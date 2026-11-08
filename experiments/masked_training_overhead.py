"""Fig. 14 (first panel: 1 node, 1 GPU): overhead of masked sparse training with STen on BERT-base.

One training step of BERT-base masked language modelling (mini-batch 16, as in the paper) is timed for
seven configurations, exactly the bars of Fig. 14:
  dense                       - plain PyTorch
  fixed 50% / 3:6 / 3:6:8     - every encoder Linear weight is a masked STen tensor; the mask is computed
                                once and kept fixed (after each optimizer step the same mask is re-applied)
  new   50% / 3:6 / 3:6:8     - the mask is recomputed after every optimizer step
  50% = unstructured magnitude pruning, 3:6 = n:m (STen's PerfectNMTensor), 3:6:8 = n:m:g (STen's
  GroupedNMTensor conversion; it runs on the CPU, so the weight is copied to the CPU and back)

The step time is split like the paper's stacked bars:
  PyTorch         = dense step time (same model, same batch)
  Sparsification  = time spent inside our sparsifier implementations (mask computation), measured
  STen            = the rest (dispatcher, masked dense fallback, extra copies)

The paper's training code is not in the repository; the masked layout and sparsifier registration follow
the authors' own test (tests/test_patching.py) and Section 5 of the paper. Sequence length is not stated
in the paper; default 128 so that BERT-base training fits a 6 GB GPU.

Usage (GPU):
  python experiments/masked_training_overhead.py --out results/fig14_masked_training.csv
If the GPU runs out of memory, lower --seq (e.g. 64) or --batch, and write the change in the notes.
The new-3:6:8 configuration is slow: the n:m:g conversion in the repository runs on the CPU
(the paper mentions an optimized GPU conversion that is not available), roughly 1-2 min per step.
Smoke test (CPU, tiny model, no download):
  python experiments/masked_training_overhead.py --tiny --device cpu --steps 2 --warmup 1 --out /tmp/fig14_test.csv
"""
import argparse
import copy
import csv
import os
import statistics
import time

import torch
import sten
from transformers import BertConfig, BertForMaskedLM


# ----------------------------------------------------------------------------------------------
# Masked layout: dense values + boolean mask (same idea as FixedMaskTensor in the authors' tests)
# ----------------------------------------------------------------------------------------------
class MaskedWeight:
    def __init__(self, val, mask):
        self.val = val
        self.mask = mask

    def to_dense(self):
        return copy.deepcopy(self.val.data)

    @property
    def shape(self):
        return self.val.shape


class MaskSparsifier:
    """kind: 'unstructured' | 'nm' | 'nmg'; recompute: False = fixed mask, True = new mask every step."""

    def __init__(self, kind, recompute, fraction=0.5, n=3, m=6, g=8):
        self.kind, self.recompute = kind, recompute
        self.fraction, self.n, self.m, self.g = fraction, n, m, g


SPARSIFICATION_TIME = [0.0]          # seconds spent computing masks (summed by the timer below)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def compute_mask(sp, w):
    """Boolean mask that keeps the entries chosen by the format rule."""
    if sp.kind == "unstructured":
        num_zeros = int(sp.fraction * w.numel())
        order = torch.argsort(torch.argsort(w.flatten().abs())).reshape(w.shape)
        return order >= num_zeros
    if sp.kind == "nm":
        t = sten.PerfectNMTensor.from_dense(w, sp.n, sp.m, sparse_dim=w.ndim - 2)
        return t.to_dense() != 0
    if sp.kind == "nmg":
        t = sten.GroupedNMTensor.from_dense(w.detach().cpu(), sp.n, sp.m, sparse_dim=w.ndim - 2,
                                            group_size=sp.g, group_dim=w.ndim - 1)
        return (t.to_dense() != 0).to(w.device)
    raise ValueError(sp.kind)


def timed_mask(sp, w):
    sync(w.device)
    t0 = time.perf_counter()
    mask = compute_mask(sp, w)
    sync(w.device)
    SPARSIFICATION_TIME[0] += time.perf_counter() - t0
    return mask


@sten.register_sparsifier_implementation(sparsifier=MaskSparsifier, inp=torch.Tensor, out=MaskedWeight)
def dense_to_masked(sparsifier, tensor, grad_fmt=None):
    mask = timed_mask(sparsifier, tensor)
    return sten.SparseTensorWrapper.wrapped_from_dense(
        MaskedWeight((tensor * mask).detach().clone(), mask), tensor, grad_fmt)


@sten.register_sparsifier_implementation(sparsifier=sten.SameFormatSparsifier, inp=torch.Tensor, out=MaskedWeight)
def reapply_mask(sparsifier, tensor, grad_fmt=None):
    """Called by STen after in-place updates (optimizer step): keep the mask or recompute it."""
    ref = sparsifier.ref_sp_ten.wrapped_tensor
    sp = sparsifier.ref_sp_ten.wrapped_tensor.sparsifier
    mask = timed_mask(sp, tensor) if sp.recompute else ref.mask
    out = MaskedWeight((tensor * mask).detach().clone(), mask)
    out.sparsifier = sp
    return sten.SparseTensorWrapper.wrapped_from_dense(out, sparsifier.ref_sp_ten, grad_fmt)


def encoder_linear_weights(model):
    return [n + ".weight" for n, mod in model.named_modules()
            if isinstance(mod, torch.nn.Linear) and "encoder.layer" in n]


def make_sparse(model, sp):
    sb = sten.SparsityBuilder()
    for name in encoder_linear_weights(model):
        # Gradients stay dense; after the optimizer's in-place update STen calls reapply_mask
        # (SameFormatSparsifier), which keeps or recomputes the mask.
        sb.set_weight(name=name, initial_sparsifier=sp, out_format=MaskedWeight)
    sparse = sb.get_sparse_model(model)
    for name in encoder_linear_weights(model):                  # remember the rule for re-sparsification
        obj = sparse
        for part in name.split("."):
            obj = getattr(obj, part)
        obj.wrapped_tensor.sparsifier = sp
    return sparse


def weight_sparsity(model):
    zeros = total = 0
    params = dict(model.named_parameters())
    for name in encoder_linear_weights(model):
        p = params[name]
        w = p.wrapped_tensor.to_dense() if isinstance(p, sten.SparseTensorWrapper) else p.detach()
        zeros += int((w == 0).sum())
        total += w.numel()
    return zeros / total


def time_steps(model, batch, device, steps, warmup):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()
    times, spars = [], []
    for i in range(warmup + steps):
        SPARSIFICATION_TIME[0] = 0.0
        sync(device)
        t0 = time.perf_counter()
        loss = model(**batch).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        sync(device)
        if i >= warmup:
            times.append(time.perf_counter() - t0)
            spars.append(SPARSIFICATION_TIME[0])
    return statistics.median(times), min(times), max(times), statistics.median(spars)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="bert-base-uncased")
    p.add_argument("--tiny", action="store_true", help="random 2-layer BERT (smoke test, no download)")
    p.add_argument("--batch", type=int, default=16, help="paper: 16")
    p.add_argument("--seq", type=int, default=128, help="not stated in the paper")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--configs", default="dense,fixed-50%,fixed-3:6,fixed-3:6:8,new-50%,new-3:6,new-3:6:8")
    p.add_argument("--out", default="results/fig14_masked_training.csv")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)
    if args.tiny:
        base = BertForMaskedLM(BertConfig(num_hidden_layers=2, hidden_size=96, num_attention_heads=4,
                                          intermediate_size=384))
    else:
        base = BertForMaskedLM.from_pretrained(args.model)
    # the base model stays on the CPU; each configuration gets its own copy on the GPU (saves GPU memory)
    vocab = base.config.vocab_size
    ids = torch.randint(0, vocab, (args.batch, args.seq), device=device)
    batch = dict(input_ids=ids, labels=ids.clone())
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"torch {torch.__version__} | device {gpu} | model {'tiny' if args.tiny else args.model} | "
          f"batch {args.batch} x seq {args.seq} | {args.steps} steps after {args.warmup} warm-up", flush=True)

    specs = {"50%": dict(kind="unstructured"), "3:6": dict(kind="nm"), "3:6:8": dict(kind="nmg")}
    rows, dense_time = [], None
    for cfg in args.configs.split(","):
        model = copy.deepcopy(base).to(device)
        if cfg != "dense":
            mode, fmt = cfg.split("-", 1)
            model = make_sparse(model, MaskSparsifier(recompute=(mode == "new"), **specs[fmt]))
        med, mn, mx, spars = time_steps(model, batch, device, args.steps, args.warmup)
        if cfg == "dense":
            dense_time = med
        pytorch = dense_time if dense_time is not None else float("nan")
        sten_part = max(med - pytorch - spars, 0.0)
        row = dict(config=cfg, step_s=med, min_s=mn, max_s=mx, pytorch_s=pytorch, sten_s=sten_part,
                   sparsification_s=spars, overhead_vs_dense=med / pytorch, weight_sparsity=weight_sparsity(model))
        rows.append(row)
        print(f"{cfg:>12}  step {med:7.3f} s  [pytorch {pytorch:.3f} | sten {sten_part:.3f} | "
              f"sparsification {spars:.3f}]  {med/pytorch:5.2f}x dense  sparsity {row['weight_sparsity']:.1%}"
              + (f"  gpu mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB" if device.type == "cuda" else ""),
              flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
