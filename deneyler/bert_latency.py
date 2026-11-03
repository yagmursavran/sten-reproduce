"""Mini reproduction of STen Fig. 15b: dense vs n:m:g sparse BERT inference latency on CPU.

Every Linear weight inside the BERT encoder is converted to the n:m:g format with STen's
SparsityBuilder, exactly like the repository's tests/test_nmg.py. For each format the script
  1. checks correctness against a dense model that holds the same pruned weights,
  2. measures the median latency over several runs (after a warm-up call that compiles the kernel).

Usage examples:
  python deneyler/bert_latency.py                                   # 2-layer random BERT, quick
  python deneyler/bert_latency.py --pretrained prajjwal1/bert-small # small pretrained BERT
  python deneyler/bert_latency.py --pretrained bert-base-uncased --batch 8 --seq 512  # paper setting
  python deneyler/bert_latency.py --kernel avx2                     # force the AVX2 kernel
"""
import argparse
import copy
import csv
import os
import platform
import statistics
import time

import torch
import sten
from transformers import BertConfig, BertModel


def parse_formats(text):
    out = []
    for item in text.split(","):
        n, m = item.strip().split(":")
        out.append((int(n), int(m)))
    return out


def build_model(args):
    if args.pretrained:
        model = BertModel.from_pretrained(args.pretrained)
    else:
        model = BertModel(BertConfig(num_hidden_layers=args.layers))
    return model.eval()


def encoder_linear_weights(model):
    return [
        name + ".weight"
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and "encoder.layer" in name
    ]


def get_attr(root, dotted):
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def sparsify(model, n, m, g, kernel):
    sb = sten.SparsityBuilder()
    for name in encoder_linear_weights(model):
        sb.set_weight(
            name=name,
            initial_sparsifier=sten.GroupedNMSparsifier(n=n, m=m, g=g),
            out_format=sten.GroupedNMTensor,
        )
    sparse_model = sb.get_sparse_model(model)
    if kernel != "auto":
        for name in encoder_linear_weights(model):
            get_attr(sparse_model, name).wrapped_tensor.set_spmm_opt(kernel=kernel)
    return sparse_model


def masked_dense_reference(model, sparse_model):
    """Dense model whose Linear weights equal the pruned n:m:g weights (for correctness checks)."""
    ref = copy.deepcopy(model)
    with torch.no_grad():
        for name in encoder_linear_weights(model):
            pruned = get_attr(sparse_model, name).wrapped_tensor.to_dense()
            get_attr(ref, name).copy_(pruned)
    return ref


def median_latency(fn, repeats, warmup):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), min(times), max(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", default=None, help="HuggingFace model name; default: random weights")
    p.add_argument("--layers", type=int, default=2, help="encoder layers when using random weights")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--formats", default="3:6,2:7,1:10", help="comma-separated n:m pairs")
    p.add_argument("--g", type=int, default=4, help="group size")
    p.add_argument("--kernel", default="auto", choices=["auto", "avx2", "avx512"])
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--threads", type=int, default=None, help="torch threads (also set OMP_NUM_THREADS)")
    p.add_argument("--out", default="results/bert_latency.csv")
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    model = build_model(args)
    x = torch.randint(0, model.config.vocab_size, (args.batch, args.seq))
    print(f"torch {torch.__version__} | threads {torch.get_num_threads()} | "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')} | {platform.processor() or platform.machine()}")
    print(f"model: {args.pretrained or f'random BERT, {args.layers} layers'} | input {args.batch}x{args.seq}")

    rows = []
    with torch.no_grad():
        dense_med, dense_min, dense_max = median_latency(lambda: model(x), args.repeats, args.warmup)
        rows.append(dict(format="dense", sparsity=0.0, kernel="-", median_s=dense_med,
                         min_s=dense_min, max_s=dense_max, speedup=1.0, max_abs_err=0.0))
        print(f"{'dense':>10}  median {dense_med*1000:9.1f} ms")

        for n, m in parse_formats(args.formats):
            label = f"{n}:{m}:{args.g}"
            t0 = time.perf_counter()
            sparse_model = sparsify(model, n, m, args.g, args.kernel)
            conv_s = time.perf_counter() - t0
            ref = masked_dense_reference(model, sparse_model)

            y_sparse = sparse_model(x).last_hidden_state   # first call also compiles the kernel
            y_ref = ref(x).last_hidden_state
            err = (y_sparse - y_ref).abs().max().item()

            med, mn, mx = median_latency(lambda: sparse_model(x), args.repeats, args.warmup)
            kernel = get_attr(sparse_model, encoder_linear_weights(model)[0]).wrapped_tensor.nm_strides["kernel"]
            rows.append(dict(format=label, sparsity=round(1 - n / m, 3), kernel=kernel, median_s=med,
                             min_s=mn, max_s=mx, speedup=dense_med / med, max_abs_err=err))
            print(f"{label:>10}  median {med*1000:9.1f} ms  speedup {dense_med/med:5.2f}x  "
                  f"sparsity {1-n/m:.0%}  kernel {kernel}  max|err| {err:.2e}  (convert {conv_s:.1f}s)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
