"""Table 4 / Fig. 16: magnitude pruning of a Wide ResNet-16-8 on CIFAR-10 to 50% sparsity with STen.

The paper's training code is not in the repository, so this script follows the description in
Section 7.2 of the paper:
  dense      - train WRN-16-8 from scratch (Zagoruyko & Komodakis recipe: SGD, momentum 0.9, nesterov,
               weight decay 5e-4, lr 0.1, x0.2 at 30%, 60%, 80% of the epochs)
  oneshot    - prune all weights to 50% at once, then fine-tune with the same schedule
  iterative  - start at 10% sparsity, fine-tune N epochs, increase by 10% until 50%
  layerwise  - prune one layer at a time to 50% (first layer first), fine-tune N epochs after each

Sparsity is implemented with STen exactly like the authors' masked-training test
(tests/test_patching.py): a FixedMaskTensor layout, sten.ScalarFractionSparsifier (magnitude
pruning) and sten.SameFormatSparsifier, which re-applies the sparsity after every optimizer step.
Every conv and linear weight is pruned (the paper does not say which layers; this is our choice).

The dense checkpoint is needed by the three pruning modes. Each run writes one CSV row per epoch
(train loss, test accuracy, seconds per batch) and a checkpoint after every epoch, so an interrupted
run can be resumed with --resume.

Usage (GPU laptop):
  python experiments/wrn_cifar10.py dense     --epochs 200 --out results/wrn/dense
  python experiments/wrn_cifar10.py oneshot   --epochs 200 --dense-ckpt results/wrn/dense/model.pt --out results/wrn/oneshot
  python experiments/wrn_cifar10.py iterative --epochs-per-stage 75 --dense-ckpt ... --out results/wrn/iterative
  python experiments/wrn_cifar10.py layerwise --epochs-per-stage 30 --dense-ckpt ... --out results/wrn/layerwise
Quick smoke test (CPU, a few batches):
  python experiments/wrn_cifar10.py dense --epochs 1 --limit-batches 3 --width 1 --fake-data --out /tmp/wrn_test
"""
import argparse
import copy
import csv
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import sten


# ----------------------------------------------------------------------------------------------
# Model: Wide ResNet (Zagoruyko & Komodakis, 2016). Depth 16, widen factor 8 -> 11.0 M parameters.
# ----------------------------------------------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride, dropout):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.dropout = dropout
        self.shortcut = None if (cin == cout and stride == 1) else nn.Conv2d(cin, cout, 1, stride, 0, bias=False)

    def forward(self, x):
        o = F.relu(self.bn1(x))
        y = self.conv1(o)
        y = F.relu(self.bn2(y))
        if self.dropout > 0:
            y = F.dropout(y, p=self.dropout, training=self.training)
        y = self.conv2(y)
        return y + (x if self.shortcut is None else self.shortcut(o))


class WideResNet(nn.Module):
    def __init__(self, depth=16, width=8, num_classes=10, dropout=0.0):
        super().__init__()
        n = (depth - 4) // 6
        widths = [16, 16 * width, 32 * width, 64 * width]
        self.conv1 = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        layers, cin = [], widths[0]
        for i, (cout, stride) in enumerate(zip(widths[1:], [1, 2, 2])):
            for j in range(n):
                layers.append(BasicBlock(cin, cout, stride if j == 0 else 1, dropout))
                cin = cout
        self.blocks = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(cin)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        x = self.blocks(self.conv1(x))
        x = F.relu(self.bn(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def prunable_weights(model):
    """Names of all conv and linear weights, in forward order (used by layer-wise pruning)."""
    return [name + ".weight" for name, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


# ----------------------------------------------------------------------------------------------
# STen masked layout (same as the authors' tests/test_patching.py)
# ----------------------------------------------------------------------------------------------
class FixedMaskTensor:
    def __init__(self, val, mask):
        self.val = val
        self.mask = mask

    def to_dense(self):
        return copy.deepcopy(self.val.data)


def magnitude_mask(tensor, num_zeros):
    indices = torch.argsort(torch.argsort(tensor.flatten().abs())).reshape(tensor.shape)
    return indices >= num_zeros


@sten.register_sparsifier_implementation(
    sparsifier=sten.ScalarFractionSparsifier, inp=torch.Tensor, out=FixedMaskTensor
)
def dense_to_fixed(sparsifier, tensor, grad_fmt=None):
    mask = magnitude_mask(tensor, int(sparsifier.fraction * tensor.numel()))
    val = (tensor * mask).detach().clone()
    return sten.SparseTensorWrapper.wrapped_from_dense(FixedMaskTensor(val, mask), tensor, grad_fmt)


@sten.register_sparsifier_implementation(
    sparsifier=sten.SameFormatSparsifier, inp=torch.Tensor, out=FixedMaskTensor
)
def same_to_fixed(sparsifier, tensor, grad_fmt=None):
    ref = sparsifier.ref_sp_ten.wrapped_tensor
    mask = magnitude_mask(tensor, tensor.numel() - int(torch.sum(ref.mask)))
    val = (tensor * mask).detach().clone()
    return sten.SparseTensorWrapper.wrapped_from_dense(FixedMaskTensor(val, mask), sparsifier.ref_sp_ten, grad_fmt)


def sparsify(dense_model, fractions):
    """Copy of dense_model where every weight in `fractions` ({name: fraction}) is magnitude-pruned."""
    sb = sten.SparsityBuilder()
    for name, frac in fractions.items():
        sp = sten.ScalarFractionSparsifier(frac)
        sb.set_weight(name=name, initial_sparsifier=sp, out_format=FixedMaskTensor)
        sb.set_weight_grad(name=name, external_sparsifier=sp, out_format=FixedMaskTensor)
    return sb.get_sparse_model(dense_model)


def densify(model, template):
    """Dense copy of a (possibly sparse) model: wrapped weights are replaced by their dense values."""
    dense = copy.deepcopy(template)
    state = {}
    for name, p in list(model.named_parameters()) + list(model.named_buffers()):
        state[name] = p.wrapped_tensor.to_dense() if isinstance(p, sten.SparseTensorWrapper) else p.detach()
    dense.load_state_dict(state)
    return dense


def sparsity_report(model, names):
    zeros = total = 0
    for name in names:
        w = dict(model.named_parameters())[name]
        w = w.wrapped_tensor.to_dense() if isinstance(w, sten.SparseTensorWrapper) else w.detach()
        zeros += int((w == 0).sum())
        total += w.numel()
    return zeros / total


# ----------------------------------------------------------------------------------------------
# Data and training loop (identical for dense and sparse models)
# ----------------------------------------------------------------------------------------------
def loaders(args):
    if args.fake_data:   # random images, only to test that the code runs (no download)
        ds = lambda n: torch.utils.data.TensorDataset(torch.randn(n, 3, 32, 32), torch.randint(0, 10, (n,)))
        return (torch.utils.data.DataLoader(ds(512), batch_size=args.batch, shuffle=True, drop_last=True),
                torch.utils.data.DataLoader(ds(256), batch_size=128))
    import torchvision
    import torchvision.transforms as T
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train = torchvision.datasets.CIFAR10(args.data, train=True, download=True, transform=train_tf)
    test = torchvision.datasets.CIFAR10(args.data, train=False, download=True, transform=test_tf)
    kw = dict(num_workers=args.workers, pin_memory=torch.cuda.is_available())
    return (torch.utils.data.DataLoader(train, batch_size=args.batch, shuffle=True, drop_last=True, **kw),
            torch.utils.data.DataLoader(test, batch_size=512, shuffle=False, **kw))


def make_optimizer(model, args, epochs):
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    milestones = [int(epochs * f) for f in (0.3, 0.6, 0.8)]
    return opt, torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.2)


def evaluate(model, test_loader, device, limit):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if limit and i >= limit:
                break
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return 100.0 * correct / total


def train_epochs(model, epochs, args, device, train_loader, test_loader, log, stage, names, state):
    """Standard training loop. `state` holds what is needed to resume (epoch counters)."""
    opt, sched = make_optimizer(model, args, epochs)
    for _ in range(state.get("sched_steps", 0)):
        sched.step()
    for epoch in range(state.get("epoch_in_stage", 0), epochs):
        model.train()
        loss_sum, n, t0 = 0.0, 0, time.perf_counter()
        for i, (x, y) in enumerate(train_loader):
            if args.limit_batches and i >= args.limit_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            n += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        sec_per_batch = (time.perf_counter() - t0) / max(n, 1)
        sched.step()
        acc = evaluate(model, test_loader, device, args.limit_batches)
        row = dict(stage=stage, epoch=epoch + 1, global_epoch=state["global_epoch"] + 1,
                   lr=opt.param_groups[0]["lr"], train_loss=loss_sum / max(n, 1), test_acc=acc,
                   sec_per_batch=sec_per_batch, sparsity=sparsity_report(model, names))
        log(row)
        state["global_epoch"] += 1
        state["epoch_in_stage"] = epoch + 1
        state["sched_steps"] = epoch + 1
        yield model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["dense", "oneshot", "iterative", "layerwise"])
    p.add_argument("--out", required=True, help="output folder (log.csv, model.pt, state.json)")
    p.add_argument("--dense-ckpt", help="dense model.pt, required for the pruning modes")
    p.add_argument("--epochs", type=int, default=200, help="dense and one-shot")
    p.add_argument("--epochs-per-stage", type=int, default=None,
                   help="iterative (paper: 75) and layer-wise (paper: 30)")
    p.add_argument("--sparsity", type=float, default=0.5)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1, help="0.1 for dense; consider 0.01 for fine-tuning")
    p.add_argument("--width", type=int, default=8, help="widen factor (8 = WRN-16-8)")
    p.add_argument("--data", default=os.path.expanduser("~/data/cifar10"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit-batches", type=int, default=0, help="for smoke tests only")
    p.add_argument("--fake-data", action="store_true", help="random data instead of CIFAR-10 (smoke test)")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)
    log_path, ckpt_path, state_path = (os.path.join(args.out, f) for f in ("log.csv", "model.pt", "state.json"))

    template = WideResNet(16, args.width).to(device)
    names = prunable_weights(template)
    print(f"device {device} | WRN-16-{args.width} | {sum(p.numel() for p in template.parameters())/1e6:.1f} M params "
          f"| {len(names)} prunable weights | mode {args.mode}", flush=True)

    state = {"global_epoch": 0, "stage": 0, "epoch_in_stage": 0, "sched_steps": 0}
    dense = copy.deepcopy(template)
    if args.resume and os.path.exists(state_path):
        state = json.load(open(state_path))
        dense.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"resuming: {state}", flush=True)
    elif args.mode != "dense":
        assert args.dense_ckpt, "--dense-ckpt is required for pruning modes"
        dense.load_state_dict(torch.load(args.dense_ckpt, map_location=device))

    fieldnames = ["mode", "stage", "epoch", "global_epoch", "lr", "train_loss", "test_acc", "sec_per_batch", "sparsity"]
    new_log = not (args.resume and os.path.exists(log_path))

    def log(row):
        nonlocal new_log
        row = dict(mode=args.mode, **row)
        with open(log_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if new_log:
                w.writeheader()
                new_log = False
            w.writerow(row)
        print(f"[{args.mode}] stage {row['stage']} epoch {row['epoch']:>3} | loss {row['train_loss']:.4f} | "
              f"test acc {row['test_acc']:.2f}% | {row['sec_per_batch']*1000:.0f} ms/batch | "
              f"sparsity {row['sparsity']:.1%}", flush=True)

    train_loader, test_loader = loaders(args)
    if not state.get("baseline_logged") and args.mode != "dense":
        print(f"dense checkpoint test acc: {evaluate(dense, test_loader, device, args.limit_batches):.2f}%", flush=True)
        state["baseline_logged"] = True

    # Stages: list of {weight name: fraction} dicts, one per stage
    if args.mode == "dense":
        stages, epochs = [{}], args.epochs
    elif args.mode == "oneshot":
        stages, epochs = [{n: args.sparsity for n in names}], args.epochs
    elif args.mode == "iterative":
        steps = [round(0.1 * k, 2) for k in range(1, int(round(args.sparsity * 10)) + 1)]
        stages, epochs = [{n: s for n in names} for s in steps], args.epochs_per_stage or 75
    else:  # layerwise
        stages, epochs = [{n: args.sparsity for n in names[: k + 1]} for k in range(len(names))], args.epochs_per_stage or 30

    for stage_idx in range(state["stage"], len(stages)):
        fractions = stages[stage_idx]
        model = sparsify(dense, fractions) if fractions else dense
        for model in train_epochs(model, epochs, args, device, train_loader, test_loader, log,
                                  stage_idx + 1, names, state):
            dense = densify(model, template) if fractions else model
            torch.save(dense.state_dict(), ckpt_path)          # dense weights; zeros mark pruned entries
            json.dump(dict(state, stage=stage_idx), open(state_path, "w"))
        state.update(stage=stage_idx + 1, epoch_in_stage=0, sched_steps=0)
        json.dump(state, open(state_path, "w"))

    print(f"done. final test acc {evaluate(dense, test_loader, device, args.limit_batches):.2f}% | "
          f"sparsity {sparsity_report(dense, names):.1%} | model: {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
