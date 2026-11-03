"""Mini reproduction of STen Fig. 12: how much weight magnitude ("energy") each sparsity structure keeps.

energy = ||X_pruned||_1 / ||X||_1   (1.0 = nothing lost)

Compared structures, at the paper's sparsity levels (3:6, 2:5, 2:6, 2:7, 1:5, 1:10, 1:20):
  unstructured  - keep the largest values anywhere
  n:m           - STen PerfectNMTensor (optimal n:m)
  n:m:g         - STen GroupedNMTensor, g = 1, 4, 16
  blocked b     - keep whole b x b blocks with the largest L1 norm, b = 2, 4, 8

The paper uses bert-base-uncased, layer 8, attention query weight. Without --pretrained a random
Gaussian matrix is used: that only tests the pipeline, its numbers are not comparable to the paper.

Usage:
  python deneyler/energy.py --pretrained bert-base-uncased
  python deneyler/energy.py --pretrained prajjwal1/bert-small --layer 2
"""
import argparse
import csv
import os
import time

import torch
import sten

LEVELS = [(3, 6), (2, 5), (2, 6), (2, 7), (1, 5), (1, 10), (1, 20)]


def energy(orig, pruned):
    return (pruned.abs().sum() / orig.abs().sum()).item()


def prune_unstructured(w, sparsity):
    keep = int(round(w.numel() * (1 - sparsity)))
    flat = w.flatten()
    idx = flat.abs().topk(keep).indices
    out = torch.zeros_like(flat)
    out[idx] = flat[idx]
    return out.view_as(w)


def prune_blocked(w, sparsity, b):
    rows, cols = w.shape
    pr, pc = (-rows) % b, (-cols) % b
    wp = torch.nn.functional.pad(w, (0, pc, 0, pr))
    R, C = wp.shape
    blocks = wp.reshape(R // b, b, C // b, b).permute(0, 2, 1, 3)       # (R/b, C/b, b, b)
    scores = blocks.abs().sum(dim=(2, 3)).flatten()
    keep = int(round(scores.numel() * (1 - sparsity)))
    mask = torch.zeros_like(scores)
    mask[scores.topk(keep).indices] = 1
    mask = mask.view(R // b, C // b, 1, 1)
    pruned = (blocks * mask).permute(0, 2, 1, 3).reshape(R, C)
    return pruned[:rows, :cols]


def prune_nm(w, n, m):
    # same dimension convention as STen's GroupedNMSparsifier for 2-D weights
    return sten.PerfectNMTensor.from_dense(w, n=n, m=m, sparse_dim=w.ndim - 2).to_dense()


def prune_nmg(w, n, m, g):
    return sten.GroupedNMTensor.from_dense(
        w, n=n, m=m, sparse_dim=w.ndim - 2, group_size=g, group_dim=w.ndim - 1
    ).to_dense()


def load_weight(args):
    if args.pretrained:
        from transformers import BertModel
        model = BertModel.from_pretrained(args.pretrained)
        w = model.encoder.layer[args.layer].attention.self.query.weight.detach().clone()
        src = f"{args.pretrained}, layer {args.layer}, attention query"
    else:
        torch.manual_seed(0)
        w = torch.randn(args.size, args.size)
        src = f"random Gaussian {args.size}x{args.size} (pipeline test only)"
    return w.float(), src


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", default=None)
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--size", type=int, default=768, help="matrix size for the random fallback")
    p.add_argument("--out", default="results/energy.csv")
    p.add_argument("--plot", default="results/energy.png")
    args = p.parse_args()

    w, src = load_weight(args)
    print(f"weight: {src}, shape {tuple(w.shape)}")

    methods = [("unstructured", lambda n, m, s: prune_unstructured(w, s)),
               ("n:m", lambda n, m, s: prune_nm(w, n, m))]
    methods += [(f"n:m:g g={g}", (lambda g: lambda n, m, s: prune_nmg(w, n, m, g))(g)) for g in (16, 4, 1)]
    methods += [(f"blocked b={b}", (lambda b: lambda n, m, s: prune_blocked(w, s, b))(b)) for b in (2, 4, 8)]

    rows = []
    for n, m in LEVELS:
        s = 1 - n / m
        line = [f"{s:.0%} ({n}:{m})"]
        for name, fn in methods:
            t0 = time.perf_counter()
            pruned = fn(n, m, s)
            e = energy(w, pruned)
            kept = (pruned != 0).float().mean().item()
            rows.append(dict(sparsity=round(s, 3), n=n, m=m, method=name, energy=round(e, 4),
                             kept_fraction=round(kept, 4), seconds=round(time.perf_counter() - t0, 2)))
            line.append(f"{name}={e:.3f}")
        print("  ".join(line))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"saved {args.out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [mname for mname, _ in methods]
        fig, ax = plt.subplots(figsize=(12, 4))
        width = 0.8 / len(names)
        for i, mname in enumerate(names):
            vals = [r["energy"] for r in rows if r["method"] == mname]
            ax.bar([j + i * width for j in range(len(LEVELS))], vals, width, label=mname)
        ax.set_xticks([j + 0.4 - width / 2 for j in range(len(LEVELS))])
        ax.set_xticklabels([f"{1-n/m:.0%}\n({n}:{m})" for n, m in LEVELS])
        ax.set_ylabel("Energy  ||X̂||₁ / ||X||₁")
        ax.set_xlabel("Sparsity (n:m)")
        ax.set_title(src)
        ax.legend(ncol=4, fontsize=8)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"saved {args.plot}")
    except ImportError:
        print("matplotlib not installed: skipping the plot")


if __name__ == "__main__":
    main()
