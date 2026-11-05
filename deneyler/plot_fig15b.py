"""Draw our version of Fig. 15b: dense PyTorch vs STen n:m:g, with the STen bar split into
"other" (PyTorch work that sparsity does not touch) and "linear" (encoder Linear layers = STen part).

Input: one or more CSV files from bert_latency.py (one per run). For every format the median over the
runs is plotted, with a thin line showing the min-max range across runs.

Usage:
  python deneyler/plot_fig15b.py results/bert_base_run1.csv results/bert_base_run2.csv \
      --out results/fig15b_bert_base.png
"""
import argparse
import csv
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    with open(path) as f:
        return {r["format"]: r for r in csv.DictReader(f)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="+")
    p.add_argument("--out", default="results/fig15b.png")
    p.add_argument("--title", default="BERT inference latency, dense PyTorch vs STen n:m:g")
    args = p.parse_args()

    runs = [read(path) for path in args.csv]
    formats = list(runs[0].keys())                      # dense first, then the sparse formats
    med = lambda key, fmt: statistics.median(float(r[fmt][key]) for r in runs if r[fmt][key] != "")

    fig, ax = plt.subplots(figsize=(10, 4))
    xs = list(range(len(formats)))
    totals = [med("median_s", f) for f in formats]
    lows = [min(float(r[f]["median_s"]) for r in runs) for f in formats]
    highs = [max(float(r[f]["median_s"]) for r in runs) for f in formats]
    has_breakdown = all(r[f]["linear_s"] != "" for r in runs for f in formats)

    for i, f in enumerate(formats):
        if f == "dense":
            ax.bar(i, totals[i], color="C0", label="PyTorch (dense)")
            continue
        if has_breakdown:
            # rescale the hook-measured split so that the two parts add up to the measured total
            lin, oth = med("linear_s", f), med("other_s", f)
            lin_part = totals[i] * lin / (lin + oth)
            ax.bar(i, totals[i] - lin_part, color="C0", alpha=0.55,
                   label="STen run: other PyTorch ops" if i == 1 else None)
            ax.bar(i, lin_part, bottom=totals[i] - lin_part, color="C1",
                   label="STen run: n:m:g Linear layers" if i == 1 else None)
        else:
            ax.bar(i, totals[i], color="C1", label="STen n:m:g" if i == 1 else None)
        speedup = totals[0] / totals[i]
        ax.annotate(f"{speedup:.2f}x", (i, totals[i]), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8)
    if len(runs) > 1:
        ax.vlines(xs, lows, highs, color="black", linewidth=1)

    labels = ["Dense"] + [f"{float(runs[0][f]['sparsity']):.0%}\n{f}" for f in formats[1:]]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Time per batch [s]")
    ax.set_xlabel("Sparsity")
    ax.set_title(f"{args.title}  ({len(runs)} run{'s' if len(runs) > 1 else ''}, median)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
