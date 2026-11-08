"""Draw our version of Fig. 15a: sparse-dense GEMM (768x3072x4096) time per sparsity level,
STen n:m:g (AVX2 and/or AVX-512 kernel) vs DeepSparse (unstructured), with dense PyTorch as a line.

Usage:
  python experiments/plot_fig15a.py --sten results/fig15a_sten_avx2.csv results/fig15a_sten_avx512.csv \
      --deepsparse results/fig15a_deepsparse.csv --out results/fig15a.png
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sten", nargs="+", required=True, help="CSV files from spmm_sten.py")
    p.add_argument("--deepsparse", help="CSV file from spmm_deepsparse.py")
    p.add_argument("--out", default="results/fig15a.png")
    args = p.parse_args()

    series, dense_ms = [], None
    for path in args.sten:
        rows = read(path)
        dense_ms = float(rows[0]["median_ms"]) if dense_ms is None else dense_ms
        sparse = [r for r in rows if r["format"] != "dense"]
        series.append((f"STen n:m:g ({sparse[0]['kernel']})",
                       {round(float(r["sparsity"]), 2): float(r["median_ms"]) for r in sparse}))
    if args.deepsparse:
        rows = [r for r in read(args.deepsparse) if float(r["sparsity"]) > 0]
        series.append(("DeepSparse 1.1.0 (unstructured)",
                       {round(float(r["sparsity"]), 2): float(r["median_ms"]) for r in rows}))

    levels = sorted(series[0][1].keys())
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / len(series)
    for i, (label, vals) in enumerate(series):
        xs = [j + i * width for j in range(len(levels))]
        ax.bar(xs, [vals.get(l, float("nan")) for l in levels], width, label=label)
    ax.axhline(dense_ms, color="gray", linestyle="--", linewidth=1, label=f"PyTorch dense ({dense_ms:.1f} ms)")
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(levels))])
    ax.set_xticklabels([f"{l:.0%}" for l in levels])
    ax.set_xlabel("Sparsity")
    ax.set_ylabel("Time [ms]")
    ax.set_title("Sparse-dense GEMM 768x3072x4096 (BERT-base FF layer), Ryzen 7 7840HS")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
