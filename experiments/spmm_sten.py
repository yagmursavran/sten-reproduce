"""Fig. 15a, STen side: sparse-dense GEMM of a BERT-base feed-forward layer with STen's n:m:g kernel.

GEMM size as in the paper: 768x3072x4096, i.e. a 3072x768 weight (BERT-base FF layer, 768 -> 3072)
applied to 4096 tokens (batch 8 x sequence 512). For each sparsity level the weight is converted to
n:m:g and the Linear layer is timed through STen (dispatcher + n:m:g kernel). Dense PyTorch is timed
for reference. The paper shows the AVX2 kernel on AMD and the AVX-512 kernel on Intel; our CPU has
both, so run this script once per kernel.

Note: the layer has a (zero) bias on purpose. STen registers its n:m:g linear only for inputs
(Tensor, GroupedNMTensor, Tensor); with bias=None the dispatcher finds no match and silently falls
back to dense PyTorch, so the n:m:g kernel would not be measured at all. With the original code the
max|err| column is large (transpose bug); max|err| == 0 would mean the fallback was used.

Usage:
  python experiments/spmm_sten.py --kernel avx2   --threads 8 --out results/fig15a_sten_avx2.csv
  python experiments/spmm_sten.py --kernel avx512 --threads 8 --out results/fig15a_sten_avx512.csv
"""
import argparse
import csv
import os
import statistics
import time

import torch
import sten

LEVELS = [(3, 6), (2, 5), (2, 6), (2, 7), (1, 5), (1, 10), (1, 20)]   # same sparsities as Fig. 15


def median_ms(fn, repeats, warmup):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000, min(times) * 1000, max(times) * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-features", type=int, default=768)
    p.add_argument("--out-features", type=int, default=3072)
    p.add_argument("--tokens", type=int, default=4096, help="must be a multiple of 64 (AVX-512 kernel)")
    p.add_argument("--g", type=int, default=4)
    p.add_argument("--kernel", default="auto", choices=["auto", "avx2", "avx512"])
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--out", default="results/fig15a_sten.csv")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    layer = torch.nn.Linear(args.in_features, args.out_features, bias=True)
    with torch.no_grad():
        layer.bias.zero_()
    x = torch.randn(args.tokens, args.in_features)
    print(f"torch {torch.__version__} | threads {torch.get_num_threads()} | GEMM "
          f"{args.in_features}x{args.out_features}x{args.tokens} | kernel {args.kernel}", flush=True)

    rows = []
    with torch.no_grad():
        med, mn, mx = median_ms(lambda: layer(x), args.repeats, args.warmup)
        rows.append(dict(engine="pytorch-dense", format="dense", sparsity=0.0, kernel="-",
                         median_ms=med, min_ms=mn, max_ms=mx, max_abs_err=0.0))
        print(f"{'dense':>8}  PyTorch median {med:8.2f} ms", flush=True)

        for n, m in LEVELS:
            label = f"{n}:{m}:{args.g}"
            sb = sten.SparsityBuilder()
            sb.set_weight(name="weight", initial_sparsifier=sten.GroupedNMSparsifier(n=n, m=m, g=args.g),
                          out_format=sten.GroupedNMTensor)
            sparse = sb.get_sparse_model(layer)
            if args.kernel != "auto":
                sparse.weight.wrapped_tensor.set_spmm_opt(kernel=args.kernel)
            w_pruned = sparse.weight.wrapped_tensor.to_dense()
            y = sparse(x)                                   # first call compiles the kernel
            err = (y - torch.nn.functional.linear(x, w_pruned, layer.bias)).abs().max().item()
            med, mn, mx = median_ms(lambda: sparse(x), args.repeats, args.warmup)
            kernel = sparse.weight.wrapped_tensor.nm_strides["kernel"]
            rows.append(dict(engine="sten-nmg", format=label, sparsity=round(1 - n / m, 3), kernel=kernel,
                             median_ms=med, min_ms=mn, max_ms=mx, max_abs_err=err))
            print(f"{label:>8}  STen    median {med:8.2f} ms  sparsity {1-n/m:.0%}  kernel {kernel}  "
                  f"max|err| {err:.2e}", flush=True)
            del sparse

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
