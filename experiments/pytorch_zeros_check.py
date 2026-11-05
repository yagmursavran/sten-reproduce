"""Does dense PyTorch get faster when the weight contains zeros?

Expected answer: no. A dense matrix multiplication performs every multiply-add, including those with
zeros, so its time depends only on the matrix shapes. This is why Fig. 15b of the paper shows the
PyTorch time only once (the "Dense" column): it is the same at every sparsity level.

For comparison the script also times PyTorch's own built-in sparse format (CSR), which actually
skips the zeros.

Usage:
  python experiments/pytorch_zeros_check.py --threads 8
"""
import argparse
import statistics
import time

import torch


def median_ms(fn, repeats=20, warmup=3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--tokens", type=int, default=1024, help="rows of the input (batch x seq)")
    p.add_argument("--in-features", type=int, default=2048)
    p.add_argument("--out-features", type=int, default=512)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    x = torch.randn(args.tokens, args.in_features)
    w = torch.randn(args.out_features, args.in_features)
    print(f"input {tuple(x.shape)}, weight {tuple(w.shape)}, threads {torch.get_num_threads()}")

    with torch.no_grad():
        for sparsity in (0.0, 0.5, 0.9, 0.99):
            wz = w.clone()
            wz[torch.rand_like(wz) < sparsity] = 0.0
            dense_ms = median_ms(lambda: torch.nn.functional.linear(x, wz))
            csr = wz.to_sparse_csr()
            csr_ms = median_ms(lambda: torch.sparse.mm(csr, x.T))
            print(f"zeros {sparsity:4.0%}:  dense PyTorch {dense_ms:8.2f} ms   "
                  f"PyTorch sparse CSR {csr_ms:8.2f} ms")


if __name__ == "__main__":
    main()
