"""Fig. 15a, DeepSparse side: sparse-dense GEMM of a BERT-base feed-forward layer with DeepSparse 1.1.0.

GEMM size as in the paper: 768x3072x4096, i.e. a 3072x768 weight (BERT-base FF layer, 768 -> 3072)
applied to 4096 tokens (batch 8 x sequence 512). The weight is pruned UNSTRUCTURED by magnitude
(the paper compares against DeepSparse "on a sparse unstructured tensor").

The ONNX model is a single MatMul, Y = X @ W^T, built directly with the onnx helper API, so this
environment does not need PyTorch. DeepSparse detects the zeros in the weight by itself.

Runs in its OWN environment (Python 3.10, deepsparse==1.1.0, onnx, numpy<1.24), see the setup notes.

Usage:
  NM_VERSION_CHECK=false python experiments/spmm_deepsparse.py --cores 8 --out results/fig15a_deepsparse.csv
"""
import argparse
import csv
import os
import statistics
import tempfile
import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

LEVELS = [(3, 6), (2, 5), (2, 6), (2, 7), (1, 5), (1, 10), (1, 20)]   # same sparsities as Fig. 15


def prune_unstructured(w, sparsity):
    """Keep the largest-magnitude (1 - sparsity) fraction of the entries."""
    keep = int(round(w.size * (1 - sparsity)))
    threshold = np.partition(np.abs(w).ravel(), w.size - keep)[w.size - keep]
    return np.where(np.abs(w) >= threshold, w, 0.0).astype(np.float32)


def matmul_model(w_t, tokens):
    """ONNX graph: Y[tokens, out] = X[tokens, in] @ Wt[in, out]."""
    k, m = w_t.shape
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [tokens, k])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [tokens, m])
    weight = numpy_helper.from_array(w_t, name="Wt")
    node = helper.make_node("MatMul", ["X", "Wt"], ["Y"])
    graph = helper.make_graph([node], "spmm", [x], [y], initializer=[weight])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


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
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--cores", type=int, default=8, help="physical cores DeepSparse may use")
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--out", default="results/fig15a_deepsparse.csv")
    args = p.parse_args()

    from deepsparse import compile_model   # imported here so --help works without DeepSparse

    rng = np.random.default_rng(0)
    w = rng.standard_normal((args.out_features, args.in_features)).astype(np.float32)
    x = rng.standard_normal((args.tokens, args.in_features)).astype(np.float32)

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for n, m in [(1, 1)] + LEVELS:                     # (1, 1) = dense, 0% sparsity
            sparsity = 1 - n / m
            wp = prune_unstructured(w, sparsity) if sparsity > 0 else w
            path = os.path.join(tmp, f"spmm_{n}_{m}.onnx")
            onnx.save(matmul_model(np.ascontiguousarray(wp.T), args.tokens), path)
            engine = compile_model(path, batch_size=args.tokens, num_cores=args.cores)
            y = engine.run([x])[0]
            err = float(np.abs(y - x @ wp.T).max())
            med, mn, mx = median_ms(lambda: engine.run([x]), args.repeats, args.warmup)
            label = "dense" if sparsity == 0 else f"{sparsity:.0%}"
            rows.append(dict(engine="deepsparse-1.1.0", sparsity=round(sparsity, 3), median_ms=med,
                             min_ms=mn, max_ms=mx, max_abs_err=err, cores=args.cores))
            print(f"{label:>6}  DeepSparse median {med:8.2f} ms  (min {mn:.2f}, max {mx:.2f})  "
                  f"max|err| {err:.2e}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
