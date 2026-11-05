# STen Reproduction – Report Notes

Everything that should go into the report is collected here: environment, deviations, findings, obstacles.
Marks: ✅ = observed on one of our own machines | 🔬 = not yet verified on our machines

## 1. Sources
- Paper: Ivanov et al., "STen: Productive and Efficient Sparsity in PyTorch", ACM TACO 23(2), Art. 74, June 2026. doi:10.1145/3815424
- Original code: https://github.com/spcl/sten, commit 9110245 (2025-03-22, "doublesparse")
- Our fork: https://github.com/yagmursavran/sten-reproduce

## 2. Hardware
| Machine | CPU | GPU | RAM | Role |
|---|---|---|---|---|
| Victus 16 | AMD Ryzen 7 7840HS (8 cores / 16 threads, AVX2 + AVX-512) | RTX 4050 6 GB | ? (WSL sees 7.4 GiB) | CPU measurements |
| 2nd laptop | AMD Ryzen 7 7840HS | RTX 3050 6 GB | ? | Training experiments |
| Mac | Apple M4 (ARM) | integrated | ? | ARM port (candidate) |

Hardware in the paper: AMD EPYC 7742 (64 cores, AVX2), Intel Xeon Gold 6130 (16 cores, AVX-512), 3x RTX 3090, V100, Piz Daint (P100).

## 3. Software environment (paper vs ours)
| Component | Paper | Ours | Note |
|---|---|---|---|
| OS | not stated | Ubuntu 24.04 (WSL2) | |
| Python | not stated | 3.10.22 (via uv) | DaCe 0.14.4 requires <3.12; DeepSparse 1.1.0 requires <=3.10 |
| PyTorch | 1.12 | 2.4.1 | 2.15 breaks STen's fallback path; 1.12 may be tried later |
| CUDA | 11.4 | (to be filled in from the GPU setup) | |
| GCC | 8.4.1 | 13.x (Ubuntu 24.04) | |
| DaCe | not stated | 0.14.4 | 1.0+ breaks the n:m:g code generator |
| networkx | not stated | 3.4.2 | 3.5+ breaks DaCe 0.14 |
| transformers | not stated | 4.44.2 | newer versions require PyTorch >=2.6 |
| numpy | not stated | 1.26.4 | |

## 4. Obstacles during installation
1. ✅ STen's pyproject.toml pins no package versions. `pip install .` always installs the newest versions; two attempts produced different versions (DaCe 1.0.2 / 0.16).
2. ✅ `pip install .` (including GPU PyTorch) took more than 40 minutes on a slow connection.
3. ✅ With the newest versions (torch 2.15.0, dace 1.0.2, networkx 3.7, transformers 5.22.1), 3 STen tests fail:
   - test_modify_bert_encoder: torch.hub looks for HuggingFace's hubconf.py, which has been removed from the HuggingFace repository (FileNotFoundError). This test can no longer pass with any version.
   - test_fallback_implementations: in torch 2.15 the signature of torch.mm changed; STen's fallback path passes an extra None (TypeError). Passes with torch 2.4.1.
   - test_clone: a gap left as a TODO in the authors' code; also fails with torch 2.4.1.
   - ✅ Berfin/Victus, 3 Nov, pinned environment (torch 2.4.1): 2 failed, 20 passed (test_modify_bert_encoder, test_clone).
4. ✅ STen uses old PyTorch APIs (TypedStorage, torch.distributed.reduce_op): deprecation warnings with newer versions.
5. ✅ DaCe 0.14.4 does not support Python 3.12 (pip: "Requires-Python >=3.6,<3.12"). Therefore a Python 3.10 environment was set up.
6. 🔬 DaCe 1.0.2 + networkx 3.6+: "KeyError: SDFGState" while compiling the n:m:g code. Cause: networkx 3.5+; fixed with networkx 3.4.2. (Seen in our test environment, not on the team machines.)
7. 🔬 DaCe 0.16 + sympy 1.14: "TypeError: 'Attr' object is not callable" during n:m:g compilation.
8. 🔬 TVM 0.10.0 (used in the paper) is not on PyPI; PyPI only has 0.25+. Relay and te.Schedule were removed in TVM 0.20, so the paper's TVM experiment cannot be repeated one-to-one with a new version.
9. 🔬 DeepSparse 1.1.0 only ships packages for Linux x86_64 and Python 3.6–3.10. The latest release, 1.9.0 (June 2025), is source-only.
10. ✅ GitHub push timed out on some networks (SSH port 22 blocked). Fixed by using SSH over port 443 (ssh.github.com) in ~/.ssh/config.

## 5. Bug found in the code: the n:m:g multiplication reads its input transposed
- Location: src/sten/grouped_nm/sten_impls.py, line 53: `trans_b = not dense.is_contiguous()`
- The linear() implementation passes trans_b=True (line 166); this line overwrites it. The input coming from PyTorch is contiguous, so trans_b becomes False and the kernel reads the input transposed. The comment above the line (line 52) contradicts what the code does.
- Evidence (experiments/correctness_test.py, 64x64 input, 2:4:2):
  - STen vs dense reference: ~2.75 (wrong)
  - STen vs linear(x.T, W): ~4.8e-07 (i.e., the input is read transposed)
- Speed is not affected (same number of operations). Since the paper measures accuracy with masked dense training, the paper's accuracy results are probably not affected.
- Proposed fix: `trans_b = trans_b and dense.is_contiguous()` → error drops to ~1e-6 (tested only in our test environment so far).
- Status: ✅ confirmed on Berfin/Victus (3 Nov): STen vs reference = 2.7545, STen vs linear(x.T, W) = 4.77e-07.
- DaCe warning during compilation: "Incompatible strides: inner (4, 1, 6144, 1536, 96) - outer (4, 1, 1536, 96)". Relation to the bug unknown.
- Important: STen's own tests (tests/test_nmg.py) do not check the numerical result, which is why the bug was not caught.

## 6. Gaps in the paper (from a reproduction point of view)
- Only the PyTorch, CUDA and GCC versions are given; no versions for the other dependencies.
- No scripts in the repository for Fig. 12 (energy), Fig. 15 (DeepSparse/TVM comparison) or Table 3 (OPT + SparseGPT).
- The group size g used in Fig. 15b is not stated (the authors' test tests/test_nmg.py uses 3:6:4 on bert-base, 8x512).
- The abstract says "best available implementation"; the code has no cost model, it only looks for an exact match.

## 7. Experiment log

### 3 Nov – Fig. 15b, small scale (bert-small, original code)
- Victus, Ryzen 7 7840HS, on AC power, OMP_NUM_THREADS=8, torch 2.4.1
- Input 8x128, g=4, kernel avx512 (auto-selected), median of 10 repeats, 2 warm-up runs
- Script: experiments/bert_latency.py | Results: results/bert_small.csv, results/bert_small_run2.csv, results/bert_small_latency.png

| Format | Sparsity | Run 1 | Run 2 | Speedup 1 | Speedup 2 |
|---|---|---|---|---|---|
| dense | 0% | 80.0 ms | 82.2 ms | – | – |
| 3:6:4 | 50% | 59.2 ms | 61.4 ms | 1.35x | 1.34x |
| 2:7:4 | 71% | 53.7 ms | 52.8 ms | 1.49x | 1.56x |
| 1:10:4 | 90% | 36.6 ms | 38.2 ms | 2.19x | 2.15x |

- max|err| 6.29 / 5.74 / 4.84 (identical in both runs): the transpose bug shows up in the whole model; it does not affect the speed measurement
- First run: 3 compilations per format (3 different matrix shapes); weight conversion ~35 s per format
- Same trend as the paper (sparsity ↑ → speedup ↑). The paper reports up to 3.2x on BERT-base; in a small model the matrices are small and non-matmul work has a larger share → lower speedup expected

### 3 Nov – Fig. 12 energy (bert-base-uncased, layer 8, attention query, 768x768)
- Script: experiments/energy.py | Results: results/energy.csv, results/energy.png
- Energy = ||pruned||₁ / ||original||₁
- Same ranking as the paper at all 7 levels:
  unstructured > n:m > n:m:g g=16 > g=4 > g=1 > blocked b=2 > b=4 > b=8
- 50% (3:6): unstructured 0.798, n:m 0.760, g=16 0.754, g=4 0.744, g=1 0.720, b=2 0.655, b=4 0.582, b=8 0.549
- 90% (1:10): unstructured 0.262, n:m 0.237, g=16 0.232, g=4 0.225, g=1 0.207, b=2 0.177, b=4 0.139, b=8 0.123
- Paper values at 50% (read from the figure, approx.): 0.80, 0.76, 0.75, 0.74, 0.72, 0.62, 0.56, 0.53 → unstructured, n:m and n:m:g match almost exactly; our blocked values are ~0.02–0.03 higher (our own blocked implementation; the paper's code for it is not available)
- g=16 is very close to n:m (difference 0.006 at 50%); g=4 keeps ~95–98% of n:m
- At 90%, n:m:g g=4 keeps ~60% more energy than blocked b=4
- Together with the speed experiment: 1:10:4 → ~2.2x faster, energy close to unstructured

### 5 Nov – Does dense PyTorch get faster with zeros? (experiments/pytorch_zeros_check.py)
- Linear, input 1024x2048, weight 512x2048, 8 threads, Victus
- Dense PyTorch: 0% 4.69 ms, 50% 4.75, 90% 4.86, 99% 4.94 → independent of the fraction of zeros
- PyTorch CSR: 0% 53.04, 50% 34.72, 90% 9.51, 99% 3.58 → beats dense only at 99%
- Conclusion: showing PyTorch as a single "Dense" bar in Fig. 15b is correct; unstructured sparsity is slow at moderate levels → motivation for a structured format such as n:m:g

### 5 Nov – Fig. 15b, full setting (bert-base, 8x512, 7 levels, original code)
- Victus, Ryzen 7 7840HS, 8 threads, on AC power, torch 2.4.1, g=4 (not stated in the paper; value from the authors' tests/test_nmg.py), median of 10 repeats after 2 warm-up runs
- Script: experiments/bert_latency.py | Results: results/bert_base_run1.*, run2.*, run2_part2.*, run2_merged.csv, fig15b_bert_base.png
- Dense PyTorch: 1934.5 ms (run 1), 1907.6 ms (run 2) — paper ~2.87 s
- Speedup (run 1 / run 2): 3:6 1.29/1.30, 2:5 1.53/1.51, 2:6 1.55/1.55, 2:7 1.73/1.64, 1:5 1.83/1.73, 1:10 2.06/2.06, 1:20 2.44/2.34
- Paper (read from the figure, approx.): 50% ~2.3x, 90% ~3.0x, 95% ~3.2x → same trend, lower values for us.
  Possible reason: our dense baseline is faster (PyTorch 2.4 vs 1.12, different CPU) — not verified
- Comparing the stacked bars: the paper's "PyTorch" part of the STen run is ~0.73 s, ours ("other") ~0.42 s; the paper's STen part is ~0.51 s at 50% and ~0.17 s at 95%, ours ~1.03 s and ~0.37 s. So the gap comes mainly from the n:m:g Linear part, not from the rest of the model. To investigate: AVX2 vs AVX-512 kernel on Zen 4, thread count
- Time breakdown: linear 1605 → 373 ms (decreases with sparsity); other ~395–475 ms (roughly constant, ~330 in dense).
  At 90% and 95%, other > linear → the non-sparse part dominates; ceiling ≈ 1934/420 ≈ 4.6x
- In the STen model "other" is ~100 ms higher than in dense: probably the memory layout of STen's Linear output (transposed + sliced, sten_impls.py lines 168–170) slows down the following operations — to investigate
- max|err| 6.4–11.0 (transpose bug)
- Run 2 was killed by the out-of-memory killer (OOM, 7.3 GB) while converting 1:20:4. 1:10 and 1:20 were re-measured in a separate run (run2_part2); the 1:10 value from before the kill (1.84x, other 564 ms) is considered corrupted by memory pressure
