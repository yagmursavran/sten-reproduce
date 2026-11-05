"""Correctness test for STen's n:m:g sparse-dense multiplication kernel.

Same pruned weight, two ways of computing a Linear layer:
  1. STen's n:m:g kernel (through the SparsityBuilder / dispatcher path),
  2. plain dense PyTorch (F.linear) with the same pruned weight.
They must agree up to float rounding (~1e-6).

A third computation feeds the TRANSPOSED input to dense PyTorch. If STen matches this one instead,
STen reads its input transposed (bug in src/sten/grouped_nm/sten_impls.py, line 53).
The input is square (64x64) so that x.T is also a valid input for the layer.

Usage:
  python experiments/correctness_test.py
"""
import torch
import sten

F = torch.nn.functional
torch.manual_seed(0)

layer = torch.nn.Linear(64, 96)          # 64 inputs -> 96 outputs, weight W is 96x64

sb = sten.SparsityBuilder()
sb.set_weight(
    name="weight",
    initial_sparsifier=sten.GroupedNMSparsifier(n=2, m=4, g=2),   # keep 2 of every 4, groups of 2
    out_format=sten.GroupedNMTensor,
)
sparse_layer = sb.get_sparse_model(layer)                  # sparse copy; `layer` is unchanged
W = sparse_layer.weight.wrapped_tensor.to_dense()          # the same pruned weight as a dense tensor

x = torch.randn(64, 64)                                    # 64 rows (tokens) x 64 features

with torch.no_grad():
    sten_result = sparse_layer(x)                          # STen n:m:g kernel
    reference = F.linear(x, W, layer.bias)                 # dense PyTorch, same weight
    transposed_input = F.linear(x.T, W, layer.bias)        # what we get if the input is read transposed

print("STen vs dense reference:     ", (sten_result - reference).abs().max().item())
print("STen vs transposed input:    ", (sten_result - transposed_input).abs().max().item())
