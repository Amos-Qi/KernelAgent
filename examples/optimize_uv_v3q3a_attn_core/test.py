"""Correctness test for the v3-q3a batched-MHA attention core.

Contract: kernel_function(*inputs) must match the eager reference Model on
the real serving shapes within bf16 tolerances. Inputs: F queries, F keys,
F values (floating -> cast to bf16 on cuda), F bool key-padding masks
(moved to cuda, kept bool). Both sides return one (F, B, H, Lq, D) tensor.
"""

import sys

import torch

# The reference does fp32 matmuls that mirror the kernel's exact numerics
# (ieee fp32 accumulate); keep TF32 out of the comparison.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

try:
    from kernel import kernel_function  # worker sandbox: candidate under test
except ImportError:  # standalone: validate the starting kernel
    from input import kernel_function

from problem import get_inputs, get_init_inputs, Model


def test_kernel():
    device = "cuda"
    dtype = torch.bfloat16

    model = Model(*get_init_inputs()).to(device)
    inputs = [
        x.to(device).to(dtype) if x.is_floating_point() else x.to(device)
        for x in get_inputs()
    ]

    with torch.no_grad():
        ref_output = model(*inputs)

    kernel_output = kernel_function(*inputs)

    if kernel_output.shape != ref_output.shape:
        print(
            f"FAIL: shape mismatch, expected {tuple(ref_output.shape)}, "
            f"got {tuple(kernel_output.shape)}"
        )
        return False

    ref32 = ref_output.float()
    out32 = kernel_output.float()
    if torch.allclose(ref32, out32, rtol=1e-2, atol=1e-2):
        print("PASS")
        return True
    max_diff = (ref32 - out32).abs().max().item()
    print(f"FAIL: max difference = {max_diff}")
    return False


if __name__ == "__main__":
    success = test_kernel()
    sys.exit(0 if success else 1)
