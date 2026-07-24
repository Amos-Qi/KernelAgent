"""Parity test for the CVR v62 DHEN ensemble-sum kernel (dir #2).

BAND gate, not bit-exact: the fused kernel keeps ONE fp32 accumulator
across all three GEMM segments and rounds once, while the serving chain
rounds each addmm to fp16 before the weighted sum (plus the softmax fold
pre-rounds the packed weights). The fused result is the MORE accurate one;
differences sit at the 1-2 fp16-ulp scale of the output magnitude.
Band: |diff| <= 3e-3 + 3e-3 * |ref| with ZERO tolerance for outliers.
"""

import sys

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

try:
    from kernel import kernel_function  # worker sandbox: candidate under test
except ImportError:  # standalone: validate the starting kernel
    from input import kernel_function

from problem import Model, get_init_inputs, get_inputs


def _run_one(seed: int) -> bool:
    device = "cuda"
    dtype = torch.float16

    model = Model(*get_init_inputs()).to(device)
    inputs = [x.to(device).to(dtype) for x in get_inputs(seed=seed)]

    with torch.no_grad():
        ref = model(*inputs).float()
    out = kernel_function(*inputs).float()

    if out.shape != ref.shape:
        print(f"FAIL seed={seed}: shape {tuple(out.shape)} != {tuple(ref.shape)}")
        return False

    diff = (out - ref).abs()
    band = 3e-3 + 3e-3 * ref.abs()
    over = int((diff > band).sum().item())
    med = diff.median().item()
    mx = diff.max().item()
    status = "PASS" if over == 0 else "FAIL"
    print(
        f"parity seed={seed}: median {med:.6f}, over-band {over}, "
        f"max {mx:.5f} -> {status}"
    )
    return over == 0


def test_kernel() -> bool:
    ok = all([_run_one(0), _run_one(1234)])
    print("ALL PASS" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if test_kernel() else 1)
