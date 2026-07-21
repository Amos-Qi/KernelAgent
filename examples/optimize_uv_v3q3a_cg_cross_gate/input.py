# Seed kernel for dir #6 (cross + EPNet gate): the UNFUSED incumbent — the
# exact GEMM + separate-pointwise-epilogue sequence serving runs (compiled by
# Inductor into `mm_mul_sigmoid` and `add_addmm_mul` at 87-90% Mem SOL).
# The target is byte reduction: fuse each epilogue into its producing GEMM's
# store, keep the x tile L2/register-resident across the elementwise re-read
# (the deployed epilogue's 1% L2 hit rate is the exploit), and consider
# fusing gate-out into cross-U (both consume y0).
#
# Capture-safety: pure device-side ops, static shapes -> capture-legal as-is.

import torch

from problem import Model

_model = Model()


def kernel_function(*tensors: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return _model(*tensors)
