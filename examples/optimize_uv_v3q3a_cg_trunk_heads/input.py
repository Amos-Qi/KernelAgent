# Seed kernel for dir #5 (trunk tail): the UNFUSED incumbent — the same
# per-op sequence Inductor currently compiles into the 60-/27-callsite
# pointwise swarm plus dozens of tiny head GEMM launches. Intentionally seeded
# as plain torch so the search space starts from the served structure; the
# optimization target is horizontal fusion (heads x windows x components into
# few wide kernels), NOT tuning any single op.
#
# Capture-safety: pure device-side torch ops, static shapes -> capture-legal
# as-is (gate 2 passes on the seed). Candidates must preserve that.

import torch

from problem import Model

_model = Model()


def kernel_function(*tensors: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return _model(*tensors)
