"""Gates for the feature-front kernel: numeric parity vs the eager reference
(gathers must be exact; bag means / dense projection within bf16 rounding) and
CUDA-graph capture+replay (the v3-q3a-cg serving contract)."""

import sys

import torch

from problem import Model, get_inputs
from input import kernel_function

DEVICE = "cuda"
DTYPE = torch.bfloat16  # serving deploy precision


def _to_dev(ts):
    # ints (indices) keep their dtype; only float tensors move to bf16.
    return [t.to(DEVICE) if not t.is_floating_point() else t.to(DEVICE, DTYPE) for t in ts]


def _check(ref, out, label):
    diff = (ref.float() - out.float()).abs()
    exact = (diff == 0).float().mean().item()
    md, mx = diff.median().item(), diff.max().item()
    ok = md == 0.0 and mx <= 2e-2 and exact >= 0.95
    print(f"{label}: exact-frac {exact:.4f}, median {md:.6f}, max {mx:.6f} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    torch.backends.cuda.matmul.allow_tf32 = False
    model = Model().to(DEVICE)
    ok = True

    for seed in (0, 1):
        inputs = _to_dev(get_inputs(seed))
        with torch.no_grad():
            ref = model(*inputs)
            out = kernel_function(*inputs)
        ok &= _check(ref, out, f"parity seed={seed}")

    # --- capture/replay gate (dir #4 contract): capture once, mutate input
    # CONTENTS in place (same storages -> same data_ptrs), replay, compare
    # against an eager recompute of the mutated inputs.
    static = _to_dev(get_inputs(0))
    for _ in range(3):
        kernel_function(*static)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        graph_out = kernel_function(*static)
    fresh = _to_dev(get_inputs(7))
    with torch.no_grad():
        for dst, src in zip(static, fresh):
            dst.copy_(src)
    g.replay()
    torch.cuda.synchronize()
    with torch.no_grad():
        ref = Model().to(DEVICE)(*static)
    ok &= _check(ref, graph_out, "capture/replay")

    print("ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
