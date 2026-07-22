"""Gates for the feature-front kernel: numeric parity vs the eager reference
(gathers must be exact; bag means / dense projection within bf16 rounding) and
CUDA-graph capture+replay (the v3-q3a-cg serving contract)."""

import sys

import torch

from problem import COL_OFF, LAYOUT, Model, get_inputs

try:
    from kernel import kernel_function  # worker sandbox: candidate under test
except ImportError:
    from input import kernel_function

DEVICE = "cuda"
DTYPE = torch.bfloat16  # serving deploy precision

# Region-aware gate: gathers (attn + embed cols) are pure copies and must be
# BIT-EXACT — any diff there is an indexing bug. Bag/dense cols are arithmetic;
# a faithful kernel matches the eager rounding chain, but ties in fp32
# accumulation order may legitimately flip the last bf16 bit — allow <= 1 ulp
# at |v| < 1 (4e-3) with median 0.
_GATHER_COLS, _COMPUTE_COLS = [], []
for (_n, _kind, *_rest), _c in zip(LAYOUT, COL_OFF):
    (_GATHER_COLS if _kind in ("attn", "embed") else _COMPUTE_COLS).extend(range(_c, _c + _rest[2]))


def _to_dev(ts):
    # ints (indices) keep their dtype; only float tensors move to bf16.
    return [t.to(DEVICE) if not t.is_floating_point() else t.to(DEVICE, DTYPE) for t in ts]


def _check(ref, out, label):
    diff = (ref.float() - out.float()).abs()
    g = diff[:, _GATHER_COLS]
    c = diff[:, _COMPUTE_COLS]
    g_exact = (g == 0).float().mean().item()
    c_md, c_mx = c.median().item(), c.max().item()
    ok = g_exact == 1.0 and c_md == 0.0 and c_mx <= 4e-3
    print(
        f"{label}: gather exact-frac {g_exact:.6f} (need 1.0) | "
        f"compute median {c_md:.6f} max {c_mx:.6f} (need <=4e-3) -> {'PASS' if ok else 'FAIL'}"
    )
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
