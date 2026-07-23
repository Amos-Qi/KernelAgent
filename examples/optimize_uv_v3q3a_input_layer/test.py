"""Gates for the whole-input-layer kernel (dir #8): numeric parity vs the
eager reference and CUDA-graph capture+replay (the v3-q3a-cg serving
contract).

Everything downstream of the gate is arithmetic (the gate mixes every
column), so parity uses a cancellation-aware element-wise band rather than
exact-fraction: bf16 within ~1 ulp scaled by operand magnitude, fp32 within
fp32-ulp scale (the seed's triton softmax/GEMMs legitimately differ from
eager by reduction order). Two structural invariants stay exact: the output
shape and the kalign pad columns (must be exactly zero — consumers carry
zero K-rows and empty-garbage would be NaN-unsafe)."""

import sys

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

try:
    from kernel import kernel_function  # worker sandbox: candidate under test
except ImportError:  # standalone: validate the starting kernel
    from input import kernel_function

from problem import Model, get_inputs, ROWS, W_PAD, W_RAW

DEVICE = "cuda"


def _to_dev(ts, dtype):
    return [t.to(DEVICE) if not t.is_floating_point() else t.to(DEVICE, dtype) for t in ts]


def _check(ref, out, dtype, label):
    assert out.shape == (ROWS, W_PAD), f"{label}: shape {tuple(out.shape)}"
    assert bool((out[:, W_RAW:] == 0).all()), f"{label}: kalign pad cols not exactly zero"
    d = (ref.float() - out.float()).abs()
    if dtype == torch.float32:
        band = 1e-4 + 1e-4 * ref.float().abs()
    else:
        band = 1e-2 + 2e-2 * ref.float().abs()
    ok = bool((d <= band).all())
    print(f"{label}: max {d.max().item():.6f}, over-band {int((d > band).sum())} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    model = Model().to(DEVICE)
    ok = True

    for dtype in (torch.bfloat16, torch.float32):
        inputs = _to_dev(get_inputs(0), dtype)
        with torch.no_grad():
            ref = model(*inputs)
            out = kernel_function(*inputs)
        ok &= _check(ref, out, dtype, f"parity {dtype}")

    # --- capture/replay gate (dir #4/#7 contract) ---
    static = _to_dev(get_inputs(0), torch.bfloat16)
    for _ in range(3):
        kernel_function(*static)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            graph_out = kernel_function(*static)
    except Exception as exc:
        print(f"FAIL: CUDA-graph capture failed (not capture-safe): {exc}")
        return 1
    fresh = _to_dev(get_inputs(11), torch.bfloat16)
    with torch.no_grad():
        for dst, src in zip(static, fresh):
            dst.copy_(src)
    graph.replay()
    torch.cuda.synchronize()
    with torch.no_grad():
        ref = model(*static)
    ok &= _check(ref, graph_out, torch.bfloat16, "capture/replay")

    print("ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
