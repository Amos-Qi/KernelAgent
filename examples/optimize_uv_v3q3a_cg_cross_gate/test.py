"""Correctness + CUDA-graph capture test for the v3-q3a-cg cross + EPNet gate (dir #6).

Two gates, both mandatory (same contract as dir #4):
1. Numeric: kernel_function(*inputs) matches the eager reference Model within
   a CANCELLATION-AWARE tolerance. The final epilogue ``x0*expand + y0`` can
   cancel O(8) operands down to O(0.1); any faithful reimplementation carries
   +/-1 ULP per stage from reduction order, and on cancelling elements that
   absolute error survives while the result shrinks — a result-relative
   tolerance rejects bit-faithful kernels there. The error budget is
   therefore scaled by the final add's OPERAND magnitudes
   (|x0*expand| + |y0| + |x|), which stays strict against real bugs (a wrong
   formula shifts the bulk, and the bulk must remain essentially exact).
2. Capture-safety: kernel_function must survive torch.cuda.CUDAGraph capture
   and produce correct results on REPLAY after the input contents are
   overwritten in place. Any per-call host work fails this gate.
"""

import sys

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

try:
    from kernel import kernel_function  # worker sandbox: candidate under test
except ImportError:  # standalone: validate the starting kernel
    from input import kernel_function

from problem import get_inputs, get_init_inputs, Model


def _fresh_contents(inputs, seed):
    """Fresh contents with identical shapes and the same distributional
    invariants (same scaling) — generated through get_inputs itself."""
    return [
        n.to(x.device).to(x.dtype)
        for x, n in zip(inputs, get_inputs(seed=seed))
    ]


def _operand_scale(inputs):
    """Forward-error scale of the final add's operands, |x0*expand| + |y0| + |x|,
    recomputed with the reference math (see module docstring)."""
    x, x0, w1, w2, u, v, b = inputs
    with torch.no_grad():
        sd = x.dtype
        h = F.silu(torch.matmul(x, w1).float()).to(sd)
        g = 2.0 * torch.sigmoid(torch.matmul(h, w2).float())
        y0 = (x.float() * g).to(sd)
        low = torch.matmul(y0, u)
        expand = (torch.matmul(low, v) + b).float()
        # |x| term: the gate stage y0 = x * 2*sigmoid(g) amplifies a 1-ULP
        # wobble in g by up to ~|x| even where y0 itself lands near zero.
        # Row term: the cross GEMM (V @ (U @ y0)) mixes each row's y0 error
        # across all D columns, so every element also carries a row-coupled
        # budget proportional to |x0| times the row's typical |x|.
        row = x.float().abs().mean(dim=1, keepdim=True)
        return (
            (x0.float() * expand).abs()
            + y0.float().abs()
            + x.float().abs()
            + x0.float().abs() * row
        )


def _close(ref, out, inputs, label):
    diff = (ref.float() - out.float()).abs()
    # Bulk exactness: ULP/reduction-order noise keeps the median diff near
    # zero; any real formula/logic bug shifts the bulk and fails here.
    med = diff.median().item()
    if med > 5e-3:
        print(f"FAIL: {label} — bulk deviates (median diff {med:.4f}); formula/logic error")
        return False
    tol = 1e-2 + 2e-2 * _operand_scale(inputs)
    if bool((diff <= tol).all()):
        return True
    worst = (diff - tol).argmax()
    print(
        f"FAIL: {label} — max over-tolerance diff {diff.flatten()[worst].item():.4f} "
        f"(tol there {tol.flatten()[worst].item():.4f}), "
        f"{int((diff > tol).sum())} elements over"
    )
    return False


def test_kernel():
    device = "cuda"
    dtype = torch.bfloat16

    model = Model(*get_init_inputs()).to(device)
    inputs = [x.to(device).to(dtype) for x in get_inputs()]

    # ---- Gate 1: numeric parity -----------------------------------------
    with torch.no_grad():
        ref_output = model(*inputs)
    kernel_output = kernel_function(*inputs)

    if kernel_output.shape != ref_output.shape:
        print(
            f"FAIL: shape mismatch, expected {tuple(ref_output.shape)}, "
            f"got {tuple(kernel_output.shape)}"
        )
        return False
    if not _close(ref_output, kernel_output, inputs, "numeric parity"):
        return False

    # ---- Gate 2: CUDA-graph capture + replay ----------------------------
    static_inputs = [x.clone() for x in inputs]
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # warmup on a side stream (capture rule)
            for _ in range(3):
                kernel_function(*static_inputs)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = kernel_function(*static_inputs)
    except Exception as exc:
        print(f"FAIL: CUDA-graph capture failed (kernel is not capture-safe): {exc}")
        return False

    fresh = _fresh_contents(inputs, seed=1234)
    for si, fi in zip(static_inputs, fresh):
        si.copy_(fi)
    graph.replay()
    torch.cuda.synchronize()

    with torch.no_grad():
        ref_replay = model(*static_inputs)
    if not _close(ref_replay, graph_output, static_inputs, "replay parity"):
        return False

    print("PASS")
    return True


if __name__ == "__main__":
    success = test_kernel()
    sys.exit(0 if success else 1)
