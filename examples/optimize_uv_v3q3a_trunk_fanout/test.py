"""Correctness + CUDA-graph capture test for the v3-q3a-cg trunk fan-out (dir #9).

Two gates, both mandatory (same contract as dir #4):
1. Numeric: kernel_function(*inputs) matches the eager reference Model within
   bf16 tolerances (floating inputs cast to bf16 on cuda).
2. Capture-safety: kernel_function must survive torch.cuda.CUDAGraph capture
   and produce correct results on REPLAY after the input contents are
   overwritten in place. Any per-call host work fails this gate.
"""

import sys

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

try:
    from kernel import kernel_function  # worker sandbox: candidate under test
except ImportError:  # standalone: validate the starting kernel
    from input import kernel_function

from problem import get_inputs, get_init_inputs, Model


def _fresh_contents(inputs, seed):
    """Fresh contents with identical shapes and the same distributional
    invariants (head-weight down-scaling keeps the ziln exp finite) —
    generated through get_inputs itself."""
    return [
        n.to(x.device).to(x.dtype)
        for x, n in zip(inputs, get_inputs(seed=seed))
    ]


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
    if not torch.allclose(ref_output.float(), kernel_output.float(), rtol=1e-2, atol=1e-2):
        max_diff = (ref_output.float() - kernel_output.float()).abs().max().item()
        print(f"FAIL: max difference = {max_diff}")
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
    if not torch.allclose(ref_replay.float(), graph_output.float(), rtol=1e-2, atol=1e-2):
        max_diff = (ref_replay.float() - graph_output.float()).abs().max().item()
        print(f"FAIL: replay output wrong (max difference = {max_diff}) — "
              "kernel captured stale state or does host-side work per call")
        return False

    print("PASS")
    return True


if __name__ == "__main__":
    success = test_kernel()
    sys.exit(0 if success else 1)
