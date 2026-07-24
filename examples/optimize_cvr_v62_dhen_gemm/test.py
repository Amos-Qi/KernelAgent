"""Gates: fp16 parity vs the eager pair (relu-exact on the MLP half; 1-ulp
operand-scaled band for GEMM accumulation-order wobble)."""

import sys

import torch

from problem import Model, get_inputs

try:
    from kernel import kernel_function  # worker sandbox: candidate under test
except ImportError:
    from input import kernel_function

DEVICE = "cuda"
DTYPE = torch.float16  # v62 serving precision


def main() -> int:
    torch.backends.cuda.matmul.allow_tf32 = False
    model = Model().to(DEVICE)
    ok = True
    for seed in (0, 1):
        inputs = [t.to(DEVICE, DTYPE) for t in get_inputs(seed)]
        with torch.no_grad():
            ref = model(*inputs).float()
            out = kernel_function(*inputs).float()
        diff = (ref - out).abs()
        band = 2e-3 + 2e-3 * ref.abs()  # ~1 fp16 ulp at magnitude
        over = int((diff > band).sum())
        md = diff.median().item()
        good = over == 0 and md == 0.0
        print(f"parity seed={seed}: median {md:.6f}, over-band {over}, max {diff.max().item():.5f}"
              f" -> {'PASS' if good else 'FAIL'}")
        ok &= good
    print("ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
