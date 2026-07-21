# Unity user-value model v3-q3a-cg — CROSS LAYER + EPNet GATE with their
# pointwise epilogues: the two heaviest memory-bound families in serving.
#
# Kernel dir #6: GEMM-EPILOGUE FUSION. In the deployed profile these are
# the #2 and #3 GPU kernels overall:
#   `triton_poi_fused_add_addmm_mul`  — 4.69 ms total, 90% Mem SOL, **1% L2
#     hit**: the low-rank cross layer's epilogue (x0 * (V·(U·x) + b) + x)
#     re-reads the full (rows, D) activation from DRAM after the GEMM pair.
#   `triton_poi_fused_mm_mul_sigmoid` — 2.89 ms total, 87-90% Mem SOL: the
#     EPNet gate epilogue (x * sigmoid(gate)) — another full (rows, D)
#     round-trip.
# Both epilogue kernels are "optimal" as written — they run at the memory
# roof. The ONLY speedup is moving fewer bytes: fuse the epilogue into the
# producing GEMM's store (write-once), and keep the x tile resident so the
# elementwise re-read never touches DRAM (that 1% L2 hit is the exploit).
#
# Structure (exact, from the served model; dims from config.json + the
# deployed profile):
#   rows = 24 * 1024 = 24576 (cg bucket rows);  D = 2756 (input-layer width)
#   EPNet gate: g = 2 * sigmoid(W2 @ silu(W1 @ x_domain));  y0 = x * g
#     (gate_hidden_dim = 512; x_domain here = x for a self-contained problem
#      — serving computes it from domain features; same shapes/traffic)
#   Cross (low-rank DCN, rank 128): y = x0 * (V·(U·y0) + b) + y0
#     U: (D, 128), V: (128, D), b: (D,)
#
#   [VERIFY before integration: the exact gate formula (the 2x factor and
#    silu hidden activation) against EPNet's module code, and whether the
#    gate multiplies before or after the cross in this config — the fusion
#    structure and byte math are identical either way.]
#
# Numerics contract: bf16 I/O and native-dtype matmul operands with fp32
# accumulation (ieee, no TF32); sigmoid/silu in fp32; each served stage
# stores bf16, and the reference rounds at the same boundaries.

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

ROWS = 24 * 1024
D = 2756
GATE_HIDDEN = 512
CROSS_RANK = 128


class Model(nn.Module):
    """Eager reference: the unfused sequence as served (GEMMs + separate
    pointwise epilogues, bf16 stage storage, fp32 activations)."""

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        x, x0, w1, w2, u, v, b = tensors
        sd = x.dtype

        # EPNet gate: hidden GEMM -> silu -> out GEMM -> sigmoid*2 -> mul
        h = (torch.matmul(x, w1)).float()
        h = F.silu(h).to(sd)
        g = torch.matmul(h, w2).float()
        g = 2.0 * torch.sigmoid(g)
        y0 = (x.float() * g).to(sd)  # the mm_mul_sigmoid epilogue round-trip

        # Low-rank cross: two GEMMs -> add/mul epilogue (add_addmm_mul)
        low = torch.matmul(y0, u)          # (rows, 128)
        expand = torch.matmul(low, v) + b  # (rows, D)
        y = (x0.float() * expand.float() + y0.float()).to(sd)
        return y


def get_inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(ROWS, D, generator=g)
    x0 = torch.randn(ROWS, D, generator=g)
    w1 = torch.randn(D, GATE_HIDDEN, generator=g) * (D**-0.5)
    w2 = torch.randn(GATE_HIDDEN, D, generator=g) * (GATE_HIDDEN**-0.5)
    u = torch.randn(D, CROSS_RANK, generator=g) * (D**-0.5)
    v = torch.randn(CROSS_RANK, D, generator=g) * (CROSS_RANK**-0.5)
    b = torch.randn(D, generator=g) * 0.02
    return [x, x0, w1, w2, u, v, b]


def get_init_inputs():
    return []
