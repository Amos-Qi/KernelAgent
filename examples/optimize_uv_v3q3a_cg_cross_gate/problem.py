# Unity user-value model v3-q3a-cg — CROSS LAYER + EPNet GATE with their
# pointwise epilogues: the two heaviest memory-bound families in serving.
#
# Kernel dir #6: GEMM-EPILOGUE FUSION. In the deployed profile these are
# the #2 and #3 GPU kernels overall:
#   `triton_poi_fused_add_addmm_mul`  — the low-rank cross layer's epilogue
#     re-reading the full (rows, D) activation from DRAM after the GEMM pair
#     (~90% Mem SOL, ~1% L2 hit).
#   `triton_poi_fused_mm_mul_sigmoid` — the EPNet gate epilogue, another
#     full (rows, D) round-trip (~87-90% Mem SOL).
# Both epilogue kernels are "optimal" as written — they run at the memory
# roof. The ONLY speedup is moving fewer bytes: fuse the epilogue into the
# producing GEMM's store (write-once), and keep the x tile resident so the
# elementwise re-read never touches DRAM (that 1% L2 hit is the exploit).
#
# Structure and dims are VERIFIED against the deployed v3_q3a checkpoint and
# module source (VERIFIED_DIMS.md; layers/gate.py::GateNU,
# layers/cross.py::CrossLayerV2). NOTE: this dir originally ran the search at
# D = 2756 — a stale config-dump width, 2.08x the real trunk — and that
# winner did not transfer to serving (underutilized at the true shape). Dims
# below are the real ones; re-tune before integrating any prior winner.
#
#   rows = 24 * 1024 = 24576 (cg bucket rows)
#   D = 1328  # POST-KALIGN trunk (raw 1322 + 6 zero pad cols, as served) (trunk width), DOM = 144  # POST-KALIGN padded EPNet domain (raw 131-or-134 -> ceil16 = 144 either way) (domain features)
#   gate: input 1453 = DOM + D, hidden 512, gamma = 2.0
#   cross: low-rank 128, layer 0 of a 1-layer stack -> x0 == xl == y0,
#          ONE aliased tensor (not two independent streams)
#
#   EPNet gate (GateNU):              Low-rank cross (CrossLayerV2):
#     xc = cat([dom, x], -1)            low = y0 @ w_down        (rounds)
#     h  = relu(addmm(b1, xc, w1))      t   = low @ w_up         (rounds)
#          (addmm rounds ONCE)          t   = t + cb             (rounds)
#     s  = sigmoid(addmm(b2, h, w2))    y   = y0 * t + y0        (mul rounds,
#          (addmm rounds; sig rounds)                             add rounds)
#     l2 = s * 2.0            (rounds)
#     y0 = l2 * x             (rounds)
#
#   Module -> math weight mapping (materialize contiguous at swap time):
#     w1 = linear1.weight.T (1453, 512),  b1 = linear1.bias (512,)
#     w2 = linear2.weight.T (512, 1322),  b2 = linear2.bias (1322,)
#     w_down = U.T (1322, 128), w_up = V.T (128, 1322), cb = bias (1322,)
#     (In CrossLayerV2, U is applied FIRST via .T — U names the
#      down-projection, V the up-projection.)
#
# Numerics contract: bf16 I/O; matmuls accumulate fp32 (ieee, no TF32);
# every eager op boundary rounds to the I/O dtype exactly where the module
# code rounds (the reference below IS the module op sequence, inlined; a
# fp32 run makes every round a no-op).
#
# SERVING-ENVELOPE REVIVAL (2026-07-24): on the MIG 1g.24gb slice at R=6 the
# split epilogues are BACK at multi-ms scale in the optimized artifact --
# tem_fused_addmm_relu class 13% of GPU time + poi_fused_mm_mul_sigmoid 4.5%
# -- which is exactly the revival condition recorded when this dir was
# removed. Dims updated to POST-KALIGN serving (D 1328, domain padded 144,
# gate input 1472); target hardware = the 1g.24gb slice (46 SMs, 448 GB/s,
# 99KB smem ceiling) via strategy beam_search_uv_mig. VERIFY at integration:
# the RAW domain width (131 in this dir's original checkpoint read vs 134 =
# 13+18+63+40 in the input-layer arithmetic) -- the padded GEMM shapes are
# identical either way, only synthetic-input composition differs.

import torch
import torch.nn as nn

ROWS = 6 * 1024  # SERVING shape: prod max_batch_size=6 x 1024 bucket (cands capped at 1000)
D = 1328  # POST-KALIGN trunk (raw 1322 + 6 zero pad cols, as served)
DOM = 144  # POST-KALIGN padded EPNet domain (raw 131-or-134 -> ceil16 = 144 either way)
GATE_IN = DOM + D  # 1453
GATE_HIDDEN = 512
CROSS_RANK = 128
GAMMA = 2.0


class Model(nn.Module):
    """Eager reference: GateNU + CrossLayerV2(layer 0) exactly as served
    (GEMMs + separate pointwise epilogues, per-op storage-dtype rounding)."""

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        x, dom, w1, b1, w2, b2, w_down, w_up, cb = tensors

        # GateNU (inference: the .detach() on x is a no-op)
        xc = torch.cat([dom, x], dim=-1)
        h = torch.relu(torch.addmm(b1, xc, w1))
        l2 = torch.sigmoid(torch.addmm(b2, h, w2)) * GAMMA
        y0 = l2 * x  # the mm_mul_sigmoid epilogue round-trip

        # CrossLayerV2 layer 0 (dropout is Identity at inference);
        # x0 == xl == y0. The .to() mirrors the module (no-op at uniform dtype).
        low = torch.matmul(y0, w_down)
        t = torch.matmul(low, w_up).to(y0.dtype) + cb
        return y0 * t + y0  # the add_addmm_mul epilogue round-trip


def get_inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(ROWS, D, generator=g)
    x[:, 1322:] = 0.0  # kalign pad columns are exact zeros as served
    dom = torch.randn(ROWS, DOM, generator=g)
    w1 = torch.randn(GATE_IN, GATE_HIDDEN, generator=g) * (GATE_IN**-0.5)
    b1 = torch.randn(GATE_HIDDEN, generator=g) * 0.02
    w2 = torch.randn(GATE_HIDDEN, D, generator=g) * (GATE_HIDDEN**-0.5)
    b2 = torch.randn(D, generator=g) * 0.02
    w_down = torch.randn(D, CROSS_RANK, generator=g) * (D**-0.5)
    w_up = torch.randn(CROSS_RANK, D, generator=g) * (CROSS_RANK**-0.5)
    cb = torch.randn(D, generator=g) * 0.02
    return [x, dom, w1, b1, w2, b2, w_down, w_up, cb]


def get_init_inputs():
    return []
