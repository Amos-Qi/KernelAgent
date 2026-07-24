# dir #9 winner + hand polish (v2), bfloat16 serving contract.
# Changes vs the round-6 winner (both grounded in the run's own AOTI autotune
# tables printed at reference-compile time on this exact 2g slice):
#   1. Per-shape schedule dispatch in _mm (the winner used one config for all
#      17 GEMM shapes; the autotune tables show per-shape winners up to 40%
#      faster, e.g. 659x659 wants num_warps=8, tiny-N heads want BLOCK_N=16).
#   2. Layer-1 fused across BOTH ensemble components (weight cat N=960 -> 1920):
#      the trunk (6144x1328 bf16, ~16MB) is read ONCE per forward instead of
#      twice. Per-call cat keeps replay-correctness (no cached weight contents).
# The gate epilogue (ACT=2), fused decode/ensemble/stack kernel, and weight
# walking are the winner's, unchanged.

import torch
import triton
import triton.language as tl
from problem import (
    ADREV_CALIBRATION, GROUPS, N_COMPONENTS, OUT_COLS, PAYER_COL, RET_COL,
    ROWS, TOWER_DIMS, TOWER_HEAD_COLS, TOWER_ORDER,
)

TOWER_DIMS_L1 = [TOWER_DIMS[n][0] for n in TOWER_ORDER]

@triton.jit
def _mm_kernel(x_ptr, w_ptr, b_ptr, y_ptr, M, K, N, x_stride,
               ACT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    m_mask = offs_m < M
    n_mask = offs_n < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        k_mask = offs_k < K
        a = tl.load(x_ptr + offs_m[:, None] * x_stride + offs_k[None, :],
                    mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptr + offs_k[:, None] * N + offs_n[None, :],
                    mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(a, w, acc, out_dtype=tl.float32, input_precision="ieee")
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    y = acc + bias[None, :]
    if ACT == 1:
        y_bf = y.to(y_ptr.dtype.element_ty)
        y_f = y_bf.to(tl.float32)
        y_f = y_f / (1.0 + tl.exp(-y_f))
        y_store = y_f.to(y_ptr.dtype.element_ty)
    elif ACT == 2:
        z = y.to(y_ptr.dtype.element_ty)
        g = (1.0 / (1.0 + tl.exp(-z.to(tl.float32)))).to(y_ptr.dtype.element_ty)
        x_tile = tl.load(x_ptr + offs_m[:, None] * x_stride + offs_n[None, :],
                         mask=m_mask[:, None] & n_mask[None, :], other=0.0)
        y_store = (x_tile.to(tl.float32) * g.to(tl.float32)).to(y_ptr.dtype.element_ty)
    else:
        y_store = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], y_store,
             mask=m_mask[:, None] & n_mask[None, :])

@triton.jit
def _decode_kernel(g_iap0, g_adv0, t_ret0, t_iap0,
                   g_iap1, g_adv1, t_ret1, t_iap1,
                   out, M,
                   RET_COL: tl.constexpr, PAYER_COL: tl.constexpr,
                   ADREV_CAL: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M

    i0_0 = tl.load(g_iap0 + offs * 6 + 0, mask=mask, other=0.0).to(tl.float32)
    i0_1 = tl.load(g_iap0 + offs * 6 + 1, mask=mask, other=0.0).to(tl.float32)
    i0_2 = tl.load(g_iap0 + offs * 6 + 2, mask=mask, other=0.0).to(tl.float32)
    i0_3 = tl.load(g_iap0 + offs * 6 + 3, mask=mask, other=0.0).to(tl.float32)
    i0_4 = tl.load(g_iap0 + offs * 6 + 4, mask=mask, other=0.0).to(tl.float32)
    i0_5 = tl.load(g_iap0 + offs * 6 + 5, mask=mask, other=0.0).to(tl.float32)
    a0_0 = tl.load(g_adv0 + offs * 3 + 0, mask=mask, other=0.0).to(tl.float32)
    a0_1 = tl.load(g_adv0 + offs * 3 + 1, mask=mask, other=0.0).to(tl.float32)
    a0_2 = tl.load(g_adv0 + offs * 3 + 2, mask=mask, other=0.0).to(tl.float32)
    r0 = tl.load(t_ret0 + offs * 6 + RET_COL, mask=mask, other=0.0).to(tl.float32)
    p0 = tl.load(t_iap0 + offs * 13 + PAYER_COL, mask=mask, other=0.0).to(tl.float32)

    i1_0 = tl.load(g_iap1 + offs * 6 + 0, mask=mask, other=0.0).to(tl.float32)
    i1_1 = tl.load(g_iap1 + offs * 6 + 1, mask=mask, other=0.0).to(tl.float32)
    i1_2 = tl.load(g_iap1 + offs * 6 + 2, mask=mask, other=0.0).to(tl.float32)
    i1_3 = tl.load(g_iap1 + offs * 6 + 3, mask=mask, other=0.0).to(tl.float32)
    i1_4 = tl.load(g_iap1 + offs * 6 + 4, mask=mask, other=0.0).to(tl.float32)
    i1_5 = tl.load(g_iap1 + offs * 6 + 5, mask=mask, other=0.0).to(tl.float32)
    a1_0 = tl.load(g_adv1 + offs * 3 + 0, mask=mask, other=0.0).to(tl.float32)
    a1_1 = tl.load(g_adv1 + offs * 3 + 1, mask=mask, other=0.0).to(tl.float32)
    a1_2 = tl.load(g_adv1 + offs * 3 + 2, mask=mask, other=0.0).to(tl.float32)
    r1 = tl.load(t_ret1 + offs * 6 + RET_COL, mask=mask, other=0.0).to(tl.float32)
    p1 = tl.load(t_iap1 + offs * 13 + PAYER_COL, mask=mask, other=0.0).to(tl.float32)

    p7_0 = 1.0 / (1.0 + tl.exp(-i0_0))
    l7_0 = 10.0 * (1.0 - 2.0 / (tl.exp(2.0 * (i0_1 / 10.0)) + 1.0))
    tanh2_0 = 1.0 - 2.0 / (tl.exp(2.0 * (i0_2 / 3.0)) + 1.0)
    sp_in_0 = 3.0 * tanh2_0
    s7_0 = tl.where(sp_in_0 > 0.0, sp_in_0, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(sp_in_0)))
    v7_0 = tl.exp(l7_0 + 0.5 * s7_0 * s7_0)
    f7_0 = p7_0 * v7_0

    p28_0 = 1.0 / (1.0 + tl.exp(-i0_3))
    l28_0 = 10.0 * (1.0 - 2.0 / (tl.exp(2.0 * (i0_4 / 10.0)) + 1.0))
    tanh2_1 = 1.0 - 2.0 / (tl.exp(2.0 * (i0_5 / 3.0)) + 1.0)
    sp_in_1 = 3.0 * tanh2_1
    s28_0 = tl.where(sp_in_1 > 0.0, sp_in_1, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(sp_in_1)))
    v28_0 = tl.exp(l28_0 + 0.5 * s28_0 * s28_0)
    f28_0 = p28_0 * v28_0

    p7_1 = 1.0 / (1.0 + tl.exp(-i1_0))
    l7_1 = 10.0 * (1.0 - 2.0 / (tl.exp(2.0 * (i1_1 / 10.0)) + 1.0))
    tanh2_2 = 1.0 - 2.0 / (tl.exp(2.0 * (i1_2 / 3.0)) + 1.0)
    sp_in_2 = 3.0 * tanh2_2
    s7_1 = tl.where(sp_in_2 > 0.0, sp_in_2, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(sp_in_2)))
    v7_1 = tl.exp(l7_1 + 0.5 * s7_1 * s7_1)
    f7_1 = p7_1 * v7_1

    p28_1 = 1.0 / (1.0 + tl.exp(-i1_3))
    l28_1 = 10.0 * (1.0 - 2.0 / (tl.exp(2.0 * (i1_4 / 10.0)) + 1.0))
    tanh2_3 = 1.0 - 2.0 / (tl.exp(2.0 * (i1_5 / 3.0)) + 1.0)
    sp_in_3 = 3.0 * tanh2_3
    s28_1 = tl.where(sp_in_3 > 0.0, sp_in_3, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(sp_in_3)))
    v28_1 = tl.exp(l28_1 + 0.5 * s28_1 * s28_1)
    f28_1 = p28_1 * v28_1

    p7 = (p7_0 + p7_1) * 0.5
    v7 = (v7_0 + v7_1) * 0.5
    f7 = (f7_0 + f7_1) * 0.5
    l7 = (l7_0 + l7_1) * 0.5
    s7 = (s7_0 + s7_1) * 0.5
    p28 = (p28_0 + p28_1) * 0.5
    v28 = (v28_0 + v28_1) * 0.5
    f28 = (f28_0 + f28_1) * 0.5
    l28 = (l28_0 + l28_1) * 0.5
    s28 = (s28_0 + s28_1) * 0.5

    sp_a0_0 = tl.where(a0_0 > 0.0, a0_0, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(a0_0)))
    sp_a1_0 = tl.where(a1_0 > 0.0, a1_0, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(a1_0)))
    ad0 = (sp_a0_0 + sp_a1_0) * 0.5 * ADREV_CAL

    sp_a0_1 = tl.where(a0_1 > 0.0, a0_1, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(a0_1)))
    sp_a1_1 = tl.where(a1_1 > 0.0, a1_1, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(a1_1)))
    ad7 = (sp_a0_1 + sp_a1_1) * 0.5 * ADREV_CAL

    sp_a0_2 = tl.where(a0_2 > 0.0, a0_2, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(a0_2)))
    sp_a1_2 = tl.where(a1_2 > 0.0, a1_2, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(a1_2)))
    ad28 = (sp_a0_2 + sp_a1_2) * 0.5 * ADREV_CAL

    ret = (1.0 / (1.0 + tl.exp(-r0)) + 1.0 / (1.0 + tl.exp(-r1))) * 0.5
    payer = (1.0 / (1.0 + tl.exp(-p0)) + 1.0 / (1.0 + tl.exp(-p1))) * 0.5

    odt = out.dtype.element_ty
    tl.store(out + offs * 16 + 0, p7.to(odt), mask=mask)
    tl.store(out + offs * 16 + 1, v7.to(odt), mask=mask)
    tl.store(out + offs * 16 + 2, f7.to(odt), mask=mask)
    tl.store(out + offs * 16 + 3, l7.to(odt), mask=mask)
    tl.store(out + offs * 16 + 4, s7.to(odt), mask=mask)
    tl.store(out + offs * 16 + 5, p28.to(odt), mask=mask)
    tl.store(out + offs * 16 + 6, v28.to(odt), mask=mask)
    tl.store(out + offs * 16 + 7, f28.to(odt), mask=mask)
    tl.store(out + offs * 16 + 8, l28.to(odt), mask=mask)
    tl.store(out + offs * 16 + 9, s28.to(odt), mask=mask)
    tl.store(out + offs * 16 + 10, ad0.to(odt), mask=mask)
    tl.store(out + offs * 16 + 11, ad7.to(odt), mask=mask)
    tl.store(out + offs * 16 + 12, ad28.to(odt), mask=mask)
    tl.store(out + offs * 16 + 13, ret.to(odt), mask=mask)
    tl.store(out + offs * 16 + 14, payer.to(odt), mask=mask)

# Per-(K, N) launch schedules, taken from the AOTI reference's autotune
# tables printed by THIS dir's run on the same slice (execution-proven
# configs; ACT epilogues shift the balance little at these shapes).
# (BM, BN, BK, num_warps, num_stages)
_SCHED = {
    (1328, 1920): (128, 128, 32, 4, 3),  # dual-component layer-1 (from 1328xN winners)
    (384, 384):   (64, 128, 64, 4, 3),
    (192, 192):   (64, 64, 64, 8, 3),
    (256, 256):   (64, 128, 64, 4, 3),
    (128, 128):   (64, 64, 64, 8, 3),
    (659, 659):   (128, 64, 32, 8, 4),
    (593, 593):   (64, 128, 32, 4, 3),
    (384, 13):    (64, 16, 64, 4, 3),
    (192, 4):     (64, 16, 64, 4, 3),
    (256, 6):     (128, 16, 64, 8, 5),
    (128, 7):     (64, 16, 128, 4, 4),
    (659, 6):     (32, 16, 128, 2, 2),
    (593, 3):     (32, 16, 128, 2, 2),
}
_DEFAULT_SCHED = (128, 64, 32, 4, 4)  # the winner's single config

def _mm(x, w, b, act=0):
    m, k = x.shape
    n = w.shape[1]
    y = torch.empty(m, n, dtype=x.dtype, device=x.device)
    BM, BN, BK, warps, stages = _SCHED.get((k, n), _DEFAULT_SCHED)
    grid = (triton.cdiv(m, BM), triton.cdiv(n, BN))
    _mm_kernel[grid](x, w, b, y, m, k, n, x.stride(0),
                     ACT=act, BM=BM, BN=BN, BK=BK,
                     num_warps=warps, num_stages=stages)
    return y

def kernel_function(*tensors: torch.Tensor) -> torch.Tensor:
    trunk = tensors[0]
    w = list(tensors[1:])
    sd = trunk.dtype
    assert sd in (torch.bfloat16, torch.float32), "v3-q3a serves bfloat16"

    M = ROWS

    # Layer-1 for BOTH components in ONE GEMM: trunk read once per forward.
    # Weight order per component c (stride 40): W1_i at c*40 + i*8.
    l1_w = [w[c * 40 + i * 8] for c in range(N_COMPONENTS) for i in range(4)]
    l1_b = [w[c * 40 + i * 8 + 1] for c in range(N_COMPONENTS) for i in range(4)]
    W1cat = torch.cat(l1_w, dim=1)          # (1328, 1920)
    b1cat = torch.cat(l1_b, dim=0)
    hidden1 = _mm(trunk, W1cat, b1cat, act=1)
    h1_all = torch.split(hidden1, TOWER_DIMS_L1 * N_COMPONENTS, dim=1)

    comp_data = []
    wi = 0
    for comp in range(N_COMPONENTS):
        towers = {}
        tlogits = {}
        for ti, name in enumerate(TOWER_ORDER):
            base = wi + ti * 8
            x = _mm(h1_all[comp * 4 + ti], w[base + 2], w[base + 3], act=1)
            towers[name] = x
            gated = _mm(x, w[base + 4], w[base + 5], act=2)
            tlogits[name] = _mm(gated, w[base + 6], w[base + 7], act=0)

        wi += 32

        glogits = {}
        for gname, (seq, _hcols) in GROUPS.items():
            gin = torch.cat([t for n in seq for t in (towers[n], tlogits[n])], dim=1)
            gated = _mm(gin, w[wi], w[wi + 1], act=2)
            glogits[gname] = _mm(gated, w[wi + 2], w[wi + 3], act=0)
            wi += 4

        comp_data.append((glogits["iap_main"], glogits["adrev_main"],
                          tlogits["retention"], tlogits["iap"]))

    out = torch.zeros(M, OUT_COLS, dtype=sd, device=trunk.device)
    g_iap0, g_adv0, t_ret0, t_iap0 = comp_data[0]
    g_iap1, g_adv1, t_ret1, t_iap1 = comp_data[1]
    BLOCK = 256
    _decode_kernel[(triton.cdiv(M, BLOCK),)](
        g_iap0, g_adv0, t_ret0, t_iap0,
        g_iap1, g_adv1, t_ret1, t_iap1,
        out, M,
        RET_COL=RET_COL, PAYER_COL=PAYER_COL,
        ADREV_CAL=ADREV_CALIBRATION, BLOCK=BLOCK,
    )
    return out
