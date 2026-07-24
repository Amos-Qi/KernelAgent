# Seed for dir #9 (trunk fan-out): a 1:1 per-op Triton mirror of the eager
# reference — every GEMM through one generic `_lin_kernel` (fp32-accumulate
# ieee, masked K so 659/593 need no CUTLASS-style alignment cliff), every
# GateLayer through `_mul_sigmoid_kernel`, decodes/stack as gatekeeper-safe
# tensor arithmetic. No host-side state, no caches: trivially capture-safe
# and replay-correct when input CONTENTS are overwritten in place.
#
# DECLARED HEADROOM (the structural moves this seed deliberately does not
# make, in expected-value order on the 1g.24gb slice):
#   1. The trunk (6144x1328 bf16) is read from DRAM once PER TOWER LAYER-1
#      GEMM (4x per component, 8x per forward). A grouped / horizontally
#      fused kernel reads each trunk M-tile ONCE and feeds all four towers
#      (total N = 384+192+256+128 = 960).
#   2. GateLayer = gate GEMM -> sigmoid-mul: carry the sigmoid-multiply as
#      the gate GEMM's epilogue (and the heads GEMM can consume the gated
#      value from registers/SMEM instead of a DRAM round-trip).
#   3. The group inputs are materialized cats (659/593 wide); a fused kernel
#      can index the tower outputs directly and never build them.
#   4. The decode pointwise swarm (sigmoid/softplus/exp on 1-3 col slices)
#      and the 2-component mean/calibration/stack are all tiny launches --
#      one wide epilogue kernel covers them.

import torch
import triton
import triton.language as tl

from problem import (
    ADREV_CALIBRATION, GROUPS, N_COMPONENTS, OUT_COLS, PAYER_COL, RET_COL,
    ROWS, TOWER_DIMS, TOWER_HEAD_COLS, TOWER_ORDER,
)


@triton.jit
def _lin_kernel(x_ptr, w_ptr, b_ptr, y_ptr, M, K, N,
                ACT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """y = act(x @ w + b): fp32 accumulate (ieee), single round to the I/O
    dtype after the bias (addmm semantics); ACT==1 applies silu on the
    ROUNDED value in fp32 and rounds again (mirrors F.silu(addmm(...)))."""
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
        a = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptr + offs_k[:, None] * N + offs_n[None, :],
                    mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(a, w, acc, out_dtype=tl.float32, input_precision="ieee")
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    y = (acc + bias[None, :]).to(y_ptr.dtype.element_ty)
    if ACT == 1:
        f = y.to(tl.float32)
        f = f / (1.0 + tl.exp(-f))
        y = f.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], y,
             mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _mul_sigmoid_kernel(x_ptr, z_ptr, y_ptr, NUMEL, BLOCK: tl.constexpr):
    """y = x * sigmoid(z), elementwise on contiguous same-shape tensors.
    sigmoid computed in fp32 and rounded to the I/O dtype BEFORE the multiply
    (mirrors torch.sigmoid(bf16) -> bf16 gate, then x * gate)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NUMEL
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    z = tl.load(z_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = (1.0 / (1.0 + tl.exp(-z))).to(y_ptr.dtype.element_ty)
    y = (x.to(tl.float32) * g.to(tl.float32)).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y, mask=mask)


def _lin(x, w, b, act=0):
    m, k = x.shape
    n = w.shape[1]
    y = torch.empty(m, n, dtype=x.dtype, device=x.device)
    _lin_kernel[(triton.cdiv(m, 64), triton.cdiv(n, 64))](
        x.contiguous(), w, b, y, m, k, n, ACT=act, BM=64, BN=64, BK=32,
        num_warps=4, num_stages=3,
    )
    return y


def _gate(x, wg, bg):
    z = _lin(x, wg, bg, act=0)
    y = torch.empty_like(x)
    numel = x.numel()
    _mul_sigmoid_kernel[(triton.cdiv(numel, 1024),)](
        x.contiguous(), z, y, numel, BLOCK=1024, num_warps=4,
    )
    return y


def _sigmoid(t):
    return 1.0 / (1.0 + torch.exp(-t))


def _softplus(t):
    # torch-form softplus, overflow-safe: max(t,0) + log1p(exp(-|t|))
    return torch.clamp(t, min=0.0) + torch.log1p(torch.exp(-torch.abs(t)))


def kernel_function(*tensors: torch.Tensor) -> torch.Tensor:
    trunk = tensors[0]
    w = list(tensors[1:])
    sd = trunk.dtype

    per_component = []
    wi = 0
    for _ in range(N_COMPONENTS):
        towers = {}
        tlogits = {}
        for name in TOWER_ORDER:
            x = trunk
            for _layer in range(2):
                x = _lin(x, w[wi], w[wi + 1], act=1)
                wi += 2
            towers[name] = x
            gated = _gate(x, w[wi], w[wi + 1])
            wi += 2
            tlogits[name] = _lin(gated, w[wi], w[wi + 1], act=0)
            wi += 2

        glogits = {}
        for gname, (seq, _hcols) in GROUPS.items():
            gin = torch.cat([t for n in seq for t in (towers[n], tlogits[n])], dim=1)
            gated = _gate(gin, w[wi], w[wi + 1])
            wi += 2
            glogits[gname] = _lin(gated, w[wi], w[wi + 1], act=0)
            wi += 2

        iap = glogits["iap_main"].float()
        adrev = glogits["adrev_main"].float()
        ret_d7 = tlogits["retention"][:, RET_COL : RET_COL + 1].float()
        payer_d7 = tlogits["iap"][:, PAYER_COL : PAYER_COL + 1].float()

        def ziln(t):
            prob = _sigmoid(t[:, 0:1])
            scale = _softplus(t[:, 2:3])
            value = torch.exp(t[:, 1:2] + 0.5 * torch.square(scale))
            return prob, value, prob * value, t[:, 1:2], scale

        p7, v7, f7, l7, s7 = ziln(iap[:, 0:3])
        p28, v28, f28, l28, s28 = ziln(iap[:, 3:6])
        cols = [
            p7, v7, f7, l7, s7,
            p28, v28, f28, l28, s28,
            _softplus(adrev[:, 0:1]),
            _softplus(adrev[:, 1:2]),
            _softplus(adrev[:, 2:3]),
            _sigmoid(ret_d7),
            _sigmoid(payer_d7),
        ]
        per_component.append(torch.cat(cols, dim=1))

    mean = (per_component[0] + per_component[1]) * 0.5
    mean[:, 10:13] = mean[:, 10:13] * ADREV_CALIBRATION
    out = torch.cat([mean, torch.zeros(ROWS, 1, device=mean.device)], dim=1)
    return out.to(sd)
