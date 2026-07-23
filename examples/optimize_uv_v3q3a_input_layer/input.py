# Starting kernel for dir #8 (whole input-layer forward): the SHIPPED stack
# composed — dir #7's arena feature-front kernels + dirs #1-#4's ragged
# attention pipeline — with the not-yet-fused stages left as EAGER torch:
# K/V + query assembly (gathers, cats, pad masks), the EPNet mask/iap/domain
# chain, and the GateNU gate. Those eager stages ARE the optimization
# headroom, alongside cross-stage fusions (masks inside gathers, Q-projection
# into the attention core, domain emitted by the front) and schedule tuning
# of the shipped kernels. Capture-safety contract as dirs #4/#7: after the
# first (warmup) call, steady-state does no host->device meta builds.

from typing import List

import torch
import triton
import triton.language as tl

@triton.jit
def _ragged_gemm_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    m_sizes,
    k_sizes,
    m_offsets,
    x_offsets,
    N: tl.constexpr,
    MAX_KV,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    g = tl.program_id(2)

    M = tl.load(m_sizes + g)
    K = tl.load(k_sizes + g)
    m_off = tl.load(m_offsets + g)
    x_off = tl.load(x_offsets + g)

    if pid_m * BLOCK_M >= M:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    x_base = x_ptr + x_off
    w_base = w_ptr + g * (MAX_KV * N)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        x = tl.load(
            x_base + offs_m[:, None] * K + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        w = tl.load(
            w_base + offs_k[:, None] * N + offs_n[None, :],
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc = tl.dot(x, w, acc, out_dtype=tl.float32, input_precision="ieee")

    b_vals = tl.load(b_ptr + g * N + offs_n, mask=n_mask, other=0.0)
    acc = acc + b_vals[None, :]

    y_ptrs = y_ptr + (m_off + offs_m[:, None]) * N + offs_n[None, :]
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _attn_core_ragged_kernel(
    q_ptr,
    y_ptr,
    bias_k_ptr,
    bias_v_ptr,
    pad_ptr,
    out_ptr,
    m_offs,
    s_sizes,
    B,
    Lq,
    scale,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_f = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_bh = tl.program_id(2)
    b = pid_bh // H
    h = pid_bh % H

    S_i = tl.load(s_sizes + pid_f)
    m_off = tl.load(m_offs + pid_f)
    S_ext = S_i + 2

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    m_valid = offs_m < Lq
    n_valid = offs_n < S_ext
    n_in_range = offs_n < S_i
    n_is_bias = offs_n == S_i

    kv_row = m_off + b * S_i + offs_n

    q_base = q_ptr + pid_f * (B * Lq * E) + b * (Lq * E)
    q = tl.load(
        q_base + offs_m[:, None] * E + (h * D + offs_d[None, :]),
        mask=m_valid[:, None],
        other=0.0,
    )

    k_ragged = tl.load(
        y_ptr + kv_row[:, None] * (2 * E) + (h * D + offs_d[None, :]),
        mask=n_in_range[:, None],
        other=0.0,
    )
    k_bias = tl.load(bias_k_ptr + pid_f * E + h * D + offs_d)
    k = tl.where(n_in_range[:, None], k_ragged, 0.0)
    k = tl.where(n_is_bias[:, None], k_bias[None, :].to(k.dtype), k)

    kt = tl.trans(k)
    scores = (tl.dot(q, kt, out_dtype=tl.float32, input_precision="ieee") * scale).to(tl.float32)

    add_mask = tl.load(pad_ptr + kv_row, mask=n_in_range, other=0.0)
    scores = scores + add_mask[None, :]
    scores = tl.where(n_valid[None, :], scores, float("-inf"))

    m_i = tl.max(scores, axis=1)
    p = tl.exp(scores - m_i[:, None])
    p = p / tl.sum(p, axis=1)[:, None]  # softmax stays fp32

    v_ragged = tl.load(
        y_ptr + kv_row[:, None] * (2 * E) + (E + h * D + offs_d[None, :]),
        mask=n_in_range[:, None],
        other=0.0,
    )
    v_bias = tl.load(bias_v_ptr + pid_f * E + h * D + offs_d)
    v = tl.where(n_in_range[:, None], v_ragged, 0.0)
    v = tl.where(n_is_bias[:, None], v_bias[None, :].to(v.dtype), v)

    out = tl.dot(p, v.to(tl.float32), input_precision="ieee")

    o_base = out_ptr + pid_f * (B * Lq * E) + b * (Lq * E)
    tl.store(
        o_base + offs_m[:, None] * E + (h * D + offs_d[None, :]),
        out.to(out_ptr.dtype.element_ty),
        mask=m_valid[:, None],
    )


@triton.jit
def _proj_ee_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    R,
    E: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_r = tl.program_id(0)
    f = tl.program_id(1)

    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_e = tl.arange(0, E)
    r_mask = offs_r < R

    x = tl.load(
        x_ptr + f * (R * E) + offs_r[:, None] * E + offs_e[None, :],
        mask=r_mask[:, None],
        other=0.0,
    )
    w = tl.load(w_ptr + f * (E * E) + offs_e[:, None] * E + offs_e[None, :])
    acc = tl.dot(x, w, out_dtype=tl.float32, input_precision="ieee")
    # served double rounding: round the matmul, then round again after bias
    acc = acc.to(y_ptr.dtype.element_ty).to(tl.float32)
    bias = tl.load(b_ptr + f * E + offs_e).to(tl.float32)
    acc = acc + bias[None, :]
    tl.store(
        y_ptr + f * (R * E) + offs_r[:, None] * E + offs_e[None, :],
        acc.to(y_ptr.dtype.element_ty),
        mask=r_mask[:, None],
    )


def _proj_ee(x_stacked: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    f, b, lq, e = x_stacked.shape
    y = torch.empty_like(x_stacked)
    rows = b * lq
    grid = (triton.cdiv(rows, 64), f)
    _proj_ee_kernel[grid](x_stacked, w, bias, y, rows, E=e, BLOCK_R=64)
    return y


@triton.jit
def _bf16_rtne(x):
    """fp32 -> bf16 round-to-nearest-even quantize in integer space (a plain
    .to(bf16).to(f32) cast pair is folded away by the compiler). Finite inputs
    only — satisfied here (bounded dense-projection values)."""
    u = x.to(tl.uint32, bitcast=True)
    bias = 0x00007FFF + ((u >> 16) & 1)
    q = (u + bias) & 0xFFFF0000
    return q.to(tl.float32, bitcast=True)


@triton.jit
def _ff_embed_kernel(arena_ptr, idx_ptr, meta_aoff, meta_ioff, meta_w, meta_col, meta_user,
                     expand_ptr, out_ptr, M,
                     W_TOT: tl.constexpr, BM: tl.constexpr, WMAX: tl.constexpr):
    """One width-class group of single-index gathers: out[r, col:col+w] =
    arena[aoff + idx[ioff + (expand[r] if user else r)]*w : ...+w]."""
    f = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    m_mask = offs_m < M
    aoff = tl.load(meta_aoff + f)
    ioff = tl.load(meta_ioff + f)
    w = tl.load(meta_w + f)
    col = tl.load(meta_col + f)
    user = tl.load(meta_user + f)
    rows = tl.where(user > 0, tl.load(expand_ptr + offs_m, mask=m_mask, other=0),
                    offs_m.to(tl.int64))
    idx = tl.load(idx_ptr + ioff + rows, mask=m_mask, other=0)
    ow = tl.arange(0, WMAX)
    w_mask = ow < w
    val = tl.load(arena_ptr + aoff + idx[:, None] * w + ow[None, :],
                  mask=m_mask[:, None] & w_mask[None, :], other=0.0)
    tl.store(out_ptr + offs_m[:, None] * W_TOT + col + ow[None, :], val,
             mask=m_mask[:, None] & w_mask[None, :])


@triton.jit
def _ff_bag_kernel(arena_ptr, vals_ptr, meta_aoff, meta_voff, meta_len, meta_w, meta_col,
                   expand_ptr, out_ptr, M,
                   W_TOT: tl.constexpr, BM: tl.constexpr, LMAX: tl.constexpr, WMAX: tl.constexpr):
    """One length-class group of mean-mode EmbeddingBags (user-level, fixed
    length, padding_idx==0: padded slots leave both sum and count; an
    all-padding bag yields zeros — aten semantics)."""
    f = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    m_mask = offs_m < M
    aoff = tl.load(meta_aoff + f)
    voff = tl.load(meta_voff + f)
    L = tl.load(meta_len + f)
    w = tl.load(meta_w + f)
    col = tl.load(meta_col + f)
    u = tl.load(expand_ptr + offs_m, mask=m_mask, other=0)
    ow = tl.arange(0, WMAX)
    w_mask = ow < w
    acc = tl.zeros((BM, WMAX), dtype=tl.float32)
    cnt = tl.zeros((BM,), dtype=tl.float32)
    for l in range(0, LMAX):
        in_len = l < L
        ii = tl.load(vals_ptr + voff + u * L + l, mask=m_mask & in_len, other=0)
        keep = (ii != 0) & in_len & m_mask
        row = tl.load(arena_ptr + aoff + ii[:, None] * w + ow[None, :],
                      mask=keep[:, None] & w_mask[None, :], other=0.0).to(tl.float32)
        acc += row
        cnt += keep.to(tl.float32)
    denom = tl.where(cnt > 0, cnt, 1.0)
    mean = acc / denom[:, None]
    tl.store(out_ptr + offs_m[:, None] * W_TOT + col + ow[None, :],
             mean.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & w_mask[None, :])


@triton.jit
def _ff_dense_kernel(x_ptr, w_ptr, b_ptr, out_ptr, M,
                     COL: tl.constexpr, NF: tl.constexpr, NO: tl.constexpr,
                     W_TOT: tl.constexpr, BM: tl.constexpr, WPAD: tl.constexpr, NW: tl.constexpr,
                     IO_FP32: tl.constexpr):
    """out[:, COL + f*NO + j] = rnd(rnd(x[:, f] * W[f, j]) + b[f, j]) — the two
    bf16 rounding steps torch's broadcast mul/add kernels perform; the
    intermediate round is the unfoldable arithmetic quantize."""
    pid_m = tl.program_id(0)
    pid_w = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    m_mask = offs_m < M
    ow = pid_w * WPAD + tl.arange(0, WPAD)
    w_mask = ow < NW
    fidx = ow // NO
    j = ow % NO
    x = tl.load(x_ptr + offs_m[:, None] * NF + fidx[None, :],
                mask=m_mask[:, None] & w_mask[None, :], other=0.0).to(tl.float32)
    wv = tl.load(w_ptr + fidx * NO + j, mask=w_mask, other=0.0).to(tl.float32)
    bv = tl.load(b_ptr + fidx * NO + j, mask=w_mask, other=0.0).to(tl.float32)
    if IO_FP32:
        y = x * wv[None, :] + bv[None, :]
    else:
        y = _bf16_rtne(x * wv[None, :])  # torch's mul-kernel round, unfoldable
        y = y + bv[None, :]
    tl.store(out_ptr + offs_m[:, None] * W_TOT + COL + ow[None, :],
             y.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & w_mask[None, :])


@triton.jit
def _ff_attn_copy_kernel(src_ptr, out_ptr, M,
                         W_TOT: tl.constexpr, AW: tl.constexpr, COL: tl.constexpr,
                         BM: tl.constexpr, WPAD: tl.constexpr):
    """Copy the concatenated attention block (rows, AW) into out[:, COL:COL+AW]."""
    pid_m = tl.program_id(0)
    pid_w = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    m_mask = offs_m < M
    ow = pid_w * WPAD + tl.arange(0, WPAD)
    w_mask = ow < AW
    val = tl.load(src_ptr + offs_m[:, None] * AW + ow[None, :],
                  mask=m_mask[:, None] & w_mask[None, :], other=0.0)
    tl.store(out_ptr + offs_m[:, None] * W_TOT + COL + ow[None, :], val,
             mask=m_mask[:, None] & w_mask[None, :])


@triton.jit
def _lin_kernel(x_ptr, w_ptr, b_ptr, y_ptr, M, K, N,
                ACT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """y = act(x @ w + b), fp32 accumulate (ieee), single round to the I/O
    dtype after the bias (addmm semantics), relu applied on the rounded value
    (mirrors eager relu(linear(...)))."""
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
        y = tl.maximum(y, 0.0)
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], y,
             mask=m_mask[:, None] & n_mask[None, :])


def _lin(x, w, b, act=0):
    m, k = x.shape
    n = w.shape[1]
    y = torch.empty(m, n, dtype=x.dtype, device=x.device)
    _lin_kernel[(triton.cdiv(m, 64), triton.cdiv(n, 64))](
        x.contiguous(), w, b, y, m, k, n, ACT=act, BM=64, BN=64, BK=32,
        num_warps=4, num_stages=3,
    )
    return y


# ---------------------------------------------------------------------------
# Composition wrapper: fused front + shipped attention pipeline + EAGER glue
# (K/V + query assembly, EPNet chain, gate) — the eager parts are the search
# headroom. Interface constants come from problem.py (single source of truth
# for the 198-tensor input list).
# ---------------------------------------------------------------------------
from problem import (
    ATTN_W, B, COL_OFF, DOM_PAD, DOM_RAW, E, EMB_W, EPNET_EMBED_FEATURES,
    GAMMA, H, KV_DIMS, KV_LENS, LAYOUT, LQ, MAX_KV, NF, N_SUM_SCALARS,
    PURCHASE_IDX, ROWS, SCAL_W, W_PAD, W_RAW,
)

_W_CLASSES = (8, 32, 64)
_L_CLASSES = (16, 64)
_meta_cache = {}


def _front_meta(device):
    hit = _meta_cache.get(device)
    if hit is not None:
        return hit
    embeds, bags = [], []
    aoffs_e, aoffs_b = [], []
    ioff = 0
    acc = 0
    for (name, kind, t_rows, _x, width, level, bag_len), col in zip(LAYOUT, COL_OFF):
        if kind == "embed":
            embeds.append((ioff, width, col, 1 if level == "user" else 0))
            aoffs_e.append(acc)
            acc += t_rows * width
            ioff += B if level == "user" else ROWS
        elif kind == "bag":
            bags.append((bag_len, width, col))
            aoffs_b.append(None)  # filled after the embed region size is known
    boff = acc
    ab = []
    for (name, kind, t_rows, _x, width, level, bag_len) in LAYOUT:
        if kind == "bag":
            ab.append(boff)
            boff += t_rows * width
    aoffs_b = ab
    def wcls(w):
        return next(i for i, wm in enumerate(_W_CLASSES) if w <= wm)
    def lcls(ln):
        return next(i for i, lm in enumerate(_L_CLASSES) if ln <= lm)
    e_order = sorted(range(len(embeds)), key=lambda i: wcls(embeds[i][1]))
    b_order = sorted(range(len(bags)), key=lambda i: lcls(bags[i][0]))
    e_bounds = [0] * (len(_W_CLASSES) + 1)
    for i in e_order:
        e_bounds[wcls(embeds[i][1]) + 1] += 1
    b_bounds = [0] * (len(_L_CLASSES) + 1)
    for i in b_order:
        b_bounds[lcls(bags[i][0]) + 1] += 1
    for a in (e_bounds, b_bounds):
        for i in range(1, len(a)):
            a[i] += a[i - 1]
    # voff accumulates in B_ORDER — bag_vals_flat is concatenated in that order.
    voffs = []
    v = 0
    for i in b_order:
        voffs.append(v)
        v += B * bags[i][0]
    i64 = lambda x: torch.tensor(x, dtype=torch.int64, device=device)  # noqa: E731
    meta = (
        i64([aoffs_e[i] for i in e_order]),
        i64([embeds[i][0] for i in e_order]), i64([embeds[i][1] for i in e_order]),
        i64([embeds[i][2] for i in e_order]), i64([embeds[i][3] for i in e_order]),
        e_bounds, e_order,
        i64([aoffs_b[i] for i in b_order]), i64(voffs),
        i64([bags[i][0] for i in b_order]), i64([bags[i][1] for i in b_order]),
        i64([bags[i][2] for i in b_order]),
        b_bounds, b_order,
    )
    _meta_cache[device] = meta
    return meta


def kernel_function(*tensors: torch.Tensor) -> torch.Tensor:
    expand_idx = tensors[0]
    device, f = expand_idx.device, NF
    a = tensors[1 : 1 + 6 * f]
    seq_ids = [a[6 * i + 0] for i in range(f)]
    scal = [a[6 * i + 1] for i in range(f)]
    tbls = [a[6 * i + 2] for i in range(f)]
    qids = [a[6 * i + 3] for i in range(f)]
    qp_w = [a[6 * i + 4] for i in range(f)]
    qp_b = [a[6 * i + 5] for i in range(f)]
    (q_w, q_b, k_w, v_w, k_b, v_b, out_w, out_b, bias_k, bias_v) = tensors[1 + 6 * f : 11 + 6 * f]
    (m_sizes_t, k_sizes_t, m_offs_t, x_offs_t, s_sizes_t) = tensors[11 + 6 * f : 16 + 6 * f]
    front = tensors[16 + 6 * f : 16 + 6 * f + 99]
    frozen_tbl = tensors[16 + 6 * f + 99]
    gw1, gb1, gw2, gb2 = tensors[16 + 6 * f + 100 : 16 + 6 * f + 104]
    sd = tbls[0].dtype

    # ---- EAGER K/V + query assembly (fusion headroom) ----
    kv_list, pad_list, q_list = [], [], []
    for i in range(f):
        emb = tbls[i][seq_ids[i]]  # exact row gather
        kv = torch.cat([emb, scal[i]], dim=-1) if SCAL_W[i] else emb
        kv_list.append(kv)
        pad_list.append(seq_ids[i] == 0)
        q = _lin(tbls[i][qids[i]], qp_w[i], qp_b[i])  # query projection
        q_list.append(q.view(B, LQ, E))

    # ---- shipped attention pipeline (dirs #1-#4 winner kernels) ----
    d = E // H
    scale = 1.0 / (d ** 0.5)
    neg = torch.finfo(torch.float32).min
    q_stacked = _proj_ee(torch.stack(q_list).contiguous(), q_w, q_b)
    kv_w = torch.cat([k_w, v_w], dim=2)
    kv_b = torch.cat([k_b, v_b], dim=1)
    n2 = 2 * E
    m_list = [B * s for s in KV_LENS]
    x_flat = torch.cat([x.reshape(-1) for x in kv_list])
    y = torch.empty((sum(m_list), n2), device=device, dtype=sd)
    grid = (triton.cdiv(max(m_list), 16), triton.cdiv(n2, 32), f)
    _ragged_gemm_kernel[grid](
        x_flat, kv_w, kv_b, y, m_sizes_t, k_sizes_t, m_offs_t, x_offs_t,
        N=n2, MAX_KV=k_w.shape[1], BLOCK_M=16, BLOCK_K=32, BLOCK_N=32,
        num_warps=4, num_stages=4,
    )
    pad_flat = torch.cat([p.reshape(-1) for p in pad_list]).to(torch.float32) * neg
    attn = torch.empty((f, B, LQ, E), device=device, dtype=sd)
    grid2 = (f, triton.cdiv(LQ, 32), B * H)
    _attn_core_ragged_kernel[grid2](
        q_stacked, y, bias_k, bias_v, pad_flat, attn, m_offs_t, s_sizes_t,
        B, LQ, scale, H=H, D=d, E=E, BLOCK_M=32, BLOCK_N=64,
        num_warps=4, num_stages=3,
    )
    attn = _proj_ee(attn, out_w, out_b)
    attn_block = attn.permute(1, 2, 0, 3).reshape(ROWS, ATTN_W).contiguous()

    # ---- fused feature front (dir #7 kernels) into (ROWS, W_PAD) ----
    (e_aoff, e_ioff, e_w, e_col, e_user, e_bounds, e_order,
     b_aoff, b_voff, b_len, b_w, b_col, b_bounds, b_order) = _front_meta(device)
    e_idx_t, b_idx_t, dense = [], [], None
    pos = 0
    for name, kind, *_rest in LAYOUT:
        if kind == "embed":
            e_idx_t.append((front[pos].reshape(-1), front[pos + 1]))
            pos += 2
        elif kind == "bag":
            b_idx_t.append((front[pos], front[pos + 1]))
            pos += 2
        else:
            dense = (front[pos], front[pos + 1], front[pos + 2])
            pos += 3
    arena = torch.cat([t[1].reshape(-1) for t in e_idx_t] + [t[1].reshape(-1) for t in b_idx_t])
    embed_idx_flat = torch.cat([e_idx_t[i][0].to(torch.int64) for i in range(len(e_idx_t))])
    bag_vals_flat = torch.cat([b_idx_t[i][0].reshape(-1).to(torch.int64) for i in b_order])

    out = torch.empty(ROWS, W_PAD, dtype=sd, device=device)
    for gi, wmax in enumerate(_W_CLASSES):
        g0, g1 = e_bounds[gi], e_bounds[gi + 1]
        if g1 == g0:
            continue
        bm = 128 if wmax <= 8 else 64
        _ff_embed_kernel[(g1 - g0, triton.cdiv(ROWS, bm))](
            arena, embed_idx_flat,
            e_aoff[g0:g1], e_ioff[g0:g1], e_w[g0:g1], e_col[g0:g1], e_user[g0:g1],
            expand_idx, out, ROWS,
            W_TOT=W_PAD, BM=bm, WMAX=wmax, num_warps=4 if wmax <= 8 else 2,
        )
    for gi, lmax in enumerate(_L_CLASSES):
        g0, g1 = b_bounds[gi], b_bounds[gi + 1]
        if g1 == g0:
            continue
        _ff_bag_kernel[(g1 - g0, triton.cdiv(ROWS, 128))](
            arena, bag_vals_flat,
            b_aoff[g0:g1], b_voff[g0:g1], b_len[g0:g1], b_w[g0:g1], b_col[g0:g1],
            expand_idx, out, ROWS,
            W_TOT=W_PAD, BM=128, LMAX=lmax, WMAX=8, num_warps=4,
        )
    dx, dw, db = dense
    dense_col = COL_OFF[[k for k, e in enumerate(LAYOUT) if e[1] == "dense"][0]]
    nw_d = dw.shape[0] * dw.shape[1]
    BM_D = 128
    _ff_dense_kernel[(triton.cdiv(ROWS, BM_D), triton.cdiv(nw_d, 32))](
        dx.to(sd), dw, db, out, ROWS,
        COL=dense_col, NF=dw.shape[0], NO=dw.shape[1], W_TOT=W_PAD, BM=BM_D, WPAD=32, NW=nw_d,
        IO_FP32=sd == torch.float32, num_warps=4,
    )
    _ff_attn_copy_kernel[(triton.cdiv(ROWS, 128), triton.cdiv(ATTN_W, 128))](
        attn_block, out, ROWS, W_TOT=W_PAD, AW=ATTN_W, COL=0, BM=128, WPAD=128, num_warps=4,
    )
    out[:, W_RAW:].zero_()  # kalign pad columns: consumers have zero K-rows

    # ---- EAGER EPNet chain + gate (fusion headroom) ----
    mask_cols = [torch.where(ids.ne(0).sum(-1, keepdim=True) > 0, 1.0, 0.0) for ids in seq_ids]
    inst_len = seq_ids[0].ne(0).sum(-1, keepdim=True).clamp(min=1)
    iap_cols = []
    for pi in PURCHASE_IDX:
        slen = seq_ids[pi].ne(0).sum(-1, keepdim=True)
        iap_cols.append(slen.to(torch.float32))
        iap_cols.append(slen.to(torch.float32) / inst_len.to(torch.float32))
        sc = kv_list[pi][..., EMB_W[pi]:]
        for j in range(N_SUM_SCALARS):
            iap_cols.append(sc[..., j].max(dim=1).values.unsqueeze(-1).to(torch.float32))
    dom_b = torch.cat([torch.cat(mask_cols, dim=-1).to(torch.float32),
                       torch.cat(iap_cols, dim=-1)], dim=-1)
    dom_rows = [dom_b.index_select(0, expand_idx)]
    ei = 0
    front_by_name = {}
    for k, (name, kind, *_r) in enumerate(LAYOUT):
        if kind == "embed":
            front_by_name[name] = e_idx_t[ei]
            ei += 1
    for name in EPNET_EMBED_FEATURES:
        idx, tab = front_by_name[name]
        dom_rows.append(tab[idx].index_select(0, expand_idx).float())
    t_idx, _t = front_by_name["target_store_id"]
    dom_rows.append(frozen_tbl[t_idx].float())
    dom = torch.cat(dom_rows, dim=-1)
    dom = torch.cat([dom, dom.new_zeros(dom.shape[0], DOM_PAD - DOM_RAW)], dim=-1).to(sd)
    h = _lin(torch.cat([dom, out], dim=-1), gw1, gb1, act=1)  # relu(linear)
    g = _lin(h, gw2, gb2)
    l2 = ((1.0 / (1.0 + torch.exp(-g.float()))) * GAMMA).to(sd)
    return l2 * out
