# Starting kernel: the verified full-forward winner (kernel dir #3 — ragged
# GEMM feeding the attention core directly, bit-identical to serving, 4.1x
# over the integrated seed) made CAPTURE-SAFE for CUDA-graph serving:
#   - every offset/size array arrives as a PRECOMPUTED device input — the
#     per-call torch.tensor(list, device=...) host->device builds are gone
#     (they are illegal inside torch.cuda.graph capture);
#   - shapes are the fixed CG bucket (B=24, Lq=1024) — grids are static;
#   - remaining per-call ops are all device-side (stack/cat/elementwise) and
#     allocations (graph-pool-managed), both capture-legal. Reducing them
#     further — e.g. reading per-feature q/kv/pad tensors in-kernel instead
#     of cat/stacking them, or persisting the flat buffers — is exactly the
#     optimization headroom, since replay cost = sum of captured work.
# Numerics contract (do not weaken): native-dtype matmul operands with fp32
# accumulation (ieee, no TF32); softmax and P.V in fp32; stage outputs stored
# in the I/O dtype (projections keep the double-rounded bias add). Every
# attention row keeps >= 1 attendable key.

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


# Static per-config constants (CG bucket shapes are fixed at capture time).
_KV_LENS = [50, 50, 30, 10, 10, 10, 30, 5, 5, 10, 30, 10, 30]
_KV_DIMS = [40, 37, 58, 54, 54, 58, 58, 54, 54, 59, 71, 71, 71]


def kernel_function(*tensors: torch.Tensor) -> torch.Tensor:
    f = 13
    q_list = list(tensors[0:f])
    kv_list = list(tensors[f : 2 * f])
    pad_list = list(tensors[2 * f : 3 * f])
    (q_w, q_b, k_w, v_w, k_b, v_b, out_w, out_b, bias_k, bias_v) = tensors[3 * f : 3 * f + 10]
    # Precomputed, device-resident, replay-static index tensors (NO per-call
    # host->device builds — capture-safety requirement).
    (m_sizes_t, k_sizes_t, m_offs_t, x_offs_t, s_sizes_t) = tensors[3 * f + 10 :]

    b, lq, e = q_list[0].shape
    h = 2
    d = e // h
    scale = 1.0 / (d**0.5)
    device, dtype = q_list[0].device, q_list[0].dtype
    neg = torch.finfo(torch.float32).min

    # a. Q projection (device-side stack + per-feature E->E GEMM)
    q_stacked = _proj_ee(torch.stack(q_list).contiguous(), q_w, q_b)

    # b. ragged K/V projection into a compact (sum_M, 2E) buffer
    kv_w = torch.cat([k_w, v_w], dim=2)  # (F, max_kv, 2E)
    kv_b = torch.cat([k_b, v_b], dim=1)  # (F, 2E)
    max_kv = k_w.shape[1]
    n2 = 2 * e
    m_list = [b * s for s in _KV_LENS]  # python constants: shapes are static
    sum_m = sum(m_list)
    x_flat = torch.cat([x.reshape(-1) for x in kv_list])
    y = torch.empty((sum_m, n2), device=device, dtype=dtype)
    grid = (triton.cdiv(max(m_list), 16), triton.cdiv(n2, 32), f)
    _ragged_gemm_kernel[grid](
        x_flat, kv_w, kv_b, y,
        m_sizes_t, k_sizes_t, m_offs_t, x_offs_t,
        N=n2, MAX_KV=max_kv,
        BLOCK_M=16, BLOCK_K=32, BLOCK_N=32,
        num_warps=4, num_stages=4,
    )

    # c. attention core reading the ragged buffer directly (device-side flat
    # additive mask; bias_kv slot synthesized in-register, zero_attn masked-in)
    pad_flat = torch.cat([p.reshape(-1) for p in pad_list]).to(torch.float32) * neg
    out = torch.empty((f, b, lq, e), device=device, dtype=dtype)
    grid2 = (f, triton.cdiv(lq, 32), b * h)
    _attn_core_ragged_kernel[grid2](
        q_stacked, y, bias_k, bias_v, pad_flat, out,
        m_offs_t, s_sizes_t,
        b, lq, scale,
        H=h, D=d, E=e, BLOCK_M=32, BLOCK_N=64,
        num_warps=4, num_stages=3,
    )

    # d. output projection
    return _proj_ee(out, out_w, out_b)
