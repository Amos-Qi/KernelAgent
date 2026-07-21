# Starting kernel: the CURRENT BEST integrated pipeline — the verified
# winners of kernel dirs #1 (fused attention core, 5.5x) and #2 (ragged K/V
# GEMM, 2.6x) plus the torch glue between them, exactly as integrated into
# serving. Stage-by-stage it is already near-optimal; the remaining cost is
# the SEAMS, all inside this timed region:
#   a. Q projection: torch.stack + einsum materializes (F, B, Lq, E) before
#      the core reads it (could fold into the core's Q read or one kernel);
#   b. the ragged GEMM writes a compact (sum_M, 2E) buffer that the epilogue
#      then re-copies into the core's concatenated (B, S_total, E) layout,
#      adds bias_kv/zero_attn slots, and builds the additive mask (~26 small
#      torch ops -> the GEMM could write the final layout directly, with the
#      bias slots and mask baked in);
#   c. the output projection einsum (could fold into the core's store).
# Numerics contract (do not weaken): matmuls take native-dtype operands with
# fp32 accumulation and input_precision="ieee" (no TF32); softmax and P.V in
# fp32; outputs cast to the I/O dtype. Every attention row keeps >= 1
# attendable key (the appended slots carry additive mask 0).

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
def _attn_core_fused_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    mask_ptr,
    off_ptr,
    len_ptr,
    B,
    Lq,
    S_total,
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

    S = tl.load(len_ptr + pid_f)
    off = tl.load(off_ptr + pid_f)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    m_valid = offs_m < Lq
    n_valid = offs_n < S

    # q/o: (F, B, Lq, E) contiguous; head h at E-columns [h*D, (h+1)*D)
    q_base = q_ptr + pid_f * (B * Lq * E) + b * (Lq * E)
    q = tl.load(
        q_base + offs_m[:, None] * E + (h * D + offs_d[None, :]),
        mask=m_valid[:, None],
        other=0.0,
    )

    k_base = k_ptr + b * (S_total * E)
    kt = tl.load(
        k_base + (off + offs_n[None, :]) * E + (h * D + offs_d[:, None]),
        mask=n_valid[None, :],
        other=0.0,
    )

    scores = (tl.dot(q, kt, out_dtype=tl.float32, input_precision="ieee") * scale).to(tl.float32)
    mvals = tl.load(mask_ptr + b * S_total + (off + offs_n), mask=n_valid, other=0.0)
    scores = scores + mvals[None, :]
    scores = tl.where(n_valid[None, :], scores, float("-inf"))

    m_i = tl.max(scores, axis=1)
    p = tl.exp(scores - m_i[:, None])
    p = p / tl.sum(p, axis=1)[:, None]  # fp32 softmax

    v_base = v_ptr + b * (S_total * E)
    v = tl.load(
        v_base + (off + offs_n[:, None]) * E + (h * D + offs_d[None, :]),
        mask=n_valid[:, None],
        other=0.0,
    ).to(tl.float32)
    out = tl.dot(p, v, input_precision="ieee")  # fp32 P.V

    o_base = o_ptr + pid_f * (B * Lq * E) + b * (Lq * E)
    tl.store(
        o_base + offs_m[:, None] * E + (h * D + offs_d[None, :]),
        out.to(o_ptr.dtype.element_ty),
        mask=m_valid[:, None],
    )


def kernel_function(*tensors: torch.Tensor) -> torch.Tensor:
    f = 13
    q_list = list(tensors[0:f])
    kv_list = list(tensors[f : 2 * f])
    pad_list = list(tensors[2 * f : 3 * f])
    (q_w, q_b, k_w, v_w, k_b, v_b, out_w, out_b, bias_k, bias_v) = tensors[3 * f :]

    b, lq, e = q_list[0].shape
    h = 2
    d = e // h
    scale = 1.0 / (d**0.5)
    device, dtype = q_list[0].device, q_list[0].dtype
    neg = torch.finfo(torch.float32).min

    # a. Q projection (stack + einsum, as served)
    q_stacked = torch.einsum("fble,fed->fbld", torch.stack(q_list), q_w) + q_b[:, None, None, :]
    q_stacked = q_stacked.contiguous()

    # b1. ragged K/V projection into a compact (sum_M, 2E) buffer
    kv_w = torch.cat([k_w, v_w], dim=2)  # (F, max_kv, 2E)
    kv_b = torch.cat([k_b, v_b], dim=1)  # (F, 2E)
    max_kv = k_w.shape[1]
    n2 = 2 * e
    s_list = [x.shape[1] for x in kv_list]
    d_list = [x.shape[2] for x in kv_list]
    m_list = [b * s for s in s_list]
    m_offs = [0]
    x_offs = [0]
    for i in range(f - 1):
        m_offs.append(m_offs[-1] + m_list[i])
        x_offs.append(x_offs[-1] + m_list[i] * d_list[i])
    m_sizes_t = torch.tensor(m_list, device=device, dtype=torch.int32)
    k_sizes_t = torch.tensor(d_list, device=device, dtype=torch.int32)
    m_offs_t = torch.tensor(m_offs, device=device, dtype=torch.int32)
    x_offs_t = torch.tensor(x_offs, device=device, dtype=torch.int32)
    x_flat = torch.cat([x.reshape(-1) for x in kv_list])
    sum_m = m_offs[-1] + m_list[-1]
    y = torch.empty((sum_m, n2), device=device, dtype=dtype)
    grid = (triton.cdiv(max(m_list), 16), triton.cdiv(n2, 32), f)
    _ragged_gemm_kernel[grid](
        x_flat, kv_w, kv_b, y,
        m_sizes_t, k_sizes_t, m_offs_t, x_offs_t,
        N=n2, MAX_KV=max_kv,
        BLOCK_M=16, BLOCK_K=32, BLOCK_N=32,
        num_warps=4, num_stages=4,
    )

    # b2. epilogue: scatter into the core's concatenated layout + bias slots + mask
    s_ext = [s + 2 for s in s_list]
    s_offsets: List[int] = []
    s_total = 0
    for s in s_ext:
        s_offsets.append(s_total)
        s_total += s
    k_cat = y.new_zeros(b, s_total, e)
    v_cat = y.new_zeros(b, s_total, e)
    add_cat = torch.zeros(b, s_total, device=device, dtype=torch.float32)
    for i in range(f):
        off, s_i = s_offsets[i], s_list[i]
        yi = y[m_offs[i] : m_offs[i] + m_list[i]].reshape(b, s_i, n2)
        k_cat[:, off : off + s_i] = yi[..., :e]
        v_cat[:, off : off + s_i] = yi[..., e:]
        k_cat[:, off + s_i] = bias_k[i].to(dtype)
        v_cat[:, off + s_i] = bias_v[i].to(dtype)
        add_cat[:, off : off + s_i] = torch.where(pad_list[i], neg, 0.0)

    # c1. fused attention core (single launch over all features)
    off_t = torch.tensor(s_offsets, device=device, dtype=torch.int32)
    len_t = torch.tensor(s_ext, device=device, dtype=torch.int32)
    out = torch.empty((f, b, lq, e), device=device, dtype=dtype)
    grid2 = (f, triton.cdiv(lq, 32), b * h)
    _attn_core_fused_kernel[grid2](
        q_stacked, k_cat, v_cat, out, add_cat, off_t, len_t,
        b, lq, s_total, scale,
        H=h, D=d, E=e, BLOCK_M=32, BLOCK_N=64,
    )

    # c2. output projection (einsum, as served)
    return torch.einsum("fble,fed->fbld", out, out_w) + out_b[:, None, None, :]
