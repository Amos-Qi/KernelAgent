# Starting kernel: faithful port of the v3-q3a serving implementation
# (vector-ai-unity-learner deploy/triton_attention/{grouped_gemm,proj,
# grouped_gemm_op}.py, minus the torch.library.triton_op wrapper).
#
# Structure being timed, exactly as served today:
#   1. cat k_weight|v_weight (F, max_kv, 2E) and k_bias|v_bias (F, 2E) per call
#   2. pack_grouped: zero-fill x (F, max_M, max_K) and w (F, max_K, 2E), then
#      copy every ragged x_i / sliced w_i into the padded tensors
#   3. one grouped-GEMM launch on the PADDED shapes: grid
#      (cdiv(max_M, 32), F) — every feature pays max_M rows even when its
#      M_i = B*S_i is 10x smaller (~58% of row tiles are pure padding)
#   4. per-feature output slices y[i, :M_i] concatenated back together
#
# The per-call weight cats, the zero-fill + gather copies, and the padded-tile
# compute are real serving overhead and inside the timed region on purpose —
# eliminating the packing (ragged access via per-feature pointers), skipping
# padded tiles (per-group M array), and right-sizing BLOCK_K per group are all
# valid optimization directions. Numerics contract: products of bf16 values
# with fp32 accumulate (ieee), bias added in fp32, output cast to I/O dtype.

from typing import List, Optional, Sequence

import torch
import triton
import triton.language as tl
from torch import Tensor

BLOCK_M = 32  # output rows per program (serving _config.BLOCK_M)


@triton.jit
def _grouped_gemm_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    M,
    K,
    N,
    sx_g,
    sx_m,
    sx_k,
    sw_g,
    sw_k,
    sw_n,
    sy_g,
    sy_m,
    sy_n,
    sb_g,
    sb_n,
    HAS_BIAS: tl.constexpr,
    BLOCK_M_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)  # row tile
    g = tl.program_id(1)  # problem index

    offs_m = pid_m * BLOCK_M_C + tl.arange(0, BLOCK_M_C)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N
    k_mask = offs_k < K

    x_ptrs = x_ptr + g * sx_g + offs_m[:, None] * sx_m + offs_k[None, :] * sx_k
    x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

    w_ptrs = w_ptr + g * sw_g + offs_k[:, None] * sw_k + offs_n[None, :] * sw_n
    w = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

    # Native-dtype operands -> tensor cores on bf16; fp32 accumulate; ieee
    # keeps the fp32 path exact (no TF32).
    acc = tl.dot(x, w, out_dtype=tl.float32, input_precision="ieee")
    if HAS_BIAS:
        b = tl.load(b_ptr + g * sb_g + offs_n * sb_n, mask=n_mask, other=0.0)
        acc = acc + b[None, :]

    y_ptrs = y_ptr + g * sy_g + offs_m[:, None] * sy_m + offs_n[None, :] * sy_n
    tl.store(
        y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :]
    )


def _pack_grouped(
    x_list: Sequence[Tensor],
    w_list: Sequence[Tensor],
    b_list: Optional[Sequence[Tensor]],
):
    """Zero-pad the ragged (x_i (M_i,K_i), w_i (K_i,N)) list into the fixed 3-D
    layout the kernel reads (verbatim serving pack_grouped)."""
    g = len(x_list)
    n = w_list[0].shape[1]
    m_sizes = [int(x.shape[0]) for x in x_list]
    k_sizes = [int(x.shape[1]) for x in x_list]
    max_m, max_k = max(m_sizes), max(k_sizes)
    x = x_list[0].new_zeros(g, max_m, max_k)
    w = w_list[0].new_zeros(g, max_k, n)
    for i in range(g):
        x[i, : m_sizes[i], : k_sizes[i]] = x_list[i]
        w[i, : k_sizes[i], :] = w_list[i]
    b = torch.stack(list(b_list), dim=0) if b_list is not None else None
    return x, w, b, m_sizes


def kernel_function(*tensors: torch.Tensor) -> torch.Tensor:
    """Fused ragged K/V projection over F features.

    Input: F ragged tensors x_i (B, S_i, k_i), then k_weight (F, MAX_KV, E),
    v_weight (F, MAX_KV, E), k_bias (F, E), v_bias (F, E).
    Output: (sum(B*S_i), 2E) in the input dtype, feature blocks in order.
    """
    f = len(tensors) - 4
    x_raw = tensors[0:f]
    k_w, v_w, k_b, v_b = tensors[f : f + 4]
    e = k_w.shape[2]

    # 1. fuse K and V: one grouped GEMM with N = 2E (as served)
    kv_w = torch.cat([k_w, v_w], dim=2)  # (F, max_kv, 2E)
    kv_b = torch.cat([k_b, v_b], dim=1)  # (F, 2E)

    x_list: List[Tensor] = []
    w_list: List[Tensor] = []
    b_list: List[Tensor] = []
    m_bs = []
    for i in range(f):
        b_i, s_i, k_i = x_raw[i].shape
        x_list.append(x_raw[i].reshape(b_i * s_i, k_i))
        w_list.append(kv_w[i, :k_i, :])
        b_list.append(kv_b[i])
        m_bs.append(b_i * s_i)

    # 2. zero-pad into the fixed 3-D layout
    x, w, b, m_sizes = _pack_grouped(x_list, w_list, b_list)
    g, max_m, max_k = x.shape
    n = w.shape[2]

    # 3. single launch on padded shapes
    y = x.new_empty(g, max_m, n)
    block_n = max(16, triton.next_power_of_2(n))
    block_k = max(16, triton.next_power_of_2(max_k))
    grid = (triton.cdiv(max_m, BLOCK_M), g)
    _grouped_gemm_kernel[grid](
        x,
        w,
        b,
        y,
        max_m,
        max_k,
        n,
        *x.stride(),
        *w.stride(),
        *y.stride(),
        *b.stride(),
        HAS_BIAS=True,
        BLOCK_M_C=BLOCK_M,
        BLOCK_K=block_k,
        BLOCK_N=block_n,
    )

    # 4. slice off the padded rows and rebuild the flat output
    return torch.cat([y[i, : m_sizes[i], :] for i in range(g)], dim=0)
