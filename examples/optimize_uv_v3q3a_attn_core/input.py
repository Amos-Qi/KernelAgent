# Starting kernel: faithful port of the v3-q3a serving implementation
# (vector-ai-unity-learner deploy/triton_attention/{kernel,op,grouped}.py,
# minus the torch.library.triton_op wrapper, which is re-applied at
# integration time).
#
# Structure being timed, exactly as served today:
#   1. per-feature additive-mask build from the bool key-padding mask
#   2. features bucketed by KV length S -> one launch per distinct S
#      ({52, 32, 12, 7} -> 4 launches), with torch.stack gathers into
#      contiguous batched tensors per bucket
#   3. single-block-over-KV fused attention kernel per bucket
#      (S <= 52 fits one BLOCK_N; no online softmax needed)
#   4. per-bucket outputs scattered back into one (F, B, H, Lq, D) tensor
#
# The stacks/fills in 1-2-4 and the 4 separate launches are real serving
# overhead and are inside the timed region on purpose — fusing buckets into
# fewer launches and eliminating the gather/scatter copies are valid
# optimization directions, as is tuning BLOCK_M/num_warps for the
# launch-bound grid. Numerics contract: QK products of bf16 values with fp32
# accumulate, softmax and P.V in fp32, output cast back to the I/O dtype.
# Every mask row has >= 1 attendable key (2 appended slots), so the softmax
# is always finite.

from collections import defaultdict

import torch
import triton
import triton.language as tl

BLOCK_M = 32  # query rows per program (serving _config.BLOCK_M)


@triton.jit
def _attn_core_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    mask_ptr,
    Lq,
    S,
    sq_bh,
    sq_l,
    sq_d,
    sk_bh,
    sk_s,
    sk_d,
    sv_bh,
    sv_s,
    sv_d,
    so_bh,
    so_l,
    so_d,
    sm_b,
    sm_s,
    scale,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    pid_m = tl.program_id(0)  # query-row block
    pid_bh = tl.program_id(1)  # flattened (batch, head)
    b = pid_bh // H

    offs_m = pid_m * BLOCK_M_C + tl.arange(0, BLOCK_M_C)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    m_valid = offs_m < Lq
    n_valid = offs_n < S

    q_ptrs = q_ptr + pid_bh * sq_bh + offs_m[:, None] * sq_l + offs_d[None, :] * sq_d
    q = tl.load(q_ptrs, mask=m_valid[:, None], other=0.0)

    kt_ptrs = k_ptr + pid_bh * sk_bh + offs_d[:, None] * sk_d + offs_n[None, :] * sk_s
    kt = tl.load(kt_ptrs, mask=n_valid[None, :], other=0.0)

    # QK in native dtype (bf16 -> tensor cores); out_dtype=fp32 gives fp32
    # scores for a stable softmax. The trailing .to(fp32) pins the dtype after
    # the scale multiply (Triton types the Python-float scale as fp64).
    scores = (tl.dot(q, kt, out_dtype=tl.float32, input_precision="ieee") * scale).to(
        tl.float32
    )
    if HAS_MASK:
        mvals = tl.load(mask_ptr + b * sm_b + offs_n * sm_s, mask=n_valid, other=0.0)
        scores = scores + mvals[None, :]
    scores = tl.where(n_valid[None, :], scores, float("-inf"))

    m_i = tl.max(scores, axis=1)
    p = tl.exp(scores - m_i[:, None])
    p = p / tl.sum(p, axis=1)[:, None]  # softmax stays fp32

    v_ptrs = v_ptr + pid_bh * sv_bh + offs_n[:, None] * sv_s + offs_d[None, :] * sv_d
    # V upcast only to match the fp32 probs (tl.dot needs one shared dtype).
    v = tl.load(v_ptrs, mask=n_valid[:, None], other=0.0).to(tl.float32)
    out = tl.dot(p, v, input_precision="ieee")

    o_ptrs = o_ptr + pid_bh * so_bh + offs_m[:, None] * so_l + offs_d[None, :] * so_d
    tl.store(o_ptrs, out.to(o_ptr.dtype.element_ty), mask=m_valid[:, None])


def _attn_core(q, k, v, mask, scale, heads):
    """One fused-core launch. q/k/v: (B*H, L, D) contiguous; mask: (B, S) fp32
    additive (0 = attend, finfo.min = ignore)."""
    bh, lq, d = q.shape
    s = k.shape[1]
    out = torch.empty_like(q)
    block_n = max(16, triton.next_power_of_2(s))
    grid = (triton.cdiv(lq, BLOCK_M), bh)
    _attn_core_kernel[grid](
        q,
        k,
        v,
        out,
        mask,
        lq,
        s,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        *mask.stride(),
        scale,
        H=heads,
        D=d,
        BLOCK_M_C=BLOCK_M,
        BLOCK_N=block_n,
        HAS_MASK=True,
    )
    return out


def kernel_function(*tensors: torch.Tensor) -> torch.Tensor:
    """Grouped attention core over F features.

    Input: F queries (B,H,Lq,D), F keys (B,H,S_i,D), F values (B,H,S_i,D),
    F bool key-padding masks (B,S_i) with True = ignore, concatenated in that
    order. Output: (F, B, H, Lq, D) in the query dtype.
    """
    f = len(tensors) // 4
    q_list = tensors[0:f]
    k_list = tensors[f : 2 * f]
    v_list = tensors[2 * f : 3 * f]
    pad_list = tensors[3 * f : 4 * f]

    b, h, lq, d = q_list[0].shape
    scale = 1.0 / (d**0.5)
    neg = torch.finfo(torch.float32).min

    # 1. bool key-padding -> fp32 additive mask, per feature (as served)
    add_list = []
    for pad in pad_list:
        add = torch.zeros(pad.shape, device=pad.device, dtype=torch.float32)
        add_list.append(add.masked_fill(pad, neg))

    # 2. bucket features by KV length
    buckets: "dict[int, list[int]]" = defaultdict(list)
    for i, k in enumerate(k_list):
        buckets[k.shape[2]].append(i)

    out = torch.empty((f, b, h, lq, d), device=q_list[0].device, dtype=q_list[0].dtype)
    for s, idxs in buckets.items():
        n = len(idxs)
        qb = torch.stack([q_list[i] for i in idxs]).reshape(n * b * h, lq, d)
        kb = torch.stack([k_list[i] for i in idxs]).reshape(n * b * h, s, d)
        vb = torch.stack([v_list[i] for i in idxs]).reshape(n * b * h, s, d)
        mb = torch.stack([add_list[i] for i in idxs]).reshape(n * b, s)

        # 3. one fused launch per bucket; 4. scatter back per feature
        ob = _attn_core(qb, kb, vb, mb, scale, h)
        out[idxs] = ob.view(n, b, h, lq, d)

    return out
