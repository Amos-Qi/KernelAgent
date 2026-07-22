import torch
import triton
import triton.language as tl


@triton.jit
def _attn_kernel(
    q_ptr_arr,
    k_ptr_arr,
    v_ptr_arr,
    pad_ptr_arr,
    s_arr,
    out_ptr,
    Lq,
    scale,
    H: tl.constexpr,
    D: tl.constexpr,
    B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_f = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_bh = tl.program_id(2)
    b = pid_bh // H

    S = tl.load(s_arr + pid_f)

    q_base = tl.load(q_ptr_arr + pid_f).to(tl.pointer_type(tl.bfloat16))
    k_base = tl.load(k_ptr_arr + pid_f).to(tl.pointer_type(tl.bfloat16))
    v_base = tl.load(v_ptr_arr + pid_f).to(tl.pointer_type(tl.bfloat16))
    pad_base = tl.load(pad_ptr_arr + pid_f).to(tl.pointer_type(tl.uint8))

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    m_valid = offs_m < Lq
    n_valid = offs_n < S

    q_ptrs = q_base + pid_bh * (Lq * D) + offs_m[:, None] * D + offs_d[None, :]
    q = tl.load(q_ptrs, mask=m_valid[:, None], other=0.0)

    k_ptrs = k_base + pid_bh * (S * D) + offs_n[:, None] * D + offs_d[None, :]
    k = tl.load(k_ptrs, mask=n_valid[:, None], other=0.0)
    kt = tl.trans(k)

    scores = tl.dot(q, kt, out_dtype=tl.float32) * scale

    pad_ptrs = pad_base + b * S + offs_n
    pad = tl.load(pad_ptrs, mask=n_valid, other=1)
    scores = tl.where((pad != 0)[None, :], float("-inf"), scores)

    m_i = tl.max(scores, axis=1)
    p = tl.exp(scores - m_i[:, None])
    p = p / tl.sum(p, axis=1)[:, None]

    v_ptrs = v_base + pid_bh * (S * D) + offs_n[:, None] * D + offs_d[None, :]
    v = tl.load(v_ptrs, mask=n_valid[:, None], other=0.0)

    out = tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)

    o_ptrs = (
        out_ptr
        + pid_f * (B * H * Lq * D)
        + pid_bh * (Lq * D)
        + offs_m[:, None] * D
        + offs_d[None, :]
    )
    tl.store(o_ptrs, out.to(out_ptr.dtype.element_ty), mask=m_valid[:, None])


def kernel_function(*tensors: torch.Tensor) -> torch.Tensor:
    f = len(tensors) // 4
    q_list = list(tensors[0:f])
    k_list = list(tensors[f : 2 * f])
    v_list = list(tensors[2 * f : 3 * f])
    pad_list = list(tensors[3 * f : 4 * f])

    b, h, lq, d = q_list[0].shape
    scale = 1.0 / (d ** 0.5)

    device = q_list[0].device
    q_ptrs = torch.tensor(
        [q.data_ptr() for q in q_list], device=device, dtype=torch.int64
    )
    k_ptrs = torch.tensor(
        [k.data_ptr() for k in k_list], device=device, dtype=torch.int64
    )
    v_ptrs = torch.tensor(
        [v.data_ptr() for v in v_list], device=device, dtype=torch.int64
    )
    pad_ptrs = torch.tensor(
        [p.data_ptr() for p in pad_list], device=device, dtype=torch.int64
    )
    s_arr = torch.tensor(
        [k.shape[2] for k in k_list], device=device, dtype=torch.int32
    )

    out = torch.empty(
        (f, b, h, lq, d), device=device, dtype=q_list[0].dtype
    )

    block_m = 64
    block_n = 64
    grid = (f, triton.cdiv(lq, block_m), b * h)

    _attn_kernel[grid](
        q_ptrs,
        k_ptrs,
        v_ptrs,
        pad_ptrs,
        s_arr,
        out,
        lq,
        scale,
        H=h,
        D=d,
        B=b,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return out