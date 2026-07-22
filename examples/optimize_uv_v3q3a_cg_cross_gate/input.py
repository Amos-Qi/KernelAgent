# Starting kernel (hand-written re-seed): a CORRECT, working Triton pipeline
# for the EPNet-gate + low-rank-cross region, with each pointwise epilogue
# fused into its producing GEMM's store — the byte-saving structure the
# search should refine (tiling, persistence, cross-stage fusion, keeping the
# x tile resident across stages).
#
# Shapes are the VERIFIED serving dims (problem.py): D = 1322, gate input
# 1453 = cat([dom (131), x]), hidden 512, rank 128, and the cross's
# x0 == xl == y0 aliasing (the epilogue reads ONE stream, not two).
#
# Stage-rounding semantics mirror the eager reference EXACTLY: addmm rounds
# once (bias added on the fp32 accumulator, then one round), every elementwise
# op rounds its output to the storage dtype, and dots accumulate fp32 (ieee,
# no TF32). A fp32 run makes every round a no-op.
#
# Capture-safety: static shapes, no per-call host->device transfers, only
# device-side launches and pool-managed allocations. Both test.py gates pass.

import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_bias_relu_kernel(
    x_ptr, w_ptr, b_ptr, h_ptr,
    M, K: tl.constexpr, N: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """h = relu(addmm(b, x, w)): bias on the fp32 accumulator, ONE round
    (as torch.addmm), relu on the rounded value. x:(M,K) w:(K,N) h:(M,N)."""
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
    hb = (acc + bias[None, :]).to(h_ptr.dtype.element_ty).to(tl.float32)
    hr = tl.maximum(hb, 0.0)  # relu is exact on the rounded value
    tl.store(h_ptr + offs_m[:, None] * N + offs_n[None, :],
             hr.to(h_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _gemm_gate_mul_kernel(
    h_ptr, w2_ptr, b2_ptr, x_ptr, y0_ptr,
    M, K: tl.constexpr, N: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """y0 = x * (2*sigmoid(addmm(b2, h, w2))) — the mm_mul_sigmoid epilogue
    fused into the GEMM store (x tile loaded once, no full-tensor round trip).
    Rounds at each eager op: addmm once, sigmoid, *gamma, *x."""
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
        a = tl.load(h_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w2_ptr + offs_k[:, None] * N + offs_n[None, :],
                    mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(a, w, acc, out_dtype=tl.float32, input_precision="ieee")

    dt = y0_ptr.dtype.element_ty
    bias = tl.load(b2_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    gb = (acc + bias[None, :]).to(dt).to(tl.float32)   # addmm's single round
    s = tl.sigmoid(gb).to(dt).to(tl.float32)           # sigmoid rounds
    l2 = (s * 2.0).to(dt).to(tl.float32)               # *gamma rounds
    xv = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :],
                 mask=m_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)
    tl.store(y0_ptr + offs_m[:, None] * N + offs_n[None, :],
             (l2 * xv).to(dt), mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _gemm_low_kernel(
    y0_ptr, wd_ptr, low_ptr,
    M, K: tl.constexpr, N: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """low = y0 @ w_down. (M,K)@(K,N) with N = rank 128; stored bf16 as served."""
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
        a = tl.load(y0_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(wd_ptr + offs_k[:, None] * N + offs_n[None, :],
                    mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(a, w, acc, out_dtype=tl.float32, input_precision="ieee")

    tl.store(low_ptr + offs_m[:, None] * N + offs_n[None, :],
             acc.to(low_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _gemm_cross_epilogue_kernel(
    low_ptr, wu_ptr, b_ptr, y0_ptr, y_ptr,
    M, K: tl.constexpr, N: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """y = y0 * (low @ w_up + cb) + y0 — the add_addmm_mul epilogue fused into
    the GEMM store. x0 == xl == y0 in the served model, so the epilogue loads
    ONE (rows, D) stream and writes one. Rounds at each eager op: matmul,
    +bias, *y0, +y0."""
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
        a = tl.load(low_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(wu_ptr + offs_k[:, None] * N + offs_n[None, :],
                    mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(a, w, acc, out_dtype=tl.float32, input_precision="ieee")

    dt = y_ptr.dtype.element_ty
    eb = acc.to(dt).to(tl.float32)                       # matmul rounds
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    t = (eb + bias[None, :]).to(dt).to(tl.float32)       # + cb rounds
    y0v = tl.load(y0_ptr + offs_m[:, None] * N + offs_n[None, :],
                  mask=m_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)
    c = (y0v * t).to(dt).to(tl.float32)                  # mul rounds
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :],
             (c + y0v).to(dt), mask=m_mask[:, None] & n_mask[None, :])


def kernel_function(*tensors: torch.Tensor) -> torch.Tensor:
    x, dom, w1, b1, w2, b2, w_down, w_up, cb = tensors
    m, d = x.shape
    gate_in = w1.shape[0]
    gate_hidden = w1.shape[1]
    rank = w_down.shape[1]
    device, dtype = x.device, x.dtype

    # Gate input as served: cat([dom, x]) — pool-managed alloc, capture-safe.
    xc = torch.cat([dom, x], dim=-1)

    h = torch.empty((m, gate_hidden), device=device, dtype=dtype)
    _gemm_bias_relu_kernel[(triton.cdiv(m, 64), triton.cdiv(gate_hidden, 64))](
        xc, w1, b1, h, m, K=gate_in, N=gate_hidden, BM=64, BN=64, BK=64,
        num_warps=8, num_stages=3,
    )

    y0 = torch.empty((m, d), device=device, dtype=dtype)
    _gemm_gate_mul_kernel[(triton.cdiv(m, 64), triton.cdiv(d, 64))](
        h, w2, b2, x, y0, m, K=gate_hidden, N=d, BM=64, BN=64, BK=64,
        num_warps=8, num_stages=3,
    )

    low = torch.empty((m, rank), device=device, dtype=dtype)
    _gemm_low_kernel[(triton.cdiv(m, 64), triton.cdiv(rank, 64))](
        y0, w_down, low, m, K=d, N=rank, BM=64, BN=64, BK=64,
        num_warps=8, num_stages=3,
    )

    y = torch.empty((m, d), device=device, dtype=dtype)
    _gemm_cross_epilogue_kernel[(triton.cdiv(m, 64), triton.cdiv(d, 64))](
        low, w_up, cb, y0, y, m, K=rank, N=d, BM=64, BN=64, BK=64,
        num_warps=8, num_stages=3,
    )
    return y
