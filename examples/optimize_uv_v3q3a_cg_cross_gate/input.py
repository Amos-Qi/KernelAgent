# Starting kernel (hand-written re-seed): a CORRECT, working Triton pipeline
# for the EPNet-gate + low-rank-cross region, with each pointwise epilogue
# fused into its producing GEMM's store — the byte-saving structure the
# search should refine (tiling, persistence, cross-stage fusion, keeping the
# x tile resident across stages).
#
# Stage-rounding semantics mirror the eager reference EXACTLY: torch's bf16
# matmul rounds its output to bf16, activations run in fp32, and every served
# stage stores bf16 — so each kernel computes fp32-accumulated dots (ieee, no
# TF32), rounds to the storage dtype where torch does, and applies the
# fp32 epilogue on the rounded value.
#
# Capture-safety: static shapes, no per-call host->device transfers, only
# device-side launches and pool-managed allocations. Both test.py gates pass.

import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_silu_kernel(
    x_ptr, w_ptr, h_ptr,
    M, K: tl.constexpr, N: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """h = silu(x @ w1), silu fused into the store. x:(M,K) w:(K,N) h:(M,N)."""
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

    # torch: matmul rounds to bf16, then silu in fp32, then store bf16
    hb = acc.to(h_ptr.dtype.element_ty).to(tl.float32)
    hs = hb * tl.sigmoid(hb)
    tl.store(h_ptr + offs_m[:, None] * N + offs_n[None, :],
             hs.to(h_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _gemm_gate_mul_kernel(
    h_ptr, w2_ptr, x_ptr, y0_ptr,
    M, K: tl.constexpr, N: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """y0 = x * (2*sigmoid(h @ w2)) — the mm_mul_sigmoid epilogue fused into
    the GEMM store (x tile loaded once, no full-tensor round trip)."""
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

    gb = acc.to(y0_ptr.dtype.element_ty).to(tl.float32)  # matmul's bf16 round
    g = 2.0 * tl.sigmoid(gb)
    xv = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :],
                 mask=m_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)
    tl.store(y0_ptr + offs_m[:, None] * N + offs_n[None, :],
             (xv * g).to(y0_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _gemm_low_kernel(
    y0_ptr, u_ptr, low_ptr,
    M, K: tl.constexpr, N: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """low = y0 @ u. (M,K)@(K,N) with N = rank 128; stored bf16 as served."""
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
        w = tl.load(u_ptr + offs_k[:, None] * N + offs_n[None, :],
                    mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(a, w, acc, out_dtype=tl.float32, input_precision="ieee")

    tl.store(low_ptr + offs_m[:, None] * N + offs_n[None, :],
             acc.to(low_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _gemm_cross_epilogue_kernel(
    low_ptr, v_ptr, b_ptr, x0_ptr, y0_ptr, y_ptr,
    M, K: tl.constexpr, N: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """y = x0 * (low @ v + b) + y0 — the add_addmm_mul epilogue fused into the
    GEMM store (x0 and y0 tiles loaded once each, written once)."""
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
        w = tl.load(v_ptr + offs_k[:, None] * N + offs_n[None, :],
                    mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(a, w, acc, out_dtype=tl.float32, input_precision="ieee")

    # torch: matmul rounds bf16, then + b (bf16 add rounds again), then the
    # fp32 mul-add epilogue, then the final bf16 store.
    eb = acc.to(y_ptr.dtype.element_ty).to(tl.float32)
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    e2 = (eb + bias).to(y_ptr.dtype.element_ty).to(tl.float32)
    x0v = tl.load(x0_ptr + offs_m[:, None] * N + offs_n[None, :],
                  mask=m_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)
    y0v = tl.load(y0_ptr + offs_m[:, None] * N + offs_n[None, :],
                  mask=m_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :],
             (x0v * e2 + y0v).to(y_ptr.dtype.element_ty),
             mask=m_mask[:, None] & n_mask[None, :])


def kernel_function(*tensors: torch.Tensor) -> torch.Tensor:
    x, x0, w1, w2, u, v, b = tensors
    m, d = x.shape
    gate_hidden = w1.shape[1]
    rank = u.shape[1]
    device, dtype = x.device, x.dtype

    h = torch.empty((m, gate_hidden), device=device, dtype=dtype)
    _gemm_silu_kernel[(triton.cdiv(m, 64), triton.cdiv(gate_hidden, 64))](
        x, w1, h, m, K=d, N=gate_hidden, BM=64, BN=64, BK=64,
        num_warps=8, num_stages=3,
    )

    y0 = torch.empty((m, d), device=device, dtype=dtype)
    _gemm_gate_mul_kernel[(triton.cdiv(m, 64), triton.cdiv(d, 64))](
        h, w2, x, y0, m, K=gate_hidden, N=d, BM=64, BN=64, BK=64,
        num_warps=8, num_stages=3,
    )

    low = torch.empty((m, rank), device=device, dtype=dtype)
    _gemm_low_kernel[(triton.cdiv(m, 64), triton.cdiv(rank, 64))](
        y0, u, low, m, K=d, N=rank, BM=64, BN=64, BK=64,
        num_warps=8, num_stages=3,
    )

    y = torch.empty((m, d), device=device, dtype=dtype)
    _gemm_cross_epilogue_kernel[(triton.cdiv(m, 64), triton.cdiv(d, 64))](
        low, v, b, x0, y0, y, m, K=rank, N=d, BM=64, BN=64, BK=64,
        num_warps=8, num_stages=3,
    )
    return y
