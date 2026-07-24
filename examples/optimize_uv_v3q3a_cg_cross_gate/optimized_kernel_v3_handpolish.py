# dir #10 hand-polish (v3) — EPNet GateNU + CrossLayerV2, bfloat16 serving contract.
# Structure: the round-2 search winner's 4-kernel chain (cat-free two-segment
# gate1; sigmoid-gate epilogue fused into gate2; cross epilogue fused into the
# up-projection). Hand-polish = per-kernel launch schedules re-derived from
# Inductor autotune tables and re-swept IN-CHAIN on the 2g MIG slice:
#   gate1 (6144x1472->512): BK 32->64, warps 4->8
#   gate2 (6144x512->1328): BM 64->128 (autotune-confirmed)
#   low   (6144x1328->128): r2w1 config confirmed best in-chain (standalone
#                           autotune BK64/w8 LOSES in-chain; L2 state differs)
#   epi   (6144x128->1328): BM 64->128
# Structural fusions were built, verified bit-identical, and REJECTED on
# measurement: mega gate2+low+epi 0.383ms; light low+epi 0.245ms; vs 0.219ms
# for this file. On sm_120 schedule quality beats DRAM-byte savings.
# Numerics: gate1 BK change alters fp32 accumulation chunk order (1-ulp
# class); all stage round points match eager exactly (addmm rounds once;
# every elementwise op rounds to storage dtype).
import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_bias_relu_kernel(
    dom_ptr, x_ptr, w_ptr, b_ptr, h_ptr,
    M, K_DOM: tl.constexpr, K_X: tl.constexpr, N: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """h = relu(addmm(b, cat([dom, x]), w))"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    
    for k0 in range(0, K_DOM, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(dom_ptr + offs_m[:, None] * K_DOM + offs_k[None, :],
                    mask=m_mask[:, None] & (offs_k[None, :] < K_DOM), other=0.0)
        w = tl.load(w_ptr + offs_k[:, None] * N + offs_n[None, :],
                    mask=(offs_k[:, None] < K_DOM) & n_mask[None, :], other=0.0)
        acc = tl.dot(a, w, acc, out_dtype=tl.float32, input_precision="ieee")

    for k0 in range(0, K_X, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(x_ptr + offs_m[:, None] * K_X + offs_k[None, :],
                    mask=m_mask[:, None] & (offs_k[None, :] < K_X), other=0.0)
        w = tl.load(w_ptr + (K_DOM + offs_k)[:, None] * N + offs_n[None, :],
                    mask=(offs_k[:, None] < K_X) & n_mask[None, :], other=0.0)
        acc = tl.dot(a, w, acc, out_dtype=tl.float32, input_precision="ieee")

    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    hb = (acc + bias[None, :]).to(h_ptr.dtype.element_ty).to(tl.float32)
    hr = tl.maximum(hb, 0.0)
    tl.store(h_ptr + offs_m[:, None] * N + offs_n[None, :],
             hr.to(h_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _gemm_gate_mul_kernel(
    h_ptr, w2_ptr, b2_ptr, x_ptr, y0_ptr,
    M, K: tl.constexpr, N: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """y0 = x * (2*sigmoid(addmm(b2, h, w2)))"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(h_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=m_mask[:, None] & (offs_k[None, :] < K), other=0.0)
        w = tl.load(w2_ptr + offs_k[:, None] * N + offs_n[None, :],
                    mask=(offs_k[:, None] < K) & n_mask[None, :], other=0.0)
        acc = tl.dot(a, w, acc, out_dtype=tl.float32, input_precision="ieee")

    dt = y0_ptr.dtype.element_ty
    bias = tl.load(b2_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    gb = (acc + bias[None, :]).to(dt).to(tl.float32)
    s = tl.sigmoid(gb).to(dt).to(tl.float32)
    l2 = (s * 2.0).to(dt).to(tl.float32)
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
    """low = y0 @ w_down."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(y0_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=m_mask[:, None] & (offs_k[None, :] < K), other=0.0)
        w = tl.load(wd_ptr + offs_k[:, None] * N + offs_n[None, :],
                    mask=(offs_k[:, None] < K) & n_mask[None, :], other=0.0)
        acc = tl.dot(a, w, acc, out_dtype=tl.float32, input_precision="ieee")

    tl.store(low_ptr + offs_m[:, None] * N + offs_n[None, :],
             acc.to(low_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _gemm_cross_epilogue_kernel(
    low_ptr, wu_ptr, b_ptr, y0_ptr, y_ptr,
    M, K: tl.constexpr, N: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """y = y0 * (low @ w_up + cb) + y0"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(low_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=m_mask[:, None] & (offs_k[None, :] < K), other=0.0)
        w = tl.load(wu_ptr + offs_k[:, None] * N + offs_n[None, :],
                    mask=(offs_k[:, None] < K) & n_mask[None, :], other=0.0)
        acc = tl.dot(a, w, acc, out_dtype=tl.float32, input_precision="ieee")

    dt = y_ptr.dtype.element_ty
    eb = acc.to(dt).to(tl.float32)
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    t = (eb + bias[None, :]).to(dt).to(tl.float32)
    y0v = tl.load(y0_ptr + offs_m[:, None] * N + offs_n[None, :],
                  mask=m_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)
    c = (y0v * t).to(dt).to(tl.float32)
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :],
             (c + y0v).to(dt), mask=m_mask[:, None] & n_mask[None, :])



def kernel_function(*tensors: torch.Tensor) -> torch.Tensor:
    x, dom, w1, b1, w2, b2, w_down, w_up, cb = tensors
    assert x.dtype in (torch.bfloat16, torch.float32), "v3-q3a serves bfloat16"
    m, d = x.shape
    k_dom = dom.shape[1]
    gate_hidden = w1.shape[1]
    rank = w_down.shape[1]
    device, dtype = x.device, x.dtype

    h = torch.empty((m, gate_hidden), device=device, dtype=dtype)
    _gemm_bias_relu_kernel[(triton.cdiv(m, 64), triton.cdiv(gate_hidden, 128))](
        dom, x, w1, b1, h, m, K_DOM=k_dom, K_X=d, N=gate_hidden, BM=64, BN=128, BK=64,
        num_warps=8, num_stages=3,
    )

    y0 = torch.empty((m, d), device=device, dtype=dtype)
    _gemm_gate_mul_kernel[(triton.cdiv(m, 128), triton.cdiv(d, 128))](
        h, w2, b2, x, y0, m, K=gate_hidden, N=d, BM=128, BN=128, BK=32,
        num_warps=4, num_stages=3,
    )

    low = torch.empty((m, rank), device=device, dtype=dtype)
    _gemm_low_kernel[(triton.cdiv(m, 64), triton.cdiv(rank, 64))](
        y0, w_down, low, m, K=d, N=rank, BM=64, BN=64, BK=32,
        num_warps=4, num_stages=3,
    )

    y = torch.empty((m, d), device=device, dtype=dtype)
    _gemm_cross_epilogue_kernel[(triton.cdiv(m, 128), triton.cdiv(d, 128))](
        low, w_up, cb, y0, y, m, K=rank, N=d, BM=128, BN=128, BK=32,
        num_warps=4, num_stages=3,
    )
    return y
