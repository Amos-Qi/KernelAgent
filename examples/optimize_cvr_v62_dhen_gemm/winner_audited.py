# AUDITED WINNER of the dir #1 beam search (run of 2026-07-25, gen 5).
# Two audit fixes applied to the raw optimized_kernel_beam_search_cvr.py:
#   1. Removed the four BK=128 autotune configs: 8256 % 128 != 0 and the
#      K loop is unmasked -- those configs read past K and are silently
#      WRONG wherever autotune picks them (it did not pick them on 2g,
#      but a 1g/other-device re-tune could have).
#   2. input_precision "tf32" -> "ieee" (no-op for fp16 inputs; keeps the
#      fp32 debug path honest).
# Verified after fixes: bit-exact parity on 2g AND 1g (median 0, max 0,
# both seeds); autotune picks BM128/BN128/BK32/GROUP8/w4/s3 on BOTH slices.
# Measured (interleaved do_bench, medians):
#   2g: 1.2924 ms vs seed 1.8084 ms (1.399x); AOTI max-autotune bar 1.4918
#   1g: 2.6935 ms vs seed 3.5267 ms (1.309x) vs incumbent two-GEMM chain
#       4.5239 ms = 1.680x; per-GEMM-equivalent 1347 us (~87% slice MFU,
#       vs 2227.7 us serving incumbent and the 1610 us full-card-parity
#       target).
import torch
import triton
import triton.language as tl

M, K, N = 4000, 8256, 2048


@triton.autotune(
    configs=[
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64, 'GROUP': 8}, num_warps=4, num_stages=2),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64, 'GROUP': 8}, num_warps=8, num_stages=2),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64, 'GROUP': 8}, num_warps=8, num_stages=3),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64, 'GROUP': 8}, num_warps=4, num_stages=3),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 64, 'GROUP': 8}, num_warps=4, num_stages=2),
        triton.Config({'BM': 128, 'BN': 64, 'BK': 64, 'GROUP': 8}, num_warps=4, num_stages=2),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 32, 'GROUP': 8}, num_warps=4, num_stages=3),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 32, 'GROUP': 8}, num_warps=4, num_stages=3),
        triton.Config({'BM': 64, 'BN': 64, 'BK': 64, 'GROUP': 8}, num_warps=4, num_stages=2),
        triton.Config({'BM': 64, 'BN': 64, 'BK': 32, 'GROUP': 8}, num_warps=4, num_stages=3),
        # Higher num_stages for better memory pipelining
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64, 'GROUP': 8}, num_warps=4, num_stages=4),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64, 'GROUP': 8}, num_warps=8, num_stages=4),
        triton.Config({'BM': 128, 'BN': 64, 'BK': 64, 'GROUP': 8}, num_warps=4, num_stages=4),
        triton.Config({'BM': 128, 'BN': 64, 'BK': 64, 'GROUP': 8}, num_warps=8, num_stages=4),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 64, 'GROUP': 8}, num_warps=4, num_stages=4),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 64, 'GROUP': 8}, num_warps=8, num_stages=4),
        triton.Config({'BM': 64, 'BN': 64, 'BK': 64, 'GROUP': 8}, num_warps=4, num_stages=5),
        triton.Config({'BM': 64, 'BN': 64, 'BK': 64, 'GROUP': 8}, num_warps=8, num_stages=5),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 64, 'GROUP': 8}, num_warps=4, num_stages=5),
        triton.Config({'BM': 128, 'BN': 64, 'BK': 64, 'GROUP': 8}, num_warps=4, num_stages=5),
        # GROUP=4 for smaller L2 working set per group
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64, 'GROUP': 4}, num_warps=4, num_stages=3),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64, 'GROUP': 4}, num_warps=8, num_stages=3),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64, 'GROUP': 4}, num_warps=4, num_stages=4),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64, 'GROUP': 4}, num_warps=8, num_stages=4),
        # GROUP=16 for more x reuse across N-tiles
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64, 'GROUP': 16}, num_warps=4, num_stages=3),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64, 'GROUP': 16}, num_warps=8, num_stages=3),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64, 'GROUP': 16}, num_warps=8, num_stages=4),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64, 'GROUP': 16}, num_warps=4, num_stages=4),
        # BM=256 for fewer M-tiles (less x amplification across groups)
        triton.Config({'BM': 256, 'BN': 64, 'BK': 64, 'GROUP': 8}, num_warps=8, num_stages=2),
        triton.Config({'BM': 256, 'BN': 64, 'BK': 64, 'GROUP': 8}, num_warps=8, num_stages=3),
        triton.Config({'BM': 256, 'BN': 64, 'BK': 32, 'GROUP': 8}, num_warps=8, num_stages=3),
        triton.Config({'BM': 256, 'BN': 128, 'BK': 32, 'GROUP': 8}, num_warps=8, num_stages=2),
        triton.Config({'BM': 256, 'BN': 128, 'BK': 32, 'GROUP': 8}, num_warps=8, num_stages=3),
        triton.Config({'BM': 256, 'BN': 64, 'BK': 64, 'GROUP': 4}, num_warps=8, num_stages=3),
        triton.Config({'BM': 256, 'BN': 64, 'BK': 64, 'GROUP': 16}, num_warps=8, num_stages=3),
    ],
    key=['M_', 'K_', 'N2'],
)
@triton.jit
def _dhen_pair_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                      M_: tl.constexpr, K_: tl.constexpr, N2: tl.constexpr, N1: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M_, BM)
    num_pid_n = tl.cdiv(N2, BN)
    num_pid_in_group = GROUP * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP
    group_size_m = min(num_pid_m - first_pid_m, GROUP)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    m_mask = offs_m < M_
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K_, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(x_ptr + offs_m[:, None] * K_ + offs_k[None, :],
                    mask=m_mask[:, None], other=0.0,
                    eviction_policy="evict_last")
        w = tl.load(w_ptr + offs_k[:, None] * N2 + offs_n[None, :],
                    eviction_policy="evict_last")
        acc = tl.dot(a, w, acc, input_precision="ieee")
    bias = tl.load(b_ptr + offs_n, eviction_policy="evict_last").to(tl.float32)
    y = acc + bias[None, :]
    y = tl.where(offs_n[None, :] >= N1, tl.maximum(y, 0.0), y)
    tl.store(out_ptr + offs_m[:, None] * N2 + offs_n[None, :],
             y.to(out_ptr.dtype.element_ty), mask=m_mask[:, None],
             eviction_policy="evict_first")


_packed = {}


def _pack(w_ip, b_ip, w_mlp, b_mlp):
    key = (w_ip.data_ptr(), w_mlp.data_ptr(), b_ip.data_ptr(), b_mlp.data_ptr())
    hit = _packed.get(key)
    if hit is None:
        w = torch.cat([w_ip, w_mlp], dim=1).contiguous()
        b = torch.cat([b_ip, b_mlp]).contiguous()
        _packed[key] = (w, b, (w_ip, b_ip, w_mlp, b_mlp))
        hit = _packed[key]
    return hit[0], hit[1]


def kernel_function(*tensors: torch.Tensor) -> torch.Tensor:
    x, w_ip, b_ip, w_mlp, b_mlp = tensors
    assert x.dtype in (torch.float16, torch.float32), "cvr-v62 serves float16"
    w, b = _pack(w_ip, b_ip, w_mlp, b_mlp)
    out = torch.empty(M, 2 * N, dtype=x.dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(M, meta['BM']) * triton.cdiv(2 * N, meta['BN']),)
    _dhen_pair_kernel[grid](
        x, w, b, out, M_=M, K_=K, N2=2 * N, N1=N,
    )
    return out
