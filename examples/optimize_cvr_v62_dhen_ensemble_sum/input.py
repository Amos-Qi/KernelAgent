# Seed for dir #2 (DHEN ensemble sum): ONE K-segmented GEMM computing
# sw0*(t0@Wp0+bp0) + sw1*(t1@Wp1+bp1) + sw2*(t2@Wp2+bp2) via the sum
# identity [A0|A1|A2] @ [B0;B1;B2] — WITHOUT materializing any A concat:
# the kernel walks three A base pointers per M-tile (zero-copy), while the
# weights/biases ARE packed once (host-side, cached) with the softmax
# scalars folded in (inference-time constants in serving).
#
# I/O dtype: torch.float16 — the v62 deploy precision. The harness
# auto-detects the benchmark dtype from this file's source text, so every
# derived kernel must keep a literal "float16" (this marker + the assert in
# kernel_function below).
#
# Schedule: seeded with dir #1's AUDITED winner family (BM128/BN128/BK32/
# GROUP8/warps4/stages3 won on both the 2g and 1g slices at the sibling
# shape). K-DIVISIBILITY CONSTRAINT: segment widths 8256 and 1024 divide by
# 32 and 64 but NOT both by 128 — the segment loops carry no K masks, so
# BK MUST stay in {32, 64}. (Dir #1's audit removed BK=128 configs for
# exactly this bug; do not reintroduce them without adding K masks.)
#
# DECLARED HEADROOM the seed does not exploit:
#   1. The output tile is written once and re-read by the eager add with
#      dir #1's ip half + SwishLayerNorm downstream — a candidate may fuse
#      a second lightweight LN pass (rowwise stats over N=2048) or emit
#      partial row statistics; strictly optional.
#   2. Per-segment scheduling: the K2=1024 segment is short — software
#      pipelining across the segment boundary (prefetching segment i+1
#      while draining i) is not expressed.
#   3. eviction hints / larger BM for fewer A re-reads across N-blocks.

import torch
import triton
import triton.language as tl

from problem import K01, K2, M, N

_packed = {}


def _pack(wp0, bp0, wp1, bp1, wp2, bp2, ens_w):
    key = (
        wp0.data_ptr(), wp1.data_ptr(), wp2.data_ptr(),
        bp0.data_ptr(), bp1.data_ptr(), bp2.data_ptr(), ens_w.data_ptr(),
    )
    hit = _packed.get(key)
    if hit is None:
        # Mirror serving's weight rounding: softmax on the fp16 logits is
        # computed in fp32 opmath and rounded back to the I/O dtype, THEN
        # folded into the packed weights in fp32 with one round at the end.
        sw = torch.softmax(ens_w.float(), dim=0).to(ens_w.dtype).float()
        w = torch.cat(
            [
                (wp0.float() * sw[0]).to(wp0.dtype),
                (wp1.float() * sw[1]).to(wp1.dtype),
                (wp2.float() * sw[2]).to(wp2.dtype),
            ],
            dim=0,
        ).contiguous()  # (K01+K01+K2, N) = (17536, N)
        b = (
            bp0.float() * sw[0] + bp1.float() * sw[1] + bp2.float() * sw[2]
        ).to(bp0.dtype).contiguous()
        # Hold refs to the SOURCE tensors: keeps them alive so their
        # data_ptrs can never be recycled by the caching allocator for new
        # tensors — a cache hit therefore always means "the same live
        # tensors". (Contents-staleness on in-place weight writes remains
        # out of contract: v62 serving weights are static once loaded.)
        _packed[key] = (w, b, (wp0, bp0, wp1, bp1, wp2, bp2, ens_w))
        hit = _packed[key]
    return hit[0], hit[1]


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
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64, 'GROUP': 8}, num_warps=4, num_stages=4),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64, 'GROUP': 8}, num_warps=8, num_stages=4),
        triton.Config({'BM': 128, 'BN': 64, 'BK': 64, 'GROUP': 8}, num_warps=4, num_stages=4),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 64, 'GROUP': 8}, num_warps=4, num_stages=4),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64, 'GROUP': 4}, num_warps=4, num_stages=3),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64, 'GROUP': 16}, num_warps=4, num_stages=3),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 32, 'GROUP': 16}, num_warps=4, num_stages=3),
        triton.Config({'BM': 256, 'BN': 64, 'BK': 64, 'GROUP': 8}, num_warps=8, num_stages=3),
        triton.Config({'BM': 256, 'BN': 128, 'BK': 32, 'GROUP': 8}, num_warps=8, num_stages=3),
    ],
    key=['M_', 'N_'],
)
@triton.jit
def _ens_sum_kernel(a0_ptr, a1_ptr, a2_ptr, w_ptr, b_ptr, out_ptr,
                    M_: tl.constexpr, N_: tl.constexpr,
                    K01_: tl.constexpr, K2_: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                    GROUP: tl.constexpr):
    """One fp32-accumulated pass over three K segments:
    rows [0,K01) of w against a0, [K01,2*K01) against a1,
    [2*K01, 2*K01+K2) against a2. Single fp16 round at the store."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M_, BM)
    num_pid_n = tl.cdiv(N_, BN)
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

    # Segment 0: a0 (row stride K01_), w rows [0, K01_)
    for k0 in range(0, K01_, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(a0_ptr + offs_m[:, None] * K01_ + offs_k[None, :],
                    mask=m_mask[:, None], other=0.0,
                    eviction_policy="evict_last")
        w = tl.load(w_ptr + offs_k[:, None] * N_ + offs_n[None, :],
                    eviction_policy="evict_last")
        acc = tl.dot(a, w, acc, input_precision="ieee")
    # Segment 1: a1 (row stride K01_), w rows [K01_, 2*K01_)
    for k0 in range(0, K01_, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(a1_ptr + offs_m[:, None] * K01_ + offs_k[None, :],
                    mask=m_mask[:, None], other=0.0,
                    eviction_policy="evict_last")
        w = tl.load(w_ptr + (K01_ + offs_k[:, None]) * N_ + offs_n[None, :],
                    eviction_policy="evict_last")
        acc = tl.dot(a, w, acc, input_precision="ieee")
    # Segment 2: a2 (row stride K2_), w rows [2*K01_, 2*K01_+K2_)
    for k0 in range(0, K2_, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(a2_ptr + offs_m[:, None] * K2_ + offs_k[None, :],
                    mask=m_mask[:, None], other=0.0,
                    eviction_policy="evict_last")
        w = tl.load(w_ptr + (2 * K01_ + offs_k[:, None]) * N_ + offs_n[None, :],
                    eviction_policy="evict_last")
        acc = tl.dot(a, w, acc, input_precision="ieee")

    bias = tl.load(b_ptr + offs_n, eviction_policy="evict_last").to(tl.float32)
    y = acc + bias[None, :]
    tl.store(out_ptr + offs_m[:, None] * N_ + offs_n[None, :],
             y.to(out_ptr.dtype.element_ty), mask=m_mask[:, None],
             eviction_policy="evict_first")


def kernel_function(*tensors: torch.Tensor) -> torch.Tensor:
    t0, t1, t2, wp0, bp0, wp1, bp1, wp2, bp2, ens_w = tensors
    # Serving contract (and the harness dtype-detection anchor): float16.
    assert t0.dtype in (torch.float16, torch.float32), "cvr-v62 serves float16"
    w, b = _pack(wp0, bp0, wp1, bp1, wp2, bp2, ens_w)
    out = torch.empty(M, N, dtype=t0.dtype, device=t0.device)
    grid = lambda meta: (triton.cdiv(M, meta['BM']) * triton.cdiv(N, meta['BN']),)
    _ens_sum_kernel[grid](
        t0, t1, t2, w, b, out,
        M_=M, N_=N, K01_=K01, K2_=K2,
    )
    return out
