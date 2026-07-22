import torch
import triton
import triton.language as tl

ROWS = 24 * 1024
SHARED_DIM = 512
SKIP_DIM = 64
FEATS_DIM = 576
TOWER_OUT = 960
IAP_DIM = 384
ADREV_DIM = 192
RET_DIM = 192
ABCE_DIM = 192
IAP_IN_DIM = IAP_DIM + RET_DIM
ADREV_IN_DIM = ADREV_DIM + ABCE_DIM + RET_DIM


@triton.jit
def copy_kernel(src_ptr, dst_ptr, M, D, stride_src_m, stride_src_d, stride_dst_m, stride_dst_d,
                BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = (offs_m[:, None] < M) & (offs_d[None, :] < D)
    src = src_ptr + offs_m[:, None] * stride_src_m + offs_d[None, :] * stride_src_d
    dst = dst_ptr + offs_m[:, None] * stride_dst_m + offs_d[None, :] * stride_dst_d
    val = tl.load(src, mask=mask, other=0.0)
    tl.store(dst, val, mask=mask)


@triton.jit
def matmul_silu_kernel(A_ptr, W_ptr, B_ptr, C_ptr, M, N, K,
                       stride_am, stride_ak, stride_wk, stride_wn, stride_cm, stride_cn,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)
        w = tl.load(W_ptr + (k + offs_k[:, None]) * stride_wk + offs_n[None, :] * stride_wn)
        acc += tl.dot(a, w)
    b = tl.load(B_ptr + offs_n)
    acc += b[None, :]
    acc = acc * tl.sigmoid(acc)
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc.to(tl.bfloat16))


@triton.jit
def iap_head_kernel(iap_ptr, ret_ptr, w7_ptr, b7_ptr, w28_ptr, b28_ptr, acc_ptr,
                    M, K_iap, K_ret,
                    stride_iap_m, stride_iap_k, stride_ret_m, stride_ret_k,
                    stride_wk, stride_wn, stride_acc_m, stride_acc_c,
                    ADD: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, N_PAD: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_c = tl.arange(0, N_PAD)
    acc7 = tl.zeros((BLOCK_M, N_PAD), dtype=tl.float32)
    acc28 = tl.zeros((BLOCK_M, N_PAD), dtype=tl.float32)
    for k in range(0, K_iap, BLOCK_K):
        a = tl.load(iap_ptr + offs_m[:, None] * stride_iap_m + (k + offs_k[None, :]) * stride_iap_k)
        w7 = tl.load(w7_ptr + (k + offs_k[:, None]) * stride_wk + offs_c[None, :] * stride_wn,
                     mask=offs_c[None, :] < 3, other=0.0)
        w28 = tl.load(w28_ptr + (k + offs_k[:, None]) * stride_wk + offs_c[None, :] * stride_wn,
                      mask=offs_c[None, :] < 3, other=0.0)
        acc7 += tl.dot(a, w7)
        acc28 += tl.dot(a, w28)
    for k in range(0, K_ret, BLOCK_K):
        a = tl.load(ret_ptr + offs_m[:, None] * stride_ret_m + (k + offs_k[None, :]) * stride_ret_k)
        w7 = tl.load(w7_ptr + ((K_iap + k) + offs_k[:, None]) * stride_wk + offs_c[None, :] * stride_wn,
                     mask=offs_c[None, :] < 3, other=0.0)
        w28 = tl.load(w28_ptr + ((K_iap + k) + offs_k[:, None]) * stride_wk + offs_c[None, :] * stride_wn,
                      mask=offs_c[None, :] < 3, other=0.0)
        acc7 += tl.dot(a, w7)
        acc28 += tl.dot(a, w28)
    b7 = tl.load(b7_ptr + offs_c, mask=offs_c < 3, other=0.0)
    b28 = tl.load(b28_ptr + offs_c, mask=offs_c < 3, other=0.0)
    acc7 += b7[None, :]
    acc28 += b28[None, :]

    # Extract columns via masked reduction
    t0_7 = tl.sum(tl.where(offs_c[None, :] == 0, acc7, 0.0), axis=1)
    t1_7 = tl.sum(tl.where(offs_c[None, :] == 1, acc7, 0.0), axis=1)
    t2_7 = tl.sum(tl.where(offs_c[None, :] == 2, acc7, 0.0), axis=1)
    prob7 = tl.sigmoid(t0_7)
    scale7 = tl.where(t2_7 > 0, t2_7 + tl.log(1.0 + tl.exp(-t2_7)), tl.log(1.0 + tl.exp(t2_7)))
    value7 = tl.exp(t1_7 + 0.5 * scale7 * scale7)
    final7 = prob7 * value7

    t0_28 = tl.sum(tl.where(offs_c[None, :] == 0, acc28, 0.0), axis=1)
    t1_28 = tl.sum(tl.where(offs_c[None, :] == 1, acc28, 0.0), axis=1)
    t2_28 = tl.sum(tl.where(offs_c[None, :] == 2, acc28, 0.0), axis=1)
    prob28 = tl.sigmoid(t0_28)
    scale28 = tl.where(t2_28 > 0, t2_28 + tl.log(1.0 + tl.exp(-t2_28)), tl.log(1.0 + tl.exp(t2_28)))
    value28 = tl.exp(t1_28 + 0.5 * scale28 * scale28)
    final28 = prob28 * value28

    # Store each column explicitly (unrolled to avoid list indexing in JIT)
    p0 = acc_ptr + offs_m * stride_acc_m + 0 * stride_acc_c
    p1 = acc_ptr + offs_m * stride_acc_m + 1 * stride_acc_c
    p2 = acc_ptr + offs_m * stride_acc_m + 2 * stride_acc_c
    p3 = acc_ptr + offs_m * stride_acc_m + 3 * stride_acc_c
    p4 = acc_ptr + offs_m * stride_acc_m + 4 * stride_acc_c
    p5 = acc_ptr + offs_m * stride_acc_m + 5 * stride_acc_c
    p6 = acc_ptr + offs_m * stride_acc_m + 6 * stride_acc_c
    p7 = acc_ptr + offs_m * stride_acc_m + 7 * stride_acc_c
    p8 = acc_ptr + offs_m * stride_acc_m + 8 * stride_acc_c
    p9 = acc_ptr + offs_m * stride_acc_m + 9 * stride_acc_c

    if ADD:
        tl.store(p0, tl.load(p0) + prob7)
        tl.store(p1, tl.load(p1) + value7)
        tl.store(p2, tl.load(p2) + final7)
        tl.store(p3, tl.load(p3) + t1_7)
        tl.store(p4, tl.load(p4) + scale7)
        tl.store(p5, tl.load(p5) + prob28)
        tl.store(p6, tl.load(p6) + value28)
        tl.store(p7, tl.load(p7) + final28)
        tl.store(p8, tl.load(p8) + t1_28)
        tl.store(p9, tl.load(p9) + scale28)
    else:
        tl.store(p0, prob7)
        tl.store(p1, value7)
        tl.store(p2, final7)
        tl.store(p3, t1_7)
        tl.store(p4, scale7)
        tl.store(p5, prob28)
        tl.store(p6, value28)
        tl.store(p7, final28)
        tl.store(p8, t1_28)
        tl.store(p9, scale28)


@triton.jit
def adrev_head_kernel(adrev_ptr, abce_ptr, ret_ptr, w0_ptr, b0_ptr, w7_ptr, b7_ptr, w28_ptr, b28_ptr, acc_ptr,
                      M, K0, K1, K2,
                      stride_adrev_m, stride_adrev_k, stride_abce_m, stride_abce_k, stride_ret_m, stride_ret_k,
                      stride_wk, stride_acc_m, stride_acc_c,
                      ADD: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    acc0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc7 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc28 = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k in range(0, K0, BLOCK_K):
        a = tl.load(adrev_ptr + offs_m[:, None] * stride_adrev_m + (k + offs_k[None, :]) * stride_adrev_k).to(tl.float32)
        w0 = tl.load(w0_ptr + (k + offs_k) * stride_wk).to(tl.float32)
        w7 = tl.load(w7_ptr + (k + offs_k) * stride_wk).to(tl.float32)
        w28 = tl.load(w28_ptr + (k + offs_k) * stride_wk).to(tl.float32)
        acc0 += tl.sum(a * w0[None, :], axis=1)
        acc7 += tl.sum(a * w7[None, :], axis=1)
        acc28 += tl.sum(a * w28[None, :], axis=1)
    for k in range(0, K1, BLOCK_K):
        a = tl.load(abce_ptr + offs_m[:, None] * stride_abce_m + (k + offs_k[None, :]) * stride_abce_k).to(tl.float32)
        w0 = tl.load(w0_ptr + ((K0 + k) + offs_k) * stride_wk).to(tl.float32)
        w7 = tl.load(w7_ptr + ((K0 + k) + offs_k) * stride_wk).to(tl.float32)
        w28 = tl.load(w28_ptr + ((K0 + k) + offs_k) * stride_wk).to(tl.float32)
        acc0 += tl.sum(a * w0[None, :], axis=1)
        acc7 += tl.sum(a * w7[None, :], axis=1)
        acc28 += tl.sum(a * w28[None, :], axis=1)
    for k in range(0, K2, BLOCK_K):
        a = tl.load(ret_ptr + offs_m[:, None] * stride_ret_m + (k + offs_k[None, :]) * stride_ret_k).to(tl.float32)
        w0 = tl.load(w0_ptr + ((K0 + K1 + k) + offs_k) * stride_wk).to(tl.float32)
        w7 = tl.load(w7_ptr + ((K0 + K1 + k) + offs_k) * stride_wk).to(tl.float32)
        w28 = tl.load(w28_ptr + ((K0 + K1 + k) + offs_k) * stride_wk).to(tl.float32)
        acc0 += tl.sum(a * w0[None, :], axis=1)
        acc7 += tl.sum(a * w7[None, :], axis=1)
        acc28 += tl.sum(a * w28[None, :], axis=1)

    acc0 += tl.load(b0_ptr).to(tl.float32)
    acc7 += tl.load(b7_ptr).to(tl.float32)
    acc28 += tl.load(b28_ptr).to(tl.float32)

    v0 = tl.where(acc0 > 0, acc0 + tl.log(1.0 + tl.exp(-acc0)), tl.log(1.0 + tl.exp(acc0)))
    v7 = tl.where(acc7 > 0, acc7 + tl.log(1.0 + tl.exp(-acc7)), tl.log(1.0 + tl.exp(acc7)))
    v28 = tl.where(acc28 > 0, acc28 + tl.log(1.0 + tl.exp(-acc28)), tl.log(1.0 + tl.exp(acc28)))

    # Store each column explicitly (unrolled to avoid list indexing in JIT)
    p10 = acc_ptr + offs_m * stride_acc_m + 10 * stride_acc_c
    p11 = acc_ptr + offs_m * stride_acc_m + 11 * stride_acc_c
    p12 = acc_ptr + offs_m * stride_acc_m + 12 * stride_acc_c

    if ADD:
        tl.store(p10, tl.load(p10) + v0)
        tl.store(p11, tl.load(p11) + v7)
        tl.store(p12, tl.load(p12) + v28)
    else:
        tl.store(p10, v0)
        tl.store(p11, v7)
        tl.store(p12, v28)


@triton.jit
def simple_head_kernel(x_ptr, w_ptr, b_ptr, acc_ptr, M, K, stride_xm, stride_xk, stride_wk,
                       stride_acc_m, stride_acc_c, col, ADD: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(x_ptr + offs_m[:, None] * stride_xm + (k + offs_k[None, :]) * stride_xk).to(tl.float32)
        w = tl.load(w_ptr + (k + offs_k) * stride_wk).to(tl.float32)
        acc += tl.sum(a * w[None, :], axis=1)
    acc += tl.load(b_ptr).to(tl.float32)
    prob = tl.sigmoid(acc)
    p = acc_ptr + offs_m * stride_acc_m + col * stride_acc_c
    if ADD:
        tl.store(p, tl.load(p) + prob)
    else:
        tl.store(p, prob)


@triton.jit
def finalize_kernel(acc_ptr, out_ptr, M, stride_acc_m, stride_acc_c, stride_out_m, stride_out_c,
                    BLOCK_M: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, 16)
    mask_c = offs_c < 15
    val = tl.load(acc_ptr + offs_m[:, None] * stride_acc_m + offs_c[None, :] * stride_acc_c,
                  mask=mask_c[None, :], other=0.0)
    val = val * 0.5
    calib = tl.where((offs_c >= 10) & (offs_c < 13), 1.053, 1.0)
    val = val * calib[None, :]
    tl.store(out_ptr + offs_m[:, None] * stride_out_m + offs_c[None, :] * stride_out_c, val.to(tl.bfloat16))


_cache = {}


def _get_buffers(device, dtype):
    key = (device, dtype)
    if key not in _cache:
        feats = torch.empty(ROWS, FEATS_DIM, dtype=dtype, device=device)
        tower1 = torch.empty(ROWS, TOWER_OUT, dtype=dtype, device=device)
        tower2 = torch.empty(ROWS, TOWER_OUT, dtype=dtype, device=device)
        acc = torch.empty(ROWS, 15, dtype=torch.float32, device=device)
        out = torch.empty(ROWS, 16, dtype=dtype, device=device)
        _cache[key] = (feats, tower1, tower2, acc, out)
    return _cache[key]


def kernel_function(*tensors):
    shared = tensors[0]
    skip = tensors[1]
    weights = list(tensors[2:])
    device = shared.device
    dtype = shared.dtype
    feats, tower1, tower2, acc, out = _get_buffers(device, dtype)

    BM = 128
    BD = 128
    grid_s = (triton.cdiv(ROWS, BM), triton.cdiv(SHARED_DIM, BD))
    copy_kernel[grid_s](shared, feats, ROWS, SHARED_DIM,
                        shared.stride(0), shared.stride(1), feats.stride(0), feats.stride(1), BM, BD)
    skip_dst = feats[:, SHARED_DIM:]
    grid_k = (triton.cdiv(ROWS, BM), triton.cdiv(SKIP_DIM, BD))
    copy_kernel[grid_k](skip, skip_dst, ROWS, SKIP_DIM,
                        skip.stride(0), skip.stride(1), skip_dst.stride(0), skip_dst.stride(1), BM, BD)

    MM = 128
    NN = 64
    KK = 32
    grid_m = (ROWS // MM,)

    for c in range(2):
        base = c * 30
        add = 1 if c == 1 else 0

        # Tower first layer
        matmul_silu_kernel[(ROWS // MM, IAP_DIM // NN)](
            feats, weights[base + 0], weights[base + 1], tower1[:, :IAP_DIM],
            ROWS, IAP_DIM, FEATS_DIM,
            feats.stride(0), feats.stride(1), weights[base + 0].stride(0), weights[base + 0].stride(1),
            tower1.stride(0), tower1.stride(1), MM, NN, KK)
        matmul_silu_kernel[(ROWS // MM, ADREV_DIM // NN)](
            feats, weights[base + 4], weights[base + 5], tower1[:, IAP_DIM:IAP_DIM + ADREV_DIM],
            ROWS, ADREV_DIM, FEATS_DIM,
            feats.stride(0), feats.stride(1), weights[base + 4].stride(0), weights[base + 4].stride(1),
            tower1.stride(0), tower1.stride(1), MM, NN, KK)
        matmul_silu_kernel[(ROWS // MM, RET_DIM // NN)](
            feats, weights[base + 8], weights[base + 9], tower1[:, IAP_DIM + ADREV_DIM:IAP_DIM + ADREV_DIM + RET_DIM],
            ROWS, RET_DIM, FEATS_DIM,
            feats.stride(0), feats.stride(1), weights[base + 8].stride(0), weights[base + 8].stride(1),
            tower1.stride(0), tower1.stride(1), MM, NN, KK)
        matmul_silu_kernel[(ROWS // MM, ABCE_DIM // NN)](
            feats, weights[base + 12], weights[base + 13], tower1[:, IAP_DIM + ADREV_DIM + RET_DIM:],
            ROWS, ABCE_DIM, FEATS_DIM,
            feats.stride(0), feats.stride(1), weights[base + 12].stride(0), weights[base + 12].stride(1),
            tower1.stride(0), tower1.stride(1), MM, NN, KK)

        # Tower second layer
        matmul_silu_kernel[(ROWS // MM, IAP_DIM // NN)](
            tower1[:, :IAP_DIM], weights[base + 2], weights[base + 3], tower2[:, :IAP_DIM],
            ROWS, IAP_DIM, IAP_DIM,
            tower1.stride(0), tower1.stride(1), weights[base + 2].stride(0), weights[base + 2].stride(1),
            tower2.stride(0), tower2.stride(1), MM, NN, KK)
        matmul_silu_kernel[(ROWS // MM, ADREV_DIM // NN)](
            tower1[:, IAP_DIM:IAP_DIM + ADREV_DIM], weights[base + 6], weights[base + 7], tower2[:, IAP_DIM:IAP_DIM + ADREV_DIM],
            ROWS, ADREV_DIM, ADREV_DIM,
            tower1.stride(0), tower1.stride(1), weights[base + 6].stride(0), weights[base + 6].stride(1),
            tower2.stride(0), tower2.stride(1), MM, NN, KK)
        matmul_silu_kernel[(ROWS // MM, RET_DIM // NN)](
            tower1[:, IAP_DIM + ADREV_DIM:IAP_DIM + ADREV_DIM + RET_DIM], weights[base + 10], weights[base + 11],
            tower2[:, IAP_DIM + ADREV_DIM:IAP_DIM + ADREV_DIM + RET_DIM],
            ROWS, RET_DIM, RET_DIM,
            tower1.stride(0), tower1.stride(1), weights[base + 10].stride(0), weights[base + 10].stride(1),
            tower2.stride(0), tower2.stride(1), MM, NN, KK)
        matmul_silu_kernel[(ROWS // MM, ABCE_DIM // NN)](
            tower1[:, IAP_DIM + ADREV_DIM + RET_DIM:], weights[base + 14], weights[base + 15],
            tower2[:, IAP_DIM + ADREV_DIM + RET_DIM:],
            ROWS, ABCE_DIM, ABCE_DIM,
            tower1.stride(0), tower1.stride(1), weights[base + 14].stride(0), weights[base + 14].stride(1),
            tower2.stride(0), tower2.stride(1), MM, NN, KK)

        # Heads
        iap_head_kernel[grid_m](
            tower2[:, :IAP_DIM], tower2[:, IAP_DIM + ADREV_DIM:IAP_DIM + ADREV_DIM + RET_DIM],
            weights[base + 16], weights[base + 17], weights[base + 18], weights[base + 19], acc,
            ROWS, IAP_DIM, RET_DIM,
            tower2.stride(0), tower2.stride(1), tower2.stride(0), tower2.stride(1),
            weights[base + 16].stride(0), weights[base + 16].stride(1), acc.stride(0), acc.stride(1),
            add, MM, 64, 16)

        adrev_head_kernel[grid_m](
            tower2[:, IAP_DIM:IAP_DIM + ADREV_DIM], tower2[:, IAP_DIM + ADREV_DIM + RET_DIM:],
            tower2[:, IAP_DIM + ADREV_DIM:IAP_DIM + ADREV_DIM + RET_DIM],
            weights[base + 20], weights[base + 21], weights[base + 22], weights[base + 23],
            weights[base + 24], weights[base + 25], acc,
            ROWS, ADREV_DIM, ABCE_DIM, RET_DIM,
            tower2.stride(0), tower2.stride(1), tower2.stride(0), tower2.stride(1), tower2.stride(0), tower2.stride(1),
            weights[base + 20].stride(0), acc.stride(0), acc.stride(1),
            add, MM, 64)

        simple_head_kernel[grid_m](
            tower2[:, IAP_DIM + ADREV_DIM:IAP_DIM + ADREV_DIM + RET_DIM], weights[base + 26], weights[base + 27], acc,
            ROWS, RET_DIM, tower2.stride(0), tower2.stride(1), weights[base + 26].stride(0),
            acc.stride(0), acc.stride(1), 13, add, MM, 64)

        simple_head_kernel[grid_m](
            tower2[:, :IAP_DIM], weights[base + 28], weights[base + 29], acc,
            ROWS, IAP_DIM, tower2.stride(0), tower2.stride(1), weights[base + 28].stride(0),
            acc.stride(0), acc.stride(1), 14, add, MM, 64)

    finalize_kernel[(ROWS // MM,)](acc, out, ROWS, acc.stride(0), acc.stride(1), out.stride(0), out.stride(1), MM)
    return out