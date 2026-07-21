# Unity user-value model v3-q3a — FULL TritonBatchedMHA forward.
#
# Kernel dir #3: the whole batched-MHA module as one problem — Q projection,
# ragged K/V projection, bias_kv/zero_attn epilogue + masks, the attention
# core, and the output projection. Kernel dirs #1/#2 optimized the core and
# the K/V GEMM in isolation (verified 5.5x / 2.6x, both integrated into
# serving); this problem seeds from that integrated pipeline so the search
# only has to FUSE THE SEAMS between near-optimal stages:
#   - the K/V GEMM writes a compact (sum_M, 2E) buffer that the epilogue then
#     re-copies into the core's concatenated (B, S_total, E) layout (+ bias
#     slots + mask build, ~26 small ops) — it could write there directly;
#   - the Q projection materializes torch.stack + einsum before the core
#     reads it — it could be fused into the core's Q read or one kernel.
#
# Real v3-q3a serving shapes (same provenance as dirs #1/#2):
#   F = 13, B = 20, Lq = 500, E = 32, H = 2 (D = 16)
#   S_i  = [50, 50, 30, 10, 10, 10, 30, 5, 5, 10, 30, 10, 30]
#   k_i  = [40, 37, 58, 54, 54, 58, 58, 54, 54, 59, 71, 71, 71] (max_kv 71)
#   add_bias_kv + add_zero_attn -> S_ext_i = S_i + 2, S_total = sum = 306
#
# Numerics contract (as served, bf16 I/O): all matmuls take native-dtype
# operands with fp32 accumulation (ieee, no TF32); softmax and P.V in fp32;
# outputs cast back to the I/O dtype. Every attention row has >= 1
# attendable key (the appended slots), keeping the softmax finite.
#
# Input order (49 tensors): 13x q_i (B, Lq, E); 13x kv_i (B, S_i, k_i);
# 13x pad_i (B, S_i) bool (True = ignore); then q_weight (F, E, E),
# q_bias (F, E), k_weight (F, max_kv, E), v_weight (F, max_kv, E),
# k_bias (F, E), v_bias (F, E), out_weight (F, E, E), out_bias (F, E),
# bias_k (F, E), bias_v (F, E). Output: (F, B, Lq, E).

from typing import List

import torch
import torch.nn as nn

NUM_FEATURES = 13
NUM_REQUESTS = 20  # B
QUERY_LEN = 500  # Lq
EMBED_DIM = 32  # E
NUM_HEADS = 2  # H; D = E // H
KV_LENS = [50, 50, 30, 10, 10, 10, 30, 5, 5, 10, 30, 10, 30]  # S_i (raw)
KV_DIMS = [40, 37, 58, 54, 54, 58, 58, 54, 54, 59, 71, 71, 71]  # k_i
MAX_KV = max(KV_DIMS)
SCALE = 1.0 / ((EMBED_DIM // NUM_HEADS) ** 0.5)


class Model(nn.Module):
    """Eager reference for the full batched-MHA forward (fp32 math mirroring
    the served kernel numerics; bf16 values upcast exactly into fp32)."""

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        f, e, h = NUM_FEATURES, EMBED_DIM, NUM_HEADS
        d = e // h
        assert len(tensors) == 3 * f + 10, f"expected {3 * f + 10} tensors"
        q_list = tensors[0:f]
        kv_list = tensors[f : 2 * f]
        pad_list = tensors[2 * f : 3 * f]
        (q_w, q_b, k_w, v_w, k_b, v_b, out_w, out_b, bias_k, bias_v) = tensors[3 * f :]

        neg = torch.finfo(torch.float32).min
        # Serving stores each stage's output in the I/O dtype (each kernel
        # computes in fp32 but writes bf16), so the reference rounds at the
        # same stage boundaries. A candidate that fuses a seam and carries
        # fp32 across it only gets MORE accurate — still within tolerance.
        sd = tensors[0].dtype  # stage/storage dtype
        outs: List[torch.Tensor] = []
        for i in range(f):
            b, lq, _ = q_list[i].shape
            s, kdim = kv_list[i].shape[1], kv_list[i].shape[2]

            # Q projection in the storage dtype, exactly as served (torch bf16
            # matmul accumulates fp32 internally, rounds the matmul output,
            # then the bias add rounds again).
            q = torch.matmul(q_list[i], q_w[i]) + q_b[i]
            x = kv_list[i].reshape(b * s, kdim).float()
            k = (torch.matmul(x, k_w[i, :kdim].float()) + k_b[i].float()).to(sd).reshape(b, s, e)
            v = (torch.matmul(x, v_w[i, :kdim].float()) + v_b[i].float()).to(sd).reshape(b, s, e)

            # epilogue: bias_kv slot + zero_attn slot (stored in sd), additive mask
            k = torch.cat([k, bias_k[i].to(sd).expand(b, 1, e), torch.zeros(b, 1, e, device=k.device, dtype=sd)], dim=1)
            v = torch.cat([v, bias_v[i].to(sd).expand(b, 1, e), torch.zeros(b, 1, e, device=v.device, dtype=sd)], dim=1)
            add = torch.zeros(b, s + 2, device=k.device, dtype=torch.float32)
            add[:, :s] = torch.where(pad_list[i], neg, 0.0)

            # attention core per head (fp32 scores/softmax/P.V, output stored in sd)
            qh = q.view(b, lq, h, d).transpose(1, 2).float()
            kh = k.view(b, s + 2, h, d).transpose(1, 2).float()
            vh = v.view(b, s + 2, h, d).transpose(1, 2).float()
            scores = torch.matmul(qh, kh.transpose(-1, -2)) * SCALE + add[:, None, None, :]
            o = torch.matmul(scores.softmax(dim=-1), vh)  # (b, h, lq, d) fp32
            merged = o.transpose(1, 2).reshape(b, lq, e).to(sd)

            # output projection in the storage dtype, exactly as served
            outs.append(torch.matmul(merged, out_w[i]) + out_b[i])

        return torch.stack(outs, dim=0)  # (F, B, Lq, E)


def get_inputs():
    torch.manual_seed(0)
    f, b, lq, e = NUM_FEATURES, NUM_REQUESTS, QUERY_LEN, EMBED_DIM
    qs = [torch.randn(b, lq, e) for _ in range(f)]
    kvs = [torch.randn(b, s, k) for s, k in zip(KV_LENS, KV_DIMS)]
    pads = []
    for s in KV_LENS:
        valid = torch.randint(1, s + 1, (b, 1))
        pads.append(torch.arange(s).unsqueeze(0) >= valid)  # True = ignore
    q_w = torch.randn(f, e, e) * (e**-0.5)
    q_b = torch.randn(f, e) * 0.02
    k_w = torch.randn(f, MAX_KV, e) * (MAX_KV**-0.5)
    v_w = torch.randn(f, MAX_KV, e) * (MAX_KV**-0.5)
    for i, k in enumerate(KV_DIMS):
        k_w[i, k:, :] = 0.0
        v_w[i, k:, :] = 0.0
    k_b = torch.randn(f, e) * 0.02
    v_b = torch.randn(f, e) * 0.02
    out_w = torch.randn(f, e, e) * (e**-0.5)
    out_b = torch.randn(f, e) * 0.02
    bias_k = torch.randn(f, e) * 0.02
    bias_v = torch.randn(f, e) * 0.02
    return qs + kvs + pads + [q_w, q_b, k_w, v_w, k_b, v_b, out_w, out_b, bias_k, bias_v]


def get_init_inputs():
    return []
