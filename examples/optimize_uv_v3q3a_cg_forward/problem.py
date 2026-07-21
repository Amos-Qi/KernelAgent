# Unity user-value model v3-q3a-cg — full batched-MHA forward under
# CUDA-GRAPH serving constraints.
#
# Kernel dir #4. Same computation as optimize_uv_v3q3a_full_forward, but for
# the CUDA-graph serving variant (v3_q3a_cg): the front of the model is
# captured once per request-count bucket and REPLAYED, so the kernel must be
# **capture-safe** at **fixed bucket shapes**:
#   - B is the request bucket (24) and Lq is max_candidates_bucket (1024) —
#     no dynamic shapes, no data-dependent host control flow;
#   - NO per-call host->device transfers (torch.tensor(list, device=cuda) of
#     offsets is illegal under capture): every index/offset array is
#     PRECOMPUTED and passed in as a device-resident input tensor;
#   - per-call device-side ops (cat/stack/elementwise) and torch.empty
#     allocations are fine (graph memory pool), but fewer is better — replay
#     cost is the sum of captured work.
# test.py enforces the contract: after the numeric check it captures
# kernel_function in a torch.cuda.CUDAGraph, mutates the input CONTENTS in
# place, replays, and compares against an eager recompute. Candidates that do
# host-side work in the hot path fail capture and are rejected.
#
# Shapes (CG bucket profile; S_i/k_i provenance as dirs #1-#3):
#   F = 13, B = 24 (request bucket), Lq = 1024 (max_candidates_bucket)
#   E = 32, H = 2 (D = 16)
#   S_i  = [50, 50, 30, 10, 10, 10, 30, 5, 5, 10, 30, 10, 30]
#   k_i  = [40, 37, 58, 54, 54, 58, 58, 54, 54, 59, 71, 71, 71] (max_kv 71)
#   add_bias_kv + add_zero_attn -> S_ext_i = S_i + 2
#
# Numerics contract (as served, bf16 I/O): native-dtype matmul operands with
# fp32 accumulation (ieee, no TF32); softmax and P.V in fp32; each stage's
# output stored in the I/O dtype (the reference rounds at the same stage
# boundaries serving does, including the double-rounded projection bias adds).
#
# Input order (59 tensors):
#   13x q_i (B, Lq, E); 13x kv_i (B, S_i, k_i); 13x pad_i (B, S_i) bool;
#   q_weight (F,E,E), q_bias (F,E), k_weight (F,max_kv,E), v_weight
#   (F,max_kv,E), k_bias (F,E), v_bias (F,E), out_weight (F,E,E),
#   out_bias (F,E), bias_k (F,E), bias_v (F,E);
#   then the PRECOMPUTED int32 index tensors (device-resident, replay-static):
#   m_sizes (F,)   = B*S_i rows per feature in the ragged GEMM
#   k_sizes (F,)   = k_i
#   m_offsets (F,) = exclusive cumsum of m_sizes (row offsets)
#   x_offsets (F,) = exclusive cumsum of B*S_i*k_i (element offsets)
#   s_sizes (F,)   = S_i
# The reference ignores the index tensors (they are derivable from shapes);
# they exist so capture-safe kernels never build them on the host per call.
# Output: (F, B, Lq, E).

from typing import List

import torch
import torch.nn as nn

NUM_FEATURES = 13
NUM_REQUESTS = 24  # B: CG request bucket
QUERY_LEN = 1024  # Lq: max_candidates_bucket
EMBED_DIM = 32  # E
NUM_HEADS = 2  # H; D = E // H
KV_LENS = [50, 50, 30, 10, 10, 10, 30, 5, 5, 10, 30, 10, 30]  # S_i (raw)
KV_DIMS = [40, 37, 58, 54, 54, 58, 58, 54, 54, 59, 71, 71, 71]  # k_i
MAX_KV = max(KV_DIMS)
SCALE = 1.0 / ((EMBED_DIM // NUM_HEADS) ** 0.5)


class Model(nn.Module):
    """Eager reference (fp32 math mirroring served kernel numerics with
    storage-dtype rounding at each stage boundary). Ignores the trailing
    index tensors — they are an interface contract for capture-safe kernels,
    not part of the math."""

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        f, e, h = NUM_FEATURES, EMBED_DIM, NUM_HEADS
        d = e // h
        assert len(tensors) == 3 * f + 10 + 5, f"expected {3 * f + 15} tensors"
        q_list = tensors[0:f]
        kv_list = tensors[f : 2 * f]
        pad_list = tensors[2 * f : 3 * f]
        (q_w, q_b, k_w, v_w, k_b, v_b, out_w, out_b, bias_k, bias_v) = tensors[3 * f : 3 * f + 10]
        # tensors[3f+10 : 3f+15] are the precomputed index tensors — unused here.

        neg = torch.finfo(torch.float32).min
        sd = tensors[0].dtype  # stage/storage dtype
        outs: List[torch.Tensor] = []
        for i in range(f):
            b, lq, _ = q_list[i].shape
            s, kdim = kv_list[i].shape[1], kv_list[i].shape[2]

            # Q projection in the storage dtype, exactly as served (double
            # rounding: matmul output, then bias add).
            q = torch.matmul(q_list[i], q_w[i]) + q_b[i]

            x = kv_list[i].reshape(b * s, kdim).float()
            k = (torch.matmul(x, k_w[i, :kdim].float()) + k_b[i].float()).to(sd).reshape(b, s, e)
            v = (torch.matmul(x, v_w[i, :kdim].float()) + v_b[i].float()).to(sd).reshape(b, s, e)

            k = torch.cat([k, bias_k[i].to(sd).expand(b, 1, e), torch.zeros(b, 1, e, device=k.device, dtype=sd)], dim=1)
            v = torch.cat([v, bias_v[i].to(sd).expand(b, 1, e), torch.zeros(b, 1, e, device=v.device, dtype=sd)], dim=1)
            add = torch.zeros(b, s + 2, device=k.device, dtype=torch.float32)
            add[:, :s] = torch.where(pad_list[i], neg, 0.0)

            qh = q.view(b, lq, h, d).transpose(1, 2).float()
            kh = k.view(b, s + 2, h, d).transpose(1, 2).float()
            vh = v.view(b, s + 2, h, d).transpose(1, 2).float()
            scores = torch.matmul(qh, kh.transpose(-1, -2)) * SCALE + add[:, None, None, :]
            o = torch.matmul(scores.softmax(dim=-1), vh)  # fp32
            merged = o.transpose(1, 2).reshape(b, lq, e).to(sd)

            outs.append(torch.matmul(merged, out_w[i]) + out_b[i])

        return torch.stack(outs, dim=0)  # (F, B, Lq, E)


def make_index_tensors(batch: int):
    """Precomputed, replay-static index tensors for the ragged pipeline."""
    m_list = [batch * s for s in KV_LENS]
    m_offs, x_offs = [0], [0]
    for i in range(NUM_FEATURES - 1):
        m_offs.append(m_offs[-1] + m_list[i])
        x_offs.append(x_offs[-1] + m_list[i] * KV_DIMS[i])
    return [
        torch.tensor(m_list, dtype=torch.int32),
        torch.tensor(KV_DIMS, dtype=torch.int32),
        torch.tensor(m_offs, dtype=torch.int32),
        torch.tensor(x_offs, dtype=torch.int32),
        torch.tensor(KV_LENS, dtype=torch.int32),
    ]


def get_inputs(seed: int = 0):
    torch.manual_seed(seed)
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
    weights = [q_w, q_b, k_w, v_w, k_b, v_b, out_w, out_b, bias_k, bias_v]
    return qs + kvs + pads + weights + make_index_tensors(b)


def get_init_inputs():
    return []
