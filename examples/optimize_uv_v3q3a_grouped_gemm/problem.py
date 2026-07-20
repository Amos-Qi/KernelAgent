# Unity user-value model v3-q3a — ragged K/V projection (grouped GEMM).
#
# Reference problem for project_kv in TritonBatchedMHA
# (vector-ai-unity-learner, deploy/triton_attention/{proj,grouped_gemm}.py):
# for each of F attention features, project its ragged KV input
#   x_i (B, S_i, k_i)  @  [k_weight_i | v_weight_i] (k_i, 2E)  +  [k_bias|v_bias]
# K and V are fused into one grouped GEMM with shared output dim N = 2E.
#
# Real v3-q3a serving shapes (provenance in README.md; kv_dims read from the
# deployed checkpoint's attn_module.k_proj_weight shapes, 2026-07-17 artifact):
#   F = 13, B = 20, E = 32 (N = 2E = 64)
#   S_i  = [50, 50, 30, 10, 10, 10, 30, 5, 5, 10, 30, 10, 30]  (pre bias_kv/zero_attn)
#   k_i  = [40, 37, 58, 54, 54, 58, 58, 54, 54, 59, 71, 71, 71] (max_kv = 71)
#   rows M_i = B*S_i -> sum(M) = 5400, max_M = 1000
#
# Weights use BatchedMHA's stacked layout: k_weight/v_weight (F, max_kv, E)
# with rows >= k_i zero-padded, k_bias/v_bias (F, E). bf16 I/O; products of
# bf16 values accumulated in fp32 (ieee), bias added in fp32, output cast back.
#
# Output contract: one (sum(B*S_i), 2E) tensor — feature blocks in order, each
# block the (B*S_i, 2E) projection [K_i | V_i].

from typing import List

import torch
import torch.nn as nn

NUM_FEATURES = 13
NUM_REQUESTS = 20  # B
EMBED_DIM = 32  # E; grouped GEMM output N = 2E
KV_LENS = [50, 50, 30, 10, 10, 10, 30, 5, 5, 10, 30, 10, 30]  # S_i
KV_DIMS = [40, 37, 58, 54, 54, 58, 58, 54, 54, 59, 71, 71, 71]  # k_i
MAX_KV = max(KV_DIMS)


class Model(nn.Module):
    """Eager reference for the fused ragged K/V projection.

    forward(*tensors): F ragged inputs x_i (B, S_i, k_i), then k_weight
    (F, MAX_KV, E), v_weight (F, MAX_KV, E), k_bias (F, E), v_bias (F, E).
    Returns (sum(B*S_i), 2E) in the input dtype.
    """

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        f = NUM_FEATURES
        assert len(tensors) == f + 4, f"expected {f + 4} tensors, got {len(tensors)}"
        x_list = tensors[0:f]
        k_w, v_w, k_b, v_b = tensors[f : f + 4]

        outs: List[torch.Tensor] = []
        for i, x in enumerate(x_list):
            b, s, k = x.shape
            w = torch.cat([k_w[i, :k, :], v_w[i, :k, :]], dim=1)  # (k_i, 2E)
            bias = torch.cat([k_b[i], v_b[i]], dim=0)  # (2E,)
            # fp32 accumulation over bf16 values == tl.dot(..., out_dtype=fp32)
            y = torch.matmul(x.reshape(b * s, k).float(), w.float()) + bias.float()
            outs.append(y.to(x.dtype))
        return torch.cat(outs, dim=0)  # (sum(B*S_i), 2E)


def get_inputs():
    torch.manual_seed(0)
    b, e = NUM_REQUESTS, EMBED_DIM
    xs = [torch.randn(b, s, k) for s, k in zip(KV_LENS, KV_DIMS)]
    k_w = torch.randn(NUM_FEATURES, MAX_KV, e)
    v_w = torch.randn(NUM_FEATURES, MAX_KV, e)
    # BatchedMHA layout guarantee: rows beyond each feature's k_i are zero.
    for i, k in enumerate(KV_DIMS):
        k_w[i, k:, :] = 0.0
        v_w[i, k:, :] = 0.0
    k_b = torch.randn(NUM_FEATURES, e)
    v_b = torch.randn(NUM_FEATURES, e)
    return xs + [k_w, v_w, k_b, v_b]


def get_init_inputs():
    return []
