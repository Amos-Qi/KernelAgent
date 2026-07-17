# Unity user-value model v3-q3a — batched-MHA attention core.
#
# Reference problem for the fused attention core inside TritonBatchedMHA
# (vector-ai-unity-learner, src/unity_learner/deploy/triton_attention/):
# per attention feature, scores = (Q @ K^T) * scale, additive key-padding
# mask, softmax, @ V. The K/V projections and output projection are separate
# kernels and are NOT part of this problem.
#
# Real v3-q3a serving shapes (provenance in README.md):
#   F  = 13 attention feature groups        (config.json mha_features)
#   B  = 20 requests per batch              (AOT export default, bucket 16-32)
#   H  = 2 heads, E = 32, D = E/H = 16      (attn_heads / attn_embed_dim)
#   Lq = 500 padded candidates (pad-Q path) (request-batching sampler bound)
#   S  = per-feature KV length incl. the 2 always-attendable slots appended
#        by add_bias_kv/add_zero_attn: {52 x2, 32 x4, 12 x5, 7 x2}
#
# Inputs are the post-projection, post-head-split per-feature tensors:
#   q_i (B, H, Lq, D)  k_i/v_i (B, H, S_i, D)  bf16
#   pad_i (B, S_i) bool key-padding mask, True = ignore. The last 2 slots of
#   every row are False (the bias_kv/zero_attn slots), so every row has at
#   least one attendable key and the softmax is always finite.
#
# Served numerics, mirrored exactly here: Q.K^T products of bf16 values
# accumulated in fp32, mask-add + softmax in fp32, P.V in fp32, output cast
# to the I/O dtype.

from typing import List

import torch
import torch.nn as nn

NUM_FEATURES = 13
NUM_REQUESTS = 20  # B
NUM_HEADS = 2  # H
HEAD_DIM = 16  # D (attn_embed_dim 32 / 2 heads)
QUERY_LEN = 500  # Lq: padded max candidates per request (pad-Q path)

# Per-feature KV lengths, in config.json mha_features order, each equal to
# _seq_max_length(feature) + 2 appended slots (bias_kv + zero_attn):
#   installed_store_ids, ad_req_project_id, adrev_levelplay_v2,
#   adrev_oecpm_interstitial_v2, adrev_oecpm_rewarded_v2,
#   adrev_s2s_attributed_v2, adrev_s2s_unattributed_v2,
#   purchase_attributed_v3, purchase_uasdk_v3, purchase_unattributed_v3,
#   fs_gamer_levelplay_adrev, fs_gamer_mmp_s2s_adrev_attributed,
#   fs_gamer_mmp_s2s_adrev_unattributed
KV_LENS = [52, 52, 32, 12, 12, 12, 32, 7, 7, 12, 32, 12, 32]

SCALE = 1.0 / (HEAD_DIM**0.5)


class Model(nn.Module):
    """Eager reference for the grouped attention core.

    forward(*tensors) takes the flat input list produced by get_inputs():
    F query tensors, then F key tensors, then F value tensors, then F bool
    key-padding masks. Returns one stacked (F, B, H, Lq, D) tensor in the
    query dtype so outputs can be compared with a single allclose.
    """

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        f = NUM_FEATURES
        assert len(tensors) == 4 * f, f"expected {4 * f} tensors, got {len(tensors)}"
        q_list = tensors[0:f]
        k_list = tensors[f : 2 * f]
        v_list = tensors[2 * f : 3 * f]
        pad_list = tensors[3 * f : 4 * f]

        outs: List[torch.Tensor] = []
        neg = torch.finfo(torch.float32).min
        for q, k, v, pad in zip(q_list, k_list, v_list, pad_list):
            # fp32 accumulation over bf16 values == tl.dot(..., out_dtype=fp32)
            scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * SCALE
            add = torch.zeros(
                pad.shape[0], 1, 1, pad.shape[1], device=pad.device, dtype=torch.float32
            ).masked_fill(pad[:, None, None, :], neg)
            probs = torch.softmax(scores + add, dim=-1)  # fp32
            out = torch.matmul(probs, v.float())  # P.V in fp32
            outs.append(out.to(q.dtype))
        return torch.stack(outs, dim=0)  # (F, B, H, Lq, D)


def get_inputs():
    torch.manual_seed(0)
    b, h, lq, d = NUM_REQUESTS, NUM_HEADS, QUERY_LEN, HEAD_DIM
    qs, ks, vs, pads = [], [], [], []
    for s in KV_LENS:
        qs.append(torch.randn(b, h, lq, d))
        ks.append(torch.randn(b, h, s, d))
        vs.append(torch.randn(b, h, s, d))
        # Per-request history length in [1, s-2]; the tail of the real slots
        # is padding. The 2 appended slots (bias_kv/zero_attn) always attend.
        valid = torch.randint(1, s - 1, (b, 1))
        pad = torch.arange(s - 2).unsqueeze(0) >= valid  # (b, s-2) True = ignore
        pad = torch.cat([pad, torch.zeros(b, 2, dtype=torch.bool)], dim=1)
        pads.append(pad)
    return qs + ks + vs + pads


def get_init_inputs():
    return []
