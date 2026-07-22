# Unity user-value model v3-q3a-cg — INPUT-LAYER FEATURE FRONT: the gather/
# assemble swarm between the raw batch and the 1322-wide trunk input.
#
# Kernel dir #7: HORIZONTAL FUSION of the non-attention feature modules. In the
# prod roofline this region is the 160-instance EmbeddingBag aten-fallback
# family (3.5us each), the to_copy/cat/expand/index_select pointwise families,
# and ~40% of GPU time spread across the sub-top-20 tail — all launch-floor +
# small-transfer bound. AOTInductor cannot fuse across features, cannot fuse
# aten::EmbeddingBag at all, and cannot fold the user->candidate expansion into
# consumers. The measured lesson from dirs #5/#6: only regions with this
# structure (many launches, skinny work, data movement) beat max_autotune.
#
# Geometry extracted from the deployed prod checkpoint (2026-07-22 state dict
# + preprocessor-input sample; see README): 62 features in cat order =
# 13 attention outputs (32 cols each; ALREADY fused via TritonBatchedMHA —
# passed through here as precomputed inputs) + 40 single-index embeddings +
# 8 fixed-length mean-mode EmbeddingBags (padding_idx 0) + one vectorized
# 130-scalar x 3-col dense projection. Total width == 1322 (verified).
#
# Serving semantics mirrored exactly:
#   * user-level features are computed at USERS rows then expanded to
#     candidate rows via index_select with the shared expand index
#     (ul_internal_expand_user_idx); candidate-level features compute at ROWS.
#   * EmbeddingBag(mode="mean", padding_idx=0): padded slots excluded from
#     numerator AND denominator; all-padding bag -> zeros.
#   * dense projection: out = x*W + b broadcast (torch: two bf16-rounding
#     elementwise kernels), reshape to 390 cols.
#   * bf16 I/O; gathers are exact copies; bag means accumulate fp32.
#
# CUDA-graph serving constraints (contract of dir #4): fixed bucket shapes
# (USERS=24 requests x CAND=1024 candidates), no per-call host->device
# transfers in steady state, capture+replay must match eager recompute.

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

USERS = 24
CAND = 1024
ROWS = USERS * CAND
ATTN_W = 32
DENSE_N = 130
DENSE_OUT = 3

# (name, kind, table_rows, _, width, level, bag_len) — cat order = feature_modules order
LAYOUT = [
    ('installed_store_ids', 'attn', 0, 0, 32, 'cand', 0),
    ('ad_req_project_id', 'attn', 0, 0, 32, 'cand', 0),
    ('raw_uups_adrev_levelplay_v2', 'attn', 0, 0, 32, 'cand', 0),
    ('raw_uups_adrev_oecpm_interstitial_v2', 'attn', 0, 0, 32, 'cand', 0),
    ('raw_uups_adrev_oecpm_rewarded_v2', 'attn', 0, 0, 32, 'cand', 0),
    ('raw_uups_adrev_s2s_attributed_v2', 'attn', 0, 0, 32, 'cand', 0),
    ('raw_uups_adrev_s2s_unattributed_v2', 'attn', 0, 0, 32, 'cand', 0),
    ('raw_uups_purchase_attributed_v3', 'attn', 0, 0, 32, 'cand', 0),
    ('raw_uups_purchase_uasdk_v3', 'attn', 0, 0, 32, 'cand', 0),
    ('raw_uups_purchase_unattributed_v3', 'attn', 0, 0, 32, 'cand', 0),
    ('feature_store_gamer_levelplay_adrev', 'attn', 0, 0, 32, 'cand', 0),
    ('feature_store_gamer_mmp_s2s_adrev_attributed', 'attn', 0, 0, 32, 'cand', 0),
    ('feature_store_gamer_mmp_s2s_adrev_unattributed', 'attn', 0, 0, 32, 'cand', 0),
    ('ad_format', 'embed', 10, 0, 5, 'user', 0),
    ('audience_id', 'embed', 46000, 0, 29, 'cand', 0),
    ('gamer_id_scope', 'embed', 10, 0, 5, 'user', 0),
    ('geolocation_country', 'embed', 300, 0, 8, 'user', 0),
    ('platform', 'embed', 10, 0, 5, 'user', 0),
    ('publisher_developer_id', 'embed', 40000, 0, 28, 'user', 0),
    ('publisher_game_id', 'embed', 56000, 0, 30, 'user', 0),
    ('publisher_store_id', 'embed', 110000, 0, 40, 'user', 0),
    ('target_developer_id', 'embed', 40000, 0, 28, 'cand', 0),
    ('target_game_id', 'embed', 56000, 0, 30, 'cand', 0),
    ('target_store_id', 'embed', 110000, 0, 40, 'cand', 0),
    ('device_connection_type', 'embed', 10, 0, 5, 'user', 0),
    ('device_orientation', 'embed', 10, 0, 5, 'user', 0),
    ('device_type', 'embed', 67000, 0, 32, 'user', 0),
    ('apptopia_storeinfo_source_category_id', 'embed', 50, 0, 5, 'user', 0),
    ('apptopia_storeinfo_source_cross_store_app_id', 'embed', 250000, 0, 20, 'user', 0),
    ('apptopia_storeinfo_source_offers_in_app_purchases', 'embed', 10, 0, 5, 'user', 0),
    ('apptopia_storeinfo_source_subcategory_id', 'embed', 200, 0, 7, 'user', 0),
    ('apptopia_storeinfo_target_category_id', 'embed', 50, 0, 5, 'cand', 0),
    ('apptopia_storeinfo_target_cross_store_app_id', 'embed', 250000, 0, 20, 'cand', 0),
    ('apptopia_storeinfo_target_offers_in_app_purchases', 'embed', 10, 0, 5, 'cand', 0),
    ('apptopia_storeinfo_target_subcategory_id', 'embed', 200, 0, 7, 'cand', 0),
    ('device_atlas_device_vendor', 'embed', 5000, 0, 16, 'user', 0),
    ('device_atlas_hardware_classification', 'embed', 20, 0, 5, 'user', 0),
    ('device_atlas_highest_cellular_generation', 'embed', 20, 0, 5, 'user', 0),
    ('device_atlas_os_vendor', 'embed', 50, 0, 5, 'user', 0),
    ('sensor_tower_source_game_class', 'embed', 14, 0, 5, 'user', 0),
    ('sensor_tower_source_game_genre', 'embed', 27, 0, 5, 'user', 0),
    ('sensor_tower_source_game_subcategory', 'embed', 35, 0, 5, 'user', 0),
    ('sensor_tower_source_game_subgenre', 'embed', 150, 0, 6, 'user', 0),
    ('sensor_tower_source_game_theme', 'embed', 188, 0, 7, 'user', 0),
    ('sensor_tower_source_primary_category', 'embed', 85, 0, 6, 'user', 0),
    ('sensor_tower_target_game_class', 'embed', 14, 0, 5, 'cand', 0),
    ('sensor_tower_target_game_genre', 'embed', 27, 0, 5, 'cand', 0),
    ('sensor_tower_target_game_monetization_model', 'embed', 15, 0, 5, 'cand', 0),
    ('sensor_tower_target_game_subcategory', 'embed', 35, 0, 5, 'cand', 0),
    ('sensor_tower_target_game_subgenre', 'embed', 150, 0, 6, 'cand', 0),
    ('sensor_tower_target_game_theme', 'embed', 188, 0, 7, 'cand', 0),
    ('sensor_tower_target_primary_category', 'embed', 85, 0, 6, 'cand', 0),
    ('platform_os_version', 'embed', 400, 0, 8, 'user', 0),
    ('installed_store_ids_channel', 'bag', 15, 0, 5, 'user', 50),
    ('raw_uups_adrev_levelplay_v2_channel', 'bag', 15, 0, 5, 'user', 30),
    ('raw_uups_adrev_oecpm_interstitial_v2_channel', 'bag', 15, 0, 5, 'user', 10),
    ('raw_uups_adrev_oecpm_rewarded_v2_channel', 'bag', 15, 0, 5, 'user', 10),
    ('raw_uups_adrev_s2s_attributed_v2_channel', 'bag', 15, 0, 5, 'user', 10),
    ('raw_uups_adrev_s2s_unattributed_v2_channel', 'bag', 15, 0, 5, 'user', 30),
    ('raw_uups_purchase_attributed_v3_channel', 'bag', 15, 0, 5, 'user', 5),
    ('raw_uups_purchase_uasdk_v3_channel', 'bag', 15, 0, 5, 'user', 5),
    ('dense_features', 'dense', 130, 0, 390, 'cand', 0),
]


COL_OFF = []
_c = 0
for _e in LAYOUT:
    COL_OFF.append(_c)
    _c += _e[4]
TOTAL_W = _c
assert TOTAL_W == 1322


class Model(nn.Module):
    """Eager reference: the per-feature module sequence exactly as
    AttentionInputLayer.forward runs it (gather -> maybe_expand -> cat)."""

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        it = iter(tensors)
        expand_idx = next(it)  # (ROWS,) int64: candidate row -> user row
        outs: List[torch.Tensor] = []
        for name, kind, t_rows, _, width, level, bag_len in LAYOUT:
            if kind == "attn":
                outs.append(next(it))  # (ROWS, 32) precomputed BatchedMHA output
                continue
            if kind == "embed":
                idx, table = next(it), next(it)
                e = F.embedding(idx, table)  # exact row gather
            elif kind == "bag":
                idx, table = next(it), next(it)  # (USERS, L) int64, (rows, w)
                e = F.embedding_bag(idx, table, mode="mean", padding_idx=0)
            elif kind == "dense":
                x, w, b = next(it), next(it), next(it)
                e = (x.unsqueeze(-1) * w.unsqueeze(0) + b.unsqueeze(0)).reshape(x.shape[0], -1)
            else:
                raise AssertionError(kind)
            if level == "user":
                e = e.index_select(0, expand_idx)  # user -> candidate expansion
            outs.append(e)
        return torch.cat(outs, dim=1)  # (ROWS, 1322)


def get_inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    ts: List[torch.Tensor] = [torch.arange(USERS, dtype=torch.int64).repeat_interleave(CAND)]
    for name, kind, t_rows, _, width, level, bag_len in LAYOUT:
        n = USERS if level == "user" else ROWS
        if kind == "attn":
            ts.append(torch.randn(ROWS, ATTN_W, generator=g))
        elif kind == "embed":
            ts.append(torch.randint(0, t_rows, (n,), generator=g, dtype=torch.int64))
            ts.append(torch.randn(t_rows, width, generator=g) * 0.05)
        elif kind == "bag":
            idx = torch.randint(0, t_rows, (n, bag_len), generator=g, dtype=torch.int64)
            # ~35% padding incl. some all-padding bags (empty bag -> zeros path)
            pad = torch.rand(n, bag_len, generator=g) < 0.35
            idx[pad] = 0
            idx[: max(n // 8, 1)] = 0
            ts.append(idx)
            ts.append(torch.randn(t_rows, width, generator=g) * 0.05)
        elif kind == "dense":
            ts.append(torch.randn(n, DENSE_N, generator=g))
            ts.append(torch.randn(DENSE_N, DENSE_OUT, generator=g) * 0.05)
            ts.append(torch.randn(DENSE_N, DENSE_OUT, generator=g) * 0.02)
    return ts


def get_init_inputs():
    return []
