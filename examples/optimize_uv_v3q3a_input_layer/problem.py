# Unity user-value model v3-q3a — THE WHOLE INPUT-LAYER FORWARD (dir #8).
#
# One search over the full pipeline between the raw batch and the trunk input,
# replacing the segmented dirs: the fused feature front (dir #7, shipped), the
# batched-MHA attention path (dirs #1-#4, shipped) INCLUDING the not-yet-fused
# K/V + query ASSEMBLY (seq-embedding gathers, scalar concat, padding masks —
# the 14%-efficiency to_copy/gt/split/stack families), and the EPNet chain
# (the 14%-efficiency log1p/div/expand mask + iap-len-max block, domain
# assembly, and the GateNU gate). Joint search enables the cross-stage fusions
# the segmented dirs cannot see (masks built inside gathers, Q-projection
# folded into the attention core's Q read, domain vector emitted by the front).
#
# Geometry is the POST-kalign SHIPPED configuration (see the UL branch
# qi/v3-q3a-cuda-graph deploy/ passes): trunk width 1322 padded to 1328 (six
# exact-zero columns), EPNet domain 134 padded to 144, gate 1472 -> 512 ->
# 1328. CG bucket shapes (B=24 requests x Lq=1024 candidates); every offset
# needed by a capture-safe kernel is precomputed and passed as a device input.
#
# VERIFY-before-integration caveats (README): the per-feature K/V split
# EMB_W/SCAL_W, the iap-len-max block composition (3 purchase features x
# (len, len/inst_len, 4 seq-max sum-scalars)), and the domain-embed widths are
# taken from config arithmetic (sums verified: attn 416 + front 906 = 1322;
# domain 13+18+63+40 = 134) — confirm the splits against the checkpoint before
# integrating a winner. The fusion STRUCTURE is split-agnostic.
#
# Numerics contract (as served, bf16 I/O): gathers exact; matmuls fp32-accum
# (ieee, no TF32); softmax and P.V fp32; stage outputs round to the I/O dtype
# at the same boundaries the eager modules round.
#
# ── OPTIMIZATION GUIDANCE (measured on this exact problem, RTX PRO 6000) ──
# The AOTI max-autotune reference compiles this whole graph to ~1.37 ms; the
# seed implementation times ~4.4 ms. NCU on the seed: underutilized-bound,
# ~25% SOL. The gap is (a) eager glue between kernel launches — per-feature
# `tbl[ids]` gathers, concats, mask building, padding copies — and (b) the
# generic `_lin_kernel` GEMM schedule. It is NOT kernel-internal math:
# schedule-only tweaks of the seed's Triton kernels measured 0.99–1.01x of
# the seed, i.e. wasted attempts. Only structural moves close a 3.2x gap:
#   1. Fold the K/V + query assembly (gathers, scalar concat, additive masks)
#      into the ragged attention kernels using the flat-buffer + per-feature
#      offset-tensor pattern already used by `_front_meta` (aoffs/voffs);
#      masks can be computed inside the kernel from lengths and never
#      materialized in global memory.
#   2. Specialize or fuse the `_lin_kernel` GEMMs: the query projection can
#      fold into the attention core's Q read; the (24576x1472)@(1472x512)
#      gate GEMM chain can carry its relu / sigmoid-scale / pad-write
#      epilogues instead of round-tripping intermediates through DRAM.
#   3. Merge `_ff_*` launches that sweep the same rows; emit the EPNet
#      domain vector during the front sweep instead of re-gathering.
#
# RAGGEDNESS WARNING: per-feature sizes are heterogeneous (KV_LENS, KV_DIMS,
# EMB_W, table heights all differ by feature). `torch.stack` across features
# FAILS (e.g. [40,32] vs [37,32]); do not try to rectangularize per-feature
# tensors. The layout for any restructuring is one flat device buffer plus
# per-feature offset tensors (see `_front_meta`) — which is also the only
# layout that stays CUDA-graph-capture-safe (no host-side shape decisions).

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

B = 24          # request bucket
LQ = 1024       # max_candidates bucket
ROWS = B * LQ
E = 32          # attention embed dim
H = 2           # attention heads
SCALE = 1.0 / ((E // H) ** 0.5)
KV_LENS = [50, 50, 30, 10, 10, 10, 30, 5, 5, 10, 30, 10, 30]
KV_DIMS = [40, 37, 58, 54, 54, 58, 58, 54, 54, 59, 71, 71, 71]
MAX_KV = max(KV_DIMS)
EMB_W = [40, 37, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40]   # VERIFY
SCAL_W = [k - e for k, e in zip(KV_DIMS, EMB_W)]
SEQ_TABLE_ROWS = [110000, 46000] + [60000] * 11                # VERIFY heights
NF = 13
ATTN_W = NF * E  # 416: attention block occupies cols [0, 416)

PURCHASE_IDX = (7, 8, 9)   # raw_uups_purchase_{attributed,uasdk,unattributed}
N_SUM_SCALARS = 4          # per purchase feature, seq-max'd into the domain  VERIFY
IAP_BLOCK = len(PURCHASE_IDX) * (2 + N_SUM_SCALARS)  # 18

# EPNet domain-embed features REUSE the front features' tables/ids by name.
EPNET_EMBED_FEATURES = (
    "platform", "gamer_id_scope", "geolocation_country", "device_type",
    "device_atlas_hardware_classification", "platform_os_version",
)
TARGET_STORE_W = 40
DOM_RAW = NF + IAP_BLOCK + 63 + TARGET_STORE_W  # 13+18+63+40 = 134
DOM_PAD = 144            # kalign: next multiple of 16
W_RAW = ATTN_W + 906     # 1322 (front widths sum = 906)
W_PAD = 1328             # kalign trunk pad
GATE_HIDDEN = 512
GAMMA = 2.0

# Front layout (from dir #7, cols offset by ATTN_W): (name, kind, table_rows,
# _, width, level, bag_len); cat order = feature_modules registration order.
LAYOUT = [
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
_c = ATTN_W
for _e in LAYOUT:
    COL_OFF.append(_c)
    _c += _e[4]
assert _c == W_RAW, _c


class Model(nn.Module):
    """Eager reference: AttentionInputLayer.forward as served, post-kalign."""

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        it = iter(tensors)
        expand_idx = next(it)
        sd = None

        # ---- attention path (13 features): assembly + batched MHA ----
        seq_ids, kv_list, pad_list, q_list = [], [], [], []
        for i in range(NF):
            ids = next(it)                      # (B, S_i) int64, 0 = pad
            scal = next(it)                     # (B, S_i, SCAL_W_i)
            tbl = next(it)                      # (T_i, EMB_W_i)
            qids = next(it)                     # (ROWS,) int64
            qp_w = next(it)                     # (EMB_W_i, E)
            qp_b = next(it)                     # (E,)
            sd = tbl.dtype
            emb = F.embedding(ids, tbl)         # (B, S_i, EMB_W_i)
            kv = torch.cat([emb, scal], dim=-1) if scal.shape[-1] else emb
            q = torch.addmm(qp_b, F.embedding(qids, tbl), qp_w)  # query_projection (nn.Linear: single round)
            seq_ids.append(ids)
            kv_list.append(kv)
            pad_list.append(ids == 0)
            q_list.append(q.view(B, LQ, E))

        (q_w, q_b, k_w, v_w, k_b, v_b, out_w, out_b, bias_k, bias_v) = (
            next(it), next(it), next(it), next(it), next(it),
            next(it), next(it), next(it), next(it), next(it),
        )
        for _ in range(5):
            next(it)  # precomputed index tensors: kernel interface, not math

        neg = torch.finfo(torch.float32).min
        attn_outs: List[torch.Tensor] = []
        for i in range(NF):
            b_, lq_ = B, LQ
            s, kdim = kv_list[i].shape[1], kv_list[i].shape[2]
            q = torch.matmul(q_list[i], q_w[i]) + q_b[i]
            x = kv_list[i].reshape(b_ * s, kdim).float()
            k = (torch.matmul(x, k_w[i, :kdim].float()) + k_b[i].float()).to(sd).reshape(b_, s, E)
            v = (torch.matmul(x, v_w[i, :kdim].float()) + v_b[i].float()).to(sd).reshape(b_, s, E)
            k = torch.cat([k, bias_k[i].to(sd).expand(b_, 1, E),
                           torch.zeros(b_, 1, E, device=k.device, dtype=sd)], dim=1)
            v = torch.cat([v, bias_v[i].to(sd).expand(b_, 1, E),
                           torch.zeros(b_, 1, E, device=v.device, dtype=sd)], dim=1)
            add = torch.zeros(b_, s + 2, device=k.device, dtype=torch.float32)
            add[:, :s] = torch.where(pad_list[i], neg, 0.0)
            qh = q.view(b_, lq_, H, E // H).transpose(1, 2).float()
            kh = k.view(b_, s + 2, H, E // H).transpose(1, 2).float()
            vh = v.view(b_, s + 2, H, E // H).transpose(1, 2).float()
            scores = torch.matmul(qh, kh.transpose(-1, -2)) * SCALE + add[:, None, None, :]
            o = torch.matmul(scores.softmax(dim=-1), vh)
            merged = o.transpose(1, 2).reshape(b_, lq_, E).to(sd)
            attn_outs.append((torch.matmul(merged, out_w[i]) + out_b[i]).reshape(ROWS, E))

        # ---- feature front (dir #7 semantics), cols offset by ATTN_W ----
        front_inputs = {}
        outs: List[torch.Tensor] = list(attn_outs)
        for name, kind, t_rows, _, width, level, bag_len in LAYOUT:
            if kind == "embed":
                idx, table = next(it), next(it)
                front_inputs[name] = (idx, table)
                e_ = F.embedding(idx, table)
            elif kind == "bag":
                idx, table = next(it), next(it)
                e_ = F.embedding_bag(idx, table, mode="mean", padding_idx=0)
            elif kind == "dense":
                x_, w_, b2_ = next(it), next(it), next(it)
                e_ = (x_.unsqueeze(-1) * w_.unsqueeze(0) + b2_.unsqueeze(0)).reshape(x_.shape[0], -1)
            else:
                raise AssertionError(kind)
            if level == "user":
                e_ = e_.index_select(0, expand_idx)
            outs.append(e_)
        x = torch.cat(outs, dim=1)                       # (ROWS, 1322)
        x = F.pad(x, (0, W_PAD - W_RAW))                 # kalign: -> 1328, zeros

        # ---- EPNet domain (serving levels; user-level pieces expand) ----
        frozen_tbl = next(it)                            # (110000, TARGET_STORE_W)
        gw1, gb1, gw2, gb2 = next(it), next(it), next(it), next(it)

        mask_cols = [torch.where(ids.ne(0).sum(-1, keepdim=True) > 0, 1.0, 0.0) for ids in seq_ids]
        inst_len = seq_ids[0].ne(0).sum(-1, keepdim=True).clamp(min=1)
        iap_cols: List[torch.Tensor] = []
        for pi in PURCHASE_IDX:
            slen = seq_ids[pi].ne(0).sum(-1, keepdim=True)
            iap_cols.append(slen.to(torch.float32))
            iap_cols.append(slen.to(torch.float32) / inst_len.to(torch.float32))
            scal = kv_list[pi][..., EMB_W[pi]:]
            for j in range(N_SUM_SCALARS):
                iap_cols.append(scal[..., j].max(dim=1).values.unsqueeze(-1).to(torch.float32))
        dom_b = torch.cat([torch.cat(mask_cols, dim=-1).to(torch.float32),
                           torch.cat(iap_cols, dim=-1)], dim=-1)      # (B, 31)
        dom_rows = [dom_b.index_select(0, expand_idx)]
        for name in EPNET_EMBED_FEATURES:
            idx, table = front_inputs[name]
            dom_rows.append(F.embedding(idx, table).index_select(0, expand_idx).float())
        t_idx, _ = front_inputs["target_store_id"]
        dom_rows.append(F.embedding(t_idx, frozen_tbl).float())        # (ROWS, 40)
        dom = torch.cat(dom_rows, dim=-1)
        assert dom.shape[1] == DOM_RAW, dom.shape
        dom = F.pad(dom, (0, DOM_PAD - DOM_RAW)).to(x.dtype)           # kalign

        # ---- GateNU (kalign dims): relu -> sigmoid -> *gamma -> hadamard ----
        h = torch.relu(torch.addmm(gb1, torch.cat([dom, x], dim=-1), gw1))
        l2 = torch.sigmoid(torch.addmm(gb2, h, gw2)) * GAMMA
        return l2 * x                                                   # (ROWS, 1328)


def get_inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    ts: List[torch.Tensor] = [torch.arange(B, dtype=torch.int64).repeat_interleave(LQ)]
    for i in range(NF):
        s, t_rows, ew, sw = KV_LENS[i], SEQ_TABLE_ROWS[i], EMB_W[i], SCAL_W[i]
        ids = torch.randint(1, t_rows, (B, s), generator=g, dtype=torch.int64)
        valid = torch.randint(1, s + 1, (B, 1), generator=g)
        ids[torch.arange(s).unsqueeze(0) >= valid] = 0                  # pad tail
        ts.append(ids)
        ts.append(torch.randn(B, s, sw, generator=g) * 0.05)
        ts.append(torch.randn(t_rows, ew, generator=g) * 0.05)
        ts.append(torch.randint(0, t_rows, (ROWS,), generator=g, dtype=torch.int64))
        ts.append(torch.randn(ew, E, generator=g) * (ew ** -0.5))
        ts.append(torch.randn(E, generator=g) * 0.02)
    f = NF
    ts += [torch.randn(f, E, E, generator=g) * (E ** -0.5),
           torch.randn(f, E, generator=g) * 0.02]
    k_w = torch.randn(f, MAX_KV, E, generator=g) * (MAX_KV ** -0.5)
    v_w = torch.randn(f, MAX_KV, E, generator=g) * (MAX_KV ** -0.5)
    for i, k in enumerate(KV_DIMS):
        k_w[i, k:, :] = 0.0
        v_w[i, k:, :] = 0.0
    ts += [k_w, v_w,
           torch.randn(f, E, generator=g) * 0.02, torch.randn(f, E, generator=g) * 0.02,
           torch.randn(f, E, E, generator=g) * (E ** -0.5), torch.randn(f, E, generator=g) * 0.02,
           torch.randn(f, E, generator=g) * 0.02, torch.randn(f, E, generator=g) * 0.02]
    m_list = [B * s for s in KV_LENS]
    m_offs, x_offs = [0], [0]
    for i in range(NF - 1):
        m_offs.append(m_offs[-1] + m_list[i])
        x_offs.append(x_offs[-1] + m_list[i] * KV_DIMS[i])
    ts += [torch.tensor(m_list, dtype=torch.int32), torch.tensor(KV_DIMS, dtype=torch.int32),
           torch.tensor(m_offs, dtype=torch.int32), torch.tensor(x_offs, dtype=torch.int32),
           torch.tensor(KV_LENS, dtype=torch.int32)]
    for name, kind, t_rows, _, width, level, bag_len in LAYOUT:
        n = B if level == "user" else ROWS
        if kind == "embed":
            ts.append(torch.randint(0, t_rows, (n,), generator=g, dtype=torch.int64))
            ts.append(torch.randn(t_rows, width, generator=g) * 0.05)
        elif kind == "bag":
            idx = torch.randint(0, t_rows, (n, bag_len), generator=g, dtype=torch.int64)
            pad = torch.rand(n, bag_len, generator=g) < 0.35
            idx[pad] = 0
            idx[: max(n // 8, 1)] = 0
            ts.append(idx)
            ts.append(torch.randn(t_rows, width, generator=g) * 0.05)
        elif kind == "dense":
            ts.append(torch.randn(n, 130, generator=g))
            ts.append(torch.randn(130, 3, generator=g) * 0.05)
            ts.append(torch.randn(130, 3, generator=g) * 0.02)
    ts.append(torch.randn(110000, TARGET_STORE_W, generator=g) * 0.05)  # frozen target-store
    gin = DOM_PAD + W_PAD
    ts += [torch.randn(gin, GATE_HIDDEN, generator=g) * (gin ** -0.5),
           torch.randn(GATE_HIDDEN, generator=g) * 0.02,
           torch.randn(GATE_HIDDEN, W_PAD, generator=g) * (GATE_HIDDEN ** -0.5),
           torch.randn(W_PAD, generator=g) * 0.02]
    return ts


def get_init_inputs():
    return []
