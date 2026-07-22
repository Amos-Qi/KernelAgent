# optimize_uv_v3q3a_cg_feature_front (kernel dir #7)

Horizontal fusion of the v3-q3a input-layer FEATURE FRONT: the 49 non-attention
feature gathers (36 single-index embeddings, 8 user-level mean-mode
EmbeddingBags, one 130-scalar x 3 dense projection) + the user->candidate
expansion + the 1322-wide concat, with the 13 fused-attention outputs passed
through as inputs. This is the biggest remaining non-max_autotune region in the
2026-07-22 prod roofline (aten EmbeddingBag fallbacks, to_copy/cat/expand
families, ~40% of GPU time below the top-20).

- Geometry: extracted from the deployed prod checkpoint state dict +
  preprocessor-input sample (real table heights/widths; total width == 1322,
  verified). Cat order = feature_modules registration order.
- Seed: 4 launches (pointer-table gathers with in-kernel expansion, bag means,
  dense projection, attention copies), ptr/meta tables cached on data_ptrs
  (capture-safe steady state).
- VERIFIED against the UL repo (2026-07-22): bags are 2D fixed-length at
  serving (`use_fixed_length_features = deploy_config.use_static_shape`,
  true in v3_q3a config:741) with `nn.EmbeddingBag(mode="mean",
  padding_idx=ZERO_PADDING_VALUE)` and ZERO_PADDING_VALUE == 0
  (constant.py:14) — caveat (2) resolved. Dense is the vectorized identity
  path (`continuous_projection_activation: "identity"`, state dict
  `dense_features.W/b (130, 3)`). Expansion is `index_select(0,
  ul_internal_expand_user_idx)` (maybe_expand_feature). Cat order matches
  the registration order (attn -> embed -> bag -> dense; the 13 mha_features
  names match this LAYOUT).
- CAVEATS (verify before integration): (1) user-vs-candidate level is
  assigned here by NAME RULE, but serving decides at RUNTIME by
  `feature.shape[0] == num_requests` — dump one real model-input batch and
  check every feature's row count (including `dense_features`, modeled cand
  here); (2) bag lengths for the 7 raw_uups_*_channel bags did NOT come
  from `_seq_max_length` (it defines only installed_store_ids=50 and
  returns -1 otherwise) — they came from a preprocessor sample; re-confirm.
  (3) publisher_/target_/installed_ store_id (and game_id, developer_id,
  cross_store_app_id) share JOINT tables in serving (`joint_feature_groups`)
  — the harness models separate tables, which UNDERSTATES cache reuse;
  integration must point both features at the one shared table.
- INTEGRATION DESIGN NOTE: the seed's raw-`data_ptr` pointer tables are
  harness-only. An AOTI-exported artifact loads in a fresh process where
  baked addresses dangle — the integration pass must switch to a single
  concatenated table ARENA + int64 row-offset metadata (export-safe
  constants), or pass the tables as real tensor args.

Run (VM, from ~/KernelAgent):
  source .venv/bin/activate && cd examples
  python ../examples/optimize_uv_v3q3a_cg_feature_front/test.py   # gates
  python run_opt_manager.py --kernel-dir optimize_uv_v3q3a_cg_feature_front \
    --strategy beam_search_uv --max-rounds 5
