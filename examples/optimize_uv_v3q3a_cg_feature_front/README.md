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
- CAVEATS (verify before integration): (1) user-vs-candidate level for the
  preprocessor-DERIVED features (apptopia/sensor_tower/device_atlas/
  platform_os_version) is assigned by name rule (source/device=user,
  target=cand) — confirm against a real model-input batch; (2) EmbeddingBag
  padding_idx assumed 0; (3) bag fixed lengths from _seq_max_length.

Run (VM, from ~/KernelAgent):
  source .venv/bin/activate && cd examples
  python ../examples/optimize_uv_v3q3a_cg_feature_front/test.py   # gates
  python run_opt_manager.py --kernel-dir optimize_uv_v3q3a_cg_feature_front \
    --strategy beam_search_uv --max-rounds 5
