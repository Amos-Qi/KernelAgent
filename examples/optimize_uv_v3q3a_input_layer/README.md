# UV v3-q3a — WHOLE INPUT-LAYER forward (kernel dir #8)

One search over the full pipeline between the raw batch and the trunk input,
consolidating what used to be three segmented targets:

1. **Attention input assembly** — the 14.2%-efficiency `to_copy/gt/split/
   stack` families: per-feature seq-embedding gathers, scalar concat, padding
   masks, the flat additive-mask cat, and the query gather+projection.
2. **The EPNet chain** — the 14.1%-efficiency `log1p/div/expand/where`
   family: the 13-feature non-empty-seq mask, the iap-len-max block, domain
   embed assembly, frozen target-store gather, and the GateNU gate.
3. **Schedule re-tuning of the shipped kernels** — `_ff_*` (dir #7) and the
   ragged attention pipeline (dirs #1–#4) are the SEED here, so the search
   tunes their schedules jointly and can find cross-stage fusions the
   segmented dirs structurally could not see (masks built inside gathers,
   Q-projection folded into the attention core's Q read, the domain vector
   emitted by the front pass).

Applies to BOTH serving variants: the input layer is identical in prod
(dynamic export) and CG (bucketed export); kernels take runtime row counts,
and the integration point is the existing `triton_feature_front` pass growing
to own the whole layer. Harness shapes are the CG bucket (24×1024) like all
prior dirs; the online-shape re-tune (R≈2, Σ≈1800) is the known follow-up.

## Geometry (post-kalign SHIPPED configuration)

Trunk 1322 → **1328** (six exact-zero pad columns — the output invariant
`out[:, 1322:] == 0` is enforced by test.py); EPNet domain 134 → **144**;
gate 1472 → 512 → 1328 (γ=2, relu hidden). Attention: 13 features, E=32,
H=2, real S_i/k_i from dirs #1–#4. Width arithmetic verified: attn 416 +
front 906 = 1322; domain 13 + 18 + 63 + 40 = 134.

## VERIFY before integrating a winner

- The per-feature K/V **EMB_W / SCAL_W split** (40/…-mostly assumed) and the
  **seq table heights** — confirm against the deployed checkpoint.
- The **iap-len-max block composition**: 3 purchase features × (len,
  len/inst_len, 4 seq-max sum-scalars) = 18 — confirm the sum-scalar count
  and ordering against `_get_non_empty_seq_mask` / config.
- The **domain-embed widths** (63 across the six named features) and the
  frozen target-store width (40).
The fusion structure is split-agnostic; these only shape synthetic traffic
and column bookkeeping.

## Numerics gates

Gathers are exact copies; matmuls fp32-accumulate (ieee, no TF32); softmax
and P·V in fp32; stage outputs round at eager boundaries. Because the gate
mixes every column, parity is a cancellation-aware band (bf16 ~1 ulp scaled;
fp32 ulp-scale) plus two exact invariants (shape, zero pad columns), and the
dir-#4-style CUDA-graph capture/replay gate.

## Running (VM)

```bash
cd ~/KernelAgent && git pull --ff-only && source .venv/bin/activate
cd examples/optimize_uv_v3q3a_input_layer && python test.py && cd ..   # gates must PASS
nohup python run_opt_manager.py --kernel-dir optimize_uv_v3q3a_input_layer \
  --strategy beam_search_uv --max-rounds 6 > ~/opt_run8.log 2>&1 &
```

Reminder on reading results: the seed already contains the shipped fused
kernels, so the honest bar is the harness's AOTI max-autotune reference and,
ultimately, the integrated re-profile — not the ratio vs the eager reference.
