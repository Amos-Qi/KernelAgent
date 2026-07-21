# UV v3-q3a-cg — trunk tail: towers → heads → decodes → stack (kernel dir #5)

The **horizontal-fusion** problem the roofline has pointed at since day one:
the served trunk tail is dozens of tiny launches — 8 tower-layer GEMMs, 14
head GEMMs (3- and 1-column outputs!), and the 60-/27-callsite pointwise
decode swarm (`sigmoid/softplus/exp/where`) — every one "optimal for its
size" at 8–16% SOL, which is exactly why per-kernel optimization plateaued
at ~10%. The win here is structural: run heads × windows × ensemble
components as FEW WIDE KERNELS.

Fusion shape suggestions (the search should find better):
- The 14 head GEMMs per component share two input matrices (`iap_in`,
  `adrev_in`, plus two tower outputs) — pack all heads that share an input
  into ONE GEMM with concatenated output columns, decode in the epilogue.
- The decode swarm is pure per-row math over ≤ 16 columns — one kernel.
- Both ensemble components are independent until the final mean — batch them
  (stack the weight matrices; grid over components).
- The 16-col output stack + calibration + mean folds into the same epilogue.

## CUDA-graph contract (as dir #4)

Fixed shapes (rows = 24 × 1024), capture-safe, no per-call host work.
`test.py` gates numeric parity AND capture/replay correctness.

## Provenance and the two UNVERIFIED dims

Structure and decode math are exact (config.json `nn_config.model`,
`forward_trunk`, `ZilnHead.prob_value_pred`, branch `qi/partial-graph-split`).
Two dims were truncated in the config dump and are **placeholders to verify
before integration** (they do not change the fusion structure):
- `SKIP_DIM = 64` — check `input_layer.skip_output_size()`
- `retention` / `adrev_bce` tower dims `[192, 192]` — check
  `nn_config.model.towers`

## Running on the VM

```bash
cd ~/KernelAgent && git pull --ff-only origin v3-q3a-opt
source .venv/bin/activate
cd examples/optimize_uv_v3q3a_cg_trunk_heads && python test.py && cd ..   # must PASS

nohup python run_opt_manager.py --kernel-dir optimize_uv_v3q3a_cg_trunk_heads \
    --strategy beam_search_uv --max-rounds 5 > opt_run5.log 2>&1 &
```

Integration target: `forward_trunk` / `forward_flat` tail on the UL branch —
the same seam the partial-graph split's TRUNK wrapper calls, so a winner here
speeds the whole-cg replay AND the split's eager trunk identically. Verify
the winner with a strict A/B, read its diff (the numerics contract: fp32
decode math, bf16 stage storage, ieee accumulate), and re-run shadow parity
after integration as with dirs #1–#3.
