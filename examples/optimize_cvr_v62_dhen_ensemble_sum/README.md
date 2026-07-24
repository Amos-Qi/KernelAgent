# CVR v62 — DHEN layer-0 ensemble sum (kernel dir #2)

`out = sw0*proj_dlrm(t0) + sw1*proj_dcn(t1) + sw2*proj_mlp(t2)` with
`sw = softmax(WeightedSum.weight)` — folded into ONE K-segmented GEMM
(sum identity `Σ A_i@B_i = [A_0|A_1|A_2] @ [B_0;B_1;B_2]`), K segments
8256 + 8256 + 1024 = 17536, N = 2048, M = 4000 (serving rows), fp16.
Weights/biases pack once per weight-set (softmax scalars folded, host
cache holds source refs); the A activations are walked by three base
pointers — never concatenated.

## Why

Serving-slice profile (1g.24gb, 2026-07-24): proj_dlrm / proj_dcn are two
of the four identical 4000x8256@8256x2048 addmms at 2227.7 us each; the
compiler additionally emits stack/mul/sum eltwise passes for WeightedSum.
Dir #1 (audited winner: 1.68x vs incumbent pair on 1g, per-GEMM-equiv
1347 us) owns the other two big GEMMs (input_projection + MLP first
linear, shared input x). Integration composes them with ZERO overlap:

```
layer_out = SwishLayerNorm( dir1_pair.ip_half + dir2_sum )   # one add + LN
```

## Contracts

- fp16 I/O (assert inside kernel_function = the dtype-detection anchor).
- BAND parity gate (3e-3 + 3e-3*|ref|, zero outliers): single fp32
  accumulator vs serving's per-GEMM fp16 rounds is more accurate, not
  bit-identical. See test.py docstring.
- **BK must stay in {32, 64}**: 8256 % 128 != 0 and the segment loops are
  unmasked in K (dir #1's audit removed silently-wrong BK=128 configs;
  adding BK=128 back requires K masks).
- Autotune configs seeded from dir #1's audited winner family; both slices
  picked BM128/BN128/BK32/GROUP8/w4/s3 at the sibling shape.

## Running (VM, ~/KernelAgent, Dev 0 2g slice)

```bash
cd ~/KernelAgent && source .venv/bin/activate && source ~/campaigns/v3q3a-kernelagent.env
cd examples/optimize_cvr_v62_dhen_ensemble_sum && python test.py && cd ..   # gates must PASS
nohup python run_opt_manager.py --kernel-dir optimize_cvr_v62_dhen_ensemble_sum \
  --strategy beam_search_cvr --max-rounds 5 > opt_cvr_dir2.log 2>&1 &
```

Final winner re-validation on the 1g slice (Dev 2, cvr-profile.env)
against the eager three-addmm chain — the AOTI reference bar compiled by
the harness runs on the search slice only.

## RESULT (2026-07-25) — search complete, winner audited, IDEAL

| where | winner | seed | eager 3-GEMM + wsum chain |
|---|---|---|---|
| 2g search slice | **1.4944 ms** | 1.5169 | 2.2116 (early-session) |
| 1g serving slice | **2.9942 ms** | 3.2194 | 5.5053 |

1g verdict: **1.839x vs the serving chain** (~83% slice MFU). Honest
decomposition: the fusion itself is the payload (the seed already ran
3.22 ms); the search added a 2D super-tile swizzle + eviction split worth
1.075x on 1g. The manager-reported 1.127x-vs-seed was early-session DVFS
inflation (MIG clocks unlocked) — re-measured interleaved.
`winner_audited.py` is the integration source (ieee restored; shape-
fragility note on the floor-divided super-tile swizzle inside).

Combined with dir #1 (pair 4.524→2.694), the layer-0 big-GEMM cluster is
measured at **-4.34 ms GPU time** on the serving slice before integration
glue (eager add + SwishLayerNorm) is counted.

## FURTHER-OPTIMIZATION AUDIT (2026-07-25) — ceiling confirmed

Post-search hand exploration on the 1g serving slice, all variants pinned
to the winner's config with one knob changed, parity-gated, interleaved:

| variant | 1g time | vs winner |
|---|---|---|
| pinned winner (control) | 3.133 ms | = (validates pin) |
| eviction hints swapped (A resident / W stream) | 3.133 ms | 0.0% |
| no eviction hints at all | 3.134 ms | 0.0% |
| all evict_last (seed hints) | 3.134 ms | 0.0% |
| G_M2/G_N2 super-tiles (fits 1g 32MB L2) | 3.145 ms | -0.4% |
| BN=256 (BM128/w8/s3 — outside search space) | 3.407 ms | -8.7% |
| BN=256 (BM64/w4/s4) | 3.467 ms | -10.7% |
| BN=256/BK64 (s2/w8) | 4.388 ms | -40% |

Program DB shows the beam plateaued in round 1 (1.4875 -> 1.4856 over 5
rounds, one PTX family). Eviction hints are measurably inert here; the
kernel sits at ~83% slice MFU, TC-pipeline-bound, and every schedule
direction adjacent to (and beyond) the search space loses. Remaining
headroom (~17% to the naive compute floor) would need TMA / warp
specialization (CUTLASS-grade pipelining) outside this Triton toolchain's
demonstrated reach, or problem-boundary changes (dir #1 + dir #2 + LN
composition) which belong to integration.

## Adversarial re-audit (v3q3a session, 2026-07-24) — ceiling CONFIRMED

Three-arm probe on the 1g slice (control reproduced at 3.0213 vs the audited
2.9942; all arms bit-exact):
  winner (supertile swizzle)      3.0213 ms
  1D GROUP swizzle, masked        3.1764 ms  -> the supertile swizzle EARNS
                                               +5.1% over standard grouping
                                               (stronger than the grid showed)
  1D GROUP, EVEN_M two-launch     3.1890 ms  -> mask effect ~0.4% ~= zero
Mask despecialization refuted as a win axis; the swizzle is load-bearing.
