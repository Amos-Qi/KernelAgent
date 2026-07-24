# optimize_cvr_v62_dhen_gemm (cvr dir #1)

The dominant cost of android_conversion.v62 serving: the fp16
(4000x8256)@(8256x2048) GEMM class (4 instances in DHEN layer 0; the
cutlass_80 f16 64x64 family = 31% of all GPU time on the serving slice).
Evidence: same shape autotuned to 395us / 73% MFU on the FULL card at build
time but runs 2228us / 53% on the 1g.24gb slice -> ~1.4x recoverable by a
slice-tuned schedule alone. This dir is the SHARED-INPUT pair
(input_projection + MLP first linear, both consume the same x), so the seed
fuses them into one x@[W_ip|W_mlp] GEMM (N=4096) with the relu epilogue on
the MLP column half.

- Numerics: fp16 I/O (marker: torch.float16), fp32 ieee accumulate, addmm
  single-round; input_projection has NO activation, MLP half has ReLU
  (assumption from the gemm_relu family — integration re-reads live modules).
- GOAL: minimize GPU kernel time on the SERVING slice (gpu_name in
  beam_search_cvr.yaml = the MIG 1g.24gb entry, 46 SMs).
- Known bar caveat: the harness AOTI reference defaults to the UV inductor
  configs; v62 additionally sets coordinate_descent_tuning/aggressive_fusion.
  max_autotune_gemm=true (the load-bearing flag) is in both — acceptable, but
  the in-profile 2228us incumbent is the honest serving bar.
- Final validation: re-time the winner on a 1g slice (search runs on the 2g).

Run (VM, ~/KernelAgent on cvr-v62-opt):
  source .venv/bin/activate && source ~/campaigns/v3q3a-kernelagent.env  # Dev0 2g
  cd examples && python ../examples/optimize_cvr_v62_dhen_gemm/test.py  # gates
  python run_opt_manager.py --kernel-dir optimize_cvr_v62_dhen_gemm \
    --strategy beam_search_cvr --max-rounds 5

## RESULT (2026-07-25) — search complete, winner audited, IDEAL

Beam search (5 rounds, rounds 4/4 except one 3/4) converged to the seed
structure + `@triton.autotune` over a block-schedule space. Audit found and
fixed a latent correctness bug (BK=128 configs with unmasked K at
8256 % 128 != 0 — silently wrong wherever a re-tune picks them) and
restored ieee input precision; see `winner_audited.py` (the integration
source).

| where | winner | seed | incumbent 2-GEMM | AOTI bar |
|---|---|---|---|---|
| 2g search slice | **1.2924 ms** | 1.8084 | — | 1.4918 |
| 1g serving slice | **2.6935 ms** | 3.5267 | 4.5239 | (not buildable) |

1g verdict: **1.680x vs incumbent**, per-GEMM-equivalent 1347 us (~87%
slice MFU) — beats the 1610 us full-card-parity target. Parity bit-exact
on both slices; autotune picks BM128/BN128/BK32/GROUP8/w4/s3 on both.
Follow-up region wired as dir #2: `optimize_cvr_v62_dhen_ensemble_sum`
(the three interaction projections as one K-segmented weighted-sum GEMM;
composes with this dir's pair at integration with zero overlap).

## FURTHER-OPTIMIZATION AUDIT (2026-07-25) — ceiling confirmed

Program DB: round-2 jump (1.4531 -> 1.2850, autotune adoption), flat
thereafter (1.2843 by r5, 0.05%). Hand check on the 1g slice: pinned
winner control 2.6904 ms reproduces the audited 2.6935; BN=256
(BM128/w8/s3, outside the search space) loses 17.5% (3.1615 ms). At ~87%
slice MFU the kernel is compute-bound (memory floor ~0.83 ms vs compute
floor 2.35 ms); the BM128/BN128/BK32 family is the verified optimum for
this shape family on the slice.
