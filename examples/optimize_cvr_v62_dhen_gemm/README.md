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
