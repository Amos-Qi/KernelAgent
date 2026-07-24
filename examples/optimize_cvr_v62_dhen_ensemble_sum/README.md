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
