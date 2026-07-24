# UV v3-q3a — TRUNK FAN-OUT (kernel dir #9)

The per-component GEMM cluster consuming the trunk: 4 towers (silu DNNs) ->
COMBINED GateLayers everywhere (x*sigmoid(Wx+b); tower gates 384/192/256/128
square, group gates 659x659 / 593x593) -> tower head logits (13/4/6/7) ->
main head groups (659/593 -> 6/3) -> ZiLN/Poisson/BCE decodes -> 2-component
ensemble mean (+1.053 adrev calibration) -> 16-col nn_output stack.

TARGET = the SERVING envelope: MIG 1g.24gb slice (46 SMs, 448 GB/s), R=6
(rows 6*1024). Measured motivation (2026-07-24 slice profile of the optimized
prod artifact): align2 group-gate GEMMs 7.6% of GPU, split mm_mul_sigmoid
gate epilogues 4.5%, per-GEMM tower/head templates ~9% — with the trunk
re-read from DRAM per tower GEMM on a quarter-bandwidth slice.

Geometry is VERIFIED (trunk_tail_fixture.py + config + the measured GEMM
table: 659/593 derive exactly as tower_dims + head_cols sums).
VERIFY before integrating a winner: the group-input concat ORDER (assumed
[tower, its head logits] per input_towers sequence) and the serving-head
column offsets (payer=iap[6], ret=retention[3]) against DepositorModel
forward / the real head dict order.

Integration point: grow the `triton_trunk_tail` pass (it already owns
DepositorModel wrapping; towers/gates stayed eager there because Inductor
won on the FULL card — the slice inverts that).

## Running (VM, v3q3a KernelAgent worktree, 2g.48gb slice)

```bash
cd ~/ka-v3q3a && source .venv/bin/activate && source ~/campaigns/v3q3a-kernelagent.env
cd examples/optimize_uv_v3q3a_trunk_fanout && python test.py && cd ..   # gates must PASS
nohup python run_opt_manager.py --kernel-dir optimize_uv_v3q3a_trunk_fanout \
  --strategy beam_search_uv_mig --max-rounds 6 > ~/ka-v3q3a/examples/opt_run10_fanout.log 2>&1 &
```

Winner re-validation happens on the SPARE 1g slice (Dev 1 when idle) against
the incumbent kernels — do NOT freshly compile an AOTI reference there
(is_big_gpu < 68 SMs silently disables max-autotune templates).
