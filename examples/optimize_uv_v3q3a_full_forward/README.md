# UV v3-q3a — full batched-MHA forward (kernel dir #3)

The whole `TritonBatchedMHA` forward as one problem, **seeded from the
integrated winners of dirs #1 and #2** (fused attention core 5.5x + ragged
K/V GEMM 2.6x, both verified and shipped to the UL branch
`qi/v3q3a-triton-kernel-fusion`). Every stage is near-optimal in isolation;
the optimization target is the SEAMS:

1. Q projection materializes `torch.stack` + einsum before the core reads
   `(F, B, Lq, E)` — foldable into the core's Q read or one Triton kernel
   (the 13 q inputs are often views of one shared tensor in serving).
2. The ragged GEMM writes a compact `(sum_M, 2E)` buffer that ~26 small
   torch ops then re-copy into the core's concatenated `(B, S_total, E)`
   layout with bias_kv/zero_attn slots and the additive mask — the GEMM
   could emit that final layout directly.
3. The output projection einsum could fold into the core's store.

## Numerics contract (do not weaken)

Native-dtype matmul operands with fp32 accumulation and
`input_precision="ieee"` (no TF32); softmax and P·V in fp32; outputs cast to
the I/O dtype. Every attention row keeps >= 1 attendable key.

## Shapes (same provenance as dirs #1/#2 — configs + deployed checkpoint)

F=13, B=20, Lq=500, E=32, H=2; S_i = [50,50,30,10,10,10,30,5,5,10,30,10,30];
kv_dims = [40,37,58,54,54,58,58,54,54,59,71,71,71]; S_ext = S_i+2,
S_total = 306. bf16 I/O. 49 input tensors (13 q, 13 kv, 13 bool masks,
10 weight/bias tensors); output (F, B, Lq, E).

## Running on the VM

```bash
cd ~/KernelAgent && git pull --ff-only origin v3-q3a-opt
source .venv/bin/activate
cd examples/optimize_uv_v3q3a_full_forward && python test.py && cd ..   # must PASS

nohup python run_opt_manager.py --kernel-dir optimize_uv_v3q3a_full_forward \
    --strategy beam_search_uv --max-rounds 5 > opt_run3.log 2>&1 &
```

This run includes the interleaved-ranking fix (candidates are timed in
alternating same-session blocks against the incumbent; ranking uses the
session-invariant ratio), so the leaderboard is trustworthy — but still
verify the winner with a strict A/B and READ its diff before integrating:
the seams it may fuse carry the serving numerics contract, and export
constraints (torch.export-safe layouts, no data_ptr tricks) apply at
integration time as with dirs #1/#2.
