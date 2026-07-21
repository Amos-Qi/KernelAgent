# UV v3-q3a-cg — full batched-MHA forward under CUDA-graph constraints (kernel dir #4)

Same computation as `optimize_uv_v3q3a_full_forward`, retargeted at the
**CUDA-graph serving variant** (`v3_q3a_cg`): the model front is captured
once per request bucket and replayed per request, so kernels here must be
**capture-safe at fixed bucket shapes**. Seeded from the verified dir-#3
winner (ragged GEMM → attention core reading the ragged buffer directly →
Triton E→E projections) with its per-call host→device offset builds hoisted
into precomputed device-resident inputs.

## What "capture-safe" means (enforced by test.py gate 2)

- NO per-call host→device transfers: `torch.tensor([...], device="cuda")`
  inside the hot path is illegal under `torch.cuda.graph` capture. All
  index/offset arrays arrive as inputs (see `make_index_tensors`).
- NO data-dependent host control flow; shapes and grids are static
  (B = 24 request bucket, Lq = 1024 `max_candidates_bucket`).
- Device-side ops (stack/cat/elementwise) and `torch.empty` allocations are
  capture-legal (graph memory pool) — but every captured op replays every
  request, so REMOVING them is the optimization: e.g. read the 13 per-feature
  q/kv/pad tensors in-kernel instead of cat/stacking them per call, fold the
  mask conversion into the attention kernel, fuse the projections.
- test.py captures `kernel_function` in a CUDA graph, overwrites the input
  contents in place, replays, and checks the output against an eager
  recompute — candidates that fail capture or replay stale state are
  rejected regardless of their numeric parity in gate 1.

## Shapes (CG bucket profile; S/kv provenance as dirs #1–#3)

F=13, B=24, Lq=1024, E=32, H=2; S_i = [50,50,30,10,10,10,30,5,5,10,30,10,30];
kv_dims = [40,37,58,54,54,58,58,54,54,59,71,71,71]; S_ext = S_i+2. bf16 I/O.
59 input tensors: 13 q, 13 kv, 13 bool masks, 10 weights, 5 int32 index
tensors (replay-static). Output (F, B, Lq, E).

Numerics contract (do not weaken): native-dtype matmul operands, fp32
accumulate (ieee, no TF32), fp32 softmax and P·V, stage outputs stored in the
I/O dtype (projections keep the double-rounded bias add).

## Running on the VM

```bash
cd ~/KernelAgent && git pull --ff-only origin v3-q3a-opt
source .venv/bin/activate
cd examples/optimize_uv_v3q3a_cg_forward && python test.py && cd ..   # must PASS (both gates)

nohup python run_opt_manager.py --kernel-dir optimize_uv_v3q3a_cg_forward \
    --strategy beam_search_uv --max-rounds 5 > opt_run4.log 2>&1 &
```

Interleaved-ranking fix applies. Verify any winner with a strict A/B AND
rerun this test.py standalone (the capture gate) before integration; at
integration time the offset inputs map to persistent buffers owned by the
CG serving backend, and the triton_op wrappers follow the dir-#3 pattern.
