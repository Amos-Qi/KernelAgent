# UV v3-q3a — ragged K/V projection grouped GEMM (kernel dir #2)

KernelAgent kernel dir for `project_kv` inside `TritonBatchedMHA`
(`vector-ai-unity-learner`, `deploy/triton_attention/{proj,grouped_gemm,
grouped_gemm_op}.py`): the single-launch grouped GEMM that projects 13 ragged
per-feature KV inputs through their K and V weights (fused, N = 2E). Kernel
dir #1 (`optimize_uv_v3q3a_attn_core`) covered the attention core; the
verified 5.5x winner for it is parked in that directory.

## Files

- `problem.py` — eager reference (`Model`) + real-shape input generator.
- `input.py` — starting kernel: faithful port of the serving implementation
  (per-call weight cats, `pack_grouped` zero-pad packing, one launch on
  padded shapes, output slicing). `triton_op` wrapper stripped for the loop.
- `test.py` — correctness gate (bf16, rtol/atol 1e-2). Standalone it tests
  `input.py`; in the optimizer sandbox it tests the candidate `kernel.py`.

## Real serving shapes (and where they come from)

| Quantity | Value | Source |
|---|---|---|
| F features | 13 | `v3_q3a/config.json` `mha_features` |
| B requests/batch | 20 | AOT export default; serving bucket 16–32 |
| E / N | 32 / 64 (K,V fused) | config `attn_embed_dim`; `proj.py` |
| S_i (pre bias_kv/zero_attn) | 50,50,30,10,10,10,30,5,5,10,30,10,30 | `_seq_max_length` |
| kv_dims k_i | 40,37,58,54,54,58,58,54,54,59,71,71,71 | **deployed checkpoint** `attn_module.k_proj_weight` shapes (prod artifact 2026-07-17, `unified-user-value-v3-q3a-10-model`) |
| rows M_i = B·S_i | Σ 5400, max 1000 | derived |
| dtype | bf16 I/O, fp32 accumulate (ieee) | `deploy_precision: bf16`; kernel |

## Why there's headroom

The served path pays, every forward: two weight cats, a ~1.9MB zero-fill +
26 gather copies (`pack_grouped`), a launch whose grid covers **max_M rows
for every feature** — Σ padded row-tiles ≈ 58% of the grid does nothing —
and 13 output slices + a cat. The math itself is ~39 MFLOP. Obvious
directions (mirroring the kernel-dir-#1 winner): ragged access via
per-feature pointer tables (no packing), a per-group M array so the grid
skips padded tiles, right-sized BLOCK_K per kv_dim bucket, and keeping
everything CUDA-graph capture-safe (fixed shapes, no data-dependent host
control flow).

## Running on the VM (wweic-dev-g4)

```bash
cd ~/KernelAgent && git pull --ff-only origin v3-q3a-opt
source .venv/bin/activate

# correctness gate — must print PASS
cd examples/optimize_uv_v3q3a_grouped_gemm && python test.py && cd ..

# optimization loop (thinking mode: effort=high + discipline block, all default)
python run_opt_manager.py \
    --kernel-dir optimize_uv_v3q3a_grouped_gemm \
    --strategy beam_search_uv \
    --max-rounds 5
```

LLM traces stream to `~/.kernelagent/llm_traces/`. Artifacts land in
`optimize_uv_v3q3a_grouped_gemm/opt_manager_logs/` and the winner in
`optimized_kernel_beam_search_uv.py`. Verify any winner with a strict
interleaved A/B against `input.py` before trusting the manager's ranking.

## Integration back into serving

1. Wrap the winner behind `torch.library.triton_op` following
   `grouped_gemm_op.py` (`unity_learner::grouped_gemm`).
2. Re-point `proj.py::project_kv` (or replace `pack_grouped` if the winner
   goes ragged/pointer-based — mind `torch.export(strict=True)`).
3. Rebuild the `.pt2`, validate with `ul-cli` benchmark.
