# UV v3-q3a — batched-MHA attention core (kernel dir #1)

KernelAgent kernel dir for the fused attention core inside `TritonBatchedMHA`
(`vector-ai-unity-learner`, `src/unity_learner/deploy/triton_attention/`).
This is the scores + key-padding mask + softmax + AV stage only; the K/V/Q/out
projections (`grouped_gemm.py` / `proj.py`) will be packaged as separate
kernel dirs.

## Files

- `problem.py` — eager reference (`Model`) + real-shape input generator.
- `input.py` — starting kernel: faithful port of the serving implementation
  (bucket-by-KV-length, one fused launch per distinct S, stack/scatter
  around it). The `torch.library.triton_op` wrapper is stripped for the
  optimization loop and re-applied at integration time.
- `test.py` — correctness gate (bf16, rtol/atol 1e-2). Standalone it tests
  `input.py`; inside the optimizer sandbox it tests the candidate `kernel.py`.

## Real serving shapes (and where they come from)

| Quantity | Value | Source (vector-ai-unity-learner) |
|---|---|---|
| F attention features | 13 | `experiment_repo/unified_user_value/v3_q3a/config.json` `mha_features` |
| B requests/batch | 20 | `deploy/utils.py` `build_torch_aot_model(num_requests=20)`; serving bucket 16–32 |
| Heads H / head dim D | 2 / 16 | config `attn_heads: 2`, `attn_embed_dim: 32` |
| Lq (padded candidates, pad-Q path) | 500 | `data/raw_data.py` `load_raw_request_batching_data(max_batch_size=500)` |
| KV lengths S (incl. +2 slots) | 52×2, 32×4, 12×5, 7×2 | `attention_input_layer.py` `_seq_max_length` + `add_bias_kv`/`add_zero_attn` |
| dtype | bf16 I/O, fp32 softmax | `deploy_config.deploy_precision: bf16`; kernel numerics |

If the true serving candidate bucket differs from 500 (it is the one number
above not read from a config), edit `QUERY_LEN` in `problem.py` before
optimizing — everything else follows the configs.

## Running on the VM (wweic-dev-g4)

```bash
# 0. Get this branch (repo already tracks the Amos-Qi fork)
cd ~/KernelAgent && git pull --ff-only origin v3-q3a-opt

# 1. Environment: the LLM endpoint + key must be in the .env that
#    examples/run_opt_manager.py loads (cwd .env via python-dotenv):
#      OPENAI_API_KEY=<unity-internal key>
#      OPENAI_BASE_URL=<unity-internal OpenAI-compatible endpoint>
grep -c OPENAI_API_KEY .env examples/.env 2>/dev/null

# 2. Sanity: GPU + NCU profiling permission
nvidia-smi -L
ncu --version && ncu --query-metrics >/dev/null 2>&1 && echo "ncu ok"
# If ncu reports ERR_NVGPUCTRPERM, the driver restricts perf counters:
# needs NVreg_RestrictProfilingToAdminUsers=0 (modprobe option + reboot)
# or run the optimizer under sudo with the venv python.

# 3. Correctness gate on the starting kernel (must print PASS)
cd examples/optimize_uv_v3q3a_attn_core && python test.py && cd ..

# 4. Optimization loop (from examples/, paths in configs are relative)
python run_opt_manager.py \
    --kernel-dir optimize_uv_v3q3a_attn_core \
    --strategy beam_search_uv \
    --max-rounds 5
```

Artifacts land in `optimize_uv_v3q3a_attn_core/`:

- `opt_manager_logs/beam_search_uv/` — per-round logs, NCU profiles,
  `program_db.json` (resumable optimization history).
- `optimized_kernel_beam_search_uv.py` — best kernel found.

Commit results on the VM (or copy them off) so the laptop can pull them.

## What the optimizer should find

The starting point is launch-overhead-bound: per forward it does 13 additive
mask builds, 13×3 stack gathers, 4 kernel launches (one per distinct S), and
4 scatters — for ~100 MFLOP of attention math. Obvious directions: fuse the
4 buckets into 1 launch (pad to S=52 or use a per-program feature index
table), fold the bool→additive mask conversion into the kernel, eliminate
the stack/scatter copies by reading per-feature pointers directly, and tune
`BLOCK_M`/`num_warps` for the (⌈Lq/BLOCK_M⌉ × n·B·H) grid. All candidates
must stay CUDA-graph capture-safe for the `v3_q3a_cg` variant: fixed shapes,
no host-side data-dependent control flow, no dynamic allocation patterns
that break capture.

## Integration back into serving

1. Wrap the winning kernel behind `torch.library.triton_op` following
   `deploy/triton_attention/op.py` (`unity_learner::attn_core`).
2. Re-point `grouped_attn_core` (or replace it, if the winner fuses the
   bucketing) in `deploy/triton_attention/grouped.py`.
3. Rebuild the `.pt2` via the AOT export path and validate with the `ul-cli`
   benchmark task; the swap itself stays registered in
   `deploy/optimize.py::swap_modules`.
