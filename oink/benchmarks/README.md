# Blackwell SM10x Benchmarks (KernelAgent-Oink vs Quack)

This folder contains SM10x (GB200 / GB300 / Blackwell) microbenchmarks for the
Oink CuTeDSL kernels, comparing against Quack’s SM100 kernels where Quack
provides an equivalent API.

## Prereqs

- GPU: **SM10x / Blackwell** (`torch.cuda.get_device_capability()[0] == 10`).
- Python deps in your environment:
  - `torch`
  - `nvidia-cutlass-dsl>=4.4.2` (CuTeDSL); optimized dense GEMM and MoE grouped-GEMM backends require `nvidia-cutlass-dsl>=4.5.0`
  - `cuda-python`
  - `triton` (only for `triton.testing.do_bench`)
  - `quack` / `quack-kernels` (optional; only needed for Oink-vs-Quack comparisons)

Recommended env vars:

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
# GB300 / SM103. Use the `a` suffix for GEMM/tcgen05 paths:
export CUTE_DSL_ARCH=sm_103a
# GB200/B200 / SM100 historical runs:
# export CUTE_DSL_ARCH=sm_100a
```

For the pinned GB300 / SM103 benchmark environment used by the current README
numbers:

```bash
conda create -y -n cute python=3.12
conda run -n cute python -m pip install --upgrade pip setuptools wheel packaging ninja
conda run -n cute python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu130 torch
conda run -n cute python -m pip install 'nvidia-cutlass-dsl==4.5.0' cuda-python triton matplotlib pytest pytest-cov
# Oink GEMM paths use CUTLASS DSL 4.5-style Blackwell/tcgen05 helpers.
conda run -n cute python -m pip install -e '.[bench]'
conda run -n cute python -m pip install 'git+https://github.com/Dao-AILab/quack.git'  # optional comparison baseline
```

## Shape suites

- **Quack-suite**: `(batch, seq) ∈ {1,4,8,16,32} × {8192,16384,32768,65536,131072}`,
  with `hidden = 4096` so `M = batch * seq`, `N = 4096`.
- **DeepSeek-V3-like (DSv3)**
  - RMSNorm / LayerNorm / Softmax: `M ∈ {4096, 16384, 65536}`, `N ∈ {6144, 7168, 8192}`
  - LayerNorm backward's `--dsv3` suite uses `N ∈ {6144, 8192}`; use `--dsv4` for the `N = 7168` hidden-state sweep.
  - Cross-entropy: `M ∈ {4096, 16384, 65536}`, `N ∈ {3072, 6144, 8192, 12288}`
- **DeepSeek-V4-Flash norm shapes (DSv4)** from `deepseek-ai/DeepSeek-V4-Flash/inference/model.py`
  - hidden-state RMSNorm / LayerNorm: `M ∈ {4096, 16384, 65536}`, `N = 7168`
  - q_lora RMSNorm: `M ∈ {4096, 16384, 65536}`, `N = 1536`
  - kv latent / per-head RMSNorm: `M ∈ {4096, 16384, 65536}`, `N = 512`

## Correctness gates

By default, each script runs a per-shape `torch.testing.assert_close` check vs a
**pure-PyTorch reference** **before** emitting timing numbers. When Quack is
available for that op/path, the script also validates Quack vs the *same*
reference (so speedups can’t come from looser numerics).

Disable with `--skip-verify` only for quick smoke tests. Do not use
`--skip-verify` for README or release performance numbers.

## Roofline reporting

Most benchmark JSONs include `*_hbm_frac` using `bench_utils.detect_hbm_peak_gbps()`.
That helper is a coarse fallback (`8000 GB/s` for SM10x) so old JSONs can be
compared consistently. For GB300/SM103 published results, use a measured roofline
run instead.

Current measured GB300 BF16 STREAM-like roof used in the README:

- **7.140 TB/s** (triad, `BLOCK=2048`, `warps=8`)
- 90% target: **6.426 TB/s**

Regenerate on the current machine:

```bash
conda run -n cute bash -lc 'PYTHONNOUSERSITE=1 CUTE_DSL_ARCH=sm_103a \
  python benchmarks/benchmark/benchmark_hbm_roofline_sm100.py --dtype bf16 --op both --gb 1 \
  --json /tmp/oink_sm103_hbm_roofline_bf16_current.json'
```

## Running benchmarks

All primary scripts support:

- `--quack-suite` or `--dsv3` (and `--dsv4` where applicable)
- `--configs MxN,...`
- `--dtype {bf16,fp16,fp32}`
- `--iters <ms>` and `--warmup-ms <ms>` for kernel-only timing
- `--json <path>` and/or `--csv <path>` outputs (meta + rows)

### One-command suite

Run the full Quack-suite + DSv3 set (Oink vs Quack) and write all JSON artifacts
to a timestamped directory:

```bash
conda run -n cute bash -lc 'PYTHONNOUSERSITE=1 CUTE_DSL_ARCH=sm_103a \
  python benchmarks/readme/run_sm100_suite.py --dtype bf16'

# Include DeepSeek-V4-Flash norm workloads:
conda run -n cute bash -lc 'PYTHONNOUSERSITE=1 CUTE_DSL_ARCH=sm_103a \
  python benchmarks/readme/run_sm100_suite.py --dtype bf16 --include-dsv4 \
  --out-dir /tmp/oink_sm103_suite_bf16_current'
```

Turn JSON artifacts into Markdown tables (with geomean speedups):

```bash
conda run -n cute bash -lc 'python benchmarks/readme/summarize_results.py \
  --in-dir /tmp/oink_sm103_suite_bf16_current \
  --out /tmp/oink_sm103_suite_bf16_current_summary.md'
```

Generate SM103 SVGs from current JSONs and measured roofline:

```bash
conda run -n cute bash -lc 'python benchmarks/readme/plot_quack_style_svg.py \
  --in-dir /tmp/oink_sm103_suite_bf16_current \
  --suite quack_suite --include-layernorm \
  --roofline-json /tmp/oink_sm103_hbm_roofline_bf16_current.json \
  --arch-label "SM103 / GB300" \
  --out benchmarks/media/sm103_bf16_oink_vs_quack_with_layernorm.svg'

conda run -n cute bash -lc 'python benchmarks/readme/plot_quack_style_svg.py \
  --in-dir /tmp/oink_sm103_suite_bf16_current \
  --suite dsv3_all --shape-policy first \
  --roofline-json /tmp/oink_sm103_hbm_roofline_bf16_current.json \
  --arch-label "SM103 / GB300" \
  --out benchmarks/media/sm103_bf16_oink_vs_quack_dsv3_all.svg'
```

The existing `sm100_*` SVGs in `benchmarks/media/` are historical SM100/B200
plots. Do not use them as GB300 evidence.

### Dense GEMM

```bash
PYTHONNOUSERSITE=1 CUTE_DSL_ARCH=sm_103a PYTORCH_ALLOC_CONF=expandable_segments:True \
  python benchmarks/benchmark/benchmark_gemm_dense_sm100.py \
    --shapes 128x128x128,4096x4096x4096 --iters 50 --warmup-ms 10 \
    --json /tmp/oink_dense_gemm_sm103.json
```

The dense benchmark validates `torch.ops.oink.gemm` against fp32-accumulation
PyTorch before timing. The current optimized scope is contiguous CUDA BF16
`mat_a[M,K] @ mat_b[K,N] -> out[M,N]`; unsupported shapes/dtypes use the Python
reference path in the public wrapper. Use `torch.ops.oink.gemm_out(a, b, out)`
when the caller can provide the output allocation.

For real model dense GEMM comparisons against a local Quack checkout:

```bash
PYTHONPATH=references/cute_kernels/quack:oink/src \
PYTHONNOUSERSITE=1 CUTE_DSL_ARCH=sm_103a PYTORCH_ALLOC_CONF=expandable_segments:True \
QUACK_COMPILE_WORKERS=4 \
conda run -n cute python -u oink/benchmarks/benchmark/benchmark_gemm_real_workloads_sm100.py \
  --suite all --dtype bf16 --iters 20 --warmup-ms 5 \
  --oink-mode public-out --quack-mode public \
  --json /tmp/oink_gemm_real_workloads_sm103.json
```

The real-workload harness reports Oink/Quack/Torch timings, TFLOP/s, arithmetic
intensity, per-row correctness stats, and a geomean `oink_over_quack_x` summary.
`--oink-mode backend-out` can be used for controlled kernel-config sweeps, with
`--oink-tile`, `--oink-cluster`, `--oink-2cta`, `--oink-tma-store`, and scheduler
swizzle/cluster overrides.

Current GB300 / SM103 BF16 real-workload result, measured with correctness checks
before timing, `--oink-mode public-out --quack-mode public`, and local Quack
reference import via `PYTHONPATH=references/cute_kernels/quack:oink/src`:

| suite | shape | M | N | K | Oink ms | Quack ms | torch ms | Oink TFLOP/s | Oink/Quack |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| quack_transformer | qkv_proj | 8192 | 6144 | 4096 | 0.2240 | 0.2243 | 0.2260 | 1841 | 1.002x |
| quack_transformer | attn_out | 8192 | 4096 | 4096 | 0.1485 | 0.1534 | 0.1565 | 1851 | 1.033x |
| quack_transformer | ffn_down | 8192 | 4096 | 14336 | 0.5529 | 0.6054 | 0.6422 | 1740 | 1.095x |
| quack_transformer | ffn_up_gate_dense_unfused | 8192 | 28672 | 4096 | 1.1070 | 1.1492 | 1.2542 | 1738 | 1.038x |
| deepseek_v3 | hidden_proj | 4096 | 7168 | 7168 | 0.2310 | 0.2423 | 0.2622 | 1822 | 1.049x |
| deepseek_v3 | q_lora_a | 4096 | 1536 | 7168 | 0.0620 | 0.1207 | 0.0591 | 1455 | 1.948x |
| deepseek_v3 | q_lora_b | 4096 | 24576 | 1536 | 0.1690 | 0.1785 | 0.1932 | 1830 | 1.056x |
| deepseek_v3 | kv_lora_a | 4096 | 576 | 7168 | 0.0577 | 0.1091 | 0.0419 | 586 | 1.891x |
| deepseek_v3 | kv_lora_b | 4096 | 32768 | 512 | 0.0848 | 0.1201 | 0.0878 | 1621 | 1.416x |
| deepseek_v4_flash | hidden_proj | 8192 | 7168 | 7168 | 0.5107 | 0.4908 | 0.5003 | 1648 | 0.961x |
| deepseek_v4_flash | q_lora_a | 8192 | 1536 | 7168 | 0.0972 | 0.1270 | 0.1105 | 1856 | 1.306x |
| deepseek_v4_flash | q_lora_b | 8192 | 24576 | 1536 | 0.3430 | 0.3642 | 0.3623 | 1803 | 1.062x |
| **geomean** | 12 rows |  |  |  |  |  |  |  | **1.202x** |

Roofline notes:

- The large dense rows have arithmetic intensity in the hundreds to thousands of
  FLOP/byte, so they are compute/tensor-pipe bound rather than useful-HBM-bound.
- The qkv row is at parity with Quack (`1.002x`) and reaches about `1841 TFLOP/s`,
  `~71%` of the nominal `2577.5 TFLOP/s` BF16 dense peak used by the harness.
- DSv4 hidden is the remaining hard row (`0.961x` vs Quack). NCU profiling of the
  current Oink path showed `~93.5%` sustained SM throughput and `~35.3%` DRAM
  throughput, indicating further margin likely needs a deeper kernel-family change
  rather than another simple selector tweak.

### MoE grouped GEMM

```bash
PYTHONNOUSERSITE=1 CUTE_DSL_ARCH=sm_103a PYTORCH_ALLOC_CONF=expandable_segments:True \
  python benchmarks/benchmark/benchmark_moe_grouped_gemm_sm100.py \
    --scenario 2Dx3D --tokens 4096 --experts 8 --hidden 4096 --intermediate 8192 \
    --distribution skewed --iters 100 --warmup-ms 25 \
    --json /tmp/oink_moe_grouped_gemm_sm103_2dx3d.json
```

The benchmark always validates against a PyTorch fp32-accumulation reference before
timing. On environments with `nvidia-cutlass-dsl<4.5.0`, `torch.ops.oink.grouped_mm`
uses Torch/reference fallback semantics and the benchmark prints
`backend=torch_or_reference_fallback`; do not treat that row as optimized Oink
kernel performance. Use the same command with `nvidia-cutlass-dsl>=4.5.0` to
exercise the self-contained CUTLASS 4.5-style CuTeDSL backend.

### RMSNorm forward

```bash
python benchmarks/benchmark/benchmark_rmsnorm_sm100.py --dtype bf16 --weight-dtype fp32 --quack-suite --iters 200 --warmup-ms 25 \
  --json /tmp/oink_rmsnorm_fwd_quack_suite.json

python benchmarks/benchmark/benchmark_rmsnorm_sm100.py --dtype bf16 --weight-dtype fp32 --dsv3 --iters 200 --warmup-ms 25 \
  --json /tmp/oink_rmsnorm_fwd_dsv3.json

# vLLM-style inference weights (weight dtype == activation dtype)
python benchmarks/benchmark/benchmark_rmsnorm_sm100.py --dtype bf16 --weight-dtype same --quack-suite --iters 200 --warmup-ms 25 \
  --json /tmp/oink_rmsnorm_fwd_quack_suite_wsame.json

# DeepSeek-V4-Flash norm grid
python benchmarks/benchmark/benchmark_rmsnorm_sm100.py --dtype bf16 --weight-dtype same --dsv4 --iters 200 --warmup-ms 25 \
  --json /tmp/oink_rmsnorm_fwd_dsv4_wsame.json
```

### Fused Add + RMSNorm (vLLM-style, in-place)

This is a good roofline case study kernel (heavy read/write traffic, very little
extra math). Oink exposes an **in-place** fused op that updates `x` and
`residual`. Quack's fused kernel writes separate `out` and `residual_out`
buffers, so the default benchmark baseline (`--quack-baseline kernel_inplace`)
times Quack plus the copies needed to match Oink's in-place semantics. Use
`--quack-baseline kernel` to time only the Quack kernel with preallocated
outputs.

```bash
# DeepSeek-V3 hidden-size sweep
PYTHONNOUSERSITE=1 CUTE_DSL_ARCH=sm_103 \
  python benchmarks/benchmark/benchmark_fused_add_rmsnorm_sm100.py \
    --dtype bf16 --dsv3 --iters 80 --warmup-ms 15 \
    --quack-baseline kernel_inplace \
    --json /tmp/oink_sm103_fused_add_rmsnorm_dsv3_bf16.json

# DeepSeek-V4-Flash hidden-state sweep (N=7168)
PYTHONNOUSERSITE=1 CUTE_DSL_ARCH=sm_103 \
  python benchmarks/benchmark/benchmark_fused_add_rmsnorm_sm100.py \
    --dtype bf16 --dsv4 --iters 80 --warmup-ms 15 \
    --quack-baseline kernel_inplace \
    --json /tmp/oink_sm103_fused_add_rmsnorm_dsv4_bf16.json
```

Current GB300 / SM103 BF16 results from correctness-gated runs:

| suite | rows | speedup vs Quack (min / geomean / max) |
|---|---:|---:|
| DSv3 fused-add RMSNorm | 9 | 2.022x / 2.045x / 2.089x |
| DSv4 fused-add RMSNorm | 3 | 2.030x / 2.192x / 2.521x |

DSv3 per-shape results:

| M | N | Oink ms | Quack ms | speedup | Oink TB/s |
|---:|---:|---:|---:|---:|---:|
| 4096 | 6144 | 0.0360 | 0.0727 | 2.022x | 5.598 |
| 4096 | 7168 | 0.0396 | 0.0828 | 2.089x | 5.926 |
| 4096 | 8192 | 0.0479 | 0.0993 | 2.076x | 5.610 |
| 16384 | 6144 | 0.1206 | 0.2463 | 2.043x | 6.678 |
| 16384 | 7168 | 0.1393 | 0.2830 | 2.031x | 6.742 |
| 16384 | 8192 | 0.1574 | 0.3212 | 2.040x | 6.821 |
| 65536 | 6144 | 0.4575 | 0.9285 | 2.030x | 7.041 |
| 65536 | 7168 | 0.5329 | 1.0785 | 2.024x | 7.052 |
| 65536 | 8192 | 0.6077 | 1.2466 | 2.052x | 7.068 |

DSv4 per-shape results:

| M | N | Oink ms | Quack ms | speedup | Oink TB/s |
|---:|---:|---:|---:|---:|---:|
| 4096 | 7168 | 0.0415 | 0.1047 | 2.521x | 5.655 |
| 16384 | 7168 | 0.1388 | 0.2855 | 2.057x | 6.769 |
| 65536 | 7168 | 0.5314 | 1.0785 | 2.030x | 7.072 |

### RMSNorm backward

```bash
python benchmarks/benchmark/benchmark_rmsnorm_bwd_sm100.py --dtype bf16 --weight-dtype fp32 --quack-suite --iters 100 --warmup-ms 25 \
  --csv /tmp/oink_rmsnorm_bwd_quack_suite.csv

python benchmarks/benchmark/benchmark_rmsnorm_bwd_sm100.py --dtype bf16 --weight-dtype fp32 --dsv3 --iters 100 --warmup-ms 25 \
  --csv /tmp/oink_rmsnorm_bwd_dsv3.csv
```

### Softmax (forward + backward)

```bash
python benchmarks/benchmark/benchmark_softmax_sm100.py --dtype bf16 --mode fwd_bwd --quack-suite --iters 50 --warmup-ms 25 \
  --json /tmp/oink_softmax_fwd_bwd_quack_suite.json

python benchmarks/benchmark/benchmark_softmax_sm100.py --dtype bf16 --mode fwd_bwd --dsv3 --iters 50 --warmup-ms 25 \
  --json /tmp/oink_softmax_fwd_bwd_dsv3.json
```

### Cross-entropy (forward + backward)

```bash
python benchmarks/benchmark/benchmark_cross_entropy_sm100.py --dtype bf16 --mode fwd_bwd --quack-suite --iters 50 --warmup-ms 25 \
  --json /tmp/oink_cross_entropy_fwd_bwd_quack_suite.json

python benchmarks/benchmark/benchmark_cross_entropy_sm100.py --dtype bf16 --mode fwd_bwd --dsv3 --iters 50 --warmup-ms 25 \
  --json /tmp/oink_cross_entropy_fwd_bwd_dsv3.json
```

### LayerNorm forward

```bash
python benchmarks/benchmark/benchmark_layernorm_sm100.py --dtype bf16 --quack-suite --iters 200 --warmup-ms 25 \
  --json /tmp/oink_layernorm_fwd_quack_suite.json

python benchmarks/benchmark/benchmark_layernorm_sm100.py --dtype bf16 --dsv3 --iters 200 --warmup-ms 25 \
  --json /tmp/oink_layernorm_fwd_dsv3.json
```

### LayerNorm backward

This compares Oink against ATen's native LayerNorm backward reference and,
when the installed OSS Quack package exposes `quack.rmsnorm.layernorm_bwd`, Quack
LayerNorm backward. The benchmark validates each available backend against a
chunked fp32 PyTorch formula before timing. Current table numbers use CUDA graph
warm replay (`--cuda-graph`). The local Quack package used for these runs exposes
LayerNorm forward but not `layernorm_bwd`, so Quack timing columns are omitted.

DSv3 CUDA-graph replay results (`N ∈ {6144,8192}`):

| M | N | Oink ms | Oink TB/s | ATen ref ms | Oink/ref |
|---:|---:|---:|---:|---:|---:|
| 4096 | 6144 | 0.0548 | 2.7574 | 0.0777 | 1.4190x |
| 4096 | 8192 | 0.0611 | 3.2951 | 0.0970 | 1.5873x |
| 16384 | 6144 | 0.1840 | 3.2833 | 0.2794 | 1.5183x |
| 16384 | 8192 | 0.2093 | 3.8480 | 0.3387 | 1.6183x |
| 65536 | 6144 | 0.6896 | 3.5043 | 1.0652 | 1.5447x |
| 65536 | 8192 | 0.7372 | 4.3705 | 1.3138 | 1.7823x |

DSv4 hidden LayerNorm CUDA-graph replay results (`N = 7168`):

| M | N | Oink ms | Oink TB/s | ATen ref ms | Oink/ref |
|---:|---:|---:|---:|---:|---:|
| 4096 | 7168 | 0.0591 | 2.9800 | 0.0858 | 1.4503x |
| 16384 | 7168 | 0.1990 | 3.5425 | 0.3077 | 1.5467x |
| 65536 | 7168 | 0.7467 | 3.7753 | 1.1711 | 1.5684x |

```bash
# DeepSeek-V4-Flash hidden LayerNorm shape sweep (N=7168)
env PYTHONNOUSERSITE=1 CUTE_DSL_ARCH=sm_103 PYTORCH_ALLOC_CONF=expandable_segments:True \
  conda run -n cute python -u benchmarks/benchmark/benchmark_layernorm_bwd_sm100.py \
    --dtype bf16 --weight-dtype same --dsv4 --iters 80 --warmup-ms 10 --cuda-graph \
    --json /tmp/oink_layernorm_bwd_sm103_dsv4_cuda_graph_seq.json

# DeepSeek-V3 shape sweep (N in {6144,8192})
env PYTHONNOUSERSITE=1 CUTE_DSL_ARCH=sm_103 PYTORCH_ALLOC_CONF=expandable_segments:True \
  conda run -n cute python -u benchmarks/benchmark/benchmark_layernorm_bwd_sm100.py \
    --dtype bf16 --weight-dtype same --dsv3 --iters 80 --warmup-ms 10 --cuda-graph \
    --json /tmp/oink_layernorm_bwd_sm103_dsv3_cuda_graph_seq.json
```

## Notes

- These scripts intentionally avoid importing any external Oink checkout so the
  results reflect the in-tree KernelAgent-Oink kernels.
- `src/kernelagent_oink/blackwell/rmsnorm_with_stage2.py` is a compatibility
  facade. The stage-2 scheduling policy lives in `_rmsnorm_impl.py`; keep the
  facade for downstream imports.
- For RMSNorm, the stage-2 path is a fallback used when the pointer-based fast
  path cannot be used (for example when layouts/alignments are incompatible). You
  can force it for A/B testing via `KERNELAGENT_OINK_FORCE_RMSNORM_STAGE2=1`.
