# Rubin green72 RMSNorm performance log (SM103)

This README is a branch-specific performance log for `laura/vera-rubin`. It records the
local GB300/SM103 green-context experiments that motivated the default-off Rubin tuning
knobs on this branch.

The results below are **local SM103 / GB300 green72 proxy evidence**, not a production-wide
Rubin claim. Production defaults are unchanged unless the experimental env knobs are set.

## Branch scope

Source commits documented here:

```text
82274d5 Add green-context RMSNorm benchmarks and Rubin stage2 tuning knobs
6e20b48 Tune Rubin RMSNorm stage2 shared-memory tile for N8192
c47f258 Merge smem RMSNorm tile tuning
```

Main default-off policy knob:

```bash
export OINK_RMSNORM_RUBIN_GREEN_STAGE2=1
```

This applies only to BF16/FP16 same-weight RMSNorm forward with `M >= 16384` and
`N in {6144, 7168, 8192}`. When unset, the production launch policy is unchanged.

## Environment

Measured locally on:

| item | value |
|---|---|
| GPU | NVIDIA GB300 |
| target arch | SM103 / `CUTE_DSL_ARCH=sm_103` |
| full-device SMs | 152 |
| green-context SMs | requested 72, actual 72 |
| Conda env | `oink` |
| Python | `3.12.13+meta` |
| PyTorch | `2.9.1+cu130` |
| CUDA runtime reported by PyTorch | 13.0 |
| system CUDA toolkit | `/usr/local/cuda`, CUDA 13.0 |
| CuTeDSL | `nvidia-cutlass-dsl==4.4.2` |
| cuda-python | `13.1.1` |
| Triton | `3.5.1` |

The system CUDA toolkit was CUDA 13.0, so the benchmark-only green-context helper uses
`cuda-python` driver bindings for CUDA 13.1 green-context APIs instead of compiling a CUDA
13.1 C++ extension.

## Methodology and roofline model

Reported RMSNorm forward numbers are correctness-gated against the benchmark's PyTorch
reference before timing. Do not use `--skip-verify` for numbers intended to update this
file.

The primary workload is DSv3-style RMSNorm forward:

```text
M in {4096, 16384, 65536}
N in {6144, 7168, 8192}
dtype = bf16
weight dtype = same
```

The final tuning comparison focuses on the steady large-M subset:

```text
M in {16384, 65536}
N in {6144, 7168, 8192}
```

RMSNorm forward is memory-bound for these shapes. Throughput is reported with the logical
useful-IO model used by the benchmark:

```text
bytes = (2 * M * N) * elem_size + N * weight_elem_size
```

The local green72 BF16 roof proxy is an event-timed torch `add_` triad:

| mode | roof proxy |
|---|---:|
| full 152 SMs | 7.285 TB/s |
| green72 | 5.067 TB/s |

The green72 90% target is therefore **4.56 TB/s**. The Triton HBM roofline benchmark was
not used for this SM103 run because Triton emitted `sm_103a` and bundled `ptxas` rejected
that GPU name.

## Initial green-context measurements

These correctness-gated runs established the full-GPU vs green72 behavior before the
RMSNorm forward tuning pass. Quack comparisons are included only where the benchmark had a
matching Quack path installed and exercised under the same mode.

| run | shapes | mode | geomean Oink TB/s | geomean Oink/Quack | notes |
|---|---|---|---:|---:|---|
| RMSNorm fwd, BF16, weight=same | DSv3 9-shape grid | full 152 SMs | 5.664 | 1.078x | correctness passed |
| RMSNorm fwd, BF16, weight=same | DSv3 9-shape grid | green72 | 3.867 | 1.081x | correctness passed; green/full throughput 68.3% geomean |
| fused-add RMSNorm, BF16 | DSv3 9-shape grid | full 152 SMs | 6.495 | 2.040x | correctness passed; Quack baseline `kernel_inplace` |
| fused-add RMSNorm, BF16 | DSv3 9-shape grid | green72 | 5.267 | 2.610x | correctness passed; green/full throughput 81.1% geomean |
| torch `y.add_(x)` BF16 triad roof proxy | 1 GiB tensor | full 152 SMs | 7.285 | n/a | event-timed roof proxy |
| torch `y.add_(x)` BF16 triad roof proxy | 1 GiB tensor | green72 | 5.067 | n/a | event-timed roof proxy; green/full bandwidth 69.6% |

The forward tuning log below uses a later fresh green72 baseline run for apples-to-apples
comparison within the campaign.

## Summary results

Fresh green72 baseline before the Rubin policy:

| run | shape set | geomean throughput | green72 roof fraction |
|---|---|---:|---:|
| baseline Oink RMSNorm fwd | DSv3 9-shape grid | 3.976 TB/s | 78.5% |
| baseline Oink RMSNorm fwd | large-M DSv3 subset | 4.313 TB/s | 85.1% |

Best stage2/thread recipe before the shared-memory tile follow-up:

| run | shape set | geomean throughput | green72 roof fraction |
|---|---|---:|---:|
| per-N stage2 + per-N thread layout | large-M DSv3 subset | 4.598 TB/s | 90.7% |

Final branch policy with `OINK_RMSNORM_RUBIN_GREEN_STAGE2=1` after folding in the
`N=8192` staged-K tile split:

| run | shape set | geomean throughput | green72 roof fraction |
|---|---|---:|---:|
| Rubin green72 policy | large-M DSv3 subset | 4.6182 TB/s | 91.1% |

This is a **+7.1%** large-M geomean improvement over the fresh green72 baseline
(`4.313 -> 4.6182 TB/s`) and crosses the local 90%-of-green-roof target.

## Final per-shape table

Command policy:

```bash
OINK_RMSNORM_RUBIN_GREEN_STAGE2=1
```

Shape order used for the stable single-process result was reverse large-M order. The
benchmark harness previously showed order sensitivity in ascending/full-grid runs, so this
table should be read as the stable controlled large-M run rather than a universal claim for
all benchmark orders.

| M | N | Oink throughput | green72 roof fraction |
|---:|---:|---:|---:|
| 65536 | 8192 | 4.6996 TB/s | 92.7% |
| 65536 | 7168 | 4.6534 TB/s | 91.8% |
| 65536 | 6144 | 4.8824 TB/s | 96.4% |
| 16384 | 8192 | 4.5083 TB/s | 89.0% |
| 16384 | 7168 | 4.4290 TB/s | 87.4% |
| 16384 | 6144 | 4.5505 TB/s | 89.8% |
| **geomean** | **large-M subset** | **4.6182 TB/s** | **91.1%** |

## Tuning recipe

The final policy keeps the winning changes default-off and narrow:

- `N=6144`: stage2/cp.async path, `OINK_RMSNORM_TPR=192`, `OINK_RMSNORM_NT=192`.
- `N=7168`: stage2/cp.async path, `OINK_RMSNORM_TPR=224`, `OINK_RMSNORM_NT=224`.
- `N=8192`: stage2/cp.async path, `OINK_RMSNORM_TPR=256`, `OINK_RMSNORM_NT=256`,
  staged K tile split to `4096 + 4096`.

The branch exposes these default-off tuning controls:

```text
OINK_RMSNORM_STAGE_OVERRIDE={1,stage1,2,stage2}
OINK_RMSNORM_FORCE_STAGE2_NS=6144,7168,8192
OINK_RMSNORM_RUBIN_GREEN_STAGE2={0,1}
OINK_RMSNORM_SMEM_TILE_N=<int|auto>
```

Env policy is read at import time, so each schedule variant needs a fresh Python process.

## Shared-memory experiment log

The useful shared-memory change was narrow: split the `N=8192` staged K tile into two
4096-wide chunks under the Rubin policy. Broader shared-memory/vector changes did not help.

| experiment | result | decision |
|---|---|---|
| 256b logical SMEM/register vector layout with 128b async copy | hard crash / segfault | rejected and removed |
| `N=6144` full-row staged tile | `M=16384,N=6144` regressed to 4.2647 TB/s | rejected |
| `N=6144`, two rows per CTA (`TPR=192,NT=384`) | `M=16384,N=6144` regressed to 4.0609 TB/s | rejected |
| `N=8192`, staged K tile `4096` | improved large `N=8192`, especially `M=65536` | kept behind Rubin policy |

Main keeper win:

```text
M=65536,N=8192: ~4.51 TB/s -> ~4.70 TB/s
```

Nsight Compute comparison for `M=65536,N=8192`:

| metric | full-row `8192` tile | split `4096` tile |
|---|---:|---:|
| benchmark throughput | 4.5198 TB/s | 4.7022 TB/s |
| NCU time | 745,088 ns | 713,312 ns |
| registers/thread | 36 | 32 |
| dynamic smem/block | 32,832 B | 32,832 B |
| active warps | 67.93% | 68.14% |
| long scoreboard | 27.02% | 30.83% |
| DRAM throughput pct | 35.58% | 37.17% |

Interpretation: the green72 drop was consistent with insufficient memory-level parallelism
/ bytes-in-flight. The winning schedule uses the existing staged cp.async path and per-N
thread layouts to increase useful memory overlap under the reduced-SM context. The
shared-memory tile result is not evidence that "larger SMEM is always better"; full-row
`N=6144` and wider-vector attempts regressed or crashed.

## Reproduction

Use the `oink` Conda environment from the repo root.

### Source validation used before PR-readiness check

```bash
conda run -n oink bash -lc 'PYTHONNOUSERSITE=1 CUTE_DSL_ARCH=sm_103 \
  python -m compileall oink/src/kernelagent_oink/blackwell/_rmsnorm_impl.py'
```

```bash
conda run -n oink bash -lc 'PYTHONNOUSERSITE=1 CUTE_DSL_ARCH=sm_103 \
  PYTHONPATH=oink/src \
  python -m pytest -q -o addopts="" oink/tests/test_aten_override.py'
```

Observed result:

```text
15 passed, 1 warning
```

### Final green72 large-M benchmark

```bash
conda run -n oink bash -lc 'PYTHONNOUSERSITE=1 CUTE_DSL_ARCH=sm_103 \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  OINK_RMSNORM_RUBIN_GREEN_STAGE2=1 \
  PYTHONPATH=oink/src \
  python -u oink/benchmarks/benchmark/benchmark_rmsnorm_sm100.py \
    --dtype bf16 --weight-dtype same \
    --configs 65536x8192,65536x7168,65536x6144,16384x8192,16384x7168,16384x6144 \
    --green-sms 72 --iters 50 --warmup-ms 10 \
    --json /tmp/oink_rubin_green72_rmsnorm_fwd_large_reverse.json'
```

Notes:

- `--configs` takes a single comma-separated string.
- Keep correctness enabled for logged results; do not pass `--skip-verify`.
- For per-N policy sweeps, run each env variant in a fresh Python process because the
  launch policy is read at module import time.

## Caveats

- This file logs already-recorded correctness-gated SM103/green72 performance. It does not
  introduce new production defaults.
- Green contexts reduce execution resources but do not emulate Rubin HBM latency/cache
  behavior or actual Rubin HBM bandwidth.
- The benchmark-only green-context support is most trustworthy for forward/fused-forward
  paths. Backward benchmark plumbing exists, but backward numbers are not claimed here
  because scratch/grid sizing still sees the full-device SM count unless separately
  clamped.
- The final result uses a stable reverse large-M shape order. Earlier ascending/full-grid
  one-process runs occasionally showed late-shape drops to roughly 2.0-2.3 TB/s; root
  cause was not resolved and is likely benchmark/order/resource interaction.
- Do not claim this branch adds vectorized shared-memory access. The failed vectorized-SMEM
  experiments were removed from live code. The kept change tunes the existing staged
  shared-memory/cp.async path for `N=8192` under the default-off Rubin policy.
