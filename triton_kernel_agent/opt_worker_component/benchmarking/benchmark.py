# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unified benchmarking for Triton kernels and PyTorch baselines.

This module consolidates kernel and PyTorch benchmarking with improved timing
utilities, L2 cache clearing, and comprehensive statistics.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Optional

import torch

from triton_kernel_agent.opt_worker_component.searching.ptx_fingerprint import (
    ptx_hash_from_cache,
)

from triton_kernel_agent.opt_worker_component.benchmarking.timing import (
    compute_timing_stats,
    prepare_pytorch_model,
    time_with_cuda_events,
    time_with_triton_do_bench,
)


class BenchmarkLockManager:
    """Manages GPU benchmarking locks to prevent resource contention."""

    def __init__(self, lock: Any, worker_id: int, logger: logging.Logger):
        """Initialize the lock manager.

        Args:
            lock: Shared multiprocessing lock for serializing GPU access
            worker_id: Worker ID for logging
            logger: Logger instance
        """
        self.lock = lock
        self.worker_id = worker_id
        self.logger = logger

    def __enter__(self):
        """Acquire the benchmarking lock."""
        self.logger.info(f"⏳ Waiting for benchmark lock (worker {self.worker_id})...")
        self.lock.acquire()
        self.logger.info(f"🔓 Acquired benchmark lock (worker {self.worker_id})")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release the benchmarking lock."""
        try:
            self.lock.release()
            self.logger.info(f"🔒 Released benchmark lock (worker {self.worker_id})")
        except Exception as e:
            self.logger.warning(f"Failed to release benchmark lock: {e}")
        return False


class Benchmark:
    """Unified benchmark for Triton kernels and PyTorch baselines.

    Supports two modes:
    1. Subprocess mode: Runs benchmarks in isolated processes (for compatibility)
    2. Direct mode: Uses in-process timing utilities (faster, more flexible)
    """

    def __init__(
        self,
        logger: logging.Logger,
        artifacts_dir: Path,
        benchmark_lock: Any,
        worker_id: int = 0,
        warmup: int = 25,
        repeat: int = 100,
        timing_method: str = "cuda_event",
    ):
        """Initialize the benchmark.

        Args:
            logger: Logger instance
            artifacts_dir: Directory for benchmark artifacts
            benchmark_lock: Shared lock to serialize GPU benchmarking
            worker_id: Worker ID
            warmup: Number of warmup iterations (or warmup time in ms for do_bench)
            repeat: Number of repeat iterations (or rep time in ms for do_bench)
            timing_method: Timing method ("cuda_event", "do_bench", "host_time")
        """
        self.logger = logger
        self.artifacts_dir = artifacts_dir
        self.lock_manager = BenchmarkLockManager(benchmark_lock, worker_id, logger)
        self.warmup = warmup
        self.repeat = repeat
        self.timing_method = timing_method

    def benchmark_kernel(
        self,
        kernel_file: Path,
        problem_file: Path,
        baseline_file: Optional[Path] = None,
    ) -> dict[str, Any]:
        """Benchmark Triton kernel performance using subprocess isolation.

        Uses subprocess for crash protection of potentially buggy kernels.

        Args:
            kernel_file: Path to kernel file
            problem_file: Path to problem file
            baseline_file: Path to baseline kernel (optional)

        Returns:
            Dictionary with benchmark results:
                - time_ms: Mean time in ms
                - speedup: Speedup vs baseline
        """
        ptx_cache_dir = Path(tempfile.mkdtemp(prefix="triton_cache_bench_"))
        try:
            with self.lock_manager:
                results_json = self.artifacts_dir / "benchmark_results.json"
                benchmark_script = Path(__file__).parent / "kernel_subprocess.py"

                # Use KERNEL_PROFILER_PYTHON (the PAR bootstrap) when set, like
                # ncu_profiler.py; bare sys.executable is un-bootstrapped in a PAR.
                bench_python = (
                    os.environ.get("KERNEL_PROFILER_PYTHON") or sys.executable
                )

                cmd = [
                    bench_python,
                    str(benchmark_script),
                    "--problem",
                    str(problem_file),
                    "--kernel",
                    str(kernel_file),
                    "--warmup",
                    str(self.warmup),
                    "--repeat",
                    str(self.repeat),
                    "--json",
                    str(results_json),
                    "--quiet",
                ]

                if baseline_file:
                    cmd.extend(["--parent", str(baseline_file)])

                # Isolate this benchmark's Triton compilation cache so we can
                # capture its PTX for fingerprint-based dedup without being
                # contaminated by sibling workers' artifacts.
                env = {**os.environ, "TRITON_CACHE_DIR": str(ptx_cache_dir)}

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    env=env,
                )

                if result.returncode != 0:
                    error_msg = (
                        result.stderr.strip()
                        or result.stdout.strip()
                        or "Unknown error"
                    )
                    self.logger.error(f"Kernel benchmark failed: {error_msg}")
                    return {"time_ms": float("inf"), "speedup": 0.0, "ptx_hash": None}

                with open(results_json, "r") as f:
                    results = json.load(f)

                kernel_name = kernel_file.stem
                kernel_results = results.get("kernels", {}).get(kernel_name, {})

                # Capture the PTX fingerprint from the isolated cache dir.
                # A None result is graceful — dedup treats it as a singleton.
                ptx_hash = ptx_hash_from_cache(ptx_cache_dir)

                return {
                    "time_ms": kernel_results.get("time_ms", float("inf")),
                    "speedup": kernel_results.get("speedup", 1.0),
                    "parent_time_ms": kernel_results.get("parent_time_ms"),
                    "time_vs_parent": kernel_results.get("time_vs_parent"),
                    "ptx_hash": ptx_hash,
                }

        except Exception as e:
            self.logger.error(f"Kernel benchmark failed: {e}")
            return {"time_ms": float("inf"), "speedup": 0.0, "ptx_hash": None}
        finally:
            shutil.rmtree(ptx_cache_dir, ignore_errors=True)

    def benchmark_pytorch(
        self,
        problem_file: Path,
        dtype: Optional[torch.dtype] = None,
    ) -> dict[str, Any]:
        """Benchmark PyTorch baseline using direct in-process timing.

        Always uses direct mode (PyTorch is stable, doesn't need subprocess isolation).

        Args:
            problem_file: Path to problem file (must define Model class and get_inputs())
            dtype: Data type to use (default: auto-detect based on model parameters)

        Returns:
            Dictionary with benchmark results:
                - time_ms: Mean time in ms
                - stats: Full timing statistics (mean, std, min, max, all_times, etc.)
        """
        try:
            with self.lock_manager:
                model, inputs = prepare_pytorch_model(
                    problem_file=problem_file,
                    device="cuda",
                    dtype=dtype,
                )

                if self.timing_method == "do_bench":
                    times = time_with_triton_do_bench(
                        lambda: model(*inputs),
                        [],
                        warmup=self.warmup,
                        rep=self.repeat,
                        verbose=False,
                    )
                else:  # cuda_event
                    times = time_with_cuda_events(
                        lambda: model(*inputs),
                        [],
                        num_warmup=self.warmup,
                        num_trials=self.repeat,
                        clear_cache=True,
                        verbose=False,
                    )

                stats = compute_timing_stats(times)

                return {
                    "time_ms": stats["mean"],
                    "stats": stats,
                }

        except Exception as e:
            self.logger.error(f"PyTorch baseline benchmark failed: {e}")
            self.logger.error(traceback.format_exc())
            return {"time_ms": float("inf")}

    def benchmark_pytorch_compile(
        self,
        problem_file: Path,
        dtype: Optional[torch.dtype] = None,
        mode: Optional[str] = None,
    ) -> dict[str, Any]:
        """Benchmark torch.compile'd PyTorch baseline using direct in-process timing.

        Mirrors benchmark_pytorch() but wraps the model with torch.compile()
        and uses extended warmup (3 forward calls) before timing to allow
        compilation and warm caches.

        Args:
            problem_file: Path to problem file (must define Model class and get_inputs())
            dtype: Data type to use (default: auto-detect based on model parameters)
            mode: torch.compile mode. ``"max-autotune"`` measures the
                Inductor-autotuned reference — the kernels a production AOT
                deploy with ``max_autotune_gemm=True`` actually competes with
                (plain ``torch.compile`` understates that bar). None keeps
                torch.compile's default mode.

        Returns:
            Dictionary with benchmark results:
                - time_ms: Mean time in ms
                - stats: Full timing statistics (mean, std, min, max, all_times, etc.)
        """
        tf32_prev = torch.backends.cuda.matmul.allow_tf32
        try:
            with self.lock_manager:
                model, inputs = prepare_pytorch_model(
                    problem_file=problem_file,
                    device="cuda",
                    dtype=dtype,
                )

                # Match the UV problems' numerics contract (fp32 accumulate,
                # no TF32) so the reference competes on equal numeric terms
                # regardless of process defaults.
                torch.backends.cuda.matmul.allow_tf32 = False

                model = torch.compile(model, mode=mode)

                # Extended warmup: 3 forward calls to trigger compilation
                for _ in range(3):
                    model(*inputs)
                torch.cuda.synchronize()

                if self.timing_method == "do_bench":
                    times = time_with_triton_do_bench(
                        lambda: model(*inputs),
                        [],
                        warmup=self.warmup,
                        rep=self.repeat,
                        verbose=False,
                    )
                else:  # cuda_event
                    times = time_with_cuda_events(
                        lambda: model(*inputs),
                        [],
                        num_warmup=self.warmup,
                        num_trials=self.repeat,
                        clear_cache=True,
                        verbose=False,
                    )

                stats = compute_timing_stats(times)

                return {
                    "time_ms": stats["mean"],
                    "stats": stats,
                }

        except Exception as e:
            self.logger.error(f"PyTorch compile benchmark failed: {e}")
            self.logger.error(traceback.format_exc())
            return {"time_ms": float("inf")}
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32_prev

    # The exact inductor options the UL production AOT deploy compiles with
    # (vector-ai-unity-learner deploy/utils.py aoti_compile_and_package call +
    # the v3_q3a deploy_config.compile_config). The reference must compete
    # with production's kernels, not a guessed config.
    _AOTI_PROD_INDUCTOR_CONFIGS: dict[str, Any] = {
        "layout_optimization": False,
        "assert_indirect_indexing": False,
        "max_autotune_gemm": True,
        "benchmark_epilogue_fusion": False,
        "triton.autotune_at_compile_time": False,
    }

    def benchmark_pytorch_aoti(
        self,
        problem_file: Path,
        dtype: Optional[torch.dtype] = None,
        inductor_configs: Optional[dict[str, Any]] = None,
        cuda_graph: bool = False,
    ) -> dict[str, Any]:
        """Benchmark the AOTI reference — the production-parity bar.

        The UL serving stack builds a ``.pt2`` via ``torch.export`` +
        ``torch._inductor.aoti_compile_and_package`` (``max_autotune_gemm=True``).
        This measures export + AOTI compile with the production inductor
        configs, timed EAGER-LAUNCH by default — candidates are timed raw
        (launches included), so the reference must be too; a symmetric A/B is
        the point of the harness, and serving-mode effects (CUDA-graph launch
        collapse) are measured by the serving profiles, not here.
        ``cuda_graph=True`` opts into whole-forward capture/replay timing for
        mode studies (under CG production BOTH sides would be graphed, so the
        one-sided graphed number must not be used as the candidate bar);
        capture failure falls back to eager-launch timing with a warning.

        Returns dict with ``time_ms``, ``stats``, and ``graphed`` (whether the
        CUDA-graph wrap succeeded).
        """
        tf32_prev = torch.backends.cuda.matmul.allow_tf32
        tmp_dir = None
        try:
            with self.lock_manager:
                from torch._inductor import aoti_compile_and_package, aoti_load_package

                model, inputs = prepare_pytorch_model(
                    problem_file=problem_file,
                    device="cuda",
                    dtype=dtype,
                )
                # Match the UV problems' numerics contract (no TF32).
                torch.backends.cuda.matmul.allow_tf32 = False

                exported = torch.export.export(model, tuple(inputs))
                tmp_dir = tempfile.mkdtemp(prefix="ka_aoti_ref_")
                pt2_path = str(Path(tmp_dir) / "reference.pt2")
                cfgs = dict(
                    self._AOTI_PROD_INDUCTOR_CONFIGS
                    if inductor_configs is None
                    else inductor_configs
                )
                aoti_compile_and_package(
                    exported, package_path=pt2_path, inductor_configs=cfgs
                )
                # run_single_threaded so the CUDA-graph wrap can capture it —
                # the same flag UL's CG serving manager passes
                # (pytorch/pytorch@85467ed). The default multi-threaded runner
                # fails capture and silently degrades this reference to
                # eager-launch.
                runner = aoti_load_package(pt2_path, run_single_threaded=cuda_graph)

                # Warmup (also primes the CUDA-graph memory pool paths).
                for _ in range(3):
                    runner(*inputs)
                torch.cuda.synchronize()

                graphed = False
                fn = lambda: runner(*inputs)  # noqa: E731
                if cuda_graph:
                    try:
                        side = torch.cuda.Stream()
                        side.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(side):
                            runner(*inputs)
                        torch.cuda.current_stream().wait_stream(side)
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            runner(*inputs)
                        fn = graph.replay
                        graphed = True
                    except Exception as exc:
                        self.logger.warning(
                            f"AOTI reference CUDA-graph capture failed ({exc}); "
                            "timing the eager-launch AOTI runner instead"
                        )

                if self.timing_method == "do_bench":
                    times = time_with_triton_do_bench(
                        fn, [], warmup=self.warmup, rep=self.repeat, verbose=False
                    )
                else:  # cuda_event
                    times = time_with_cuda_events(
                        fn,
                        [],
                        num_warmup=self.warmup,
                        num_trials=self.repeat,
                        clear_cache=True,
                        verbose=False,
                    )

                stats = compute_timing_stats(times)
                return {
                    "time_ms": stats["mean"],
                    "stats": stats,
                    "graphed": graphed,
                }

        except Exception as e:
            self.logger.error(f"AOTI reference benchmark failed: {e}")
            self.logger.error(traceback.format_exc())
            return {"time_ms": float("inf"), "graphed": False}
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32_prev
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)
