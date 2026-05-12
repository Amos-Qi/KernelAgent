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

"""
Torch custom ops wrapping Oink's Blackwell RMSNorm, dense GEMM, and MoE kernels.

These ops are designed to be:
- Architecture-aware (use CuTeDSL Blackwell SM10x kernels when available, fall back
  to a safe reference elsewhere).
- Layout-preserving for 2D row-major inputs, including padded MLA-style
  layouts where stride(0) > N and stride(1) == 1.
- torch.compile-friendly via proper fake implementations that mirror
  runtime shapes and strides.

Public ops (Python signatures):

  torch.ops.oink.rmsnorm(x: Tensor, weight: Tensor, eps: float) -> Tensor
      Functional RMSNorm. Returns a new tensor with the same shape and
      stride as x when using the fast CuTeDSL path.

  torch.ops.oink.fused_add_rms_norm(
      x: Tensor, residual: Tensor, weight: Tensor, eps: float
  ) -> None
      In-place fused residual-add + RMSNorm matching vLLM semantics:
          residual = x + residual   (stored into `residual`)
          x = RMSNorm(residual, w)  (stored into `x`)
      Mutates `x` and `residual` in-place and returns None.

  torch.ops.oink.gemm(mat_a: Tensor, mat_b: Tensor, out_dtype: Optional[torch.dtype] = None) -> Tensor
      Dense BF16 GEMM (`mat_a[M,K] @ mat_b[K,N]`) with a self-contained
      Blackwell CuTeDSL backend when available, otherwise a correctness-first
      PyTorch reference path.

  torch.ops.oink.gemm_out(mat_a: Tensor, mat_b: Tensor, out: Tensor) -> None
      Dense BF16 GEMM into caller-owned `out[M,N]`, avoiding output allocation
      overhead for integration paths that can preallocate activations.

  torch.ops.oink.grouped_mm(
      mat_a: Tensor, mat_b: Tensor, offs: Tensor,
      scenario: str = "2Dx3D", out_dtype: Optional[torch.dtype] = None
  ) -> Tensor
      MoE grouped GEMM aligned with torch.nn.functional.grouped_mm. The
      self-contained CUTLASS 4.5-style CuTeDSL backend is used when available;
      otherwise a correctness-first Torch/reference path is used.
"""

from __future__ import annotations

import importlib
import threading
from typing import Optional

import torch
from torch.library import custom_op

_RMSNORM_MOD: object | None = None
_RMSNORM_MOD_LOCK = threading.Lock()
_MOE_GEMM_MOD: object | None = None
_MOE_GEMM_MOD_LOCK = threading.Lock()
_DENSE_GEMM_MOD: object | None = None
_DENSE_GEMM_MOD_LOCK = threading.Lock()
_DENSE_SP_GEMM_MOD: object | None = None
_DENSE_SP_GEMM_MOD_LOCK = threading.Lock()
_SM_CACHE: dict[int, int] = {}
_SM_CACHE_LOCK = threading.Lock()


def _get_dense_gemm_mod():
    """Lazy import for the dense GEMM module."""
    global _DENSE_GEMM_MOD

    cached = _DENSE_GEMM_MOD
    if cached is not None:
        return cached

    with _DENSE_GEMM_MOD_LOCK:
        if _DENSE_GEMM_MOD is None:
            _DENSE_GEMM_MOD = importlib.import_module(
                "kernelagent_oink.blackwell.gemm_dense"
            )
        return _DENSE_GEMM_MOD


def _get_dense_sp_gemm_mod():
    """Lazy import for the dense software-pipeline GEMM module."""
    global _DENSE_SP_GEMM_MOD

    cached = _DENSE_SP_GEMM_MOD
    if cached is not None:
        return cached

    with _DENSE_SP_GEMM_MOD_LOCK:
        if _DENSE_SP_GEMM_MOD is None:
            _DENSE_SP_GEMM_MOD = importlib.import_module(
                "kernelagent_oink.blackwell.gemm_dense_software_pipeline"
            )
        return _DENSE_SP_GEMM_MOD


def _is_cuda_bf16_contiguous_exact_shape(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out: torch.Tensor,
    mat_a_shape: tuple[int, int],
    mat_b_shape: tuple[int, int],
    out_shape: tuple[int, int],
) -> bool:
    # Keep exact-shape rejection cheap because every gemm_out call reaches these
    # predicates before falling back to the generic GEMM selector.
    if tuple(mat_a.shape) != mat_a_shape or tuple(mat_b.shape) != mat_b_shape or tuple(out.shape) != out_shape:
        return False
    return (
        mat_a.is_cuda
        and mat_b.is_cuda
        and out.is_cuda
        and mat_a.dtype is torch.bfloat16
        and mat_b.dtype is torch.bfloat16
        and out.dtype is torch.bfloat16
        and mat_a.is_contiguous()
        and mat_b.is_contiguous()
        and out.is_contiguous()
        and _get_sm(mat_a.device) >= 100
    )


def _is_kv_lora_a_sp_shape(mat_a: torch.Tensor, mat_b: torch.Tensor, out: torch.Tensor) -> bool:
    return _is_cuda_bf16_contiguous_exact_shape(
        mat_a,
        mat_b,
        out,
        (4096, 7168),
        (7168, 576),
        (4096, 576),
    )


def _is_dsv3_q_lora_a_shape(mat_a: torch.Tensor, mat_b: torch.Tensor, out: torch.Tensor) -> bool:
    return _is_cuda_bf16_contiguous_exact_shape(
        mat_a,
        mat_b,
        out,
        (4096, 7168),
        (7168, 1536),
        (4096, 1536),
    )


def _is_dsv4_q_lora_a_shape(mat_a: torch.Tensor, mat_b: torch.Tensor, out: torch.Tensor) -> bool:
    return _is_cuda_bf16_contiguous_exact_shape(
        mat_a,
        mat_b,
        out,
        (8192, 7168),
        (7168, 1536),
        (8192, 1536),
    )


def _is_dsv3_kv_lora_b_shape(mat_a: torch.Tensor, mat_b: torch.Tensor, out: torch.Tensor) -> bool:
    return _is_cuda_bf16_contiguous_exact_shape(
        mat_a,
        mat_b,
        out,
        (4096, 512),
        (512, 32768),
        (4096, 32768),
    )


def _get_moe_gemm_mod():
    """Lazy import for the MoE grouped-GEMM module."""
    global _MOE_GEMM_MOD

    cached = _MOE_GEMM_MOD
    if cached is not None:
        return cached

    with _MOE_GEMM_MOD_LOCK:
        if _MOE_GEMM_MOD is None:
            _MOE_GEMM_MOD = importlib.import_module(
                "kernelagent_oink.blackwell.moe_grouped_gemm"
            )
        return _MOE_GEMM_MOD


def _get_rmsnorm_mod():
    """Lazy import to keep plugin registration lightweight.

    Importing the CuTeDSL kernel stack can be expensive and may require a CUDA
    context. We defer it until the first actual execution of the custom op.
    """
    global _RMSNORM_MOD

    cached = _RMSNORM_MOD
    if cached is not None:
        return cached

    with _RMSNORM_MOD_LOCK:
        if _RMSNORM_MOD is None:
            _RMSNORM_MOD = importlib.import_module("kernelagent_oink.blackwell.rmsnorm")
        return _RMSNORM_MOD


def _get_sm(device: torch.device | None = None) -> int:
    """Return SM version as an int (e.g., 103 for SM103 / Blackwell)."""
    if device is None:
        device = torch.device("cuda")
    idx = torch.device(device).index
    if idx is None:
        idx = torch.cuda.current_device()
    cached = _SM_CACHE.get(int(idx))
    if cached is not None:
        return cached
    with _SM_CACHE_LOCK:
        cached = _SM_CACHE.get(int(idx))
        if cached is None:
            major, minor = torch.cuda.get_device_capability(device)
            cached = 10 * int(major) + int(minor)
            _SM_CACHE[int(idx)] = cached
        return cached


#
# RMSNorm (functional)
#


@custom_op("oink::rmsnorm", mutates_args=())
def oink_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    Functional RMSNorm entrypoint.

    This op is model-agnostic. It expects a 2D [M, N] view of the input
    where the last dimension is contiguous (stride(1) == 1). The leading
    dimension stride(0) may be larger than N (padded-row layouts), and
    will be preserved on the fast CuTeDSL path.

    On Blackwell SM10x (SM100 and newer), this dispatches to the tuned CuTeDSL Blackwell
    RMSNorm kernel in rmsnorm.rmsnorm_forward, which in turn selects the
    best internal schedule (including DSv3-specific stage-2 kernels where
    applicable) and preserves the input's 2D stride when using the
    pointer-based path.

    On older architectures it falls back to a safe PyTorch reference
    implementation for correctness.
    """
    assert x.is_cuda, "oink::rmsnorm requires CUDA tensors"
    assert x.dim() == 2, "oink::rmsnorm expects a 2D [M, N] tensor view"
    assert weight.dim() == 1, "weight must be 1D [N]"

    sm = _get_sm(x.device)
    _rms = _get_rmsnorm_mod()
    if sm >= 100:
        # Use the tuned CuTeDSL Blackwell kernel. The public API already
        # contains all necessary gating and layout checks internally.
        y, _rstd, _res = _rms.rmsnorm_forward(
            x,
            weight=weight,
            bias=None,
            residual=None,
            eps=eps,
            store_rstd=False,
        )
        return y

    # Fallback: reference implementation (correctness-first).
    return _rms.rmsnorm_ref(
        x,
        w=weight,
        b=None,
        residual=None,
        eps=eps,
    )


@oink_rmsnorm.register_fake
def oink_rmsnorm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    Fake (meta) implementation for oink::rmsnorm.

    We must preserve x's logical layout (shape + stride) so that Inductor's
    CUDA graph capture sees the same stride contract as the real kernel.
    """
    # x is a FakeTensor here; x.shape/x.stride()/x.device/x.dtype are defined.
    return torch.empty_strided(
        x.shape,
        x.stride(),
        device=x.device,
        dtype=x.dtype,
    )


#
# Fused residual-add + RMSNorm (in-place, vLLM semantics)
#


@custom_op("oink::fused_add_rms_norm", mutates_args=("x", "residual"))
def oink_fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    """
    In-place fused residual-add + RMSNorm:

        residual <- x + residual
        x <- RMSNorm(residual, weight, eps)

    Returns:
        None (mutates `x` and `residual` in-place).
    """
    assert x.is_cuda and residual.is_cuda, (
        "oink::fused_add_rms_norm requires CUDA tensors"
    )
    assert x.shape == residual.shape, "x and residual must have the same shape"
    assert x.dtype == residual.dtype, "x and residual must have the same dtype"
    assert weight.dim() == 1, "weight must be 1D [N]"

    sm = _get_sm(x.device)
    _rms = _get_rmsnorm_mod()

    if sm < 100:
        # Non-SM10x fallback: keep semantics in-place (correctness-first).
        residual.add_(x)
        y = _rms.rmsnorm_ref(residual, w=weight, b=None, residual=None, eps=eps)
        x.copy_(y)
        return None

    # SM10x+: prefer the lowest-overhead in-place entrypoint (returns None).
    if hasattr(_rms, "fused_add_rmsnorm_inplace_"):
        _rms.fused_add_rmsnorm_inplace_(  # type: ignore[misc]
            x,
            residual,
            weight,
            eps=eps,
        )
        return None

    # Backward-compatible wrapper (returns (x, residual)).
    if hasattr(_rms, "fused_add_rmsnorm_forward_inplace"):
        _rms.fused_add_rmsnorm_forward_inplace(  # type: ignore[misc]
            x,
            residual,
            weight,
            eps=eps,
        )
        return None

    # Extremely defensive fallback if the Oink module doesn't provide
    # the in-place entrypoint.
    y, z = _rms.fused_add_rmsnorm_forward(x, residual, weight, eps=eps)
    x.copy_(y)
    residual.copy_(z)
    return None


@oink_fused_add_rms_norm.register_fake
def oink_fused_add_rms_norm_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    """
    Fake (meta) implementation for oink::fused_add_rms_norm.

    Because this op mutates its inputs in-place, the outputs alias the input
    buffers and therefore have the same shapes and strides.
    """
    return None


#
# Dense GEMM (functional)
#


@custom_op("oink::gemm", mutates_args=())
def oink_gemm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Dense GEMM: ``mat_a[M,K] @ mat_b[K,N] -> out[M,N]``."""
    _gemm = _get_dense_gemm_mod()
    return _gemm.gemm(mat_a, mat_b, out_dtype=out_dtype)  # type: ignore[attr-defined]


@oink_gemm.register_fake
def oink_gemm_fake(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if out_dtype is None:
        out_dtype = mat_a.dtype
    return torch.empty((mat_a.shape[0], mat_b.shape[1]), device=mat_a.device, dtype=out_dtype)


@custom_op("oink::gemm_out", mutates_args=("out",))
def oink_gemm_out(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Dense GEMM into caller-owned ``out[M,N]``."""
    if _is_kv_lora_a_sp_shape(mat_a, mat_b, out):
        _sp = _get_dense_sp_gemm_mod()
        _sp.gemm_backend_out(  # type: ignore[attr-defined]
            mat_a,
            mat_b,
            out,
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, 1),
            use_2cta_instrs=False,
            use_tma_store=True,
        )
        return None

    _gemm = _get_dense_gemm_mod()
    if _is_dsv3_q_lora_a_shape(mat_a, mat_b, out):
        _gemm.gemm_backend_out(  # type: ignore[attr-defined]
            mat_a,
            mat_b,
            out,
            mma_tiler_mn=(256, 192),
            cluster_shape_mn=(2, 1),
            use_2cta_instrs=True,
            use_tma_store=True,
            scheduler_swizzle_size=8,
        )
        return None
    if _is_dsv4_q_lora_a_shape(mat_a, mat_b, out):
        _gemm.gemm_backend_out(  # type: ignore[attr-defined]
            mat_a,
            mat_b,
            out,
            mma_tiler_mn=(256, 224),
            cluster_shape_mn=(2, 1),
            use_2cta_instrs=True,
            use_tma_store=True,
            use_dynamic_scheduler=True,
            swap_ab=True,
        )
        return None
    if _is_dsv3_kv_lora_b_shape(mat_a, mat_b, out):
        _gemm.gemm_backend_out(  # type: ignore[attr-defined]
            mat_a,
            mat_b,
            out,
            mma_tiler_mn=(256, 256),
            cluster_shape_mn=(2, 1),
            use_2cta_instrs=True,
            use_tma_store=True,
            scheduler_swizzle_size=8,
        )
        return None

    _gemm.gemm_out(mat_a, mat_b, out)  # type: ignore[attr-defined]
    return None


@oink_gemm_out.register_fake
def oink_gemm_out_fake(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out: torch.Tensor,
) -> None:
    return None


#
# MoE grouped GEMM (functional)
#


@custom_op("oink::grouped_mm", mutates_args=())
def oink_grouped_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    offs: torch.Tensor,
    scenario: str = "2Dx3D",
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """MoE grouped GEMM aligned with ``torch.nn.functional.grouped_mm``.

    Initial optimized scope is BF16 Blackwell ``2Dx3D`` / ``2Dx2D``. On this
    repository's current CuTeDSL 4.4.x environment the CUTLASS 4.5 descriptor
    path is intentionally gated inside the backend module, so this public op
    falls back to Torch grouped_mm or a manual PyTorch reference for correctness.
    """
    _moe = _get_moe_gemm_mod()
    return _moe.grouped_mm(  # type: ignore[attr-defined]
        mat_a,
        mat_b,
        offs,
        scenario=scenario,
        out_dtype=out_dtype,
    )


@oink_grouped_mm.register_fake
def oink_grouped_mm_fake(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    offs: torch.Tensor,
    scenario: str = "2Dx3D",
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if out_dtype is None:
        out_dtype = mat_a.dtype
    if scenario == "2Dx3D":
        shape = (mat_a.shape[0], mat_b.shape[2])
    elif scenario == "2Dx2D":
        shape = (offs.shape[0], mat_a.shape[0], mat_b.shape[1])
    else:
        raise ValueError(f"unsupported grouped_mm scenario: {scenario!r}")
    return torch.empty(shape, device=mat_a.device, dtype=out_dtype)


__all__ = [
    "oink_rmsnorm",
    "oink_fused_add_rms_norm",
    "oink_gemm",
    "oink_gemm_out",
    "oink_grouped_mm",
]
