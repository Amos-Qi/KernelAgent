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

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def _add_oink_to_syspath() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    oink_src = repo_root / "oink" / "src"
    sys.path.insert(0, str(oink_src))


def _manual_ref_2dx3d(a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    out = torch.empty((a.shape[0], b.shape[2]), device=a.device, dtype=a.dtype)
    prev = 0
    for expert_idx, cur_t in enumerate(offs.detach().cpu().tolist()):
        cur = int(cur_t)
        if cur > prev:
            out[prev:cur].copy_((a[prev:cur].float() @ b[expert_idx].float()).to(a.dtype))
        prev = cur
    return out


def _manual_ref_2dx2d(a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    out = torch.empty((offs.numel(), a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    prev = 0
    for expert_idx, cur_t in enumerate(offs.detach().cpu().tolist()):
        cur = int(cur_t)
        if cur > prev:
            out[expert_idx].copy_((a[:, prev:cur].float() @ b[prev:cur].float()).to(a.dtype))
        else:
            out[expert_idx].zero_()
        prev = cur
    return out


@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTeDSL not installed")
def test_grouped_mm_reference_cpu_2dx3d_and_2dx2d() -> None:
    _add_oink_to_syspath()
    from kernelagent_oink.blackwell import moe_grouped_gemm as moe

    torch.manual_seed(0)
    a = torch.randn(7, 8, dtype=torch.bfloat16)
    b = torch.randn(3, 8, 16, dtype=torch.bfloat16)
    offs = torch.tensor([2, 2, 7], dtype=torch.int32)
    torch.testing.assert_close(
        moe.grouped_mm_ref(a, b, offs, scenario="2Dx3D"),
        _manual_ref_2dx3d(a, b, offs),
        atol=1e-2,
        rtol=1e-2,
    )

    a2 = torch.randn(8, 7, dtype=torch.bfloat16)
    b2 = torch.randn(7, 16, dtype=torch.bfloat16)
    torch.testing.assert_close(
        moe.grouped_mm_ref(a2, b2, offs, scenario="2Dx2D"),
        _manual_ref_2dx2d(a2, b2, offs),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTeDSL not installed")
def test_oink_grouped_mm_custom_op_cuda_correctness() -> None:
    _add_oink_to_syspath()
    import kernelagent_oink

    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("requires Blackwell / SM10x for Oink custom-op registration path")

    kernelagent_oink.register(force=True)
    if not hasattr(torch.ops, "oink") or not hasattr(torch.ops.oink, "grouped_mm"):
        pytest.skip("oink grouped_mm custom op was not registered")

    torch.manual_seed(1)
    a = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2, 16, 16, device="cuda", dtype=torch.bfloat16)
    offs = torch.tensor([8, 16], device="cuda", dtype=torch.int32)
    out = torch.ops.oink.grouped_mm(a, b, offs, "2Dx3D", torch.bfloat16)
    torch.testing.assert_close(out, _manual_ref_2dx3d(a, b, offs), atol=1e-2, rtol=1e-2)

    # Reuse the same compiled pointer-specialized 2Dx3D shape with different data.
    a_next = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16)
    b_next = torch.randn(2, 16, 16, device="cuda", dtype=torch.bfloat16)
    out_next = torch.ops.oink.grouped_mm(a_next, b_next, offs, "2Dx3D", torch.bfloat16)
    torch.testing.assert_close(out_next, _manual_ref_2dx3d(a_next, b_next, offs), atol=1e-2, rtol=1e-2)

    a2 = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16)
    b2 = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16)
    out2 = torch.ops.oink.grouped_mm(a2, b2, offs, "2Dx2D", torch.bfloat16)
    torch.testing.assert_close(out2, _manual_ref_2dx2d(a2, b2, offs), atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTeDSL not installed")
def test_backend_gate_reports_cutlass_dsl_requirement() -> None:
    _add_oink_to_syspath()
    from kernelagent_oink.blackwell import moe_grouped_gemm as moe

    if moe._cutlass_dsl_supports_moe_tensormap():  # type: ignore[attr-defined]
        pytest.skip("optimized backend available in this environment")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    a = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2, 16, 16, device="cuda", dtype=torch.bfloat16)
    offs = torch.tensor([8, 16], device="cuda", dtype=torch.int32)
    out = torch.empty((16, 16), device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty((moe.get_workspace_size(2),), device="cuda", dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="nvidia-cutlass-dsl>=4.5.0"):
        moe.grouped_mm_backend_out(a, b, offs, out, workspace)
