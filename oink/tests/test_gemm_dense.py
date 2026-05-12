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


@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTeDSL not installed")
def test_gemm_ref_cpu() -> None:
    _add_oink_to_syspath()
    from kernelagent_oink.blackwell import gemm_dense

    torch.manual_seed(0)
    a = torch.randn(7, 8, dtype=torch.bfloat16)
    b = torch.randn(8, 16, dtype=torch.bfloat16)
    out = gemm_dense.gemm_ref(a, b)
    torch.testing.assert_close(out, (a.float() @ b.float()).to(torch.bfloat16), atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTeDSL not installed")
def test_oink_gemm_custom_op_cuda_correctness() -> None:
    _add_oink_to_syspath()
    import kernelagent_oink

    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("requires Blackwell / SM10x")

    kernelagent_oink.register(force=True)
    if not hasattr(torch.ops, "oink") or not hasattr(torch.ops.oink, "gemm"):
        pytest.skip("oink gemm custom op was not registered")

    torch.manual_seed(1)
    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    out = torch.ops.oink.gemm(a, b, torch.bfloat16)
    ref = (a.float() @ b.float()).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    # Reuse the same compiled pointer-specialized shape with different data.
    a2 = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    b2 = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    out2 = torch.ops.oink.gemm(a2, b2, torch.bfloat16)
    ref2 = (a2.float() @ b2.float()).to(torch.bfloat16)
    torch.testing.assert_close(out2, ref2, atol=1e-2, rtol=1e-2)

    out3 = torch.empty_like(out2)
    ret = torch.ops.oink.gemm_out(a2, b2, out3)
    assert ret is None
    torch.testing.assert_close(out3, ref2, atol=1e-2, rtol=1e-2)

    from kernelagent_oink.blackwell import gemm_dense

    a_swap = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    b_swap = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    out_swap = torch.empty((256, 128), device="cuda", dtype=torch.bfloat16)
    gemm_dense.gemm_backend_out(
        a_swap,
        b_swap,
        out_swap,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(2, 1),
        use_2cta_instrs=True,
        use_tma_store=True,
        use_dynamic_scheduler=True,
        swap_ab=True,
    )
    torch.testing.assert_close(
        out_swap,
        (a_swap.float() @ b_swap.float()).to(torch.bfloat16),
        atol=1.0,
        rtol=1e-2,
    )
