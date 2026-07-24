# Unity CVR android-conversion v62 — DHEN layer-0 ENSEMBLE SUM (kernel dir #2):
# the weighted sum of the three interaction projections,
#
#   out = sw0*proj_dlrm(t0) + sw1*proj_dcn(t1) + sw2*proj_mlp(t2)
#   sw  = softmax(ensemble_weight)          (WeightedSum, 3 learned scalars)
#
# WHY (serving-slice profile 2026-07-24, 1g.24gb, request_batch parity):
# proj_dlrm / proj_dcn are two of the four identical 4000x8256 @ 8256x2048
# fp16 addmms measured at 2227.7 us EACH on the slice (53% of roofline);
# proj_mlp is the same pattern at K=1024. Plus the WeightedSum glue the
# compiler emits around them (stack/mul/sum eltwise passes over 4000x2048).
# Kernel dir #1 (optimize_cvr_v62_dhen_gemm, AUDITED winner 1.68x vs the
# incumbent pair on the 1g slice) already owns the OTHER two big GEMMs of
# the cluster: input_projection + the MLP interaction's first linear (the
# shared-input pair). This dir owns the remaining three projections as ONE
# GEMM via the sum identity:
#
#   sum_i (A_i @ B_i)  ==  [A_0 | A_1 | A_2] @ [B_0 ; B_1 ; B_2]
#
# i.e. a single K-segmented GEMM, K_TOT = 8256 + 8256 + 1024 = 17536, with
# the softmax scalars (inference-time constants) and the three biases folded
# into the packed weight/bias. The A matrices are runtime activations and
# must NOT be materialized into a concat — the kernel walks three A base
# pointers per M-tile instead (zero-copy).
#
# DELIBERATELY OUT OF SCOPE: input_projection (composes with dir #1's pair
# at integration: layer_out = SwishLayerNorm(dir1.ip_half + this_sum)), and
# SwishLayerNorm itself (rowwise stats need a cross-tile pass; candidates MAY
# try a fused second pass, but it is not required).
#
# Provenance (verified in UL source, branch qi/android-row-v62-opt):
#   config.json dhen_config: dhen_cross_methods[0] = ["dlrm","dcn","mlp"],
#     layer_output_dims [2048, 2048], ensemble_mode "weighted_sum"
#   dhen_layer.py: DHENLayer.forward =
#     final_layer_norm(input_projection(x) + ensembler([proj_i(mod_i(x))]))
#   weighted_sum.py: w = softmax(weight); (stack(inputs) * w).sum(dim=0)
#   dlrm/dcn preserve the 8256-wide input (hence 8256->2048 projections,
#   matching the measured GEMM table); mlp ends at 1024 (mlp_layers
#   [2048, 1024]) so proj_mlp is 1024->2048.
#
# Shapes are the SERVING shape (request_batch_size 2 envelope: M=4000 rows).
# Target hardware: MIG 1g.24gb slice (46 SMs, 448 GB/s) via beam_search_cvr.
#
# Numerics contract: I/O torch.float16 (v62 deploy precision). The serving
# chain rounds each addmm to fp16, multiplies by fp16-rounded softmax
# weights (fp32 opmath), and does ONE fp32-accumulated sum with a final
# fp16 round. A fused single-accumulator kernel replaces the intermediate
# rounds with one round at the end — MORE accurate, but not bit-identical:
# the parity gate is therefore a BAND (test.py: 3e-3 + 3e-3*|ref|, no
# zero-median requirement), unlike dir #1's bit-exact gate.
# K-divisibility: 8256 and 1024 are divisible by 32 and 64 but NOT both by
# 128 — BK must stay in {32, 64} unless K masks are added.

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

M = 4000
K01 = 8256   # dlrm / dcn projection input width
K2 = 1024    # mlp projection input width
N = 2048     # layer_output_dim
K_TOT = K01 + K01 + K2


class Model(nn.Module):
    """Eager reference: the exact serving chain — three addmms (each with a
    single fp16 round), softmax weights, stack -> broadcast mul -> sum(dim=0)
    exactly as WeightedSum.forward runs it."""

    def forward(
        self,
        t0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor,
        wp0: torch.Tensor, bp0: torch.Tensor,
        wp1: torch.Tensor, bp1: torch.Tensor,
        wp2: torch.Tensor, bp2: torch.Tensor,
        ens_w: torch.Tensor,
    ) -> torch.Tensor:
        p0 = torch.addmm(bp0, t0, wp0)
        p1 = torch.addmm(bp1, t1, wp1)
        p2 = torch.addmm(bp2, t2, wp2)
        w = F.softmax(ens_w, dim=0)
        stacked = torch.stack([p0, p1, p2], dim=0)
        return (stacked * w.view(-1, 1, 1)).sum(dim=0)


def get_inputs(seed: int = 0) -> List[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    t0 = torch.randn(M, K01, generator=g)
    t1 = torch.randn(M, K01, generator=g)
    t2 = torch.randn(M, K2, generator=g)
    wp0 = torch.randn(K01, N, generator=g) * (K01 ** -0.5)
    bp0 = torch.randn(N, generator=g) * 0.02
    wp1 = torch.randn(K01, N, generator=g) * (K01 ** -0.5)
    bp1 = torch.randn(N, generator=g) * 0.02
    wp2 = torch.randn(K2, N, generator=g) * (K2 ** -0.5)
    bp2 = torch.randn(N, generator=g) * 0.02
    # Spread the raw ensemble logits so softmax is meaningfully non-uniform.
    ens_w = torch.randn(3, generator=g) * 0.5
    return [t0, t1, t2, wp0, bp0, wp1, bp1, wp2, bp2, ens_w]


def get_init_inputs():
    return []
