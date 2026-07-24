# Unity user-value model v3-q3a — TRUNK FAN-OUT (kernel dir #9): the full
# per-component GEMM cluster consuming the trunk — towers -> COMBINED-gated
# tower heads -> main head groups (gated) -> ZiLN/Poisson/BCE decodes ->
# 2-component ensemble aggregate -> nn_output stack.
#
# WHY (measured on the SERVING envelope, 2026-07-24): on the MIG 1g.24gb
# quarter-slice at R=6 (the real prod partition + batching), this cluster is
# the dominant optimizable GPU cost of the optimized artifact:
#   - cutlass ...align2 GEMMs, 7.6% of GPU time (18.0ms/capture, 80 inst):
#     the 659x659 / 593x593 group-gate GEMMs — K/N=659/593 are odd, CUTLASS
#     falls to the align2 slow path (~25-35% penalty).
#   - triton_poi_fused_mm_mul_sigmoid, 4.5% (10.6ms): the GateLayer
#     x*sigmoid(gate(x)) epilogues Inductor splits from their GEMMs.
#   - tem_fused_addmm(_silu/_relu) template GEMMs ~9%: tower layers + heads,
#     each launched separately, each RE-READING its input from DRAM on a
#     slice with 1/4 the bandwidth (448 GB/s).
# Structural headroom the per-GEMM incumbent cannot reach: read the trunk
# ONCE per M-tile for all four towers (grouped/horizontally-fused GEMM),
# carry the GateLayer sigmoid-multiply as a GEMM epilogue, keep group inputs
# in registers/SMEM through gate+heads, fuse the tiny decode pointwise ops.
# Alignment note: tl.dot with masked K handles 659/593 without the CUTLASS
# align cliff; padding to 672/608 (x16) is also legal — pad cols are exact
# zeros through sigmoid-mul when the padded gate rows/cols are zero.
#
# Provenance (VERIFIED serving geometry — test/triton/trunk_tail_fixture.py
# at the v3-q3a checkpoint + nn_config.model in v3_q3a config.json + the
# measured GEMM table):
#   trunk width = 1328 (kalign-padded; raw 1322, cols [1322:1328) are ZERO)
#   fc_shared = Identity (towers read the trunk directly)
#   towers (DNN, activation swish/silu, 2 layers, no BN, inference dropout 0):
#     iap [384,384]; adrev [192,192]; retention [256,256]; adrev_bce [128,128]
#   every HeadsModule uses COMBINED gating (cross.GateLayer):
#     gated = x * sigmoid(x @ Wg + bg)   (Wg square: input_dim x input_dim)
#     logits = gated @ Wh + bh           (Wh: input_dim x linear_output_dim)
#   tower head logits widths: iap 13, adrev 4, retention 6, adrev_bce 7
#     serving slices: main_payer_d7 = iap logits[:, 6:7],
#                     main_retention_d7 = retention logits[:, 3:4]
#   main head groups (input = concat of [tower output, tower head logits] per
#   input tower, in input_towers order — PROVEN against DepositorModel
#   forward_flat (model.py: input_parts.append(tower_x) then .append(
#   transformed) per config.input_towers); sizes are verified:
#     iap_main:  [iap(384), iap_logits(13), retention(256), ret_logits(6)]
#                -> 659 (matches the measured 659x659 gate GEMM)
#     adrev_main:[adrev(192), adrev_logits(4), adrev_bce(128), abce_logits(7),
#                retention(256), ret_logits(6)] -> 593 (matches 593x593)
#   group heads: iap_main -> main_iap_d7 (ziln,3) ++ main_iap_d28 (ziln,3);
#                adrev_main -> main_adrev_d0/d7/d28 (poisson, 1 each)
#   ensemble: 2 components, mean; adrev cols x 1.053 calibration after mean.
#
# Decodes (exact per ZilnHead.p_loc_scale + prob_value_pred, head_module.py:
#   p_loc_scale applies the tanh squashes BEFORE softplus/exp — this is why
#   the integrated trunk_tail kernel needs libdevice tanh):
#   ziln(t[:,0:3]): prob=sigmoid(t0); loc=10*tanh(t1/10);
#                   scale=softplus(3*tanh(t2/3));
#                   value=exp(loc + 0.5*scale^2); final=prob*value
#   poisson(t): softplus(t0);  bce(t): sigmoid(t0)
# nn_output stack (16 cols): [iap_d7 (p,v,f,loc,scale), iap_d28 (p,v,f,loc,
#   scale), adrev_d0, adrev_d7, adrev_d28, ret_prob_d7, payer_prob_d7, 0.0]
#
# Shapes are the SERVING shape: rows = 6*1024 (prod max_batch_size=6 x the
# 1024 candidate bucket; online candidates are hard-capped at 1000, p50 965,
# so 6144 rows ~= the real max-fill Sigma of ~5.7k). Target hardware is the
# MIG 1g.24gb slice (strategy beam_search_uv_mig -> 46 SMs, 448 GB/s).
#
# Numerics contract: inputs/weights torch.bfloat16 (the serving deploy
# precision — the seed carries the literal so the harness auto-detects it);
# GEMMs fp32-accumulate (ieee, no
# TF32) with a single round after bias (addmm semantics — the reference uses
# torch.addmm); silu/sigmoid computed on the rounded value; decode pointwise
# math in fp32; stacked output stored in the I/O dtype. Random head weights
# are scaled 0.05x so the ziln exp stays finite (served logits are small).

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROWS = 6 * 1024
TRUNK_W = 1328
TRUNK_RAW = 1322
TOWER_ORDER = ("iap", "adrev", "retention", "adrev_bce")
TOWER_DIMS: Dict[str, List[int]] = {
    "iap": [384, 384],
    "adrev": [192, 192],
    "retention": [256, 256],
    "adrev_bce": [128, 128],
}
TOWER_HEAD_COLS: Dict[str, int] = {"iap": 13, "adrev": 4, "retention": 6, "adrev_bce": 7}
PAYER_COL = 6  # main_payer_d7 within iap tower logits
RET_COL = 3    # main_retention_d7 within retention tower logits
GROUPS: Dict[str, Tuple[Tuple[str, ...], int]] = {
    "iap_main": (("iap", "retention"), 6),
    "adrev_main": (("adrev", "adrev_bce", "retention"), 3),
}
GROUP_IN: Dict[str, int] = {"iap_main": 659, "adrev_main": 593}
N_COMPONENTS = 2
ADREV_CALIBRATION = 1.053
OUT_COLS = 16

assert GROUP_IN["iap_main"] == sum(
    TOWER_DIMS[n][-1] + TOWER_HEAD_COLS[n] for n in GROUPS["iap_main"][0]
)
assert GROUP_IN["adrev_main"] == sum(
    TOWER_DIMS[n][-1] + TOWER_HEAD_COLS[n] for n in GROUPS["adrev_main"][0]
)


class Model(nn.Module):
    """Eager reference: the unfused per-op sequence exactly as forward_trunk
    runs it per ensemble component (addmm single-round, bf16 stage storage,
    fp32 decode math)."""

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        trunk = tensors[0]
        w = list(tensors[1:])
        sd = trunk.dtype

        per_component: List[torch.Tensor] = []
        wi = 0
        for _ in range(N_COMPONENTS):
            towers: Dict[str, torch.Tensor] = {}
            tlogits: Dict[str, torch.Tensor] = {}
            for name in TOWER_ORDER:
                x = trunk
                for _layer in range(2):
                    x = F.silu(torch.addmm(w[wi + 1], x, w[wi])).to(sd)
                    wi += 2
                towers[name] = x
                gate = torch.sigmoid(torch.addmm(w[wi + 1], x, w[wi]))
                wi += 2
                gated = (x * gate).to(sd)
                tlogits[name] = torch.addmm(w[wi + 1], gated, w[wi]).to(sd)
                wi += 2

            glogits: Dict[str, torch.Tensor] = {}
            for gname, (seq, _hcols) in GROUPS.items():
                gin = torch.cat(
                    [t for n in seq for t in (towers[n], tlogits[n])], dim=1
                )
                gate = torch.sigmoid(torch.addmm(w[wi + 1], gin, w[wi]))
                wi += 2
                gated = (gin * gate).to(sd)
                glogits[gname] = torch.addmm(w[wi + 1], gated, w[wi]).to(sd)
                wi += 2

            iap_d7 = glogits["iap_main"][:, 0:3]
            iap_d28 = glogits["iap_main"][:, 3:6]
            adrev_d0 = glogits["adrev_main"][:, 0:1]
            adrev_d7 = glogits["adrev_main"][:, 1:2]
            adrev_d28 = glogits["adrev_main"][:, 2:3]
            ret_d7 = tlogits["retention"][:, RET_COL : RET_COL + 1]
            payer_d7 = tlogits["iap"][:, PAYER_COL : PAYER_COL + 1]

            def ziln(t):
                t = t.float()
                prob = torch.sigmoid(t[:, 0:1])
                loc = 10.0 * torch.tanh(t[:, 1:2] / 10.0)
                scale = F.softplus(3.0 * torch.tanh(t[:, 2:3] / 3.0))
                value = torch.exp(loc + 0.5 * torch.square(scale))
                return prob, value, prob * value, loc, scale

            p7, v7, f7, l7, s7 = ziln(iap_d7)
            p28, v28, f28, l28, s28 = ziln(iap_d28)
            cols = [
                p7, v7, f7, l7, s7,
                p28, v28, f28, l28, s28,
                F.softplus(adrev_d0.float()),
                F.softplus(adrev_d7.float()),
                F.softplus(adrev_d28.float()),
                torch.sigmoid(ret_d7.float()),
                torch.sigmoid(payer_d7.float()),
            ]
            per_component.append(torch.cat(cols, dim=1))  # (rows, 15) fp32

        mean = torch.stack(per_component, dim=0).mean(dim=0)
        mean[:, 10:13] = mean[:, 10:13] * ADREV_CALIBRATION
        out = torch.cat([mean, torch.zeros(ROWS, 1, device=mean.device)], dim=1)
        return out.to(sd)  # (rows, 16)


def _component_weights(g: torch.Generator) -> List[torch.Tensor]:
    ws: List[torch.Tensor] = []
    for name in TOWER_ORDER:
        d_in = TRUNK_W
        for d_out in TOWER_DIMS[name]:
            ws += [
                torch.randn(d_in, d_out, generator=g) * (d_in**-0.5),
                torch.randn(d_out, generator=g) * 0.02,
            ]
            d_in = d_out
        d = TOWER_DIMS[name][-1]
        # GateLayer: square gate GEMM + sigmoid-mul.
        ws += [
            torch.randn(d, d, generator=g) * (d**-0.5),
            torch.randn(d, generator=g) * 0.02,
        ]
        # Tower heads linear. 0.05x keeps downstream ziln exp finite and
        # mirrors small served logits (these logits also join the group cat).
        ws += [
            torch.randn(d, TOWER_HEAD_COLS[name], generator=g) * (d**-0.5) * 0.05,
            torch.randn(TOWER_HEAD_COLS[name], generator=g) * 0.02,
        ]
    for gname, (_seq, hcols) in GROUPS.items():
        n = GROUP_IN[gname]
        ws += [
            torch.randn(n, n, generator=g) * (n**-0.5),
            torch.randn(n, generator=g) * 0.02,
        ]
        ws += [
            torch.randn(n, hcols, generator=g) * (n**-0.5) * 0.05,
            torch.randn(hcols, generator=g) * 0.02,
        ]
    return ws


def get_inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    trunk = torch.randn(ROWS, TRUNK_W, generator=g)
    trunk[:, TRUNK_RAW:] = 0.0  # kalign pad columns are exact zeros as served
    weights: List[torch.Tensor] = []
    for _ in range(N_COMPONENTS):
        weights += _component_weights(g)
    return [trunk] + weights


def get_init_inputs():
    return []
