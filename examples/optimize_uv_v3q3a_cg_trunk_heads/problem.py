# Unity user-value model v3-q3a-cg — TRUNK TAIL: towers -> head linears ->
# ZiLN/Poisson/BCE decodes -> 2-component ensemble aggregate -> nn_output stack.
#
# Kernel dir #5: HORIZONTAL FUSION of the served head swarm. In the deployed
# roofline this region is the 60-callsite `..._softplus_..._where` family plus
# the 27-callsite variant plus dozens of tiny head addmm/stack launches — all
# at 8-16% SOL. Each op is "optimal for its size"; the win is running them as
# ONE (or few) wide kernels across heads x windows x ensemble components.
#
# CUDA-graph serving constraints (same contract as dir #4): fixed shapes,
# capture-safe (no per-call host work), fewer captured launches = faster
# replay. test.py enforces capture+replay correctness.
#
# Provenance (v3_q3a_cg config.json nn_config.model + model.py forward_trunk
# + head_module.ZilnHead.prob_value_pred, branch qi/partial-graph-split):
#   rows       = 24 * 1024 = 24576   (request bucket x max_candidates_bucket)
#   shared dim = 512 (fc_shared [512,512] silu output)  [+ skip concat --
#                skip width set to 64 here; VERIFY against
#                input_layer.skip_output_size() before integration]
#   towers: iap [384,384] swish; adrev [192,192] swish;
#           retention [192,192]; adrev_bce [192,192]
#           [retention/adrev_bce dims TRUNCATED in the config dump --
#            VERIFY nn_config.model.towers before integration]
#   main head groups:
#     iap_main  (input: iap ++ retention towers)  -> main_iap_d7 (ziln, 3),
#                                                    main_iap_d28 (ziln, 3)
#     adrev_main (input: adrev ++ adrev_bce ++ retention)
#                -> main_adrev_d0/d7/d28 (poisson, 1 each)
#   tower heads: main_retention_d7 (bce, 1, on retention tower),
#                main_payer_d7 (bce, 1, on iap tower)
#   ensemble: 2 components, aggregate avg; adrev heads use avg with
#   calibration factor 1.053 (multiplied after the mean, as served).
#
# Decodes (exact, from ZilnHead.prob_value_pred / forward_trunk):
#   ziln(t[:, 0:3]): prob = sigmoid(t0); scale = softplus(t2);
#                    value = exp(t1 + 0.5*scale^2); final = prob*value
#   poisson(t[:, 0:1]): value = softplus(t0)
#   bce(t[:, 0:1]):     prob  = sigmoid(t0)
#
# nn_output stack (16 cols, order from BaseModel._stack_served_outputs):
#   [iap_prob_d7, iap_value_d7, iap_final_d7, iap_loc_d7, iap_scale_d7,
#    iap_prob_d28, iap_value_d28, iap_final_d28, iap_loc_d28, iap_scale_d28,
#    adrev_d0_value, adrev_d7_value, adrev_d28_value, ret_prob_d7,
#    payer_prob_d7, 0.0 pad]
#
# Numerics contract: inputs/weights bf16; matmuls native-dtype operands with
# fp32 accumulate (ieee, no TF32), stage outputs stored bf16; the decode
# pointwise math runs in fp32 (sigmoid/softplus/exp) and the stacked output
# is stored in the I/O dtype. exp() of an unclamped loc overflows with random
# weights, so get_inputs scales head weights down — integration must keep the
# served clamp-free behavior (real logits are small).

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

ROWS = 24 * 1024
SHARED_DIM = 512
SKIP_DIM = 64  # VERIFY: input_layer.skip_output_size()
TOWER_DIMS = {
    "iap": [384, 384],
    "adrev": [192, 192],
    "retention": [192, 192],  # VERIFY
    "adrev_bce": [192, 192],  # VERIFY
}
N_COMPONENTS = 2
ADREV_CALIBRATION = 1.053
OUT_COLS = 16


def _tower_in() -> int:
    return SHARED_DIM + SKIP_DIM


class Model(nn.Module):
    """Eager reference: the unfused per-op sequence exactly as forward_trunk
    runs it per ensemble component (fp32 decode math, bf16 stage storage)."""

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        shared, skip = tensors[0], tensors[1]
        w = list(tensors[2:])
        sd = shared.dtype
        feats = torch.cat([shared, skip], dim=1)

        per_component: List[torch.Tensor] = []
        wi = 0
        for _ in range(N_COMPONENTS):
            towers = {}
            for name in ("iap", "adrev", "retention", "adrev_bce"):
                x = feats
                for _layer in range(2):
                    x = torch.matmul(x, w[wi]) + w[wi + 1]
                    x = F.silu(x)  # swish
                    x = x.to(sd)
                    wi += 2
                towers[name] = x

            iap_in = torch.cat([towers["iap"], towers["retention"]], dim=1)
            adrev_in = torch.cat([towers["adrev"], towers["adrev_bce"], towers["retention"]], dim=1)

            def head(x, weight, bias):
                return (torch.matmul(x, weight) + bias).to(sd)

            iap_d7 = head(iap_in, w[wi], w[wi + 1]); wi += 2      # (rows, 3)
            iap_d28 = head(iap_in, w[wi], w[wi + 1]); wi += 2     # (rows, 3)
            adrev_d0 = head(adrev_in, w[wi], w[wi + 1]); wi += 2  # (rows, 1)
            adrev_d7 = head(adrev_in, w[wi], w[wi + 1]); wi += 2
            adrev_d28 = head(adrev_in, w[wi], w[wi + 1]); wi += 2
            ret_d7 = head(towers["retention"], w[wi], w[wi + 1]); wi += 2  # (rows, 1)
            payer_d7 = head(towers["iap"], w[wi], w[wi + 1]); wi += 2      # (rows, 1)

            def ziln(t):
                t = t.float()
                prob = torch.sigmoid(t[:, 0:1])
                scale = F.softplus(t[:, 2:3])
                value = torch.exp(t[:, 1:2] + 0.5 * torch.square(scale))
                return prob, value, prob * value, t[:, 1:2], scale

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
    for name in ("iap", "adrev", "retention", "adrev_bce"):
        d_in = _tower_in()
        for d_out in TOWER_DIMS[name]:
            ws += [torch.randn(d_in, d_out, generator=g) * (d_in**-0.5), torch.randn(d_out, generator=g) * 0.02]
            d_in = d_out
    iap_in = TOWER_DIMS["iap"][-1] + TOWER_DIMS["retention"][-1]
    adrev_in = TOWER_DIMS["adrev"][-1] + TOWER_DIMS["adrev_bce"][-1] + TOWER_DIMS["retention"][-1]
    # 0.05x: keeps ziln exp() finite with random inputs (served logits are small).
    for d_in, d_out in [(iap_in, 3), (iap_in, 3), (adrev_in, 1), (adrev_in, 1), (adrev_in, 1),
                        (TOWER_DIMS["retention"][-1], 1), (TOWER_DIMS["iap"][-1], 1)]:
        ws += [torch.randn(d_in, d_out, generator=g) * (d_in**-0.5) * 0.05,
               torch.randn(d_out, generator=g) * 0.02]
    return ws


def get_inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    shared = torch.randn(ROWS, SHARED_DIM, generator=g)
    skip = torch.randn(ROWS, SKIP_DIM, generator=g)
    weights: List[torch.Tensor] = []
    for _ in range(N_COMPONENTS):
        weights += _component_weights(g)
    return [shared, skip] + weights


def get_init_inputs():
    return []
