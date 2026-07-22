# Verified dims + gate formula (deployed checkpoint + gate.py)

Source: `model.pth` state-dict shapes (`cross1.*`, `input_layer.epnet_module.*`)
and `model/user_value/layers/gate.py::GateNU` (v3_q3a config: epnet
gate_hidden_dim 512).

- Cross layer: DCNv2 low-rank confirmed — `V (1322, 128)`, `U (128, 1322)`,
  `bias (1322,)`; trunk width 1322.
- EPNet gate (GateNU): `linear1 (512, 1453)`, `linear2 (1322, 512)` —
  gate input = cat([domain_ft (131), x.detach() (1322)]) = 1453.
- Exact formula and ordering (the flagged VERIFY):

      layer1 = relu(linear1(cat([domain_ft, x_detached])))
      layer2 = sigmoid(linear2(layer1)) * gamma      # gamma = 2.0
      out    = layer2 * x                            # elementwise product

  i.e. sigmoid FIRST, then the 2.0 scale, then the elementwise product with
  the (grad-attached) embedding. The detach only matters for training; at
  inference cat -> relu -> sigmoid -> *2 -> hadamard is the served math.

Do NOT edit problem.py while an optimization run is live; apply after the
run, re-run test.py, re-verify the winner before integration.

**2026-07-22: applied.** `problem.py` / `input.py` / `test.py` now carry
these dims (D 2756 → 1322, gate input 1453, relu + both linear biases,
gamma 2.0, x0 ≡ xl ≡ y0 aliasing). The D=2756 round's winner (integrated as
`triton_cross_gate.py` on the UL branch, then removed after a flat profile)
does NOT transfer — re-run the search from the fresh seed.
