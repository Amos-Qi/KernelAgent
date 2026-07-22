# UV v3-q3a-cg — cross layer + EPNet gate epilogue fusion (kernel dir #6)

The **GEMM-epilogue fusion** problem: serving's #2 and #3 kernels overall are
the separate pointwise epilogues of the low-rank cross layer and the EPNet
gate — `add_addmm_mul` (4.69 ms, 90% Mem SOL, **1% L2 hit**) and
`mm_mul_sigmoid` (2.89 ms, 87–90% Mem SOL). Both already run at the memory
roof; per-kernel tuning is exhausted by construction. The only win is BYTES:

- fuse each epilogue into the producing GEMM's store (write-once, no
  (rows, 1322) round-trip);
- keep the `x` / `y0` tile register- or L2-resident across the elementwise
  re-read — the deployed epilogue's 1% L2 hit rate IS the opportunity;
- optionally fuse gate-output into the cross's down-GEMM (both consume `y0`).

Byte math at rows=24576, D=1322, bf16, with the served x0 ≡ xl ≡ y0
aliasing (the cross epilogue reads ONE 62 MB stream, not two): each fused
epilogue removes ~124 MB of DRAM traffic (the intermediate's write+read
round-trip) per component per forward, ~496 MB across gate+cross × 2
components — ≈ 280 µs at a ~1.8 TB/s roof, worth ~4–5% of the ~6.2 ms
forward. A clean fusion must WIN AT THE GEMM TOO: the replaced inductor
template GEMMs run at 70–88% SOL, so a fused kernel that sacrifices GEMM
throughput for epilogue locality loses net (measured on the first
integration attempt).

## ⚠ 2026-07-22 re-seed at VERIFIED dims

This dir originally ran the search at **D = 2756 — a stale config-dump
width, 2.08× the real trunk (1322)** — with a silu/no-bias gate and a
two-stream cross epilogue. That round's winner was integrated
(`deploy/triton_cross_gate.py` on UL branch `qi/v3-q3a-cuda-graph`,
commit `6e1bfd374`, since removed) and profiled **flat**: its kernels ran
at 39–66% SOL (underutilized) at the true shape. `problem.py`, `input.py`
and `test.py` now carry the verified geometry (VERIFIED_DIMS.md): D=1322,
gate input 1453 = cat([dom 131, x]), relu hidden with both linear biases,
gamma 2.0, and the x0 ≡ xl aliasing. **Prior winners are invalid; re-run
the search from the fresh seed before any re-integration.**

## CUDA-graph contract (as dirs #4/#5)

Fixed shapes, capture-safe, no per-call host work; `test.py` gates numeric
parity AND capture/replay correctness.

## Provenance and what to VERIFY before integration

Shapes, the gate formula (relu hidden, sigmoid → ×2.0 → hadamard) and the
gate→cross ordering are VERIFIED against the deployed checkpoint and module
source (VERIFIED_DIMS.md; `layers/gate.py::GateNU`,
`layers/cross.py::CrossLayerV2`). Before integrating a winner, re-verify
only that the target branch's modules haven't drifted from those files.

## Running on the VM

```bash
cd ~/KernelAgent && git pull --ff-only origin v3-q3a-opt
source .venv/bin/activate
cd examples/optimize_uv_v3q3a_cg_cross_gate && python test.py && cd ..   # must PASS

nohup python run_opt_manager.py --kernel-dir optimize_uv_v3q3a_cg_cross_gate \
    --strategy beam_search_uv --max-rounds 5 > opt_run6.log 2>&1 &
```

A winner integrates at `forward_flat`'s cross1 + the EPNet gate on the UL
branch — like dir #5, it speeds the whole-cg replay and the split's trunk
identically. Strict A/B + diff read + shadow parity after integration.
