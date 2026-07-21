# UV v3-q3a-cg — cross layer + EPNet gate epilogue fusion (kernel dir #6)

The **GEMM-epilogue fusion** problem: serving's #2 and #3 kernels overall are
the separate pointwise epilogues of the low-rank cross layer and the EPNet
gate — `add_addmm_mul` (4.69 ms, 90% Mem SOL, **1% L2 hit**) and
`mm_mul_sigmoid` (2.89 ms, 87–90% Mem SOL). Both already run at the memory
roof; per-kernel tuning is exhausted by construction. The only win is BYTES:

- fuse each epilogue into the producing GEMM's store (write-once, no
  (rows, 2756) round-trip);
- keep the `x` / `x0` tile register- or L2-resident across the elementwise
  re-read — the deployed epilogue's 1% L2 hit rate IS the opportunity;
- optionally fuse gate-output into the cross's U-GEMM (both consume `y0`).

Byte math at rows=24576, D=2756, bf16: each eliminated full-activation
round-trip saves ~2×135 MB of DRAM traffic per forward — against a ~measured
7.6 ms epilogue budget, a clean fusion is worth 3–5% of total kernel time.

## CUDA-graph contract (as dirs #4/#5)

Fixed shapes, capture-safe, no per-call host work; `test.py` gates numeric
parity AND capture/replay correctness.

## Provenance and what to VERIFY before integration

Shapes and the epilogue patterns are from the deployed profile + config
(D=2756 input-layer width, cross rank 128, gate hidden 512). Verify against
the EPNet module before integrating a winner: the exact gate formula (the
2× factor, silu hidden activation) and gate-vs-cross ordering — the fusion
structure and byte math are identical either way, but the integrated code
must match serving numerics exactly.

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
