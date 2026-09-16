# Execution Modes and Capacity

`streamed_edges="direct"` and `execution_profile="capacity"` are the default
for Kokkos evaluation. The model precision defaults to FP32. Direct execution
requires an artifact matching that precision and fails closed rather than
silently choosing a different algorithm.

Prepare a direct host artifact before a reproducible CPU deployment:

```bash
export SYMMETRIX_JIT_CACHE=/path/to/private/jit-cache
symmetrix_prepare_jit_host_artifact \
    --model srtio3-mace.json --precision float32
```

For explicit `dtype="float64"` evaluation, prepare the corresponding artifact
with `--precision float64`.

Use `streamed_edges="non-compiled"` only as a compiler-free fallback or diagnostic;
it has no performance guarantee and is not the primary execution path.

`capacity` ranks qualified internal MH-0 plans using an advisory device-memory
estimate. It chooses the fastest estimated fit; when no candidate fits, it
attempts the smallest qualified plan and lets the allocator determine final
feasibility. `execution_profile="speed"` fixes the backend-qualified throughput
plan without considering available memory. For standard compact MACE, that
plan retains harmonic gradients, reuses eligible MH-0 state, and recomputes
readout state. For larger graphs, CPU and CUDA FP32 select Y-only harmonics
when that qualified path is faster. FP32 also uses compact unit-direction
geometry. FP64 retains Cartesian geometry; CUDA FP64 and HIP retain harmonic
gradients.
Bounded fixed-workspace plans require the separate
`allow_fixed_workspace=True` opt-in and are excluded by default.

```python
from ase.spacegroup import crystal
from symmetrix import Symmetrix

calc = Symmetrix(
    "srtio3-mace.json",
    execution_profile="capacity",
    allow_fixed_workspace=True,
)
a = 3.905
atoms = crystal(
    symbols=["Sr", "Ti", "O"],
    basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
    spacegroup=221,
    cellpar=[a, a, a, 90, 90, 90],
)
atoms.calc = calc
atoms.get_forces()  # prepares the graph
print(calc.execution_plan)  # MH-0 graph-time report
```

`low_memory=True` and `low_memory=False` remain compatibility spellings for
`capacity` and `speed`. `generic` and `materialized` are explicit temporary
compatibility modes. The full support matrix, diagnostics, and private
development plan pins are in {doc}`/streamed_edge_execution`.

The graph-time execution-plan report currently covers MH-0 MACE and MACEField.
MH-1 direct execution has its own qualified policies but does not yet expose
the corresponding planner report.
