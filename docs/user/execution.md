# Execution Modes and Capacity

`streamed_edges="direct"` and `execution_profile="capacity"` are the default
for Kokkos evaluation. The model precision defaults to FP32. Two-interaction
direct execution requires an artifact matching that precision and fails closed
rather than silently choosing a different algorithm. Two-interaction direct
execution refuses graph preparation until a qualified R1 artifact is loaded and
reports one of four artifact causes: an empty artifact cache with a working
compiler, an unavailable compiler, a missing prebuilt artifact, or a rejected
stale artifact. Generated direct execution is the only direct implementation for
standard MACE energy and forces; the `reference` routing remains available as
the in-evaluator reference, and is the implementation the removed standalone
CPU-native evaluators were consolidated onto. Execution observers and parameter gradients are admitted
on that compiled reference routing in direct mode and do not require an R1
artifact. Admitted single-layer
models have no R1 stage and use built-in direct execution. Their capacity plans
may still compile or load specialized M0/M1/R0 operator modules.

CUDA artifact preparation compiles independent R1, M0, M1, and R0 modules in
parallel when they are all needed. Each NVRTC compilation uses two internal
split-compile threads; at most two outer compiler jobs run concurrently, and
module loading and validation remain ordered as R1, M0, M1, then R0.

On CPU, low-memory preparation compiles generated R1, M0, and M1 host plugins
concurrently with at most two compiler processes. Loading and native validation
remain ordered as R1, M0, then M1.

Generated operator resolution normally tries a matching artifact installed
with the backend package, then a validated user-cache artifact, then runtime
generation. There is one temporary exception while the frozen M0/R0 retirement
gates remain open: for an eligible low-memory M0 or R0 node whose frozen
built-in is admitted, the shadow-qualification path tries an exact installed
artifact first when one is available, but may choose the built-in after an
installed miss or rejection, before consulting cache/runtime. It may also
choose the built-in immediately when that package mode installs no candidate
for the stage. Generic execution is used only when the explicitly selected
execution mode permits it; direct execution does not silently change
algorithms when specialization is unavailable. Installed and runtime-generated
artifacts use the same canonical implementation and identity. Their origin is
reported separately as `prebuilt`, `cache`, or `runtime`; the frozen built-in
has a null artifact origin.

Official CPU and CUDA packages include the six verified MACE R0 artifacts
(three foundation-model channel shapes in FP32 and FP64).
OAM wheel inclusion is pending post-merge repaired-wheel build/install gates
and the final release decision; the release build plan contains eight known
S/M/L/XL generated architectures in FP32 and FP64. HIP packages include only
the MACE artifacts.
Published packages do not include the larger optional MACE R1/R0/M0/M1 bundle.
OAM selection compares the exact
resolved layer, irrep/channel, angular, tensor-product, radial, and
normalization descriptor; paths and learned values are not selection keys.
Models with an uncommon architecture or weighted-edge contract compile the
exact missing artifacts on first use and reuse them from
`SYMMETRIX_JIT_CACHE`; this is supported behavior, not a compatibility
fallback. A runtime compiler is still required unless every required stage has
an exact installed artifact or was prepared earlier.

R0 diagnostics describe four independent questions. `r0_implementation` is
`generic` or `generated`; `r0_backend` is `host`, `cuda`, or `hip`;
`r0_artifact_origin` is `prebuilt`, `cache`, `runtime`, or null; and
`r0_schedule`, `r0_profile`, and `r0_module_id` identify the selected generated
schedule and canonical artifact. Do not treat origin or a loader filename as
an implementation name. The temporary frozen R0 qualification oracle has no
artifact origin and remains in the package because the CPU and HIP retirement
gates remain open. The exact-current CUDA retirement campaign has passed.

Prepare a direct host artifact before a reproducible CPU deployment:

```bash
export SYMMETRIX_JIT_CACHE=/path/to/jit-cache
symmetrix_prepare_jit_host_artifact \
    --model srtio3-mace.json --precision float32
```

For explicit `dtype="float64"` evaluation, prepare the corresponding artifact
with `--precision float64`.

Use `streamed_edges="non-compiled"` only for compiler-free debugging or
compatibility checks. It is not optimized, has no performance guarantee, and is
not the primary execution path.

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
`allow_fixed_workspace=True` opt-in, currently require CUDA, and support FP32
and FP64 ordinary MH0 standard MACE. MACEField and MACE-MH-1 do not support
fixed-workspace plans. They are excluded by default. In LAMMPS MPI, the
ordinary two-layer standard-MACE plan uses the existing forward-H1 and
reverse-H1-adjoint exchanges between its three tiled phases. Its receiver and
edge workspace is bounded, while the hidden state remains proportional to the
local-plus-ghost feature count. Two-layer execution requires a
precision-matched generated device artifact with tiled R1 support.

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
`capacity` and `speed`. Every documented streamed-edge spelling resolves
through one explicit alias table before the native call: `direct` (with the
compatibility aliases `factorized` and `direct_streamed`),
`non-compiled` (with the deprecated alias `generic` and the compatibility
alias `all_interactions`), `materialized`, and the temporary compatibility
request `auto`. `factorized` and `direct_streamed` alias `direct` and change an
otherwise-`capacity` request to the old `speed` behavior; use literal `direct`
for capacity behavior. `materialized` is the original Symmetrix execution path
used by the original Symmetrix model format (named v1 here). The public mode and
backend matrix is in
{doc}`/reference/execution_support_matrix`; implementation architecture is
described in {doc}`/streamed_edge_execution`.

The graph-time execution-plan report currently covers MH-0 MACE and MACEField.
MH-1 direct execution has its own qualified policies but does not yet expose
the corresponding planner report.

NequIP-OAM exposes the same `execution_profile="speed"` and `"capacity"`
request spelling through `NequIPCalculator`. Both use generated execution and
the same exact arena planner, select the sole `full_message` edge variant, and
retain no more memory in capacity than in speed. Their plan identifiers are
`oam-direct-speed` and `oam-direct-capacity`; current release qualification is
pending for both after the artifact-contract change. `oam-fixed-workspace-v1`
is deliberately unavailable until its separate matrix passes. Direct execution
fails closed on a stale ABI, missing capability, wrong target, or missing exact
artifact; the artifact-free `reference` route remains the same-spline
diagnostic oracle.
