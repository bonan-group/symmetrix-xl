# Changelog

All notable changes to Symmetrix-XL are documented in this file.

## 0.2.0 - 2026-10-25

### Execution and JIT

- Refactored every model family onto a declared registry and exactly two
  execution routings. `direct` executes a contract-bound generated artifact
  and fails closed when the artifact cannot be prepared; `reference` executes
  the generic implementation without an artifact. Removed the standalone
  serial MACE and MACE-MH-1 evaluators, the serial LAMMPS pair style, the
  `use_kokkos` argument, and silent JIT fallback paths.
- Made artifact identities bind the generated architecture, scalar width,
  backend, target architecture, compiler/toolchain, source-content hashes, ABI,
  and generation version. Runtime caches retain the private-directory trust
  policy; prebuilt artifacts are startup optimizations rather than correctness
  fallbacks. Learned weights and model paths are not artifact-selection keys.
- Added generated host and CUDA M0, R0, R1, and M1 operators and generation-19
  catalogs. Generated M1 passed CUDA parity and routing qualification on the
  exact 5,000-atom MACE-OMAT-0 medium workload with zero fallback. Frozen R0
  remains available while CPU and HIP retirement gates are open.
- Unified MACE-MH-1 symmetric-product contractions with ordinary MACE's
  canonical generated operators: non-final layers use M0 and the final scalar
  layer uses M1. MH-1 retains its model-specific nonlinear stages, while every
  product layer reports its own exact artifact identity and origin.
- Removed the compiled `team_cached` and `tiled_coupling` R1 strategies and
  runtime executor overrides. Generated execution is the only `direct` route
  for standard two-interaction MACE energy and forces; the in-evaluator serial
  reference remains an internal numerical oracle.

### NequIP-OAM support

- Added the OAM v2 JSON contract, `NequIPCalculator`, and
  `symmetrix_extract_nequip`. JSON-only evaluation depends on ASE and the native
  Symmetrix backend; PyTorch, NequIP, and e3nn are optional exporter/reference
  dependencies. This is support for the OAM S/M/L/XL graph family, not an
  arbitrary NequIP graph compiler.
- Added a release build plan for FP32/FP64 OAM-S/M/L/XL artifacts in CPU and
  CUDA wheels. Loading requires an exact match of the resolved layer,
  irrep/channel, angular-order, tensor-product, radial, target, and
  floating-point-policy identity; custom architectures continue to use runtime
  generation when a compiler is available. Repaired-wheel inclusion remains a
  release gate.
- Added full-message-only generated host ABI-v7 and CUDA ABI-v3 artifacts
  with energy, forces, and stress. The artifact-free `reference` route serves
  energy and forces. Fresh correctness, performance, capacity, and repaired-wheel
  qualification remain release gates for the current artifact contracts.
- Added OAM-XL graph support, including degree-four real harmonics, analytic
  derivatives, and ZBL. Live upstream-reference FP64 parity, HIP, multi-GPU,
  and LAMMPS/MPI remain outside the current qualified boundary.

### Models, properties, and backends

- Added frozen-weight alchemical message mixing for standard two-interaction
  compact MACE models, including Agnesi radial functions, A0/A1 density
  conditioning, ZBL, energy, forces, stress, and reusable coefficient updates.
  CPU/OpenMP and `gfx1151` HIP FP32/FP64 parity is qualified; CUDA alchemical
  execution remains outside that record.
- Published KokkosKernels TPL configuration to the native targets so compiled
  accelerator batched-BLAS routes and runtime capability reporting agree.
- Retained architecture-qualified CPU, CUDA, and HIP backend packages with
  exact frontend-version dependencies and source/native provenance. HIP support
  for ordinary MACE remains available, but NequIP-OAM HIP execution is not part
  of this release.

### Migration

- Remove `use_kokkos` from `Symmetrix(...)` and `NequIPCalculator(...)`. Use
  `streamed_edges="direct"` for generated execution or
  `streamed_edges="reference"` for the artifact-free generic route.
- Re-export original Symmetrix pair-spline (v1) models with
  `radial_format="compact"`. For LAMMPS, replace `pair_style symmetrix/mace`
  with a Kokkos spelling and rerun `pair_symmetrix/install.sh`.
- Regenerate JIT artifacts created before generation 19; incompatible cached
  artifacts are rejected rather than reused.

## 0.1.1 - 2026-09-22

- Added automatic installation of architecture-qualified accelerator backends.
- Improved LAMMPS MPI execution, including fixed-workspace and FP64 support.
- Hardened periodic neighbor-graph indexing for large flattened edge arrays.
- Updated public documentation and qualified performance summaries.

## 0.1.0 - 2026-09-16

Initial public Symmetrix-XL release.

- Added streamed-edge MACE and MACEField execution for reduced graph workspace.
- Added model-specialized host, CUDA, and HIP execution with runtime-compiled
  artifacts.
- Added separately installable CPU, CUDA, and HIP backend packages with runtime
  diagnostics.
- Added ASE calculators, ensemble evaluation, prediction heads, and LAMMPS pair
  styles.
- Added reproducible correctness, capacity, performance, and backend
  qualification harnesses.
