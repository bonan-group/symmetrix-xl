# Changelog

All notable changes to Symmetrix-XL are documented in this file.

## 0.2.0 - Unreleased

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
- `prepare_jit_device_artifact --lammps-arguments` prints the M0 and R0 device
  modules' own persistent block counts and the R0 schedule
  (`jit_m0_device_blocks_per_compute_unit`,
  `jit_r0_device_blocks_per_compute_unit`, `jit_r0_device_schedule`), and the
  pair style passes them to those loaders only. Before, R0 received R1's count
  and the documented flow stopped at the R0 load ("persistent-block profile
  does not match the generated backend schedule"). A fragment without the new
  keywords behaves as before.
- Fixed run-to-run variation of stresses and edge forces for A0-scaled models
  in the capacity profile with the generated M0: the R0 reverse prepare now
  reduces the A0 scale adjoint in a fixed order instead of the generated M0
  adding it with an FP64 atomic. The built-in M0 is unchanged.
- Fixed periodic calculations with the factorized observer or execution
  parameter gradients where the native Kokkos neighbor builder is selected
  (CUDA and HIP at 512 atoms or more, or `SYMMETRIX_NEIGHBOR_BACKEND=kokkos`):
  they now build the graph on the host instead of stopping with `Execution R1
  source schedule has not been prepared`, and enabling either after an
  evaluation rebuilds the topology instead of reusing a released graph token.
- Fixed the Kokkos pair style accepting fixed-workspace tiled plans under
  `no_domain_decomposition` (`allow_fixed_workspace yes` or a tiled
  `_debug_execution_plan`), which then failed during `run`. `pair_style` now
  refuses them and names `mpi_message_passing`, and the generic evaluation
  entry refuses edge vectors under an active tiled plan unless they are the
  registered geometry or the prepared workspace.
- Fixed the Kokkos pair style under `pair_style hybrid/overlay/kk` with
  `mpi_message_passing`, the default on more than one MPI rank: the wrapper
  kept the message sizes it had before `pair_coeff` set this style's, LAMMPS
  sized the host communication buffers from them, and the per-atom message
  passing wrote past those buffers (a crash in CPU Kokkos builds).
  `init_style()` now raises the wrapper's sizes to this style's.
- Fixed the default float32 calculator route (`streamed_edges="direct"` with
  compact edge geometry) on frames without edges, such as an isolated atom or
  atoms farther apart than the cutoff: it raised "Spherical-harmonic
  coordinates do not cover all edge radii." instead of returning the atomic
  energies. The float64 and reference routes already evaluated these frames.
- Fixed the direct route on a graph without receivers, as in the first run of
  a box created without atoms or on an MPI rank that owns no atom and no
  ghost: H1 kept no trailing extents, the R1 launch stopped with "R1 tensor
  shape is inconsistent with the plugin contract", and the run aborted. H1 and
  its adjoint now keep the model's trailing extents with zero rows.
- Fixed the Kokkos pair style on type-filtered neighbor lists, where
  `pair_style hybrid` or `hybrid/overlay` gives the pairs of some atom types to
  another sub-style and the list skips those atoms. `no_domain_decomposition`
  indexed each neighbor by its local atom instead of its receiver row, which
  read the wrong features or crashed whenever a skipped atom came before a
  MACE atom; it now maps atoms to receiver rows. `mpi_message_passing`, which
  needs every owned atom in its list, stops with an error naming the cause
  instead of writing past its receiver rows.
- Fixed the prepared `no_domain_decomposition` route of the Kokkos pair style
  (`streamed_edges direct`) after an atom type changes without a neighbor list
  build, as in `fix sgcmc`: it kept the graph built for the old types and
  returned the energy and forces of the old species. The pair style now
  compares the receivers' types with the graph's on each step and rebuilds the
  graph when one differs.
- Fixed the Kokkos and Polar pair styles after a re-issued `pair_style` line
  changes their options following `pair_coeff`. LAMMPS passes such a line to
  the existing pair style, which kept the model loaded with the earlier
  options: a new `streamed_edges`, `profile`, `head` or artifact was ignored
  silently, and a change of the domain mode crashed, for
  `no_mpi_message_passing` also after a new `pair_coeff`. The next run now
  stops with an error; issue `pair_coeff` again after the change. An identical
  line, or one that changes only the electric field components or the Polar
  charge, spin, Fermi level or electrostatics settings, still needs none.
- Fixed `pair_coeff` of the Kokkos pair style terminating the process when
  the model cannot be loaded (an unreadable file, a malformed model or an
  unknown `head`). It now reports a LAMMPS error, which a Python `lammps` host
  can catch.
- Fixed both pair styles reading positions and returning energies in the
  wrong units, without a message, under unit styles other than `metal`. They
  now require `units metal` and stop with an error otherwise.

### NequIP-OAM support

- Bound runtime OAM artifact preparation to the backend's verified full source
  fingerprint in cache keys and `_src<fullhash>` artifact basenames. The three
  `prepare_nequip_oam_{cuda,host,receiver_host}_artifact` adapters require the
  caller's explicit `native_module`; normal frontend calls already provide
  it. Strict installed-prebuilt behavior is unchanged, and compatible learned
  weights still share artifacts within one package identity. Kernels, model
  architecture, ABI and numerical tolerances are unchanged.
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

### Dispersion corrections

- Added native DFT-D2 and two-body DFT-D3 through `D2Correction` and
  `D3Correction`, including energy, per-atom energies, forces, and stress.
  D3 supports zero, Becke-Johnson, modified-zero, and modified-Becke-Johnson
  damping. Corrections own independent reusable neighbor graphs and do not
  require PyTorch or torch-dftd at runtime; three-body ATM is not implemented.
- Added reciprocal-space `FourierD3Correction` for fully periodic systems,
  with a short coordination graph, low-rank particle-mesh evaluation,
  analytic forces and stress, and reusable FFT plans and workspaces. FourierD3
  is a modified two-body BJ-damped model, not a numerically interchangeable
  replacement for finite-cutoff D3, and never falls back to real-space D3.
- Added shared particle-mesh infrastructure with backend-selected FFTW on
  CPU, cuFFT on CUDA, and hipFFT on HIP. FourierD3 requires the matching FFT
  provider; implementation availability does not replace architecture-specific
  runtime qualification.

### Models, properties, and backends

- Added native MACE-POLAR-1 evaluation for polarizable MACE models, including
  energy, forces, stress, and total dipole, with total charge, total spin,
  Fermi level, and external electric field as graph inputs. Generated `direct`
  and artifact-free `reference` backbones compose with the native Polar
  forward and reverse program. Nonperiodic systems use exact real-space
  electrostatics; periodic systems support direct reciprocal electrostatics.
  PME implementation and automatic selection exist but remain outside the
  qualified public release contract. Distributed Polar LAMMPS is unsupported;
  the Polar pair style
  currently requires one MPI rank owning the complete system.
- Added specialized Polar-S `direct` execution for the one-interaction,
  retained scalar-only backbone with a linear energy readout, alongside the
  artifact-free `reference` route. Generated forward and seeded reverse use
  the validated layer contract; unsupported layouts fail explicitly. Polar
  S/M/L HIP `gfx1151` source-build/runtime evidence covers FP32/FP64,
  `reference` and `direct`, single-GPU molecule, slab, and bulk
  energy/forces/stress. HIP wheels, other HIP architectures,
  multi-GPU/distributed execution, PME publication, and distributed Polar
  LAMMPS remain outside this release contract.
- Corrected the generated retained-state bias-readout reverse and Polar-S
  reference retained-feature/skin-graph handling. Strict frozen Polar oracle
  comparisons explicitly use 4096 radial spline points; the public 256-point
  default and frozen numerical tolerances are unchanged.
- Added native MACELES evaluation and checkpoint export for two-interaction
  MACE models with learned scalar charges and intrinsic LES electrostatics.
  Both execution routings evaluate the complete fitted model, including charge
  derivatives in forces and stress, and expose `les_energy`,
  `mace_les_energy`, and `latent_charges`. Fully periodic and nonperiodic
  systems are supported; mixed periodicity, fixed-workspace plans, alchemical
  mixing, distributed evaluation, and LAMMPS are outside the supported boundary.
  checkpoint-free qualification passed on CPU/OpenMP and HIP `gfx1151` for
  FP32/FP64, reference/direct routing, periodic and nonperiodic structures,
  energy, forces, stress, and latent charges. HIP remains experimental and
  source-build/runtime only; the v0.2.0 decision keeps MACELES single-GPU.
- Added frozen-weight alchemical message mixing for standard two-interaction
  compact MACE models, including Agnesi radial functions, A0/A1 density
  conditioning, ZBL, energy, forces, stress, and reusable coefficient updates.
  CPU/OpenMP and `gfx1151` HIP FP32/FP64 parity is qualified; CUDA alchemical
  execution remains outside that record.
- Published KokkosKernels TPL configuration to the native targets so compiled
  accelerator batched-BLAS routes and runtime capability reporting agree.
- Fixed two Clang build stops. The Polar Kokkos pair style instantiated its
  templates outside `namespace LAMMPS_NS`, which Clang rejects, so every
  HIP (and other Clang) LAMMPS build stopped there. `-fbracket-depth=1024`,
  needed by the several-hundred-operand standard M0 folds, was passed to
  Apple builds only; it now reaches every Clang build, including ROCm 7.1
  (clang 20). GCC and nvcc builds are unchanged.
- Retained architecture-qualified CPU, CUDA, and HIP backend packages with
  exact frontend-version dependencies and source/native provenance. HIP support
  for ordinary MACE remains available, but NequIP-OAM HIP execution is not part
  of this release.
- Fixed MACE export for checkpoints whose stored symmetric-contraction U
  matrices differ in sign from the recomputed ones, such as MACE-MP-0b2-large
  (`mace-large-density-agnesi-stress.model`). The exporter takes each recomputed
  U column's sign from the checkpoint's `U_matrix_<nu>` buffer and refuses a
  buffer that matches neither sign. Exports of checkpoints that already
  matched are unchanged.
- Fixed MACE export of options the JSON cannot represent. A non-linear readout
  gate other than SiLU (for example `tanh`) or a joint embedding
  (`embedding_specs`) was exported silently and evaluated a different model;
  `NonLinearBiasReadoutBlock` lost its biases; a missing gate,
  `use_agnostic_product`, `use_last_readout_only` and plain `MACE` models
  failed with an `AttributeError` or `IndexError`. The exporter now refuses
  each with a `ValueError` that names the option. Supported models export
  unchanged.
- Fixed MACE export of models with correlation order 1 or 2 in any layer,
  including mixed orders such as `correlation=[3, 2]`, which failed with
  `IndexError: index 1 is out of range`. Orders above 3 are now refused with
  a message instead of failing or exporting a truncated product, and so are
  mixed orders in a multi-head model, which MACE's head selection cannot
  rebuild. Order-3 exports are unchanged.
- Fixed pair-spline export (`radial_format="pair-splines"`) of models with
  `apply_cutoff=False`: MACE then returns the polynomial cutoff separately and
  multiplies it into the radial weights and edge densities, but the exporter
  dropped it, so the R0, R1, A0 and A1 tables did not vanish at the cutoff
  (forces off by 2e-2 eV/A on a small test model, 0.46 eV/A with density
  blocks). The tables now include the cutoff. Exports of models with the
  cutoff in the radial basis are unchanged.

### CUDA packaging

- Added lazy external cuFFT loading in place of a direct cuFFT link or
  bundled vendor payload. Normal installs still require the matching NVIDIA
  cuFFT package; this reduces the Symmetrix wheel itself, not the total
  dependency download. FFT requests and explicit runtime probes load cuFFT;
  non-FFT execution can proceed when cuFFT is unavailable.
- A CUDA 13/sm120 development wheel passed FP32/FP64 FFT/particle-mesh checks
  and the cuFFT dependency/content audit. Final integrated release wheels,
  CUDA 12 execution, and portable manylinux repair remain unqualified; the
  development wheel is not a publication artifact.

### Migration

- Remove `use_kokkos` from `Symmetrix(...)` and `NequIPCalculator(...)`. Use
  `streamed_edges="direct"` for generated execution or
  `streamed_edges="reference"` for the artifact-free generic route.
- Re-export original Symmetrix pair-spline (v1) models with
  `radial_format="compact"`. For LAMMPS, replace `pair_style symmetrix/mace`
  with a Kokkos spelling and rerun `pair_symmetrix/install.sh`.
- Regenerate JIT artifacts created before generation 19; incompatible cached
  artifacts are rejected rather than reused.

## 0.1.2 - 2026-11-04

- Added an opt-in Apple Metal backend for FP32 standard-MACE execution on
  Apple silicon, with installation guidance, examples, and runtime
  qualification tools.
- Added native response and capacity execution for MACEField models, including
  electric-field observables, polarization, Born effective charges,
  polarizability, trusted execution-JIT cache handling, and streamed-edge
  coverage.
- Added source provenance to PEP 517 builds and expanded backend diagnostics,
  build qualification, and runtime checks.
- Added extractor support for forwarding `--num-spline-points` to
  MACE_Nonlinear exports.
- Expanded CPU and CUDA wheel builds and hardened release automation for
  reruns, ABI-tagged extensions, compound manylinux tags, and AVX-512 codegen
  checks.
- Refreshed user and developer documentation, paper benchmarks, citations, and
  benchmark guidance; removed superseded benchmark drivers.
- Improved CUDA execution and its qualification workflow.
- Hardened host OpenMP execution by blocking unsafe worker GEMMs, selecting the
  compatible OpenBLAS runtime, and fixing BLAS/LAPACK wiring.
- Model JSON format version 4 for every exported variant: the first readout and
  the H1->H2 skip are stored without the inverse of `interactions[1].linear_up`
  (`h1_affine_mode` `m0_composite` or `pre_linear_up`). Format-1/2 files still
  load with the old arithmetic and print a warning with an fp32 cancellation
  estimate computed from the file; `symmetrix_recompose_mace_json` converts
  them. Loading an old file does not opt it into the improved arithmetic:
  re-exporting the original MACE checkpoint to v4 is strongly recommended.
  Use `symmetrix_recompose_mace_json` only when the original checkpoint is not
  available.
- In the 192-atom MACE-MPA-0 water validation against the original fp64 e3nn
  model, a v0.1.2 v4 export measured maximum fp32 force deviations of
  `6.24e-5 eV/A` with 256 radial spline points and `1.98e-6 eV/A` with 512
  points. The v0.1.1 LAMMPS comparison in
  [Liu et al., arXiv:2610.23371v1](https://arxiv.org/abs/2610.23371v1)
  reported `1.6e-4 eV/A` for its legacy 256-point export; its scripts and input
  structure are archived in the
  [`mace-million` paper snapshot](https://github.com/ShiQiaoL/mace-million/tree/paper-d572103).
  The v4 fp32 result with 256 points is already small, and the corresponding
  fp64 256-point results agree (`6.15e-5 eV/A` here and `6.1e-5 eV/A` in the
  paper of Liu et al.). Because the fp32 error decreased while retaining the
  same 256-point grid, the larger legacy fp32 deviation was primarily caused by
  the inverse-based H1 affine arithmetic removed by format v4, rather than by
  the radial-grid density. Increasing the grid to 512 points further reduces
  interpolation error but is not required to address that legacy format issue.
  The benefit of a denser radial grid is workload-dependent and is generally
  more pronounced when structures sample short bonds, compressed environments,
  or close contacts, where radial derivatives vary rapidly. For those
  workloads, export with `--num-spline-points 512` and validate representative
  configurations.
- The serial `symmetrix/mace` pair style and the field-free `compute_H1` entry
  refuse MACEField models (the serial pair style crashed on them); MACEField
  runs through `symmetrix/mace/kk electric_field Ex Ey Ez` or the field entries.
- Fixed the LAMMPS pair styles' `mpi_message_passing` and
  `no_mpi_message_passing` modes, which omitted the first-interaction residual
  of residual-first models.


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
