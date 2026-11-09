# Changelog

All notable changes to Symmetrix-XL are documented in this file.

## Unreleased

- Fixed MACE export for checkpoints whose stored symmetric-contraction U
  matrices differ in sign from the recomputed ones, such as MACE-MP-0b2-large
  (`mace-large-density-agnesi-stress.model`). The exporter takes each recomputed
  U column's sign from the checkpoint's `U_matrix_<nu>` buffer and refuses a
  buffer that matches neither sign. Exports of checkpoints that already
  matched are unchanged.
- Fixed the Kokkos pair style accepting fixed-workspace tiled plans under
  `no_domain_decomposition` (`allow_fixed_workspace yes` or a tiled
  `_debug_execution_plan`), which then failed during `run`. `pair_style` now
  refuses them and names `mpi_message_passing`, and the generic evaluation
  entry refuses edge vectors under an active tiled plan unless they are the
  registered geometry or the prepared workspace.
- Fixed the A1 channel-tile kernels of the Phi1 `channel-tiled-64` policy,
  which the CUDA dual-layer fixed-workspace plan selects: they read the
  evaluator's channel count through the host object pointer, which a GPU can
  dereference only where it can access host memory. They now copy it before
  the launch, as the neighbouring kernels do; the values are unchanged.

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
