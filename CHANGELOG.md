# Changelog

All notable changes to Symmetrix-XL are documented in this file.

## Unreleased

- Model JSON format version 4 for every exported variant: the first readout and
  the H1->H2 skip are stored without the inverse of `interactions[1].linear_up`
  (`h1_affine_mode` `m0_composite` or `pre_linear_up`). Format-1/2 files still
  load with the old arithmetic and print a warning with an fp32 cancellation
  estimate computed from the file; `symmetrix_recompose_mace_json` converts
  them.

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
