# Benchmarks

This directory contains reproducible performance drivers. It does not serve as
storage for benchmark results, raw profiler captures, generated kernels, model
files, machine-specific run output, or historical qualification reports.

## Maintained drivers

Current Python files directly under this directory are the maintained entry
points. Names identify the workload or comparison they run. In particular:

- `factorized_operator_benchmark.py` compares the extracted factorized R1
  operator across Symmetrix and the upstream implementation.
- `execution_upstream_r1_runner.py` is the isolated upstream runner used by that
  comparison.
- `standard_mace_streamed_benchmark.py` exercises standard MACE streamed-edge
  execution.
- `streamed_edge_release_benchmark.py` runs the milestone current/develop/
  PyTorch energy, forces, and stress comparison in isolated processes.
- `ase_omat_md_backend_scale.py` and
  `ase_omat_static_graph_backend_scale.py` measure OMAT backend scaling.
- The remaining `*_benchmark.py`, profiling, generation, and validation scripts
  support the workload named by each file.
- `volc_lammps_kokkos_comm_device_reproducer.sh` and
  `lammps_kokkos_comm_device_migration.in` exercise eight-rank Kokkos
  ownership migration and generic communication-buffer growth on `volc`.

Drivers may require an existing model checkpoint, compiled extension, GPU, or
external profiler. Use their command-line help and the relevant user or build
documentation to establish the environment and invocation.

## Output and retention

Write generated output beneath `benchmarks/.artifacts/<descriptive-run-name>/`.
The entire `.artifacts` tree is ignored by Git. Raw JSON/CSV logs, profiler
sessions, traces, generated source and binaries, downloaded models, and bulk
per-sample output must not be committed.

Keep raw evidence locally only as long as it is useful for analysis or external
archival. User-facing, current performance claims belong in the README or
documentation and must state the workload and hardware context. A raw artifact
may be added to version control only when a focused test consumes it as a stable
fixture; test fixtures belong with that test rather than in `.artifacts`.
