# Benchmarks

This directory contains reproducible performance drivers. It does not serve as
storage for benchmark results, raw profiler captures, generated kernels, model
files, machine-specific run output, or historical qualification reports.

## Maintained drivers

Current Python files directly under this directory are the maintained entry
points. Names identify the workload or comparison they run. In particular:

- `standard_mace_streamed_benchmark.py` exercises standard MACE streamed-edge
  execution.
- `streamed_edge_release_benchmark.py` runs the milestone current/develop/
  PyTorch energy, forces, and stress comparison in isolated processes.
- `ase_omat_md_backend_scale.py` measures OMAT backend scaling through ASE.
- `srtio3_capacity_probe.py` probes a fresh-process GPU capacity point on a
  periodic cubic SrTiO3 workload and records the selected plan and sampled
  memory.
- `a100_srtio3_plan_benchmark.py` measures an explicit speed or capacity plan
  on the same workload. These two drivers generated the single-GPU capacity
  measurements used in the paper; use them only with a prepared compact model
  and a matching GPU backend.
- `oam_release_qualification.py` declares the schema-v3 OAM release contract:
  S-FP32, M-FP32, M-FP64, L-FP32, and XL-FP32 on the balanced 100-atom
  SrTiO3 `[2, 2, 5]` host graph at t1/t4; the unchanged S/M/L/XL × FP32/FP64
  CUDA small/production matrix; and exactly one XL/CUDA/FP32 production
  capacity boundary. It resumes only strictly validated cell
  records and leaves missing models, hardware, evidence, or baselines explicitly
  unqualified. Start from `oam_release_qualification_manifest.example.json` and
  replace its source/native paths, host/CUDA targets, and exact direct/reference
  binding declarations with values from the qualified builds. Every retained record and JIT
  namespace is bound to the resulting request fingerprint; a model, graph,
  target, baseline, provenance, route, profile, precision, thread, or boundary
  change cannot resume the old evidence. Use `--execute` only in a prepared
  qualification environment.
  Every canonical route writes a compact, request-path-bound NPZ containing
  total energy, full forces, and stress. The evaluator opens it without
  following symlinks, accepts only a bounded regular file, reads one immutable
  byte payload, and uses those exact bytes for SHA-256, ZIP bounds, and NumPy
  parsing. Parity is checked with the authoritative tolerances in
  `oam_release_contract.py`: FP32 energy/forces/stress use
  `1e-5/5e-4/5e-4`, and FP64 uses `5e-6/1e-5/1e-6`. These preserve the
  established OAM equivalence-driver contract, including the FP64 package's
  float32 internal-feature rounding floor. Stale v2
  manifests, driver records, cells, and results are not accepted. Graph rows
  are strict: `srtio3` accepts exactly `structure`, `repeat`, `seed`,
  `rattle`, and `skin`; `chain` replaces `repeat` with `atoms`. Baselines are
  keyed by exact cell ID and carry `identity`, a hexadecimal evidence
  `sha256`, and `us_per_atom`. Unknown graph fields or baseline cell IDs are
  rejected rather than ignored.
  A release pass requires the exact declared evidence set: ten host rows and
  seventeen CUDA rows. Undeclared, duplicate, extra, or stale-schema rows make
  correctness incomplete and prevent a top-level pass.
  `foundation_model_support_manifest.json` is a v5-only release-harness
  contract: older and future versions fail closed instead of being interpreted
  with v5 rules. Its top-level `tolerances` remain the established generic MACE
  direct-versus-noncompiled worker limits; OAM values live separately under
  `nequip_support.tolerances` and are projected exactly into each OAM row.
- The remaining analysis, profiling, generation, and validation scripts support
  the workload named by each file. One-off machine-specific probes and
  superseded single-layer or experimental backend drivers are intentionally not
  retained here.
- `lammps_kokkos_comm_device_migration.in` is the portable LAMMPS input for
  Kokkos ownership migration and communication-buffer growth. Site-specific
  launchers and machine paths are intentionally kept out of this repository.
- `polar_mace_serial_benchmark.py` measures the PolarMACE family on one pinned
  physical core and, with `--device cuda`, on the accelerator: the whole
  family's `energy`, `energy+forces`, and `energy+forces+stress` cost in
  `us/atom` with the full graph and long-range context (`n_k`, dense-table
  bytes, correction mode, retention bytes), the equal-shape MH-1 backbone on
  the same prepared graph (verified bit-identical), a stage breakdown, and a
  size ladder. It requires the cached `MACE-POLAR-1-{S,M,L}.model` checkpoints
  and the oracle `graph_longrange` checkout for the extraction. Record source/model hashes, backend, graph facts, and fresh-process timings
  with each result. See the current execution support matrix for release gates.
- `polar_torch_baseline.py` times the frozen torch oracle stack (mace
  `PolarMACE` + `graph_longrange`) on the same fixtures and protocol, on the
  CPU oracle environment or `--device cuda` through the repository
  environment, giving the reference implementation's own speed for the
  symmetrix comparisons. CUDA defaults to the preserved standard PyTorch/e3nn
  backend; `--backend cueq` creates a separate MACE-Torch+cuEquivariance
  record, verifies the accelerated ops runtime, and checks energy/force/stress
  parity against a fresh e3nn calculator outside the timed region.
- `polar_paper_benchmark.py` compares MACE-Torch and Symmetrix in isolated
  processes on deterministic reconstructions of the MACE-POLAR-1 paper's main
  size/boundary regimes. The suite contains an isolated molecule, a 384-atom
  finite-cluster proxy, the published 48-atom X23 benzene cell, and 64- and
  333-water periodic cells. Run the two `run` workers with the same checkpoint,
  dtype, GPU, warmups, and repeats, then use its `compare` command to verify
  fixture hashes, energy/force/stress tolerances, and synchronized E/EF/EFS
  timing ratios. The water boxes reproduce reported counts and cell sizes, not
  unpublished trajectory frames; every record labels exact structures and
  proxies explicitly. CUDA workers reject an occupied GPU unless
  `--allow-busy-gpu` is supplied. Only EFS timing is like-for-like because the
  pinned MACE-Torch calculator computes forces and stress for every request.

## Paper driver map

The headline paper measurements were generated by these reproducible drivers:

| Measurement | Drivers |
|---|---|
| ASE throughput and MACE-Torch comparison | `streamed_edge_release_benchmark.py`, `mace_torch_cuda_benchmark.py` |
| Single-GPU SrTiO3 capacity | `srtio3_capacity_probe.py`, `a100_srtio3_plan_benchmark.py` |
| LAMMPS fixture generation | `generate_lammps_mace_omat0_md.py` and the public `pair_symmetrix` tests |

The drivers emit machine-readable records only when an output path is supplied.
Keep those records under `benchmarks/.artifacts/`; they are not source data for
the repository and should not be committed.

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
