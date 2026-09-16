# Runtime M0/R0 specialization qualification (2026-08-29)

## Scope

This record qualifies runtime-generated M0 and R0 device modules for
`streamed_edges="direct", low_memory=True`. The implementation prepares pure
NVRTC/hipRTC modules when a compact model's exact structural contract does not
match a built-in module. Learned weights, channels, species, node/edge counts,
and graph data remain runtime inputs.

The source state is the uncommitted implementation based on
`8cd95e17d4d7b38eaa41db137a9d67b48872621c` on `streamed-edge`. The CUDA
extension SHA-256 was
`a7459bd5749f5fb2d385e46462aa456e016ae5a3290d4e978a91c6ad35ec3042`.

## Correctness matrix

- Full fresh OpenMP/Python regression suite: 1,220 passed, 115 skipped.
- hipRTC compile-only M0/R0 matrix: 6 passed for FP32/FP64, both M0 schedules,
  and R0 without `hipcc` or AMD hardware.
- CUDA custom-model and MACEField matrix: 6 passed for FP32/FP64 and
  `chunk32`/`table`, with
  energy, per-atom energy, forces, stress, density scaling, input-scale
  adjoints, zero M0 polynomial capacity, cache reuse, and negative loader tests.
- Device artifact and CLI tests: 17 passed, including complete LAMMPS argument
  rendering and true R1-only `--path-only` preparation.

An additional 864-atom MACEField comparison used retained generic M0 as the
reference. Maximum absolute FP32 differences were:

| Comparison | Energy (eV) | Forces (eV/A) | Stress (eV/A^3) | Polarization |
|---|---:|---:|---:|---:|
| `chunk32` vs retained | 2.452e-4 | 1.388e-5 | 1.214e-8 | 8.719e-9 |
| `table` vs retained | 2.452e-4 | 1.392e-5 | 9.753e-9 | 8.719e-9 |
| `chunk32` vs `table` | 0 | 3.721e-7 | 2.386e-9 | 0 |

The retained comparison also changes from Cartesian retained geometry to the
qualified compact FP32 low-memory geometry; the schedule-to-schedule row
isolates generated M0 ordering differences.

## Schedule benchmark

The schedule A/B used the 128-channel MACEField dielectric model with one
additional valid M0 monomial so built-in M0 admission fails. Its JSON SHA-256
was `8bd8f9ccc833f78faa2008431288d7b9093955d2119f75e4cc769c97129bf427`;
the source checkpoint SHA-256 was
`f92e043aaf2cd8879919db8452503553fe7b608cb749d8d169dd96d4aa094aa2`.

- GPU: NVIDIA GeForce RTX 5090, compute capability 12.0, driver 610.43.02.
- Backend: CUDA 13.3, Kokkos CUDA, NVRTC, FP32.
- Structure: 864-atom periodic wurtzite AlN, 94,176 directed edges.
- Cutoff: 6.0 A; skin: 0.5 A; effective cutoff: 6.5 A.
- Properties in native evaluator scope: energy and force reverse; the directed
  pair-force result is the input to device stress reduction.
- Policy: direct low memory, compact FP32 directions/FP64 radii, M1
  recomputation tile 16, one prepared graph, five warmups, 30 samples.
- Selection: generated R1, runtime M0 device module, built-in R0, zero fallback.

| M0 schedule | Median ms | us/atom | atom/s | sampled VRAM | M0 artifact |
|---|---:|---:|---:|---:|---:|
| `chunk32` | 4.5897 | **5.3122** | **188,246** | 844 MiB | 96,704 B |
| `table` | 5.0567 | 5.8526 | 170,863 | 844 MiB | 74,792 B |

`chunk32` is 9.2% faster end to end. It remains the production default.
`table` remains available as a conservative schedule with lower cold compile
time and a smaller artifact.

## Profiler attribution

Nsight Systems 2026.1.3 traced eight identical evaluations per schedule. The
M0 kernel medians were:

| Schedule | Forward | Reverse | Combined | Relative combined |
|---|---:|---:|---:|---:|
| `chunk32` | 27.1 us | 46.0 us | **73.1 us** | 1.00x |
| `table` | 125.0 us | 268.4 us | 393.3 us | 5.38x |

Nsight Compute 2026.2.1 replayed one forward and one reverse launch per
schedule. Replay durations are not end-to-end timings.

| Schedule/kernel | registers/thread | achieved occupancy | local spills | NCU duration |
|---|---:|---:|---:|---:|
| `chunk32` forward | 128 | 31.47% | 0 | 41.22 us |
| `chunk32` reverse | 255 | 15.24% | 0 | 60.74 us |
| `table` forward | 40 | 42.21% | 0 | 220.45 us |
| `table` reverse | 48 | 42.05% | 0 | 355.52 us |

The table interpreter obtains higher occupancy with fewer registers, but it
executes the monomial table loop far more slowly. The chunked topology-specific
program has no local-memory spilling and wins despite its register-bound
occupancy, so occupancy alone would have selected the wrong schedule.

Reports are retained for this work session under `/tmp/symmetrix-rtc-profile/`:

- `chunk32-system.nsys-rep` and `table-system.nsys-rep`;
- `chunk32-m0.ncu-rep` and `table-m0.ncu-rep`.

## Built-in regression and capacity

The built-in OMAT-0 medium and MACEField contracts were requalified with the
installed current-source CUDA extension, SHA-256
`7861d4c1c2818666016ee38611d8335d016b55fa4dc79661853b2594a2b7b4d6`.
This installed artifact differs from the build-tree profiler artifact above
because CMake installation rewrites its runtime path. Both were produced by
the same fresh CUDA 13.3 BLACKWELL120 build.

The matched workload used 131,072 atoms (`n32`), 14,286,848 directed edges, a
6.0 A model cutoff, 0.5 A skin, and 6.5 A effective cutoff. Each current row is
the median of 15 calls after five warmups. The OMAT-0 comparison also reran an
immutable pre-change extension from the implementation base to isolate the
effect of this change.

| Model/policy | Current us/atom | atom/s | sampled VRAM | Base us/atom |
|---|---:|---:|---:|---:|
| OMAT-0 retained | 3.6678 | 272,645 | 19,332 MiB | 3.6714 |
| OMAT-0 low memory | 4.0396 | 247,549 | 11,266 MiB | 4.0463 |
| MACEField retained | 4.7872 | 208,890 | 20,106 MiB | n/a |
| MACEField low memory | 4.7364 | 211,129 | 12,298 MiB | n/a |

Current OMAT-0 differs from the immutable base by -0.10% for retained state
and -0.17% for low memory, so the selected built-in M0/R0 path has no measured
runtime-specialization regression. Low memory saves 8,066 MiB (41.7%) for
OMAT-0 and 7,808 MiB (38.8%) for MACEField. On this build OMAT-0 low memory is
10.1% slower than retained at matched size, while MACEField low memory is 1.1%
faster; the base binary reproduces the OMAT-0 policy difference, so it is not
caused by the new selection abstraction.

Retained-versus-low-memory numerical summaries remain within the qualified
FP32 compact-geometry tolerance. OMAT-0 differs by `6.28e-7 eV/atom`,
`5.01e-6 eV/A` in maximum force, and `1.06e-7 eV/A^3` in maximum stress.
MACEField differs by `6.76e-7 eV/atom`, `1.15e-6 eV/A` in maximum force,
`8.24e-8 eV/A^3` in maximum stress, and `3.52e-9` in polarization.

One current fresh process was run at each previously established capacity
endpoint and first failure. All four brackets reproduce exactly:

| Model | Policy | Largest success | Directed edges | First failure | Failure class |
|---|---|---:|---:|---:|---|
| OMAT-0 | retained | 219,488 (`n38`) | 23,924,192 | 237,276 (`n39`) | CUDA OOM, 463.4 MiB allocation |
| OMAT-0 | low memory | 389,344 (`n46`) | 42,438,496 | 415,292 (`n47`) | CUDA OOM, 405.6 MiB allocation |
| MACEField | retained | 202,612 (`n37`) | 22,084,708 | 219,488 (`n38`) | CUDA OOM, 1.675 GiB allocation |
| MACEField | low memory | 340,736 (`n44`) | 37,140,224 | 364,500 (`n45`) | CUDA OOM, 711.9 MiB allocation |

Every success selected generated direct R1, built-in M0, built-in R0, and zero
fallback. Low-memory successes additionally selected adjoint reuse and compact
FP32 geometry. No illegal-address failure occurred. Raw current and immutable
base JSON records are retained under the ignored
`benchmarks/.artifacts/runtime_m0_r0_capacity_20260829/` directory.

## Reproduction

The steady-state worker command was:

```bash
SYMMETRIX_SOURCE_ROOT=$PWD \
SYMMETRIX_EXTENSION=/absolute/path/to/symmetrix.cpython-312-x86_64-linux-gnu.so \
SYMMETRIX_JIT_POLICY=required \
SYMMETRIX_JIT_CUDA_JIT_BACKEND=nvrtc \
SYMMETRIX_JIT_CACHE=/tmp/symmetrix-rtc-profile/jit-cache \
SYMMETRIX_M0_RTC_SCHEDULE=chunk32 \
python benchmarks/macefield_standard_aln_scale.py worker \
  --backend cuda --kind field --policy optimized_recompute \
  --model /tmp/symmetrix-rtc-profile/runtime-specialized-macefield.json \
  --repeat 6 --warmups 5 --samples 30 --low-memory \
  --m1-tile-channels 16 \
  --edge-geometry-policy unit-f32-radius-f64-v1 --response none
```

Repeat with `SYMMETRIX_M0_RTC_SCHEDULE=table` for the comparator.

## Deployment status

Python/ASE automatically prepares only missing M0/R0 modules and validates
them natively before low-memory admission. The device prewarm CLI returns R1
plus optional M0/R0 metadata and now offers `--lammps-arguments` to render the
complete pair-style fragment. LAMMPS loads R1, then M0/R0, then activates low
memory; it never invokes NVRTC/hipRTC itself.

The checked-in compile-time M0/R0 topology specializations remain operational
for zero-cold-start compatibility, but are now explicitly marked deprecated
and frozen as an extension mechanism. New model contracts must be covered by
the RTC generators rather than by adding another built-in topology.

No compatible local LAMMPS source/build was available for executable
qualification in this session. Parser/load ordering has focused source tests,
and the MPI test harness accepts precision-specific operator artifacts. A
fresh CUDA/HIP LAMMPS build and device MPI execution remain deployment gates.

HIP execution remains unqualified because this machine has no AMD GPU. The
hipRTC compile-only matrix is evidence for compiler feasibility, not runtime
correctness or performance.
