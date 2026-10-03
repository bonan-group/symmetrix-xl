# What Symmetrix-XL Does

Symmetrix-XL is a package for functions equivariant under translation, rotation,
inversion, and exchange of particles. It provides compact MACE and MACEField
inference through a Python frontend, an ASE calculator, Kokkos CPU/GPU
backends, and LAMMPS pair styles.

The package is split into a stable Python frontend and a native execution
backend. The default installation includes the CPU/OpenMP backend. Additional
CUDA and HIP backends are installed separately for a specific GPU architecture
and are selected at process startup. This lets the same Python-facing API be
used on a host CPU, an NVIDIA GPU, or an AMD GPU without mixing incompatible
native extensions.

The frontend provides the Python API, ASE calculator, model conversion, and CLI;
the backend provides the native graph and model evaluation kernels. When a
matching wheel is available, a pre-compiled pip package avoids the local
C++/Kokkos/CUDA or HIP toolchain and supplies a tested architecture-specific
extension. Source builds remain available for unsupported targets and backend
development.

## What It Computes

For ordinary MACE models, Symmetrix-XL evaluates total energy, per-atom energies,
forces, and stress. MACEField models extend this interface with polarization
and supported analytical field-response properties. Multi-head model files can
retain every prediction head, and `SymmetrixEnsemble` can evaluate compatible
models together and report member values, means, and population variances.

Compatible format-version-2 MACE and MACEField JSON can be used from Python/ASE
or from the `pair_symmetrix` LAMMPS integration. Two-interaction direct LAMMPS
runs consume a prepared JIT artifact; LAMMPS does not invoke the runtime
compiler during a simulation. Compiler-free and single-layer paths do not
require an R1 artifact.

## Demonstrated Scale

The paper's FP32 single-GPU LAMMPS capacity benchmark uses the MACE-OMAT-0
medium checkpoint on cubic SrTiO3, with a 6.0 A model cutoff and a 0.5 A
neighbor-list skin:

| Execution | A100 80 GB atoms | RTX 5090 32 GB atoms |
|---|---:|---:|
| ML-IAP + cuEquivariance | 24,565 | 8,640 |
| Symmetrix-XL standard streaming | 1,250,235 | 486,680 |
| Symmetrix-XL tiled streaming | 11,240,455 | 4,152,920 |

Tiled streaming is an explicitly enabled capacity mode; standard streaming
remains the default. These are capacity-boundary probes under the stated
protocol, not general capacity guarantees or sustained 20-step MD results. On
64 A800 GPUs, tiled streaming weak-scaled to 703.04 million atoms at 93.81%
efficiency.

## Demonstrated Speed

A matched FP32 RTX 5090 qualification compared complete warmed ASE
energy/forces/stress calls for the standard MACE-OMAT-0 medium checkpoint:

| Atoms | MACE-Torch + cuEquivariance (us/atom) | Symmetrix-XL direct (us/atom) | Speedup | Sampled VRAM: MACE-Torch / Symmetrix-XL (MiB) |
|---:|---:|---:|---:|---:|
| 864 | 44.487 | 4.241 | 10.49x | 2,136 / 924 |
| 4,000 | 31.011 | 3.248 | 9.55x | 7,136 / 1,386 |

The MACE-Torch baseline used cuEquivariance with the exact 6.0 A graph.
Symmetrix-XL used the same model cutoff plus a 0.5 A neighbor-list skin, giving
an effective cutoff of 6.5 A; the graph policies are therefore not identical.

Across the matched FP32 LAMMPS benchmarks reported in the paper, Symmetrix-XL
reduces complete step time by 3.1-5.0x relative to ML-IAP + cuEquivariance on
the tested A100 and RTX 5090 workloads. A representative RTX 5090 result uses
5,000-atom perturbed cubic SrTiO3 with MACE-OMAT-0 medium:

| Implementation | Time (us/atom/step) | Speedup vs. ML-IAP |
|---|---:|---:|
| MACE-Torch + cuEquivariance through LAMMPS ML-IAP | 12.813 | 1.00x |
| Symmetrix-XL LAMMPS pair style | 2.559 | 5.01x |

Each value is the median of three 20-step runs after warmup. Both deployments
used a 6.0 A model cutoff and a 0.5 A neighbor-list skin; see the paper for the
full benchmark protocol and graph-policy details.

## How Execution Is Chosen

The first-class performance path is streamed-edge `direct` execution. For
Kokkos evaluation, the default `capacity` profile estimates device memory and
chooses the fastest qualified plan that fits. A matching runtime-specialized
R1 artifact is required for two-interaction direct execution. Admitted
single-layer models have no R1 stage and use built-in direct execution. Their
capacity plans may still compile or load specialized M0/R0 operator modules.
Incompatibilities fail clearly rather than silently changing algorithms.

Model evaluation defaults to FP32. This is the primary performance and
capacity mode on CPU and GPU backends. Request `dtype="float64"` explicitly
when a calculation requires higher numerical precision; JIT artifacts are
precision-specific.

`non-compiled` is the explicit compiler-free compatibility and diagnostic path.
It is useful when preparing or diagnosing a direct artifact, but it has no
performance guarantee. `generic` is its deprecated internal-facing alias.
Historical `materialized` and other compatibility modes remain available only
where the current support matrix permits them.

## What Is Installed

Installing the Python package provides the frontend and CPU backend. Converting
a PyTorch checkpoint to compact JSON is a separate optional workflow that uses
`mace-torch`; once JSON has been created, normal inference and LAMMPS execution
do not require the converter package. See {doc}`models` for conversion and
{doc}`installation` for backend setup.
