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

An FP32 NVIDIA A100-SXM4-80GB qualification evaluated energy, forces, and
stress for the standard two-layer MACE-OMAT-0 model on cubic SrTiO3:

| Execution | Maximum atoms | Directed edges | Speed (us/atom) | Sampled peak VRAM (MiB) |
|---|---:|---:|---:|---:|
| Standard | 1,373,125 | 141,157,250 | 6.330 | 79,313 |
| Fixed workspace | 13,140,360 | 1,350,829,008 | 6.868 | 80,639 |

Both used a 6.0 A model cutoff and 0.5 A neighbor-list skin, giving a 6.5 A
effective cutoff, and completed with zero fallbacks. These are workload-specific
demonstrations, not capacity guarantees.

## Demonstrated Speed

A matched FP32 RTX 5090 qualification compared complete warmed ASE
energy/forces/stress calls for the standard OMAT-0-medium checkpoint:

| Atoms | Directed edges: MACE-Torch / Symmetrix-XL | MACE-Torch + cuEquivariance (us/atom) | Symmetrix-XL direct (us/atom) | Speedup | Sampled VRAM: MACE-Torch / Symmetrix-XL (MiB) |
|---:|---:|---:|---:|---:|---:|
| 864 | 78,624 / 97,762 | 44.487 | 4.241 | 10.49x | 2,136 / 924 |
| 4,000 | 364,000 / 452,342 | 31.011 | 3.248 | 9.55x | 7,136 / 1,386 |

MACE-Torch 0.3.15 with cuEquivariance 0.11.0 used the exact 6.0 A graph.
Symmetrix-XL used the same model cutoff plus a 0.5 A neighbor-list skin, giving a
candidate graph with an effective cutoff of 6.5 A; the graph policies are not
identical, and Symmetrix-XL processed more directed candidates.

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
