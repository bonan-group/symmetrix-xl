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

The same compact model data can be used from Python/ASE or from the
`pair_symmetrix` LAMMPS integration. LAMMPS runs consume prepared model and JIT
artifacts; they do not invoke the runtime compiler during a simulation.

## Demonstrated Scale

Capacity is specific to the model, precision, requested properties, graph
density, and hardware. Current FP32 qualification highlights are:

- A standard two-layer OMAT-0-medium MACE model evaluated 1,372,000 atoms and
  149,548,000 directed edges on one NVIDIA A100-SXM4-80GB. Two fresh-process
  trials took 6.931 and 6.983 us/atom and reached 80,411 MiB sampled peak
  device memory; the adjacent 1,431,644-atom case failed twice with CUDA OOM.
  The workload requested energy, forces, and stress with a 6.0 A model cutoff
  and 0.5 A neighbor-list skin, giving an effective cutoff of 6.5 A.
- A purpose-built, nonstandard single-layer qualification model
  (`single-v2`) evaluated 11,943,936 atoms and 1,301,889,024 directed edges on
  one NVIDIA RTX 5090 with 32,607 MiB at 1.830 us/atom and 22,414 MiB peak
  device memory. This used the
  separately enabled fixed-workspace plan and requested energy, per-atom
  energies, forces, and stress with the same 6.0 A model cutoff, 0.5 A skin,
  and an effective cutoff of 6.5 A. It was the largest host-feasible ASE
  construction tested, not a general GPU capacity limit.

These results are exact workload demonstrations, not capacity guarantees for
other configurations. See the repository's
[A100 qualification record](https://github.com/bonan-group/symmetrix-xl/blob/main/benchmarks/extreme_scale_indexing_20260829.md#superseding-a100-capacity-result-2026-09-15)
and
[single-layer fixed-workspace record](https://github.com/bonan-group/symmetrix-xl/blob/main/benchmarks/single_layer_fixed_workspace_cuda_20260910.md#capacity-result)
for complete context.

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
identical, and Symmetrix-XL processed more directed candidates. See the
[matched speed qualification](https://github.com/bonan-group/symmetrix-xl/blob/main/benchmarks/streamed_edge_milestone_20260822.md#superseding-cuda-paper-qualification-2026-09-14).

## How Execution Is Chosen

The first-class performance path is streamed-edge `direct` execution. For
Kokkos evaluation, the default `capacity` profile estimates device memory and
chooses the fastest qualified plan that fits. A matching runtime-specialized
artifact is required for direct execution, and incompatibilities fail clearly
rather than silently changing algorithms.

Model evaluation defaults to FP32. This is the primary performance and
capacity mode on CPU and GPU backends. Request `dtype="float64"` explicitly
when a calculation requires higher numerical precision; JIT artifacts are
precision-specific.

`generic` is an explicit compiler-free compatibility and diagnostic path. It is
useful when preparing or diagnosing a direct artifact, but it has no performance
guarantee. Historical `materialized` and other compatibility modes remain
available only where the current support matrix permits them.

## What Is Installed

Installing the Python package provides the frontend and CPU backend. Converting
a PyTorch checkpoint to compact JSON is a separate optional workflow that uses
`mace-torch`; once JSON has been created, normal inference and LAMMPS execution
do not require the converter package. See {doc}`models` for conversion and
{doc}`installation` for backend setup.
