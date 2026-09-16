# What Symmetrix Does

Symmetrix is a package for functions equivariant under translation, rotation,
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

For ordinary MACE models, Symmetrix evaluates total energy, per-atom energies,
forces, and stress. MACEField models extend this interface with polarization
and supported analytical field-response properties. Multi-head model files can
retain every prediction head, and `SymmetrixEnsemble` can evaluate compatible
models together and report member values, means, and population variances.

The same compact model data can be used from Python/ASE or from the
`pair_symmetrix` LAMMPS integration. LAMMPS runs consume prepared model and JIT
artifacts; they do not invoke the runtime compiler during a simulation.

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
