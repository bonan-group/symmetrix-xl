# Symmetrix-XL

In Symmetrix-XL, **XL** stands for **eXtreme scale, Low latency**.

Symmetrix-XL provides native CPU and GPU inference for MACE and
MACEField models. The published distribution is named `symmetrix-xl`; the
Python import namespace and command-line interface remain `symmetrix`.

This package is a fork of the original
[`symmetrix`](https://github.com/wcwitt/symmetrix) project. Symmetrix-XL adds
two execution strategies:

1. **Edge streaming.** Edge messages are consumed immediately instead of being
   fully materialized in DRAM.
2. **Specialized code generation.** Source for critical operations is generated
   from the model architecture and parameters, enabling compiler specialization
   and vectorization.

The base distribution contains the Python frontend and an x86-64-v3
CPU/OpenMP backend, which requires AVX2, FMA, and the other x86-64-v3 features.
CUDA and HIP backends are installed as separate, architecture-qualified
packages so that they can coexist without overwriting the frontend or CPU
extension.

## Demonstrated scale

An FP32 NVIDIA A100-SXM4-80GB qualification evaluated energy, forces, and
stress for the standard two-layer MACE-OMAT-0 model on cubic SrTiO3:

| Execution | Maximum atoms | Directed edges | Speed (us/atom) | Sampled peak VRAM (MiB) |
|---|---:|---:|---:|---:|
| Standard | 1,373,125 | 141,157,250 | 6.330 | 79,313 |
| Fixed workspace | 13,140,360 | 1,350,829,008 | 6.868 | 80,639 |

Both used a 6.0 A model cutoff and 0.5 A neighbor-list skin, giving a 6.5 A
effective cutoff, and completed with zero fallbacks. These are workload-specific
demonstrations, not capacity guarantees.

## Demonstrated speed

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
[matched speed qualification](https://github.com/bonan-group/symmetrix-xl/blob/main/benchmarks/streamed_edge_milestone_20260822.md#user-content-superseding-cuda-paper-qualification-2026-09-14).

## Installation

Symmetrix-XL supports CPython 3.10 through 3.14 on Linux x86-64.

```bash
python -m pip install symmetrix-xl
```

Install a GPU package whose architecture exactly matches the target device.
For example:

```bash
python -m pip install symmetrix-xl-cuda12-sm80
python -m pip install symmetrix-xl-cuda13-sm120
```

GPU packages depend on the matching version of `symmetrix-xl`, so installing a
GPU backend also installs the frontend and CPU fallback. CUDA runtime, cuBLAS,
and NVRTC libraries are supplied by declared NVIDIA Python packages. A
compatible NVIDIA driver remains a host requirement.

Inspect the installed backends and verify the selected runtime in a fresh
process:

```bash
symmetrix backend list
symmetrix backend show
symmetrix doctor
```

The frontend loads exactly one native backend per process. Without an explicit
selection, it chooses an installed accelerator backend only when its
architecture exactly matches a visible device; otherwise it uses CPU. Select a
backend before importing a calculator when more than one usable backend is
installed:

```bash
SYMMETRIX_BACKEND=cuda13-sm120 python calculation.py
```

An extension compiled for one GPU architecture must not be used on another
architecture. NVRTC specializes model kernels at runtime, but Kokkos and
SpheriCart device code is compiled ahead of time into the backend extension.

## Source installation

Clone the repository with its submodules, create an environment, and use the
build frontend from the repository root:

```bash
git clone --recursive https://github.com/bonan-group/symmetrix-xl.git
cd symmetrix-xl
uv venv
source .venv/bin/activate
python tools/symmetrix_build.py install --backend cpu --cpu-target native
```

Use `--cpu-target x86-64-v3` for a portable x86-64-v3 CPU build. A native build
is optimized for the build host and should not be copied to heterogeneous
machines.

Build an exact CUDA backend with a matching development toolkit:

```bash
python tools/symmetrix_build.py install \
    --backend cuda --arch sm120 --cuda-root /usr/local/cuda-13.3
```

Build HIP in a separate environment and build directory:

```bash
python tools/symmetrix_build.py install \
    --backend hip --arch gfx1151 --rocm-root /opt/rocm
```

Source builds require Python 3.10 or newer, CMake 3.27 or newer, a C++20
compiler, and a recursive checkout. CPU builds additionally require a Fortran
compiler and optimized BLAS. CUDA builds require `nvcc`; HIP builds require
`hipcc`. Do not reuse a build directory across CPU, CUDA, and HIP backends.

The maintained build and cluster instructions are in the
[installation guide](https://github.com/bonan-group/symmetrix-xl/blob/main/docs/user/installation.md)
and
[developer build guide](https://github.com/bonan-group/symmetrix-xl/blob/main/docs/developer/build_and_test.md).

## Model conversion

Compact Symmetrix-XL JSON is the preferred deployment format. JSON evaluation does
not require PyTorch or `mace-torch`; those packages are needed only when
loading or converting an upstream checkpoint.

Install the optional converter dependencies and export a checkpoint:

```bash
python -m pip install "symmetrix-xl[mace]"
symmetrix_extract_mace \
    --model mace-omat-0-medium.model \
    --output srtio3-mace.json
```

Universal compact exports are preferred. By omitting `--chemical-symbols` and
`--atomic-numbers`, format v2 retains the checkpoint's complete element domain
without generating the per-element-pair spline tables used by the original
Symmetrix format (named v1 here). Format-v3 nonlinear MACE exports likewise
retain the complete domain and do not support element subsets. Universal files
still contain the checkpoint's element-indexed learned parameters. Use a subset
only for an intentionally restricted format-v2 deployment or an original-format
`--radial-format pair-splines` file.

The converter retains all compatible prediction heads. Use `--head` to select
the default head without discarding the others:

```bash
symmetrix_extract_mace \
    --model MACEField-MH-0-omat-dielectric.model \
    --head mp-dielectric \
    --output srtio3-macefield.json
```

MACEField checkpoints must be converted before use. Passing an original
MACEField checkpoint directly to the calculator is rejected rather than
silently delegating evaluation to PyTorch.

See the
[model guide](https://github.com/bonan-group/symmetrix-xl/blob/main/docs/user/models.md)
for supported model families, multi-head models, precision, and conversion
options.

## ASE calculator

Use the compact model with the normal ASE calculator interface:

```python
from ase.spacegroup import crystal
from symmetrix import Symmetrix

a = 3.905
atoms = crystal(
    symbols=["Sr", "Ti", "O"],
    basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
    spacegroup=221,
    cellpar=[a, a, a, 90, 90, 90],
)
atoms.calc = Symmetrix("srtio3-mace.json")

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()
```

Model evaluation defaults to FP32. Request FP64 explicitly when needed:

```python
atoms.calc = Symmetrix("srtio3-mace.json", dtype="float64")
```

Direct execution is the default for supported compact models. Two-interaction
models compile or reuse a precision- and backend-specific artifact and fail
clearly when it cannot be produced or loaded. Admitted single-layer models use
built-in direct execution without an R1 artifact; capacity planning may still
compile or load specialized M0/R0 operator modules. Use
`streamed_edges="non-compiled"` only as a compiler-free compatibility or
diagnostic mode.

A cold CPU specialization requires a C++20 compiler at runtime. CUDA
specialization uses NVRTC from the backend package's declared NVIDIA
dependencies and still requires a compatible host driver.

Runtime artifacts are stored in a private per-user cache. Set
`SYMMETRIX_JIT_CACHE` before starting Python to choose an explicit location:

```bash
export SYMMETRIX_JIT_CACHE=/path/to/private/symmetrix-jit-cache
symmetrix_prepare_jit_host_artifact \
    --model srtio3-mace.json --precision float32
```

The cache contains executable code and must not be writable by other users.
Generated artifacts are specific to the model contract, precision, backend,
compiler/runtime identity, and host or GPU target.

## MACEField

MACEField models expose energy, forces, stress, polarization, Born effective
charges, and polarizability. Set the electric field on the calculator or in the
ASE structure:

```python
import numpy as np
from symmetrix import Symmetrix

atoms.info["electric_field"] = np.array([0.01, -0.02, 0.03])
atoms.calc = Symmetrix("srtio3-macefield.json", dtype="float64")

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
polarization = atoms.calc.get_property("polarization", atoms)
becs = atoms.calc.get_property("becs", atoms)
polarizability = atoms.calc.get_property("polarizability", atoms)
```

Response properties are computed only when requested. Calculator instances are
stateful and should not be called concurrently from multiple host threads.

## Model ensembles

`SymmetrixEnsemble` evaluates compatible models sequentially in one process and
returns the member mean through ordinary ASE properties. Member values use the
`_comm` suffix and population variances use `_var`:

```python
from symmetrix import SymmetrixEnsemble

atoms.calc = SymmetrixEnsemble(["model-1.json", "model-2.json"])
mean_energy = atoms.get_potential_energy()
member_forces = atoms.calc.get_property("forces_comm", atoms)
force_variance = atoms.calc.get_property("forces_var", atoms)
```

Ensemble members must agree on model type, species, cutoff, precision, backend,
and execution mode.

## LAMMPS

LAMMPS integration is provided by `pair_symmetrix`. It does not embed Python or
compile model-specific code at runtime, so required host or device artifacts
must be prepared before launch. See the
[LAMMPS guide](https://github.com/bonan-group/symmetrix-xl/blob/main/docs/user/lammps.md)
for installation, artifact preparation, precision, and MPI requirements.

## Troubleshooting

Always diagnose the exact environment used for evaluation:

```bash
symmetrix backend list
symmetrix backend show --probe
symmetrix doctor --json
```

For CPU deployments, `doctor` checks OpenMP runtime identity and worker
participation. For CUDA and HIP, it checks device compatibility and launches a
sentinel kernel. Set `SYMMETRIX_OPENMP_RUNTIME_CHECK=strict` for production CPU
qualification.

When reporting a failure, include the backend selector, model hash, precision,
compiler or toolkit version, and the JSON output from `symmetrix doctor`.
Detailed diagnostic guidance is available in the
[troubleshooting guide](https://github.com/bonan-group/symmetrix-xl/blob/main/docs/user/troubleshooting.md).

## Documentation

- [User guide](https://github.com/bonan-group/symmetrix-xl/blob/main/docs/user/index.md)
- [Python API](https://github.com/bonan-group/symmetrix-xl/blob/main/docs/reference/python_api.md)
- [Backend and diagnostics guide](https://github.com/bonan-group/symmetrix-xl/blob/main/docs/user/backends.md)
- [Execution modes and artifacts](https://github.com/bonan-group/symmetrix-xl/blob/main/docs/user/execution.md)
- [Developer guide](https://github.com/bonan-group/symmetrix-xl/blob/main/docs/developer/index.md)

## Development

From a recursive source checkout:

```bash
uv venv
source .venv/bin/activate
uv pip install -e "./symmetrix[test]"
pytest symmetrix/test
uvx pre-commit run --all-files
```

Keep CPU, CUDA, HIP, and LAMMPS builds in separate environments and build
directories.

## License

Symmetrix-XL is distributed under the
[MIT License](https://github.com/bonan-group/symmetrix-xl/blob/main/LICENSE).
`pair_symmetrix` is GPLv2 to remain compatible with LAMMPS.
