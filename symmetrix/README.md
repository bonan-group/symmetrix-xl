# Symmetrix-XL

In Symmetrix-XL, **XL** stands for **eXtreme scale, Low latency**.

Symmetrix-XL provides native CPU and GPU inference for MACE and
MACEField models. The published distribution is named `symmetrix-xl`; the
Python import namespace and command-line interface remain `symmetrix`.

The methods and benchmark protocol are described in
[Train for Accuracy, Execute at Scale: Architecture-Preserving Inference for Equivariant Atomistic Foundation Models](https://arxiv.org/abs/2610.01036).

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

The **frontend** is the Python API, ASE calculator, model-conversion tools, and
CLI. The **backend** is the native CPU/OpenMP, CUDA, or HIP extension that
constructs graphs and evaluates the model. Pre-compiled pip packages are the
recommended path when available because they provide tested, architecture-
matched native code without requiring a local C++/Kokkos/CUDA or HIP toolchain;
build from source when no suitable wheel exists or when developing the backend.

## Demonstrated scale

The paper's FP32 single-GPU LAMMPS capacity benchmark uses the MACE-OMAT-0
medium checkpoint on cubic SrTiO3, with a 6.0 A model cutoff and a 0.5 A
neighbor-list skin:

| Execution | A100 80 GB atoms | RTX 5090 32 GB atoms |
|---|---:|---:|
| ML-IAP + cuEquivariance | 24,565 | 8,640 |
| Symmetrix-XL standard streaming | 1,250,235 | 486,680 |
| Symmetrix-XL tiled streaming | 11,240,455 | 4,152,920 |

Tiled streaming is an explicitly enabled capacity mode; standard streaming
remains the default. These values are capacity-boundary probes under the stated
protocol, not general capacity guarantees or sustained 20-step MD results. On
64 A800 GPUs, tiled streaming weak-scaled to 703.04 million atoms at 93.81%
efficiency.

## Demonstrated speed

The Symmetrix-XL paper reports the matched performance study, including the
model, graph policy, hardware, warmup, and measurement protocol. Release
documentation does not reproduce timings from comparison packages; consult the
paper when evaluating performance claims.

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

Published accelerator wheels can also be installed from the CLI:

```bash
symmetrix backend install
symmetrix backend install --arch cuda13-sm120
```

The default `--arch auto` detects one visible GPU architecture. The command
uses `uv` when available and otherwise invokes `python -m pip`; it reports the
detected architecture, toolkit version and source, selected package, and
Python environment before installing. If the detected CUDA-major wheel is not
published, automatic selection probes the lower CUDA 12 wheel for the same
architecture.
On CPU-only hosts, the bundled CPU backend needs no download. If no pre-built
wheel is available, the CLI reports a source-build command.

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
compiler, a Fortran compiler, OpenBLAS or another compatible optimized BLAS,
and a recursive checkout. CUDA builds additionally require a matching CUDA
toolkit and `nvcc`; HIP builds require a matching ROCm toolkit and `hipcc`.
Do not reuse a build directory across CPU, CUDA, and HIP backends.

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

### NequIP OAM (experimental)

NequIP support uses the OAM v2 versioned JSON contract. Install the optional
exporter stack in a fresh environment and export an upstream OAM package or
checkpoint:

```bash
uv pip install "symmetrix-xl[nequip]"   # torch, nequip, and e3nn
symmetrix_extract_nequip \
    --model NequIP-OAM-S-0.1.nequip.zip \
    --output oam-s-nequip.json
```

Runtime evaluation accepts the OAM v2 JSON contract with
`symmetrix_nequip_format_version=2` and `architecture="oam"`. The contract
retains a universal H through Pu element mapping, model cutoff, ordered layers,
and precision-independent parameters. Runtime evaluation uses the exported
JSON, ASE, and the native Symmetrix extension; PyTorch, NequIP, and e3nn remain
exporter/reference dependencies. OAM-S, OAM-M, OAM-L, and OAM-XL are
experimentally supported.
Retained OAM-v2 qualification covers energy, forces, and stress for host
S-FP32, M-FP32/FP64, L-FP32, and XL-FP32 and CUDA `sm120` S/M/L/XL FP32/FP64.
The independent upstream comparison is narrower and does not establish live
upstream FP64 parity. HIP and multi-GPU execution are unsupported; other CUDA
targets are artifact-available but runtime-unqualified. The earlier `sm80`
package/generated-path record predates ABI v4 and requires a merged-source
rerun.

Unsupported modules fail during export with a qualified diagnostic. Original
`.nequip`/`.nequip.zip` packages, PyTorch checkpoints, and AOTInductor `.pt2`
artifacts are upstream reference inputs and are not loaded implicitly by the
native calculator. See the [model guide](../docs/user/models.md#nequip-and-oam)
for current OAM support and artifact boundaries.

The CPU and CUDA release-wheel build plan includes FP32 and FP64 generated
artifacts for the known OAM-S/M/L/XL architectures; repaired-wheel inclusion
remains a release gate. `NequIPCalculator` selects one only when the
model's explicit ordered layer, irrep/channel, angular-order, tensor-product,
and radial-table layout matches exactly; model paths and learned-weight hashes
are not selection keys. A custom architecture is generated during preparation
and stored in the configured runtime JIT cache when the matching compiler is
available. As for MACE, the installed catalog reduces time to first energy but
is not a correctness fallback.

See the
[model guide](https://github.com/bonan-group/symmetrix-xl/blob/main/docs/user/models.md)
for supported model families, multi-head models, precision, and conversion
options. NequIP-OAM users should also follow the
[publication guide](https://github.com/bonan-group/symmetrix-xl/blob/main/docs/reference/publications.md#nequip-oam)
for the NequIP and OAM papers to cite.

## ASE calculator

Use the compact model as a drop-in replacement for the usual MACECalculator in
ASE workflows:

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

Native DFT-D2 and two-body DFT-D3 corrections can be added without PyTorch or
`torch-dftd`:

```python
from symmetrix import D3Correction, Symmetrix

atoms.calc = Symmetrix(
    "srtio3-mace.json",
    corrections=[D3Correction(functional="pbe", damping="bj")],
)
```

The correction runs through the loaded Kokkos CPU, CUDA, or HIP backend and
owns a neighbor graph independent of the model graph. See the
[dispersion guide](https://github.com/bonan-group/symmetrix-xl/blob/main/docs/user/dispersion_corrections.md)
for D2, damping choices, cutoffs and skin, diagnostics, and limitations. The
deprecated `symmetrix-xl[dispersion]` extra remains an empty compatibility
extra; ordinary `symmetrix-xl` already contains the native implementation.

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
test -x .venv/bin/python || uv venv .venv
source .venv/bin/activate
uv pip install scikit-build-core pybind11 ninja
source_fingerprint=$(python -c 'from pathlib import Path; from tools._symmetrix_build.source_provenance import symmetrix_source_fingerprint; print(symmetrix_source_fingerprint(Path.cwd()))')
uv pip install --no-build-isolation --reinstall-package symmetrix-xl \
  -e "./symmetrix[test]" \
  --config-setting="cmake.define.SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256=$source_fingerprint"
pytest symmetrix/test
uvx pre-commit run --all-files
```

Keep CPU, CUDA, HIP, and LAMMPS builds in separate environments and build
directories.

## License

Symmetrix-XL is distributed under the
[MIT License](https://github.com/bonan-group/symmetrix-xl/blob/main/LICENSE).
`pair_symmetrix` is GPLv2 to remain compatible with LAMMPS.
