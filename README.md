# Symmetrix-XL

[![CI](https://github.com/bonan-group/symmetrix-xl/actions/workflows/ci.yaml/badge.svg)](https://github.com/bonan-group/symmetrix-xl/actions/workflows/ci.yaml?branch=main)

In Symmetrix-XL, **XL** stands for **eXtreme scale, Low latency**.

Symmetrix-XL builds on `symmetrix`, a cross-platform evaluator for MACE models,
and extends it with scalable, low-latency native execution paths for CPUs and
GPUs. The Python distribution is named `symmetrix-xl`; the import namespace and
command-line interface remain `symmetrix`.

Symmetrix-XL preserves the learned MACE model while changing how its equivariant
operations are scheduled, stored, and compiled. Its two main contributions are:

1. **Memory-bounded direct execution.** The default executor consumes,
   aggregates, or recomputes edge intermediates without retaining every
   materialized tensor, reducing graph-sized workspace.
2. **Model-specialized execution with runtime compilation.** Symmetrix-XL lowers
   each admitted MACE contraction structure into generated CPU, CUDA, or HIP
   kernels and caches the resulting artifact.

See the [user guide](docs/user/index.md) for supported workflows and the
[direct-execution architecture](docs/streamed_edge_execution.md) for the
implementation contract.

### Qualification highlights

Current FP32 records include a standard two-layer OMAT-0-medium evaluation of
1,372,000 atoms and 149,548,000 directed edges on one A100-80GB, and a
purpose-built single-layer fixed-workspace evaluation of 11,943,936 atoms and
1,301,889,024 directed edges on one RTX 5090. A matched RTX 5090 comparison for
the standard OMAT-0-medium checkpoint measured 9.55-10.49x lower warmed
energy/forces/stress time per atom than MACE-Torch with cuEquivariance for the
tested 864- and 4,000-atom cells.

These are workload-specific demonstrations, not general capacity or speed
guarantees. The exact models, cutoffs, graph policies, memory measurements, and
failure boundaries are recorded in the [user overview](docs/user/overview.md)
and the linked benchmark reports.

-----

### Quick Start

Symmetrix-XL supports CPU/OpenMP, CUDA, and HIP through separately built backend
packages. Install the Python frontend and CPU backend first; the CPU package is
the required base installation and remains available as a fallback. Additional
GPU backends can then be installed alongside it without replacing the frontend
or CPU backend. Each GPU backend must be built for the exact architecture of
the target device.

Start from a source checkout and install the CPU backend:

```bash
git clone --recursive https://github.com/bonan-group/symmetrix-xl.git
cd symmetrix-xl
uv venv
source .venv/bin/activate
python tools/symmetrix_build.py install --backend cpu --cpu-target native
```

For example, detect the visible NVIDIA GPU architecture and install the
matching CUDA 13.3 backend:

```bash
python tools/symmetrix_build.py install --backend cuda \
    --cuda-root /usr/local/cuda-13.3
```

Detection requires exactly one visible CUDA architecture. If no GPU is visible
on the build host, or visible GPUs have different architectures, specify the
deployment target explicitly, for example `--arch sm120`.

Inspect the installed backends and test the one selected for the current
machine:

```bash
symmetrix backend list
symmetrix doctor
```

Convert a MACE checkpoint to the compact JSON format used by Symmetrix-XL. The
converter is optional; JSON-only evaluation does not require `mace-torch`.

```bash
uv pip install mace-torch
symmetrix_extract_mace \
    --model mace-omat-0-medium.model \
    --chemical-symbols Sr Ti O \
    --output srtio3-mace.json
```

Run an energy, force, or stress calculation with ASE:

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
print(atoms.get_potential_energy())
print(atoms.get_forces())
print(atoms.get_stress())
```

The calculator defaults to FP32 model evaluation with generated direct,
capacity-aware execution. Pass `dtype="float64"` explicitly for high-precision
calculations. CUDA and HIP backends are installed separately for a matching GPU
architecture; see the [installation guide](docs/user/installation.md) and
[Symmetrix-XL package README](symmetrix/README.md) for backend-specific builds.

### LAMMPS integration

See the `pair_symmetrix` [README](pair_symmetrix/README.md) for use from LAMMPS.

### Development Setup

Use `uv` to create a virtual environment and install the package with its test dependencies:

```bash
uv venv
source .venv/bin/activate
uv pip install -e "./symmetrix[test]"
```

Run Python formatting and lint checks with `uvx pre-commit run --all-files`.

### Citing Symmetrix-XL

The earliest `symmetrix` results are reported in:
* D. P. Kovács, J. H. Moore, N. J. Browning, I. Batatia, J. T. Horton, Y. Pu, V. Kapil, W. C. Witt, I.-B. Magdău, D. J. Cole, G. Csányi, "MACE-OFF: Short-Range Transferable Machine Learning Force Fields for Organic Molecules", _Journal of the American Chemical Society_ **147**, 17598 (2025). [[arxiv]](https://arxiv.org/abs/2312.15211) [[journal]](https://doi.org/10.1021/jacs.4c07099)

MACE foundation models and implementations are described in:
* I. Batatia, P. Benner, Y. Chiang, A. M. Elena, D. P. Kovács, J. Riebesell, ...+78 others..., W. C. Witt, T. Wolf, F. Zills, G. Csányi, "A foundation model for atomistic materials chemistry," _Journal of Chemical Physics_ **163**, 184110 (2025). [[arxiv]](https://arxiv.org/abs/2401.00096) [[journal]](https://doi.org/10.1063/5.0297006)

Please cite these papers when using Symmetrix-XL.

### Licensing

The default license for this project is the [MIT License](./LICENSE).

The `pair_symmetrix` subdirectory, which enables integration with LAMMPS,
is licensed under the [GNU General Public License (GPLv2)](pair_symmetrix/LICENSE)
to maintain consistency with LAMMPS.

### Acknowledgements

An early phase of this project, leading to the Kokkos-based MACE implementation, was supported by the Schmidt Sciences Virtual Institute for Scientific Software (VISS). This engagement involved key contributions from Dave Brownell and Ketan Bhardwaj of the Center for Scientific and Software Engineering at Georgia Tech.
