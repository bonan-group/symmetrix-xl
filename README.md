# Symmetrix-XL

[![CI](https://github.com/bonan-group/symmetrix-xl/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/bonan-group/symmetrix-xl/actions/workflows/ci.yaml?query=branch%3Amain)
[![Documentation](https://github.com/bonan-group/symmetrix-xl/actions/workflows/docs-pages.yaml/badge.svg)](https://bonan-group.github.io/symmetrix-xl/)

In Symmetrix-XL, **XL** stands for **eXtreme scale, Low latency**.

Symmetrix-XL builds on [Symmetrix](https://github.com/wcwitt/symmetrix), a cross-platform evaluator for MACE models,
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

See the [documentation](https://bonan-group.github.io/symmetrix-xl/) for
installation, supported workflows, and developer references.

### Demonstrated scale

An FP32 NVIDIA A100-SXM4-80GB qualification evaluated energy, forces, and
stress for the standard two-layer MACE-OMAT-0 model on cubic SrTiO3:

| Execution | Maximum atoms | Directed edges | Speed (us/atom) | Sampled peak VRAM (MiB) |
|---|---:|---:|---:|---:|
| Standard | 1,373,125 | 141,157,250 | 6.330 | 79,313 |
| Extended | 13,140,360 | 1,350,829,008 | 6.868 | 80,639 |

Both used a 6.0 A model cutoff and 0.5 A neighbor-list skin, giving a 6.5 A
effective cutoff, and completed with zero fallbacks. These are workload-specific
demonstrations, not capacity guarantees.

Fixed-workspace execution traded about 8.5% throughput in this qualification
for the larger demonstrated capacity. Standard execution remains the default;
large production systems can also be distributed across multiple GPUs with
LAMMPS.

### Demonstrated speed

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
[matched speed qualification](benchmarks/streamed_edge_milestone_20260822.md#superseding-cuda-paper-qualification-2026-09-14).

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
    --output mace-omat-0-medium.json
```

The default is the preferred universal compact export, retaining every element
supported by the checkpoint. Compact format v2 stores the shared radial model
once instead of generating pair-specific spline tables, while format v3 also
retains the complete checkpoint domain. Universal compact files therefore
avoid quadratic pair-table growth and can be reused across compositions; they
still include the checkpoint's element-indexed learned parameters. Element
selectors are intended only for deliberately restricted format-v2 deployments
or exports in the original Symmetrix pair-spline format (named v1 here);
format-v3 models reject subsets.

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
atoms.calc = Symmetrix("mace-omat-0-medium.json")
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

### Citing Symmetrix

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

Symmetrix-XL is based on [Symmetrix](https://github.com/wcwitt/symmetrix),
developed by Chuck Witt.

The original Symmetrix project also has the following acknowledgement statement:

An early phase of this project, leading to the Kokkos-based MACE implementation, was supported by the Schmidt Sciences Virtual Institute for Scientific Software (VISS). This engagement involved key contributions from Dave Brownell and Ketan Bhardwaj of the Center for Scientific and Software Engineering at Georgia Tech.
