# Getting Started

Install the Python frontend and the default CPU/OpenMP backend from a source
checkout:

```bash
git clone --recursive https://github.com/bonan-group/symmetrix-xl.git
cd symmetrix-xl
uv venv
source .venv/bin/activate
python tools/symmetrix_build.py install --backend cpu --cpu-target native
symmetrix backend list
symmetrix doctor
```

Use `--cpu-target x86-64-v3` for a portable deployment build. The frontend
loads exactly one backend per process; inspect choices with `symmetrix backend
list` or select one explicitly with `SYMMETRIX_BACKEND=cpu` before importing a
calculator.

Run the built-in benchmark on a deterministic, rattled 6x6x6 SrTiO3 supercell
(1,080 atoms) with the MACE-OMAT-0 medium checkpoint:

```bash
uv pip install mace-torch
symmetrix bench
```

The benchmark uses direct execution and the capacity profile, defaults to
FP32, and reports median `us/atom` and atoms per second. `--model` accepts a
local checkpoint or extracted JSON path, as well as a name recognized by
MACE's `mace_mp` downloader. Symmetrix-XL caches a compact extraction restricted
to the benchmark species and reuses it on subsequent runs. When the original
checkpoint and MACE-Torch are available, an untimed reference evaluation also
validates energy, forces, and stress; JSON-only inputs report an explicit skip.
Adjust `--dtype`, `--threads`, `--warmups`, and `--repeats` for a qualification run.
For a large cell, use `--skip-validation` to avoid constructing the separate
MACE-Torch reference model:

```bash
symmetrix bench --supercell-repeat 20 --skip-validation
```

Timing is finalized before reference validation begins. If the reference
calculation runs out of memory, Symmetrix-XL retains and reports the benchmark
result and marks validation unavailable.
It fails clearly if the selected direct backend or required JIT artifact is
not available; it does not silently benchmark the slower generic path.

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
```

This is the first-class path: FP32, `direct`, and capacity-aware plan selection
are the defaults. It requires an admitted model and a usable FP32 host or
device JIT artifact; prepare one as shown in {doc}`execution`. Request
`dtype="float64"` explicitly for high-precision evaluation and prepare the
matching FP64 artifact. `non-compiled` is only a compiler-free fallback/diagnostic
mode and has no performance guarantee:

```python
atoms.calc = Symmetrix("srtio3-mace.json", streamed_edges="non-compiled")
```

See {doc}`models` to create `srtio3-mace.json`, {doc}`execution` for direct plans
and capacity, and {doc}`backends` for CUDA/HIP installation.
