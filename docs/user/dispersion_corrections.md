# Native Dispersion Corrections

Symmetrix-XL can add native DFT-D2, two-body DFT-D3, or reciprocal-space
FourierD3 to an ASE calculation.
The implementation is part of the standard `symmetrix-xl` installation and
does not import PyTorch or `torch-dftd` at runtime. The old `dispersion` package
extra is retained temporarily as an empty compatibility extra.

Published calculations should cite the D2 or D3 method and the damping variant
actually used. FourierD3 calculations should additionally cite the FourierD3
method. See {doc}`/reference/publications` for the complete references and the
torch-dftd numerical-provenance statement.

## Configure D2 or D3

Pass one immutable correction configuration to `Symmetrix`:

```python
from symmetrix import D2Correction, Symmetrix

calc = Symmetrix(
    "model.json",
    corrections=[
        D2Correction(functional="pbe", cutoff=12.0, skin=1.0),
    ],
)
```

D2 uses fixed pair coefficients and zero damping. D3 includes
coordination-dependent pair coefficients and supports exactly `zero`, `bj`,
`zerom`, and `bjm` damping:

```python
from symmetrix import D3Correction, Symmetrix

calc = Symmetrix(
    "model.json",
    corrections=[
        D3Correction(
            functional="pbe",
            damping="bj",
            cutoff=12.0,
            coordination_cutoff=8.0,
            skin=1.0,
        ),
    ],
)
```

All public lengths are in Angstrom. Energies are returned in eV, forces in
eV/Angstrom, and ASE stress in eV/Angstrom\^3 and Voigt order
`(xx, yy, zz, yz, xz, xy)`. Defaults convert the torch-dftd 0.5.3 cutoff of 40
Bohr once at configuration construction; callers still pass Angstrom values.
The correction contributes to `energy`, `free_energy`, per-atom `energies`,
`forces`, and `stress`. Duplicate dispersion corrections are rejected to avoid
double counting.

## Configure reciprocal-space FourierD3

`FourierD3Correction` is a modified model designed for periodic particle-mesh
evaluation. It is not a drop-in numerical replacement for a finite-cutoff
`D3Correction`: it uses a softened coordination law, retains periodic
self-image interactions while cancelling only the zero-cell self term, and
uses the continuous B-spline deconvolution convention. Damping is fixed to
Becke-Johnson and cannot be selected.

```python
from symmetrix import FourierD3Correction, Symmetrix

calc = Symmetrix(
    "model.json",
    corrections=[
        FourierD3Correction(
            functional="pbe",
            coordination_cutoff=8.0,
            mesh_spacing=1.0,
            spline_order=4,
            decomposition_tolerance=1.0e-4,
            chunk_size=512,
            skin=1.0,
        ),
    ],
)
```

FourierD3 requires a finite, nonsingular, fully periodic three-dimensional
cell (`pbc=[True, True, True]`). Slabs, wires, molecules, and singular cells
are rejected before a native evaluator is allocated. There is no real-space D3 fallback.
Absence of the dtype-specific native evaluator or its FFT provider is an
actionable error rather than a model substitution.

Choose exactly one effective mesh policy:

- `mesh_spacing=<Angstrom>` resolves each lattice-vector length with
  `ceil(length / mesh_spacing)`, applies a floor of `spline_order`, and rounds
  upward to a 2/3/5/7-smooth FFT dimension. Omitting both policies selects the
  neutral 1.0 Angstrom spacing.
- `mesh=(nx, ny, nz)` fixes the three dimensions and does not change when the
  cell changes. It is mutually exclusive with `mesh_spacing`.

`spline_order` accepts 2 through 6. `decomposition_tolerance` controls the
low-rank canonical-table approximation; optional `rank_cap` imposes a hard
rank ceiling, while `chunk_size` bounds the rank channels processed per pass.
Diagnostics report whether the tolerance was met, the retained rank, resolved
mesh and realized axis spacings. Production settings should be converged for
the target energy, forces, and stress; a smaller spacing is more accurate and
uses more FFT memory.

The build must discover KokkosFFT 2.0.0 and its backend provider: FFTW on CPU,
cuFFT on CUDA, or hipFFT on HIP. `symmetrix backend show --probe` and
`symmetrix doctor --json` report this capability. CPU/FFTW and CUDA/cuFFT have
been runtime-qualified in both precisions. HIP/hipFFT is also runtime-qualified
in both precisions on the recorded `rocm7-gfx1151` configuration. This
architecture-specific evidence does not qualify other HIP architectures.

## Graphs, skin, and backends

Each correction owns a graph separate from the model graph because its range
and node state differ from those of the neural model. D2 builds candidates to
`cutoff + skin`. D3 builds one candidate pair graph to
`max(cutoff, coordination_cutoff) + skin`, then applies the pair and
coordination cutoffs in their respective kernels. Sharing the shorter model
graph would silently omit long-range interactions.

FourierD3 owns only the short coordination graph at
`coordination_cutoff + skin`; the long-range pair interaction is evaluated on
the reciprocal mesh. It therefore avoids the cubic cutoff growth of a
real-space D3 pair graph. Its short coordination topology may use the native
Kokkos neighbor builder or the host builder and upload path according to the
policy below. That choice affects topology construction only: FourierD3 still
evaluates the long-range dispersion on the reciprocal mesh, with no real-space
D3 fallback.

The correction uses the same half-skin displacement and affine-cell validity
rules as the model graph. A valid step uploads only positions and cell data;
it does not rebuild topology. Set `skin=0` to rebuild on every call.

Correction kernels are Kokkos implementations available in CPU/OpenMP, CUDA,
and HIP builds. Fully periodic cells with a nonsingular cell matrix are
eligible for native Kokkos neighbor construction. Under
`SYMMETRIX_NEIGHBOR_BACKEND=automatic`, CUDA and HIP use that builder at 512
atoms or more; smaller systems and CPU builds use the host builder by default.
`SYMMETRIX_NEIGHBOR_BACKEND=kokkos` forces native construction when eligible,
and `host` forces the host builder. Partial or nonperiodic cells use the host
builder and upload receiver-major topology once. A requested Kokkos build
failure is reported; it is not silently retried on the host.

The D3 evaluator uses the same source on every Kokkos backend. It evaluates
the coordination switching function and its radial derivative once per
directed edge and caches both for the coordination and force-adjoint passes.
It then computes, once per atom, normalized weights over that element's active
coordination references and the derivatives of those weights. Pairwise C6 and
both coordination derivatives are contractions of the two atoms' cached
weight rows with the reference C6 table; the expensive normalization is not
repeated for every edge. One Kokkos team owns each receiver and reduces all of
its edges, both for pair energy/direct force and for the coordination adjoint,
without atomics. A final single reduction accumulates and exports all six
stress components.

The D3 cache uses grow-only device views. For scalar size `sizeof(Precision)`,
its capacity allocation is
`2 * atom_capacity * 5 * sizeof(Precision) + 2 * edge_capacity *
sizeof(Precision)`: two `[atom_capacity][5]` reference-weight arrays and two
`[edge_capacity]` coordination-switch arrays. Diagnostic fields whose names
end in `workspace_bytes` report this capacity allocation, not the number of
currently active bytes. A graph rebuilt for a larger system can therefore
retain its high-water capacity for later smaller systems.

The implementation is backend-portable, but a particular deployment is only
qualified by tests run with its exact backend package and device architecture.
Accelerator required-test manifests exercise graph residency, reuse, and
precision on appropriately equipped CI or deployment hosts. A successful
compile for an architecture does not replace runtime qualification on a device
with a visible driver.

After a calculation, `calc.correction_diagnostics` is a tuple with one entry
per correction. Each entry reports `method`, `graph_backend`,
`graph_build_count`, `graph_reuse_count`, `geometry_update_count`,
`backend_fallback_count`, `num_edges`, `pair_cutoff`, `coordination_cutoff`, and
`skin`. A nonzero fallback count is not expected: native graph failures are
fail-fast.

FourierD3 adds provider/version, requested and resolved mesh, realized
spacing, decomposition accuracy/rank, graph and cache generations, exactly
reconcilable active/capacity memory components, and FFT-plan ownership fields.
Synchronized stage timing is an opt-in diagnostic because device fencing
perturbs execution. Performance measurements should use
`benchmarks/dispersion_kokkos_timing.py`, which excludes construction and
warmup and validates the serialized record before writing it.

## Legacy migration

The former constructor spelling remains for one compatibility cycle:

```python
# Deprecated: emits FutureWarning.
calc = Symmetrix(
    "model.json",
    dispersion=True,
    dispersion_damping="bj",
    dispersion_xc="pbe",
    dispersion_cutoff=12.0,
)
```

It maps to one `D3Correction`; an explicit legacy cutoff also caps the mapped
coordination cutoff at the default 40-Bohr value. Do not pass `corrections`
with `dispersion=True`. `dispersion_device` no longer chooses a device: the
correction follows the loaded Symmetrix backend, and an incompatible legacy
request is rejected.

## Scope and future methods

The real-space D3 correction is two-body only. FourierD3 is likewise a
two-body modified model, not a real-space/reciprocal Ewald split. D3
Axilrod-Teller-Muto (ATM) is not
implemented because it needs triplet topology. DFT-D4 is not implemented; it
needs charge equilibration and pair/triplet topology. Tkatchenko-Scheffler and
many-body dispersion (MBD) are also unsupported; MBD needs a global
coupled-dipole solve. These methods cannot be represented honestly as another
pair kernel and will require new runtime implementations and graph
requirements.

The pinned torch-dftd 0.5.3 source and compact reference fixtures are used only
to regenerate and validate numerical data during development. They are not
loaded by an installed calculator. LAMMPS calculations can continue to use
LAMMPS's own `pair_style dispersion/d3`; this native integration targets ASE.
