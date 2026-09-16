# LAMMPS Integration

Use LAMMPS 10 Dec 2025 or newer, CMake 3.27 or newer, a C++20 compiler, and a
recursive Symmetrix-XL checkout. CPU builds additionally require a Fortran
compiler and an optimized OpenBLAS installation. The build helper installs the
pair style, enables Kokkos, builds LAMMPS, and verifies its pair styles:

```bash
python tools/symmetrix_build.py lammps \
    --source /path/to/lammps --prefix /path/to/lammps-install \
    --backend cpu --cpu-target x86-64-v3 --mpi on
```

Use `--cpu-target native` for a machine-local CPU build. `--mpi on` fails if an
MPI C++ wrapper is unavailable; pass `--mpi-cxx /path/to/mpicxx` when the module
environment does not expose it under the usual name. Use `--mpi auto` only when
silently producing a non-MPI build is acceptable.

CUDA and HIP LAMMPS executables are separate builds. Select the architecture
that will run the executable:

```bash
python tools/symmetrix_build.py lammps \
    --source /path/to/lammps --prefix /path/to/lammps-cuda-sm120 \
    --backend cuda --arch sm120 --cuda-root /usr/local/cuda-13.3 --mpi on

python tools/symmetrix_build.py lammps \
    --source /path/to/lammps --prefix /path/to/lammps-hip-gfx1151 \
    --backend hip --arch gfx1151 --rocm-root /opt/rocm --mpi on
```

The currently qualified accelerator configuration uses CUDA or HIP for device
execution and Kokkos Serial for host work; Kokkos OpenMP is disabled. This is a
tested configuration rather than an intrinsic Kokkos requirement.

For a quick diagnostic CPU run without a prepared artifact, use the
compiler-free Kokkos `non-compiled` fallback with a compact JSON model. It has no
performance guarantee. The first-class production path is the direct example
below. The atom-ID map and Newton pair setting are required:

```text
units metal
atom_style atomic
atom_modify map yes
package kokkos neigh half newton on
newton on
pair_style symmetrix/mace/kk streamed_edges non-compiled
pair_coeff * * /absolute/path/srtio3-mace.json Sr Ti O
run 0
```

The three element names map LAMMPS atom types 1, 2, and 3 to Sr, Ti, and O.
They must match the atom types in the SrTiO3 data file and be supported by the
model.

LAMMPS does not compile direct artifacts. Prepare the matching artifact before
a direct production run. A CPU/OpenMP direct run uses a host artifact:

```bash
jit_artifact=$(symmetrix_prepare_jit_host_artifact \
    --model srtio3-mace.json --precision float32 --path-only)
lmp -var jit_artifact "$jit_artifact" -in in.mace
```

```text
pair_style symmetrix/mace/float32/kk mpi_message_passing streamed_edges direct \
           profile capacity jit_host_artifact ${jit_artifact}
pair_coeff * * srtio3-mace.json Sr Ti O
```

`profile capacity` is the default and matches ASE's capacity profile. Use
`profile speed` to select the retained throughput plan. The deprecated
`low_memory yes|no` spelling maps to `profile capacity|speed`; the alias never
enables fixed workspace.
`allow_fixed_workspace` defaults to `no`. Set it to `yes` only with
`profile capacity` to permit bounded tiled workspace selection; it does not
force that plan when another qualified capacity plan is preferred.
Single-layer direct MPI supports retained and fixed-workspace plans without H1
communication and does not require an R1 artifact. Dual-layer fixed-workspace
MPI is not supported yet.

Artifact preparation uses the Python runtime, not the LAMMPS executable. Install
the `symmetrix-xl` CPU frontend first and then install the backend matching the
LAMMPS executable. For CUDA/HIP, activate that backend and generate the device
artifact on the deployment GPU:

```bash
python tools/symmetrix_build.py install --backend cpu --cpu-target native
python tools/symmetrix_build.py install \
    --backend cuda --arch sm120 --cuda-root /usr/local/cuda-13.3

jit_arguments=$(symmetrix_prepare_jit_device_artifact \
    --model srtio3-mace.json --precision float32 --lammps-arguments)
lmp -var jit_arguments "$jit_arguments" -in in.mace
```

For HIP, replace the second install command with `--backend hip --arch gfx1151
--rocm-root /opt/rocm`. Backend packages are architecture-specific; do not use
an artifact or native backend built for a different device target.

Use the rendered arguments with `symmetrix/mace/float32/kk`. Use
`--precision float64` with `symmetrix/mace/kk` when FP64 is required. Use
`no_domain_decomposition` for a one-rank GPU input or
`mpi_message_passing` for the qualified direct MPI mode. Use the float32 pair
style only with a float32 artifact. MACEField is Kokkos-only and needs
`electric_field Ex Ey Ez`. The detailed, retained `pair_symmetrix/README.md`
covers pair-style options, MPI constraints, and CUDA-aware MPI qualification.
