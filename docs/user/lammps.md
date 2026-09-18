# LAMMPS Integration

Use LAMMPS 10 Dec 2025 or newer, CMake 3.27 or newer, a C++20 compiler, and a
recursive Symmetrix-XL checkout. CPU builds additionally require a Fortran
compiler and a supported optimized BLAS installation (OpenBLAS or MKL). The
build helper installs the pair style, enables Kokkos, builds LAMMPS, and
verifies its pair styles:

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
    --backend cuda --arch sm120 --cuda-root /usr/local/cuda-13.3 \
    --lammps-package KSPACE --lammps-package EXTRA-PAIR \
    --lammps-cmake-define FFT=FFTW3 \
    --require-gpu-aware-mpi

python tools/symmetrix_build.py lammps \
    --source /path/to/lammps --prefix /path/to/lammps-hip-gfx1151 \
    --backend hip --arch gfx1151 --rocm-root /opt/rocm \
    --require-gpu-aware-mpi
```

The helper does not download LAMMPS. `--source` must point to an existing local
LAMMPS source checkout; a binary-only LAMMPS installation cannot be extended
with this pair style. The installer adds Symmetrix source links and a marked
CMake block to that checkout, then builds into a separate fingerprinted
directory and installs under `--prefix`. Repeat `--lammps-package` for each
additional package. In the example, `KSPACE` supplies long-range electrostatics
such as Ewald and PPPM, while `EXTRA-PAIR` supplies `pair_style dispersion/d3`.
Requested packages are checked against the source tree before configuration and
against `lmp -help` after installation.

Use repeatable `--lammps-cmake-define NAME=VALUE` arguments for package-specific
choices such as `FFT=FFTW3`, `FFT=KISS`, or external-library discovery options.
These values and package selections are part of the build fingerprint, so a
different LAMMPS composition gets a fresh build directory. Backend, MPI,
Kokkos, architecture, `SYMMETRIX_*`, and `PKG_*` definitions remain managed by
the frontend; select packages with `--lammps-package` instead of passing
`PKG_NAME=ON` directly.

Package installation does not select the corrections at runtime. For example,
the following illustrates a learned potential overlaid with explicit Coulomb
and D3 terms:

```text
atom_style charge
pair_style hybrid/overlay \
    symmetrix/mace/kk streamed_edges non-compiled \
    coul/long 10.0 \
    dispersion/d3 original pbe 30.0 20.0
pair_coeff * * symmetrix/mace/kk /absolute/path/model.json C H O
pair_coeff * * coul/long
pair_coeff * * dispersion/d3 C H O
kspace_style pppm 1.0e-5
```

Set atom charges and replace the elements, D3 damping method, functional, and
cutoffs with values appropriate to the model. Only overlay a correction that
is absent from the model's training target, otherwise its energy and forces are
double counted. See the LAMMPS
[`hybrid/overlay`](https://docs.lammps.org/pair_hybrid.html),
[`dispersion/d3`](https://docs.lammps.org/pair_dispersion_d3.html), and
[`kspace_style`](https://docs.lammps.org/kspace_style.html) documentation for
the full runtime syntax.

`--require-gpu-aware-mpi` implies `--mpi on`. Before CMake runs, the helper
compiles a small program with the selected `mpicxx` and requires the same
provider extension that LAMMPS queries: `MPIX_Query_cuda_support()` for CUDA or
`MPIX_Query_rocm_support()` for HIP. An MPI installation without that extension,
or one whose query returns false, fails immediately. Select a site or vendor
wrapper explicitly with `--mpi-cxx /path/to/mpicxx`. The positive query, wrapper
identity, and wrapper path are recorded under
`provenance.mpi.gpu_aware` in `qualification.json` and
`lammps-invocation.json`.

Use ordinary `--mpi on` without `--require-gpu-aware-mpi` when host-staged
communication is intentional. The helper cannot prove arbitrary vendor-specific
GPU transport interfaces, so the required mode currently accepts providers
that expose the Open MPI-compatible `MPIX` query used by LAMMPS.

The currently qualified accelerator configuration uses CUDA or HIP for device
execution and Kokkos Serial for host work; Kokkos OpenMP is disabled. This is a
tested configuration rather than an intrinsic Kokkos requirement.

The build-time query proves that the linked MPI provider advertises device
buffers; it does not prove the selected inter-node transport. Qualify the
installed executable in the same module and scheduler environment used for
production. Launch one rank per GPU and let LAMMPS auto-detect GPU awareness:

```bash
mpirun -np 2 /path/to/lammps-cuda-sm120/bin/lmp \
    -k on g 2 -sf kk -pk kokkos neigh half newton on \
    -in in.mace 2>&1 | tee lammps-gpu-aware.log

rg 'Turning off GPU-aware MPI|GPU-aware MPI' lammps-gpu-aware.log
```

When the scheduler exposes one GPU to each rank, use `-k on g 1`. A qualified
run has a positive recorded `MPIX` query and no `Turning off GPU-aware MPI since
it is not detected` warning. Do not force `-pk kokkos gpu/aware on` to suppress
auto-detection: a successful multi-rank model calculation, correct provider
capability query, and absence of the fallback warning are the required evidence.
Cross-node GPUDirect qualification remains specific to the site's MPI provider,
network, and scheduler configuration.

## See whether communication dominates

LAMMPS accounts the Symmetrix hidden-state exchange inside its `Pair` category,
because the exchange occurs during pair evaluation. Add the Symmetrix timing
compute to separate that work and display its share directly:

```text
compute sxt all symmetrix/timing
timer full
thermo 1000
thermo_style custom step atoms c_sxt c_sxt[1] c_sxt[2] c_sxt[3] c_sxt[10]
thermo_modify colname auto
run 1000
```

The headings are supplied by the compute, so the final row is self-describing:

```text
Step  Atoms  SxH1CommPct  SxPair  SxNonComm  SxH1Comm  SxH1Imbal
1000  32768          58.3    4.82       2.01      2.81       1.07
```

`SxPair`, `SxNonComm`, and `SxH1Comm` are run-to-date averages in
`us/atom/pair-evaluation` from the rank with the largest measured Symmetrix pair
time. The normalization assumes the atom count remains fixed during the measured
run. They are also `us/atom/step` when the pair is evaluated once per timestep.
`SxPair = SxNonComm + SxH1Comm`, and `SxH1CommPct` is the matching,
rank-correlated hidden-state communication share. A value of 50% or more means
hidden-state communication dominates Symmetrix pair evaluation on the critical
rank. It does not mean that communication occupies the same fraction of the
complete LAMMPS timestep; inspect LAMMPS's `timer full` breakdown for that
broader question. `SxH1Imbal` is maximum divided by mean hidden-state
communication time across ranks; values appreciably above one indicate
imbalance or uneven MPI wait.

The timed communication path includes packing, host/device staging, MPI send and
wait, unpacking, and completion at the existing Kokkos fences. It is deliberately
not called network latency: load imbalance can appear as MPI wait. Ordinary
LAMMPS halo and force communication remains in LAMMPS's `Comm` category.
Defining `compute symmetrix/timing` also places fences around the complete
Kokkos pair call so its denominator includes completed device work. Those
measurement-only fences are absent when the compute is not defined; compare a
warmed run with and without the compute when quantifying instrumentation
overhead.

The run-to-date measurement window begins after LAMMPS's setup force evaluation.
The initial thermo row therefore reports `nan` for the percentage because no
timed pair evaluation has occurred. Querying the compute performs an MPI
reduction, so use a thermo interval equal to the measurement block instead of
printing it every timestep. The remaining vector fields provide forward/reverse
and rank-spread diagnostics:

| Field | Meaning |
| --- | --- |
| `c_sxt[4]` | Same critical-rank percentage as scalar `c_sxt` |
| `c_sxt[5]`, `c_sxt[6]` | Critical-rank forward and reverse communication in `us/atom/pair-evaluation` |
| `c_sxt[7:9]` | Minimum, mean, and maximum communication percentage across ranks |
| `c_sxt[11:12]` | Minimum and maximum forward calls per pair evaluation |
| `c_sxt[13:14]` | Minimum and maximum reverse calls per pair evaluation |

Dual-layer `mpi_message_passing` normally reports one forward and one reverse
call per pair evaluation. Single-layer and non-message-passing modes report zero
communication time and calls. Library clients can also read the cumulative
rank-local `symmetrix_mpi_hidden_state_seconds`, forward/reverse seconds and call
counters, and `symmetrix_pair_seconds` through `extract_pair`.
The complete-pair counter is populated after `compute symmetrix/timing` has
initialized for the current pair style, because accurate accelerator timing
requires the bounding fences.

`--dry-run` performs target and MPI preflight checks but does not create or
modify a build directory. It prints one JSON document containing the resolved
commands, selected environment, fingerprint, and provenance.

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
Single-layer direct MPI does not require an R1 artifact. Its retained plan is
backend-independent; its fixed-workspace plan currently requires CUDA FP32.
Dual-layer fixed-workspace MPI requires an ordinary two-layer standard-MACE
model, CUDA FP32, direct prepared execution, and a generated device artifact
with tiled R1 support. It bounds receiver and edge intermediates but retains
local-plus-ghost H1 and H1-adjoint state. MACEField, FP64, HIP, parameter
gradients, and execution observers remain unsupported for this tiled MPI plan.
LAMMPS communication counts remain signed `int` values, so the H1 width times
the largest rank-local ghost count must fit that ABI. The bound applies to a
halo message, not to the full local-plus-ghost feature tensor.

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
