# Automatic build, backend-wheel, and LAMMPS packaging plan

Date: 2026-08-27

Implementation status: repository-local `detect`, `preflight`, `install`,
`wheel`, `wheel-matrix`, `sdist`, and `lammps` commands are implemented. The
base distribution now owns the Python frontend plus CPU/OpenMP, and exact
CUDA/HIP builds generate non-overlapping native-only packages discovered by
the runtime selector; `x86-64-v4` CPU builds are additional
architecture-qualified backend packages published as
`symmetrix-xl-cpu-avx512`; GPU backend wheels are built locally with
`tools/build_gpu_wheels.sh`. The sdist command vendors submodules at their
pinned commits, refuses a dirty worktree (override with `--allow-dirty`),
and smoke-checks the produced sdist by rebuilding it from itself. The CI
release workflow pins its toolchain downloads with SHA-256 checksums and
gates publication on wheel installation plus an AVX-512 code-generation
sentinel. The release index/publication gate, upgrade
qualification, and `lammps-prepare-model` remain planned release work.

## Objective

Make a backend-correct Symmetrix installation a short, explicit operation while
preserving the backend and architecture constraints imposed by Kokkos.

The implementation must support three related workflows from one configuration
model:

1. build and install Symmetrix from a source checkout;
2. build architecture-specific Python wheels for release;
3. integrate and build `pair_symmetrix` with a compatible LAMMPS source tree.

The public source-build entry point is a standalone repository script:

```bash
python tools/symmetrix_build.py install --backend cpu --cpu-target native
# Optional after the base install:
python tools/symmetrix_build.py install --backend cuda --arch sm120
```

It is build tooling, not a Symmetrix runtime command. It must not be installed
by any Symmetrix wheel, must not be registered in `[project.scripts]`, and must
not import the `symmetrix` package. Model conversion, runtime diagnostics, and
JIT artifact preparation remain installed runtime commands because those
operations require the native evaluator.

## Fixed design decisions

### One native target per build

Every native build selects exactly one of:

- CPU plus one host target, initially `native` or `x86-64-v3`;
- CUDA plus one compute capability, such as `sm86` or `sm120`;
- HIP plus one exact AMD ISA, such as `gfx90a` or `gfx1151`.

The bundled Kokkos configuration permits one GPU architecture per build. A
single "fat" CUDA/HIP Symmetrix wheel is therefore not a supported release
unit. CPU, CUDA, and HIP builds never share a CMake cache.

### Detection before CMake compiler initialization

Compiler selection cannot be implemented only inside the current
`symmetrix/CMakeLists.txt`: CMake initializes C and C++ compilers at
`project()`, before the existing backend policy executes. The standalone tool
must resolve the backend, target, compiler, toolkit, and generator first, then
invoke pip/scikit-build or CMake with a complete configuration.

CMake remains an independent validation boundary. It must reject a manifest
translated into contradictory flags, an unsupported compiler, multiple GPU
traits, or a target that does not match the selected backend.

### The build tool is not installed

The build implementation lives outside `symmetrix/source/symmetrix`:

```text
tools/
  symmetrix_build.py
  _symmetrix_build/
    __init__.py
    cli.py
    command.py
    detect.py
    manifest.py
    targets.py
    python_build.py
    lammps_build.py
```

The modules use the Python standard library. Subprocesses are always invoked
with argument arrays, never shell command strings. The tool is tested by path
from the checkout. Python wheels must contain none of these files and must not
expose a `symmetrix-build` entry point.

The existing installed commands remain separate:

```text
symmetrix doctor
symmetrix_extract_mace
symmetrix_prepare_jit_host_artifact
symmetrix_prepare_jit_device_artifact
```

### One versioned target manifest

Detection produces a JSON manifest. Python and LAMMPS command generation must
consume this manifest rather than maintaining separate architecture tables or
backend defaults.

An illustrative version-1 record is:

```json
{
  "architecture": {
    "compiler_target": "sm_120",
    "device_target": "sm120",
    "kokkos_trait": "BLACKWELL120",
    "requested": "auto"
  },
  "backend": "cuda",
  "blas_policy": "kokkos",
  "device_probe": "/usr/bin/nvidia-smi",
  "host_target": "none",
  "python_abi": "cpython-314",
  "python_executable": "/path/to/venv/bin/python",
  "schema_version": 1,
  "toolchain": {
    "cmake": "/absolute/path/to/cmake",
    "cmake_version": "4.2.0",
    "cxx": "/absolute/path/to/nvcc_wrapper",
    "cxx_version": "g++ 15.2.0",
    "generator": "Unix Makefiles",
    "host_cxx": "/usr/bin/g++",
    "runtime_library_dirs": ["/usr/local/cuda-13.3/lib64"],
    "toolkit_root": "/usr/local/cuda-13.3",
    "toolkit_version": "13.3"
  },
  "visible_devices": ["sm120"]
}
```

The committed schema defines required fields, normalized target spelling, and
forward-compatibility behavior. Unknown schema versions fail closed. Absolute
compiler and toolkit paths, tool versions, Python ABI, generator, backend,
architecture, host target, and BLAS policy contribute to the configuration
fingerprint.

### Explicit selection overrides automatic detection

The primary selectors are command-line options:

```text
--backend auto|cpu|cuda|hip
--arch auto|sm70|sm80|sm86|sm89|sm90|sm100|sm120|gfx...
--cpu-target native|x86-64-v3|none
--cuda-root PATH
--rocm-root PATH
--cxx PATH
--host-cxx PATH
--generator NAME
--build-root PATH
```

Precedence is:

1. explicit command-line value;
2. value loaded with `--target-manifest`;
3. automatic probe;
4. documented backend default.

The first implementation does not add a second set of build environment
variables. CI and cluster scripts should use explicit options or a checked
manifest. Existing low-level CMake settings remain available as an advanced
escape hatch but are not the normal interface.

## Automatic detection policy

### Source checkout preflight

Before selecting a backend, validate:

- Python 3.10 or newer and its ABI tag;
- CMake 3.27 or newer;
- a recursive checkout with the pinned Kokkos, KokkosKernels, SpheriCart,
  pybind11, and JSON sources;
- a C++20-capable host compiler;
- sufficient write access to the selected build root;
- a supported platform, initially Linux x86-64;
- the requested build directory does not contain a different manifest
  fingerprint.

CPU preflight additionally resolves OpenMP, a Fortran compiler, and an
optimized BLAS. It must distinguish an optimized provider from reference BLAS
and report the resolved library before compilation. OpenBLAS remains the
`auto` preference when available; Intel oneMKL is selectable with `--blas mkl`
or `--blas mkl-sequential` and is discovered from `MKLROOT`, an explicit
`--mkl-root`, or the standard oneAPI installation. Only MKL LP64 CMake vendors
are supported: `Intel10_64lp` selects GNU OpenMP threading and
`Intel10_64lp_seq` selects the single-threaded layer. For CPU Kokkos
deployments, the sequential MKL policy is recommended because Kokkos supplies
the outer parallelism and avoids nested teams and OpenMP runtime coupling.
Use the GNU-threaded policy only after qualifying its shared `libgomp` runtime
and measuring a benefit. Fortran must be enabled
before BLAS discovery so CMake can configure MKL's imported targets. Both
policies record and pass `gfortran` explicitly so CMake chooses MKL's GNU
LP64 interface. The GNU-threaded policy also requires a GNU C++ compiler; this
makes Kokkos and MKL share `libgomp`. Clang builds with an available OpenMP
development runtime must use the sequential MKL policy with `gfortran` retained
as the Fortran compiler.

### Backend auto-selection

`--backend auto` follows this policy:

1. Probe visible NVIDIA devices and visible AMD agents without initializing
   Kokkos.
2. If exactly one vendor is visible and its complete toolchain is usable,
   select that vendor.
3. If both vendors are visible, fail and require `--backend`.
4. If visible devices from one vendor have different architectures, fail and
   require `--arch`; do not pick device zero silently.
5. If a GPU is visible but its compiler/toolkit is missing, fail with the
   missing paths and versions. Do not silently produce a CPU build.
6. If no GPU is visible, select the OpenMP CPU build.

This makes accidental CPU builds on GPU workstations visible while retaining a
simple default on ordinary CPU systems. Login-node and cross-compilation builds
must use an explicit backend and architecture because device discovery is not
authoritative there.

### CUDA detection

Use `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` when a device is
visible. Normalize `8.6` to `sm86`, for example, and map it through a committed
table to the pinned Kokkos trait:

| Device target | Kokkos trait |
| --- | --- |
| `sm70` | `VOLTA70` |
| `sm80` | `AMPERE80` |
| `sm86` | `AMPERE86` |
| `sm87` | `AMPERE87` |
| `sm89` | `ADA89` |
| `sm90` | `HOPPER90` |
| `sm100` | `BLACKWELL100` |
| `sm120` | `BLACKWELL120` |

Reject a detected target not supported by the pinned Kokkos version. Locate
`nvcc` and derive the toolkit root from its canonical path. Validate the
toolkit version against the target, select the bundled `nvcc_wrapper`, resolve
a supported host C++ compiler, and set `NVCC_WRAPPER_DEFAULT_COMPILER` plus
`CMAKE_CUDA_HOST_COMPILER` explicitly. CUDA uses `Unix Makefiles` until the
documented CMake/Ninja dependency-scanning incompatibility is removed.

### HIP detection

Use `rocm_agent_enumerator`, filter non-GPU entries such as `gfx000`, retain
feature suffixes for diagnostic evidence, and normalize the base ISA. Locate
`hipcc`, derive the ROCm root, and require hipcc 6.2 or newer for the pinned
Kokkos.

Map exact ISAs to a Kokkos trait and compiler target separately. This is
required for targets such as `gfx1151`, where pinned Kokkos provides compatible
RDNA3 `GFX1100` traits but must compile and link for the exact `gfx1151` ISA:

```text
Kokkos_ARCH_AMD_GFX1100=ON
Kokkos_IMPL_AMDGPU_FLAGS=--offload-arch=gfx1151
Kokkos_IMPL_AMDGPU_LINK=--offload-arch=gfx1151
```

HIP forces Kokkos Serial as the host execution space, disables Kokkos OpenMP,
host-native architecture flags, CUDA SpheriCart, and global fast-math. rocBLAS
selection uses the existing `SYMMETRIX_HIP_BLAS=AUTO` policy unless explicitly
overridden.

### CPU detection

The source-build default is `--cpu-target native`. Release wheel jobs must
select an explicit portable baseline. The initial portable x86 target is
`x86-64-v3`; a baseline wheel for older x86-64 systems is a separate release
decision and must not be mislabeled as v3.

CPU detection selects Kokkos OpenMP, disables Kokkos Serial and accelerator
backends, enables the host BLAS TPL, and disables CUDA SpheriCart. It records
the exact OpenMP and BLAS libraries for post-install verification. Runtime
worker admission allows MKL sequential CBLAS and GNU-threaded MKL only when it
shares Kokkos's `libgomp`; Intel-threaded MKL is kept on the Kokkos contraction
path to avoid two OpenMP runtimes in one process.

## Source-build interface

### Commands

The implemented source-build commands are:

```bash
# Print probes and the resolved configuration without writing a build.
python tools/symmetrix_build.py detect --backend auto

# Validate prerequisites only.
python tools/symmetrix_build.py preflight --backend hip --arch gfx1151

# Install the frontend and default CPU backend.
python tools/symmetrix_build.py install --backend cpu --cpu-target native

# Add an architecture-qualified backend to that environment.
python tools/symmetrix_build.py install --backend cuda --arch sm120

# Build one wheel without installing it.
python tools/symmetrix_build.py wheel \
    --backend cuda --arch sm120 --wheel-dir dist
```

The matrix command builds several release targets in isolated child
processes. It accepts only add-on targets: base-distribution CPU targets
(`native`, `x86-64-v3`, `none`) belong to the `symmetrix-xl` release
pipeline and are rejected here.

```bash
python tools/symmetrix_build.py wheel-matrix \
    --target cpu:avx512 \
    --target cuda:13:sm86 \
    --target cuda:13:sm120 \
    --target hip:7:gfx1151 \
    --wheel-dir dist
```

`detect` emits a human summary by default and JSON with `--json`. Every build
stores the normalized manifest, exact invoked command, and combined configure
and build log under its build directory. Successful installs additionally
store the imported package and extension paths, extension SHA-256, device
sentinel, and doctor report.

### Isolated build directories

Replace the single hard-coded scikit-build directory with a target-specific
directory:

```text
symmetrix/build-<configuration-fingerprint>/
```

The tool may reuse that directory only when its manifest matches exactly.
Changing Python, compiler, backend, toolkit, architecture, host target,
SpheriCart policy, or BLAS policy selects a different fingerprint. It never
repairs a mixed cache and never deletes an unvalidated broad path.

Direct `pip install .` remains a low-level developer operation. Documentation
must recommend the standalone tool whenever automatic backend selection is
desired.

### Post-install gate

After installation, start a fresh Python process and record:

- imported `symmetrix` package path;
- imported native extension path and SHA-256;
- Kokkos default execution space;
- recorded compiler and architecture metadata;
- runtime device architecture for CUDA/HIP;
- loaded OpenMP and BLAS libraries for CPU;
- `symmetrix doctor --json` result.

Run a minimal native device sentinel for the selected backend. A successful
compile without a successful fresh-process import is not an accepted install.

## Architecture-specific wheel design

### Distribution ownership

Do not publish several distributions that overwrite the same
`symmetrix/__init__.py` or `symmetrix/symmetrix*.so`. Python packaging has no
portable conflict declaration, and uninstalling either distribution could
remove files owned by another.

Published wheels are split into a usable base distribution and non-overlapping
GPU distributions:

```text
symmetrix-xl
symmetrix-xl-cuda13-sm120
symmetrix-xl-rocm7-gfx1151
```

The base distribution owns all files in the public `symmetrix` package and its
`_native_cpu` extension. GPU distributions own only unique top-level packages:

```text
symmetrix_backend_cuda13_sm120
symmetrix_backend_rocm7_gfx1151
```

Each GPU package pins the exact `symmetrix-xl` release and native ABI generation.
Installing or uninstalling one GPU target cannot remove frontend or CPU files.

Examples:

```bash
pip install symmetrix-xl
pip install symmetrix-xl-cuda13-sm120
pip install symmetrix-xl-rocm7-gfx1151
```

The toolkit generation belongs in the project name because standard wheel
compatibility tags do not encode CUDA, ROCm, or GPU ISA compatibility.

### Unique native modules

The pybind module token, output name, and install package are build parameters.
Backend packages therefore use unique extensions, for example:

```text
symmetrix._native_cpu
symmetrix_backend_cuda13_sm120._native_cuda13_sm120
symmetrix_backend_rocm7_gfx1151._native_rocm7_gfx1151
```

The old monolithic module remains a one-release loader fallback. New base and
backend builds always use the unique names above.

Each GPU distribution publishes a lightweight entry point in the
`symmetrix.backends` group. Its value names a pure metadata package containing
`backend.json`; discovery reads that file through distribution metadata and
does not import the package, native extension, CUDA/HIP, or Kokkos.

The `symmetrix` frontend selects one compatible descriptor and imports only that native
module. It then registers the selected module as `symmetrix.symmetrix` in
`sys.modules` before importing existing calculator and JIT modules. This
preserves the established Python API while avoiding duplicate files.

### Runtime backend selection

Selection occurs once, before Kokkos initialization:

1. enumerate installed backend descriptors;
2. honor an explicit `SYMMETRIX_BACKEND` selector;
3. otherwise probe the visible device vendor and exact architecture;
4. choose an exact compatible GPU backend;
5. use an installed CPU backend when no accelerator is visible;
6. fail with installed and detected targets when selection is ambiguous.

Multiple backend wheels may coexist, but one process loads only one. A request
for `sm120` never loads an `sm86` extension merely because CUDA can JIT some
embedded code. Kokkos and SpheriCart ahead-of-time device code must match the
qualified architecture.

Extend runtime diagnostics with:

```bash
symmetrix backend list
symmetrix backend show
symmetrix doctor --json
```

These are runtime commands and are distinct from the non-installed source
build tool.

### Wheel release manifest

Every release publishes a machine-readable index containing:

- distribution and wheel filenames;
- Symmetrix version, commit, and native ABI generation;
- backend, toolkit generation, device target, and Kokkos trait;
- Python and platform tags;
- compiler/toolkit versions;
- external runtime requirements;
- wheel and native extension SHA-256 values;
- qualification status and test record links.

Wheel publication is blocked if two artifacts claim the same normalized target
or if the distribution name disagrees with its embedded descriptor.

## LAMMPS build design

### Why LAMMPS does not link a Python backend wheel

LAMMPS normally builds Kokkos statically. `pair_symmetrix` must build
`libsymmetrix` against LAMMPS's existing `Kokkos::kokkos` target so the process
contains one Kokkos runtime. A Python backend extension owns a different Kokkos
build and is not a linkable LAMMPS plugin artifact.

Consequently, the automatic LAMMPS path is a coordinated source build, not
installation of a Python wheel into an existing `lmp` executable.

### Commands

The same standalone tool provides:

```bash
python tools/symmetrix_build.py lammps \
    --source /path/to/lammps \
    --backend auto \
    --mpi auto \
    --prefix /path/to/install

python tools/symmetrix_build.py lammps \
    --source /path/to/lammps \
    --target-manifest /path/to/target.json \
    --mpi-cxx /path/to/mpicxx \
    --prefix /path/to/install
```

The command validates the source tree and version, invokes the existing
idempotent pair-style installer, configures a fresh fingerprinted LAMMPS build,
builds and installs `lmp`, and runs post-build checks.

### LAMMPS configuration policy

All Kokkos backend and architecture settings come from the common target
manifest. The LAMMPS adapter additionally sets:

- `CMAKE_CXX_STANDARD=20`;
- `PKG_KOKKOS=ON`;
- the matching LAMMPS Kokkos host/device backend;
- `BUILD_MPI` and `MPI_CXX_COMPILER` from explicit or detected MPI policy;
- the matching Symmetrix SpheriCart and BLAS policies.

For CUDA, use the `nvcc_wrapper` belonging to the Kokkos source used by LAMMPS
and the manifest's host compiler and compute capability. For HIP, use the
manifest's hipcc, exact offload ISA, and Kokkos Serial host execution space.

Before adding `libsymmetrix`, verify that LAMMPS exposes an existing compatible
Kokkos target. Do not add the bundled Symmetrix Kokkos as a second runtime.

### MPI policy

`--mpi auto` locates an MPI C++ wrapper and records its implementation, version,
compiler, and library paths. It does not claim GPU-aware transport merely from
a successful build. CUDA-aware or HIP-aware qualification requires the
provider capability query and runtime evidence described by the existing MPI
tests.

Prebuilt complete LAMMPS binaries may be published for a small qualified
matrix, but they are not the primary deployment mechanism because MPI ABI,
GPU-aware provider, scheduler, and communication policy are site-specific.
Non-MPI CPU binaries are the first reasonable prebuilt target.

### LAMMPS post-build gate

Record and verify:

- LAMMPS version, executable path, and SHA-256;
- compiler, MPI provider, Kokkos execution space, and architecture;
- availability of `symmetrix/mace` and `symmetrix/mace/kk`;
- one native energy/force/stress smoke calculation;
- generic and direct mode behavior appropriate to the available artifacts;
- for MPI builds, a two-rank smoke test when execution is available.

The generated build manifest is installed adjacent to `lmp` for provenance.
It must agree with the Python/native target manifest used to prepare device or
host JIT artifacts.

### Model-specific JIT artifacts

LAMMPS continues not to invoke a runtime compiler. After the backend-specific
Python package is installed, use its existing artifact commands to prepare a
model- and precision-specific host shared library, CUDA cubin, or HIP hsaco.

The standalone tool may orchestrate this as a convenience operation:

```bash
python tools/symmetrix_build.py lammps-prepare-model \
    --model model.json \
    --lammps /path/to/lmp \
    --precision float32,float64 \
    --output model-bundle
```

This operation calls the installed artifact-preparation command in a fresh
process, validates its target against the LAMMPS manifest, and writes a bundle
manifest plus ready-to-use LAMMPS variable definitions. The compiler remains
outside LAMMPS.

Format-v3 nonlinear and MH-1 models remain unsupported by the current LAMMPS
pair styles. The build work must report that limitation clearly and must not
claim that installing a GPU backend enables MH-1 in LAMMPS.

## Implementation milestones

### Milestone 1: target model and detector

1. Add the standalone tool skeleton and versioned manifest dataclasses.
2. Implement normalized CPU, CUDA, and HIP target tables.
3. Add command execution with captured evidence and actionable diagnostics.
4. Implement source, compiler, toolkit, device, BLAS, and OpenMP preflight.
5. Add `detect`, `preflight`, and `--json` without invoking a build.
6. Add unit tests using fake executable outputs and temporary source trees.

Acceptance:

- local CPU, CUDA, and HIP fixtures resolve deterministic manifests;
- ambiguous vendors and heterogeneous architectures fail;
- explicit cross-compilation works without a visible GPU;
- unsupported targets fail before CMake;
- no build-tool file appears in a constructed runtime-wheel file list.

### Milestone 2: automatic source install

1. Translate one manifest into scikit-build CMake definitions and compiler
   environment.
2. Replace the fixed build directory with a configuration fingerprint.
3. Implement `install` and single-target `wheel`.
4. Retain CMake-side contradiction and version checks.
5. Run fresh-process import, backend, architecture, hash, and doctor checks.
6. Reduce the README default installation to the standalone command and move
   raw CMake flags into an advanced reference section.

Acceptance:

- one command builds CPU OpenMP on a CPU-only host;
- one command builds exact-target HIP on the local `gfx1151` system;
- one command builds CUDA on a qualified CUDA worker;
- rebuilding another backend cannot reuse the first backend's cache;
- explicit architecture selection is recorded and verified at runtime.

### Milestone 3: release wheel split and loader

1. Implemented: make `symmetrix` own the frontend and CPU backend while GPU
   builds use target-specific native projects.
2. Implemented: parameterize the pybind module name and native output package.
3. Implemented: add import-free descriptors and the one-time runtime selector.
4. Implemented: preserve `symmetrix.symmetrix` as the selected-module alias.
5. Implemented: generate exact-version, architecture-qualified GPU packages.
6. Implement the release wheel matrix and manifest.
7. Verify wheel contents and uninstall behavior for co-installed backends.

Acceptance:

- CPU and GPU backend distributions share no installed files;
- two backend wheels can coexist while only one native module is imported;
- exact architecture selection succeeds and a mismatch fails clearly;
- `pip uninstall` of one backend does not damage core or another backend;
- wheel tags, project names, descriptors, and embedded manifests agree;
- `tools/symmetrix_build.py` and `_symmetrix_build` are absent from all wheels.

### Milestone 4: automatic LAMMPS build

1. Add LAMMPS version/source validation around the existing installer.
2. Generate LAMMPS CMake options from the common target manifest.
3. Detect and record MPI without overclaiming GPU awareness.
4. Add fingerprinted configure/build/install directories.
5. Add executable provenance and serial smoke tests.
6. Add artifact-target comparison and model-bundle generation.
7. Run the existing CPU and accelerator LAMMPS qualification matrices.

Acceptance:

- CPU and local HIP LAMMPS builds use the same target selection as Python;
- the final executable contains one Kokkos runtime;
- pair styles are present and a smoke calculation passes;
- incompatible Python artifact and LAMMPS targets are rejected before launch;
- existing MPI ownership, migration, and correctness tests remain passing.

### Milestone 5: release automation and documentation

1. Add CI jobs for detector tests, CPU wheels, selected CUDA/HIP wheels, and
   wheel-content audits.
2. Publish the release target manifest with hashes and qualification links.
3. Document local auto-detection, explicit target selection, cluster
   cross-compilation, wheel choice, and LAMMPS source builds separately.
4. Keep raw low-level commands as reproducibility references rather than the
   first installation path.

## Test strategy

### Unit tests

- Target spelling, toolkit generation, and Kokkos trait mappings.
- NVIDIA and AMD probe parsing, including empty, duplicate, feature-suffixed,
  heterogeneous, malformed, and unsupported outputs.
- Selection precedence and ambiguity errors.
- Compiler and toolkit version parsing.
- Manifest serialization, schema rejection, and stable fingerprinting.
- CMake and pip argument generation without shell interpretation.
- LAMMPS version parsing and CMake option generation.
- Safe build-root validation.

### Build tests

- Fresh CPU native and x86-64-v3 builds.
- Fresh local HIP `gfx1151` build with exact AOT target verification.
- CUDA compile and runtime builds for each published compute capability on
  appropriate workers.
- Repeated same-target build reuse and different-target isolation.
- Wheel archive inspection proving no build-tool files are installed.

### Runtime tests

- Core plus one backend.
- Core plus CPU and one GPU backend.
- Core plus multiple GPU targets with automatic and explicit selection.
- Missing, incompatible, and ambiguous backend diagnostics.
- One Kokkos initialization per process.
- Existing native, calculator, JIT, OpenMP doctor, and GPU correctness suites.

### LAMMPS tests

- Existing installer location, idempotence, and minimum-version tests.
- CPU non-MPI and MPI builds.
- CUDA/HIP builds on qualified workers.
- Pair-style discovery and energy/force/stress smoke tests.
- Direct artifact target mismatch rejection.
- Multi-rank ownership migration and prepared-graph reuse.
- Positive provider evidence before claiming GPU-aware MPI.

## Non-goals for the first release

- Loading two Kokkos backends in one Python process.
- A single multi-architecture CUDA or HIP native wheel.
- Treating CUDA PTX compatibility as qualification for a different AOT Kokkos
  target.
- Automatically choosing an architecture on a GPU-less login node.
- Bundling arbitrary site MPI stacks in Python wheels.
- Making LAMMPS compile JIT artifacts.
- Enabling format-v3 nonlinear or MH-1 LAMMPS execution as part of packaging.
- Windows, macOS, or non-x86 Linux release wheels before the Linux x86-64
  design is qualified.

## Completion criteria

The work is complete when a local user can build the correct source backend
without writing Kokkos flags, a release user can select an explicitly named
architecture wheel without file conflicts, and a LAMMPS user can build an
architecture-consistent executable from the same target manifest. Every path
must fail early on ambiguous or incompatible hardware, preserve one Kokkos
runtime per process, and retain enough provenance to reproduce the resulting
binary.
