# Build Frontend

`tools/symmetrix_build.py` is the maintained entry point for isolated backend
builds and release artifacts. It resolves a target, validates its toolchain,
and records the selected package and backend configuration in the build
manifest. Run commands from the repository root with the Python interpreter
for the target environment.

| Command | Purpose |
|---|---|
| `detect` | Resolve a CPU, CUDA, or HIP target and print or write its manifest. |
| `preflight` | Validate the requested target and toolchain without compiling. |
| `install` | Build and install the selected frontend/backend into the active Python environment. |
| `develop` | Incrementally rebuild and replace only an already installed native runtime. |
| `wheel` | Build one backend-specific wheel into a required output directory. |
| `wheel-matrix` | Build repeated `--target` release specifications in isolated child processes. |
| `sdist` | Stage a self-contained source distribution at the pinned submodule commits. |
| `lammps` | Integrate `pair_symmetrix` with a specified LAMMPS source and install prefix. |

For accelerator LAMMPS builds, `--require-gpu-aware-mpi` implies `--mpi on` and
requires the selected wrapper to pass the CUDA or ROCm `MPIX` provider query
before configuration. The evidence becomes part of the fingerprinted build
provenance. See {doc}`/user/lammps` for the complete build and runtime
qualification workflow.

The `lammps` command accepts repeatable `--lammps-package NAME` arguments and
repeatable `--lammps-cmake-define NAME=VALUE` arguments. Package selections are
validated against the supplied source tree and installed executable. Both forms
are recorded in the build provenance and therefore select a distinct build
directory. The frontend rejects extra definitions in its managed `PKG_*`,
`Kokkos_*`, and `SYMMETRIX_*` namespaces. `lammps --dry-run` performs preflight
without creating the selected build directory and emits one JSON document with
the resolved invocation and provenance.

The command help is the authoritative option reference:

```bash
python tools/symmetrix_build.py --help
python tools/symmetrix_build.py <command> --help
```

Use a fresh virtual environment and a separate `--build-root` for every
backend and architecture. `wheel-matrix` target specifications include forms
such as `cpu:x86-64-v3`, `cuda:13:sm120`, and `hip:7:gfx1151`. Direct CMake
configuration remains a diagnostic path; the supported installation workflow
is described in {doc}`/user/installation`, with backend-specific requirements
in {doc}`/user/backends`.

Within one fixed backend/toolchain configuration, repeated `install` commands
are incremental and should reuse the same build root. The command installs
`scikit-build-core`, `pybind11`, and `ninja` into the target environment once,
then uses no-build-isolation and a forced reinstall of the selected
distribution. It records the current source content fingerprint and passes
that fingerprint into the native build metadata. Use a fresh process to verify
the installed extension after each rebuild. Source changes require new
fingerprinted JIT artifact keys, not a new package build root or cache
directory.

## Fast native edit loop

After one ordinary `install` has installed the frontend and selected backend
distribution, `develop` bypasses Python package and wheel construction:

```bash
python tools/symmetrix_build.py develop \
  --backend cpu --cpu-target native \
  --build-root "$HOME/tmp/symmetrix-cpu" --generator "Unix Makefiles"
```

Use exactly the same target arguments and build root as the initial install.
The command resolves and preflights the normal target manifest, configures the
fingerprinted CMake directory with current source provenance, builds only
`symmetrix_bindings`, and installs the dedicated runtime component into the
selected interpreter's platform-library directory. That component contains
only the native extension and matching backend descriptor. Accelerator use
requires its architecture-qualified distribution to have been installed once
so its entry point already exists.

Every invocation validates the replacement in a fresh process, rejects legacy
fallback loading or provenance drift, and prints JSON with
`development_only: true`, the target identity, build directory, installed
paths, source provenance, and validation result. `--dry-run` prints the three
direct CMake commands without compiling or installing. Repeating `develop`
without source changes reuses CMake's dependency graph; the build step should
be a no-op apart from configuration, runtime-component installation, and
fresh-process validation.

`develop` is an iteration aid, not release evidence. Final CPU, CUDA, or HIP
qualification must use the ordinary `install` command from a fresh exact build
environment and follow the repository qualification requirements.

`source_dirty` has one policy for the packaged descriptor and native build
report: it covers superproject tracked/untracked changes and recursive
submodule state, including a checked-out submodule at the wrong commit and
local tracked or untracked submodule changes. The frontend passes that single
value to CMake, and backend loading rejects descriptor/native disagreement.

## Prebuilt generated operators

Every `detect`, `preflight`, `install`, `wheel`, and `wheel-matrix` target
manifest records `prebuilt_operator_artifacts` and `prebuilt_oam_artifacts`.
Select the MACE bundle with `--prebuilt-operator-artifacts`:

- `off` installs no generated operator catalog;
- `r0` installs six deduplicated verified R0 artifacts for the exact backend
  target and is the source, development, and publication default;
- `full` installs the 32-artifact verified R1/R0/M0/M1 foundation bundle.

Select the OAM bundle independently with `--prebuilt-oam-artifacts`:

- `off` installs no generated OAM artifacts;
- `known` installs FP32 and FP64 artifacts for the explicit OAM-S/M/L/XL
  architecture descriptors. Runtime lookup compares those descriptors, not a
  filename, package label, or learned-weight hash. Catalog inclusion is a build
  fact and does not qualify the current artifact ABI at runtime.

The mode is part of the build-manifest fingerprint. Do not reuse a build root
across modes. Accelerator artifacts are compiled only for the manifest's exact
`smNN` or `gfxNNNN` target; CPU artifacts use the package CPU target and
OpenMP policy.

```bash
python tools/symmetrix_build.py wheel \
    --backend cpu --cpu-target x86-64-v3 \
    --prebuilt-operator-artifacts r0 \
    --prebuilt-oam-artifacts known \
    --build-root "$HOME/tmp/symmetrix-cpu-r0" --wheel-dir dist
```

Full mode is an opt-in deployment experiment, not the publication default. Its
measured cold-setup benefit comes with 26 additional binaries and mixed
per-case first-call results. No explicit package-size, build-time, and
first-call release budgets have been accepted. The manylinux publication
scripts therefore pin `r0` explicitly; promotion to `full` requires a reviewed
policy change and fresh CPU, CUDA, and HIP qualification evidence. Official CPU
and CUDA publication scripts select the independent `known` OAM bundle; HIP
does not. See
{doc}`/developer/jit_and_artifacts` for the measured comparison and security
model.
