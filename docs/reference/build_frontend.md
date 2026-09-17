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
