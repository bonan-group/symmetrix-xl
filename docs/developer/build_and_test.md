# Build and Test

Keep CPU, CUDA, HIP, and LAMMPS configurations in separate fresh build
directories and virtual environments. Do not reconfigure an editable
installation to qualify a different backend. The maintained build frontend
records the resolved toolchain and package identity; `AGENTS.md` records the
qualification environment rules. See {doc}`/reference/build_frontend` for its
command surface.

For an isolated OpenMP build:

```bash
mkdir -p "$HOME/tmp"
build_root=$(mktemp -d "$HOME/tmp/symmetrix-openmp.XXXXXX")
uv venv "$build_root/venv"
source "$build_root/venv/bin/activate"
uv pip install scikit-build-core pybind11 ninja
python tools/symmetrix_build.py install \
  --backend cpu --cpu-target native \
  --build-root "$build_root/build" --generator "Unix Makefiles"
symmetrix doctor --json
```

Run focused tests while iterating, then the full applicable suite. Accelerator
tests must use an isolated backend build and matching hardware. LAMMPS requires
a rebuild after changes to the native library.

## Incremental source experiments

Create one task virtual environment with `uv venv`, install the CPU frontend
into it, and add an accelerator backend only when required. Reuse one build
root per fixed Python/backend/toolchain configuration during source iteration.
The maintained installer bootstraps its build backend and Ninja once, then uses
no-build-isolation and a forced reinstall of the selected distribution,
records a content hash of the source inputs in `invocation.json` and
`qualification.json`, and embeds that hash in the native extension. This keeps
incremental compilation fast while making a same-version rebuild observable.

After a relevant edit, stop processes that imported Symmetrix, rerun `install`
against the same build root, and verify the new extension from a fresh process.
The `source_content_sha256` in `_backend_build_info()` must match the current
`symmetrix_source_fingerprint(...)`. JIT artifact keys, manifests, and basenames
include that content hash, so one task/backend cache root can be reused safely
across source edits as described in {doc}`/developer/jit_and_artifacts`.

In a Git worktree, a `.venv` directory can contain only the interpreter. Check
imports rather than directory existence. The default CPU developer loop uses
the worktree-local `.venv` and editable `symmetrix/build`; pass the current
source fingerprint as
`cmake.define.SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256` and force reinstall of
`symmetrix-xl` on every source iteration. Isolated qualification uses the
maintained build frontend and its separate task environment instead. Do not
alternate between these installation routes within one result series.

Direct CMake configuration is a diagnostic path. If it is needed, use a fresh
build directory and pin both `Python_EXECUTABLE` and `PYTHON_EXECUTABLE` during
the first configure so bundled pybind11 cannot select another interpreter.
