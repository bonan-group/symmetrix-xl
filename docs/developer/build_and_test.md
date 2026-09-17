# Build and Test

Keep CPU, CUDA, HIP, and LAMMPS configurations in separate fresh build
directories and virtual environments. Do not reconfigure an editable
installation to qualify a different backend. The maintained build frontend
records the resolved toolchain and package identity; `AGENTS.md` records the
qualification environment rules. See {doc}`/reference/build_frontend` for its
command surface.

For an isolated OpenMP build:

```bash
build_root=$(mktemp -d /tmp/symmetrix-openmp.XXXXXX)
uv venv "$build_root/venv"
source "$build_root/venv/bin/activate"
python tools/symmetrix_build.py install \
  --backend cpu --cpu-target native \
  --build-root "$build_root/build" --generator "Unix Makefiles"
symmetrix doctor --json
```

Run focused tests while iterating, then the full applicable suite. Accelerator
tests must use an isolated backend build and matching hardware. LAMMPS requires
a rebuild after changes to the native library.

Direct CMake configuration is a diagnostic path. If it is needed, use a fresh
build directory and pin both `Python_EXECUTABLE` and `PYTHON_EXECUTABLE` during
the first configure so bundled pybind11 cannot select another interpreter.
