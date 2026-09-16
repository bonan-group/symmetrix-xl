# Build and Test

Keep CPU, CUDA, HIP, and LAMMPS configurations in separate fresh build
directories. Do not reconfigure an editable installation to qualify a
different backend. The source-tree package `README.md` documents the supported
build commands; `AGENTS.md` records the qualification environment rules.

For an isolated OpenMP build, pin both CMake Python variables and stage the
result outside the editable environment:

```bash
build_root=$(mktemp -d /tmp/symmetrix-openmp.XXXXXX)
cmake -S symmetrix -B "$build_root/build" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$build_root/stage" \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
  -DPYTHON_EXECUTABLE="$PWD/.venv/bin/python" \
  -DSYMMETRIX_KOKKOS=ON -DSYMMETRIX_DEVICE_BACKEND=NONE \
  -DKokkos_ENABLE_CUDA=OFF -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_SERIAL=OFF -DSYMMETRIX_SPHERICART_CUDA=OFF
cmake --build "$build_root/build" --parallel
cmake --install "$build_root/build"
```

Run focused tests while iterating, then the full applicable suite. Accelerator
tests must use an isolated backend build and matching hardware. LAMMPS requires
a rebuild after changes to the native library.
