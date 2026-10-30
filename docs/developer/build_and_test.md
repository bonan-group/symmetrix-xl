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

For the shortest native-only edit loop, perform the ordinary install once,
then rerun the same target with `develop`:

```bash
python tools/symmetrix_build.py develop \
  --backend cpu --cpu-target native \
  --build-root "$build_root/build" --generator "Unix Makefiles"
```

This directly reconfigures CMake with the current source hashes, builds only
`symmetrix_bindings`, replaces only the extension and backend descriptor, and
validates exact provenance from a fresh process. It intentionally does not
install Python sources, dependencies, distribution metadata, or entry points,
so the matching frontend and accelerator distribution (when applicable) must
already be normally installed in the selected interpreter.

After a relevant edit, stop processes that imported Symmetrix, rerun `develop`
against the same build root, and let its fresh-process validation verify the
new extension.
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

The `develop` result is marked `development_only` and must not be used as final
qualification. Before reporting release or performance evidence, build with
the ordinary full `install` workflow in a fresh exact build environment.

Direct CMake configuration is a diagnostic path. If it is needed, use a fresh
build directory and pin both `Python_EXECUTABLE` and `PYTHON_EXECUTABLE` during
the first configure so bundled pybind11 cannot select another interpreter.

## Generated-operator release qualification

Qualify `off`, `r0`, and `full` in separate build roots. For each packaged
mode, audit the wheel, install it into a fresh environment, relocate that
environment, and run `symmetrix version --probe --json` plus
`symmetrix doctor --json`. Confirm the catalog mode, exact target, artifact
count, installed bytes, catalog/source hashes, and one OpenMP runtime. An sdist
must contain generator inputs but no generated source, object, shared library,
cubin, code object, catalog, cache, or private model. Its audit requires every
archive member path to be unique and records both raw and unique counts.

The focused Python qualification covers canonical one-field identity
mismatches, manifest normalization and alias deduplication, package relocation,
catalog/binary tampering and symlinks, normal installed/cache/runtime precedence
plus the explicitly frozen builtin exception,
unavailable compilers, policy behavior, and real host/CUDA/HIP compiler
fixtures. Runtime qualification still requires a fresh exact CPU build and an
exact accelerator build on matching hardware. Compiler/package fixtures do not
substitute for CUDA or HIP execution. Publication scripts pin `r0`; see
{doc}`/developer/jit_and_artifacts` for the recorded full-mode tradeoff and the
currently unmet frozen-R0 retirement gates.

## Dispersion reference and backend qualification

Native D2/D3 is part of the base package. `torch-dftd==0.5.3` is an offline
reference input only: use an isolated environment and the provenance-checking
generators under `tools/` when intentionally regenerating parameter tables or
fixtures. Ordinary builds and installed ASE calculations must not import
PyTorch or torch-dftd. The `dispersion` extra is an empty, deprecated
compatibility spelling.

Run the CPU correctness, finite-difference, integration, metadata, and device
contract tests with:

```bash
pytest symmetrix/test/test_dispersion_parameters.py \
  symmetrix/test/test_dispersion_graph.py \
  symmetrix/test/test_d2_kokkos.py \
  symmetrix/test/test_d3_kokkos.py \
  symmetrix/test/test_corrections.py \
  symmetrix/test/test_dispersion_device.py \
  symmetrix/test/test_dispersion_documentation.py
```

CUDA and HIP runtime cases skip when the matching extension or device is not
available. A release or deployment qualification must run the backend's
required-test manifest in a fresh architecture-qualified environment and
verify device graph construction, graph generation reuse inside half the skin,
zero backend fallbacks, and float32/float64 parity. Compile-only evidence
does not establish runtime correctness or performance.

The large-cell driver is `benchmarks/dispersion_kokkos_timing.py`. Record
`us/atom`, method, precision, backend, atom count, pair and coordination
cutoffs, skin/effective graph cutoff, directed edges, graph-build and reuse
time, peak memory, and provenance hashes. Exclude compilation, first graph
construction, and warmup from steady-state timing. For CPU scaling, use
`--cpu <first-logical-cpu> --cpu-count <physical-cores>`; the driver selects
that many distinct package/core topology keys from its inherited affinity mask
before importing NumPy or native modules and sets Kokkos, OpenMP, OpenBLAS,
MKL, BLIS, and NumExpr thread counts to the same value. Schema version 3 records
the requested count, selected affinity, and each thread setting, and rejects a
mismatch when it reparses the output. Use `OMP_PROC_BIND=close` and
`OMP_PLACES=cores` so the OpenMP runtime keeps workers on the selected cores.

Current D3 diagnostics report grow-only workspace capacity. The reference
weight and derivative capacity is `2 * atom_capacity * 5 * sizeof(Precision)`;
the cached coordination switch and radial-derivative capacity is
`2 * edge_capacity * sizeof(Precision)`. Treat the corresponding byte fields
as capacity allocation, not active bytes, and record both active and capacity
counts when investigating graph-size changes.
