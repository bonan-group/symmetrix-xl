# Repository Guidelines

## Layout

- `libsymmetrix/source/`: C++20 core, Kokkos backends, standard M0/R0 modules,
  factorized execution, and runtime JIT support. Generated R1 sources/binaries
  are cache artifacts, not source files.
- `symmetrix/source/symmetrix/`: Python package and ASE calculator;
  `symmetrix/source/cpp/`: pybind11 bindings; `symmetrix/test/`: Python tests.
- `pair_symmetrix/`: LAMMPS pair styles and tests.
- `benchmarks/`: reproducible performance drivers and result notes;
  `docs/`: streamed-edge and JIT documentation.
- Third-party code is kept in Git submodules under `libsymmetrix/external/` and
  `symmetrix/external/`.

## Development and tests

Use Python 3.12 for development, testing, and qualification. Initialize
submodules, then from the repository root:

```bash
test -x .venv/bin/python || uv venv .venv
source .venv/bin/activate
uv pip install scikit-build-core pybind11 ninja
source_fingerprint=$(python -c 'from pathlib import Path; from tools._symmetrix_build.source_provenance import symmetrix_source_fingerprint; print(symmetrix_source_fingerprint(Path.cwd()))')
uv pip install --no-build-isolation --reinstall-package symmetrix-xl -e './symmetrix[test]' \
  --config-setting="cmake.define.SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256=$source_fingerprint"
pytest symmetrix/test
uvx pre-commit run --all-files
```

The published distribution is `symmetrix-xl`; the import namespace remains
`symmetrix`. Do not assume an existing `.venv` is complete: install the package
and verify imports. The shared `symmetrix/build/` is the CPU development build;
do not reconfigure it for another backend. LAMMPS requires
`./pair_symmetrix/install.sh /path/to/lammps`, followed by
`pytest pair_symmetrix/test` in the compatible LAMMPS environment.

Add focused tests beside changed code and parametrize backend/precision cases.
Tests requiring unavailable hardware, toolchains, LAMMPS, or checkpoints should
skip with a clear reason. Run focused tests first, then the applicable full
suite and pre-commit. In a dirty worktree, run pre-commit only on intended
files before using `--all-files`; never restore unrelated formatter changes.

## Builds and qualification

Keep CPU, CUDA, HIP, LAMMPS, and comparison builds in separate fresh virtual
environments and task-specific build roots. Use
`python tools/symmetrix_build.py install`; use `--generator 'Unix Makefiles'`
when Ninja or CUDA `nvcc_wrapper` requires it. Use `uv` for dependencies.
Discard a build root when the compiler, Python, backend, architecture,
SpheriCart policy, incompatible CMake options, or cache validity changes;
never repair a mixed cache in place.

CPU development example:

```bash
root=$(mktemp -d /tmp/symmetrix-cpu.XXXXXX)
uv venv "$root/venv" && source "$root/venv/bin/activate"
python tools/symmetrix_build.py install --backend cpu --cpu-target native \
  --build-root "$root/build" --generator 'Unix Makefiles'
SYMMETRIX_OPENMP_RUNTIME_CHECK=strict symmetrix doctor --json
```

For portable deployment use `--cpu-target x86-64-v3`; do not copy a native
extension to a heterogeneous host. Verify BLAS/OpenMP dependencies with the
doctor output, `ldd`, the build manifest, and wheel audit as appropriate.
OpenMP qualification must use the exact interpreter and module stack, report
one loaded OpenMP runtime, and finish with `status: ok`; do not statically link
libgomp.

CUDA example (replace `sm120` with the target compute capability):

```bash
root=$(mktemp -d /tmp/symmetrix-cuda.XXXXXX)
uv venv "$root/venv" && source "$root/venv/bin/activate"
python tools/symmetrix_build.py install --backend cpu --cpu-target native \
  --build-root "$root/cpu"
python tools/symmetrix_build.py install --backend cuda --arch sm120 \
  --cuda-root /path/to/cuda --build-root "$root/cuda" \
  --generator 'Unix Makefiles'
symmetrix backend show --probe
symmetrix doctor --json
```

Use a fresh backend build for each GPU architecture; NVRTC does not make an
extension built for another architecture valid. HIP likewise requires a fresh
build, `hipcc`, Kokkos Serial, and one explicit AMD target. Prefer the
maintained build frontend over direct CMake; if direct CMake is unavoidable,
pin both `Python_EXECUTABLE` and `PYTHON_EXECUTABLE` on first configure.

After an architecture-qualified build, run `symmetrix version --probe --json`
with the task interpreter. Require `probe_status: "ok"`, source commit/dirty
fields, and non-unknown source and native-content hashes in both the backend
descriptor and `_backend_build_info()`.

Compiled comparison packages may not support newer Python versions; keep their
environment separate and record Python, PyTorch, MACE, cuEquivariance, and CUDA
runtime versions.

## Rebuild and runtime provenance

After edits affecting Python generation, C++, headers, CMake, or packaging:

1. Stop processes that loaded the old extension.
2. Recompute and record the source fingerprint.
3. Rerun the same install command and build root; use a fresh root only when
   the toolchain or cache conditions above require it.
4. In a fresh process, print `sys.executable`, package and extension paths,
   extension SHA-256, selected backend, Kokkos execution space, and
   `_backend_build_info()`. Its content hash must match the fingerprint.
5. Set one task/backend-specific `SYMMETRIX_JIT_CACHE` before importing or
   preparing artifacts. Use `SYMMETRIX_JIT_POLICY=required` for direct-mode
   qualification. Assert the requested algorithm, zero fallback count, and
   matching source hash in the JIT manifest/artifact name.
6. For drivers supporting explicit loading, set `SYMMETRIX_SOURCE_ROOT` and
   `SYMMETRIX_EXTENSION` from the verified build. Warm compilation and module
   loading before timing.

Do not compare a result from a process that survived a rebuild. Record source
fingerprint, extension/model hashes, JIT cache and artifact path, and the exact
benchmark command for every baseline/candidate result. Initialize only one
Kokkos backend per process and release native owners before finalization; a
teardown error may be secondary to an earlier exception.

For CPU timings, apply affinity before importing NumPy/Kokkos/BLAS and set
Kokkos, OpenMP, BLAS, MKL, BLIS, and NumExpr thread counts to one. Record the
physical CPU and exclude its SMT sibling. Exclude compilation, graph creation,
neighbor-list construction, and JIT warmup unless they are the subject of the
benchmark.

## GPU, profiling, LAMMPS, and MPI

Run `nvidia-smi` first and record GPU model, driver, compute capability, memory,
and competing processes. Before diagnosing a comparison failure, probe
`torch.cuda.is_available()` and device count using the exact benchmark
interpreter: a restricted shell can hide the GPU from PyTorch. For a baseline
using another CUDA major version, verify imports/device execution, inspect
dependencies with `ldd`, keep matching runtime/NVRTC libraries inside that
isolated environment, and label the result as cross-major; never prepend those
libraries to the Symmetrix process.

Use Nsight Systems first for timelines and kernel families, then narrowly
filtered Nsight Compute runs. Nsight Compute replays kernels and must not be
used for end-to-end timing. Use built-in timers, `/usr/bin/time -v`, or
`strace -c` when `perf` is unavailable; use `rocprofv3 --kernel-trace --stats`
for AMD after warming hipRTC. Report missing counter permissions instead of
substituting unsupported estimates.

Relink LAMMPS after every native-library change and record its hash. Prepare
matching FP32/FP64 JIT host/device artifacts first. MPI tests need working
OpenMPI/PMIx sockets and shared memory; use one rank per GPU and state the rank
grid and whether MPI is host-staged or CUDA-aware. Positive CUDA-aware evidence
requires the provider capability query and no `Turning off GPU-aware MPI`
warning. Report `us/atom/step` with Pair and Comm separated, and qualify
ownership migration plus multi-step graph reuse, not only `run 0`.

## Coding and benchmark standards

Use four-space Python, Ruff formatting/linting (`E741` is intentionally
ignored), `snake_case` functions/modules, `PascalCase` classes, and
`test_<behavior>` tests. Keep C++20 style and established snake-case APIs.
Comments should explain only non-obvious numerical, ownership, or backend
constraints.

Use `std::size_t` or an explicitly 64-bit type for flattened indices and large
execution ranges; widen operands before multiplication. Follow
`docs/openmp_simd_coding_standard.md`: SIMD writes must be lane-private,
disjoint, or reduced. Use host-launched KokkosBlas for whole-array operations
and KokkosBatched/native SIMD inside workers. External CBLAS from OpenMP
workers is allowed only under the centralized runtime-compatibility policy;
otherwise use Kokkos contractions. Parallel tests must prove worker
participation and nontrivial speedup.

Benchmark records must use `us/atom` (MPI: `us/atom/step`) as the primary
metric and include model, precision, backend, atom count, cutoff, skin,
effective cutoff, directed edges, timing, and memory. After every run, parse
the result record and verify atoms/repeat, cutoff, skin, precision, properties,
and edges against the command; filenames are not evidence that the requested
case ran. Primary CPU optimization results use one physical core; scaling runs
must state physical cores and Kokkos/BLAS thread counts.

OpenMP deployment qualification additionally requires fresh 1/2/4/8-thread
MACEField processes for energy, forces, stress, polarization, BEC, and
polarizability in generic and direct modes, using
`benchmarks/openmp_full_property_qualification.py`. Prove worker participation,
tile-width 1/4/8/16 parity, and nontrivial speedup; import success, `ldd`, or an
energy-only test is insufficient.

## Commits

Use short, imperative, sentence-case subjects and cohesive commits. PRs should
describe behavior/backend impact, list commands run, state CPU/CUDA and
precision coverage, and include benchmark evidence for performance changes.
Do not commit transient agent plans, scratch notes, task-state files, generated
binaries, caches, private models, machine names, or unpublished benchmark
records.
