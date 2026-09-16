# Repository Guidelines

## Project Structure & Module Organization

- `libsymmetrix/source/` contains the C++20 core, Kokkos implementations, standard M0/R0 modules, factorized execution, and runtime JIT support. JIT-generated R1 sources and binaries are runtime cache artifacts, not checked-in source modules.
- `symmetrix/source/symmetrix/` contains the Python package and ASE calculator; `symmetrix/source/cpp/` provides pybind11 bindings. Python tests live in `symmetrix/test/`.
- `pair_symmetrix/` provides the LAMMPS pair styles and its separate pytest suite in `pair_symmetrix/test/`.
- `benchmarks/` holds reproducible performance drivers and result notes. `docs/` documents streamed-edge execution and JIT deployment. Third-party dependencies are Git submodules under `libsymmetrix/external/` and `symmetrix/external/`.

## Build, Test, and Development Commands

Clone with submodules (`git clone --recursive ...`) or run `git submodule update --init --recursive` first. From the repository root:

```bash
uv venv
source .venv/bin/activate
uv pip install -e "./symmetrix[test]"
pytest symmetrix/test
uvx pre-commit run --all-files
```

The editable install invokes scikit-build-core/CMake and builds the C++ extension. Treat its shared `symmetrix/build/` directory as the default CPU development build only; do not reconfigure it to qualify another backend. The published frontend distribution is named `symmetrix-xl`, while the Python import namespace and source directory remain `symmetrix`. CUDA and explicit CPU build options are documented in `symmetrix/README.md`. LAMMPS integration requires installing `pair_symmetrix` into a compatible LAMMPS checkout with `./pair_symmetrix/install.sh /path/to/lammps`, then running `pytest pair_symmetrix/test` in that built environment.

## Compilation Environments

Keep CPU, CUDA, HIP, and LAMMPS builds in separate fresh directories and
virtual environments. Do not reconfigure the developer's editable installation
to qualify another backend. Use the maintained build frontend with a
task-specific `--build-root`; it records the resolved toolchain and package
identity and installs into the active Python environment. If Ninja is
unavailable, or for CUDA through Kokkos `nvcc_wrapper`, use
`--generator "Unix Makefiles"`.

```bash
build_root=$(mktemp -d /tmp/symmetrix-openmp.XXXXXX)
uv venv "$build_root/venv"
source "$build_root/venv/bin/activate"
python tools/symmetrix_build.py install \
  --backend cpu --cpu-target native \
  --build-root "$build_root/build" --generator "Unix Makefiles"
python -c 'import pathlib, symmetrix; print(pathlib.Path(symmetrix.__file__).resolve())'
SYMMETRIX_OPENMP_RUNTIME_CHECK=strict symmetrix doctor --json
```

Check the recorded backend summary and use `ldd` on the extension path reported
by `symmetrix doctor --json` to verify its dynamic OpenBLAS and OpenMP
dependencies. Verify static BLAS selection from the build manifest and wheel
audit rather than `ldd`, which cannot report statically linked libraries. For
OpenMP deployments, run `doctor` with the exact Python executable and module
stack used for evaluation. It must report the expected compiler/runtime hash,
one actual loaded OpenMP runtime, and `status: ok`; `ldd` does not model the
main executable's `DT_RPATH`. Prefer standalone CPython when Anaconda's legacy
RPATH would substitute libgomp; use a matching compiler-runtime `LD_PRELOAD`
only as a diagnosed workaround, and do not statically link libgomp.

If configure or an
incremental rebuild loses an imported BLAS target, or if the compiler, Python,
backend, architecture, or SpheriCart policy changes, discard that task build
and configure a new one. Do not repair a mixed cache in place. CPU builds
default to `--cpu-target native` with Kokkos native architecture detection. Do
not copy that machine-local extension to a heterogeneous node. Use
`--cpu-target x86-64-v3` for a portable x86-64-v3 deployment build; the build
frontend disables `Kokkos_ARCH_NATIVE` and applies the baseline to the bundled
Kokkos, KokkosKernels, SpheriCart, core, and Python bindings.

For CUDA builds, use a fresh environment and separate build roots for the base
frontend and architecture-qualified accelerator package. Select the exact
device architecture matching the qualification GPU:

```bash
cuda_root=/path/to/cuda
build_root=$(mktemp -d /tmp/symmetrix-cuda.XXXXXX)
uv venv "$build_root/venv"
source "$build_root/venv/bin/activate"
python tools/symmetrix_build.py install \
  --backend cpu --cpu-target native --build-root "$build_root/cpu"
python tools/symmetrix_build.py install \
  --backend cuda --arch sm120 --cuda-root "$cuda_root" \
  --build-root "$build_root/cuda" --generator "Unix Makefiles"
symmetrix backend show --probe
symmetrix doctor --json
```

Replace `sm120` with the compute capability of the qualification target. The
build frontend selects the pinned Kokkos wrapper, host compiler, Kokkos
architecture trait, SpheriCart CUDA policy, backend package identity, and
runtime paths. Direct CMake configuration is a low-level diagnostic path, not
the supported package-install procedure. If it is required, pin both
`Python_EXECUTABLE` and `PYTHON_EXECUTABLE` on the first configure so bundled
pybind11 cannot select a different interpreter.

Do not reuse an extension built for another GPU architecture merely because
NVRTC recompiles the generated direct kernel. Kokkos and SpheriCart device code
is part of the ahead-of-time extension. HIP likewise requires its own fresh
build, `hipcc`, Kokkos Serial, and one explicit AMD architecture/offload target;
follow the qualified recipe in `symmetrix/README.md`.

Use `uv` when dependency installation is required. Keep backend-specific test
environments isolated and do not replace the developer's installed extension.

## Runtime Selection

Before every qualification, print and record the imported Python package,
native extension, Kokkos execution space, and binary hash. An editable
scikit-build finder can override `PYTHONPATH`; setting `PYTHONPATH` alone does
not prove that the fresh extension was loaded. Prefer a fresh virtual
environment. Maintained benchmark drivers that support explicit loading accept
both variables below and remove the editable finder before importing:

```bash
export SYMMETRIX_SOURCE_ROOT="$PWD"
export SYMMETRIX_EXTENSION=/absolute/path/to/the-built-extension.so
```

Use one task-specific RTC cache per backend and source state, for example
`SYMMETRIX_JIT_CACHE=/tmp/symmetrix-TASK-jit-cache`. Use
`SYMMETRIX_JIT_POLICY=required` when generated direct execution is part of the
qualification, assert the selected algorithm/artifact and zero fallback count,
and warm compilation and module loading before measurement. Host artifacts are
model-, precision-, ABI-, generation-, and target-specific. Prepare them with
`symmetrix_prepare_jit_host_artifact`; use `--host-target portable` only when an
x86-64-v3 deployment artifact is intended.

Initialize only one Kokkos backend per process. Release evaluators and other
native owners before explicit Kokkos finalization. A teardown error after an
earlier exception is often secondary; diagnose the first exception before
changing lifecycle code.

For primary CPU timings, apply process affinity before Python imports NumPy,
Kokkos, or BLAS, and set `KOKKOS_NUM_THREADS`, `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `BLIS_NUM_THREADS`, and
`NUMEXPR_NUM_THREADS` to one. Record the selected physical CPU and exclude its
SMT sibling. Compilation, graph construction, neighbor-list creation, and JIT
warmup are outside the steady-state region unless the benchmark explicitly
targets them.

## Profiling Environments

Run `nvidia-smi` first and record the GPU model, driver, compute capability,
memory use, and competing processes. Do not stop unrelated GPU services unless
the user explicitly authorizes it. CUDA profiling requires access to the target
device and driver. Store reports under a task-specific temporary or benchmark
artifact directory and retain the exact command, model/binary hashes, warmup
count, cutoff, skin, effective cutoff, atom count, and directed-edge count.

Use Nsight Systems first to locate expensive kernel families, synchronization,
and transfers:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --force-overwrite=true -o /tmp/symmetrix-profile/system \
  /path/to/warmed-profile-command
nsys stats /tmp/symmetrix-profile/system.nsys-rep
```

Use Nsight Compute only for the kernel families identified by the timeline.
Limit kernel names and launch counts before requesting expensive sections:

```bash
ncu \
  --kernel-name 'regex:.*target_kernel.*' --launch-skip 10 \
  --launch-count 2 --section SpeedOfLight \
  --section MemoryWorkloadAnalysis -o /tmp/symmetrix-profile/kernel \
  /path/to/profile-command
```

Nsight Compute replays kernels and materially perturbs end-to-end timing; do
not use its elapsed process time as a performance result. Add Occupancy,
LaunchStatistics, SchedulerStatistics, or WarpStateStatistics only to answer a
specific hypothesis. Collect a baseline and candidate with identical inputs
and launch filters. If hardware counters are permission-restricted, report the
missing capability instead of silently substituting timeline estimates.

When system policy prevents `perf record`, use built-in phase timers,
`/usr/bin/time -v`, and `strace -c` for CPU attribution. For AMD GPU work,
prefer `rocprofv3 --kernel-trace --stats` and selected ROCTx regions after
warming hipRTC.

## LAMMPS and MPI Environments

Relink the LAMMPS executable after every native library change and record its
hash. LAMMPS does not run the direct-kernel compiler: prepare matching FP32 and
FP64 host/device artifacts in the Symmetrix environment first. The MPI pytest
matrix uses `SYMMETRIX_LAMMPS_EXECUTABLE`,
`SYMMETRIX_LAMMPS_COMPACT_MODEL`, `SYMMETRIX_LAMMPS_FIELD_MODEL`, and the
precision-specific `SYMMETRIX_LAMMPS_JIT_HOST_ARTIFACT_FLOAT32` and
`SYMMETRIX_LAMMPS_JIT_HOST_ARTIFACT_FLOAT64` variables.

MPI execution requires an environment in which OpenMPI/PMIx and the selected
transport can create their sockets and shared-memory resources. For multi-GPU
tests use one MPI rank per GPU, state the rank grid, and distinguish host-staged
from CUDA-aware MPI. Positive CUDA-aware evidence requires the provider
capability query and absence of LAMMPS's `Turning off GPU-aware MPI` warning;
zero Symmetrix staged bytes alone is not proof because no callback may have run.

Report MPI performance in `us/atom/step`, with Pair and Comm timing separated.
Staging synchronization can be charged to Pair. Qualify ownership migration
and multi-step prepared-graph reuse, not only `run 0` or successful process
exit.

## Coding Style & Naming Conventions

Python uses four-space indentation, Ruff formatting, and Ruff linting; `E741` is intentionally ignored. Use `snake_case` for functions/modules, `PascalCase` for classes, and `test_<behavior>` for tests. For C++, retain the surrounding style, C++20 compatibility, and established `snake_case` APIs. Keep comments focused on non-obvious numerical, ownership, or backend constraints.

Use `std::size_t` or an explicitly 64-bit type for flattened indices and execution ranges whose extents can scale with atom, edge, spline, or channel counts. Cast operands before multiplication so the product cannot overflow in 32-bit arithmetic.

For OpenMP and SIMD kernels, follow `docs/openmp_simd_coding_standard.md`.
Every value written by an `omp simd` loop must be lane-private, disjointly
indexed, or covered by an explicit reduction. Use host-launched `KokkosBlas`
for whole rank-1/rank-2 operations and `KokkosBatched::Team*`, `Serial*`, or
native SIMD kernels inside Kokkos workers. External CBLAS calls from Kokkos
OpenMP workers are permitted only through the centralized host-worker policy
when runtime probing verifies OpenMP OpenBLAS, confirms that its query symbols
belong to the CBLAS provider, proves that OpenBLAS and Kokkos use the same
OpenMP runtime, and finds nested active levels disabled. Pthread, unidentified,
runtime-incompatible, or nested-enabled BLAS configurations must use Kokkos
contractions inside workers. One BLAS thread alone does not establish
concurrent-caller safety; the unsafe override is for controlled diagnostics
only and must fail strict runtime qualification.
Parallel correctness tests must prove actual worker participation and
nontrivial speedup, not only requested thread counts.

## Testing Guidelines

Add focused pytest coverage beside the affected component. Parametrize backend/precision variants when behavior differs. Test coverage must not be limited to the currently installed extension. When compatible CUDA or HIP hardware and toolchains are available, create isolated temporary virtual environments and fresh backend-specific build directories, build the corresponding Kokkos backend, and run its applicable tests. Do not replace or reconfigure the developer's installed extension to obtain this coverage. Tests requiring unavailable accelerator hardware, toolchains, compilers, LAMMPS, or downloaded model checkpoints should skip clearly with the missing prerequisite in the reason. Run the smallest relevant test file while iterating, then the full applicable suite and pre-commit before submission.

If the worktree has no usable test environment, create the repository-local
`.venv` with `uv venv`, activate it, and install the editable test package with
`uv pip install -e "./symmetrix[test]"`. Do not treat a missing `pytest`
executable as a reason to skip tests when this CPU development environment can
be built.

Inspect `git status` before running formatters. In a dirty worktree, run
pre-commit on the exact intended files first; `--all-files` can rewrite
unrelated tracked files. Never restore a formatter change unless the file was
known clean immediately before that invocation.

Benchmark records and result notes must use `us/atom` as the primary performance metric. Total step or call time may also be reported as secondary context, but performance targets, comparisons, and optimization decisions must be stated in `us/atom`. Include the atom count and the exact model cutoff used, in addition to the existing model, backend, precision, timing, and memory context. When a neighbor-list skin or other graph expansion is active, also report that value, the resulting effective neighbor-list cutoff, and the directed-edge count so capacity and performance results can be compared on the same workload.

CPU optimization benchmarks and stage profiles default to one physical CPU
core, one Kokkos/OpenMP thread, and one BLAS thread, with process affinity
applied before Python initializes either runtime. Use this one-thread result for
primary optimization decisions and before/after acceptance. Multithread runs
are secondary scaling qualifications and must state their physical-core set and
both Kokkos and BLAS thread counts explicitly.

OpenMP deployment qualification must additionally compare fresh one-, two-,
four-, and eight-thread MACEField processes for energy, forces, stress,
polarization, BEC, and polarizability in both generic and direct modes. Use
`benchmarks/openmp_full_property_qualification.py` with explicit model,
structure, physical CPU sets, and output record. The gate must prove actual
worker participation, tile-width 1/4/8/16 parity, and nontrivial multithread
speedup. A successful import, static `ldd`, or energy-only calculation is not a
substitute for this gate.

## Commit & Pull Request Guidelines

Follow the history’s short, imperative, sentence-case subjects, for example `Harden indexing for million-atom models`. Keep commits cohesive. Pull requests should explain the behavior and backend impact, list commands run, link relevant issues, and include benchmark evidence for performance changes. Note CPU/CUDA and precision coverage explicitly; attach screenshots only for documentation or user-visible output changes.
