# Direct CPU single-thread timing

Date: 2026-08-20

## Result

Canonical `streamed_edges="direct"` takes **216.869 us/atom**, or
**187.375 ms/call**, on the local AMD Ryzen 9 9950X3D2. This is the median of
three fresh-process medians. Each process used three warmups and five measured
energy, force, and stress evaluations.

| Fresh process | Median us/atom | Median ms/call |
|---|---:|---:|
| 1 | 216.347 | 186.924 |
| 2 | 216.869 | 187.375 |
| 3 | 218.306 | 188.617 |

The standard deviation of the three process medians is 1.014 us/atom and their
coefficient of variation is 0.467%. Across all 15 calls, the median is 216.966
us/atom and the range is 215.547 to 220.068 us/atom.

## Workload

- Source commit: `9547dfa3e2b6f0e9918922fa62b0f625498cb03e`
- Model source checkpoint:
  `/path/to/mace-mh-0.model`
- Source checkpoint SHA-256:
  `d62ff8f293664e6556cfa49364b28ee72a70cf1e6150f3c111a578397fed609d`
- Current Al/N compact export: `/tmp/mace-mh-0-current-Al-N.json`
- Export SHA-256:
  `03ba3e74170c2a51ba93dd0309728c9dbd094ce67d01cf8338993784b7eee574`
- Structure: 6x6x6 periodic wurtzite AlN, 864 atoms
- Model cutoff: 6.0 A
- Neighbor-list skin: 0.0 A
- Effective neighbor-list cutoff: 6.0 A
- Directed edges: 78,624
- Backend and precision: Kokkos OpenMP FP32
- CPU affinity: logical CPU 0 only; its SMT sibling CPU 16 was excluded
- Thread counts: one Kokkos/OpenMP thread and one BLAS thread
- CPU: AMD Ryzen 9 9950X3D2, 16 physical cores, 32 logical CPUs
- Compiler: GCC 15.2.0, Release, `-march=native`
- BLAS: `/usr/lib/x86_64-linux-gnu/libopenblas.so`

The current-HEAD extension was built in an isolated `/tmp` directory. Its
SHA-256 is
`a602585f56ba0ae0e2e9ad2411ad4de006d04704df75f24483ba480a20845eb0`.
The host RTC artifact was built before timing with `-O3 -ffast-math
-march=native`; its ID is `jit-r1-gen2-f32-29f6298ae84d2005` and its SHA-256 is
`1064ddeebe985a6a45b9eead45382dfd7bdd8ad9944d64ed3071e6f27df1b61b`.

## Validation

All runs report requested mode, canonical mode, and execution algorithm as
`direct`. The RTC status is `cached`, R1 forward selects `jit_all`, R1 reverse
selects `jit`, and the measured calls report zero factorized fallback
evaluations. Every run has the same graph hash and directed-edge count. Energy,
forces, and stress are finite in every run; total energy is identical across
fresh processes at `-646502.3501055818` eV.

The prior full-evaluator result in `receiver_factorized_rtc_mh0_cpu.md` was
230.820 us/atom. The new value is 6.04% lower, but this is not a controlled
code-only speedup: that result used an AMD Ryzen AI MAX+ 395 and model export
SHA-256 `c202e5bc...eb22`. The historical temporary JSON is no longer present,
and regenerating from the local checkpoint with the current extractor produces
the export recorded above. Use 216.869 us/atom as the current-machine baseline.

## Raw samples

Times are milliseconds per 864-atom call:

```text
process 1: 186.232275, 186.647467, 187.007757, 186.924212, 187.458977
process 2: 187.021674, 187.552611, 188.009010, 187.375211, 187.149231
process 3: 190.138813, 188.616761, 188.222107, 188.395530, 188.893887
```

Raw JSON records:

```text
/tmp/direct-cpu-9547dfa-run1.json
/tmp/direct-cpu-9547dfa-run2.json
/tmp/direct-cpu-9547dfa-run3.json
```

## Reproduction

The isolated extension was configured with:

```bash
cmake -S symmetrix -B /tmp/symmetrix-direct-cpu-9547dfa-make-v2 \
  -G 'Unix Makefiles' -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/tmp/symmetrix-direct-cpu-9547dfa-stage-v2 \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
  -DPYTHON_EXECUTABLE="$PWD/.venv/bin/python" \
  -DSYMMETRIX_KOKKOS=ON -DSYMMETRIX_DEVICE_BACKEND=NONE \
  -DKokkos_ENABLE_CUDA=OFF -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_SERIAL=OFF -DKokkosKernels_ENABLE_TPL_BLAS=ON \
  -DBLA_VENDOR=OpenBLAS -DSYMMETRIX_SPHERICART_CUDA=OFF \
  -DSPHERICART_ENABLE_CUDA=OFF
cmake --build /tmp/symmetrix-direct-cpu-9547dfa-make-v2 --parallel 8
```

The benchmark was launched three times in fresh processes. The `runpy` wrapper
removes the existing editable-install finder so the isolated current-HEAD
extension is loaded without modifying `.venv`:

```bash
env SYMMETRIX_JIT_CACHE=/tmp/symmetrix-direct-cpu-9547dfa-jit-cache \
  SYMMETRIX_JIT_POLICY=required SYMMETRIX_BENCHMARK_THREADS=1 \
  SYMMETRIX_BENCHMARK_BLAS_THREADS=1 KOKKOS_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  OMP_PROC_BIND=true OMP_PLACES=cores \
  taskset -c 0 .venv/bin/python -c \
  "import runpy,sys; sys.meta_path[:]=[f for f in sys.meta_path if type(f).__module__!='_editable_skbc_symmetrix']; sys.path.insert(0,'/tmp/symmetrix-direct-cpu-9547dfa-stage-v2'); sys.argv=['benchmarks/standard_mace_streamed_benchmark.py',*sys.argv[1:]]; runpy.run_path('benchmarks/standard_mace_streamed_benchmark.py',run_name='__main__')" \
  /tmp/mace-mh-0-current-Al-N.json --backend kokkos --dtype float32 \
  --modes direct --sizes 6 --neighbor-skin 0 --warmups 3 --repeats 5 \
  --source-commit 9547dfa --output /tmp/direct-cpu-9547dfa-run1.json
```
