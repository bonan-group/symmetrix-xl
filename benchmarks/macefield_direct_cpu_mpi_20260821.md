# MACEField direct CPU timing and MPI qualification

Date: 2026-08-21

## Outcome

MACEField energy, forces, stress, and polarization are now within 10.3% of
matched standard-MACE energy, forces, and stress on the public low-memory path.
The native evaluator gap is 11.8%. This is the same performance class as
retained-state execution, where MACEField was 10.9% slower.

The low-memory-specific regression came from excluding field-coupled models
from the host FP32 standard-M1 module. Removing only that exclusion makes the
structurally matching MACEField model use standard M1 in both directions and
removes 36,352 bytes/team of generic M1 scratch.

| Scope and policy | Standard MACE (us/atom) | MACEField (us/atom) | Field / standard |
|---|---:|---:|---:|
| Native, retained | 217.002 | 240.704 | 1.109x |
| Public results, retained | 238.587 | 264.367 | 1.109x |
| Native, low memory, before | 190.822 | 261.364 | 1.370x |
| Native, low memory, after | 189.922 | 212.364 | 1.118x |
| Public results, low memory, before | 210.284 | 286.551 | 1.363x |
| Public results, low memory, after | 211.276 | 232.989 | 1.103x |

The optimized MACEField low-memory path is 1.231x faster in the native scope
and 1.230x faster through public results. Matched standard-MACE medians moved by
-0.47% and +0.47%, respectively.

## CPU workload

- Source baseline: `9547dfa3e2b6f0e9918922fa62b0f625498cb03e` plus the scoped changes in this record
- Structure: 6x6x6 periodic wurtzite AlN, 864 atoms
- Model cutoff: 6.0 A
- Neighbor-list skin: 0.0 A
- Effective neighbor-list cutoff: 6.0 A
- Directed edges: 78,624
- Backend and precision: Kokkos OpenMP FP32
- Execution: canonical `streamed_edges="direct"`, cached host RTC, R1 `jit_all` forward and `jit` reverse
- Low memory: M1 recomputation, MH-0 `reuse-adjoints-v1`, compact `unit-f32-radius-f64-v1` geometry
- CPU: AMD Ryzen 9 9950X3D2
- Affinity: logical CPU 0 only; its SMT sibling CPU 16 was excluded
- Threads: one Kokkos/OpenMP thread and one BLAS thread
- Compiler and BLAS: GCC 15.2.0, Release, `-march=native`, OpenBLAS
- Sampling: three fresh processes, three warmups, five measured calls per process
- Electric field: `[0.01, -0.02, 0.03]`

The standard and field artifacts have matched 128-channel, `l_max=3`,
94-term M1 structure and current M0/R0/R1 contracts. They are different trained
models, so this is an implementation-cost comparison rather than a comparison
of physical predictions.

| Artifact | SHA-256 |
|---|---|
| Standard MACE compact JSON | `03ba3e74170c2a51ba93dd0309728c9dbd094ce67d01cf8338993784b7eee574` |
| MACEField compact JSON | `52dceeca5ed876bc12e82835574cbc83b85a5055f834560cbb3acd426be31fdf` |
| Timed optimized OpenMP extension | `d2db67292650d2752fd017a958f050620532bff749613d01b8854f2759274743` |
| Final admission-qualified OpenMP extension | `ef8b5af1766f7b1860f6847438efbf5ba68efcbad8e2c1212b203097754162d0` |
| FP32 host RTC plugin | `1064ddeebe985a6a45b9eead45382dfd7bdd8ad9944d64ed3071e6f27df1b61b` |

The final rebuild adds only the ordinary standard-M1 zero-scratch admission
correction. The timed MACEField and explicitly enabled standard-MACE low-memory
paths select the same standard-M1 kernels in both binaries, so this policy-only
correction does not invalidate the recorded steady-state samples.

## CPU samples

Each entry is a fresh-process median in us/atom.

| Scope | Model | Process 1 | Process 2 | Process 3 | Median |
|---|---|---:|---:|---:|---:|
| Native low memory | Standard MACE | 189.412 | 189.922 | 190.081 | 189.922 |
| Native low memory | MACEField | 212.364 | 211.140 | 213.664 | 212.364 |
| Public low memory | Standard MACE | 211.674 | 210.499 | 211.276 | 211.276 |
| Public low memory | MACEField | 231.677 | 233.834 | 232.989 | 232.989 |

Public MACEField calls request energy, forces, six-component ASE stress, and
three-component polarization. Public standard-MACE calls request energy,
forces, and stress. Analytical BEC and polarizability response code is not in
the timed region.

Every optimized MACEField record reports standard-M1 readiness, eight forward
and eight reverse launches after warmup plus measurement, zero generic M1
scratch, one prepared graph, one schedule build, and zero fallback evaluations.
Ordinary standard M1 also bypasses the generic team-scratch admission check;
MACEField retains the separate analytical-response scratch check because its
field tangent kernels still use four generic polynomial workspaces.
The output changes relative to the generic M1 ordering are within FP32
expectations: total energy changes by `2.00e-7 eV`, the largest polarization
component change is `1.40e-10`, and the reported maximum stress magnitude
changes by `4.31e-9 eV/A^3`.

The raw timing JSON files are ephemeral `/tmp` artifacts rather than durable
repository evidence. Their SHA-256 hashes identify the exact inputs used to
derive the tables above:

| Record | SHA-256 |
|---|---|
| Retained native | `b4c4d6ec2ed7128980af54cb46f1732abc52106a494df70f9e38990b6d749343` |
| Retained public | `2a1eb7b681285696f4963de258f474b5db5db251dae9cdf8fe28b4668f346bb7` |
| Low-memory native, before | `869e74342fcd9c2520317979c88f6d63f5f87ce654676aefdce8a1c185434209` |
| Low-memory public, before | `04f845c9bc0bad04bd60f78a907777ec73e7c027727322634f75654c9705a7fe` |
| Low-memory native, optimized | `ce76b7af26a0302654696cb754da175a8f124953ed3fcda4799036106ff31b84` |
| Low-memory public, optimized | `37d2788c39b415d433bacf387556dc117da886e8e950ab7a610a86a832d5e7e7` |

## LAMMPS MPI

The current-head OpenMP/MPI LAMMPS executable was relinked after the standard-M1
change and the standard-M1 scratch-admission correction. Its SHA-256 is
`19d689c26c6c1c98d66afa820ab995202fb78cd35808025aa166b59e14bac60c`.
The FP64 host RTC plugin SHA-256 is
`43145332242618e45d5df52620f5b6798c82c1e87c5176a00dc2bbdcb0486e2c`.

The full MPI file passes 21/21 outside the filesystem sandbox. The matrix uses
a 64-atom perturbed wurtzite AlN cell, 6.0-A model cutoff, 1.0-A skin, 7.0-A
effective list cutoff, and 9,152 directed neighbor-list candidates. It checks
the initial state and a state where atom 3 crosses an ownership boundary.

Qualified coverage includes:

- Standard MACE direct retained and low-memory FP32/FP64, two ranks, orthogonal and triclinic cells
- Standard MACE generic and compatibility-alias parity in FP32/FP64, one/two/four ranks, plus forced legacy callback parity
- MACEField generic FP32/FP64, one and two ranks, orthogonal and triclinic cells
- MACEField direct retained and low memory in FP32/FP64, one and two ranks, orthogonal and triclinic cells
- MACEField direct low-memory FP32 with four ranks in a 1x2x2 decomposition

Maximum observed MACEField differences over initial and migrated states are:

| Comparison | Energy (eV) | Pressure component (bar) | Force component (eV/A) | Per-atom energy (eV) |
|---|---:|---:|---:|---:|
| Generic vs direct retained | 2.81e-5 | 4.59e-1 | 4.29e-5 | 1.35e-5 |
| Direct retained vs direct low memory | 1.14e-13 | 5.99e-5 | 1.75e-9 | 1.78e-15 |
| Two-rank low memory vs single rank | 4.87e-5 | 5.69e-2 | 6.25e-6 | 1.35e-5 |
| Four-rank low memory vs single rank | 1.99e-5 | 4.61e-2 | 6.81e-6 | 1.16e-5 |

These rank-dependent differences are consistent with FP32 reduction order.
FP64 cases use `rtol=1e-11`, atom `atol=1e-10`, and global `atol=1e-8` or
tighter. FP32 cases use `rtol=2e-4`, atom `atol=1e-4`, and global
`atol=1e-1`.

The MPI pair style returns energy, forces, and global virial/pressure. It
accepts the graph-level electric field but does not expose polarization, BEC,
polarizability, per-atom fields, time-dependent fields, or atomic virials.
CUDA MACEField MPI, FP64 CUDA MPI, HIP MPI, and cross-node MACEField execution
remain qualification targets.

## Verification

```text
24 passed  symmetrix/test/test_standard_modules.py and benchmark harness tests
8 passed   ordinary-MACE M1 recompute, automatic admission, and low-memory bundle tests
6 passed   focused FP32/FP64 MACEField low-memory primal, response, stress, and calculator tests
2 passed   FP32/FP64 MACEField analytical-response scratch-limit tests
21 passed  pair_symmetrix/test/test_pair_symmetrix_factorized_mpi.py
```

The focused native tests compare low memory with retained execution, compare
stress with finite-difference cell derivatives, verify public energy/forces/
stress/polarization, validate graph-token lifecycle behavior, and assert that
host FP32 executes the standard-M1 forward and reverse modules with zero generic
M1 scratch.

The broader native MACEField file still has a pre-existing analytical-response
crash in
`test_kokkos_field_standard_m0_m1_lifecycle[MACEKokkosFloat-0.0005-all_interactions]`
while computing the electric-field force derivative. The same crash reproduces
with the preserved pre-change extension, so it is not caused by admitting
MACEField to standard M1. It remains outside this energy/forces/stress/
polarization and direct-MPI qualification.

## Reproduction

The source checkpoints are pinned by SHA-256 before extraction:

| Checkpoint and head | SHA-256 |
|---|---|
| `mace-mh-0.model`, `omat_pbe` | `d62ff8f293664e6556cfa49364b28ee72a70cf1e6150f3c111a578397fed609d` |
| `MACEField-MH-0-omat-dielectric.model`, `mp-dielectric` | `f92e043aaf2cd8879919db8452503553fe7b608cb749d8d169dd96d4aa094aa2` |

Extract the Al/N compact models with the current source environment:

```bash
.venv/bin/symmetrix_extract_mace \
  --model /path/to/mace-mh-0.model \
  --head omat_pbe --atomic-numbers 13 7 \
  --output /tmp/mace-mh-0-current-Al-N.json
.venv/bin/symmetrix_extract_mace \
  --model /path/to/MACEField-MH-0-omat-dielectric.model \
  --head mp-dielectric --atomic-numbers 13 7 \
  --output /tmp/macefield-response-current-contract.json
```

The final extension was built as Release, OpenMP-only, native-architecture, and
OpenBLAS, then installed into an isolated staging root:

```bash
cmake -S symmetrix -B /tmp/symmetrix-macefield-m1-9547dfa-build \
  -G 'Unix Makefiles' -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/tmp/symmetrix-macefield-m1-9547dfa-stage \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
  -DPYTHON_EXECUTABLE="$PWD/.venv/bin/python" \
  -DSYMMETRIX_KOKKOS=ON -DSYMMETRIX_DEVICE_BACKEND=NONE \
  -DKokkos_ENABLE_CUDA=OFF -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_SERIAL=OFF -DKokkosKernels_ENABLE_TPL_BLAS=ON \
  -DBLA_VENDOR=OpenBLAS -DSYMMETRIX_SPHERICART_CUDA=OFF \
  -DSPHERICART_ENABLE_CUDA=OFF
cmake --build /tmp/symmetrix-macefield-m1-9547dfa-build --parallel 8
cmake --install /tmp/symmetrix-macefield-m1-9547dfa-build
```

Run each scope three times in fresh workers through the maintained supervisor.
For the optimized low-memory native record:

```bash
env SYMMETRIX_SOURCE_ROOT="$PWD" \
  SYMMETRIX_EXTENSION=/tmp/symmetrix-macefield-m1-9547dfa-stage/symmetrix/symmetrix.cpython-312-x86_64-linux-gnu.so \
  SYMMETRIX_JIT_CACHE=/tmp/symmetrix-macefield-final-jit-cache \
  SYMMETRIX_JIT_POLICY=required KOKKOS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 OMP_PROC_BIND=true OMP_PLACES=cores \
  taskset -c 0 .venv/bin/python benchmarks/macefield_standard_aln_scale.py run \
  --backend openmp --field-model /tmp/macefield-response-current-contract.json \
  --standard-model /tmp/mace-mh-0-current-Al-N.json \
  --output /tmp/macefield-standard-n6-low-core-standard-m1.json \
  --sizes n6 --policies optimized --trials 3 --warmups 3 --samples 5 \
  --low-memory --neighbor-skin 0 --response none
```

For public energy, forces, stress, and polarization timing, change the output
path and replace `--response none` with
`--response polarization --response-routing public-native`. Remove
`--low-memory` to reproduce retained-state results. The before-optimization
records require the preserved pre-change extension and are identified by their
hashes above; they cannot be regenerated from the final source alone.

Run the MPI qualification with the current OpenMP/MPI LAMMPS executable and
matching FP32/FP64 host artifacts:

```bash
env SYMMETRIX_LAMMPS_EXECUTABLE=/path/to/lmp \
  SYMMETRIX_LAMMPS_COMPACT_MODEL=/tmp/mace-mh-0-current-Al-N.json \
  SYMMETRIX_LAMMPS_FIELD_MODEL=/tmp/macefield-response-current-contract.json \
  SYMMETRIX_LAMMPS_JIT_HOST_ARTIFACT_FLOAT32=/path/to/fp32-plugin.so \
  SYMMETRIX_LAMMPS_JIT_HOST_ARTIFACT_FLOAT64=/path/to/fp64-plugin.so \
  .venv/bin/python -m pytest -q \
  pair_symmetrix/test/test_pair_symmetrix_factorized_mpi.py
```

## Current-head OpenMP scaling refresh (2026-08-24)

This refresh measures the committed `streamed-edge` head
`6e40d9a6db132e14f3a7cf226ef2fc156c14e3ce` at 1, 2, 4, 8, and 16 physical
cores. It supersedes the need for a current-source OpenMP scaling rerun but
does not replace the historical before/after results above.

The extension was rebuilt from scratch as Release, OpenMP-only, native
architecture on the AMD Ryzen 9 9950X3D2. GCC/GFortran 15.2.0 detected
AVX-512, Kokkos Serial and CUDA were disabled, and OpenBLAS was enabled and
pinned to one thread. The installed extension is:

```text
/tmp/symmetrix-openmp-scaling-6e40d9a-20260824/stage/symmetrix/symmetrix.cpython-312-x86_64-linux-gnu.so
SHA-256: 8a0edbeb6e64be0d8a1236eaa1417b149234ce06d06dd733822cc6df9f565130
```

Every point uses the same 864-atom 6x6x6 periodic wurtzite AlN structure,
FP32 Kokkos OpenMP, a 6.0-A model cutoff, zero skin, a 6.0-A effective cutoff,
and 78,624 directed edges. Logical CPUs 0-15 are distinct physical cores;
SMT siblings 16-31 were excluded. The electric field is
`[0.01, -0.02, 0.03]`. MACEField requests public energy, forces, six-component
ASE stress, and polarization. The matched standard-MACE control requests
energy, forces, and stress.

Both cases select `streamed_edges="direct"`, `low_memory=True`, compact
`unit-f32-radius-f64-v1` geometry, standard M0, R0 `v2_receiver`, standard M1
with recomputation, R1 `jit_all` forward, and R1 `jit` reverse. Required host
JIT selected artifact `jit-r1-gen2-f32-29f6298ae84d2005` from the task-local
cache, with zero fallback evaluations. The model hashes remain:

| Artifact | SHA-256 |
|---|---|
| Standard MACE compact JSON | `03ba3e74170c2a51ba93dd0309728c9dbd094ce67d01cf8338993784b7eee574` |
| MACEField compact JSON | `52dceeca5ed876bc12e82835574cbc83b85a5055f834560cbb3acd426be31fdf` |

These are structurally matched 128-channel, `l_max=3` models but have
different trained parameters. The comparison therefore measures the extra
implementation cost of the field-aware public result path, not physical-output
parity between the models.

Each table entry is the median of three fresh-process medians, after three
warmups and over five measured calls per process. Speedup and efficiency use
the corresponding one-thread median as the baseline.

| Physical cores | MACEField ms/call | MACEField us/atom | Speedup | Efficiency | Standard MACE us/atom | Field overhead |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 200.881 | 232.502 | 1.000x | 100.0% | 211.376 | 9.99% |
| 2 | 119.838 | 138.701 | 1.676x | 83.8% | 120.286 | 15.31% |
| 4 | 71.036 | 82.218 | 2.828x | 70.7% | 73.260 | 12.23% |
| 8 | 49.695 | 57.517 | 4.042x | 50.5% | 52.697 | 9.15% |
| 16 | 34.246 | 39.636 | 5.866x | 36.7% | 36.959 | 7.25% |

The standard-MACE speedups are 1.000x, 1.757x, 2.885x, 4.011x, and 5.719x.
MACEField therefore remains in the same performance class as standard MACE at
every core count. Scaling is useful but sublinear above four cores on this
864-atom workload; the 16-core result is 5.866x faster than one core, rather
than a near-linear 16x.

Fresh-process MACEField median ranges were 200.719-204.260 ms at one core,
118.874-122.148 ms at two, 70.588-71.328 ms at four, 48.820-49.993 ms at
eight, and 33.928-34.515 ms at 16. Across all 15 MACEField records, energy and
both reported force norms were bitwise stable. The total range in maximum
stress was `6.80e-16 eV/A^3`; polarization component ranges were `2.33e-18`,
`2.87e-18`, and `1.39e-17`.

The raw result hashes are:

| Physical cores | Result JSON SHA-256 |
|---:|---|
| 1 | `cb4cde20294b72f49d87b90e66c9cd0f0b723761673f307cd58e562fe3e16895` |
| 2 | `f1d28a0abbc5c8f4d1d7165f9cdadc371b44e5c63eaad0d21d9f2a5f3555b63b` |
| 4 | `407ac69b58c4cd9e0bb8fb7117b7b1b2aa58c1e25ca6dbe834edc47f39cc2d6b` |
| 8 | `b125fb118ca9f765ed1b337bf9ed4dcfe5435ab0964049d89c58810113cd0e26` |
| 16 | `ec5fcfe079b6c929fdd773c44d5f5fd963cc2bfba10a0cab0d33eb0df1ca5678` |

The per-point command template below uses `<N>` in both the affinity and
thread settings. For these measurements, `<CPUSET>` was `0`, `0-1`, `0-3`,
`0-7`, or `0-15` for `<N>` equal to 1, 2, 4, 8, or 16, respectively.

```bash
env SYMMETRIX_SOURCE_ROOT="$PWD" \
  SYMMETRIX_EXTENSION=/tmp/symmetrix-openmp-scaling-6e40d9a-20260824/stage/symmetrix/symmetrix.cpython-312-x86_64-linux-gnu.so \
  SYMMETRIX_JIT_CACHE=/tmp/symmetrix-openmp-scaling-6e40d9a-20260824/jit-cache \
  SYMMETRIX_JIT_POLICY=required KOKKOS_NUM_THREADS=<N> OMP_NUM_THREADS=<N> \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 OMP_PLACES=threads OMP_PROC_BIND=spread \
  taskset -c <CPUSET> .venv/bin/python \
  benchmarks/macefield_standard_aln_scale.py run \
  --backend openmp \
  --field-model /tmp/macefield-response-current-contract.json \
  --standard-model /tmp/mace-mh-0-current-Al-N.json \
  --output /tmp/symmetrix-openmp-scaling-6e40d9a-20260824/results/threads-<N>.json \
  --sizes n6 --policies optimized --trials 3 --warmups 3 --samples 5 \
  --low-memory --neighbor-skin 0 \
  --response polarization --response-routing public-native
```

## OpenMP efficiency investigation (2026-08-24)

The apparent fall from 70.7% four-core efficiency to 36.7% at 16 cores is
primarily a property of the exact-cutoff public benchmark boundary, not the
parallel factorized evaluator. The refresh above deliberately uses zero skin.
In that configuration `_can_use_native_geometry()` rejects cached native
geometry, and every public `calculate()` call reconstructs the 6.0-A graph
through ASE `neighbor_list("ijdD", atoms, cutoff)`.

Matched cached-host result routing reuses the exact same 78,624-edge input
graph. The difference between public and cached-host medians is nearly fixed at
15.2-15.9 ms as OpenMP threads increase:

| Physical cores | Public exact cutoff (ms) | Cached-host results (ms) | Fixed public overhead (ms) | Cached-host speedup | Cached-host efficiency |
|---:|---:|---:|---:|---:|---:|
| 1 | 200.881 | 185.729 | 15.152 | 1.000x | 100.0% |
| 4 | 71.036 | 55.894 | 15.142 | 3.323x | 83.1% |
| 8 | 49.695 | 34.435 | 15.260 | 5.394x | 67.4% |
| 16 | 34.246 | 18.368 | 15.878 | 10.112x | 63.2% |

Evaluator-only measurements, which exclude graph and public result handling,
are 186.332, 54.070, 32.232, and 16.859 ms at 1/4/8/16 cores. The native
factorized evaluator therefore reaches 11.052x speedup and 69.1% efficiency at
16 cores.

The production-style 0.5-A skin activates native prepared-geometry reuse. It
has a 6.5-A effective neighbor-list cutoff and 94,176 directed candidates, but
the model cutoff remains exactly 6.0 A. Inactive skin edges are clipped to the
model cutoff in prepared geometry, where the radial basis and cutoff envelope
make their contribution zero.

| Physical cores | Public 0.5-A skin (ms) | us/atom | Speedup | Efficiency |
|---:|---:|---:|---:|---:|
| 1 | 219.582 | 254.146 | 1.000x | 100.0% |
| 4 | 60.837 | 70.413 | 3.609x | 90.2% |
| 8 | 35.169 | 40.705 | 6.244x | 78.0% |
| 16 | 19.063 | 22.063 | 11.519x | 72.0% |

The skin-expanded one-thread call is slower because it carries 19.8% more
candidate edges, but at 16 cores it is 1.796x faster than rebuilding the exact
graph on every call. This is the relevant steady-state molecular-dynamics
route. Zero skin remains useful for exact graph-controlled comparisons, but
its end-to-end efficiency should not be presented as evaluator-only OpenMP
scaling.

Increasing the exact-cutoff structure to 6,912 atoms only partly reduces the
effect. MACEField public medians are 1,633.830, 549.354, 380.641, and 240.628
ms at 1/4/8/16 cores, or 236.376, 79.478, 55.070, and 34.813 us/atom. The
16-core speedup is 6.790x and efficiency is 42.4%. The modest improvement over
36.7% confirms that fixed overhead is important, while the remaining evaluator
also has intrinsic cache and bandwidth limits.

### Kernel attribution

A temporary Kokkos Tools callback timer profiled five complete exact-cutoff
evaluations at 1/4/8/16 cores. Callback overhead perturbs end-to-end timing, so
these leaf times are diagnostic attribution rather than benchmark results.
The two ranges responsible for most of the intrinsic 16-core loss are:

| Range | 1 core (ms) | 16 cores (ms) | Speedup | Time above ideal 16-core scaling (ms) |
|---|---:|---:|---:|---:|
| R1 source reverse | 34.616 | 6.054 | 5.718x | 3.890 |
| Field-H1 forward | 18.956 | 3.558 | 5.328x | 2.373 |
| R1 edge reverse | 31.716 | 2.709 | 11.708x | 0.727 |
| R1 forward | 18.691 | 1.822 | 10.258x | 0.654 |
| Standard R0 forward | 15.143 | 1.606 | 9.429x | 0.660 |
| Standard R0 coordinate reverse | 15.213 | 1.355 | 11.231x | 0.404 |

The generated host R1 source reverse schedules one source-owner task per node
and 16-channel tile, or 6,912 tasks for this model and structure. Each task
walks the source-owned edge list serially, performs irregular receiver gathers
and radial evaluations, and maintains compensated accumulators. The edge-owned
reverse has much finer independent work and scales nearly twice as well.

Field-H1 forward is a scalar node/output-harmonic/output-channel MDRange whose
inner loop scans field entries and all 128 input channels. It repeatedly reads
node features and field path matrices instead of expressing the contraction as
a packed matrix operation. Both ranges are consequently sensitive to shared
cache and memory traffic. At eight threads, spreading work over cores 0-3 and
8-11, across both L3 domains, improves evaluator-only time from 32.232 to
28.045 ms (13.0%). Splitting four threads across both domains improves only
1.5%, indicating that one L3 domain becomes limiting near eight active cores.

The raw timer TSV SHA-256 values at 1/4/8/16 cores are respectively
`8897566491a6669648ba034fec89be5c3df4fac74261bc3d00e4a5f1046ff491`,
`2331df6139837345e776a5a269d4696fe75e405dea99b518b4140bde933ce995`,
`60f01a862887d18ecfe9b4e24bd4453b558a1d14d88cb4a1c8356410f5643c6e`,
and `4fdc4daa3a94c6b24410eb97b9e3add536e51fb2d4f78d537978832c1f5a61a9`.

### Multithread correctness

Fresh 1/2/4/8/16-thread workers compared every public energy, force, stress,
and polarization component. Relative to one thread, energy is bitwise
identical at every point. The largest absolute differences over all thread
counts are `2.46e-15 eV/A` for force, `6.80e-16 eV/A^3` for stress, and
`1.39e-17` for polarization. All are reduction-order roundoff and pass the
normal FP32 tolerances by a wide margin.

The 0.5-A skin route is equally stable between one and 16 threads: energy is
bitwise identical, while maximum force, stress, and polarization differences
are `3.00e-15 eV/A`, `3.66e-15 eV/A^3`, and `1.39e-17`. Changing the graph
construction from exact cutoff to a skin-expanded candidate list changes total
energy by `3.49e-4 eV` (`4.04e-7 eV/atom`), maximum force by
`7.94e-6 eV/A`, stress by `1.02e-7 eV/A^3`, and polarization by `4.90e-9`;
these remain within the established FP32 graph-order tolerance.
