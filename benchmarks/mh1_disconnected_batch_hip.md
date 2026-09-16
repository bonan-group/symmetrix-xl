# MH-1 disconnected small-structure batching on HIP

Date: 2026-08-27

## Scope

This experiment tests steady-state geometry-optimization throughput for several
independent 20-atom periodic AlN cells. It concatenates their node and directed
edge arrays into one disconnected graph, offsets every source and receiver
index by its slot's node offset, and introduces no cross-slot edges. This is
not a periodic supercell and does not change the physical neighborhoods.

The prepared topology is retained with a `0.5 A` neighbor-list skin. The model
cutoff is `6.0 A`, the effective list cutoff is `6.5 A`, and inactive skin
edges are clamped to the model cutoff by the existing prepared execution path.
Each slot has 20 atoms and 2,180 directed list edges.

## Environment

- Device: AMD Radeon 8060S Graphics, `gfx1151`, 20 compute units, wave width 32
- ROCm runtime/compiler: 7.14
- Backend and precision: Kokkos HIP, FP32
- Model: `/tmp/mace-mh-1-current-direct-Al-N.json`
- Model SHA-256:
  `fb1dc908fd0f7b99aa84279dd5ae598dbde62cb15b95c2b076581872b05d6917`
- Native extension:
  `/tmp/symmetrix-mh1-shared-r1-hip.l8prRb/build/symmetrix.cpython-314-x86_64-linux-gnu.so`
- Extension SHA-256:
  `d4d793596e0bec27092a9ecc562996c9e9c313054f4f4302aa80364a53398739`
- Direct executor: MH-1 `pair_spline_v1`, cached hipRTC artifact
- Timing: five warmups, 20 samples, median wall time with a device fence
- Result record:
  `/tmp/mh1-disconnected-batch-device-stress-int64-fp32.json`
- Result SHA-256:
  `b619d1d54b4d7cd40e5fc67edf05037bd7d7296e660c417a07f61cf3fbcae4f2`

The AOT Kokkos extension reports that it was compiled for `gfx1100` while the
runtime device is `gfx1151`. The hipRTC direct module targets `gfx1151`. These
results qualify the batching direction on this local system, but a fully
native `gfx1151` rebuild remains necessary for a release number.

## Throughput results

The sequential control executes the same prepared 20-atom graph once per
slot, changing only geometry. The batched case executes one disconnected graph.

### Native energy and analytic-force evaluation

| Slots | Atoms | Sequential group (ms) | Batched group (ms) | Batched us/atom | Batched us/structure | Structures/s | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 3.380 | 3.374 | 168.98 | 3,374 | 296 | 1.00x |
| 2 | 40 | 6.917 | 4.329 | 108.22 | 2,164 | 462 | 1.60x |
| 4 | 80 | 13.758 | 5.769 | 72.11 | 1,442 | 693 | 2.38x |
| 8 | 160 | 27.921 | 9.118 | 56.99 | 1,140 | 877 | 3.06x |
| 16 | 320 | 57.604 | 18.685 | 58.39 | 1,168 | 856 | 3.08x |

### Energy and atom-force result materialization

This scope also copies node energies and runs the native GPU edge-to-atom
force reduction before returning atom forces to the host.

| Slots | Sequential group (ms) | Batched group (ms) | Batched us/atom | Batched us/structure | Structures/s | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.532 | 3.529 | 176.45 | 3,529 | 283 | 1.00x |
| 2 | 7.211 | 4.468 | 111.70 | 2,234 | 448 | 1.61x |
| 4 | 14.298 | 5.978 | 74.72 | 1,494 | 669 | 2.39x |
| 8 | 29.909 | 9.499 | 59.37 | 1,187 | 842 | 3.15x |
| 16 | 60.082 | 19.359 | 60.50 | 1,210 | 826 | 3.10x |

### Energy, forces, and individual slot stresses

The evaluator stores prepared edge prefix offsets once per batch topology.
Each call copies the current graph volumes to the device and launches a Kokkos
team reduction for every `(graph, tensor component)`. Directed edge forces stay
on the device and the binding returns one 3x3 tensor per graph.

| Slots | Sequential group (ms) | Batched group (ms) | Batched us/atom | Batched us/structure | Structures/s | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.551 | 3.561 | 178.07 | 3,561 | 281 | 1.00x |
| 2 | 7.279 | 4.494 | 112.36 | 2,247 | 445 | 1.62x |
| 4 | 14.411 | 6.036 | 75.45 | 1,509 | 663 | 2.39x |
| 8 | 30.125 | 9.529 | 59.56 | 1,191 | 840 | 3.16x |
| 16 | 60.570 | 19.383 | 60.57 | 1,211 | 825 | 3.12x |

Eight slots are the practical knee. Sixteen slots do not improve per-structure
latency and slightly reduce native-evaluation throughput.

## Correctness

Across 1, 2, 4, 8, and 16 slots:

- Maximum per-slot energy difference: exactly `0 eV`
- Maximum atom-force difference: `1.18e-15 eV/A`
- Maximum individual stress difference: `1.39e-17 eV/A^3`
- Maximum aggregate native-stress difference: `1.39e-17 eV/A^3`

The same two-slot check passed in FP64 on the OpenMP backend with a maximum
per-slot stress difference of `3.47e-17 eV/A^3`. Batch edge prefix offsets are
64-bit on the Python, host, and device sides, including the device reduction
range, so cumulative edge counts do not truncate at 32-bit limits.

The legacy scalar stress API still returns the volume-weighted aggregate of a
disconnected batch. The new prepared-batch API returns independent per-slot
stress tensors suitable for concurrent cell optimization.

## Device stress implementation

`_prepare_factorized_batch(token, edge_offsets)` validates and retains the
topology-owned edge prefix offsets on the Kokkos device. A new graph token
invalidates this metadata. `_reduce_batched_stress(volumes, token)` validates
the current completed evaluation, updates the small volume array, and launches
`9 * graph_count` Kokkos teams. Each team reduces one tensor component over its
graph's directed-edge interval and writes directly into a `graph_count * 9`
device buffer.

At eight slots, the independently timed energy/force and complete medians differ
by `0.030 ms` for the whole batch (`9.499 -> 9.529 ms`), or about
`3.79 us/structure`. The former diagnostic host split took `0.956 ms` beyond
energy and forces and achieved 761 structures/s; the device implementation
achieves 840 structures/s. A selected-region ROCm trace records one
`reduce_prepared_batched_stress` HIP dispatch per evaluation at a mean
`6.214 us`; the remaining API overhead is metadata transfer and returning the
72 stress values.

## Kernel profile

`rocprofv3 --kernel-trace --marker-trace --stats --selected-regions` captured
ten warmed evaluations. Both graph sizes launch the same 103 kernels per
evaluation. Device kernel time grows from 3.225 ms for one slot to 8.759 ms for
eight slots: 8x the graph work costs only 2.72x the device time.

| Stage family | 1-slot ms | 8-slot ms | 1-slot share | 8-slot share |
|---|---:|---:|---:|---:|
| R forward | 0.121 | 0.311 | 3.7% | 3.6% |
| R reverse | 0.513 | 1.049 | 15.9% | 12.0% |
| M0 node program | 1.057 | 3.162 | 32.8% | 36.1% |
| M1 node program | 1.466 | 3.845 | 45.5% | 43.9% |
| Harmonics | 0.018 | 0.203 | 0.6% | 2.3% |
| ZBL, fills, and other | 0.050 | 0.189 | 1.5% | 2.2% |

The R-forward kernels expand from 10 to 80 blocks and the R-reverse kernels
from 20 to 160 blocks. On a 20-CU GPU, the single-slot R reverse provides only
one block per CU, while the eight-slot graph provides eight scheduling waves.
Launch count and generated code remain unchanged.

Selected-kernel hardware counters show the resulting utilization improvement:

| Stage family | Occupancy 1 slot | Occupancy 8 slots | Memory busy 1 slot | Memory busy 8 slots |
|---|---:|---:|---:|---:|
| R forward | 5.1% | 43.8% | 27.3% | 92.8% |
| R reverse | 3.0% | 24.5% | 38.0% | 68.9% |
| M0 node program | 24.6% | 67.3% | 69.9% | 79.2% |
| M1 node program | 19.1% | 61.5% | 61.4% | 77.0% |

R reverse remains occupancy-limited even after batching. Its two generated
kernels use 184 and 256 VGPRs per thread respectively, so the larger graph
fixes insufficient grid depth but does not remove the per-block register limit.

## Commands

The throughput matrix used:

```bash
SYMMETRIX_SOURCE_ROOT="$PWD" \
SYMMETRIX_EXTENSION=/tmp/symmetrix-mh1-shared-r1-hip.l8prRb/build/symmetrix.cpython-314-x86_64-linux-gnu.so \
SYMMETRIX_JIT_CACHE=/tmp/symmetrix-mh1-shared-r1-diag.dCU41O \
LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64 \
.venv/bin/python benchmarks/mh1_disconnected_batch_benchmark.py \
  /tmp/mace-mh-1-current-direct-Al-N.json \
  --slots 1,2,4,8,16 --neighbor-skin 0.5 \
  --warmups 5 --repeats 20 \
  --output /tmp/mh1-disconnected-batch-device-stress-int64-fp32.json
```

The isolated stage profiles used the benchmark's `--profile-batch-repeats 10`
mode under `rocprofv3 --selected-regions`. Reports are in
`/tmp/mh1-batch-profile-slot1`, `/tmp/mh1-batch-profile-slot8`,
`/tmp/mh1-batch-counters-slot1`, and `/tmp/mh1-batch-counters-slot8`.
The current-binary device-stress trace used three warmed evaluations and is in
`/tmp/mh1-batched-stress-int64-profile`; its kernel statistics SHA-256 is
`891a49977bee43c58b351161c92c544f03f256559be35702b7f4349c13b7bd73`.

## Implementation direction

The experiment accepts disconnected batching. The next production interface
should use a fixed-capacity slot batch with:

1. Node and edge prefix offsets plus active node/edge extents per slot.
2. One prepared disconnected topology token while every active slot remains
   inside its neighbor skin.
3. Segmented device energy reduction; segmented virial/stress and atom-force
   reduction are already available.
4. An active-slot mask so converged geometry optimizations can be replaced
   without rebuilding unaffected slots.
5. An initial default capacity of eight 20-atom-class slots on this 20-CU GPU,
   followed by adaptive sizing based on atom and directed-edge counts.

Batching improves utilization; it does not make one individual optimization
finish 3x sooner unless the optimizer can advance multiple structures together.
