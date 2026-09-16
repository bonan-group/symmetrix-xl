# Extreme-scale CUDA indexing qualification

Date: 2026-08-29

## Scope

This record distinguishes the historical OMAT-0-medium CUDA illegal address
from a true memory-capacity boundary. It covers generated R1 global offsets,
the maintained SpheriCart 1.0.3 CUDA patch, JIT cache invalidation, and the
capacity driver's failure contract. It does not change the graph ABI: node and
edge identifiers remain signed 32-bit values, so one process is still limited
to at most 2,147,483,647 directed edges.

## Root cause and correction

Generated R1 previously evaluated flattened expressions while `receiver` was
still `int32`, including

```cpp
(receiver * 40 + lme) * 128 + channel
```

The 5,120-element node stride first overflows signed 32-bit arithmetic at
receiver 419,431. The `n48` workload has 442,368 atoms and therefore crossed
that boundary. Casting the completed expression would be too late; the
scalable operand must be widened before the first multiplication.

Generated host, CUDA, and HIP R1 now form node, source, edge, harmonic,
gradient, coordinate, and directed-force offsets with `std::size_t` arithmetic.
The maintained SpheriCart patch similarly uses `unsigned long long` for global
sample and edge offsets in forward, gradient, Hessian, and backward indexing.
Its launch ceiling division uses

```cpp
x / bdim + (x % bdim != 0)
```

instead of the overflow-prone `(x + bdim - 1) / bdim`. Symmetrix generated
launch helpers use the same division form. Bounded angular-component,
thread/lane, and shared-memory indices remain 32-bit.

Runtime M0/R0 modules were audited separately. Their ABI already uses 64-bit
node and edge counts; M0 offsets derive from a `long long node`, and R0 uses a
`long long edge` or widens the edge before applying the harmonic stride.

The shared JIT generation is advanced from 2 to 3. Existing generation-2 host,
CUDA, and HIP artifacts therefore cannot be reused after this source change.

## Pre-execution cardinality admission

Python neighbor-list construction invokes both the common topology admission
and, when available, the loaded evaluator's model-dependent admission. Every
standard-MACE native Kokkos graph/evaluation entry point repeats authoritative
validation for LAMMPS and direct callers. MH-1 retains its separate topology
admission and widened generated indexing. Invalid standard-MACE graphs are
therefore rejected before topology narrowing, model workspace allocation, or
kernel launch. The compact topology contract remains

\[
N_r,N_f \leq \mathrm{INT32\_MAX}-1,
\qquad E \leq \mathrm{INT32\_MAX},
\]

where $N_r$ is the receiver-atom count, $N_f$ is the feature-node count in a
domain-decomposed evaluation, and $E$ is the directed-edge count. The one-node
margin supports CSR offsets of length $N+1$.

The native guard also derives limits from the loaded model. Execution ranges
that are still genuinely signed 32-bit must satisfy

\[
N_r C,\;N_f C,\;N_r n_{lm},\;N_r n_{LM},\;
N_r n_{\mathrm{coupled}} \leq \mathrm{INT32\_MAX}.
\]

For the 128-channel OMAT-0-small test contract, the receiver-channel range is
the first of these limits and admits at most 16,777,215 receivers. This is a
representability limit, not a practical memory-capacity claim.

Flattened storage extents are checked separately with overflow-detecting
`size_t` products, including $3E$ coordinates, $En_{lm}$ harmonics,
$3En_{lm}$ harmonic gradients, $E(l_{\max}+1)C$ R0 values, $EPC$ R1 values,
and the corresponding node/channel/angular state tensors. In particular,
$EC$ is not capped at `INT32_MAX`: direct edge kernels launch over $E$ and use
64-bit flattened addressing for channel-coupled storage. A graph at the
`INT32_MAX` edge boundary therefore passes allocation-free admission when all
byte extents remain representable, although a real evaluation will normally
reach the device allocator limit much earlier.

Admission failures report the rejected count or named product and its limit.
They do not replace ordinary CUDA OOM handling: a representable graph can still
be too large for the available device memory.

Fresh post-guard qualification used a GNU 15.2/Kokkos OpenMP build and a CUDA
13.3/Kokkos CUDA+Serial `BLACKWELL120` build. The staged extension hashes were:

- OpenMP: `471b5c9cbd422046db3501393104bdd7a3e30347b25f9bd37d02efd13e3c169b`;
- CUDA: `4cc1a661a4e0def483530ad9c41f540ec4f2744a9139023fe6cb28aa0db4c1fd`.

The allocation-free native tests admit the historical 1,000,188-atom,
109,020,492-edge graph, the full `INT32_MAX` edge boundary, and flattened
edge/channel storage larger than `INT32_MAX`. They reject the first atom,
feature-node, edge, receiver-channel, and feature-node-channel values beyond
their limits. A real two-node prepared factorized graph also passes admission.
The focused OpenMP guard/Python/indexing suite passed 9 tests; the broader
calculator, capacity-driver, and JIT-generation suite passed 131 tests with 19
dependency skips. On the RTX 5090, 8 CUDA guard/prepared/indexing tests passed,
and the compact NVRTC direct-versus-generic energy/force regression passed.

## A100 capacity result

The qualification used one NVIDIA A100-SXM4-80GB, OMAT-0-medium FP32,
`low_memory=True`, energy, forces, and stress, a 6.0 A model cutoff, a 0.5 A
skin, and a 6.5 A effective cutoff. Each probe ran in a fresh process.

| Repeat | Atoms | Directed edges | Result | us/atom |
|---:|---:|---:|---|---:|
| `n48` | 442,368 | 48,218,112 | success | 8.982 |
| `n58` | 780,448 | 85,068,832 | success | 8.887 |
| `n63` | 1,000,188 | 109,020,492 | success twice | 9.607, 9.612 |
| `n64` | 1,048,576 | 114,294,784 expected | CUDA OOM twice | n/a |

The `n64` failure is a 2 GiB Kokkos CUDA allocation failure. No post-fix
extreme-size probe produced an illegal address. Thus `n63` is the largest
demonstrated success for this exact 80 GB hardware/model/graph contract, while
`n64` is a true allocator upper bound. These values must not be hardcoded for
other GPU memory sizes, models, precisions, cutoffs, skins, or neighbor
densities.

Installed qualification identities:

- extension SHA-256:
  `431dd5f9c25e4de92be6f9c4e4b87a4228b131843da8de840776629c2d69d6d9`;
- OMAT-0-medium Al/N JSON SHA-256:
  `ec3a81da0d4aac597f7cb249d5743c60c81215f28da7725382c0f3ff1ee0a54f`;
- LAMMPS SHA-256:
  `06d698f1844115e128e090288162c4f3dea378f5d0d0571152e7f038635dbcd1`.

Remote raw records are under
`/opt/symmetrix/qualification/index64-20260829/capacity-oom`. The boundary
summary SHA-256 is
`56d5c330ae26b895dac335075399867ab194628261c6a9cc493d89fad69574d7`.

## RTX 5090 current-source confirmation

A fresh CUDA 13.3/Kokkos `BLACKWELL120` build of the reviewed source was also
checked on the local NVIDIA GeForce RTX 5090 (32,607 MiB). The staged extension
SHA-256 was
`3cd33a45957683dc09319979cb81c0fbba6d4ebfaabc018962a7de36af7a438a`;
the OMAT-0-medium Al/N compact-model SHA-256 was
`cd0fde722f21125a7640f4e8a9c45495d36b6a1463a379583ed9c71e949b89b2`.

With the same FP32 direct low-memory property contract and cutoff policy,
`n46` completed with 389,344 atoms and 42,438,496 directed edges at 5.105
us/atom. Total sampled device use was 32,144 MiB. The adjacent `n47` graph was
constructed successfully with 415,292 atoms and 45,266,828 directed edges,
then the first native evaluation failed with the original exception
`Cuda memory space failed to allocate 405.6 MiB`. No illegal address occurred,
and total device use returned to 68 MiB after process exit. This confirms the
historical local boundary remains a true allocation ceiling with the fresh
generation-3 artifact.

## Capacity-driver contract

`benchmarks/low_memory_capacity.py` now:

- writes an atomic schema-v2 partial record after host neighbor-graph
  construction and before the first native evaluation;
- preserves atom count, cutoff, skin, effective cutoff, directed-edge count,
  active allocation phase, and the original exception on failure;
- samples total used memory for the selected physical GPU, while retaining the
  process query only as secondary telemetry;
- records CUDA logical-device metadata, physical index, UUID, name, and memory;
- accepts only `cuda_oom` as a capacity upper bound and aborts immediately on
  `cuda_illegal_address` or any other failure class;
- classifies the captured original exception before later Kokkos-finalization
  diagnostics; and
- requires GPU memory to return to its pre-probe baseline, within a configurable
  tolerance, before the next fresh-process trial.

The campaign remains hardware-aware. Use `--gpu-device` when more than one
physical GPU is visible or when an explicit GPU/MIG UUID is needed.

## Reproduction

```bash
source /opt/symmetrix/activate.sh
export SYMMETRIX_EXTENSION=/opt/symmetrix/.venv/lib/python3.12/site-packages/symmetrix/symmetrix.cpython-312-x86_64-linux-gnu.so
export SYMMETRIX_JIT_POLICY=required

/opt/symmetrix/.venv/bin/python \
  /opt/symmetrix/source/benchmarks/low_memory_capacity.py worker \
  --model-kind omat0 \
  --model /opt/symmetrix/models/mace-omat-0-medium-Al-N.json \
  --low-memory true --dtype float32 --repeat 63 \
  --neighbor-skin 0.5 --warmups 1 --samples 1 \
  --memory-sample-interval 0.05 --output /path/to/n63.json
```

Repeat with `--repeat 64`. On the exact recorded A100 contract the required
classification is `cuda_oom`; an illegal address is always a correctness
failure, never capacity evidence.
