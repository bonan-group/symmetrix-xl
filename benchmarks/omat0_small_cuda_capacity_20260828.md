# OMAT-0 small CUDA capacity

Date: 2026-08-28

## Scope

This campaign measures FP32 MACE-OMAT-0 small inference on one local RTX 5090.
Every successful worker requests energy, forces, and stress through direct
streamed-edge execution. Each capacity point runs in a fresh process with one
warmup and two measured evaluations. The largest success and first failure are
each repeated twice. A separate 16,384-atom throughput point uses three warmups
and seven measured evaluations away from the allocator boundary.

Throughput is reported both as `us/atom` and `atoms/s`, with

$$
\mathrm{atoms/s} = \frac{10^6}{\mathrm{us/atom}}.
$$

## Environment and provenance

- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB, compute capability 12.0
- GPU UUID: `GPU-a7b75c49-a576-be8a-e560-4852a3c2afd7`
- Driver: 610.43.02; 110 MiB baseline use; no competing compute process after
  the campaign
- CUDA toolkit: 13.3.73; GCC/G++ 15.2; Python 3.12.13
- Kokkos execution spaces: CUDA + Serial; SpheriCart CUDA;
  `Kokkos_ARCH_BLACKWELL120=ON`
- Pre-specialization source HEAD:
  `934484f9ae6b779189990d24660f101b65f7676e`
- At baseline, uncommitted source changes were confined to the automatic build
  frontend and LAMMPS integration; the native evaluator and calculator sources
  used by the baseline extension matched HEAD.
- Pre-specialization extension SHA-256:
  `d97a64b70e3c159e2f1666a0e2286fbc23dd82458ef91b377b3a98a14843c88a`
- Official checkpoint SHA-256:
  `0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5`
- Al/N compact model SHA-256:
  `83d9008621eea80755701cb5b2d653466513f8576d79d6a48caacc7f9572bb46`
- Compact model size: 8,324,981 bytes
- Model: OMAT-0 small, 128 channels, `l_max=3`, `L_max=0`, 6.0 A cutoff
- Structure: periodic wurtzite AlN, four atoms per primitive cell
- Neighbor skin: 0.5 A; effective cutoff: 6.5 A; 109 directed candidates/atom

The extension was freshly configured and compiled from the absolute current
source path. The generated CMake `git-state.txt` incorrectly retained
`dfceb7727-CLEAN`; it is not used as provenance evidence. The source HEAD,
source-scope diff, build configuration, and extension hash above identify this
qualification binary.

## Pre-specialization baseline

The first campaign established the implementation gap before the scalar M0
specialization was added.

### Execution selection

Successful workers select:

- direct R1 forward/reverse through cached NVRTC artifact
  `jit-r1-gen2-f32-44c7e72f932e893e`;
- runtime M0;
- generated R0 `v2_edge16`;
- M1 polynomial recomputation;
- full-retention MH-0 state; and
- Cartesian float64 retained-edge geometry.

### Capacity boundary

| Policy | Largest success | Directed candidates | Sampled VRAM | First failure | Result |
|---|---:|---:|---:|---:|---|
| Direct, `low_memory=False` | **119,164 (`n31`)** | 12,988,876 | 30,592 MiB | 131,072 (`n32`) | CUDA OOM, 8.875 GiB `M0_poly_adjoints_0` allocation |
| Direct, `low_memory=True` | Not runnable | n/a | n/a | n/a | Rejected: generated standard M0 is required |

Both `n31` fresh processes succeeded and both `n32` fresh processes failed.
The successful endpoint reports a 6,710,839,984-byte Cartesian geometry
workspace. The runtime M0 polynomial arrays dominate the remaining capacity.

OMAT-0 small has only the scalar `LM=0` M0 output with 94 terms. The built-in
generated standard M0 module is fixed to `L_max=1`, four output components, and
422 terms. Consequently the small model falls back to runtime M0, while the
public low-memory bundle fails closed because adjoint-state reuse requires the
generated M0 module. There is no runtime standard-M0 artifact preparation
mechanism that can enable the scalar-only model.

### Throughput

| Workload | Trial | ms/call | us/atom | atoms/s | Sampled VRAM |
|---|---:|---:|---:|---:|---:|
| Capacity endpoint, 119,164 atoms | 1 | 441.155 | 3.702085 | 270,118 | 30,592 MiB |
| Capacity endpoint, 119,164 atoms | 2 | 445.226 | 3.736243 | 267,649 | 30,592 MiB |
| Stable throughput, 16,384 atoms | 1 | 56.445 | **3.445139** | **290,264** | 4,844 MiB |

The stable row is the primary throughput result. The endpoint rows demonstrate
capacity behavior and may include near-boundary allocator or memory-pressure
effects.

The two endpoint workers produce identical total energy, force L2, and stress.
Their maximum absolute forces differ by only `1.4e-16 eV/A`.

## L_max=0 standard-M0 qualification

The post-specialization capacity extension has SHA-256
`450941978f48a345609b5eb581743815ead21cf8772085c121c296cc305b1d42`.
After review, the immutable term tables were changed from `std::array` to
equivalent device-safe `constexpr` arrays to eliminate NVCC host-constexpr
warnings. The final rebuilt extension has SHA-256
`e61783e30626b8bd639cec92f3ba271363c8a6799fb7f3ecc1d27308437f7113`.
It reproduced the matched-size selection, numerical summaries, and memory
footprints; the capacity boundary below was measured with the pre-cleanup
capacity binary. Both were built fresh with the same CUDA 13.3 BLACKWELL120
configuration and compact model. The specialization selects only after exact
checks of `l_max`, `L_max`, correlation, output count, term count, and all 94
canonical monomials. Channels, type count, and weights remain runtime
quantities.

Successful workers selected:

- R1 `jit-r1-gen2-f32-44c7e72f932e893e` through NVRTC;
- scalar standard M0 `standard-m0-lmax0-module-v1`;
- R0 `v2_edge16`;
- M1 recomputation;
- direct forward `jit_all` and direct reverse `jit`; and
- zero factorized fallback in the focused direct-execution tests.

The retained policy used full MH-0 state and Cartesian float64 edge geometry.
The low-memory policy used aliased adjoints and float32 unit vectors with
float64 radii. Both standard-M0 polynomial value and adjoint capacities were
zero.

### Matched-size throughput and correctness

The primary comparison uses 16,384 atoms and 1,785,856 directed edges, three
warmups, and seven measured calls.

| Policy | ms/call | us/atom | atoms/s | Sampled VRAM |
|---|---:|---:|---:|---:|
| Retained, `low_memory=False` | 44.734 | **2.730321** | **366,257** | 2,568 MiB |
| Low memory, `low_memory=True` | 48.328 | **2.949721** | **339,015** | 1,766 MiB |

Relative to the 3.445139 us/atom runtime-M0 baseline, retained execution is
20.7% faster and low-memory execution is 14.4% faster. Low memory is 8.0%
slower than the newly specialized retained path at this matched size, while
saving 802 MiB, or 31.2% of sampled process VRAM.

The retained-versus-low-memory numerical differences are:

- total energy: `0.01009785 eV`, or `6.16e-7 eV/atom`;
- maximum absolute force summary: `5.18e-7 eV/A`; and
- maximum stress component: `2.38e-7 eV/A^3`.

These differences are consistent with the low-memory compact-geometry FP32
policy. The focused FP32 and FP64 tests additionally compare full force and
stress arrays against retained and runtime-M0 execution.

### Post-specialization capacity boundary

| Policy | Largest success | Directed edges | us/atom at boundary | atoms/s | Sampled VRAM | First repeated failure |
|---|---:|---:|---:|---:|---:|---|
| Retained | **275,684 (`n41`)** | 30,049,556 | 2.889-2.903 | 344,477-346,125 | 31,574 MiB | 296,352 (`n42`), CUDA OOM allocating 2.261 GiB |
| Low memory | **470,596 (`n49`)** | 51,294,964 | 3.157-3.164 | 316,054-316,797 | 30,668 MiB | 500,000 (`n50`), CUDA OOM allocating 553.1 MiB |

Both largest successes and both first failures were reproduced in two fresh
processes. The failure class is a clean CUDA allocator OOM in every case, not
an illegal-address execution ceiling. Boundary timings are capacity evidence;
the 16,384-atom rows above remain the primary throughput comparison.

The specialization raises retained capacity from 119,164 to 275,684 atoms,
or 2.31x. Low memory reaches 470,596 atoms, 3.95x the pre-specialization
baseline and 1.71x the new retained capacity.

## Comparison with OMAT-0 medium

The earlier matched-device medium campaign reached 219,488 atoms with retained
state and 389,344 atoms with `low_memory=True`. Before specialization, the
small model reached only 119,164 atoms because it could not use standard M0 or
the complete low-memory bundle. With scalar standard M0, small reaches 275,684
retained and 470,596 low-memory atoms, or 1.26x and 1.21x the corresponding
medium limits. The prior inversion was therefore an implementation
specialization gap, not an intrinsic property of the smaller architecture.

## Reproduction

The maintained worker is `benchmarks/low_memory_capacity.py worker`. The
baseline capacity search used repeats `16`, `31`, `46`, `38`, `34`, and `32`,
then confirmed `31` and `32` in fresh processes. The post-specialization
retained search used `16`, `31`, `38`, `40`, `41`, `42`, and `46`; the
low-memory search used `16`, `31`, `46`, `49`, `50`, `51`, `53`, and `61`.
The stable runs used repeat `16`, three warmups, and seven samples. Every
invocation set:

```bash
export CUDA_VISIBLE_DEVICES=0
export SYMMETRIX_SOURCE_ROOT="$PWD"
export SYMMETRIX_EXTENSION=/tmp/symmetrix-omat0-small-capacity-20260828/build/symmetrix.cpython-312-x86_64-linux-gnu.so
export SYMMETRIX_JIT_CACHE=/tmp/symmetrix-omat0-small-capacity-20260828/jit-cache
export SYMMETRIX_JIT_POLICY=required
```

Raw JSON and logs are under
`/tmp/symmetrix-omat0-small-capacity-20260828/`, including
`retained-capacity.json`, `retained-boundary.json`, per-worker records, and
stdout/stderr logs.

Post-specialization raw JSON, capacity records, failure logs, build cache, and
the exact campaign wrapper are under
`/tmp/symmetrix-omat0-m0-scalar-cuda-20260828/`.
