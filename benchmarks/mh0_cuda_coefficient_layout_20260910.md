# MH-0 CUDA cubic-coefficient layout experiment

Date: 2026-09-10. The standalone microbenchmark is retained; the production
prototype was rejected and removed.

## Decision

Keep the `float4` coefficient-layout experiment in
[`mh0_radial_point_microbench.cu`](mh0_radial_point_microbench.cu), but keep the
generation-10 scalar coefficient layout in production. Packing gives a large
synthetic R1-forward improvement, but matched full-evaluator timing is neutral
at 4,000 atoms and regresses at 864 atoms. Nsight Systems also shows no R1
kernel improvement in the actual generated implementation.

This direction does not require edge sorting or any graph-builder interface.
The rejected prototype packed the model's R1 coefficients once during model
initialization. It therefore would have worked through the existing device
graph and LAMMPS pair-style paths, provided both used a matching generation-11
artifact. The lack of production speedup, rather than integration complexity,
is the reason it was removed.

## Layouts

The scalar control stores coefficients as
`[pair][interval][coefficient][function]`. A thread evaluating one radial
function issues four loads separated by a full function-count stride. Across
adjacent channel threads each individual coefficient load is coalesced.

The candidate stores one aligned `float4` at
`[pair][interval][function]`. It transfers the same four FP32 values per spline
evaluation, but expresses them as one 16-byte load and uses one base-address
calculation. The standalone benchmark maintains packed R0 and R1 arrays. The
production prototype added a second, model-owned packed R1 view only for CUDA
FP32; CPU, HIP, FP64, and Kokkos evaluation continued to use the scalar view.

For this MH-0 model, the extra production R1 view was
`3 * 255 * 1280 * 4 * sizeof(float) = 15,667,200` bytes (14.94 MiB). Its
one-time packing kernel took 0.0392 ms in the profiled process. An in-place
CUDA-only layout could avoid the duplicate storage, but would not change the
measured steady-state conclusion.

## Workload

- GPU: NVIDIA GeForce RTX 5090, 170 SMs, compute capability 12.0
- Driver: 610.43.02
- CUDA compiler/runtime: 13.3.73 / 13.3
- Precision: FP32 model values with the existing FP64 radius and cached local
  spline coordinate
- Model: MACE-MH-0 Al/N, SHA256
  `c202e5bc6da9a2b0ca4974595869fa5f4f0cdb7f37b58d87415ecd95f048eb22`
- Cutoff: 6 A; skin: 0 A; effective cutoff: 6 A
- 864 atoms: 78,624 directed edges
- 4,000 atoms: 364,000 directed edges

The production cells use wurtzite AlN coordinates displaced by independent
Gaussian noise with standard deviation 0.02 A and seed `20260908`. Edge sorting
is disabled and the generation-10 radial-point cache is enabled. Both graph
hashes match the earlier cache-only controls:

- 864 atoms: `038ff6f89a81c20f7e37b09859c4216fb6f253071a8c2496b50675e61ea9b3f4`
- 4,000 atoms: `8d2eba37287114dd37adf4bc9ec6d4954644d24b4f12f8e022f650a6e7769759`

## Microbenchmark

Each cell is the median of three fresh-process medians. Every process performs
10 warmups and 31 CUDA-event measurements. The coefficient suite uses the
production dimensions, access layouts, owner mappings, launch shapes, and
allocation sizes, with deterministic synthetic tensor contents. All packed
versus scalar output comparisons have `max_abs_difference == 0`.

| Stage | 864 scalar | 864 `float4` | Change | 4,000 scalar | 4,000 `float4` | Change |
|---|---:|---:|---:|---:|---:|---:|
| R0 forward | 0.28344 | 0.27878 | -1.65% | 0.26502 | 0.26502 | 0.00% |
| R0 reverse | 0.27522 | 0.27270 | -0.92% | 0.40524 | 0.40447 | -0.19% |
| R1 forward | 0.46852 | 0.36881 | **-21.28%** | 0.36352 | 0.27391 | **-24.65%** |
| R1 reverse | 0.43237 | 0.43519 | +0.65% | 0.42137 | 0.41342 | -1.89% |

All values are `us/atom`; negative changes are faster. Only R1 forward shows a
large, size-independent signal, so the production prototype was limited to
CUDA FP32 R1 rather than expanding the change to R0.

## Production timing

The control is the retained generation-10 cache-only result. The candidate is
a fresh generation-11 CUDA build with packed R1 coefficients. Edge ordering is
disabled in both. Each cell is the median of three fresh-process medians with
10 warmups and 30 synchronized evaluations.

| Atoms | Gen10 scalar process medians | Gen10 scalar | Packed process medians | Packed | Change |
|---:|---|---:|---|---:|---:|
| 864 | 3.22622, 3.22888, 3.23365 | 3.22888 | 3.26092, 3.26326, 3.26482 | 3.26326 | **+1.07%** |
| 4,000 | 2.87028, 2.87179, 2.87262 | 2.87179 | 2.85909, 2.86245, 2.86416 | 2.86245 | -0.33% |

Values are `us/atom`. The 4,000-atom difference is smaller than the spread
between individual process medians and is not a useful production improvement.
All candidate runs selected `jit_all` forward and `jit` reverse, used a
generation-11 artifact, and reported zero fallback evaluations.

## Nsight and generated code

The matched 4,000-atom, rattled, cache-only Nsight Systems traces give:

| Kernel | Generation 10 scalar | Packed candidate | Change |
|---|---:|---:|---:|
| R1 forward | 1.2245 ms | 1.2277 ms | +0.26% |
| R1 fused reverse | 2.4033 ms | 2.4392 ms | +1.49% |

The scalar trace contains one measured launch. The candidate report contains
11 stable instances of each target kernel; its table uses their mean. The
capture ranges differ, so whole-trace totals are not compared. These
un-replayed timeline measurements show that neither generated R1 stage
reproduces the synthetic R1-forward gain.

SASS inspection confirms that the layout transformation survived NVRTC. In
`symmetrix_factorized_forward_v2`, the scalar artifact contains 68 `LDG.E` and
two `LDG.E.64` instructions. The packed artifact contains 28 `LDG.E`, two
`LDG.E.64`, and ten `LDG.E.128` instructions. Thus forty scalar loads became
ten 128-bit loads. Resource usage changed as follows:

| Generated function | Scalar registers/thread | Packed registers/thread |
|---|---:|---:|
| R1 forward | 96 | 96 |
| R1 fused reverse | 96 | 80 |
| R1 edge reverse | 167 | 128 |

The instruction reduction is real, but it does not shorten the production
critical path. The current scalar loads are already coalesced across channel
threads, and coefficient bytes are only part of the generated kernel's L2
traffic. The full kernel also loads features, harmonics, weights, and adjoints
and performs substantially more contraction work than the synthetic body. A
single vector load makes all four coefficients available together, but the
dependent polynomial FMAs still wait for that load. The scalar form gives the
compiler more independent loads to interleave with surrounding work. Lower
register counts in reverse also did not change the limiting occupancy or
scheduler behavior enough to improve time.

Nsight Compute was attempted earlier on this host, but hardware-counter
collection fails with `ERR_NVGPUCTRPERM`. No replay time or inferred timeline
metric is substituted for the unavailable counter data.

## Reproduction and artifacts

Build and run the retained microbenchmark coefficient suite with:

```bash
/usr/local/cuda-13.3/bin/nvcc -O3 -std=c++20 -arch=sm_120 -lineinfo \
  -Xptxas=-v benchmarks/mh0_radial_point_microbench.cu \
  -o /tmp/symmetrix-mh0-coefficient-layout-20260910/mh0_radial_point_microbench

/tmp/symmetrix-mh0-coefficient-layout-20260910/mh0_radial_point_microbench \
  --atoms 4000 --edges-per-atom 91 --warmups 10 --repeats 31 \
  --suite coefficient \
  --output /tmp/symmetrix-mh0-coefficient-layout-20260910/result-4000.json
```

Use `--atoms 864` for the small workload. The retained source SHA256 is
`659933144896527abc575cb07035d045736957998b5a72d770d490c641ddf45a` and
the measured binary SHA256 is
`f4847f5a8fa3370d1a832796aff782562132a7003f404bb0198f9ae1c55c7bf9`.

Raw microbenchmark records are under
`/tmp/symmetrix-mh0-coefficient-layout-20260910/`. Production timing records,
the generated source, cubin, smoke result, and candidate Nsight Systems report
are under `/tmp/symmetrix-mh0-coeff-cuda-20260910/`. The candidate extension
SHA256 is
`9c2b4cfbb4de86b8bb428bbe63c6fae1e85f32e85f6cfa14489d7a0ecbdb78b9`.
The matched scalar trace is
`/tmp/symmetrix-mh0-radial-final-profile-20260910/cache-only-4000-perturbed.nsys-rep`.
