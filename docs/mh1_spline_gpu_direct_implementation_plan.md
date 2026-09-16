# MH-1 spline GPU direct implementation plan

Date: 2026-08-26

## Objective

Extend the MH-1 `pair_spline_v1` executor from CPU/OpenMP to prepared `direct`
execution on accelerators. The first qualification target is the local HIP
`gfx1151` system. Numerical kernels and contracts must remain backend-neutral
so the same implementation can be built by Kokkos CUDA or Kokkos HIP and can
later be lowered into the shared NVRTC/hipRTC renderer without changing the
algorithm.

The path must preserve the staged standard-MACE vocabulary:

1. R0 spline forward;
2. M0 node forward;
3. R1 spline forward;
4. M1 node forward;
5. M1 node reverse;
6. R1 spline reverse;
7. M0 node reverse;
8. R0 spline reverse.

No steady-state R stage may invoke the conditioner or density MLP per edge.

## Existing foundations

- `RadialFunctionSetKokkos` already stores cubic coefficients in the active
  Kokkos memory space and exposes device-callable value and derivative
  evaluation.
- The compiler-free spline R path already computes messages, density, harmonic
  adjoints, and distance adjoints without radial-basis or conditioned-MLP
  materialization.
- The V4 MH-1 node program is rendered from one shared numerical program for
  CUDA and HIP and already owns packed M0/M1 execution.
- Prepared direct graphs already provide receiver-contiguous edges and a
  source-owned reverse schedule on both device backends.

The current blocker is policy rather than data representation: construction
marks spline tables unavailable on accelerators, the setter rejects the
executor there, and selecting the spline executor disables the generated V4
node program.

## Milestone 1: portable staged-device baseline

Enable spline construction for Kokkos device execution and remove the host-only
executor gate. Retain the existing Kokkos R implementation initially, but run
it under `streamed_edges='direct'` on the prepared graph and on the evaluator's
execution space. Re-enable the generated V4 node program for spline execution,
because its input/output packet is independent of how the R message and density
were produced.

The selected backend identity is
`pair_spline_v1_staged_device_v5`. It means:

- R0/R1: ordered-pair cubic splines plus Kokkos tensor-product owners;
- M0/M1: shared generated V4 device node owners;
- graph and stream ownership: existing prepared direct execution;
- precision: FP32 for accelerator qualification.

This milestone adds no vendor API, device intrinsic, CUDA token, HIP token, or
new RTC/module ABI. CUDA and HIP compile the same templated numerical code.

### Correctness gates

- Device spline tables are available only when all layers prepare successfully.
- `generic` `mlp_reference` and `direct` `pair_spline_v1` agree for total and
  atomic energies, forces, and stress within the existing MH-1 FP32 tolerance.
- Tests include multiple species, perturbed coordinates, repeated evaluation,
  and prepared-graph reuse.
- Runtime metadata proves the staged-device backend, generated node execution,
  and zero direct fallback evaluations.
- The spline path reports no generated conditioning launch.

### HIP qualification contract

- Fresh Kokkos HIP build for `gfx1151`; do not reuse a CPU or older HIP cache.
- Record imported package and extension paths, execution space, extension hash,
  GPU identity, model hash, atom count, directed-edge count, model cutoff,
  neighbor skin, effective cutoff, warmups, and samples.
- Primary workload: official MH-1 `omat_pbe`, FP32, 864-atom wurtzite AlN,
  78,624 directed edges, 6.0 A model cutoff, zero skin.
- Compare against the same extension's generated direct `mlp_reference` path.
- Report total and per-stage performance in `us/atom`.

## Milestone 2: dedicated spline device R owners

Profile Milestone 1 with `rocprofv3`. If either spline R stage is dominated by
temporary edge-weight materialization, atomics, launch fragmentation, or tensor
product layout conversion, lower the accepted host-V5 ownership model into
portable graph-wide Kokkos owners first:

- receiver-owned forward reads one spline interval per edge and accumulates
  packed messages and density;
- source-owned reverse traverses prepared source edges, recomputes spline value
  and derivative, and produces source, harmonic, and distance adjoints;
- owners consume spline coefficients as runtime device data rather than
  embedding the tables in source;
- schedules describe receiver/source ownership and channel/component tiles;
- the numerical implementation uses only Kokkos policies, views, team scratch,
  and device-callable spline evaluation.

The retained implementation does not add an RTC packet or module ABI. Existing
V4 compact-radial artifacts remain loadable and retain identical dispatch. A
future generated spline owner remains possible, but real CUDA execution is
deferred until an NVIDIA qualification host is explicitly in scope.

## Performance decision

Milestone 1 is retained if it provides a meaningful end-to-end improvement and
its R stages are no longer the dominant cost. Milestone 2 is required if spline
R remains materially slower than the corresponding generated compact-radial R
stages or if the portable Kokkos baseline retains graph-sized intermediates that
the generated ownership model can eliminate.

No experimental public execution mode is added. `direct` remains the public
algorithm, and `pair_spline_v1` remains an internal edge-executor selector until
device correctness and performance are qualified.

## HIP implementation outcome

The accepted device path is graph-wide and contains no per-edge MLP:

- R0/R1 forward uses receiver/channel ownership and evaluates cubic spline
  weights on demand. An explicit `{receiver=1, channel=64}` Kokkos MDRange tile
  groups one wave64 or two wave32/CUDA warps on one receiver edge list and
  coalesces channel access.
- R0/R1 source reverse uses prepared source segments with an explicit
  `{source=1, channel=64, component=4}` tile.
- Harmonic and distance reverse are fused in one edge-owned Kokkos TeamPolicy
  kernel. Each channel evaluates a spline value and derivative once, reuses the
  tensor contraction for both adjoints, and reduces through per-team scratch.
- M0/M1 continue to use the shared generated V4 device node program.

The implementation contains no HIP or CUDA runtime calls, vendor intrinsics, or
backend-specific source. The C++ templates and launch bounds are valid for both
Kokkos HIP and Kokkos CUDA. Only the local HIP build and execution were
qualified in this milestone; CUDA compilation and performance remain untested.

### Rejected experiments

- Graph-wide materialization of all spline weights and derivatives regressed
  the 864-atom result to `1882.96 us/atom` and increased edge workspace to
  `1104.54 MiB`. MH-1 has thousands of path weights, so this representation is
  not comparable to materializing a compact radial basis.
- Changing receiver forward from instruction-first to edge-first measured
  `712.42 us/atom` versus the then-current `700.14 us/atom`. Reusing the spline
  interval did not compensate for reloading dynamic instruction metadata per
  edge.

### Local HIP qualification

Qualification used the official MH-1 AlN model (`sha256:fb1dc908fd0f7b99a`
prefix), FP32, 864 atoms, a `6.0 A` model and effective cutoff, zero neighbor
skin, and 78,624 directed edges on the local `gfx1151` AMD Radeon 8060S. The
fresh extension hash is
`sha256:928ce61ddadeefc5dadeee7f6b5f4edc3bc7332e3ba9fc5133f7874715b27b79`.
The build emitted the existing `gfx1100` compatibility warning for part of the
Kokkos code while targeting the local `gfx1151` device; no CUDA build or run
was performed.

The first five-warmup, ten-sample comparison measured `132.560 us/atom` for
the generated `mlp_reference` control and `131.566 us/atom` for spline direct.
That near tie prompted a matched attribution audit. A fresh ten-warmup,
twenty-sample run with the same extension measured:

| Executor | Median (us/atom) | Min (us/atom) | Max (us/atom) |
| --- | ---: | ---: | ---: |
| Standard MH-0 direct | 17.095 | 16.990 | 17.250 |
| MH-1 generated `mlp_reference` control | 125.975 | 125.526 | 127.404 |
| MH-1 spline `pair_spline_v1` direct | 130.169 | 129.799 | 130.682 |

The MLP control is not spline-backed. In one measured evaluation its generated
conditioning-forward, conditioning-reverse, graph-forward, source-reverse, and
edge-reverse counters each increased by two. All five counter families remained
zero for spline direct. Spline therefore removes the two per-layer conditioning
MLPs, rather than reaching the same computation through another entry point.

Matched selected-region `rocprofv3` traces used the Kokkos rocprofiler
connector so the portable spline kernels inherit their enclosing CPU stage
regions. The analyzer reports the same `R0`, `M0`, `R1`, and `M1` forward and
reverse names for generated and portable GPU execution. Device time was:

| Stage | MH-0 direct (us/atom) | MH-1 MLP (us/atom) | MH-1 spline (us/atom) | Spline - MLP |
| --- | ---: | ---: | ---: | ---: |
| R0 forward | 1.594 | 4.889 | 10.937 | +6.049 |
| M0 forward | 0.323 | 9.003 | 8.849 | -0.153 |
| R1 forward | 0.733 | 14.030 | 27.022 | +12.992 |
| M1 forward | 0.932 | 9.809 | 9.852 | +0.043 |
| M1 reverse | 0.282 | 13.328 | 15.746 | +2.419 |
| R1 reverse | 3.223 | 45.268 | 31.019 | -14.249 |
| M0 reverse | 0.649 | 10.064 | 10.776 | +0.711 |
| R0 reverse | 2.164 | 18.579 | 12.192 | -6.386 |
| Outside canonical stages | 7.377 | 3.941 | 5.226 | +1.286 |
| Total device time | 17.279 | 128.909 | 131.620 | +2.711 |

The MH-0 column retains only identically named R/M owners; its outside row also
contains the standard-only H/A stages, common geometry, readout, and ZBL. The
MH-1 outside rows contain only common work and force assembly. Profiler device
times are attribution data and are not substituted for the unprofiled medians.

The MLP trace spends `10.115 us/atom` in explicit conditioning. Spline removes
all of it and makes reverse R work substantially cheaper, but its portable
forward contractions give back more: relative to the complete generated MLP R
owners, spline regresses R0 forward by `6.049 us/atom` and R1 forward by
`12.992 us/atom`. Across all four R owners it saves only `1.595 us/atom`.
Small M-stage and common-work differences then turn that saving into a
`2.711 us/atom` device-time regression and a `4.194 us/atom` end-to-end
regression. The next performance target is therefore specialized spline R
owners, led by R1 forward, rather than further work on the already shared
generated M implementation or the cubic interpolation itself.

### HIP BLAS node-stage control

The HIP build enables `KokkosKernels_ENABLE_TPL_ROCBLAS` and links rocBLAS,
but the generated MH-1 M owners do not call it. A temporary control retained
the spline R owners and replaced the complete generated node path with the
existing Kokkos E3-linear `packed_gemm` path. The source switch was removed
after the experiment; it is not a retained execution mode.

The control passed the profile harness scalar validation but regressed from
`130.169 us/atom` to `140.503 us/atom`. Node workspace increased from
`305,309,848 B` to `694,631,832 B`. The matched device trace attributed
`48.503 ms` to the four M stages, versus `39.073 ms` for the generated owner.
The Kokkos control decomposed as follows:

| Device work | Time (ms) | Time (us/atom) |
| --- | ---: | ---: |
| rocBLAS Tensile GEMMs, 70 launches | 16.335 | 18.906 |
| E3-linear pack and unpack | 4.931 | 5.707 |
| Other E3-linear kernels | 6.142 | 7.109 |
| Generic gate kernels | 11.522 | 13.336 |
| Generic product-basis kernels | 9.693 | 11.218 |

This rejects wholesale replacement of the generated M path, not rocBLAS
itself. Correlation of each Tensile launch with its enclosing E3-linear region
identified narrower candidates. Including their measured pack/unpack costs,
rocBLAS reduced the four product-output forward/reverse transforms from about
`3.652 ms` to `1.911 ms`; all four `linear_2` transforms from about `13.257 ms`
to `12.716 ms`; and the four `linear_1` transforms from about `9.751 ms` to
`9.316 ms`. Layer-1 residual forward/reverse also saved about `0.322 ms`.

A retained hybrid must therefore split the generated node schedule only at
selected dense transforms, keep generated gate and product kernels, reuse one
bounded packing workspace, and preserve the Kokkos stream. Product-output is
the first M candidate because it has the largest measured margin. R1 and R0
forward specialization remains higher priority because the current spline R
gap is materially larger than the approximately `3 ms/step` aggregate M-linear
opportunity.

### Channel-contiguous spline R forward

The first retained R specialization removes the device spline adapter's output
transpose and changes the portable receiver owner to consume and produce the
generated node program's channel-contiguous `ir-mul` layout. Each channel now
accumulates one instruction's at-most-seven angular outputs across all incoming
edges before writing them once. This replaces an output read/modify/write for
every edge with one write per instruction, makes source and output channel
accesses coalesced across the wavefront, and bounds the per-thread accumulator
at 16 values. The bound applies only to this device spline owner; it does not
restrict general factorable tensor products. No new runtime selector or
workspace was added.

The matched FP32 HIP workload remained 864 atoms, 78,624 directed edges, a
6.0 A model/effective cutoff, and zero skin. Ten warmups and twenty samples
measured:

| Implementation | Median (us/atom) | Min (us/atom) | Max (us/atom) |
| --- | ---: | ---: | ---: |
| Previous spline direct | 130.169 | 129.799 | 130.682 |
| Channel-contiguous R forward | 116.123 | 115.935 | 116.427 |

This is a `10.79%` end-to-end improvement. The marker trace attributes the
change as follows; unrelated stages remain within run-to-run noise:

| Canonical stage | Previous (us/atom) | Candidate (us/atom) | Change |
| --- | ---: | ---: | ---: |
| R0 forward | 10.937 | 6.535 | -40.25% |
| R1 forward | 27.022 | 17.599 | -34.87% |
| M0 forward | 8.849 | 8.836 | -0.15% |
| M1 forward | 9.852 | 9.784 | -0.69% |
| M1 reverse | 15.746 | 15.792 | +0.29% |
| R1 reverse | 31.019 | 30.947 | -0.23% |
| M0 reverse | 10.776 | 10.608 | -1.55% |
| R0 reverse | 12.192 | 12.236 | +0.36% |

The perturbed 864-atom full-array comparison against generated
`mlp_reference` measured maximum absolute differences of `3.615e-5 eV` for
total energy, `1.937e-6 eV` for atomic energies, `2.726e-5 eV/A` for force
components, and `4.906e-7` for stress. The direct fallback delta remained zero.
The measured extension SHA-256 was
`7536fb481cfa70620b1c3ba6b0486b72dd82aa41a0c1ea68ee47c00a4f17bbcb`.

Partitioning the spline forward owner by instruction was tested and rejected.
On the matched 864-atom workload, adding instruction to the launch grid changed
the median from `100.330 ms/step` (`116.123 us/atom`) to `103.411 ms/step`
(`119.689 us/atom`), a `3.07%` regression. The receiver/channel grid already
supplies enough blocks; the extra dimension repeats incoming-edge traversal and
dynamic spline metadata work. The source was restored to the rank-2
receiver/channel owner, so the rejected implementation is not retained as a
runtime path.

### Shared spline evaluation-point cache

The next accepted R optimization removes repeated interval searches from all
four spline R owners. Previously every channel independently evaluated
`floor((r-x0)/h)` and reconstructed the local spline coordinate for the same
edge, and reverse repeated that work for both source-owned and edge-owned
contractions. R0 now prepares one compact cache per edge containing the
interval index and local double-precision coordinate. R0/R1 forward and reverse
reuse it while still forming the two cubic powers in registers. The cache is
shared by both layers and adds 943,488 bytes for 78,624 directed edges.

On the matched 864-atom FP32 HIP workload, ten warmups and twenty samples
measured `87.157 ms/step` (`100.876 us/atom`), versus the preceding accepted
`100.330 ms/step` (`116.123 us/atom`). This is a `13.13%` end-to-end reduction.
Scalar validation passed, the direct fallback count remained zero, and no
conditioner launches were observed. The selected-region trace changed as
follows:

| Canonical stage | Previous (us/atom) | Point cache (us/atom) | Change |
| --- | ---: | ---: | ---: |
| R0 forward | 6.535 | 3.199 | -51.05% |
| R1 forward | 17.599 | 10.456 | -40.59% |
| R1 reverse | 30.947 | 29.013 | -6.25% |
| R0 reverse | 12.236 | 10.338 | -15.51% |
| Total device time | 115.444 | 100.102 | -13.29% |

The M owners stayed within profiling variation. Relative to the matched MH-0
trace, R1 remains the largest gap: MH-1 spline R1 forward is `10.456 us/atom`
versus `0.768 us/atom`, and reverse is `29.013 us/atom` versus
`3.212 us/atom`. The remaining R work therefore needs a more specialized
contraction schedule; repeated spline interval construction is no longer the
dominant explanation.

The next M experiment remains a single product-output forward/reverse BLAS
handoff. It must reuse bounded packing storage, execute on the Kokkos HIP/CUDA
stream, and include packing, synchronization, and workspace in acceptance. The
complete node-stage rocBLAS replacement remains rejected.

That selective handoff was subsequently tested and rejected. On the matched
864-atom workload it measured `100.573 ms/step` (`116.404 us/atom`) versus the
accepted `100.330 ms/step` (`116.123 us/atom`), a `0.24%` regression within a
very narrow margin but with no demonstrated benefit. Scalar validation passed
and the direct fallback count remained zero. The split added exactly `2 MiB`
of node workspace and increased the traced launch count from 336 to 428.

The trace explains the flat result. Removing the generated product-output
linears saved about `3.41 ms` of generated node-kernel time, but the replacement
added about `1.74 ms` of Tensile GEMMs and `1.52 ms` of E3-linear packing and
unpacking. The R kernels changed by less than `0.06 ms` in aggregate. Thus the
generated kernels and the rocBLAS path perform comparable useful work for this
tile size, while the BLAS interface requires layout conversion. The split code
and its extra workspace were removed. A future BLAS attempt should start by
making product contraction emit the packed matrix layout expected by GEMM;
another ir-mul-to-packed adapter is not justified.

The final perturbed 256-atom full-array comparison against `mlp_reference`
measured maximum absolute differences of `1.190e-5 eV` for total energy,
`2.951e-6 eV` for atomic energies, `3.915e-5 eV/A` for force components, and
`4.157e-7` for stress. The direct fallback delta was zero and generated
conditioning counters were unchanged.

### Precision-matched spline evaluation cache

The retained follow-up stores the cached local spline coordinate and its
squared and cubed powers in the model precision. Distance construction,
interval selection, and cutoff handling remain double precision. Consequently,
the qualified FP32 path evaluates the cubic spline polynomial entirely in
FP32, while an FP64 model continues to use FP64 throughout. For 78,624 directed
edges the cache shrinks from 943,488 to 628,992 bytes.

On the same 864-atom FP32 HIP workload, with a 6.0 A model/effective cutoff,
zero skin, ten warmups, and twenty samples, the median changed from
`87.157 ms/step` (`100.876 us/atom`) to `81.867 ms/step`
(`94.753 us/atom`). This is a `6.07%` reduction over the double-coordinate
cache and an `18.41%` reduction over the preceding channel-contiguous R
implementation. Scalar validation passed and the direct fallback count
remained zero.

The marker-aware single-step traces attribute the direct R-stage changes as
follows. M stages are included to show that the node program did not change.

| Canonical stage | Double cache (us/atom) | Precision cache (us/atom) | Change |
| --- | ---: | ---: | ---: |
| R0 forward | 3.199 | 2.471 | -22.77% |
| M0 forward | 8.806 | 8.700 | -1.20% |
| R1 forward | 10.456 | 9.269 | -11.35% |
| M1 forward | 9.394 | 9.342 | -0.55% |
| M1 reverse | 15.099 | 15.093 | -0.04% |
| R1 reverse | 29.013 | 28.118 | -3.08% |
| M0 reverse | 10.699 | 10.655 | -0.41% |
| R0 reverse | 10.338 | 10.419 | +0.78% |

The `SQ_INSTS_VALU_DP` counter over the six spline forward/reverse launches
fell from 36,953,280 to zero. Grid sizes, workgroup sizes, VGPR counts, and
launch counts were unchanged, so the result isolates arithmetic precision
rather than a scheduling change. The perturbed 256-atom full-array comparison
against generated `mlp_reference` measured maximum absolute differences of
`4.150e-5 eV` for total energy, `2.029e-6 eV` for atomic energies,
`2.690e-5 eV/A` for force components, and `4.740e-7` for stress. The measured
extension SHA-256 was
`f02ba3052971cf13fdcedd225ae8c4c5c7f6b12a0adab54fdd5a42733a4e81a9`.

R1 reverse remains the dominant non-BLAS owner at `28.118 us/atom`. The next R
experiment should therefore split its source-owned and fused edge-owned
contractions in the trace and optimize the larger sub-owner without adding a
second runtime path. The rejected rocBLAS adapter result still applies: any
future M-stage handoff must receive a GEMM-ready layout directly from the
product contraction instead of paying separate packing and unpacking kernels.

### Source-major R reverse contraction

The retained source-reverse owner collapses the launch domain from
`(source, channel, input component)` to `(source, channel)`. A thread now walks
each source's outgoing edge list once, evaluates each instruction's spline
weight once per edge, and accumulates the at-most-16 input angular components
in registers. A compact `(input component, instruction)` offset table gives the
thread direct sparse-term ranges. This removes repeated edge metadata,
evaluation-point, and spline-coefficient loads without atomics. It replaces the
existing source owner and does not add a runtime path.

On the same 864-atom, 78,624-directed-edge FP32 HIP workload, ten warmups and
twenty samples measured `72.474 ms/step` (`83.881 us/atom`), versus
`81.867 ms/step` (`94.753 us/atom`) for the precision-matched cache. This is an
additional `11.48%` end-to-end reduction and a `27.77%` reduction from the
pre-cache channel-contiguous implementation. Scalar validation passed and the
direct fallback count remained zero.

The marker trace shows that the optimization is isolated to R reverse:

| Canonical stage | Previous (us/atom) | Source-major (us/atom) | Change |
| --- | ---: | ---: | ---: |
| R0 forward | 2.471 | 2.521 | +2.04% |
| M0 forward | 8.700 | 8.775 | +0.87% |
| R1 forward | 9.269 | 9.102 | -1.80% |
| M1 forward | 9.342 | 9.364 | +0.23% |
| M1 reverse | 15.093 | 15.016 | -0.51% |
| R1 reverse | 28.118 | 19.564 | -30.42% |
| M0 reverse | 10.655 | 10.672 | +0.16% |
| R0 reverse | 10.419 | 7.188 | -31.01% |

Within R1 reverse, the source-owned kernel fell from `13.413 ms` to
`6.236 ms`; the fused edge-owned kernel remained near `10 ms`. Within R0
reverse, source-owned work fell from `4.415 ms` to `1.369 ms`. The perturbed
256-atom full-array comparison against generated `mlp_reference` measured
maximum absolute differences of `4.150e-5 eV` for total energy,
`2.029e-6 eV` for atomic energies, `2.658e-5 eV/A` for force components, and
`4.747e-7` for stress. The measured extension SHA-256 was
`910340cb07b60f8c8a7e6cd1d8a014ea3ca98f171334100d705800bb176e1980`.

The active staged-direct M owners remain generated RTC node kernels. Although
the extension links rocBLAS and the generic E3-linear selector reports
`packed_gemm`, this graphwide generated path does not launch Tensile kernels.
The earlier selective handoff remains the relevant control: generated node
kernels saved about `3.41 ms`, but rocBLAS plus packing/unpacking added about
`3.26 ms` and increased launches. A future M experiment must therefore change
the product contraction's output layout and hand only adapter-free, sufficiently
large transforms to rocBLAS on the Kokkos stream.

### Shape-adaptive fused edge-reverse teams

The fused edge owner originally used 128 threads for every tensor-product
shape. A uniform 64-thread experiment improved the light four-instruction R0
edge kernel from `4.836 ms` to `2.465 ms`, but regressed the ten-instruction R1
edge kernel from `9.937 ms` to `10.909 ms`. The uniform change still improved
the full call to `81.959 us/atom`, but it left performance on the table and was
not retained.

The accepted schedule uses 64 threads only when the contraction has at most
four instructions and otherwise retains 128 threads. The smaller R0 team halves
per-team reduction scratch and lets each thread process two channels, which is
profitable for the light contraction. R1 has enough contraction work to use all
128 threads and avoids the uniform-64 regression. This is a model-shape policy
inside the existing portable Kokkos owner, not a HIP-specific branch or a new
runtime code path.

The matched ten-warmup, twenty-sample result is `70.549 ms/step`
(`81.654 us/atom`), compared with `72.474 ms/step` (`83.881 us/atom`) for
uniform 128 threads. That is a further `2.66%` reduction. The marker trace
measured R1 reverse at `16.611 ms` (`19.226 us/atom`) and R0 reverse at
`3.854 ms` (`4.460 us/atom`). The perturbed 256-atom full-array comparison
against generated `mlp_reference` measured maximum absolute differences of
`4.150e-5 eV` for total energy, `2.029e-6 eV` for atomic energies,
`2.643e-5 eV/A` for force components, and `4.751e-7` for stress. The direct
fallback delta remained zero. The measured extension SHA-256 was
`1078fb606cc6521359095f315f55dc727ba9f83294cfd94bab8da8ba7b4c03d3`.

Across the retained R experiments, the end-to-end median has moved from
`116.123 us/atom` to `81.654 us/atom`, a `29.68%` reduction. M0/M1 now account
for about half of the selected device time, so the next material step is the
adapter-free product-layout/rocBLAS experiment rather than another generic
BLAS wrapper.

## Shared generated R architecture

The final device milestone aligns MH-1 spline R with optimized MH-0 rather
than retaining a second portable GPU contraction. MH-1 tensor-product paths
are projected once into the same generated sparse-row representation used by
standard R1. The shared renderer now accepts ordered pair types, runtime spline
function offsets, output masks, density fusion, and compact source owners while
preserving the default MH-0 source exactly.

The resulting steady-state architecture is:

- R0/R1 forward: one receiver/channel-persistent generated kernel. It reads
  channel-contiguous `ir-mul` node state, evaluates spline weights on demand,
  accumulates sparse angular rows, and writes messages and density directly.
- M0/M1: the existing generated V4 node program consumes and produces the same
  persistent `ir-mul` layout, with no spline-specific transpose adapter.
- R0/R1 reverse: one source-major generated kernel per interaction. A block
  walks one compact source segment, accumulates source adjoints in shared
  storage, and fuses radial and harmonic contractions into directed edge
  forces. The later generic harmonic/distance force assembly is skipped.
- Runtime: the neutral V5 spline packets and launch-plan schema are resolved by
  the existing CUDA/HIP module loader. Only the local HIP module was executed;
  CUDA remains a compile-portability target, not a qualified runtime result.

During integration, the first device-generated result incorrectly ran both the
new generated R owner and the generic spline owner. The host-generated and
portable-device skip predicates did not include the new device-generated
state. This doubled R work, produced `-5978.381 eV` instead of the reference
energy, and measured about `1762.9 us/atom`. Forward and reverse dispatch now
have explicit `generated_device_spline_*` ownership predicates, so exactly one
R implementation runs. No experimental public mode or retained control branch
was added.

### Final local HIP qualification

The exact-cutoff acceptance workload used the official MH-1 `omat_pbe` model,
FP32, 864-atom wurtzite AlN, a `6.0 A` model/effective cutoff, zero skin, and
78,624 directed edges. Ten warmups and twenty samples measured a median of
`45.504 ms/step` (`52.666 us/atom`), with a `45.252-58.573 ms` sample range.
The retained portable spline result was `81.654 us/atom`, so shared generated R
reduces end-to-end time by `35.50%`. Direct fallback remained zero, and the
measured launch-counter deltas were exactly two forward, two source-reverse,
and two edge-reverse owners, one of each per interaction. The edge workspace
high-water mark fell from 78,624 portable rows to 864 compact source rows.

A perturbed 256-atom, 23,296-directed-edge full-array comparison against the
generated `mlp_reference` control measured maximum absolute differences of
`2.628e-5 eV` for total energy, `1.937e-6 eV` for atomic energies,
`2.648e-5 eV/A` for force components, and `5.320e-7` for stress. The fallback
delta was zero.

The marker-aware one-step `rocprofv3` trace measured `45.929 ms` end to end and
assigned the staged owners as follows:

| Canonical stage | ms/step | us/atom | Device-time share |
| --- | ---: | ---: | ---: |
| R0 forward | 0.522 | 0.604 | 1.16% |
| M0 forward | 7.576 | 8.769 | 16.85% |
| R1 forward | 1.444 | 1.672 | 3.21% |
| M1 forward | 8.161 | 9.445 | 18.15% |
| M1 reverse | 10.992 | 12.722 | 24.45% |
| R1 reverse | 3.828 | 4.431 | 8.52% |
| M0 reverse | 8.534 | 9.877 | 18.98% |
| R0 reverse | 2.119 | 2.452 | 4.71% |
| Outside staged owners | 1.775 | 2.055 | 3.95% |

The aligned R owners now total `9.159 us/atom`, down from `23.686 us/atom` in
the retained portable trace. M stages now dominate the call; further R work
should be driven by targeted occupancy or instruction evidence rather than by
adding another tensor-product implementation.

A second raw-name trace independently confirmed the generated launch shape.
It contained exactly one `spline_r_forward_kernel_0`, one
`spline_r_forward_kernel_1`, one `spline_r_reverse_kernel_1`, and one
`spline_r_reverse_kernel_0`. Their respective times were `0.377`, `0.976`,
`3.219`, and `2.116 ms`; no portable spline contraction kernel appeared. The
rocprof analyzer maps these symbols directly onto the same R0/R1 stage names
used by the host implementation and standard MACE.

### Portability review

The spline R algorithm and data ownership are shared between CUDA and HIP.
Both backends consume the same sparse rows and V5 packets, use the same
receiver/channel forward owner and source-major fused reverse owner, and emit
the same four exported kernel symbols. Backend dialects are limited to runtime
headers, shuffle intrinsics, module APIs, target identity, and launch-plan
metadata. The 256-thread forward and 64-thread reverse launch shapes are also
currently shared.

One numerical policy remains backend-specific: local HIP FP32 artifacts use
FP32 spline coordinates to avoid the measured `gfx1151` FP64 penalty, while
CUDA retains the existing double-coordinate evaluation policy until CUDA
correctness and performance are qualified. CUDA source generation is covered
by the shared source-contract tests, but this milestone does not claim a CUDA
compile or runtime result.

The milestone review tightened packet validation before launch. Nonempty
graphs now require nodes and at least one compact source owner; reverse packets
must use the supported unit-float or Cartesian-double coordinate pairing; and
the source and edge spline descriptors must match exactly. Backend reporting
also requires the module to contain all four spline R symbols before claiming
the staged-device V5 path, so older V4 artifacts correctly report the portable
device fallback.
