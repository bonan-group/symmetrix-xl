# MH-0 versus MH-1 5,000-atom SrTiO3 CUDA profile

## Conclusion

Moving MH-1 retained-edge geometry construction and atom-force reduction to
Kokkos device kernels reduces the controlled 5,000-atom SrTiO3 calculator
evaluation from 254.378 to 149.934 ms, a 1.70x speedup. Energy and force
qualification is bitwise identical to the baseline; positions differ by at
most 2.22e-16 Angstrom.

The optimized Nsight run spends 151.459 ms/evaluation in GPU kernels under
profiling and 153.096 ms in the synchronized calculator call. Its 1.637 ms
subtraction residual is 63.8 times smaller than the 104.441 ms baseline
residual. Steady-state transfers are now 120,000 bytes of positions H2D,
40,000 bytes of node energies D2H, and 120,000 bytes of atom forces D2H. The
12,286,224-byte pair-force D2H transfer and the 12,286,224- plus
4,095,408-byte retained-edge geometry H2D transfers no longer occur in a
measured step.

For comparison, MH-0 takes 20.953 ms per calculator evaluation and spends
19.236 ms in GPU kernels. Its 1.717 ms subtraction residual remains similar to
the optimized MH-1 residual. MH-1 kernel optimization is still needed because
its device work remains approximately 7.9 times MH-0.

The residual is defined only as synchronized calculator wall time minus the
sum of GPU kernel durations. It is not a measured CPU phase and must not be
called CPU time. Nsight Systems shows that the MH-1 device is idle for 113.120
and 103.881 ms before the two measured kernel clusters. Transfers in those
intervals take only 1.709 and 1.389 ms. The trace therefore localizes most of
the MH-1 residual to host-side work between evaluations, but the current MH-1
trace has no NVTX ranges that can divide that work exactly among force-result
materialization, ASE integration, cached-graph geometry construction, and
input conversion.

The previously quoted MH-1 kernel time of 151.534 ms/evaluation included the
slower warm-up cluster. Consequently, the derived 102.845 ms residual and
59.9x ratio are superseded. The controlled comparison below uses only the two
measured evaluation clusters.

## Device stress extension

The device-boundary optimization now also covers stress. When energy, forces,
and stress are requested together, MH-1 keeps both retained-edge vectors and
pair forces on CUDA and reduces
`stress_ab = -sum_e(force_e,a * vector_e,b) / volume` there. Python receives
only the nine-element tensor before applying the existing ASE Voigt conversion.

This qualification uses the same deterministic 5,000-atom state, float32 MH-1
model, 6.0 Angstrom cutoff, 0.5 Angstrom skin, 6.5 Angstrom effective cutoff,
and 511,926 retained directed edges as the energy/forces run. The current
stress-enabled native extension SHA-256 is
`c087a23326b117925d6b0bd44451170416af9994c82ceb35582f776f4a344c01`.

| Component | Energy/forces only | Energy/forces/stress | Change |
| --- | ---: | ---: | ---: |
| Unprofiled synchronized calculator evaluation | 149.934 ms | 155.877 ms | +5.943 ms |
| Profiled synchronized calculator evaluation | 153.096 ms | 156.667 ms | +3.571 ms |
| Profiled GPU kernels | 151.459 ms | 154.990 ms | +3.531 ms |
| Profiled subtraction residual | 1.637 ms | 1.676 ms | +0.039 ms |
| Complete VelocityVerlet step, unprofiled | 150.533 ms | 156.489 ms | +5.956 ms |

These are separate two-sample runs, so their differences include normal kernel
timing variation. The added virial work is nine reductions per evaluation and
totals only 0.071 ms in the first profiled stress step. The unprofiled stress
calculator samples are 155.958 and 155.797 ms. The profiled samples are
156.532 and 156.801 ms; corresponding kernel sums are 154.655 and 155.325 ms.
The two complete profiled step ranges are 157.131 and 157.155 ms, with largest
internal kernel gaps of 0.364 and 0.350 ms.

The steady-state stress transfer contract is:

| Direction | Bytes per step | Array |
| --- | ---: | --- |
| Host to device | 120,000 | 5,000 current positions in float64 |
| Device to host | 40,000 | 5,000 node energies in float64 |
| Device to host | 120,000 | 5,000 atom forces in float64 |
| Device to host | 72 | Reduced 3 by 3 stress tensor in float64 |
| Device to host | 4 | Geometry-validity flag |

Neither measured stress step contains a 12,286,224-byte pair-force D2H copy,
a 12,286,224-byte retained-edge-vector H2D copy, nor a 4,095,408-byte distance
H2D copy. The one process-wide 12,286,224-byte H2D transfer is still the
reference-edge initialization before warm-up and is not repeated while the
cached graph remains valid.

For numerical qualification, the compact device stress was compared to the
previous host expression `(-pair_forces.T @ xyz) / volume` from the same two
evaluations. The maximum absolute difference is
`5.551115123125783e-17 eV/Angstrom^3`, and the RMS difference is
`2.0428069205511727e-17 eV/Angstrom^3`. Profiled and unprofiled energy and
forces are bitwise identical; their maximum stress and position differences
are `3.3881317890172014e-21 eV/Angstrom^3` and
`8.881784197001252e-16 Angstrom`, respectively.

## Device-boundary optimization result

The optimized run uses the same 5,000-atom state, 6.0 Angstrom model cutoff,
0.5 Angstrom skin, 6.5 Angstrom effective neighbor-list cutoff, and 511,926
retained directed edges as the baseline. The optimized native extension
SHA-256 is
`454e7217f8c1e9c2726194300c4015407c08a0a21ada7a8460b6d19579c560d9`.
The generated JIT artifact remains `jit-mh1-v4-86823d33d34d12b5` with the
same generated variant and edge policy as the baseline.

| Component | MH-1 baseline | MH-1 optimized | Change |
| --- | ---: | ---: | ---: |
| Unprofiled synchronized calculator evaluation | 254.378 ms | 149.934 ms | 1.70x faster |
| Profiled synchronized calculator evaluation | 254.378 ms | 153.096 ms | 1.66x faster |
| Profiled GPU kernels | 149.937 ms | 151.459 ms | 1.01x slower |
| Profiled subtraction residual | 104.441 ms | 1.637 ms | 63.8x smaller |
| Complete VelocityVerlet step, unprofiled | 255.015 ms | 150.533 ms | 1.69x faster |

The small kernel-time difference is between separate Nsight runs and includes
the two new geometry kernels plus atom-force reduction. It is not evidence of
a regression in the generated model kernels. The unprofiled calculator samples
are 150.280 and 149.588 ms; the profiled samples are 153.134 and 153.058 ms.

The measured-step transfer contract is:

| Direction | Bytes per step | Array |
| --- | ---: | --- |
| Host to device | 120,000 | 5,000 current positions in float64 |
| Device to host | 40,000 | 5,000 node energies in float64 |
| Device to host | 120,000 | 5,000 atom forces in float64 |
| Device to host | 4 | Geometry-validity flag |

The only process-wide 12,286,224-byte H2D copy occurs before warm-up when the
retained reference geometry is prepared. It is outside both measured NVTX
steps and is not repeated while the neighbor cache remains valid. Each step
contains 106 kernels; their mean summed duration is 151.459 ms, and the
largest internal kernel gap is 0.172 ms.

## Workload identity

- Source base revision: `35202488cffa3da536df97b234fbf0234f3891bc`.
  The optimized binary also contains the device-boundary working-tree changes
  described in this report and is identified exactly by its SHA-256 above.
- Device: NVIDIA GeForce RTX 5090, CUDA execution, float32.
- State: deterministic `mlmd-medium-5000` SrTiO3, 5,000 atoms, state identity
  SHA-256 `3e581ace7edf6c2d974deb9107274da13ac723e6f1c72e88e2ca2980f6370d34`.
- Model cutoff: 6.0 Angstrom. The calculator's 0.5 Angstrom neighbor skin
  gives a 6.5 Angstrom effective neighbor-list cutoff and 511,926 retained
  directed candidates; the initial exact-cutoff graph has 370,000 directed
  edges.
- Properties: energy and forces. Stress was not requested or measured.
- Scenario: two 1 fs ASE VelocityVerlet steps after warm-up.
- Execution: `streamed_edges="factorized"`, NVRTC JIT, automatic launch
  policy with unresolved launch tuning.
- Baseline MH-0/MH-1 native CUDA extension SHA-256:
  `d6e6b30ad88470c131f79fdd39197892e0fb16fab3db632cc238c00e8a69391b`
  for both models.
- MH-0 checkpoint SHA-256:
  `d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a`;
  compact model SHA-256:
  `af3b41b1e0de93dc1b8ba946f71ae3cb84c16b7c7a48840529cddd32382380a7`;
  JIT artifact `jit-r1-f32-6fdf3e63624a0e45`.
- MH-1 checkpoint SHA-256:
  `a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47`;
  compact model SHA-256:
  `c7bc71ce6058e96a43e5606051e675a91069c2720b93b5da0a058f6cc002c3fb`;
  JIT artifact `jit-mh1-v4-86823d33d34d12b5`.

## Controlled timing comparison

Calculator timings are medians of the same two measured evaluations. Kernel
time is the mean of the corresponding two Nsight kernel sums; it is used for
the subtraction residual because a median of two samples is their mean.

| Component | MH-0 ms/evaluation | MH-1 ms/evaluation | MH-1 / MH-0 |
| --- | ---: | ---: | ---: |
| Synchronized calculator evaluation | 20.953 | 254.378 | 12.1x |
| GPU kernels | 19.236 | 149.937 | 7.8x |
| Non-kernel subtraction residual | **1.717** | **104.441** | **60.8x** |
| Complete VelocityVerlet step | 21.545 | 255.015 | 11.8x |

The MH-0 per-step trace is:

| Step | Complete step | GPU kernels | Memcpy | Memset |
| --- | ---: | ---: | ---: | ---: |
| 1 | 21.720 ms | 19.257 ms | 0.010 ms | 0.012 ms |
| 2 | 21.380 ms | 19.214 ms | 0.009 ms | 0.012 ms |

The MH-0 calculator samples were 21.002 and 20.904 ms. The complete-step
samples were 21.715 and 21.376 ms; the small difference from the trace ranges
above is timestamp-boundary and rounding noise.

The MH-0 worker JSON reports `neighbor_skin_A: 0.0`, but that field was written
as a hard-coded workload value rather than read from the calculator. The worker
did not override the calculator default of 0.5 Angstrom. The 6.5 Angstrom
candidate count was independently recomputed from the recorded state and also
matches the 511,926-element transfer extents in the traces. This report records
the effective runtime configuration and supersedes that JSON metadata field.

## GPU kernel decomposition

MH-0's principal kernel groups per evaluation are:

| Kernel group | ms/evaluation |
| --- | ---: |
| R1 reverse | 3.955 |
| M1 reverse | 2.245 |
| R0 reverse | 1.912 |
| R0 forward | 1.806 |
| R1 forward | 1.590 |
| A1 forward | 1.238 |
| A1 reverse | 1.046 |
| H2 reverse | 1.028 |
| Remaining stages | 4.417 |

MH-1's two measured GPU clusters span 150.353 and 150.025 ms and contain
150.098 and 149.776 ms of kernels. Internal device gaps are therefore only
0.255 and 0.248 ms: after an MH-1 evaluation reaches the GPU, the device is
almost continuously occupied. Kernel optimization remains necessary because
149.937 ms is 58.9% of calculator wall time, but it does not explain the
additional approximately 104 ms wall-time residual.

## What the MH-1 residual contains

The trace has this measured sequence at each evaluation boundary:

```text
previous MH-1 kernel cluster
  -> device-to-host result copies
  -> long GPU-idle host interval
  -> host-to-device edge geometry copies
  -> next MH-1 kernel cluster
```

| Boundary before measured evaluation | GPU-idle gap | Copies in gap | Copy time | CUDA runtime API time |
| --- | ---: | ---: | ---: | ---: |
| 1 | 113.120 ms | 28,707,856 bytes in 4 copies | 1.709 ms | 1.935 ms |
| 2 | 103.881 ms | 28,707,856 bytes in 4 copies | 1.389 ms | 1.596 ms |

Each boundary contains these four transfers:

| Direction | Bytes | Inferred array |
| --- | ---: | --- |
| Device to host | 40,000 | 5,000 node energies in float64 |
| Device to host | 12,286,224 | 511,926 by 3 pair-force values in float64 |
| Host to device | 12,286,224 | 511,926 by 3 edge-vector values in float64 |
| Host to device | 4,095,408 | 511,926 edge distances in float64 |

The inference follows exactly from the array extents and the calculator source.
The MH-1 `MACENonlinearKokkos` Python binding does not expose
`_reduce_atom_forces`, `_prepare_factorized_geometry`, or
`_compute_prepared_factorized_positions`. Their absence forces two Python
fallbacks:

1. `_collect_mace_results` materializes the full pair-force array and performs
   six NumPy `bincount` reductions instead of reducing to 5,000 atom forces on
   the device.
2. `_cached_neighbor_geometry` constructs `xyz` from cached reference geometry
   and atomic displacements, computes `np.linalg.norm` over every retained edge,
   and clamps inactive skin candidates. `_compute_mace` converts those arrays
   to contiguous float64 input and passes all edge vectors and distances to the
   prepared factorized evaluator.

The ordinary MH-0 `MACEKokkos` binding exposes device atom-force reduction and
native prepared-position geometry. Its measured steady-state boundary copies
only 120,000 bytes of current positions to the device and returns 40,000 bytes
of node energies plus 120,000 bytes of atom forces. The binding difference
therefore explains why the large MH-1 pair-force and edge-geometry transfers
exist. It also identifies the host routines that can occupy the long gap,
although it does not measure their individual durations.

The copies themselves are not the problem: they consume at most 1.7 ms of a
104-113 ms inter-cluster interval. Subtracting copy durations from a GPU-idle
span does not produce a precise CPU-phase measurement, but it establishes that
roughly 102-111 ms per boundary is neither GPU kernels nor GPU copy execution.
Source and binding analysis make host pair-force handling and edge-geometry
preparation the leading candidates. Exact shares require a repeat trace with
NVTX ranges around `_mace_inputs`, `_cached_neighbor_geometry`, `_compute_mace`,
`_collect_mace_results`, and the ASE integrator boundary, or a CPU sampling
profile correlated to those ranges.

CUDA runtime API duration is reported only as diagnostic evidence. Runtime API
calls can overlap GPU work and must not be added to kernel or copy durations to
construct an end-to-end total.

## Artifacts

- MH-0 machine-readable result:
  `benchmarks/.artifacts/mh0_omat_cuda_plan_profile_20260812/n5000-worker.json`.
- MH-0 Nsight trace:
  `benchmarks/.artifacts/mh0_omat_cuda_plan_profile_20260812/n5000-baseline.nsys-rep`.
- MH-0 exported trace database:
  `benchmarks/.artifacts/mh0_omat_cuda_plan_profile_20260812/n5000-baseline.sqlite`.
- MH-0 qualification arrays:
  `benchmarks/.artifacts/mh0_omat_cuda_plan_profile_20260812/n5000-qualification.npz`.
- MH-1 machine-readable result:
  `benchmarks/.artifacts/mh1_omat_cuda_plan_profile_20260812/n5000-worker.json`.
- MH-1 Nsight trace:
  `benchmarks/.artifacts/mh1_omat_cuda_plan_profile_20260812/n5000-baseline.nsys-rep`.
- MH-1 exported trace database:
  `benchmarks/.artifacts/mh1_omat_cuda_plan_profile_20260812/n5000-baseline.sqlite`.
- Optimized MH-1 machine-readable result:
  `benchmarks/.artifacts/mh1_device_boundary_optimized_20260813/n5000-worker.json`.
- Optimized MH-1 Nsight trace:
  `benchmarks/.artifacts/mh1_device_boundary_optimized_20260813/n5000-optimized.nsys-rep`.
- Optimized MH-1 exported trace database:
  `benchmarks/.artifacts/mh1_device_boundary_optimized_20260813/n5000-optimized.sqlite`.
- Optimized MH-1 qualification arrays:
  `benchmarks/.artifacts/mh1_device_boundary_optimized_20260813/n5000-qualification.npz`.
- Stress-enabled MH-1 machine-readable result:
  `benchmarks/.artifacts/mh1_device_boundary_stress_optimized_20260813/n5000-worker.json`.
- Stress-enabled MH-1 legacy-formula comparison:
  `benchmarks/.artifacts/mh1_device_boundary_stress_optimized_20260813/n5000-stress-reference.json`.
- Stress-enabled MH-1 Nsight trace:
  `benchmarks/.artifacts/mh1_device_boundary_stress_optimized_20260813/n5000-stress-optimized.nsys-rep`.
- Stress-enabled MH-1 exported trace database:
  `benchmarks/.artifacts/mh1_device_boundary_stress_optimized_20260813/n5000-stress-optimized.sqlite`.
- Stress-enabled MH-1 profiled machine-readable result and qualification arrays:
  `benchmarks/.artifacts/mh1_device_boundary_stress_optimized_20260813/n5000-profile-worker.json`
  and
  `benchmarks/.artifacts/mh1_device_boundary_stress_optimized_20260813/n5000-profile-qualification.npz`.

The `.artifacts` files are local raw evidence and are intentionally ignored by
Git. This report is the tracked record of the controlled result.
