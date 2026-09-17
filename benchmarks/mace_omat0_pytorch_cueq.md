# MACE-OMAT-0 PyTorch/cuEquivariance MD comparison

## Summary

Symmetrix direct execution was compared with MACE-OMAT-0 medium in 20-step NVE molecular
dynamics trajectories on a local AMD Ryzen 9 9950X3D2 and RTX 5090. GPU
reference runs enabled cuEquivariance. cuEquivariance has no CPU backend, so
CPU reference runs use PyTorch/e3nn and are labelled separately.

The primary metric is median microseconds per atom for one complete 1 fs MD
step. All six comparisons passed trajectory equivalence qualification.

| Device | Threads | Atoms | Symmetrix (us/atom) | Reference (us/atom) | Speedup | Reference |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| CPU | 1 | 256 | 2119.02 | 12129.17 | 5.724x | PyTorch/e3nn |
| CPU | 1 | 2048 | 2090.69 | 12642.92 | 6.047x | PyTorch/e3nn |
| CPU | 16 | 256 | 224.79 | 4554.77 | 20.262x | PyTorch/e3nn |
| CPU | 16 | 2048 | 205.51 | 4743.36 | 23.081x | PyTorch/e3nn |
| RTX 5090 | - | 256 | 66.33 | 93.11 | 1.404x | PyTorch+cuEquivariance |
| RTX 5090 | - | 2048 | 36.62 | 32.56 | 0.889x | PyTorch+cuEquivariance |

At 2048 atoms on the RTX 5090, a speedup below one means the PyTorch+
cuEquivariance reference is about 1.125x faster for this end-to-end MD step.

### Idle-GPU retest and regression check

The GPU comparison was repeated three times per size after the initial report.
Every backend ran in a separate process, and every process launch required two
consecutive `nvidia-smi` samples with zero utilization and no compute-process
PIDs. Each accepted preflight showed 66 MiB device memory, P8, and 0%
utilization.

A new private registry at
`benchmarks/.artifacts/omat0_md_gpu_retest/direct-jit-fresh` was empty before the
campaign and had mode `0700`. The first accepted Symmetrix run built exactly one
NVRTC artifact in that registry; subsequent runs reported `cached` and used the
same registry.

| Atoms | Backend | Median of trial medians (us/atom) | Trial range (us/atom) |
| ---: | --- | ---: | ---: |
| 256 | Symmetrix direct R1 execution | 66.180 | 63.557-68.630 |
| 256 | PyTorch+cuEquivariance | 92.874 | 89.628-95.812 |
| 2048 | Symmetrix direct R1 execution | 36.756 | 36.732-38.452 |
| 2048 | PyTorch+cuEquivariance | 33.662 | 33.027-35.715 |

The crossover is reproducible at the complete MD-step boundary: direct execution is
1.354x-1.461x faster at 256 atoms, while cuEquivariance is 1.077x-1.113x faster
at 2048 atoms.

To test whether this was a generated-kernel regression, the prepared execution
paths were measured separately with 10 warm-ups and 20 timed evaluations. Both
drivers used the same periodic AlN graphs (23,296 edges at 256 atoms and
186,368 at 2048 atoms). Symmetrix timed `_compute_prepared_direct`; upstream
MACE timed model forwards on its prebuilt batch.

| Atoms | Prepared direct R1 execution (us/atom) | Prepared cuEquivariance (us/atom) | direct execution speedup |
| ---: | ---: | ---: | ---: |
| 256 | 6.670 | 47.873 | 7.178x |
| 2048 | 4.813 | 7.341 | 1.525x |

The prepared direct execution path reported NVRTC artifact
`jit-r1-f32-5a92711776c5960f`, 60 measured generated R1 launches, and zero
fallback evaluations at each size. It was built from a second initially empty
private registry, then loaded as `cached` for 2048 atoms. This result does not
support a generated direct R1 execution regression: direct execution remains faster than
cuEquivariance at the model-execution boundary, including at 2048 atoms. The
end-to-end crossover is therefore in work outside the prepared evaluator, such
as rebuilding and transferring the changing neighbor graph through the ASE
calculator boundary. The prepared diagnostic is not substituted for the
primary MD metric.

A follow-up workload profile attributed the gap to neighbor/input construction,
order-sensitive direct execution schedule rebuilding, and host-side result
reduction rather than a generated CUDA kernel regression.

### Optimized MD result

After reusing a deterministic skin topology and prepared direct execution schedule,
reducing atom forces on the device, and skipping unrequested stress, the
first optimized 2048-atom Symmetrix result was 15.899 us/atom. Native
retained-topology geometry construction and cached topology-derived type arrays
then reduced the independently qualified result to **7.951 us/atom**. The same
saved PyTorch+cuEquivariance reference is 32.557 us/atom, making final
Symmetrix **4.095x faster** at the complete MD-step boundary. The original
Symmetrix result was 36.619 us/atom, so the final implementation reduces step
time by 78.3%.

The optimized 20-step trajectory independently passed the same qualification
limits: 2.768e-5 eV/atom energy error, 1.231e-4 eV/A maximum force error, and
2.485e-6 A maximum position error. The final records, fresh registries, and
profile traces are in `benchmarks/.artifacts/omat0_md_native/`. The original
six-case table above is retained as the pre-optimization benchmark baseline;
only the profiled 2048-atom GPU case has been rerun after these changes.

## MD protocol

- Structure: wurtzite AlN repeated to 256 or 2048 atoms, with deterministic
  0.01 A Cartesian position noise.
- Velocities: ASE `MaxwellBoltzmannDistribution` at 300 K using seed 20260808,
  followed by `Stationary(..., preserve_temperature=True)` to remove center-of-
  mass translation.
- Ensemble and integrator: NVE, ASE `VelocityVerlet`, 1 fs timestep, 20 steps.
- Precision: float32 for both implementations.
- Timing: calculator construction and one initial force/energy evaluation are
  recorded separately and excluded. That initial evaluation performs model
  warm-up, first graph construction, and direct execution JIT build/cache loading. Each
  measured sample is one complete `VelocityVerlet.run(1)` step, including its
  calculator evaluation, graph construction, synchronization, and ASE state
  update.
- Statistic: median of the 20 individual MD-step wall times. `us/atom` is
  `1000 * median_step_ms / atom_count`.

The initial evaluation plus 20 integrated steps gives 21 calculator
evaluations per backend trajectory. Timing a single raw `Calculator.calculate`
call is intentionally not used for the performance result.

## Threading

CPU thread count is an explicit case parameter. The driver sets
`torch.set_num_threads()` for PyTorch and sets `OMP_NUM_THREADS` plus
`KOKKOS_NUM_THREADS` for Symmetrix. The records confirm:

| Requested threads | PyTorch `get_num_threads()` | `OMP_NUM_THREADS` | `KOKKOS_NUM_THREADS` |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 1 | 1 for Symmetrix |
| 16 | 16 | 16 | 16 for Symmetrix |

Thus the 16-thread PyTorch/e3nn results are threaded; they are not serial
results with only an environment label.

## direct execution JIT

The comparison command rejects Symmetrix records unless they report all of:

- `implementation=symmetrix-direct`
- `streamed_edges=direct`
- `direct_jit_status=built|cached`

The 256-atom GPU record reports `built`, meaning it generated the CUDA direct R1 execution
artifact. The 2048-atom record reports `cached`, meaning it reused that artifact.
Both report the Kokkos execution space `Cuda`.

New records also capture the JIT compiler backend, artifact ID, variant ID, and
edge policy. For this ordinary MACE-OMAT-0 registry miss, the generated CUDA
identity is NVRTC artifact `jit-r1-f32-5a92711776c5960f`; a separate static
variant ID and edge policy are not assigned.

## Trajectory equivalence

Qualification compares every step, not only the initial or final state. Limits
are `2e-4 eV/atom` maximum potential-energy error, `5e-3 eV/A` maximum force-
component error, and `2e-4 A` maximum position error across all 20 steps.

| Device | Threads | Atoms | Energy (eV/atom) | Force (eV/A) | Position (A) |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 1 | 256 | 4.950e-6 | 9.104e-5 | 1.939e-6 |
| CPU | 1 | 2048 | 2.795e-5 | 1.263e-4 | 2.380e-6 |
| CPU | 16 | 256 | 5.904e-6 | 9.271e-5 | 1.933e-6 |
| CPU | 16 | 2048 | 2.795e-5 | 1.249e-4 | 2.378e-6 |
| RTX 5090 | - | 256 | 4.848e-6 | 1.017e-4 | 1.939e-6 |
| RTX 5090 | - | 2048 | 2.772e-5 | 1.232e-4 | 2.485e-6 |

## Environment

- Repository revision: `bb8e8e1651e6423fa2d34b4d9576aaf5bb29d78c`
- CPU: AMD Ryzen 9 9950X3D2, 16 cores/32 threads
- GPU: NVIDIA GeForce RTX 5090, compute capability 12.0, 32,607 MiB
- NVIDIA driver 610.43.02; CUDA toolkit `/usr/local/cuda-13.3`
- Python 3.12.13, PyTorch 2.13.0, MACE 0.3.16
- cuequivariance 0.11.0 and cuequivariance-torch 0.11.1
- Checkpoint SHA256: `d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a`
- Compact model SHA256: `55ed1b6f689eb92f77fb7fd10a4eb11008f4520934e63c0b0cf5cd5c7b111100`

## Reproduction

Use separate fresh processes for each backend. CPU example:

```bash
python benchmarks/mace_omat0_equivalence_benchmark.py run \
  --backend symmetrix --device cpu --threads 16 \
  --checkpoint "$CHECKPOINT" --compact-model "$COMPACT_MODEL" \
  --repeat 4 --output symmetrix.json

python benchmarks/mace_omat0_equivalence_benchmark.py run \
  --backend torch --device cpu --threads 16 \
  --checkpoint "$CHECKPOINT" --compact-model "$COMPACT_MODEL" \
  --repeat 4 --output reference.json

python benchmarks/mace_omat0_equivalence_benchmark.py compare \
  --symmetrix symmetrix.json --reference reference.json \
  --output comparison.json
```

Use `--device cuda` with the CUDA environments; the reference then enables
cuEquivariance. Repeat 4 gives 256 atoms and repeat 8 gives 2048 atoms. Raw MD
records are retained under `benchmarks/.artifacts/omat0_md_equivalence/`.
The idle-GPU trials and their fresh registry are under
`benchmarks/.artifacts/omat0_md_gpu_retest/`; prepared-path diagnostic records
and their independent fresh registry are under
`benchmarks/.artifacts/omat0_prepared_gpu_retest/`.
