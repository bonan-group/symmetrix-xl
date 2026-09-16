# Generated MACEField versus MH-0 benchmark

## Decision

The merged generated MACEField R1 path is working correctly, but its current
end-to-end direct execution evaluator is 61.9-71.6% slower and uses substantially more GPU
memory than standard MH-0's best admitted path. This is not a regression in the
shared generated R1 kernel. MH-0 additionally admits generated M0 and R0-v2,
while MACEField currently selects runtime M0 and R0-v1.

When both models are restricted to generated R1 plus the same R0-v1/runtime-M0
stages, MACEField is 11.8-13.0% faster and GPU memory is equal within 10 MiB.
The next useful optimization is therefore MACEField M0/R0 admission, not a new
R1 implementation.

## Method

- Commit: `d36bcad` (`Merge commit '98b4f3bd5' into direct-opt`)
- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB, driver 610.43.02
- Extension: Release CUDA/sm_120, SHA-256
  `cc1e100a1753eddc488ddd4cfdce595bf3972e733fae8d3871e141ddf77bc7df`
- Precision and mode: float32, `streamed_edges="direct"`,
  `direct_jit="required"`, retained M1 default
- Systems: periodic wurtzite AlN, 864 and 4,000 atoms
- MACEField input: `mp-dielectric` head, electric field
  `[0.01, -0.02, 0.03]` V/A
- MH-0 input: `omat_pbe` head, ordinary energy/force evaluation
- Sampling: 3 fresh processes per model/profile/size, alternating model order;
  10 warmups and 30 synchronized evaluator samples per process
- Timing excludes checkpoint extraction, JSON parsing, calculator setup, graph
  construction, and generated-artifact setup.

Both current-contract V2 exports have 128 channels and the identical R1
generation fingerprint `sha256:1d1a9f8c...f4073b36`. MACEField retains 79
elements and one field coupling; MH-0 retains 89 elements and no field
coupling. The comparison measures deployed model performance, not model
accuracy or identical learned weights.

## Results

| Profile | Atoms | MACEField median | MH-0 median | MACEField delta | MACEField GPU | MH-0 GPU |
|---|---:|---:|---:|---:|---:|---:|
| optimized | 864 | 6.943 ms | 4.046 ms | +71.6% | 1,630 MiB | 1,080 MiB |
| optimized | 4,000 | 27.582 ms | 17.034 ms | +61.9% | 4,500 MiB | 1,934 MiB |
| matched R1 | 864 | 6.951 ms | 7.877 ms | -11.8% | 1,630 MiB | 1,636 MiB |
| matched R1 | 4,000 | 27.632 ms | 31.765 ms | -13.0% | 4,500 MiB | 4,490 MiB |

The `optimized` profile uses each model's automatically admitted stages:

| Model | R1 forward/reverse | R0 | M0 |
|---|---|---|---|
| MACEField | generated-all / generated | v1 | runtime |
| MH-0 | generated-all / generated | v2-edge16 | generated |

The `matched R1` profile forces R0-v1 and runtime-M0 for both models. Compared
with that control, MH-0's extra generated M0/R0-v2 coverage reduces evaluator
time by 48.6% at 864 atoms and 46.4% at 4,000 atoms. MACEField is unchanged
within 0.2% because the matched stages are already its automatic selections.

## Generated-path validation

All 24 fresh-process records report:

- static generated R1 admission;
- `generated_all` forward and `generated` reverse selection;
- 30 generated forward and 60 generated reverse launches for 30 measured calls;
- zero direct execution fallback evaluations;
- zero retained R0 or R1 edge-radial storage.

Raw records and worker logs are under
`benchmarks/.artifacts/macefield_mh0_generated/paired-generated.json` and its
adjacent `paired-generated-logs/` directory. The reproducible driver is
`benchmarks/macefield_mh0_generated_benchmark.py`.

## Compatibility scope

Model schema V1 and V2 support remains required. The generated direct execution contract,
plugin ABI, cache identity, and generated artifacts are pre-production and may
change incompatibly; models and artifacts should be re-extracted/regenerated
with the active Symmetrix version after such a change.

## Phase 22 generated M0/R0-v2 update

MACEField now admits the existing structural generated M0 and exact R0-v2
executors in compatible float32 CUDA `direct` evaluations. Runtime M0 and
R0-v1 remain independent controls; the global `all` default is unchanged.

Three fresh processes per profile and size, with 10 warmups and 30 samples,
produced the following MACEField medians:

| Atoms | retained | M0 only | R0-v2 only | combined | combined gain |
|---:|---:|---:|---:|---:|---:|
| 864 | 6.997 ms | 5.789 ms | 6.402 ms | 5.176 ms | 26.0% |
| 4,000 | 27.795 ms | 23.025 ms | 25.918 ms | 21.175 ms | 23.8% |
| 6,912 | 47.385 ms | 39.577 ms | 44.570 ms | 36.623 ms | 22.7% |
| 32,000 | 226.855 ms | 184.601 ms | 214.068 ms | 171.345 ms | 24.5% |

At 864 atoms, generated M0 reduces both active 288,866,304-byte polynomial
tensors to zero and lowers sampled GPU process memory by about 556 MiB. Every
paired record passed exact launch and zero-fallback assertions. Direct,
prepared, response, deterministic, executor-switch, and retained rollback
tests pass without relaxed production tolerances.

The optimized n31 capacity probe still fails on a subsequent 2.273 GiB device
allocation. M0 and R0 storage are already eliminated there; the remaining
boundary includes retained field-coupled M1, whose separate recomputation
prototype remains out of scope. Nsight Systems confirms the expected generated
launch sequence. Nsight Compute counters were unavailable due to
`ERR_NVGPUCTRPERM`; cubin metadata reports zero local memory for generated M0
and R0-v2, with 114/254 registers for M0 forward/reverse and 38-48 registers
for R0-v2 kernels.

Raw Phase 22 records are in
`benchmarks/.artifacts/macefield_mh0_generated/phase22-paired.json`; the Systems
trace and compiled resource report are adjacent.
