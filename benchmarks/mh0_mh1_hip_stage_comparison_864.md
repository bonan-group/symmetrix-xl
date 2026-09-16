# Standard MACE and MH-1 HIP stage comparison

Date: 2026-08-22. Source revision: `589bc78489e5b295df64ba570f3b23a57542c302`
with reporting-only stage-label changes in the worktree. This record compares
standard MACE (the `mace-mh-0` model) with MH-1 using matched local HIP direct
execution on the Radeon 8060S (`gfx1151`). No CUDA execution was performed.

## Workload and execution contract

- Precision: FP32.
- Properties: energy and forces.
- Structure: 864-atom wurtzite AlN, repeated `6 x 6 x 6`.
- Model cutoff: 6.0 A.
- Neighbor-list skin: 0.0 A; effective cutoff: 6.0 A.
- Directed edges: 78,624.
- Graph SHA-256: `a6990939929e747e577bbd8b65c8954a13f712b972e40cad9869f78f54a521b7`.
- Execution: prepared graph, `streamed_edges="direct"`, required hipRTC,
  five warmups, zero fallback evaluations.
- Native extension SHA-256:
  `0791a2eb0fdaf5af519f2d215560e56704d553af2f338ef61a531f1793aaffae`.
- Standard MACE model SHA-256:
  `c202e5bc6da9a2b0ca4974595869fa5f4f0cdb7f37b58d87415ecd95f048eb22`.
- MH-1 model SHA-256:
  `fb1dc908fd0f7b99aa84279dd5ae598dbde62cb15b95c2b076581872b05d6917`.

The GPU was idle before capture: zero reported GPU use, no KFD processes, and
166,252,544 of 8,589,934,592 VRAM bytes in use. Both diagnostic numerical
validations passed.

## Low-overhead timing

These are ten synchronized evaluator-only samples after five warmups. They are
the primary performance results; rocprof timings below are for attribution.

| Model | Median (ms/step) | Median (us/atom) | Sample range (ms) | Relative time |
|---|---:|---:|---:|---:|
| Standard MACE | 14.797 | 17.126 | 14.709-15.045 | 1.00x |
| MH-1 | 109.789 | 127.071 | 109.150-110.208 | 7.42x |

## Per-stage GPU timing

Each trace contains one selected prepared evaluation after five warmups.
`Share` is the fraction of that model's total kernel duration. A dash denotes a
stage that is absent or fused into a differently named MH-1 stage, not measured
zero work.

| Stage | Standard ms | Standard us/atom | Share | MH-1 ms | MH-1 us/atom | Share |
|---|---:|---:|---:|---:|---:|---:|
| Common: geometry/harmonics | 1.088 | 1.260 | 7.25% | 1.360 | 1.575 | 1.22% |
| Common: ZBL | 0.166 | 0.192 | 1.11% | 0.163 | 0.189 | 0.15% |
| Runtime: copy/fill | 0.030 | 0.034 | 0.20% | 0.111 | 0.129 | 0.10% |
| R0 forward | 1.403 | 1.624 | 9.34% | - | - | - |
| M0 forward | 0.281 | 0.325 | 1.87% | - | - | - |
| H1 forward | 0.180 | 0.208 | 1.20% | - | - | - |
| R1 edge conditioning forward | - | - | - | 2.478 | 2.868 | 2.23% |
| R1 forward | 0.634 | 0.734 | 4.22% | 13.903 | 16.092 | 12.49% |
| A1 forward | 1.551 | 1.796 | 10.33% | - | - | - |
| M1 forward | 0.826 | 0.956 | 5.50% | - | - | - |
| M1/node forward | - | - | - | 16.250 | 18.807 | 14.60% |
| H2 forward | 0.348 | 0.403 | 2.32% | - | - | - |
| Readout/reduction | 0.360 | 0.417 | 2.40% | 0.651 | 0.754 | 0.59% |
| H2 reverse | 0.743 | 0.860 | 4.95% | - | - | - |
| M1 reverse | 0.258 | 0.299 | 1.72% | - | - | - |
| M1/node reverse | - | - | - | 19.636 | 22.727 | 17.65% |
| A1 reverse | 1.498 | 1.734 | 9.97% | - | - | - |
| R1 reverse | 2.813 | 3.256 | 18.73% | - | - | - |
| R1 reverse: source | - | - | - | 16.381 | 18.960 | 14.72% |
| R1 reverse: edge (compact fused) | - | - | - | 32.192 | 37.260 | 28.93% |
| R1 edge conditioning reverse | - | - | - | 6.351 | 7.350 | 5.71% |
| H1 reverse | 0.466 | 0.540 | 3.11% | - | - | - |
| M0 reverse | 0.550 | 0.636 | 3.66% | - | - | - |
| R0 reverse | 1.824 | 2.111 | 12.14% | - | - | - |
| Common: force assembly | - | - | - | 1.800 | 2.084 | 1.62% |
| **All device kernels** | **15.019** | **17.383** | **100%** | **111.277** | **128.793** | **100%** |

The standard stages marked absent in MH-1 have not disappeared as mathematical
work one-for-one. The generated MH-1 `M1/node` programs fuse message transforms,
nonlinear products, residuals, gates, density, H2-like work, and readout state.
The MH-1 graph reverse is split into source, compact-edge, and learned edge
conditioning stages, whereas the standard generated R1 reverse is one fused
kernel. Consequently, only the common stages and the grouped families below
are meaningful direct comparisons.

## Comparable stage families

| Family | Standard ms | Standard us/atom | MH-1 ms | MH-1 us/atom | MH-1 minus standard (ms) |
|---|---:|---:|---:|---:|---:|
| R1 graph forward | 0.634 | 0.734 | 16.381 | 18.960 | 15.747 |
| Node forward and readout | 1.534 | 1.775 | 16.901 | 19.561 | 15.367 |
| Node reverse | 1.001 | 1.159 | 19.636 | 22.727 | 18.634 |
| R1 graph reverse | 2.813 | 3.256 | 54.924 | 63.570 | 52.111 |
| Common geometry/ZBL/runtime/assembly | 1.284 | 1.486 | 3.435 | 3.976 | 2.151 |

For MH-1, `R1 graph forward` includes edge conditioning forward, and `R1 graph
reverse` includes source reverse, compact-edge reverse, and edge conditioning
reverse. The standard node families combine M1/H2/readout stages to match the
fused MH-1 node boundary as closely as the two architectures permit.

The R1 graph reverse family accounts for `52.111 ms`, or 54.1%, of the
`96.258 ms` profiled kernel-time gap. Compact-edge reverse alone is the largest
MH-1 stage at `32.192 ms` (`37.260 us/atom`, 28.93% of MH-1 device time).

## Artifacts and reproduction

Local artifacts are under `/tmp/mh0-mh1-stage-comparison/`:

- `mh0/timing.json` and `mh1/timing.json`: low-overhead ten-sample timings.
- `mh0/profile-harness.json` and `mh1/profile-harness.json`: profiled run
  contracts and numerical validation.
- `mh0/stage-profile_kernel_trace.csv` and
  `mh1/stage-profile_kernel_trace.csv`: selected-region traces.
- `mh0/kernel-summary.json` and `mh1/kernel-summary.json`: semantic summaries.

The standard and MH-1 hipRTC cache keys are respectively
`ef93f80238c723720664ebd552bb6f6165481d19d24a7448aba0a2db224d4682` and
`870b71972c9074ea4dad281fedcdf0d761d01c81a2a0b360d9f66e748722ea4f`.
Their HSACO SHA-256 values are respectively
`4df769e20b2a0372355c19e376a497d3ab9c3732b9868dd985aa1bb28b1dbcc2` and
`1a27d6718d54e0eba53260199a0c315f13ee14d97cc5c052b1999c6d3855a5a8`.

Both traces were generated with:

```bash
env LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/core-7.14/lib \
  SYMMETRIX_JIT_CACHE=/tmp/mh0-mh1-stage-comparison/jit-cache \
  SYMMETRIX_JIT_POLICY=required \
  /usr/bin/rocprofv3 --kernel-trace --stats --selected-regions -f csv \
  -d /tmp/mh0-mh1-stage-comparison/ROLE -o stage-profile -- \
  .venv/bin/python benchmarks/symmetrix_mh_rocm_profile.py ROLE MODEL_JSON \
  --warmups 5 --steps 1 \
  --output /tmp/mh0-mh1-stage-comparison/ROLE/profile-harness.json \
  --extension /tmp/symmetrix-mh1-hip-final/symmetrix.cpython-314-x86_64-linux-gnu.so \
  --source-root "$PWD"
```

Use `ROLE=mh0` with `/tmp/mace-mh-0-current-direct-Al-N.json`, and `ROLE=mh1`
with `/tmp/mace-mh-1-current-direct-Al-N.json`. The low-overhead runs use the
same driver with `--steps 10` and without the `rocprofv3` prefix.
