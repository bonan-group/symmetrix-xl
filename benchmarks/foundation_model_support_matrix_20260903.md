# MACE Foundation direct-execution support matrix

This record qualifies representative released MACE Foundation checkpoints on
source base commit `a8cb0f6f7e11c383db697481fe55d1c6a725cf2b` plus the uncommitted fixes
described below. Native binary and normalized-result hashes identify the exact
matrix builds. This is execution evidence, not a claim that checkpoints
sharing a similar architecture have been tested.

## Method

- CPU: fresh native OpenMP backend, one physical core, one Kokkos/OpenMP
  thread, and one BLAS thread.
- NVIDIA GPU: fresh CUDA 13.3 SM120 backend on an RTX 5090.
- AMD GPU: fresh ROCm 7.14 `gfx1151` backend on an 8 GiB Radeon 8060S.
- Structures: every checkpoint uses a shaken 64-atom conventional diamond C
  `2x2x2` cell. Inorganic and multihead checkpoints additionally use a shaken
  40-atom cubic SrTiO3 `2x2x2` cell; OFF23 instead uses an isolated shaken
  three-atom H2O molecule because Sr and Ti are outside OFF23's chemistry.
  Every structure uses seed `20260903` and Gaussian Cartesian displacements
  with standard deviation `0.03 A`.
- Properties: energy, forces, and ASE stress.
- Precisions: FP32 and FP64.
- Execution: `streamed_edges="direct"`, `execution_profile="speed"`, and
  `SYMMETRIX_JIT_POLICY=required`.
- Reference: the generic executor evaluated the identical compact model and
  structure in the same worker. Each model, structure, precision, and backend
  combination ran in a fresh process.

The direct gate requires a generated artifact, observed generated forward and
reverse launches, the expected direct executors, zero fallback evaluations,
finite outputs, and errors below the manifest tolerances. Worker teardown is
part of the gate: a nonzero process exit fails the case even if it wrote a
result first.

The C and SrTiO3 graphs contain 12,160 and 3,998 directed edges respectively
for the 6.0 A models with a 0.5 A skin. MP-0b2 large uses a 5.0 A cutoff and
contains 7,808 and 2,144 directed edges. OFF23 uses a 4.5 A cutoff and its H2O
graph contains six directed edges. Effective neighbor-list cutoffs are 6.5 A,
5.5 A, and 5.0 A respectively.

## Results

The corrected full campaigns cover all 16 manifest checkpoints. CPU passed all
66 declared cases. CUDA passed all 68 declared speed and capacity cases after
the MH-1 device module was made precision-aware. HIP passed all 68 declared
speed and capacity cases. The controller schedules only structures declared
for each model family, so the matrix has no artificial not-applicable rows.

| Backend | Declared cases | Passed | Not applicable | Failed |
|---|---:|---:|---:|---:|
| CPU OpenMP | 66 | 66 | 0 | 0 |
| CUDA SM120 | 68 | 68 | 0 | 0 |
| HIP gfx1151 | 68 | 68 | 0 | 0 |

Every passing case selected direct streamed-edge execution, loaded a generated
R1 or MH-1 artifact, observed generated forward and reverse launches, and
reported zero fallback evaluations. The additional CUDA cases are capacity
probes, including the hardware-aware MPA FP64 tile fallback. The detailed
per-checkpoint numerical results and immutable record hashes follow.

All 18 OFF23/H2O speed cases passed. Each H2O graph had three atoms and six
directed edges; every run selected generated `jit_all` forward and `jit`
reverse execution with zero fallback. The maximum direct-versus-generic force
difference for each backend and precision was:

| Checkpoint | CPU FP32 | CPU FP64 | CUDA FP32 | CUDA FP64 | HIP FP32 | HIP FP64 |
|---|---:|---:|---:|---:|---:|---:|
| OFF23 small | `1.912e-6` | `3.153e-14` | `1.269e-6` | `7.994e-15` | `1.045e-5` | `2.442e-14` |
| OFF23 medium | `3.503e-5` | `1.292e-13` | `2.109e-5` | `8.971e-14` | `3.285e-5` | `6.151e-14` |
| OFF23 large | `2.475e-6` | `6.106e-15` | `1.918e-6` | `5.995e-15` | `2.851e-6` | `6.661e-15` |

All values in this table are in `eV/A`.

The accelerator MH-1 runtime retains the stable v4 packet layout while the
NVRTC/hipRTC module renderer now emits precision-specific pointee types and
validates the descriptor scalar width. CUDA FP64 uses NVRTC; the separately
compiled shared-plugin route remains FP32-only. On both C64 and SrTiO3, FP64
selected `pair_spline_v1_staged_device_v5`, launched generated forward,
source-reverse, and edge-reverse kernels twice, and reported zero fallback.
Maximum direct-versus-generic force differences were `1.15e-14 eV/A` and
`5.22e-15 eV/A`, respectively. A CUDA FP32 regression rerun also passed with a
`3.45e-6 eV/A` maximum force difference.

### Expanded current-source CPU qualification

A consolidated CPU campaign subsequently ran all 16 manifest checkpoints on
C64 and their family-specific second structure: corrected
face-centred-oxygen SrTiO3 for inorganic and multihead models, or isolated H2O
for OFF23. The inorganic campaign used the fresh native OpenMP extension built
from commit
`a8cb0f6f7e11c383db697481fe55d1c6a725cf2b`, whose SHA-256 was
`fe266beb486529635242429e28491bb181c12d7309a17ac415ed2188b9a67289`.
The H2O supplement used a current-source extension with SHA-256
`06d0ec1e8e8d9080386bdb9ccf03b54592d385aaadcc08e27237425c302e8175`.
Processes were pinned to physical CPU 0 with one Kokkos/OpenMP thread and one
BLAS thread.

All 64 speed-profile cases passed in FP32 and FP64. Every case selected direct streamed-edge
execution, loaded a generated artifact, observed the required generated
forward and reverse launches, and retained zero fallback evaluations. The
largest force differences across the two structures were:

| Checkpoint | Applicable speed cases | FP32 direct vs generic | FP64 direct vs generic | FP32 direct vs FP64 direct |
|---|---:|---:|---:|---:|
| OMAT-0 small | 4 | `8.77e-5 eV/A` | `1.57e-13 eV/A` | `6.13e-6 eV/A` |
| OMAT-0 medium | 4 | `4.98e-5 eV/A` | `1.26e-13 eV/A` | `5.57e-6 eV/A` |
| MPA-0 medium | 4 | `6.02e-4 eV/A` | `1.93e-12 eV/A` | `6.90e-5 eV/A` |
| MP-0b small | 4 | `9.13e-5 eV/A` | `1.58e-13 eV/A` | `5.19e-6 eV/A` |
| MP-0b medium | 4 | `3.08e-5 eV/A` | `5.00e-14 eV/A` | `3.99e-6 eV/A` |
| MP-0b2 small | 4 | `1.70e-5 eV/A` | `3.55e-14 eV/A` | `4.21e-6 eV/A` |
| MP-0b2 medium | 4 | `3.14e-5 eV/A` | `6.59e-14 eV/A` | `3.56e-6 eV/A` |
| MP-0b2 large | 4 | `7.48e-5 eV/A` | `1.56e-13 eV/A` | `5.87e-6 eV/A` |
| MP-0b3 medium | 4 | `4.93e-6 eV/A` | `3.69e-14 eV/A` | `2.67e-6 eV/A` |
| MATPES PBE | 4 | `5.32e-4 eV/A` | `1.21e-12 eV/A` | `3.76e-5 eV/A` |
| MATPES r2SCAN | 4 | `1.20e-5 eV/A` | `3.82e-14 eV/A` | `3.50e-6 eV/A` |
| MH-0, `omat_pbe` head | 4 | `7.92e-5 eV/A` | `1.04e-13 eV/A` | `4.73e-6 eV/A` |
| MH-1, `omat_pbe` head | 4 | `2.31e-6 eV/A` | `5.69e-15 eV/A` | `4.30e-6 eV/A` |
| OFF23 small | 4 | `1.25e-4 eV/A` | `3.10e-13 eV/A` | `1.97e-5 eV/A` |
| OFF23 medium | 4 | `1.23e-3 eV/A` | `2.87e-12 eV/A` | `1.06e-4 eV/A` |
| OFF23 large | 4 | `1.42e-4 eV/A` | `2.06e-13 eV/A` | `1.31e-5 eV/A` |

The two declared CPU capacity probes for OFF23 large also passed. Both used a
generated 224-channel host M0 plugin, built-in standard R0 v2, generated R1
forward and reverse execution, and zero fallback. FP32 and FP64 both selected
an M1 recomputation tile width of 32. This distinguishes the capacity
contract, which requires runtime-specialized M0/R0, from the speed contract,
where retaining generic M0 is valid when generated R1 remains active.

The authoritative normalized CPU JSONL is
`/tmp/symmetrix-foundation-cpu-consolidated-20260903/cpu-results-h2o.jsonl`
with SHA-256
`1e25f19f6ab334f5c995adf22f1ee2403cf48b96f4aa6cbe9aa49d30b11248af`.

### Expanded current-source CUDA qualification

A subsequent campaign covered all 16 checkpoints in the manifest using C64
and the same family-specific SrTiO3 or H2O second structures. The backend was
rebuilt from the current dirty source at commit
`a8cb0f6f7e11c383db697481fe55d1c6a725cf2b` with CUDA 13.3 for SM120 and
executed on the RTX 5090. The native extension SHA-256 was
`6427c4a0cf0a8649e0991b7f21713b255303bb463cb93a9e0ca96c2d3968d767`.
The H2O supplement used a current-source extension with SHA-256
`9c564909c865e70134f70d03a10d4f5c1a8af90048766640ce9f6cdde24ffd90`.

The corrected normalized record contains 68 unique cases and all 68 passed.
Every checkpoint passed direct FP32 and FP64 execution on both of its declared
structures. Ordinary
MACE used generated `jit_all` forward and `jit` reverse execution with zero
fallback evaluations. Speed-profile runs record the selected M0/R0
implementations but do not require specialization; capacity-profile runs
require specialized M0 and R0 executors.

Four retained-capacity probes also passed:

| Checkpoint | Precision | Selected M1 tile | M0 | R0 |
|---|---:|---:|---|---|
| OMAT-0 small, SrTiO3 | FP32 | 32 | built-in `standard-m0-lmax0-module-v1` | built-in standard R0 v2 |
| MPA-0 medium, SrTiO3 | FP64 | 16 | built-in standard M0 v1 | built-in standard R0 v2 |
| OFF23 large, C | FP32 | 32 | generated CUDA M0, 224 channels | built-in standard R0 v2 |
| OFF23 large, C | FP64 | 16 | generated CUDA M0, 224 channels | built-in standard R0 v2 |

The MPA FP64 probe did not set a tile override. Tile 32 required `72,704 B`
of team scratch and was rejected against Kokkos's `40,936 B` usable SM120
limit; automatic selection admitted tile 16 at `36,352 B`. It retained
generated R1 forward/reverse execution, specialized M0/R0 modules, and zero
fallback evaluations. Its direct-versus-generic maximum force difference was
`9.83e-13 eV/A` in the normalized run.

Across the corrected CUDA MPA speed cases, the largest FP32
direct-versus-generic force difference was `2.280e-4 eV/A`; the largest direct
FP32-versus-direct FP64 difference was `2.572e-4 eV/A`. Both remain within the
manifest gate and are consistent with the accumulation-order diagnosis below.
The corrected normalized JSONL is
`.task-cuda-foundation-matrix-20260903/results/cuda-all-current-source-h2o.jsonl`
with SHA-256
`1e0be1398ae4d57f292b55138311925b3c1eb27e60d7808931207e1e8fe54c92`.
The original matrix cases used the native hash above; the two corrected MH-1
FP64 records used the rebuilt native SHA-256
`1706853f8a91b2ccd94233f3ec549fa66f688725fe822f57cb81a46189c64128`.

### AMD HIP qualification

The AMD campaign ran on a dedicated HIP host using a Radeon 8060S with 8 GiB VRAM and
native target `gfx1151`. The fresh backend used ROCm 7.14, hipcc
7.14.60850/Clang 23.0.0, Kokkos HIP plus Serial, the `AMD_GFX1100` Kokkos
trait, and an explicit `--offload-arch=gfx1151` compiler target. The backend
doctor detected the exact GPU target and passed its device execution sentinel.
The installed native extension SHA-256 was
`20eaf14114eac2381619b495984d2461ea8c8cd4e75beb2c5c700ba78f4e0294`.

All 64 speed-profile cases passed in FP32 and FP64. Ordinary MACE selected
generated `jit_all` forward and `jit` reverse execution. MH-1 selected
`pair_spline_v1_staged_device_v5`. Every case reported zero fallback and an
empty direct gate failure list. Ordinary FP32 cases selected an M1 tile width
of 32; FP64 cases selected 16, demonstrating precision- and hardware-aware
scratch admission rather than a fixed tile width.

Four capacity-profile probes also passed: OMAT-0 small FP32, MPA-0 medium
FP64, and OFF23 large in FP32 and FP64. They selected specialized M0/R0
executors, retained generated R1 forward and reverse execution, and reported
zero fallback. MPA-0 FP64 selected tile 16 without a manual override.

The largest direct-versus-generic force difference was `2.103e-4 eV/A` in
FP32 and `1.198e-12 eV/A` in FP64. The largest direct FP32-versus-direct FP64
force difference was `5.208e-4 eV/A`. All were below the existing matrix
tolerances. The authoritative normalized HIP JSONL is
`.task-hip-foundation-matrix-20260903/results/hip-all.jsonl` with SHA-256
`d0e6ce30a2dfeb00e57867d757d0c7996dd7cb5c1555851f6b9dce6d110e4048`.
The retained qualification artifacts include the target manifest and backend
qualification JSON.

### MPA-0 medium FP32 numerical diagnosis and remediation

An additional CPU check investigated the larger MPA-0 medium FP32
direct-versus-generic force differences: `5.994e-4 eV/A` on C and
`4.999e-4 eV/A` on SrTiO3. These differences do not indicate an error in the
generated direct executor. Against the common FP64 result, direct FP32 was
substantially more accurate:

| Structure | Generic FP32 max force error | Direct FP32 max force error | Generic FP32 RMS force error | Direct FP32 RMS force error |
|---|---:|---:|---:|---:|
| C64 | `5.878e-4 eV/A` | `3.824e-5 eV/A` | `2.484e-4 eV/A` | `1.426e-5 eV/A` |
| SrTiO3-40 | `5.288e-4 eV/A` | `6.891e-5 eV/A` | `1.384e-4 eV/A` | `1.743e-5 eV/A` |

The direct FP32 result was also closer to independent MACE PyTorch FP32 on
the same structures. On C64, the maximum force difference was
`6.264e-5 eV/A` for direct and `5.712e-4 eV/A` for generic. A central
finite-difference check of the worst direct-versus-generic C coordinate agreed
with the FP64 analytic force within `6.8e-9 eV/A` at a `3e-4 A` displacement.

Executor controls excluded generated code, M0, R0, and host fast-math as the
cause. Direct execution with runtime R1 forward and reverse retained a
`4.270e-5 eV/A` maximum force error against FP64 on C64. Runtime and standard
M0, the R0 variants, and an IEEE-oriented host JIT artifact changed the result
only at the ordinary FP32 rounding scale. In contrast, selecting the serial
reference source-reduction strategy reproduced the generic error scale.

The former generic reverse path accumulated edge and coupling contributions
into source adjoints using FP32 atomic additions. Direct execution uses
source-owned or team-cached reductions with a different, more favorable
association. The MPA checkpoint is more sensitive to that accumulation order
than the other models in this matrix.

Generic FP32 reverse now accumulates the shared source adjoint in a persistent
node-sized FP64 buffer and converts it to FP32 once after the reduction. The
buffer costs `8 * nodes * num_LM * channels` bytes and is reused when the graph
shape does not grow. On the MPA workloads it reduced the maximum generic versus
direct force difference as follows:

| Backend | Structure | Before | After |
|---|---|---:|---:|
| CPU | C64 | `5.994e-4 eV/A` | `3.926e-5 eV/A` |
| CPU | SrTiO3-40 | `4.999e-4 eV/A` | `2.001e-5 eV/A` |
| CUDA | C64 | `2.235e-4 eV/A` | `4.425e-5 eV/A` |
| CUDA | SrTiO3-40 | `2.280e-4 eV/A` | `2.516e-5 eV/A` |

One-core CPU generic timing changed by less than `0.1%`. CUDA generic timing
changed by `-0.17%` on C64 and `+0.42%` on SrTiO3, which is within measurement
noise. FP64 results were unchanged at approximately `1e-12 eV/A` or better.
The MPA checkpoint SHA-256 was
`75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`.

## Checkpoints

| Checkpoint | SHA-256 |
|---|---|
| OMAT-0 small | `0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5` |
| OMAT-0 medium | `d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a` |
| MPA-0 medium | `75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638` |
| MP-0b small | `7e3a0abcaf41e03a80e69f778e1b11b29de1cca704783dc25917a736392f8cf0` |
| MP-0b medium | `ab8baff639a8f295f3eccad3d3ccf574efb6fb63220bd52cd88664211569e521` |
| MP-0b2 small | `d5773bf9440e96d6eb8c598f84bd0e6369fcfa432f626a87f890e07da3c651c9` |
| MP-0b2 medium | `a90be07c8aa6623c390fcc4653d3e319c4a356f8b4a53d323f96b2012d375caf` |
| MP-0b2 large | `348390e758e1c90011c7675e864850f8e9c5b3e7c217f79a2b4bad1baaa657ad` |
| MP-0b3 medium | `2f2be696351ac9e94fbe01cdfb6f017679acdbd2db7645209ef55fec9826b012` |
| MATPES PBE | `e618ad582b84239905b9c3b77ce6e9ce111b0ecd1533223a1a6aac7a696b8aa0` |
| MATPES r2SCAN | `8f147ecffa1d06a696e5648b13b81abda9c96fda9d0fb3faba3cd3fb27d9bba9` |
| MH-0 | `d62ff8f293664e6556cfa49364b28ee72a70cf1e6150f3c111a578397fed609d` |
| MH-1 | `a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47` |
| OFF23 small | `165cce4cfec5a34b9c64d4ebf95de15d71106bb584b7291c8470f0749977c46f` |
| OFF23 medium | `4842c52ad210d6e1f84d6cf1ffa70fae25a7e0d755ed55cf223f43913f587db7` |
| OFF23 large | `a29e397dbf3e7a24ac50a9b0dfc919bd5a62efa346f5895a6237b0950c1d76f4` |

The reusable controller and pinned model manifest are
`benchmarks/foundation_model_support_matrix.py` and
`benchmarks/foundation_model_support_manifest.json`. The authoritative CPU and
CUDA record locations and hashes are given in their respective sections; build
trees and downloaded checkpoints are qualification artifacts, not repository
source.
