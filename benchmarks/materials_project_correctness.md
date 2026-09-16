# Materials Project cross-backend correctness

## Decision

All tested Symmetrix paths pass energy, force, and stress checks against
MACE-PyTorch on ten deterministically sampled Materials Project structures.
The sample contains unique MP material IDs and uses only the endpoints of the
published ferroelectric trajectories, not interpolated images.

The optimized paths were admitted as requested:

- CUDA `all` retained selected runtime M0 and retained M1.
- CUDA `all` AOT M0 selected generated M0 and retained M1.
- CUDA `all` AOT M0 plus M1 recomputation selected generated M0 and recomputed M1.
- CUDA `direct` selected the static direct fixed-weight R1 artifact,
  generated M0, and retained M1.

## Provenance

- Source commit: `86fdfc21f9709312e2e8b152c1a8a32e245a20f2`.
- Sampling seed: `20260806`.
- Public dataset: MACEField 1.0.2 Materials Project response release.
- Public source-asset SHA-256:
  `4d7c3d3e8642b5691ccfb93bf3a41b7b89425d9018eeedef8cb73efabcfd350a`.
- Prepared test-split SHA-256:
  `76daa9b0b76fccaaa93c91bde82bba27c20cce1a64504fa1b3018e93cceb1ba8`.
- MH-0 checkpoint SHA-256:
  `d62ff8f293664e6556cfa49364b28ee72a70cf1e6150f3c111a578397fed609d`.
- Compact `omat_pbe` model SHA-256:
  `66f9eb745d2346dfd23bf961e0479817e198e3397772ce0958af70592a8b8698`.
- CUDA extension SHA-256:
  `779345aef68778dabcaabd232aad28a983227e16f809e9919d09fe4fb6a74ee6`.
- OpenMP extension SHA-256:
  `adbcbb3c3e47a6d5ada926eea8110d804642641b258a5806bda966069bd1afcd`.
- Benchmark-driver SHA-256:
  `4f5934bb66b90279c4b25d007e7ab303f149528b98246f3204607509aeb41e1f`.

The legacy `2023-12-03-mace-mp.model` was considered first but is not a valid
oracle for this suite because Symmetrix correctly rejects its unsupported first
interaction block. The tested model is the existing qualified ordinary MH-0
checkpoint with its explicit `omat_pbe` head and matching compact model.

## Sample

| MP ID | Endpoint | Formula | Atoms | Source index |
|---|---|---:|---:|---:|
| mp-568350 | polar | Cl48Rb24Zn12 | 84 | 749 |
| mp-551176 | polar | C2O8Rb8 | 18 | 1079 |
| mp-608314 | nonpolar | Cl48Rb24Zn12 | 84 | 740 |
| mp-561189 | nonpolar | O46Rb12Si20 | 78 | 1890 |
| mp-12924 | nonpolar | K2La2S8Si2 | 14 | 1190 |
| mp-667338 | nonpolar | N72Pb12 | 84 | 280 |
| mp-681439 | polar | Li4O48P12Zr8 | 72 | 1819 |
| mp-653454 | nonpolar | Cl16K8Zn4 | 28 | 140 |
| mp-13664 | polar | O28Sr8Ta8 | 44 | 369 |
| mp-620058 | polar | N72Pb12 | 84 | 289 |

## Correctness results

The float64 limits are `2e-4 eV/atom`, `3e-3 eV/A`, and `4e-3 eV/A^3`.
The float32 limits are `5e-4 eV/atom`, `5e-3 eV/A`, and `8e-3 eV/A^3`.
Every value below is the maximum over all ten structures.

| Profile | Result | max dE (eV/atom) | max dF (eV/A) | max dStress (eV/A^3) | median ms/structure |
|---|---:|---:|---:|---:|---:|
| Symmetrix serial float64 | PASS | 7.293e-7 | 3.701e-4 | 1.733e-6 | 58.523 |
| Kokkos OpenMP `all` float64 | PASS | 7.293e-7 | 3.701e-4 | 1.733e-6 | 21.963 |
| CUDA `all` retained float32 | PASS | 1.849e-6 | 3.503e-4 | 1.756e-6 | 3.926 |
| CUDA `all` AOT M0 float32 | PASS | 1.849e-6 | 3.516e-4 | 1.754e-6 | 2.694 |
| CUDA `all` AOT M0 + M1 recompute float32 | PASS | 1.849e-6 | 3.499e-4 | 1.755e-6 | 2.683 |
| CUDA generated `direct` float32 | PASS | 2.033e-6 | 3.662e-4 | 1.830e-6 | 1.761 |

MACE-PyTorch CPU float64 took a median 230.190 ms/structure. Timing is included
as a diagnostic only: this suite uses one process per profile and is designed
for correctness and admission checks, not paired performance qualification.

## Artifacts

The machine-readable report, sampled extxyz snapshot, per-profile stdout/stderr,
and concise generated Markdown report are under
`benchmarks/.artifacts/materials_project_correctness/`. The final native worker
stderr logs are empty; the PyTorch log contains only upstream deprecation and
float64-conversion warnings.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/symmetrix-mpl-cache \
SYMMETRIX_JIT_CACHE=/tmp/symmetrix-mp-correctness-jit \
.venv/bin/python \
  benchmarks/materials_project_correctness.py run \
  --dataset /path/to/mp-ferroelectric-test.xyz \
  --checkpoint /path/to/mace-mh-0.model \
  --model-json benchmarks/.artifacts/macefield_mh0_generated/mh0-current-contract.json \
  --head omat_pbe \
  --source-root "$PWD" \
  --cuda-extension /tmp/symmetrix-omat-profile-sm120/symmetrix.cpython-312-x86_64-linux-gnu.so \
  --openmp-extension /tmp/symmetrix-openmp-120a/symmetrix.cpython-312-x86_64-linux-gnu.so \
  --output benchmarks/.artifacts/materials_project_correctness/results.json \
  --count 10 --warmups 1 --samples 3
```
