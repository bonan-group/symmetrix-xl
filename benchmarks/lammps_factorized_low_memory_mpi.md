# LAMMPS direct low-memory MPI qualification

## CPU host-artifact qualification (2026-08-19)

CPU LAMMPS now accepts a pre-generated Execution host artifact through
`jit_host_artifact`. LAMMPS does not invoke Python or a compiler. The artifact
was prepared and native-loader-validated before launch with:

```bash
symmetrix_prepare_jit_host_artifact \
    --model /tmp/mace-mh-0-current-Al-N.json \
    --precision float32 \
    --path-only
```

The correctness matrix uses the physical 64-atom wurtzite AlN fixture in
`test_pair_symmetrix_factorized_mpi.py`. Two-rank retained direct, two-rank
direct low memory, and one-rank direct low memory agree within the existing
FP32 gates for periodic orthogonal and restricted-triclinic cells, before and
after one nitrogen crosses the `1x1x2` MPI ownership boundary.

The timing workload is MH-0 FP32 on 864 perturbed wurtzite AlN atoms. The model
cutoff is 6.0 A, the neighbor skin is 0.5 A, and the effective candidate cutoff
is 6.5 A. It has 78,624 active directed edges and 97,698 neighbor-list
candidates. Both configurations use two OpenMPI ranks bound to two physical
cores, one Kokkos/OpenMP thread and one OpenBLAS thread per rank, three warmup
steps, and ten measured static steps.

| Execution | us/atom | ms/step | Relative time |
|---|---:|---:|---:|
| `direct low_memory`, generated host artifact | **219.178** | 189.370 | 1.000x |
| `generic`, compiler-free control | 3417.627 | 2952.830 | 15.59x |

The measured direct loop is 1.89370 s for ten steps. Pair evaluation accounts
for 99.95% of that loop and MPI communication for 0.03%. The generic control is
the previously recorded 29.5283 s loop on the same executable configuration,
model, graph, ranks, and thread placement.

Environment identities:

- LAMMPS 4 Jul 2026 Development, OpenMPI 5.0.10
- Kokkos 5.1.99, OpenMP + Serial, native AVX-512, OpenBLAS
- LAMMPS executable SHA-256:
  `13670b7abc1b38428f1cf6f13f1867e07bec12427e6604ba3c7faf0585c4cff6`
- MH-0 model SHA-256:
  `c202e5bc6da9a2b0ca4974595869fa5f4f0cdb7f37b58d87415ecd95f048eb22`
- Host artifact ID: `jit-r1-gen2-f32-29f6298ae84d2005`
- Host artifact SHA-256:
  `1064ddeebe985a6a45b9eead45382dfd7bdd8ad9944d64ed3071e6f27df1b61b`

The focused correctness test is
`test_direct_low_memory_mpi_matches_retained_after_migration`. The retained
control selects the compatibility alias `factorized -> direct`. Set
`SYMMETRIX_LAMMPS_JIT_HOST_ARTIFACT` to the matching `.so`; GPU qualification
continues to use `SYMMETRIX_LAMMPS_JIT_ARTIFACT` for the device module.

### CPU MPI strong scaling (2026-08-20)

The same 864-atom MH-0 FP32 `direct low_memory` workload was run with 1, 2, 4,
8, and 16 MPI ranks. The host has one socket with 16 physical cores and SMT2.
OpenMPI maps and binds one rank to each physical core; every rank uses one
Kokkos/OpenMP thread and one OpenBLAS thread. SMT oversubscription is not used.
The rank sweeps were interleaved and each fresh process performed three warmup
steps followed by ten measured static steps.

| MPI ranks | Decomposition | us/atom samples | Median us/atom | ms/step | Speedup | Efficiency |
|---:|---|---|---:|---:|---:|---:|
| 1 | `1x1x1` | 425.418, 429.097, 434.485 | **429.097** | 370.740 | 1.000x | 100.00% |
| 2 | `1x1x2` | 219.907, 220.568, 226.329 | **220.568** | 190.571 | 1.945x | 97.27% |
| 4 | `1x2x2` | 123.087, 121.591, 122.125 | **122.125** | 105.516 | 3.514x | 87.84% |
| 8 | `2x2x2` | 78.850, 70.594, 70.046 | **70.594** | 60.993 | 6.078x | 75.98% |
| 16 | `2x2x4` | 40.902, 40.806, 45.719 | **40.902** | 35.339 | 10.491x | 65.57% |

The 6.0 A active graph remains 78,624 directed edges and the 6.5 A candidate
graph remains exactly 97,698 directed entries at every rank count. There are
no neighbor rebuilds or dangerous builds. The final total energy is
`-6405.0652 eV` for every run at the printed FP32 precision. Pressure ranges
from 46,916.161 to 46,916.175 bar, a maximum decomposition-dependent span of
0.014 bar.

| MPI ranks | Average owned atoms/rank | Average ghost atoms/rank | Candidate edges/rank |
|---:|---:|---:|---:|
| 1 | 864 | 3,250 | 97,698.0 |
| 2 | 432 | 2,230 | 48,849.0 |
| 4 | 216 | 1,720 | 24,424.5 |
| 8 | 108 | 1,300 | 12,212.2 |
| 16 | 54 | 970 | 6,106.1 |

The loss of efficiency at high rank count is consistent with this very small
strong-scaling workload: the owned work halves at every step, while the ghost
state falls much more slowly. At 16 ranks there are about 18 ghost atoms per
owned atom on each rank. LAMMPS attributes 97.6% of the median 16-rank loop to
`Pair` and 2.3% to `Comm`; Symmetrix H1 packet preparation and message passing
inside the pair evaluation are included in the `Pair` row.

Date: 2026-08-17

## Scope

This qualification exercises standard MACE FP32 through the actual LAMMPS
low-memory bundle with two MPI ranks sharing one CUDA GPU. It compares:

- two-rank low memory against two-rank full retention; and
- two-rank low memory against one-rank low memory.

Both periodic orthogonal and restricted-triclinic cells are evaluated before
and after one nitrogen migrates across the `1x1x2` rank boundary. The checks
cover global energy and pressure, per-atom energy, force components, positions,
and ownership migration.

## Environment

- Source baseline: `7632c54fc1411bb2ac6b4b39aaa75265de955d91` plus the
  reviewed LAMMPS low-memory and per-atom synchronization changes in this work
- LAMMPS: 4 Jul 2026 Development; OpenMPI 5.0.10
- Kokkos: CUDA + Serial, version 5.1.99, CUDA 13.3, `sm_120`
- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB, driver 610.43.02
- MPI transport: two ranks sharing one GPU; GPU-aware MPI was not detected, so
  Kokkos selected host staging
- LAMMPS executable SHA-256:
  `1e32e5f13bb0f705d6b5a3f51fd152d745022cff422cc0adcef86e2665a277ad`
- LAMMPS shared-library SHA-256:
  `d6cfe8147125612826316bcf3ecfae3f1cf3e6705a22e90ac6464e753dbc167d`
- OMAT-0 compact model SHA-256:
  `af3b41b1e0de93dc1b8ba946f71ae3cb84c16b7c7a48840529cddd32382380a7`
- NVRTC R1 artifact: `jit-r1-gen2-f32-6fdf3e63624a0e45`, SHA-256
  `226382dcdb5a306746ca847f008217bbda199174438392fc9c89fcd503790cbd`

The model cutoff is 6.0 A, the LAMMPS skin is 1.0 A, and the effective
candidate cutoff is 7.0 A. The 64-atom graph has 5,824 directed active edges.
The orthogonal candidate graph has 9,152 directed edges; the triclinic graph
has 9,056 initially and 9,058 after migration.

CUDA uses `-k on g 1`, `package kokkos neigh half newton on`, and
`newton on`. The generated R1 artifact is loaded on every rank before native
`set_low_memory(true)` admission. The logs contain both the generated-artifact
identity and `Activated factorized low-memory execution`.

## Numerical results

All two-rank low-memory cases pass the retained and rank-invariance gates:

| Cell | Comparison | max force (eV/A) | max atom energy (eV) | max global energy (eV) | max pressure (bar) |
|---|---|---:|---:|---:|---:|
| Orthogonal | low memory vs retained | 8.21e-6 | 4.34e-6 | 1.09e-5 | 0.1280 |
| Orthogonal | two ranks vs one rank, low memory | 8.66e-6 | 5.39e-6 | 2.73e-5 | 0.1204 |
| Triclinic | low memory vs retained | 8.64e-6 | 6.46e-6 | 1.36e-5 | 0.0131 |
| Triclinic | two ranks vs one rank, low memory | 9.92e-6 | 8.17e-6 | 1.27e-5 | 0.0366 |

Positions agree exactly. The migrant changes owner in each two-rank case.
These are static `run 0` correctness cases, so their LAMMPS loop time is not a
performance result and is not reported as us/atom.

## Issues found

The CUDA fixture originally issued 64 separate `create_atoms single` commands.
That enters `AtomKokkos::map_set_device` while a rank can transiently own zero
atoms and produced an unrelated illegal access. Reading the complete physical
fixture from one LAMMPS data file avoids that invalid setup state.

CUDA Kokkos defaults to full neighbor lists with Newton communication off.
Factorized MPI requires half lists and `newton pair on` for reverse H1-adjoint
communication, so the qualification makes both settings explicit.

The first successful pair evaluation also exposed zero values from
`compute pe/atom`: Symmetrix marked its Kokkos per-atom energy DualView modified
on device but did not synchronize the host view read by LAMMPS. Adding
`k_eatom.sync_host()` to all three Kokkos pair execution modes restores
per-atom energy publication without copying any additional pairwise arrays.

## Reproduction

The focused test is
`pair_symmetrix/test/test_pair_symmetrix_factorized_mpi.py::test_direct_low_memory_mpi_matches_retained_after_migration`.
Set `SYMMETRIX_LAMMPS_EXECUTABLE`, `SYMMETRIX_LAMMPS_COMPACT_MODEL`,
`SYMMETRIX_LAMMPS_JIT_ARTIFACT`, `SYMMETRIX_LAMMPS_KOKKOS_GPUS=1`, and
`MPIEXEC_EXECUTABLE`, then run the two parameterized cases with pytest.

## Host-staging optimization qualification (2026-08-18)

The original host-staged callbacks mirrored the complete per-rank H1 or
H1-adjoint tensor on every pack and unpack. The revised callbacks retain the
same LAMMPS double-packet ABI but use reusable device packet/index buffers and
transfer only the active boundary packet. Forward unpack overwrites the
contiguous ghost range; reverse unpack atomically accumulates into the send
list. The four device pack/unpack callbacks also no longer fence internally;
LAMMPS transport fences and the evaluator phase boundaries remain in place.

For `n` boundary atoms, `num_LM=4`, and 128 channels, each staged packet is now
exactly `n * 4 * 128 * 8 = 4,096n` bytes in each applicable direction. Forward
pack and reverse unpack additionally transfer `4n` bytes of indices. Transfer
volume no longer depends on the total local-plus-ghost feature-node count.

Fresh builds completed for Kokkos OpenMP+MPI and CUDA 13.3+MPI (`sm_120`). The
two-rank OpenMP migration matrix passes eight cases: standard MACE and
MACEField, FP32 and FP64, and periodic orthogonal and triclinic cells. Each case
compares factorized execution with all-interactions and a one-rank factorized
reference before and after ownership migration.

After the GPU became available, the complete CUDA/MPI migration module passed
all ten cases. This adds CUDA coverage for standard MACE and MACEField, FP32 and
FP64, orthogonal and triclinic cells, and the factorized low-memory cases.

### Timed host-staged workload

The timing workload is a 4,096-atom periodic rocksalt AlN cell, standard OMAT-0
FP32, factorized low memory, and a `1x1x2` MPI decomposition with both ranks
sharing the RTX 5090. The model cutoff is 6.0 A, the skin is 1.0 A, and the
effective candidate cutoff is 7.0 A. The measured trajectory starts with
438,502 directed active edges and ends with 438,482; the 7.0-A candidate list
contains 729,088 directed entries. Each rank owns 2,048 atoms and has 5,887
ghost atoms.

Each fresh process performs five warmup steps followed by 20 measured NVE
steps. BLAS, OpenMP, and Kokkos use one host thread per rank. The primary timing
results are:

| Sample | us/atom | ms/step | final directed edges |
|---|---:|---:|---:|
| 1 | 10.175918 | 41.680562 | 438,758 |
| 2 | 10.272247 | 42.075122 | 438,428 |
| 3 | 10.224744 | 41.880552 | 438,576 |
| Median | **10.224744** | **41.880552** | - |

The Nsight-instrumented run is 10.522657 us/atom, or 43.100802 ms/step. The
trace is `/tmp/symmetrix-mpi-staging-4096.nsys-rep`; its SQLite export is
`/tmp/symmetrix-mpi-staging-4096.sqlite`.

### Transfer and synchronization profile

The pair counters report the following aggregate transfer volume for the 20
measured steps across both ranks:

| Transfer | 20-step bytes | bytes/step |
|---|---:|---:|
| Boundary packet D2H | 1,929,052,160 | 96,452,608 |
| Boundary packet H2D | 1,929,052,160 | 96,452,608 |
| Boundary index H2D | 1,883,840 | 94,192 |

The packet-to-index ratio is exactly 1,024, matching 4,096 packet bytes per
4-byte boundary index. In the steady-state trace, the largest H2D or D2H copy
is 8,667,136 bytes, exactly the largest MPI packet. No 16,250,880-byte complete
H1/H1-adjoint copy occurs. This verifies that steady-state host staging no
longer mirrors the complete feature tensor.

For comparison, the removed implementation copied the complete
`7,935 * 4 * 128 * 4 = 16,250,880`-byte FP32 state in every callback. With six
swaps in each communication phase and two ranks, its exact transfer lower
bound on this layout is:

| Direction | Removed full-state bytes/step | Packet-only bytes/step | Reduction |
|---|---:|---:|---:|
| D2H | 780,042,240 | 96,452,608 | 87.6% |
| H2D, including indices | 390,021,120 | 96,546,800 | 75.2% |
| Total | 1,170,063,360 | 192,999,408 | 83.5% |

Across the full Nsight trace, all four staging kernels consume 8.576 ms out of
1,107.099 ms of GPU kernel time, or 0.775%. CUDA stream and device
synchronization account for 76.7% of CUDA API time, while synchronous
`cudaMemcpy` accounts for 21.7%. MPI records 418 sends totaling 1,586,839,184
bytes and 120.645 ms of aggregate send-call time. The MPI volume is lower than
the staging counters because self-neighbor swaps still stage packets but do
not traverse MPI.

The result qualifies packet-only host staging and confirms that the staging
kernels themselves are not a GPU bottleneck. Remaining optimization should
target synchronization and self-swap staging first. Dependency-level message
batching still requires a LAMMPS communication API or maintained patch and a
real multi-GPU profile; this two-rank, one-GPU result is not a multi-card
scaling qualification.

## Four-A5000 strong scaling (2026-08-19)

The multi-card qualification uses a portable cluster build: GCC
12.2, OpenMPI 4.1.5, CUDA 12.9, Kokkos `AMPERE86`, and an x86-64-v3 host
baseline. Each Slurm step assigns one physical RTX A5000 to one MPI rank;
Slurm renumbers that device to logical device 0, so every rank launches LAMMPS
with `-k on g 1`.

The workload is a `17 x 17 x 17` replication of the eight-site cubic AlN cell,
or 39,304 atoms. It uses OMAT-0 FP32, `streamed_edges factorized`,
`low_memory yes`, and generated artifact
`jit-r1-gen2-f32-6fdf3e63624a0e45`. The model cutoff is 6.0 A, the neighbor
skin is 1.0 A, the effective neighbor-list cutoff is 7.0 A, and every run has
3,615,968 directed edges at the model cutoff. Each fresh job performs five
warmup steps and then one measured 20-step loop. The campaign interleaves
1-, 2-, and 4-GPU jobs and repeats that sequence three times.

### Timing results

`us/atom/step` is computed from the measured 20-step LAMMPS loop. `ms/step`
and the speedup use the median of the same three fresh jobs.

| GPUs / MPI ranks | Slurm jobs | us/atom/step samples | median us/atom/step | median ms/step | speedup | efficiency |
|---:|---|---|---:|---:|---:|---:|
| 1 | 543627, 543630, 543633 | 22.085284, 22.594138, 22.719443 | **22.594138** | 888.0400 | 1.000x | 100.00% |
| 2 | 543628, 543631, 543634 | 12.425427, 12.531714, 12.544576 | **12.531714** | 492.5465 | 1.803x | 90.15% |
| 4 | 543629, 543632, 543635 | 7.275901, 7.279017, 7.320578 | **7.279017** | 286.0945 | 3.104x | 77.60% |

Sample standard deviations are 0.335855, 0.065394, and
0.024944 us/atom/step for 1, 2, and 4 GPUs. Their coefficients of variation
are 1.486%, 0.522%, and 0.343%, respectively. Slurm allocation elapsed times
are consistently 40 s, 29 s, and 24 s; all nine jobs completed with
exit code `0:0`.

Median LAMMPS timing attribution is:

| GPUs | Pair share | Comm share | median Pair time (s/20 steps) | median Comm time (s/20 steps) |
|---:|---:|---:|---:|---:|
| 1 | 99.89% | 0.09% | 17.7410 | 0.016656 |
| 2 | 99.48% | 0.45% | 9.8019 | 0.043914 |
| 4 | 99.33% | 0.57% | 5.6870 | 0.032619 |

The `Comm` row is not a complete measure of MPI overhead. Symmetrix packet
packing, device/host copies, synchronization, and some staged communication
occur inside the pair-style call and are therefore charged to `Pair` by
LAMMPS. The observed loss from 90.15% two-GPU efficiency to 77.60% four-GPU
efficiency should be profiled at that boundary rather than inferred from the
small `Comm` percentage alone.

### Correctness and transport boundary

Every run activates the same generated artifact and factorized low-memory
path, reports the same atom and directed-edge counts, completes with finite
energy and pressure, performs no neighbor rebuilds, and reports zero dangerous
builds. Relative to the matched one-GPU repetition, the maximum precise
initial-state differences are:

| Comparison | total energy (eV) | energy per atom (micro-eV) | pressure (bar) |
|---|---:|---:|---:|
| 2 GPUs vs 1 GPU | 0.001618 | 0.0412 | 0.008003 |
| 4 GPUs vs 1 GPU | 0.002741 | 0.0697 | 0.027600 |

LAMMPS prints `Turning off GPU-aware MPI since it is not detected` for every
multi-rank job. These results therefore qualify the optimized host-staged
boundary-packet implementation, not CUDA-aware MPI or NCCL. Full logs are
stored with the qualification run as
`logs/scaling-39304-r{1,2,4}-rep{1,2,3}.lammps.log`.

## HPC-X CUDA-aware MPI strong scaling (2026-08-19)

The CUDA-aware comparison uses the same cluster build, 39,304-atom input,
OMAT-0 model, generated artifact, FP32 factorized low-memory policy, rank
decompositions, five warmup steps, and 20 measured steps as the host-staged
campaign above. Only the MPI-linked LAMMPS executable and runtime change:

- NVIDIA HPC-X module `nvhpc-hpcx-2.20-cuda12/25.1`
- OpenMPI 4.1.7a1 and UCX 1.17.0 with CUDA support enabled
- LAMMPS `build/lammps-cuda-mpi-sm86-hpcx/lmp`, SHA-256
  `ac9d6737725ba47b8dc7cb1b5c79c0432f9b1d3af5407b85a57daa564dc07efb`
- Symmetrix pair implementation corresponding to commit
  `112fd3cebbb4633ae100a2ec6ef168d0659989ff`; installed extension SHA-256
  `fce2d0b39bee48f4080a296f9d750a2daa8da02699133dc35152446ce618d8f4`
- LAMMPS 4 Jul 2026 Development; the dedicated copied source tree does not
  retain Git metadata, so a more specific upstream revision is unavailable
- Compact model SHA-256
  `af3b41b1e0de93dc1b8ba946f71ae3cb84c16b7c7a48840529cddd32382380a7`
- Generated `sm_86` cubin SHA-256
  `15c06e1ce574e273fe1090fe8b2de6167a3c849bfa5facbee024d8cc44f243ab`
- `OMPI_MCA_coll_hcoll_enable=0`, disabling an optional collective plugin that
  could not initialize on the qualification node

The model cutoff is 6.0 A, the neighbor skin is 1.0 A, the effective cutoff is
7.0 A, and every case has 3,615,968 directed edges at the model cutoff. Each
Slurm step assigns one A5000 to one MPI rank, with the same `1x1x1`, `1x1x2`,
and `1x2x2` processor grids used by the OpenMPI 4.1.5 baseline.

### CUDA-aware timing results

| GPUs / MPI ranks | Slurm jobs | us/atom/step samples | median us/atom/step | median ms/step | speedup | efficiency |
|---:|---|---|---:|---:|---:|---:|
| 1 | 543778, 543781, 543785 | 22.011118, 22.591467, 22.624542 | **22.591467** | 887.9350 | 1.000x | 100.00% |
| 2 | 543779, 543782, 543786 | 11.474659, 11.588681, 11.651181 | **11.588681** | 455.4815 | 1.949x | 97.47% |
| 4 | 543780, 543784, 543787 | 6.018802, 6.058022, 6.074878 | **6.058022** | 238.1045 | 3.729x | 93.23% |

Sample standard deviations are 0.345009, 0.089505, and
0.028772 us/atom/step, corresponding to 1.527%, 0.772%, and 0.475%
coefficients of variation. Median Slurm allocation elapsed times are 39 s,
26 s, and 22 s. All nine jobs completed with exit code `0:0`.

Compared at the same GPU count:

| GPUs | host-staged us/atom/step | HPC-X us/atom/step | HPC-X speedup | time reduction |
|---:|---:|---:|---:|---:|
| 1 | 22.594138 | 22.591467 | 1.0001x | 0.01% |
| 2 | 12.531714 | 11.588681 | 1.0814x | 7.53% |
| 4 | 7.279017 | 6.058022 | 1.2016x | 16.77% |

The unchanged one-GPU result provides a useful control: the multi-GPU gains
come from the communication path rather than a different evaluator or device
kernel. Four-GPU efficiency improves from 77.60% with host staging to 93.23%
with CUDA-aware MPI.

Median LAMMPS attribution changes as follows:

| GPUs | transport | Pair time (s/20 steps) | Pair share | Comm time (s/20 steps) | Comm share |
|---:|---|---:|---:|---:|---:|
| 1 | host staged | 17.7410 | 99.89% | 0.016656 | 0.09% |
| 1 | HPC-X | 17.7390 | 99.89% | 0.016872 | 0.10% |
| 2 | host staged | 9.8019 | 99.48% | 0.043914 | 0.45% |
| 2 | HPC-X | 9.0634 | 99.54% | 0.038573 | 0.43% |
| 4 | host staged | 5.6870 | 99.33% | 0.032619 | 0.57% |
| 4 | HPC-X | 4.7053 | 98.81% | 0.053960 | 1.13% |

The higher four-GPU `Comm` share does not negate the gain: host packet
packing, device/host copies, and synchronization occur inside the pair-style
call and are charged to `Pair`. With HPC-X, measured Pair time drops by
0.9817 s over 20 steps while explicit Comm time rises by only 0.0213 s.

### Selection and correctness

HPC-X reports CUDA-aware support, and LAMMPS does not print
`Turning off GPU-aware MPI since it is not detected` in any multi-rank log.
LAMMPS selects this path through `MPIX_Query_cuda_support()` without forcing
`gpu/aware on`. No HCOLL warning occurs. The logs do not independently report
zero staged bytes, so this establishes automatic runtime selection rather than
a direct host-transfer measurement.

Every job activates the same generated artifact and factorized low-memory
path, reports identical atom and edge counts, completes with finite energy and
pressure, performs no neighbor rebuilds, and has zero dangerous builds.
Maximum matched differences relative to one GPU are:

| Comparison | total energy (eV) | energy per atom (micro-eV) | pressure (bar) |
|---|---:|---:|---:|
| 2 GPUs vs 1 GPU | 0.002294 | 0.0584 | 0.017845 |
| 4 GPUs vs 1 GPU | 0.002805 | 0.0714 | 0.028497 |

The HPC-X logs are stored remotely as
`logs/scaling-hpcx-39304-r{1,2,4}-rep{1,2,3}.lammps.log`. Separate
`run-scaling-39304-hpcx.sh` and `launch-scaling-39304-hpcx.sh` helpers preserve
the OpenMPI launchers and all prior logs. Their reproducibility hashes are:

| Remote file | SHA-256 |
|---|---|
| `scripts/scaling-39304.in` | `b8137e85b524a6b4691d063215a1b40fca13c9ed2754861a8490322093bbc6cb` |
| `scripts/run-scaling-39304-hpcx.sh` | `7616c5b8650c6d6597edc5805ffb2308180f92be498ad172427093a8096ce590` |
| `scripts/launch-scaling-39304-hpcx.sh` | `5477a25caa43d884011c10916b72144db18378aba7487f35c5665d5e091bfafc` |
