# Repeated LAMMPS run GPU OOM remediation

## Scope

This record covers the repeated-`run` CUDA allocation failure reported for
distributed OMAT-0-medium LAMMPS workloads. On A100-80GB, a first run block
completed near 62.3 GiB/GPU, while setup of a second block attempted an
additional unlabeled 18.61 GiB allocation and failed.

The qualified failure used one MPI rank per A100, CUDA-aware MPI, 975,560
atoms/rank, about 100,287,500 directed edges/rank, FP32 direct low-memory
execution, a 6.0 A model cutoff, a 0.5 A neighbor skin, and a 6.5 A effective
neighbor cutoff.

## Root cause and correction

The 18.61 GiB request matches the distributed first-interaction adjoint state:

\[
N_{\mathrm{feature\ nodes}} \times 16\ \mathrm{LM}
\times 128\ \mathrm{channels} \times \mathrm{sizeof(float)}.
\]

During reverse MPI communication, the pair style temporarily aliases the
evaluator-owned `H1_adj` view. The pair-side alias remained alive after the
communication fence. When the next LAMMPS run boundary produced a larger
feature-node extent, Kokkos could not release the old allocation before
allocating the replacement because the old view still had multiple owners.

The correction:

- releases the pair-side `H1_adj` alias immediately after the reverse
  communication fence in both direct/prepared and generic/materialized MPI
  paths;
- centralizes evaluator `H1_adj` growth in
  `ensure_h1_adjoint_capacity()`;
- allocates resized transient H1 storage without preserving unused contents;
- assigns stable Kokkos labels: `MACE H1 adjoint`,
  `Distributed factorized prefix H1`, and
  `Pair Symmetrix communicated H1`.

A second source-level attribution found that the remaining bounded local
high-water increase was the geometry workspace's unconditional `2x` growth.
Prepared graph admission now prices the actual geometry capacity. Both speed
and capacity execution retain geometric headroom when it fits, narrow to the
exact edge count when only exact storage fits, and release the old geometry
generation before allocating the replacement. `low_memory=False` retains the
speed execution policy and fails early if its exact working set does not fit;
it never switches to capacity execution. MH-1 uses the same guarded geometry
growth and release-before-replacement contract.

`neighbor->ago == 0` remains part of the rebuild decision. Counts and feature
fingerprints alone do not prove that edge connectivity or ordering is
unchanged, so bypassing a legitimate run-boundary graph rebuild would be
incorrect.

## Build and artifacts

The final local qualification build used LAMMPS 4 Jul 2026, CUDA 13.3,
Kokkos CUDA+Serial, `Kokkos_ARCH_BLACKWELL120`, GCC 15.2, and OpenMPI 5.0.10
on an RTX 5090 (SM120, driver 610.43.02, 32,607 MiB).

| Artifact | SHA-256 |
|---|---|
| Final LAMMPS executable | `f398d40739b858ad69f7d8b77bd068ba446625c797e6caae17c15ec81900833c` |
| OMAT-0-medium Al/N compact model | `cd0fde722f21125a7640f4e8a9c45495d36b6a1463a379583ed9c71e949b89b2` |
| R1 CUDA cubin | `40994f57ba484cd31c58e604d1db1d3b7ed549ebbd996be18108b894aa2efdad` |
| Fresh CUDA extension | `dbc3a701a5669762941046aaf5c70e428455067502e39c67ff0b553b621db191` |
| Fresh Serial extension | `fe9e4e525de451c5c5f81e88b2f4434bacbfa8c2347f6fe055050f46b5cff428` |

The reused cubin manifest calls the file generation 3, while current runtime
diagnostics identify its compatible execution contract as
`jit-r1-gen4-f32-6fdf3e63624a0e45`. No JIT source changed in this lifecycle
fix; the different names describe cache-generation and execution-contract
metadata, respectively.

## Local CUDA/MPI qualification

The local workload used 32,768 Al/N atoms, two MPI ranks sharing one RTX 5090,
FP32 direct low-memory execution, cutoff 6.0 A, skin 0.5 A, and effective
cutoff 6.5 A. Three identical 20-step trajectories differed only in LAMMPS
run-block partitioning. JIT and initial setup were warm before timed dynamics.

| Run partition | Final directed edges | Total timed s | us/atom/step | atoms/s | sampled peak GPU memory |
|---|---:|---:|---:|---:|---:|
| `run 20` | 3,656,704 | 3.125700 | 4.7694 | 209,668 | 7,001 MiB |
| `run 2 post no; reset_timestep 0; run 18` | 3,699,308 | 3.201226 | 4.8847 | 204,722 | 8,691 MiB |
| twenty one-step blocks | 3,682,042 | 3.178326 | 4.8497 | 206,197 | 8,745 MiB |

The two-block and twenty-block cases are 2.4% and 1.7% slower than the
monolithic control, respectively. Every short block pays LAMMPS setup overhead,
and the split graphs contain up to 1.2% more directed edges than the
monolithic final graph.

LAMMPS reports the following Pair/Comm averages. For split inputs, LAMMPS only
prints the detailed breakdown for the final block.

| Run partition and reported block | Pair | Comm |
|---|---:|---:|
| Monolithic 20 steps | 3.0845 s | 0.035445 s |
| Split final 18 steps | 2.8411 s | 0.038440 s |
| Twenty-block final one step | 0.15860 s | 0.00044681 s |

The first run boundary raises the observed high-water mark by 1,690 MiB. The
speed bundle uses 492 geometry bytes per directed edge, so doubling the initial
3,656,704-edge capacity prices 1,715.8 MiB of additional geometry. This GPU had
ample free memory after the 5% reserve, so the new contract intentionally kept
that speed-oriented geometric headroom. The observed increase is consistent
with the priced capacity rather than an untracked allocation. Twenty boundaries
do not reproduce the old full-state duplication: peak memory reaches 8,745
MiB, only 54 MiB above the two-block peak. Memory remains bounded by the maximum
selected graph/workspace capacity rather than increasing per `run` command.

Focused synthetic-memory tests cover the boundary behavior that this 32 GB
workload does not trigger. They verify that `low_memory=False` keeps speed
execution while narrowing to exact edge capacity and reuses an existing
capacity without pricing a second doubling. When its estimate exceeds the
advisory budget, it still attempts exact speed-path allocation. FP32/FP64
`low_memory=True` tests likewise verify that an over-budget minimum estimate
selects exact capacity without rejecting graph preparation. The MH-1 matrix
independently verifies exact growth, failure before releasing the current
workspace, and subsequent capacity reuse for three model shapes.

## Numerical qualification

Focused MPI tests compare monolithic and split trajectories by atom ID and
type, minimum-image coordinates, forces, per-atom energies, rank ownership,
global energy, and all six stress components. They cover:

- `run 2 post no; reset_timestep 0; run 10` versus `run 12`;
- twenty `run 1` blocks versus `run 20`;
- direct low-memory ownership migration and graph replacement;
- generic/materialized ownership migration and graph replacement.

All four final-binary CUDA/MPI tests pass. The final generic/materialized FP32
migration case was rerun explicitly with `SYMMETRIX_LAMMPS_KOKKOS_GPUS=1` on
the final executable. In the larger manual trajectories,
the earlier full-state comparison found maximum coordinate and force
differences of `1.68e-7 A` and `1.65e-5 eV/A`, respectively. For the final
20-step timing rerun, the split-versus-monolithic final energy difference is
`1.33e-4 eV`, and the largest pressure-component difference is
`2.19e-3 bar`.

The affected fresh Serial test matrix also passes. Its initial 41 failures
were solely attempts to use an unwritable default JIT cache; rerunning those
tests with a task-local cache produced no failures.

## Two-A100 CUDA-aware acceptance

The final production-style gate used two A100-SXM4-80GB GPUs, one CUDA-aware
OpenMPI 4.1.6 rank per GPU, and a `1x1x2` decomposition. All runs used the same
Sr/Ti/O FP32 direct low-memory MACEField model, a 6.0 A model cutoff, a 0.5 A
skin, and a 6.5 A effective neighbor-list cutoff. Every fresh trial executed
separate `run 0`, `run 1`, and `run 2` commands. LAMMPS emitted no `Turning off
GPU-aware MPI` warning, and process teardown returned both devices to the
4 MiB baseline.

| Repeat | Atoms | Active 6.0 A directed edges | Retained 6.5 A candidates | Outcome | Final 2-step us/atom/step | sampled peak MiB/GPU |
|---|---:|---:|---:|---|---:|---:|
| n74 trial 1 | 2,026,120 | 149,932,880 | 208,285,136 | success | 4.790486 | 72,871 / 72,871 |
| n74 trial 2 | 2,026,120 | 149,932,880 | 208,285,136 | success | 4.789104 | 72,871 / 72,871 |
| n75 trial 1 | 2,109,375 | 156,093,750 | 216,843,750 | success | 4.794406 | 75,749 / 75,907 |
| n75 trial 2 | 2,109,375 | 156,093,750 | 216,843,750 | success | 4.791917 | 75,749 / 75,907 |
| n76 trials 1 and 2 | 2,194,880 | 162,421,120 | 225,633,664 | deterministic preflight | n/a | 6,731 / 6,731 |

n74 initially planned `1,015,593` receivers, `1,225,513` feature nodes, and
`104,402,925` retained candidates per rank from active values near `1,013,060`,
`1,222,456`, and `104,142,568`. Both ranks remained inside that capacity after
ownership changed. Graph updates reached two while graph, geometry, result,
M0, and communicated-H1 allocation counts remained one. The M0-adjoint
allocation count remained zero and the M0 alias was restored after reverse.

n75 is a deliberate revision to the plan's expected boundary. Rank zero grew
beyond its initial receiver/edge allowance, so the alias-safe exceptional path
released the old owners before allocating one successor generation. Its graph,
geometry, result, and M0 replacement counts became two and its M0 alias-detach
count became one; the standalone M0-adjoint allocation count remained zero.
Rank one stayed inside its initial plan. Both independent trials completed with
identical lifecycle counters, showing that exceptional replacement is viable
without weakening the reserve.

n76 is the new deterministic boundary. Each trial rejected the graph before a
large allocation because `74,792,897,336` required bytes exceeded
`73,679,994,880` available bytes after the `4,249,387,008`-byte reserve. The
limiting rank retained `112,816,832` candidate edges. Both attempts produced
the same capacity error rather than a CUDA allocator failure or illegal
address.

The final Pair/Comm measurements were `9.6476/0.015091 s` and
`19.387/0.011014 s` for n74 trial 1's one- and two-step blocks, and
`9.6473/0.015051 s` and `19.378/0.015653 s` for trial 2. For n75 they were
`10.039/0.016221 s`, `20.196/0.013973 s`, `10.055/0.020336 s`, and
`20.183/0.019018 s`, respectively. Pair includes the model evaluation and may
also absorb synchronization around MPI staging or ownership boundaries.

## Standard MACE two-A100 boundary

A second campaign used the standard OMAT-0-medium Al/N model on the same two
A100-SXM4-80GB GPUs, CUDA-aware OpenMPI 4.1.6 provider, current extension and
LAMMPS binary, FP32 direct low-memory policy, 6.0 A cutoff, 0.5 A skin, and
`1x1x2` rank grid. The periodic orthorhombic wurtzite-AlN cell has eight atoms,
91 exact-cutoff directed edges/atom, and 109 retained candidates/atom. Each
fresh process executed separate `run 0`, `run 1`, and `run 2` commands.

The instance currently exposes driver 535.129.03 while the qualified candidate
links CUDA 13. The runs therefore prepend the installed NVIDIA data-center
forward-compatibility directory `/usr/local/cuda-13.0/compat` to
`LD_LIBRARY_PATH`. The unchanged candidate then loads and runs normally; no
binary was rebuilt for this campaign.

| Trial | Repeat | Atoms | Active 6.0 A edges | Retained 6.5 A candidates | Outcome | Final 2-step us/atom/step | atoms/s | sampled peak MiB/GPU |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | n65 | 2,197,000 | 199,927,000 | 239,473,000 | success | 4.067660 | 245,841 | 72,181 / 72,183 |
| 2 | n65 | 2,197,000 | 199,927,000 | 239,473,000 | success | 4.055826 | 246,559 | 72,181 / 72,183 |
| 1 | n66 | 2,299,968 | 209,297,088 | 250,696,512 | deterministic preflight | n/a | n/a | 7,133 / 7,133 |
| 2 | n66 | 2,299,968 | 209,297,088 | 250,696,512 | deterministic preflight | n/a | n/a | 7,133 / 7,133 |

At n65 each rank initially owns 1,098,500 receivers and 119,736,500 retained
candidate edges. The admitted plan is 1,101,247 receivers, 1,316,834 feature
nodes, and 120,035,842 candidates. NVT motion increases the final rank-local
candidate counts to 121,556,017 and 121,569,577, beyond the initial 0.25% edge
allowance. Both ranks consequently perform one release-before-allocation graph,
geometry, and result replacement. M0 remains inside its receiver capacity:
M0 replacements remain one, standalone M0-adjoint allocations remain zero,
and its reuse alias is active after reverse. Both trials report the same
lifecycle counters and zero dangerous neighbor builds.

The final one-step Pair/Comm times are `8.7316/0.011700 s` and
`8.7472/0.009195 s`; the final two-step values are `17.756/0.096737 s` and
`17.705/0.101660 s`. n66 rejects before a large allocation in both trials:
`73,679,497,632` required bytes exceed `73,258,467,328` available after the
`4,249,387,008`-byte reserve. The limiting rank has 125,348,256 retained
candidates. Neither endpoint emits a GPU-aware MPI fallback warning, and both
devices return to 4 MiB after every process.

These n66 and MACEField n76 failures were measured with the preceding candidate,
which treated the 5%/512 MiB reserve as a hard preflight gate. Final review
realigned the contract: memory estimates now select the speed/capacity policy
and suppress optional slack, but do not reject execution. The exact-capacity
allocation is attempted and the backend allocator determines feasibility. The
historical outcomes above remain useful lower-bound evidence, but are not the
final capacity boundary for the revised policy.

The standard-MACE boundary is not a chemistry-matched performance comparison
with the SrTiO3 MACEField result. It establishes that repeated-run capacity
management also works for the ordinary MACE evaluator, including exceptional
growth near its own model- and topology-specific memory boundary.

The retained JSON record SHA-256 values are:

- n65 trial 1: `180ebe3af48fdb7fcf5cfeeb91049898cffe5c620f82d48c6be5b1dc76bec229`;
- n65 trial 2: `bf44ff3a51dd8fe478cb49dfa1e42f70a46d2fced41b83079c1a99ca1e244e5b`;
- n66 trial 1: `48d5c66e1ee610fb63d123016fd122e2896341d7236db1d038e93f755f3e10d3`;
- n66 trial 2: `0685b01735d31d8476eab44dcfa3eb6fb29f2bac99754c50bb2690ca76453225`.

## Final remote artifacts

The candidate remained isolated below
`/dev/shm/symmetrix-repeated-capacity-v1`; `/opt/symmetrix` was not replaced.

| Artifact | SHA-256 |
|---|---|
| Source baseline commit | `82e3acc8c27acd5d24d4385f4f466f24ad32afa8` |
| Candidate source patch against baseline | `20b40460fb3be062cf645a02bd45170c7c215ae7664c1c5a3197ec8c4f16569c` |
| SM80 Python extension | `1c773b15f126b26da339e526706147a58228a7c18707aa72571f546621fc2dad` |
| CUDA/MPI LAMMPS executable | `764b2353a400dacb173cac1a6330d64f782b236228476e6a5aeed5adb8ef0b9c` |
| Sr/Ti/O compact MACEField JSON | `9b1d0f825d37381740afcfb5e51c41fa16de118a62e780a3f7935c3e831fb323` |
| Generation-4 SM80 R1 cubin | `220edd81a30a988560e5e83bdb93a6c7cd1a2c59cddbd6e2a8e1f9479b98a257` |
| OMAT-0-medium Al/N compact JSON | `ec3a81da0d4aac597f7cb249d5743c60c81215f28da7725382c0f3ff1ee0a54f` |
| OMAT-0-medium SM80 R1 cubin | `8e333dafe310c32736992b5cdb8efb9a5da3040ed2cf81739402d17ecdf1235f` |
| Standard-MACE boundary runner | `1bce9cb5fd2502da08b10c6140fe121bd5cab97ac0341c255f88fc821622c64e` |

The regenerated compact JSON is byte-identical to the historical production
model. Its source checkpoint SHA-256 is
`f92e043aaf2cd8879919db8452503553fe7b608cb749d8d169dd96d4aa094aa2`.
The final capacity records are retained under
`.task-build-repeated-capacity-lammps-20260831/remote-results/`.
Standard-MACE records are retained beside them under
`.task-build-repeated-capacity-lammps-20260831/remote-results-standard/`,
including the exact executed runner whose hash is listed above.

## Remaining qualification limits

The accepted gates cover same-node FP32 CUDA-aware MACEField and standard MACE
with two A100s. FP64 CUDA MPI, HIP MPI, other MPI providers, cross-node
GPU-aware transport, and eight-rank capacity-boundary execution remain separate
qualification targets. Symmetrix reports rank-local allocation lifecycle
counters, but it does not independently count bytes transferred by the MPI
provider.
