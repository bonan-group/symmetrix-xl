# Native Kokkos ASE neighbor graph, 2026-09-10

## Scope

This qualification covers the first native Kokkos cell-list implementation for
the ASE calculator. It preserves the `neighbor_skin` Verlet policy and prepares
receiver-major direct-execution topology without materializing edge arrays in
Python. The implementation supports fully periodic orthorhombic and triclinic
cells. Partial and non-periodic cells retain the host builder.

The design follows the useful parts of LAMMPS Kokkos full-bin neighbor lists:
spatial bins, a conservative neighboring-bin stencil, device-side membership,
and receiver-parallel neighbor enumeration. It does not reuse LAMMPS atom
storage, ghost ownership, MPI communication, page allocators, half-list rules,
or dynamic per-atom neighbor capacity. Symmetrix instead performs an exact
64-bit count and prefix scan followed by an exact-sized fill of its directed
full list. Counts are narrowed to int32 graph IDs only after the existing graph
cardinality guard succeeds.

Bin members are ordered by atom index with a bin-local device sort. This makes
identical rebuilds deterministic without restoring the former edge-sized
Python `lexsort`. Eight identical 1,080-atom OpenMP rebuilds produced zero span
in energy, forces, and stress.

## Workload

- Model: MPA-0 medium compact JSON
- Model SHA256: `d9db17c40b6cd22ac7cb1b00072157ebaf4e79856452d7848017341a45463fa8`
- Structure: rattled cubic SrTiO3 supercell
- Precision and execution: FP32, generated direct-speed
- Model cutoff: 6.0 A
- Neighbor skin: 0.5 A
- Effective candidate cutoff: 6.5 A
- Properties: energy, forces, stress
- Timing: median after two or three warmups; graph reuse and forced rebuild are
  measured separately

The maintained command is:

```bash
python benchmarks/ase_neighbor_graph_benchmark.py \
  --model /path/to/compact.json \
  --neighbor-backend kokkos \
  --supercell-repeat 10 \
  --warmups 2 --repeats 5
```

Use `--neighbor-backend host` for the matched host control.

## OpenMP results

The extension used GCC with Kokkos OpenMP and had SHA256
`219fb4ad034afa448679e7f85056a97a7aa43b2ed7654047ba74b87c4676c905`.
Runs were pinned to physical CPUs 0-7 with eight Kokkos/OpenMP threads and one
OpenBLAS thread.

| Builder | Atoms | Directed edges | Reused ms | Reused us/atom | Rebuilt ms | Rebuilt us/atom | Rebuild increment ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Host | 5,000 | 504,618 | 174.308 | 34.862 | 221.944 | 44.389 | 47.636 |
| Kokkos | 5,000 | 504,618 | 172.931 | 34.586 | 180.975 | 36.195 | 8.045 |

The native builder reduces the rebuild-specific increment by `5.92x` and the
complete rebuilt ASE call by `1.23x`. Cache-hit time changes by less than 0.8%.
The corresponding rebuilt throughputs are approximately 22,528 atoms/s for
the host builder and 27,628 atoms/s for the Kokkos builder.

At 1,080 atoms on the same eight cores, the final deterministic implementation
measured 48.633 us/atom for a host-rebuilt call and 37.086 us/atom for a native
rebuilt call. Cache-hit results were 34.520 and 34.438 us/atom, respectively.

An OpenMP MACEField comparison on a 40-atom SrTiO3 graph covered energy,
forces, stress, polarization, and polarizability. Both builders produced 4,046
directed edges. Energy, polarization, and polarizability were identical; the
maximum force and stress differences were `1.33e-15 eV/A` and
`3.47e-18 eV/A^3`. BEC remains intentionally routed through the host builder.

### CPU automatic-selection qualification

A follow-up on the capacity-preserving implementation used a fresh OpenMP
extension (SHA256
`8290a917da92a822911c4855cae4e7be9ba95ac07fbedae75d213308e4c11426`) and
the universal MACEField MH-0 model (SHA256
`cde5c40a4ea8c8db852225eb8e4f25ac2fd4f3b4b676bf03f31b49880b41f495`).
The model cutoff, skin, effective cutoff, structure, precision, and properties
matched the workload above. Host and Kokkos runs alternated in one process to
control evaluation drift. The primary one-physical-core results were:

| Builder | Atoms | Directed edges | Rebuilt us/atom | Graph preparation us/atom |
|---|---:|---:|---:|---:|
| Host | 320 | 32,328 | 231.543 | 13.049 |
| Kokkos | 320 | 32,328 | 233.335 | 16.462 |
| Host | 625 | 63,122 | 226.000 | 8.837 |
| Kokkos | 625 | 63,122 | 224.839 | 10.241 |
| Host | 1,080 | 109,028 | 230.646 | 12.875 |
| Kokkos | 1,080 | 109,028 | 231.334 | 15.989 |

The complete call varied by less than 0.8% and did not establish a stable CPU
crossover. Isolated graph preparation remained slower with Kokkos. A paired
eight-core scan confirmed that result through 20,480 atoms: at 5,000 atoms the
host and Kokkos preparation costs were 8.897 and 9.940 us/atom, respectively;
at 20,480 atoms they were 10.429 and 11.278 us/atom. The eight workers were
pinned to physical CPUs 0-7, with one BLAS thread.

Automatic selection therefore stays on the host builder for Serial and
OpenMP. `SYMMETRIX_NEIGHBOR_BACKEND=kokkos` remains available as an explicit
CPU override. This avoids using a CUDA-derived crossover for a CPU execution
space where the current implementation does not show the same gain.

## CUDA results

Before deterministic bin ordering was added, the RTX 5090 result at 5,000
atoms and 504,618 directed edges was:

| Builder | Reused us/atom | Rebuilt us/atom | Rebuild increment ms |
|---|---:|---:|---:|
| Host | 3.065 | 11.826 | 43.802 |
| Kokkos | 2.931 | 4.032 | 5.506 |

This was a `2.93x` reduction in total rebuilt-call time and a `7.96x` reduction
in rebuild-specific overhead. FP32 host/native comparison gave identical
energy, force differences of a few micro-eV/A, and stress differences around
`1e-8 eV/A^3`; FP64 agreed near machine precision. MACEField energy, forces,
stress, polarization, and polarizability also passed.

After the device was restored, a fresh CUDA 13.3 `BLACKWELL120` build qualified
the deterministic implementation. The extension SHA256 was
`70fa19eccb0b93ba0761489d24c355ac870101e4601606794982068149aade35`.

| Builder | Atoms | Directed edges | Reused us/atom | Rebuilt us/atom | Rebuild increment ms |
|---|---:|---:|---:|---:|---:|
| Host | 5,000 | 504,618 | 3.062 | 11.994 | 44.662 |
| Kokkos | 5,000 | 504,618 | 3.018 | 4.214 | 5.980 |
| Host | 40,000 | 4,036,706 | 3.117 | 12.713 | 383.849 |
| Kokkos | 40,000 | 4,036,706 | 2.954 | 3.152 | 7.902 |

The 40,000-atom native rebuilt call sustained approximately 317,265 atoms/s.
A matched 320-atom host/native calculation gave zero energy difference, a
maximum force difference of `3.77e-15 eV/A`, and a maximum stress difference
of `3.47e-18 eV/A^3`.

### Small-cell automatic selection

The CUDA builder has a small-cell floor because each receiver owns one GPU
thread. In a 320-atom `4x4x4` SrTiO3 cell, the three bin counts are `2x2x2`
while the conservative stencil spans `5x5x5` entries. Periodic wrapping makes
each receiver revisit the eight physical bins through multiple images, and the
edge-count and edge-fill kernels do not expose enough parallelism to fill the
GPU. Nsight Systems attributed about 5.48 ms per rebuild to those two kernels,
which accounts for nearly all of the 6.05 ms native graph increment.

Matched CUDA 13.3 `BLACKWELL120` measurements with the normal two-layer MACE
checkpoint locate the crossover between 320 and 625 atoms:

- Model: `maceomat0smallmodel`, SHA256
  `0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5`
- Extension SHA256:
  `63a41ee865cfe3070a77f50b76bd4916daf6779362b0ea02bdb6b532e7832b97`
- Model cutoff: 6.0 A; neighbor skin: 0.5 A; effective cutoff: 6.5 A

| Builder | Atoms | Directed edges | Rebuilt us/atom | Rebuild increment ms |
|---|---:|---:|---:|---:|
| Host | 320 | 32,328 | 18.173 | 4.17 |
| Kokkos | 320 | 32,328 | 23.867 | 6.05 |
| Host | 625 | 63,122 | 12.250 | 5.54 |
| Kokkos | 625 | 63,122 | 10.648 | 4.63 |
| Host | 1,080 | 109,028 | 15.989 | 13.93 |
| Kokkos | 1,080 | 109,028 | 8.865 | 6.39 |

The accelerator `automatic` policy therefore uses the host builder below 512
atoms and the Kokkos builder at 512 atoms or above. Explicit `host` and
`kokkos` requests still pin their respective implementations. Atom count is an
intentionally simple proxy for the amount of available parallelism; unusually
anisotropic cells can still be compared with the explicit overrides.

## Rebuild peak-memory policy

The first implementation returned edge-sized `sources` and FP64 fractional
geometry and then copied them into evaluator-owned storage. At 40,000 atoms
this raised sampled process VRAM from the 6,386 MiB evaluation plateau to
6,500 MiB during rebuilding. The measured 114 MiB increment agreed with its
`140*N + 12*B + 28*E` temporary-allocation model.

Production graph construction now follows the capacity-preserving part of the
LAMMPS Kokkos design: exact counting happens first, after which the builder
writes sources and fractional geometry directly into grow-only evaluator
capacity. The public graph-export helper still owns its returned arrays. The
same 40,000-atom run now peaked at 6,390 MiB, only 4 MiB above the evaluation
plateau. Rebuild time remained within run-to-run noise.

Consequently, normal production rebuilds no longer allocate temporary storage
proportional to the directed-edge count. Their remaining builder workspace is
proportional to atoms and spatial bins. This prevents device neighbor-list
construction from materially reducing the maximum simulatable graph size.
