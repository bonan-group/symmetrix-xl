# Generated R1 reverse edge tiling

## Scope

This experiment evaluated source-edge parallelism in the shared CUDA/HIP RTC
renderer. The CSR-tiled controls consume the existing source-major
`source_offsets/source_edges` schedule directly. The packed candidates implement
the originally proposed precomputed `source_tile_offsets`, `source_edge_tiles`,
and `source_edge_masks` schedule. The experimental implementation made the
generated reverse strategy and block size part of artifact metadata, launch-plan
identity, the cache key, and runtime diagnostics. It was reverted after the
results below; production retains the original 64-thread serial source policy.

The available production checkpoint is the 128-channel MH-0 model, not the
32-channel target that motivated the experiment. The 32-channel automatic
specialization is therefore compile-qualified but not performance-qualified.

## HIP workload

- GPU: Radeon 8060S, `gfx1151`, wave32
- Model: `/tmp/mace-mh-0-current-Al-N.json`, SHA256
  `c202e5bc6da9a2b0ca4974595869fa5f4f0cdb7f37b58d87415ecd95f048eb22`
- Model cutoff: 6.0 A
- Neighbor-list skin: 0 A; effective cutoff: 6.0 A
- Structure: 864-atom `6x6x6` wurtzite AlN
- Directed edges: 78,624
- Precision: FP32
- Execution: low-memory direct path and hipRTC generated R1. Final ABI-matched
  controls use 20 warmups and 50 measured calls; initial rejected variants used
  10 warmups and 20 measured calls.

## Results

| Reverse strategy | Threads | Median (us/atom) | Range (us/atom) | Result |
|---|---:|---:|---:|---|
| `source_serial_64` | 64 | 22.751 | 22.612-23.013 | 128-channel control |
| `source_serial_32` | 32 | 24.016 | 23.766-24.193 | 5.6% slower on 128 channels |
| `source_edge_tiled8`, path group 5 | 64 | 38.643 | 38.419-38.853 | Rejected: spills |
| `source_edge_tiled16`, path group 5 | 128 | 39.907 | 39.394-40.824 | Rejected: spills |
| `source_edge_tiled8`, path group 1 | 64 | 31.204 | 30.312-32.472 | Rejected for 128 channels |
| `source_edge_tiled16`, path group 1 | 128 | 35.065 | 34.357-36.038 | Rejected for 128 channels |
| `source_edge_packed8`, path group 1 | 64 | 29.052 | 28.547-29.545 | 6.9% faster than CSR tiled-8 |
| `source_edge_packed16`, path group 1 | 128 | 34.125 | 33.551-34.720 | 2.7% faster than CSR tiled-16 |

The path-group-1 renderer removes the tiled kernel's spills. hipRTC code-object
metadata for tiled-8 changes from 256 VGPRs, 92 VGPR spills and a 364-byte
private segment per thread to 195 VGPRs with no spills or private segment. The
serial64 control uses 230 VGPRs without spills.

`rocprofv3` measured the original tiled-8 reverse at 17.570 ms per launch,
versus 3.462 ms for serial64, or 5.08x slower. Scratch allocation was 14.38 MiB
for tiled-8 versus 1.56 MiB for the full serial-control process. The profiler
reported 4,096 B LDS for tiled-8 and 2,560 B for serial64. This MH-0 R1
contract has four source harmonics, so the generated static shared partials are
4,096 B for tiled-8 and 2,072 B for serial64; LDS capacity is not the limiting
resource.

The remaining 128-channel regression after eliminating spills comes from the
ownership algorithm: every width-8 lane traverses 16 channel slices, with
repeated path dispatch, shuffle reduction, and shared accumulation. It does not
recover enough edge parallelism to offset that work.

The literal packed schedule does reduce indexing work. Paired one-evaluation
`rocprofv3` traces measured the path-group-1 fused reverse at 8.931 ms for
packed-8 and 9.788 ms for CSR tiled-8, an 8.8% kernel reduction. Both kernels
used 200 VGPRs, 4,096 B LDS, and no scratch allocation. The improvement is real,
but the 29.052 us/atom packed-8 end-to-end result remains 27.7% slower than the
22.751 us/atom serial64 control on this 128-channel model.

## Packed schedule memory

Only the selected packed width is allocated, and the schedule is reused while
the graph topology remains unchanged. No geometry, harmonics, gradients, or
node features are duplicated.

| Strategy | Tiles | Flattened slots | Bytes | B/atom | B/directed edge |
|---|---:|---:|---:|---:|---:|
| packed-8 | 10,368 | 82,944 | 380,168 | 440.01 | 4.835 |
| packed-16 | 5,184 | 82,944 | 359,432 | 416.01 | 4.572 |

The byte count includes 64-bit per-source tile offsets, 32-bit flattened edge
indices, and one 32-bit validity mask per tile. Tile counts and flattened
allocation arithmetic use 64-bit checked indexing.

## Correctness and selection

On the same 864-atom graph, the retained path-group-1 variants versus serial64
produced:

- tiled-8: 0 eV energy, `2.1400e-6 eV/A` maximum force, and
  `1.8751e-8 eV/A^3` maximum stress difference
- tiled-16: 0 eV energy, `1.9005e-6 eV/A` maximum force, and
  `2.1832e-8 eV/A^3` maximum stress difference
- packed-8: 0 eV energy, `2.1400e-6 eV/A` maximum force, and
  `1.8751e-8 eV/A^3` maximum stress difference
- packed-16: 0 eV energy, `1.9005e-6 eV/A` maximum force, and
  `2.1832e-8 eV/A^3` maximum stress difference

The numerical reordering is within the existing FP32 gates, but no tested
alternative outperformed the original 64-thread serial source policy. The
experimental strategy selector, alternate kernels, packed schedule storage, and
ABI additions were therefore removed. A real 32-channel checkpoint would be
required before reconsidering source-edge tiling for that distinct model shape.

CUDA 13.1 NVRTC compile qualification for SM120 produced cubins for
`source_serial_32` (137,144 B), `source_edge_tiled8` (141,392 B), and
`source_edge_tiled16` (141,752 B). The appended packet and packed kernels also
compile-qualified as `source_edge_packed8` (141,824 B) and
`source_edge_packed16` (142,536 B). This machine has no CUDA GPU, so CUDA
execution, numerical parity, resource use, and performance remain unqualified.
