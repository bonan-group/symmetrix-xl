# OMAT-0 reference CUDA capacity

Date: 2026-08-23

## Scope

This fresh-process campaign adds measured capacity brackets for the three
reference implementations used by the streamed-edge milestone report:

- original Symmetrix `origin/develop` materialized Kokkos;
- PyTorch MACE/e3nn; and
- PyTorch MACE/cuEquivariance.

Every worker used the OMAT-0 medium checkpoint, FP32, the `default` head, and
ASE energy, forces, and stress. Structures are deterministically perturbed
periodic wurtzite AlN with a 6.0 A exact neighbor cutoff and zero skin. The
successful and immediately larger failed repeat were each observed in two
fresh processes, counting the existing 4,000-atom milestone record as the first
e3nn success. Each new successful worker used one warmup and one measured call;
capacity timing is not used as a throughput comparison.

## Environment

- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB, compute capability 12.0
- Driver: 610.43.02; no competing compute process before the campaign
- Python 3.12.13; PyTorch 2.13.0+cu130; MACE-Torch 0.3.15
- cuEquivariance, cuEquivariance-Torch, and CUDA operators: 0.11.1
- Checkpoint SHA-256:
  `d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a`
- Origin/develop extension SHA-256:
  `65a3e2c4a1e76266b33a7988a76d457a1b1915b17023e83d26496940dfdb26d1`
- Benchmark driver SHA-256:
  `12f35d42bc9f765f7fd1332e037cde4c726ad99244b430644aaaa331a7d849f1`

## Capacity boundaries

| Implementation | Largest demonstrated success | Directed edges | Boundary us/atom | Resident VRAM | First failed size | Failure |
|---|---:|---:|---:|---:|---:|---|
| PyTorch MACE/e3nn | 4,000 (`n10`) | 364,000 | 98.995 stable milestone; 120.106 repeat | 29,246-29,248 MiB | 5,324 (`n11`) | CUDA OOM, 9.24 GiB tensor-product allocation |
| PyTorch MACE/cuEquivariance | 19,652 (`n17`) | 1,788,332 | 50.256-50.914 | 30,878 MiB | 23,328 (`n18`) | CUDA OOM, 10.12 GiB `uniform_1d` reverse allocation |
| Origin/develop Kokkos | 13,500 (`n15`) | 1,228,500 | 61.510-61.647 | 30,850 MiB | 16,384 (`n16`) | Kokkos CUDA OOM, 128 MiB allocation |

The cuEquivariance `n17` successes and the e3nn `n10` repeat emit recoverable
allocator warnings before completing. Their boundary timing and allocation
state are not suitable throughput baselines. Origin/develop's primary OOM is
followed by a secondary Kokkos teardown abort; the first allocation exception
defines the failure class.

For comparison, the separate current-direct campaign uses an unperturbed
wurtzite series with a 6.0 A model cutoff and 0.5 A skin: 109 candidate edges
per atom rather than the references' 91 exact-cutoff edges per atom. Current
direct low memory reaches 389,344 atoms and 42,438,496 candidates before the
415,292-atom case fails. Its maximum demonstrated atom count is therefore
28.840x origin/develop, 19.812x PyTorch/cuEquivariance, and 97.336x
PyTorch/e3nn on this device. These ratios compare measured workload boundaries,
not a topology-normalized theoretical capacity.

## Matched 4,000-atom resident memory

The earlier steady-state milestone comparison remains the controlled
small-system memory result:

| Implementation | Directed edges/candidates | Resident VRAM | Relative to direct |
|---|---:|---:|---:|
| Current direct low memory | 452,342 candidates at 6.5 A | 1,118 MiB | 1.00x |
| Origin/develop materialized | 364,000 edges at 6.0 A | 9,612 MiB | 8.60x |
| PyTorch/cuEquivariance | 364,000 edges at 6.0 A | 7,136 MiB | 6.38x |
| PyTorch/e3nn | 364,000 edges at 6.0 A | 29,248 MiB | 26.16x |

## Raw evidence

Successful worker records remain under `/tmp` for this local campaign. Their
SHA-256 values are:

- e3nn `n10` repeat:
  `be8c1b12460060b6b6fd0bc3a2193f349970220e3e1a3a713105cde995742bed`;
- cuEquivariance `n17` trials:
  `de4910d21cc441e39c8157f5515d1192b4fc88a9f71cc32a0798f7c4e94572eb`
  and `a248071c748771a00e631c0192fb850ebb79ef83a3cd61149dd7ba5169e0d64e`;
- origin/develop `n15` trials:
  `5074bab544ba97974643e62c0e0284b1ba41e34836415894a43586ac36bc9f27`
  and `0dd3f2aea107a32c51c0b68dd84ea469efe4becb0322eeed2ccd81e5179d8e96`.

The exact invocation is the milestone driver's `run` subcommand with
`--implementation develop` or `pytorch`, `--pytorch-backend e3nn|cueq`,
`--device cuda`, `--dtype float32`, `--cutoff 6.0`, `--skin 0.0`, one warmup,
one sample, and the reported integer repeat. The process exit and first CUDA
exception classify failed workers because no JSON record is written after an
OOM.
