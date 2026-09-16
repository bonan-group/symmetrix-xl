# GPU spill-reduction milestone

## Scope

This milestone adds compiler-resource guardrails, fuses standard MH-0 ZBL
value/gradient evaluation, splits the generated MH1 conditioner reverse by
network, and replaces the misleading CUDA-80 HIP policy record with a neutral
GPU policy bound to the real target ID. CUDA and HIP continue to render the
same numerical program; only their RTC compilation and module launch adapters
differ.

No accepted change adds atom- or edge-scaled workspace. The conditioner split
adds one physical reverse launch per interaction and runs both launches on the
existing evaluator stream.

## Resource guardrail

`benchmarks/gpu_kernel_resources.py` normalizes these compiler reports:

- HIP: `llvm-readobj --notes artifact.hsaco`
- CUDA: `cuobjdump --dump-resource-usage artifact.cubin`

It supports one or more `--match` regular expressions and exits with status 1
when a selected kernel exceeds a supplied `--max-*` limit. CUDA's report does
not expose separate VGPR/SGPR spill counts, so those fields are recorded as
unknown; stack and local bytes remain enforceable.

Example:

```bash
python benchmarks/gpu_kernel_resources.py hip notes.txt \
  --match 'conditioning_reverse' \
  --max-private-bytes 0 --max-vgpr-spills 0
```

## Standard MH-0 ZBL

The ZBL kernel now evaluates screening exponentials, envelope value, and both
derivatives once per edge. Model-sized pair constants replace repeated powers
of atomic numbers. The fixed pair tables are independent of atom and edge
count.

Qualification used FP32 Kokkos HIP with hipRTC, generated factorized execution,
M1 tile 16, MH-0 adjoint reuse, and compact FP32 direction/FP64 radius geometry.
The structure was a `30x30x30` wurtzite AlN cell: 108,000 atoms, 9,828,000
directed edges, model cutoff 6.0 A, skin 0 A, and effective cutoff 6.0 A.

| Measurement | Before | After |
|---|---:|---:|
| End-to-end | 27.516 us/atom | 26.499 us/atom |
| ZBL kernel | 1.160 us/atom | 0.175 us/atom |
| Private bytes/thread | 8 | 0 |
| VGPR spills | not retained | 0 |

The new compact ZBL kernel uses 60 VGPRs and 107 SGPRs on gfx1151. Its metadata
reports 18 SGPR spills but no private segment, so they do not create scratch
traffic. Energy and maximum force magnitude matched the retained large-cell
record, and the focused Kokkos ZBL tests pass.

## MH1 conditioner reverse

Qualification used the official MH1 Al/N checkpoint, FP32 Kokkos HIP, hipRTC,
full-retention factorized execution, and a `6x6x6` wurtzite AlN cell: 864 atoms,
78,624 directed edges, model cutoff 6.0 A, skin 0 A, and effective cutoff 6.0 A.

The retained single reverse kernel inlined the conditioned-prefix and density
networks together. The new RTC module emits two same-stream kernels so their
64-wide reverse states are never simultaneously live. Each kernel accumulates
only its ten radial derivatives into existing storage.

| Measurement | Single kernel control | Split networks |
|---|---:|---:|
| Median | 176.751 us/atom | 171.470 us/atom |
| Median step | 152.712 ms | 148.150 ms |
| Sample range | 152.554-153.271 ms | 147.539-148.468 ms |

Both runs used the same 864-atom/78,624-edge graph. The split result used five
warmups and 15 measured evaluations. Energy was `-6423.654872930765 eV`; force
L2 was `3.1685264253 eV/A`; no factorized fallback occurred.

Per interaction, the old reverse kernel used 976 private bytes, 80 VGPR spills,
and 383 SGPR spills. The split resources are:

| Kernel | Private bytes | VGPR spills | SGPR spills |
|---|---:|---:|---:|
| Conditioned prefix | 912 | 34 | 116 |
| Density | 20 | 4 | 194 |

The split does not eliminate prefix scratch, but it reduces pressure and is 3.0%
faster end to end on the matched current-branch control. The schedule ID was
versioned so an older one-symbol RTC cache entry cannot be loaded as the new
two-symbol module.

## Deferred kernels

Standard M0 reverse is only 0.492 us/atom. Its 16-input generated polynomial
reverse has 36 private bytes, but splitting it would add complete node/channel
passes for a small absolute ceiling. It remains unchanged pending evidence for
an in-kernel live-range reduction.

The dormant generic R1 edge fallback is not launched by the qualified MH-0 or
MH1 RTC paths. Its large spill frame remains visible to the resource guardrail,
but changing it cannot improve current end-to-end timing. It should be removed
or separately specialized only when a supported execution contract still needs
that fallback.
