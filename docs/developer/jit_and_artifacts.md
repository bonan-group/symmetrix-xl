# JIT and Generated Operator Artifacts

M0, M1, R0, and Execution R1 use one build-safe generation pipeline. Build-time
precompilation and runtime JIT/RTC are adapters over the same normalization,
semantic lowering, source renderer, metadata builder, and identity functions.
There is no prebuilt-only numerical implementation. An installed artifact and
a runtime-generated artifact made from the same explicit inputs have identical
source, metadata, and `artifact_id`; origin is deliberately not part of the
identity.

The canonical identity is `symmetrix.generated-operator-identity/1`. Every
operator includes its generation revision, precision, backend, exact target,
architecture dimensions, schedule, and structure/contract fingerprints as
applicable. R0 additionally binds the compatibility and schedule revisions,
channel count, angular cutoff, derived harmonic count, and unity-versus-weighted
edge mode. The catalog repeats the exact identity and adds only physical
packaging information: binary format, package-relative path, SHA-256, ABI,
capabilities, and compiler/source provenance. Changing any identity field
produces a different artifact ID and a mismatched artifact is rejected before
execution.

## MACE-MH-1 composition boundary

Generation 19 projects every admitted MH-1 product layer onto the same
canonical M0 contract used by ordinary MACE, except that the final scalar layer
uses canonical M1. The projected module has type count one and receives an
explicit zero-filled type-index view; physical element indices never address
its agnostic coefficient table. Coefficient order, logical output-component
groups, forward contraction, and reverse contraction belong to that common
module. Density normalization, gates, conditioned interaction/message work,
product output linear, skip, readout, tensor product, and the rest of nonlinear
reverse remain in the model-specific MH-1 plugin.

The two artifact classes resolve independently. Each product layer reports its
own `prebuilt`, `cache`, or `runtime` origin, identity, schedule, target, and
hash. The surrounding MH-1 plugin reports a separate origin. An exact installed
M0/M1 binary can therefore be shared by ordinary MACE and MH-1, but a prebuilt
product module never makes the full nonlinear plugin prebuilt. An unmatched
product contract follows the normal cache/runtime path.

Current device product modules use common operator ABI v3 and production
schedule `chunk32`. The experimental `tiled128x4-v1` schedule is disabled and
unqualified. The nonlinear device module uses version 5 and launch plan version
6; the current host plugin ABI is v7; generated artifacts are JIT generation
19. Loaders fail closed on a retired
generation, ABI, module or plan version, wrong layer/stage, precision, target,
schedule, topology, or partial module set. The generation-18 monolithic MH-1
product artifact is historical benchmark evidence only and cannot be rendered
or loaded as a generation-19 oracle.

Common M0/M1 product buffers are disjoint. Device M0 uses `table` revision 1,
`chunk32` revision 2, and `tiled128x4-v1` revision 3 with capability mask
`0x1b`. Device M1 uses revisions 2, 3, and 4 respectively with capability mask
`0x0b`. Host M1 `chunk32` remains revision 2 with the host-ABI `0x1b` mask.
No device schedule advertises alias safety; loaders reject exact input/output
or input/input-adjoint aliases before launch.

Release records are generation-strict rather than compatibility records: they
must identify launch-plan schema `/6` and version 6, device module ABI 5, JIT
generation 19, and `split_phase_composition=true`. The retired generation-18
`external_product` flag is not accepted as a substitute. Each product record
also binds `coefficient_count = term_count * channels`; absent manifest fields
or incomplete component identities invalidate the record.

External product timing evidence uses two explicit benchmark commands. The
`profiled_command` is the command run under the profiler and must equal the
profiled record's `exact_command`; the later `attachment_command` includes
`--product-timings-json` and must equal the release record's `exact_command`.
They are intentionally different, so attaching a profile does not require a
profiled record to predict its own later attachment. Both commands and the
exact profiler command are hash-bound. Release validation also requires
external expectations for the full-plugin identity, binary and manifest, each
common module's runtime-manifest or installed-catalog provenance, and the
profile command, source/native/extension hashes, model, system, properties and
sample scope. Nsight product medians are accepted only from a parsed
`cuda_gpu_kern_sum` CSV whose explicitly named forward/reverse kernel rows
match the declared layer timings; a merely hashed opaque timing blob is not
evidence.

HIP uses the same canonical boundary and strict identity checks, but release
claims require an architecture-qualified HIP build and execution on accessible
AMD hardware. Source rendering or hipcc/hipRTC compilation without `/dev/kfd`
does not establish runtime correctness or performance.

## Resolution and policy

For an operator that needs specialization, resolution is exact and ordered:

1. Validate the native package's catalog and try the exact installed artifact.
2. Validate and load an exact artifact from `SYMMETRIX_JIT_CACHE`.
3. Compile the same canonical request with C++20, NVRTC, or hipRTC, publish it
   to the cache, then validate and load it.
4. Use generic execution only when the selected execution mode permits it.

That ordering describes the generated-artifact resolver, but is not yet
universal for every admitted low-memory M0/R0 node. During shadow
qualification, an eligible node with a frozen built-in first tries an exact
installed candidate when that stage/package mode provides one, then may select
the frozen built-in after an installed miss or rejection instead of continuing
to cache/runtime. With no installed candidate it may select the built-in
immediately. This temporary oracle exception ends only when the frozen M0/R0
retirement gates pass; it must not be presented as a cache or runtime artifact.

Installed lookup requires an exact operator identity, backend target, ABI, and
capability word; it never copies package data into the cache. A rejected
installed artifact produces an origin-specific diagnostic and runtime
resolution continues when the caller permits it. Mutable cache load failures
use quarantine and one rebuild/reload attempt. `streamed_edges="direct"`
requires its generated R1 artifact and fails closed when resolution is
exhausted. `SYMMETRIX_JIT_POLICY=required` is the default. Setting it to
`none` is a diagnostic way to disable compilation, not permission to silently
change algorithms; select `streamed_edges="non-compiled"` explicitly for the
compiler-free generic route.

The published R0 state is semantic rather than loader-specific:

- `implementation`: `generic` or `generated`;
- `backend`: `host`, `cuda`, or `hip`;
- `artifact_origin`: `prebuilt`, `cache`, `runtime`, or null when there is no
  external artifact;
- `schedule`, `profile`, and `artifact_id`: independent fields.

Do not infer implementation from a `.so`, cubin, or code-object path. In
particular, `prebuilt` and `runtime` are origins of the same generated
implementation. During the remaining R0 retirement campaign, the internal
frozen oracle reports through this public schema with a null origin and a
`frozen` schedule; its private loader route is qualification-only.

## Package modes and release policy

`SYMMETRIX_PREBUILT_OPERATOR_ARTIFACTS` and
`--prebuilt-operator-artifacts` accept three values:

| Mode | Installed generated operators | Unique artifacts per target | Intended use |
|---|---|---:|---|
| `off` | none | 0 | minimal/debug build; required stages resolve through cache/runtime |
| `r0` | verified unity-weight R0 | 6 | normal development and published-package default |
| `full` | exact R1, R0, M0, and M1 foundation bundles | 32 | opt-in deployment experiment |

The six R0 artifacts are three deduplicated channel shapes (96, 128, and 224)
in FP32 and FP64 at angular cutoff 3. Full mode adds five exact R1 and five
exact M0 bundle identities plus three scalar M1 profiles per precision. Model
aliases share identities and do not create binaries. Uncommon channel/angular
shapes, contracts outside the
verified manifest, and weighted-edge R0 use runtime generation and then the
normal cache path.

The following CPU `x86-64-v3` measurements predate the six M1 additions and
must not be used as size or build-time evidence for the current 32-artifact
full bundle:

| Metric | `r0` | `full` | Full-mode increment |
|---|---:|---:|---:|
| Installed catalog/artifact bytes | 138,014 | 1,336,832 | 1,198,818 |
| Compressed wheel bytes | 10,489,473 | 10,926,363 | 436,890 |
| Fresh wheel build wall time | 118.66 s | 123.41 s | 4.75 s |

On the five verified bundles in both precisions, full mode removed runtime
R1/M0 compilation and reduced cold calculator setup from roughly 0.7--14.2 s
to 0.04--0.33 s. The subsequent first energy/forces/stress call remained
comparable and was more than 5% slower in several larger-model observations.
No explicit release budget or per-case first-call threshold has therefore been
accepted. Publication scripts pin `r0`; changing them to `full` requires a
separate policy decision with recorded CPU, CUDA, and HIP size, build-time,
and first-call evidence.

## Integrity and cache security

Installed artifacts are immutable package data. The catalog checksum is bound
into native build information; validation rejects a symlinked package root or
data member, unsafe/absolute path, catalog or binary checksum mismatch,
unexpected or missing member, wrong byte count, target, ABI, capability, or
source/compiler provenance. `symmetrix version --probe --json` and
`symmetrix doctor --json` expose this validation and the catalog mode, count,
bytes, duration, target, and hashes.

Release benchmark evidence preserves the same distinction. Runtime-generated
common products bind the sibling manifest and binary hashes; installed common
products bind the already validated package catalog, the exact catalog-entry
hash, and the binary hash, because an installed binary need not have a sibling
manifest. Per-layer product bytes are not invented: scratch and retention are
reported by the composed MH-1 arena, while the layer record explicitly marks
that apportionment unavailable. Launch counters use a captured benchmark
window and separate warmup, timed-sample, and total deltas. Release validation
requires an explicit in-process affinity request and exact expected product
dimensions, identities, and artifact hashes.

The JIT cache is mutable executable content and has a different trust model.
New roots are created owner-only, entries are content-addressed, and manifests,
target identity, size, and SHA-256 are validated before load. Globally writable
cache and artifact directories produce a runtime warning because another user
can replace code. Production and qualification should treat that warning as a
failure, use one user-owned task/backend cache, and never use cache contents as
evidence of what a wheel shipped. Cache quarantine/rebuild is valid only for
mutable cache artifacts; package data is never modified or quarantined.

Host artifacts are model-, precision-, ABI-, generation-, and target-specific.
Automatic CUDA and HIP specialization uses NVRTC and hipRTC. Deprecated
explicit NVCC and hipcc selections remain only for compatible standard R1
artifacts. LAMMPS consumes prepared artifacts but never invokes a compiler.

Use a task- and backend-specific cache and require specialization when
qualifying it:

```bash
export SYMMETRIX_JIT_CACHE="$HOME/tmp/symmetrix-task-jit-cache"
export SYMMETRIX_JIT_POLICY=required
symmetrix_prepare_jit_host_artifact \
    --model srtio3-mace.json --precision float64
```

Production preparation includes the package's complete
`source_content_sha256` in the cache key, manifest, and `_src<hash>` artifact
basename. After rebuilding the extension, stop old processes and prepare/load
artifacts in a fresh process. Verify that the manifest source hash matches the
loaded extension and that the selected path belongs to the intended cache or
installed package root.

Prepare standard MACE or MACEField R1 device artifacts with
`symmetrix_prepare_jit_device_artifact` on the exact target GPU. The broader
cache concurrency and execution invariants are in
{doc}`/streamed_edge_execution`.

## R0 retirement status

The generated artifact pipeline is complete, but frozen R0 retirement is not.
The final shadow campaign passed all 48 CPU correctness/route cases and the CPU
aggregate performance gate (`0.979538` generated/frozen), but one FP64 weighted
small-graph stress case was `1.054858`, above the 1.05 per-case limit. CUDA
schedule parity subsequently passed all 48 performance cases at the kernel
commit, with a `0.983760` aggregate ratio and `0.996812` worst case. Two fresh
exact-source confirmation runs after an identity/catalog-only change retained
aggregate wins (`0.983635` and `0.985595`) but each had a different isolated
warm-up/timing-transition outlier that exceeded the per-case ceiling. That
pattern is consistent with cold-clock behavior, leaving both runs at 47/48. A
later exact-current generation-17 campaign passed all 48 cases with aggregate
`0.982764`, FP32 `0.956079`, FP64 `0.991949`, and worst case `0.996605`, closing
the CUDA gate. HIP is compiler/package-qualified only because the qualification
user could not open `/dev/kfd`, so its execution gates are unmeasured.
Consequently `mace_standard_r0.hpp` and frozen dispatch remain as the
qualification oracle. Do not describe the numerical implementation as retired
or the broader R0 migration as complete until the CPU and HIP gates also pass
stable campaigns and the removal is separately reviewed.
