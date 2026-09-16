# MH0/MH1 PyTorch and Symmetrix Profile Comparison

Date: 2026-08-09

Repository milestone: `475d16a Reuse MH1 message weights across nodes`

Device: Radeon 8060S (`gfx1151`), ROCm 7.14

## Workload and scope

All four cells use wurtzite AlN with `a=3.112 A`, `c=4.982 A`, repeated
`6x6x6`: 864 atoms and 78,624 directed edges. They use FP32, the `omat_pbe`
head, energy plus analytic forces, no stress, and a prebuilt graph.

Official PyTorch checkpoint hashes are:

- MH0: `d62ff8f293664e6556cfa49364b28ee72a70cf1e6150f3c111a578397fed609d`
- MH1: `a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47`

The matched Symmetrix Al/N extraction hashes are:

- MH0: `d3e825d439555bb859c2b226894f9c7ae02a15d20b4e90e6c5d6f2eb9c84a802`
- MH1: `8be929a1d3e3b8de2b181bdaca7aeef0769756f4301b5009920f795bd2bf69ae`

The native HIP extension used for the Symmetrix cells has SHA-256
`1dceb69def3f5a6380c3d31d7b43ef77bc5f2bc55708d2524cfb3ed4691caa`.

PyTorch 2.12.0+rocm7.14.0 used eager e3nn for both checkpoints. A compatible
ROCm cuEquivariance package is not installed. Batch cloning is outside the
profiled range. Kineto records one synchronized post-warmup model call and
correlates each device kernel with its launching CPU operator. MACE module
ranges attribute forward launches; launches on the autograd worker form the
reverse bucket.

Symmetrix uses a prepared native graph, `direct_jit="required"`, and hipRTC for
both models. MH0 is forced away from its checked-in static artifact. rocprofv3
selected-region tracing covers exactly one synchronized
`_compute_prepared_direct` call. Generated MH1 reports
`generated_hip_v4`.

These are single-step diagnostic profiles after three warmups, not the final
three-process 20-warmup/20-sample qualification. All four cells have canonical
physical-graph SHA-256
`a6990939929e747e577bbd8b65c8954a13f712b972e40cad9869f78f54a521b7`.
The fingerprint covers atom order, cell and positions quantized to `1e-4 A`,
and sorted `(first atom, second atom, integer periodic shift)` records. Raw
displacements are excluded because PyTorch consumes FP32 geometry while
Symmetrix receives FP64 geometry. The order-sensitive edge-record hash also
matches all cells:
`ef74d9e1e6030a13983bd5789c90cd7f12ff2af2ed1ba55e68f5d567722fd415`.
Compare MH1/MH0 within each runtime; do not treat cross-runtime absolute time
as a product benchmark.

## Whole-step result

| Runtime and profiler | MH0 device ms | MH0 launches | MH1 device ms | MH1 launches | MH1/MH0 |
|---|---:|---:|---:|---:|---:|
| PyTorch e3nn, Kineto | 541.696 | 1,064 | 2056.546 | 1,473 | 3.796x |
| Symmetrix hipRTC, rocprofv3 | 24.735 | 40 | 138.098 | 167 | 5.583x |

The synchronized PyTorch wall brackets were `541.990` and `2056.583 ms`.
The profiled Symmetrix brackets were `25.346` and `138.861 ms`. Kernel sums,
not profiler-perturbed wall time, are used below.

## Pathway growth

| Runtime pathway | MH0 ms / launches | MH1 ms / launches | Delta ms | Delta share | Ratio |
|---|---:|---:|---:|---:|---:|
| PyTorch forward | 177.139 / 390 | 1555.280 / 511 | 1378.141 | 91.0% | 8.780x |
| PyTorch autograd reverse | 364.557 / 674 | 501.266 / 962 | 136.709 | 9.0% | 1.375x |
| Symmetrix common geometry/force/ZBL | 2.279 / 8 | 3.371 / 10 | 1.092 | 1.0% | 1.479x |
| Symmetrix model forward | 6.591 / 19 | 35.099 / 62 | 28.507 | 25.1% | 5.325x |
| Symmetrix model reverse | 15.865 / 13 | 99.629 / 95 | 83.764 | 73.9% | 6.280x |

PyTorch's MH1 increase is not a uniform architectural multiplier. Product
modules rise from `50.011` to `1386.899 ms` and account for `88.3%` of the
whole-step delta. Device work attributed to `aten::copy_` inside those modules
rises from `7.000` to `1239.859 ms`, accounting for `81.4%` of the delta.
The two interaction modules rise only from `126.114` to `167.054 ms`.

Symmetrix has already eliminated the upstream product-copy pathology. Its
remaining implementation gap is instead the generated reverse schedule.

## Current Symmetrix MH1 phases

| Phase | Device ms | Share | Launches |
|---|---:|---:|---:|
| M1/node reverse, including reverse readout | 39.174 | 28.4% | 80 |
| R1 reverse: edge phi | 22.621 | 16.4% | 2 |
| M1/node forward | 17.672 | 12.8% | 35 |
| R1 reverse: source | 16.875 | 12.2% | 2 |
| R1 forward | 14.278 | 10.3% | 2 |
| R1 reverse: edge harmonic | 13.443 | 9.7% | 2 |
| R1 edge conditioning reverse | 7.040 | 5.1% | 2 |
| R1 edge conditioning forward | 2.550 | 1.8% | 2 |
| Common geometry, force assembly, ZBL, copies | 4.386 | 3.2% | 36 |
| Opaque E3Linear/Tensile helpers | 0.060 | 0.0% | 4 |

Within node reverse, transpose/adjoint work is `18.945 ms`. Forward replay and
recomputation are another `19.157 ms` across 42 launches; reverse setup and
readout seeding account for the remaining `1.071 ms`. The replay/recompute
portion is avoidable if forward values survive to reverse.

## Next shared-RTC optimization

Implement a neutral selective-retention schedule for the node program:

1. Retain pre-gate values (`9,728` FP32 values per node per layer) and
   interaction outputs (`8,192` per node per layer) directly from forward.
2. Add neutral ABI fields and launch-plan/cache identity for retained-state
   pointers. CUDA Driver and HIP Module adapters only marshal those pointers.
3. Make reverse consume retained values and omit residual/message replay, gate
   replay, linear2 replay, and residual/message recomputation.
4. Keep the current recompute schedule as a memory-budget policy available to
   both backends. Do not introduce CUDA- or HIP-specific numerical kernels.

For two layers at 864 atoms, the retained state is `35,840` floats per atom,
or `118.125 MiB`. The hard Amdahl ceiling is `19.157 ms`; actual savings must
include any changed memory traffic. Accept the change only after an
`old-new-old-new` cached-module pair shows a stable benefit, followed by fresh
process timing. Energy, per-atom energy, forces, and stress must match at 864,
9, and 17 atoms, and the record must report the exact workspace increase.

This optimization alone cannot reach the 2.5x gate. After it, profile the
`52.940 ms` combined interaction-reverse family (source, edge-phi, harmonic)
for cross-kernel state reuse and ownership changes. Residual and product
dual-node tiling is lower priority: those families total only about `10.5 ms`
across forward and reverse before any achievable fractional reduction.

## Reproduction and artifacts

The tracked entry points are:

- `benchmarks/mace_torch_rocm_profile.py`
- `benchmarks/symmetrix_mh_rocm_profile.py`
- `benchmarks/profile_graph_fingerprint.py`
- `benchmarks/analyze_kineto_trace.py`
- `benchmarks/analyze_rocprof_kernels.py`

Raw local outputs are under the ignored directory
`benchmarks/.artifacts/mh1-profile-comparison-20260809/`.

Run the following from the repository root. The checkpoint directory and ROCm
Python path are the ones used for this record. `PROFILE_EXTRACTOR` must be the
`symmetrix_extract_mace` entry point in an environment containing MACE and a
CPU-built Symmetrix extension. Loading the HIP extension and ROCm PyTorch into
one extraction process triggers the duplicate-LLVM failure described below.

```bash
PROFILE_PYTHON=/path/to/pytorch-env/bin/python
PROFILE_EXTRACTOR=/path/to/cpu-symmetrix-env/bin/symmetrix_extract_mace
PROFILE_CHECKPOINT_DIR=/path/to/checkpoints
PROFILE_OUTPUT_ROOT=benchmarks/.artifacts/mh1-profile-comparison-20260809
PROFILE_MODEL_DIR="$PROFILE_OUTPUT_ROOT/models"

mkdir -p "$PROFILE_MODEL_DIR"
"$PROFILE_EXTRACTOR" \
    --model "$PROFILE_CHECKPOINT_DIR/mace-mh-0.model" \
    --head omat_pbe --atomic-numbers 13 7 \
    --output "$PROFILE_MODEL_DIR/mace-mh-0-omat-pbe-Al-N.json"
"$PROFILE_EXTRACTOR" \
    --model "$PROFILE_CHECKPOINT_DIR/mace-mh-1.model" \
    --head omat_pbe --atomic-numbers 13 7 \
    --output "$PROFILE_MODEL_DIR/mace-mh-1-omat-pbe-Al-N.json"
sha256sum "$PROFILE_CHECKPOINT_DIR/mace-mh-0.model" \
    "$PROFILE_CHECKPOINT_DIR/mace-mh-1.model" \
    "$PROFILE_MODEL_DIR/mace-mh-0-omat-pbe-Al-N.json" \
    "$PROFILE_MODEL_DIR/mace-mh-1-omat-pbe-Al-N.json"
```

Build the HIP extension from the recorded milestone. The output filename below
is for the Python 3.14 environment used here.

```bash
PROFILE_BUILD=/tmp/symmetrix-mh-profile-build
cmake -S symmetrix -B "$PROFILE_BUILD" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/usr/bin/gcc \
    -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DSYMMETRIX_DEVICE_BACKEND=HIP \
    -DKokkos_ARCH_AMD_GFX1100=ON \
    -DKokkos_IMPL_AMDGPU_FLAGS=--offload-arch=gfx1151 \
    -DKokkos_IMPL_AMDGPU_LINK=--offload-arch=gfx1151
cmake --build "$PROFILE_BUILD" --target symmetrix_bindings -j 4
PROFILE_EXTENSION="$PROFILE_BUILD/symmetrix.cpython-314-x86_64-linux-gnu.so"
sha256sum "$PROFILE_EXTENSION"
```

Generate and analyze the two PyTorch Kineto traces:

```bash
mkdir -p "$PROFILE_OUTPUT_ROOT/pytorch-mh0" \
    "$PROFILE_OUTPUT_ROOT/pytorch-mh1"
env LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/core-7.14/lib \
    "$PROFILE_PYTHON" benchmarks/mace_torch_rocm_profile.py mh0 \
    --warmups 3 \
    --output "$PROFILE_OUTPUT_ROOT/pytorch-mh0/kineto_report.json" \
    --torch-trace "$PROFILE_OUTPUT_ROOT/pytorch-mh0/kineto_trace.json"
env LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/core-7.14/lib \
    "$PROFILE_PYTHON" benchmarks/mace_torch_rocm_profile.py mh1 \
    --warmups 3 \
    --output "$PROFILE_OUTPUT_ROOT/pytorch-mh1/kineto_report.json" \
    --torch-trace "$PROFILE_OUTPUT_ROOT/pytorch-mh1/kineto_trace.json"
.venv/bin/python benchmarks/analyze_kineto_trace.py \
    "$PROFILE_OUTPUT_ROOT/pytorch-mh0/kineto_trace.json" \
    --json-output "$PROFILE_OUTPUT_ROOT/pytorch-mh0/kineto_summary.json"
.venv/bin/python benchmarks/analyze_kineto_trace.py \
    "$PROFILE_OUTPUT_ROOT/pytorch-mh1/kineto_trace.json" \
    --json-output "$PROFILE_OUTPUT_ROOT/pytorch-mh1/kineto_summary.json"
```

Generate and analyze the selected-region Symmetrix rocprofv3 traces:

```bash
mkdir -p "$PROFILE_OUTPUT_ROOT/symmetrix-mh0" \
    "$PROFILE_OUTPUT_ROOT/symmetrix-mh1"
env LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/core-7.14/lib \
    /opt/rocm/bin/rocprofv3 --kernel-trace --stats --selected-regions \
    -f csv -d "$PROFILE_OUTPUT_ROOT/symmetrix-mh0" \
    -o symmetrix_mh0_steady -- \
    "$PROFILE_PYTHON" benchmarks/symmetrix_mh_rocm_profile.py mh0 \
    "$PROFILE_MODEL_DIR/mace-mh-0-omat-pbe-Al-N.json" --warmups 3 \
    --output "$PROFILE_OUTPUT_ROOT/symmetrix-mh0/harness_report.json" \
    --extension "$PROFILE_EXTENSION" --source-root .
env LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/core-7.14/lib \
    /opt/rocm/bin/rocprofv3 --kernel-trace --stats --selected-regions \
    -f csv -d "$PROFILE_OUTPUT_ROOT/symmetrix-mh1" \
    -o symmetrix_mh1_steady -- \
    "$PROFILE_PYTHON" benchmarks/symmetrix_mh_rocm_profile.py mh1 \
    "$PROFILE_MODEL_DIR/mace-mh-1-omat-pbe-Al-N.json" --warmups 3 \
    --output "$PROFILE_OUTPUT_ROOT/symmetrix-mh1/harness_report.json" \
    --extension "$PROFILE_EXTENSION" --source-root .
.venv/bin/python benchmarks/analyze_rocprof_kernels.py \
    "$PROFILE_OUTPUT_ROOT/symmetrix-mh0/symmetrix_mh0_steady_kernel_trace.csv" \
    --json-output "$PROFILE_OUTPUT_ROOT/symmetrix-mh0/kernel_summary.json"
.venv/bin/python benchmarks/analyze_rocprof_kernels.py \
    "$PROFILE_OUTPUT_ROOT/symmetrix-mh1/symmetrix_mh1_steady_kernel_trace.csv" \
    --json-output "$PROFILE_OUTPUT_ROOT/symmetrix-mh1/kernel_summary.json"
```

Each harness report contains the canonical and order-sensitive graph hashes.
The four values can be checked together with:

```bash
jq -r \
    '[.workload.generation // .workload.role, \
      .graph.canonical_fingerprint.sha256, \
      .graph.canonical_fingerprint.input_order_edge_records_sha256] | @tsv' \
    "$PROFILE_OUTPUT_ROOT"/{pytorch,symmetrix}-mh{0,1}/*report.json
```

The attempted direct rocprofv3/PyTorch run aborted during startup because two
loaded LLVM components registered `spirv-expand-step`. PyTorch device evidence
therefore comes from Kineto's ROCm kernel activities. Symmetrix rocprofv3 is
unaffected. The Chrome traces retain exact kernel count, duration, external
operator ID, and stream correlation.

The Symmetrix harness pauses profiling before setup and warmups, then brackets
the measured call with `roctxProfilerResume(0)` and `roctxProfilerPause(0)`.
Use rocprofv3 `--selected-regions`; the harness also rejects the wrong model
hash, non-hipRTC compilation, fallback execution, a graph fingerprint mismatch,
or diagnostic numerical scalar checks outside their recorded tolerances.
