# MACE-Torch Comparison Protocol

This document defines the repeatable MACE-Torch/cuEquivariance versus
Symmetrix-XL benchmark used for README performance claims. It reproduces the
complete ASE energy, forces, and stress call protocol used for the paper's
single-GPU MACE comparison. It does not define a separate molecular-dynamics
protocol.

## Scope

The comparison uses:

- the original MACE checkpoint as the common source of model parameters;
- MACE-Torch with cuEquivariance as the reference implementation;
- Symmetrix-XL direct/streamed execution as the candidate implementation;
- one clean Python 3.12 virtual environment created with `uv` and containing
  both implementations;
- one fresh process per implementation and workload size; and
- identical coordinates, requested properties, precision, hardware, and timing
  boundary for both implementations.

Separate virtual environments are allowed only when incompatible binary or CUDA
dependencies make a shared environment impossible. Such a result must be
labelled as cross-environment and must not replace the matched README result.

## Environment

Create a new environment for every benchmark campaign. Do not reuse the
repository development environment or a previous qualification environment.

```bash
run_id=<date>-<symmetrix-version>-<system>
env_root="$HOME/tmp/symmetrix-benchmark/$run_id"
artifact_root="$PWD/benchmarks/.artifacts/$run_id"

uv venv --python 3.12 "$env_root/venv"
mkdir -p "$artifact_root"
python="$env_root/venv/bin/python"
```

Install pinned, mutually compatible builds of Torch, MACE-Torch,
cuEquivariance, ASE, NumPy, and Symmetrix-XL into this environment. Install the
exact wheel or source revision under qualification rather than an unidentified
editable checkout.

```bash
uv pip install --python "$python" \
  "torch==<version>" \
  "mace-torch==<version>" \
  "cuequivariance==<version>" \
  "cuequivariance-torch==<version>" \
  "ase==<version>" \
  "numpy==<version>"

uv pip install --python "$python" /path/to/symmetrix_xl-<version>-<tag>.whl
uv pip freeze --python "$python" > "$artifact_root/packages.txt"
```

Record the exact installation commands, wheel hashes, and package versions. A
campaign is invalid if either implementation loads a package or native
extension outside the campaign environment.

## Paper reference case

The README reference workload is the paper's 5,000-atom case:

| Setting | Value |
|---|---|
| Model | MACE-OMAT-0 medium |
| MACE head | `default` |
| Structure | Cubic Pm-3m SrTiO3, `a = 3.905 A` |
| Supercell | `10 x 10 x 10`, 5,000 atoms |
| Position perturbation | Gaussian, `0.01 A` standard deviation |
| Position seed | `17` |
| Precision | FP32 |
| Requested properties | Energy, forces, and stress |
| Model cutoff | `6.0 A` |
| Warm-up calls | 3 |
| Measured calls | 10 |
| Primary statistic | Median complete-call time in `us/atom` |

The same protocol may be applied to another system or size, but the structure,
atom count, composition, cell, perturbation, seed, and edge counts must be
reported. Results from different systems or sizes are separate qualification
points and are not pooled.

MACE-Torch builds the exact `6.0 A` model graph. Symmetrix-XL uses a retained
neighbor graph with a `0.5 A` skin and therefore a `6.5 A` effective cutoff.
Record the directed edge count for both cutoffs and disclose this graph-policy
difference with every published result.

## Model and input identity

Record:

- checkpoint path and SHA-256;
- selected and available MACE heads;
- compact-model path and SHA-256;
- export command and exporter version or source revision;
- model cutoff, precision, and Symmetrix execution-policy options; and
- atomic numbers, positions, cell, periodic flags, and a digest of the generated
  input state.

Generate the perturbed structure independently in every fresh process using the
same driver and seed. Before accepting results, verify that the generated state
digest, checkpoint digest, selected head, precision, atom count, species, and
cell match between implementations.

## System preflight

Before each campaign:

1. Record the operating system, CPU, GPU model, compute capability, driver,
   visible GPU count, GPU memory, and CUDA toolkit and runtime versions.
2. Verify that the selected GPU is idle and has no competing compute process.
3. Verify `torch.cuda.is_available()` and execute a small CUDA operation with
   the campaign interpreter.
4. Construct MACE-Torch with cuEquivariance enabled and verify the selected head
   and accelerated module configuration.
5. Run `symmetrix version --probe --json` and `symmetrix doctor --json` with the
   same interpreter. Require the intended backend, execution space,
   architecture, source hash, and native-content hash.
6. Set host thread pools to one and record the effective values.

Apply CPU affinity, GPU clock, or power-limit policies before either process
starts, and retain the same policy for both implementations.

## Timing boundary

A complete timed call invokes the ASE calculator for energy, forces, and stress
with all system changes enabled. The boundary includes:

- Python calculator dispatch;
- graph construction for MACE-Torch or retained-neighbor input update for
  Symmetrix-XL;
- framework or native model evaluation; and
- assembly of the returned energy, forces, and stress.

Calculator construction, model loading, package import, backend initialization,
JIT compilation, and the three warm-up calls are outside the timing boundary.
Do not substitute an isolated model forward pass, kernel timing, profiler replay,
or molecular-dynamics step timing.

## Timing procedure

Run each implementation in a fresh process using the same interpreter:

1. Generate and validate the deterministic input structure.
2. Construct the calculator and verify its model, head, backend, and execution
   policy.
3. Perform three untimed complete energy/forces/stress calls.
4. For each of ten measured calls, synchronize the GPU, start the host timer,
   perform one complete call, synchronize the GPU again, and stop the timer.
5. Record every call time, median, mean, minimum, maximum, setup time, sampled
   CUDA process memory, and final energy, forces, and stress.

The headline latency is the median of the ten measured complete-call samples,
converted to `us/atom`. The matched speedup is:

```text
MACE-Torch + cuEquivariance median us/atom
------------------------------------------------
Symmetrix-XL direct/streamed median us/atom
```

Retain valid slow samples. Repeat the entire campaign only when a run is
invalidated or when an additional independent qualification point is desired;
do not replace the ten-sample median with a median-of-trial-medians statistic.

## Numerical comparison

Compare the final energy, force components, and stress components from the two
implementations on the identical input state. Report:

- absolute energy difference in `eV/atom`;
- maximum absolute force-component difference in `eV/A`; and
- maximum absolute stress-component difference in `eV/A^3`.

The paper reported these quantities rather than defining a new MD drift gate.
Any release acceptance limits must be declared before running the campaign and
published alongside the observed errors.

## Required provenance

Every raw result must include:

- benchmark driver path and SHA-256;
- exact command and relevant environment variables;
- Python executable and version;
- installed package versions and wheel or source identities;
- Symmetrix source commit and dirty state, native extension path and SHA-256,
  backend descriptor, and JIT artifact identity;
- checkpoint, compact-model, and input-state SHA-256 values;
- model head, precision, properties, cutoff, skin, effective cutoff, directed
  edge counts, atom count, composition, cell, and seed;
- GPU, driver, CUDA runtime, CPU affinity, and thread counts;
- all ten timing samples and sampled process memory; and
- energy, force, and stress comparison results.

Parse every record after the run and verify these fields against the requested
case. Filenames and commands alone are not evidence that the requested case ran.

## Invalid runs

Discard and repeat a run when:

- another process uses the benchmark GPU during measurement;
- the wrong device, checkpoint, head, precision, backend, execution mode, or JIT
  artifact is selected;
- Symmetrix falls back when direct execution is required;
- input or model identities differ between implementations;
- a timing sample is missing, non-finite, or contains setup or compilation;
- requested properties or graph settings differ from the protocol; or
- package, driver, hardware, or numerical provenance is incomplete.

Do not remove an otherwise valid slow sample merely because it is an outlier.

## README publication

Publish the matched ASE comparison separately from LAMMPS results:

| Implementation | Atoms | Median time (us/atom) | Sampled VRAM (MiB) | Matched speedup |
|---|---:|---:|---:|---:|
| MACE-Torch + cuEquivariance | 5,000 | `<value>` | `<value>` | 1.00x |
| Symmetrix-XL direct/streamed | 5,000 | `<value>` | `<value>` | `<value>x` |

The surrounding text must state the Symmetrix version, qualification date,
hardware, model and checkpoint hash, precision, atom count, properties,
cutoff/skin policies, warm-up and sample counts, aggregation method, and
numerical differences. Do not compute a speedup across different hardware,
models, structures, properties, precision, or timing boundaries.

Store raw JSON, logs, environment snapshots, and the aggregate report under
`benchmarks/.artifacts/<run-id>/`. These machine-specific artifacts are not
committed. Attach long-term evidence to the corresponding release or archive it
externally.

## Repeating the campaign

For a new Symmetrix release or system:

1. Create a new run ID and clean shared `uv` environment.
2. Install and freeze the complete dependency set.
3. Export or validate the compact model from the original checkpoint.
4. Run preflight and verify deterministic input generation.
5. Run MACE-Torch + cuEquivariance in a fresh process.
6. Run Symmetrix-XL in a fresh process.
7. Validate the records, numerical differences, and matched timing boundary.
8. Update the README only from the validated aggregate report.

Never combine samples across GPU models, CUDA stacks, package sets, Symmetrix
versions, or workload definitions into one headline number.
