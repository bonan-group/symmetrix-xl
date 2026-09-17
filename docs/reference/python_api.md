# Python API

This reference is generated from the public source contract and is deliberately
kept independent of a native extension, so documentation can render without a
particular CPU/GPU backend installed.

## Frontend and Backend Selection

`symmetrix.available_backends()` returns installed backend descriptors without
loading their native extensions. `symmetrix.load_backend(request=None)` loads
the automatic or named backend and permanently selects it for the process.
`symmetrix.selected_backend()` exposes the resolved descriptor after selection.

Select a backend before loading a calculator or native symbol.

## Symmetrix

```python
Symmetrix(
    model_file, dtype="float32", use_kokkos=True,
    streamed_edges="direct", execution_profile="capacity",
    allow_fixed_workspace=False,
    low_memory=<compatibility switch>, neighbor_skin=0.5, head=None,
    **ase_kwargs,
)
```

`Symmetrix` is an ASE calculator. `dtype` defaults to `"float32"` and also
accepts `"float64"`. Select float64 explicitly for high-precision calculations;
the non-Kokkos serial `MACE_Nonlinear` evaluator currently requires it.
`streamed_edges` accepts `direct` (default), `non-compiled`, `materialized`,
and temporary compatibility request `auto`. `generic` and `all_interactions`
are deprecated aliases for `non-compiled`. `factorized` and `direct_streamed`
are compatibility aliases for `direct`; in the Python frontend they preserve
the former throughput behavior by changing an otherwise-`capacity` request to
the `speed` profile. Use literal `direct` for capacity behavior. The removed
`receiver_factorized` selector is rejected.
`execution_profile` is `capacity` (default) or `speed`;
the compatibility `low_memory=True` and `low_memory=False` spellings map to
those profiles.
`allow_fixed_workspace=True` separately permits bounded tiled workspace plans
for qualified CUDA FP32 direct capacity execution. It defaults to `False` and
does not force the planner to select such a plan.

`direct` is the first-class performance path. Two-interaction models require a
matching admitted R1 artifact; admitted single-layer models use built-in R1
execution, although capacity planning may still compile or load M0/R0 operator
modules. `non-compiled` is an explicit compiler-free fallback/diagnostic mode
with no performance guarantee.

Set `dispersion=True` to add the D3 correction from the optional `torch-dftd`
package. Symmetrix-XL evaluates the neural model and D3 calculator together and
adds their energy, forces, and stress. Install it with
`uv pip install 'symmetrix-xl[dispersion]'`.

Ordinary models implement `energy`, `free_energy`, `energies`, `forces`, and
`stress`. MACEField models additionally expose `polarization`, `becs`, and
`polarizability` when supported by the model and requested calculation.

With a positive `neighbor_skin`, fully periodic `direct` Kokkos calculations
construct and retain the candidate neighbor graph in native Kokkos memory.
The ASE layer keeps only node-sized state for Verlet validity checks. Energy,
forces, stress, polarization, and polarizability use this path; MACEField BECs
currently retain host edge arrays for their response reduction. The diagnostic
environment variable `SYMMETRIX_NEIGHBOR_BACKEND=host|kokkos` pins either
implementation. On CUDA and HIP, the default `automatic` selects the host
builder below 512 atoms and Kokkos at 512 atoms or above when native
construction is eligible. Serial and OpenMP use the host builder by default.
The calculator attribute `neighbor_graph_backend` reports the implementation
that built the current cached graph.

Useful run-time attributes include `execution_plan`, `jit_status`,
`jit_reason`, `jit_artifact_id`, `jit_variant_id`, `low_memory_policy`, `head`,
and `available_heads`.

## Field Calculators

```python
FieldContributionCalculator(field_calculator, **ase_kwargs)
FieldAwareCalculator(base_calculator, field_calculator, **ase_kwargs)
```

`FieldContributionCalculator` returns the MACEField contribution relative to
zero field. `FieldAwareCalculator` combines that contribution with an arbitrary
ASE base calculator.

## Ensembles

```python
SymmetrixEnsemble(model_files=None, *, calculators=None, **symmetrix_kwargs)
```

Provide exactly one of `model_files` or existing `calculators`. For each
supported base property `x`, an ensemble exposes mean `x`, stacked member
values `x_comm`, and population variance `x_var`.

## Direct Artifacts

`prepare_jit_host_artifact(...)` and `prepare_jit_device_artifact(...)` are
the Python APIs behind the corresponding command-line tools. Their artifacts
are specific to the model contract, precision, ABI, implementation generation,
and backend target. Use the command-line wrappers for repeatable deployments.
