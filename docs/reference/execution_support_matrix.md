# Execution Support Matrix

This matrix describes the public execution algorithms. Internal kernels,
storage policies, and historical selector names are not separate public modes.

| Public request | Model boundary | Kokkos backends | Runtime artifact | LAMMPS |
|---|---|---|---|---|
| `direct` | Compact standard MACE and MACEField with admitted contracts; compatible compact MACE-MH-1 | Serial, OpenMP, CUDA, HIP | Standard two-interaction MACE/MACEField requires an R1 artifact; single-layer standard MACE runs without R1 specialization, but capacity may specialize M0/R0; MACE-MH-1 requires its generated program | Standard MACE and MACEField; R1 artifact required for two-interaction models, but not single-layer models; MACE-MH-1 is not supported |
| `non-compiled` | Compact standard MACE and MACEField; compatible compact MACE-MH-1 | Serial, OpenMP, CUDA, HIP | None | Standard MACE and MACEField |
| `materialized` | Legacy format-v1 models and retained numerical controls | Serial, OpenMP, CUDA, HIP | None | Legacy and supported standard MACE/MACEField controls |
| `auto` | Compatibility request resolved from model format and evaluator capability | As resolved above | As resolved above | Format-aware compatibility default |

The non-Kokkos evaluator selected by `use_kokkos=False` does not provide
prepared direct execution: a `direct` request resolves to serial generic
execution with a warning. An explicit `materialized` request remains
materialized.

`generic` and `all_interactions` are deprecated aliases for `non-compiled`.
`factorized` and `direct_streamed` are compatibility aliases for `direct` and,
in the Python frontend, preserve the former throughput-oriented behavior by
changing an otherwise-`capacity` request to the `speed` profile. Use the
literal `direct` request for capacity behavior. These aliases do not emit a
deprecation warning. The removed `receiver_factorized` selector is rejected.

Ordinary MACE exposes energy, free energy, per-atom energies, forces, and
stress. MACEField additionally exposes polarization and admitted analytical
Born effective charge and polarizability calculations. Property, precision,
and MPI qualification can be narrower than implementation availability; dated
benchmark reports provide the evidence for a specific backend and workload.

`execution_profile="capacity"` and `"speed"` select resource objectives inside
the direct algorithm. `allow_fixed_workspace=True` separately permits bounded
tiled workspace for qualified CUDA FP32 models. These controls do not create
additional execution modes.

See {doc}`/streamed_edge_execution` for the architecture, {doc}`/user/execution`
for user controls, and {doc}`/user/lammps` for the narrower LAMMPS boundary.
