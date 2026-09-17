# MACE Kokkos source layout

The MACE Kokkos evaluator is split into semantic translation units. Each
translation unit owns a disjoint family of `MACEKokkos` method definitions and
explicitly instantiates both supported precisions. This keeps implementation
edits local and lets expensive CPU and accelerator template work compile in
parallel.

## Owners

| Source | Responsibility |
|---|---|
| `mace_kokkos_runtime.cpp` | Construction, destruction, backend and plugin policy, workspace admission, diagnostics, observers, and parameter-gradient setup |
| `mace_kokkos_factorized_lifecycle.cpp` | Streamed modes and schedules, prepared graph state, Cartesian/fractional geometry lifecycle and device mapping, active types, and factorized model preparation |
| `mace_kokkos_factorized_execution.cpp` | Factorized R1 forward execution and direct and generic reverse execution |
| `mace_kokkos_factorized_analysis.cpp` | Factorized operator benchmarking and parameter-gradient kernels |
| `mace_kokkos_evaluate.cpp` | Force reductions and ordinary and field evaluator entry points |
| `mace_kokkos_response.cpp` | Analytic field response; includes `mace_kokkos_response.tpp` as its private implementation body |
| `mace_kokkos_first_interaction.cpp` | R0, R1, spherical harmonics, A0 variants, and M0 |
| `mace_kokkos_h1_phi1.cpp` | H1, field H1, and Phi1 forward and reverse paths |
| `mace_kokkos_second_interaction.cpp` | A1 variants, M1, H2, and readouts |
| `mace_kokkos_model.cpp` | JSON model loading and model-contract validation |

The authoritative build list is the `SYMMETRIX_KOKKOS` source block in
`libsymmetrix/CMakeLists.txt`. The layout contract is enforced by
`symmetrix/test/test_mace_kokkos_source_partitioning.py`. Its order is a build
scheduling contract: with the current equal-priority Ninja graph, longer CUDA
owners are listed first so they enter the ready queue before shorter owners.

## Private shared contracts

Four headers under `libsymmetrix/source/` carry implementation details that
must be complete in more than one owner:

| Header | Consumers | Contract |
|---|---|---|
| `mace_kokkos_factorized_blas_detail.hpp` | runtime, factorized execution | BLAS context lifetime, stream binding, and CUDA/HIP batched GEMM launch helpers |
| `mace_kokkos_kernel_launch_detail.hpp` | runtime, factorized lifecycle, factorized execution, model | Backend capability probing and standard-module launch-profile selection |
| `mace_kokkos_jit_plugin_detail.hpp` | factorized execution, first interaction, H1/Phi1 | Validated host and device JIT-plugin launch-packet construction |
| `mace_kokkos_spherical_harmonics_detail.hpp` | runtime, first interaction | Complete nested spherical-harmonics state and backend-specific teardown |

Owner-local helpers stay in the unnamed namespace of their source file. Do not
add a common implementation prelude: it recreates broad invalidation and makes
unrelated owners parse the same implementation.

## Placement rules

- Declare public and cross-owner members in `mace_kokkos.hpp`; define each
  member in exactly one semantic owner.
- Place a new method with the state or kernel family it changes. Calls between
  owners use class declarations rather than source inclusion.
- End every owner with exactly one `template class MACEKokkos<float>;` and one
  `template class MACEKokkos<double>;`. Do not add per-method or `extern`
  instantiation as part of an unrelated change.
- Keep `mace_kokkos_response.tpp` included only by
  `mace_kokkos_response.cpp`. It is an implementation body, not a shared
  header.
- Add a private detail header only when multiple owners require the same
  complete type or inline helper. Keep its consumer set explicit and acyclic.
- Keep the standard M0/R0 module bodies and launch profiles under
  `libsymmetrix/source/`. Runtime-generated R1 code belongs to the JIT compiler
  and cache rather than a checked-in generated-source directory.

## Rebuild expectations

Editing one `.cpp` owner should recompile that owner and relink its targets, but
must not recompile the other nine owners. Editing a private detail header should
recompile only the consumers listed above. Changes to `mace_kokkos.hpp` remain
class-wide and are expected to rebuild every owner.

After changing ownership or shared dependencies, run the source-partitioning
test, build at least one supported backend, and compare the normalized
`MACEKokkos<float>` and `MACEKokkos<double>` symbol inventory with the previous
layout. Backend-specific build and runtime coverage must be reported
separately.
