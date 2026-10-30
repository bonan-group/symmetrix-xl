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
| `mace_kokkos_factorized_execution.cpp` | Factorized R1 forward execution and direct and non-compiled reverse execution |
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

## The MACE_Nonlinear (MH-1) partition

The MH-1 evaluator is partitioned the same way, and the same test covers it.
Three translation units instantiate `MaceNonlinearKokkosT` from one
implementation body:

| Source | Responsibility |
|---|---|
| `mace_nonlinear_kokkos_components.cpp` | `SYMMETRIX_MACE_NONLINEAR_KOKKOS_PART 1`; includes the body |
| `mace_nonlinear_kokkos_runtime.cpp` | `SYMMETRIX_MACE_NONLINEAR_KOKKOS_PART 2`; includes the body and asserts the family-adapter interface |
| `mace_nonlinear_kokkos_evaluate.cpp` | `SYMMETRIX_MACE_NONLINEAR_KOKKOS_PART 3`; includes the body |

`mace_nonlinear_kokkos_impl.tpp` is an implementation body, not a shared
header: it is included only by those three owners, so it is listed in no build
file and carried by no shared layer.

`mace_nonlinear_stage.hpp` is the family's stage description. The interaction
layer count, every layer's R and M forward and reverse profiling region name,
and the ordered forward and reverse execution schedules are its **data**, built
from a layer count, so an evaluator that admits a different interaction count
profiles and dispatches it through the same value instead of a compiled-in
two-layer table. The region spelling it emits is the one
`benchmarks/analyze_rocprof_kernels.py` decodes, so a description built for any
count keeps the profile contract readable. The evaluator holds one description
built from the validated model's layer count and reads its regions and phase
ordinals from it.

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
- A family-owned extension point on a shared class stays behind an opaque
  handle: the shared class declares the handle and the entry points, the family
  layer defines them and owns the state. `E3TensorProductKokkosT`
  (`e3nn_kokkos.hpp`) holds `direct_execution_plan_` this way, with the plan
  layout and every kernel that reads it in
  `mace_nonlinear_mh1_fast_path.{hpp,cpp}`, and `E3ProductBasisKokkosT`
  (`e3nn_product_kokkos.hpp`) holds `compiled_host_plan_` this way, with the
  frozen term plan and its accumulation kernels in
  `mace_nonlinear_product_host_plan.cpp`. Shared translation units and shared
  headers must not include family headers or carry family state, so a consumer
  that never installs a plan keeps the generic schedule.
- A family-owned specialization of a shared template lives with the family that
  owns the types it names: `device_backend.hpp` keeps the host-neutral
  environment probes and the primary `DeviceBackendTraits` template, while the
  CUDA/HIP specializations and device-scope guards that alias the MACE JIT ABI
  packet structs and plugin classes live in `mace_device_backend.{hpp,cpp}`.

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
