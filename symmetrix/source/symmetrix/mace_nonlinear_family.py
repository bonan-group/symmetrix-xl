"""Family declaration for the nonlinear compact models (MH-1).

The entry is discovered by the family registry from this module's ``_family``
suffix.  The family is detected by its own model type, admits only the
generated node-program contract, refuses the serial float32 form its serial
evaluator cannot serve, and shares the compact frontend's execution-mode
vocabulary, which the family registry declares beside its data model.
"""

from __future__ import annotations

from .family_registry import (
    COMPACT_MODE_VOCABULARY,
    ArtifactAdapter,
    CapabilityDeclaration,
    ContractDeclaration,
    Detection,
    EvaluatorSurface,
    EvaluatorVariant,
    FamilyEntry,
    Routing,
    Symbol,
)

# Both compact families share one frontend and therefore one vocabulary; the
# entry still declares it, so a consumer reads the accepted spellings from the
# family it detected instead of from the frontend it happens to run inside.

FAMILY_ENTRY = FamilyEntry(
    key="MACE_Nonlinear",
    # The compact frontend resolved this family before the compact model family.
    precedence=0,
    detection=Detection(model_types=("MACE_Nonlinear",)),
    evaluators=(
        EvaluatorVariant(
            kokkos=True,
            precision="float64",
            class_name="MACENonlinearKokkos",
            declared_error=(
                "This Symmetrix build does not provide the requested "
                "native MACE_Nonlinear Kokkos evaluator."
            ),
        ),
        EvaluatorVariant(
            kokkos=True,
            precision="float32",
            class_name="MACENonlinearKokkosFloat",
            declared_error=(
                "This Symmetrix build does not provide the requested "
                "native MACE_Nonlinear Kokkos evaluator."
            ),
        ),
    ),
    # The serial MH-1 evaluator required float64, and its removal retires that
    # restriction with it: both Kokkos precisions serve the family, so no
    # precision and execution space is refused by contract.
    modes=COMPACT_MODE_VOCABULARY,
    routings=(
        Routing(key="direct", native_mode="direct", surface="prepared"),
        Routing(key="reference", native_mode="generic", surface="reduced"),
    ),
    schema=Symbol("mace_nonlinear_execution_contract"),
    extractor=Symbol("extract_mace_nonlinear", "extract_mace_nonlinear_data"),
    contract=ContractDeclaration(
        builder=Symbol(
            "mace_nonlinear_execution_contract", "make_execution_mh1_contract"
        ),
        validator=Symbol(
            "mace_nonlinear_execution_contract",
            "validate_execution_mh1_v4_contract_for_model",
        ),
        json_key="MH1_UVU",
        specialization="mh1",
    ),
    generated_contract=ContractDeclaration(
        builder=Symbol(
            "mace_nonlinear_execution_contract", "make_execution_mh1_contract"
        ),
        validator=Symbol(
            "mace_nonlinear_execution_contract",
            "validate_execution_mh1_v4_contract_for_model",
        ),
        json_key="MH1_UVU",
        specialization="mh1",
    ),
    # The nonlinear generator owns interaction, conditioning, gates,
    # product-linear/skip/readout, and their reverse phases. Every product
    # contraction crosses the canonical, type-agnostic M0/M1 boundary and is
    # emitted only by the ordinary generated-operator codegen.
    codegen=(
        Symbol("mace_nonlinear_jit_codegen"),
        Symbol("generated_operator_codegen"),
    ),
    # The full staged plugin and each canonical product module resolve
    # independently. A prebuilt M0/M1 module does not make the model-specific
    # MH-1 plugin prebuilt; diagnostics must preserve both origins.
    adapters=(
        Symbol("jit", "prepare_jit_artifact"),
        Symbol("mace_jit_operator_artifact"),
    ),
    # The direct routing runs a prepared artifact.  The shared JIT entry point
    # takes rendered source, and the MH-1 host plugin's source is rendered from
    # the model's staged node contract by this family's own code generator, so
    # the family declares its own host entry point instead.  The frontend hands
    # the evaluator the prepared library positionally through
    # ``_load_jit_mh1_host_plugin_v7(path)`` -- the binding declares no keyword
    # for it -- and reads the published identity back off the evaluator
    # (``jit_mh1_host_plugin_artifact_id``), so neither an evaluator keyword nor
    # an identity keyword is declared.  MH-1 declares no device adapter: its
    # generated CUDA and HIP modules are rendered against the device plan during
    # calculator construction, not prepared from a model file. The common
    # M0/M1 modules still use the operator adapter declared above.
    artifacts=(
        ArtifactAdapter(
            routing="direct",
            target="host",
            prepare=Symbol("mace_nonlinear_jit_artifact", "prepare_mh1_host_artifact"),
            manifest_path="manifest_path",
        ),
    ),
    capabilities=CapabilityDeclaration(family="mace-nonlinear"),
    # MH-1 shares the compact frontend, so it declares the same surface keys
    # with the methods its own evaluator binds: it has no field coupling, no
    # fractional geometry, and no all-interactions block.  The mode handshake
    # and the prepared surfaces are the evaluator's work; the reduced surface is
    # the whole-evaluator entry point every routing can reach.
    surfaces=(
        EvaluatorSurface(
            key="mode_handshake",
            methods=("set_streamed_edges",),
            kokkos=True,
        ),
        EvaluatorSurface(
            key="prepared",
            methods=("_prepare_factorized_graph", "_compute_prepared_factorized"),
            readiness=("portable", "generated"),
            kokkos=True,
        ),
        EvaluatorSurface(
            key="prepared_graph",
            methods=("_prepare_factorized_graph",),
            kokkos=True,
        ),
        EvaluatorSurface(
            key="prepared_positions",
            methods=("_compute_prepared_factorized_positions",),
            kokkos=True,
        ),
        EvaluatorSurface(
            key="prepared_geometry",
            methods=("_prepare_factorized_geometry",),
            kokkos=True,
        ),
        EvaluatorSurface(
            key="reduced",
            methods=("compute_node_energies_forces",),
        ),
        EvaluatorSurface(
            key="reduced_stress",
            methods=("_reduce_stress",),
            kokkos=True,
        ),
    ),
)
