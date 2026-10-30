"""Family declaration for compact MACE, MACEField, and MACELES models.

The entry is discovered by the family registry from this module's
``_family`` suffix, so the basename and the exported constant are the whole
declaration surface.  Detection, evaluators, the model variant that routes
without an R1 specialization, the R1 contract, and the lazy symbols for the
family's schema, extractor, generators, artifact adapters, and native
capability query all live here and nowhere else; the execution-mode vocabulary
the compact frontend resolves is shared by every family it serves and is
declared beside the registry's data model instead.
"""

from __future__ import annotations

from typing import Any, Mapping

from .family_registry import (
    ArtifactAdapter,
    CapabilityDeclaration,
    COMPACT_MODE_VOCABULARY,
    ContractDeclaration,
    Detection,
    EvaluatorSurface,
    EvaluatorVariant,
    FamilyEntry,
    Routing,
    Symbol,
    Variant,
)


def _is_single_layer_compact_model(model_data: Mapping[str, Any]) -> bool:
    """Return whether an MH-0 model carries the single-layer invariant readout.

    The readout variant is declared for the compact model type itself: a field
    variant that carries the flag keeps its R1 specialization, which is the
    routing the frontend applied before this declaration existed.
    """

    return model_data.get("model_type", "MACE") == "MACE" and bool(
        model_data.get("single_layer_readout")
    )


FAMILY_ENTRY = FamilyEntry(
    key="MACE",
    # The compact frontend resolved the nonlinear family first, then this one.
    precedence=1,
    detection=Detection(model_types=("MACE", "MACELES", "MACEField")),
    evaluators=(
        EvaluatorVariant(
            kokkos=True,
            precision="float64",
            class_name="MACEKokkos",
            model_types=("MACE", "MACEField"),
        ),
        EvaluatorVariant(
            kokkos=True,
            precision="float32",
            class_name="MACEKokkosFloat",
            model_types=("MACE", "MACEField"),
        ),
        EvaluatorVariant(
            kokkos=True,
            precision="float64",
            class_name="MACELESKokkos",
            model_types=("MACELES",),
        ),
        EvaluatorVariant(
            kokkos=True,
            precision="float32",
            class_name="MACELESKokkosFloat",
            model_types=("MACELES",),
        ),
    ),
    modes=COMPACT_MODE_VOCABULARY,
    # The two routings: ``direct`` runs the prepared factorized plan and needs a
    # qualified generated artifact; ``reference`` runs the generic streamed
    # implementation through the whole-evaluator entry point on the default host
    # execution space with no artifact at all.  Both reduce stress.
    routings=(
        Routing(key="direct", native_mode="direct", surface="prepared"),
        Routing(key="reference", native_mode="generic", surface="reduced"),
    ),
    schema=Symbol("mace_execution_contract"),
    extractor=Symbol("extract_mace_data", "extract_mace_data"),
    contract=ContractDeclaration(
        builder=Symbol("mace_execution_contract", "normalize_jit_r1_contract"),
        json_key="R1",
        specialization="r1",
    ),
    generated_contract=ContractDeclaration(
        builder=Symbol("mace_execution_contract", "normalize_jit_r1_contract"),
        json_key="R1",
        specialization="r1",
    ),
    codegen=(
        Symbol("mace_jit_codegen"),
        Symbol("mace_jit_operator_codegen"),
    ),
    adapters=(
        Symbol("mace_jit_host_artifact"),
        Symbol("mace_jit_device_artifact"),
        Symbol("mace_jit_operator_artifact"),
    ),
    # The direct routing runs a prepared artifact, so it declares the entry
    # point that prepares one per execution target: the factorized host plugin
    # for the host target, and the same device entry point for the CUDA and HIP
    # targets, which derives the active backend from the evaluator's own device
    # execution environment.  The frontend hands the evaluator the prepared
    # library positionally -- ``_load_jit_host_plugin(path)`` and
    # ``_load_jit_cuda_plugin(path, ...)`` -- and reads the published identity
    # back off the evaluator (``jit_host_plugin_artifact_id`` and
    # ``jit_cuda_plugin_artifact_id``), so neither adapter declares an evaluator
    # or identity keyword.  The reference routing runs the whole-evaluator entry
    # point and declares no artifact at all.
    artifacts=(
        ArtifactAdapter(
            routing="direct",
            target="host",
            prepare=Symbol("mace_jit_host_artifact", "prepare_jit_host_artifact"),
        ),
        ArtifactAdapter(
            routing="direct",
            target="cuda",
            prepare=Symbol("mace_jit_device_artifact", "prepare_jit_device_artifact"),
            manifest_path="manifest_path",
        ),
        ArtifactAdapter(
            routing="direct",
            target="hip",
            prepare=Symbol("mace_jit_device_artifact", "prepare_jit_device_artifact"),
            manifest_path="manifest_path",
        ),
    ),
    capabilities=CapabilityDeclaration(family="MH-0 compact model"),
    variants=(Variant(key="single_layer", matches=_is_single_layer_compact_model),),
    # The evaluator surface the compact frontend reads.  Every method is named
    # once here, so a frontend that has committed to a mode fails with the
    # missing method instead of silently evaluating through a slower path, and
    # the sites that choose between two admitted policies ask the same
    # declaration instead of probing the evaluator.  The mode handshake and the
    # prepared and field-coupled surfaces are the evaluator's work; the reduced
    # surfaces are the whole-evaluator entry points every routing can reach.
    surfaces=(
        EvaluatorSurface(
            key="mode_handshake",
            methods=("set_streamed_edges",),
            kokkos=True,
        ),
        EvaluatorSurface(
            key="mace_les_geometry",
            methods=("_set_les_geometry",),
            kokkos=True,
            model_types=("MACELES",),
        ),
        EvaluatorSurface(
            key="prepared",
            methods=("_prepare_factorized_graph", "_compute_prepared_factorized"),
            readiness=("portable", "generated"),
            kokkos=True,
        ),
        # The weight-aware packet surface the alchemical execution of this
        # family's ``direct`` routing requires.  The frontend requires it by
        # name when a caller asks for alchemical mixing, so a backend without it
        # fails naming the missing method instead of evaluating through an
        # unweighted path.  Only the publishing entry point is bound: the
        # ``alchemical_weights_active`` query is a state property the suites read
        # for diagnostics, not a method the frontend calls.
        EvaluatorSurface(
            key="alchemical",
            methods=("_set_alchemical_weights",),
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
            key="prepared_shift_geometry",
            methods=("_prepare_factorized_shift_geometry",),
            kokkos=True,
        ),
        EvaluatorSurface(
            key="prepared_fractional_geometry",
            methods=("_prepare_factorized_fractional_geometry",),
            kokkos=True,
        ),
        EvaluatorSurface(
            key="prepared_cell_update",
            methods=("_update_factorized_cell",),
            kokkos=True,
        ),
        EvaluatorSurface(
            key="prepared_field",
            methods=("_prepare_factorized_graph", "_compute_prepared_factorized_field"),
            readiness=("portable", "generated"),
            kokkos=True,
        ),
        EvaluatorSurface(
            key="prepared_field_positions",
            methods=("_compute_prepared_factorized_positions_field",),
            kokkos=True,
        ),
        EvaluatorSurface(
            key="prepared_field_response",
            methods=("_compute_prepared_factorized_field_response",),
            kokkos=True,
        ),
        EvaluatorSurface(
            key="all_interactions",
            methods=(
                "_prepare_all_interactions_graph",
                "_prepare_all_interactions_geometry",
                "_compute_prepared_all_interactions_positions",
            ),
            kokkos=True,
        ),
        EvaluatorSurface(
            key="reduced",
            methods=("compute_node_energies_forces",),
        ),
        EvaluatorSurface(
            key="reduced_field",
            methods=("compute_node_energies_forces_field",),
        ),
        EvaluatorSurface(
            key="reduced_stress",
            methods=("_reduce_stress",),
            kokkos=True,
        ),
    ),
)
