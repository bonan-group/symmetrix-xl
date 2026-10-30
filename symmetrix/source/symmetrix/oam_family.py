"""Family declaration for the OAM models exported from NequIP contracts.

The entry is discovered by the family registry from this module's ``_family``
suffix.  The family is detected by the architecture marker its exported JSON
carries, resolves its own literal execution-mode set, and declares the Kokkos
evaluator, the architecture fingerprint, the generator modules, and the artifact
adapters its direct routing needs.
"""

from __future__ import annotations

from .family_registry import (
    ArtifactAdapter,
    CapabilityDeclaration,
    ContractDeclaration,
    Detection,
    EvaluatorSurface,
    EvaluatorVariant,
    FamilyEntry,
    ModeVocabulary,
    Routing,
    Symbol,
)

_MISSING_EVALUATOR = (
    "native backend does not provide {class_name}; build Symmetrix with NequIP support"
)

FAMILY_ENTRY = FamilyEntry(
    key="OAM",
    precedence=3,
    # The architecture marker is this family's detection key: the frontend that
    # serves it reads the architecture and never the model type, so an exported
    # OAM contract cannot be captured by a model-type lookup.
    detection=Detection(architectures=("oam",)),
    evaluators=(
        EvaluatorVariant(
            kokkos=True,
            precision="float64",
            class_name="NequIPOAMKokkos",
            declared_error=_MISSING_EVALUATOR,
        ),
        EvaluatorVariant(
            kokkos=True,
            precision="float32",
            class_name="NequIPOAMKokkosFloat",
            declared_error=_MISSING_EVALUATOR,
        ),
    ),
    # The OAM execution modes are a literal native set: the runtime admits
    # direct receiver execution, the debug_v1 edge-owned ABI v1 artifact, and
    # the mlp_reference whole-evaluator entry point.  The frontend resolves the
    # two non-direct spellings onto the single reference routing, and no
    # compact spelling resolves here.
    modes=ModeVocabulary(
        aliases={
            "direct": "direct",
            "reference": "reference",
            "debug_v1": "reference",
            "mlp_reference": "reference",
        },
        primary=("direct", "reference"),
    ),
    # ``direct`` runs the prepared receiver lifecycle against a generated
    # artifact; ``reference`` runs the whole-evaluator generic entry point on
    # the compiler-free Kokkos implementation with no artifact at all.  The
    # one-shot entry point leaves no completed generation, so only the direct
    # routing reduces stress.
    routings=(
        Routing(key="direct", native_mode="direct", surface="receiver"),
        Routing(
            key="reference",
            native_mode="reference",
            surface="reference",
            reduces_stress=False,
        ),
    ),
    schema=Symbol("nequip_schema"),
    extractor=Symbol("extract_nequip_data", "extract_nequip_data"),
    # The exported contract is the whole model file, so this family declares no
    # in-file contract key and no frontend specialization.
    contract=ContractDeclaration(
        builder=Symbol("nequip_schema", "nequip_contract_fingerprint")
    ),
    generated_contract=ContractDeclaration(
        builder=Symbol("nequip_schema", "nequip_contract_fingerprint")
    ),
    codegen=(
        Symbol("oam_jit_codegen"),
        Symbol("oam_receiver_codegen"),
        Symbol("oam_cuda_codegen"),
    ),
    adapters=(
        Symbol("oam_jit_artifact"),
        Symbol("oam_cuda_artifact"),
    ),
    # The direct routing runs a prepared artifact, so it declares the entry
    # point that prepares one per execution target: the receiver/source ABI v7
    # host plugin for the host target, and the generated CUDA R1 plugin for the
    # CUDA target.  The frontend resolves both through this declaration, and the
    # CUDA artifact's identity is the envelope its manifest publishes, so the
    # frontend never reconstructs it.  The reference routing runs the
    # whole-evaluator entry point and declares no artifact at all.
    artifacts=(
        ArtifactAdapter(
            routing="direct",
            target="host",
            prepare=Symbol(
                "oam_jit_artifact", "prepare_nequip_oam_receiver_host_artifact"
            ),
        ),
        ArtifactAdapter(
            routing="direct",
            target="cuda",
            prepare=Symbol("oam_cuda_artifact", "prepare_nequip_oam_cuda_artifact"),
            manifest_path="manifest_path",
            evaluator_keyword="cuda_plugin_path",
            identity_keyword="cuda_artifact_identity",
        ),
    ),
    capabilities=CapabilityDeclaration(family="OAM v2 contract"),
    # The OAM frontend reads three surfaces: the mode handshake every evaluator
    # carries, the receiver prepared path the `direct` routing executes, and the
    # whole-evaluator entry point the `reference` routing executes.  A `direct`
    # request therefore fails naming `prepare_graph` rather than silently
    # evaluating through the reference path.
    surfaces=(
        EvaluatorSurface(
            key="mode_handshake",
            methods=("set_streamed_edges",),
            kokkos=True,
        ),
        EvaluatorSurface(
            key="receiver",
            methods=(
                "prepare_graph",
                "prepare_geometry",
                "compute_prepared",
                "reduce_forces",
            ),
            readiness=("portable", "generated"),
            kokkos=True,
        ),
        EvaluatorSurface(
            key="reference",
            methods=("compute_node_energies_forces",),
            kokkos=True,
        ),
    ),
)
