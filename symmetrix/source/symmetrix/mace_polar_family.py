"""Family declaration for the PolarMACE models (MACE-POLAR-1).

The entry is discovered by the family registry from this module's ``_family``
suffix.  The family is detected by its own model type -- the stamp the family's
extractor writes, so an MH-1 file is never admitted here and a polar file is
never captured by the MH-1 entry -- and it declares the public Polar sibling
evaluator, the Polar frontend schema and extractor, the
backbone-delta contract identity its models carry, and the two routings every
family declares.

What the family's *evaluator* is: ``MacePolarKokkosT``, a public family
evaluator that is a sibling of ``MaceNonlinearKokkosT`` over the same
non-public backbone engine.  It owns the ``POLAR_BACKBONE`` contract, native
whole-model transaction, electrostatic solver, reverse, and public outputs; it
does not construct or invoke the public nonlinear evaluator.  The thin Python
adapter prepares compact inputs and performs one whole-model call.

The global (non-neighbor-list) half of the family's program is tier-1: the
electrostatic, correction, equilibration, and readout nodes have no generated
implementation, which is why the family declares the MH-1 node program's
generator as the only code generator it owns.  The family's execution tiers are
declared by the program and the implementation registry, not by this entry.
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
    Routing,
    Symbol,
)
from .family_registry import COMPACT_MODE_VOCABULARY

# The polar family shares the compact frontend, so it declares the compact
# execution-mode vocabulary the compact model family owns, exactly as the MH-1
# family does.
FAMILY_ENTRY = FamilyEntry(
    key="MACE_Polar",
    # The compact frontend resolved the nonlinear family first, then the compact
    # model family, then this one, then the OAM contracts.
    precedence=2,
    detection=Detection(model_types=("MACE_Polar",)),
    evaluators=(
        EvaluatorVariant(
            kokkos=True,
            precision="float64",
            class_name="MACEPolarKokkos",
            declared_error=(
                "This Symmetrix build does not provide the requested "
                "native MACE_Polar Kokkos evaluator."
            ),
        ),
        EvaluatorVariant(
            kokkos=True,
            precision="float32",
            class_name="MACEPolarKokkosFloat",
            declared_error=(
                "This Symmetrix build does not provide the requested "
                "native MACE_Polar Kokkos evaluator."
            ),
        ),
    ),
    modes=COMPACT_MODE_VOCABULARY,
    # ``direct`` loads the generated MH-1 backbone subprogram; ``reference``
    # keeps the artifact-free portable MH-1 backbone through the whole-
    # evaluator entry point. The native-mode spelling is a frontend JIT control
    # token for this family and is not handed to the evaluator because the
    # public family frontend does not expose the generic ``mode_handshake``
    # surface. The internal backbone still supports prepared streamed-edge
    # execution. Both routes use the same native public-result collector:
    # isolated stress is exactly zero, while a periodic result includes the
    # reciprocal and analytic cell adjoints.
    routings=(
        Routing(key="direct", native_mode="direct", surface="whole_model"),
        Routing(key="reference", native_mode="materialized", surface="whole_model"),
    ),
    schema=Symbol("mace_polar_schema"),
    extractor=Symbol("extract_mace_polar", "extract_mace_polar_data"),
    # The family's models carry the backbone-delta identity beside the backbone's
    # own MH1_UVU contract; the builder is the polar identity's own.
    contract=ContractDeclaration(
        builder=Symbol(
            "mace_polar_execution_contract",
            "make_execution_polar_backbone_contract",
        ),
        json_key="POLAR_BACKBONE",
    ),
    generated_contract=ContractDeclaration(
        builder=Symbol(
            "mace_nonlinear_execution_contract", "make_execution_mh1_contract"
        ),
        validator=Symbol(
            "mace_polar_execution_contract",
            "validate_execution_polar_mh1_v4_contract_for_model",
        ),
        json_key="MH1_UVU",
        specialization="mh1",
    ),
    # The global nodes are tier-1 with no generated implementation, so the
    # family's only code generator is the backbone node program's -- the same
    # generator the MH-1 family's evaluator shares.
    codegen=(Symbol("mace_nonlinear_jit_codegen"),),
    # The staged backbone plugins are prepared by the shared JIT adapter from
    # that generator; the group's host plugin is the MH-1 family's own.
    adapters=(
        Symbol("jit", "prepare_jit_artifact"),
        Symbol("mace_nonlinear_jit_artifact"),
    ),
    artifacts=(
        ArtifactAdapter(
            routing="direct",
            target="host",
            prepare=Symbol("mace_nonlinear_jit_artifact", "prepare_mh1_host_artifact"),
            manifest_path="manifest_path",
        ),
    ),
    capabilities=CapabilityDeclaration(family="polar-mace"),
    frontend_adapter=Symbol("mace_polar_frontend", "MacePolarFrontendAdapter"),
    # The surfaces the frontend reads.  Both public routings execute the same
    # family-owned whole-model lifecycle; ``direct`` additionally prepares a
    # generated backbone artifact.  The prepared graph/geometry surfaces are
    # compact inputs to that lifecycle, not a second scientific evaluator.
    #
    # The public family frontend declares no generic streamed-edge
    # ``mode_handshake`` surface; its routing set is declared above instead.
    # This does not remove the internal backbone's streamed-edge capability.
    # The frontend resolves the declared route without handing a public mode to
    # the evaluator: direct loads its generated subprogram, while reference
    # keeps the native portable path without compiling or loading an artifact.
    surfaces=(
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
            key="polar_graph_inputs",
            methods=("set_polar_graph_inputs",),
            kokkos=True,
        ),
        EvaluatorSurface(
            key="whole_model",
            methods=("_evaluate_polar_prepared",),
            readiness=("portable", "generated"),
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
