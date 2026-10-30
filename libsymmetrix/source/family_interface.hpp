#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sha256.hpp"

// The native family-adapter interface: what a model family computes, expressed
// as pure versioned data, plus the lifecycle and capability queries the
// execution layer drives.  This header is family-neutral -- it carries no
// family token and no family state -- so every retained family adopts one
// contract instead of a per-family stack of parallel entry points.
//
// Two layers are separated here and nowhere else:
//
//   * the model-architecture layer (this header) declares what a family
//     computes: a compiled program, a DAG of operator nodes that carry contract
//     parameters and buffer/layout requirements, plus the capability set and
//     the artifact readiness of the family-backend combination;
//   * the execution layer (`implementation_registry.hpp`) decides how each
//     node runs and binds one implementation tier per node.
//
// A family adopts the interface by defining the neutral members the
// `FamilyEvaluator` concept requires.  Families own their definitions: the
// declarations for each retained evaluator live in that evaluator's own header
// and the definitions in its own semantic owner, so a family that changes an
// operator list never rebuilds a sibling.
namespace symmetrix::execution {

// ---------------------------------------------------------------------------
// Capability queries
// ---------------------------------------------------------------------------

// What one family-backend combination declares it can do.  `stress` is a
// first-class capability because it is not universal: a family-backend may
// evaluate energy and forces while its stress reduction is unavailable on that
// backend.  Declaration is honest: a capability reported true must be served by
// the corresponding entry point on this backend, and an absent one is reported
// false rather than silently degraded.
struct FamilyCapabilities {
    // Execution target this declaration describes: "host" for the default host
    // execution space, otherwise the accelerator name.
    std::string target;
    std::string precision;
    // Implementation tiers this family-backend admits (see ImplementationTier).
    bool reference_ready = false;
    bool portable_ready = false;
    bool generated_ready = false;
    // Properties.
    bool stress = false;
    bool field = false;
    bool parameter_gradients = false;
    // Artifact loading form the family-backend accepts.
    bool loads_host_artifact = false;
    bool loads_device_artifact = false;
};

// ---------------------------------------------------------------------------
// Model-to-program compilation: the operator DAG as pure versioned data
// ---------------------------------------------------------------------------

// The neutral operator vocabulary.  `custom` keeps a family free to name an
// operator the shared vocabulary does not cover without inventing a new shared
// kind for it.
enum class OperatorKind {
    geometry,
    radial,
    spherical_harmonic,
    tensor_product,
    linear,
    gate,
    readout,
    pair_energy,
    reduction,
    // The long-range (global, non-neighbor-list) operator group.  These
    // kinds are outside the edge vocabulary: their addressing extents are
    // k-points, graphs, and nodes rather than directed edges.  The
    // declarative layer -- buffer contracts, precision policy, capability
    // declarations, and the registry descriptors -- lives in
    // `long_range_operator_kinds.hpp`; no family wires them yet.
    kspace_geometry_cache,
    kspace_coulomb_features,
    kspace_coulomb_energy,
    self_interaction_linear,
    external_field_projection,
    fukui_normalize,
    dipole_readout,
    analytic_correction_energy,
    analytic_correction_features,
    node_program,
    custom,
};

inline const char* operator_kind_name(const OperatorKind kind)
{
    switch (kind) {
    case OperatorKind::geometry: return "geometry";
    case OperatorKind::radial: return "radial";
    case OperatorKind::spherical_harmonic: return "spherical_harmonic";
    case OperatorKind::tensor_product: return "tensor_product";
    case OperatorKind::linear: return "linear";
    case OperatorKind::gate: return "gate";
    case OperatorKind::readout: return "readout";
    case OperatorKind::pair_energy: return "pair_energy";
    case OperatorKind::reduction: return "reduction";
    case OperatorKind::kspace_geometry_cache: return "kspace_geometry_cache";
    case OperatorKind::kspace_coulomb_features: return "kspace_coulomb_features";
    case OperatorKind::kspace_coulomb_energy: return "kspace_coulomb_energy";
    case OperatorKind::self_interaction_linear: return "self_interaction_linear";
    case OperatorKind::external_field_projection:
        return "external_field_projection";
    case OperatorKind::fukui_normalize: return "fukui_normalize";
    case OperatorKind::dipole_readout: return "dipole_readout";
    case OperatorKind::analytic_correction_energy:
        return "analytic_correction_energy";
    case OperatorKind::analytic_correction_features:
        return "analytic_correction_features";
    case OperatorKind::node_program: return "node_program";
    case OperatorKind::custom: return "custom";
    }
    return "custom";
}

// Family-owned names occupy a namespace disjoint from the neutral enum.  A
// custom spelling that aliases a built-in would make the program fingerprint
// look built-in while the implementation registry still binds it as custom.
// Keep the identity check next to the vocabulary so every consumer applies
// the same collision rule.
inline bool reserved_operator_kind_name(const std::string_view name)
{
    static constexpr std::array<OperatorKind, 20> kinds = {
        OperatorKind::geometry,
        OperatorKind::radial,
        OperatorKind::spherical_harmonic,
        OperatorKind::tensor_product,
        OperatorKind::linear,
        OperatorKind::gate,
        OperatorKind::readout,
        OperatorKind::pair_energy,
        OperatorKind::reduction,
        OperatorKind::kspace_geometry_cache,
        OperatorKind::kspace_coulomb_features,
        OperatorKind::kspace_coulomb_energy,
        OperatorKind::self_interaction_linear,
        OperatorKind::external_field_projection,
        OperatorKind::fukui_normalize,
        OperatorKind::dipole_readout,
        OperatorKind::analytic_correction_energy,
        OperatorKind::analytic_correction_features,
        OperatorKind::node_program,
        OperatorKind::custom,
    };
    for (const OperatorKind kind : kinds)
        if (name == operator_kind_name(kind))
            return true;
    return false;
}

inline std::string validate_operator_kind_identity(
    const OperatorKind kind, const std::string_view custom_kind)
{
    if (kind == OperatorKind::custom) {
        if (custom_kind.empty())
            return "a custom operator kind must declare a non-empty family-owned name";
        if (reserved_operator_kind_name(custom_kind))
            return "custom operator kind '" + std::string(custom_kind)
                + "' collides with the reserved neutral operator vocabulary";
        return {};
    }
    if (!custom_kind.empty())
        return "built-in operator kind '" + std::string(operator_kind_name(kind))
            + "' must not carry a custom operator kind name";
    return {};
}

// The layout order of one buffer.  `node_major` stores a block as
// offset + channel * width + component and `lane_major` as
// offset + component * multiplicity + channel; both are the two orders the
// shared tensor kernels address.  `edge_major` addresses one row per directed
// edge and `block_major` one row per irrep block.
struct LayoutContract {
    std::string order = "node_major";
    // The irreps string whose block decomposition the order applies to.  It is
    // contract data, not a parsed object, so the program stays pure data.
    std::string irreps;
    std::uint32_t version = 1;
};

// The buffer contract of one operator operand or result.
struct BufferContract {
    std::string name;
    std::string dtype = "float64";
    std::vector<std::size_t> shape;
    LayoutContract layout;
    std::uint32_t version = 1;
};

// One node of the compiled program: its operator kind, a family-scoped stable
// id, the capability key the registry binds by, its buffer contracts, and the
// scalar contract parameters that define its numerics.
struct OperatorNode {
    OperatorKind kind = OperatorKind::custom;
    // Family-owned kinds use ``custom`` plus this stable identifier rather
    // than extending the shared enumeration with family vocabulary.
    std::string custom_kind;
    // Family-scoped stable node id, for example a layer-qualified operator name.
    std::string id;
    // The capability key the implementation registry binds by.  Distinct
    // operator instances that share an implementation family share this key.
    std::string capability_key;
    std::vector<BufferContract> inputs;
    std::vector<BufferContract> outputs;
    std::vector<std::pair<std::string, double>> parameters;
    std::uint32_t version = 1;
};

inline const char* operator_kind_name(const OperatorNode& node)
{
    if (node.kind == OperatorKind::custom && !node.custom_kind.empty())
        return node.custom_kind.c_str();
    return operator_kind_name(node.kind);
}

inline std::string validate_operator_kind_identity(const OperatorNode& node)
{
    return validate_operator_kind_identity(node.kind, node.custom_kind);
}

// The compiled program of one loaded model: pure versioned data, so the
// execution layer can plan it without reading family state.
struct Program {
    // Opaque family key supplied by the family layer.
    std::string family;
    std::string precision;
    std::uint32_t version = 1;
    std::vector<OperatorNode> operators;
    // Identity of the compiled program.  Families stamp this through
    // `program_fingerprint`; an empty stamp means the program was not compiled
    // against a contract identity yet.
    std::string fingerprint;
};

// The deterministic identity of a compiled program: kind, id, capability key,
// buffer layout orders, and the scalar parameters, in declaration order.  Only
// contract data contributes, so a program that plans identically fingerprints
// identically regardless of the process that compiled it.
inline std::string program_fingerprint(const Program& program)
{
    // Parameters are folded at full precision: two contracts whose extents
    // differ beyond a short decimal spelling must not fingerprint alike.
    std::ostringstream parameters;
    parameters << std::setprecision(17);
    sha256::Digest digest;
    digest.update("symmetrix-program-v1\n");
    digest.update(program.family + "\n");
    digest.update(program.precision + "\n");
    digest.update(std::to_string(program.version) + "\n");
    for (const auto& node : program.operators) {
        const std::string identity_reason =
            validate_operator_kind_identity(node);
        if (!identity_reason.empty())
            throw std::invalid_argument(identity_reason);
        // The neutral MH-1 node program predates its dedicated enum identity
        // and historically fingerprinted as the enum fallback "custom".
        // Preserve that stable artifact/contract byte while the registry and
        // validation surfaces use the collision-safe node_program identity.
        digest.update(node.kind == OperatorKind::node_program
                ? "custom"
                : operator_kind_name(node));
        digest.update("|" + node.id + "|" + node.capability_key + "|");
        digest.update(std::to_string(node.version) + "\n");
        for (const auto& input : node.inputs) {
            digest.update("in:" + input.name + "," + input.dtype + ",");
            digest.update(input.layout.order + "," + input.layout.irreps + ",");
            for (const auto size : input.shape)
                digest.update(std::to_string(size) + "x");
            digest.update("\n");
        }
        for (const auto& output : node.outputs) {
            digest.update("out:" + output.name + "," + output.dtype + ",");
            digest.update(output.layout.order + "," + output.layout.irreps + ",");
            for (const auto size : output.shape)
                digest.update(std::to_string(size) + "x");
            digest.update("\n");
        }
        for (const auto& [name, value] : node.parameters) {
            parameters.str(std::string());
            parameters.clear();
            parameters << "p:" << name << "=" << value << "\n";
            digest.update(parameters.str());
        }
    }
    return digest.finish();
}

// ---------------------------------------------------------------------------
// Artifact loading: readiness and identity through the artifact envelope
// ---------------------------------------------------------------------------

// One artifact load request.  `path` is empty for a readiness query that must
// not load anything; a nonempty path asks the family to admit that artifact.
struct ArtifactRequest {
    std::string path;
    std::string target;
};

// The readiness and identity a family reports for its generated artifacts.  The
// identity members are exactly the artifact-envelope fields every loader
// validates, so a family populates them from the envelope it already checks
// rather than from a parallel identity path.
struct ArtifactReadiness {
    bool ready = false;
    std::string target;
    std::string artifact_id;
    std::string generation_fingerprint;
    std::string semantic_fingerprint;
    std::string structure_fingerprint;
    // Why the artifact is not ready, or how it was admitted.
    std::string reason;
};

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

// The diagnostics the interface exposes: how often an evaluation fell back
// from a generated implementation to a portable or reference one, and the
// human-readable state of the last evaluation.  This is deliberately not a
// mirror of any family's full diagnostic surface.
struct FamilyDiagnostics {
    std::size_t fallback_count = 0;
    std::string diagnostic;
};

// ---------------------------------------------------------------------------
// The lifecycle contract
// ---------------------------------------------------------------------------

// A family evaluator satisfies this concept when it exposes the neutral
// lifecycle, capability, program-compilation, artifact, and diagnostic
// surface.  The lifecycle is the seven steps every retained family already
// performs under family-specific spellings:
//
//   1. compile_program()            model -> program (pure data)
//   2. prepare_graph()              admit a topology, return its generation
//   3. prepare_geometry()           bind geometry to that graph generation
//   4. compute_prepared()           evaluate the active geometry
//   5. reduce_forces()              reduce the retained adjoint to atom forces
//   6. reduce_stress()              reduce it to the stress tensor
//   7. family_diagnostics()         report fallbacks and the last state
//
// The generation counters make the reuse contract observable: a caller may
// re-evaluate within a generation and must prepare a new one after a topology
// or reference-geometry change.
template <typename Evaluator>
concept FamilyEvaluator = requires (
    Evaluator& evaluator,
    const Evaluator& const_evaluator,
    const std::uint64_t generation,
    const double volume,
    std::span<const int> integers,
    std::span<const double> reals) {
    {
        const_evaluator.family_capabilities()
    } -> std::convertible_to<FamilyCapabilities>;
    { const_evaluator.compile_program() } -> std::convertible_to<Program>;
    {
        evaluator.prepare_graph(0, integers, integers, integers, integers)
    } -> std::convertible_to<std::uint64_t>;
    { evaluator.prepare_geometry(generation, reals, reals, reals, reals, integers) };
    { evaluator.compute_prepared(generation, reals, reals) };
    { evaluator.reduce_forces(generation) };
    { evaluator.reduce_stress(volume, generation) };
    { const_evaluator.graph_generation() } -> std::convertible_to<std::uint64_t>;
    { const_evaluator.geometry_generation() } -> std::convertible_to<std::uint64_t>;
    {
        evaluator.load_family_artifact(ArtifactRequest{})
    } -> std::convertible_to<ArtifactReadiness>;
    { const_evaluator.family_artifact_ready() } -> std::convertible_to<bool>;
    {
        const_evaluator.family_artifact_readiness()
    } -> std::convertible_to<ArtifactReadiness>;
    { const_evaluator.family_diagnostics() } -> std::convertible_to<FamilyDiagnostics>;
};

} // namespace symmetrix::execution
