#pragma once

#include <array>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "family_interface.hpp"
#include "implementation_registry.hpp"
#include "long_range_operator_kinds.hpp"
#include "mace_polar_long_range_contract.hpp"
#include "sha256.hpp"

// The PolarMACE backbone deltas as tiered-registry kinds.
//
// The family's node-local execution layer (mace_polar_deltas.hpp) is driven by
// the unrolled chain, and the chain's buffers are what the long-range kind
// group's nodes consume: the spin-resolved multipoles the electrostatic features and
// the self-interaction kind read, the equilibrated per-channel densities the
// energy, dipole, and correction kinds read, and the Fukui sources the
// equilibration reads.  Those buffers had no producer while the deltas had no
// declaration, so this group declares them -- one node per delta region, in
// the same declarative style as the long-range group (buffer contracts, scalar
// parameters, capability state, one portable registry descriptor each), and
// the family's program assembly folds both groups into the same program.
//
// The group is node-local: every kind's forward is per-node arithmetic (the
// dense equivariant linears, the tied-channel tensor products, the affine
// MLPs, the gated readout, and the per-graph normalization the equilibration
// performs), so no kind contributes a position or cell adjoint -- the
// geometry/cell reverse stays exactly where it was.  What the kinds declare is
// their *feature* derivative surface: the seeds of their inputs' feature
// spaces.
namespace symmetrix::execution::polar {

// ---------------------------------------------------------------------------
// Vocabulary
// ---------------------------------------------------------------------------

// The delta kind tokens, in declaration order.  They name the backbone's
// regions, not a second vocabulary: the schedule's backbone entries
// (``interactions``, ``feature_mixer``, ``field_update``,
// ``local_electron_energy``) are the regions, and these kinds are how a region
// produces and consumes the program's buffers.
inline constexpr std::array<std::string_view, 6> delta_kind_vocabulary{
    "polar_spin_source_heads",
    "polar_feature_mixer",
    "polar_source_map",
    "polar_field_update",
    "polar_charge_state",
    "polar_local_electron_energy",
};

struct DeltaKindSpec {
    OperatorKind kind = OperatorKind::custom;
    std::string_view name;
};

inline const std::vector<DeltaKindSpec>& delta_kind_specs()
{
    static const std::vector<DeltaKindSpec> specs = {
        {OperatorKind::custom, "polar_spin_source_heads"},
        {OperatorKind::custom, "polar_feature_mixer"},
        {OperatorKind::custom, "polar_source_map"},
        {OperatorKind::custom, "polar_field_update"},
        {OperatorKind::custom, "polar_charge_state"},
        {OperatorKind::custom, "polar_local_electron_energy"},
    };
    return specs;
}

inline std::optional<DeltaKindSpec> find_delta_kind(const std::string_view name)
{
    for (const DeltaKindSpec& spec : delta_kind_specs())
        if (spec.name == name)
            return spec;
    return std::nullopt;
}

// The delta group's own contract data: the backbone's per-layer feature width
// (the products' target irreps), the model's element count, the interaction
// count, and the checkpoint's self-interaction flag.  These are family data
// the parse already knows; the kinds parameterize their buffer contracts with
// them exactly as the long-range group parameterizes its own with the basis.
struct DeltaContract {
    int node_feature_dimension = 0;
    int num_elements = 0;
    int interaction_count = 0;
    bool subtract_feature_self_terms = true;

    bool declared() const
    {
        return node_feature_dimension > 0 && num_elements > 0
            && interaction_count > 0;
    }
};

// ---------------------------------------------------------------------------
// Buffer contracts
// ---------------------------------------------------------------------------

namespace detail {

inline BufferContract delta_buffer(std::string name, std::string dtype,
    std::vector<std::size_t> shape, std::string order)
{
    BufferContract contract;
    contract.name = std::move(name);
    contract.dtype = std::move(dtype);
    contract.shape = std::move(shape);
    contract.layout = LayoutContract{std::move(order), std::string(), 1};
    return contract;
}

} // namespace detail

// ---------------------------------------------------------------------------
// Node contracts
// ---------------------------------------------------------------------------

// Compile the delta group's nodes from the contract data that parameterizes
// them.  ``node_feature_dimension`` is the backbone's per-layer width (the
// products' target irreps) and ``num_elements`` the model's element count;
// both are contract data the family's parse already knows, so the nodes are
// as readable without a model as the long-range group's are.
inline std::vector<OperatorNode> delta_operator_nodes(
    const PolarLongRangeContract& contract,
    const int node_feature_dimension,
    const int num_elements,
    const int interaction_count)
{
    if (!validate_polar_long_range_contract(contract).empty()
        || node_feature_dimension < 1 || num_elements < 1
        || interaction_count < 1)
        return {};
    const std::size_t density_width = static_cast<std::size_t>(
        (contract.density_max_l + 1) * (contract.density_max_l + 1));
    const std::string layers = "1x1o";
    std::vector<OperatorNode> nodes;

    // The per-layer spin-source heads: one dense equivariant linear per
    // interaction layer, summed into the spin-resolved multipole state.
    {
        OperatorNode node;
        node.kind = OperatorKind::custom;
        node.custom_kind = "polar_spin_source_heads";
        node.id = "polar.delta.spin_source_heads";
        node.capability_key = "polar_spin_source_heads";
        node.inputs.push_back(detail::delta_buffer("node_features_layers",
            "float64", {0, 0}, "node_major"));
        node.outputs.push_back(detail::delta_buffer("spin_charge_density_sources",
            "float64", {0, contract.charges_dimension}, "node_major"));
        // One head per interaction layer (the extractor validates the count),
        // so the layer count is program structure and the feature width is
        // contract data: both participate in the program's identity.
        node.parameters.emplace_back(
            "interaction_count", static_cast<double>(interaction_count));
        node.parameters.emplace_back(
            "node_feature_dimension",
            static_cast<double>(node_feature_dimension));
        nodes.push_back(std::move(node));
    }

    // The feature mixer: one dense equivariant linear per layer over the
    // stacked per-layer features.
    {
        OperatorNode node;
        node.kind = OperatorKind::custom;
        node.custom_kind = "polar_feature_mixer";
        node.id = "polar.delta.feature_mixer";
        node.capability_key = "polar_feature_mixer";
        node.inputs.push_back(detail::delta_buffer("node_features_layers",
            "float64", {0, 0}, "node_major"));
        node.outputs.push_back(detail::delta_buffer("features_mixed",
            "float64", {0, static_cast<std::size_t>(node_feature_dimension)},
            "node_major"));
        node.parameters.emplace_back(
            "interaction_count", static_cast<double>(interaction_count));
        nodes.push_back(std::move(node));
    }

    // The pre-loop Fukui source map: the oracle's non-linear-bias readout over
    // the last layer's features, which the equilibration normalizes per graph.
    {
        OperatorNode node;
        node.kind = OperatorKind::custom;
        node.custom_kind = "polar_source_map";
        node.id = "polar.delta.source_map";
        node.capability_key = "polar_source_map";
        node.outputs.push_back(detail::delta_buffer("fukui_sources",
            "float64", {0, 2}, "node_major"));
        node.inputs.push_back(detail::delta_buffer("node_features",
            "float64", {0, static_cast<std::size_t>(node_feature_dimension)},
            "node_major"));
        node.parameters.emplace_back(
            "node_feature_dimension",
            static_cast<double>(node_feature_dimension));
        nodes.push_back(std::move(node));
    }

    // The field update: the potential embedding, the tied-channel contraction,
    // the affine MLP, the modulation, and the gated readout, once per unrolled
    // step.  Its readout emits the charge channels and the step's Fukui source
    // pair, which is why the sources have two producers (this node's per-step
    // output and the source map's pre-loop one).
    {
        OperatorNode node;
        node.kind = OperatorKind::custom;
        node.custom_kind = "polar_field_update";
        node.id = "polar.delta.field_update";
        node.capability_key = "polar_field_update";
        node.inputs.push_back(detail::delta_buffer("features_mixed",
            "float64", {0, static_cast<std::size_t>(node_feature_dimension)},
            "node_major"));
        node.inputs.push_back(detail::delta_buffer("potential_features_alpha",
            "float64", {0, contract.field_dimension}, "node_major"));
        node.inputs.push_back(detail::delta_buffer("potential_features_beta",
            "float64", {0, contract.field_dimension}, "node_major"));
        node.inputs.push_back(detail::delta_buffer("spin_charge_density",
            "float64", {0, contract.charges_dimension}, "node_major"));
        node.outputs.push_back(detail::delta_buffer("charge_sources",
            "float64", {0, contract.charges_dimension}, "node_major"));
        node.outputs.push_back(detail::delta_buffer("fukui_sources",
            "float64", {0, 2}, "node_major"));
        node.parameters.emplace_back("evaluation_count",
            static_cast<double>(contract.recursion_step_count));
        nodes.push_back(std::move(node));
    }

    // The charge state: the chain's accumulator.  It sums the heads' sources,
    // every unrolled step's charge sources, and the equilibration's l=0
    // corrections, and publishes the equilibrated state the long-range nodes
    // consume -- the per-channel densities (``multipoles_alpha/beta``), their
    // l=0 channels (``charge_l0_alpha/beta``), the spin-summed density
    // (``charge_density``), and the per-channel densities under the names the
    // correction kinds read (``charge_density_alpha/beta``).
    {
        OperatorNode node;
        node.kind = OperatorKind::custom;
        node.custom_kind = "polar_charge_state";
        node.id = "polar.delta.charge_state";
        node.capability_key = "polar_charge_state";
        node.inputs.push_back(detail::delta_buffer("spin_charge_density_sources",
            "float64", {0, contract.charges_dimension}, "node_major"));
        node.inputs.push_back(detail::delta_buffer("charge_sources",
            "float64", {0, contract.charges_dimension}, "node_major"));
        node.inputs.push_back(detail::delta_buffer("charge_correction_alpha",
            "float64", {0}, "node_major"));
        node.inputs.push_back(detail::delta_buffer("charge_correction_beta",
            "float64", {0}, "node_major"));
        // The state itself is published under the name the update and the
        // readout consume (the accumulator's feedback edge), and as the
        // per-channel and spin-summed densities the long-range kinds read.
        node.outputs.push_back(detail::delta_buffer("spin_charge_density",
            "float64", {0, contract.charges_dimension}, "node_major"));
        for (const char* name : {"multipoles_alpha", "multipoles_beta",
                "charge_density", "charge_density_alpha", "charge_density_beta"})
            node.outputs.push_back(detail::delta_buffer(name, "float64",
                {0, density_width}, "node_major"));
        for (const char* name : {"charge_l0_alpha", "charge_l0_beta"})
            node.outputs.push_back(detail::delta_buffer(name, "float64", {0},
                "node_major"));
        node.parameters.emplace_back("spin_channel_count",
            static_cast<double>(contract.spin_channel_count));
        // Once before the loop (the pre-loop equilibration) and once per
        // step: the state's multiplicity is the chain's own signature.
        node.parameters.emplace_back("evaluation_count",
            static_cast<double>(1 + contract.recursion_step_count));
        nodes.push_back(std::move(node));
    }

    // The post-loop local-electron readout: the per-node energy the family's
    // orchestration sums into the graph's local-electron term.
    {
        OperatorNode node;
        node.kind = OperatorKind::custom;
        node.custom_kind = "polar_local_electron_energy";
        node.id = "polar.delta.local_electron_energy";
        node.capability_key = "polar_local_electron_energy";
        node.inputs.push_back(detail::delta_buffer("node_features",
            "float64", {0, static_cast<std::size_t>(node_feature_dimension)},
            "node_major"));
        node.inputs.push_back(detail::delta_buffer("potential_features_alpha",
            "float64", {0, contract.field_dimension}, "node_major"));
        node.inputs.push_back(detail::delta_buffer("potential_features_beta",
            "float64", {0, contract.field_dimension}, "node_major"));
        node.inputs.push_back(detail::delta_buffer("spin_charge_density",
            "float64", {0, contract.charges_dimension}, "node_major"));
        node.outputs.push_back(detail::delta_buffer("local_electron_energy",
            "float64", {0}, "node_major"));
        nodes.push_back(std::move(node));
    }

    for (const DeltaKindSpec& spec : delta_kind_specs()) {
        bool declared = false;
        for (const OperatorNode& node : nodes)
            declared = declared or operator_kind_name(node) == spec.name;
        if (!declared)
            return {};
    }
    return nodes;
}

// ---------------------------------------------------------------------------
// Shape validation and the producer/consumer coherence rule
// ---------------------------------------------------------------------------

// Validate the delta nodes' buffer contracts against the contract data.  The
// rules mirror the long-range group's pass: the node set is complete (every
// declared kind exactly once), every buffer carries a dtype, an order, and at
// least one extent, the leading extent is the dynamic zero, and the buffers
// whose width the contract fixes carry it.
inline std::string validate_delta_node_shapes(
    const std::vector<OperatorNode>& nodes,
    const PolarLongRangeContract& contract,
    const int node_feature_dimension,
    const int num_elements,
    const int interaction_count)
{
    const std::string contract_reason =
        validate_polar_long_range_contract(contract);
    if (!contract_reason.empty())
        return contract_reason;
    if (node_feature_dimension < 1 || num_elements < 1 || interaction_count < 1)
        return "the delta group requires the backbone's feature width, the "
            "model's element count, and the interaction count";
    const std::size_t density_width = static_cast<std::size_t>(
        (contract.density_max_l + 1) * (contract.density_max_l + 1));
    std::map<std::string, const OperatorNode*> by_kind;
    for (const OperatorNode& node : nodes) {
        const std::string identity_reason =
            validate_operator_kind_identity(node);
        if (!identity_reason.empty())
            return identity_reason;
        const std::string kind = operator_kind_name(node);
        const auto spec = find_delta_kind(kind);
        if (!spec.has_value())
            return "the node set carries a non-delta kind: '" + kind + "'";
        if (node.kind != spec->kind)
            return "the delta kind '" + kind
                + "' must use its family-owned custom identity";
        if (by_kind.find(kind) != by_kind.end())
            return "the kind appears more than once: '" + kind + "'";
        by_kind[kind] = &node;
    }
    for (const DeltaKindSpec& spec : delta_kind_specs())
        if (by_kind.find(std::string(spec.name)) == by_kind.end())
            return "the node set is missing the declared kind: '"
                + std::string(spec.name) + "'";
    for (const auto& [kind, node] : by_kind) {
        for (const BufferContract& buffer : node->inputs)
            if (buffer.name.empty() || buffer.shape.empty()
                || buffer.layout.order != "node_major")
                return "'" + kind + "' declares an invalid input buffer";
        for (const BufferContract& buffer : node->outputs)
            if (buffer.name.empty() || buffer.shape.empty()
                || buffer.layout.order != "node_major")
                return "'" + kind + "' declares an invalid output buffer";
        // The densities are the buffers the long-range kinds consume, so their
        // columns are the density basis' width when the node hashes them out.
        for (const BufferContract& buffer : node->outputs) {
            const bool density_column =
                buffer.name == "multipoles_alpha" || buffer.name == "multipoles_beta"
                || buffer.name == "charge_density"
                || buffer.name == "charge_density_alpha"
                || buffer.name == "charge_density_beta";
            if (density_column && buffer.shape.back() != density_width)
                return "'" + kind + "' declares '" + buffer.name
                    + "' with a density channel width of "
                    + std::to_string(buffer.shape.back()) + ", expected "
                    + std::to_string(density_width);
            const bool charges_column =
                buffer.name == "spin_charge_density_sources"
                || buffer.name == "charge_sources"
                || buffer.name == "spin_charge_density";
            if (charges_column
                && buffer.shape.back() != contract.charges_dimension)
                return "'" + kind + "' declares '" + buffer.name
                    + "' with a charge width of "
                    + std::to_string(buffer.shape.back()) + ", expected "
                    + std::to_string(contract.charges_dimension);
            const bool feature_column = buffer.name == "features_mixed"
                || buffer.name == "node_features";
            if (feature_column
                && buffer.shape.back()
                    != static_cast<std::size_t>(node_feature_dimension))
                return "'" + kind + "' declares '" + buffer.name
                    + "' with a feature width of "
                    + std::to_string(buffer.shape.back()) + ", expected "
                    + std::to_string(node_feature_dimension);
            if (buffer.name == "fukui_sources"
                && buffer.shape.back() != 2)
                return "'" + kind + "' declares '" + buffer.name
                    + "' with a width other than the source pair";
        }
    }
    return {};
}

// ---------------------------------------------------------------------------
// Registry integration
// ---------------------------------------------------------------------------

// Declare the delta group's implementations into a family-backend registry.
// One portable (T1) descriptor per kind, exactly as the long-range group
// declares its own: the layer's kernels are node-local arithmetic over the
// parsed model's weights, so the portable tier is the production path and the
// host admission invariant is the group's only admission rule.  The identity
// fingerprint folds the delta contract data and the long-range contract, so
// two builds whose widths or basis differ never declare the same identity.
inline void declare_delta_implementations(
    ImplementationRegistry& registry,
    const std::string& target,
    const PolarLongRangeContract& contract,
    const std::string& compute,
    const DeltaContract& delta);

} // namespace symmetrix::execution::polar
