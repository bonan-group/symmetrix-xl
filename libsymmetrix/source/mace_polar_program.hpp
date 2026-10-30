#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "family_interface.hpp"
#include "implementation_registry.hpp"
#include "long_range_operator_kinds.hpp"
#include "mace_polar_long_range_contract.hpp"
#include "mace_polar_delta_kinds.hpp"

// The MACE Polar family program combines the neutral internal backbone program
// with the family's own delta schedule and global electrostatic operators.
//
//   * the backbone -- the neutral internal backbone program the public Polar
//     sibling compiles from its own parsed model, node for node;
//   * the global (non-neighbor-list) operators -- the long-range kind group
//     (long_range_operator_kinds.hpp), parameterized by the POLAR_BACKBONE
//     contract's long-range slice.
//
// Plus the piece that makes the family a *schedule-driven* client: the
// unrolled non-self-consistent global<->local chain, expressed here as
// contract data exactly the way the backbone node program expresses its own
// stage schedule (``node_execution_schedule``). The scheme has no convergence
// criterion and
// no data-dependent control flow -- ``num_update`` fixed-depth iterations of
// electrostatic features -> external field -> the local charge update -> the
// closed-form Fukui equilibration -- so the chain is a fixed linear sequence
// whose only parameter is the recursion depth the checkpoint declares.  That
// is what makes the model a valid fixed-orchestration program client.
//
// This header is family-owned -- the tree's naming contract spells the MACE
// families with the ``mace_`` prefix, and PolarMACE is one of them -- and it
// holds no family state:
// every entry point is a pure function over the program, the contract data,
// and the schedule, so the assembly is testable without a model, a backend,
// or an evaluator -- the same property the long-range kind group's
// declarative layer has.  The schedule vocabulary is mirrored by the Python
// contract builder and pinned cross-language by the polar program probe.
namespace symmetrix::execution::polar {


// The family key the compiled program declares.  It is the frontend family
// entry's key spelled the way the other families spell theirs
// (``MACE_Nonlinear`` -> ``mace-nonlinear``).
inline constexpr std::string_view program_family = "polar-mace";

// ---------------------------------------------------------------------------
// The unrolled chain as contract data
// ---------------------------------------------------------------------------

// One entry of the chain: either a backbone region (the local work the MH-1
// node program and the polar-specific dense blocks run) or one global
// operator node of the long-range kind group.
struct PolarScheduleEntry {
    enum class Kind {
        backbone,
        global_operator,
    };
    Kind kind = Kind::backbone;
    // ``backbone``: a region token of the local vocabulary below.
    // ``global_operator``: a long-range kind token.
    std::string name;
    // The unrolled iteration the entry belongs to: 0 is the pre-loop
    // equilibration and the stages that precede it, 1..U are the unrolled
    // updates, and U+1 is the post-loop readout.  The index is what makes the
    // unrolled chain explicit rather than an implied loop.
    int iteration = 0;
};

// The local (backbone) region vocabulary.  The regions are the oracle's own
// stages (mace/modules/extensions.py, PolarMACE.forward): the interaction
// layers (which carry the per-layer spin-source multipole heads), the feature
// mixer, the per-iteration field-dependent charge update, and the post-loop
// local-electron-energy readout.  A region is a *region*, not a node: the
// backbone's own node program describes how it runs.
inline constexpr std::array<std::string_view, 4> backbone_region_vocabulary{
    "interactions",
    "feature_mixer",
    "field_update",
    "local_electron_energy",
};

// An empty entry list is this group's refusal idiom -- invalid input has no
// spelling, exactly as ``long_range_operator_nodes`` refuses invalid contract
// data -- so it means "no chain was declared", never "a chain of length zero".
// Two callers reach it: an invalid recursion depth, and an evaluator that
// carries no long-range contract at all.  A consumer that must tell those
// apart asks the accessor that owns the state
// (``has_polar_long_range_contract``), not the schedule.
struct PolarSchedule {
    std::uint32_t version = 1;
    std::uint32_t recursion_step_count = 1;
    std::vector<PolarScheduleEntry> entries;
};

inline bool polar_region_declared(const std::string_view name)
{
    for (const std::string_view region : backbone_region_vocabulary)
        if (region == name)
            return true;
    return false;
}

// The unrolled chain for a recursion depth.  ``recursion_step_count`` is the
// checkpoint fact the POLAR_BACKBONE contract carries (``num_update``); an
// invalid depth has no schedule spelling.
inline PolarSchedule polar_execution_schedule(const int recursion_step_count)
{
    PolarSchedule schedule;
    if (recursion_step_count < 1) {
        // The refusal carries a zero depth, so an empty chain never claims a
        // depth it does not have.
        PolarSchedule refused;
        refused.recursion_step_count = 0;
        return refused;
    }
    schedule.recursion_step_count =
        static_cast<std::uint32_t>(recursion_step_count);
    const auto entry = [] (const PolarScheduleEntry::Kind kind,
        std::string name, const int iteration) {
        PolarScheduleEntry value;
        value.kind = kind;
        value.name = std::move(name);
        value.iteration = iteration;
        return value;
    };
    const auto global = [&entry] (const char* name, const int iteration) {
        return entry(PolarScheduleEntry::Kind::global_operator, name, iteration);
    };
    const auto backbone = [&entry] (const char* name, const int iteration) {
        return entry(PolarScheduleEntry::Kind::backbone, name, iteration);
    };

    // The backbone's interaction layers (their own node schedule runs inside),
    // then the shared geometry cache, then the feature mixer and the pre-loop
    // Fukui equilibration -- the oracle's order.
    schedule.entries.push_back(backbone("interactions", 0));
    schedule.entries.push_back(global("kspace_geometry_cache", 0));
    schedule.entries.push_back(backbone("feature_mixer", 0));
    schedule.entries.push_back(global("fukui_normalize", 0));
    for (int step = 1; step <= recursion_step_count; ++step) {
        // One unrolled update: the long-range electrostatic features, the external-field
        // coupling, the local charge update, and the equilibration.
        schedule.entries.push_back(global("kspace_coulomb_features", step));
        schedule.entries.push_back(global("external_field_projection", step));
        schedule.entries.push_back(backbone("field_update", step));
        schedule.entries.push_back(global("fukui_normalize", step));
    }
    const int readout_iteration = recursion_step_count + 1;
    schedule.entries.push_back(
        backbone("local_electron_energy", readout_iteration));
    schedule.entries.push_back(global("dipole_readout", readout_iteration));
    schedule.entries.push_back(global("kspace_coulomb_energy", readout_iteration));
    schedule.entries.push_back(global("self_interaction_linear", readout_iteration));
    schedule.entries.push_back(
        global("analytic_correction_features", readout_iteration));
    schedule.entries.push_back(
        global("analytic_correction_energy", readout_iteration));
    return schedule;
}

// ---------------------------------------------------------------------------
// Assembly
// ---------------------------------------------------------------------------

// The buffers the *program* supplies rather than a node producing them: the
// prepared graph's geometry and indexing, the polar block's contract constant,
// and the family's graph-level scalar channel.  A delta node may consume these
// without a producer; anything else must be produced inside the program, which
// is the coherence rule the assembly enforces.
inline bool polar_program_input_buffer(const std::string_view name)
{
    return name == "positions" || name == "cell" || name == "node_graph_index"
        || name == "graph_pbc" || name == "field_feature_norms"
        || name == "external_potential" || name == "total_charge"
        || name == "total_spin" || name == "node_attrs"
        || name == "node_features_layers" || name == "node_features";
}

// The chain-state buffers the delta group *owns*: the long-range group's nodes
// consume them, and while the deltas had no declaration they had no producer at
// all -- which is what made the compiled program unrunnable.  The pass below
// uses the set explicitly: a long-range node consuming one of these buffers
// requires a *delta* producer, not merely some producer, so the rule cannot
// silently stop checking a renamed buffer by finding an unrelated name.
inline bool delta_produced_buffer(const std::string_view name)
{
    return name == "spin_charge_density_sources"
        || name == "charge_sources" || name == "features_mixed"
        || name == "fukui_sources" || name == "fukui_sources_update"
        || name == "multipoles_alpha" || name == "multipoles_beta"
        || name == "charge_l0_alpha" || name == "charge_l0_beta"
        || name == "charge_density" || name == "charge_density_alpha"
        || name == "charge_density_beta" || name == "local_electron_energy";
}

inline long_range::LongRangeChannelLayout polar_long_range_channel_layout()
{
    return long_range::LongRangeChannelLayout{
        .node_id_prefix = "polar.long_range",
        .channel_names = {"alpha", "beta"},
    };
}

inline BufferContract polar_buffer(std::string name, std::string dtype,
    std::vector<std::size_t> shape, std::string order, std::string irreps)
{
    BufferContract contract;
    contract.name = std::move(name);
    contract.dtype = std::move(dtype);
    contract.shape = std::move(shape);
    contract.layout = LayoutContract{std::move(order), std::move(irreps), 1};
    return contract;
}

// Fukui equilibration is Polar program semantics, not an electrostatic
// primitive: it assigns alpha/beta targets and applies the family's
// charge/spin constraint after each update.
inline OperatorNode polar_fukui_normalize_node()
{
    OperatorNode node;
    node.kind = OperatorKind::fukui_normalize;
    node.id = "polar.long_range.fukui_normalize";
    node.capability_key = "fukui_normalize";
    node.inputs.push_back(polar_buffer(
        "fukui_sources", "float64", {0, 2}, "node_major", "2x0e"));
    node.inputs.push_back(polar_buffer(
        "charge_l0_alpha", "float64", {0}, "node_major", "1x0e"));
    node.inputs.push_back(polar_buffer(
        "charge_l0_beta", "float64", {0}, "node_major", "1x0e"));
    node.inputs.push_back(polar_buffer(
        "total_charge", "float64", {0}, "graph_major", "1x0e"));
    node.inputs.push_back(polar_buffer(
        "total_spin", "float64", {0}, "graph_major", "1x0e"));
    node.inputs.push_back(polar_buffer(
        "node_graph_index", "int32", {0}, "node_major", "1x0e"));
    node.outputs.push_back(polar_buffer("charge_correction_alpha",
        "float64", {0}, "node_major", "1x0e"));
    node.outputs.push_back(polar_buffer("charge_correction_beta",
        "float64", {0}, "node_major", "1x0e"));
    return node;
}

inline bool same_buffer_contract(
    const BufferContract& lhs, const BufferContract& rhs)
{
    return lhs.name == rhs.name && lhs.dtype == rhs.dtype
        && lhs.shape == rhs.shape && lhs.layout.order == rhs.layout.order
        && lhs.layout.irreps == rhs.layout.irreps;
}

inline std::string validate_polar_long_range_node_shapes(
    const std::vector<OperatorNode>& nodes,
    const PolarLongRangeContract& contract)
{
    std::vector<OperatorNode> primitive_nodes;
    const OperatorNode* fukui = nullptr;
    for (const OperatorNode& node : nodes) {
        if (node.kind != OperatorKind::fukui_normalize) {
            primitive_nodes.push_back(node);
            continue;
        }
        if (fukui != nullptr)
            return "the Polar program declares Fukui normalization more than once";
        fukui = &node;
    }
    const std::string primitive_reason =
        long_range::validate_long_range_primitive_node_shapes(
            primitive_nodes, contract, polar_long_range_channel_layout());
    if (!primitive_reason.empty())
        return primitive_reason;
    if (fukui == nullptr)
        return "the Polar program is missing Fukui normalization";
    const OperatorNode expected = polar_fukui_normalize_node();
    if (fukui->id != expected.id
        || fukui->capability_key != expected.capability_key
        || fukui->inputs.size() != expected.inputs.size()
        || fukui->outputs.size() != expected.outputs.size())
        return "the Polar Fukui normalization node has the wrong contract";
    for (std::size_t index = 0; index < expected.inputs.size(); ++index)
        if (!same_buffer_contract(fukui->inputs[index], expected.inputs[index]))
            return "the Polar Fukui normalization input contract is invalid";
    for (std::size_t index = 0; index < expected.outputs.size(); ++index)
        if (!same_buffer_contract(fukui->outputs[index], expected.outputs[index]))
            return "the Polar Fukui normalization output contract is invalid";
    return {};
}

inline std::vector<OperatorNode> polar_long_range_operator_nodes(
    const PolarLongRangeContract& contract)
{
    if (!validate_polar_long_range_contract(contract).empty())
        return {};
    auto nodes = long_range::long_range_primitive_operator_nodes(
        contract, polar_long_range_channel_layout());
    const auto dipole = std::find_if(nodes.begin(), nodes.end(),
        [](const OperatorNode& node) {
            return node.kind == OperatorKind::dipole_readout;
        });
    nodes.insert(dipole, polar_fukui_normalize_node());
    for(auto& node:nodes) {
        switch(node.kind) {
        case OperatorKind::kspace_geometry_cache:
            node.parameters.emplace_back("spin_channel_count",
                static_cast<double>(contract.spin_channel_count));
            break;
        case OperatorKind::kspace_coulomb_features:
        case OperatorKind::external_field_projection:
            node.parameters.emplace_back("evaluation_count",
                static_cast<double>(contract.recursion_step_count));
            break;
        case OperatorKind::kspace_coulomb_energy:
            node.parameters.emplace_back("include_self_interaction",
                contract.include_electrostatic_self_interaction?1.0:0.0);
            break;
        case OperatorKind::self_interaction_linear:
            node.parameters.emplace_back("include_feature_self_interaction",
                contract.field_si?1.0:0.0);
            node.parameters.emplace_back("include_energy_self_interaction",
                contract.include_electrostatic_self_interaction?1.0:0.0);
            break;
        case OperatorKind::fukui_normalize:
            node.parameters.emplace_back("spin_channel_count",
                static_cast<double>(contract.spin_channel_count));
            node.parameters.emplace_back("equilibration_count",
                static_cast<double>(1+contract.recursion_step_count));
            break;
        default:
            break;
        }
    }
    return nodes;
}

// Validate one assembled PolarMACE program against the contract data and the
// schedule it was compiled from.  Returns the empty string when the program is
// coherent; otherwise a human-readable reason.  The rules:
//
//   * the family-owned long-range node set is complete and shape-consistent
//     with both the primitive contract and Polar's Fukui contract;
//   * every global schedule entry names a kind the program declares, and every
//     long-range node is reached by at least one schedule entry -- a chain
//     that skipped a node or named an absent one is not a program;
//   * every backbone region is in the region vocabulary, and the schedule's
//     depth is the contract's recursion depth.
inline std::string validate_polar_program(
    const Program& program,
    const PolarLongRangeContract& contract,
    const PolarSchedule& schedule,
    const DeltaContract& delta = {})
{
    const std::string contract_reason=
        validate_polar_long_range_contract(contract);
    if(!contract_reason.empty())return contract_reason;
    for (const OperatorNode& node : program.operators) {
        const std::string identity_reason =
            validate_operator_kind_identity(node);
        if (!identity_reason.empty())
            return identity_reason;
    }
    if (schedule.recursion_step_count
        != static_cast<std::uint32_t>(contract.recursion_step_count))
        return "the schedule depth must be the contract's recursion depth";
    if (schedule.entries.empty())
        return "the schedule must carry at least one entry";
    std::vector<OperatorNode> long_range_nodes;
    std::vector<std::string> long_range_names;
    for (const OperatorNode& node : program.operators) {
        const std::string kind = operator_kind_name(node);
        if (long_range::find_long_range_kind(kind).has_value()) {
            long_range_nodes.push_back(node);
            long_range_names.push_back(kind);
        }
    }
    const std::string shape_reason =
        validate_polar_long_range_node_shapes(long_range_nodes, contract);
    if (!shape_reason.empty())
        return shape_reason;
    // The delta group: its node set is complete and shape-consistent with the
    // contract data, and every buffer it consumes is either produced by the
    // program's own nodes or supplied by the program's inputs.  The rule is
    // what makes the assembled program runnable: a node whose inputs nothing
    // produces used to validate, which is how an unrunnable program shipped.
    std::vector<OperatorNode> delta_nodes;
    if (delta.declared()) {
        for (const OperatorNode& node : program.operators)
            if (find_delta_kind(operator_kind_name(node)).has_value())
                delta_nodes.push_back(node);
        const std::string delta_shape_reason = validate_delta_node_shapes(
            delta_nodes, contract, delta.node_feature_dimension,
            delta.num_elements, delta.interaction_count);
        if (!delta_shape_reason.empty())
            return delta_shape_reason;
        std::map<std::string, std::size_t> producers;
        std::map<std::string, std::size_t> delta_producers;
        for (const OperatorNode& node : program.operators)
            for (const BufferContract& output : node.outputs) {
                ++producers[output.name];
                if (find_delta_kind(operator_kind_name(node)).has_value())
                    ++delta_producers[output.name];
            }
        for (const std::string_view name : delta_kind_vocabulary) {
            const auto spec = find_delta_kind(name);
            if (!spec.has_value())
                continue;
            const OperatorNode* node = nullptr;
            for (const OperatorNode& candidate : delta_nodes)
                if (operator_kind_name(candidate) == name)
                    node = &candidate;
            if (node == nullptr)
                return "the program must declare the delta kind '"
                    + std::string(name) + "'";
            for (const BufferContract& input : node->inputs) {
                if (polar_program_input_buffer(input.name)
                    || producers.count(input.name) != 0)
                    continue;
                return std::string(name) + " consumes '" + input.name
                    + "', which no node of the program produces and which is "
                    "not a declared program input";
            }
        }
        // Every buffer that had no producer before the deltas were declared
        // must now have one, and -- where the buffer is the chain state -- that
        // producer must be a *delta* node: a same-named output from anywhere
        // else in the program is not the chain the long-range kinds read.
        for (const OperatorNode& node : program.operators) {
            const std::string kind = operator_kind_name(node);
            if (!long_range::find_long_range_kind(kind).has_value())
                continue;
            for (const BufferContract& input : node.inputs) {
                if (polar_program_input_buffer(input.name))
                    continue;
                if (delta_produced_buffer(input.name)) {
                    if (delta_producers.count(input.name) != 0)
                        continue;
                    return std::string(kind) + " consumes the chain-state "
                        "buffer '" + input.name
                        + "', which no delta node of the program produces";
                }
                if (producers.count(input.name) != 0)
                    continue;
                return std::string(kind) + " consumes '" + input.name
                    + "', which no node of the program produces and which is "
                    "not a declared program input";
            }
        }
    }
    const std::vector<OperatorNode> expected=
        polar_long_range_operator_nodes(contract);
    if (long_range_nodes.size() != expected.size())
        return "the program must declare every long-range kind exactly once";
    for (const PolarScheduleEntry& entry : schedule.entries) {
        if (entry.kind == PolarScheduleEntry::Kind::backbone) {
            if (!polar_region_declared(entry.name))
                return "unknown backbone region '" + entry.name + "'";
            continue;
        }
        if (!long_range::find_long_range_kind(entry.name).has_value())
            return "unknown long-range operator '" + entry.name + "'";
        bool declared = false;
        for (const std::string& name : long_range_names)
            declared = declared || name == entry.name;
        if (!declared)
            return "the schedule names a long-range operator the program does "
                "not declare: '" + entry.name + "'";
    }
    for (const std::string& name : long_range_names) {
        bool scheduled = false;
        for (const PolarScheduleEntry& entry : schedule.entries)
            scheduled = scheduled || (entry.kind
                    == PolarScheduleEntry::Kind::global_operator
                && entry.name == name);
        if (!scheduled)
            return "the program declares a long-range operator the schedule "
                "never reaches: '" + name + "'";
    }

    // The chain's multiplicity is program data.  Each kind whose evaluation
    // count the unrolled chain determines declares that count, so a program
    // and its chain cannot disagree about the depth they were compiled for --
    // and the count is what makes two depths different programs (the node set
    // itself is depth-independent).
    const auto scheduled_count = [&schedule] (const std::string_view kind) {
        std::size_t count = 0;
        for (const PolarScheduleEntry& entry : schedule.entries)
            if (entry.kind == PolarScheduleEntry::Kind::global_operator
                && entry.name == kind)
                ++count;
        return count;
    };
    for (const auto& [kind, parameter] :
        std::array<std::pair<const char*, const char*>, 3>{{
            {"kspace_coulomb_features", "evaluation_count"},
            {"external_field_projection", "evaluation_count"},
            {"fukui_normalize", "equilibration_count"}}}) {
        const OperatorNode* node = nullptr;
        for (const OperatorNode& candidate : long_range_nodes)
            if (operator_kind_name(candidate) == kind)
                node = &candidate;
        if (node == nullptr)
            continue; // the completeness check above already refused this
        std::optional<double> declared;
        for (const auto& [name, value] : node->parameters)
            if (name == parameter)
                declared = value;
        const std::size_t performed = scheduled_count(kind);
        if (!declared.has_value())
            return std::string(kind) + " must declare its chain multiplicity ('"
                + parameter + "')";
        if (*declared != static_cast<double>(performed))
            return std::string(kind) + " declares " + parameter + " = "
                + std::to_string(static_cast<int>(*declared))
                + ", but its chain performs it " + std::to_string(performed)
                + " time(s)";
    }
    return {};
}

// Compile the PolarMACE program from the backbone program and the contract
// data: the backbone's nodes first, then the long-range group's nodes, with
// the program identity stamped from the assembled node set.  The unrolled
// chain is derived from the contract's recursion depth and validated against
// the assembled program, so a compiled PolarMACE program is schedule-coherent
// by construction.  An invalid contract, an incomplete node set, or an
// incoherent schedule throws -- a program that cannot be validated has no
// compiled spelling.
inline Program compile_polar_program(
    const Program& backbone,
    const PolarLongRangeContract& contract,
    const DeltaContract& delta = {})
{
    const std::string reason =
        validate_polar_long_range_contract(contract);
    if (!reason.empty())
        throw std::invalid_argument(
            "the PolarMACE program requires valid contract data: " + reason);
    Program program;
    program.family = std::string(program_family);
    program.precision = backbone.precision;
    program.version = backbone.version;
    program.operators = backbone.operators;
    // The two family-owned groups, in the order the chain runs them: the
    // delta group's nodes produce the chain state the long-range nodes read.
    if (delta.declared())
        for (const OperatorNode& node : delta_operator_nodes(contract,
                delta.node_feature_dimension, delta.num_elements,
                delta.interaction_count))
            program.operators.push_back(node);
    for (const OperatorNode& node : polar_long_range_operator_nodes(contract))
        program.operators.push_back(node);
    const PolarSchedule schedule =
        polar_execution_schedule(contract.recursion_step_count);
    const std::string schedule_reason =
        validate_polar_program(program, contract, schedule, delta);
    if (!schedule_reason.empty())
        throw std::invalid_argument(
            "the PolarMACE program is incoherent: " + schedule_reason);
    program.fingerprint = program_fingerprint(program);
    return program;
}

// The family-backend's tiered registry: the backbone's own declarations plus
// the long-range group's portable descriptors.
// The delta group's registry declarations (declared in
// mace_polar_delta_kinds.hpp; defined here because the descriptor's identity
// folds both contracts).
inline void declare_delta_implementations(
    ImplementationRegistry& registry,
    const std::string& target,
    const PolarLongRangeContract& contract,
    const std::string& compute,
    const DeltaContract& delta)
{
    if (!delta.declared())
        return;
    sha256::Digest digest;
    digest.update("symmetrix-polar-delta-v1\n");
    digest.update("node_feature_dimension="
        + std::to_string(delta.node_feature_dimension) + "\n");
    digest.update("num_elements=" + std::to_string(delta.num_elements) + "\n");
    digest.update("interaction_count="
        + std::to_string(delta.interaction_count) + "\n");
    digest.update("self_terms=" + std::string(
        delta.subtract_feature_self_terms ? "subtract" : "include") + "\n");
    digest.update(polar_long_range_contract_fingerprint(contract));
    const std::string fingerprint = digest.finish();
    for (const DeltaKindSpec& spec : delta_kind_specs()) {
        ImplementationDescriptor descriptor;
        descriptor.id = "portable_polar_delta_" + std::string(spec.name);
        descriptor.kind = spec.kind;
        descriptor.custom_kind = std::string(spec.name);
        descriptor.capability_key = std::string(spec.name);
        descriptor.tier = ImplementationTier::portable;
        descriptor.target = target;
        descriptor.precision =
            PrecisionPolicy{compute, contract.accumulator_dtype};
        descriptor.layout = LayoutCompatibility{"node_major", "node_major"};
        descriptor.capabilities.target = target;
        descriptor.capabilities.precision = compute;
        descriptor.capabilities.portable_ready = true;
        descriptor.capabilities.stress = false;
        descriptor.capabilities.field = false;
        descriptor.capabilities.parameter_gradients = false;
        descriptor.identity_fingerprint = fingerprint;
        descriptor.qualified =
            [] (const ImplementationContext&) { return true; };
        registry.declare(std::move(descriptor));
    }
}

inline ImplementationRegistry polar_implementation_registry(
    const ImplementationRegistry& backbone,
    const std::string& target,
    const PolarLongRangeContract& contract,
    const std::string& compute = "float64",
    const DeltaContract& delta = {})
{
    ImplementationRegistry registry;
    for (const ImplementationDescriptor& descriptor : backbone.implementations())
        registry.declare(descriptor);
    long_range::declare_long_range_implementations(
        registry,target,contract,compute,
        polar_long_range_contract_fingerprint(contract));
    declare_delta_implementations(registry, target, contract, compute, delta);
    return registry;
}

} // namespace symmetrix::execution::polar
