#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "family_interface.hpp"
#include "implementation_registry.hpp"
#include "sha256.hpp"

// The declarative layer of the long-range operator group: the global
// (non-neighbor-list) electrostatic primitives used by long-range model
// families as tiered-registry kinds.
//
// This header declares reusable electrostatic data and predicates for the
// registry it feeds: the operator-kind vocabulary, the buffer
// contracts and precision policy of every kind, the capability state each
// kind admits, and the implementation descriptors a family folds into its
// registry.  No family is *defined* here: the PolarMACE family (the fourth
// adapter) folds `long_range_primitive_operator_nodes` into its compiled program and
// `declare_long_range_implementations` into its registry through its own
// assembly (mace_polar_program.hpp), so the group stays a set of descriptors
// and node contracts an adopting family consumes.  The registry's own shape
// needs no global instance for this: an `ImplementationRegistry` is
// per-family-backend data.  This layer is exercised by the registry probe
// (symmetrix/test/test_long_range_operator_kinds.py) and by the conformance
// battery, which reads the adopted program's own node contracts
// (symmetrix/test/test_long_range_conformance.py).
//
// Shape decisions this layer freezes (the design note carries the physics):
//
//   * The analytic box/slab corrections are TWO kinds -- one per output path
//     (a per-graph energy scalar, per-node corrective feature fields) --
//     because a kind is the registry's unit of uniform buffer contract.  The
//     correction MODE is not a kind. Production Polar evaluation admits one
//     complete system, whose pbc pattern selects one runtime mode through the
//     `graph_pbc` input. Mixed or multi-system batches are rejected before
//     solver allocation.
//   * Every long-range feature/energy path runs per spin channel under one node per kind: the
//     channel is a data axis of the multipoles/features buffers, not a
//     program fork.  The energy, dipole, and energy-correction paths consume
//     the spin-summed charge density; the family's orchestration performs
//     that one sum.
//   * One geometry cache serves features, energy, both spin channels, and
//     every recursion step: the features and energy nodes declare the cache
//     outputs as identically-named identically-shaped inputs, which is the
//     binding-level sharing the conformance suite asserts.
//   * Precision policy: fp64 accumulators everywhere the upstream
//     `safe_double` casts sit in the electrostatic reductions. Delivered
//     contracts also use fp64 dense
//     tables. The accumulator and table dtype are explicit contract data; an
//     fp32-table experiment would require a different qualified contract.
namespace symmetrix::execution::long_range {

// ---------------------------------------------------------------------------
// Vocabulary
// ---------------------------------------------------------------------------

// The analytic correction-mode tokens, exactly the long-range capacity
// vocabulary of the support manifest (benchmarks/
// foundation_model_support_matrix.py, LONG_RANGE_CORRECTION_MODES).  A
// graph's pbc class selects one token at runtime.
inline constexpr std::array<std::string_view, 3> correction_mode_vocabulary{
    "makov-payne-dabo",
    "slab-dipole",
    "none",
};

// The derivative-mode vocabulary of the group: forward everywhere, the
// position adjoints of the reverse-mode stage (every kind whose forward
// consumes positions), the normalization adjoint of the global
// scatter-and-broadcast kind (the Fukui equilibration), and the
// reciprocal-space cell adjoint (the Ewald stress), which stays
// experimental because no analytic serial oracle ships for it.
inline constexpr std::array<std::string_view, 4> derivative_mode_vocabulary{
    "forward",
    "position_adjoint",
    "normalization_adjoint",
    "cell_adjoint",
};

// Fixed unit-convention constants of the layer (design note, Kernel
// inventory and Unit convention): the half-sphere factor-2 weights that live
// in the features and energy kernels, the 0.5 the external-field gauge
// carries per spin channel, and the cubic Madelung constant of the molecular
// box term.
inline constexpr double half_sphere_weight = 2.0;
inline constexpr double external_field_scale = 0.5;
inline constexpr double cubic_madelung = 2.837297;

// ---------------------------------------------------------------------------
// Contract data
// ---------------------------------------------------------------------------

// Reusable electrostatic primitive data: density/feature basis descriptors,
// public tensor dimensions, and precision policy. Family programs supply
// channel names and node namespaces separately; recursion, spin layout, and
// family self-term policy do not belong to this neutral contract.
struct LongRangeContract {
    // field.kspace_cutoff: the model's own k-space cutoff (the checkpoint
    // fact kspace_cutoff, not the heuristic or the factor).
    double kspace_cutoff = 0.0;
    // charges.max_l and charges.smearing_width: the density basis is one
    // smearing width wide per spin channel.
    int density_max_l = -1;
    double density_smearing_width = 0.0;
    // field.max_l and field.smearing_widths: the feature basis carries one
    // spherical-harmonics set per smearing width (the sigma set).
    int feature_max_l = -1;
    std::vector<double> feature_smearing_widths;
    // charges.dimension is an integer number of density bases,
    // field.dimension = (feature_max_l + 1)^2 * sigma count,
    // potential.dimension is the same number of feature bases.
    std::size_t charges_dimension = 0;
    std::size_t field_dimension = 0;
    std::size_t potential_dimension = 0;
    double field_norm_factor = 1.0;
    // Precision policy: the dense tables' dtype is contract data, and the
    // delivered implementation's is fp64.  The tier-1 kernels carry every
    // table in double (long_range_kernels.hpp: the geometry cache's cos/sin
    // pair, the per-k factors, the analytic Fourier factors), and the
    // measured fp32 quantization of those tables misses the group's own
    // parity gate: recomputing the fixture energies with fp32-rounded tables
    // moves them by 1.2e-9 (mol5) and 7.8e-9 (pbc4) relative -- three to four
    // orders of magnitude above the 1e-12 fp64 gate the group holds, and the
    // known fp32-Coulomb-sum hazard the design note records.  The
    // declaration therefore states what is delivered; an fp32 table delivery
    // is a tolerance-gated optimization that would have to move this field
    // with it (and the capacity record's table_bytes follows it).
    std::string table_dtype = "float64";
    // Precision policy: fp64 accumulators, mirroring upstream safe_double.
    std::string accumulator_dtype = "float64";
};

// Family-owned programs supply the names of their independent electrostatic
// channels and their node-id namespace explicitly.  The primitive layer uses
// this data only to declare repeated kernel inputs and outputs; it does not
// assign spin meaning or add a family equilibration program.
struct LongRangeChannelLayout {
    std::string node_id_prefix;
    std::vector<std::string> channel_names;
};

// Validate contract data for the kind group.  Returns the empty string when
// the data admits the group; otherwise a human-readable rejection reason.
inline std::string validate_long_range_contract(const LongRangeContract& contract)
{
    const auto finite_positive = [] (const double value) {
        return std::isfinite(value) && value > 0.0;
    };
    if (!finite_positive(contract.kspace_cutoff))
        return "kspace_cutoff must be finite and positive";
    // The analytic GTO radial factors are closed forms that hard-reject
    // l > 1 (RadialIntegralDirect), so the kind group admits exactly the
    // shipped bases and says so at declaration time rather than at kernel
    // time.
    if (contract.density_max_l < 0 || contract.density_max_l > 1)
        return "density_max_l must be 0 or 1 (the analytic radial Fourier "
            "factors reject l > 1)";
    if (contract.feature_max_l < 0 || contract.feature_max_l > 1)
        return "feature_max_l must be 0 or 1 (the analytic radial Fourier "
            "factors reject l > 1)";
    if (!finite_positive(contract.density_smearing_width))
        return "density_smearing_width must be finite and positive";
    if (contract.feature_smearing_widths.empty())
        return "feature_smearing_widths must carry at least one smearing width";
    for (const double width : contract.feature_smearing_widths) {
        if (!finite_positive(width))
            return "feature_smearing_widths must be finite and positive";
    }
    const std::size_t density_width = static_cast<std::size_t>(
        (contract.density_max_l + 1) * (contract.density_max_l + 1));
    const std::size_t field_width = static_cast<std::size_t>(
        (contract.feature_max_l + 1) * (contract.feature_max_l + 1))
        * contract.feature_smearing_widths.size();
    if (contract.charges_dimension == 0
        || contract.charges_dimension % density_width != 0)
        return "charges_dimension must be a positive integer multiple of "
            "(density_max_l + 1)^2";
    const std::size_t channels=contract.charges_dimension/density_width;
    if (contract.field_dimension != field_width)
        return "field_dimension must be (feature_max_l + 1)^2 * the sigma "
            "count = " + std::to_string(field_width);
    if (contract.potential_dimension != channels * contract.field_dimension)
        return "potential_dimension must be channel_count * "
            "field_dimension = "
            + std::to_string(channels * contract.field_dimension);
    if (!std::isfinite(contract.field_norm_factor)
        || contract.field_norm_factor == 0.0)
        return "field_norm_factor must be finite and nonzero";
    if (contract.table_dtype != "float32" && contract.table_dtype != "float64")
        return "table_dtype must be float32 or float64";
    if (contract.accumulator_dtype != "float64")
        return "accumulator_dtype must be float64 (the long-range precision "
            "policy keeps fp64 accumulators)";
    return {};
}

inline std::string validate_long_range_channel_layout(
    const LongRangeContract& contract,
    const LongRangeChannelLayout& layout)
{
    const std::string contract_reason = validate_long_range_contract(contract);
    if (!contract_reason.empty())
        return contract_reason;
    const std::size_t density_width = static_cast<std::size_t>(
        (contract.density_max_l + 1) * (contract.density_max_l + 1));
    const std::size_t channels = contract.charges_dimension / density_width;
    if (layout.node_id_prefix.empty())
        return "the long-range node-id prefix must not be empty";
    if (layout.channel_names.size() != channels)
        return "the long-range channel layout must name every contract channel";
    std::map<std::string, bool> names;
    for (const std::string& name : layout.channel_names) {
        if (name.empty())
            return "long-range channel names must not be empty";
        if (!names.emplace(name, true).second)
            return "long-range channel names must be unique";
    }
    return {};
}

// The deterministic identity of one contract instance: every field that
// parameterizes the kinds, folded at full precision so two contracts whose
// extents differ beyond a short decimal spelling never fingerprint alike.
inline std::string long_range_contract_fingerprint(const LongRangeContract& contract)
{
    std::ostringstream parameters;
    parameters << std::setprecision(17);
    sha256::Digest digest;
    digest.update("symmetrix-long-range-contract-v1\n");
    const auto fold = [&digest, &parameters] (
        const std::string& name, const double value) {
        parameters.str(std::string());
        parameters.clear();
        parameters << name << "=" << value << "\n";
        digest.update(parameters.str());
    };
    fold("kspace_cutoff", contract.kspace_cutoff);
    fold("density_max_l", static_cast<double>(contract.density_max_l));
    fold("density_smearing_width", contract.density_smearing_width);
    fold("feature_max_l", static_cast<double>(contract.feature_max_l));
    for (std::size_t index = 0;
        index < contract.feature_smearing_widths.size(); ++index) {
        fold("feature_smearing_width_" + std::to_string(index),
            contract.feature_smearing_widths[index]);
    }
    fold("charges_dimension",
        static_cast<double>(contract.charges_dimension));
    fold("field_dimension", static_cast<double>(contract.field_dimension));
    fold("potential_dimension",
        static_cast<double>(contract.potential_dimension));
    fold("field_norm_factor", contract.field_norm_factor);
    digest.update("table:" + contract.table_dtype + "\n");
    digest.update("accumulator:" + contract.accumulator_dtype + "\n");
    return digest.finish();
}

// ---------------------------------------------------------------------------
// Kind specs
// ---------------------------------------------------------------------------

// The capability declaration of one kind: which derivative and reduction
// surfaces the group serves at this declaration.  Every flag is backed by an
// implemented kernel surface -- the reverse-mode stage's adjoints
// (long_range_kernels.hpp, FD-verified in test_long_range_adjoints.py and
// test_long_range_stress.py) are what the flags declare:
//
//   * `position_adjoint` -- the kind's forward consumes positions and its
//     position adjoint exists (the geometry cache, the electrostatic feature
//     and energy chains, the external-field gauge, the dipole readout, and
//     both analytic corrections). The self-interaction constants and the
//     Fukui normalization have no position dependence and declare it false.
//   * `cell_adjoint` -- declared on the compatibility kinds whose periodic
//     forward carries cell-dependent terms and whose reciprocal cell adjoint
//     exists: geometry/tables, electrostatic features and energy, and analytic
//     correction volume terms. On the fully non-periodic real-space route,
//     fixed-Cartesian cell adjoints are exactly zero; public isolated stress is
//     therefore exactly zero rather than a reciprocal virial. The historical
//     `kspace_*` kind identifiers remain part of the compatible contract.
//   * `normalization_adjoint` -- the global scatter-and-broadcast adjoint:
//     the Fukui equilibration's own derivative surface (the forward's
//     per-graph norm and broadcast, both differentiated).
struct LongRangeKindCapabilities {
    bool forward = true;
    bool position_adjoint = false;
    bool cell_adjoint = false;
    bool normalization_adjoint = false;
    // The kind couples an external electric field into the features.
    bool serves_external_field = false;
    // The kind is the group's global scatter-and-broadcast operator: its
    // forward normalizes by a per-graph scatter and broadcast, so its
    // adjoint is the scatter of a broadcast gradient -- the first of that
    // class in the framework.
    bool global_scatter_broadcast = false;
};

// One kind's declarative identity: the registry vocabulary token and the
// capability state.  Buffer contracts and parameters come from contract data
// (`long_range_operator_nodes`); the implementation descriptors from
// `declare_long_range_implementations`.
struct LongRangeKindSpec {
    OperatorKind kind = OperatorKind::custom;
    std::string_view name;
    LongRangeKindCapabilities capabilities;
};

inline const std::vector<LongRangeKindSpec>& long_range_kind_specs()
{
    // The capability flags follow the implemented adjoint surfaces exactly
    // (the design note's Adjoint set, all FD-verified): position adjoints
    // where the forward consumes positions, the cell adjoint where the
    // forward carries cell-dependent terms, and the normalization adjoint
    // on the global scatter-and-broadcast kind.
    static const std::vector<LongRangeKindSpec> specs = {
        {OperatorKind::kspace_geometry_cache, "kspace_geometry_cache",
            LongRangeKindCapabilities{
                .position_adjoint = true, .cell_adjoint = true}},
        {OperatorKind::kspace_coulomb_features, "kspace_coulomb_features",
            LongRangeKindCapabilities{
                .position_adjoint = true, .cell_adjoint = true}},
        {OperatorKind::kspace_coulomb_energy, "kspace_coulomb_energy",
            LongRangeKindCapabilities{
                .position_adjoint = true, .cell_adjoint = true}},
        {OperatorKind::self_interaction_linear, "self_interaction_linear",
            {}},
        {OperatorKind::external_field_projection, "external_field_projection",
            LongRangeKindCapabilities{
                .position_adjoint = true, .serves_external_field = true}},
        {OperatorKind::fukui_normalize, "fukui_normalize",
            LongRangeKindCapabilities{
                .normalization_adjoint = true,
                .global_scatter_broadcast = true}},
        {OperatorKind::dipole_readout, "dipole_readout",
            LongRangeKindCapabilities{.position_adjoint = true}},
        {OperatorKind::analytic_correction_energy, "analytic_correction_energy",
            LongRangeKindCapabilities{
                .position_adjoint = true, .cell_adjoint = true}},
        {OperatorKind::analytic_correction_features,
            "analytic_correction_features",
            LongRangeKindCapabilities{
                .position_adjoint = true, .cell_adjoint = true}},
    };
    return specs;
}

// The vocabulary query.  An unknown name is a clean empty result, never a
// fallback to `custom` and never a crash; callers surface the miss.
inline std::optional<LongRangeKindSpec> find_long_range_kind(
    const std::string_view name)
{
    for (const LongRangeKindSpec& spec : long_range_kind_specs()) {
        if (spec.name == name)
            return spec;
    }
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// Shape validation
// ---------------------------------------------------------------------------

namespace detail {

// The buffers dimensioned by the density basis' channel width, and those
// dimensioned by the feature basis' channel width.  The sets are explicit:
// a kind that adds a buffer must add it here, so the pass cannot silently
// stop checking a renamed buffer.
inline bool channel_buffer(const std::string_view name,
    const std::string_view prefix,
    const LongRangeChannelLayout& layout)
{
    if (!name.starts_with(prefix))
        return false;
    const std::string_view channel = name.substr(prefix.size());
    return std::find(layout.channel_names.begin(), layout.channel_names.end(),
        channel) != layout.channel_names.end();
}

inline bool density_column_buffer(const std::string_view name,
    const LongRangeChannelLayout& layout)
{
    return name == "charge_density" || name == "energy_self_fields"
        || channel_buffer(name, "multipoles_", layout)
        || channel_buffer(name, "charge_density_", layout);
}

inline bool field_column_buffer(const std::string_view name,
    const LongRangeChannelLayout& layout)
{
    return channel_buffer(name, "potential_features_", layout)
        || channel_buffer(name, "feature_self_terms_", layout)
        || channel_buffer(name, "external_field_features_", layout)
        || channel_buffer(name, "correction_fields_", layout);
}

// The fully static constants: their leading extent is a positive literal, not
// the dynamic row count.
inline bool static_extent_buffer(const std::string_view name)
{
    return name == "cell" || name == "field_feature_norms";
}

inline bool buffer_dtype_declared(const std::string_view dtype)
{
    return dtype == "float32" || dtype == "float64" || dtype == "int32";
}

inline bool buffer_order_declared(const std::string_view order)
{
    return order == "node_major" || order == "k_major"
        || order == "graph_major";
}

} // namespace detail

// Validate the buffer contracts of the kind group's nodes against the contract
// data that parameterizes them.  Returns the empty string when every node's
// buffers are shape-consistent; otherwise a human-readable reason naming the
// offending buffer.  The pass is what makes the declarative layer's extents
// program-readable rather than conventional:
//
//   * the node set is complete -- every declared kind exactly once, and no
//     node of an undeclared kind;
//   * every buffer carries a declared dtype and layout order, a non-empty
//     name, and at least one extent;
//   * the leading extent is the dynamic 0 (the row count is runtime capacity)
//     except for the fully static constants, whose leading extent is a
//     positive literal;
//   * every density-column buffer carries the contract's density channel
//     width and every field-column buffer the contract's feature dimension --
//     including the self-interaction kind's two adjoint seed surfaces, whose
//     columns are exactly the seeds its transposes consume;
//   * the shared cache tables the features and energy nodes consume match the
//     geometry cache's own contracts field for field (the one-shared-cache
//     rule at the buffer level).
inline std::string validate_long_range_primitive_node_shapes(
    const std::vector<OperatorNode>& nodes,
    const LongRangeContract& contract,
    const LongRangeChannelLayout& layout)
{
    const std::string contract_reason = validate_long_range_contract(contract);
    if (!contract_reason.empty())
        return contract_reason;
    const std::string layout_reason =
        validate_long_range_channel_layout(contract, layout);
    if (!layout_reason.empty())
        return layout_reason;
    const std::size_t density_width = static_cast<std::size_t>(
        (contract.density_max_l + 1) * (contract.density_max_l + 1));
    const std::size_t field_width = static_cast<std::size_t>(
        (contract.feature_max_l + 1) * (contract.feature_max_l + 1))
        * contract.feature_smearing_widths.size();

    std::map<std::string, const OperatorNode*> by_kind;
    for (const OperatorNode& node : nodes) {
        const std::string identity_reason =
            validate_operator_kind_identity(node);
        if (!identity_reason.empty())
            return identity_reason;
        const std::string kind = operator_kind_name(node);
        const auto spec = find_long_range_kind(kind);
        if (!spec.has_value())
            return "the node set carries a non-long-range kind: '" + kind + "'";
        if (spec->kind == OperatorKind::fukui_normalize)
            return "the primitive node set carries a family normalization kind";
        if (node.kind != spec->kind)
            return "the long-range kind '" + kind
                + "' must use its reserved neutral enum identity";
        if (node.id != layout.node_id_prefix + "." + kind)
            return "the long-range kind '" + kind
                + "' must use the family-owned node-id namespace";
        if (by_kind.find(kind) != by_kind.end())
            return "the kind appears more than once: '" + kind + "'";
        by_kind[kind] = &node;
    }
    for (const LongRangeKindSpec& spec : long_range_kind_specs()) {
        if (spec.kind == OperatorKind::fukui_normalize)
            continue;
        if (by_kind.find(std::string(spec.name)) == by_kind.end())
            return "the node set is missing the declared kind: '"
                + std::string(spec.name) + "'";
    }

    for (const auto& [kind, node] : by_kind) {
        const auto check = [&kind, density_width, field_width, &layout] (
            const BufferContract& buffer) -> std::string {
            if (buffer.name.empty())
                return "a buffer of '" + kind + "' is unnamed";
            if (!detail::buffer_dtype_declared(buffer.dtype))
                return "'" + kind + "' declares an unknown dtype '" + buffer.dtype
                    + "' for '" + buffer.name + "'";
            if (!detail::buffer_order_declared(buffer.layout.order))
                return "'" + kind + "' declares an unknown layout order '"
                    + buffer.layout.order + "' for '" + buffer.name + "'";
            if (buffer.shape.empty())
                return "'" + kind + "' declares '" + buffer.name
                    + "' without a shape";
            if (detail::static_extent_buffer(buffer.name)) {
                if (buffer.shape.front() == 0)
                    return "'" + kind + "' declares the static buffer '"
                        + buffer.name + "' with a dynamic leading extent";
            } else if (buffer.shape.front() != 0) {
                return "'" + kind + "' declares '" + buffer.name
                    + "' with a static leading extent";
            }
            if (detail::density_column_buffer(buffer.name, layout)
                && buffer.shape.back() != density_width)
                return "'" + kind + "' declares '" + buffer.name
                    + "' with a density channel width of "
                    + std::to_string(buffer.shape.back()) + ", expected "
                    + std::to_string(density_width);
            if (detail::field_column_buffer(buffer.name, layout)
                && buffer.shape.back() != field_width)
                return "'" + kind + "' declares '" + buffer.name
                    + "' with a feature channel width of "
                    + std::to_string(buffer.shape.back()) + ", expected "
                    + std::to_string(field_width);
            // The two analytic Fourier tables carry their basis' channel
            // width as the middle extent (the trailing extent is the real/
            // imaginary pair).
            if (buffer.name == "density_fourier_factors"
                || buffer.name == "feature_fourier_factors") {
                const std::size_t expected =
                    buffer.name == "density_fourier_factors"
                    ? density_width : field_width;
                if (buffer.shape.size() != 3 || buffer.shape[1] != expected)
                    return "'" + kind + "' declares '" + buffer.name
                        + "' with a channel width of "
                        + (buffer.shape.size() == 3
                            ? std::to_string(buffer.shape[1]) : "the wrong rank")
                        + ", expected " + std::to_string(expected);
            }
            return {};
        };
        for (const BufferContract& buffer : node->inputs) {
            const std::string reason = check(buffer);
            if (!reason.empty())
                return reason;
        }
        for (const BufferContract& buffer : node->outputs) {
            const std::string reason = check(buffer);
            if (!reason.empty())
                return reason;
        }
    }

    const auto require_channels = [&by_kind, &layout](
        const std::string& kind,
        const std::vector<std::string>& input_prefixes,
        const std::vector<std::string>& output_prefixes) -> std::string {
        const OperatorNode& node = *by_kind.at(kind);
        const auto require = [&kind, &layout](
            const std::vector<BufferContract>& buffers,
            const std::string& prefix) -> std::string {
            for (const std::string& channel : layout.channel_names) {
                const std::string expected = prefix + channel;
                const std::size_t count = static_cast<std::size_t>(
                    std::count_if(buffers.begin(), buffers.end(),
                        [&expected](const BufferContract& buffer) {
                            return buffer.name == expected;
                        }));
                if (count != 1)
                    return "'" + kind + "' must declare exactly one '"
                        + expected + "' buffer";
            }
            const std::size_t prefixed = static_cast<std::size_t>(
                std::count_if(buffers.begin(), buffers.end(),
                    [&prefix](const BufferContract& buffer) {
                        return buffer.name.starts_with(prefix);
                    }));
            if (prefixed != layout.channel_names.size())
                return "'" + kind + "' must declare exactly one '" + prefix
                    + "<channel>' buffer per channel";
            return {};
        };
        for (const std::string& prefix : input_prefixes) {
            const std::string reason = require(node.inputs, prefix);
            if (!reason.empty())
                return reason;
        }
        for (const std::string& prefix : output_prefixes) {
            const std::string reason = require(node.outputs, prefix);
            if (!reason.empty())
                return reason;
        }
        return {};
    };
    for (const auto& [kind, inputs, outputs] :
        std::vector<std::tuple<std::string, std::vector<std::string>,
            std::vector<std::string>>>{
            {"kspace_coulomb_features", {"multipoles_"},
                {"potential_features_"}},
            {"self_interaction_linear", {"multipoles_"},
                {"feature_self_terms_"}},
            {"external_field_projection", {},
                {"external_field_features_"}},
            {"analytic_correction_features", {"charge_density_"},
                {"correction_fields_"}},
        }) {
        const std::string reason = require_channels(kind, inputs, outputs);
        if (!reason.empty())
            return reason;
    }

    // The one-shared-cache rule at the buffer level: the features and energy
    // nodes consume the cache's tables under their production names with
    // identical contracts.
    const OperatorNode& cache = *by_kind["kspace_geometry_cache"];
    std::map<std::string, const BufferContract*> produced;
    for (const BufferContract& output : cache.outputs)
        produced[output.name] = &output;
    for (const char* kind : {"kspace_coulomb_features", "kspace_coulomb_energy"}) {
        const OperatorNode& consumer = *by_kind[kind];
        for (const BufferContract& input : consumer.inputs) {
            const auto it = produced.find(input.name);
            if (it == produced.end())
                continue;
            const BufferContract& shared = *it->second;
            if (shared.dtype != input.dtype || shared.shape != input.shape
                || shared.layout.order != input.layout.order)
                return std::string(kind) + " consumes '" + input.name
                    + "' with a contract the cache does not produce";
        }
    }
    return {};
}

// ---------------------------------------------------------------------------
// Operator nodes
// ---------------------------------------------------------------------------

namespace detail {

inline std::string spherical_harmonics_irreps(const int max_l)
{
    std::string irreps;
    for (int l = 0; l <= max_l; ++l) {
        if (!irreps.empty())
            irreps += "+";
        irreps += "1x" + std::to_string(l) + (l % 2 == 0 ? "e" : "o");
    }
    return irreps;
}

// The e3nn-sorted feature spelling: one multiplicity block per l, ascending.
inline std::string multiplicity_irreps(const int max_l, const std::size_t count)
{
    std::string irreps;
    for (int l = 0; l <= max_l; ++l) {
        if (!irreps.empty())
            irreps += "+";
        irreps += std::to_string(count) + "x" + std::to_string(l)
            + (l % 2 == 0 ? "e" : "o");
    }
    return irreps;
}

inline BufferContract buffer(std::string name, std::string dtype,
    std::vector<std::size_t> shape, std::string order, std::string irreps)
{
    BufferContract contract;
    contract.name = std::move(name);
    contract.dtype = std::move(dtype);
    contract.shape = std::move(shape);
    contract.layout = LayoutContract{std::move(order), std::move(irreps), 1};
    return contract;
}

} // namespace detail

// Compile the kind group's operator nodes from contract data.  The nodes are
// family-scoped pure data: the adopting family folds them into its compiled
// program and supplies the node-id namespace and channel names. Buffer names
// are the linking vocabulary --
// the features and energy nodes consume the cache outputs under their
// production names, which keeps the one-shared-cache rule assertable at the
// binding level.  Invalid contract data has no operator spelling: the empty
// node list is the refusal, and callers validate first (the probe does).
//
// Layout orders: `node_major` for per-node blocks, `k_major` for the
// [N_k, ...] tables and per-k factors (one row per half-sphere k-point,
// flattened over graphs), `graph_major` for per-graph rows.  The k- and
// graph-major orders are the long-range group's addition to the layout
// vocabulary; the shared tensor kernels' node/edge orders do not address
// these extents.
inline std::vector<OperatorNode> long_range_primitive_operator_nodes(
    const LongRangeContract& contract,
    const LongRangeChannelLayout& layout)
{
    if (!validate_long_range_channel_layout(contract, layout).empty())
        return {};
    const std::size_t density_width = static_cast<std::size_t>(
        (contract.density_max_l + 1) * (contract.density_max_l + 1));
    const std::string density_irreps =
        detail::spherical_harmonics_irreps(contract.density_max_l);
    const std::string field_irreps = detail::multiplicity_irreps(
        contract.feature_max_l, contract.feature_smearing_widths.size());
    const std::string& tables = contract.table_dtype;
    const auto node_id = [&layout](const std::string_view kind) {
        return layout.node_id_prefix + "." + std::string(kind);
    };

    std::vector<OperatorNode> nodes;

    // The geometry cache: one per evaluation, shared by features, energy,
    // both spin channels, and every recursion step.  Its extents define the
    // capacity currency (N_k and table_bytes).
    {
        OperatorNode node;
        node.kind = OperatorKind::kspace_geometry_cache;
        node.id = node_id("kspace_geometry_cache");
        node.capability_key = "kspace_geometry_cache";
        node.inputs.push_back(detail::buffer(
            "positions", "float64", {0, 3}, "node_major", "1x1o"));
        node.inputs.push_back(detail::buffer(
            "cell", "float64", {3, 3}, "node_major", ""));
        node.inputs.push_back(detail::buffer(
            "node_graph_index", "int32", {0}, "node_major", "1x0e"));
        node.outputs.push_back(detail::buffer(
            "k_vectors", "float64", {0, 3}, "k_major", "1x1o"));
        node.outputs.push_back(detail::buffer(
            "k_norm2", "float64", {0}, "k_major", "1x0e"));
        node.outputs.push_back(detail::buffer(
            "k_graph_index", "int32", {0}, "k_major", "1x0e"));
        // The k=0 handling stays an explicit mask (exact branchy logic), so
        // the factor-2 half weight at the origin is data rather than a
        // floating scale factor.
        node.outputs.push_back(detail::buffer(
            "k0_mask", "float64", {0}, "k_major", "1x0e"));
        node.outputs.push_back(detail::buffer(
            "volume_per_k", "float64", {0}, "k_major", "1x0e"));
        node.outputs.push_back(detail::buffer(
            "k_factor_coulomb", tables, {0}, "k_major", "1x0e"));
        node.outputs.push_back(detail::buffer(
            "k_factor_proj", tables, {0}, "k_major", "1x0e"));
        node.outputs.push_back(detail::buffer(
            "cos_table", tables, {0, 0}, "k_major", ""));
        node.outputs.push_back(detail::buffer(
            "sin_table", tables, {0, 0}, "k_major", ""));
        node.outputs.push_back(detail::buffer("density_fourier_factors",
            tables, {0, density_width, 2}, "k_major", density_irreps));
        node.outputs.push_back(detail::buffer("feature_fourier_factors",
            tables, {0, contract.field_dimension, 2}, "k_major",
            field_irreps));
        node.parameters.emplace_back("kspace_cutoff", contract.kspace_cutoff);
        node.parameters.emplace_back(
            "density_max_l", static_cast<double>(contract.density_max_l));
        node.parameters.emplace_back(
            "feature_max_l", static_cast<double>(contract.feature_max_l));
        node.parameters.emplace_back(
            "density_smearing_width", contract.density_smearing_width);
        for (std::size_t index = 0;
            index < contract.feature_smearing_widths.size(); ++index) {
            node.parameters.emplace_back(
                "feature_smearing_width_" + std::to_string(index),
                contract.feature_smearing_widths[index]);
        }
        nodes.push_back(std::move(node));
    }

    // The electrostatic feature path: assemble, Coulomb-convolve, and project to
    // GTO features, per spin channel under one node.  The half-sphere
    // factor-2 weights live here (and in the energy kernel), not in the
    // cache.
    {
        OperatorNode node;
        node.kind = OperatorKind::kspace_coulomb_features;
        node.id = node_id("kspace_coulomb_features");
        node.capability_key = "kspace_coulomb_features";
        for (const char* name : {"cos_table", "sin_table"}) {
            node.inputs.push_back(
                detail::buffer(name, tables, {0, 0}, "k_major", ""));
        }
        node.inputs.push_back(detail::buffer(
            "k_factor_coulomb", tables, {0}, "k_major", "1x0e"));
        node.inputs.push_back(detail::buffer(
            "k_factor_proj", tables, {0}, "k_major", "1x0e"));
        node.inputs.push_back(detail::buffer(
            "volume_per_k", "float64", {0}, "k_major", "1x0e"));
        node.inputs.push_back(detail::buffer(
            "k0_mask", "float64", {0}, "k_major", "1x0e"));
        node.inputs.push_back(detail::buffer("density_fourier_factors",
            tables, {0, density_width, 2}, "k_major", density_irreps));
        node.inputs.push_back(detail::buffer("feature_fourier_factors",
            tables, {0, contract.field_dimension, 2}, "k_major",
            field_irreps));
        // The model's per-channel feature norms are contract constants the
        // family supplies; the normed output keeps the external-field
        // contribution independently combinable (the norm divide is linear).
        node.inputs.push_back(detail::buffer("field_feature_norms", "float64",
            {contract.field_dimension}, "node_major", field_irreps));
        for (const std::string& channel : layout.channel_names) {
            node.inputs.push_back(detail::buffer("multipoles_" + channel,
                "float64", {0, density_width}, "node_major", density_irreps));
            node.outputs.push_back(detail::buffer(
                "potential_features_" + channel, "float64",
                {0, contract.field_dimension}, "node_major", field_irreps));
        }
        node.parameters.emplace_back("half_sphere_weight", half_sphere_weight);
        nodes.push_back(std::move(node));
    }

    // The electrostatic energy operator: the periodic implementation evaluates
    // V/(2pi)^6 * sum over the half sphere of the
    // real part, per graph, minus the self-interaction overlaps.  The oracle
    // recomputes its own cos/sin tables here; the port consumes the shared
    // cache (the inputs are identical), which this node's input contracts
    // state.
    {
        OperatorNode node;
        node.kind = OperatorKind::kspace_coulomb_energy;
        node.id = node_id("kspace_coulomb_energy");
        node.capability_key = "kspace_coulomb_energy";
        for (const char* name : {"cos_table", "sin_table"}) {
            node.inputs.push_back(
                detail::buffer(name, tables, {0, 0}, "k_major", ""));
        }
        node.inputs.push_back(detail::buffer(
            "k_factor_coulomb", tables, {0}, "k_major", "1x0e"));
        node.inputs.push_back(detail::buffer(
            "volume_per_k", "float64", {0}, "k_major", "1x0e"));
        node.inputs.push_back(detail::buffer(
            "k0_mask", "float64", {0}, "k_major", "1x0e"));
        node.inputs.push_back(detail::buffer(
            "k_graph_index", "int32", {0}, "k_major", "1x0e"));
        node.inputs.push_back(detail::buffer("density_fourier_factors",
            tables, {0, density_width, 2}, "k_major", density_irreps));
        node.inputs.push_back(detail::buffer("charge_density", "float64",
            {0, density_width}, "node_major", density_irreps));
        node.inputs.push_back(detail::buffer(
            "cell", "float64", {3, 3}, "node_major", ""));
        node.inputs.push_back(detail::buffer("energy_self_fields", "float64",
            {0, density_width}, "node_major", density_irreps));
        node.outputs.push_back(detail::buffer(
            "energy_kspace", "float64", {0}, "graph_major", "1x0e"));
        node.parameters.emplace_back("half_sphere_weight", half_sphere_weight);
        nodes.push_back(std::move(node));
    }

    // The self-interaction overlaps: constant linear ops derived in closed
    // form from the basis parameters (the sigma set and widths) exactly as
    // the oracle's gto_utils derives them -- fixed init-time quadrature, so
    // the forward is a gather and broadcast scale with no runtime
    // quadrature.  The contract carries no coefficients; the checkpoint
    // facts pin their digests only for later cross-checks.  Feature terms
    // are per spin channel; the energy fields are gathered from the
    // spin-summed density.
    {
        OperatorNode node;
        node.kind = OperatorKind::self_interaction_linear;
        node.id = node_id("self_interaction_linear");
        node.capability_key = "self_interaction_linear";
        for (const std::string& channel : layout.channel_names) {
            node.inputs.push_back(detail::buffer("multipoles_" + channel,
                "float64", {0, density_width}, "node_major", density_irreps));
            node.outputs.push_back(detail::buffer(
                "feature_self_terms_" + channel, "float64",
                {0, contract.field_dimension}, "node_major", field_irreps));
        }
        node.outputs.push_back(detail::buffer("energy_self_fields", "float64",
            {0, density_width}, "node_major", density_irreps));
        nodes.push_back(std::move(node));
    }

    // The external-field projection: the gauge term E.(r - barycenter)
    // through the constant [features_dim, 4] matrix, scaled by one half per
    // spin channel and divided by the feature norms.  The barycenter is a
    // per-graph scatter inside the kind (fp64 accumulator).
    {
        OperatorNode node;
        node.kind = OperatorKind::external_field_projection;
        node.id = node_id("external_field_projection");
        node.capability_key = "external_field_projection";
        node.inputs.push_back(detail::buffer(
            "external_potential", "float64", {0, 4}, "graph_major", ""));
        node.inputs.push_back(detail::buffer(
            "positions", "float64", {0, 3}, "node_major", "1x1o"));
        node.inputs.push_back(detail::buffer(
            "node_graph_index", "int32", {0}, "node_major", "1x0e"));
        node.inputs.push_back(detail::buffer("field_feature_norms", "float64",
            {contract.field_dimension}, "node_major", field_irreps));
        for (const std::string& channel : layout.channel_names) {
            node.outputs.push_back(detail::buffer(
                "external_field_features_" + channel, "float64",
                {0, contract.field_dimension}, "node_major", field_irreps));
        }
        node.parameters.emplace_back(
            "external_field_scale", external_field_scale);
        node.parameters.emplace_back(
            "field_norm_factor", contract.field_norm_factor);
        nodes.push_back(std::move(node));
    }

    // The dipole readout: total charge and dipole of the equilibrated
    // (spin-summed, mul_ir-permuted) density.  The external-field energy
    // -E.mu is a graph-level contraction of the dipole with the external
    // potential the family's orchestration owns.
    {
        OperatorNode node;
        node.kind = OperatorKind::dipole_readout;
        node.id = node_id("dipole_readout");
        node.capability_key = "dipole_readout";
        node.inputs.push_back(detail::buffer("charge_density", "float64",
            {0, density_width}, "node_major", density_irreps));
        node.inputs.push_back(detail::buffer(
            "positions", "float64", {0, 3}, "node_major", "1x1o"));
        node.inputs.push_back(detail::buffer(
            "node_graph_index", "int32", {0}, "node_major", "1x0e"));
        node.outputs.push_back(detail::buffer(
            "total_charge", "float64", {0}, "graph_major", "1x0e"));
        node.outputs.push_back(detail::buffer(
            "total_dipole", "float64", {0, 3}, "graph_major", "1x1o"));
        nodes.push_back(std::move(node));
    }

    // The analytic energy correction: the Makov-Payne/Dabo
    // monopole-dipole-quadrupole box term for molecule graphs, the mu_z^2
    // slab dipole term for slab graphs, nothing for bulk and any other pbc
    // pattern.  The graph's own pbc row selects the mode at runtime.
    {
        OperatorNode node;
        node.kind = OperatorKind::analytic_correction_energy;
        node.id = node_id("analytic_correction_energy");
        node.capability_key = "analytic_correction_energy";
        node.inputs.push_back(detail::buffer("charge_density", "float64",
            {0, density_width}, "node_major", density_irreps));
        node.inputs.push_back(detail::buffer(
            "positions", "float64", {0, 3}, "node_major", "1x1o"));
        node.inputs.push_back(detail::buffer(
            "cell", "float64", {3, 3}, "node_major", ""));
        node.inputs.push_back(detail::buffer(
            "node_graph_index", "int32", {0}, "node_major", "1x0e"));
        node.inputs.push_back(detail::buffer(
            "graph_pbc", "int32", {0, 3}, "graph_major", ""));
        node.outputs.push_back(detail::buffer(
            "correction_energy", "float64", {0}, "graph_major", "1x0e"));
        node.parameters.emplace_back("cubic_madelung", cubic_madelung);
        nodes.push_back(std::move(node));
    }

    // The analytic feature correction: point-charge corrective potential
    // node fields for molecule graphs, z-dipole corrective fields for slab
    // graphs, projected onto GTO features by the constant matrix derived
    // from the basis parameters (the sigma set and widths), exactly as the
    // oracle's gto_utils derives it -- the contract carries no coefficients.
    // Per spin channel; mode selected per graph by pbc.
    {
        OperatorNode node;
        node.kind = OperatorKind::analytic_correction_features;
        node.id = node_id("analytic_correction_features");
        node.capability_key = "analytic_correction_features";
        for (const std::string& channel : layout.channel_names) {
            node.inputs.push_back(detail::buffer("charge_density_" + channel,
                "float64", {0, density_width}, "node_major", density_irreps));
        }
        node.inputs.push_back(detail::buffer(
            "positions", "float64", {0, 3}, "node_major", "1x1o"));
        node.inputs.push_back(detail::buffer(
            "cell", "float64", {3, 3}, "node_major", ""));
        node.inputs.push_back(detail::buffer(
            "node_graph_index", "int32", {0}, "node_major", "1x0e"));
        node.inputs.push_back(detail::buffer(
            "graph_pbc", "int32", {0, 3}, "graph_major", ""));
        for (const std::string& channel : layout.channel_names) {
            node.outputs.push_back(detail::buffer(
                "correction_fields_" + channel, "float64",
                {0, contract.field_dimension}, "node_major", field_irreps));
        }
        nodes.push_back(std::move(node));
    }

    return nodes;
}

// A family must name its channels and node namespace before primitive nodes
// have a valid spelling.  The no-layout overload is a deliberate refusal,
// retained so callers cannot accidentally obtain a Polar-shaped program from
// a neutral contract.
inline std::vector<OperatorNode> long_range_operator_nodes(
    const LongRangeContract&)
{
    return {};
}

// ---------------------------------------------------------------------------
// Registry integration
// ---------------------------------------------------------------------------

// Declare the kind group's implementations into a family-backend registry.
// One portable (T1) descriptor per kind: the generic host-first
// implementation is the production path for fixed-form physics kernels with
// no learned weights, so no generated tier is declared and the host
// admission invariant (portable coverage per node) is the group's only
// admission rule.  Descriptors are qualified exactly on the target they are
// declared for, so a plan on another target reports the nodes unbound with
// the target named rather than silently degrading.  `compute` is the
// family-backend's compute precision; the accumulator is the contract's
// fp64 policy.
inline void declare_long_range_implementations(
    ImplementationRegistry& registry,
    const std::string& target,
    const LongRangeContract& contract,
    const std::string& compute = "float64",
    const std::string& identity_override = {})
{
    const std::string fingerprint=identity_override.empty()
        ?long_range_contract_fingerprint(contract):identity_override;
    for (const LongRangeKindSpec& spec : long_range_kind_specs()) {
        ImplementationDescriptor descriptor;
        descriptor.id = "portable_long_range_" + std::string(spec.name);
        descriptor.kind = spec.kind;
        descriptor.capability_key = std::string(spec.name);
        descriptor.tier = ImplementationTier::portable;
        descriptor.target = target;
        descriptor.precision =
            PrecisionPolicy{compute, contract.accumulator_dtype};
        descriptor.layout = LayoutCompatibility{"node_major", "node_major"};
        descriptor.capabilities.target = target;
        descriptor.capabilities.precision = compute;
        descriptor.capabilities.portable_ready = true;
        // Stress is served exactly where the experimental reciprocal-space
        // cell adjoint is declared (the finite-strain oracle is its gate;
        // no analytic serial oracle ships for it).  Field coupling is the
        // external-field projection's declaration.
        descriptor.capabilities.stress = spec.capabilities.cell_adjoint;
        descriptor.capabilities.field = spec.capabilities.serves_external_field;
        descriptor.capabilities.parameter_gradients = false;
        descriptor.identity_fingerprint = fingerprint;
        descriptor.qualified =
            [] (const ImplementationContext&) { return true; };
        registry.declare(std::move(descriptor));
    }
}

// ---------------------------------------------------------------------------
// Capacity hooks
// ---------------------------------------------------------------------------

// The capacity fields, in the manifest's long_range vocabulary
// (benchmarks/foundation_model_support_matrix.py,
// LONG_RANGE_CAPACITY_VERSION 1): N_k is the half-sphere k-point count
// flattened over graphs, table_bytes the dense cos/sin table bytes that
// dominate the layer's memory and are rebuilt every step, the correction
// mode is recorded per graph class, and the rebuild trigger states whether a
// cell change rebuilds the k-grid as well as the tables.
struct LongRangeCapacity {
    std::size_t n_k = 0;
    std::size_t table_bytes = 0;
    std::string correction_mode = "none";
    bool rebuild_on_cell_change = true;
};

inline std::size_t long_range_dtype_itemsize(const std::string& dtype)
{
    if (dtype == "float32")
        return 4;
    if (dtype == "float64")
        return 8;
    return 0;
}

// Validate the capacity inputs.  Returns the empty string when they admit a
// record; otherwise a human-readable rejection reason.  The rules are the
// Python mirror's (`PolarLongRangeContract.capacity_record` validates inline
// with the same three): a positive k-point count, a table dtype in the
// precision policy's vocabulary, and a correction mode in the manifest
// vocabulary.
inline std::string validate_long_range_capacity(
    const std::size_t n_k, const std::string& correction_mode,
    const std::string& table_dtype = "float64")
{
    if (n_k == 0)
        return "n_k must be at least one k-point";
    if (long_range_dtype_itemsize(table_dtype) == 0)
        return "table_dtype must be float32 or float64";
    for (const std::string_view token : correction_mode_vocabulary) {
        if (correction_mode == token)
            return {};
    }
    return "correction mode '" + correction_mode
        + "' is not in the correction-mode vocabulary "
          "(makov-payne-dabo, slab-dipole, none)";
}

inline LongRangeCapacity make_long_range_capacity(
    const std::size_t n_k,
    const std::size_t atoms,
    const std::string& correction_mode,
    const std::string& table_dtype,
    const bool rebuild_on_cell_change = true)
{
    // The constructor validates, exactly like the node builder refuses
    // invalid contract data: a capacity record is the same class of
    // declaration-time data, so an invalid record has no spelling.  The
    // refusal lives at construction (not in every admission path) so no
    // caller can emit an unvalidated record -- the asymmetry the Task-6
    // review flagged against the validating Python mirror.
    const std::string reason =
        validate_long_range_capacity(n_k, correction_mode, table_dtype);
    if (!reason.empty())
        throw std::invalid_argument("long-range capacity: " + reason);
    LongRangeCapacity capacity;
    capacity.n_k = n_k;
    capacity.table_bytes =
        2 * n_k * atoms * long_range_dtype_itemsize(table_dtype);
    capacity.correction_mode = correction_mode;
    capacity.rebuild_on_cell_change = rebuild_on_cell_change;
    return capacity;
}

} // namespace symmetrix::execution::long_range
