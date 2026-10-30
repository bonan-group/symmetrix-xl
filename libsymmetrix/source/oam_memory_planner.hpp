#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace symmetrix::oam {

inline constexpr std::size_t oam_arena_alignment = 64;
inline constexpr std::size_t oam_arena_canary_bytes = 64;

enum class OAMRetentionPolicy { speed, capacity };

struct OAMLayerMemoryDimensions {
    std::size_t input_width = 0;
    std::size_t prelinear_width = 0;
    std::size_t edge_width = 0;
    std::size_t message_width = 0;
    std::size_t pre_gate_width = 0;
    std::size_t output_width = 0;
    std::size_t attribute_width = 0;
    std::size_t self_forward_scratch_width = 0;
    std::size_t self_reverse_scratch_width = 0;
};

struct OAMMemoryDimensions {
    std::size_t num_nodes = 0;
    std::size_t num_edges = 0;
    std::size_t scalar_bytes = 0;
    std::size_t harmonic_count = 0;
    std::size_t readout_width = 0;
    std::size_t readout_output_width = 0;
    std::size_t packed_input_values = 0;
    std::size_t packed_output_values = 0;
    std::size_t bilinear_forward_values = 0;
    std::size_t bilinear_reverse_values = 0;
    std::size_t reverse_partial_values = 0;
    std::size_t edge_chunk_values = 0;
    std::size_t source_harmonic_partial_values = 0;
    std::size_t source_distance_partial_values = 0;
    std::vector<OAMLayerMemoryDimensions> layers;
};

struct OAMMemorySlot {
    std::string name;
    std::size_t offset = 0;
    std::size_t bytes = 0;
    std::size_t live_begin = 0;
    std::size_t live_end = 0;
};

struct OAMMemoryLayout {
    OAMRetentionPolicy policy = OAMRetentionPolicy::capacity;
    std::size_t alignment = oam_arena_alignment;
    std::size_t active_bytes = 0;
    std::size_t peak_transient_bytes = 0;
    std::vector<OAMMemorySlot> slots;

    const OAMMemorySlot& slot(const std::string_view name) const
    {
        const auto found = std::find_if(slots.begin(), slots.end(),
            [name](const OAMMemorySlot& candidate) {
                return candidate.name == name;
            });
        if (found == slots.end())
            throw std::out_of_range("unknown OAM memory slot");
        return *found;
    }
};

inline std::size_t checked_oam_add(
    const std::size_t left, const std::size_t right)
{
    if (right > std::numeric_limits<std::size_t>::max() - left)
        throw std::overflow_error("OAM memory layout addition overflow");
    return left + right;
}

inline std::size_t checked_oam_multiply(
    const std::size_t left, const std::size_t right)
{
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
        throw std::overflow_error("OAM memory layout multiplication overflow");
    return left * right;
}

inline std::size_t align_oam_bytes(const std::size_t bytes)
{
    const std::size_t remainder = bytes % oam_arena_alignment;
    return remainder == 0 ? bytes
                          : checked_oam_add(bytes, oam_arena_alignment - remainder);
}

inline void validate_oam_alias(
    const OAMMemorySlot& left, const OAMMemorySlot& right)
{
    const bool byte_overlap = left.offset < checked_oam_add(right.offset, right.bytes)
        && right.offset < checked_oam_add(left.offset, left.bytes);
    const bool lifetime_overlap = left.live_begin < right.live_end
        && right.live_begin < left.live_end;
    if (byte_overlap && lifetime_overlap)
        throw std::logic_error("OAM arena aliases have overlapping live ranges");
}

namespace detail {

inline std::size_t matrix_bytes(const std::size_t rows,
    const std::size_t columns, const std::size_t scalar_bytes)
{
    return checked_oam_multiply(
        checked_oam_multiply(rows, columns), scalar_bytes);
}

inline void append_slot(OAMMemoryLayout& layout, std::string name,
    const std::size_t bytes, const std::size_t live_begin,
    const std::size_t live_end)
{
    const std::size_t guard_begin = align_oam_bytes(layout.active_bytes);
    const std::size_t offset = checked_oam_add(
        guard_begin, oam_arena_canary_bytes);
    layout.slots.push_back(
        {std::move(name), offset, bytes, live_begin, live_end});
    layout.active_bytes = checked_oam_add(
        checked_oam_add(offset, bytes), oam_arena_canary_bytes);
}

inline std::size_t maximum_layer_matrix_bytes(
    const OAMMemoryDimensions& dimensions, const bool reverse)
{
    std::size_t result = 0;
    for (const auto& layer : dimensions.layers) {
        std::size_t width = 0;
        if (reverse)
            width = checked_oam_add(
                checked_oam_add(layer.pre_gate_width, layer.message_width),
                checked_oam_add(layer.prelinear_width,
                    layer.self_reverse_scratch_width));
        else
            width = checked_oam_add(
                layer.message_width, layer.self_forward_scratch_width);
        result = std::max(result, matrix_bytes(
            dimensions.num_nodes, width, dimensions.scalar_bytes));
    }
    return result;
}

}  // namespace detail

inline OAMMemoryLayout plan_oam_memory(const OAMMemoryDimensions& dimensions,
    const OAMRetentionPolicy policy)
{
    if (dimensions.scalar_bytes == 0)
        throw std::invalid_argument("OAM scalar size must be nonzero");

    OAMMemoryLayout layout;
    layout.policy = policy;
    const std::size_t scalar = dimensions.scalar_bytes;
    const std::size_t nodes = dimensions.num_nodes;
    const std::size_t edges = dimensions.num_edges;

    if (!dimensions.layers.empty())
        detail::append_slot(layout, "initial_features",
            detail::matrix_bytes(
                nodes, dimensions.layers.front().input_width, scalar),
            0, 7);
    for (std::size_t index = 0; index < dimensions.layers.size(); ++index) {
        const auto& layer = dimensions.layers[index];
        const std::string prefix = "layer_" + std::to_string(index);
        detail::append_slot(layout, prefix + "_pre_gate",
            detail::matrix_bytes(nodes, layer.pre_gate_width, scalar), 2, 7);
        detail::append_slot(layout, prefix + "_output",
            detail::matrix_bytes(nodes, layer.output_width, scalar), 3, 7);
        if (policy == OAMRetentionPolicy::speed)
            detail::append_slot(layout, prefix + "_prelinear",
                detail::matrix_bytes(nodes, layer.prelinear_width, scalar), 1, 7);
    }

    std::size_t forward_bytes = detail::maximum_layer_matrix_bytes(dimensions, false);
    std::size_t reverse_bytes = detail::maximum_layer_matrix_bytes(dimensions, true);
    for (const auto& layer : dimensions.layers) {
        if (policy == OAMRetentionPolicy::capacity) {
            forward_bytes = std::max(forward_bytes,
                detail::matrix_bytes(nodes,
                    checked_oam_add(layer.prelinear_width,
                        checked_oam_add(layer.message_width,
                            layer.self_forward_scratch_width)),
                    scalar));
            reverse_bytes = std::max(reverse_bytes,
                detail::matrix_bytes(nodes,
                    checked_oam_add(layer.prelinear_width,
                        checked_oam_add(layer.pre_gate_width,
                            checked_oam_add(layer.message_width,
                                checked_oam_add(layer.prelinear_width,
                                    layer.self_reverse_scratch_width)))),
                    scalar));
        }
    }
    forward_bytes = std::max(forward_bytes,
        detail::matrix_bytes(
            nodes, dimensions.readout_output_width, scalar));
    reverse_bytes = std::max(reverse_bytes,
        detail::matrix_bytes(
            nodes, dimensions.readout_output_width, scalar));
    detail::append_slot(layout, "forward_scratch", forward_bytes, 1, 4);
    if (policy == OAMRetentionPolicy::capacity) {
        const std::size_t shared_bytes = std::max(forward_bytes, reverse_bytes);
        auto& forward = layout.slots.back();
        forward.bytes = shared_bytes;
        layout.active_bytes = checked_oam_add(
            checked_oam_add(forward.offset, shared_bytes),
            oam_arena_canary_bytes);
        const auto forward_slot = forward;
        layout.slots.push_back(
            {"reverse_scratch", forward_slot.offset, shared_bytes, 4, 7});
        validate_oam_alias(forward_slot, layout.slots.back());
    } else {
        detail::append_slot(layout, "reverse_scratch", reverse_bytes, 2, 7);
    }

    std::size_t maximum_adjoint_width = dimensions.readout_width;
    for (const auto& layer : dimensions.layers) {
        maximum_adjoint_width = std::max(maximum_adjoint_width,
            std::max(layer.input_width, layer.output_width));
    }
    detail::append_slot(layout, "node_adjoint_a",
        detail::matrix_bytes(nodes, maximum_adjoint_width, scalar), 4, 8);
    detail::append_slot(layout, "node_adjoint_b",
        detail::matrix_bytes(nodes, maximum_adjoint_width, scalar), 4, 8);
    detail::append_slot(layout, "reverse_partials",
        checked_oam_multiply(dimensions.reverse_partial_values, scalar), 5, 7);
    detail::append_slot(layout, "edge_chunk_scratch",
        checked_oam_multiply(dimensions.edge_chunk_values, scalar), 1, 7);
    detail::append_slot(layout, "source_harmonic_partials",
        checked_oam_multiply(
            dimensions.source_harmonic_partial_values, scalar), 5, 7);
    detail::append_slot(layout, "source_distance_partials",
        checked_oam_multiply(
            dimensions.source_distance_partial_values, sizeof(double)), 5, 7);
    detail::append_slot(layout, "packed_linear_scratch",
        checked_oam_multiply(checked_oam_add(dimensions.packed_input_values,
            dimensions.packed_output_values), scalar), 1, 7);
    // Self-connection bilinear forward and reverse contractions never overlap.
    // Record both lifetimes over one allocation so the four unmanaged views
    // are reconstructed from the same checked maxima that produced the plan.
    const std::size_t bilinear_bytes = checked_oam_multiply(
        std::max(dimensions.bilinear_forward_values,
            dimensions.bilinear_reverse_values), scalar);
    detail::append_slot(layout, "bilinear_forward_scratch",
        bilinear_bytes, 1, 4);
    const auto bilinear_forward = layout.slots.back();
    layout.slots.push_back({"bilinear_reverse_scratch",
        bilinear_forward.offset, bilinear_bytes, 4, 7});
    validate_oam_alias(bilinear_forward, layout.slots.back());

    std::size_t reduction_values = checked_oam_add(edges, nodes);
    reduction_values = checked_oam_add(reduction_values,
        checked_oam_multiply(nodes, std::size_t(3)));
    reduction_values = checked_oam_add(reduction_values, 6);
    detail::append_slot(layout, "final_reductions",
        checked_oam_multiply(reduction_values, sizeof(double)), 7, 9);

    layout.active_bytes = align_oam_bytes(layout.active_bytes);
    layout.peak_transient_bytes = layout.active_bytes;
    return layout;
}

struct OAMTransientArenaState {
    std::size_t active_bytes = 0;
    std::size_t retained_capacity_bytes = 0;
    std::size_t peak_transient_bytes = 0;
    std::size_t replacements = 0;
    std::string high_water_reason;

    bool admit(const OAMMemoryLayout& layout, std::string reason)
    {
        active_bytes = layout.active_bytes;
        peak_transient_bytes = std::max(
            peak_transient_bytes, layout.peak_transient_bytes);
        if (layout.active_bytes <= retained_capacity_bytes) return false;
        retained_capacity_bytes = layout.active_bytes;
        ++replacements;
        high_water_reason = std::move(reason);
        return true;
    }
};

}  // namespace symmetrix::oam
