#pragma once

#include <cstdint>
#include <string_view>

namespace symmetrix::execution {

inline constexpr std::uint32_t r0_density_prepare_capability = 1u << 0;
inline constexpr std::uint32_t r0_forward_capability = 1u << 1;
inline constexpr std::uint32_t r0_reverse_prepare_capability = 1u << 2;
inline constexpr std::uint32_t r0_coordinate_reverse_capability = 1u << 3;
inline constexpr std::uint32_t r0_compact_geometry_capability = 1u << 4;
inline constexpr std::uint32_t r0_precomputed_scale_capability = 1u << 5;
inline constexpr std::uint32_t r0_receiver_batch_capability = 1u << 6;
inline constexpr std::uint32_t r0_required_capabilities =
    r0_density_prepare_capability|r0_forward_capability
    |r0_reverse_prepare_capability|r0_coordinate_reverse_capability
    |r0_compact_geometry_capability|r0_precomputed_scale_capability
    |r0_receiver_batch_capability;
inline constexpr std::uint32_t r0_compatibility_version = 2;

// This is the semantic descriptor shared by the host-plugin and accelerator
// loaders.  Their physical descriptors intentionally remain separate.
struct R0LogicalDescriptor {
    std::string_view artifact_id;
    std::string_view structure_fingerprint;
    std::string_view target;
    std::string_view schedule;
    std::uint32_t scalar_kind = 0;
    std::uint32_t scalar_size = 0;
    std::uint32_t capabilities = 0;
    std::int32_t channels = 0;
    std::int32_t l_max = 0;
    std::int32_t harmonic_count = 0;
    std::int32_t edge_weight_mode = 0;
    std::int32_t schedule_revision = 0;
    std::uint32_t compatibility_version = 0;
};

inline std::string_view r0_descriptor_mismatch(
    const R0LogicalDescriptor& actual,
    const R0LogicalDescriptor& expected) noexcept
{
    if (actual.artifact_id != expected.artifact_id)
        return "artifact_id";
    if (actual.structure_fingerprint != expected.structure_fingerprint)
        return "structure_fingerprint";
    if (actual.target != expected.target)
        return "target";
    if (actual.schedule != expected.schedule)
        return "schedule";
    if (actual.scalar_kind != expected.scalar_kind)
        return "scalar_kind";
    if (actual.scalar_size != expected.scalar_size)
        return "scalar_size";
    if ((actual.capabilities&expected.capabilities) != expected.capabilities)
        return "capabilities";
    if (actual.channels != expected.channels)
        return "channels";
    if (actual.l_max != expected.l_max)
        return "l_max";
    if (actual.harmonic_count != expected.harmonic_count)
        return "harmonic_count";
    if (actual.edge_weight_mode != expected.edge_weight_mode)
        return "edge_weight_mode";
    if (actual.schedule_revision != expected.schedule_revision)
        return "schedule_revision";
    if (actual.compatibility_version != expected.compatibility_version)
        return "compatibility_version";
    return {};
}

} // namespace symmetrix::execution
