#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <span>
#include <string_view>
#include <tuple>

namespace symmetrix::execution {

struct LaunchProfile {
    std::string_view implementation_id;
    std::string_view profile_id;
    std::string_view backend;
    std::string_view precision;
    std::string_view launch_geometry;
    int required_warp_width;
    int source_vector_width;
    int edge_vector_width;
    int max_threads_per_block;
    int persistent_blocks_per_sm;
    std::string_view accumulator_policy;
    int minimum_compute_capability;
    int maximum_compute_capability;
    std::string_view architecture_family;
    std::array<int,8> calibration_candidates;
    int calibration_candidate_count;
    bool calibration_permitted;
};

inline constexpr char m0_module_id[] = "standard-m0-module-module-v1";
inline constexpr char m0_scalar_module_id[] = "standard-m0-lmax0-module-v1";
inline constexpr char r0_module_id[] = "standard-r0-module-module-v2";
inline constexpr char field_h1_reverse_id[] = "field-h1-reverse-wave32-v1";

inline constexpr std::array<LaunchProfile,23> launch_profiles = {{
    {m0_module_id, "standard-m0-module-module-v1-cuda-float32-generic", "cuda", "float32", "persistent_warp8x32_node_channel", 32, 0, 0, 256, 4, "fixed_order_fp32", 0, 0, "", std::array<int,8>{2, 4, 8, 0, 0, 0, 0, 0}, 3, true},
    {m0_module_id, "standard-m0-module-module-v1-cuda-float64-generic", "cuda", "float64", "persistent_warp8x32_node_channel", 32, 0, 0, 256, 4, "fixed_order_fp64", 0, 0, "", std::array<int,8>{2, 4, 8, 0, 0, 0, 0, 0}, 3, true},
    {m0_module_id, "standard-m0-module-module-v1-hip-float32-generic", "hip", "float32", "persistent_wave8x32_node_channel", 32, 0, 0, 256, 4, "fixed_order_fp32", 0, 0, "", std::array<int,8>{2, 4, 8, 0, 0, 0, 0, 0}, 3, true},
    {m0_module_id, "standard-m0-module-module-v1-hip-float64-generic", "hip", "float64", "persistent_wave8x32_node_channel", 32, 0, 0, 256, 4, "fixed_order_fp64", 0, 0, "", std::array<int,8>{2, 4, 8, 0, 0, 0, 0, 0}, 3, true},
    {m0_module_id, "standard-m0-module-module-v1-host-float32-generic", "host", "float32", "flat_node_channel_range", 0, 0, 0, 0, 0, "fixed_order_fp32", 0, 0, "", std::array<int,8>{0, 0, 0, 0, 0, 0, 0, 0}, 0, false},
    {m0_module_id, "standard-m0-module-module-v1-host-float64-generic", "host", "float64", "range_node_channel", 0, 0, 0, 0, 0, "fixed_order_fp64", 0, 0, "", std::array<int,8>{0, 0, 0, 0, 0, 0, 0, 0}, 0, false},
    {m0_scalar_module_id, "standard-m0-lmax0-module-v1-cuda-float32-generic", "cuda", "float32", "persistent_warp8x32_node_channel", 32, 0, 0, 256, 4, "fixed_order_fp32", 0, 0, "", std::array<int,8>{2, 4, 8, 0, 0, 0, 0, 0}, 3, true},
    {m0_scalar_module_id, "standard-m0-lmax0-module-v1-cuda-float64-generic", "cuda", "float64", "persistent_warp8x32_node_channel", 32, 0, 0, 256, 4, "fixed_order_fp64", 0, 0, "", std::array<int,8>{2, 4, 8, 0, 0, 0, 0, 0}, 3, true},
    {m0_scalar_module_id, "standard-m0-lmax0-module-v1-hip-float32-generic", "hip", "float32", "persistent_wave8x32_node_channel", 32, 0, 0, 256, 4, "fixed_order_fp32", 0, 0, "", std::array<int,8>{2, 4, 8, 0, 0, 0, 0, 0}, 3, true},
    {m0_scalar_module_id, "standard-m0-lmax0-module-v1-hip-float64-generic", "hip", "float64", "persistent_wave8x32_node_channel", 32, 0, 0, 256, 4, "fixed_order_fp64", 0, 0, "", std::array<int,8>{2, 4, 8, 0, 0, 0, 0, 0}, 3, true},
    {m0_scalar_module_id, "standard-m0-lmax0-module-v1-host-float32-generic", "host", "float32", "range_node_channel_tiled", 0, 0, 0, 0, 0, "fixed_order_fp32", 0, 0, "", std::array<int,8>{0, 0, 0, 0, 0, 0, 0, 0}, 0, false},
    {m0_scalar_module_id, "standard-m0-lmax0-module-v1-host-float64-generic", "host", "float64", "range_node_channel_tiled", 0, 0, 0, 0, 0, "fixed_order_fp64", 0, 0, "", std::array<int,8>{0, 0, 0, 0, 0, 0, 0, 0}, 0, false},
    {r0_module_id, "standard-r0-module-module-v2-cuda-float32-generic", "cuda", "float32", "runtime_channel_split_density_receiver_l32_reverse_receiver4x32_edge16x16_edge8x32", 32, 0, 16, 256, 4, "fixed_order_fp32_density_fp64", 0, 0, "", std::array<int,8>{2, 4, 8, 0, 0, 0, 0, 0}, 3, true},
    {r0_module_id, "standard-r0-module-module-v2-cuda-float64-generic", "cuda", "float64", "runtime_channel_split_density_receiver_l32_reverse_receiver4x32_edge16x16_edge8x32", 32, 0, 16, 256, 4, "fixed_order_fp64_density_fp64", 0, 0, "", std::array<int,8>{2, 4, 8, 0, 0, 0, 0, 0}, 3, true},
    {r0_module_id, "standard-r0-module-module-v2-hip-float32-generic", "hip", "float32", "runtime_channel_split_density_receiver_l32_reverse_receiver4x32_edge16x16_edge8x32", 32, 0, 16, 256, 4, "fixed_order_fp32_density_fp64", 0, 0, "", std::array<int,8>{2, 4, 8, 0, 0, 0, 0, 0}, 3, true},
    {r0_module_id, "standard-r0-module-module-v2-hip-float32-gfx1151", "hip", "float32", "runtime_channel_split_density_receiver_l32_reverse_receiver4x32_edge16x16_edge8x32", 32, 0, 16, 256, 8, "fixed_order_fp32_density_fp64", 0, 0, "gfx1151", std::array<int,8>{4, 8, 16, 0, 0, 0, 0, 0}, 3, true},
    {r0_module_id, "standard-r0-module-module-v2-hip-float64-generic", "hip", "float64", "runtime_channel_split_density_receiver_l32_reverse_receiver4x32_edge16x16_edge8x32", 32, 0, 16, 256, 4, "fixed_order_fp64_density_fp64", 0, 0, "", std::array<int,8>{2, 4, 8, 0, 0, 0, 0, 0}, 3, true},
    {r0_module_id, "standard-r0-module-module-v2-host-float32-generic", "host", "float32", "runtime_channel_flat_receiver_component_channel_edge_range", 0, 0, 0, 0, 0, "fixed_order_fp32_density_fp64", 0, 0, "", std::array<int,8>{0, 0, 0, 0, 0, 0, 0, 0}, 0, false},
    {r0_module_id, "standard-r0-module-module-v2-host-float64-generic", "host", "float64", "runtime_channel_flat_receiver_component_channel_edge_range", 0, 0, 0, 0, 0, "fixed_order_fp64_density_fp64", 0, 0, "", std::array<int,8>{0, 0, 0, 0, 0, 0, 0, 0}, 0, false},
    {field_h1_reverse_id, "field-h1-reverse-wave32-v1-cuda-float32-generic", "cuda", "float32", "persistent_warp8x32_grid_stride", 32, 0, 0, 256, 4, "wave32_fp64", 0, 0, "", std::array<int,8>{2, 4, 8, 0, 0, 0, 0, 0}, 3, true},
    {field_h1_reverse_id, "field-h1-reverse-wave32-v1-cuda-float64-generic", "cuda", "float64", "persistent_warp8x32_grid_stride", 32, 0, 0, 256, 4, "wave32_fp64", 0, 0, "", std::array<int,8>{2, 4, 8, 0, 0, 0, 0, 0}, 3, true},
    {field_h1_reverse_id, "field-h1-reverse-wave32-v1-hip-float32-generic", "hip", "float32", "persistent_wave8x32_grid_stride", 32, 0, 0, 256, 8, "wave32_fp64", 0, 0, "", std::array<int,8>{4, 8, 16, 0, 0, 0, 0, 0}, 3, true},
    {field_h1_reverse_id, "field-h1-reverse-wave32-v1-hip-float64-generic", "hip", "float64", "persistent_wave8x32_grid_stride", 32, 0, 0, 256, 8, "wave32_fp64", 0, 0, "", std::array<int,8>{4, 8, 16, 0, 0, 0, 0, 0}, 3, true},
}};

inline const LaunchProfile* find_launch_profile(
    const std::string_view profile_id)
{
    for (const auto& profile : launch_profiles)
        if (profile.profile_id == profile_id)
            return &profile;
    return nullptr;
}

inline bool is_calibration_candidate(
    const LaunchProfile& profile, const int blocks_per_compute_unit)
{
    for (int index=0; index<profile.calibration_candidate_count; ++index)
        if (profile.calibration_candidates[static_cast<std::size_t>(index)]
            == blocks_per_compute_unit)
            return true;
    return false;
}

inline const LaunchProfile* select_launch_profile(
    const std::string_view implementation_id,
    const std::string_view backend,
    const std::string_view precision,
    const int warp_width,
    const int compute_capability,
    const std::string_view architecture_family = {})
{
    const LaunchProfile* selected = nullptr;
    auto selected_rank = std::tuple{-1, -1, std::numeric_limits<int>::min()};
    for (const auto& profile : launch_profiles)
        if (profile.implementation_id == implementation_id
            && profile.backend == backend
            && profile.precision == precision
            && profile.required_warp_width == warp_width
            && compute_capability >= profile.minimum_compute_capability
            && (profile.maximum_compute_capability == 0
                || compute_capability <= profile.maximum_compute_capability)
            && (profile.architecture_family.empty()
                || profile.architecture_family == architecture_family)) {
            const auto rank = std::tuple{
                profile.architecture_family.empty() ? 0 : 1,
                profile.minimum_compute_capability,
                profile.maximum_compute_capability == 0
                    ? std::numeric_limits<int>::min()
                    : -profile.maximum_compute_capability};
            if (selected == nullptr || rank > selected_rank) {
                selected = &profile;
                selected_rank = rank;
            }
        }
    return selected;
}

} // namespace symmetrix::execution
