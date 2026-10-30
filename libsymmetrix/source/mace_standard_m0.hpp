#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

namespace symmetrix::standard_m0 {

// Frozen compatibility accelerator. New M0 topology coverage belongs in RTC.
inline constexpr bool deprecated_for_new_specializations = true;
inline constexpr char deprecation_notice[] =
    "Built-in M0 topology specialization is deprecated; use RTC for new contracts.";
inline constexpr char schema[] = "symmetrix.execution.symmetric_contraction";
inline constexpr int schema_version = 1;
inline constexpr char module_id[] = "standard-m0-module-module-v1";
inline constexpr char scalar_module_id[] = "standard-m0-lmax0-module-v1";
inline constexpr char execution_profile[] = "fixed_weight_coordinate";
inline constexpr char derivative_signature[] = "fixed_weight_coordinate";
inline constexpr char launch_geometry[] = "persistent_warp8x32_node_channel";
inline constexpr char accumulator_policy[] = "fixed_order_scalar";
inline constexpr int module_revision = 2;
inline constexpr int scalar_module_revision = 1;
inline constexpr char structure_fingerprint[] =
    "sha256:3ab9a8681d9a992095eb3a8afbc5c481ed6b50eace16ad62ae367a07d1fc34e1";
inline constexpr char scalar_structure_fingerprint[] =
    "sha256:3b4ab2d979488798a05c07b5d54c4c96c1026da3ee1eb93069fd2bacec76ac90";
inline constexpr int channels = 0;
inline constexpr int type_count = 0;
inline constexpr int input_l_max = 3;
inline constexpr int output_l_max = 1;
inline constexpr int input_components = 16;
inline constexpr int output_components = 4;
inline constexpr int correlation = 3;
inline constexpr int total_terms = 422;
inline constexpr int scalar_output_l_max = 0;
inline constexpr int scalar_output_components = 1;
inline constexpr int scalar_total_terms = 94;
inline constexpr int team_size = 8;
inline constexpr int vector_length = 32;
inline constexpr int threads_per_block = team_size*vector_length;
inline constexpr int blocks_per_sm = 4;
inline constexpr int forward_launch_count = 1;
inline constexpr int reverse_launch_count = 1;
inline constexpr std::size_t scratch_bytes_per_node = 0;
inline constexpr bool uses_runtime_weights = true;
inline constexpr bool supports_parameter_gradients = false;
inline constexpr bool dynamic_channels = true;
inline constexpr bool dynamic_type_count = true;

inline constexpr int term_offsets[output_components+1] =
    {0, 94, 212, 304, 422};
inline constexpr int term_degrees[total_terms] =
    {1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
inline constexpr int term_components[total_terms][correlation] = {
        {0, -1, -1},
        {0, 0, -1},
        {1, 1, -1},
        {2, 2, -1},
        {3, 3, -1},
        {4, 4, -1},
        {5, 5, -1},
        {6, 6, -1},
        {7, 7, -1},
        {8, 8, -1},
        {9, 9, -1},
        {10, 10, -1},
        {11, 11, -1},
        {12, 12, -1},
        {13, 13, -1},
        {14, 14, -1},
        {15, 15, -1},
        {0, 0, 0},
        {0, 1, 1},
        {0, 2, 2},
        {0, 3, 3},
        {0, 4, 4},
        {0, 5, 5},
        {0, 6, 6},
        {0, 7, 7},
        {0, 8, 8},
        {0, 9, 9},
        {0, 10, 10},
        {0, 11, 11},
        {0, 12, 12},
        {0, 13, 13},
        {0, 14, 14},
        {0, 15, 15},
        {1, 1, 6},
        {1, 1, 8},
        {1, 2, 5},
        {1, 3, 4},
        {1, 4, 13},
        {1, 4, 15},
        {1, 5, 12},
        {1, 5, 14},
        {1, 6, 11},
        {1, 7, 10},
        {1, 8, 9},
        {1, 8, 11},
        {2, 2, 6},
        {2, 3, 7},
        {2, 4, 10},
        {2, 5, 11},
        {2, 6, 12},
        {2, 7, 13},
        {2, 8, 14},
        {3, 3, 6},
        {3, 3, 8},
        {3, 4, 9},
        {3, 4, 11},
        {3, 5, 10},
        {3, 6, 13},
        {3, 7, 12},
        {3, 7, 14},
        {3, 8, 13},
        {3, 8, 15},
        {4, 4, 6},
        {4, 5, 7},
        {4, 9, 13},
        {4, 10, 12},
        {4, 11, 13},
        {4, 11, 15},
        {5, 5, 6},
        {5, 5, 8},
        {5, 9, 14},
        {5, 10, 13},
        {5, 10, 15},
        {5, 11, 12},
        {5, 11, 14},
        {6, 6, 6},
        {6, 7, 7},
        {6, 8, 8},
        {6, 9, 9},
        {6, 11, 11},
        {6, 12, 12},
        {6, 13, 13},
        {6, 15, 15},
        {7, 7, 8},
        {7, 9, 10},
        {7, 10, 11},
        {7, 12, 13},
        {7, 13, 14},
        {7, 14, 15},
        {8, 9, 11},
        {8, 11, 11},
        {8, 12, 14},
        {8, 13, 13},
        {8, 13, 15},
        {1, -1, -1},
        {0, 1, -1},
        {1, 6, -1},
        {1, 8, -1},
        {2, 5, -1},
        {3, 4, -1},
        {4, 13, -1},
        {4, 15, -1},
        {5, 12, -1},
        {5, 14, -1},
        {6, 11, -1},
        {7, 10, -1},
        {8, 9, -1},
        {8, 11, -1},
        {0, 0, 1},
        {0, 1, 6},
        {0, 1, 8},
        {0, 2, 5},
        {0, 3, 4},
        {0, 4, 13},
        {0, 4, 15},
        {0, 5, 12},
        {0, 5, 14},
        {0, 6, 11},
        {0, 7, 10},
        {0, 8, 9},
        {0, 8, 11},
        {1, 1, 1},
        {1, 1, 9},
        {1, 1, 11},
        {1, 2, 2},
        {1, 2, 12},
        {1, 2, 14},
        {1, 3, 3},
        {1, 3, 13},
        {1, 3, 15},
        {1, 4, 4},
        {1, 5, 5},
        {1, 6, 6},
        {1, 6, 8},
        {1, 7, 7},
        {1, 8, 8},
        {1, 9, 9},
        {1, 9, 11},
        {1, 10, 10},
        {1, 11, 11},
        {1, 12, 12},
        {1, 12, 14},
        {1, 13, 13},
        {1, 13, 15},
        {1, 14, 14},
        {1, 15, 15},
        {2, 2, 11},
        {2, 3, 10},
        {2, 4, 7},
        {2, 5, 6},
        {2, 5, 8},
        {2, 9, 14},
        {2, 10, 13},
        {2, 10, 15},
        {2, 11, 12},
        {2, 11, 14},
        {3, 3, 9},
        {3, 3, 11},
        {3, 4, 6},
        {3, 4, 8},
        {3, 5, 7},
        {3, 9, 13},
        {3, 9, 15},
        {3, 10, 12},
        {3, 10, 14},
        {3, 11, 13},
        {3, 11, 15},
        {4, 4, 9},
        {4, 4, 11},
        {4, 5, 10},
        {4, 6, 13},
        {4, 6, 15},
        {4, 7, 12},
        {4, 7, 14},
        {4, 8, 13},
        {4, 8, 15},
        {5, 5, 9},
        {5, 5, 11},
        {5, 6, 12},
        {5, 6, 14},
        {5, 7, 13},
        {5, 7, 15},
        {5, 8, 12},
        {5, 8, 14},
        {6, 6, 11},
        {6, 7, 10},
        {6, 8, 9},
        {6, 8, 11},
        {7, 7, 9},
        {7, 7, 11},
        {7, 8, 10},
        {8, 8, 9},
        {8, 8, 11},
        {9, 9, 11},
        {9, 10, 10},
        {9, 11, 11},
        {9, 12, 14},
        {9, 13, 13},
        {9, 13, 15},
        {9, 14, 14},
        {10, 10, 11},
        {10, 12, 13},
        {10, 12, 15},
        {10, 13, 14},
        {10, 14, 15},
        {11, 11, 11},
        {11, 12, 12},
        {11, 12, 14},
        {11, 13, 13},
        {11, 13, 15},
        {11, 14, 14},
        {11, 15, 15},
        {2, -1, -1},
        {0, 2, -1},
        {1, 5, -1},
        {2, 6, -1},
        {3, 7, -1},
        {4, 10, -1},
        {5, 11, -1},
        {6, 12, -1},
        {7, 13, -1},
        {8, 14, -1},
        {0, 0, 2},
        {0, 1, 5},
        {0, 2, 6},
        {0, 3, 7},
        {0, 4, 10},
        {0, 5, 11},
        {0, 6, 12},
        {0, 7, 13},
        {0, 8, 14},
        {1, 1, 2},
        {1, 1, 12},
        {1, 1, 14},
        {1, 2, 11},
        {1, 3, 10},
        {1, 4, 7},
        {1, 5, 6},
        {1, 5, 8},
        {1, 9, 14},
        {1, 10, 13},
        {1, 10, 15},
        {1, 11, 12},
        {1, 11, 14},
        {2, 2, 2},
        {2, 2, 12},
        {2, 3, 3},
        {2, 3, 13},
        {2, 4, 4},
        {2, 5, 5},
        {2, 6, 6},
        {2, 7, 7},
        {2, 8, 8},
        {2, 9, 9},
        {2, 10, 10},
        {2, 11, 11},
        {2, 12, 12},
        {2, 13, 13},
        {2, 14, 14},
        {2, 15, 15},
        {3, 3, 12},
        {3, 3, 14},
        {3, 4, 5},
        {3, 6, 7},
        {3, 7, 8},
        {3, 9, 10},
        {3, 10, 11},
        {3, 12, 13},
        {3, 13, 14},
        {3, 14, 15},
        {4, 4, 12},
        {4, 5, 13},
        {4, 5, 15},
        {4, 6, 10},
        {4, 7, 9},
        {4, 7, 11},
        {5, 5, 12},
        {5, 5, 14},
        {5, 6, 11},
        {5, 7, 10},
        {5, 8, 9},
        {5, 8, 11},
        {6, 6, 12},
        {6, 7, 13},
        {6, 8, 14},
        {7, 7, 12},
        {7, 7, 14},
        {7, 8, 13},
        {7, 8, 15},
        {8, 8, 12},
        {9, 9, 12},
        {9, 10, 13},
        {9, 11, 14},
        {10, 10, 12},
        {10, 11, 13},
        {10, 11, 15},
        {11, 11, 12},
        {11, 11, 14},
        {12, 12, 12},
        {12, 13, 13},
        {12, 14, 14},
        {12, 15, 15},
        {13, 13, 14},
        {13, 14, 15},
        {3, -1, -1},
        {0, 3, -1},
        {1, 4, -1},
        {2, 7, -1},
        {3, 6, -1},
        {3, 8, -1},
        {4, 9, -1},
        {4, 11, -1},
        {5, 10, -1},
        {6, 13, -1},
        {7, 12, -1},
        {7, 14, -1},
        {8, 13, -1},
        {8, 15, -1},
        {0, 0, 3},
        {0, 1, 4},
        {0, 2, 7},
        {0, 3, 6},
        {0, 3, 8},
        {0, 4, 9},
        {0, 4, 11},
        {0, 5, 10},
        {0, 6, 13},
        {0, 7, 12},
        {0, 7, 14},
        {0, 8, 13},
        {0, 8, 15},
        {1, 1, 3},
        {1, 1, 13},
        {1, 1, 15},
        {1, 2, 10},
        {1, 3, 9},
        {1, 3, 11},
        {1, 4, 6},
        {1, 4, 8},
        {1, 5, 7},
        {1, 9, 13},
        {1, 9, 15},
        {1, 10, 12},
        {1, 10, 14},
        {1, 11, 13},
        {1, 11, 15},
        {2, 2, 3},
        {2, 2, 13},
        {2, 3, 12},
        {2, 3, 14},
        {2, 4, 5},
        {2, 6, 7},
        {2, 7, 8},
        {2, 9, 10},
        {2, 10, 11},
        {2, 12, 13},
        {2, 13, 14},
        {2, 14, 15},
        {3, 3, 3},
        {3, 3, 13},
        {3, 3, 15},
        {3, 4, 4},
        {3, 5, 5},
        {3, 6, 6},
        {3, 6, 8},
        {3, 7, 7},
        {3, 8, 8},
        {3, 9, 9},
        {3, 9, 11},
        {3, 10, 10},
        {3, 11, 11},
        {3, 12, 12},
        {3, 12, 14},
        {3, 13, 13},
        {3, 13, 15},
        {3, 14, 14},
        {3, 15, 15},
        {4, 4, 13},
        {4, 4, 15},
        {4, 5, 12},
        {4, 5, 14},
        {4, 6, 9},
        {4, 6, 11},
        {4, 7, 10},
        {4, 8, 9},
        {4, 8, 11},
        {5, 5, 13},
        {5, 5, 15},
        {5, 6, 10},
        {5, 7, 9},
        {5, 7, 11},
        {5, 8, 10},
        {6, 6, 13},
        {6, 7, 12},
        {6, 7, 14},
        {6, 8, 13},
        {6, 8, 15},
        {7, 7, 13},
        {7, 7, 15},
        {7, 8, 12},
        {7, 8, 14},
        {8, 8, 13},
        {8, 8, 15},
        {9, 9, 13},
        {9, 10, 12},
        {9, 10, 14},
        {9, 11, 13},
        {9, 11, 15},
        {10, 10, 13},
        {10, 10, 15},
        {10, 11, 12},
        {10, 11, 14},
        {11, 11, 13},
        {11, 11, 15},
        {12, 12, 13},
        {12, 13, 14},
        {12, 14, 15},
        {13, 13, 13},
        {13, 13, 15},
        {13, 14, 14},
        {13, 15, 15},
        {14, 14, 15}
    };

template <typename ExecutionSpace>
inline constexpr bool host_execution_space = std::is_same_v<
    typename ExecutionSpace::memory_space, Kokkos::HostSpace>;
template <typename ExecutionSpace>
inline constexpr bool execution_space_supported =
    host_execution_space<ExecutionSpace>;
#ifdef KOKKOS_ENABLE_CUDA
template <>
inline constexpr bool execution_space_supported<Kokkos::Cuda> = true;
#endif
#ifdef KOKKOS_ENABLE_HIP
template <>
inline constexpr bool execution_space_supported<Kokkos::HIP> = true;
#endif

inline bool matches_structure(
    int runtime_channels,
    int runtime_type_count,
    int runtime_input_components,
    int runtime_output_components,
    int runtime_correlation,
    std::span<const int> runtime_term_counts)
{
    if (runtime_channels <= 0 || runtime_type_count <= 0
        || runtime_input_components != input_components
        || runtime_output_components != output_components
        || runtime_correlation != correlation
        || runtime_term_counts.size() != output_components)
        return false;
    for (int output=0; output<output_components; ++output)
        if (runtime_term_counts[output]
                != term_offsets[output+1]-term_offsets[output])
            return false;
    return true;
}

inline bool matches_term(
    int output_component,
    int term_index,
    std::span<const int> runtime_components)
{
    if (output_component < 0 || output_component >= output_components
        || term_index < 0
        || term_index >= term_offsets[output_component+1]
            -term_offsets[output_component])
        return false;
    const int global_term = term_offsets[output_component]+term_index;
    if (runtime_components.size()
            != static_cast<std::size_t>(term_degrees[global_term]))
        return false;
    for (std::size_t index=0; index<runtime_components.size(); ++index)
        if (runtime_components[index] != term_components[global_term][index])
            return false;
    return true;
}

inline bool matches_scalar_structure(
    int runtime_channels,
    int runtime_type_count,
    int runtime_input_components,
    int runtime_output_components,
    int runtime_correlation,
    std::span<const int> runtime_term_counts)
{
    return runtime_channels > 0 && runtime_type_count > 0
        && runtime_input_components == input_components
        && runtime_output_components == scalar_output_components
        && runtime_correlation == correlation
        && runtime_term_counts.size() == scalar_output_components
        && runtime_term_counts[0] == scalar_total_terms;
}

inline bool matches_scalar_term(
    int output_component,
    int term_index,
    std::span<const int> runtime_components)
{
    return output_component == 0
        && term_index >= 0
        && term_index < scalar_total_terms
        && matches_term(output_component, term_index, runtime_components);
}

inline constexpr int default_host_channel_tile = 16;

inline int host_channel_tile()
{
    static const int tile = [] {
        const char* value = std::getenv("SYMMETRIX_STANDARD_M0_HOST_TILE");
        if (value == nullptr || std::string_view(value).empty())
            return default_host_channel_tile;
        if (std::string_view(value) == "1")
            return 1;
        if (std::string_view(value) == "4")
            return 4;
        if (std::string_view(value) == "8")
            return 8;
        if (std::string_view(value) == "16")
            return 16;
        return default_host_channel_tile;
    }();
    return tile;
}

template <std::size_t Term>
KOKKOS_INLINE_FUNCTION constexpr int output_for_term()
{
    if constexpr (Term < static_cast<std::size_t>(term_offsets[1]))
        return 0;
    else if constexpr (Term < static_cast<std::size_t>(term_offsets[2]))
        return 1;
    else if constexpr (Term < static_cast<std::size_t>(term_offsets[3]))
        return 2;
    else
        return 3;
}

template <int Tile, std::size_t Term, typename Scalar, typename WeightsView>
KOKKOS_INLINE_FUNCTION void accumulate_forward_term(
    const int type,
    const int channel_begin,
    const int active_channels,
    const Scalar (&input_values)[input_components][Tile],
    const WeightsView& weights,
    Scalar (&output_values)[output_components][Tile])
{
    constexpr int output_component = output_for_term<Term>();
    constexpr int degree = term_degrees[Term];
    constexpr int component_0 = term_components[Term][0];
    constexpr int component_1 = term_components[Term][1];
    constexpr int component_2 = term_components[Term][2];
#if defined(_OPENMP)
#pragma omp simd
#endif
    for (int lane=0; lane<active_channels; ++lane) {
        Scalar product = input_values[component_0][lane];
        if constexpr (degree >= 2)
            product *= input_values[component_1][lane];
        if constexpr (degree >= 3)
            product *= input_values[component_2][lane];
        output_values[output_component][lane] +=
            weights(type,Term,channel_begin+lane)*product;
    }
}

template <int Tile, typename Scalar, typename WeightsView, std::size_t... Terms>
KOKKOS_INLINE_FUNCTION void accumulate_forward_terms(
    const int type,
    const int channel_begin,
    const int active_channels,
    const Scalar (&input_values)[input_components][Tile],
    const WeightsView& weights,
    Scalar (&output_values)[output_components][Tile],
    std::index_sequence<Terms...>)
{
    (accumulate_forward_term<Tile,Terms>(
        type, channel_begin, active_channels,
        input_values, weights, output_values), ...);
}

template <int Tile, int ActiveOutputComponents = output_components,
          std::size_t TermCount = total_terms,
          typename ExecutionSpace, typename NodeTypesView,
          typename InputView, typename WeightsView, typename OutputView>
void launch_forward_host_tiled(
    const ExecutionSpace& execution_space,
    const int num_nodes,
    const int channel_count,
    const NodeTypesView node_types,
    const InputView input,
    const WeightsView weights,
    const OutputView output)
{
    using Scalar = typename InputView::non_const_value_type;
    Kokkos::parallel_for(
        "StandardM0::forward_host_tiled",
        Kokkos::RangePolicy<ExecutionSpace,
            Kokkos::IndexType<std::size_t>>(
            execution_space, 0, static_cast<std::size_t>(num_nodes)),
        KOKKOS_LAMBDA (const std::size_t node) {
            const int type = node_types(node);
            for (int channel_begin=0; channel_begin<channel_count;
                 channel_begin+=Tile) {
                const int active_channels = Kokkos::min(
                    Tile, channel_count-channel_begin);
                Scalar input_values[input_components][Tile];
                Scalar output_values[output_components][Tile] = {};
#if defined(_OPENMP)
#pragma omp simd
#endif
                for (int lane=0; lane<active_channels; ++lane)
                    for (int component=0; component<input_components; ++component)
                        input_values[component][lane] =
                            input(node,component,channel_begin+lane);
                accumulate_forward_terms<Tile>(
                    type, channel_begin, active_channels,
                    input_values, weights, output_values,
                    std::make_index_sequence<TermCount>{});
                for (int component=0; component<ActiveOutputComponents; ++component) {
#if defined(_OPENMP)
#pragma omp simd
#endif
                    for (int lane=0; lane<active_channels; ++lane)
                        output(node,component,channel_begin+lane) =
                            output_values[component][lane];
                }
            }
        });
}

template <int Tile, std::size_t Term, typename Scalar, typename WeightsView>
KOKKOS_INLINE_FUNCTION void accumulate_reverse_term(
    const int type,
    const int channel_begin,
    const int active_channels,
    const Scalar (&input_values)[input_components][Tile],
    const Scalar (&output_adjoints)[output_components][Tile],
    const WeightsView& weights,
    Scalar (&input_adjoints)[input_components][Tile])
{
    constexpr int output_component = output_for_term<Term>();
    constexpr int degree = term_degrees[Term];
    constexpr int component_0 = term_components[Term][0];
    constexpr int component_1 = term_components[Term][1];
    constexpr int component_2 = term_components[Term][2];
#if defined(_OPENMP)
#pragma omp simd
#endif
    for (int lane=0; lane<active_channels; ++lane) {
        const Scalar scale = output_adjoints[output_component][lane]
            *weights(type,Term,channel_begin+lane);
        if constexpr (degree == 1) {
            input_adjoints[component_0][lane] += scale;
        } else if constexpr (degree == 2) {
            if constexpr (component_0 == component_1) {
                input_adjoints[component_0][lane] +=
                    scale*(Scalar(2)*input_values[component_0][lane]);
            } else {
                input_adjoints[component_0][lane] +=
                    scale*input_values[component_1][lane];
                input_adjoints[component_1][lane] +=
                    scale*input_values[component_0][lane];
            }
        } else if constexpr (
                component_0 == component_1 && component_1 == component_2) {
            input_adjoints[component_0][lane] += scale
                *(Scalar(3)*input_values[component_0][lane]
                    *input_values[component_0][lane]);
        } else if constexpr (component_0 == component_1) {
            input_adjoints[component_0][lane] += scale
                *(Scalar(2)*input_values[component_0][lane]
                    *input_values[component_2][lane]);
            input_adjoints[component_2][lane] += scale
                *(input_values[component_0][lane]
                    *input_values[component_0][lane]);
        } else if constexpr (component_0 == component_2) {
            input_adjoints[component_0][lane] += scale
                *(Scalar(2)*input_values[component_0][lane]
                    *input_values[component_1][lane]);
            input_adjoints[component_1][lane] += scale
                *(input_values[component_0][lane]
                    *input_values[component_0][lane]);
        } else if constexpr (component_1 == component_2) {
            input_adjoints[component_1][lane] += scale
                *(Scalar(2)*input_values[component_1][lane]
                    *input_values[component_0][lane]);
            input_adjoints[component_0][lane] += scale
                *(input_values[component_1][lane]
                    *input_values[component_1][lane]);
        } else {
            input_adjoints[component_0][lane] += scale
                *(input_values[component_1][lane]
                    *input_values[component_2][lane]);
            input_adjoints[component_1][lane] += scale
                *(input_values[component_0][lane]
                    *input_values[component_2][lane]);
            input_adjoints[component_2][lane] += scale
                *(input_values[component_0][lane]
                    *input_values[component_1][lane]);
        }
    }
}

template <int Tile, typename Scalar, typename WeightsView, std::size_t... Terms>
KOKKOS_INLINE_FUNCTION void accumulate_reverse_terms(
    const int type,
    const int channel_begin,
    const int active_channels,
    const Scalar (&input_values)[input_components][Tile],
    const Scalar (&output_adjoints)[output_components][Tile],
    const WeightsView& weights,
    Scalar (&input_adjoints)[input_components][Tile],
    std::index_sequence<Terms...>)
{
    (accumulate_reverse_term<Tile,Terms>(
        type, channel_begin, active_channels,
        input_values, output_adjoints, weights, input_adjoints), ...);
}

template <int Tile, int ActiveOutputComponents = output_components,
          std::size_t TermCount = total_terms,
          typename ExecutionSpace, typename NodeTypesView,
          typename InputView, typename WeightsView,
          typename OutputAdjointView, typename InputAdjointView,
          typename InputScaleAdjointView>
void launch_reverse_host_tiled(
    const ExecutionSpace& execution_space,
    const int num_nodes,
    const int channel_count,
    const NodeTypesView node_types,
    const InputView input,
    const WeightsView weights,
    const OutputAdjointView output_adjoint,
    const InputAdjointView input_adjoint,
    const InputScaleAdjointView input_scale_adjoint,
    const bool capture_input_scale_adjoint)
{
    using Scalar = typename InputView::non_const_value_type;
    Kokkos::parallel_for(
        "StandardM0::reverse_host_tiled",
        Kokkos::RangePolicy<ExecutionSpace,
            Kokkos::IndexType<std::size_t>>(
            execution_space, 0, static_cast<std::size_t>(num_nodes)),
        KOKKOS_LAMBDA (const std::size_t node) {
            const int type = node_types(node);
            double scale_adjoint = 0.0;
            for (int channel_begin=0; channel_begin<channel_count;
                 channel_begin+=Tile) {
                const int active_channels = Kokkos::min(
                    Tile, channel_count-channel_begin);
                Scalar input_values[input_components][Tile];
                Scalar output_adjoints[output_components][Tile];
                Scalar input_adjoints[input_components][Tile] = {};
#if defined(_OPENMP)
#pragma omp simd
#endif
                for (int lane=0; lane<active_channels; ++lane) {
                    for (int component=0; component<input_components; ++component)
                        input_values[component][lane] =
                            input(node,component,channel_begin+lane);
                    for (int component=0; component<ActiveOutputComponents; ++component)
                        output_adjoints[component][lane] =
                            output_adjoint(node,component,channel_begin+lane);
                }
                accumulate_reverse_terms<Tile>(
                    type, channel_begin, active_channels,
                    input_values, output_adjoints, weights, input_adjoints,
                    std::make_index_sequence<TermCount>{});
                for (int component=0; component<input_components; ++component) {
#if defined(_OPENMP)
#pragma omp simd reduction(+:scale_adjoint)
#endif
                    for (int lane=0; lane<active_channels; ++lane) {
                        input_adjoint(node,component,channel_begin+lane) =
                            input_adjoints[component][lane];
                        if (capture_input_scale_adjoint)
                            scale_adjoint += static_cast<double>(
                                input_values[component][lane])
                                *input_adjoints[component][lane];
                    }
                }
            }
            if (capture_input_scale_adjoint)
                input_scale_adjoint(node) += scale_adjoint;
        });
}

template <typename ExecutionSpace, typename NodeTypesView,
          typename InputView, typename WeightsView, typename OutputView>
bool launch_forward(
    const ExecutionSpace& execution_space,
    int persistent_blocks,
    int num_nodes,
    NodeTypesView node_types,
    InputView input,
    WeightsView weights,
    OutputView output)
{
    if constexpr (!execution_space_supported<ExecutionSpace>) {
        return false;
    } else {
        using Scalar = typename InputView::non_const_value_type;
        static_assert(std::is_same_v<Scalar, float>
            || std::is_same_v<Scalar, double>);
        static_assert(std::is_same_v<
            typename WeightsView::non_const_value_type, Scalar>);
        static_assert(std::is_same_v<
            typename OutputView::non_const_value_type, Scalar>);
        static_assert(std::is_same_v<
            typename InputView::array_layout, Kokkos::LayoutRight>);
        static_assert(std::is_same_v<
            typename WeightsView::array_layout, Kokkos::LayoutRight>);
        static_assert(std::is_same_v<
            typename OutputView::array_layout, Kokkos::LayoutRight>);
        if (num_nodes == 0)
            return true;
        if (num_nodes < 0)
            return false;
        if constexpr (!host_execution_space<ExecutionSpace>)
            if (persistent_blocks <= 0)
                return false;
        const int channel_count = input.extent_int(2);
        if (channel_count <= 0
            || node_types.extent_int(0) < num_nodes
            || input.extent_int(0) < num_nodes
            || input.extent_int(1) != input_components
            || weights.extent_int(0) <= 0
            || weights.extent_int(1) != total_terms
            || weights.extent_int(2) != channel_count
            || output.extent_int(0) < num_nodes
            || output.extent_int(1) != output_components
            || output.extent_int(2) != channel_count)
            return false;
        const auto process_owner = KOKKOS_LAMBDA (
            const int node, const int channel) {
                            const std::size_t node_index = node;
                            const int type = node_types(node);
                            const Scalar x_0 = input(node_index,0,channel);
                            const Scalar x_1 = input(node_index,1,channel);
                            const Scalar x_2 = input(node_index,2,channel);
                            const Scalar x_3 = input(node_index,3,channel);
                            const Scalar x_4 = input(node_index,4,channel);
                            const Scalar x_5 = input(node_index,5,channel);
                            const Scalar x_6 = input(node_index,6,channel);
                            const Scalar x_7 = input(node_index,7,channel);
                            const Scalar x_8 = input(node_index,8,channel);
                            const Scalar x_9 = input(node_index,9,channel);
                            const Scalar x_10 = input(node_index,10,channel);
                            const Scalar x_11 = input(node_index,11,channel);
                            const Scalar x_12 = input(node_index,12,channel);
                            const Scalar x_13 = input(node_index,13,channel);
                            const Scalar x_14 = input(node_index,14,channel);
                            const Scalar x_15 = input(node_index,15,channel);
                            Scalar value_0 = Scalar(0);
                            Scalar value_1 = Scalar(0);
                            Scalar value_2 = Scalar(0);
                            Scalar value_3 = Scalar(0);
                            {
                                const Scalar product = x_0;
                                value_0 +=
                                    weights(type,0,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*x_0);
                                value_0 +=
                                    weights(type,1,channel)*product;
                            }
                            {
                                const Scalar product = (x_1*x_1);
                                value_0 +=
                                    weights(type,2,channel)*product;
                            }
                            {
                                const Scalar product = (x_2*x_2);
                                value_0 +=
                                    weights(type,3,channel)*product;
                            }
                            {
                                const Scalar product = (x_3*x_3);
                                value_0 +=
                                    weights(type,4,channel)*product;
                            }
                            {
                                const Scalar product = (x_4*x_4);
                                value_0 +=
                                    weights(type,5,channel)*product;
                            }
                            {
                                const Scalar product = (x_5*x_5);
                                value_0 +=
                                    weights(type,6,channel)*product;
                            }
                            {
                                const Scalar product = (x_6*x_6);
                                value_0 +=
                                    weights(type,7,channel)*product;
                            }
                            {
                                const Scalar product = (x_7*x_7);
                                value_0 +=
                                    weights(type,8,channel)*product;
                            }
                            {
                                const Scalar product = (x_8*x_8);
                                value_0 +=
                                    weights(type,9,channel)*product;
                            }
                            {
                                const Scalar product = (x_9*x_9);
                                value_0 +=
                                    weights(type,10,channel)*product;
                            }
                            {
                                const Scalar product = (x_10*x_10);
                                value_0 +=
                                    weights(type,11,channel)*product;
                            }
                            {
                                const Scalar product = (x_11*x_11);
                                value_0 +=
                                    weights(type,12,channel)*product;
                            }
                            {
                                const Scalar product = (x_12*x_12);
                                value_0 +=
                                    weights(type,13,channel)*product;
                            }
                            {
                                const Scalar product = (x_13*x_13);
                                value_0 +=
                                    weights(type,14,channel)*product;
                            }
                            {
                                const Scalar product = (x_14*x_14);
                                value_0 +=
                                    weights(type,15,channel)*product;
                            }
                            {
                                const Scalar product = (x_15*x_15);
                                value_0 +=
                                    weights(type,16,channel)*product;
                            }
                            {
                                const Scalar product = ((x_0*x_0)*x_0);
                                value_0 +=
                                    weights(type,17,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_1*x_1));
                                value_0 +=
                                    weights(type,18,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_2*x_2));
                                value_0 +=
                                    weights(type,19,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_3*x_3));
                                value_0 +=
                                    weights(type,20,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_4*x_4));
                                value_0 +=
                                    weights(type,21,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_5*x_5));
                                value_0 +=
                                    weights(type,22,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_6*x_6));
                                value_0 +=
                                    weights(type,23,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_7*x_7));
                                value_0 +=
                                    weights(type,24,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_8*x_8));
                                value_0 +=
                                    weights(type,25,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_9*x_9));
                                value_0 +=
                                    weights(type,26,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_10*x_10));
                                value_0 +=
                                    weights(type,27,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_11*x_11));
                                value_0 +=
                                    weights(type,28,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_12*x_12));
                                value_0 +=
                                    weights(type,29,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_13*x_13));
                                value_0 +=
                                    weights(type,30,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_14*x_14));
                                value_0 +=
                                    weights(type,31,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_15*x_15));
                                value_0 +=
                                    weights(type,32,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_6)*x_1);
                                value_0 +=
                                    weights(type,33,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_8)*x_1);
                                value_0 +=
                                    weights(type,34,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_5)*x_2);
                                value_0 +=
                                    weights(type,35,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_4)*x_3);
                                value_0 +=
                                    weights(type,36,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_4)*x_13);
                                value_0 +=
                                    weights(type,37,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_4)*x_15);
                                value_0 +=
                                    weights(type,38,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_5)*x_12);
                                value_0 +=
                                    weights(type,39,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_5)*x_14);
                                value_0 +=
                                    weights(type,40,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_6)*x_11);
                                value_0 +=
                                    weights(type,41,channel)*product;
                            }
                            {
                                const Scalar product = (x_1*(x_7*x_10));
                                value_0 +=
                                    weights(type,42,channel)*product;
                            }
                            {
                                const Scalar product = (x_1*(x_8*x_9));
                                value_0 +=
                                    weights(type,43,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_8)*x_11);
                                value_0 +=
                                    weights(type,44,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_6)*x_2);
                                value_0 +=
                                    weights(type,45,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_7)*x_3);
                                value_0 +=
                                    weights(type,46,channel)*product;
                            }
                            {
                                const Scalar product = (x_2*(x_4*x_10));
                                value_0 +=
                                    weights(type,47,channel)*product;
                            }
                            {
                                const Scalar product = (x_2*(x_5*x_11));
                                value_0 +=
                                    weights(type,48,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_6)*x_12);
                                value_0 +=
                                    weights(type,49,channel)*product;
                            }
                            {
                                const Scalar product = (x_2*(x_7*x_13));
                                value_0 +=
                                    weights(type,50,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_8)*x_14);
                                value_0 +=
                                    weights(type,51,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_6)*x_3);
                                value_0 +=
                                    weights(type,52,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_8)*x_3);
                                value_0 +=
                                    weights(type,53,channel)*product;
                            }
                            {
                                const Scalar product = (x_3*(x_4*x_9));
                                value_0 +=
                                    weights(type,54,channel)*product;
                            }
                            {
                                const Scalar product = (x_3*(x_4*x_11));
                                value_0 +=
                                    weights(type,55,channel)*product;
                            }
                            {
                                const Scalar product = (x_3*(x_5*x_10));
                                value_0 +=
                                    weights(type,56,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_6)*x_13);
                                value_0 +=
                                    weights(type,57,channel)*product;
                            }
                            {
                                const Scalar product = (x_3*(x_7*x_12));
                                value_0 +=
                                    weights(type,58,channel)*product;
                            }
                            {
                                const Scalar product = (x_3*(x_7*x_14));
                                value_0 +=
                                    weights(type,59,channel)*product;
                            }
                            {
                                const Scalar product = (x_3*(x_8*x_13));
                                value_0 +=
                                    weights(type,60,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_8)*x_15);
                                value_0 +=
                                    weights(type,61,channel)*product;
                            }
                            {
                                const Scalar product = ((x_4*x_4)*x_6);
                                value_0 +=
                                    weights(type,62,channel)*product;
                            }
                            {
                                const Scalar product = ((x_4*x_5)*x_7);
                                value_0 +=
                                    weights(type,63,channel)*product;
                            }
                            {
                                const Scalar product = ((x_4*x_9)*x_13);
                                value_0 +=
                                    weights(type,64,channel)*product;
                            }
                            {
                                const Scalar product = ((x_4*x_10)*x_12);
                                value_0 +=
                                    weights(type,65,channel)*product;
                            }
                            {
                                const Scalar product = ((x_4*x_11)*x_13);
                                value_0 +=
                                    weights(type,66,channel)*product;
                            }
                            {
                                const Scalar product = ((x_4*x_11)*x_15);
                                value_0 +=
                                    weights(type,67,channel)*product;
                            }
                            {
                                const Scalar product = ((x_5*x_5)*x_6);
                                value_0 +=
                                    weights(type,68,channel)*product;
                            }
                            {
                                const Scalar product = ((x_5*x_5)*x_8);
                                value_0 +=
                                    weights(type,69,channel)*product;
                            }
                            {
                                const Scalar product = ((x_5*x_9)*x_14);
                                value_0 +=
                                    weights(type,70,channel)*product;
                            }
                            {
                                const Scalar product = ((x_5*x_10)*x_13);
                                value_0 +=
                                    weights(type,71,channel)*product;
                            }
                            {
                                const Scalar product = ((x_5*x_10)*x_15);
                                value_0 +=
                                    weights(type,72,channel)*product;
                            }
                            {
                                const Scalar product = ((x_5*x_11)*x_12);
                                value_0 +=
                                    weights(type,73,channel)*product;
                            }
                            {
                                const Scalar product = ((x_5*x_11)*x_14);
                                value_0 +=
                                    weights(type,74,channel)*product;
                            }
                            {
                                const Scalar product = ((x_6*x_6)*x_6);
                                value_0 +=
                                    weights(type,75,channel)*product;
                            }
                            {
                                const Scalar product = (x_6*(x_7*x_7));
                                value_0 +=
                                    weights(type,76,channel)*product;
                            }
                            {
                                const Scalar product = (x_6*(x_8*x_8));
                                value_0 +=
                                    weights(type,77,channel)*product;
                            }
                            {
                                const Scalar product = (x_6*(x_9*x_9));
                                value_0 +=
                                    weights(type,78,channel)*product;
                            }
                            {
                                const Scalar product = (x_6*(x_11*x_11));
                                value_0 +=
                                    weights(type,79,channel)*product;
                            }
                            {
                                const Scalar product = (x_6*(x_12*x_12));
                                value_0 +=
                                    weights(type,80,channel)*product;
                            }
                            {
                                const Scalar product = (x_6*(x_13*x_13));
                                value_0 +=
                                    weights(type,81,channel)*product;
                            }
                            {
                                const Scalar product = (x_6*(x_15*x_15));
                                value_0 +=
                                    weights(type,82,channel)*product;
                            }
                            {
                                const Scalar product = ((x_7*x_7)*x_8);
                                value_0 +=
                                    weights(type,83,channel)*product;
                            }
                            {
                                const Scalar product = ((x_7*x_10)*x_9);
                                value_0 +=
                                    weights(type,84,channel)*product;
                            }
                            {
                                const Scalar product = ((x_7*x_10)*x_11);
                                value_0 +=
                                    weights(type,85,channel)*product;
                            }
                            {
                                const Scalar product = ((x_7*x_13)*x_12);
                                value_0 +=
                                    weights(type,86,channel)*product;
                            }
                            {
                                const Scalar product = ((x_7*x_14)*x_13);
                                value_0 +=
                                    weights(type,87,channel)*product;
                            }
                            {
                                const Scalar product = ((x_7*x_14)*x_15);
                                value_0 +=
                                    weights(type,88,channel)*product;
                            }
                            {
                                const Scalar product = ((x_8*x_9)*x_11);
                                value_0 +=
                                    weights(type,89,channel)*product;
                            }
                            {
                                const Scalar product = (x_8*(x_11*x_11));
                                value_0 +=
                                    weights(type,90,channel)*product;
                            }
                            {
                                const Scalar product = ((x_8*x_12)*x_14);
                                value_0 +=
                                    weights(type,91,channel)*product;
                            }
                            {
                                const Scalar product = ((x_8*x_13)*x_13);
                                value_0 +=
                                    weights(type,92,channel)*product;
                            }
                            {
                                const Scalar product = ((x_8*x_13)*x_15);
                                value_0 +=
                                    weights(type,93,channel)*product;
                            }
                            {
                                const Scalar product = x_1;
                                value_1 +=
                                    weights(type,94,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*x_1);
                                value_1 +=
                                    weights(type,95,channel)*product;
                            }
                            {
                                const Scalar product = (x_1*x_6);
                                value_1 +=
                                    weights(type,96,channel)*product;
                            }
                            {
                                const Scalar product = (x_1*x_8);
                                value_1 +=
                                    weights(type,97,channel)*product;
                            }
                            {
                                const Scalar product = (x_2*x_5);
                                value_1 +=
                                    weights(type,98,channel)*product;
                            }
                            {
                                const Scalar product = (x_3*x_4);
                                value_1 +=
                                    weights(type,99,channel)*product;
                            }
                            {
                                const Scalar product = (x_4*x_13);
                                value_1 +=
                                    weights(type,100,channel)*product;
                            }
                            {
                                const Scalar product = (x_4*x_15);
                                value_1 +=
                                    weights(type,101,channel)*product;
                            }
                            {
                                const Scalar product = (x_5*x_12);
                                value_1 +=
                                    weights(type,102,channel)*product;
                            }
                            {
                                const Scalar product = (x_5*x_14);
                                value_1 +=
                                    weights(type,103,channel)*product;
                            }
                            {
                                const Scalar product = (x_6*x_11);
                                value_1 +=
                                    weights(type,104,channel)*product;
                            }
                            {
                                const Scalar product = (x_7*x_10);
                                value_1 +=
                                    weights(type,105,channel)*product;
                            }
                            {
                                const Scalar product = (x_8*x_9);
                                value_1 +=
                                    weights(type,106,channel)*product;
                            }
                            {
                                const Scalar product = (x_8*x_11);
                                value_1 +=
                                    weights(type,107,channel)*product;
                            }
                            {
                                const Scalar product = ((x_0*x_1)*x_0);
                                value_1 +=
                                    weights(type,108,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_1*x_6));
                                value_1 +=
                                    weights(type,109,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_1*x_8));
                                value_1 +=
                                    weights(type,110,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_2*x_5));
                                value_1 +=
                                    weights(type,111,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_3*x_4));
                                value_1 +=
                                    weights(type,112,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_4*x_13));
                                value_1 +=
                                    weights(type,113,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_4*x_15));
                                value_1 +=
                                    weights(type,114,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_5*x_12));
                                value_1 +=
                                    weights(type,115,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_5*x_14));
                                value_1 +=
                                    weights(type,116,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_6*x_11));
                                value_1 +=
                                    weights(type,117,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_7*x_10));
                                value_1 +=
                                    weights(type,118,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_8*x_9));
                                value_1 +=
                                    weights(type,119,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_8*x_11));
                                value_1 +=
                                    weights(type,120,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_1)*x_1);
                                value_1 +=
                                    weights(type,121,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_9)*x_1);
                                value_1 +=
                                    weights(type,122,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_11)*x_1);
                                value_1 +=
                                    weights(type,123,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_2)*x_2);
                                value_1 +=
                                    weights(type,124,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_12)*x_2);
                                value_1 +=
                                    weights(type,125,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_14)*x_2);
                                value_1 +=
                                    weights(type,126,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_3)*x_3);
                                value_1 +=
                                    weights(type,127,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_13)*x_3);
                                value_1 +=
                                    weights(type,128,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_15)*x_3);
                                value_1 +=
                                    weights(type,129,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_4)*x_4);
                                value_1 +=
                                    weights(type,130,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_5)*x_5);
                                value_1 +=
                                    weights(type,131,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_6)*x_6);
                                value_1 +=
                                    weights(type,132,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_8)*x_6);
                                value_1 +=
                                    weights(type,133,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_7)*x_7);
                                value_1 +=
                                    weights(type,134,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_8)*x_8);
                                value_1 +=
                                    weights(type,135,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_9)*x_9);
                                value_1 +=
                                    weights(type,136,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_11)*x_9);
                                value_1 +=
                                    weights(type,137,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_10)*x_10);
                                value_1 +=
                                    weights(type,138,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_11)*x_11);
                                value_1 +=
                                    weights(type,139,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_12)*x_12);
                                value_1 +=
                                    weights(type,140,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_14)*x_12);
                                value_1 +=
                                    weights(type,141,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_13)*x_13);
                                value_1 +=
                                    weights(type,142,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_15)*x_13);
                                value_1 +=
                                    weights(type,143,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_14)*x_14);
                                value_1 +=
                                    weights(type,144,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_15)*x_15);
                                value_1 +=
                                    weights(type,145,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_11)*x_2);
                                value_1 +=
                                    weights(type,146,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_10)*x_3);
                                value_1 +=
                                    weights(type,147,channel)*product;
                            }
                            {
                                const Scalar product = (x_2*(x_4*x_7));
                                value_1 +=
                                    weights(type,148,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_5)*x_6);
                                value_1 +=
                                    weights(type,149,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_5)*x_8);
                                value_1 +=
                                    weights(type,150,channel)*product;
                            }
                            {
                                const Scalar product = (x_2*(x_9*x_14));
                                value_1 +=
                                    weights(type,151,channel)*product;
                            }
                            {
                                const Scalar product = (x_2*(x_10*x_13));
                                value_1 +=
                                    weights(type,152,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_10)*x_15);
                                value_1 +=
                                    weights(type,153,channel)*product;
                            }
                            {
                                const Scalar product = (x_2*(x_11*x_12));
                                value_1 +=
                                    weights(type,154,channel)*product;
                            }
                            {
                                const Scalar product = (x_2*(x_11*x_14));
                                value_1 +=
                                    weights(type,155,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_9)*x_3);
                                value_1 +=
                                    weights(type,156,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_11)*x_3);
                                value_1 +=
                                    weights(type,157,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_4)*x_6);
                                value_1 +=
                                    weights(type,158,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_4)*x_8);
                                value_1 +=
                                    weights(type,159,channel)*product;
                            }
                            {
                                const Scalar product = (x_3*(x_5*x_7));
                                value_1 +=
                                    weights(type,160,channel)*product;
                            }
                            {
                                const Scalar product = (x_3*(x_9*x_13));
                                value_1 +=
                                    weights(type,161,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_9)*x_15);
                                value_1 +=
                                    weights(type,162,channel)*product;
                            }
                            {
                                const Scalar product = (x_3*(x_10*x_12));
                                value_1 +=
                                    weights(type,163,channel)*product;
                            }
                            {
                                const Scalar product = (x_3*(x_10*x_14));
                                value_1 +=
                                    weights(type,164,channel)*product;
                            }
                            {
                                const Scalar product = (x_3*(x_11*x_13));
                                value_1 +=
                                    weights(type,165,channel)*product;
                            }
                            {
                                const Scalar product = (x_3*(x_11*x_15));
                                value_1 +=
                                    weights(type,166,channel)*product;
                            }
                            {
                                const Scalar product = ((x_4*x_4)*x_9);
                                value_1 +=
                                    weights(type,167,channel)*product;
                            }
                            {
                                const Scalar product = ((x_4*x_4)*x_11);
                                value_1 +=
                                    weights(type,168,channel)*product;
                            }
                            {
                                const Scalar product = ((x_4*x_5)*x_10);
                                value_1 +=
                                    weights(type,169,channel)*product;
                            }
                            {
                                const Scalar product = ((x_4*x_13)*x_6);
                                value_1 +=
                                    weights(type,170,channel)*product;
                            }
                            {
                                const Scalar product = ((x_4*x_15)*x_6);
                                value_1 +=
                                    weights(type,171,channel)*product;
                            }
                            {
                                const Scalar product = ((x_4*x_7)*x_12);
                                value_1 +=
                                    weights(type,172,channel)*product;
                            }
                            {
                                const Scalar product = ((x_4*x_7)*x_14);
                                value_1 +=
                                    weights(type,173,channel)*product;
                            }
                            {
                                const Scalar product = ((x_4*x_13)*x_8);
                                value_1 +=
                                    weights(type,174,channel)*product;
                            }
                            {
                                const Scalar product = ((x_4*x_15)*x_8);
                                value_1 +=
                                    weights(type,175,channel)*product;
                            }
                            {
                                const Scalar product = ((x_5*x_5)*x_9);
                                value_1 +=
                                    weights(type,176,channel)*product;
                            }
                            {
                                const Scalar product = ((x_5*x_5)*x_11);
                                value_1 +=
                                    weights(type,177,channel)*product;
                            }
                            {
                                const Scalar product = ((x_5*x_12)*x_6);
                                value_1 +=
                                    weights(type,178,channel)*product;
                            }
                            {
                                const Scalar product = ((x_5*x_14)*x_6);
                                value_1 +=
                                    weights(type,179,channel)*product;
                            }
                            {
                                const Scalar product = ((x_5*x_7)*x_13);
                                value_1 +=
                                    weights(type,180,channel)*product;
                            }
                            {
                                const Scalar product = ((x_5*x_7)*x_15);
                                value_1 +=
                                    weights(type,181,channel)*product;
                            }
                            {
                                const Scalar product = ((x_5*x_12)*x_8);
                                value_1 +=
                                    weights(type,182,channel)*product;
                            }
                            {
                                const Scalar product = ((x_5*x_14)*x_8);
                                value_1 +=
                                    weights(type,183,channel)*product;
                            }
                            {
                                const Scalar product = ((x_6*x_11)*x_6);
                                value_1 +=
                                    weights(type,184,channel)*product;
                            }
                            {
                                const Scalar product = (x_6*(x_7*x_10));
                                value_1 +=
                                    weights(type,185,channel)*product;
                            }
                            {
                                const Scalar product = (x_6*(x_8*x_9));
                                value_1 +=
                                    weights(type,186,channel)*product;
                            }
                            {
                                const Scalar product = ((x_6*x_11)*x_8);
                                value_1 +=
                                    weights(type,187,channel)*product;
                            }
                            {
                                const Scalar product = ((x_7*x_7)*x_9);
                                value_1 +=
                                    weights(type,188,channel)*product;
                            }
                            {
                                const Scalar product = ((x_7*x_7)*x_11);
                                value_1 +=
                                    weights(type,189,channel)*product;
                            }
                            {
                                const Scalar product = ((x_7*x_10)*x_8);
                                value_1 +=
                                    weights(type,190,channel)*product;
                            }
                            {
                                const Scalar product = ((x_8*x_9)*x_8);
                                value_1 +=
                                    weights(type,191,channel)*product;
                            }
                            {
                                const Scalar product = ((x_8*x_11)*x_8);
                                value_1 +=
                                    weights(type,192,channel)*product;
                            }
                            {
                                const Scalar product = ((x_9*x_11)*x_9);
                                value_1 +=
                                    weights(type,193,channel)*product;
                            }
                            {
                                const Scalar product = ((x_9*x_10)*x_10);
                                value_1 +=
                                    weights(type,194,channel)*product;
                            }
                            {
                                const Scalar product = ((x_9*x_11)*x_11);
                                value_1 +=
                                    weights(type,195,channel)*product;
                            }
                            {
                                const Scalar product = ((x_9*x_14)*x_12);
                                value_1 +=
                                    weights(type,196,channel)*product;
                            }
                            {
                                const Scalar product = ((x_9*x_13)*x_13);
                                value_1 +=
                                    weights(type,197,channel)*product;
                            }
                            {
                                const Scalar product = ((x_9*x_13)*x_15);
                                value_1 +=
                                    weights(type,198,channel)*product;
                            }
                            {
                                const Scalar product = ((x_9*x_14)*x_14);
                                value_1 +=
                                    weights(type,199,channel)*product;
                            }
                            {
                                const Scalar product = ((x_10*x_10)*x_11);
                                value_1 +=
                                    weights(type,200,channel)*product;
                            }
                            {
                                const Scalar product = ((x_10*x_13)*x_12);
                                value_1 +=
                                    weights(type,201,channel)*product;
                            }
                            {
                                const Scalar product = ((x_10*x_12)*x_15);
                                value_1 +=
                                    weights(type,202,channel)*product;
                            }
                            {
                                const Scalar product = ((x_10*x_14)*x_13);
                                value_1 +=
                                    weights(type,203,channel)*product;
                            }
                            {
                                const Scalar product = ((x_10*x_14)*x_15);
                                value_1 +=
                                    weights(type,204,channel)*product;
                            }
                            {
                                const Scalar product = ((x_11*x_11)*x_11);
                                value_1 +=
                                    weights(type,205,channel)*product;
                            }
                            {
                                const Scalar product = ((x_11*x_12)*x_12);
                                value_1 +=
                                    weights(type,206,channel)*product;
                            }
                            {
                                const Scalar product = ((x_11*x_14)*x_12);
                                value_1 +=
                                    weights(type,207,channel)*product;
                            }
                            {
                                const Scalar product = ((x_11*x_13)*x_13);
                                value_1 +=
                                    weights(type,208,channel)*product;
                            }
                            {
                                const Scalar product = ((x_11*x_15)*x_13);
                                value_1 +=
                                    weights(type,209,channel)*product;
                            }
                            {
                                const Scalar product = ((x_11*x_14)*x_14);
                                value_1 +=
                                    weights(type,210,channel)*product;
                            }
                            {
                                const Scalar product = ((x_11*x_15)*x_15);
                                value_1 +=
                                    weights(type,211,channel)*product;
                            }
                            {
                                const Scalar product = x_2;
                                value_2 +=
                                    weights(type,212,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*x_2);
                                value_2 +=
                                    weights(type,213,channel)*product;
                            }
                            {
                                const Scalar product = (x_1*x_5);
                                value_2 +=
                                    weights(type,214,channel)*product;
                            }
                            {
                                const Scalar product = (x_2*x_6);
                                value_2 +=
                                    weights(type,215,channel)*product;
                            }
                            {
                                const Scalar product = (x_3*x_7);
                                value_2 +=
                                    weights(type,216,channel)*product;
                            }
                            {
                                const Scalar product = (x_4*x_10);
                                value_2 +=
                                    weights(type,217,channel)*product;
                            }
                            {
                                const Scalar product = (x_5*x_11);
                                value_2 +=
                                    weights(type,218,channel)*product;
                            }
                            {
                                const Scalar product = (x_6*x_12);
                                value_2 +=
                                    weights(type,219,channel)*product;
                            }
                            {
                                const Scalar product = (x_7*x_13);
                                value_2 +=
                                    weights(type,220,channel)*product;
                            }
                            {
                                const Scalar product = (x_8*x_14);
                                value_2 +=
                                    weights(type,221,channel)*product;
                            }
                            {
                                const Scalar product = ((x_0*x_2)*x_0);
                                value_2 +=
                                    weights(type,222,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_1*x_5));
                                value_2 +=
                                    weights(type,223,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_2*x_6));
                                value_2 +=
                                    weights(type,224,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_3*x_7));
                                value_2 +=
                                    weights(type,225,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_4*x_10));
                                value_2 +=
                                    weights(type,226,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_5*x_11));
                                value_2 +=
                                    weights(type,227,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_6*x_12));
                                value_2 +=
                                    weights(type,228,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_7*x_13));
                                value_2 +=
                                    weights(type,229,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_8*x_14));
                                value_2 +=
                                    weights(type,230,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_2)*x_1);
                                value_2 +=
                                    weights(type,231,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_1)*x_12);
                                value_2 +=
                                    weights(type,232,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_1)*x_14);
                                value_2 +=
                                    weights(type,233,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_11)*x_2);
                                value_2 +=
                                    weights(type,234,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_10)*x_3);
                                value_2 +=
                                    weights(type,235,channel)*product;
                            }
                            {
                                const Scalar product = (x_1*(x_4*x_7));
                                value_2 +=
                                    weights(type,236,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_5)*x_6);
                                value_2 +=
                                    weights(type,237,channel)*product;
                            }
                            {
                                const Scalar product = (x_1*(x_5*x_8));
                                value_2 +=
                                    weights(type,238,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_9)*x_14);
                                value_2 +=
                                    weights(type,239,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_10)*x_13);
                                value_2 +=
                                    weights(type,240,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_10)*x_15);
                                value_2 +=
                                    weights(type,241,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_11)*x_12);
                                value_2 +=
                                    weights(type,242,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_11)*x_14);
                                value_2 +=
                                    weights(type,243,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_2)*x_2);
                                value_2 +=
                                    weights(type,244,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_12)*x_2);
                                value_2 +=
                                    weights(type,245,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_3)*x_3);
                                value_2 +=
                                    weights(type,246,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_13)*x_3);
                                value_2 +=
                                    weights(type,247,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_4)*x_4);
                                value_2 +=
                                    weights(type,248,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_5)*x_5);
                                value_2 +=
                                    weights(type,249,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_6)*x_6);
                                value_2 +=
                                    weights(type,250,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_7)*x_7);
                                value_2 +=
                                    weights(type,251,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_8)*x_8);
                                value_2 +=
                                    weights(type,252,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_9)*x_9);
                                value_2 +=
                                    weights(type,253,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_10)*x_10);
                                value_2 +=
                                    weights(type,254,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_11)*x_11);
                                value_2 +=
                                    weights(type,255,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_12)*x_12);
                                value_2 +=
                                    weights(type,256,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_13)*x_13);
                                value_2 +=
                                    weights(type,257,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_14)*x_14);
                                value_2 +=
                                    weights(type,258,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_15)*x_15);
                                value_2 +=
                                    weights(type,259,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_12)*x_3);
                                value_2 +=
                                    weights(type,260,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_14)*x_3);
                                value_2 +=
                                    weights(type,261,channel)*product;
                            }
                            {
                                const Scalar product = (x_3*(x_4*x_5));
                                value_2 +=
                                    weights(type,262,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_7)*x_6);
                                value_2 +=
                                    weights(type,263,channel)*product;
                            }
                            {
                                const Scalar product = (x_3*(x_7*x_8));
                                value_2 +=
                                    weights(type,264,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_10)*x_9);
                                value_2 +=
                                    weights(type,265,channel)*product;
                            }
                            {
                                const Scalar product = (x_3*(x_10*x_11));
                                value_2 +=
                                    weights(type,266,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_13)*x_12);
                                value_2 +=
                                    weights(type,267,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_14)*x_13);
                                value_2 +=
                                    weights(type,268,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_14)*x_15);
                                value_2 +=
                                    weights(type,269,channel)*product;
                            }
                            {
                                const Scalar product = ((x_4*x_4)*x_12);
                                value_2 +=
                                    weights(type,270,channel)*product;
                            }
                            {
                                const Scalar product = ((x_4*x_5)*x_13);
                                value_2 +=
                                    weights(type,271,channel)*product;
                            }
                            {
                                const Scalar product = ((x_4*x_5)*x_15);
                                value_2 +=
                                    weights(type,272,channel)*product;
                            }
                            {
                                const Scalar product = ((x_4*x_10)*x_6);
                                value_2 +=
                                    weights(type,273,channel)*product;
                            }
                            {
                                const Scalar product = ((x_4*x_7)*x_9);
                                value_2 +=
                                    weights(type,274,channel)*product;
                            }
                            {
                                const Scalar product = ((x_4*x_7)*x_11);
                                value_2 +=
                                    weights(type,275,channel)*product;
                            }
                            {
                                const Scalar product = ((x_5*x_5)*x_12);
                                value_2 +=
                                    weights(type,276,channel)*product;
                            }
                            {
                                const Scalar product = ((x_5*x_5)*x_14);
                                value_2 +=
                                    weights(type,277,channel)*product;
                            }
                            {
                                const Scalar product = ((x_5*x_11)*x_6);
                                value_2 +=
                                    weights(type,278,channel)*product;
                            }
                            {
                                const Scalar product = ((x_5*x_7)*x_10);
                                value_2 +=
                                    weights(type,279,channel)*product;
                            }
                            {
                                const Scalar product = ((x_5*x_8)*x_9);
                                value_2 +=
                                    weights(type,280,channel)*product;
                            }
                            {
                                const Scalar product = ((x_5*x_11)*x_8);
                                value_2 +=
                                    weights(type,281,channel)*product;
                            }
                            {
                                const Scalar product = ((x_6*x_12)*x_6);
                                value_2 +=
                                    weights(type,282,channel)*product;
                            }
                            {
                                const Scalar product = (x_6*(x_7*x_13));
                                value_2 +=
                                    weights(type,283,channel)*product;
                            }
                            {
                                const Scalar product = (x_6*(x_8*x_14));
                                value_2 +=
                                    weights(type,284,channel)*product;
                            }
                            {
                                const Scalar product = ((x_7*x_7)*x_12);
                                value_2 +=
                                    weights(type,285,channel)*product;
                            }
                            {
                                const Scalar product = ((x_7*x_7)*x_14);
                                value_2 +=
                                    weights(type,286,channel)*product;
                            }
                            {
                                const Scalar product = ((x_7*x_13)*x_8);
                                value_2 +=
                                    weights(type,287,channel)*product;
                            }
                            {
                                const Scalar product = ((x_7*x_8)*x_15);
                                value_2 +=
                                    weights(type,288,channel)*product;
                            }
                            {
                                const Scalar product = ((x_8*x_8)*x_12);
                                value_2 +=
                                    weights(type,289,channel)*product;
                            }
                            {
                                const Scalar product = ((x_9*x_9)*x_12);
                                value_2 +=
                                    weights(type,290,channel)*product;
                            }
                            {
                                const Scalar product = ((x_9*x_10)*x_13);
                                value_2 +=
                                    weights(type,291,channel)*product;
                            }
                            {
                                const Scalar product = ((x_9*x_11)*x_14);
                                value_2 +=
                                    weights(type,292,channel)*product;
                            }
                            {
                                const Scalar product = ((x_10*x_10)*x_12);
                                value_2 +=
                                    weights(type,293,channel)*product;
                            }
                            {
                                const Scalar product = ((x_10*x_11)*x_13);
                                value_2 +=
                                    weights(type,294,channel)*product;
                            }
                            {
                                const Scalar product = ((x_10*x_11)*x_15);
                                value_2 +=
                                    weights(type,295,channel)*product;
                            }
                            {
                                const Scalar product = ((x_11*x_11)*x_12);
                                value_2 +=
                                    weights(type,296,channel)*product;
                            }
                            {
                                const Scalar product = ((x_11*x_11)*x_14);
                                value_2 +=
                                    weights(type,297,channel)*product;
                            }
                            {
                                const Scalar product = ((x_12*x_12)*x_12);
                                value_2 +=
                                    weights(type,298,channel)*product;
                            }
                            {
                                const Scalar product = ((x_12*x_13)*x_13);
                                value_2 +=
                                    weights(type,299,channel)*product;
                            }
                            {
                                const Scalar product = ((x_12*x_14)*x_14);
                                value_2 +=
                                    weights(type,300,channel)*product;
                            }
                            {
                                const Scalar product = ((x_12*x_15)*x_15);
                                value_2 +=
                                    weights(type,301,channel)*product;
                            }
                            {
                                const Scalar product = ((x_13*x_14)*x_13);
                                value_2 +=
                                    weights(type,302,channel)*product;
                            }
                            {
                                const Scalar product = ((x_13*x_14)*x_15);
                                value_2 +=
                                    weights(type,303,channel)*product;
                            }
                            {
                                const Scalar product = x_3;
                                value_3 +=
                                    weights(type,304,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*x_3);
                                value_3 +=
                                    weights(type,305,channel)*product;
                            }
                            {
                                const Scalar product = (x_1*x_4);
                                value_3 +=
                                    weights(type,306,channel)*product;
                            }
                            {
                                const Scalar product = (x_2*x_7);
                                value_3 +=
                                    weights(type,307,channel)*product;
                            }
                            {
                                const Scalar product = (x_3*x_6);
                                value_3 +=
                                    weights(type,308,channel)*product;
                            }
                            {
                                const Scalar product = (x_3*x_8);
                                value_3 +=
                                    weights(type,309,channel)*product;
                            }
                            {
                                const Scalar product = (x_4*x_9);
                                value_3 +=
                                    weights(type,310,channel)*product;
                            }
                            {
                                const Scalar product = (x_4*x_11);
                                value_3 +=
                                    weights(type,311,channel)*product;
                            }
                            {
                                const Scalar product = (x_5*x_10);
                                value_3 +=
                                    weights(type,312,channel)*product;
                            }
                            {
                                const Scalar product = (x_6*x_13);
                                value_3 +=
                                    weights(type,313,channel)*product;
                            }
                            {
                                const Scalar product = (x_7*x_12);
                                value_3 +=
                                    weights(type,314,channel)*product;
                            }
                            {
                                const Scalar product = (x_7*x_14);
                                value_3 +=
                                    weights(type,315,channel)*product;
                            }
                            {
                                const Scalar product = (x_8*x_13);
                                value_3 +=
                                    weights(type,316,channel)*product;
                            }
                            {
                                const Scalar product = (x_8*x_15);
                                value_3 +=
                                    weights(type,317,channel)*product;
                            }
                            {
                                const Scalar product = ((x_0*x_3)*x_0);
                                value_3 +=
                                    weights(type,318,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_1*x_4));
                                value_3 +=
                                    weights(type,319,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_2*x_7));
                                value_3 +=
                                    weights(type,320,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_3*x_6));
                                value_3 +=
                                    weights(type,321,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_3*x_8));
                                value_3 +=
                                    weights(type,322,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_4*x_9));
                                value_3 +=
                                    weights(type,323,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_4*x_11));
                                value_3 +=
                                    weights(type,324,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_5*x_10));
                                value_3 +=
                                    weights(type,325,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_6*x_13));
                                value_3 +=
                                    weights(type,326,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_7*x_12));
                                value_3 +=
                                    weights(type,327,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_7*x_14));
                                value_3 +=
                                    weights(type,328,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_8*x_13));
                                value_3 +=
                                    weights(type,329,channel)*product;
                            }
                            {
                                const Scalar product = (x_0*(x_8*x_15));
                                value_3 +=
                                    weights(type,330,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_3)*x_1);
                                value_3 +=
                                    weights(type,331,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_1)*x_13);
                                value_3 +=
                                    weights(type,332,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_1)*x_15);
                                value_3 +=
                                    weights(type,333,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_10)*x_2);
                                value_3 +=
                                    weights(type,334,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_9)*x_3);
                                value_3 +=
                                    weights(type,335,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_11)*x_3);
                                value_3 +=
                                    weights(type,336,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_4)*x_6);
                                value_3 +=
                                    weights(type,337,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_4)*x_8);
                                value_3 +=
                                    weights(type,338,channel)*product;
                            }
                            {
                                const Scalar product = (x_1*(x_5*x_7));
                                value_3 +=
                                    weights(type,339,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_9)*x_13);
                                value_3 +=
                                    weights(type,340,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_9)*x_15);
                                value_3 +=
                                    weights(type,341,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_10)*x_12);
                                value_3 +=
                                    weights(type,342,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_10)*x_14);
                                value_3 +=
                                    weights(type,343,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_11)*x_13);
                                value_3 +=
                                    weights(type,344,channel)*product;
                            }
                            {
                                const Scalar product = ((x_1*x_11)*x_15);
                                value_3 +=
                                    weights(type,345,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_3)*x_2);
                                value_3 +=
                                    weights(type,346,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_13)*x_2);
                                value_3 +=
                                    weights(type,347,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_12)*x_3);
                                value_3 +=
                                    weights(type,348,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_14)*x_3);
                                value_3 +=
                                    weights(type,349,channel)*product;
                            }
                            {
                                const Scalar product = (x_2*(x_4*x_5));
                                value_3 +=
                                    weights(type,350,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_7)*x_6);
                                value_3 +=
                                    weights(type,351,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_7)*x_8);
                                value_3 +=
                                    weights(type,352,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_10)*x_9);
                                value_3 +=
                                    weights(type,353,channel)*product;
                            }
                            {
                                const Scalar product = (x_2*(x_10*x_11));
                                value_3 +=
                                    weights(type,354,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_13)*x_12);
                                value_3 +=
                                    weights(type,355,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_14)*x_13);
                                value_3 +=
                                    weights(type,356,channel)*product;
                            }
                            {
                                const Scalar product = ((x_2*x_14)*x_15);
                                value_3 +=
                                    weights(type,357,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_3)*x_3);
                                value_3 +=
                                    weights(type,358,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_13)*x_3);
                                value_3 +=
                                    weights(type,359,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_15)*x_3);
                                value_3 +=
                                    weights(type,360,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_4)*x_4);
                                value_3 +=
                                    weights(type,361,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_5)*x_5);
                                value_3 +=
                                    weights(type,362,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_6)*x_6);
                                value_3 +=
                                    weights(type,363,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_8)*x_6);
                                value_3 +=
                                    weights(type,364,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_7)*x_7);
                                value_3 +=
                                    weights(type,365,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_8)*x_8);
                                value_3 +=
                                    weights(type,366,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_9)*x_9);
                                value_3 +=
                                    weights(type,367,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_11)*x_9);
                                value_3 +=
                                    weights(type,368,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_10)*x_10);
                                value_3 +=
                                    weights(type,369,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_11)*x_11);
                                value_3 +=
                                    weights(type,370,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_12)*x_12);
                                value_3 +=
                                    weights(type,371,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_14)*x_12);
                                value_3 +=
                                    weights(type,372,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_13)*x_13);
                                value_3 +=
                                    weights(type,373,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_15)*x_13);
                                value_3 +=
                                    weights(type,374,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_14)*x_14);
                                value_3 +=
                                    weights(type,375,channel)*product;
                            }
                            {
                                const Scalar product = ((x_3*x_15)*x_15);
                                value_3 +=
                                    weights(type,376,channel)*product;
                            }
                            {
                                const Scalar product = ((x_4*x_4)*x_13);
                                value_3 +=
                                    weights(type,377,channel)*product;
                            }
                            {
                                const Scalar product = ((x_4*x_4)*x_15);
                                value_3 +=
                                    weights(type,378,channel)*product;
                            }
                            {
                                const Scalar product = ((x_4*x_5)*x_12);
                                value_3 +=
                                    weights(type,379,channel)*product;
                            }
                            {
                                const Scalar product = ((x_4*x_5)*x_14);
                                value_3 +=
                                    weights(type,380,channel)*product;
                            }
                            {
                                const Scalar product = ((x_4*x_9)*x_6);
                                value_3 +=
                                    weights(type,381,channel)*product;
                            }
                            {
                                const Scalar product = ((x_4*x_11)*x_6);
                                value_3 +=
                                    weights(type,382,channel)*product;
                            }
                            {
                                const Scalar product = ((x_4*x_7)*x_10);
                                value_3 +=
                                    weights(type,383,channel)*product;
                            }
                            {
                                const Scalar product = ((x_4*x_9)*x_8);
                                value_3 +=
                                    weights(type,384,channel)*product;
                            }
                            {
                                const Scalar product = ((x_4*x_11)*x_8);
                                value_3 +=
                                    weights(type,385,channel)*product;
                            }
                            {
                                const Scalar product = ((x_5*x_5)*x_13);
                                value_3 +=
                                    weights(type,386,channel)*product;
                            }
                            {
                                const Scalar product = ((x_5*x_5)*x_15);
                                value_3 +=
                                    weights(type,387,channel)*product;
                            }
                            {
                                const Scalar product = ((x_5*x_10)*x_6);
                                value_3 +=
                                    weights(type,388,channel)*product;
                            }
                            {
                                const Scalar product = ((x_5*x_7)*x_9);
                                value_3 +=
                                    weights(type,389,channel)*product;
                            }
                            {
                                const Scalar product = ((x_5*x_7)*x_11);
                                value_3 +=
                                    weights(type,390,channel)*product;
                            }
                            {
                                const Scalar product = ((x_5*x_10)*x_8);
                                value_3 +=
                                    weights(type,391,channel)*product;
                            }
                            {
                                const Scalar product = ((x_6*x_13)*x_6);
                                value_3 +=
                                    weights(type,392,channel)*product;
                            }
                            {
                                const Scalar product = (x_6*(x_7*x_12));
                                value_3 +=
                                    weights(type,393,channel)*product;
                            }
                            {
                                const Scalar product = (x_6*(x_7*x_14));
                                value_3 +=
                                    weights(type,394,channel)*product;
                            }
                            {
                                const Scalar product = ((x_6*x_13)*x_8);
                                value_3 +=
                                    weights(type,395,channel)*product;
                            }
                            {
                                const Scalar product = (x_6*(x_8*x_15));
                                value_3 +=
                                    weights(type,396,channel)*product;
                            }
                            {
                                const Scalar product = ((x_7*x_7)*x_13);
                                value_3 +=
                                    weights(type,397,channel)*product;
                            }
                            {
                                const Scalar product = ((x_7*x_7)*x_15);
                                value_3 +=
                                    weights(type,398,channel)*product;
                            }
                            {
                                const Scalar product = ((x_7*x_12)*x_8);
                                value_3 +=
                                    weights(type,399,channel)*product;
                            }
                            {
                                const Scalar product = ((x_7*x_14)*x_8);
                                value_3 +=
                                    weights(type,400,channel)*product;
                            }
                            {
                                const Scalar product = ((x_8*x_13)*x_8);
                                value_3 +=
                                    weights(type,401,channel)*product;
                            }
                            {
                                const Scalar product = ((x_8*x_15)*x_8);
                                value_3 +=
                                    weights(type,402,channel)*product;
                            }
                            {
                                const Scalar product = ((x_9*x_9)*x_13);
                                value_3 +=
                                    weights(type,403,channel)*product;
                            }
                            {
                                const Scalar product = ((x_9*x_10)*x_12);
                                value_3 +=
                                    weights(type,404,channel)*product;
                            }
                            {
                                const Scalar product = ((x_9*x_10)*x_14);
                                value_3 +=
                                    weights(type,405,channel)*product;
                            }
                            {
                                const Scalar product = ((x_9*x_11)*x_13);
                                value_3 +=
                                    weights(type,406,channel)*product;
                            }
                            {
                                const Scalar product = ((x_9*x_11)*x_15);
                                value_3 +=
                                    weights(type,407,channel)*product;
                            }
                            {
                                const Scalar product = ((x_10*x_10)*x_13);
                                value_3 +=
                                    weights(type,408,channel)*product;
                            }
                            {
                                const Scalar product = ((x_10*x_10)*x_15);
                                value_3 +=
                                    weights(type,409,channel)*product;
                            }
                            {
                                const Scalar product = ((x_10*x_11)*x_12);
                                value_3 +=
                                    weights(type,410,channel)*product;
                            }
                            {
                                const Scalar product = ((x_10*x_11)*x_14);
                                value_3 +=
                                    weights(type,411,channel)*product;
                            }
                            {
                                const Scalar product = ((x_11*x_11)*x_13);
                                value_3 +=
                                    weights(type,412,channel)*product;
                            }
                            {
                                const Scalar product = ((x_11*x_11)*x_15);
                                value_3 +=
                                    weights(type,413,channel)*product;
                            }
                            {
                                const Scalar product = ((x_12*x_13)*x_12);
                                value_3 +=
                                    weights(type,414,channel)*product;
                            }
                            {
                                const Scalar product = ((x_12*x_14)*x_13);
                                value_3 +=
                                    weights(type,415,channel)*product;
                            }
                            {
                                const Scalar product = ((x_12*x_14)*x_15);
                                value_3 +=
                                    weights(type,416,channel)*product;
                            }
                            {
                                const Scalar product = ((x_13*x_13)*x_13);
                                value_3 +=
                                    weights(type,417,channel)*product;
                            }
                            {
                                const Scalar product = ((x_13*x_15)*x_13);
                                value_3 +=
                                    weights(type,418,channel)*product;
                            }
                            {
                                const Scalar product = ((x_13*x_14)*x_14);
                                value_3 +=
                                    weights(type,419,channel)*product;
                            }
                            {
                                const Scalar product = ((x_13*x_15)*x_15);
                                value_3 +=
                                    weights(type,420,channel)*product;
                            }
                            {
                                const Scalar product = ((x_14*x_14)*x_15);
                                value_3 +=
                                    weights(type,421,channel)*product;
                            }
                            output(node_index,0,channel) = value_0;
                            output(node_index,1,channel) = value_1;
                            output(node_index,2,channel) = value_2;
                            output(node_index,3,channel) = value_3;
        };
        if constexpr (host_execution_space<ExecutionSpace>) {
            switch (host_channel_tile()) {
            case 4:
                launch_forward_host_tiled<4>(
                    execution_space, num_nodes, channel_count,
                    node_types, input, weights, output);
                return true;
            case 8:
                launch_forward_host_tiled<8>(
                    execution_space, num_nodes, channel_count,
                    node_types, input, weights, output);
                return true;
            case 16:
                launch_forward_host_tiled<16>(
                    execution_space, num_nodes, channel_count,
                    node_types, input, weights, output);
                return true;
            default:
                break;
            }
            Kokkos::parallel_for(
                "StandardM0::forward_host",
                Kokkos::RangePolicy<ExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(
                    execution_space, 0,
                    static_cast<std::size_t>(num_nodes)*channel_count),
                KOKKOS_LAMBDA (const std::size_t owner) {
                    process_owner(
                        owner/channel_count, owner%channel_count);
                });
        } else {
            using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
            using Member = typename Policy::member_type;
            const int channel_tiles =
                (channel_count+vector_length-1)/vector_length;
            const std::int64_t work_count =
                static_cast<std::int64_t>(num_nodes)*channel_tiles;
            Kokkos::parallel_for(
                "StandardM0::forward",
                Policy(
                    execution_space, persistent_blocks,
                    team_size, vector_length),
                KOKKOS_LAMBDA (const Member& team) {
                    for (std::int64_t work_base=
                             static_cast<std::int64_t>(team.league_rank())*team_size;
                         work_base<work_count;
                         work_base+=team.league_size()*team_size) {
                        const std::int64_t work =
                            work_base+team.team_rank();
                        const bool work_active = work < work_count;
                        Kokkos::parallel_for(
                            Kokkos::ThreadVectorRange(
                                team, vector_length),
                            [=] (const int lane) {
                                if (!work_active)
                                    return;
                                const int node = static_cast<int>(
                                    work/channel_tiles);
                                const int channel =
                                    (work%channel_tiles)
                                        *vector_length+lane;
                                if (channel >= channel_count)
                                    return;
                                process_owner(node, channel);
                            });
                    }
                });
        }
        return true;
    }
}

template <typename ExecutionSpace, typename NodeTypesView,
          typename InputView, typename WeightsView,
          typename OutputAdjointView, typename InputAdjointView,
          typename InputScaleAdjointView>
bool launch_reverse(
    const ExecutionSpace& execution_space,
    int persistent_blocks,
    int num_nodes,
    NodeTypesView node_types,
    InputView input,
    WeightsView weights,
    OutputAdjointView output_adjoint,
    InputAdjointView input_adjoint,
    InputScaleAdjointView input_scale_adjoint,
    bool capture_input_scale_adjoint)
{
    if constexpr (!execution_space_supported<ExecutionSpace>) {
        return false;
    } else {
        using Scalar = typename InputView::non_const_value_type;
        static_assert(std::is_same_v<Scalar, float>
            || std::is_same_v<Scalar, double>);
        static_assert(std::is_same_v<
            typename WeightsView::non_const_value_type, Scalar>);
        static_assert(std::is_same_v<
            typename OutputAdjointView::non_const_value_type, Scalar>);
        static_assert(std::is_same_v<
            typename InputAdjointView::non_const_value_type, Scalar>);
        static_assert(std::is_same_v<
            typename InputView::array_layout, Kokkos::LayoutRight>);
        static_assert(std::is_same_v<
            typename WeightsView::array_layout, Kokkos::LayoutRight>);
        static_assert(std::is_same_v<
            typename OutputAdjointView::array_layout, Kokkos::LayoutRight>);
        static_assert(std::is_same_v<
            typename InputAdjointView::array_layout, Kokkos::LayoutRight>);
        if (num_nodes == 0)
            return true;
        if (num_nodes < 0)
            return false;
        if constexpr (!host_execution_space<ExecutionSpace>)
            if (persistent_blocks <= 0)
                return false;
        const int channel_count = input.extent_int(2);
        if (channel_count <= 0
            || node_types.extent_int(0) < num_nodes
            || input.extent_int(0) < num_nodes
            || input.extent_int(1) != input_components
            || weights.extent_int(0) <= 0
            || weights.extent_int(1) != total_terms
            || weights.extent_int(2) != channel_count
            || output_adjoint.extent_int(0) < num_nodes
            || output_adjoint.extent_int(1) != output_components
            || output_adjoint.extent_int(2) != channel_count
            || input_adjoint.extent_int(0) < num_nodes
            || input_adjoint.extent_int(1) != input_components
            || input_adjoint.extent_int(2) != channel_count)
            return false;
        const auto process_owner = KOKKOS_LAMBDA (
            const int node, const int channel) {
                            const std::size_t node_index = node;
                            const int type = node_types(node);
                            const Scalar x_0 = input(node_index,0,channel);
                            const Scalar x_1 = input(node_index,1,channel);
                            const Scalar x_2 = input(node_index,2,channel);
                            const Scalar x_3 = input(node_index,3,channel);
                            const Scalar x_4 = input(node_index,4,channel);
                            const Scalar x_5 = input(node_index,5,channel);
                            const Scalar x_6 = input(node_index,6,channel);
                            const Scalar x_7 = input(node_index,7,channel);
                            const Scalar x_8 = input(node_index,8,channel);
                            const Scalar x_9 = input(node_index,9,channel);
                            const Scalar x_10 = input(node_index,10,channel);
                            const Scalar x_11 = input(node_index,11,channel);
                            const Scalar x_12 = input(node_index,12,channel);
                            const Scalar x_13 = input(node_index,13,channel);
                            const Scalar x_14 = input(node_index,14,channel);
                            const Scalar x_15 = input(node_index,15,channel);
                            const Scalar adj_0 = output_adjoint(node_index,0,channel);
                            const Scalar adj_1 = output_adjoint(node_index,1,channel);
                            const Scalar adj_2 = output_adjoint(node_index,2,channel);
                            const Scalar adj_3 = output_adjoint(node_index,3,channel);
                            Scalar grad_0 = Scalar(0);
                            Scalar grad_1 = Scalar(0);
                            Scalar grad_2 = Scalar(0);
                            Scalar grad_3 = Scalar(0);
                            Scalar grad_4 = Scalar(0);
                            Scalar grad_5 = Scalar(0);
                            Scalar grad_6 = Scalar(0);
                            Scalar grad_7 = Scalar(0);
                            Scalar grad_8 = Scalar(0);
                            Scalar grad_9 = Scalar(0);
                            Scalar grad_10 = Scalar(0);
                            Scalar grad_11 = Scalar(0);
                            Scalar grad_12 = Scalar(0);
                            Scalar grad_13 = Scalar(0);
                            Scalar grad_14 = Scalar(0);
                            Scalar grad_15 = Scalar(0);
                            {
                                const Scalar scale =
                                    adj_0*weights(type,0,channel);
                                grad_0 += scale*1.0f;
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,1,channel);
                                grad_0 += scale*(2.0f*x_0);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,2,channel);
                                grad_1 += scale*(2.0f*x_1);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,3,channel);
                                grad_2 += scale*(2.0f*x_2);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,4,channel);
                                grad_3 += scale*(2.0f*x_3);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,5,channel);
                                grad_4 += scale*(2.0f*x_4);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,6,channel);
                                grad_5 += scale*(2.0f*x_5);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,7,channel);
                                grad_6 += scale*(2.0f*x_6);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,8,channel);
                                grad_7 += scale*(2.0f*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,9,channel);
                                grad_8 += scale*(2.0f*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,10,channel);
                                grad_9 += scale*(2.0f*x_9);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,11,channel);
                                grad_10 += scale*(2.0f*x_10);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,12,channel);
                                grad_11 += scale*(2.0f*x_11);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,13,channel);
                                grad_12 += scale*(2.0f*x_12);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,14,channel);
                                grad_13 += scale*(2.0f*x_13);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,15,channel);
                                grad_14 += scale*(2.0f*x_14);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,16,channel);
                                grad_15 += scale*(2.0f*x_15);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,17,channel);
                                grad_0 += scale*(3.0f*(x_0*x_0));
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,18,channel);
                                grad_0 += scale*(x_1*x_1);
                                grad_1 += scale*(2.0f*(x_0*x_1));
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,19,channel);
                                grad_0 += scale*(x_2*x_2);
                                grad_2 += scale*(2.0f*(x_0*x_2));
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,20,channel);
                                grad_0 += scale*(x_3*x_3);
                                grad_3 += scale*(2.0f*(x_0*x_3));
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,21,channel);
                                grad_0 += scale*(x_4*x_4);
                                grad_4 += scale*(2.0f*(x_0*x_4));
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,22,channel);
                                grad_0 += scale*(x_5*x_5);
                                grad_5 += scale*(2.0f*(x_0*x_5));
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,23,channel);
                                grad_0 += scale*(x_6*x_6);
                                grad_6 += scale*(2.0f*(x_0*x_6));
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,24,channel);
                                grad_0 += scale*(x_7*x_7);
                                grad_7 += scale*(2.0f*(x_0*x_7));
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,25,channel);
                                grad_0 += scale*(x_8*x_8);
                                grad_8 += scale*(2.0f*(x_0*x_8));
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,26,channel);
                                grad_0 += scale*(x_9*x_9);
                                grad_9 += scale*(2.0f*(x_0*x_9));
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,27,channel);
                                grad_0 += scale*(x_10*x_10);
                                grad_10 += scale*(2.0f*(x_0*x_10));
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,28,channel);
                                grad_0 += scale*(x_11*x_11);
                                grad_11 += scale*(2.0f*(x_0*x_11));
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,29,channel);
                                grad_0 += scale*(x_12*x_12);
                                grad_12 += scale*(2.0f*(x_0*x_12));
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,30,channel);
                                grad_0 += scale*(x_13*x_13);
                                grad_13 += scale*(2.0f*(x_0*x_13));
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,31,channel);
                                grad_0 += scale*(x_14*x_14);
                                grad_14 += scale*(2.0f*(x_0*x_14));
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,32,channel);
                                grad_0 += scale*(x_15*x_15);
                                grad_15 += scale*(2.0f*(x_0*x_15));
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,33,channel);
                                grad_1 += scale*(2.0f*(x_1*x_6));
                                grad_6 += scale*(x_1*x_1);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,34,channel);
                                grad_1 += scale*(2.0f*(x_1*x_8));
                                grad_8 += scale*(x_1*x_1);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,35,channel);
                                grad_1 += scale*(x_2*x_5);
                                grad_2 += scale*(x_1*x_5);
                                grad_5 += scale*(x_1*x_2);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,36,channel);
                                grad_1 += scale*(x_3*x_4);
                                grad_3 += scale*(x_1*x_4);
                                grad_4 += scale*(x_1*x_3);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,37,channel);
                                grad_1 += scale*(x_4*x_13);
                                grad_4 += scale*(x_1*x_13);
                                grad_13 += scale*(x_1*x_4);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,38,channel);
                                grad_1 += scale*(x_4*x_15);
                                grad_4 += scale*(x_1*x_15);
                                grad_15 += scale*(x_1*x_4);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,39,channel);
                                grad_1 += scale*(x_5*x_12);
                                grad_5 += scale*(x_1*x_12);
                                grad_12 += scale*(x_1*x_5);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,40,channel);
                                grad_1 += scale*(x_5*x_14);
                                grad_5 += scale*(x_1*x_14);
                                grad_14 += scale*(x_1*x_5);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,41,channel);
                                grad_1 += scale*(x_6*x_11);
                                grad_6 += scale*(x_1*x_11);
                                grad_11 += scale*(x_1*x_6);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,42,channel);
                                grad_1 += scale*(x_7*x_10);
                                grad_7 += scale*(x_1*x_10);
                                grad_10 += scale*(x_1*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,43,channel);
                                grad_1 += scale*(x_8*x_9);
                                grad_8 += scale*(x_1*x_9);
                                grad_9 += scale*(x_1*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,44,channel);
                                grad_1 += scale*(x_8*x_11);
                                grad_8 += scale*(x_1*x_11);
                                grad_11 += scale*(x_1*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,45,channel);
                                grad_2 += scale*(2.0f*(x_2*x_6));
                                grad_6 += scale*(x_2*x_2);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,46,channel);
                                grad_2 += scale*(x_3*x_7);
                                grad_3 += scale*(x_2*x_7);
                                grad_7 += scale*(x_2*x_3);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,47,channel);
                                grad_2 += scale*(x_4*x_10);
                                grad_4 += scale*(x_2*x_10);
                                grad_10 += scale*(x_2*x_4);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,48,channel);
                                grad_2 += scale*(x_5*x_11);
                                grad_5 += scale*(x_2*x_11);
                                grad_11 += scale*(x_2*x_5);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,49,channel);
                                grad_2 += scale*(x_6*x_12);
                                grad_6 += scale*(x_2*x_12);
                                grad_12 += scale*(x_2*x_6);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,50,channel);
                                grad_2 += scale*(x_7*x_13);
                                grad_7 += scale*(x_2*x_13);
                                grad_13 += scale*(x_2*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,51,channel);
                                grad_2 += scale*(x_8*x_14);
                                grad_8 += scale*(x_2*x_14);
                                grad_14 += scale*(x_2*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,52,channel);
                                grad_3 += scale*(2.0f*(x_3*x_6));
                                grad_6 += scale*(x_3*x_3);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,53,channel);
                                grad_3 += scale*(2.0f*(x_3*x_8));
                                grad_8 += scale*(x_3*x_3);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,54,channel);
                                grad_3 += scale*(x_4*x_9);
                                grad_4 += scale*(x_3*x_9);
                                grad_9 += scale*(x_3*x_4);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,55,channel);
                                grad_3 += scale*(x_4*x_11);
                                grad_4 += scale*(x_3*x_11);
                                grad_11 += scale*(x_3*x_4);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,56,channel);
                                grad_3 += scale*(x_5*x_10);
                                grad_5 += scale*(x_3*x_10);
                                grad_10 += scale*(x_3*x_5);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,57,channel);
                                grad_3 += scale*(x_6*x_13);
                                grad_6 += scale*(x_3*x_13);
                                grad_13 += scale*(x_3*x_6);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,58,channel);
                                grad_3 += scale*(x_7*x_12);
                                grad_7 += scale*(x_3*x_12);
                                grad_12 += scale*(x_3*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,59,channel);
                                grad_3 += scale*(x_7*x_14);
                                grad_7 += scale*(x_3*x_14);
                                grad_14 += scale*(x_3*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,60,channel);
                                grad_3 += scale*(x_8*x_13);
                                grad_8 += scale*(x_3*x_13);
                                grad_13 += scale*(x_3*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,61,channel);
                                grad_3 += scale*(x_8*x_15);
                                grad_8 += scale*(x_3*x_15);
                                grad_15 += scale*(x_3*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,62,channel);
                                grad_4 += scale*(2.0f*(x_4*x_6));
                                grad_6 += scale*(x_4*x_4);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,63,channel);
                                grad_4 += scale*(x_5*x_7);
                                grad_5 += scale*(x_4*x_7);
                                grad_7 += scale*(x_4*x_5);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,64,channel);
                                grad_4 += scale*(x_9*x_13);
                                grad_9 += scale*(x_4*x_13);
                                grad_13 += scale*(x_4*x_9);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,65,channel);
                                grad_4 += scale*(x_10*x_12);
                                grad_10 += scale*(x_4*x_12);
                                grad_12 += scale*(x_4*x_10);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,66,channel);
                                grad_4 += scale*(x_11*x_13);
                                grad_11 += scale*(x_4*x_13);
                                grad_13 += scale*(x_4*x_11);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,67,channel);
                                grad_4 += scale*(x_11*x_15);
                                grad_11 += scale*(x_4*x_15);
                                grad_15 += scale*(x_4*x_11);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,68,channel);
                                grad_5 += scale*(2.0f*(x_5*x_6));
                                grad_6 += scale*(x_5*x_5);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,69,channel);
                                grad_5 += scale*(2.0f*(x_5*x_8));
                                grad_8 += scale*(x_5*x_5);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,70,channel);
                                grad_5 += scale*(x_9*x_14);
                                grad_9 += scale*(x_5*x_14);
                                grad_14 += scale*(x_5*x_9);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,71,channel);
                                grad_5 += scale*(x_10*x_13);
                                grad_10 += scale*(x_5*x_13);
                                grad_13 += scale*(x_5*x_10);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,72,channel);
                                grad_5 += scale*(x_10*x_15);
                                grad_10 += scale*(x_5*x_15);
                                grad_15 += scale*(x_5*x_10);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,73,channel);
                                grad_5 += scale*(x_11*x_12);
                                grad_11 += scale*(x_5*x_12);
                                grad_12 += scale*(x_5*x_11);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,74,channel);
                                grad_5 += scale*(x_11*x_14);
                                grad_11 += scale*(x_5*x_14);
                                grad_14 += scale*(x_5*x_11);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,75,channel);
                                grad_6 += scale*(3.0f*(x_6*x_6));
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,76,channel);
                                grad_6 += scale*(x_7*x_7);
                                grad_7 += scale*(2.0f*(x_6*x_7));
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,77,channel);
                                grad_6 += scale*(x_8*x_8);
                                grad_8 += scale*(2.0f*(x_6*x_8));
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,78,channel);
                                grad_6 += scale*(x_9*x_9);
                                grad_9 += scale*(2.0f*(x_6*x_9));
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,79,channel);
                                grad_6 += scale*(x_11*x_11);
                                grad_11 += scale*(2.0f*(x_6*x_11));
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,80,channel);
                                grad_6 += scale*(x_12*x_12);
                                grad_12 += scale*(2.0f*(x_6*x_12));
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,81,channel);
                                grad_6 += scale*(x_13*x_13);
                                grad_13 += scale*(2.0f*(x_6*x_13));
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,82,channel);
                                grad_6 += scale*(x_15*x_15);
                                grad_15 += scale*(2.0f*(x_6*x_15));
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,83,channel);
                                grad_7 += scale*(2.0f*(x_7*x_8));
                                grad_8 += scale*(x_7*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,84,channel);
                                grad_7 += scale*(x_9*x_10);
                                grad_9 += scale*(x_7*x_10);
                                grad_10 += scale*(x_7*x_9);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,85,channel);
                                grad_7 += scale*(x_10*x_11);
                                grad_10 += scale*(x_7*x_11);
                                grad_11 += scale*(x_7*x_10);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,86,channel);
                                grad_7 += scale*(x_12*x_13);
                                grad_12 += scale*(x_7*x_13);
                                grad_13 += scale*(x_7*x_12);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,87,channel);
                                grad_7 += scale*(x_13*x_14);
                                grad_13 += scale*(x_7*x_14);
                                grad_14 += scale*(x_7*x_13);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,88,channel);
                                grad_7 += scale*(x_14*x_15);
                                grad_14 += scale*(x_7*x_15);
                                grad_15 += scale*(x_7*x_14);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,89,channel);
                                grad_8 += scale*(x_9*x_11);
                                grad_9 += scale*(x_8*x_11);
                                grad_11 += scale*(x_8*x_9);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,90,channel);
                                grad_8 += scale*(x_11*x_11);
                                grad_11 += scale*(2.0f*(x_8*x_11));
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,91,channel);
                                grad_8 += scale*(x_12*x_14);
                                grad_12 += scale*(x_8*x_14);
                                grad_14 += scale*(x_8*x_12);
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,92,channel);
                                grad_8 += scale*(x_13*x_13);
                                grad_13 += scale*(2.0f*(x_8*x_13));
                            }
                            {
                                const Scalar scale =
                                    adj_0*weights(type,93,channel);
                                grad_8 += scale*(x_13*x_15);
                                grad_13 += scale*(x_8*x_15);
                                grad_15 += scale*(x_8*x_13);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,94,channel);
                                grad_1 += scale*1.0f;
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,95,channel);
                                grad_0 += scale*x_1;
                                grad_1 += scale*x_0;
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,96,channel);
                                grad_1 += scale*x_6;
                                grad_6 += scale*x_1;
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,97,channel);
                                grad_1 += scale*x_8;
                                grad_8 += scale*x_1;
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,98,channel);
                                grad_2 += scale*x_5;
                                grad_5 += scale*x_2;
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,99,channel);
                                grad_3 += scale*x_4;
                                grad_4 += scale*x_3;
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,100,channel);
                                grad_4 += scale*x_13;
                                grad_13 += scale*x_4;
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,101,channel);
                                grad_4 += scale*x_15;
                                grad_15 += scale*x_4;
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,102,channel);
                                grad_5 += scale*x_12;
                                grad_12 += scale*x_5;
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,103,channel);
                                grad_5 += scale*x_14;
                                grad_14 += scale*x_5;
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,104,channel);
                                grad_6 += scale*x_11;
                                grad_11 += scale*x_6;
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,105,channel);
                                grad_7 += scale*x_10;
                                grad_10 += scale*x_7;
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,106,channel);
                                grad_8 += scale*x_9;
                                grad_9 += scale*x_8;
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,107,channel);
                                grad_8 += scale*x_11;
                                grad_11 += scale*x_8;
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,108,channel);
                                grad_0 += scale*(2.0f*(x_0*x_1));
                                grad_1 += scale*(x_0*x_0);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,109,channel);
                                grad_0 += scale*(x_1*x_6);
                                grad_1 += scale*(x_0*x_6);
                                grad_6 += scale*(x_0*x_1);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,110,channel);
                                grad_0 += scale*(x_1*x_8);
                                grad_1 += scale*(x_0*x_8);
                                grad_8 += scale*(x_0*x_1);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,111,channel);
                                grad_0 += scale*(x_2*x_5);
                                grad_2 += scale*(x_0*x_5);
                                grad_5 += scale*(x_0*x_2);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,112,channel);
                                grad_0 += scale*(x_3*x_4);
                                grad_3 += scale*(x_0*x_4);
                                grad_4 += scale*(x_0*x_3);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,113,channel);
                                grad_0 += scale*(x_4*x_13);
                                grad_4 += scale*(x_0*x_13);
                                grad_13 += scale*(x_0*x_4);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,114,channel);
                                grad_0 += scale*(x_4*x_15);
                                grad_4 += scale*(x_0*x_15);
                                grad_15 += scale*(x_0*x_4);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,115,channel);
                                grad_0 += scale*(x_5*x_12);
                                grad_5 += scale*(x_0*x_12);
                                grad_12 += scale*(x_0*x_5);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,116,channel);
                                grad_0 += scale*(x_5*x_14);
                                grad_5 += scale*(x_0*x_14);
                                grad_14 += scale*(x_0*x_5);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,117,channel);
                                grad_0 += scale*(x_6*x_11);
                                grad_6 += scale*(x_0*x_11);
                                grad_11 += scale*(x_0*x_6);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,118,channel);
                                grad_0 += scale*(x_7*x_10);
                                grad_7 += scale*(x_0*x_10);
                                grad_10 += scale*(x_0*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,119,channel);
                                grad_0 += scale*(x_8*x_9);
                                grad_8 += scale*(x_0*x_9);
                                grad_9 += scale*(x_0*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,120,channel);
                                grad_0 += scale*(x_8*x_11);
                                grad_8 += scale*(x_0*x_11);
                                grad_11 += scale*(x_0*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,121,channel);
                                grad_1 += scale*(3.0f*(x_1*x_1));
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,122,channel);
                                grad_1 += scale*(2.0f*(x_1*x_9));
                                grad_9 += scale*(x_1*x_1);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,123,channel);
                                grad_1 += scale*(2.0f*(x_1*x_11));
                                grad_11 += scale*(x_1*x_1);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,124,channel);
                                grad_1 += scale*(x_2*x_2);
                                grad_2 += scale*(2.0f*(x_1*x_2));
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,125,channel);
                                grad_1 += scale*(x_2*x_12);
                                grad_2 += scale*(x_1*x_12);
                                grad_12 += scale*(x_1*x_2);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,126,channel);
                                grad_1 += scale*(x_2*x_14);
                                grad_2 += scale*(x_1*x_14);
                                grad_14 += scale*(x_1*x_2);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,127,channel);
                                grad_1 += scale*(x_3*x_3);
                                grad_3 += scale*(2.0f*(x_1*x_3));
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,128,channel);
                                grad_1 += scale*(x_3*x_13);
                                grad_3 += scale*(x_1*x_13);
                                grad_13 += scale*(x_1*x_3);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,129,channel);
                                grad_1 += scale*(x_3*x_15);
                                grad_3 += scale*(x_1*x_15);
                                grad_15 += scale*(x_1*x_3);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,130,channel);
                                grad_1 += scale*(x_4*x_4);
                                grad_4 += scale*(2.0f*(x_1*x_4));
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,131,channel);
                                grad_1 += scale*(x_5*x_5);
                                grad_5 += scale*(2.0f*(x_1*x_5));
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,132,channel);
                                grad_1 += scale*(x_6*x_6);
                                grad_6 += scale*(2.0f*(x_1*x_6));
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,133,channel);
                                grad_1 += scale*(x_6*x_8);
                                grad_6 += scale*(x_1*x_8);
                                grad_8 += scale*(x_1*x_6);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,134,channel);
                                grad_1 += scale*(x_7*x_7);
                                grad_7 += scale*(2.0f*(x_1*x_7));
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,135,channel);
                                grad_1 += scale*(x_8*x_8);
                                grad_8 += scale*(2.0f*(x_1*x_8));
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,136,channel);
                                grad_1 += scale*(x_9*x_9);
                                grad_9 += scale*(2.0f*(x_1*x_9));
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,137,channel);
                                grad_1 += scale*(x_9*x_11);
                                grad_9 += scale*(x_1*x_11);
                                grad_11 += scale*(x_1*x_9);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,138,channel);
                                grad_1 += scale*(x_10*x_10);
                                grad_10 += scale*(2.0f*(x_1*x_10));
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,139,channel);
                                grad_1 += scale*(x_11*x_11);
                                grad_11 += scale*(2.0f*(x_1*x_11));
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,140,channel);
                                grad_1 += scale*(x_12*x_12);
                                grad_12 += scale*(2.0f*(x_1*x_12));
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,141,channel);
                                grad_1 += scale*(x_12*x_14);
                                grad_12 += scale*(x_1*x_14);
                                grad_14 += scale*(x_1*x_12);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,142,channel);
                                grad_1 += scale*(x_13*x_13);
                                grad_13 += scale*(2.0f*(x_1*x_13));
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,143,channel);
                                grad_1 += scale*(x_13*x_15);
                                grad_13 += scale*(x_1*x_15);
                                grad_15 += scale*(x_1*x_13);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,144,channel);
                                grad_1 += scale*(x_14*x_14);
                                grad_14 += scale*(2.0f*(x_1*x_14));
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,145,channel);
                                grad_1 += scale*(x_15*x_15);
                                grad_15 += scale*(2.0f*(x_1*x_15));
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,146,channel);
                                grad_2 += scale*(2.0f*(x_2*x_11));
                                grad_11 += scale*(x_2*x_2);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,147,channel);
                                grad_2 += scale*(x_3*x_10);
                                grad_3 += scale*(x_2*x_10);
                                grad_10 += scale*(x_2*x_3);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,148,channel);
                                grad_2 += scale*(x_4*x_7);
                                grad_4 += scale*(x_2*x_7);
                                grad_7 += scale*(x_2*x_4);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,149,channel);
                                grad_2 += scale*(x_5*x_6);
                                grad_5 += scale*(x_2*x_6);
                                grad_6 += scale*(x_2*x_5);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,150,channel);
                                grad_2 += scale*(x_5*x_8);
                                grad_5 += scale*(x_2*x_8);
                                grad_8 += scale*(x_2*x_5);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,151,channel);
                                grad_2 += scale*(x_9*x_14);
                                grad_9 += scale*(x_2*x_14);
                                grad_14 += scale*(x_2*x_9);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,152,channel);
                                grad_2 += scale*(x_10*x_13);
                                grad_10 += scale*(x_2*x_13);
                                grad_13 += scale*(x_2*x_10);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,153,channel);
                                grad_2 += scale*(x_10*x_15);
                                grad_10 += scale*(x_2*x_15);
                                grad_15 += scale*(x_2*x_10);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,154,channel);
                                grad_2 += scale*(x_11*x_12);
                                grad_11 += scale*(x_2*x_12);
                                grad_12 += scale*(x_2*x_11);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,155,channel);
                                grad_2 += scale*(x_11*x_14);
                                grad_11 += scale*(x_2*x_14);
                                grad_14 += scale*(x_2*x_11);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,156,channel);
                                grad_3 += scale*(2.0f*(x_3*x_9));
                                grad_9 += scale*(x_3*x_3);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,157,channel);
                                grad_3 += scale*(2.0f*(x_3*x_11));
                                grad_11 += scale*(x_3*x_3);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,158,channel);
                                grad_3 += scale*(x_4*x_6);
                                grad_4 += scale*(x_3*x_6);
                                grad_6 += scale*(x_3*x_4);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,159,channel);
                                grad_3 += scale*(x_4*x_8);
                                grad_4 += scale*(x_3*x_8);
                                grad_8 += scale*(x_3*x_4);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,160,channel);
                                grad_3 += scale*(x_5*x_7);
                                grad_5 += scale*(x_3*x_7);
                                grad_7 += scale*(x_3*x_5);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,161,channel);
                                grad_3 += scale*(x_9*x_13);
                                grad_9 += scale*(x_3*x_13);
                                grad_13 += scale*(x_3*x_9);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,162,channel);
                                grad_3 += scale*(x_9*x_15);
                                grad_9 += scale*(x_3*x_15);
                                grad_15 += scale*(x_3*x_9);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,163,channel);
                                grad_3 += scale*(x_10*x_12);
                                grad_10 += scale*(x_3*x_12);
                                grad_12 += scale*(x_3*x_10);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,164,channel);
                                grad_3 += scale*(x_10*x_14);
                                grad_10 += scale*(x_3*x_14);
                                grad_14 += scale*(x_3*x_10);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,165,channel);
                                grad_3 += scale*(x_11*x_13);
                                grad_11 += scale*(x_3*x_13);
                                grad_13 += scale*(x_3*x_11);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,166,channel);
                                grad_3 += scale*(x_11*x_15);
                                grad_11 += scale*(x_3*x_15);
                                grad_15 += scale*(x_3*x_11);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,167,channel);
                                grad_4 += scale*(2.0f*(x_4*x_9));
                                grad_9 += scale*(x_4*x_4);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,168,channel);
                                grad_4 += scale*(2.0f*(x_4*x_11));
                                grad_11 += scale*(x_4*x_4);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,169,channel);
                                grad_4 += scale*(x_5*x_10);
                                grad_5 += scale*(x_4*x_10);
                                grad_10 += scale*(x_4*x_5);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,170,channel);
                                grad_4 += scale*(x_6*x_13);
                                grad_6 += scale*(x_4*x_13);
                                grad_13 += scale*(x_4*x_6);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,171,channel);
                                grad_4 += scale*(x_6*x_15);
                                grad_6 += scale*(x_4*x_15);
                                grad_15 += scale*(x_4*x_6);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,172,channel);
                                grad_4 += scale*(x_7*x_12);
                                grad_7 += scale*(x_4*x_12);
                                grad_12 += scale*(x_4*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,173,channel);
                                grad_4 += scale*(x_7*x_14);
                                grad_7 += scale*(x_4*x_14);
                                grad_14 += scale*(x_4*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,174,channel);
                                grad_4 += scale*(x_8*x_13);
                                grad_8 += scale*(x_4*x_13);
                                grad_13 += scale*(x_4*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,175,channel);
                                grad_4 += scale*(x_8*x_15);
                                grad_8 += scale*(x_4*x_15);
                                grad_15 += scale*(x_4*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,176,channel);
                                grad_5 += scale*(2.0f*(x_5*x_9));
                                grad_9 += scale*(x_5*x_5);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,177,channel);
                                grad_5 += scale*(2.0f*(x_5*x_11));
                                grad_11 += scale*(x_5*x_5);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,178,channel);
                                grad_5 += scale*(x_6*x_12);
                                grad_6 += scale*(x_5*x_12);
                                grad_12 += scale*(x_5*x_6);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,179,channel);
                                grad_5 += scale*(x_6*x_14);
                                grad_6 += scale*(x_5*x_14);
                                grad_14 += scale*(x_5*x_6);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,180,channel);
                                grad_5 += scale*(x_7*x_13);
                                grad_7 += scale*(x_5*x_13);
                                grad_13 += scale*(x_5*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,181,channel);
                                grad_5 += scale*(x_7*x_15);
                                grad_7 += scale*(x_5*x_15);
                                grad_15 += scale*(x_5*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,182,channel);
                                grad_5 += scale*(x_8*x_12);
                                grad_8 += scale*(x_5*x_12);
                                grad_12 += scale*(x_5*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,183,channel);
                                grad_5 += scale*(x_8*x_14);
                                grad_8 += scale*(x_5*x_14);
                                grad_14 += scale*(x_5*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,184,channel);
                                grad_6 += scale*(2.0f*(x_6*x_11));
                                grad_11 += scale*(x_6*x_6);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,185,channel);
                                grad_6 += scale*(x_7*x_10);
                                grad_7 += scale*(x_6*x_10);
                                grad_10 += scale*(x_6*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,186,channel);
                                grad_6 += scale*(x_8*x_9);
                                grad_8 += scale*(x_6*x_9);
                                grad_9 += scale*(x_6*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,187,channel);
                                grad_6 += scale*(x_8*x_11);
                                grad_8 += scale*(x_6*x_11);
                                grad_11 += scale*(x_6*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,188,channel);
                                grad_7 += scale*(2.0f*(x_7*x_9));
                                grad_9 += scale*(x_7*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,189,channel);
                                grad_7 += scale*(2.0f*(x_7*x_11));
                                grad_11 += scale*(x_7*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,190,channel);
                                grad_7 += scale*(x_8*x_10);
                                grad_8 += scale*(x_7*x_10);
                                grad_10 += scale*(x_7*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,191,channel);
                                grad_8 += scale*(2.0f*(x_8*x_9));
                                grad_9 += scale*(x_8*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,192,channel);
                                grad_8 += scale*(2.0f*(x_8*x_11));
                                grad_11 += scale*(x_8*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,193,channel);
                                grad_9 += scale*(2.0f*(x_9*x_11));
                                grad_11 += scale*(x_9*x_9);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,194,channel);
                                grad_9 += scale*(x_10*x_10);
                                grad_10 += scale*(2.0f*(x_9*x_10));
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,195,channel);
                                grad_9 += scale*(x_11*x_11);
                                grad_11 += scale*(2.0f*(x_9*x_11));
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,196,channel);
                                grad_9 += scale*(x_12*x_14);
                                grad_12 += scale*(x_9*x_14);
                                grad_14 += scale*(x_9*x_12);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,197,channel);
                                grad_9 += scale*(x_13*x_13);
                                grad_13 += scale*(2.0f*(x_9*x_13));
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,198,channel);
                                grad_9 += scale*(x_13*x_15);
                                grad_13 += scale*(x_9*x_15);
                                grad_15 += scale*(x_9*x_13);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,199,channel);
                                grad_9 += scale*(x_14*x_14);
                                grad_14 += scale*(2.0f*(x_9*x_14));
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,200,channel);
                                grad_10 += scale*(2.0f*(x_10*x_11));
                                grad_11 += scale*(x_10*x_10);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,201,channel);
                                grad_10 += scale*(x_12*x_13);
                                grad_12 += scale*(x_10*x_13);
                                grad_13 += scale*(x_10*x_12);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,202,channel);
                                grad_10 += scale*(x_12*x_15);
                                grad_12 += scale*(x_10*x_15);
                                grad_15 += scale*(x_10*x_12);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,203,channel);
                                grad_10 += scale*(x_13*x_14);
                                grad_13 += scale*(x_10*x_14);
                                grad_14 += scale*(x_10*x_13);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,204,channel);
                                grad_10 += scale*(x_14*x_15);
                                grad_14 += scale*(x_10*x_15);
                                grad_15 += scale*(x_10*x_14);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,205,channel);
                                grad_11 += scale*(3.0f*(x_11*x_11));
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,206,channel);
                                grad_11 += scale*(x_12*x_12);
                                grad_12 += scale*(2.0f*(x_11*x_12));
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,207,channel);
                                grad_11 += scale*(x_12*x_14);
                                grad_12 += scale*(x_11*x_14);
                                grad_14 += scale*(x_11*x_12);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,208,channel);
                                grad_11 += scale*(x_13*x_13);
                                grad_13 += scale*(2.0f*(x_11*x_13));
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,209,channel);
                                grad_11 += scale*(x_13*x_15);
                                grad_13 += scale*(x_11*x_15);
                                grad_15 += scale*(x_11*x_13);
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,210,channel);
                                grad_11 += scale*(x_14*x_14);
                                grad_14 += scale*(2.0f*(x_11*x_14));
                            }
                            {
                                const Scalar scale =
                                    adj_1*weights(type,211,channel);
                                grad_11 += scale*(x_15*x_15);
                                grad_15 += scale*(2.0f*(x_11*x_15));
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,212,channel);
                                grad_2 += scale*1.0f;
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,213,channel);
                                grad_0 += scale*x_2;
                                grad_2 += scale*x_0;
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,214,channel);
                                grad_1 += scale*x_5;
                                grad_5 += scale*x_1;
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,215,channel);
                                grad_2 += scale*x_6;
                                grad_6 += scale*x_2;
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,216,channel);
                                grad_3 += scale*x_7;
                                grad_7 += scale*x_3;
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,217,channel);
                                grad_4 += scale*x_10;
                                grad_10 += scale*x_4;
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,218,channel);
                                grad_5 += scale*x_11;
                                grad_11 += scale*x_5;
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,219,channel);
                                grad_6 += scale*x_12;
                                grad_12 += scale*x_6;
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,220,channel);
                                grad_7 += scale*x_13;
                                grad_13 += scale*x_7;
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,221,channel);
                                grad_8 += scale*x_14;
                                grad_14 += scale*x_8;
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,222,channel);
                                grad_0 += scale*(2.0f*(x_0*x_2));
                                grad_2 += scale*(x_0*x_0);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,223,channel);
                                grad_0 += scale*(x_1*x_5);
                                grad_1 += scale*(x_0*x_5);
                                grad_5 += scale*(x_0*x_1);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,224,channel);
                                grad_0 += scale*(x_2*x_6);
                                grad_2 += scale*(x_0*x_6);
                                grad_6 += scale*(x_0*x_2);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,225,channel);
                                grad_0 += scale*(x_3*x_7);
                                grad_3 += scale*(x_0*x_7);
                                grad_7 += scale*(x_0*x_3);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,226,channel);
                                grad_0 += scale*(x_4*x_10);
                                grad_4 += scale*(x_0*x_10);
                                grad_10 += scale*(x_0*x_4);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,227,channel);
                                grad_0 += scale*(x_5*x_11);
                                grad_5 += scale*(x_0*x_11);
                                grad_11 += scale*(x_0*x_5);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,228,channel);
                                grad_0 += scale*(x_6*x_12);
                                grad_6 += scale*(x_0*x_12);
                                grad_12 += scale*(x_0*x_6);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,229,channel);
                                grad_0 += scale*(x_7*x_13);
                                grad_7 += scale*(x_0*x_13);
                                grad_13 += scale*(x_0*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,230,channel);
                                grad_0 += scale*(x_8*x_14);
                                grad_8 += scale*(x_0*x_14);
                                grad_14 += scale*(x_0*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,231,channel);
                                grad_1 += scale*(2.0f*(x_1*x_2));
                                grad_2 += scale*(x_1*x_1);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,232,channel);
                                grad_1 += scale*(2.0f*(x_1*x_12));
                                grad_12 += scale*(x_1*x_1);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,233,channel);
                                grad_1 += scale*(2.0f*(x_1*x_14));
                                grad_14 += scale*(x_1*x_1);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,234,channel);
                                grad_1 += scale*(x_2*x_11);
                                grad_2 += scale*(x_1*x_11);
                                grad_11 += scale*(x_1*x_2);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,235,channel);
                                grad_1 += scale*(x_3*x_10);
                                grad_3 += scale*(x_1*x_10);
                                grad_10 += scale*(x_1*x_3);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,236,channel);
                                grad_1 += scale*(x_4*x_7);
                                grad_4 += scale*(x_1*x_7);
                                grad_7 += scale*(x_1*x_4);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,237,channel);
                                grad_1 += scale*(x_5*x_6);
                                grad_5 += scale*(x_1*x_6);
                                grad_6 += scale*(x_1*x_5);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,238,channel);
                                grad_1 += scale*(x_5*x_8);
                                grad_5 += scale*(x_1*x_8);
                                grad_8 += scale*(x_1*x_5);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,239,channel);
                                grad_1 += scale*(x_9*x_14);
                                grad_9 += scale*(x_1*x_14);
                                grad_14 += scale*(x_1*x_9);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,240,channel);
                                grad_1 += scale*(x_10*x_13);
                                grad_10 += scale*(x_1*x_13);
                                grad_13 += scale*(x_1*x_10);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,241,channel);
                                grad_1 += scale*(x_10*x_15);
                                grad_10 += scale*(x_1*x_15);
                                grad_15 += scale*(x_1*x_10);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,242,channel);
                                grad_1 += scale*(x_11*x_12);
                                grad_11 += scale*(x_1*x_12);
                                grad_12 += scale*(x_1*x_11);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,243,channel);
                                grad_1 += scale*(x_11*x_14);
                                grad_11 += scale*(x_1*x_14);
                                grad_14 += scale*(x_1*x_11);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,244,channel);
                                grad_2 += scale*(3.0f*(x_2*x_2));
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,245,channel);
                                grad_2 += scale*(2.0f*(x_2*x_12));
                                grad_12 += scale*(x_2*x_2);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,246,channel);
                                grad_2 += scale*(x_3*x_3);
                                grad_3 += scale*(2.0f*(x_2*x_3));
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,247,channel);
                                grad_2 += scale*(x_3*x_13);
                                grad_3 += scale*(x_2*x_13);
                                grad_13 += scale*(x_2*x_3);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,248,channel);
                                grad_2 += scale*(x_4*x_4);
                                grad_4 += scale*(2.0f*(x_2*x_4));
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,249,channel);
                                grad_2 += scale*(x_5*x_5);
                                grad_5 += scale*(2.0f*(x_2*x_5));
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,250,channel);
                                grad_2 += scale*(x_6*x_6);
                                grad_6 += scale*(2.0f*(x_2*x_6));
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,251,channel);
                                grad_2 += scale*(x_7*x_7);
                                grad_7 += scale*(2.0f*(x_2*x_7));
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,252,channel);
                                grad_2 += scale*(x_8*x_8);
                                grad_8 += scale*(2.0f*(x_2*x_8));
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,253,channel);
                                grad_2 += scale*(x_9*x_9);
                                grad_9 += scale*(2.0f*(x_2*x_9));
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,254,channel);
                                grad_2 += scale*(x_10*x_10);
                                grad_10 += scale*(2.0f*(x_2*x_10));
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,255,channel);
                                grad_2 += scale*(x_11*x_11);
                                grad_11 += scale*(2.0f*(x_2*x_11));
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,256,channel);
                                grad_2 += scale*(x_12*x_12);
                                grad_12 += scale*(2.0f*(x_2*x_12));
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,257,channel);
                                grad_2 += scale*(x_13*x_13);
                                grad_13 += scale*(2.0f*(x_2*x_13));
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,258,channel);
                                grad_2 += scale*(x_14*x_14);
                                grad_14 += scale*(2.0f*(x_2*x_14));
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,259,channel);
                                grad_2 += scale*(x_15*x_15);
                                grad_15 += scale*(2.0f*(x_2*x_15));
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,260,channel);
                                grad_3 += scale*(2.0f*(x_3*x_12));
                                grad_12 += scale*(x_3*x_3);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,261,channel);
                                grad_3 += scale*(2.0f*(x_3*x_14));
                                grad_14 += scale*(x_3*x_3);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,262,channel);
                                grad_3 += scale*(x_4*x_5);
                                grad_4 += scale*(x_3*x_5);
                                grad_5 += scale*(x_3*x_4);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,263,channel);
                                grad_3 += scale*(x_6*x_7);
                                grad_6 += scale*(x_3*x_7);
                                grad_7 += scale*(x_3*x_6);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,264,channel);
                                grad_3 += scale*(x_7*x_8);
                                grad_7 += scale*(x_3*x_8);
                                grad_8 += scale*(x_3*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,265,channel);
                                grad_3 += scale*(x_9*x_10);
                                grad_9 += scale*(x_3*x_10);
                                grad_10 += scale*(x_3*x_9);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,266,channel);
                                grad_3 += scale*(x_10*x_11);
                                grad_10 += scale*(x_3*x_11);
                                grad_11 += scale*(x_3*x_10);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,267,channel);
                                grad_3 += scale*(x_12*x_13);
                                grad_12 += scale*(x_3*x_13);
                                grad_13 += scale*(x_3*x_12);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,268,channel);
                                grad_3 += scale*(x_13*x_14);
                                grad_13 += scale*(x_3*x_14);
                                grad_14 += scale*(x_3*x_13);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,269,channel);
                                grad_3 += scale*(x_14*x_15);
                                grad_14 += scale*(x_3*x_15);
                                grad_15 += scale*(x_3*x_14);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,270,channel);
                                grad_4 += scale*(2.0f*(x_4*x_12));
                                grad_12 += scale*(x_4*x_4);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,271,channel);
                                grad_4 += scale*(x_5*x_13);
                                grad_5 += scale*(x_4*x_13);
                                grad_13 += scale*(x_4*x_5);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,272,channel);
                                grad_4 += scale*(x_5*x_15);
                                grad_5 += scale*(x_4*x_15);
                                grad_15 += scale*(x_4*x_5);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,273,channel);
                                grad_4 += scale*(x_6*x_10);
                                grad_6 += scale*(x_4*x_10);
                                grad_10 += scale*(x_4*x_6);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,274,channel);
                                grad_4 += scale*(x_7*x_9);
                                grad_7 += scale*(x_4*x_9);
                                grad_9 += scale*(x_4*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,275,channel);
                                grad_4 += scale*(x_7*x_11);
                                grad_7 += scale*(x_4*x_11);
                                grad_11 += scale*(x_4*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,276,channel);
                                grad_5 += scale*(2.0f*(x_5*x_12));
                                grad_12 += scale*(x_5*x_5);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,277,channel);
                                grad_5 += scale*(2.0f*(x_5*x_14));
                                grad_14 += scale*(x_5*x_5);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,278,channel);
                                grad_5 += scale*(x_6*x_11);
                                grad_6 += scale*(x_5*x_11);
                                grad_11 += scale*(x_5*x_6);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,279,channel);
                                grad_5 += scale*(x_7*x_10);
                                grad_7 += scale*(x_5*x_10);
                                grad_10 += scale*(x_5*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,280,channel);
                                grad_5 += scale*(x_8*x_9);
                                grad_8 += scale*(x_5*x_9);
                                grad_9 += scale*(x_5*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,281,channel);
                                grad_5 += scale*(x_8*x_11);
                                grad_8 += scale*(x_5*x_11);
                                grad_11 += scale*(x_5*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,282,channel);
                                grad_6 += scale*(2.0f*(x_6*x_12));
                                grad_12 += scale*(x_6*x_6);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,283,channel);
                                grad_6 += scale*(x_7*x_13);
                                grad_7 += scale*(x_6*x_13);
                                grad_13 += scale*(x_6*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,284,channel);
                                grad_6 += scale*(x_8*x_14);
                                grad_8 += scale*(x_6*x_14);
                                grad_14 += scale*(x_6*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,285,channel);
                                grad_7 += scale*(2.0f*(x_7*x_12));
                                grad_12 += scale*(x_7*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,286,channel);
                                grad_7 += scale*(2.0f*(x_7*x_14));
                                grad_14 += scale*(x_7*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,287,channel);
                                grad_7 += scale*(x_8*x_13);
                                grad_8 += scale*(x_7*x_13);
                                grad_13 += scale*(x_7*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,288,channel);
                                grad_7 += scale*(x_8*x_15);
                                grad_8 += scale*(x_7*x_15);
                                grad_15 += scale*(x_7*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,289,channel);
                                grad_8 += scale*(2.0f*(x_8*x_12));
                                grad_12 += scale*(x_8*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,290,channel);
                                grad_9 += scale*(2.0f*(x_9*x_12));
                                grad_12 += scale*(x_9*x_9);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,291,channel);
                                grad_9 += scale*(x_10*x_13);
                                grad_10 += scale*(x_9*x_13);
                                grad_13 += scale*(x_9*x_10);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,292,channel);
                                grad_9 += scale*(x_11*x_14);
                                grad_11 += scale*(x_9*x_14);
                                grad_14 += scale*(x_9*x_11);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,293,channel);
                                grad_10 += scale*(2.0f*(x_10*x_12));
                                grad_12 += scale*(x_10*x_10);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,294,channel);
                                grad_10 += scale*(x_11*x_13);
                                grad_11 += scale*(x_10*x_13);
                                grad_13 += scale*(x_10*x_11);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,295,channel);
                                grad_10 += scale*(x_11*x_15);
                                grad_11 += scale*(x_10*x_15);
                                grad_15 += scale*(x_10*x_11);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,296,channel);
                                grad_11 += scale*(2.0f*(x_11*x_12));
                                grad_12 += scale*(x_11*x_11);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,297,channel);
                                grad_11 += scale*(2.0f*(x_11*x_14));
                                grad_14 += scale*(x_11*x_11);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,298,channel);
                                grad_12 += scale*(3.0f*(x_12*x_12));
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,299,channel);
                                grad_12 += scale*(x_13*x_13);
                                grad_13 += scale*(2.0f*(x_12*x_13));
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,300,channel);
                                grad_12 += scale*(x_14*x_14);
                                grad_14 += scale*(2.0f*(x_12*x_14));
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,301,channel);
                                grad_12 += scale*(x_15*x_15);
                                grad_15 += scale*(2.0f*(x_12*x_15));
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,302,channel);
                                grad_13 += scale*(2.0f*(x_13*x_14));
                                grad_14 += scale*(x_13*x_13);
                            }
                            {
                                const Scalar scale =
                                    adj_2*weights(type,303,channel);
                                grad_13 += scale*(x_14*x_15);
                                grad_14 += scale*(x_13*x_15);
                                grad_15 += scale*(x_13*x_14);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,304,channel);
                                grad_3 += scale*1.0f;
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,305,channel);
                                grad_0 += scale*x_3;
                                grad_3 += scale*x_0;
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,306,channel);
                                grad_1 += scale*x_4;
                                grad_4 += scale*x_1;
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,307,channel);
                                grad_2 += scale*x_7;
                                grad_7 += scale*x_2;
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,308,channel);
                                grad_3 += scale*x_6;
                                grad_6 += scale*x_3;
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,309,channel);
                                grad_3 += scale*x_8;
                                grad_8 += scale*x_3;
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,310,channel);
                                grad_4 += scale*x_9;
                                grad_9 += scale*x_4;
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,311,channel);
                                grad_4 += scale*x_11;
                                grad_11 += scale*x_4;
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,312,channel);
                                grad_5 += scale*x_10;
                                grad_10 += scale*x_5;
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,313,channel);
                                grad_6 += scale*x_13;
                                grad_13 += scale*x_6;
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,314,channel);
                                grad_7 += scale*x_12;
                                grad_12 += scale*x_7;
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,315,channel);
                                grad_7 += scale*x_14;
                                grad_14 += scale*x_7;
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,316,channel);
                                grad_8 += scale*x_13;
                                grad_13 += scale*x_8;
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,317,channel);
                                grad_8 += scale*x_15;
                                grad_15 += scale*x_8;
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,318,channel);
                                grad_0 += scale*(2.0f*(x_0*x_3));
                                grad_3 += scale*(x_0*x_0);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,319,channel);
                                grad_0 += scale*(x_1*x_4);
                                grad_1 += scale*(x_0*x_4);
                                grad_4 += scale*(x_0*x_1);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,320,channel);
                                grad_0 += scale*(x_2*x_7);
                                grad_2 += scale*(x_0*x_7);
                                grad_7 += scale*(x_0*x_2);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,321,channel);
                                grad_0 += scale*(x_3*x_6);
                                grad_3 += scale*(x_0*x_6);
                                grad_6 += scale*(x_0*x_3);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,322,channel);
                                grad_0 += scale*(x_3*x_8);
                                grad_3 += scale*(x_0*x_8);
                                grad_8 += scale*(x_0*x_3);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,323,channel);
                                grad_0 += scale*(x_4*x_9);
                                grad_4 += scale*(x_0*x_9);
                                grad_9 += scale*(x_0*x_4);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,324,channel);
                                grad_0 += scale*(x_4*x_11);
                                grad_4 += scale*(x_0*x_11);
                                grad_11 += scale*(x_0*x_4);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,325,channel);
                                grad_0 += scale*(x_5*x_10);
                                grad_5 += scale*(x_0*x_10);
                                grad_10 += scale*(x_0*x_5);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,326,channel);
                                grad_0 += scale*(x_6*x_13);
                                grad_6 += scale*(x_0*x_13);
                                grad_13 += scale*(x_0*x_6);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,327,channel);
                                grad_0 += scale*(x_7*x_12);
                                grad_7 += scale*(x_0*x_12);
                                grad_12 += scale*(x_0*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,328,channel);
                                grad_0 += scale*(x_7*x_14);
                                grad_7 += scale*(x_0*x_14);
                                grad_14 += scale*(x_0*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,329,channel);
                                grad_0 += scale*(x_8*x_13);
                                grad_8 += scale*(x_0*x_13);
                                grad_13 += scale*(x_0*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,330,channel);
                                grad_0 += scale*(x_8*x_15);
                                grad_8 += scale*(x_0*x_15);
                                grad_15 += scale*(x_0*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,331,channel);
                                grad_1 += scale*(2.0f*(x_1*x_3));
                                grad_3 += scale*(x_1*x_1);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,332,channel);
                                grad_1 += scale*(2.0f*(x_1*x_13));
                                grad_13 += scale*(x_1*x_1);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,333,channel);
                                grad_1 += scale*(2.0f*(x_1*x_15));
                                grad_15 += scale*(x_1*x_1);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,334,channel);
                                grad_1 += scale*(x_2*x_10);
                                grad_2 += scale*(x_1*x_10);
                                grad_10 += scale*(x_1*x_2);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,335,channel);
                                grad_1 += scale*(x_3*x_9);
                                grad_3 += scale*(x_1*x_9);
                                grad_9 += scale*(x_1*x_3);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,336,channel);
                                grad_1 += scale*(x_3*x_11);
                                grad_3 += scale*(x_1*x_11);
                                grad_11 += scale*(x_1*x_3);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,337,channel);
                                grad_1 += scale*(x_4*x_6);
                                grad_4 += scale*(x_1*x_6);
                                grad_6 += scale*(x_1*x_4);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,338,channel);
                                grad_1 += scale*(x_4*x_8);
                                grad_4 += scale*(x_1*x_8);
                                grad_8 += scale*(x_1*x_4);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,339,channel);
                                grad_1 += scale*(x_5*x_7);
                                grad_5 += scale*(x_1*x_7);
                                grad_7 += scale*(x_1*x_5);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,340,channel);
                                grad_1 += scale*(x_9*x_13);
                                grad_9 += scale*(x_1*x_13);
                                grad_13 += scale*(x_1*x_9);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,341,channel);
                                grad_1 += scale*(x_9*x_15);
                                grad_9 += scale*(x_1*x_15);
                                grad_15 += scale*(x_1*x_9);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,342,channel);
                                grad_1 += scale*(x_10*x_12);
                                grad_10 += scale*(x_1*x_12);
                                grad_12 += scale*(x_1*x_10);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,343,channel);
                                grad_1 += scale*(x_10*x_14);
                                grad_10 += scale*(x_1*x_14);
                                grad_14 += scale*(x_1*x_10);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,344,channel);
                                grad_1 += scale*(x_11*x_13);
                                grad_11 += scale*(x_1*x_13);
                                grad_13 += scale*(x_1*x_11);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,345,channel);
                                grad_1 += scale*(x_11*x_15);
                                grad_11 += scale*(x_1*x_15);
                                grad_15 += scale*(x_1*x_11);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,346,channel);
                                grad_2 += scale*(2.0f*(x_2*x_3));
                                grad_3 += scale*(x_2*x_2);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,347,channel);
                                grad_2 += scale*(2.0f*(x_2*x_13));
                                grad_13 += scale*(x_2*x_2);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,348,channel);
                                grad_2 += scale*(x_3*x_12);
                                grad_3 += scale*(x_2*x_12);
                                grad_12 += scale*(x_2*x_3);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,349,channel);
                                grad_2 += scale*(x_3*x_14);
                                grad_3 += scale*(x_2*x_14);
                                grad_14 += scale*(x_2*x_3);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,350,channel);
                                grad_2 += scale*(x_4*x_5);
                                grad_4 += scale*(x_2*x_5);
                                grad_5 += scale*(x_2*x_4);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,351,channel);
                                grad_2 += scale*(x_6*x_7);
                                grad_6 += scale*(x_2*x_7);
                                grad_7 += scale*(x_2*x_6);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,352,channel);
                                grad_2 += scale*(x_7*x_8);
                                grad_7 += scale*(x_2*x_8);
                                grad_8 += scale*(x_2*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,353,channel);
                                grad_2 += scale*(x_9*x_10);
                                grad_9 += scale*(x_2*x_10);
                                grad_10 += scale*(x_2*x_9);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,354,channel);
                                grad_2 += scale*(x_10*x_11);
                                grad_10 += scale*(x_2*x_11);
                                grad_11 += scale*(x_2*x_10);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,355,channel);
                                grad_2 += scale*(x_12*x_13);
                                grad_12 += scale*(x_2*x_13);
                                grad_13 += scale*(x_2*x_12);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,356,channel);
                                grad_2 += scale*(x_13*x_14);
                                grad_13 += scale*(x_2*x_14);
                                grad_14 += scale*(x_2*x_13);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,357,channel);
                                grad_2 += scale*(x_14*x_15);
                                grad_14 += scale*(x_2*x_15);
                                grad_15 += scale*(x_2*x_14);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,358,channel);
                                grad_3 += scale*(3.0f*(x_3*x_3));
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,359,channel);
                                grad_3 += scale*(2.0f*(x_3*x_13));
                                grad_13 += scale*(x_3*x_3);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,360,channel);
                                grad_3 += scale*(2.0f*(x_3*x_15));
                                grad_15 += scale*(x_3*x_3);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,361,channel);
                                grad_3 += scale*(x_4*x_4);
                                grad_4 += scale*(2.0f*(x_3*x_4));
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,362,channel);
                                grad_3 += scale*(x_5*x_5);
                                grad_5 += scale*(2.0f*(x_3*x_5));
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,363,channel);
                                grad_3 += scale*(x_6*x_6);
                                grad_6 += scale*(2.0f*(x_3*x_6));
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,364,channel);
                                grad_3 += scale*(x_6*x_8);
                                grad_6 += scale*(x_3*x_8);
                                grad_8 += scale*(x_3*x_6);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,365,channel);
                                grad_3 += scale*(x_7*x_7);
                                grad_7 += scale*(2.0f*(x_3*x_7));
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,366,channel);
                                grad_3 += scale*(x_8*x_8);
                                grad_8 += scale*(2.0f*(x_3*x_8));
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,367,channel);
                                grad_3 += scale*(x_9*x_9);
                                grad_9 += scale*(2.0f*(x_3*x_9));
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,368,channel);
                                grad_3 += scale*(x_9*x_11);
                                grad_9 += scale*(x_3*x_11);
                                grad_11 += scale*(x_3*x_9);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,369,channel);
                                grad_3 += scale*(x_10*x_10);
                                grad_10 += scale*(2.0f*(x_3*x_10));
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,370,channel);
                                grad_3 += scale*(x_11*x_11);
                                grad_11 += scale*(2.0f*(x_3*x_11));
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,371,channel);
                                grad_3 += scale*(x_12*x_12);
                                grad_12 += scale*(2.0f*(x_3*x_12));
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,372,channel);
                                grad_3 += scale*(x_12*x_14);
                                grad_12 += scale*(x_3*x_14);
                                grad_14 += scale*(x_3*x_12);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,373,channel);
                                grad_3 += scale*(x_13*x_13);
                                grad_13 += scale*(2.0f*(x_3*x_13));
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,374,channel);
                                grad_3 += scale*(x_13*x_15);
                                grad_13 += scale*(x_3*x_15);
                                grad_15 += scale*(x_3*x_13);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,375,channel);
                                grad_3 += scale*(x_14*x_14);
                                grad_14 += scale*(2.0f*(x_3*x_14));
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,376,channel);
                                grad_3 += scale*(x_15*x_15);
                                grad_15 += scale*(2.0f*(x_3*x_15));
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,377,channel);
                                grad_4 += scale*(2.0f*(x_4*x_13));
                                grad_13 += scale*(x_4*x_4);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,378,channel);
                                grad_4 += scale*(2.0f*(x_4*x_15));
                                grad_15 += scale*(x_4*x_4);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,379,channel);
                                grad_4 += scale*(x_5*x_12);
                                grad_5 += scale*(x_4*x_12);
                                grad_12 += scale*(x_4*x_5);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,380,channel);
                                grad_4 += scale*(x_5*x_14);
                                grad_5 += scale*(x_4*x_14);
                                grad_14 += scale*(x_4*x_5);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,381,channel);
                                grad_4 += scale*(x_6*x_9);
                                grad_6 += scale*(x_4*x_9);
                                grad_9 += scale*(x_4*x_6);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,382,channel);
                                grad_4 += scale*(x_6*x_11);
                                grad_6 += scale*(x_4*x_11);
                                grad_11 += scale*(x_4*x_6);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,383,channel);
                                grad_4 += scale*(x_7*x_10);
                                grad_7 += scale*(x_4*x_10);
                                grad_10 += scale*(x_4*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,384,channel);
                                grad_4 += scale*(x_8*x_9);
                                grad_8 += scale*(x_4*x_9);
                                grad_9 += scale*(x_4*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,385,channel);
                                grad_4 += scale*(x_8*x_11);
                                grad_8 += scale*(x_4*x_11);
                                grad_11 += scale*(x_4*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,386,channel);
                                grad_5 += scale*(2.0f*(x_5*x_13));
                                grad_13 += scale*(x_5*x_5);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,387,channel);
                                grad_5 += scale*(2.0f*(x_5*x_15));
                                grad_15 += scale*(x_5*x_5);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,388,channel);
                                grad_5 += scale*(x_6*x_10);
                                grad_6 += scale*(x_5*x_10);
                                grad_10 += scale*(x_5*x_6);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,389,channel);
                                grad_5 += scale*(x_7*x_9);
                                grad_7 += scale*(x_5*x_9);
                                grad_9 += scale*(x_5*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,390,channel);
                                grad_5 += scale*(x_7*x_11);
                                grad_7 += scale*(x_5*x_11);
                                grad_11 += scale*(x_5*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,391,channel);
                                grad_5 += scale*(x_8*x_10);
                                grad_8 += scale*(x_5*x_10);
                                grad_10 += scale*(x_5*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,392,channel);
                                grad_6 += scale*(2.0f*(x_6*x_13));
                                grad_13 += scale*(x_6*x_6);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,393,channel);
                                grad_6 += scale*(x_7*x_12);
                                grad_7 += scale*(x_6*x_12);
                                grad_12 += scale*(x_6*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,394,channel);
                                grad_6 += scale*(x_7*x_14);
                                grad_7 += scale*(x_6*x_14);
                                grad_14 += scale*(x_6*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,395,channel);
                                grad_6 += scale*(x_8*x_13);
                                grad_8 += scale*(x_6*x_13);
                                grad_13 += scale*(x_6*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,396,channel);
                                grad_6 += scale*(x_8*x_15);
                                grad_8 += scale*(x_6*x_15);
                                grad_15 += scale*(x_6*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,397,channel);
                                grad_7 += scale*(2.0f*(x_7*x_13));
                                grad_13 += scale*(x_7*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,398,channel);
                                grad_7 += scale*(2.0f*(x_7*x_15));
                                grad_15 += scale*(x_7*x_7);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,399,channel);
                                grad_7 += scale*(x_8*x_12);
                                grad_8 += scale*(x_7*x_12);
                                grad_12 += scale*(x_7*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,400,channel);
                                grad_7 += scale*(x_8*x_14);
                                grad_8 += scale*(x_7*x_14);
                                grad_14 += scale*(x_7*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,401,channel);
                                grad_8 += scale*(2.0f*(x_8*x_13));
                                grad_13 += scale*(x_8*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,402,channel);
                                grad_8 += scale*(2.0f*(x_8*x_15));
                                grad_15 += scale*(x_8*x_8);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,403,channel);
                                grad_9 += scale*(2.0f*(x_9*x_13));
                                grad_13 += scale*(x_9*x_9);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,404,channel);
                                grad_9 += scale*(x_10*x_12);
                                grad_10 += scale*(x_9*x_12);
                                grad_12 += scale*(x_9*x_10);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,405,channel);
                                grad_9 += scale*(x_10*x_14);
                                grad_10 += scale*(x_9*x_14);
                                grad_14 += scale*(x_9*x_10);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,406,channel);
                                grad_9 += scale*(x_11*x_13);
                                grad_11 += scale*(x_9*x_13);
                                grad_13 += scale*(x_9*x_11);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,407,channel);
                                grad_9 += scale*(x_11*x_15);
                                grad_11 += scale*(x_9*x_15);
                                grad_15 += scale*(x_9*x_11);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,408,channel);
                                grad_10 += scale*(2.0f*(x_10*x_13));
                                grad_13 += scale*(x_10*x_10);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,409,channel);
                                grad_10 += scale*(2.0f*(x_10*x_15));
                                grad_15 += scale*(x_10*x_10);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,410,channel);
                                grad_10 += scale*(x_11*x_12);
                                grad_11 += scale*(x_10*x_12);
                                grad_12 += scale*(x_10*x_11);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,411,channel);
                                grad_10 += scale*(x_11*x_14);
                                grad_11 += scale*(x_10*x_14);
                                grad_14 += scale*(x_10*x_11);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,412,channel);
                                grad_11 += scale*(2.0f*(x_11*x_13));
                                grad_13 += scale*(x_11*x_11);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,413,channel);
                                grad_11 += scale*(2.0f*(x_11*x_15));
                                grad_15 += scale*(x_11*x_11);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,414,channel);
                                grad_12 += scale*(2.0f*(x_12*x_13));
                                grad_13 += scale*(x_12*x_12);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,415,channel);
                                grad_12 += scale*(x_13*x_14);
                                grad_13 += scale*(x_12*x_14);
                                grad_14 += scale*(x_12*x_13);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,416,channel);
                                grad_12 += scale*(x_14*x_15);
                                grad_14 += scale*(x_12*x_15);
                                grad_15 += scale*(x_12*x_14);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,417,channel);
                                grad_13 += scale*(3.0f*(x_13*x_13));
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,418,channel);
                                grad_13 += scale*(2.0f*(x_13*x_15));
                                grad_15 += scale*(x_13*x_13);
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,419,channel);
                                grad_13 += scale*(x_14*x_14);
                                grad_14 += scale*(2.0f*(x_13*x_14));
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,420,channel);
                                grad_13 += scale*(x_15*x_15);
                                grad_15 += scale*(2.0f*(x_13*x_15));
                            }
                            {
                                const Scalar scale =
                                    adj_3*weights(type,421,channel);
                                grad_14 += scale*(2.0f*(x_14*x_15));
                                grad_15 += scale*(x_14*x_14);
                            }
                            if (capture_input_scale_adjoint) {
                                const double scale_adjoint =
                                    static_cast<double>(x_0)*grad_0
                                    +static_cast<double>(x_1)*grad_1
                                    +static_cast<double>(x_2)*grad_2
                                    +static_cast<double>(x_3)*grad_3
                                    +static_cast<double>(x_4)*grad_4
                                    +static_cast<double>(x_5)*grad_5
                                    +static_cast<double>(x_6)*grad_6
                                    +static_cast<double>(x_7)*grad_7
                                    +static_cast<double>(x_8)*grad_8
                                    +static_cast<double>(x_9)*grad_9
                                    +static_cast<double>(x_10)*grad_10
                                    +static_cast<double>(x_11)*grad_11
                                    +static_cast<double>(x_12)*grad_12
                                    +static_cast<double>(x_13)*grad_13
                                    +static_cast<double>(x_14)*grad_14
                                    +static_cast<double>(x_15)*grad_15;
                                Kokkos::atomic_add(
                                    &input_scale_adjoint(node_index),
                                    scale_adjoint);
                            }
                            input_adjoint(node_index,0,channel) = grad_0;
                            input_adjoint(node_index,1,channel) = grad_1;
                            input_adjoint(node_index,2,channel) = grad_2;
                            input_adjoint(node_index,3,channel) = grad_3;
                            input_adjoint(node_index,4,channel) = grad_4;
                            input_adjoint(node_index,5,channel) = grad_5;
                            input_adjoint(node_index,6,channel) = grad_6;
                            input_adjoint(node_index,7,channel) = grad_7;
                            input_adjoint(node_index,8,channel) = grad_8;
                            input_adjoint(node_index,9,channel) = grad_9;
                            input_adjoint(node_index,10,channel) = grad_10;
                            input_adjoint(node_index,11,channel) = grad_11;
                            input_adjoint(node_index,12,channel) = grad_12;
                            input_adjoint(node_index,13,channel) = grad_13;
                            input_adjoint(node_index,14,channel) = grad_14;
                            input_adjoint(node_index,15,channel) = grad_15;
        };
        if constexpr (host_execution_space<ExecutionSpace>) {
            switch (host_channel_tile()) {
            case 4:
                launch_reverse_host_tiled<4>(
                    execution_space, num_nodes, channel_count,
                    node_types, input, weights, output_adjoint,
                    input_adjoint, input_scale_adjoint,
                    capture_input_scale_adjoint);
                return true;
            case 8:
                launch_reverse_host_tiled<8>(
                    execution_space, num_nodes, channel_count,
                    node_types, input, weights, output_adjoint,
                    input_adjoint, input_scale_adjoint,
                    capture_input_scale_adjoint);
                return true;
            case 16:
                launch_reverse_host_tiled<16>(
                    execution_space, num_nodes, channel_count,
                    node_types, input, weights, output_adjoint,
                    input_adjoint, input_scale_adjoint,
                    capture_input_scale_adjoint);
                return true;
            default:
                break;
            }
            Kokkos::parallel_for(
                "StandardM0::reverse_host",
                Kokkos::RangePolicy<ExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(
                    execution_space, 0,
                    static_cast<std::size_t>(num_nodes)*channel_count),
                KOKKOS_LAMBDA (const std::size_t owner) {
                    process_owner(
                        owner/channel_count, owner%channel_count);
                });
        } else {
            using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
            using Member = typename Policy::member_type;
            const int channel_tiles =
                (channel_count+vector_length-1)/vector_length;
            const std::int64_t work_count =
                static_cast<std::int64_t>(num_nodes)*channel_tiles;
            Kokkos::parallel_for(
                "StandardM0::reverse",
                Policy(
                    execution_space, persistent_blocks,
                    team_size, vector_length),
                KOKKOS_LAMBDA (const Member& team) {
                    for (std::int64_t work_base=
                             static_cast<std::int64_t>(team.league_rank())*team_size;
                         work_base<work_count;
                         work_base+=team.league_size()*team_size) {
                        const std::int64_t work =
                            work_base+team.team_rank();
                        const bool work_active = work < work_count;
                        Kokkos::parallel_for(
                            Kokkos::ThreadVectorRange(
                                team, vector_length),
                            [=] (const int lane) {
                                if (!work_active)
                                    return;
                                const int node = static_cast<int>(
                                    work/channel_tiles);
                                const int channel =
                                    (work%channel_tiles)
                                        *vector_length+lane;
                                if (channel >= channel_count)
                                    return;
                                process_owner(node, channel);
                            });
                    }
                });
        }
        return true;
    }
}

template <typename ExecutionSpace, typename NodeTypesView,
          typename InputView, typename WeightsView, typename OutputView>
bool launch_scalar_forward(
    const ExecutionSpace& execution_space,
    int persistent_blocks,
    int num_nodes,
    NodeTypesView node_types,
    InputView input,
    WeightsView weights,
    OutputView output)
{
    if constexpr (!execution_space_supported<ExecutionSpace>) {
        return false;
    } else {
        using Scalar = typename InputView::non_const_value_type;
        static_assert(std::is_same_v<Scalar, float>
            || std::is_same_v<Scalar, double>);
        static_assert(std::is_same_v<
            typename WeightsView::non_const_value_type, Scalar>);
        static_assert(std::is_same_v<
            typename OutputView::non_const_value_type, Scalar>);
        static_assert(std::is_same_v<
            typename InputView::array_layout, Kokkos::LayoutRight>);
        static_assert(std::is_same_v<
            typename WeightsView::array_layout, Kokkos::LayoutRight>);
        static_assert(std::is_same_v<
            typename OutputView::array_layout, Kokkos::LayoutRight>);
        if (num_nodes == 0)
            return true;
        if (num_nodes < 0)
            return false;
        if constexpr (!host_execution_space<ExecutionSpace>)
            if (persistent_blocks <= 0)
                return false;
        const int channel_count = input.extent_int(2);
        if (channel_count <= 0
            || node_types.extent_int(0) < num_nodes
            || input.extent_int(0) < num_nodes
            || input.extent_int(1) != input_components
            || weights.extent_int(0) <= 0
            || weights.extent_int(1) != scalar_total_terms
            || weights.extent_int(2) != channel_count
            || output.extent_int(0) < num_nodes
            || output.extent_int(1) != scalar_output_components
            || output.extent_int(2) != channel_count)
            return false;

        if constexpr (host_execution_space<ExecutionSpace>) {
            switch (host_channel_tile()) {
            case 4:
                launch_forward_host_tiled<4, scalar_output_components,
                    scalar_total_terms>(
                    execution_space, num_nodes, channel_count,
                    node_types, input, weights, output);
                break;
            case 8:
                launch_forward_host_tiled<8, scalar_output_components,
                    scalar_total_terms>(
                    execution_space, num_nodes, channel_count,
                    node_types, input, weights, output);
                break;
            case 16:
                launch_forward_host_tiled<16, scalar_output_components,
                    scalar_total_terms>(
                    execution_space, num_nodes, channel_count,
                    node_types, input, weights, output);
                break;
            default:
                launch_forward_host_tiled<1, scalar_output_components,
                    scalar_total_terms>(
                    execution_space, num_nodes, channel_count,
                    node_types, input, weights, output);
                break;
            }
        } else {
            const auto process_owner = KOKKOS_LAMBDA (
                const int node, const int channel) {
                const int type = node_types(node);
                Scalar input_values[input_components][1];
                Scalar output_values[output_components][1] = {};
                for (int component=0; component<input_components; ++component)
                    input_values[component][0] = input(node,component,channel);
                accumulate_forward_terms<1>(
                    type, channel, 1, input_values, weights, output_values,
                    std::make_index_sequence<scalar_total_terms>{});
                output(node,0,channel) = output_values[0][0];
            };
            using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
            using Member = typename Policy::member_type;
            const int channel_tiles =
                (channel_count+vector_length-1)/vector_length;
            const std::int64_t work_count =
                static_cast<std::int64_t>(num_nodes)*channel_tiles;
            Kokkos::parallel_for(
                "StandardM0Scalar::forward",
                Policy(execution_space, persistent_blocks,
                    team_size, vector_length),
                KOKKOS_LAMBDA (const Member& team) {
                    for (std::int64_t work_base=
                             static_cast<std::int64_t>(team.league_rank())*team_size;
                         work_base<work_count;
                         work_base+=team.league_size()*team_size) {
                        const std::int64_t work = work_base+team.team_rank();
                        const bool work_active = work < work_count;
                        Kokkos::parallel_for(
                            Kokkos::ThreadVectorRange(team, vector_length),
                            [=] (const int lane) {
                                if (!work_active)
                                    return;
                                const int node = static_cast<int>(
                                    work/channel_tiles);
                                const int channel =
                                    (work%channel_tiles)*vector_length+lane;
                                if (channel < channel_count)
                                    process_owner(node, channel);
                            });
                    }
                });
        }
        return true;
    }
}

template <typename ExecutionSpace, typename NodeTypesView,
          typename InputView, typename WeightsView,
          typename OutputAdjointView, typename InputAdjointView,
          typename InputScaleAdjointView>
bool launch_scalar_reverse(
    const ExecutionSpace& execution_space,
    int persistent_blocks,
    int num_nodes,
    NodeTypesView node_types,
    InputView input,
    WeightsView weights,
    OutputAdjointView output_adjoint,
    InputAdjointView input_adjoint,
    InputScaleAdjointView input_scale_adjoint,
    bool capture_input_scale_adjoint)
{
    if constexpr (!execution_space_supported<ExecutionSpace>) {
        return false;
    } else {
        using Scalar = typename InputView::non_const_value_type;
        static_assert(std::is_same_v<Scalar, float>
            || std::is_same_v<Scalar, double>);
        static_assert(std::is_same_v<
            typename WeightsView::non_const_value_type, Scalar>);
        static_assert(std::is_same_v<
            typename OutputAdjointView::non_const_value_type, Scalar>);
        static_assert(std::is_same_v<
            typename InputAdjointView::non_const_value_type, Scalar>);
        static_assert(std::is_same_v<
            typename InputView::array_layout, Kokkos::LayoutRight>);
        static_assert(std::is_same_v<
            typename WeightsView::array_layout, Kokkos::LayoutRight>);
        static_assert(std::is_same_v<
            typename OutputAdjointView::array_layout, Kokkos::LayoutRight>);
        static_assert(std::is_same_v<
            typename InputAdjointView::array_layout, Kokkos::LayoutRight>);
        if (num_nodes == 0)
            return true;
        if (num_nodes < 0)
            return false;
        if constexpr (!host_execution_space<ExecutionSpace>)
            if (persistent_blocks <= 0)
                return false;
        const int channel_count = input.extent_int(2);
        if (channel_count <= 0
            || node_types.extent_int(0) < num_nodes
            || input.extent_int(0) < num_nodes
            || input.extent_int(1) != input_components
            || weights.extent_int(0) <= 0
            || weights.extent_int(1) != scalar_total_terms
            || weights.extent_int(2) != channel_count
            || output_adjoint.extent_int(0) < num_nodes
            || output_adjoint.extent_int(1) != scalar_output_components
            || output_adjoint.extent_int(2) != channel_count
            || input_adjoint.extent_int(0) < num_nodes
            || input_adjoint.extent_int(1) != input_components
            || input_adjoint.extent_int(2) != channel_count
            || (capture_input_scale_adjoint
                && input_scale_adjoint.extent_int(0) < num_nodes))
            return false;

        if constexpr (host_execution_space<ExecutionSpace>) {
            switch (host_channel_tile()) {
            case 4:
                launch_reverse_host_tiled<4, scalar_output_components,
                    scalar_total_terms>(
                    execution_space, num_nodes, channel_count,
                    node_types, input, weights, output_adjoint,
                    input_adjoint, input_scale_adjoint,
                    capture_input_scale_adjoint);
                break;
            case 8:
                launch_reverse_host_tiled<8, scalar_output_components,
                    scalar_total_terms>(
                    execution_space, num_nodes, channel_count,
                    node_types, input, weights, output_adjoint,
                    input_adjoint, input_scale_adjoint,
                    capture_input_scale_adjoint);
                break;
            case 16:
                launch_reverse_host_tiled<16, scalar_output_components,
                    scalar_total_terms>(
                    execution_space, num_nodes, channel_count,
                    node_types, input, weights, output_adjoint,
                    input_adjoint, input_scale_adjoint,
                    capture_input_scale_adjoint);
                break;
            default:
                launch_reverse_host_tiled<1, scalar_output_components,
                    scalar_total_terms>(
                    execution_space, num_nodes, channel_count,
                    node_types, input, weights, output_adjoint,
                    input_adjoint, input_scale_adjoint,
                    capture_input_scale_adjoint);
                break;
            }
        } else {
            const auto process_owner = KOKKOS_LAMBDA (
                const int node, const int channel) {
                const int type = node_types(node);
                Scalar input_values[input_components][1];
                Scalar output_adjoints[output_components][1] = {};
                Scalar input_adjoints[input_components][1] = {};
                for (int component=0; component<input_components; ++component)
                    input_values[component][0] = input(node,component,channel);
                output_adjoints[0][0] = output_adjoint(node,0,channel);
                accumulate_reverse_terms<1>(
                    type, channel, 1, input_values, output_adjoints,
                    weights, input_adjoints,
                    std::make_index_sequence<scalar_total_terms>{});
                if (capture_input_scale_adjoint) {
                    double scale_adjoint = 0.0;
                    for (int component=0; component<input_components; ++component)
                        scale_adjoint += static_cast<double>(
                            input_values[component][0])
                            *input_adjoints[component][0];
                    Kokkos::atomic_add(
                        &input_scale_adjoint(node), scale_adjoint);
                }
                for (int component=0; component<input_components; ++component)
                    input_adjoint(node,component,channel) =
                        input_adjoints[component][0];
            };
            using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
            using Member = typename Policy::member_type;
            const int channel_tiles =
                (channel_count+vector_length-1)/vector_length;
            const std::int64_t work_count =
                static_cast<std::int64_t>(num_nodes)*channel_tiles;
            Kokkos::parallel_for(
                "StandardM0Scalar::reverse",
                Policy(execution_space, persistent_blocks,
                    team_size, vector_length),
                KOKKOS_LAMBDA (const Member& team) {
                    for (std::int64_t work_base=
                             static_cast<std::int64_t>(team.league_rank())*team_size;
                         work_base<work_count;
                         work_base+=team.league_size()*team_size) {
                        const std::int64_t work = work_base+team.team_rank();
                        const bool work_active = work < work_count;
                        Kokkos::parallel_for(
                            Kokkos::ThreadVectorRange(team, vector_length),
                            [=] (const int lane) {
                                if (!work_active)
                                    return;
                                const int node = static_cast<int>(
                                    work/channel_tiles);
                                const int channel =
                                    (work%channel_tiles)*vector_length+lane;
                                if (channel < channel_count)
                                    process_owner(node, channel);
                            });
                    }
                });
        }
        return true;
    }
}

} // namespace symmetrix::standard_m0
