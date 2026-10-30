#pragma once

#include "mace_jit_operator_module_abi.h"
#include "mace_jit_r0_host_plugin_abi.h"

#include <cstddef>
#include <cstdint>

namespace symmetrix::execution {

template<class Precision>
struct R0LogicalSpline {
    std::uint32_t edge_types = 0;
    std::uint32_t intervals = 0;
    std::uint32_t functions = 0;
    double h = 0.0;
    double x0 = 0.0;
    const Precision* coefficients = nullptr;
};

template<class Precision>
struct R0LogicalArguments {
    std::uint32_t receiver_base = 0;
    std::int64_t num_nodes = 0;
    std::int64_t num_edges = 0;
    std::int64_t edge_begin = 0;
    std::int64_t workspace_edge_begin = 0;
    std::int32_t active_type_count = 0;
    std::int64_t type_to_active_count = 0;
    std::int32_t channels = 0;
    std::int32_t l_max = 0;
    std::int32_t harmonic_count = 0;
    bool coordinates_are_unit = false;
    bool apply_density_scale = false;
    bool use_precomputed_scale_adjoint = false;
    const std::int32_t* node_types = nullptr;
    const std::int32_t* num_neigh = nullptr;
    const std::int32_t* first_neigh = nullptr;
    const std::int32_t* neigh_types = nullptr;
    const std::int32_t* edge_receivers = nullptr;
    const std::int32_t* type_to_active = nullptr;
    const void* coordinates = nullptr;
    const double* radius = nullptr;
    R0LogicalSpline<Precision> radial;
    R0LogicalSpline<double> density;
    const Precision* harmonics = nullptr;
    const Precision* harmonic_gradients = nullptr;
    Precision* output = nullptr;
    Precision* output_adjoint = nullptr;
    double* density_state = nullptr;
    const double* precomputed_scale_adjoint = nullptr;
    double* directed_forces = nullptr;
    double cutoff = 0.0;
    const double* edge_weights = nullptr;
};

template<class Value>
constexpr const Value* r0_offset(
    const Value* pointer, const std::int64_t offset) noexcept
{
    return pointer == nullptr ? nullptr : pointer+offset;
}

template<class Value>
constexpr Value* r0_offset(Value* pointer, const std::int64_t offset) noexcept
{
    return pointer == nullptr ? nullptr : pointer+offset;
}

template<class Precision>
inline SymmetrixJitR0HostArgsV1 lower_r0_host_arguments(
    const R0LogicalArguments<Precision>& value) noexcept
{
    const void* coordinates = value.coordinates == nullptr ? nullptr
        : value.coordinates_are_unit
            ? static_cast<const void*>(
                static_cast<const Precision*>(value.coordinates)
                    +3*value.edge_begin)
            : static_cast<const void*>(
                static_cast<const double*>(value.coordinates)
                    +3*value.edge_begin);
    const auto radial = SymmetrixJitR0HostSplineV1{
        sizeof(SymmetrixJitR0HostSplineV1), value.radial.edge_types,
        value.radial.intervals, value.radial.functions, value.radial.h,
        value.radial.x0, value.radial.coefficients};
    const auto density = SymmetrixJitR0HostSplineV1{
        sizeof(SymmetrixJitR0HostSplineV1), value.density.edge_types,
        value.density.intervals, value.density.functions, value.density.h,
        value.density.x0, value.density.coefficients};
    return {
        sizeof(SymmetrixJitR0HostArgsV1), value.receiver_base,
        value.num_nodes, value.num_edges, value.active_type_count,
        value.channels, value.l_max, value.coordinates_are_unit ? 1 : 0,
        value.apply_density_scale ? 1 : 0,
        value.use_precomputed_scale_adjoint ? 1 : 0,
        value.node_types, value.num_neigh, value.first_neigh,
        r0_offset(value.neigh_types, value.edge_begin),
        r0_offset(value.edge_receivers, value.edge_begin), value.type_to_active,
        coordinates,
        r0_offset(value.radius, value.edge_begin), radial, density,
        r0_offset(value.harmonics, value.edge_begin*value.harmonic_count),
        r0_offset(value.harmonic_gradients,
            3*value.edge_begin*value.harmonic_count),
        value.output, value.output_adjoint, value.density_state,
        value.precomputed_scale_adjoint,
        r0_offset(value.directed_forces, 3*value.edge_begin), value.cutoff,
        r0_offset(value.edge_weights, value.edge_begin), value.edge_begin};
}

template<class Precision>
inline SymmetrixJitR0ArgsV1 lower_r0_device_arguments(
    const R0LogicalArguments<Precision>& value) noexcept
{
    const void* coordinates = value.coordinates == nullptr ? nullptr
        : value.coordinates_are_unit
            ? static_cast<const void*>(
                static_cast<const Precision*>(value.coordinates)
                    +3*value.edge_begin)
            : static_cast<const void*>(
                static_cast<const double*>(value.coordinates)
                    +3*value.edge_begin);
    const auto radial = SymmetrixJitR0SplineV1{
        sizeof(SymmetrixJitR0SplineV1), value.radial.edge_types,
        value.radial.intervals, value.radial.functions, value.radial.h,
        value.radial.x0, value.radial.coefficients};
    const auto density = SymmetrixJitR0SplineV1{
        sizeof(SymmetrixJitR0SplineV1), value.density.edge_types,
        value.density.intervals, value.density.functions, value.density.h,
        value.density.x0, value.density.coefficients};
    return {
        sizeof(SymmetrixJitR0ArgsV1), value.receiver_base, value.num_nodes,
        value.num_edges, value.active_type_count, value.channels, value.l_max,
        value.coordinates_are_unit ? 1 : 0,
        value.apply_density_scale ? 1 : 0,
        value.use_precomputed_scale_adjoint ? 1 : 0,
        value.node_types, value.num_neigh, value.first_neigh,
        r0_offset(value.neigh_types, value.edge_begin),
        r0_offset(value.edge_receivers, value.edge_begin), value.type_to_active,
        coordinates,
        r0_offset(value.radius, value.edge_begin), radial, density,
        r0_offset(value.harmonics, value.edge_begin*value.harmonic_count),
        r0_offset(value.harmonic_gradients,
            3*value.edge_begin*value.harmonic_count),
        value.output, value.output_adjoint, value.density_state,
        value.precomputed_scale_adjoint,
        r0_offset(value.directed_forces, 3*value.edge_begin), value.cutoff,
        r0_offset(value.edge_weights, value.edge_begin)};
}

} // namespace symmetrix::execution
