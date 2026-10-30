#ifndef SYMMETRIX_PARTICLE_MESH_STENCIL_HPP
#define SYMMETRIX_PARTICLE_MESH_STENCIL_HPP

// Physics-neutral ownership and construction of particle assignment stencils.

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "particle_mesh_geometry.hpp"
#include "particle_mesh_storage.hpp"

namespace symmetrix::particle_mesh {

enum class StencilPreparationPath {
    device_non_boundary,
    host_boundary_fallback,
    host_diagnostic_baseline,
};

namespace detail {

template <typename Float, typename ExecutionSpace, typename DevicePositionsView,
          typename BaseView, typename ValueView>
void prepare_device_stencil(
    const ExecutionSpace& execution_space,
    const std::int64_t nodes,
    const std::int64_t order,
    const PackedMatrix3& cell_inverse,
    const DevicePositionsView& positions,
    const double extent0,
    const double extent1,
    const double extent2,
    const BaseView& base,
    const ValueView& weights,
    const ValueView& slopes,
    const bool positions_are_grid_coordinates)
{
    Kokkos::parallel_for(
        "symmetrix_particle_mesh_stencil",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space, 0, nodes),
        KOKKOS_LAMBDA(const std::int64_t node) {
            const double extents[3] = {extent0, extent1, extent2};
            for (std::int64_t axis = 0; axis < 3; ++axis) {
                double y = positions(3 * node + axis);
                if (!positions_are_grid_coordinates) {
                    double fractional = 0.0;
                    for (std::int64_t component = 0; component < 3; ++component) {
                        fractional += positions(3 * node + component)
                            * cell_inverse(component, axis);
                    }
                    y = extents[axis] * fractional;
                }
                const double base_value = Kokkos::floor(y);
                const double theta = y - base_value;
                const double start_argument =
                    theta - (static_cast<double>(order) - 2.0) / 2.0;
                const double start = Kokkos::floor(start_argument);
                base(3 * node + axis) =
                    static_cast<std::int32_t>(base_value + start);
                for (std::int64_t offset = 0; offset < order; ++offset) {
                    const double argument = static_cast<double>(order) / 2.0 + theta
                        - (start + static_cast<double>(offset));
                    const std::int64_t index = (3 * node + axis) * order + offset;
                    weights(index) = bspline::cardinal<Float>(
                        static_cast<int>(order), static_cast<Float>(argument));
                    slopes(index) = static_cast<Float>(extents[axis])
                        * bspline::cardinal_derivative<Float>(
                            static_cast<int>(order), static_cast<Float>(argument));
                }
            }
        });
}

inline double host_grid_coordinate(
    const std::span<const double> positions,
    const std::int64_t node,
    const std::int64_t axis,
    const std::array<double, 9>& cell_inverse,
    const std::int64_t extent)
{
    // Materialize the same binary64 operation order everywhere this host
    // policy is used. The core is compiled with fast-math, so ordinary source
    // expressions may otherwise be reassociated differently after inlining.
    double fractional = 0.0;
    for (std::int64_t component = 0; component < 3; ++component) {
        volatile double product =
            positions[static_cast<std::size_t>(3 * node + component)]
            * cell_inverse[static_cast<std::size_t>(3 * component + axis)];
        volatile double sum = fractional + product;
        fractional = sum;
    }
    volatile double coordinate = static_cast<double>(extent) * fractional;
    return coordinate;
}

}  // namespace detail

[[nodiscard]] inline const char* stencil_preparation_path_token(
    const StencilPreparationPath path) noexcept
{
    switch (path) {
    case StencilPreparationPath::device_non_boundary:
        return "device_non_boundary";
    case StencilPreparationPath::host_boundary_fallback:
        return "host_boundary_fallback";
    case StencilPreparationPath::host_diagnostic_baseline:
        return "host_diagnostic_baseline";
    }
    return "unknown";
}

template <typename Float>
class ParticleStencil {
public:
    using base_view_type = Kokkos::View<std::int32_t*>;
    using value_view_type = Kokkos::View<Float*>;

    ParticleStencil() = default;
    ParticleStencil(base_view_type base, value_view_type weights, value_view_type slopes)
        : base_(std::move(base)), weights_(std::move(weights)), slopes_(std::move(slopes))
    {
    }

    [[nodiscard]] base_view_type base() const { return base_; }
    [[nodiscard]] value_view_type weights() const { return weights_; }
    [[nodiscard]] value_view_type slopes() const { return slopes_; }

    // This is the shared safe preparation entry point for particle-mesh
    // workspaces. The host and device positions must represent the same input.
    // The host doubles select the backend-independent boundary policy; only
    // inputs proved away from a guarded floor boundary enter the device kernel.
    template <typename ExecutionSpace, typename DevicePositionsView>
    [[nodiscard]] StencilPreparationPath prepare_safe(
        const ExecutionSpace& execution_space,
        const std::int64_t nodes,
        const std::int64_t order,
        const std::array<double, 9>& cell_inverse,
        const std::span<const double> host_positions,
        const DevicePositionsView& device_positions,
        const MeshShape& mesh) const
    {
        if (requires_host_boundary_fallback(
                nodes, order, cell_inverse, host_positions, mesh)) {
            prepare_host_on_execution_space(
                execution_space, nodes, order, cell_inverse, host_positions, mesh);
            return StencilPreparationPath::host_boundary_fallback;
        }
        prepare_device_unchecked_non_boundary(
            execution_space, nodes, order, cell_inverse, device_positions, mesh);
        return StencilPreparationPath::device_non_boundary;
    }

    // Resident consumers with compact host positions may upload the three grid
    // coordinates selected by the host policy. Bases, weights, and slopes are
    // still constructed on device without a particle-scale return transfer.
    template <typename ExecutionSpace, typename DevicePositionsView>
    void prepare_device_grid_coordinates(
        const ExecutionSpace& execution_space,
        const std::int64_t nodes,
        const std::int64_t order,
        const DevicePositionsView& device_grid_coordinates,
        const MeshShape& mesh) const
    {
        const double extent0 = static_cast<double>(mesh.extent(0));
        const double extent1 = static_cast<double>(mesh.extent(1));
        const double extent2 = static_cast<double>(mesh.extent(2));
        detail::prepare_device_stencil<Float>(execution_space, nodes, order,
            PackedMatrix3{}, device_grid_coordinates, extent0, extent1,
            extent2, base_, weights_, slopes_, true);
    }

    // This path exists only for private host/device parity diagnostics. Normal
    // workspaces must call prepare_safe so they cannot bypass boundary policy.
    template <typename ExecutionSpace>
    void prepare_host_diagnostic_baseline(
        const ExecutionSpace& execution_space,
        const std::int64_t nodes,
        const std::int64_t order,
        const std::array<double, 9>& cell_inverse,
        const std::span<const double> host_positions,
        const MeshShape& mesh) const
    {
        prepare_host_on_execution_space(
            execution_space, nodes, order, cell_inverse, host_positions, mesh);
    }

private:
    [[nodiscard]] static bool requires_host_boundary_fallback(
        const std::int64_t nodes,
        const std::int64_t order,
        const std::array<double, 9>& cell_inverse,
        const std::span<const double> positions,
        const MeshShape& mesh)
    {
        // Host and accelerator arithmetic can differ in the last few bits where
        // either floor selects the grid base or the order-dependent stencil
        // start. Bound each three-product dot by six rounded operations, then
        // include the mesh multiplication, two independent host/device error
        // paths, and the rounding in the distance-to-boundary comparison.
        constexpr double epsilon = std::numeric_limits<double>::epsilon();
        constexpr double dot_gamma = (6.0 * epsilon) / (1.0 - 6.0 * epsilon);
        constexpr double positive_sum_gamma =
            (3.0 * epsilon) / (1.0 - 3.0 * epsilon);
        for (std::int64_t node = 0; node < nodes; ++node) {
            for (std::int64_t axis = 0; axis < 3; ++axis) {
                double absolute_product_sum = 0.0;
                for (std::int64_t component = 0; component < 3; ++component) {
                    const double product =
                        positions[static_cast<std::size_t>(3 * node + component)]
                        * cell_inverse[static_cast<std::size_t>(3 * component + axis)];
                    absolute_product_sum += std::abs(product);
                }
                const double extent = static_cast<double>(mesh.extent(axis));
                const double y = detail::host_grid_coordinate(
                    positions, node, axis, cell_inverse, mesh.extent(axis));
                const double y_scale = std::max(1.0, std::abs(y));
                const double exact_product_sum_bound =
                    absolute_product_sum
                    / ((1.0 - epsilon) * (1.0 - positive_sum_gamma));
                const double dot_error = dot_gamma * exact_product_sum_bound;
                const double mesh_multiply_error =
                    (epsilon / (1.0 - epsilon)) * std::abs(y);
                const double one_path_error = extent * dot_error + mesh_multiply_error;
                const double boundary_guard =
                    2.0 * one_path_error + 4.0 * epsilon * y_scale;
                const double base_value = std::floor(y);
                const double theta = y - base_value;
                const double start_argument =
                    theta - (static_cast<double>(order) - 2.0) / 2.0;
                if (std::abs(y - std::round(y)) <= boundary_guard
                    || std::abs(start_argument - std::round(start_argument))
                        <= boundary_guard) {
                    return true;
                }
            }
        }
        return false;
    }

    template <typename ExecutionSpace, typename DevicePositionsView>
    void prepare_device_unchecked_non_boundary(
        const ExecutionSpace& execution_space,
        const std::int64_t nodes,
        const std::int64_t order,
        const std::array<double, 9>& cell_inverse,
        const DevicePositionsView& positions,
        const MeshShape& mesh) const
    {
        const double extent0 = static_cast<double>(mesh.extent(0));
        const double extent1 = static_cast<double>(mesh.extent(1));
        const double extent2 = static_cast<double>(mesh.extent(2));
        auto base = base_;
        auto weights = weights_;
        auto slopes = slopes_;
        detail::prepare_device_stencil<Float>(
            execution_space, nodes, order, pack_matrix(cell_inverse), positions,
            extent0, extent1, extent2, base, weights, slopes, false);
    }

    template <typename ExecutionSpace>
    void prepare_host_on_execution_space(
        const ExecutionSpace& execution_space,
        const std::int64_t nodes,
        const std::int64_t order,
        const std::array<double, 9>& cell_inverse,
        const std::span<const double> positions,
        const MeshShape& mesh) const
    {
        // Preserve the established host construction and base/weight/slope
        // copy order. All transfers use and complete on the caller's immutable
        // execution instance before these temporary host views are destroyed.
        std::vector<std::int32_t> base(
            static_cast<std::size_t>(3 * std::max<std::int64_t>(nodes, 0)), 0);
        std::vector<Float> weights(
            static_cast<std::size_t>(std::max<std::int64_t>(3 * nodes * order, 0)),
            Float(0));
        std::vector<Float> slopes(
            static_cast<std::size_t>(std::max<std::int64_t>(3 * nodes * order, 0)),
            Float(0));

        const std::array<double, 3> extents{
            static_cast<double>(mesh.extent(0)), static_cast<double>(mesh.extent(1)),
            static_cast<double>(mesh.extent(2))};
        for (std::int64_t node = 0; node < nodes; ++node) {
            for (std::int64_t axis = 0; axis < 3; ++axis) {
                const double y = detail::host_grid_coordinate(
                    positions, node, axis, cell_inverse, mesh.extent(axis));
                const double base_value = std::floor(y);
                const double theta = y - base_value;
                const double start =
                    std::floor(theta - (static_cast<double>(order) - 2.0) / 2.0);
                base[static_cast<std::size_t>(3 * node + axis)] =
                    static_cast<std::int32_t>(base_value + start);
                for (std::int64_t offset = 0; offset < order; ++offset) {
                    const double argument = static_cast<double>(order) / 2.0 + theta
                        - (start + static_cast<double>(offset));
                    weights[static_cast<std::size_t>((3 * node + axis) * order + offset)] =
                        bspline::cardinal<Float>(
                            static_cast<int>(order), static_cast<Float>(argument));
                    slopes[static_cast<std::size_t>((3 * node + axis) * order + offset)] =
                        static_cast<Float>(extents[static_cast<std::size_t>(axis)])
                        * bspline::cardinal_derivative<Float>(
                            static_cast<int>(order), static_cast<Float>(argument));
                }
            }
        }

        if (nodes == 0) {
            return;
        }
        Kokkos::View<std::int32_t*, Kokkos::HostSpace> host_base(
            "symmetrix_particle_mesh_stencil_base_host", base.size());
        Kokkos::View<Float*, Kokkos::HostSpace> host_weight(
            "symmetrix_particle_mesh_stencil_weight_host", weights.size());
        Kokkos::View<Float*, Kokkos::HostSpace> host_slope(
            "symmetrix_particle_mesh_stencil_slope_host", slopes.size());
        for (std::size_t index = 0; index < base.size(); ++index) {
            host_base(index) = base[index];
        }
        for (std::size_t index = 0; index < weights.size(); ++index) {
            host_weight(index) = weights[index];
            host_slope(index) = slopes[index];
        }
        Kokkos::deep_copy(execution_space, base_, host_base);
        Kokkos::deep_copy(execution_space, weights_, host_weight);
        Kokkos::deep_copy(execution_space, slopes_, host_slope);
        execution_space.fence("particle-mesh host stencil copy completion");
    }

    base_view_type base_;
    value_view_type weights_;
    value_view_type slopes_;
};

}  // namespace symmetrix::particle_mesh

#endif
