#include "mace_polar_particle_mesh.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <type_traits>

namespace symmetrix::mace_polar {
namespace {

constexpr double field_constant = 1.0 / 5.526349406e-3;

bool finite_double(const double value)
{
    constexpr std::uint64_t exponent_mask = UINT64_C(0x7ff0000000000000);
    return (std::bit_cast<std::uint64_t>(value) & exponent_mask) != exponent_mask;
}

template <typename Float>
KOKKOS_INLINE_FUNCTION bool finite_device(const Float value)
{
    // The library's historical -ffast-math policy permits library finiteness
    // predicates to be folded away. Inspect the IEEE-754 exponent directly so
    // resident validation remains effective on every execution space.
    if constexpr (std::is_same_v<Float, float>) {
        constexpr std::uint32_t exponent_mask = UINT32_C(0x7f800000);
        return (Kokkos::bit_cast<std::uint32_t>(value) & exponent_mask)
            != exponent_mask;
    } else {
        static_assert(std::is_same_v<Float, double>);
        constexpr std::uint64_t exponent_mask = UINT64_C(0x7ff0000000000000);
        return (Kokkos::bit_cast<std::uint64_t>(value) & exponent_mask)
            != exponent_mask;
    }
}

double determinant(const std::array<double, 9>& cell)
{
    return cell[0] * (cell[4] * cell[8] - cell[5] * cell[7])
        - cell[1] * (cell[3] * cell[8] - cell[5] * cell[6])
        + cell[2] * (cell[3] * cell[7] - cell[4] * cell[6]);
}

std::array<double, 9> inverse(const std::array<double, 9>& cell, double det)
{
    const double scale = 1.0 / det;
    return {
        (cell[4] * cell[8] - cell[5] * cell[7]) * scale,
        (cell[2] * cell[7] - cell[1] * cell[8]) * scale,
        (cell[1] * cell[5] - cell[2] * cell[4]) * scale,
        (cell[5] * cell[6] - cell[3] * cell[8]) * scale,
        (cell[0] * cell[8] - cell[2] * cell[6]) * scale,
        (cell[2] * cell[3] - cell[0] * cell[5]) * scale,
        (cell[3] * cell[7] - cell[4] * cell[6]) * scale,
        (cell[1] * cell[6] - cell[0] * cell[7]) * scale,
        (cell[0] * cell[4] - cell[1] * cell[3]) * scale};
}

double cl_sigma(const int angular_order, const double sigma, const bool receiver)
{
    if (receiver) {
        return 1.0
            / (std::pow(2.0, (angular_order + 1) / 2.0)
                * std::tgamma((angular_order + 3) / 2.0)
                * std::pow(sigma, angular_order + 3));
    }
    return 1.0
        / (std::sqrt(4.0 * std::numbers::pi / (2 * angular_order + 1))
            * std::pow(2.0, (2 * angular_order + 1) / 2.0)
            * std::tgamma((2 * angular_order + 3) / 2.0)
            * std::pow(sigma, 2 * angular_order + 3));
}

double gammainc_5_2(const double x)
{
    const double root_x = std::sqrt(x);
    const double exponential = std::exp(-x);
    const double gamma_3_2 = 0.5 * std::sqrt(std::numbers::pi)
            * std::erf(root_x)
        - root_x * exponential;
    const double gamma_5_2 = 1.5 * gamma_3_2 - x * root_x * exponential;
    return gamma_5_2 / (0.75 * std::sqrt(std::numbers::pi));
}

double self_constant(
    const int angular_order,
    const double source_sigma,
    const double receiver_sigma,
    const bool normalize_receiver)
{
    constexpr std::size_t grid_points = 10000;
    constexpr double grid_start = 0.0001;
    const double grid_stop = 10.0 * std::max(source_sigma, receiver_sigma);
    const double step = (grid_stop - grid_start)
        / static_cast<double>(grid_points - 1);
    double accumulator = 0.0;
    double previous = 0.0;
    for (std::size_t index = 0; index < grid_points; ++index) {
        const double radius = index + 1 == grid_points
            ? grid_stop
            : grid_start + static_cast<double>(index) * step;
        const double x = 0.5 * radius * radius
            / (receiver_sigma * receiver_sigma);
        const double regularized = angular_order == 0
            ? std::erf(std::sqrt(x))
                - 2.0 * std::sqrt(x / std::numbers::pi) * std::exp(-x)
            : gammainc_5_2(x);
        const double f1 = std::pow(2.0, angular_order + 0.5)
            * std::pow(receiver_sigma, 2 * angular_order + 3)
            * std::tgamma(angular_order + 1.5) * regularized
            * std::pow(radius, -(angular_order + 1));
        const double f2 = receiver_sigma * receiver_sigma
            * std::pow(radius, angular_order)
            * std::exp(-0.5 * radius * radius
                / (receiver_sigma * receiver_sigma));
        const double integrand = std::pow(radius, angular_order + 2)
            * std::exp(-0.5 * radius * radius
                / (source_sigma * source_sigma))
            * (f1 + f2);
        if (index > 0) {
            accumulator += step * (previous + integrand);
        }
        previous = integrand;
    }
    const double integral = 0.5 * accumulator;
    return cl_sigma(angular_order, source_sigma, false)
        * cl_sigma(angular_order, receiver_sigma, normalize_receiver)
        * field_constant * integral / (2 * angular_order + 1);
}

double fourier_l0_constant(const double sigma, const bool receiver)
{
    const double prefactor = 4.0 * std::numbers::pi
        * std::sqrt(std::numbers::pi / 2.0) * sigma * sigma * sigma;
    return prefactor * cl_sigma(0, sigma, receiver)
        * (0.5 / std::sqrt(std::numbers::pi));
}

double fourier_l1_coefficient(const double sigma, const bool receiver)
{
    const double sigma_squared = sigma * sigma;
    const double prefactor = 4.0 * std::numbers::pi
        * std::sqrt(std::numbers::pi / 2.0) * sigma_squared * sigma;
    return prefactor * sigma_squared * cl_sigma(1, sigma, receiver)
        * std::sqrt(3.0 / (4.0 * std::numbers::pi));
}

}  // namespace

template <typename Precision>
PolarParticleMeshForwardResult PolarParticleMeshWorkspaceT<Precision>::scalar_forward(
    const std::span<const double> positions,
    const std::span<const double> sources,
    const std::array<double, 9>& cell,
    const std::array<int, 3>& pbc,
    const double source_sigma,
    const std::span<const double> receiver_sigmas,
    const std::array<std::int64_t, 3>& mesh,
    const std::int64_t spline_order,
    const bool include_feature_self_interaction,
    const bool include_energy_self_interaction)
{
    if (layout_.density_max_l != 0 || layout_.feature_max_l != 0) {
        throw std::invalid_argument(
            "POLAR scalar particle-mesh forward requires an l=0 source and "
            "receiver layout.");
    }
    return forward(positions, sources, cell, pbc, source_sigma, receiver_sigmas,
        mesh, spline_order, include_feature_self_interaction,
        include_energy_self_interaction);
}

template <typename Precision>
PolarParticleMeshForwardResult PolarParticleMeshWorkspaceT<Precision>::forward(
    const std::span<const double> positions,
    const std::span<const double> source_multipoles,
    const std::array<double, 9>& cell,
    const std::array<int, 3>& pbc,
    const double source_sigma,
    const std::span<const double> receiver_sigmas,
    const std::array<std::int64_t, 3>& mesh,
    const std::int64_t spline_order,
    const bool include_feature_self_interaction,
    const bool include_energy_self_interaction)
{
    auto channels = forward_impl(positions, source_multipoles, 1, cell, pbc,
        source_sigma, receiver_sigmas, mesh, spline_order,
        include_feature_self_interaction, include_energy_self_interaction);
    PolarParticleMeshForwardResult result;
    result.features = std::move(channels.features);
    result.energy = channels.energies.front();
    result.nodes = channels.nodes;
    result.receiver_widths = layout_.receiver_widths;
    result.feature_dimension = channels.feature_dimension;
    return result;
}

template <typename Precision>
PolarParticleMeshChannelsForwardResult
PolarParticleMeshWorkspaceT<Precision>::forward_channels(
    const std::span<const double> positions,
    const std::span<const double> source_multipoles,
    const std::array<double, 9>& cell,
    const std::array<int, 3>& pbc,
    const double source_sigma,
    const std::span<const double> receiver_sigmas,
    const std::array<std::int64_t, 3>& mesh,
    const std::int64_t spline_order,
    const bool include_feature_self_interaction,
    const bool include_energy_self_interaction)
{
    return forward_impl(positions, source_multipoles, layout_.spin_channels,
        cell, pbc, source_sigma, receiver_sigmas, mesh, spline_order,
        include_feature_self_interaction, include_energy_self_interaction);
}

template <typename Precision>
void PolarParticleMeshWorkspaceT<Precision>::forward_channels_device(
    const std::span<const double> positions,
    const const_device_matrix& source_multipoles,
    const device_matrix& receiver_features,
    const device_vector& channel_energies,
    const std::array<double, 9>& cell,
    const std::array<int, 3>& pbc,
    const double source_sigma,
    const std::span<const double> receiver_sigmas,
    const std::array<std::int64_t, 3>& mesh,
    const std::int64_t spline_order,
    const bool include_feature_self_interaction,
    const bool include_energy_self_interaction,
    const std::int64_t active_channels,
    const bool compute_features)
{
    static_cast<void>(forward_impl(positions, {}, active_channels,
        cell, pbc, source_sigma, receiver_sigmas, mesh, spline_order,
        include_feature_self_interaction, include_energy_self_interaction,
        &source_multipoles,
        &receiver_features, &channel_energies, nullptr, nullptr,
        compute_features));
}

template <typename Precision>
void PolarParticleMeshWorkspaceT<Precision>::forward_channels_device(
    const const_device_double_matrix& positions,
    const const_device_matrix& source_multipoles,
    const device_matrix& receiver_features,
    const device_vector& channel_energies,
    const std::array<double, 9>& cell,
    const std::array<int, 3>& pbc,
    const double source_sigma,
    const std::span<const double> receiver_sigmas,
    const std::array<std::int64_t, 3>& mesh,
    const std::int64_t spline_order,
    const bool include_feature_self_interaction,
    const bool include_energy_self_interaction,
    const std::int64_t active_channels,
    const bool compute_features)
{
    static_cast<void>(forward_impl({}, {}, active_channels,
        cell, pbc, source_sigma, receiver_sigmas, mesh, spline_order,
        include_feature_self_interaction, include_energy_self_interaction,
        &source_multipoles, &receiver_features, &channel_energies, &positions,
        nullptr, compute_features));
}

template <typename Precision>
PolarParticleMeshChannelsForwardResult
PolarParticleMeshWorkspaceT<Precision>::forward_impl(
    const std::span<const double> positions,
    const std::span<const double> source_multipoles,
    const std::int64_t active_channels,
    const std::array<double, 9>& cell,
    const std::array<int, 3>& pbc,
    const double source_sigma,
    const std::span<const double> receiver_sigmas,
    const std::array<std::int64_t, 3>& mesh,
    const std::int64_t spline_order,
    const bool include_feature_self_interaction,
    const bool include_energy_self_interaction,
    const const_device_matrix* resident_sources,
    const device_matrix* resident_features,
    const device_vector* resident_energies,
    const const_device_double_matrix* resident_positions,
    Kokkos::View<complex_type*>* source_spectrum_snapshot,
    const bool compute_features)
{
    // Scope checks intentionally precede reserve: unsupported requests cannot
    // provoke a particle- or mesh-scale allocation.
    if (pbc != std::array<int, 3>{1, 1, 1}) {
        throw std::invalid_argument(
            "POLAR particle-mesh forward requires fully periodic PBC TTT.");
    }
    if (receiver_sigmas.size()
        != static_cast<std::size_t>(layout_.receiver_widths)) {
        throw std::invalid_argument(
            "POLAR particle-mesh receiver widths do not match the "
            "workspace layout.");
    }
    const auto valid_sigma = [](const double sigma) {
        return finite_double(sigma) && sigma > 0.0;
    };
    if (!valid_sigma(source_sigma)
        || !std::all_of(receiver_sigmas.begin(), receiver_sigmas.end(), valid_sigma)) {
        throw std::invalid_argument(
            "POLAR particle-mesh Gaussian widths must be positive and finite.");
    }
    const std::int64_t source_components = layout_.source_components;
    if (active_channels <= 0 || active_channels > layout_.spin_channels) {
        throw std::invalid_argument(
            "POLAR particle-mesh active channels exceed the workspace layout.");
    }
    std::int64_t nodes = 0;
    if (resident_positions == nullptr) {
        if (positions.size()
            > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
            throw std::overflow_error(
                "POLAR particle-mesh position count exceeds int64.");
        }
        if (positions.size() % 3 != 0) {
            throw std::invalid_argument(
                "POLAR particle-mesh positions must have shape [nodes, 3].");
        }
        nodes = static_cast<std::int64_t>(positions.size() / 3);
    } else {
        if (resident_positions->extent(0)
            > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
            throw std::overflow_error(
                "POLAR resident particle-mesh position rows exceed int64.");
        }
        if (resident_positions->extent(1) != 3) {
            throw std::invalid_argument(
                "POLAR resident particle-mesh positions must have shape [nodes, 3].");
        }
        nodes = static_cast<std::int64_t>(resident_positions->extent(0));
    }
    const std::int64_t coordinates = particle_mesh::detail::checked_product(
        nodes, 3, "POLAR particle-mesh position coordinates");
    if (resident_positions == nullptr
        && positions.size() != static_cast<std::size_t>(coordinates)) {
        throw std::invalid_argument(
            "POLAR particle-mesh positions must have shape [nodes, 3].");
    }
    const std::int64_t source_columns = particle_mesh::detail::checked_product(
        active_channels, source_components,
        "POLAR particle-mesh active source lanes");
    const std::int64_t source_elements = particle_mesh::detail::checked_product(
        nodes, source_columns, "POLAR particle-mesh active source elements");
    if (resident_sources == nullptr) {
        if (resident_features != nullptr || resident_energies != nullptr) {
            throw std::invalid_argument(
                "POLAR particle-mesh resident outputs require resident sources.");
        }
    } else {
        if (resident_features == nullptr || resident_energies == nullptr) {
            throw std::invalid_argument(
                "POLAR resident particle-mesh outputs are required.");
        }
        if (resident_sources->extent(0)
            > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
            throw std::overflow_error(
                "POLAR resident particle-mesh source rows exceed int64.");
        }
        if (resident_sources->extent(0) != static_cast<std::size_t>(nodes)) {
            throw std::invalid_argument(
                "POLAR resident particle-mesh source rows must match positions.");
        }
        const std::int64_t resident_feature_dimension =
            particle_mesh::detail::checked_product(layout_.receiver_widths,
                layout_.receiver_components,
                "POLAR particle-mesh resident feature dimension");
        const std::int64_t resident_feature_columns =
            particle_mesh::detail::checked_product(active_channels,
                resident_feature_dimension,
                "POLAR particle-mesh resident feature lanes");
        if (resident_sources->extent(1)
                != static_cast<std::size_t>(source_columns)
            || resident_features->extent(0) != static_cast<std::size_t>(nodes)
            || resident_features->extent(1)
                != static_cast<std::size_t>(resident_feature_columns)
            || resident_energies->extent(0)
                != static_cast<std::size_t>(active_channels)) {
            throw std::invalid_argument(
                "POLAR particle-mesh resident source/output shapes do not match "
                "the active workspace layout.");
        }
        static_cast<void>(particle_mesh::detail::checked_product(
            nodes, source_columns,
            "POLAR particle-mesh resident source elements"));
        // The caller may have produced the source view on another default-space
        // instance. Complete that compact-input handoff before this workspace's
        // private execution instance validates or consumes it.
        Kokkos::fence("POLAR particle-mesh resident source readiness");
        int invalid_sources = 0;
        const auto sources = *resident_sources;
        Kokkos::parallel_reduce(
            "POLAR particle-mesh resident source validation",
            Kokkos::RangePolicy<execution_space>(
                execution_space_, 0, sources.size()),
            KOKKOS_LAMBDA(const std::size_t element, int& invalid) {
                const std::size_t row = element / sources.extent(1);
                const std::size_t column = element % sources.extent(1);
                if (!finite_device(sources(row, column))) {
                    ++invalid;
                }
            }, Kokkos::Sum<int>(invalid_sources));
        ++resident_scalar_validation_transfers_;
        if (invalid_sources != 0) {
            throw std::invalid_argument(
                "POLAR particle-mesh requires finite resident sources.");
        }
    }
    if (resident_sources == nullptr
        && source_multipoles.size() != static_cast<std::size_t>(source_elements)) {
        throw std::invalid_argument(
            "POLAR particle-mesh source multipoles must have shape "
            "[nodes, active_channels * source_components].");
    }
    if (resident_positions != nullptr) {
        Kokkos::fence("POLAR particle-mesh resident position readiness");
        std::size_t invalid_positions = 0;
        const auto position_view = *resident_positions;
        Kokkos::parallel_reduce(
            "POLAR particle-mesh resident position validation",
            Kokkos::RangePolicy<execution_space>(
                execution_space_, 0, position_view.size()),
            KOKKOS_LAMBDA(const std::size_t element, std::size_t& invalid) {
                const std::size_t row = element / 3;
                const std::size_t column = element % 3;
                if (!finite_device(position_view(row, column))) invalid = 1;
            },
            Kokkos::Sum<std::size_t>(invalid_positions));
        ++resident_scalar_validation_transfers_;
        if (invalid_positions != 0) {
            throw std::invalid_argument(
                "POLAR particle-mesh requires finite resident positions.");
        }
    }
    if (resident_positions == nullptr
        && !std::all_of(positions.begin(), positions.end(), finite_double)) {
        throw std::invalid_argument(
            "POLAR particle-mesh requires finite positions.");
    }
    if (resident_sources == nullptr
        && !std::all_of(source_multipoles.begin(), source_multipoles.end(),
            [](const double value) {
            return finite_double(value);
        })) {
        throw std::invalid_argument(
            "POLAR particle-mesh requires finite sources/multipoles.");
    }
    const double volume = determinant(cell);
    if (!finite_double(volume) || volume <= 0.0) {
        throw std::invalid_argument(
            "POLAR particle-mesh cell must have positive finite volume.");
    }
    const auto cell_inverse = inverse(cell, volume);
    if (!std::all_of(cell_inverse.begin(), cell_inverse.end(),
            [](const double value) { return finite_double(value); })) {
        throw std::invalid_argument(
            "POLAR particle-mesh cell must have a finite inverse.");
    }

    reserve(mesh[0], mesh[1], mesh[2], nodes, spline_order);
    neutral_.prepare_transforms();
    neutral_.set_cell(cell_inverse, volume);
    neutral_.prepare_reciprocal_grid();

    if (resident_positions != nullptr) {
        const auto cartesian = *resident_positions;
        const auto grid = positions_;
        const double inverse_00 = cell_inverse[0];
        const double inverse_01 = cell_inverse[1];
        const double inverse_02 = cell_inverse[2];
        const double inverse_10 = cell_inverse[3];
        const double inverse_11 = cell_inverse[4];
        const double inverse_12 = cell_inverse[5];
        const double inverse_20 = cell_inverse[6];
        const double inverse_21 = cell_inverse[7];
        const double inverse_22 = cell_inverse[8];
        const double extent_x = static_cast<double>(mesh[0]);
        const double extent_y = static_cast<double>(mesh[1]);
        const double extent_z = static_cast<double>(mesh[2]);
        Kokkos::parallel_for("POLAR particle-mesh resident grid coordinates",
            Kokkos::RangePolicy<execution_space>(
                execution_space_, 0, coordinates),
            KOKKOS_LAMBDA(const std::int64_t element) {
                const std::int64_t node = element / 3;
                const std::int64_t axis = element % 3;
                const double x = cartesian(node, 0);
                const double y = cartesian(node, 1);
                const double z = cartesian(node, 2);
                if (axis == 0)
                    grid(element) = extent_x
                        * (x * inverse_00 + y * inverse_10 + z * inverse_20);
                else if (axis == 1)
                    grid(element) = extent_y
                        * (x * inverse_01 + y * inverse_11 + z * inverse_21);
                else
                    grid(element) = extent_z
                        * (x * inverse_02 + y * inverse_12 + z * inverse_22);
            });
    } else {
        auto host_positions = Kokkos::create_mirror_view(positions_);
        for (std::int64_t node = 0; node < nodes; ++node) {
            for (int axis = 0; axis < 3; ++axis) {
                if (resident_sources == nullptr) {
                    host_positions(3 * node + axis) =
                        positions[static_cast<std::size_t>(3 * node + axis)];
                    continue;
                }
                host_positions(3 * node + axis) =
                    particle_mesh::detail::host_grid_coordinate(positions, node,
                        axis, cell_inverse,
                        mesh[static_cast<std::size_t>(axis)]);
            }
        }
        Kokkos::deep_copy(execution_space_, positions_, host_positions);
    }
    if (resident_sources != nullptr) {
        const auto sources = *resident_sources;
        const auto weights = transform_weights_;
        const std::int64_t transform_lanes = layout_.transform_lanes;
        const std::int64_t elements = particle_mesh::detail::checked_product(
            nodes, transform_lanes,
            "POLAR particle-mesh resident transform weights");
        Kokkos::parallel_for("POLAR particle-mesh resident source lanes",
            Kokkos::RangePolicy<execution_space>(execution_space_, 0, elements),
            KOKKOS_LAMBDA(const std::int64_t element) {
                const std::int64_t node = element / transform_lanes;
                const std::int64_t lane = element % transform_lanes;
                weights(element) = lane < source_columns
                    ? sources(node, lane) : Precision(0);
            });
    } else {
        auto host_weights = Kokkos::create_mirror_view(transform_weights_);
        std::fill(host_weights.data(),
            host_weights.data() + host_weights.extent(0), Precision(0));
        for (std::int64_t node = 0; node < nodes; ++node) {
            for (std::int64_t lane = 0; lane < source_columns; ++lane) {
                host_weights(node * layout_.transform_lanes + lane) =
                    static_cast<Precision>(source_multipoles[
                        static_cast<std::size_t>(
                            node * source_columns + lane)]);
            }
        }
        Kokkos::deep_copy(execution_space_, transform_weights_, host_weights);
    }
    if (resident_sources != nullptr) {
        neutral_.prepare_stencil_grid_coordinates_device(positions_);
        resident_stencil_path_ = resident_positions == nullptr
            ? "device_host_grid_coordinates"
            : "device_resident_grid_coordinates";
    } else {
        const auto stencil_path = neutral_.prepare_stencil(positions, positions_);
        static_cast<void>(stencil_path);
    }

    const typename particle_mesh::ParticleMeshWorkspace<Precision>::weights_view
        weights(transform_weights_.data(), nodes, layout_.transform_lanes);
    try {
        neutral_.spread(weights);
    } catch (const std::exception& error) {
        throw std::runtime_error(
            std::string("POLAR particle-mesh source spread failed: ")
            + error.what());
    }
    neutral_.transform_forward();
    neutral_.deconvolve_amplitudes();

    const auto spectrum = neutral_.spectrum();
    const auto grid = neutral_.reciprocal_grid();
    auto host_sigmas = Kokkos::create_mirror_view(receiver_sigmas_);
    for (std::size_t index = 0; index < receiver_sigmas.size(); ++index) {
        host_sigmas(index) = receiver_sigmas[index];
    }
    Kokkos::deep_copy(execution_space_, receiver_sigmas_, host_sigmas);
    const auto sigmas = receiver_sigmas_;
    const std::int64_t bins = grid.bins;
    if (source_spectrum_snapshot != nullptr) {
        const std::int64_t snapshot_elements =
            particle_mesh::detail::checked_product(source_columns, bins,
                "POLAR particle-mesh forward source spectrum snapshot");
        if (source_spectrum_snapshot->extent(0)
            < static_cast<std::size_t>(snapshot_elements)) {
            throw std::logic_error(
                "POLAR particle-mesh source spectrum snapshot is undersized.");
        }
        Kokkos::View<complex_type*, typename execution_space::memory_space,
            Kokkos::MemoryTraits<Kokkos::Unmanaged>> source_spectrum(
            spectrum.data(), snapshot_elements);
        Kokkos::View<complex_type*, typename execution_space::memory_space,
            Kokkos::MemoryTraits<Kokkos::Unmanaged>> snapshot_active(
            source_spectrum_snapshot->data(), snapshot_elements);
        Kokkos::deep_copy(execution_space_, snapshot_active, source_spectrum);
    }
    const std::int64_t widths = layout_.receiver_widths;
    const std::int64_t receiver_components = layout_.receiver_components;
    const std::int64_t feature_dimension = particle_mesh::detail::checked_product(
        widths, receiver_components, "POLAR particle-mesh feature dimension");
    const double source_sigma_squared = source_sigma * source_sigma;
    const double source_l0 = fourier_l0_constant(source_sigma, false);
    const double source_l1 = layout_.density_max_l == 0
        ? 0.0
        : fourier_l1_coefficient(source_sigma, false);
    const double feature_prefactor = field_constant / volume;
    const double energy_prefactor = 0.5 * field_constant / volume;
    const double receiver_l0_prefactor = 2.0 * std::sqrt(std::numbers::pi);
    const double receiver_l1_prefactor = std::numbers::pi * std::sqrt(1.5);
    using complex_type = Kokkos::complex<Precision>;
    const std::int64_t active_feature_lanes =
        particle_mesh::detail::checked_product(active_channels,
            feature_dimension, "POLAR particle-mesh active feature lanes");
    const auto channel_energies = channel_energies_;
    for (std::int64_t channel = 0; channel < active_channels; ++channel) {
        const std::int64_t source_offset = channel * source_components;
        const auto energy = Kokkos::subview(channel_energies, channel);
        Kokkos::parallel_reduce(
            "POLAR particle-mesh channel energy",
            Kokkos::RangePolicy<execution_space>(execution_space_, 0, bins),
            KOKKOS_LAMBDA(const std::int64_t bin, double& sum) {
                    const double norm = grid.k_norm(bin);
                    const double squared = norm * norm;
                    if (squared == 0.0) {
                        return;
                    }
                    const double source_gaussian = Kokkos::exp(
                        -0.5 * source_sigma_squared * squared);
                    double wave[3];
                    particle_mesh::wave_vector(bin, grid.nx, grid.ny,
                        grid.spectrum_extent, grid.cell_inverse, wave);
                    const auto scalar =
                        spectrum.data()[source_offset * bins + bin];
                    double density_real = source_l0 * source_gaussian
                        * static_cast<double>(scalar.real());
                    double density_imaginary = source_l0 * source_gaussian
                        * static_cast<double>(scalar.imag());
                    if (source_components == 4) {
                        const double permuted_wave[3] = {
                            wave[1], wave[2], wave[0]};
                        for (int component = 0; component < 3; ++component) {
                            const auto dipole = spectrum.data()[
                                (source_offset + 1 + component) * bins + bin];
                            const double factor = source_l1 * source_gaussian
                                * permuted_wave[component];
                            density_real += factor
                                * static_cast<double>(dipole.imag());
                            density_imaginary -= factor
                                * static_cast<double>(dipole.real());
                        }
                    }
                    sum += static_cast<double>(grid.hermitian_weight(bin))
                        * energy_prefactor / squared
                        * (density_real * density_real
                            + density_imaginary * density_imaginary);
                }, energy);
    }
    if (compute_features) {
        Kokkos::parallel_for(
            "POLAR particle-mesh spectral forward",
            Kokkos::RangePolicy<execution_space>(execution_space_, 0, bins),
            KOKKOS_LAMBDA(const std::int64_t bin) {
            complex_type source_amplitudes[8] = {
                complex_type(0, 0), complex_type(0, 0),
                complex_type(0, 0), complex_type(0, 0),
                complex_type(0, 0), complex_type(0, 0),
                complex_type(0, 0), complex_type(0, 0)};
            // All source lanes must be captured before receiver lanes are
            // written: for two spin channels their ranges overlap.
            for (std::int64_t lane = 0; lane < source_columns; ++lane) {
                source_amplitudes[lane] = spectrum.data()[lane * bins + bin];
            }
            const double norm = grid.k_norm(bin);
            const double squared = norm * norm;
            if (squared == 0.0) {
                for (std::int64_t lane = 0; lane < active_feature_lanes; ++lane) {
                    spectrum.data()[lane * bins + bin] = complex_type(0, 0);
                }
                return;
            }
            const double source_gaussian =
                Kokkos::exp(-0.5 * source_sigma_squared * squared);
            double wave[3];
            particle_mesh::wave_vector(bin, grid.nx, grid.ny,
                grid.spectrum_extent, grid.cell_inverse, wave);
            for (std::int64_t channel = 0; channel < active_channels; ++channel) {
                const std::int64_t source_offset = channel * source_components;
                const std::int64_t feature_offset = channel * feature_dimension;
                double density_real = source_l0 * source_gaussian
                    * static_cast<double>(
                        source_amplitudes[source_offset].real());
                double density_imaginary = source_l0 * source_gaussian
                    * static_cast<double>(
                        source_amplitudes[source_offset].imag());
                if (source_components == 4) {
                    const double permuted_wave[3] = {wave[1], wave[2], wave[0]};
                    for (int component = 0; component < 3; ++component) {
                        const double factor = source_l1 * source_gaussian
                            * permuted_wave[component];
                        density_real += factor
                            * static_cast<double>(source_amplitudes[
                                source_offset + 1 + component].imag());
                        density_imaginary -= factor
                            * static_cast<double>(source_amplitudes[
                                source_offset + 1 + component].real());
                    }
                }
                for (std::int64_t width = 0; width < widths; ++width) {
                    const double receiver_sigma = sigmas(width);
                    const double receiver_gaussian = Kokkos::exp(
                        -0.5 * receiver_sigma * receiver_sigma * squared);
                    const double kernel = feature_prefactor
                        / squared / static_cast<double>(grid.deconvolution(bin));
                    const double receiver_l0 =
                        receiver_l0_prefactor * receiver_gaussian;
                    spectrum.data()[(feature_offset + width) * bins + bin] =
                        complex_type(static_cast<Precision>(
                                         kernel * receiver_l0 * density_real),
                            static_cast<Precision>(
                                kernel * receiver_l0 * density_imaginary));
                    if (receiver_components == 4) {
                        const double receiver_l1 = receiver_l1_prefactor
                            * receiver_sigma * receiver_gaussian;
                        const double permuted_wave[3] = {
                            wave[1], wave[2], wave[0]};
                        for (int component = 0; component < 3; ++component) {
                            const std::int64_t feature = widths
                                + width * 3 + component;
                            const double factor = kernel * receiver_l1
                                * permuted_wave[component];
                            spectrum.data()[
                                (feature_offset + feature) * bins + bin] =
                                complex_type(static_cast<Precision>(
                                                 -factor * density_imaginary),
                                    static_cast<Precision>(
                                        factor * density_real));
                        }
                    }
                }
            }
            });
        // Lanes not used by active channels must not leak a previous pass.
        if (layout_.transform_lanes > active_feature_lanes) {
            const std::int64_t unused = particle_mesh::detail::checked_product(
                layout_.transform_lanes - active_feature_lanes, bins,
                "POLAR particle-mesh unused spectrum lanes");
            Kokkos::parallel_for(
                "POLAR particle-mesh clear unused lanes",
                Kokkos::RangePolicy<execution_space>(execution_space_, 0, unused),
                KOKKOS_LAMBDA(const std::int64_t element) {
                    spectrum.data()[active_feature_lanes * bins + element] =
                        complex_type(0, 0);
                });
        }
        neutral_.transform_inverse();

        const typename particle_mesh::ParticleMeshWorkspace<
            Precision>::weight_adjoints_view gathered(
            receiver_features_.data(), nodes, layout_.transform_lanes);
        const typename particle_mesh::ParticleMeshWorkspace<
            Precision>::position_forces_view forces(
            transform_position_forces_.data(), nodes, 3);
        try {
            neutral_.gather(weights,
                typename particle_mesh::ParticleMeshWorkspace<
                    Precision>::groups_view(),
                0, gathered, forces);
        } catch (const std::exception& error) {
            throw std::runtime_error(
                std::string("POLAR particle-mesh receiver gather failed: ")
                + error.what());
        }
    }
    if (resident_sources != nullptr) {
        const auto output_features = *resident_features;
        const auto output_energies = *resident_energies;
        const auto workspace_features = receiver_features_;
        const auto workspace_energies = channel_energies_;
        const std::int64_t transform_lanes = layout_.transform_lanes;
        if (compute_features) {
            const std::int64_t output_elements =
                particle_mesh::detail::checked_product(nodes,
                    active_feature_lanes,
                    "POLAR particle-mesh resident output elements");
            Kokkos::parallel_for(
                "POLAR particle-mesh resident feature outputs",
                Kokkos::RangePolicy<execution_space>(
                    execution_space_, 0, output_elements),
                KOKKOS_LAMBDA(const std::int64_t element) {
                    const std::int64_t node = element / active_feature_lanes;
                    const std::int64_t lane = element % active_feature_lanes;
                    output_features(node, lane) =
                        workspace_features(node * transform_lanes + lane);
                });
        }
        Kokkos::parallel_for("POLAR particle-mesh resident energy outputs",
            Kokkos::RangePolicy<execution_space>(
                execution_space_, 0, active_channels),
            KOKKOS_LAMBDA(const std::int64_t channel) {
                output_energies(channel) = workspace_energies(channel);
            });
        if (compute_features && !include_feature_self_interaction) {
            for (std::int64_t width = 0; width < widths; ++width) {
                const double scalar_self = self_constant(0, source_sigma,
                    receiver_sigmas[static_cast<std::size_t>(width)], true);
                const double vector_self = receiver_components == 4
                    ? self_constant(1, source_sigma,
                        receiver_sigmas[static_cast<std::size_t>(width)], true)
                    : 0.0;
                const auto sources = *resident_sources;
                Kokkos::parallel_for(
                    "POLAR particle-mesh resident feature self correction",
                    Kokkos::RangePolicy<execution_space>(execution_space_, 0,
                        particle_mesh::detail::checked_product(nodes,
                            active_channels,
                            "POLAR particle-mesh resident self range")),
                    KOKKOS_LAMBDA(const std::int64_t element) {
                        const std::int64_t node = element / active_channels;
                        const std::int64_t channel = element % active_channels;
                        const std::int64_t source_offset =
                            channel * source_components;
                        const std::int64_t feature_offset =
                            channel * feature_dimension;
                        output_features(node, feature_offset + width) -=
                            static_cast<Precision>(scalar_self)
                            * sources(node, source_offset);
                        if (receiver_components == 4) {
                            for (int component = 0; component < 3; ++component) {
                                const std::int64_t feature = widths
                                    + width * 3 + component;
                                output_features(node,
                                    feature_offset + feature) -=
                                    static_cast<Precision>(vector_self)
                                    * sources(
                                        node, source_offset + 1 + component);
                            }
                        }
                    });
            }
        }
        if (!include_energy_self_interaction) {
            const double scalar_self = self_constant(
                0, source_sigma, source_sigma, false);
            const double vector_self = source_components == 4
                ? self_constant(1, source_sigma, source_sigma, false)
                : 0.0;
            const auto sources = *resident_sources;
            using EnergyPolicy = Kokkos::TeamPolicy<execution_space>;
            using EnergyMember = typename EnergyPolicy::member_type;
            Kokkos::parallel_for(
                "POLAR particle-mesh resident energy self correction",
                EnergyPolicy(execution_space_, active_channels, Kokkos::AUTO),
                KOKKOS_LAMBDA(const EnergyMember& team) {
                    const std::int64_t channel = team.league_rank();
                    const std::int64_t offset = channel * source_components;
                    double correction = 0.0;
                    Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(team, nodes),
                        [=](const std::int64_t node, double& sum) {
                            const double scalar = sources(node, offset);
                            sum += 0.5 * scalar_self * scalar * scalar;
                            if (source_components == 4) {
                                for (int component = 0; component < 3;
                                    ++component) {
                                    const double source = sources(
                                        node, offset + 1 + component);
                                    sum += 0.5 * vector_self * source * source;
                                }
                            }
                        }, correction);
                    Kokkos::single(Kokkos::PerTeam(team), [=]() {
                        output_energies(channel) -= correction;
                    });
                });
        }
        execution_space_.fence("POLAR particle-mesh resident completion");
        ++resident_completion_fences_;
        PolarParticleMeshChannelsForwardResult result;
        result.nodes = nodes;
        result.channels = active_channels;
        result.feature_dimension = feature_dimension;
        return result;
    }
    execution_space_.fence("POLAR particle-mesh forward outputs");

    auto host_features = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), receiver_features_);
    auto host_energies = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), channel_energies_);
    std::vector<double> feature_self_l0;
    std::vector<double> feature_self_l1;
    double energy_self_l0 = 0.0;
    double energy_self_l1 = 0.0;
    if (!include_feature_self_interaction) {
        feature_self_l0.reserve(receiver_sigmas.size());
        if (receiver_components == 4) {
            feature_self_l1.reserve(receiver_sigmas.size());
        }
        for (const double receiver_sigma : receiver_sigmas) {
            feature_self_l0.push_back(self_constant(
                0, source_sigma, receiver_sigma, true));
            if (receiver_components == 4) {
                feature_self_l1.push_back(self_constant(
                    1, source_sigma, receiver_sigma, true));
            }
        }
    }
    if (!include_energy_self_interaction) {
        energy_self_l0 = self_constant(
            0, source_sigma, source_sigma, false);
        if (source_components == 4) {
            energy_self_l1 = self_constant(
                1, source_sigma, source_sigma, false);
        }
    }
    PolarParticleMeshChannelsForwardResult result;
    result.nodes = nodes;
    result.channels = active_channels;
    result.feature_dimension = feature_dimension;
    result.features.resize(static_cast<std::size_t>(
        nodes * active_feature_lanes));
    result.energies.resize(static_cast<std::size_t>(active_channels));
    for (std::int64_t channel = 0; channel < active_channels; ++channel) {
        result.energies[static_cast<std::size_t>(channel)] =
            host_energies(channel);
    }
    for (std::int64_t node = 0; node < nodes; ++node) {
        for (std::int64_t channel = 0; channel < active_channels; ++channel) {
            const std::int64_t source_offset = channel * source_components;
            const std::int64_t feature_offset = channel * feature_dimension;
            for (std::int64_t width = 0; width < widths; ++width) {
                double scalar = static_cast<double>(host_features(
                    node * layout_.transform_lanes + feature_offset + width));
                if (!include_feature_self_interaction) {
                    scalar -= feature_self_l0[static_cast<std::size_t>(width)]
                        * source_multipoles[static_cast<std::size_t>(
                            node * source_columns + source_offset)];
                }
                result.features[static_cast<std::size_t>(
                    node * active_feature_lanes + feature_offset + width)] = scalar;
                if (receiver_components == 4) {
                    for (int component = 0; component < 3; ++component) {
                        const std::int64_t feature =
                            widths + width * 3 + component;
                        double vector = static_cast<double>(host_features(
                            node * layout_.transform_lanes
                            + feature_offset + feature));
                        if (!include_feature_self_interaction) {
                            vector -= feature_self_l1[
                                static_cast<std::size_t>(width)]
                                * source_multipoles[static_cast<std::size_t>(
                                    node * source_columns + source_offset
                                    + 1 + component)];
                        }
                        result.features[static_cast<std::size_t>(
                            node * active_feature_lanes
                            + feature_offset + feature)] = vector;
                    }
                }
            }
            if (!include_energy_self_interaction) {
                const double scalar = source_multipoles[static_cast<std::size_t>(
                    node * source_columns + source_offset)];
                result.energies[static_cast<std::size_t>(channel)] -=
                    0.5 * energy_self_l0 * scalar * scalar;
                if (source_components == 4) {
                    for (int component = 0; component < 3; ++component) {
                        const double source = source_multipoles[
                            static_cast<std::size_t>(node * source_columns
                                + source_offset + 1 + component)];
                        result.energies[static_cast<std::size_t>(channel)] -=
                            0.5 * energy_self_l1 * source * source;
                    }
                }
            }
        }
    }
    return result;
}

template <typename Precision>
PolarParticleMeshReverseResult
PolarParticleMeshWorkspaceT<Precision>::reverse_channels(
    const std::span<const double> positions,
    const std::span<const double> source_multipoles,
    const std::span<const double> feature_seeds,
    const std::span<const double> energy_seeds,
    const std::array<double, 9>& cell,
    const std::array<int, 3>& pbc,
    const double source_sigma,
    const std::span<const double> receiver_sigmas,
    const std::array<std::int64_t, 3>& mesh,
    const std::int64_t spline_order,
    const bool include_feature_self_interaction,
    const bool include_energy_self_interaction)
{
    return reverse_impl(positions, source_multipoles, feature_seeds,
        energy_seeds, cell, pbc, source_sigma, receiver_sigmas, mesh,
        spline_order, include_feature_self_interaction,
        include_energy_self_interaction, layout_.spin_channels);
}

template <typename Precision>
void PolarParticleMeshWorkspaceT<Precision>::reverse_channels_device(
    const std::span<const double> positions,
    const const_device_matrix& source_multipoles,
    const const_device_matrix& feature_seeds,
    const const_device_vector& energy_seeds,
    const device_matrix& receiver_features,
    const device_vector& channel_energies,
    const device_matrix& source_multipole_adjoints,
    const device_double_matrix& position_adjoints,
    const device_vector& cell_adjoints,
    const std::array<double, 9>& cell,
    const std::array<int, 3>& pbc,
    const double source_sigma,
    const std::span<const double> receiver_sigmas,
    const std::array<std::int64_t, 3>& mesh,
    const std::int64_t spline_order,
    const bool include_feature_self_interaction,
    const bool include_energy_self_interaction,
    const std::int64_t active_channels,
    const bool feature_seed_active,
    const bool energy_seed_active)
{
    static_cast<void>(reverse_impl(positions, {}, {}, {}, cell, pbc,
        source_sigma, receiver_sigmas, mesh, spline_order,
        include_feature_self_interaction, include_energy_self_interaction,
        active_channels, &source_multipoles, &feature_seeds, &energy_seeds,
        &receiver_features, &channel_energies, &source_multipole_adjoints,
        &position_adjoints, &cell_adjoints, nullptr, feature_seed_active,
        energy_seed_active));
}

template <typename Precision>
void PolarParticleMeshWorkspaceT<Precision>::reverse_channels_device(
    const const_device_double_matrix& positions,
    const const_device_matrix& source_multipoles,
    const const_device_matrix& feature_seeds,
    const const_device_vector& energy_seeds,
    const device_matrix& receiver_features,
    const device_vector& channel_energies,
    const device_matrix& source_multipole_adjoints,
    const device_double_matrix& position_adjoints,
    const device_vector& cell_adjoints,
    const std::array<double, 9>& cell,
    const std::array<int, 3>& pbc,
    const double source_sigma,
    const std::span<const double> receiver_sigmas,
    const std::array<std::int64_t, 3>& mesh,
    const std::int64_t spline_order,
    const bool include_feature_self_interaction,
    const bool include_energy_self_interaction,
    const std::int64_t active_channels,
    const bool feature_seed_active,
    const bool energy_seed_active)
{
    static_cast<void>(reverse_impl({}, {}, {}, {}, cell, pbc,
        source_sigma, receiver_sigmas, mesh, spline_order,
        include_feature_self_interaction, include_energy_self_interaction,
        active_channels, &source_multipoles, &feature_seeds, &energy_seeds,
        &receiver_features, &channel_energies, &source_multipole_adjoints,
        &position_adjoints, &cell_adjoints, &positions, feature_seed_active,
        energy_seed_active));
}

template <typename Precision>
PolarParticleMeshReverseResult
PolarParticleMeshWorkspaceT<Precision>::reverse_impl(
    const std::span<const double> positions,
    const std::span<const double> source_multipoles,
    const std::span<const double> feature_seeds,
    const std::span<const double> energy_seeds,
    const std::array<double, 9>& cell,
    const std::array<int, 3>& pbc,
    const double source_sigma,
    const std::span<const double> receiver_sigmas,
    const std::array<std::int64_t, 3>& mesh,
    const std::int64_t spline_order,
    const bool include_feature_self_interaction,
    const bool include_energy_self_interaction,
    const std::int64_t active_channels,
    const const_device_matrix* resident_sources,
    const const_device_matrix* resident_feature_seeds,
    const const_device_vector* resident_energy_seeds,
    const device_matrix* resident_forward_features,
    const device_vector* resident_forward_energies,
    const device_matrix* resident_source_adjoints,
    const device_double_matrix* resident_position_adjoints,
    const device_vector* resident_cell_adjoints,
    const const_device_double_matrix* resident_positions,
    const bool feature_seed_active,
    const bool energy_seed_active)
{
    if (!feature_seed_active && !energy_seed_active) {
        throw std::invalid_argument(
            "POLAR particle-mesh reverse requires an active seed branch.");
    }
    if (resident_positions == nullptr
        && (positions.size() % 3 != 0
            || positions.size() / 3
                > static_cast<std::size_t>(
                    std::numeric_limits<std::int64_t>::max()))) {
        throw std::invalid_argument(
            "POLAR particle-mesh reverse positions must have shape [nodes, 3].");
    }
    if (resident_positions != nullptr
        && (resident_positions->extent(1) != 3
            || resident_positions->extent(0)
                > static_cast<std::size_t>(
                    std::numeric_limits<std::int64_t>::max()))) {
        throw std::invalid_argument(
            "POLAR resident particle-mesh reverse positions must have shape "
            "[nodes, 3].");
    }
    const std::int64_t nodes = resident_positions == nullptr
        ? static_cast<std::int64_t>(positions.size() / 3)
        : static_cast<std::int64_t>(resident_positions->extent(0));
    const std::int64_t feature_dimension = particle_mesh::detail::checked_product(
        layout_.receiver_widths, layout_.receiver_components,
        "POLAR particle-mesh reverse feature dimension");
    if (active_channels <= 0 || active_channels > layout_.spin_channels) {
        throw std::invalid_argument(
            "POLAR particle-mesh reverse active channels exceed the workspace "
            "layout.");
    }
    const std::int64_t feature_lanes = particle_mesh::detail::checked_product(
        active_channels, feature_dimension,
        "POLAR particle-mesh reverse feature lanes");
    const std::int64_t source_lanes = particle_mesh::detail::checked_product(
        active_channels, layout_.source_components,
        "POLAR particle-mesh reverse source lanes");
    const std::int64_t feature_elements = particle_mesh::detail::checked_product(
        nodes, feature_lanes, "POLAR particle-mesh reverse feature seeds");
    const std::int64_t source_elements = particle_mesh::detail::checked_product(
        nodes, source_lanes,
        "POLAR particle-mesh reverse source elements");
    const std::int64_t position_elements = particle_mesh::detail::checked_product(
        nodes, 3, "POLAR particle-mesh reverse position elements");
    const std::int64_t channel_elements = particle_mesh::detail::checked_product(
        nodes, active_channels,
        "POLAR particle-mesh reverse node-channel elements");
    const bool resident = resident_sources != nullptr;
    const bool all_resident_views = resident && resident_feature_seeds != nullptr
        && resident_energy_seeds != nullptr
        && resident_forward_features != nullptr
        && resident_forward_energies != nullptr
        && resident_source_adjoints != nullptr
        && resident_position_adjoints != nullptr
        && resident_cell_adjoints != nullptr;
    if (resident != all_resident_views) {
        throw std::invalid_argument(
            "POLAR resident particle-mesh reverse requires all resident views.");
    }
    if (!resident) {
        if (source_multipoles.size() != static_cast<std::size_t>(source_elements)
            || feature_seeds.size() != static_cast<std::size_t>(feature_elements)
            || energy_seeds.size() != static_cast<std::size_t>(active_channels)) {
            throw std::invalid_argument(
                "POLAR particle-mesh reverse source/seed shapes do not match the "
                "workspace layout.");
        }
        if (!std::all_of(feature_seeds.begin(), feature_seeds.end(), finite_double)
            || !std::all_of(
                energy_seeds.begin(), energy_seeds.end(), finite_double)) {
            throw std::invalid_argument(
                "POLAR particle-mesh reverse requires finite seeds.");
        }
    } else {
        if (resident_sources->extent(0) != static_cast<std::size_t>(nodes)
            || resident_sources->extent(1)
                != static_cast<std::size_t>(source_lanes)
            || resident_feature_seeds->extent(0)
                != static_cast<std::size_t>(nodes)
            || resident_feature_seeds->extent(1)
                != static_cast<std::size_t>(feature_lanes)
            || resident_energy_seeds->extent(0)
                != static_cast<std::size_t>(active_channels)
            || resident_forward_features->extent(0)
                != static_cast<std::size_t>(nodes)
            || resident_forward_features->extent(1)
                != static_cast<std::size_t>(feature_lanes)
            || resident_forward_energies->extent(0)
                != static_cast<std::size_t>(active_channels)
            || resident_source_adjoints->extent(0)
                != static_cast<std::size_t>(nodes)
            || resident_source_adjoints->extent(1)
                != static_cast<std::size_t>(source_lanes)
            || resident_position_adjoints->extent(0)
                != static_cast<std::size_t>(nodes)
            || resident_position_adjoints->extent(1) != 3
            || resident_cell_adjoints->extent(0) != 9) {
            throw std::invalid_argument(
                "POLAR resident particle-mesh reverse views do not match the "
                "active workspace layout.");
        }
        Kokkos::fence("POLAR particle-mesh resident reverse seed readiness");
        std::size_t invalid_seeds = 0;
        const auto feature_seed_view = *resident_feature_seeds;
        const auto energy_seed_view = *resident_energy_seeds;
        const auto source_view = *resident_sources;
        const auto checked_add = [] (const std::size_t left,
                                     const std::size_t right) {
            if (right > std::numeric_limits<std::size_t>::max() - left) {
                throw std::length_error(
                    "POLAR resident reverse validation range overflow.");
            }
            return left + right;
        };
        const std::size_t seed_count = checked_add(
            checked_add(feature_seed_view.size(), energy_seed_view.size()),
            source_view.size());
        Kokkos::parallel_reduce(
            "POLAR particle-mesh resident reverse seed validation",
            Kokkos::RangePolicy<execution_space>(
                execution_space_, 0, seed_count),
            KOKKOS_LAMBDA(const std::size_t element, std::size_t& invalid) {
                if (element < feature_seed_view.size()) {
                    const std::size_t row =
                        element / feature_seed_view.extent(1);
                    const std::size_t column =
                        element % feature_seed_view.extent(1);
                    if (!finite_device(feature_seed_view(row, column))) {
                        invalid = 1;
                    }
                } else {
                    const std::size_t remaining =
                        element - feature_seed_view.size();
                    if (remaining < energy_seed_view.size()) {
                        if (!finite_device(energy_seed_view(remaining))) {
                            invalid = 1;
                        }
                    } else {
                        const std::size_t source_element =
                            remaining - energy_seed_view.size();
                        const std::size_t row =
                            source_element / source_view.extent(1);
                        const std::size_t column =
                            source_element % source_view.extent(1);
                        if (!finite_device(source_view(row, column))) {
                            invalid = 1;
                        }
                    }
                }
            },
            Kokkos::Sum<std::size_t>(invalid_seeds));
        ++resident_scalar_validation_transfers_;
        if (invalid_seeds != 0) {
            throw std::invalid_argument(
                "POLAR particle-mesh reverse requires finite resident sources "
                "and seeds.");
        }
    }
    if (pbc != std::array<int, 3>{1, 1, 1}) {
        throw std::invalid_argument(
            "POLAR particle-mesh reverse requires fully periodic PBC TTT.");
    }
    if (receiver_sigmas.size()
        != static_cast<std::size_t>(layout_.receiver_widths)) {
        throw std::invalid_argument(
            "POLAR particle-mesh receiver widths do not match the "
            "workspace layout.");
    }
    const auto valid_sigma = [](const double sigma) {
        return finite_double(sigma) && sigma > 0.0;
    };
    if (!valid_sigma(source_sigma)
        || !std::all_of(receiver_sigmas.begin(), receiver_sigmas.end(),
            valid_sigma)) {
        throw std::invalid_argument(
            "POLAR particle-mesh Gaussian widths must be positive and finite.");
    }
    if (!std::all_of(positions.begin(), positions.end(), finite_double)) {
        throw std::invalid_argument(
            "POLAR particle-mesh requires finite positions.");
    }
    if (!resident && !std::all_of(
            source_multipoles.begin(), source_multipoles.end(), finite_double)) {
        throw std::invalid_argument(
            "POLAR particle-mesh requires finite sources/multipoles.");
    }
    const double requested_volume = determinant(cell);
    if (!finite_double(requested_volume) || requested_volume <= 0.0) {
        throw std::invalid_argument(
            "POLAR particle-mesh cell must have positive finite volume.");
    }
    const auto requested_cell_inverse = inverse(cell, requested_volume);
    if (!std::all_of(requested_cell_inverse.begin(),
            requested_cell_inverse.end(), finite_double)) {
        throw std::invalid_argument(
            "POLAR particle-mesh cell must have a finite inverse.");
    }
    if (mesh[0] < spline_order || mesh[1] < spline_order
        || mesh[2] < spline_order) {
        throw std::invalid_argument(
            "POLAR particle-mesh mesh axes must be at least the spline order.");
    }
    neutral_.validate_reserve(mesh[0], mesh[1], mesh[2], nodes,
        layout_.transform_lanes, spline_order);

    // Reserve the reverse-only spectrum snapshot before forward_impl can
    // commit an incompatible mesh.  Keep the candidate local until that
    // forward succeeds, so either allocation failure leaves the complete
    // previously committed shape usable.
    const particle_mesh::MeshShape requested_shape(
        mesh[0], mesh[1], mesh[2], layout_.transform_lanes,
        particle_mesh::MeshLayout::right);
    const std::int64_t requested_snapshot_elements =
        particle_mesh::detail::checked_product(source_lanes,
            requested_shape.spectrum_point_count(),
            "POLAR particle-mesh reverse source spectrum snapshot");
    auto snapshot_candidate = replacement(source_spectrum_snapshot_,
        requested_snapshot_elements,
        "symmetrix_polar_pme_source_spectrum_snapshot",
        "POLAR particle-mesh source spectrum snapshot");

    // Re-establish the exact forward state. Besides making the diagnostic API
    // self-contained, this leaves the receiver fields in the neutral real mesh
    // for the receiver-interpolation position adjoint below.
    auto snapshot_target = snapshot_candidate.extent(0) != 0
        ? snapshot_candidate
        : source_spectrum_snapshot_;
    static_cast<void>(forward_impl(positions, source_multipoles,
        active_channels, cell, pbc, source_sigma, receiver_sigmas, mesh,
        spline_order, include_feature_self_interaction,
        include_energy_self_interaction, resident_sources,
        resident_forward_features, resident_forward_energies,
        resident_positions, &snapshot_target, feature_seed_active));
    if (snapshot_candidate.extent(0) != 0) {
        source_spectrum_snapshot_ = std::move(snapshot_candidate);
        allocation_ids_[16] = ++allocation_serial_;
    }

    PolarParticleMeshReverseResult result;
    result.nodes = nodes;
    result.source_lanes = source_lanes;
    if (!resident) {
        result.source_multipole_adjoints.assign(
            static_cast<std::size_t>(source_elements), 0.0);
        result.position_adjoints.assign(
            static_cast<std::size_t>(particle_mesh::detail::checked_product(
                nodes, 3, "POLAR particle-mesh reverse position adjoints")),
            0.0);
        result.cell_adjoints.assign(9, 0.0);
    } else {
        Kokkos::deep_copy(
            execution_space_, *resident_source_adjoints, Precision(0));
        Kokkos::deep_copy(
            execution_space_, *resident_position_adjoints, 0.0);
        Kokkos::deep_copy(execution_space_, *resident_cell_adjoints, 0.0);
    }

    const std::int64_t transform_lanes = layout_.transform_lanes;
    const std::int64_t transform_elements = particle_mesh::detail::checked_product(
        nodes, transform_lanes,
        "POLAR particle-mesh reverse transform elements");
    const auto weights = typename particle_mesh::ParticleMeshWorkspace<
        Precision>::weights_view(
        transform_weights_.data(), nodes, transform_lanes);
    const auto gathered = typename particle_mesh::ParticleMeshWorkspace<
        Precision>::weight_adjoints_view(
        receiver_features_.data(), nodes, transform_lanes);
    const auto forces = typename particle_mesh::ParticleMeshWorkspace<
        Precision>::position_forces_view(
        transform_position_forces_.data(), nodes, 3);
    const auto no_groups = typename particle_mesh::ParticleMeshWorkspace<
        Precision>::groups_view();

    Kokkos::View<Precision*, Kokkos::HostSpace> host_weights;
    if (!resident) {
        host_weights = Kokkos::create_mirror_view(transform_weights_);
    }
    if (feature_seed_active) {
        if (!resident) {
            std::fill(host_weights.data(),
                host_weights.data() + host_weights.extent(0), Precision(0));
            for (std::int64_t node = 0; node < nodes; ++node) {
                for (std::int64_t lane = 0; lane < feature_lanes; ++lane) {
                    host_weights(node * transform_lanes + lane) =
                        static_cast<Precision>(feature_seeds[
                            static_cast<std::size_t>(
                                node * feature_lanes + lane)]);
                }
            }
            Kokkos::deep_copy(
                execution_space_, transform_weights_, host_weights);
        } else {
            const auto seed_view = *resident_feature_seeds;
            const auto transform = transform_weights_;
            Kokkos::parallel_for(
                "POLAR particle-mesh resident feature seed lanes",
                Kokkos::RangePolicy<execution_space>(
                    execution_space_, 0, transform_elements),
                KOKKOS_LAMBDA(const std::int64_t element) {
                    const std::int64_t node = element / transform_lanes;
                    const std::int64_t lane = element % transform_lanes;
                    transform(element) = lane < feature_lanes
                        ? seed_view(node, lane)
                        : Precision(0);
                });
        }
        // The neutral gather reports physical forces (minus position adjoints).
        // Sampling the already-computed receiver fields with the feature seeds
        // captures the receiver-side interpolation contribution.
        neutral_.gather(weights, no_groups, 0, gathered, forces);
        if (!resident) {
            auto host_forces = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), transform_position_forces_);
            for (std::size_t index = 0;
                index < result.position_adjoints.size(); ++index) {
                result.position_adjoints[index] -=
                    static_cast<double>(host_forces(index));
            }
        } else {
            const auto output_position = *resident_position_adjoints;
            const auto receiver_forces = transform_position_forces_;
            Kokkos::parallel_for(
                "POLAR particle-mesh resident receiver position adjoint",
                Kokkos::RangePolicy<execution_space>(
                    execution_space_, 0, position_elements),
                KOKKOS_LAMBDA(const std::int64_t element) {
                    output_position(element / 3, element % 3) =
                        -static_cast<double>(receiver_forces(element));
                });
        }
        // The same feature weights feed the receiver-seed transform; gather
        // does not mutate them.
        neutral_.spread(weights);
        neutral_.transform_forward();
        neutral_.deconvolve_amplitudes();
    }

    const auto grid = neutral_.reciprocal_grid();
    const auto sigmas = receiver_sigmas_;
    const std::int64_t bins = grid.bins;
    const std::int64_t widths = layout_.receiver_widths;
    const std::int64_t source_components = layout_.source_components;
    const std::int64_t spin_channels = active_channels;
    const std::int64_t receiver_components = layout_.receiver_components;
    const std::int64_t snapshot_elements =
        particle_mesh::detail::checked_product(source_lanes, bins,
            "POLAR particle-mesh reverse source spectrum snapshot");
    // Reverse the receiver projection. The seed assignment is deconvolved
    // once; the explicit receiver-side assignment divisor in the forward
    // spectral field then cancels under the discrete Parseval pairing.
    auto spectrum = neutral_.spectrum();
    Kokkos::View<complex_type*, typename execution_space::memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>> source_spectrum(
        spectrum.data(), snapshot_elements);
    Kokkos::View<complex_type*, typename execution_space::memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>> snapshot_active(
        source_spectrum_snapshot_.data(), snapshot_elements);
    const double volume = determinant(cell);
    const double source_sigma_squared = source_sigma * source_sigma;
    const double source_l0 = fourier_l0_constant(source_sigma, false);
    const double source_l1 = layout_.density_max_l == 0
        ? 0.0
        : fourier_l1_coefficient(source_sigma, false);
    const double feature_prefactor = field_constant / volume;
    const double receiver_l0_prefactor = 2.0 * std::sqrt(std::numbers::pi);
    const double receiver_l1_prefactor = std::numbers::pi * std::sqrt(1.5);
    using complex_type = Kokkos::complex<Precision>;
    const auto device_energy_seeds = channel_energies_;
    if (!resident) {
        auto host_energy_seeds = Kokkos::create_mirror_view(device_energy_seeds);
        for (std::int64_t channel = 0; channel < spin_channels; ++channel) {
            host_energy_seeds(channel) =
                energy_seeds[static_cast<std::size_t>(channel)];
        }
        Kokkos::deep_copy(
            execution_space_, device_energy_seeds, host_energy_seeds);
    } else {
        const auto active_energy_seeds = Kokkos::subview(
            device_energy_seeds,
            std::make_pair(std::int64_t(0), spin_channels));
        Kokkos::deep_copy(execution_space_, active_energy_seeds,
            *resident_energy_seeds);
    }

    // Under cell -> cell*(I+eps) with positions transformed by the same
    // affine map, assignment coordinates and FFT amplitudes are invariant.
    // Only volume, |k|, and the Cartesian l=1 wave factors differentiate.
    // Each grid-wide reduction owns one strain component. A nine-team kernel
    // leaves large meshes almost serial on GPUs; RangePolicy reductions expose
    // the full spectrum while retaining only nine scalar outputs.
    const auto source_snapshot = source_spectrum_snapshot_;
    const auto cell_adjoints = resident
        ? *resident_cell_adjoints
        : cell_adjoints_;
    const auto cell_bin_adjoints = cell_bin_adjoints_;
    Kokkos::parallel_for(
        "POLAR particle-mesh affine cell bins",
        Kokkos::RangePolicy<execution_space>(execution_space_, 0, bins),
        KOKKOS_LAMBDA(const std::int64_t bin) {
            double bin_derivatives[9] = {
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            double wave[3];
            particle_mesh::wave_vector(bin, grid.nx, grid.ny,
                grid.spectrum_extent, grid.cell_inverse, wave);
            const double squared = wave[0] * wave[0]
                + wave[1] * wave[1] + wave[2] * wave[2];
            if (!(squared > 0.0)) {
                for (int tensor = 0; tensor < 9; ++tensor)
                    cell_bin_adjoints(bin * 9 + tensor) = 0.0;
                return;
            }
            const double kernel = feature_prefactor / squared;
            const double source_gaussian = Kokkos::exp(
                -0.5 * source_sigma_squared * squared);
            const double permuted_wave[3] = {wave[1], wave[2], wave[0]};
            double squared_derivatives[9];
            double kernel_derivatives[9];
            double source_gaussian_derivatives[9];
            for (int tensor = 0; tensor < 9; ++tensor) {
                const int row = tensor / 3;
                const int column = tensor % 3;
                const double squared_derivative =
                    -2.0 * wave[row] * wave[column];
                squared_derivatives[tensor] = squared_derivative;
                kernel_derivatives[tensor] = kernel
                    * (-(row == column ? 1.0 : 0.0)
                        - squared_derivative / squared);
                source_gaussian_derivatives[tensor] = source_gaussian
                    * (-0.5 * source_sigma_squared * squared_derivative);
            }
            for (std::int64_t channel = 0;
                channel < spin_channels; ++channel) {
                const std::int64_t source_offset =
                    channel * source_components;
                const std::int64_t feature_offset =
                    channel * feature_dimension;
                const auto scalar_source = source_snapshot[
                    source_offset * bins + bin];
                const double scalar_real =
                    static_cast<double>(scalar_source.real());
                const double scalar_imaginary =
                    static_cast<double>(scalar_source.imag());
                double density_real =
                    source_l0 * source_gaussian * scalar_real;
                double density_imaginary =
                    source_l0 * source_gaussian * scalar_imaginary;
                double density_real_derivatives[9];
                double density_imaginary_derivatives[9];
                for (int tensor = 0; tensor < 9; ++tensor) {
                    density_real_derivatives[tensor] = source_l0
                        * source_gaussian_derivatives[tensor] * scalar_real;
                    density_imaginary_derivatives[tensor] = source_l0
                        * source_gaussian_derivatives[tensor]
                        * scalar_imaginary;
                }
                if (source_components == 4) {
                    for (int component = 0; component < 3; ++component) {
                        const auto dipole = source_snapshot[
                            (source_offset + 1 + component) * bins + bin];
                        const double dipole_real =
                            static_cast<double>(dipole.real());
                        const double dipole_imaginary =
                            static_cast<double>(dipole.imag());
                        const double factor = source_l1 * source_gaussian
                            * permuted_wave[component];
                        density_real += factor * dipole_imaginary;
                        density_imaginary -= factor * dipole_real;
                        const int wave_axis = component == 0 ? 1
                            : (component == 1 ? 2 : 0);
                        for (int tensor = 0; tensor < 9; ++tensor) {
                            const int row = tensor / 3;
                            const int column = tensor % 3;
                            const double permuted_derivative =
                                wave_axis == row ? -wave[column] : 0.0;
                            const double factor_derivative = source_l1
                                * (source_gaussian_derivatives[tensor]
                                        * permuted_wave[component]
                                    + source_gaussian
                                        * permuted_derivative);
                            density_real_derivatives[tensor] +=
                                factor_derivative * dipole_imaginary;
                            density_imaginary_derivatives[tensor] -=
                                factor_derivative * dipole_real;
                        }
                    }
                }
                if (feature_seed_active) {
                    for (std::int64_t width = 0; width < widths; ++width) {
                        const double receiver_sigma = sigmas(width);
                        const double receiver_squared =
                            receiver_sigma * receiver_sigma;
                        const double receiver_gaussian = Kokkos::exp(
                            -0.5 * receiver_squared * squared);
                        const double scalar_factor = kernel
                            * receiver_l0_prefactor * receiver_gaussian;
                        const auto scalar_seed = spectrum.data()[
                            (feature_offset + width) * bins + bin];
                        const double scalar_seed_real =
                            static_cast<double>(scalar_seed.real());
                        const double scalar_seed_imaginary =
                            static_cast<double>(scalar_seed.imag());
                        for (int tensor = 0; tensor < 9; ++tensor) {
                            const double receiver_gaussian_derivative =
                                receiver_gaussian
                                * (-0.5 * receiver_squared
                                    * squared_derivatives[tensor]);
                            const double scalar_factor_derivative =
                                receiver_l0_prefactor
                                * (kernel_derivatives[tensor]
                                        * receiver_gaussian
                                    + kernel
                                        * receiver_gaussian_derivative);
                            bin_derivatives[tensor] += scalar_seed_real
                                    * (scalar_factor_derivative * density_real
                                        + scalar_factor
                                            * density_real_derivatives[tensor])
                                + scalar_seed_imaginary
                                    * (scalar_factor_derivative
                                            * density_imaginary
                                        + scalar_factor
                                            * density_imaginary_derivatives[
                                                tensor]);
                        }
                        if (receiver_components == 4) {
                            const double vector_scale =
                                receiver_l1_prefactor * receiver_sigma;
                            for (int component = 0; component < 3; ++component) {
                                const std::int64_t feature = widths
                                    + width * 3 + component;
                                const double factor = kernel * vector_scale
                                    * receiver_gaussian
                                    * permuted_wave[component];
                                const auto vector_seed = spectrum.data()[
                                    (feature_offset + feature) * bins + bin];
                                const double vector_seed_real =
                                    static_cast<double>(vector_seed.real());
                                const double vector_seed_imaginary =
                                    static_cast<double>(vector_seed.imag());
                                const int wave_axis = component == 0 ? 1
                                    : (component == 1 ? 2 : 0);
                                for (int tensor = 0; tensor < 9; ++tensor) {
                                    const int row = tensor / 3;
                                    const int column = tensor % 3;
                                    const double receiver_gaussian_derivative =
                                        receiver_gaussian
                                        * (-0.5 * receiver_squared
                                            * squared_derivatives[tensor]);
                                    const double permuted_derivative =
                                        wave_axis == row ? -wave[column] : 0.0;
                                    const double factor_derivative =
                                        vector_scale
                                        * ((kernel_derivatives[tensor]
                                                    * receiver_gaussian
                                                + kernel
                                                    * receiver_gaussian_derivative)
                                                * permuted_wave[component]
                                            + kernel * receiver_gaussian
                                                * permuted_derivative);
                                    bin_derivatives[tensor] += vector_seed_real
                                            * (-factor_derivative
                                                    * density_imaginary
                                                - factor
                                                    * density_imaginary_derivatives[
                                                        tensor])
                                        + vector_seed_imaginary
                                            * (factor_derivative * density_real
                                                + factor
                                                    * density_real_derivatives[
                                                        tensor]);
                                }
                            }
                        }
                    }
                }
                if (energy_seed_active) {
                    const double norm_squared = density_real * density_real
                        + density_imaginary * density_imaginary;
                    for (int tensor = 0; tensor < 9; ++tensor) {
                        const double norm_squared_derivative = 2.0
                            * (density_real * density_real_derivatives[tensor]
                                + density_imaginary
                                    * density_imaginary_derivatives[tensor]);
                        bin_derivatives[tensor] += device_energy_seeds(channel)
                            * (0.5 * kernel_derivatives[tensor] * norm_squared
                                + 0.5 * kernel * norm_squared_derivative);
                    }
                }
            }
            const double weight =
                static_cast<double>(grid.hermitian_weight(bin));
            for (int tensor = 0; tensor < 9; ++tensor)
                cell_bin_adjoints(bin * 9 + tensor) =
                    weight * bin_derivatives[tensor];
        });
    for (int tensor = 0; tensor < 9; ++tensor) {
        const auto derivative = Kokkos::subview(cell_adjoints, tensor);
        Kokkos::parallel_reduce(
            "POLAR particle-mesh affine cell reduce",
            Kokkos::RangePolicy<execution_space>(execution_space_, 0, bins),
            KOKKOS_LAMBDA(const std::int64_t bin, double& sum) {
                sum += cell_bin_adjoints(bin * 9 + tensor);
            }, derivative);
    }

    if (feature_seed_active) {
        Kokkos::parallel_for(
            "POLAR particle-mesh feature spectral reverse",
            Kokkos::RangePolicy<execution_space>(execution_space_, 0, bins),
            KOKKOS_LAMBDA(const std::int64_t bin) {
            complex_type density_cotangent[2] = {
                complex_type(0, 0), complex_type(0, 0)};
            const double norm = grid.k_norm(bin);
            const double squared = norm * norm;
            if (squared > 0.0) {
                double wave[3];
                particle_mesh::wave_vector(bin, grid.nx, grid.ny,
                    grid.spectrum_extent, grid.cell_inverse, wave);
                const double permuted_wave[3] = {wave[1], wave[2], wave[0]};
                const double kernel = feature_prefactor / squared;
                for (std::int64_t channel = 0;
                    channel < spin_channels; ++channel) {
                    const std::int64_t feature_offset =
                        channel * feature_dimension;
                    double real = 0.0;
                    double imaginary = 0.0;
                    for (std::int64_t width = 0; width < widths; ++width) {
                        const double receiver_sigma = sigmas(width);
                        const double gaussian = Kokkos::exp(
                            -0.5 * receiver_sigma * receiver_sigma * squared);
                        const double scalar_factor = kernel
                            * receiver_l0_prefactor * gaussian;
                        const auto scalar = spectrum.data()[
                            (feature_offset + width) * bins + bin];
                        real += scalar_factor * static_cast<double>(scalar.real());
                        imaginary += scalar_factor
                            * static_cast<double>(scalar.imag());
                        if (receiver_components == 4) {
                            const double vector_base = kernel
                                * receiver_l1_prefactor * receiver_sigma * gaussian;
                            for (int component = 0; component < 3; ++component) {
                                const std::int64_t feature = widths
                                    + width * 3 + component;
                                const auto vector = spectrum.data()[
                                    (feature_offset + feature) * bins + bin];
                                const double factor = vector_base
                                    * permuted_wave[component];
                                // Adjoint of multiplication by +i*factor.
                                real += factor
                                    * static_cast<double>(vector.imag());
                                imaginary -= factor
                                    * static_cast<double>(vector.real());
                            }
                        }
                    }
                    density_cotangent[channel] = complex_type(
                        static_cast<Precision>(real),
                        static_cast<Precision>(imaginary));
                }
            }
            const double source_gaussian = Kokkos::exp(
                -0.5 * source_sigma_squared * squared);
            double wave[3] = {0.0, 0.0, 0.0};
            if (squared > 0.0) {
                particle_mesh::wave_vector(bin, grid.nx, grid.ny,
                    grid.spectrum_extent, grid.cell_inverse, wave);
            }
            const double permuted_wave[3] = {wave[1], wave[2], wave[0]};
            for (std::int64_t channel = 0;
                channel < spin_channels; ++channel) {
                const std::int64_t source_offset = channel * source_components;
                const auto cotangent = density_cotangent[channel];
                spectrum.data()[source_offset * bins + bin] = complex_type(
                    static_cast<Precision>(source_l0 * source_gaussian
                        * static_cast<double>(cotangent.real())),
                    static_cast<Precision>(source_l0 * source_gaussian
                        * static_cast<double>(cotangent.imag())));
                if (source_components == 4) {
                    for (int component = 0; component < 3; ++component) {
                        const double factor = source_l1 * source_gaussian
                            * permuted_wave[component];
                        // Adjoint of multiplication by -i*factor.
                        spectrum.data()[(source_offset + 1 + component)
                                * bins + bin] = complex_type(
                            static_cast<Precision>(-factor
                                * static_cast<double>(cotangent.imag())),
                            static_cast<Precision>(factor
                                * static_cast<double>(cotangent.real())));
                    }
                }
            }
            for (std::int64_t lane = source_lanes;
                lane < transform_lanes; ++lane) {
                spectrum.data()[lane * bins + bin] = complex_type(0, 0);
            }
            });
        neutral_.deconvolve_cotangent();
        neutral_.transform_inverse();
    }

    if (!resident) {
        std::fill(host_weights.data(),
            host_weights.data() + host_weights.extent(0), Precision(0));
        for (std::int64_t node = 0; node < nodes; ++node) {
            for (std::int64_t lane = 0; lane < source_lanes; ++lane) {
                host_weights(node * transform_lanes + lane) =
                    static_cast<Precision>(source_multipoles[
                        static_cast<std::size_t>(node * source_lanes + lane)]);
            }
        }
        Kokkos::deep_copy(execution_space_, transform_weights_, host_weights);
        execution_space_.fence(
            "POLAR particle-mesh reverse gather-weight staging handoff");
    } else {
        const auto source_view = *resident_sources;
        const auto transform = transform_weights_;
        Kokkos::parallel_for(
            "POLAR particle-mesh resident reverse gather weights",
            Kokkos::RangePolicy<execution_space>(
                execution_space_, 0, transform_elements),
            KOKKOS_LAMBDA(const std::int64_t element) {
                const std::int64_t node = element / transform_lanes;
                const std::int64_t lane = element % transform_lanes;
                transform(element) = lane < source_lanes
                    ? source_view(node, lane)
                    : Precision(0);
            });
    }
    auto accumulate_source_pass = [&] {
        if (!resident) {
            auto host_gathered = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), receiver_features_);
            auto pass_forces = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), transform_position_forces_);
            for (std::int64_t node = 0; node < nodes; ++node) {
                for (std::int64_t lane = 0; lane < source_lanes; ++lane) {
                    result.source_multipole_adjoints[
                        static_cast<std::size_t>(node * source_lanes + lane)] +=
                        static_cast<double>(
                            host_gathered(node * transform_lanes + lane));
                }
                for (int component = 0; component < 3; ++component) {
                    const std::size_t index = static_cast<std::size_t>(
                        3 * node + component);
                    result.position_adjoints[index] -=
                        static_cast<double>(pass_forces(index));
                }
            }
        } else {
            const auto output_source = *resident_source_adjoints;
            const auto output_position = *resident_position_adjoints;
            const auto pass_gathered = receiver_features_;
            const auto pass_forces = transform_position_forces_;
            Kokkos::parallel_for(
                "POLAR particle-mesh resident source reverse accumulate",
                Kokkos::RangePolicy<execution_space>(
                    execution_space_, 0, nodes),
                KOKKOS_LAMBDA(const std::int64_t node) {
                    for (std::int64_t lane = 0; lane < source_lanes; ++lane) {
                        output_source(node, lane) +=
                            pass_gathered(node * transform_lanes + lane);
                    }
                    for (int component = 0; component < 3; ++component) {
                        output_position(node, component) -=
                            static_cast<double>(pass_forces(3 * node + component));
                    }
                });
        }
    };
    if (feature_seed_active) {
        neutral_.gather(weights, no_groups, 0, gathered, forces);
        accumulate_source_pass();
    }

    if (energy_seed_active) {
        // Restore the saved source amplitudes for the energy reverse. The
        // feature pass may already have consumed its receiver-seed spectrum.
        spectrum = neutral_.spectrum();
        Kokkos::deep_copy(execution_space_, source_spectrum, snapshot_active);
        Kokkos::parallel_for(
            "POLAR particle-mesh energy spectral reverse",
            Kokkos::RangePolicy<execution_space>(execution_space_, 0, bins),
            KOKKOS_LAMBDA(const std::int64_t bin) {
            complex_type amplitudes[8] = {
                complex_type(0, 0), complex_type(0, 0),
                complex_type(0, 0), complex_type(0, 0),
                complex_type(0, 0), complex_type(0, 0),
                complex_type(0, 0), complex_type(0, 0)};
            for (std::int64_t lane = 0; lane < source_lanes; ++lane) {
                amplitudes[lane] = spectrum.data()[lane * bins + bin];
            }
            const double norm = grid.k_norm(bin);
            const double squared = norm * norm;
            double wave[3] = {0.0, 0.0, 0.0};
            if (squared > 0.0) {
                particle_mesh::wave_vector(bin, grid.nx, grid.ny,
                    grid.spectrum_extent, grid.cell_inverse, wave);
            }
            const double permuted_wave[3] = {wave[1], wave[2], wave[0]};
            const double gaussian = Kokkos::exp(
                -0.5 * source_sigma_squared * squared);
            for (std::int64_t channel = 0;
                channel < spin_channels; ++channel) {
                const std::int64_t offset = channel * source_components;
                double density_real = source_l0 * gaussian
                    * static_cast<double>(amplitudes[offset].real());
                double density_imaginary = source_l0 * gaussian
                    * static_cast<double>(amplitudes[offset].imag());
                if (source_components == 4) {
                    for (int component = 0; component < 3; ++component) {
                        const double factor = source_l1 * gaussian
                            * permuted_wave[component];
                        density_real += factor
                            * static_cast<double>(
                                amplitudes[offset + 1 + component].imag());
                        density_imaginary -= factor
                            * static_cast<double>(
                                amplitudes[offset + 1 + component].real());
                    }
                }
                const double kernel = squared > 0.0
                    ? device_energy_seeds(channel) * feature_prefactor / squared
                    : 0.0;
                const double cotangent_real = kernel * density_real;
                const double cotangent_imaginary = kernel * density_imaginary;
                spectrum.data()[offset * bins + bin] = complex_type(
                    static_cast<Precision>(source_l0 * gaussian
                        * cotangent_real),
                    static_cast<Precision>(source_l0 * gaussian
                        * cotangent_imaginary));
                if (source_components == 4) {
                    for (int component = 0; component < 3; ++component) {
                        const double factor = source_l1 * gaussian
                            * permuted_wave[component];
                        spectrum.data()[(offset + 1 + component) * bins + bin] =
                            complex_type(
                                static_cast<Precision>(
                                    -factor * cotangent_imaginary),
                                static_cast<Precision>(factor * cotangent_real));
                    }
                }
            }
            for (std::int64_t lane = source_lanes;
                lane < transform_lanes; ++lane) {
                spectrum.data()[lane * bins + bin] = complex_type(0, 0);
            }
            });
        neutral_.deconvolve_cotangent();
        neutral_.transform_inverse();
        neutral_.gather(weights, no_groups, 0, gathered, forces);
        accumulate_source_pass();
    }

    if (feature_seed_active && !include_feature_self_interaction) {
        for (std::int64_t width = 0; width < widths; ++width) {
            const double scalar_self = self_constant(
                0, source_sigma,
                receiver_sigmas[static_cast<std::size_t>(width)], true);
            const double vector_self = layout_.receiver_components == 4
                ? self_constant(1, source_sigma,
                    receiver_sigmas[static_cast<std::size_t>(width)], true)
                : 0.0;
            if (!resident) {
                for (std::int64_t node = 0; node < nodes; ++node) {
                    for (std::int64_t channel = 0;
                        channel < spin_channels; ++channel) {
                        const std::int64_t source_offset =
                            channel * source_components;
                        const std::int64_t feature_offset =
                            channel * feature_dimension;
                        result.source_multipole_adjoints[
                            static_cast<std::size_t>(
                                node * source_lanes + source_offset)] -=
                            scalar_self * feature_seeds[
                                static_cast<std::size_t>(node * feature_lanes
                                    + feature_offset + width)];
                        if (source_components == 4) {
                            for (int component = 0; component < 3; ++component) {
                                const std::int64_t feature = widths
                                    + width * 3 + component;
                                result.source_multipole_adjoints[
                                    static_cast<std::size_t>(node * source_lanes
                                        + source_offset + 1 + component)] -=
                                    vector_self * feature_seeds[
                                        static_cast<std::size_t>(
                                            node * feature_lanes
                                            + feature_offset + feature)];
                            }
                        }
                    }
                }
            } else {
                const auto seed_view = *resident_feature_seeds;
                const auto output_source = *resident_source_adjoints;
                Kokkos::parallel_for(
                    "POLAR particle-mesh resident feature self reverse",
                    Kokkos::RangePolicy<execution_space>(
                        execution_space_, 0, channel_elements),
                    KOKKOS_LAMBDA(const std::int64_t element) {
                        const std::int64_t node = element / spin_channels;
                        const std::int64_t channel = element % spin_channels;
                        const std::int64_t source_offset =
                            channel * source_components;
                        const std::int64_t feature_offset =
                            channel * feature_dimension;
                        output_source(node, source_offset) -=
                            static_cast<Precision>(scalar_self)
                            * seed_view(node, feature_offset + width);
                        if (source_components == 4) {
                            for (int component = 0; component < 3; ++component) {
                                const std::int64_t feature = widths
                                    + width * 3 + component;
                                output_source(
                                    node, source_offset + 1 + component) -=
                                    static_cast<Precision>(vector_self)
                                    * seed_view(node, feature_offset + feature);
                            }
                        }
                    });
            }
        }
    }
    if (energy_seed_active && !include_energy_self_interaction) {
        const double scalar_self = self_constant(
            0, source_sigma, source_sigma, false);
        const double vector_self = source_components == 4
            ? self_constant(1, source_sigma, source_sigma, false) : 0.0;
        if (!resident) {
            for (std::int64_t node = 0; node < nodes; ++node) {
                for (std::int64_t channel = 0;
                    channel < spin_channels; ++channel) {
                    const std::int64_t offset = channel * source_components;
                    const double seed =
                        energy_seeds[static_cast<std::size_t>(channel)];
                    result.source_multipole_adjoints[
                        static_cast<std::size_t>(node * source_lanes + offset)] -=
                        seed * scalar_self * source_multipoles[
                            static_cast<std::size_t>(
                                node * source_lanes + offset)];
                    for (int component = 0; component < source_components - 1;
                        ++component) {
                        result.source_multipole_adjoints[
                            static_cast<std::size_t>(node * source_lanes + offset
                                + 1 + component)] -= seed * vector_self
                            * source_multipoles[static_cast<std::size_t>(
                                node * source_lanes + offset + 1 + component)];
                    }
                }
            }
        } else {
            const auto seed_view = *resident_energy_seeds;
            const auto source_view = *resident_sources;
            const auto output_source = *resident_source_adjoints;
            Kokkos::parallel_for(
                "POLAR particle-mesh resident energy self reverse",
                Kokkos::RangePolicy<execution_space>(
                    execution_space_, 0, channel_elements),
                KOKKOS_LAMBDA(const std::int64_t element) {
                    const std::int64_t node = element / spin_channels;
                    const std::int64_t channel = element % spin_channels;
                    const std::int64_t offset = channel * source_components;
                    const double seed = seed_view(channel);
                    output_source(node, offset) -= static_cast<Precision>(
                        seed * scalar_self * source_view(node, offset));
                    for (int component = 0; component < source_components - 1;
                        ++component) {
                        output_source(node, offset + 1 + component) -=
                            static_cast<Precision>(seed * vector_self
                                * source_view(node, offset + 1 + component));
                    }
                });
        }
    }
    execution_space_.fence("POLAR particle-mesh reverse outputs");
    if (!resident) {
        auto host_cell_adjoints = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), cell_adjoints_);
        for (std::size_t component = 0; component < result.cell_adjoints.size();
            ++component) {
            result.cell_adjoints[component] = host_cell_adjoints(component);
        }
    } else {
        ++resident_completion_fences_;
    }
    return result;
}

template class PolarParticleMeshWorkspaceT<float>;
template class PolarParticleMeshWorkspaceT<double>;

}  // namespace symmetrix::mace_polar
