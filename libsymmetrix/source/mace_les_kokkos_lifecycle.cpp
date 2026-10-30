#include "mace_les_kokkos.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <vector>

template <typename Precision>
void MACELESKokkos<Precision>::validate_model_capability(
    const MACEModelCapability capability) const
{
    switch (capability) {
    case MACEModelCapability::alchemical:
        throw std::invalid_argument(
            "MACELES does not support alchemical coefficients.");
    case MACEModelCapability::distributed_prepared:
        throw std::invalid_argument(
            "MACELES does not support distributed prepared evaluation.");
    case MACEModelCapability::fixed_workspace:
        throw std::invalid_argument(
            "MACELES does not support fixed-workspace tiled execution.");
    case MACEModelCapability::metal_artifact:
        throw MACEModelCapabilityError(
            "MACELES does not support Metal artifact loader");
    }
    throw std::logic_error("Unknown MACE model capability.");
}

template <typename Precision>
void MACELESKokkos<Precision>::validate_model_execution_mode(
    const ExecutionMode mode) const
{
    if (mode != ExecutionMode::generic && !uses_prepared_execution(mode))
        throw std::invalid_argument(
            "MACELES streamed_edges='"+execution_mode_name(mode)
            +"' is unavailable; select reference or direct execution.");
}

template <typename Precision>
void MACELESKokkos<Precision>::on_model_execution_mode_change(
    const ExecutionMode previous,
    const ExecutionMode next)
{
    if (previous == next
        || this->les_.periodic_algorithm
            != symmetrix::les::LESPeriodicAlgorithm::particle_mesh)
        return;
    this->fence_les_particle_mesh_owner(
        "Replace LES particle-mesh execution mode");
    this->les_.particle_mesh_owner_valid = false;
    this->les_.geometry_ready = false;
    this->les_.results_ready = false;
}

template void MACELESKokkos<float>::validate_model_capability(
    MACEModelCapability) const;
template void MACELESKokkos<double>::validate_model_capability(
    MACEModelCapability) const;
template void MACELESKokkos<float>::validate_model_execution_mode(
    ExecutionMode) const;
template void MACELESKokkos<double>::validate_model_execution_mode(
    ExecutionMode) const;
template void MACELESKokkos<float>::on_model_execution_mode_change(
    ExecutionMode, ExecutionMode);
template void MACELESKokkos<double>::on_model_execution_mode_change(
    ExecutionMode, ExecutionMode);

template <typename Precision>
MACELESKokkos<Precision>::~MACELESKokkos()
{
    fence_les_particle_mesh_owner("Destroy MACELES evaluator");
    Kokkos::fence("Destroy MACELES evaluator state");
}

template <typename Precision>
int MACELESKokkos<Precision>::validate_les_position_count(
    const std::size_t position_count) const
{
    if (position_count%3 != 0)
        throw std::invalid_argument("LES positions must contain three values per node.");
    const std::size_t nodes = position_count/3;
    if (nodes > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("LES geometry node count exceeds the supported int range.");
    return static_cast<int>(nodes);
}

template <typename Precision>
void MACELESKokkos<Precision>::set_les_geometry(
    const std::span<const double> positions,
    const std::span<const double> cell,
    const std::span<const int> pbc)
{
    if (!is_mace_les())
        throw std::logic_error("LES geometry requires a MACELES model.");
    les_.charge_readouts_ready = false;
    les_.charge_readout_nodes = 0;
    if (positions.size()%3 != 0 || cell.size() != 9 || pbc.size() != 3)
        throw std::invalid_argument(
            "LES geometry requires positions [N,3], cell [3,3], and PBC [3].");
    const int geometry_nodes = validate_les_position_count(positions.size());
    if (std::any_of(positions.begin(), positions.end(),
            [] (const double value) { return !std::isfinite(value); })
        || std::any_of(cell.begin(), cell.end(),
            [] (const double value) { return !std::isfinite(value); }))
        throw std::invalid_argument("LES geometry values must be finite.");
    if (std::any_of(pbc.begin(), pbc.end(),
            [] (const int value) { return value != 0 && value != 1; }))
        throw std::invalid_argument("LES PBC entries must be boolean values.");
    const bool periodic = pbc.front() != 0;
    if (std::any_of(pbc.begin(), pbc.end(),
            [periodic] (const int value) { return (value != 0) != periodic; }))
        throw std::invalid_argument("LES PBC must be all false or all true.");

    les_.geometry_ready = false;
    les_.geometry_active_nodes = 0;
    les_.results_ready = false;
    les_.geometry_execution_space_selected = true;
    les_.geometry_execution_space_factorized =
        uses_prepared_execution(this->streamed_edges);
    const auto execution_space = les_.geometry_execution_space_factorized
        ? this->factorized_execution_space : Kokkos::DefaultExecutionSpace();
    const bool particle_mesh_periodic = periodic
        && les_.periodic_algorithm
            == symmetrix::les::LESPeriodicAlgorithm::particle_mesh;

    if (particle_mesh_periodic) {
        const std::uint32_t instance_id = execution_space.impl_instance_id();
        if (!les_.particle_mesh_owner_valid
            || les_.particle_mesh_owner_mode != this->streamed_edges
            || les_.particle_mesh_execution_instance_id != instance_id) {
            fence_les_particle_mesh_owner(
                "Replace LES particle-mesh execution instance");
            les_.monopole->configure_particle_mesh_for_testing(
                les_.particle_mesh_shape[0],
                les_.particle_mesh_shape[1],
                les_.particle_mesh_shape[2],
                les_.particle_mesh_spline_order);
            les_.particle_mesh_owner_valid = true;
            les_.particle_mesh_owner_mode = this->streamed_edges;
            les_.particle_mesh_owner_execution_space = execution_space;
            les_.particle_mesh_execution_instance_id =
                les_.particle_mesh_owner_execution_space.impl_instance_id();
            ++les_.particle_mesh_owner_generation;
        }
    }

    if (les_.particle_mesh_owner_valid
        && les_.periodic_algorithm
            == symmetrix::les::LESPeriodicAlgorithm::particle_mesh) {
        // This must precede a count-driven positions reallocation, including
        // a nonperiodic geometry that reuses the shared MACE position buffer.
        // It fences the exact workspace owner once and lets reserve or the
        // nonperiodic path consume that fence.
        les_.monopole->prepare_particle_count_change(
            execution_space,
            static_cast<std::size_t>(geometry_nodes));
    }

    if (les_.positions.extent(0) < positions.size())
        Kokkos::realloc(
            Kokkos::WithoutInitializing, les_.positions, positions.size());
    if (les_.cell.extent(0) != 9)
        Kokkos::realloc(Kokkos::WithoutInitializing, les_.cell, 9);
    if (les_.pbc.extent(0) != 3)
        Kokkos::realloc(Kokkos::WithoutInitializing, les_.pbc, 3);
    auto positions_host = Kokkos::create_mirror_view(les_.positions);
    auto cell_host = Kokkos::create_mirror_view(les_.cell);
    auto pbc_host = Kokkos::create_mirror_view(les_.pbc);
    std::copy(positions.begin(), positions.end(), positions_host.data());
    std::copy(cell.begin(), cell.end(), cell_host.data());
    std::copy(pbc.begin(), pbc.end(), pbc_host.data());
    Kokkos::deep_copy(execution_space, les_.positions, positions_host);
    Kokkos::deep_copy(execution_space, les_.cell, cell_host);
    Kokkos::deep_copy(execution_space, les_.pbc, pbc_host);
    les_.periodic = periodic;
    if (periodic) {
        const std::vector<double> cell_values(cell.begin(), cell.end());
        if (particle_mesh_periodic) {
            les_.monopole->prepare_periodic_geometry(
                execution_space, cell_values, positions, les_.positions,
                static_cast<std::size_t>(geometry_nodes));
        } else {
            les_.monopole->prepare_periodic(execution_space, cell_values);
        }
    }
    les_.geometry_active_nodes = geometry_nodes;
    les_.geometry_ready = true;
    les_.results_ready = false;
}

template <typename Precision>
void MACELESKokkos<Precision>::configure_les_particle_mesh_for_testing(
    const std::int64_t nx,
    const std::int64_t ny,
    const std::int64_t nz,
    const std::int64_t spline_order)
{
    if (!is_mace_les() || !les_.monopole) {
        throw std::logic_error(
            "LES particle-mesh selection requires a MACELES model.");
    }
    const std::array<std::int64_t, 3> requested_shape = {nx, ny, nz};
    if (les_.periodic_algorithm
            == symmetrix::les::LESPeriodicAlgorithm::particle_mesh
        && les_.particle_mesh_shape == requested_shape
        && les_.particle_mesh_spline_order == spline_order) {
        return;
    }
    fence_les_particle_mesh_owner("Replace LES particle-mesh configuration");
    les_.monopole->configure_particle_mesh_for_testing(nx, ny, nz, spline_order);
    les_.periodic_algorithm = symmetrix::les::LESPeriodicAlgorithm::particle_mesh;
    les_.particle_mesh_shape = requested_shape;
    les_.particle_mesh_spline_order = spline_order;
    les_.particle_mesh_owner_valid = false;
    les_.geometry_ready = false;
    les_.results_ready = false;
}

template <typename Precision>
void MACELESKokkos<Precision>::use_explicit_les_periodic_for_testing()
{
    if (!is_mace_les() || !les_.monopole) {
        throw std::logic_error(
            "Explicit LES selection requires a MACELES model.");
    }
    fence_les_particle_mesh_owner("Replace LES particle mesh with explicit sum");
    if (les_.periodic_algorithm == symmetrix::les::LESPeriodicAlgorithm::particle_mesh
        && les_.particle_mesh_owner_valid) {
        ++les_.particle_mesh_owner_generation;
    }
    les_.monopole->use_explicit_periodic_for_testing();
    les_.periodic_algorithm = symmetrix::les::LESPeriodicAlgorithm::explicit_sum;
    les_.particle_mesh_owner_valid = false;
    les_.geometry_ready = false;
    les_.results_ready = false;
}

template <typename Precision>
void MACELESKokkos<Precision>::fence_les_particle_mesh_owner(const char* label)
{
    if (!les_.particle_mesh_owner_valid) {
        return;
    }
    les_.particle_mesh_owner_execution_space.fence(label);
    ++les_.particle_mesh_owner_fences;
}


template <typename Precision>
std::tuple<bool, bool, std::string, std::uint32_t, std::uint32_t, std::uint64_t>
MACELESKokkos<Precision>::les_particle_mesh_stream_probe_for_testing()
{
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    using execution_space = Kokkos::DefaultExecutionSpace;
    try {
        auto instances = Kokkos::Experimental::partition_space(
            execution_space(), 1, 1);
        const auto asynchronous = instances[0];
        const auto reference_execution = instances[1];
        const auto default_execution = execution_space();
        constexpr std::size_t initial_count = 3;
        constexpr std::size_t grown_count = 4;
        constexpr std::size_t final_count = 2;
        const std::array<double, 9> initial_positions_values = {
            0.21, 0.37, 0.53,
            1.17, 0.82, 0.44,
            0.69, 1.51, 1.08};
        const std::array<double, 12> grown_positions_values = {
            0.21, 0.37, 0.53,
            1.17, 0.82, 0.44,
            0.69, 1.51, 1.08,
            1.83, 1.24, 0.77};
        const std::array<double, 6> final_positions_values = {
            0.21, 0.37, 0.53,
            1.17, 0.82, 0.44};
        const std::vector<double> cell = {
            5.71, 0.17, 0.09,
            0.11, 5.33, 0.21,
            0.07, 0.19, 5.49};

        const auto run = [&] (const execution_space& instance, const bool fence_fill) {
            Kokkos::View<double*> initial_positions(
                "LES stream probe initial positions", 3 * initial_count);
            Kokkos::View<double*> grown_positions(
                "LES stream probe grown positions", 3 * grown_count);
            Kokkos::View<double*> final_positions(
                "LES stream probe final positions", 3 * final_count);
            Kokkos::View<double*> charges("LES stream probe charges", final_count);
            Kokkos::parallel_for(
                "LES stream probe initial position fill",
                Kokkos::RangePolicy<execution_space>(instance, 0, 3 * initial_count),
                KOKKOS_LAMBDA(const std::size_t index) {
                    constexpr double values[9] = {
                        0.21, 0.37, 0.53,
                        1.17, 0.82, 0.44,
                        0.69, 1.51, 1.08};
                    initial_positions(index) = values[index];
                });
            Kokkos::parallel_for(
                "LES stream probe grown position fill",
                Kokkos::RangePolicy<execution_space>(instance, 0, 3 * grown_count),
                KOKKOS_LAMBDA(const std::size_t index) {
                    constexpr double values[12] = {
                        0.21, 0.37, 0.53,
                        1.17, 0.82, 0.44,
                        0.69, 1.51, 1.08,
                        1.83, 1.24, 0.77};
                    grown_positions(index) = values[index];
                });
            Kokkos::parallel_for(
                "LES stream probe final position fill",
                Kokkos::RangePolicy<execution_space>(instance, 0, 3 * final_count),
                KOKKOS_LAMBDA(const std::size_t index) {
                    constexpr double values[6] = {
                        0.21, 0.37, 0.53,
                        1.17, 0.82, 0.44};
                    final_positions(index) = values[index];
                });
            if (fence_fill) {
                instance.fence("LES stream probe fenced reference positions");
            }
            symmetrix::les::LESMonopoleKokkos<double> monopole(0.8, 1.35, true);
            monopole.configure_particle_mesh_for_testing(24, 24, 24, 4);
            monopole.prepare_periodic_geometry(
                instance, cell, initial_positions_values, initial_positions, initial_count);
            monopole.prepare_periodic_geometry(
                instance, cell, grown_positions_values, grown_positions, grown_count);
            monopole.prepare_periodic_geometry(
                instance, cell, final_positions_values, final_positions, final_count);
            Kokkos::parallel_for(
                "LES stream probe charge fill",
                Kokkos::RangePolicy<execution_space>(instance, 0, final_count),
                KOKKOS_LAMBDA(const std::size_t index) {
                    constexpr double values[2] = {0.41, -0.73};
                    charges(index) = values[index];
                });
            if (fence_fill) {
                instance.fence("LES stream probe fenced reference charges");
            }
            const auto result = monopole.compute_periodic_device(
                instance, charges, final_positions, final_count);
            std::array<double, 20> host{};
            Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                host_atomic(host.data() + 1, final_count);
            Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                host_forces(host.data() + 3, 3 * final_count);
            Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                host_adjoints(host.data() + 9, final_count);
            Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                host_strain(host.data() + 11, 9);
            Kokkos::View<double, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                host_energy(host.data());
            Kokkos::deep_copy(instance, host_energy, result.energy);
            Kokkos::deep_copy(
                instance, host_atomic,
                Kokkos::subview(result.atomic_energies,
                    std::make_pair(std::size_t(0), final_count)));
            Kokkos::deep_copy(
                instance, host_forces,
                Kokkos::subview(result.forces,
                    std::make_pair(std::size_t(0), 3 * final_count)));
            Kokkos::deep_copy(
                instance, host_adjoints,
                Kokkos::subview(result.charge_adjoints,
                    std::make_pair(std::size_t(0), final_count)));
            Kokkos::deep_copy(
                instance, host_strain,
                Kokkos::subview(result.strain_derivative, std::make_pair(std::size_t(0), std::size_t(9))));
            instance.fence("LES stream probe terminal readback");
            return std::make_pair(
                host, monopole.diagnostics().particle_mesh_particle_count_fences);
        };

        const auto asynchronous_result = run(asynchronous, false);
        const auto reference_result = run(reference_execution, true);
        const bool passed = std::equal(
            asynchronous_result.first.begin(), asynchronous_result.first.end(),
            reference_result.first.begin(), [] (const double left, const double right) {
                return std::abs(left - right) <= 1.0e-12;
            });
        return {
            true,
            passed,
            passed ? std::string() : std::string("asynchronous and fenced results differ"),
            asynchronous.impl_instance_id(),
            default_execution.impl_instance_id(),
            asynchronous_result.second};
    } catch (const std::exception& error) {
        return {true, false, error.what(), 0, 0, 0};
    }
#else
    return {
        false,
        false,
        "requires a CUDA or HIP Kokkos execution space",
        0,
        0,
        0};
#endif
}


template MACELESKokkos<float>::~MACELESKokkos();
template MACELESKokkos<double>::~MACELESKokkos();
template int MACELESKokkos<float>::validate_les_position_count(std::size_t) const;
template int MACELESKokkos<double>::validate_les_position_count(std::size_t) const;
template void MACELESKokkos<float>::set_les_geometry(
    std::span<const double>, std::span<const double>, std::span<const int>);
template void MACELESKokkos<double>::set_les_geometry(
    std::span<const double>, std::span<const double>, std::span<const int>);
template void MACELESKokkos<float>::configure_les_particle_mesh_for_testing(
    std::int64_t, std::int64_t, std::int64_t, std::int64_t);
template void MACELESKokkos<double>::configure_les_particle_mesh_for_testing(
    std::int64_t, std::int64_t, std::int64_t, std::int64_t);
template void MACELESKokkos<float>::use_explicit_les_periodic_for_testing();
template void MACELESKokkos<double>::use_explicit_les_periodic_for_testing();
template void MACELESKokkos<float>::fence_les_particle_mesh_owner(const char*);
template void MACELESKokkos<double>::fence_les_particle_mesh_owner(const char*);
template std::tuple<bool, bool, std::string, std::uint32_t, std::uint32_t, std::uint64_t>
MACELESKokkos<float>::les_particle_mesh_stream_probe_for_testing();
template std::tuple<bool, bool, std::string, std::uint32_t, std::uint32_t, std::uint64_t>
MACELESKokkos<double>::les_particle_mesh_stream_probe_for_testing();
