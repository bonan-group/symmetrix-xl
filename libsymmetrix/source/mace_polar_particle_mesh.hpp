#ifndef SYMMETRIX_POLAR_PARTICLE_MESH_HPP
#define SYMMETRIX_POLAR_PARTICLE_MESH_HPP

// POLAR-owned storage boundary over the physics-neutral particle-mesh layer.
//
// This type declares the POLAR-1 source/receiver lane layout, owns atom-scale
// feature and adjoint buffers, and transactionally composes them with the
// shared mesh/stencil/FFT workspace. POLAR-specific filters, normalization,
// self terms, and reverse equations live here rather than in the neutral layer.

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <span>
#include <utility>
#include <vector>

#include "particle_mesh_transport.hpp"

namespace symmetrix::mace_polar {

struct PolarParticleMeshLayout {
    int density_max_l = 0;
    int feature_max_l = 0;
    int spin_channels = 0;
    std::int64_t receiver_widths = 0;
    std::int64_t source_components = 0;
    std::int64_t receiver_components = 0;
    std::int64_t source_lanes = 0;
    std::int64_t receiver_lanes = 0;
    std::int64_t transform_lanes = 0;
};

struct PolarParticleMeshReserveOutcome {
    bool rebuilt = false;
    std::uint64_t generation = 0;
};

struct PolarParticleMeshUsage {
    std::vector<particle_mesh::BufferUsage> buffers;
    std::uint64_t active_bytes = 0;
    std::uint64_t capacity_bytes = 0;
    std::uint64_t reservations = 0;
    std::uint64_t rebuilds = 0;
    std::uint64_t reuses = 0;
    std::uint64_t generation = 0;
    std::int64_t nodes = 0;
    std::array<std::int64_t, 3> mesh{0, 0, 0};
    std::int64_t spline_order = 0;
    std::uint64_t forward_plan_builds = 0;
    std::uint64_t forward_plan_reuses = 0;
    std::uint64_t inverse_plan_builds = 0;
    std::uint64_t inverse_plan_reuses = 0;
    std::uint64_t resident_scalar_validation_transfers = 0;
    std::uint64_t resident_completion_fences = 0;
    std::string resident_stencil_path;
    PolarParticleMeshLayout layout;
};

struct PolarParticleMeshForwardResult {
    std::vector<double> features;
    double energy = 0.0;
    std::int64_t nodes = 0;
    std::int64_t receiver_widths = 0;
    std::int64_t feature_dimension = 0;
};

struct PolarParticleMeshChannelsForwardResult {
    std::vector<double> features;
    std::vector<double> energies;
    std::int64_t nodes = 0;
    std::int64_t channels = 0;
    std::int64_t feature_dimension = 0;
};

struct PolarParticleMeshReverseResult {
    std::vector<double> source_multipole_adjoints;
    std::vector<double> position_adjoints;
    std::vector<double> cell_adjoints;
    std::int64_t nodes = 0;
    std::int64_t source_lanes = 0;
};

template <typename Precision>
class PolarParticleMeshWorkspaceT final {
public:
    using execution_space = Kokkos::DefaultExecutionSpace;
    using precision_type = Precision;
    using const_device_matrix =
        Kokkos::View<const Precision**, Kokkos::LayoutRight>;
    using device_matrix = Kokkos::View<Precision**, Kokkos::LayoutRight>;
    using const_device_vector = Kokkos::View<const double*>;
    using device_vector = Kokkos::View<double*>;
    using device_double_matrix =
        Kokkos::View<double**, Kokkos::LayoutRight>;
    using const_device_double_matrix =
        Kokkos::View<const double**, Kokkos::LayoutRight>;
    using complex_type = Kokkos::complex<Precision>;

    PolarParticleMeshWorkspaceT(
        execution_space execution_space,
        int density_max_l,
        int feature_max_l,
        int spin_channels,
        std::int64_t receiver_widths);

    PolarParticleMeshWorkspaceT(const PolarParticleMeshWorkspaceT&) = delete;
    PolarParticleMeshWorkspaceT& operator=(const PolarParticleMeshWorkspaceT&) = delete;
    PolarParticleMeshWorkspaceT(PolarParticleMeshWorkspaceT&&) = delete;
    PolarParticleMeshWorkspaceT& operator=(PolarParticleMeshWorkspaceT&&) = delete;

    [[nodiscard]] const PolarParticleMeshLayout& layout() const noexcept
    {
        return layout_;
    }

    PolarParticleMeshReserveOutcome reserve(
        std::int64_t nx,
        std::int64_t ny,
        std::int64_t nz,
        std::int64_t nodes,
        std::int64_t spline_order);

    [[nodiscard]] PolarParticleMeshUsage usage() const;
    [[nodiscard]] PolarParticleMeshForwardResult scalar_forward(
        std::span<const double> positions,
        std::span<const double> sources,
        const std::array<double, 9>& cell,
        const std::array<int, 3>& pbc,
        double source_sigma,
        std::span<const double> receiver_sigmas,
        const std::array<std::int64_t, 3>& mesh,
        std::int64_t spline_order,
        bool include_feature_self_interaction,
        bool include_energy_self_interaction);
    [[nodiscard]] PolarParticleMeshForwardResult forward(
        std::span<const double> positions,
        std::span<const double> source_multipoles,
        const std::array<double, 9>& cell,
        const std::array<int, 3>& pbc,
        double source_sigma,
        std::span<const double> receiver_sigmas,
        const std::array<std::int64_t, 3>& mesh,
        std::int64_t spline_order,
        bool include_feature_self_interaction,
        bool include_energy_self_interaction);
    [[nodiscard]] PolarParticleMeshChannelsForwardResult forward_channels(
        std::span<const double> positions,
        std::span<const double> source_multipoles,
        const std::array<double, 9>& cell,
        const std::array<int, 3>& pbc,
        double source_sigma,
        std::span<const double> receiver_sigmas,
        const std::array<std::int64_t, 3>& mesh,
        std::int64_t spline_order,
        bool include_feature_self_interaction,
        bool include_energy_self_interaction);
    [[nodiscard]] PolarParticleMeshReverseResult reverse_channels(
        std::span<const double> positions,
        std::span<const double> source_multipoles,
        std::span<const double> feature_seeds,
        std::span<const double> energy_seeds,
        const std::array<double, 9>& cell,
        const std::array<int, 3>& pbc,
        double source_sigma,
        std::span<const double> receiver_sigmas,
        const std::array<std::int64_t, 3>& mesh,
        std::int64_t spline_order,
        bool include_feature_self_interaction,
        bool include_energy_self_interaction);
    void forward_channels_device(
        std::span<const double> positions,
        const const_device_matrix& source_multipoles,
        const device_matrix& receiver_features,
        const device_vector& channel_energies,
        const std::array<double, 9>& cell,
        const std::array<int, 3>& pbc,
        double source_sigma,
        std::span<const double> receiver_sigmas,
        const std::array<std::int64_t, 3>& mesh,
        std::int64_t spline_order,
        bool include_feature_self_interaction,
        bool include_energy_self_interaction,
        std::int64_t active_channels,
        bool compute_features = true);
    void forward_channels_device(
        const const_device_double_matrix& positions,
        const const_device_matrix& source_multipoles,
        const device_matrix& receiver_features,
        const device_vector& channel_energies,
        const std::array<double, 9>& cell,
        const std::array<int, 3>& pbc,
        double source_sigma,
        std::span<const double> receiver_sigmas,
        const std::array<std::int64_t, 3>& mesh,
        std::int64_t spline_order,
        bool include_feature_self_interaction,
        bool include_energy_self_interaction,
        std::int64_t active_channels,
        bool compute_features = true);
    void reverse_channels_device(
        std::span<const double> positions,
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
        double source_sigma,
        std::span<const double> receiver_sigmas,
        const std::array<std::int64_t, 3>& mesh,
        std::int64_t spline_order,
        bool include_feature_self_interaction,
        bool include_energy_self_interaction,
        std::int64_t active_channels,
        bool feature_seed_active = true,
        bool energy_seed_active = true);
    void reverse_channels_device(
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
        double source_sigma,
        std::span<const double> receiver_sigmas,
        const std::array<std::int64_t, 3>& mesh,
        std::int64_t spline_order,
        bool include_feature_self_interaction,
        bool include_energy_self_interaction,
        std::int64_t active_channels,
        bool feature_seed_active = true,
        bool energy_seed_active = true);
    [[nodiscard]] particle_mesh::ParticleMeshWorkspace<Precision>& neutral_workspace()
        noexcept
    {
        return neutral_;
    }
    [[nodiscard]] const particle_mesh::ParticleMeshWorkspace<Precision>&
    neutral_workspace() const noexcept
    {
        return neutral_;
    }

    // NVCC extended device lambdas require their enclosing member to be
    // publicly accessible. This remains an implementation seam; callers use
    // the checked host or resident entry points above.
    [[nodiscard]] PolarParticleMeshChannelsForwardResult forward_impl(
        std::span<const double> positions,
        std::span<const double> source_multipoles,
        std::int64_t active_channels,
        const std::array<double, 9>& cell,
        const std::array<int, 3>& pbc,
        double source_sigma,
        std::span<const double> receiver_sigmas,
        const std::array<std::int64_t, 3>& mesh,
        std::int64_t spline_order,
        bool include_feature_self_interaction,
        bool include_energy_self_interaction,
        const const_device_matrix* resident_sources = nullptr,
        const device_matrix* resident_features = nullptr,
        const device_vector* resident_energies = nullptr,
        const const_device_double_matrix* resident_positions = nullptr,
        Kokkos::View<complex_type*>* source_spectrum_snapshot = nullptr,
        bool compute_features = true);
    [[nodiscard]] PolarParticleMeshReverseResult reverse_impl(
        std::span<const double> positions,
        std::span<const double> source_multipoles,
        std::span<const double> feature_seeds,
        std::span<const double> energy_seeds,
        const std::array<double, 9>& cell,
        const std::array<int, 3>& pbc,
        double source_sigma,
        std::span<const double> receiver_sigmas,
        const std::array<std::int64_t, 3>& mesh,
        std::int64_t spline_order,
        bool include_feature_self_interaction,
        bool include_energy_self_interaction,
        std::int64_t active_channels,
        const const_device_matrix* resident_sources = nullptr,
        const const_device_matrix* resident_feature_seeds = nullptr,
        const const_device_vector* resident_energy_seeds = nullptr,
        const device_matrix* resident_forward_features = nullptr,
        const device_vector* resident_forward_energies = nullptr,
        const device_matrix* resident_source_adjoints = nullptr,
        const device_double_matrix* resident_position_adjoints = nullptr,
        const device_vector* resident_cell_adjoints = nullptr,
        const const_device_double_matrix* resident_positions = nullptr,
        bool feature_seed_active = true,
        bool energy_seed_active = true);

private:
    struct Key {
        std::array<std::int64_t, 3> mesh{};
        std::int64_t nodes = 0;
        std::int64_t spline_order = 0;
        friend bool operator==(const Key&, const Key&) = default;
    };

    template <typename Value>
    static Kokkos::View<Value*> replacement(
        const Kokkos::View<Value*>& current,
        std::int64_t required,
        const char* label,
        const char* fault_name)
    {
        if (required <= static_cast<std::int64_t>(current.extent(0))) {
            return {};
        }
        const auto extent =
            particle_mesh::detail::checked_flat_index(required, fault_name);
        particle_mesh::detail::maybe_fail_allocation(fault_name);
        return Kokkos::View<Value*>(
            std::string(label), static_cast<std::size_t>(extent));
    }

    template <typename Value>
    static void install_if_grown(
        Kokkos::View<Value*>& current, Kokkos::View<Value*>& candidate)
    {
        if (candidate.extent(0) != 0) {
            current = std::move(candidate);
        }
    }

    execution_space execution_space_;
    PolarParticleMeshLayout layout_;
    particle_mesh::ParticleMeshWorkspace<Precision> neutral_;
    std::optional<Key> key_;
    std::int64_t active_nodes_ = 0;
    std::uint64_t reservations_ = 0;
    std::uint64_t rebuilds_ = 0;
    std::uint64_t reuses_ = 0;
    std::uint64_t generation_ = 0;
    std::uint64_t allocation_serial_ = 0;
    std::uint64_t resident_scalar_validation_transfers_ = 0;
    std::uint64_t resident_completion_fences_ = 0;
    std::string resident_stencil_path_;
    // POLAR-owned buffers followed by the six neutral stencil/grid buffers.
    std::array<std::uint64_t, 18> allocation_ids_{};

    Kokkos::View<double*> positions_;
    Kokkos::View<Precision*> source_multipoles_;
    Kokkos::View<Precision*> receiver_features_;
    Kokkos::View<Precision*> transform_weights_;
    Kokkos::View<Precision*> transform_position_forces_;
    Kokkos::View<Precision*> source_multipole_adjoints_;
    Kokkos::View<double*> position_adjoints_;
    Kokkos::View<double*> cell_adjoints_;
    Kokkos::View<double*> receiver_sigmas_;
    Kokkos::View<double*> channel_energies_;
    Kokkos::View<complex_type*> source_spectrum_snapshot_;
    Kokkos::View<double*> cell_bin_adjoints_;
};

template <typename Precision>
PolarParticleMeshWorkspaceT<Precision>::PolarParticleMeshWorkspaceT(
    execution_space execution_space,
    const int density_max_l,
    const int feature_max_l,
    const int spin_channels,
    const std::int64_t receiver_widths)
    : execution_space_(std::move(execution_space)),
      neutral_(particle_mesh::MeshLayout::right, [&] {
          if (density_max_l < 0 || density_max_l > 1) {
              throw std::invalid_argument(
                  "POLAR particle-mesh density_max_l must be 0 or 1.");
          }
          if (feature_max_l < 0 || feature_max_l > 1) {
              throw std::invalid_argument(
                  "POLAR particle-mesh feature_max_l must be 0 or 1.");
          }
          if (spin_channels != 2) {
              throw std::invalid_argument(
                  "POLAR particle-mesh spin_channels must be 2.");
          }
          if (receiver_widths <= 0) {
              throw std::invalid_argument(
                  "POLAR particle-mesh receiver_widths must be positive.");
          }
          layout_.density_max_l = density_max_l;
          layout_.feature_max_l = feature_max_l;
          layout_.spin_channels = spin_channels;
          layout_.receiver_widths = receiver_widths;
          layout_.source_components = static_cast<std::int64_t>(density_max_l + 1)
              * static_cast<std::int64_t>(density_max_l + 1);
          layout_.receiver_components = static_cast<std::int64_t>(feature_max_l + 1)
              * static_cast<std::int64_t>(feature_max_l + 1);
          layout_.source_lanes = particle_mesh::detail::checked_product(
              spin_channels, layout_.source_components,
              "POLAR particle-mesh source lanes");
          const std::int64_t receiver_channels =
              particle_mesh::detail::checked_product(
                  spin_channels, receiver_widths,
                  "POLAR particle-mesh spin/receiver-width lanes");
          layout_.receiver_lanes = particle_mesh::detail::checked_product(
              receiver_channels, layout_.receiver_components,
              "POLAR particle-mesh receiver lanes");
          layout_.transform_lanes =
              std::max(layout_.source_lanes, layout_.receiver_lanes);
          return layout_.transform_lanes;
      }(), execution_space_)
{
    receiver_sigmas_ = Kokkos::View<double*>(
        "symmetrix_polar_pme_receiver_sigmas", layout_.receiver_widths);
    allocation_ids_[14] = ++allocation_serial_;
    channel_energies_ = Kokkos::View<double*>(
        "symmetrix_polar_pme_channel_energies", layout_.spin_channels);
    allocation_ids_[15] = ++allocation_serial_;
}

template <typename Precision>
PolarParticleMeshReserveOutcome PolarParticleMeshWorkspaceT<Precision>::reserve(
    const std::int64_t nx,
    const std::int64_t ny,
    const std::int64_t nz,
    const std::int64_t nodes,
    const std::int64_t spline_order)
{
    if (nodes < 0) {
        throw std::invalid_argument(
            "POLAR particle-mesh node count must not be negative.");
    }
    if (nx < spline_order || ny < spline_order || nz < spline_order) {
        throw std::invalid_argument(
            "POLAR particle-mesh mesh axes must be at least the spline order.");
    }
    const Key requested{{nx, ny, nz}, nodes, spline_order};
    if (key_ && *key_ == requested) {
        ++reservations_;
        ++reuses_;
        return {false, generation_};
    }

    const std::int64_t coordinates = particle_mesh::detail::checked_product(
        nodes, 3, "POLAR particle-mesh position coordinates");
    const std::int64_t source_elements = particle_mesh::detail::checked_product(
        nodes, layout_.source_lanes, "POLAR particle-mesh source elements");
    // Gather always writes the neutral transform batch, including padding when
    // source lanes outnumber active receiver lanes.
    const std::int64_t receiver_elements = particle_mesh::detail::checked_product(
        nodes, layout_.transform_lanes, "POLAR particle-mesh receiver elements");
    const particle_mesh::MeshShape requested_shape(
        nx, ny, nz, layout_.transform_lanes,
        particle_mesh::MeshLayout::right);
    const std::int64_t cell_bin_elements =
        particle_mesh::detail::checked_product(
            requested_shape.spectrum_point_count(), 9,
            "POLAR particle-mesh cell-bin adjoints");
    // Reject invalid mesh/order/stencil ranges before any outer allocation.
    neutral_.validate_reserve(
        nx, ny, nz, nodes, layout_.transform_lanes, spline_order);

    auto positions = replacement(
        positions_, coordinates, "symmetrix_polar_pme_positions",
        "POLAR particle-mesh positions");
    auto source = replacement(
        source_multipoles_, source_elements, "symmetrix_polar_pme_sources",
        "POLAR particle-mesh source multipoles");
    auto receiver = replacement(
        receiver_features_, receiver_elements, "symmetrix_polar_pme_receiver_features",
        "POLAR particle-mesh receiver features");
    const std::int64_t transform_elements = particle_mesh::detail::checked_product(
        nodes, layout_.transform_lanes, "POLAR particle-mesh transform weights");
    auto transform_weights = replacement(
        transform_weights_, transform_elements,
        "symmetrix_polar_pme_transform_weights",
        "POLAR particle-mesh transform weights");
    auto transform_position_forces = replacement(
        transform_position_forces_, coordinates,
        "symmetrix_polar_pme_transform_position_forces",
        "POLAR particle-mesh transform position forces");
    auto source_adjoints = replacement(
        source_multipole_adjoints_, source_elements,
        "symmetrix_polar_pme_source_adjoints",
        "POLAR particle-mesh source multipole adjoints");
    auto position_adjoints = replacement(
        position_adjoints_, coordinates, "symmetrix_polar_pme_position_adjoints",
        "POLAR particle-mesh position adjoints");
    auto cell_adjoints = replacement(
        cell_adjoints_, 9, "symmetrix_polar_pme_cell_adjoints",
        "POLAR particle-mesh cell adjoints");
    auto cell_bin_adjoints = replacement(cell_bin_adjoints_,
        cell_bin_elements, "symmetrix_polar_pme_cell_bin_adjoints",
        "POLAR particle-mesh cell-bin adjoints");
    auto allocation_ids = allocation_ids_;
    auto allocation_serial = allocation_serial_;
    const auto mark_replacement = [&](const std::size_t index, const auto& candidate) {
        if (candidate.extent(0) != 0) {
            allocation_ids[index] = ++allocation_serial;
        }
    };
    mark_replacement(0, positions);
    mark_replacement(1, source);
    mark_replacement(2, receiver);
    mark_replacement(3, transform_weights);
    mark_replacement(4, transform_position_forces);
    mark_replacement(5, source_adjoints);
    mark_replacement(6, position_adjoints);
    mark_replacement(7, cell_adjoints);
    mark_replacement(17, cell_bin_adjoints);
    auto neutral = neutral_.plan_reserve(
        nx, ny, nz, nodes, layout_.transform_lanes, spline_order);
    mark_replacement(8, neutral.stencil_base);
    mark_replacement(9, neutral.stencil_weight);
    mark_replacement(10, neutral.stencil_slope);
    mark_replacement(11, neutral.k_norm);
    mark_replacement(12, neutral.deconvolution);
    mark_replacement(13, neutral.hermitian);

    neutral_.commit(std::move(neutral));
    install_if_grown(positions_, positions);
    install_if_grown(source_multipoles_, source);
    install_if_grown(receiver_features_, receiver);
    install_if_grown(transform_weights_, transform_weights);
    install_if_grown(transform_position_forces_, transform_position_forces);
    install_if_grown(source_multipole_adjoints_, source_adjoints);
    install_if_grown(position_adjoints_, position_adjoints);
    install_if_grown(cell_adjoints_, cell_adjoints);
    install_if_grown(cell_bin_adjoints_, cell_bin_adjoints);
    allocation_serial_ = allocation_serial;
    allocation_ids_ = allocation_ids;
    key_ = requested;
    active_nodes_ = nodes;
    ++reservations_;
    ++rebuilds_;
    ++generation_;
    return {true, generation_};
}

template <typename Precision>
PolarParticleMeshUsage PolarParticleMeshWorkspaceT<Precision>::usage() const
{
    PolarParticleMeshUsage result;
    result.reservations = reservations_;
    result.rebuilds = rebuilds_;
    result.reuses = reuses_;
    result.generation = generation_;
    result.nodes = active_nodes_;
    result.layout = layout_;
    if (key_) {
        result.mesh = key_->mesh;
        result.spline_order = key_->spline_order;
    }
    const auto add = [&](const char* name, const std::int64_t active,
                         const std::int64_t capacity, const std::uint64_t bytes,
                         const std::uint64_t allocation_id) {
        result.buffers.push_back(particle_mesh::BufferUsage{
            name, active, capacity,
            particle_mesh::detail::element_bytes(active, bytes),
            particle_mesh::detail::element_bytes(capacity, bytes), allocation_id});
    };
    const std::int64_t coordinates = particle_mesh::detail::checked_product(
        active_nodes_, 3, "POLAR particle-mesh active position coordinates");
    const std::int64_t source = particle_mesh::detail::checked_product(
        active_nodes_, layout_.source_lanes,
        "POLAR particle-mesh active source elements");
    const std::int64_t receiver = particle_mesh::detail::checked_product(
        active_nodes_, layout_.transform_lanes,
        "POLAR particle-mesh active receiver elements");
    const std::int64_t transform = particle_mesh::detail::checked_product(
        active_nodes_, layout_.transform_lanes,
        "POLAR particle-mesh active transform elements");
    add("positions", coordinates, positions_.extent(0), sizeof(double), allocation_ids_[0]);
    add("source_multipoles", source, source_multipoles_.extent(0), sizeof(Precision),
        allocation_ids_[1]);
    add("receiver_features", receiver, receiver_features_.extent(0), sizeof(Precision),
        allocation_ids_[2]);
    add("transform_weights", transform, transform_weights_.extent(0),
        sizeof(Precision), allocation_ids_[3]);
    add("transform_position_forces", coordinates,
        transform_position_forces_.extent(0), sizeof(Precision), allocation_ids_[4]);
    add("source_multipole_adjoints", source,
        source_multipole_adjoints_.extent(0), sizeof(Precision), allocation_ids_[5]);
    add("position_adjoints", coordinates, position_adjoints_.extent(0), sizeof(double),
        allocation_ids_[6]);
    add("cell_adjoints", key_ ? 9 : 0, cell_adjoints_.extent(0), sizeof(double),
        allocation_ids_[7]);
    add("receiver_sigmas", layout_.receiver_widths, receiver_sigmas_.extent(0),
        sizeof(double), allocation_ids_[14]);
    add("channel_energies", layout_.spin_channels, channel_energies_.extent(0),
        sizeof(double), allocation_ids_[15]);

    if (key_) {
        const auto mesh_usage = neutral_.storage().usage();
        for (auto item : mesh_usage.buffers) {
            result.buffers.push_back(std::move(item));
        }
        const std::int64_t offsets = particle_mesh::detail::checked_product(
            active_nodes_, 3, "POLAR particle-mesh active stencil offsets");
        const std::int64_t stencil = particle_mesh::detail::checked_product(
            offsets, key_->spline_order,
            "POLAR particle-mesh active stencil elements");
        const std::int64_t bins = neutral_.reciprocal_grid().bins;
        const std::int64_t cell_bins =
            particle_mesh::detail::checked_product(bins, 9,
                "POLAR particle-mesh active cell-bin adjoints");
        add("cell_bin_adjoints", cell_bins, cell_bin_adjoints_.extent(0),
            sizeof(double), allocation_ids_[17]);
        add("stencil_base", offsets, neutral_.stencil().base().extent(0),
            sizeof(std::int32_t), allocation_ids_[8]);
        add("stencil_weight", stencil, neutral_.stencil().weights().extent(0),
            sizeof(Precision), allocation_ids_[9]);
        add("stencil_slope", stencil, neutral_.stencil().slopes().extent(0),
            sizeof(Precision), allocation_ids_[10]);
        add("k_norm", bins, neutral_.reciprocal_grid().k_norm.extent(0), sizeof(double),
            allocation_ids_[11]);
        add("deconvolution", bins,
            neutral_.reciprocal_grid().deconvolution.extent(0), sizeof(Precision),
            allocation_ids_[12]);
        add("hermitian", bins,
            neutral_.reciprocal_grid().hermitian_weight.extent(0),
            sizeof(Precision), allocation_ids_[13]);
        if (source_spectrum_snapshot_.extent(0) != 0) {
            const std::int64_t snapshot = particle_mesh::detail::checked_product(
                layout_.source_lanes, bins,
                "POLAR particle-mesh active source spectrum snapshot");
            add("source_spectrum_snapshot", snapshot,
                source_spectrum_snapshot_.extent(0), sizeof(complex_type),
                allocation_ids_[16]);
        }
    }
    const auto forward = neutral_.plans().usage(
        particle_mesh::FftPlanRole::forward_r2c);
    const auto inverse = neutral_.plans().usage(
        particle_mesh::FftPlanRole::inverse_c2r);
    result.forward_plan_builds = forward.builds;
    result.forward_plan_reuses = forward.reuses;
    result.inverse_plan_builds = inverse.builds;
    result.inverse_plan_reuses = inverse.reuses;
    result.resident_scalar_validation_transfers =
        resident_scalar_validation_transfers_;
    result.resident_completion_fences = resident_completion_fences_;
    result.resident_stencil_path = resident_stencil_path_;
    particle_mesh::detail::finalize_usage(
        result.buffers, result.active_bytes, result.capacity_bytes);
    return result;
}

extern template class PolarParticleMeshWorkspaceT<float>;
extern template class PolarParticleMeshWorkspaceT<double>;

}  // namespace symmetrix::mace_polar

#endif  // SYMMETRIX_POLAR_PARTICLE_MESH_HPP
