#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <numbers>
#include <numeric>
#include <set>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <type_traits>

#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime_api.h>
#elif defined(KOKKOS_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

// TODO: remove some of these headers?
#include "KokkosBatched_Util.hpp"
#include "KokkosBlas.hpp"
#include "KokkosBatched_Gemm_Decl.hpp"
#include "KokkosBlas_tpl_spec.hpp"
#if defined(KOKKOS_ENABLE_HIP) \
    && defined(KOKKOSKERNELS_ENABLE_TPL_ROCBLAS)
#include <rocblas/rocblas.h>
#endif
#include "nlohmann/json.hpp"
#include "sphericart.hpp"
#include "sphericart_cuda.hpp"
#include "spherical_harmonic_device.hpp"

#include "tools_kokkos.hpp"
#include "mace_kokkos.hpp"
#include "device_backend.hpp"
#include "kernel_launch_profile.hpp"
#include "standard_m0.hpp"
#include "standard_m1.hpp"
#include "standard_r0.hpp"
#include "jit_generation_version.hpp"
#include "factorized_blas.hpp"

using Kokkos::ALL;
using Kokkos::LayoutRight;
using Kokkos::make_pair;
using Kokkos::MemoryUnmanaged;
using Kokkos::parallel_for;
using Kokkos::PerTeam;
using Kokkos::subview;
using Kokkos::TeamPolicy;
using Kokkos::TeamVectorRange;
using Kokkos::TeamVectorMDRange;
using Kokkos::View;
#include "mace_kokkos_factorized_blas_detail.hpp"

#include "mace_kokkos_kernel_launch_detail.hpp"

#include "mace_kokkos_spherical_harmonics_detail.hpp"

#ifdef KOKKOS_ENABLE_CUDA
using ExecutionDeviceBackend = DeviceBackendTraits<Kokkos::Cuda>;
#elif defined(KOKKOS_ENABLE_HIP)
using ExecutionDeviceBackend = DeviceBackendTraits<Kokkos::HIP>;
#endif

namespace {

std::size_t checked_extent_product(
    const std::string_view name,
    const std::initializer_list<std::size_t> factors)
{
    std::size_t product = 1;
    for (const std::size_t factor : factors) {
        if (factor != 0
            && product > std::numeric_limits<std::size_t>::max()/factor)
            throw std::length_error(
                "MACE graph "+std::string(name)+" extent overflows size_t.");
        product *= factor;
    }
    return product;
}

std::size_t checked_extent_sum(
    const std::string_view name,
    const std::initializer_list<std::size_t> terms)
{
    std::size_t total = 0;
    for (const std::size_t term : terms) {
        if (total > std::numeric_limits<std::size_t>::max()-term)
            throw std::length_error(
                "MACE graph "+std::string(name)+" byte estimate overflows size_t.");
        total += term;
    }
    return total;
}

std::size_t checked_extent_align_up(
    const std::string_view name,
    const std::size_t value,
    const std::size_t alignment)
{
    if (alignment == 0)
        throw std::logic_error("MACE graph alignment is zero.");
    const std::size_t remainder = value%alignment;
    return remainder == 0 ? value : checked_extent_sum(
        name, {value, alignment-remainder});
}

void require_int_launch_extent(
    const std::string_view name,
    const std::initializer_list<std::size_t> factors)
{
    const std::size_t extent = checked_extent_product(name, factors);
    if (extent > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::length_error(
            "MACE graph "+std::string(name)+" launch extent "+
            std::to_string(extent)+" exceeds the 32-bit execution-range limit "+
            std::to_string(std::numeric_limits<int>::max())+".");
}

template <typename Precision, typename PackedView>
void capture_packed_execution_tile(
    std::vector<Precision>& destination,
    const int receiver_begin,
    const int receiver_count,
    const int components,
    const int num_eta,
    const int embedding,
    const int channels,
    PackedView packed)
{
    const auto host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), packed);
    for (int local_receiver=0; local_receiver<receiver_count; ++local_receiver)
        for (int component=0; component<components; ++component)
            for (int eta=0; eta<num_eta; ++eta)
                for (int q=0; q<embedding; ++q)
                    for (int channel=0; channel<channels; ++channel) {
                        const std::size_t index =
                            (((static_cast<std::size_t>(
                                    receiver_begin+local_receiver)
                                    *components+component)
                                *num_eta+eta)
                              *embedding+q)
                             *channels+channel;
                        const int column =
                            (eta*components+component)*channels+channel;
                        destination[index] = host(local_receiver,q,column);
                    }
}

template <typename Precision, typename StateView>
void capture_matrix_execution_tile(
    std::vector<Precision>& destination,
    const int receiver_begin,
    const int receiver_count,
    const int components,
    const int num_eta,
    const int embedding,
    const int channels,
    StateView state)
{
    const auto host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), state);
    for (int local_receiver=0; local_receiver<receiver_count; ++local_receiver)
        for (int component=0; component<components; ++component)
            for (int eta=0; eta<num_eta; ++eta)
                for (int q=0; q<embedding; ++q)
                    for (int channel=0; channel<channels; ++channel) {
                        const std::size_t index =
                            (((static_cast<std::size_t>(
                                    receiver_begin+local_receiver)
                                    *components+component)
                                *num_eta+eta)
                              *embedding+q)
                             *channels+channel;
                        const int row = local_receiver*components+component;
                        const int column =
                            (eta*embedding+q)*channels+channel;
                        destination[index] = host(row,column);
                    }
}

}

template <typename Precision>
void MACEKokkos<Precision>::validate_graph_cardinality(
    const std::size_t num_receivers,
    const std::size_t num_feature_nodes,
    const std::size_t num_edges) const
{
    constexpr std::size_t int_max = static_cast<std::size_t>(
        std::numeric_limits<int>::max());
    if (num_receivers >= int_max)
        throw std::length_error(
            "MACE graph atom count "+std::to_string(num_receivers)+
            " exceeds the 32-bit CSR limit "+std::to_string(int_max-1)+".");
    if (num_feature_nodes >= int_max)
        throw std::length_error(
            "MACE graph feature-node count "+std::to_string(num_feature_nodes)+
            " exceeds the 32-bit CSR limit "+std::to_string(int_max-1)+".");
    if (num_feature_nodes < num_receivers)
        throw std::invalid_argument(
            "MACE graph feature-node count is smaller than its receiver count.");
    if (num_edges > int_max)
        throw std::length_error(
            "MACE graph directed-edge count "+std::to_string(num_edges)+
            " exceeds the 32-bit topology limit "+std::to_string(int_max)+".");
    if (num_channels <= 0 || l_max < 0 || num_lm <= 0 || num_LM <= 0
        || (!single_layer_readout
            && (num_lelm1lm2 <= 0 || num_lme <= 0)))
        throw std::logic_error(
            "MACE graph cardinality validation requires positive model dimensions.");

    const std::size_t channels = static_cast<std::size_t>(num_channels);
    const std::size_t edge_harmonics = static_cast<std::size_t>(num_lm);
    const std::size_t output_harmonics = static_cast<std::size_t>(num_LM);
    const std::size_t radial_bands = static_cast<std::size_t>(l_max)+1;
    const std::size_t paths = Phi1_l.extent(0);
    const std::size_t coupled_rows = static_cast<std::size_t>(num_lelm1lm2);
    const std::size_t coupled_outputs = static_cast<std::size_t>(num_lme);

    checked_extent_product(
        "edge-coordinate", {num_edges, std::size_t(3), sizeof(double)});
    checked_extent_product(
        "harmonic-value",
        {num_edges, edge_harmonics, sizeof(Precision)});
    checked_extent_product(
        "harmonic-gradient",
        {num_edges, std::size_t(3), edge_harmonics, sizeof(Precision)});
    checked_extent_product(
        "R0 edge-channel",
        {num_edges, radial_bands, channels, sizeof(Precision)});
    checked_extent_product(
        "R1 edge-channel",
        {num_edges, paths, channels, sizeof(Precision)});
    checked_extent_product(
        "receiver input-state",
        {num_receivers, edge_harmonics, channels, sizeof(Precision)});
    checked_extent_product(
        "receiver output-state",
        {num_receivers, output_harmonics, channels, sizeof(Precision)});
    checked_extent_product(
        "receiver coupled-row",
        {num_receivers, coupled_rows, channels, sizeof(Precision)});
    checked_extent_product(
        "receiver coupled-output",
        {num_receivers, coupled_outputs, channels, sizeof(Precision)});
    checked_extent_product(
        "feature-node state",
        {num_feature_nodes, output_harmonics, channels, sizeof(Precision)});

    require_int_launch_extent(
        "receiver-channel", {num_receivers, channels});
    require_int_launch_extent(
        "feature-node-channel", {num_feature_nodes, channels});
    require_int_launch_extent(
        "receiver-input-component", {num_receivers, edge_harmonics});
    require_int_launch_extent(
        "receiver-output-component", {num_receivers, output_harmonics});
    require_int_launch_extent(
        "receiver-coupled-row", {num_receivers, coupled_rows});
}

template <typename Precision>
MACEKokkos<Precision>::MACEKokkos(
    std::string filename, std::string requested_head)
{
    factorized_blas_context = std::make_unique<FactorizedBlasContext>();
    load_from_json(filename, requested_head);
    set_m1_polynomial_policy("automatic");
    // Match the Python calculator: compact models use direct execution by
    // default.  Direct still fails closed until its required RTC artifact is
    // loaded; callers can explicitly request non-compiled/materialized modes.
    if (supports_streamed_edges())
        streamed_edges = MACEStreamedEdgesMode::direct;
}

template <typename Precision>
void MACEKokkos<Precision>::ensure_spherical_harmonics_state()
{
    if (spherical_harmonics_state)
        return;
    spherical_harmonics_state =
        std::make_unique<SphericalHarmonicsState>(l_max);
    execution_sphericart_initialization_count += 1;
}

template <typename Precision>
void MACEKokkos<Precision>::reserve_execution_geometry_workspace(
    const int num_edges)
{
    if (num_edges < 0)
        throw std::invalid_argument(
            "Execution geometry workspace requires a non-negative edge count.");
    ensure_spherical_harmonics_state();
    const bool tiled_workspace = single_layer_tiled_plan_active
        || dual_layer_tiled_plan_active;
    const std::size_t tiled_edge_capacity = static_cast<std::size_t>(
        dual_layer_tiled_plan_active
            ? dual_layer_workspace_planned_edge_capacity
            : single_layer_workspace_planned_edge_capacity);
    const std::size_t required_harmonic_edges = tiled_workspace
        ? tiled_edge_capacity
        : static_cast<std::size_t>(num_edges);
    if (num_edges <= execution_geometry_capacity_edges
        && Y.extent(0) >= required_harmonic_edges*static_cast<std::size_t>(num_lm)
        && (!tiled_workspace
            || (execution_prepared_unit_direction.extent(0)
                    >= 3*tiled_edge_capacity
                && execution_prepared_r.extent(0) >= tiled_edge_capacity
                && (dual_layer_tiled_plan_active
                    ? dual_layer_workspace_neigh_types.extent(0)
                    : single_layer_workspace_neigh_types.extent(0))
                    >= tiled_edge_capacity)))
        return;

    const int grown_capacity = execution_geometry_capacity_edges
            > std::numeric_limits<int>::max()/2
        ? std::numeric_limits<int>::max()
        : 2*execution_geometry_capacity_edges;
    const int planned_capacity = execution_geometry_planned_capacity_edges
            >= num_edges
        ? execution_geometry_planned_capacity_edges
        : std::max(num_edges, std::max(1, grown_capacity));
    const int capacity = std::max(num_edges, planned_capacity);
    const std::size_t edge_capacity = static_cast<std::size_t>(capacity);
    const std::size_t geometry_edge_capacity = tiled_workspace
        ? tiled_edge_capacity : edge_capacity;
    const std::size_t harmonic_edge_capacity = tiled_workspace
        ? tiled_edge_capacity
        : edge_capacity;
    const std::size_t harmonic_capacity =
        harmonic_edge_capacity*static_cast<std::size_t>(num_lm);
    const bool y_only_direct = use_y_only_direct_harmonics();
    factorized_execution_space.fence(
        "Replace execution geometry workspace capacity");
    Y_grad_shuffled = decltype(Y_grad_shuffled)();
    Y_grad = decltype(Y_grad)();
    Y = decltype(Y)();
    xyz_shuffled = decltype(xyz_shuffled)();
    execution_prepared_xyz = decltype(execution_prepared_xyz)();
    execution_prepared_unit_direction =
        decltype(execution_prepared_unit_direction)();
    execution_prepared_r = decltype(execution_prepared_r)();
    single_layer_workspace_neigh_types = {};
    dual_layer_workspace_neigh_types = {};
    execution_geometry_capacity_edges = 0;
    if (edge_geometry_policy == EdgeGeometryPolicy::unit_f32_radius_f64) {
        execution_prepared_unit_direction =
            decltype(execution_prepared_unit_direction)(
                Kokkos::view_alloc(
                    "Execution prepared unit direction",
                    Kokkos::WithoutInitializing),
                3*geometry_edge_capacity);
    } else {
        execution_prepared_xyz = decltype(execution_prepared_xyz)(
            Kokkos::view_alloc(
                "Execution prepared xyz", Kokkos::WithoutInitializing),
            3*edge_capacity);
    }
    execution_prepared_r = decltype(execution_prepared_r)(
            Kokkos::view_alloc(
                "Execution prepared radii", Kokkos::WithoutInitializing),
        geometry_edge_capacity);
    if (tiled_workspace) {
        const std::string workspace_label = dual_layer_tiled_plan_active
            ? "Dual-layer tiled neighbor types"
            : "Single-layer tiled neighbor types";
        auto workspace_neigh_types = Kokkos::View<int*>(
            Kokkos::view_alloc(
                workspace_label,
                Kokkos::WithoutInitializing),
            tiled_edge_capacity);
        if (dual_layer_tiled_plan_active)
            dual_layer_workspace_neigh_types = workspace_neigh_types;
        else
            single_layer_workspace_neigh_types = workspace_neigh_types;
    }
    if (!y_only_direct)
        xyz_shuffled = decltype(xyz_shuffled)(
            Kokkos::view_alloc(
                "Execution shuffled xyz", Kokkos::WithoutInitializing),
            3*edge_capacity);
    Y = decltype(Y)(
        Kokkos::view_alloc(
            "Execution spherical harmonics", Kokkos::WithoutInitializing),
        harmonic_capacity);
    if (!y_only_direct) {
        Y_grad = decltype(Y_grad)(
            Kokkos::view_alloc(
                "Execution spherical harmonic gradients",
                Kokkos::WithoutInitializing),
            3*harmonic_capacity);
        if (use_mh0_adjoint_reuse())
            Y_grad_shuffled = Y_grad;
        else
            Y_grad_shuffled = decltype(Y_grad_shuffled)(
                Kokkos::view_alloc(
                    "Execution raw spherical harmonic gradients",
                    Kokkos::WithoutInitializing),
                3*harmonic_capacity);
    }
    execution_geometry_capacity_edges = capacity;
    execution_geometry_planned_capacity_edges = capacity;
    execution_geometry_allocation_count += 1;
}

template <typename Precision>
void MACEKokkos<Precision>::ensure_execution_result_capacity(
    const int num_receivers,
    const int num_feature_nodes,
    const int num_edges)
{
    if (num_receivers < 0 || num_feature_nodes < num_receivers || num_edges < 0)
        throw std::invalid_argument(
            "Execution result capacity requires valid graph extents.");
    const std::size_t receiver_capacity = static_cast<std::size_t>(std::max(
        num_receivers, execution_planned_receivers));
    const std::size_t feature_capacity = static_cast<std::size_t>(std::max(
        num_feature_nodes, execution_planned_feature_nodes));
    const bool tiled_workspace = single_layer_tiled_plan_active
        || dual_layer_tiled_plan_active;
    const std::size_t edge_capacity = tiled_workspace
        ? static_cast<std::size_t>(dual_layer_tiled_plan_active
            ? dual_layer_workspace_planned_edge_capacity
            : single_layer_workspace_planned_edge_capacity)
        : static_cast<std::size_t>(std::max(num_edges, execution_planned_edges));
    const std::size_t force_capacity = checked_extent_product(
        "directed-force capacity", {std::size_t(3), edge_capacity});
    const bool fits = node_energies_storage.extent(0) >= receiver_capacity
        && node_forces_storage.extent(0) >= force_capacity
        && (!tiled_workspace
            || (atom_forces.extent(0) >= checked_extent_product(
                    "atom-force capacity", {std::size_t(3), feature_capacity})
                && (dual_layer_tiled_plan_active
                    ? dual_layer_workspace_virial.extent(0)
                    : single_layer_workspace_virial.extent(0)) >= 9
                && stress_tensor.extent(0) >= 9));
    if (!fits) {
        factorized_execution_space.fence("Replace execution result capacity");
        node_energies = {};
        node_forces = {};
        single_layer_workspace_xyz = {};
        node_energies_storage = {};
        node_forces_storage = {};
        node_energies_storage = decltype(node_energies_storage)(
            Kokkos::view_alloc(
                "Execution node energies capacity", Kokkos::WithoutInitializing),
            receiver_capacity);
        node_forces_storage = decltype(node_forces_storage)(
            Kokkos::view_alloc(
                "Execution directed forces capacity", Kokkos::WithoutInitializing),
            force_capacity);
        if (tiled_workspace) {
            const std::string workspace_label = dual_layer_tiled_plan_active
                ? "Dual-layer virial" : "Single-layer virial";
            atom_forces = decltype(atom_forces)(
                Kokkos::view_alloc(
                    "Single-layer atom forces capacity",
                    Kokkos::WithoutInitializing),
                checked_extent_product(
                    "atom-force capacity", {std::size_t(3), feature_capacity}));
            auto workspace_virial = Kokkos::View<double*>(
                    Kokkos::view_alloc(
                        workspace_label,
                        Kokkos::WithoutInitializing),
                    9);
            if (dual_layer_tiled_plan_active)
                dual_layer_workspace_virial = workspace_virial;
            else
                single_layer_workspace_virial = workspace_virial;
            stress_tensor = decltype(stress_tensor)(
                Kokkos::view_alloc(
                    "Single-layer batch virial", Kokkos::WithoutInitializing),
                9);
        }
        execution_result_allocation_count += 1;
    }
    node_energies = Kokkos::subview(
        node_energies_storage,
        Kokkos::make_pair(std::size_t(0), static_cast<std::size_t>(num_receivers)));
    node_forces = Kokkos::subview(
        node_forces_storage,
        Kokkos::make_pair(
            std::size_t(0), checked_extent_product(
                "active directed forces",
                {std::size_t(3), edge_capacity})));
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::execution_geometry_workspace_bytes() const
{
    return sizeof(double)*(execution_prepared_xyz.size()+execution_prepared_r.size()
        +execution_prepared_positions_storage.size()
        +execution_prepared_reference_positions_storage.size()
        +execution_prepared_reference_xyz_storage.size()
        +execution_prepared_displacements_storage.size()
        +execution_prepared_cell.size()+execution_prepared_inverse_cell.size()
        +execution_prepared_electric_field.size())
        +sizeof(int)*(execution_prepared_pbc.size()
            +execution_prepared_geometry_invalid.size()
            +execution_prepared_edge_shifts_storage.size()
            +single_layer_workspace_neigh_types.size()
            +dual_layer_workspace_neigh_types.size())
        +sizeof(Precision)*(execution_prepared_unit_direction.size()
            +xyz_shuffled.size()+Y.size()+Y_grad.size()
            +(Y_grad_shuffled.data() == Y_grad.data()
                ? 0 : Y_grad_shuffled.size()));
}

template <typename Precision>
MACEKokkos<Precision>::~MACEKokkos()
{
    Kokkos::fence();

    spherical_harmonics_state.reset();
    execution_prepared_xyz = decltype(execution_prepared_xyz)();
    execution_prepared_unit_direction =
        decltype(execution_prepared_unit_direction)();
    execution_prepared_r = decltype(execution_prepared_r)();
    single_layer_workspace_xyz = {};
    single_layer_explicit_xyz_host = {};
    single_layer_explicit_r_host = {};
    execution_prepared_positions = decltype(execution_prepared_positions)();
    execution_prepared_reference_positions =
        decltype(execution_prepared_reference_positions)();
    execution_prepared_reference_xyz = decltype(execution_prepared_reference_xyz)();
    execution_prepared_displacements = decltype(execution_prepared_displacements)();
    execution_prepared_edge_shifts = {};
    execution_prepared_edge_shifts_storage = {};
    single_layer_workspace_neigh_types = {};
    execution_prepared_positions_storage = {};
    execution_prepared_reference_positions_storage = {};
    execution_prepared_reference_xyz_storage = {};
    execution_prepared_displacements_storage = {};
    execution_prepared_cell = decltype(execution_prepared_cell)();
    execution_prepared_inverse_cell = decltype(execution_prepared_inverse_cell)();
    execution_prepared_pbc = decltype(execution_prepared_pbc)();
    execution_prepared_geometry_invalid =
        decltype(execution_prepared_geometry_invalid)();
    execution_prepared_electric_field =
        decltype(execution_prepared_electric_field)();
    xyz_shuffled = decltype(xyz_shuffled)();
    Y = decltype(Y)();
    Y_grad = decltype(Y_grad)();
    Y_grad_shuffled = decltype(Y_grad_shuffled)();
    node_energies = {};
    node_forces = {};
    node_energies_storage = {};
    node_forces_storage = {};

    M0_monomials = Kokkos::View<Kokkos::View<int**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>();
    M0_weights = Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace>();
    M0_poly_spec = Kokkos::View<Kokkos::View<int**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>();
    M0_poly_coeff = Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace>();
    M0_poly_values = Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace>();
    M0_poly_adjoints = Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace>();
    standard_m0_module_weights = decltype(standard_m0_module_weights)();

    A1_weights = Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>();
    A1_weights_trans = Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>();
    A1_channel_tile_weights = decltype(A1_channel_tile_weights)();
    A1_channel_tile_weights_trans =
        decltype(A1_channel_tile_weights_trans)();
    execution_projection = decltype(execution_projection)();
    execution_projection_trans = decltype(execution_projection_trans)();
    execution_receiver_projection = decltype(execution_receiver_projection)();
    execution_a1_projection_weights =
        decltype(execution_a1_projection_weights)();
    execution_final_projection = decltype(execution_final_projection)();
    factorized_workspace_arena = decltype(factorized_workspace_arena)();
    factorized_blas_context.reset();
}

template <typename Precision>
bool MACEKokkos<Precision>::supports_streamed_edges() const
{
    return uses_compact_radial;
}

template <typename Precision>
bool MACEKokkos<Precision>::supports_fused_streamed_reverse() const
{
    // The fused kernel trades path parallelism for fewer source adjoint atomics.
    return num_LM > 0
        && num_LM <= streamed_fused_max_num_LM
        && Phi1_l.extent(0) <= streamed_fused_max_num_paths
        && Phi1_lm1.extent(0) >= 2*static_cast<std::size_t>(num_lm);
}

template <typename Precision>
bool MACEKokkos<Precision>::supports_factorized() const
{
    return supports_streamed_edges()
        && num_LM > 0
        && num_LM <= streamed_fused_max_num_LM;
}

template <typename Precision>
bool MACEKokkos<Precision>::device_cuda_available() const
{
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (std::is_same_v<
            Kokkos::DefaultExecutionSpace,Kokkos::Cuda>)
        return true;
#endif
    return false;
}

template <typename Precision>
std::string MACEKokkos<Precision>::execution_cuda_device_name() const
{
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (std::is_same_v<
            Kokkos::DefaultExecutionSpace,Kokkos::Cuda>)
        return factorized_execution_space.cuda_device_prop().name;
#endif
    return {};
}

template <typename Precision>
int MACEKokkos<Precision>::execution_cuda_device_ordinal() const
{
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (std::is_same_v<
            Kokkos::DefaultExecutionSpace,Kokkos::Cuda>)
        return factorized_execution_space.cuda_device();
#endif
    return -1;
}

template <typename Precision>
int MACEKokkos<Precision>::execution_cuda_compute_capability() const
{
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (std::is_same_v<
            Kokkos::DefaultExecutionSpace,Kokkos::Cuda>) {
        const auto& device = factorized_execution_space.cuda_device_prop();
        return 10*device.major+device.minor;
    }
#endif
    return 0;
}

template <typename Precision>
int MACEKokkos<Precision>::execution_cuda_multiprocessor_count() const
{
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (std::is_same_v<
            Kokkos::DefaultExecutionSpace,Kokkos::Cuda>)
        return factorized_execution_space.cuda_device_prop().multiProcessorCount;
#endif
    return 0;
}

template <typename Precision>
int MACEKokkos<Precision>::execution_cuda_warp_width() const
{
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (std::is_same_v<
            Kokkos::DefaultExecutionSpace,Kokkos::Cuda>)
        return factorized_execution_space.cuda_device_prop().warpSize;
#endif
    return 0;
}

template <typename Precision>
int MACEKokkos<Precision>::execution_cuda_runtime_version() const
{
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (std::is_same_v<
            Kokkos::DefaultExecutionSpace,Kokkos::Cuda>) {
        int version = 0;
        DeviceBackendTraits<Kokkos::Cuda>::check_status(
            static_cast<std::int32_t>(cudaRuntimeGetVersion(&version)),
            "Querying the CUDA runtime version");
        return version;
    }
#endif
    return 0;
}

template <typename Precision>
int MACEKokkos<Precision>::execution_cuda_driver_version() const
{
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (std::is_same_v<
            Kokkos::DefaultExecutionSpace,Kokkos::Cuda>) {
        int version = 0;
        DeviceBackendTraits<Kokkos::Cuda>::check_status(
            static_cast<std::int32_t>(cudaDriverGetVersion(&version)),
            "Querying the CUDA driver version");
        return version;
    }
#endif
    return 0;
}

template <typename Precision>
void MACEKokkos<Precision>::load_jit_cuda_plugin(
    std::string path, const int persistent_blocks_per_compute_unit)
{
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (!std::is_same_v<
            Kokkos::DefaultExecutionSpace,Kokkos::Cuda>) {
        throw std::invalid_argument(
            "Execution CUDA plugins require a CUDA Kokkos execution space.");
    } else {
        if (!uses_compact_radial || !factorized_has_model_contract
            || !factorized_model_payload_verified)
            throw std::invalid_argument(
                "Execution CUDA plugins require a validated compact-v2 R1 contract.");
        constexpr std::string_view fingerprint_prefix = "sha256:";
        if (!std::string_view(factorized_model_contract_fingerprint).starts_with(
                fingerprint_prefix)
            || factorized_model_contract_fingerprint.size()
                < fingerprint_prefix.size()+16)
            throw std::invalid_argument(
                "Execution CUDA plugin contract fingerprint is malformed.");

        symmetrix::execution::CudaDeviceGuard device_guard(
            factorized_execution_space.cuda_device());
        const auto& device = factorized_execution_space.cuda_device_prop();
        constexpr std::string_view precision_suffix =
            std::is_same_v<Precision,float> ? "f32" : "f64";
        const std::string artifact_id = "jit-r1-gen"
            +std::to_string(symmetrix::execution::required_jit_generation_version)
            +"-"+std::string(precision_suffix)+"-"
            +factorized_model_contract_fingerprint.substr(
                fingerprint_prefix.size(), 16);
        const std::uint32_t capabilities =
            SYMMETRIX_JIT_CUDA_R1_FORWARD_LAUNCH_V2
            |SYMMETRIX_JIT_CUDA_R1_COORDINATE_REVERSE_LAUNCH_V2;
        symmetrix::execution::CudaPluginExpectation expectation{
            artifact_id,
            factorized_model_contract_fingerprint,
            factorized_model_semantic_fingerprint,
            {},
            num_channels,
            factorized_model_embedding,
            l_max,
            L_max,
            10*device.major+device.minor,
            device.maxThreadsPerBlock,
            persistent_blocks_per_compute_unit,
            capabilities,
            std::is_same_v<Precision,float>
                ? SYMMETRIX_JIT_CUDA_SCALAR_FLOAT32_V2
                : SYMMETRIX_JIT_CUDA_SCALAR_FLOAT64_V2,
            sizeof(Precision),
        };
        auto loaded = std::string_view(path).ends_with(".cubin")
            ? symmetrix::execution::CudaPlugin::load_cubin(
                std::move(path), expectation)
            : symmetrix::execution::CudaPlugin::load(
                std::move(path), expectation);
        factorized_execution_space.fence("Replace Execution CUDA plugin");
        jit_device_plugin =
            std::make_unique<symmetrix::execution::CudaPlugin>(std::move(loaded));
        set_factorized_source_strategy("jit_plugin");
    }
#else
    (void)path;
    throw std::invalid_argument(
        "Execution CUDA plugins require a CUDA-enabled Kokkos build.");
#endif
}

template <typename Precision>
bool MACEKokkos<Precision>::jit_cuda_plugin_ready() const
{
#ifdef KOKKOS_ENABLE_CUDA
    return std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>
        && jit_device_plugin
        && ExecutionDeviceBackend::plugin_ready(*jit_device_plugin);
#else
    return false;
#endif
}

template <typename Precision>
std::string MACEKokkos<Precision>::jit_cuda_plugin_path() const
{
#ifdef KOKKOS_ENABLE_CUDA
    return jit_cuda_plugin_ready()
        ? jit_device_plugin->path() : std::string();
#else
    return {};
#endif
}

template <typename Precision>
std::string MACEKokkos<Precision>::jit_cuda_plugin_artifact_id() const
{
#ifdef KOKKOS_ENABLE_CUDA
    return jit_cuda_plugin_ready()
        ? std::string(jit_device_plugin->artifact_id())
        : std::string();
#else
    return {};
#endif
}

template <typename Precision>
void MACEKokkos<Precision>::load_jit_hip_plugin(std::string path)
{
#ifdef KOKKOS_ENABLE_HIP
    if constexpr (!std::is_same_v<
            Kokkos::DefaultExecutionSpace,Kokkos::HIP>) {
        throw std::invalid_argument(
            "Execution HIP plugins require a HIP Kokkos execution space.");
    } else {
        if (!uses_compact_radial || !factorized_has_model_contract
            || !factorized_model_payload_verified)
            throw std::invalid_argument(
                "Execution HIP plugins require a validated compact-v2 R1 contract.");
        constexpr std::string_view fingerprint_prefix = "sha256:";
        if (!std::string_view(factorized_model_contract_fingerprint).starts_with(
                fingerprint_prefix)
            || factorized_model_contract_fingerprint.size()
                < fingerprint_prefix.size()+16)
            throw std::invalid_argument(
                "Execution HIP plugin contract fingerprint is malformed.");

        symmetrix::execution::HipDeviceGuard device_guard(
            factorized_execution_space.hip_device());
        const auto& device = factorized_execution_space.hip_device_prop();
        const auto target = normalize_hip_agent_target(device.gcnArchName);
        constexpr std::string_view precision_suffix =
            std::is_same_v<Precision,float> ? "f32" : "f64";
        const std::string artifact_id = "jit-r1-gen"
            +std::to_string(symmetrix::execution::required_jit_generation_version)
            +"-"+std::string(precision_suffix)+"-"
            +factorized_model_contract_fingerprint.substr(
                fingerprint_prefix.size(), 16);
        const std::uint32_t capabilities =
            SYMMETRIX_JIT_HIP_R1_FORWARD_LAUNCH_V1
            |SYMMETRIX_JIT_HIP_R1_COORDINATE_REVERSE_LAUNCH_V1;
        symmetrix::execution::HipPluginExpectation expectation{
            artifact_id,
            factorized_model_contract_fingerprint,
            factorized_model_semantic_fingerprint,
            {},
            target.base_isa,
            target.features,
            device.warpSize,
            num_channels,
            factorized_model_embedding,
            l_max,
            L_max,
            device.maxThreadsPerBlock,
            32,
            capabilities,
            std::is_same_v<Precision,float>
                ? SYMMETRIX_JIT_HIP_SCALAR_FLOAT32_V1
                : SYMMETRIX_JIT_HIP_SCALAR_FLOAT64_V1,
            sizeof(Precision),
        };
        auto loaded = std::string_view(path).ends_with(".hsaco")
            ? symmetrix::execution::HipPlugin::load_module(
                std::move(path), expectation)
            : symmetrix::execution::HipPlugin::load(
                std::move(path), expectation);
        factorized_execution_space.fence("Replace Execution HIP plugin");
        jit_device_plugin =
            std::make_unique<symmetrix::execution::HipPlugin>(std::move(loaded));
        set_factorized_source_strategy("jit_plugin");
    }
#else
    (void)path;
    throw std::invalid_argument(
        "Execution HIP plugins require a HIP-enabled Kokkos build.");
#endif
}

template <typename Precision>
bool MACEKokkos<Precision>::jit_hip_plugin_ready() const
{
#ifdef KOKKOS_ENABLE_HIP
    return std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::HIP>
        && jit_device_plugin
        && ExecutionDeviceBackend::plugin_ready(*jit_device_plugin);
#else
    return false;
#endif
}

template <typename Precision>
std::string MACEKokkos<Precision>::jit_hip_plugin_path() const
{
#ifdef KOKKOS_ENABLE_HIP
    return jit_hip_plugin_ready()
        ? jit_device_plugin->path() : std::string();
#else
    return {};
#endif
}

template <typename Precision>
std::string MACEKokkos<Precision>::jit_hip_plugin_artifact_id() const
{
#ifdef KOKKOS_ENABLE_HIP
    return jit_hip_plugin_ready()
        ? std::string(jit_device_plugin->artifact_id())
        : std::string();
#else
    return {};
#endif
}

template <typename Precision>
void MACEKokkos<Precision>::load_jit_device_plugin(
    std::string path, const int persistent_blocks_per_compute_unit)
{
#ifdef KOKKOS_ENABLE_CUDA
    load_jit_cuda_plugin(
        std::move(path), persistent_blocks_per_compute_unit);
#elif defined(KOKKOS_ENABLE_HIP)
    (void)persistent_blocks_per_compute_unit;
    load_jit_hip_plugin(std::move(path));
#else
    (void)path;
    (void)persistent_blocks_per_compute_unit;
    throw std::invalid_argument(
        "Execution device plugins require a CUDA- or HIP-enabled Kokkos build.");
#endif
}

template <typename Precision>
bool MACEKokkos<Precision>::jit_device_plugin_ready() const
{
#ifdef KOKKOS_ENABLE_CUDA
    return jit_cuda_plugin_ready();
#elif defined(KOKKOS_ENABLE_HIP)
    return jit_hip_plugin_ready();
#else
    return false;
#endif
}

template <typename Precision>
std::string MACEKokkos<Precision>::jit_device_plugin_path() const
{
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    return jit_device_plugin_ready()
        ? jit_device_plugin->path() : std::string();
#else
    return {};
#endif
}

template <typename Precision>
std::string MACEKokkos<Precision>::jit_device_plugin_artifact_id() const
{
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    return jit_device_plugin_ready()
        ? std::string(jit_device_plugin->artifact_id()) : std::string();
#else
    return {};
#endif
}

template <typename Precision>
void MACEKokkos<Precision>::load_m0_device_module(
    std::string path, std::string schedule,
    const int persistent_blocks_per_compute_unit)
{
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    if constexpr (!device_execution_space<Kokkos::DefaultExecutionSpace>) {
        throw std::invalid_argument(
            "Execution M0 device modules require a device Kokkos execution space.");
    } else {
        if (!uses_compact_radial || !standard_m0_has_model_contract
            || !m0_model_payload_verified)
            throw std::invalid_argument(
                "Execution M0 device modules require a verified compact-v2 M0 contract.");
        if (m0_correlation <= 0 || m0_correlation > 4)
            throw std::invalid_argument(
                "Execution M0 device modules support correlation up to four.");
        if (standard_m0_module_weights.data() == nullptr
            || m0_module_term_count <= 0)
            throw std::invalid_argument(
                "Execution M0 device module weights are unavailable.");
        if (schedule != "chunk32" && schedule != "table")
            throw std::invalid_argument(
                "Execution M0 device schedule must be 'chunk32' or 'table'.");
        if (persistent_blocks_per_compute_unit <= 0
            || persistent_blocks_per_compute_unit > 32)
            throw std::invalid_argument(
                "Execution M0 persistent blocks per compute unit must be in [1, 32].");

        constexpr std::string_view fingerprint_prefix = "sha256:";
        if (!std::string_view(standard_m0_model_structure_fingerprint)
                .starts_with(fingerprint_prefix)
            || standard_m0_model_structure_fingerprint.size()
                < fingerprint_prefix.size()+16)
            throw std::invalid_argument(
                "Execution M0 structure fingerprint is malformed.");
        constexpr std::string_view precision_suffix =
            std::is_same_v<Precision,float> ? "f32" : "f64";
        std::string target;
        int max_threads = 0;
#ifdef KOKKOS_ENABLE_CUDA
        symmetrix::execution::CudaDeviceGuard device_guard(
            factorized_execution_space.cuda_device());
        const auto& device = factorized_execution_space.cuda_device_prop();
        target = "sm_"+std::to_string(10*device.major+device.minor);
        max_threads = device.maxThreadsPerBlock;
#else
        symmetrix::execution::HipDeviceGuard device_guard(
            factorized_execution_space.hip_device());
        const auto& device = factorized_execution_space.hip_device_prop();
        const auto normalized = normalize_hip_agent_target(device.gcnArchName);
        target = normalized.base_isa
            +(normalized.features.empty() ? std::string() : ":"+normalized.features);
        max_threads = device.maxThreadsPerBlock;
#endif
        const std::string artifact_id = "jit-m0-gen"
            +std::to_string(symmetrix::execution::required_jit_generation_version)
            +"-"+std::string(precision_suffix)+"-"
            +standard_m0_model_structure_fingerprint.substr(
                fingerprint_prefix.size(), 16)+"-c"+std::to_string(num_channels)
            +"-"+schedule+"-"+target;
        constexpr std::uint32_t capabilities =
            SYMMETRIX_JIT_OPERATOR_M0_FORWARD_V1
            |SYMMETRIX_JIT_OPERATOR_M0_REVERSE_V1
            |SYMMETRIX_JIT_OPERATOR_M0_ALIAS_SAFE_V1
            |SYMMETRIX_JIT_OPERATOR_M0_SCALE_ADJOINT_V1
            |SYMMETRIX_JIT_OPERATOR_M0_RESPONSE_REVERSE_V1;
        const symmetrix::execution::OperatorModuleExpectation expectation{
            SYMMETRIX_JIT_OPERATOR_KIND_M0_V1,
            artifact_id,
            standard_m0_model_structure_fingerprint,
            target,
            std::is_same_v<Precision,float>
                ? SYMMETRIX_JIT_OPERATOR_SCALAR_FLOAT32_V1
                : SYMMETRIX_JIT_OPERATOR_SCALAR_FLOAT64_V1,
            sizeof(Precision),
            capabilities,
            num_lm,
            num_LM,
            m0_correlation,
            m0_module_term_count,
            max_threads,
        };
        auto loaded = symmetrix::execution::OperatorModule::load(
            std::move(path), expectation);
        factorized_execution_space.fence("Replace Execution M0 device module");
        m0_device_module =
            std::make_unique<symmetrix::execution::OperatorModule>(
                std::move(loaded));
        m0_device_module_schedule = std::move(schedule);
        m0_device_persistent_blocks_per_compute_unit =
            persistent_blocks_per_compute_unit;
        selected_m0_implementation = M0Implementation::device_module;
        if (use_m0_module())
            release_m0_polynomial_workspace();
    }
#else
    (void)path;
    (void)schedule;
    (void)persistent_blocks_per_compute_unit;
    throw std::invalid_argument(
        "Execution M0 device modules require a CUDA- or HIP-enabled build.");
#endif
}

template <typename Precision>
bool MACEKokkos<Precision>::m0_device_module_ready() const
{
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    return m0_device_module != nullptr && static_cast<bool>(*m0_device_module);
#else
    return false;
#endif
}

template <typename Precision>
std::string MACEKokkos<Precision>::m0_device_module_path() const
{
    return m0_device_module_ready() ? m0_device_module->path() : std::string();
}

template <typename Precision>
std::string MACEKokkos<Precision>::m0_device_module_artifact_id() const
{
    return m0_device_module_ready()
        ? std::string(m0_device_module->artifact_id()) : std::string();
}

template <typename Precision>
std::string MACEKokkos<Precision>::m0_device_module_schedule_name() const
{
    return m0_device_module_ready() ? m0_device_module_schedule : "none";
}

template <typename Precision>
void MACEKokkos<Precision>::load_m0_host_plugin(std::string path)
{
    if constexpr (!symmetrix::standard_r0::host_execution_space<
            Kokkos::DefaultExecutionSpace>) {
        throw std::invalid_argument(
            "Execution M0 host plugins require a host Kokkos execution space.");
    } else {
        if (!uses_compact_radial || !standard_m0_has_model_contract
            || !m0_model_payload_verified)
            throw std::invalid_argument(
                "Execution M0 host plugins require a verified compact-v2 M0 contract.");
        if (m0_correlation <= 0 || m0_correlation > 4
            || standard_m0_module_weights.data() == nullptr
            || m0_module_term_count <= 0)
            throw std::invalid_argument(
                "Execution M0 host plugin model data is unsupported (correlation must be 1-4).");
        constexpr std::string_view fingerprint_prefix = "sha256:";
        if (!std::string_view(standard_m0_model_structure_fingerprint)
                .starts_with(fingerprint_prefix))
            throw std::invalid_argument(
                "Execution M0 structure fingerprint is malformed.");
        constexpr std::string_view suffix =
            std::is_same_v<Precision,float> ? "f32" : "f64";
        const std::string artifact_id = "jit-m0-host-gen"
            +std::to_string(symmetrix::execution::required_jit_generation_version)
            +"-"+std::string(suffix)+"-"
            +standard_m0_model_structure_fingerprint.substr(
                fingerprint_prefix.size(), 16)
            +"-c"+std::to_string(num_channels)+"-chunk32";
        const symmetrix::execution::M0HostPluginExpectation expectation{
            artifact_id,
            standard_m0_model_structure_fingerprint,
            std::is_same_v<Precision,float>
                ? SYMMETRIX_JIT_M0_HOST_SCALAR_FLOAT32_V1
                : SYMMETRIX_JIT_M0_HOST_SCALAR_FLOAT64_V1,
            sizeof(Precision), num_channels, num_lm, num_LM,
            m0_correlation, m0_module_term_count,
        };
        auto loaded = symmetrix::execution::M0HostPlugin::load(
            std::move(path), expectation);
        factorized_execution_space.fence("Replace Execution M0 host plugin");
        m0_host_plugin =
            std::make_unique<symmetrix::execution::M0HostPlugin>(
                std::move(loaded));
        selected_m0_implementation = M0Implementation::host_plugin;
        if (use_m0_module())
            release_m0_polynomial_workspace();
    }
}

template <typename Precision>
bool MACEKokkos<Precision>::m0_host_plugin_ready() const
{
    return symmetrix::standard_r0::host_execution_space<
            Kokkos::DefaultExecutionSpace>
        && m0_host_plugin != nullptr && static_cast<bool>(*m0_host_plugin);
}

template <typename Precision>
std::string MACEKokkos<Precision>::m0_host_plugin_path() const
{
    return m0_host_plugin_ready() ? m0_host_plugin->path() : std::string();
}

template <typename Precision>
std::string MACEKokkos<Precision>::m0_host_plugin_artifact_id() const
{
    return m0_host_plugin_ready()
        ? std::string(m0_host_plugin->descriptor().artifact_id) : std::string();
}

template <typename Precision>
void MACEKokkos<Precision>::load_r0_device_module(
    std::string path, const int persistent_blocks_per_compute_unit)
{
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    if constexpr (!device_execution_space<Kokkos::DefaultExecutionSpace>) {
        throw std::invalid_argument(
            "Execution R0 device modules require a device Kokkos execution space.");
    } else {
        if (!uses_compact_radial || !standard_r0_has_model_contract
            || !r0_model_payload_verified)
            throw std::invalid_argument(
                "Execution R0 device modules require a verified scalar-source R0 contract.");
        if (persistent_blocks_per_compute_unit <= 0
            || persistent_blocks_per_compute_unit > 32)
            throw std::invalid_argument(
                "Execution R0 persistent blocks per compute unit must be in [1, 32].");
        constexpr std::string_view fingerprint_prefix = "sha256:";
        if (!std::string_view(standard_r0_model_structure_fingerprint)
                .starts_with(fingerprint_prefix)
            || standard_r0_model_structure_fingerprint.size()
                < fingerprint_prefix.size()+16)
            throw std::invalid_argument(
                "Execution R0 structure fingerprint is malformed.");
        constexpr std::string_view precision_suffix =
            std::is_same_v<Precision,float> ? "f32" : "f64";
        std::string target;
        int max_threads = 0;
#ifdef KOKKOS_ENABLE_CUDA
        symmetrix::execution::CudaDeviceGuard device_guard(
            factorized_execution_space.cuda_device());
        const auto& device = factorized_execution_space.cuda_device_prop();
        target = "sm_"+std::to_string(10*device.major+device.minor);
        max_threads = device.maxThreadsPerBlock;
#else
        symmetrix::execution::HipDeviceGuard device_guard(
            factorized_execution_space.hip_device());
        const auto& device = factorized_execution_space.hip_device_prop();
        const auto normalized = normalize_hip_agent_target(device.gcnArchName);
        target = normalized.base_isa
            +(normalized.features.empty() ? std::string() : ":"+normalized.features);
        max_threads = device.maxThreadsPerBlock;
#endif
        const std::string artifact_id = "jit-r0-gen"
            +std::to_string(symmetrix::execution::required_jit_generation_version)
            +"-"+std::string(precision_suffix)+"-"
            +standard_r0_model_structure_fingerprint.substr(
                fingerprint_prefix.size(), 16)+"-edge-batch1-"+target;
        constexpr std::uint32_t capabilities =
            SYMMETRIX_JIT_OPERATOR_R0_DENSITY_PREPARE_V1
            |SYMMETRIX_JIT_OPERATOR_R0_FORWARD_V1
            |SYMMETRIX_JIT_OPERATOR_R0_REVERSE_PREPARE_V1
            |SYMMETRIX_JIT_OPERATOR_R0_COORDINATE_REVERSE_V1
            |SYMMETRIX_JIT_OPERATOR_R0_COMPACT_GEOMETRY_V1
            |SYMMETRIX_JIT_OPERATOR_R0_PRECOMPUTED_SCALE_V1
            |SYMMETRIX_JIT_OPERATOR_R0_RECEIVER_BATCH_V1;
        const symmetrix::execution::OperatorModuleExpectation expectation{
            SYMMETRIX_JIT_OPERATOR_KIND_R0_V1,
            artifact_id,
            standard_r0_model_structure_fingerprint,
            target,
            std::is_same_v<Precision,float>
                ? SYMMETRIX_JIT_OPERATOR_SCALAR_FLOAT32_V1
                : SYMMETRIX_JIT_OPERATOR_SCALAR_FLOAT64_V1,
            sizeof(Precision),
            capabilities,
            num_lm,
            num_lm,
            1,
            num_lm,
            max_threads,
        };
        auto loaded = symmetrix::execution::OperatorModule::load(
            std::move(path), expectation);
        factorized_execution_space.fence("Replace Execution R0 device module");
        r0_device_module =
            std::make_unique<symmetrix::execution::OperatorModule>(
                std::move(loaded));
        r0_device_persistent_blocks_per_compute_unit =
            persistent_blocks_per_compute_unit;
        selected_r0_implementation = R0Implementation::device_module;
    }
#else
    (void)path;
    (void)persistent_blocks_per_compute_unit;
    throw std::invalid_argument(
        "Execution R0 device modules require a CUDA- or HIP-enabled build.");
#endif
}

template <typename Precision>
bool MACEKokkos<Precision>::r0_device_module_ready() const
{
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    return r0_device_module != nullptr && static_cast<bool>(*r0_device_module);
#else
    return false;
#endif
}

template <typename Precision>
std::string MACEKokkos<Precision>::r0_device_module_path() const
{
    return r0_device_module_ready() ? r0_device_module->path() : std::string();
}

template <typename Precision>
std::string MACEKokkos<Precision>::r0_device_module_artifact_id() const
{
    return r0_device_module_ready()
        ? std::string(r0_device_module->artifact_id()) : std::string();
}

template <typename Precision>
void MACEKokkos<Precision>::load_jit_host_plugin(std::string path)
{
    if (!uses_compact_radial || !factorized_has_model_contract
        || !factorized_model_payload_verified)
        throw std::invalid_argument(
            "Execution host plugins require a validated compact-v2 R1 contract.");
    const auto environment =
        kernel_launch_environment(factorized_execution_space);
    if (environment.backend != "host")
        throw std::invalid_argument(
            "Execution host plugins require a host Kokkos execution space.");
    constexpr std::string_view fingerprint_prefix = "sha256:";
    if (!std::string_view(factorized_model_contract_fingerprint).starts_with(
            fingerprint_prefix)
        || factorized_model_contract_fingerprint.size()
            < fingerprint_prefix.size()+16)
        throw std::invalid_argument(
            "Execution host plugin contract fingerprint is malformed.");
    constexpr std::string_view precision_suffix =
        std::is_same_v<Precision,float> ? "f32" : "f64";
    const std::string artifact_id = "jit-r1-gen"
        +std::to_string(symmetrix::execution::required_jit_generation_version)
        +"-"+std::string(precision_suffix)+"-"
        +factorized_model_contract_fingerprint.substr(
            fingerprint_prefix.size(), 16);
    const std::uint32_t capabilities =
        SYMMETRIX_JIT_HOST_R1_FORWARD_OWNER_V2
        |SYMMETRIX_JIT_HOST_R1_SOURCE_OWNER_V2
        |SYMMETRIX_JIT_HOST_R1_COMPENSATED_SOURCE_OWNER_V2
        |SYMMETRIX_JIT_HOST_R1_EDGE_OWNER_V2
        |SYMMETRIX_JIT_HOST_R1_SOURCE_CHANNEL_TILE_32_V2
        |SYMMETRIX_JIT_HOST_R1_FORWARD_CHANNEL_TILE_16_V2;
    // Structural fingerprints are migratable derived metadata. The exact
    // generation/semantic fingerprints and extents remain mandatory.
    symmetrix::execution::HostPluginExpectation expectation{
        artifact_id,
        factorized_model_contract_fingerprint,
        factorized_model_semantic_fingerprint,
        {},
        num_channels,
        factorized_model_embedding,
        l_max,
        L_max,
        capabilities,
        std::is_same_v<Precision,float>
            ? SYMMETRIX_JIT_HOST_SCALAR_FLOAT32_V2
            : SYMMETRIX_JIT_HOST_SCALAR_FLOAT64_V2,
        sizeof(Precision),
    };
    auto loaded = symmetrix::execution::HostPlugin::load(
        std::move(path), expectation);
    Kokkos::fence("Replace Execution host plugin");
    jit_host_plugin = std::make_unique<symmetrix::execution::HostPlugin>(
        std::move(loaded));
    set_factorized_source_strategy("jit_plugin");
}

template <typename Precision>
bool MACEKokkos<Precision>::jit_host_plugin_ready() const
{
    return jit_host_plugin
        && static_cast<bool>(*jit_host_plugin)
        && jit_host_plugin->has_v2_descriptor();
}

template <typename Precision>
std::string MACEKokkos<Precision>::jit_host_plugin_path() const
{
    return jit_host_plugin_ready() ? jit_host_plugin->path() : std::string();
}

template <typename Precision>
std::string MACEKokkos<Precision>::jit_host_plugin_artifact_id() const
{
    return jit_host_plugin_ready()
        ? std::string(jit_host_plugin->descriptor_v2().artifact_id)
        : std::string();
}

template <typename Precision>
bool MACEKokkos<Precision>::use_factorized_async_inference() const
{
    return mace_uses_prepared_execution(streamed_edges)
        && !factorized_observer_enabled
        && !execution_parameter_gradients_enabled;
}

template <typename Precision>
bool MACEKokkos<Precision>::use_gpu_field_h1_reverse() const
{
    return (streamed_edges == MACEStreamedEdgesMode::generic
            || mace_uses_prepared_execution(streamed_edges))
        && !factorized_observer_enabled
        && !execution_parameter_gradients_enabled;
}

template <typename Precision>
void MACEKokkos<Precision>::complete_device_stage(const char* label)
{
    static const bool force_execution_stage_fences = [] {
        const char* value = std::getenv("SYMMETRIX_STREAMED_STAGE_FENCES");
        return value != nullptr && std::string_view(value) != ""
            && std::string_view(value) != "0";
    }();
    if (use_factorized_async_inference() && !force_execution_stage_fences)
        return;
    Kokkos::fence(label);
    if (mace_uses_prepared_execution(streamed_edges))
        factorized_stage_fence_count += 1;
}

template <typename Precision>
void MACEKokkos<Precision>::set_factorized_source_strategy(std::string strategy)
{
    if (strategy == "serial_reference")
        factorized_source_strategy = FactorizedSourceStrategy::serial_reference;
    else if (strategy == "team_cached")
        factorized_source_strategy = FactorizedSourceStrategy::team_cached;
    else if (strategy == "tiled_coupling")
        factorized_source_strategy = FactorizedSourceStrategy::tiled_coupling;
    else if (strategy == "jit_plugin")
        factorized_source_strategy = FactorizedSourceStrategy::jit_plugin;
    else
        throw std::invalid_argument(
            "Execution R1 source strategy must be 'serial_reference', 'team_cached', "
            "'tiled_coupling', or 'jit_plugin'.");
    if (factorized_source_strategy != FactorizedSourceStrategy::tiled_coupling)
        release_factorized_coupling_workspace();
    plan_factorized_reverse_cache();
    if (use_factorized_direct_inference())
        release_factorized_stateful_workspace();
}

template <typename Precision>
void MACEKokkos<Precision>::set_factorized_direct_forward_executor(
    std::string executor)
{
    if (executor == "automatic")
        factorized_direct_forward_executor =
            FactorizedDirectForwardExecutor::automatic;
    else if (executor == "runtime")
        factorized_direct_forward_executor =
            FactorizedDirectForwardExecutor::runtime;
    else if (executor == "jit_all")
        factorized_direct_forward_executor =
            FactorizedDirectForwardExecutor::jit_all;
    else
        throw std::invalid_argument(
            "Execution R1 direct forward executor must be 'automatic', 'runtime', "
            "or 'jit_all'.");
}

template <typename Precision>
void MACEKokkos<Precision>::set_factorized_direct_reverse_executor(
    std::string executor)
{
    if (executor == "automatic")
        factorized_direct_reverse_executor =
            FactorizedDirectReverseExecutor::automatic;
    else if (executor == "runtime")
        factorized_direct_reverse_executor = FactorizedDirectReverseExecutor::runtime;
    else if (executor == "jit")
        factorized_direct_reverse_executor = FactorizedDirectReverseExecutor::jit;
    else
        throw std::invalid_argument(
            "Execution R1 direct reverse executor must be 'automatic', 'runtime', "
            "or 'jit'.");
}

template <typename Precision>
void MACEKokkos<Precision>::set_standard_r0_executor(std::string executor)
{
    if (executor == "automatic")
        standard_r0_executor = StandardR0Executor::automatic;
    else if (executor == "v1")
        standard_r0_executor = StandardR0Executor::v1;
    else if (executor == "v2_receiver")
        standard_r0_executor = StandardR0Executor::v2_receiver;
    else if (executor == "v2_edge16")
        standard_r0_executor = StandardR0Executor::v2_edge16;
    else if (executor == "v2_edge32")
        standard_r0_executor = StandardR0Executor::v2_edge32;
    else
        throw std::invalid_argument(
            "Execution R0 executor must be 'automatic', 'v1', 'v2_receiver', "
            "'v2_edge16', or 'v2_edge32'.");
}

template <typename Precision>
void MACEKokkos<Precision>::set_standard_m0_executor(std::string executor)
{
    StandardM0Executor requested;
    if (executor == "automatic")
        requested = StandardM0Executor::automatic;
    else if (executor == "runtime")
        requested = StandardM0Executor::runtime;
    else if (executor == "standard")
        requested = StandardM0Executor::standard;
    else
        throw std::invalid_argument(
            "Execution M0 executor must be 'automatic', 'runtime', or 'standard'.");

    const bool supported_mode = streamed_edges == MACEStreamedEdgesMode::generic
        || mace_uses_prepared_execution(streamed_edges);
    if (requested == StandardM0Executor::standard) {
        if (!supported_mode)
            throw std::invalid_argument(
                "Standard M0 requires streamed_edges='generic', 'direct', or "
                "'direct'.");
        if (!standard_m0_module_ready)
            throw std::invalid_argument(
                "Standard M0 is unavailable: "+standard_m0_module_fallback_reason);
        if (factorized_observer_enabled)
            throw std::invalid_argument(
                "Standard M0 does not support Execution observation.");
        if (execution_parameter_gradients_enabled)
            throw std::invalid_argument(
                "Standard M0 does not support parameter gradients.");
    }
    if (requested == standard_m0_executor)
        return;
    standard_m0_executor = requested;
    if (use_m0_module())
        release_m0_polynomial_workspace();
    else
        Kokkos::fence("Change M0 executor");
}

template <typename Precision>
void MACEKokkos<Precision>::ensure_m0_polynomial_workspace()
{
    using workspace_type =
        Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace>;
    if (M0_poly_values.extent(0) != static_cast<std::size_t>(num_LM))
        M0_poly_values = workspace_type(
            Kokkos::view_alloc("M0_poly_values",Kokkos::SequentialHostInit),
            num_LM);
    if (M0_poly_adjoints.extent(0) != static_cast<std::size_t>(num_LM))
        M0_poly_adjoints = workspace_type(
            Kokkos::view_alloc("M0_poly_adjoints",Kokkos::SequentialHostInit),
            num_LM);
}

template <typename Precision>
void MACEKokkos<Precision>::release_m0_polynomial_workspace()
{
    if (M0_poly_values.size() == 0 && M0_poly_adjoints.size() == 0)
        return;
    Kokkos::fence("Release M0 polynomial workspace");
    M0_poly_values = decltype(M0_poly_values)();
    M0_poly_adjoints = decltype(M0_poly_adjoints)();
}

template <typename Precision>
void MACEKokkos<Precision>::release_m1_polynomial_workspace()
{
    if (M1_poly_values.size() == 0 && M1_poly_adjoints.size() == 0)
        return;
    Kokkos::fence("Release M1 polynomial workspace");
    M1_poly_values = decltype(M1_poly_values)();
    M1_poly_adjoints = decltype(M1_poly_adjoints)();
}

template <typename Precision>
void MACEKokkos<Precision>::set_m1_polynomial_policy(std::string policy)
{
    if (policy != "automatic" && policy != "retained" && policy != "recompute")
        throw std::invalid_argument(
            "M1 polynomial policy must be 'automatic', 'retained', or "
            "'recompute'.");
    m1_polynomial_policy_request = policy;
    M1PolynomialPolicy requested = M1PolynomialPolicy::retained;
    int selected_tile = m1_recompute_tile_channels;
    if (policy != "retained") {
        selected_tile = select_m1_recompute_tile_channels(
            policy == "automatic" ? 32 : m1_recompute_tile_channels,
            mh0_state_policy == MH0StatePolicy::reuse_adjoints);
        if (selected_tile == 0) {
            const auto limit = m1_recompute_scratch_limit_bytes();
            std::size_t minimum = 0;
            try {
                minimum = m1_recompute_effective_scratch_bytes(
                    8,
                    has_field_coupling,
                    mh0_state_policy == MH0StatePolicy::reuse_adjoints);
            } catch (const std::length_error&) {
                // Keep zero in the diagnostic when the requirement overflowed.
            }
            m1_recompute_fallback_reason =
                "M1 recomputation on "+m1_recompute_backend_name()
                +" requires at least "+std::to_string(minimum)
                +" scratch bytes per team at tile 8; available limit is "
                +std::to_string(limit)+" bytes.";
            if (policy == "recompute")
                throw std::invalid_argument(m1_recompute_fallback_reason);
        } else {
            requested = M1PolynomialPolicy::recompute;
            m1_recompute_fallback_reason.clear();
        }
    } else {
        m1_recompute_fallback_reason.clear();
    }
    if (requested == m1_polynomial_policy
        && selected_tile == m1_recompute_tile_channels)
        return;
    if (requested == M1PolynomialPolicy::recompute)
        release_m1_polynomial_workspace();
    else
        Kokkos::fence("Change M1 polynomial policy");
    m1_polynomial_policy = requested;
    if (requested == M1PolynomialPolicy::recompute)
        m1_recompute_tile_channels = selected_tile;
}

template <typename Precision>
void MACEKokkos<Precision>::set_m1_recompute_tile_channels(const int channels)
{
    if (channels != 8 && channels != 16 && channels != 32)
        throw std::invalid_argument(
            "M1 recompute tile channels must be 8, 16, or 32.");
    if (channels == m1_recompute_tile_channels)
        return;
    if (m1_polynomial_policy == M1PolynomialPolicy::recompute) {
        const bool require_adjoint_overlap =
            mh0_state_policy == MH0StatePolicy::reuse_adjoints;
        const int selected = select_m1_recompute_tile_channels(
            channels, require_adjoint_overlap);
        if (selected != channels)
            validate_m1_recompute_scratch(
                channels, require_adjoint_overlap);
    }
    Kokkos::fence("Change M1 recompute tile channels");
    m1_recompute_tile_channels = channels;
}

template <typename Precision>
void MACEKokkos<Precision>::set_m1_recompute_scratch_limit_for_testing(
    const std::size_t bytes)
{
    m1_recompute_scratch_limit_override_bytes = bytes;
}

template <typename Precision>
bool MACEKokkos<Precision>::use_mh0_adjoint_reuse() const
{
    return mh0_state_policy == MH0StatePolicy::reuse_adjoints
        && mace_admits_low_memory(streamed_edges)
        && m1_polynomial_policy == M1PolynomialPolicy::recompute
        && (single_layer_readout
            || factorized_source_strategy == FactorizedSourceStrategy::jit_plugin)
        && r0_supports_low_memory()
        && m0_supports_low_memory()
        && !factorized_observer_enabled
        && !execution_parameter_gradients_enabled;
}

template <typename Precision>
bool MACEKokkos<Precision>::use_h1_m0_adjoint_ping_pong() const
{
    if constexpr (!symmetrix::standard_r0::host_execution_space<
            Kokkos::DefaultExecutionSpace>) {
        if (!single_layer_tiled_plan_active || dual_layer_tiled_plan_active)
            return false;
    }
    return use_mh0_adjoint_reuse() && !has_field_coupling;
}

template <typename Precision>
void MACEKokkos<Precision>::prepare_mh0_state_policy_views()
{
    const bool reuse = use_mh0_adjoint_reuse();
    const bool ping_pong = use_h1_m0_adjoint_ping_pong();
    const auto prepare = [reuse] (auto& adjoint, const auto& forward) {
        const bool aliases = adjoint.data() != nullptr
            && adjoint.data() == forward.data();
        if ((!reuse && aliases) || (reuse && !aliases))
            adjoint = std::decay_t<decltype(adjoint)>();
    };
    prepare(M1_adj, M1);
    prepare(A1_adj, A1);
    prepare(M0_adj, M0);
    prepare(A0_adj, A0);
    prepare(H2_adj, H2);
    const bool h1_adjoint_aliases_m0 = H1_adj.data() != nullptr
        && H1_adj.data() == M0.data();
    const bool m0_adjoint_aliases_h1 = M0_adj.data() != nullptr
        && M0_adj.data() == H1.data();
    if (!ping_pong && h1_adjoint_aliases_m0)
        H1_adj = decltype(H1_adj)();
    if (!ping_pong && m0_adjoint_aliases_h1)
        M0_adj = decltype(M0_adj)();
    const bool gradients_alias = Y_grad.data() != nullptr
        && Y_grad.data() == Y_grad_shuffled.data();
    if (reuse && !gradients_alias)
        Y_grad_shuffled = Y_grad;
    else if (!reuse && gradients_alias)
        Y_grad_shuffled = decltype(Y_grad_shuffled)(
            Kokkos::view_alloc(
                "Execution raw spherical harmonic gradients",
                Kokkos::WithoutInitializing),
            Y_grad.size());
    if (!reuse) {
        mh0_a1_scale_adjoint = decltype(mh0_a1_scale_adjoint)();
        mh0_a0_scale_adjoint = decltype(mh0_a0_scale_adjoint)();
    }
}

template <typename Precision>
void MACEKokkos<Precision>::set_mh0_state_policy(std::string policy)
{
    if (policy != "full-retention-v1" && policy != "reuse-adjoints-v1")
        throw std::invalid_argument(
            "MH-0 state policy must be 'full-retention-v1' or "
            "'reuse-adjoints-v1'.");
    if (policy == "reuse-adjoints-v1") {
        if (!mace_admits_low_memory(streamed_edges))
            throw std::invalid_argument(
                "MH-0 adjoint reuse requires direct execution.");
        if (m1_polynomial_policy != M1PolynomialPolicy::recompute)
            throw std::invalid_argument(
                "MH-0 adjoint reuse requires M1 polynomial recomputation.");
        if ((!single_layer_readout
             && factorized_source_strategy != FactorizedSourceStrategy::jit_plugin)
            || !r0_supports_low_memory())
            throw std::invalid_argument(
                "MH-0 adjoint reuse requires a state-free R0 implementation.");
        if (!m0_supports_low_memory())
            throw std::invalid_argument(
                "MH-0 adjoint reuse requires a state-free M0 implementation.");
        if (factorized_observer_enabled || execution_parameter_gradients_enabled)
            throw std::invalid_argument(
                "MH-0 adjoint reuse does not support observers or parameter "
                "gradients.");
        validate_m1_recompute_scratch(
            m1_recompute_tile_channels, true);
    }
    Kokkos::fence("Change MH-0 state policy");
    mh0_state_policy_request = std::move(policy);
    mh0_state_policy = mh0_state_policy_request == "reuse-adjoints-v1"
        ? MH0StatePolicy::reuse_adjoints : MH0StatePolicy::full_retention;
    prepare_mh0_state_policy_views();
}

template <typename Precision>
std::string MACEKokkos<Precision>::mh0_state_policy_name() const
{
    return use_mh0_adjoint_reuse()
        ? "reuse-adjoints-v1" : "full-retention-v1";
}

template <typename Precision>
std::string MACEKokkos<Precision>::mh0_state_policy_request_name() const
{
    return mh0_state_policy_request;
}

template <typename Precision>
std::string MACEKokkos<Precision>::mh0_state_policy_fallback_reason_name() const
{
    if (mh0_state_policy != MH0StatePolicy::reuse_adjoints
        || use_mh0_adjoint_reuse())
        return {};
    if (!mace_uses_prepared_execution(streamed_edges))
        return "prepared execution is not active";
    if (m1_polynomial_policy != M1PolynomialPolicy::recompute)
        return "M1 polynomial recomputation is not active";
    if ((!single_layer_readout
         && factorized_source_strategy != FactorizedSourceStrategy::jit_plugin)
        || !r0_supports_low_memory())
        return "a state-free R0 implementation is not active";
    if (!m0_supports_low_memory())
        return "a state-free M0 implementation is not active";
    if (factorized_observer_enabled)
        return "the factorized observer is active";
    if (execution_parameter_gradients_enabled)
        return "parameter gradients are active";
    return "MH-0 adjoint reuse is unavailable";
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::mh0_reused_state_bytes() const
{
    std::size_t bytes = 0;
    if (M1_adj.data() != nullptr && M1_adj.data() == M1.data())
        bytes += sizeof(Precision)*M1.size();
    if (A1_adj.data() != nullptr && A1_adj.data() == A1.data())
        bytes += sizeof(Precision)*A1.size();
    if (M0_adj.data() != nullptr
        && (M0_adj.data() == M0.data() || M0_adj.data() == H1.data()))
        bytes += sizeof(Precision)*M0.size();
    if (H1_adj.data() != nullptr && H1_adj.data() == M0.data())
        bytes += sizeof(Precision)*H1_adj.size();
    if (A0_adj.data() != nullptr && A0_adj.data() == A0.data())
        bytes += sizeof(Precision)*A0.size();
    if (H2_adj.data() != nullptr && H2_adj.data() == H2.data())
        bytes += sizeof(double)*H2.size();
    return bytes;
}

template <typename Precision>
bool MACEKokkos<Precision>::h1_m0_adjoint_ping_pong_active() const
{
    return use_h1_m0_adjoint_ping_pong()
        && H1_adj.data() != nullptr && H1_adj.data() == M0.data()
        && M0_adj.data() != nullptr && M0_adj.data() == H1.data();
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::h1_m0_adjoint_ping_pong_bytes() const
{
    return h1_m0_adjoint_ping_pong_active()
        ? sizeof(Precision)*H1_adj.size() : 0;
}

template <typename Precision>
bool MACEKokkos<Precision>::mh0_m0_adjoint_alias_active() const
{
    return M0.data() != nullptr
        && (M0_adj.data() == M0.data() || H1_adj.data() == M0.data());
}

template <typename Precision>
std::size_t
MACEKokkos<Precision>::mh0_m0_forward_replacement_count_value() const
{
    return mh0_m0_forward_replacement_count;
}

template <typename Precision>
std::size_t
MACEKokkos<Precision>::mh0_m0_adjoint_allocation_count_value() const
{
    return mh0_m0_adjoint_allocation_count;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::mh0_m0_alias_detach_count_value() const
{
    return mh0_m0_alias_detach_count;
}

template <typename Precision>
void MACEKokkos<Precision>::ensure_mh0_y_gradient_capacity(
    const std::size_t harmonic_values)
{
    const std::size_t planned_harmonic_values = checked_extent_product(
        "MH0 spherical harmonic gradient capacity",
        {static_cast<std::size_t>(std::max(0, execution_planned_edges)),
         static_cast<std::size_t>(num_lm)});
    const std::size_t elements = checked_extent_product(
        "MH0 spherical harmonic gradient capacity",
        {std::size_t(3), std::max(harmonic_values, planned_harmonic_values)});
    if (Y_grad.extent(0) >= elements)
        return;
    const auto execution_space = use_factorized_async_inference()
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    execution_space.fence("Replace MH0 spherical harmonic gradients");
    if (Y_grad.data() != nullptr && Y_grad_shuffled.data() == Y_grad.data())
        Y_grad_shuffled = decltype(Y_grad_shuffled)();
    Y_grad = decltype(Y_grad)();
    Y_grad = decltype(Y_grad)(
        Kokkos::view_alloc(
            "MH0 spherical harmonic gradients", Kokkos::WithoutInitializing),
        elements);
}

template <typename Precision>
void MACEKokkos<Precision>::ensure_mh0_a0_forward_capacity(
    const int num_nodes)
{
    const int capacity = (single_layer_tiled_plan_active
            || dual_layer_tiled_plan_active)
        ? num_nodes : std::max(num_nodes, execution_planned_receivers);
    if (A0.extent(0) >= static_cast<std::size_t>(capacity)
        && A0.extent_int(1) == num_lm
        && A0.extent_int(2) == num_channels)
        return;
    const auto execution_space = use_factorized_async_inference()
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    execution_space.fence("Replace MH0 A0 forward state");
    if (A0.data() != nullptr && A0_adj.data() == A0.data())
        A0_adj = decltype(A0_adj)();
    A0 = decltype(A0)();
    A0 = decltype(A0)(
        Kokkos::view_alloc(
            "MH0 A0 forward state", Kokkos::WithoutInitializing),
        capacity, num_lm, num_channels);
}

template <typename Precision>
void MACEKokkos<Precision>::ensure_mh0_m0_forward_capacity(
    const int num_nodes)
{
    const int capacity = (single_layer_tiled_plan_active
            || dual_layer_tiled_plan_active)
        ? num_nodes : std::max(num_nodes, execution_planned_receivers);
    if (M0.extent(0) >= static_cast<std::size_t>(capacity)
        && M0.extent_int(1) == num_LM
        && M0.extent_int(2) == num_channels)
        return;
    const auto execution_space = use_factorized_async_inference()
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    execution_space.fence("Replace MH0 M0 forward state");
    bool detached_alias = false;
    if (M0.data() != nullptr && M0_adj.data() == M0.data()) {
        M0_adj = decltype(M0_adj)();
        detached_alias = true;
    }
    if (M0.data() != nullptr && H1_adj.data() == M0.data()) {
        H1_adj = decltype(H1_adj)();
        detached_alias = true;
    }
    if (detached_alias)
        mh0_m0_alias_detach_count += 1;
    M0 = decltype(M0)();
    M0 = decltype(M0)(
        Kokkos::view_alloc(
            "MH0 M0 forward state", Kokkos::WithoutInitializing),
        capacity, num_LM, num_channels);
    mh0_m0_forward_replacement_count += 1;
}

template <typename Precision>
void MACEKokkos<Precision>::ensure_mh0_m0_adjoint_capacity()
{
    if (use_h1_m0_adjoint_ping_pong()
        && H1_adj.data() != nullptr && H1_adj.data() == M0.data()) {
        // H1 is dead when its reverse begins, so it can hold the M0 adjoint.
        M0_adj = H1;
        return;
    }
    if (use_mh0_adjoint_reuse()) {
        M0_adj = M0;
        return;
    }
    if (M0_adj.extent(0) >= M0.extent(0)
        && M0_adj.extent(1) == M0.extent(1)
        && M0_adj.extent(2) == M0.extent(2))
        return;
    const auto execution_space = use_factorized_async_inference()
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    execution_space.fence("Replace MH0 M0 adjoint state");
    M0_adj = decltype(M0_adj)();
    M0_adj = decltype(M0_adj)(
        Kokkos::view_alloc(
            "MH0 M0 adjoint state", Kokkos::WithoutInitializing),
        M0.extent(0), M0.extent(1), M0.extent(2));
    mh0_m0_adjoint_allocation_count += 1;
}

template <typename Precision>
void MACEKokkos<Precision>::ensure_mh0_a1_forward_capacity(
    const int num_nodes)
{
    const int capacity = dual_layer_tiled_plan_active
        ? num_nodes : std::max(num_nodes, execution_planned_receivers);
    if (A1.extent(0) >= static_cast<std::size_t>(capacity)
        && A1.extent_int(1) == num_lm
        && A1.extent_int(2) == num_channels)
        return;
    const auto execution_space = use_factorized_async_inference()
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    execution_space.fence("Replace MH0 A1 forward state");
    if (A1.data() != nullptr && A1_adj.data() == A1.data())
        A1_adj = decltype(A1_adj)();
    A1 = decltype(A1)();
    A1 = decltype(A1)(
        Kokkos::view_alloc(
            "MH0 A1 forward state", Kokkos::WithoutInitializing),
        capacity, num_lm, num_channels);
}

template <typename Precision>
void MACEKokkos<Precision>::ensure_mh0_phi1_forward_capacity(
    const int num_nodes,
    const int channel_count)
{
    if (channel_count <= 0)
        throw std::invalid_argument(
            "MH0 Phi1 capacity requires a positive channel count.");
    const int capacity = dual_layer_tiled_plan_active
        ? num_nodes : std::max(num_nodes, execution_planned_receivers);
    if (Phi1.extent(0) >= static_cast<std::size_t>(capacity)
        && Phi1.extent_int(1) == num_lme
        && Phi1.extent_int(2) == channel_count)
        return;
    const auto execution_space = use_factorized_async_inference()
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    execution_space.fence("Replace MH0 Phi1 forward state");
    if (Phi1.data() != nullptr && dPhi1.data() == Phi1.data())
        dPhi1 = decltype(dPhi1)();
    Phi1 = decltype(Phi1)();
    Phi1 = decltype(Phi1)(
        Kokkos::view_alloc(
            "MH0 Phi1 forward state", Kokkos::WithoutInitializing),
        capacity, num_lme, channel_count);
}

template <typename Precision>
void MACEKokkos<Precision>::ensure_mh0_m1_forward_capacity(
    const int num_nodes)
{
    const int capacity = (single_layer_tiled_plan_active
            || dual_layer_tiled_plan_active)
        ? num_nodes : std::max(num_nodes, execution_planned_receivers);
    if (M1.extent(0) >= static_cast<std::size_t>(capacity)
        && M1.extent_int(1) == num_channels)
        return;
    const auto execution_space = use_factorized_async_inference()
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    execution_space.fence("Replace MH0 M1 forward state");
    if (M1.data() != nullptr && M1_adj.data() == M1.data())
        M1_adj = decltype(M1_adj)();
    M1 = decltype(M1)();
    M1 = decltype(M1)(
        Kokkos::view_alloc(
            "MH0 M1 forward state", Kokkos::WithoutInitializing),
        capacity, num_channels);
}

template <typename Precision>
void MACEKokkos<Precision>::ensure_mh0_h2_forward_capacity(
    const int num_nodes)
{
    const int capacity = (single_layer_tiled_plan_active
            || dual_layer_tiled_plan_active)
        ? num_nodes : std::max(num_nodes, execution_planned_receivers);
    if (H2.extent(0) >= static_cast<std::size_t>(capacity)
        && H2.extent_int(1) == num_channels)
        return;
    const auto execution_space = use_factorized_async_inference()
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    execution_space.fence("Replace MH0 H2 forward state");
    if (H2.data() != nullptr && H2_adj.data() == H2.data())
        H2_adj = decltype(H2_adj)();
    H2 = decltype(H2)();
    H2 = decltype(H2)(
        Kokkos::view_alloc(
            "MH0 H2 forward state", Kokkos::WithoutInitializing),
        capacity, num_channels);
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::mh0_auxiliary_state_bytes() const
{
    return sizeof(double)*(
        mh0_a1_scale_adjoint.size()+mh0_a0_scale_adjoint.size());
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::readout_workspace_bytes() const
{
    return readout_2.workspace_bytes()
        +sizeof(double)*readout_2_output.size();
}

template <typename Precision>
std::string MACEKokkos<Precision>::readout_policy_name() const
{
    return readout_recompute ? "recompute" : "retained";
}

template <typename Precision>
void MACEKokkos<Precision>::set_readout_policy(std::string policy)
{
    bool recompute;
    if (policy == "retained")
        recompute = false;
    else if (policy == "recompute")
        recompute = true;
    else
        throw std::invalid_argument(
            "readout policy must be 'retained' or 'recompute'.");
    if (recompute == readout_recompute)
        return;
    Kokkos::fence("Change readout policy");
    readout_recompute = recompute;
    if (readout_recompute) {
        readout_2.release_workspace();
        readout_2_output = decltype(readout_2_output)();
    }
}

template <typename Precision>
bool MACEKokkos<Precision>::use_channel_tiled_phi1() const
{
#if defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_CUDA)
    return phi1_policy == Phi1Policy::channel_tiled_64
        && std::is_same_v<Precision,float>
        && !has_field_coupling
        && mace_admits_low_memory(streamed_edges)
        && factorized_source_strategy == FactorizedSourceStrategy::jit_plugin
        && jit_device_plugin_ready()
        && jit_device_plugin->supports_tiled_r1()
        && !factorized_observer_enabled
        && !execution_parameter_gradients_enabled;
#else
    return false;
#endif
}

template <typename Precision>
bool MACEKokkos<Precision>::use_receiver_local_phi1() const
{
#if defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_CUDA)
    return phi1_policy == Phi1Policy::receiver_local
        && std::is_same_v<Precision,float>
        && !has_field_coupling
        && mace_admits_low_memory(streamed_edges)
        && factorized_source_strategy == FactorizedSourceStrategy::jit_plugin
        && jit_device_plugin_ready()
        && jit_device_plugin->supports_projected_r1()
        && !factorized_observer_enabled
        && !execution_parameter_gradients_enabled;
#else
    return false;
#endif
}

template <typename Precision>
void MACEKokkos<Precision>::set_phi1_policy(std::string policy)
{
    Phi1Policy requested;
    if (policy == "retained")
        requested = Phi1Policy::retained;
    else if (policy == "channel-tiled-64")
        requested = Phi1Policy::channel_tiled_64;
    else if (policy == "receiver-local")
        requested = Phi1Policy::receiver_local;
    else
        throw std::invalid_argument(
            "Phi1 policy must be 'retained', 'channel-tiled-64', or "
            "'receiver-local'.");
    if (requested != Phi1Policy::retained) {
        if constexpr (!std::is_same_v<Precision,float>)
            throw std::invalid_argument(
                "experimental Phi1 policies currently require float32 inference.");
        if (has_field_coupling)
            throw std::invalid_argument(
                "experimental Phi1 policies currently support ordinary MACE only.");
        if (!mace_admits_low_memory(streamed_edges)
            || factorized_source_strategy != FactorizedSourceStrategy::jit_plugin)
            throw std::invalid_argument(
                "experimental Phi1 policies require direct generated execution.");
#if defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_CUDA)
        const bool module_ready = jit_device_plugin_ready()
            && (requested == Phi1Policy::receiver_local
                ? jit_device_plugin->supports_projected_r1()
                : jit_device_plugin->supports_tiled_r1());
        if (!module_ready)
            throw std::invalid_argument(
                "the requested Phi1 policy requires a matching RTC module.");
#else
        throw std::invalid_argument(
            "experimental Phi1 policies require HIP or CUDA execution.");
#endif
        if (factorized_observer_enabled || execution_parameter_gradients_enabled)
            throw std::invalid_argument(
                "experimental Phi1 policies do not support observers or parameter gradients.");
        if (requested == Phi1Policy::channel_tiled_64
            && num_channels%phi1_channel_tile_size != 0)
            throw std::invalid_argument(
                "channel-tiled-64 Phi1 requires a channel count divisible by 64.");
    }
    if (requested == phi1_policy)
        return;
    if (requested == Phi1Policy::receiver_local
        && harmonic_storage_policy == HarmonicStoragePolicy::y_only_direct)
        set_harmonic_storage_policy("retained");
    factorized_execution_space.fence("Change Phi1 policy");
    phi1_policy = requested;
    if (phi1_policy != Phi1Policy::retained) {
        Phi1 = decltype(Phi1)();
        dPhi1 = decltype(dPhi1)();
    }
}

template <typename Precision>
std::string MACEKokkos<Precision>::phi1_policy_name() const
{
    if (use_receiver_local_phi1())
        return "receiver-local";
    if (use_channel_tiled_phi1())
        return "channel-tiled-64";
    return "retained";
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::phi1_workspace_bytes() const
{
    std::size_t elements = Phi1.size();
    if (dPhi1.data() != Phi1.data())
        elements += dPhi1.size();
    return sizeof(Precision)*elements
        +generic_phi1_source_adjoint_workspace_bytes();
}

template <typename Precision>
std::size_t
MACEKokkos<Precision>::generic_phi1_source_adjoint_workspace_bytes() const
{
    return sizeof(double)*H1_adj_fp64_accumulator.size();
}

template <typename Precision>
bool MACEKokkos<Precision>::use_compact_edge_geometry() const
{
    return compact_edge_geometry_active
        && edge_geometry_policy == EdgeGeometryPolicy::unit_f32_radius_f64;
}

template <typename Precision>
std::string
MACEKokkos<Precision>::y_only_direct_harmonics_admission_reason() const
{
    const bool host_execution =
        symmetrix::standard_r0::host_execution_space<
            Kokkos::DefaultExecutionSpace>;
    const bool host_generated_r1 = host_execution
        && factorized_source_strategy == FactorizedSourceStrategy::jit_plugin
        && jit_host_plugin_ready();
#if defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_CUDA)
    constexpr bool accelerator_execution = true;
#else
    constexpr bool accelerator_execution = false;
#endif
    if (!accelerator_execution && !single_layer_readout && !host_generated_r1)
        return "a matching generated host R1 plugin is not active";
    if constexpr (std::is_same_v<Precision,float>) {
        if (edge_geometry_policy != EdgeGeometryPolicy::unit_f32_radius_f64)
            return "float32 execution requires compact unit-direction geometry";
    } else if (edge_geometry_policy != EdgeGeometryPolicy::cartesian_f64) {
        return "float64 execution requires Cartesian float64 geometry";
    }
    if (!mace_admits_low_memory(streamed_edges))
        return "direct execution is not active";
    if (!single_layer_readout
        && (factorized_source_strategy != FactorizedSourceStrategy::jit_plugin
            || !(jit_host_plugin_ready() || jit_device_plugin_ready())))
        return "a matching generated R1 module is not active";
    if (!standard_r0_module_ready
        || standard_r0_executor == StandardR0Executor::v1)
        return "a compatible standard R0 reverse is not active";
    if (phi1_policy != Phi1Policy::retained && !use_channel_tiled_phi1())
        return "retained or channel-tiled Phi1 is not active";
    if (factorized_observer_enabled)
        return "the factorized observer is active";
    if (execution_parameter_gradients_enabled)
        return "parameter gradients are active";
    if (l_max != 3)
        return "l_max is not 3";
    return "";
}

template <typename Precision>
bool MACEKokkos<Precision>::use_y_only_direct_harmonics() const
{
    return harmonic_storage_policy == HarmonicStoragePolicy::y_only_direct
        && y_only_direct_harmonics_admission_reason().empty();
}

template <typename Precision>
void MACEKokkos<Precision>::set_harmonic_storage_policy(std::string policy)
{
    HarmonicStoragePolicy requested;
    if (policy == "automatic")
        requested = HarmonicStoragePolicy::automatic;
    else if (policy == "retained")
        requested = HarmonicStoragePolicy::retained;
    else if (policy == "y-only-direct-v1")
        requested = HarmonicStoragePolicy::y_only_direct;
    else
        throw std::invalid_argument(
            "harmonic_storage_policy must be 'automatic', 'retained', or "
            "'y-only-direct-v1'.");
    if (requested == HarmonicStoragePolicy::y_only_direct) {
        if constexpr (std::is_same_v<Precision,float>) {
            if (edge_geometry_policy
                != EdgeGeometryPolicy::unit_f32_radius_f64)
                throw std::invalid_argument(
                    "float32 y-only-direct-v1 requires compact "
                    "unit-direction geometry.");
        } else {
            if (edge_geometry_policy != EdgeGeometryPolicy::cartesian_f64)
                throw std::invalid_argument(
                    "float64 y-only-direct-v1 requires Cartesian float64 "
                    "geometry.");
        }
        if (!mace_admits_low_memory(streamed_edges)
            || (!single_layer_readout
                && (factorized_source_strategy
                        != FactorizedSourceStrategy::jit_plugin
                    || !(jit_host_plugin_ready() || jit_device_plugin_ready()))))
            throw std::invalid_argument(
                "y-only-direct-v1 requires direct generated execution.");
        if (!standard_r0_module_ready
            || standard_r0_executor == StandardR0Executor::v1)
            throw std::invalid_argument(
                "y-only-direct-v1 requires a compatible standard R0 reverse.");
        if (phi1_policy != Phi1Policy::retained && !use_channel_tiled_phi1())
            throw std::invalid_argument(
                "y-only-direct-v1 requires retained or channel-tiled-64 Phi1.");
        if (factorized_observer_enabled || execution_parameter_gradients_enabled)
            throw std::invalid_argument(
                "y-only-direct-v1 does not support observers or parameter gradients.");
        if (l_max != 3)
            throw std::invalid_argument(
                "y-only-direct-v1 currently requires l_max=3.");
        if (!y_only_direct_harmonics_admission_reason().empty())
            throw std::invalid_argument(
                "y-only-direct-v1 admission failed: "
                +y_only_direct_harmonics_admission_reason());
    }
    if (requested == harmonic_storage_policy_request
        && requested != HarmonicStoragePolicy::automatic)
        return;
    factorized_execution_space.fence("Change harmonic storage policy");
    invalidate_factorized_prepared_graph();
    harmonic_storage_policy_request = requested;
    harmonic_storage_policy = requested == HarmonicStoragePolicy::y_only_direct
        ? HarmonicStoragePolicy::y_only_direct
        : HarmonicStoragePolicy::retained;
    harmonic_storage_selection_reason =
        requested == HarmonicStoragePolicy::automatic
        ? "automatic selection pending prepared graph"
        : requested == HarmonicStoragePolicy::retained
            ? "explicit retained policy" : "explicit Y-only policy";
    low_memory_device_free = 0;
    low_memory_device_total = 0;
    low_memory_reserve = 0;
    low_memory_capacity_y_only_estimate = 0;
    low_memory_capacity_retained_estimate = 0;
    xyz_shuffled = decltype(xyz_shuffled)();
    Y = decltype(Y)();
    Y_grad = decltype(Y_grad)();
    Y_grad_shuffled = decltype(Y_grad_shuffled)();
    execution_geometry_capacity_edges = 0;
    execution_geometry_planned_capacity_edges = 0;
    execution_geometry_growth_reason = "geometric growth";
}

template <typename Precision>
std::string MACEKokkos<Precision>::harmonic_storage_policy_name() const
{
    return use_y_only_direct_harmonics()
        ? "y-only-direct-v1" : "retained";
}

template <typename Precision>
std::string MACEKokkos<Precision>::harmonic_storage_policy_request_name() const
{
    switch (harmonic_storage_policy_request) {
    case HarmonicStoragePolicy::automatic:
        return "automatic";
    case HarmonicStoragePolicy::retained:
        return "retained";
    case HarmonicStoragePolicy::y_only_direct:
        return "y-only-direct-v1";
    }
    throw std::logic_error("Invalid harmonic storage policy request.");
}

template <typename Precision>
std::string
MACEKokkos<Precision>::harmonic_storage_selection_reason_name() const
{
    return harmonic_storage_selection_reason;
}

template <typename Precision>
std::string
MACEKokkos<Precision>::harmonic_storage_fallback_reason_name() const
{
    if (!low_memory_enabled()
        || harmonic_storage_policy_request != HarmonicStoragePolicy::automatic
        || use_y_only_direct_harmonics())
        return "";
    return y_only_direct_harmonics_admission_reason();
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::low_memory_device_free_bytes() const
{
    return low_memory_device_free;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::low_memory_device_total_bytes() const
{
    return low_memory_device_total;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::low_memory_reserve_bytes() const
{
    return low_memory_reserve;
}

template <typename Precision>
std::size_t
MACEKokkos<Precision>::low_memory_capacity_y_only_estimated_bytes() const
{
    return low_memory_capacity_y_only_estimate;
}

template <typename Precision>
std::size_t
MACEKokkos<Precision>::low_memory_capacity_retained_estimated_bytes() const
{
    return low_memory_capacity_retained_estimate;
}

template <typename Precision>
void MACEKokkos<Precision>::set_low_memory_device_memory_info_for_testing(
    const std::size_t free_bytes,
    const std::size_t total_bytes)
{
    factorized_execution_space.fence("Change low-memory test memory limits");
    invalidate_factorized_prepared_graph();
    if (free_bytes == 0 && total_bytes == 0) {
        low_memory_device_memory_info_override = false;
        low_memory_device_free = 0;
        low_memory_device_total = 0;
        return;
    }
    if (total_bytes == 0 || free_bytes > total_bytes)
        throw std::invalid_argument(
            "Low-memory device test override requires 0 < free <= total bytes.");
    low_memory_device_memory_info_override = true;
    low_memory_device_free = free_bytes;
    low_memory_device_total = total_bytes;
}

template <typename Precision>
void MACEKokkos<Precision>::set_single_layer_workspace_receiver_limit_for_testing(
    const int receivers)
{
    if (receivers <= 0)
        throw std::invalid_argument(
            "Single-layer workspace receiver limit must be positive.");
    factorized_execution_space.fence(
        "Change single-layer workspace receiver limit");
    invalidate_factorized_prepared_graph();
    single_layer_workspace_receiver_limit = receivers;
}

template <typename Precision>
void MACEKokkos<Precision>::set_dual_layer_workspace_receiver_limit_for_testing(
    const int receivers)
{
    if (receivers <= 0)
        throw std::invalid_argument(
            "Dual-layer workspace receiver limit must be positive.");
    factorized_execution_space.fence(
        "Change dual-layer workspace receiver limit");
    invalidate_factorized_prepared_graph();
    dual_layer_workspace_receiver_limit = receivers;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::estimate_graph_bytes(
    const int num_receivers,
    const int num_feature_nodes,
    const std::size_t num_edges,
    const bool capacity_bundle,
    const bool retain_harmonic_gradients) const
{
    if (num_receivers < 0 || num_feature_nodes < num_receivers)
        throw std::invalid_argument(
            "Low-memory estimate requires valid graph cardinalities "
            "(receivers="+std::to_string(num_receivers)+", feature_nodes="
            +std::to_string(num_feature_nodes)+").");
    const std::size_t receivers = static_cast<std::size_t>(num_receivers);
    const std::size_t features = static_cast<std::size_t>(num_feature_nodes);
    const std::size_t channels = static_cast<std::size_t>(num_channels);
    const std::size_t harmonics = static_cast<std::size_t>(num_lm);
    const std::size_t outputs = static_cast<std::size_t>(num_LM);
    const std::size_t coupled = static_cast<std::size_t>(num_lme);

    // Direct execution retains four edge-index arrays plus the direct-source
    // offsets, receiver metadata, reference geometry, active geometry,
    // harmonic values, and directed pair forces. The generic-only chunked
    // source schedule is deliberately excluded: direct execution leaves those
    // storage views empty.
    const std::size_t topology_edge_bytes = checked_extent_product(
        "harmonic topology edge", {num_edges, 4, sizeof(int)});
    const std::size_t topology_node_bytes = checked_extent_product(
        "harmonic topology node",
        {checked_extent_sum(
             "harmonic topology node entries",
             {features, std::size_t(1), std::size_t(3)*receivers}),
         sizeof(int)});
    const std::size_t reference_edge_bytes = checked_extent_product(
        "harmonic reference edge", {num_edges, 3, sizeof(double)});
    const std::size_t directed_force_bytes = checked_extent_product(
        "harmonic directed force", {num_edges, 3, sizeof(double)});
    const bool compact_geometry = capacity_bundle
        && std::is_same_v<Precision,float>;
    const std::size_t geometry_scalars = compact_geometry
        ? checked_extent_sum(
            "compact harmonic geometry",
            {3*sizeof(Precision), sizeof(double), harmonics*sizeof(Precision)})
        : checked_extent_sum(
            "Cartesian harmonic geometry",
            {4*sizeof(double), harmonics*sizeof(Precision)});
    const std::size_t geometry_bytes = checked_extent_product(
        "harmonic geometry", {num_edges, geometry_scalars});

    const std::size_t harmonic_channels = checked_extent_product(
        "harmonic-channel", {harmonics, channels});
    const std::size_t output_channels = checked_extent_product(
        "output-channel", {outputs, channels});
    const std::size_t coupled_channels = checked_extent_product(
        "coupled-channel", {coupled, channels});
    std::size_t receiver_precision_scalars = checked_extent_sum(
        "receiver forward state",
        {harmonic_channels, output_channels, coupled_channels,
         harmonic_channels, channels});
    if (!capacity_bundle)
        receiver_precision_scalars = checked_extent_sum(
            "receiver retained adjoint state",
            {receiver_precision_scalars, harmonic_channels, output_channels,
             coupled_channels, harmonic_channels, channels});
    const std::size_t receiver_precision_bytes = checked_extent_product(
        "low-memory receiver state",
        {receivers, receiver_precision_scalars, sizeof(Precision)});
    const std::size_t receiver_double_bytes = checked_extent_product(
        "low-memory receiver double state",
        {receivers, checked_extent_sum(
            "low-memory receiver double scalars", {
                channels,             // H2
                capacity_bundle ? 0 : channels, // separate H2 adjoint
                std::size_t(4),        // energies and reduced atom forces
                std::size_t(capacity_bundle && A0_scaled),
                std::size_t(capacity_bundle && A1_scaled),
            }), sizeof(double)});

    // H1, its adjoint, and the linear-up input remain live across R1 reverse.
    std::size_t feature_precision_scalars = checked_extent_product(
        "feature H1 state", {std::size_t(3), outputs, channels});
    if (has_field_coupling)
        feature_precision_scalars = checked_extent_sum(
            "field-aware feature state",
            {feature_precision_scalars, checked_extent_product(
                "field-aware feature state", {std::size_t(2), outputs, channels})});
    const std::size_t feature_precision_bytes = checked_extent_product(
        "low-memory feature state",
        {features, feature_precision_scalars, sizeof(Precision)});
    const std::size_t feature_bookkeeping_bytes = checked_extent_product(
        "low-memory feature bookkeeping",
        {features, checked_extent_sum(
            "feature bookkeeping scalars",
            {std::size_t(6)*sizeof(int), std::size_t(9)*sizeof(double),
             has_field_coupling ? std::size_t(3)*sizeof(double) : 0})});

    const std::size_t gradient_copies = capacity_bundle ? 1 : 2;
    const std::size_t retained_delta = retain_harmonic_gradients
        ? checked_extent_product(
            "retained harmonic coordinates and gradients",
            {num_edges, checked_extent_sum(
                "retained harmonic edge scalars",
                {3*sizeof(Precision), checked_extent_product(
                    "retained harmonic gradients",
                    {gradient_copies, std::size_t(3), harmonics,
                     sizeof(Precision)})})})
        : 0;
    std::size_t policy_workspace_bytes = 0;
    if (!capacity_bundle) {
        policy_workspace_bytes = checked_extent_sum(
            "speed policy workspace",
            {readout_2.estimated_workspace_bytes(
                 receivers, true, has_field_coupling),
             checked_extent_product(
                 "readout output", {receivers, sizeof(double)})});
        if (!low_memory_speed_m1_recompute)
            policy_workspace_bytes = checked_extent_sum(
                "retained M1 polynomial workspace",
                {policy_workspace_bytes, checked_extent_product(
                    "retained M1 polynomial values and adjoints",
                    {std::size_t(2), receivers, M1_poly_coeff.extent(1),
                     channels, sizeof(Precision)})});
    }
    return checked_extent_sum(
        capacity_bundle ? "low-memory graph" : "speed graph",
        {topology_edge_bytes, topology_node_bytes, reference_edge_bytes,
         directed_force_bytes,
         geometry_bytes, receiver_precision_bytes, receiver_double_bytes,
         feature_precision_bytes, feature_bookkeeping_bytes, retained_delta,
         policy_workspace_bytes});
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::estimate_single_layer_tiled_graph_bytes(
    const int num_receivers,
    const int num_feature_nodes,
    const std::size_t num_edges,
    const int workspace_receivers,
    const std::size_t workspace_edges) const
{
    if (num_receivers < 0 || num_feature_nodes < num_receivers
        || workspace_receivers < 0 || workspace_receivers > num_receivers
        || workspace_edges > num_edges)
        throw std::invalid_argument(
            "Single-layer tiled estimate requires valid graph and workspace "
            "cardinalities.");
    const std::size_t receivers = static_cast<std::size_t>(num_receivers);
    const std::size_t features = static_cast<std::size_t>(num_feature_nodes);
    const std::size_t workspace = static_cast<std::size_t>(workspace_receivers);
    const std::size_t channels = static_cast<std::size_t>(num_channels);
    const std::size_t harmonics = static_cast<std::size_t>(num_lm);
    const std::size_t outputs = static_cast<std::size_t>(num_LM);

    const std::size_t topology_edge_bytes = checked_extent_product(
        "single-layer tiled topology edge", {num_edges, sizeof(int)});
    const std::size_t topology_node_bytes = checked_extent_product(
        "single-layer tiled topology node",
        {std::size_t(3)*receivers+features, sizeof(int)});
    const std::size_t edge_bytes = checked_extent_product(
        "single-layer tiled persistent edge",
        {num_edges, checked_extent_sum(
            "single-layer tiled persistent edge scalars",
            {std::size_t(3)*sizeof(int)})});
    const std::size_t receiver_output_bytes = checked_extent_product(
        "single-layer tiled receiver outputs",
        {receivers, std::size_t(4), sizeof(double)});
    const std::size_t feature_geometry_bytes = checked_extent_product(
        "single-layer tiled feature geometry",
        {features, std::size_t(6)*sizeof(double)});
    const std::size_t workspace_precision_scalars = checked_extent_sum(
        "single-layer tiled precision workspace",
        {checked_extent_product(
             "single-layer tiled A0 workspace", {harmonics, channels}),
         checked_extent_product(
             "single-layer tiled equivariant workspace",
             {std::size_t(2), outputs, channels}),
         channels});
    const std::size_t workspace_precision_bytes = checked_extent_product(
        "single-layer tiled precision workspace",
        {workspace, workspace_precision_scalars, sizeof(Precision)});
    const std::size_t workspace_double_offset = checked_extent_align_up(
        "single-layer tiled FP64 workspace alignment",
        workspace_precision_bytes, alignof(double));
    const std::size_t workspace_bytes = checked_extent_sum(
        "single-layer tiled workspace",
        {workspace_double_offset,
         checked_extent_product(
             "single-layer tiled H2 workspace",
             {workspace, channels, sizeof(double)}),
         checked_extent_product(
             "single-layer tiled density workspace",
             {workspace, sizeof(double)}),
         checked_extent_product(
             "single-layer tiled receiver-offset workspace",
             {workspace, sizeof(int)})});
    const std::size_t edge_workspace_bytes = checked_extent_product(
        "single-layer tiled edge workspace",
        {workspace_edges, checked_extent_sum(
            "single-layer tiled workspace per edge",
            {harmonics*sizeof(Precision), std::size_t(3)*sizeof(double),
             std::size_t(3)*sizeof(Precision), sizeof(double), sizeof(int)})});
    return checked_extent_sum(
        "single-layer tiled graph",
        {topology_edge_bytes, topology_node_bytes, edge_bytes,
         receiver_output_bytes, feature_geometry_bytes, workspace_bytes,
         edge_workspace_bytes, std::size_t(9)*sizeof(double)});
}

template <typename Precision>
std::string MACEKokkos<Precision>::single_layer_tiled_admission_reason() const
{
#ifndef KOKKOS_ENABLE_CUDA
    return "mh0-single-layer-tiled-v1 currently requires CUDA";
#else
    if constexpr (!std::is_same_v<Precision,float>)
        return "mh0-single-layer-tiled-v1 requires float32 precision";
    if (!single_layer_readout)
        return "mh0-single-layer-tiled-v1 requires single_layer_readout=true";
    if (!mace_uses_prepared_execution(streamed_edges))
        return "mh0-single-layer-tiled-v1 requires direct prepared execution";
    if (has_field_coupling)
        return "mh0-single-layer-tiled-v1 does not support field coupling";
    if (factorized_observer_enabled)
        return "mh0-single-layer-tiled-v1 does not support execution observers";
    if (execution_parameter_gradients_enabled)
        return "mh0-single-layer-tiled-v1 does not support parameter gradients";
    if (!r0_supports_low_memory())
        return "mh0-single-layer-tiled-v1 requires a state-free R0 operator";
    if (!m0_supports_low_memory())
        return "mh0-single-layer-tiled-v1 requires a state-free M0 operator";
    const auto harmonic_reason = y_only_direct_harmonics_admission_reason();
    if (!harmonic_reason.empty())
        return "mh0-single-layer-tiled-v1 requires Y-only harmonics: "
            +harmonic_reason;
    return {};
#endif
}

template <typename Precision>
std::string MACEKokkos<Precision>::dual_layer_tiled_admission_reason() const
{
#ifndef KOKKOS_ENABLE_CUDA
    return "mh0-dual-layer-tiled-v1 currently requires CUDA";
#else
    if constexpr (!std::is_same_v<Precision,float>)
        return "mh0-dual-layer-tiled-v1 requires float32 precision";
    if (single_layer_readout || num_interactions != 2)
        return "mh0-dual-layer-tiled-v1 requires an ordinary two-layer MACE model";
    if (!mace_uses_prepared_execution(streamed_edges))
        return "mh0-dual-layer-tiled-v1 requires direct prepared execution";
    if (has_field_coupling)
        return "mh0-dual-layer-tiled-v1 does not support field coupling";
    if (factorized_distributed_phase != FactorizedDistributedPhase::idle)
        return "mh0-dual-layer-tiled-v1 does not support distributed execution";
    if (factorized_observer_enabled)
        return "mh0-dual-layer-tiled-v1 does not support execution observers";
    if (execution_parameter_gradients_enabled)
        return "mh0-dual-layer-tiled-v1 does not support parameter gradients";
    if (!r0_supports_low_memory())
        return "mh0-dual-layer-tiled-v1 requires a state-free R0 operator";
    if (!m0_supports_low_memory())
        return "mh0-dual-layer-tiled-v1 requires a state-free M0 operator";
    if (!jit_device_plugin_ready() || !jit_device_plugin->supports_tiled_r1())
        return "mh0-dual-layer-tiled-v1 requires a generated tiled R1 module";
    if (num_channels%phi1_channel_tile_size != 0)
        return "mh0-dual-layer-tiled-v1 requires channels divisible by 64";
    const auto harmonic_reason = y_only_direct_harmonics_admission_reason();
    if (!harmonic_reason.empty())
        return "mh0-dual-layer-tiled-v1 requires Y-only harmonics: "
            +harmonic_reason;
    return {};
#endif
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::estimate_dual_layer_tiled_graph_bytes(
    const int num_receivers,
    const int num_feature_nodes,
    const std::size_t num_edges,
    const int workspace_receivers,
    const std::size_t workspace_edges,
    const std::size_t source_segments) const
{
    if (num_receivers < 0 || num_feature_nodes != num_receivers
        || workspace_receivers < 0 || workspace_receivers > num_receivers
        || workspace_edges > num_edges || source_segments > num_edges)
        throw std::invalid_argument(
            "Dual-layer tiled estimate requires a monolithic graph and valid "
            "workspace cardinalities.");
    const std::size_t receivers = static_cast<std::size_t>(num_receivers);
    const std::size_t workspace = static_cast<std::size_t>(workspace_receivers);
    const std::size_t channels = static_cast<std::size_t>(num_channels);
    const std::size_t harmonics = static_cast<std::size_t>(num_lm);
    const std::size_t outputs = static_cast<std::size_t>(num_LM);
    const std::size_t coupled = static_cast<std::size_t>(num_lme);
    const std::size_t tile_count = workspace_receivers == 0 ? 0
        : receivers/workspace
            +(receivers%workspace != 0);

    const std::size_t persistent_topology = checked_extent_sum(
        "dual-layer tiled topology",
        {checked_extent_product(
             "dual-layer tiled topology edge", {num_edges, std::size_t(4)*sizeof(int)}),
         checked_extent_product(
             "dual-layer tiled topology node", {receivers, std::size_t(2)*sizeof(int)}),
         checked_extent_product(
             "dual-layer tiled receiver outputs", {receivers, std::size_t(4)*sizeof(double)}),
         checked_extent_product(
             "dual-layer tiled feature geometry", {receivers, std::size_t(6)*sizeof(double)})});
    const std::size_t learned_state = checked_extent_product(
        "dual-layer tiled H1 state",
        {std::size_t(2), receivers, outputs, channels, sizeof(Precision)});
    const std::size_t schedule = checked_extent_sum(
        "dual-layer tiled source schedule",
        {checked_extent_product(
             "dual-layer tiled edge schedule", {num_edges, sizeof(int)}),
         checked_extent_product(
             "dual-layer tiled segment schedule",
             {source_segments, std::size_t(2)*sizeof(int)}),
         checked_extent_product(
             "dual-layer tiled schedule sentinels",
             {tile_count+2, sizeof(int)})});
    const std::size_t workspace_precision = checked_extent_sum(
        "dual-layer tiled precision workspace",
        {checked_extent_product(
             "dual-layer tiled A slot", {workspace, harmonics, channels}),
         checked_extent_product(
             "dual-layer tiled equivariant slot", {workspace, outputs, channels}),
         checked_extent_product(
             "dual-layer tiled Phi1 slot",
             {workspace, coupled, static_cast<std::size_t>(phi1_channel_tile_size)}),
         checked_extent_product(
             "dual-layer tiled M1 slot", {workspace, channels})});
    const std::size_t precision_bytes = checked_extent_product(
        "dual-layer tiled precision workspace bytes",
        {workspace_precision, sizeof(Precision)});
    const std::size_t aligned_precision_bytes = checked_extent_align_up(
        "dual-layer tiled FP64 alignment", precision_bytes, alignof(double));
    const std::size_t receiver_workspace = checked_extent_sum(
        "dual-layer tiled receiver workspace",
        {aligned_precision_bytes,
         checked_extent_product(
             "dual-layer tiled H2 slot", {workspace, channels, sizeof(double)}),
         checked_extent_product(
             "dual-layer tiled density slots",
             {workspace, std::size_t(2), sizeof(double)}),
         checked_extent_product(
             "dual-layer tiled receiver offsets", {workspace, sizeof(int)})});
    const std::size_t edge_workspace = checked_extent_product(
        "dual-layer tiled edge workspace",
        {workspace_edges, checked_extent_sum(
            "dual-layer tiled workspace per edge",
            {harmonics*sizeof(Precision), std::size_t(3)*sizeof(double),
             std::size_t(3)*sizeof(Precision), sizeof(double),
             std::size_t(2)*sizeof(int)})});
    return checked_extent_sum(
        "dual-layer tiled graph",
        {persistent_topology, learned_state, schedule, receiver_workspace,
         edge_workspace, std::size_t(9)*sizeof(double)});
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::estimate_geometry_bytes_per_edge(
    const bool capacity_bundle,
    const bool retain_harmonic_gradients) const
{
    const std::size_t harmonics = static_cast<std::size_t>(num_lm);
    const bool compact_geometry = capacity_bundle
        && std::is_same_v<Precision,float>;
    const std::size_t base = compact_geometry
        ? checked_extent_sum(
            "compact harmonic geometry per edge",
            {3*sizeof(Precision), sizeof(double), harmonics*sizeof(Precision)})
        : checked_extent_sum(
            "Cartesian harmonic geometry per edge",
            {4*sizeof(double), harmonics*sizeof(Precision)});
    if (!retain_harmonic_gradients)
        return base;
    const std::size_t gradient_copies = capacity_bundle ? 1 : 2;
    return checked_extent_sum(
        "retained harmonic geometry per edge",
        {base, 3*sizeof(Precision), checked_extent_product(
            "retained harmonic gradients per edge",
            {gradient_copies, std::size_t(3), harmonics, sizeof(Precision)})});
}

template <typename Precision>
typename MACEKokkos<Precision>::LowMemoryPolicy
MACEKokkos<Precision>::preferred_speed_policy(const std::size_t num_edges) const
{
    if (!mace_admits_low_memory(streamed_edges)
        || (!single_layer_readout
            && (factorized_source_strategy != FactorizedSourceStrategy::jit_plugin
                || !(jit_host_plugin_ready() || jit_device_plugin_ready())))
        || !r0_supports_low_memory()
        || !m0_supports_low_memory()
        || factorized_observer_enabled
        || execution_parameter_gradients_enabled
        || select_m1_recompute_tile_channels(32, true) == 0)
        return LowMemoryPolicy::speed;

    constexpr std::size_t y_only_throughput_threshold_edges = 40960;
    constexpr bool host_execution =
        symmetrix::standard_r0::host_execution_space<
            Kokkos::DefaultExecutionSpace>;
#if defined(KOKKOS_ENABLE_CUDA)
    constexpr bool cuda_fp32_execution = std::is_same_v<Precision, float>;
#else
    constexpr bool cuda_fp32_execution = false;
#endif
    if (num_edges >= y_only_throughput_threshold_edges
        && (host_execution || cuda_fp32_execution)
        && y_only_direct_harmonics_admission_reason().empty())
        return LowMemoryPolicy::capacity_y_only;

    return LowMemoryPolicy::capacity_retained;
}

template <typename Precision>
void MACEKokkos<Precision>::apply_speed_policy(const LowMemoryPolicy preferred)
{
    apply_low_memory_policy(preferred);
    low_memory_policy = LowMemoryPolicy::speed;
    if (preferred == LowMemoryPolicy::capacity_y_only)
        harmonic_storage_selection_reason =
            "throughput-qualified graph-size Y-only selection: speed bundle is active";
    else if (preferred == LowMemoryPolicy::capacity_retained)
        harmonic_storage_selection_reason =
            "throughput-qualified retained selection: speed bundle is active";
}

template <typename Precision>
void MACEKokkos<Precision>::apply_preferred_speed_policy(
    const std::size_t num_edges)
{
    apply_speed_policy(preferred_speed_policy(num_edges));
}

template <typename Precision>
void MACEKokkos<Precision>::apply_low_memory_policy(
    const LowMemoryPolicy policy)
{
    const bool capacity = policy == LowMemoryPolicy::capacity_retained
        || policy == LowMemoryPolicy::capacity_y_only;
    const auto select_harmonics = [&] (const HarmonicStoragePolicy selected) {
        if (selected != harmonic_storage_policy) {
            harmonic_storage_policy = selected;
            xyz_shuffled = decltype(xyz_shuffled)();
            Y = decltype(Y)();
            Y_grad = decltype(Y_grad)();
            Y_grad_shuffled = decltype(Y_grad_shuffled)();
            execution_prepared_xyz = decltype(execution_prepared_xyz)();
            execution_prepared_unit_direction =
                decltype(execution_prepared_unit_direction)();
            execution_prepared_r = decltype(execution_prepared_r)();
            execution_geometry_capacity_edges = 0;
            execution_geometry_planned_capacity_edges = 0;
        }
        harmonic_storage_policy_request = HarmonicStoragePolicy::automatic;
    };

    if (!capacity) {
        select_harmonics(HarmonicStoragePolicy::retained);
        set_phi1_policy("retained");
        set_mh0_state_policy("full-retention-v1");
        set_m1_polynomial_policy(low_memory_speed_m1_policy_request);
        set_readout_policy("retained");
        set_edge_geometry_policy("cartesian-f64-v1");
        harmonic_storage_selection_reason =
            "automatic retained selection: speed bundle is active";
    } else {
        set_m1_polynomial_policy("recompute");
        set_mh0_state_policy("reuse-adjoints-v1");
        set_phi1_policy("retained");
        if constexpr (std::is_same_v<Precision,float>)
            set_edge_geometry_policy("unit-f32-radius-f64-v1");
        else
            set_edge_geometry_policy("cartesian-f64-v1");
        set_readout_policy("recompute");
        if (symmetrix::standard_r0::host_execution_space<
                Kokkos::DefaultExecutionSpace>)
            set_standard_r0_executor("v2_receiver");
        select_harmonics(policy == LowMemoryPolicy::capacity_y_only
            ? HarmonicStoragePolicy::y_only_direct
            : HarmonicStoragePolicy::retained);
        harmonic_storage_selection_reason = policy
                == LowMemoryPolicy::capacity_y_only
            ? "automatic Y-only selection: capacity bundle is active"
            : "automatic retained selection: capacity fallback is active";
    }
    low_memory_policy = policy;
}

template <typename Precision>
void MACEKokkos<Precision>::select_low_memory_policy_for_graph(
    const int num_receivers,
    const int num_feature_nodes,
    const std::size_t num_edges,
    const std::span<const int> num_neigh)
{
    if (num_neigh.size() != static_cast<std::size_t>(num_receivers))
        throw std::invalid_argument(
            "Single-layer workspace planning requires one degree per receiver.");
    single_layer_tiled_plan_active = false;
    dual_layer_tiled_plan_active = false;
    single_layer_workspace_active_capacity = 0;
    single_layer_workspace_active_edge_capacity = 0;
    dual_layer_workspace_active_capacity = 0;
    dual_layer_workspace_active_edge_capacity = 0;
    const std::string y_only_admission =
        y_only_direct_harmonics_admission_reason();
    const LowMemoryPolicy speed_policy = preferred_speed_policy(num_edges);
    const bool speed_uses_capacity_bundle =
        speed_policy != LowMemoryPolicy::speed;
    const bool speed_retains_harmonic_gradients =
        speed_policy != LowMemoryPolicy::capacity_y_only;
    const std::string single_layer_tiled_admission =
        single_layer_tiled_admission_reason();
    const std::size_t previous_selected_estimate =
        low_memory_selected_estimate;
    const int previous_planned_receivers = execution_planned_receivers;
    const int previous_planned_feature_nodes = execution_planned_feature_nodes;
    const int previous_planned_edges = execution_planned_edges;
    low_memory_capacity_y_only_estimate = estimate_graph_bytes(
        num_receivers, num_feature_nodes, num_edges, true, false);
    low_memory_capacity_retained_estimate = estimate_graph_bytes(
        num_receivers, num_feature_nodes, num_edges, true, true);
    low_memory_speed_estimate = estimate_graph_bytes(
        num_receivers, num_feature_nodes, num_edges,
        speed_uses_capacity_bundle, speed_retains_harmonic_gradients);
    const int tiled_workspace_receivers = std::min(
        num_receivers, single_layer_workspace_receiver_limit);
    std::size_t tiled_workspace_edges = 0;
    for (int receiver_begin=0; receiver_begin<num_receivers;) {
        const int receiver_end = receiver_begin+std::min(
            tiled_workspace_receivers, num_receivers-receiver_begin);
        std::size_t batch_edges = 0;
        for (int receiver=receiver_begin; receiver<receiver_end; ++receiver) {
            if (num_neigh[static_cast<std::size_t>(receiver)] < 0)
                throw std::invalid_argument(
                    "Single-layer workspace planning found a negative degree.");
            batch_edges = checked_extent_sum(
                "single-layer batch edge count",
                {batch_edges, static_cast<std::size_t>(
                    num_neigh[static_cast<std::size_t>(receiver)])});
        }
        tiled_workspace_edges = std::max(tiled_workspace_edges, batch_edges);
        receiver_begin = receiver_end;
    }
    if (tiled_workspace_edges > num_edges)
        throw std::invalid_argument(
            "Single-layer workspace edge count exceeds the graph extent.");
    single_layer_tiled_estimate = 0;
    if (single_layer_readout)
        single_layer_tiled_estimate = estimate_single_layer_tiled_graph_bytes(
            num_receivers, num_feature_nodes, num_edges,
            tiled_workspace_receivers, tiled_workspace_edges);
    const int dual_workspace_receivers = std::min(
        num_receivers, dual_layer_workspace_receiver_limit);
    std::size_t dual_workspace_edges = 0;
    if (dual_workspace_receivers > 0) {
        for (int receiver_begin=0; receiver_begin<num_receivers;) {
            const int receiver_end = receiver_begin+std::min(
                dual_workspace_receivers, num_receivers-receiver_begin);
            std::size_t batch_edges = 0;
            for (int receiver=receiver_begin; receiver<receiver_end; ++receiver)
                batch_edges = checked_extent_sum(
                    "dual-layer batch edge count",
                    {batch_edges, static_cast<std::size_t>(
                        num_neigh[static_cast<std::size_t>(receiver)])});
            dual_workspace_edges = std::max(dual_workspace_edges, batch_edges);
            receiver_begin = receiver_end;
        }
    }
    dual_layer_tiled_estimate = 0;
    if (!single_layer_readout && num_interactions == 2
        && num_feature_nodes == num_receivers)
        dual_layer_tiled_estimate = estimate_dual_layer_tiled_graph_bytes(
            num_receivers, num_feature_nodes, num_edges,
            dual_workspace_receivers, dual_workspace_edges, num_edges);
    const bool debug_capacity_retained =
        execution_plan_debug_id_ == "mh0-direct-capacity-retained";
    const bool debug_capacity_y_only =
        execution_plan_debug_id_ == "mh0-direct-capacity-y-only";
    const bool debug_single_layer_tiled =
        execution_plan_debug_id_ == "mh0-single-layer-tiled-v1";
    const bool debug_dual_layer_tiled =
        execution_plan_debug_id_ == "mh0-dual-layer-tiled-v1";

    const auto selected_capacity_policy = [&] {
        return y_only_admission.empty()
            ? LowMemoryPolicy::capacity_y_only
            : LowMemoryPolicy::capacity_retained;
    };
    const auto capacity_selection_reason = [&] (std::string reason) {
        if (!y_only_admission.empty())
            reason += "; Y-only unavailable: "+y_only_admission;
        return reason;
    };

    bool memory_info_available = true;
    std::string memory_query_failure;
    if (!low_memory_device_memory_info_override) {
#ifdef KOKKOS_ENABLE_CUDA
        ExecutionDeviceBackend::DeviceGuard device_guard(
            ExecutionDeviceBackend::device_ordinal(factorized_execution_space));
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (status != cudaSuccess) {
            memory_info_available = false;
            memory_query_failure = "cudaMemGetInfo failed: "
                +std::string(cudaGetErrorString(status));
        } else {
            low_memory_device_free = free_bytes;
            low_memory_device_total = total_bytes;
        }
#elif defined(KOKKOS_ENABLE_HIP)
        ExecutionDeviceBackend::DeviceGuard device_guard(
            ExecutionDeviceBackend::device_ordinal(factorized_execution_space));
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        const hipError_t status = hipMemGetInfo(&free_bytes, &total_bytes);
        if (status != hipSuccess) {
            memory_info_available = false;
            memory_query_failure = "hipMemGetInfo failed: "
                +std::string(hipGetErrorString(status));
        } else {
            low_memory_device_free = free_bytes;
            low_memory_device_total = total_bytes;
        }
#else
        memory_info_available = false;
        memory_query_failure = "device memory query is unavailable";
#endif
    }

    const auto select_geometric_capacity = [&] {
        const int current = execution_geometry_capacity_edges;
        if (num_edges <= static_cast<std::size_t>(current))
            return current;
        const int doubled = current > std::numeric_limits<int>::max()/2
            ? std::numeric_limits<int>::max() : 2*current;
        return std::max(
            static_cast<int>(num_edges), std::max(1, doubled));
    };
    const auto estimate_with_geometry_capacity = [&] (
        const std::size_t exact_estimate,
        const std::size_t bytes_per_edge,
        const int capacity) {
        const std::size_t extra_edges = static_cast<std::size_t>(capacity)
            -num_edges;
        return checked_extent_sum(
            "geometry capacity headroom",
            {exact_estimate, checked_extent_product(
                "geometry capacity headroom",
                {extra_edges, bytes_per_edge})});
    };
    if (!memory_info_available) {
        low_memory_device_free = 0;
        low_memory_device_total = 0;
        low_memory_reserve = 0;
        low_memory_available = 0;
        if (debug_dual_layer_tiled) {
            const auto rejection = dual_layer_tiled_admission_reason();
            if (!rejection.empty())
                throw std::invalid_argument(rejection);
            apply_low_memory_policy(LowMemoryPolicy::capacity_y_only);
            dual_layer_tiled_plan_active = true;
            set_phi1_policy("channel-tiled-64");
            dual_layer_workspace_active_capacity = dual_workspace_receivers;
            dual_layer_workspace_active_edge_capacity =
                static_cast<int>(dual_workspace_edges);
            low_memory_selected_estimate = dual_layer_tiled_estimate;
            execution_geometry_planned_capacity_edges =
                static_cast<int>(num_edges);
            low_memory_selection_reason =
                "debug override selected mh0-dual-layer-tiled-v1; "
                +memory_query_failure;
        } else if (debug_single_layer_tiled) {
            const auto rejection = single_layer_tiled_admission_reason();
            if (!rejection.empty())
                throw std::invalid_argument(rejection);
            apply_low_memory_policy(LowMemoryPolicy::capacity_y_only);
            single_layer_tiled_plan_active = true;
            single_layer_workspace_active_capacity = tiled_workspace_receivers;
            single_layer_workspace_active_edge_capacity =
                static_cast<int>(tiled_workspace_edges);
            low_memory_selected_estimate = single_layer_tiled_estimate;
            execution_geometry_planned_capacity_edges =
                static_cast<int>(num_edges);
            low_memory_selection_reason =
                "debug override selected mh0-single-layer-tiled-v1; "
                +memory_query_failure;
        } else if (debug_capacity_retained || debug_capacity_y_only) {
            const LowMemoryPolicy selected = debug_capacity_y_only
                ? LowMemoryPolicy::capacity_y_only
                : LowMemoryPolicy::capacity_retained;
            if (debug_capacity_y_only && !y_only_admission.empty())
                throw std::invalid_argument(
                    "mh0-direct-capacity-y-only debug plan is not qualified: "
                    +y_only_admission);
            apply_low_memory_policy(selected);
            low_memory_selection_reason = "debug override selected "
                +execution_plan_debug_id_+"; "+memory_query_failure;
            const bool y_only = selected == LowMemoryPolicy::capacity_y_only;
            const int capacity = select_geometric_capacity();
            execution_geometry_planned_capacity_edges = capacity;
            low_memory_selected_estimate = estimate_with_geometry_capacity(
                y_only ? low_memory_capacity_y_only_estimate
                       : low_memory_capacity_retained_estimate,
                estimate_geometry_bytes_per_edge(true, !y_only), capacity);
            execution_geometry_growth_reason = num_edges
                    <= static_cast<std::size_t>(
                        execution_geometry_capacity_edges)
                ? "existing capacity reused"
                : "geometric growth: device memory query unavailable";
        } else if (low_memory_request) {
            const LowMemoryPolicy selected = selected_capacity_policy();
            apply_low_memory_policy(selected);
            low_memory_selection_reason = capacity_selection_reason(
                "capacity selection: "+memory_query_failure);
            const bool y_only = selected == LowMemoryPolicy::capacity_y_only;
            const int capacity = select_geometric_capacity();
            execution_geometry_planned_capacity_edges = capacity;
            low_memory_selected_estimate = estimate_with_geometry_capacity(
                y_only ? low_memory_capacity_y_only_estimate
                       : low_memory_capacity_retained_estimate,
                estimate_geometry_bytes_per_edge(true, !y_only), capacity);
            execution_geometry_growth_reason = num_edges
                    <= static_cast<std::size_t>(
                        execution_geometry_capacity_edges)
                ? "existing capacity reused"
                : "geometric growth: device memory query unavailable";
        } else {
            apply_speed_policy(speed_policy);
            const int capacity = select_geometric_capacity();
            execution_geometry_planned_capacity_edges = capacity;
            low_memory_selected_estimate = estimate_with_geometry_capacity(
                low_memory_speed_estimate,
                estimate_geometry_bytes_per_edge(
                    speed_uses_capacity_bundle,
                    speed_retains_harmonic_gradients),
                capacity);
            low_memory_selection_reason =
                "low_memory=False: speed policy selected; "+memory_query_failure;
            execution_geometry_growth_reason = num_edges
                    <= static_cast<std::size_t>(
                        execution_geometry_capacity_edges)
                ? "existing capacity reused"
                : "geometric growth: device memory query unavailable";
        }
        execution_active_receivers = num_receivers;
        execution_active_feature_nodes = num_feature_nodes;
        execution_active_edges = static_cast<int>(num_edges);
        execution_planned_receivers = num_receivers;
        execution_planned_feature_nodes = num_feature_nodes;
        execution_planned_edges = static_cast<int>(num_edges);
        execution_planned_bytes = low_memory_selected_estimate;
        execution_capacity_selection_reason =
            "exact active capacity: device memory query unavailable";
        refresh_execution_plan_report();
        return;
    }

    constexpr std::size_t minimum_reserve = std::size_t(512)*1024*1024;
    low_memory_reserve = std::max(
        minimum_reserve, low_memory_device_total/20);
    const std::size_t allocated = low_memory_device_total-low_memory_device_free;
    const std::size_t reclaimable = previous_selected_estimate
            > allocated
        ? low_memory_device_total
        : low_memory_device_free+previous_selected_estimate;
    low_memory_available = reclaimable > low_memory_reserve
        ? reclaimable-low_memory_reserve : 0;

    bool capacity_bundle = false;
    bool retain_harmonic_gradients = true;
    std::size_t exact_estimate = low_memory_speed_estimate;
    if (debug_dual_layer_tiled) {
        const auto rejection = dual_layer_tiled_admission_reason();
        if (!rejection.empty())
            throw std::invalid_argument(rejection);
        capacity_bundle = true;
        retain_harmonic_gradients = false;
        exact_estimate = dual_layer_tiled_estimate;
        apply_low_memory_policy(LowMemoryPolicy::capacity_y_only);
        dual_layer_tiled_plan_active = true;
        set_phi1_policy("channel-tiled-64");
        dual_layer_workspace_active_capacity = dual_workspace_receivers;
        dual_layer_workspace_active_edge_capacity =
            static_cast<int>(dual_workspace_edges);
        low_memory_selection_reason =
            "debug override selected mh0-dual-layer-tiled-v1; bounded receiver "
            "workspace with capacity-y-only rollback remains available";
    } else if (debug_single_layer_tiled) {
        const auto rejection = single_layer_tiled_admission_reason();
        if (!rejection.empty())
            throw std::invalid_argument(rejection);
        capacity_bundle = true;
        retain_harmonic_gradients = false;
        exact_estimate = single_layer_tiled_estimate;
        apply_low_memory_policy(LowMemoryPolicy::capacity_y_only);
        single_layer_tiled_plan_active = true;
        single_layer_workspace_active_capacity = tiled_workspace_receivers;
        single_layer_workspace_active_edge_capacity =
            static_cast<int>(tiled_workspace_edges);
        low_memory_selection_reason =
            "debug override selected mh0-single-layer-tiled-v1; bounded receiver "
            "workspace with capacity-y-only rollback remains available";
    } else if (debug_capacity_retained || debug_capacity_y_only) {
        const LowMemoryPolicy selected = debug_capacity_y_only
            ? LowMemoryPolicy::capacity_y_only
            : LowMemoryPolicy::capacity_retained;
        if (debug_capacity_y_only && !y_only_admission.empty())
            throw std::invalid_argument(
                "mh0-direct-capacity-y-only debug plan is not qualified: "
                +y_only_admission);
        capacity_bundle = true;
        retain_harmonic_gradients = !debug_capacity_y_only;
        exact_estimate = retain_harmonic_gradients
            ? low_memory_capacity_retained_estimate
            : low_memory_capacity_y_only_estimate;
        apply_low_memory_policy(selected);
        low_memory_selection_reason = "debug override selected "+execution_plan_debug_id_
            +"; device-memory estimates remain advisory";
    } else if (!low_memory_request) {
        capacity_bundle = speed_uses_capacity_bundle;
        retain_harmonic_gradients = speed_retains_harmonic_gradients;
        apply_speed_policy(speed_policy);
        low_memory_selection_reason = exact_estimate <= low_memory_available
            ? "low_memory=False: exact speed graph fits advisory device-memory budget"
            : "low_memory=False: speed policy selected despite advisory "
              "device-memory estimate; allocation determines feasibility";
    } else if (low_memory_speed_estimate <= low_memory_available) {
        capacity_bundle = speed_uses_capacity_bundle;
        retain_harmonic_gradients = speed_retains_harmonic_gradients;
        apply_speed_policy(speed_policy);
        low_memory_selection_reason =
            "speed selection: estimated speed graph bytes fit available "
            "device memory after reserve";
    } else if (allow_fixed_workspace_
        && single_layer_tiled_admission.empty()
        && low_memory_capacity_y_only_estimate > low_memory_available
        && single_layer_tiled_estimate
            < low_memory_capacity_y_only_estimate) {
        capacity_bundle = true;
        retain_harmonic_gradients = false;
        exact_estimate = single_layer_tiled_estimate;
        apply_low_memory_policy(LowMemoryPolicy::capacity_y_only);
        single_layer_tiled_plan_active = true;
        single_layer_workspace_active_capacity = tiled_workspace_receivers;
        single_layer_workspace_active_edge_capacity =
            static_cast<int>(tiled_workspace_edges);
        low_memory_selection_reason = exact_estimate <= low_memory_available
            ? "fixed-workspace selection: capacity Y-only exceeds the advisory "
              "device-memory budget"
            : "fixed-workspace boundary selection: every qualified plan exceeds "
              "the advisory device-memory budget; exact allocation determines "
              "feasibility";
    } else {
        const LowMemoryPolicy selected = selected_capacity_policy();
        capacity_bundle = true;
        retain_harmonic_gradients =
            selected == LowMemoryPolicy::capacity_retained;
        exact_estimate = retain_harmonic_gradients
            ? low_memory_capacity_retained_estimate
            : low_memory_capacity_y_only_estimate;
        apply_low_memory_policy(selected);
        low_memory_selection_reason = capacity_selection_reason(
            exact_estimate <= low_memory_available
                ? "capacity selection: estimated speed graph bytes exceed "
                  "advisory device-memory budget"
                : "capacity selection: minimum estimate exceeds advisory "
                  "device-memory budget; exact allocation determines feasibility");
    }

    const int geometric_capacity = select_geometric_capacity();
    const std::size_t geometry_bytes_per_edge =
        estimate_geometry_bytes_per_edge(
            capacity_bundle, retain_harmonic_gradients);
    const std::size_t geometric_estimate = estimate_with_geometry_capacity(
        exact_estimate, geometry_bytes_per_edge, geometric_capacity);
    if (capacity_bundle) {
        execution_geometry_planned_capacity_edges = static_cast<int>(num_edges);
        low_memory_selected_estimate = exact_estimate;
        execution_geometry_growth_reason =
            "capacity high-water pending post-reserve admission";
    } else if (num_edges <= static_cast<std::size_t>(
            execution_geometry_capacity_edges)) {
        execution_geometry_planned_capacity_edges = geometric_capacity;
        low_memory_selected_estimate = geometric_estimate;
        execution_geometry_growth_reason = "existing capacity reused";
    } else if (geometric_estimate <= low_memory_available) {
        execution_geometry_planned_capacity_edges = geometric_capacity;
        low_memory_selected_estimate = geometric_estimate;
        execution_geometry_growth_reason =
            "geometric growth fits available device memory";
    } else {
        execution_geometry_planned_capacity_edges = static_cast<int>(num_edges);
        low_memory_selected_estimate = exact_estimate;
        execution_geometry_growth_reason =
            "exact growth: geometric headroom exceeds available device memory";
    }

    execution_active_receivers = num_receivers;
    execution_active_feature_nodes = num_feature_nodes;
    execution_active_edges = static_cast<int>(num_edges);
    execution_planned_receivers = num_receivers;
    execution_planned_feature_nodes = num_feature_nodes;
    execution_planned_edges = static_cast<int>(num_edges);
    execution_planned_bytes = low_memory_selected_estimate;
    execution_capacity_selection_reason = "exact active capacity";

    if (single_layer_tiled_plan_active || dual_layer_tiled_plan_active) {
        refresh_execution_plan_report();
        return;
    }
    if (!low_memory_request || !capacity_bundle) {
        refresh_execution_plan_report();
        return;
    }

    const auto desired_slack = [] (const int active) {
        constexpr int minimum_slack = 64;
        constexpr int denominator = 400; // 0.25 percent.
        const int proportional = active/denominator
            +(active%denominator != 0);
        const int requested = std::max(minimum_slack, proportional);
        return active > std::numeric_limits<int>::max()-requested
            ? std::numeric_limits<int>::max()-active : requested;
    };
    std::size_t remaining = low_memory_available > execution_planned_bytes
        ? low_memory_available-execution_planned_bytes : 0;
    const auto admit = [&] (
        const int active,
        const int previous,
        const std::size_t bytes_per_item,
        int& planned,
        const int maximum = std::numeric_limits<int>::max()) {
        if (bytes_per_item == 0 || remaining < bytes_per_item)
            return;
        const int requested = previous >= active
            ? previous : active+desired_slack(active);
        const int target = std::min(requested, maximum);
        const std::size_t admitted = std::min(
            static_cast<std::size_t>(target-active),
            remaining/bytes_per_item);
        planned = active+static_cast<int>(admitted);
        const std::size_t admitted_bytes = checked_extent_product(
            "repeated-run admitted high-water capacity",
            {admitted, bytes_per_item});
        execution_planned_bytes = checked_extent_sum(
            "repeated-run admitted high-water capacity",
            {execution_planned_bytes, admitted_bytes});
        remaining -= admitted_bytes;
    };

    const std::size_t feature_increment =
        num_feature_nodes == std::numeric_limits<int>::max()
        ? 0 : estimate_graph_bytes(
            num_receivers, num_feature_nodes+1, num_edges,
            capacity_bundle, retain_harmonic_gradients)-exact_estimate;
    std::size_t receiver_increment = 0;
    if (num_receivers != std::numeric_limits<int>::max()) {
        const bool feature_must_grow = num_feature_nodes == num_receivers;
        const std::size_t coupled_increment = estimate_graph_bytes(
            num_receivers+1,
            num_feature_nodes+(feature_must_grow ? 1 : 0),
            num_edges, capacity_bundle, retain_harmonic_gradients)-exact_estimate;
        receiver_increment = feature_must_grow
            ? coupled_increment-feature_increment : coupled_increment;
    }
    const std::size_t edge_increment = num_edges
            == static_cast<std::size_t>(std::numeric_limits<int>::max())
        ? 0 : estimate_graph_bytes(
            num_receivers, num_feature_nodes, num_edges+1,
            capacity_bundle, retain_harmonic_gradients)-exact_estimate;
    admit(
        static_cast<int>(num_edges), previous_planned_edges,
        edge_increment, execution_planned_edges);
    execution_geometry_planned_capacity_edges = execution_planned_edges;
    if (num_receivers == num_feature_nodes) {
        int common_capacity = num_receivers;
        const std::size_t coupled_increment = checked_extent_sum(
            "coupled receiver and feature high-water",
            {receiver_increment, feature_increment});
        admit(
            num_receivers,
            std::min(previous_planned_receivers, previous_planned_feature_nodes),
            coupled_increment, common_capacity);
        execution_planned_receivers = common_capacity;
        execution_planned_feature_nodes = common_capacity;
    } else {
        admit(
            num_receivers, previous_planned_receivers,
            receiver_increment, execution_planned_receivers);
        admit(
            num_feature_nodes, previous_planned_feature_nodes, feature_increment,
            execution_planned_feature_nodes);
    }
    low_memory_selected_estimate = execution_planned_bytes;
    execution_capacity_selection_reason =
        "admitted 0.25% repeated-run high-water within post-reserve bytes; "
        "active=("+std::to_string(num_receivers)+","+
        std::to_string(num_feature_nodes)+","+std::to_string(num_edges)+
        "), previous=("+std::to_string(previous_planned_receivers)+","+
        std::to_string(previous_planned_feature_nodes)+","+
        std::to_string(previous_planned_edges)+"), planned=("+
        std::to_string(execution_planned_receivers)+","+
        std::to_string(execution_planned_feature_nodes)+","+
        std::to_string(execution_planned_edges)+")";
    refresh_execution_plan_report();
}

template <typename Precision>
void MACEKokkos<Precision>::refresh_execution_plan_report()
{
    if (!mace_uses_direct_execution(streamed_edges))
        return;

    const auto y_only_reason = y_only_direct_harmonics_admission_reason();
    const bool memory_budget_available = low_memory_device_total != 0;
    const auto single_layer_tiled_reason =
        single_layer_tiled_admission_reason();
    const auto dual_layer_tiled_reason = dual_layer_tiled_admission_reason();
    const bool single_layer_tiled_permitted = allow_fixed_workspace_
        || execution_plan_debug_id_ == "mh0-single-layer-tiled-v1";
    const bool dual_layer_tiled_permitted = allow_fixed_workspace_
        || execution_plan_debug_id_ == "mh0-dual-layer-tiled-v1";
    const std::string fixed_workspace_opt_in_reason =
        "fixed-workspace execution requires explicit opt-in via "
        "allow_fixed_workspace";
    execution_plan_.available_bytes = low_memory_available;
    execution_plan_.reserve_bytes = low_memory_reserve;
    execution_plan_.state = "active";
    execution_plan_.boundary_attempt = memory_budget_available
        && execution_plan_.requested_profile
            == symmetrix::execution::ExecutionProfile::capacity
        &&
        low_memory_speed_estimate > low_memory_available
        && low_memory_capacity_y_only_estimate > low_memory_available
        && low_memory_capacity_retained_estimate > low_memory_available
        && (!single_layer_tiled_permitted
            || !single_layer_tiled_reason.empty()
            || single_layer_tiled_estimate > low_memory_available);
    execution_plan_.selected_id = execution_plan_.requested_profile
            == symmetrix::execution::ExecutionProfile::speed
        ? "mh0-direct-speed"
        : low_memory_policy == LowMemoryPolicy::speed
        ? "mh0-direct-speed"
        : dual_layer_tiled_plan_active
            ? "mh0-dual-layer-tiled-v1"
        : single_layer_tiled_plan_active
            ? "mh0-single-layer-tiled-v1"
        : low_memory_policy == LowMemoryPolicy::capacity_y_only
            ? "mh0-direct-capacity-y-only"
            : "mh0-direct-capacity-retained";
    execution_plan_.selection_reason = low_memory_selection_reason;
    execution_plan_.candidates = {
        {"mh0-direct-speed", true, low_memory_speed_estimate,
            !memory_budget_available
                ? "memory query unavailable"
                : low_memory_speed_estimate <= low_memory_available
                ? "estimated fit" : "exceeds advisory memory budget"},
        {"mh0-direct-capacity-y-only", y_only_reason.empty(),
            low_memory_capacity_y_only_estimate,
            y_only_reason.empty() ? "qualified" : y_only_reason},
        {"mh0-direct-capacity-retained", true,
            low_memory_capacity_retained_estimate, "qualified"},
        {"mh0-single-layer-tiled-v1",
            single_layer_tiled_permitted && single_layer_tiled_reason.empty(),
            single_layer_tiled_estimate,
            !single_layer_tiled_permitted
                ? fixed_workspace_opt_in_reason
                : single_layer_tiled_reason.empty()
                    ? "qualified" : single_layer_tiled_reason},
        {"mh0-dual-layer-tiled-v1",
            dual_layer_tiled_permitted && dual_layer_tiled_reason.empty(),
            dual_layer_tiled_estimate,
            !dual_layer_tiled_permitted
                ? fixed_workspace_opt_in_reason
                : dual_layer_tiled_reason.empty()
                    ? "qualified" : dual_layer_tiled_reason},
    };
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::harmonic_value_bytes() const
{
    return sizeof(Precision)*Y.size();
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::harmonic_gradient_bytes() const
{
    const std::size_t elements = Y_grad.size()
        +(Y_grad_shuffled.data() == Y_grad.data()
            ? 0 : Y_grad_shuffled.size());
    return sizeof(Precision)*elements;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::shuffled_coordinate_bytes() const
{
    return sizeof(Precision)*xyz_shuffled.size();
}

template <typename Precision>
void MACEKokkos<Precision>::set_edge_geometry_policy(std::string policy)
{
    EdgeGeometryPolicy requested;
    if (policy == "cartesian-f64-v1")
        requested = EdgeGeometryPolicy::cartesian_f64;
    else if (policy == "unit-f32-radius-f64-v1")
        requested = EdgeGeometryPolicy::unit_f32_radius_f64;
    else
        throw std::invalid_argument(
            "edge_geometry_policy must be 'cartesian-f64-v1' or "
            "'unit-f32-radius-f64-v1'.");

    if (requested == EdgeGeometryPolicy::unit_f32_radius_f64) {
        if constexpr (!std::is_same_v<Precision,float>)
            throw std::invalid_argument(
                "unit-f32-radius-f64-v1 requires float32 inference.");
        if (!mace_admits_low_memory(streamed_edges))
            throw std::invalid_argument(
                "unit-f32-radius-f64-v1 requires direct execution.");
        if (!single_layer_readout
            && factorized_source_strategy != FactorizedSourceStrategy::jit_plugin)
            throw std::invalid_argument(
                "unit-f32-radius-f64-v1 requires generated RTC execution.");
        if (!r0_supports_low_memory() || !m0_supports_low_memory())
            throw std::invalid_argument(
                "unit-f32-radius-f64-v1 requires state-free R0 and M0 implementations.");
        if (m1_polynomial_policy != M1PolynomialPolicy::recompute)
            throw std::invalid_argument(
                "unit-f32-radius-f64-v1 requires M1 recomputation.");
        if (factorized_observer_enabled || execution_parameter_gradients_enabled)
            throw std::invalid_argument(
                "unit-f32-radius-f64-v1 does not support observers or parameter gradients.");
    }
    if (requested == edge_geometry_policy)
        return;
    if (requested != EdgeGeometryPolicy::unit_f32_radius_f64
        && harmonic_storage_policy == HarmonicStoragePolicy::y_only_direct)
        set_harmonic_storage_policy("retained");
    factorized_execution_space.fence("Change compact edge geometry policy");
    invalidate_factorized_prepared_graph();
    edge_geometry_policy = requested;
    compact_edge_geometry_active = false;
    execution_prepared_xyz = decltype(execution_prepared_xyz)();
    execution_prepared_unit_direction =
        decltype(execution_prepared_unit_direction)();
    execution_prepared_r = decltype(execution_prepared_r)();
    single_layer_workspace_xyz = {};
    execution_geometry_capacity_edges = 0;
    execution_geometry_planned_capacity_edges = 0;
}

template <typename Precision>
void MACEKokkos<Precision>::set_allow_fixed_workspace(const bool enabled)
{
    if (enabled == allow_fixed_workspace_)
        return;
    factorized_execution_space.fence("Change fixed-workspace opt-in");
    invalidate_factorized_prepared_graph();
    allow_fixed_workspace_ = enabled;
}

template <typename Precision>
bool MACEKokkos<Precision>::allow_fixed_workspace() const
{
    return allow_fixed_workspace_;
}

template <typename Precision>
void MACEKokkos<Precision>::set_low_memory(const bool enabled)
{
    if (!enabled) {
        factorized_execution_space.fence("Disable low-memory policy request");
        invalidate_factorized_prepared_graph();
        apply_low_memory_policy(LowMemoryPolicy::speed);
        low_memory_request = false;
        low_memory_policy = LowMemoryPolicy::disabled;
        low_memory_selection_reason = "low_memory=False";
        low_memory_available = 0;
        low_memory_speed_estimate = 0;
        low_memory_selected_estimate = 0;
        set_harmonic_storage_policy("retained");
        return;
    }

    if (!mace_admits_low_memory(streamed_edges))
        throw std::invalid_argument(
            "Low-memory execution requires direct execution.");
    if (!single_layer_readout
        && factorized_source_strategy != FactorizedSourceStrategy::jit_plugin)
        throw std::invalid_argument(
            "Low-memory execution requires generated RTC execution.");
    if (!r0_supports_low_memory())
        throw std::invalid_argument(
            "Low-memory execution requires a state-free R0 implementation.");
    if (!m0_supports_low_memory())
        throw std::invalid_argument(
            "Low-memory execution requires a state-free M0 implementation.");
    if (factorized_observer_enabled || execution_parameter_gradients_enabled)
        throw std::invalid_argument(
            "Low-memory execution does not support observers or parameter gradients.");

    const int selected_tile = select_m1_recompute_tile_channels(
        m1_recompute_tile_channels, true);
    if (selected_tile == 0) {
        const auto limit = m1_recompute_scratch_limit_bytes();
        std::size_t minimum = 0;
        try {
            minimum = m1_recompute_effective_scratch_bytes(
                8, has_field_coupling, true);
        } catch (const std::length_error&) {
            // Keep zero in the diagnostic when the requirement overflowed.
        }
        throw std::invalid_argument(
            "M1 recomputation on "+m1_recompute_backend_name()
            +" requires at least "+std::to_string(minimum)
            +" scratch bytes per team at tile 8; available limit is "
            +std::to_string(limit)+" bytes.");
    }
    validate_m1_recompute_scratch(selected_tile, true);

    factorized_execution_space.fence("Enable low-memory policy request");
    invalidate_factorized_prepared_graph();
    if (!low_memory_request) {
        low_memory_speed_m1_policy_request = m1_polynomial_policy_request;
        low_memory_speed_m1_recompute =
            m1_polynomial_policy == M1PolynomialPolicy::recompute;
    }
    low_memory_request = true;
    set_m1_recompute_tile_channels(selected_tile);
    apply_low_memory_policy(LowMemoryPolicy::capacity_retained);
    // The retained capacity bundle establishes compact geometry and the host
    // single-layer R0 executor needed by Y-only admission. Select Y-only now
    // when those prerequisites make it available; graph sizing may revise the
    // pending policy later.
    if (y_only_direct_harmonics_admission_reason().empty())
        apply_low_memory_policy(LowMemoryPolicy::capacity_y_only);
    low_memory_policy = LowMemoryPolicy::pending;
    low_memory_selection_reason = "automatic selection pending prepared graph";
    low_memory_available = 0;
    low_memory_speed_estimate = 0;
    low_memory_selected_estimate = 0;
    if (!low_memory_device_memory_info_override) {
        low_memory_device_free = 0;
        low_memory_device_total = 0;
    }
    low_memory_reserve = 0;
    low_memory_capacity_y_only_estimate = 0;
    low_memory_capacity_retained_estimate = 0;
    harmonic_storage_selection_reason =
        "automatic selection pending prepared graph";
}

template <typename Precision>
bool MACEKokkos<Precision>::low_memory_enabled() const
{
    if (!use_mh0_adjoint_reuse() || !readout_recompute)
        return false;
    if constexpr (std::is_same_v<Precision, float>)
        return edge_geometry_policy == EdgeGeometryPolicy::unit_f32_radius_f64;
    return edge_geometry_policy == EdgeGeometryPolicy::cartesian_f64;
}

template <typename Precision>
bool MACEKokkos<Precision>::low_memory_requested() const
{
    return low_memory_request;
}

template <typename Precision>
std::string MACEKokkos<Precision>::low_memory_policy_name() const
{
    switch (low_memory_policy) {
    case LowMemoryPolicy::disabled:
        return "disabled";
    case LowMemoryPolicy::pending:
        return "pending";
    case LowMemoryPolicy::speed:
        return "speed";
    case LowMemoryPolicy::capacity_retained:
        return "capacity-retained";
    case LowMemoryPolicy::capacity_y_only:
        return "capacity-y-only";
    }
    throw std::logic_error("Invalid low-memory policy.");
}

template <typename Precision>
std::string MACEKokkos<Precision>::low_memory_selection_reason_name() const
{
    return low_memory_selection_reason;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::low_memory_available_bytes() const
{
    return low_memory_available;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::low_memory_speed_estimated_bytes() const
{
    return low_memory_speed_estimate;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::low_memory_capacity_estimated_bytes() const
{
    return y_only_direct_harmonics_admission_reason().empty()
        ? low_memory_capacity_y_only_estimate : low_memory_capacity_retained_estimate;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::low_memory_selected_estimated_bytes() const
{
    return low_memory_selected_estimate;
}

template <typename Precision>
std::string MACEKokkos<Precision>::execution_geometry_growth_reason_name() const
{
    return execution_geometry_growth_reason;
}

template <typename Precision>
int MACEKokkos<Precision>::execution_active_receiver_count() const
{
    return execution_active_receivers;
}

template <typename Precision>
int MACEKokkos<Precision>::execution_active_feature_node_count() const
{
    return execution_active_feature_nodes;
}

template <typename Precision>
int MACEKokkos<Precision>::execution_active_edge_count() const
{
    return execution_active_edges;
}

template <typename Precision>
int MACEKokkos<Precision>::execution_planned_receiver_capacity() const
{
    return execution_planned_receivers;
}

template <typename Precision>
int MACEKokkos<Precision>::execution_planned_feature_node_capacity() const
{
    return execution_planned_feature_nodes;
}

template <typename Precision>
int MACEKokkos<Precision>::execution_planned_edge_capacity() const
{
    return execution_planned_edges;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::execution_planned_capacity_bytes() const
{
    return execution_planned_bytes;
}

template <typename Precision>
std::string
MACEKokkos<Precision>::execution_capacity_selection_reason_name() const
{
    return execution_capacity_selection_reason;
}

template <typename Precision>
std::size_t
MACEKokkos<Precision>::execution_result_allocation_count_value() const
{
    return execution_result_allocation_count;
}

template <typename Precision>
int MACEKokkos<Precision>::single_layer_workspace_active_receivers() const
{
    return single_layer_workspace_active_capacity;
}

template <typename Precision>
int MACEKokkos<Precision>::single_layer_workspace_planned_receivers() const
{
    return single_layer_workspace_planned_capacity;
}

template <typename Precision>
int MACEKokkos<Precision>::single_layer_workspace_active_edges() const
{
    return single_layer_workspace_active_edge_capacity;
}

template <typename Precision>
int MACEKokkos<Precision>::single_layer_workspace_planned_edges() const
{
    return single_layer_workspace_planned_edge_capacity;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::single_layer_workspace_bytes() const
{
    return single_layer_workspace_payload_bytes+single_layer_workspace_edge_bytes;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::single_layer_workspace_bytes_per_receiver() const
{
    return single_layer_workspace_receiver_bytes;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::single_layer_workspace_bytes_per_edge() const
{
    return static_cast<std::size_t>(num_lm)*sizeof(Precision)
        +std::size_t(3)*sizeof(double)
        +std::size_t(3)*sizeof(Precision)+sizeof(double)+sizeof(int);
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::single_layer_workspace_replacement_count() const
{
    return single_layer_workspace_replacements;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::single_layer_workspace_reuse_count() const
{
    return single_layer_workspace_reuses;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::single_layer_workspace_batch_count() const
{
    return single_layer_workspace_batches;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::single_layer_tiled_evaluation_count() const
{
    return single_layer_tiled_evaluations;
}

template <typename Precision>
int MACEKokkos<Precision>::dual_layer_workspace_active_receivers() const
{
    return dual_layer_workspace_active_capacity;
}

template <typename Precision>
int MACEKokkos<Precision>::dual_layer_workspace_planned_receivers() const
{
    return dual_layer_workspace_planned_capacity;
}

template <typename Precision>
int MACEKokkos<Precision>::dual_layer_workspace_active_edges() const
{
    return dual_layer_workspace_active_edge_capacity;
}

template <typename Precision>
int MACEKokkos<Precision>::dual_layer_workspace_planned_edges() const
{
    return dual_layer_workspace_planned_edge_capacity;
}

template <typename Precision>
int MACEKokkos<Precision>::dual_layer_source_segment_count() const
{
    return dual_layer_source_segments;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::dual_layer_workspace_bytes() const
{
    return dual_layer_workspace_payload_bytes+dual_layer_workspace_edge_bytes;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::dual_layer_workspace_bytes_per_receiver() const
{
    return dual_layer_workspace_receiver_bytes;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::dual_layer_workspace_bytes_per_edge() const
{
    return static_cast<std::size_t>(num_lm)*sizeof(Precision)
        +std::size_t(3)*sizeof(double)
        +std::size_t(3)*sizeof(Precision)+sizeof(double)
        +std::size_t(2)*sizeof(int);
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::dual_layer_schedule_bytes() const
{
    return dual_layer_source_schedule_bytes;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::
    dual_layer_schedule_preparation_explicit_scratch_bytes() const
{
    return dual_layer_source_schedule_preparation_explicit_scratch_bytes;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::dual_layer_workspace_replacement_count() const
{
    return dual_layer_workspace_replacements;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::dual_layer_workspace_reuse_count() const
{
    return dual_layer_workspace_reuses;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::dual_layer_workspace_batch_count() const
{
    return dual_layer_workspace_batches;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::dual_layer_tiled_evaluation_count() const
{
    return dual_layer_tiled_evaluations;
}

template <typename Precision>
std::string MACEKokkos<Precision>::edge_geometry_policy_name() const
{
    return edge_geometry_policy == EdgeGeometryPolicy::unit_f32_radius_f64
        ? "unit-f32-radius-f64-v1" : "cartesian-f64-v1";
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::compact_edge_geometry_bytes() const
{
    if (edge_geometry_policy != EdgeGeometryPolicy::unit_f32_radius_f64)
        return 0;
    return sizeof(Precision)*execution_prepared_unit_direction.size()
        +sizeof(double)*execution_prepared_r.size();
}

template <typename Precision>
std::string MACEKokkos<Precision>::m1_polynomial_policy_name() const
{
    switch (m1_polynomial_policy) {
    case M1PolynomialPolicy::retained:
        return "retained";
    case M1PolynomialPolicy::recompute:
        return "recompute";
    }
    throw std::logic_error("Invalid M1 polynomial policy.");
}

template <typename Precision>
std::string MACEKokkos<Precision>::m1_polynomial_policy_request_name() const
{
    return m1_polynomial_policy_request;
}

template <typename Precision>
std::string MACEKokkos<Precision>::m1_recompute_backend_name() const
{
    const auto environment = kernel_launch_environment(
        Kokkos::DefaultExecutionSpace());
    return environment.backend.empty() ? "unsupported" : environment.backend;
}

template <typename Precision>
std::string MACEKokkos<Precision>::m1_recompute_fallback_reason_name() const
{
    return m1_recompute_fallback_reason;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::m1_poly_values_capacity_bytes() const
{
    return sizeof(Precision)*M1_poly_values.size();
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::m1_poly_values_active_bytes() const
{
    return m1_polynomial_policy == M1PolynomialPolicy::retained
        ? m1_poly_values_capacity_bytes() : 0;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::m1_poly_adjoints_capacity_bytes() const
{
    return sizeof(Precision)*M1_poly_adjoints.size();
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::m1_poly_adjoints_active_bytes() const
{
    return m1_polynomial_policy == M1PolynomialPolicy::retained
        ? m1_poly_adjoints_capacity_bytes() : 0;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::m1_recompute_scratch_bytes() const
{
    if (m1_polynomial_policy != M1PolynomialPolicy::recompute)
        return 0;
    return m1_recompute_effective_scratch_bytes(
        m1_recompute_tile_channels,
        false,
        mh0_state_policy == MH0StatePolicy::reuse_adjoints);
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::m1_recompute_effective_scratch_bytes(
    const int tile_channels,
    const bool include_field_response,
    const bool require_adjoint_overlap) const
{
    const std::size_t generic_reverse = m1_recompute_required_scratch_bytes(
        tile_channels, include_field_response);
    if (include_field_response)
        return generic_reverse;
    if (!standard_m1_module_ready)
        return generic_reverse;
    if constexpr (std::is_same_v<
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>)
        return symmetrix::standard_m1::host_channel_tile() != 0
            ? 0 : generic_reverse;
#if defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_CUDA)
    if constexpr (symmetrix::standard_m1::supports_device_direct_reverse<
            Kokkos::DefaultExecutionSpace>()) {
        if (!require_adjoint_overlap
            && symmetrix::standard_m1::views_overlap(A1, A1_adj)
            && (A1.data() != A1_adj.data() || A1.span() != A1_adj.span()))
            return generic_reverse;
        // Direct and exact-alias reverse need less scratch than the generic
        // forward polynomial-values tile.
        return generic_reverse/2;
    }
#endif
    return generic_reverse;
}

template <typename Precision>
std::size_t
MACEKokkos<Precision>::m1_recompute_scratch_limit_bytes() const
{
    if (m1_recompute_scratch_limit_override_bytes != 0)
        return m1_recompute_scratch_limit_override_bytes;
    const auto environment = kernel_launch_environment(
        Kokkos::DefaultExecutionSpace());
    const int scratch_level = environment.backend == "host" ? 1 : 0;
    const std::size_t kokkos_limit = static_cast<std::size_t>(
        Kokkos::TeamPolicy<>::scratch_size_max(scratch_level));
    if (environment.backend == "cuda" || environment.backend == "hip")
        return std::min(kokkos_limit, environment.max_shared_memory_per_block);
    return environment.backend == "host" ? kokkos_limit : 0;
}

template <typename Precision>
std::size_t
MACEKokkos<Precision>::macefield_response_m1_recompute_scratch_limit_bytes() const
{
    return m1_recompute_scratch_limit_bytes();
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::m1_recompute_required_scratch_bytes(
    const int tile_channels, const bool include_field_response) const
{
    const std::size_t factor = include_field_response ? 4 : 2;
    const std::size_t nodes = M1_poly_coeff.extent(1);
    const std::size_t tile = static_cast<std::size_t>(tile_channels);
    constexpr std::size_t maximum = std::numeric_limits<std::size_t>::max();
    if (nodes != 0 && factor > maximum/sizeof(Precision)/nodes)
        throw std::length_error("M1 recomputation scratch size overflow.");
    const std::size_t per_channel = factor*sizeof(Precision)*nodes;
    if (tile != 0 && per_channel > maximum/tile)
        throw std::length_error("M1 recomputation scratch size overflow.");
    return per_channel*tile;
}

template <typename Precision>
int MACEKokkos<Precision>::select_m1_recompute_tile_channels(
    const int requested, const bool require_adjoint_overlap) const
{
    if constexpr (std::is_same_v<
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>) {
        if (!has_field_coupling && standard_m1_module_ready
            && symmetrix::standard_m1::host_channel_tile() != 0)
            return requested;
    }
    const std::size_t limit = m1_recompute_scratch_limit_bytes();
    for (const int candidate : {32, 16, 8}) {
        if (candidate > requested)
            continue;
        try {
            if (m1_recompute_effective_scratch_bytes(
                    candidate,
                    has_field_coupling,
                    require_adjoint_overlap) <= limit)
                return candidate;
        } catch (const std::length_error&) {
        }
    }
    return 0;
}

template <typename Precision>
int MACEKokkos<Precision>::m1_recompute_vector_length() const
{
    return m1_recompute_backend_name() == "host"
        ? 1 : m1_recompute_tile_channels;
}

template <typename Precision>
int MACEKokkos<Precision>::m1_recompute_scratch_level() const
{
    return m1_recompute_backend_name() == "host" ? 1 : 0;
}

template <typename Precision>
int MACEKokkos<Precision>::macefield_response_m1_recompute_tile_channels_for(
    const int requested) const
{
    if (!has_field_coupling)
        return select_m1_recompute_tile_channels(requested);
    return select_m1_recompute_tile_channels(requested);
}

template <typename Precision>
void MACEKokkos<Precision>::validate_m1_recompute_scratch(
    const int requested, const bool require_adjoint_overlap) const
{
    const std::size_t required = m1_recompute_effective_scratch_bytes(
        requested, has_field_coupling, require_adjoint_overlap);
    const std::size_t limit = m1_recompute_scratch_limit_bytes();
    if (required > limit)
        throw std::invalid_argument(
            "M1 recomputation on "+m1_recompute_backend_name()+" requires "
            +std::to_string(required)+" scratch bytes per team at tile "
            +std::to_string(requested)+"; available limit is "
            +std::to_string(limit)+" bytes.");
}

template <typename Precision>
int MACEKokkos<Precision>::macefield_response_m1_recompute_tile_channels() const
{
    return macefield_response_m1_recompute_tile_channels_for(
        m1_recompute_tile_channels);
}

template <typename Precision>
std::size_t
MACEKokkos<Precision>::macefield_response_m1_recompute_scratch_bytes() const
{
    if (!has_field_coupling
        || m1_polynomial_policy != M1PolynomialPolicy::recompute)
        return 0;
    return m1_recompute_required_scratch_bytes(
        macefield_response_m1_recompute_tile_channels(), true);
}

template <typename Precision>
void MACEKokkos<Precision>::set_kernel_launch_policy(std::string policy)
{
    if (policy == "automatic")
        kernel_launch_policy = KernelLaunchPolicy::automatic;
    else if (policy == "static")
        kernel_launch_policy = KernelLaunchPolicy::static_profile;
    else
        throw std::invalid_argument(
            "Execution launch policy must be 'automatic' or 'static'.");
    Kokkos::fence();
    kernel_launch_profile_diagnostic_records.clear();
    invalidate_factorized_prepared_graph();
}

template <typename Precision>
void MACEKokkos<Precision>::set_kernel_launch_profile_override(
    std::string profile_id, const int blocks)
{
    const auto* profile =
        symmetrix::execution::find_launch_profile(profile_id);
    if (profile == nullptr)
        throw std::invalid_argument("Unknown Execution launch profile ID.");
    if (!profile->calibration_permitted
        || !symmetrix::execution::is_calibration_candidate(
            *profile, blocks))
        throw std::invalid_argument(
            "Execution launch override is not an admitted calibration candidate.");
    Kokkos::fence();
    kernel_launch_profile_overrides[std::move(profile_id)] = blocks;
    kernel_launch_profile_diagnostic_records.clear();
    invalidate_factorized_prepared_graph();
}

template <typename Precision>
void MACEKokkos<Precision>::set_jit_device_plugin_launch_override(
    std::string stage, const int blocks)
{
    if (stage != "r1_forward" && stage != "r1_reverse")
        throw std::invalid_argument(
            "Execution device plugin launch stage must be 'r1_forward' or "
            "'r1_reverse'.");
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    if (!jit_device_plugin_ready())
        throw std::logic_error(
            "Execution device plugin launch overrides require a loaded plugin.");
    const int default_blocks =
        jit_device_plugin->persistent_blocks_per_compute_unit();
    const int lower = std::max(1, default_blocks/2);
    const int upper = std::min(32, default_blocks*2);
    if (blocks != lower && blocks != default_blocks && blocks != upper)
        throw std::invalid_argument(
            "Execution device plugin launch override is not an admitted candidate.");
    Kokkos::fence();
    jit_device_plugin_launch_overrides[std::move(stage)] = blocks;
    kernel_launch_profile_diagnostic_records.clear();
    invalidate_factorized_prepared_graph();
#else
    static_cast<void>(blocks);
    throw std::logic_error(
        "Execution device plugin launch overrides require CUDA or HIP.");
#endif
}

template <typename Precision>
void MACEKokkos<Precision>::clear_kernel_launch_profile_overrides()
{
    Kokkos::fence();
    kernel_launch_profile_overrides.clear();
    jit_device_plugin_launch_overrides.clear();
    kernel_launch_profile_diagnostic_records.clear();
    invalidate_factorized_prepared_graph();
}

template <typename Precision>
std::string MACEKokkos<Precision>::kernel_launch_policy_name() const
{
    return kernel_launch_policy == KernelLaunchPolicy::automatic
        ? "automatic" : "static";
}

template <typename Precision>
std::vector<KernelLaunchProfileDiagnostic>
MACEKokkos<Precision>::kernel_launch_profile_diagnostics() const
{
    std::vector<KernelLaunchProfileDiagnostic> result;
    result.reserve(kernel_launch_profile_diagnostic_records.size());
    for (const auto& [stage, diagnostic] :
         kernel_launch_profile_diagnostic_records) {
        static_cast<void>(stage);
        result.push_back(diagnostic);
    }
    return result;
}

template <typename Precision>
int MACEKokkos<Precision>::apply_kernel_launch_profile(
    const symmetrix::execution::LaunchProfile* profile,
    const std::string_view implementation_id,
    const std::string_view implementation_kind,
    const bool host,
    const int compute_units,
    const int max_threads_per_block,
    const std::size_t work_items,
    const std::string_view stage)
{
    KernelLaunchProfileDiagnostic diagnostic;
    diagnostic.stage = stage;
    diagnostic.implementation_id = implementation_id;
    diagnostic.implementation_kind = implementation_kind;
    if (profile == nullptr) {
        diagnostic.selection_source = "rejected";
        diagnostic.rejection_reason = "no compatible launch profile";
        kernel_launch_profile_diagnostic_records[diagnostic.stage] = diagnostic;
        return 0;
    }
    diagnostic.profile_id = profile->profile_id;
    diagnostic.default_blocks_per_compute_unit =
        profile->persistent_blocks_per_sm;
    diagnostic.calibration_permitted = profile->calibration_permitted;
    for (int index=0; index<profile->calibration_candidate_count; ++index)
        diagnostic.calibration_candidates.push_back(
            profile->calibration_candidates[static_cast<std::size_t>(index)]);
    if (host) {
        diagnostic.selection_source = "static";
        diagnostic.persistent_blocks = 1;
        kernel_launch_profile_diagnostic_records[diagnostic.stage] = diagnostic;
        return 1;
    }
    if (profile->max_threads_per_block > max_threads_per_block) {
        diagnostic.selection_source = "rejected";
        diagnostic.rejection_reason = "compiled block size exceeds device limit";
        kernel_launch_profile_diagnostic_records[diagnostic.stage] = diagnostic;
        return 0;
    }
    int blocks_per_compute_unit = profile->persistent_blocks_per_sm;
    diagnostic.selection_source = kernel_launch_policy
            == KernelLaunchPolicy::static_profile
        ? "static" : "resource_default";
    const auto override = kernel_launch_profile_overrides.find(
        std::string(profile->profile_id));
    if (kernel_launch_policy == KernelLaunchPolicy::automatic
        && override != kernel_launch_profile_overrides.end()) {
        if (!profile->calibration_permitted
            || !symmetrix::execution::is_calibration_candidate(
                *profile, override->second))
            throw std::invalid_argument(
                "Execution launch override is incompatible with the selected profile.");
        blocks_per_compute_unit = override->second;
        diagnostic.selection_source = "calibrated";
    } else if (kernel_launch_policy == KernelLaunchPolicy::static_profile
        && override != kernel_launch_profile_overrides.end()) {
        diagnostic.rejection_reason = "static policy ignores calibration override";
    }
    if (blocks_per_compute_unit <= 0 || compute_units <= 0) {
        diagnostic.selection_source = "rejected";
        diagnostic.rejection_reason = "persistent launch profile is invalid";
        kernel_launch_profile_diagnostic_records[diagnostic.stage] = diagnostic;
        return 0;
    }
    const std::size_t requested =
        static_cast<std::size_t>(blocks_per_compute_unit)
        *static_cast<std::size_t>(compute_units);
    std::size_t total = requested;
    if (work_items > 0)
        total = std::min(total, work_items);
    if (total > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("Execution persistent grid exceeds INT_MAX blocks.");
    diagnostic.active_blocks_per_compute_unit = blocks_per_compute_unit;
    diagnostic.persistent_blocks = static_cast<int>(std::max<std::size_t>(1, total));
    kernel_launch_profile_diagnostic_records[diagnostic.stage] = diagnostic;
    return diagnostic.persistent_blocks;
}

template <typename Precision>
int MACEKokkos<Precision>::resolve_execution_persistent_blocks(
    const std::string_view implementation_id,
    const Kokkos::DefaultExecutionSpace& execution_space,
    const std::size_t work_items,
    const std::string_view stage)
{
    const auto environment =
        kernel_launch_environment(execution_space);
    return apply_kernel_launch_profile(
        kernel_launch_profile<Precision>(implementation_id, environment),
        implementation_id,
        "module",
        environment.backend == "host",
        environment.multiprocessor_count,
        environment.max_threads_per_block,
        work_items,
        stage);
}

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
template <typename Precision>
int MACEKokkos<Precision>::resolve_jit_device_plugin_persistent_blocks(
    const JitDevicePlugin& plugin,
    const Kokkos::DefaultExecutionSpace& execution_space,
    const std::size_t work_items,
    const std::string_view stage)
{
    const auto environment =
        kernel_launch_environment(execution_space);
    KernelLaunchProfileDiagnostic diagnostic;
    diagnostic.stage = stage;
    diagnostic.implementation_id = plugin.artifact_id();
    diagnostic.implementation_kind = "artifact";
    diagnostic.profile_id = "rtc-"+diagnostic.implementation_id;
    const int default_blocks = plugin.persistent_blocks_per_compute_unit();
    diagnostic.default_blocks_per_compute_unit = default_blocks;
    diagnostic.calibration_permitted = true;
    diagnostic.calibration_candidates = {
        std::max(1, default_blocks/2), default_blocks,
        std::min(32, default_blocks*2)};
    std::sort(
        diagnostic.calibration_candidates.begin(),
        diagnostic.calibration_candidates.end());
    diagnostic.calibration_candidates.erase(
        std::unique(
            diagnostic.calibration_candidates.begin(),
            diagnostic.calibration_candidates.end()),
        diagnostic.calibration_candidates.end());
    int active_blocks = default_blocks;
    diagnostic.selection_source = kernel_launch_policy
            == KernelLaunchPolicy::static_profile
        ? "static" : "resource_default";
    const auto override = jit_device_plugin_launch_overrides.find(
        std::string(stage));
    if (kernel_launch_policy == KernelLaunchPolicy::automatic
        && override != jit_device_plugin_launch_overrides.end()) {
        const int lower = std::max(1, default_blocks/2);
        const int upper = std::min(32, default_blocks*2);
        if (override->second != lower && override->second != default_blocks
            && override->second != upper)
            throw std::invalid_argument(
                "Execution device plugin override is incompatible with its descriptor.");
        active_blocks = override->second;
        diagnostic.selection_source = "calibrated";
    } else if (kernel_launch_policy == KernelLaunchPolicy::static_profile
        && override != jit_device_plugin_launch_overrides.end()) {
        diagnostic.rejection_reason = "static policy ignores calibration override";
    }
    if (active_blocks <= 0 || environment.multiprocessor_count <= 0)
        throw std::runtime_error(
            "Execution device plugin contains an invalid persistent launch default.");
    std::size_t total = static_cast<std::size_t>(active_blocks)
        *static_cast<std::size_t>(environment.multiprocessor_count);
    if (work_items > 0)
        total = std::min(total, work_items);
    if (total > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error(
            "Execution device plugin grid exceeds INT_MAX blocks.");
    diagnostic.active_blocks_per_compute_unit = active_blocks;
    diagnostic.persistent_blocks = static_cast<int>(std::max<std::size_t>(1, total));
    kernel_launch_profile_diagnostic_records[diagnostic.stage] = diagnostic;
    return diagnostic.persistent_blocks;
}
#endif

template <typename Precision>
void MACEKokkos<Precision>::set_factorized_reverse_cache_policy(std::string policy)
{
    if (policy == "automatic")
        factorized_reverse_cache_policy = FactorizedReverseCachePolicy::automatic;
    else if (policy == "recompute")
        factorized_reverse_cache_policy = FactorizedReverseCachePolicy::recompute;
    else if (policy == "retain")
        factorized_reverse_cache_policy = FactorizedReverseCachePolicy::retain;
    else
        throw std::invalid_argument(
            "Execution R1 reverse cache policy must be 'automatic', 'recompute', "
            "or 'retain'.");
    release_factorized_coupling_workspace();
    factorized_schedule_dirty = true;
    invalidate_factorized_prepared_graph();
    plan_factorized_reverse_cache();
}

template <typename Precision>
void MACEKokkos<Precision>::set_factorized_planner_budget_bytes(
    const std::size_t budget_bytes)
{
    if (budget_bytes == 0)
        throw std::invalid_argument(
            "Execution R1 planner budget must be positive.");
    release_factorized_coupling_workspace();
    factorized_planner_budget_bytes = budget_bytes;
    factorized_schedule_dirty = true;
    invalidate_factorized_prepared_graph();
    plan_factorized_reverse_cache();
}

template <typename Precision>
std::string MACEKokkos<Precision>::factorized_source_strategy_name() const
{
    switch (factorized_source_strategy) {
    case FactorizedSourceStrategy::serial_reference:
        return "serial_reference";
    case FactorizedSourceStrategy::team_cached:
        return "team_cached";
    case FactorizedSourceStrategy::tiled_coupling:
        return "tiled_coupling";
    case FactorizedSourceStrategy::jit_plugin:
        return "jit_plugin";
    }
    throw std::logic_error("Invalid Execution R1 source strategy.");
}

template <typename Precision>
std::string MACEKokkos<Precision>::factorized_execution_strategy_name() const
{
    if (factorized_source_strategy == FactorizedSourceStrategy::jit_plugin) {
        if (use_factorized_direct_inference())
            return use_factorized_direct_jit_reverse()
                ? "direct_jit_reverse" : "direct_runtime";
        if (factorized_tiled_ready)
            return "tiled_coupling";
        return "team_cached";
    }
    return factorized_source_strategy_name();
}

template <typename Precision>
std::string MACEKokkos<Precision>::factorized_direct_forward_executor_name() const
{
    switch (factorized_direct_forward_executor) {
    case FactorizedDirectForwardExecutor::automatic:
        return "automatic";
    case FactorizedDirectForwardExecutor::runtime:
        return "runtime";
    case FactorizedDirectForwardExecutor::jit_all:
        return "jit_all";
    }
    throw std::logic_error("Invalid Execution R1 direct forward executor.");
}

template <typename Precision>
std::string MACEKokkos<Precision>::factorized_selected_direct_forward_executor_name()
    const
{
    if (!use_factorized_direct_jit_forward())
        return "runtime";
    return "jit_all";
}

template <typename Precision>
std::string MACEKokkos<Precision>::factorized_direct_reverse_executor_name() const
{
    switch (factorized_direct_reverse_executor) {
    case FactorizedDirectReverseExecutor::automatic:
        return "automatic";
    case FactorizedDirectReverseExecutor::runtime:
        return "runtime";
    case FactorizedDirectReverseExecutor::jit:
        return "jit";
    }
    throw std::logic_error("Invalid Execution R1 direct reverse executor.");
}

template <typename Precision>
std::string MACEKokkos<Precision>::factorized_selected_direct_reverse_executor_name()
    const
{
    return use_factorized_direct_jit_reverse() ? "jit" : "runtime";
}

template <typename Precision>
std::string MACEKokkos<Precision>::standard_r0_executor_name() const
{
    switch (standard_r0_executor) {
    case StandardR0Executor::automatic:
        return "automatic";
    case StandardR0Executor::v1:
        return "v1";
    case StandardR0Executor::v2_receiver:
        return "v2_receiver";
    case StandardR0Executor::v2_edge16:
        return "v2_edge16";
    case StandardR0Executor::v2_edge32:
        return "v2_edge32";
    }
    throw std::logic_error("Invalid Execution R0 executor.");
}

template <typename Precision>
std::string MACEKokkos<Precision>::standard_r0_selected_executor_name() const
{
    if (standard_r0_module_active) {
        switch (standard_r0_executor) {
        case StandardR0Executor::v2_edge16:
            return "v2_edge16";
        case StandardR0Executor::automatic:
            if constexpr (std::is_same_v<
                    typename Kokkos::DefaultExecutionSpace::memory_space,
                    Kokkos::HostSpace>)
                return "v2_receiver";
            return "v2_edge16";
        case StandardR0Executor::v2_edge32:
            return "v2_edge32";
        default:
            return "v2_receiver";
        }
    }
    return "v1";
}

template <typename Precision>
std::string MACEKokkos<Precision>::standard_m0_executor_name() const
{
    switch (standard_m0_executor) {
    case StandardM0Executor::automatic:
        return "automatic";
    case StandardM0Executor::runtime:
        return "runtime";
    case StandardM0Executor::standard:
        return "standard";
    }
    throw std::logic_error("Invalid Execution M0 executor.");
}

template <typename Precision>
std::string MACEKokkos<Precision>::standard_m0_selected_executor_name() const
{
    return use_standard_m0_module() ? "standard" : "runtime";
}

template <typename Precision>
bool MACEKokkos<Precision>::use_standard_m0_module() const
{
    return selected_m0_implementation == M0Implementation::builtin
        && use_m0_module();
}

template <typename Precision>
bool MACEKokkos<Precision>::use_r0_module() const
{
    return (streamed_edges == MACEStreamedEdgesMode::generic
            || mace_uses_prepared_execution(streamed_edges))
        && selected_r0_implementation != R0Implementation::generic
        && !factorized_observer_enabled
        && !execution_parameter_gradients_enabled;
}

template <typename Precision>
bool MACEKokkos<Precision>::use_m0_module() const
{
    return (streamed_edges == MACEStreamedEdgesMode::generic
            || mace_uses_prepared_execution(streamed_edges))
        && selected_m0_implementation != M0Implementation::generic
        && standard_m0_executor != StandardM0Executor::runtime
        && !factorized_observer_enabled
        && !execution_parameter_gradients_enabled;
}

template <typename Precision>
bool MACEKokkos<Precision>::r0_supports_low_memory() const
{
    if (!use_r0_module())
        return false;
    if (selected_r0_implementation == R0Implementation::builtin)
        return standard_r0_module_ready
            && standard_r0_executor != StandardR0Executor::v1;
    return selected_r0_implementation == R0Implementation::device_module
        && r0_device_module_ready();
}

template <typename Precision>
bool MACEKokkos<Precision>::m0_supports_low_memory() const
{
    if (!use_m0_module())
        return false;
    if (selected_m0_implementation == M0Implementation::builtin)
        return standard_m0_module_ready;
    if (selected_m0_implementation == M0Implementation::host_plugin)
        return m0_host_plugin_ready();
    return selected_m0_implementation == M0Implementation::device_module
        && m0_device_module_ready();
}

template <typename Precision>
std::string MACEKokkos<Precision>::r0_implementation_name() const
{
    switch (selected_r0_implementation) {
    case R0Implementation::generic:
        return "generic";
    case R0Implementation::builtin:
        return "builtin";
    case R0Implementation::device_module:
        return "device_module";
    }
    throw std::logic_error("Invalid R0 implementation.");
}

template <typename Precision>
std::string MACEKokkos<Precision>::m0_implementation_name() const
{
    switch (selected_m0_implementation) {
    case M0Implementation::generic:
        return "generic";
    case M0Implementation::builtin:
        return "builtin";
    case M0Implementation::device_module:
        return "device_module";
    case M0Implementation::host_plugin:
        return "host_plugin";
    }
    throw std::logic_error("Invalid M0 implementation.");
}

template <typename Precision>
std::string MACEKokkos<Precision>::r0_selected_module_id() const
{
    if (selected_r0_implementation == R0Implementation::builtin)
        return standard_r0_module_id();
    if (selected_r0_implementation == R0Implementation::device_module)
        return r0_device_module_artifact_id();
    return {};
}

template <typename Precision>
std::string MACEKokkos<Precision>::m0_selected_module_id() const
{
    if (selected_m0_implementation == M0Implementation::builtin)
        return standard_m0_module_id();
    if (selected_m0_implementation == M0Implementation::device_module)
        return m0_device_module_artifact_id();
    if (selected_m0_implementation == M0Implementation::host_plugin)
        return m0_host_plugin_artifact_id();
    return {};
}

template <typename Precision>
bool MACEKokkos<Precision>::factorized_direct_profile_requested() const
{
    return mace_uses_direct_execution(streamed_edges)
        && factorized_source_strategy == FactorizedSourceStrategy::jit_plugin
        && (jit_host_plugin_ready() || jit_device_plugin_ready())
        && !factorized_observer_enabled
        && !execution_parameter_gradients_enabled;
}

template <typename Precision>
bool MACEKokkos<Precision>::use_factorized_direct_inference() const
{
    return factorized_direct_profile_requested();
}

template <typename Precision>
bool MACEKokkos<Precision>::use_factorized_direct_jit_forward() const
{
    return use_factorized_direct_inference()
        && (jit_host_plugin_ready() || jit_device_plugin_ready())
        && factorized_direct_forward_executor
            != FactorizedDirectForwardExecutor::runtime;
}

template <typename Precision>
bool MACEKokkos<Precision>::use_factorized_direct_jit_reverse() const
{
    return use_factorized_direct_inference()
        && (jit_host_plugin_ready() || jit_device_plugin_ready())
        && factorized_direct_reverse_executor
            != FactorizedDirectReverseExecutor::runtime;
}

template <typename Precision>
std::string MACEKokkos<Precision>::factorized_execution_profile_name() const
{
    return use_factorized_direct_inference()
        ? "direct_fixed_weight" : "factorized_state";
}

template <typename Precision>
std::string MACEKokkos<Precision>::factorized_derivative_signature_name() const
{
    if (execution_parameter_gradients_enabled)
        return "parameter_coordinate";
    return use_factorized_direct_inference()
        ? "fixed_weight_coordinate" : "factorized_coordinate";
}

template <typename Precision>
std::string MACEKokkos<Precision>::factorized_reverse_cache_policy_name() const
{
    switch (factorized_reverse_cache_policy) {
    case FactorizedReverseCachePolicy::automatic:
        return "automatic";
    case FactorizedReverseCachePolicy::recompute:
        return "recompute";
    case FactorizedReverseCachePolicy::retain:
        return "retain";
    }
    throw std::logic_error("Invalid Execution R1 reverse cache policy.");
}

template <typename Precision>
std::string
MACEKokkos<Precision>::factorized_selected_reverse_cache_policy_name() const
{
    if (use_factorized_direct_inference())
        return "not_applicable";
    const auto retained = std::count(
        factorized_reverse_group_retain.begin(),
        factorized_reverse_group_retain.end(), true);
    if (retained == 0)
        return "recompute";
    if (retained == static_cast<std::ptrdiff_t>(
            factorized_reverse_group_retain.size()))
        return "retain";
    return "mixed";
}

template <typename Precision>
std::vector<int> MACEKokkos<Precision>::factorized_retained_reverse_groups() const
{
    std::vector<int> groups;
    if (use_factorized_direct_inference())
        return groups;
    for (int l=0; l<static_cast<int>(factorized_reverse_group_retain.size()); ++l)
        if (factorized_reverse_group_retain[l])
            groups.push_back(l);
    return groups;
}

template <typename Precision>
std::vector<int> MACEKokkos<Precision>::factorized_recomputed_reverse_groups() const
{
    std::vector<int> groups;
    if (use_factorized_direct_inference())
        return groups;
    for (int l=0; l<static_cast<int>(factorized_reverse_group_retain.size()); ++l)
        if (!factorized_reverse_group_retain[l])
            groups.push_back(l);
    return groups;
}

template <typename Precision>
std::string MACEKokkos<Precision>::factorized_jit_artifact_id() const
{
    if (jit_device_plugin_ready())
        return jit_device_plugin_artifact_id();
    if (jit_host_plugin_ready())
        return jit_host_plugin_artifact_id();
    return {};
}

template <typename Precision>
std::string MACEKokkos<Precision>::factorized_jit_contract_fingerprint() const
{
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    if (jit_device_plugin_ready())
        return std::string(jit_device_plugin->contract_fingerprint());
#endif
    if (jit_host_plugin_ready())
        return std::string(
            jit_host_plugin->descriptor_v2().contract_fingerprint);
    return {};
}

template <typename Precision>
std::string MACEKokkos<Precision>::standard_r0_module_id() const
{
    return standard_r0_module_ready
        ? std::string(symmetrix::standard_r0::module_id) : std::string();
}

template <typename Precision>
int MACEKokkos<Precision>::standard_r0_module_revision() const
{
    return standard_r0_module_ready ? symmetrix::standard_r0::module_revision : 0;
}

template <typename Precision>
std::string MACEKokkos<Precision>::standard_m0_module_id() const
{
    if (!standard_m0_module_ready)
        return {};
    switch (standard_m0_module_variant) {
    case StandardM0ModuleVariant::scalar_lmax0:
        return symmetrix::standard_m0::scalar_module_id;
    case StandardM0ModuleVariant::full_lmax1:
        return symmetrix::standard_m0::module_id;
    case StandardM0ModuleVariant::none:
        return {};
    }
    return {};
}

template <typename Precision>
int MACEKokkos<Precision>::standard_m0_module_revision() const
{
    if (!standard_m0_module_ready)
        return 0;
    switch (standard_m0_module_variant) {
    case StandardM0ModuleVariant::scalar_lmax0:
        return symmetrix::standard_m0::scalar_module_revision;
    case StandardM0ModuleVariant::full_lmax1:
        return symmetrix::standard_m0::module_revision;
    case StandardM0ModuleVariant::none:
        return 0;
    }
    return 0;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::standard_m0_poly_values_capacity_bytes() const
{
    std::size_t elements = 0;
    for (std::size_t LM=0; LM<M0_poly_values.extent(0); ++LM)
        elements += M0_poly_values(LM).size();
    return sizeof(Precision)*elements;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::standard_m0_poly_values_active_bytes() const
{
    return use_m0_module()
        ? 0 : standard_m0_poly_values_capacity_bytes();
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::standard_m0_poly_adjoints_capacity_bytes() const
{
    std::size_t elements = 0;
    for (std::size_t LM=0; LM<M0_poly_adjoints.extent(0); ++LM)
        elements += M0_poly_adjoints(LM).size();
    return sizeof(Precision)*elements;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::standard_m0_poly_adjoints_active_bytes() const
{
    return use_m0_module()
        ? 0 : standard_m0_poly_adjoints_capacity_bytes();
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::factorized_forward_coupling_workspace_bytes() const
{
    return factorized_coupling_workspace_bytes;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::factorized_reverse_coupling_workspace_bytes() const
{
    return factorized_coupling_workspace_bytes;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::factorized_custom_blas_launch_count() const
{
    return factorized_blas_context == nullptr
        ? 0 : factorized_blas_context->launch_count;
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::factorized_blas_stream_bind_count() const
{
    return factorized_blas_context == nullptr
        ? 0 : factorized_blas_context->stream_bind_count;
}

template <typename Precision>
std::size_t
MACEKokkos<Precision>::estimate_factorized_factorized_core_workspace_bytes() const
{
    int maximum_chunk_edges = 0;
    const int available_chunks = std::min(
        execution_schedule_num_chunks,
        std::max(0, static_cast<int>(execution_chunk_edge_offsets_host.size())-1));
    for (int chunk=0; chunk<available_chunks; ++chunk)
        maximum_chunk_edges = std::max(
            maximum_chunk_edges,
            execution_chunk_edge_offsets_host[chunk+1]
                -execution_chunk_edge_offsets_host[chunk]);
    const std::size_t radial_bytes = 2*sizeof(Precision)
        *static_cast<std::size_t>(maximum_chunk_edges)
        *static_cast<std::size_t>(factorized_embedding_width);
    return estimate_factorized_state_workspace_bytes(
        factorized_chunk_size, execution_schedule_num_feature_nodes)
        +radial_bytes+factorized_schedule_bytes;
}

template <typename Precision>
void MACEKokkos<Precision>::refresh_factorized_workspace_readiness()
{
    const std::size_t factorized_core_bytes =
        estimate_factorized_factorized_core_workspace_bytes();
    factorized_tiled_ready = factorized_core_bytes
        +factorized_coupling_capacity_bytes
        <= factorized_default_planner_budget_bytes;
}

template <typename Precision>
void MACEKokkos<Precision>::update_factorized_workspace_accounting()
{
    factorized_radial_workspace_bytes = sizeof(Precision)*(
        execution_radial_values.size()+execution_radial_derivatives.size());
    factorized_coupling_workspace_bytes =
        sizeof(Precision)*execution_coupling_adjoint.size();
    factorized_workspace_bytes = factorized_state_workspace_bytes
        +factorized_radial_workspace_bytes+factorized_coupling_workspace_bytes
        +factorized_schedule_bytes;
    factorized_workspace_capacity_bytes = factorized_state_workspace_bytes
        +factorized_radial_workspace_bytes
        +std::max(
            factorized_coupling_workspace_bytes,
            factorized_coupling_capacity_bytes)
        +factorized_schedule_bytes;
    refresh_factorized_workspace_readiness();
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::estimate_factorized_state_workspace_bytes(
    const int chunk_size,
    const int num_nodes) const
{
    const std::size_t packed_elements = static_cast<std::size_t>(chunk_size)
        *static_cast<std::size_t>(factorized_embedding_width)
        *static_cast<std::size_t>(factorized_max_coupling_columns);
    int maximum_message_rows = 0;
    int maximum_compact_width = 0;
    std::size_t maximum_legacy_elements = 0;
    for (int l=0;
         l+1<static_cast<int>(execution_group_path_offsets_host.size()); ++l) {
        const int paths = execution_group_path_offsets_host[l+1]
            -execution_group_path_offsets_host[l];
        const int rows = chunk_size*(2*l+1);
        const int contracted_width =
            paths*factorized_embedding_width*num_channels;
        maximum_legacy_elements = std::max(
            maximum_legacy_elements,
            static_cast<std::size_t>(rows)
                *static_cast<std::size_t>(contracted_width+num_channels));
        maximum_message_rows = std::max(maximum_message_rows, rows);
        maximum_compact_width = std::max(
            maximum_compact_width, paths*num_channels);
    }
    const std::size_t compact_elements =
        static_cast<std::size_t>(maximum_message_rows)
        *static_cast<std::size_t>(maximum_compact_width);
    const std::size_t source_compensation_elements =
        static_cast<std::size_t>(num_nodes)
        *static_cast<std::size_t>(num_LM)
        *static_cast<std::size_t>(num_channels);
    return sizeof(Precision)*(packed_elements+std::max(
        maximum_legacy_elements, compact_elements)
        +source_compensation_elements);
}

template <typename Precision>
typename MACEKokkos<Precision>::ExecutionScratchMatrix
MACEKokkos<Precision>::factorized_state_view(const int l) const
{
    const int paths = execution_group_path_offsets_host[l+1]
        -execution_group_path_offsets_host[l];
    const std::size_t packed_elements =
        static_cast<std::size_t>(factorized_chunk_size)
        *static_cast<std::size_t>(factorized_embedding_width)
        *static_cast<std::size_t>(factorized_max_coupling_columns);
    return ExecutionScratchMatrix(
        factorized_workspace_arena.data()+packed_elements,
        factorized_chunk_size*(2*l+1),
        paths*factorized_embedding_width*num_channels);
}

template <typename Precision>
typename MACEKokkos<Precision>::ExecutionScratchMatrix
MACEKokkos<Precision>::factorized_output_view(const int l) const
{
    const int paths = execution_group_path_offsets_host[l+1]
        -execution_group_path_offsets_host[l];
    const int rows = factorized_chunk_size*(2*l+1);
    const std::size_t packed_elements =
        static_cast<std::size_t>(factorized_chunk_size)
        *static_cast<std::size_t>(factorized_embedding_width)
        *static_cast<std::size_t>(factorized_max_coupling_columns);
    const std::size_t state_elements = static_cast<std::size_t>(rows)
        *static_cast<std::size_t>(
            paths*factorized_embedding_width*num_channels);
    return ExecutionScratchMatrix(
        factorized_workspace_arena.data()+packed_elements+state_elements,
        rows, num_channels);
}

template <typename Precision>
typename MACEKokkos<Precision>::ExecutionScratchTensor
MACEKokkos<Precision>::factorized_packed_state_view() const
{
    return ExecutionScratchTensor(
        factorized_workspace_arena.data(),
        factorized_chunk_size, factorized_embedding_width,
        factorized_max_coupling_columns);
}

template <typename Precision>
typename MACEKokkos<Precision>::ExecutionScratchMatrix
MACEKokkos<Precision>::factorized_compact_message_view() const
{
    int maximum_message_rows = 0;
    int maximum_compact_width = 0;
    for (int l=0;
         l+1<static_cast<int>(execution_group_path_offsets_host.size()); ++l) {
        const int paths = execution_group_path_offsets_host[l+1]
            -execution_group_path_offsets_host[l];
        maximum_message_rows = std::max(
            maximum_message_rows, factorized_chunk_size*(2*l+1));
        maximum_compact_width = std::max(
            maximum_compact_width, paths*num_channels);
    }
    const std::size_t packed_elements =
        static_cast<std::size_t>(factorized_chunk_size)
        *static_cast<std::size_t>(factorized_embedding_width)
        *static_cast<std::size_t>(factorized_max_coupling_columns);
    return ExecutionScratchMatrix(
        factorized_workspace_arena.data()+packed_elements,
        maximum_message_rows, maximum_compact_width);
}

template <typename Precision>
typename MACEKokkos<Precision>::ExecutionScratchTensor
MACEKokkos<Precision>::factorized_source_compensation_view() const
{
    const std::size_t packed_elements =
        static_cast<std::size_t>(factorized_chunk_size)
        *static_cast<std::size_t>(factorized_embedding_width)
        *static_cast<std::size_t>(factorized_max_coupling_columns);
    int maximum_message_rows = 0;
    int maximum_compact_width = 0;
    std::size_t maximum_legacy_elements = 0;
    for (int l=0;
         l+1<static_cast<int>(execution_group_path_offsets_host.size()); ++l) {
        const int paths = execution_group_path_offsets_host[l+1]
            -execution_group_path_offsets_host[l];
        const int rows = factorized_chunk_size*(2*l+1);
        maximum_message_rows = std::max(maximum_message_rows, rows);
        maximum_compact_width = std::max(
            maximum_compact_width, paths*num_channels);
        maximum_legacy_elements = std::max(
            maximum_legacy_elements,
            static_cast<std::size_t>(rows)
                *static_cast<std::size_t>(
                    paths*factorized_embedding_width*num_channels+num_channels));
    }
    const std::size_t compact_elements =
        static_cast<std::size_t>(maximum_message_rows)
        *static_cast<std::size_t>(maximum_compact_width);
    return ExecutionScratchTensor(
        factorized_workspace_arena.data()+packed_elements
            +std::max(maximum_legacy_elements, compact_elements),
        execution_schedule_num_feature_nodes, num_LM, num_channels);
}

template <typename Precision>
int MACEKokkos<Precision>::select_factorized_chunk_size(
    const int num_receivers,
    const int num_feature_nodes,
    const int num_edges,
    const std::vector<int>& num_neigh)
{
    struct Candidate {
        int chunk_size;
        std::size_t core_bytes;
        std::size_t retained_bytes;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(factorized_default_chunk_size);
    for (int chunk_size=factorized_default_chunk_size;
         chunk_size>=1; --chunk_size) {
        const int num_chunks =
            (num_receivers+chunk_size-1)/chunk_size;
        int maximum_chunk_edges = 0;
        for (int receiver_begin=0; receiver_begin<num_receivers;
             receiver_begin += chunk_size) {
            int chunk_edges = 0;
            const int receiver_end = std::min(
                num_receivers, receiver_begin+chunk_size);
            for (int receiver=receiver_begin;
                 receiver<receiver_end; ++receiver)
                chunk_edges += num_neigh[receiver];
            maximum_chunk_edges = std::max(
                maximum_chunk_edges, chunk_edges);
        }
        const std::size_t state_bytes =
            estimate_factorized_state_workspace_bytes(
                chunk_size, num_feature_nodes);
        const std::size_t radial_bytes = 2*sizeof(Precision)
            *static_cast<std::size_t>(maximum_chunk_edges)
            *static_cast<std::size_t>(factorized_embedding_width);
        const std::size_t schedule_bytes = sizeof(int)*(
            static_cast<std::size_t>(num_chunks)
                *(static_cast<std::size_t>(num_feature_nodes)+1)
            +3*static_cast<std::size_t>(num_edges)
            +static_cast<std::size_t>(num_feature_nodes)+1
            +static_cast<std::size_t>(num_receivers));
        const std::size_t retained_bytes = sizeof(Precision)
            *static_cast<std::size_t>(maximum_chunk_edges)
            *static_cast<std::size_t>(factorized_max_coupling_columns);
        candidates.push_back({
            chunk_size,
            state_bytes+radial_bytes+schedule_bytes,
            retained_bytes,
        });
    }

    if (factorized_direct_profile_requested()) {
        factorized_tile_selection_reason =
            "JIT direct profile uses compact source schedule";
        return factorized_default_chunk_size;
    }

    if (factorized_reverse_cache_policy
            != FactorizedReverseCachePolicy::recompute)
        for (const auto& candidate : candidates)
            if (candidate.core_bytes+candidate.retained_bytes
                    <= factorized_planner_budget_bytes) {
                factorized_tile_selection_reason =
                    candidate.chunk_size == factorized_default_chunk_size
                    ? "default tile retains all reverse groups"
                    : "reduced tile retains all reverse groups";
                return candidate.chunk_size;
            }
    for (const auto& candidate : candidates)
        if (candidate.core_bytes <= factorized_planner_budget_bytes) {
            factorized_tile_selection_reason =
                factorized_reverse_cache_policy
                    == FactorizedReverseCachePolicy::recompute
                ? "largest tile fitting forced recompute workspace"
                : "largest tile fitting core workspace with group recompute";
            return candidate.chunk_size;
        }
    const auto minimum = std::min_element(
        candidates.begin(), candidates.end(),
        [] (const Candidate& left, const Candidate& right) {
            if (left.core_bytes != right.core_bytes)
                return left.core_bytes < right.core_bytes;
            return left.chunk_size > right.chunk_size;
        });
    factorized_tile_selection_reason =
        "minimum footprint exceeds planner budget";
    return minimum->chunk_size;
}

template <typename Precision>
void MACEKokkos<Precision>::resize_factorized_tile_workspace(
    const int chunk_size,
    const int num_nodes)
{
    const std::size_t required_bytes =
        estimate_factorized_state_workspace_bytes(chunk_size, num_nodes);
    if (chunk_size == factorized_chunk_size
        && sizeof(Precision)*factorized_workspace_arena.size()
            == required_bytes)
        return;
    Kokkos::fence();
    release_factorized_coupling_workspace();
    int maximum_message_rows = 0;
    int maximum_compact_width = 0;
    for (int l=0;
         l+1<static_cast<int>(execution_group_path_offsets_host.size()); ++l) {
        const int paths = execution_group_path_offsets_host[l+1]
            -execution_group_path_offsets_host[l];
        const int rows = chunk_size*(2*l+1);
        maximum_message_rows = std::max(maximum_message_rows, rows);
        maximum_compact_width = std::max(
            maximum_compact_width, paths*num_channels);
    }
    factorized_workspace_arena = decltype(factorized_workspace_arena)(
        Kokkos::view_alloc(
            "Execution R1 workspace arena", Kokkos::WithoutInitializing),
        required_bytes/sizeof(Precision));
    factorized_arena_allocation_count += 1;
    factorized_chunk_size = chunk_size;
    factorized_compact_workspace_bytes =
        sizeof(Precision)*static_cast<std::size_t>(maximum_message_rows)
        *static_cast<std::size_t>(maximum_compact_width);
    factorized_state_workspace_bytes = required_bytes;
    update_factorized_workspace_accounting();
}

template <typename Precision>
void MACEKokkos<Precision>::ensure_factorized_stateful_workspace()
{
    resize_factorized_tile_workspace(
        factorized_chunk_size, execution_schedule_num_feature_nodes);
    int maximum_chunk_edges = 0;
    for (int chunk=0; chunk<execution_schedule_num_chunks; ++chunk)
        maximum_chunk_edges = std::max(
            maximum_chunk_edges,
            execution_chunk_edge_offsets_host[chunk+1]
                -execution_chunk_edge_offsets_host[chunk]);
    if (execution_radial_values.extent(0)
            != static_cast<std::size_t>(maximum_chunk_edges)
        || execution_radial_values.extent(1)
            != static_cast<std::size_t>(factorized_embedding_width))
        Kokkos::realloc(
            execution_radial_values,
            maximum_chunk_edges, factorized_embedding_width);
    if (execution_radial_derivatives.extent(0)
            != static_cast<std::size_t>(maximum_chunk_edges)
        || execution_radial_derivatives.extent(1)
            != static_cast<std::size_t>(factorized_embedding_width))
        Kokkos::realloc(
            execution_radial_derivatives,
            maximum_chunk_edges, factorized_embedding_width);
    update_factorized_workspace_accounting();
    plan_factorized_reverse_cache();
}

template <typename Precision>
void MACEKokkos<Precision>::release_factorized_stateful_workspace()
{
    if (use_factorized_direct_inference()) {
        std::fill(
            factorized_reverse_group_retain.begin(),
            factorized_reverse_group_retain.end(), false);
        factorized_planned_coupling_workspace_bytes = 0;
        factorized_planned_coupling_columns = 0;
    }
    if (factorized_workspace_arena.data() == nullptr
        && execution_radial_values.data() == nullptr
        && execution_radial_derivatives.data() == nullptr
        && execution_coupling_adjoint.data() == nullptr) {
        refresh_factorized_workspace_readiness();
        return;
    }
    Kokkos::fence();
    release_factorized_coupling_workspace();
    factorized_workspace_arena = {};
    execution_radial_values = {};
    execution_radial_derivatives = {};
    factorized_state_workspace_bytes = 0;
    factorized_compact_workspace_bytes = 0;
    update_factorized_workspace_accounting();
}

template <typename Precision>
void MACEKokkos<Precision>::plan_factorized_reverse_cache()
{
    const int group_count = static_cast<int>(
        execution_group_path_offsets_host.empty()
            ? 0 : execution_group_path_offsets_host.size()-1);
    factorized_reverse_group_retain.assign(group_count, false);
    factorized_reverse_group_workspace_bytes.assign(group_count, 0);
    factorized_planned_coupling_workspace_bytes = 0;
    factorized_planned_coupling_columns = 0;

    int maximum_chunk_edges = 0;
    for (int chunk=0; chunk<execution_schedule_num_chunks; ++chunk)
        maximum_chunk_edges = std::max(
            maximum_chunk_edges,
            execution_chunk_edge_offsets_host[chunk+1]
                -execution_chunk_edge_offsets_host[chunk]);
    const std::size_t core_workspace_bytes =
        estimate_factorized_factorized_core_workspace_bytes();
    const bool direct_inference = use_factorized_direct_inference();
    for (int l=0; l<group_count; ++l) {
        const int eta = execution_group_path_offsets_host[l+1]
            -execution_group_path_offsets_host[l];
        const int columns = eta*(2*l+1)*num_channels;
        const std::size_t group_bytes = sizeof(Precision)
            *static_cast<std::size_t>(maximum_chunk_edges)
            *static_cast<std::size_t>(columns);
        factorized_reverse_group_workspace_bytes[l] = group_bytes;
        const bool retain =
            !direct_inference
            && factorized_reverse_cache_policy
                != FactorizedReverseCachePolicy::recompute
            && core_workspace_bytes+group_bytes <= factorized_planner_budget_bytes;
        factorized_reverse_group_retain[l] = retain;
        if (retain && columns > factorized_planned_coupling_columns) {
            factorized_planned_coupling_columns = columns;
            factorized_planned_coupling_workspace_bytes = group_bytes;
        }
    }
    refresh_factorized_workspace_readiness();
}

template <typename Precision>
void MACEKokkos<Precision>::ensure_factorized_coupling_workspace(
    const int coupling_columns)
{
    int maximum_chunk_edges = 0;
    for (int chunk=0; chunk<execution_schedule_num_chunks; ++chunk)
        maximum_chunk_edges = std::max(
            maximum_chunk_edges,
            execution_chunk_edge_offsets_host[chunk+1]
                -execution_chunk_edge_offsets_host[chunk]);
    if (execution_coupling_adjoint.extent(0)
            != static_cast<std::size_t>(maximum_chunk_edges)
        || execution_coupling_adjoint.extent(1)
            != static_cast<std::size_t>(coupling_columns))
        Kokkos::realloc(
            execution_coupling_adjoint,
            maximum_chunk_edges,
            coupling_columns);
    update_factorized_workspace_accounting();
}

template <typename Precision>
void MACEKokkos<Precision>::release_factorized_coupling_workspace()
{
    if (execution_coupling_adjoint.size() == 0)
        return;
    Kokkos::fence();
    execution_coupling_adjoint = {};
    update_factorized_workspace_accounting();
}

template <typename Precision>
void MACEKokkos<Precision>::set_factorized_observer(
    const bool enabled,
    const std::size_t max_bytes)
{
    if (enabled && max_bytes == 0)
        throw std::invalid_argument("Execution R1 observer byte limit must be positive.");
    if (enabled && !mace_uses_direct_execution(streamed_edges))
        throw std::invalid_argument(
            "Execution R1 observation requires "
            "streamed_edges='direct'.");
    if (enabled && !low_memory_request
        && low_memory_policy == LowMemoryPolicy::speed
        && (use_mh0_adjoint_reuse()
            || edge_geometry_policy == EdgeGeometryPolicy::unit_f32_radius_f64)) {
        apply_low_memory_policy(LowMemoryPolicy::speed);
        low_memory_policy = LowMemoryPolicy::speed;
    }
    if (enabled
        && edge_geometry_policy == EdgeGeometryPolicy::unit_f32_radius_f64)
        throw std::invalid_argument(
            "Compact edge geometry does not support the factorized observer.");
    const bool profile_changed = factorized_observer_enabled != enabled;
    factorized_observer_enabled = enabled;
    if (enabled)
        factorized_observer_max_bytes = max_bytes;
    factorized_observer_ready = false;
    factorized_observer_bytes = 0;
    factorized_observer_num_nodes = 0;
    factorized_observer_num_edges = 0;
    std::vector<std::vector<Precision>>().swap(factorized_observer_state);
    std::vector<std::vector<Precision>>().swap(
        factorized_observer_state_adjoint);
    std::vector<Precision>().swap(factorized_observer_output);
    std::vector<Precision>().swap(factorized_observer_output_adjoint);
    std::vector<Precision>().swap(standard_r0_observer_output);
    std::vector<Precision>().swap(standard_r0_observer_output_adjoint);
    std::vector<Precision>().swap(factorized_observer_source_h1_delta);
    std::vector<Precision>().swap(factorized_observer_source_h1_before);
    std::vector<double>().swap(factorized_observer_directed_force_delta);
    std::vector<double>().swap(factorized_observer_directed_force_before);
    std::vector<double>().swap(standard_r0_observer_directed_force_delta);
    std::vector<double>().swap(standard_r0_observer_directed_force_before);
    if (profile_changed) {
        factorized_schedule_dirty = true;
        invalidate_factorized_prepared_graph();
    }
    plan_factorized_reverse_cache();
    if (use_factorized_direct_inference())
        release_factorized_stateful_workspace();
}

template <typename Precision>
void MACEKokkos<Precision>::set_execution_parameter_gradients(
    const bool enabled,
    const std::size_t max_bytes)
{
    if (!enabled) {
        const bool profile_changed = execution_parameter_gradients_enabled;
        execution_parameter_gradients_enabled = false;
        execution_parameter_gradients_ready = false;
        execution_parameter_gradients_result_bytes = 0;
        execution_parameter_gradients_workspace_bytes = 0;
        execution_parameter_gradients_r1_ms = 0.0;
        execution_parameter_gradients_r0_ms = 0.0;
        execution_parameter_gradients_density_ms = 0.0;
        execution_parameter_gradients_r1_workers = 0;
        execution_parameter_gradients_r0_workers = 0;
        std::vector<ExecutionParameterGradientGroup>().swap(
            execution_parameter_gradient_groups);
        if (profile_changed) {
            factorized_schedule_dirty = true;
            invalidate_factorized_prepared_graph();
        }
        plan_factorized_reverse_cache();
        if (use_factorized_direct_inference())
            release_factorized_stateful_workspace();
        return;
    }
    if (max_bytes == 0)
        throw std::invalid_argument(
            "Execution parameter-gradient byte limit must be positive.");
    if (!low_memory_request && low_memory_policy == LowMemoryPolicy::speed
        && (use_mh0_adjoint_reuse()
            || edge_geometry_policy == EdgeGeometryPolicy::unit_f32_radius_f64)) {
        apply_low_memory_policy(LowMemoryPolicy::speed);
        low_memory_policy = LowMemoryPolicy::speed;
    }
    if (!mace_uses_direct_execution(streamed_edges))
        throw std::invalid_argument(
            "Execution parameter gradients require "
            "streamed_edges='direct'.");
    if (edge_geometry_policy == EdgeGeometryPolicy::unit_f32_radius_f64)
        throw std::invalid_argument(
            "Compact edge geometry does not support parameter gradients.");
    if (first_interaction_residual)
        throw std::invalid_argument(
            "Execution parameter gradients do not yet support residual-first MACE models.");
    if (!supports_factorized() || !compact_radial_model)
        throw std::invalid_argument(
            "Execution parameter gradients require an admitted compact Execution model.");

    auto groups = std::vector<ExecutionParameterGradientGroup>();
    std::size_t result_bytes = 0;
    const auto add_group = [&] (
        std::string name,
        std::string layout,
        std::vector<std::size_t> shape) {
        std::size_t elements = 1;
        for (const auto extent : shape) {
            if (extent != 0
                && elements > std::numeric_limits<std::size_t>::max()/extent)
                throw std::length_error(
                    "Execution parameter-gradient result size overflow.");
            elements *= extent;
        }
        if (elements > std::numeric_limits<std::size_t>::max()/sizeof(double))
            throw std::length_error(
                "Execution parameter-gradient result size overflow.");
        const std::size_t bytes = elements*sizeof(double);
        if (bytes > max_bytes-result_bytes)
            throw std::length_error(
                "Execution parameter gradients require more than the configured limit of "
                +std::to_string(max_bytes)+" host bytes.");
        result_bytes += bytes;
        groups.push_back({
            std::move(name), std::move(layout), std::move(shape),
            std::vector<double>(elements, 0.0)});
    };

    const auto add_compact_network = [&] (const std::string& network) {
        const auto shapes = compact_radial_model->network_weight_shapes(network);
        for (std::size_t layer=0; layer<shapes.size(); ++layer)
            add_group(
                "compact_radial."+network+".weights."+std::to_string(layer),
                "output,input",
                {static_cast<std::size_t>(shapes[layer].first),
                 static_cast<std::size_t>(shapes[layer].second)});
    };
    add_compact_network("R0");
    add_compact_network("R1");
    if (compact_radial_model->has_A0())
        add_compact_network("A0");
    if (compact_radial_model->has_A1())
        add_compact_network("A1");

    const std::size_t species = atomic_numbers_host.size();
    if (H0_weights_host.size()
        != species*static_cast<std::size_t>(num_channels))
        throw std::runtime_error(
            "Execution H0 parameter layout is inconsistent with model extents.");
    add_group(
        "H0_weights", "source_type,channel",
        {species, static_cast<std::size_t>(num_channels)});

    if (A0_weights_host.size() != species || A0_weights_host.empty())
        throw std::runtime_error(
            "Execution A0 parameter layout is inconsistent with model species.");
    const std::size_t A0_group_count = A0_weights_host.front().size();
    for (const auto& receiver : A0_weights_host)
        if (receiver.size() != A0_group_count)
            throw std::runtime_error(
                "Execution A0 parameter groups have inconsistent extents.");
    for (std::size_t l=0; l<A0_group_count; ++l) {
        const std::size_t matrix_elements = A0_weights_host.front()[l].size();
        if (matrix_elements%num_channels != 0)
            throw std::runtime_error(
                "Execution A0 parameter matrix has an invalid channel extent.");
        for (const auto& receiver : A0_weights_host)
            if (receiver[l].size() != matrix_elements)
                throw std::runtime_error(
                    "Execution A0 parameter matrices have inconsistent extents.");
        add_group(
            "A0_weights.l"+std::to_string(l),
            "receiver_type,input_channel,output_channel",
            {species, matrix_elements/static_cast<std::size_t>(num_channels),
             static_cast<std::size_t>(num_channels)});
    }

    for (int l=0; l<=l_max; ++l)
        add_group(
            "A1_weights.l"+std::to_string(l),
            "path_input_channel,output_channel",
            {A1_weights(l).extent(0), A1_weights(l).extent(1)});

    const bool profile_changed = !execution_parameter_gradients_enabled;
    execution_parameter_gradients_enabled = true;
    execution_parameter_gradients_ready = false;
    execution_parameter_gradients_max_bytes = max_bytes;
    execution_parameter_gradients_result_bytes = result_bytes;
    execution_parameter_gradients_workspace_bytes = result_bytes;
    execution_parameter_gradient_groups = std::move(groups);
    if (profile_changed) {
        factorized_schedule_dirty = true;
        invalidate_factorized_prepared_graph();
    }
    plan_factorized_reverse_cache();
}

template <typename Precision>
ExecutionParameterGradientGroup&
MACEKokkos<Precision>::execution_parameter_gradient_group(const std::string& name)
{
    const auto group = std::find_if(
        execution_parameter_gradient_groups.begin(),
        execution_parameter_gradient_groups.end(),
        [&] (const ExecutionParameterGradientGroup& candidate) {
            return candidate.name == name;
        });
    if (group == execution_parameter_gradient_groups.end())
        throw std::runtime_error(
            "Execution parameter-gradient group is missing: "+name+".");
    return *group;
}

template <typename Precision>
void MACEKokkos<Precision>::begin_execution_parameter_gradients()
{
    if (!execution_parameter_gradients_enabled)
        return;
    if (!mace_uses_direct_execution(streamed_edges))
        throw std::invalid_argument(
            "Execution parameter gradients require "
            "streamed_edges='direct'.");
    for (auto& group : execution_parameter_gradient_groups)
        std::fill(group.values.begin(), group.values.end(), 0.0);
    execution_parameter_gradients_ready = false;
    execution_parameter_gradients_workspace_bytes =
        execution_parameter_gradients_result_bytes;
    execution_parameter_gradients_r1_ms = 0.0;
    execution_parameter_gradients_r0_ms = 0.0;
    execution_parameter_gradients_density_ms = 0.0;
    execution_parameter_gradients_r1_workers = 0;
    execution_parameter_gradients_r0_workers = 0;
}

template <typename Precision>
void MACEKokkos<Precision>::admit_execution_parameter_gradient_scratch(
    const std::size_t scratch_bytes)
{
    if (scratch_bytes
        > execution_parameter_gradients_max_bytes
            -execution_parameter_gradients_result_bytes)
        throw std::length_error(
            "Execution parameter gradients require "
            +std::to_string(
                execution_parameter_gradients_result_bytes+scratch_bytes)
            +" host bytes, exceeding the configured limit of "
            +std::to_string(execution_parameter_gradients_max_bytes)+".");
    execution_parameter_gradients_workspace_bytes = std::max(
        execution_parameter_gradients_workspace_bytes,
        execution_parameter_gradients_result_bytes+scratch_bytes);
}

template <typename Precision>
void MACEKokkos<Precision>::begin_factorized_observation(
    const int num_nodes,
    const int num_edges)
{
    if (!factorized_observer_enabled)
        return;
    if (num_nodes < 0 || num_edges < 0)
        throw std::invalid_argument("Execution R1 observer extents must be nonnegative.");

    const auto checked_add = [] (std::size_t& total, const std::size_t value) {
        if (value > std::numeric_limits<std::size_t>::max()-total)
            throw std::length_error("Execution R1 observer size overflow.");
        total += value;
    };
    const auto checked_product = [] (std::initializer_list<std::size_t> factors) {
        std::size_t product = 1;
        for (const auto factor : factors) {
            if (factor != 0
                && product > std::numeric_limits<std::size_t>::max()/factor)
                throw std::length_error("Execution R1 observer size overflow.");
            product *= factor;
        }
        return product;
    };

    std::size_t precision_elements = 0;
    for (int l=0; l<=l_max; ++l) {
        const std::size_t components = 2*l+1;
        const std::size_t num_eta =
            execution_group_path_offsets_host[l+1]-execution_group_path_offsets_host[l];
        const std::size_t state_elements = checked_product({
            static_cast<std::size_t>(num_nodes), components, num_eta,
            static_cast<std::size_t>(factorized_embedding_width),
            static_cast<std::size_t>(num_channels)});
        checked_add(precision_elements, state_elements);
        checked_add(precision_elements, state_elements);
    }
    const std::size_t output_elements = checked_product({
        static_cast<std::size_t>(num_nodes), static_cast<std::size_t>(num_lm),
        static_cast<std::size_t>(num_channels)});
    checked_add(precision_elements, output_elements);
    checked_add(precision_elements, output_elements);
    checked_add(precision_elements, output_elements);
    checked_add(precision_elements, output_elements);
    const std::size_t source_elements = checked_product({
        static_cast<std::size_t>(num_nodes), static_cast<std::size_t>(num_LM),
        static_cast<std::size_t>(num_channels)});
    checked_add(precision_elements, source_elements);
    checked_add(precision_elements, source_elements);
    const std::size_t force_elements = checked_product({
        static_cast<std::size_t>(num_edges), std::size_t(3)});
    std::size_t double_elements = force_elements;
    checked_add(double_elements, force_elements);
    checked_add(double_elements, force_elements);
    checked_add(double_elements, force_elements);
    const std::size_t precision_bytes = checked_product({
        precision_elements, sizeof(Precision)});
    const std::size_t double_bytes = checked_product({
        double_elements, sizeof(double)});
    std::size_t required_bytes = precision_bytes;
    checked_add(required_bytes, double_bytes);
    if (required_bytes > factorized_observer_max_bytes)
        throw std::length_error(
            "Execution R1 observer requires " + std::to_string(required_bytes)
            + " host bytes, exceeding the configured limit of "
            + std::to_string(factorized_observer_max_bytes) + ".");

    factorized_observer_num_nodes = num_nodes;
    factorized_observer_num_edges = num_edges;
    factorized_observer_bytes = required_bytes;
    factorized_observer_ready = false;
    factorized_observer_state.assign(l_max+1, {});
    factorized_observer_state_adjoint.assign(l_max+1, {});
    for (int l=0; l<=l_max; ++l) {
        const std::size_t components = 2*l+1;
        const std::size_t num_eta =
            execution_group_path_offsets_host[l+1]-execution_group_path_offsets_host[l];
        const std::size_t size = checked_product({
            static_cast<std::size_t>(num_nodes), components, num_eta,
            static_cast<std::size_t>(factorized_embedding_width),
            static_cast<std::size_t>(num_channels)});
        factorized_observer_state[l].assign(size, Precision(0));
        factorized_observer_state_adjoint[l].assign(size, Precision(0));
    }
    factorized_observer_output.assign(
        checked_product({static_cast<std::size_t>(num_nodes),
                         static_cast<std::size_t>(num_lm),
                         static_cast<std::size_t>(num_channels)}),
        Precision(0));
    factorized_observer_output_adjoint.assign(
        factorized_observer_output.size(), Precision(0));
    standard_r0_observer_output.assign(
        factorized_observer_output.size(), Precision(0));
    standard_r0_observer_output_adjoint.assign(
        factorized_observer_output.size(), Precision(0));
    factorized_observer_source_h1_delta.assign(
        checked_product({static_cast<std::size_t>(num_nodes),
                         static_cast<std::size_t>(num_LM),
                         static_cast<std::size_t>(num_channels)}),
        Precision(0));
    factorized_observer_source_h1_before.assign(
        factorized_observer_source_h1_delta.size(), Precision(0));
    factorized_observer_directed_force_delta.assign(
        checked_product({static_cast<std::size_t>(num_edges), std::size_t(3)}),
        0.0);
    factorized_observer_directed_force_before.assign(
        factorized_observer_directed_force_delta.size(), 0.0);
    standard_r0_observer_directed_force_delta.assign(
        factorized_observer_directed_force_delta.size(), 0.0);
    standard_r0_observer_directed_force_before.assign(
        factorized_observer_directed_force_delta.size(), 0.0);
}

template <typename Precision>
void MACEKokkos<Precision>::capture_factorized_state_tile(
    const int l,
    const int receiver_begin,
    const int receiver_count,
    ExecutionScratchTensor packed_state)
{
    const int num_eta =
        execution_group_path_offsets_host[l+1]-execution_group_path_offsets_host[l];
    capture_packed_execution_tile(
        factorized_observer_state[l], receiver_begin, receiver_count,
        2*l+1, num_eta,
        factorized_embedding_width, num_channels, packed_state);
}

template <typename Precision>
void MACEKokkos<Precision>::capture_factorized_state_tile(
    const int l,
    const int receiver_begin,
    const int receiver_count,
    ExecutionScratchMatrix state)
{
    const int num_eta =
        execution_group_path_offsets_host[l+1]-execution_group_path_offsets_host[l];
    capture_matrix_execution_tile(
        factorized_observer_state[l], receiver_begin, receiver_count,
        2*l+1, num_eta,
        factorized_embedding_width, num_channels, state);
}

template <typename Precision>
void MACEKokkos<Precision>::capture_factorized_state_adjoint_tile(
    const int l,
    const int receiver_begin,
    const int receiver_count,
    ExecutionScratchTensor packed_state_adjoint)
{
    const int num_eta =
        execution_group_path_offsets_host[l+1]-execution_group_path_offsets_host[l];
    capture_packed_execution_tile(
        factorized_observer_state_adjoint[l], receiver_begin, receiver_count,
        2*l+1, num_eta,
        factorized_embedding_width, num_channels, packed_state_adjoint);
}

template <typename Precision>
void MACEKokkos<Precision>::capture_factorized_state_adjoint_tile(
    const int l,
    const int receiver_begin,
    const int receiver_count,
    ExecutionScratchMatrix state_adjoint)
{
    const int num_eta =
        execution_group_path_offsets_host[l+1]-execution_group_path_offsets_host[l];
    capture_matrix_execution_tile(
        factorized_observer_state_adjoint[l], receiver_begin, receiver_count,
        2*l+1, num_eta,
        factorized_embedding_width, num_channels, state_adjoint);
}

template <typename Precision>
void MACEKokkos<Precision>::capture_factorized_output(const int num_nodes)
{
    if (!factorized_observer_enabled)
        return;
    const auto host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), A1);
    for (int node=0; node<num_nodes; ++node)
        for (int lm=0; lm<num_lm; ++lm)
            for (int channel=0; channel<num_channels; ++channel) {
                const std::size_t index =
                    (static_cast<std::size_t>(node)*num_lm+lm)*num_channels
                    +channel;
                factorized_observer_output[index] = host(node,lm,channel);
            }
}

template <typename Precision>
void MACEKokkos<Precision>::capture_standard_r0_output(const int num_nodes)
{
    if (!factorized_observer_enabled)
        return;
    const auto host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), A0);
    for (int node=0; node<num_nodes; ++node)
        for (int lm=0; lm<num_lm; ++lm)
            for (int channel=0; channel<num_channels; ++channel) {
                const std::size_t index =
                    (static_cast<std::size_t>(node)*num_lm+lm)*num_channels
                    +channel;
                standard_r0_observer_output[index] = host(node,lm,channel);
            }
}

template <typename Precision>
void MACEKokkos<Precision>::begin_factorized_reverse_observation(
    const int num_nodes,
    const int num_edges)
{
    if (!factorized_observer_enabled)
        return;
    if (factorized_observer_num_nodes != num_nodes
        || factorized_observer_num_edges != num_edges)
        throw std::logic_error(
            "Execution R1 observer reverse requires a matching observed forward pass.");
    factorized_observer_ready = false;
    const auto output_adjoint_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), A1_adj);
    for (int node=0; node<num_nodes; ++node)
        for (int lm=0; lm<num_lm; ++lm)
            for (int channel=0; channel<num_channels; ++channel) {
                const std::size_t index =
                    (static_cast<std::size_t>(node)*num_lm+lm)*num_channels
                    +channel;
                factorized_observer_output_adjoint[index] =
                    output_adjoint_host(node,lm,channel);
            }
    const auto h1_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), H1_adj);
    for (int node=0; node<num_nodes; ++node)
        for (int lm=0; lm<num_LM; ++lm)
            for (int channel=0; channel<num_channels; ++channel) {
                const std::size_t index =
                    (static_cast<std::size_t>(node)*num_LM+lm)*num_channels
                    +channel;
                factorized_observer_source_h1_before[index] =
                    h1_host(node,lm,channel);
            }
    const auto force_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), node_forces);
    for (std::size_t index=0;
         index<factorized_observer_directed_force_before.size(); ++index)
        factorized_observer_directed_force_before[index] = force_host(index);
}

template <typename Precision>
void MACEKokkos<Precision>::finish_factorized_reverse_observation()
{
    if (!factorized_observer_enabled)
        return;
    const auto h1_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), H1_adj);
    for (int node=0; node<factorized_observer_num_nodes; ++node)
        for (int lm=0; lm<num_LM; ++lm)
            for (int channel=0; channel<num_channels; ++channel) {
                const std::size_t index =
                    (static_cast<std::size_t>(node)*num_LM+lm)*num_channels
                    +channel;
                factorized_observer_source_h1_delta[index] =
                    h1_host(node,lm,channel)
                    -factorized_observer_source_h1_before[index];
            }
    const auto force_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), node_forces);
    for (std::size_t index=0;
         index<factorized_observer_directed_force_delta.size(); ++index)
        factorized_observer_directed_force_delta[index] = force_host(index)
            -factorized_observer_directed_force_before[index];
}

template <typename Precision>
void MACEKokkos<Precision>::begin_standard_r0_reverse_observation(
    const int num_nodes,
    const int num_edges)
{
    if (!factorized_observer_enabled)
        return;
    if (factorized_observer_num_nodes != num_nodes
        || factorized_observer_num_edges != num_edges)
        throw std::logic_error(
            "Execution R0 observer reverse requires a matching observed forward pass.");
    const auto adjoint_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), A0_adj);
    for (int node=0; node<num_nodes; ++node)
        for (int lm=0; lm<num_lm; ++lm)
            for (int channel=0; channel<num_channels; ++channel) {
                const std::size_t index =
                    (static_cast<std::size_t>(node)*num_lm+lm)*num_channels
                    +channel;
                standard_r0_observer_output_adjoint[index] =
                    adjoint_host(node,lm,channel);
            }
    const auto force_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), node_forces);
    for (std::size_t index=0;
         index<standard_r0_observer_directed_force_before.size(); ++index)
        standard_r0_observer_directed_force_before[index] = force_host(index);
}

template <typename Precision>
void MACEKokkos<Precision>::finish_standard_r0_reverse_observation()
{
    if (!factorized_observer_enabled)
        return;
    const auto force_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), node_forces);
    for (std::size_t index=0;
         index<standard_r0_observer_directed_force_delta.size(); ++index)
        standard_r0_observer_directed_force_delta[index] = force_host(index)
            -standard_r0_observer_directed_force_before[index];
    factorized_observer_ready = true;
}


template class MACEKokkos<float>;
template class MACEKokkos<double>;
