#include "mace_standard_r0.hpp"

#include <Kokkos_Core.hpp>

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr std::size_t prefix_edges = 2;
constexpr std::size_t active_edges = 3;
constexpr std::size_t suffix_edges = 3;
constexpr std::size_t total_edges = prefix_edges+active_edges+suffix_edges;
constexpr std::size_t workspace_edge_begin = 1;
constexpr double sentinel = 73.0;

void reset_workspace(
    std::array<double,total_edges>& harmonics,
    std::array<double,3*total_edges>& gradients,
    std::array<double,3*total_edges>& forces)
{
    harmonics.fill(sentinel);
    gradients.fill(sentinel);
    forces.fill(sentinel);
    for (std::size_t edge=0; edge<active_edges; ++edge) {
        harmonics[prefix_edges+edge] = 1.0;
        for (std::size_t xyz=0; xyz<3; ++xyz) {
            gradients[3*(prefix_edges+edge)+xyz] = 0.0;
            forces[3*(prefix_edges+edge)+xyz] = 0.0;
        }
    }
}

void check_workspace(const std::array<double,3*total_edges>& forces)
{
    for (std::size_t edge=0; edge<prefix_edges; ++edge)
        for (std::size_t xyz=0; xyz<3; ++xyz)
            assert(forces[3*edge+xyz] == sentinel);
    for (std::size_t edge=0; edge<active_edges; ++edge) {
        assert(std::abs(forces[3*(prefix_edges+edge)]+1.0) < 1e-12);
        assert(forces[3*(prefix_edges+edge)+1] == 0.0);
        assert(forces[3*(prefix_edges+edge)+2] == 0.0);
    }
    for (std::size_t edge=prefix_edges+active_edges;
         edge<total_edges; ++edge)
        for (std::size_t xyz=0; xyz<3; ++xyz)
            assert(forces[3*edge+xyz] == sentinel);
}

} // namespace

int main(int argc, char** argv)
{
    Kokkos::ScopeGuard guard(argc, argv);
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;

    std::array<std::int32_t,3> node_types{0,0,0};
    std::array<std::int32_t,3> num_neigh{3,3,2};
    std::array<std::int32_t,3> first_neigh{0,3,6};
    std::array<std::int32_t,total_edges> neigh_types{};
    std::array<std::int32_t,total_edges> edge_receivers{0,0,0,1,1,1,2,2};
    std::array<std::int32_t,1> type_to_active{0};
    std::array<double,3*total_edges> coordinates{};
    std::array<double,total_edges> radius{};
    for (std::size_t edge=0; edge<total_edges; ++edge) {
        coordinates[3*edge] = edge >= prefix_edges
                && edge < prefix_edges+active_edges
            ? 1.0 : 0.0;
        coordinates[3*edge+1] = edge >= prefix_edges
                && edge < prefix_edges+active_edges
            ? 0.0 : 1.0;
        radius[edge] = 0.5;
    }
    std::array<double,4> radial_coefficients{0.0,1.0,0.0,0.0};
    std::array<double,3> output_adjoint{1.0,1.0,1.0};
    std::array<double,3> density_state{};
    std::array<double,total_edges> harmonics{};
    std::array<double,3*total_edges> gradients{};
    std::array<double,3*total_edges> forces{};

    symmetrix::execution::R0LogicalArguments<double> args{
        .num_nodes = 3,
        .num_edges = active_edges,
        .edge_begin = prefix_edges,
        .workspace_edge_begin = workspace_edge_begin,
        .active_type_count = 1,
        .type_to_active_count = 1,
        .channels = 1,
        .l_max = 0,
        .harmonic_count = 1,
        .coordinates_are_unit = true,
        .node_types = node_types.data(),
        .num_neigh = num_neigh.data(),
        .first_neigh = first_neigh.data(),
        .neigh_types = neigh_types.data(),
        .edge_receivers = edge_receivers.data(),
        .type_to_active = type_to_active.data(),
        .coordinates = coordinates.data(),
        .radius = radius.data(),
        .radial = {1,1,1,1.0,0.0,radial_coefficients.data()},
        .harmonics = harmonics.data()+workspace_edge_begin,
        .harmonic_gradients = gradients.data()+3*workspace_edge_begin,
        .output_adjoint = output_adjoint.data(),
        .density_state = density_state.data(),
        .directed_forces = forces.data()+3*workspace_edge_begin,
        .cutoff = 1.0,
    };

    reset_workspace(harmonics, gradients, forces);
    symmetrix::standard_r0::launch_coordinate_reverse_edge16<false>(
        ExecutionSpace{}, 1, args);
    Kokkos::fence();
    check_workspace(forces);

    reset_workspace(harmonics, gradients, forces);
    symmetrix::standard_r0::launch_coordinate_reverse<false>(
        ExecutionSpace{}, args);
    Kokkos::fence();
    check_workspace(forces);
}
