#ifndef SYMMETRIX_NEIGHBOR_GRAPH_KOKKOS_HPP
#define SYMMETRIX_NEIGHBOR_GRAPH_KOKKOS_HPP

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <functional>
#include <span>

namespace symmetrix::execution {

struct KokkosNeighborGraph {
    int num_nodes = 0;
    std::size_t num_edges = 0;
    Kokkos::View<int*> num_neigh;
    Kokkos::View<int*> receiver_offsets;
    Kokkos::View<int*> sources;
    Kokkos::View<int*[3],Kokkos::LayoutRight> shifts;
    Kokkos::View<double*[3]> fractional_positions;
    Kokkos::View<double*[3],Kokkos::LayoutRight> fractional_xyz;
    Kokkos::View<double*[3]> wrapped_positions;
};

using KokkosNeighborGraphOutputFactory =
    std::function<void(KokkosNeighborGraph&)>;

KokkosNeighborGraph build_periodic_neighbor_graph_kokkos(
    std::span<const double> positions,
    std::span<const double> cell,
    std::span<const double> inverse_cell,
    double cutoff,
    bool retain_shifts = true,
    // Invoked after exact counting so production can fill grow-only storage.
    const KokkosNeighborGraphOutputFactory& output_factory = {});

} // namespace symmetrix::execution

#endif
