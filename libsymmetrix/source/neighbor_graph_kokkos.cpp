#include "neighbor_graph_kokkos.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>

namespace symmetrix::execution {
namespace {

KOKKOS_INLINE_FUNCTION
int floor_div(const int numerator, const int denominator)
{
    const int quotient = numerator/denominator;
    const int remainder = numerator%denominator;
    return quotient-((remainder != 0 && remainder < 0) ? 1 : 0);
}

KOKKOS_INLINE_FUNCTION
int flatten_bin(
    const int x, const int y, const int z,
    const int nx, const int ny)
{
    return (z*ny+y)*nx+x;
}

} // namespace

KokkosNeighborGraph build_periodic_neighbor_graph_kokkos(
    const std::span<const double> positions,
    const std::span<const double> cell,
    const std::span<const double> inverse_cell,
    const double cutoff,
    const bool retain_shifts,
    const KokkosNeighborGraphOutputFactory& output_factory)
{
    if (positions.size()%3 != 0 || cell.size() != 9
        || inverse_cell.size() != 9)
        throw std::invalid_argument(
            "Kokkos neighbor graph input extents are inconsistent.");
    if (!std::isfinite(cutoff) || !(cutoff > 0.0))
        throw std::invalid_argument(
            "Kokkos neighbor graph cutoff must be finite and positive.");
    if (positions.size()/3
            >= static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::length_error(
            "Kokkos neighbor graph exceeds 32-bit node indexing.");
    for (const double value : positions)
        if (!std::isfinite(value))
            throw std::invalid_argument(
                "Kokkos neighbor graph positions must be finite.");
    for (const double value : cell)
        if (!std::isfinite(value))
            throw std::invalid_argument(
                "Kokkos neighbor graph cell must be finite.");
    for (const double value : inverse_cell)
        if (!std::isfinite(value))
            throw std::invalid_argument(
                "Kokkos neighbor graph inverse cell must be finite.");
    for (int row=0; row<3; ++row)
        for (int column=0; column<3; ++column) {
            double product = 0.0;
            for (int inner=0; inner<3; ++inner)
                product += cell[3*row+inner]*inverse_cell[3*inner+column];
            const double expected = row == column ? 1.0 : 0.0;
            if (std::abs(product-expected) > 1e-9)
                throw std::invalid_argument(
                    "Kokkos neighbor graph cell and inverse cell are inconsistent.");
        }

    const int num_nodes = static_cast<int>(positions.size()/3);
    auto host_positions = Kokkos::View<double*[3],Kokkos::HostSpace,
        Kokkos::MemoryUnmanaged>(
            const_cast<double*>(positions.data()), num_nodes);
    auto host_cell = Kokkos::View<double*[3],Kokkos::HostSpace,
        Kokkos::MemoryUnmanaged>(const_cast<double*>(cell.data()), 3);
    auto host_inverse_cell = Kokkos::View<double*[3],Kokkos::HostSpace,
        Kokkos::MemoryUnmanaged>(const_cast<double*>(inverse_cell.data()), 3);
    auto device_positions = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultExecutionSpace::memory_space(), host_positions);
    auto device_cell = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultExecutionSpace::memory_space(), host_cell);
    auto device_inverse_cell = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultExecutionSpace::memory_space(), host_inverse_cell);

    auto graph = KokkosNeighborGraph{};
    graph.num_nodes = num_nodes;
    graph.num_neigh = Kokkos::View<int*>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
            "SymmetrixNeighborGraph::num_neigh"), num_nodes);
    graph.receiver_offsets = Kokkos::View<int*>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
            "SymmetrixNeighborGraph::receiver_offsets"), num_nodes+1);
    graph.wrapped_positions = Kokkos::View<double*[3]>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
            "SymmetrixNeighborGraph::wrapped_positions"), num_nodes);
    graph.fractional_positions = Kokkos::View<double*[3]>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
            "SymmetrixNeighborGraph::reference_fractional_positions"), num_nodes);
    auto wrapped_fractional_positions = Kokkos::View<double*[3]>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
            "SymmetrixNeighborGraph::wrapped_fractional_positions"), num_nodes);
    auto wrap_shifts = Kokkos::View<int*[3]>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
            "SymmetrixNeighborGraph::wrap_shifts"), num_nodes);

    auto execution_space = Kokkos::DefaultExecutionSpace{};
    Kokkos::parallel_for(
        "SymmetrixNeighborGraph::wrap_positions",
        Kokkos::RangePolicy<decltype(execution_space)>(
            execution_space, 0, num_nodes),
        KOKKOS_LAMBDA (const int atom) {
            double fractional[3];
            for (int axis=0; axis<3; ++axis) {
                double value = 0.0;
                for (int component=0; component<3; ++component)
                    value += device_positions(atom,component)
                        *device_inverse_cell(component,axis);
                graph.fractional_positions(atom,axis) = value;
                const int wrap = static_cast<int>(floor(value));
                value -= wrap;
                fractional[axis] = value;
                wrapped_fractional_positions(atom,axis) = value;
                wrap_shifts(atom,axis) = wrap;
            }
            for (int component=0; component<3; ++component) {
                double value = 0.0;
                for (int axis=0; axis<3; ++axis)
                    value += fractional[axis]*device_cell(axis,component);
                graph.wrapped_positions(atom,component) = value;
            }
        });

    std::array<int,3> bin_counts{};
    std::array<int,3> stencil_reach{};
    for (int axis=0; axis<3; ++axis) {
        double inverse_norm_squared = 0.0;
        for (int component=0; component<3; ++component) {
            const double value = inverse_cell[3*component+axis];
            inverse_norm_squared += value*value;
        }
        if (!(inverse_norm_squared > 0.0))
            throw std::invalid_argument(
                "Kokkos neighbor graph cell must be invertible.");
        const double face_height = 1.0/std::sqrt(inverse_norm_squared);
        const double bins = std::floor(face_height/cutoff);
        if (bins >= static_cast<double>(std::numeric_limits<int>::max()))
            throw std::length_error(
                "Kokkos neighbor graph bin extent exceeds 32-bit indexing.");
        bin_counts[axis] = std::max(1, static_cast<int>(bins));
        const double reach = std::ceil(
            cutoff*std::sqrt(inverse_norm_squared)*bin_counts[axis])+1.0;
        if (reach >= static_cast<double>(std::numeric_limits<int>::max()/2))
            throw std::length_error(
                "Kokkos neighbor graph stencil exceeds 32-bit indexing.");
        stencil_reach[axis] = std::max(1, static_cast<int>(reach));
    }
    const std::size_t num_bins_size = static_cast<std::size_t>(bin_counts[0])
        *static_cast<std::size_t>(bin_counts[1])
        *static_cast<std::size_t>(bin_counts[2]);
    if (num_bins_size
            >= static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::length_error(
            "Kokkos neighbor graph bin count exceeds 32-bit indexing.");
    const int num_bins = static_cast<int>(num_bins_size);
    auto atom_bins = Kokkos::View<int*>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
            "SymmetrixNeighborGraph::atom_bins"), num_nodes);
    auto bin_sizes = Kokkos::View<int*>("SymmetrixNeighborGraph::bin_sizes", num_bins);
    const int bins_x = bin_counts[0];
    const int bins_y = bin_counts[1];
    const int bins_z = bin_counts[2];
    Kokkos::parallel_for(
        "SymmetrixNeighborGraph::count_bins",
        Kokkos::RangePolicy<decltype(execution_space)>(
            execution_space, 0, num_nodes),
        KOKKOS_LAMBDA (const int atom) {
            const int x = Kokkos::min(
                bins_x-1,
                static_cast<int>(wrapped_fractional_positions(atom,0)*bins_x));
            const int y = Kokkos::min(
                bins_y-1,
                static_cast<int>(wrapped_fractional_positions(atom,1)*bins_y));
            const int z = Kokkos::min(
                bins_z-1,
                static_cast<int>(wrapped_fractional_positions(atom,2)*bins_z));
            const int bin = flatten_bin(x,y,z,bins_x,bins_y);
            atom_bins(atom) = bin;
            Kokkos::atomic_fetch_add(&bin_sizes(bin), 1);
        });
    auto bin_offsets = Kokkos::View<int*>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
            "SymmetrixNeighborGraph::bin_offsets"), num_bins+1);
    Kokkos::parallel_scan(
        "SymmetrixNeighborGraph::scan_bins",
        Kokkos::RangePolicy<decltype(execution_space)>(
            execution_space, 0, num_bins+1),
        KOKKOS_LAMBDA (const int bin, int& update, const bool final) {
            if (final)
                bin_offsets(bin) = update;
            if (bin < num_bins)
                update += bin_sizes(bin);
        });
    auto bin_cursors = Kokkos::View<int*>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
            "SymmetrixNeighborGraph::bin_cursors"), num_bins);
    Kokkos::parallel_for(
        "SymmetrixNeighborGraph::initialize_bin_cursors",
        Kokkos::RangePolicy<decltype(execution_space)>(
            execution_space, 0, num_bins),
        KOKKOS_LAMBDA (const int bin) { bin_cursors(bin) = bin_offsets(bin); });
    auto bin_atoms = Kokkos::View<int*>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
            "SymmetrixNeighborGraph::bin_atoms"), num_nodes);
    Kokkos::parallel_for(
        "SymmetrixNeighborGraph::fill_bins",
        Kokkos::RangePolicy<decltype(execution_space)>(
            execution_space, 0, num_nodes),
        KOKKOS_LAMBDA (const int atom) {
            const int index = Kokkos::atomic_fetch_add(
                &bin_cursors(atom_bins(atom)), 1);
            bin_atoms(index) = atom;
        });
    Kokkos::parallel_for(
        "SymmetrixNeighborGraph::sort_bin_atoms",
        Kokkos::RangePolicy<decltype(execution_space)>(
            execution_space, 0, num_bins),
        KOKKOS_LAMBDA (const int bin) {
            const int begin = bin_offsets(bin);
            const int end = bin_offsets(bin+1);
            for (int index=begin+1; index<end; ++index) {
                const int atom = bin_atoms(index);
                int insertion = index;
                while (insertion > begin && bin_atoms(insertion-1) > atom) {
                    bin_atoms(insertion) = bin_atoms(insertion-1);
                    --insertion;
                }
                bin_atoms(insertion) = atom;
            }
        });

    const int reach_x = stencil_reach[0];
    const int reach_y = stencil_reach[1];
    const int reach_z = stencil_reach[2];
    const double cutoff_squared = cutoff*cutoff;
    auto edge_counts = Kokkos::View<long long*>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
            "SymmetrixNeighborGraph::edge_counts"), num_nodes);
    const auto count_or_fill = KOKKOS_LAMBDA (
            const int receiver, const bool fill,
            const Kokkos::View<int*> output_sources,
            const Kokkos::View<int*[3],Kokkos::LayoutRight> output_shifts,
            const Kokkos::View<double*[3],Kokkos::LayoutRight>
                output_fractional_xyz) {
        const int receiver_bin = atom_bins(receiver);
        const int receiver_x = receiver_bin%bins_x;
        const int receiver_y = (receiver_bin/bins_x)%bins_y;
        const int receiver_z = receiver_bin/(bins_x*bins_y);
        long long local_edge = 0;
        for (int dz=-reach_z; dz<=reach_z; ++dz) {
            const int raw_z = receiver_z+dz;
            const int shift_z = floor_div(raw_z,bins_z);
            const int source_z = raw_z-shift_z*bins_z;
            for (int dy=-reach_y; dy<=reach_y; ++dy) {
                const int raw_y = receiver_y+dy;
                const int shift_y = floor_div(raw_y,bins_y);
                const int source_y = raw_y-shift_y*bins_y;
                for (int dx=-reach_x; dx<=reach_x; ++dx) {
                    const int raw_x = receiver_x+dx;
                    const int shift_x = floor_div(raw_x,bins_x);
                    const int source_x = raw_x-shift_x*bins_x;
                    const int source_bin = flatten_bin(
                        source_x,source_y,source_z,bins_x,bins_y);
                    for (int index=bin_offsets(source_bin);
                         index<bin_offsets(source_bin+1); ++index) {
                        const int source = bin_atoms(index);
                        if (source == receiver && shift_x == 0
                            && shift_y == 0 && shift_z == 0)
                            continue;
                        double distance_squared = 0.0;
                        const int image[3] = {shift_x,shift_y,shift_z};
                        for (int component=0; component<3; ++component) {
                            double displacement = graph.wrapped_positions(
                                source,component)-graph.wrapped_positions(
                                    receiver,component);
                            for (int axis=0; axis<3; ++axis)
                                displacement += image[axis]
                                    *device_cell(axis,component);
                            distance_squared += displacement*displacement;
                        }
                        if (distance_squared > cutoff_squared)
                            continue;
                        if (fill) {
                            const int edge = graph.receiver_offsets(receiver)
                                +static_cast<int>(local_edge);
                            output_sources(edge) = source;
                            if (output_shifts.extent(0) != 0) {
                                output_shifts(edge,0) = shift_x
                                    -wrap_shifts(source,0)+wrap_shifts(receiver,0);
                                output_shifts(edge,1) = shift_y
                                    -wrap_shifts(source,1)+wrap_shifts(receiver,1);
                                output_shifts(edge,2) = shift_z
                                    -wrap_shifts(source,2)+wrap_shifts(receiver,2);
                            }
                            if (output_fractional_xyz.extent(0) != 0)
                                for (int axis=0; axis<3; ++axis)
                                    output_fractional_xyz(edge,axis) =
                                        wrapped_fractional_positions(source,axis)
                                        -wrapped_fractional_positions(receiver,axis)
                                        +image[axis];
                        }
                        ++local_edge;
                    }
                }
            }
        }
        if (!fill)
            edge_counts(receiver) = local_edge;
    };
    Kokkos::parallel_for(
        "SymmetrixNeighborGraph::count_edges",
        Kokkos::RangePolicy<decltype(execution_space)>(
            execution_space, 0, num_nodes),
        KOKKOS_LAMBDA (const int receiver) {
            count_or_fill(receiver, false, {}, {}, {});
        });
    auto edge_offsets = Kokkos::View<long long*>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
            "SymmetrixNeighborGraph::edge_offsets"), num_nodes+1);
    Kokkos::parallel_scan(
        "SymmetrixNeighborGraph::scan_receivers",
        Kokkos::RangePolicy<decltype(execution_space)>(
            execution_space, 0, num_nodes+1),
        KOKKOS_LAMBDA (const int receiver, long long& update, const bool final) {
            if (final)
                edge_offsets(receiver) = update;
            if (receiver < num_nodes)
                update += edge_counts(receiver);
        });
    auto host_edge_count = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), Kokkos::subview(edge_offsets,num_nodes));
    if (host_edge_count()
            > static_cast<long long>(std::numeric_limits<int>::max()))
        throw std::length_error(
            "Kokkos neighbor graph exceeds 32-bit edge indexing.");
    graph.num_edges = static_cast<std::size_t>(host_edge_count());
    Kokkos::parallel_for(
        "SymmetrixNeighborGraph::narrow_admitted_offsets",
        Kokkos::RangePolicy<decltype(execution_space)>(
            execution_space, 0, num_nodes+1),
        KOKKOS_LAMBDA (const int receiver) {
            graph.receiver_offsets(receiver) = static_cast<int>(
                edge_offsets(receiver));
            if (receiver < num_nodes)
                graph.num_neigh(receiver) = static_cast<int>(
                    edge_counts(receiver));
        });
    if (output_factory) {
        output_factory(graph);
        const bool has_fractional_xyz =
            graph.fractional_xyz.extent(0) == graph.num_edges;
        const bool has_shifts = graph.shifts.extent(0) == graph.num_edges;
        if (graph.sources.extent(0) != graph.num_edges
            || (!has_fractional_xyz && !has_shifts)
            || (retain_shifts && !has_shifts))
            throw std::invalid_argument(
                "Kokkos neighbor graph output factory returned invalid extents.");
    } else {
        graph.sources = Kokkos::View<int*>(
            Kokkos::view_alloc(Kokkos::WithoutInitializing,
                "SymmetrixNeighborGraph::sources"), graph.num_edges);
        if (retain_shifts)
            graph.shifts = Kokkos::View<int*[3],Kokkos::LayoutRight>(
                Kokkos::view_alloc(Kokkos::WithoutInitializing,
                    "SymmetrixNeighborGraph::shifts"), graph.num_edges);
        graph.fractional_xyz = Kokkos::View<
            double*[3],Kokkos::LayoutRight>(
            Kokkos::view_alloc(Kokkos::WithoutInitializing,
                "SymmetrixNeighborGraph::fractional_xyz"), graph.num_edges);
    }
    Kokkos::parallel_for(
        "SymmetrixNeighborGraph::fill_edges",
        Kokkos::RangePolicy<decltype(execution_space)>(
            execution_space, 0, num_nodes),
        KOKKOS_LAMBDA (const int receiver) {
            count_or_fill(
                receiver, true, graph.sources, graph.shifts,
                graph.fractional_xyz);
        });
    execution_space.fence("Symmetrix Kokkos neighbor graph construction");
    return graph;
}

} // namespace symmetrix::execution
