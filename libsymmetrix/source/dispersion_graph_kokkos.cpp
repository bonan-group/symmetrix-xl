#include "dispersion_graph_kokkos.hpp"

#include "dispersion_parameters.hpp"
#include "neighbor_graph_kokkos.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

namespace symmetrix::dispersion {
namespace {

void validate_cutoff(const double cutoff)
{
    if (!std::isfinite(cutoff) || !(cutoff > 0.0))
        throw std::invalid_argument(
            "Dispersion graph build cutoff must be finite and positive.");
}

void validate_finite(
    const std::span<const double> values, const char* description)
{
    for (const double value : values)
        if (!std::isfinite(value))
            throw std::invalid_argument(
                std::string("Dispersion graph ")+description+" must be finite.");
}

void validate_geometry_extents(
    const std::span<const double> positions,
    const std::span<const double> cell,
    const std::span<const double> inverse_cell,
    const std::size_t nodes)
{
    if (positions.size() != std::size_t(3)*nodes)
        throw std::invalid_argument(
            "Dispersion graph positions extent is inconsistent.");
    if (cell.size() != 9)
        throw std::invalid_argument(
            "Dispersion graph cell extent is inconsistent.");
    if (inverse_cell.size() != 9)
        throw std::invalid_argument(
            "Dispersion graph inverse cell extent is inconsistent.");
    validate_finite(positions, "positions");
    validate_finite(cell, "cell");
    validate_finite(inverse_cell, "inverse cell");
    for (int row=0; row<3; ++row)
        for (int column=0; column<3; ++column) {
            double product = 0.0;
            for (int inner=0; inner<3; ++inner)
                product += cell[std::size_t(3)*row+inner]
                    *inverse_cell[std::size_t(3)*inner+column];
            const double expected = row == column ? 1.0 : 0.0;
            if (std::abs(product-expected) > 1e-9)
                throw std::invalid_argument(
                    "Dispersion graph cell and inverse cell are inconsistent.");
        }
}

template<class View>
bool grow_view(View& view, const std::size_t extent, const char* label)
{
    if (view.extent(0) < extent) {
        view = View(
            Kokkos::view_alloc(
                Kokkos::WithoutInitializing,std::string(label)), extent);
        return true;
    }
    return false;
}

void launch_build_receivers(
    const DispersionGraphKokkos::IntView& offsets,
    const DispersionGraphKokkos::IntView& receivers,
    const int nodes)
{
    Kokkos::parallel_for(
        "DispersionGraph::build_receivers",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,nodes),
        KOKKOS_LAMBDA (const int receiver) {
            for (int edge=offsets(receiver); edge<offsets(receiver+1); ++edge)
                receivers(edge) = receiver;
        });
}

void launch_update_geometry(
    const DispersionGraphKokkos::Double3View& positions,
    const DispersionGraphKokkos::Double3View& cell,
    const DispersionGraphKokkos::IntView& receivers,
    const DispersionGraphKokkos::IntView& sources,
    const DispersionGraphKokkos::Int3View& shifts,
    const DispersionGraphKokkos::Double3View& edge_vectors,
    const DispersionGraphKokkos::DoubleView& distances,
    const std::size_t edges)
{
    Kokkos::parallel_for(
        "DispersionGraph::update_geometry",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,edges),
        KOKKOS_LAMBDA (const std::size_t edge) {
            const std::size_t coordinate_offset = std::size_t(3)*edge;
            double squared_distance = 0.0;
            for (std::size_t component=0; component<3; ++component) {
                double displacement = positions(sources(edge),component)
                    -positions(receivers(edge),component);
                for (std::size_t axis=0; axis<3; ++axis)
                    displacement += shifts.data()[coordinate_offset+axis]
                        *cell(axis,component);
                edge_vectors.data()[coordinate_offset+component] = displacement;
                squared_distance += displacement*displacement;
            }
            distances(edge) = Kokkos::sqrt(squared_distance);
        });
}

} // namespace

void DispersionGraphKokkos::validate_atomic_numbers(
    const std::span<const int> atomic_numbers) const
{
    if (atomic_numbers.size()
            >= static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::length_error(
            "Dispersion graph exceeds 32-bit node indexing.");
    for (const int atomic_number : atomic_numbers)
        if (atomic_number < 1 || atomic_number > d3_max_atomic_number)
            throw std::invalid_argument(
                "Dispersion graph atomic number is unsupported.");
}

void DispersionGraphKokkos::ensure_node_capacity(const std::size_t nodes)
{
    view_allocation_count_ += grow_view(
        atomic_numbers_,nodes,"DispersionGraph::atomic_numbers");
    view_allocation_count_ += grow_view(
        positions_,nodes,"DispersionGraph::positions");
    view_allocation_count_ += grow_view(
        host_positions_,nodes,"DispersionGraph::host_positions");
    view_allocation_count_ += grow_view(
        receiver_offsets_,nodes+1,"DispersionGraph::receiver_offsets");
}

void DispersionGraphKokkos::ensure_edge_capacity(const std::size_t edges)
{
    if (edges > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::length_error(
            "Dispersion graph exceeds 32-bit edge indexing.");
    view_allocation_count_ += grow_view(
        receivers_,edges,"DispersionGraph::receivers");
    view_allocation_count_ += grow_view(
        sources_,edges,"DispersionGraph::sources");
    view_allocation_count_ += grow_view(
        shifts_,edges,"DispersionGraph::shifts");
    view_allocation_count_ += grow_view(
        edge_vectors_,edges,"DispersionGraph::edge_vectors");
    view_allocation_count_ += grow_view(
        distances_,edges,"DispersionGraph::distances");
}

void DispersionGraphKokkos::copy_atomic_numbers(
    const std::span<const int> atomic_numbers)
{
    auto active_numbers = Kokkos::subview(
        atomic_numbers_,
        std::make_pair(std::size_t(0),atomic_numbers.size()));
    auto host = Kokkos::create_mirror_view(active_numbers);
    for (std::size_t index=0; index<atomic_numbers.size(); ++index)
        host(index) = atomic_numbers[index];
    Kokkos::deep_copy(active_numbers,host);
}

void DispersionGraphKokkos::prepare_periodic(
    const std::span<const int> atomic_numbers,
    const std::span<const double> positions,
    const std::span<const double> cell,
    const std::span<const double> inverse_cell,
    const double build_cutoff)
{
    validate_atomic_numbers(atomic_numbers);
    validate_cutoff(build_cutoff);
    validate_geometry_extents(
        positions,cell,inverse_cell,atomic_numbers.size());
    ensure_node_capacity(atomic_numbers.size());

    const auto graph = execution::build_periodic_neighbor_graph_kokkos(
        positions,cell,inverse_cell,build_cutoff,true,
        [this] (execution::KokkosNeighborGraph& output) {
            ensure_edge_capacity(output.num_edges);
            output.sources = Kokkos::subview(
                sources_,std::make_pair(std::size_t(0),output.num_edges));
            output.shifts = Kokkos::subview(
                shifts_,std::make_pair(std::size_t(0),output.num_edges),
                Kokkos::ALL());
        });

    num_nodes_ = graph.num_nodes;
    num_edges_ = graph.num_edges;
    auto output_offsets = Kokkos::subview(
        receiver_offsets_,std::make_pair(
            std::size_t(0),static_cast<std::size_t>(num_nodes_)+1));
    Kokkos::deep_copy(output_offsets,graph.receiver_offsets);
    copy_atomic_numbers(atomic_numbers);
    graph_backend_ = "kokkos";
    build_cutoff_ = build_cutoff;
    ++generation_;
    build_receivers();
    update_geometry_impl(positions,cell);
}

void DispersionGraphKokkos::prepare_host(
    const std::span<const int> atomic_numbers,
    const std::span<const int> receiver_offsets,
    const std::span<const int> sources,
    const std::span<const int> shifts,
    const std::span<const double> reference_positions,
    const std::span<const double> cell,
    const double build_cutoff)
{
    validate_atomic_numbers(atomic_numbers);
    validate_cutoff(build_cutoff);
    const std::size_t nodes = atomic_numbers.size();
    const std::size_t edges = sources.size();
    if (receiver_offsets.size() != nodes+1)
        throw std::invalid_argument(
            "Dispersion graph receiver offsets extent is inconsistent.");
    if (reference_positions.size() != std::size_t(3)*nodes)
        throw std::invalid_argument(
            "Dispersion graph reference positions extent is inconsistent.");
    if (cell.size() != 9)
        throw std::invalid_argument(
            "Dispersion graph cell extent is inconsistent.");
    if (shifts.size() != std::size_t(3)*edges)
        throw std::invalid_argument(
            "Dispersion graph shift extent is inconsistent.");
    if (receiver_offsets.front() != 0)
        throw std::invalid_argument(
            "Dispersion graph receiver offsets must start at zero.");
    for (std::size_t index=1; index<receiver_offsets.size(); ++index)
        if (receiver_offsets[index] < receiver_offsets[index-1])
            throw std::invalid_argument(
                "Dispersion graph receiver offsets must be nondecreasing.");
    if (receiver_offsets.back() < 0
            || static_cast<std::size_t>(receiver_offsets.back()) != edges)
        throw std::invalid_argument(
            "Dispersion graph terminal receiver offset must equal the edge count.");
    for (const int source : sources)
        if (source < 0 || static_cast<std::size_t>(source) >= nodes)
            throw std::invalid_argument(
                "Dispersion graph source index is out of range.");
    validate_finite(reference_positions,"reference positions");
    validate_finite(cell,"cell");
    ensure_node_capacity(nodes);
    ensure_edge_capacity(edges);

    auto active_offsets = Kokkos::subview(
        receiver_offsets_,
        std::make_pair(std::size_t(0),receiver_offsets.size()));
    auto host_offsets = Kokkos::create_mirror_view(active_offsets);
    for (std::size_t index=0; index<receiver_offsets.size(); ++index)
        host_offsets(index) = receiver_offsets[index];
    Kokkos::deep_copy(active_offsets,host_offsets);
    auto active_sources = Kokkos::subview(
        sources_,std::make_pair(std::size_t(0),edges));
    auto host_sources = Kokkos::create_mirror_view(active_sources);
    for (std::size_t edge=0; edge<edges; ++edge)
        host_sources(edge) = sources[edge];
    Kokkos::deep_copy(active_sources,host_sources);
    auto active_shifts = Kokkos::subview(
        shifts_,std::make_pair(std::size_t(0),edges),Kokkos::ALL());
    auto host_shifts = Kokkos::create_mirror_view(active_shifts);
    for (std::size_t edge=0; edge<edges; ++edge)
        for (std::size_t axis=0; axis<3; ++axis)
            host_shifts(edge,axis) = shifts[std::size_t(3)*edge+axis];
    Kokkos::deep_copy(active_shifts,host_shifts);
    copy_atomic_numbers(atomic_numbers);

    num_nodes_ = static_cast<int>(nodes);
    num_edges_ = edges;
    graph_backend_ = "host";
    build_cutoff_ = build_cutoff;
    ++generation_;
    build_receivers();
    update_geometry_impl(reference_positions,cell);
}

void DispersionGraphKokkos::build_receivers()
{
    launch_build_receivers(receiver_offsets_,receivers_,num_nodes_);
}

void DispersionGraphKokkos::update_geometry(
    const std::span<const double> positions,
    const std::span<const double> cell,
    const std::span<const double> inverse_cell)
{
    if (generation_ == 0)
        throw std::logic_error(
            "Dispersion graph must be prepared before updating geometry.");
    validate_geometry_extents(
        positions,cell,inverse_cell,static_cast<std::size_t>(num_nodes_));
    update_geometry_impl(positions,cell);
}

void DispersionGraphKokkos::update_geometry_impl(
    const std::span<const double> positions,
    const std::span<const double> cell)
{
    auto active_positions = Kokkos::subview(
        positions_,
        std::make_pair(
            std::size_t(0),static_cast<std::size_t>(num_nodes_)),
        Kokkos::ALL());
    for (int atom=0; atom<num_nodes_; ++atom)
        for (std::size_t axis=0; axis<3; ++axis)
            host_positions_(atom,axis) =
                positions[std::size_t(3)*static_cast<std::size_t>(atom)+axis];
    const auto active_host_positions = Kokkos::subview(
        host_positions_,
        std::make_pair(std::size_t(0),static_cast<std::size_t>(num_nodes_)),
        Kokkos::ALL());
    Kokkos::deep_copy(active_positions,active_host_positions);
    if (cell_.extent(0) == 0) {
        cell_ = Double3View(
            Kokkos::view_alloc(Kokkos::WithoutInitializing,
                "DispersionGraph::cell"),3);
        ++view_allocation_count_;
    }
    if (host_cell_.extent(0) == 0) {
        host_cell_ = HostDouble3View(
            Kokkos::view_alloc(Kokkos::WithoutInitializing,
                "DispersionGraph::host_cell"),3);
        ++view_allocation_count_;
    }
    for (std::size_t row=0; row<3; ++row)
        for (std::size_t column=0; column<3; ++column)
            host_cell_(row,column) = cell[std::size_t(3)*row+column];
    Kokkos::deep_copy(cell_,host_cell_);
    const double determinant =
        cell[0]*(cell[4]*cell[8]-cell[5]*cell[7])
        -cell[1]*(cell[3]*cell[8]-cell[5]*cell[6])
        +cell[2]*(cell[3]*cell[7]-cell[4]*cell[6]);
    cell_volume_ = std::abs(determinant);

    launch_update_geometry(
        positions_,cell_,receivers_,sources_,shifts_,edge_vectors_,distances_,
        num_edges_);
    Kokkos::DefaultExecutionSpace().fence(
        "Symmetrix dispersion graph geometry update");
    ++geometry_update_count_;
}

DispersionGraphDiagnostics DispersionGraphKokkos::diagnostics() const
{
    return {
        graph_backend_,generation_,geometry_update_count_,num_nodes_,num_edges_,
        atomic_numbers_.extent(0),receiver_offsets_.extent(0),sources_.extent(0),
        build_cutoff_,view_allocation_count_};
}

} // namespace symmetrix::dispersion
