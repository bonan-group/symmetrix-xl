#ifndef SYMMETRIX_DISPERSION_GRAPH_KOKKOS_HPP
#define SYMMETRIX_DISPERSION_GRAPH_KOKKOS_HPP

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace symmetrix::dispersion {

struct DispersionGraphDiagnostics {
    std::string graph_backend;
    std::uint64_t generation = 0;
    std::uint64_t geometry_update_count = 0;
    int num_nodes = 0;
    std::size_t num_edges = 0;
    std::size_t node_capacity = 0;
    std::size_t offset_capacity = 0;
    std::size_t edge_capacity = 0;
    double build_cutoff = 0.0;
    std::size_t view_allocation_count = 0;
};

class DispersionGraphKokkos {
public:
    using IntView = Kokkos::View<int*>;
    using Int3View = Kokkos::View<int*[3],Kokkos::LayoutRight>;
    using DoubleView = Kokkos::View<double*>;
    using Double3View = Kokkos::View<double*[3],Kokkos::LayoutRight>;

    void prepare_periodic(
        std::span<const int> atomic_numbers,
        std::span<const double> positions,
        std::span<const double> cell,
        std::span<const double> inverse_cell,
        double build_cutoff);
    void prepare_host(
        std::span<const int> atomic_numbers,
        std::span<const int> receiver_offsets,
        std::span<const int> sources,
        std::span<const int> shifts,
        std::span<const double> reference_positions,
        std::span<const double> cell,
        double build_cutoff);
    void update_geometry(
        std::span<const double> positions,
        std::span<const double> cell,
        std::span<const double> inverse_cell);

    [[nodiscard]] int num_nodes() const { return num_nodes_; }
    [[nodiscard]] std::size_t num_edges() const { return num_edges_; }
    [[nodiscard]] bool prepared() const { return generation_ != 0; }

    // Consumer views always expose active ranges, never retained capacity.
    [[nodiscard]] auto atomic_numbers() const
    {
        require_prepared();
        return Kokkos::subview(
            atomic_numbers_,std::make_pair(
                std::size_t(0),static_cast<std::size_t>(num_nodes_)));
    }
    [[nodiscard]] auto receiver_offsets() const
    {
        require_prepared();
        return Kokkos::subview(
            receiver_offsets_,std::make_pair(
                std::size_t(0),static_cast<std::size_t>(num_nodes_)+1));
    }
    [[nodiscard]] auto receivers() const
    {
        require_prepared();
        return Kokkos::subview(
            receivers_,std::make_pair(std::size_t(0),num_edges_));
    }
    [[nodiscard]] auto sources() const
    {
        require_prepared();
        return Kokkos::subview(
            sources_,std::make_pair(std::size_t(0),num_edges_));
    }
    [[nodiscard]] auto shifts() const
    {
        require_prepared();
        return Kokkos::subview(
            shifts_,std::make_pair(std::size_t(0),num_edges_),Kokkos::ALL());
    }
    [[nodiscard]] auto positions() const
    {
        require_prepared();
        return Kokkos::subview(
            positions_,std::make_pair(
                std::size_t(0),static_cast<std::size_t>(num_nodes_)),
            Kokkos::ALL());
    }
    [[nodiscard]] const Double3View& cell() const
    {
        require_prepared();
        return cell_;
    }
    [[nodiscard]] double cell_volume() const
    {
        require_prepared();
        return cell_volume_;
    }
    [[nodiscard]] auto edge_vectors() const
    {
        require_prepared();
        return Kokkos::subview(
            edge_vectors_,std::make_pair(std::size_t(0),num_edges_),
            Kokkos::ALL());
    }
    [[nodiscard]] auto distances() const
    {
        require_prepared();
        return Kokkos::subview(
            distances_,std::make_pair(std::size_t(0),num_edges_));
    }
    [[nodiscard]] DispersionGraphDiagnostics diagnostics() const;

private:
    void require_prepared() const
    {
        if (!prepared())
            throw std::logic_error(
                "Dispersion graph must be prepared before accessing views.");
    }
    void validate_atomic_numbers(std::span<const int> atomic_numbers) const;
    void ensure_node_capacity(std::size_t nodes);
    void ensure_edge_capacity(std::size_t edges);
    void copy_atomic_numbers(std::span<const int> atomic_numbers);
    void build_receivers();
    void update_geometry_impl(
        std::span<const double> positions, std::span<const double> cell);

    int num_nodes_ = 0;
    std::size_t num_edges_ = 0;
    std::uint64_t generation_ = 0;
    std::uint64_t geometry_update_count_ = 0;
    double build_cutoff_ = 0.0;
    std::string graph_backend_;
    IntView atomic_numbers_;
    IntView receiver_offsets_;
    IntView receivers_;
    IntView sources_;
    Int3View shifts_;
    Double3View positions_;
    Double3View cell_;
    using HostDouble3View =
        Kokkos::View<double*[3],Kokkos::LayoutRight,Kokkos::HostSpace>;
    HostDouble3View host_positions_;
    HostDouble3View host_cell_;
    Double3View edge_vectors_;
    DoubleView distances_;
    double cell_volume_ = 0.0;
    std::size_t view_allocation_count_ = 0;
};

} // namespace symmetrix::dispersion

#endif
