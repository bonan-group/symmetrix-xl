#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace symmetrix::dispersion {
namespace d2_detail {

template<class View>
void grow_view(View& view, const std::size_t extent, const char* label)
{
    if (view.extent(0) < extent)
        view = View(Kokkos::view_alloc(std::string(label)),extent);
}

} // namespace d2_detail

template<class Precision>
D2Kokkos<Precision>::D2Kokkos(
    std::string functional, const double pair_cutoff)
    : functional_(std::move(functional)), pair_cutoff_(pair_cutoff)
{
    if (!std::isfinite(pair_cutoff_) || !(pair_cutoff_ > 0.0))
        throw std::invalid_argument(
            "D2 pair cutoff must be finite and positive.");
    parameters_ = lookup_d2_parameters(functional_).parameters;
    const auto host_r0 = d2_r0_angstrom();
    const auto host_c6 = d2_c6_ev_angstrom6();
    r0_ = ParameterView("D2::r0",host_r0.size());
    c6_ = ParameterView("D2::c6",host_c6.size());
    auto r0_mirror = Kokkos::create_mirror_view(r0_);
    auto c6_mirror = Kokkos::create_mirror_view(c6_);
    for (std::size_t index=0; index<host_r0.size(); ++index)
        r0_mirror(index) = static_cast<Precision>(host_r0[index]);
    for (std::size_t index=0; index<host_c6.size(); ++index)
        c6_mirror(index) = static_cast<Precision>(host_c6[index]);
    Kokkos::deep_copy(r0_,r0_mirror);
    Kokkos::deep_copy(c6_,c6_mirror);
    result_.stress = decltype(result_.stress)("D2::stress");
}

template<class Precision>
void D2Kokkos<Precision>::validate_build_cutoff(
    const double build_cutoff) const
{
    if (!std::isfinite(build_cutoff) || build_cutoff < pair_cutoff_)
        throw std::invalid_argument(
            "D2 graph build cutoff must cover the pair cutoff.");
}

template<class Precision>
void D2Kokkos<Precision>::validate_atomic_numbers(
    const std::span<const int> atomic_numbers) const
{
    for (const int atomic_number : atomic_numbers)
        if (atomic_number < 1 || atomic_number > d2_max_atomic_number)
            throw std::invalid_argument(
                "D2 atomic number is outside the parameter table.");
}

template<class Precision>
void D2Kokkos<Precision>::prepare_periodic_graph(
    const std::span<const int> atomic_numbers,
    const std::span<const double> positions,
    const std::span<const double> cell,
    const std::span<const double> inverse_cell,
    const double build_cutoff)
{
    validate_build_cutoff(build_cutoff);
    validate_atomic_numbers(atomic_numbers);
    graph_.prepare_periodic(
        atomic_numbers,positions,cell,inverse_cell,build_cutoff);
    computed_ = false;
}

template<class Precision>
void D2Kokkos<Precision>::prepare_host_graph(
    const std::span<const int> atomic_numbers,
    const std::span<const int> receiver_offsets,
    const std::span<const int> sources,
    const std::span<const int> shifts,
    const std::span<const double> reference_positions,
    const std::span<const double> cell,
    const double build_cutoff)
{
    validate_build_cutoff(build_cutoff);
    validate_atomic_numbers(atomic_numbers);
    graph_.prepare_host(
        atomic_numbers,receiver_offsets,sources,shifts,reference_positions,
        cell,build_cutoff);
    computed_ = false;
}

template<class Precision>
void D2Kokkos<Precision>::update_geometry(
    const std::span<const double> positions,
    const std::span<const double> cell,
    const std::span<const double> inverse_cell)
{
    graph_.update_geometry(positions,cell,inverse_cell);
    computed_ = false;
}

template<class Precision>
void D2Kokkos<Precision>::ensure_result_capacity(const std::size_t nodes)
{
    d2_detail::grow_view(
        atomic_energies_storage_,nodes,"D2::atomic_energies");
    d2_detail::grow_view(forces_storage_,nodes,"D2::forces");
    d2_detail::grow_view(virial_storage_,nodes,"D2::virial");
    result_.atomic_energies = Kokkos::subview(
        atomic_energies_storage_,std::make_pair(std::size_t(0),nodes));
    result_.forces = Kokkos::subview(
        forces_storage_,std::make_pair(std::size_t(0),nodes),Kokkos::ALL());
}

template<class Precision>
void D2Kokkos<Precision>::compute(
    const bool compute_forces, const bool compute_stress)
{
    computed_ = false;
    forces_available_ = false;
    stress_available_ = false;
    if (!graph_.prepared())
        throw std::logic_error("D2 graph must be prepared before compute.");
    double volume = 0.0;
    if (compute_stress) {
        volume = graph_.cell_volume();
        if (!std::isfinite(volume) || !(volume > 0.0))
            throw std::invalid_argument(
                "D2 stress requires a finite positive cell volume.");
    }
    const std::size_t nodes = static_cast<std::size_t>(graph_.num_nodes());
    ensure_result_capacity(nodes);

    const auto numbers = graph_.atomic_numbers();
    const auto offsets = graph_.receiver_offsets();
    const auto sources = graph_.sources();
    const auto vectors = graph_.edge_vectors();
    const auto distances = graph_.distances();
    const auto atomic_energies = result_.atomic_energies;
    const auto forces = result_.forces;
    const auto virial = virial_storage_;
    const auto r0 = r0_;
    const auto c6 = c6_;
    const Precision cutoff = static_cast<Precision>(pair_cutoff_);
    const Precision s6 = static_cast<Precision>(parameters_.s6);
    const Precision rs6 = static_cast<Precision>(parameters_.rs6);
    const Precision alpha = static_cast<Precision>(parameters_.alp);
    constexpr std::size_t table_extent =
        static_cast<std::size_t>(d2_max_atomic_number)+1;

    Kokkos::parallel_for(
        "D2::receiver_kernel",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,nodes),
        KOKKOS_LAMBDA (const std::size_t receiver) {
            double node_energy = 0.0;
            double node_force[3] = {0.0,0.0,0.0};
            double node_virial[6] = {0.0,0.0,0.0,0.0,0.0,0.0};
            const int zi = numbers(receiver);
            for (int edge=offsets(receiver); edge<offsets(receiver+1); ++edge) {
                const Precision distance =
                    static_cast<Precision>(distances(edge));
                if (!(distance > Precision(0)) || distance >= cutoff)
                    continue;
                const int zj = numbers(sources(edge));
                const std::size_t pair = static_cast<std::size_t>(zi)
                    *table_extent+static_cast<std::size_t>(zj);
                const Precision radius = rs6*r0(pair);
                const Precision damping = Precision(1)/(
                    Precision(1)+Kokkos::exp(
                        -alpha*(distance/radius-Precision(1))));
                const Precision distance2 = distance*distance;
                const Precision inverse_distance6 = Precision(1)/(
                    distance2*distance2*distance2);
                const Precision pair_prefactor = -s6*c6(pair);
                const Precision pair_energy =
                    pair_prefactor*damping*inverse_distance6;
                node_energy += 0.5*static_cast<double>(pair_energy);
                if (compute_forces || compute_stress) {
                    const Precision damping_derivative = damping
                        *(Precision(1)-damping)*alpha/radius;
                    const Precision radial_derivative = pair_prefactor
                        *(damping_derivative*inverse_distance6
                            -Precision(6)*damping*inverse_distance6/distance);
                    const double force_scale =
                        static_cast<double>(radial_derivative/distance);
                    double edge_force[3];
                    for (int axis=0; axis<3; ++axis) {
                        edge_force[axis] = force_scale*vectors(edge,axis);
                        if (compute_forces)
                            node_force[axis] += edge_force[axis];
                    }
                    if (compute_stress) {
                        node_virial[0] += 0.5*vectors(edge,0)*edge_force[0];
                        node_virial[1] += 0.5*vectors(edge,1)*edge_force[1];
                        node_virial[2] += 0.5*vectors(edge,2)*edge_force[2];
                        node_virial[3] += 0.25*(
                            vectors(edge,1)*edge_force[2]
                            +vectors(edge,2)*edge_force[1]);
                        node_virial[4] += 0.25*(
                            vectors(edge,0)*edge_force[2]
                            +vectors(edge,2)*edge_force[0]);
                        node_virial[5] += 0.25*(
                            vectors(edge,0)*edge_force[1]
                            +vectors(edge,1)*edge_force[0]);
                    }
                }
            }
            atomic_energies(receiver) = node_energy;
            if (compute_forces)
                for (int axis=0; axis<3; ++axis)
                    forces(receiver,axis) = node_force[axis];
            if (compute_stress)
                for (int component=0; component<6; ++component)
                    virial(receiver,component) = node_virial[component];
        });

    double energy = 0.0;
    Kokkos::parallel_reduce(
        "D2::total_energy",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,nodes),
        KOKKOS_LAMBDA (const std::size_t atom, double& sum) {
            sum += atomic_energies(atom);
        },energy);
    result_.energy = energy;

    if (compute_stress) {
        for (int component=0; component<6; ++component) {
            double component_virial = 0.0;
            Kokkos::parallel_reduce(
                "D2::stress_component",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,nodes),
                KOKKOS_LAMBDA (const std::size_t atom, double& sum) {
                    sum += virial(atom,component);
                },component_virial);
            Kokkos::deep_copy(
                Kokkos::subview(result_.stress,component),
                component_virial/volume);
        }
    }
    Kokkos::DefaultExecutionSpace().fence("Symmetrix D2 compute");
    computed_ = true;
    forces_available_ = compute_forces;
    stress_available_ = compute_stress;
}

template<class Precision>
const DispersionResultKokkos& D2Kokkos<Precision>::result() const
{
    if (!computed_)
        throw std::logic_error("D2 result is unavailable before compute.");
    return result_;
}

} // namespace symmetrix::dispersion
