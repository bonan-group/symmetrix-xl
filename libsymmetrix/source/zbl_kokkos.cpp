#include <cmath>
#include <stdexcept>

#include "tools_kokkos.hpp"

#include "zbl_kokkos.hpp"

ZBLKokkos::ZBLKokkos()
{
}

ZBLKokkos::ZBLKokkos(
        double a_exp,
        double a_prefactor,
        std::vector<double> c,
        std::vector<double> covalent_radii,
        int p)
    : envelope_coefficient_0((p + 1.0) * (p + 2.0) / 2.0),
      envelope_coefficient_1(p * (p + 2.0)),
      envelope_coefficient_2(p * (p + 1.0) / 2.0),
      pair_dimension(covalent_radii.size()),
      p(p)
{
    if (c.size() != 4)
        throw std::invalid_argument("ZBL requires four screening coefficients");
    if (covalent_radii.empty())
        throw std::invalid_argument("ZBL requires at least one covalent radius");
    if (a_prefactor <= 0.0)
        throw std::invalid_argument("ZBL screening prefactor must be positive");
    if (p < 1)
        throw std::invalid_argument("ZBL envelope power must be positive");
    c_0 = c[0];
    c_1 = c[1];
    c_2 = c[2];
    c_3 = c[3];
    const std::size_t pair_count =
        covalent_radii.size() * covalent_radii.size();
    std::vector<double> inverse_a(pair_count);
    std::vector<double> prefactor(pair_count);
    std::vector<double> cutoff(pair_count);
    for (std::size_t Z_u=0; Z_u<covalent_radii.size(); ++Z_u) {
        for (std::size_t Z_v=0; Z_v<covalent_radii.size(); ++Z_v) {
            const std::size_t pair = Z_u*covalent_radii.size() + Z_v;
            const double denominator =
                std::pow(static_cast<double>(Z_u), a_exp)
                + std::pow(static_cast<double>(Z_v), a_exp);
            inverse_a[pair] = denominator / (a_prefactor * 0.529);
            prefactor[pair] =
                v_prefactor * static_cast<double>(Z_u * Z_v);
            cutoff[pair] = covalent_radii[Z_u] + covalent_radii[Z_v];
        }
    }
    pair_inverse_a = toKokkosView("zbl_pair_inverse_a", inverse_a);
    pair_prefactor = toKokkosView("zbl_pair_prefactor", prefactor);
    pair_cutoff = toKokkosView("zbl_pair_cutoff", cutoff);
    host_pair_inverse_a = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), pair_inverse_a);
    host_pair_prefactor = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), pair_prefactor);
    host_pair_cutoff = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), pair_cutoff);
}

KOKKOS_FUNCTION
double ZBLKokkos::compute(const int Z_u, const int Z_v, const double r) const
{
    return compute_value_gradient(Z_u, Z_v, r).value;
}

KOKKOS_FUNCTION
double ZBLKokkos::compute_gradient(const int Z_u, const int Z_v, const double r) const
{
    return compute_value_gradient(Z_u, Z_v, r).gradient;
}

KOKKOS_FUNCTION
ZBLKokkos::ValueGradient ZBLKokkos::compute_value_gradient(
    const int Z_u, const int Z_v, const double r) const
{
    const std::size_t pair = static_cast<std::size_t>(Z_u)*pair_dimension + Z_v;
    double inverse_a;
    double bare_prefactor;
    double r_max;
    KOKKOS_IF_ON_DEVICE((
        inverse_a = pair_inverse_a(pair);
        bare_prefactor = pair_prefactor(pair);
        r_max = pair_cutoff(pair);
    ))
    KOKKOS_IF_ON_HOST((
        inverse_a = host_pair_inverse_a(pair);
        bare_prefactor = host_pair_prefactor(pair);
        r_max = host_pair_cutoff(pair);
    ))
    if (r >= r_max)
        return {0.0, 0.0};

    const double r_over_a = r*inverse_a;
    const double e_0 = c_0*std::exp(c_exps_0*r_over_a);
    const double e_1 = c_1*std::exp(c_exps_1*r_over_a);
    const double e_2 = c_2*std::exp(c_exps_2*r_over_a);
    const double e_3 = c_3*std::exp(c_exps_3*r_over_a);
    const double phi = e_0 + e_1 + e_2 + e_3;
    const double d_phi__d_r = inverse_a * (
        c_exps_0*e_0 + c_exps_1*e_1 + c_exps_2*e_2 + c_exps_3*e_3);
    const double inverse_r = 1.0/r;
    const double bare_value = bare_prefactor*inverse_r*phi;
    const double bare_gradient = bare_prefactor*inverse_r*inverse_r
        * (r*d_phi__d_r - phi);

    const double x = r/r_max;
    const double x_p_minus_1 = std::pow(x, p - 1);
    const double x_p = x_p_minus_1*x;
    const double x_p_plus_1 = x_p*x;
    const double x_p_plus_2 = x_p_plus_1*x;
    const double envelope = 1.0 - envelope_coefficient_0*x_p
        + envelope_coefficient_1*x_p_plus_1
        - envelope_coefficient_2*x_p_plus_2;
    const double envelope_gradient = (
        - envelope_coefficient_0*p*x_p_minus_1
        + envelope_coefficient_1*(p + 1.0)*x_p
        - envelope_coefficient_2*(p + 2.0)*x_p_plus_1) / r_max;
    return {
        0.5*bare_value*envelope,
        0.5*(bare_value*envelope_gradient + bare_gradient*envelope)};
}

KOKKOS_FUNCTION
double ZBLKokkos::compute_envelope(const double r, const double r_max, const int p) const
{
    if (r >= r_max) {
        return 0.0;
    }
    double r_over_r_max = r / r_max;
    double v = (1.0 - ((p + 1.0) * (p + 2.0) / 2.0) * std::pow(r_over_r_max, p)
                    + p * (p + 2.0) * std::pow(r_over_r_max, p + 1)
                    - (p * (p + 1.0) / 2.0) * std::pow(r_over_r_max, p + 2));
    return v;
}

KOKKOS_FUNCTION
double ZBLKokkos::compute_envelope_gradient(const double r, const double r_max, const int p) const
{
    if (r >= r_max) {
        return 0.0;
    }
    double r_over_r_max = r / r_max;
    double v = (- ((p + 1.0) * (p + 2.0) / 2.0) * p * std::pow(r_over_r_max, p - 1)
                + p * (p + 2.0) * (p + 1.0) * std::pow(r_over_r_max, p)
                - (p * (p + 1.0) / 2.0) * (p + 2.0) * std::pow(r_over_r_max, p + 1));
    v /= r_max;
    return v;
}

void ZBLKokkos::compute_ZBL(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const int*> atomic_numbers,
    Kokkos::View<const double*> r,
    Kokkos::View<const double*> xyz,
    Kokkos::View<double*> node_energies,
    Kokkos::View<double*> node_forces)
{
    Kokkos::View<int*> first_neigh("first_neigh", num_nodes);
    Kokkos::parallel_scan("first_neigh",
        num_nodes,
        KOKKOS_LAMBDA (const int i, int& update, const bool final) {
            const int num_neigh_i = num_neigh(i);
            if (final)
                first_neigh(i) = update;
            update += num_neigh_i;
        });
    Kokkos::fence();

    Kokkos::parallel_for("ZBLKokkos::compute_ZBL",
        Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
        KOKKOS_CLASS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            const int i0 = first_neigh(i);
            const int type_i = node_types(i);
            const int Z_i = atomic_numbers(type_i);
            double e_i;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team_member, num_neigh(i)),
                [&] (const int j, double& e_i) {
                    const int ij = i0 + j;
                    const int type_j = neigh_types(ij);
                    const int Z_j = atomic_numbers(type_j);
                    const auto result = compute_value_gradient(
                        Z_i, Z_j, r(ij));
                    e_i += result.value;
                    node_forces(3*ij)   -= xyz(3*ij)   / r(ij) * result.gradient;
                    node_forces(3*ij+1) -= xyz(3*ij+1) / r(ij) * result.gradient;
                    node_forces(3*ij+2) -= xyz(3*ij+2) / r(ij) * result.gradient;
                }, e_i);
            Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
                node_energies(i) += e_i;
            });
        });
}

void ZBLKokkos::compute_ZBL(
    const Kokkos::DefaultExecutionSpace& execution_space,
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const int*> atomic_numbers,
    Kokkos::View<const int*> first_neigh,
    Kokkos::View<const double*> r,
    Kokkos::View<const double*> xyz,
    Kokkos::View<double*> node_energies,
    Kokkos::View<double*> node_forces,
    const int workspace_edge_begin)
{
    Kokkos::parallel_for(
        "ZBLKokkos::compute_ZBL_prepared",
        Kokkos::TeamPolicy<>(execution_space, num_nodes, Kokkos::AUTO),
        KOKKOS_CLASS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            const int i0 = first_neigh(i);
            const int Z_i = atomic_numbers(node_types(i));
            double e_i;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team_member, num_neigh(i)),
                [&] (const int j, double& local_energy) {
                    const int edge = i0+j;
                    const int workspace_edge = edge-workspace_edge_begin;
                    const int Z_j = atomic_numbers(neigh_types(edge));
                    const auto result = compute_value_gradient(
                        Z_i, Z_j, r(edge));
                    local_energy += result.value;
                    node_forces(3*workspace_edge) -=
                        xyz(3*edge)/r(edge)*result.gradient;
                    node_forces(3*workspace_edge+1) -=
                        xyz(3*edge+1)/r(edge)*result.gradient;
                    node_forces(3*workspace_edge+2) -=
                        xyz(3*edge+2)/r(edge)*result.gradient;
                }, e_i);
            Kokkos::single(Kokkos::PerTeam(team_member), [&] () {
                node_energies(i) += e_i;
            });
        });
}

void ZBLKokkos::compute_ZBL(
    const Kokkos::DefaultExecutionSpace& execution_space,
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const int*> atomic_numbers,
    Kokkos::View<const int*> first_neigh,
    Kokkos::View<const double*> r,
    Kokkos::View<const float*> unit_direction,
    Kokkos::View<double*> node_energies,
    Kokkos::View<double*> node_forces,
    const int workspace_edge_begin)
{
    Kokkos::parallel_for(
        "ZBLKokkos::compute_ZBL_prepared_compact",
        Kokkos::TeamPolicy<>(execution_space, num_nodes, Kokkos::AUTO),
        KOKKOS_CLASS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            const int i0 = first_neigh(i);
            const int Z_i = atomic_numbers(node_types(i));
            double e_i;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team_member, num_neigh(i)),
                [&] (const int j, double& local_energy) {
                    const int edge = i0+j;
                    const int workspace_edge = edge-workspace_edge_begin;
                    const int Z_j = atomic_numbers(neigh_types(edge));
                    const auto result = compute_value_gradient(
                        Z_i, Z_j, r(edge));
                    local_energy += result.value;
                    node_forces(3*workspace_edge) -=
                        unit_direction(3*edge)*result.gradient;
                    node_forces(3*workspace_edge+1) -=
                        unit_direction(3*edge+1)*result.gradient;
                    node_forces(3*workspace_edge+2) -=
                        unit_direction(3*edge+2)*result.gradient;
                }, e_i);
            Kokkos::single(Kokkos::PerTeam(team_member), [&] () {
                node_energies(i) += e_i;
            });
        });
}
