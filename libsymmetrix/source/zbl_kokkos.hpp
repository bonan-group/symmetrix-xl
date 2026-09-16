#pragma once

#include <cstddef>
#include <vector>

#include "Kokkos_Core.hpp"

class ZBLKokkos {

public:

ZBLKokkos();

ZBLKokkos(double a_exp,
          double a_prefactor,
          std::vector<double> c,
          std::vector<double> covalent_radii,
          int p);

KOKKOS_FUNCTION
double compute(const int Z_u, const int Z_v, const double r) const;

KOKKOS_FUNCTION
double compute_gradient(const int Z_u, const int Z_v, const double r) const;

KOKKOS_FUNCTION
double compute_envelope(const double r, const double r_max, const int p) const;

KOKKOS_FUNCTION
double compute_envelope_gradient(const double r, const double r_max, const int p) const;

void compute_ZBL(const int num_nodes,
                 Kokkos::View<const int*> node_types,
                 Kokkos::View<const int*> num_neigh,
                 Kokkos::View<const int*> neigh_types,
                 Kokkos::View<const int*> atomic_numbers,
                 Kokkos::View<const double*> r,
                 Kokkos::View<const double*> xyz,
                 Kokkos::View<double*> node_energies,
                 Kokkos::View<double*> node_forces);

void compute_ZBL(
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
    int workspace_edge_begin = 0);

void compute_ZBL(
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
    int workspace_edge_begin = 0);

private:

struct ValueGradient {
    double value;
    double gradient;
};

KOKKOS_FUNCTION
ValueGradient compute_value_gradient(
    const int Z_u, const int Z_v, const double r) const;

// values set in constructor
double c_0 = 0.0;
double c_1 = 0.0;
double c_2 = 0.0;
double c_3 = 0.0;
double envelope_coefficient_0 = 0.0;
double envelope_coefficient_1 = 0.0;
double envelope_coefficient_2 = 0.0;
std::size_t pair_dimension = 0;
Kokkos::View<double*> pair_inverse_a;
Kokkos::View<double*> pair_prefactor;
Kokkos::View<double*> pair_cutoff;
Kokkos::View<double*, Kokkos::HostSpace> host_pair_inverse_a;
Kokkos::View<double*, Kokkos::HostSpace> host_pair_prefactor;
Kokkos::View<double*, Kokkos::HostSpace> host_pair_cutoff;
int p = 1;

// values taken from mace/modules/radial.py
static constexpr double c_exps_0 = -3.2;
static constexpr double c_exps_1 = -0.9423;
static constexpr double c_exps_2 = -0.4028;
static constexpr double c_exps_3 = -0.2016;
static constexpr double v_prefactor = 14.3996;

};
