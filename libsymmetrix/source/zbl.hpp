#pragma once

#include <span>
#include <vector>

template <typename Precision>
class ZBLT {

public:

ZBLT();

ZBLT(Precision a_exp,
    Precision a_prefactor,
    std::vector<Precision> c,
    std::vector<Precision> covalent_radii,
    int p);

Precision compute(const int Z_u, const int Z_v, const Precision r);

Precision compute_gradient(const int Z_u, const int Z_v, const Precision r);

Precision compute_envelope(const Precision r, const Precision r_max, const int p);

Precision compute_envelope_gradient(const Precision r, const Precision r_max, const int p);

void compute_ZBL(const int num_nodes,
                 std::span<const int> node_types,
                 std::span<const int> num_neigh,
                 std::span<const int> neigh_types,
                 std::span<const int> atomic_numbers,
                 std::span<const double> r,
                 std::span<const double> xyz,
                 std::span<Precision> node_energies,
                 std::span<Precision> node_forces);

private:

// values set in constructor
Precision a_exp;
Precision a_prefactor;
std::vector<Precision> c;
std::vector<Precision> covalent_radii;
int p;

// values taken from mace/modules/radial.py
static constexpr Precision c_exps_0 = -3.2;
static constexpr Precision c_exps_1 = -0.9423;
static constexpr Precision c_exps_2 = -0.4028;
static constexpr Precision c_exps_3 = -0.2016;
static constexpr Precision v_prefactor = 14.3996;

};

using ZBL = ZBLT<double>;

extern template class ZBLT<float>;
extern template class ZBLT<double>;
