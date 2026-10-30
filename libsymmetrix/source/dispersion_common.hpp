#ifndef SYMMETRIX_DISPERSION_COMMON_HPP
#define SYMMETRIX_DISPERSION_COMMON_HPP

#include <Kokkos_Core.hpp>

namespace symmetrix::dispersion {

struct DispersionResultKokkos {
    double energy = 0.0;
    Kokkos::View<double*> atomic_energies;
    Kokkos::View<double*[3]> forces;
    Kokkos::View<double[6]> stress;
};

} // namespace symmetrix::dispersion

#endif
