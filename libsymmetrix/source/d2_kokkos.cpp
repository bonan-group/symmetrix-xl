#include "d2_kokkos.hpp"

#include "d2_kokkos_impl.tpp"

namespace symmetrix::dispersion {

template class D2Kokkos<double>;
template class D2Kokkos<float>;

} // namespace symmetrix::dispersion
