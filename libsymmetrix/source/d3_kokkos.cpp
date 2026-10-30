#include "d3_kokkos.hpp"

#include "d3_kokkos_impl.tpp"

namespace symmetrix::dispersion {

template class D3Kokkos<double>;
template class D3Kokkos<float>;

} // namespace symmetrix::dispersion
