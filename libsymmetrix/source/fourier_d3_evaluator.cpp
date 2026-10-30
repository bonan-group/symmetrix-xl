// Explicit instantiation of the private FourierD3 evaluator.
//
// The evaluator is a class template whose members live in the header, so every
// consumer would otherwise instantiate its own copy of the whole composition.
// This translation unit is the one definition, and the header's `extern
// template` declarations keep the binding translation unit from repeating it.

#include "fourier_d3_evaluator.hpp"

namespace symmetrix::fourier_d3 {

template class FourierD3Kokkos<float>;
template class FourierD3Kokkos<double>;

}  // namespace symmetrix::fourier_d3
