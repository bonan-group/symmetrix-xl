// Explicit instantiation of the private analytic reverse path.
//
// The orchestrator is a class template whose members live in the header, so
// every consumer would otherwise instantiate its own copy of the whole reverse
// chain.  This translation unit is the one definition, and the header's
// `extern template` declarations keep the binding translation unit from
// repeating it.

#include "fourier_d3_reverse.hpp"

namespace symmetrix::fourier_d3 {

template class ReciprocalReverse<float>;
template class ReciprocalReverse<double>;

}  // namespace symmetrix::fourier_d3
