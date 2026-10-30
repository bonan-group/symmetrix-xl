// Explicit instantiation of the private particle-mesh forward path.
//
// The evaluator is a class template whose members live in the header, so every
// consumer would otherwise instantiate its own copy of the whole pipeline.  This
// translation unit is the one definition, and the header's `extern template`
// declarations keep the binding translation unit from repeating it.

#include "fourier_d3_reciprocal.hpp"

namespace symmetrix::fourier_d3 {

template class ReciprocalForward<float>;
template class ReciprocalForward<double>;

template std::vector<ReciprocalKernelProbe> probe_reciprocal_kernel<float>(
    std::span<const double>, double, double, double, double);
template std::vector<ReciprocalKernelProbe> probe_reciprocal_kernel<double>(
    std::span<const double>, double, double, double, double);

}  // namespace symmetrix::fourier_d3
