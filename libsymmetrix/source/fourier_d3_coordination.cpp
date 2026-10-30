// Native modified coordination-number and separable-coefficient kernels.
//
// This translation unit is compiled only when the FFT build capability
// resolved a KokkosFFT FFT provider.  It contains no FFT, no mesh assignment,
// and no pair-force accumulation: the reciprocal transforms belong to the reciprocal kernel and
// the coordination adjoint to the reverse chain.  Both precisions are instantiated here so
// that the kernels are compiled exactly once for the library.

#include "fourier_d3_coordination.hpp"

namespace symmetrix::fourier_d3 {

template class CoordinationWorkspace<float>;
template class CoordinationWorkspace<double>;

}  // namespace symmetrix::fourier_d3
