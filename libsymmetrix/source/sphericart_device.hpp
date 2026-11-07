#pragma once

#include <Kokkos_Core.hpp>

#include "sphericart.hpp"
#if defined(SYMMETRIX_SPHERICART_CUDA)
#include "sphericart_cuda.hpp"
#elif defined(SYMMETRIX_SPHERICART_HIP)
#include "sphericart_hip.hpp"
#endif

namespace symmetrix {

#if defined(SYMMETRIX_SPHERICART_CUDA) || defined(SYMMETRIX_SPHERICART_HIP)

// SpheriCart's CUDA and HIP calculators expose the same lifecycle (compute,
// opt-in asynchronous gradients, explicit device release), so the evaluator
// drives them through one alias and only the raw launch stream differs.
#if defined(SYMMETRIX_SPHERICART_CUDA)
template <class Precision>
using DeviceSphericartCalculator = sphericart::cuda::SphericalHarmonics<Precision>;
#else
template <class Precision>
using DeviceSphericartCalculator = sphericart::hip::SphericalHarmonics<Precision>;
#endif

// Kokkos names the raw stream accessor after the backend (cuda_stream vs
// hip_stream), so the shared device branch resolves it here.
inline void* native_device_stream(const Kokkos::DefaultExecutionSpace& execution_space) {
#if defined(SYMMETRIX_SPHERICART_CUDA)
    return reinterpret_cast<void*>(execution_space.cuda_stream());
#else
    return reinterpret_cast<void*>(execution_space.hip_stream());
#endif
}

#endif

} // namespace symmetrix
