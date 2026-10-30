#include "mace_device_backend.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif
#ifdef KOKKOS_ENABLE_HIP
#include <KokkosKernels_config.h>
#include <hip/hip_runtime_api.h>
#endif

#ifdef KOKKOS_ENABLE_CUDA
void DeviceBackendTraits<Kokkos::Cuda>::check_status(
    const std::int32_t status, const char* operation)
{
    if (status != static_cast<std::int32_t>(cudaSuccess))
        throw std::runtime_error(
            std::string(operation)+": "
            +cudaGetErrorString(static_cast<cudaError_t>(status)));
}

namespace symmetrix::execution {

CudaDeviceGuard::CudaDeviceGuard(const int requested_device)
{
    DeviceBackendTraits<Kokkos::Cuda>::check_status(
        static_cast<std::int32_t>(cudaGetDevice(&previous_device_)),
        "Querying the caller CUDA device");
    if (previous_device_ != requested_device) {
        DeviceBackendTraits<Kokkos::Cuda>::check_status(
            static_cast<std::int32_t>(cudaSetDevice(requested_device)),
            "Selecting the Execution CUDA plugin device");
        restore_ = true;
    }
}

CudaDeviceGuard::~CudaDeviceGuard()
{
    if (restore_)
        static_cast<void>(cudaSetDevice(previous_device_));
}

} // namespace symmetrix::execution
#endif

#ifdef KOKKOS_ENABLE_HIP
void DeviceBackendTraits<Kokkos::HIP>::check_status(
    const std::int32_t status, const char* operation)
{
    if (status != static_cast<std::int32_t>(hipSuccess))
        throw std::runtime_error(
            std::string(operation)+": "
            +hipGetErrorString(static_cast<hipError_t>(status)));
}

namespace symmetrix::execution {

HipDeviceGuard::HipDeviceGuard(const int requested_device)
{
    DeviceBackendTraits<Kokkos::HIP>::check_status(
        static_cast<std::int32_t>(hipGetDevice(&previous_device_)),
        "Querying the caller HIP device");
    if (previous_device_ != requested_device) {
        DeviceBackendTraits<Kokkos::HIP>::check_status(
            static_cast<std::int32_t>(hipSetDevice(requested_device)),
            "Selecting the Execution HIP plugin device");
        restore_ = true;
    }
}

HipDeviceGuard::~HipDeviceGuard()
{
    if (restore_)
        static_cast<void>(hipSetDevice(previous_device_));
}

} // namespace symmetrix::execution
#endif
