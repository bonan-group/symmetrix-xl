#include "device_backend.hpp"
#include "spherical_harmonic_device.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

#ifdef KOKKOS_ENABLE_HIP
#include <KokkosKernels_config.h>
#include <hip/hip_runtime_api.h>
#endif

namespace {

std::string version_string(int version)
{
    if (version <= 0)
        return {};
    const int major = version / 10000000 != 0
        ? version / 10000000 : version / 1000;
    const int minor = version / 10000000 != 0
        ? (version / 100000) % 100 : (version / 10) % 100;
    std::ostringstream result;
    result << major << '.' << minor;
    return result.str();
}

std::string compiler_version()
{
#if defined(__HIPCC__) && defined(HIP_VERSION_MAJOR)
    return std::to_string(HIP_VERSION_MAJOR) + "."
        + std::to_string(HIP_VERSION_MINOR) + "."
        + std::to_string(HIP_VERSION_PATCH);
#elif defined(__CUDACC_VER_MAJOR__)
    return std::to_string(__CUDACC_VER_MAJOR__) + "."
        + std::to_string(__CUDACC_VER_MINOR__) + "."
        + std::to_string(__CUDACC_VER_BUILD__);
#elif defined(__clang__)
    return __clang_version__;
#elif defined(__GNUC__)
    return std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__);
#else
    return "unknown";
#endif
}

} // namespace

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

HipAgentTarget normalize_hip_agent_target(const std::string& target)
{
    HipAgentTarget result;
    result.raw = target;
    const auto separator = target.find(':');
    result.base_isa = target.substr(0, separator);
    std::transform(
        result.base_isa.begin(), result.base_isa.end(), result.base_isa.begin(),
        [](unsigned char value) { return std::tolower(value); });
    if (result.base_isa.rfind("gfx", 0) != 0)
        throw std::invalid_argument("HIP agent target must begin with 'gfx'");
    if (separator != std::string::npos) {
        std::vector<std::string> features;
        std::istringstream stream(target.substr(separator + 1));
        std::string feature;
        while (std::getline(stream, feature, ':')) {
            if (!feature.empty())
                features.push_back(feature);
        }
        std::sort(features.begin(), features.end());
        for (std::size_t index = 0; index < features.size(); ++index) {
            if (index != 0)
                result.features += ':';
            result.features += features[index];
        }
    }
    return result;
}

DeviceExecutionEnvironment execution_device_execution_environment()
{
    DeviceExecutionEnvironment result;
    result.available = Kokkos::is_initialized() && !Kokkos::is_finalized();
    result.execution_space = Kokkos::DefaultExecutionSpace::name();
    result.memory_space = Kokkos::DefaultExecutionSpace::memory_space::name();
    result.compiler_version = compiler_version();

#ifdef KOKKOS_ENABLE_CUDA
    result.backend = "cuda";
    result.device_memory = true;
    if (!result.available)
        return result;
    int device = -1;
    cudaDeviceProp properties{};
    int runtime = 0;
    int driver = 0;
    if (cudaGetDevice(&device) != cudaSuccess
        || cudaGetDeviceProperties(&properties, device) != cudaSuccess)
        return result;
    cudaRuntimeGetVersion(&runtime);
    cudaDriverGetVersion(&driver);
    result.device_ordinal = device;
    result.device_name = properties.name;
    result.architecture = "sm_" + std::to_string(properties.major)
        + std::to_string(properties.minor);
    result.compiler_offload_target = result.architecture;
    result.compute_capability = 10 * properties.major + properties.minor;
    result.native_subgroup_width = properties.warpSize;
    result.compute_unit_count = properties.multiProcessorCount;
    result.max_team_size = properties.maxThreadsPerBlock;
    result.max_shared_memory_per_block = properties.sharedMemPerBlock;
    result.runtime_version = version_string(runtime);
    result.driver_version = version_string(driver);
    result.aot_available = true;
    result.jit_available = true;
    result.batched_blas_available = true;
#elif defined(KOKKOS_ENABLE_HIP)
    result.backend = "hip";
    result.device_memory = true;
    if (!result.available)
        return result;
    int device = -1;
    hipDeviceProp_t properties{};
    int runtime = 0;
    int driver = 0;
    if (hipGetDevice(&device) != hipSuccess
        || hipGetDeviceProperties(&properties, device) != hipSuccess)
        return result;
    static_cast<void>(hipRuntimeGetVersion(&runtime));
    static_cast<void>(hipDriverGetVersion(&driver));
    const auto target = normalize_hip_agent_target(properties.gcnArchName);
    result.device_ordinal = device;
    result.device_name = properties.name;
    result.raw_agent_target = target.raw;
    result.architecture = target.base_isa;
    result.target_features = target.features;
    result.compiler_offload_target = target.base_isa;
    result.native_subgroup_width = properties.warpSize;
    result.compute_unit_count = properties.multiProcessorCount;
    result.max_team_size = properties.maxThreadsPerBlock;
    result.max_shared_memory_per_block = properties.sharedMemPerBlock;
    result.runtime_version = version_string(runtime);
    result.driver_version = version_string(driver);
    result.aot_available = false;
    result.jit_available = true;
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCBLAS
    result.batched_blas_available = true;
#else
    result.batched_blas_available = false;
#endif
#else
    result.backend = "host";
    result.device_name = "host";
    result.architecture = "host";
    result.compiler_offload_target = "host";
#endif
    return result;
}

bool kokkos_device_sentinel()
{
    if (!Kokkos::is_initialized() || Kokkos::is_finalized())
        return false;
    Kokkos::View<int*> value("symmetrix device sentinel", 1);
    const Kokkos::DefaultExecutionSpace execution_space;
    Kokkos::parallel_for(
        "symmetrix device sentinel",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(execution_space, 0, 1),
        KOKKOS_LAMBDA(const int) { value(0) = 0x5a17; });
    auto host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), value);
    return host(0) == 0x5a17;
}

std::pair<std::vector<double>, std::vector<double>>
kokkos_spherical_harmonics_reference(
    const std::vector<double>& coordinates, const int l_max)
{
    if (coordinates.size() % 3 != 0)
        throw std::invalid_argument("coordinates must contain complete xyz triples");
    if (l_max < 0 || l_max > 6)
        throw std::invalid_argument(
            "portable spherical harmonics support l_max values from 0 through 6");
    const int samples = static_cast<int>(coordinates.size() / 3);
    const int size = (l_max + 1) * (l_max + 1);
    Kokkos::View<double*> xyz(
        "spherical harmonic coordinates", coordinates.size());
    Kokkos::View<double*> values(
        "spherical harmonic values", static_cast<std::size_t>(samples) * size);
    Kokkos::View<double*> gradients(
        "spherical harmonic gradients",
        static_cast<std::size_t>(samples) * 3 * size);
    auto host_xyz = Kokkos::create_mirror_view(xyz);
    for (std::size_t index = 0; index < coordinates.size(); ++index)
        host_xyz(index) = coordinates[index];
    Kokkos::deep_copy(xyz, host_xyz);
    const Kokkos::DefaultExecutionSpace execution_space;
    symmetrix::launch_spherical_harmonics_device(
        execution_space, xyz, values, gradients, samples, l_max);
    auto host_values = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), values);
    auto host_gradients = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), gradients);
    std::vector<double> result_values(host_values.extent(0));
    std::vector<double> result_gradients(host_gradients.extent(0));
    for (std::size_t index = 0; index < result_values.size(); ++index)
        result_values[index] = host_values(index);
    for (std::size_t index = 0; index < result_gradients.size(); ++index)
        result_gradients[index] = host_gradients(index);
    return {std::move(result_values), std::move(result_gradients)};
}
