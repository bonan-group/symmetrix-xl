#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(hipError_t status, const char* operation)
{
    if (status != hipSuccess)
        throw std::runtime_error(
            std::string(operation)+": "+hipGetErrorString(status));
}

template<class T>
class DeviceBuffer
{
public:
    explicit DeviceBuffer(std::size_t count) : count_(count)
    {
        check(hipMalloc(&data_, count*sizeof(T)), "hipMalloc");
    }
    ~DeviceBuffer() { static_cast<void>(hipFree(data_)); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* data() { return data_; }
    const T* data() const { return data_; }
    std::size_t bytes() const { return count_*sizeof(T); }
private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

__global__ void project_raw_geometry(
    const double* xyz, const double* radius, float* output,
    std::size_t edges)
{
    const std::size_t edge = blockIdx.x*blockDim.x+threadIdx.x;
    if (edge >= edges) return;
    const double r = radius[edge];
    const float x = static_cast<float>(xyz[3*edge]/r);
    const float y = static_cast<float>(xyz[3*edge+1]/r);
    const float z = static_cast<float>(xyz[3*edge+2]/r);
    const float radial = static_cast<float>((r-0.5)*0.25);
    output[edge] = radial*(0.25f*x-0.5f*y+0.75f*z);
}

__global__ void project_unit_geometry(
    const float* direction, const double* radius, float* output,
    std::size_t edges)
{
    const std::size_t edge = blockIdx.x*blockDim.x+threadIdx.x;
    if (edge >= edges) return;
    const double r = radius[edge];
    const float x = direction[3*edge];
    const float y = direction[3*edge+1];
    const float z = direction[3*edge+2];
    const float radial = static_cast<float>((r-0.5)*0.25);
    output[edge] = radial*(0.25f*x-0.5f*y+0.75f*z);
}

template<class Launch>
double time_kernel(Launch launch, int warmups, int repeats)
{
    for (int repeat=0; repeat<warmups; ++repeat) launch();
    check(hipDeviceSynchronize(), "warmup synchronization");
    hipEvent_t begin = nullptr;
    hipEvent_t end = nullptr;
    check(hipEventCreate(&begin), "hipEventCreate begin");
    check(hipEventCreate(&end), "hipEventCreate end");
    check(hipEventRecord(begin), "hipEventRecord begin");
    for (int repeat=0; repeat<repeats; ++repeat) launch();
    check(hipEventRecord(end), "hipEventRecord end");
    check(hipEventSynchronize(end), "hipEventSynchronize");
    float milliseconds = 0.0f;
    check(hipEventElapsedTime(&milliseconds, begin, end), "hipEventElapsedTime");
    check(hipEventDestroy(begin), "hipEventDestroy begin");
    check(hipEventDestroy(end), "hipEventDestroy end");
    return milliseconds/repeats;
}

void run(std::size_t edges, int repeats)
{
    std::mt19937 generator(20260815);
    std::uniform_real_distribution<double> coordinate(-4.0, 4.0);
    std::vector<double> xyz(3*edges);
    std::vector<double> radius(edges);
    std::vector<float> direction(3*edges);
    for (std::size_t edge=0; edge<edges; ++edge) {
        double squared = 0.0;
        do {
            squared = 0.0;
            for (int component=0; component<3; ++component) {
                const double value = coordinate(generator);
                xyz[3*edge+component] = value;
                squared += value*value;
            }
        } while (squared < 0.25 || squared > 36.0);
        radius[edge] = std::sqrt(squared);
        for (int component=0; component<3; ++component)
            direction[3*edge+component] = static_cast<float>(
                xyz[3*edge+component]/radius[edge]);
    }

    DeviceBuffer<double> d_xyz(xyz.size());
    DeviceBuffer<double> d_radius(radius.size());
    DeviceBuffer<float> d_direction(direction.size());
    DeviceBuffer<float> d_raw_output(edges);
    DeviceBuffer<float> d_unit_output(edges);
    check(hipMemcpy(d_xyz.data(), xyz.data(), d_xyz.bytes(), hipMemcpyHostToDevice),
        "copy xyz");
    check(hipMemcpy(
        d_radius.data(), radius.data(), d_radius.bytes(), hipMemcpyHostToDevice),
        "copy radius");
    check(hipMemcpy(d_direction.data(), direction.data(), d_direction.bytes(),
        hipMemcpyHostToDevice), "copy direction");

    constexpr int threads = 256;
    const int blocks = static_cast<int>((edges+threads-1)/threads);
    const auto launch_raw = [&] {
        hipLaunchKernelGGL(project_raw_geometry, dim3(blocks), dim3(threads), 0, 0,
            d_xyz.data(), d_radius.data(), d_raw_output.data(), edges);
    };
    const auto launch_unit = [&] {
        hipLaunchKernelGGL(project_unit_geometry, dim3(blocks), dim3(threads), 0, 0,
            d_direction.data(), d_radius.data(), d_unit_output.data(), edges);
    };
    const double raw_ms = time_kernel(launch_raw, 20, repeats);
    const double unit_ms = time_kernel(launch_unit, 20, repeats);

    std::vector<float> raw_output(edges);
    std::vector<float> unit_output(edges);
    check(hipMemcpy(raw_output.data(), d_raw_output.data(), d_raw_output.bytes(),
        hipMemcpyDeviceToHost), "copy raw output");
    check(hipMemcpy(unit_output.data(), d_unit_output.data(), d_unit_output.bytes(),
        hipMemcpyDeviceToHost), "copy unit output");
    double max_absolute = 0.0;
    double rms = 0.0;
    for (std::size_t edge=0; edge<edges; ++edge) {
        const double difference =
            static_cast<double>(unit_output[edge])-raw_output[edge];
        max_absolute = std::max(max_absolute, std::abs(difference));
        rms += difference*difference;
    }
    rms = std::sqrt(rms/edges);

    const auto gib = [] (std::size_t bytes) {
        return static_cast<double>(bytes)/(1024.0*1024.0*1024.0);
    };
    std::cout << std::fixed << std::setprecision(6)
        << "edges=" << edges << " repeats=" << repeats << '\n'
        << "raw_ms=" << raw_ms
        << " raw_ns_per_edge=" << raw_ms*1.0e6/edges << '\n'
        << "unit_ms=" << unit_ms
        << " unit_ns_per_edge=" << unit_ms*1.0e6/edges << '\n'
        << "speedup=" << raw_ms/unit_ms << '\n'
        << "raw_geometry_GiB=" << gib(edges*4*sizeof(double)) << '\n'
        << "unit_f32_radius_f64_GiB="
        << gib(edges*(3*sizeof(float)+sizeof(double))) << '\n'
        << std::scientific
        << "max_absolute_difference=" << max_absolute << '\n'
        << "rms_difference=" << rms << '\n';
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const std::size_t edges = argc > 1
            ? std::stoull(argv[1]) : std::size_t(78624);
        const int repeats = argc > 2 ? std::stoi(argv[2]) : 200;
        if (edges == 0 || repeats <= 0)
            throw std::invalid_argument("edges and repeats must be positive");
        run(edges, repeats);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "compact edge geometry benchmark failed: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
