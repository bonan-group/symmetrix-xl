// Verifies that the Metal runtime layer compiles MSL at runtime and executes
// FP32 kernels on the GPU. GPU execution is established by the command
// buffer's GPU timestamps, the hardware SIMD-group width observed inside the
// kernel, and agreement with a host reference.

#include "metal_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <limits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <numeric>
#include <string>
#include <vector>

namespace mtl = symmetrix::execution::metal;

namespace {

constexpr const char* kernels = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void axpy(
    device float* y [[buffer(0)]],
    device const float* x [[buffer(1)]],
    constant float& a [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i < n)
        y[i] = fma(a, x[i], y[i]);
}

// Two-level reduction: simd_sum within a SIMD group, threadgroup memory across
// SIMD groups, one FP32 device atomic per threadgroup.
kernel void reduce_sum(
    device const float* x [[buffer(0)]],
    device atomic_float* total [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    threadgroup float* partial [[threadgroup(0)]],
    uint i [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_groups [[simdgroups_per_threadgroup]])
{
    const float value = i < n ? x[i] : 0.0f;
    const float simd_total = simd_sum(value);
    if (lane == 0)
        partial[simd_group] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        const float group_value = lane < simd_groups ? partial[lane] : 0.0f;
        const float group_total = simd_sum(group_value);
        if (lane == 0)
            atomic_fetch_add_explicit(total, group_total, memory_order_relaxed);
    }
}

// Packets with embedded device pointers, as used by generated R1 kernels.
struct PointerPacket {
    device const float* x;
    device float* y;
    uint n;
};

kernel void packet_scale(
    constant PointerPacket& packet [[buffer(0)]],
    uint i [[thread_position_in_grid]])
{
    if (i < packet.n)
        packet.y[i] = 3.0f*packet.x[i];
}

kernel void probe(
    device uint* out [[buffer(0)]],
    uint width [[threads_per_simdgroup]],
    uint i [[thread_position_in_grid]])
{
    if (i == 0)
        out[0] = width;
}
)MSL";

int failures = 0;

void check(const bool condition, const std::string& message)
{
    std::printf("%s %s\n", condition ? "[ ok ]" : "[FAIL]", message.c_str());
    if (!condition)
        ++failures;
}

template <class Function>
bool throws(Function&& function)
{
    try {
        function();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

float max_relative_error(
    const std::vector<float>& expected, const float* actual)
{
    float worst = 0.0f;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const float scale = std::max(1.0f, std::fabs(expected[i]));
        worst = std::max(worst, std::fabs(expected[i]-actual[i])/scale);
    }
    return worst;
}

}  // namespace

int main()
{
    const mtl::MetalInformation probe_info = mtl::metal_information();
    if (!probe_info.available) {
        std::printf("Metal unavailable: %s\n", probe_info.reason.c_str());
        return 1;
    }

    const mtl::Device device = mtl::Device::system_default();
    const mtl::MetalInformation& info = device.information();
    std::printf(
        "device: %s family=%s metal3=%d unified=%d max_tg_threads=%u "
        "max_tg_mem=%u working_set_MiB=%llu\n",
        info.device_name.c_str(), info.gpu_family.c_str(),
        info.supports_metal3, info.unified_memory,
        info.max_threads_per_threadgroup, info.max_threadgroup_memory_bytes,
        static_cast<unsigned long long>(
            info.recommended_working_set_bytes >> 20));

    const mtl::Library library = device.compile(kernels);
    const mtl::Pipeline axpy = device.pipeline(library, "axpy");
    const mtl::Pipeline reduce = device.pipeline(library, "reduce_sum");
    const mtl::Pipeline probe = device.pipeline(library, "probe");
    std::printf(
        "axpy: execution_width=%u max_threads=%u\n",
        axpy.thread_execution_width(),
        axpy.max_total_threads_per_threadgroup());

    // SIMD width observed inside a kernel; host fallbacks would not report
    // the Apple GPU's 32-wide SIMD groups.
    {
        const mtl::Buffer out = device.allocate(sizeof(std::uint32_t));
        auto batch = device.begin();
        batch.set_buffer(0, out).dispatch_threads(probe, {1}, {1});
        const mtl::BatchTiming timing = batch.submit_and_wait();
        check(*out.data<std::uint32_t>() == 32,
            "kernel observed SIMD-group width "
            +std::to_string(*out.data<std::uint32_t>()));
        check(timing.gpu_seconds > 0.0, "command buffer has GPU timestamps");
    }

    // AXPY with a grid that is not a threadgroup multiple.
    const std::uint32_t n = 3'000'017;
    const float a = 1.5f;
    std::vector<float> x(n), y(n), expected(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        x[i] = std::sin(0.001f*static_cast<float>(i));
        y[i] = std::cos(0.002f*static_cast<float>(i));
        expected[i] = std::fma(a, x[i], y[i]);
    }
    const mtl::Buffer x_buffer = device.allocate(n*sizeof(float));
    const mtl::Buffer y_buffer = device.allocate(n*sizeof(float));
    std::copy(x.begin(), x.end(), x_buffer.data<float>());
    std::copy(y.begin(), y.end(), y_buffer.data<float>());
    {
        auto batch = device.begin();
        batch.set_buffer(0, y_buffer).set_buffer(1, x_buffer)
            .set_value(2, a).set_value(3, n)
            .dispatch_threads(axpy, {n}, {256});
        const mtl::BatchTiming timing = batch.submit_and_wait();
        const float error = max_relative_error(expected, y_buffer.data<float>());
        check(error == 0.0f,
            "axpy matches host fma, max relative error "+std::to_string(error));
        std::printf(
            "axpy n=%u gpu=%.3f ms (%.1f GB/s)\n", n, 1e3*timing.gpu_seconds,
            3.0*n*sizeof(float)/timing.gpu_seconds/1e9);
    }

    // Reduction with dynamic threadgroup memory and FP32 device atomics.
    {
        const std::uint32_t threadgroup = 256;
        const std::uint32_t simd_groups =
            threadgroup/reduce.thread_execution_width();
        const mtl::Buffer total = device.allocate(sizeof(float));
        auto batch = device.begin();
        batch.set_buffer(0, x_buffer).set_buffer(1, total).set_value(2, n)
            .set_threadgroup_memory(0, simd_groups*sizeof(float))
            .dispatch_threads(reduce, {n}, {threadgroup});
        batch.submit_and_wait();
        const double reference =
            std::accumulate(x.begin(), x.end(), 0.0);
        const double gpu = *total.data<float>();
        const double relative =
            std::fabs(gpu-reference)/std::max(1.0, std::fabs(reference));
        check(relative < 1e-4,
            "reduce_sum within 1e-4 of FP64 host sum, relative error "
            +std::to_string(relative));
    }

    // Two dispatches in one serial batch observe each other's writes.
    {
        std::copy(y.begin(), y.end(), y_buffer.data<float>());
        auto batch = device.begin();
        batch.set_buffer(0, y_buffer).set_buffer(1, x_buffer)
            .set_value(2, a).set_value(3, n)
            .dispatch_threads(axpy, {n}, {256})
            .dispatch_threads(axpy, {n}, {256});
        const mtl::BatchTiming timing = batch.submit_and_wait();
        for (std::uint32_t i = 0; i < n; ++i)
            expected[i] = std::fma(a, x[i], std::fma(a, x[i], y[i]));
        check(timing.dispatch_count == 2
                && max_relative_error(expected, y_buffer.data<float>()) == 0.0f,
            "serial dispatches are ordered within a batch");
    }

    // Zero-copy wrapping of page-aligned host memory.
    {
        const std::size_t page = info.page_bytes;
        const std::size_t count = 4*page/sizeof(float);
        void* host = nullptr;
        check(posix_memalign(&host, page, count*sizeof(float)) == 0,
            "page-aligned host allocation");
        auto* values = static_cast<float*>(host);
        std::fill(values, values+count, 2.0f);
        {
            const mtl::Buffer wrapped =
                device.wrap(host, count*sizeof(float));
            const mtl::Buffer ones = device.allocate(count*sizeof(float));
            std::fill(ones.data<float>(), ones.data<float>()+count, 1.0f);
            const auto count32 = static_cast<std::uint32_t>(count);
            auto batch = device.begin();
            batch.set_buffer(0, wrapped).set_buffer(1, ones)
                .set_value(2, 3.0f).set_value(3, count32)
                .dispatch_threads(axpy, {count}, {256});
            batch.submit_and_wait();
            check(wrapped.contents() == host && !wrapped.owns_storage(),
                "wrapped buffer aliases host memory");
            check(std::all_of(values, values+count,
                    [](float v) { return v == 5.0f; }),
                "GPU wrote through to wrapped host memory");
            check(throws([&] { device.wrap(values+1, page); }),
                "unaligned host memory is rejected");
        }
        std::free(host);
    }

    // Embedded GPU addresses in an inline argument packet.
    {
        const std::uint32_t count = 1000;
        const mtl::Buffer in = device.allocate(count*sizeof(float));
        const mtl::Buffer out = device.allocate(count*sizeof(float));
        for (std::uint32_t i = 0; i < count; ++i)
            in.data<float>()[i] = static_cast<float>(i);
        const struct {
            std::uint64_t x;
            std::uint64_t y;
            std::uint32_t n;
            std::uint32_t pad;
        } packet{in.gpu_address(), out.gpu_address(), count, 0};
        const mtl::Pipeline scale = device.pipeline(library, "packet_scale");
        auto batch = device.begin();
        batch.use_buffer(in, false).use_buffer(out, true)
            .set_value(0, packet).dispatch_threads(scale, {count}, {64});
        batch.submit_and_wait();
        bool exact = true;
        for (std::uint32_t i = 0; i < count; ++i)
            exact = exact && out.data<float>()[i] == 3.0f*static_cast<float>(i);
        check(in.gpu_address() != 0 && exact,
            "kernel dereferences device pointers embedded in a packet");
    }

    // Strided GEMM followed by a compute dispatch in the same batch.
    {
        const std::uint32_t rows = 37, inner = 70, columns = 50;
        const std::uint32_t left_stride = 96, result_stride = 64;
        const std::size_t left_offset = 5, result_offset = 3;
        const mtl::Buffer left = device.allocate(
            (left_offset+rows*left_stride)*sizeof(float));
        const mtl::Buffer right = device.allocate(inner*columns*sizeof(float));
        const mtl::Buffer result = device.allocate(
            (result_offset+rows*result_stride)*sizeof(float));
        const mtl::Buffer transposed = device.allocate(inner*rows*sizeof(float));
        for (std::size_t i = 0; i < left.size()/sizeof(float); ++i)
            left.data<float>()[i] = std::sin(0.37f*static_cast<float>(i));
        for (std::size_t i = 0; i < inner*columns; ++i)
            right.data<float>()[i] = std::cos(0.11f*static_cast<float>(i));
        const mtl::Buffer probe_out = device.allocate(sizeof(std::uint32_t));
        auto batch = device.begin();
        batch.use_buffer(probe_out, true);
        batch.gemm(
            {&left, left_offset*sizeof(float), rows, inner, left_stride*sizeof(float)},
            {&right, 0, inner, columns, columns*sizeof(float)},
            {&result, result_offset*sizeof(float), rows, columns,
                result_stride*sizeof(float)});
        // left^T written densely: [inner, rows] = op(left)^T via identity right.
        const mtl::Buffer identity = device.allocate(rows*rows*sizeof(float));
        for (std::uint32_t i = 0; i < rows; ++i)
            identity.data<float>()[i*rows+i] = 1.0f;
        batch.gemm(
            {&left, left_offset*sizeof(float), rows, inner, left_stride*sizeof(float)},
            {&identity, 0, rows, rows, rows*sizeof(float)},
            {&transposed, 0, inner, rows, rows*sizeof(float)}, true, false);
        batch.set_buffer(0, probe_out).dispatch_threads(probe, {1}, {1});
        batch.submit_and_wait();
        float worst = 0.0f;
        for (std::uint32_t i = 0; i < rows; ++i)
            for (std::uint32_t j = 0; j < columns; ++j) {
                double expected = 0.0;
                for (std::uint32_t k = 0; k < inner; ++k)
                    expected += static_cast<double>(
                        left.data<float>()[left_offset+i*left_stride+k])
                        *right.data<float>()[k*columns+j];
                const float actual =
                    result.data<float>()[result_offset+i*result_stride+j];
                worst = std::max(worst, static_cast<float>(
                    std::fabs(expected-actual)/std::max(1.0, std::fabs(expected))));
            }
        bool transposed_exact = true;
        for (std::uint32_t k = 0; k < inner; ++k)
            for (std::uint32_t i = 0; i < rows; ++i)
                transposed_exact = transposed_exact
                    && transposed.data<float>()[k*rows+i]
                        == left.data<float>()[left_offset+i*left_stride+k];
        check(worst < 1e-5f,
            "strided GEMM matches FP64 reference, relative error "
            +std::to_string(worst));
        check(transposed_exact, "transposed GEMM operand is exact");
        check(*probe_out.data<std::uint32_t>() == 32,
            "compute dispatch after GEMM reopens its encoder");
    }

    // Large strided GEMM at evaluator scale writes every output row; an
    // earlier MPS-based implementation silently skipped 384-row blocks here.
    {
        const std::uint32_t rows = 78125, inner = 128, columns = 128;
        const std::size_t left_pitch = 2048, result_pitch = 5120;
        const mtl::Buffer left = device.allocate(rows*left_pitch*sizeof(float));
        const mtl::Buffer right = device.allocate(inner*columns*sizeof(float));
        const mtl::Buffer result = device.allocate(rows*result_pitch*sizeof(float));
        std::mt19937 generator(1);
        std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
        for (std::size_t i = 0; i < rows*left_pitch; ++i)
            left.data<float>()[i] = uniform(generator);
        for (std::size_t i = 0; i < inner*columns; ++i)
            right.data<float>()[i] = uniform(generator);
        std::fill(result.data<float>(), result.data<float>()+rows*result_pitch,
            std::numeric_limits<float>::quiet_NaN());
        auto batch = device.begin();
        batch.gemm({&left, 0, rows, inner, left_pitch*sizeof(float)},
            {&right, 0, inner, columns, columns*sizeof(float)},
            {&result, 0, rows, columns, result_pitch*sizeof(float)});
        batch.submit_and_wait();
        std::size_t wrong_rows = 0;
        for (std::size_t r = 0; r < rows; ++r)
            for (std::uint32_t j = 0; j < columns; ++j) {
                double expected = 0.0;
                for (std::uint32_t k = 0; k < inner; ++k)
                    expected += static_cast<double>(left.data<float>()[r*left_pitch+k])
                        *right.data<float>()[k*columns+j];
                if (!(std::fabs(result.data<float>()[r*result_pitch+j]-expected) < 1e-3)) {
                    ++wrong_rows;
                    break;
                }
            }
        check(wrong_rows == 0, "large strided GEMM writes all "+std::to_string(rows)
            +" rows ("+std::to_string(wrong_rows)+" wrong)");
    }

    // Error reporting.
    check(throws([&] { device.compile("kernel void broken( {"); }),
        "MSL compile errors throw");
    check(throws([&] { device.pipeline(library, "missing"); }),
        "missing functions throw");
    check(throws([&] {
            auto batch = device.begin();
            batch.dispatch_threads(axpy, {n}, {4096});
        }),
        "oversized threadgroups are rejected");

    std::printf("%s: %d failure(s)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
