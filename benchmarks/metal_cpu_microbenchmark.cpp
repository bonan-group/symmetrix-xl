// FP32 Metal GPU versus CPU microbenchmarks on Apple silicon.
//
// The kernels are proxies for MACE stages, not the Symmetrix evaluator:
//   triad      bandwidth-bound elementwise update
//   sgemm      dense channel mixing; CPU uses Accelerate (AMX), GPU uses a
//              simdgroup_matrix kernel through the Symmetrix Metal runtime
//   edge       receiver-sorted A-basis aggregation: Bessel radial basis with a
//              polynomial envelope, learned radial mixing, real spherical
//              harmonics through l=3, and a gather of sender features
//
// CPU timings use one thread and all performance cores. GPU timings report the
// command-buffer GPU interval and the host wall time including submission.

#include "metal_runtime.hpp"

#include <Accelerate/Accelerate.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace mtl = symmetrix::execution::metal;

namespace {

constexpr std::uint32_t channels = 32;
constexpr std::uint32_t harmonics = 16;  // l = 0..3
constexpr std::uint32_t radial_basis = 8;
constexpr float cutoff = 5.0f;
constexpr int repetitions = 10;

constexpr const char* gpu_source = R"MSL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void triad(
    device float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* c [[buffer(2)]],
    constant float& s [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i < n)
        a[i] = fma(s, c[i], b[i]);
}

// Row-major C = A*B with M, N multiples of 64 and K a multiple of 8. Each
// threadgroup owns a 64x64 tile; each of its four SIMD groups owns 32x32.
kernel void sgemm(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint3& dims [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    const uint K = dims.z, N = dims.y;
    const uint row0 = group.y*64 + (simd_group/2)*32;
    const uint col0 = group.x*64 + (simd_group%2)*32;
    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 4; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0; k < K; k += 8) {
        simdgroup_float8x8 a[4], b[4];
        for (uint i = 0; i < 4; ++i)
            simdgroup_load(a[i], A + ulong(row0 + 8*i)*K + k, K);
        for (uint j = 0; j < 4; ++j)
            simdgroup_load(b[j], B + ulong(k)*N + col0 + 8*j, N);
        for (uint i = 0; i < 4; ++i)
            for (uint j = 0; j < 4; ++j)
                simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 4; ++j)
            simdgroup_store(acc[i][j], C + ulong(row0 + 8*i)*N + col0 + 8*j, N);
}

constant constexpr uint CH = 32;
constant constexpr uint LM = 16;
constant constexpr uint NB = 8;
constant constexpr uint CHUNK = 32;
constant constexpr uint THREADS = 128;

inline void edge_geometry(
    device const float* v, float rc, threadgroup float* Y, threadgroup float* B)
{
    const float r = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    const float x = v[0]/r, y = v[1]/r, z = v[2]/r;
    Y[0] = 0.28209479f;
    Y[1] = 0.48860251f*y; Y[2] = 0.48860251f*z; Y[3] = 0.48860251f*x;
    Y[4] = 1.09254843f*x*y; Y[5] = 1.09254843f*y*z;
    Y[6] = 0.31539157f*(3.0f*z*z - 1.0f);
    Y[7] = 1.09254843f*x*z; Y[8] = 0.54627422f*(x*x - y*y);
    Y[9] = 0.59004359f*y*(3.0f*x*x - y*y);
    Y[10] = 2.89061144f*x*y*z;
    Y[11] = 0.45704580f*y*(5.0f*z*z - 1.0f);
    Y[12] = 0.37317633f*z*(5.0f*z*z - 3.0f);
    Y[13] = 0.45704580f*x*(5.0f*z*z - 1.0f);
    Y[14] = 1.44530572f*z*(x*x - y*y);
    Y[15] = 0.59004359f*x*(x*x - 3.0f*y*y);
    const float u = r/rc;
    const float u2 = u*u, u6 = u2*u2*u2;
    const float envelope =
        u < 1.0f ? 1.0f - 28.0f*u6 + 48.0f*u6*u - 21.0f*u6*u2 : 0.0f;
    const float scale = sqrt(2.0f/rc)*envelope/r;
    for (uint k = 0; k < NB; ++k)
        B[k] = scale*sin(float(k + 1)*M_PI_F*u);
}

// One threadgroup per receiving atom. Edge geometry is computed once per edge
// into threadgroup memory; each thread then owns one channel and four
// harmonic components of the atom's A-basis.
kernel void edge_aggregate(
    device const float* edge_vectors [[buffer(0)]],
    device const uint* senders [[buffer(1)]],
    device const uint* offsets [[buffer(2)]],
    device const float* h [[buffer(3)]],
    constant float* W [[buffer(4)]],
    device float* A [[buffer(5)]],
    constant float& rc [[buffer(6)]],
    uint node [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]])
{
    threadgroup float Y[CHUNK][LM];
    threadgroup float B[CHUNK][NB];
    threadgroup float RH[CHUNK][CH];
    threadgroup uint S[CHUNK];
    const uint c = t % CH;
    const uint lm0 = (t / CH)*4;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const uint begin = offsets[node], end = offsets[node + 1];
    for (uint base = begin; base < end; base += CHUNK) {
        const uint count = min(CHUNK, end - base);
        if (t < count) {
            edge_geometry(edge_vectors + 3*(base + t), rc, Y[t], B[t]);
            S[t] = senders[base + t];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint index = t; index < count*CH; index += THREADS) {
            const uint e = index / CH, cc = index % CH;
            float radial = 0.0f;
            for (uint k = 0; k < NB; ++k)
                radial = fma(W[k*CH + cc], B[e][k], radial);
            RH[e][cc] = radial*h[S[e]*CH + cc];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = 0; e < count; ++e) {
            const float rh = RH[e][c];
            acc0 = fma(rh, Y[e][lm0], acc0);
            acc1 = fma(rh, Y[e][lm0 + 1], acc1);
            acc2 = fma(rh, Y[e][lm0 + 2], acc2);
            acc3 = fma(rh, Y[e][lm0 + 3], acc3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float* out = A + (ulong(node)*CH + c)*LM + lm0;
    out[0] = acc0; out[1] = acc1; out[2] = acc2; out[3] = acc3;
}
)MSL";

using Clock = std::chrono::steady_clock;

double seconds_since(const Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    return values[values.size()/2];
}

unsigned performance_cores()
{
    unsigned value = 0;
    std::size_t size = sizeof(value);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &value, &size, nullptr, 0) != 0
        || value == 0)
        value = std::max(1u, std::thread::hardware_concurrency());
    return value;
}

void parallel_for(
    const unsigned threads, const std::size_t n,
    const std::function<void(std::size_t, std::size_t)>& body)
{
    if (threads <= 1) {
        body(0, n);
        return;
    }
    std::vector<std::thread> workers;
    const std::size_t block = (n + threads - 1)/threads;
    for (unsigned w = 0; w < threads; ++w) {
        const std::size_t begin = std::min(n, w*block);
        const std::size_t end = std::min(n, begin + block);
        if (begin < end)
            workers.emplace_back(body, begin, end);
    }
    for (auto& worker : workers)
        worker.join();
}

template <class Function>
double time_cpu(Function&& run)
{
    run();
    std::vector<double> samples;
    for (int i = 0; i < repetitions; ++i) {
        const auto start = Clock::now();
        run();
        samples.push_back(seconds_since(start));
    }
    return median(samples);
}

struct GpuTime {
    double gpu = 0.0;
    double wall = 0.0;
};

template <class Encode>
GpuTime time_gpu(const mtl::Device& device, Encode&& encode)
{
    auto once = [&] {
        auto batch = device.begin();
        encode(batch);
        const auto start = Clock::now();
        const mtl::BatchTiming timing = batch.submit_and_wait();
        return GpuTime{timing.gpu_seconds, seconds_since(start)};
    };
    once();
    std::vector<double> gpu, wall;
    for (int i = 0; i < repetitions; ++i) {
        const GpuTime t = once();
        gpu.push_back(t.gpu);
        wall.push_back(t.wall);
    }
    return {median(gpu), median(wall)};
}

float max_relative_error(const float* expected, const float* actual, std::size_t n)
{
    float scale = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
        scale = std::max(scale, std::fabs(expected[i]));
    float worst = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
        worst = std::max(worst, std::fabs(expected[i] - actual[i]));
    return scale > 0.0f ? worst/scale : worst;
}

void report(
    const char* name, const char* metric, double cpu1, double cpun,
    const GpuTime& gpu, unsigned threads, double scale, float error)
{
    std::printf(
        "%-24s %-10s cpu1=%10.3f cpu%u=%10.3f gpu=%10.3f gpu_wall=%10.3f "
        "speedup_vs_cpu1=%6.1fx speedup_vs_cpu%u=%6.1fx max_rel_err=%.2e\n",
        name, metric, cpu1*scale, threads, cpun*scale, gpu.gpu*scale,
        gpu.wall*scale, cpu1/gpu.wall, threads, cpun/gpu.wall, error);
}

void benchmark_triad(const mtl::Device& device, const mtl::Pipeline& pipeline,
    unsigned threads)
{
    const std::uint32_t n = 64u << 20;
    const float s = 0.5f;
    const mtl::Buffer a = device.allocate(std::size_t(n)*4);
    const mtl::Buffer b = device.allocate(std::size_t(n)*4);
    const mtl::Buffer c = device.allocate(std::size_t(n)*4);
    float* pa = a.data<float>();
    float* pb = b.data<float>();
    float* pc = c.data<float>();
    for (std::uint32_t i = 0; i < n; ++i) {
        pb[i] = float(i % 1024)*1e-3f;
        pc[i] = float(i % 977)*1e-3f;
    }
    auto cpu = [&](unsigned workers) {
        parallel_for(workers, n, [&](std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; ++i)
                pa[i] = std::fma(s, pc[i], pb[i]);
        });
    };
    std::vector<float> expected(n);
    const double cpu1 = time_cpu([&] { cpu(1); });
    const double cpun = time_cpu([&] { cpu(threads); });
    std::copy(pa, pa + n, expected.begin());
    std::fill(pa, pa + n, 0.0f);
    const GpuTime gpu = time_gpu(device, [&](mtl::CommandBatch& batch) {
        batch.set_buffer(0, a).set_buffer(1, b).set_buffer(2, c)
            .set_value(3, s).set_value(4, n)
            .dispatch_threads(pipeline, {n}, {256});
    });
    const double bytes = 3.0*n*4;
    std::printf(
        "triad n=%u bandwidth GB/s: cpu1=%.1f cpu%u=%.1f gpu=%.1f\n", n,
        bytes/cpu1/1e9, threads, bytes/cpun/1e9, bytes/gpu.gpu/1e9);
    report("triad", "ms", cpu1, cpun, gpu, threads, 1e3,
        max_relative_error(expected.data(), pa, n));
}

void benchmark_sgemm(const mtl::Device& device, const mtl::Pipeline& pipeline,
    std::uint32_t size)
{
    const std::size_t elements = std::size_t(size)*size;
    const mtl::Buffer A = device.allocate(elements*4);
    const mtl::Buffer B = device.allocate(elements*4);
    const mtl::Buffer C = device.allocate(elements*4);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < elements; ++i) {
        A.data<float>()[i] = dist(rng);
        B.data<float>()[i] = dist(rng);
    }
    std::vector<float> expected(elements);
    const int m = static_cast<int>(size);
    const double cpu = time_cpu([&] {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, m, m, 1.0f,
            A.data<float>(), m, B.data<float>(), m, 0.0f, expected.data(), m);
    });
    const struct { std::uint32_t M, N, K, pad; } dims{size, size, size, 0};
    const GpuTime gpu = time_gpu(device, [&](mtl::CommandBatch& batch) {
        batch.set_buffer(0, A).set_buffer(1, B).set_buffer(2, C)
            .set_value(3, dims)
            .dispatch_threadgroups(pipeline, {size/64, size/64}, {128});
    });
    const double flops = 2.0*double(size)*size*size;
    const float error =
        max_relative_error(expected.data(), C.data<float>(), elements);
    std::printf(
        "sgemm %u^3 GFLOP/s: accelerate=%.0f gpu=%.0f (gpu_wall %.0f) "
        "speedup_vs_accelerate=%.2fx max_rel_err=%.2e\n",
        size, flops/cpu/1e9, flops/gpu.gpu/1e9, flops/gpu.wall/1e9,
        cpu/gpu.wall, error);
}

struct EdgeProblem {
    std::uint32_t atoms = 0;
    std::uint32_t edges = 0;
    std::vector<float> edge_vectors;
    std::vector<std::uint32_t> senders;
    std::vector<std::uint32_t> offsets;
    std::vector<float> h;
    std::vector<float> W;
};

EdgeProblem make_edge_problem(std::uint32_t atoms, std::uint32_t neighbors)
{
    EdgeProblem p;
    p.atoms = atoms;
    p.edges = atoms*neighbors;
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::uniform_real_distribution<float> radius(0.8f, 0.98f*cutoff);
    std::uniform_int_distribution<std::uint32_t> sender(0, atoms - 1);
    p.offsets.resize(atoms + 1);
    for (std::uint32_t i = 0; i <= atoms; ++i)
        p.offsets[i] = i*neighbors;
    p.edge_vectors.resize(3*std::size_t(p.edges));
    p.senders.resize(p.edges);
    for (std::uint32_t e = 0; e < p.edges; ++e) {
        float x, y, z, n2;
        do {
            x = unit(rng); y = unit(rng); z = unit(rng);
            n2 = x*x + y*y + z*z;
        } while (n2 < 1e-2f || n2 > 1.0f);
        const float r = radius(rng)/std::sqrt(n2);
        p.edge_vectors[3*e] = x*r;
        p.edge_vectors[3*e + 1] = y*r;
        p.edge_vectors[3*e + 2] = z*r;
        p.senders[e] = sender(rng);
    }
    p.h.resize(std::size_t(atoms)*channels);
    for (float& v : p.h)
        v = unit(rng);
    p.W.resize(radial_basis*channels);
    for (float& v : p.W)
        v = 0.3f*unit(rng);
    return p;
}

void cpu_edge_aggregate(
    const EdgeProblem& p, float* A, std::size_t begin, std::size_t end)
{
    constexpr float pi = 3.14159265358979f;
    const float norm = std::sqrt(2.0f/cutoff);
    for (std::size_t node = begin; node < end; ++node) {
        float acc[channels][harmonics] = {};
        for (std::uint32_t e = p.offsets[node]; e < p.offsets[node + 1]; ++e) {
            const float* v = &p.edge_vectors[3*std::size_t(e)];
            const float r = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
            const float x = v[0]/r, y = v[1]/r, z = v[2]/r;
            const float Y[harmonics] = {
                0.28209479f,
                0.48860251f*y, 0.48860251f*z, 0.48860251f*x,
                1.09254843f*x*y, 1.09254843f*y*z,
                0.31539157f*(3.0f*z*z - 1.0f),
                1.09254843f*x*z, 0.54627422f*(x*x - y*y),
                0.59004359f*y*(3.0f*x*x - y*y),
                2.89061144f*x*y*z,
                0.45704580f*y*(5.0f*z*z - 1.0f),
                0.37317633f*z*(5.0f*z*z - 3.0f),
                0.45704580f*x*(5.0f*z*z - 1.0f),
                1.44530572f*z*(x*x - y*y),
                0.59004359f*x*(x*x - 3.0f*y*y)};
            const float u = r/cutoff;
            const float u2 = u*u, u6 = u2*u2*u2;
            const float envelope =
                u < 1.0f ? 1.0f - 28.0f*u6 + 48.0f*u6*u - 21.0f*u6*u2 : 0.0f;
            const float scale = norm*envelope/r;
            float B[radial_basis];
            for (std::uint32_t k = 0; k < radial_basis; ++k)
                B[k] = scale*std::sin(float(k + 1)*pi*u);
            const float* hj = &p.h[std::size_t(p.senders[e])*channels];
            for (std::uint32_t c = 0; c < channels; ++c) {
                float radial = 0.0f;
                for (std::uint32_t k = 0; k < radial_basis; ++k)
                    radial = std::fma(p.W[k*channels + c], B[k], radial);
                const float rh = radial*hj[c];
                for (std::uint32_t lm = 0; lm < harmonics; ++lm)
                    acc[c][lm] = std::fma(rh, Y[lm], acc[c][lm]);
            }
        }
        std::copy(&acc[0][0], &acc[0][0] + channels*harmonics,
            A + node*channels*harmonics);
    }
}

void benchmark_edge(const mtl::Device& device, const mtl::Pipeline& pipeline,
    unsigned threads, std::uint32_t atoms, std::uint32_t neighbors)
{
    const EdgeProblem p = make_edge_problem(atoms, neighbors);
    const std::size_t outputs = std::size_t(atoms)*channels*harmonics;
    std::vector<float> expected(outputs);
    const double cpu1 = time_cpu([&] {
        cpu_edge_aggregate(p, expected.data(), 0, atoms);
    });
    const double cpun = time_cpu([&] {
        parallel_for(threads, atoms, [&](std::size_t begin, std::size_t end) {
            cpu_edge_aggregate(p, expected.data(), begin, end);
        });
    });

    auto upload = [&](const auto& values) {
        using T = typename std::decay_t<decltype(values)>::value_type;
        const mtl::Buffer buffer = device.allocate(values.size()*sizeof(T));
        std::copy(values.begin(), values.end(), buffer.data<T>());
        return buffer;
    };
    const mtl::Buffer vectors = upload(p.edge_vectors);
    const mtl::Buffer senders = upload(p.senders);
    const mtl::Buffer offsets = upload(p.offsets);
    const mtl::Buffer h = upload(p.h);
    const mtl::Buffer W = upload(p.W);
    const mtl::Buffer A = device.allocate(outputs*sizeof(float));
    const GpuTime gpu = time_gpu(device, [&](mtl::CommandBatch& batch) {
        batch.set_buffer(0, vectors).set_buffer(1, senders)
            .set_buffer(2, offsets).set_buffer(3, h).set_buffer(4, W)
            .set_buffer(5, A).set_value(6, cutoff)
            .dispatch_threadgroups(pipeline, {atoms}, {128});
    });
    char name[64];
    std::snprintf(name, sizeof(name), "edge N=%u E=%u", atoms, p.edges);
    report(name, "us/atom", cpu1, cpun, gpu, threads, 1e6/atoms,
        max_relative_error(expected.data(), A.data<float>(), outputs));
}

}  // namespace

int main()
{
    const mtl::Device device = mtl::Device::system_default();
    const mtl::MetalInformation& info = device.information();
    const unsigned threads = performance_cores();
    std::printf(
        "device=%s family=%s performance_cores=%u repetitions=%d (median)\n"
        "edge workload: cutoff=%.1f A, channels=%u, l_max=3, radial_basis=%u, "
        "synthetic neighbor lists\n\n",
        info.device_name.c_str(), info.gpu_family.c_str(), threads,
        repetitions, cutoff, channels, radial_basis);

    const mtl::Library library = device.compile(gpu_source);
    const mtl::Pipeline triad = device.pipeline(library, "triad");
    const mtl::Pipeline sgemm = device.pipeline(library, "sgemm");
    const mtl::Pipeline edge = device.pipeline(library, "edge_aggregate");

    benchmark_triad(device, triad, threads);
    std::printf("\n");
    benchmark_sgemm(device, sgemm, 2048);
    benchmark_sgemm(device, sgemm, 4096);
    std::printf("\n");
    for (const std::uint32_t atoms : {1'000u, 10'000u, 100'000u})
        benchmark_edge(device, edge, threads, atoms, 40);
    return 0;
}
