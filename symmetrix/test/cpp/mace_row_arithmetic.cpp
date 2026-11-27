// Run the production row functor with a host team and compare ordered sums.
#include "mace_a1_scale_divide.hpp"
#include "mace_row_gemm.hpp"

#include <array>
#include <barrier>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace row = symmetrix::optimized;

namespace {
struct Scratch {
    char* data;
    std::size_t offset = 0;
    void* get_shmem_aligned(std::size_t size, std::size_t alignment)
    {
        offset = (offset + alignment-1)/alignment*alignment;
        auto* ptr = data + offset;
        offset += size;
        return ptr;
    }
};

struct Member {
    int league, rank;
    std::barrier<>& barrier;
    mutable Scratch scratch;
    int league_rank() const { return league; }
    int team_rank() const { return rank; }
    void team_barrier() const { barrier.arrive_and_wait(); }
    Scratch& team_scratch(int) const { return scratch; }
};

template <class Kernel>
void run_team(const Kernel& kernel)
{
    alignas(16) std::array<char, Kernel::shmem_bytes> scratch{};
    std::barrier barrier(Kernel::team_size);
    std::vector<std::jthread> workers;
    for (int rank = 0; rank < Kernel::team_size; ++rank)
        workers.emplace_back([&, rank] {
            for (int league = 0; league < kernel.gb.total_tiles; ++league) {
                kernel(Member{league, rank, barrier, {scratch.data()}});
                barrier.arrive_and_wait();
            }
        });
}

std::uint32_t word(float value)
{
    return std::bit_cast<std::uint32_t>(value);
}

template <int BN, int BK, int TM, int TN, int Epilogue = row::kRowGemmPlain>
bool contraction(const char* name, int channels, int paths, bool reverse)
{
    constexpr int nodes = 23;
    constexpr int components = 4;
    const int inputs = reverse ? channels : channels*paths;
    const int outputs = reverse ? channels*paths : channels;
    const auto input_stride = static_cast<std::size_t>(components*inputs + 4);
    const auto output_stride = static_cast<std::size_t>(components*outputs + 4);
    std::vector<float> a(nodes*input_stride), b(inputs*outputs);
    std::vector<float> c(nodes*output_stride, -123.f), reference = c;
    std::vector<double> scale(nodes);
    for (std::size_t i = 0; i < a.size(); ++i)
        a[i] = static_cast<float>(static_cast<int>(i%11)-5)*0.0715f;
    for (std::size_t i = 0; i < b.size(); ++i)
        b[i] = static_cast<float>(static_cast<int>(i%19)-9)*0.0931f;
    // One dot separates fused and separately rounded products.
    a[0] = -1.f;
    a[1] = 1.f + 0x1p-23f;
    b[0] = 1.f;
    b[outputs] = 1.f - 0x1p-23f;
    for (int k = 2; k < inputs; ++k)
        b[k*outputs] = 0.f;
    for (int node = 0; node < nodes; ++node) {
        scale[node] = 1.0/(1.0 + node*0.03);
        for (int component = 0; component < components; ++component)
            for (int j = 0; j < outputs; ++j) {
                float dot = 0.f;
                for (int k = 0; k < inputs; ++k) {
                    const float value = a[node*input_stride + component*inputs + k];
                    const float operand = Epilogue == row::kRowGemmScaledLoad
                        ? static_cast<float>(static_cast<double>(value)*scale[node])
                        : value;
                    dot += operand*b[k*outputs+j];
                }
                reference[node*output_stride + component*outputs+j] = 0.f + dot;
            }
    }

    row::RowGemmBatch batch{};
    batch.A = a.data();
    batch.C = c.data();
    batch.a_ns = input_stride;
    batch.c_ns = output_stride;
    batch.num_nodes = nodes;
    batch.count = 2;
    batch.p[0] = {b.data(), 0, static_cast<std::size_t>(inputs), 0,
        static_cast<std::size_t>(outputs), inputs, outputs, 1, 0, 0};
    batch.p[1] = batch.p[0];
    batch.p[1].a_off = inputs;
    batch.p[1].c_off = outputs;
    batch.p[1].comps = components - 1;
    batch.load_scale = scale.data();
    if (!row::row_gemm_prepare<64,BN,BK>(batch))
        return false;
    run_team(row::RowGemmKernel<64,BN,BK,TM,TN,Epilogue>{batch});

    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < c.size(); ++i)
        mismatches += word(c[i]) != word(reference[i]);
    std::cout << name << " outputs=" << nodes*components*outputs
        << " bitdiff=" << mismatches << '\n';
    return mismatches == 0;
}

#if defined(__linux__)
bool wide_node_stride()
{
    constexpr std::size_t stride = (std::size_t{1} << 31) + 64;
    constexpr auto bytes = (stride + 64)*sizeof(float);
    const auto reserve = [] {
        return static_cast<float*>(mmap(nullptr, bytes, PROT_READ|PROT_WRITE,
            MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0));
    };
    auto* input = reserve();
    auto* output = reserve();
    if (input == MAP_FAILED || output == MAP_FAILED)
        return false;

    alignas(16) std::array<float,64*64> weights{};
    for (int k = 0; k < 64; ++k) {
        weights[k*64+k] = 1.f;
        input[k] = static_cast<float>(k);
        input[stride+k] = static_cast<float>(101+k);
    }
    row::RowGemmBatch batch{};
    batch.A = input;
    batch.C = output;
    batch.a_ns = batch.c_ns = stride;
    batch.num_nodes = 2;
    batch.count = 1;
    batch.p[0] = {weights.data(), 0, 64, 0, 64, 64, 64, 1, 0, 0};
    if (!row::row_gemm_prepare<64,64,32>(batch))
        return false;
    run_team(row::RowGemmKernel<64,64,32,4,4>{batch});

    int mismatches = 0;
    for (int k = 0; k < 64; ++k) {
        mismatches += word(input[k]) != word(output[k]);
        mismatches += word(input[stride+k]) != word(output[stride+k]);
    }
    munmap(input, bytes);
    munmap(output, bytes);
    std::cout << "H1 wide node stride outputs=128 bitdiff=" << mismatches << '\n';
    return mismatches == 0;
}
#endif

bool density_scale()
{
    std::mt19937 random(127);
    std::uniform_real_distribution<float> values(1.f, 2.f);
    int mismatches = 0, midpoint_misses = 0;
    const auto compare = [&](float value, double divisor) {
        const float expected = static_cast<float>(static_cast<double>(value)/divisor);
        const float got = row::a1_scale_divide(value, row::a1_scale_divisor(divisor));
        mismatches += word(got) != word(expected)
            && !(std::isnan(got) && std::isnan(expected));
        return expected;
    };
    for (int i = 0; i < 25000; ++i) {
        const auto value = std::ldexp(values(random), i%61-30);
        const auto scale = 1.0 + values(random);
        compare(value, scale);
        const auto base = std::ldexp(values(random), i%255-130);
        const double midpoint = (static_cast<double>(base)
            + std::nextafter(base, std::numeric_limits<float>::infinity()))*0.5;
        const double divisor = value/midpoint;
        const auto expected = compare(value, divisor);
        midpoint_misses += word(expected) != word(static_cast<float>(value*(1.0/divisor)));
    }
    for (const float value : {0.f, -0.f, 1.f, -1.f,
            std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::infinity()})
        for (const double divisor : {0.0, -0.0, 1.0, -1.0,
                std::numeric_limits<double>::infinity(), std::numeric_limits<double>::denorm_min()})
            compare(value, divisor);
    std::cout << "A1 density scale bitdiff=" << mismatches
        << " unguarded_midpoint_bitdiff=" << midpoint_misses << '\n';
    return mismatches == 0 && midpoint_misses > 0;
}
}

int main()
{
    bool equal = contraction<64,32,4,4>("H1 forward", 64, 1, false);
    equal &= contraction<64,32,4,4>("H1 reverse", 128, 1, true);
    equal &= contraction<128,16,8,4>("A1 forward", 128, 3, false);
    equal &= contraction<128,16,8,4,row::kRowGemmScaledLoad>("A1 forward scaled", 256, 2, false);
    equal &= contraction<128,16,8,4>("A1 reverse", 128, 3, true);
#if defined(__linux__)
    equal &= wide_node_stride();
#endif
    equal &= density_scale();
    return equal ? 0 : 1;
}
