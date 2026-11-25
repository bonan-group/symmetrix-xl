#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <string_view>

namespace symmetrix::execution {

enum class MaceKernels { standard, optimized };

inline MaceKernels parse_mace_kernels(std::string_view value)
{
    if (value == "standard")
        return MaceKernels::standard;
    if (value == "optimized")
        return MaceKernels::optimized;
    throw std::invalid_argument("kernels must be standard or optimized");
}

enum class MaceKernel : std::size_t { r1_slab, h1_gemm, a1_gemm, a1_divide };

inline constexpr std::array mace_kernel_names{
    "r1_type_map", "h1_contraction", "a1_contraction", "a1_scale_divide"};
struct MaceKernelDevice {
    MaceKernel row;
    std::string_view architecture;
    bool admitted = true;
    int subgroup_width = 0;
    std::size_t scratch_limit = 0;
};

inline constexpr std::array mace_kernel_devices{
    MaceKernelDevice{MaceKernel::r1_slab, "gfx1201"},
    MaceKernelDevice{MaceKernel::r1_slab, "sm_86"},
    MaceKernelDevice{MaceKernel::h1_gemm, "gfx1201", true, 32, 64*1024},
    MaceKernelDevice{MaceKernel::h1_gemm, "sm_86", true, 32, 48*1024},
    MaceKernelDevice{MaceKernel::a1_gemm, "gfx1201", true, 32, 64*1024},
    MaceKernelDevice{MaceKernel::a1_gemm, "sm_86", true, 32, 48*1024},
    MaceKernelDevice{MaceKernel::a1_divide, "gfx1201"},
};

inline const MaceKernelDevice* mace_kernel_device(
    MaceKernel row, std::string_view architecture)
{
    for (const auto& device : mace_kernel_devices)
        if (device.row == row && device.architecture == architecture)
            return &device;
    return nullptr;
}

struct MaceKernelDirection {
    const char* form = "not_run";
    std::uint64_t launches = 0;
};

struct MaceKernelBinding {
    MaceKernelDirection forward, reverse;
    const char* refusal = nullptr;
};

struct MaceKernelReport {
    const char* mode;
    std::string plan, architecture, m0, r0, m1;
    std::array<MaceKernelBinding, mace_kernel_names.size()> rows;

    std::vector<std::string> lines() const;
};

} // namespace symmetrix::execution
