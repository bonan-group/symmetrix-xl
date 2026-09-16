#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace symmetrix::execution {

struct NvrtcInformation {
    bool available = false;
    std::string library;
    std::string reason;
    int major = 0;
    int minor = 0;
    std::vector<int> supported_architectures;
};

struct NvrtcCompilation {
    std::vector<std::uint8_t> cubin;
    std::string log;
    int major = 0;
    int minor = 0;
};

NvrtcInformation nvrtc_information();

NvrtcCompilation compile_cuda_with_nvrtc(
    std::string_view source,
    const std::vector<std::string>& options);

}  // namespace symmetrix::execution
