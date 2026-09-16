#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace symmetrix::execution {

struct HiprtcInformation {
    bool available = false;
    std::string library;
    std::string reason;
    int major = 0;
    int minor = 0;
};

struct HiprtcCompilation {
    std::vector<std::uint8_t> code;
    std::string log;
    int major = 0;
    int minor = 0;
};

HiprtcInformation hiprtc_information();

HiprtcCompilation compile_hip_with_hiprtc(
    std::string_view source,
    const std::vector<std::string>& options);

}  // namespace symmetrix::execution
