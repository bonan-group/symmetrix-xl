#pragma once

#include <cstdint>

namespace symmetrix::execution {

// Increment with symmetrix/jit.py whenever generated code or its consumer changes.
inline constexpr std::uint32_t required_jit_generation_version = 11;

}
