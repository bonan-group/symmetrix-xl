#pragma once

#include <cstdint>

namespace symmetrix::execution {

// Keep synchronized with symmetrix/generated_operator_version.py whenever
// generated code or its native consumer changes.
inline constexpr std::uint32_t required_jit_generation_version = 19;

}
