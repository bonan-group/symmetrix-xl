#pragma once

// MSL spherical harmonics for Metal kernels. The polynomials are SpheriCart's
// hardcoded macros, and the axis order, normalization, and gradient
// projection mirror spherical_harmonic_device.hpp so Metal and Kokkos owners
// evaluate identical definitions.

#include <string>

#include "metal_sphericart_macros.hpp"

namespace symmetrix::execution::metal {

inline constexpr const char spherical_harmonics_functions_msl[] = R"MSL(
#include <metal_stdlib>

#define SYMMETRIX_SPH_IDENTITY_INDEX

constant float symmetrix_sph_mace_normalization = 3.5449077018110318f;

// normalized_spherical_harmonic_values_from_direction<LMax>.
template <int LMax>
inline void symmetrix_sph_values(const float3 direction, thread float* values)
{
    constexpr int size = (LMax + 1) * (LMax + 1);
    float x = direction.z;
    float y = direction.x;
    float z = direction.y;
    const float radius_squared = x * x + y * y + z * z;
    if (radius_squared != 0.0f) {
        const float inverse_radius = metal::rsqrt(radius_squared);
        x *= inverse_radius;
        y *= inverse_radius;
        z *= inverse_radius;
    }
    const float x2 = x * x;
    const float y2 = y * y;
    const float z2 = z * z;
    HARDCODED_SPH_MACRO(LMax, x, y, z, x2, y2, z2, values, SYMMETRIX_SPH_IDENTITY_INDEX);
    for (int index = 0; index < size; ++index)
        values[index] *= symmetrix_sph_mace_normalization;
}

// Values and the Cartesian gradients of
// normalized_spherical_harmonic_gradients_from_direction<LMax>, stored as
// gradients[component * size + index] in the evaluator's x, y, z order.
template <int LMax>
inline void symmetrix_sph_values_gradients(
    const float3 direction, const float radius,
    thread float* values, thread float* gradients)
{
    constexpr int size = (LMax + 1) * (LMax + 1);
    float x = direction.z;
    float y = direction.x;
    float z = direction.y;
    const float radius_squared = x * x + y * y + z * z;
    if (radius_squared == 0.0f) {
        for (int index = 0; index < size; ++index) {
            values[index] = index == 0
                ? 0.282094791773878f * symmetrix_sph_mace_normalization : 0.0f;
            gradients[index] = 0.0f;
            gradients[size + index] = 0.0f;
            gradients[2 * size + index] = 0.0f;
        }
        return;
    }
    const float inverse_radius = metal::rsqrt(radius_squared);
    x *= inverse_radius;
    y *= inverse_radius;
    z *= inverse_radius;
    const float x2 = x * x;
    const float y2 = y * y;
    const float z2 = z * z;
    float dx[size];
    float dy[size];
    float dz[size];
    HARDCODED_SPH_MACRO(LMax, x, y, z, x2, y2, z2, values, SYMMETRIX_SPH_IDENTITY_INDEX);
    HARDCODED_SPH_DERIVATIVE_MACRO(
        LMax, x, y, z, x2, y2, z2, values, dx, dy, dz, SYMMETRIX_SPH_IDENTITY_INDEX);
    const float scale = symmetrix_sph_mace_normalization / radius;
    for (int index = 0; index < size; ++index) {
        const float radial = dx[index] * x + dy[index] * y + dz[index] * z;
        // Shuffled (z, x, y) components map back to evaluator x, y, z.
        gradients[index] = scale * (dy[index] - y * radial) * inverse_radius;
        gradients[size + index] = scale * (dz[index] - z * radial) * inverse_radius;
        gradients[2 * size + index] = scale * (dx[index] - x * radial) * inverse_radius;
        values[index] *= symmetrix_sph_mace_normalization;
    }
}
)MSL";

// Self-contained preamble for MSL sources that call the functions above.
inline std::string spherical_harmonics_msl()
{
    return std::string(sphericart_macros_msl)+spherical_harmonics_functions_msl;
}

}  // namespace symmetrix::execution::metal
