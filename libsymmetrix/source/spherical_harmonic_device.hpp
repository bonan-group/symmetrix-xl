#pragma once

#include <stdexcept>

#include <Kokkos_Core.hpp>

// Reuse SpheriCart's public, checked low-l polynomial definitions. Symmetrix
// owns the Kokkos lifecycle, launch, normalization, and stream ordering.
#include "macros.hpp"

#define SYMMETRIX_SPH_IDENTITY_INDEX

namespace symmetrix {

// Converts the orthonormal real spherical harmonics produced by the shared
// device kernels to the unit-l=0 convention carried by the serialized model
// definitions: sqrt(4*pi).
inline constexpr double spherical_harmonic_real_normalization =
    3.5449077018110318;

// Y normalization of the direct (y-only) plan, equal to the retained plan's
// float(double(Y) * C) (normalize_Y in mace_kokkos_first_interaction.cpp) for
// every float input.  C = hi + lo with hi = C rounded down to float and
// lo = float(C - hi) > 0 (the sign of zero is kept); both are scaled by 2^64 so
// value * lo never underflows.  The product rounds once in the FMA and the
// scale back by 2^-64 is exact while the result is normal.  Nonzero inputs
// below 2^-126 and |value| >= 2^62 (unreachable: |Y| <= 1 on a unit
// direction) take the retained double product.  The values of one edge are
// tested together (one branch per edge), so the double product is not
// evaluated, nor predicated, on the common path.
inline constexpr float spherical_harmonic_mace_normalization_hi64 =
    0x1.c5bf88p+65f;
inline constexpr float spherical_harmonic_mace_normalization_lo64 =
    0x1.1b4ef6p+41f;
static_assert(
    static_cast<double>(spherical_harmonic_mace_normalization_hi64)*0x1p-64
        <= spherical_harmonic_mace_normalization
    && spherical_harmonic_mace_normalization_lo64 > 0.0f
    && spherical_harmonic_mace_normalization
        - static_cast<double>(spherical_harmonic_mace_normalization_hi64)*0x1p-64
        - static_cast<double>(spherical_harmonic_mace_normalization_lo64)*0x1p-64
        < 0x1p-46
    && spherical_harmonic_mace_normalization
        - static_cast<double>(spherical_harmonic_mace_normalization_hi64)*0x1p-64
        - static_cast<double>(spherical_harmonic_mace_normalization_lo64)*0x1p-64
        > -0x1p-46,
    "hi/lo split of the MACE harmonic normalization (C - hi - lo within 2^-46)");

KOKKOS_INLINE_FUNCTION bool spherical_harmonic_mace_normalize_in_fast_range(
    const float value)
{
    const float magnitude = Kokkos::fabs(value);
    return magnitude < 0x1p62f
        && (magnitude >= 0x1p-126f || magnitude == 0.0f);
}

KOKKOS_INLINE_FUNCTION float spherical_harmonic_mace_normalize_fast(
    const float value)
{
    return Kokkos::fma(
        value, spherical_harmonic_mace_normalization_hi64,
        value*spherical_harmonic_mace_normalization_lo64)*0x1p-64f;
}

KOKKOS_INLINE_FUNCTION float spherical_harmonic_mace_normalize_retained(
    const float value)
{
    return static_cast<float>(
        static_cast<double>(value)*spherical_harmonic_mace_normalization);
}

KOKKOS_INLINE_FUNCTION float spherical_harmonic_mace_normalize(
    const float value)
{
    return spherical_harmonic_mace_normalize_in_fast_range(value)
        ? spherical_harmonic_mace_normalize_fast(value)
        : spherical_harmonic_mace_normalize_retained(value);
}

KOKKOS_INLINE_FUNCTION double spherical_harmonic_mace_normalize(
    const double value)
{
    return value*spherical_harmonic_mace_normalization;
}

// raw: the unnormalized values in registers; values: the output, written once
// per entry (normalizing in place in the output made nvcc store the raw
// values first, doubling the global stores of the direct Y kernel).
template<int Size>
KOKKOS_INLINE_FUNCTION void spherical_harmonic_mace_normalize_values(
    const float* raw, float* values)
{
    bool fast = true;
    for (int index = 0; index < Size; ++index)
        fast = fast & spherical_harmonic_mace_normalize_in_fast_range(
            raw[index]);
    if (fast) {
        for (int index = 0; index < Size; ++index)
            values[index] = spherical_harmonic_mace_normalize_fast(
                raw[index]);
    } else {
        for (int index = 0; index < Size; ++index)
            values[index] = spherical_harmonic_mace_normalize_retained(
                raw[index]);
    }
}

template<int Size>
KOKKOS_INLINE_FUNCTION void spherical_harmonic_mace_normalize_values(
    const double* raw, double* values)
{
    for (int index = 0; index < Size; ++index)
        values[index] = raw[index]*spherical_harmonic_mace_normalization;
}

template<int LMax, class Precision>
KOKKOS_INLINE_FUNCTION void spherical_harmonic_device_sample(
    const Precision* xyz, Precision* values, Precision* gradients)
{
    constexpr int size = (LMax + 1) * (LMax + 1);
    Precision x = xyz[0];
    Precision y = xyz[1];
    Precision z = xyz[2];
    const Precision radius_squared = x*x + y*y + z*z;
    if (radius_squared == Precision(0)) {
        for (int index = 0; index < size; ++index) {
            values[index] = index == 0
                ? Precision(0.282094791773878) : Precision(0);
            gradients[index] = Precision(0);
            gradients[size + index] = Precision(0);
            gradients[2*size + index] = Precision(0);
        }
        return;
    }
    const Precision inverse_radius = Precision(1) / Kokkos::sqrt(radius_squared);
    x *= inverse_radius;
    y *= inverse_radius;
    z *= inverse_radius;
    const Precision x2 = x*x;
    const Precision y2 = y*y;
    const Precision z2 = z*z;
    Precision* dx = gradients;
    Precision* dy = gradients + size;
    Precision* dz = gradients + 2*size;
    HARDCODED_SPH_MACRO(
        LMax, x, y, z, x2, y2, z2, values, SYMMETRIX_SPH_IDENTITY_INDEX);
    HARDCODED_SPH_DERIVATIVE_MACRO(
        LMax, x, y, z, x2, y2, z2, values, dx, dy, dz,
        SYMMETRIX_SPH_IDENTITY_INDEX);
    for (int index = 0; index < size; ++index) {
        const Precision radial = dx[index]*x + dy[index]*y + dz[index]*z;
        dx[index] = (dx[index] - x*radial)*inverse_radius;
        dy[index] = (dy[index] - y*radial)*inverse_radius;
        dz[index] = (dz[index] - z*radial)*inverse_radius;
    }
}

template<int LMax, class Precision>
KOKKOS_INLINE_FUNCTION void normalized_spherical_harmonic_values_from_direction(
    const Precision* direction, Precision* values)
{
    constexpr int size = (LMax + 1) * (LMax + 1);
    Precision x = direction[2];
    Precision y = direction[0];
    Precision z = direction[1];
    const Precision radius_squared = x*x + y*y + z*z;
    if (radius_squared != Precision(0)) {
        const Precision inverse_radius =
            Precision(1)/Kokkos::sqrt(radius_squared);
        x *= inverse_radius;
        y *= inverse_radius;
        z *= inverse_radius;
    }
    const Precision x2 = x*x;
    const Precision y2 = y*y;
    const Precision z2 = z*z;
    Precision raw[size];
    Precision* raw_values = raw;   // the macro's cast() takes a plain pointer
    HARDCODED_SPH_MACRO(
        LMax, x, y, z, x2, y2, z2, raw_values,
        SYMMETRIX_SPH_IDENTITY_INDEX);
    spherical_harmonic_mace_normalize_values<size>(raw, values);
}

template<int LMax, class Precision>
KOKKOS_INLINE_FUNCTION void normalized_spherical_harmonic_gradients_from_direction(
    const Precision* direction, const Precision radius, Precision* gradients)
{
    constexpr int size = (LMax + 1) * (LMax + 1);
    Precision shuffled[3] = {direction[2], direction[0], direction[1]};
    Precision values[size];
    Precision shuffled_gradients[3*size];
    spherical_harmonic_device_sample<LMax>(
        shuffled, values, shuffled_gradients);
    const Precision scale = static_cast<Precision>(
        spherical_harmonic_real_normalization)/radius;
    for (int index = 0; index < size; ++index) {
        gradients[index] = scale*shuffled_gradients[size+index];
        gradients[size+index] = scale*shuffled_gradients[2*size+index];
        gradients[2*size+index] = scale*shuffled_gradients[index];
    }
}

template<int LMax, class ExecutionSpace, class DirectionView, class RadiusView,
    class ValueView>
void launch_spherical_harmonic_values_from_directions(
    const ExecutionSpace& execution_space, const DirectionView& directions,
    const RadiusView& radii, const double cutoff, const ValueView& values,
    const int samples)
{
    constexpr int size = (LMax + 1) * (LMax + 1);
    using Precision = typename ValueView::non_const_value_type;
    Kokkos::parallel_for(
        "Symmetrix direct spherical harmonic values",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space, 0, samples),
        KOKKOS_LAMBDA(const int sample) {
            const std::size_t sample_index = static_cast<std::size_t>(sample);
            Precision* output = values.data()
                +static_cast<std::size_t>(size)*sample_index;
            if (!(radii(sample) < cutoff)) {
                for (int index=0; index<size; ++index)
                    output[index] = Precision(0);
                return;
            }
            const std::size_t coordinate_offset = 3*sample_index;
            const Precision direction[3] = {
                static_cast<Precision>(directions(coordinate_offset)),
                static_cast<Precision>(directions(coordinate_offset+1)),
                static_cast<Precision>(directions(coordinate_offset+2))};
            normalized_spherical_harmonic_values_from_direction<LMax>(
                direction, output);
        });
}

template<class ExecutionSpace, class DirectionView, class RadiusView,
    class ValueView>
void launch_spherical_harmonic_values_from_directions(
    const ExecutionSpace& execution_space, const DirectionView& directions,
    const RadiusView& radii, const double cutoff, const ValueView& values,
    const int samples, const int l_max)
{
    switch (l_max) {
    case 0:
        return launch_spherical_harmonic_values_from_directions<0>(
            execution_space, directions, radii, cutoff, values, samples);
    case 1:
        return launch_spherical_harmonic_values_from_directions<1>(
            execution_space, directions, radii, cutoff, values, samples);
    case 2:
        return launch_spherical_harmonic_values_from_directions<2>(
            execution_space, directions, radii, cutoff, values, samples);
    case 3:
        return launch_spherical_harmonic_values_from_directions<3>(
            execution_space, directions, radii, cutoff, values, samples);
    case 4:
        return launch_spherical_harmonic_values_from_directions<4>(
            execution_space, directions, radii, cutoff, values, samples);
    case 5:
        return launch_spherical_harmonic_values_from_directions<5>(
            execution_space, directions, radii, cutoff, values, samples);
    case 6:
        return launch_spherical_harmonic_values_from_directions<6>(
            execution_space, directions, radii, cutoff, values, samples);
    default:
        throw std::invalid_argument(
            "Spherical harmonic values support l_max values from 0 through 6.");
    }
}

template<int LMax, class ExecutionSpace, class CoordinateView,
    class ValueView, class GradientView>
void launch_spherical_harmonics_device_impl(
    const ExecutionSpace& execution_space, const CoordinateView& xyz,
    const ValueView& values, const GradientView& gradients, const int samples)
{
    constexpr int size = (LMax + 1) * (LMax + 1);
    Kokkos::parallel_for(
        "Symmetrix device spherical harmonics",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space, 0, samples),
        KOKKOS_LAMBDA(const int sample) {
            const std::size_t sample_index = static_cast<std::size_t>(sample);
            spherical_harmonic_device_sample<LMax>(
                xyz.data() + 3*sample_index,
                values.data() + static_cast<std::size_t>(size)*sample_index,
                gradients.data()
                    +3*static_cast<std::size_t>(size)*sample_index);
        });
}

template<class ExecutionSpace, class CoordinateView, class ValueView,
    class GradientView>
void launch_spherical_harmonics_device(
    const ExecutionSpace& execution_space, const CoordinateView& xyz,
    const ValueView& values, const GradientView& gradients, const int samples,
    const int l_max)
{
    switch (l_max) {
    case 0:
        return launch_spherical_harmonics_device_impl<0>(
            execution_space, xyz, values, gradients, samples);
    case 1:
        return launch_spherical_harmonics_device_impl<1>(
            execution_space, xyz, values, gradients, samples);
    case 2:
        return launch_spherical_harmonics_device_impl<2>(
            execution_space, xyz, values, gradients, samples);
    case 3:
        return launch_spherical_harmonics_device_impl<3>(
            execution_space, xyz, values, gradients, samples);
    case 4:
        return launch_spherical_harmonics_device_impl<4>(
            execution_space, xyz, values, gradients, samples);
    case 5:
        return launch_spherical_harmonics_device_impl<5>(
            execution_space, xyz, values, gradients, samples);
    case 6:
        return launch_spherical_harmonics_device_impl<6>(
            execution_space, xyz, values, gradients, samples);
    default:
        throw std::invalid_argument(
            "HIP spherical harmonics support l_max values from 0 through 6.");
    }
}

} // namespace symmetrix

#undef SYMMETRIX_SPH_IDENTITY_INDEX
