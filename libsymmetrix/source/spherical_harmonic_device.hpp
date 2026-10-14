#pragma once

#include <stdexcept>

#include <Kokkos_Core.hpp>

// Reuse SpheriCart's public, checked low-l polynomial definitions. Symmetrix
// owns the Kokkos lifecycle, launch, normalization, and stream ordering.
#include "macros.hpp"

#define SYMMETRIX_SPH_IDENTITY_INDEX

namespace symmetrix {

inline constexpr double spherical_harmonic_mace_normalization =
    3.5449077018110318;

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
    HARDCODED_SPH_MACRO(
        LMax, x, y, z, x2, y2, z2, values,
        SYMMETRIX_SPH_IDENTITY_INDEX);
    const Precision normalization = static_cast<Precision>(
        spherical_harmonic_mace_normalization);
    for (int index = 0; index < size; ++index)
        values[index] *= normalization;
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
        spherical_harmonic_mace_normalization)/radius;
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

template<int LMax, class ExecutionSpace, class CoordinateView,
    class ValueView, class GradientView>
void launch_spherical_harmonics_device_impl(
    const ExecutionSpace& execution_space, const CoordinateView& xyz,
    const ValueView& values, const GradientView& gradients, const int samples)
{
    constexpr int size = (LMax + 1) * (LMax + 1);
    using Precision = typename ValueView::non_const_value_type;
    // A thread per sample writes size (and 3*size) consecutive outputs, so
    // neighbouring lanes are far apart in memory. Stage a team's samples in
    // scratch instead and write its contiguous output span cooperatively.
    constexpr int stride = 4*size + 1;  // values+gradients, padded for LDS banks
    constexpr int fit = 32768 / (stride * static_cast<int>(sizeof(Precision)));
    constexpr int team_samples = fit >= 64 ? 64 : fit >= 32 ? 32 : 16;
    using policy_type = Kokkos::TeamPolicy<ExecutionSpace>;
    using scratch_view = Kokkos::View<Precision*,
        typename ExecutionSpace::scratch_memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    const int teams = (samples + team_samples - 1) / team_samples;
    if (teams <= 0)
        return;
    const auto* coordinates = xyz.data();
    auto* out_values = values.data();
    auto* out_gradients = gradients.data();
    Kokkos::parallel_for(
        "Symmetrix device spherical harmonics",
        policy_type(execution_space, teams, team_samples).set_scratch_size(
            0, Kokkos::PerTeam(scratch_view::shmem_size(team_samples*stride))),
        KOKKOS_LAMBDA(const typename policy_type::member_type& team) {
            scratch_view staged(team.team_scratch(0), team_samples*stride);
            const int first_sample = team.league_rank()*team_samples;
            const int count = Kokkos::min(team_samples, samples - first_sample);
            const int local = team.team_rank();
            const std::size_t first = static_cast<std::size_t>(first_sample);
            if (local < count) {
                Precision* slot = staged.data() + local*stride;
                spherical_harmonic_device_sample<LMax>(
                    coordinates + 3*(first + local), slot, slot + size);
            }
            team.team_barrier();
            for (int k = local; k < count*size; k += team_samples)
                out_values[first*size + k] = staged((k/size)*stride + k%size);
            for (int k = local; k < 3*count*size; k += team_samples)
                out_gradients[3*first*size + k] =
                    staged((k/(3*size))*stride + size + k%(3*size));
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
