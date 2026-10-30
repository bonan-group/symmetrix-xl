#ifndef SYMMETRIX_PARTICLE_MESH_STRAIN_HPP
#define SYMMETRIX_PARTICLE_MESH_STRAIN_HPP

// Device-resident affine-strain reduction for isotropic reciprocal kernels.

#include <Kokkos_Core.hpp>

#include <array>
#include <cstdint>
#include <stdexcept>

#include "particle_mesh_geometry.hpp"

namespace symmetrix::particle_mesh {

template <typename Float>
[[nodiscard]] std::int64_t radial_strain_work_items(
    const ReciprocalGridDeviceView<Float>& grid) noexcept
{
    return grid.bins;
}

namespace detail {

struct RadialStrainSum {
    double value[9];
};

template <typename Float, typename EnergyView, typename RadialView, typename OutputView>
struct RadialStrainReduction {
    using value_type = RadialStrainSum;

    std::int64_t nx;
    std::int64_t ny;
    std::int64_t spectrum_extent;
    PackedMatrix3 inverse;
    Kokkos::View<const double*> norms;
    EnergyView bin_energy;
    RadialView bin_radial_derivative;
    OutputView output;

    KOKKOS_INLINE_FUNCTION
    void init(value_type& update) const
    {
        for (int component = 0; component < 9; ++component) {
            update.value[component] = 0.0;
        }
    }

    KOKKOS_INLINE_FUNCTION
    void join(value_type& destination, const value_type& source) const
    {
        for (int component = 0; component < 9; ++component) {
            destination.value[component] += source.value[component];
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::int64_t bin, value_type& update) const
    {
        double wave[3];
        wave_vector(bin, nx, ny, spectrum_extent, inverse, wave);
        const double energy = static_cast<double>(bin_energy(bin));
        const double norm = static_cast<double>(norms(bin));
        const double radial_scale = norm > 0.0
            ? -static_cast<double>(bin_radial_derivative(bin)) / norm
            : 0.0;
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                const int component = 3 * a + b;
                if (a == b) {
                    update.value[component] -= energy;
                }
                update.value[component] += radial_scale * wave[a] * wave[b];
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    void final(value_type& update) const
    {
        for (int component = 0; component < 9; ++component) {
            output(component) = update.value[component];
        }
    }
};

}  // namespace detail

// `bin_energy` is e_b and `bin_radial_derivative` is de_b/d|k|, both already
// including the adapter's Hermitian multiplicity and physical prefactors. For
// affine deformation of positions and cell together,
//
//   dE/d strain_ab = -delta_ab sum_b e_b
//                    -sum_b de_b/d|k| k_a k_b / |k|.
//
// One reduction work item owns each spectrum bin and contributes all tensor
// components, so the wave vector is decoded once per bin. The result remains in
// device memory; callers decide whether and when a terminal readback is required.
template <typename ExecutionSpace, typename Float, typename EnergyView, typename RadialView>
void reduce_radial_strain(
    const ExecutionSpace& execution_space,
    const ReciprocalGridDeviceView<Float>& grid,
    const EnergyView& bin_energy,
    const RadialView& bin_radial_derivative,
    const Kokkos::View<double*, typename ExecutionSpace::memory_space>& output)
{
    if (static_cast<std::int64_t>(bin_energy.extent(0)) != grid.bins
        || static_cast<std::int64_t>(bin_radial_derivative.extent(0)) != grid.bins) {
        throw std::invalid_argument(
            "Particle-mesh radial strain inputs must hold one value per spectrum bin.");
    }
    if (output.extent(0) != 9) {
        throw std::invalid_argument(
            "Particle-mesh radial strain output must hold a 3 x 3 tensor.");
    }
    using output_view =
        Kokkos::View<double*, typename ExecutionSpace::memory_space>;
    Kokkos::parallel_reduce(
        "symmetrix_particle_mesh_radial_strain",
        Kokkos::RangePolicy<ExecutionSpace>(
            execution_space, 0, radial_strain_work_items(grid)),
        detail::RadialStrainReduction<
            Float, EnergyView, RadialView, output_view>{
            grid.nx, grid.ny, grid.spectrum_extent, grid.cell_inverse, grid.k_norm,
            bin_energy, bin_radial_derivative, output});
}

}  // namespace symmetrix::particle_mesh

#endif
