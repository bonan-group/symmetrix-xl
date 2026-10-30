#ifndef SYMMETRIX_PARTICLE_MESH_GEOMETRY_HPP
#define SYMMETRIX_PARTICLE_MESH_GEOMETRY_HPP

// Physics-neutral reciprocal-grid geometry for real-to-complex particle meshes.

#include <Kokkos_Core.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace symmetrix::particle_mesh {

inline constexpr double pi = 3.14159265358979323846;
// This remains a divisor guard. For supported spline orders, the continuous
// symbol of every stored bin stays well above the floor.
inline constexpr double deconvolution_floor = 1e-10;

namespace bspline {

inline constexpr int max_order = 6;

template <typename Float>
KOKKOS_INLINE_FUNCTION Float cardinal(int order, Float x)
{
    Float basis[max_order + 1];
    for (int index = 0; index <= max_order; ++index) {
        basis[index] = Float(0);
    }
    for (int index = 0; index < order; ++index) {
        const Float lower = static_cast<Float>(index);
        basis[index] = (x >= lower && x < lower + Float(1)) ? Float(1) : Float(0);
    }
    for (int degree = 1; degree < order; ++degree) {
        const Float scale = Float(1) / static_cast<Float>(degree);
        for (int index = 0; index + degree < order; ++index) {
            const Float left = (x - static_cast<Float>(index)) * scale * basis[index];
            const Float right =
                (static_cast<Float>(index + degree + 1) - x) * scale * basis[index + 1];
            basis[index] = left + right;
        }
    }
    return basis[0];
}

template <typename Float>
KOKKOS_INLINE_FUNCTION Float cardinal_derivative(int order, Float x)
{
    return cardinal<Float>(order - 1, x) - cardinal<Float>(order - 1, x - Float(1));
}

}  // namespace bspline

KOKKOS_INLINE_FUNCTION std::int64_t wrap_node(
    std::int64_t value, std::int64_t extent) noexcept
{
    const std::int64_t remainder = value % extent;
    return remainder < 0 ? remainder + extent : remainder;
}

KOKKOS_INLINE_FUNCTION std::int64_t signed_miller(
    std::int64_t index, std::int64_t extent) noexcept
{
    return index <= extent / 2 ? index : index - extent;
}

KOKKOS_INLINE_FUNCTION double hermitian_multiplicity(
    std::int64_t k_index, std::int64_t nz) noexcept
{
    if (k_index == 0) {
        return 1.0;
    }
    if (nz % 2 == 0 && 2 * k_index == nz) {
        return 1.0;
    }
    return 2.0;
}

// Trivially copyable row-major matrix for extended device lambdas.  Capturing
// std::array directly can make NVCC instantiate host-only accessors.
struct PackedMatrix3 {
    double values[9] = {};

    KOKKOS_INLINE_FUNCTION double operator()(
        const std::int64_t row, const std::int64_t column) const noexcept
    {
        return values[3 * row + column];
    }
};

[[nodiscard]] inline PackedMatrix3 pack_matrix(
    const std::array<double, 9>& source) noexcept
{
    PackedMatrix3 packed;
    for (std::size_t index = 0; index < source.size(); ++index) {
        packed.values[index] = source[index];
    }
    return packed;
}

KOKKOS_INLINE_FUNCTION void wave_vector(
    std::int64_t bin,
    std::int64_t nx,
    std::int64_t ny,
    std::int64_t nzh,
    const PackedMatrix3& inverse,
    double (&wave)[3]) noexcept
{
    const std::int64_t i = bin / (ny * nzh);
    const std::int64_t rest = bin % (ny * nzh);
    const std::int64_t j = rest / nzh;
    const std::int64_t k = rest % nzh;
    const std::int64_t miller[3] = {signed_miller(i, nx), signed_miller(j, ny), k};
    for (std::int64_t a = 0; a < 3; ++a) {
        double component = 0.0;
        for (std::int64_t b = 0; b < 3; ++b) {
            component += inverse(a, b) * static_cast<double>(miller[b]);
        }
        wave[a] = 2.0 * pi * component;
    }
}

template <typename Float>
struct ReciprocalGridDeviceView {
    // Mesh geometry and the row-convention reciprocal basis travel with the
    // device arrays so every physics adapter decodes bins identically.
    std::int64_t nx = 0;
    std::int64_t ny = 0;
    std::int64_t nz = 0;
    std::int64_t spectrum_extent = 0;
    std::int64_t points = 0;
    std::int64_t bins = 0;
    PackedMatrix3 cell_inverse{};
    double volume = 0.0;
    Kokkos::View<double*> k_norm;
    Kokkos::View<Float*> deconvolution;
    Kokkos::View<Float*> hermitian_weight;
};

template <typename ExecutionSpace, typename Float>
void prepare_reciprocal_grid(
    const ExecutionSpace& execution_space,
    const ReciprocalGridDeviceView<Float>& grid,
    const std::int64_t spline_order)
{
    const double order = static_cast<double>(spline_order);
    Kokkos::parallel_for(
        "symmetrix_particle_mesh_reciprocal_grid",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space, 0, grid.bins),
        KOKKOS_LAMBDA(const std::int64_t bin) {
            double wave[3];
            wave_vector(
                bin, grid.nx, grid.ny, grid.spectrum_extent, grid.cell_inverse, wave);
            double norm = 0.0;
            for (const double component : wave) {
                norm += component * component;
            }
            grid.k_norm(bin) = Kokkos::sqrt(norm);

            const std::int64_t i = bin / (grid.ny * grid.spectrum_extent);
            const std::int64_t rest = bin % (grid.ny * grid.spectrum_extent);
            const std::int64_t j = rest / grid.spectrum_extent;
            const std::int64_t k = rest % grid.spectrum_extent;
            const std::int64_t miller[3] = {
                signed_miller(i, grid.nx), signed_miller(j, grid.ny), k};
            const std::int64_t extents[3] = {grid.nx, grid.ny, grid.nz};
            double modulus = 1.0;
            for (std::int64_t axis = 0; axis < 3; ++axis) {
                const double ratio = static_cast<double>(miller[axis])
                    / static_cast<double>(extents[axis]);
                const double sinc = ratio == 0.0
                    ? 1.0
                    : Kokkos::sin(pi * ratio) / (pi * ratio);
                modulus *= Kokkos::pow(Kokkos::abs(sinc), order);
            }
            const double divisor = modulus > deconvolution_floor
                ? modulus
                : deconvolution_floor;
            grid.deconvolution(bin) = static_cast<Float>(divisor);
            grid.hermitian_weight(bin) =
                static_cast<Float>(hermitian_multiplicity(k, grid.nz));
        });
}

}  // namespace symmetrix::particle_mesh

#endif
