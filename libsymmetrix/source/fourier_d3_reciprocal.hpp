#ifndef SYMMETRIX_FOURIER_D3_RECIPROCAL_HPP
#define SYMMETRIX_FOURIER_D3_RECIPROCAL_HPP

// Private particle-mesh forward path and reciprocal contraction of the
// reciprocal-space FourierD3 correction and its mesh state.
//
// This layer owns the forward evaluation and nothing else:
//
//   * the fractional wrap and the cardinal B-spline assignment of
//     specification section 5.1;
//   * the continuous-symbol deconvolution of section 5.2;
//   * the batched real-to-complex transform of section 5.4, executed only
//     through the FFT plan owner and with no normalization on either side;
//   * the Becke-Johnson reciprocal contraction of sections 4.4 and 5.3,
//     including the k = 0 term and the Hermitian multiplicity of the stored
//     half spectrum;
//   * the self term of section 4.5, added exactly once and outside every
//     species-pair and transform loop;
//   * the unscaled inverse transform that materializes dE/dm for the reverse chain.
//
// The mesh is a semi-analytic device.  Every quantity here is a discrete sum on
// a finite mesh, and the only use of the continuous symbol is as the
// deconvolution divisor recorded in the geometry report; there is no real-space
// dispersion split and no fallback path.
//
// The reverse chain owns forces, stress, and per-atom energies; nothing in this header
// publishes an energy-only state or a public evaluator.  The whole translation
// unit exists only when the FFT build capability `SYMMETRIX_FOURIER_D3_ENABLED` is
// set and a KokkosFFT provider was linked.

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "fourier_d3_storage.hpp"
#include "fourier_d3_timing.hpp"
#include "particle_mesh_transport.hpp"
#include "tools_kokkos.hpp"

namespace symmetrix::fourier_d3 {

namespace bspline = particle_mesh::bspline;

// ---------------------------------------------------------------------------
// Specification constants
// ---------------------------------------------------------------------------

// Specification section 4.4: the series form is used below this argument of the
// BJ shape functions, and it is the normative crossover.
inline constexpr double bj_series_cutoff = 0.1;

// Specification section 8: the small-argument guard that keeps the reciprocal
// kernel finite at |k| = 0 without changing any other bin.
inline constexpr double bj_k_norm_guard = 1e-12;

// ---------------------------------------------------------------------------
// Becke-Johnson reciprocal kernel (specification section 4.4)
// ---------------------------------------------------------------------------

namespace bj {

inline constexpr double pi = 3.14159265358979323846;

// Specification section 4.4 series coefficients of the R6 and R8 shape
// functions, evaluated as scalar Horner expressions.  Namespace-scope
// constexpr arrays are host symbols under NVCC and cannot be read from these
// device functions; spelling the same coefficients locally keeps the formula
// portable without introducing device-global state.
template <typename Float>
KOKKOS_INLINE_FUNCTION Float r6_series_value(Float x)
{
    return ((((((Float(1.0 / 5040.0) * x) * x - Float(1.0 / 60.0)) * x
                   + Float(0.125))
                  * x
              - Float(1.0 / 3.0))
             * x)
            * x
        + Float(1));
}

template <typename Float>
KOKKOS_INLINE_FUNCTION Float r8_series_value(Float x)
{
    return ((((((Float(-0.00025923868350721731) * x + Float(1.0 / 360.0)) * x
                    - Float(0.010888024707303151))
                   * x)
                  * x
              + Float(0.090199350024366076))
             * x)
            * x
        - Float(0.54119610014619668));
}

template <typename Float>
KOKKOS_INLINE_FUNCTION Float r6_series_derivative(Float x)
{
    return (((((Float(6.0 / 5040.0) * x) * x - Float(4.0 / 60.0)) * x
                  + Float(3.0 * 0.125))
                 * x
             - Float(2.0 / 3.0))
        * x);
}

template <typename Float>
KOKKOS_INLINE_FUNCTION Float r8_series_derivative(Float x)
{
    return (((((Float(-6.0 * 0.00025923868350721731) * x + Float(5.0 / 360.0)) * x
                   - Float(4.0 * 0.010888024707303151))
                  * x)
                 * x
             + Float(2.0 * 0.090199350024366076))
        * x);
}

// G6 and G8 of specification section 4.4: the closed form above the normative
// crossover and the truncated power series below it, where the closed form
// divides two quantities that both vanish.
template <typename Float>
KOKKOS_INLINE_FUNCTION void shape_functions(Float x, Float& g6, Float& g8)
{
    if (x < static_cast<Float>(bj_series_cutoff)) {
        g6 = r6_series_value(x);
        g8 = r8_series_value(x);
        return;
    }
    const Float root3 = Kokkos::sqrt(Float(3));
    const Float n6 = Kokkos::exp(-x)
        - Float(2) * Kokkos::exp(-x / Float(2))
            * Kokkos::cos(static_cast<Float>(pi / 3.0) + x * root3 / Float(2));
    const Float sin8 = Kokkos::sin(static_cast<Float>(pi / 8.0));
    const Float cos8 = Kokkos::cos(static_cast<Float>(pi / 8.0));
    const Float n8 =
        Kokkos::exp(-x * sin8) * Kokkos::cos(static_cast<Float>(pi / 4.0) + x * cos8)
        + Kokkos::exp(-x * cos8) * Kokkos::cos(static_cast<Float>(3.0 * pi / 4.0) + x * sin8);
    g6 = n6 / x;
    g8 = n8 / x;
}

// dG6/dx and dG8/dx on the same branch as `shape_functions`.
template <typename Float>
KOKKOS_INLINE_FUNCTION void shape_derivatives(Float x, Float& d6, Float& d8)
{
    if (x < static_cast<Float>(bj_series_cutoff)) {
        d6 = r6_series_derivative(x);
        d8 = r8_series_derivative(x);
        return;
    }
    const Float root3 = Kokkos::sqrt(Float(3));
    const Float phase6 = static_cast<Float>(pi / 3.0) + x * root3 / Float(2);
    const Float cos6 = Kokkos::cos(phase6);
    const Float sin6 = Kokkos::sin(phase6);
    const Float n6 = Kokkos::exp(-x) - Float(2) * Kokkos::exp(-x / Float(2)) * cos6;
    const Float dn6 = -Kokkos::exp(-x) + Kokkos::exp(-x / Float(2)) * cos6
        + Kokkos::exp(-x / Float(2)) * sin6 * root3;

    const Float sin8 = Kokkos::sin(static_cast<Float>(pi / 8.0));
    const Float cos8 = Kokkos::cos(static_cast<Float>(pi / 8.0));
    const Float phase8a = static_cast<Float>(pi / 4.0) + x * cos8;
    const Float phase8b = static_cast<Float>(3.0 * pi / 4.0) + x * sin8;
    const Float n8 = Kokkos::exp(-x * sin8) * Kokkos::cos(phase8a)
        + Kokkos::exp(-x * cos8) * Kokkos::cos(phase8b);
    const Float dn8 = -sin8 * Kokkos::exp(-x * sin8) * Kokkos::cos(phase8a)
        - cos8 * Kokkos::exp(-x * sin8) * Kokkos::sin(phase8a)
        - cos8 * Kokkos::exp(-x * cos8) * Kokkos::cos(phase8b)
        - sin8 * Kokkos::exp(-x * cos8) * Kokkos::sin(phase8b);

    d6 = dn6 / x - n6 / (x * x);
    d8 = dn8 / x - n8 / (x * x);
}

// phi_hat_XY(|k|) and d phi_hat_XY / d|k| of specification section 4.4.  The
// radial derivative carries one factor R0 because the shape functions are
// evaluated at x = |k| R0.  R0, the q product, and the damping coefficients
// arrive already unit converted.
template <typename Float>
KOKKOS_INLINE_FUNCTION void pair_kernel(
    Float k_norm,
    Float r0,
    Float q_product,
    Float s6,
    Float s8,
    Float& value,
    Float& derivative)
{
    const Float pi_squared = static_cast<Float>(pi) * static_cast<Float>(pi);
    const Float r0_squared = r0 * r0;
    const Float r0_cubed = r0_squared * r0;
    const Float r0_fifth = r0_cubed * r0_squared;
    const Float prefactor6 = Float(2) * pi_squared / (Float(3) * r0_cubed);
    const Float prefactor8 = -pi_squared / r0_fifth;

    Float g6 = Float(0);
    Float g8 = Float(0);
    shape_functions(k_norm * r0, g6, g8);
    value = s6 * prefactor6 * g6 + Float(3) * s8 * q_product * prefactor8 * g8;

    Float dg6 = Float(0);
    Float dg8 = Float(0);
    shape_derivatives(k_norm * r0, dg6, dg8);
    derivative = r0
        * (s6 * prefactor6 * dg6 + Float(3) * s8 * q_product * prefactor8 * dg8);
}

// phi_XY(0) of specification section 3.1: the r = 0 value of the damped pair
// potential, which is the self term's kernel.
template <typename Float>
KOKKOS_INLINE_FUNCTION Float zero_distance_potential(
    Float r0, Float q_product, Float s6, Float s8)
{
    const Float r0_squared = r0 * r0;
    const Float r0_6 = r0_squared * r0_squared * r0_squared;
    const Float r0_8 = r0_6 * r0_squared;
    return s6 / r0_6 + Float(3) * s8 * q_product / r0_8;
}

}  // namespace bj

// ---------------------------------------------------------------------------
// Reports
// ---------------------------------------------------------------------------

struct ReciprocalGeometry {
    std::string deconvolution;
    std::string forward_normalization;
    std::string inverse_normalization;
    std::string hermitian_axis;
    std::string mesh;
    std::int64_t nx = 0;
    std::int64_t ny = 0;
    std::int64_t nz = 0;
    std::int64_t spectrum_extent = 0;
    std::int64_t mesh_points = 0;
    std::int64_t spectrum_bins = 0;
    std::int64_t species_pairs = 0;
    std::int64_t stencil_points = 0;
    std::int64_t active_rank_chunk = 0;
    std::int64_t rank_begin = 0;
    std::int64_t rank_total = 0;
    std::int64_t num_nodes = 0;
    std::int64_t num_species = 0;
    std::int64_t spline_order = 0;
};

struct ReciprocalUsage {
    std::vector<particle_mesh::BufferUsage> buffers;
    std::uint64_t active_bytes = 0;
    std::uint64_t capacity_bytes = 0;
    // Bytes the held plans retain through their mesh allocation tokens.  The
    // evaluator itemizes a retained token separately only after storage has moved
    // to a newer generation; while both share one token, `mesh` counts it once.
    std::uint64_t plan_retained_bytes = 0;
    std::uint64_t reservations = 0;
    std::uint64_t allocations = 0;
    particle_mesh::MeshStorageUsage mesh;
    KernelDomainUsage kernels;
    std::array<particle_mesh::FftPlanUsage, particle_mesh::plan_role_count> plans;
};

struct ReciprocalOutcome {
    double reciprocal_energy = 0.0;
    double self_energy = 0.0;
    double energy = 0.0;
};

// One probe of the BJ kernel and its radial derivative, for the private
// kernel-value seam.
struct ReciprocalKernelProbe {
    double value = 0.0;
    double derivative = 0.0;
};

template <typename Float>
struct StencilDiagnostic {
    std::vector<std::int32_t> base;
    std::vector<Float> weights;
    std::vector<Float> slopes;
    bool host_fallback = false;
    particle_mesh::StencilPreparationPath preparation =
        particle_mesh::StencilPreparationPath::device_non_boundary;
};

// ---------------------------------------------------------------------------
// Forward path and reciprocal contraction
// ---------------------------------------------------------------------------

// The reverse orchestrator composes this class and the
// coordination workspace.  It is declared here so the friendship below can name
// it; it is defined in `fourier_d3_reverse.hpp`.
template <typename Float>
class ReciprocalReverse;

// One particle mesh, one rank chunk, one mesh geometry.
//
// The stages are the pipeline, in order, and each one refuses to run without
// what it reads:
//
//   reserve             shapes, bounds, and 64-bit products, all-or-nothing
//   set_cell            the lattice, its determinant, and its inverse
//   set_species         atomic numbers and their r2r4 factors
//   set_atoms           positions, and with them the assignment stencil
//   set_damping         BJ damping, already unit converted
//   set_rank            the rank window and its signed eigenvalues
//   set_coefficients    c_il, one column per rank slot in the window
//
//   assign(slot)        spread one species channel onto the real mesh
//   set_field(values)   load the real mesh directly (diagnostics)
//   transform_forward   unnormalized real-to-complex transform of the mesh
//   deconvolve          divide the stored half spectrum by the continuous symbol
//   store_spectrum(slot) keep that species' deconvolved amplitude A_X(m)
//   contract            reciprocal energy, per-bin subtotals, and the cotangent
//   self_energy         the diagonal self term and its coefficient derivative
//   invert(slot)        unnormalized inverse transform of the cotangent
//   run                 the whole pipeline for every species, self term included
//
// `field`, `spectrum`, `structure_factors`, `potential`, `pair_kernel`,
// `pair_kernel_derivative`, `bin_energies`, and `self_coefficient_derivative`
// read the state those stages produced and throw a named error when it is
// missing rather than returning a stale or zero substitute.
//
// Every spread is a scatter of every atom onto p**3 mesh nodes with no weight
// threshold: dropping the smallest stencil weights would break the gather as the
// gradient of the spread, which the reverse chain depends on.
template <typename Float>
class ReciprocalForward {
public:
    using real_view_type =
        Kokkos::View<Float*, Kokkos::DefaultExecutionSpace::memory_space>;
    using spectrum_view_type = Kokkos::View<
        Kokkos::complex<Float>*, Kokkos::DefaultExecutionSpace::memory_space>;

    [[nodiscard]] static constexpr std::string_view precision_token() noexcept
    {
        if constexpr (std::is_same_v<Float, float>) {
            return "float32";
        } else {
            static_assert(std::is_same_v<Float, double>, "unsupported FourierD3 precision");
            return "float64";
        }
    }

    ReciprocalForward(particle_mesh::MeshLayout layout, std::int64_t channel_bound);
    ~ReciprocalForward() = default;

    ReciprocalForward(const ReciprocalForward&) = delete;
    ReciprocalForward& operator=(const ReciprocalForward&) = delete;
    ReciprocalForward(ReciprocalForward&&) = delete;
    ReciprocalForward& operator=(ReciprocalForward&&) = delete;

    // -- configuration -----------------------------------------------------

    void reserve(
        std::int64_t nx,
        std::int64_t ny,
        std::int64_t nz,
        std::int64_t num_species,
        std::int64_t num_nodes,
        std::int64_t rank_total,
        std::int64_t rank_chunk,
        std::int64_t spline_order);

    void set_cell(std::span<const double> cell);
    void set_species(std::span<const std::int32_t> species, std::span<const double> q_table);
    void set_atoms(std::span<const double> positions, std::span<const std::int32_t> numbers);
    void set_damping(double a1, double a2, double s6, double s8);
    void set_rank(std::int64_t rank_begin, std::span<const double> eigenvalues);
    void set_coefficients(std::span<const Float> coefficients);

    // -- stages ------------------------------------------------------------

    // Everything an evaluation can fail on: the configuration checks, the
    // retained assignment stencil, the mesh grid, the reciprocal kernel tables,
    // and both FFT plans.  None of it reads the separable coefficients, and none
    // of it writes an output - it fills caches only - so an evaluation that fails
    // here has changed nothing a caller can read.  `run` is `prepare` followed by
    // `execute`, and the reverse path calls the two halves separately so its own
    // coordination chain runs strictly between them.
    void prepare();
    // The kernel stages.  Every allocation and every validation has already
    // happened in `prepare`, so once `prepare` has returned an evaluation either
    // completes or was never started.
    [[nodiscard]] ReciprocalOutcome execute();

    void assign(std::int64_t species_slot);
    void set_field(std::span<const Float> values);
    void transform_forward();
    // The unnormalized inverse kernel of the same schedule: after invert words
    // the cotangent into the spectrum, this is the transform that produces
    // dE/dm for the reverse chain.
    void transform_inverse();
    void deconvolve();
    void store_spectrum(std::int64_t species_slot);
    void contract();
    void self_energy();
    void invert(std::int64_t species_slot);
    [[nodiscard]] ReciprocalOutcome run();
    // Spreads nothing: the supplied real mesh is transformed, deconvolved, and
    // stored per species, then contracted.  This is the acceptance seam for the
    // cotangent, and it is a diagnostic rather than a forward path.
    [[nodiscard]] double debug_evaluate_field(std::span<const Float> values);

    // -- readbacks ---------------------------------------------------------

    [[nodiscard]] std::vector<Float> field() const;
    [[nodiscard]] std::vector<std::complex<Float>> spectrum() const;
    [[nodiscard]] std::vector<std::complex<Float>> structure_factors() const;
    [[nodiscard]] std::vector<Float> potential(std::int64_t species_slot) const;
    [[nodiscard]] std::vector<Float> pair_kernel() const;
    [[nodiscard]] std::vector<Float> pair_kernel_derivative() const;
    [[nodiscard]] std::vector<double> bin_energies() const;
    [[nodiscard]] double reciprocal_energy() const;
    [[nodiscard]] double self_energy_value() const;
    [[nodiscard]] double energy_value() const;
    [[nodiscard]] std::vector<Float> self_coefficient_derivative() const;
    void prepare_host_stencil_baseline();
    [[nodiscard]] StencilDiagnostic<Float> stencil_diagnostic(bool host_baseline);
    [[nodiscard]] ReciprocalGeometry geometry() const;
    [[nodiscard]] ReciprocalUsage usage() const;

    // -- opt-in stage timing ----------------------------------------------

    // Disabled by default: a disabled timer records nothing, and the diagnostic
    // record then says timing was not collected rather than reporting zeros.
    void set_timing_enabled(const bool enabled) noexcept { timing_enabled_ = enabled; }
    [[nodiscard]] bool timing_enabled() const noexcept { return timing_enabled_; }
    [[nodiscard]] FourierD3StageTimings timings() const;

private:
    using complex_type = Kokkos::complex<Float>;

public:
    using value_type = Float;

private:
    // The reverse orchestrator reads the retained adjoint state - the
    // potential the inverse transform produced, the assignment stencil, the
    // coefficients, the eigenvalues, the self derivative, the mesh bins, the
    // species amplitudes, the kernel derivative table, and the owned plans - and
    // drives the stages that produce them.  Those are all internals of this
    // class by design, so the coupling is stated once, here, instead of widening
    // the forward surface with accessors nothing else would use.
    template <typename>
    friend class ReciprocalReverse;

    enum class RealContent { none, loaded, potential };

    // Everything a successful `reserve` would install, allocated and validated
    // but not yet installed.  `plan_reserve` allocates and throws; `commit`
    // neither allocates nor throws.  The split exists so a composite owner - the
    // reverse path, which composes this path with a coordination
    // workspace - can prepare both before it installs either, and therefore
    // cannot leave one grown and the other untouched when the second refuses.
    struct Reservation {
        Kokkos::View<double*> cell;
        Kokkos::View<std::int32_t*> species;
        Kokkos::View<double*> q_table;
        Kokkos::View<double*> positions;
        Kokkos::View<std::int32_t*> numbers;
        Kokkos::View<std::int32_t*> channels;
        Kokkos::View<double*> eigenvalues;
        Kokkos::View<Float*> coefficients;
        Kokkos::View<complex_type*> species_spectra;
        Kokkos::View<complex_type*> cotangent;
        Kokkos::View<double*> bin_energy;
        Kokkos::View<double*> bin_radial_derivative;
        Kokkos::View<double*> node_energy;
        Kokkos::View<Float*> self_derivative;
        Kokkos::View<Float*> zero_distance;
        std::vector<char> spectra_stored;
        typename particle_mesh::ParticleMeshWorkspace<Float>::Reservation workspace;
        std::int64_t num_species = 0;
        std::int64_t num_nodes = 0;
        std::int64_t rank_total = 0;
        std::int64_t rank_chunk = 0;
        std::int64_t spline_order = 0;
    };

    [[nodiscard]] Reservation plan_reserve(
        std::int64_t nx,
        std::int64_t ny,
        std::int64_t nz,
        std::int64_t num_species,
        std::int64_t num_nodes,
        std::int64_t rank_total,
        std::int64_t rank_chunk,
        std::int64_t spline_order) const;
    void commit(Reservation&& reservation);
    // Returned by value: the storage publishes its shape by value, and a
    // reference into a temporary would decode a destroyed geometry.
    [[nodiscard]] particle_mesh::MeshShape require_mesh(std::string_view what) const;
    void require_cell(std::string_view what) const;
    void require_species(std::string_view what) const;
    void require_atoms(std::string_view what) const;
    void require_rank(std::string_view what) const;
    void require_coefficients(std::string_view what) const;
    void require_configuration() const;
    void invalidate_stages();
    void build_stencil();
    void prepare_stencil();

public:
    // NVCC requires an enclosing member function that contains an extended
    // host/device lambda to have public access.  These remain native-internal
    // launch stages (they are not part of the Python binding), but their access
    // must be public for the CUDA compilation path.
    void prepare_grid();
    void prepare_kernels();

private:
    [[nodiscard]] std::int64_t pair_index(std::int64_t a, std::int64_t b) const noexcept;
    [[nodiscard]] bool spectra_complete() const noexcept;
    void clear_stored_spectra() noexcept;
    void require_stored_spectra() const;

    // Declared first so that Kokkos allocations are destroyed before this object
    // stops counting as a live Kokkos owner.
    KokkosLiveObjectGuard kokkos_live_object_guard_;
    particle_mesh::MeshLayout layout_ = particle_mesh::MeshLayout::right;
    std::int64_t channel_bound_ = 0;
    Kokkos::DefaultExecutionSpace execution_space_;

    // The shared workspace owns assignment, reciprocal geometry, mesh storage,
    // and FFT plans. The references preserve the private reverse seam until its
    // gather is moved into that workspace in the next extraction step.
    particle_mesh::ParticleMeshWorkspace<Float> workspace_;
    particle_mesh::MeshStorage<Float>& storage_;
    particle_mesh::FftPlanDomain<Float>& plans_;
    KernelDomain<Float> kernels_;

    particle_mesh::ReciprocalGridDeviceView<Float>& grid_;
    std::int64_t num_species_ = 0;
    std::int64_t num_nodes_ = 0;
    std::int64_t rank_total_ = 0;
    std::int64_t rank_chunk_ = 0;
    std::int64_t spline_order_ = 0;

    bool cell_set_ = false;
    std::array<double, 9> cell_{};
    double volume_ = 0.0;
    double inverse_volume_ = 0.0;
    std::array<double, 9> cell_inverse_{};

    bool species_set_ = false;
    std::vector<std::int32_t> species_numbers_;
    std::vector<double> q_table_;

    bool atoms_set_ = false;
    std::vector<double> positions_;
    bool rank_set_ = false;
    bool coefficients_set_ = false;
    bool damping_set_ = false;
    DampingCoefficients damping_{};
    std::int64_t rank_begin_ = 0;

    // Retained device buffers.  Every one of them is sized by `reserve` and
    // reused across steps; `active_` counts below are the active extents.
    Kokkos::View<double*> cell_view_;
    Kokkos::View<std::int32_t*> species_view_;
    Kokkos::View<double*> q_table_view_;
    Kokkos::View<double*> positions_view_;
    Kokkos::View<std::int32_t*> numbers_view_;
    Kokkos::View<std::int32_t*> channel_view_;
    Kokkos::View<double*> eigenvalues_view_;
    Kokkos::View<Float*> coefficients_view_;
    particle_mesh::ParticleStencil<Float>& stencil_;
    Kokkos::View<complex_type*> species_spectra_view_;
    Kokkos::View<complex_type*> cotangent_view_;
    Kokkos::View<double*> bin_energy_view_;
    Kokkos::View<double*> bin_radial_derivative_view_;
    Kokkos::View<double*> node_energy_view_;
    Kokkos::View<Float*> self_derivative_view_;
    Kokkos::View<Float*> zero_distance_view_;

    std::uint64_t reservations_ = 0;
    std::uint64_t allocations_ = 0;
    std::int64_t allocated_nodes_ = 0;
    // True once a reserve committed the buffers above; the accounting reads it
    // so an unconfigured object reports no retained memory.
    bool buffers_live_ = false;

    // Opt-in stage timing.  The three stages are the forward path's own; the
    // reverse orchestrator's stages live in its object and the evaluator's in the
    // evaluator's, so no stage is timed twice.
    bool timing_enabled_ = false;
    StageTimer forward_preparation_timer_;
    StageTimer assignment_forward_fft_timer_;
    StageTimer reciprocal_contraction_timer_;

    // Stage state.  `spectrum_valid_` says the mesh spectrum is the forward
    // transform of the current real mesh; `spectra_stored_` says which species
    // amplitudes have been kept since the last invalidation.
    RealContent real_content_ = RealContent::none;
    std::int64_t real_slot_ = -1;
    bool spectrum_valid_ = false;
    bool spectrum_deconvolved_ = false;
    std::vector<char> spectra_stored_;
    bool contracted_ = false;
    std::int64_t potential_slot_ = -1;
    bool self_valid_ = false;

    // Cache identity of the two derived tables, so a cell, mesh, order, or
    // spline change refills exactly what it invalidates.
    std::optional<std::array<double, 9>> grid_cell_;
    std::optional<particle_mesh::MeshShape> grid_mesh_;
    int grid_order_ = 0;
    std::optional<KernelKey> kernel_key_;
    std::optional<particle_mesh::MeshShape> kernel_mesh_;
    // The assignment stencil is derived from the cell inverse *and* the
    // positions, so its identity is both of them: a cell change invalidates it
    // even when the atoms are unchanged, and the positions generation moves
    // whenever set_atoms stores new coordinates.  An absent cell leaves it
    // unbuilt rather than deriving it from a zero inverse.
    std::optional<std::array<double, 9>> stencil_cell_;
    std::int64_t stencil_positions_generation_ = -1;
    std::int64_t positions_generation_ = 0;
    particle_mesh::StencilPreparationPath stencil_preparation_ =
        particle_mesh::StencilPreparationPath::device_non_boundary;

    double reciprocal_energy_ = 0.0;
    double self_energy_ = 0.0;
    double energy_ = 0.0;
};

// ---------------------------------------------------------------------------
// Kernel probe: the BJ value and radial derivative at supplied norms
// ---------------------------------------------------------------------------

template <typename Float>
[[nodiscard]] std::vector<ReciprocalKernelProbe> probe_reciprocal_kernel(
    std::span<const double> k_norms, double r0, double q_product, double s6, double s8)
{
    std::vector<ReciprocalKernelProbe> probes;
    probes.reserve(k_norms.size());
    for (const double k_norm : k_norms) {
        Float value = Float(0);
        Float derivative = Float(0);
        bj::pair_kernel<Float>(
            static_cast<Float>(k_norm), static_cast<Float>(r0),
            static_cast<Float>(q_product), static_cast<Float>(s6), static_cast<Float>(s8),
            value, derivative);
        ReciprocalKernelProbe probe;
        probe.value = static_cast<double>(value);
        probe.derivative = static_cast<double>(derivative);
        probes.push_back(probe);
    }
    return probes;
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

// Index of the unordered species pair (min, max) in the packed kernel table.
KOKKOS_INLINE_FUNCTION std::int64_t unordered_pair_index(
    std::int64_t first, std::int64_t second, std::int64_t num_species) noexcept
{
    if (first > second) {
        const std::int64_t swap = first;
        first = second;
        second = swap;
    }
    return first * num_species - first * (first - 1) / 2 + (second - first);
}


template <typename Float>
ReciprocalForward<Float>::ReciprocalForward(
    particle_mesh::MeshLayout layout, std::int64_t channel_bound)
    : layout_(layout),
      channel_bound_(detail::checked_extent(channel_bound, "FourierD3 rank channel bound")),
      workspace_(layout, channel_bound, execution_space_),
      storage_(workspace_.storage()),
      plans_(workspace_.plans()),
      grid_(workspace_.reciprocal_grid()),
      stencil_(workspace_.stencil())
{
    if (layout_ != particle_mesh::MeshLayout::right) {
        throw std::invalid_argument(
            "FourierD3 reciprocal storage is compiled for the layout_right transform axis; "
            "a layout_left plan would need an internal transpose, and none is inserted.");
    }
}

template <typename Float>
particle_mesh::MeshShape ReciprocalForward<Float>::require_mesh(std::string_view what) const
{
    const std::optional<particle_mesh::MeshShape> shape = storage_.shape();
    if (!shape) {
        throw std::runtime_error(
            "FourierD3 " + std::string(what)
            + " requires a reserved mesh; call reserve first.");
    }
    return *storage_.shape();
}

template <typename Float>
void ReciprocalForward<Float>::require_cell(std::string_view what) const
{
    if (!cell_set_) {
        throw std::runtime_error(
            "FourierD3 " + std::string(what)
            + " requires a cell matrix; call set_cell first.");
    }
}

template <typename Float>
void ReciprocalForward<Float>::require_species(std::string_view what) const
{
    if (!species_set_) {
        throw std::runtime_error(
            "FourierD3 " + std::string(what)
            + " requires species channels and their r2r4 factors; call set_species first.");
    }
}

template <typename Float>
void ReciprocalForward<Float>::require_atoms(std::string_view what) const
{
    if (!atoms_set_) {
        throw std::runtime_error(
            "FourierD3 " + std::string(what)
            + " requires atomic positions; call set_atoms first.");
    }
}

template <typename Float>
void ReciprocalForward<Float>::require_rank(std::string_view what) const
{
    if (!rank_set_) {
        throw std::runtime_error(
            "FourierD3 " + std::string(what)
            + " requires a rank window and its eigenvalues; call set_rank first.");
    }
}

template <typename Float>
void ReciprocalForward<Float>::require_coefficients(std::string_view what) const
{
    if (!coefficients_set_) {
        throw std::runtime_error(
            "FourierD3 " + std::string(what)
            + " requires separable coefficients; call set_coefficients first.");
    }
}

template <typename Float>
void ReciprocalForward<Float>::require_configuration() const
{
    static_cast<void>(require_mesh("the forward evaluation"));
    require_cell("the forward evaluation");
    require_species("the forward evaluation");
    require_atoms("the forward evaluation");
    require_rank("the forward evaluation");
    require_coefficients("the forward evaluation");
    if (!damping_set_) {
        throw std::runtime_error(
            "FourierD3 the forward evaluation requires BJ damping coefficients; call "
            "set_damping first.");
    }
}

template <typename Float>
void ReciprocalForward<Float>::invalidate_stages()
{
    clear_stored_spectra();
    real_content_ = RealContent::none;
    real_slot_ = -1;
    spectrum_valid_ = false;
    spectrum_deconvolved_ = false;
    contracted_ = false;
    potential_slot_ = -1;
    self_valid_ = false;
    energy_ = 0.0;
}

template <typename Float>
void ReciprocalForward<Float>::clear_stored_spectra() noexcept
{
    std::fill(spectra_stored_.begin(), spectra_stored_.end(), static_cast<char>(0));
    contracted_ = false;
}

template <typename Float>
bool ReciprocalForward<Float>::spectra_complete() const noexcept
{
    for (const char stored : spectra_stored_) {
        if (stored == 0) {
            return false;
        }
    }
    return !spectra_stored_.empty();
}

template <typename Float>
void ReciprocalForward<Float>::require_stored_spectra() const
{
    if (!spectra_complete()) {
        throw std::runtime_error(
            "FourierD3 contraction requires one stored spectrum per species channel; no "
            "species amplitude has been stored, so nothing is substituted for it.");
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

template <typename Float>
void ReciprocalForward<Float>::reserve(
    const std::int64_t nx,
    const std::int64_t ny,
    const std::int64_t nz,
    const std::int64_t num_species,
    const std::int64_t num_nodes,
    const std::int64_t rank_total,
    const std::int64_t rank_chunk,
    const std::int64_t spline_order)
{
    commit(plan_reserve(nx, ny, nz, num_species, num_nodes, rank_total, rank_chunk, spline_order));
}

template <typename Float>
auto ReciprocalForward<Float>::plan_reserve(
    const std::int64_t nx,
    const std::int64_t ny,
    const std::int64_t nz,
    const std::int64_t num_species,
    const std::int64_t num_nodes,
    const std::int64_t rank_total,
    const std::int64_t rank_chunk,
    const std::int64_t spline_order) const -> Reservation
{
    if (spline_order < 2 || spline_order > particle_mesh::bspline::max_order) {
        throw std::invalid_argument(
            "FourierD3 spline order must be between 2 and 6, got "
            + std::to_string(spline_order) + ".");
    }
    const std::array<std::int64_t, 3> extents{nx, ny, nz};
    for (const std::int64_t extent : extents) {
        if (extent < spline_order) {
            throw std::invalid_argument(
                "FourierD3 mesh axis " + std::to_string(extent)
                + " must hold at least " + std::to_string(spline_order)
                + " nodes: the assignment stencil is that many nodes wide at the "
                  "requested spline order.");
        }
    }
    if (num_species < 1) {
        throw std::invalid_argument(
            "FourierD3 must cover at least one species channel, got "
            + std::to_string(num_species) + ".");
    }
    if (num_nodes < 0) {
        throw std::invalid_argument(
            "FourierD3 atom count must not be negative, got "
            + std::to_string(num_nodes) + ".");
    }
    if (rank_total < 1) {
        throw std::invalid_argument(
            "FourierD3 rank must hold at least one slot, got "
            + std::to_string(rank_total) + ".");
    }
    if (rank_chunk < 1) {
        throw std::invalid_argument(
            "FourierD3 rank chunk must hold at least one slot, got "
            + std::to_string(rank_chunk) + ".");
    }
    if (rank_chunk > channel_bound_) {
        throw std::invalid_argument(
            "FourierD3 rank chunk of " + std::to_string(rank_chunk)
            + " slot(s) exceeds the configured rank chunk bound of "
            + std::to_string(channel_bound_)
            + "; the request is rejected rather than clipped.");
    }
    if (rank_chunk > rank_total) {
        throw std::invalid_argument(
            "FourierD3 rank chunk of " + std::to_string(rank_chunk)
            + " slot(s) is wider than the supplied rank of "
            + std::to_string(rank_total) + " slot(s).");
    }
    if (num_nodes > 0 && rank_chunk > std::numeric_limits<std::int64_t>::max() / num_nodes) {
        throw std::overflow_error(
            "FourierD3 coefficient table size overflows a signed 64-bit index.");
    }
    // The active geometry is checked by particle_mesh::MeshShape itself,
    // including every 64-bit product.
    const particle_mesh::MeshShape requested(nx, ny, nz, rank_chunk, layout_);
    const std::int64_t nodes = num_nodes;
    const std::int64_t bins = requested.spectrum_point_count();
    const std::int64_t species_slots = detail::checked_product(
        num_species, rank_chunk, "FourierD3 species amplitude slots");
    const std::int64_t species_elements = detail::checked_product(
        species_slots, bins, "FourierD3 species amplitude elements");
    const std::int64_t coefficient_elements =
        detail::checked_product(num_nodes, rank_chunk, "FourierD3 coefficient table");
    // The position extent is checked before it is formed rather than after it
    // has wrapped. The shared workspace validates the stencil product.
    const std::int64_t stencil_offsets =
        detail::checked_product(3, num_nodes, "FourierD3 stencil offsets");
    const std::int64_t stencil_elements = detail::checked_product(
        stencil_offsets, spline_order, "FourierD3 stencil values");

    // Every buffer count this owner allocates is checked against the flat index
    // a kernel can address before the first candidate view is built, so an
    // unaddressable request is refused rather than attempted.  The mesh counts
    // are included because this owner allocates them too, and the mesh is grown
    // last - after the views above - so without this check the refusal would
    // arrive only after those views had already been allocated.
    static_cast<void>(detail::checked_flat_index(num_species, "species"));
    static_cast<void>(detail::checked_flat_index(num_nodes, "numbers"));
    static_cast<void>(detail::checked_flat_index(stencil_offsets, "positions"));
    static_cast<void>(detail::checked_flat_index(stencil_elements, "stencil_weight"));
    static_cast<void>(detail::checked_flat_index(coefficient_elements, "coefficients"));
    static_cast<void>(detail::checked_flat_index(bins, "k_norm"));
    static_cast<void>(detail::checked_flat_index(species_elements, "species_spectra"));
    static_cast<void>(detail::checked_flat_index(requested.real_element_count(), "real_mesh"));
    static_cast<void>(detail::checked_flat_index(requested.spectrum_element_count(), "spectrum"));

    // All-or-nothing: every replacement buffer is built into the reservation
    // first, and the committed state is replaced only after all of them
    // succeeded.  A failure here leaves the object exactly as it was, with no
    // partially committed geometry and no drained plan.
    Reservation reservation;
    detail::maybe_fail_allocation("the cell matrix");
    reservation.cell = Kokkos::View<double*>("symmetrix_fourier_d3_cell", 9);
    detail::maybe_fail_allocation("the species table");
    reservation.species =
        Kokkos::View<std::int32_t*>("symmetrix_fourier_d3_species", num_species);
    detail::maybe_fail_allocation("the r2r4 table");
    reservation.q_table = Kokkos::View<double*>("symmetrix_fourier_d3_q_table", num_species);
    detail::maybe_fail_allocation("the atomic positions");
    reservation.positions =
        Kokkos::View<double*>("symmetrix_fourier_d3_positions", stencil_offsets);
    detail::maybe_fail_allocation("the atomic numbers");
    reservation.numbers =
        Kokkos::View<std::int32_t*>("symmetrix_fourier_d3_numbers", num_nodes);
    detail::maybe_fail_allocation("the channel map");
    reservation.channels =
        Kokkos::View<std::int32_t*>("symmetrix_fourier_d3_channels", num_nodes);
    detail::maybe_fail_allocation("the eigenvalue window");
    reservation.eigenvalues =
        Kokkos::View<double*>("symmetrix_fourier_d3_eigenvalues", rank_chunk);
    detail::maybe_fail_allocation("the coefficient table");
    reservation.coefficients =
        Kokkos::View<Float*>("symmetrix_fourier_d3_coefficients", coefficient_elements);
    detail::maybe_fail_allocation("the species amplitudes");
    reservation.species_spectra =
        Kokkos::View<complex_type*>("symmetrix_fourier_d3_species_spectra", species_elements);
    detail::maybe_fail_allocation("the cotangent spectrum");
    reservation.cotangent =
        Kokkos::View<complex_type*>("symmetrix_fourier_d3_cotangent", species_elements);
    detail::maybe_fail_allocation("the per-bin subtotals");
    reservation.bin_energy =
        Kokkos::View<double*>("symmetrix_fourier_d3_bin_energy", bins);
    detail::maybe_fail_allocation("the per-bin radial derivatives");
    reservation.bin_radial_derivative = Kokkos::View<double*>(
        "symmetrix_fourier_d3_bin_radial_derivative", bins);
    detail::maybe_fail_allocation("the per-node subtotals");
    reservation.node_energy =
        Kokkos::View<double*>("symmetrix_fourier_d3_node_energy", num_nodes);
    detail::maybe_fail_allocation("the self-term coefficient derivative");
    reservation.self_derivative = Kokkos::View<Float*>(
        "symmetrix_fourier_d3_self_derivative",
        detail::checked_product(num_nodes, rank_chunk, "FourierD3 self derivative"));
    detail::maybe_fail_allocation("the zero-distance potentials");
    reservation.zero_distance =
        Kokkos::View<Float*>("symmetrix_fourier_d3_zero_distance", num_species);
    reservation.spectra_stored.assign(static_cast<std::size_t>(num_species), 0);

    reservation.workspace = workspace_.plan_reserve(
        nx, ny, nz, num_nodes, rank_chunk, spline_order);
    reservation.num_species = num_species;
    reservation.num_nodes = nodes;
    reservation.rank_total = rank_total;
    reservation.rank_chunk = rank_chunk;
    reservation.spline_order = spline_order;
    return reservation;
}

template <typename Float>
void ReciprocalForward<Float>::commit(Reservation&& reservation)
{
    // Nothing below allocates or throws.
    workspace_.commit(std::move(reservation.workspace));
    cell_view_ = std::move(reservation.cell);
    species_view_ = std::move(reservation.species);
    q_table_view_ = std::move(reservation.q_table);
    positions_view_ = std::move(reservation.positions);
    numbers_view_ = std::move(reservation.numbers);
    channel_view_ = std::move(reservation.channels);
    eigenvalues_view_ = std::move(reservation.eigenvalues);
    coefficients_view_ = std::move(reservation.coefficients);
    species_spectra_view_ = std::move(reservation.species_spectra);
    cotangent_view_ = std::move(reservation.cotangent);
    bin_energy_view_ = std::move(reservation.bin_energy);
    bin_radial_derivative_view_ = std::move(reservation.bin_radial_derivative);
    node_energy_view_ = std::move(reservation.node_energy);
    self_derivative_view_ = std::move(reservation.self_derivative);
    zero_distance_view_ = std::move(reservation.zero_distance);
    num_species_ = reservation.num_species;
    num_nodes_ = reservation.num_nodes;
    rank_total_ = reservation.rank_total;
    rank_chunk_ = reservation.rank_chunk;
    spline_order_ = reservation.spline_order;
    allocated_nodes_ = num_nodes_;
    spectra_stored_ = std::move(reservation.spectra_stored);

    cell_set_ = false;
    species_set_ = false;
    atoms_set_ = false;
    positions_.clear();
    rank_set_ = false;
    coefficients_set_ = false;
    damping_set_ = false;
    grid_cell_.reset();
    grid_mesh_.reset();
    grid_order_ = 0;
    kernel_key_.reset();
    kernel_mesh_.reset();
    stencil_cell_.reset();
    stencil_positions_generation_ = -1;
    positions_generation_ = 0;
    stencil_preparation_ = particle_mesh::StencilPreparationPath::device_non_boundary;
    invalidate_stages();
    buffers_live_ = true;
    ++reservations_;
    // One Kokkos allocation per retained buffer above.
    allocations_ += 21;
}

template <typename Float>
void ReciprocalForward<Float>::set_cell(std::span<const double> cell)
{
    if (cell.size() != 9) {
        throw std::invalid_argument(
            "FourierD3 cell matrix must hold nine entries in row-major order, got "
            + std::to_string(cell.size()) + ".");
    }
    std::array<double, 9> values{};
    for (std::size_t index = 0; index < 9; ++index) {
        values[index] = cell[index];
        detail::require_finite(values[index], "cell entries must be finite");
    }
    const double determinant =
        values[0] * (values[4] * values[8] - values[5] * values[7])
        - values[1] * (values[3] * values[8] - values[5] * values[6])
        + values[2] * (values[3] * values[7] - values[4] * values[6]);
    detail::require_finite(determinant, "cell determinant must be finite");
    if (!(std::abs(determinant) > 0.0)) {
        throw std::invalid_argument(
            "FourierD3 cell matrix is singular: its determinant is zero, so the lattice "
            "has no reciprocal basis and no wave vectors exist.");
    }
    std::array<double, 9> inverse{};
    inverse[0] = (values[4] * values[8] - values[5] * values[7]) / determinant;
    inverse[1] = (values[2] * values[7] - values[1] * values[8]) / determinant;
    inverse[2] = (values[1] * values[5] - values[2] * values[4]) / determinant;
    inverse[3] = (values[5] * values[6] - values[3] * values[8]) / determinant;
    inverse[4] = (values[0] * values[8] - values[2] * values[6]) / determinant;
    inverse[5] = (values[2] * values[3] - values[0] * values[5]) / determinant;
    inverse[6] = (values[3] * values[7] - values[4] * values[6]) / determinant;
    inverse[7] = (values[1] * values[6] - values[0] * values[7]) / determinant;
    inverse[8] = (values[0] * values[4] - values[1] * values[3]) / determinant;
    for (const double entry : inverse) {
        detail::require_finite(entry, "cell inverse entries must be finite");
    }

    Kokkos::View<double*, Kokkos::HostSpace> host("symmetrix_fourier_d3_cell_host", 9);
    for (std::size_t index = 0; index < 9; ++index) {
        host(index) = values[index];
    }
    Kokkos::deep_copy(execution_space_, cell_view_, host);
    execution_space_.fence("FourierD3 cell copy completion");
    cell_ = values;
    cell_inverse_ = inverse;
    volume_ = std::abs(determinant);
    inverse_volume_ = 1.0 / volume_;
    workspace_.set_cell(inverse, volume_);
    cell_set_ = true;
    grid_cell_.reset();
    // The assignment stencil is derived from the cell inverse, so a new cell
    // invalidates it.  It is rebuilt on the next spread, once both the cell and
    // the atoms are present, rather than here: set_cell must not require atoms
    // and must not derive anything from a half-configured object.
    stencil_cell_.reset();
    invalidate_stages();
}

template <typename Float>
void ReciprocalForward<Float>::set_species(
    std::span<const std::int32_t> species, std::span<const double> q_table)
{
    if (species.size() != static_cast<std::size_t>(num_species_)) {
        throw std::invalid_argument(
            "FourierD3 species list must hold exactly the reserved number of species "
            "channels, got " + std::to_string(species.size()) + " for "
            + std::to_string(num_species_) + ".");
    }
    if (q_table.size() != static_cast<std::size_t>(num_species_)) {
        throw std::invalid_argument(
            "FourierD3 q table must hold one r2r4 value per species channel, got "
            + std::to_string(q_table.size()) + " for " + std::to_string(num_species_) + ".");
    }
    std::vector<std::int32_t> numbers(species.begin(), species.end());
    std::vector<double> factors(q_table.begin(), q_table.end());
    for (std::size_t index = 0; index < numbers.size(); ++index) {
        if (numbers[index] < 1) {
            throw std::invalid_argument(
                "FourierD3 species atomic numbers must be real atomic numbers; got "
                + std::to_string(numbers[index]) + " at channel "
                + std::to_string(index) + ".");
        }
        if (index != 0 && numbers[index] <= numbers[index - 1]) {
            throw std::invalid_argument(
                "FourierD3 species must be unique and sorted ascending; got the "
                "repeated or out-of-order entry "
                + std::to_string(numbers[index]) + " at channel "
                + std::to_string(index) + ".");
        }
        detail::require_finite(factors[index], "q table entries must be finite");
        if (!(factors[index] > 0.0)) {
            throw std::invalid_argument(
                "FourierD3 q table entries must be positive r2r4 values; got "
                + detail::format_scalar(factors[index]) + " at channel "
                + std::to_string(index) + ".");
        }
    }

    Kokkos::View<std::int32_t*, Kokkos::HostSpace> host_species(
        "symmetrix_fourier_d3_species_host", num_species_);
    Kokkos::View<double*, Kokkos::HostSpace> host_q(
        "symmetrix_fourier_d3_q_table_host", num_species_);
    for (std::int64_t index = 0; index < num_species_; ++index) {
        host_species(index) = numbers[static_cast<std::size_t>(index)];
        host_q(index) = factors[static_cast<std::size_t>(index)];
    }
    Kokkos::deep_copy(execution_space_, species_view_, host_species);
    Kokkos::deep_copy(execution_space_, q_table_view_, host_q);
    execution_space_.fence("FourierD3 species copy completion");
    species_numbers_ = std::move(numbers);
    q_table_ = std::move(factors);
    species_set_ = true;
    kernel_key_.reset();
    invalidate_stages();
}

template <typename Float>
void ReciprocalForward<Float>::set_atoms(
    std::span<const double> positions, std::span<const std::int32_t> numbers)
{
    const std::int64_t position_elements =
        detail::checked_product(3, num_nodes_, "FourierD3 atom positions");
    if (positions.size() != static_cast<std::size_t>(position_elements)) {
        throw std::invalid_argument(
            "FourierD3 atom positions must hold three coordinates per atom, got "
            + std::to_string(positions.size()) + " for " + std::to_string(num_nodes_)
            + " atom(s).");
    }
    if (numbers.size() != static_cast<std::size_t>(num_nodes_)) {
        throw std::invalid_argument(
            "FourierD3 atomic numbers must hold one entry per atom, got "
            + std::to_string(numbers.size()) + " for " + std::to_string(num_nodes_)
            + " atom(s).");
    }
    if (!species_set_) {
        throw std::runtime_error(
            "FourierD3 atomic positions require the species channels they are grouped by; "
            "call set_species first.");
    }
    std::vector<double> coordinates(positions.begin(), positions.end());
    for (const double value : coordinates) {
        detail::require_finite(value, "atom positions must be finite");
    }
    std::vector<std::int32_t> atom_numbers(numbers.begin(), numbers.end());
    std::vector<std::int32_t> channels(atom_numbers.size(), -1);
    for (std::size_t index = 0; index < atom_numbers.size(); ++index) {
        if (atom_numbers[index] < 0) {
            throw std::invalid_argument(
                "FourierD3 atomic numbers must not be negative; got "
                + std::to_string(atom_numbers[index]) + " at atom "
                + std::to_string(index) + ".");
        }
        const auto found = std::lower_bound(
            species_numbers_.begin(), species_numbers_.end(), atom_numbers[index]);
        if (found != species_numbers_.end() && *found == atom_numbers[index]) {
            channels[index] =
                static_cast<std::int32_t>(std::distance(species_numbers_.begin(), found));
        }
    }

    Kokkos::View<double*, Kokkos::HostSpace> host_positions(
        "symmetrix_fourier_d3_positions_host", coordinates.size());
    Kokkos::View<std::int32_t*, Kokkos::HostSpace> host_numbers(
        "symmetrix_fourier_d3_numbers_host", atom_numbers.size());
    Kokkos::View<std::int32_t*, Kokkos::HostSpace> host_channels(
        "symmetrix_fourier_d3_channels_host", channels.size());
    for (std::size_t index = 0; index < coordinates.size(); ++index) {
        host_positions(index) = coordinates[index];
    }
    for (std::size_t index = 0; index < atom_numbers.size(); ++index) {
        host_numbers(index) = atom_numbers[index];
        host_channels(index) = channels[index];
    }
    Kokkos::deep_copy(execution_space_, positions_view_, host_positions);
    Kokkos::deep_copy(execution_space_, numbers_view_, host_numbers);
    Kokkos::deep_copy(execution_space_, channel_view_, host_channels);
    execution_space_.fence("FourierD3 atom copy completion");
    positions_ = std::move(coordinates);
    // The stencil is derived from these positions *and* the cell inverse, so it
    // is prepared on the next spread: set_atoms may legitimately run before
    // set_cell, and a stencil built from a zero inverse would silently place
    // every atom at the origin.
    ++positions_generation_;
    atoms_set_ = true;
    invalidate_stages();
}

template <typename Float>
void ReciprocalForward<Float>::set_damping(double a1, double a2, double s6, double s8)
{
    detail::require_finite(a1, "damping a1 must be finite");
    detail::require_finite(a2, "damping a2 must be finite");
    detail::require_finite(s6, "damping s6 must be finite");
    detail::require_finite(s8, "damping s8 must be finite");
    if (!(a1 >= 0.0)) {
        throw std::invalid_argument(
            "FourierD3 BJ damping a1 must not be negative, got "
            + detail::format_scalar(a1) + ".");
    }
    if (!(a2 > 0.0)) {
        throw std::invalid_argument(
            "FourierD3 BJ damping a2 must be positive, got " + detail::format_scalar(a2)
            + "; a zero damping radius leaves the reciprocal kernel undefined at the "
              "origin.");
    }
    if (!(s6 >= 0.0) || !(s8 >= 0.0)) {
        throw std::invalid_argument(
            "FourierD3 BJ damping orders s6 and s8 must not be negative, got "
            + detail::format_scalar(s6) + " and " + detail::format_scalar(s8) + ".");
    }
    damping_ = DampingCoefficients{a1, a2, s6, s8};
    damping_set_ = true;
    invalidate_stages();
}

template <typename Float>
void ReciprocalForward<Float>::set_rank(
    std::int64_t rank_begin, std::span<const double> eigenvalues)
{
    if (rank_begin < 0) {
        throw std::invalid_argument(
            "FourierD3 rank window must start at a non-negative slot, got "
            + std::to_string(rank_begin) + ".");
    }
    if (rank_begin > rank_total_ || rank_chunk_ > rank_total_ - rank_begin) {
        throw std::invalid_argument(
            "FourierD3 rank window [" + std::to_string(rank_begin) + ", "
            + std::to_string(rank_begin + rank_chunk_)
            + ") exceeds the rank of " + std::to_string(rank_total_) + " slot(s).");
    }
    if (eigenvalues.size() != static_cast<std::size_t>(rank_chunk_)) {
        throw std::invalid_argument(
            "FourierD3 eigenvalue window must hold one value per active rank slot, got "
            + std::to_string(eigenvalues.size()) + " for " + std::to_string(rank_chunk_)
            + ".");
    }
    for (const double value : eigenvalues) {
        detail::require_finite(value, "eigenvalues must be finite");
    }
    Kokkos::View<double*, Kokkos::HostSpace> host("symmetrix_fourier_d3_eigenvalues_host", rank_chunk_);
    for (std::int64_t index = 0; index < rank_chunk_; ++index) {
        host(index) = eigenvalues[static_cast<std::size_t>(index)];
    }
    Kokkos::deep_copy(execution_space_, eigenvalues_view_, host);
    execution_space_.fence("FourierD3 eigenvalue copy completion");
    rank_begin_ = rank_begin;
    rank_set_ = true;
    invalidate_stages();
}

template <typename Float>
void ReciprocalForward<Float>::set_coefficients(std::span<const Float> coefficients)
{
    const std::int64_t expected =
        detail::checked_product(num_nodes_, rank_chunk_, "FourierD3 coefficient table");
    if (coefficients.size() != static_cast<std::size_t>(expected)) {
        throw std::invalid_argument(
            "FourierD3 coefficient table must hold one value per atom and active rank "
            "slot, got " + std::to_string(coefficients.size()) + " for "
            + std::to_string(expected) + ".");
    }
    for (const Float value : coefficients) {
        detail::require_finite(static_cast<double>(value), "coefficients must be finite");
    }
    Kokkos::View<Float*, Kokkos::HostSpace> host(
        "symmetrix_fourier_d3_coefficients_host", static_cast<std::size_t>(expected));
    for (std::int64_t index = 0; index < expected; ++index) {
        host(index) = coefficients[static_cast<std::size_t>(index)];
    }
    Kokkos::deep_copy(execution_space_, coefficients_view_, host);
    execution_space_.fence("FourierD3 coefficient copy completion");
    coefficients_set_ = true;
    invalidate_stages();
}

template <typename Float>
void ReciprocalForward<Float>::prepare_stencil()
{
    require_cell("the assignment stencil");
    require_atoms("the assignment stencil");
    if (stencil_cell_ && *stencil_cell_ == cell_
        && stencil_positions_generation_ == positions_generation_) {
        return;
    }
    build_stencil();
    stencil_cell_ = cell_;
    stencil_positions_generation_ = positions_generation_;
}

template <typename Float>
void ReciprocalForward<Float>::build_stencil()
{
    (void)require_mesh("the assignment stencil");
    stencil_preparation_ = workspace_.prepare_stencil(positions_, positions_view_);
}

// ---------------------------------------------------------------------------
// Derived tables: the bin geometry, the kernel tables, and the kernel values
// ---------------------------------------------------------------------------

template <typename Float>
void ReciprocalForward<Float>::prepare_grid()
{
    static_cast<void>(require_mesh("the mesh geometry"));
    require_cell("the mesh geometry");
    if (grid_cell_ && *grid_cell_ == cell_ && grid_mesh_ && *grid_mesh_ == *storage_.shape()
        && grid_order_ == static_cast<int>(spline_order_)) {
        return;
    }
    workspace_.prepare_reciprocal_grid();

    grid_cell_ = cell_;
    grid_mesh_ = storage_.shape();
    grid_order_ = static_cast<int>(spline_order_);
}

template <typename Float>
void ReciprocalForward<Float>::prepare_kernels()
{
    static_cast<void>(require_mesh("the reciprocal kernel tables"));
    require_cell("the reciprocal kernel tables");
    require_species("the reciprocal kernel tables");
    if (!damping_set_) {
        throw std::runtime_error(
            "FourierD3 the reciprocal kernel tables require BJ damping coefficients; call "
            "set_damping first.");
    }
    prepare_grid();

    const KernelKey key(
        cell_, damping_, std::vector<int>(species_numbers_.begin(), species_numbers_.end()),
        static_cast<int>(spline_order_), compiled_plan_identity<Float>().backend,
        std::string(precision_token()));
    kernels_.reserve(key, *storage_.shape());

    const bool refill = !kernel_key_ || !(*kernel_key_ == key) || !kernel_mesh_
        || !(*kernel_mesh_ == *storage_.shape());
    if (!refill) {
        return;
    }

    const std::int64_t nspec = num_species_;
    const std::int64_t bins = grid_.bins;
    auto k_norm = grid_.k_norm;
    auto q_table = q_table_view_;
    auto kernel = kernels_.pair_kernel();
    auto derivative = kernels_.pair_kernel_derivative();
    auto zero_distance = zero_distance_view_;
    const double s6 = damping_.s6;
    const double s8 = damping_.s8;
    const double a1 = damping_.a1;
    const double a2 = damping_.a2;
    const std::int64_t pairs = detail::checked_product(
        nspec, nspec + 1, "FourierD3 kernel species pairs")
        / 2;
    const std::int64_t kernel_elements =
        detail::checked_product(pairs, bins, "FourierD3 kernel table elements");

    Kokkos::parallel_for(
        "symmetrix_fourier_d3_kernel_table",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            execution_space_, 0, kernel_elements),
        KOKKOS_LAMBDA(const std::int64_t element) {
            const std::int64_t row = element / bins;
            const std::int64_t bin = element % bins;
            // Decode the packed unordered pair: row = a * n - a (a - 1) / 2 + (b - a).
            std::int64_t a = 0;
            while (row >= (a + 1) * nspec - a * (a + 1) / 2) {
                ++a;
            }
            const std::int64_t b =
                a + (row - (a * nspec - a * (a - 1) / 2));
            const double q_product = q_table(a) * q_table(b);
            const double r0 = a1 * Kokkos::sqrt(3.0 * q_product) + a2;
            Float value = Float(0);
            Float slope = Float(0);
            bj::pair_kernel<Float>(
                static_cast<Float>(k_norm(bin)), static_cast<Float>(r0),
                static_cast<Float>(q_product), static_cast<Float>(s6),
                static_cast<Float>(s8), value, slope);
            kernel(element) = value;
            derivative(element) = slope;
        });
    Kokkos::parallel_for(
        "symmetrix_fourier_d3_zero_distance",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            execution_space_, 0, nspec),
        KOKKOS_LAMBDA(const std::int64_t channel) {
            const double q_product = q_table(channel) * q_table(channel);
            const double r0 = a1 * Kokkos::sqrt(3.0 * q_product) + a2;
            zero_distance(channel) = bj::zero_distance_potential<Float>(
                static_cast<Float>(r0), static_cast<Float>(q_product), static_cast<Float>(s6),
                static_cast<Float>(s8));
        });
    kernel_key_ = key;
    kernel_mesh_ = storage_.shape();
}

// ---------------------------------------------------------------------------
// Stages
// ---------------------------------------------------------------------------

template <typename Float>
void ReciprocalForward<Float>::assign(std::int64_t species_slot)
{
    require_configuration();
    prepare_stencil();
    if (species_slot < 0 || species_slot >= num_species_) {
        throw std::out_of_range(
            "FourierD3 species slot " + std::to_string(species_slot)
            + " is outside the reserved range [0," + std::to_string(num_species_) + ").");
    }
    const auto coefficients = workspace_.checked_weights_view(
        coefficients_view_, num_nodes_, rank_chunk_, "FourierD3 coefficient table");
    const auto channels = workspace_.checked_groups_view(
        channel_view_, num_nodes_, "FourierD3 channel map");
    workspace_.spread(
        coefficients, channels, static_cast<std::int32_t>(species_slot));

    real_content_ = RealContent::loaded;
    real_slot_ = species_slot;
    spectrum_valid_ = false;
    spectrum_deconvolved_ = false;
    potential_slot_ = -1;
    contracted_ = false;
    energy_ = 0.0;
}

template <typename Float>
void ReciprocalForward<Float>::set_field(std::span<const Float> values)
{
    static_cast<void>(require_mesh("the real field"));
    const std::int64_t expected =
        detail::checked_product(rank_chunk_, grid_.points, "FourierD3 real field elements");
    if (values.size() != static_cast<std::size_t>(expected)) {
        throw std::invalid_argument(
            "FourierD3 real field must hold one value per rank slot and mesh point, got "
            + std::to_string(values.size()) + " for " + std::to_string(expected) + ".");
    }
    for (const Float value : values) {
        detail::require_finite(static_cast<double>(value), "field values must be finite");
    }
    Kokkos::View<Float*, Kokkos::HostSpace> host(
        "symmetrix_fourier_d3_field_host", static_cast<std::size_t>(expected));
    for (std::int64_t index = 0; index < expected; ++index) {
        host(index) = values[static_cast<std::size_t>(index)];
    }
    auto field = workspace_.real_mesh();
    using flat_field_view = Kokkos::View<
        Float*, Kokkos::DefaultExecutionSpace::memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    Kokkos::deep_copy(execution_space_, flat_field_view(field.data(), expected), host);
    execution_space_.fence("FourierD3 field copy completion");

    real_content_ = RealContent::loaded;
    real_slot_ = -1;
    spectrum_valid_ = false;
    spectrum_deconvolved_ = false;
    potential_slot_ = -1;
    contracted_ = false;
    energy_ = 0.0;
}

template <typename Float>
void ReciprocalForward<Float>::transform_forward()
{
    static_cast<void>(require_mesh("the forward transform"));
    if (real_content_ == RealContent::none) {
        throw std::runtime_error(
            "FourierD3 the forward transform requires a real mesh; spread a species or "
            "load a field first.");
    }
    // The transform goes through the FFT plan owner: the direction, the
    // batch, and the lack of normalization are decided there and nowhere else.
    workspace_.transform_forward();
    spectrum_valid_ = true;
    spectrum_deconvolved_ = false;
    potential_slot_ = -1;
    contracted_ = false;
    energy_ = 0.0;
}

template <typename Float>
void ReciprocalForward<Float>::transform_inverse()
{
    static_cast<void>(require_mesh("the inverse transform"));
    if (!spectrum_valid_) {
        throw std::runtime_error(
            "FourierD3 the inverse transform requires a mesh spectrum; run "
            "transform_forward first.");
    }
    workspace_.transform_inverse();
    real_content_ = RealContent::loaded;
    real_slot_ = -1;
    potential_slot_ = -1;
}

template <typename Float>
void ReciprocalForward<Float>::deconvolve()
{
    static_cast<void>(require_mesh("the deconvolution"));
    if (!spectrum_valid_) {
        throw std::runtime_error(
            "FourierD3 the deconvolution requires a forward-transformed spectrum; run "
            "transform_forward first.");
    }
    if (spectrum_deconvolved_) {
        throw std::logic_error(
            "FourierD3 the deconvolution was already applied to this spectrum; dividing "
            "the continuous symbol twice would silently rescale every amplitude.");
    }
    prepare_grid();
    // One division per stored bin, in place: the divisor table is the continuous
    // symbol of specification section 5.2 and nothing else, so this stage does
    // exactly one deconvolution.
    workspace_.deconvolve_amplitudes();
    spectrum_deconvolved_ = true;
    contracted_ = false;
}

template <typename Float>
void ReciprocalForward<Float>::store_spectrum(std::int64_t species_slot)
{
    static_cast<void>(require_mesh("the species amplitude"));
    if (!spectrum_valid_ || !spectrum_deconvolved_) {
        throw std::runtime_error(
            "FourierD3 a species amplitude must be stored from a deconvolved spectrum; "
            "run transform_forward and deconvolve first.");
    }
    if (species_slot < 0 || species_slot >= num_species_) {
        throw std::out_of_range(
            "FourierD3 species slot " + std::to_string(species_slot)
            + " is outside the reserved range [0," + std::to_string(num_species_) + ").");
    }
    auto spectrum = storage_.spectrum();
    auto spectra = species_spectra_view_;
    const std::int64_t chunk = rank_chunk_;
    const std::int64_t bins = grid_.bins;
    // The stored amplitude is the deconvolved spectrum itself.  The assignment
    // is a partition of unity, so the unnormalized transform of the spread mesh
    // already is S_X^l(k) = sum_i c_il exp(i k . r_i) attenuated by the sampled
    // stencil symbol; dividing by the continuous symbol once is what leaves the
    // structure factor, and no point count enters here.  A 1/N here would make
    // the contraction N^2 times too small and would not be recoverable later.
    Kokkos::parallel_for(
        "symmetrix_fourier_d3_store_spectrum",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            execution_space_, 0, chunk * bins),
        KOKKOS_LAMBDA(const std::int64_t element) {
            const std::int64_t l = element / bins;
            const std::int64_t bin = element % bins;
            spectra((species_slot * chunk + l) * bins + bin) =
                spectrum.data()[l * bins + bin];
        });
    spectra_stored_[static_cast<std::size_t>(species_slot)] = 1;
    contracted_ = false;
    potential_slot_ = -1;
    energy_ = 0.0;
}

template <typename Float>
void ReciprocalForward<Float>::contract()
{
    require_stored_spectra();
    require_species("the reciprocal contraction");
    require_cell("the reciprocal contraction");
    require_rank("the reciprocal contraction");
    prepare_kernels();

    auto spectra = species_spectra_view_;
    auto cotangent = cotangent_view_;
    auto kernels = kernels_.pair_kernel();
    auto eigenvalues = eigenvalues_view_;
    auto hermitian = grid_.hermitian_weight;
    auto bins_energy = bin_energy_view_;
    const std::int64_t nspec = num_species_;
    const std::int64_t chunk = rank_chunk_;
    const std::int64_t bins = grid_.bins;
    const double inverse_volume = inverse_volume_;

    Kokkos::parallel_for(
        "symmetrix_fourier_d3_contract",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(execution_space_, 0, bins),
        KOKKOS_LAMBDA(const std::int64_t bin) {
            const double weight = static_cast<double>(hermitian(bin));
            double total = 0.0;
            for (std::int64_t l = 0; l < chunk; ++l) {
                const double eigenvalue = eigenvalues(l);
                for (std::int64_t x = 0; x < nspec; ++x) {
                    const complex_type amplitude = spectra((x * chunk + l) * bins + bin);
                    complex_type sum(0.0, 0.0);
                    for (std::int64_t y = 0; y < nspec; ++y) {
                        const std::int64_t pair = unordered_pair_index(x, y, nspec);
                        sum += static_cast<Float>(kernels(pair * bins + bin))
                            * spectra((y * chunk + l) * bins + bin);
                    }
                    total += eigenvalue * (amplitude * Kokkos::conj(sum)).real();
                    // The cotangent is *not* scaled by the Hermitian multiplicity:
                    // the inverse transform's own Hermitian completion supplies
                    // the conjugate bins of every stored bin, and carrying the
                    // multiplicity here as well would double count them. This
                    // retained value is exactly dE/dA for A = S/B; the shared
                    // reverse transport applies the chain-rule division by B.
                    cotangent((x * chunk + l) * bins + bin) =
                        -static_cast<Float>(eigenvalue * inverse_volume) * sum;
                }
            }
            bins_energy(bin) = -0.5 * weight * total * inverse_volume;
        });

    auto mirror = Kokkos::create_mirror_view(bin_energy_view_);
    Kokkos::deep_copy(execution_space_, mirror, bin_energy_view_);
    execution_space_.fence("FourierD3 reciprocal-energy readback");
    double total = 0.0;
    for (std::int64_t bin = 0; bin < bins; ++bin) {
        total += mirror(bin);
    }
    reciprocal_energy_ = total;
    contracted_ = true;
    energy_ = 0.0;
}

template <typename Float>
void ReciprocalForward<Float>::self_energy()
{
    require_species("the self term");
    require_rank("the self term");
    require_coefficients("the self term");
    if (!damping_set_) {
        throw std::runtime_error(
            "FourierD3 the self term requires BJ damping coefficients; call set_damping "
            "first.");
    }
    static_cast<void>(require_mesh("the self term"));
    prepare_kernels();

    auto coefficients = coefficients_view_;
    auto channels = channel_view_;
    auto eigenvalues = eigenvalues_view_;
    auto zero_distance = zero_distance_view_;
    auto nodes_energy = node_energy_view_;
    auto derivative = self_derivative_view_;
    const std::int64_t nodes = num_nodes_;
    const std::int64_t chunk = rank_chunk_;

    // The self term is diagonal in the atoms: every atom contributes
    // 0.5 * sum_l lambda_l c_il^2 phi_XX(0), added exactly once and outside the
    // species-pair and transform loops.  Only the zero-translation diagonal is
    // cancelled; the periodic self images are retained by the reciprocal sum,
    // which is why an isolated atom in a periodic cell has a non-zero energy.
    Kokkos::parallel_for(
        "symmetrix_fourier_d3_self",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(execution_space_, 0, nodes),
        KOKKOS_LAMBDA(const std::int64_t node) {
            const std::int32_t channel = channels(node);
            double accumulator = 0.0;
            for (std::int64_t l = 0; l < chunk; ++l) {
                const Float coefficient = coefficients(node * chunk + l);
                const double lambda = eigenvalues(l);
                if (channel < 0) {
                    derivative(node * chunk + l) = Float(0);
                    continue;
                }
                const double potential = static_cast<double>(zero_distance(channel));
                accumulator += lambda * static_cast<double>(coefficient)
                    * static_cast<double>(coefficient) * potential;
                derivative(node * chunk + l) = static_cast<Float>(
                    lambda * static_cast<double>(coefficient) * potential);
            }
            nodes_energy(node) = 0.5 * accumulator;
        });

    auto mirror = Kokkos::create_mirror_view(node_energy_view_);
    Kokkos::deep_copy(execution_space_, mirror, node_energy_view_);
    execution_space_.fence("FourierD3 self-energy readback");
    double total = 0.0;
    for (std::int64_t node = 0; node < nodes; ++node) {
        total += mirror(node);
    }
    self_energy_ = total;
    self_valid_ = true;
}

template <typename Float>
void ReciprocalForward<Float>::invert(std::int64_t species_slot)
{
    static_cast<void>(require_mesh("the inverse transform"));
    if (!contracted_) {
        throw std::runtime_error(
            "FourierD3 the inverse transform requires a contracted cotangent; run "
            "contract first.");
    }
    if (species_slot < 0 || species_slot >= num_species_
        || spectra_stored_[static_cast<std::size_t>(species_slot)] == 0) {
        throw std::runtime_error(
            "FourierD3 the inverse transform requires the stored spectrum of species slot "
            + std::to_string(species_slot) + ".");
    }
    auto cotangent = cotangent_view_;
    const std::int64_t chunk = rank_chunk_;
    const std::int64_t bins = grid_.bins;
    using cotangent_input =
        typename particle_mesh::ParticleMeshWorkspace<Float>::complex_input_view;
    const cotangent_input physical_cotangent(
        cotangent.data() + species_slot * chunk * bins, chunk * bins);
    workspace_.load_physical_cotangent(physical_cotangent);
    workspace_.deconvolve_cotangent();
    workspace_.transform_inverse();
    real_content_ = RealContent::potential;
    real_slot_ = species_slot;
    potential_slot_ = species_slot;
    spectrum_valid_ = false;
    spectrum_deconvolved_ = false;
}

template <typename Float>
void ReciprocalForward<Float>::prepare()
{
    // The coefficients are deliberately not required here: the reverse path
    // adopts them from the coordination workspace and installs them between this
    // call and `execute`, and everything below is independent of them.
    static_cast<void>(require_mesh("the forward evaluation"));
    require_cell("the forward evaluation");
    require_species("the forward evaluation");
    require_atoms("the forward evaluation");
    require_rank("the forward evaluation");
    if (!damping_set_) {
        throw std::runtime_error(
            "FourierD3 the forward evaluation requires BJ damping coefficients; call "
            "set_damping first.");
    }
    // Every retained table and both roles of the plan domain are built here, so
    // the stages below cannot allocate.  Each of them is guarded by its own key
    // and is a no-op once the geometry is unchanged.
    time_stage(forward_preparation_timer_, timing_enabled_, [&] {
        prepare_stencil();
        prepare_grid();
        prepare_kernels();
        workspace_.prepare_transforms();
    });
}

template <typename Float>
ReciprocalOutcome ReciprocalForward<Float>::execute()
{
    require_coefficients("the forward evaluation");
    clear_stored_spectra();
    time_stage(assignment_forward_fft_timer_, timing_enabled_, [&] {
        for (std::int64_t slot = 0; slot < num_species_; ++slot) {
            assign(slot);
            transform_forward();
            deconvolve();
            store_spectrum(slot);
        }
    });
    time_stage(reciprocal_contraction_timer_, timing_enabled_, [&] {
        contract();
        self_energy();
    });
    energy_ = reciprocal_energy_ + self_energy_;
    return ReciprocalOutcome{reciprocal_energy_, self_energy_, energy_};
}

template <typename Float>
FourierD3StageTimings ReciprocalForward<Float>::timings() const
{
    FourierD3StageTimings report;
    report.forward_preparation = forward_preparation_timer_.timing();
    report.assignment_forward_fft = assignment_forward_fft_timer_.timing();
    report.reciprocal_contraction = reciprocal_contraction_timer_.timing();
    return report;
}

template <typename Float>
ReciprocalOutcome ReciprocalForward<Float>::run()
{
    prepare();
    return execute();
}

template <typename Float>
double ReciprocalForward<Float>::debug_evaluate_field(std::span<const Float> values)
{
    static_cast<void>(require_mesh("the diagnostic evaluation"));
    require_cell("the diagnostic evaluation");
    require_species("the diagnostic evaluation");
    require_rank("the diagnostic evaluation");
    if (!damping_set_) {
        throw std::runtime_error(
            "FourierD3 the diagnostic evaluation requires BJ damping coefficients; call "
            "set_damping first.");
    }
    const std::int64_t slice =
        detail::checked_product(rank_chunk_, grid_.points, "FourierD3 field slice");
    if (values.size() != static_cast<std::size_t>(num_species_ * slice)) {
        throw std::invalid_argument(
            "FourierD3 the diagnostic field must hold one value per species, rank slot, and "
            "mesh point, got " + std::to_string(values.size()) + " for "
            + std::to_string(num_species_ * slice) + ".");
    }
    clear_stored_spectra();
    workspace_.prepare_transforms();
    for (std::int64_t slot = 0; slot < num_species_; ++slot) {
        set_field(values.subspan(static_cast<std::size_t>(slot * slice),
            static_cast<std::size_t>(slice)));
        transform_forward();
        deconvolve();
        store_spectrum(slot);
    }
    contract();
    return reciprocal_energy_;
}

// ---------------------------------------------------------------------------
// Readbacks
// ---------------------------------------------------------------------------

template <typename Float>
std::vector<Float> ReciprocalForward<Float>::field() const
{
    const particle_mesh::MeshShape mesh = require_mesh("the real field");
    if (real_content_ == RealContent::none) {
        throw std::runtime_error(
            "FourierD3 the real field holds no mesh; spread a species or load a field "
            "first.");
    }
    const auto allocation = storage_.allocation();
    const std::int64_t elements = mesh.real_element_count();
    const Kokkos::View<const Float*, Kokkos::DefaultExecutionSpace::memory_space,
        Kokkos::MemoryUnmanaged>
        active(allocation->real.data(), elements);
    Kokkos::View<Float*, Kokkos::HostSpace> mirror(
        "symmetrix_fourier_d3_field_mirror", elements);
    Kokkos::deep_copy(execution_space_, mirror, active);
    execution_space_.fence("FourierD3 field readback");
    std::vector<Float> values(static_cast<std::size_t>(elements));
    for (std::int64_t index = 0; index < elements; ++index) {
        values[static_cast<std::size_t>(index)] = mirror(index);
    }
    return values;
}

template <typename Float>
std::vector<std::complex<Float>> ReciprocalForward<Float>::spectrum() const
{
    const particle_mesh::MeshShape mesh = require_mesh("the mesh spectrum");
    if (!spectrum_valid_) {
        throw std::runtime_error(
            "FourierD3 the mesh spectrum holds no transform; run transform_forward first.");
    }
    const auto allocation = storage_.allocation();
    const std::int64_t elements = mesh.spectrum_element_count();
    const Kokkos::View<const complex_type*, Kokkos::DefaultExecutionSpace::memory_space,
        Kokkos::MemoryUnmanaged>
        active(allocation->spectrum.data(), elements);
    Kokkos::View<complex_type*, Kokkos::HostSpace> mirror(
        "symmetrix_fourier_d3_spectrum_mirror", elements);
    Kokkos::deep_copy(execution_space_, mirror, active);
    execution_space_.fence("FourierD3 spectrum readback");
    std::vector<std::complex<Float>> values(static_cast<std::size_t>(elements));
    for (std::int64_t index = 0; index < elements; ++index) {
        values[static_cast<std::size_t>(index)] =
            std::complex<Float>(mirror(index).real(), mirror(index).imag());
    }
    return values;
}

template <typename Float>
std::vector<std::complex<Float>> ReciprocalForward<Float>::structure_factors() const
{
    require_stored_spectra();
    auto mirror = Kokkos::create_mirror_view(species_spectra_view_);
    Kokkos::deep_copy(execution_space_, mirror, species_spectra_view_);
    execution_space_.fence("FourierD3 structure-factor readback");
    const std::int64_t active = num_species_ * rank_chunk_ * grid_.bins;
    std::vector<std::complex<Float>> values(static_cast<std::size_t>(active));
    for (std::int64_t index = 0; index < active; ++index) {
        values[static_cast<std::size_t>(index)] =
            std::complex<Float>(mirror(index).real(), mirror(index).imag());
    }
    return values;
}

template <typename Float>
std::vector<Float> ReciprocalForward<Float>::potential(std::int64_t species_slot) const
{
    const particle_mesh::MeshShape mesh = require_mesh("the potential");
    if (potential_slot_ != species_slot) {
        throw std::runtime_error(
            "FourierD3 the potential buffer holds no inverse transform of species slot "
            + std::to_string(species_slot)
            + "; run invert for that species after a contraction.");
    }
    const auto allocation = storage_.allocation();
    const std::int64_t elements = mesh.real_element_count();
    const Kokkos::View<const Float*, Kokkos::DefaultExecutionSpace::memory_space,
        Kokkos::MemoryUnmanaged>
        active(allocation->real.data(), elements);
    Kokkos::View<Float*, Kokkos::HostSpace> mirror(
        "symmetrix_fourier_d3_potential_mirror", elements);
    Kokkos::deep_copy(execution_space_, mirror, active);
    execution_space_.fence("FourierD3 potential readback");
    std::vector<Float> values(static_cast<std::size_t>(elements));
    for (std::int64_t index = 0; index < elements; ++index) {
        values[static_cast<std::size_t>(index)] = mirror(index);
    }
    return values;
}

template <typename Float>
std::vector<Float> ReciprocalForward<Float>::pair_kernel() const
{
    const auto view = kernels_.pair_kernel();
    auto mirror = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(execution_space_, mirror, view);
    execution_space_.fence("FourierD3 pair-kernel readback");
    std::vector<Float> values(static_cast<std::size_t>(view.extent(0)));
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = mirror(index);
    }
    return values;
}

template <typename Float>
std::vector<Float> ReciprocalForward<Float>::pair_kernel_derivative() const
{
    const auto view = kernels_.pair_kernel_derivative();
    auto mirror = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(execution_space_, mirror, view);
    execution_space_.fence("FourierD3 pair-kernel-derivative readback");
    std::vector<Float> values(static_cast<std::size_t>(view.extent(0)));
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = mirror(index);
    }
    return values;
}

template <typename Float>
std::vector<double> ReciprocalForward<Float>::bin_energies() const
{
    static_cast<void>(require_mesh("the per-bin subtotals"));
    if (!contracted_) {
        throw std::runtime_error(
            "FourierD3 the per-bin subtotals require a contraction; run contract first.");
    }
    auto mirror = Kokkos::create_mirror_view(bin_energy_view_);
    Kokkos::deep_copy(execution_space_, mirror, bin_energy_view_);
    execution_space_.fence("FourierD3 bin-energy readback");
    std::vector<double> values(static_cast<std::size_t>(grid_.bins));
    for (std::int64_t bin = 0; bin < grid_.bins; ++bin) {
        values[static_cast<std::size_t>(bin)] = mirror(bin);
    }
    return values;
}

template <typename Float>
double ReciprocalForward<Float>::reciprocal_energy() const
{
    if (!contracted_) {
        throw std::runtime_error(
            "FourierD3 the reciprocal energy requires a contraction; run contract first.");
    }
    return reciprocal_energy_;
}

template <typename Float>
double ReciprocalForward<Float>::self_energy_value() const
{
    if (!self_valid_) {
        throw std::runtime_error(
            "FourierD3 the self term has not been evaluated; run self_energy first.");
    }
    return self_energy_;
}

template <typename Float>
double ReciprocalForward<Float>::energy_value() const
{
    if (!self_valid_ || !contracted_) {
        throw std::runtime_error(
            "FourierD3 the total energy requires both a contraction and the self term; run "
            "run or call contract and self_energy.");
    }
    return reciprocal_energy_ + self_energy_;
}

template <typename Float>
std::vector<Float> ReciprocalForward<Float>::self_coefficient_derivative() const
{
    if (!self_valid_) {
        throw std::runtime_error(
            "FourierD3 the self-term coefficient derivative has not been evaluated; run "
            "self_energy first.");
    }
    auto mirror = Kokkos::create_mirror_view(self_derivative_view_);
    Kokkos::deep_copy(execution_space_, mirror, self_derivative_view_);
    execution_space_.fence("FourierD3 self-derivative readback");
    const std::int64_t active = num_nodes_ * rank_chunk_;
    std::vector<Float> values(static_cast<std::size_t>(active));
    for (std::int64_t index = 0; index < active; ++index) {
        values[static_cast<std::size_t>(index)] = mirror(index);
    }
    return values;
}

template <typename Float>
void ReciprocalForward<Float>::prepare_host_stencil_baseline()
{
    require_cell("the host stencil diagnostic");
    require_atoms("the host stencil diagnostic");
    (void)require_mesh("the host stencil diagnostic");
    workspace_.prepare_host_stencil_baseline(positions_);
    stencil_cell_ = cell_;
    stencil_positions_generation_ = positions_generation_;
    stencil_preparation_ =
        particle_mesh::StencilPreparationPath::host_diagnostic_baseline;
}

template <typename Float>
StencilDiagnostic<Float> ReciprocalForward<Float>::stencil_diagnostic(
    const bool host_baseline)
{
    require_cell("the stencil diagnostic");
    require_atoms("the stencil diagnostic");
    if (host_baseline) {
        prepare_host_stencil_baseline();
    } else {
        prepare_stencil();
    }

    StencilDiagnostic<Float> diagnostic;
    diagnostic.preparation = stencil_preparation_;
    diagnostic.host_fallback =
        stencil_preparation_ != particle_mesh::StencilPreparationPath::device_non_boundary;
    Kokkos::View<std::int32_t*, Kokkos::HostSpace> mirror(
        "symmetrix_fourier_d3_stencil_base_mirror", 3 * num_nodes_);
    Kokkos::deep_copy(execution_space_, mirror, stencil_.base());
    execution_space_.fence("FourierD3 stencil-base diagnostic readback");
    diagnostic.base.resize(static_cast<std::size_t>(3 * num_nodes_));
    for (std::size_t index = 0; index < diagnostic.base.size(); ++index) {
        diagnostic.base[index] = mirror(index);
    }

    const std::int64_t active = 3 * num_nodes_ * spline_order_;
    Kokkos::View<Float*, Kokkos::HostSpace> weight_mirror(
        "symmetrix_fourier_d3_stencil_weight_mirror", active);
    Kokkos::deep_copy(execution_space_, weight_mirror, stencil_.weights());
    execution_space_.fence("FourierD3 stencil-weight diagnostic readback");
    diagnostic.weights.resize(static_cast<std::size_t>(active));
    for (std::size_t index = 0; index < diagnostic.weights.size(); ++index) {
        diagnostic.weights[index] = weight_mirror(index);
    }

    Kokkos::View<Float*, Kokkos::HostSpace> slope_mirror(
        "symmetrix_fourier_d3_stencil_slope_mirror", active);
    Kokkos::deep_copy(execution_space_, slope_mirror, stencil_.slopes());
    execution_space_.fence("FourierD3 stencil-slope diagnostic readback");
    diagnostic.slopes.resize(static_cast<std::size_t>(active));
    for (std::size_t index = 0; index < diagnostic.slopes.size(); ++index) {
        diagnostic.slopes[index] = slope_mirror(index);
    }

    return diagnostic;
}

template <typename Float>
ReciprocalGeometry ReciprocalForward<Float>::geometry() const
{
    const particle_mesh::MeshShape mesh = require_mesh("the geometry report");
    ReciprocalGeometry report;
    report.deconvolution = "continuous";
    report.forward_normalization = "unnormalized";
    report.inverse_normalization = "unnormalized";
    report.hermitian_axis = "z";
    report.mesh = mesh.description();
    report.nx = grid_.nx;
    report.ny = grid_.ny;
    report.nz = grid_.nz;
    report.spectrum_extent = grid_.spectrum_extent;
    report.mesh_points = mesh.point_count();
    report.spectrum_bins = mesh.spectrum_point_count();
    report.species_pairs = num_species_ * (num_species_ + 1) / 2;
    report.stencil_points = spline_order_ * spline_order_ * spline_order_;
    report.active_rank_chunk = rank_chunk_;
    report.rank_begin = rank_begin_;
    report.rank_total = rank_total_;
    report.num_nodes = num_nodes_;
    report.num_species = num_species_;
    report.spline_order = spline_order_;
    return report;
}

template <typename Float>
ReciprocalUsage ReciprocalForward<Float>::usage() const
{
    ReciprocalUsage usage;
    // Before the first successful reserve the object holds no geometry and no
    // buffer, so the itemization must report zeroes rather than the shapes the
    // committed geometry would have.
    const auto push = [&usage, live = buffers_live_](
                          std::string name, std::int64_t active, std::uint64_t element_size) {
        const std::int64_t elements = live ? active : 0;
        usage.buffers.push_back(particle_mesh::BufferUsage{
            std::move(name),
            elements,
            elements,
            detail::element_bytes(elements, element_size),
            detail::element_bytes(elements, element_size),
            0});
    };
    const std::int64_t bins = grid_.bins;
    push("cell", 9, sizeof(double));
    push("species", num_species_, sizeof(std::int32_t));
    push("q_table", num_species_, sizeof(double));
    push("positions", 3 * num_nodes_, sizeof(double));
    push("numbers", num_nodes_, sizeof(std::int32_t));
    push("channel_map", num_nodes_, sizeof(std::int32_t));
    push("eigenvalues", rank_chunk_, sizeof(double));
    push(
        "coefficients", detail::checked_product(num_nodes_, rank_chunk_, "coefficients"),
        sizeof(Float));
    push("stencil_base", 3 * num_nodes_, sizeof(std::int32_t));
    push(
        "stencil_weight",
        detail::checked_product(3 * num_nodes_, spline_order_, "stencil weights"),
        sizeof(Float));
    push(
        "stencil_slope",
        detail::checked_product(3 * num_nodes_, spline_order_, "stencil slopes"),
        sizeof(Float));
    push("k_norm", bins, sizeof(double));
    push("deconvolution", bins, sizeof(Float));
    push("hermitian_weight", bins, sizeof(Float));
    push(
        "species_spectra",
        detail::checked_product(
            detail::checked_product(num_species_, rank_chunk_, "species amplitude slots"),
            bins, "species amplitude elements"),
        sizeof(complex_type));
    push(
        "cotangent",
        detail::checked_product(
            detail::checked_product(num_species_, rank_chunk_, "cotangent slots"), bins,
            "cotangent elements"),
        sizeof(complex_type));
    push("bin_energy", bins, sizeof(double));
    push("bin_radial_derivative", bins, sizeof(double));
    push("node_energy", num_nodes_, sizeof(double));
    push(
        "self_coefficient_derivative",
        detail::checked_product(num_nodes_, rank_chunk_, "self derivative"), sizeof(Float));
    push("zero_distance_potential", num_species_, sizeof(Float));
    detail::finalize_usage(usage.buffers, usage.active_bytes, usage.capacity_bytes);
    usage.mesh = storage_.usage();
    usage.kernels = kernels_.usage();
    usage.plans = plans_.plan_usage();
    usage.reservations = reservations_;
    usage.allocations = allocations_;
    // A plan retains the allocation it was built from, so the bytes reported here
    // are the bytes of the allocation the plans actually own - not the current
    // storage token, which a mesh replacement may already have moved on from.  The
    // two roles normally retain the same allocation, so each distinct retained
    // allocation is counted once rather than once per plan.
    std::vector<std::uint64_t> counted_generations;
    for (const particle_mesh::FftPlanUsage& plan : usage.plans) {
        if (!plan.held || !plan.allocation.owns) {
            continue;
        }
        if (std::find(
                counted_generations.begin(), counted_generations.end(),
                plan.allocation.generation)
            != counted_generations.end()) {
            continue;
        }
        counted_generations.push_back(plan.allocation.generation);
        usage.plan_retained_bytes += plan.allocation.retained_bytes;
    }
    usage.active_bytes += usage.mesh.active_bytes + usage.kernels.active_bytes;
    usage.capacity_bytes += usage.mesh.capacity_bytes + usage.kernels.capacity_bytes;
    return usage;
}

}  // namespace symmetrix::fourier_d3

#endif  // SYMMETRIX_FOURIER_D3_RECIPROCAL_HPP
