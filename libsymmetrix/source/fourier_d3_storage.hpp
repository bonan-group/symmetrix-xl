#ifndef SYMMETRIX_FOURIER_D3_STORAGE_HPP
#define SYMMETRIX_FOURIER_D3_STORAGE_HPP

// FourierD3-specific reciprocal-kernel storage. Particle-mesh geometry,
// workspace storage, and FFT-plan ownership live in particle_mesh_storage.

#include "particle_mesh_storage.hpp"

namespace symmetrix::fourier_d3 {

using particle_mesh::BufferUsage;
using particle_mesh::FftPlanDomain;
using particle_mesh::FftPlanHandle;
using particle_mesh::FftPlanKey;
using particle_mesh::FftPlanRole;
using particle_mesh::FftPlanUsage;
using particle_mesh::MeshAllocation;
using particle_mesh::MeshLayout;
using particle_mesh::MeshReserveOutcome;
using particle_mesh::MeshShape;
using particle_mesh::MeshStorage;
using particle_mesh::MeshStorageUsage;
using particle_mesh::PlanIdentity;
using particle_mesh::RetainedAllocationReport;
using particle_mesh::build_fft_plan;
using particle_mesh::compiled_plan_identity;
using particle_mesh::layout_from_token;
using particle_mesh::layout_token;
using particle_mesh::plan_role_count;
using particle_mesh::plan_role_from_token;
using particle_mesh::plan_role_token;

namespace detail {
using particle_mesh::detail::allocation_fault_armed;
using particle_mesh::detail::arm_allocation_fault;
using particle_mesh::detail::checked_axis_extent;
using particle_mesh::detail::checked_extent;
using particle_mesh::detail::checked_flat_index;
using particle_mesh::detail::checked_product;
using particle_mesh::detail::clear_allocation_fault;
using particle_mesh::detail::element_bytes;
using particle_mesh::detail::finalize_usage;
using particle_mesh::detail::format_scalar;
using particle_mesh::detail::maybe_fail_allocation;
using particle_mesh::detail::require_finite;
}  // namespace detail

// ---------------------------------------------------------------------------
// Reciprocal-kernel domain
// ---------------------------------------------------------------------------

// BJ damping parameters, already unit converted by the configuration layer.
struct DampingCoefficients {
    double a1 = 0.0;
    double a2 = 0.0;
    double s6 = 0.0;
    double s8 = 0.0;

    friend bool operator==(const DampingCoefficients&, const DampingCoefficients&) = default;
};

// Cache key of the reciprocal kernel domain: cell, damping, species, and
// spline order, plus the backend and precision the tables are filled for.
// Graph and decomposition state are deliberately absent - they invalidate
// their own domains - and so is the FFT provider, because a kernel table is a
// value table rather than a backend plan.
class KernelKey {
public:
    KernelKey(
        std::array<double, 9> cell,
        DampingCoefficients damping,
        std::vector<int> species,
        int spline_order,
        std::string backend,
        std::string precision);

    [[nodiscard]] const std::array<double, 9>& cell() const noexcept { return cell_; }
    [[nodiscard]] const DampingCoefficients& damping() const noexcept { return damping_; }
    [[nodiscard]] const std::vector<int>& species() const noexcept { return species_; }
    [[nodiscard]] int spline_order() const noexcept { return spline_order_; }
    [[nodiscard]] const std::string& backend() const noexcept { return backend_; }
    [[nodiscard]] const std::string& precision() const noexcept { return precision_; }
    // Unordered species pairs, including the diagonal (spec 4.2).
    [[nodiscard]] std::int64_t pair_count() const noexcept;
    [[nodiscard]] std::string description() const;

    friend bool operator==(const KernelKey&, const KernelKey&) = default;

private:
    std::array<double, 9> cell_{};
    DampingCoefficients damping_{};
    std::vector<int> species_;
    int spline_order_ = 0;
    std::string backend_;
    std::string precision_;
};

struct KernelDomainUsage {
    std::vector<BufferUsage> buffers;
    std::uint64_t active_bytes = 0;
    std::uint64_t capacity_bytes = 0;
    std::uint64_t reservations = 0;
    std::uint64_t allocations = 0;
    std::uint64_t key_hits = 0;
    std::uint64_t key_misses = 0;
    std::uint64_t key_rebuilds = 0;
    bool has_key = false;
    std::string key;
    PlanIdentity identity;
};

// Grow-only kernel tables for one species set and mesh spectrum: the BJ kernel
// values and their radial derivatives, both consumed by the reciprocal
// contraction (the reciprocal kernel owns their contents).  A key change asks the caller to
// refill the tables; the retained allocation is reused when it already fits.
//
// `reserve` is all-or-nothing, exactly like `MeshStorage::reserve`: the two
// replacement tables are allocated locally, the key is copied locally, and the
// committed key, geometry, tables, ids, and counters are replaced together only
// after all of that succeeded.  A failed reserve therefore leaves the domain
// describing - and holding - exactly the tables it held before, so the tables
// the reciprocal kernel reads never become a mismatched pair.
template <typename Float>
class KernelDomain {
public:
    struct Outcome {
        bool key_reused = false;
        bool buffers_grown = false;
    };

    KernelDomain() = default;
    ~KernelDomain() = default;

    KernelDomain(const KernelDomain&) = delete;
    KernelDomain& operator=(const KernelDomain&) = delete;
    KernelDomain(KernelDomain&&) = delete;
    KernelDomain& operator=(KernelDomain&&) = delete;

    Outcome reserve(const KernelKey& key, const MeshShape& mesh);
    [[nodiscard]] const KernelKey* key() const noexcept { return key_ ? &*key_ : nullptr; }
    [[nodiscard]] KernelDomainUsage usage() const;

    // Active tables alias the retained allocations; the caller fills them
    // through these views.
    using active_table_view = Kokkos::View<
        Float*, Kokkos::DefaultExecutionSpace::memory_space, Kokkos::MemoryUnmanaged>;

    [[nodiscard]] active_table_view pair_kernel() const;
    [[nodiscard]] active_table_view pair_kernel_derivative() const;

private:
    KokkosLiveObjectGuard kokkos_live_object_guard_;
    std::optional<KernelKey> key_;
    std::int64_t active_elements_ = 0;
    std::int64_t capacity_elements_ = 0;
    std::uint64_t reservations_ = 0;
    std::uint64_t allocations_ = 0;
    std::uint64_t key_hits_ = 0;
    std::uint64_t key_misses_ = 0;
    std::uint64_t key_rebuilds_ = 0;
    std::uint64_t allocation_serial_ = 0;
    std::uint64_t pair_kernel_id_ = 0;
    std::uint64_t pair_kernel_derivative_id_ = 0;
    Kokkos::View<Float*> pair_kernel_;
    Kokkos::View<Float*> pair_kernel_derivative_;
};

template <typename Float>
auto KernelDomain<Float>::reserve(const KernelKey& key, const MeshShape& mesh) -> Outcome
{
    const std::int64_t elements = detail::checked_product(
        key.pair_count(), mesh.spectrum_point_count(),
        "FourierD3 reciprocal-kernel table elements");

    const bool key_reused = key_ && *key_ == key;
    const bool grown = elements > capacity_elements_;

    // All-or-nothing: every potentially throwing step happens before the first
    // committed mutation, so a failure of the second table leaves the committed
    // key, tables, ids, and counters untouched.
    std::optional<KernelKey> committed_key = key;
    Kokkos::View<Float*> kernel_replacement;
    Kokkos::View<Float*> derivative_replacement;
    if (grown) {
        detail::maybe_fail_allocation("the pair-kernel table");
        kernel_replacement =
            Kokkos::View<Float*>("symmetrix_fourier_d3_pair_kernel", elements);
        detail::maybe_fail_allocation("the pair-kernel derivative table");
        derivative_replacement = Kokkos::View<Float*>(
            "symmetrix_fourier_d3_pair_kernel_derivative", elements);
    }
    std::uint64_t serial = allocation_serial_;
    const std::uint64_t kernel_id = grown ? serial + 1 : pair_kernel_id_;
    const std::uint64_t derivative_id = grown ? serial + 2 : pair_kernel_derivative_id_;
    const std::uint64_t committed_allocations = allocations_ + (grown ? 2U : 0U);

    // Commit.
    ++reservations_;
    if (key_reused) {
        ++key_hits_;
    } else {
        ++key_misses_;
        if (key_) {
            ++key_rebuilds_;
        }
    }
    if (grown) {
        pair_kernel_ = std::move(kernel_replacement);
        pair_kernel_derivative_ = std::move(derivative_replacement);
        capacity_elements_ = elements;
        allocation_serial_ = serial + 2;
        pair_kernel_id_ = kernel_id;
        pair_kernel_derivative_id_ = derivative_id;
        allocations_ = committed_allocations;
    }
    key_ = std::move(committed_key);
    active_elements_ = elements;
    return Outcome{key_reused, grown};
}

template <typename Float>
KernelDomainUsage KernelDomain<Float>::usage() const
{
    KernelDomainUsage usage;
    usage.reservations = reservations_;
    usage.allocations = allocations_;
    usage.key_hits = key_hits_;
    usage.key_misses = key_misses_;
    usage.key_rebuilds = key_rebuilds_;
    usage.has_key = static_cast<bool>(key_);
    usage.identity = compiled_plan_identity<Float>();
    if (key_) {
        usage.key = key_->description();
    }
    usage.buffers.push_back(BufferUsage{
        "pair_kernel",
        active_elements_,
        capacity_elements_,
        detail::element_bytes(active_elements_, sizeof(Float)),
        detail::element_bytes(capacity_elements_, sizeof(Float)),
        pair_kernel_id_});
    usage.buffers.push_back(BufferUsage{
        "pair_kernel_derivative",
        active_elements_,
        capacity_elements_,
        detail::element_bytes(active_elements_, sizeof(Float)),
        detail::element_bytes(capacity_elements_, sizeof(Float)),
        pair_kernel_derivative_id_});
    detail::finalize_usage(usage.buffers, usage.active_bytes, usage.capacity_bytes);
    return usage;
}

template <typename Float>
typename KernelDomain<Float>::active_table_view KernelDomain<Float>::pair_kernel() const
{
    if (!key_) {
        throw std::runtime_error(
            "FourierD3 reciprocal-kernel tables hold no domain; reserve a key first.");
    }
    return active_table_view(pair_kernel_.data(), active_elements_);
}

template <typename Float>
typename KernelDomain<Float>::active_table_view
KernelDomain<Float>::pair_kernel_derivative() const
{
    if (!key_) {
        throw std::runtime_error(
            "FourierD3 reciprocal-kernel tables hold no domain; reserve a key first.");
    }
    return active_table_view(pair_kernel_derivative_.data(), active_elements_);
}

extern template class KernelDomain<float>;
extern template class KernelDomain<double>;

}  // namespace symmetrix::fourier_d3

#endif  // SYMMETRIX_FOURIER_D3_STORAGE_HPP
