// Physics-neutral particle-mesh storage, cache keys, and FFT-plan ownership.
//
// This translation unit is compiled only when the build resolves a KokkosFFT
// provider, and it is the only place that names the provider-specific plan
// type. It contains no model-specific reciprocal kernel or reverse chain.

#include "particle_mesh_storage.hpp"

#include <KokkosFFT.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace symmetrix::particle_mesh {

namespace {

// The provider token is read from KokkosFFT's own generated configuration, so
// it cannot drift from the provider that the configure step linked.
[[nodiscard]] constexpr const char* compiled_provider() noexcept
{
#if defined(KOKKOSFFT_ENABLE_TPL_FFTW)
    return "fftw3";
#elif defined(KOKKOSFFT_ENABLE_TPL_CUFFT)
    return "cufft";
#elif defined(KOKKOSFFT_ENABLE_TPL_HIPFFT)
    return "hipfft";
#else
#error "Particle-mesh storage requires a KokkosFFT FFT provider"
#endif
}


}  // namespace

namespace detail {

namespace {

// Single-threaded test fault state: `next_failure` is the ordinal of the next
// allocation to fail within the armed window, or -1 when disarmed.
struct AllocationFaultState {
    std::int64_t next_failure = -1;
    std::int64_t ordinal = 0;
};

AllocationFaultState& allocation_fault_state()
{
    static AllocationFaultState state;
    return state;
}

}  // namespace

void arm_allocation_fault(std::int64_t fail_at)
{
    if (fail_at < 0) {
        throw std::invalid_argument(
            "the allocation fault index must not be negative, got "
            + std::to_string(fail_at) + ".");
    }
    auto& state = allocation_fault_state();
    state.next_failure = fail_at;
    state.ordinal = 0;
}

void clear_allocation_fault()
{
    allocation_fault_state() = AllocationFaultState{};
}

bool allocation_fault_armed()
{
    return allocation_fault_state().next_failure >= 0;
}

void maybe_fail_allocation(std::string_view what)
{
    auto& state = allocation_fault_state();
    if (state.next_failure < 0) {
        return;
    }
    if (state.ordinal++ != state.next_failure) {
        return;
    }
    state.next_failure = -1;
    throw std::runtime_error(
        "Particle-mesh injected allocation fault while allocating "
        + std::string(what) + ".");
}

std::int64_t checked_extent(std::int64_t value, std::string_view what)
{
    if (value < 1) {
        throw std::invalid_argument(
            std::string(what) + " must be a positive extent, got " + std::to_string(value)
            + ".");
    }
    return value;
}

std::int64_t checked_flat_index(
    const std::int64_t elements, const std::string_view what)
{
    if (elements < 0) {
        throw std::invalid_argument(
            "Particle-mesh buffer '" + std::string(what)
            + "' has a negative element count.");
    }
    if (elements
        > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(
            "Particle-mesh buffer '" + std::string(what) + "' needs "
            + std::to_string(elements)
            + " elements, which exceeds the flat Kokkos index range.");
    }
    return elements;
}

std::int64_t checked_axis_extent(std::int64_t value, std::string_view what)
{
    if (value < 2) {
        throw std::invalid_argument(
            std::string(what) + " must hold at least two nodes, got "
            + std::to_string(value)
            + "; the assignment stencil is at least two nodes wide.");
    }
    return value;
}

std::int64_t checked_product(std::int64_t lhs, std::int64_t rhs, std::string_view what)
{
    if (lhs < 0 || rhs < 0) {
        throw std::invalid_argument(
            std::string(what) + " must not be negative.");
    }
    if (lhs != 0 && rhs > std::numeric_limits<std::int64_t>::max() / lhs) {
        throw std::overflow_error(
            std::string(what) + " overflows a signed 64-bit index.");
    }
    return lhs * rhs;
}

std::uint64_t element_bytes(std::int64_t elements, std::uint64_t element_size)
{
    if (elements < 0) {
        throw std::invalid_argument("element count must not be negative.");
    }
    const auto count = static_cast<std::uint64_t>(elements);
    if (element_size != 0 && count > std::numeric_limits<std::uint64_t>::max() / element_size) {
        throw std::overflow_error("element byte count overflows an unsigned 64-bit size.");
    }
    return count * element_size;
}

std::string format_scalar(double value)
{
    // Seventeen significant digits round-trip a double exactly, so a key
    // description never hides a cache-key scalar change.
    std::array<char, 32> buffer{};
    const int written = std::snprintf(buffer.data(), buffer.size(), "%.17g", value);
    if (written <= 0) {
        throw std::runtime_error("failed to format a particle-mesh cache-key value.");
    }
    return std::string(buffer.data());
}

void require_finite(double value, std::string_view what)
{
    // A non-finite entry would silently poison every comparison of the cache
    // key, so it is rejected at the boundary.  The IEEE-754 exponent is read
    // directly because this library is compiled with the historical
    // -ffast-math policy, under which std::isfinite is optimized to a constant
    // and cannot detect a NaN or an infinity.
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7ff0000000000000ULL) == UINT64_C(0x7ff0000000000000)) {
        throw std::invalid_argument(
            "Particle-mesh cache key " + std::string(what) + " must be finite.");
    }
}

void finalize_usage(
    std::vector<BufferUsage>& buffers,
    std::uint64_t& active_bytes,
    std::uint64_t& capacity_bytes)
{
    active_bytes = 0;
    capacity_bytes = 0;
    for (std::size_t index = 0; index < buffers.size(); ++index) {
        const BufferUsage& buffer = buffers[index];
        if (buffer.active_elements < 0 || buffer.capacity_elements < buffer.active_elements) {
            throw std::logic_error(
                "Particle-mesh buffer '" + buffer.name
                + "' reports active elements beyond its capacity.");
        }
        for (std::size_t other = 0; other < index; ++other) {
            if (buffers[other].name == buffer.name) {
                throw std::logic_error(
                    "Particle-mesh buffer '" + buffer.name
                    + "' is itemized twice; aliases must be counted once.");
            }
        }
        active_bytes += buffer.active_bytes;
        capacity_bytes += buffer.capacity_bytes;
    }
}

}  // namespace detail

std::string_view layout_token(MeshLayout layout)
{
    switch (layout) {
        case MeshLayout::left:
            return "layout_left";
        case MeshLayout::right:
            return "layout_right";
    }
    throw std::invalid_argument("unknown particle-mesh layout.");
}

MeshLayout layout_from_token(std::string_view token)
{
    if (token == "layout_left") {
        return MeshLayout::left;
    }
    if (token == "layout_right") {
        return MeshLayout::right;
    }
    throw std::invalid_argument(
        "unknown particle-mesh layout '" + std::string(token)
        + "'; expected layout_right (the compiled layout) or layout_left.");
}

// ---------------------------------------------------------------------------
// MeshShape
// ---------------------------------------------------------------------------

MeshShape::MeshShape(
    std::int64_t nx,
    std::int64_t ny,
    std::int64_t nz,
    std::int64_t batch,
    MeshLayout layout)
    : extents_{
          detail::checked_axis_extent(nx, "mesh x extent"),
          detail::checked_axis_extent(ny, "mesh y extent"),
          detail::checked_axis_extent(nz, "mesh z extent")},
      batch_(detail::checked_extent(batch, "mesh batch")),
      layout_(layout)
{
    const std::int64_t plane = detail::checked_product(nx, ny, "mesh xy plane");
    point_count_ = detail::checked_product(plane, nz, "mesh point count");
    real_element_count_ = detail::checked_product(batch_, point_count_, "mesh element count");
    spectrum_extent_ = nz / 2 + 1;
    spectrum_point_count_ =
        detail::checked_product(plane, spectrum_extent_, "mesh spectrum point count");
    spectrum_element_count_ = detail::checked_product(
        batch_, spectrum_point_count_, "mesh spectrum element count");
}

std::int64_t MeshShape::extent(int axis) const
{
    if (axis < 0 || axis >= rank) {
        throw std::out_of_range(
            "mesh axis " + std::to_string(axis) + " is outside [0,3).");
    }
    return extents_[static_cast<std::size_t>(axis)];
}

std::int64_t MeshShape::stride(int axis) const
{
    // LayoutRight: the transform axis is contiguous.
    switch (axis) {
        case 0:
            return detail::checked_product(extent(1), extent(2), "mesh x stride");
        case 1:
            return extent(2);
        case 2:
            return 1;
        default:
            throw std::out_of_range(
                "mesh axis " + std::to_string(axis) + " is outside [0,3).");
    }
}

std::int64_t MeshShape::spectrum_stride(int axis) const
{
    switch (axis) {
        case 0:
            return detail::checked_product(
                extent(1), spectrum_extent_, "mesh spectrum x stride");
        case 1:
            return spectrum_extent_;
        case 2:
            return 1;
        default:
            throw std::out_of_range(
                "mesh axis " + std::to_string(axis) + " is outside [0,3).");
    }
}

std::int64_t MeshShape::offset(std::int64_t i, std::int64_t j, std::int64_t k) const
{
    if (i < 0 || i >= extents_[0] || j < 0 || j >= extents_[1] || k < 0 || k >= extents_[2]) {
        throw std::out_of_range(
            "mesh index (" + std::to_string(i) + "," + std::to_string(j) + ","
            + std::to_string(k) + ") is outside " + description() + ".");
    }
    const std::int64_t flat = detail::checked_product(i, stride(0), "mesh offset")
        + detail::checked_product(j, stride(1), "mesh offset") + k;
    if (flat >= point_count_) {
        throw std::out_of_range("mesh offset exceeds the flattened point count.");
    }
    return flat;
}

std::int64_t MeshShape::spectrum_offset(std::int64_t i, std::int64_t j, std::int64_t k) const
{
    if (i < 0 || i >= extents_[0] || j < 0 || j >= extents_[1] || k < 0
        || k >= spectrum_extent_) {
        throw std::out_of_range(
            "mesh spectrum index (" + std::to_string(i) + "," + std::to_string(j) + ","
            + std::to_string(k) + ") is outside " + description() + ".");
    }
    const std::int64_t flat = detail::checked_product(i, spectrum_stride(0), "mesh spectrum offset")
        + detail::checked_product(j, spectrum_stride(1), "mesh spectrum offset") + k;
    if (flat >= spectrum_point_count_) {
        throw std::out_of_range("mesh spectrum offset exceeds the flattened bin count.");
    }
    return flat;
}

std::string MeshShape::description() const
{
    return std::string(layout_token(layout_)) + ":" + std::to_string(extents_[0]) + "x"
        + std::to_string(extents_[1]) + "x" + std::to_string(extents_[2])
        + ":batch=" + std::to_string(batch_);
}

// ---------------------------------------------------------------------------
// FFT-plan key and identity
// ---------------------------------------------------------------------------

template <typename Float>
PlanIdentity compiled_plan_identity()
{
    return PlanIdentity{
        Kokkos::DefaultExecutionSpace::name(),
        std::string(MeshStorage<Float>::precision_token()),
        compiled_provider()};
}

template PlanIdentity compiled_plan_identity<float>();
template PlanIdentity compiled_plan_identity<double>();

FftPlanKey::FftPlanKey(MeshShape mesh, PlanIdentity identity, FftPlanRole role)
    : mesh_(std::move(mesh)), identity_(std::move(identity)), role_(role)
{
}

std::string FftPlanKey::description() const
{
    return "role=" + std::string(plan_role_token(role_)) + " backend=" + identity_.backend
        + " precision=" + identity_.precision + " provider=" + identity_.provider
        + " mesh=[" + mesh_.description() + "]";
}

std::string_view plan_role_token(FftPlanRole role)
{
    switch (role) {
        case FftPlanRole::forward_r2c:
            return "forward_r2c";
        case FftPlanRole::inverse_c2r:
            return "inverse_c2r";
    }
    throw std::invalid_argument("unknown particle-mesh FFT plan role.");
}

FftPlanRole plan_role_from_token(std::string_view token)
{
    if (token == "forward_r2c") {
        return FftPlanRole::forward_r2c;
    }
    if (token == "inverse_c2r") {
        return FftPlanRole::inverse_c2r;
    }
    throw std::invalid_argument(
        "unknown particle-mesh FFT plan role '" + std::string(token)
        + "'; expected forward_r2c or inverse_c2r.");
}

template <typename Float>
FftPlanHandle<Float>::~FftPlanHandle() = default;

template class FftPlanHandle<float>;
template class FftPlanHandle<double>;

namespace {

// One provider plan over one committed mesh allocation.  The KokkosFFT plan
// captures the active buffer addresses while the active views are unmanaged, so
// this class retains the storage's `MeshAllocation` token: the buffers stay
// allocated for as long as the plan exists, and the plan may safely outlive the
// storage that created them.  `kokkos_live_object_guard_` keeps the plan
// counted as a live Kokkos owner, so Kokkos cannot be finalized while it - or
// the allocation it retains - is alive.
//
// The forward specialization reads the real mesh and writes the half spectrum;
// the inverse specialization reads the half spectrum and writes the real mesh.
// Both run with `Normalization::none`, so neither transform is scaled: a round
// trip scales by the point count, and the inverse is the exact adjoint of the
// unnormalized forward transform.
template <typename Float, FftPlanRole Role>
class HostRfftPlan final : public FftPlanHandle<Float> {
public:
    using real_view_type = typename MeshStorage<Float>::real_active_view;
    using spectrum_view_type = typename MeshStorage<Float>::spectrum_active_view;
    using in_view_type = std::conditional_t<
        Role == FftPlanRole::forward_r2c, real_view_type, spectrum_view_type>;
    using out_view_type = std::conditional_t<
        Role == FftPlanRole::forward_r2c, spectrum_view_type, real_view_type>;
    static constexpr KokkosFFT::Direction direction =
        Role == FftPlanRole::forward_r2c ? KokkosFFT::Direction::forward
                                         : KokkosFFT::Direction::backward;
    using plan_type =
        KokkosFFT::Plan<Kokkos::DefaultExecutionSpace, in_view_type, out_view_type, MeshShape::rank>;

    [[nodiscard]] static std::unique_ptr<HostRfftPlan> create(
        const Kokkos::DefaultExecutionSpace& execution_space,
        FftPlanKey key,
        std::shared_ptr<const MeshAllocation<Float>> allocation)
    {
        const Views views = active_views(key, *allocation);
        return std::unique_ptr<HostRfftPlan>(
            new HostRfftPlan(execution_space, std::move(key), std::move(allocation), views));
    }

    [[nodiscard]] const FftPlanKey& key() const noexcept override { return key_; }

    [[nodiscard]] std::uint32_t execution_instance_id() const noexcept override
    {
        return execution_instance_id_;
    }

    [[nodiscard]] const void* retained_allocation_token() const noexcept override
    {
        return allocation_.get();
    }

    [[nodiscard]] RetainedAllocationReport retained_allocation() const override
    {
        RetainedAllocationReport report;
        if (!allocation_) {
            return report;
        }
        report.owns = true;
        report.allocation_owners = static_cast<std::int64_t>(allocation_.use_count());
        report.generation = allocation_->generation;
        report.real_elements = allocation_->real_elements();
        report.spectrum_elements = allocation_->spectrum_elements();
        report.retained_bytes = allocation_->retained_bytes();
        report.real_view_use_count = allocation_->real.use_count();
        report.spectrum_view_use_count = allocation_->spectrum.use_count();
        return report;
    }

    void execute(const real_view_type& real, const spectrum_view_type& spectrum) const override
    {
        if constexpr (Role == FftPlanRole::forward_r2c) {
            KokkosFFT::execute(plan_, real, spectrum, KokkosFFT::Normalization::none);
        } else {
            KokkosFFT::execute(plan_, spectrum, real, KokkosFFT::Normalization::none);
        }
    }

private:
    // The active geometry is the plan's key, and it is applied to the retained
    // allocation's managed views rather than to hand-built unmanaged views, so
    // the plan can only ever alias memory the token owns.
    struct Views {
        in_view_type in;
        out_view_type out;
    };

    [[nodiscard]] static Views active_views(
        const FftPlanKey& key, const MeshAllocation<Float>& allocation)
    {
        const MeshShape& mesh = key.mesh();
        real_view_type real(
            allocation.real.data(), mesh.batch(), mesh.extent(0), mesh.extent(1),
            mesh.extent(2));
        spectrum_view_type spectrum(
            allocation.spectrum.data(), mesh.batch(), mesh.extent(0), mesh.extent(1),
            mesh.spectrum_extent());
        if constexpr (Role == FftPlanRole::forward_r2c) {
            return Views{real, spectrum};
        } else {
            return Views{spectrum, real};
        }
    }

    HostRfftPlan(
        const Kokkos::DefaultExecutionSpace& execution_space,
        FftPlanKey key,
        std::shared_ptr<const MeshAllocation<Float>> allocation,
        const Views& views)
        : kokkos_live_object_guard_(),
          key_(std::move(key)),
          allocation_(std::move(allocation)),
          execution_instance_id_(execution_space.impl_instance_id()),
          plan_(
              execution_space,
              views.in,
              views.out,
              // The transform covers the three mesh axes; the leading axis is
              // the transform batch, so the halved axis stays innermost and no
              // transpose is inserted.
              direction,
              KokkosFFT::axis_type<MeshShape::rank>{1, 2, 3})
    {
    }

    KokkosLiveObjectGuard kokkos_live_object_guard_;
    FftPlanKey key_;
    std::shared_ptr<const MeshAllocation<Float>> allocation_;
    std::uint32_t execution_instance_id_ = 0;
    plan_type plan_;
};

}  // namespace

template <typename Float>
std::unique_ptr<FftPlanHandle<Float>> build_fft_plan(
    const Kokkos::DefaultExecutionSpace& execution_space,
    const FftPlanKey& key,
    std::shared_ptr<const MeshAllocation<Float>> allocation)
{
    // KokkosFFT registers a finalize hook and aborts when it is called before
    // initialization, so the precondition is checked here and reported as an
    // error instead of a process abort.
    if (!Kokkos::is_initialized()) {
        throw std::runtime_error(
            "Particle-mesh FFT plans require an initialized Kokkos execution space; "
            "initialize Kokkos before building a plan.");
    }
    if (key.mesh().layout() != MeshLayout::right) {
        throw std::invalid_argument(
            "Particle-mesh plans support the layout_right mesh only; a "
            "layout_left plan would need an internal transpose, and none is inserted.");
    }
    if (!allocation) {
        throw std::runtime_error(
            "Particle-mesh FFT plans require a committed mesh allocation; the storage "
            "holds none.");
    }
    if (key.role() == FftPlanRole::forward_r2c) {
        return HostRfftPlan<Float, FftPlanRole::forward_r2c>::create(
            execution_space, key, std::move(allocation));
    }
    return HostRfftPlan<Float, FftPlanRole::inverse_c2r>::create(
        execution_space, key, std::move(allocation));
}

template std::unique_ptr<FftPlanHandle<float>> build_fft_plan<float>(
    const Kokkos::DefaultExecutionSpace&,
    const FftPlanKey&,
    std::shared_ptr<const MeshAllocation<float>>);
template std::unique_ptr<FftPlanHandle<double>> build_fft_plan<double>(
    const Kokkos::DefaultExecutionSpace&,
    const FftPlanKey&,
    std::shared_ptr<const MeshAllocation<double>>);

// Both precisions are instantiated here, once for the whole library.
template class MeshStorage<float>;
template class MeshStorage<double>;
template class FftPlanDomain<float>;
template class FftPlanDomain<double>;

}  // namespace symmetrix::particle_mesh
