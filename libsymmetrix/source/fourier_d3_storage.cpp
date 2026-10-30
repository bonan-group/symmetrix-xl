// FourierD3-specific reciprocal-kernel cache ownership.

#include "fourier_d3_storage.hpp"

#include <array>
#include <string>
#include <utility>
#include <vector>

namespace symmetrix::fourier_d3 {

namespace {

[[nodiscard]] std::string cell_description(const std::array<double, 9>& cell)
{
    std::string text = "cell=[";
    for (std::size_t index = 0; index < cell.size(); ++index) {
        if (index != 0) {
            text += ",";
        }
        text += detail::format_scalar(cell[index]);
    }
    text += "]";
    return text;
}


}  // namespace

// ---------------------------------------------------------------------------
// Reciprocal-kernel key
// ---------------------------------------------------------------------------

KernelKey::KernelKey(
    std::array<double, 9> cell,
    DampingCoefficients damping,
    std::vector<int> species,
    int spline_order,
    std::string backend,
    std::string precision)
    : cell_(cell), damping_(damping), species_(std::move(species)),
      spline_order_(spline_order), backend_(std::move(backend)),
      precision_(std::move(precision))
{
    for (std::size_t index = 0; index < cell_.size(); ++index) {
        detail::require_finite(cell_[index], "cell entry");
    }
    detail::require_finite(damping_.a1, "damping a1");
    detail::require_finite(damping_.a2, "damping a2");
    detail::require_finite(damping_.s6, "damping s6");
    detail::require_finite(damping_.s8, "damping s8");
    // Order 2..6 is the frozen spline range of the specification (5.6); the
    // configuration layer owns the user-facing bound and this is its native
    // precondition.
    if (spline_order_ < 2 || spline_order_ > 6) {
        throw std::invalid_argument(
            "FourierD3 spline order must be between 2 and 6, got "
            + std::to_string(spline_order_) + ".");
    }
    for (std::size_t index = 0; index < species_.size(); ++index) {
        if (species_[index] < 1) {
            throw std::invalid_argument(
                "FourierD3 kernel species channels must be real atomic numbers; padding "
                "(Z=0) is not a channel.");
        }
        if (index != 0 && species_[index] <= species_[index - 1]) {
            throw std::invalid_argument(
                "FourierD3 kernel species must be unique and sorted ascending; got the "
                "repeated or out-of-order entry "
                + std::to_string(species_[index]) + ".");
        }
    }
}

std::int64_t KernelKey::pair_count() const noexcept
{
    const std::int64_t species = static_cast<std::int64_t>(species_.size());
    return species * (species + 1) / 2;
}

std::string KernelKey::description() const
{
    std::string species = "species=[";
    for (std::size_t index = 0; index < species_.size(); ++index) {
        if (index != 0) {
            species += ",";
        }
        species += std::to_string(species_[index]);
    }
    species += "]";

    return "backend=" + backend_ + " precision=" + precision_
        + " " + cell_description(cell_) + " damping=[" + detail::format_scalar(damping_.a1)
        + "," + detail::format_scalar(damping_.a2) + ","
        + detail::format_scalar(damping_.s6) + "," + detail::format_scalar(damping_.s8)
        + "] " + species + " spline_order=" + std::to_string(spline_order_);
}

template class KernelDomain<float>;
template class KernelDomain<double>;

}  // namespace symmetrix::fourier_d3
