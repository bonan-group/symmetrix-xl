#pragma once

#include <bit>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace nequip_validation {

inline bool is_finite(const double value)
{
    static_assert(std::numeric_limits<double>::is_iec559);
    constexpr std::uint64_t exponent_mask = UINT64_C(0x7ff0000000000000);
    return (std::bit_cast<std::uint64_t>(value) & exponent_mask) != exponent_mask;
}

inline bool all_finite(const std::span<const double> values)
{
    for (const double value : values)
        if (!is_finite(value)) return false;
    return true;
}

}  // namespace nequip_validation

namespace symmetrix::execution {

struct OAMOwnerWorkRange {
    int tile_count;
    int work_items;
};

inline OAMOwnerWorkRange checked_oam_owner_work_range(
    const int owner_width,
    const int channel_tile,
    const int owner_count,
    const bool channel_tiled)
{
    if (owner_width <= 0 || channel_tile <= 0 || owner_count < 0)
        throw std::overflow_error("generated OAM owner work range is invalid");
    const std::int64_t tile_count = channel_tiled
        ? (static_cast<std::int64_t>(owner_width)
            + static_cast<std::int64_t>(channel_tile) - 1)
            / static_cast<std::int64_t>(channel_tile)
        : 1;
    if (tile_count <= 0 || tile_count > std::numeric_limits<int>::max()
        || (owner_count > 0
            && tile_count > std::numeric_limits<int>::max() / owner_count))
        throw std::overflow_error(
            "generated OAM owner/tile work range exceeds ABI bounds");
    return {
        static_cast<int>(tile_count),
        owner_count * static_cast<int>(tile_count)};
}

}  // namespace symmetrix::execution
