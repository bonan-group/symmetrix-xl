#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>

namespace symmetrix::mh1 {

inline constexpr int layer_count = 2;

enum class Layer : int {
    first = 0,
    second = 1,
};

inline Layer layer_from_index(const int index)
{
    if (index == 0)
        return Layer::first;
    if (index == 1)
        return Layer::second;
    throw std::out_of_range("MH-1 direct stage index must be zero or one.");
}

inline constexpr int layer_index(const Layer layer) noexcept
{
    return static_cast<int>(layer);
}

inline constexpr std::array<const char*,layer_count> r_forward_regions = {
    "symmetrix/mh1/R0/forward",
    "symmetrix/mh1/R1/forward",
};

inline constexpr std::array<const char*,layer_count> m_forward_regions = {
    "symmetrix/mh1/M0/forward",
    "symmetrix/mh1/M1/forward",
};

inline constexpr std::array<const char*,layer_count> r_reverse_regions = {
    "symmetrix/mh1/R0/reverse",
    "symmetrix/mh1/R1/reverse",
};

inline constexpr std::array<const char*,layer_count> m_reverse_regions = {
    "symmetrix/mh1/M0/reverse",
    "symmetrix/mh1/M1/reverse",
};

template<std::size_t Size>
inline constexpr const char* region_name(
    const std::array<const char*,Size>& names,const Layer layer) noexcept
{
    return names[static_cast<std::size_t>(layer_index(layer))];
}

}  // namespace symmetrix::mh1
