#pragma once

#include <array>

namespace symmetrix::detail {

struct ProductMonomialKey {
    int degree = 0;
    std::array<int,3> indices{};

    bool operator<(const ProductMonomialKey& other) const
    {
        if (degree != other.degree) return degree < other.degree;
        return indices < other.indices;
    }
};

} // namespace symmetrix::detail
