#pragma once
// a1_scale_divide=reciprocal, class ID: the A1 density scale of an FP32 value v by its node's FP64 scale s, the standard expression
// float(double(v)/s), as one FP64 multiply by r = RN64(1/s) wherever that gives the same bits, and as that division everywhere else.
//   t = RN64(double(v)*r).  If t's exponent is in [-125, 126], then |s| lies in about [2^-277, 2^254] (|v| is an FP32 value), so 1/s,
//   v/s and v*r are FP64-normal, t and d = RN64(double(v)/s) both lie within (1 + u)^2 of v/s (u = 2^-53) and |t - d| < 4 ulp64(t);
//   both are FP32-normal and below the FP32 overflow threshold.  The only FP32 rounding boundary within 2^27 ulp64(t) of t is the
//   midpoint of t's FP32 cell, at |R - 2^28| ulp64(t), R the 29 low significand bits of t.  If that is more than 4 ulp64(t), no boundary
//   separates t from d: float(t) == float(d).  A zero t gives the division's signed zero: t is zero only for a zero v or r (an infinite
//   s), or when |v*r| underflows below 2^-1074 (|s| above 2^926), and every quotient that small rounds to an FP32 zero of that sign.
// Every other t takes the division: one outside the window or NaN (a zero, subnormal or NaN s, an infinite or NaN v, an FP32-subnormal
// or near-overflow result) and one near an FP32 midpoint.  Assumes IEEE FP64 multiply and divide and round-to-nearest
// conversions: no value-changing compiler flag (host builds take -ffast-math and keep the division; HIP builds take none).  The guard
// reads bits only, so it does not depend on the finite-math assumptions a fast-math build would make.
#include <Kokkos_Core.hpp>

#include <cstdint>

namespace symmetrix::optimized {

struct A1ScaleDivisor {
    double s = 1.0;   // the node's scale, the divisor of the standard expression
    double r = 1.0;   // RN64(1/s)
};

// Once per node.
KOKKOS_INLINE_FUNCTION A1ScaleDivisor a1_scale_divisor(const double s)
{
    return {s, 1.0/s};
}

// Does float(t) equal float(double(v)/s) for t = double(v)*r?  True only where the proof above says so.
KOKKOS_INLINE_FUNCTION bool a1_scale_product_exact(const double t)
{
    const std::uint64_t bits = Kokkos::bit_cast<std::uint64_t>(t);
    const auto biased = static_cast<unsigned>(bits >> 52) & 0x7ffu;
    const auto low = static_cast<unsigned>(bits) & 0x1fffffffu;
    const bool normal = biased - 898u <= 251u;                 // unbiased exponent in [-125, 126]
    const bool clear = low - ((1u << 28) - 4u) > 8u;           // |R - 2^28| > 4
    const bool zero = (bits << 1) == 0;
    return (normal && clear) || zero;
}

// float(double(v)/d.s), bit for bit.
KOKKOS_INLINE_FUNCTION float a1_scale_divide(const float v, const A1ScaleDivisor& d)
{
    const double t = static_cast<double>(v)*d.r;
    if (a1_scale_product_exact(t))
        return static_cast<float>(t);
    return static_cast<float>(static_cast<double>(v)/d.s);
}

} // namespace symmetrix::optimized
