"""Host check of the direct-plan Y normalization against the retained rounding.

The direct (y-only) plan normalizes the hardcoded harmonics with
``spherical_harmonic_mace_normalize`` from ``spherical_harmonic_device.hpp``;
the retained plan computes ``float(double(y) * C)``. The header is compiled
here with a small Kokkos stand-in (only the scalar math it uses), so the test
needs a C++17 compiler but no Kokkos build, and it checks every float input.
"""

import json
import os
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

_SOURCE = Path(__file__).resolve().parents[2] / "libsymmetrix" / "source"

_KOKKOS_STAND_IN = """\
#pragma once
#include <cmath>
#define KOKKOS_INLINE_FUNCTION inline
#define KOKKOS_LAMBDA [=]
namespace Kokkos {
using std::fabs;
using std::fma;
using std::sqrt;
// Serial stand-ins for the launch helpers, which this test does not call.
template <class Space>
struct RangePolicy {
    RangePolicy(const Space&, int begin, int end) : begin(begin), end(end) {}
    int begin;
    int end;
};
template <class Policy, class Functor>
void parallel_for(const char*, const Policy& policy, const Functor& functor)
{
    for (int index = policy.begin; index < policy.end; ++index)
        functor(index);
}
}
"""

# The normalization does not use the SpheriCart polynomial macros.
_MACROS_STAND_IN = """\
#pragma once
#define HARDCODED_SPH_MACRO(...)
#define HARDCODED_SPH_DERIVATIVE_MACRO(...)
"""

_DRIVER = r"""
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "spherical_harmonic_device.hpp"

namespace {

float from_bits(std::uint32_t bits)
{
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::uint32_t to_bits(float value)
{
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float retained(float value)
{
    return static_cast<float>(
        static_cast<double>(value)*symmetrix::spherical_harmonic_mace_normalization);
}

// Previous direct-plan product: raw times the float-rounded constant.
float float_constant(float value)
{
    return value*static_cast<float>(symmetrix::spherical_harmonic_mace_normalization);
}

bool same(float a, float b)
{
    return (std::isnan(a) && std::isnan(b)) || to_bits(a) == to_bits(b);
}

// Distance in units in the last place between two finite floats.
std::int64_t ulp_distance(float a, float b)
{
    auto ordered = [](float value) {
        const std::int64_t bits = static_cast<std::int32_t>(to_bits(value));
        return bits < 0 ? std::int64_t(INT32_MIN) - bits : bits;
    };
    const std::int64_t difference = ordered(a) - ordered(b);
    return difference < 0 ? -difference : difference;
}

struct Tally {
    std::uint64_t checked = 0;
    std::uint64_t mismatches = 0;
    std::uint32_t first_mismatch = 0;
    std::int64_t max_ulp_vs_float_constant = 0;
};

void sweep(std::uint64_t begin, std::uint64_t end, Tally& tally)
{
    for (std::uint64_t bits = begin; bits < end; ++bits) {
        const float value = from_bits(static_cast<std::uint32_t>(bits));
        const float normalized = symmetrix::spherical_harmonic_mace_normalize(value);
        const float expected = retained(value);
        tally.checked += 1;
        if (!same(normalized, expected)) {
            if (tally.mismatches == 0)
                tally.first_mismatch = static_cast<std::uint32_t>(bits);
            tally.mismatches += 1;
        }
        const float previous = float_constant(value);
        if (std::isfinite(expected) && std::isfinite(previous))
            tally.max_ulp_vs_float_constant = std::max(
                tally.max_ulp_vs_float_constant, ulp_distance(previous, expected));
    }
}

// The array form tests all values of one edge together and takes one path.
std::uint64_t check_values()
{
    constexpr int size = 16;
    std::uint64_t mismatches = 0;
    std::uint32_t state = 0x9e3779b9u;
    auto next = [&state]() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    };
    for (int trial = 0; trial < 200000; ++trial) {
        float raw[size];
        float values[size];
        for (int index = 0; index < size; ++index) {
            const std::uint32_t bits = next();
            // Mostly harmonic-sized values; some arbitrary bit patterns put
            // denormal, huge, infinite or NaN entries next to ordinary ones.
            raw[index] = (bits & 7u) == 0
                ? from_bits(next())
                : static_cast<float>(static_cast<std::int32_t>(bits))*0x1p-31f;
        }
        symmetrix::spherical_harmonic_mace_normalize_values<size>(raw, values);
        for (int index = 0; index < size; ++index)
            if (!same(values[index], retained(raw[index])))
                mismatches += 1;
    }
    return mismatches;
}

}  // namespace

int main()
{
    const unsigned workers = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    constexpr std::uint64_t total = std::uint64_t(1) << 32;
    std::vector<Tally> tallies(workers);
    std::vector<std::thread> threads;
    for (unsigned worker = 0; worker < workers; ++worker)
        threads.emplace_back(
            sweep, total*worker/workers, total*(worker + 1)/workers,
            std::ref(tallies[worker]));
    for (auto& thread : threads)
        thread.join();
    Tally sum;
    for (const auto& tally : tallies) {
        sum.checked += tally.checked;
        if (sum.mismatches == 0 && tally.mismatches != 0)
            sum.first_mismatch = tally.first_mismatch;
        sum.mismatches += tally.mismatches;
        sum.max_ulp_vs_float_constant = std::max(
            sum.max_ulp_vs_float_constant, tally.max_ulp_vs_float_constant);
    }

    const float y00 = static_cast<float>(0.28209479177387814);
    double raw_double[1] = {0.28209479177387814};
    double normalized_double[1];
    symmetrix::spherical_harmonic_mace_normalize_values<1>(raw_double, normalized_double);
    std::printf(
        "{\"checked\": %llu, \"mismatches\": %llu, \"first_mismatch\": %u, "
        "\"max_ulp_vs_float_constant\": %lld, \"values_mismatches\": %llu, "
        "\"y00_bits\": %u, \"y00_float_constant_bits\": %u, "
        "\"y00_double\": %.17g}\n",
        static_cast<unsigned long long>(sum.checked),
        static_cast<unsigned long long>(sum.mismatches),
        sum.first_mismatch,
        static_cast<long long>(sum.max_ulp_vs_float_constant),
        static_cast<unsigned long long>(check_values()),
        to_bits(symmetrix::spherical_harmonic_mace_normalize(y00)),
        to_bits(float_constant(y00)),
        normalized_double[0]);
    return 0;
}
"""


def _compiler():
    for candidate in (os.environ.get("CXX"), "c++", "g++", "clang++"):
        if candidate and shutil.which(candidate):
            return shutil.which(candidate)
    pytest.skip("a C++17 host compiler (CXX, c++, g++ or clang++) is required")


def _float_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def test_direct_y_normalize_matches_retained_rounding_for_every_float(tmp_path):
    compiler = _compiler()
    stand_in = tmp_path / "include"
    stand_in.mkdir()
    (stand_in / "Kokkos_Core.hpp").write_text(_KOKKOS_STAND_IN)
    (stand_in / "macros.hpp").write_text(_MACROS_STAND_IN)
    driver = tmp_path / "normalize_sweep.cpp"
    driver.write_text(_DRIVER)
    executable = tmp_path / "normalize_sweep"
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-pthread",
            f"-I{stand_in}",
            f"-I{_SOURCE}",
            str(driver),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert compiled.returncode == 0, compiled.stderr
    completed = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=1200
    )
    summary = json.loads(completed.stdout)

    assert summary["checked"] == 2**32
    assert summary["mismatches"] == 0, hex(summary["first_mismatch"])
    assert summary["values_mismatches"] == 0
    # The float-constant product it replaces differs by at most one ulp.
    assert summary["max_ulp_vs_float_constant"] <= 1
    # Y00 of every edge: 1 exactly now, 1 + 1 ulp with the float constant.
    assert summary["y00_bits"] == _float_bits(1.0)
    assert summary["y00_float_constant_bits"] == _float_bits(1.0) + 1
    # FP64 keeps raw * C.
    assert summary["y00_double"] == pytest.approx(1.0, rel=1.0e-15)
