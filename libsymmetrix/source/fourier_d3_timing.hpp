#ifndef SYMMETRIX_FOURIER_D3_TIMING_HPP
#define SYMMETRIX_FOURIER_D3_TIMING_HPP

// Opt-in, synchronized stage timing for the reciprocal-space FourierD3 path
// and its stage diagnostics.
//
// The stages a benchmark needs are the stage boundaries the evaluation already
// has, so the timing seam is a counter beside each of them rather than a second
// implementation.  Two rules make the record usable:
//
//   * opt-in only.  A disabled timer records nothing, and a caller reads "no
//     timing was collected" from the absence of the record rather than from a
//     zero that looks like a measurement.
//   * synchronized.  A stage is bracketed by a Kokkos fence before it starts and
//     after it finishes, so the interval covers the device work the stage
//     launched instead of the time spent enqueueing it.  Without the fences a
//     GPU stage would be timed as microseconds of host submission.
//
// The unit is seconds, `count` is the number of completed executions of the
// stage, `last_seconds` the most recent one, and `cumulative_seconds` the sum
// over every completed one.  A throwing stage records nothing because it never
// reaches the closing fence and record step; a later failure does not erase a
// stage whose body already completed.
//
// Nothing here is a performance claim.  These numbers locate where an evaluation
// spends time; benchmark drivers measure performance with their own protocols.

#include <chrono>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

namespace symmetrix::fourier_d3 {

// One stage's timing record.
struct StageTiming {
    std::uint64_t count = 0;
    double last_seconds = 0.0;
    double cumulative_seconds = 0.0;
};

// Every stage the diagnostic record can carry.  One struct rather than one type
// per owner, so a stage's identity is its field name and merging a composed
// path's timings is a field-wise copy.  Each owner fills only the fields it
// times; a field with `count == 0` was never timed by anyone.
struct FourierD3StageTimings {
    // The evaluator's own stages.
    StageTiming preparation;
    StageTiming result_reduction;
    StageTiming result_export;
    // The composed forward path's stages.
    StageTiming forward_preparation;
    StageTiming assignment_forward_fft;
    StageTiming reciprocal_contraction;
    // The reverse orchestrator's stages.
    StageTiming coordination;
    StageTiming inverse_fft_gather_adjoint;
    StageTiming virial_and_strain;

};

// Accumulates one stage across every completed execution.
class StageTimer {
public:
    void record(const double seconds) noexcept
    {
        ++timing_.count;
        timing_.last_seconds = seconds;
        timing_.cumulative_seconds += seconds;
    }

    [[nodiscard]] const StageTiming& timing() const noexcept { return timing_; }

private:
    StageTiming timing_{};
};

// Runs `body` and records its synchronized wall time when `enabled`.  The return
// value of `body` is forwarded, so a timed stage and an untimed one are the same
// call site.
template <typename Body>
decltype(auto) time_stage(StageTimer& timer, const bool enabled, Body&& body)
{
    if (!enabled) {
        return std::forward<Body>(body)();
    }
    Kokkos::fence();
    const auto start = std::chrono::steady_clock::now();
    if constexpr (std::is_void_v<std::invoke_result_t<Body&>>) {
        std::forward<Body>(body)();
        Kokkos::fence();
        const auto stop = std::chrono::steady_clock::now();
        timer.record(std::chrono::duration<double>(stop - start).count());
    } else {
        auto result = std::forward<Body>(body)();
        Kokkos::fence();
        const auto stop = std::chrono::steady_clock::now();
        timer.record(std::chrono::duration<double>(stop - start).count());
        return result;
    }
}

}  // namespace symmetrix::fourier_d3

#endif  // SYMMETRIX_FOURIER_D3_TIMING_HPP
