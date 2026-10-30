// The one serialization of the FourierD3 diagnostic records.
//
// The private seams each report their own usage, and the runtime merges them into
// one record; if every seam rendered the same structure its own way, the merged
// record would depend on which seam a field came from.  The builders here are
// therefore shared by every binding translation unit that publishes a usage,
// plan, or timing record, so a field means the same thing wherever it appears.
//
// Nothing here evaluates anything, and nothing here is a public capability: the
// header is included by the private FourierD3 binding translation units only.

#ifndef SYMMETRIX_FOURIER_D3_DIAGNOSTICS_REPORT_HPP
#define SYMMETRIX_FOURIER_D3_DIAGNOSTICS_REPORT_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "fourier_d3_storage.hpp"
#include "fourier_d3_timing.hpp"

namespace py = pybind11;

namespace symmetrix::fourier_d3::binding {

[[nodiscard]] inline py::dict buffer_report(const BufferUsage& buffer)
{
    py::dict report;
    report["name"] = buffer.name;
    report["active_elements"] = buffer.active_elements;
    report["capacity_elements"] = buffer.capacity_elements;
    report["active_bytes"] = buffer.active_bytes;
    report["capacity_bytes"] = buffer.capacity_bytes;
    report["allocation_id"] = buffer.allocation_id;
    return report;
}

[[nodiscard]] inline py::list buffer_reports(const std::vector<BufferUsage>& buffers)
{
    py::list reports;
    for (const BufferUsage& buffer : buffers) {
        reports.append(buffer_report(buffer));
    }
    return reports;
}

// The retained allocation a plan holds, or None when it holds none.  A released
// plan renders as None rather than as a zeroed allocation, so "holds nothing" is
// distinguishable from "holds an empty allocation"; an owned one carries an
// explicit `owns` flag so a caller reads ownership from the record instead of
// inferring it from the shape of a zero.
[[nodiscard]] inline py::object retained_allocation_report(
    const RetainedAllocationReport& report)
{
    if (!report.owns) {
        return py::none();
    }
    py::dict rendered;
    rendered["owns"] = true;
    rendered["allocation_owners"] = report.allocation_owners;
    rendered["generation"] = report.generation;
    rendered["real_elements"] = report.real_elements;
    rendered["spectrum_elements"] = report.spectrum_elements;
    rendered["retained_bytes"] = report.retained_bytes;
    rendered["real_view_use_count"] = report.real_view_use_count;
    rendered["spectrum_view_use_count"] = report.spectrum_view_use_count;
    return rendered;
}

[[nodiscard]] inline py::dict identity_report(const PlanIdentity& identity)
{
    py::dict report;
    report["backend"] = identity.backend;
    report["precision"] = identity.precision;
    report["provider"] = identity.provider;
    return report;
}

[[nodiscard]] inline py::dict plan_usage_report(const FftPlanUsage& usage)
{
    py::dict report;
    report["role"] = std::string(plan_role_token(usage.role));
    report["held"] = usage.held;
    report["builds"] = usage.builds;
    report["reuses"] = usage.reuses;
    report["releases"] = usage.releases;
    report["key_rebuilds"] = usage.key_rebuilds;
    report["identity_rebuilds"] = usage.identity_rebuilds;
    report["allocation_generation"] = usage.allocation_generation;
    report["key"] = usage.key;
    report["identity"] = identity_report(usage.identity);
    report["allocation"] = retained_allocation_report(usage.allocation);
    return report;
}

[[nodiscard]] inline py::list plan_usage_reports(
    const std::array<FftPlanUsage, plan_role_count>& plans)
{
    py::list reports;
    for (const FftPlanUsage& plan : plans) {
        reports.append(plan_usage_report(plan));
    }
    return reports;
}

// One stage's timing in seconds, with the count of completed executions.
[[nodiscard]] inline py::dict stage_timing_report(const StageTiming& timing)
{
    py::dict report;
    report["last_seconds"] = timing.last_seconds;
    report["cumulative_seconds"] = timing.cumulative_seconds;
    report["count"] = timing.count;
    return report;
}

// Every stage, so a caller reads one mapping and never has to know which owner
// timed which stage.
[[nodiscard]] inline py::dict stage_timings_report(const FourierD3StageTimings& timings)
{
    py::dict report;
    report["preparation"] = stage_timing_report(timings.preparation);
    report["forward_preparation"] = stage_timing_report(timings.forward_preparation);
    report["coordination"] = stage_timing_report(timings.coordination);
    report["assignment_forward_fft"] =
        stage_timing_report(timings.assignment_forward_fft);
    report["reciprocal_contraction"] =
        stage_timing_report(timings.reciprocal_contraction);
    report["inverse_fft_gather_adjoint"] =
        stage_timing_report(timings.inverse_fft_gather_adjoint);
    report["virial_and_strain"] = stage_timing_report(timings.virial_and_strain);
    report["result_reduction"] = stage_timing_report(timings.result_reduction);
    report["result_export"] = stage_timing_report(timings.result_export);
    return report;
}

}  // namespace symmetrix::fourier_d3::binding

#endif  // SYMMETRIX_FOURIER_D3_DIAGNOSTICS_REPORT_HPP
