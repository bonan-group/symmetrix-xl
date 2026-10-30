"""Boundary-aware runtime policy for the single-system POLAR evaluator."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "libsymmetrix" / "source"

HARNESS = r"""
#include "long_range_solver_policy.hpp"

#include <array>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

using namespace symmetrix::execution::long_range;

static int failures = 0;

static void check(const bool condition, const char* message)
{
    if (!condition) {
        std::printf("FAIL %s\n", message);
        ++failures;
    }
}

static void check_route(
    const char* requested,
    const std::array<int, 3>& pbc,
    const char* expected_solver,
    const char* expected_reason)
{
    const auto decision = resolve_long_range_solver(
        parse_long_range_solver_request(requested), pbc, 1, 1);
    check(decision.accepted, "route is admitted");
    check(to_string(decision.resolved) == expected_solver,
        "resolved solver matches");
    check(decision.reason == expected_reason, "route reason matches");
}

static LongRangeSolverCapabilities pme_capabilities(
    const bool provider, const bool backend,
    const std::size_t nodes = 0, const std::size_t modes = 0)
{
    LongRangeSolverCapabilities capabilities;
    capabilities.pme_fft_provider_available = provider;
    capabilities.pme_backend_qualified = backend;
    capabilities.automatic_nodes = nodes;
    capabilities.automatic_reciprocal_modes = modes;
    return capabilities;
}

int main()
{
    check_route("auto", {0, 0, 0}, "realspace",
        "fully_nonperiodic_input");
    check_route("auto", {1, 0, 0}, "direct_reciprocal",
        "periodic_input_delivery_1");
    check_route("periodic", {0, 0, 0}, "direct_reciprocal",
        "periodic_compatibility_override");
    check_route("periodic", {0, 1, 0}, "direct_reciprocal",
        "periodic_compatibility_override");
    check_route("direct_reciprocal", {0, 0, 0}, "direct_reciprocal",
        "explicit_direct_reciprocal");
    check_route("realspace", {0, 0, 0}, "realspace",
        "explicit_realspace");

    const auto periodic_realspace = resolve_long_range_solver(
        LongRangeSolverRequest::realspace, {0, 0, 1}, 1, 1);
    check(!periodic_realspace.accepted,
        "forced realspace rejects periodic input");
    check(periodic_realspace.reason == "realspace_requires_fully_nonperiodic_input",
        "periodic rejection reason matches");

    const auto unavailable_pme = resolve_long_range_solver(
        LongRangeSolverRequest::pme, {1, 1, 1}, 1, 1);
    check(!unavailable_pme.accepted, "pme requires an FFT provider");
    check(unavailable_pme.reason == "pme_fft_provider_unavailable",
        "provider rejection reason matches");
    check(unavailable_pme.error.find("no particle-mesh FFT provider")
            != std::string::npos,
        "provider rejection error is stable");

    const auto unqualified_pme = resolve_long_range_solver(
        LongRangeSolverRequest::pme, {1, 1, 1}, 1, 1,
        pme_capabilities(true, false));
    check(!unqualified_pme.accepted, "pme requires a qualified backend");
    check(unqualified_pme.reason == "pme_backend_not_qualified",
        "backend rejection reason matches");

    const auto partial_pme = resolve_long_range_solver(
        LongRangeSolverRequest::pme, {1, 1, 0}, 1, 1,
        pme_capabilities(true, true));
    check(!partial_pme.accepted, "pme rejects partial periodicity");
    check(partial_pme.reason == "pme_requires_fully_periodic_input",
        "partial-periodic rejection reason matches");
    check(partial_pme.error.find("PBC TTT") != std::string::npos,
        "partial-periodic rejection error is stable");

    const auto high_order_pme = resolve_long_range_solver(
        LongRangeSolverRequest::pme, {1, 1, 1}, 2, 1,
        pme_capabilities(true, true));
    check(!high_order_pme.accepted, "pme rejects source l>1");
    check(high_order_pme.reason == "pme_supports_l_at_most_1",
        "pme angular-order rejection reason matches");
    const auto high_receiver_order_pme = resolve_long_range_solver(
        LongRangeSolverRequest::pme, {1, 1, 1}, 1, 2,
        pme_capabilities(true, true));
    check(!high_receiver_order_pme.accepted, "pme rejects receiver l>1");
    check(high_receiver_order_pme.error.find("l<=1") != std::string::npos,
        "pme angular-order rejection error is stable");

    const auto admitted_pme = resolve_long_range_solver(
        LongRangeSolverRequest::pme, {1, 1, 1}, 1, 1,
        pme_capabilities(true, true));
    check(admitted_pme.accepted, "qualified fully periodic pme is admitted");
    check(admitted_pme.resolved == LongRangeSolver::pme,
        "qualified pme resolves to pme");
    check(admitted_pme.reason == "explicit_pme",
        "qualified pme reason matches");

    const auto below_crossover = resolve_long_range_solver(
        LongRangeSolverRequest::automatic, {1, 1, 1}, 1, 1,
        pme_capabilities(true, true, 9999, 4000));
    check(below_crossover.resolved == LongRangeSolver::direct_reciprocal,
        "automatic route stays direct below forty million atom-mode pairs");
    const auto at_crossover = resolve_long_range_solver(
        LongRangeSolverRequest::automatic, {1, 1, 1}, 1, 1,
        pme_capabilities(true, true, 10000, 4000));
    check(at_crossover.resolved == LongRangeSolver::pme,
        "automatic route selects pme at forty million atom-mode pairs");
    check(at_crossover.reason == "automatic_pme_atom_mode_pairs",
        "automatic pme reason matches");
    const auto overflow_free = resolve_long_range_solver(
        LongRangeSolverRequest::automatic, {1, 1, 1}, 1, 1,
        pme_capabilities(true, true,
            std::numeric_limits<std::size_t>::max(), 2));
    check(overflow_free.resolved == LongRangeSolver::pme,
        "automatic threshold comparison does not multiply overflowing factors");
    const auto explicit_direct_above_crossover = resolve_long_range_solver(
        LongRangeSolverRequest::direct_reciprocal, {1, 1, 1}, 1, 1,
        pme_capabilities(true, true, 10000, 4000));
    check(explicit_direct_above_crossover.resolved
            == LongRangeSolver::direct_reciprocal,
        "explicit direct override wins above the automatic crossover");
    const auto unavailable_auto = resolve_long_range_solver(
        LongRangeSolverRequest::automatic, {1, 1, 1}, 1, 1,
        pme_capabilities(false, true, 10000, 4000));
    check(unavailable_auto.resolved == LongRangeSolver::direct_reciprocal,
        "automatic route stays direct without an FFT provider");
    const auto partial_auto = resolve_long_range_solver(
        LongRangeSolverRequest::automatic, {1, 1, 0}, 1, 1,
        pme_capabilities(true, true, 10000, 4000));
    check(partial_auto.resolved == LongRangeSolver::direct_reciprocal,
        "automatic route keeps partial periodicity on direct reciprocal");

    const auto source_l2 = resolve_long_range_solver(
        LongRangeSolverRequest::automatic, {0, 0, 0}, 2, 1);
    check(!source_l2.accepted, "realspace rejects source l>1");
    check(source_l2.reason == "realspace_supports_l_at_most_1",
        "source admission reason matches");
    const auto receiver_l2 = resolve_long_range_solver(
        LongRangeSolverRequest::realspace, {0, 0, 0}, 1, 2);
    check(!receiver_l2.accepted, "realspace rejects receiver l>1");

    bool unknown_rejected = false;
    try {
        (void)parse_long_range_solver_request("ewald");
    } catch (const std::invalid_argument&) {
        unknown_rejected = true;
    }
    check(unknown_rejected, "unknown solver request is rejected");

    if (failures != 0)
        return 1;
    std::printf("long-range solver policy probe: all checks passed\n");
    return 0;
}
"""


def test_delivery_1_solver_policy_matrix_and_admission(tmp_path):
    """The policy seam rejects invalid routes and admits only qualified PME."""

    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++ compiler is required for the solver-policy probe")
    source = tmp_path / "long_range_solver_policy_probe.cpp"
    source.write_text(HARNESS)
    binary = tmp_path / "long_range_solver_policy_probe"
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            f"-I{SOURCE_ROOT}",
            "-o",
            str(binary),
            str(source),
        ],
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stderr
    completed = subprocess.run([str(binary)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout == ("long-range solver policy probe: all checks passed\n")
