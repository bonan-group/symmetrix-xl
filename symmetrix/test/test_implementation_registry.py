"""Tier, binding, and admission semantics of the implementation registry.

The registry is family-neutral data plus predicates, so its contract is
testable without any model, backend, or evaluator: a small native probe drives
synthetic descriptors through the planner and asserts the admission rules that
the family backends rely on.  In particular it pins the two rules a family
report can otherwise only be inspected for:

* a host backend admits a node only when an implementation *below* the
  generated tier serves it, so a generated-only host program is refused even
  with its artifact ready;
* an accelerator backend may admit a generated-only program, and then the
  report records the artifact readiness that qualified it, so a missing
  artifact is a refusal rather than a silent downgrade.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "libsymmetrix" / "source"

HARNESS = r"""
#include "implementation_registry.hpp"

#include <cstdio>
#include <string>

using symmetrix::execution::family_backend_admits;
using symmetrix::execution::ImplementationBinding;
using symmetrix::execution::ImplementationDescriptor;
using symmetrix::execution::ImplementationRegistry;
using symmetrix::execution::ImplementationTier;
using symmetrix::execution::OperatorKind;
using symmetrix::execution::OperatorNode;
using symmetrix::execution::Program;

static int failures = 0;

static void check(const bool condition, const char* what)
{
    if (!condition) {
        std::printf("FAIL %s\n", what);
        ++failures;
    }
}

static Program single_node_program(const OperatorKind kind, const std::string& key)
{
    Program program;
    program.family = "admission probe";
    program.precision = "float64";
    OperatorNode node;
    node.kind = kind;
    node.id = "probe." + key;
    node.capability_key = key;
    program.operators.push_back(std::move(node));
    return program;
}

static ImplementationDescriptor tier_descriptor(
    const std::string& id, const OperatorKind kind, const std::string& key,
    const ImplementationTier tier, const std::string& target,
    const bool requires_artifact)
{
    ImplementationDescriptor descriptor;
    descriptor.id = id;
    descriptor.kind = kind;
    descriptor.capability_key = key;
    descriptor.tier = tier;
    descriptor.target = target;
    if (requires_artifact)
        descriptor.qualified = [] (const auto& context) {
            return context.artifact_ready;
        };
    else
        descriptor.qualified = [] (const auto&) { return true; };
    return descriptor;
}

static void check_artifact_flag_consistency(const ImplementationBinding& binding)
{
    check(binding.artifact_required
            == (binding.qualified && !binding.portable_covered),
        "artifact_required is qualified && !portable_covered");
}

int main()
{
    const std::string host = "host";
    const std::string accelerator = "cuda";
    const auto kind = OperatorKind::tensor_product;
    const std::string key = "m0";
    const Program program = single_node_program(kind, key);

    // The planner binds the highest qualified tier, and a generated selection
    // does not remove the portable coverage the host backend requires.
    {
        ImplementationRegistry registry;
        registry.declare(tier_descriptor(
            "portable", kind, key, ImplementationTier::portable, host, false));
        registry.declare(tier_descriptor(
            "frozen", kind, key, ImplementationTier::frozen, host, false));
        registry.declare(tier_descriptor(
            "generated", kind, key, ImplementationTier::generated, host, true));
        auto report = symmetrix::execution::plan_program(
            program, registry, host, "float64", true, "artifact");
        check(report.bindings.size() == 1, "one binding");
        check(report.bindings[0].implementation_id == "generated",
            "generated outranks frozen and portable");
        check(report.bindings[0].portable_covered,
            "portable coverage survives a generated selection");
        check(!report.bindings[0].artifact_required,
            "a portable-covered node needs no artifact");
        check_artifact_flag_consistency(report.bindings[0]);
        check(family_backend_admits(report),
            "host admits a generated selection that keeps portable coverage");
        check(report.portable_node_count == 1, "portable coverage is counted");
    }
    {
        ImplementationRegistry registry;
        registry.declare(tier_descriptor(
            "portable", kind, key, ImplementationTier::portable, host, false));
        registry.declare(tier_descriptor(
            "frozen", kind, key, ImplementationTier::frozen, host, false));
        auto report = symmetrix::execution::plan_program(
            program, registry, host, "float64", false, {});
        check(report.bindings[0].implementation_id == "frozen",
            "frozen outranks portable");
        check(report.bindings[0].portable_covered, "frozen is portable coverage");
        check(!report.bindings[0].artifact_required, "frozen needs no artifact");
        check(family_backend_admits(report), "host admits portable coverage");
    }
    {
        ImplementationRegistry registry;
        registry.declare(tier_descriptor(
            "portable", kind, key, ImplementationTier::portable, host, false));
        auto report = symmetrix::execution::plan_program(
            program, registry, host, "float64", false, {});
        check(report.bindings[0].implementation_id == "portable",
            "portable is the fallback");
        check(family_backend_admits(report), "host admits the portable tier");
    }

    // A host program served only by a generated implementation is refused, even
    // with its artifact ready: portable coverage is the host invariant.
    {
        ImplementationRegistry registry;
        registry.declare(tier_descriptor(
            "generated", kind, key, ImplementationTier::generated, host, true));
        auto report = symmetrix::execution::plan_program(
            program, registry, host, "float64", true, "artifact");
        check(report.admitted, "the node is bound");
        check(report.bindings[0].qualified, "the generated node is qualified");
        check(!report.bindings[0].portable_covered, "no coverage below generated");
        check(report.bindings[0].artifact_required, "the artifact is required");
        check_artifact_flag_consistency(report.bindings[0]);
        check(!family_backend_admits(report),
            "host refuses a generated-only program");
    }

    // An accelerator program served only by a generated implementation is
    // admitted exactly when the artifact that qualified it is ready.
    {
        ImplementationRegistry registry;
        registry.declare(tier_descriptor("generated", kind, key,
            ImplementationTier::generated, accelerator, true));
        auto missing = symmetrix::execution::plan_program(
            program, registry, accelerator, "float64", false, {});
        check(!missing.admitted, "no artifact: nothing qualifies");
        check(!missing.artifact_ready, "the report records the readiness");
        check(!family_backend_admits(missing),
            "accelerator refuses a generated-only program without its artifact");

        auto ready = symmetrix::execution::plan_program(
            program, registry, accelerator, "float64", true, "artifact");
        check(ready.admitted, "with the artifact the node is bound");
        check(ready.artifact_ready, "the report records the readiness");
        check(ready.bindings[0].artifact_required, "the artifact is required");
        check_artifact_flag_consistency(ready.bindings[0]);
        check(family_backend_admits(ready),
            "accelerator admits a generated-only program with its artifact");
    }

    // An accelerator program with coverage below the generated tier needs no
    // artifact at all.
    {
        ImplementationRegistry registry;
        registry.declare(tier_descriptor(
            "portable", kind, key, ImplementationTier::portable, accelerator, false));
        registry.declare(tier_descriptor(
            "generated", kind, key, ImplementationTier::generated, accelerator, true));
        auto report = symmetrix::execution::plan_program(
            program, registry, accelerator, "float64", false, {});
        check(report.bindings[0].implementation_id == "portable",
            "the portable tier serves the node without an artifact");
        check(report.bindings[0].portable_covered, "portable coverage recorded");
        check(!report.bindings[0].artifact_required, "no artifact required");
        check(family_backend_admits(report),
            "accelerator admits portable coverage without an artifact");
    }

    // The executed implementation is the binding's statement of what the
    // dispatch runs.  A family whose dispatch gates are narrower than the
    // readiness flags supplies it, the registry records it, and an empty answer
    // leaves the selection in place.  The planned id and the tier keep naming
    // what qualified, so a dispatch that declines the qualified tier is visible
    // in the report rather than silently recomputed.
    {
        ImplementationRegistry registry;
        registry.declare(tier_descriptor(
            "portable", kind, key, ImplementationTier::portable, host, false));
        registry.declare(tier_descriptor(
            "frozen", kind, key, ImplementationTier::frozen, host, false));
        auto report = symmetrix::execution::plan_program(
            program, registry, host, "float64", false, {});
        check(report.bindings[0].implementation_id == "frozen",
            "the plan names the highest qualified tier");
        check(report.bindings[0].selected_implementation_id == "frozen",
            "without a resolver the selection is what executes");

        registry.set_selected_implementation_resolver(
            [] (const auto&, const ImplementationBinding&) {
                return std::string("portable");
            });
        report = symmetrix::execution::plan_program(
            program, registry, host, "float64", false, {});
        check(report.bindings[0].implementation_id == "frozen",
            "a resolver does not move the planned tier");
        check(report.bindings[0].selected_implementation_id == "portable",
            "the resolver's answer is the executed implementation");

        registry.set_selected_implementation_resolver(
            [] (const auto&, const ImplementationBinding&) {
                return std::string();
            });
        report = symmetrix::execution::plan_program(
            program, registry, host, "float64", false, {});
        check(report.bindings[0].selected_implementation_id == "frozen",
            "an empty resolver answer leaves the selection in place");
    }

    // An unbound node names the target it had no implementation for.
    {
        ImplementationRegistry registry;
        auto report = symmetrix::execution::plan_program(
            program, registry, accelerator, "float64", false, {});
        check(report.bindings.size() == 1, "the unbound node is still reported");
        check(!report.bindings[0].qualified, "the node is unbound");
        check(report.bindings[0].selected_implementation_id.empty(),
            "an unbound node states no executed implementation");
        check(!report.admitted, "the report is not admitted");
        check(!family_backend_admits(report), "an unbound node is refused");
        check(report.bindings[0].reason.find(accelerator) != std::string::npos,
            "the reason names the target");
        check_artifact_flag_consistency(report.bindings[0]);
    }

    if (failures != 0) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("registry admission probe: all checks passed\n");
    return 0;
}
"""


def test_implementation_registry_tier_binding_and_admission(tmp_path):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++ compiler is required for the registry probe")
    source = tmp_path / "registry_probe.cpp"
    source.write_text(HARNESS)
    binary = tmp_path / "registry_probe"
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
    assert "FAIL" not in completed.stdout
