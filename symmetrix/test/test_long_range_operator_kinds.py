"""Declarative-layer contract of the long-range operator kind group.

The kind group (``libsymmetrix/source/long_range_operator_kinds.hpp``) is
family-neutral data plus predicates, so its contract is testable without any
model, backend, or evaluator, exactly like the registry itself: a small
native probe drives the declarations through the planner and asserts the
rules the future PolarMACE family will rely on.  In particular it pins:

* every kind's contract data validates -- the shipped MACE-POLAR-1 shape
  (the checkpoint-facts M entry) passes, and each garbage mutation (wrong
  precision policy, inconsistent dimensions, an out-of-support max_l, an
  unknown correction mode) rejects with an error that names the rule;
* registry integration -- the kinds appear in the binding-report vocabulary
  with portable coverage per node (the host admission invariant), and a
  query for an unknown long-range kind fails cleanly instead of falling back
  to ``custom`` or crashing;
* the declaration-time capability state -- forward everywhere, the
  experimental cell adjoint (the reciprocal-space stress) on exactly the
  cell-dependent kinds, the external-field coupling on the projection
  kind, and the global scatter-and-broadcast class on the Fukui kind;
* the capacity hooks stay in the manifest's long_range vocabulary (N_k,
  table_bytes, correction mode, rebuild trigger).

Runs in the plain repository environment: a C++20 compiler and nothing else.
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
#include "long_range_operator_kinds.hpp"
#include "mace_polar_program.hpp"

#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

using symmetrix::execution::ImplementationBinding;
using symmetrix::execution::ImplementationRegistry;
using symmetrix::execution::OperatorKind;
using symmetrix::execution::OperatorNode;
using symmetrix::execution::Program;
using symmetrix::execution::family_backend_admits;
using symmetrix::execution::operator_kind_name;
using symmetrix::execution::plan_program;
using symmetrix::execution::long_range::LongRangeCapacity;
using symmetrix::execution::long_range::LongRangeChannelLayout;
using LongRangeContract =
    symmetrix::execution::polar::PolarLongRangeContract;
using symmetrix::execution::long_range::LongRangeKindSpec;
using symmetrix::execution::long_range::correction_mode_vocabulary;
using symmetrix::execution::long_range::declare_long_range_implementations;
using symmetrix::execution::long_range::derivative_mode_vocabulary;
using symmetrix::execution::long_range::find_long_range_kind;
using symmetrix::execution::long_range::long_range_contract_fingerprint;
using symmetrix::execution::long_range::long_range_dtype_itemsize;
using symmetrix::execution::long_range::long_range_kind_specs;
using symmetrix::execution::long_range::long_range_operator_nodes;
using symmetrix::execution::long_range::long_range_primitive_operator_nodes;
using symmetrix::execution::long_range::make_long_range_capacity;
using symmetrix::execution::long_range::validate_long_range_capacity;
using symmetrix::execution::long_range::validate_long_range_contract;
using symmetrix::execution::long_range::validate_long_range_primitive_node_shapes;
using symmetrix::execution::polar::PolarSchedule;
using symmetrix::execution::polar::PolarScheduleEntry;
using symmetrix::execution::polar::compile_polar_program;
using symmetrix::execution::polar::polar_execution_schedule;
using symmetrix::execution::polar::polar_implementation_registry;
using symmetrix::execution::polar::polar_region_declared;
using symmetrix::execution::polar::validate_polar_long_range_node_shapes;
using symmetrix::execution::polar::validate_polar_program;

static int failures = 0;

static void check(const bool condition, const std::string& what)
{
    if (!condition) {
        std::printf("FAIL %s\n", what.c_str());
        ++failures;
    }
}

// The shipped MACE-POLAR-1 M checkpoint shape (symmetrix/test/data/polar/
// checkpoint_facts.json, models.M): sigma set (1.5, 3.0), both max_l at 1,
// the model's own k-space cutoff, two recursion steps, 8/8/16 dimensions.
static LongRangeContract checkpoint_m_contract()
{
    LongRangeContract contract;
    contract.kspace_cutoff = 1.8467166423797607;
    contract.density_max_l = 1;
    contract.density_smearing_width = 1.5;
    contract.feature_max_l = 1;
    contract.feature_smearing_widths = {1.5, 3.0};
    contract.charges_dimension = 8;
    contract.field_dimension = 8;
    contract.potential_dimension = 16;
    contract.field_norm_factor = 1.0;
    contract.recursion_step_count = 2;
    contract.spin_channel_count = 2;
    // The shipped M checkpoint's self-interaction policy: the k-space energy
    // carries its overlaps, the projected features do not carry the feature
    // self-terms (checkpoint_facts.json, models.M).
    contract.include_electrostatic_self_interaction = true;
    contract.field_si = false;
    return contract;
}

static void check_buffer_contract(
    const symmetrix::execution::BufferContract& buffer)
{
    check(!buffer.name.empty(), "buffers are named");
    check(buffer.dtype == "float64" || buffer.dtype == "float32"
        || buffer.dtype == "int32", "buffer dtype is declared: " + buffer.name);
    check(!buffer.shape.empty(), "buffer shape is non-empty: " + buffer.name);
    check(buffer.layout.order == "node_major" || buffer.layout.order == "k_major"
        || buffer.layout.order == "graph_major",
        "layout order is in the group vocabulary: " + buffer.name);
}

int main()
{
    // The vocabularies this probe asserts are also the Python side's
    // declaration constants (symmetrix/polar_long_range_kinds.py), so the
    // probe prints them in a stable form: the Python test pins the two
    // sides equal (tokens and declaration order) instead of each side
    // checking only its own literals while both drift.
    for (const LongRangeKindSpec& spec : long_range_kind_specs())
        std::printf("vocabulary kind %s\n",
            std::string(spec.name).c_str());
    for (const std::string_view mode : correction_mode_vocabulary)
        std::printf("vocabulary correction_mode %s\n",
            std::string(mode).c_str());
    for (const std::string_view mode : derivative_mode_vocabulary)
        std::printf("vocabulary derivative_mode %s\n",
            std::string(mode).c_str());
    const LongRangeContract good = checkpoint_m_contract();
    const std::string validation = validate_long_range_contract(good);
    check(validation.empty(), "the checkpoint-M contract validates: " + validation);

    // -----------------------------------------------------------------
    // Node contracts
    // -----------------------------------------------------------------
    const std::vector<OperatorNode> nodes = polar_long_range_operator_nodes(good);
    check(nodes.size() == long_range_kind_specs().size(),
        "one node per declared kind");
    std::map<std::string, const OperatorNode*> by_kind;
    for (const OperatorNode& node : nodes) {
        const std::string kind = operator_kind_name(node);
        check(by_kind.find(kind) == by_kind.end(),
            "kind appears once: " + kind);
        by_kind[kind] = &node;
        for (const auto& input : node.inputs)
            check_buffer_contract(input);
        for (const auto& output : node.outputs)
            check_buffer_contract(output);
        // Runtime rows stay the leading dynamic extent everywhere except
        // the fully static constants (the 3x3 cell, the norm vector).
        for (const auto& buffer : node.inputs) {
            if (buffer.shape.front() == 0)
                continue;
            check(buffer.name == "cell" || buffer.name == "field_feature_norms",
                "only the cell and norm constants are fully static: "
                + buffer.name);
        }
    }
    check(by_kind.size() == 9, "the plan's kind set is complete");
    for (const char* kind : {"kspace_geometry_cache", "kspace_coulomb_features",
        "kspace_coulomb_energy", "self_interaction_linear",
        "external_field_projection", "fukui_normalize", "dipole_readout",
        "analytic_correction_energy", "analytic_correction_features"}) {
        check(by_kind.find(kind) != by_kind.end(),
            std::string("kind is declared: ") + kind);
    }
    // The layout orders each kind's buffers declare are checked as data by
    // ``check_buffer_contract`` above and, on the family side, by the
    // conformance battery reading the compiled program's own buffer
    // contracts; the probe deliberately prints no per-kind layout lines, so
    // no consumer can mistake a printout for a second authority.

    // The cache is the capacity owner and the shared-table source: the
    // features and energy nodes consume its outputs under their production
    // names with identical contracts.
    const OperatorNode& cache = *by_kind["kspace_geometry_cache"];
    check(cache.id == "polar.long_range.kspace_geometry_cache",
        "node ids carry the family prefix");
    std::map<std::string, const symmetrix::execution::BufferContract*> produced;
    for (const auto& output : cache.outputs)
        produced[output.name] = &output;
    for (const char* name : {"cos_table", "sin_table"}) {
        check(produced.find(name) != produced.end(),
            std::string("the cache produces ") + name);
        // The declared dtype is the delivered one: the tier-1 kernels carry
        // every table in double, and fp32 quantization of them misses the
        // group's own parity gate (the contract comment records the
        // measurement), so the declaration follows the implementation.
        check(produced[name]->dtype == good.table_dtype,
            std::string("the dense tables are the contract's dtype: ") + name);
        check(produced[name]->dtype == "float64",
            std::string("the delivered dense tables are fp64: ") + name);
        check(produced[name]->shape == std::vector<std::size_t>({0, 0}),
            std::string("both table extents are dynamic: ") + name);
    }
    for (const char* kind : {"kspace_coulomb_features", "kspace_coulomb_energy"}) {
        const OperatorNode& consumer = *by_kind[kind];
        for (const char* name : {"cos_table", "sin_table"}) {
            const auto it = std::find_if(consumer.inputs.begin(),
                consumer.inputs.end(),
                [name] (const auto& buffer) { return buffer.name == name; });
            check(it != consumer.inputs.end(),
                std::string(kind) + " consumes " + name);
            if (it != consumer.inputs.end()) {
                check(it->dtype == produced[name]->dtype
                    && it->shape == produced[name]->shape
                    && it->layout.order == produced[name]->layout.order,
                    std::string(kind) + " consumes the cache contract of " + name);
            }
        }
    }

    // The correction mode is runtime data: both correction kinds consume the
    // per-graph pbc rows, never a mode parameter.
    for (const char* kind : {"analytic_correction_energy",
        "analytic_correction_features"}) {
        const OperatorNode& correction = *by_kind[kind];
        const auto pbc = std::find_if(correction.inputs.begin(),
            correction.inputs.end(),
            [] (const auto& buffer) { return buffer.name == "graph_pbc"; });
        check(pbc != correction.inputs.end(),
            std::string(kind) + " selects its mode from graph pbc data");
        bool mode_parameter = false;
        for (const auto& [name, value] : correction.parameters) {
            (void)value;
            if (name.find("mode") != std::string::npos)
                mode_parameter = true;
        }
        check(!mode_parameter, std::string(kind) + " carries no mode parameter");
    }

    // The Fukui kind is the global scatter-and-broadcast declaration, and its
    // own derivative surface is the scatter of the broadcast gradient.
    const LongRangeKindSpec fukui = *find_long_range_kind("fukui_normalize");
    check(fukui.capabilities.global_scatter_broadcast,
        "fukui_normalize declares the scatter-and-broadcast class");
    check(fukui.capabilities.forward && fukui.capabilities.normalization_adjoint
        && !fukui.capabilities.position_adjoint
        && !fukui.capabilities.cell_adjoint,
        "the Fukui kind serves forward and its normalization adjoint only "
        "(no position or cell dependence)");
    const LongRangeKindSpec projection =
        *find_long_range_kind("external_field_projection");
    check(projection.capabilities.serves_external_field,
        "external_field_projection declares the field coupling");
    // The capability flags follow the implemented adjoint surfaces exactly:
    // position adjoints where the forward consumes positions, the
    // experimental cell adjoint (the reciprocal-space stress) where the
    // forward carries cell-dependent terms, and the normalization adjoint on
    // the global scatter-and-broadcast kind.
    int cell_adjoint_kinds = 0;
    int position_adjoint_kinds = 0;
    for (const LongRangeKindSpec& spec : long_range_kind_specs()) {
        check(spec.capabilities.forward, "every kind serves forward");
        const std::string name(spec.name);
        const bool expects_position = name == "kspace_geometry_cache"
            || name == "kspace_coulomb_features"
            || name == "kspace_coulomb_energy"
            || name == "external_field_projection"
            || name == "dipole_readout"
            || name == "analytic_correction_energy"
            || name == "analytic_correction_features";
        check(spec.capabilities.position_adjoint == expects_position,
            "the position adjoint is declared exactly on the "
            "position-dependent kinds: " + name);
        const bool expected_cell = name == "kspace_geometry_cache"
            || name == "kspace_coulomb_features"
            || name == "kspace_coulomb_energy"
            || name == "analytic_correction_energy"
            || name == "analytic_correction_features";
        check(spec.capabilities.cell_adjoint == expected_cell,
            std::string("the cell adjoint is declared exactly on the "
                        "cell-dependent kinds: ")
                + std::string(spec.name));
        const bool expects_normalization = name == "fukui_normalize";
        check(spec.capabilities.normalization_adjoint == expects_normalization,
            "the normalization adjoint is declared exactly on the "
            "scatter-and-broadcast kind: " + name);
        if (spec.capabilities.cell_adjoint)
            ++cell_adjoint_kinds;
        if (spec.capabilities.position_adjoint)
            ++position_adjoint_kinds;
    }
    check(cell_adjoint_kinds == 5,
        "five kinds declare the experimental cell adjoint");
    check(position_adjoint_kinds == 7,
        "seven kinds declare a position adjoint");

    // Scalar contract parameters land on the kinds that consume them.
    {
        const OperatorNode& features = *by_kind["kspace_coulomb_features"];
        bool half_weight = false;
        for (const auto& [name, value] : features.parameters)
            if (name == "half_sphere_weight" && value == 2.0)
                half_weight = true;
        check(half_weight, "the features kind carries the half-sphere weight");
        const OperatorNode& energy = *by_kind["kspace_coulomb_energy"];
        bool self_flag = false;
        for (const auto& [name, value] : energy.parameters)
            if (name == "include_self_interaction" && value == 1.0)
                self_flag = true;
        check(self_flag,
            "the energy kind carries the contract's self-interaction flag");
        // The policy is contract data, so flipping it flips the stamped value
        // on both nodes that consume it.
        LongRangeContract reversed = good;
        reversed.include_electrostatic_self_interaction = false;
        reversed.field_si = true;
        const std::vector<OperatorNode> flipped =
            polar_long_range_operator_nodes(reversed);
        const auto flipped_value = [] (const std::vector<OperatorNode>& nodes,
            const std::string_view kind, const std::string_view name) {
            for (const auto& node : nodes)
                if (operator_kind_name(node) == kind)
                    for (const auto& [parameter, value] : node.parameters)
                        if (parameter == name)
                            return value;
            return -1.0;
        };
        check(flipped_value(flipped, "kspace_coulomb_energy",
                "include_self_interaction") == 0.0,
            "the energy flag follows the contract");
        check(flipped_value(flipped, "self_interaction_linear",
                "include_energy_self_interaction") == 0.0,
            "the energy self-term flag follows the contract");
        check(flipped_value(flipped, "self_interaction_linear",
                "include_feature_self_interaction") == 1.0,
            "the feature self-term flag follows the contract");
        const OperatorNode& correction =
            *by_kind["analytic_correction_energy"];
        bool box_constant = false;
        for (const auto& [name, value] : correction.parameters)
            if (name == "cubic_madelung" && value == 2.837297)
                box_constant = true;
        check(box_constant, "the box term carries the cubic Madelung constant");
    }

    // -----------------------------------------------------------------
    // Shape validation (the probe-extent pass)
    // -----------------------------------------------------------------
    // The pass the family's program assembly runs before it stamps an
    // identity: the declared extents, the basis channel widths (including the
    // self-interaction kind's two adjoint seed surfaces), and the shared
    // cache tables are checked against the contract data rather than trusted.
    {
        const std::string reason =
            validate_polar_long_range_node_shapes(nodes, good);
        check(reason.empty(),
            "the declared node contracts are shape-consistent: " + reason);
        const auto self_term_columns = [] (std::vector<OperatorNode> mutated) {
            for (auto& node : mutated)
                if (node.kind == OperatorKind::self_interaction_linear)
                    for (auto& output : node.outputs)
                        if (output.name == "energy_self_fields")
                            output.shape = {0, 7};
            return mutated;
        };
        const auto with_buffer = [] (std::vector<OperatorNode> mutated,
            const OperatorKind kind, const std::string& name,
            auto&& mutate) {
            for (auto& node : mutated)
                if (node.kind == kind)
                    for (auto& buffer : node.inputs)
                        if (buffer.name == name)
                            mutate(buffer);
            return mutated;
        };
        {
            std::vector<OperatorNode> mutated = nodes;
            mutated.front().outputs.front().shape = {5, 3};
            const std::string reason =
                validate_polar_long_range_node_shapes(mutated, good);
            check(reason.find("static leading extent") != std::string::npos,
                "a static leading extent rejects: " + reason);
        }
        {
            const std::string reason = validate_polar_long_range_node_shapes(
                self_term_columns(nodes), good);
            check(reason.find("energy_self_fields") != std::string::npos
                    && reason.find("density channel width") != std::string::npos,
                "a wrong self-term seed column rejects: " + reason);
        }
        {
            const std::string reason = validate_polar_long_range_node_shapes(
                with_buffer(nodes, OperatorKind::kspace_coulomb_features,
                    "cos_table",
                    [] (auto& buffer) { buffer.dtype = "float32"; }),
                good);
            check(reason.find("contract the cache does not produce")
                    != std::string::npos,
                "a consumer whose shared table contract differs rejects: "
                + reason);
        }
        {
            std::vector<OperatorNode> mutated = nodes;
            mutated.pop_back();
            const std::string reason =
                validate_polar_long_range_node_shapes(mutated, good);
            check(reason.find("missing the declared kind") != std::string::npos,
                "an incomplete node set rejects: " + reason);
        }
        {
            LongRangeContract narrower = good;
            narrower.feature_smearing_widths = {1.5};
            narrower.field_dimension = 4;
            narrower.potential_dimension = 8;
            const std::string reason =
                validate_polar_long_range_node_shapes(nodes, narrower);
            check(reason.find("feature channel width") != std::string::npos,
                "a contract whose feature width moved rejects: " + reason);
        }
        {
            LongRangeContract wider = good;
            wider.density_max_l = 0;
            wider.charges_dimension = 2;
            const std::string reason =
                validate_polar_long_range_node_shapes(nodes, wider);
            check(reason.find("density channel width") != std::string::npos,
                "a contract whose density width moved rejects: " + reason);
        }
    }

    // -----------------------------------------------------------------
    // Registry integration
    // -----------------------------------------------------------------
    Program program;
    program.family = "polar-long-range-probe";
    program.precision = "float64";
    program.operators = nodes;
    ImplementationRegistry registry;
    declare_long_range_implementations(registry, "host", good);
    check(registry.size() == long_range_kind_specs().size(),
        "one descriptor per kind");

    // The registry's stress capability follows the experimental cell
    // adjoint declaration exactly (the descriptor-level vocabulary the
    // family surface reads).
    for (const auto& descriptor : registry.implementations()) {
        const std::string name = descriptor.capability_key;
        const bool expected = name == "kspace_geometry_cache"
            || name == "kspace_coulomb_features"
            || name == "kspace_coulomb_energy"
            || name == "analytic_correction_energy"
            || name == "analytic_correction_features";
        check(descriptor.capabilities.stress == expected,
            "the stress capability follows the cell adjoint: " + name);
        check(!descriptor.capabilities.parameter_gradients,
            "no parameter gradients are declared: " + name);
    }

    const auto report = plan_program(program, registry, "host", "float64");
    check(report.admitted, "every long-range node is bound");
    check(report.node_count == nodes.size(), "the report covers every node");
    check(report.portable_node_count == nodes.size(),
        "portable coverage is per node (the host admission invariant)");
    check(family_backend_admits(report), "the host admits the group");
    check(report.bindings.size() == nodes.size(), "one binding per node");
    std::map<std::string, const ImplementationBinding*> vocabulary;
    for (const auto& binding : report.bindings) {
        const std::string kind = operator_kind_name(binding.kind);
        vocabulary[kind] = &binding;
        check(binding.qualified, "the node is qualified: " + kind);
        check(binding.portable_covered, "portable coverage: " + kind);
        check(!binding.artifact_required, "no artifact is required: " + kind);
        check(binding.tier == symmetrix::execution::ImplementationTier::portable,
            "the portable tier is the production path: " + kind);
        check(binding.implementation_id == "portable_long_range_" + kind,
            "the implementation id names the kind: " + kind);
        check(binding.precision.accumulator == "float64",
            "the accumulator policy is fp64: " + kind);
        check(binding.precision.compute == "float64",
            "the compute precision is the plan's: " + kind);
    }
    check(vocabulary.size() == 9,
        "the binding-report vocabulary lists every kind");

    // A plan on a target the group is not declared for is refused with the
    // target named, not silently degraded.
    const auto cuda_report = plan_program(program, registry, "cuda", "float64");
    check(!cuda_report.admitted, "an undeclared target leaves nodes unbound");
    check(cuda_report.bindings.size() == nodes.size(), "the refusal is reported");
    for (const auto& binding : cuda_report.bindings) {
        check(!binding.qualified, "no implementation is qualified on cuda");
        check(binding.reason.find("cuda") != std::string::npos,
            "the reason names the target");
    }
    check(!family_backend_admits(cuda_report), "the cuda plan is not admitted");

    // An unknown long-range kind fails cleanly: the vocabulary query returns
    // nothing and a node of an unlisted kind stays unbound.
    check(!find_long_range_kind("kspace-coulomb-feature").has_value(),
        "a misspelled kind name is a clean empty result");
    check(!find_long_range_kind("").has_value(),
        "the empty name is a clean empty result");
    {
        Program unknown;
        unknown.family = "polar-long-range-probe";
        unknown.precision = "float64";
        OperatorNode node;
        node.kind = OperatorKind::custom;
        node.id = "polar.long_range.mystery";
        node.capability_key = "mystery";
        unknown.operators.push_back(node);
        const auto unknown_report =
            plan_program(unknown, registry, "host", "float64");
        check(!unknown_report.admitted, "an unknown kind is not admitted");
        check(unknown_report.bindings.size() == 1, "the node is still reported");
        check(!unknown_report.bindings[0].qualified, "the node is unbound");
        check(!unknown_report.bindings[0].portable_covered,
            "the node has no portable coverage");
    }

    // A family-owned name cannot alias the neutral vocabulary.  The node
    // identity, the neutral kind validator, the registry plan, and the
    // fingerprint must all reject the same collision instead of treating a
    // custom spelling as a built-in operator.
    for (const std::string custom_kind :
        {std::string(), std::string("geometry"),
            std::string("kspace_geometry_cache"), std::string("custom")}) {
        Program collision;
        collision.family = "custom-collision-probe";
        collision.precision = "float64";
        OperatorNode node;
        node.kind = OperatorKind::custom;
        node.custom_kind = custom_kind;
        node.id = "family.custom";
        node.capability_key = "custom";
        collision.operators.push_back(node);

        bool fingerprint_rejected = false;
        try {
            (void)symmetrix::execution::program_fingerprint(collision);
        } catch (const std::invalid_argument&) {
            fingerprint_rejected = true;
        }
        check(fingerprint_rejected,
            "the fingerprint rejects an empty or reserved custom kind: '"
                + custom_kind + "'");

        const auto collision_report =
            plan_program(collision, registry, "host", "float64");
        check(!collision_report.admitted
                && collision_report.bindings.size() == 1
                && collision_report.bindings[0].reason.find("custom operator kind")
                    != std::string::npos,
            "the registry rejects the same custom identity: '"
                + custom_kind + "'");

        std::vector<OperatorNode> collision_nodes = nodes;
        collision_nodes.front().kind = OperatorKind::custom;
        collision_nodes.front().custom_kind = custom_kind;
        const std::string collision_reason =
            validate_polar_long_range_node_shapes(collision_nodes, good);
        check(collision_reason.find("custom operator kind") != std::string::npos,
            "the neutral long-range validator rejects the custom identity: '"
                + custom_kind + "': " + collision_reason);
    }

    // -----------------------------------------------------------------
    // Garbage contract data rejects with the rule named
    // -----------------------------------------------------------------
    {
        LongRangeContract bad = good;
        bad.accumulator_dtype = "float32";
        const std::string reason = validate_long_range_contract(bad);
        check(!reason.empty() && reason.find("accumulator") != std::string::npos,
            "a wrong precision policy rejects: " + reason);
    }
    {
        LongRangeContract bad = good;
        bad.table_dtype = "float16";
        check(validate_long_range_contract(bad).find("table_dtype")
                != std::string::npos,
            "an unknown table dtype rejects");
    }
    {
        LongRangeContract bad = good;
        bad.kspace_cutoff = 0.0;
        check(validate_long_range_contract(bad).find("kspace_cutoff")
                != std::string::npos,
            "a non-positive cutoff rejects");
    }
    {
        LongRangeContract bad = good;
        bad.feature_max_l = 2;
        check(validate_long_range_contract(bad).find("l > 1") != std::string::npos,
            "an out-of-support feature max_l rejects");
    }
    {
        LongRangeContract bad = good;
        bad.density_max_l = -1;
        check(validate_long_range_contract(bad).find("density_max_l")
                != std::string::npos,
            "a negative density max_l rejects");
    }
    {
        LongRangeContract bad = good;
        bad.charges_dimension = 7;
        check(validate_long_range_contract(bad).find("charges_dimension")
                != std::string::npos,
            "an inconsistent charges dimension rejects");
    }
    {
        LongRangeContract bad = good;
        bad.potential_dimension = 15;
        check(validate_long_range_contract(bad).find("potential_dimension")
                != std::string::npos,
            "an inconsistent potential dimension rejects");
    }
    {
        LongRangeContract bad = good;
        bad.feature_smearing_widths = {1.5, -3.0};
        check(validate_long_range_contract(bad).find("smearing")
                != std::string::npos,
            "a non-positive smearing width rejects");
    }
    {
        LongRangeContract bad = good;
        bad.recursion_step_count = 0;
        check(validate_polar_long_range_contract(bad).find("recursion")
                != std::string::npos,
            "a zero recursion depth rejects");
    }
    {
        LongRangeContract bad = good;
        bad.spin_channel_count = 3;
        bad.charges_dimension = 12;
        bad.potential_dimension = 24;
        check(validate_polar_long_range_contract(bad).find("spin_channel_count")
                != std::string::npos,
            "a non-alpha/beta spin channel count rejects");
    }
    {
        symmetrix::execution::long_range::LongRangeContract neutral = good;
        neutral.charges_dimension = 12;
        neutral.potential_dimension = 24;
        check(validate_long_range_contract(neutral).empty(),
            "a three-channel electrostatic primitive contract validates");
        check(long_range_operator_nodes(neutral).empty(),
            "the neutral primitive layer refuses program assembly without a "
            "family-owned channel layout");

        LongRangeChannelLayout layout;
        layout.node_id_prefix = "test.long_range";
        layout.channel_names = {"red", "green", "blue"};
        const auto primitives =
            long_range_primitive_operator_nodes(neutral, layout);
        check(primitives.size() == long_range_kind_specs().size() - 1,
            "the neutral builder emits only reusable primitive kinds");
        std::map<std::string, const OperatorNode*> primitive_by_kind;
        for (const auto& node : primitives)
            primitive_by_kind[operator_kind_name(node)] = &node;
        check(primitive_by_kind.find("fukui_normalize")
                == primitive_by_kind.end(),
            "the neutral primitive program carries no Polar equilibration");
        const auto& features =
            *primitive_by_kind.at("kspace_coulomb_features");
        for (const char* channel : {"red", "green", "blue"}) {
            const std::string multipoles =
                std::string("multipoles_") + channel;
            const std::string potentials =
                std::string("potential_features_") + channel;
            check(std::count_if(features.inputs.begin(), features.inputs.end(),
                      [&](const auto& buffer) {
                          return buffer.name == multipoles;
                      }) == 1,
                "the neutral feature node emits " + multipoles);
            check(std::count_if(features.outputs.begin(), features.outputs.end(),
                      [&](const auto& buffer) {
                          return buffer.name == potentials;
                      }) == 1,
                "the neutral feature node emits " + potentials);
        }
        const std::string primitive_reason =
            validate_long_range_primitive_node_shapes(
                primitives, neutral, layout);
        check(primitive_reason.empty(),
            "the three-channel primitive program validates: "
                + primitive_reason);
        auto missing_channel = primitives;
        for (auto& node : missing_channel) {
            if (node.kind != OperatorKind::analytic_correction_features)
                continue;
            node.outputs.erase(std::remove_if(node.outputs.begin(),
                                   node.outputs.end(), [](const auto& buffer) {
                                       return buffer.name
                                           == "correction_fields_blue";
                                   }),
                node.outputs.end());
        }
        const std::string missing_reason =
            validate_long_range_primitive_node_shapes(
                missing_channel, neutral, layout);
        check(missing_reason.find("correction_fields_blue")
                != std::string::npos,
            "the neutral validator rejects an omitted channel: "
                + missing_reason);
    }
    {
        // Invalid contract data has no node spelling: the builder refuses.
        LongRangeContract bad = good;
        bad.kspace_cutoff = -1.0;
        check(polar_long_range_operator_nodes(bad).empty(),
            "invalid contract data produces no operator nodes");
    }

    // -----------------------------------------------------------------
    // Identity
    // -----------------------------------------------------------------
    check(long_range_contract_fingerprint(good)
            == long_range_contract_fingerprint(checkpoint_m_contract()),
        "identical contract data fingerprints identically");
    {
        LongRangeContract changed = good;
        changed.kspace_cutoff = 1.85;
        check(long_range_contract_fingerprint(changed)
                != long_range_contract_fingerprint(good),
            "the cutoff participates in the identity");
        changed = good;
        changed.feature_smearing_widths = {1.5, 3.5};
        check(long_range_contract_fingerprint(changed)
                != long_range_contract_fingerprint(good),
            "the sigma set participates in the identity");
    }

    // -----------------------------------------------------------------
    // Capacity hooks (the manifest long_range vocabulary)
    // -----------------------------------------------------------------
    for (const std::string_view mode : correction_mode_vocabulary) {
        const std::string reason =
            validate_long_range_capacity(62, std::string(mode));
        check(reason.empty(),
            "the correction mode is in the vocabulary: " + reason);
    }
    const std::string unknown_mode = validate_long_range_capacity(
        62, "makov-payne");
    check(!unknown_mode.empty()
        && unknown_mode.find("vocabulary") != std::string::npos,
        "an unknown correction mode rejects naming the vocabulary");
    check(!validate_long_range_capacity(0, "none").empty(),
        "a zero k-point count rejects");
    check(validate_long_range_capacity(62, "none", "float16")
              .find("table_dtype") != std::string::npos,
        "an unknown table dtype rejects");
    // The constructor validates: garbage capacity data has no record
    // spelling, the same refusal the node builder makes of invalid contract
    // data (the Task-8 conformance admission decision).
    auto throws_invalid_argument = [] (auto&& make) {
        try {
            make();
        } catch (const std::invalid_argument&) {
            return true;
        } catch (...) {
        }
        return false;
    };
    check(throws_invalid_argument([] {
        return make_long_range_capacity(0, 5, "none", "float32");
    }), "a zero k-point count rejects at construction");
    check(throws_invalid_argument([] {
        return make_long_range_capacity(62, 5, "box", "float32");
    }), "an unknown correction mode rejects at construction");
    check(throws_invalid_argument([] {
        return make_long_range_capacity(62, 5, "none", "float16");
    }), "an unknown table dtype rejects at construction");
    const LongRangeCapacity mol5 = make_long_range_capacity(
        62, 5, "makov-payne-dabo", "float32");
    check(mol5.table_bytes == 2 * 62 * 5 * long_range_dtype_itemsize("float32"),
        "table_bytes is 2 * n_k * atoms * itemsize (fp32)");
    check(mol5.rebuild_on_cell_change,
        "a cell change rebuilds the grid by default");
    const LongRangeCapacity fp64 = make_long_range_capacity(
        55, 4, "none", "float64", false);
    check(fp64.table_bytes == 2 * 55 * 4 * 8, "fp64 tables double the bytes");
    check(!fp64.rebuild_on_cell_change, "the rebuild trigger is contract data");

    // -----------------------------------------------------------------
    // The PolarMACE program assembly
    // -----------------------------------------------------------------
    // The family's assembly is a pure function over the backbone program and
    // the contract data (mace_polar_program.hpp), so its rules -- the
    // unrolled chain, the node composition, the schedule coverage, and the
    // registry folding -- are testable here without a model, a backend, or an
    // evaluator.
    {
        const PolarSchedule schedule = polar_execution_schedule(2);
        check(schedule.version == 1, "the schedule is versioned");
        check(schedule.recursion_step_count == 2, "the schedule carries the depth");
        check(schedule.entries.size() == 4 + 4 * 2 + 6,
            "the chain is the pre-loop stages, one entry set per unrolled step, "
            "and the post-loop readout");
        check(polar_execution_schedule(0).entries.empty(),
            "an invalid depth has no schedule spelling");
        int previous_iteration = -1;
        int fukui_entries = 0;
        int global_entries = 0;
        for (const PolarScheduleEntry& entry : schedule.entries) {
            check(entry.iteration >= previous_iteration,
                "the chain is linear: iterations never go backwards");
            previous_iteration = entry.iteration;
            if (entry.kind == PolarScheduleEntry::Kind::backbone) {
                check(polar_region_declared(entry.name),
                    "every backbone entry names a declared region: " + entry.name);
            } else {
                ++global_entries;
                check(find_long_range_kind(entry.name).has_value(),
                    "every global entry names a declared kind: " + entry.name);
                if (entry.name == "fukui_normalize")
                    ++fukui_entries;
            }
        }
        // Two before the loop (the cache and the equilibration), three per
        // unrolled step (the features, the external field, the
        // equilibration), and five after it (the readout and the energy and
        // correction kinds).
        check(global_entries == 2 + 3 * 2 + 5,
            "the global entries are one per scheduled operator instance");
        check(fukui_entries == 1 + 2,
            "the equilibration runs once before the loop and once per step");

        // A synthetic backbone program: the assembly copies it verbatim.
        Program backbone;
        backbone.family = "mace-nonlinear";
        backbone.precision = "float64";
        backbone.version = 1;
        {
            OperatorNode node;
            node.kind = OperatorKind::geometry;
            node.id = "mh1.geometry";
            node.capability_key = "geometry";
            backbone.operators.push_back(node);
            OperatorNode readout;
            readout.kind = OperatorKind::readout;
            readout.id = "mh1.readout";
            readout.capability_key = "readout";
            backbone.operators.push_back(readout);
        }
        const Program assembled = compile_polar_program(backbone, good);
        check(assembled.family == "polar-mace",
            "the assembled program declares the family");
        check(assembled.precision == backbone.precision,
            "the assembly keeps the backbone's precision");
        check(assembled.operators.size() == backbone.operators.size() + 9,
            "the assembly appends one node per long-range kind");
        check(assembled.operators.front().id == "mh1.geometry"
                && assembled.operators[1].id == "mh1.readout",
            "the backbone's nodes stay first, in order");
        check(assembled.fingerprint
                == symmetrix::execution::program_fingerprint(assembled),
            "the assembly stamps the program identity");
        {
            LongRangeContract invalid = good;
            invalid.kspace_cutoff = 0.0;
            bool threw = false;
            try {
                compile_polar_program(backbone, invalid);
            } catch (const std::invalid_argument&) {
                threw = true;
            }
            check(threw, "invalid contract data has no assembled program");
        }

        // The chain's multiplicity is program data: the scheduled nodes
        // declare how many times the unrolled iteration evaluates them, which
        // is what makes two recursion depths different programs -- the node
        // set itself is depth-independent.
        {
            const auto declared = [] (const std::vector<OperatorNode>& nodes,
                const char* kind, const char* parameter) {
                for (const OperatorNode& node : nodes)
                    if (operator_kind_name(node) == kind)
                        for (const auto& [name, value] : node.parameters)
                            if (name == parameter)
                                return value;
                return -1.0;
            };
            check(declared(assembled.operators, "kspace_coulomb_features",
                    "evaluation_count") == 2.0,
                "the features node declares its per-step count");
            check(declared(assembled.operators, "external_field_projection",
                    "evaluation_count") == 2.0,
                "the external-field node declares its per-step count");
            check(declared(assembled.operators, "fukui_normalize",
                    "equilibration_count") == 3.0,
                "the equilibration node declares its pre-loop plus per-step "
                "count");
            LongRangeContract shallower = good;
            shallower.recursion_step_count = 1;
            const Program one_step = compile_polar_program(backbone, shallower);
            const Program rebuilt = compile_polar_program(backbone, good);
            check(one_step.fingerprint != assembled.fingerprint,
                "a program of another recursion depth is another program");
            check(rebuilt.fingerprint == assembled.fingerprint,
                "a same-depth rebuild fingerprints identically");
            check(declared(one_step.operators, "fukui_normalize",
                    "equilibration_count") == 2.0,
                "the shallower program declares the shallower chain");
            check(validate_polar_program(one_step, shallower,
                    polar_execution_schedule(1)).empty(),
                "the shallower program is coherent with its own chain");
        }

        // Schedule coverage: an incoherent chain or an incomplete node set is
        // refused with the reason named, never silently planned.
        check(validate_polar_program(assembled, good, schedule).empty(),
            "the assembled program is coherent with its chain");
        {
            Program incomplete = assembled;
            incomplete.operators.pop_back();
            const std::string reason =
                validate_polar_program(incomplete, good, schedule);
            check(reason.find("missing the declared kind") != std::string::npos,
                "an incomplete node set is refused: " + reason);
        }
        {
            PolarSchedule changed = schedule;
            changed.entries.front().name = "mystery_region";
            const std::string reason =
                validate_polar_program(assembled, good, changed);
            check(reason.find("unknown backbone region") != std::string::npos,
                "an unknown backbone region is refused: " + reason);
        }
        {
            PolarSchedule changed = schedule;
            changed.entries[1].name = "kspace_coulomb_energy";
            const std::string reason =
                validate_polar_program(assembled, good, changed);
            check(reason.find("the schedule never reaches")
                    != std::string::npos,
                "a node the chain never reaches is refused: " + reason);
        }
        {
            Program mismatched = assembled;
            for (OperatorNode& node : mismatched.operators)
                if (node.kind == OperatorKind::fukui_normalize)
                    for (auto& [name, value] : node.parameters)
                        if (name == "equilibration_count")
                            value = 5.0;
            const std::string reason =
                validate_polar_program(mismatched, good, schedule);
            check(reason.find("equilibration_count") != std::string::npos,
                "a program whose chain multiplicity disagrees with its chain "
                "is refused: " + reason);
        }
        {
            PolarSchedule changed = schedule;
            changed.recursion_step_count = 3;
            const std::string reason =
                validate_polar_program(assembled, good, changed);
            check(reason.find("the contract's recursion depth")
                    != std::string::npos,
                "a depth the contract does not declare is refused: " + reason);
        }
        {
            Program extra = assembled;
            extra.operators.pop_back();
            PolarSchedule changed = schedule;
            for (auto& entry : changed.entries)
                if (entry.name == "analytic_correction_features")
                    entry.name = "kspace_geometry_cache";
            const std::string reason =
                validate_polar_program(extra, good, changed);
            check(!reason.empty(),
                "an incoherent program never validates silently: " + reason);
        }

        // Registry folding: the family's registry is the backbone's own plus
        // the group's portable descriptors, and the group's host-only
        // delivery leaves another target's plan unbound.
        ImplementationRegistry backbone_registry;
        {
            // A minimal stand-in for the backbone evaluator's own registry:
            // the folding copies whatever the family declared and adds the
            // group's descriptors, so the probe states two backbone entries
            // and checks the sum.
            const auto declare = [&backbone_registry] (
                const char* id, const OperatorKind kind, const char* key) {
                symmetrix::execution::ImplementationDescriptor descriptor;
                descriptor.id = id;
                descriptor.kind = kind;
                descriptor.capability_key = key;
                descriptor.tier = symmetrix::execution::ImplementationTier::portable;
                descriptor.target = "host";
                descriptor.qualified = [] (const auto&) { return true; };
                backbone_registry.declare(std::move(descriptor));
            };
            declare("generic_kokkos_geometry", OperatorKind::geometry, "geometry");
            declare("generic_kokkos_readout", OperatorKind::readout, "readout");
        }
        const ImplementationRegistry folded = polar_implementation_registry(
            backbone_registry, "host", good, "float64");
        check(folded.size()
                == backbone_registry.size() + long_range_kind_specs().size(),
            "the folded registry is the backbone's own plus the group's");
        const auto folded_report =
            plan_program(assembled, folded, "host", "float64");
        check(folded_report.admitted
                && folded_report.node_count == assembled.operators.size(),
            "the assembled program binds against the folded registry");
        const auto folded_cuda =
            plan_program(assembled, folded, "cuda", "float64");
        check(!folded_cuda.admitted,
            "the host-only delivery leaves a cuda plan unbound");
    }

    if (failures != 0) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("long-range kind group probe: all checks passed\n");
    return 0;
}
"""


def run_kind_probe(tmp_path: Path) -> str:
    """Compile and run the kind-group probe, returning its stdout.

    The probe owns the declarative layer's rules (the vocabulary pins, the
    contract admission, the capability state, the shape pass, and the program
    assembly), so it is the compiled-side gate beside the conformance
    battery's data-side one.
    """
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++ compiler is required for the kind group probe")
    source = tmp_path / "long_range_probe.cpp"
    source.write_text(HARNESS)
    binary = tmp_path / "long_range_probe"
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
    return completed.stdout


def test_long_range_kind_group_declares_and_binds(tmp_path):
    stdout = run_kind_probe(tmp_path)
    # The cross-language vocabulary pin: the probe prints the C++ kind and
    # correction-mode vocabularies, and the Python declaration constants
    # must carry exactly the same tokens in the same declaration order --
    # otherwise the two sides drift while each suite checks only its own
    # literals.
    from symmetrix.polar_long_range_kinds import (
        POLAR_LONG_RANGE_CORRECTION_MODES,
        POLAR_LONG_RANGE_KINDS,
    )

    def vocabulary(token: str) -> list[str]:
        prefix = f"vocabulary {token} "
        return [
            line[len(prefix) :]
            for line in stdout.splitlines()
            if line.startswith(prefix)
        ]

    assert tuple(vocabulary("kind")) == POLAR_LONG_RANGE_KINDS
    assert tuple(vocabulary("correction_mode")) == POLAR_LONG_RANGE_CORRECTION_MODES
