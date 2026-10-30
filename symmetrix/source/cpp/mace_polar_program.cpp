#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "family_interface.hpp"
#include "implementation_registry.hpp"
#include "mace_polar_binding.hpp"
#include "mace_polar_program.hpp"

namespace py = pybind11;

namespace symmetrix::execution::polar {

polar::PolarLongRangeContract long_range_contract_from_python(
    const double kspace_cutoff,
    const int density_max_l,
    const double density_smearing_width,
    const int feature_max_l,
    const std::vector<double>& feature_smearing_widths,
    const std::size_t charges_dimension,
    const std::size_t field_dimension,
    const std::size_t potential_dimension,
    const double field_norm_factor,
    const int recursion_step_count,
    const std::string& table_dtype,
    const bool include_electrostatic_self_interaction,
    const bool field_si)
{
    polar::PolarLongRangeContract contract;
    contract.kspace_cutoff = kspace_cutoff;
    contract.density_max_l = density_max_l;
    contract.density_smearing_width = density_smearing_width;
    contract.feature_max_l = feature_max_l;
    contract.feature_smearing_widths = feature_smearing_widths;
    contract.charges_dimension = charges_dimension;
    contract.field_dimension = field_dimension;
    contract.potential_dimension = potential_dimension;
    contract.field_norm_factor = field_norm_factor;
    contract.recursion_step_count = recursion_step_count;
    contract.table_dtype = table_dtype;
    contract.include_electrostatic_self_interaction =
        include_electrostatic_self_interaction;
    contract.field_si = field_si;
    return contract;
}

namespace {

py::dict buffer_dict(const BufferContract& buffer)
{
    py::dict entry;
    entry["name"] = buffer.name;
    entry["dtype"] = buffer.dtype;
    py::list shape;
    for (const auto size : buffer.shape)
        shape.append(size);
    entry["shape"] = shape;
    entry["layout"] = buffer.layout.order;
    entry["irreps"] = buffer.layout.irreps;
    return entry;
}

py::dict node_dict(const OperatorNode& node)
{
    py::dict entry;
    entry["kind"] = operator_kind_name(node);
    entry["id"] = node.id;
    entry["capability_key"] = node.capability_key;
    py::dict parameters;
    for (const auto& [name, value] : node.parameters)
        parameters[py::cast(name)] = value;
    entry["parameters"] = parameters;
    py::list inputs;
    for (const auto& input : node.inputs)
        inputs.append(buffer_dict(input));
    entry["inputs"] = inputs;
    py::list outputs;
    for (const auto& output : node.outputs)
        outputs.append(buffer_dict(output));
    entry["outputs"] = outputs;
    return entry;
}

py::dict schedule_dict(const PolarSchedule& schedule)
{
    py::dict result;
    result["version"] = schedule.version;
    result["recursion_step_count"] = schedule.recursion_step_count;
    py::list entries;
    for (const auto& entry : schedule.entries) {
        py::dict item;
        item["kind"] = entry.kind == PolarScheduleEntry::Kind::backbone
            ? "backbone" : "global_operator";
        item["name"] = entry.name;
        item["iteration"] = entry.iteration;
        entries.append(item);
    }
    result["entries"] = entries;
    return result;
}

py::dict binding_report_dict(const ImplementationBindingReport& report)
{
    py::dict result;
    result["version"] = report.version;
    result["target"] = report.target;
    result["precision"] = report.precision;
    result["artifact_ready"] = report.artifact_ready;
    result["admitted"] = report.admitted;
    result["node_count"] = report.node_count;
    result["portable_node_count"] = report.portable_node_count;
    py::list bindings;
    for (const auto& binding : report.bindings) {
        py::dict entry;
        entry["operator_id"] = binding.operator_id;
        entry["kind"] = !binding.custom_kind.empty()
            ? binding.custom_kind
            : operator_kind_name(binding.kind);
        entry["capability_key"] = binding.capability_key;
        entry["implementation_id"] = binding.implementation_id;
        entry["tier"] = implementation_tier_name(binding.tier);
        entry["target"] = binding.target;
        py::dict precision;
        precision["compute"] = binding.precision.compute;
        precision["accumulator"] = binding.precision.accumulator;
        entry["precision"] = precision;
        entry["identity_fingerprint"] = binding.identity_fingerprint;
        entry["qualified"] = binding.qualified;
        entry["portable_covered"] = binding.portable_covered;
        entry["artifact_required"] = binding.artifact_required;
        entry["reason"] = binding.reason;
        bindings.append(entry);
    }
    result["bindings"] = bindings;
    return result;
}

} // namespace

} // namespace symmetrix::execution::polar

void bind_mace_polar_program(py::module_& module)
{
    using symmetrix::execution::polar::PolarSchedule;
    using symmetrix::execution::polar::polar_execution_schedule;
    using symmetrix::execution::polar::schedule_dict;

    // The unrolled chain as contract data: the native mirror of the Python
    // contract builder's schedule, exposed so the two sides are compared by a
    // test rather than by inspection.
    module.def(
        "polar_schedule",
        [] (const int recursion_step_count) {
            return schedule_dict(polar_execution_schedule(recursion_step_count));
        },
        py::arg("recursion_step_count"),
        "The unrolled PolarMACE global<->local chain for a recursion depth.  "
        "A depth below one has no chain: the empty entry list is the refusal "
        "(the group's idiom for invalid input), not a zero-length chain.");

    // The long-range half of the PolarMACE program, compiled from contract
    // data alone: the node contracts, the tiered binding report, and the
    // schedule.  This is the program slice the conformance battery reads --
    // the backbone half is the family evaluator's own program, which needs a
    // model to compile.
    module.def(
        "polar_long_range_program",
        [] (const double kspace_cutoff, const int density_max_l,
            const double density_smearing_width, const int feature_max_l,
            const std::vector<double>& feature_smearing_widths,
            const std::size_t charges_dimension,
            const std::size_t field_dimension,
            const std::size_t potential_dimension,
            const double field_norm_factor, const int recursion_step_count,
            const std::string& table_dtype, const std::string& target,
            const std::string& compute, const int node_feature_dimension,
            const int num_elements, const int interaction_count,
            const bool subtract_feature_self_terms,
            const bool include_electrostatic_self_interaction,
            const bool field_si) {
            using symmetrix::execution::Program;
            using symmetrix::execution::plan_program;
            using symmetrix::execution::long_range::declare_long_range_implementations;
            using symmetrix::execution::polar::compile_polar_program;
            using symmetrix::execution::polar::polar_execution_schedule;
            const symmetrix::execution::polar::PolarLongRangeContract contract =
                symmetrix::execution::polar::long_range_contract_from_python(
                    kspace_cutoff, density_max_l, density_smearing_width,
                    feature_max_l, feature_smearing_widths, charges_dimension,
                    field_dimension, potential_dimension, field_norm_factor,
                    recursion_step_count, table_dtype,
                    include_electrostatic_self_interaction, field_si);
            // The slice is the family's own assembly over an empty backbone,
            // so this query exercises exactly the code path the evaluator
            // runs: the same contract validation, node set, shape pass,
            // producer/consumer coherence, schedule derivation, and program
            // identity.  The delta group joins the assembly when its own
            // contract data is supplied, which is what gives the chain
            // state's buffers their producers.
            symmetrix::execution::polar::DeltaContract delta;
            delta.node_feature_dimension = node_feature_dimension;
            delta.num_elements = num_elements;
            delta.interaction_count = interaction_count;
            delta.subtract_feature_self_terms = subtract_feature_self_terms;
            Program backbone;
            backbone.precision = compute;
            Program program;
            try {
                program = compile_polar_program(backbone, contract, delta);
            } catch (const std::invalid_argument& error) {
                throw py::value_error(error.what());
            }
            const PolarSchedule schedule =
                polar_execution_schedule(contract.recursion_step_count);
            // The group's delivered implementation is the portable (T1) host
            // one -- the global nodes stay tier-1 until a device delivery
            // exists -- so the registry declares it for the host and the plan
            // asks the requested target.  A plan on another target therefore
            // reports every node unbound with the target named, which is the
            // honest refusal the kind-group probe asserts on the C++ side.
            symmetrix::execution::ImplementationRegistry registry;
            declare_long_range_implementations(
                registry,"host",contract,compute,
                symmetrix::execution::polar::
                    polar_long_range_contract_fingerprint(contract));
            if (delta.declared())
                symmetrix::execution::polar::declare_delta_implementations(
                    registry, "host", contract, compute, delta);
            py::dict result;
            result["family"] = program.family;
            result["precision"] = program.precision;
            result["fingerprint"] = program.fingerprint;
            result["contract_fingerprint"] =
                symmetrix::execution::polar::polar_long_range_contract_fingerprint(
                    contract);
            py::list nodes;
            py::list delta_nodes;
            for (const auto& node : program.operators) {
                if (symmetrix::execution::polar::find_delta_kind(
                        operator_kind_name(node)).has_value())
                    delta_nodes.append(
                        symmetrix::execution::polar::node_dict(node));
                else
                    nodes.append(symmetrix::execution::polar::node_dict(node));
            }
            result["nodes"] = nodes;
            result["delta_nodes"] = delta_nodes;
            result["binding_report"] = symmetrix::execution::polar::
                binding_report_dict(plan_program(program, registry, target, compute));
            result["schedule"] = symmetrix::execution::polar::schedule_dict(schedule);
            return result;
        },
        py::arg("kspace_cutoff"), py::arg("density_max_l"),
        py::arg("density_smearing_width"), py::arg("feature_max_l"),
        py::arg("feature_smearing_widths"), py::arg("charges_dimension"),
        py::arg("field_dimension"), py::arg("potential_dimension"),
        py::arg("field_norm_factor") = 1.0,
        py::arg("recursion_step_count"), py::arg("table_dtype") = "float64",
        py::arg("target") = "host", py::arg("compute") = "float64",
        py::arg("node_feature_dimension") = 0, py::arg("num_elements") = 0,
        py::arg("interaction_count") = 0,
        py::arg("subtract_feature_self_terms") = true,
        py::arg("include_electrostatic_self_interaction") = false,
        py::arg("field_si") = false,
        "The long-range half of the PolarMACE program: node contracts, the "
        "tiered binding report, and the unrolled schedule, from contract data "
        "alone.  ``target`` is the plan's target: the group's tier-1 "
        "implementation is delivered for the host, so a plan for another "
        "target reports its nodes unbound with that target named.");
}
