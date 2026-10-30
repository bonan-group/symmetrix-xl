#include "mace_polar_schema.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "mace_nonlinear_schema.hpp"

namespace symmetrix::mace_polar {

namespace {

const nlohmann::json& object(
    const nlohmann::json& parent, const char* key, const char* name)
{
    const auto& value = parent.at(key);
    if (!value.is_object())
        throw std::invalid_argument(
            std::string("PolarMACE ") + name + " must be an object.");
    return value;
}

const nlohmann::json& array(
    const nlohmann::json& parent, const char* key, const char* name)
{
    const auto& value = parent.at(key);
    if (!value.is_array())
        throw std::invalid_argument(
            std::string("PolarMACE ") + name + " must be an array.");
    return value;
}

bool boolean_flag(
    const nlohmann::json& parent, const char* key, const char* name)
{
    const auto& value = parent.at(key);
    if (!value.is_boolean())
        throw std::invalid_argument(
            std::string("PolarMACE ") + name + " must be a boolean.");
    return value.get<bool>();
}

int positive_int(const nlohmann::json& parent, const char* key, const char* name)
{
    const auto& value = parent.at(key);
    if (!value.is_number_integer() || value.get<long long>() < 1)
        throw std::invalid_argument(
            std::string("PolarMACE ") + name + " must be a positive integer.");
    return value.get<int>();
}

double finite_number(
    const nlohmann::json& parent, const char* key, const char* name)
{
    const auto& value = parent.at(key);
    if (!value.is_number() || !std::isfinite(value.get<double>()))
        throw std::invalid_argument(
            std::string("PolarMACE ") + name + " must be a finite number.");
    return value.get<double>();
}

const nlohmann::json& embedded_contract(const nlohmann::json& data)
{
    const auto& contracts = object(data, "execution_contracts", "execution_contracts");
    if (!contracts.contains(contract_key))
        throw std::invalid_argument(
            std::string("the PolarMACE model file must carry the ")
            + contract_key + " contract.");
    const auto& contract = contracts.at(contract_key);
    if (!contract.is_object() || contract.value("tag", std::string()) != contract_tag)
        throw std::invalid_argument(
            std::string("the PolarMACE model file must carry a ") + contract_tag
            + " contract; version 1 (without the unrolled chain as contract "
            "data) is not served.");
    return contract;
}

} // namespace

void validate_mace_polar_schema(const nlohmann::json& data)
{
    // The shared backbone structure, under this family's rule row.  The polar
    // block and the identity below are what is genuinely family-specific; every
    // other rule is the one the MH-1 family's files are held to.
    validate_mace_backbone_schema(data, MaceBackboneSchema::polar);

    const auto& polar = object(data, "polar", "polar backbone deltas");
    if (!boolean_flag(
            polar, "keep_last_layer_irreps",
            "polar.keep_last_layer_irreps"))
        throw std::invalid_argument(
            "PolarMACE requires keep_last_layer_irreps backbone readouts.");
    const int interaction_count = data.at("num_interactions").get<int>();
    const int recursion_steps =
        positive_int(polar, "num_recursion_steps", "polar.num_recursion_steps");
    const std::size_t head_count =
        array(polar, "spin_source_heads", "polar.spin_source_heads").size();
    if (head_count != static_cast<std::size_t>(interaction_count))
        throw std::invalid_argument(
            "PolarMACE polar.spin_source_heads must match the interaction count.");
    const auto& mixer = object(polar, "feature_mixer", "polar.feature_mixer");
    const std::size_t mixer_count =
        array(mixer, "linears", "polar.feature_mixer.linears").size();
    if (mixer_count != static_cast<std::size_t>(interaction_count))
        throw std::invalid_argument(
            "PolarMACE polar.feature_mixer.linears must match the interaction "
            "count.");
    const std::size_t update_count =
        array(polar, "field_update_blocks", "polar.field_update_blocks").size();
    if (update_count != static_cast<std::size_t>(recursion_steps))
        throw std::invalid_argument(
            "PolarMACE polar.field_update_blocks must match "
            "polar.num_recursion_steps.");
    for (const char* key : {"charges_irreps", "field_irreps", "potential_irreps"}) {
        const auto& irreps = polar.at(key);
        if (!irreps.is_string() || irreps.get<std::string>().empty())
            throw std::invalid_argument(
                std::string("PolarMACE polar.") + key
                + " must be a non-empty irreps string.");
    }
    if (finite_number(polar, "kspace_cutoff", "polar.kspace_cutoff") <= 0.0)
        throw std::invalid_argument("PolarMACE polar.kspace_cutoff must be positive.");

    // The identity's counts are contract data the program assembly validates
    // against its own chain and node set; the schema checks the two the
    // evaluator's configuration depends on.
    const auto& identity = embedded_contract(data);
    if (identity.value("version", 0) != contract_version
        || identity.value("schema", std::string()) != contract_schema)
        throw std::invalid_argument(
            std::string("the PolarMACE ") + contract_key
            + " contract must declare the generated schema and its version.");
    if (identity.value("interaction_count", 0) != interaction_count)
        throw std::invalid_argument(
            std::string("the PolarMACE ") + contract_key
            + " contract's interaction count must match the backbone.");
    if (!boolean_flag(
            identity, "keep_last_layer_irreps",
            "POLAR_BACKBONE keep_last_layer_irreps"))
        throw std::invalid_argument(
            std::string("the PolarMACE ") + contract_key
            + " contract must declare keep_last_layer_irreps=true.");
    if (identity.value("recursion_step_count", 0) != recursion_steps)
        throw std::invalid_argument(
            std::string("the PolarMACE ") + contract_key
            + " contract's recursion depth must match the backbone's.");
}

execution::polar::PolarLongRangeContract long_range_contract_from_model(
    const nlohmann::json& data)
{
    const auto& identity = embedded_contract(data);
    const auto& charges = object(identity, "charges", "charges descriptor");
    const auto& polar_field = object(identity, "field", "field descriptor");
    const auto& potential = object(identity, "potential", "potential descriptor");

    execution::polar::PolarLongRangeContract contract;
    contract.kspace_cutoff =
        finite_number(polar_field, "kspace_cutoff", "field.kspace_cutoff");
    contract.density_max_l =
        charges.at("max_l").get<int>();
    contract.density_smearing_width = finite_number(
        charges, "smearing_width", "charges.smearing_width");
    contract.feature_max_l = polar_field.at("max_l").get<int>();
    contract.feature_smearing_widths =
        array(polar_field, "smearing_widths", "field.smearing_widths")
            .get<std::vector<double>>();
    contract.charges_dimension = charges.at("dimension").get<std::size_t>();
    contract.field_dimension = polar_field.at("dimension").get<std::size_t>();
    contract.potential_dimension = potential.at("dimension").get<std::size_t>();
    contract.field_norm_factor = finite_number(
        identity, "field_norm_factor", "field_norm_factor");
    contract.recursion_step_count =
        positive_int(identity, "recursion_step_count", "recursion_step_count");
    // The checkpoint default remains fp64 for standalone contract queries.
    // A concrete evaluator rewrites this field to its own precision when it
    // admits the contract, matching the native table/workspace specialization.
    contract.table_dtype = identity.value("table_dtype", std::string("float64"));
    // The self-interaction policy the checkpoint's polar block carries: the
    // contract's flags object is the admitted identity's own record of it, so
    // the energy node's overlap term and the feature self-terms are declared
    // from the model rather than from a kind-group default.
    {
        const auto& flags = object(identity, "flags", "flags descriptor");
        contract.include_electrostatic_self_interaction = boolean_flag(flags,
            "include_electrostatic_self_interaction",
            "flags.include_electrostatic_self_interaction");
        contract.field_si =
            boolean_flag(flags, "field_si", "flags.field_si");
    }

    const std::string reason =
        execution::polar::validate_polar_long_range_contract(contract);
    if (!reason.empty())
        throw std::invalid_argument(
            std::string("the PolarMACE ") + contract_key
            + " contract's long-range slice is invalid: " + reason);
    return contract;
}

} // namespace symmetrix::mace_polar
