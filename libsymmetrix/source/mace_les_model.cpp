#include "mace_les_model.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace {

void require_fields(const nlohmann::json& value,
                    const std::vector<std::string>& fields,
                    const char* name)
{
    if (!value.is_object() || value.size() != fields.size())
        throw std::invalid_argument(std::string(name)+" has an unsupported field set.");
    for (const auto& field : fields)
        if (!value.contains(field))
            throw std::invalid_argument(std::string(name)+" is missing "+field+".");
}

double finite(const nlohmann::json& value, const char* name, bool positive = false)
{
    if (!value.is_number())
        throw std::invalid_argument(std::string(name)+" must be numeric.");
    const double result = value.get<double>();
    if (!std::isfinite(result) || (positive && result <= 0.0))
        throw std::invalid_argument(std::string(name)+" must be finite and positive.");
    return result;
}

std::vector<double> weights(const nlohmann::json& value,
                            const char* name, std::size_t expected)
{
    if (!value.is_array() || value.size() != expected)
        throw std::invalid_argument(std::string(name)+" has an incompatible length.");
    auto result = value.get<std::vector<double>>();
    for (const double weight : result)
        if (!std::isfinite(weight))
            throw std::invalid_argument(std::string(name)+" must be finite.");
    return result;
}

} // namespace

template <typename Precision>
std::unique_ptr<MACELESParameters<Precision>> MACELESParameters<Precision>::from_json(
    const nlohmann::json& data, const int interactions, const int channels,
    const nlohmann::json& execution_contracts)
{
    require_fields(data, {"schema", "version", "kind", "periodicity", "ewald",
        "output_scale", "mace_num_interactions", "mace_num_channels",
        "mace_contract_fingerprints", "readout_1_weights",
        "readout_2_hidden_size", "readout_2_weights_1",
        "readout_2_weights_2", "readout_2_scale_factor"}, "les");
    if (data.at("schema") != "symmetrix.model.mace_les" || data.at("version") != 1
        || data.at("kind") != "monopole" || data.at("periodicity") != "geometry")
        throw std::invalid_argument("les has an unsupported MACELES schema.");
    if (data.at("mace_num_interactions").get<int>() != interactions)
        throw std::invalid_argument("les.mace_num_interactions does not match MACE.");
    if (data.at("mace_num_channels").get<int>() != channels)
        throw std::invalid_argument("les.mace_num_channels does not match MACE.");
    const auto& ewald = data.at("ewald");
    require_fields(ewald, {"sigma", "dl", "remove_self_interaction"}, "les.ewald");
    if (!ewald.at("remove_self_interaction").is_boolean())
        throw std::invalid_argument("les.ewald.remove_self_interaction must be boolean.");
    const int hidden = data.at("readout_2_hidden_size").get<int>();
    if (hidden < 1)
        throw std::invalid_argument("les.readout_2_hidden_size must be positive.");
    auto result = std::make_unique<MACELESParameters>();
    result->schema_version = data.at("version").get<int>();
    const auto& fingerprints = data.at("mace_contract_fingerprints");
    require_fields(
        fingerprints, {"R0", "M0", "R1"}, "les.mace_contract_fingerprints");
    if (!execution_contracts.is_object())
        throw std::invalid_argument(
            "MACELES requires loaded R0, M0, and R1 execution contracts.");
    const auto compare_contract = [&] (
        const char* name, const char* field) -> std::string {
        if (!execution_contracts.contains(name)
            || !execution_contracts.at(name).is_object()
            || !execution_contracts.at(name).contains(field))
            throw std::invalid_argument(
                std::string("MACELES requires a loaded ")+name
                +" execution contract.");
        const std::string expected = fingerprints.at(name).get<std::string>();
        const std::string actual =
            execution_contracts.at(name).at(field).get<std::string>();
        if (expected != actual)
            throw std::invalid_argument(
                std::string("MACELES MACE ")+name
                +" contract fingerprint does not match the loaded MACE backbone.");
        return expected;
    };
    result->mace_r0_fingerprint = compare_contract("R0", "fingerprint");
    result->mace_m0_fingerprint =
        compare_contract("M0", "semantic_fingerprint");
    result->mace_r1_fingerprint = compare_contract("R1", "fingerprint");
    result->sigma = finite(ewald.at("sigma"), "les.ewald.sigma", true);
    result->dl = finite(ewald.at("dl"), "les.ewald.dl", true);
    result->remove_self_interaction = ewald.at("remove_self_interaction").get<bool>();
    result->output_scale = finite(data.at("output_scale"), "les.output_scale");
    result->readout_2_hidden_size = hidden;
    result->readout_2_scale_factor = finite(
        data.at("readout_2_scale_factor"), "les.readout_2_scale_factor");
    result->readout_1_weights = weights(
        data.at("readout_1_weights"), "les.readout_1_weights", channels);
    result->readout_2_weights_1 = weights(data.at("readout_2_weights_1"),
        "les.readout_2_weights_1", static_cast<std::size_t>(channels)*hidden);
    result->readout_2_weights_2 = weights(data.at("readout_2_weights_2"),
        "les.readout_2_weights_2", hidden);
    return result;
}

template struct MACELESParameters<float>;
template struct MACELESParameters<double>;
