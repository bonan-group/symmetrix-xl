#pragma once

#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

inline nlohmann::json select_prediction_head(
    nlohmann::json data, const std::string& requested_head = {})
{
    const auto section_iterator = data.find("prediction_heads");
    if (section_iterator == data.end()) {
        const std::string legacy_head = data.value("head", std::string());
        if (!requested_head.empty() && requested_head != legacy_head) {
            if (legacy_head.empty())
                throw std::invalid_argument(
                    "This legacy Symmetrix JSON does not identify a prediction head.");
            throw std::invalid_argument(
                "Requested prediction head '" + requested_head
                + "' is unavailable; the JSON contains '" + legacy_head + "'.");
        }
        data["selected_head"] = legacy_head;
        data["available_heads"] = legacy_head.empty()
            ? nlohmann::json::array() : nlohmann::json::array({legacy_head});
        return data;
    }

    const auto& section = *section_iterator;
    if (!section.is_object() || section.value("schema_version", 0) != 1)
        throw std::invalid_argument(
            "prediction_heads has an unsupported schema version.");
    if (!section.contains("names") || !section.at("names").is_array())
        throw std::invalid_argument("prediction_heads.names must be an array.");
    const auto names = section.at("names").get<std::vector<std::string>>();
    const std::set<std::string> unique_names(names.begin(), names.end());
    if (names.empty() || unique_names.size() != names.size()
            || unique_names.contains(std::string()))
        throw std::invalid_argument(
            "prediction_heads.names must contain unique non-empty strings.");
    if (!section.contains("parameters") || !section.at("parameters").is_object()
            || section.at("parameters").size() != names.size())
        throw std::invalid_argument(
            "prediction_heads.parameters must define every named head.");
    std::set<std::string> expected_fields;
    if (data.value("model_type", std::string("MACE")) == "MACE_Nonlinear") {
        expected_fields = {"atomic_energies", "scale_shift", "readouts"};
    } else {
        expected_fields = {
            "atomic_energies", "readout_1_weights",
            "readout_2_weights_1", "readout_2_weights_2"};
        if (data.value("has_zbl", false))
            expected_fields.insert("zbl_c");
    }
    for (const auto& name : names) {
        if (!section.at("parameters").contains(name)
                || !section.at("parameters").at(name).is_object())
            throw std::invalid_argument(
                "prediction_heads.parameters must define every named head.");
        std::set<std::string> fields;
        for (const auto& [field, unused] : section.at("parameters").at(name).items())
            fields.insert(field);
        if (fields != expected_fields)
            throw std::invalid_argument(
                "Every prediction-head payload must contain exactly the supported fields.");
    }
    const std::string default_head = section.value("default", std::string());
    if (!unique_names.contains(default_head))
        throw std::invalid_argument(
            "prediction_heads.default must name a stored head.");
    const std::string selected = requested_head.empty() ? default_head : requested_head;
    if (!unique_names.contains(selected))
        throw std::invalid_argument(
            "Requested prediction head '" + selected + "' is unavailable.");

    for (const auto& [name, value] : section.at("parameters").at(selected).items())
        data[name] = value;
    data["head"] = selected;
    data["selected_head"] = selected;
    data["available_heads"] = names;
    return data;
}
