#include "model_metadata.hpp"

#include <fstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

std::pair<std::string, bool> symmetrix_model_metadata(const std::string& filename)
{
    std::ifstream stream(filename);
    if (!stream)
        throw std::runtime_error("Could not open Symmetrix model file: " + filename);
    const nlohmann::json data = nlohmann::json::parse(stream);
    if (!data.is_object())
        throw std::invalid_argument("Symmetrix model JSON must contain an object.");
    return {
        data.value("model_type", std::string("MACE")),
        data.value("has_field_coupling", false),
    };
}

std::string symmetrix_model_type(const std::string& filename)
{
    return symmetrix_model_metadata(filename).first;
}

bool symmetrix_is_nonlinear_mace_model(const std::string& filename)
{
    return symmetrix_model_type(filename) == "MACE_Nonlinear";
}
