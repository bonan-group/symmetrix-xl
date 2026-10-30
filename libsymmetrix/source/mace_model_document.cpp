#include "mace_model_document.hpp"

#include <chrono>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "mace_prediction_heads.hpp"

SelectedMACEModel read_selected_mace_model(
    const std::string& filename,
    const std::string& requested_head)
{
    std::ifstream stream(filename);
    const auto parse_start = std::chrono::steady_clock::now();
    auto data = select_prediction_head(
        nlohmann::json::parse(stream), requested_head);
    const auto json_parse_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - parse_start).count();
    const auto model_type = data.value("model_type", std::string("MACE"));

    MACEModelKind kind;
    if (model_type == "MACE")
        kind = MACEModelKind::mace;
    else if (model_type == "MACEField")
        kind = MACEModelKind::mace_field;
    else if (model_type == "MACELES")
        kind = MACEModelKind::mace_les;
    else if (model_type == "MACE_Nonlinear")
        throw std::invalid_argument(
            "MACE_Nonlinear JSON must be loaded through the nonlinear MACE "
            "Kokkos evaluator, not legacy MACEKokkos.");
    else
        throw std::invalid_argument(
            "Unsupported MACE model_type '" + model_type + "'.");

    auto selected_head = data.value("selected_head", std::string());
    auto available_heads =
        data.value("available_heads", std::vector<std::string>());
    return SelectedMACEModel{
        std::move(data),
        std::move(selected_head),
        std::move(available_heads),
        kind,
        json_parse_ms,
    };
}
