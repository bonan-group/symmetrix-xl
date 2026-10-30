#pragma once

#include <string>
#include <vector>

#include "nlohmann/json.hpp"

enum class MACEModelKind { mace, mace_field, mace_les };

struct SelectedMACEModel {
    nlohmann::json data;
    std::string selected_head;
    std::vector<std::string> available_heads;
    MACEModelKind kind = MACEModelKind::mace;
    double json_parse_ms = 0.0;
};

SelectedMACEModel read_selected_mace_model(
    const std::string& filename,
    const std::string& requested_head);
