#pragma once

#include <string>

#include "nlohmann/json.hpp"

struct MaceMH1FamilyDescriptor {
    bool compatible = false;
    int l_max = 0;
    int radial_size = 0;
    int node_channels = 0;
    int edge_channels = 0;
    std::string rejection_reason;
};

void validate_mace_nonlinear_schema(const nlohmann::json& data);
MaceMH1FamilyDescriptor analyze_mh1_family_architecture(
    const nlohmann::json& data);
