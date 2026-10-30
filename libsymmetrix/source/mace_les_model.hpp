#pragma once

#include <memory>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

template <typename Precision>
struct MACELESParameters {
    int schema_version = 0;
    double sigma = 0.0;
    double dl = 0.0;
    bool remove_self_interaction = true;
    double output_scale = 1.0;
    int readout_2_hidden_size = 0;
    double readout_2_scale_factor = 1.0;
    std::vector<double> readout_1_weights;
    std::vector<double> readout_2_weights_1;
    std::vector<double> readout_2_weights_2;
    std::string mace_r0_fingerprint;
    std::string mace_m0_fingerprint;
    std::string mace_r1_fingerprint;

    static std::unique_ptr<MACELESParameters> from_json(
        const nlohmann::json& data, int interactions, int channels,
        const nlohmann::json& execution_contracts);
};
