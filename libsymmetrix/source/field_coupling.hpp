#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"

struct FieldAngularEntry {
    int input_lm;
    int field_component;
    int output_lm;
    double coefficient;
};

struct FieldCouplingPath {
    int input_l;
    int output_l;
    std::vector<FieldAngularEntry> angular_entries;
    std::vector<double> channel_matrix;
    std::vector<double> channel_up_matrix;
};

inline void validate_macefield_L_max(const int L_max)
{
    if (L_max < 1 || L_max > 2)
        throw std::runtime_error(
            "MACEField field coupling currently supports only 1 <= L_max <= 2.");
}

inline std::string canonical_field_irreps(const int L_max, const int channels)
{
    std::string result;
    for (int l = 0; l <= L_max; ++l) {
        if (!result.empty())
            result += "+";
        result += std::to_string(channels) + "x" + std::to_string(l)
            + (l % 2 == 0 ? "e" : "o");
    }
    return result;
}

inline void require_unit_field_mask(
    const nlohmann::json& coupling,
    const char* key,
    const std::size_t expected_size)
{
    const auto mask = coupling.at(key).get<std::vector<double>>();
    if (mask.size() != expected_size)
        throw std::runtime_error(std::string("MACEField ") + key + " has an invalid size.");
    for (const double value : mask) {
        if (value != 1.0)
            throw std::runtime_error(std::string("MACEField ") + key + " must contain only ones.");
    }
}

inline std::vector<FieldCouplingPath> compile_field_coupling(
    const nlohmann::json& coupling,
    const int L_max,
    const int channels,
    const std::vector<double>& H1_linear_up_weights)
{
    validate_macefield_L_max(L_max);
    const bool legacy = !coupling.contains("schema_version");
    if (!legacy && coupling.at("schema_version").get<int>() != 1)
        throw std::runtime_error("Unsupported MACEField field-coupling schema version.");
    if (legacy && L_max != 1)
        throw std::runtime_error(
            "Unversioned MACEField field coupling is supported only for legacy L_max == 1 data.");

    const std::string hidden_irreps = canonical_field_irreps(L_max, channels);
    if (coupling.at("field_feats_irreps_in1").get<std::string>() != hidden_irreps
        || coupling.at("field_feats_irreps_in2").get<std::string>() != "1x1o"
        || coupling.at("field_feats_irreps_out").get<std::string>() != hidden_irreps
        || coupling.at("field_linear_irreps_in").get<std::string>() != hidden_irreps
        || coupling.at("field_linear_irreps_out").get<std::string>() != hidden_irreps)
        throw std::runtime_error(
            "MACEField field coupling must use the canonical equal-channel hidden irreps.");

    const std::size_t hidden_size = static_cast<std::size_t>((L_max + 1) * (L_max + 1))
        * static_cast<std::size_t>(channels);
    require_unit_field_mask(coupling, "field_feats_output_mask", hidden_size);
    require_unit_field_mask(coupling, "field_linear_output_mask", hidden_size);
    if (!coupling.at("field_linear_bias").get<std::vector<double>>().empty())
        throw std::runtime_error("MACEField field_linear bias must be empty.");

    const auto field_weights = coupling.at("field_feats_weight").get<std::vector<double>>();
    const auto linear_weights = coupling.at("field_linear_weight").get<std::vector<double>>();
    const auto& field_instructions = coupling.at("field_feats_instructions");
    const auto& linear_instructions = coupling.at("field_linear_instructions");
    const std::size_t channel_pairs = static_cast<std::size_t>(channels) * channels;
    if (H1_linear_up_weights.size()
        != static_cast<std::size_t>(L_max + 1) * channel_pairs)
        throw std::runtime_error("MACEField split H1 linear_up weights have an invalid size.");

    std::vector<std::size_t> field_offsets;
    std::size_t field_offset = 0;
    for (const auto& instruction : field_instructions) {
        field_offsets.push_back(field_offset);
        const auto shape = instruction.at("path_shape").get<std::vector<int>>();
        if (instruction.at("connection_mode").get<std::string>() != "uvw"
            || shape != std::vector<int>{channels, 1, channels})
            throw std::runtime_error(
                "MACEField field_feats instructions must use uvw paths shaped (channels, 1, channels).");
        field_offset += channel_pairs;
    }
    if (field_offset != field_weights.size())
        throw std::runtime_error("MACEField field_feats weight size does not match its instructions.");

    std::vector<std::size_t> linear_offsets;
    std::size_t linear_offset = 0;
    for (const auto& instruction : linear_instructions) {
        linear_offsets.push_back(linear_offset);
        const auto shape = instruction.at("path_shape").get<std::vector<int>>();
        if (shape != std::vector<int>{channels, channels})
            throw std::runtime_error(
                "MACEField field_linear instructions must be shaped (channels, channels).");
        const int input_block = instruction.at("i_in").get<int>();
        const int output_block = instruction.at("i_out").get<int>();
        if (input_block < 0 || input_block > L_max || output_block != input_block)
            throw std::runtime_error(
                "MACEField field_linear instructions must preserve each canonical angular block.");
        linear_offset += channel_pairs;
    }
    if (linear_offset != linear_weights.size())
        throw std::runtime_error("MACEField field_linear weight size does not match its instructions.");

    std::vector<FieldCouplingPath> paths;
    for (std::size_t fi = 0; fi < field_instructions.size(); ++fi) {
        const auto& field_instruction = field_instructions[fi];
        const int input_l = field_instruction.at("i_in1").get<int>();
        const int field_block = field_instruction.at("i_in2").get<int>();
        const int intermediate_l = field_instruction.at("i_out").get<int>();
        if (field_block != 0 || input_l < 0 || input_l > L_max
            || intermediate_l < 0 || intermediate_l > L_max
            || std::abs(input_l - intermediate_l) != 1)
            throw std::runtime_error(
                "MACEField field_feats instruction is not an adjacent-rank 1o coupling.");

        std::vector<double> wigner;
        if (legacy) {
            wigner.assign(
                static_cast<std::size_t>(2 * input_l + 1) * 3
                    * static_cast<std::size_t>(2 * intermediate_l + 1),
                0.0);
            const double coefficient = 1.0 / std::sqrt(3.0);
            for (int component = 0; component < 3; ++component) {
                const std::size_t index = input_l == 0
                    ? static_cast<std::size_t>(component * 3 + component)
                    : static_cast<std::size_t>(component * 4);
                wigner[index] = coefficient;
            }
        } else {
            const auto shape = field_instruction.at("wigner_3j_shape").get<std::vector<int>>();
            const std::vector<int> expected_shape{
                2 * input_l + 1, 3, 2 * intermediate_l + 1};
            if (shape != expected_shape)
                throw std::runtime_error("MACEField Wigner tensor shape does not match its angular path.");
            wigner = field_instruction.at("wigner_3j").get<std::vector<double>>();
            if (wigner.size() != static_cast<std::size_t>(shape[0] * shape[1] * shape[2]))
                throw std::runtime_error("MACEField Wigner tensor data has an invalid size.");
        }

        bool found_linear = false;
        for (std::size_t li = 0; li < linear_instructions.size(); ++li) {
            const auto& linear_instruction = linear_instructions[li];
            if (linear_instruction.at("i_in").get<int>() != intermediate_l)
                continue;
            found_linear = true;
            FieldCouplingPath path;
            path.input_l = input_l;
            path.output_l = linear_instruction.at("i_out").get<int>();
            const int input_components = 2 * input_l + 1;
            const int output_components = 2 * path.output_l + 1;
            for (int input_m = 0; input_m < input_components; ++input_m) {
                for (int component = 0; component < 3; ++component) {
                    for (int output_m = 0; output_m < output_components; ++output_m) {
                        const std::size_t index =
                            (static_cast<std::size_t>(input_m) * 3 + component)
                                * output_components + output_m;
                        if (std::abs(wigner[index]) > 1e-14) {
                            path.angular_entries.push_back({
                                input_l * input_l + input_m,
                                component,
                                path.output_l * path.output_l + output_m,
                                wigner[index]});
                        }
                    }
                }
            }

            path.channel_matrix.assign(channel_pairs, 0.0);
            const double path_scale = field_instruction.at("path_weight").get<double>()
                * linear_instruction.at("path_weight").get<double>();
            for (int input = 0; input < channels; ++input) {
                for (int output = 0; output < channels; ++output) {
                    double value = 0.0;
                    for (int intermediate = 0; intermediate < channels; ++intermediate) {
                        value += field_weights[
                            field_offsets[fi] + static_cast<std::size_t>(input) * channels
                                + intermediate]
                            * linear_weights[
                                linear_offsets[li] + static_cast<std::size_t>(intermediate) * channels
                                    + output];
                    }
                    path.channel_matrix[static_cast<std::size_t>(input) * channels + output]
                        = path_scale * value;
                }
            }

            path.channel_up_matrix.assign(channel_pairs, 0.0);
            const std::size_t up_offset = static_cast<std::size_t>(path.output_l) * channel_pairs;
            for (int input = 0; input < channels; ++input) {
                for (int output = 0; output < channels; ++output) {
                    double value = 0.0;
                    for (int intermediate = 0; intermediate < channels; ++intermediate) {
                        value += path.channel_matrix[
                            static_cast<std::size_t>(input) * channels + intermediate]
                            * H1_linear_up_weights[
                                up_offset + static_cast<std::size_t>(intermediate) * channels + output];
                    }
                    path.channel_up_matrix[
                        static_cast<std::size_t>(input) * channels + output] = value;
                }
            }
            paths.push_back(std::move(path));
        }
        if (!found_linear)
            throw std::runtime_error(
                "MACEField field_feats output has no compatible field_linear instruction.");
    }
    if (paths.empty())
        throw std::runtime_error("MACEField field coupling contains no executable paths.");
    return paths;
}
