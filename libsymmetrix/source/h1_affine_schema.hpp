#pragma once

// H1 affine basis within the common format-version-4 model container.
//
// legacy        (format 1/2, no h1_affine fields): readout_1_weights and
//               H2_weights_for_H1 act on the fused H1 = M0 C (C = P U), i.e.
//               they carry inv(U) from the exporter.
// m0_composite  (format 4, h1_affine_mode "m0_composite", h1_affine_version 1):
//               readout_1_weights_for_M0 (q = C r) and H2_weights_for_M0
//               (T_z = C S_z, same row-major [input][output] layout) act on the
//               scalar M0 block; H1_weights still feeds the second message
//               interaction only.  The legacy keys must be absent.
//
// Nonstandard families and single-layer MACE keep their existing payload in
// the v4 container and do not declare h1_affine fields.
//
// Scope of m0_composite: single-head two-interaction compact MACE without
// field coupling, first-interaction residual or single-layer readout.  Every
// executor that does not implement the composite path must refuse it.

#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace symmetrix::h1_affine {

// New evaluators start in the v4 representation. Loading a v1/v2 standard-MACE
// payload explicitly selects the legacy representation during schema validation.
enum class Mode { m0_composite, legacy };

inline constexpr int m0_composite_format_version = 4;
inline constexpr int m0_composite_affine_version = 1;

inline const char* mode_name(Mode mode)
{
    return mode == Mode::m0_composite ? "m0_composite" : "legacy";
}

// Validates the H1 affine contract of a head-selected MACE JSON and returns the
// declared basis.  Throws std::invalid_argument on any mixed, partial,
// out-of-scope or malformed payload.
inline Mode validate(const nlohmann::json& file, int num_elements, int num_channels)
{
    const int format_version = file.value("symmetrix_format_version", 1);
    const std::string model_type = file.value("model_type", std::string("MACE"));
    const bool declares_mode = file.contains("h1_affine_mode")
        || file.contains("h1_affine_version");
    const bool has_composite_keys = file.contains("readout_1_weights_for_M0")
        || file.contains("H2_weights_for_M0");
    if (model_type != "MACE") {
        if (declares_mode || has_composite_keys)
            throw std::invalid_argument(
                "Only standard MACE JSON may declare h1_affine fields.");
        return Mode::legacy;
    }
    if (format_version != m0_composite_format_version) {
        if (declares_mode || has_composite_keys)
            throw std::invalid_argument(
                "Symmetrix MACE format version " + std::to_string(format_version)
                + " must not declare h1_affine fields; the m0_composite basis "
                  "requires format version 4.");
        return Mode::legacy;
    }
    if (file.value("num_interactions", 2) == 1) {
        if (declares_mode || has_composite_keys)
            throw std::invalid_argument(
                "Single-layer MACE format version 4 must not declare h1_affine fields.");
        return Mode::legacy;
    }
    if (file.value("h1_affine_mode", std::string()) != "m0_composite")
        throw std::invalid_argument(
            "Symmetrix MACE format version 4 requires h1_affine_mode 'm0_composite'.");
    if (file.value("h1_affine_version", 0) != m0_composite_affine_version)
        throw std::invalid_argument(
            "Unsupported h1_affine_version for the m0_composite basis (supported: 1).");
    if (file.contains("readout_1_weights") || file.contains("H2_weights_for_H1"))
        throw std::invalid_argument(
            "An m0_composite MACE JSON must not contain the legacy readout_1_weights or "
            "H2_weights_for_H1 fields.");
    if (file.contains("prediction_heads"))
        throw std::invalid_argument(
            "The m0_composite basis supports single-head MACE JSON only.");
    if (file.value("model_type", std::string("MACE")) != "MACE"
            || file.value("num_interactions", 2) != 2
            || file.value("single_layer_readout", false)
            || file.value("has_field_coupling", false)
            || file.value("first_interaction_residual", false)
            || file.value("radial_representation", std::string()) != "compact")
        throw std::invalid_argument(
            "The m0_composite basis supports only two-interaction compact MACE without "
            "field coupling, first-interaction residual or single-layer readout.");
    if (!file.contains("readout_1_weights_for_M0") || !file.contains("H2_weights_for_M0"))
        throw std::invalid_argument(
            "An m0_composite MACE JSON requires readout_1_weights_for_M0 and "
            "H2_weights_for_M0.");
    const auto& readout = file.at("readout_1_weights_for_M0");
    if (!readout.is_array() || readout.size() != static_cast<std::size_t>(num_channels))
        throw std::invalid_argument(
            "readout_1_weights_for_M0 must hold num_channels values.");
    const auto& skip = file.at("H2_weights_for_M0");
    if (!skip.is_array() || skip.size() != static_cast<std::size_t>(num_elements))
        throw std::invalid_argument(
            "H2_weights_for_M0 must hold one row per element.");
    for (const auto& row : skip)
        if (!row.is_array()
                || row.size() != static_cast<std::size_t>(num_channels)*num_channels)
            throw std::invalid_argument(
                "Every H2_weights_for_M0 row must hold num_channels^2 values.");
    return Mode::m0_composite;
}

}  // namespace symmetrix::h1_affine
