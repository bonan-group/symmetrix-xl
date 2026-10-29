#pragma once

// H1 affine basis of a standard MACE / MACEField JSON (format version 4).
//
// The first readout and the H1->H2 skip act on the input of interactions[1],
// i.e. on the features *before* interactions[1].linear_up (U).  Format
// versions 1 and 2 stored them after U, which needs inv(U) at export time
// (still loaded, with a warning; see below).  Format 4 stores them in one of
// two inverse-free bases:
//
// m0_composite   readout_1_weights_for_M0 (q = P0 r0 * scale, K values) and
//                H2_weights_for_M0 (T_z = P0 S0_z, row-major [input][output])
//                act on the scalar M0 block.  Used when the pre-linear_up
//                features are linear in M0 (no first-interaction residual, no
//                field coupling) and for single-layer models (zero maps).
// pre_linear_up  readout_1_weights_for_H1_pre_linear_up (r0 * scale) and
//                H2_weights_for_H1_pre_linear_up (S0_z) are the checkpoint's own
//                maps on the scalar block of the pre-linear_up features
//                (P0 M0 + first-interaction residual + field coupling).  Used for
//                two-interaction models whose JSON keeps H1_product_weights and
//                H1_linear_up_weights split (residual-first, MACEField).
//
// H1_weights (C = P U) still feeds the second message interaction.  In format
// 4 the readout_1_weights / H2_weights_for_H1 keys of the old basis are absent.
//
// Format 1/2 files (old basis, Mode::legacy) are still evaluated with the old
// arithmetic; validate() prints one warning per load with a cancellation
// estimate computed from the file (legacy_cancellation) and recommends
// re-export or symmetrix_recompose_mace_json.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace symmetrix::h1_affine {

// legacy: format 1/2, readout_1_weights / H2_weights_for_H1 after linear_up.
enum class Mode { legacy, m0_composite, pre_linear_up };

inline constexpr int format_version = 4;
inline constexpr int affine_version = 1;
// Kept for existing callers.
inline constexpr int m0_composite_format_version = format_version;

inline const char* mode_name(Mode mode)
{
    switch (mode) {
    case Mode::m0_composite: return "m0_composite";
    case Mode::pre_linear_up: return "pre_linear_up";
    default: return "legacy";
    }
}

inline const char* readout_key(Mode mode)
{
    return mode == Mode::pre_linear_up
        ? "readout_1_weights_for_H1_pre_linear_up" : "readout_1_weights_for_M0";
}

inline const char* skip_key(Mode mode)
{
    return mode == Mode::pre_linear_up
        ? "H2_weights_for_H1_pre_linear_up" : "H2_weights_for_M0";
}

// fp32 cancellation estimate of a format-1/2 file, from the file alone.  The
// old runtime forms the scalar M0 (or, with field coupling, pre-linear_up)
// adjoint as r A^T and S_z g A^T with A = C = P U (H1_weights, l = 0) or A = U
// (H1_linear_up_weights, field models); in working precision the rounding of
// those sums scales with the magnitude of their terms, |A| |r|, while the
// result is A r = P r0 (or r0).  The estimate is the ratio of the largest term
// sum to the largest result component (readout), and the same for the skip
// over every element; 1 when the maps are zero (single-layer models).
struct LegacyCancellation {
    double readout = 1.0;
    double skip = 1.0;
};

inline double cancellation_ratio(double terms, double result)
{
    if (result > 0.0)
        return terms/result;
    return terms > 0.0 ? INFINITY : 1.0;
}

inline LegacyCancellation legacy_cancellation(
    const nlohmann::json& file, int num_elements, int num_channels)
{
    LegacyCancellation out;
    const std::size_t K = static_cast<std::size_t>(num_channels);
    const bool field = file.value("has_field_coupling", false);
    const char* map_key = field ? "H1_linear_up_weights" : "H1_weights";
    if (!file.contains(map_key) || !file.contains("readout_1_weights")
            || !file.contains("H2_weights_for_H1"))
        return out;
    const auto A = file.at(map_key).get<std::vector<double>>();
    const auto r = file.at("readout_1_weights").get<std::vector<double>>();
    if (A.size() < K*K || r.size() != K)
        return out;
    double terms = 0.0, result = 0.0;
    for (std::size_t j=0; j<K; ++j) {
        double t = 0.0, q = 0.0;
        for (std::size_t k=0; k<K; ++k) {
            t += std::abs(A[j*K+k]*r[k]);
            q += A[j*K+k]*r[k];
        }
        terms = std::max(terms, t);
        result = std::max(result, std::abs(q));
    }
    out.readout = cancellation_ratio(terms, result);
    const auto& rows = file.at("H2_weights_for_H1");
    double worst = 1.0;
    for (int z=0; z<num_elements && z<static_cast<int>(rows.size()); ++z) {
        const auto S = rows.at(z).get<std::vector<double>>();
        if (S.size() != K*K)
            continue;
        double z_terms = 0.0, z_result = 0.0;
        std::vector<double> t(K), q(K);
        for (std::size_t j=0; j<K; ++j) {
            std::fill(t.begin(), t.end(), 0.0);
            std::fill(q.begin(), q.end(), 0.0);
            for (std::size_t k=0; k<K; ++k) {
                const double a = A[j*K+k];
                for (std::size_t c=0; c<K; ++c) {
                    t[c] += std::abs(a*S[k*K+c]);
                    q[c] += a*S[k*K+c];
                }
            }
            for (std::size_t c=0; c<K; ++c) {
                z_terms = std::max(z_terms, t[c]);
                z_result = std::max(z_result, std::abs(q[c]));
            }
        }
        worst = std::max(worst, cancellation_ratio(z_terms, z_result));
    }
    out.skip = worst;
    return out;
}

inline std::string legacy_format_message(int version, const LegacyCancellation& estimate)
{
    char numbers[96];
    std::snprintf(numbers, sizeof(numbers), "readout %.3g, skip %.3g",
                  estimate.readout, estimate.skip);
    return "Symmetrix MACE JSON format version " + std::to_string(version)
        + " stores readout_1 and the H1->H2 skip multiplied by inv(linear_up) "
          "(fp32 cancellation estimate from this file: " + numbers
        + "); it is evaluated with the old arithmetic. Re-export the checkpoint "
          "with `symmetrix_extract_mace` or convert the file with "
          "`symmetrix_recompose_mace_json --input OLD.json --output NEW.json` "
          "(format version 4).";
}

// Validates the H1 affine contract of a head-selected MACE JSON and returns its
// basis.  Format 1/2 returns Mode::legacy after printing the warning above;
// throws std::invalid_argument for any mixed, partial, out-of-scope or
// malformed format-4 payload.
inline Mode validate(const nlohmann::json& file, int num_elements, int num_channels)
{
    const int version = file.value("symmetrix_format_version", 1);
    if (version == 1 || version == 2) {
        if (file.contains("h1_affine_mode") || file.contains("h1_affine_version")
                || file.contains(readout_key(Mode::m0_composite))
                || file.contains(skip_key(Mode::m0_composite))
                || file.contains(readout_key(Mode::pre_linear_up))
                || file.contains(skip_key(Mode::pre_linear_up)))
            throw std::invalid_argument(
                "Symmetrix MACE format version " + std::to_string(version)
                + " must not declare h1_affine fields; they require format version 4.");
        std::cerr << "Warning: " << legacy_format_message(
            version, legacy_cancellation(file, num_elements, num_channels)) << std::endl;
        return Mode::legacy;
    }
    if (version != format_version)
        throw std::invalid_argument("Unsupported Symmetrix model format version.");
    const std::string name = file.value("h1_affine_mode", std::string());
    Mode mode = Mode::legacy;
    if (name == "m0_composite")
        mode = Mode::m0_composite;
    else if (name == "pre_linear_up")
        mode = Mode::pre_linear_up;
    else
        throw std::invalid_argument(
            "Symmetrix MACE format version 4 requires h1_affine_mode 'm0_composite' "
            "or 'pre_linear_up'.");
    if (file.value("h1_affine_version", 0) != affine_version)
        throw std::invalid_argument(
            "Unsupported h1_affine_version (supported: 1).");
    if (file.contains("readout_1_weights") || file.contains("H2_weights_for_H1"))
        throw std::invalid_argument(
            "A format-version-4 MACE JSON must not contain the old-basis "
            "readout_1_weights or H2_weights_for_H1 fields.");
    const Mode other = mode == Mode::m0_composite ? Mode::pre_linear_up : Mode::m0_composite;
    if (file.contains(readout_key(other)) || file.contains(skip_key(other)))
        throw std::invalid_argument(
            std::string("A '") + mode_name(mode) + "' MACE JSON must not contain "
            + readout_key(other) + " or " + skip_key(other) + ".");
    const std::string model_type = file.value("model_type", std::string("MACE"));
    if (model_type != "MACE" && model_type != "MACEField")
        throw std::invalid_argument(
            "Format version 4 is the standard MACE / MACEField format.");
    const std::string radial = file.value("radial_representation", std::string());
    if (radial != "compact" && radial != "pair-splines")
        throw std::invalid_argument(
            "Format version 4 requires radial_representation 'compact' or 'pair-splines'.");
    const bool single_layer = file.value("single_layer_readout", false);
    const bool field = file.value("has_field_coupling", false);
    const bool residual = file.value("first_interaction_residual", false);
    if (mode == Mode::m0_composite && !single_layer && (field || residual))
        throw std::invalid_argument(
            "The m0_composite basis cannot represent first-interaction residual or "
            "field-coupled features; such models use h1_affine_mode 'pre_linear_up'.");
    if (mode == Mode::pre_linear_up
        && (single_layer || file.value("num_interactions", 2) != 2))
        throw std::invalid_argument(
            "The pre_linear_up basis applies to two-interaction models only.");
    if (mode == Mode::pre_linear_up && !file.contains("H1_product_weights"))
        throw std::invalid_argument(
            "The pre_linear_up basis requires H1_product_weights.");
    if (!file.contains(readout_key(mode)) || !file.contains(skip_key(mode)))
        throw std::invalid_argument(
            std::string("A '") + mode_name(mode) + "' MACE JSON requires "
            + readout_key(mode) + " and " + skip_key(mode) + ".");
    const auto& readout = file.at(readout_key(mode));
    if (!readout.is_array() || readout.size() != static_cast<std::size_t>(num_channels))
        throw std::invalid_argument(
            std::string(readout_key(mode)) + " must hold num_channels values.");
    const auto& skip = file.at(skip_key(mode));
    if (!skip.is_array() || skip.size() != static_cast<std::size_t>(num_elements))
        throw std::invalid_argument(
            std::string(skip_key(mode)) + " must hold one row per element.");
    for (const auto& row : skip)
        if (!row.is_array()
                || row.size() != static_cast<std::size_t>(num_channels)*num_channels)
            throw std::invalid_argument(
                std::string("Every ") + skip_key(mode) + " row must hold num_channels^2 values.");
    return mode;
}

}  // namespace symmetrix::h1_affine
