#include "oam_schema.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <set>
#include <limits>
#include <stdexcept>
#include <string>
#include <sstream>

#include "sha256.hpp"

namespace {

constexpr std::size_t max_feature_dimension = 1u << 20;
constexpr std::size_t max_tensor_elements = 1u << 30;

[[noreturn]] void fail(const std::string& path, const std::string& message)
{
    throw std::invalid_argument("NequIP schema " + path + " " + message);
}

const nlohmann::json& object_member(
    const nlohmann::json& object, const char* key, const std::string& path)
{
    if (!object.is_object() || !object.contains(key))
        fail(path, std::string("must contain ") + key + ".");
    return object.at(key);
}

std::vector<int> checked_shape(const nlohmann::json& tensor, const std::string& path)
{
    const auto& shape = object_member(tensor, "shape", path);
    const auto& values = object_member(tensor, "values", path);
    if (!shape.is_array() || !values.is_array()) fail(path, "shape and values must be arrays.");
    std::size_t expected = 1;
    std::vector<int> result;
    result.reserve(shape.size());
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (!shape.at(index).is_number_integer())
            fail(path + ".shape[" + std::to_string(index) + "]", "must be an integer.");
        const auto extent = shape.at(index).get<std::int64_t>();
        if (extent < 0 || static_cast<std::uint64_t>(extent)
                > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
            fail(path + ".shape[" + std::to_string(index) + "]", "is outside the supported range.");
        const auto converted = static_cast<std::size_t>(extent);
        if (converted != 0 && expected > max_tensor_elements / converted)
            fail(path + ".shape", "product exceeds the supported tensor bound.");
        expected *= converted;
        result.push_back(static_cast<int>(extent));
    }
    if (expected != values.size())
        fail(path + ".values", "length does not agree with shape product.");
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!values.at(index).is_number() || !std::isfinite(values.at(index).get<double>()))
            fail(path + ".values[" + std::to_string(index) + "]", "must be finite.");
    }
    return result;
}


bool sha256_fingerprint(const nlohmann::json& value)
{
    if (!value.is_string()) return false;
    const auto text = value.get<std::string>();
    if (text.size() != 71 || text.rfind("sha256:", 0) != 0) return false;
    return std::all_of(text.begin()+7, text.end(), [](unsigned char value) {
        return std::isdigit(value) || (value >= 'a' && value <= 'f');
    });
}

// Keys the exporter drops before hashing a contract (dtype/device style runtime
// settings that do not change the model semantics).
constexpr const char* runtime_keys[] = {"dtype", "device", "precision", "runtime_dtype"};

// One irrep block as (offset, multiplicity, angular width), and the same list
// with adjacent equal-width blocks merged into runs.  The node-major element
// order of a tensor is fixed by the ordered component widths, so two
// descriptions with equal runs hold the same (l, m) component at every index:
// splitting or merging adjacent same-angular blocks is index-preserving, while
// any other difference re-indexes the row.
struct IrrepRun {
    int offset, multiplicity, width;
};

std::vector<IrrepRun> irrep_runs(const Irreps& irreps)
{
    std::vector<IrrepRun> runs;
    for (const auto& block : irreps.blocks) {
        const int width = 2 * block.l + 1;
        if (!runs.empty() && runs.back().width == width
            && runs.back().offset
                + runs.back().multiplicity * runs.back().width == block.offset) {
            runs.back().multiplicity += block.multiplicity;
            continue;
        }
        runs.push_back({block.offset, block.multiplicity, width});
    }
    return runs;
}

bool same_node_major_order(const Irreps& first, const Irreps& second)
{
    const auto left = irrep_runs(first);
    const auto right = irrep_runs(second);
    return left.size() == right.size()
        && std::equal(left.begin(), left.end(), right.begin(),
            [](const IrrepRun& a, const IrrepRun& b) {
                return a.offset == b.offset
                    && a.multiplicity == b.multiplicity
                    && a.width == b.width;
            });
}

// Python's `repr` for a finite double: shortest round-trip digits, fixed
// notation while -4 <= decimal exponent < 16 and scientific otherwise, with an
// integral value always keeping a trailing ".0".  The canonical contract
// fingerprint is defined over this spelling, so it has to be reproduced
// exactly rather than approximated with a locale or iostream format.
void write_python_number(double value, std::string& out)
{
    if (!std::isfinite(value)) fail("$", "contains a non-finite number.");
    char buffer[64];
    const auto converted = std::to_chars(
        buffer, buffer + sizeof buffer, value, std::chars_format::scientific);
    if (converted.ec != std::errc()) fail("$", "contains an unrepresentable number.");
    std::string digits;
    int exponent = 0;
    int exponent_sign = 1;
    bool in_exponent = false;
    bool negative = false;
    for (const char* cursor = buffer; cursor != converted.ptr; ++cursor) {
        const char character = *cursor;
        if (character == '.') continue;
        if (character == 'e' || character == 'E') { in_exponent = true; continue; }
        if (in_exponent) {
            if (character == '-') { exponent_sign = -1; continue; }
            if (character == '+') { exponent_sign = 1; continue; }
            exponent = exponent * 10 + (character - '0');
            continue;
        }
        if (character == '-') { negative = true; continue; }
        digits.push_back(character);
    }
    exponent *= exponent_sign;
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    const int decimal_point = exponent + 1;
    if (negative) out.push_back('-');
    if (decimal_point <= -4 || decimal_point > 16) {
        out.push_back(digits.front());
        if (digits.size() > 1) {
            out.push_back('.');
            out.append(digits, 1, std::string::npos);
        }
        out.push_back('e');
        int written = decimal_point - 1;
        out.push_back(written < 0 ? '-' : '+');
        if (written < 0) written = -written;
        std::string tail = std::to_string(written);
        if (tail.size() < 2) tail.insert(tail.begin(), '0');
        out.append(tail);
        return;
    }
    if (decimal_point <= 0) {
        out.append("0.");
        out.append(static_cast<std::size_t>(-decimal_point), '0');
        out.append(digits);
        return;
    }
    if (static_cast<int>(digits.size()) <= decimal_point) {
        out.append(digits);
        out.append(static_cast<std::size_t>(decimal_point) - digits.size(), '0');
        out.append(".0");
        return;
    }
    out.append(digits, 0, static_cast<std::size_t>(decimal_point));
    out.push_back('.');
    out.append(digits, static_cast<std::size_t>(decimal_point), std::string::npos);
}

// `json.dumps(..., ensure_ascii=True)` escaping: every code point above the
// ASCII range is written as one or two \uXXXX units.
void write_python_string(const std::string& text, std::string& out)
{
    static const char* hex = "0123456789abcdef";
    const auto write_unit = [&out](unsigned int unit) {
        out += "\\u";
        out.push_back(hex[(unit >> 12) & 0xf]);
        out.push_back(hex[(unit >> 8) & 0xf]);
        out.push_back(hex[(unit >> 4) & 0xf]);
        out.push_back(hex[unit & 0xf]);
    };
    for (std::size_t index = 0; index < text.size();) {
        const unsigned char byte = static_cast<unsigned char>(text[index]);
        if (byte == '"') { out += "\\\""; ++index; continue; }
        if (byte == '\\') { out += "\\\\"; ++index; continue; }
        if (byte == '\n') { out += "\\n"; ++index; continue; }
        if (byte == '\r') { out += "\\r"; ++index; continue; }
        if (byte == '\t') { out += "\\t"; ++index; continue; }
        if (byte == '\b') { out += "\\b"; ++index; continue; }
        if (byte == '\f') { out += "\\f"; ++index; continue; }
        if (byte >= 0x20 && byte < 0x7f) { out.push_back(static_cast<char>(byte)); ++index; continue; }
        unsigned int code_point = 0;
        int continuation = 0;
        if (byte < 0x80) { code_point = byte; }
        else if ((byte >> 5) == 0x6) { code_point = byte & 0x1fu; continuation = 1; }
        else if ((byte >> 4) == 0xe) { code_point = byte & 0x0fu; continuation = 2; }
        else if ((byte >> 3) == 0x1e) { code_point = byte & 0x07u; continuation = 3; }
        else fail("$", "contains invalid UTF-8 text.");
        ++index;
        for (int extra = 0; extra < continuation; ++extra) {
            if (index == text.size()) fail("$", "contains truncated UTF-8 text.");
            code_point = (code_point << 6)
                | (static_cast<unsigned char>(text[index]) & 0x3fu);
            ++index;
        }
        if (code_point > 0xffff) {
            const unsigned int adjusted = code_point - 0x10000u;
            write_unit(0xd800u + (adjusted >> 10));
            write_unit(0xdc00u + (adjusted & 0x3ffu));
        } else write_unit(code_point);
    }
}

void write_canonical_contract(
    const nlohmann::json& value,
    std::string& out,
    bool root = false,
    bool model_metadata = false)
{
    if (value.is_object()) {
        out.push_back('{');
        bool first = true;
        // nlohmann::json keeps object members sorted by key, which is the
        // ordering `json.dumps(..., sort_keys=True)` uses as well.
        for (auto member = value.begin(); member != value.end(); ++member) {
            if (model_metadata && member.key() == "archive") continue;
            bool skip = false;
            for (const auto* runtime_key : runtime_keys) {
                if (member.key() != runtime_key) continue;
                // `precision` is a root-level runtime setting in the exporter's
                // canonical form; a nested `precision` tag, such as the radial
                // table precision, is part of the contract identity.
                if (member.key() == "precision" && !root) continue;
                skip = true;
            }
            if (skip) continue;
            if (!first) out.push_back(',');
            first = false;
            out.push_back('"');
            write_python_string(member.key(), out);
            out += "\":";
            write_canonical_contract(
                member.value(), out, false,
                root && member.key() == "model_metadata");
        }
        out.push_back('}');
        return;
    }
    if (value.is_array()) {
        out.push_back('[');
        bool first = true;
        for (const auto& item : value) {
            if (!first) out.push_back(',');
            first = false;
            write_canonical_contract(item, out);
        }
        out.push_back(']');
        return;
    }
    if (value.is_string()) {
        out.push_back('"');
        write_python_string(value.get<std::string>(), out);
        out.push_back('"');
        return;
    }
    if (value.is_boolean()) { out += value.get<bool>() ? "true" : "false"; return; }
    if (value.is_number_float()) {
        write_python_number(value.get<double>(), out);
        return;
    }
    if (value.is_number_integer()) {
        out += std::to_string(value.get<std::int64_t>());
        return;
    }
    if (value.is_number_unsigned()) {
        out += std::to_string(value.get<std::uint64_t>());
        return;
    }
    if (value.is_null()) { out += "null"; return; }
    fail("$", "contains a value that cannot be fingerprinted.");
}

// Identity of a contract, byte-identical to the exporter's
// `nequip_contract_fingerprint`: SHA-256 over the canonical compact JSON
// spelling of the document with the exporter's runtime keys removed.
std::string contract_fingerprint(const nlohmann::json& root)
{
    std::string canonical;
    write_canonical_contract(root, canonical, true);
    sha256::Digest hash;
    hash.update(canonical);
    return hash.finish();
}

OAMRadialSplineContract checked_kokkos_radial(
    const nlohmann::json& value, const std::string& path,
    int expected_function_count, double expected_cutoff)
{
    if (!value.is_object()) fail(path, "must be an object.");
    for (const auto key : {"pair_tables", "element_pair_tables", "edge_tables", "tables"})
        if (value.contains(key)) fail(path + "." + key, "element-pair radial tables are unsupported.");
    for (const auto key : {"grid_min", "cutoff", "num_spline_points", "spacing",
        "function_count", "edge_type_count", "model_fingerprint", "table_fingerprint",
        "precision", "exporter_version", "certification"})
        object_member(value, key, path);
    OAMRadialSplineContract result;
    result.grid_min = value.at("grid_min").get<double>();
    result.cutoff = value.at("cutoff").get<double>();
    result.num_points = value.at("num_spline_points").get<int>();
    result.spacing = value.at("spacing").get<double>();
    result.function_count = value.at("function_count").get<int>();
    result.edge_type_count = value.at("edge_type_count").get<int>();
    result.model_fingerprint = value.at("model_fingerprint").get<std::string>();
    result.table_fingerprint = value.at("table_fingerprint").get<std::string>();
    result.precision = value.at("precision").get<std::string>();
    result.exporter_version = value.at("exporter_version").get<std::string>();
    result.certification = value.at("certification");
    if (!std::isfinite(result.grid_min) || result.grid_min < 0.0
        || !std::isfinite(result.cutoff) || result.cutoff <= result.grid_min
        || std::abs(result.cutoff-expected_cutoff) > 1.0e-12
        || result.num_points < 4 || !std::isfinite(result.spacing) || result.spacing <= 0.0
        || std::abs(result.spacing-(result.cutoff-result.grid_min)/(result.num_points-1)) > 1.0e-12
        || result.function_count != expected_function_count || result.edge_type_count != 1
        || !sha256_fingerprint(result.model_fingerprint)
        || !sha256_fingerprint(result.table_fingerprint)
        || (result.precision != "float32" && result.precision != "float64")
        || result.exporter_version.empty())
        fail(path, "has invalid compact radial grid, function, fingerprint, or precision metadata.");
    const auto& certification = result.certification;
    if (!certification.is_object()
        || !certification.contains("grid_points") || !certification.contains("clamped_points")
        || !certification.contains("max_abs_value_error")
        || !certification.contains("max_abs_derivative_error")
        || !certification.at("grid_points").is_number_integer()
        || certification.at("grid_points").get<int>() != result.num_points
        || !certification.at("clamped_points").is_number_integer()
        || certification.at("clamped_points").get<int>() < 0
        || certification.at("clamped_points").get<int>() > result.num_points
        || !certification.at("max_abs_value_error").is_number()
        || !certification.at("max_abs_derivative_error").is_number()
        || !std::isfinite(certification.at("max_abs_value_error").get<double>())
        || !std::isfinite(certification.at("max_abs_derivative_error").get<double>()))
        fail(path + ".certification", "is incomplete or invalid.");
    result.certified = true;
    return result;
}

Irreps checked_irreps(const nlohmann::json& value, const std::string& path)
{
    if (!value.is_string() || value.get<std::string>().empty()) fail(path, "must be a non-empty string.");
    const auto specification = value.get<std::string>();
    std::size_t token_start = 0;
    std::size_t aggregate_dimension = 0;
    while (token_start < specification.size()) {
        const auto token_end = specification.find('+', token_start);
        auto token = specification.substr(token_start, token_end == std::string::npos
            ? std::string::npos : token_end-token_start);
        token.erase(std::remove_if(token.begin(), token.end(), [](unsigned char character) {
            return std::isspace(character);
        }), token.end());
        const auto x = token.find('x');
        const auto parity = token.find_first_of("eo", x == std::string::npos ? 0 : x+1);
        if (x == std::string::npos || parity == std::string::npos || parity + 1 != token.size())
            fail(path, "contains an invalid irrep token.");
        try {
            const auto multiplicity_text = token.substr(0, x);
            const auto degree_text = token.substr(x+1, parity-x-1);
            if (multiplicity_text.empty() || multiplicity_text.front() == '0'
                || !std::all_of(multiplicity_text.begin(), multiplicity_text.end(),
                    [](unsigned char character) { return std::isdigit(character); })
                || degree_text.empty()
                || !std::all_of(degree_text.begin(), degree_text.end(),
                    [](unsigned char character) { return std::isdigit(character); }))
                fail(path, "contains an invalid irrep token.");
            const auto multiplicity = std::stoull(multiplicity_text);
            const auto degree = std::stoull(degree_text);
            if (multiplicity == 0 || degree > max_feature_dimension
                || multiplicity > max_feature_dimension
                || multiplicity > max_feature_dimension / (2*degree+1)
                || aggregate_dimension > max_feature_dimension
                    - multiplicity*(2*degree+1))
                fail(path, "contains an irrep block beyond the feature bound.");
            aggregate_dimension += multiplicity*(2*degree+1);
        } catch (const std::exception&) {
            fail(path, "contains an invalid irrep block.");
        }
        if (token_end == std::string::npos) break;
        if (token_end + 1 == specification.size())
            fail(path, "contains a trailing '+'.");
        token_start = token_end + 1;
    }
    try {
        Irreps irreps(specification);
        if (irreps.dimension() <= 0
            || static_cast<std::size_t>(irreps.dimension()) > max_feature_dimension)
            fail(path, "dimension exceeds the supported feature bound.");
        for (std::size_t index = 0; index < irreps.blocks.size(); ++index) {
            if (irreps.blocks[index].multiplicity <= 0 || irreps.blocks[index].l < 0
                || irreps.blocks[index].dimension() <= 0)
                fail(path, "contains an invalid block.");
        }
        return irreps;
    } catch (const std::exception& error) {
        throw std::invalid_argument("NequIP schema " + path + " is invalid: " + error.what());
    }
}

nlohmann::json generated_architecture(const nlohmann::json& root)
{
    nlohmann::json layers = nlohmann::json::array();
    for (std::size_t index = 0; index < root.at("layers").size(); ++index) {
        const auto& layer = root.at("layers").at(index);
        const auto& product = layer.at("edge_tensor_product");
        const auto& radial = layer.at("kokkos_radial");
        const auto source_irreps =
            layer.at("pre_linear").at("irreps_out").get<std::string>();
        const auto edge_irreps = layer.at("edge_irreps").get<std::string>();
        const auto output_irreps = layer.at("tp_output_irreps").get<std::string>();
        const auto source = checked_irreps(source_irreps, "generated source irreps");
        const auto edge = checked_irreps(edge_irreps, "generated edge irreps");
        const auto output = checked_irreps(output_irreps, "generated output irreps");
        int edge_l_max = 0;
        for (const auto& block : edge.blocks)
            edge_l_max = std::max(edge_l_max, block.l);
        nlohmann::json tensor_product = nlohmann::json::object();
        for (const auto* key : {
                 "irreps_in1", "irreps_in2", "irreps_out", "instructions",
                 "output_mask", "weight_numel"})
            tensor_product[key] = product.at(key);
        layers.push_back({
            {"index", index},
            {"source_irreps", source_irreps},
            {"source_dimension", source.dimension()},
            {"edge_irreps", edge_irreps},
            {"edge_dimension", edge.dimension()},
            {"edge_l_max", edge_l_max},
            {"output_irreps", output_irreps},
            {"output_dimension", output.dimension()},
            {"tensor_product", std::move(tensor_product)},
            {"spline_function_count", radial.at("function_count").get<int>()},
            {"spline_interval_count",
                radial.at("num_spline_points").get<int>() - 1},
            {"normalization_factor",
                layer.at("normalization").value("factor", 1.0)},
        });
    }
    return {
        {"schema", "symmetrix.oam.generated-architecture/1"},
        {"layer_count", layers.size()},
        {"layers", std::move(layers)},
    };
}

} // namespace

NequIPOAMModelSchema load_nequip_oam_schema(const std::string& filename)
{
    std::ifstream stream(filename);
    if (!stream) throw std::runtime_error("Could not open NequIP OAM model file: " + filename);
    nlohmann::json data;
    try { data = nlohmann::json::parse(stream); }
    catch (const std::exception& error) { throw std::invalid_argument("NequIP OAM schema JSON parse failed: " + std::string(error.what())); }
    if (!data.is_object() || data.value("model_type", std::string{}) != "NequIP"
        || data.value("architecture", std::string{}) != "oam"
        || data.value("symmetrix_nequip_format_version", 0) != 2)
        throw std::invalid_argument("NequIP OAM schema requires model_type=NequIP, architecture=oam, and format version 2.");
    for (const auto key : {"model_metadata", "atomic_numbers", "model_atomic_numbers", "atomic_number_to_model_type", "r_cut", "edge_embedding", "node_attributes", "layers", "readout", "per_type_shift", "per_type_scale", "zbl"})
        object_member(data, key, "$");
    const auto& metadata = data.at("model_metadata");
    const auto fingerprint = metadata.value("exporter_fingerprint", std::string{});
    if (fingerprint.rfind("sha256:", 0) != 0 || fingerprint.size() <= 7)
        fail("$.model_metadata.exporter_fingerprint", "must start with sha256:.");
    if (!data.at("atomic_numbers").is_array() || !data.at("model_atomic_numbers").is_array()
        || data.at("atomic_numbers").empty() || data.at("model_atomic_numbers").empty())
        fail("$.atomic_numbers", "must be non-empty arrays.");
    if (!data.at("atomic_number_to_model_type").is_object()
        || data.at("atomic_number_to_model_type").size() != data.at("model_atomic_numbers").size())
        fail("$.atomic_number_to_model_type", "must map every model atomic number exactly once.");
    if (!data.at("r_cut").is_number() || !std::isfinite(data.at("r_cut").get<double>()) || data.at("r_cut").get<double>() <= 0.0)
        fail("$.r_cut", "must be positive and finite.");
    const auto& embedding = data.at("edge_embedding");
    const auto& frequencies = object_member(embedding, "bessel_weights", "$.edge_embedding");
    const auto frequency_shape = checked_shape(frequencies, "$.edge_embedding.bessel_weights");
    if (frequency_shape.size() != 1 || frequency_shape[0] <= 0)
        fail("$.edge_embedding.bessel_weights.shape", "must be a non-empty rank-one tensor.");
    const auto& cutoff = object_member(embedding, "cutoff", "$.edge_embedding");
    if (cutoff.value("type", std::string{}) != "polynomial" || !cutoff.contains("p")
        || !cutoff.at("p").is_number() || !std::isfinite(cutoff.at("p").get<double>()) || cutoff.at("p").get<double>() <= 0.0)
        fail("$.edge_embedding.cutoff", "must define a positive polynomial exponent.");
    if (!embedding.contains("factor") || !embedding.at("factor").is_number() || !std::isfinite(embedding.at("factor").get<double>()))
        fail("$.edge_embedding.factor", "must be finite.");
    if (!data.at("layers").is_array() || data.at("layers").empty())
        fail("$.layers", "must be a non-empty array.");
    const auto& zbl = data.at("zbl");
    if (zbl.value("type", std::string{}) != "zbl" || !zbl.contains("atomic_numbers")
        || !zbl.contains("screening_exponents") || !zbl.contains("screening_coefficients"))
        fail("$.zbl", "must define zbl atomic numbers and screening arrays.");
    if (!zbl.at("atomic_numbers").is_array() || !zbl.at("screening_exponents").is_array()
        || !zbl.at("screening_coefficients").is_array()
        || zbl.at("atomic_numbers").empty()
        || zbl.at("screening_exponents").empty()
        || zbl.at("screening_exponents").size() != zbl.at("screening_coefficients").size())
        fail("$.zbl", "screening arrays must be non-empty and have matching lengths.");
    if (!zbl.contains("cutoff") || !zbl.at("cutoff").is_number() || zbl.at("cutoff").get<double>() <= 0.0
        || !zbl.contains("half_energy_prefactor") || !zbl.at("half_energy_prefactor").is_number()
        || zbl.at("half_energy_prefactor").get<double>() <= 0.0)
        fail("$.zbl", "cutoff and half_energy_prefactor must be positive.");
    if (zbl.contains("cutoff_p")
        && (!zbl.at("cutoff_p").is_number() || !std::isfinite(zbl.at("cutoff_p").get<double>())
            || zbl.at("cutoff_p").get<double>() <= 0.0))
        fail("$.zbl.cutoff_p", "must be positive and finite when present.");
    NequIPOAMModelSchema result;
    result.atomic_numbers = data.at("atomic_numbers").get<std::vector<int>>();
    result.model_atomic_numbers = data.at("model_atomic_numbers").get<std::vector<int>>();
    const auto zbl_numbers = zbl.at("atomic_numbers").get<std::vector<int>>();
    auto sorted_zbl_numbers = zbl_numbers;
    auto sorted_model_numbers = result.model_atomic_numbers;
    std::sort(sorted_zbl_numbers.begin(), sorted_zbl_numbers.end());
    std::sort(sorted_model_numbers.begin(), sorted_model_numbers.end());
    if (sorted_zbl_numbers != sorted_model_numbers)
        fail("$.zbl.atomic_numbers", "must match the complete model element table.");
    if (std::set<int>(result.atomic_numbers.begin(), result.atomic_numbers.end()).size()
            != result.atomic_numbers.size()
        || std::set<int>(result.model_atomic_numbers.begin(), result.model_atomic_numbers.end()).size()
            != result.model_atomic_numbers.size())
        fail("$.atomic_numbers", "must not contain duplicates.");
    std::set<int> model_numbers(result.model_atomic_numbers.begin(), result.model_atomic_numbers.end());
    std::set<int> universal_numbers(result.atomic_numbers.begin(), result.atomic_numbers.end());
    for (const auto number : result.model_atomic_numbers)
        if (!universal_numbers.contains(number))
            fail("$.model_atomic_numbers", "contains an atomic number absent from atomic_numbers.");
    std::set<int> mapping_values;
    for (const auto number : result.model_atomic_numbers) {
        const auto key = std::to_string(number);
        if (!data.at("atomic_number_to_model_type").contains(key)
                || !data.at("atomic_number_to_model_type").at(key).is_number_integer())
            fail("$.atomic_number_to_model_type", "must contain an integer entry for every model atomic number.");
        const int mapped = data.at("atomic_number_to_model_type").at(key).get<int>();
        if (mapped < 0 || mapped >= static_cast<int>(result.model_atomic_numbers.size())
                || !mapping_values.insert(mapped).second)
            fail("$.atomic_number_to_model_type", "values must be a permutation of model type indices.");
        result.atomic_number_to_model_type.push_back(mapped);
    }
    result.r_cut = data.at("r_cut").get<double>();
    result.edge_embedding = data.at("edge_embedding");
    result.node_attributes = data.at("node_attributes");
    for (std::size_t index = 0; index < data.at("layers").size(); ++index) {
        const auto& layer = data.at("layers").at(index);
        const auto layer_path = "$.layers[" + std::to_string(index) + "]";
        result.layers.emplace_back(layer);
        // Both the serial and the Kokkos scalar gate consume the gate region
        // sequentially: one gate channel per gated irrep copy, in definition
        // order, and the scalars are silu-activated as l=0 values.  A
        // non-scalar gate block, or a gate channel count that differs from the
        // number of gated copies, would make that flat indexing mean something
        // other than the per-multiplet gate the contract describes ("0x0e"
        // means the layer has no gate at all), so it is rejected here.
        const auto& gate = object_member(layer, "gate", layer_path);
        if (gate.value("type", std::string{}) != "scalar_gate")
            fail(layer_path + ".gate", "must be a scalar_gate.");
        // "0x0e" (or an absent key) is how the exporter spells "no such irreps".
        const auto has_irreps = [](const std::string& text) {
            return !text.empty() && text != "0x0e";
        };
        const std::string scalar_text = gate.value("scalar_irreps", std::string{});
        const std::string gate_text = gate.value("gate_irreps", std::string{});
        const std::string gated_text = gate.value("gated_irreps", std::string{});
        if (has_irreps(scalar_text)) {
            const auto scalar = checked_irreps(
                gate.at("scalar_irreps"), layer_path + ".gate.scalar_irreps");
            for (const auto& block : scalar.blocks)
                if (block.l != 0)
                    fail(layer_path + ".gate.scalar_irreps",
                        "must contain only l=0 scalar blocks.");
        }
        if (has_irreps(gate_text)) {
            const auto gates = checked_irreps(
                gate.at("gate_irreps"), layer_path + ".gate.gate_irreps");
            for (const auto& block : gates.blocks)
                if (block.l != 0 || block.parity != 1)
                    fail(layer_path + ".gate.gate_irreps",
                        "must contain only scalar-even (0e) blocks.");
            int gated_copies = 0;
            if (has_irreps(gated_text)) {
                const auto gated = checked_irreps(
                    gate.at("gated_irreps"), layer_path + ".gate.gated_irreps");
                for (const auto& block : gated.blocks)
                    gated_copies += block.multiplicity;
            }
            if (gates.dimension() != gated_copies)
                fail(layer_path + ".gate",
                    "must define exactly one gate channel per gated irrep copy.");
        }
        // A self connection given as a tensor product is applied to node-major
        // operands that both evaluators lay out from the layer node plans, and
        // its result is added to the pre-gate tensor.  A decomposition that
        // does not preserve that node-major element order would silently
        // re-index the row; the lane-major kernels additionally require an
        // identical block plan and fall back to the definition-order path when
        // the descriptions merely aggregate adjacent same-angular blocks
        // differently.
        const auto& self_connection = layer.at("self_connection");
        if (self_connection.is_object() && self_connection.contains("irreps_in1")) {
            const auto self_in = checked_irreps(
                object_member(self_connection, "irreps_in1", layer_path + ".self_connection"),
                layer_path + ".self_connection.irreps_in1");
            const auto self_out = checked_irreps(
                object_member(self_connection, "irreps_out", layer_path + ".self_connection"),
                layer_path + ".self_connection.irreps_out");
            const auto node_in = checked_irreps(
                layer.at("node_irreps_in"), layer_path + ".node_irreps_in");
            const auto node_pre_gate = checked_irreps(
                layer.at("node_irreps_pre_gate"), layer_path + ".node_irreps_pre_gate");
            if (!same_node_major_order(self_in, node_in)
                || !same_node_major_order(self_out, node_pre_gate))
                fail(layer_path + ".self_connection",
                    "irrep decomposition does not keep the layer node-major element order "
                    "(irreps_in1=" + self_connection.at("irreps_in1").get<std::string>()
                    + " versus node_irreps_in="
                    + layer.at("node_irreps_in").get<std::string>() + ", irreps_out="
                    + self_connection.at("irreps_out").get<std::string>()
                    + " versus node_irreps_pre_gate="
                    + layer.at("node_irreps_pre_gate").get<std::string>() + ").");
        }
        // Legacy OAM v2 files remain loadable for the materialized path.  A
        // Kokkos owner must require this certificate before construction.
        if (layer.contains("kokkos_radial")) {
            const int function_count = layer.at("edge_tensor_product").value("weight_numel", 0);
            result.kokkos_radial.push_back(checked_kokkos_radial(
                layer.at("kokkos_radial"),
                "$.layers[" + std::to_string(index) + "].kokkos_radial",
                function_count, result.r_cut));
        } else {
            result.kokkos_radial.emplace_back();
        }
    }
    result.readout = data.at("readout");
    const auto expected_types = result.model_atomic_numbers.size();
    for (const auto* key : {"per_type_shift", "per_type_scale"}) {
        const auto& tensor = data.at(key);
        if (!tensor.is_object() || !tensor.contains("shape") || !tensor.contains("values")
                || !tensor.at("shape").is_array() || tensor.at("shape").size() != 1
                || tensor.at("shape").at(0).get<std::size_t>() != expected_types
                || !tensor.at("values").is_array() || tensor.at("values").size() != expected_types)
            fail(std::string("$.") + key, "must be a rank-one tensor with one finite value per model type.");
        for (const auto& value : tensor.at("values"))
            if (!value.is_number() || !std::isfinite(value.get<double>()))
                fail(std::string("$.") + key, "values must be finite numbers.");
    }
    result.per_type_shift = data.at("per_type_shift").at("values").get<std::vector<double>>();
    result.per_type_scale = data.at("per_type_scale").at("values").get<std::vector<double>>();
    result.zbl = data.at("zbl");
    result.exporter_fingerprint = fingerprint;
    result.contract_fingerprint_value = contract_fingerprint(data);
    result.generated_architecture_fingerprint_value =
        contract_fingerprint(generated_architecture(data));
    return result;
}

NequIPOAMModel::NequIPOAMModel(const std::string& filename)
    : schema_(load_nequip_oam_schema(filename))
{
    std::vector<int> type_to_model(schema_.atomic_numbers.size(), -1);
    for (std::size_t type = 0; type < schema_.atomic_numbers.size(); ++type) {
        const auto it = std::find(
            schema_.model_atomic_numbers.begin(), schema_.model_atomic_numbers.end(),
            schema_.atomic_numbers[type]);
        if (it == schema_.model_atomic_numbers.end())
            throw std::invalid_argument("NequIP OAM atomic-number mapping is incomplete.");
        type_to_model[type] = schema_.atomic_number_to_model_type[
            static_cast<std::size_t>(it - schema_.model_atomic_numbers.begin())];
    }
    const auto embedding_shape =
        schema_.node_attributes.at("shape").get<std::vector<int>>();
    if (embedding_shape.size() != 2
        || embedding_shape[0] != static_cast<int>(schema_.model_atomic_numbers.size())
        || embedding_shape[1] <= 0)
        throw std::invalid_argument("NequIP OAM node_attributes shape is inconsistent.");
    const int node_attribute_width = embedding_shape[1];
    const auto node_attributes =
        schema_.node_attributes.at("values").get<std::vector<double>>();
    if (static_cast<int>(node_attributes.size())
        != embedding_shape[0] * embedding_shape[1])
        throw std::invalid_argument("NequIP OAM node_attributes values are inconsistent.");
    if (schema_.layers.empty()
        || Irreps(schema_.layers.front().node_irreps_in).dimension()
            != node_attribute_width)
        throw std::invalid_argument(
            "NequIP OAM node embedding does not match the first layer input.");
    for (std::size_t index = 1; index < schema_.layers.size(); ++index)
        if (Irreps(schema_.layers[index - 1].node_irreps_out).dimension()
            != Irreps(schema_.layers[index].node_irreps_in).dimension())
            throw std::invalid_argument(
                "NequIP OAM layer dimensions do not form a chain.");
    const int edge_dimension =
        Irreps(schema_.layers.front().edge_irreps).dimension();
    for (const auto& layer : schema_.layers)
        if (Irreps(layer.edge_irreps).dimension() != edge_dimension)
            throw std::invalid_argument(
                "NequIP OAM layers must use a consistent edge irreps dimension.");
    if (Irreps(schema_.readout.at("irreps_in").get<std::string>()).dimension()
        != Irreps(schema_.layers.back().node_irreps_out).dimension())
        throw std::invalid_argument(
            "NequIP OAM readout input does not match the final layer.");
    const auto& edge = schema_.edge_embedding;
    const auto edge_frequencies = edge.at("bessel_weights").at("values")
        .get<std::vector<double>>();
    const double edge_cutoff_p = edge.at("cutoff").at("p").get<double>();
    const double edge_factor = edge.at("factor").get<double>();
    if (edge_frequencies.empty() || !(edge_cutoff_p > 0.0)
        || !std::isfinite(edge_factor) || !std::isfinite(edge_cutoff_p))
        throw std::invalid_argument("NequIP OAM edge embedding metadata is invalid.");
    const auto& zbl = schema_.zbl;
    const auto zbl_exponents = zbl.at("screening_exponents").get<std::vector<double>>();
    const auto zbl_coefficients =
        zbl.at("screening_coefficients").get<std::vector<double>>();
    const double zbl_cutoff = zbl.at("cutoff").get<double>();
    const double zbl_cutoff_p = zbl.value("cutoff_p", edge_cutoff_p);
    const double zbl_half_prefactor = zbl.at("half_energy_prefactor").get<double>();
    if (zbl_exponents.size() != zbl_coefficients.size() || zbl_exponents.empty()
        || !(zbl_cutoff > 0.0) || !(zbl_cutoff_p > 0.0) || !(zbl_half_prefactor > 0.0))
        throw std::invalid_argument("NequIP OAM ZBL metadata is inconsistent.");
}
