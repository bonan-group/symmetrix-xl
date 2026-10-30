#include "oam_radial.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <locale>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "compact_radial.hpp"
#include "sha256.hpp"


namespace {

nlohmann::json radial_mlp_definition(const nlohmann::json& data)
{
    if (!data.contains("mlp") || !data.at("mlp").is_object()
        || !data.at("mlp").contains("layers"))
        throw std::invalid_argument("OAM radial_basis.mlp must contain layers.");
    const auto& layers = data.at("mlp").at("layers");
    if (!layers.is_array() || layers.empty())
        throw std::invalid_argument("OAM radial_basis.mlp.layers must not be empty.");
    nlohmann::json definition{{"layers", nlohmann::json::array()}};
    for (std::size_t index = 0; index < layers.size(); ++index) {
        const auto& layer = layers.at(index);
        if (layer.value("type", std::string{}) == "silu") {
            definition["layers"].push_back({{"type", "silu"}});
            continue;
        }
        if (!layer.is_object() || !layer.contains("weight") || !layer.contains("bias"))
            throw std::invalid_argument(
                "OAM radial_basis.mlp.layers[" + std::to_string(index)
                + "] must contain weight and bias.");
        definition["layers"].push_back({
            {"type", "linear"}, {"weight", layer.at("weight")},
            {"bias", layer.at("bias")} });
        if (index + 1 != layers.size() && !layers.at(index + 1).contains("type"))
            definition["layers"].push_back({{"type", "silu"}});
    }
    return definition;
}

std::vector<double> radial_bessel_weights(const nlohmann::json& data)
{
    if (!data.contains("bessel_weights")) return {};
    const auto& weights = data.at("bessel_weights");
    if (weights.is_array()) return weights.get<std::vector<double>>();
    if (weights.is_object() && weights.contains("values"))
        return weights.at("values").get<std::vector<double>>();
    throw std::invalid_argument("OAM radial bessel_weights must be an array or tensor.");
}

}  // namespace

OAMRadialNetwork::OAMRadialNetwork(const nlohmann::json& data)
    : type_(data.at("type").get<std::string>()),
      cutoff_p_(data.at("cutoff").value("p", 0.0)),
      num_basis_(data.at("num_basis").get<int>()),
      bessel_weights_(radial_bessel_weights(data)),
      input_factor_(data.value("input_factor", 1.0)),
      mlp_(radial_mlp_definition(data))
{
    if (type_.empty() || num_basis_ <= 0 || !std::isfinite(cutoff_p_)
        || (data.at("cutoff").contains("p") && cutoff_p_ <= 0.0)
        || !std::isfinite(input_factor_))
        throw std::invalid_argument("OAM radial basis metadata is invalid.");
    if (!bessel_weights_.empty()) {
        if (static_cast<int>(bessel_weights_.size()) != num_basis_
            || !std::all_of(bessel_weights_.begin(), bessel_weights_.end(),
                [](double value) { return std::isfinite(value) && value > 0.0; }))
            throw std::invalid_argument("OAM radial Bessel weights are invalid.");
    }
    if (mlp_.input_size() != num_basis_)
        throw std::invalid_argument("OAM radial basis MLP input size does not match num_basis.");
}
namespace {

std::string model_fingerprint(const std::string& exporter_fingerprint)
{
    sha256::Digest hash;
    hash.update(exporter_fingerprint);
    return hash.finish();
}

std::string table_hash(
    double grid_min, double cutoff, int num_points, int function_count,
    const std::vector<std::vector<double>>& nodal_values)
{
    // Derivative solves can differ by language at roundoff; nodal values are portable.
    // Match the exporter with a snapped fixed-decimal projection.  The
    // independently evaluated radial MLP differs between Python and C++ by at
    // most a few 1e-14, so the projection step is kept six orders of magnitude
    // coarser than that noise floor; otherwise a value sitting on a rounding
    // boundary would hash differently for two identical tables.
    std::ostringstream canonical;
    canonical.imbue(std::locale::classic());
    canonical << std::fixed << std::setprecision(6);
    const auto write_value = [&canonical](const double value) {
        // Round half to even, as the exporter's round() does, and emit every
        // sub-step magnitude through an explicit zero literal.  The sign of a
        // value below half a step is noise that the two language runtimes
        // disagree about, and a magnitude test is what survives aggressive
        // floating-point optimization of a signed-zero normalization.
        const double scaled = std::nearbyint(value * 1.0e6);
        if (std::abs(scaled) < 0.5)
            canonical << "0.000000";
        else
            canonical << scaled / 1.0e6;
        canonical << ';';
    };
    write_value(grid_min);
    write_value(cutoff);
    canonical << num_points << ';' << function_count << ';';
    for (const auto& values : nodal_values)
        for (const double value : values)
            write_value(value);
    sha256::Digest hash;
    hash.update(canonical.str());
    return hash.finish();
}

double smooth_cutoff(double x, double p)
{
    if (x >= 1.0) return 0.0;
    if (x <= 0.0) return 1.0;
    const double xp = std::pow(x, p);
    return 1.0 - ((p+1.0)*(p+2.0)/2.0)*xp
        + p*(p+2.0)*std::pow(x, p+1.0)
        - (p*(p+1.0)/2.0)*std::pow(x, p+2.0);
}

std::vector<double> radial_features(
    const OAMRadialNetwork& radial, double radius, double cutoff)
{
    const double x = radius/cutoff;
    std::vector<double> basis(static_cast<std::size_t>(radial.num_basis()), 0.0);
    if (x >= 1.0) return basis;
    const auto& frequencies = radial.bessel_weights();
    const bool oam_basis = !frequencies.empty();
    const double scale = std::sqrt(2.0/cutoff);
    for (int index = 0; index < radial.num_basis(); ++index) {
        const double frequency = oam_basis
            ? frequencies[static_cast<std::size_t>(index)]
            : static_cast<double>(index+1);
        if (oam_basis) {
            const double argument = x*frequency;
            basis[static_cast<std::size_t>(index)] = std::abs(argument) < 1.0e-14
                ? frequency : std::sin(std::numbers::pi*argument)*frequency
                    /(std::numbers::pi*argument);
        } else if (radius > 0.0) {
            basis[static_cast<std::size_t>(index)] = scale
                * std::sin(std::numbers::pi*frequency*x)/radius;
        } else {
            basis[static_cast<std::size_t>(index)] = scale
                * std::numbers::pi*frequency/cutoff;
        }
        basis[static_cast<std::size_t>(index)] *= smooth_cutoff(x, radial.cutoff_p())
            * radial.input_factor();
    }
    return basis;
}

OAMRadialSplineData build_from_network(
    const OAMRadialNetwork& radial, int num_points, double grid_min, double cutoff)
{
    if (num_points < 4) throw std::invalid_argument("OAM radial spline requires at least four points.");
    if (!std::isfinite(grid_min) || grid_min < 0.0)
        throw std::invalid_argument("OAM radial spline grid minimum must be finite and non-negative.");
    if (!std::isfinite(cutoff) || cutoff <= 0.0 || cutoff <= grid_min)
        throw std::invalid_argument("OAM radial spline cutoff must be positive and exceed grid minimum.");
    const double h = (cutoff-grid_min)/static_cast<double>(num_points-1);
    if (!std::isfinite(h) || !(h > 0.0))
        throw std::invalid_argument("OAM radial spline spacing must be positive and finite.");
    OAMRadialSplineData result;
    result.grid_min = grid_min;
    result.cutoff = cutoff;
    result.spacing = h;
    result.num_points = num_points;
    result.function_count = radial.output_size();
    if (result.function_count <= 0)
        throw std::invalid_argument("OAM radial spline network has no output functions.");
    result.nodal_values.assign(result.function_count,
        std::vector<double>(static_cast<std::size_t>(num_points), 0.0));
    result.nodal_derivatives.assign(result.function_count,
        std::vector<double>(static_cast<std::size_t>(num_points), 0.0));
    for (int node = 0; node < num_points; ++node) {
        const double radius = grid_min+h*node;
        const auto output = radial.mlp().evaluate(radial_features(radial, radius, cutoff));
        if (static_cast<int>(output.size()) != result.function_count)
            throw std::invalid_argument("OAM radial spline network output size is inconsistent.");
        for (int function = 0; function < result.function_count; ++function)
            result.nodal_values[function][static_cast<std::size_t>(node)] = output[function];
    }
    for (int function = 0; function < result.function_count; ++function) {
        auto& values = result.nodal_values[function];
        auto& derivatives = result.nodal_derivatives[function];
        derivatives = CompactRadialModel::spline_derivatives(values, h);
        for (const double value : values)
            if (!std::isfinite(value)) throw std::runtime_error("OAM radial spline contains non-finite nodal values.");
        for (const double value : derivatives)
            if (!std::isfinite(value)) throw std::runtime_error("OAM radial spline contains non-finite nodal derivatives.");
    }
    result.coefficients.resize(static_cast<std::size_t>(num_points-1)*4*result.function_count);
    for (int interval = 0; interval < num_points-1; ++interval) {
        for (int function = 0; function < result.function_count; ++function) {
            const double y0 = result.nodal_values[function][static_cast<std::size_t>(interval)];
            const double y1 = result.nodal_values[function][static_cast<std::size_t>(interval+1)];
            const double d0 = result.nodal_derivatives[function][static_cast<std::size_t>(interval)];
            const double d1 = result.nodal_derivatives[function][static_cast<std::size_t>(interval+1)];
            const std::size_t offset = (static_cast<std::size_t>(interval)*4)*result.function_count
                + static_cast<std::size_t>(function);
            result.coefficients[offset] = y0;
            result.coefficients[offset+static_cast<std::size_t>(result.function_count)] = d0;
            result.coefficients[offset+2*static_cast<std::size_t>(result.function_count)] =
                (3.0*(y1-y0)/h-2.0*d0-d1)/h;
            result.coefficients[offset+3*static_cast<std::size_t>(result.function_count)] =
                (2.0*(y0-y1)/h+d0+d1)/(h*h);
        }
    }
    result.table_fingerprint = table_hash(
        grid_min, cutoff, num_points, result.function_count, result.nodal_values);
    return result;
}

template <typename Precision>
RadialFunctionSetKokkos<Precision> load_radial(const OAMRadialSplineData& data)
{
    std::vector<std::vector<std::vector<double>>> values(1, data.nodal_values);
    std::vector<std::vector<std::vector<double>>> derivatives(1, data.nodal_derivatives);
    return RadialFunctionSetKokkos<Precision>(
        data.spacing, std::move(values), std::move(derivatives), data.grid_min, data.cutoff);
}

template <typename Precision>
void validate_contract(
    const OAMRadialSplineContract& contract,
    const OAMRadialSplineData& generated,
    const std::string& exporter_fingerprint,
    int requested_points,
    double requested_grid_min,
    double cutoff)
{
    if (!contract.certified)
        throw std::invalid_argument("OAM radial Kokkos construction requires certified compact metadata.");
    if (contract.num_points != requested_points
        || std::abs(contract.grid_min-requested_grid_min) > 1.0e-12
        || std::abs(contract.cutoff-cutoff) > 1.0e-12
        || contract.function_count != generated.function_count
        || contract.edge_type_count != 1
        || std::abs(contract.spacing-generated.spacing) > 1.0e-12)
        throw std::invalid_argument("OAM radial compact metadata does not match the requested spline grid.");
    // The spline is always built and certified in double precision and cast
    // on load, so a float evaluator may consume a certified float64 export;
    // that keeps one export per model while FP32 evaluation stays opt-in.
    // A double evaluator must not silently promote a float32 export.
    const std::string promoted_precision = std::is_same_v<Precision, float>
        ? "float32" : "";
    if (contract.precision != "float64"
        && contract.precision != promoted_precision)
        throw std::invalid_argument(
            std::string("OAM radial compact metadata precision ")
            + contract.precision
            + " is not loadable by this evaluator (expected float64"
            + (promoted_precision.empty() ? "" : " or float32") + ").");
    if (contract.model_fingerprint != model_fingerprint(exporter_fingerprint))
        throw std::invalid_argument("OAM radial model fingerprint does not match the schema.");
    if (contract.table_fingerprint != generated.table_fingerprint)
        throw std::invalid_argument(
            "OAM radial table fingerprint does not match generated nodal values (contract="
            + contract.table_fingerprint + ", generated=" + generated.table_fingerprint
            + ", first=" + std::to_string(generated.nodal_values.front().front()) + ").");
}

} // namespace

template <typename Precision>
NequIPOAMRadialSet<Precision>::NequIPOAMRadialSet(
    const NequIPOAMModelSchema& schema, int layer_index, int num_points, double grid_min)
{
    if (layer_index < 0 || static_cast<std::size_t>(layer_index) >= schema.layers.size())
        throw std::out_of_range("OAM radial layer index is out of range.");
    if (schema.kokkos_radial.size() != schema.layers.size())
        throw std::invalid_argument("OAM radial schema is missing certified compact metadata.");
    const auto data = build_oam_radial_spline(
        OAMRadialNetwork(schema.layers[static_cast<std::size_t>(layer_index)].radial_basis),
        num_points, grid_min, schema.r_cut);
    validate_contract<Precision>(
        schema.kokkos_radial[static_cast<std::size_t>(layer_index)], data,
        schema.exporter_fingerprint, num_points, grid_min, schema.r_cut);
    function_count_ = data.function_count;
    radial_ = load_radial<Precision>(data);
}

template <typename Precision>
void NequIPOAMRadialSet<Precision>::evaluate(
    double radius, int function, Precision& value, Precision& derivative) const
{
    if (!std::isfinite(radius) || function < 0 || function >= function_count_)
        throw std::out_of_range("OAM radial spline evaluation index is out of range.");
    radial_.evaluate_function(0, radius, function, value, derivative);
}

OAMRadialSplineData build_oam_radial_spline(
    const OAMRadialNetwork& radial, int num_points, double grid_min, double cutoff)
{
    return build_from_network(radial, num_points, grid_min, cutoff);
}

template class NequIPOAMRadialSet<float>;
template class NequIPOAMRadialSet<double>;
