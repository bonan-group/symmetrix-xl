#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

#include "compact_radial.hpp"

namespace {

CompactRadialModel::Network parse_network(
    const nlohmann::json& definition,
    const std::string& name)
{
    CompactRadialModel::Network network;
    network.shape = definition.at("shape").get<std::vector<int>>();
    network.weights = definition.at("weights").get<std::vector<std::vector<double>>>();
    network.activation_scale = definition.at("activation_scale").get<double>();
    const auto postprocess = definition.value("postprocess", std::string());
    if (!postprocess.empty() && postprocess != "tanh-square")
        throw std::invalid_argument(
            "Compact radial network " + name + " has unsupported postprocessing.");
    network.tanh_square = postprocess == "tanh-square";

    if (definition.at("activation").get<std::string>() != "silu")
        throw std::invalid_argument("Compact radial network " + name + " must use SiLU.");
    if (network.shape.size() < 2 || network.weights.size() != network.shape.size()-1)
        throw std::invalid_argument("Compact radial network " + name + " has an invalid shape.");
    if (!std::isfinite(network.activation_scale) || network.activation_scale <= 0.0)
        throw std::invalid_argument(
            "Compact radial network " + name + " has an invalid activation scale.");
    for (int layer=0; layer<network.weights.size(); ++layer) {
        if (network.shape[layer] <= 0 || network.shape[layer+1] <= 0)
            throw std::invalid_argument(
                "Compact radial network " + name + " has a non-positive layer size.");
        const auto expected = static_cast<std::size_t>(network.shape[layer]*network.shape[layer+1]);
        if (network.weights[layer].size() != expected)
            throw std::invalid_argument("Compact radial network " + name + " has invalid weights.");
        if (!std::all_of(
                network.weights[layer].begin(),
                network.weights[layer].end(),
                [] (double value) { return std::isfinite(value); }))
            throw std::invalid_argument(
                "Compact radial network " + name + " has non-finite weights.");
    }

    network.mlp = std::make_unique<MultilayerPerceptron>(
        network.shape,
        network.weights,
        network.activation_scale);
    return network;
}

}  // namespace

RadialSplineData FactorizedRadialData::reconstruct_projected() const
{
    if (projection_layout
        != CompactRadialProjectionLayout::row_major_output_by_embedding)
        throw std::invalid_argument("Compact radial projection has an unsupported layout.");
    if (embedding_width <= 0 || output_width <= 0)
        throw std::invalid_argument("Compact radial factorization has invalid extents.");
    if (penultimate.values.size() != static_cast<std::size_t>(embedding_width)
        || penultimate.derivatives.size() != static_cast<std::size_t>(embedding_width)
        || final_projection.size()
            != static_cast<std::size_t>(embedding_width)*output_width)
        throw std::invalid_argument("Compact radial factorization has inconsistent extents.");

    std::size_t node_count = 0;
    if (!penultimate.values.empty())
        node_count = penultimate.values.front().size();
    for (int embedding=0; embedding<embedding_width; ++embedding) {
        if (penultimate.values[embedding].size() != node_count
            || penultimate.derivatives[embedding].size() != node_count)
            throw std::invalid_argument(
                "Compact radial penultimate splines have inconsistent node counts.");
        if (!std::all_of(
                penultimate.values[embedding].begin(),
                penultimate.values[embedding].end(),
                [] (double value) { return std::isfinite(value); })
            || !std::all_of(
                penultimate.derivatives[embedding].begin(),
                penultimate.derivatives[embedding].end(),
                [] (double value) { return std::isfinite(value); }))
            throw std::invalid_argument(
                "Compact radial penultimate splines contain a non-finite value.");
    }

    RadialSplineData result;
    result.values.assign(output_width, std::vector<double>(node_count, 0.0));
    result.derivatives.assign(output_width, std::vector<double>(node_count, 0.0));
    for (int output=0; output<output_width; ++output) {
        for (int embedding=0; embedding<embedding_width; ++embedding) {
            const double weight = final_projection[output*embedding_width+embedding];
            if (!std::isfinite(weight))
                throw std::invalid_argument(
                    "Compact radial final projection contains a non-finite value.");
            for (std::size_t node=0; node<node_count; ++node) {
                result.values[output][node] +=
                    weight*penultimate.values[embedding][node];
                result.derivatives[output][node] +=
                    weight*penultimate.derivatives[embedding][node];
            }
        }
    }
    return result;
}

CompactRadialModel::CompactRadialModel(
    std::string definition_json,
    std::vector<int> atomic_numbers,
    double r_cut)
    : atomic_numbers(std::move(atomic_numbers))
{
    const auto definition = nlohmann::json::parse(definition_json);
    grid_min = definition.at("spline_grid_min").get<double>();
    spline_points = definition.at("num_spline_points").get<int>();
    if (!std::isfinite(grid_min) || grid_min <= 0.0)
        throw std::invalid_argument("Compact radial model has an invalid spline grid minimum.");
    if (spline_points < 4)
        throw std::invalid_argument("Compact radial model requires at least four spline points.");
    h = (r_cut-grid_min)/(spline_points-1);
    if (!(h > 0.0))
        throw std::invalid_argument("Compact radial model has an invalid spline grid.");

    const auto& basis = definition.at("basis");
    if (basis.at("type").get<std::string>() != "bessel")
        throw std::invalid_argument("Compact radial model only supports a Bessel basis.");
    bessel_weights = basis.at("weights").get<std::vector<double>>();
    bessel_prefactor = basis.at("prefactor").get<double>();
    if (bessel_weights.empty()
        || !std::isfinite(bessel_prefactor)
        || !std::all_of(
            bessel_weights.begin(), bessel_weights.end(),
            [] (double value) { return std::isfinite(value); }))
        throw std::invalid_argument("Compact radial model has invalid Bessel data.");

    const auto& cutoff = definition.at("cutoff");
    if (cutoff.at("type").get<std::string>() != "polynomial")
        throw std::invalid_argument("Compact radial model only supports a polynomial cutoff.");
    cutoff_r_max = cutoff.at("r_max").get<double>();
    cutoff_p = cutoff.at("p").get<int>();
    if (!std::isfinite(cutoff_r_max) || cutoff_p <= 0)
        throw std::invalid_argument("Compact radial model has invalid cutoff data.");
    if (std::abs(cutoff_r_max-r_cut) > 1e-12)
        throw std::invalid_argument("Compact radial cutoff does not match model r_cut.");

    const auto& transform = definition.at("distance_transform");
    const auto transform_type = transform.at("type").get<std::string>();
    use_agnesi = transform_type == "agnesi";
    if (use_agnesi) {
        agnesi_a = transform.at("a").get<double>();
        agnesi_q = transform.at("q").get<double>();
        agnesi_p = transform.at("p").get<double>();
        covalent_radii = transform.at("covalent_radii").get<std::vector<double>>();
        if (covalent_radii.size() != this->atomic_numbers.size())
            throw std::invalid_argument("Compact radial covalent radii do not match atomic numbers.");
        if (!std::isfinite(agnesi_a)
            || !std::isfinite(agnesi_q)
            || !std::isfinite(agnesi_p)
            || !std::all_of(
                covalent_radii.begin(), covalent_radii.end(),
                [] (double value) { return std::isfinite(value) && value > 0.0; }))
            throw std::invalid_argument("Compact radial model has invalid Agnesi data.");
    } else if (transform_type != "none") {
        throw std::invalid_argument("Compact radial model has an unsupported distance transform.");
    }

    const auto& networks = definition.at("networks");
    R0_network = parse_network(networks.at("R0"), "R0");
    if (networks.contains("R1"))
        R1_network = std::make_unique<Network>(
            parse_network(networks.at("R1"), "R1"));
    if (R0_network.shape.front() != bessel_weights.size()
        || (R1_network
            && R1_network->shape.front() != bessel_weights.size()))
        throw std::invalid_argument("Compact radial network input does not match the Bessel basis.");
    if (R0_network.tanh_square || (R1_network && R1_network->tanh_square))
        throw std::invalid_argument("Compact radial R0/R1 networks cannot use postprocessing.");
    if (networks.contains("A0")) {
        A0_network = std::make_unique<Network>(parse_network(networks.at("A0"), "A0"));
        if (A0_network->shape.front() != bessel_weights.size()
            || A0_network->shape.back() != 1
            || !A0_network->tanh_square)
            throw std::invalid_argument("Compact radial A0 network has an invalid shape or postprocessing.");
    }
    if (networks.contains("A1")) {
        A1_network = std::make_unique<Network>(parse_network(networks.at("A1"), "A1"));
        if (A1_network->shape.front() != bessel_weights.size()
            || A1_network->shape.back() != 1
            || !A1_network->tanh_square)
            throw std::invalid_argument("Compact radial A1 network has an invalid shape or postprocessing.");
    }
}

double CompactRadialModel::spline_h() const
{
    return h;
}

double CompactRadialModel::spline_min() const
{
    return grid_min;
}

int CompactRadialModel::num_spline_points() const
{
    return spline_points;
}

bool CompactRadialModel::has_A0() const
{
    return static_cast<bool>(A0_network);
}

bool CompactRadialModel::has_A1() const
{
    return static_cast<bool>(A1_network);
}

bool CompactRadialModel::has_R1() const
{
    return static_cast<bool>(R1_network);
}

std::vector<double> CompactRadialModel::radial_features(int type_i, int type_j) const
{
    if (type_i < 0 || type_i >= atomic_numbers.size()
        || type_j < 0 || type_j >= atomic_numbers.size())
        throw std::out_of_range("Compact radial model type index is out of range.");

    double r0 = 1.0;
    if (use_agnesi)
        r0 = 0.5*(covalent_radii[type_i]+covalent_radii[type_j]);

    auto features = std::vector<double>(spline_points*bessel_weights.size());
    for (int node=0; node<spline_points; ++node) {
        const double r = grid_min+node*h;
        double transformed_r = r;
        if (use_agnesi) {
            const double scaled_r = r/r0;
            transformed_r = 1.0/(
                1.0
                + agnesi_a*std::pow(scaled_r, agnesi_q)
                    /(1.0+std::pow(scaled_r, agnesi_q-agnesi_p)));
        }

        double cutoff_value = 0.0;
        if (r < cutoff_r_max) {
            const double scaled_r = r/cutoff_r_max;
            cutoff_value = 1.0
                - ((cutoff_p+1.0)*(cutoff_p+2.0)/2.0)*std::pow(scaled_r, cutoff_p)
                + cutoff_p*(cutoff_p+2.0)*std::pow(scaled_r, cutoff_p+1)
                - (cutoff_p*(cutoff_p+1.0)/2.0)*std::pow(scaled_r, cutoff_p+2);
        }

        for (int basis=0; basis<bessel_weights.size(); ++basis) {
            features[node*bessel_weights.size()+basis] =
                bessel_prefactor
                * std::sin(bessel_weights[basis]*transformed_r)
                / transformed_r
                * cutoff_value;
        }
    }
    return features;
}

RadialSplineData CompactRadialModel::evaluate_network(
    Network& network,
    const std::vector<double>& features)
{
    auto batch_values = network.mlp->evaluate_batch(features, spline_points);
    const int num_functions = network.shape.back();
    if (network.tanh_square) {
        for (auto& value : batch_values)
            value = std::tanh(value*value);
    }

    RadialSplineData result;
    result.values.resize(num_functions, std::vector<double>(spline_points));
    result.derivatives.resize(num_functions, std::vector<double>(spline_points));
    for (int function=0; function<num_functions; ++function) {
        for (int node=0; node<spline_points; ++node)
            result.values[function][node] = batch_values[node*num_functions+function];
        result.derivatives[function] = spline_derivatives(result.values[function]);
    }
    return result;
}

FactorizedRadialData CompactRadialModel::evaluate_factorized_network(
    Network& network,
    const std::vector<double>& features)
{
    if (network.shape.size() < 2)
        throw std::invalid_argument("Compact radial network cannot be factorized.");
    FactorizedRadialData result;
    result.embedding_width = network.shape.end()[-2];
    result.output_width = network.shape.back();
    result.final_projection = network.weights.back();
    result.projection_layout =
        CompactRadialProjectionLayout::row_major_output_by_embedding;

    const auto batch_values =
        network.mlp->evaluate_penultimate_batch(features, spline_points);
    const auto expected = static_cast<std::size_t>(spline_points)*result.embedding_width;
    if (batch_values.size() != expected)
        throw std::runtime_error(
            "Compact radial penultimate network output has an invalid size.");
    result.penultimate.values.assign(
        result.embedding_width, std::vector<double>(spline_points));
    result.penultimate.derivatives.assign(
        result.embedding_width, std::vector<double>(spline_points));
    for (int embedding=0; embedding<result.embedding_width; ++embedding) {
        for (int node=0; node<spline_points; ++node)
            result.penultimate.values[embedding][node] =
                batch_values[node*result.embedding_width+embedding];
        result.penultimate.derivatives[embedding] =
            spline_derivatives(result.penultimate.values[embedding]);
    }
    // Validate the complete contract before it can reach device planning.
    static_cast<void>(result.reconstruct_projected());
    return result;
}

std::vector<double> CompactRadialModel::spline_derivatives(
    const std::vector<double>& values) const
{
    if (values.size() != spline_points)
        throw std::invalid_argument("Compact radial spline values have an invalid size.");

    const int unknowns = spline_points-2;
    auto lower = std::vector<double>(unknowns, 0.0);
    auto diagonal = std::vector<double>(unknowns, 4.0);
    auto upper = std::vector<double>(unknowns, 0.0);
    auto rhs = std::vector<double>(unknowns, 0.0);

    const double left_not_a_knot =
        2.0*(-values[0]+2.0*values[1]-values[2])/h;
    upper[0] = 2.0;
    rhs[0] = 3.0*(values[2]-values[0])/h-left_not_a_knot;
    for (int node=2; node<spline_points-1; ++node) {
        const int row = node-1;
        lower[row] = 1.0;
        if (node+1 <= spline_points-2)
            upper[row] = 1.0;
        rhs[row] = 3.0*(values[node+1]-values[node-1])/h;
    }

    for (int row=1; row<unknowns; ++row) {
        const double factor = lower[row]/diagonal[row-1];
        diagonal[row] -= factor*upper[row-1];
        rhs[row] -= factor*rhs[row-1];
    }
    auto solution = std::vector<double>(unknowns, 0.0);
    solution.back() = rhs.back()/diagonal.back();
    for (int row=unknowns-2; row>=0; --row)
        solution[row] = (rhs[row]-upper[row]*solution[row+1])/diagonal[row];

    auto derivatives = std::vector<double>(spline_points, 0.0);
    for (int node=1; node<spline_points-1; ++node)
        derivatives[node] = solution[node-1];
    derivatives[0] = derivatives[2]+left_not_a_knot;
    derivatives.back() = 0.0;
    return derivatives;
}

CompactRadialPairTables CompactRadialModel::materialize_pair(int type_i, int type_j)
{
    return materialize_projected_pair(type_i, type_j);
}

CompactRadialPairTables CompactRadialModel::materialize_projected_pair(
    int type_i,
    int type_j)
{
    const auto features = radial_features(type_i, type_j);
    CompactRadialPairTables result;
    result.R0 = evaluate_network(R0_network, features);
    if (R1_network)
        result.R1 = evaluate_network(*R1_network, features);
    if (A0_network)
        result.A0 = evaluate_network(*A0_network, features);
    if (A1_network)
        result.A1 = evaluate_network(*A1_network, features);
    return result;
}

CompactRadialFactorizedPairTables CompactRadialModel::materialize_factorized_pair(
    int type_i,
    int type_j)
{
    const auto features = radial_features(type_i, type_j);
    CompactRadialFactorizedPairTables result;
    result.R0 = evaluate_factorized_network(R0_network, features);
    if (!R1_network)
        throw std::invalid_argument(
            "Compact radial factorization requires an R1 network.");
    result.R1 = evaluate_factorized_network(*R1_network, features);
    return result;
}

CompactRadialModel::Network& CompactRadialModel::radial_network(
    const std::string& name)
{
    if (name == "R0")
        return R0_network;
    if (name == "R1" && R1_network)
        return *R1_network;
    if (name == "A0" && A0_network)
        return *A0_network;
    if (name == "A1" && A1_network)
        return *A1_network;
    throw std::invalid_argument(
        "Compact radial network must be R0, R1, A0, or A1 and present in the model.");
}

std::vector<std::pair<int,int>> CompactRadialModel::network_weight_shapes(
    const std::string& network_name)
{
    const auto& network = radial_network(network_name);
    auto result = std::vector<std::pair<int,int>>();
    result.reserve(network.weights.size());
    for (std::size_t layer=0; layer<network.weights.size(); ++layer)
        result.emplace_back(network.shape[layer+1], network.shape[layer]);
    return result;
}

std::vector<double> CompactRadialModel::transpose_spline_coefficients(
    const std::vector<double>& coefficient_adjoints,
    int pair_count,
    int function_count)
{
    if (pair_count < 0 || function_count <= 0)
        throw std::invalid_argument("Compact radial spline adjoint extents are invalid.");
    const int interval_count = spline_points-1;
    const auto expected = static_cast<std::size_t>(pair_count)
        *interval_count*4*function_count;
    if (coefficient_adjoints.size() != expected)
        throw std::invalid_argument("Compact radial spline coefficient adjoints have an invalid size.");

    if (spline_derivative_operator.empty()) {
        spline_derivative_operator.resize(
            static_cast<std::size_t>(spline_points)*spline_points);
        for (int input_node=0; input_node<spline_points; ++input_node) {
            auto basis = std::vector<double>(spline_points, 0.0);
            basis[input_node] = 1.0;
            const auto derivatives = spline_derivatives(basis);
            for (int output_node=0; output_node<spline_points; ++output_node)
                spline_derivative_operator[
                    output_node*spline_points+input_node] =
                        derivatives[output_node];
        }
    }

    auto nodal_adjoints = std::vector<double>(
        static_cast<std::size_t>(pair_count)*spline_points*function_count,
        0.0);
    auto derivative_adjoints = std::vector<double>(spline_points, 0.0);
    for (int pair=0; pair<pair_count; ++pair) {
        for (int function=0; function<function_count; ++function) {
            std::fill(derivative_adjoints.begin(), derivative_adjoints.end(), 0.0);
            auto nodal = [&] (int node) -> double& {
                return nodal_adjoints[
                    (static_cast<std::size_t>(pair)*spline_points+node)
                        *function_count+function];
            };
            auto coefficient = [&] (int interval, int order) {
                return coefficient_adjoints[
                    ((static_cast<std::size_t>(pair)*interval_count+interval)*4
                        +order)*function_count+function];
            };
            for (int interval=0; interval<interval_count; ++interval) {
                const double c0 = coefficient(interval, 0);
                const double c1 = coefficient(interval, 1);
                const double c2 = coefficient(interval, 2);
                const double c3 = coefficient(interval, 3);
                nodal(interval) += c0-3.0*c2/(h*h)+2.0*c3/(h*h*h);
                nodal(interval+1) += 3.0*c2/(h*h)-2.0*c3/(h*h*h);
                derivative_adjoints[interval] += c1-2.0*c2/h+c3/(h*h);
                derivative_adjoints[interval+1] += -c2/h+c3/(h*h);
            }
            for (int input_node=0; input_node<spline_points; ++input_node)
                for (int output_node=0; output_node<spline_points; ++output_node)
                    nodal(input_node) += spline_derivative_operator[
                        output_node*spline_points+input_node]
                        *derivative_adjoints[output_node];
        }
    }
    return nodal_adjoints;
}

CompactRadialNetworkGradients
CompactRadialModel::backpropagate_network_nodal_values(
    const std::string& network_name,
    const std::vector<std::pair<int,int>>& pair_types,
    const std::vector<double>& nodal_value_adjoints,
    bool factorized,
    const std::vector<double>& final_projection_adjoints)
{
    auto& network = radial_network(network_name);
    const int output_width = factorized
        ? network.shape.end()[-2] : network.shape.back();
    const auto expected = pair_types.size()
        *static_cast<std::size_t>(spline_points)*output_width;
    if (nodal_value_adjoints.size() != expected)
        throw std::invalid_argument("Compact radial nodal adjoints have an invalid size.");
    if (factorized) {
        if (network.tanh_square)
            throw std::invalid_argument(
                "Factorized compact radial reverse does not support postprocessing.");
        if (final_projection_adjoints.size() != network.weights.back().size())
            throw std::invalid_argument(
                "Compact radial final projection adjoints have an invalid size.");
    } else if (!final_projection_adjoints.empty()) {
        throw std::invalid_argument(
            "Full compact radial reverse does not accept separate projection adjoints.");
    }

    CompactRadialNetworkGradients result;
    result.weights.resize(network.weights.size());
    for (std::size_t layer=0; layer<network.weights.size(); ++layer)
        result.weights[layer].assign(network.weights[layer].size(), 0.0);
    if (factorized)
        result.weights.back() = final_projection_adjoints;

    const int reverse_layer_count = factorized
        ? static_cast<int>(network.weights.size())-1
        : static_cast<int>(network.weights.size());
    for (std::size_t pair=0; pair<pair_types.size(); ++pair) {
        const auto features = radial_features(
            pair_types[pair].first, pair_types[pair].second);
        for (int node=0; node<spline_points; ++node) {
            auto values = std::vector<std::vector<double>>(network.shape.size());
            auto preactivations = std::vector<std::vector<double>>(
                network.shape.size());
            values[0].assign(
                features.begin()+static_cast<std::size_t>(node)*network.shape[0],
                features.begin()+static_cast<std::size_t>(node+1)*network.shape[0]);
            for (int layer=0; layer<reverse_layer_count; ++layer) {
                const int input_width = network.shape[layer];
                const int layer_output_width = network.shape[layer+1];
                values[layer+1].assign(layer_output_width, 0.0);
                preactivations[layer+1].assign(layer_output_width, 0.0);
                for (int output=0; output<layer_output_width; ++output)
                    for (int input=0; input<input_width; ++input)
                        preactivations[layer+1][output] +=
                            network.weights[layer][output*input_width+input]
                            *values[layer][input];
                const bool activation = layer+1 < network.shape.size()-1;
                for (int output=0; output<layer_output_width; ++output) {
                    const double z = preactivations[layer+1][output];
                    values[layer+1][output] = activation
                        ? network.activation_scale*z/(1.0+std::exp(-z)) : z;
                }
            }

            auto adjoint = std::vector<double>(output_width);
            for (int output=0; output<output_width; ++output)
                adjoint[output] = nodal_value_adjoints[
                    (pair*spline_points+node)*output_width+output];
            if (!factorized && network.tanh_square) {
                for (int output=0; output<output_width; ++output) {
                    const double z = values.back()[output];
                    const double t = std::tanh(z*z);
                    adjoint[output] *= 2.0*z*(1.0-t*t);
                }
            }

            for (int layer=reverse_layer_count-1; layer>=0; --layer) {
                const int input_width = network.shape[layer];
                const int layer_output_width = network.shape[layer+1];
                if (layer+1 < network.shape.size()-1) {
                    for (int output=0; output<layer_output_width; ++output) {
                        const double z = preactivations[layer+1][output];
                        const double sigmoid = 1.0/(1.0+std::exp(-z));
                        adjoint[output] *= network.activation_scale*sigmoid
                            *(1.0+z*(1.0-sigmoid));
                    }
                }
                auto input_adjoint = std::vector<double>(input_width, 0.0);
                for (int output=0; output<layer_output_width; ++output) {
                    for (int input=0; input<input_width; ++input) {
                        const int index = output*input_width+input;
                        result.weights[layer][index] +=
                            adjoint[output]*values[layer][input];
                        input_adjoint[input] +=
                            adjoint[output]*network.weights[layer][index];
                    }
                }
                adjoint = std::move(input_adjoint);
            }
        }
    }
    return result;
}

CompactRadialFactorizationReport CompactRadialModel::factorization_report(
    const std::string& network_name,
    int type_i,
    int type_j)
{
    const auto features = radial_features(type_i, type_j);
    auto& network = radial_network(network_name);
    const auto projected = evaluate_network(network, features);
    const auto factorized = evaluate_factorized_network(network, features);
    const auto reconstructed = factorized.reconstruct_projected();
    if (projected.values.size() != reconstructed.values.size()
        || projected.derivatives.size() != reconstructed.derivatives.size())
        throw std::runtime_error(
            "Compact radial factorization reconstruction has an invalid size.");

    double value_error = 0.0;
    double derivative_error = 0.0;
    for (std::size_t output=0; output<projected.values.size(); ++output) {
        if (projected.values[output].size() != reconstructed.values[output].size()
            || projected.derivatives[output].size()
                != reconstructed.derivatives[output].size())
            throw std::runtime_error(
                "Compact radial factorization reconstruction has inconsistent nodes.");
        for (std::size_t node=0; node<projected.values[output].size(); ++node) {
            value_error = std::max(
                value_error,
                std::abs(projected.values[output][node]
                    - reconstructed.values[output][node]));
            derivative_error = std::max(
                derivative_error,
                std::abs(projected.derivatives[output][node]
                    - reconstructed.derivatives[output][node]));
        }
    }
    return {
        factorized.embedding_width,
        factorized.output_width,
        value_error,
        derivative_error,
    };
}
