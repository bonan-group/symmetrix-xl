#include "mace_nonlinear.hpp"
#include "prediction_heads.hpp"
#include "mace_nonlinear_schema.hpp"

#include <cmath>
#include <fstream>
#include <numbers>
#include <stdexcept>

#include "sphericart.hpp"

namespace {
std::vector<double> tensor_values(const nlohmann::json& value)
{
    return value.at("values").get<std::vector<double>>();
}

nlohmann::json load_model_json(const std::string& filename)
{
    std::ifstream stream(filename);
    if (!stream)
        throw std::runtime_error("Could not open MACE_Nonlinear model file: " + filename);
    return nlohmann::json::parse(stream);
}

double sigmoid(double value)
{
    if (value >= 0.0) return 1.0/(1.0+std::exp(-value));
    const double exponential = std::exp(value);
    return exponential/(1.0+exponential);
}

double silu(double value) { return value*sigmoid(value); }

const nlohmann::json& checked_interaction(const nlohmann::json& data)
{
    if (data.at("class").get<std::string>()
        != "RealAgnosticResidualNonLinearInteractionBlock")
        throw std::invalid_argument("MACE_Nonlinear interaction class is unsupported.");
    return data;
}

const nlohmann::json& readout_linear(const nlohmann::json& data)
{
    const auto readout_class = data.at("class").get<std::string>();
    if (readout_class == "LinearReadoutBlock")
        return data.at("linear");
    if (readout_class == "NonLinearReadoutBlock")
        return data.at("linear_1");
    throw std::invalid_argument(
        "MACE_Nonlinear readout class is unsupported: " + readout_class);
}

} // namespace

MaceNonlinear::Gate::Gate(const nlohmann::json& data)
    : input(data.at("irreps_in").get<std::string>()),
      output(data.at("irreps_out").get<std::string>()),
      scalars(data.at("irreps_scalars").get<std::string>()),
      gates(data.at("irreps_gates").get<std::string>()),
      gated(data.at("irreps_gated").get<std::string>()),
      scalar_constants(data.at("scalar_activation_constants").get<std::vector<double>>()),
      gate_constants(data.at("gate_activation_constants").get<std::vector<double>>())
{
    if (data.at("scalar_activation").get<std::string>() != "silu"
        || data.at("gate_activation").get<std::string>() != "sigmoid")
        throw std::invalid_argument("MACE_Nonlinear gate has unsupported activations.");
    if (input.dimension() != scalars.dimension()+gates.dimension()+gated.dimension()
        || output.dimension() != scalars.dimension()+gated.dimension()
        || scalar_constants.size() != scalars.blocks.size()
        || gate_constants.size() != gated.blocks.size())
        throw std::invalid_argument("MACE_Nonlinear gate irreps are inconsistent.");
}

std::vector<double> MaceNonlinear::Gate::evaluate(const std::vector<double>& values) const
{
    if (static_cast<int>(values.size()) != input.dimension())
        throw std::invalid_argument("MACE_Nonlinear gate input size is invalid.");
    std::vector<double> result(output.dimension());
    for (int block_index=0; block_index<static_cast<int>(scalars.blocks.size()); ++block_index) {
        const auto& block = scalars.blocks[block_index];
        for (int index=0; index<block.dimension(); ++index) {
            const int offset = block.offset+index;
            result[offset] = scalar_constants[block_index]*silu(values[offset]);
        }
    }
    int gate_offset = scalars.dimension();
    int gated_offset = scalars.dimension()+gates.dimension();
    int output_offset = scalars.dimension();
    for (int block_index=0; block_index<static_cast<int>(gated.blocks.size()); ++block_index) {
        const auto& block = gated.blocks[block_index];
        const int width = 2*block.l+1;
        for (int feature=0; feature<block.multiplicity; ++feature) {
            const double gate = gate_constants[block_index]*sigmoid(values[gate_offset++]);
            for (int component=0; component<width; ++component)
                result[output_offset++] = gate*values[gated_offset++];
        }
    }
    return result;
}

std::vector<double> MaceNonlinear::Gate::reverse(
    const std::vector<double>& values,
    const std::vector<double>& output_adjoint) const
{
    if (static_cast<int>(values.size()) != input.dimension()
        || static_cast<int>(output_adjoint.size()) != output.dimension())
        throw std::invalid_argument("MACE_Nonlinear gate adjoint size is invalid.");
    std::vector<double> result(input.dimension(), 0.0);
    for (int block_index=0; block_index<static_cast<int>(scalars.blocks.size()); ++block_index) {
        const auto& block = scalars.blocks[block_index];
        for (int index=0; index<block.dimension(); ++index) {
            const int offset = block.offset+index;
            const double probability = sigmoid(values[offset]);
            result[offset] = output_adjoint[offset]*scalar_constants[block_index]
                *(probability+values[offset]*probability*(1.0-probability));
        }
    }
    int gate_offset = scalars.dimension();
    int gated_offset = scalars.dimension()+gates.dimension();
    int output_offset = scalars.dimension();
    for (int block_index=0; block_index<static_cast<int>(gated.blocks.size()); ++block_index) {
        const auto& block = gated.blocks[block_index];
        const int width = 2*block.l+1;
        for (int feature=0; feature<block.multiplicity; ++feature) {
            const double probability = sigmoid(values[gate_offset]);
            const double gate = gate_constants[block_index]*probability;
            double gate_adjoint = 0.0;
            for (int component=0; component<width; ++component) {
                result[gated_offset] += gate*output_adjoint[output_offset];
                gate_adjoint += values[gated_offset]*output_adjoint[output_offset];
                ++gated_offset;
                ++output_offset;
            }
            result[gate_offset] = gate_adjoint*gate_constants[block_index]
                * probability*(1.0-probability);
            ++gate_offset;
        }
    }
    return result;
}

MaceNonlinear::Interaction::Interaction(const nlohmann::json& data)
    : source_embedding(checked_interaction(data).at("source_embedding")),
      target_embedding(data.at("target_embedding")),
      linear_up(data.at("linear_up")),
      skip(data.at("skip_tp")),
      linear_res(data.at("linear_res")),
      linear_1(data.at("linear_1")),
      linear_2(data.at("linear_2")),
      convolution(data.at("conv_tp")),
      convolution_weights(data.at("conv_tp_weights")),
      density(data.at("density_fn")),
      gate(data.at("gate")),
      alpha(data.at("alpha").get<double>()),
      beta(data.at("beta").get<double>())
{}

void MaceNonlinear::Interaction::prepare_pair_conditioning(
    int radial_size,
    int model_element_count,
    const std::vector<int>& selected_model_indices)
{
    if (!supports_pair_conditioning(radial_size, model_element_count))
        throw std::invalid_argument("MACE-MH-1 pair-conditioned MLP dimensions are inconsistent.");

    active_type_count = selected_model_indices.size();
    std::vector<std::vector<double>> source_embeddings(active_type_count);
    std::vector<std::vector<double>> target_embeddings(active_type_count);
    for (int type=0; type<active_type_count; ++type) {
        std::vector<double> attrs(model_element_count, 0.0);
        attrs.at(selected_model_indices[type]) = 1.0;
        source_embeddings[type] = source_embedding.evaluate(attrs);
        target_embeddings[type] = target_embedding.evaluate(attrs);
    }

    convolution_source_contributions.reserve(active_type_count);
    convolution_target_contributions.reserve(active_type_count);
    density_source_contributions.reserve(active_type_count);
    density_target_contributions.reserve(active_type_count);
    const int target_offset = radial_size+source_embedding.output_dimension();
    for (int type=0; type<active_type_count; ++type) {
        convolution_source_contributions.push_back(
            convolution_weights.first_layer_contribution(
                radial_size, source_embeddings[type]));
        convolution_target_contributions.push_back(
            convolution_weights.first_layer_contribution(
                target_offset, target_embeddings[type]));
        density_source_contributions.push_back(
            density.first_layer_contribution(radial_size, source_embeddings[type]));
        density_target_contributions.push_back(
            density.first_layer_contribution(target_offset, target_embeddings[type]));
    }
}

bool MaceNonlinear::Interaction::supports_pair_conditioning(
    int radial_size,
    int model_element_count) const
{
    const int source_width = source_embedding.output_dimension();
    const int target_width = target_embedding.output_dimension();
    const int conditioned_input_size = radial_size+source_width+target_width;
    return source_embedding.input_dimension() == model_element_count
        && target_embedding.input_dimension() == model_element_count
        && source_width > 0 && target_width > 0
        && convolution_weights.input_size() == conditioned_input_size
        && density.input_size() == conditioned_input_size
        && convolution_weights.supports_conditioned_input(radial_size)
        && density.supports_conditioned_input(radial_size);
}

void MaceNonlinear::Interaction::validate_conditioned_type(int type) const
{
    if (type < 0 || type >= active_type_count)
        throw std::out_of_range("MACE-MH-1 conditioned type is out of range.");
}

MaceNonlinear::Readout::Readout(const nlohmann::json& data)
    : nonlinear(data.at("class").get<std::string>() == "NonLinearReadoutBlock"),
      linear(readout_linear(data)),
      activation_constant(1.0)
{
    if (nonlinear) {
        if (data.at("activation").get<std::string>() != "silu")
            throw std::invalid_argument("MACE_Nonlinear readout activation is unsupported.");
        linear_1 = std::make_unique<E3Linear>(data.at("linear_1"));
        linear_2 = std::make_unique<E3Linear>(data.at("linear_2"));
        activation_constant = data.at("activation_constants").at(0).get<double>();
    }
}

double MaceNonlinear::Readout::evaluate(const std::vector<double>& features) const
{
    if (!nonlinear) return linear.evaluate(features).at(0);
    auto hidden = linear_1->evaluate(features);
    for (double& value : hidden) value = activation_constant*silu(value);
    return linear_2->evaluate(hidden).at(0);
}

std::vector<double> MaceNonlinear::Readout::reverse(
    const std::vector<double>& features,
    double output_adjoint) const
{
    if (!nonlinear) {
        std::vector<double> result;
        linear.reverse({output_adjoint}, result);
        return result;
    }
    auto hidden = linear_1->evaluate(features);
    auto activated = hidden;
    for (double& value : activated) value = activation_constant*silu(value);
    std::vector<double> activated_adjoint;
    linear_2->reverse({output_adjoint}, activated_adjoint);
    for (int index=0; index<static_cast<int>(hidden.size()); ++index) {
        const double probability = sigmoid(hidden[index]);
        activated_adjoint[index] *= activation_constant
            *(probability+hidden[index]*probability*(1.0-probability));
    }
    std::vector<double> result;
    linear_1->reverse(activated_adjoint, result);
    return result;
}

MaceNonlinear::MaceNonlinear(
    const std::string& filename, const std::string& requested_head)
    : MaceNonlinear(select_prediction_head(
        load_model_json(filename), requested_head))
{}

std::string MaceNonlinear::streamed_edges_mode() const
{
    return mace_streamed_edges_mode_name(streamed_edges);
}

void MaceNonlinear::set_streamed_edges(std::string mode)
{
    const auto requested = parse_mace_streamed_edges_mode(mode);
    if (mh1_fast_path && requested == MACEStreamedEdgesMode::materialized)
        throw std::invalid_argument(
            "MACE-MH-1 native serial execution supports only "
            "streamed_edges='generic'; 'materialized' is disabled.");
    if (requested != MACEStreamedEdgesMode::materialized && !supports_streamed_edges())
        throw std::invalid_argument(
            "Streamed edges require a compatible MACE-MH-1 family fast path.");
    if (mace_uses_prepared_execution(requested))
        throw std::invalid_argument(
            "MACE_Nonlinear direct execution currently "
            "requires the Kokkos evaluator.");
    streamed_edges = requested;
}

bool MaceNonlinear::streams_layer(int) const
{
    return streamed_edges == MACEStreamedEdgesMode::generic;
}

MaceNonlinear::MaceNonlinear(const nlohmann::json& data)
    : node_embedding(data.at("node_embedding"))
{
    validate_mace_nonlinear_schema(data);
    selected_head = data.value("selected_head", std::string());
    available_heads = data.value("available_heads", std::vector<std::string>());
    atomic_numbers = data.at("atomic_numbers").get<std::vector<int>>();
    model_atomic_numbers = data.at("model_atomic_numbers").get<std::vector<int>>();
    model_indices = data.at("model_indices").get<std::vector<int>>();
    model_num_elements = model_atomic_numbers.size();
    r_cut = data.at("r_cut").get<double>();
    l_max = data.at("l_max").get<int>();
    num_lm = (l_max+1)*(l_max+1);
    const auto& radial = data.at("radial_embedding");
    apply_cutoff = radial.at("apply_cutoff").get<bool>();
    bessel_weights = tensor_values(radial.at("basis").at("weights"));
    radial_prefactor = radial.at("basis").at("prefactor").get<double>();
    cutoff_power = radial.at("cutoff").at("p").get<int>();
    const auto& transform = radial.at("distance_transform");
    has_agnesi = transform.at("type").get<std::string>() == "agnesi";
    if (has_agnesi) {
        agnesi_a = transform.at("a").get<double>();
        agnesi_q = transform.at("q").get<double>();
        agnesi_p = transform.at("p").get<double>();
        covalent_radii = transform.at("covalent_radii").get<std::vector<double>>();
    }
    const auto scales = tensor_values(data.at("scale_shift").at("scale"));
    const auto shifts = tensor_values(data.at("scale_shift").at("shift"));
    scale = scales.at(0);
    shift = shifts.at(0);
    const auto all_atomic_energies = tensor_values(data.at("atomic_energies"));
    atomic_energies.resize(atomic_numbers.size());
    for (int type=0; type<static_cast<int>(atomic_numbers.size()); ++type)
        atomic_energies[type] = all_atomic_energies.at(model_indices[type]);
    for (const auto& value : data.at("interactions")) interactions.emplace_back(value);
    for (const auto& value : data.at("products")) {
        products.emplace_back(value);
        product_agnostic.push_back(value.value("use_agnostic_product", false));
    }
    for (const auto& value : data.at("readouts")) readouts.emplace_back(value);
    if (interactions.size() != products.size() || interactions.size() != readouts.size())
        throw std::invalid_argument("MACE_Nonlinear layer counts are inconsistent.");
    mh1_family = analyze_mh1_family_architecture(data);
    mh1_compiled_products = interactions.size() == 2 && products.size() == 2
        && products[0].uses_compiled_plan() && products[1].uses_compiled_plan();
    mh1_pair_conditioning = std::all_of(
            interactions.begin(), interactions.end(), [&](const auto& interaction) {
                return interaction.supports_pair_conditioning(
                    bessel_weights.size(), model_num_elements);
            });
    mh1_fast_path = mh1_family.compatible
        && mh1_compiled_products && mh1_pair_conditioning;
    if (!mh1_fast_path) {
        if (!mh1_family.compatible)
            mh1_fast_path_rejection = mh1_family.rejection_reason;
        else if (!mh1_compiled_products)
            mh1_fast_path_rejection = "Compiled correlation-three product plan is unavailable.";
        else
            mh1_fast_path_rejection = "Conditioned edge MLP plan is unavailable.";
    }
    if (mh1_fast_path)
        for (auto& interaction : interactions)
            interaction.prepare_pair_conditioning(
                bessel_weights.size(), model_num_elements, model_indices);
    if (supports_streamed_edges())
        streamed_edges = MACEStreamedEdgesMode::generic;
    has_zbl = data.at("has_zbl").get<bool>();
    if (has_zbl) {
        const auto& value = data.at("zbl");
        zbl = ZBL(value.at("a_exp").get<double>(), value.at("a_prefactor").get<double>(),
                  tensor_values(value.at("c")), tensor_values(value.at("covalent_radii")),
                  static_cast<int>(tensor_values(value.at("p")).at(0)));
    }
}

std::vector<double> MaceNonlinear::one_hot(int local_type) const
{
    if (local_type < 0 || local_type >= static_cast<int>(model_indices.size()))
        throw std::out_of_range("MACE_Nonlinear node type is out of range.");
    std::vector<double> result(model_num_elements, 0.0);
    result[model_indices[local_type]] = 1.0;
    return result;
}

double MaceNonlinear::cutoff(double distance) const
{
    if (distance >= r_cut) return 0.0;
    const double x = distance/r_cut;
    const double p = cutoff_power;
    return 1.0-0.5*(p+1.0)*(p+2.0)*std::pow(x, cutoff_power)
        +p*(p+2.0)*std::pow(x, cutoff_power+1)
        -0.5*p*(p+1.0)*std::pow(x, cutoff_power+2);
}

double MaceNonlinear::cutoff_derivative(double distance) const
{
    if (distance >= r_cut) return 0.0;
    const double x = distance/r_cut;
    const double p = cutoff_power;
    return (-0.5*p*(p+1.0)*(p+2.0)*std::pow(x,cutoff_power-1)
            +p*(p+1.0)*(p+2.0)*std::pow(x,cutoff_power)
            -0.5*p*(p+1.0)*(p+2.0)*std::pow(x,cutoff_power+1))/r_cut;
}

std::vector<double> MaceNonlinear::radial_features(double distance, int source_type, int target_type) const
{
    double transformed = distance;
    if (has_agnesi) {
        const int source_z = model_atomic_numbers.at(model_indices.at(source_type));
        const int target_z = model_atomic_numbers.at(model_indices.at(target_type));
        const double r0 = 0.5*(covalent_radii.at(source_z)+covalent_radii.at(target_z));
        const double ratio = distance/r0;
        transformed = 1.0/(1.0+agnesi_a*std::pow(ratio, agnesi_q)
                          /(1.0+std::pow(ratio, agnesi_q-agnesi_p)));
    }
    std::vector<double> result(bessel_weights.size());
    for (int index=0; index<static_cast<int>(result.size()); ++index)
        result[index] = radial_prefactor*std::sin(bessel_weights[index]*transformed)/transformed;
    if (apply_cutoff) for (double& value : result) value *= cutoff(distance);
    return result;
}

std::vector<double> MaceNonlinear::radial_feature_derivatives(
    double distance, int source_type, int target_type) const
{
    double transformed = distance;
    double transform_derivative = 1.0;
    if (has_agnesi) {
        const int source_z = model_atomic_numbers.at(model_indices.at(source_type));
        const int target_z = model_atomic_numbers.at(model_indices.at(target_type));
        const double r0 = 0.5*(covalent_radii.at(source_z)+covalent_radii.at(target_z));
        const double x = distance/r0;
        const double xs = std::pow(x,agnesi_q-agnesi_p);
        const double denominator = 1.0+xs;
        const double g = agnesi_a*std::pow(x,agnesi_q)/denominator;
        const double dgdx = agnesi_a*(
            agnesi_q*std::pow(x,agnesi_q-1.0)*denominator
            -(agnesi_q-agnesi_p)*std::pow(x,2.0*agnesi_q-agnesi_p-1.0)
        )/(denominator*denominator);
        transformed = 1.0/(1.0+g);
        transform_derivative = -dgdx/(r0*(1.0+g)*(1.0+g));
    }
    const double envelope = cutoff(distance);
    const double envelope_derivative = cutoff_derivative(distance);
    std::vector<double> result(bessel_weights.size());
    for (int index=0; index<static_cast<int>(result.size()); ++index) {
        const double weight = bessel_weights[index];
        const double base = radial_prefactor*std::sin(weight*transformed)/transformed;
        const double base_derivative = radial_prefactor
            *(weight*std::cos(weight*transformed)*transformed-std::sin(weight*transformed))
            /(transformed*transformed)*transform_derivative;
        result[index] = apply_cutoff
            ? base_derivative*envelope+base*envelope_derivative
            : base_derivative;
    }
    return result;
}

void MaceNonlinear::compute_spherical_harmonics(std::span<const double> xyz)
{
    std::vector<double> shuffled(xyz.size());
    for (int edge=0; edge<static_cast<int>(xyz.size()/3); ++edge) {
        shuffled[3*edge] = xyz[3*edge+2];
        shuffled[3*edge+1] = xyz[3*edge];
        shuffled[3*edge+2] = xyz[3*edge+1];
    }
    sphericart::SphericalHarmonics<double> calculator(l_max);
    calculator.compute_with_gradients(shuffled, spherical_harmonics, spherical_harmonic_gradients);
    for (double& value : spherical_harmonics) value *= 2.0*std::sqrt(std::numbers::pi);
    for (double& value : spherical_harmonic_gradients) value *= 2.0*std::sqrt(std::numbers::pi);
    auto shuffled_gradients = spherical_harmonic_gradients;
    for (int edge=0; edge<static_cast<int>(xyz.size()/3); ++edge)
        for (int lm=0; lm<num_lm; ++lm) {
            spherical_harmonic_gradients[(3*edge+0)*num_lm+lm] = shuffled_gradients[(3*edge+1)*num_lm+lm];
            spherical_harmonic_gradients[(3*edge+1)*num_lm+lm] = shuffled_gradients[(3*edge+2)*num_lm+lm];
            spherical_harmonic_gradients[(3*edge+2)*num_lm+lm] = shuffled_gradients[(3*edge+0)*num_lm+lm];
        }
}

std::vector<double> MaceNonlinear::node_slice(const std::vector<double>& values, int node, int width) const
{
    return std::vector<double>(values.begin()+node*width, values.begin()+(node+1)*width);
}

void MaceNonlinear::add_node_slice(std::vector<double>& values, int node, const std::vector<double>& addend) const
{
    for (int index=0; index<static_cast<int>(addend.size()); ++index)
        values[node*addend.size()+index] += addend[index];
}

void MaceNonlinear::compute_node_energies_forces(
    int num_nodes, std::span<const int> node_types, std::span<const int> num_neigh,
    std::span<const int> neigh_indices, std::span<const int> neigh_types,
    std::span<const double> xyz, std::span<const double> r)
{
    if (num_nodes < 0 || node_types.size() != static_cast<std::size_t>(num_nodes)
        || num_neigh.size() != static_cast<std::size_t>(num_nodes)
        || neigh_indices.size() != r.size() || neigh_types.size() != r.size()
        || xyz.size() != 3*r.size())
        throw std::invalid_argument("MACE_Nonlinear graph input sizes are inconsistent.");
    std::size_t edge_count = 0;
    for (int node=0; node<num_nodes; ++node) {
        if (node_types[node] < 0
            || node_types[node] >= static_cast<int>(atomic_numbers.size())
            || num_neigh[node] < 0)
            throw std::invalid_argument("MACE_Nonlinear graph has an invalid node type or neighbor count.");
        edge_count += static_cast<std::size_t>(num_neigh[node]);
        if (edge_count > r.size())
            throw std::invalid_argument("MACE_Nonlinear neighbor counts do not match the edge arrays.");
    }
    if (edge_count != r.size())
        throw std::invalid_argument("MACE_Nonlinear neighbor counts do not match the edge arrays.");
    last_edge_workspace_rows = streamed_edges == MACEStreamedEdgesMode::generic
        ? std::min(streamed_edge_block_size,static_cast<int>(r.size()))
        : static_cast<int>(r.size());
    for (std::size_t edge=0; edge<r.size(); ++edge) {
        if (neigh_indices[edge] < 0 || neigh_indices[edge] >= num_nodes
            || neigh_types[edge] < 0
            || neigh_types[edge] >= static_cast<int>(atomic_numbers.size())
            || neigh_types[edge] != node_types[neigh_indices[edge]]
            || !std::isfinite(r[edge]) || !(r[edge] > 0.0)
            || !std::isfinite(xyz[3*edge])
            || !std::isfinite(xyz[3*edge+1])
            || !std::isfinite(xyz[3*edge+2]))
            throw std::invalid_argument("MACE_Nonlinear graph has an invalid edge index, type, distance, or vector.");
    }
    compute_spherical_harmonics(xyz);
    std::vector<std::vector<double>> attrs(num_nodes);
    const int embedding_width = node_embedding.output_dimension();
    std::vector<double> features(num_nodes*embedding_width);
    for (int node=0; node<num_nodes; ++node) {
        attrs[node] = one_hot(node_types[node]);
        const auto embedded = node_embedding.evaluate(attrs[node]);
        std::copy(embedded.begin(), embedded.end(), features.begin()+node*embedding_width);
    }
    std::vector<double> cutoffs(r.size());
    const int radial_width = static_cast<int>(bessel_weights.size());
    std::vector<double> radial(r.size()*radial_width);
    std::vector<int> edge_targets(r.size());
    int edge = 0;
    for (int target=0; target<num_nodes; ++target)
        for (int local=0; local<num_neigh[target]; ++local, ++edge) {
            edge_targets[edge] = target;
            cutoffs[edge] = cutoff(r[edge]);
            const auto values = radial_features(
                r[edge], neigh_types[edge], node_types[target]);
            std::copy(
                values.begin(), values.end(),
                radial.begin()+static_cast<std::size_t>(edge)*radial_width);
        }
    auto prepare_conditioned_block = [&] (
        const Interaction& interaction, int first_edge, int samples,
        std::vector<double>& radial_block,
        std::vector<double>& convolution_contributions,
        std::vector<double>& density_contributions,
        AffineMLPBatchTape& convolution_tape,
        AffineMLPBatchTape& density_tape) {
        const int convolution_width =
            interaction.convolution_source_contributions.front().size();
        const int density_width =
            interaction.density_source_contributions.front().size();
        radial_block.assign(
            radial.begin()+static_cast<std::size_t>(first_edge)*radial_width,
            radial.begin()+static_cast<std::size_t>(first_edge+samples)*radial_width);
        convolution_contributions.resize(
            static_cast<std::size_t>(samples)*convolution_width);
        density_contributions.resize(static_cast<std::size_t>(samples)*density_width);
        for (int local_edge=0; local_edge<samples; ++local_edge) {
            const int edge_index = first_edge+local_edge;
            const int source_type = node_types[neigh_indices[edge_index]];
            const int target_type = node_types[edge_targets[edge_index]];
            interaction.validate_conditioned_type(source_type);
            interaction.validate_conditioned_type(target_type);
            for (int index=0; index<convolution_width; ++index)
                convolution_contributions[
                    static_cast<std::size_t>(local_edge)*convolution_width+index]
                    = interaction.convolution_source_contributions[source_type][index]
                    + interaction.convolution_target_contributions[target_type][index];
            for (int index=0; index<density_width; ++index)
                density_contributions[
                    static_cast<std::size_t>(local_edge)*density_width+index]
                    = interaction.density_source_contributions[source_type][index]
                    + interaction.density_target_contributions[target_type][index];
        }
        interaction.convolution_weights.evaluate_conditioned_batch(
            radial_block, samples, radial_width,
            convolution_contributions, convolution_tape);
        interaction.density.evaluate_conditioned_batch(
            radial_block, samples, radial_width, density_contributions, density_tape);
    };
    struct LayerState {
        std::vector<double> input, up, residual, skip, messages, linear_1_output;
        std::vector<double> pre_gate, interaction_output, output, densities;
        std::vector<int> product_elements;
        std::vector<std::vector<double>> source_embeddings, target_embeddings;
        std::vector<std::vector<double>> edge_features, raw_weights, weights;
        std::vector<double> density_raw, density_base;
        AffineMLPBatchTape convolution_tape, density_tape;
    };
    std::vector<LayerState> states(interactions.size());
    std::vector<std::vector<double>> layer_features;
    for (int layer=0; layer<static_cast<int>(interactions.size()); ++layer) {
        const auto& interaction = interactions[layer];
        auto& state = states[layer];
        state.input = features;
        const int up_width = interaction.linear_up.output_dimension();
        const int message_width = interaction.convolution.output_dimension();
        interaction.linear_up.evaluate_batch(
            features, num_nodes, state.up, linear_batch_workspace);
        interaction.linear_res.evaluate_batch(
            state.up, num_nodes, state.residual, linear_batch_workspace);
        interaction.skip.evaluate_batch(
            features, num_nodes, state.skip, linear_batch_workspace);
        if (!mh1_fast_path) {
            state.source_embeddings.resize(num_nodes);
            state.target_embeddings.resize(num_nodes);
        }
        for (int node=0; node<num_nodes; ++node) {
            if (!mh1_fast_path) {
                state.source_embeddings[node] = interaction.source_embedding.evaluate(attrs[node]);
                state.target_embeddings[node] = interaction.target_embedding.evaluate(attrs[node]);
            }
        }
        state.messages.assign(num_nodes*message_width,0.0);
        state.densities.assign(num_nodes,0.0);
        const bool stream_layer = streams_layer(layer);
        if (!mh1_fast_path) {
            state.edge_features.resize(r.size());
            state.raw_weights.resize(r.size());
        }
        if (!stream_layer) {
            state.weights.resize(r.size());
            state.density_raw.resize(r.size());
            state.density_base.resize(r.size());
        }
        if (mh1_fast_path && !stream_layer) {
            const int convolution_conditioning_width =
                interaction.convolution_source_contributions.front().size();
            const int density_conditioning_width =
                interaction.density_source_contributions.front().size();
            std::vector<double> convolution_contributions(
                r.size()*convolution_conditioning_width);
            std::vector<double> density_contributions(
                r.size()*density_conditioning_width);
            edge = 0;
            for (int target=0; target<num_nodes; ++target)
                for (int local=0; local<num_neigh[target]; ++local, ++edge) {
                    const int source_type = node_types[neigh_indices[edge]];
                    const int target_type = node_types[target];
                    interaction.validate_conditioned_type(source_type);
                    interaction.validate_conditioned_type(target_type);
                    for (int index=0; index<convolution_conditioning_width; ++index)
                        convolution_contributions[
                            static_cast<std::size_t>(edge)*convolution_conditioning_width+index]
                            = interaction.convolution_source_contributions[source_type][index]
                            + interaction.convolution_target_contributions[target_type][index];
                    for (int index=0; index<density_conditioning_width; ++index)
                        density_contributions[
                            static_cast<std::size_t>(edge)*density_conditioning_width+index]
                            = interaction.density_source_contributions[source_type][index]
                            + interaction.density_target_contributions[target_type][index];
                }
            interaction.convolution_weights.evaluate_conditioned_batch(
                radial, static_cast<int>(r.size()), radial_width,
                convolution_contributions, state.convolution_tape);
            interaction.density.evaluate_conditioned_batch(
                radial, static_cast<int>(r.size()), radial_width,
                density_contributions, state.density_tape);
        }
        if (!stream_layer) {
            edge = 0;
            for (int target=0; target<num_nodes; ++target)
                for (int local=0; local<num_neigh[target]; ++local, ++edge) {
                const int source = neigh_indices[edge];
                std::vector<double> edge_features;
                if (!mh1_fast_path) {
                    edge_features = node_slice(radial,edge,radial_width);
                    edge_features.insert(
                        edge_features.end(), state.source_embeddings[source].begin(),
                        state.source_embeddings[source].end());
                    edge_features.insert(
                        edge_features.end(), state.target_embeddings[target].begin(),
                        state.target_embeddings[target].end());
                    state.edge_features[edge] = edge_features;
                }
                std::vector<double> weights;
                if (mh1_fast_path) {
                    const auto& raw_weights = state.convolution_tape.values.back();
                    const int width = interaction.convolution_weights.output_size();
                    const auto first = raw_weights.begin()+static_cast<std::size_t>(edge)*width;
                    weights.assign(first, first+width);
                } else {
                    weights = interaction.convolution_weights.evaluate(edge_features);
                    state.raw_weights[edge] = weights;
                }
                if (!apply_cutoff) for (double& value : weights) value *= cutoffs[edge];
                state.weights[edge] = weights;
                const auto up_source = node_slice(state.up,source,up_width);
                const auto harmonics = std::vector<double>(
                    spherical_harmonics.begin()+edge*num_lm,
                    spherical_harmonics.begin()+(edge+1)*num_lm);
                add_node_slice(state.messages,target,interaction.convolution.evaluate(up_source,harmonics,weights));
                const double density_raw = mh1_fast_path
                    ? state.density_tape.values.back().at(edge)
                    : interaction.density.evaluate(edge_features).at(0);
                state.density_raw[edge] = density_raw;
                double density_value = std::tanh(density_raw*density_raw);
                state.density_base[edge] = density_value;
                if (!apply_cutoff) density_value *= cutoffs[edge];
                    state.densities[target] += density_value;
                }
        } else {
            for (int first_edge=0; first_edge<static_cast<int>(r.size());
                 first_edge+=streamed_edge_block_size) {
                const int samples = std::min(
                    streamed_edge_block_size,
                    static_cast<int>(r.size())-first_edge);
                std::vector<double> radial_block;
                std::vector<double> convolution_contributions;
                std::vector<double> density_contributions;
                AffineMLPBatchTape convolution_tape;
                AffineMLPBatchTape density_tape;
                prepare_conditioned_block(
                    interaction, first_edge, samples, radial_block,
                    convolution_contributions, density_contributions,
                    convolution_tape, density_tape);
                const auto& raw_weights = convolution_tape.values.back();
                const auto& density_raw_values = density_tape.values.back();
                const int weight_width = interaction.convolution_weights.output_size();
                for (int local_edge=0; local_edge<samples; ++local_edge) {
                    const int edge_index = first_edge+local_edge;
                    const int source = neigh_indices[edge_index];
                    const int target = edge_targets[edge_index];
                    const auto first = raw_weights.begin()
                        +static_cast<std::size_t>(local_edge)*weight_width;
                    std::vector<double> weights(first,first+weight_width);
                    if (!apply_cutoff)
                        for (double& value : weights) value *= cutoffs[edge_index];
                    const auto up_source = node_slice(state.up,source,up_width);
                    const auto harmonics = std::vector<double>(
                        spherical_harmonics.begin()+edge_index*num_lm,
                        spherical_harmonics.begin()+(edge_index+1)*num_lm);
                    add_node_slice(
                        state.messages,target,
                        interaction.convolution.evaluate(up_source,harmonics,weights));
                    const double raw = density_raw_values[local_edge];
                    double density_value = std::tanh(raw*raw);
                    if (!apply_cutoff) density_value *= cutoffs[edge_index];
                    state.densities[target] += density_value;
                }
            }
        }
        interaction.linear_1.evaluate_batch(
            state.messages, num_nodes, state.linear_1_output, linear_batch_workspace);
        state.pre_gate.resize(state.linear_1_output.size());
        std::vector<double> gated_values(
            static_cast<std::size_t>(num_nodes)*interaction.linear_2.input_dimension());
        for (int node=0; node<num_nodes; ++node) {
            const double normalization = interaction.alpha+interaction.beta*state.densities[node];
            for (int index=0; index<interaction.linear_1.output_dimension(); ++index)
                state.pre_gate[static_cast<std::size_t>(node)*interaction.linear_1.output_dimension()+index]
                    = state.linear_1_output[static_cast<std::size_t>(node)*interaction.linear_1.output_dimension()+index]
                        /normalization
                    +state.residual[static_cast<std::size_t>(node)*interaction.linear_res.output_dimension()+index];
            const auto gated = interaction.gate.evaluate(
                node_slice(state.pre_gate,node,interaction.gate.input.dimension()));
            std::copy(
                gated.begin(), gated.end(),
                gated_values.begin()
                    +static_cast<std::size_t>(node)*interaction.linear_2.input_dimension());
        }
        interaction.linear_2.evaluate_batch(
            gated_values, num_nodes, state.interaction_output, linear_batch_workspace);
        state.product_elements.resize(num_nodes);
        for (int node=0; node<num_nodes; ++node)
            state.product_elements[node] = product_agnostic[layer]
                ? 0 : model_indices[node_types[node]];
        products[layer].evaluate_batch(
            state.interaction_output, state.skip, state.product_elements,
            num_nodes, features, product_batch_workspace);
        state.output = features;
        layer_features.push_back(features);
    }
    node_energies.resize(num_nodes);
    for (int node=0; node<num_nodes; ++node) {
        double interaction_energy = has_zbl ? 0.0 : 0.0;
        for (int layer=0; layer<static_cast<int>(readouts.size()); ++layer)
            interaction_energy += readouts[layer].evaluate(node_slice(
                layer_features[layer],node,readouts[layer].linear.input_dimension()));
        node_energies[node] = atomic_energies[node_types[node]]+scale*interaction_energy+shift;
    }
    std::vector<std::vector<double>> layer_adjoints(interactions.size());
    for (int layer=0; layer<static_cast<int>(interactions.size()); ++layer)
        layer_adjoints[layer].assign(layer_features[layer].size(),0.0);
    for (int node=0; node<num_nodes; ++node)
        for (int layer=0; layer<static_cast<int>(readouts.size()); ++layer) {
            const int width = readouts[layer].linear.input_dimension();
            const auto contribution = readouts[layer].reverse(
                node_slice(layer_features[layer],node,width),scale);
            add_node_slice(layer_adjoints[layer],node,contribution);
        }

    std::vector<std::vector<double>> radial_adjoints(r.size(),std::vector<double>(bessel_weights.size(),0.0));
    std::vector<double> harmonic_adjoints(r.size()*num_lm,0.0);
    std::vector<double> cutoff_adjoints(r.size(),0.0);
    for (int layer=static_cast<int>(interactions.size())-1; layer>=0; --layer) {
        const auto& interaction = interactions[layer];
        const auto& state = states[layer];
        const int up_width = interaction.linear_up.output_dimension();
        const int message_width = interaction.convolution.output_dimension();
        const int pre_gate_width = interaction.gate.input.dimension();
        std::vector<double> interaction_output_adj;
        std::vector<double> skip_adj;
        products[layer].reverse_batch(
            state.interaction_output, state.product_elements, layer_adjoints[layer],
            num_nodes, interaction_output_adj, skip_adj, product_batch_workspace);
        std::vector<double> gated_adj;
        interaction.linear_2.reverse_batch(
            interaction_output_adj, num_nodes, gated_adj, linear_batch_workspace);
        std::vector<double> linear_adj(
            static_cast<std::size_t>(num_nodes)*pre_gate_width);
        std::vector<double> residual_adj(
            static_cast<std::size_t>(num_nodes)*interaction.linear_res.output_dimension());
        std::vector<double> density_adj(num_nodes,0.0);
        for (int node=0; node<num_nodes; ++node) {
            const auto pre_gate = node_slice(state.pre_gate,node,pre_gate_width);
            const auto pre_gate_adj = interaction.gate.reverse(
                pre_gate,
                node_slice(gated_adj,node,interaction.linear_2.input_dimension()));
            const double normalization = interaction.alpha+interaction.beta*state.densities[node];
            for (int index=0; index<pre_gate_width; ++index) {
                const std::size_t offset = static_cast<std::size_t>(node)*pre_gate_width+index;
                linear_adj[offset] = pre_gate_adj[index]/normalization;
                residual_adj[offset] = pre_gate_adj[index];
                density_adj[node] -= pre_gate_adj[index]*state.linear_1_output[offset]
                    *interaction.beta/(normalization*normalization);
            }
        }
        std::vector<double> message_adj;
        interaction.linear_1.reverse_batch(
            linear_adj, num_nodes, message_adj, linear_batch_workspace);
        std::vector<double> up_adj;
        interaction.linear_res.reverse_batch(
            residual_adj, num_nodes, up_adj, linear_batch_workspace);
        const bool stream_layer = streams_layer(layer);
        if (!stream_layer) {
            std::vector<double> convolution_output_adjoints;
            std::vector<double> density_output_adjoints;
            if (mh1_fast_path) {
                convolution_output_adjoints.assign(
                    r.size()*interaction.convolution_weights.output_size(), 0.0);
                density_output_adjoints.assign(r.size(), 0.0);
            }
            edge = 0;
            for (int target=0; target<num_nodes; ++target)
                for (int local=0; local<num_neigh[target]; ++local, ++edge) {
                const int source = neigh_indices[edge];
                const auto up_source = node_slice(state.up,source,up_width);
                const auto harmonics = std::vector<double>(
                    spherical_harmonics.begin()+edge*num_lm,
                    spherical_harmonics.begin()+(edge+1)*num_lm);
                std::vector<double> up_source_adj,harmonics_adj,weights_adj;
                interaction.convolution.reverse(
                    up_source,harmonics,state.weights[edge],
                    node_slice(message_adj,target,message_width),
                    up_source_adj,harmonics_adj,weights_adj);
                add_node_slice(up_adj,source,up_source_adj);
                for (int lm=0; lm<num_lm; ++lm) harmonic_adjoints[edge*num_lm+lm] += harmonics_adj[lm];
                if (!apply_cutoff) {
                    for (int index=0; index<static_cast<int>(weights_adj.size()); ++index) {
                        const double raw_weight = mh1_fast_path
                            ? state.convolution_tape.values.back()[
                                static_cast<std::size_t>(edge)*weights_adj.size()+index]
                            : state.raw_weights[edge][index];
                        cutoff_adjoints[edge] += weights_adj[index]*raw_weight;
                        weights_adj[index] *= cutoffs[edge];
                    }
                }
                const double raw = state.density_raw[edge];
                double density_raw_adj = density_adj[target]
                    *(1.0-state.density_base[edge]*state.density_base[edge])*2.0*raw;
                if (!apply_cutoff) {
                    cutoff_adjoints[edge] += density_adj[target]*state.density_base[edge];
                    density_raw_adj *= cutoffs[edge];
                }
                if (mh1_fast_path) {
                    std::copy(
                        weights_adj.begin(), weights_adj.end(),
                        convolution_output_adjoints.begin()
                            +static_cast<std::size_t>(edge)*weights_adj.size());
                    density_output_adjoints[edge] = density_raw_adj;
                } else {
                    auto edge_feature_adj = interaction.convolution_weights.evaluate_gradient(
                        state.edge_features[edge], weights_adj);
                    const auto density_feature_adj = interaction.density.evaluate_gradient(
                        state.edge_features[edge], {density_raw_adj});
                    for (int index=0; index<static_cast<int>(edge_feature_adj.size()); ++index)
                        edge_feature_adj[index] += density_feature_adj[index];
                    for (int index=0; index<radial_width; ++index)
                        radial_adjoints[edge][index] += edge_feature_adj[index];
                }
                }
            if (mh1_fast_path) {
                std::vector<double> convolution_input_adjoints;
                interaction.convolution_weights.reverse_conditioned_batch(
                    convolution_output_adjoints, state.convolution_tape,
                    convolution_input_adjoints, affine_batch_workspace);
                std::vector<double> density_input_adjoints;
                interaction.density.reverse_conditioned_batch(
                    density_output_adjoints, state.density_tape,
                    density_input_adjoints, affine_batch_workspace);
                for (std::size_t edge_index=0; edge_index<r.size(); ++edge_index)
                    for (int index=0; index<radial_width; ++index)
                        radial_adjoints[edge_index][index]
                            += convolution_input_adjoints[edge_index*radial_width+index]
                            + density_input_adjoints[edge_index*radial_width+index];
            }
        } else {
            const int weight_width = interaction.convolution_weights.output_size();
            for (int first_edge=0; first_edge<static_cast<int>(r.size());
                 first_edge+=streamed_edge_block_size) {
                const int samples = std::min(
                    streamed_edge_block_size,
                    static_cast<int>(r.size())-first_edge);
                std::vector<double> radial_block;
                std::vector<double> convolution_contributions;
                std::vector<double> density_contributions;
                AffineMLPBatchTape convolution_tape;
                AffineMLPBatchTape density_tape;
                prepare_conditioned_block(
                    interaction, first_edge, samples, radial_block,
                    convolution_contributions, density_contributions,
                    convolution_tape, density_tape);
                const auto& raw_weights = convolution_tape.values.back();
                const auto& density_raw_values = density_tape.values.back();
                std::vector<double> convolution_output_adjoints(
                    static_cast<std::size_t>(samples)*weight_width,0.0);
                std::vector<double> density_output_adjoints(samples,0.0);
                for (int local_edge=0; local_edge<samples; ++local_edge) {
                    const int edge_index = first_edge+local_edge;
                    const int source = neigh_indices[edge_index];
                    const int target = edge_targets[edge_index];
                    const auto raw_first = raw_weights.begin()
                        +static_cast<std::size_t>(local_edge)*weight_width;
                    std::vector<double> weights(raw_first,raw_first+weight_width);
                    if (!apply_cutoff)
                        for (double& value : weights) value *= cutoffs[edge_index];
                    const auto up_source = node_slice(state.up,source,up_width);
                    const auto harmonics = std::vector<double>(
                        spherical_harmonics.begin()+edge_index*num_lm,
                        spherical_harmonics.begin()+(edge_index+1)*num_lm);
                    std::vector<double> up_source_adj,harmonics_adj,weights_adj;
                    interaction.convolution.reverse(
                        up_source,harmonics,weights,
                        node_slice(message_adj,target,message_width),
                        up_source_adj,harmonics_adj,weights_adj);
                    add_node_slice(up_adj,source,up_source_adj);
                    for (int lm=0; lm<num_lm; ++lm)
                        harmonic_adjoints[edge_index*num_lm+lm] += harmonics_adj[lm];
                    if (!apply_cutoff)
                        for (int index=0; index<weight_width; ++index) {
                            cutoff_adjoints[edge_index]
                                += weights_adj[index]*raw_first[index];
                            weights_adj[index] *= cutoffs[edge_index];
                        }
                    const double raw = density_raw_values[local_edge];
                    const double density_base = std::tanh(raw*raw);
                    double density_raw_adj = density_adj[target]
                        *(1.0-density_base*density_base)*2.0*raw;
                    if (!apply_cutoff) {
                        cutoff_adjoints[edge_index] += density_adj[target]*density_base;
                        density_raw_adj *= cutoffs[edge_index];
                    }
                    std::copy(
                        weights_adj.begin(), weights_adj.end(),
                        convolution_output_adjoints.begin()
                            +static_cast<std::size_t>(local_edge)*weight_width);
                    density_output_adjoints[local_edge] = density_raw_adj;
                }
                std::vector<double> convolution_input_adjoints;
                interaction.convolution_weights.reverse_conditioned_batch(
                    convolution_output_adjoints, convolution_tape,
                    convolution_input_adjoints, affine_batch_workspace);
                std::vector<double> density_input_adjoints;
                interaction.density.reverse_conditioned_batch(
                    density_output_adjoints, density_tape,
                    density_input_adjoints, affine_batch_workspace);
                for (int local_edge=0; local_edge<samples; ++local_edge)
                    for (int index=0; index<radial_width; ++index)
                        radial_adjoints[first_edge+local_edge][index]
                            += convolution_input_adjoints[
                                static_cast<std::size_t>(local_edge)*radial_width+index]
                            + density_input_adjoints[
                                static_cast<std::size_t>(local_edge)*radial_width+index];
            }
        }
        std::vector<double> input_adj(state.input.size(),0.0);
        interaction.linear_up.reverse_batch(
            up_adj, num_nodes, input_adj, linear_batch_workspace);
        interaction.skip.reverse_batch(
            skip_adj, num_nodes, input_adj, linear_batch_workspace);
        if (layer > 0)
            for (int index=0; index<static_cast<int>(input_adj.size()); ++index)
                layer_adjoints[layer-1][index] += input_adj[index];
    }

    node_forces.assign(xyz.size(),0.0);
    edge = 0;
    for (int target=0; target<num_nodes; ++target)
        for (int local=0; local<num_neigh[target]; ++local, ++edge) {
            const auto derivatives = radial_feature_derivatives(r[edge],neigh_types[edge],node_types[target]);
            double radial_distance_adjoint = cutoff_adjoints[edge]*cutoff_derivative(r[edge]);
            for (int index=0; index<static_cast<int>(derivatives.size()); ++index)
                radial_distance_adjoint += radial_adjoints[edge][index]*derivatives[index];
            for (int component=0; component<3; ++component) {
                double vector_adjoint = radial_distance_adjoint*xyz[3*edge+component]/r[edge];
                for (int lm=0; lm<num_lm; ++lm)
                    vector_adjoint += harmonic_adjoints[edge*num_lm+lm]
                        *spherical_harmonic_gradients[(3*edge+component)*num_lm+lm];
                node_forces[3*edge+component] = -vector_adjoint;
            }
        }
    if (has_zbl) {
        std::vector<double> zbl_energies(num_nodes,0.0), zbl_forces(xyz.size(),0.0);
        zbl.compute_ZBL(num_nodes,node_types,num_neigh,neigh_types,atomic_numbers,r,xyz,zbl_energies,zbl_forces);
        for (int node=0; node<num_nodes; ++node) node_energies[node] += scale*zbl_energies[node];
        for (int index=0; index<static_cast<int>(node_forces.size()); ++index)
            node_forces[index] += scale*zbl_forces[index];
    }
}
