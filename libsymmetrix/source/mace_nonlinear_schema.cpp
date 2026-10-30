#include "mace_nonlinear_schema.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "e3nn.hpp"

namespace {

const nlohmann::json& tensor(
    const nlohmann::json& parent, const char* key, const char* name,
    const std::string& prefix)
{
    const auto& value = parent.at(key);
    if (!value.is_object() || !value.contains("shape") || !value.contains("values")
        || !value.at("shape").is_array() || !value.at("values").is_array())
        throw std::invalid_argument(
            prefix + name + " tensor is malformed.");
    std::size_t size = 1;
    for (const auto& extent_value : value.at("shape")) {
        const int extent = extent_value.get<int>();
        if (extent < 0
            || (extent != 0
                && size > std::numeric_limits<std::size_t>::max()
                    / static_cast<std::size_t>(extent)))
            throw std::invalid_argument(
                prefix + name + " tensor shape is invalid.");
        size *= static_cast<std::size_t>(extent);
    }
    if (size != value.at("values").size())
        throw std::invalid_argument(
            prefix + name + " tensor shape is invalid.");
    return value;
}

void require_finite(double value, const char* name, const std::string& prefix)
{
    if (!std::isfinite(value))
        throw std::invalid_argument(prefix + name + " must be finite.");
}

bool blocks_equal(
    const Irreps& actual,
    const std::vector<std::tuple<int, int, int>>& expected)
{
    if (actual.blocks.size() != expected.size()) return false;
    for (int index=0; index<static_cast<int>(actual.blocks.size()); ++index) {
        const auto& block = actual.blocks[index];
        const auto& [multiplicity, l, parity] = expected[index];
        if (block.multiplicity != multiplicity || block.l != l
            || block.parity != parity)
            return false;
    }
    return true;
}

bool blocks_equal(const Irreps& actual, const Irreps& expected)
{
    if (actual.blocks.size() != expected.blocks.size()) return false;
    for (std::size_t index=0; index<actual.blocks.size(); ++index) {
        const auto& actual_block = actual.blocks[index];
        const auto& expected_block = expected.blocks[index];
        if (actual_block.multiplicity != expected_block.multiplicity
            || actual_block.l != expected_block.l
            || actual_block.parity != expected_block.parity)
            return false;
    }
    return true;
}

bool is_exact_true(const nlohmann::json& object, const char* key)
{
    return object.is_object() && object.contains(key)
        && object.at(key).is_boolean() && object.at(key).get<bool>();
}

bool has_polar_keep_last_policy(
    const nlohmann::json& data, const MaceBackboneSchema schema)
{
    // The caller has already validated the selected family schema and its
    // contract identity.  Keep the policy decision tied to that identity: a
    // plain MACE_Nonlinear file cannot activate it by copying polar fields.
    return schema == MaceBackboneSchema::polar && data.contains("polar")
        && is_exact_true(data.at("polar"), "keep_last_layer_irreps");
}

bool has_only_even_scalar_blocks(const Irreps& irreps)
{
    if (irreps.blocks.empty()) return false;
    for (const auto& block : irreps.blocks)
        if (block.multiplicity <= 0 || block.l != 0 || block.parity != 1)
            return false;
    return true;
}

bool has_scalar_only_linear_paths(const nlohmann::json& linear)
{
    const Irreps input(linear.at("irreps_in").get<std::string>());
    const Irreps output(linear.at("irreps_out").get<std::string>());
    const auto& instructions = linear.at("instructions");
    if (!has_only_even_scalar_blocks(output) || instructions.empty())
        return false;
    for (const auto& instruction : instructions) {
        const int input_index = instruction.at("i_in").get<int>();
        const int output_index = instruction.at("i_out").get<int>();
        if (input_index < 0
            || input_index >= static_cast<int>(input.blocks.size())
            || output_index < 0
            || output_index >= static_cast<int>(output.blocks.size()))
            return false;
        const auto& input_block = input.blocks[input_index];
        const auto& output_block = output.blocks[output_index];
        if (input_block.l != 0 || input_block.parity != 1
            || output_block.l != 0 || output_block.parity != 1
            || instruction.at("path_shape").get<std::vector<int>>()
                != std::vector<int>{
                    input_block.multiplicity, output_block.multiplicity})
            return false;
    }
    return true;
}

bool has_scalar_only_energy_readout(
    const nlohmann::json& readout,
    const std::vector<std::tuple<int, int, int>>& product_output)
{
    const auto& linear_1 = readout.at("linear_1");
    const auto& linear_2 = readout.at("linear_2");
    const Irreps linear_1_output(
        linear_1.at("irreps_out").get<std::string>());
    const Irreps linear_2_input(
        linear_2.at("irreps_in").get<std::string>());
    if (!blocks_equal(
            Irreps(linear_1.at("irreps_in").get<std::string>()),
            product_output)
        || !blocks_equal(linear_1_output, linear_2_input)
        || !has_scalar_only_linear_paths(linear_1)
        || !has_scalar_only_linear_paths(linear_2)
        || !blocks_equal(
            Irreps(linear_2.at("irreps_out").get<std::string>()),
            std::vector<std::tuple<int, int, int>>{{1, 0, 1}}))
        return false;
    if (readout.at("class").get<std::string>() != "NonLinearBiasReadoutBlock")
        return true;
    const auto& linear_mid = readout.at("linear_mid");
    return blocks_equal(
               Irreps(linear_mid.at("irreps_in").get<std::string>()),
               linear_1_output)
        && blocks_equal(
            Irreps(linear_mid.at("irreps_out").get<std::string>()),
            linear_2_input)
        && has_scalar_only_linear_paths(linear_mid);
}

std::vector<std::tuple<int, int, int>> spherical_blocks(
    int multiplicity, int l_max, int scalar_copies=1)
{
    std::vector<std::tuple<int, int, int>> result;
    result.emplace_back(scalar_copies*multiplicity, 0, 1);
    for (int l=1; l<=l_max; ++l)
        result.emplace_back(multiplicity, l, l%2 == 0 ? 1 : -1);
    return result;
}

std::vector<MaceMH1IrrepBlockPattern> block_pattern(const Irreps& irreps)
{
    std::vector<MaceMH1IrrepBlockPattern> result;
    result.reserve(irreps.blocks.size());
    for (const auto& block : irreps.blocks)
        result.push_back({block.multiplicity, block.l, block.parity});
    return result;
}

int layer_uvu_channels(const nlohmann::json& interaction)
{
    const auto& tensor_product = interaction.at("conv_tp");
    const Irreps input_1(tensor_product.at("irreps_in1").get<std::string>());
    const Irreps output(tensor_product.at("irreps_out").get<std::string>());
    int channels = 0;
    for (const auto& instruction : tensor_product.at("instructions")) {
        const int input_index = instruction.at("i_in1").get<int>();
        const int output_index = instruction.at("i_out").get<int>();
        if (instruction.at("connection_mode").get<std::string>() != "uvu"
            || input_index < 0
            || input_index >= static_cast<int>(input_1.blocks.size())
            || output_index < 0
            || output_index >= static_cast<int>(output.blocks.size()))
            return 0;
        const int input_channels = input_1.blocks[input_index].multiplicity;
        const int output_channels = output.blocks[output_index].multiplicity;
        if (input_channels != output_channels
            || (channels != 0 && channels != input_channels))
            return 0;
        channels = input_channels;
    }
    return channels;
}

bool has_external_uvu_tensor_layout(
    const nlohmann::json& interaction, int multiplicity)
{
    const auto& tensor_product = interaction.at("conv_tp");
    const Irreps input_1(tensor_product.at("irreps_in1").get<std::string>());
    const Irreps input_2(tensor_product.at("irreps_in2").get<std::string>());
    const Irreps output(tensor_product.at("irreps_out").get<std::string>());
    if (!tensor_product.at("weight").at("values").empty()
        || tensor_product.at("instructions").empty())
        return false;
    for (const auto& instruction : tensor_product.at("instructions")) {
        const int i1 = instruction.at("i_in1").get<int>();
        const int i2 = instruction.at("i_in2").get<int>();
        const int io = instruction.at("i_out").get<int>();
        if (i1 < 0 || i1 >= static_cast<int>(input_1.blocks.size())
            || i2 < 0 || i2 >= static_cast<int>(input_2.blocks.size())
            || io < 0 || io >= static_cast<int>(output.blocks.size()))
            return false;
        const auto& a = input_1.blocks[i1];
        const auto& b = input_2.blocks[i2];
        const auto& c = output.blocks[io];
        if (instruction.at("connection_mode").get<std::string>() != "uvu"
            || !instruction.at("has_weight").get<bool>()
            || a.multiplicity != multiplicity || b.multiplicity != 1
            || c.multiplicity != multiplicity
            || instruction.at("path_shape").get<std::vector<int>>()
                != std::vector<int>{multiplicity, 1})
            return false;
    }
    return true;
}

} // namespace

const char* mace_mh1_final_output_policy_name(
    const MaceMH1FinalOutputPolicy policy)
{
    switch (policy) {
    case MaceMH1FinalOutputPolicy::not_final:
        return "not_final";
    case MaceMH1FinalOutputPolicy::scalar:
        return "scalar";
    case MaceMH1FinalOutputPolicy::keep_last:
        return "keep_last";
    }
    return "unknown";
}

namespace {

//: The three rule differences between the two families' schemas, and nothing
//: else: the model type a file declares, whether ``scale_shift`` must be a
//: 0-dimensional scalar, and whether ``atomic_energies`` must be the
//: ``[1, num_elements]`` matrix.  Both families' evaluators run the same
//: structural checks below under their own rule row.
struct BackboneSchemaRules {
    const char* model_type;
    bool scalar_scale_shift;
    bool matrix_atomic_energies;
};

const BackboneSchemaRules& backbone_schema_rules(const MaceBackboneSchema schema)
{
    static const BackboneSchemaRules mh1_rules{
        "MACE_Nonlinear", true, true};
    static const BackboneSchemaRules polar_rules{"MACE_Polar", false, false};
    return schema == MaceBackboneSchema::polar ? polar_rules : mh1_rules;
}

} // namespace

void validate_mace_backbone_schema(
    const nlohmann::json& data, const MaceBackboneSchema schema)
{
    const BackboneSchemaRules& rules = backbone_schema_rules(schema);
    const std::string prefix = std::string(rules.model_type) + " ";
    const auto fail = [&prefix](const std::string& message) {
        throw std::invalid_argument(prefix + message);
    };
    if (!data.is_object()
        || data.value("model_type", std::string()) != rules.model_type
        || data.at("symmetrix_format_version").get<int>() != 3)
        fail(std::string("evaluator requires model_type ") + rules.model_type
            + " and format-version-3 nonlinear JSON.");

    const auto atomic_numbers = data.at("atomic_numbers").get<std::vector<int>>();
    const auto model_atomic_numbers =
        data.at("model_atomic_numbers").get<std::vector<int>>();
    const auto model_indices = data.at("model_indices").get<std::vector<int>>();
    if (atomic_numbers.empty() || model_atomic_numbers.empty()
        || model_indices.size() != atomic_numbers.size()
        || data.at("num_elements").get<int>() != static_cast<int>(atomic_numbers.size())
        || std::set<int>(atomic_numbers.begin(), atomic_numbers.end()).size()
            != atomic_numbers.size()
        || std::set<int>(model_atomic_numbers.begin(), model_atomic_numbers.end()).size()
            != model_atomic_numbers.size())
        fail("atomic-number metadata is inconsistent.");
    if (std::any_of(atomic_numbers.begin(), atomic_numbers.end(), [](int value) { return value <= 0; })
        || std::any_of(model_atomic_numbers.begin(), model_atomic_numbers.end(), [](int value) { return value <= 0; }))
        fail("atomic numbers must be positive.");
    for (int local_type=0; local_type<static_cast<int>(atomic_numbers.size()); ++local_type) {
        const int model_type = model_indices[local_type];
        if (model_type < 0 || model_type >= static_cast<int>(model_atomic_numbers.size())
            || model_atomic_numbers[model_type] != atomic_numbers[local_type])
            fail("model_indices are inconsistent.");
    }

    const double r_cut = data.at("r_cut").get<double>();
    const int l_max = data.at("l_max").get<int>();
    if (!(r_cut > 0.0) || !std::isfinite(r_cut) || l_max < 0
        || l_max >= static_cast<int>(std::sqrt(std::numeric_limits<int>::max()))-1)
        fail("cutoff or l_max is invalid.");

    const int num_interactions = data.at("num_interactions").get<int>();
    if (num_interactions <= 0 || !data.at("interactions").is_array()
        || !data.at("products").is_array() || !data.at("readouts").is_array()
        || data.at("interactions").size() != static_cast<std::size_t>(num_interactions)
        || data.at("products").size() != static_cast<std::size_t>(num_interactions)
        || data.at("readouts").size() != static_cast<std::size_t>(num_interactions))
        fail("layer counts are inconsistent.");

    const auto& radial = data.at("radial_embedding");
    const int num_spline_points = radial.value("num_spline_points", 256);
    if (num_spline_points < 4)
        fail("radial num_spline_points must be at least 4.");
    const auto& basis = radial.at("basis");
    const auto& cutoff = radial.at("cutoff");
    if (basis.at("type").get<std::string>() != "bessel"
        || cutoff.at("type").get<std::string>() != "polynomial"
        || cutoff.at("p").get<int>() < 1
        || cutoff.at("r_max").get<double>() != r_cut)
        fail("radial basis or cutoff is unsupported.");
    const auto& basis_weights =
        tensor(basis, "weights", "radial basis weights", prefix);
    if (basis_weights.at("shape").size() != 1 || basis_weights.at("values").empty())
        fail("radial basis weights are invalid.");
    require_finite(
        basis.at("prefactor").get<double>(), "radial prefactor", prefix);

    const auto& transform = radial.at("distance_transform");
    const auto transform_type = transform.at("type").get<std::string>();
    if (transform_type != "none" && transform_type != "agnesi")
        fail("distance transform is unsupported.");
    if (transform_type == "agnesi") {
        require_finite(transform.at("a").get<double>(), "Agnesi a", prefix);
        require_finite(transform.at("q").get<double>(), "Agnesi q", prefix);
        require_finite(transform.at("p").get<double>(), "Agnesi p", prefix);
        const auto radii = transform.at("covalent_radii").get<std::vector<double>>();
        const int maximum_atomic_number =
            *std::max_element(model_atomic_numbers.begin(), model_atomic_numbers.end());
        if (maximum_atomic_number < 0
            || radii.size() <= static_cast<std::size_t>(maximum_atomic_number))
            fail("Agnesi covalent radii are incomplete.");
        for (int atomic_number : model_atomic_numbers)
            if (!(radii[atomic_number] > 0.0) || !std::isfinite(radii[atomic_number]))
                fail("Agnesi covalent radii are invalid.");
    }

    const auto& scale =
        tensor(data.at("scale_shift"), "scale", "scale", prefix);
    const auto& shift =
        tensor(data.at("scale_shift"), "shift", "shift", prefix);
    // The polar family's released checkpoints serialize each scalar as a
    // leading size-1 tensor; both spellings hold one value, and the evaluator
    // reads element zero either way.
    const auto is_scalar = [](const nlohmann::json& value) {
        return value.at("shape").empty()
            || value.at("shape") == nlohmann::json{1};
    };
    if (rules.scalar_scale_shift) {
        if (!scale.at("shape").empty() || !shift.at("shape").empty())
            fail("scale and shift must be scalar tensors.");
    } else if (!is_scalar(scale) || !is_scalar(shift)) {
        fail("scale and shift must be scalar tensors (a leading size-1 extent is "
            "admitted for the polar checkpoints' spelling).");
    }
    require_finite(scale.at("values").at(0).get<double>(), "scale", prefix);
    require_finite(shift.at("values").at(0).get<double>(), "shift", prefix);

    const auto& atomic_energies =
        tensor(data, "atomic_energies", "atomic energies", prefix);
    const auto energy_shape = atomic_energies.at("shape").get<std::vector<int>>();
    const int element_count = static_cast<int>(model_atomic_numbers.size());
    const std::vector<int> matrix_shape{1, element_count};
    const std::vector<int> vector_shape{element_count};
    if (rules.matrix_atomic_energies) {
        if (energy_shape != matrix_shape)
            fail("atomic-energy tensor has an invalid shape.");
    } else if (energy_shape != matrix_shape && energy_shape != vector_shape) {
        fail("atomic-energy tensor must be shaped like its element count, as a "
            "vector or a leading size-1 matrix.");
    }
}

void validate_mace_nonlinear_schema(const nlohmann::json& data)
{
    validate_mace_backbone_schema(data, MaceBackboneSchema::mh1);
}

MaceMH1FamilyDescriptor analyze_mh1_family_architecture(
    const nlohmann::json& data, const MaceBackboneSchema schema)
{
    MaceMH1FamilyDescriptor result;
    auto reject = [&](const std::string& reason) {
        result.rejection_reason = reason;
        return result;
    };
    const auto reject_layer = [&](const std::size_t layer, const char* invariant) {
        return reject(
            "MH-1 layer " + std::to_string(layer) + " " + invariant
            + " is inconsistent.");
    };
    const auto interaction_count = data.at("num_interactions").get<int>();
    if (interaction_count < 1
        || data.at("interactions").size()
            != static_cast<std::size_t>(interaction_count)
        || data.at("products").size() != data.at("interactions").size()
        || data.at("readouts").size() != data.at("interactions").size())
        return reject(
            "MH-1 family execution requires one product and one readout per "
            "interaction.");
    const auto last = data.at("interactions").size() - 1;
    const bool keep_last = has_polar_keep_last_policy(data, schema);
    result.layers.reserve(data.at("interactions").size());
    for (std::size_t layer = 0;
         layer < data.at("interactions").size(); ++layer) {
        const auto& interaction = data.at("interactions").at(layer);
        const auto& product = data.at("products").at(layer);
        const auto edge_blocks = block_pattern(
            Irreps(interaction.at("edge_irreps").get<std::string>()));
        const auto product_output_blocks = block_pattern(Irreps(
            product.at("symmetric_contractions")
                .at("irreps_out").get<std::string>()));
        const auto output_policy = layer != last
            ? MaceMH1FinalOutputPolicy::not_final
            : (keep_last ? MaceMH1FinalOutputPolicy::keep_last
                         : MaceMH1FinalOutputPolicy::scalar);
        result.layers.push_back({
            layer, layer_uvu_channels(interaction), edge_blocks,
            product_output_blocks, output_policy});
    }
    result.l_max = data.at("l_max").get<int>();
    if (result.l_max != 2 && result.l_max != 3)
        return reject("MH-1 family execution supports l_max=2 or l_max=3.");
    result.radial_size = static_cast<int>(
        data.at("radial_embedding").at("basis").at("weights").at("values").size());
    if (result.radial_size <= 0)
        return reject("MH-1 family execution requires a non-empty radial basis.");

    const auto& first = data.at("interactions").at(0);
    for (std::size_t layer = 0;
         layer < data.at("interactions").size(); ++layer) {
        const auto& interaction = data.at("interactions").at(layer);
        if (interaction.at("class").get<std::string>()
            != "RealAgnosticResidualNonLinearInteractionBlock")
            return reject_layer(layer, "interaction class");
    }

    const Irreps first_nodes(first.at("node_feats_irreps").get<std::string>());
    if (first_nodes.blocks.size() != 1 || first_nodes.blocks[0].l != 0
        || first_nodes.blocks[0].parity != 1)
        return reject_layer(0, "node representation");
    result.node_channels = first_nodes.blocks[0].multiplicity;
    const auto full = spherical_blocks(result.node_channels, result.l_max);
    const auto gate_input = spherical_blocks(
        result.node_channels, result.l_max, result.l_max+1);
    const std::vector<std::tuple<int, int, int>> node_hidden = {
        {result.node_channels, 0, 1}, {result.node_channels, 1, -1}};
    const std::vector<std::tuple<int, int, int>> scalar_hidden = {
        {result.node_channels, 0, 1}};
    // Every layer but the last carries the node representation its successor
    // consumes; the last layer carries the scalar representation its readout
    // consumes.  The rule is stated against the layer's position, so a model
    // with any interaction count is described by the same support matrix.
    for (std::size_t layer = 0;
         layer < data.at("interactions").size(); ++layer) {
        const auto& interaction = data.at("interactions").at(layer);
        const auto& layer_descriptor = result.layer(layer);
        if (layer_descriptor.uvu_channels <= 0
            || !has_external_uvu_tensor_layout(
                interaction, layer_descriptor.uvu_channels))
            return reject_layer(layer, "tied-channel UVU layout");

        const std::vector<std::tuple<int, int, int>> expected_nodes =
            layer == 0
            ? std::vector<std::tuple<int, int, int>>{
                {result.node_channels, 0, 1}}
            : node_hidden;
        if (!blocks_equal(
                Irreps(interaction.at("node_feats_irreps").get<std::string>()),
                expected_nodes))
            return reject_layer(layer, "node representation");

        const std::vector<std::tuple<int, int, int>> expected_edges =
            layer == 0
            ? std::vector<std::tuple<int, int, int>>{
                {layer_descriptor.uvu_channels, 0, 1}}
            : std::vector<std::tuple<int, int, int>>{
                {layer_descriptor.uvu_channels, 0, 1},
                {layer_descriptor.uvu_channels, 1, -1}};
        if (!blocks_equal(
                Irreps(interaction.at("edge_irreps").get<std::string>()),
                expected_edges))
            return reject_layer(layer, "edge representation");

        // Preserve the original one-interaction support matrix: layer zero
        // always carries the node-hidden representation, even when it is also
        // the final layer.
        const auto& expected_hidden = layer != 0 && layer == last && !keep_last
            ? scalar_hidden : node_hidden;
        if (!blocks_equal(
                Irreps(interaction.at("hidden_irreps").get<std::string>()),
                expected_hidden))
            return reject_layer(layer, "hidden representation");
        const auto& skip = interaction.at("skip_tp");
        if (!blocks_equal(
                Irreps(skip.at("irreps_in").get<std::string>()), expected_nodes)
            || !blocks_equal(
                Irreps(skip.at("irreps_out").get<std::string>()),
                expected_hidden))
            return reject_layer(layer, "skip representation");

        if (!blocks_equal(
                Irreps(interaction.at("target_irreps").get<std::string>()), full))
            return reject_layer(layer, "target representation");
        if (!blocks_equal(
                Irreps(interaction.at("gate").at("irreps_in").get<std::string>()),
                gate_input))
            return reject_layer(layer, "gate input representation");
        if (!blocks_equal(
                Irreps(interaction.at("gate").at("irreps_out").get<std::string>()),
                full))
            return reject_layer(layer, "gate output representation");
    }
    for (std::size_t layer = 0;
         layer < data.at("products").size(); ++layer) {
        const std::vector<std::tuple<int, int, int>> product_output =
            layer == last && !keep_last ? scalar_hidden : node_hidden;
        const auto& product = data.at("products").at(layer);
        const auto& contractions =
            product.at("symmetric_contractions").at("contractions");
        if (!product.at("use_sc").get<bool>()
            || !product.value("use_agnostic_product", false)
            || !blocks_equal(
                Irreps(product.at("symmetric_contractions")
                    .at("irreps_in").get<std::string>()), full)
            || !blocks_equal(
                Irreps(product.at("symmetric_contractions")
                    .at("irreps_out").get<std::string>()), product_output)
            || contractions.size() != product_output.size())
            return reject_layer(layer, "product representation");
        if (!blocks_equal(
                Irreps(product.at("node_feats_irreps").get<std::string>()),
                product_output)
            || !blocks_equal(
                Irreps(product.at("target_irreps").get<std::string>()),
                product_output)
            || !blocks_equal(
                Irreps(product.at("linear").at("irreps_in").get<std::string>()),
                product_output)
            || !blocks_equal(
                Irreps(product.at("linear").at("irreps_out").get<std::string>()),
                product_output))
            return reject_layer(layer, "product connectivity");
        for (const auto& contraction : contractions) {
            const auto shape = contraction.at("weights_max").at("shape")
                .get<std::vector<int>>();
            if (contraction.at("correlation").get<int>() != 3
                || shape.empty() || shape.front() != 1)
                return reject(
                    "MH-1 layer " + std::to_string(layer)
                    + " product requires agnostic correlation-three contractions.");
        }
    }
    for (std::size_t layer = 0;
         layer < data.at("readouts").size(); ++layer) {
        const auto& readout = data.at("readouts").at(layer);
        const auto& readout_class = readout.at("class").get<std::string>();
        if (layer != last) {
            if (readout_class != "LinearReadoutBlock")
                return reject_layer(layer, "readout class");
            continue;
        }
        // The last layer's readout consumes the scalar representation it
        // produced; the plain nonlinear form and the bias variant (a second
        // hidden layer between its two linears) are both admitted, and the
        // node program's generated readout serves either.
        if (readout_class != "NonLinearReadoutBlock"
            && readout_class != "NonLinearBiasReadoutBlock")
            return reject_layer(layer, "readout class");
        const std::vector<std::tuple<int, int, int>> product_output =
            keep_last ? node_hidden : scalar_hidden;
        if (!has_scalar_only_energy_readout(readout, product_output))
            return reject_layer(layer, "readout connectivity");
    }
    result.compatible = true;
    return result;
}
