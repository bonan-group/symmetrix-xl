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
    const nlohmann::json& parent, const char* key, const char* name)
{
    const auto& value = parent.at(key);
    if (!value.is_object() || !value.contains("shape") || !value.contains("values")
        || !value.at("shape").is_array() || !value.at("values").is_array())
        throw std::invalid_argument(
            std::string("MACE_Nonlinear ") + name + " tensor is malformed.");
    std::size_t size = 1;
    for (const auto& extent_value : value.at("shape")) {
        const int extent = extent_value.get<int>();
        if (extent < 0
            || (extent != 0
                && size > std::numeric_limits<std::size_t>::max()
                    / static_cast<std::size_t>(extent)))
            throw std::invalid_argument(
                std::string("MACE_Nonlinear ") + name + " tensor shape is invalid.");
        size *= static_cast<std::size_t>(extent);
    }
    if (size != value.at("values").size())
        throw std::invalid_argument(
            std::string("MACE_Nonlinear ") + name + " tensor shape is invalid.");
    return value;
}

void require_finite(double value, const char* name)
{
    if (!std::isfinite(value))
        throw std::invalid_argument(std::string("MACE_Nonlinear ") + name + " must be finite.");
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

std::vector<std::tuple<int, int, int>> spherical_blocks(
    int multiplicity, int l_max, int scalar_copies=1)
{
    std::vector<std::tuple<int, int, int>> result;
    result.emplace_back(scalar_copies*multiplicity, 0, 1);
    for (int l=1; l<=l_max; ++l)
        result.emplace_back(multiplicity, l, l%2 == 0 ? 1 : -1);
    return result;
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

MaceMH1FamilyDescriptor analyze_mh1_family_architecture(
    const nlohmann::json& data)
{
    MaceMH1FamilyDescriptor result;
    auto reject = [&](const char* reason) {
        result.rejection_reason = reason;
        return result;
    };
    if (data.at("num_interactions").get<int>() != 2
        || data.at("interactions").size() != 2
        || data.at("products").size() != 2
        || data.at("readouts").size() != 2)
        return reject("MH-1 family execution requires exactly two interactions.");
    result.l_max = data.at("l_max").get<int>();
    if (result.l_max != 2 && result.l_max != 3)
        return reject("MH-1 family execution supports l_max=2 or l_max=3.");
    result.radial_size = static_cast<int>(
        data.at("radial_embedding").at("basis").at("weights").at("values").size());
    if (result.radial_size <= 0)
        return reject("MH-1 family execution requires a non-empty radial basis.");

    const auto& first = data.at("interactions").at(0);
    const auto& second = data.at("interactions").at(1);
    if (first.at("class").get<std::string>()
            != "RealAgnosticResidualNonLinearInteractionBlock"
        || second.at("class").get<std::string>()
            != "RealAgnosticResidualNonLinearInteractionBlock")
        return reject("MH-1 family execution requires nonlinear residual interactions.");

    const Irreps first_nodes(first.at("node_feats_irreps").get<std::string>());
    const Irreps first_edges(first.at("edge_irreps").get<std::string>());
    if (first_nodes.blocks.size() != 1 || first_nodes.blocks[0].l != 0
        || first_nodes.blocks[0].parity != 1)
        return reject("The first MH-1 node representation must be scalar.");
    if (first_edges.blocks.size() != 1 || first_edges.blocks[0].l != 0
        || first_edges.blocks[0].parity != 1)
        return reject("The first MH-1 edge representation must be scalar.");
    result.node_channels = first_nodes.blocks[0].multiplicity;
    result.edge_channels = first_edges.blocks[0].multiplicity;
    const auto full = spherical_blocks(result.node_channels, result.l_max);
    const auto gate_input = spherical_blocks(
        result.node_channels, result.l_max, result.l_max+1);
    const std::vector<std::tuple<int, int, int>> node_hidden = {
        {result.node_channels, 0, 1}, {result.node_channels, 1, -1}};
    const std::vector<std::tuple<int, int, int>> scalar_hidden = {
        {result.node_channels, 0, 1}};
    const std::vector<std::tuple<int, int, int>> second_edges = {
        {result.edge_channels, 0, 1}, {result.edge_channels, 1, -1}};

    if (!blocks_equal(
            Irreps(second.at("node_feats_irreps").get<std::string>()), node_hidden)
        || !blocks_equal(
            Irreps(second.at("edge_irreps").get<std::string>()), second_edges)
        || !blocks_equal(
            Irreps(first.at("hidden_irreps").get<std::string>()), node_hidden)
        || !blocks_equal(
            Irreps(second.at("hidden_irreps").get<std::string>()), scalar_hidden))
        return reject("MH-1 node, edge, and hidden irreps are inconsistent.");
    for (const auto* interaction : {&first, &second}) {
        if (!blocks_equal(
                Irreps(interaction->at("target_irreps").get<std::string>()), full)
            || !blocks_equal(
                Irreps(interaction->at("gate").at("irreps_in").get<std::string>()),
                gate_input)
            || !blocks_equal(
                Irreps(interaction->at("gate").at("irreps_out").get<std::string>()),
                full))
            return reject("MH-1 target or gate irreps are inconsistent.");
    }
    const std::vector<std::vector<std::tuple<int, int, int>>> product_outputs = {
        node_hidden, scalar_hidden};
    for (int layer=0; layer<2; ++layer) {
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
                    .at("irreps_out").get<std::string>()), product_outputs[layer])
            || contractions.size() != product_outputs[layer].size())
            return reject("MH-1 family execution requires agnostic products with skip connections.");
        for (const auto& contraction : contractions) {
            const auto shape = contraction.at("weights_max").at("shape")
                .get<std::vector<int>>();
            if (contraction.at("correlation").get<int>() != 3
                || shape.empty() || shape.front() != 1)
                return reject("MH-1 family execution requires agnostic correlation-three products.");
        }
    }
    if (!has_external_uvu_tensor_layout(first, result.edge_channels)
        || !has_external_uvu_tensor_layout(second, result.edge_channels))
        return reject("MH-1 family execution requires external weighted uvu tensor products.");
    if (data.at("readouts").at(0).at("class").get<std::string>()
            != "LinearReadoutBlock"
        || data.at("readouts").at(1).at("class").get<std::string>()
            != "NonLinearReadoutBlock")
        return reject("MH-1 family execution requires linear then nonlinear readouts.");
    result.compatible = true;
    return result;
}

void validate_mace_nonlinear_schema(const nlohmann::json& data)
{
    if (!data.is_object()
        || data.value("model_type", std::string()) != "MACE_Nonlinear"
        || data.at("symmetrix_format_version").get<int>() != 3)
        throw std::invalid_argument(
            "MACE_Nonlinear evaluator requires format-version-3 nonlinear JSON.");

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
        throw std::invalid_argument("MACE_Nonlinear atomic-number metadata is inconsistent.");
    if (std::any_of(atomic_numbers.begin(), atomic_numbers.end(), [](int value) { return value <= 0; })
        || std::any_of(model_atomic_numbers.begin(), model_atomic_numbers.end(), [](int value) { return value <= 0; }))
        throw std::invalid_argument("MACE_Nonlinear atomic numbers must be positive.");
    for (int local_type=0; local_type<static_cast<int>(atomic_numbers.size()); ++local_type) {
        const int model_type = model_indices[local_type];
        if (model_type < 0 || model_type >= static_cast<int>(model_atomic_numbers.size())
            || model_atomic_numbers[model_type] != atomic_numbers[local_type])
            throw std::invalid_argument("MACE_Nonlinear model_indices are inconsistent.");
    }

    const double r_cut = data.at("r_cut").get<double>();
    const int l_max = data.at("l_max").get<int>();
    if (!(r_cut > 0.0) || !std::isfinite(r_cut) || l_max < 0
        || l_max >= static_cast<int>(std::sqrt(std::numeric_limits<int>::max()))-1)
        throw std::invalid_argument("MACE_Nonlinear cutoff or l_max is invalid.");

    const int num_interactions = data.at("num_interactions").get<int>();
    if (num_interactions <= 0 || !data.at("interactions").is_array()
        || !data.at("products").is_array() || !data.at("readouts").is_array()
        || data.at("interactions").size() != static_cast<std::size_t>(num_interactions)
        || data.at("products").size() != static_cast<std::size_t>(num_interactions)
        || data.at("readouts").size() != static_cast<std::size_t>(num_interactions))
        throw std::invalid_argument("MACE_Nonlinear layer counts are inconsistent.");

    const auto& radial = data.at("radial_embedding");
    const int num_spline_points = radial.value("num_spline_points", 256);
    if (num_spline_points < 4)
        throw std::invalid_argument(
            "MACE_Nonlinear radial num_spline_points must be at least 4.");
    const auto& basis = radial.at("basis");
    const auto& cutoff = radial.at("cutoff");
    if (basis.at("type").get<std::string>() != "bessel"
        || cutoff.at("type").get<std::string>() != "polynomial"
        || cutoff.at("p").get<int>() < 1
        || cutoff.at("r_max").get<double>() != r_cut)
        throw std::invalid_argument("MACE_Nonlinear radial basis or cutoff is unsupported.");
    const auto& basis_weights = tensor(basis, "weights", "radial basis weights");
    if (basis_weights.at("shape").size() != 1 || basis_weights.at("values").empty())
        throw std::invalid_argument("MACE_Nonlinear radial basis weights are invalid.");
    require_finite(basis.at("prefactor").get<double>(), "radial prefactor");

    const auto& transform = radial.at("distance_transform");
    const auto transform_type = transform.at("type").get<std::string>();
    if (transform_type != "none" && transform_type != "agnesi")
        throw std::invalid_argument("MACE_Nonlinear distance transform is unsupported.");
    if (transform_type == "agnesi") {
        require_finite(transform.at("a").get<double>(), "Agnesi a");
        require_finite(transform.at("q").get<double>(), "Agnesi q");
        require_finite(transform.at("p").get<double>(), "Agnesi p");
        const auto radii = transform.at("covalent_radii").get<std::vector<double>>();
        const int maximum_atomic_number =
            *std::max_element(model_atomic_numbers.begin(), model_atomic_numbers.end());
        if (maximum_atomic_number < 0
            || radii.size() <= static_cast<std::size_t>(maximum_atomic_number))
            throw std::invalid_argument("MACE_Nonlinear Agnesi covalent radii are incomplete.");
        for (int atomic_number : model_atomic_numbers)
            if (!(radii[atomic_number] > 0.0) || !std::isfinite(radii[atomic_number]))
                throw std::invalid_argument("MACE_Nonlinear Agnesi covalent radii are invalid.");
    }

    const auto& scale = tensor(data.at("scale_shift"), "scale", "scale");
    const auto& shift = tensor(data.at("scale_shift"), "shift", "shift");
    if (!scale.at("shape").empty() || !shift.at("shape").empty())
        throw std::invalid_argument("MACE_Nonlinear scale and shift must be scalar tensors.");
    require_finite(scale.at("values").at(0).get<double>(), "scale");
    require_finite(shift.at("values").at(0).get<double>(), "shift");

    const auto& atomic_energies = tensor(data, "atomic_energies", "atomic energies");
    const auto energy_shape = atomic_energies.at("shape").get<std::vector<int>>();
    if (energy_shape != std::vector<int>{1, static_cast<int>(model_atomic_numbers.size())})
        throw std::invalid_argument("MACE_Nonlinear atomic-energy tensor has an invalid shape.");
}
