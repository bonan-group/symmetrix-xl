#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "e3nn.hpp"
#include "nlohmann/json.hpp"

struct OAMRadialSplineContract {
    double grid_min = 0.0;
    double cutoff = 0.0;
    int num_points = 0;
    double spacing = 0.0;
    int function_count = 0;
    int edge_type_count = 1;
    std::string model_fingerprint;
    std::string table_fingerprint;
    std::string precision;
    std::string exporter_version;
    nlohmann::json certification = nlohmann::json::object();
    bool certified = false;
};

struct NequIPOAMLayer {
    std::string node_irreps_in;
    std::string edge_irreps;
    std::string tp_output_irreps;
    std::string node_irreps_pre_gate;
    std::string node_irreps_out;
    nlohmann::json pre_linear;
    nlohmann::json radial_basis;
    nlohmann::json edge_tensor_product;
    nlohmann::json post_linear;
    nlohmann::json self_connection;
    nlohmann::json normalization;
    nlohmann::json gate;
    explicit NequIPOAMLayer(const nlohmann::json& data)
        : node_irreps_in(data.at("node_irreps_in").get<std::string>()),
          edge_irreps(data.at("edge_irreps").get<std::string>()),
          tp_output_irreps(data.at("tp_output_irreps").get<std::string>()),
          node_irreps_pre_gate(data.at("node_irreps_pre_gate").get<std::string>()),
          node_irreps_out(data.at("node_irreps_out").get<std::string>()),
          pre_linear(data.at("pre_linear")), radial_basis(data.at("radial_basis")),
          edge_tensor_product(data.at("edge_tensor_product")), post_linear(data.at("post_linear")),
          self_connection(data.at("self_connection")), normalization(data.at("normalization")),
          gate(data.at("gate")) {}
};

struct NequIPOAMModelSchema {
    std::vector<int> atomic_numbers;
    std::vector<int> model_atomic_numbers;
    std::vector<int> atomic_number_to_model_type;
    double r_cut = 0.0;
    nlohmann::json edge_embedding;
    nlohmann::json node_attributes;
    std::vector<NequIPOAMLayer> layers;
    nlohmann::json readout;
    std::vector<double> per_type_shift;
    std::vector<double> per_type_scale;
    nlohmann::json zbl;
    std::vector<OAMRadialSplineContract> kokkos_radial;
    std::string exporter_fingerprint;
    std::string contract_fingerprint_value;
    std::string generated_architecture_fingerprint_value;
    std::string contract_fingerprint() const { return contract_fingerprint_value; }
    std::string generated_architecture_fingerprint() const
    {
        return generated_architecture_fingerprint_value;
    }
};

NequIPOAMModelSchema load_nequip_oam_schema(const std::string& filename);

// The validated, immutable owner of one version-2 NequIP-OAM contract.
//
// A schema is only executable once its cross-field invariants hold: the
// atomic-number to model-type mapping is complete, the node-attribute tensor
// matches the first layer's input, the layer dimensions form a chain, the
// readout consumes the final layer's output, and the edge-embedding and ZBL
// metadata are consistent.  This type owns the schema and checks exactly those
// invariants at construction, so an execution owner can depend on the metadata
// without depending on any evaluator.
class NequIPOAMModel {
public:
    explicit NequIPOAMModel(const std::string& filename);

    const NequIPOAMModelSchema& schema() const { return schema_; }
    const std::vector<int>& atomic_numbers() const { return schema_.atomic_numbers; }
    double r_cut() const { return schema_.r_cut; }
    std::size_t num_layers() const { return schema_.layers.size(); }
    const std::string& contract_fingerprint() const
    {
        return schema_.contract_fingerprint_value;
    }

private:
    NequIPOAMModelSchema schema_;
};
