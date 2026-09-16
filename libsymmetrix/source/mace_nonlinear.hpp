#pragma once

#include <span>
#include <string>
#include <vector>

#include "affine_mlp.hpp"
#include "e3nn.hpp"
#include "e3nn_product.hpp"
#include "mace_nonlinear_schema.hpp"
#include "mace_streamed_edges.hpp"
#include "zbl.hpp"

class MaceNonlinear {
public:
    explicit MaceNonlinear(
        const std::string& filename, const std::string& requested_head = {});

    std::vector<int> atomic_numbers;
    std::vector<double> atomic_energies;
    std::string selected_head;
    std::vector<std::string> available_heads;
    std::vector<double> node_energies;
    std::vector<double> node_forces;
    double r_cut = 0.0;
    bool has_field_coupling = false;
    bool is_mh1_family() const { return mh1_family.compatible; }
    int mh1_node_channels() const { return mh1_family.node_channels; }
    int mh1_edge_channels() const { return mh1_family.edge_channels; }
    int mh1_radial_size() const { return mh1_family.radial_size; }
    int mh1_l_max() const { return mh1_family.l_max; }
    const std::string& mh1_family_rejection_reason() const {
        return mh1_family.rejection_reason;
    }
    bool mh1_uses_compiled_products() const { return mh1_compiled_products; }
    bool mh1_uses_pair_conditioning() const { return mh1_pair_conditioning; }
    const std::string& mh1_fast_path_rejection_reason() const {
        return mh1_fast_path_rejection;
    }
    bool uses_mh1_fast_path() const { return mh1_fast_path; }
    bool supports_streamed_edges() const { return mh1_fast_path; }
    std::string streamed_edges_mode() const;
    void set_streamed_edges(std::string mode);
    int edge_workspace_rows() const { return last_edge_workspace_rows; }

    void compute_node_energies_forces(
        int num_nodes,
        std::span<const int> node_types,
        std::span<const int> num_neigh,
        std::span<const int> neigh_indices,
        std::span<const int> neigh_types,
        std::span<const double> xyz,
        std::span<const double> r);

private:
    explicit MaceNonlinear(const nlohmann::json& data);

    struct Gate {
        Irreps input;
        Irreps output;
        Irreps scalars;
        Irreps gates;
        Irreps gated;
        std::vector<double> scalar_constants;
        std::vector<double> gate_constants;
        explicit Gate(const nlohmann::json& data);
        std::vector<double> evaluate(const std::vector<double>& values) const;
        std::vector<double> reverse(
            const std::vector<double>& values,
            const std::vector<double>& output_adjoint) const;
    };
    struct Interaction {
        E3Linear source_embedding;
        E3Linear target_embedding;
        E3Linear linear_up;
        E3Linear skip;
        E3Linear linear_res;
        E3Linear linear_1;
        E3Linear linear_2;
        E3TensorProduct convolution;
        AffineMLP convolution_weights;
        AffineMLP density;
        std::vector<std::vector<double>> convolution_source_contributions;
        std::vector<std::vector<double>> convolution_target_contributions;
        std::vector<std::vector<double>> density_source_contributions;
        std::vector<std::vector<double>> density_target_contributions;
        int active_type_count = 0;
        Gate gate;
        double alpha;
        double beta;
        explicit Interaction(const nlohmann::json& data);
        void prepare_pair_conditioning(
            int radial_size,
            int model_element_count,
            const std::vector<int>& selected_model_indices);
        bool supports_pair_conditioning(
            int radial_size,
            int model_element_count) const;
        void validate_conditioned_type(int type) const;
    };
    struct Readout {
        bool nonlinear;
        E3Linear linear;
        std::unique_ptr<E3Linear> linear_1;
        std::unique_ptr<E3Linear> linear_2;
        double activation_constant;
        explicit Readout(const nlohmann::json& data);
        double evaluate(const std::vector<double>& features) const;
        std::vector<double> reverse(const std::vector<double>& features, double output_adjoint) const;
    };

    std::vector<double> one_hot(int local_type) const;
    std::vector<double> radial_features(double distance, int source_type, int target_type) const;
    std::vector<double> radial_feature_derivatives(double distance, int source_type, int target_type) const;
    double cutoff(double distance) const;
    double cutoff_derivative(double distance) const;
    void compute_spherical_harmonics(std::span<const double> xyz);
    std::vector<double> node_slice(const std::vector<double>& values, int node, int width) const;
    void add_node_slice(std::vector<double>& values, int node, const std::vector<double>& addend) const;

    int l_max = 0;
    int num_lm = 0;
    int model_num_elements = 0;
    int cutoff_power = 0;
    bool apply_cutoff = false;
    bool has_agnesi = false;
    double radial_prefactor = 0.0;
    double agnesi_a = 0.0;
    double agnesi_q = 0.0;
    double agnesi_p = 0.0;
    double scale = 1.0;
    double shift = 0.0;
    std::vector<int> model_indices;
    std::vector<int> model_atomic_numbers;
    std::vector<double> bessel_weights;
    std::vector<double> covalent_radii;
    E3Linear node_embedding;
    std::vector<Interaction> interactions;
    std::vector<E3ProductBasis> products;
    std::vector<bool> product_agnostic;
    std::vector<Readout> readouts;
    E3LinearBatchWorkspace linear_batch_workspace;
    E3ProductBasisBatchWorkspace product_batch_workspace;
    AffineMLPBatchWorkspace affine_batch_workspace;
    bool mh1_fast_path = false;
    bool mh1_compiled_products = false;
    bool mh1_pair_conditioning = false;
    MaceMH1FamilyDescriptor mh1_family;
    std::string mh1_fast_path_rejection;
    int last_edge_workspace_rows = 0;
    MACEStreamedEdgesMode streamed_edges = MACEStreamedEdgesMode::materialized;
    static constexpr int streamed_edge_block_size = 1024;
    bool streams_layer(int layer) const;
    bool has_zbl = false;
    ZBL zbl;
    std::vector<double> spherical_harmonics;
    std::vector<double> spherical_harmonic_gradients;
};
