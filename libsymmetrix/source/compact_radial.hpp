#pragma once

#include <memory>
#include <string>
#include <vector>

#include "multilayer_perceptron.hpp"

struct RadialSplineData {
    std::vector<std::vector<double>> values;
    std::vector<std::vector<double>> derivatives;
};

enum class CompactRadialProjectionLayout {
    row_major_output_by_embedding,
};

struct FactorizedRadialData {
    RadialSplineData penultimate;
    std::vector<double> final_projection;
    int embedding_width = 0;
    int output_width = 0;
    CompactRadialProjectionLayout projection_layout =
        CompactRadialProjectionLayout::row_major_output_by_embedding;

    RadialSplineData reconstruct_projected() const;
};

struct CompactRadialFactorizedPairTables {
    FactorizedRadialData R0;
    FactorizedRadialData R1;
};

struct CompactRadialFactorizationReport {
    int embedding_width;
    int output_width;
    double max_value_error;
    double max_derivative_error;
};

struct CompactRadialNetworkGradients {
    std::vector<std::vector<double>> weights;
};

struct CompactRadialPairTables {
    RadialSplineData R0;
    RadialSplineData R1;
    RadialSplineData A0;
    RadialSplineData A1;
};

class CompactRadialModel {
public:
    struct Network {
        std::vector<int> shape;
        std::vector<std::vector<double>> weights;
        double activation_scale;
        bool tanh_square;
        std::unique_ptr<MultilayerPerceptron> mlp;
    };

    CompactRadialModel(
        std::string definition_json,
        std::vector<int> atomic_numbers,
        double r_cut);

    CompactRadialPairTables materialize_pair(int type_i, int type_j);
    CompactRadialPairTables materialize_projected_pair(int type_i, int type_j);
    CompactRadialFactorizedPairTables materialize_factorized_pair(
        int type_i,
        int type_j);
    CompactRadialFactorizationReport factorization_report(
        const std::string& network_name,
        int type_i,
        int type_j);
    std::vector<double> transpose_spline_coefficients(
        const std::vector<double>& coefficient_adjoints,
        int pair_count,
        int function_count);
    CompactRadialNetworkGradients backpropagate_network_nodal_values(
        const std::string& network_name,
        const std::vector<std::pair<int,int>>& pair_types,
        const std::vector<double>& nodal_value_adjoints,
        bool factorized,
        const std::vector<double>& final_projection_adjoints = {});
    std::vector<std::pair<int,int>> network_weight_shapes(
        const std::string& network_name);

    double spline_h() const;
    double spline_min() const;
    int num_spline_points() const;
    bool has_A0() const;
    bool has_A1() const;
    bool has_R1() const;

private:
    double grid_min;
    int spline_points;
    double cutoff_r_max;
    int cutoff_p;
    double h;
    std::vector<int> atomic_numbers;
    std::vector<double> bessel_weights;
    double bessel_prefactor;

    bool use_agnesi;
    double agnesi_a;
    double agnesi_q;
    double agnesi_p;
    std::vector<double> covalent_radii;

    Network R0_network;
    std::unique_ptr<Network> R1_network;
    std::unique_ptr<Network> A0_network;
    std::unique_ptr<Network> A1_network;
    std::vector<double> spline_derivative_operator;

    std::vector<double> radial_features(int type_i, int type_j) const;
    RadialSplineData evaluate_network(Network& network, const std::vector<double>& features);
    FactorizedRadialData evaluate_factorized_network(
        Network& network,
        const std::vector<double>& features);
    Network& radial_network(const std::string& name);
    std::vector<double> spline_derivatives(const std::vector<double>& values) const;
};
