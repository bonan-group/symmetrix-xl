#pragma once

#include <string>
#include <vector>

#include "affine_mlp.hpp"
#include "oam_schema.hpp"
#include "radial_function_set_kokkos.hpp"

class OAMRadialNetwork {
public:
    explicit OAMRadialNetwork(const nlohmann::json& data);

    const std::string& type() const { return type_; }
    double cutoff_p() const { return cutoff_p_; }
    int num_basis() const { return num_basis_; }
    int output_size() const { return mlp_.output_size(); }
    const std::vector<double>& bessel_weights() const { return bessel_weights_; }
    double input_factor() const { return input_factor_; }
    const AffineMLP& mlp() const { return mlp_; }

private:
    std::string type_;
    double cutoff_p_ = 0.0;
    int num_basis_ = 0;
    std::vector<double> bessel_weights_;
    double input_factor_ = 1.0;
    AffineMLP mlp_;
};

struct OAMRadialSplineData {
    double grid_min = 0.0;
    double cutoff = 0.0;
    double spacing = 0.0;
    int num_points = 0;
    int function_count = 0;
    std::vector<std::vector<double>> nodal_values;
    std::vector<std::vector<double>> nodal_derivatives;
    std::vector<double> coefficients;
    std::string table_fingerprint;
};

OAMRadialSplineData build_oam_radial_spline(
    const OAMRadialNetwork& radial,
    int num_points,
    double grid_min,
    double cutoff);

template <typename Precision>
class NequIPOAMRadialSet {
public:
    NequIPOAMRadialSet(
        const NequIPOAMModelSchema& schema,
        int layer_index,
        int num_points,
        double grid_min);

    void evaluate(double radius, int function, Precision& value, Precision& derivative) const;
    int function_count() const noexcept { return function_count_; }
    int interval_count() const noexcept { return radial_.interval_count(); }
    double spline_h() const noexcept { return radial_.spline_h(); }
    double spline_x0() const noexcept { return radial_.spline_x0(); }
    const Precision* coefficient_data() const noexcept { return radial_.coefficient_data(); }
    const RadialFunctionSetKokkos<Precision>& kokkos_set() const noexcept { return radial_; }

private:
    RadialFunctionSetKokkos<Precision> radial_;
    int function_count_ = 0;
};

extern template class NequIPOAMRadialSet<float>;
extern template class NequIPOAMRadialSet<double>;
