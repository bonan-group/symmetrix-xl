#include <algorithm>
#include <iostream> //TODO
#include <fstream>
#include <cmath>
#include <numbers>
#include <numeric>
#include <limits>
#include <stdexcept>

#include "nlohmann/json.hpp"
#include "sphericart.hpp"

#include "cblas.hpp"
#include "mace.hpp"
#include "prediction_heads.hpp"

namespace {

template <typename Precision>
Precision checked_precision_cast(const double value)
{
    if (!std::isfinite(value)
        || std::abs(value) > static_cast<double>(std::numeric_limits<Precision>::max()))
        throw std::invalid_argument("MACE model contains a value outside the requested precision range.");
    return static_cast<Precision>(value);
}

template <typename Precision, typename Input>
struct PrecisionConverter;

template <typename Precision>
struct PrecisionConverter<Precision, double> {
    using Output = Precision;

    static Output convert(const double value)
    {
        return checked_precision_cast<Precision>(value);
    }
};

template <typename Precision, typename Value>
struct PrecisionConverter<Precision, std::vector<Value>> {
    using ConvertedValue = typename PrecisionConverter<Precision, Value>::Output;
    using Output = std::vector<ConvertedValue>;

    static Output convert(const std::vector<Value>& values)
    {
        auto result = Output();
        result.reserve(values.size());
        for (const auto& value : values)
            result.push_back(PrecisionConverter<Precision, Value>::convert(value));
        return result;
    }
};

template <typename Precision, typename Key, typename Value>
struct PrecisionConverter<Precision, std::map<Key, Value>> {
    using ConvertedValue = typename PrecisionConverter<Precision, Value>::Output;
    using Output = std::map<Key, ConvertedValue>;

    static Output convert(const std::map<Key, Value>& values)
    {
        auto result = Output();
        for (const auto& [key, value] : values)
            result.emplace(key, PrecisionConverter<Precision, Value>::convert(value));
        return result;
    }
};

template <typename Precision, typename Input>
auto checked_precision_data(const Input& values)
    -> typename PrecisionConverter<Precision, Input>::Output
{
    return PrecisionConverter<Precision, Input>::convert(values);
}

}

template <typename Precision>
MACECPU<Precision>::MACECPU(std::string filename, std::string requested_head)
{
    load_from_json(filename, requested_head);
    if (supports_streamed_edges())
        streamed_edges = MACEStreamedEdgesMode::direct;
}

template <typename Precision>
bool MACECPU<Precision>::supports_streamed_edges() const
{
    return uses_compact_radial;
}

template <typename Precision>
std::string MACECPU<Precision>::streamed_edges_mode() const
{
    return mace_streamed_edges_mode_name(streamed_edges);
}

template <typename Precision>
void MACECPU<Precision>::set_streamed_edges(std::string mode)
{
    const auto requested = parse_mace_streamed_edges_mode(mode);
    if (mace_uses_prepared_execution(requested))
        throw std::invalid_argument(
            "direct execution requires the Kokkos evaluator.");
    if (requested != MACEStreamedEdgesMode::materialized && !supports_streamed_edges())
        throw std::invalid_argument(
            "Streamed edges require a format-v2 compact MACE or MACEField model.");
    streamed_edges = requested;
    if (streamed_edges == MACEStreamedEdgesMode::generic) {
        std::vector<Precision>().swap(R1);
        std::vector<Precision>().swap(R1_deriv);
        std::vector<Precision>().swap(R0);
        std::vector<Precision>().swap(R0_deriv);
    }
}

template <typename Precision>
void MACECPU<Precision>::prepare_active_types(std::span<const int> node_types)
{
    if (!uses_compact_radial)
        return;

    auto requested_types = std::vector<int>(node_types.begin(), node_types.end());
    std::sort(requested_types.begin(), requested_types.end());
    requested_types.erase(
        std::unique(requested_types.begin(), requested_types.end()),
        requested_types.end());
    if (requested_types.empty())
        throw std::invalid_argument("MACE compact radial cache requires at least one active type.");
    for (const int type : requested_types)
        if (type < 0 || type >= atomic_numbers.size())
            throw std::out_of_range("MACE active type index is out of range.");
    if (requested_types == active_types)
        return;

    auto new_spl_set_0 = std::vector<std::unique_ptr<CubicSplineSetT<Precision>>>();
    auto new_spl_set_1 = std::vector<std::unique_ptr<CubicSplineSetT<Precision>>>();
    auto new_A0_splines = std::vector<CubicSplineT<Precision>>();
    auto new_A1_splines = std::vector<CubicSplineT<Precision>>();
    const int pair_count = requested_types.size()*(requested_types.size()+1)/2;
    new_spl_set_0.reserve(pair_count);
    new_spl_set_1.reserve(pair_count);
    if (A0_scaled)
        new_A0_splines.reserve(pair_count);
    if (A1_scaled)
        new_A1_splines.reserve(pair_count);

    const Precision h = checked_precision_cast<Precision>(
        compact_radial_model->spline_h());
    const Precision x0 = checked_precision_cast<Precision>(
        compact_radial_model->spline_min());
    for (int local_i=0; local_i<requested_types.size(); ++local_i) {
        for (int local_j=local_i; local_j<requested_types.size(); ++local_j) {
            auto tables = compact_radial_model->materialize_pair(
                requested_types[local_i], requested_types[local_j]);
            const auto expected_R0 = static_cast<std::size_t>((l_max+1)*num_channels);
            const auto expected_R1 = static_cast<std::size_t>(Phi1_l.size()*num_channels);
            if (tables.R0.values.size() != expected_R0
                || tables.R0.derivatives.size() != expected_R0)
                throw std::runtime_error("Compact radial R0 output has an invalid size.");
            if (!single_layer_readout
                && (tables.R1.values.size() != expected_R1
                    || tables.R1.derivatives.size() != expected_R1))
                throw std::runtime_error("Compact radial R1 output has an invalid size.");
            new_spl_set_0.push_back(std::make_unique<CubicSplineSetT<Precision>>(
                h,
                checked_precision_data<Precision>(tables.R0.values),
                checked_precision_data<Precision>(tables.R0.derivatives),
                x0));
            if (!single_layer_readout)
                new_spl_set_1.push_back(std::make_unique<CubicSplineSetT<Precision>>(
                    h,
                    checked_precision_data<Precision>(tables.R1.values),
                    checked_precision_data<Precision>(tables.R1.derivatives),
                    x0));
            if (A0_scaled) {
                if (tables.A0.values.size() != 1 || tables.A0.derivatives.size() != 1)
                    throw std::runtime_error("Compact radial A0 network must have one output.");
                new_A0_splines.emplace_back(
                    h,
                    checked_precision_data<Precision>(tables.A0.values[0]),
                    checked_precision_data<Precision>(tables.A0.derivatives[0]),
                    x0);
            }
            if (A1_scaled) {
                if (tables.A1.values.size() != 1 || tables.A1.derivatives.size() != 1)
                    throw std::runtime_error("Compact radial A1 network must have one output.");
                new_A1_splines.emplace_back(
                    h,
                    checked_precision_data<Precision>(tables.A1.values[0]),
                    checked_precision_data<Precision>(tables.A1.derivatives[0]),
                    x0);
            }
        }
    }

    auto new_type_to_active = std::vector<int>(atomic_numbers.size(), -1);
    auto new_active_atomic_numbers = std::vector<int>();
    new_active_atomic_numbers.reserve(requested_types.size());
    for (int active=0; active<requested_types.size(); ++active) {
        new_type_to_active[requested_types[active]] = active;
        new_active_atomic_numbers.push_back(atomic_numbers[requested_types[active]]);
    }

    spl_set_0 = std::move(new_spl_set_0);
    spl_set_1 = std::move(new_spl_set_1);
    A0_splines = std::move(new_A0_splines);
    A1_splines = std::move(new_A1_splines);
    type_to_active = std::move(new_type_to_active);
    active_atomic_numbers = std::move(new_active_atomic_numbers);
    active_types = std::move(requested_types);
}

template <typename Precision>
int MACECPU<Precision>::radial_pair_index(int type_i, int type_j) const
{
    if (type_i < 0 || type_i >= type_to_active.size()
        || type_j < 0 || type_j >= type_to_active.size())
        throw std::out_of_range("MACE radial type index is out of range.");
    const int active_i = type_to_active[type_i];
    const int active_j = type_to_active[type_j];
    if (active_i < 0 || active_j < 0)
        throw std::runtime_error("MACE radial cache is not prepared for an active type.");
    const int num_active = active_types.size();
    return (active_i <= active_j)
        ? active_i*(2*num_active-active_i-1)/2+active_j
        : active_j*(2*num_active-active_j-1)/2+active_i;
}

template <typename Precision>
void MACECPU<Precision>::compute_node_energies_forces(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r)
{
    // TODO: best to resize these within individual routines
    node_energies.resize(num_nodes);
    std::fill(node_energies.begin(), node_energies.end(), 0.0);
    node_forces.resize(xyz.size());
    std::fill(node_forces.begin(), node_forces.end(), 0.0);

    if (has_zbl)
        zbl.compute_ZBL(
            num_nodes, node_types, num_neigh, neigh_types,
            atomic_numbers, r, xyz, node_energies, node_forces);

    compute_Y(xyz);

    if (streamed_edges == MACEStreamedEdgesMode::generic) {
        compute_A0_streamed(num_nodes, node_types, num_neigh, neigh_types, r);
    } else {
        compute_R0(num_nodes, node_types, num_neigh, neigh_types, r);
        compute_A0(num_nodes, node_types, num_neigh, neigh_types);
    }
    compute_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_M0(num_nodes, node_types);
    compute_H1(num_nodes);
    add_H1_first_residual(num_nodes, node_types, true);

    if (single_layer_readout) {
        std::fill(H1_adj.begin(), H1_adj.end(), Precision(0));
        compute_M1(num_nodes, node_types);
        compute_H2(num_nodes, node_types);
        compute_readouts(num_nodes, node_types);
        std::fill(H1_adj.begin(), H1_adj.end(), Precision(0));
        reverse_H2(num_nodes, node_types, false);
        reverse_M1(num_nodes, node_types);
        reverse_H1(num_nodes);
        reverse_M0(num_nodes, node_types);
        reverse_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
        if (streamed_edges == MACEStreamedEdgesMode::generic)
            reverse_A0_streamed(
                num_nodes, node_types, num_neigh, neigh_types, xyz, r);
        else
            reverse_A0(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
        return;
    }

    if (streamed_edges == MACEStreamedEdgesMode::materialized) {
        compute_R1(num_nodes, node_types, num_neigh, neigh_types, r);
        compute_Phi1(num_nodes, num_neigh, neigh_indices);
    } else {
        compute_Phi1_streamed(
            num_nodes, node_types, num_neigh, neigh_indices, neigh_types, r);
    }
    compute_A1(num_nodes);
    compute_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_M1(num_nodes, node_types);
    compute_H2(num_nodes, node_types);

    compute_readouts(num_nodes, node_types);

    reverse_H2(num_nodes, node_types, false);
    reverse_M1(num_nodes, node_types);
    reverse_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r, false);
    reverse_A1(num_nodes);
    if (streamed_edges == MACEStreamedEdgesMode::materialized) {
        reverse_Phi1(num_nodes, num_neigh, neigh_indices, xyz, r, false, false);
    } else {
        reverse_Phi1_streamed(
            num_nodes, node_types, num_neigh, neigh_indices, neigh_types,
            xyz, r, false, false);
    }

    reverse_H1(num_nodes);
    reverse_M0(num_nodes, node_types);
    reverse_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
    if (streamed_edges == MACEStreamedEdgesMode::generic) {
        reverse_A0_streamed(
            num_nodes, node_types, num_neigh, neigh_types, xyz, r);
    } else {
        reverse_A0(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
    }
}

template <typename Precision>
void MACECPU<Precision>::compute_node_energies_forces_field(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r,
    std::span<const double> electric_field)
{
    node_energies.resize(num_nodes);
    std::fill(node_energies.begin(), node_energies.end(), 0.0);
    node_forces.resize(xyz.size());
    std::fill(node_forces.begin(), node_forces.end(), 0.0);

    if (has_zbl)
        zbl.compute_ZBL(
            num_nodes, node_types, num_neigh, neigh_types,
            atomic_numbers, r, xyz, node_energies, node_forces);

    compute_Y(xyz);

    if (streamed_edges == MACEStreamedEdgesMode::generic) {
        compute_A0_streamed(num_nodes, node_types, num_neigh, neigh_types, r);
    } else {
        compute_R0(num_nodes, node_types, num_neigh, neigh_types, r);
        compute_A0(num_nodes, node_types, num_neigh, neigh_types);
    }
    compute_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_M0(num_nodes, node_types);
    compute_H1_product(num_nodes);
    add_H1_first_residual(num_nodes, node_types, false);
    compute_field_H1(num_nodes, electric_field);
    compute_H1_linear_up(num_nodes);

    if (streamed_edges == MACEStreamedEdgesMode::materialized) {
        compute_R1(num_nodes, node_types, num_neigh, neigh_types, r);
        compute_Phi1(num_nodes, num_neigh, neigh_indices);
    } else {
        compute_Phi1_streamed(
            num_nodes, node_types, num_neigh, neigh_indices, neigh_types, r);
    }
    compute_A1(num_nodes);
    compute_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_M1(num_nodes, node_types);
    compute_H2(num_nodes, node_types);

    compute_readouts(num_nodes, node_types);

    reverse_H2(num_nodes, node_types, false);
    reverse_M1(num_nodes, node_types);
    reverse_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r, false);
    reverse_A1(num_nodes);
    if (streamed_edges == MACEStreamedEdgesMode::materialized) {
        reverse_Phi1(num_nodes, num_neigh, neigh_indices, xyz, r, false, false);
    } else {
        reverse_Phi1_streamed(
            num_nodes, node_types, num_neigh, neigh_indices, neigh_types,
            xyz, r, false, false);
    }
    reverse_H1_linear_up(num_nodes);
    reverse_field_H1(num_nodes, electric_field);

    reverse_H1_product(num_nodes);
    reverse_M0(num_nodes, node_types);
    reverse_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
    if (streamed_edges == MACEStreamedEdgesMode::generic) {
        reverse_A0_streamed(
            num_nodes, node_types, num_neigh, neigh_types, xyz, r);
    } else {
        reverse_A0(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
    }
}

template <typename Precision>
void MACECPU<Precision>::compute_electric_field_hessian(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r,
    std::span<const double> electric_field)
{
    if (not has_field_coupling)
        throw std::invalid_argument("MACECPU<Precision>::compute_electric_field_hessian requires field coupling.");
    if (electric_field.size() != 3)
        throw std::invalid_argument("MACECPU<Precision>::compute_electric_field_hessian requires a graph-level electric field.");

    const Precision field[3] = {
        static_cast<Precision>(electric_field[0]),
        static_cast<Precision>(electric_field[1]),
        static_cast<Precision>(electric_field[2]),
    };

    node_energies.resize(num_nodes);
    std::fill(node_energies.begin(), node_energies.end(), 0.0);
    node_forces.resize(xyz.size());
    std::fill(node_forces.begin(), node_forces.end(), 0.0);

    compute_Y(xyz);
    compute_R0(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_A0(num_nodes, node_types, num_neigh, neigh_types);
    compute_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_M0(num_nodes, node_types);
    compute_H1_product(num_nodes);
    add_H1_first_residual(num_nodes, node_types, false);
    compute_field_H1(num_nodes, electric_field);
    compute_H1_linear_up(num_nodes);
    compute_R1(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_Phi1(num_nodes, num_neigh, neigh_indices);
    compute_A1(num_nodes);
    compute_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_M1(num_nodes, node_types);
    compute_H2(num_nodes, node_types);
    compute_readouts(num_nodes, node_types);

    electric_field_hessian.assign(9, 0.0);
    electric_field_force_derivative.assign(3*xyz.size(), 0.0);

    const auto h1_index = [this](int i, int lm, int k) {
        return (i*num_LM + lm)*num_channels + k;
    };

    auto A1_scale_factors = std::vector<Precision>(num_nodes, 1.0);
    if (A1_scaled) {
        int ij = 0;
        for (int i=0; i<num_nodes; ++i) {
            const int type_i = node_types[i];
            for (int j=0; j<num_neigh[i]; ++j) {
                const int type_j = neigh_types[ij];
                const int type_ij = radial_pair_index(type_i, type_j);
                A1_scale_factors[i] += A1_splines[type_ij].evaluate(r[ij]);
                ij += 1;
            }
        }
    }

    auto A0_scale_factors = std::vector<Precision>(num_nodes, 1.0);
    if (A0_scaled) {
        int ij = 0;
        for (int i=0; i<num_nodes; ++i) {
            const int type_i = node_types[i];
            for (int j=0; j<num_neigh[i]; ++j) {
                const int type_j = neigh_types[ij];
                const int type_ij = radial_pair_index(type_i, type_j);
                A0_scale_factors[i] += A0_splines[type_ij].evaluate(r[ij]);
                ij += 1;
            }
        }
    }

    int num_lme_local = 0;
    std::vector<int> num_e(l_max+1,0);
    for (auto l : Phi1_l) {
        num_lme_local += 2*l+1;
        num_e[l] += 1;
    }

    for (int seed=0; seed<3; ++seed) {
        auto H1_dot = std::vector<Precision>(H1.size(), 0.0);
        for (int i=0; i<num_nodes; ++i) {
            for (const auto& path : field_coupling_paths) {
                for (const auto& angular : path.angular_entries) {
                    if (angular.field_component != seed)
                        continue;
                    const Precision coefficient =
                        static_cast<Precision>(angular.coefficient);
                    for (int input=0; input<num_channels; ++input) {
                        const Precision input_value =
                            H1_pre_field[h1_index(i, angular.input_lm, input)];
                        for (int output=0; output<num_channels; ++output) {
                            H1_dot[h1_index(i, angular.output_lm, output)] +=
                                coefficient * input_value
                                * static_cast<Precision>(
                                    path.channel_up_matrix[input*num_channels+output]);
                        }
                    }
                }
            }
        }

        auto Phi1r_dot = std::vector<Precision>(Phi1r.size(), 0.0);
        auto Phi1_dot = std::vector<Precision>(Phi1.size(), 0.0);
        int ij = 0;
        for (int i=0; i<num_nodes; ++i) {
            auto Phi1r_dot_i = Phi1r_dot.data()+i*num_lelm1lm2*num_channels;
            for (int j=0; j<num_neigh[i]; ++j) {
                auto R1_ij = R1.data()+ij*spl_set_1[0]->num_splines;
                auto Y_ij = Y.data()+ij*num_lm;
                auto H1_dot_ij = H1_dot.data()+neigh_indices[ij]*num_LM*num_channels;
                int lelm1lm2 = 0;
                for (int lel1l2=0; lel1l2<Phi1_l.size(); ++lel1l2) {
                    const int l1 = Phi1_l1[lel1l2];
                    const int l2 = Phi1_l2[lel1l2];
                    auto R1_ij_lel1l2 = R1_ij+lel1l2*num_channels;
                    for (int lm1=l1*l1; lm1<=l1*(l1+2); ++lm1) {
                        const Precision Y_ij_lm1 = Y_ij[lm1];
                        for (int lm2=l2*l2; lm2<=l2*(l2+2); ++lm2) {
                            auto H1_dot_ij_lm2 = H1_dot_ij+lm2*num_channels;
                            auto Phi1r_dot_i_lelm1lm2 = Phi1r_dot_i+lelm1lm2*num_channels;
                            for (int k=0; k<num_channels; ++k)
                                Phi1r_dot_i_lelm1lm2[k] +=
                                    R1_ij_lel1l2[k] * Y_ij_lm1 * H1_dot_ij_lm2[k];
                            lelm1lm2 += 1;
                        }
                    }
                }
                ij += 1;
            }
        }
        for (int i=0; i<num_nodes; ++i) {
            auto Phi1_dot_i = Phi1_dot.data()+i*num_lme*num_channels;
            auto Phi1r_dot_i = Phi1r_dot.data()+i*num_lelm1lm2*num_channels;
            for (int p=0; p<Phi1_clebsch_gordan.size(); ++p) {
                const Precision C = Phi1_clebsch_gordan[p];
                auto Phi1_dot_i_lme = Phi1_dot_i+Phi1_lme[p]*num_channels;
                auto Phi1r_dot_i_lelm1lm2 = Phi1r_dot_i+Phi1_lelm1lm2[p]*num_channels;
                for (int k=0; k<num_channels; ++k)
                    Phi1_dot_i_lme[k] += C * Phi1r_dot_i_lelm1lm2[k];
            }
        }

        auto A1_dot = std::vector<Precision>(A1.size(), 0.0);
        for (int i=0; i<num_nodes; ++i) {
            auto Phi1_dot_il = Phi1_dot.data()+i*num_lme_local*num_channels;
            auto A1_dot_il = A1_dot.data()+i*num_lm*num_channels;
            for (int l=0; l<=l_max; ++l) {
                symmetrix_blas_gemm<Precision>(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            2*l+1, num_channels, num_e[l]*num_channels,
                            1.0, Phi1_dot_il, num_e[l]*num_channels,
                            A1_weights[l].data(), num_channels,
                            0.0, A1_dot_il, num_channels);
                Phi1_dot_il += (2*l+1)*num_e[l]*num_channels;
                A1_dot_il += (2*l+1)*num_channels;
            }
        }
        if (A1_scaled) {
            for (int i=0; i<num_nodes; ++i) {
                auto A1_dot_i = A1_dot.data()+i*num_lm*num_channels;
                for (int lmk=0; lmk<num_lm*num_channels; ++lmk)
                    A1_dot_i[lmk] /= A1_scale_factors[i];
            }
        }

        auto M1_dot = std::vector<Precision>(M1.size(), 0.0);
        auto M1_grad_dot = std::vector<Precision>(M1_grad.size(), 0.0);
        for (int i=0; i<num_nodes; ++i) {
            auto A1_i = A1.data()+i*num_lm*num_channels;
            auto A1_dot_i = A1_dot.data()+i*num_lm*num_channels;
            auto M1_grad_dot_i = M1_grad_dot.data()+i*num_channels*num_lm;
            auto x = std::vector<Precision>(num_lm);
            auto x_dot = std::vector<Precision>(num_lm);
            for (int k=0; k<num_channels; ++k) {
                symmetrix_blas_copy<Precision>(num_lm, A1_i+k, num_channels, x.data(), 1);
                symmetrix_blas_copy<Precision>(num_lm, A1_dot_i+k, num_channels, x_dot.data(), 1);
                auto [f, g, g_dot] =
                    P1[node_types[i]*num_channels+k].evaluate_gradient_directional(x, x_dot);
                M1_dot[i*num_channels+k] = symmetrix_blas_dot<Precision>(num_lm, g.data(), 1, x_dot.data(), 1);
                symmetrix_blas_copy<Precision>(num_lm, g_dot.data(), 1, M1_grad_dot_i+k, num_channels);
            }
        }

        auto H2_dot = std::vector<Precision>(H2.size(), 0.0);
        for (int i=0; i<num_nodes; ++i) {
            auto H2_dot_i = H2_dot.data()+i*num_channels;
            auto H1_dot_i = H1_dot.data()+i*num_LM*num_channels;
            symmetrix_blas_gemv<Precision>(CblasRowMajor, CblasTrans,
                        num_channels, num_channels,
                        1.0, H2_weights_for_H1[node_types[i]].data(), num_channels,
                        H1_dot_i, 1,
                        0.0, H2_dot_i, 1);
            auto M1_dot_i = M1_dot.data()+i*num_channels;
            symmetrix_blas_gemv<Precision>(CblasRowMajor, CblasTrans,
                        num_channels, num_channels,
                        1.0, H2_weights_for_M1.data(), num_channels,
                        M1_dot_i, 1,
                        1.0, H2_dot_i, 1);
        }

        auto H1_adj_local = std::vector<Precision>(H1.size(), 0.0);
        auto H1_adj_dot = std::vector<Precision>(H1.size(), 0.0);
        auto H2_adj_local = std::vector<Precision>(H2.size(), 0.0);
        auto H2_adj_dot = std::vector<Precision>(H2.size(), 0.0);
        for (int i=0; i<num_nodes; ++i) {
            for (int k=0; k<num_channels; ++k)
                H1_adj_local[i*num_LM*num_channels+k] = readout_1_weights[k];
            auto x = std::vector<Precision>(H2.begin()+i*num_channels, H2.begin()+(i+1)*num_channels);
            auto x_dot = std::vector<Precision>(H2_dot.begin()+i*num_channels, H2_dot.begin()+(i+1)*num_channels);
            auto [f, g, g_dot] = readout_2->evaluate_gradient_directional(x, x_dot);
            for (int k=0; k<num_channels; ++k) {
                H2_adj_local[i*num_channels+k] = g[k];
                H2_adj_dot[i*num_channels+k] = g_dot[k];
            }
        }

        auto M1_adj_local = std::vector<Precision>(M1.size(), 0.0);
        auto M1_adj_dot = std::vector<Precision>(M1.size(), 0.0);
        for (int i=0; i<num_nodes; ++i) {
            auto H2_adj_i = H2_adj_local.data()+i*num_channels;
            auto H2_adj_dot_i = H2_adj_dot.data()+i*num_channels;
            auto H1_adj_i = H1_adj_local.data()+i*num_LM*num_channels;
            auto H1_adj_dot_i = H1_adj_dot.data()+i*num_LM*num_channels;
            symmetrix_blas_gemv<Precision>(CblasRowMajor, CblasNoTrans,
                        num_channels, num_channels,
                        1.0, H2_weights_for_H1[node_types[i]].data(), num_channels,
                        H2_adj_i, 1,
                        1.0, H1_adj_i, 1);
            symmetrix_blas_gemv<Precision>(CblasRowMajor, CblasNoTrans,
                        num_channels, num_channels,
                        1.0, H2_weights_for_H1[node_types[i]].data(), num_channels,
                        H2_adj_dot_i, 1,
                        1.0, H1_adj_dot_i, 1);
            auto M1_adj_i = M1_adj_local.data()+i*num_channels;
            auto M1_adj_dot_i = M1_adj_dot.data()+i*num_channels;
            symmetrix_blas_gemv<Precision>(CblasRowMajor, CblasNoTrans,
                        num_channels, num_channels,
                        1.0, H2_weights_for_M1.data(), num_channels,
                        H2_adj_i, 1,
                        0.0, M1_adj_i, 1);
            symmetrix_blas_gemv<Precision>(CblasRowMajor, CblasNoTrans,
                        num_channels, num_channels,
                        1.0, H2_weights_for_M1.data(), num_channels,
                        H2_adj_dot_i, 1,
                        0.0, M1_adj_dot_i, 1);
        }

        auto A1_adj_local = std::vector<Precision>(A1.size(), 0.0);
        auto A1_adj_dot = std::vector<Precision>(A1.size(), 0.0);
        for (int i=0; i<num_nodes; ++i) {
            auto M1_adj_i = M1_adj_local.data()+i*num_channels;
            auto M1_adj_dot_i = M1_adj_dot.data()+i*num_channels;
            for (int lm=0; lm<num_lm; ++lm) {
                auto A1_adj_ilm = A1_adj_local.data()+(i*num_lm+lm)*num_channels;
                auto A1_adj_dot_ilm = A1_adj_dot.data()+(i*num_lm+lm)*num_channels;
                auto M1_grad_ilm = M1_grad.data()+(i*num_lm+lm)*num_channels;
                auto M1_grad_dot_ilm = M1_grad_dot.data()+(i*num_lm+lm)*num_channels;
                for (int k=0; k<num_channels; ++k) {
                    A1_adj_ilm[k] = M1_grad_ilm[k] * M1_adj_i[k];
                    A1_adj_dot_ilm[k] =
                        M1_grad_dot_ilm[k] * M1_adj_i[k]
                        + M1_grad_ilm[k] * M1_adj_dot_i[k];
                }
            }
        }
        if (A1_scaled) {
            int ij_scale = 0;
            for (int i=0; i<num_nodes; ++i) {
                const int type_i = node_types[i];
                auto A1_i = A1.data()+i*num_lm*num_channels;
                auto A1_dot_i = A1_dot.data()+i*num_lm*num_channels;
                auto A1_adj_i = A1_adj_local.data()+i*num_lm*num_channels;
                auto A1_adj_dot_i = A1_adj_dot.data()+i*num_lm*num_channels;
                Precision dA1_dot_A1_dot = 0.0;
                for (int lmk=0; lmk<num_lm*num_channels; ++lmk) {
                    dA1_dot_A1_dot +=
                        A1_adj_dot_i[lmk] * A1_i[lmk]
                        + A1_adj_i[lmk] * A1_dot_i[lmk];
                }
                for (int j=0; j<num_neigh[i]; ++j) {
                    const int type_j = neigh_types[ij_scale];
                    const int type_ij = radial_pair_index(type_i, type_j);
                    auto [f,d] = A1_splines[type_ij].evaluate_deriv(r[ij_scale]);
                    const auto xyz_ij = xyz.data()+ij_scale*3;
                    const Precision direction[3] = {
                        static_cast<Precision>(xyz_ij[0]/r[ij_scale]),
                        static_cast<Precision>(xyz_ij[1]/r[ij_scale]),
                        static_cast<Precision>(xyz_ij[2]/r[ij_scale]),
                    };
                    auto force_deriv_ij =
                        electric_field_force_derivative.data()+seed*xyz.size()+ij_scale*3;
                    const Precision force_scale =
                        dA1_dot_A1_dot/A1_scale_factors[i]*d;
                    force_deriv_ij[0] += force_scale*direction[0];
                    force_deriv_ij[1] += force_scale*direction[1];
                    force_deriv_ij[2] += force_scale*direction[2];
                    ij_scale += 1;
                }
            }
            for (int i=0; i<num_nodes; ++i) {
                auto A1_adj_i = A1_adj_local.data()+i*num_lm*num_channels;
                auto A1_adj_dot_i = A1_adj_dot.data()+i*num_lm*num_channels;
                for (int lmk=0; lmk<num_lm*num_channels; ++lmk) {
                    A1_adj_i[lmk] /= A1_scale_factors[i];
                    A1_adj_dot_i[lmk] /= A1_scale_factors[i];
                }
            }
        }

        auto dPhi1_local = std::vector<Precision>(Phi1.size(), 0.0);
        auto dPhi1_dot = std::vector<Precision>(Phi1.size(), 0.0);
        for (int i=0; i<num_nodes; ++i) {
            auto A1_adj_il = A1_adj_local.data()+i*num_lm*num_channels;
            auto A1_adj_dot_il = A1_adj_dot.data()+i*num_lm*num_channels;
            auto dPhi1_il = dPhi1_local.data()+i*num_lme_local*num_channels;
            auto dPhi1_dot_il = dPhi1_dot.data()+i*num_lme_local*num_channels;
            for (int l=0; l<=l_max; ++l) {
                symmetrix_blas_gemm<Precision>(CblasRowMajor, CblasNoTrans, CblasTrans,
                            2*l+1, num_e[l]*num_channels, num_channels,
                            1.0, A1_adj_il, num_channels,
                            A1_weights[l].data(), num_channels,
                            0.0, dPhi1_il, num_e[l]*num_channels);
                symmetrix_blas_gemm<Precision>(CblasRowMajor, CblasNoTrans, CblasTrans,
                            2*l+1, num_e[l]*num_channels, num_channels,
                            1.0, A1_adj_dot_il, num_channels,
                            A1_weights[l].data(), num_channels,
                            0.0, dPhi1_dot_il, num_e[l]*num_channels);
                A1_adj_il += (2*l+1)*num_channels;
                A1_adj_dot_il += (2*l+1)*num_channels;
                dPhi1_il += (2*l+1)*num_e[l]*num_channels;
                dPhi1_dot_il += (2*l+1)*num_e[l]*num_channels;
            }
        }

        auto dPhi1r_local = std::vector<Precision>(Phi1r.size(), 0.0);
        auto dPhi1r_dot = std::vector<Precision>(Phi1r.size(), 0.0);
        for (int i=0; i<num_nodes; ++i) {
            auto dPhi1r_i = dPhi1r_local.data()+i*num_lelm1lm2*num_channels;
            auto dPhi1r_dot_i = dPhi1r_dot.data()+i*num_lelm1lm2*num_channels;
            auto dPhi1_i = dPhi1_local.data()+i*num_lme*num_channels;
            auto dPhi1_dot_i = dPhi1_dot.data()+i*num_lme*num_channels;
            for (int p=0; p<Phi1_clebsch_gordan.size(); ++p) {
                const Precision C = Phi1_clebsch_gordan[p];
                auto dPhi1r_i_lelm1lm2 = dPhi1r_i+Phi1_lelm1lm2[p]*num_channels;
                auto dPhi1r_dot_i_lelm1lm2 = dPhi1r_dot_i+Phi1_lelm1lm2[p]*num_channels;
                auto dPhi1_i_lme = dPhi1_i+Phi1_lme[p]*num_channels;
                auto dPhi1_dot_i_lme = dPhi1_dot_i+Phi1_lme[p]*num_channels;
                for (int k=0; k<num_channels; ++k) {
                    dPhi1r_i_lelm1lm2[k] += C * dPhi1_i_lme[k];
                    dPhi1r_dot_i_lelm1lm2[k] += C * dPhi1_dot_i_lme[k];
                }
            }
        }

        ij = 0;
        for (int i=0; i<num_nodes; ++i) {
            auto dPhi1r_i = dPhi1r_local.data()+i*num_lelm1lm2*num_channels;
            auto dPhi1r_dot_i = dPhi1r_dot.data()+i*num_lelm1lm2*num_channels;
            for (int j=0; j<num_neigh[i]; ++j) {
                auto R1_ij = R1.data()+ij*spl_set_1[0]->num_splines;
                auto R1_deriv_ij = R1_deriv.data()+ij*spl_set_1[0]->num_splines;
                auto Y_ij = Y.data()+ij*num_lm;
                auto Y_grad_ij = Y_grad.data()+ij*3*num_lm;
                auto H1_ij = H1.data()+neigh_indices[ij]*num_LM*num_channels;
                auto H1_dot_ij = H1_dot.data()+neigh_indices[ij]*num_LM*num_channels;
                auto H1_adj_ij = H1_adj_local.data()+neigh_indices[ij]*num_LM*num_channels;
                auto H1_adj_dot_ij = H1_adj_dot.data()+neigh_indices[ij]*num_LM*num_channels;
                const auto xyz_ij = xyz.data()+ij*3;
                const Precision direction_x =
                    static_cast<Precision>(xyz_ij[0]/r[ij]);
                const Precision direction_y =
                    static_cast<Precision>(xyz_ij[1]/r[ij]);
                const Precision direction_z =
                    static_cast<Precision>(xyz_ij[2]/r[ij]);
                auto force_deriv_ij = electric_field_force_derivative.data()+seed*xyz.size()+ij*3;
                int lelm1lm2 = 0;
                for (int lel1l2=0; lel1l2<Phi1_l.size(); ++lel1l2) {
                    const int l1 = Phi1_l1[lel1l2];
                    const int l2 = Phi1_l2[lel1l2];
                    auto R1_ij_lel1l2 = R1_ij+lel1l2*num_channels;
                    auto R1_deriv_ij_lel1l2 = R1_deriv_ij+lel1l2*num_channels;
                    for (int lm1=l1*l1; lm1<=l1*(l1+2); ++lm1) {
                        const Precision Y_ij_lm1 = Y_ij[lm1];
                        const Precision Y_grad_ij_x_lm1 = Y_grad_ij[0*num_lm+lm1];
                        const Precision Y_grad_ij_y_lm1 = Y_grad_ij[1*num_lm+lm1];
                        const Precision Y_grad_ij_z_lm1 = Y_grad_ij[2*num_lm+lm1];
                        for (int lm2=l2*l2; lm2<=l2*(l2+2); ++lm2) {
                            auto H1_ij_lm2 = H1_ij+lm2*num_channels;
                            auto H1_dot_ij_lm2 = H1_dot_ij+lm2*num_channels;
                            auto H1_adj_ij_lm2 = H1_adj_ij+lm2*num_channels;
                            auto H1_adj_dot_ij_lm2 = H1_adj_dot_ij+lm2*num_channels;
                            auto dPhi1r_i_lelm1lm2 = dPhi1r_i+lelm1lm2*num_channels;
                            auto dPhi1r_dot_i_lelm1lm2 = dPhi1r_dot_i+lelm1lm2*num_channels;
                            for (int k=0; k<num_channels; ++k) {
                                const Precision force_factor_x =
                                    direction_x * R1_deriv_ij_lel1l2[k] * Y_ij_lm1 * H1_ij_lm2[k]
                                    + R1_ij_lel1l2[k] * Y_grad_ij_x_lm1 * H1_ij_lm2[k];
                                const Precision force_factor_y =
                                    direction_y * R1_deriv_ij_lel1l2[k] * Y_ij_lm1 * H1_ij_lm2[k]
                                    + R1_ij_lel1l2[k] * Y_grad_ij_y_lm1 * H1_ij_lm2[k];
                                const Precision force_factor_z =
                                    direction_z * R1_deriv_ij_lel1l2[k] * Y_ij_lm1 * H1_ij_lm2[k]
                                    + R1_ij_lel1l2[k] * Y_grad_ij_z_lm1 * H1_ij_lm2[k];
                                const Precision force_factor_dot_x =
                                    direction_x * R1_deriv_ij_lel1l2[k] * Y_ij_lm1 * H1_dot_ij_lm2[k]
                                    + R1_ij_lel1l2[k] * Y_grad_ij_x_lm1 * H1_dot_ij_lm2[k];
                                const Precision force_factor_dot_y =
                                    direction_y * R1_deriv_ij_lel1l2[k] * Y_ij_lm1 * H1_dot_ij_lm2[k]
                                    + R1_ij_lel1l2[k] * Y_grad_ij_y_lm1 * H1_dot_ij_lm2[k];
                                const Precision force_factor_dot_z =
                                    direction_z * R1_deriv_ij_lel1l2[k] * Y_ij_lm1 * H1_dot_ij_lm2[k]
                                    + R1_ij_lel1l2[k] * Y_grad_ij_z_lm1 * H1_dot_ij_lm2[k];
                                force_deriv_ij[0] += -(
                                    dPhi1r_dot_i_lelm1lm2[k] * force_factor_x
                                    + dPhi1r_i_lelm1lm2[k] * force_factor_dot_x);
                                force_deriv_ij[1] += -(
                                    dPhi1r_dot_i_lelm1lm2[k] * force_factor_y
                                    + dPhi1r_i_lelm1lm2[k] * force_factor_dot_y);
                                force_deriv_ij[2] += -(
                                    dPhi1r_dot_i_lelm1lm2[k] * force_factor_z
                                    + dPhi1r_i_lelm1lm2[k] * force_factor_dot_z);
                                H1_adj_ij_lm2[k] +=
                                    R1_ij_lel1l2[k]*Y_ij_lm1*dPhi1r_i_lelm1lm2[k];
                                H1_adj_dot_ij_lm2[k] +=
                                    R1_ij_lel1l2[k]*Y_ij_lm1*dPhi1r_dot_i_lelm1lm2[k];
                            }
                            lelm1lm2 += 1;
                        }
                    }
                }
                ij += 1;
            }
        }

        auto H1_adj_before_linear_up = H1_adj_local;
        auto H1_adj_dot_before_linear_up = H1_adj_dot;
        std::fill(H1_adj_local.begin(), H1_adj_local.end(), 0.0);
        std::fill(H1_adj_dot.begin(), H1_adj_dot.end(), 0.0);
        for (int i=0; i<num_nodes; ++i) {
            for (int l=0; l<=L_max; ++l) {
                const auto H1_adj_il = H1_adj_before_linear_up.data()+(i*num_LM+l*l)*num_channels;
                const auto H1_adj_dot_il = H1_adj_dot_before_linear_up.data()+(i*num_LM+l*l)*num_channels;
                const auto weights_l = H1_linear_up_weights.data()+l*num_channels*num_channels;
                auto H1_pre_adj_il = H1_adj_local.data()+(i*num_LM+l*l)*num_channels;
                auto H1_pre_adj_dot_il = H1_adj_dot.data()+(i*num_LM+l*l)*num_channels;
                symmetrix_blas_gemm<Precision>(CblasRowMajor, CblasNoTrans, CblasTrans,
                            2*l+1, num_channels, num_channels,
                            1.0, H1_adj_il, num_channels,
                            weights_l, num_channels,
                            0.0, H1_pre_adj_il, num_channels);
                symmetrix_blas_gemm<Precision>(CblasRowMajor, CblasNoTrans, CblasTrans,
                            2*l+1, num_channels, num_channels,
                            1.0, H1_adj_dot_il, num_channels,
                            weights_l, num_channels,
                            0.0, H1_pre_adj_dot_il, num_channels);
            }
        }

        auto H1_pre_adj = H1_adj_local;
        auto H1_pre_adj_dot = H1_adj_dot;
        for (int i=0; i<num_nodes; ++i) {
            for (const auto& path : field_coupling_paths) {
                for (const auto& angular : path.angular_entries) {
                    const Precision coefficient =
                        static_cast<Precision>(angular.coefficient);
                    const Precision field_value = field[angular.field_component];
                    const Precision field_dot =
                        angular.field_component == seed ? 1.0 : 0.0;
                    for (int input=0; input<num_channels; ++input) {
                        const Precision input_value =
                            H1_pre_field[h1_index(i, angular.input_lm, input)];
                        for (int output=0; output<num_channels; ++output) {
                            const Precision weight = coefficient
                                * static_cast<Precision>(
                                    path.channel_matrix[input*num_channels+output]);
                            const Precision output_adj =
                                H1_adj_local[h1_index(i, angular.output_lm, output)];
                            const Precision output_adj_dot =
                                H1_adj_dot[h1_index(i, angular.output_lm, output)];
                            H1_pre_adj[h1_index(i, angular.input_lm, input)] +=
                                field_value * weight * output_adj;
                            H1_pre_adj_dot[h1_index(i, angular.input_lm, input)] +=
                                field_value * weight * output_adj_dot
                                + field_dot * weight * output_adj;
                            electric_field_hessian[
                                angular.field_component*3 + seed] +=
                                weight * input_value * output_adj_dot;
                        }
                    }
                }
            }
        }
        H1_adj_local = std::move(H1_pre_adj);
        H1_adj_dot = std::move(H1_pre_adj_dot);

        auto M0_adj_local = std::vector<Precision>(M0.size(), 0.0);
        auto M0_adj_dot = std::vector<Precision>(M0.size(), 0.0);
        for (int i=0; i<num_nodes; ++i) {
            for (int l=0; l<=L_max; ++l) {
                const auto H1_adj_il = H1_adj_local.data()+(i*num_LM+l*l)*num_channels;
                const auto H1_adj_dot_il = H1_adj_dot.data()+(i*num_LM+l*l)*num_channels;
                const auto weights_l = H1_product_weights.data()+l*num_channels*num_channels;
                auto M0_adj_il = M0_adj_local.data()+(i*num_LM+l*l)*num_channels;
                auto M0_adj_dot_il = M0_adj_dot.data()+(i*num_LM+l*l)*num_channels;
                symmetrix_blas_gemm<Precision>(CblasRowMajor, CblasNoTrans, CblasTrans,
                            2*l+1, num_channels, num_channels,
                            1.0, H1_adj_il, num_channels,
                            weights_l, num_channels,
                            0.0, M0_adj_il, num_channels);
                symmetrix_blas_gemm<Precision>(CblasRowMajor, CblasNoTrans, CblasTrans,
                            2*l+1, num_channels, num_channels,
                            1.0, H1_adj_dot_il, num_channels,
                            weights_l, num_channels,
                            0.0, M0_adj_dot_il, num_channels);
            }
        }

        auto A0_adj_local = std::vector<Precision>(A0.size(), 0.0);
        auto A0_adj_dot = std::vector<Precision>(A0.size(), 0.0);
        for (int i=0; i<num_nodes; ++i) {
            auto A0_adj_i = A0_adj_local.data()+i*num_lm*num_channels;
            auto A0_adj_dot_i = A0_adj_dot.data()+i*num_lm*num_channels;
            auto M0_adj_i = M0_adj_local.data()+i*num_LM*num_channels;
            auto M0_adj_dot_i = M0_adj_dot.data()+i*num_LM*num_channels;
            auto M0_grad_i = M0_grad.data()+i*num_LM*num_channels*num_lm;
            for (int lm=0; lm<num_lm; ++lm) {
                auto A0_adj_ilm = A0_adj_i + lm*num_channels;
                auto A0_adj_dot_ilm = A0_adj_dot_i + lm*num_channels;
                for (int lmp=0; lmp<num_LM; ++lmp) {
                    auto M0_adj_ilmp = M0_adj_i + lmp*num_channels;
                    auto M0_adj_dot_ilmp = M0_adj_dot_i + lmp*num_channels;
                    auto M0_grad_ilmplm =
                        M0_grad_i + lmp*num_lm*num_channels + lm*num_channels;
                    for (int k=0; k<num_channels; ++k) {
                        A0_adj_ilm[k] += M0_grad_ilmplm[k] * M0_adj_ilmp[k];
                        A0_adj_dot_ilm[k] += M0_grad_ilmplm[k] * M0_adj_dot_ilmp[k];
                    }
                }
            }
        }

        if (A0_scaled) {
            int ij_scale = 0;
            for (int i=0; i<num_nodes; ++i) {
                const int type_i = node_types[i];
                auto A0_i = A0.data()+i*num_lm*num_channels;
                auto A0_adj_dot_i = A0_adj_dot.data()+i*num_lm*num_channels;
                Precision dA0_dot_A0_dot = 0.0;
                for (int lmk=0; lmk<num_lm*num_channels; ++lmk)
                    dA0_dot_A0_dot += A0_adj_dot_i[lmk] * A0_i[lmk];
                for (int j=0; j<num_neigh[i]; ++j) {
                    const int type_j = neigh_types[ij_scale];
                    const int type_ij = radial_pair_index(type_i, type_j);
                    auto [f,d] = A0_splines[type_ij].evaluate_deriv(r[ij_scale]);
                    const auto xyz_ij = xyz.data()+ij_scale*3;
                    const Precision direction[3] = {
                        static_cast<Precision>(xyz_ij[0]/r[ij_scale]),
                        static_cast<Precision>(xyz_ij[1]/r[ij_scale]),
                        static_cast<Precision>(xyz_ij[2]/r[ij_scale]),
                    };
                    auto force_deriv_ij =
                        electric_field_force_derivative.data()+seed*xyz.size()+ij_scale*3;
                    const Precision force_scale =
                        dA0_dot_A0_dot/A0_scale_factors[i]*d;
                    force_deriv_ij[0] += force_scale*direction[0];
                    force_deriv_ij[1] += force_scale*direction[1];
                    force_deriv_ij[2] += force_scale*direction[2];
                    ij_scale += 1;
                }
            }
            for (int i=0; i<num_nodes; ++i) {
                auto A0_adj_i = A0_adj_local.data()+i*num_lm*num_channels;
                auto A0_adj_dot_i = A0_adj_dot.data()+i*num_lm*num_channels;
                for (int lmk=0; lmk<num_lm*num_channels; ++lmk) {
                    A0_adj_i[lmk] /= A0_scale_factors[i];
                    A0_adj_dot_i[lmk] /= A0_scale_factors[i];
                }
            }
        }

        int ij_a0 = 0;
        for (int i=0; i<num_nodes; ++i) {
            auto Phi0_adj_dot_i = std::vector<Precision>(num_lm*num_channels, 0.0);
            for (int l=0; l<=l_max; ++l) {
                auto Phi0_adj_dot_il = Phi0_adj_dot_i.data()+l*l*num_channels;
                auto A0_adj_dot_il = A0_adj_dot.data()+(i*num_lm+l*l)*num_channels;
                symmetrix_blas_gemm<Precision>(CblasRowMajor, CblasNoTrans, CblasTrans,
                            2*l+1, num_channels, num_channels,
                            1.0, A0_adj_dot_il, num_channels,
                            A0_weights[node_types[i]][l].data(), num_channels,
                            0.0, Phi0_adj_dot_il, num_channels);
            }

            for (int j=0; j<num_neigh[i]; ++j) {
                const auto xyz_ij = xyz.data()+ij_a0*3;
                const Precision direction_x =
                    static_cast<Precision>(xyz_ij[0]/r[ij_a0]);
                const Precision direction_y =
                    static_cast<Precision>(xyz_ij[1]/r[ij_a0]);
                const Precision direction_z =
                    static_cast<Precision>(xyz_ij[2]/r[ij_a0]);
                auto Y_ij = Y.data()+ij_a0*num_lm;
                auto Y_grad_ij = Y_grad.data()+ij_a0*3*num_lm;
                auto H0_ij = H0_weights.data()+neigh_types[ij_a0]*num_channels;
                auto force_deriv_ij = electric_field_force_derivative.data()+seed*xyz.size()+ij_a0*3;
                for (int l=0; l<=l_max; ++l) {
                    auto R0_ij_l = R0.data()+ij_a0*(l_max+1)*num_channels+l*num_channels;
                    auto R0_deriv_ij_l = R0_deriv.data()+ij_a0*(l_max+1)*num_channels+l*num_channels;
                    for (int m=-l; m<=l; ++m) {
                        const int lm = l*l+l+m;
                        const Precision Y_ij_lm = Y_ij[lm];
                        const Precision Y_grad_ij_lm_x = Y_grad_ij[lm];
                        const Precision Y_grad_ij_lm_y = Y_grad_ij[num_lm+lm];
                        const Precision Y_grad_ij_lm_z = Y_grad_ij[2*num_lm+lm];
                        auto Phi0_adj_dot_i_lm = Phi0_adj_dot_i.data()+lm*num_channels;
                        for (int k=0; k<num_channels; ++k) {
                            force_deriv_ij[0] += -Phi0_adj_dot_i_lm[k] * (
                                direction_x * R0_deriv_ij_l[k] * Y_ij_lm * H0_ij[k]
                                + R0_ij_l[k] * Y_grad_ij_lm_x * H0_ij[k]);
                            force_deriv_ij[1] += -Phi0_adj_dot_i_lm[k] * (
                                direction_y * R0_deriv_ij_l[k] * Y_ij_lm * H0_ij[k]
                                + R0_ij_l[k] * Y_grad_ij_lm_y * H0_ij[k]);
                            force_deriv_ij[2] += -Phi0_adj_dot_i_lm[k] * (
                                direction_z * R0_deriv_ij_l[k] * Y_ij_lm * H0_ij[k]
                                + R0_ij_l[k] * Y_grad_ij_lm_z * H0_ij[k]);
                        }
                    }
                }
                ij_a0 += 1;
            }
        }
    }

    if (streamed_edges == MACEStreamedEdgesMode::generic) {
        std::vector<Precision>().swap(R1);
        std::vector<Precision>().swap(R1_deriv);
    }
    if (streamed_edges == MACEStreamedEdgesMode::generic) {
        std::vector<Precision>().swap(R0);
        std::vector<Precision>().swap(R0_deriv);
    }
}

template <typename Precision>
void MACECPU<Precision>::compute_electric_field_force_derivative(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r,
    std::span<const double> electric_field)
{
    compute_electric_field_hessian(
        num_nodes,
        node_types,
        num_neigh,
        neigh_indices,
        neigh_types,
        xyz,
        r,
        electric_field);
}

template <typename Precision>
void MACECPU<Precision>::compute_R0(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> r)
{
    if (spl_set_0.empty())
        throw std::runtime_error(
            "MACE compact radial cache is not prepared; call prepare_active_types first.");
    const int num_spl = spl_set_0[0]->num_splines;
    R0.resize(r.size()*num_spl);
    R0_deriv.resize(R0.size());
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        const int type_i = node_types[i];
        for (int j=0; j<num_neigh[i]; ++j) {
            const int type_j = neigh_types[ij];
            const int type_ij = radial_pair_index(type_i, type_j);
            auto R0_ij = std::span<Precision>(R0.data()+ij*num_spl,num_spl);
            auto R0_deriv_ij = std::span<Precision>(R0_deriv.data()+ij*num_spl,num_spl);
            spl_set_0[type_ij]->evaluate_derivs(r[ij], R0_ij, R0_deriv_ij);
            ij += 1;
        }
    }
}

template <typename Precision>
void MACECPU<Precision>::compute_field_H1(
    const int num_nodes,
    std::span<const double> electric_field)
{
    if (!has_field_coupling)
        return;

    if (electric_field.size() != 3 && electric_field.size() != 3*num_nodes)
        throw std::runtime_error("MACEField electric_field must have shape (3,) or (num_nodes, 3).");

    const int hidden_size = num_LM*num_channels;
    if (H1.size() != num_nodes*hidden_size)
        throw std::runtime_error("MACEField H1 buffer size does not match num_nodes.");

    H1_pre_field = H1;
    const auto h1_index = [this](int i, int lm, int k) {
        return (i*num_LM + lm)*num_channels + k;
    };

    for (int i=0; i<num_nodes; ++i) {
        const double* field_i = electric_field.data()
            + (electric_field.size() == 3 ? 0 : 3*i);
        for (const auto& path : field_coupling_paths) {
            for (const auto& angular : path.angular_entries) {
                const Precision angular_field = static_cast<Precision>(
                    angular.coefficient * field_i[angular.field_component]);
                for (int input=0; input<num_channels; ++input) {
                    const Precision input_value =
                        H1_pre_field[h1_index(i, angular.input_lm, input)];
                    for (int output=0; output<num_channels; ++output) {
                        H1[h1_index(i, angular.output_lm, output)] +=
                            angular_field * input_value
                            * static_cast<Precision>(
                                path.channel_matrix[input*num_channels+output]);
                    }
                }
            }
        }
    }
}

template <typename Precision>
void MACECPU<Precision>::reverse_field_H1(
    const int num_nodes,
    std::span<const double> electric_field)
{
    if (!has_field_coupling)
        return;

    if (electric_field.size() != 3 && electric_field.size() != 3*num_nodes)
        throw std::runtime_error("MACEField electric_field must have shape (3,) or (num_nodes, 3).");

    const int hidden_size = num_LM*num_channels;
    if (H1_adj.size() != num_nodes*hidden_size || H1_pre_field.size() != num_nodes*hidden_size)
        throw std::runtime_error("MACEField reverse_field_H1 requires H1_adj and saved pre-field H1 buffers.");

    const bool global_field = electric_field.size() == 3;
    electric_field_adj.assign(electric_field.size(), 0.0);
    std::vector<Precision> H1_pre_adj = H1_adj;

    const auto h1_index = [this](int i, int lm, int k) {
        return (i*num_LM + lm)*num_channels + k;
    };

    for (int i=0; i<num_nodes; ++i) {
        const double* field_i = electric_field.data() + (global_field ? 0 : 3*i);
        Precision* field_adj_i = electric_field_adj.data() + (global_field ? 0 : 3*i);
        for (const auto& path : field_coupling_paths) {
            for (const auto& angular : path.angular_entries) {
                const Precision angular_field = static_cast<Precision>(
                    angular.coefficient * field_i[angular.field_component]);
                for (int input=0; input<num_channels; ++input) {
                    const Precision input_value =
                        H1_pre_field[h1_index(i, angular.input_lm, input)];
                    for (int output=0; output<num_channels; ++output) {
                        const Precision weight = static_cast<Precision>(
                            path.channel_matrix[input*num_channels+output]);
                        const Precision output_adj =
                            H1_adj[h1_index(i, angular.output_lm, output)];
                        H1_pre_adj[h1_index(i, angular.input_lm, input)] +=
                            angular_field * weight * output_adj;
                        field_adj_i[angular.field_component] +=
                            static_cast<Precision>(angular.coefficient)
                            * weight * input_value * output_adj;
                    }
                }
            }
        }
    }

    H1_adj = std::move(H1_pre_adj);
}

template <typename Precision>
void MACECPU<Precision>::compute_R1(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> r)
{
    if (spl_set_1.empty())
        throw std::runtime_error(
            "MACE compact radial cache is not prepared; call prepare_active_types first.");
    const int num_spl = spl_set_1[0]->num_splines;
    R1.resize(r.size()*num_spl);
    R1_deriv.resize(R1.size());
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        const int type_i = node_types[i];
        for (int j=0; j<num_neigh[i]; ++j) {
            const int type_j = neigh_types[ij];
            const int type_ij = radial_pair_index(type_i, type_j);
            auto R1_ij = std::span<Precision>(R1.data()+ij*num_spl,num_spl);
            auto R1_deriv_ij = std::span<Precision>(R1_deriv.data()+ij*num_spl,num_spl);
            spl_set_1[type_ij]->evaluate_derivs(r[ij], R1_ij, R1_deriv_ij);
            ij += 1;
        }
    }
}

template <typename Precision>
void MACECPU<Precision>::compute_Y(
    std::span<const double> xyz)
{
    if (xyz.size() == 0) return;
    const int num = xyz.size()/3;
    Y.resize(num*num_lm);
    Y_grad.resize(3*num*num_lm);
    // shuffle to match e3nn conventions
    xyz_shuffled.resize(3*num);
    for (int i=0; i<num; ++i) {
        xyz_shuffled[3*i]   = xyz[3*i+2];
        xyz_shuffled[3*i+1] = xyz[3*i];
        xyz_shuffled[3*i+2] = xyz[3*i+1];
    }
    sphericart::SphericalHarmonics<Precision> sphericart(l_max);
    sphericart.compute_with_gradients(xyz_shuffled, Y, Y_grad);
    // normalize to match e3nn conventions
    for (int i=0; i<Y.size(); ++i)
        Y[i] *= 2*std::sqrt(std::numbers::pi);
    for (int i=0; i<Y_grad.size(); ++i)
        Y_grad[i] *= 2*std::sqrt(std::numbers::pi);
    // unshuffle gradient
    auto Y_grad_shuffled = Y_grad;
    for (int i=0; i<num; ++i) {
        for (int lm=0; lm<num_lm; ++lm) {
            Y_grad[3*i*num_lm+0*num_lm+lm] = Y_grad_shuffled[3*i*num_lm+1*num_lm+lm];
            Y_grad[3*i*num_lm+1*num_lm+lm] = Y_grad_shuffled[3*i*num_lm+2*num_lm+lm];
            Y_grad[3*i*num_lm+2*num_lm+lm] = Y_grad_shuffled[3*i*num_lm+0*num_lm+lm];
        }
    }
}

template <typename Precision>
void MACECPU<Precision>::compute_A0(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types)
{
    A0.resize(static_cast<std::size_t>(num_nodes)*num_lm*num_channels);

    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {

        // compute Phi0_i
        auto Phi0_i = std::vector<Precision>(num_lm*num_channels, 0.0);
        for (int j=0; j<num_neigh[i]; ++j) {
            auto Y_ij = Y.data()+ij*num_lm;
            auto H0_ij = H0_weights.data()+neigh_types[ij]*num_channels;
            for (int l=0; l<=l_max; ++l) {
                auto R0_ij_l = R0.data()+ij*(l_max+1)*num_channels+l*num_channels;
                for (int m=-l; m<=l; ++m) {
                    const int lm = l*l+l+m;
                    const Precision Y_ij_lm = Y_ij[lm];
                    auto Phi0_i_lm = Phi0_i.data()+lm*num_channels;
                    for (int k=0; k<num_channels; ++k) {
                        Phi0_i_lm[k] += R0_ij_l[k] * Y_ij_lm * H0_ij[k];
                    }
                }
            }
            ij += 1;
        }

        // [A0_il]_mk = \sum_k' [Phi0_il]_mk' [W_il]_k'k
        for (int l=0; l<=l_max; ++l) {
            auto Phi0_il = Phi0_i.data()+l*l*num_channels;
            auto A0_il = A0.data()+(i*num_lm+l*l)*num_channels;
            symmetrix_blas_gemm<Precision>(
                CblasRowMajor,                        // const CBLAS_LAYOUT Layout
                CblasNoTrans,                         // const CBLAS_TRANSPOSE transa
                CblasNoTrans,                         // const CBLAS_TRANSPOSE transb
                (2*l+1),                              // const MKL_INT m
                num_channels,                         // const MKL_INT n
                num_channels,                         // const MKL_INT k
                1.0,                                  // const double alpha
                Phi0_il,                              // const double *a
                num_channels,                         // const MKL_INT lda
                A0_weights[node_types[i]][l].data(),  // const double *b
                num_channels,                         // const MKL_INT ldb
                0.0,                                  // const double beta
                A0_il,                                // double *c
                num_channels);                        // const MKL_INT ldc
        }
    }
}

template <typename Precision>
void MACECPU<Precision>::compute_A0_streamed(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> r)
{
    A0.resize(static_cast<std::size_t>(num_nodes)*num_lm*num_channels);
    auto radial_values = std::vector<Precision>(num_channels);

    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        auto Phi0_i = std::vector<Precision>(num_lm*num_channels, 0.0);
        const int type_i = node_types[i];
        for (int j=0; j<num_neigh[i]; ++j) {
            const int pair = radial_pair_index(type_i, neigh_types[ij]);
            const auto& spline = *spl_set_0[pair];
            const auto point = spline.evaluation_point(r[ij]);
            const auto Y_ij = Y.data()+ij*num_lm;
            const auto H0_ij = H0_weights.data()+neigh_types[ij]*num_channels;
            for (int l=0; l<=l_max; ++l) {
                for (int k=0; k<num_channels; ++k)
                    radial_values[k] =
                        spline.evaluate_function(point, l*num_channels+k);
                for (int m=-l; m<=l; ++m) {
                    const int lm = l*l+l+m;
                    const Precision Y_ij_lm = Y_ij[lm];
                    auto Phi0_i_lm = Phi0_i.data()+lm*num_channels;
                    for (int k=0; k<num_channels; ++k)
                        Phi0_i_lm[k] +=
                            radial_values[k]*Y_ij_lm*H0_ij[k];
                }
            }
            ij += 1;
        }

        for (int l=0; l<=l_max; ++l) {
            auto Phi0_il = Phi0_i.data()+l*l*num_channels;
            auto A0_il = A0.data()+(i*num_lm+l*l)*num_channels;
            symmetrix_blas_gemm<Precision>(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                2*l+1, num_channels, num_channels,
                1.0, Phi0_il, num_channels,
                A0_weights[node_types[i]][l].data(), num_channels,
                0.0, A0_il, num_channels);
        }
    }
}

template <typename Precision>
void MACECPU<Precision>::reverse_A0(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r)
{
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {

        auto Phi0_adj_i = std::vector<Precision>(num_lm*num_channels);

        // [dE/dPhi0_il]_mk = \sum_k' [dE/dA0_il]_mk' [trans(W_il)]_k'k
        for (int l=0; l<=l_max; ++l) {
            auto Phi0_adj_il = Phi0_adj_i.data()+l*l*num_channels;
            auto A0_adj_il = A0_adj.data()+(i*num_lm+l*l)*num_channels;
            symmetrix_blas_gemm<Precision>(
                CblasRowMajor,                        // const CBLAS_LAYOUT Layout
                CblasNoTrans,                         // const CBLAS_TRANSPOSE transa
                CblasTrans,                           // const CBLAS_TRANSPOSE transb
                (2*l+1),                              // const MKL_INT m
                num_channels,                         // const MKL_INT n
                num_channels,                         // const MKL_INT k
                1.0,                                  // const double alpha
                A0_adj_il,                            // const double *a
                num_channels,                         // const MKL_INT lda
                A0_weights[node_types[i]][l].data(),  // const double *b
                num_channels,                         // const MKL_INT ldb
                0.0,                                  // const double beta
                Phi0_adj_il,                          // double *c
                num_channels);                        // const MKL_INT ldc
        }

        // Warning: Assumes node_forces have been initialized elsewhere
        for (int j=0; j<num_neigh[i]; ++j) {
            auto xyz_ij = xyz.data()+ij*3;
            auto r_ij = r[ij];
            const Precision direction_x = static_cast<Precision>(xyz_ij[0]/r_ij);
            const Precision direction_y = static_cast<Precision>(xyz_ij[1]/r_ij);
            const Precision direction_z = static_cast<Precision>(xyz_ij[2]/r_ij);
            auto Y_ij = Y.data()+ij*num_lm;
            auto Y_grad_ij = Y_grad.data()+ij*3*num_lm;
            auto H0_ij = H0_weights.data()+neigh_types[ij]*num_channels;
            auto node_forces_ij = node_forces.data()+ij*3;
            Precision force_x = 0.0;
            Precision force_y = 0.0;
            Precision force_z = 0.0;
            for (int l=0; l<=l_max; ++l) {
                auto R0_ij_l = R0.data()+ij*(l_max+1)*num_channels+l*num_channels;
                auto R0_deriv_ij_l = R0_deriv.data()+ij*(l_max+1)*num_channels+l*num_channels;
                for (int m=-l; m<=l; ++m) {
                    const int lm = l*l+l+m;
                    const Precision Y_ij_lm = Y_ij[lm];
                    const Precision Y_grad_ij_lm_x = Y_grad_ij[lm];
                    const Precision Y_grad_ij_lm_y = Y_grad_ij[num_lm+lm];
                    const Precision Y_grad_ij_lm_z = Y_grad_ij[2*num_lm+lm];
                    auto Phi0_adj_i_lm = Phi0_adj_i.data()+lm*num_channels;
                    for (int k=0; k<num_channels; ++k) {
                        force_x += -Phi0_adj_i_lm[k] * (
                            direction_x * R0_deriv_ij_l[k] * Y_ij_lm * H0_ij[k]
                            + R0_ij_l[k] * Y_grad_ij_lm_x * H0_ij[k] );
                        force_y += -Phi0_adj_i_lm[k] * (
                            direction_y * R0_deriv_ij_l[k] * Y_ij_lm * H0_ij[k]
                            + R0_ij_l[k] * Y_grad_ij_lm_y * H0_ij[k] );
                        force_z += -Phi0_adj_i_lm[k] * (
                            direction_z * R0_deriv_ij_l[k] * Y_ij_lm * H0_ij[k]
                            + R0_ij_l[k] * Y_grad_ij_lm_z * H0_ij[k]);
                    }
                }
            }
            node_forces_ij[0] += force_x;
            node_forces_ij[1] += force_y;
            node_forces_ij[2] += force_z;
            ij += 1;
        }
    }
}

template <typename Precision>
void MACECPU<Precision>::reverse_A0_streamed(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r)
{
    auto radial_values = std::vector<Precision>(num_channels);
    auto radial_derivatives = std::vector<Precision>(num_channels);
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        auto Phi0_adj_i = std::vector<Precision>(num_lm*num_channels);
        for (int l=0; l<=l_max; ++l) {
            auto Phi0_adj_il = Phi0_adj_i.data()+l*l*num_channels;
            auto A0_adj_il = A0_adj.data()+(i*num_lm+l*l)*num_channels;
            symmetrix_blas_gemm<Precision>(
                CblasRowMajor, CblasNoTrans, CblasTrans,
                2*l+1, num_channels, num_channels,
                1.0, A0_adj_il, num_channels,
                A0_weights[node_types[i]][l].data(), num_channels,
                0.0, Phi0_adj_il, num_channels);
        }

        const int type_i = node_types[i];
        for (int j=0; j<num_neigh[i]; ++j) {
            const int pair = radial_pair_index(type_i, neigh_types[ij]);
            const auto& spline = *spl_set_0[pair];
            const auto point = spline.evaluation_point(r[ij]);
            const auto xyz_ij = xyz.data()+ij*3;
            const Precision direction_x = static_cast<Precision>(xyz_ij[0]/r[ij]);
            const Precision direction_y = static_cast<Precision>(xyz_ij[1]/r[ij]);
            const Precision direction_z = static_cast<Precision>(xyz_ij[2]/r[ij]);
            const auto Y_ij = Y.data()+ij*num_lm;
            const auto Y_grad_ij = Y_grad.data()+ij*3*num_lm;
            const auto H0_ij = H0_weights.data()+neigh_types[ij]*num_channels;
            auto node_forces_ij = node_forces.data()+ij*3;
            Precision force_x = 0.0;
            Precision force_y = 0.0;
            Precision force_z = 0.0;
            for (int l=0; l<=l_max; ++l) {
                for (int k=0; k<num_channels; ++k) {
                    spline.evaluate_function_derivs(
                        point, l*num_channels+k,
                        radial_values[k], radial_derivatives[k]);
                }
                for (int m=-l; m<=l; ++m) {
                    const int lm = l*l+l+m;
                    const Precision Y_ij_lm = Y_ij[lm];
                    auto Phi0_adj_i_lm = Phi0_adj_i.data()+lm*num_channels;
                    for (int k=0; k<num_channels; ++k) {
                        const Precision radial_force =
                            radial_derivatives[k]*Y_ij_lm*H0_ij[k];
                        const Precision angular_force = radial_values[k]*H0_ij[k];
                        const Precision adjoint = Phi0_adj_i_lm[k];
                        force_x -= adjoint*(
                            direction_x*radial_force
                            +angular_force*Y_grad_ij[lm]);
                        force_y -= adjoint*(
                            direction_y*radial_force
                            +angular_force*Y_grad_ij[num_lm+lm]);
                        force_z -= adjoint*(
                            direction_z*radial_force
                            +angular_force*Y_grad_ij[2*num_lm+lm]);
                    }
                }
            }
            node_forces_ij[0] += force_x;
            node_forces_ij[1] += force_y;
            node_forces_ij[2] += force_z;
            ij += 1;
        }
    }
}

template <typename Precision>
void MACECPU<Precision>::compute_A0_scaled(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> r)
{
    if (not A0_scaled) return;
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        const int type_i = node_types[i];
        Precision A0_scale_factor = 1.0;
        for (int j=0; j<num_neigh[i]; ++j) {
            const int type_j = neigh_types[ij];
            const int type_ij = radial_pair_index(type_i, type_j);
            A0_scale_factor += A0_splines[type_ij].evaluate(r[ij]);
            ij += 1;
        }
        auto A0_i = A0.data()+i*num_lm*num_channels;
        for (int lmk=0; lmk<num_lm*num_channels; ++lmk)
            A0_i[lmk] /= A0_scale_factor;
    }
}

template <typename Precision>
void MACECPU<Precision>::reverse_A0_scaled(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r)
{
    if (not A0_scaled) return;
    // Warning: Assumes node_forces have been initialized elsewhere
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        const int type_i = node_types[i];
        auto A0_i = A0.data()+i*num_lm*num_channels;
        auto A0_adj_i = A0_adj.data()+i*num_lm*num_channels;
        // recompute the scale factor
        Precision A0_scale_factor = 1.0;
        for (int j=0; j<num_neigh[i]; ++j) {
            const int type_j = neigh_types[ij];
            const int type_ij = radial_pair_index(type_i, type_j);
            A0_scale_factor += A0_splines[type_ij].evaluate(r[ij]);
            ij += 1;
        }
        // update dE/dxyz
        Precision dA0_dot_A0 = 0.0;
        for (int lmk=0; lmk<num_lm*num_channels; ++lmk)
            dA0_dot_A0 += A0_adj_i[lmk] * A0_i[lmk];
        ij = ij - num_neigh[i];
        for (int j=0; j<num_neigh[i]; ++j) {
            const int type_j = neigh_types[ij];
            const int type_ij = radial_pair_index(type_i, type_j);
            auto [f,d] = A0_splines[type_ij].evaluate_deriv(r[ij]);
            const auto xyz_ij = xyz.data()+ij*3;
            const Precision direction[3] = {
                static_cast<Precision>(xyz_ij[0]/r[ij]),
                static_cast<Precision>(xyz_ij[1]/r[ij]),
                static_cast<Precision>(xyz_ij[2]/r[ij]),
            };
            auto node_forces_ij = node_forces.data()+ij*3;
            const Precision force_scale = dA0_dot_A0/A0_scale_factor*d;
            node_forces_ij[0] += force_scale*direction[0];
            node_forces_ij[1] += force_scale*direction[1];
            node_forces_ij[2] += force_scale*direction[2];
            ij += 1;
        }
        // update dE/dA0
        for (int lmk=0; lmk<num_lm*num_channels; ++lmk)
            A0_adj_i[lmk] /= A0_scale_factor;
    }
}

template <typename Precision>
void MACECPU<Precision>::compute_M0(
    const int num_nodes,
    std::span<const int> node_types)
{
    M0.resize(static_cast<std::size_t>(num_nodes)*num_LM*num_channels);
    M0_grad.resize(
        static_cast<std::size_t>(num_nodes)*num_channels*num_LM*num_lm);
    for (int i=0; i<num_nodes; ++i) {
        auto A0_i = A0.data()+i*num_lm*num_channels;
        auto M0_i = M0.data()+i*num_LM*num_channels;
        auto M0_grad_i = M0_grad.data()+i*num_LM*num_channels*num_lm;
        auto x = std::vector<Precision>(num_lm);
        int lmk = 0;
        for (int lm=0; lm<num_LM; ++lm) {
            for (int k=0; k<num_channels; ++k) {
                symmetrix_blas_copy<Precision>(num_lm, A0_i+k, num_channels, x.data(), 1);
                auto [f,g] = P0[node_types[i]*num_LM*num_channels+lmk].evaluate_gradient(x);
                M0_i[lmk] = f;
                symmetrix_blas_copy<Precision>(num_lm, g.data(), 1, M0_grad_i+lm*num_lm*num_channels+k, num_channels);
                lmk += 1;
            }
        }
    }
}

template <typename Precision>
void MACECPU<Precision>::reverse_M0(
    const int num_nodes,
    std::span<const int> node_types)
{
    A0_adj.resize(A0.size());
    std::fill(A0_adj.begin(), A0_adj.end(), 0.0);
    for (int i=0; i<num_nodes; ++i) {
        auto A0_adj_i = A0_adj.data()+i*num_lm*num_channels;
        auto M0_adj_i = M0_adj.data()+i*num_LM*num_channels;
        auto M0_grad_i = M0_grad.data()+i*num_LM*num_lm*num_channels;
        for (int lm=0; lm<num_lm; ++lm) {
            auto A0_adj_ilm = A0_adj_i + lm*num_channels;
            for (int lmp=0; lmp<num_LM; ++lmp) {
                auto M0_adj_ilmp = M0_adj_i + lmp*num_channels;
                auto M0_grad_ilmplm = M0_grad_i +
                    + lmp*num_lm*num_channels
                    + lm*num_channels;
                for (int k=0; k<num_channels; ++k) {
                    A0_adj_ilm[k] += M0_grad_ilmplm[k] * M0_adj_ilmp[k];
                }
            }
        }
    }
}

template <typename Precision>
void MACECPU<Precision>::compute_H1(
    const int num_nodes)
{
    H1.resize(M0.size());
    for (int i=0; i<num_nodes; ++i) {
        for (int l=0; l<=L_max; ++l) {
            const auto M0_il = M0.data()+(i*num_LM+l*l)*num_channels;
            const auto H1_weights_l = H1_weights.data()+l*num_channels*num_channels;
            auto H1_il = H1.data()+(i*num_LM+l*l)*num_channels;
            symmetrix_blas_gemm<Precision>(
                CblasRowMajor,  // const CBLAS_LAYOUT Layout
                CblasNoTrans,   // const CBLAS_TRANSPOSE transa
                CblasNoTrans,   // const CBLAS_TRANSPOSE transb
                2*l+1,          // const MKL_INT m
                num_channels,   // const MKL_INT n
                num_channels,   // const MKL_INT k
                1.0,            // const double alpha
                M0_il,          // const double *a
                num_channels,   // const MKL_INT lda
                H1_weights_l,   // const double *b
                num_channels,   // const MKL_INT ldb
                0.0,            // const double beta
                H1_il,          // double *c
                num_channels);  // const MKL_INT ldc
        }
    }
}

template <typename Precision>
void MACECPU<Precision>::reverse_H1(
    const int num_nodes)
{
    M0_adj.resize(M0.size());
    for (int i=0; i<num_nodes; ++i) {
        for (int l=0; l<=L_max; ++l) {
            const auto H1_adj_il = H1_adj.data()+(i*num_LM+l*l)*num_channels;
            const auto H1_weights_l = H1_weights.data()+l*num_channels*num_channels;
            auto M0_adj_il = M0_adj.data()+(i*num_LM+l*l)*num_channels;
            symmetrix_blas_gemm<Precision>(
                CblasRowMajor,  // const CBLAS_LAYOUT Layout
                CblasNoTrans,   // const CBLAS_TRANSPOSE transa
                CblasTrans,     // const CBLAS_TRANSPOSE transb
                2*l+1,          // const MKL_INT m
                num_channels,   // const MKL_INT n
                num_channels,   // const MKL_INT k
                1.0,            // const double alpha
                H1_adj_il,      // const double *a
                num_channels,   // const MKL_INT lda
                H1_weights_l,   // const double *b
                num_channels,   // const MKL_INT ldb
                0.0,            // const double beta
                M0_adj_il,      // double *c
                num_channels);  // const MKL_INT ldc
        }
    }
}

template <typename Precision>
void MACECPU<Precision>::compute_H1_product(
    const int num_nodes)
{
    H1.resize(M0.size());
    for (int i=0; i<num_nodes; ++i) {
        for (int l=0; l<=L_max; ++l) {
            const auto M0_il = M0.data()+(i*num_LM+l*l)*num_channels;
            const auto weights_l = H1_product_weights.data()+l*num_channels*num_channels;
            auto H1_il = H1.data()+(i*num_LM+l*l)*num_channels;
            symmetrix_blas_gemm<Precision>(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                2*l+1,
                num_channels,
                num_channels,
                1.0,
                M0_il,
                num_channels,
                weights_l,
                num_channels,
                0.0,
                H1_il,
                num_channels);
        }
    }
}

template <typename Precision>
void MACECPU<Precision>::add_H1_first_residual(
    const int num_nodes,
    std::span<const int> node_types,
    const bool fused)
{
    if (!first_interaction_residual)
        return;
    const auto& weights = fused
        ? H1_first_residual_fused_weights
        : H1_first_residual_weights;
    for (int i=0; i<num_nodes; ++i) {
        const auto offset = static_cast<std::size_t>(node_types[i])
            *num_LM*num_channels;
        for (int lm=0; lm<num_LM; ++lm)
            for (int k=0; k<num_channels; ++k)
                H1[(i*num_LM+lm)*num_channels+k]
                    += weights[offset+lm*num_channels+k];
    }
}

template <typename Precision>
void MACECPU<Precision>::compute_H1_linear_up(
    const int num_nodes)
{
    auto H1_before_linear_up = H1;
    for (int i=0; i<num_nodes; ++i) {
        for (int l=0; l<=L_max; ++l) {
            const auto H1_in_il = H1_before_linear_up.data()+(i*num_LM+l*l)*num_channels;
            const auto weights_l = H1_linear_up_weights.data()+l*num_channels*num_channels;
            auto H1_il = H1.data()+(i*num_LM+l*l)*num_channels;
            symmetrix_blas_gemm<Precision>(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                2*l+1,
                num_channels,
                num_channels,
                1.0,
                H1_in_il,
                num_channels,
                weights_l,
                num_channels,
                0.0,
                H1_il,
                num_channels);
        }
    }
}

template <typename Precision>
void MACECPU<Precision>::reverse_H1_linear_up(
    const int num_nodes)
{
    auto H1_adj_before_linear_up = H1_adj;
    for (int i=0; i<num_nodes; ++i) {
        for (int l=0; l<=L_max; ++l) {
            const auto H1_adj_il = H1_adj_before_linear_up.data()+(i*num_LM+l*l)*num_channels;
            const auto weights_l = H1_linear_up_weights.data()+l*num_channels*num_channels;
            auto H1_pre_adj_il = H1_adj.data()+(i*num_LM+l*l)*num_channels;
            symmetrix_blas_gemm<Precision>(
                CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                2*l+1,
                num_channels,
                num_channels,
                1.0,
                H1_adj_il,
                num_channels,
                weights_l,
                num_channels,
                0.0,
                H1_pre_adj_il,
                num_channels);
        }
    }
}

template <typename Precision>
void MACECPU<Precision>::reverse_H1_product(
    const int num_nodes)
{
    M0_adj.resize(M0.size());
    for (int i=0; i<num_nodes; ++i) {
        for (int l=0; l<=L_max; ++l) {
            const auto H1_adj_il = H1_adj.data()+(i*num_LM+l*l)*num_channels;
            const auto weights_l = H1_product_weights.data()+l*num_channels*num_channels;
            auto M0_adj_il = M0_adj.data()+(i*num_LM+l*l)*num_channels;
            symmetrix_blas_gemm<Precision>(
                CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                2*l+1,
                num_channels,
                num_channels,
                1.0,
                H1_adj_il,
                num_channels,
                weights_l,
                num_channels,
                0.0,
                M0_adj_il,
                num_channels);
        }
    }
}

template <typename Precision>
void MACECPU<Precision>::compute_Phi1(
    const int num_nodes,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices)
{
    // Compute Phi1_lelm1lm2 (named Phi1r)
    Phi1r.resize(
        static_cast<std::size_t>(num_nodes)*num_lelm1lm2*num_channels);
    std::fill(Phi1r.begin(), Phi1r.end(), 0.0);
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        auto Phi1r_i = Phi1r.data()+i*num_lelm1lm2*num_channels;
        for (int j=0; j<num_neigh[i]; ++j) {
            auto R1_ij = R1.data()+ij*spl_set_1[0]->num_splines;
            auto Y_ij = Y.data()+ij*num_lm;
            auto H1_ij = H1.data()+neigh_indices[ij]*num_LM*num_channels;
            int lelm1lm2 = 0;
            for (int lel1l2=0; lel1l2<Phi1_l.size(); ++lel1l2) {
                const int l1 = Phi1_l1[lel1l2];
                const int l2 = Phi1_l2[lel1l2];
                auto R1_ij_lel1l2 = R1_ij+lel1l2*num_channels;
                for (int lm1=l1*l1; lm1<=l1*(l1+2); ++lm1) {
                    const Precision Y_ij_lm1 = Y_ij[lm1];
                    for (int lm2=l2*l2; lm2<=l2*(l2+2); ++lm2) {
                        auto H1_ij_lm2 = H1_ij+lm2*num_channels;
                        auto Phi1r_i_lelm1lm2 = Phi1r_i+lelm1lm2*num_channels;
                        for (int k=0; k<num_channels; ++k) {
                            Phi1r_i_lelm1lm2[k] += R1_ij_lel1l2[k] * Y_ij_lm1 * H1_ij_lm2[k];
                        }
                        lelm1lm2 += 1;
                    }
                }
            }
            ij += 1;
        }
    }
    // Compute Phi1 using CG coefficients
    Phi1.resize(static_cast<std::size_t>(num_nodes)*num_lme*num_channels);
    std::fill(Phi1.begin(), Phi1.end(), 0.0);
    for (int i=0; i<num_nodes; ++i) {
        auto Phi1_i = Phi1.data()+i*num_lme*num_channels;
        auto Phi1r_i = Phi1r.data()+i*num_lelm1lm2*num_channels;
        for (int p=0; p<Phi1_clebsch_gordan.size(); ++p) {
            auto Phi1_i_lme = Phi1_i+Phi1_lme[p]*num_channels;
            const Precision C = Phi1_clebsch_gordan[p];
            auto Phi1r_i_lelm1lm2 = Phi1r_i+Phi1_lelm1lm2[p]*num_channels;
            for (int k=0; k<num_channels; ++k)
                Phi1_i_lme[k] += C * Phi1r_i_lelm1lm2[k];
        }
    }
}

template <typename Precision>
void MACECPU<Precision>::compute_Phi1_streamed(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const int> neigh_types,
    std::span<const double> r)
{
    Phi1r.assign(
        static_cast<std::size_t>(num_nodes)*num_lelm1lm2*num_channels, 0.0);
    auto radial_values = std::vector<Precision>(num_channels);
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        auto Phi1r_i = Phi1r.data()+i*num_lelm1lm2*num_channels;
        const int type_i = node_types[i];
        for (int j=0; j<num_neigh[i]; ++j) {
            const int pair = radial_pair_index(type_i, neigh_types[ij]);
            const auto& spline = *spl_set_1[pair];
            const auto point = spline.evaluation_point(r[ij]);
            const auto Y_ij = Y.data()+ij*num_lm;
            const auto H1_ij =
                H1.data()+neigh_indices[ij]*num_LM*num_channels;
            int lelm1lm2 = 0;
            for (int lel1l2=0; lel1l2<Phi1_l.size(); ++lel1l2) {
                for (int k=0; k<num_channels; ++k)
                    radial_values[k] = spline.evaluate_function(
                        point, lel1l2*num_channels+k);
                const int l1 = Phi1_l1[lel1l2];
                const int l2 = Phi1_l2[lel1l2];
                for (int lm1=l1*l1; lm1<=l1*(l1+2); ++lm1) {
                    const Precision Y_ij_lm1 = Y_ij[lm1];
                    for (int lm2=l2*l2; lm2<=l2*(l2+2); ++lm2) {
                        const auto H1_ij_lm2 = H1_ij+lm2*num_channels;
                        auto Phi1r_i_row =
                            Phi1r_i+lelm1lm2*num_channels;
                        for (int k=0; k<num_channels; ++k)
                            Phi1r_i_row[k] +=
                                radial_values[k]*Y_ij_lm1*H1_ij_lm2[k];
                        lelm1lm2 += 1;
                    }
                }
            }
            ij += 1;
        }
    }

    Phi1.assign(
        static_cast<std::size_t>(num_nodes)*num_lme*num_channels, 0.0);
    for (int i=0; i<num_nodes; ++i) {
        auto Phi1_i = Phi1.data()+i*num_lme*num_channels;
        auto Phi1r_i = Phi1r.data()+i*num_lelm1lm2*num_channels;
        for (int p=0; p<Phi1_clebsch_gordan.size(); ++p) {
            auto Phi1_i_lme = Phi1_i+Phi1_lme[p]*num_channels;
            const Precision coefficient = Phi1_clebsch_gordan[p];
            const auto Phi1r_i_row =
                Phi1r_i+Phi1_lelm1lm2[p]*num_channels;
            for (int k=0; k<num_channels; ++k)
                Phi1_i_lme[k] += coefficient*Phi1r_i_row[k];
        }
    }
}

template <typename Precision>
void MACECPU<Precision>::reverse_Phi1(
    const int num_nodes,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const double> xyz,
    std::span<const double> r,
    bool zero_dxyz,
    bool zero_H1_adj)
{
    // Compute dE/dPhi1 (named dPhi1)
    dPhi1r.resize(Phi1r.size());
    std::fill(dPhi1r.begin(), dPhi1r.end(), 0.0);
    for (int i=0; i<num_nodes; ++i) {
        auto dPhi1r_i = dPhi1r.data()+i*num_lelm1lm2*num_channels;
        auto dPhi1_i = dPhi1.data()+i*num_lme*num_channels;
        for (int p=0; p<Phi1_clebsch_gordan.size(); ++p) {
            auto dPhi1r_i_lelm1lm2 = dPhi1r_i+Phi1_lelm1lm2[p]*num_channels;
            const Precision C = Phi1_clebsch_gordan[p];
            auto dPhi1_i_lme = dPhi1_i+Phi1_lme[p]*num_channels;
            for (int k=0; k<num_channels; ++k)
                dPhi1r_i_lelm1lm2[k] += C * dPhi1_i_lme[k];
        }
    }
    // Compute partial forces
    node_forces.resize(xyz.size());
    if (zero_dxyz)
        std::fill(node_forces.begin(), node_forces.end(), 0.0);
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        auto dPhi1r_i = dPhi1r.data()+i*num_lelm1lm2*num_channels;
        for (int j=0; j<num_neigh[i]; ++j) {
            auto node_forces_ij = node_forces.data()+3*ij;
            auto xyz_ij = xyz.data()+3*ij;
            auto r_ij = r[ij];
            const Precision direction_x = static_cast<Precision>(xyz_ij[0]/r_ij);
            const Precision direction_y = static_cast<Precision>(xyz_ij[1]/r_ij);
            const Precision direction_z = static_cast<Precision>(xyz_ij[2]/r_ij);
            auto R1_ij = R1.data()+ij*spl_set_1[0]->num_splines;
            auto R1_deriv_ij = R1_deriv.data()+ij*spl_set_1[0]->num_splines;
            auto Y_ij = Y.data()+ij*num_lm;
            auto Y_grad_ij = Y_grad.data()+ij*3*num_lm;
            auto H1_ij = H1.data()+neigh_indices[ij]*num_LM*num_channels;
            Precision force_x = 0.0;
            Precision force_y = 0.0;
            Precision force_z = 0.0;
            int lelm1lm2 = 0;
            for (int lel1l2=0; lel1l2<Phi1_l.size(); ++lel1l2) {
                const int l1 = Phi1_l1[lel1l2];
                const int l2 = Phi1_l2[lel1l2];
                auto R1_ij_lel1l2 = R1_ij+lel1l2*num_channels;
                auto R1_deriv_ij_lel1l2 = R1_deriv_ij+lel1l2*num_channels;
                for (int lm1=l1*l1; lm1<=l1*(l1+2); ++lm1) {
                    const Precision Y_ij_lm1 = Y_ij[lm1];
                    const Precision Y_grad_ij_x_lm1 = Y_grad_ij[0*num_lm+lm1];
                    const Precision Y_grad_ij_y_lm1 = Y_grad_ij[1*num_lm+lm1];
                    const Precision Y_grad_ij_z_lm1 = Y_grad_ij[2*num_lm+lm1];
                    for (int lm2=l2*l2; lm2<=l2*(l2+2); ++lm2) {
                        auto H1_ij_lm2 = H1_ij+lm2*num_channels;
                        auto dPhi1r_i_lelm1lm2 = dPhi1r_i+lelm1lm2*num_channels;
                        for (int k=0; k<num_channels; ++k) {
                            force_x += -dPhi1r_i_lelm1lm2[k] * (
                                direction_x * R1_deriv_ij_lel1l2[k] * Y_ij_lm1 * H1_ij_lm2[k]
                                    + R1_ij_lel1l2[k] * Y_grad_ij_x_lm1 * H1_ij_lm2[k]);
                            force_y += -dPhi1r_i_lelm1lm2[k] * (
                                direction_y * R1_deriv_ij_lel1l2[k] * Y_ij_lm1 * H1_ij_lm2[k]
                                    + R1_ij_lel1l2[k] * Y_grad_ij_y_lm1 * H1_ij_lm2[k]);
                            force_z += -dPhi1r_i_lelm1lm2[k] * (
                                direction_z * R1_deriv_ij_lel1l2[k] * Y_ij_lm1 * H1_ij_lm2[k]
                                    + R1_ij_lel1l2[k] * Y_grad_ij_z_lm1 * H1_ij_lm2[k]);
                        }
                        lelm1lm2 += 1;
                    }
                }
            }
            node_forces_ij[0] += force_x;
            node_forces_ij[1] += force_y;
            node_forces_ij[2] += force_z;
            ij += 1;
        }
    }
    // Compute dE/dH1 (named dH1)
    H1_adj.resize(H1.size());
    if (zero_H1_adj)
        std::fill(H1_adj.begin(), H1_adj.end(), 0.0);
    ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        auto dPhi1r_i = dPhi1r.data()+i*num_lelm1lm2*num_channels;
        for (int j=0; j<num_neigh[i]; ++j) {
            auto R1_ij = R1.data()+ij*spl_set_1[0]->num_splines;
            auto Y_ij = Y.data()+ij*num_lm;
            auto H1_adj_ij = H1_adj.data()+neigh_indices[ij]*num_LM*num_channels;
            int lelm1lm2 = 0;
            for (int lel1l2=0; lel1l2<Phi1_l.size(); ++lel1l2) {
                const int l1 = Phi1_l1[lel1l2];
                const int l2 = Phi1_l2[lel1l2];
                auto R1_ij_lel1l2 = R1_ij+lel1l2*num_channels;
                for (int lm1=l1*l1; lm1<=l1*(l1+2); ++lm1) {
                    for (int lm2=l2*l2; lm2<=l2*(l2+2); ++lm2) {
                        auto H1_adj_ij_lm2 = H1_adj_ij+lm2*num_channels;
                        auto dPhi1r_i_lelm1lm2 = dPhi1r_i+lelm1lm2*num_channels;
                        for (int k=0; k<num_channels; ++k) {
                            H1_adj_ij_lm2[k] += R1_ij_lel1l2[k]*Y_ij[lm1]*dPhi1r_i_lelm1lm2[k];
                        }
                        lelm1lm2 += 1;
                    }
                }
            }
            ij += 1;
        }
    }
}

template <typename Precision>
void MACECPU<Precision>::reverse_Phi1_streamed(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r,
    bool zero_dxyz,
    bool zero_H1_adj)
{
    dPhi1r.assign(Phi1r.size(), 0.0);
    for (int i=0; i<num_nodes; ++i) {
        auto dPhi1r_i = dPhi1r.data()+i*num_lelm1lm2*num_channels;
        const auto dPhi1_i = dPhi1.data()+i*num_lme*num_channels;
        for (int p=0; p<Phi1_clebsch_gordan.size(); ++p) {
            auto dPhi1r_i_row =
                dPhi1r_i+Phi1_lelm1lm2[p]*num_channels;
            const Precision coefficient = Phi1_clebsch_gordan[p];
            const auto dPhi1_i_lme = dPhi1_i+Phi1_lme[p]*num_channels;
            for (int k=0; k<num_channels; ++k)
                dPhi1r_i_row[k] += coefficient*dPhi1_i_lme[k];
        }
    }

    node_forces.resize(xyz.size());
    if (zero_dxyz)
        std::fill(node_forces.begin(), node_forces.end(), 0.0);
    H1_adj.resize(H1.size());
    if (zero_H1_adj)
        std::fill(H1_adj.begin(), H1_adj.end(), 0.0);

    auto radial_values = std::vector<Precision>(num_channels);
    auto radial_derivatives = std::vector<Precision>(num_channels);
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        const auto dPhi1r_i =
            dPhi1r.data()+i*num_lelm1lm2*num_channels;
        const int type_i = node_types[i];
        for (int j=0; j<num_neigh[i]; ++j) {
            const int pair = radial_pair_index(type_i, neigh_types[ij]);
            const auto& spline = *spl_set_1[pair];
            const auto point = spline.evaluation_point(r[ij]);
            auto node_forces_ij = node_forces.data()+3*ij;
            const auto xyz_ij = xyz.data()+3*ij;
            const Precision direction_x = static_cast<Precision>(xyz_ij[0]/r[ij]);
            const Precision direction_y = static_cast<Precision>(xyz_ij[1]/r[ij]);
            const Precision direction_z = static_cast<Precision>(xyz_ij[2]/r[ij]);
            const auto Y_ij = Y.data()+ij*num_lm;
            const auto Y_grad_ij = Y_grad.data()+ij*3*num_lm;
            const auto H1_ij =
                H1.data()+neigh_indices[ij]*num_LM*num_channels;
            auto H1_adj_ij =
                H1_adj.data()+neigh_indices[ij]*num_LM*num_channels;
            Precision force_x = 0.0;
            Precision force_y = 0.0;
            Precision force_z = 0.0;
            int lelm1lm2 = 0;
            for (int lel1l2=0; lel1l2<Phi1_l.size(); ++lel1l2) {
                for (int k=0; k<num_channels; ++k) {
                    spline.evaluate_function_derivs(
                        point, lel1l2*num_channels+k,
                        radial_values[k], radial_derivatives[k]);
                }
                const int l1 = Phi1_l1[lel1l2];
                const int l2 = Phi1_l2[lel1l2];
                for (int lm1=l1*l1; lm1<=l1*(l1+2); ++lm1) {
                    const Precision Y_ij_lm1 = Y_ij[lm1];
                    for (int lm2=l2*l2; lm2<=l2*(l2+2); ++lm2) {
                        const auto H1_ij_lm2 = H1_ij+lm2*num_channels;
                        auto H1_adj_ij_lm2 =
                            H1_adj_ij+lm2*num_channels;
                        const auto dPhi1r_i_row =
                            dPhi1r_i+lelm1lm2*num_channels;
                        for (int k=0; k<num_channels; ++k) {
                            const Precision adjoint = dPhi1r_i_row[k];
                            const Precision source_feature = H1_ij_lm2[k];
                            const Precision radial_force =
                                radial_derivatives[k]*Y_ij_lm1*source_feature;
                            const Precision angular_force =
                                radial_values[k]*source_feature;
                            force_x -= adjoint*(
                                direction_x*radial_force
                                +angular_force*Y_grad_ij[lm1]);
                            force_y -= adjoint*(
                                direction_y*radial_force
                                +angular_force*Y_grad_ij[num_lm+lm1]);
                            force_z -= adjoint*(
                                direction_z*radial_force
                                +angular_force*Y_grad_ij[2*num_lm+lm1]);
                            H1_adj_ij_lm2[k] +=
                                radial_values[k]*Y_ij_lm1*adjoint;
                        }
                        lelm1lm2 += 1;
                    }
                }
            }
            node_forces_ij[0] += force_x;
            node_forces_ij[1] += force_y;
            node_forces_ij[2] += force_z;
            ij += 1;
        }
    }
}

template <typename Precision>
void MACECPU<Precision>::compute_A1(
    const int num_nodes)
{
    // The core matrix multiplication is:
    //         [A1_il]_mk = \sum_k' [Phi1_il]_m(ek') [W_il]_(ek')k
    A1.resize(static_cast<std::size_t>(num_nodes)*num_lm*num_channels);
    int num_lme = 0;
    std::vector<int> num_e(l_max+1,0);
    for (auto l : Phi1_l) {
        num_lme += 2*l+1;
        num_e[l] += 1;
    }
    for (int i=0; i<num_nodes; ++i) {
        auto Phi1_il = Phi1.data()+i*num_lme*num_channels;
        auto A1_il = A1.data()+i*num_lm*num_channels;
        for (int l=0; l<=l_max; ++l) {
            symmetrix_blas_gemm<Precision>(
                CblasRowMajor,          // const CBLAS_LAYOUT Layout
                CblasNoTrans,           // const CBLAS_TRANSPOSE transa
                CblasNoTrans,           // const CBLAS_TRANSPOSE transb
                (2*l+1),                // const MKL_INT m
                num_channels,           // const MKL_INT n
                num_e[l]*num_channels,  // const MKL_INT k
                1.0,                    // const double alpha
                Phi1_il,                // const double *a
                num_e[l]*num_channels,  // const MKL_INT lda
                A1_weights[l].data(),   // const double *b
                num_channels,           // const MKL_INT ldb
                0.0,                    // const double beta
                A1_il,                  // double *c
                num_channels);          // const MKL_INT ldc
            Phi1_il += (2*l+1)*num_e[l]*num_channels;
            A1_il += (2*l+1)*num_channels;
        }
    }
}

template <typename Precision>
void MACECPU<Precision>::reverse_A1(
    const int num_nodes)
{
    // The core matrix multiplication is:
    //         [dE/dPhi1_il]_m(ek) = \sum_k' [dE/dA1_il]_mk' [trans(W_il)]_k'(ek)
    dPhi1.resize(Phi1.size());
    int num_lme = 0;
    std::vector<int> num_e(l_max+1,0);
    for (auto l : Phi1_l) {
        num_lme += 2*l+1;
        num_e[l] += 1;
    }
    for (int i=0; i<num_nodes; ++i) {
        auto A1_adj_il = A1_adj.data()+i*num_lm*num_channels;
        auto dPhi1_il = dPhi1.data()+i*num_lme*num_channels;
        for (int l=0; l<=l_max; ++l) {
            symmetrix_blas_gemm<Precision>(
                CblasRowMajor,          // const CBLAS_LAYOUT Layout
                CblasNoTrans,           // const CBLAS_TRANSPOSE transa
                CblasTrans,             // const CBLAS_TRANSPOSE transb
                (2*l+1),                // const MKL_INT m
                num_e[l]*num_channels,  // const MKL_INT n
                num_channels,           // const MKL_INT k
                1.0,                    // const double alpha
                A1_adj_il,              // const double *a
                num_channels,           // const MKL_INT lda
                A1_weights[l].data(),   // const double *b
                num_channels,           // const MKL_INT ldb
                0.0,                    // const double beta
                dPhi1_il,            // double *c
                num_e[l]*num_channels); // const MKL_INT ldc
            A1_adj_il += (2*l+1)*num_channels;
            dPhi1_il += (2*l+1)*num_e[l]*num_channels;
        }
    }
}

template <typename Precision>
void MACECPU<Precision>::compute_A1_scaled(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> r)
{
    if (not A1_scaled) return;
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        const int type_i = node_types[i];
        Precision A1_scale_factor = 1.0;
        for (int j=0; j<num_neigh[i]; ++j) {
            const int type_j = neigh_types[ij];
            const int type_ij = radial_pair_index(type_i, type_j);
            A1_scale_factor += A1_splines[type_ij].evaluate(r[ij]);
            ij += 1;
        }
        auto A1_i = A1.data()+i*num_lm*num_channels;
        for (int lmk=0; lmk<num_lm*num_channels; ++lmk)
            A1_i[lmk] /= A1_scale_factor;
    }
}

template <typename Precision>
void MACECPU<Precision>::reverse_A1_scaled(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r,
    bool zero_dxyz)
{
    if (not A1_scaled) return;
    node_forces.resize(xyz.size());
    if (zero_dxyz)
        std::fill(node_forces.begin(), node_forces.end(), 0.0);
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        const int type_i = node_types[i];
        auto A1_i = A1.data()+i*num_lm*num_channels;
        auto A1_adj_i = A1_adj.data()+i*num_lm*num_channels;
        // recompute the scale factor
        Precision A1_scale_factor = 1.0;
        for (int j=0; j<num_neigh[i]; ++j) {
            const int type_j = neigh_types[ij];
            const int type_ij = radial_pair_index(type_i, type_j);
            A1_scale_factor += A1_splines[type_ij].evaluate(r[ij]);
            ij += 1;
        }
        // update dE/dxyz
        Precision dA1_dot_A1 = 0.0;
        for (int lmk=0; lmk<num_lm*num_channels; ++lmk)
            dA1_dot_A1 += A1_adj_i[lmk] * A1_i[lmk];
        ij = ij - num_neigh[i];
        for (int j=0; j<num_neigh[i]; ++j) {
            const int type_j = neigh_types[ij];
            const int type_ij = radial_pair_index(type_i, type_j);
            auto [f,d] = A1_splines[type_ij].evaluate_deriv(r[ij]);
            const auto xyz_ij = xyz.data()+ij*3;
            const Precision direction[3] = {
                static_cast<Precision>(xyz_ij[0]/r[ij]),
                static_cast<Precision>(xyz_ij[1]/r[ij]),
                static_cast<Precision>(xyz_ij[2]/r[ij]),
            };
            auto node_forces_ij = node_forces.data()+ij*3;
            const Precision force_scale = dA1_dot_A1/A1_scale_factor*d;
            node_forces_ij[0] += force_scale*direction[0];
            node_forces_ij[1] += force_scale*direction[1];
            node_forces_ij[2] += force_scale*direction[2];
            ij += 1;
        }
        // update dE/dA1
        for (int lmk=0; lmk<num_lm*num_channels; ++lmk)
            A1_adj_i[lmk] /= A1_scale_factor;
    }
}

template <typename Precision>
void MACECPU<Precision>::compute_M1(
    const int num_nodes,
    std::span<const int> node_types)
{
    const int harmonic_count = single_layer_readout ? num_LM : num_lm;
    const auto* polynomial_input = single_layer_readout ? H1.data() : A1.data();
    M1.resize(static_cast<std::size_t>(num_nodes)*num_channels);
    M1_grad.resize(
        static_cast<std::size_t>(num_nodes)*harmonic_count*num_channels);
    for (int i=0; i<num_nodes; ++i) {
        auto input_i = polynomial_input+i*harmonic_count*num_channels;
        auto M1_i = M1.data()+i*num_channels;
        auto M1_grad_i = M1_grad.data()+i*num_channels*harmonic_count;
        auto x = std::vector<Precision>(harmonic_count);
        for (int k=0; k<num_channels; ++k) {
            symmetrix_blas_copy<Precision>(
                harmonic_count, input_i+k, num_channels, x.data(), 1);
            auto [f,g] = P1[node_types[i]*num_channels+k].evaluate_gradient(x);
            M1_i[k] = f;
            symmetrix_blas_copy<Precision>(
                harmonic_count, g.data(), 1, M1_grad_i+k, num_channels);
        }
    }
}

template <typename Precision>
void MACECPU<Precision>::reverse_M1(
    const int num_nodes,
    std::span<const int> node_types)
{
    const int harmonic_count = single_layer_readout ? num_LM : num_lm;
    if (single_layer_readout)
        H1_adj.resize(H1.size());
    else
        A1_adj.resize(A1.size());
    for (int i=0; i<num_nodes; ++i) {
        auto M1_adj_i = M1_adj.data() + i*num_channels;
        for (int lm=0; lm<harmonic_count; ++lm) {
            auto* input_adj = single_layer_readout
                ? H1_adj.data() + (i*num_LM+lm)*num_channels
                : A1_adj.data() + (i*num_lm+lm)*num_channels;
            auto M1_grad_ilm =
                M1_grad.begin() + (i*harmonic_count+lm)*num_channels;
            for (int k=0; k<num_channels; ++k) {
                const auto adjoint = M1_grad_ilm[k] * M1_adj_i[k];
                if (single_layer_readout)
                    input_adj[k] += adjoint;
                else
                    input_adj[k] = adjoint;
            }
        }
    }
}

template <typename Precision>
void MACECPU<Precision>::compute_H2(
    const int num_nodes,
    std::span<const int> node_types)
{
    H2.resize(static_cast<std::size_t>(num_nodes)*num_channels);
    for (int i=0; i<num_nodes; ++i) {
        auto H2_i = H2.data()+i*num_channels;
        auto H1_i = H1.data()+i*num_LM*num_channels;
        symmetrix_blas_gemv<Precision>(
            CblasRowMajor,                            // const CBLAS_LAYOUT Layout
            CblasTrans,                               // const CBLAS_TRANSPOSE trans
            num_channels,                             // const MKL_INT m
            num_channels,                             // const MKL_INT n
            1.0,                                      // const double alpha
            H2_weights_for_H1[node_types[i]].data(),  // const double *a
            num_channels,                             // const MKL_INT lda
            H1_i,                                     // const double *x
            1,                                        // const MKL_INT incx
            0.0,                                      // const double beta
            H2_i,                                     // double *y
            1);                                       // const MKL_INT incy
        auto M1_i = M1.data()+i*num_channels;
        symmetrix_blas_gemv<Precision>(
            CblasRowMajor,             // const CBLAS_LAYOUT Layout
            CblasTrans,                // const CBLAS_TRANSPOSE trans
            num_channels,              // const MKL_INT m
            num_channels,              // const MKL_INT n
            1.0,                       // const double alpha
            H2_weights_for_M1.data(),  // const double *a
            num_channels,              // const MKL_INT lda
            M1_i,                      // const double *x
            1,                         // const MKL_INT incx
            1.0,                       // const double beta
            H2_i,                      // double *y
            1);                        // const MKL_INT incy
    }
}

template <typename Precision>
void MACECPU<Precision>::reverse_H2(
    const int num_nodes,
    std::span<const int> node_types,
    bool zero_H1_adj)
{
    H1_adj.resize(H1.size());
    M1_adj.resize(M1.size());
    if (zero_H1_adj)
        std::fill(H1_adj.begin(), H1_adj.end(), 0.0);
    for (int i=0; i<num_nodes; ++i) {
        auto H2_adj_i = H2_adj.data()+i*num_channels;
        auto H1_adj_i = H1_adj.data()+i*num_LM*num_channels;
        symmetrix_blas_gemv<Precision>(
            CblasRowMajor,                            // const CBLAS_LAYOUT Layout
            CblasNoTrans,                             // const CBLAS_TRANSPOSE trans
            num_channels,                             // const MKL_INT m
            num_channels,                             // const MKL_INT n
            1.0,                                      // const double alpha
            H2_weights_for_H1[node_types[i]].data(),  // const double *a
            num_channels,                             // const MKL_INT lda
            H2_adj_i,                                 // const double *x
            1,                                        // const MKL_INT incx
            1.0,                                      // const double beta
            H1_adj_i,                                 // double *y
            1);                                       // const MKL_INT incy
        auto M1_adj_i = M1_adj.data()+i*num_channels;
        symmetrix_blas_gemv<Precision>(
            CblasRowMajor,             // const CBLAS_LAYOUT Layout
            CblasNoTrans,              // const CBLAS_TRANSPOSE trans
            num_channels,              // const MKL_INT m
            num_channels,              // const MKL_INT n
            1.0,                       // const double alpha
            H2_weights_for_M1.data(),  // const double *a
            num_channels,              // const MKL_INT lda
            H2_adj_i,                  // const double *x
            1,                         // const MKL_INT incx
            0.0,                       // const double beta
            M1_adj_i,                  // double *y
            1);                        // const MKL_INT incy
    }
}

template <typename Precision>
void MACECPU<Precision>::compute_readouts(
    const int num_nodes,
    std::span<const int> node_types)
{
    node_energies.resize(num_nodes);
    H1_adj.resize(H1.size());
    // Warning: Although it doesn't appear necessary to set H1_adj to zero,
    //          it matters when the number of nodes associated with H1 is greater than num_nodes.
    //          There is probably a better way to manage this.
    std::fill(H1_adj.begin(), H1_adj.end(), 0.0);
    H2_adj.resize(H2.size());
    for (int i=0; i<num_nodes; ++i) {
        // atomic energies
        node_energies[i] += atomic_energies[node_types[i]];
        // first readout
        for (int k=0; k<num_channels; ++k) {
            node_energies[i] += readout_1_weights[k]*H1[i*num_LM*num_channels+k];
            H1_adj[i*num_LM*num_channels+k] = readout_1_weights[k];
        }
        // second readout
        auto x = std::vector<Precision>(H2.begin()+i*num_channels, H2.begin()+(i+1)*num_channels);
        auto [f, g] = readout_2->evaluate_gradient(x);
        node_energies[i] += f[0];
        for (int k=0; k<num_channels; ++k) {
            H2_adj[i*num_channels+k] = g[k];
        }
    }
}

template <typename Precision>
void MACECPU<Precision>::load_from_json(
    const std::string filename, const std::string& requested_head)
{
    std::ifstream f(filename);
    nlohmann::json file = select_prediction_head(
        nlohmann::json::parse(f), requested_head);
    selected_head = file.value("selected_head", std::string());
    available_heads = file.value("available_heads", std::vector<std::string>());
    if (file.value("model_type", std::string("MACE")) == "MACE_Nonlinear")
        throw std::invalid_argument(
            "MACE_Nonlinear JSON must be loaded through the nonlinear MACE evaluator, not legacy MACE.");

    // Basic model information
    num_elements = file["num_elements"];
    num_channels = file["num_channels"];
    num_interactions = file.value("num_interactions", 2);
    single_layer_readout = file.value("single_layer_readout", false);
    if (num_interactions != (single_layer_readout ? 1 : 2))
        throw std::invalid_argument(
            "MACE interaction count and single-layer readout metadata disagree.");
    r_cut = file["r_cut"];
    l_max = file["l_max"];
    num_lm = (l_max+1)*(l_max+1);
    L_max = file["L_max"];
    if (file.value("has_field_coupling", false))
        validate_macefield_L_max(L_max);
    num_LM = (L_max+1)*(L_max+1);
    atomic_numbers = file["atomic_numbers"].get<std::vector<int>>();
    atomic_energies = file["atomic_energies"].get<std::vector<double>>();

    // ZBL
    has_zbl = file["has_zbl"].get<bool>();
    if (has_zbl)
        zbl = ZBLT<Precision>(
            checked_precision_cast<Precision>(file["zbl_a_exp"].get<double>()),
            checked_precision_cast<Precision>(file["zbl_a_prefactor"].get<double>()),
            checked_precision_data<Precision>(file["zbl_c"].get<std::vector<double>>()),
            checked_precision_data<Precision>(
                file["zbl_covalent_radii"].get<std::vector<double>>()),
            file["zbl_p"].get<int>());

    // Radial representation
    const int format_version = file.value("symmetrix_format_version", 1);
    uses_compact_radial = format_version == 2;
    if (uses_compact_radial) {
        if (file.value("radial_representation", std::string()) != "compact")
            throw std::invalid_argument("Symmetrix format version 2 requires compact radial data.");
        compact_radial_model = std::make_unique<CompactRadialModel>(
            file.at("compact_radial").dump(), atomic_numbers, r_cut);
        type_to_active.assign(atomic_numbers.size(), -1);
    } else if (format_version == 1) {
        const Precision spl_h = checked_precision_cast<Precision>(
            file["radial_spline_h"].get<double>());
        const Precision spl_min = checked_precision_cast<Precision>(
            file.value("radial_spline_min", 0.0));
        auto spl_values_0 = checked_precision_data<Precision>(
            file["radial_spline_values_0"].get<std::vector<std::vector<std::vector<double>>>>());
        auto spl_derivs_0 = checked_precision_data<Precision>(
            file["radial_spline_derivs_0"].get<std::vector<std::vector<std::vector<double>>>>());
        for (int i=0; i<spl_values_0.size(); ++i)
            spl_set_0.push_back(std::make_unique<CubicSplineSetT<Precision>>(
                spl_h, spl_values_0[i], spl_derivs_0[i], spl_min));
        if (!single_layer_readout) {
            auto spl_values_1 = checked_precision_data<Precision>(
                file["radial_spline_values_1"].get<std::vector<std::vector<std::vector<double>>>>());
            auto spl_derivs_1 = checked_precision_data<Precision>(
                file["radial_spline_derivs_1"].get<std::vector<std::vector<std::vector<double>>>>());
            for (int i=0; i<spl_values_1.size(); ++i)
                spl_set_1.push_back(std::make_unique<CubicSplineSetT<Precision>>(
                    spl_h, spl_values_1[i], spl_derivs_1[i], spl_min));
        }
        active_types.resize(atomic_numbers.size());
        std::iota(active_types.begin(), active_types.end(), 0);
        type_to_active = active_types;
        active_atomic_numbers = atomic_numbers;
    } else {
        throw std::invalid_argument("Unsupported Symmetrix model format version.");
    }

    // H0
    H0_weights = checked_precision_data<Precision>(
        file["H0_weights"].get<std::vector<double>>());

    // A0
    A0_weights = checked_precision_data<Precision>(
        file["A0_weights"].get<std::vector<std::vector<std::vector<double>>>>());

    // A0 scaling
    A0_scaled = file["A0_scaled"].get<bool>();
    if (A0_scaled && !uses_compact_radial) {
        const Precision A0_spline_h = checked_precision_cast<Precision>(
            file["A0_spline_h"].get<double>());
        const Precision A0_spline_min = checked_precision_cast<Precision>(
            file.value("A0_spline_min", 0.0));
        auto A0_spline_values = checked_precision_data<Precision>(
            file["A0_spline_values"].get<std::vector<std::vector<double>>>());
        auto A0_spline_derivs = checked_precision_data<Precision>(
            file["A0_spline_derivs"].get<std::vector<std::vector<double>>>());
        for (int i=0; i<A0_spline_values.size(); ++i)
            A0_splines.push_back(CubicSplineT<Precision>(
                A0_spline_h, A0_spline_values[i], A0_spline_derivs[i], A0_spline_min));
    }
    if (uses_compact_radial && A0_scaled != compact_radial_model->has_A0())
        throw std::invalid_argument("Compact radial A0 network does not match A0_scaled.");
    if (uses_compact_radial
        && compact_radial_model->has_R1() == single_layer_readout)
        throw std::invalid_argument(
            "Compact radial R1 network does not match the MACE interaction count.");

    // M0
    auto M0_weights = checked_precision_data<Precision>(
        file["M0_weights"].get<std::map<std::string,std::map<std::string,std::map<std::string,std::vector<double>>>>>());
    auto M0_monomials = file["M0_monomials"].get<std::map<std::string,std::vector<std::vector<int>>>>();
    P0 = std::vector<MultivariatePolynomialT<Precision>>();
    for (int a=0; a<atomic_numbers.size(); ++a) {
        for (int lm=0; lm<num_LM; ++lm) {
            for (int k=0; k<num_channels; ++k) {
                P0.push_back(MultivariatePolynomialT<Precision>(
                    num_lm,
                    M0_weights[std::to_string(a)][std::to_string(lm)][std::to_string(k)],
                    M0_monomials[std::to_string(lm)]));
            }
        }
    }

    // H1
    H1_weights = checked_precision_data<Precision>(
        file["H1_weights"].get<std::vector<double>>());
    H1_product_weights = checked_precision_data<Precision>(
        file.value("H1_product_weights", std::vector<double>{}));
    H1_linear_up_weights = checked_precision_data<Precision>(
        file.value("H1_linear_up_weights", std::vector<double>{}));
    first_interaction_residual = file.value("first_interaction_residual", false);
    H1_first_residual_weights = checked_precision_data<Precision>(
        file.value("H1_first_residual_weights", std::vector<double>{}));
    const auto expected_residual_size = static_cast<std::size_t>(num_elements)
        *num_LM*num_channels;
    if (first_interaction_residual) {
        if (H1_first_residual_weights.size() != expected_residual_size)
            throw std::invalid_argument(
                "Residual-first MACE JSON has an invalid H1 residual extent.");
        if (H1_product_weights.size() != H1_weights.size()
            || H1_linear_up_weights.size() != H1_weights.size())
            throw std::invalid_argument(
                "Residual-first MACE JSON requires split H1 weights.");
        H1_first_residual_fused_weights.assign(expected_residual_size, Precision(0));
        for (int a=0; a<num_elements; ++a)
            for (int l=0; l<=L_max; ++l)
                for (int m=0; m<2*l+1; ++m)
                    for (int output=0; output<num_channels; ++output)
                        for (int input=0; input<num_channels; ++input) {
                            const int lm = l*l+m;
                            H1_first_residual_fused_weights[
                                (a*num_LM+lm)*num_channels+output]
                                += H1_first_residual_weights[
                                    (a*num_LM+lm)*num_channels+input]
                                *H1_linear_up_weights[
                                    (l*num_channels+input)*num_channels+output];
                        }
    } else if (!H1_first_residual_weights.empty()) {
        throw std::invalid_argument(
            "Non-residual MACE JSON must not contain H1 residual weights.");
    }

    // MACEField H1 coupling
    has_field_coupling = file.value("has_field_coupling", false);
    field_coupling_paths.clear();
    field_feats_weight.clear();
    field_feats_output_mask.clear();
    field_linear_weight.clear();
    field_linear_bias.clear();
    field_linear_output_mask.clear();
    if (has_field_coupling) {
        auto field_couplings = file["field_couplings"];
        if (field_couplings.size() != 1)
            throw std::runtime_error("MACEField JSON must contain exactly one field coupling.");
        auto coupling = field_couplings[0];
        if (H1_product_weights.size() != H1_weights.size()
            || H1_linear_up_weights.size() != H1_weights.size())
            throw std::runtime_error("MACEField JSON must contain split H1 product and linear_up weights.");

        field_feats_weight = checked_precision_data<Precision>(
            coupling["field_feats_weight"].get<std::vector<double>>());
        field_feats_output_mask = checked_precision_data<Precision>(
            coupling["field_feats_output_mask"].get<std::vector<double>>());
        field_linear_weight = checked_precision_data<Precision>(
            coupling["field_linear_weight"].get<std::vector<double>>());
        field_linear_bias = checked_precision_data<Precision>(
            coupling["field_linear_bias"].get<std::vector<double>>());
        field_linear_output_mask = checked_precision_data<Precision>(
            coupling["field_linear_output_mask"].get<std::vector<double>>());
        const std::vector<double> linear_up_weights(
            H1_linear_up_weights.begin(), H1_linear_up_weights.end());
        field_coupling_paths = compile_field_coupling(
            coupling, L_max, num_channels, linear_up_weights);
    }

    // Phi1
    Phi1_l = file.value("Phi1_l", std::vector<int>{});
    Phi1_l1 = file.value("Phi1_l1", std::vector<int>{});
    Phi1_l2 = file.value("Phi1_l2", std::vector<int>{});
    Phi1_lme = file.value("Phi1_lme", std::vector<int>{});
    Phi1_clebsch_gordan = checked_precision_data<Precision>(
        file.value("Phi1_clebsch_gordan", std::vector<double>{}));
    Phi1_lelm1lm2 = file.value("Phi1_lelm1lm2", std::vector<int>{});
    num_lme = 0;
    for (auto l : Phi1_l)
        num_lme += 2*l+1;
    num_lelm1lm2 = 0;
    for (int le=0; le<Phi1_l.size(); ++le)
        num_lelm1lm2 += (2*Phi1_l1[le]+1)*(2*Phi1_l2[le]+1);

    // A1
    A1_weights = checked_precision_data<Precision>(
        file.value("A1_weights", std::vector<std::vector<double>>{}));

    // A1 scaling
    A1_scaled = file.value("A1_scaled", false);
    if (A1_scaled && !uses_compact_radial) {
        const Precision A1_spline_h = checked_precision_cast<Precision>(
            file["A1_spline_h"].get<double>());
        const Precision A1_spline_min = checked_precision_cast<Precision>(
            file.value("A1_spline_min", 0.0));
        auto A1_spline_values = checked_precision_data<Precision>(
            file["A1_spline_values"].get<std::vector<std::vector<double>>>());
        auto A1_spline_derivs = checked_precision_data<Precision>(
            file["A1_spline_derivs"].get<std::vector<std::vector<double>>>());
        for (int i=0; i<A1_spline_values.size(); ++i)
            A1_splines.push_back(CubicSplineT<Precision>(
                A1_spline_h, A1_spline_values[i], A1_spline_derivs[i], A1_spline_min));
    }
    if (uses_compact_radial && A1_scaled != compact_radial_model->has_A1())
        throw std::invalid_argument("Compact radial A1 network does not match A1_scaled.");

    // M1
    auto M1_weights = checked_precision_data<Precision>(
        file["M1_weights"].get<std::map<std::string,std::map<std::string,std::vector<double>>>>());
    auto M1_monomials = file["M1_monomials"].get<std::vector<std::vector<int>>>();
    P1 = std::vector<MultivariatePolynomialT<Precision>>();
    for (int a=0; a<atomic_numbers.size(); ++a) {
        for (int k=0; k<num_channels; ++k) {
            P1.push_back(MultivariatePolynomialT<Precision>(
                single_layer_readout ? num_LM : num_lm,
                M1_weights[std::to_string(a)][std::to_string(k)],
                M1_monomials));
        }
    }

    // H2
    H2_weights_for_H1 = checked_precision_data<Precision>(
        file["H2_weights_for_H1"].get<std::vector<std::vector<double>>>());
    H2_weights_for_M1 = checked_precision_data<Precision>(
        file["H2_weights_for_M1"].get<std::vector<double>>());

    // Readouts
    readout_1_weights = checked_precision_data<Precision>(
        file["readout_1_weights"].get<std::vector<double>>());
    auto readout_2_weights_1 = checked_precision_data<Precision>(
        file["readout_2_weights_1"].get<std::vector<double>>());
    auto readout_2_weights_2 = checked_precision_data<Precision>(
        file["readout_2_weights_2"].get<std::vector<double>>());
    const int readout_2_hidden_size = file.value("readout_2_hidden_size", 16);
    if (readout_2_hidden_size <= 0
            || readout_2_weights_1.size()
                != static_cast<std::size_t>(num_channels * readout_2_hidden_size)
            || readout_2_weights_2.size()
                != static_cast<std::size_t>(readout_2_hidden_size)) {
        throw std::invalid_argument(
            "MACE nonlinear readout weights do not match the declared hidden size.");
    }
    readout_2 = std::make_unique<MultilayerPerceptronT<Precision>>(
        std::vector<int>{num_channels, readout_2_hidden_size, 1},
        std::vector<std::vector<Precision>>{readout_2_weights_1, readout_2_weights_2},
        checked_precision_cast<Precision>(
            file["readout_2_scale_factor"].get<double>()));
}

template class MACECPU<float>;
template class MACECPU<double>;
