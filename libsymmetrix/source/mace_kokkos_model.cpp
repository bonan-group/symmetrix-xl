#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <limits>
#include <numbers>
#include <numeric>
#include <set>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <type_traits>

// TODO: remove some of these headers?
#include "KokkosBatched_Util.hpp"
#include "KokkosBlas.hpp"
#include "KokkosBatched_Gemm_Decl.hpp"
#include "KokkosBlas_tpl_spec.hpp"
#if defined(KOKKOS_ENABLE_HIP) \
    && defined(KOKKOSKERNELS_ENABLE_TPL_ROCBLAS)
#include <rocblas/rocblas.h>
#endif
#include "nlohmann/json.hpp"
#include "sphericart.hpp"
#include "sphericart_cuda.hpp"
#include "spherical_harmonic_device.hpp"

#include "tools_kokkos.hpp"
#include "mace_kokkos.hpp"
#include "prediction_heads.hpp"
#include "device_backend.hpp"
#include "kernel_launch_profile.hpp"
#include "standard_m0.hpp"
#include "standard_m1.hpp"
#include "standard_r0.hpp"
#include "factorized_blas.hpp"

using Kokkos::ALL;
using Kokkos::LayoutRight;
using Kokkos::make_pair;
using Kokkos::MemoryUnmanaged;
using Kokkos::parallel_for;
using Kokkos::PerTeam;
using Kokkos::subview;
using Kokkos::TeamPolicy;
using Kokkos::TeamVectorRange;
using Kokkos::TeamVectorMDRange;
using Kokkos::View;
#include "mace_kokkos_kernel_launch_detail.hpp"

namespace {

std::vector<int> polynomial_coefficient_nodes(
    const MultivariatePolynomial& polynomial,
    const std::vector<std::vector<int>>& monomials)
{
    auto result = std::vector<int>();
    result.reserve(monomials.size());
    auto assigned_nodes = std::set<int>();
    for (const auto& monomial : monomials) {
        const auto found = std::find(
            polynomial.nodes.begin(), polynomial.nodes.end(), monomial);
        if (found == polynomial.nodes.end())
            throw std::logic_error(
                "Polynomial monomial is missing from its evaluation graph.");
        const int node = static_cast<int>(
            std::distance(polynomial.nodes.begin(), found));
        // MultivariatePolynomial's ordered map retains the first coefficient
        // when a legacy payload contains a repeated monomial.
        result.push_back(assigned_nodes.insert(node).second ? node : -1);
    }
    return result;
}

template <typename Precision>
std::string validate_factorized_contract_payload(
    const nlohmann::json& contract,
    const std::vector<int>& path_l,
    const std::vector<int>& path_l1,
    const std::vector<int>& path_l2,
    const std::vector<int>& term_lme,
    const std::vector<int>& term_rows,
    const std::vector<Precision>& term_coefficients,
    const std::vector<int>& row_lm1,
    const std::vector<int>& row_lm2)
{
    const auto mismatch = [] (const std::string& field) {
        return "loaded "+field
            +" does not match the exact Execution R1 contract payload";
    };
    try {
        const auto& paths = contract.at("paths");
        if (!paths.is_array() || paths.size() != path_l.size())
            return mismatch("Phi1_l/Phi1_l1/Phi1_l2");
        for (std::size_t path=0; path<path_l.size(); ++path) {
            const auto& expected = paths.at(path);
            if (!expected.is_object()
                || expected.at("index").get<int>()
                    != static_cast<int>(path)
                || expected.at("output_l").get<int>() != path_l[path]
                || expected.at("edge_l").get<int>() != path_l1[path]
                || expected.at("source_l").get<int>() != path_l2[path])
                return mismatch("Phi1_l/Phi1_l1/Phi1_l2");
        }

        const auto& sparse = contract.at("sparse_coupling");
        const auto& terms = sparse.at("terms");
        if (!sparse.is_object() || !terms.is_array()
            || sparse.at("ordering").get<std::string>()
                != "edge_lm,source_lm,path,output_m"
            || sparse.at("term_count").get<std::size_t>()
                != term_coefficients.size()
            || terms.size() != term_coefficients.size())
            return mismatch("Phi1 sparse coupling");

        for (std::size_t term=0; term<term_coefficients.size(); ++term) {
            const int row = term_rows[term];
            if (row < 0 || row >= static_cast<int>(row_lm1.size())
                || row >= static_cast<int>(row_lm2.size()))
                return mismatch("Phi1_lelm1lm2");
            const auto& expected = terms.at(term);
            if (!expected.is_object()
                || expected.at("lme").get<int>() != term_lme[term]
                || expected.at("row").get<int>() != row
                || expected.at("lm1").get<int>() != row_lm1[row]
                || expected.at("lm2").get<int>() != row_lm2[row]
                || expected.at("coefficient").get<Precision>()
                    != term_coefficients[term])
                return mismatch(
                    "Phi1_lme/Phi1_lelm1lm2/Phi1_clebsch_gordan");
        }
        return "";
    } catch (const std::exception& error) {
        return std::string(
            "malformed exact Execution R1 contract payload metadata: ")
            +error.what();
    }
}

}

template <typename Precision>
void MACEKokkos<Precision>::load_from_json(
    std::string filename, const std::string& requested_head)
{
    model_load_phase_names.clear();
    model_load_phase_ms.clear();
    auto phase_start = std::chrono::steady_clock::now();
    const auto record_phase = [&] (const std::string& name) {
        const auto now = std::chrono::steady_clock::now();
        model_load_phase_names.push_back(name);
        model_load_phase_ms.push_back(
            std::chrono::duration<double,std::milli>(now-phase_start).count());
        phase_start = now;
    };
    std::ifstream f(filename);
    nlohmann::json file = select_prediction_head(
        nlohmann::json::parse(f), requested_head);
    selected_head = file.value("selected_head", std::string());
    available_heads = file.value("available_heads", std::vector<std::string>());
    record_phase("json_parse");
    if (file.value("model_type", std::string("MACE")) == "MACE_Nonlinear")
        throw std::invalid_argument(
            "MACE_Nonlinear JSON must be loaded through the nonlinear MACE Kokkos evaluator, not legacy MACEKokkos.");

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
    atomic_numbers_host = file["atomic_numbers"].get<std::vector<int>>();
    atomic_numbers = toKokkosView("atomic_numbers", atomic_numbers_host);
    atomic_energies = toKokkosView("atomic_energies", file["atomic_energies"].get<std::vector<double>>());
    H0_weights_host = file["H0_weights"].get<std::vector<double>>();
    A0_weights_host = file["A0_weights"].get<std::vector<std::vector<std::vector<double>>>>();

    // ZBL
    has_zbl = file["has_zbl"].get<bool>();
    if (has_zbl)
        zbl = ZBLKokkos(
            file["zbl_a_exp"].get<double>(),
            file["zbl_a_prefactor"].get<double>(),
            file["zbl_c"].get<std::vector<double>>(),
            file["zbl_covalent_radii"].get<std::vector<double>>(),
            file["zbl_p"].get<int>());

    A0_scaled = file["A0_scaled"].get<bool>();
    const int format_version = file.value("symmetrix_format_version", 1);
    std::string standard_m0_contract_diagnostic;
    nlohmann::json m0_contract_metadata;
    uses_compact_radial = format_version == 2;
    if (uses_compact_radial) {
        if (file.value("radial_representation", std::string()) != "compact")
            throw std::invalid_argument("Symmetrix format version 2 requires compact radial data.");
        compact_radial_model = std::make_unique<CompactRadialModel>(
            file.at("compact_radial").dump(), atomic_numbers_host, r_cut);
        if (A0_scaled != compact_radial_model->has_A0())
            throw std::invalid_argument("Compact radial A0 network does not match A0_scaled.");
        if (compact_radial_model->has_R1() == single_layer_readout)
            throw std::invalid_argument(
                "Compact radial R1 network does not match the MACE interaction count.");
        R0_spline_h = compact_radial_model->spline_h();
        R0_spline_min = compact_radial_model->spline_min();
        type_to_active = Kokkos::View<int*>("compact type_to_active", atomic_numbers.size());
        Kokkos::deep_copy(type_to_active, -1);
        if (file.contains("execution_contracts")
            && file.at("execution_contracts").contains("R0")) {
            const auto& contract = file.at("execution_contracts").at("R0");
            if (contract.value("schema", std::string())
                    != "symmetrix.execution.tensor_product"
                || contract.value("version", 0) != 1
                || contract.value("interaction", std::string()) != "R0")
                throw std::invalid_argument(
                    "Compact radial Execution R0 contract has an unsupported schema.");
            standard_r0_model_contract_fingerprint =
                contract.at("generation_fingerprint").get<std::string>();
            standard_r0_model_semantic_fingerprint =
                contract.at("fingerprint").get<std::string>();
            standard_r0_model_structure_fingerprint =
                contract.value("structure_fingerprint", std::string());
            standard_r0_has_model_contract = true;
            r0_model_payload_verified = true;
            try {
                const auto& edge = contract.at("edge_harmonics");
                const auto& source = contract.at("source_harmonics");
                const auto& paths = contract.at("paths");
                const auto& groups = contract.at("groups");
                const auto& terms = contract.at("sparse_coupling").at("terms");
                if (edge.at("l_max").get<int>() != l_max
                    || source.at("l_max").get<int>() != 0
                    || paths.size() != static_cast<std::size_t>(l_max+1)
                    || groups.size() != static_cast<std::size_t>(l_max+1)
                    || terms.size() != static_cast<std::size_t>((l_max+1)*(l_max+1)))
                    r0_model_payload_verified = false;
                for (int l=0; l<=l_max && r0_model_payload_verified; ++l) {
                    const auto& path = paths.at(l);
                    const auto& group = groups.at(l);
                    if (path.value("index", -1) != l
                        || path.value("output_l", -1) != l
                        || path.value("edge_l", -1) != l
                        || path.value("source_l", -1) != 0
                        || path.value("connection_mode", std::string()) != "uvu"
                        || !path.value("has_weight", false)
                        || path.at("path_shape").get<std::vector<int>>()
                            != std::vector<int>{num_channels, 1}
                        || group.value("l", -1) != l
                        || group.value("components", -1) != 2*l+1
                        || group.at("path_indices").get<std::vector<int>>()
                            != std::vector<int>{l})
                        r0_model_payload_verified = false;
                }
                for (int lm=0; lm<(l_max+1)*(l_max+1)
                     && r0_model_payload_verified; ++lm) {
                    const auto& term = terms.at(lm);
                    if (term.value("row", -1) != lm
                        || term.value("lme", -1) != lm
                        || term.value("lm1", -1) != lm
                        || term.value("lm2", -1) != 0
                        || term.value("coefficient", 0.0) != 1.0)
                        r0_model_payload_verified = false;
                }
            } catch (const std::exception&) {
                r0_model_payload_verified = false;
            }
        }
        const auto launch_environment =
            kernel_launch_environment(factorized_execution_space);
        constexpr std::string_view precision_name =
            std::is_same_v<Precision,float> ? "float32" : "float64";
        standard_r0_module_ready = standard_r0_has_model_contract
            && num_channels > 0
            && l_max == symmetrix::standard_r0::l_max
            && standard_r0_model_structure_fingerprint
                == symmetrix::standard_r0::structure_fingerprint
            && launch_environment.supported()
            && symmetrix::execution::select_launch_profile(
                symmetrix::standard_r0::module_id,
                launch_environment.backend,
                precision_name,
                launch_environment.warp_width,
                launch_environment.compute_capability,
                launch_environment.architecture) != nullptr;
        if (standard_r0_module_ready) {
            selected_r0_implementation = R0Implementation::builtin;
            standard_r0_module_fallback_reason = "";
        } else if (!standard_r0_has_model_contract) {
            standard_r0_module_fallback_reason =
                "model does not contain a Execution R0 contract";
        } else {
            standard_r0_module_fallback_reason =
                "model R0 topology is not supported by the built-in module";
        }
        if (file.contains("execution_contracts")
            && file.at("execution_contracts").contains("R1")) {
            const auto& contract = file.at("execution_contracts").at("R1");
            try {
                if (!contract.is_object()
                    || contract.value("schema", std::string())
                        != "symmetrix.execution.tensor_product"
                    || contract.value("version", 0) != 1
                    || contract.value("interaction", std::string()) != "R1")
                    throw std::invalid_argument("unsupported schema");
                factorized_model_contract_fingerprint =
                    contract.at("generation_fingerprint").get<std::string>();
                factorized_model_semantic_fingerprint =
                    contract.at("fingerprint").get<std::string>();
                factorized_model_structure_fingerprint =
                    contract.value("structure_fingerprint", std::string());
                factorized_model_embedding =
                    contract.at("radial_embedding").get<int>();
                factorized_has_model_contract = true;
            } catch (const std::exception& error) {
                factorized_has_model_contract = false;
                factorized_model_embedding = 0;
                factorized_model_contract_fingerprint.clear();
                factorized_model_semantic_fingerprint.clear();
                factorized_model_structure_fingerprint.clear();
                factorized_model_payload_fallback_reason =
                    std::string("malformed Execution R1 contract metadata: ")
                    +error.what();
            }
        }
        if (file.contains("execution_contracts")
            && file.at("execution_contracts").contains("M0")) {
            const auto& contract = file.at("execution_contracts").at("M0");
            try {
                if (!contract.is_object()
                    || contract.value("schema", std::string())
                        != "symmetrix.execution.symmetric_contraction"
                    || contract.value("version", 0) != 1
                    || contract.value("interaction", std::string()) != "M0")
                    throw std::invalid_argument("unsupported schema");
                standard_m0_model_contract_fingerprint =
                    contract.at("generation_fingerprint").get<std::string>();
                standard_m0_model_semantic_fingerprint =
                    contract.at("semantic_fingerprint").get<std::string>();
                standard_m0_model_structure_fingerprint =
                    contract.at("structure_fingerprint").get<std::string>();
                m0_contract_metadata = contract;
                standard_m0_has_model_contract = true;
            } catch (const std::exception& error) {
                standard_m0_has_model_contract = false;
                standard_m0_model_contract_fingerprint.clear();
                standard_m0_model_semantic_fingerprint.clear();
                standard_m0_model_structure_fingerprint.clear();
                standard_m0_contract_diagnostic =
                    std::string("invalid optional Execution M0 contract metadata: ")
                    +error.what();
            }
        }
    } else if (format_version == 1) {
        const double spl_h = file["radial_spline_h"];
        const double spl_min = file.value("radial_spline_min", 0.0);
        auto spl_values_0 = file["radial_spline_values_0"].get<std::vector<std::vector<std::vector<double>>>>();
        auto spl_derivs_0 = file["radial_spline_derivs_0"].get<std::vector<std::vector<std::vector<double>>>>();
        auto c = Kokkos::View<Precision****,Kokkos::LayoutRight>(
            "c", atomic_numbers.size()*atomic_numbers.size(), spl_values_0[0][0].size()-1, 4, (l_max+1)*num_channels);
        auto h_c = Kokkos::create_mirror_view(c);
        for (int a=0; a<atomic_numbers.size(); ++a) {
            for (int b=0; b<atomic_numbers.size(); ++b) {
                const int ab = a*atomic_numbers.size()+b;
                const int ab_unordered = (a <= b)
                    ? a*(2*atomic_numbers.size()-a-1)/2+b
                    : b*(2*atomic_numbers.size()-b-1)/2+a;
                const auto& spl_values = spl_values_0[ab_unordered];
                const auto& spl_derivs = spl_derivs_0[ab_unordered];
                for (int i=0; i<spl_values_0[0][0].size()-1; ++i) {
                    for (int lk=0; lk<(l_max+1)*num_channels; ++lk) {
                        h_c(ab,i,0,lk) = spl_values[lk][i];
                        h_c(ab,i,1,lk) = spl_derivs[lk][i];
                        h_c(ab,i,2,lk) = (-3*spl_values[lk][i]-2*spl_h*spl_derivs[lk][i]
                            +3*spl_values[lk][i+1]-spl_h*spl_derivs[lk][i+1])/(spl_h*spl_h);
                        h_c(ab,i,3,lk) = (2*spl_values[lk][i]+spl_h*spl_derivs[lk][i]
                            -2*spl_values[lk][i+1]+spl_h*spl_derivs[lk][i+1])/(spl_h*spl_h*spl_h);
                    }
                    for (int lk=0; lk<(l_max+1)*num_channels; ++lk) {
                        const int k = lk%num_channels;
                        for (int coefficient=0; coefficient<4; ++coefficient)
                            h_c(ab,i,coefficient,lk) *= H0_weights_host[b*num_channels+k];
                    }
                    for (int l=0; l<=l_max; ++l) {
                        auto original = std::vector<double>(4*num_channels);
                        for (int coefficient=0; coefficient<4; ++coefficient)
                            for (int k=0; k<num_channels; ++k)
                                original[coefficient*num_channels+k] = h_c(ab,i,coefficient,l*num_channels+k);
                        for (int coefficient=0; coefficient<4; ++coefficient) {
                            for (int k=0; k<num_channels; ++k) {
                                h_c(ab,i,coefficient,l*num_channels+k) = 0.0;
                                for (int kp=0; kp<num_channels; ++kp)
                                    h_c(ab,i,coefficient,l*num_channels+k) +=
                                        A0_weights_host[a][l][kp*num_channels+k]
                                        * original[coefficient*num_channels+kp];
                            }
                        }
                    }
                }
            }
        }
        Kokkos::deep_copy(c, h_c);
        R0_spline_h = spl_h;
        R0_spline_min = spl_min;
        R0_spline_coefficients = c;

        auto spl_values_1 = file["radial_spline_values_1"].get<std::vector<std::vector<std::vector<double>>>>();
        auto spl_derivs_1 = file["radial_spline_derivs_1"].get<std::vector<std::vector<std::vector<double>>>>();
        radial_1 =
            RadialFunctionSetKokkos<Precision>(spl_h, spl_values_1, spl_derivs_1, spl_min);

        if (A0_scaled) {
            const double A0_spline_h = file["A0_spline_h"];
            auto A0_spline_values = std::vector<std::vector<std::vector<double>>>();
            for (auto& values : file["A0_spline_values"].get<std::vector<std::vector<double>>>())
                A0_spline_values.push_back({values});
            auto A0_spline_derivs = std::vector<std::vector<std::vector<double>>>();
            for (auto& derivs : file["A0_spline_derivs"].get<std::vector<std::vector<double>>>())
                A0_spline_derivs.push_back({derivs});
            const double A0_spline_min = file.value("A0_spline_min", 0.0);
            A0_splines = RadialFunctionSetKokkos<double>(
                A0_spline_h, A0_spline_values, A0_spline_derivs, A0_spline_min);
        }

        active_types.resize(atomic_numbers.size());
        std::iota(active_types.begin(), active_types.end(), 0);
        active_atomic_numbers = atomic_numbers_host;
        num_active_types = active_types.size();
        type_to_active = toKokkosView("type_to_active", active_types);
    } else {
        throw std::invalid_argument("Unsupported Symmetrix model format version.");
    }

    record_phase("metadata_compact_radial_and_contracts");
    // M0 weights and monomials
    auto M0_weights_file = file["M0_weights"].get<std::map<std::string,std::map<std::string,std::map<std::string,std::vector<double>>>>>();
    auto M0_monomials_file = file["M0_monomials"].get<std::map<std::string,std::vector<std::vector<int>>>>();
    auto canonical_m0_rows = std::vector<std::vector<int>>(num_LM);
    auto m0_term_counts = std::vector<int>(num_LM, 0);
    int m0_correlation = 0;
    for (int LM=0; LM<num_LM; ++LM) {
        const auto key = std::to_string(LM);
        const auto found = M0_monomials_file.find(key);
        if (found == M0_monomials_file.end())
            throw std::invalid_argument(
                "M0 payload is missing monomials for output component "+key+".");
        const auto& monomials = found->second;
        auto& rows = canonical_m0_rows[LM];
        rows.resize(monomials.size());
        std::iota(rows.begin(), rows.end(), 0);
        std::sort(rows.begin(), rows.end(), [&] (const int left, const int right) {
            const auto& lhs = monomials[left];
            const auto& rhs = monomials[right];
            if (lhs.size() != rhs.size())
                return lhs.size() < rhs.size();
            return lhs < rhs;
        });
        m0_term_counts[LM] = static_cast<int>(rows.size());
        for (int term=0; term<static_cast<int>(rows.size()); ++term) {
            const auto& components = monomials[rows[term]];
            if (components.empty() || components.size() > 4)
                throw std::invalid_argument(
                    "M0 monomials must have degree between one and four.");
            for (const int component : components)
                if (component < 0 || component >= num_lm)
                    throw std::invalid_argument(
                        "M0 monomial component is outside the A0 component range.");
            m0_correlation = std::max(
                m0_correlation, static_cast<int>(components.size()));
        }
        for (int type=0; type<static_cast<int>(atomic_numbers.size()); ++type) {
            const auto type_found = M0_weights_file.find(std::to_string(type));
            if (type_found == M0_weights_file.end())
                throw std::invalid_argument(
                    "M0 payload is missing weights for node type "
                    +std::to_string(type)+".");
            const auto output_found = type_found->second.find(key);
            if (output_found == type_found->second.end())
                throw std::invalid_argument(
                    "M0 payload is missing weights for output component "+key+".");
            for (int channel=0; channel<num_channels; ++channel) {
                const auto channel_found =
                    output_found->second.find(std::to_string(channel));
                if (channel_found == output_found->second.end()
                    || channel_found->second.size() != monomials.size())
                    throw std::invalid_argument(
                        "M0 weight row length does not match the monomial count.");
            }
        }
    }
    this->m0_correlation = m0_correlation;
    m0_module_term_count = std::accumulate(
        m0_term_counts.begin(), m0_term_counts.end(), 0);
    m0_model_payload_verified = standard_m0_has_model_contract;
    if (m0_model_payload_verified) {
        try {
            const auto& groups = m0_contract_metadata.at("monomial_groups");
            if (!groups.is_array()
                || groups.size() != static_cast<std::size_t>(num_LM))
                m0_model_payload_verified = false;
            for (int LM=0; LM<num_LM && m0_model_payload_verified; ++LM) {
                const auto terms = groups.at(LM).at("terms")
                    .get<std::vector<std::vector<int>>>();
                if (groups.at(LM).value("output_component", -1) != LM
                    || terms.size() != canonical_m0_rows[LM].size()) {
                    m0_model_payload_verified = false;
                    break;
                }
                const auto& payload = M0_monomials_file.at(std::to_string(LM));
                for (int term=0; term<static_cast<int>(terms.size()); ++term)
                    if (terms[term] != payload[canonical_m0_rows[LM][term]]) {
                        m0_model_payload_verified = false;
                        break;
                    }
            }
        } catch (const std::exception&) {
            m0_model_payload_verified = false;
        }
    }
    const auto launch_environment =
        kernel_launch_environment(factorized_execution_space);
    {
        constexpr std::string_view precision_name =
            std::is_same_v<Precision,float> ? "float32" : "float64";
        StandardM0ModuleVariant candidate = StandardM0ModuleVariant::none;
        if (l_max == symmetrix::standard_m0::input_l_max) {
            if (L_max == symmetrix::standard_m0::scalar_output_l_max)
                candidate = StandardM0ModuleVariant::scalar_lmax0;
            else if (L_max == symmetrix::standard_m0::output_l_max)
                candidate = StandardM0ModuleVariant::full_lmax1;
        }
        const bool scalar_candidate =
            candidate == StandardM0ModuleVariant::scalar_lmax0;
        const std::string_view candidate_module_id = scalar_candidate
            ? std::string_view(symmetrix::standard_m0::scalar_module_id)
            : std::string_view(symmetrix::standard_m0::module_id);
        const bool structure_matches = scalar_candidate
            ? symmetrix::standard_m0::matches_scalar_structure(
                num_channels,
                static_cast<int>(atomic_numbers.size()),
                num_lm,
                num_LM,
                m0_correlation,
                std::span<const int>(
                    m0_term_counts.data(), m0_term_counts.size()))
            : symmetrix::standard_m0::matches_structure(
                num_channels,
                static_cast<int>(atomic_numbers.size()),
                num_lm,
                num_LM,
                m0_correlation,
                std::span<const int>(
                    m0_term_counts.data(), m0_term_counts.size()));
        const bool m0_module_admitted =
            candidate != StandardM0ModuleVariant::none
            && structure_matches
            && launch_environment.supported()
            && symmetrix::execution::select_launch_profile(
                candidate_module_id,
                launch_environment.backend,
                precision_name,
                launch_environment.warp_width,
                launch_environment.compute_capability,
                launch_environment.architecture) != nullptr;

        bool m0_payload_matches = m0_module_admitted;
        if (m0_payload_matches) {
            for (int LM=0; LM<num_LM && m0_payload_matches; ++LM) {
                const auto& monomials =
                    M0_monomials_file.at(std::to_string(LM));
                for (int term=0;
                     term<static_cast<int>(canonical_m0_rows[LM].size());
                     ++term) {
                    const auto& components =
                        monomials[canonical_m0_rows[LM][term]];
                    const auto component_span = std::span<const int>(
                        components.data(), components.size());
                    const bool term_matches = scalar_candidate
                        ? symmetrix::standard_m0::matches_scalar_term(
                            LM, term, component_span)
                        : symmetrix::standard_m0::matches_term(
                            LM, term, component_span);
                    if (!term_matches) {
                        m0_payload_matches = false;
                        break;
                    }
                }
            }
        }
        standard_m0_module_ready = m0_payload_matches;
        standard_m0_module_variant = standard_m0_module_ready
            ? candidate : StandardM0ModuleVariant::none;
        if (standard_m0_module_ready) {
            selected_m0_implementation = M0Implementation::builtin;
            standard_m0_module_fallback_reason.clear();
        } else if (m0_module_admitted) {
            standard_m0_module_fallback_reason =
                "model M0 payload does not match the standard module";
        } else if (!standard_m0_contract_diagnostic.empty()) {
            standard_m0_module_fallback_reason = standard_m0_contract_diagnostic;
        } else if (!standard_m0_has_model_contract) {
            standard_m0_module_fallback_reason =
                "model M0 topology is not supported by the built-in module";
        } else {
            standard_m0_module_fallback_reason =
                "no compatible built-in Execution M0 module";
        }
        if ((standard_m0_module_ready || m0_model_payload_verified)
            && m0_correlation <= 4) {
            standard_m0_module_weights = decltype(standard_m0_module_weights)(
                Kokkos::view_alloc(
                    "m0_module_weights", Kokkos::WithoutInitializing),
                atomic_numbers.size(), m0_module_term_count, num_channels);
            auto host_weights =
                Kokkos::create_mirror_view(standard_m0_module_weights);
            int packed_offset = 0;
            for (int LM=0; LM<num_LM; ++LM) {
                for (int term=0;
                     term<static_cast<int>(canonical_m0_rows[LM].size());
                     ++term) {
                    const int source_row = canonical_m0_rows[LM][term];
                    for (int type=0;
                         type<static_cast<int>(atomic_numbers.size()); ++type)
                        for (int channel=0; channel<num_channels; ++channel)
                            host_weights(type,packed_offset+term,channel) =
                                M0_weights_file.at(std::to_string(type))
                                    .at(std::to_string(LM))
                                    .at(std::to_string(channel))[source_row];
                }
                packed_offset += static_cast<int>(canonical_m0_rows[LM].size());
            }
            Kokkos::deep_copy(standard_m0_module_weights, host_weights);
        }
    }
    M0_weights = Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
        Kokkos::view_alloc("M0_weights", Kokkos::SequentialHostInit), num_LM);
    M0_monomials = Kokkos::View<Kokkos::View<int**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
        Kokkos::view_alloc("M0_monomials", Kokkos::SequentialHostInit), num_LM);
    for (int LM=0; LM<num_LM; ++LM) {
        M0_weights(LM) = Kokkos::View<Precision***,Kokkos::LayoutRight>(
            Kokkos::view_alloc(std::string("M0_weights_") + std::to_string(LM), Kokkos::WithoutInitializing),
            atomic_numbers.size(), num_channels, M0_monomials_file[std::to_string(LM)].size());
        auto h_M0_weights_LM = Kokkos::create_mirror_view(M0_weights(LM));
        for (int a=0; a<atomic_numbers.size(); ++a)
            for (int k=0; k<num_channels; ++k)
                for (int w=0; w<M0_monomials_file[std::to_string(LM)].size(); ++w)
                    h_M0_weights_LM(a,k,w) = M0_weights_file[std::to_string(a)][std::to_string(LM)][std::to_string(k)][w];
        Kokkos::deep_copy(M0_weights(LM), h_M0_weights_LM);
        M0_monomials(LM) = Kokkos::View<int**,Kokkos::LayoutRight>(
            Kokkos::view_alloc(std::string("M0_monomials_") + std::to_string(LM), Kokkos::WithoutInitializing),
            M0_monomials_file[std::to_string(LM)].size(), 4);
        auto h_M0_monomials_LM = Kokkos::create_mirror_view(M0_monomials(LM));
        Kokkos::deep_copy(h_M0_monomials_LM, -1);
        for (int i=0; i<M0_monomials_file[std::to_string(LM)].size(); ++i) {
            for (int j=0; j<M0_monomials_file[std::to_string(LM)][i].size(); ++j) {
                h_M0_monomials_LM(i,j) = M0_monomials_file[std::to_string(LM)][i][j];
            }
        }
        Kokkos::deep_copy(M0_monomials(LM), h_M0_monomials_LM);
    }

    // M0_poly_spec
    M0_poly_spec = Kokkos::View<Kokkos::View<int**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
        Kokkos::view_alloc("M0_poly_spec",Kokkos::SequentialHostInit), num_LM);
    for (int LM=0; LM<num_LM; ++LM) {
        auto P = MultivariatePolynomial(
            num_lm,
            M0_weights_file[std::to_string(0)][std::to_string(LM)][std::to_string(0)],
            M0_monomials_file[std::to_string(LM)]);
        M0_poly_spec(LM) = Kokkos::View<int**,Kokkos::LayoutRight>(
            Kokkos::view_alloc(std::string("M0_poly_spec_")+std::to_string(LM),Kokkos::WithoutInitializing),
            P.edges.size(), 2);
        auto h_M0_poly_spec_LM = Kokkos::create_mirror_view(M0_poly_spec(LM));
        for (int p=0; p<P.edges.size(); ++p) {
            h_M0_poly_spec_LM(p,0) = P.edges[p][0];
            h_M0_poly_spec_LM(p,1) = P.edges[p][1];
        }
        Kokkos::deep_copy(M0_poly_spec(LM), h_M0_poly_spec_LM);
    }
    // M0_poly_coeff
    M0_poly_coeff = Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
        Kokkos::view_alloc("M0_poly_coeff",Kokkos::SequentialHostInit), num_LM);
    for (int LM=0; LM<num_LM; ++LM) {
        const auto P = MultivariatePolynomial(
            num_lm,
            M0_weights_file[std::to_string(0)][std::to_string(LM)][std::to_string(0)],
            M0_monomials_file[std::to_string(LM)]);
        const auto coefficient_nodes = polynomial_coefficient_nodes(
            P, M0_monomials_file[std::to_string(LM)]);
        M0_poly_coeff(LM) = Kokkos::View<Precision***,Kokkos::LayoutRight>(
            Kokkos::view_alloc(std::string("M0_poly_coeff_")+std::to_string(LM),Kokkos::WithoutInitializing),
            atomic_numbers.size(), P.node_coefficients.size(), num_channels);
        auto h_M0_poly_coeff_LM = Kokkos::create_mirror_view(M0_poly_coeff(LM));
        Kokkos::deep_copy(h_M0_poly_coeff_LM, Precision(0));
        for (int a=0; a<atomic_numbers.size(); ++a) {
            for (int k=0; k<num_channels; ++k) {
                const auto& coefficients = M0_weights_file[std::to_string(a)]
                    [std::to_string(LM)][std::to_string(k)];
                for (int term=0;
                     term<static_cast<int>(coefficient_nodes.size()); ++term)
                    if (coefficient_nodes[term] >= 0)
                        h_M0_poly_coeff_LM(a,coefficient_nodes[term],k) =
                            coefficients[term];
            }
        }
        Kokkos::deep_copy(M0_poly_coeff(LM), h_M0_poly_coeff_LM);
    }
    M0_poly_values = Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
        Kokkos::view_alloc("M0_poly_values",Kokkos::SequentialHostInit), num_LM);
    M0_poly_adjoints = Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
        Kokkos::view_alloc("M0_poly_adjoints",Kokkos::SequentialHostInit), num_LM);

    record_phase("m0_weights_and_polynomials");
    // H1 weights
    const auto H1_weights_vec =
        file["H1_weights"].get<std::vector<Precision>>();
    set_kokkos_view(
        H1_weights,
        H1_weights_vec,
        L_max+1,
        num_channels,
        num_channels);
    auto H1_weights_trans_vec =
        std::vector<Precision>(H1_weights_vec.size());
    for (int l=0; l<=L_max; ++l)
        for (int input=0; input<num_channels; ++input)
            for (int output=0; output<num_channels; ++output)
                H1_weights_trans_vec[
                    (l*num_channels+output)*num_channels+input]
                    = H1_weights_vec[
                        (l*num_channels+input)*num_channels+output];
    set_kokkos_view(
        H1_weights_trans,
        H1_weights_trans_vec,
        L_max+1,
        num_channels,
        num_channels);
    first_interaction_residual = file.value("first_interaction_residual", false);
    const auto H1_first_residual_weights_vec = file.value(
        "H1_first_residual_weights", std::vector<Precision>{});
    const auto expected_residual_size = static_cast<std::size_t>(num_elements)
        *num_LM*num_channels;
    const auto H1_product_weights_vec =
        file.value("H1_product_weights", std::vector<Precision>{});
    const auto H1_linear_up_weights_vec =
        file.value("H1_linear_up_weights", std::vector<Precision>{});
    if (first_interaction_residual) {
        if (H1_first_residual_weights_vec.size() != expected_residual_size)
            throw std::invalid_argument(
                "Residual-first MACE JSON has an invalid H1 residual extent.");
        if (H1_product_weights_vec.size() != H1_weights.size()
            || H1_linear_up_weights_vec.size() != H1_weights.size())
            throw std::invalid_argument(
                "Residual-first MACE JSON requires split H1 weights.");
        auto fused_residual = std::vector<Precision>(
            expected_residual_size, Precision(0));
        for (int a=0; a<num_elements; ++a)
            for (int l=0; l<=L_max; ++l)
                for (int m=0; m<2*l+1; ++m)
                    for (int output=0; output<num_channels; ++output)
                        for (int input=0; input<num_channels; ++input) {
                            const int lm = l*l+m;
                            fused_residual[(a*num_LM+lm)*num_channels+output]
                                += H1_first_residual_weights_vec[
                                    (a*num_LM+lm)*num_channels+input]
                                *H1_linear_up_weights_vec[
                                    (l*num_channels+input)*num_channels+output];
                        }
        set_kokkos_view(
            H1_first_residual_weights,
            H1_first_residual_weights_vec,
            num_elements,
            num_LM,
            num_channels);
        set_kokkos_view(
            H1_first_residual_fused_weights,
            fused_residual,
            num_elements,
            num_LM,
            num_channels);
        set_kokkos_view(
            H1_product_weights,
            H1_product_weights_vec,
            L_max+1,
            num_channels,
            num_channels);
        set_kokkos_view(
            H1_linear_up_weights,
            H1_linear_up_weights_vec,
            L_max+1,
            num_channels,
            num_channels);
    } else if (!H1_first_residual_weights_vec.empty()) {
        throw std::invalid_argument(
            "Non-residual MACE JSON must not contain H1 residual weights.");
    }

    // MACEField H1 coupling
    has_field_coupling = file.value("has_field_coupling", false);
    num_field_paths = 0;
    num_field_angular_entries = 0;
    if (has_field_coupling) {
        auto field_couplings = file["field_couplings"];
        if (field_couplings.size() != 1)
            throw std::runtime_error("MACEField JSON must contain exactly one field coupling.");
        auto coupling = field_couplings[0];

        if (H1_product_weights_vec.size() != H1_weights.size()
            || H1_linear_up_weights_vec.size() != H1_weights.size())
            throw std::runtime_error("MACEField JSON must contain split H1 product and linear_up weights.");
        set_kokkos_view(
            H1_product_weights,
            H1_product_weights_vec,
            L_max+1,
            num_channels,
            num_channels);
        set_kokkos_view(
            H1_linear_up_weights,
            H1_linear_up_weights_vec,
            L_max+1,
            num_channels,
            num_channels);
        const std::vector<double> linear_up_weights(
            H1_linear_up_weights_vec.begin(), H1_linear_up_weights_vec.end());
        const auto paths = compile_field_coupling(
            coupling, L_max, num_channels, linear_up_weights);
        num_field_paths = static_cast<int>(paths.size());

        std::vector<int> entry_input_lm;
        std::vector<int> entry_component;
        std::vector<int> entry_output_lm;
        std::vector<int> entry_path;
        std::vector<Precision> entry_coefficient;
        std::vector<Precision> path_matrix;
        std::vector<Precision> path_up_matrix;
        for (std::size_t path_index=0; path_index<paths.size(); ++path_index) {
            const auto& path = paths[path_index];
            for (const auto& entry : path.angular_entries) {
                entry_input_lm.push_back(entry.input_lm);
                entry_component.push_back(entry.field_component);
                entry_output_lm.push_back(entry.output_lm);
                entry_path.push_back(static_cast<int>(path_index));
                entry_coefficient.push_back(static_cast<Precision>(entry.coefficient));
            }
            path_matrix.insert(
                path_matrix.end(), path.channel_matrix.begin(), path.channel_matrix.end());
            path_up_matrix.insert(
                path_up_matrix.end(),
                path.channel_up_matrix.begin(),
                path.channel_up_matrix.end());
        }
        num_field_angular_entries = static_cast<int>(entry_input_lm.size());
        std::vector<int> component_entry_offsets(4, 0);
        std::vector<int> component_entries;
        component_entries.reserve(entry_component.size());
        for (int component=0; component<3; ++component) {
            component_entry_offsets[component] = static_cast<int>(
                component_entries.size());
            for (int entry=0; entry<num_field_angular_entries; ++entry)
                if (entry_component[entry] == component)
                    component_entries.push_back(entry);
        }
        component_entry_offsets[3] = static_cast<int>(component_entries.size());
        field_entry_input_lm = toKokkosView("field_entry_input_lm", entry_input_lm);
        field_entry_component = toKokkosView("field_entry_component", entry_component);
        field_entry_output_lm = toKokkosView("field_entry_output_lm", entry_output_lm);
        field_entry_path = toKokkosView("field_entry_path", entry_path);
        field_component_entry_offsets = toKokkosView(
            "field_component_entry_offsets", component_entry_offsets);
        field_component_entries = toKokkosView(
            "field_component_entries", component_entries);
        field_entry_coefficient = toKokkosView(
            "field_entry_coefficient", entry_coefficient);
        set_kokkos_view(
            field_path_matrix,
            path_matrix,
            num_field_paths,
            num_channels,
            num_channels);
        set_kokkos_view(
            field_path_up_matrix,
            path_up_matrix,
            num_field_paths,
            num_channels,
            num_channels);
    }

    // Phi1
    const auto file_Phi1_l = file.value("Phi1_l", std::vector<int>{});
    const auto file_Phi1_l1 = file.value("Phi1_l1", std::vector<int>{});
    const auto file_Phi1_l2 = file.value("Phi1_l2", std::vector<int>{});
    const auto file_Phi1_lme = file.value("Phi1_lme", std::vector<int>{});
    const auto file_Phi1_clebsch_gordan =
        file.value("Phi1_clebsch_gordan", std::vector<Precision>{});
    const auto file_Phi1_lelm1lm2 =
        file.value("Phi1_lelm1lm2", std::vector<int>{});
    if (file_Phi1_l.size() != file_Phi1_l1.size()
        || file_Phi1_l.size() != file_Phi1_l2.size())
        throw std::invalid_argument(
            "Phi1_l, Phi1_l1, and Phi1_l2 must have equal lengths.");
    if (file_Phi1_lme.size() != file_Phi1_lelm1lm2.size()
        || file_Phi1_lme.size() != file_Phi1_clebsch_gordan.size())
        throw std::invalid_argument(
            "Phi1 sparse coupling arrays must have equal lengths.");
    Phi1_l = toKokkosView("Phi1_l", file_Phi1_l);
    Phi1_l1 = toKokkosView("Phi1_l1", file_Phi1_l1);
    Phi1_l2 = toKokkosView("Phi1_l2", file_Phi1_l2);
    Phi1_lme = toKokkosView("Phi1_lme", file_Phi1_lme);
    Phi1_clebsch_gordan = toKokkosView(
        "Phi1_clebsch_gordan", file_Phi1_clebsch_gordan);
    Phi1_lelm1lm2 = toKokkosView(
        "Phi1_lelm1lm2", file_Phi1_lelm1lm2);
    num_lme = 0;
    auto h_Phi1_l = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Phi1_l);
    for (int i=0; i<h_Phi1_l.size(); ++i)
        num_lme += 2*h_Phi1_l(i)+1;
    num_lelm1lm2 = 0;
    auto h_Phi1_l1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Phi1_l1);
    auto h_Phi1_l2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Phi1_l2);
    if (h_Phi1_l1.size() != h_Phi1_l.size()
        || h_Phi1_l2.size() != h_Phi1_l.size())
        throw std::runtime_error(
            "Phi1_l, Phi1_l1, and Phi1_l2 must contain the same number of paths.");
    for (int le=0; le<h_Phi1_l.size(); ++le) {
        if (h_Phi1_l1(le) < 0 || h_Phi1_l1(le) > l_max)
            throw std::runtime_error(
                "Phi1_l1 contains an out-of-range edge degree.");
        if (h_Phi1_l2(le) < 0 || h_Phi1_l2(le) > L_max)
            throw std::runtime_error(
                "Phi1_l2 contains an out-of-range hidden degree.");
        num_lelm1lm2 += (2*h_Phi1_l1(le)+1)*(2*h_Phi1_l2(le)+1);
    }

    // for new approach to Phi1
    std::vector<int> Phi1_lm1, Phi1_lm2, Phi1_lel1l2;
    std::vector<int> Phi1_path_row_offsets;
    Phi1_path_row_offsets.reserve(Phi1_l.size()+1);
    int lelm1lm2 = 0;
    for (int lel1l2=0; lel1l2<Phi1_l.size(); ++lel1l2) {
        Phi1_path_row_offsets.push_back(lelm1lm2);
        const int l1 = h_Phi1_l1(lel1l2);
        const int l2 = h_Phi1_l2(lel1l2);
        for (int lm1=l1*l1; lm1<=l1*(l1+2); ++lm1) {
            for (int lm2=l2*l2; lm2<=l2*(l2+2); ++lm2) {
                Phi1_lm1.push_back(lm1);
                Phi1_lm2.push_back(lm2);
                Phi1_lel1l2.push_back(lel1l2);
                lelm1lm2 += 1;
            }
        }
    }
    Phi1_path_row_offsets.push_back(lelm1lm2);
    if (lelm1lm2 != num_lelm1lm2)
        throw std::runtime_error("Inconsistent Phi1 coupled-row layout.");
    if (factorized_has_model_contract) {
        const auto& contract = file.at("execution_contracts").at("R1");
        factorized_model_payload_fallback_reason =
            validate_factorized_contract_payload<Precision>(
                contract,
                file_Phi1_l,
                file_Phi1_l1,
                file_Phi1_l2,
                file_Phi1_lme,
                file_Phi1_lelm1lm2,
                file_Phi1_clebsch_gordan,
                Phi1_lm1,
                Phi1_lm2);
        factorized_model_payload_verified =
            factorized_model_payload_fallback_reason.empty();
    }
    this->Phi1_lm1 = toKokkosView("Phi1_lm1", Phi1_lm1);
    this->Phi1_lm2 = toKokkosView("Phi1_lm2", Phi1_lm2);
    this->Phi1_lel1l2 = toKokkosView("Phi1_lel1l2", Phi1_lel1l2);
    this->Phi1_path_row_offsets = toKokkosView(
        "Phi1_path_row_offsets", Phi1_path_row_offsets);

    record_phase("phi1_coupling");
    // A1 weights
    auto file_A1_weights = file.value(
        "A1_weights", std::vector<std::vector<Precision>>(l_max+1));
    std::size_t a1_projection_elements = 0;
    for (const auto& weights : file_A1_weights)
        a1_projection_elements += weights.size();
    execution_a1_projection_weights = Kokkos::View<Precision*>(
        Kokkos::view_alloc(
            "Execution A1 projection weights", Kokkos::WithoutInitializing),
        a1_projection_elements);
    auto h_execution_a1_projection_weights =
        Kokkos::create_mirror_view(execution_a1_projection_weights);
    std::size_t a1_projection_offset = 0;
    A1_weights = Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
        Kokkos::view_alloc("A1_weights", Kokkos::SequentialHostInit), l_max+1);
    A1_weights_trans = Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
        Kokkos::view_alloc("A1_weights_trans", Kokkos::SequentialHostInit), l_max+1);
    const int a1_tile_phases =
        (num_channels+phi1_channel_tile_size-1)/phi1_channel_tile_size;
    A1_channel_tile_weights = Kokkos::View<
        Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
            Kokkos::view_alloc(
                "A1_channel_tile_weights", Kokkos::SequentialHostInit),
            a1_tile_phases*(l_max+1));
    A1_channel_tile_weights_trans = Kokkos::View<
        Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
            Kokkos::view_alloc(
                "A1_channel_tile_weights_trans", Kokkos::SequentialHostInit),
            a1_tile_phases*(l_max+1));
    for (int l=0; l<=l_max; ++l) {
        int num_eta = 0;
        auto h_Phi1_l = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Phi1_l);
        for (int i=0; i<h_Phi1_l.size(); ++i)
            num_eta += (h_Phi1_l(i) == l);
        A1_weights(l) = Kokkos::View<Precision**,Kokkos::LayoutRight>(
            Kokkos::view_alloc(std::string("A1_weights_") + std::to_string(l), Kokkos::WithoutInitializing),
            num_eta*num_channels, num_channels);
        A1_weights_trans(l) = Kokkos::View<Precision**,Kokkos::LayoutRight>(
            Kokkos::view_alloc(std::string("A1_weights_") + std::to_string(l), Kokkos::WithoutInitializing),
            num_channels, num_eta*num_channels);
        auto h_A1_weights_l = Kokkos::create_mirror_view(A1_weights(l));
        auto h_A1_weights_trans_l = Kokkos::create_mirror_view(A1_weights_trans(l));
        for (int i=0; i<num_eta*num_channels; ++i) {
            for (int j=0; j<num_channels; ++j) {
                h_A1_weights_l(i,j) = file_A1_weights[l][i*num_channels+j];
                h_A1_weights_trans_l(j,i) = file_A1_weights[l][i*num_channels+j];
                h_execution_a1_projection_weights(
                    a1_projection_offset
                        +static_cast<std::size_t>(i)*num_channels+j) =
                    file_A1_weights[l][i*num_channels+j];
            }
        }
        Kokkos::deep_copy(A1_weights(l), h_A1_weights_l);
        Kokkos::deep_copy(A1_weights_trans(l), h_A1_weights_trans_l);
        for (int phase=0; phase<a1_tile_phases; ++phase) {
            const int channel_begin = phase*phi1_channel_tile_size;
            const int channel_count = std::min(
                phi1_channel_tile_size, num_channels-channel_begin);
            const int tile_index = phase*(l_max+1)+l;
            A1_channel_tile_weights(tile_index) =
                Kokkos::View<Precision**,Kokkos::LayoutRight>(
                    Kokkos::view_alloc(
                        std::string("A1_channel_tile_weights_")
                            +std::to_string(phase)+"_"+std::to_string(l),
                        Kokkos::WithoutInitializing),
                    num_eta*channel_count, num_channels);
            A1_channel_tile_weights_trans(tile_index) =
                Kokkos::View<Precision**,Kokkos::LayoutRight>(
                    Kokkos::view_alloc(
                        std::string("A1_channel_tile_weights_trans_")
                            +std::to_string(phase)+"_"+std::to_string(l),
                        Kokkos::WithoutInitializing),
                    num_channels, num_eta*channel_count);
            auto h_tile = Kokkos::create_mirror_view(
                A1_channel_tile_weights(tile_index));
            auto h_tile_trans = Kokkos::create_mirror_view(
                A1_channel_tile_weights_trans(tile_index));
            for (int eta=0; eta<num_eta; ++eta) {
                for (int local=0; local<channel_count; ++local) {
                    const int original_row =
                        eta*num_channels+channel_begin+local;
                    const int packed_row = eta*channel_count+local;
                    for (int output_channel=0;
                         output_channel<num_channels; ++output_channel) {
                        const Precision value = file_A1_weights[l][
                            original_row*num_channels+output_channel];
                        h_tile(packed_row,output_channel) = value;
                        h_tile_trans(output_channel,packed_row) = value;
                    }
                }
            }
            Kokkos::deep_copy(
                A1_channel_tile_weights(tile_index), h_tile);
            Kokkos::deep_copy(
                A1_channel_tile_weights_trans(tile_index), h_tile_trans);
        }
        a1_projection_offset += A1_weights(l).size();
    }
    Kokkos::deep_copy(
        execution_a1_projection_weights,
        h_execution_a1_projection_weights);

    // A1 scaling
    A1_scaled = file.value("A1_scaled", false);
    if (A1_scaled && !uses_compact_radial) {
        const double A1_spline_h = file["A1_spline_h"];
        auto A1_spline_values = std::vector<std::vector<std::vector<double>>>();
        for (auto& values : file["A1_spline_values"].get<std::vector<std::vector<double>>>())
            A1_spline_values.push_back({values});  // adds dimension to reach 3d
        auto A1_spline_derivs = std::vector<std::vector<std::vector<double>>>();
        for (auto& derivs : file["A1_spline_derivs"].get<std::vector<std::vector<double>>>())
            A1_spline_derivs.push_back({derivs});  // adds dimension to reach 3d
        const double A1_spline_min = file.value("A1_spline_min", 0.0);
        A1_splines = RadialFunctionSetKokkos<double>(
            A1_spline_h, A1_spline_values, A1_spline_derivs, A1_spline_min);
    }
    if (uses_compact_radial && A1_scaled != compact_radial_model->has_A1())
        throw std::invalid_argument("Compact radial A1 network does not match A1_scaled.");

    record_phase("a1_weights");
    // M1 weights and monomials
    auto M1_weights = file["M1_weights"].get<std::map<std::string,std::map<std::string,std::vector<double>>>>();
    const int num_terms = M1_weights[std::to_string(0)][std::to_string(0)].size();
    auto M1_monomials = file["M1_monomials"].get<std::vector<std::vector<int>>>();
    Kokkos::resize(this->M1_monomials, num_terms, 3);// TODO: hardcoded 3
    auto h_M1_monomials = Kokkos::create_mirror_view(this->M1_monomials);
    Kokkos::deep_copy(h_M1_monomials, -1);
    for (int i=0; i<num_terms; ++i)
        for (int j=0; j<M1_monomials[i].size(); ++j)
            h_M1_monomials(i,j) = M1_monomials[i][j];
    Kokkos::deep_copy(this->M1_monomials, h_M1_monomials);
    auto canonical_m1_rows = std::vector<int>();
    if constexpr (symmetrix::standard_m1::supports_backend<
            Precision,Kokkos::DefaultExecutionSpace>()) {
        constexpr bool host_backend = std::is_same_v<
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>;
        standard_m1_module_ready = !single_layer_readout
            && (!has_field_coupling || host_backend)
            && standard_m0_module_ready
            && symmetrix::standard_m1::matches_structure(
                num_lm, M1_monomials, &canonical_m1_rows);
    }
    Kokkos::resize(this->M1_weights, atomic_numbers.size(), num_terms, num_channels);
    auto h_M1_weights = Kokkos::create_mirror_view(this->M1_weights);
    for (int type=0; type<static_cast<int>(atomic_numbers.size()); ++type)
        for (int term=0; term<num_terms; ++term)
            for (int channel=0; channel<num_channels; ++channel) {
                const int source_term = standard_m1_module_ready
                    ? canonical_m1_rows[term] : term;
                h_M1_weights(type,term,channel) =
                    M1_weights[std::to_string(type)]
                        [std::to_string(channel)][source_term];
            }
    Kokkos::deep_copy(this->M1_weights, h_M1_weights);
    // Begin recursive
    const auto P1 = MultivariatePolynomial(
        single_layer_readout ? num_LM : num_lm,
        M1_weights[std::to_string(0)][std::to_string(0)],
        M1_monomials);
    const auto M1_coefficient_nodes = polynomial_coefficient_nodes(
        P1, M1_monomials);
    // M1_poly_spec
    Kokkos::realloc(M1_poly_spec, P1.edges.size(), 2);
    auto h_M1_poly_spec = Kokkos::create_mirror_view(M1_poly_spec);
    for (int p=0; p<P1.edges.size(); ++p) {
        h_M1_poly_spec(p,0) = P1.edges[p][0];
        h_M1_poly_spec(p,1) = P1.edges[p][1];
    }
    Kokkos::deep_copy(M1_poly_spec, h_M1_poly_spec);
    // M1_poly_coeff
    Kokkos::realloc(
        M1_poly_coeff, atomic_numbers.size(), P1.node_coefficients.size(),
        num_channels);
    auto h_M1_poly_coeff = Kokkos::create_mirror_view(M1_poly_coeff);
    Kokkos::deep_copy(h_M1_poly_coeff, Precision(0));
    for (int a=0; a<atomic_numbers.size(); ++a) {
        for (int k=0; k<num_channels; ++k) {
            const auto& coefficients =
                M1_weights[std::to_string(a)][std::to_string(k)];
            for (int term=0;
                 term<static_cast<int>(M1_coefficient_nodes.size()); ++term)
                if (M1_coefficient_nodes[term] >= 0)
                    h_M1_poly_coeff(a,M1_coefficient_nodes[term],k) =
                        coefficients[term];
        }
    }
    Kokkos::deep_copy(M1_poly_coeff, h_M1_poly_coeff);

    record_phase("m1_weights_and_polynomials");
    // H2
    auto H2_weights_for_H1_vec = file["H2_weights_for_H1"].get<std::vector<std::vector<double>>>();
    H2_weights_for_H1 = Kokkos::View<
        H2WeightPrecision**,Kokkos::LayoutRight>(
        "H2_weights_for_H2", num_elements, num_channels*num_channels);
    auto h_H2_weights_for_H1 = Kokkos::create_mirror_view(H2_weights_for_H1);
    for (int i=0; i<num_elements; ++i) {
        for (int j=0; j<num_channels*num_channels; ++j) {
            h_H2_weights_for_H1(i,j) =
                static_cast<H2WeightPrecision>(H2_weights_for_H1_vec[i][j]);
        }
    }
    Kokkos::deep_copy(H2_weights_for_H1, h_H2_weights_for_H1);
    if constexpr (!std::is_same_v<
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>) {
        H2_weights_for_H1_reverse = Kokkos::View<
            H2WeightPrecision**,Kokkos::LayoutRight>(
            "H2_weights_for_H1_reverse", num_elements,
            num_channels*num_channels);
        auto h_H2_weights_for_H1_reverse =
            Kokkos::create_mirror_view(H2_weights_for_H1_reverse);
        for (int type=0; type<num_elements; ++type) {
            for (int k=0; k<num_channels; ++k) {
                for (int kp=0; kp<num_channels; ++kp) {
                    h_H2_weights_for_H1_reverse(
                        type,kp*num_channels+k) = static_cast<H2WeightPrecision>(
                            H2_weights_for_H1_vec[type][k*num_channels+kp]);
                }
            }
        }
        Kokkos::deep_copy(
            H2_weights_for_H1_reverse, h_H2_weights_for_H1_reverse);
    }
    const auto H2_weights_for_M1_vec =
        file["H2_weights_for_M1"].get<std::vector<double>>();
    H2_weights_for_M1 = Kokkos::View<H2WeightPrecision*>(
        "H2_weights_for_M1", H2_weights_for_M1_vec.size());
    auto h_H2_weights_for_M1 = Kokkos::create_mirror_view(H2_weights_for_M1);
    for (std::size_t i=0; i<H2_weights_for_M1_vec.size(); ++i)
        h_H2_weights_for_M1(i) =
            static_cast<H2WeightPrecision>(H2_weights_for_M1_vec[i]);
    Kokkos::deep_copy(H2_weights_for_M1, h_H2_weights_for_M1);
    if constexpr (!std::is_same_v<
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>) {
        H2_weights_for_M1_reverse = Kokkos::View<H2WeightPrecision*>(
            "H2_weights_for_M1_reverse", H2_weights_for_M1_vec.size());
        auto h_H2_weights_for_M1_reverse =
            Kokkos::create_mirror_view(H2_weights_for_M1_reverse);
        for (int k=0; k<num_channels; ++k) {
            for (int kp=0; kp<num_channels; ++kp) {
                h_H2_weights_for_M1_reverse(kp*num_channels+k) =
                    static_cast<H2WeightPrecision>(
                        H2_weights_for_M1_vec[k*num_channels+kp]);
            }
        }
        Kokkos::deep_copy(
            H2_weights_for_M1_reverse, h_H2_weights_for_M1_reverse);
    }

    record_phase("h2_weights");
    // Readouts
    readout_1_weights = toKokkosView("readout_1_weights", file["readout_1_weights"].get<std::vector<double>>());
    auto readout_2_weights_1 = file["readout_2_weights_1"].get<std::vector<double>>();
    auto readout_2_weights_2 = file["readout_2_weights_2"].get<std::vector<double>>();
    const int readout_2_hidden_size = file.value("readout_2_hidden_size", 16);
    if (readout_2_hidden_size <= 0
            || readout_2_weights_1.size()
                != static_cast<std::size_t>(num_channels * readout_2_hidden_size)
            || readout_2_weights_2.size()
                != static_cast<std::size_t>(readout_2_hidden_size)) {
        throw std::invalid_argument(
            "MACE nonlinear readout weights do not match the declared hidden size.");
    }
    readout_2 = MultilayerPerceptronKokkos(
        std::vector<int>{num_channels, readout_2_hidden_size, 1},
        std::vector<std::vector<double>>{readout_2_weights_1, readout_2_weights_2},
        file["readout_2_scale_factor"]);
    record_phase("readouts");
}

template class MACEKokkos<float>;
template class MACEKokkos<double>;
