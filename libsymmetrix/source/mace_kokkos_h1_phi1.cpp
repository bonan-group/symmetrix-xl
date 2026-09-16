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
#include "cblas.hpp"
#include "device_backend.hpp"
#include "kernel_launch_profile.hpp"
#include "factorized_blas.hpp"
#include "host_dense_kernels.hpp"
#include "host_worker_blas.hpp"

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
#include "mace_kokkos_jit_plugin_detail.hpp"

namespace {

template <bool TransposeWeight, typename Precision, typename InputView,
          typename WeightView, typename OutputView>
void launch_host_h1_gemm(
    const char* label,
    const Kokkos::DefaultExecutionSpace& execution_space,
    const int num_nodes,
    const int l_max,
    const int num_channels,
    const InputView input,
    const WeightView weight,
    const OutputView output)
{
    Kokkos::parallel_for(
        label,
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
            Kokkos::IndexType<std::size_t>>(
            execution_space, 0,
            static_cast<std::size_t>(num_nodes)*(l_max+1)),
        [=] (const std::size_t owner) {
            const std::size_t node = owner/(l_max+1);
            const int l = owner%(l_max+1);
            const int component_count = 2*l+1;
            symmetrix_blas_gemm<Precision>(
                CblasRowMajor, CblasNoTrans,
                TransposeWeight ? CblasTrans : CblasNoTrans,
                component_count, num_channels, num_channels,
                Precision(1), &input(node,l*l,0), num_channels,
                &weight(l,0,0), num_channels,
                Precision(0), &output(node,l*l,0), num_channels);
        });
}

}

template <typename Precision>
struct GlobalFieldH1ReverseReducer {
    using value_type = double[];

    static constexpr unsigned value_count = 3;

    int num_channels;
    int num_LM;
    int num_entries;
    Kokkos::View<Precision***,Kokkos::LayoutRight> H1_adj;
    Kokkos::View<Precision***,Kokkos::LayoutRight> H1_pre_adj;
    Kokkos::View<Precision***,Kokkos::LayoutRight> H1_pre_field;
    Kokkos::View<int*> entry_input_lm;
    Kokkos::View<int*> entry_component;
    Kokkos::View<int*> entry_output_lm;
    Kokkos::View<int*> entry_path;
    Kokkos::View<Precision*> entry_coefficient;
    Kokkos::View<Precision***,Kokkos::LayoutRight> path_matrix;
    Kokkos::View<const double*> electric_field;
    Kokkos::View<double*> electric_field_adj;

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t index, double local_field_adj[]) const {
        const std::size_t row = static_cast<std::size_t>(num_LM)*num_channels;
        const std::size_t i = index/row;
        const int input_lm =
            (index/static_cast<std::size_t>(num_channels))%num_LM;
        const int input = index%static_cast<std::size_t>(num_channels);
        Precision value = H1_adj(i,input_lm,input);
        const Precision input_feature = H1_pre_field(i,input_lm,input);
        for (int entry=0; entry<num_entries; ++entry) {
            if (entry_input_lm(entry) != input_lm)
                continue;
            const int component = entry_component(entry);
            const int output_lm = entry_output_lm(entry);
            const int path = entry_path(entry);
            const Precision coefficient = entry_coefficient(entry);
            Precision transformed_adjoint = Precision(0);
            for (int output=0; output<num_channels; ++output) {
                transformed_adjoint += path_matrix(path,input,output)
                    *H1_adj(i,output_lm,output);
            }
            const Precision contribution = coefficient*transformed_adjoint;
            value += static_cast<Precision>(electric_field(component))
                *contribution;
            local_field_adj[component] += static_cast<double>(
                input_feature*contribution);
        }
        H1_pre_adj(i,input_lm,input) = value;
    }

    KOKKOS_INLINE_FUNCTION
    void init(double update[]) const {
        for (int component=0; component<3; ++component)
            update[component] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    void join(double dst[], const double src[]) const {
        for (int component=0; component<3; ++component)
            dst[component] += src[component];
    }

    KOKKOS_INLINE_FUNCTION
    void final(double update[]) const {
        for (int component=0; component<3; ++component)
            electric_field_adj(component) = update[component];
    }
};

KOKKOS_INLINE_FUNCTION
double deterministic_gpu_wave32_sum(double value) {
#if defined(__CUDA_ARCH__)
    const unsigned mask = __activemask();
    const int lane = (
        threadIdx.x+blockDim.x*(threadIdx.y+blockDim.y*threadIdx.z))&31;
    for (int offset=16; offset>0; offset/=2) {
        const double other = Kokkos::shfl_down(value, offset, 32, mask);
        const int source_lane = lane+offset;
        if (source_lane < 32 && (mask&(1u<<source_lane)) != 0)
            value += other;
    }
#elif defined(__HIP_DEVICE_COMPILE__)
    for (int offset=16; offset>0; offset/=2)
        value += Kokkos::shfl_down(value, offset, 32);
#endif
    return value;
}
template <typename Precision>
void MACEKokkos<Precision>::compute_H1(
    const int num_nodes)
{
    const auto execution_space = use_factorized_async_inference()
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    if (H1.extent(0) < M0.extent(0))
        Kokkos::realloc(H1, M0.extent(0), M0.extent(1), M0.extent(2));

    auto L_max = this->L_max;
    const int channels = num_channels;
    auto H1 = this->H1;
    auto H1_weights = this->H1_weights;
    auto M0 = this->M0;

    if constexpr (std::is_same_v<
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>) {
        if (symmetrix::host_worker_cblas_enabled()
                && !symmetrix::host_dense_backend_overrides_cblas()) {
            launch_host_h1_gemm<false,Precision>(
                "Compute H1", execution_space, num_nodes, L_max,
                num_channels, M0, H1_weights, H1);
            complete_device_stage("MACEKokkos::compute_H1");
            return;
        }
        if (symmetrix::host_flat_dense_enabled()) {
            Kokkos::parallel_for(
                "Compute H1 flat",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(
                    execution_space, 0,
                    static_cast<std::size_t>(num_nodes)*(L_max+1)),
                [=] (const std::size_t owner) {
                    const std::size_t node = owner/(L_max+1);
                    const int l = owner%(L_max+1);
                    symmetrix::host_dense_gemm_nn(
                        &M0(node,l*l,0), &H1_weights(l,0,0),
                        &H1(node,l*l,0), 2*l+1,
                        channels, channels);
                });
            complete_device_stage("MACEKokkos::compute_H1");
            return;
        }
    }

    Kokkos::parallel_for("Compute H1",
        Kokkos::TeamPolicy<>(
            execution_space, num_nodes*(L_max+1),
            Kokkos::AUTO, Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank() / (L_max+1);
            const int l = team_member.league_rank() % (L_max+1);
            auto M0_il = Kokkos::subview(M0, i, Kokkos::make_pair(l*l, l*(l+2)+1), Kokkos::ALL);
            auto W_il = Kokkos::subview(H1_weights, l, Kokkos::ALL, Kokkos::ALL);
            auto H1_il = Kokkos::subview(H1, i, Kokkos::make_pair(l*l, l*(l+2)+1), Kokkos::ALL);
            KokkosBatched::TeamGemm<Kokkos::TeamPolicy<>::member_type,
                                    KokkosBatched::Trans::NoTranspose,
                                    KokkosBatched::Trans::NoTranspose,
                                    KokkosBatched::Algo::Gemm::Unblocked>
                ::invoke(team_member, 1.0, M0_il, W_il, 0.0, H1_il);
        });
    complete_device_stage("MACEKokkos::compute_H1");
}

template <typename Precision>
void MACEKokkos<Precision>::compute_H1_product(
    const int num_nodes)
{
    const auto execution_space = use_factorized_async_inference()
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    if (H1.extent(0) < M0.extent(0))
        Kokkos::realloc(H1, M0.extent(0), M0.extent(1), M0.extent(2));

    auto L_max = this->L_max;
    const int channels = num_channels;
    auto H1 = this->H1;
    auto H1_product_weights = this->H1_product_weights;
    auto M0 = this->M0;

    if constexpr (std::is_same_v<
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>) {
        if (symmetrix::host_worker_cblas_enabled()
                && !symmetrix::host_dense_backend_overrides_cblas()) {
            launch_host_h1_gemm<false,Precision>(
                "MACEKokkos::compute_H1_product", execution_space,
                num_nodes, L_max, num_channels,
                M0, H1_product_weights, H1);
            complete_device_stage("MACEKokkos::compute_H1_product");
            return;
        }
        if (symmetrix::host_flat_dense_enabled()) {
            Kokkos::parallel_for(
                "MACEKokkos::compute_H1_product_flat",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(
                    execution_space, 0,
                    static_cast<std::size_t>(num_nodes)*(L_max+1)),
                [=] (const std::size_t owner) {
                    const std::size_t node = owner/(L_max+1);
                    const int l = owner%(L_max+1);
                    symmetrix::host_dense_gemm_nn(
                        &M0(node,l*l,0), &H1_product_weights(l,0,0),
                        &H1(node,l*l,0), 2*l+1,
                        channels, channels);
                });
            complete_device_stage("MACEKokkos::compute_H1_product");
            return;
        }
    }

    Kokkos::parallel_for("MACEKokkos::compute_H1_product",
        Kokkos::TeamPolicy<>(
            execution_space, num_nodes*(L_max+1),
            Kokkos::AUTO, Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank() / (L_max+1);
            const int l = team_member.league_rank() % (L_max+1);
            auto M0_il = Kokkos::subview(M0, i, Kokkos::make_pair(l*l, l*(l+2)+1), Kokkos::ALL);
            auto W_il = Kokkos::subview(H1_product_weights, l, Kokkos::ALL, Kokkos::ALL);
            auto H1_il = Kokkos::subview(H1, i, Kokkos::make_pair(l*l, l*(l+2)+1), Kokkos::ALL);
            KokkosBatched::TeamGemm<Kokkos::TeamPolicy<>::member_type,
                                    KokkosBatched::Trans::NoTranspose,
                                    KokkosBatched::Trans::NoTranspose,
                                    KokkosBatched::Algo::Gemm::Unblocked>
                ::invoke(team_member, 1.0, M0_il, W_il, 0.0, H1_il);
        });
    complete_device_stage("MACEKokkos::compute_H1_product");
}

template <typename Precision>
void MACEKokkos<Precision>::add_H1_first_residual(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    const bool fused)
{
    if (!first_interaction_residual)
        return;
    const auto execution_space = use_factorized_async_inference()
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    const auto num_LM = this->num_LM;
    const auto num_channels = this->num_channels;
    const auto H1 = this->H1;
    const auto residual = fused
        ? H1_first_residual_fused_weights
        : H1_first_residual_weights;
    Kokkos::parallel_for(
        "MACEKokkos::add_H1_first_residual",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            execution_space, 0,
            static_cast<std::size_t>(num_nodes)*num_LM*num_channels),
        KOKKOS_LAMBDA (const std::size_t work) {
            const int k = work%num_channels;
            const std::size_t node_lm = work/num_channels;
            const int lm = node_lm%num_LM;
            const int node = node_lm/num_LM;
            H1(node,lm,k) += residual(node_types(node),lm,k);
        });
    complete_device_stage("MACEKokkos::add_H1_first_residual");
}

template <typename Precision>
void MACEKokkos<Precision>::compute_H1_linear_up(
    const int num_nodes)
{
    const auto execution_space = use_factorized_async_inference()
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    if (H1_pre_linear_up.extent(0) < H1.extent(0))
        Kokkos::realloc(H1_pre_linear_up, H1.extent(0), H1.extent(1), H1.extent(2));
    Kokkos::deep_copy(execution_space, H1_pre_linear_up, H1);

    auto L_max = this->L_max;
    const int channels = num_channels;
    auto H1 = this->H1;
    auto H1_pre_linear_up = this->H1_pre_linear_up;
    auto H1_linear_up_weights = this->H1_linear_up_weights;

    if constexpr (std::is_same_v<
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>) {
        if (symmetrix::host_worker_cblas_enabled()
                && !symmetrix::host_dense_backend_overrides_cblas()) {
            launch_host_h1_gemm<false,Precision>(
                "MACEKokkos::compute_H1_linear_up", execution_space,
                num_nodes, L_max, num_channels,
                H1_pre_linear_up, H1_linear_up_weights, H1);
            complete_device_stage("MACEKokkos::compute_H1_linear_up");
            return;
        }
        if (symmetrix::host_flat_dense_enabled()) {
            Kokkos::parallel_for(
                "MACEKokkos::compute_H1_linear_up_flat",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(
                    execution_space, 0,
                    static_cast<std::size_t>(num_nodes)*(L_max+1)),
                [=] (const std::size_t owner) {
                    const std::size_t node = owner/(L_max+1);
                    const int l = owner%(L_max+1);
                    symmetrix::host_dense_gemm_nn(
                        &H1_pre_linear_up(node,l*l,0),
                        &H1_linear_up_weights(l,0,0),
                        &H1(node,l*l,0), 2*l+1,
                        channels, channels);
                });
            complete_device_stage("MACEKokkos::compute_H1_linear_up");
            return;
        }
    }

    Kokkos::parallel_for("MACEKokkos::compute_H1_linear_up",
        Kokkos::TeamPolicy<>(
            execution_space, num_nodes*(L_max+1),
            Kokkos::AUTO, Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank() / (L_max+1);
            const int l = team_member.league_rank() % (L_max+1);
            auto H1_in_il = Kokkos::subview(H1_pre_linear_up, i, Kokkos::make_pair(l*l, l*(l+2)+1), Kokkos::ALL);
            auto W_il = Kokkos::subview(H1_linear_up_weights, l, Kokkos::ALL, Kokkos::ALL);
            auto H1_il = Kokkos::subview(H1, i, Kokkos::make_pair(l*l, l*(l+2)+1), Kokkos::ALL);
            KokkosBatched::TeamGemm<Kokkos::TeamPolicy<>::member_type,
                                    KokkosBatched::Trans::NoTranspose,
                                    KokkosBatched::Trans::NoTranspose,
                                    KokkosBatched::Algo::Gemm::Unblocked>
                ::invoke(team_member, 1.0, H1_in_il, W_il, 0.0, H1_il);
        });
    complete_device_stage("MACEKokkos::compute_H1_linear_up");
}

template <typename Precision>
void MACEKokkos<Precision>::reverse_H1(
    const int num_nodes)
{
    const auto execution_space = use_factorized_async_inference()
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    ensure_mh0_m0_adjoint_capacity();

    auto L_max = this->L_max;
    const int channels = num_channels;
    auto M0_adj = this->M0_adj;
    auto H1_weights = this->H1_weights;
    auto H1_weights_trans = this->H1_weights_trans;
    auto H1_adj = this->H1_adj;

    if constexpr (std::is_same_v<
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>) {
        if (symmetrix::host_worker_cblas_enabled()
                && !symmetrix::host_dense_backend_overrides_cblas()) {
            launch_host_h1_gemm<true,Precision>(
                "Reverse H1", execution_space, num_nodes, L_max,
                num_channels, H1_adj, H1_weights, M0_adj);
            complete_device_stage("MACEKokkos::reverse_H1");
            return;
        }
        if (symmetrix::host_flat_dense_enabled()) {
            Kokkos::parallel_for(
                "Reverse H1 flat",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(
                    execution_space, 0,
                    static_cast<std::size_t>(num_nodes)*(L_max+1)),
                [=] (const std::size_t owner) {
                    const std::size_t node = owner/(L_max+1);
                    const int l = owner%(L_max+1);
                    symmetrix::host_dense_gemm_nt(
                        &H1_adj(node,l*l,0), &H1_weights(l,0,0),
                        &M0_adj(node,l*l,0), 2*l+1,
                        channels, channels);
                });
            complete_device_stage("MACEKokkos::reverse_H1");
            return;
        }
    }

    Kokkos::parallel_for("Reverse H1",
        Kokkos::TeamPolicy<>(
            execution_space, num_nodes*(L_max+1),
            Kokkos::AUTO, Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank() / (L_max+1);
            const int l = team_member.league_rank() % (L_max+1);
            auto H1_adj_il = Kokkos::subview(H1_adj, i, Kokkos::make_pair(l*l, l*(l+2)+1), Kokkos::ALL);
            auto W_il = Kokkos::subview(
                H1_weights_trans, l, Kokkos::ALL, Kokkos::ALL);
            auto M0_adj_il = Kokkos::subview(M0_adj, i, Kokkos::make_pair(l*l, l*(l+2)+1), Kokkos::ALL);
            KokkosBatched::TeamGemm<Kokkos::TeamPolicy<>::member_type,
                                    KokkosBatched::Trans::NoTranspose,
                                    KokkosBatched::Trans::NoTranspose,
                                    KokkosBatched::Algo::Gemm::Unblocked>
                ::invoke(team_member, 1.0, H1_adj_il, W_il, 0.0, M0_adj_il);
        });
    complete_device_stage("MACEKokkos::reverse_H1");
}

template <typename Precision>
void MACEKokkos<Precision>::reverse_H1_linear_up(
    const int num_nodes)
{
    const auto execution_space = use_factorized_async_inference()
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    if (H1_pre_linear_up.extent(0) < H1.extent(0))
        throw std::runtime_error("MACEKokkos::reverse_H1_linear_up requires saved pre-linear-up H1.");

    auto L_max = this->L_max;
    const int channels = num_channels;
    auto H1_adj = this->H1_adj;
    auto H1_linear_up_weights = this->H1_linear_up_weights;
    auto H1_pre_linear_up = this->H1_pre_linear_up;

    if constexpr (std::is_same_v<
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>) {
        if (symmetrix::host_worker_cblas_enabled()
                && !symmetrix::host_dense_backend_overrides_cblas()) {
            launch_host_h1_gemm<true,Precision>(
                "MACEKokkos::reverse_H1_linear_up", execution_space,
                num_nodes, L_max, num_channels,
                H1_adj, H1_linear_up_weights, H1_pre_linear_up);
            Kokkos::deep_copy(execution_space, H1_adj, H1_pre_linear_up);
            complete_device_stage("MACEKokkos::reverse_H1_linear_up");
            return;
        }
        if (symmetrix::host_flat_dense_enabled()) {
            Kokkos::parallel_for(
                "MACEKokkos::reverse_H1_linear_up_flat",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(
                    execution_space, 0,
                    static_cast<std::size_t>(num_nodes)*(L_max+1)),
                [=] (const std::size_t owner) {
                    const std::size_t node = owner/(L_max+1);
                    const int l = owner%(L_max+1);
                    symmetrix::host_dense_gemm_nt(
                        &H1_adj(node,l*l,0),
                        &H1_linear_up_weights(l,0,0),
                        &H1_pre_linear_up(node,l*l,0), 2*l+1,
                        channels, channels);
                });
            Kokkos::deep_copy(execution_space, H1_adj, H1_pre_linear_up);
            complete_device_stage("MACEKokkos::reverse_H1_linear_up");
            return;
        }
    }

    Kokkos::parallel_for("MACEKokkos::reverse_H1_linear_up",
        Kokkos::TeamPolicy<>(
            execution_space, num_nodes*(L_max+1),
            Kokkos::AUTO, Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank() / (L_max+1);
            const int l = team_member.league_rank() % (L_max+1);
            auto H1_adj_il = Kokkos::subview(H1_adj, i, Kokkos::make_pair(l*l, l*(l+2)+1), Kokkos::ALL);
            auto W_il = Kokkos::subview(H1_linear_up_weights, l, Kokkos::ALL, Kokkos::ALL);
            auto H1_pre_adj_il = Kokkos::subview(H1_pre_linear_up, i, Kokkos::make_pair(l*l, l*(l+2)+1), Kokkos::ALL);
            KokkosBatched::TeamGemm<Kokkos::TeamPolicy<>::member_type,
                                    KokkosBatched::Trans::NoTranspose,
                                    KokkosBatched::Trans::Transpose,
                                    KokkosBatched::Algo::Gemm::Unblocked>
                ::invoke(team_member, 1.0, H1_adj_il, W_il, 0.0, H1_pre_adj_il);
        });
    Kokkos::deep_copy(execution_space, H1_adj, H1_pre_linear_up);
    complete_device_stage("MACEKokkos::reverse_H1_linear_up");
}

template <typename Precision>
void MACEKokkos<Precision>::reverse_H1_product(
    const int num_nodes)
{
    const auto execution_space = use_factorized_async_inference()
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    ensure_mh0_m0_adjoint_capacity();

    auto L_max = this->L_max;
    const int channels = num_channels;
    auto M0_adj = this->M0_adj;
    auto H1_product_weights = this->H1_product_weights;
    auto H1_adj = this->H1_adj;

    if constexpr (std::is_same_v<
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>) {
        if (symmetrix::host_worker_cblas_enabled()
                && !symmetrix::host_dense_backend_overrides_cblas()) {
            launch_host_h1_gemm<true,Precision>(
                "MACEKokkos::reverse_H1_product", execution_space,
                num_nodes, L_max, num_channels,
                H1_adj, H1_product_weights, M0_adj);
            complete_device_stage("MACEKokkos::reverse_H1_product");
            return;
        }
        if (symmetrix::host_flat_dense_enabled()) {
            Kokkos::parallel_for(
                "MACEKokkos::reverse_H1_product_flat",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(
                    execution_space, 0,
                    static_cast<std::size_t>(num_nodes)*(L_max+1)),
                [=] (const std::size_t owner) {
                    const std::size_t node = owner/(L_max+1);
                    const int l = owner%(L_max+1);
                    symmetrix::host_dense_gemm_nt(
                        &H1_adj(node,l*l,0),
                        &H1_product_weights(l,0,0),
                        &M0_adj(node,l*l,0), 2*l+1,
                        channels, channels);
                });
            complete_device_stage("MACEKokkos::reverse_H1_product");
            return;
        }
    }

    Kokkos::parallel_for("MACEKokkos::reverse_H1_product",
        Kokkos::TeamPolicy<>(
            execution_space, num_nodes*(L_max+1),
            Kokkos::AUTO, Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank() / (L_max+1);
            const int l = team_member.league_rank() % (L_max+1);
            auto H1_adj_il = Kokkos::subview(H1_adj, i, Kokkos::make_pair(l*l, l*(l+2)+1), Kokkos::ALL);
            auto W_il = Kokkos::subview(H1_product_weights, l, Kokkos::ALL, Kokkos::ALL);
            auto M0_adj_il = Kokkos::subview(M0_adj, i, Kokkos::make_pair(l*l, l*(l+2)+1), Kokkos::ALL);
            KokkosBatched::TeamGemm<Kokkos::TeamPolicy<>::member_type,
                                    KokkosBatched::Trans::NoTranspose,
                                    KokkosBatched::Trans::Transpose,
                                    KokkosBatched::Algo::Gemm::Unblocked>
                ::invoke(team_member, 1.0, H1_adj_il, W_il, 0.0, M0_adj_il);
        });
    complete_device_stage("MACEKokkos::reverse_H1_product");
}

template <typename Precision>
void MACEKokkos<Precision>::compute_field_H1(
    const int num_nodes,
    Kokkos::View<const double*> electric_field)
{
    const auto execution_space = use_factorized_async_inference()
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    if (!has_field_coupling)
        return;

    if (electric_field.size() != 3 && electric_field.size() != 3*num_nodes)
        throw std::runtime_error("MACEField electric_field must have shape (3,) or (num_nodes, 3).");
    if (H1.extent(0) < num_nodes || H1.extent(1) != num_LM || H1.extent(2) != num_channels)
        throw std::runtime_error("MACEField H1 buffer size does not match num_nodes.");

    if (H1_pre_field.extent(0) < H1.extent(0))
        Kokkos::realloc(H1_pre_field, H1.extent(0), H1.extent(1), H1.extent(2));
    Kokkos::deep_copy(execution_space, H1_pre_field, H1);

    const bool global_field = electric_field.size() == 3;
    const auto num_channels = this->num_channels;
    const auto num_LM = this->num_LM;
    const auto num_entries = this->num_field_angular_entries;
    const auto H1_pre_field = this->H1_pre_field;
    const auto H1 = this->H1;
    const auto entry_input_lm = field_entry_input_lm;
    const auto entry_component = field_entry_component;
    const auto entry_output_lm = field_entry_output_lm;
    const auto entry_path = field_entry_path;
    const auto entry_coefficient = field_entry_coefficient;
    const auto path_matrix = field_path_matrix;

    Kokkos::parallel_for(
        "MACEKokkos::compute_field_H1",
        Kokkos::MDRangePolicy<
            Kokkos::Rank<3,Kokkos::Iterate::Right>,
            Kokkos::IndexType<std::size_t>>(
            execution_space, {0,0,0}, {num_nodes,num_LM,num_channels}),
        KOKKOS_LAMBDA (
            const std::size_t i,
            const std::size_t output_lm,
            const std::size_t output) {
        Precision value = H1_pre_field(i,output_lm,output);
        const std::size_t field_offset = global_field ? 0 : 3*i;
        for (int entry=0; entry<num_entries; ++entry) {
            if (entry_output_lm(entry) != output_lm)
                continue;
            const Precision angular_field = entry_coefficient(entry)
                * static_cast<Precision>(
                    electric_field(field_offset+entry_component(entry)));
            const int path = entry_path(entry);
            for (int input=0; input<num_channels; ++input) {
                value += angular_field
                    * H1_pre_field(i,entry_input_lm(entry),input)
                    * path_matrix(path,input,output);
            }
        }
        H1(i,output_lm,output) = value;
    });
    complete_device_stage("MACEKokkos::compute_field_H1");
}

template <typename Precision>
void MACEKokkos<Precision>::reverse_field_H1(
    const int num_nodes,
    Kokkos::View<const double*> electric_field)
{
    const auto execution_space = use_factorized_async_inference()
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    if (!has_field_coupling)
        return;

    if (electric_field.size() != 3
        && electric_field.size() != 3*static_cast<std::size_t>(num_nodes))
        throw std::runtime_error("MACEField electric_field must have shape (3,) or (num_nodes, 3).");
    if (H1_adj.extent(0) < num_nodes || H1_pre_field.extent(0) < num_nodes)
        throw std::runtime_error("MACEField reverse_field_H1 requires H1_adj and saved pre-field H1 buffers.");

    const auto num_channels = this->num_channels;
    const auto num_entries = this->num_field_angular_entries;
    const auto H1_adj = this->H1_adj;
    const auto H1_pre_field = this->H1_pre_field;
    const auto entry_input_lm = field_entry_input_lm;
    const auto entry_component = field_entry_component;
    const auto entry_output_lm = field_entry_output_lm;
    const auto entry_path = field_entry_path;
    const auto component_entry_offsets = field_component_entry_offsets;
    const auto component_entries = field_component_entries;
    const auto entry_coefficient = field_entry_coefficient;
    const auto path_matrix = field_path_matrix;

    const bool global_field = electric_field.size() == 3;
    const std::int64_t h1_lm = H1_adj.extent(1);
    const std::int64_t h1_channels = H1_adj.extent(2);

    if (this->electric_field_adj.size() != electric_field.size())
        Kokkos::realloc(this->electric_field_adj, electric_field.size());
    auto electric_field_adj = this->electric_field_adj;

    if (field_H1_pre_adj.extent(0) < num_nodes
        || field_H1_pre_adj.extent(1) < h1_lm
        || field_H1_pre_adj.extent(2) < h1_channels)
        Kokkos::realloc(field_H1_pre_adj, num_nodes, h1_lm, h1_channels);

    auto H1_pre_adj = this->field_H1_pre_adj;
    if (global_field) {
        const std::size_t input_count =
            static_cast<std::size_t>(num_nodes)*h1_lm*h1_channels;
        #if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
        if (use_gpu_field_h1_reverse()) {
            constexpr std::size_t wave_size = 32;
            constexpr std::size_t threads_per_block = 256;
            const std::size_t required_blocks =
                (input_count+threads_per_block-1)/threads_per_block;
            const int persistent_blocks = resolve_execution_persistent_blocks(
                symmetrix::execution::field_h1_reverse_id,
                execution_space, required_blocks, "field_h1_reverse");
            if (persistent_blocks <= 0)
                throw std::runtime_error(
                    "MACEField H1 reverse has no compatible GPU launch profile.");
            const std::size_t worker_count =
                static_cast<std::size_t>(persistent_blocks)*threads_per_block;
            const std::size_t partial_count = worker_count/wave_size;
            if (field_H1_global_partial.extent(0) < partial_count)
                Kokkos::realloc(
                    Kokkos::WithoutInitializing,
                    field_H1_global_partial, partial_count, 3);
            auto field_partial = field_H1_global_partial;
            Kokkos::parallel_for(
                "MACEKokkos::reverse_field_H1_global_features",
                Kokkos::TeamPolicy<>(
                    execution_space, persistent_blocks, threads_per_block),
                KOKKOS_LAMBDA (
                    const Kokkos::TeamPolicy<>::member_type& team_member) {
                const std::size_t worker =
                    static_cast<std::size_t>(team_member.league_rank())
                        *threads_per_block
                    +static_cast<std::size_t>(team_member.team_rank());
                double field_0 = 0.0;
                double field_1 = 0.0;
                double field_2 = 0.0;
                const std::size_t row =
                    static_cast<std::size_t>(h1_lm)*h1_channels;
                for (std::size_t index=worker; index<input_count;
                     index+=worker_count) {
                    const std::size_t i = index/row;
                    const int input_lm = static_cast<int>(
                        (index/static_cast<std::size_t>(h1_channels))%h1_lm);
                    const int input =
                        index%static_cast<std::size_t>(h1_channels);
                    Precision value = H1_adj(i,input_lm,input);
                    const Precision input_feature =
                        H1_pre_field(i,input_lm,input);
                    for (int entry=0; entry<num_entries; ++entry) {
                        if (entry_input_lm(entry) != input_lm)
                            continue;
                        const int component = entry_component(entry);
                        const int path = entry_path(entry);
                        Precision transformed_adjoint = Precision(0);
                        for (int output=0; output<num_channels; ++output) {
                            transformed_adjoint += path_matrix(path,input,output)
                                *H1_adj(i,entry_output_lm(entry),output);
                        }
                        const Precision contribution =
                            entry_coefficient(entry)*transformed_adjoint;
                        value += static_cast<Precision>(electric_field(component))
                            *contribution;
                        const double field_contribution = static_cast<double>(
                            input_feature*contribution);
                        if (component == 0)
                            field_0 += field_contribution;
                        else if (component == 1)
                            field_1 += field_contribution;
                        else
                            field_2 += field_contribution;
                    }
                    H1_pre_adj(i,input_lm,input) = value;
                }
                field_0 = deterministic_gpu_wave32_sum(field_0);
                field_1 = deterministic_gpu_wave32_sum(field_1);
                field_2 = deterministic_gpu_wave32_sum(field_2);
                if ((worker&(wave_size-1)) == 0) {
                    const std::size_t partial = worker/wave_size;
                    field_partial(partial,0) = field_0;
                    field_partial(partial,1) = field_1;
                    field_partial(partial,2) = field_2;
                }
                });
            for (int component=0; component<3; ++component) {
                auto component_adjoint =
                    Kokkos::subview(electric_field_adj, component);
                using ComponentMemorySpace =
                    typename decltype(component_adjoint)::memory_space;
                Kokkos::parallel_reduce(
                    "MACEKokkos::reverse_field_H1_global_reduce",
                    Kokkos::RangePolicy<
                        Kokkos::DefaultExecutionSpace,
                        Kokkos::IndexType<std::size_t>>(
                        execution_space, 0, partial_count),
                    KOKKOS_LAMBDA (
                        const std::size_t partial, double& value) {
                        value += field_partial(partial,component);
                    },
                    Kokkos::Sum<double,ComponentMemorySpace>(
                        component_adjoint));
            }
        } else
        #endif
        {
            Kokkos::parallel_reduce(
                "MACEKokkos::reverse_field_H1_global",
                Kokkos::RangePolicy<
                    Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(
                    execution_space, 0, input_count),
                GlobalFieldH1ReverseReducer<Precision>{
                    num_channels, static_cast<int>(h1_lm), num_entries,
                    H1_adj, H1_pre_adj, H1_pre_field,
                    entry_input_lm, entry_component, entry_output_lm,
                    entry_path, entry_coefficient, path_matrix,
                    electric_field, electric_field_adj});
        }
    } else {
        Kokkos::parallel_for(
            "MACEKokkos::reverse_field_H1_features",
            Kokkos::MDRangePolicy<
                Kokkos::Rank<3,Kokkos::Iterate::Right>,
                Kokkos::IndexType<std::size_t>>(
                execution_space, {0,0,0}, {num_nodes,h1_lm,h1_channels}),
            KOKKOS_LAMBDA (
                const std::size_t i,
                const std::size_t input_lm,
                const std::size_t input) {
                Precision value = H1_adj(i,input_lm,input);
                const std::size_t field_offset = 3*i;
                for (int entry=0; entry<num_entries; ++entry) {
                    if (entry_input_lm(entry) != input_lm)
                        continue;
                    const Precision angular_field = entry_coefficient(entry)
                        *static_cast<Precision>(electric_field(
                            field_offset+entry_component(entry)));
                    const int path = entry_path(entry);
                    for (int output=0; output<num_channels; ++output) {
                        value += angular_field*path_matrix(path,input,output)
                            *H1_adj(i,entry_output_lm(entry),output);
                    }
                }
                H1_pre_adj(i,input_lm,input) = value;
            });

        Kokkos::View<double**,Kokkos::LayoutRight> node_field_adj(
            "MACEField node field adjoint", num_nodes, 3);
        Kokkos::parallel_for(
            "MACEKokkos::reverse_field_H1_field",
            Kokkos::MDRangePolicy<
                Kokkos::Rank<2,Kokkos::Iterate::Right>,
                Kokkos::IndexType<std::size_t>>(
                execution_space, {0,0}, {num_nodes,3}),
            KOKKOS_LAMBDA (const std::size_t i, const std::size_t component) {
                double value = 0.0;
                for (int entry=0; entry<num_entries; ++entry) {
                    if (entry_component(entry) != component)
                        continue;
                    const int path = entry_path(entry);
                    for (int input=0; input<num_channels; ++input) {
                        const Precision input_value =
                            H1_pre_field(i,entry_input_lm(entry),input);
                        for (int output=0; output<num_channels; ++output) {
                            value += static_cast<double>(entry_coefficient(entry)
                                *path_matrix(path,input,output)*input_value
                                *H1_adj(i,entry_output_lm(entry),output));
                        }
                    }
                }
                node_field_adj(i,component) = value;
            });
        Kokkos::parallel_for(
            "MACEKokkos::reverse_field_H1_field_copy",
            Kokkos::MDRangePolicy<
                Kokkos::Rank<2,Kokkos::Iterate::Right>,
                Kokkos::IndexType<std::size_t>>(
                execution_space, {0,0}, {num_nodes,3}),
            KOKKOS_LAMBDA (const std::size_t i, const std::size_t component) {
                electric_field_adj(3*i+component) = node_field_adj(i,component);
            });
    }
    Kokkos::parallel_for(
        "MACEKokkos::reverse_field_H1_copy_out",
        Kokkos::MDRangePolicy<
            Kokkos::Rank<3,Kokkos::Iterate::Right>,
            Kokkos::IndexType<std::size_t>>(
            execution_space, {0,0,0}, {num_nodes,h1_lm,h1_channels}),
        KOKKOS_LAMBDA (
            const std::size_t i,
            const std::size_t lm,
            const std::size_t channel) {
            H1_adj(i,lm,channel) = H1_pre_adj(i,lm,channel);
        });
    complete_device_stage("MACEKokkos::reverse_field_H1");
}

template <typename Precision>
void MACEKokkos<Precision>::compute_Phi1(
    const int num_nodes,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices)
{
    // Compute Phi1_lelm1lm2 (named Phi1r)
    if (Phi1r.extent(0) < num_nodes)
        Kokkos::realloc(Phi1r, num_nodes, num_lelm1lm2, num_channels);
    ensure_mh0_phi1_forward_capacity(num_nodes, num_channels);
    Kokkos::deep_copy(Phi1r, 0.0);
    Kokkos::deep_copy(Phi1, 0.0);

    Kokkos::View<int*> first_neigh("first_neigh", num_nodes);
    Kokkos::parallel_scan("Compute first_neigh",
        num_nodes,
        KOKKOS_LAMBDA (const int i, int& update, const bool final) {
            if (final)
                first_neigh(i) = update;
            update += num_neigh(i);
        });

    const auto num_channels = this->num_channels;
    const auto num_lm = this->num_lm;
    const auto num_lelm1lm2 = this->num_lelm1lm2;
    const auto Phi1_lm1 = this->Phi1_lm1;
    const auto Phi1_lm2 = this->Phi1_lm2;
    const auto Phi1_lel1l2 = this->Phi1_lel1l2;
    const auto Phi1_lme = this->Phi1_lme;
    const auto Phi1_lelm1lm2 = this->Phi1_lelm1lm2;
    const auto Phi1_clebsch_gordan = this->Phi1_clebsch_gordan;
    const auto R1 = this->R1;
    const auto Y = this->Y;
    const auto H1 = this->H1;
    auto Phi1 = this->Phi1;
    auto Phi1r = this->Phi1r;

    const auto phi1r_scratch_bytes = admitted_team_scratch_bytes<>(
        "MACEKokkos::compute_Phi1r",
        {static_cast<std::size_t>(num_channels), sizeof(double)});
    Kokkos::parallel_for("Compute Phi1r",
        Kokkos::TeamPolicy<>(num_nodes*num_lelm1lm2, Kokkos::AUTO, 32)
             .set_scratch_size(0, Kokkos::PerTeam(phi1r_scratch_bytes)),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const std::size_t node = static_cast<std::size_t>(
                team_member.league_rank()/num_lelm1lm2);
            const int lelm1lm2 = team_member.league_rank() % num_lelm1lm2;
            const int i0 = first_neigh(node);
            const int lm1 = Phi1_lm1(lelm1lm2);
            const int lm2 = Phi1_lm2(lelm1lm2);
            const int lel1l2 = Phi1_lel1l2(lelm1lm2);
            // initialize Phi1r_i_lelm1lm2 in scratch space
            auto Phi1r_i_lelm1lm2 = Kokkos::View<double*>(team_member.team_scratch(0), num_channels);
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team_member, num_channels),
                [=] (const int k) {
                    Phi1r_i_lelm1lm2(k) = 0.0;
                });
            team_member.team_barrier();
            // compute Phi1r_i_lelm1lm2
            for (int j=0; j<num_neigh(node); ++j) {
                const int ij = i0 + j;
                const std::size_t edge = static_cast<std::size_t>(ij);
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team_member, num_channels),
                    [=] (const int k) {
                        Phi1r_i_lelm1lm2(k) += R1(ij,lel1l2*num_channels+k) * Y(edge*num_lm+lm1) * H1(neigh_indices(ij),lm2,k);
                    });
            }
            team_member.team_barrier();
            // store Phi1r_i_lelm1lm2
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team_member, num_channels),
                [=] (const int k) {
                    Phi1r(node,lelm1lm2,k) = Phi1r_i_lelm1lm2(k);
                });
        });
    Kokkos::fence();
    // Compute Phi1 using CG coefficients
    Kokkos::parallel_for("Compute Phi1",
        Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const std::size_t node = static_cast<std::size_t>(
                team_member.league_rank());
            for (int p=0; p<Phi1_clebsch_gordan.size(); ++p) {
                const double C = Phi1_clebsch_gordan(p);
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team_member, num_channels),
                    [&] (const int k) {
                        Phi1(node,Phi1_lme(p),k) += C * Phi1r(node,Phi1_lelm1lm2(p),k);
                    });
            }
        });
    Kokkos::fence();
}

template <typename Precision>
void MACEKokkos<Precision>::compute_Phi1_streamed(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r)
{
    const bool async_inference = use_factorized_async_inference();
    const auto execution_space = async_inference
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    if (Phi1r.extent(0) < num_nodes)
        Kokkos::realloc(Phi1r, num_nodes, num_lelm1lm2, num_channels);
    ensure_mh0_phi1_forward_capacity(num_nodes, num_channels);
    if (async_inference) {
        Kokkos::deep_copy(execution_space, Phi1r, Precision(0));
        Kokkos::deep_copy(execution_space, Phi1, Precision(0));
    } else {
        Kokkos::deep_copy(Phi1r, Precision(0));
        Kokkos::deep_copy(Phi1, Precision(0));
    }

    const int num_channels = this->num_channels;
    const int num_lm = this->num_lm;
    const int num_paths = Phi1_l.extent(0);
    const int num_types = num_active_types;
    const auto type_to_active = this->type_to_active;
    const auto first_neigh = streamed_first_neigh;
    const auto Phi1_lm1 = this->Phi1_lm1;
    const auto Phi1_lm2 = this->Phi1_lm2;
    const auto path_row_offsets = this->Phi1_path_row_offsets;
    const auto radial_1 = this->radial_1;
    const auto Y = this->Y;
    const auto H1 = this->H1;
    auto Phi1r = this->Phi1r;

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    const Kokkos::TeamPolicy<> streamed_policy(
        execution_space, num_nodes*num_paths, 1, 32);
#else
    const Kokkos::TeamPolicy<> streamed_policy(
        execution_space, num_nodes*num_paths, Kokkos::AUTO, 32);
#endif
    Kokkos::parallel_for(
        "MACEKokkos::compute_Phi1r_streamed",
        streamed_policy,
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const std::size_t node = static_cast<std::size_t>(
                team_member.league_rank()/num_paths);
            const int path = team_member.league_rank()%num_paths;
            const int row_begin = path_row_offsets(path);
            const int row_end = path_row_offsets(path+1);
            const int type_i = type_to_active(node_types(node));
            const int i0 = first_neigh(node);
            for (int j=0; j<num_neigh(node); ++j) {
                const int ij = i0+j;
                const std::size_t edge = static_cast<std::size_t>(ij);
                const int type_j = type_to_active(neigh_types(ij));
                const int edge_type = (type_i <= type_j)
                    ? type_i*(2*num_types-type_i-1)/2+type_j
                    : type_j*(2*num_types-type_j-1)/2+type_i;
                const auto point = radial_1.evaluation_point(r(ij));
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team_member, num_channels),
                    [=] (const int k) {
                        const Precision radial = radial_1.evaluate_function(
                            edge_type, point, path*num_channels+k);
                        for (int row=row_begin; row<row_end; ++row)
                            Phi1r(node,row,k) += radial
                                *Y(edge*num_lm+Phi1_lm1(row))
                                *H1(neigh_indices(ij),Phi1_lm2(row),k);
                    });
            }
        });

    const auto Phi1_lme = this->Phi1_lme;
    const auto Phi1_lelm1lm2 = this->Phi1_lelm1lm2;
    const auto Phi1_clebsch_gordan = this->Phi1_clebsch_gordan;
    auto Phi1 = this->Phi1;
    Kokkos::parallel_for(
        "MACEKokkos::compute_Phi1_streamed",
        Kokkos::TeamPolicy<>(
            execution_space, num_nodes, Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const std::size_t node = static_cast<std::size_t>(
                team_member.league_rank());
            for (int p=0; p<Phi1_clebsch_gordan.size(); ++p) {
                const Precision coefficient = Phi1_clebsch_gordan(p);
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team_member, num_channels),
                    [=] (const int k) {
                        Phi1(node,Phi1_lme(p),k) += coefficient
                            *Phi1r(node,Phi1_lelm1lm2(p),k);
                    });
            }
        });
    complete_device_stage("MACEKokkos::compute_Phi1_streamed");
}

template <typename Precision>
void MACEKokkos<Precision>::compute_Phi1_streamed_jit(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r)
{
    if (!use_factorized_direct_jit_forward())
        throw std::logic_error(
            "Generated Execution R1 forward was called without an admitted artifact.");
    const bool receiver_local_phi1 = use_receiver_local_phi1();
    const bool channel_tiled_phi1 = use_channel_tiled_phi1();
    if (receiver_local_phi1) {
        ensure_mh0_a1_forward_capacity(num_nodes);
        dPhi1 = decltype(dPhi1)();
        Phi1 = decltype(Phi1)();
    } else if (channel_tiled_phi1) {
        ensure_mh0_a1_forward_capacity(num_nodes);
        ensure_mh0_phi1_forward_capacity(num_nodes, phi1_channel_tile_size);
    } else
        ensure_mh0_phi1_forward_capacity(num_nodes, num_channels);

    // Runtime rollback may have populated this capacity in an earlier evaluation.
    Phi1r = decltype(Phi1r)();

    {
#if defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_CUDA)
        if (jit_device_plugin_ready()) {
            if constexpr (!device_execution_space<
                    Kokkos::DefaultExecutionSpace>) {
                throw std::runtime_error(
                    "Execution device plugins require device execution.");
            } else {
#endif
#if defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_CUDA)
                static_assert(std::is_same_v<int,std::int32_t>);
                require_execution_device_packet(num_nodes >= 0, "negative node count");
                require_execution_device_packet(
                    neigh_indices.extent(0)
                        <= static_cast<std::size_t>(
                            std::numeric_limits<std::int32_t>::max()),
                    "edge count exceeds the CUDA ABI");
                const int num_edges = static_cast<int>(neigh_indices.extent(0));
                const int channels = num_channels;
                const auto radial = make_execution_device_radial_spline(radial_1);
                const int expected_edge_types =
                    num_active_types*(num_active_types+1)/2;
                const int expected_functions =
                    static_cast<int>(Phi1_l.extent(0))*channels;

                if (receiver_local_phi1) {
                    require_execution_device_packet(
                        node_types.extent(0) >= static_cast<std::size_t>(num_nodes)
                            && num_neigh.extent(0)
                                >= static_cast<std::size_t>(num_nodes)
                            && streamed_first_neigh.extent(0)
                                >= static_cast<std::size_t>(num_nodes)
                            && H1.extent(0) >= static_cast<std::size_t>(num_nodes)
                            && A1.extent(0) >= static_cast<std::size_t>(num_nodes),
                        "projected R1 node tensor extent is too small");
                    require_execution_device_packet(
                        execution_a1_projection_weights.data() != nullptr
                            && A1.span_is_contiguous(),
                        "projected R1 weights or output storage is invalid");
                    const ExecutionDeviceR1ProjectedForwardArgs args {
                        sizeof(ExecutionDeviceR1ProjectedForwardArgs),
                        static_cast<std::uint32_t>(num_active_types),
                        num_nodes,
                        num_edges,
                        reinterpret_cast<const std::int32_t*>(node_types.data()),
                        reinterpret_cast<const std::int32_t*>(num_neigh.data()),
                        reinterpret_cast<const std::int32_t*>(
                            streamed_first_neigh.data()),
                        reinterpret_cast<const std::int32_t*>(neigh_indices.data()),
                        reinterpret_cast<const std::int32_t*>(neigh_types.data()),
                        reinterpret_cast<const std::int32_t*>(type_to_active.data()),
                        r.data(),
                        radial,
                        Y.data(),
                        H1.data(),
                        execution_a1_projection_weights.data(),
                        A1.data(),
                        r_cut};
                    const int persistent_blocks =
                        resolve_jit_device_plugin_persistent_blocks(
                            *jit_device_plugin,
                            factorized_execution_space,
                            static_cast<std::size_t>(num_nodes),
                            "r1_projected_forward");
                    ExecutionDeviceBackend::DeviceGuard device_guard(
                        ExecutionDeviceBackend::device_ordinal(
                            factorized_execution_space));
                    ExecutionDeviceBackend::check_status(
                        jit_device_plugin->launch_r1_projected_forward(
                            &args,
                            reinterpret_cast<void*>(
                                ExecutionDeviceBackend::native_stream(
                                    factorized_execution_space)),
                            persistent_blocks),
                        "Launching the projected R1 forward module");
                    factorized_jit_launch_count += 1;
                    factorized_jit_forward_launch_count += 1;
                    return;
                }

                if (channel_tiled_phi1) {
                    require_execution_device_packet(
                        radial.edge_types == static_cast<std::uint32_t>(
                            expected_edge_types)
                            && radial.functions == static_cast<std::uint32_t>(
                                expected_functions)
                            && radial.intervals > 0 && radial.h > 0.0
                            && std::isfinite(r_cut) && r_cut > 0.0
                            && radial.coefficients != nullptr,
                        "tiled R1 radial spline storage is invalid");
                    require_execution_device_packet(
                        node_types.extent(0) >= static_cast<std::size_t>(num_nodes)
                            && num_neigh.extent(0)
                                >= static_cast<std::size_t>(num_nodes)
                            && streamed_first_neigh.extent(0)
                                >= static_cast<std::size_t>(num_nodes)
                            && H1.extent(0) >= static_cast<std::size_t>(num_nodes)
                            && Phi1.extent(0)
                                >= static_cast<std::size_t>(num_nodes)
                            && Phi1.extent_int(1) == num_lme
                            && Phi1.extent_int(2) == phi1_channel_tile_size,
                        "tiled R1 node tensor shape is inconsistent");
                    require_execution_device_packet(
                        Phi1.span_is_contiguous() && A1.span_is_contiguous(),
                        "tiled R1 storage is not contiguous");
                    const int persistent_blocks =
                        resolve_jit_device_plugin_persistent_blocks(
                            *jit_device_plugin,
                            factorized_execution_space,
                            static_cast<std::size_t>(num_nodes)
                                *phi1_channel_tile_size,
                            "r1_tiled_forward");
                    ExecutionDeviceBackend::DeviceGuard device_guard(
                        ExecutionDeviceBackend::device_ordinal(
                            factorized_execution_space));
                    for (int channel_begin=0; channel_begin<channels;
                         channel_begin += phi1_channel_tile_size) {
                        const int channel_count = std::min(
                            phi1_channel_tile_size, channels-channel_begin);
                        const ExecutionDeviceR1TiledForwardArgs args {
                            sizeof(ExecutionDeviceR1TiledForwardArgs),
                            static_cast<std::uint32_t>(num_active_types),
                            static_cast<std::uint32_t>(channel_begin),
                            static_cast<std::uint32_t>(channel_count),
                            num_nodes,
                            num_edges,
                            reinterpret_cast<const std::int32_t*>(node_types.data()),
                            reinterpret_cast<const std::int32_t*>(num_neigh.data()),
                            reinterpret_cast<const std::int32_t*>(
                                streamed_first_neigh.data()),
                            reinterpret_cast<const std::int32_t*>(
                                neigh_indices.data()),
                            reinterpret_cast<const std::int32_t*>(
                                neigh_types.data()),
                            reinterpret_cast<const std::int32_t*>(
                                type_to_active.data()),
                            r.data(),
                            radial,
                            Y.data(),
                            H1.data(),
                            Phi1.data(),
                            r_cut};
                        ExecutionDeviceBackend::check_status(
                            jit_device_plugin->launch_r1_tiled_forward(
                                &args,
                                reinterpret_cast<void*>(
                                    ExecutionDeviceBackend::native_stream(
                                        factorized_execution_space)),
                                persistent_blocks),
                            "Launching the tiled R1 forward module");
                        compute_A1_channel_tile(num_nodes, channel_begin);
                        factorized_jit_launch_count += 1;
                        factorized_jit_forward_launch_count += 1;
                    }
                    return;
                }

                require_execution_device_packet(
                    radial.edge_types == static_cast<std::uint32_t>(
                        expected_edge_types),
                    "radial edge-type extent does not match active types");
                require_execution_device_packet(
                    radial.functions == static_cast<std::uint32_t>(
                        expected_functions),
                    "radial function extent does not match R1 paths");
                require_execution_device_packet(
                    radial.intervals > 0 && radial.h > 0.0
                        && std::isfinite(r_cut) && r_cut > 0.0
                        && radial.coefficients != nullptr,
                    "radial spline storage is invalid");
                require_execution_device_packet(
                    node_types.extent(0) >= static_cast<std::size_t>(num_nodes)
                        && num_neigh.extent(0)
                            >= static_cast<std::size_t>(num_nodes)
                        && streamed_first_neigh.extent(0)
                            >= static_cast<std::size_t>(num_nodes)
                        && H1.extent(0) >= static_cast<std::size_t>(num_nodes)
                        && Phi1.extent(0)
                            >= static_cast<std::size_t>(num_nodes),
                    "node tensor extent is too small");
                require_execution_device_packet(
                    neigh_types.extent(0) >= static_cast<std::size_t>(num_edges)
                        && r.extent(0) >= static_cast<std::size_t>(num_edges),
                    "edge tensor extent is too small");
                require_execution_device_packet(
                    type_to_active.extent(0)
                        >= static_cast<std::size_t>(num_elements),
                    "type map extent is too small");
                require_execution_device_packet(
                    Y.extent(0) >= static_cast<std::size_t>(num_edges)*num_lm
                        && H1.extent(1)
                            >= static_cast<std::size_t>(num_LM)
                        && H1.extent(2)
                            >= static_cast<std::size_t>(channels)
                        && Phi1.extent(1)
                            >= static_cast<std::size_t>(num_lme)
                        && Phi1.extent(2)
                            >= static_cast<std::size_t>(channels),
                    "R1 tensor shape is inconsistent with the plugin contract");
                require_execution_device_packet(
                    num_nodes == 0
                        || (node_types.data() != nullptr
                            && num_neigh.data() != nullptr
                            && streamed_first_neigh.data() != nullptr
                            && type_to_active.data() != nullptr
                            && H1.data() != nullptr
                            && Phi1.data() != nullptr),
                    "required node tensor pointer is null");
                require_execution_device_packet(
                    num_edges == 0
                        || (neigh_indices.data() != nullptr
                            && neigh_types.data() != nullptr
                            && r.data() != nullptr
                            && Y.data() != nullptr),
                    "required edge tensor pointer is null");
                require_execution_device_packet(
                    node_types.span_is_contiguous()
                        && num_neigh.span_is_contiguous()
                        && streamed_first_neigh.span_is_contiguous()
                        && neigh_indices.span_is_contiguous()
                        && neigh_types.span_is_contiguous()
                        && type_to_active.span_is_contiguous()
                        && r.span_is_contiguous()
                        && Y.span_is_contiguous()
                        && H1.span_is_contiguous()
                        && Phi1.span_is_contiguous(),
                    "tensor storage is not contiguous");

                const ExecutionDeviceR1ForwardArgs args {
                    sizeof(ExecutionDeviceR1ForwardArgs),
                    static_cast<std::uint32_t>(num_active_types),
                    num_nodes,
                    num_edges,
                    reinterpret_cast<const std::int32_t*>(node_types.data()),
                    reinterpret_cast<const std::int32_t*>(num_neigh.data()),
                    reinterpret_cast<const std::int32_t*>(
                        streamed_first_neigh.data()),
                    reinterpret_cast<const std::int32_t*>(neigh_indices.data()),
                    reinterpret_cast<const std::int32_t*>(neigh_types.data()),
                    reinterpret_cast<const std::int32_t*>(type_to_active.data()),
                    r.data(),
                    radial,
                    Y.data(),
                    H1.data(),
                    Phi1.data(),
                    r_cut};
                const int persistent_blocks =
                    resolve_jit_device_plugin_persistent_blocks(
                        *jit_device_plugin,
                        factorized_execution_space,
                        static_cast<std::size_t>(num_nodes)*num_channels,
                        "r1_forward");
                ExecutionDeviceBackend::DeviceGuard device_guard(
                    ExecutionDeviceBackend::device_ordinal(
                        factorized_execution_space));
                ExecutionDeviceBackend::check_status(
                    jit_device_plugin->launch_r1_forward(
                        &args,
                        reinterpret_cast<void*>(
                            ExecutionDeviceBackend::native_stream(
                                factorized_execution_space)),
                        persistent_blocks),
                    "Launching the Execution device R1 forward plugin");
                factorized_jit_launch_count += 1;
                factorized_jit_forward_launch_count += 1;
                return;
            }
        }
#endif
        if (jit_host_plugin_ready()) {
            if constexpr (!std::is_same_v<
                typename Kokkos::DefaultExecutionSpace::memory_space,
                Kokkos::HostSpace>) {
                throw std::runtime_error(
                    "Execution host plugins require host execution.");
            } else {
                static_assert(std::is_same_v<int,std::int32_t>);
                const int num_edges = static_cast<int>(neigh_indices.extent(0));
                const int channels = num_channels;
                const auto radial = make_execution_host_radial_spline(radial_1);
                const int expected_edge_types =
                    num_active_types*(num_active_types+1)/2;
                const int expected_functions =
                    static_cast<int>(Phi1_l.extent(0))*channels;

                require_execution_host_packet(num_nodes >= 0, "negative node count");
                require_execution_host_packet(
                    num_edges >= 0, "edge count exceeds the host ABI");
                require_execution_host_packet(
                    radial.edge_types == static_cast<std::uint32_t>(
                        expected_edge_types),
                    "radial edge-type extent does not match active types");
                require_execution_host_packet(
                    radial.functions == static_cast<std::uint32_t>(
                        expected_functions),
                    "radial function extent does not match R1 paths");
                require_execution_host_packet(
                    radial.intervals > 0 && radial.h > 0.0
                        && std::isfinite(r_cut) && r_cut > 0.0
                        && radial.coefficients != nullptr,
                    "radial spline storage is invalid");
                require_execution_host_packet(
                    node_types.extent(0) >= static_cast<std::size_t>(num_nodes)
                        && num_neigh.extent(0)
                            >= static_cast<std::size_t>(num_nodes)
                        && streamed_first_neigh.extent(0)
                            >= static_cast<std::size_t>(num_nodes)
                        && H1.extent(0) >= static_cast<std::size_t>(num_nodes)
                        && Phi1.extent(0)
                            >= static_cast<std::size_t>(num_nodes),
                    "node tensor extent is too small");
                require_execution_host_packet(
                    neigh_types.extent(0) >= static_cast<std::size_t>(num_edges)
                        && r.extent(0) >= static_cast<std::size_t>(num_edges),
                    "edge tensor extent is too small");
                require_execution_host_packet(
                    type_to_active.extent(0)
                        >= static_cast<std::size_t>(num_elements),
                    "type map extent is too small");
                require_execution_host_packet(
                    Y.extent(0) >= static_cast<std::size_t>(num_edges)*num_lm
                        && H1.extent(1)
                            >= static_cast<std::size_t>(num_LM)
                        && H1.extent(2)
                            >= static_cast<std::size_t>(channels)
                        && Phi1.extent(1)
                            >= static_cast<std::size_t>(num_lme)
                        && Phi1.extent(2)
                            >= static_cast<std::size_t>(channels),
                    "R1 tensor shape is inconsistent with the plugin contract");
                require_execution_host_packet(
                    num_nodes == 0
                        || (node_types.data() != nullptr
                            && num_neigh.data() != nullptr
                            && streamed_first_neigh.data() != nullptr
                            && type_to_active.data() != nullptr
                            && H1.data() != nullptr
                            && Phi1.data() != nullptr),
                    "required node tensor pointer is null");
                require_execution_host_packet(
                    num_edges == 0
                        || (neigh_indices.data() != nullptr
                            && neigh_types.data() != nullptr
                            && r.data() != nullptr
                            && Y.data() != nullptr),
                    "required edge tensor pointer is null");
                require_execution_host_packet(
                    node_types.span_is_contiguous()
                        && num_neigh.span_is_contiguous()
                        && streamed_first_neigh.span_is_contiguous()
                        && neigh_indices.span_is_contiguous()
                        && neigh_types.span_is_contiguous()
                        && type_to_active.span_is_contiguous()
                        && r.span_is_contiguous()
                        && Y.span_is_contiguous()
                        && H1.span_is_contiguous()
                        && Phi1.span_is_contiguous(),
                    "tensor storage is not contiguous");

                const SymmetrixJitHostR1ForwardArgsV2 args {
                    sizeof(SymmetrixJitHostR1ForwardArgsV2),
                    static_cast<std::uint32_t>(num_active_types),
                    num_nodes,
                    num_edges,
                    reinterpret_cast<const std::int32_t*>(node_types.data()),
                    reinterpret_cast<const std::int32_t*>(num_neigh.data()),
                    reinterpret_cast<const std::int32_t*>(
                        streamed_first_neigh.data()),
                    reinterpret_cast<const std::int32_t*>(neigh_indices.data()),
                    reinterpret_cast<const std::int32_t*>(neigh_types.data()),
                    reinterpret_cast<const std::int32_t*>(type_to_active.data()),
                    r.data(),
                    radial,
                    Y.data(),
                    H1.data(),
                    Phi1.data(),
                    r_cut};
                const auto owner =
                    jit_host_plugin->descriptor_v2().r1_forward_owner;
                constexpr int forward_channel_tile = 16;
                const int forward_channel_tiles =
                    (channels+forward_channel_tile-1)/forward_channel_tile;
                Kokkos::parallel_for(
                    "ExecutionHostPlugin::r1_forward",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                        factorized_execution_space, 0,
                        num_nodes*forward_channel_tiles),
                    [=] (const int flat) {
                        owner(
                            &args,
                            flat/forward_channel_tiles,
                            (flat%forward_channel_tiles)*forward_channel_tile);
                    });
                factorized_jit_launch_count += 1;
                factorized_jit_forward_launch_count += 1;
                return;
            }
        }
        throw std::runtime_error(
            "Execution R1 JIT forward selected without a loaded plugin.");
    }
    throw std::runtime_error(
        "Execution R1 JIT forward requires a loaded plugin.");
}

template <typename Precision>
void MACEKokkos<Precision>::reverse_Phi1(
    const int num_nodes,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r,
    bool zero_dxyz,
    bool zero_H1_adj)
{
    if (dPhi1r.extent(0) < Phi1r.extent(0))
        Kokkos::realloc(dPhi1r, Phi1r.extent(0), Phi1r.extent(1), Phi1r.extent(2));
    if (node_forces.size() != xyz.size())
        Kokkos::resize(node_forces, xyz.size());
    if (H1_adj.extent(0) < H1.extent(0))
        Kokkos::resize(H1_adj, H1.extent(0), H1.extent(1), H1.extent(2));
    if (zero_dxyz)
        Kokkos::deep_copy(node_forces, 0.0);
    Kokkos::deep_copy(dPhi1r, 0.0);
    if (zero_H1_adj)
        Kokkos::deep_copy(H1_adj, 0.0);

    const auto num_lm = this->num_lm;
    const auto num_channels = this->num_channels;
    const auto num_lelm1lm2 = this->num_lelm1lm2;
    const auto Phi1_lm1 = this->Phi1_lm1;
    const auto Phi1_lm2 = this->Phi1_lm2;
    const auto Phi1_lel1l2 = this->Phi1_lel1l2;
    const auto Phi1_lme = this->Phi1_lme;
    const auto Phi1_lelm1lm2 = this->Phi1_lelm1lm2;
    const auto Phi1_clebsch_gordan = this->Phi1_clebsch_gordan;
    const auto R1 = this->R1;
    const auto R1_deriv = this->R1_deriv;
    const auto Y = this->Y;
    const auto Y_grad = this->Y_grad;
    const auto H1 = this->H1;
    const auto H1_adj = this->H1_adj;
    const auto node_forces = this->node_forces;
    auto dPhi1r = this->dPhi1r;
    auto dPhi1 = this->dPhi1;

    // Compute dE/dPhi1 (named dPhi1)
    Kokkos::parallel_for("Reverse Phi1",
        Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const std::size_t node = static_cast<std::size_t>(
                team_member.league_rank());
            for (int p=0; p<Phi1_clebsch_gordan.size(); ++p) {
                const double C = Phi1_clebsch_gordan(p);
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team_member, num_channels),
                    [&] (const int k) {
                        dPhi1r(node,Phi1_lelm1lm2(p),k) += C * dPhi1(node,Phi1_lme(p),k);
                    });
            }
        });

    Kokkos::View<int*> first_neigh("first_neigh", num_nodes);
    Kokkos::parallel_scan("first_neigh",
        num_nodes,
        KOKKOS_LAMBDA (const int i, int& update, const bool final) {
            const int num_neigh_i = num_neigh(i);
            if (final)
                first_neigh(i) = update;
            update += num_neigh_i;
        });
    Kokkos::fence();

    Kokkos::parallel_for("Reverse Phi1r",
        Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const std::size_t node = static_cast<std::size_t>(
                team_member.league_rank());
            const int i0 = first_neigh(node);
            for (int j=0; j<num_neigh(node); ++j) {
                const int ij = i0 + j;
                const std::size_t edge = static_cast<std::size_t>(ij);
                const std::size_t source = static_cast<std::size_t>(
                    neigh_indices(ij));
                double f_x, f_y, f_z;
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team_member, num_lelm1lm2),
                    [=] (const int lelm1lm2, double& f_x, double& f_y, double& f_z) {
                        const int lm1 = Phi1_lm1(lelm1lm2);
                        const int lm2 = Phi1_lm2(lelm1lm2);
                        const int lel1l2 = Phi1_lel1l2(lelm1lm2);
                        double t1, t2;
                        Kokkos::parallel_reduce(
                            Kokkos::ThreadVectorRange(team_member, num_channels),
                            [=] (const int k, double& t1, double& t2) {
                                t1 += R1_deriv(ij,lel1l2*num_channels+k) * H1(source,lm2,k) * dPhi1r(node,lelm1lm2,k);
                                t2 += R1(ij,lel1l2*num_channels+k) * H1(source,lm2,k) * dPhi1r(node,lelm1lm2,k);
                                Kokkos::atomic_add(
                                    &H1_adj(source,lm2,k),
                                    R1(ij,lel1l2*num_channels+k) * Y(edge*num_lm+lm1) * dPhi1r(node,lelm1lm2,k));
                            }, t1, t2);
                        f_x += t1*xyz(3*edge)/r(ij)*Y(edge*num_lm+lm1) + t2*Y_grad(3*edge*num_lm+lm1);
                        f_y += t1*xyz(3*edge+1)/r(ij)*Y(edge*num_lm+lm1) + t2*Y_grad((3*edge+1)*num_lm+lm1);
                        f_z += t1*xyz(3*edge+2)/r(ij)*Y(edge*num_lm+lm1) + t2*Y_grad((3*edge+2)*num_lm+lm1);
                    }, f_x, f_y, f_z);
                team_member.team_barrier();
                Kokkos::single(Kokkos::PerTeam(team_member), [=]() {
                    node_forces(3*edge)   -= f_x;
                    node_forces(3*edge+1) -= f_y;
                    node_forces(3*edge+2) -= f_z;
                });
            }
        });
    Kokkos::fence();
}

template <typename Precision>
void MACEKokkos<Precision>::reverse_Phi1_streamed(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r,
    bool zero_dxyz,
    bool zero_H1_adj)
{
    if (dPhi1r.extent(0) < static_cast<std::size_t>(num_nodes))
        Kokkos::realloc(dPhi1r, num_nodes, num_lelm1lm2, num_channels);
    if (node_forces.size() != xyz.size())
        Kokkos::resize(node_forces, xyz.size());
    if (H1_adj.extent(0) < H1.extent(0))
        Kokkos::resize(H1_adj, H1.extent(0), H1.extent(1), H1.extent(2));
    if (zero_dxyz)
        Kokkos::deep_copy(node_forces, 0.0);
    Kokkos::deep_copy(dPhi1r, 0.0);
    if (zero_H1_adj)
        Kokkos::deep_copy(H1_adj, 0.0);

    const int num_channels = this->num_channels;
    const int num_lm = this->num_lm;
    const int source_harmonics = this->num_LM;
    const int num_paths = Phi1_l.extent(0);
    const int num_types = num_active_types;
    const auto Phi1_lme = this->Phi1_lme;
    const auto Phi1_lelm1lm2 = this->Phi1_lelm1lm2;
    const auto Phi1_clebsch_gordan = this->Phi1_clebsch_gordan;
    const auto dPhi1 = this->dPhi1;
    auto dPhi1r = this->dPhi1r;

    Kokkos::parallel_for(
        "MACEKokkos::reverse_Phi1_streamed",
        Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const std::size_t node = static_cast<std::size_t>(
                team_member.league_rank());
            for (int p=0; p<Phi1_clebsch_gordan.size(); ++p) {
                const Precision coefficient = Phi1_clebsch_gordan(p);
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team_member, num_channels),
                    [=] (const int k) {
                        dPhi1r(node,Phi1_lelm1lm2(p),k) += coefficient
                            *dPhi1(node,Phi1_lme(p),k);
                    });
            }
        });

    const auto type_to_active = this->type_to_active;
    const auto first_neigh = streamed_first_neigh;
    const auto Phi1_lm1 = this->Phi1_lm1;
    const auto Phi1_lm2 = this->Phi1_lm2;
    const auto path_row_offsets = this->Phi1_path_row_offsets;
    const auto radial_1 = this->radial_1;
    const auto edge_receivers = streamed_edge_receivers;
    const auto Y = this->Y;
    const auto Y_grad = this->Y_grad;
    const auto H1 = this->H1;
    auto H1_adj = this->H1_adj;
    Kokkos::View<double***,Kokkos::LayoutRight> source_adjoint;
    if constexpr (std::is_same_v<Precision,float>) {
        if (H1_adj_fp64_accumulator.extent(0) != H1_adj.extent(0)
            || H1_adj_fp64_accumulator.extent(1) != H1_adj.extent(1)
            || H1_adj_fp64_accumulator.extent(2) != H1_adj.extent(2)) {
            ++generic_phi1_source_adjoint_allocation_count;
            Kokkos::realloc(
                H1_adj_fp64_accumulator,
                H1_adj.extent(0), H1_adj.extent(1), H1_adj.extent(2));
        }
        source_adjoint = H1_adj_fp64_accumulator;
        const std::size_t entries = H1_adj.size();
        Kokkos::parallel_for(
            "MACEKokkos::initialize_Phi1_source_adjoint_fp64",
            Kokkos::RangePolicy<
                Kokkos::DefaultExecutionSpace,
                Kokkos::IndexType<std::size_t>>(0, entries),
            KOKKOS_LAMBDA (const std::size_t entry) {
                source_adjoint.data()[entry] = H1_adj.data()[entry];
            });
    } else {
        source_adjoint = H1_adj;
    }
    auto node_forces = this->node_forces;

    const auto finalize_source_adjoint = [&] () {
        if constexpr (std::is_same_v<Precision,float>) {
            const std::size_t entries = H1_adj.size();
            Kokkos::parallel_for(
                "MACEKokkos::finalize_Phi1_source_adjoint_fp32",
                Kokkos::RangePolicy<
                    Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(0, entries),
                KOKKOS_LAMBDA (const std::size_t entry) {
                    H1_adj.data()[entry] = static_cast<float>(
                        source_adjoint.data()[entry]);
                });
        }
    };

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    const int num_edges = neigh_indices.extent(0);
    using team_member_type = Kokkos::TeamPolicy<>::member_type;
    if (supports_fused_streamed_reverse()) {
        constexpr int edges_per_team = 8;
        Kokkos::parallel_for(
            "MACEKokkos::reverse_Phi1r_streamed_fused_atomics",
            Kokkos::TeamPolicy<>(
                (num_edges+edges_per_team-1)/edges_per_team,
                edges_per_team, 32),
            KOKKOS_LAMBDA (team_member_type team_member) {
                const int edge_begin = team_member.league_rank()*edges_per_team;
                const int edge_count = Kokkos::min(
                    edges_per_team, num_edges-edge_begin);
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team_member, edge_count),
                    [=] (const int edge_offset) {
                        const int ij = edge_begin+edge_offset;
                        const std::size_t edge = static_cast<std::size_t>(ij);
                        const std::size_t receiver = static_cast<std::size_t>(
                            edge_receivers(ij));
                        const std::size_t source = static_cast<std::size_t>(
                            neigh_indices(ij));
                        const int type_i = type_to_active(node_types(receiver));
                        const int type_j = type_to_active(neigh_types(ij));
                        const int edge_type = (type_i <= type_j)
                            ? type_i*(2*num_types-type_i-1)/2+type_j
                            : type_j*(2*num_types-type_j-1)/2+type_i;
                        const auto point = radial_1.evaluation_point(r(ij));
                        const Precision x_over_r = static_cast<Precision>(
                            xyz(3*edge)/r(ij));
                        const Precision y_over_r = static_cast<Precision>(
                            xyz(3*edge+1)/r(ij));
                        const Precision z_over_r = static_cast<Precision>(
                            xyz(3*edge+2)/r(ij));
                        Precision f_x, f_y, f_z;
                        Kokkos::parallel_reduce(
                            Kokkos::ThreadVectorRange(team_member, num_channels),
                            [=] (const int k,
                                 Precision& f_x,
                                 Precision& f_y,
                                 Precision& f_z) {
                                Precision h1_contributions[
                                    streamed_fused_max_num_LM] = {};
                                for (int path=0; path<num_paths; ++path) {
                                    Precision radial, radial_derivative;
                                    radial_1.evaluate_function(
                                        edge_type, point, path*num_channels+k,
                                        radial, radial_derivative);
                                    const int row_begin = path_row_offsets(path);
                                    const int row_end = path_row_offsets(path+1);
                                    for (int row=row_begin; row<row_end; ++row) {
                                        const int lm1 = Phi1_lm1(row);
                                        const int lm2 = Phi1_lm2(row);
                                        const Precision adjoint = dPhi1r(receiver,row,k);
                                        const Precision neighbor_feature =
                                            H1(source,lm2,k);
                                        const Precision radial_force =
                                            radial_derivative*neighbor_feature*adjoint;
                                        const Precision angular_force =
                                            radial*neighbor_feature*adjoint;
                                        f_x += radial_force*x_over_r
                                            *Y(edge*num_lm+lm1)
                                            +angular_force*Y_grad(3*edge*num_lm+lm1);
                                        f_y += radial_force*y_over_r
                                            *Y(edge*num_lm+lm1)
                                            +angular_force*Y_grad(
                                                (3*edge+1)*num_lm+lm1);
                                        f_z += radial_force*z_over_r
                                            *Y(edge*num_lm+lm1)
                                            +angular_force*Y_grad(
                                                (3*edge+2)*num_lm+lm1);
                                        h1_contributions[lm2] += radial
                                            *Y(edge*num_lm+lm1)*adjoint;
                                    }
                                }
                                for (int lm2=0; lm2<source_harmonics; ++lm2)
                                    Kokkos::atomic_add(
                                        &source_adjoint(source,lm2,k),
                                        static_cast<double>(h1_contributions[lm2]));
                            }, f_x, f_y, f_z);
                        Kokkos::single(
                            Kokkos::PerThread(team_member), [=]() {
                                node_forces(3*edge) -= f_x;
                                node_forces(3*edge+1) -= f_y;
                                node_forces(3*edge+2) -= f_z;
                            });
                    });
            });
        finalize_source_adjoint();
        Kokkos::fence();
        return;
    }
    if (num_edges <= streamed_edge_owned_limit) {
        const auto process_edge = KOKKOS_LAMBDA(
        const team_member_type& team_member,
        const int ij, const int i, const int type_i) {
        const std::size_t edge = static_cast<std::size_t>(ij);
        const std::size_t receiver = static_cast<std::size_t>(i);
        const std::size_t source = static_cast<std::size_t>(neigh_indices(ij));
        const int type_j = type_to_active(neigh_types(ij));
        const int edge_type = (type_i <= type_j)
            ? type_i*(2*num_types-type_i-1)/2+type_j
            : type_j*(2*num_types-type_j-1)/2+type_i;
        const auto point = radial_1.evaluation_point(r(ij));
        const double r_inv = 1.0/r(ij);
        double f_x, f_y, f_z;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, num_paths),
            [=] (const int path, double& f_x, double& f_y, double& f_z) {
                const int row_begin = path_row_offsets(path);
                const int row_end = path_row_offsets(path+1);
                double path_fx, path_fy, path_fz;
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(team_member, num_channels),
                    [=] (const int k,
                         double& path_fx,
                         double& path_fy,
                         double& path_fz) {
                        Precision radial, radial_derivative;
                        radial_1.evaluate_function(
                            edge_type, point, path*num_channels+k,
                            radial, radial_derivative);
                        for (int row=row_begin; row<row_end; ++row) {
                            const int lm1 = Phi1_lm1(row);
                            const int lm2 = Phi1_lm2(row);
                            const Precision adjoint = dPhi1r(receiver,row,k);
                            const Precision neighbor_feature = H1(source,lm2,k);
                            const Precision radial_force =
                                radial_derivative*neighbor_feature*adjoint;
                            const Precision angular_force =
                                radial*neighbor_feature*adjoint;
                            path_fx += radial_force*xyz(3*edge)*r_inv
                                *Y(edge*num_lm+lm1)
                                +angular_force*Y_grad(3*edge*num_lm+lm1);
                            path_fy += radial_force*xyz(3*edge+1)*r_inv
                                *Y(edge*num_lm+lm1)
                                +angular_force*Y_grad((3*edge+1)*num_lm+lm1);
                            path_fz += radial_force*xyz(3*edge+2)*r_inv
                                *Y(edge*num_lm+lm1)
                                +angular_force*Y_grad((3*edge+2)*num_lm+lm1);
                            Kokkos::atomic_add(
                                &source_adjoint(source,lm2,k),
                                static_cast<double>(
                                    radial*Y(edge*num_lm+lm1)*adjoint));
                        }
                    }, path_fx, path_fy, path_fz);
                f_x += path_fx;
                f_y += path_fy;
                f_z += path_fz;
            }, f_x, f_y, f_z);
        Kokkos::single(Kokkos::PerTeam(team_member), [=]() {
            node_forces(3*edge) -= f_x;
            node_forces(3*edge+1) -= f_y;
            node_forces(3*edge+2) -= f_z;
        });
    };

        constexpr int edges_per_team = 8;
        const int team_size = std::max(1, std::min(num_paths, 8));
        Kokkos::parallel_for(
            "MACEKokkos::reverse_Phi1r_streamed_edge_owned",
            Kokkos::TeamPolicy<>(
                (num_edges+edges_per_team-1)/edges_per_team,
                team_size, 32),
            KOKKOS_LAMBDA (team_member_type team_member) {
                const int edge_begin = team_member.league_rank()*edges_per_team;
                const int proposed_end = edge_begin+edges_per_team;
                const int edge_end = proposed_end < num_edges
                    ? proposed_end : num_edges;
                for (int ij=edge_begin; ij<edge_end; ++ij) {
                    const int i = edge_receivers(ij);
                    const int type_i = type_to_active(node_types(i));
                    process_edge(team_member, ij, i, type_i);
                }
            });
        finalize_source_adjoint();
        Kokkos::fence();
        return;
    }
#endif

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    const int team_size = std::max(1, std::min(num_paths, 8));
    const Kokkos::TeamPolicy<> streamed_policy(
        num_nodes, team_size, 32);
#else
    const Kokkos::TeamPolicy<> streamed_policy(
        num_nodes, Kokkos::AUTO, 32);
#endif
    Kokkos::parallel_for(
        "MACEKokkos::reverse_Phi1r_streamed",
        streamed_policy,
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const std::size_t node = static_cast<std::size_t>(
                team_member.league_rank());
            const int type_i = type_to_active(node_types(node));
            const int i0 = first_neigh(node);
            for (int j=0; j<num_neigh(node); ++j) {
                const int ij = i0+j;
                const std::size_t edge = static_cast<std::size_t>(ij);
                const std::size_t source = static_cast<std::size_t>(
                    neigh_indices(ij));
                const int type_j = type_to_active(neigh_types(ij));
                const int edge_type = (type_i <= type_j)
                    ? type_i*(2*num_types-type_i-1)/2+type_j
                    : type_j*(2*num_types-type_j-1)/2+type_i;
                const auto point = radial_1.evaluation_point(r(ij));
                const double r_inv = 1.0/r(ij);
                double f_x, f_y, f_z;
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team_member, num_paths),
                    [=] (const int path, double& f_x, double& f_y, double& f_z) {
                        const int row_begin = path_row_offsets(path);
                        const int row_end = path_row_offsets(path+1);
                        double path_fx, path_fy, path_fz;
                        Kokkos::parallel_reduce(
                            Kokkos::ThreadVectorRange(team_member, num_channels),
                            [=] (const int k,
                                 double& path_fx,
                                 double& path_fy,
                                 double& path_fz) {
                                Precision radial, radial_derivative;
                                radial_1.evaluate_function(
                                    edge_type, point, path*num_channels+k,
                                    radial, radial_derivative);
                                for (int row=row_begin; row<row_end; ++row) {
                                    const int lm1 = Phi1_lm1(row);
                                    const int lm2 = Phi1_lm2(row);
                                    const Precision adjoint = dPhi1r(node,row,k);
                                    const Precision neighbor_feature = H1(source,lm2,k);
                                    const Precision radial_force =
                                        radial_derivative*neighbor_feature*adjoint;
                                    const Precision angular_force =
                                        radial*neighbor_feature*adjoint;
                                    path_fx += radial_force*xyz(3*edge)*r_inv
                                        *Y(edge*num_lm+lm1)
                                        +angular_force*Y_grad(3*edge*num_lm+lm1);
                                    path_fy += radial_force*xyz(3*edge+1)*r_inv
                                        *Y(edge*num_lm+lm1)
                                        +angular_force*Y_grad((3*edge+1)*num_lm+lm1);
                                    path_fz += radial_force*xyz(3*edge+2)*r_inv
                                        *Y(edge*num_lm+lm1)
                                        +angular_force*Y_grad((3*edge+2)*num_lm+lm1);
                                    Kokkos::atomic_add(
                                        &source_adjoint(source,lm2,k),
                                        static_cast<double>(
                                            radial*Y(edge*num_lm+lm1)*adjoint));
                                }
                            }, path_fx, path_fy, path_fz);
                        f_x += path_fx;
                        f_y += path_fy;
                        f_z += path_fz;
                    }, f_x, f_y, f_z);
                Kokkos::single(Kokkos::PerTeam(team_member), [=]() {
                    node_forces(3*edge) -= f_x;
                    node_forces(3*edge+1) -= f_y;
                    node_forces(3*edge+2) -= f_z;
                });
            }
        });
    finalize_source_adjoint();
    Kokkos::fence();
}


template class MACEKokkos<float>;
template class MACEKokkos<double>;
