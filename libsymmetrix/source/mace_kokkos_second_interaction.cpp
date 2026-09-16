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
#include "factorized_blas.hpp"
#include "host_dense_kernels.hpp"
#include "standard_m1.hpp"
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

namespace {

constexpr int host_h2_blas_max_channels = 1024;

}

template <typename Precision>
void MACEKokkos<Precision>::compute_A1(
    int num_nodes,
    const bool completion_fence)
{
    // The core matrix multiplication is:
    //         [A1_il]_mk = \sum_(ek') [Phi1_il]_m(ek') [W_il]_(ek')k
    ensure_mh0_a1_forward_capacity(num_nodes);

    const auto l_max = this->l_max;
    const auto num_channels = this->num_channels;
    const auto Phi1_l = this->Phi1_l;
    const auto Phi1 = this->Phi1;
    auto A1_weights = this->A1_weights;
    auto A1 = this->A1;

    // These small fixed-l host contractions need no packing before BLAS.
    if constexpr (std::is_same_v<
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>) {
        if (symmetrix::host_worker_cblas_enabled()
                && !symmetrix::host_dense_backend_overrides_cblas()) {
            Kokkos::parallel_for(
                "Compute A1 host BLAS",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(
                    factorized_execution_space, 0,
                    static_cast<std::size_t>(num_nodes)*(l_max+1)),
                [=] (const std::size_t owner) {
                    const std::size_t node = owner/(l_max+1);
                    const int l = owner%(l_max+1);
                    int lme = 0;
                    int num_eta = 0;
                    for (int p=0; p<Phi1_l.size(); ++p) {
                        const int ll = Phi1_l(p);
                        if (ll < l)
                            lme += 2*ll+1;
                        if (ll == l)
                            ++num_eta;
                    }
                    const int component_count = 2*l+1;
                    const int input_count = num_eta*num_channels;
                    const auto weights = A1_weights(l);
                    symmetrix_blas_gemm<Precision>(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        component_count, num_channels, input_count,
                        Precision(1), &Phi1(node,lme,0), input_count,
                        weights.data(), num_channels,
                        Precision(0), &A1(node,l*l,0), num_channels);
                });
            if (completion_fence)
                factorized_execution_space.fence("Compute A1 completion");
            return;
        }
        if (symmetrix::host_flat_dense_enabled()) {
            Kokkos::parallel_for(
                "Compute A1 flat",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(
                    factorized_execution_space, 0,
                    static_cast<std::size_t>(num_nodes)*(l_max+1)),
                [=] (const std::size_t owner) {
                    const std::size_t node = owner/(l_max+1);
                    const int l = owner%(l_max+1);
                    int lme = 0;
                    int num_eta = 0;
                    for (int p=0; p<Phi1_l.size(); ++p) {
                        const int ll = Phi1_l(p);
                        if (ll < l)
                            lme += 2*ll+1;
                        if (ll == l)
                            ++num_eta;
                    }
                    const auto weights = A1_weights(l);
                    symmetrix::host_dense_gemm_nn(
                        &Phi1(node,lme,0), weights.data(),
                        &A1(node,l*l,0), 2*l+1,
                        num_eta*num_channels, num_channels);
                });
            if (completion_fence)
                factorized_execution_space.fence("Compute A1 completion");
            return;
        }
    }

    Kokkos::parallel_for("Compute A1",
        Kokkos::TeamPolicy<>(
            factorized_execution_space,
            num_nodes*(l_max+1), Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const std::size_t node = static_cast<std::size_t>(
                team_member.league_rank()/(l_max+1));
            const int l = team_member.league_rank() % (l_max+1);
            int lme = 0;
            int num_eta = 0;
            for (int p=0; p<Phi1_l.size(); ++p) {
                const int ll = Phi1_l(p);
                if (ll < l)
                    lme += 2*ll+1;
                if (ll == l)
                    num_eta += 1;
            }
            auto Phi1_il = Kokkos::View<Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>(
                &Phi1(node,lme,0), 2*l+1, num_eta*num_channels);
            auto A1_il = Kokkos::subview(A1, node, Kokkos::make_pair(l*l,l*(l+2)+1), Kokkos::ALL);
            KokkosBatched::TeamGemm<Kokkos::TeamPolicy<>::member_type,
                                    KokkosBatched::Trans::NoTranspose,
                                    KokkosBatched::Trans::NoTranspose,
                                    KokkosBatched::Algo::Gemm::Blocked>
                ::invoke(team_member, 1.0, Phi1_il, A1_weights(l), 0.0, A1_il);
        });
    if (completion_fence)
        factorized_execution_space.fence("Compute A1 completion");
}

template <typename Precision>
void MACEKokkos<Precision>::compute_A1_channel_tile(
    const int num_nodes,
    const int channel_begin)
{
    const int channel_count = std::min(
        phi1_channel_tile_size, num_channels-channel_begin);
    if (channel_begin < 0 || channel_count <= 0
        || channel_begin%phi1_channel_tile_size != 0
        || Phi1.extent_int(0) < num_nodes
        || Phi1.extent_int(1) != num_lme
        || Phi1.extent_int(2) != channel_count)
        throw std::logic_error("Invalid forward Phi1 channel tile.");
    ensure_mh0_a1_forward_capacity(num_nodes);

    const auto l_max = this->l_max;
    const auto Phi1_l = this->Phi1_l;
    const auto Phi1 = this->Phi1;
    const auto tile_weights = this->A1_channel_tile_weights;
    auto A1 = this->A1;
    const int phase = channel_begin/phi1_channel_tile_size;
    const Precision beta = channel_begin == 0 ? Precision(0) : Precision(1);
    Kokkos::parallel_for(
        "Compute A1 channel tile",
        Kokkos::TeamPolicy<>(
            factorized_execution_space,
            num_nodes*(l_max+1), Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const std::size_t node = static_cast<std::size_t>(
                team_member.league_rank()/(l_max+1));
            const int l = team_member.league_rank()%(l_max+1);
            int lme = 0;
            int num_eta = 0;
            for (int p=0; p<Phi1_l.size(); ++p) {
                const int ll = Phi1_l(p);
                if (ll < l)
                    lme += 2*ll+1;
                if (ll == l)
                    ++num_eta;
            }
            auto Phi1_il = Kokkos::View<
                Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>(
                    &Phi1(node,lme,0), 2*l+1, num_eta*channel_count);
            auto A1_il = Kokkos::subview(
                A1, node,
                Kokkos::make_pair(l*l,l*l+2*l+1), Kokkos::ALL);
            const auto weights = tile_weights(phase*(l_max+1)+l);
            KokkosBatched::TeamGemm<
                Kokkos::TeamPolicy<>::member_type,
                KokkosBatched::Trans::NoTranspose,
                KokkosBatched::Trans::NoTranspose,
                KokkosBatched::Algo::Gemm::Blocked>::invoke(
                    team_member, Precision(1), Phi1_il, weights, beta, A1_il);
        });
}

template <typename Precision>
void MACEKokkos<Precision>::reverse_A1(
    int num_nodes,
    const bool completion_fence)
{
    reverse_A1_from(num_nodes, A1_adj, completion_fence);
}

template <typename Precision>
void MACEKokkos<Precision>::reverse_A1_channel_tile(
    const int num_nodes,
    const int channel_begin)
{
    const int channel_count = std::min(
        phi1_channel_tile_size, num_channels-channel_begin);
    if (channel_begin < 0 || channel_count <= 0
        || channel_begin%phi1_channel_tile_size != 0
        || Phi1.extent_int(0) < num_nodes
        || Phi1.extent_int(1) != num_lme
        || Phi1.extent_int(2) != channel_count)
        throw std::logic_error("Invalid reverse Phi1 channel tile.");

    const auto l_max = this->l_max;
    const auto Phi1_l = this->Phi1_l;
    auto Phi1 = this->Phi1;
    const auto tile_weights_trans = this->A1_channel_tile_weights_trans;
    const auto A1_adj = this->A1_adj;
    const int phase = channel_begin/phi1_channel_tile_size;
    Kokkos::parallel_for(
        "Reverse A1 channel tile",
        Kokkos::TeamPolicy<>(
            factorized_execution_space,
            num_nodes*(l_max+1), Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const std::size_t node = static_cast<std::size_t>(
                team_member.league_rank()/(l_max+1));
            const int l = team_member.league_rank()%(l_max+1);
            int lme = 0;
            int num_eta = 0;
            for (int p=0; p<Phi1_l.size(); ++p) {
                const int ll = Phi1_l(p);
                if (ll < l)
                    lme += 2*ll+1;
                if (ll == l)
                    ++num_eta;
            }
            auto dA1_il = Kokkos::subview(
                A1_adj, node,
                Kokkos::make_pair(l*l,l*l+2*l+1), Kokkos::ALL);
            auto dPhi1_il = Kokkos::View<
                Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>(
                    &Phi1(node,lme,0), 2*l+1, num_eta*channel_count);
            const auto weights =
                tile_weights_trans(phase*(l_max+1)+l);
            KokkosBatched::TeamGemm<
                Kokkos::TeamPolicy<>::member_type,
                KokkosBatched::Trans::NoTranspose,
                KokkosBatched::Trans::NoTranspose,
                KokkosBatched::Algo::Gemm::Blocked>::invoke(
                    team_member, Precision(1), dA1_il, weights,
                    Precision(0), dPhi1_il);
        });
}

template <typename Precision>
void MACEKokkos<Precision>::reverse_A1_from(
    int num_nodes,
    Kokkos::View<const Precision***,Kokkos::LayoutRight> output_adjoint,
    const bool completion_fence)
{
    // The core matrix multiplication is:
    //         [dE/dPhi1_il]_m(ek) = \sum_k' [dE/dA1_il]_mk' [trans(W_il)]_k'(ek)
    if (use_mh0_adjoint_reuse())
        dPhi1 = Phi1;
    else if (dPhi1.data() != nullptr && dPhi1.data() == Phi1.data())
        dPhi1 = decltype(dPhi1)();
    if (dPhi1.extent(0) < static_cast<std::size_t>(num_nodes)
        || dPhi1.extent_int(1) != num_lme
        || dPhi1.extent_int(2) != num_channels)
        Kokkos::realloc(dPhi1, num_nodes, num_lme, num_channels);

    const auto l_max = this->l_max;
    const auto num_channels = this->num_channels;
    const auto Phi1_l = this->Phi1_l;
    const auto A1_weights_trans = this->A1_weights_trans;
    auto dPhi1 = this->dPhi1;

    // Keep device execution on the backend-portable batched team kernel.
    if constexpr (std::is_same_v<
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>) {
        if (symmetrix::host_worker_cblas_enabled()
                && !symmetrix::host_dense_backend_overrides_cblas()) {
            Kokkos::parallel_for(
                "Reverse A1 host BLAS",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(
                    factorized_execution_space, 0,
                    static_cast<std::size_t>(num_nodes)*(l_max+1)),
                [=] (const std::size_t owner) {
                    const std::size_t node = owner/(l_max+1);
                    const int l = owner%(l_max+1);
                    int lme = 0;
                    int num_eta = 0;
                    for (int p=0; p<Phi1_l.size(); ++p) {
                        const int ll = Phi1_l(p);
                        if (ll < l)
                            lme += 2*ll+1;
                        if (ll == l)
                            ++num_eta;
                    }
                    const int component_count = 2*l+1;
                    const int output_count = num_eta*num_channels;
                    const auto weights = A1_weights_trans(l);
                    symmetrix_blas_gemm<Precision>(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        component_count, output_count, num_channels,
                        Precision(1), &output_adjoint(node,l*l,0), num_channels,
                        weights.data(), output_count,
                        Precision(0), &dPhi1(node,lme,0), output_count);
                });
            if (completion_fence)
                factorized_execution_space.fence("Reverse A1 completion");
            return;
        }
        if (symmetrix::host_flat_dense_enabled()) {
            Kokkos::parallel_for(
                "Reverse A1 flat",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(
                    factorized_execution_space, 0,
                    static_cast<std::size_t>(num_nodes)*(l_max+1)),
                [=] (const std::size_t owner) {
                    const std::size_t node = owner/(l_max+1);
                    const int l = owner%(l_max+1);
                    int lme = 0;
                    int num_eta = 0;
                    for (int p=0; p<Phi1_l.size(); ++p) {
                        const int ll = Phi1_l(p);
                        if (ll < l)
                            lme += 2*ll+1;
                        if (ll == l)
                            ++num_eta;
                    }
                    const auto weights = A1_weights_trans(l);
                    symmetrix::host_dense_gemm_nn(
                        &output_adjoint(node,l*l,0), weights.data(),
                        &dPhi1(node,lme,0), 2*l+1,
                        num_channels, num_eta*num_channels);
                });
            if (completion_fence)
                factorized_execution_space.fence("Reverse A1 completion");
            return;
        }
    }

    Kokkos::parallel_for("Reverse A1",
        Kokkos::TeamPolicy<>(
            factorized_execution_space,
            num_nodes*(l_max+1), Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const std::size_t node = static_cast<std::size_t>(
                team_member.league_rank()/(l_max+1));
            const int l = team_member.league_rank() % (l_max+1);
            int lme = 0;
            int num_eta = 0;
            for (int p=0; p<Phi1_l.size(); ++p) {
                const int ll = Phi1_l(p);
                if (ll < l)
                    lme += 2*ll+1;
                if (ll == l)
                    num_eta += 1;
            }
            auto dA1_il = Kokkos::subview(
                output_adjoint, node,
                Kokkos::make_pair(l*l,l*l+2*l+1), Kokkos::ALL);
            auto dPhi1_il = Kokkos::View<Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>(
                &dPhi1(node,lme,0), 2*l+1, num_eta*num_channels);
            KokkosBatched::TeamGemm<Kokkos::TeamPolicy<>::member_type,
                                    KokkosBatched::Trans::NoTranspose,
                                    KokkosBatched::Trans::NoTranspose,
                                    KokkosBatched::Algo::Gemm::Blocked>
                ::invoke(team_member, 1.0, dA1_il, A1_weights_trans(l), 0.0, dPhi1_il);
        });
    if (completion_fence)
        factorized_execution_space.fence("Reverse A1 completion");
}

template <typename Precision>
void MACEKokkos<Precision>::compute_A1_scaled(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r)
{
    if (not A1_scaled) return;
    const bool async_inference = use_factorized_async_inference();
    const auto execution_space = async_inference
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    const bool recompute_splines = async_inference && use_mh0_adjoint_reuse();

    // compute A1 splines
    if (recompute_splines) {
        A1_spline_values = decltype(A1_spline_values)();
        A1_spline_derivs = decltype(A1_spline_derivs)();
    } else if (A1_spline_values.extent(0) < r.size()) {
        Kokkos::realloc(A1_spline_values, r.size(), 1);
        Kokkos::realloc(A1_spline_derivs, r.size(), 1);
    }
    if (async_inference && !recompute_splines) {
        const auto splines = A1_splines;
        const auto edge_receivers = execution_edge_receivers;
        const auto type_map = type_to_active;
        const int active_type_count = num_active_types;
        const auto values = A1_spline_values;
        const auto derivatives = A1_spline_derivs;
        Kokkos::parallel_for(
            "MACEKokkos::compute_A1_scale_splines",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                execution_space, 0, r.extent_int(0)),
            KOKKOS_LAMBDA (const int edge) {
                const int type_i = type_map(node_types(edge_receivers(edge)));
                const int type_j = type_map(neigh_types(edge));
                const int edge_type = type_i <= type_j
                    ? type_i*(2*active_type_count-type_i-1)/2+type_j
                    : type_j*(2*active_type_count-type_j-1)/2+type_i;
                splines.evaluate_function(
                    edge_type, r(edge), 0,
                    values(edge,0), derivatives(edge,0));
            });
    } else if (!async_inference)
        A1_splines.evaluate(
            num_nodes, node_types, num_neigh, neigh_types,
            type_to_active, num_active_types, r,
            A1_spline_values, A1_spline_derivs);

    // compute first_neigh
    Kokkos::View<int*> first_neigh;
    if (async_inference)
        first_neigh = streamed_first_neigh;
    else {
        first_neigh = Kokkos::View<int*>("first_neigh", num_nodes);
        Kokkos::parallel_scan("Compute first_neigh",
            num_nodes,
            KOKKOS_LAMBDA (const int i, int& update, const bool final) {
                if (final)
                    first_neigh(i) = update;
                update += num_neigh(i);
            });
    }

    // perform the scaling
    auto A1 = this->A1;
    auto A1_spline_values = this->A1_spline_values;
    auto A1_spline_derivs = this->A1_spline_derivs;
    const auto splines = A1_splines;
    const auto type_map = type_to_active;
    const int active_type_count = num_active_types;
    const auto num_channels = this->num_channels;
    const auto num_lm = this->num_lm;
    Kokkos::parallel_for(
        "MACEKokkos::compute_A1_scaled",
        Kokkos::TeamPolicy<>(
            execution_space, num_nodes, Kokkos::AUTO, Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const std::size_t node = static_cast<std::size_t>(
                team_member.league_rank());
            const int i0 = first_neigh(node);
            const int type_i = recompute_splines
                ? type_map(node_types(node)) : 0;
            // compute scale factor
            double A1_scale_factor;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team_member, num_neigh(node)),
                [=] (const int j, double& lsum) {
                    const int edge = i0+j;
                    if (recompute_splines) {
                        const int type_j = type_map(neigh_types(edge));
                        const int edge_type = type_i <= type_j
                            ? type_i*(2*active_type_count-type_i-1)/2+type_j
                            : type_j*(2*active_type_count-type_j-1)/2+type_i;
                        lsum += splines.evaluate_function(edge_type, r(edge), 0);
                    } else
                        lsum += A1_spline_values(edge,0);
                }, A1_scale_factor);
            A1_scale_factor += 1.0;
            team_member.team_barrier();
            // perform the scaling
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, num_lm*num_channels),
                [=] (int lmk) {
                    A1(node,lmk/num_channels,lmk%num_channels) /= A1_scale_factor;
                });
        });
    complete_device_stage("MACEKokkos::compute_A1_scaled");
}

template <typename Precision>
void MACEKokkos<Precision>::reverse_A1_scaled(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r)
{
    if (not A1_scaled) return;
    const bool async_inference = use_factorized_async_inference();
    const auto execution_space = async_inference
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();

    Kokkos::View<int*> first_neigh;
    if (async_inference)
        first_neigh = streamed_first_neigh;
    else {
        first_neigh = Kokkos::View<int*>("first_neigh", num_nodes);
        Kokkos::parallel_scan("Compute first_neigh",
            num_nodes,
            KOKKOS_LAMBDA (const int i, int& update, const bool final) {
                if (final)
                    first_neigh(i) = update;
                update += num_neigh(i);
            });
    }

    // update the derivatives
    // Warning: Assumes node_forces have been initialized elsewhere
    const auto A1 = this->A1;
    const auto A1_adj = this->A1_adj;
    const auto A1_spline_values = this->A1_spline_values;
    const auto A1_spline_derivs = this->A1_spline_derivs;
    const bool reuse_adjoint = use_mh0_adjoint_reuse();
    const bool recompute_splines = async_inference && reuse_adjoint;
    const auto splines = A1_splines;
    const auto type_map = type_to_active;
    const int active_type_count = num_active_types;
    const auto scale_adjoint = mh0_a1_scale_adjoint;
    const auto num_channels = this->num_channels;
    const auto num_lm = this->num_lm;
    const bool compact_geometry = use_compact_edge_geometry();
    const auto unit_direction = execution_prepared_unit_direction;
    auto node_forces = this->node_forces;
    Kokkos::parallel_for(
        "MACEKokkos::reverse_A1_scaled",
        Kokkos::TeamPolicy<>(
            execution_space, num_nodes, Kokkos::AUTO, Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const std::size_t node = static_cast<std::size_t>(
                team_member.league_rank());
            const int i0 = first_neigh(node);
            const int type_i = recompute_splines
                ? type_map(node_types(node)) : 0;
            // scale factor
            double A1_scale_factor;// = 1.0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team_member, num_neigh(node)),
                [=] (const int j, double& lsum) {
                    const int edge = i0+j;
                    if (recompute_splines) {
                        const int type_j = type_map(neigh_types(edge));
                        const int edge_type = type_i <= type_j
                            ? type_i*(2*active_type_count-type_i-1)/2+type_j
                            : type_j*(2*active_type_count-type_j-1)/2+type_i;
                        lsum += splines.evaluate_function(edge_type, r(edge), 0);
                    } else
                        lsum += A1_spline_values(edge,0);
                }, A1_scale_factor);
            A1_scale_factor += 1.0;
            team_member.team_barrier();
            // update dE/dxyz
            double dA1_dot_A1 = 0.0;
            if (reuse_adjoint)
                dA1_dot_A1 = scale_adjoint(node);
            else
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team_member, num_lm*num_channels),
                    [=] (const int lmk, double& lsum) {
                        const int lm = lmk / num_channels;
                        const int k = lmk % num_channels;
                        lsum += A1_adj(node,lm,k) * A1(node,lm,k);
                    }, dA1_dot_A1);
            team_member.team_barrier();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, num_neigh(node)),
                [=] (const int j) {
                    const int ij = first_neigh(node) + j;
                    const std::size_t edge = static_cast<std::size_t>(ij);
                    double f;
                    double d;
                    if (recompute_splines) {
                        const int type_i = type_map(node_types(node));
                        const int type_j = type_map(neigh_types(ij));
                        const int edge_type = type_i <= type_j
                            ? type_i*(2*active_type_count-type_i-1)/2+type_j
                            : type_j*(2*active_type_count-type_j-1)/2+type_i;
                        splines.evaluate_function(edge_type, r(ij), 0, f, d);
                    } else {
                        f = A1_spline_values(ij,0);
                        d = A1_spline_derivs(ij,0);
                    }
                    const double scale = dA1_dot_A1/A1_scale_factor*d;
                    node_forces(3*edge+0) += scale*(compact_geometry
                        ? unit_direction(3*edge+0) : xyz(3*edge+0)/r(ij));
                    node_forces(3*edge+1) += scale*(compact_geometry
                        ? unit_direction(3*edge+1) : xyz(3*edge+1)/r(ij));
                    node_forces(3*edge+2) += scale*(compact_geometry
                        ? unit_direction(3*edge+2) : xyz(3*edge+2)/r(ij));
                });
            // update dE/dA1
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, num_lm*num_channels),
                [=] (int lmk) {
                    A1_adj(node,lmk/num_channels,lmk%num_channels) /= A1_scale_factor;
                });
        });
    complete_device_stage("MACEKokkos::reverse_A1_scaled");
}

template <typename Precision>
void MACEKokkos<Precision>::compute_M1(int num_nodes, Kokkos::View<const int*> node_types)
{
    const auto execution_space = use_factorized_async_inference()
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    ensure_mh0_m1_forward_capacity(num_nodes);
    if (m1_polynomial_policy == M1PolynomialPolicy::recompute) {
        release_m1_polynomial_workspace();
        if (!single_layer_readout && standard_m1_module_ready
            && symmetrix::standard_m1::launch_forward(
                execution_space, num_nodes, num_channels, node_types,
                A1, M1_weights, M1)) {
            standard_m1_module_forward_launch_count += 1;
            m1_recompute_forward_launch_count += 1;
            complete_device_stage("MACEKokkos::compute_M1_recompute");
            return;
        }
        const auto polynomial_input = single_layer_readout ? H1 : A1;
        const auto M1_poly_spec = this->M1_poly_spec;
        const auto M1_poly_coeff = this->M1_poly_coeff;
        const int channels = num_channels;
        const int harmonics = single_layer_readout ? num_LM : num_lm;
        const int polynomial_nodes = static_cast<int>(M1_poly_coeff.extent(1));
        const int multiplication_nodes = static_cast<int>(M1_poly_spec.extent(0));
        const int tile_channels = m1_recompute_tile_channels;
        const int vector_length = m1_recompute_vector_length();
        const int scratch_level = m1_recompute_scratch_level();
        auto M1 = this->M1;
        using TeamMember = typename Kokkos::TeamPolicy<>::member_type;
        using ScratchView = Kokkos::View<
            Precision**, Kokkos::LayoutRight,
            typename TeamMember::scratch_memory_space,
            Kokkos::MemoryUnmanaged>;
        auto policy = vector_length == 1
            ? Kokkos::TeamPolicy<>(
                execution_space, num_nodes, Kokkos::AUTO, 1)
            : Kokkos::TeamPolicy<>(
                execution_space, num_nodes, 1, vector_length);
        policy.set_scratch_size(
            scratch_level, Kokkos::PerTeam(
                sizeof(Precision)*polynomial_nodes*tile_channels));
        Kokkos::parallel_for(
            "Compute M1 recompute", policy,
            KOKKOS_LAMBDA (const TeamMember& member) {
                const std::size_t node =
                    static_cast<std::size_t>(member.league_rank());
                ScratchView values(
                    member.team_scratch(scratch_level),
                    polynomial_nodes, tile_channels);
                for (int channel_begin=0; channel_begin<channels;
                     channel_begin += tile_channels) {
                    const int active_channels =
                        Kokkos::min(tile_channels, channels-channel_begin);
                    Kokkos::parallel_for(
                        Kokkos::TeamVectorRange(member, active_channels),
                        [=] (const int lane) {
                            const int channel = channel_begin+lane;
                            Precision output = 0;
                            for (int p=0; p<harmonics; ++p) {
                                values(p,lane) = polynomial_input(node,p,channel);
                                output += M1_poly_coeff(
                                    node_types(node),p,channel)*values(p,lane);
                            }
                            for (int p=0; p<multiplication_nodes; ++p) {
                                const int output_node = harmonics+p;
                                values(output_node,lane) =
                                    values(M1_poly_spec(p,0),lane)
                                    *values(M1_poly_spec(p,1),lane);
                                output += M1_poly_coeff(
                                    node_types(node),output_node,channel)
                                    *values(output_node,lane);
                            }
                            M1(node,channel) = output;
                        });
                    member.team_barrier();
                }
            });
        m1_recompute_forward_launch_count += 1;
        complete_device_stage("MACEKokkos::compute_M1_recompute");
        return;
    }
    if (M1_poly_values.extent(0) < num_nodes)
        Kokkos::realloc(
            Kokkos::WithoutInitializing, M1_poly_values,
            num_nodes, (single_layer_readout ? num_LM : num_lm)
                +M1_poly_spec.extent(0), num_channels);
    if (use_factorized_async_inference())
        Kokkos::deep_copy(execution_space, M1, Precision(0));
    else
        Kokkos::deep_copy(M1, Precision(0));

    const auto polynomial_input = single_layer_readout ? H1 : A1;
    const auto M1_monomials = this->M1_monomials;
    const auto M1_weights = this->M1_weights;
    const auto M1_poly_spec = this->M1_poly_spec;
    const auto M1_poly_coeff = this->M1_poly_coeff;
    const auto M1_poly_values = this->M1_poly_values;
    const auto num_channels = this->num_channels;
    const int harmonics = single_layer_readout ? num_LM : num_lm;
    auto M1 = this->M1;

    Kokkos::parallel_for("Compute M1",
        Kokkos::TeamPolicy<>(execution_space, num_nodes, Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            const std::size_t node = static_cast<std::size_t>(i);
            // initialize
            Kokkos::parallel_for(
                Kokkos::TeamVectorMDRange<
                    Kokkos::Rank<2,Kokkos::Iterate::Right>,Kokkos::TeamPolicy<>::member_type>(
                        team_member, harmonics, num_channels),
                [&] (const int p, const int k) {
                    M1_poly_values(node,p,k) = polynomial_input(node,p,k);
                    Kokkos::atomic_add(
                        &M1(node,k), M1_poly_coeff(node_types(node),p,k)
                            *M1_poly_values(node,p,k));
                });
            team_member.team_barrier();
            // forward pass
            for (int p=0; p<M1_poly_spec.extent(0); ++p) {
                const int p0 = M1_poly_spec(p,0);
                const int p1 = M1_poly_spec(p,1);
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team_member, num_channels),
                    [&] (const int k) {
                        M1_poly_values(node,harmonics+p,k) =
                            M1_poly_values(node,p0,k)
                            *M1_poly_values(node,p1,k);
                        M1(node,k) += M1_poly_coeff(
                            node_types(node),harmonics+p,k)
                            *M1_poly_values(node,harmonics+p,k);
                    });
            }
        });
    complete_device_stage("MACEKokkos::compute_M1");
}
template <typename Precision>
void MACEKokkos<Precision>::reverse_M1(int num_nodes, Kokkos::View<const int*> node_types)
{
    const auto execution_space = use_factorized_async_inference()
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    const bool reuse_adjoint = use_mh0_adjoint_reuse() && !single_layer_readout;
    if (reuse_adjoint) {
        A1_adj = A1;
        if (A1_scaled && mh0_a1_scale_adjoint.extent(0)
                != static_cast<std::size_t>(num_nodes))
            Kokkos::realloc(mh0_a1_scale_adjoint, num_nodes);
    }
    auto polynomial_adjoint = single_layer_readout ? H1_adj : A1_adj;
    const auto polynomial_input = single_layer_readout ? H1 : A1;
    const int harmonics = single_layer_readout ? num_LM : num_lm;
    const bool accumulate_h1_adjoint = single_layer_readout;
    if (polynomial_adjoint.extent(0) < static_cast<std::size_t>(num_nodes)) {
        if (single_layer_readout)
            Kokkos::realloc(H1_adj, H1.extent(0), H1.extent(1), H1.extent(2));
        else
            Kokkos::realloc(A1_adj, A1.extent(0), A1.extent(1), A1.extent(2));
        polynomial_adjoint = single_layer_readout ? H1_adj : A1_adj;
    }
    if (m1_polynomial_policy == M1PolynomialPolicy::recompute) {
        release_m1_polynomial_workspace();
        const bool capture_scale_adjoint = reuse_adjoint && A1_scaled;
        if (!single_layer_readout && standard_m1_module_ready
            && symmetrix::standard_m1::launch_reverse(
                execution_space, num_nodes, num_channels, node_types,
                A1, M1_weights, M1_adj, A1_adj,
                mh0_a1_scale_adjoint, capture_scale_adjoint)) {
            standard_m1_module_reverse_launch_count += 1;
            m1_recompute_reverse_launch_count += 1;
            complete_device_stage("MACEKokkos::reverse_M1_recompute");
            return;
        }
        const auto M1_adj = this->M1_adj;
        const auto M1_poly_spec = this->M1_poly_spec;
        const auto M1_poly_coeff = this->M1_poly_coeff;
        const int channels = num_channels;
        const int polynomial_nodes = static_cast<int>(M1_poly_coeff.extent(1));
        const int multiplication_nodes = static_cast<int>(M1_poly_spec.extent(0));
        const int tile_channels = m1_recompute_tile_channels;
        const int vector_length = m1_recompute_vector_length();
        const int scratch_level = m1_recompute_scratch_level();
        const auto scale_adjoint = mh0_a1_scale_adjoint;
        using TeamMember = typename Kokkos::TeamPolicy<>::member_type;
        using ScratchView = Kokkos::View<
            Precision**, Kokkos::LayoutRight,
            typename TeamMember::scratch_memory_space,
            Kokkos::MemoryUnmanaged>;
        auto policy = vector_length == 1
            ? Kokkos::TeamPolicy<>(
                execution_space, num_nodes, Kokkos::AUTO, 1)
            : Kokkos::TeamPolicy<>(
                execution_space, num_nodes, 1, vector_length);
        policy.set_scratch_size(
            scratch_level, Kokkos::PerTeam(
                2*sizeof(Precision)*polynomial_nodes*tile_channels));
        Kokkos::parallel_for(
            "Reverse M1 recompute", policy,
            KOKKOS_LAMBDA (const TeamMember& member) {
                const std::size_t node =
                    static_cast<std::size_t>(member.league_rank());
                ScratchView workspace(
                    member.team_scratch(scratch_level),
                    2*polynomial_nodes, tile_channels);
                if (capture_scale_adjoint)
                    Kokkos::single(Kokkos::PerTeam(member), [=] () {
                        scale_adjoint(node) = 0.0;
                    });
                member.team_barrier();
                for (int channel_begin=0; channel_begin<channels;
                     channel_begin += tile_channels) {
                    const int active_channels =
                        Kokkos::min(tile_channels, channels-channel_begin);
                    double tile_dot = 0.0;
                    Kokkos::parallel_reduce(
                        Kokkos::TeamVectorRange(member, active_channels),
                        [=] (const int lane, double& local_dot) {
                            const int channel = channel_begin+lane;
                            for (int p=0; p<harmonics; ++p)
                                workspace(p,lane) = polynomial_input(node,p,channel);
                            for (int p=0; p<multiplication_nodes; ++p) {
                                const int output_node = harmonics+p;
                                workspace(output_node,lane) =
                                    workspace(M1_poly_spec(p,0),lane)
                                    *workspace(M1_poly_spec(p,1),lane);
                            }
                            for (int p=0; p<polynomial_nodes; ++p)
                                workspace(polynomial_nodes+p,lane) =
                                    M1_poly_coeff(node_types(node),p,channel);
                            for (int p=multiplication_nodes-1; p>=0; --p) {
                                const int output_node = harmonics+p;
                                const int p0 = M1_poly_spec(p,0);
                                const int p1 = M1_poly_spec(p,1);
                                const Precision output_adjoint =
                                    workspace(polynomial_nodes+output_node,lane);
                                workspace(polynomial_nodes+p0,lane) +=
                                    output_adjoint*workspace(p1,lane);
                                workspace(polynomial_nodes+p1,lane) +=
                                    output_adjoint*workspace(p0,lane);
                            }
                            for (int lm=0; lm<harmonics; ++lm) {
                                const Precision adjoint =
                                    workspace(polynomial_nodes+lm,lane)
                                    *M1_adj(node,channel);
                                if (capture_scale_adjoint)
                                    local_dot += static_cast<double>(
                                        workspace(lm,lane))*adjoint;
                                if (accumulate_h1_adjoint)
                                    polynomial_adjoint(node,lm,channel) += adjoint;
                                else
                                    polynomial_adjoint(node,lm,channel) = adjoint;
                            }
                        }, tile_dot);
                    if (capture_scale_adjoint)
                        Kokkos::single(Kokkos::PerTeam(member), [=] () {
                            scale_adjoint(node) += tile_dot;
                        });
                    member.team_barrier();
                }
            });
        m1_recompute_reverse_launch_count += 1;
        complete_device_stage("MACEKokkos::reverse_M1_recompute");
        return;
    }
    if (use_factorized_async_inference())
        if (!single_layer_readout)
            Kokkos::deep_copy(execution_space, polynomial_adjoint, Precision(0));
    else if (!single_layer_readout)
        Kokkos::deep_copy(polynomial_adjoint, Precision(0));
    if (M1_poly_adjoints.extent(0) < num_nodes)
        Kokkos::realloc(
            Kokkos::WithoutInitializing, M1_poly_adjoints,
            num_nodes, M1_poly_coeff.extent(1), num_channels);

    // TODO: prune
    const auto M1_adj = this->M1_adj;
    const auto M1_monomials = this->M1_monomials;
    const auto M1_weights = this->M1_weights;
    const auto M1_poly_spec = this->M1_poly_spec;
    const auto M1_poly_coeff = this->M1_poly_coeff;
    const auto M1_poly_adjoints = this->M1_poly_adjoints;
    const auto M1_poly_values = this->M1_poly_values;
    const auto num_channels = this->num_channels;
    auto M1 = this->M1;

    Kokkos::parallel_for("Reverse M1",
        Kokkos::TeamPolicy<>(execution_space, num_nodes, Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            const std::size_t node = static_cast<std::size_t>(i);
            // initialize
            Kokkos::parallel_for(
                Kokkos::TeamVectorMDRange<
                    Kokkos::Rank<2,Kokkos::Iterate::Right>,Kokkos::TeamPolicy<>::member_type>(
                        team_member, M1_poly_coeff.extent(1), num_channels),
                [&] (const int p, const int k) {
                    M1_poly_adjoints(node,p,k) =
                        M1_poly_coeff(node_types(node),p,k);
                });
            team_member.team_barrier();
            // backwards pass
            for (int p=M1_poly_spec.extent(0)-1; p>=0; --p) {
                const int p0 = M1_poly_spec(p,0);
                const int p1 = M1_poly_spec(p,1);
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team_member, num_channels),
                    [&] (const int k) {
                        M1_poly_adjoints(node,p0,k) +=
                            M1_poly_adjoints(node,harmonics+p,k)
                            *M1_poly_values(node,p1,k);
                        M1_poly_adjoints(node,p1,k) +=
                            M1_poly_adjoints(node,harmonics+p,k)
                            *M1_poly_values(node,p0,k);
                    });
            }
            team_member.team_barrier();
            Kokkos::parallel_for(
                Kokkos::TeamVectorMDRange<
                    Kokkos::Rank<2,Kokkos::Iterate::Right>,Kokkos::TeamPolicy<>::member_type>(
                        team_member, harmonics, num_channels),
                [&] (const int lm, const int k) {
                    const Precision adjoint =
                        M1_poly_adjoints(node,lm,k) * M1_adj(node,k);
                    if (accumulate_h1_adjoint)
                        polynomial_adjoint(node,lm,k) += adjoint;
                    else
                        polynomial_adjoint(node,lm,k) = adjoint;
                });
        });
    complete_device_stage("MACEKokkos::reverse_M1");
}

template <typename Precision>
void MACEKokkos<Precision>::compute_H2(int num_nodes, Kokkos::View<const int*> node_types)
{
    const auto execution_space = use_factorized_async_inference()
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    ensure_mh0_h2_forward_capacity(num_nodes);
    if (use_factorized_async_inference())
        Kokkos::deep_copy(execution_space, H2, 0.0);
    else
        Kokkos::deep_copy(H2, 0.0);

    auto num_channels = this->num_channels;
    auto H2 = this->H2;
    auto H2_weights_for_H1 = this->H2_weights_for_H1;
    auto H1 = this->H1;
    auto H2_weights_for_M1 = this->H2_weights_for_M1;
    auto M1 = this->M1;

    if constexpr (std::is_same_v<
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>) {
        if (symmetrix::host_worker_cblas_enabled()
                && !symmetrix::host_dense_backend_overrides_cblas()
                && num_channels <= host_h2_blas_max_channels) {
            Kokkos::parallel_for(
                "Compute H2 host BLAS",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                    execution_space, 0, num_nodes),
                [=] (const int node) {
                    Precision output[host_h2_blas_max_channels];
                    const int type = node_types(node);
                    symmetrix_blas_gemv<Precision>(
                        CblasRowMajor, CblasTrans,
                        num_channels, num_channels,
                        Precision(1), &H2_weights_for_H1(type,0), num_channels,
                        &H1(node,0,0), 1,
                        Precision(0), output, 1);
                    symmetrix_blas_gemv<Precision>(
                        CblasRowMajor, CblasTrans,
                        num_channels, num_channels,
                        Precision(1), H2_weights_for_M1.data(), num_channels,
                        &M1(node,0), 1,
                        Precision(1), output, 1);
                    for (int channel=0; channel<num_channels; ++channel)
                        H2(node,channel) = static_cast<double>(output[channel]);
                });
            complete_device_stage("MACEKokkos::compute_H2");
            return;
        }
        if (symmetrix::host_flat_dense_enabled()
                && num_channels <= host_h2_blas_max_channels) {
            Kokkos::parallel_for(
                "Compute H2 flat",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                    execution_space, 0, num_nodes),
                [=] (const int node) {
                    Precision output[host_h2_blas_max_channels];
                    const int type = node_types(node);
                    symmetrix::host_dense_gemm_nn(
                        &H1(node,0,0), &H2_weights_for_H1(type,0),
                        output, 1, num_channels, num_channels);
                    symmetrix::host_dense_gemm_nn(
                        &M1(node,0), H2_weights_for_M1.data(),
                        output, 1, num_channels, num_channels, true);
                    for (int channel=0; channel<num_channels; ++channel)
                        H2(node,channel) = static_cast<double>(output[channel]);
                });
            complete_device_stage("MACEKokkos::compute_H2");
            return;
        }
    }

    Kokkos::parallel_for(
        "Compute H2 from H1",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            execution_space, 0, num_nodes*num_channels),
        KOKKOS_LAMBDA (const int ik) {
            const int i = ik / num_channels;
            const int k = ik % num_channels;
                for (int kp=0; kp<num_channels; ++kp) {
                    H2(i,k) += H2_weights_for_H1(node_types(i),kp*num_channels+k) * H1(i,0,kp);
                }
        });
    Kokkos::parallel_for(
        "Compute H2 from M1",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            execution_space, 0, num_nodes*num_channels),
        KOKKOS_LAMBDA (const int ik) {
            const int i = ik / num_channels;
            const int k = ik % num_channels;
            for (int kp=0; kp<num_channels; ++kp) {
                H2(i,k) += H2_weights_for_M1(kp*num_channels+k) * M1(i,kp);
            }
        });
    complete_device_stage("MACEKokkos::compute_H2");
}

template <typename Precision>
void MACEKokkos<Precision>::reverse_H2(int num_nodes, Kokkos::View<const int*> node_types, bool zero_H1_adj)
{
    const auto execution_space = use_factorized_async_inference()
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    ensure_h1_adjoint_capacity();
    if (use_mh0_adjoint_reuse())
        M1_adj = M1;
    if (M1_adj.extent(0) < M1.extent(0))
        Kokkos::realloc(M1_adj, M1.extent(0), M1.extent(1));

    if (use_factorized_async_inference()) {
        if (zero_H1_adj)
            Kokkos::deep_copy(execution_space, H1_adj, 0.0);
        Kokkos::deep_copy(execution_space, M1_adj, 0.0);
    } else {
        if (zero_H1_adj)
            Kokkos::deep_copy(H1_adj, 0.0);
        Kokkos::deep_copy(M1_adj, 0.0);
    }

    auto num_channels = this->num_channels;
    auto H1_adj = this->H1_adj;
    auto M1_adj = this->M1_adj;
    auto H2_adj = this->H2_adj;
    auto H2_weights_for_H1 = this->H2_weights_for_H1;
    auto H2_weights_for_M1 = this->H2_weights_for_M1;
    auto H2_weights_for_H1_reverse = this->H2_weights_for_H1_reverse;
    auto H2_weights_for_M1_reverse = this->H2_weights_for_M1_reverse;

    if constexpr (std::is_same_v<
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>) {
        if (symmetrix::host_worker_cblas_enabled()
                && !symmetrix::host_dense_backend_overrides_cblas()
                && num_channels <= host_h2_blas_max_channels) {
            Kokkos::parallel_for(
                "Reverse H2 host BLAS",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                    execution_space, 0, num_nodes),
                [=] (const int node) {
                    Precision output_adjoint[host_h2_blas_max_channels];
                    for (int channel=0; channel<num_channels; ++channel)
                        output_adjoint[channel] = static_cast<Precision>(
                            H2_adj(node,channel));
                    const int type = node_types(node);
                    symmetrix_blas_gemv<Precision>(
                        CblasRowMajor, CblasNoTrans,
                        num_channels, num_channels,
                        Precision(1), &H2_weights_for_H1(type,0), num_channels,
                        output_adjoint, 1,
                        Precision(1), &H1_adj(node,0,0), 1);
                    symmetrix_blas_gemv<Precision>(
                        CblasRowMajor, CblasNoTrans,
                        num_channels, num_channels,
                        Precision(1), H2_weights_for_M1.data(), num_channels,
                        output_adjoint, 1,
                        Precision(0), &M1_adj(node,0), 1);
                });
            complete_device_stage("MACEKokkos::reverse_H2");
            return;
        }
        if (symmetrix::host_flat_dense_enabled()
                && num_channels <= host_h2_blas_max_channels) {
            Kokkos::parallel_for(
                "Reverse H2 flat",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                    execution_space, 0, num_nodes),
                [=] (const int node) {
                    Precision output_adjoint[host_h2_blas_max_channels];
                    for (int channel=0; channel<num_channels; ++channel)
                        output_adjoint[channel] = static_cast<Precision>(
                            H2_adj(node,channel));
                    const int type = node_types(node);
                    symmetrix::host_dense_gemm_nt(
                        output_adjoint, &H2_weights_for_H1(type,0),
                        &H1_adj(node,0,0), 1,
                        num_channels, num_channels, true);
                    symmetrix::host_dense_gemm_nt(
                        output_adjoint, H2_weights_for_M1.data(),
                        &M1_adj(node,0), 1,
                        num_channels, num_channels);
                });
            complete_device_stage("MACEKokkos::reverse_H2");
            return;
        }
    }

    if constexpr (std::is_same_v<
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>) {
        Kokkos::parallel_for(
            "Reverse H2",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                execution_space, 0, num_nodes*num_channels),
            KOKKOS_LAMBDA (const int ik) {
                const int i = ik / num_channels;
                const int k = ik % num_channels;
                for (int kp=0; kp<num_channels; ++kp) {
                    const Precision adjoint =
                        static_cast<Precision>(H2_adj(i,kp));
                    H1_adj(i,0,k) += H2_weights_for_H1(
                        node_types(i),k*num_channels+kp)*adjoint;
                    M1_adj(i,k) +=
                        H2_weights_for_M1(k*num_channels+kp)*adjoint;
                }
            });
    } else {
        Kokkos::parallel_for(
            "Reverse H2",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                execution_space, 0, num_nodes*num_channels),
            KOKKOS_LAMBDA (const int ik) {
                const int i = ik / num_channels;
                const int k = ik % num_channels;
                for (int kp=0; kp<num_channels; ++kp) {
                    H1_adj(i,0,k) += H2_weights_for_H1_reverse(
                        node_types(i),kp*num_channels+k)*H2_adj(i,kp);
                    M1_adj(i,k) += H2_weights_for_M1_reverse(
                        kp*num_channels+k)*H2_adj(i,kp);
                }
            });
    }
    complete_device_stage("MACEKokkos::reverse_H2");
}

template <typename Precision>
void MACEKokkos<Precision>::ensure_h1_adjoint_capacity()
{
    // M0 is dead after H1 forward, so it can hold the H1 adjoint.
    if (use_h1_m0_adjoint_ping_pong()
        && M0.extent(0) >= H1.extent(0)
        && M0.extent(1) >= H1.extent(1)
        && M0.extent(2) >= H1.extent(2)) {
        H1_adj = M0;
        return;
    }
    if (H1_adj.extent(0) >= H1.extent(0))
        return;
    if (H1_adj.data() == M0.data())
        H1_adj = {};
    if (H1_adj.data() == nullptr) {
        H1_adj = decltype(H1_adj)(
            Kokkos::view_alloc(
                "MACE H1 adjoint", Kokkos::WithoutInitializing),
            H1.extent(0), H1.extent(1), H1.extent(2));
    } else {
        Kokkos::realloc(
            Kokkos::WithoutInitializing,
            H1_adj, H1.extent(0), H1.extent(1), H1.extent(2));
    }
}

template <typename Precision>
void MACEKokkos<Precision>::compute_readout_1(
    const int num_nodes,
    const Kokkos::View<const int*> node_types,
    const bool initialize_h1_adjoint)
{
    const bool async_inference = use_factorized_async_inference();
    const auto execution_space = async_inference
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    if (initialize_h1_adjoint) {
        if (async_inference)
            Kokkos::deep_copy(execution_space, H1_adj, 0.0);
        else
            Kokkos::deep_copy(H1_adj, 0.0);
    }

    auto num_channels = this->num_channels;
    auto node_energies = this->node_energies;
    auto atomic_energies = this->atomic_energies;
    auto H1 = this->H1;
    auto H1_adj = this->H1_adj;
    auto readout_1_weights = this->readout_1_weights;

    // atomic energies
    Kokkos::parallel_for(
        "Compute Readouts 1 atomic energies",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            execution_space, 0, num_nodes),
        KOKKOS_LAMBDA (const int i) {
        node_energies(i) += atomic_energies(node_types(i));
    });
    // first readout
    Kokkos::parallel_for(
        "Compute Readouts 1",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            execution_space, 0, num_nodes),
        KOKKOS_LAMBDA (const int i) {
        for (int k=0; k<num_channels; ++k) {
            node_energies(i) += readout_1_weights(k) * H1(i,0,k);
            if (initialize_h1_adjoint)
                H1_adj(i,0,k) = readout_1_weights(k);
        }
    });
}

template <typename Precision>
void MACEKokkos<Precision>::compute_readout_2(const int num_nodes)
{
    const bool async_inference = use_factorized_async_inference();
    const auto execution_space = async_inference
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    if (use_mh0_adjoint_reuse())
        H2_adj = H2;
    if (H2_adj.extent(0) < static_cast<std::size_t>(num_nodes)
        || H2_adj.extent_int(1) != num_channels)
        Kokkos::realloc(H2_adj, num_nodes, num_channels);
    const bool recompute_readout = readout_recompute;
    auto node_energies = this->node_energies;
    if (!recompute_readout && readout_2_output.extent(0)
            < static_cast<std::size_t>(num_nodes))
        Kokkos::realloc(readout_2_output, num_nodes);
    auto H2 = Kokkos::subview(this->H2, make_pair(0,num_nodes), Kokkos::ALL);
    auto H2_adj = Kokkos::subview(this->H2_adj, make_pair(0,num_nodes), Kokkos::ALL);
    if (recompute_readout) {
        readout_2.evaluate_gradient_accumulate_recompute(
            execution_space, H2, node_energies, H2_adj, false);
    } else if (async_inference) {
        auto readout_2_output = Kokkos::subview(
            this->readout_2_output, make_pair(0,num_nodes));
        readout_2.evaluate_gradient(
            execution_space, H2, readout_2_output, H2_adj, false);
    } else {
        auto readout_2_output = Kokkos::subview(
            this->readout_2_output, make_pair(0,num_nodes));
        readout_2.evaluate_gradient(H2, readout_2_output, H2_adj);
    }
    if (!recompute_readout) {
        auto readout_2_output = Kokkos::subview(
            this->readout_2_output, make_pair(0,num_nodes));
        Kokkos::parallel_for(
            "Compute Readouts 2",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                execution_space, 0, num_nodes),
            KOKKOS_LAMBDA (const int i) {
            node_energies(i) += readout_2_output(i);
        });
    }
}

template <typename Precision>
double MACEKokkos<Precision>::compute_readouts(
    int num_nodes,
    const Kokkos::View<const int*> node_types,
    const bool compute_total_energy,
    const bool completion_fence)
{
    const bool async_inference = use_factorized_async_inference();
    const auto execution_space = async_inference
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    ensure_h1_adjoint_capacity();
    compute_readout_1(num_nodes, node_types, true);
    compute_readout_2(num_nodes);
    auto node_energies = this->node_energies;

    double energy = 0.0;
    if (compute_total_energy)
        Kokkos::parallel_reduce(
            "MACEKokkos::compute_readouts_total_energy",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                execution_space, 0, num_nodes),
            KOKKOS_LAMBDA (const int i, double& local_sum) {
                local_sum += node_energies(i);
            },
            energy);

    if (completion_fence) {
        Kokkos::fence("MACEKokkos::compute_readouts");
        if (mace_uses_prepared_execution(streamed_edges)
            && !has_field_coupling)
            factorized_stage_fence_count += 1;
    }

    return energy;
}


template class MACEKokkos<float>;
template class MACEKokkos<double>;
