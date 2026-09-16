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
#include "device_backend.hpp"
#include "mace_kokkos_jit_plugin_detail.hpp"
#include "standard_m0.hpp"
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
#include "mace_kokkos_spherical_harmonics_detail.hpp"

template <typename Precision>
void MACEKokkos<Precision>::compute_R0(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r)
{
    if (num_active_types == 0)
        throw std::runtime_error("MACEKokkos radial cache has not been prepared.");
    if (r.size() > R0.extent(0)) {
        Kokkos::realloc(R0, r.size(), (l_max+1)*num_channels);
        Kokkos::realloc(R0_deriv, r.size(), (l_max+1)*num_channels);
    }

    // TODO: shouldn't need all this
    // Build i_list
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
    Kokkos::View<int*> i_list("i_list", r.size());
    Kokkos::parallel_for("ij lists",
        num_nodes,
        KOKKOS_LAMBDA (const int i) {
            int ij = first_neigh(i);
            for (int j=0; j<num_neigh(i); ++j) {
                i_list(ij) = i;
                ij += 1;
            }
        });
    Kokkos::fence();

    const int l_max = this->l_max;
    const int num_channels = this->num_channels;
    const auto num_types = num_active_types;
    const auto type_to_active = this->type_to_active;
    const auto h = R0_spline_h;
    const auto x0 = R0_spline_min;
    const auto c = R0_spline_coefficients;
    const auto num_intervals = c.extent(1);
    auto R0 = this->R0;
    auto R0_deriv = this->R0_deriv;

    Kokkos::parallel_for(
        "Compute R0",
        Kokkos::TeamPolicy<>(r.size(), Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int ij = team_member.league_rank();
            const int type_i = type_to_active(node_types(i_list(ij)));
            const int type_j = type_to_active(neigh_types(ij));
            const int type_ij = type_i*num_types+type_j;
            // compute x, x^2, x^3
            int n = static_cast<int>(Kokkos::floor((r(ij)-x0)/h));
            double x = r(ij)-x0-h*n;
            if (n < 0) {
                n = 0;
                x = 0.0;
            } else if (n >= num_intervals) {
                n = num_intervals-1;
                x = h;
            }
            const double xx = x*x;
            const double xxx = xx*x;
            const double two_x = 2*x;
            const double three_xx = 3*xx;
            // compute function values
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team_member, (l_max+1)*num_channels),
                [&] (const int lk) {
                    const double c0 = c(type_ij,n,0,lk);
                    const double c1 = c(type_ij,n,1,lk);
                    const double c2 = c(type_ij,n,2,lk);
                    const double c3 = c(type_ij,n,3,lk);
                    R0(ij,lk) = c0 + c1*x + c2*xx + c3*xxx;
                    R0_deriv(ij,lk) = c1 + c2*two_x + c3*three_xx;
                });
        });
    Kokkos::fence();
}

template <typename Precision>
void MACEKokkos<Precision>::compute_R1(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r,
    const bool include_derivatives)
{
    if (r.size() > R1.extent(0)) {
        Kokkos::realloc(R1, r.size(), Phi1_l.size()*num_channels);
    }
    if (include_derivatives && r.size() > R1_deriv.extent(0)) {
        Kokkos::realloc(R1_deriv, r.size(), Phi1_l.size()*num_channels);
    } else if (!include_derivatives) {
        R1_deriv = decltype(R1_deriv)();
    }
    radial_1.evaluate(
        num_nodes, node_types, num_neigh, neigh_types,
        type_to_active, num_active_types, r, R1, R1_deriv,
        include_derivatives);
    Kokkos::fence();
}

template <typename Precision>
void MACEKokkos<Precision>::compute_Y(
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r,
    const bool evaluator_stream,
    const int edge_begin,
    const int edge_count)
{
    const bool compact_geometry = use_compact_edge_geometry();
    const std::size_t available_edges = compact_geometry || r.extent(0) != 0
        ? r.extent(0) : xyz.extent(0)/3;
    const std::size_t num_edges = edge_count < 0
        ? available_edges : static_cast<std::size_t>(edge_count);
    if (edge_begin < 0 || edge_count < -1
        || static_cast<std::size_t>(edge_begin)+num_edges > available_edges)
        throw std::invalid_argument(
            "Spherical-harmonic edge interval exceeds the geometry extent.");
    if ((!compact_geometry
            && (xyz.extent(0)%3 != 0 || xyz.extent(0) < 3*num_edges))
        || (compact_geometry
            && (r.extent(0) == 0
                || execution_prepared_unit_direction.extent(0) < 3*num_edges)))
        throw std::invalid_argument(
            "Spherical-harmonic coordinates do not cover all edge radii.");
    const int num = static_cast<int>(num_edges);
    const int harmonics = num_lm;
    const std::size_t harmonic_values =
        static_cast<std::size_t>(num)*harmonics;
    if (Y.size() < harmonic_values)
        Kokkos::realloc(Kokkos::WithoutInitializing, Y, harmonic_values);
    if (use_y_only_direct_harmonics()) {
        if (harmonics != 16 || l_max != 3)
            throw std::logic_error(
                "Direct retained-Y execution requires l_max=3.");
        Y_grad = decltype(Y_grad)();
        Y_grad_shuffled = decltype(Y_grad_shuffled)();
        xyz_shuffled = decltype(xyz_shuffled)();
        const auto execution_space = evaluator_stream
            ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
        const auto active_r = Kokkos::subview(
            r, Kokkos::make_pair(
                static_cast<std::size_t>(edge_begin),
                static_cast<std::size_t>(edge_begin)+num_edges));
        if (compact_geometry) {
            const auto active_direction = Kokkos::subview(
                execution_prepared_unit_direction,
                Kokkos::make_pair(
                    static_cast<std::size_t>(3)*edge_begin,
                    static_cast<std::size_t>(3)*(edge_begin+num_edges)));
            symmetrix::launch_spherical_harmonic_values_from_directions<3>(
                execution_space, active_direction, active_r, r_cut,
                Y, num);
        } else {
            const auto active_xyz = Kokkos::subview(
                xyz, Kokkos::make_pair(
                    static_cast<std::size_t>(3)*edge_begin,
                    static_cast<std::size_t>(3)*(edge_begin+num_edges)));
            symmetrix::launch_spherical_harmonic_values_from_directions<3>(
                execution_space, active_xyz, active_r, r_cut, Y, num);
        }
        if (num > 0)
            execution_direct_harmonic_launch_count += 1;
        if (!evaluator_stream)
            execution_space.fence("Complete direct spherical harmonic values");
        return;
    }
    if (edge_begin != 0 || edge_count >= 0)
        throw std::logic_error(
            "Tiled spherical harmonics require direct Y-only execution.");
    ensure_mh0_y_gradient_capacity(harmonic_values);
    if (use_mh0_adjoint_reuse())
        Y_grad_shuffled = Y_grad;
    else if ((Y_grad_shuffled.data() != nullptr
            && Y_grad_shuffled.data() == Y_grad.data())
        || Y_grad_shuffled.size() < Y_grad.size()) {
        const auto execution_space = evaluator_stream
            ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
        execution_space.fence("Replace raw spherical harmonic gradients");
        Y_grad_shuffled = decltype(Y_grad_shuffled)();
        Y_grad_shuffled = decltype(Y_grad_shuffled)(
            Kokkos::view_alloc(
                "Execution raw spherical harmonic gradients",
                Kokkos::WithoutInitializing),
            Y_grad.size());
    }
    if (xyz_shuffled.size() < 3*static_cast<std::size_t>(num))
        Kokkos::realloc(
            Kokkos::WithoutInitializing, xyz_shuffled,
            3*static_cast<std::size_t>(num));
    ensure_spherical_harmonics_state();

    auto Y = this->Y;
    auto Y_grad = this->Y_grad;
    auto Y_grad_shuffled = this->Y_grad_shuffled;
    auto xyz_shuffled = this->xyz_shuffled;
    const auto unit_direction = execution_prepared_unit_direction;
    const double normalization_factor = 2*std::sqrt(M_PI);

#if defined(KOKKOS_ENABLE_HIP)
    const auto execution_space = evaluator_stream
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    Kokkos::parallel_for(
        "shuffle_xyz",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            execution_space, 0, num),
        KOKKOS_LAMBDA (const int i) {
            const std::size_t edge = static_cast<std::size_t>(i);
            xyz_shuffled(3*edge) = compact_geometry
                ? unit_direction(3*edge+2) : xyz(3*edge+2);
            xyz_shuffled(3*edge+1) = compact_geometry
                ? unit_direction(3*edge) : xyz(3*edge);
            xyz_shuffled(3*edge+2) = compact_geometry
                ? unit_direction(3*edge+1) : xyz(3*edge+1);
        });
    symmetrix::launch_spherical_harmonics_device(
        execution_space, xyz_shuffled, Y, Y_grad_shuffled, num, l_max);
    if (num > 0) {
        execution_sphericart_launch_count += 1;
        execution_sphericart_async_launch_count += 1;
    }
    using large_range_policy = Kokkos::RangePolicy<
        Kokkos::DefaultExecutionSpace,Kokkos::IndexType<std::size_t>>;
    Kokkos::parallel_for(
        "normalize_Y", large_range_policy(
            execution_space, 0, harmonic_values),
        KOKKOS_LAMBDA (const std::size_t i) { Y(i) *= normalization_factor; });
    Kokkos::parallel_for(
        "unshuffle_normalize_Y_grad",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            execution_space, 0, num),
        KOKKOS_LAMBDA (const int i) {
            const std::size_t edge = static_cast<std::size_t>(i);
            const Precision gradient_scale = static_cast<Precision>(
                normalization_factor)
                *(compact_geometry
                    ? Precision(1)/static_cast<Precision>(r(edge))
                    : Precision(1));
            for (int lm=0; lm<harmonics; ++lm) {
                const auto x = Y_grad_shuffled(3*edge*harmonics+lm);
                const auto y = Y_grad_shuffled(
                    3*edge*harmonics+harmonics+lm);
                const auto z = Y_grad_shuffled(
                    3*edge*harmonics+2*harmonics+lm);
                Y_grad(3*edge*harmonics+lm) = gradient_scale*y;
                Y_grad(3*edge*harmonics+harmonics+lm) = gradient_scale*z;
                Y_grad(3*edge*harmonics+2*harmonics+lm) = gradient_scale*x;
            }
        });
#elif !defined(SYMMETRIX_SPHERICART_CUDA)
    static_cast<void>(evaluator_stream);
    Kokkos::parallel_for("shuffle_xyz", num, KOKKOS_LAMBDA (const int i) {
        const std::size_t edge = static_cast<std::size_t>(i);
        xyz_shuffled(3*edge) = compact_geometry
            ? unit_direction(3*edge+2) : xyz(3*edge+2);
        xyz_shuffled(3*edge+1) = compact_geometry
            ? unit_direction(3*edge) : xyz(3*edge);
        xyz_shuffled(3*edge+2) = compact_geometry
            ? unit_direction(3*edge+1) : xyz(3*edge+1);
    });
    Kokkos::fence();

    const auto active_xyz = Kokkos::subview(
        xyz_shuffled,
        Kokkos::make_pair(std::size_t(0), 3*static_cast<std::size_t>(num)));
    const auto active_Y = Kokkos::subview(
        Y, Kokkos::make_pair(std::size_t(0), harmonic_values));
    const auto active_raw_gradient = Kokkos::subview(
        Y_grad_shuffled,
        Kokkos::make_pair(std::size_t(0), 3*harmonic_values));
    auto h_xyz = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), active_xyz);
    auto h_Y = Kokkos::create_mirror_view(active_Y);
    auto h_raw_gradient = Kokkos::create_mirror_view(active_raw_gradient);
    spherical_harmonics_state->calculator.compute_array_with_gradients(
        h_xyz.data(), 3*static_cast<std::size_t>(num),
        h_Y.data(), harmonic_values,
        h_raw_gradient.data(), 3*harmonic_values);
    if (num > 0)
        execution_sphericart_launch_count += 1;
    Kokkos::deep_copy(active_Y, h_Y);
    Kokkos::deep_copy(active_raw_gradient, h_raw_gradient);

    using large_range_policy = Kokkos::RangePolicy<
        Kokkos::DefaultExecutionSpace,Kokkos::IndexType<std::size_t>>;
    Kokkos::parallel_for("normalize_Y", large_range_policy(0, harmonic_values),
        KOKKOS_LAMBDA (const std::size_t i) {
            Y(i) *= normalization_factor;
        });
    Kokkos::parallel_for("unshuffle_normalize_Y_grad", num,
        KOKKOS_LAMBDA (const int i) {
            const std::size_t edge = static_cast<std::size_t>(i);
            const Precision gradient_scale = static_cast<Precision>(
                normalization_factor)
                *(compact_geometry
                    ? Precision(1)/static_cast<Precision>(r(edge))
                    : Precision(1));
            for (int lm=0; lm<harmonics; ++lm) {
                const auto x = Y_grad_shuffled(3*edge*harmonics+lm);
                const auto y = Y_grad_shuffled(
                    3*edge*harmonics+harmonics+lm);
                const auto z = Y_grad_shuffled(
                    3*edge*harmonics+2*harmonics+lm);
                Y_grad(3*edge*harmonics+lm) = gradient_scale*y;
                Y_grad(3*edge*harmonics+harmonics+lm) = gradient_scale*z;
                Y_grad(3*edge*harmonics+2*harmonics+lm) = gradient_scale*x;
            }
        });
    Kokkos::fence();

#else
    const auto execution_space = evaluator_stream
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    Kokkos::parallel_for(
        "shuffle_xyz",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            execution_space, 0, num),
        KOKKOS_LAMBDA (const int i) {
            const std::size_t edge = static_cast<std::size_t>(i);
            xyz_shuffled(3*edge) = compact_geometry
                ? unit_direction(3*edge+2) : xyz(3*edge+2);
            xyz_shuffled(3*edge+1) = compact_geometry
                ? unit_direction(3*edge) : xyz(3*edge);
            xyz_shuffled(3*edge+2) = compact_geometry
                ? unit_direction(3*edge+1) : xyz(3*edge+1);
        });

    if (evaluator_stream) {
        spherical_harmonics_state->calculator.compute_with_gradients_async(
            xyz_shuffled.data(), num, Y.data(), Y_grad_shuffled.data(),
            reinterpret_cast<void*>(execution_space.cuda_stream()));
        if (num > 0)
            execution_sphericart_async_launch_count += 1;
    } else {
        execution_space.fence("Prepare CUDA spherical-harmonic coordinates");
        spherical_harmonics_state->calculator.compute_with_gradients(
            xyz_shuffled.data(), num, Y.data(), Y_grad_shuffled.data());
    }
    if (num > 0)
        execution_sphericart_launch_count += 1;

    Kokkos::parallel_for(
        "normalize_Y",
        Kokkos::RangePolicy<
            Kokkos::DefaultExecutionSpace,Kokkos::IndexType<std::size_t>>(
            execution_space, 0, harmonic_values),
        KOKKOS_LAMBDA (const std::size_t i) {
            Y(i) *= normalization_factor;
        });
    Kokkos::parallel_for(
        "unshuffle_normalize_Y_grad",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            execution_space, 0, num),
        KOKKOS_LAMBDA (const int i) {
            const std::size_t edge = static_cast<std::size_t>(i);
            const Precision gradient_scale = static_cast<Precision>(
                normalization_factor)
                *(compact_geometry
                    ? Precision(1)/static_cast<Precision>(r(edge))
                    : Precision(1));
            for (int lm=0; lm<harmonics; ++lm) {
                const auto x = Y_grad_shuffled(3*edge*harmonics+lm);
                const auto y = Y_grad_shuffled(
                    3*edge*harmonics+harmonics+lm);
                const auto z = Y_grad_shuffled(
                    3*edge*harmonics+2*harmonics+lm);
                Y_grad(3*edge*harmonics+lm) = gradient_scale*y;
                Y_grad(3*edge*harmonics+harmonics+lm) = gradient_scale*z;
                Y_grad(3*edge*harmonics+2*harmonics+lm) = gradient_scale*x;
            }
        });
    if (!evaluator_stream)
        execution_space.fence("Complete CUDA spherical harmonics");
#endif
}

template <typename Precision>
void MACEKokkos<Precision>::compute_A0(
    const int num_nodes,
    View<const int*> node_types,
    View<const int*> num_neigh,
    View<const int*> neigh_types)
{
    ensure_mh0_a0_forward_capacity(num_nodes);
    Kokkos::deep_copy(A0, 0.0);

    Kokkos::View<int*> first_neigh("first_neigh", num_nodes);
    Kokkos::parallel_scan("Compute first_neigh",
        num_nodes,
        KOKKOS_LAMBDA (const int i, int& update, const bool final) {
            if (final)
                first_neigh(i) = update;
            update += num_neigh(i);
        });
    Kokkos::fence();

    const int num_lm = this->num_lm;
    const int num_channels = this->num_channels;
    const auto R0 = this->R0;
    const auto Y = this->Y;
    auto A0 = this->A0;

    const auto a0_scratch_bytes = admitted_team_scratch_bytes<>(
        "MACEKokkos::compute_A0",
        {static_cast<std::size_t>(num_channels), sizeof(double)});
    parallel_for("Compute A0",
        TeamPolicy<>(num_nodes*num_lm, Kokkos::AUTO, 32)
             .set_scratch_size(0, PerTeam(a0_scratch_bytes)),
        KOKKOS_LAMBDA (TeamPolicy<>::member_type team_member) {
            const int work = team_member.league_rank();
            const std::size_t node = static_cast<std::size_t>(work/num_lm);
            const int lm = work%num_lm;
            const int l = Kokkos::sqrt(lm);
            for (int j=0; j<num_neigh(node); ++j) {
                const int ij = first_neigh(node) + j;
                const std::size_t edge = static_cast<std::size_t>(ij);
                const double Y_ij_lm = Y(edge*num_lm+lm);
                parallel_for(
                    TeamVectorRange(team_member, num_channels),
                    [=] (const int k) {
                        A0(node,lm,k) += R0(ij,l*num_channels+k) * Y_ij_lm;
                    });
            }
        });

    Kokkos::fence();
}

template <typename Precision>
void MACEKokkos<Precision>::reverse_A0(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r)
{
    auto A0_adj = this->A0_adj;
    auto num_lm = this->num_lm;
    auto num_channels = this->num_channels;
    auto R0 = this->R0;
    auto R0_deriv = this->R0_deriv;
    auto Y = this->Y;
    auto Y_grad = this->Y_grad;
    auto node_forces = this->node_forces;

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

    Kokkos::parallel_for("Reverse A0",
        Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const std::size_t node = static_cast<std::size_t>(
                team_member.league_rank());
            const int i0 = first_neigh(node);
            for (int j=0; j<num_neigh(node); ++j) {
                const int ij = i0 + j;
                const std::size_t edge = static_cast<std::size_t>(ij);
                const double r_ij = r(ij);
                const double x_ij = xyz(3*edge) / r_ij;
                const double y_ij = xyz(3*edge+1) / r_ij;
                const double z_ij = xyz(3*edge+2) / r_ij;
                const Precision* Y_ij = &Y(edge*num_lm);
                const Precision* Y_grad_ij = &Y_grad(3*edge*num_lm);
                double f_x, f_y, f_z;
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team_member, num_lm),
                    [&] (const int lm, double& f_x, double& f_y, double& f_z) {
                        const int l = Kokkos::sqrt(lm);
                        double t1, t2;
                        Kokkos::parallel_reduce(
                            Kokkos::ThreadVectorRange(team_member, num_channels),
                            [&] (const int k, double& t1, double& t2) {
                                t1 += R0_deriv(ij,l*num_channels+k) * A0_adj(node,lm,k);
                                t2 += R0(ij,l*num_channels+k) * A0_adj(node,lm,k);
                            }, t1, t2);
                        f_x += t1*x_ij*Y_ij[lm] + t2*Y_grad_ij[lm];
                        f_y += t1*y_ij*Y_ij[lm] + t2*Y_grad_ij[num_lm+lm];
                        f_z += t1*z_ij*Y_ij[lm] + t2*Y_grad_ij[2*num_lm+lm];
                    }, f_x, f_y, f_z);
                    Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
                        node_forces(3*edge)   -= f_x;
                        node_forces(3*edge+1) -= f_y;
                        node_forces(3*edge+2) -= f_z;
                    });
            }
        });
    Kokkos::fence();
}

template <typename Precision>
void MACEKokkos<Precision>::compute_A0_streamed(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r,
    const int workspace_edge_begin)
{
    const auto execution_space = use_factorized_async_inference()
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    if (num_active_types == 0)
        throw std::runtime_error("MACEKokkos radial cache has not been prepared.");
    ensure_mh0_a0_forward_capacity(num_nodes);
    const bool module_scope =
        streamed_edges == MACEStreamedEdgesMode::generic
        || (mace_uses_prepared_execution(streamed_edges)
            && (single_layer_readout
                || factorized_source_strategy
                    == FactorizedSourceStrategy::jit_plugin));
    standard_r0_module_active = module_scope
        && use_r0_module()
        && (selected_r0_implementation == R0Implementation::device_module
            || standard_r0_executor != StandardR0Executor::v1);
    if (standard_r0_module_active) {
        if (A0_scaled && standard_r0_density_state.extent(0)
                < static_cast<std::size_t>(num_nodes))
            Kokkos::realloc(standard_r0_density_state, num_nodes);
        if (selected_r0_implementation == R0Implementation::device_module
            && !single_layer_tiled_plan_active
            && !dual_layer_tiled_plan_active) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
            if (!r0_device_module_ready())
                throw std::runtime_error(
                    "Execution R0 device-module forward is unavailable.");
            const auto environment = execution_device_execution_environment();
            const int persistent_blocks = std::max(
                1, r0_device_persistent_blocks_per_compute_unit
                    *environment.compute_unit_count);
            const SymmetrixJitR0SplineV1 radial{
                sizeof(SymmetrixJitR0SplineV1),
                static_cast<std::uint32_t>(R0_spline_coefficients.extent(0)),
                static_cast<std::uint32_t>(R0_spline_coefficients.extent(1)),
                static_cast<std::uint32_t>(R0_spline_coefficients.extent(3)),
                R0_spline_h, R0_spline_min,
                R0_spline_coefficients.data()};
            SymmetrixJitR0SplineV1 density{
                sizeof(SymmetrixJitR0SplineV1), 0u, 0u, 0u,
                0.0, 0.0, nullptr};
            if (A0_scaled)
                density = {
                    sizeof(SymmetrixJitR0SplineV1),
                    static_cast<std::uint32_t>(A0_splines.edge_type_count()),
                    static_cast<std::uint32_t>(A0_splines.interval_count()),
                    static_cast<std::uint32_t>(A0_splines.function_count()),
                    A0_splines.spline_h(), A0_splines.spline_x0(),
                    A0_splines.coefficient_data()};
            const SymmetrixJitR0ArgsV1 args{
                sizeof(SymmetrixJitR0ArgsV1), 0u, num_nodes,
                static_cast<std::int64_t>(r.extent(0)), num_active_types,
                num_channels, l_max, 0, A0_scaled ? 1 : 0, 0,
                node_types.data(), num_neigh.data(), streamed_first_neigh.data(),
                neigh_types.data(), nullptr, type_to_active.data(), nullptr,
                r.data(), radial, density, Y.data(), nullptr, A0.data(), nullptr,
                standard_r0_density_state.data(), nullptr, nullptr, r_cut};
            ExecutionDeviceBackend::DeviceGuard device_guard(
                ExecutionDeviceBackend::device_ordinal(execution_space));
            void* stream = reinterpret_cast<void*>(
                ExecutionDeviceBackend::native_stream(execution_space));
            if (A0_scaled) {
                r0_device_module->launch_r0_density_prepare(
                    args, stream, persistent_blocks);
                standard_r0_module_launch_count += 1;
            }
            r0_device_module->launch_r0_forward(
                args, stream, persistent_blocks);
            standard_r0_module_launch_count += 1;
            standard_r0_density_scale_fused = A0_scaled;
            return;
#else
            throw std::runtime_error(
                "Execution R0 device modules require a device backend.");
#endif
        }
        if (A0_scaled) {
            symmetrix::standard_r0::launch_density_prepare(
                execution_space,
                num_nodes,
                num_active_types,
                r_cut,
                node_types,
                num_neigh,
                neigh_types,
                streamed_first_neigh,
                r,
                type_to_active,
                A0_splines,
                standard_r0_density_state);
            standard_r0_module_launch_count += 1;
        }
        bool use_fused_r0_forward = false;
#ifdef KOKKOS_ENABLE_HIP
        if constexpr (std::is_same_v<Precision,float>)
            use_fused_r0_forward = single_layer_readout;
#endif
        if (use_fused_r0_forward)
            symmetrix::standard_r0::launch_forward_all_l(
                execution_space,
                num_nodes,
                num_active_types,
                R0_spline_h,
                R0_spline_min,
                r_cut,
                node_types,
                num_neigh,
                neigh_types,
                streamed_first_neigh,
                r,
                type_to_active,
                R0_spline_coefficients,
                Y,
                A0_scaled,
                standard_r0_density_state,
                A0,
                workspace_edge_begin);
        else
            symmetrix::standard_r0::launch_forward(
                execution_space,
                num_nodes,
                num_active_types,
                R0_spline_h,
                R0_spline_min,
                r_cut,
                node_types,
                num_neigh,
                neigh_types,
                streamed_first_neigh,
                r,
                type_to_active,
                R0_spline_coefficients,
                Y,
                A0_scaled,
                standard_r0_density_state,
                A0,
                workspace_edge_begin);
        standard_r0_module_launch_count += 1;
        standard_r0_density_scale_fused = A0_scaled;
        return;
    }
    standard_r0_density_scale_fused = false;
    if (use_factorized_async_inference())
        Kokkos::deep_copy(execution_space, A0, Precision(0));
    else
        Kokkos::deep_copy(A0, Precision(0));
    const int l_max = this->l_max;
    const int num_lm = this->num_lm;
    const int num_channels = this->num_channels;
    const int num_types = num_active_types;
    const double h = R0_spline_h;
    const double x0 = R0_spline_min;
    const auto coefficients = R0_spline_coefficients;
    const int num_intervals = coefficients.extent(1);
    const auto type_to_active = this->type_to_active;
    const auto first_neigh = streamed_first_neigh;
    const auto Y = this->Y;
    auto A0 = this->A0;

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    using spline_coordinate_type = Precision;
    const Kokkos::TeamPolicy<> policy(
        execution_space, num_nodes*(l_max+1), 1, 32);
#else
    using spline_coordinate_type = double;
    const Kokkos::TeamPolicy<> policy(
        execution_space, num_nodes*(l_max+1), Kokkos::AUTO, 32);
#endif
    Kokkos::parallel_for(
        "MACEKokkos::compute_A0_streamed",
        policy,
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int work = team_member.league_rank();
            const std::size_t node = static_cast<std::size_t>(
                work/(l_max+1));
            const int l = work%(l_max+1);
            const int lm_begin = l*l;
            const int lm_end = (l+1)*(l+1);
            const int type_i = type_to_active(node_types(node));
            const int i0 = first_neigh(node);
            for (int j=0; j<num_neigh(node); ++j) {
                const int ij = i0+j;
                const std::size_t edge = static_cast<std::size_t>(ij);
                const int type_j = type_to_active(neigh_types(ij));
                const int edge_type = type_i*num_types+type_j;
                int interval = static_cast<int>(Kokkos::floor((r(ij)-x0)/h));
                spline_coordinate_type x = static_cast<spline_coordinate_type>(
                    r(ij)-x0-h*interval);
                if (interval < 0) {
                    interval = 0;
                    x = 0.0;
                } else if (interval >= num_intervals) {
                    interval = num_intervals-1;
                    x = h;
                }
                const spline_coordinate_type xx = x*x;
                const spline_coordinate_type xxx = xx*x;
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team_member, num_channels),
                    [=] (const int k) {
                        const int function = l*num_channels+k;
                        const Precision value =
                            coefficients(edge_type,interval,0,function)
                            + coefficients(edge_type,interval,1,function)*x
                            + coefficients(edge_type,interval,2,function)*xx
                            + coefficients(edge_type,interval,3,function)*xxx;
                        const std::size_t workspace_edge =
                            edge-static_cast<std::size_t>(workspace_edge_begin);
                        for (int lm=lm_begin; lm<lm_end; ++lm)
                            A0(node,lm,k) += value*Y(
                                workspace_edge*num_lm+lm);
                    });
            }
        });
    complete_device_stage("MACEKokkos::compute_A0_streamed");
}

template <typename Precision>
void MACEKokkos<Precision>::reverse_A0_streamed(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r,
    const int receiver_base,
    const int edge_begin,
    const int edge_count)
{
    const auto execution_space = use_factorized_async_inference()
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    const int active_edge_count = edge_count < 0
        ? static_cast<int>(r.extent(0)) : edge_count;
    if (receiver_base < 0 || edge_begin < 0 || active_edge_count < 0
        || static_cast<std::size_t>(edge_begin)+active_edge_count > r.extent(0))
        throw std::invalid_argument(
            "R0 reverse requires a valid receiver base and edge interval.");
    if (standard_r0_module_active) {
        Kokkos::View<int*> edge_receivers;
        if (dual_layer_tiled_plan_active) {
            edge_receivers = Kokkos::subview(
                dual_layer_edge_local_receivers,
                Kokkos::make_pair(
                    std::size_t(0),
                    static_cast<std::size_t>(active_edge_count)));
        } else {
            edge_receivers = streamed_edges == MACEStreamedEdgesMode::generic
                ? streamed_edge_receivers : execution_edge_receivers;
        }
        if (selected_r0_implementation == R0Implementation::device_module
            && !single_layer_tiled_plan_active
            && !dual_layer_tiled_plan_active) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
            if (!r0_device_module_ready())
                throw std::runtime_error(
                    "Execution R0 device-module reverse is unavailable.");
            const auto environment = execution_device_execution_environment();
            const int persistent_blocks = std::max(
                1, r0_device_persistent_blocks_per_compute_unit
                    *environment.compute_unit_count);
            const SymmetrixJitR0SplineV1 radial{
                sizeof(SymmetrixJitR0SplineV1),
                static_cast<std::uint32_t>(R0_spline_coefficients.extent(0)),
                static_cast<std::uint32_t>(R0_spline_coefficients.extent(1)),
                static_cast<std::uint32_t>(R0_spline_coefficients.extent(3)),
                R0_spline_h, R0_spline_min,
                R0_spline_coefficients.data()};
            SymmetrixJitR0SplineV1 density{
                sizeof(SymmetrixJitR0SplineV1), 0u, 0u, 0u,
                0.0, 0.0, nullptr};
            if (A0_scaled)
                density = {
                    sizeof(SymmetrixJitR0SplineV1),
                    static_cast<std::uint32_t>(A0_splines.edge_type_count()),
                    static_cast<std::uint32_t>(A0_splines.interval_count()),
                    static_cast<std::uint32_t>(A0_splines.function_count()),
                    A0_splines.spline_h(), A0_splines.spline_x0(),
                    A0_splines.coefficient_data()};
            const bool compact = use_compact_edge_geometry();
            const void* coordinates = compact
                ? static_cast<const void*>(
                    execution_prepared_unit_direction.data()+3*edge_begin)
                : static_cast<const void*>(xyz.data()+3*edge_begin);
            const auto* harmonic_gradients = Y_grad.data() == nullptr
                ? nullptr : Y_grad.data()+static_cast<std::size_t>(3)*edge_begin*num_lm;
            const SymmetrixJitR0ArgsV1 args{
                sizeof(SymmetrixJitR0ArgsV1),
                static_cast<std::uint32_t>(receiver_base), num_nodes,
                static_cast<std::int64_t>(active_edge_count), num_active_types,
                num_channels, l_max, compact ? 1 : 0, A0_scaled ? 1 : 0, 0,
                node_types.data(), num_neigh.data(), streamed_first_neigh.data(),
                neigh_types.data()+edge_begin,
                edge_receivers.data()+edge_begin, type_to_active.data(),
                coordinates, r.data()+edge_begin, radial, density,
                Y.data()+static_cast<std::size_t>(edge_begin)*num_lm,
                harmonic_gradients,
                A0.data(), A0_adj.data(), standard_r0_density_state.data(),
                nullptr, node_forces.data()+3*edge_begin, r_cut};
            ExecutionDeviceBackend::DeviceGuard device_guard(
                ExecutionDeviceBackend::device_ordinal(execution_space));
            r0_device_module->launch_r0_coordinate_reverse(
                args,
                reinterpret_cast<void*>(
                    ExecutionDeviceBackend::native_stream(execution_space)),
                persistent_blocks);
            standard_r0_module_launch_count += 1;
            return;
#else
            throw std::runtime_error(
                "Execution R0 device modules require a device backend.");
#endif
        }
        const auto launch_standard_reverse = [&] (
                const auto& coordinates, const bool coordinates_are_unit) {
            auto resolved_executor = standard_r0_executor;
            if (resolved_executor == StandardR0Executor::automatic) {
                if (dual_layer_tiled_plan_active && receiver_base == 0)
                    resolved_executor = StandardR0Executor::v2_edge16;
                else if (edge_count >= 0)
                    resolved_executor = StandardR0Executor::v2_receiver;
                else if constexpr (std::is_same_v<
                        typename Kokkos::DefaultExecutionSpace::memory_space,
                        Kokkos::HostSpace>)
                    resolved_executor = StandardR0Executor::v2_receiver;
                else
                    resolved_executor = StandardR0Executor::v2_edge16;
            }
            switch (resolved_executor) {
            case StandardR0Executor::v2_edge16:
            case StandardR0Executor::v2_edge32: {
                const int persistent_blocks =
                    resolve_execution_persistent_blocks(
                        symmetrix::standard_r0::module_id,
                        execution_space, r.extent(0), "r0_reverse");
                if (persistent_blocks <= 0)
                    throw std::runtime_error(
                        "Execution R0 standard-module reverse has no launch profile.");
                if (standard_r0_executor != StandardR0Executor::v2_edge32)
                    symmetrix::standard_r0::launch_coordinate_reverse_edge16(
                        execution_space,
                        persistent_blocks,
                        r.extent(0),
                        num_active_types,
                        R0_spline_h,
                        R0_spline_min,
                        r_cut,
                        node_types,
                        neigh_types,
                        edge_receivers,
                        coordinates,
                        r,
                        coordinates_are_unit,
                        type_to_active,
                        R0_spline_coefficients,
                        A0_scaled,
                        A0_splines,
                        standard_r0_density_state,
                        A0_adj,
                        Y,
                        Y_grad,
                        node_forces);
                else
                    symmetrix::standard_r0::launch_coordinate_reverse_edge32(
                        execution_space,
                        persistent_blocks,
                        r.extent(0),
                        num_active_types,
                        R0_spline_h,
                        R0_spline_min,
                        r_cut,
                        node_types,
                        neigh_types,
                        edge_receivers,
                        coordinates,
                        r,
                        coordinates_are_unit,
                        type_to_active,
                        R0_spline_coefficients,
                        A0_scaled,
                        A0_splines,
                        standard_r0_density_state,
                        A0_adj,
                        Y,
                        Y_grad,
                        node_forces);
                break;
            }
            default:
                symmetrix::standard_r0::launch_coordinate_reverse(
                    execution_space,
                    num_nodes,
                    num_active_types,
                    R0_spline_h,
                    R0_spline_min,
                    r_cut,
                    node_types,
                    num_neigh,
                    neigh_types,
                    streamed_first_neigh,
                    coordinates,
                    r,
                    coordinates_are_unit,
                    type_to_active,
                    R0_spline_coefficients,
                    A0_scaled,
                    A0_splines,
                    standard_r0_density_state,
                    A0_adj,
                    Y,
                    Y_grad,
                    node_forces,
                    edge_begin);
                break;
            }
        };
        if (use_compact_edge_geometry())
            launch_standard_reverse(execution_prepared_unit_direction, true);
        else
            launch_standard_reverse(xyz, false);
        standard_r0_module_launch_count += 1;
        return;
    }
    const int l_max = this->l_max;
    const int num_lm = this->num_lm;
    const int num_channels = this->num_channels;
    const int num_types = num_active_types;
    const double h = R0_spline_h;
    const double x0 = R0_spline_min;
    const auto coefficients = R0_spline_coefficients;
    const int num_intervals = coefficients.extent(1);
    const auto type_to_active = this->type_to_active;
    const auto first_neigh = streamed_first_neigh;
    const auto A0_adj = this->A0_adj;
    const auto Y = this->Y;
    const auto Y_grad = this->Y_grad;
    auto node_forces = this->node_forces;

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    using force_accumulation_type = Precision;
    const int team_size = std::max(1, std::min(l_max+1, 8));
    const Kokkos::TeamPolicy<> policy(
        execution_space, num_nodes, team_size, 32);
#else
    using force_accumulation_type = double;
    const Kokkos::TeamPolicy<> policy(
        execution_space, num_nodes, Kokkos::AUTO, 32);
#endif
    Kokkos::parallel_for(
        "MACEKokkos::reverse_A0_streamed",
        policy,
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const std::size_t node = static_cast<std::size_t>(
                team_member.league_rank());
            const int type_i = type_to_active(node_types(node));
            const int i0 = first_neigh(node);
            for (int j=0; j<num_neigh(node); ++j) {
                const int ij = i0+j;
                const std::size_t edge = static_cast<std::size_t>(ij);
                const std::size_t workspace_edge = edge
                    -static_cast<std::size_t>(edge_begin);
                const int type_j = type_to_active(neigh_types(ij));
                const int edge_type = type_i*num_types+type_j;
                int interval = static_cast<int>(Kokkos::floor((r(ij)-x0)/h));
                force_accumulation_type x =
                    static_cast<force_accumulation_type>(
                        r(ij)-x0-h*interval);
                if (interval < 0) {
                    interval = 0;
                    x = 0.0;
                } else if (interval >= num_intervals) {
                    interval = num_intervals-1;
                    x = h;
                }
                const force_accumulation_type xx = x*x;
                const force_accumulation_type xxx = xx*x;
                const force_accumulation_type x_over_r =
                    static_cast<force_accumulation_type>(
                        xyz(3*edge)/r(ij));
                const force_accumulation_type y_over_r =
                    static_cast<force_accumulation_type>(
                        xyz(3*edge+1)/r(ij));
                const force_accumulation_type z_over_r =
                    static_cast<force_accumulation_type>(
                        xyz(3*edge+2)/r(ij));
                force_accumulation_type f_x, f_y, f_z;
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team_member, l_max+1),
                    [=] (const int l,
                         force_accumulation_type& f_x,
                         force_accumulation_type& f_y,
                         force_accumulation_type& f_z) {
                        const int lm_begin = l*l;
                        const int lm_end = (l+1)*(l+1);
                        force_accumulation_type l_fx, l_fy, l_fz;
                        Kokkos::parallel_reduce(
                            Kokkos::ThreadVectorRange(team_member, num_channels),
                            [=] (const int k,
                                 force_accumulation_type& l_fx,
                                 force_accumulation_type& l_fy,
                                 force_accumulation_type& l_fz) {
                                const int function = l*num_channels+k;
                                const Precision c1 = coefficients(edge_type,interval,1,function);
                                const Precision c2 = coefficients(edge_type,interval,2,function);
                                const Precision c3 = coefficients(edge_type,interval,3,function);
                                const Precision value =
                                    coefficients(edge_type,interval,0,function)
                                    + c1*x+c2*xx+c3*xxx;
                                const Precision derivative = c1+2*c2*x+3*c3*xx;
                                for (int lm=lm_begin; lm<lm_end; ++lm) {
                                    const Precision adjoint = A0_adj(node,lm,k);
                                    const Precision radial = derivative
                                        *Y(workspace_edge*num_lm+lm)*adjoint;
                                    const Precision angular = value*adjoint;
                                    l_fx += radial*x_over_r
                                        + angular*Y_grad(
                                            3*workspace_edge*num_lm+lm);
                                    l_fy += radial*y_over_r
                                        + angular*Y_grad(
                                            (3*workspace_edge+1)*num_lm+lm);
                                    l_fz += radial*z_over_r
                                        + angular*Y_grad(
                                            (3*workspace_edge+2)*num_lm+lm);
                                }
                            }, l_fx, l_fy, l_fz);
                        f_x += l_fx;
                        f_y += l_fy;
                        f_z += l_fz;
                    }, f_x, f_y, f_z);
                Kokkos::single(Kokkos::PerTeam(team_member), [=]() {
                    node_forces(3*workspace_edge) -= f_x;
                    node_forces(3*workspace_edge+1) -= f_y;
                    node_forces(3*workspace_edge+2) -= f_z;
                });
            }
        });
    complete_device_stage("MACEKokkos::reverse_A0_streamed");
}

template <typename Precision>
void MACEKokkos<Precision>::compute_A0_scaled(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r)
{
    if (not A0_scaled) return;
    if (standard_r0_density_scale_fused) return;
    const bool streamed_recompute =
        mace_uses_prepared_execution(streamed_edges);
    const auto execution_space = use_factorized_async_inference()
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();

    // compute A0 splines
    if (!streamed_recompute && r.size() > A0_spline_values.extent(0)) {
        Kokkos::realloc(A0_spline_values, r.size(), 1);
        Kokkos::realloc(A0_spline_derivs, r.size(), 1);
    }
    if (!streamed_recompute)
        A0_splines.evaluate(
            num_nodes, node_types, num_neigh, neigh_types,
            type_to_active, num_active_types, r,
            A0_spline_values, A0_spline_derivs);

    Kokkos::View<int*> first_neigh;
    if (streamed_recompute)
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
    const auto A0_spline_values = this->A0_spline_values;
    const auto A0_spline_derivs = this->A0_spline_derivs;
    const auto A0_splines = this->A0_splines;
    const auto type_to_active = this->type_to_active;
    const auto active_type_count = num_active_types;
    const auto num_channels = this->num_channels;
    const auto num_lm = this->num_lm;
    auto A0 = this->A0;
    Kokkos::parallel_for(
        "MACEKokkos::compute_A0_scaled",
        Kokkos::TeamPolicy<>(
            execution_space, num_nodes, Kokkos::AUTO, Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const std::size_t node = static_cast<std::size_t>(
                team_member.league_rank());
            // compute scale factor
            double A0_scale_factor;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team_member, num_neigh(node)),
                [=] (const int j, double& lsum) {
                    const int edge = first_neigh(node)+j;
                    if (streamed_recompute) {
                        const int type_i = type_to_active(node_types(node));
                        const int type_j = type_to_active(neigh_types(edge));
                        const int edge_type = type_i <= type_j
                            ? type_i*(2*active_type_count-type_i-1)/2+type_j
                            : type_j*(2*active_type_count-type_j-1)/2+type_i;
                        lsum += A0_splines.evaluate_function(
                            edge_type, r(edge), 0);
                    } else
                        lsum += A0_spline_values(edge,0);
                }, A0_scale_factor);
            A0_scale_factor += 1.0;
            team_member.team_barrier();
            // perform the scaling
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, num_lm*num_channels),
                [=] (int lmk) {
                    A0(node,lmk/num_channels,lmk%num_channels) /= A0_scale_factor;
                });
        });
    if (!streamed_recompute)
        Kokkos::fence();
}

template <typename Precision>
void MACEKokkos<Precision>::reverse_A0_scaled(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r)
{
    if (not A0_scaled) return;
    if (standard_r0_module_active) {
        if (selected_r0_implementation == R0Implementation::device_module) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
            const auto environment = execution_device_execution_environment();
            const int persistent_blocks = std::max(
                1, r0_device_persistent_blocks_per_compute_unit
                    *environment.compute_unit_count);
            const SymmetrixJitR0SplineV1 empty_spline{
                sizeof(SymmetrixJitR0SplineV1), 0u, 0u, 0u,
                0.0, 0.0, nullptr};
            const bool precomputed = use_mh0_adjoint_reuse();
            const SymmetrixJitR0ArgsV1 args{
                sizeof(SymmetrixJitR0ArgsV1), 0u, num_nodes, 0,
                num_active_types, num_channels, l_max, 0, 1,
                precomputed ? 1 : 0, node_types.data(), num_neigh.data(),
                streamed_first_neigh.data(), neigh_types.data(), nullptr,
                type_to_active.data(), nullptr, r.data(), empty_spline,
                empty_spline, nullptr, nullptr, A0.data(), A0_adj.data(),
                standard_r0_density_state.data(),
                mh0_a0_scale_adjoint.data(), nullptr, r_cut};
            ExecutionDeviceBackend::DeviceGuard device_guard(
                ExecutionDeviceBackend::device_ordinal(
                    factorized_execution_space));
            r0_device_module->launch_r0_reverse_prepare(
                args,
                reinterpret_cast<void*>(ExecutionDeviceBackend::native_stream(
                    factorized_execution_space)),
                persistent_blocks);
            standard_r0_module_launch_count += 1;
            return;
#else
            throw std::runtime_error(
                "Execution R0 device modules require a device backend.");
#endif
        }
        symmetrix::standard_r0::launch_reverse_prepare(
            factorized_execution_space,
            num_nodes,
            A0,
            A0_adj,
            standard_r0_density_state,
            mh0_a0_scale_adjoint,
            use_mh0_adjoint_reuse());
        standard_r0_module_launch_count += 1;
        return;
    }
    if (standard_r0_density_scale_fused) return;
    const bool streamed_recompute =
        mace_uses_prepared_execution(streamed_edges);
    const auto execution_space = use_factorized_async_inference()
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();

    Kokkos::View<int*> first_neigh;
    if (streamed_recompute)
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
    const auto A0 = this->A0;
    const auto A0_adj = this->A0_adj;
    const auto A0_spline_values = this->A0_spline_values;
    const auto A0_spline_derivs = this->A0_spline_derivs;
    const auto A0_splines = this->A0_splines;
    const auto type_to_active = this->type_to_active;
    const auto active_type_count = num_active_types;
    const auto num_channels = this->num_channels;
    const auto num_lm = this->num_lm;
    auto node_forces = this->node_forces;
    Kokkos::parallel_for(
        "MACEKokkos::reverse_A0_scaled",
        Kokkos::TeamPolicy<>(
            execution_space, num_nodes, Kokkos::AUTO, Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const std::size_t node = static_cast<std::size_t>(
                team_member.league_rank());
            // compute scale factor
            double A0_scale_factor;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team_member, num_neigh(node)),
                [=] (const int j, double& lsum) {
                    const int edge = first_neigh(node)+j;
                    if (streamed_recompute) {
                        const int type_i = type_to_active(node_types(node));
                        const int type_j = type_to_active(neigh_types(edge));
                        const int edge_type = type_i <= type_j
                            ? type_i*(2*active_type_count-type_i-1)/2+type_j
                            : type_j*(2*active_type_count-type_j-1)/2+type_i;
                        lsum += A0_splines.evaluate_function(
                            edge_type, r(edge), 0);
                    } else
                        lsum += A0_spline_values(edge,0);
                }, A0_scale_factor);
            A0_scale_factor += 1.0;
            team_member.team_barrier();
            // update dE/dxyz
            double dA0_dot_A0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team_member, num_lm*num_channels),
                [=] (const int lmk, double& lsum) {
                    const int lm = lmk / num_channels;
                    const int k = lmk % num_channels;
                    lsum += A0_adj(node,lm,k) * A0(node,lm,k);
                }, dA0_dot_A0);
            team_member.team_barrier();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, num_neigh(node)),
                [=] (const int j) {
                    const int ij = first_neigh(node) + j;
                    const std::size_t edge = static_cast<std::size_t>(ij);
                    double f;
                    double d;
                    if (streamed_recompute) {
                        const int type_i = type_to_active(node_types(node));
                        const int type_j = type_to_active(neigh_types(ij));
                        const int edge_type = type_i <= type_j
                            ? type_i*(2*active_type_count-type_i-1)/2+type_j
                            : type_j*(2*active_type_count-type_j-1)/2+type_i;
                        A0_splines.evaluate_function(
                            edge_type, r(ij), 0, f, d);
                    } else {
                        f = A0_spline_values(ij,0);
                        d = A0_spline_derivs(ij,0);
                    }
                    node_forces(3*edge+0) += dA0_dot_A0/A0_scale_factor*d*xyz(3*edge+0)/r(ij);
                    node_forces(3*edge+1) += dA0_dot_A0/A0_scale_factor*d*xyz(3*edge+1)/r(ij);
                    node_forces(3*edge+2) += dA0_dot_A0/A0_scale_factor*d*xyz(3*edge+2)/r(ij);
                });
            // update dE/dA0
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, num_lm*num_channels),
                [=] (int lmk) {
                    A0_adj(node,lmk/num_channels,lmk%num_channels) /= A0_scale_factor;
                });
        });
    if (!streamed_recompute)
        Kokkos::fence();
}

template <typename Precision>
void MACEKokkos<Precision>::compute_M0_module(
    const int num_nodes,
    Kokkos::View<const int*> node_types)
{
    const auto execution_space = use_factorized_async_inference()
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    ensure_mh0_m0_forward_capacity(num_nodes);
    const int persistent_blocks =
        selected_m0_implementation == M0Implementation::builtin
        ? resolve_execution_persistent_blocks(
            standard_m0_module_id(), execution_space,
            static_cast<std::size_t>(num_nodes)*num_channels,
            "m0_forward")
        : std::max(
            1, m0_device_persistent_blocks_per_compute_unit
                *execution_device_execution_environment().compute_unit_count);
    if (selected_m0_implementation == M0Implementation::device_module) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
        if (!m0_device_module_ready())
            throw std::runtime_error(
                "Execution M0 device-module forward is unavailable.");
        ExecutionDeviceBackend::DeviceGuard device_guard(
            ExecutionDeviceBackend::device_ordinal(execution_space));
        const SymmetrixJitM0ArgsV1 args{
            sizeof(SymmetrixJitM0ArgsV1), 0u, num_nodes, num_channels, 0,
            node_types.data(), A0.data(), standard_m0_module_weights.data(),
            nullptr, M0.data(), nullptr, nullptr};
        m0_device_module->launch_m0_forward(
            args,
            reinterpret_cast<void*>(
                ExecutionDeviceBackend::native_stream(execution_space)),
            persistent_blocks);
        standard_m0_module_forward_launch_count += 1;
        return;
#else
        throw std::runtime_error(
            "Execution M0 device modules require a device backend.");
#endif
    }
    if (selected_m0_implementation == M0Implementation::host_plugin) {
        if constexpr (symmetrix::standard_r0::host_execution_space<
                Kokkos::DefaultExecutionSpace>) {
            if (!m0_host_plugin_ready())
                throw std::runtime_error(
                    "Execution M0 host-plugin forward is unavailable.");
            const SymmetrixJitM0HostArgsV1 args{
                sizeof(SymmetrixJitM0HostArgsV1), 0u, num_nodes, num_channels, 0,
                node_types.data(), A0.data(), standard_m0_module_weights.data(),
                nullptr, M0.data(), nullptr, nullptr};
            const auto& descriptor = m0_host_plugin->descriptor();
            const int owner_channel_tile = descriptor.capabilities
                &SYMMETRIX_JIT_M0_HOST_CHANNEL_TILED_OWNER_V1
                ? static_cast<int>(descriptor.owner_channel_tile) : 1;
            const std::int64_t owner_channel_tiles =
                (static_cast<std::int64_t>(num_channels)+owner_channel_tile-1)
                /owner_channel_tile;
            const auto owner = descriptor.forward_owner;
            Kokkos::parallel_for(
                "ExecutionM0::host_plugin_forward",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::int64_t>>(
                    execution_space, 0,
                    static_cast<std::int64_t>(num_nodes)*owner_channel_tiles),
                [=] (const std::int64_t index) { owner(&args, index); });
            standard_m0_module_forward_launch_count += 1;
            return;
        } else {
            throw std::runtime_error(
                "Execution M0 host plugins require host execution.");
        }
    }
    const bool launched = persistent_blocks > 0
        && (standard_m0_module_variant
                == StandardM0ModuleVariant::scalar_lmax0
            ? symmetrix::standard_m0::launch_scalar_forward(
                execution_space, persistent_blocks, num_nodes,
                node_types, A0, standard_m0_module_weights, M0)
            : symmetrix::standard_m0::launch_forward(
                execution_space, persistent_blocks, num_nodes,
                node_types, A0, standard_m0_module_weights, M0));
    if (launched) {
        standard_m0_module_forward_launch_count += 1;
        return;
    }
    throw std::runtime_error(
        "Execution M0 standard-module forward failed to launch.");
}

template <typename Precision>
void MACEKokkos<Precision>::launch_M0_module_reverse(
    const Kokkos::DefaultExecutionSpace& execution_space,
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const Precision***,Kokkos::LayoutRight> input,
    Kokkos::View<const Precision***,Kokkos::LayoutRight> output_adjoint,
    Kokkos::View<Precision***,Kokkos::LayoutRight> input_adjoint,
    Kokkos::View<double*> input_scale_adjoint,
    const bool capture_input_scale_adjoint)
{
    const int persistent_blocks =
        selected_m0_implementation == M0Implementation::builtin
        ? resolve_execution_persistent_blocks(
            standard_m0_module_id(), execution_space,
            static_cast<std::size_t>(num_nodes)*num_channels,
            "m0_reverse")
        : std::max(
            1, m0_device_persistent_blocks_per_compute_unit
                *execution_device_execution_environment().compute_unit_count);
    if (selected_m0_implementation == M0Implementation::device_module) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
        if (!m0_device_module_ready())
            throw std::runtime_error(
                "Execution M0 device-module reverse is unavailable.");
        ExecutionDeviceBackend::DeviceGuard device_guard(
            ExecutionDeviceBackend::device_ordinal(execution_space));
        const SymmetrixJitM0ArgsV1 args{
            sizeof(SymmetrixJitM0ArgsV1), 0u, num_nodes, num_channels,
            capture_input_scale_adjoint ? 1 : 0,
            node_types.data(), input.data(), standard_m0_module_weights.data(),
            output_adjoint.data(), nullptr, input_adjoint.data(),
            input_scale_adjoint.data()};
        m0_device_module->launch_m0_reverse(
            args,
            reinterpret_cast<void*>(
                ExecutionDeviceBackend::native_stream(execution_space)),
            persistent_blocks);
#else
        throw std::runtime_error(
            "Execution M0 device modules require a device backend.");
#endif
    } else if (selected_m0_implementation == M0Implementation::host_plugin) {
        if constexpr (symmetrix::standard_r0::host_execution_space<
                Kokkos::DefaultExecutionSpace>) {
            if (!m0_host_plugin_ready())
                throw std::runtime_error(
                    "Execution M0 host-plugin reverse is unavailable.");
            const SymmetrixJitM0HostArgsV1 args{
                sizeof(SymmetrixJitM0HostArgsV1), 0u, num_nodes, num_channels,
                capture_input_scale_adjoint ? 1 : 0,
                node_types.data(), input.data(), standard_m0_module_weights.data(),
                output_adjoint.data(), nullptr, input_adjoint.data(),
                input_scale_adjoint.data()};
            const auto& descriptor = m0_host_plugin->descriptor();
            const int owner_channel_tile = descriptor.capabilities
                &SYMMETRIX_JIT_M0_HOST_CHANNEL_TILED_OWNER_V1
                ? static_cast<int>(descriptor.owner_channel_tile) : 1;
            const std::int64_t owner_channel_tiles =
                (static_cast<std::int64_t>(num_channels)+owner_channel_tile-1)
                /owner_channel_tile;
            const auto owner = descriptor.reverse_owner;
            Kokkos::parallel_for(
                "ExecutionM0::host_plugin_reverse",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::int64_t>>(
                    execution_space, 0,
                    static_cast<std::int64_t>(num_nodes)*owner_channel_tiles),
                [=] (const std::int64_t index) { owner(&args, index); });
        } else {
            throw std::runtime_error(
                "Execution M0 host plugins require host execution.");
        }
    } else {
        const bool launched = persistent_blocks > 0
            && (standard_m0_module_variant
                    == StandardM0ModuleVariant::scalar_lmax0
                ? symmetrix::standard_m0::launch_scalar_reverse(
                    execution_space, persistent_blocks, num_nodes,
                    node_types, input, standard_m0_module_weights,
                    output_adjoint, input_adjoint, input_scale_adjoint,
                    capture_input_scale_adjoint)
                : symmetrix::standard_m0::launch_reverse(
                    execution_space, persistent_blocks, num_nodes,
                    node_types, input, standard_m0_module_weights,
                    output_adjoint, input_adjoint, input_scale_adjoint,
                    capture_input_scale_adjoint));
        if (!launched)
            throw std::runtime_error(
                "Execution M0 built-in reverse failed to launch.");
    }
    standard_m0_module_reverse_launch_count += 1;
    if (capture_input_scale_adjoint)
        standard_m0_input_scale_adjoint_launch_count += 1;
}

template <typename Precision>
void MACEKokkos<Precision>::reverse_M0_module(
    const int num_nodes,
    Kokkos::View<const int*> node_types)
{
    const auto execution_space = use_factorized_async_inference()
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    const bool reuse_adjoint = use_mh0_adjoint_reuse();
    const bool capture_input_scale_adjoint = reuse_adjoint && A0_scaled;
    if (reuse_adjoint) {
        A0_adj = A0;
        if (A0_scaled && mh0_a0_scale_adjoint.extent(0)
                < static_cast<std::size_t>(num_nodes))
            Kokkos::realloc(mh0_a0_scale_adjoint, num_nodes);
        if (A0_scaled)
            Kokkos::deep_copy(
                execution_space, mh0_a0_scale_adjoint, double(0));
    }
    if (A0_adj.extent(0) < num_nodes)
        Kokkos::realloc(A0_adj, A0.extent(0), A0.extent(1), A0.extent(2));
    launch_M0_module_reverse(
        execution_space, num_nodes, node_types, A0, M0_adj, A0_adj,
        mh0_a0_scale_adjoint, capture_input_scale_adjoint);
}

template <typename Precision>
void MACEKokkos<Precision>::compute_M0(int num_nodes, Kokkos::View<const int*> node_types)
{
    const auto execution_space = use_factorized_async_inference()
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    ensure_mh0_m0_forward_capacity(num_nodes);
    ensure_m0_polynomial_workspace();
    if (use_factorized_async_inference())
        Kokkos::deep_copy(execution_space, M0, Precision(0));
    else
        Kokkos::deep_copy(M0, Precision(0));
    for (int LM=0; LM<num_LM; ++LM) {
        if (M0_poly_values(LM).extent(0) < num_nodes)
            M0_poly_values(LM) = Kokkos::View<Precision***,Kokkos::LayoutRight>(
                Kokkos::view_alloc(std::string("M0_poly_values_")+std::to_string(LM),Kokkos::WithoutInitializing),
                num_nodes, M0_poly_coeff(LM).extent(1), num_channels);
    }

    const auto A0 = this->A0;
    const auto M0_monomials = this->M0_monomials;
    const auto M0_weights = this->M0_weights;
    const auto M0_poly_spec = this->M0_poly_spec;
    const auto M0_poly_coeff = this->M0_poly_coeff;
    const auto M0_poly_values = this->M0_poly_values;
    const auto num_channels = this->num_channels;
    const auto num_lm = this->num_lm;
    const auto num_LM = this->num_LM;
    auto M0 = this->M0;

    Kokkos::parallel_for("Compute M0",
        Kokkos::TeamPolicy<>(
            execution_space, num_nodes*num_LM, Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int work = team_member.league_rank();
            const std::size_t node = static_cast<std::size_t>(work/num_LM);
            const int LM = work%num_LM;
            // initialize
            Kokkos::parallel_for(
                Kokkos::TeamVectorMDRange<Kokkos::Rank<2,Kokkos::Iterate::Right>,Kokkos::TeamPolicy<>::member_type>(
                    team_member, num_lm, num_channels),
                [&] (const int lm, const int k) {
                    M0_poly_values(LM)(node,lm,k) = A0(node,lm,k);
                    Kokkos::atomic_add(&M0(node,LM,k), M0_poly_coeff(LM)(node_types(node),lm,k) * M0_poly_values(LM)(node,lm,k));
                });
            team_member.team_barrier();
            // forward pass
            for (int p=0; p<M0_poly_spec(LM).extent(0); ++p) {
                const int p0 = M0_poly_spec(LM)(p,0);
                const int p1 = M0_poly_spec(LM)(p,1);
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team_member, num_channels),
                    [&] (const int k) {
                        M0_poly_values(LM)(node,num_lm+p,k) = M0_poly_values(LM)(node,p0,k) * M0_poly_values(LM)(node,p1,k);
                        M0(node,LM,k) += M0_poly_coeff(LM)(node_types(node),num_lm+p,k) * M0_poly_values(LM)(node,num_lm+p,k);
                    });
            }
        });
    complete_device_stage("MACEKokkos::compute_M0");
}
template <typename Precision>
void MACEKokkos<Precision>::reverse_M0(int num_nodes, Kokkos::View<const int*> node_types)
{
    const auto execution_space = use_factorized_async_inference()
        ? factorized_execution_space : Kokkos::DefaultExecutionSpace();
    if (A0_adj.extent(0) < num_nodes)
        Kokkos::realloc(A0_adj, A0.extent(0), A0.extent(1), A0.extent(2));
    ensure_m0_polynomial_workspace();
    if (use_factorized_async_inference())
        Kokkos::deep_copy(execution_space, A0_adj, Precision(0));
    else
        Kokkos::deep_copy(A0_adj, Precision(0));
    for (int LM=0; LM<num_LM; ++LM) {
        if (M0_poly_adjoints(LM).extent(0) < num_nodes)
            M0_poly_adjoints(LM) = Kokkos::View<Precision***,Kokkos::LayoutRight>(
                Kokkos::view_alloc(std::string("M0_poly_adjoints_")+std::to_string(LM),Kokkos::WithoutInitializing),
                num_nodes, M0_poly_coeff(LM).extent(1), num_channels);
    }

    // TODO: prune
    const auto M0_adj = this->M0_adj;
    const auto M0_poly_spec = this->M0_poly_spec;
    const auto M0_poly_coeff = this->M0_poly_coeff;
    const auto M0_poly_adjoints = this->M0_poly_adjoints;
    const auto M0_poly_values = this->M0_poly_values;
    const auto num_channels = this->num_channels;
    const auto num_lm = this->num_lm;
    const auto num_LM = this->num_LM;
    auto A0_adj = this->A0_adj;

    Kokkos::parallel_for("Reverse M0",
        Kokkos::TeamPolicy<>(
            execution_space, num_nodes*num_LM, Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int work = team_member.league_rank();
            const std::size_t node = static_cast<std::size_t>(work/num_LM);
            const int LM = work%num_LM;
            // initialize
            Kokkos::parallel_for(
                Kokkos::TeamVectorMDRange<
                    Kokkos::Rank<2,Kokkos::Iterate::Right>,Kokkos::TeamPolicy<>::member_type>(
                        team_member, M0_poly_coeff(LM).extent(1), num_channels),
                [&] (const int p, const int k) {
                    M0_poly_adjoints(LM)(node,p,k) = M0_poly_coeff(LM)(node_types(node),p,k);
                });
            team_member.team_barrier();
            // backwards pass
            for (int p=M0_poly_spec(LM).extent(0)-1; p>=0; --p) {
                const int p0 = M0_poly_spec(LM)(p,0);
                const int p1 = M0_poly_spec(LM)(p,1);
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team_member, num_channels),
                    [&] (const int k) {
                        // TODO: use scratch space
                        M0_poly_adjoints(LM)(node,p0,k) += M0_poly_adjoints(LM)(node,num_lm+p,k)*M0_poly_values(LM)(node,p1,k);
                        M0_poly_adjoints(LM)(node,p1,k) += M0_poly_adjoints(LM)(node,num_lm+p,k)*M0_poly_values(LM)(node,p0,k);
                    });
            }
            team_member.team_barrier();
        });
    Kokkos::parallel_for(
        "Reverse M0 deterministic gather",
        Kokkos::MDRangePolicy<Kokkos::Rank<3,Kokkos::Iterate::Right>>(
            execution_space, {0,0,0}, {num_nodes,num_lm,num_channels}),
        KOKKOS_LAMBDA (const int i, const int lm, const int k) {
            const std::size_t node = static_cast<std::size_t>(i);
            Precision adjoint = 0;
            for (int LM=0; LM<num_LM; ++LM)
                adjoint += M0_poly_adjoints(LM)(node,lm,k)*M0_adj(node,LM,k);
            A0_adj(node,lm,k) = adjoint;
        });
    complete_device_stage("MACEKokkos::reverse_M0");
}
template class MACEKokkos<float>;
template class MACEKokkos<double>;
