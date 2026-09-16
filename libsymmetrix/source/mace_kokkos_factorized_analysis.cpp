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
namespace {

#ifdef KOKKOS_ENABLE_CUDA
void check_factorized_benchmark_cuda(
    const cudaError_t status,
    const char* operation)
{
    if (status != cudaSuccess)
        throw std::runtime_error(
            std::string(operation)+": "+cudaGetErrorString(status));
}

class FactorizedBenchmarkCudaEvent {
public:
    FactorizedBenchmarkCudaEvent()
    {
        check_factorized_benchmark_cuda(
            cudaEventCreate(&event_), "Creating Execution R1 benchmark CUDA event");
    }

    ~FactorizedBenchmarkCudaEvent()
    {
        if (event_ != nullptr)
            cudaEventDestroy(event_);
    }

    FactorizedBenchmarkCudaEvent(const FactorizedBenchmarkCudaEvent&) = delete;
    FactorizedBenchmarkCudaEvent& operator=(
        const FactorizedBenchmarkCudaEvent&) = delete;

    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};
#endif

}

template <typename Precision>
typename MACEKokkos<Precision>::FactorizedOperatorBenchmarkMode
MACEKokkos<Precision>::factorized_operator_benchmark_mode(
    const std::string& mode) const
{
    if (mode == "forward")
        return FactorizedOperatorBenchmarkMode::forward;
    if (mode == "reverse")
        return FactorizedOperatorBenchmarkMode::reverse;
    if (mode == "forward_reverse")
        return FactorizedOperatorBenchmarkMode::forward_reverse;
    throw std::invalid_argument(
        "Execution R1 operator benchmark mode must be 'forward', 'reverse', "
        "or 'forward_reverse'.");
}

template <typename Precision>
void MACEKokkos<Precision>::validate_factorized_operator_benchmark(
    const std::uint64_t token) const
{
    if (!factorized_operator_benchmark_ready
        || token == 0
        || token != factorized_operator_benchmark_token)
        throw std::invalid_argument(
            "Execution R1 operator benchmark token is stale; prepare it again.");
    if (factorized_operator_benchmark_graph_generation
            != factorized_prepared_graph_generation
        || factorized_schedule_dirty)
        throw std::invalid_argument(
            "Execution R1 operator benchmark graph is stale; prepare it again.");
    if (factorized_operator_benchmark_evaluation_epoch == 0
        || factorized_operator_benchmark_evaluation_epoch
            != factorized_completed_evaluation_epoch
        || factorized_completed_evaluation_graph_generation
            != factorized_prepared_graph_generation)
        throw std::invalid_argument(
            "Execution R1 operator benchmark evaluation state is stale; "
            "prepare it again after a successful prepared evaluation.");
    if (!use_factorized_direct_jit_forward()
        || !use_factorized_direct_jit_reverse())
        throw std::logic_error(
            "Execution R1 operator benchmark requires the admitted generated "
            "fixed-weight forward and coordinate reverse executors.");
}

template <typename Precision>
std::uint64_t MACEKokkos<Precision>::prepare_factorized_operator_benchmark(
    const std::uint64_t graph_generation,
    const std::span<const double> xyz,
    const std::span<const double> r)
{
    if constexpr (!std::is_same_v<Precision,float>)
        throw std::logic_error(
            "Execution R1 operator benchmark requires float32 model execution.");
    if (!mace_uses_direct_execution(streamed_edges))
        throw std::logic_error(
            "Execution R1 operator benchmark requires streamed_edges='direct'.");
    if (graph_generation == 0
        || graph_generation != factorized_prepared_graph_generation
        || factorized_schedule_dirty)
        throw std::invalid_argument(
            "Execution R1 operator benchmark requires a current prepared graph token.");
    if (factorized_completed_evaluation_epoch == 0
        || factorized_completed_evaluation_graph_generation != graph_generation)
        throw std::logic_error(
            "Execution R1 operator benchmark requires a successfully completed "
            "prepared evaluation for the current graph.");
    if (!use_factorized_direct_jit_forward()
        || !use_factorized_direct_jit_reverse())
        throw std::logic_error(
            "Execution R1 operator benchmark requires the admitted generated "
            "fixed-weight forward and coordinate reverse executors.");

    const int num_nodes = static_cast<int>(execution_prepared_node_types.extent(0));
    const int num_edges =
        static_cast<int>(execution_prepared_neigh_indices.extent(0));
    if (xyz.size() != static_cast<std::size_t>(3)*num_edges
        || r.size() != static_cast<std::size_t>(num_edges))
        throw std::invalid_argument(
            "Execution R1 operator benchmark coordinate extents do not match "
            "the prepared graph.");
    if (num_channels != 128 || factorized_embedding_width != 64
        || l_max != 3 || num_lm != 16 || num_LM != 4
        || execution_group_path_offsets_host
            != std::vector<int>({0, 2, 5, 8, 10}))
        throw std::logic_error(
            "Execution R1 operator benchmark requires the extracted OMAT "
            "128-channel, 64-embedding, ten-path contract.");
    if (H1.extent(0) < static_cast<std::size_t>(num_nodes)
        || H1.extent(1) != static_cast<std::size_t>(num_LM)
        || H1.extent(2) != static_cast<std::size_t>(num_channels)
        || A1_adj.extent(0) < static_cast<std::size_t>(num_nodes)
        || A1_adj.extent(1) != static_cast<std::size_t>(num_lm)
        || A1_adj.extent(2) != static_cast<std::size_t>(num_channels)
        || Y.extent(0) < static_cast<std::size_t>(num_edges)*num_lm
        || Y_grad.extent(0)
            < static_cast<std::size_t>(3)*num_edges*num_lm)
        throw std::logic_error(
            "Execution R1 operator benchmark requires prepared production "
            "H1, Y, Y_grad, and A1 cotangent tensors.");

    if (factorized_operator_benchmark_xyz.extent(0) != xyz.size())
        Kokkos::realloc(factorized_operator_benchmark_xyz, xyz.size());
    if (factorized_operator_benchmark_r.extent(0) != r.size())
        Kokkos::realloc(factorized_operator_benchmark_r, r.size());
    auto h_xyz = Kokkos::create_mirror_view(
        factorized_operator_benchmark_xyz);
    auto h_r = Kokkos::create_mirror_view(factorized_operator_benchmark_r);
    std::copy(xyz.begin(), xyz.end(), h_xyz.data());
    std::copy(r.begin(), r.end(), h_r.data());
    Kokkos::deep_copy(
        factorized_execution_space,
        factorized_operator_benchmark_xyz, h_xyz);
    Kokkos::deep_copy(
        factorized_execution_space,
        factorized_operator_benchmark_r, h_r);

    if (factorized_operator_benchmark_a1_adjoint.extent(0)
            != static_cast<std::size_t>(num_nodes)
        || factorized_operator_benchmark_a1_adjoint.extent(1)
            != static_cast<std::size_t>(num_lm)
        || factorized_operator_benchmark_a1_adjoint.extent(2)
            != static_cast<std::size_t>(num_channels))
        Kokkos::realloc(
            factorized_operator_benchmark_a1_adjoint,
            num_nodes, num_lm, num_channels);
    Kokkos::deep_copy(
        factorized_execution_space,
        factorized_operator_benchmark_a1_adjoint,
        Kokkos::subview(
            A1_adj,
            Kokkos::make_pair(0, num_nodes), Kokkos::ALL, Kokkos::ALL));

    if (Phi1.extent(0) != static_cast<std::size_t>(num_nodes))
        Kokkos::realloc(Phi1, num_nodes, num_lme, num_channels);
    if (A1.extent(0) != static_cast<std::size_t>(num_nodes))
        Kokkos::realloc(A1, num_nodes, num_lm, num_channels);
    if (dPhi1.extent(0) != static_cast<std::size_t>(num_nodes))
        Kokkos::realloc(dPhi1, num_nodes, num_lme, num_channels);
    if (H1_adj.extent(0) != static_cast<std::size_t>(num_nodes))
        Kokkos::realloc(H1_adj, num_nodes, num_LM, num_channels);
    if (node_forces.extent(0) != xyz.size())
        Kokkos::realloc(node_forces, xyz.size());
    if (factorized_operator_benchmark_phi.extent(0)
            != static_cast<std::size_t>(num_edges)
        || factorized_operator_benchmark_phi.extent(1)
            != static_cast<std::size_t>(factorized_embedding_width)) {
        Kokkos::realloc(
            factorized_operator_benchmark_phi,
            num_edges, factorized_embedding_width);
        Kokkos::realloc(
            factorized_operator_benchmark_dphi_dr,
            num_edges, factorized_embedding_width);
    }

    const int active_type_count = num_active_types;
    const int embedding = factorized_embedding_width;
    const auto node_types = execution_prepared_node_types;
    const auto neigh_types = execution_prepared_neigh_types;
    const auto edge_receivers = execution_edge_receivers;
    const auto type_map = type_to_active;
    const auto radial = execution_radial_1;
    const auto radius = factorized_operator_benchmark_r;
    auto phi = factorized_operator_benchmark_phi;
    auto dphi_dr = factorized_operator_benchmark_dphi_dr;
    Kokkos::parallel_for(
        "FactorizedOperatorBenchmark::compact_radial",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            factorized_execution_space, 0, num_edges),
        KOKKOS_LAMBDA (const int edge) {
            const int receiver_type =
                type_map(node_types(edge_receivers(edge)));
            const int source_type = type_map(neigh_types(edge));
            const int edge_type = receiver_type <= source_type
                ? receiver_type
                    *(2*active_type_count-receiver_type-1)/2+source_type
                : source_type
                    *(2*active_type_count-source_type-1)/2+receiver_type;
            const auto point = radial.evaluation_point(radius(edge));
            for (int q=0; q<embedding; ++q)
                radial.evaluate_function(
                    edge_type, point, q, phi(edge,q), dphi_dr(edge,q));
        });

    factorized_operator_benchmark_token_counter += 1;
    if (factorized_operator_benchmark_token_counter == 0)
        factorized_operator_benchmark_token_counter += 1;
    factorized_operator_benchmark_token =
        factorized_operator_benchmark_token_counter;
    factorized_operator_benchmark_graph_generation = graph_generation;
    factorized_operator_benchmark_evaluation_epoch =
        factorized_completed_evaluation_epoch;
    factorized_operator_benchmark_num_nodes = num_nodes;
    factorized_operator_benchmark_num_edges = num_edges;
    factorized_operator_benchmark_ready = true;
    try {
        run_factorized_operator_benchmark(
            FactorizedOperatorBenchmarkMode::forward_reverse);
        factorized_execution_space.fence(
            "Execution R1 operator benchmark preparation");
    } catch (...) {
        invalidate_factorized_operator_benchmark();
        throw;
    }
    return factorized_operator_benchmark_token;
}

template <typename Precision>
void MACEKokkos<Precision>::run_factorized_operator_benchmark(
    const std::uint64_t token,
    const std::string& mode)
{
    validate_factorized_operator_benchmark(token);
    run_factorized_operator_benchmark(factorized_operator_benchmark_mode(mode));
}

template <typename Precision>
void MACEKokkos<Precision>::run_factorized_operator_benchmark(
    const FactorizedOperatorBenchmarkMode mode)
{
    const int num_nodes = factorized_operator_benchmark_num_nodes;
    const auto run_forward = [&] {
        compute_Phi1_streamed_jit(
            num_nodes,
            execution_prepared_node_types,
            execution_prepared_num_neigh,
            execution_prepared_neigh_indices,
            execution_prepared_neigh_types,
            factorized_operator_benchmark_r);
        compute_A1(num_nodes, false);
    };
    const auto run_reverse = [&] {
        Kokkos::deep_copy(factorized_execution_space, H1_adj, Precision(0));
        Kokkos::deep_copy(factorized_execution_space, node_forces, 0.0);
        reverse_A1_from(
            num_nodes,
            factorized_operator_benchmark_a1_adjoint,
            false);
        reverse_factorized_direct(
            num_nodes,
            execution_prepared_node_types,
            execution_prepared_neigh_indices,
            execution_prepared_neigh_types,
            factorized_operator_benchmark_xyz,
            factorized_operator_benchmark_r);
    };
    switch (mode) {
    case FactorizedOperatorBenchmarkMode::forward:
        run_forward();
        break;
    case FactorizedOperatorBenchmarkMode::reverse:
        run_reverse();
        break;
    case FactorizedOperatorBenchmarkMode::forward_reverse:
        run_forward();
        run_reverse();
        break;
    }
}

template <typename Precision>
void MACEKokkos<Precision>::synchronize_factorized_operator_benchmark(
    const std::uint64_t token)
{
    validate_factorized_operator_benchmark(token);
    factorized_execution_space.fence(
        "Execution R1 operator benchmark synchronization");
}

template <typename Precision>
FactorizedOperatorBenchmarkMeasurement
MACEKokkos<Precision>::measure_factorized_operator_benchmark(
    const std::uint64_t token,
    const std::string& mode,
    const std::size_t warmup_iterations,
    const double warmup_ms,
    const std::size_t min_samples,
    const std::size_t max_samples,
    const double min_sample_ms)
{
    validate_factorized_operator_benchmark(token);
    if (!std::isfinite(warmup_ms) || warmup_ms < 0.0
        || !std::isfinite(min_sample_ms) || min_sample_ms < 0.0
        || min_samples == 0 || max_samples < min_samples)
        throw std::invalid_argument(
            "Execution R1 operator benchmark measurement configuration is invalid.");
    const auto parsed_mode = factorized_operator_benchmark_mode(mode);
    FactorizedOperatorBenchmarkMeasurement result;
    result.samples_ms.reserve(max_samples);
    const auto cold_start = std::chrono::steady_clock::now();
    run_factorized_operator_benchmark(parsed_mode);
    factorized_execution_space.fence(
        "Execution R1 operator benchmark cold synchronization");
    result.cold_ms = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-cold_start).count();

#ifdef KOKKOS_ENABLE_CUDA
    FactorizedBenchmarkCudaEvent start_event;
    FactorizedBenchmarkCudaEvent end_event;
    const auto stream = factorized_execution_space.cuda_stream();
    const auto measure_once = [&] {
        check_factorized_benchmark_cuda(
            cudaEventRecord(start_event.get(), stream),
            "Recording Execution R1 benchmark start event");
        run_factorized_operator_benchmark(parsed_mode);
        check_factorized_benchmark_cuda(
            cudaEventRecord(end_event.get(), stream),
            "Recording Execution R1 benchmark end event");
        check_factorized_benchmark_cuda(
            cudaEventSynchronize(end_event.get()),
            "Synchronizing Execution R1 benchmark end event");
        float elapsed_ms = 0.0f;
        check_factorized_benchmark_cuda(
            cudaEventElapsedTime(
                &elapsed_ms, start_event.get(), end_event.get()),
            "Reading Execution R1 benchmark CUDA events");
        return static_cast<double>(elapsed_ms);
    };
#else
    const auto measure_once = [&] {
        const auto start = std::chrono::steady_clock::now();
        run_factorized_operator_benchmark(parsed_mode);
        factorized_execution_space.fence(
            "Execution R1 operator benchmark host sample");
        return std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-start).count();
    };
#endif

    const auto warmup_start = std::chrono::steady_clock::now();
    while (result.warmup_count < warmup_iterations
           || std::chrono::duration<double,std::milli>(
                  std::chrono::steady_clock::now()-warmup_start).count()
                < warmup_ms) {
        run_factorized_operator_benchmark(parsed_mode);
        result.warmup_count += 1;
    }
    factorized_execution_space.fence("Execution R1 operator benchmark warmup");

    double measured_ms = 0.0;
    while (result.samples_ms.size() < max_samples
           && (result.samples_ms.size() < min_samples
               || measured_ms < min_sample_ms)) {
        const double sample_ms = measure_once();
        result.samples_ms.push_back(sample_ms);
        measured_ms += sample_ms;
    }
    return result;
}

template <typename Precision>
std::vector<Precision>
MACEKokkos<Precision>::factorized_operator_benchmark_radial_linear(
    const std::uint64_t token) const
{
    validate_factorized_operator_benchmark(token);
    constexpr int path_count = 10;
    const int embedding = factorized_embedding_width;
    const int channels = num_channels;
    auto result = std::vector<Precision>(
        static_cast<std::size_t>(embedding)*path_count*channels*channels);
    const auto h_group_paths = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), execution_group_paths);
    for (int l=0; l<=l_max; ++l) {
        const int group_begin = execution_group_path_offsets_host[l];
        const int num_eta =
            execution_group_path_offsets_host[l+1]-group_begin;
        const auto projection = execution_projection(l);
        if (projection.extent(0)
                != static_cast<std::size_t>(num_eta)*embedding*channels
            || projection.extent(1) != static_cast<std::size_t>(channels))
            throw std::logic_error(
                "Execution R1 composed radial projection has an invalid extent.");
        const auto h_projection = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), projection);
        for (int eta=0; eta<num_eta; ++eta) {
            const int path = h_group_paths(group_begin+eta);
            if (path < 0 || path >= path_count)
                throw std::logic_error(
                    "Execution R1 composed radial projection path is invalid.");
            for (int q=0; q<embedding; ++q)
                for (int input_channel=0;
                     input_channel<channels; ++input_channel) {
                    const int source_row =
                        (eta*embedding+q)*channels+input_channel;
                    for (int output_channel=0;
                         output_channel<channels; ++output_channel) {
                        const std::size_t destination =
                            (((static_cast<std::size_t>(q)*path_count+path)
                                *channels+input_channel)
                              *channels+output_channel);
                        result[destination] =
                            h_projection(source_row, output_channel);
                    }
                }
        }
    }
    return result;
}

template <typename Precision>
void MACEKokkos<Precision>::compute_execution_density_parameter_gradients(
    const std::string& network,
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r,
    Kokkos::View<Precision***,Kokkos::LayoutRight> output,
    Kokkos::View<Precision***,Kokkos::LayoutRight> output_adjoint)
{
    if (!execution_parameter_gradients_enabled)
        return;
    if (network != "A0" && network != "A1")
        throw std::invalid_argument(
            "Execution density parameter replay requires A0 or A1.");
    if (!compact_radial_model
        || (network == "A0" && !compact_radial_model->has_A0())
        || (network == "A1" && !compact_radial_model->has_A1()))
        throw std::runtime_error(
            "Execution density parameter replay requires the requested compact network.");

    const int active_count = num_active_types;
    const int pair_count = active_count*(active_count+1)/2;
    const int spline_nodes = compact_radial_model->num_spline_points();
    const int intervals = spline_nodes-1;
    const double h = compact_radial_model->spline_h();
    const double x0 = compact_radial_model->spline_min();
    const double cutoff = r_cut;
    const std::size_t coefficient_elements =
        static_cast<std::size_t>(pair_count)*intervals*4;
    std::size_t scratch_bytes = coefficient_elements*sizeof(double);
    const auto add_view_scratch = [&] (
        const std::size_t elements, const std::size_t element_bytes) {
        if (elements != 0
            && element_bytes
                > std::numeric_limits<std::size_t>::max()/elements)
            throw std::length_error(
                "Execution density parameter replay scratch size overflow.");
        const std::size_t bytes = elements*element_bytes;
        if (bytes > std::numeric_limits<std::size_t>::max()-scratch_bytes)
            throw std::length_error(
                "Execution density parameter replay scratch size overflow.");
        scratch_bytes += bytes;
    };
    add_view_scratch(node_types.size()+num_neigh.size()+neigh_types.size()
        +type_to_active.size(), sizeof(int));
    add_view_scratch(r.size(), sizeof(double));
    add_view_scratch(output.size()+output_adjoint.size(), sizeof(Precision));
    admit_execution_parameter_gradient_scratch(scratch_bytes);

    Kokkos::fence();
    const auto h_node_types = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), node_types);
    const auto h_num_neigh = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), num_neigh);
    const auto h_neigh_types = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), neigh_types);
    const auto h_type_to_active = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), type_to_active);
    const auto h_r = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), r);
    const auto h_output = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), output);
    const auto h_output_adjoint = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), output_adjoint);

    auto coefficient_adjoints = std::vector<double>(coefficient_elements, 0.0);
    int edge_begin = 0;
    for (int receiver=0; receiver<num_nodes; ++receiver) {
        double density_adjoint = 0.0;
        for (int lm=0; lm<num_lm; ++lm)
            for (int channel=0; channel<num_channels; ++channel)
                density_adjoint -=
                    static_cast<double>(h_output(receiver,lm,channel))
                    *static_cast<double>(
                        h_output_adjoint(receiver,lm,channel));
        const int active_i = h_type_to_active(h_node_types(receiver));
        const int edge_end = edge_begin+h_num_neigh(receiver);
        for (int edge=edge_begin; edge<edge_end; ++edge) {
            if (!(h_r(edge) < cutoff))
                continue;
            const int active_j = h_type_to_active(h_neigh_types(edge));
            const int pair = active_i <= active_j
                ? active_i*(2*active_count-active_i-1)/2+active_j
                : active_j*(2*active_count-active_j-1)/2+active_i;
            int interval = static_cast<int>(std::floor((h_r(edge)-x0)/h));
            double x = h_r(edge)-x0-h*interval;
            if (interval < 0) {
                interval = 0;
                x = 0.0;
            } else if (interval >= intervals) {
                interval = intervals-1;
                x = h;
            }
            const double basis[] = {1.0, x, x*x, x*x*x};
            for (int order=0; order<4; ++order)
                coefficient_adjoints[
                    (static_cast<std::size_t>(pair)*intervals+interval)*4+order]
                        += density_adjoint*basis[order];
        }
        edge_begin = edge_end;
    }
    if (edge_begin != static_cast<int>(r.extent(0)))
        throw std::runtime_error(
            "Execution density parameter replay edge traversal is inconsistent.");

    auto pair_types = std::vector<std::pair<int,int>>();
    pair_types.reserve(pair_count);
    for (int active_i=0; active_i<active_count; ++active_i)
        for (int active_j=active_i; active_j<active_count; ++active_j)
            pair_types.emplace_back(
                active_types[active_i], active_types[active_j]);
    const auto nodal_adjoints =
        compact_radial_model->transpose_spline_coefficients(
            coefficient_adjoints, pair_count, 1);
    const auto network_gradients =
        compact_radial_model->backpropagate_network_nodal_values(
            network, pair_types, nodal_adjoints, false);
    for (std::size_t layer=0; layer<network_gradients.weights.size(); ++layer) {
        auto& destination = execution_parameter_gradient_group(
            "compact_radial."+network+".weights."+std::to_string(layer)).values;
        if (destination.size() != network_gradients.weights[layer].size())
            throw std::runtime_error(
                "Execution density compact-network gradient extent is inconsistent.");
        destination = network_gradients.weights[layer];
    }
}

template <typename Precision>
void MACEKokkos<Precision>::compute_standard_r0_parameter_gradients(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r)
{
    if (!execution_parameter_gradients_enabled)
        return;
    if (!factorized_ready || !compact_radial_model || num_active_types <= 0)
        throw std::runtime_error(
            "Execution R0 parameter replay requires a prepared compact model.");

    const int channels = num_channels;
    const int functions = (l_max+1)*channels;
    const int active_count = num_active_types;
    const int pair_count = active_count*(active_count+1)/2;
    const int spline_nodes = compact_radial_model->num_spline_points();
    const int intervals = spline_nodes-1;
    const double h = compact_radial_model->spline_h();
    const double x0 = compact_radial_model->spline_min();
    const double cutoff = r_cut;
    const std::size_t coefficient_elements =
        static_cast<std::size_t>(pair_count)*intervals*4*functions;
    const unsigned int available_threads = std::max(
        1u, std::thread::hardware_concurrency());
    std::size_t worker_count = std::max<std::size_t>(
        1, std::min<std::size_t>({16, available_threads,
            static_cast<unsigned int>(std::max(num_nodes, 1))}));
    std::size_t scratch_bytes = 0;
    const auto add_scratch = [&] (const std::size_t elements,
                                  const std::size_t element_bytes) {
        if (elements != 0
            && element_bytes
                > std::numeric_limits<std::size_t>::max()/elements)
            throw std::length_error(
                "Execution R0 parameter replay scratch size overflow.");
        const std::size_t bytes = elements*element_bytes;
        if (bytes > std::numeric_limits<std::size_t>::max()-scratch_bytes)
            throw std::length_error(
                "Execution R0 parameter replay scratch size overflow.");
        scratch_bytes += bytes;
    };
    add_scratch(node_types.size()+num_neigh.size()+neigh_types.size()
        +type_to_active.size(), sizeof(int));
    add_scratch(r.size(), sizeof(double));
    add_scratch(Y.size()+A0_adj.size(), sizeof(Precision));
    add_scratch(coefficient_elements*(1+worker_count), sizeof(double));
    add_scratch(
        static_cast<std::size_t>(pair_count)*spline_nodes*functions*2,
        sizeof(double));
    add_scratch(worker_count*channels, sizeof(double));
    add_scratch(worker_count*execution_parameter_gradient_group(
        "H0_weights").values.size(), sizeof(double));
    for (int l=0; l<=l_max; ++l)
        add_scratch(worker_count*execution_parameter_gradient_group(
            "A0_weights.l"+std::to_string(l)).values.size(), sizeof(double));
    std::size_t worker_scratch_elements =
        coefficient_elements+channels
        +execution_parameter_gradient_group("H0_weights").values.size();
    for (int l=0; l<=l_max; ++l)
        worker_scratch_elements += execution_parameter_gradient_group(
            "A0_weights.l"+std::to_string(l)).values.size();
    const std::size_t worker_scratch_bytes =
        worker_scratch_elements*sizeof(double);
    const std::size_t scratch_budget =
        execution_parameter_gradients_max_bytes
        -execution_parameter_gradients_result_bytes;
    while (scratch_bytes > scratch_budget && worker_count > 1) {
        scratch_bytes -= worker_scratch_bytes;
        worker_count -= 1;
    }
    execution_parameter_gradients_r0_workers = worker_count;
    admit_execution_parameter_gradient_scratch(scratch_bytes);

    Kokkos::fence();
    const auto h_node_types = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), node_types);
    const auto h_num_neigh = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), num_neigh);
    const auto h_neigh_types = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), neigh_types);
    const auto h_type_to_active = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), type_to_active);
    const auto h_r = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), r);
    const auto h_Y = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), Y);
    const auto h_A0_adj = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), A0_adj);

    auto pair_types = std::vector<std::pair<int,int>>();
    pair_types.reserve(pair_count);
    auto spline_coefficients = std::vector<double>(coefficient_elements, 0.0);
    for (int active_i=0; active_i<active_count; ++active_i) {
        for (int active_j=active_i; active_j<active_count; ++active_j) {
            const int pair = active_i*(2*active_count-active_i-1)/2+active_j;
            pair_types.emplace_back(
                active_types[active_i], active_types[active_j]);
            const auto data = compact_radial_model->materialize_pair(
                active_types[active_i], active_types[active_j]).R0;
            if (data.values.size() != static_cast<std::size_t>(functions)
                || data.derivatives.size() != static_cast<std::size_t>(functions))
                throw std::runtime_error(
                    "Execution R0 radial extent is inconsistent.");
            for (int function=0; function<functions; ++function) {
                if (data.values[function].size()
                        != static_cast<std::size_t>(spline_nodes)
                    || data.derivatives[function].size()
                        != static_cast<std::size_t>(spline_nodes))
                    throw std::runtime_error(
                        "Execution R0 spline node extent is inconsistent.");
                for (int interval=0; interval<intervals; ++interval) {
                    const double value = data.values[function][interval];
                    const double derivative = data.derivatives[function][interval];
                    const double next_value = data.values[function][interval+1];
                    const double next_derivative =
                        data.derivatives[function][interval+1];
                    const std::size_t base =
                        ((static_cast<std::size_t>(pair)*intervals+interval)*4)
                            *functions+function;
                    spline_coefficients[base] = value;
                    spline_coefficients[base+functions] = derivative;
                    spline_coefficients[base+2*functions] =
                        (-3.0*value-2.0*h*derivative
                            +3.0*next_value-h*next_derivative)/(h*h);
                    spline_coefficients[base+3*functions] =
                        (2.0*value+h*derivative
                            -2.0*next_value+h*next_derivative)/(h*h*h);
                }
            }
        }
    }
    auto coefficient_adjoints = std::vector<double>(coefficient_elements, 0.0);
    auto& H0_gradient = execution_parameter_gradient_group("H0_weights").values;
    struct R0ReplayWorkerGradients {
        std::vector<double> coefficients;
        std::vector<double> H0;
        std::vector<std::vector<double>> A0;
    };
    auto worker_gradients = std::vector<R0ReplayWorkerGradients>(worker_count);
    for (auto& worker : worker_gradients) {
        worker.coefficients.assign(coefficient_elements, 0.0);
        worker.H0.assign(H0_gradient.size(), 0.0);
        worker.A0.resize(l_max+1);
        for (int l=0; l<=l_max; ++l)
            worker.A0[l].assign(
                execution_parameter_gradient_group(
                    "A0_weights.l"+std::to_string(l)).values.size(),
                0.0);
    }

    const auto evaluation_point = [&] (const double radius) {
        int interval = static_cast<int>(std::floor((radius-x0)/h));
        double x = radius-x0-h*interval;
        if (interval < 0) {
            interval = 0;
            x = 0.0;
        } else if (interval >= intervals) {
            interval = intervals-1;
            x = h;
        }
        return std::pair<int,double>(interval, x);
    };
    const auto radial_value = [&] (
        const int pair,
        const int interval,
        const double x,
        const int function) {
        const std::size_t base =
            ((static_cast<std::size_t>(pair)*intervals+interval)*4)
                *functions+function;
        const double xx = x*x;
        return spline_coefficients[base]
            +x*spline_coefficients[base+functions]
            +xx*spline_coefficients[base+2*functions]
            +xx*x*spline_coefficients[base+3*functions];
    };

    auto receiver_edge_offsets = std::vector<int>(num_nodes+1, 0);
    for (int receiver=0; receiver<num_nodes; ++receiver)
        receiver_edge_offsets[receiver+1] =
            receiver_edge_offsets[receiver]+h_num_neigh(receiver);
    if (receiver_edge_offsets.back() != static_cast<int>(r.extent(0)))
        throw std::runtime_error(
            "Execution R0 parameter replay edge traversal is inconsistent.");

    const auto replay_receivers = [&] (
        const std::size_t worker_index,
        const int receiver_begin,
        const int receiver_end) {
        auto& coefficient_adjoints = worker_gradients[worker_index].coefficients;
        auto& H0_gradient = worker_gradients[worker_index].H0;
        for (int receiver=receiver_begin; receiver<receiver_end; ++receiver) {
        const int edge_begin = receiver_edge_offsets[receiver];
        const int global_i = h_node_types(receiver);
        const int active_i = h_type_to_active(global_i);
        const int edge_end = edge_begin+h_num_neigh(receiver);
        for (int edge=edge_begin; edge<edge_end; ++edge) {
            if (!(h_r(edge) < cutoff))
                continue;
            const int global_j = h_neigh_types(edge);
            const int active_j = h_type_to_active(global_j);
            const int pair = active_i <= active_j
                ? active_i*(2*active_count-active_i-1)/2+active_j
                : active_j*(2*active_count-active_j-1)/2+active_i;
            const auto [interval, x] = evaluation_point(h_r(edge));
            const double basis[] = {1.0, x, x*x, x*x*x};
            for (int l=0; l<=l_max; ++l) {
                auto& A0_gradient = worker_gradients[worker_index].A0[l];
                const auto& A0_weights = A0_weights_host[global_i][l];
                if (A0_weights.size()
                    != static_cast<std::size_t>(channels)*channels)
                    throw std::runtime_error(
                        "Execution R0 replay requires square A0 channel matrices.");
                auto output_angular_adjoints = std::vector<double>(channels, 0.0);
                for (int output=0; output<channels; ++output)
                    for (int lm=l*l; lm<(l+1)*(l+1); ++lm)
                        output_angular_adjoints[output] +=
                            static_cast<double>(h_A0_adj(receiver,lm,output))
                            *static_cast<double>(h_Y(edge*num_lm+lm));
                for (int input=0; input<channels; ++input) {
                    const int function = l*channels+input;
                    const double radial = radial_value(
                        pair, interval, x, function);
                    const double H0_weight =
                        H0_weights_host[global_j*channels+input];
                    double mixed_adjoint = 0.0;
                    for (int output=0; output<channels; ++output) {
                        const double angular_adjoint =
                            output_angular_adjoints[output];
                        const double weight =
                            A0_weights[input*channels+output];
                        A0_gradient[
                            (static_cast<std::size_t>(global_i)*channels+input)
                                *channels+output] +=
                                    angular_adjoint*H0_weight*radial;
                        mixed_adjoint += angular_adjoint*weight;
                    }
                    H0_gradient[global_j*channels+input] +=
                        mixed_adjoint*radial;
                    const double radial_adjoint = mixed_adjoint*H0_weight;
                    for (int order=0; order<4; ++order)
                        coefficient_adjoints[
                            ((static_cast<std::size_t>(pair)*intervals+interval)*4
                                +order)*functions+function] +=
                                    radial_adjoint*basis[order];
                }
            }
        }
        }
    };
    auto worker_errors = std::vector<std::exception_ptr>(worker_count);
    auto workers = std::vector<std::thread>();
    workers.reserve(worker_count);
    try {
        for (std::size_t worker=0; worker<worker_count; ++worker) {
            const int receiver_begin = static_cast<int>(
                static_cast<std::size_t>(num_nodes)*worker/worker_count);
            const int receiver_end = static_cast<int>(
                static_cast<std::size_t>(num_nodes)*(worker+1)/worker_count);
            workers.emplace_back([&, worker, receiver_begin, receiver_end] {
                try {
                    replay_receivers(worker, receiver_begin, receiver_end);
                } catch (...) {
                    worker_errors[worker] = std::current_exception();
                }
            });
        }
    } catch (...) {
        for (auto& worker : workers)
            worker.join();
        throw;
    }
    for (auto& worker : workers)
        worker.join();
    for (const auto& error : worker_errors)
        if (error)
            std::rethrow_exception(error);

    for (std::size_t worker=0; worker<worker_count; ++worker) {
        for (std::size_t index=0; index<coefficient_adjoints.size(); ++index)
            coefficient_adjoints[index] +=
                worker_gradients[worker].coefficients[index];
        for (std::size_t index=0; index<H0_gradient.size(); ++index)
            H0_gradient[index] += worker_gradients[worker].H0[index];
        for (int l=0; l<=l_max; ++l) {
            auto& destination = execution_parameter_gradient_group(
                "A0_weights.l"+std::to_string(l)).values;
            for (std::size_t index=0; index<destination.size(); ++index)
                destination[index] += worker_gradients[worker].A0[l][index];
        }
    }

    const auto nodal_adjoints =
        compact_radial_model->transpose_spline_coefficients(
            coefficient_adjoints, pair_count, functions);
    const auto network_gradients =
        compact_radial_model->backpropagate_network_nodal_values(
            "R0", pair_types, nodal_adjoints, false);
    for (std::size_t layer=0; layer<network_gradients.weights.size(); ++layer) {
        auto& destination = execution_parameter_gradient_group(
            "compact_radial.R0.weights."+std::to_string(layer)).values;
        if (destination.size() != network_gradients.weights[layer].size())
            throw std::runtime_error(
                "Execution R0 compact-network gradient extent is inconsistent.");
        destination = network_gradients.weights[layer];
    }
}

template <typename Precision>
void MACEKokkos<Precision>::compute_factorized_parameter_gradients(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r)
{
    if (!execution_parameter_gradients_enabled)
        return;
    if (!factorized_ready || !compact_radial_model)
        throw std::runtime_error(
            "Execution R1 parameter replay requires a prepared compact model.");

    const int channels = num_channels;
    const int harmonics = num_lm;
    const int embedding = factorized_embedding_width;
    const int active_count = num_active_types;
    const int pair_count = active_count*(active_count+1)/2;
    const int spline_nodes = compact_radial_model->num_spline_points();
    const int intervals = spline_nodes-1;
    const double h = compact_radial_model->spline_h();
    const double x0 = compact_radial_model->spline_min();
    const double cutoff = r_cut;
    const auto R1_weight_shapes =
        compact_radial_model->network_weight_shapes("R1");
    const std::size_t R1_final_layer = R1_weight_shapes.size()-1;
    const std::size_t coefficient_elements =
        static_cast<std::size_t>(pair_count)*intervals*4*embedding;
    const unsigned int available_threads = std::max(
        1u, std::thread::hardware_concurrency());
    std::size_t worker_count = std::max<std::size_t>(
        1, std::min<std::size_t>({16, available_threads,
            static_cast<unsigned int>(std::max(num_nodes, 1))}));
    std::size_t maximum_projected_elements_per_edge = 0;
    std::size_t maximum_message_elements = 0;
    auto message_offsets = std::vector<std::size_t>(l_max+2, 0);
    for (int l=0; l<=l_max; ++l) {
        const std::size_t components = 2*l+1;
        const std::size_t eta =
            execution_group_path_offsets_host[l+1]
            -execution_group_path_offsets_host[l];
        maximum_projected_elements_per_edge = std::max(
            maximum_projected_elements_per_edge,
            eta*static_cast<std::size_t>(channels));
        maximum_message_elements = std::max(
            maximum_message_elements,
            components*eta*static_cast<std::size_t>(channels));
        message_offsets[l+1] = message_offsets[l]
            +components*eta*static_cast<std::size_t>(channels);
    }
    std::size_t scratch_bytes = 0;
    const auto add_scratch = [&] (const std::size_t elements,
                                  const std::size_t element_bytes) {
        if (elements != 0
            && element_bytes
                > std::numeric_limits<std::size_t>::max()/elements)
            throw std::length_error(
                "Execution R1 parameter replay scratch size overflow.");
        const std::size_t bytes = elements*element_bytes;
        if (bytes > std::numeric_limits<std::size_t>::max()-scratch_bytes)
            throw std::length_error(
                "Execution R1 parameter replay scratch size overflow.");
        scratch_bytes += bytes;
    };
    add_scratch(node_types.size()+num_neigh.size()+neigh_indices.size()
        +neigh_types.size()+type_to_active.size(), sizeof(int));
    add_scratch(r.size(), sizeof(double));
    add_scratch(Y.size()+H1.size()+A1_adj.size(), sizeof(Precision));
    add_scratch(coefficient_elements*(1+worker_count), sizeof(double));
    add_scratch(
        static_cast<std::size_t>(pair_count)*spline_nodes*embedding*2,
        sizeof(double));
    const std::size_t maximum_degree = execution_schedule_num_neigh_host.empty()
        ? 0
        : *std::max_element(
            execution_schedule_num_neigh_host.begin(),
            execution_schedule_num_neigh_host.end());
    add_scratch(
        worker_count*maximum_degree*(
            2*static_cast<std::size_t>(embedding)
            +static_cast<std::size_t>(num_lme)*channels
            +maximum_projected_elements_per_edge),
        sizeof(double));
    add_scratch(worker_count*2*maximum_message_elements, sizeof(double));
    for (int l=0; l<=l_max; ++l)
        add_scratch(
            execution_final_projection(l).size()+A1_weights(l).size(),
            sizeof(double));
    add_scratch(worker_count*execution_parameter_gradient_group(
        "compact_radial.R1.weights."+std::to_string(R1_final_layer)
        ).values.size(), sizeof(double));
    for (int l=0; l<=l_max; ++l)
        add_scratch(worker_count*execution_parameter_gradient_group(
            "A1_weights.l"+std::to_string(l)).values.size(), sizeof(double));
    std::size_t worker_scratch_elements = coefficient_elements
        +maximum_degree*(
            2*static_cast<std::size_t>(embedding)
            +static_cast<std::size_t>(num_lme)*channels
            +maximum_projected_elements_per_edge)
        +2*maximum_message_elements
        +execution_parameter_gradient_group(
            "compact_radial.R1.weights."+std::to_string(R1_final_layer)
            ).values.size();
    for (int l=0; l<=l_max; ++l)
        worker_scratch_elements += execution_parameter_gradient_group(
            "A1_weights.l"+std::to_string(l)).values.size();
    const std::size_t worker_scratch_bytes =
        worker_scratch_elements*sizeof(double);
    const std::size_t scratch_budget =
        execution_parameter_gradients_max_bytes
        -execution_parameter_gradients_result_bytes;
    while (scratch_bytes > scratch_budget && worker_count > 1) {
        scratch_bytes -= worker_scratch_bytes;
        worker_count -= 1;
    }
    execution_parameter_gradients_r1_workers = worker_count;
    admit_execution_parameter_gradient_scratch(scratch_bytes);

    Kokkos::fence();
    const auto h_node_types = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), node_types);
    const auto h_num_neigh = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), num_neigh);
    const auto h_neigh_indices = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), neigh_indices);
    const auto h_neigh_types = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), neigh_types);
    const auto h_type_to_active = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), type_to_active);
    const auto h_r = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), r);
    const auto h_Y = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), Y);
    const auto h_H1 = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), H1);
    const auto h_A1_adj = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), A1_adj);
    const auto h_group_paths = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), execution_group_paths);
    const auto h_path_component_offsets = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), execution_path_component_offsets);
    const auto h_path_component_lme = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), execution_path_component_lme);
    const auto h_cg_offsets = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), execution_cg_offsets);
    const auto h_cg_rows = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), execution_cg_rows);
    const auto h_cg_coefficients = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), execution_cg_coefficients);
    const auto h_lm1_rows = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), Phi1_lm1);
    const auto h_lm2_rows = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), Phi1_lm2);

    auto pair_types = std::vector<std::pair<int,int>>();
    pair_types.reserve(pair_count);
    auto spline_coefficients = std::vector<double>(coefficient_elements, 0.0);
    for (int active_i=0; active_i<active_count; ++active_i) {
        for (int active_j=active_i; active_j<active_count; ++active_j) {
            const int pair = active_i*(2*active_count-active_i-1)/2+active_j;
            pair_types.emplace_back(
                active_types[active_i], active_types[active_j]);
            const auto data = compact_radial_model->materialize_factorized_pair(
                active_types[active_i], active_types[active_j]).R1.penultimate;
            if (data.values.size() != static_cast<std::size_t>(embedding)
                || data.derivatives.size() != static_cast<std::size_t>(embedding))
                throw std::runtime_error(
                    "Execution R1 penultimate radial extent is inconsistent.");
            for (int q=0; q<embedding; ++q) {
                if (data.values[q].size() != static_cast<std::size_t>(spline_nodes)
                    || data.derivatives[q].size()
                        != static_cast<std::size_t>(spline_nodes))
                    throw std::runtime_error(
                        "Execution R1 penultimate spline node extent is inconsistent.");
                for (int interval=0; interval<intervals; ++interval) {
                    const double value = data.values[q][interval];
                    const double derivative = data.derivatives[q][interval];
                    const double next_value = data.values[q][interval+1];
                    const double next_derivative = data.derivatives[q][interval+1];
                    const std::size_t base =
                        ((static_cast<std::size_t>(pair)*intervals+interval)*4)
                            *embedding+q;
                    spline_coefficients[base] = value;
                    spline_coefficients[base+embedding] = derivative;
                    spline_coefficients[base+2*embedding] =
                        (-3.0*value-2.0*h*derivative
                            +3.0*next_value-h*next_derivative)/(h*h);
                    spline_coefficients[base+3*embedding] =
                        (2.0*value+h*derivative
                            -2.0*next_value+h*next_derivative)/(h*h*h);
                }
            }
        }
    }
    auto coefficient_adjoints = std::vector<double>(coefficient_elements, 0.0);
    auto& final_projection_adjoints = execution_parameter_gradient_group(
        "compact_radial.R1.weights."+std::to_string(R1_final_layer)).values;
    struct ReplayWorkerGradients {
        std::vector<double> coefficients;
        std::vector<double> final_projection;
        std::vector<std::vector<double>> A1;
    };
    auto worker_gradients = std::vector<ReplayWorkerGradients>(worker_count);
    for (auto& worker : worker_gradients) {
        worker.coefficients.assign(coefficient_elements, 0.0);
        worker.final_projection.assign(final_projection_adjoints.size(), 0.0);
        worker.A1.resize(l_max+1);
        for (int l=0; l<=l_max; ++l)
            worker.A1[l].assign(
                execution_parameter_gradient_group(
                    "A1_weights.l"+std::to_string(l)).values.size(),
                0.0);
    }

    const auto evaluation_point = [&] (const double radius) {
        int interval = static_cast<int>(std::floor((radius-x0)/h));
        double x = radius-x0-h*interval;
        if (interval < 0) {
            interval = 0;
            x = 0.0;
        } else if (interval >= intervals) {
            interval = intervals-1;
            x = h;
        }
        return std::pair<int,double>(interval, x);
    };
    const auto radial_value = [&] (
        const int pair, const int interval, const double x, const int q) {
        const std::size_t base =
            ((static_cast<std::size_t>(pair)*intervals+interval)*4)
                *embedding+q;
        const double xx = x*x;
        return spline_coefficients[base]
            +x*spline_coefficients[base+embedding]
            +xx*spline_coefficients[base+2*embedding]
            +xx*x*spline_coefficients[base+3*embedding];
    };
    const auto angular_value = [&] (
        const int edge,
        const int path,
        const int component,
        const int channel) {
        const int lme = h_path_component_lme(
            h_path_component_offsets(path)+component);
        double angular = 0.0;
        for (int term=h_cg_offsets(lme); term<h_cg_offsets(lme+1); ++term) {
            const int row = h_cg_rows(term);
            angular += static_cast<double>(h_cg_coefficients(term))
                *static_cast<double>(h_Y(edge*harmonics+h_lm1_rows(row)))
                *static_cast<double>(h_H1(
                    h_neigh_indices(edge),h_lm2_rows(row),channel));
        }
        return angular;
    };

    auto final_projection_values = std::vector<std::vector<double>>(l_max+1);
    auto A1_weight_values = std::vector<std::vector<double>>(l_max+1);
    for (int l=0; l<=l_max; ++l) {
        const auto h_final_projection = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), execution_final_projection(l));
        final_projection_values[l].assign(
            h_final_projection.data(),
            h_final_projection.data()+h_final_projection.size());
        const auto h_A1_weights = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), A1_weights(l));
        A1_weight_values[l].assign(
            h_A1_weights.data(), h_A1_weights.data()+h_A1_weights.size());
    }

    auto receiver_edge_offsets = std::vector<int>(num_nodes+1, 0);
    for (int receiver=0; receiver<num_nodes; ++receiver)
        receiver_edge_offsets[receiver+1] =
            receiver_edge_offsets[receiver]+h_num_neigh(receiver);
    if (receiver_edge_offsets.back() != static_cast<int>(neigh_indices.extent(0)))
        throw std::runtime_error(
            "Execution R1 parameter replay edge traversal is inconsistent.");

    const auto replay_receivers = [&] (
        const std::size_t worker_index,
        const int receiver_begin,
        const int receiver_end) {
        auto& coefficient_adjoints = worker_gradients[worker_index].coefficients;
        auto& final_projection_adjoints =
            worker_gradients[worker_index].final_projection;
        for (int receiver=receiver_begin; receiver<receiver_end; ++receiver) {
        const int edge_begin = receiver_edge_offsets[receiver];
        const int active_i = h_type_to_active(h_node_types(receiver));
        const int edge_end = edge_begin+h_num_neigh(receiver);
        const int degree = edge_end-edge_begin;
        auto edge_pairs = std::vector<int>(degree);
        auto edge_intervals = std::vector<int>(degree);
        auto edge_x = std::vector<double>(degree);
        auto edge_active = std::vector<unsigned char>(degree, 0);
        auto edge_radial = std::vector<double>(
            static_cast<std::size_t>(degree)*embedding);
        auto edge_radial_adjoint = std::vector<double>(
            edge_radial.size(), 0.0);
        auto edge_angular = std::vector<double>(
            static_cast<std::size_t>(degree)*num_lme*channels);
        for (int local_edge=0; local_edge<degree; ++local_edge) {
            const int edge = edge_begin+local_edge;
            if (!(h_r(edge) < cutoff))
                continue;
            edge_active[local_edge] = 1;
            const int active_j = h_type_to_active(h_neigh_types(edge));
            const int pair = active_i <= active_j
                ? active_i*(2*active_count-active_i-1)/2+active_j
                : active_j*(2*active_count-active_j-1)/2+active_i;
            const auto [interval, x] = evaluation_point(h_r(edge));
            edge_pairs[local_edge] = pair;
            edge_intervals[local_edge] = interval;
            edge_x[local_edge] = x;
            for (int q=0; q<embedding; ++q)
                edge_radial[static_cast<std::size_t>(local_edge)*embedding+q] =
                    radial_value(pair, interval, x, q);
            for (int path=0; path<h_group_paths.extent(0); ++path) {
                const int components =
                    h_path_component_offsets(path+1)
                    -h_path_component_offsets(path);
                for (int component=0; component<components; ++component) {
                    const int lme = h_path_component_lme(
                        h_path_component_offsets(path)+component);
                    for (int input=0; input<channels; ++input)
                        edge_angular[
                            (static_cast<std::size_t>(local_edge)*num_lme+lme)
                                *channels+input] = angular_value(
                                    edge, path, component, input);
                }
            }
        }

        for (int l=0; l<=l_max; ++l) {
            const int components = 2*l+1;
            const int group_begin = execution_group_path_offsets_host[l];
            const int eta_count =
                execution_group_path_offsets_host[l+1]-group_begin;
            const auto message_index = [=] (
                int component, int eta, int channel) {
                return (static_cast<std::size_t>(component)*eta_count+eta)
                    *channels+channel;
            };
            const auto& final_projection = final_projection_values[l];
            const auto& a1_weights = A1_weight_values[l];
            auto message_adjoint = std::vector<double>();
            auto projected_radial = std::vector<double>(
                static_cast<std::size_t>(degree)*eta_count*channels, 0.0);
            for (int local_edge=0; local_edge<degree; ++local_edge)
                for (int eta=0; eta<eta_count; ++eta)
                    for (int input=0; input<channels; ++input) {
                        const int row = eta*channels+input;
                        double value = 0.0;
                        for (int q=0; q<embedding; ++q)
                            value += edge_radial[
                                static_cast<std::size_t>(local_edge)*embedding+q]
                                *final_projection[
                                    static_cast<std::size_t>(row)*embedding+q];
                        projected_radial[
                            (static_cast<std::size_t>(local_edge)*eta_count+eta)
                                *channels+input] = value;
                    }

            auto message = std::vector<double>(
                static_cast<std::size_t>(components)*eta_count*channels, 0.0);
            message_adjoint.assign(message.size(), 0.0);
            for (int component=0; component<components; ++component)
                for (int eta=0; eta<eta_count; ++eta) {
                    const int path = h_group_paths(group_begin+eta);
                    const int lme = h_path_component_lme(
                        h_path_component_offsets(path)+component);
                    for (int input=0; input<channels; ++input) {
                        double value = 0.0;
                        for (int local_edge=0; local_edge<degree; ++local_edge)
                            value += edge_angular[
                                (static_cast<std::size_t>(local_edge)*num_lme+lme)
                                    *channels+input]
                                *projected_radial[
                                    (static_cast<std::size_t>(local_edge)*eta_count
                                        +eta)*channels+input];
                        message[message_index(component,eta,input)] = value;
                    }
                }

            for (int component=0; component<components; ++component)
                for (int eta=0; eta<eta_count; ++eta)
                    for (int input=0; input<channels; ++input) {
                        const int row = eta*channels+input;
                        const double value =
                            message[message_index(component,eta,input)];
                        double adjoint = 0.0;
                        for (int output=0; output<channels; ++output) {
                            const double output_adjoint = static_cast<double>(
                                h_A1_adj(receiver,l*l+component,output));
                            worker_gradients[worker_index].A1[l][
                                row*channels+output] += value*output_adjoint;
                            adjoint += output_adjoint*a1_weights[
                                static_cast<std::size_t>(row)*channels+output];
                        }
                        message_adjoint[message_index(component,eta,input)] =
                            adjoint;
                    }

            for (int local_edge=0; local_edge<degree; ++local_edge) {
                for (int eta=0; eta<eta_count; ++eta) {
                    const int path = h_group_paths(group_begin+eta);
                    for (int input=0; input<channels; ++input) {
                        double path_adjoint = 0.0;
                        for (int component=0; component<components; ++component) {
                            const int lme = h_path_component_lme(
                                h_path_component_offsets(path)+component);
                            path_adjoint += edge_angular[
                                (static_cast<std::size_t>(local_edge)*num_lme+lme)
                                    *channels+input]
                                *message_adjoint[
                                    message_index(component,eta,input)];
                        }
                        const int row = eta*channels+input;
                        for (int q=0; q<embedding; ++q) {
                            const std::size_t radial_index =
                                static_cast<std::size_t>(local_edge)*embedding+q;
                            const std::size_t final_index =
                                (static_cast<std::size_t>(path)*channels+input)
                                    *embedding+q;
                            final_projection_adjoints[final_index] +=
                                edge_radial[radial_index]*path_adjoint;
                            edge_radial_adjoint[radial_index] +=
                                final_projection[
                                    static_cast<std::size_t>(row)*embedding+q]
                                *path_adjoint;
                        }
                    }
                }
            }
        }

        for (int local_edge=0; local_edge<degree; ++local_edge) {
            if (edge_active[local_edge] == 0)
                continue;
            const int pair = edge_pairs[local_edge];
            const int interval = edge_intervals[local_edge];
            const double x = edge_x[local_edge];
            const double basis[] = {1.0, x, x*x, x*x*x};
            for (int q=0; q<embedding; ++q)
                for (int order=0; order<4; ++order)
                    coefficient_adjoints[
                        ((static_cast<std::size_t>(pair)*intervals+interval)*4
                            +order)*embedding+q] +=
                                edge_radial_adjoint[
                                    static_cast<std::size_t>(local_edge)*embedding+q]
                                *basis[order];
        }
        }
    };
    auto worker_errors = std::vector<std::exception_ptr>(worker_count);
    auto workers = std::vector<std::thread>();
    workers.reserve(worker_count);
    try {
        for (std::size_t worker=0; worker<worker_count; ++worker) {
            const int receiver_begin = static_cast<int>(
                static_cast<std::size_t>(num_nodes)*worker/worker_count);
            const int receiver_end = static_cast<int>(
                static_cast<std::size_t>(num_nodes)*(worker+1)/worker_count);
            workers.emplace_back([&, worker, receiver_begin, receiver_end] {
                try {
                    replay_receivers(worker, receiver_begin, receiver_end);
                } catch (...) {
                    worker_errors[worker] = std::current_exception();
                }
            });
        }
    } catch (...) {
        for (auto& worker : workers)
            worker.join();
        throw;
    }
    for (auto& worker : workers)
        worker.join();
    for (const auto& error : worker_errors)
        if (error)
            std::rethrow_exception(error);

    for (std::size_t worker=0; worker<worker_count; ++worker) {
        for (std::size_t index=0; index<coefficient_adjoints.size(); ++index)
            coefficient_adjoints[index] +=
                worker_gradients[worker].coefficients[index];
        for (std::size_t index=0; index<final_projection_adjoints.size(); ++index)
            final_projection_adjoints[index] +=
                worker_gradients[worker].final_projection[index];
        for (int l=0; l<=l_max; ++l) {
            auto& destination = execution_parameter_gradient_group(
                "A1_weights.l"+std::to_string(l)).values;
            for (std::size_t index=0; index<destination.size(); ++index)
                destination[index] += worker_gradients[worker].A1[l][index];
        }
    }

    const auto nodal_adjoints =
        compact_radial_model->transpose_spline_coefficients(
            coefficient_adjoints, pair_count, embedding);
    const auto network_gradients =
        compact_radial_model->backpropagate_network_nodal_values(
            "R1", pair_types, nodal_adjoints, true,
            final_projection_adjoints);
    for (std::size_t layer=0; layer<network_gradients.weights.size(); ++layer) {
        auto& destination = execution_parameter_gradient_group(
            "compact_radial.R1.weights."+std::to_string(layer)).values;
        if (destination.size() != network_gradients.weights[layer].size())
            throw std::runtime_error(
                "Execution R1 compact-network gradient extent is inconsistent.");
        destination = network_gradients.weights[layer];
    }
}


template class MACEKokkos<float>;
template class MACEKokkos<double>;
