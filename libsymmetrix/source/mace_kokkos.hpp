#pragma once

#include <memory>
#include <limits>
#include <map>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <span>
#include <array>
#include <chrono>
#include <type_traits>

#include "Kokkos_UnorderedMap.hpp"

#include "compact_radial.hpp"
#include "field_coupling.hpp"
#include "cubic_spline_kokkos.hpp"
#include "cubic_spline_set_kokkos.hpp"
#include "device_backend.hpp"
#include "execution_plan.hpp"
#include "mace_streamed_edges.hpp"
#include "multilayer_perceptron_kokkos.hpp"
#include "multivariate_polynomial.hpp"//TODO
#include "multivariate_polynomial_kokkos.hpp"
#include "radial_function_set_kokkos.hpp"
#include "jit_cuda_plugin.hpp"
#include "jit_hip_plugin.hpp"
#include "jit_host_plugin.hpp"
#include "jit_m0_host_plugin.hpp"
#include "jit_operator_module.hpp"
#include "tools_kokkos.hpp"
#include "zbl_kokkos.hpp"

struct FactorizedBlasContext;

namespace symmetrix::execution {
struct LaunchProfile;
}

struct ExecutionParameterGradientGroup {
    std::string name;
    std::string layout;
    std::vector<std::size_t> shape;
    std::vector<double> values;
};

struct FactorizedOperatorBenchmarkMeasurement {
    double cold_ms = 0.0;
    std::size_t warmup_count = 0;
    std::vector<double> samples_ms;
};

struct KernelLaunchProfileDiagnostic {
    std::string stage;
    std::string implementation_id;
    std::string implementation_kind;
    std::string profile_id;
    std::string selection_source;
    std::string rejection_reason;
    int default_blocks_per_compute_unit = 0;
    int active_blocks_per_compute_unit = 0;
    int persistent_blocks = 0;
    std::vector<int> calibration_candidates;
    bool calibration_permitted = false;
};

template <typename Precision>
class MACEKokkos {
    KokkosLiveObjectGuard kokkos_live_object_guard;

public:

MACEKokkos(std::string filename, std::string requested_head = {});
~MACEKokkos();
void set_streamed_edges(std::string mode);
std::string streamed_edges_mode() const;
void set_execution_plan_request(
    std::string algorithm, std::string profile, std::string debug_plan = {});
void resolve_execution_plan();
const symmetrix::execution::ExecutionPlanReport& execution_plan_report() const;
bool supports_streamed_edges() const;
bool supports_fused_streamed_reverse() const;
bool supports_factorized() const;
void load_jit_cuda_plugin(
    std::string path, int persistent_blocks_per_compute_unit = 8);
bool jit_cuda_plugin_ready() const;
std::string jit_cuda_plugin_path() const;
std::string jit_cuda_plugin_artifact_id() const;
void load_jit_hip_plugin(std::string path);
bool jit_hip_plugin_ready() const;
std::string jit_hip_plugin_path() const;
std::string jit_hip_plugin_artifact_id() const;
void load_jit_device_plugin(
    std::string path, int persistent_blocks_per_compute_unit = 8);
bool jit_device_plugin_ready() const;
std::string jit_device_plugin_path() const;
std::string jit_device_plugin_artifact_id() const;
void load_m0_device_module(
    std::string path, std::string schedule = "chunk32",
    int persistent_blocks_per_compute_unit = 8);
bool m0_device_module_ready() const;
std::string m0_device_module_path() const;
std::string m0_device_module_artifact_id() const;
std::string m0_device_module_schedule_name() const;
void load_m0_host_plugin(std::string path);
bool m0_host_plugin_ready() const;
std::string m0_host_plugin_path() const;
std::string m0_host_plugin_artifact_id() const;
void load_r0_device_module(
    std::string path, int persistent_blocks_per_compute_unit = 8);
bool r0_device_module_ready() const;
std::string r0_device_module_path() const;
std::string r0_device_module_artifact_id() const;
bool device_cuda_available() const;
std::string execution_cuda_device_name() const;
int execution_cuda_device_ordinal() const;
int execution_cuda_compute_capability() const;
int execution_cuda_multiprocessor_count() const;
int execution_cuda_warp_width() const;
int execution_cuda_runtime_version() const;
int execution_cuda_driver_version() const;
void load_jit_host_plugin(std::string path);
bool jit_host_plugin_ready() const;
std::string jit_host_plugin_path() const;
std::string jit_host_plugin_artifact_id() const;
void set_factorized_source_strategy(std::string strategy);
void set_factorized_direct_forward_executor(std::string executor);
void set_factorized_direct_reverse_executor(std::string executor);
void set_standard_r0_executor(std::string executor);
void set_standard_m0_executor(std::string executor);
void set_kernel_launch_policy(std::string policy);
void set_kernel_launch_profile_override(std::string profile_id, int blocks);
void set_jit_device_plugin_launch_override(std::string stage, int blocks);
void clear_kernel_launch_profile_overrides();
std::string kernel_launch_policy_name() const;
std::vector<KernelLaunchProfileDiagnostic>
kernel_launch_profile_diagnostics() const;
void set_m1_polynomial_policy(std::string policy);
void set_m1_recompute_tile_channels(int channels);
void set_m1_recompute_scratch_limit_for_testing(std::size_t bytes);
void set_mh0_state_policy(std::string policy);
void set_readout_policy(std::string policy);
void set_phi1_policy(std::string policy);
void set_edge_geometry_policy(std::string policy);
void set_harmonic_storage_policy(std::string policy);
void set_low_memory_device_memory_info_for_testing(
    std::size_t free_bytes, std::size_t total_bytes);
void set_single_layer_workspace_receiver_limit_for_testing(int receivers);
void set_dual_layer_workspace_receiver_limit_for_testing(int receivers);
void set_allow_fixed_workspace(bool enabled);
bool allow_fixed_workspace() const;
void set_low_memory(bool enabled);
void set_factorized_reverse_cache_policy(std::string policy);
void set_factorized_planner_budget_bytes(std::size_t budget_bytes);
std::string factorized_source_strategy_name() const;
std::string factorized_execution_strategy_name() const;
std::string factorized_execution_profile_name() const;
std::string factorized_derivative_signature_name() const;
std::string factorized_direct_forward_executor_name() const;
std::string factorized_selected_direct_forward_executor_name() const;
std::string factorized_direct_reverse_executor_name() const;
std::string factorized_selected_direct_reverse_executor_name() const;
std::string standard_r0_executor_name() const;
std::string standard_r0_selected_executor_name() const;
std::string standard_m0_executor_name() const;
std::string standard_m0_selected_executor_name() const;
std::string r0_implementation_name() const;
std::string m0_implementation_name() const;
std::string r0_selected_module_id() const;
std::string m0_selected_module_id() const;
bool r0_supports_low_memory() const;
bool m0_supports_low_memory() const;
std::string m1_polynomial_policy_name() const;
std::string m1_polynomial_policy_request_name() const;
std::string m1_recompute_backend_name() const;
std::string m1_recompute_fallback_reason_name() const;
std::size_t m1_poly_values_active_bytes() const;
std::size_t m1_poly_values_capacity_bytes() const;
std::size_t m1_poly_adjoints_active_bytes() const;
std::size_t m1_poly_adjoints_capacity_bytes() const;
std::size_t m1_recompute_scratch_bytes() const;
std::size_t m1_recompute_scratch_limit_bytes() const;
std::string mh0_state_policy_name() const;
std::string mh0_state_policy_request_name() const;
std::string mh0_state_policy_fallback_reason_name() const;
std::size_t mh0_reused_state_bytes() const;
std::size_t mh0_auxiliary_state_bytes() const;
bool h1_m0_adjoint_ping_pong_active() const;
std::size_t h1_m0_adjoint_ping_pong_bytes() const;
bool mh0_m0_adjoint_alias_active() const;
std::size_t mh0_m0_forward_replacement_count_value() const;
std::size_t mh0_m0_adjoint_allocation_count_value() const;
std::size_t mh0_m0_alias_detach_count_value() const;
std::string readout_policy_name() const;
std::string phi1_policy_name() const;
std::size_t phi1_workspace_bytes() const;
std::size_t generic_phi1_source_adjoint_workspace_bytes() const;
std::size_t generic_phi1_source_adjoint_allocation_count = 0;
std::size_t readout_workspace_bytes() const;
bool low_memory_enabled() const;
bool low_memory_requested() const;
std::string low_memory_policy_name() const;
std::string low_memory_selection_reason_name() const;
std::size_t low_memory_device_free_bytes() const;
std::size_t low_memory_device_total_bytes() const;
std::size_t low_memory_reserve_bytes() const;
std::size_t low_memory_available_bytes() const;
std::size_t low_memory_speed_estimated_bytes() const;
std::size_t low_memory_capacity_y_only_estimated_bytes() const;
std::size_t low_memory_capacity_retained_estimated_bytes() const;
std::size_t low_memory_capacity_estimated_bytes() const;
std::size_t low_memory_selected_estimated_bytes() const;
std::string execution_geometry_growth_reason_name() const;
int execution_active_receiver_count() const;
int execution_active_feature_node_count() const;
int execution_active_edge_count() const;
int execution_planned_receiver_capacity() const;
int execution_planned_feature_node_capacity() const;
int execution_planned_edge_capacity() const;
std::size_t execution_planned_capacity_bytes() const;
std::string execution_capacity_selection_reason_name() const;
std::size_t execution_result_allocation_count_value() const;
int single_layer_workspace_active_receivers() const;
int single_layer_workspace_planned_receivers() const;
int single_layer_workspace_active_edges() const;
int single_layer_workspace_planned_edges() const;
std::size_t single_layer_workspace_bytes() const;
std::size_t single_layer_workspace_bytes_per_receiver() const;
std::size_t single_layer_workspace_bytes_per_edge() const;
std::size_t single_layer_workspace_replacement_count() const;
std::size_t single_layer_workspace_reuse_count() const;
std::size_t single_layer_workspace_batch_count() const;
std::size_t single_layer_tiled_evaluation_count() const;
int dual_layer_workspace_active_receivers() const;
int dual_layer_workspace_planned_receivers() const;
int dual_layer_workspace_active_edges() const;
int dual_layer_workspace_planned_edges() const;
int dual_layer_source_segment_count() const;
std::size_t dual_layer_workspace_bytes() const;
std::size_t dual_layer_workspace_bytes_per_receiver() const;
std::size_t dual_layer_workspace_bytes_per_edge() const;
std::size_t dual_layer_schedule_bytes() const;
std::size_t dual_layer_schedule_preparation_explicit_scratch_bytes() const;
std::size_t dual_layer_workspace_replacement_count() const;
std::size_t dual_layer_workspace_reuse_count() const;
std::size_t dual_layer_workspace_batch_count() const;
std::size_t dual_layer_tiled_evaluation_count() const;
std::string edge_geometry_policy_name() const;
std::size_t compact_edge_geometry_bytes() const;
std::string harmonic_storage_policy_name() const;
std::string harmonic_storage_policy_request_name() const;
std::string harmonic_storage_selection_reason_name() const;
std::string harmonic_storage_fallback_reason_name() const;
std::size_t harmonic_value_bytes() const;
std::size_t harmonic_gradient_bytes() const;
std::size_t shuffled_coordinate_bytes() const;
int macefield_response_m1_recompute_tile_channels() const;
std::size_t macefield_response_m1_recompute_scratch_bytes() const;
std::size_t macefield_response_m1_recompute_scratch_limit_bytes() const;
std::string factorized_reverse_cache_policy_name() const;
std::string factorized_selected_reverse_cache_policy_name() const;
std::vector<int> factorized_retained_reverse_groups() const;
std::vector<int> factorized_recomputed_reverse_groups() const;
std::string factorized_jit_artifact_id() const;
std::string factorized_jit_contract_fingerprint() const;
std::string standard_r0_module_id() const;
int standard_r0_module_revision() const;
std::string standard_m0_module_id() const;
int standard_m0_module_revision() const;
std::size_t standard_m0_poly_values_active_bytes() const;
std::size_t standard_m0_poly_values_capacity_bytes() const;
std::size_t standard_m0_poly_adjoints_active_bytes() const;
std::size_t standard_m0_poly_adjoints_capacity_bytes() const;
std::size_t factorized_forward_coupling_workspace_bytes() const;
std::size_t factorized_reverse_coupling_workspace_bytes() const;
std::size_t factorized_custom_blas_launch_count() const;
std::size_t factorized_blas_stream_bind_count() const;
std::uint64_t prepare_all_interactions_graph(
    int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const int> neigh_types);
void validate_graph_cardinality(
    std::size_t num_receivers,
    std::size_t num_feature_nodes,
    std::size_t num_edges) const;
void prepare_all_interactions_geometry(
    std::uint64_t graph_generation,
    std::span<const double> reference_positions,
    std::span<const double> reference_xyz,
    std::span<const double> cell,
    std::span<const double> inverse_cell,
    std::span<const int> pbc);
void compute_prepared_all_interactions_positions(
    std::uint64_t graph_generation,
    std::span<const double> positions);
std::uint64_t prepare_factorized_graph(
    int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const int> neigh_types);
std::uint64_t prepare_factorized_graph(
    int num_receivers,
    int num_feature_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const int> neigh_types);
std::uint64_t prepare_factorized_graph(
    int num_receivers,
    int num_feature_nodes,
    std::span<const int> receiver_feature_indices,
    std::span<const int> node_types,
    std::span<const int> feature_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const int> neigh_types);
std::uint64_t prepare_factorized_graph_device(
    int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types);
std::pair<std::uint64_t,std::size_t> prepare_periodic_factorized_graph(
    std::span<const int> node_types,
    std::span<const double> positions,
    std::span<const double> cell,
    std::span<const double> inverse_cell,
    double neighbor_cutoff);
std::uint64_t prepare_factorized_graph_device(
    int num_receivers,
    int num_feature_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types);
std::uint64_t prepare_factorized_graph_device(
    int num_receivers,
    int num_feature_nodes,
    Kokkos::View<const int*> receiver_feature_indices,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> feature_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types);
void prepare_factorized_geometry(
    std::uint64_t graph_generation,
    std::span<const double> reference_positions,
    std::span<const double> reference_xyz,
    std::span<const double> cell,
    std::span<const double> inverse_cell,
    std::span<const int> pbc);
void prepare_factorized_shift_geometry(
    std::uint64_t graph_generation,
    std::span<const double> reference_positions,
    std::span<const int> shifts,
    std::span<const double> cell,
    std::span<const double> inverse_cell,
    std::span<const int> pbc);
void prepare_factorized_fractional_geometry(
    std::uint64_t graph_generation,
    std::span<const double> fractional_positions,
    std::span<const double> fractional_xyz,
    std::span<const double> cell,
    std::span<const double> inverse_cell,
    std::span<const int> pbc);
void update_factorized_cell(
    std::uint64_t graph_generation,
    std::span<const double> cell,
    std::span<const double> inverse_cell);
void compute_prepared_factorized(
    std::uint64_t graph_generation,
    std::span<const double> xyz,
    std::span<const double> r);
void compute_prepared_factorized_positions(
    std::uint64_t graph_generation,
    std::span<const double> positions);
void compute_prepared_factorized_positions_field(
    std::uint64_t graph_generation,
    std::span<const double> positions,
    std::span<const double> electric_field);
void compute_prepared_factorized_field(
    std::uint64_t graph_generation,
    std::span<const double> xyz,
    std::span<const double> r,
    std::span<const double> electric_field);
void update_prepared_factorized_positions_geometry(
    std::uint64_t graph_generation,
    std::span<const double> positions);
void validate_prepared_factorized_positions_geometry() const;
std::size_t execution_geometry_workspace_bytes() const;
void set_factorized_observer(bool enabled, std::size_t max_bytes);
void set_execution_parameter_gradients(bool enabled, std::size_t max_bytes);
std::uint64_t prepare_factorized_operator_benchmark(
    std::uint64_t graph_generation,
    std::span<const double> xyz,
    std::span<const double> r);
void run_factorized_operator_benchmark(
    std::uint64_t token,
    const std::string& mode);
void synchronize_factorized_operator_benchmark(std::uint64_t token);
FactorizedOperatorBenchmarkMeasurement measure_factorized_operator_benchmark(
    std::uint64_t token,
    const std::string& mode,
    std::size_t warmup_iterations,
    double warmup_ms,
    std::size_t min_samples,
    std::size_t max_samples,
    double min_sample_ms);
std::vector<Precision> factorized_operator_benchmark_radial_linear(
    std::uint64_t token) const;
void validate_factorized_operator_benchmark(std::uint64_t token) const;
MACEStreamedEdgesMode streamed_edges = MACEStreamedEdgesMode::materialized;

// Basic model information
int num_elements;
int num_channels;
int num_interactions = 2;
bool single_layer_readout = false;
double r_cut;
int l_max, num_lm;
int L_max, num_LM;
Kokkos::View<int*> atomic_numbers;
Kokkos::View<double*> atomic_energies;
std::string selected_head;
std::vector<std::string> available_heads;
std::vector<int> atomic_numbers_host;
std::vector<int> active_atomic_numbers;
void prepare_active_types(std::vector<int> node_types);

// Node energies, forces, and graph stress
Kokkos::View<double*> node_energies, node_forces, atom_forces, stress_tensor;
Kokkos::View<double*> node_energies_storage, node_forces_storage;
std::size_t execution_result_allocation_count = 0;
bool single_layer_tiled_plan_active = false;
bool dual_layer_tiled_plan_active = false;
int single_layer_workspace_receiver_limit = 32768;
int single_layer_workspace_active_capacity = 0;
int single_layer_workspace_planned_capacity = 0;
int single_layer_workspace_active_edge_capacity = 0;
int single_layer_workspace_planned_edge_capacity = 0;
std::size_t single_layer_workspace_payload_bytes = 0;
std::size_t single_layer_workspace_receiver_bytes = 0;
std::size_t single_layer_workspace_edge_bytes = 0;
std::size_t single_layer_workspace_replacements = 0;
std::size_t single_layer_workspace_reuses = 0;
std::size_t single_layer_workspace_batches = 0;
std::size_t single_layer_tiled_evaluations = 0;
Kokkos::View<std::uint64_t*> single_layer_workspace_arena;
Kokkos::View<Precision***,Kokkos::LayoutRight> single_layer_workspace_a0;
Kokkos::View<Precision***,Kokkos::LayoutRight> single_layer_workspace_equivariant_a;
Kokkos::View<Precision***,Kokkos::LayoutRight> single_layer_workspace_equivariant_b;
Kokkos::View<Precision**,Kokkos::LayoutRight> single_layer_workspace_m1;
Kokkos::View<double**,Kokkos::LayoutRight> single_layer_workspace_h2;
Kokkos::View<double*> single_layer_workspace_density;
Kokkos::View<int*> single_layer_workspace_first_neigh;
Kokkos::View<int*> single_layer_workspace_neigh_types;
Kokkos::View<double*> single_layer_workspace_virial;
bool single_layer_tiled_outputs_ready = false;
void prepare_single_layer_tiled_workspace(int num_receivers);
void bind_single_layer_tiled_workspace_batch(int active_receivers);
void prepare_single_layer_tiled_geometry(
    int receiver_begin, int receiver_count, int edge_begin, int edge_count);
void accumulate_single_layer_tiled_outputs(
    int receiver_begin, int receiver_count, int edge_begin, int edge_count);
void compute_single_layer_tiled(
    int num_receivers,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r);
int dual_layer_workspace_receiver_limit = 32768;
int dual_layer_workspace_active_capacity = 0;
int dual_layer_workspace_planned_capacity = 0;
int dual_layer_workspace_active_edge_capacity = 0;
int dual_layer_workspace_planned_edge_capacity = 0;
int dual_layer_source_segments = 0;
int dual_layer_active_tile = 0;
int dual_layer_active_segment_begin = 0;
int dual_layer_active_segment_count = 0;
std::size_t dual_layer_workspace_payload_bytes = 0;
std::size_t dual_layer_workspace_receiver_bytes = 0;
std::size_t dual_layer_workspace_edge_bytes = 0;
std::size_t dual_layer_source_schedule_bytes = 0;
std::size_t dual_layer_source_schedule_preparation_explicit_scratch_bytes = 0;
std::size_t dual_layer_workspace_replacements = 0;
std::size_t dual_layer_workspace_reuses = 0;
std::size_t dual_layer_workspace_batches = 0;
std::size_t dual_layer_tiled_evaluations = 0;
Kokkos::View<std::uint64_t*> dual_layer_workspace_arena;
Kokkos::View<Precision***,Kokkos::LayoutRight> dual_layer_workspace_a0;
Kokkos::View<Precision***,Kokkos::LayoutRight> dual_layer_workspace_equivariant_a;
Kokkos::View<Precision***,Kokkos::LayoutRight> dual_layer_workspace_equivariant_b;
Kokkos::View<Precision***,Kokkos::LayoutRight> dual_layer_workspace_a1;
Kokkos::View<Precision***,Kokkos::LayoutRight> dual_layer_workspace_phi1;
Kokkos::View<Precision**,Kokkos::LayoutRight> dual_layer_workspace_m1;
Kokkos::View<double**,Kokkos::LayoutRight> dual_layer_workspace_h2;
Kokkos::View<double*> dual_layer_workspace_density_a0;
Kokkos::View<double*> dual_layer_workspace_density_a1;
Kokkos::View<int*> dual_layer_workspace_first_neigh;
Kokkos::View<int*> dual_layer_workspace_neigh_types;
Kokkos::View<double*> dual_layer_workspace_virial;
Kokkos::View<Precision***,Kokkos::LayoutRight> dual_layer_graph_h1;
Kokkos::View<Precision***,Kokkos::LayoutRight> dual_layer_graph_h1_adjoint;
Kokkos::View<int*> dual_layer_tile_segment_offsets;
Kokkos::View<int*> dual_layer_segment_source_ids;
Kokkos::View<int*> dual_layer_segment_edge_offsets;
Kokkos::View<int*> dual_layer_source_edges;
Kokkos::View<int*> dual_layer_edge_local_receivers;
std::vector<int> dual_layer_tile_segment_offsets_host;
void prepare_dual_layer_tiled_workspace(
    int num_receivers, int num_feature_nodes);
void bind_dual_layer_phase1_workspace(int receiver_begin, int receiver_count);
void bind_dual_layer_phase2_workspace(int receiver_begin, int receiver_count);
void bind_dual_layer_phase3_workspace(int receiver_begin, int receiver_count);
void prepare_dual_layer_tile_source_schedule_host(
    int num_receivers, int num_feature_nodes,
    const std::vector<int>& num_neigh,
    const std::vector<int>& neigh_indices);
void prepare_dual_layer_tile_source_schedule_device(
    int num_receivers, int num_feature_nodes, int num_edges);
void compute_dual_layer_tiled(
    int num_receivers,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r);
void compute_dual_layer_tiled_phase1(int num_receivers);
void compute_dual_layer_tiled_phase2(int num_receivers);
void compute_dual_layer_tiled_phase3(int num_receivers);
void compute_dual_layer_tiled_phase(int num_receivers, int phase);
void compute_readout_1(
    int num_nodes, Kokkos::View<const int*> node_types,
    bool initialize_h1_adjoint);
void compute_readout_2(int num_nodes);
static constexpr int streamed_fused_max_num_LM = 16;
static constexpr int streamed_fused_max_num_paths = 16;
// Crossover for the unfused edge-owned CUDA fallback.
static constexpr int streamed_edge_owned_limit = 100000;
Kokkos::View<int*> streamed_first_neigh, streamed_edge_receivers;
Kokkos::View<int*> all_interactions_candidate_node_types;
Kokkos::View<int*> all_interactions_candidate_num_neigh;
Kokkos::View<int*> all_interactions_candidate_first_neigh;
Kokkos::View<int*> all_interactions_candidate_edge_receivers;
Kokkos::View<int*> all_interactions_candidate_neigh_indices;
Kokkos::View<int*> all_interactions_candidate_neigh_types;
Kokkos::View<int*> all_interactions_candidate_active;
Kokkos::View<int*> all_interactions_active_num_neigh;
Kokkos::View<int*> all_interactions_active_edge_receivers;
Kokkos::View<int*> all_interactions_active_neigh_indices;
Kokkos::View<int*> all_interactions_active_neigh_types;
Kokkos::View<double*> all_interactions_candidate_xyz;
Kokkos::View<double*> all_interactions_candidate_r;
Kokkos::View<double*> all_interactions_active_xyz;
Kokkos::View<double*> all_interactions_active_r;
Kokkos::View<double*> all_interactions_positions;
Kokkos::View<double*> all_interactions_reference_positions;
Kokkos::View<double*> all_interactions_reference_xyz;
Kokkos::View<double*> all_interactions_displacements;
Kokkos::View<double*> all_interactions_cell;
Kokkos::View<double*> all_interactions_inverse_cell;
Kokkos::View<int*> all_interactions_pbc;
Kokkos::View<int*> all_interactions_geometry_invalid;
std::vector<int> all_interactions_node_types_host;
std::vector<int> all_interactions_num_neigh_host;
std::vector<int> all_interactions_neigh_indices_host;
std::vector<int> all_interactions_neigh_types_host;
std::uint64_t all_interactions_graph_generation_counter = 0;
std::uint64_t all_interactions_prepared_graph_generation = 0;
std::uint64_t all_interactions_prepared_geometry_graph_generation = 0;
std::uint64_t all_interactions_completed_graph_generation = 0;
std::size_t all_interactions_prepared_graph_count = 0;
std::size_t all_interactions_prepared_evaluation_count = 0;
std::size_t all_interactions_geometry_update_count = 0;
std::size_t all_interactions_schedule_build_count = 0;
int all_interactions_active_edge_count = 0;
int all_interactions_capacity_nodes = 0;
int all_interactions_capacity_edges = 0;
void prepare_streamed_edge_schedule(
    const int num_nodes,
    const int num_edges,
    Kokkos::View<const int*> num_neigh);
void compute_node_energies_forces(const int num_nodes,
                                  Kokkos::View<const int*> node_types,
                                  Kokkos::View<const int*> num_neigh,
                                  Kokkos::View<const int*> neigh_indices,
                                  Kokkos::View<const int*> neigh_types,
                                  Kokkos::View<const double*> xyz,
                                  Kokkos::View<const double*> r,
                                  std::uint64_t execution_graph_generation = 0,
                                  bool streamed_schedule_prepared = false);
void compute_node_energies_forces_field(const int num_nodes,
                                        Kokkos::View<const int*> node_types,
                                        Kokkos::View<const int*> num_neigh,
                                        Kokkos::View<const int*> neigh_indices,
                                        Kokkos::View<const int*> neigh_types,
                                        Kokkos::View<const double*> xyz,
                                        Kokkos::View<const double*> r,
                                        Kokkos::View<const double*> electric_field,
                                        std::uint64_t execution_graph_generation = 0);
void begin_factorized_distributed_evaluation(
    int num_receivers,
    int num_feature_nodes,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r,
    std::uint64_t execution_graph_generation);
void begin_factorized_distributed_positions_evaluation(
    int num_receivers,
    int num_feature_nodes,
    Kokkos::View<const double*> positions,
    std::uint64_t execution_graph_generation);
void compute_factorized_single_layer_distributed_evaluation(
    int num_receivers,
    int num_feature_nodes,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r,
    std::uint64_t execution_graph_generation);
void compute_factorized_single_layer_distributed_positions_evaluation(
    int num_receivers,
    int num_feature_nodes,
    Kokkos::View<const double*> positions,
    std::uint64_t execution_graph_generation);
void continue_factorized_distributed_evaluation(
    Kokkos::View<Precision***,Kokkos::LayoutRight> communicated_h1,
    Kokkos::View<const double*> electric_field = {});
void finish_factorized_distributed_evaluation();
void reduce_node_forces(
    int num_nodes,
    Kokkos::View<const int*> edge_receivers,
    Kokkos::View<const int*> edge_sources);
void reduce_prepared_node_forces(std::uint64_t graph_generation);
void reduce_prepared_all_interactions_node_forces(std::uint64_t graph_generation);
void reduce_stress(double volume, Kokkos::View<const double*> xyz);
void reduce_prepared_stress(double volume, std::uint64_t graph_generation);
void reduce_prepared_all_interactions_stress(
    double volume,
    std::uint64_t graph_generation);

// ZBL
bool has_zbl;
ZBLKokkos zbl;

// R0
bool uses_compact_radial = false;
std::unique_ptr<CompactRadialModel> compact_radial_model;
std::vector<int> active_types;
Kokkos::View<int*> type_to_active;
int num_active_types = 0;
std::vector<double> H0_weights_host;
std::vector<std::vector<std::vector<double>>> A0_weights_host;
double R0_spline_h;
double R0_spline_min = 0.0;
Kokkos::View<const Precision****,Kokkos::LayoutRight> R0_spline_coefficients;
Kokkos::View<Precision**,Kokkos::LayoutRight> R0, R0_deriv;
bool standard_r0_has_model_contract = false;
enum class R0Implementation {
    generic,
    builtin,
    device_module,
};
enum class StandardR0Executor {
    automatic,
    v1,
    v2_receiver,
    v2_edge16,
    v2_edge32,
};
StandardR0Executor standard_r0_executor = StandardR0Executor::automatic;
bool standard_r0_module_ready = false;
R0Implementation selected_r0_implementation = R0Implementation::generic;
bool r0_model_payload_verified = false;
std::unique_ptr<symmetrix::execution::OperatorModule> r0_device_module;
int r0_device_persistent_blocks_per_compute_unit = 8;
std::string standard_r0_module_fallback_reason;
bool standard_r0_module_active = false;
Kokkos::View<double*> standard_r0_density_state;
std::string standard_r0_model_contract_fingerprint;
std::string standard_r0_model_semantic_fingerprint;
std::string standard_r0_model_structure_fingerprint;
std::size_t standard_r0_module_launch_count = 0;
bool standard_r0_density_scale_fused = false;
void compute_R0(const int num_nodes,
                Kokkos::View<const int*> node_types,
                Kokkos::View<const int*> num_neigh,
                Kokkos::View<const int*> neigh_types,
                Kokkos::View<const double*> r);

// R1
RadialFunctionSetKokkos<Precision> radial_1;
Kokkos::View<Precision**,Kokkos::LayoutRight> R1, R1_deriv;
void compute_R1(const int num_nodes,
                Kokkos::View<const int*> node_types,
                Kokkos::View<const int*> num_neigh,
                Kokkos::View<const int*> neigh_types,
                Kokkos::View<const double*> r,
                bool include_derivatives = true);

// Spherical harmonics
Kokkos::View<Precision*> xyz_shuffled;
Kokkos::View<Precision*> Y, Y_grad;// TODO: make multidimensional
Kokkos::View<Precision*> Y_grad_shuffled;
struct SphericalHarmonicsState;
std::unique_ptr<SphericalHarmonicsState> spherical_harmonics_state;
std::size_t execution_sphericart_initialization_count = 0;
std::size_t execution_sphericart_launch_count = 0;
std::size_t execution_sphericart_async_launch_count = 0;
std::size_t execution_direct_harmonic_launch_count = 0;
void ensure_spherical_harmonics_state();
void compute_Y(
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r = {},
    bool evaluator_stream = false,
    int edge_begin = 0,
    int edge_count = -1);

// A0
Kokkos::View<Precision***,Kokkos::LayoutRight> A0, A0_adj;
void compute_A0(const int num_nodes,
                Kokkos::View<const int*> node_types,
                Kokkos::View<const int*> num_neigh,
                Kokkos::View<const int*> neigh_types);
void compute_A0_streamed(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r,
    int edge_begin = 0);
void reverse_A0(const int num_nodes,
                Kokkos::View<const int*> node_types,
                Kokkos::View<const int*> num_neigh,
                Kokkos::View<const int*> neigh_types,
                Kokkos::View<const double*> xyz,
                Kokkos::View<const double*> r);
void reverse_A0_streamed(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r,
    int receiver_base = 0,
    int edge_begin = 0,
    int edge_count = -1);

// A0 rescaling
bool A0_scaled;
RadialFunctionSetKokkos<double> A0_splines;
Kokkos::View<double**,Kokkos::LayoutRight> A0_spline_values;
Kokkos::View<double**,Kokkos::LayoutRight> A0_spline_derivs;
void compute_A0_scaled(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r);
void reverse_A0_scaled(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r);

// M0
Kokkos::View<Precision***,Kokkos::LayoutRight> M0, M0_adj;
std::size_t mh0_m0_forward_replacement_count = 0;
std::size_t mh0_m0_adjoint_allocation_count = 0;
std::size_t mh0_m0_alias_detach_count = 0;
Kokkos::View<Kokkos::View<int**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> M0_monomials;
Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace> M0_weights;
Kokkos::View<Kokkos::View<int**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> M0_poly_spec;
Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace> M0_poly_coeff;
Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace> M0_poly_values;
Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace> M0_poly_adjoints;
enum class StandardM0Executor {
    automatic,
    runtime,
    standard,
};
enum class StandardM0ModuleVariant {
    none,
    scalar_lmax0,
    full_lmax1,
};
enum class M0Implementation {
    generic,
    builtin,
    device_module,
    host_plugin,
};
StandardM0Executor standard_m0_executor = StandardM0Executor::automatic;
StandardM0ModuleVariant standard_m0_module_variant =
    StandardM0ModuleVariant::none;
bool standard_m0_module_ready = false;
M0Implementation selected_m0_implementation = M0Implementation::generic;
bool standard_m0_has_model_contract = false;
bool m0_model_payload_verified = false;
int m0_correlation = 0;
int m0_module_term_count = 0;
std::string standard_m0_model_contract_fingerprint;
std::string standard_m0_model_semantic_fingerprint;
std::string standard_m0_model_structure_fingerprint;
std::string standard_m0_module_fallback_reason;
Kokkos::View<Precision***,Kokkos::LayoutRight> standard_m0_module_weights;
std::unique_ptr<symmetrix::execution::OperatorModule> m0_device_module;
std::unique_ptr<symmetrix::execution::M0HostPlugin> m0_host_plugin;
std::string m0_device_module_schedule = "none";
int m0_device_persistent_blocks_per_compute_unit = 8;
std::size_t standard_m0_module_forward_launch_count = 0;
std::size_t standard_m0_module_reverse_launch_count = 0;
std::size_t standard_m0_input_scale_adjoint_launch_count = 0;
bool use_standard_m0_module() const;
bool use_r0_module() const;
bool use_m0_module() const;
void ensure_m0_polynomial_workspace();
void release_m0_polynomial_workspace();
void compute_M0(const int num_nodes, Kokkos::View<const int*> node_types);
void reverse_M0(const int num_nodes, Kokkos::View<const int*> node_types);
void compute_M0_module(
    const int num_nodes, Kokkos::View<const int*> node_types);
void reverse_M0_module(
    const int num_nodes, Kokkos::View<const int*> node_types);
void launch_M0_module_reverse(
    const Kokkos::DefaultExecutionSpace& execution_space,
    int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const Precision***,Kokkos::LayoutRight> input,
    Kokkos::View<const Precision***,Kokkos::LayoutRight> output_adjoint,
    Kokkos::View<Precision***,Kokkos::LayoutRight> input_adjoint,
    Kokkos::View<double*> input_scale_adjoint,
    bool capture_input_scale_adjoint);

// H1
Kokkos::View<Precision***,Kokkos::LayoutRight> H1, H1_adj, H1_pre_linear_up;
// Generic FP32 Phi1 reverse accumulates shared source adjoints in FP64 before
// one final conversion. The persistent view avoids allocation on repeated runs.
Kokkos::View<double***,Kokkos::LayoutRight> H1_adj_fp64_accumulator;
Kokkos::View<Precision***,Kokkos::LayoutRight>
    H1_weights, H1_weights_trans, H1_product_weights, H1_linear_up_weights;
bool first_interaction_residual = false;
Kokkos::View<Precision***,Kokkos::LayoutRight> H1_first_residual_weights;
Kokkos::View<Precision***,Kokkos::LayoutRight> H1_first_residual_fused_weights;
void compute_H1(const int num_nodes);
void reverse_H1(const int num_nodes);
void compute_H1_product(const int num_nodes);
void compute_H1_linear_up(const int num_nodes);
void reverse_H1_linear_up(const int num_nodes);
void reverse_H1_product(const int num_nodes);
void add_H1_first_residual(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    bool fused);

// MACEField coupling after H1 product
bool has_field_coupling;
int num_field_paths = 0;
int num_field_angular_entries = 0;
Kokkos::View<Precision***,Kokkos::LayoutRight> H1_pre_field;
Kokkos::View<int*> field_entry_input_lm;
Kokkos::View<int*> field_entry_component;
Kokkos::View<int*> field_entry_output_lm;
Kokkos::View<int*> field_entry_path;
Kokkos::View<int*> field_component_entry_offsets;
Kokkos::View<int*> field_component_entries;
Kokkos::View<Precision*> field_entry_coefficient;
Kokkos::View<Precision***,Kokkos::LayoutRight> field_path_matrix;
Kokkos::View<Precision***,Kokkos::LayoutRight> field_path_up_matrix;
Kokkos::View<double*> electric_field_adj;
Kokkos::View<double*> electric_field_hessian;
Kokkos::View<double*> electric_field_force_derivative;
Kokkos::View<Precision***,Kokkos::LayoutRight> field_H1_pre_adj;
Kokkos::View<double**,Kokkos::LayoutRight> field_H1_global_partial;
void compute_field_H1(const int num_nodes, Kokkos::View<const double*> electric_field);
void reverse_field_H1(const int num_nodes, Kokkos::View<const double*> electric_field);
void compute_electric_field_hessian(const int num_nodes,
                                    Kokkos::View<const int*> node_types,
                                    Kokkos::View<const int*> num_neigh,
                                    Kokkos::View<const int*> neigh_indices,
                                    Kokkos::View<const int*> neigh_types,
                                    Kokkos::View<const double*> xyz,
                                    Kokkos::View<const double*> r,
                                    Kokkos::View<const double*> electric_field);
void compute_current_electric_field_hessian(const int num_nodes,
                                            Kokkos::View<const int*> node_types,
                                            Kokkos::View<const int*> num_neigh,
                                            Kokkos::View<const int*> neigh_indices,
                                            Kokkos::View<const int*> neigh_types,
                                            Kokkos::View<const double*> xyz,
                                            Kokkos::View<const double*> r,
                                            Kokkos::View<const double*> electric_field,
                                            std::uint64_t graph_generation);
void compute_electric_field_force_derivative(const int num_nodes,
                                             Kokkos::View<const int*> node_types,
                                             Kokkos::View<const int*> num_neigh,
                                             Kokkos::View<const int*> neigh_indices,
                                             Kokkos::View<const int*> neigh_types,
                                             Kokkos::View<const double*> xyz,
                                             Kokkos::View<const double*> r,
                                             Kokkos::View<const double*> electric_field);
void compute_current_electric_field_force_derivative(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r,
    Kokkos::View<const double*> electric_field,
    std::uint64_t graph_generation);
void compute_prepared_factorized_field_response(
    std::uint64_t graph_generation,
    Kokkos::View<const double*> electric_field,
    bool include_force_derivative);
void compute_electric_field_response(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r,
    Kokkos::View<const double*> electric_field,
    bool recompute_primal,
    bool include_force_derivative);
void reconstruct_mh0_field_response_state(
    int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r);

// Phi1
int num_lelm1lm2, num_lme;
Kokkos::View<int*> Phi1_l, Phi1_l1, Phi1_l2;
Kokkos::View<int*> Phi1_lme, Phi1_lelm1lm2;
Kokkos::View<Precision*> Phi1_clebsch_gordan;
Kokkos::View<Precision***,Kokkos::LayoutRight> Phi1r, dPhi1r;
Kokkos::View<Precision***,Kokkos::LayoutRight> Phi1, dPhi1;
void compute_Phi1(const int num_nodes, Kokkos::View<const int*> num_neigh, Kokkos::View<const int*> neigh_indices);
void compute_Phi1_streamed(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r);
void compute_Phi1_streamed_jit(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r);
void reverse_Phi1(const int num_nodes, Kokkos::View<const int*> num_neigh, Kokkos::View<const int*> neigh_indices, Kokkos::View<const double*> xyz, Kokkos::View<const double*> r, bool zero_dxyz = true, bool zero_H1_adj = true);
void reverse_Phi1_streamed(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r,
    bool zero_dxyz = true,
    bool zero_H1_adj = true);

// TODO for testing of Phi1 strategies
Kokkos::View<int*> Phi1_lm1, Phi1_lm2, Phi1_lel1l2;
Kokkos::View<int*> Phi1_path_row_offsets;

// A1
Kokkos::View<Precision***,Kokkos::LayoutRight> A1, A1_adj;
Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> A1_weights;
Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> A1_weights_trans;
Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>
    A1_channel_tile_weights;
Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>
    A1_channel_tile_weights_trans;
static constexpr int phi1_channel_tile_size = 64;
void compute_A1(int num_nodes, bool completion_fence = true);
void compute_A1_channel_tile(int num_nodes, int channel_begin);
void reverse_A1(int num_nodes, bool completion_fence = true);
void reverse_A1_channel_tile(int num_nodes, int channel_begin);
void reverse_A1_from(
    int num_nodes,
    Kokkos::View<const Precision***,Kokkos::LayoutRight> output_adjoint,
    bool completion_fence = true);

// Receiver-factorized Execution R1 path
Kokkos::DefaultExecutionSpace factorized_execution_space;
std::unique_ptr<FactorizedBlasContext> factorized_blas_context;
static constexpr int factorized_default_chunk_size =
    std::is_same_v<Precision,float> ? 24 : 16;
int factorized_chunk_size = factorized_default_chunk_size;
std::string factorized_tile_selection_reason = "default tile";
enum class FactorizedSourceStrategy {
    serial_reference,
    team_cached,
    tiled_coupling,
    jit_plugin,
};
enum class FactorizedReverseCachePolicy {
    automatic,
    recompute,
    retain,
};
enum class FactorizedDirectReverseExecutor {
    automatic,
    runtime,
    jit,
};
enum class FactorizedDirectForwardExecutor {
    automatic,
    runtime,
    jit_all,
};
enum class FactorizedOperatorBenchmarkMode {
    forward,
    reverse,
    forward_reverse,
};
enum class KernelLaunchPolicy {
    automatic,
    static_profile,
};
KernelLaunchPolicy kernel_launch_policy = KernelLaunchPolicy::automatic;
std::map<std::string,int> kernel_launch_profile_overrides;
std::map<std::string,int> jit_device_plugin_launch_overrides;
std::map<std::string,KernelLaunchProfileDiagnostic>
    kernel_launch_profile_diagnostic_records;
bool factorized_ready = false;
bool factorized_tiled_ready = false;
FactorizedDirectReverseExecutor factorized_direct_reverse_executor =
    FactorizedDirectReverseExecutor::automatic;
FactorizedDirectForwardExecutor factorized_direct_forward_executor =
    FactorizedDirectForwardExecutor::automatic;
bool factorized_has_model_contract = false;
bool factorized_model_payload_verified = false;
int factorized_model_embedding = 0;
std::string factorized_model_contract_fingerprint;
std::string factorized_model_semantic_fingerprint;
std::string factorized_model_structure_fingerprint;
std::string factorized_model_payload_fallback_reason;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
std::unique_ptr<JitDevicePlugin> jit_device_plugin;
#endif
std::unique_ptr<symmetrix::execution::HostPlugin> jit_host_plugin;
Kokkos::View<Precision*> execution_receiver_projection;
Kokkos::View<Precision*> execution_a1_projection_weights;
FactorizedSourceStrategy factorized_source_strategy =
    FactorizedSourceStrategy::team_cached;
FactorizedReverseCachePolicy factorized_reverse_cache_policy =
    FactorizedReverseCachePolicy::automatic;
static constexpr std::size_t factorized_default_planner_budget_bytes =
    64u*1024u*1024u;
std::size_t factorized_planner_budget_bytes =
    factorized_default_planner_budget_bytes;
std::vector<bool> factorized_reverse_group_retain;
std::vector<std::size_t> factorized_reverse_group_workspace_bytes;
std::size_t factorized_planned_coupling_workspace_bytes = 0;
int factorized_planned_coupling_columns = 0;
int factorized_embedding_width = 0;
int factorized_max_coupling_columns = 0;
RadialFunctionSetKokkos<Precision> execution_radial_1;
std::vector<int> execution_group_path_offsets_host;
std::vector<int> execution_group_lme_offsets_host;
Kokkos::View<int*> execution_group_paths;
Kokkos::View<int*> execution_path_component_offsets;
Kokkos::View<int*> execution_path_component_lme;
Kokkos::View<int*> execution_cg_offsets;
Kokkos::View<int*> execution_cg_rows;
Kokkos::View<Precision*> execution_cg_coefficients;
Kokkos::View<int*> execution_source_lm_offsets;
Kokkos::View<int*> execution_source_eta_components;
Kokkos::View<int*> execution_source_lm1;
Kokkos::View<Precision*> execution_source_coefficients;
Kokkos::View<int*> execution_row_cg_offsets;
Kokkos::View<int*> execution_row_cg_lme;
Kokkos::View<Precision*> execution_row_cg_coefficients;
Kokkos::View<int*> execution_lme_component;
Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>
    execution_projection;
Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>
    execution_projection_trans;
Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>
    execution_final_projection;
using ExecutionScratchMatrix = Kokkos::View<
    Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
using ExecutionScratchTensor = Kokkos::View<
    Precision***,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
Kokkos::View<Precision*> factorized_workspace_arena;
std::size_t factorized_arena_allocation_count = 0;
Kokkos::View<Precision**,Kokkos::LayoutRight> execution_radial_values;
Kokkos::View<Precision**,Kokkos::LayoutRight> execution_radial_derivatives;
Kokkos::View<Precision**,Kokkos::LayoutRight> execution_coupling_adjoint;
Kokkos::View<int*> execution_source_chunk_offsets;
Kokkos::View<int*> execution_source_edges;
Kokkos::View<int*> execution_direct_source_offsets;
Kokkos::View<int*> execution_direct_source_edges;
Kokkos::View<int*> execution_edge_receivers;
Kokkos::View<int*> execution_prepared_receiver_feature_indices;
Kokkos::View<int*> execution_prepared_node_types;
Kokkos::View<int*> execution_prepared_feature_types;
Kokkos::View<int*> execution_prepared_num_neigh;
Kokkos::View<int*> execution_prepared_neigh_indices;
Kokkos::View<int*> execution_prepared_neigh_types;
// Owning graph views retain admitted high-water capacity. The views above and
// streamed_first_neigh are active-prefix subviews consumed by kernels.
Kokkos::View<int*> execution_source_chunk_offsets_storage;
Kokkos::View<int*> execution_source_edges_storage;
Kokkos::View<int*> execution_direct_source_offsets_storage;
Kokkos::View<int*> execution_direct_source_edges_storage;
Kokkos::View<int*> execution_edge_receivers_storage;
Kokkos::View<int*> execution_prepared_receiver_feature_indices_storage;
Kokkos::View<int*> execution_prepared_node_types_storage;
Kokkos::View<int*> execution_prepared_feature_types_storage;
Kokkos::View<int*> execution_prepared_num_neigh_storage;
Kokkos::View<int*> execution_prepared_neigh_indices_storage;
Kokkos::View<int*> execution_prepared_neigh_types_storage;
Kokkos::View<int*> streamed_first_neigh_storage;
Kokkos::View<double*> execution_prepared_xyz;
Kokkos::View<Precision*> execution_prepared_unit_direction;
Kokkos::View<double*> execution_prepared_r;
Kokkos::View<double*> single_layer_workspace_xyz;
Kokkos::View<
    const double*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>
    single_layer_explicit_xyz_host;
Kokkos::View<
    const double*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>
    single_layer_explicit_r_host;
Kokkos::View<const double*> single_layer_explicit_xyz_device;
Kokkos::View<const double*> single_layer_explicit_r_device;
Kokkos::View<const double*> single_layer_explicit_feature_positions_device;
Kokkos::View<double*> execution_prepared_positions;
Kokkos::View<double*> execution_prepared_reference_positions;
Kokkos::View<double*> execution_prepared_reference_xyz;
Kokkos::View<double*> execution_prepared_displacements;
Kokkos::View<double*> execution_prepared_positions_storage;
Kokkos::View<double*> execution_prepared_reference_positions_storage;
Kokkos::View<double*> execution_prepared_reference_xyz_storage;
Kokkos::View<double*> execution_prepared_displacements_storage;
Kokkos::View<int*> execution_prepared_edge_shifts;
Kokkos::View<int*> execution_prepared_edge_shifts_storage;
Kokkos::View<double*> execution_prepared_cell;
Kokkos::View<double*> execution_prepared_inverse_cell;
Kokkos::View<double*,Kokkos::HostSpace> execution_prepared_cell_host;
Kokkos::View<double*,Kokkos::HostSpace> execution_prepared_inverse_cell_host;
Kokkos::View<int*> execution_prepared_pbc;
Kokkos::View<int*> execution_prepared_geometry_invalid;
Kokkos::View<double*> execution_prepared_electric_field;
int execution_geometry_capacity_edges = 0;
int execution_geometry_planned_capacity_edges = 0;
std::string execution_geometry_growth_reason = "geometric growth";
std::size_t execution_geometry_allocation_count = 0;
std::size_t execution_geometry_copy_count = 0;
int execution_active_receivers = 0;
int execution_active_feature_nodes = 0;
int execution_active_edges = 0;
int execution_planned_receivers = 0;
int execution_planned_feature_nodes = 0;
int execution_planned_edges = 0;
std::size_t execution_planned_bytes = 0;
std::string execution_capacity_selection_reason = "exact active capacity";
enum class EdgeGeometryPolicy {
    cartesian_f64,
    unit_f32_radius_f64,
};
enum class Phi1Policy {
    retained,
    channel_tiled_64,
    receiver_local,
};
enum class HarmonicStoragePolicy {
    automatic,
    retained,
    y_only_direct,
};
enum class LowMemoryPolicy {
    disabled,
    pending,
    speed,
    capacity_retained,
    capacity_y_only,
};
Phi1Policy phi1_policy = Phi1Policy::retained;
HarmonicStoragePolicy harmonic_storage_policy =
    HarmonicStoragePolicy::retained;
HarmonicStoragePolicy harmonic_storage_policy_request =
    HarmonicStoragePolicy::retained;
std::string harmonic_storage_selection_reason = "explicit retained policy";
bool low_memory_request = false;
bool allow_fixed_workspace_ = false;
LowMemoryPolicy low_memory_policy = LowMemoryPolicy::disabled;
symmetrix::execution::ExecutionPlanReport execution_plan_;
std::string execution_plan_debug_id_;
std::string low_memory_speed_m1_policy_request = "automatic";
bool low_memory_speed_m1_recompute = false;
std::string low_memory_selection_reason = "low_memory=False";
std::size_t low_memory_device_free = 0;
std::size_t low_memory_device_total = 0;
std::size_t low_memory_reserve = 0;
std::size_t low_memory_capacity_y_only_estimate = 0;
std::size_t low_memory_capacity_retained_estimate = 0;
std::size_t low_memory_available = 0;
std::size_t low_memory_speed_estimate = 0;
std::size_t low_memory_selected_estimate = 0;
std::size_t single_layer_tiled_estimate = 0;
std::size_t dual_layer_tiled_estimate = 0;
bool low_memory_device_memory_info_override = false;
EdgeGeometryPolicy edge_geometry_policy = EdgeGeometryPolicy::cartesian_f64;
bool compact_edge_geometry_active = false;
bool factorized_prepared_geometry_uses_shifts = false;
bool use_compact_edge_geometry() const;
bool use_y_only_direct_harmonics() const;
std::string y_only_direct_harmonics_admission_reason() const;
void select_low_memory_policy_for_graph(
    int num_receivers, int num_feature_nodes, std::size_t num_edges,
    std::span<const int> num_neigh);
void refresh_execution_plan_report();
void apply_low_memory_policy(LowMemoryPolicy policy);
LowMemoryPolicy preferred_speed_policy(std::size_t num_edges = 0) const;
void apply_speed_policy(LowMemoryPolicy policy);
void apply_preferred_speed_policy(std::size_t num_edges = 0);
std::size_t estimate_graph_bytes(
    int num_receivers, int num_feature_nodes, std::size_t num_edges,
    bool capacity_bundle, bool retain_harmonic_gradients) const;
std::size_t estimate_single_layer_tiled_graph_bytes(
    int num_receivers, int num_feature_nodes, std::size_t num_edges,
    int workspace_receivers, std::size_t workspace_edges) const;
std::string single_layer_tiled_admission_reason() const;
std::size_t estimate_dual_layer_tiled_graph_bytes(
    int num_receivers, int num_feature_nodes, std::size_t num_edges,
    int workspace_receivers, std::size_t workspace_edges,
    std::size_t source_segments) const;
std::string dual_layer_tiled_admission_reason() const;
std::size_t estimate_geometry_bytes_per_edge(
    bool capacity_bundle, bool retain_harmonic_gradients) const;
std::vector<int> execution_schedule_num_neigh_host;
std::vector<int> execution_schedule_neigh_indices_host;
std::vector<int> execution_prepared_receiver_feature_indices_host;
std::vector<int> execution_prepared_node_types_host;
std::vector<int> execution_prepared_feature_types_host;
std::vector<int> execution_prepared_num_neigh_host;
std::vector<int> execution_prepared_neigh_indices_host;
std::vector<int> execution_prepared_neigh_types_host;
std::vector<int> execution_receiver_offsets_host;
std::vector<int> execution_chunk_edge_offsets_host;
int execution_schedule_num_nodes = 0;
int execution_schedule_num_feature_nodes = 0;
int execution_prepared_num_feature_nodes = 0;
int execution_schedule_num_chunks = 0;
enum class FactorizedDistributedPhase {
    idle,
    prefix_complete,
    middle_complete,
};
FactorizedDistributedPhase factorized_distributed_phase =
    FactorizedDistributedPhase::idle;
int factorized_distributed_num_receivers = 0;
int factorized_distributed_num_feature_nodes = 0;
std::uint64_t factorized_distributed_graph_generation = 0;
Kokkos::View<const double*> factorized_distributed_xyz;
Kokkos::View<const double*> factorized_distributed_r;
Kokkos::View<const double*> factorized_distributed_electric_field;
Kokkos::View<Precision***,Kokkos::LayoutRight>
    factorized_distributed_prefix_h1;
std::chrono::steady_clock::time_point factorized_distributed_start;
bool factorized_schedule_dirty = true;
std::uint64_t factorized_graph_generation_counter = 0;
std::uint64_t factorized_prepared_graph_generation = 0;
std::uint64_t factorized_prepared_geometry_graph_generation = 0;
std::size_t factorized_prepared_graph_count = 0;
std::size_t factorized_graph_device_replacement_count = 0;
std::size_t factorized_graph_device_update_count = 0;
std::size_t factorized_graph_device_capacity_bytes = 0;
std::size_t factorized_distributed_prefix_h1_allocation_count = 0;
std::size_t factorized_prepared_evaluation_count = 0;
std::size_t factorized_prepared_geometry_update_count = 0;
std::size_t factorized_fractional_geometry_preparation_count = 0;
std::size_t factorized_fractional_geometry_initialization_bytes = 0;
std::size_t factorized_cell_update_count = 0;
std::size_t factorized_cell_update_bytes = 0;
std::size_t factorized_geometry_state_allocation_count = 0;
bool factorized_prepared_geometry_fractional = false;
std::size_t factorized_fallback_evaluation_count = 0;
std::size_t factorized_topology_validation_count = 0;
std::size_t factorized_topology_validation_skip_count = 0;
std::size_t factorized_preparation_fence_count = 0;
std::size_t factorized_evaluation_fence_count = 0;
std::size_t factorized_stage_fence_count = 0;
std::size_t factorized_jit_launch_count = 0;
std::size_t factorized_jit_forward_launch_count = 0;
std::size_t factorized_jit_reverse_launch_count = 0;
double factorized_last_graph_prepare_ms = 0.0;
double factorized_last_evaluation_ms = 0.0;
bool factorized_operator_benchmark_ready = false;
std::uint64_t factorized_operator_benchmark_token_counter = 0;
std::uint64_t factorized_operator_benchmark_token = 0;
std::uint64_t factorized_operator_benchmark_graph_generation = 0;
std::uint64_t factorized_operator_benchmark_evaluation_epoch = 0;
std::uint64_t factorized_completed_evaluation_epoch_counter = 0;
std::uint64_t factorized_completed_evaluation_epoch = 0;
std::uint64_t factorized_completed_evaluation_graph_generation = 0;
int factorized_operator_benchmark_num_nodes = 0;
int factorized_operator_benchmark_num_edges = 0;
Kokkos::View<double*> factorized_operator_benchmark_xyz;
Kokkos::View<double*> factorized_operator_benchmark_r;
Kokkos::View<Precision***,Kokkos::LayoutRight>
    factorized_operator_benchmark_a1_adjoint;
Kokkos::View<Precision**,Kokkos::LayoutRight>
    factorized_operator_benchmark_phi;
Kokkos::View<Precision**,Kokkos::LayoutRight>
    factorized_operator_benchmark_dphi_dr;
std::size_t factorized_schedule_build_count = 0;
std::size_t factorized_schedule_bytes = 0;
std::size_t factorized_state_workspace_bytes = 0;
std::size_t factorized_radial_workspace_bytes = 0;
std::size_t factorized_coupling_workspace_bytes = 0;
std::size_t factorized_coupling_capacity_bytes = 0;
std::size_t factorized_compact_workspace_bytes = 0;
std::size_t factorized_workspace_bytes = 0;
std::vector<std::string> model_load_phase_names;
std::vector<double> model_load_phase_ms;
std::size_t factorized_workspace_capacity_bytes = 0;
bool factorized_observer_enabled = false;
bool factorized_observer_ready = false;
std::size_t factorized_observer_max_bytes = 256u*1024u*1024u;
std::size_t factorized_observer_bytes = 0;
int factorized_observer_num_nodes = 0;
int factorized_observer_num_edges = 0;
std::vector<std::vector<Precision>> factorized_observer_state;
std::vector<std::vector<Precision>> factorized_observer_state_adjoint;
std::vector<Precision> factorized_observer_output;
std::vector<Precision> factorized_observer_output_adjoint;
std::vector<Precision> standard_r0_observer_output;
std::vector<Precision> standard_r0_observer_output_adjoint;
std::vector<Precision> factorized_observer_source_h1_delta;
std::vector<Precision> factorized_observer_source_h1_before;
std::vector<double> factorized_observer_directed_force_delta;
std::vector<double> factorized_observer_directed_force_before;
std::vector<double> standard_r0_observer_directed_force_delta;
std::vector<double> standard_r0_observer_directed_force_before;
bool execution_parameter_gradients_enabled = false;
bool execution_parameter_gradients_ready = false;
std::size_t execution_parameter_gradients_max_bytes = 256u*1024u*1024u;
std::size_t execution_parameter_gradients_result_bytes = 0;
std::size_t execution_parameter_gradients_workspace_bytes = 0;
double execution_parameter_gradients_r1_ms = 0.0;
double execution_parameter_gradients_r0_ms = 0.0;
double execution_parameter_gradients_density_ms = 0.0;
std::size_t execution_parameter_gradients_r1_workers = 0;
std::size_t execution_parameter_gradients_r0_workers = 0;
std::vector<ExecutionParameterGradientGroup> execution_parameter_gradient_groups;

enum class MH0StatePolicy {
    full_retention,
    reuse_adjoints,
};
MH0StatePolicy mh0_state_policy = MH0StatePolicy::full_retention;
std::string mh0_state_policy_request = "full-retention-v1";
bool readout_recompute = false;
Kokkos::View<double*> mh0_a1_scale_adjoint;
Kokkos::View<double*> mh0_a0_scale_adjoint;
bool use_mh0_adjoint_reuse() const;
bool use_h1_m0_adjoint_ping_pong() const;
void prepare_mh0_state_policy_views();
ExecutionParameterGradientGroup& execution_parameter_gradient_group(
    const std::string& name);
void begin_execution_parameter_gradients();
void admit_execution_parameter_gradient_scratch(std::size_t scratch_bytes);
void compute_factorized_parameter_gradients(
    int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r);
void compute_standard_r0_parameter_gradients(
    int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r);
void compute_execution_density_parameter_gradients(
    const std::string& network,
    int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r,
    Kokkos::View<Precision***,Kokkos::LayoutRight> output,
    Kokkos::View<Precision***,Kokkos::LayoutRight> output_adjoint);
void begin_factorized_observation(int num_nodes, int num_edges);
void update_factorized_workspace_accounting();
void refresh_factorized_workspace_readiness();
std::size_t estimate_factorized_state_workspace_bytes(
    int chunk_size,
    int num_nodes = 0) const;
std::size_t estimate_factorized_factorized_core_workspace_bytes() const;
ExecutionScratchMatrix factorized_state_view(int l) const;
ExecutionScratchMatrix factorized_output_view(int l) const;
ExecutionScratchTensor factorized_packed_state_view() const;
ExecutionScratchMatrix factorized_compact_message_view() const;
ExecutionScratchTensor factorized_source_compensation_view() const;
int select_factorized_chunk_size(
    int num_receivers,
    int num_feature_nodes,
    int num_edges,
    const std::vector<int>& num_neigh);
void resize_factorized_tile_workspace(int chunk_size, int num_nodes = 0);
void ensure_factorized_stateful_workspace();
void release_factorized_stateful_workspace();
void plan_factorized_reverse_cache();
void ensure_factorized_coupling_workspace(int coupling_columns);
void release_factorized_coupling_workspace();
void capture_factorized_state_tile(
    int l,
    int receiver_begin,
    int receiver_count,
    ExecutionScratchTensor packed_state);
void capture_factorized_state_tile(
    int l,
    int receiver_begin,
    int receiver_count,
    ExecutionScratchMatrix state);
void capture_factorized_state_adjoint_tile(
    int l,
    int receiver_begin,
    int receiver_count,
    ExecutionScratchTensor packed_state_adjoint);
void capture_factorized_state_adjoint_tile(
    int l,
    int receiver_begin,
    int receiver_count,
    ExecutionScratchMatrix state_adjoint);
void capture_factorized_output(int num_nodes);
void capture_standard_r0_output(int num_nodes);
void begin_factorized_reverse_observation(int num_nodes, int num_edges);
void finish_factorized_reverse_observation();
void begin_standard_r0_reverse_observation(int num_nodes, int num_edges);
void finish_standard_r0_reverse_observation();
void prepare_factorized_model(
    const std::vector<CompactRadialFactorizedPairTables>& pair_tables,
    double h,
    double x0);
void prepare_factorized_schedule(
    int num_nodes,
    int num_edges,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices);
void prepare_factorized_schedule(
    int num_receivers,
    int num_feature_nodes,
    int num_edges,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices);
bool prepare_factorized_schedule_host(
    int num_nodes,
    const std::vector<int>& num_neigh,
    const std::vector<int>& neigh_indices);
bool prepare_factorized_schedule_host(
    int num_receivers,
    int num_feature_nodes,
    const std::vector<int>& num_neigh,
    const std::vector<int>& neigh_indices);
void release_dual_layer_source_schedule();
bool reserve_factorized_graph_device_capacity(
    int num_receivers,
    int num_feature_nodes,
    int num_edges,
    std::size_t source_chunk_offset_count,
    std::size_t source_edge_count);
void invalidate_factorized_operator_benchmark();
void begin_factorized_production_evaluation();
void complete_factorized_prepared_evaluation(std::uint64_t graph_generation);
void invalidate_factorized_prepared_graph();
int resolve_execution_persistent_blocks(
    std::string_view implementation_id,
    const Kokkos::DefaultExecutionSpace& execution_space,
    std::size_t work_items,
    std::string_view stage);
int apply_kernel_launch_profile(
    const symmetrix::execution::LaunchProfile* profile,
    std::string_view implementation_id,
    std::string_view implementation_kind,
    bool host,
    int compute_units,
    int max_threads_per_block,
    std::size_t work_items,
    std::string_view stage);
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
int resolve_jit_device_plugin_persistent_blocks(
    const JitDevicePlugin& plugin,
    const Kokkos::DefaultExecutionSpace& execution_space,
    std::size_t work_items,
    std::string_view stage);
#endif
void invalidate_all_interactions_prepared_graph();
void reserve_execution_geometry_workspace(int num_edges);
void ensure_execution_result_capacity(
    int num_receivers, int num_feature_nodes, int num_edges);
void reserve_prepared_geometry_state_capacity(
    std::size_t num_nodes, std::size_t num_edges,
    bool shift_geometry = false);
void ensure_mh0_y_gradient_capacity(std::size_t harmonic_values);
void ensure_mh0_a0_forward_capacity(int num_nodes);
void ensure_mh0_m0_forward_capacity(int num_nodes);
void ensure_mh0_m0_adjoint_capacity();
void ensure_mh0_a1_forward_capacity(int num_nodes);
void ensure_mh0_phi1_forward_capacity(int num_nodes, int channel_count);
void ensure_mh0_m1_forward_capacity(int num_nodes);
void ensure_mh0_h2_forward_capacity(int num_nodes);
void compute_factorized(
    int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r);
bool use_factorized_direct_inference() const;
bool factorized_direct_profile_requested() const;
bool use_factorized_async_inference() const;
bool use_gpu_field_h1_reverse() const;
void complete_device_stage(const char* label);
bool use_factorized_direct_jit_forward() const;
bool use_factorized_direct_jit_reverse() const;
bool use_channel_tiled_phi1() const;
bool use_receiver_local_phi1() const;
FactorizedOperatorBenchmarkMode factorized_operator_benchmark_mode(
    const std::string& mode) const;
void run_factorized_operator_benchmark(
    FactorizedOperatorBenchmarkMode mode);
void reverse_factorized_direct(
    int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r);
void reverse_factorized(
    int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r);

// A1 rescaling
bool A1_scaled;
RadialFunctionSetKokkos<double> A1_splines;
Kokkos::View<double**,Kokkos::LayoutRight> A1_spline_values;
Kokkos::View<double**,Kokkos::LayoutRight> A1_spline_derivs;
void compute_A1_scaled(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r);
void reverse_A1_scaled(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r);

// M1
enum class M1PolynomialPolicy {
    retained,
    recompute,
};
M1PolynomialPolicy m1_polynomial_policy = M1PolynomialPolicy::retained;
bool standard_m1_module_ready = false;
std::size_t standard_m1_module_forward_launch_count = 0;
std::size_t standard_m1_module_reverse_launch_count = 0;
std::string m1_polynomial_policy_request = "automatic";
std::string m1_recompute_fallback_reason;
int m1_recompute_tile_channels = 32;
std::size_t m1_recompute_scratch_limit_override_bytes = 0;
std::size_t m1_recompute_forward_launch_count = 0;
std::size_t m1_recompute_reverse_launch_count = 0;
std::size_t macefield_response_m1_recompute_forward_launch_count = 0;
std::size_t macefield_response_m1_recompute_reverse_launch_count = 0;
std::string macefield_response_receiver_ownership = "uninitialized";
std::string macefield_response_source_ownership = "uninitialized";
std::size_t macefield_response_call_count = 0;
std::size_t macefield_response_factorized_topology_count = 0;
std::size_t macefield_response_primal_reconstruction_count = 0;
std::size_t macefield_response_phi1_fused_launch_count = 0;
std::size_t macefield_response_phi1_generic_launch_count = 0;
std::size_t macefield_response_a0_fused_launch_count = 0;
std::size_t macefield_response_a0_generic_launch_count = 0;
Kokkos::View<Precision**,Kokkos::LayoutRight> M1, M1_adj;
Kokkos::View<int**,Kokkos::LayoutRight> M1_monomials;
Kokkos::View<Precision***,Kokkos::LayoutRight> M1_weights;
Kokkos::View<int**,Kokkos::LayoutRight> M1_poly_spec;
Kokkos::View<Precision***,Kokkos::LayoutRight> M1_poly_coeff;
Kokkos::View<Precision***,Kokkos::LayoutRight> M1_poly_values;
Kokkos::View<Precision***,Kokkos::LayoutRight> M1_poly_adjoints;
void compute_M1(int num_nodes, Kokkos::View<const int*> node_types);
void reverse_M1(int num_nodes, Kokkos::View<const int*> node_types);
void release_m1_polynomial_workspace();
int macefield_response_m1_recompute_tile_channels_for(int requested) const;
std::size_t m1_recompute_required_scratch_bytes(
    int tile_channels, bool include_field_response) const;
std::size_t m1_recompute_effective_scratch_bytes(
    int tile_channels,
    bool include_field_response,
    bool require_adjoint_overlap) const;
int select_m1_recompute_tile_channels(
    int requested, bool require_adjoint_overlap = false) const;
void validate_m1_recompute_scratch(
    int requested, bool require_adjoint_overlap = false) const;
int m1_recompute_vector_length() const;
int m1_recompute_scratch_level() const;

// H2
using H2WeightPrecision = std::conditional_t<
    std::is_same_v<
        typename Kokkos::DefaultExecutionSpace::memory_space,
        Kokkos::HostSpace>,
    Precision,
    double>;
Kokkos::View<double**,Kokkos::LayoutRight> H2, H2_adj;
Kokkos::View<H2WeightPrecision**,Kokkos::LayoutRight> H2_weights_for_H1;
Kokkos::View<H2WeightPrecision*> H2_weights_for_M1;
Kokkos::View<H2WeightPrecision**,Kokkos::LayoutRight>
    H2_weights_for_H1_reverse;
Kokkos::View<H2WeightPrecision*> H2_weights_for_M1_reverse;
void compute_H2(int num_nodes, Kokkos::View<const int*> node_types);
void reverse_H2(int num_nodes, Kokkos::View<const int*> node_types, bool zero_H1_adj = true);
void ensure_h1_adjoint_capacity();

// Readouts
Kokkos::View<double*> readout_1_weights;
MultilayerPerceptronKokkos readout_2;
Kokkos::View<double*> readout_2_output;
double compute_readouts(
    int num_nodes,
    const Kokkos::View<const int*> node_types,
    bool compute_total_energy = true,
    bool completion_fence = true);

// Initializer
void load_from_json(std::string filename, const std::string& requested_head = {});

};
