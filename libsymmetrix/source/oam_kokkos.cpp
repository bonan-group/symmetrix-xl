#include "oam_kokkos.hpp"
#include "device_backend.hpp"

using symmetrix::execution::canonical_cuda_target_architecture;
using symmetrix::execution::join_cuda_r1_fields;
using symmetrix::execution::merge_cuda_r1_identity;
using symmetrix::execution::missing_cuda_r1_identity_fields;
using symmetrix::execution::read_cuda_r1_manifest_identity;

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>

#include "oam_validation.hpp"

namespace {

template <typename T>
std::vector<T> copy_span(const std::span<const T> values)
{
    return std::vector<T>(values.begin(), values.end());
}

std::size_t checked_edge_count(const std::span<const int> counts)
{
    std::size_t result = 0;
    for (const int count : counts) {
        if (count < 0)
            throw std::invalid_argument("NequIPOAMKokkos num_neigh contains a negative count.");
        const auto extent = static_cast<std::size_t>(count);
        if (result > std::numeric_limits<std::size_t>::max() - extent)
            throw std::length_error("NequIPOAMKokkos directed-edge count overflows size_t.");
        result += extent;
    }
    return result;
}

std::vector<double> tensor_values(const nlohmann::json& tensor)
{
    return tensor.at("values").get<std::vector<double>>();
}

std::string optional_irreps(const nlohmann::json& gate, const char* key)
{
    const auto value = gate.value(key, std::string{});
    return value == "0x0e" ? std::string{} : value;
}

int optional_irreps_dimension(const nlohmann::json& gate, const char* key)
{
    const auto value = optional_irreps(gate, key);
    return value.empty() ? 0 : Irreps(value).dimension();
}

template <typename T>
Kokkos::View<T*, Kokkos::HostSpace> host_view(const std::vector<T>& values)
{
    return Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
        const_cast<T*>(values.data()), values.size());
}

template <typename T>
Kokkos::View<T*> copy_device_vector(const char* label, const std::vector<T>& values)
{
    // The legacy Kokkos::View label constructor, which LAMMPS forces on,
    // accepts a std::string but not a borrowed const char*.
    Kokkos::View<T*> result(std::string(label), values.size());
    auto source = host_view(values);
    Kokkos::deep_copy(result, source);
    return result;
}

template <typename T>
void copy_device_vector_reuse(
    Kokkos::View<T*>& destination, const char* label, const std::vector<T>& values)
{
    if (destination.extent(0) < values.size())
        destination = Kokkos::View<T*>(std::string(label), values.size());
    auto target = Kokkos::subview(destination, std::make_pair<std::size_t, std::size_t>(0, values.size()));
    auto source = host_view(values);
    Kokkos::deep_copy(target, source);
}

template <typename T>
Kokkos::View<T**, Kokkos::LayoutRight> copy_device_matrix(
    const char* label, const std::vector<T>& values, const int rows, const int columns)
{
    Kokkos::View<T**, Kokkos::LayoutRight> result(std::string(label), rows, columns);
    auto source = Kokkos::View<T**, Kokkos::LayoutRight, Kokkos::HostSpace,
        Kokkos::MemoryUnmanaged>(const_cast<T*>(values.data()), rows, columns);
    Kokkos::deep_copy(result, source);
    return result;
}

// One (offset, multiplicity, width) triple per irrep block, in definition
// order.  Lane-major addressing stores element (channel a, component x) of a
// block at offset + x*multiplicity + a.
std::vector<int> lane_block_plan(const std::string& irreps_text)
{
    std::vector<int> plan;
    for (const auto& block : Irreps(irreps_text).blocks)
        plan.insert(plan.end(),
            {block.offset, block.multiplicity, 2 * block.l + 1});
    return plan;
}

Kokkos::View<int**, Kokkos::LayoutRight> copy_block_plan(
    const char* label, const std::vector<int>& plan)
{
    if (plan.empty())
        return Kokkos::View<int**, Kokkos::LayoutRight>();
    return copy_device_matrix(label, plan, static_cast<int>(plan.size() / 3), 3);
}

struct IrrepPlanBlock {
    int offset, multiplicity, width;
};

std::vector<IrrepPlanBlock> irreps_block_plan(const std::string& irreps_text)
{
    std::vector<IrrepPlanBlock> plan;
    for (const auto& block : Irreps(irreps_text).blocks)
        plan.push_back({block.offset, block.multiplicity, 2 * block.l + 1});
    return plan;
}

bool irrep_plans_equal(const std::vector<IrrepPlanBlock>& first,
    const std::vector<IrrepPlanBlock>& second)
{
    return first.size() == second.size()
        && std::equal(first.begin(), first.end(), second.begin(),
            [](const IrrepPlanBlock& a, const IrrepPlanBlock& b) {
                return a.offset == b.offset
                    && a.multiplicity == b.multiplicity
                    && a.width == b.width;
            });
}

// For every block of `current`, the consecutive blocks of `want` that exactly
// tile it.  `want` must be an adjacent same-angular aggregation refinement of
// `current`, so both descriptions keep the node-major element order.
std::vector<std::vector<int>> irrep_plan_cover(
    const std::vector<IrrepPlanBlock>& current,
    const std::vector<IrrepPlanBlock>& want)
{
    std::vector<std::vector<int>> cover;
    cover.reserve(current.size());
    std::size_t cursor = 0;
    int expected_offset = 0;
    for (const auto& block : current) {
        expected_offset = block.offset;
        int remaining = block.multiplicity * block.width;
        std::vector<int> group;
        while (remaining > 0) {
            if (cursor >= want.size())
                throw std::invalid_argument(
                    "NequIPOAMKokkos linear decomposition is not an adjacent "
                    "same-angular aggregation of the producing tensor.");
            const auto& fine = want[cursor];
            if (fine.width != block.width || fine.offset != expected_offset)
                throw std::invalid_argument(
                    "NequIPOAMKokkos linear decomposition is not an adjacent "
                    "same-angular aggregation of the producing tensor.");
            group.push_back(static_cast<int>(cursor));
            expected_offset += fine.multiplicity * fine.width;
            remaining -= fine.multiplicity * fine.width;
            ++cursor;
        }
        if (remaining != 0)
            throw std::invalid_argument(
                "NequIPOAMKokkos linear decomposition is not an adjacent "
                "same-angular aggregation of the producing tensor.");
        cover.push_back(std::move(group));
    }
    if (cursor != want.size())
        throw std::invalid_argument(
            "NequIPOAMKokkos linear decomposition is not an adjacent "
            "same-angular aggregation of the producing tensor.");
    return cover;
}

// Rewrite a linear definition so its input and output block decomposition
// match the tensors it consumes and produces.
//
// The lane-major layout is defined per irrep block, so a tensor that two
// modules describe with different (but node-major equivalent) aggregations of
// adjacent same-angular blocks would otherwise receive two different lane
// layouts.  Splitting or merging the linear's blocks and re-slicing its weight
// matrix is exact: the node-major element order - and therefore every
// definition-order bias, mask, and weight index - is unchanged.  Descriptions
// that already agree are returned untouched.
nlohmann::json normalize_linear_irreps(const nlohmann::json& linear,
    const std::string& target_in, const std::string& target_out)
{
    const auto current_in =
        irreps_block_plan(linear.at("irreps_in").get<std::string>());
    const auto current_out =
        irreps_block_plan(linear.at("irreps_out").get<std::string>());
    const auto want_in = irreps_block_plan(target_in);
    const auto want_out = irreps_block_plan(target_out);
    if (irrep_plans_equal(current_in, want_in)
        && irrep_plans_equal(current_out, want_out))
        return linear;
    const auto cover_in = irrep_plan_cover(current_in, want_in);
    const auto cover_out = irrep_plan_cover(current_out, want_out);
    const auto& weight = linear.at("weight").at("values");
    nlohmann::json instructions = nlohmann::json::array();
    nlohmann::json values = nlohmann::json::array();
    std::size_t cursor = 0;
    for (const auto& instruction : linear.at("instructions")) {
        const int input_block = instruction.at("i_in").get<int>();
        const int output_block = instruction.at("i_out").get<int>();
        const auto shape = instruction.at("path_shape").get<std::vector<int>>();
        const int rows = shape.at(0);
        const int columns = shape.at(1);
        int row = 0;
        for (const int fine_in : cover_in.at(input_block)) {
            const int fine_rows = want_in.at(fine_in).multiplicity;
            int column = 0;
            for (const int fine_out : cover_out.at(output_block)) {
                const int fine_columns = want_out.at(fine_out).multiplicity;
                for (int r = 0; r < fine_rows; ++r)
                    for (int c = 0; c < fine_columns; ++c)
                        values.push_back(weight.at(
                            cursor + static_cast<std::size_t>(row + r) * columns
                                + static_cast<std::size_t>(column + c)));
                instructions.push_back({
                    {"i_in", fine_in}, {"i_out", fine_out},
                    {"path_shape", {fine_rows, fine_columns}},
                    {"path_weight", instruction.value("path_weight", 1.0)}});
                column += fine_columns;
            }
            row += fine_rows;
        }
        cursor += static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns);
    }
    nlohmann::json result = linear;
    result["irreps_in"] = target_in;
    result["irreps_out"] = target_out;
    result["instructions"] = std::move(instructions);
    result["weight"] = {
        {"shape", {static_cast<int>(values.size())}},
        {"values", std::move(values)}};
    return result;
}

}  // namespace

// Load the generated host artifact and bind the layer owners it publishes.
// The constructor and the neutral artifact loader share this path, so a
// reload after construction revalidates the same manifest identity and
// relinks the same owners.
template <typename Precision>
void NequIPOAMKokkos<Precision>::load_host_artifact(
    const std::string& host_plugin_path)
{
    const auto& schema = model_.schema();
    symmetrix::execution::OAMHostPluginExpectation expectation;
    std::string artifact_id;
    std::string generation_fingerprint;
    std::string semantic_fingerprint;
    std::string structure_fingerprint;
    std::string schedule_identity;
    std::string host_target;
    std::string required_isa, full_reverse_reduction_policy;
    std::string schedule_qualification, forward_schedule;
    std::string full_reverse_schedule;
    std::string tiling_policy, vector_policy, register_budget_policy;
    std::uint32_t scalar_width = 0;
    std::string contract_fingerprint;
    int host_channel_tile = -1;
    std::uint32_t metadata_abi_version = 0;
    const auto manifest_path =
        std::filesystem::path(host_plugin_path).parent_path() / "manifest.json";
    std::ifstream manifest_stream(manifest_path);
    if (!manifest_stream)
        throw std::invalid_argument(
            "NequIP OAM host plugin manifest is missing: " + manifest_path.string());
    nlohmann::json manifest;
    try {
        manifest_stream >> manifest;
        const auto& metadata = manifest.at("key_inputs").at("build").at("metadata");
        metadata_abi_version = metadata.at("abi_version").get<std::uint32_t>();
        if (metadata_abi_version >= 3 && metadata_abi_version <= 6)
            throw std::invalid_argument(
                "stale OAM host artifact ABI; regenerate for source-owned ABI v7");
        if (metadata_abi_version != 1 && metadata_abi_version != 2
            && metadata_abi_version != 7)
            throw std::invalid_argument("unsupported OAM host artifact ABI");
        artifact_id = metadata.at("artifact_id").get<std::string>();
        generation_fingerprint = metadata.at("generation_fingerprint").get<std::string>();
        semantic_fingerprint = metadata.at("semantic_fingerprint").get<std::string>();
        structure_fingerprint = metadata.at("structure_fingerprint").get<std::string>();
        if (metadata_abi_version == 7) {
            schedule_identity = metadata.at("schedule_identity").get<std::string>();
            host_target = metadata.at("host_target").get<std::string>();
            host_channel_tile = metadata.at("channel_tile").get<int>();
            scalar_width = metadata.at("scalar_width").get<std::uint32_t>();
            required_isa = metadata.at("required_isa").get<std::string>();
            full_reverse_reduction_policy =
                metadata.at("full_reverse_reduction_policy").get<std::string>();
            schedule_qualification = metadata.at("schedule_qualification").get<std::string>();
            if (metadata.at("backend").get<std::string>() != "host"
                || metadata.at("execution_variants")
                    != nlohmann::json::array({"full_message"})
                || metadata.at("ownership") != nlohmann::json({
                    {"forward", "receiver"},
                    {"reverse", "source"}}))
                throw std::invalid_argument(
                    "full-message ABI v7 execution variant, backend, or "
                    "ownership metadata is invalid");
            forward_schedule = metadata.at("forward_owner_schedule").get<std::string>();
            full_reverse_schedule =
                metadata.at("full_reverse_owner_schedule").get<std::string>();
            tiling_policy = metadata.at("tiling_policy").get<std::string>();
            vector_policy = metadata.at("vector_policy").get<std::string>();
            register_budget_policy = metadata.at("register_budget_policy").get<std::string>();
            const bool whole = forward_schedule == "receiver-owner-v1"
                && full_reverse_schedule == "source-owner-v1"
                && tiling_policy == "whole-owner-v1";
            const bool tiled =
                forward_schedule == "receiver-channel-tiled-owner-v1"
                && full_reverse_schedule == "source-channel-tiled-owner-v1"
                && tiling_policy == "channel-tiled-owner-v1";
            if (!whole && !tiled)
                throw std::invalid_argument(
                    "host owner and tiling policies are inconsistent");
            if (metadata.at("shared_memory_policy").get<std::string>() != "none"
                || metadata.at("direct_geometry_policy").get<std::string>()
                    != "evaluator-harmonic-gradient-reduction-v1"
                || full_reverse_reduction_policy
                    != "owner-local-deterministic-v1"
                || metadata.at("channel_tile").get<int>() <= 0
                || scalar_width != sizeof(Precision))
                throw std::invalid_argument(
                    "host generated schedule resource policy is invalid");
            if (metadata.at("required_isa").get<std::string>().empty()
                || metadata.at("vector_policy").get<std::string>().empty()
                || metadata.at("register_budget_policy").get<std::string>().empty())
                throw std::invalid_argument(
                    "host generated schedule identity fields are empty");
            if (schedule_identity.empty() || host_target.empty())
                throw std::invalid_argument(
                    "full-message ABI v7 metadata is empty or incomplete");
        }
        contract_fingerprint = metadata.value("contract_fingerprint", std::string{});
    } catch (const std::exception& error) {
        throw std::invalid_argument(
            "NequIP OAM host plugin manifest has invalid identity metadata: "
            + std::string(error.what()) + "; regenerate the OAM artifact.");
    }
    // Generated callbacks bake the layer/irrep/tensor-product architecture,
    // not learned tensors or archive provenance.  Reproduce that exact
    // architecture identity from the loaded model before admitting a plugin.
    const std::string architecture_fingerprint =
        schema.generated_architecture_fingerprint();
    const auto& published_architecture = contract_fingerprint.empty()
        ? semantic_fingerprint : contract_fingerprint;
    if (published_architecture != architecture_fingerprint)
        throw std::invalid_argument(
            "NequIP OAM host plugin was generated for a different architecture: "
            "manifest architecture fingerprint=" + published_architecture
            + ", model architecture fingerprint=" + architecture_fingerprint
            + " (manifest " + manifest_path.string() + ").");
    if (metadata_abi_version == 7) {
        if (host_active_target_.empty())
            throw std::invalid_argument(
                "NequIP OAM ABI v7 loading requires an independently "
                "computed active host target; regenerate or prepare the artifact again.");
        if (host_active_target_ != host_target)
            throw std::invalid_argument(
                "NequIP OAM host artifact target is incompatible with the active "
                "runtime: required=" + host_target + ", active="
                + host_active_target_ + "; regenerate the OAM artifact.");
        if (host_artifact_origin_ != "prebuilt"
            && host_artifact_origin_ != "cache"
            && host_artifact_origin_ != "runtime")
            throw std::invalid_argument(
                "NequIP OAM ABI v7 loading requires artifact origin "
                "prebuilt, cache, or runtime.");
    }
    expectation.artifact_id = artifact_id;
    expectation.generation_fingerprint = generation_fingerprint;
    expectation.semantic_fingerprint = semantic_fingerprint;
    expectation.structure_fingerprint = structure_fingerprint;
    host_artifact_generation_fingerprint_ = generation_fingerprint;
    host_artifact_semantic_fingerprint_ = semantic_fingerprint;
    host_artifact_structure_fingerprint_ = structure_fingerprint;
    expectation.scalar_size = sizeof(Precision);
    expectation.required_capabilities =
        SYMMETRIX_JIT_OAM_HOST_FORWARD_OWNER_V1
        | SYMMETRIX_JIT_OAM_HOST_REVERSE_OWNER_V1;
    expectation.layers.reserve(schema.layers.size());
    for (const auto& data : schema.layers) {
        const auto source_dimension = Irreps(data.node_irreps_in).dimension();
        const auto edge_dimension = Irreps(data.edge_irreps).dimension();
        const auto output_dimension = Irreps(data.tp_output_irreps).dimension();
        const auto weight_dimension = data.edge_tensor_product.value("weight_numel", 0);
        const auto instruction_count = data.edge_tensor_product.contains("instructions")
            ? static_cast<int>(data.edge_tensor_product.at("instructions").size()) : 0;
        expectation.layers.push_back({
            source_dimension, edge_dimension, weight_dimension,
            output_dimension, instruction_count});
    }
    // ABI probe: prefer the receiver-owned v2 contract and fall back to the
    // edge-owned v1 artifact when the file only exports the v2 query symbol or
    // the v1 query symbol.  The requested execution mode decides which of the
    // two loaded contracts may actually execute.
    symmetrix::execution::OAMHostPluginExpectationV2 expectation_v2;
    expectation_v2.artifact_id = artifact_id;
    expectation_v2.generation_fingerprint = generation_fingerprint;
    expectation_v2.semantic_fingerprint = semantic_fingerprint;
    expectation_v2.structure_fingerprint = structure_fingerprint;
    expectation_v2.scalar_size = sizeof(Precision);
    expectation_v2.required_capabilities =
        SYMMETRIX_JIT_OAM_HOST_FULL_FORWARD_V3
        | SYMMETRIX_JIT_OAM_HOST_FULL_REVERSE_V3
        | SYMMETRIX_JIT_OAM_HOST_LANE_MAJOR_V3;
    expectation_v2.schedule_identity = schedule_identity;
    expectation_v2.host_target = host_target;
    expectation_v2.scalar_width = scalar_width;
    expectation_v2.required_isa = required_isa;
    expectation_v2.full_reverse_reduction_policy =
        full_reverse_reduction_policy;
    expectation_v2.schedule_qualification = schedule_qualification;
    expectation_v2.forward_owner_schedule = forward_schedule;
    expectation_v2.full_reverse_owner_schedule = full_reverse_schedule;
    expectation_v2.tiling_policy = tiling_policy;
    expectation_v2.vector_policy = vector_policy;
    expectation_v2.register_budget_policy = register_budget_policy;
    expectation_v2.layers.reserve(schema.layers.size());
    for (const auto& data : schema.layers) {
        const auto source_irreps =
            data.pre_linear.at("irreps_out").template get<std::string>();
        const auto source_dimension = Irreps(source_irreps).dimension();
        const auto edge_dimension = Irreps(data.edge_irreps).dimension();
        const auto output_dimension = Irreps(data.tp_output_irreps).dimension();
        const auto weight_dimension = data.edge_tensor_product.value("weight_numel", 0);
        const auto instruction_count = data.edge_tensor_product.contains("instructions")
            ? static_cast<int>(data.edge_tensor_product.at("instructions").size()) : 0;
        expectation_v2.layers.push_back({
            source_dimension, edge_dimension, weight_dimension,
            output_dimension, instruction_count, host_channel_tile,
            output_dimension});
    }
    std::string receiver_owned_probe_error;
    try {
        // An empty result means the file is not an ABI v7 artifact; every
        // other rejection throws and is reported with the v1 fallback error.
        auto receiver_owned =
            symmetrix::execution::OAMHostPluginV2Handle::load(
                host_plugin_path, expectation_v2);
        if (receiver_owned)
            oam_host_plugin_v2_ =
                std::make_unique<symmetrix::execution::OAMHostPluginV2Handle>(
                    std::move(*receiver_owned));
    } catch (const std::exception& error) {
        receiver_owned_probe_error = error.what();
    }
    if (!oam_host_plugin_v2_) {
        try {
            oam_host_plugin_ = std::make_unique<symmetrix::execution::OAMHostPlugin>(
                symmetrix::execution::OAMHostPlugin::load(
                    host_plugin_path, expectation));
        } catch (const std::exception& error) {
            throw std::invalid_argument(
                "NequIP OAM Kokkos could not load a receiver/source-owned (ABI v7) or "
                "edge-owned (ABI v1) host plugin: " + std::string(error.what())
                + "; ABI v7 probe: " + receiver_owned_probe_error);
        }
    }
    bind_host_artifact_owners();
}

// Point every layer at the callback owners of the artifact handles that are
// currently loaded.  A reload replaces the handles, so the owners are
// rebound rather than captured once at construction.
template <typename Precision>
void NequIPOAMKokkos<Precision>::bind_host_artifact_owners()
{
    const SymmetrixJitOAMHostPluginV1* const plugin_descriptor =
        oam_host_plugin_ ? &oam_host_plugin_->descriptor() : nullptr;
    for (std::size_t layer_index = 0; layer_index < layer_states_.size();
         ++layer_index) {
        auto& state = layer_states_[layer_index];
        state.receiver_forward_owner = nullptr;
        state.source_reverse_owner = nullptr;
        state.generated_forward_owner = nullptr;
        state.generated_reverse_owner = nullptr;
        state.generated_channel_tile = 1;
        state.generated_channel_tiled = false;
        if (oam_host_plugin_v2_) {
            const auto& receiver_owned = oam_host_plugin_v2_->descriptor();
            state.generated_channel_tile =
                receiver_owned.layers[layer_index].channel_tile;
            state.generated_channel_tiled = std::string_view(
                receiver_owned.tiling_policy) == "channel-tiled-owner-v1";
            state.receiver_forward_owner =
                receiver_owned.full_forward_owners[layer_index];
            state.source_reverse_owner =
                receiver_owned.full_reverse_owners[layer_index];
            if (state.receiver_forward_owner == nullptr
                || state.source_reverse_owner == nullptr)
                throw std::invalid_argument(
                    "NequIPOAMKokkos receiver-owned callback is null for layer "
                    + std::to_string(layer_index) + ".");
        } else if (plugin_descriptor != nullptr) {
            state.generated_forward_owner =
                plugin_descriptor->forward_owners[layer_index];
            if (state.generated_forward_owner == nullptr)
                throw std::invalid_argument(
                    "NequIPOAMKokkos generated forward owner is null for layer "
                    + std::to_string(layer_index) + ".");
            state.generated_reverse_owner =
                plugin_descriptor->reverse_owners[layer_index];
            if (state.generated_reverse_owner == nullptr)
                throw std::invalid_argument(
                    "NequIPOAMKokkos generated reverse owner is null for layer "
                    + std::to_string(layer_index) + ".");
        }
    }
}

template <typename Precision>
NequIPOAMKokkos<Precision>::NequIPOAMKokkos(
    const std::string& filename, const std::string& host_plugin_path,
    const std::string& cuda_plugin_path,
    const symmetrix::execution::CudaR1ArtifactIdentity& cuda_artifact_identity,
    std::string host_active_target, std::string host_artifact_origin,
    std::string requested_profile, std::string requested_edge_variant,
    std::string requested_precision)
    : model_(filename),
      host_active_target_(std::move(host_active_target)),
      host_artifact_origin_(std::move(host_artifact_origin))
{
    const auto& schema = model_.schema();
    if (!requested_profile.empty()) {
        const std::string actual_precision = sizeof(Precision) == 4
            ? "float32" : "float64";
        if (requested_precision != actual_precision)
            throw std::invalid_argument(
                "OAM direct preparation precision context is inconsistent: requested="
                + requested_precision + ", native=" + actual_precision);
        if (requested_edge_variant != "full_message")
            throw std::invalid_argument("OAM direct preparation edge variant is invalid");
        execution_plan_.requested_algorithm = ExecutionMode::direct;
        execution_plan_.requested_profile =
            symmetrix::execution::parse_execution_profile(requested_profile);
        execution_plan_.selection_source = "native_preparation_request";
        streamed_edges_mode_ = "direct";
        execution_edge_variant_request_ = requested_edge_variant;
    }
    if (!host_plugin_path.empty()) {
        try {
            load_host_artifact(host_plugin_path);
        } catch (const std::invalid_argument& error) {
            if (requested_profile.empty()) throw;
            const std::string candidate = requested_profile == "capacity"
                ? "oam-direct-capacity" : "oam-direct-speed";
            throw std::invalid_argument(
                "OAM native artifact rejection for requested candidate=" + candidate
                + ", profile=" + requested_profile
                + ", variant=" + requested_edge_variant
                + ", precision=" + requested_precision + ": " + error.what());
        }
    }
    radial_sets_.reserve(schema.layers.size());
    for (std::size_t layer = 0; layer < schema.layers.size(); ++layer) {
        const auto& contract = schema.kokkos_radial.at(layer);
        radial_sets_.emplace_back(
            schema, static_cast<int>(layer), contract.num_points, contract.grid_min);
    }
    const auto& embedding = schema.node_attributes;
    const auto embedding_shape = embedding.at("shape").get<std::vector<int>>();
    if (embedding_shape.size() != 2 || embedding_shape[0] <= 0 || embedding_shape[1] <= 0)
        throw std::invalid_argument("NequIPOAMKokkos node attributes have an invalid shape.");
    node_attribute_width_ = embedding_shape[1];
    node_attributes_host_ = tensor_values(embedding);
    if (node_attributes_host_.size() != static_cast<std::size_t>(embedding_shape[0])
            * static_cast<std::size_t>(node_attribute_width_))
        throw std::invalid_argument("NequIPOAMKokkos node attributes have invalid values.");
    std::vector<Precision> node_attributes_precision(node_attributes_host_.size());
    std::transform(node_attributes_host_.begin(), node_attributes_host_.end(),
        node_attributes_precision.begin(), [](const double value) {
            return static_cast<Precision>(value);
        });
    node_attributes_ = copy_device_matrix(
        "oam node attributes", node_attributes_precision,
        embedding_shape[0], node_attribute_width_);
    if (schema.layers.empty()
        || Irreps(schema.layers.front().node_irreps_in).dimension()
            != node_attribute_width_)
        throw std::invalid_argument(
            "NequIPOAMKokkos node attributes are inconsistent with the first layer "
            "node irreps.");
    node_feature_block_plan_ = copy_block_plan(
        "oam node feature block plan",
        lane_block_plan(schema.layers.front().node_irreps_in));
    readout_ = E3LinearKokkosT<Precision>(normalize_linear_irreps(
        schema.readout, schema.layers.back().node_irreps_out,
        schema.readout.at("irreps_out").get<std::string>()));
    readout_.set_workspace(linear_workspace_);

    type_to_model_host_.resize(schema.atomic_numbers.size(), -1);
    for (std::size_t type = 0; type < schema.atomic_numbers.size(); ++type) {
        const auto it = std::find(schema.model_atomic_numbers.begin(),
            schema.model_atomic_numbers.end(), schema.atomic_numbers[type]);
        if (it == schema.model_atomic_numbers.end())
            throw std::invalid_argument("NequIPOAMKokkos atomic-number mapping is incomplete.");
        const auto model_index = static_cast<std::size_t>(it - schema.model_atomic_numbers.begin());
        if (model_index >= schema.atomic_number_to_model_type.size())
            throw std::invalid_argument("NequIPOAMKokkos atomic-number mapping is invalid.");
        type_to_model_host_[type] = schema.atomic_number_to_model_type[model_index];
    }
    type_to_model_ = copy_device_vector("oam type-to-model map", type_to_model_host_);
    per_type_shift_ = schema.per_type_shift;
    per_type_scale_ = schema.per_type_scale;
    model_atomic_numbers_ = schema.model_atomic_numbers;
    const auto& zbl = schema.zbl;
    zbl_exponents_ = zbl.at("screening_exponents").get<std::vector<double>>();
    zbl_coefficients_ = zbl.at("screening_coefficients").get<std::vector<double>>();
    zbl_cutoff_ = zbl.at("cutoff").get<double>();
    // OAM contracts before the explicit cutoff_p field used the edge
    // embedding polynomial exponent for the ZBL envelope. Keep that
    // backward-compatible default aligned with the serial evaluator.
    zbl_cutoff_p_ = zbl.value(
        "cutoff_p", schema.edge_embedding.at("cutoff").at("p").get<double>());
    zbl_half_prefactor_ = zbl.at("half_energy_prefactor").get<double>();
    per_type_shift_device_ = copy_device_vector(
        "oam per-type shifts", per_type_shift_);
    per_type_scale_device_ = copy_device_vector(
        "oam per-type scales", per_type_scale_);
    // ZBL energy and its derivative are double expressions even for FP32
    // models. Do not round the contract coefficients to model precision.
    zbl_exponents_device_ = copy_device_vector(
        "oam zbl screening exponents", zbl_exponents_);
    zbl_coefficients_device_ = copy_device_vector(
        "oam zbl screening coefficients", zbl_coefficients_);
    {
        const std::size_t types = model_atomic_numbers_.size();
        std::vector<double> pair_zsum(types * types);
        std::vector<double> pair_prefactor(types * types);
        for (std::size_t u = 0; u < types; ++u) {
            const double zi = static_cast<double>(model_atomic_numbers_[u]);
            for (std::size_t v = 0; v < types; ++v) {
                const double zj = static_cast<double>(model_atomic_numbers_[v]);
                const std::size_t pair = u * types + v;
                pair_zsum[pair] = std::pow(zi, 0.23) + std::pow(zj, 0.23);
                pair_prefactor[pair] = zbl_half_prefactor_ * zi * zj;
            }
        }
        zbl_pair_zsum_device_ = copy_device_vector(
            "oam zbl pair screening sums", pair_zsum);
        zbl_pair_prefactor_device_ = copy_device_vector(
            "oam zbl pair prefactors", pair_prefactor);
    }

    layer_states_.reserve(schema.layers.size());
    for (std::size_t layer_index = 0; layer_index < schema.layers.size(); ++layer_index) {
        const auto& data = schema.layers[layer_index];
        LayerState state;
        state.pre_linear = E3LinearKokkosT<Precision>(normalize_linear_irreps(
            data.pre_linear, data.node_irreps_in, data.node_irreps_in));
        state.pre_linear.set_workspace(linear_workspace_);
        state.generated_layer = static_cast<std::uint32_t>(layer_index);
        state.post_linear = E3LinearKokkosT<Precision>(normalize_linear_irreps(
            data.post_linear, data.tp_output_irreps, data.node_irreps_pre_gate));
        state.post_linear.set_workspace(linear_workspace_);
        // The portable edge product.  Its declared irreps are the layer's own
        // by contract -- the schema requires irreps_in1 == node_irreps_in,
        // irreps_in2 == edge_irreps, and irreps_out == tp_output_irreps -- so
        // the node-major layout the generic tensor product addresses is the
        // definition order of the plans this layer already built.
        state.edge_tensor_product =
            std::make_unique<E3TensorProductKokkosT<Precision>>(
                data.edge_tensor_product);
        if (!data.self_connection.is_null()) {
            if (data.self_connection.contains("irreps_in1"))
                state.self_tensor_product = std::make_unique<E3TensorProductKokkosT<Precision>>(
                    data.self_connection);
            else
                state.self_linear = std::make_unique<E3LinearKokkosT<Precision>>(
                    normalize_linear_irreps(data.self_connection,
                        data.node_irreps_in, data.node_irreps_pre_gate));
            if (state.self_linear)
                state.self_linear->set_workspace(linear_workspace_);
            if (state.self_tensor_product)
                state.self_tensor_product->set_bilinear_workspace(
                    bilinear_workspace_);
        }
        if (state.self_tensor_product)
            state.self_attribute_block_plan = copy_block_plan(
                "oam self attribute block plan",
                lane_block_plan(
                    data.self_connection.at("irreps_in2").get<std::string>()));
        state.input_width = Irreps(data.node_irreps_in).dimension();
        state.input_block_plan = copy_block_plan(
            "oam layer input block plan", lane_block_plan(data.node_irreps_in));
        state.prelinear_width = state.pre_linear.output_dimension();
        state.edge_width = Irreps(data.edge_irreps).dimension();
        std::vector<int> edge_harmonic_indices;
        for (const auto& block : Irreps(data.edge_irreps).blocks)
            for (int copy = 0; copy < block.multiplicity; ++copy)
                for (int component = 0; component < 2 * block.l + 1; ++component)
                    edge_harmonic_indices.push_back(block.l * block.l + component);
        if (static_cast<int>(edge_harmonic_indices.size()) != state.edge_width)
            throw std::invalid_argument("NequIPOAMKokkos edge harmonic layout is inconsistent.");
        state.edge_harmonic_indices = copy_device_vector(
            "oam edge harmonic indices", edge_harmonic_indices);
        state.tp_output_width = Irreps(data.tp_output_irreps).dimension();
        state.pre_gate_width = Irreps(data.node_irreps_pre_gate).dimension();
        state.output_width = Irreps(data.node_irreps_out).dimension();
        state.weight_count = data.edge_tensor_product.value("weight_numel", 0);
        if (state.weight_count < 0)
            throw std::invalid_argument("NequIPOAMKokkos tensor-product weight count is invalid.");
        state.scalar_dimension = optional_irreps_dimension(data.gate, "scalar_irreps");
        state.gate_dimension = optional_irreps_dimension(data.gate, "gate_irreps");
        state.gated_dimension = optional_irreps_dimension(data.gate, "gated_irreps");
        state.normalization_factor = static_cast<Precision>(
            data.normalization.value("factor", 1.0));
        state.pre_gate_block_plan = copy_block_plan(
            "oam pre-gate block plan",
            lane_block_plan(data.node_irreps_pre_gate));
        state.tp_output_block_plan = copy_block_plan(
            "oam tp output block plan",
            lane_block_plan(data.tp_output_irreps));
        {
            const std::vector<double> mask_values =
                tensor_values(data.edge_tensor_product.at("output_mask"));
            if (mask_values.size()
                != static_cast<std::size_t>(state.tp_output_width))
                throw std::invalid_argument(
                    "NequIPOAMKokkos tensor-product output mask is inconsistent.");
            std::vector<Precision> mask(mask_values.size());
            std::transform(mask_values.begin(), mask_values.end(), mask.begin(),
                [](const double value) { return static_cast<Precision>(value); });
            state.tp_output_mask = copy_device_vector(
                "oam tp output mask", mask);
        }
        if (!state.edge_tensor_product
            || state.edge_tensor_product->input_1_dimension() != state.prelinear_width
            || state.edge_tensor_product->input_2_dimension() != state.edge_width
            || state.edge_tensor_product->output_dimension() != state.tp_output_width
            || state.edge_tensor_product->weight_size() != state.weight_count)
            throw std::invalid_argument(
                "NequIPOAMKokkos portable edge tensor-product dimensions are "
                "inconsistent.");
        if (state.self_tensor_product) {
            if (state.self_tensor_product->output_dimension() != state.pre_gate_width
                || state.self_tensor_product->input_1_dimension() != state.input_width
                || state.self_tensor_product->input_2_dimension()
                    != node_attribute_width_)
                throw std::invalid_argument(
                    "NequIPOAMKokkos self tensor-product dimensions are inconsistent.");
            // The lane kernels address every element through the tensor
            // product's own irrep blocks while its operands and its producer
            // use the runtime node plans, so they are admissible only when the
            // two block plans are identical.  The schema loader already
            // guarantees that a differing definition keeps the node-major
            // element order, which is exactly what the definition-order
            // fallback below relies on when this is false.
            state.self_lane_kernels =
                state.self_tensor_product->supports_node_bilinear_uvw_lane()
                && irrep_plans_equal(
                    irreps_block_plan(
                        data.self_connection.at("irreps_in1").get<std::string>()),
                    irreps_block_plan(data.node_irreps_in))
                && irrep_plans_equal(
                    irreps_block_plan(
                        data.self_connection.at("irreps_out").get<std::string>()),
                    irreps_block_plan(data.node_irreps_pre_gate));
            if (!state.self_lane_kernels)
                self_definition_order_layers_.push_back(
                    static_cast<int>(layer_index));
        }
        prepare_conditioned_self_plan(state, data.self_connection);
        if (state.prelinear_width != state.input_width
            && state.pre_linear.input_dimension() != state.input_width)
            throw std::invalid_argument("NequIPOAMKokkos pre-linear dimensions are inconsistent.");
        {
            const auto gated_text = optional_irreps(data.gate, "gated_irreps");
            const auto gate_text = optional_irreps(data.gate, "gate_irreps");
            std::vector<int> gated_plan;
            std::vector<int> gate_channels;
            if (!gated_text.empty()) {
                const Irreps gated(gated_text);
                int gate_cursor = 0;
                int gated_offset = 0;
                for (const auto& block : gated.blocks) {
                    const int width = 2 * block.l + 1;
                    gated_plan.insert(gated_plan.end(),
                        {gated_offset, block.multiplicity, width});
                    gated_offset += block.multiplicity * width;
                    for (int copy = 0; copy < block.multiplicity; ++copy)
                        gate_channels.push_back(gate_text.empty()
                            ? -1 : state.scalar_dimension + gate_cursor++);
                }
                if (gated_offset != state.gated_dimension)
                    throw std::invalid_argument(
                        "NequIPOAMKokkos gate dimensions are inconsistent.");
            }
            state.gated_block_plan =
                copy_block_plan("oam gated block plan", gated_plan);
            if (!gate_channels.empty())
                state.gated_gate_channels = copy_device_vector(
                    "oam gated gate channels", gate_channels);
        }
        if (state.pre_gate_width != state.scalar_dimension + state.gate_dimension
                + state.gated_dimension
            || state.output_width != state.scalar_dimension + state.gated_dimension)
            throw std::invalid_argument("NequIPOAMKokkos gate dimensions are inconsistent.");
        layer_states_.push_back(std::move(state));
        for (const auto& block : Irreps(data.edge_irreps).blocks)
            max_edge_l_ = std::max(max_edge_l_, block.l);
    }
    bind_host_artifact_owners();
    edge_harmonic_count_ = (max_edge_l_ + 1) * (max_edge_l_ + 1);
    execution_reduction_mode_ = execution_backend() == "openmp" ? "openmp-atomic"
        : (execution_space_is_cuda() ? "cuda-atomic" : "serial");
    if (!cuda_plugin_path.empty()) {
        try {
            initialize_cuda_r1(cuda_plugin_path, cuda_artifact_identity);
            if (!requested_profile.empty()
                && !(cuda_r1_plugin_ && static_cast<bool>(*cuda_r1_plugin_)))
                throw std::invalid_argument(cuda_r1_reason_);
        } catch (const std::invalid_argument& error) {
            if (requested_profile.empty()) throw;
            const std::string candidate = requested_profile == "capacity"
                ? "oam-direct-capacity" : "oam-direct-speed";
            throw std::invalid_argument(
                "OAM native artifact rejection for requested candidate=" + candidate
                + ", profile=" + requested_profile
                + ", variant=" + requested_edge_variant
                + ", precision=" + requested_precision + ": " + error.what());
        }
    }
    // The packed contractions and reverse grid were qualified together on
    // sm120 FP32. Keep every other device and precision on its existing path.
    if constexpr (std::is_same_v<Precision, float>) {
        if (execution_space_is_cuda() && cuda_r1_plugin_ && *cuda_r1_plugin_) {
            const auto device = execution_device_execution_environment();
            qualified_sm120_fp32_ = device.available
                && device.compute_capability == 120
                && cuda_r1_identity_.target_compute_capability == 120;
            for (std::size_t layer = 0; qualified_sm120_fp32_
                    && layer < cuda_r1_plugin_->layer_count(); ++layer) {
                const auto& schedule = cuda_r1_plugin_->layer(layer);
                qualified_sm120_fp32_ = schedule.forward_threads_per_block == 128
                    && schedule.reverse_threads_per_block == 128;
            }
        }
    }
    if (qualified_sm120_fp32_) {
        readout_.set_backend("packed_gemm");
        for (auto& state : layer_states_) {
            state.pre_linear.set_backend("packed_gemm");
            state.post_linear.set_backend("packed_gemm");
            if (state.self_linear) state.self_linear->set_backend("packed_gemm");
        }
    }
    refresh_execution_state_diagnostic();
    refresh_execution_plan_report();
}
template <typename Precision>
void NequIPOAMKokkos<Precision>::prepare_conditioned_self_plan(
    LayerState& state, const nlohmann::json& self_connection)
{
    auto& plan = state.conditioned_self;
    if (!execution_space_is_cuda() || self_connection.is_null()
        || !state.self_tensor_product || !state.self_lane_kernels)
        return;

    const Irreps input_irreps(
        self_connection.at("irreps_in1").get<std::string>());
    const Irreps attribute_irreps(
        self_connection.at("irreps_in2").get<std::string>());
    const Irreps output_irreps(
        self_connection.at("irreps_out").get<std::string>());
    const auto& input_blocks = input_irreps.blocks;
    const auto& attribute_blocks = attribute_irreps.blocks;
    const auto& output_blocks = output_irreps.blocks;
    if (input_irreps.dimension() != state.input_width
        || attribute_irreps.dimension() != node_attribute_width_
        || output_irreps.dimension() != state.pre_gate_width)
        return;
    for (const auto& block : attribute_blocks)
        if (block.l != 0 || block.parity != 1)
            return;

    const auto output_mask_values = tensor_values(
        self_connection.at("output_mask"));
    if (output_mask_values.size() != static_cast<std::size_t>(state.pre_gate_width))
        return;
    const auto weight_values = tensor_values(self_connection.at("weight"));
    const int model_count = static_cast<int>(node_attributes_host_.size()
        / static_cast<std::size_t>(node_attribute_width_));
    if (model_count <= 0
        || node_attributes_host_.size()
            != static_cast<std::size_t>(model_count)
                * static_cast<std::size_t>(node_attribute_width_))
        return;

    const auto& instructions = self_connection.at("instructions");
    std::vector<std::int64_t> path_data;
    std::vector<Precision> wigner_diagonal;
    std::size_t serialized_wigner_scalars = 0;
    std::size_t coefficients_per_model = 0;
    std::size_t weight_cursor = 0;
    for (const auto& instruction : instructions) {
        if (instruction.value("connection_mode", std::string{}) != "uvw"
            || !instruction.value("has_weight", false))
            return;
        const int input_index = instruction.at("i_in1").get<int>();
        const int attribute_index = instruction.at("i_in2").get<int>();
        const int output_index = instruction.at("i_out").get<int>();
        if (input_index < 0 || attribute_index < 0 || output_index < 0
            || input_index >= static_cast<int>(input_blocks.size())
            || attribute_index >= static_cast<int>(attribute_blocks.size())
            || output_index >= static_cast<int>(output_blocks.size()))
            return;
        const auto& input_block = input_blocks[static_cast<std::size_t>(input_index)];
        const auto& attribute_block =
            attribute_blocks[static_cast<std::size_t>(attribute_index)];
        const auto& output_block = output_blocks[static_cast<std::size_t>(output_index)];
        if ((2 * attribute_block.l + 1) != 1 || attribute_block.l != 0
            || attribute_block.parity != 1
            || input_block.l != output_block.l
            || input_block.parity != output_block.parity
            || input_block.dimension() <= 0 || output_block.dimension() <= 0)
            return;
        const auto path_shape = instruction.at("path_shape").get<std::vector<int>>();
        if (path_shape != std::vector<int>{
                input_block.multiplicity, attribute_block.multiplicity,
                output_block.multiplicity})
            return;
        const auto& wigner = instruction.at("wigner_3j");
        const auto wigner_shape = wigner.at("shape").get<std::vector<int>>();
        const auto wigner_values = wigner.at("values").get<std::vector<double>>();
        serialized_wigner_scalars += wigner_values.size();
        const int width = 2 * input_block.l + 1;
        if (wigner_shape != std::vector<int>{width, 1, width}
            || wigner_values.size() != static_cast<std::size_t>(width * width))
            return;
        const std::size_t angular_offset = wigner_diagonal.size();
        for (int a = 0; a < width; ++a) {
            for (int c = 0; c < width; ++c) {
                const double value = wigner_values[static_cast<std::size_t>(a * width + c)];
                if (a != c && value != 0.0)
                    return;
            }
            const double diagonal = wigner_values[static_cast<std::size_t>(a * width + a)];
            if (!std::isfinite(diagonal))
                return;
            wigner_diagonal.push_back(static_cast<Precision>(diagonal));
        }
        const std::size_t matrix_size = static_cast<std::size_t>(
            input_block.multiplicity) * static_cast<std::size_t>(output_block.multiplicity);
        const std::size_t weight_size = static_cast<std::size_t>(
            input_block.multiplicity) * static_cast<std::size_t>(attribute_block.multiplicity)
            * static_cast<std::size_t>(output_block.multiplicity);
        if (weight_cursor > weight_values.size()
            || weight_size > weight_values.size() - weight_cursor)
            return;
        path_data.insert(path_data.end(), {
            static_cast<std::int64_t>(input_block.offset),
            static_cast<std::int64_t>(input_block.multiplicity),
            static_cast<std::int64_t>(width),
            static_cast<std::int64_t>(attribute_block.offset),
            static_cast<std::int64_t>(attribute_block.multiplicity),
            1,
            static_cast<std::int64_t>(output_block.offset),
            static_cast<std::int64_t>(output_block.multiplicity),
            static_cast<std::int64_t>(width),
            static_cast<std::int64_t>(angular_offset),
            static_cast<std::int64_t>(coefficients_per_model)});
        coefficients_per_model += matrix_size;
        weight_cursor += weight_size;
    }
    if (path_data.empty() || weight_cursor != weight_values.size())
        return;

    std::vector<Precision> coefficients(
        static_cast<std::size_t>(model_count) * coefficients_per_model,
        Precision(0));
    std::size_t weight_cursor_again = 0;
    const std::size_t path_count = path_data.size() / 11;
    for (std::size_t path = 0; path < path_count; ++path) {
        const auto& instruction = instructions.at(path);
        const auto* row = path_data.data() + path * 11;
        const int input_multiplicity = static_cast<int>(row[1]);
        const int attribute_offset = static_cast<int>(row[3]);
        const int attribute_multiplicity = static_cast<int>(row[4]);
        const int output_multiplicity = static_cast<int>(row[7]);
        const std::size_t matrix_offset = static_cast<std::size_t>(row[10]);
        for (int model = 0; model < model_count; ++model) {
            const std::size_t attribute_base = static_cast<std::size_t>(model)
                * static_cast<std::size_t>(node_attribute_width_)
                + static_cast<std::size_t>(attribute_offset);
            const std::size_t coefficient_base = static_cast<std::size_t>(model)
                * coefficients_per_model + matrix_offset;
            for (int input_channel = 0; input_channel < input_multiplicity;
                 ++input_channel)
                for (int output_channel = 0; output_channel < output_multiplicity;
                     ++output_channel) {
                    Precision value = Precision(0);
                    for (int attribute_channel = 0;
                         attribute_channel < attribute_multiplicity;
                         ++attribute_channel) {
                        const std::size_t weight_index = weight_cursor_again
                            + (static_cast<std::size_t>(input_channel)
                                * static_cast<std::size_t>(attribute_multiplicity)
                                + static_cast<std::size_t>(attribute_channel))
                                * static_cast<std::size_t>(output_multiplicity)
                            + static_cast<std::size_t>(output_channel);
                        value += static_cast<Precision>(
                            weight_values[weight_index])
                            * static_cast<Precision>(
                                node_attributes_host_[attribute_base
                                    + static_cast<std::size_t>(attribute_channel)]);
                    }
                    coefficients[coefficient_base
                        + static_cast<std::size_t>(input_channel)
                            * static_cast<std::size_t>(output_multiplicity)
                        + static_cast<std::size_t>(output_channel)] =
                        static_cast<Precision>(instruction.value("path_weight", 1.0))
                            * value;
                }
        }
        const auto path_shape = instruction.at("path_shape").get<std::vector<int>>();
        weight_cursor_again += static_cast<std::size_t>(path_shape[0])
            * static_cast<std::size_t>(path_shape[1])
            * static_cast<std::size_t>(path_shape[2]);
    }

    std::vector<Precision> masks(output_mask_values.size());
    std::transform(output_mask_values.begin(), output_mask_values.end(), masks.begin(),
        [](const double value) { return static_cast<Precision>(value); });
    plan.path_data = copy_device_matrix(
        "oam conditioned self path data", path_data,
        static_cast<int>(path_count), 11);
    plan.coefficients = copy_device_vector(
        "oam conditioned self coefficients", coefficients);
    plan.wigner_diagonal = copy_device_vector(
        "oam conditioned self wigner diagonal", wigner_diagonal);
    plan.output_mask = copy_device_vector(
        "oam conditioned self output mask", masks);
    plan.coefficients_per_model = coefficients_per_model;
    plan.path_count = static_cast<int>(path_count);
    plan.output_width = state.pre_gate_width;
    plan.original_tensor_bytes = (weight_values.size()
        + output_mask_values.size() + serialized_wigner_scalars)
        * sizeof(Precision);
    plan.eligible = true;
}


template <typename Precision>
NequIPOAMKokkos<Precision>::~NequIPOAMKokkos()
{
    Kokkos::fence("Destroy NequIP OAM host plugin");
    cuda_r1_plugin_.reset();
    oam_host_plugin_.reset();
    oam_host_plugin_v2_.reset();
}

template <typename Precision>
bool NequIPOAMKokkos<Precision>::execution_space_is_host() const
{
    return symmetrix::execution::host_execution_space<
        Kokkos::DefaultExecutionSpace>;
}

template <typename Precision>
bool NequIPOAMKokkos<Precision>::execution_space_is_cuda() const
{
    const std::string execution_space = _kokkos_default_execution_space();
    return execution_space.find("Cuda") != std::string::npos
        || execution_space.find("CUDA") != std::string::npos;
}

template <typename Precision>
std::string NequIPOAMKokkos<Precision>::execution_target() const
{
    if (execution_space_is_host()) return "host";
    if (execution_space_is_cuda()) return "cuda";
    const std::string execution_space = _kokkos_default_execution_space();
    if (execution_space.find("HIP") != std::string::npos
            || execution_space.find("Hip") != std::string::npos)
        return "hip";
    return "unsupported";
}

template <typename Precision>
std::string NequIPOAMKokkos<Precision>::cuda_r1_state_text() const
{
    if (cuda_r1_plugin_path_.empty())
        return {};
    std::string text =
        ", cuda_r1_ready=" + std::string(cuda_r1_ready_ ? "true" : "false")
        + ", cuda_r1_plugin_path=" + cuda_r1_plugin_path_
        + ", cuda_r1_artifact_id=" + cuda_r1_artifact_id_
        + ", cuda_r1_launch_count=" + std::to_string(cuda_r1_launch_count_)
        + ", cuda_r1_fallback_count=" + std::to_string(cuda_r1_fallback_count_)
        + ", cuda_r1_grid_blocks=" + std::to_string(cuda_r1_grid_blocks_);
    if (!cuda_r1_manifest_path_.empty())
        text += ", cuda_r1_manifest_path=" + cuda_r1_manifest_path_;
    if (!cuda_r1_reason_.empty())
        text += ", cuda_r1_reason=" + cuda_r1_reason_;
    return text;
}

template <typename Precision>
std::vector<symmetrix::execution::OAMCudaLayerExpectation>
NequIPOAMKokkos<Precision>::cuda_r1_layer_expectations() const
{
    const auto& schema = model_.schema();
    std::vector<symmetrix::execution::OAMCudaLayerExpectation> expectations;
    expectations.reserve(schema.layers.size());
    for (std::size_t index = 0; index < schema.layers.size(); ++index) {
        const auto& data = schema.layers[index];
        const auto& radial = radial_sets_[index].kokkos_set();
        symmetrix::execution::OAMCudaLayerExpectation row;
        row.source_dimension =
            Irreps(data.pre_linear.at("irreps_out").get<std::string>()).dimension();
        row.edge_dimension = Irreps(data.edge_irreps).dimension();
        row.weight_dimension = data.edge_tensor_product.value("weight_numel", 0);
        row.output_dimension = Irreps(data.tp_output_irreps).dimension();
        row.target_dimension = row.output_dimension;
        row.instruction_count =
            data.edge_tensor_product.contains("instructions")
            ? static_cast<int>(
                data.edge_tensor_product.at("instructions").size())
            : 0;
        row.spline_interval_count = radial.interval_count();
        row.spline_function_count = radial.function_count();
        row.edge_harmonic_count = edge_harmonic_count_;
        expectations.push_back(row);
    }
    return expectations;
}

template <typename Precision>
void NequIPOAMKokkos<Precision>::initialize_cuda_r1(
    const std::string& plugin_path,
    const symmetrix::execution::CudaR1ArtifactIdentity& identity)
{
    cuda_r1_plugin_path_ = plugin_path;
    cuda_r1_identity_ = identity;
    if (plugin_path.empty())
        return;
    if (!execution_space_is_cuda()) {
        ++cuda_r1_fallback_count_;
        cuda_r1_reason_ =
            "the active Kokkos execution space is not CUDA; backend="
            + execution_backend();
        last_execution_diagnostic_ += cuda_r1_state_text();
        return;
    }
    // The generated CUDA module is shared by models with the same concrete
    // layer architecture; learned tensors remain runtime inputs.
    const std::string architecture_fingerprint =
        model_.schema().generated_architecture_fingerprint();
    cuda_r1_manifest_path_ = cuda_r1_identity_.manifest_path;
    if (cuda_r1_manifest_path_.empty())
        cuda_r1_manifest_path_ = (
            std::filesystem::path(plugin_path).parent_path() / "manifest.json")
            .string();
    cuda_r1_identity_.manifest_path = cuda_r1_manifest_path_;
    symmetrix::execution::CudaR1ArtifactIdentity published;
    std::string manifest_diagnostic;
    const bool manifest_read = read_cuda_r1_manifest_identity(
        cuda_r1_manifest_path_, "oam", published, manifest_diagnostic);
    if (manifest_read
            && !published.contract_fingerprint.empty()
            && published.contract_fingerprint != architecture_fingerprint) {
        ++cuda_r1_fallback_count_;
        cuda_r1_reason_ =
            "the generated artifact was published for a different architecture: "
            "manifest contract_fingerprint=" + published.contract_fingerprint
            + ", model architecture fingerprint=" + architecture_fingerprint
            + " (" + cuda_r1_manifest_path_ + ")";
        last_execution_diagnostic_ += cuda_r1_state_text();
        return;
    }
    if (manifest_read) {
        const auto mismatches = cuda_r1_identity_mismatches(
            cuda_r1_identity_, published);
        if (!mismatches.empty()) {
            ++cuda_r1_fallback_count_;
            cuda_r1_reason_ =
                "the explicit CUDA artifact identity disagrees with the manifest: "
                + join_cuda_r1_fields(mismatches)
                + "; regenerate the artifact or remove the conflicting override";
            last_execution_diagnostic_ += cuda_r1_state_text();
            return;
        }
    }
    if (manifest_read)
        merge_cuda_r1_identity(cuda_r1_identity_, published);
    if (!cuda_r1_identity_.contract_fingerprint.empty()
            && cuda_r1_identity_.contract_fingerprint
                != architecture_fingerprint) {
        ++cuda_r1_fallback_count_;
        cuda_r1_reason_ =
            "the generated artifact identity was published for a different "
            "architecture: artifact contract_fingerprint="
            + cuda_r1_identity_.contract_fingerprint
            + ", model architecture fingerprint=" + architecture_fingerprint;
        last_execution_diagnostic_ += cuda_r1_state_text();
        return;
    }
    cuda_r1_identity_.contract_fingerprint = architecture_fingerprint;
    const auto device = execution_device_execution_environment();
    if (device.available) {
        if (cuda_r1_identity_.target_architecture.empty())
            cuda_r1_identity_.target_architecture =
                canonical_cuda_target_architecture(device.architecture);
        if (cuda_r1_identity_.target_compute_capability <= 0)
            cuda_r1_identity_.target_compute_capability =
                device.compute_capability;
    }
    const auto missing = missing_cuda_r1_identity_fields(cuda_r1_identity_);
    if (!missing.empty()) {
        ++cuda_r1_fallback_count_;
        std::string reason =
            "the generated artifact identity is incomplete: missing "
            + join_cuda_r1_fields(missing) + "; supply the codegen identity to "
            "the evaluator or publish it in " + cuda_r1_manifest_path_;
        if (!manifest_diagnostic.empty())
            reason += " (" + manifest_diagnostic + ")";
        cuda_r1_reason_ = std::move(reason);
        last_execution_diagnostic_ += cuda_r1_state_text();
        return;
    }
    if (device.available && device.compute_capability > 0
            && cuda_r1_identity_.target_compute_capability
                != device.compute_capability) {
        ++cuda_r1_fallback_count_;
        cuda_r1_reason_ =
            "the generated artifact targets compute capability "
            + std::to_string(cuda_r1_identity_.target_compute_capability)
            + " but the active device reports "
            + std::to_string(device.compute_capability);
        last_execution_diagnostic_ += cuda_r1_state_text();
        return;
    }
    symmetrix::execution::OAMCudaPluginExpectation expectation;
    expectation.artifact_id = cuda_r1_identity_.artifact_id;
    expectation.contract_fingerprint = cuda_r1_identity_.contract_fingerprint;
    expectation.semantic_fingerprint = cuda_r1_identity_.semantic_fingerprint;
    expectation.structure_fingerprint = cuda_r1_identity_.structure_fingerprint;
    expectation.target_architecture = cuda_r1_identity_.target_architecture;
    expectation.target_compute_capability =
        cuda_r1_identity_.target_compute_capability;
    expectation.max_threads_per_block =
        cuda_r1_identity_.max_threads_per_block;
    expectation.layers = cuda_r1_layer_expectations();
    expectation.scalar_kind = std::is_same_v<Precision, float>
        ? SYMMETRIX_OAM_CUDA_SCALAR_FLOAT32_V1
        : SYMMETRIX_OAM_CUDA_SCALAR_FLOAT64_V1;
    expectation.scalar_size = static_cast<std::uint32_t>(sizeof(Precision));
    expectation.target_policy = cuda_r1_identity_.target_policy;
    expectation.precision_policy = cuda_r1_identity_.precision_policy;
    expectation.schedule_qualification =
        cuda_r1_identity_.schedule_qualification;
    expectation.required_capabilities =
        SYMMETRIX_OAM_CUDA_FORWARD_LAUNCH_V1
        | SYMMETRIX_OAM_CUDA_REVERSE_LAUNCH_V1
        | SYMMETRIX_OAM_CUDA_DIRECT_GEOMETRY_V1;
    try {
        auto plugin = symmetrix::execution::OAMCudaPlugin::load(
            plugin_path, expectation);
        cuda_r1_artifact_id_ = std::string(plugin.artifact_id());
        cuda_r1_plugin_ =
            std::make_unique<symmetrix::execution::OAMCudaPlugin>(
                std::move(plugin));
        cuda_r1_reason_ =
            "the generated plugin is loaded and validated; awaiting execution";
    } catch (const std::exception& error) {
        cuda_r1_plugin_.reset();
        cuda_r1_artifact_id_.clear();
        cuda_r1_reason_ =
            "the ABI loader rejected the expectation: " + std::string(error.what());
        ++cuda_r1_fallback_count_;
    }
    cuda_r1_ready_ = false;
    last_execution_diagnostic_ += cuda_r1_state_text();
}

template <typename Precision>
bool NequIPOAMKokkos<Precision>::jit_oam_host_plugin_ready() const
{
    if (oam_host_plugin_v2_ && static_cast<bool>(*oam_host_plugin_v2_)) return true;
    return oam_host_plugin_ && static_cast<bool>(*oam_host_plugin_);
}

template <typename Precision>
std::string NequIPOAMKokkos<Precision>::jit_oam_host_plugin_path() const
{
    if (oam_host_plugin_v2_ && *oam_host_plugin_v2_) return oam_host_plugin_v2_->path();
    return oam_host_plugin_ && *oam_host_plugin_ ? oam_host_plugin_->path() : std::string();
}

template <typename Precision>
std::string NequIPOAMKokkos<Precision>::jit_oam_host_plugin_artifact_id() const
{
    if (oam_host_plugin_v2_ && *oam_host_plugin_v2_)
        return std::string(oam_host_plugin_v2_->descriptor().artifact_id);
    return oam_host_plugin_ && *oam_host_plugin_
        ? std::string(oam_host_plugin_->descriptor().artifact_id) : std::string();
}

template <typename Precision>
bool NequIPOAMKokkos<Precision>::has_portable_edge_implementation() const
{
    if (layer_states_.empty()) return false;
    return std::all_of(layer_states_.begin(), layer_states_.end(),
        [] (const LayerState& state) { return state.edge_tensor_product != nullptr; });
}

template <typename Precision>
bool NequIPOAMKokkos<Precision>::generated_edges_serve() const
{
    // The generated tier is the artifact the direct-mode dispatcher actually
    // runs.  A loaded CUDA plugin serves every edge region; on the host the
    // receiver-forward/source-reverse ABI v7 artifact serves a direct request and the
    // edge-owned (ABI v1) artifact serves only the explicit debug_v1 mode.
    // Reference mode always uses the portable implementation. Reporting mere
    // artifact readiness here
    // would claim a generated tier for an artifact the dispatcher never runs.
    if (streamed_edges_mode_ == "reference") return false;
    if (cuda_r1_plugin_ && static_cast<bool>(*cuda_r1_plugin_)) return true;
    if (!execution_space_is_host()) return false;
    if (streamed_edges_mode_ == "direct")
        return oam_host_plugin_v2_ && static_cast<bool>(*oam_host_plugin_v2_);
    return oam_host_plugin_ && static_cast<bool>(*oam_host_plugin_);
}

template <typename Precision>
std::string NequIPOAMKokkos<Precision>::execution_implementation_label() const
{
    // This name is the implementation the dispatch runs, so it follows the
    // same authority as the dispatch: a loaded generated CUDA plugin serves
    // every edge region of the device execution whether or not a forward or
    // reverse has already completed.  The execution state that used to leak
    // into this name stays reported by cuda_r1_ready/launch-count/reason.
    if (streamed_edges_mode_ == "reference")
        return "portable_oam_kokkos_generic";
    if (cuda_r1_plugin_ && *cuda_r1_plugin_)
        return "generated-r1";
    if (receiver_owned_execution())
        return "receiver_owned_full_message_oam_host_v7";
    if (streamed_edges_mode_ == "direct")
        return "unavailable_oam_direct";
    return "generated_oam_host_v1";
}

template <typename Precision>
void NequIPOAMKokkos<Precision>::refresh_execution_state_diagnostic()
{
    const bool requested_execution_ready = streamed_edges_mode_ != "direct"
        || generated_edges_serve();
    last_execution_diagnostic_ =
        "NequIP OAM Kokkos execution: backend=" + execution_backend()
        + ", implementation=" + execution_implementation_label()
        + ", state=" + (requested_execution_ready ? "ready" : "unavailable")
        + ", artifact_id=" + execution_artifact_id()
        + ", radial_sets=" + std::to_string(radial_sets_.size())
        + cuda_r1_state_text();
}

template <typename Precision>
std::string NequIPOAMKokkos<Precision>::execution_artifact_id() const
{
    if (cuda_r1_plugin_ && *cuda_r1_plugin_) return cuda_r1_artifact_id_;
    if (execution_space_is_host() && generated_edges_serve())
        return jit_oam_host_plugin_artifact_id();
    return {};
}

template <typename Precision>
std::string NequIPOAMKokkos<Precision>::execution_loaded_artifact_identity() const
{
    return execution_artifact_id();
}

template <typename Precision>
std::string NequIPOAMKokkos<Precision>::execution_forward_owner_schedule() const
{
    if (cuda_r1_plugin_ && *cuda_r1_plugin_)
        return cuda_r1_plugin_->descriptor().full_forward_schedule;
    if (oam_host_plugin_v2_ && *oam_host_plugin_v2_)
        return oam_host_plugin_v2_->descriptor().forward_owner_schedule;
    return {};
}

template <typename Precision>
std::string NequIPOAMKokkos<Precision>::execution_reverse_owner_schedule() const
{
    if (cuda_r1_plugin_ && *cuda_r1_plugin_)
        return cuda_r1_plugin_->descriptor().full_reverse_schedule;
    if (oam_host_plugin_v2_ && *oam_host_plugin_v2_)
        return oam_host_plugin_v2_->descriptor().full_reverse_owner_schedule;
    return {};
}

template <typename Precision>
std::string NequIPOAMKokkos<Precision>::execution_tiling_policy() const
{
    if (cuda_r1_plugin_ && *cuda_r1_plugin_)
        return cuda_r1_plugin_->descriptor().full_tiling_policy;
    if (oam_host_plugin_v2_ && *oam_host_plugin_v2_)
        return oam_host_plugin_v2_->descriptor().tiling_policy;
    return {};
}

template <typename Precision>
std::string NequIPOAMKokkos<Precision>::execution_vector_policy() const
{
    if (cuda_r1_plugin_ && *cuda_r1_plugin_)
        return cuda_r1_plugin_->descriptor().full_vector_policy;
    if (oam_host_plugin_v2_ && *oam_host_plugin_v2_)
        return oam_host_plugin_v2_->descriptor().vector_policy;
    return {};
}

template <typename Precision>
std::string NequIPOAMKokkos<Precision>::execution_shared_memory_policy() const
{
    if (cuda_r1_plugin_ && *cuda_r1_plugin_)
        return cuda_r1_plugin_->descriptor().full_shared_memory_policy;
    if (oam_host_plugin_v2_ && *oam_host_plugin_v2_)
        return oam_host_plugin_v2_->descriptor().shared_memory_policy;
    return {};
}

template <typename Precision>
std::string NequIPOAMKokkos<Precision>::execution_register_budget_policy() const
{
    if (cuda_r1_plugin_ && *cuda_r1_plugin_)
        return cuda_r1_plugin_->descriptor().full_register_budget_policy;
    if (oam_host_plugin_v2_ && *oam_host_plugin_v2_)
        return oam_host_plugin_v2_->descriptor().register_budget_policy;
    return {};
}

template <typename Precision>
std::string NequIPOAMKokkos<Precision>::execution_direct_geometry_policy() const
{
    if (cuda_r1_plugin_ && *cuda_r1_plugin_)
        return cuda_r1_plugin_->descriptor().direct_geometry_policy;
    if (oam_host_plugin_v2_ && *oam_host_plugin_v2_)
        return oam_host_plugin_v2_->descriptor().direct_geometry_policy;
    return {};
}

template <typename Precision>
int NequIPOAMKokkos<Precision>::execution_channel_tile() const
{
    if (cuda_r1_plugin_ && *cuda_r1_plugin_)
        return 0;
    if (oam_host_plugin_v2_ && *oam_host_plugin_v2_)
        return oam_host_plugin_v2_->descriptor().layer_count == 0 ? 0
            : oam_host_plugin_v2_->descriptor().layers[0].channel_tile;
    return 0;
}

template <typename Precision>
int NequIPOAMKokkos<Precision>::execution_scalar_width() const
{
    if (cuda_r1_plugin_ && *cuda_r1_plugin_)
        return static_cast<int>(cuda_r1_plugin_->descriptor().scalar_size);
    if (oam_host_plugin_v2_ && *oam_host_plugin_v2_)
        return static_cast<int>(oam_host_plugin_v2_->descriptor().scalar_width);
    return 0;
}

#define SYMMETRIX_OAM_RESOURCE_METHOD(name, cuda_field, host_field) \
template <typename Precision> std::string NequIPOAMKokkos<Precision>::name() const { \
    if (cuda_r1_plugin_ && *cuda_r1_plugin_) return cuda_r1_plugin_->descriptor().cuda_field; \
    if (oam_host_plugin_v2_ && *oam_host_plugin_v2_) return oam_host_plugin_v2_->descriptor().host_field; \
    return {}; }
SYMMETRIX_OAM_RESOURCE_METHOD(execution_required_isa, target_policy, required_isa)
SYMMETRIX_OAM_RESOURCE_METHOD(execution_schedule_qualification, schedule_qualification, schedule_qualification)
#undef SYMMETRIX_OAM_RESOURCE_METHOD

template <typename Precision>
std::string NequIPOAMKokkos<Precision>::execution_reduction_policy() const
{
    if (cuda_r1_plugin_ && *cuda_r1_plugin_)
        return {};
    if (oam_host_plugin_v2_ && *oam_host_plugin_v2_)
        return oam_host_plugin_v2_->descriptor().full_reverse_reduction_policy;
    return {};
}

template <typename Precision>
int NequIPOAMKokkos<Precision>::execution_target_compute_capability() const
{ return cuda_r1_plugin_ && *cuda_r1_plugin_ ? cuda_r1_plugin_->descriptor().target_compute_capability : 0; }
template <typename Precision>
std::string NequIPOAMKokkos<Precision>::execution_precision_policy() const
{ return cuda_r1_plugin_ && *cuda_r1_plugin_ ? cuda_r1_plugin_->descriptor().precision_policy : std::string{}; }

template <typename Precision>
std::size_t NequIPOAMKokkos<Precision>::conditioned_self_bytes() const
{
    std::size_t bytes = 0;
    for (const auto& state : layer_states_)
        if (state.conditioned_self.eligible)
            bytes += state.conditioned_self.coefficients.size() * sizeof(Precision);
    return bytes;
}

template <typename Precision>
std::size_t NequIPOAMKokkos<Precision>::conditioned_self_layer_count() const
{
    std::size_t count = 0;
    for (const auto& state : layer_states_)
        count += state.conditioned_self.eligible ? 1u : 0u;
    return count;
}

template <typename Precision>
bool NequIPOAMKokkos<Precision>::receiver_owned_execution() const
{
    if (streamed_edges_mode_ != "direct") return false;
    if (execution_space_is_cuda())
        return cuda_r1_plugin_ && static_cast<bool>(*cuda_r1_plugin_);
    if (!execution_space_is_host()) return false;
    return oam_host_plugin_v2_ && static_cast<bool>(*oam_host_plugin_v2_);
}

template <typename Precision>
void NequIPOAMKokkos<Precision>::set_streamed_edges(std::string mode)
{
    if (mode == "auto" || mode == "generic")
        throw std::invalid_argument(
            "NequIP OAM Kokkos supports only direct, reference, or debug_v1 "
            "streamed_edges modes.");
    if (mode == "materialized") mode = "debug_v1";
    if (mode != "direct" && mode != "reference" && mode != "debug_v1")
        throw std::invalid_argument("NequIP OAM Kokkos streamed_edges mode is invalid.");
    if (mode == "direct" && !execution_space_is_host()
            && !execution_space_is_cuda())
        throw std::invalid_argument(
            "NequIP OAM Kokkos direct execution is unsupported on target '"
            + execution_target() + "'; use a host or CUDA backend.");
    if (mode == "direct" && execution_space_is_cuda()
        && !cuda_r1_plugin_path_.empty()
        && !(cuda_r1_plugin_ && static_cast<bool>(*cuda_r1_plugin_)))
        throw std::invalid_argument(
            "NequIP OAM Kokkos rejected the requested CUDA artifact: "
            + cuda_r1_reason_);
    if (mode == "direct" && !(oam_host_plugin_v2_ && *oam_host_plugin_v2_)
        && !(cuda_r1_plugin_ && *cuda_r1_plugin_))
        throw std::invalid_argument(
            "NequIP OAM Kokkos direct execution requires an admitted generated "
            "receiver/source-owned ABI v7 host plugin or CUDA plugin; use reference "
            "mode for the portable implementation. " + cuda_r1_reason_);
    if (mode == "debug_v1" && !execution_space_is_host())
        throw std::invalid_argument(
            "NequIP OAM Kokkos debug_v1 execution requires a host execution "
            "space.");
    if (mode == "debug_v1" && !(oam_host_plugin_ && *oam_host_plugin_))
        throw std::invalid_argument(
            "NequIP OAM Kokkos debug_v1 execution requires an edge-owned ABI v1 "
            "host plugin.");
    streamed_edges_mode_ = std::move(mode);
    // The mode decides which artifact the dispatcher runs, so the tier report
    // is recomputed with it.
    refresh_execution_state_diagnostic();
    refresh_execution_plan_report();
}

template <typename Precision>
void NequIPOAMKokkos<Precision>::set_execution_plan_request(
    std::string algorithm, std::string profile)
{
    const auto requested_profile =
        symmetrix::execution::parse_execution_profile(profile);
    set_streamed_edges(std::move(algorithm));
    execution_plan_.requested_algorithm = streamed_edges_mode_ == "direct"
        ? ExecutionMode::direct
        : ExecutionMode::generic;
    execution_plan_.requested_profile = requested_profile;
    execution_plan_.selection_source = "native_request";
    refresh_execution_plan_report();
}

template <typename Precision>
void NequIPOAMKokkos<Precision>::set_execution_edge_variant(std::string variant)
{
    if (variant != "auto" && variant != "full_message")
        throw std::invalid_argument(
            "OAM execution edge variant must be auto or full_message");
    execution_edge_variant_request_ = std::move(variant);
    refresh_execution_plan_report();
}

template <typename Precision>
std::string NequIPOAMKokkos<Precision>::execution_edge_variant() const
{
    return "full_message";
}

template <typename Precision>
std::string NequIPOAMKokkos<Precision>::execution_edge_variant_reason() const
{
    if (execution_edge_variant_request_ == "full_message")
        return "explicit full-message control request";
    return "full-message is the only supported OAM direct variant";
}

template <typename Precision>
void NequIPOAMKokkos<Precision>::set_execution_plan_qualification(
    const std::string& plan_id, const std::string& artifact_identity,
    const std::string& backend, const std::string& precision,
    const std::string& schedule, const std::string& capability,
    const std::string& empirical_status,
    std::optional<std::size_t> estimated_bytes,
    const std::size_t available_bytes, const std::size_t reserve_bytes)
{
    PlanQualification* qualification = nullptr;
    std::string expected_schedule;
    std::string expected_capability;
    if (plan_id == "oam-direct-speed") {
        qualification = &execution_plan_speed_qualification_;
        expected_schedule = "throughput";
        expected_capability = "generated_edge_tensor_product";
    } else if (plan_id == "oam-direct-capacity") {
        qualification = &execution_plan_capacity_qualification_;
        expected_schedule = "capacity";
        expected_capability = "exact_capacity_layout";
    } else {
        throw std::invalid_argument("unknown OAM execution-plan qualification id");
    }
    const std::string expected_precision =
        std::is_same_v<Precision, float> ? "float32" : "float64";
    const auto policy = plan_id == "oam-direct-speed"
        ? symmetrix::oam::OAMRetentionPolicy::speed
        : symmetrix::oam::OAMRetentionPolicy::capacity;
    const std::size_t planner_estimate =
        execution_memory_layout(policy).active_bytes;
    const std::string layout_identity =
        execution_memory_layout_identity(policy);
    qualification->admitted = false;
    qualification->empirical_status = "rejected";
    qualification->artifact_identity = artifact_identity;
    qualification->estimated_bytes.reset();
    qualification->layout_identity.clear();
    const std::string current_artifact_identity =
        family_artifact_readiness().artifact_id;
    if (artifact_identity.empty()
        || artifact_identity != current_artifact_identity)
        qualification->reason = "artifact identity mismatch in qualification evidence";
    else if (backend != execution_target())
        qualification->reason = "backend mismatch in qualification evidence";
    else if (precision != expected_precision)
        qualification->reason = "precision mismatch in qualification evidence";
    else if (schedule != expected_schedule)
        qualification->reason = "schedule mismatch in qualification evidence";
    else if (capability != expected_capability)
        qualification->reason = "capability mismatch in qualification evidence";
    else if (!estimated_bytes)
        qualification->reason = "an exact estimate is required for plan admission";
    else if (*estimated_bytes != planner_estimate)
        qualification->reason = "qualification planner estimate mismatch";
    else if (empirical_status != "qualified")
        qualification->reason = "empirical qualification is required";
    else {
        qualification->admitted = true;
        qualification->empirical_status = "qualified";
        qualification->estimated_bytes = planner_estimate;
        qualification->layout_identity = layout_identity;
        qualification->reason =
            "backend, precision, schedule, capability, and empirical evidence match";
    }
    execution_plan_memory_available_ = true;
    execution_plan_.available_bytes = available_bytes;
    execution_plan_.reserve_bytes = reserve_bytes;
    refresh_execution_plan_report();
}

template <typename Precision>
std::string NequIPOAMKokkos<Precision>::execution_plan_empirical_qualification(
    const std::string& plan_id) const
{
    const PlanQualification* qualification = nullptr;
    if (plan_id == "oam-direct-speed")
        qualification = &execution_plan_speed_qualification_;
    else if (plan_id == "oam-direct-capacity")
        qualification = &execution_plan_capacity_qualification_;
    if (qualification != nullptr) {
        const auto current_identity = family_artifact_readiness().artifact_id;
        if (qualification->admitted
            && qualification->artifact_identity != current_identity)
            return "rejected";
        const auto policy = plan_id == "oam-direct-speed"
            ? symmetrix::oam::OAMRetentionPolicy::speed
            : symmetrix::oam::OAMRetentionPolicy::capacity;
        if (qualification->admitted
            && qualification->layout_identity
                != execution_memory_layout_identity(policy))
            return "rejected";
        return qualification->empirical_status;
    }
    return "required";
}

template <typename Precision>
std::string
NequIPOAMKokkos<Precision>::execution_plan_qualification_artifact_identity(
    const std::string& plan_id) const
{
    if (plan_id == "oam-direct-speed")
        return execution_plan_speed_qualification_.artifact_identity;
    if (plan_id == "oam-direct-capacity")
        return execution_plan_capacity_qualification_.artifact_identity;
    return {};
}

template <typename Precision>
std::string NequIPOAMKokkos<Precision>::execution_backend() const
{
    const std::string space = _kokkos_default_execution_space();
    if (space.find("OpenMP") != std::string::npos) return "openmp";
    if (space.find("Serial") != std::string::npos) return "serial";
    if (space.find("Cuda") != std::string::npos) return "cuda";
    if (space.find("HIP") != std::string::npos
            || space.find("Hip") != std::string::npos) return "hip";
    return "kokkos";
}

template <typename Precision>
std::size_t NequIPOAMKokkos<Precision>::edge_count() const
{
    return checked_edge_count(std::span<const int>(num_neigh_host_.data(), num_neigh_host_.size()));
}

template <typename Precision>
void NequIPOAMKokkos<Precision>::ensure_chunk_workspace(
    RuntimeLayerState& runtime, const std::size_t edge_capacity,
    const int prelinear_width, const int edge_width, const int weight_count,
    const int output_width)
{
    if (runtime.chunk_edge_capacity >= edge_capacity
        && runtime.chunk_prelinear_width == prelinear_width
        && runtime.chunk_edge_width == edge_width
        && runtime.chunk_weight_count == weight_count
        && runtime.chunk_output_width == output_width)
        return;
    const std::size_t capacity = std::max(
        edge_capacity, std::max<std::size_t>(1, runtime.chunk_edge_capacity * 2));
    Kokkos::realloc(runtime.chunk_source_features, capacity, prelinear_width);
    Kokkos::realloc(runtime.chunk_edge_features, capacity, edge_width);
    Kokkos::realloc(runtime.chunk_edge_weights, capacity, weight_count);
    Kokkos::realloc(runtime.chunk_edge_messages, capacity, output_width);
    Kokkos::realloc(runtime.chunk_source_adjoint, capacity, prelinear_width);
    Kokkos::realloc(runtime.chunk_edge_adjoint, capacity, edge_width);
    Kokkos::realloc(runtime.chunk_output_adjoint, capacity, output_width);
    Kokkos::realloc(runtime.chunk_weights_adjoint, capacity, weight_count);
    Kokkos::realloc(runtime.chunk_source_node, capacity, prelinear_width);
    Kokkos::realloc(
        runtime.chunk_source_adjoint_node, capacity, prelinear_width);
    Kokkos::realloc(runtime.chunk_messages_node, capacity, output_width);
    runtime.chunk_edge_capacity = capacity;
    runtime.chunk_prelinear_width = prelinear_width;
    runtime.chunk_edge_width = edge_width;
    runtime.chunk_weight_count = weight_count;
    runtime.chunk_output_width = output_width;
}

template <typename Precision>
typename E3LinearKokkosT<Precision>::WorkspaceShape
NequIPOAMKokkos<Precision>::execution_linear_workspace_shape() const
{
    using Shape = typename E3LinearKokkosT<Precision>::WorkspaceShape;
    Shape result;
    const auto include = [&result](const Shape& shape) {
        result.input_rows = std::max(result.input_rows, shape.input_rows);
        result.input_columns = std::max(
            result.input_columns, shape.input_columns);
        result.output_rows = std::max(result.output_rows, shape.output_rows);
        result.output_columns = std::max(
            result.output_columns, shape.output_columns);
    };
    const std::size_t nodes = static_cast<std::size_t>(num_nodes_);
    for (const auto& layer : layer_states_) {
        include(layer.pre_linear.lane_workspace_shape(nodes));
        include(layer.post_linear.lane_workspace_shape(nodes));
        if (layer.self_linear)
            include(layer.self_linear->lane_workspace_shape(nodes));
    }
    include(readout_.lane_workspace_shape(nodes));
    return result;
}

template <typename Precision>
typename E3TensorProductKokkosT<Precision>::BilinearWorkspaceShape
NequIPOAMKokkos<Precision>::execution_bilinear_workspace_shape() const
{
    using Shape =
        typename E3TensorProductKokkosT<Precision>::BilinearWorkspaceShape;
    Shape result;
    const auto include = [&result](const Shape& shape) {
        result.matrix_values = std::max(
            result.matrix_values, shape.matrix_values);
        result.result_values = std::max(
            result.result_values, shape.result_values);
        result.adjoint_values = std::max(
            result.adjoint_values, shape.adjoint_values);
        result.k_adjoint_values = std::max(
            result.k_adjoint_values, shape.k_adjoint_values);
    };
    const std::size_t nodes = static_cast<std::size_t>(num_nodes_);
    for (const auto& layer : layer_states_)
        if (layer.self_tensor_product && layer.self_lane_kernels)
            include(layer.self_tensor_product->bilinear_workspace_shape(nodes));
    return result;
}

template <typename Precision>
symmetrix::oam::OAMMemoryLayout
NequIPOAMKokkos<Precision>::execution_memory_layout(
    const symmetrix::oam::OAMRetentionPolicy policy) const
{
    symmetrix::oam::OAMMemoryDimensions dimensions;
    dimensions.num_nodes = static_cast<std::size_t>(num_nodes_);
    dimensions.num_edges = edge_count();
    dimensions.scalar_bytes = sizeof(Precision);
    dimensions.harmonic_count = static_cast<std::size_t>(edge_harmonic_count_);
    dimensions.readout_width = static_cast<std::size_t>(readout_.input_dimension());
    dimensions.readout_output_width =
        static_cast<std::size_t>(readout_.output_dimension());
    const auto workspace = execution_linear_workspace_shape();
    dimensions.packed_input_values = symmetrix::oam::checked_oam_multiply(
        workspace.input_rows, workspace.input_columns);
    dimensions.packed_output_values = symmetrix::oam::checked_oam_multiply(
        workspace.output_rows, workspace.output_columns);
    const auto bilinear = execution_bilinear_workspace_shape();
    dimensions.bilinear_forward_values = symmetrix::oam::checked_oam_add(
        bilinear.matrix_values, bilinear.result_values);
    dimensions.bilinear_reverse_values = symmetrix::oam::checked_oam_add(
        bilinear.adjoint_values, bilinear.k_adjoint_values);
    if (cuda_r1_plugin_ && static_cast<bool>(*cuda_r1_plugin_)) {
        const auto& descriptor = cuda_r1_plugin_->descriptor();
        for (std::size_t layer = 0; layer < descriptor.layer_count; ++layer) {
            const auto groups = std::max<std::int32_t>(
                descriptor.layers[layer].reverse_group_count, 0);
            dimensions.reverse_partial_values = std::max(
                dimensions.reverse_partial_values,
                symmetrix::oam::checked_oam_multiply(
                    static_cast<std::size_t>(groups),
                    symmetrix::oam::checked_oam_multiply(
                        std::size_t(3), dimensions.num_edges)));
        }
    }
    std::size_t maximum_chunk_edges = 0;
    for (int receiver_begin = 0; receiver_begin < num_nodes_;) {
        const int receiver_end = std::min(
            num_nodes_, receiver_begin + static_cast<int>(receiver_chunk_limit_));
        maximum_chunk_edges = std::max(maximum_chunk_edges,
            static_cast<std::size_t>(first_neigh_host_[receiver_end]
                - first_neigh_host_[receiver_begin]));
        receiver_begin = receiver_end;
    }
    dimensions.layers.reserve(layer_states_.size());
    for (const auto& layer : layer_states_) {
        std::size_t self_forward = 0;
        std::size_t self_reverse = 0;
        const std::size_t attributes =
            static_cast<std::size_t>(node_attribute_width_);
        if (layer.self_linear) {
            self_forward = static_cast<std::size_t>(layer.pre_gate_width);
            self_reverse = static_cast<std::size_t>(layer.input_width);
        } else if (layer.self_tensor_product) {
            if (layer.self_lane_kernels) {
                self_forward = symmetrix::oam::checked_oam_add(
                    static_cast<std::size_t>(layer.pre_gate_width), attributes);
                self_reverse = symmetrix::oam::checked_oam_multiply(
                    std::size_t(2), attributes);
            } else {
                self_forward = symmetrix::oam::checked_oam_add(
                    symmetrix::oam::checked_oam_add(
                        symmetrix::oam::checked_oam_multiply(std::size_t(2),
                            static_cast<std::size_t>(layer.pre_gate_width)),
                        static_cast<std::size_t>(layer.input_width)),
                    symmetrix::oam::checked_oam_multiply(
                        std::size_t(2), attributes));
                self_reverse = symmetrix::oam::checked_oam_add(
                    symmetrix::oam::checked_oam_add(
                        symmetrix::oam::checked_oam_multiply(std::size_t(2),
                            static_cast<std::size_t>(layer.input_width)),
                        static_cast<std::size_t>(layer.pre_gate_width)),
                    symmetrix::oam::checked_oam_multiply(
                        std::size_t(4), attributes));
            }
        }
        dimensions.layers.push_back({
            static_cast<std::size_t>(layer.input_width),
            static_cast<std::size_t>(layer.prelinear_width),
            static_cast<std::size_t>(layer.edge_width),
            static_cast<std::size_t>(layer.tp_output_width),
            static_cast<std::size_t>(layer.pre_gate_width),
            static_cast<std::size_t>(layer.output_width), attributes,
            self_forward, self_reverse});
        std::size_t chunk_width = symmetrix::oam::checked_oam_multiply(
            std::size_t(4), static_cast<std::size_t>(layer.prelinear_width));
        chunk_width = symmetrix::oam::checked_oam_add(chunk_width,
            symmetrix::oam::checked_oam_multiply(
                std::size_t(2), static_cast<std::size_t>(layer.edge_width)));
        chunk_width = symmetrix::oam::checked_oam_add(chunk_width,
            symmetrix::oam::checked_oam_multiply(
                std::size_t(2), static_cast<std::size_t>(layer.weight_count)));
        chunk_width = symmetrix::oam::checked_oam_add(chunk_width,
            symmetrix::oam::checked_oam_multiply(
                std::size_t(3), static_cast<std::size_t>(layer.tp_output_width)));
        // Generated direct owners never enter the portable edge-chunk path.
        // Reserving this buffer for the full-message direct variant inflated
        // its reported memory without backing any executed work.
        if (!receiver_owned_execution())
            dimensions.edge_chunk_values = std::max(dimensions.edge_chunk_values,
                symmetrix::oam::checked_oam_multiply(
                    maximum_chunk_edges, chunk_width));
        dimensions.source_harmonic_partial_values = std::max(
            dimensions.source_harmonic_partial_values,
            symmetrix::oam::checked_oam_multiply(dimensions.num_edges,
                static_cast<std::size_t>(layer.edge_width)));
    }
    dimensions.source_distance_partial_values = dimensions.num_edges;
    return symmetrix::oam::plan_oam_memory(dimensions, policy);
}

template <typename Precision>
std::string NequIPOAMKokkos<Precision>::execution_memory_layout_identity(
    const symmetrix::oam::OAMRetentionPolicy policy) const
{
    const auto layout = execution_memory_layout(policy);
    std::ostringstream identity;
    identity << static_cast<int>(policy) << ':' << num_nodes_ << ':'
        << edge_count() << ':' << layout.active_bytes;
    for (const auto& slot : layout.slots)
        identity << '|' << slot.name << ':' << slot.offset << ':' << slot.bytes
            << ':' << slot.live_begin << ':' << slot.live_end;
    return identity.str();
}

template <typename Precision>
void NequIPOAMKokkos<Precision>::ensure_execution_arena()
{
    const bool capacity =
        execution_plan_.selected_id == "oam-direct-capacity";
    const auto policy = capacity ? symmetrix::oam::OAMRetentionPolicy::capacity
                                 : symmetrix::oam::OAMRetentionPolicy::speed;
    const auto layout_identity = execution_memory_layout_identity(policy);
    active_memory_layout_ = execution_memory_layout(policy);
    const bool initial = transient_arena_state_.retained_capacity_bytes == 0;
    const bool replace = transient_arena_state_.admit(
        active_memory_layout_, initial ? "initial graph" : "active graph growth");
    if (replace) {
        Kokkos::fence("replace OAM transient arena");
        Kokkos::realloc(Kokkos::WithoutInitializing, transient_arena_,
            transient_arena_state_.retained_capacity_bytes);
    }
    transient_arena_runtime_subview_count_ = 0;
    if (replace || layout_identity != transient_arena_canary_layout_identity_) {
        initialize_execution_arena_canaries();
        transient_arena_canary_layout_identity_ = layout_identity;
    }
    bind_execution_arena_retained_views();
    // Candidate estimates depend on the active graph rather than the retained
    // high-water capacity, so refresh after every admission, including shrink.
    refresh_execution_plan_report();
}

template <typename Precision>
Kokkos::View<Precision**, Kokkos::LayoutRight>
NequIPOAMKokkos<Precision>::arena_matrix(const std::string& slot_name,
    const std::size_t relative_values, const std::size_t rows,
    const std::size_t columns)
{
    const std::size_t values = symmetrix::oam::checked_oam_multiply(rows, columns);
    auto flat = arena_vector<Precision>(slot_name, relative_values, values);
    return Kokkos::View<Precision**, Kokkos::LayoutRight>(
        Kokkos::view_wrap(flat.data()), rows, columns);
}

template <typename Precision>
void NequIPOAMKokkos<Precision>::initialize_execution_arena_canaries()
{
    ++transient_arena_canary_initializations_;
    std::vector<std::size_t> offsets;
    for (const auto& slot : active_memory_layout_.slots) {
        if (std::find(offsets.begin(), offsets.end(), slot.offset) != offsets.end())
            continue;
        offsets.push_back(slot.offset);
        Kokkos::deep_copy(Kokkos::subview(transient_arena_,
            std::make_pair(slot.offset - symmetrix::oam::oam_arena_canary_bytes,
                slot.offset)), static_cast<unsigned char>(0xa5));
        Kokkos::deep_copy(Kokkos::subview(transient_arena_,
            std::make_pair(slot.offset + slot.bytes,
                slot.offset + slot.bytes
                    + symmetrix::oam::oam_arena_canary_bytes)),
            static_cast<unsigned char>(0x5a));
    }
}

template <typename Precision>
bool NequIPOAMKokkos<Precision>::execution_arena_canaries_intact() const
{
    std::vector<std::size_t> offsets;
    for (const auto& slot : active_memory_layout_.slots) {
        if (std::find(offsets.begin(), offsets.end(), slot.offset) != offsets.end())
            continue;
        offsets.push_back(slot.offset);
        const auto before = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
            Kokkos::subview(transient_arena_,
                std::make_pair(slot.offset - symmetrix::oam::oam_arena_canary_bytes,
                    slot.offset)));
        const auto after = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
            Kokkos::subview(transient_arena_,
                std::make_pair(slot.offset + slot.bytes,
                    slot.offset + slot.bytes
                        + symmetrix::oam::oam_arena_canary_bytes)));
        for (std::size_t index = 0; index < before.extent(0); ++index)
            if (before(index) != static_cast<unsigned char>(0xa5)) return false;
        for (std::size_t index = 0; index < after.extent(0); ++index)
            if (after(index) != static_cast<unsigned char>(0x5a)) return false;
    }
    return true;
}

template <typename Precision>
void NequIPOAMKokkos<Precision>::bind_execution_arena_retained_views()
{
    runtime_layers_.resize(layer_states_.size());
    Kokkos::View<Precision**, Kokkos::LayoutRight> previous;
    if (!layer_states_.empty())
        previous = arena_matrix("initial_features", 0,
            static_cast<std::size_t>(num_nodes_),
            static_cast<std::size_t>(layer_states_.front().input_width));
    for (std::size_t index = 0; index < layer_states_.size(); ++index) {
        auto& runtime = runtime_layers_[index];
        const auto& layer = layer_states_[index];
        const std::string prefix = "layer_" + std::to_string(index);
        runtime.input = previous;
        runtime.pre_gate = arena_matrix(prefix + "_pre_gate", 0,
            static_cast<std::size_t>(num_nodes_),
            static_cast<std::size_t>(layer.pre_gate_width));
        runtime.output = arena_matrix(prefix + "_output", 0,
            static_cast<std::size_t>(num_nodes_),
            static_cast<std::size_t>(layer.output_width));
        if (active_memory_layout_.policy == symmetrix::oam::OAMRetentionPolicy::speed)
            runtime.prelinear = arena_matrix(prefix + "_prelinear", 0,
                static_cast<std::size_t>(num_nodes_),
                static_cast<std::size_t>(layer.prelinear_width));
        else
            runtime.prelinear = {};
        runtime.reverse_geometry_partials = arena_vector<Precision>(
            "reverse_partials", 0,
            active_memory_layout_.slot("reverse_partials").bytes
                / sizeof(Precision));
        std::size_t maximum_chunk_edges = 0;
        for (int receiver_begin = 0; receiver_begin < num_nodes_;) {
            const int receiver_end = std::min(num_nodes_, receiver_begin
                    + static_cast<int>(receiver_chunk_limit_));
            maximum_chunk_edges = std::max(maximum_chunk_edges,
                static_cast<std::size_t>(first_neigh_host_[receiver_end]
                    - first_neigh_host_[receiver_begin]));
            receiver_begin = receiver_end;
        }
        std::size_t chunk_cursor = 0;
        if (active_memory_layout_.slot("edge_chunk_scratch").bytes == 0) {
            runtime.chunk_source_features = {};
            runtime.chunk_edge_features = {};
            runtime.chunk_edge_weights = {};
            runtime.chunk_edge_messages = {};
            runtime.chunk_source_adjoint = {};
            runtime.chunk_edge_adjoint = {};
            runtime.chunk_output_adjoint = {};
            runtime.chunk_weights_adjoint = {};
            runtime.chunk_source_node = {};
            runtime.chunk_source_adjoint_node = {};
            runtime.chunk_messages_node = {};
            runtime.chunk_edge_capacity = 0;
            previous = runtime.output;
            continue;
        }
        const auto chunk_matrix = [this, &chunk_cursor, maximum_chunk_edges](
                                      const std::size_t width) {
            auto result = arena_matrix("edge_chunk_scratch", chunk_cursor,
                maximum_chunk_edges, width);
            chunk_cursor = symmetrix::oam::checked_oam_add(
                chunk_cursor, result.size());
            return result;
        };
        runtime.chunk_source_features = chunk_matrix(layer.prelinear_width);
        runtime.chunk_edge_features = chunk_matrix(layer.edge_width);
        runtime.chunk_edge_weights = chunk_matrix(layer.weight_count);
        runtime.chunk_edge_messages = chunk_matrix(layer.tp_output_width);
        runtime.chunk_source_adjoint = chunk_matrix(layer.prelinear_width);
        runtime.chunk_edge_adjoint = chunk_matrix(layer.edge_width);
        runtime.chunk_output_adjoint = chunk_matrix(layer.tp_output_width);
        runtime.chunk_weights_adjoint = chunk_matrix(layer.weight_count);
        runtime.chunk_source_node = chunk_matrix(layer.prelinear_width);
        runtime.chunk_source_adjoint_node = chunk_matrix(layer.prelinear_width);
        runtime.chunk_messages_node = chunk_matrix(layer.tp_output_width);
        runtime.chunk_edge_capacity = maximum_chunk_edges;
        runtime.chunk_prelinear_width = layer.prelinear_width;
        runtime.chunk_edge_width = layer.edge_width;
        runtime.chunk_weight_count = layer.weight_count;
        runtime.chunk_output_width = layer.tp_output_width;
        previous = runtime.output;
    }
    linear_workspace_shape_ = execution_linear_workspace_shape();
    std::size_t packed_cursor = 0;
    linear_workspace_->packed_input = arena_matrix("packed_linear_scratch",
        packed_cursor, linear_workspace_shape_.input_rows,
        linear_workspace_shape_.input_columns);
    packed_cursor = symmetrix::oam::checked_oam_add(
        packed_cursor, linear_workspace_->packed_input.size());
    linear_workspace_->packed_output = arena_matrix("packed_linear_scratch",
        packed_cursor, linear_workspace_shape_.output_rows,
        linear_workspace_shape_.output_columns);
    linear_workspace_->fixed_capacity = true;

    const auto bilinear_shape = execution_bilinear_workspace_shape();
    std::size_t bilinear_cursor = 0;
    bilinear_workspace_->matrix = arena_matrix("bilinear_forward_scratch",
        bilinear_cursor, 1, bilinear_shape.matrix_values);
    bilinear_cursor = symmetrix::oam::checked_oam_add(
        bilinear_cursor, bilinear_shape.matrix_values);
    bilinear_workspace_->result = arena_matrix("bilinear_forward_scratch",
        bilinear_cursor, 1, bilinear_shape.result_values);
    bilinear_cursor = 0;
    bilinear_workspace_->adjoint = arena_matrix("bilinear_reverse_scratch",
        bilinear_cursor, 1, bilinear_shape.adjoint_values);
    bilinear_cursor = symmetrix::oam::checked_oam_add(
        bilinear_cursor, bilinear_shape.adjoint_values);
    bilinear_workspace_->k_adjoint = arena_matrix("bilinear_reverse_scratch",
        bilinear_cursor, 1, bilinear_shape.k_adjoint_values);
    bilinear_workspace_->fixed_capacity = true;

    std::size_t reduction_cursor = 0;
    const std::size_t edges = edge_count();
    const std::size_t nodes = static_cast<std::size_t>(num_nodes_);
    active_zbl_energies_ = arena_vector<double>(
        "final_reductions", reduction_cursor, edges);
    reduction_cursor = symmetrix::oam::checked_oam_add(
        reduction_cursor, edges);
    active_node_energies_ = arena_vector<double>(
        "final_reductions", reduction_cursor, nodes);
    reduction_cursor = symmetrix::oam::checked_oam_add(
        reduction_cursor, nodes);
    active_reduced_forces_ = arena_vector<double>(
        "final_reductions", reduction_cursor,
        symmetrix::oam::checked_oam_multiply(std::size_t(3), nodes));
    reduction_cursor = symmetrix::oam::checked_oam_add(reduction_cursor,
        symmetrix::oam::checked_oam_multiply(std::size_t(3), nodes));
    active_stress_components_ = arena_vector<double>(
        "final_reductions", reduction_cursor, 6);
    receiver_owned_harmonic_adjoint_ = arena_vector<Precision>(
        "source_harmonic_partials", 0,
        active_memory_layout_.slot("source_harmonic_partials").bytes
            / sizeof(Precision));
    receiver_owned_distance_adjoint_ = arena_vector<double>(
        "source_distance_partials", 0,
        active_memory_layout_.slot("source_distance_partials").bytes
            / sizeof(double));
}

template <typename Precision>
std::size_t NequIPOAMKokkos<Precision>::retained_workspace_bytes() const
{
    // Count retained capacities, including after a smaller graph replaces a
    // larger one. Arena-backed runtime views are non-owning and therefore are
    // represented exactly once by the retained arena capacity.
    std::size_t bytes = sizeof(Precision) * (
        active_xyz_.size() + active_shuffled_xyz_.size()
        + active_geometry_adjoint_.size()
        + active_harmonics_.size() + active_harmonic_gradients_.size());
    bytes += sizeof(double) * (
        active_distances_.size() + active_stress_tensor_.size());
    bytes += transient_arena_state_.retained_capacity_bytes;
    return bytes;
}

template <typename Precision>
std::size_t NequIPOAMKokkos<Precision>::execution_linear_workspace_bytes() const
{
    return 0;
}

template <typename Precision>
std::size_t
NequIPOAMKokkos<Precision>::execution_packed_linear_workspace_bytes() const
{
    if (active_memory_layout_.slots.empty()) return 0;
    return active_memory_layout_.slot("packed_linear_scratch").bytes;
}

template <typename Precision>
std::size_t
NequIPOAMKokkos<Precision>::execution_bilinear_workspace_bytes() const
{
    if (active_memory_layout_.slots.empty()) return 0;
    return active_memory_layout_.slot("bilinear_forward_scratch").bytes;
}

template <typename Precision>
std::size_t
NequIPOAMKokkos<Precision>::execution_independent_graph_temporary_bytes() const
{
    return bilinear_workspace_->fixed_capacity
        ? 0 : bilinear_workspace_->bytes();
}

template <typename Precision>
std::size_t NequIPOAMKokkos<Precision>::execution_topology_bytes() const
{
    return sizeof(int) * (node_types_.size() + num_neigh_.size()
        + neigh_indices_.size() + neigh_types_.size() + edge_receivers_.size()
        + first_neigh_.size() + source_edge_offsets_.size()
        + source_edge_indices_.size() + reverse_segment_sources_.size()
        + reverse_segment_begin_.size() + reverse_segment_end_.size()
        + reverse_segment_rowptr_.size() + reverse_segment_order_.size());
}

template <typename Precision>
std::size_t NequIPOAMKokkos<Precision>::execution_workspace_bytes() const
{
    return retained_workspace_bytes();
}

template <typename Precision>
std::string NequIPOAMKokkos<Precision>::execution_kernel_summary() const
{
    const bool receiver_owned = receiver_owned_execution();
    bool packed_linear = true;
    std::ostringstream tensor_backends;
    for (std::size_t index = 0; index < layer_states_.size(); ++index) {
        const auto& state = layer_states_[index];
        packed_linear = packed_linear
            && state.pre_linear.selected_backend(num_nodes_) == "packed_gemm"
            && state.post_linear.selected_backend(num_nodes_) == "packed_gemm";
        if (state.self_linear)
            packed_linear = packed_linear
                && state.self_linear->selected_backend(num_nodes_) == "packed_gemm";
        if (index != 0) tensor_backends << ",";
        if (cuda_r1_plugin_ && static_cast<bool>(*cuda_r1_plugin_))
            tensor_backends << "generated_oam_cuda_full_message(layer"
                << state.generated_layer << ")";
        else if (receiver_owned)
            tensor_backends << "receiver_owned_full_message(layer"
                << state.generated_layer << ")";
        else if (streamed_edges_mode_ == "direct")
            tensor_backends << "unavailable_oam_direct(layer"
                << state.generated_layer << ")";
        else if (streamed_edges_mode_ == "reference")
            tensor_backends << "portable_oam_kokkos_generic(layer"
                << state.generated_layer << ")";
        else
            tensor_backends << "generated_oam_host_v1(layer" << state.generated_layer << ")";
    }
    packed_linear = packed_linear
        && readout_.selected_backend(num_nodes_) == "packed_gemm";
    std::size_t conditioned_self_schedule_bytes = 0;
    std::size_t conditioned_self_original_bytes = 0;
    for (const auto& state : layer_states_)
        if (state.conditioned_self.eligible) {
            conditioned_self_schedule_bytes +=
                state.conditioned_self.path_data.size() * sizeof(std::int64_t)
                + state.conditioned_self.wigner_diagonal.size() * sizeof(Precision)
                + state.conditioned_self.output_mask.size() * sizeof(Precision);
            conditioned_self_original_bytes += state.conditioned_self.original_tensor_bytes;
        }
    std::ostringstream summary;
    summary << "linear=" << (packed_linear ? "packed_gemm" : "mixed")
        << ", tensor_products=" << tensor_backends.str()
        << ", edge_variant=full_message"
        << ", edge_variant_reason=" << execution_edge_variant_reason()
        << ", generated_dispatch=";
    for (std::size_t index = 0; index < layer_states_.size(); ++index) {
        if (index != 0) summary << ",";
        summary << "layer" << layer_states_[index].generated_layer;
    }
    summary << ", workspace=full-message-control"
        // Measured retained graph-scaled state and the full retained-plus-chunk
        // footprint; the per-call node temporaries are not counted.
        << ", workspace_bytes=" << execution_workspace_bytes()
        << ", retained_workspace_bytes=" << retained_workspace_bytes()
        << ", topology_bytes=" << execution_topology_bytes()
        << ", linear_workspace_bytes=" << execution_linear_workspace_bytes()
        // Measured edge dispatch accounting: every region dispatches to a
        // loaded generated or receiver-owned callback (a layer without one
        // cannot be constructed), so the generic counters are the difference
        // between the work performed and the work that ran generated code.
        << ", edge_regions=" << edge_regions_
        << ", edge_generated_regions=" << edge_generated_regions_
        << ", edge_dispatches=" << edge_dispatches_
        << ", edge_generated_dispatches=" << edge_generated_dispatches_
        << ", edge_full_message_launches=" << edge_full_message_launches_
        << ", edge_generic_dispatches="
        << edge_dispatches_ - edge_generated_dispatches_
        << ", edge_generic_regions="
        << edge_regions_ - edge_generated_regions_
        << ", host_plugin_abi=" << (oam_host_plugin_v2_
            ? "v7" : (oam_host_plugin_ ? "v1" : "none"))
        // Which node-local self-connection path each layer executes: the
        // lane-major kernels for a matching block plan, the definition-order
        // fallback otherwise, or none when the layer has no self tensor
        // product.
        << ", self_tensor_product_layout=";
    bool has_self_tensor_product = false;
    for (const auto& state : layer_states_)
        if (state.self_tensor_product) has_self_tensor_product = true;
    if (!has_self_tensor_product)
        summary << "none";
    else if (self_definition_order_layers_.empty())
        summary << "lane";
    else {
        summary << "definition_order(";
        for (std::size_t index = 0; index < self_definition_order_layers_.size(); ++index) {
            if (index != 0) summary << ",";
            summary << "layer" << self_definition_order_layers_[index];
        }
        summary << ")";
    }
    summary << ", conditioned_self_layers=" << conditioned_self_layer_count()
        << ", conditioned_self_bytes=" << conditioned_self_bytes()
        << ", conditioned_self_schedule_bytes="
        << conditioned_self_schedule_bytes
        << ", conditioned_self_original_tensor_bytes="
        << conditioned_self_original_bytes
        << ", conditioned_self_forward_launches="
        << conditioned_self_forward_launch_count_
        << ", conditioned_self_reverse_launches="
        << conditioned_self_reverse_launch_count_;
    return summary.str() + cuda_r1_state_text();
}

template <typename Precision>
void NequIPOAMKokkos<Precision>::validate_graph_generation(
    const std::uint64_t generation) const
{
    if (!graph_prepared_ || generation == 0 || generation != graph_generation_)
        throw std::invalid_argument(
            "NequIPOAMKokkos graph generation does not match the prepared graph generation.");
}

template <typename Precision>
void NequIPOAMKokkos<Precision>::validate_geometry_generation(
    const std::uint64_t generation) const
{
    validate_graph_generation(generation);
    if (!geometry_prepared_ || geometry_generation_ != generation)
        throw std::invalid_argument(
            "NequIPOAMKokkos geometry generation does not match the prepared graph generation.");
}

template <typename Precision>
void NequIPOAMKokkos<Precision>::validate_completed_generation(
    const std::uint64_t generation) const
{
    validate_geometry_generation(generation);
    if (completed_generation_ != generation)
        throw std::invalid_argument(
            "NequIPOAMKokkos graph generation has no completed prepared evaluation.");
}

template <typename Precision>
std::uint64_t NequIPOAMKokkos<Precision>::prepare_graph(
    const int num_nodes, const std::span<const int> node_types,
    const std::span<const int> num_neigh, const std::span<const int> neigh_indices,
    const std::span<const int> neigh_types)
{
    if (num_nodes < 0 || node_types.size() != static_cast<std::size_t>(num_nodes)
        || num_neigh.size() != static_cast<std::size_t>(num_nodes))
        throw std::invalid_argument("NequIPOAMKokkos graph node arrays have invalid sizes.");
    const auto& atomic_numbers = model_.atomic_numbers();
    for (const int type : node_types)
        if (type < 0 || static_cast<std::size_t>(type) >= atomic_numbers.size())
            throw std::invalid_argument("NequIPOAMKokkos node type is outside the atomic-number map.");
    const std::size_t edges = checked_edge_count(num_neigh);
    if (neigh_indices.size() != edges || neigh_types.size() != edges)
        throw std::invalid_argument("NequIPOAMKokkos graph edge arrays have inconsistent sizes.");
    // Validation below mutates receiver metadata; no previous evaluation may
    // remain reducible if a replacement graph fails partway through.
    graph_prepared_ = false;
    geometry_prepared_ = false;
    completed_generation_ = 0;
    stress_tensor.clear();
    edge_receivers_host_.clear();
    edge_receivers_host_.reserve(edges);
    for (int receiver = 0; receiver < num_nodes; ++receiver) {
        for (int local = 0; local < num_neigh[static_cast<std::size_t>(receiver)]; ++local) {
            const std::size_t edge = edge_receivers_host_.size();
            const int source = neigh_indices[edge];
            if (source < 0 || source >= num_nodes)
                throw std::invalid_argument("NequIPOAMKokkos neighbor index is out of range.");
            if (neigh_types[edge] != node_types[static_cast<std::size_t>(source)])
                throw std::invalid_argument(
                    "NequIPOAMKokkos neighbor type does not match the source node type.");
            edge_receivers_host_.push_back(receiver);
        }
    }
    num_nodes_ = num_nodes;
    node_types_host_ = copy_span(node_types);
    num_neigh_host_ = copy_span(num_neigh);
    neigh_indices_host_ = copy_span(neigh_indices);
    neigh_types_host_ = copy_span(neigh_types);
    first_neigh_host_.assign(static_cast<std::size_t>(num_nodes_) + 1, 0);
    for (int node = 0; node < num_nodes_; ++node)
        first_neigh_host_[static_cast<std::size_t>(node) + 1]
            = first_neigh_host_[static_cast<std::size_t>(node)]
            + num_neigh_host_[static_cast<std::size_t>(node)];
    // Compact source-owned CSR: every directed edge is listed exactly once
    // under its source node so source adjoints need no atomics.
    source_edge_offsets_host_.assign(static_cast<std::size_t>(num_nodes_) + 1, 0);
    for (const int source : neigh_indices_host_)
        ++source_edge_offsets_host_[static_cast<std::size_t>(source) + 1];
    for (std::size_t node = 0; node < static_cast<std::size_t>(num_nodes_); ++node)
        source_edge_offsets_host_[node + 1] += source_edge_offsets_host_[node];
    source_edge_indices_host_.assign(edges, 0);
    {
        std::vector<int> cursor(
            source_edge_offsets_host_.begin(), source_edge_offsets_host_.end() - 1);
        for (std::size_t edge = 0; edge < edges; ++edge)
            source_edge_indices_host_[static_cast<std::size_t>(
                cursor[static_cast<std::size_t>(neigh_indices_host_[edge])]++)] =
                static_cast<int>(edge);
    }
    constexpr int reverse_segment_edges = 32;
    reverse_segment_sources_host_.clear();
    reverse_segment_begin_host_.clear();
    reverse_segment_end_host_.clear();
    reverse_segment_rowptr_host_.assign(
        static_cast<std::size_t>(num_nodes_) + 1, 0);
    for (int source = 0; source < num_nodes_; ++source) {
        const int first = source_edge_offsets_host_[static_cast<std::size_t>(source)];
        const int last = source_edge_offsets_host_[static_cast<std::size_t>(source) + 1];
        const int count = std::max(1,
            (last - first + reverse_segment_edges - 1) / reverse_segment_edges);
        for (int local = 0; local < count; ++local) {
            const int begin = std::min(last, first + local * reverse_segment_edges);
            reverse_segment_sources_host_.push_back(source);
            reverse_segment_begin_host_.push_back(begin);
            reverse_segment_end_host_.push_back(
                std::min(last, begin + reverse_segment_edges));
        }
        reverse_segment_rowptr_host_[static_cast<std::size_t>(source) + 1] =
            static_cast<int>(reverse_segment_sources_host_.size());
    }
    reverse_segment_order_host_.resize(reverse_segment_sources_host_.size());
    for (std::size_t segment = 0;
         segment < reverse_segment_order_host_.size(); ++segment)
        reverse_segment_order_host_[segment] = static_cast<int>(segment);
    std::stable_sort(reverse_segment_order_host_.begin(),
        reverse_segment_order_host_.end(), [&](const int first, const int second) {
            const int first_length = reverse_segment_end_host_[first]
                - reverse_segment_begin_host_[first];
            const int second_length = reverse_segment_end_host_[second]
                - reverse_segment_begin_host_[second];
            if (first_length != second_length)
                return first_length > second_length;
            const int first_source = reverse_segment_sources_host_[first];
            const int second_source = reverse_segment_sources_host_[second];
            const int first_degree = source_edge_offsets_host_[first_source + 1]
                - source_edge_offsets_host_[first_source];
            const int second_degree = source_edge_offsets_host_[second_source + 1]
                - source_edge_offsets_host_[second_source];
            return first_degree > second_degree;
        });
    ++generation_counter_;
    if (generation_counter_ == 0) generation_counter_ = 1;
    graph_generation_ = generation_counter_;
    geometry_generation_ = 0;
    completed_generation_ = 0;
    graph_prepared_ = true;
    geometry_prepared_ = false;
    copy_device_vector_reuse(node_types_, "oam node types", node_types_host_);
    copy_device_vector_reuse(num_neigh_, "oam node degrees", num_neigh_host_);
    copy_device_vector_reuse(neigh_indices_, "oam edge sources", neigh_indices_host_);
    copy_device_vector_reuse(neigh_types_, "oam edge types", neigh_types_host_);
    copy_device_vector_reuse(edge_receivers_, "oam edge receivers", edge_receivers_host_);
    copy_device_vector_reuse(first_neigh_, "oam receiver offsets", first_neigh_host_);
    copy_device_vector_reuse(
        source_edge_offsets_, "oam source edge offsets", source_edge_offsets_host_);
    copy_device_vector_reuse(
        source_edge_indices_, "oam source edge indices", source_edge_indices_host_);
    copy_device_vector_reuse(reverse_segment_sources_,
        "oam reverse segment sources", reverse_segment_sources_host_);
    copy_device_vector_reuse(reverse_segment_begin_,
        "oam reverse segment begin", reverse_segment_begin_host_);
    copy_device_vector_reuse(reverse_segment_end_,
        "oam reverse segment end", reverse_segment_end_host_);
    copy_device_vector_reuse(reverse_segment_rowptr_,
        "oam reverse segment rowptr", reverse_segment_rowptr_host_);
    copy_device_vector_reuse(reverse_segment_order_,
        "oam reverse segment order", reverse_segment_order_host_);
    last_execution_diagnostic_ =
        "NequIP OAM Kokkos execution: backend=" + execution_backend()
        + ", implementation=" + execution_implementation_label() + ", artifact_id="
        + execution_artifact_id()
        + ", operation=prepare_graph, active_edges=" + std::to_string(edges)
        + ", graph_generation=" + std::to_string(graph_generation_)
        + ", receiver_chunk=bounded, reduction=" + execution_reduction_mode_
        + ", " + execution_kernel_summary();
    // Candidate estimates and the selection reason describe this graph, not
    // the empty constructor state.
    refresh_execution_plan_report();
    return graph_generation_;
}

template <typename Precision>
void NequIPOAMKokkos<Precision>::prepare_geometry(
    const std::uint64_t generation, const std::span<const double> reference_positions,
    const std::span<const double> reference_xyz, const std::span<const double> cell,
    const std::span<const double> inverse_cell, const std::span<const int> pbc)
{
    validate_graph_generation(generation);
    if (reference_positions.size() != 3 * static_cast<std::size_t>(num_nodes_)
        || reference_xyz.size() != 3 * edge_count())
        throw std::invalid_argument("NequIPOAMKokkos geometry arrays have invalid sizes.");
    const bool nonperiodic = cell.empty() && inverse_cell.empty() && pbc.empty();
    const bool periodic = cell.size() == 9 && inverse_cell.size() == 9 && pbc.size() == 3;
    if ((!nonperiodic && !periodic) || !nequip_validation::all_finite(reference_positions)
        || !nequip_validation::all_finite(reference_xyz) || !nequip_validation::all_finite(cell)
        || !nequip_validation::all_finite(inverse_cell))
        throw std::invalid_argument("NequIPOAMKokkos geometry requires finite reference arrays.");
    geometry_prepared_ = false;
    completed_generation_ = 0;
    stress_tensor.clear();
    reference_positions_host_ = copy_span(reference_positions);
    reference_xyz_host_ = copy_span(reference_xyz);
    cell_host_ = copy_span(cell);
    inverse_cell_host_ = copy_span(inverse_cell);
    pbc_host_ = copy_span(pbc);
    geometry_generation_ = generation;
    geometry_prepared_ = true;
    last_execution_diagnostic_ =
        "NequIP OAM Kokkos execution: backend=" + execution_backend()
        + ", implementation=" + execution_implementation_label() + ", artifact_id="
        + execution_artifact_id()
        + ", operation=prepare_geometry, " + execution_kernel_summary();
}

template <typename Precision>
void NequIPOAMKokkos<Precision>::reduce_forces(const std::uint64_t generation)
{
    validate_completed_generation(generation);
    if (active_geometry_adjoint_.size() < 3 * edge_count())
        throw std::invalid_argument(
            "NequIP OAM Kokkos reverse force adjoints are not available.");
    // Reverse retains +dE/dr on device. Consume its negative directly rather
    // than reuploading host forces and overwriting the stress input's sign.
    const std::size_t reduced_size = 3 * static_cast<std::size_t>(num_nodes_);
    symmetrix_oam_detail::ensure_view_capacity(active_reduced_forces_,
        "oam reduced atom forces", reduced_size);
    if (reduced_size != 0)
        Kokkos::deep_copy(
            Kokkos::subview(active_reduced_forces_,
                std::make_pair(std::size_t(0), reduced_size)),
            0.0);
    const auto reduced_forces = active_reduced_forces_;
    const auto directed = active_geometry_adjoint_;
    const auto receivers = edge_receivers_;
    const auto sources = neigh_indices_;
    Kokkos::parallel_for(
        "NequIPOAMKokkos reduce directed forces",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
            Kokkos::IndexType<std::size_t>>(0, edge_count()),
        KOKKOS_LAMBDA(const std::size_t edge) {
            const int receiver = receivers(edge);
            const int source = sources(edge);
            for (int component = 0; component < 3; ++component) {
                const double force = -static_cast<double>(
                    directed(3 * edge + component));
                Kokkos::atomic_add(
                    &reduced_forces(3 * static_cast<std::size_t>(source) + component), force);
                Kokkos::atomic_add(
                    &reduced_forces(3 * static_cast<std::size_t>(receiver) + component), -force);
            }
        });
    if (reduced_size != 0) {
        const auto host_reduced = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),
            Kokkos::subview(active_reduced_forces_,
                std::make_pair(std::size_t(0), reduced_size)));
        atom_forces.assign(host_reduced.data(), host_reduced.data() + reduced_size);
    } else {
        atom_forces.clear();
    }
    Kokkos::fence();
    update_diagnostic("reduce_forces");
}

template <typename Precision>
void NequIPOAMKokkos<Precision>::reduce_stress(
    const double volume, const std::uint64_t generation)
{
    validate_completed_generation(generation);
    if (!nequip_validation::is_finite(volume) || !(volume > 0.0))
        throw std::invalid_argument("NequIP OAM Kokkos stress requires a finite positive volume.");
    const std::size_t edges = edge_count();
    if (active_xyz_.size() < 3 * edges || active_geometry_adjoint_.size() < 3 * edges)
        throw std::invalid_argument("NequIP OAM Kokkos stress requires evaluated geometry adjoints.");
    symmetrix_oam_detail::ensure_view_capacity(
        active_stress_components_, "oam symmetric virial", 6);
    symmetrix_oam_detail::ensure_view_capacity(
        active_stress_tensor_, "oam stress tensor", 9);
    const auto xyz = active_xyz_;
    const auto adjoint = active_geometry_adjoint_;
    const auto components = active_stress_components_;
    const auto tensor = active_stress_tensor_;
    // ASE positive-tension convention: sym(sum(r_edge outer dE/dr))/volume.
    // Vectors include periodic images and adjoints include ZBL. The directed
    // energy contract already includes pair normalization: no extra edge half.
    for (int component = 0; component < 6; ++component) {
        const int row = component < 3 ? component : (component == 3 ? 1 : 0);
        const int column = component < 3 ? component : (component == 5 ? 1 : 2);
        Kokkos::parallel_reduce(
            "NequIPOAMKokkos reduce stress component",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                Kokkos::IndexType<std::size_t>>(0, edges),
            KOKKOS_LAMBDA(const std::size_t edge, double& sum) {
                const double value = static_cast<double>(xyz(3 * edge + row))
                    * static_cast<double>(adjoint(3 * edge + column));
                if (row == column)
                    sum += value;
                else
                    sum += 0.5 * (value + static_cast<double>(xyz(3 * edge + column))
                        * static_cast<double>(adjoint(3 * edge + row)));
            },
            Kokkos::subview(components, component));
    }
    Kokkos::parallel_for(
        "NequIPOAMKokkos assemble stress tensor",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 9),
        KOKKOS_LAMBDA(const int index) {
            const int row = index / 3;
            const int column = index % 3;
            const int component = row == column ? row
                : ((row == 0 || column == 0)
                    ? ((row == 1 || column == 1) ? 5 : 4) : 3);
            tensor(index) = components(component) / volume;
        });
    const auto host_tensor = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), active_stress_tensor_);
    stress_tensor.assign(host_tensor.data(), host_tensor.data() + 9);
    update_diagnostic("reduce_stress");
}

template <typename Precision>
void NequIPOAMKokkos<Precision>::compute_stress()
{
    validate_completed_generation(graph_generation_);
    if (cell_host_.size() != 9)
        throw std::invalid_argument("NequIP OAM Kokkos stress requires a prepared cell volume.");
    const auto& cell = cell_host_;
    const double volume = std::abs(
        cell[0] * (cell[4] * cell[8] - cell[5] * cell[7])
        - cell[1] * (cell[3] * cell[8] - cell[5] * cell[6])
        + cell[2] * (cell[3] * cell[7] - cell[4] * cell[6]));
    if (!nequip_validation::is_finite(volume) || !(volume > 0.0))
        throw std::invalid_argument("NequIP OAM Kokkos prepared cell requires a finite positive volume.");
    reduce_stress(volume, graph_generation_);
}

template <typename Precision>
void NequIPOAMKokkos<Precision>::update_diagnostic(const char* operation)
{
    last_execution_diagnostic_ =
        "NequIP OAM Kokkos execution: backend=" + execution_backend()
        + ", implementation=" + execution_implementation_label() + ", operation=" + operation
        + ", artifact_id=" + execution_artifact_id()
        + ", streamed_edges=" + streamed_edges_mode_
        + ", " + execution_kernel_summary();
}

template <typename Precision>
symmetrix::execution::FamilyCapabilities
NequIPOAMKokkos<Precision>::family_capabilities() const
{
    using symmetrix::execution::FamilyCapabilities;
    FamilyCapabilities capabilities;
    capabilities.target = execution_target();
    capabilities.precision =
        std::is_same_v<Precision, float> ? "float32" : "float64";
    // The reference routing uses the generic Kokkos edge product in the
    // evaluator's default execution space.  It therefore needs no generated
    // artifact on either a host or device build.
    const bool supported_target = execution_space_is_host()
        || execution_space_is_cuda();
    capabilities.reference_ready = supported_target
        && has_portable_edge_implementation();
    capabilities.portable_ready = supported_target
        && has_portable_edge_implementation();
    capabilities.generated_ready = generated_edges_serve();
    // The stress reduction is execution-space generic: the portable path
    // populates the geometry adjoint it reduces on both host and device.
    capabilities.stress = supported_target;
    capabilities.field = false;
    capabilities.parameter_gradients = false;
    capabilities.loads_host_artifact = execution_space_is_host();
    // A CUDA artifact binds the device execution space, so a build whose
    // execution space is the host one cannot load one.
    capabilities.loads_device_artifact = execution_space_is_cuda();
    return capabilities;
}

template <typename Precision>
symmetrix::execution::Program NequIPOAMKokkos<Precision>::compile_program() const
{
    using symmetrix::execution::BufferContract;
    using symmetrix::execution::LayoutContract;
    using symmetrix::execution::OperatorKind;
    using symmetrix::execution::OperatorNode;
    using symmetrix::execution::Program;
    const auto& schema = model_.schema();
    const char* precision_name =
        std::is_same_v<Precision, float> ? "float32" : "float64";
    const auto node_buffer = [precision_name] (
        const std::string& name, const std::string& irreps) {
        BufferContract buffer;
        buffer.name = name;
        buffer.dtype = precision_name;
        buffer.shape = {0, static_cast<std::size_t>(Irreps(irreps).dimension())};
        buffer.layout = LayoutContract{"lane_major", irreps, 1};
        return buffer;
    };
    const auto edge_buffer = [precision_name] (
        const std::string& name, const std::string& irreps) {
        BufferContract buffer;
        buffer.name = name;
        buffer.dtype = precision_name;
        buffer.shape = {0, static_cast<std::size_t>(Irreps(irreps).dimension())};
        buffer.layout = LayoutContract{"edge_major", irreps, 1};
        return buffer;
    };
    Program program;
    program.family = "OAM v2 contract";
    program.precision = precision_name;
    program.version = 1;
    const auto append = [&program] (OperatorNode node) {
        program.operators.push_back(std::move(node));
        return &program.operators.back();
    };
    {
        OperatorNode node;
        node.kind = OperatorKind::geometry;
        node.id = "oam.geometry";
        node.capability_key = "geometry";
        node.inputs.push_back(node_buffer("reference_positions", "1x0e"));
        node.outputs.push_back(edge_buffer("edge_geometry", "1x1o"));
        node.parameters.emplace_back("r_cut", schema.r_cut);
        append(std::move(node));
    }
    for (std::size_t layer = 0; layer < schema.layers.size(); ++layer) {
        const auto& data = schema.layers[layer];
        const std::string prefix = "oam.interaction" + std::to_string(layer);
        const auto& radial = schema.kokkos_radial.at(layer);
        int edge_l_max = 0;
        for (const auto& block : Irreps(data.edge_irreps).blocks)
            edge_l_max = std::max(edge_l_max, block.l);
        append([&] {
            OperatorNode node;
            node.kind = OperatorKind::radial;
            node.id = prefix + ".radial";
            node.capability_key = "radial";
            node.inputs.push_back(edge_buffer("edge_distances", "1x0e"));
            node.outputs.push_back(edge_buffer("radial_weights", "1x0e"));
            node.parameters.emplace_back(
                "intervals", static_cast<double>(radial.num_points));
            node.parameters.emplace_back(
                "functions", static_cast<double>(radial.function_count));
            node.parameters.emplace_back("layer", static_cast<double>(layer));
            return node;
        }());
        append([&] {
            OperatorNode node;
            node.kind = OperatorKind::spherical_harmonic;
            node.id = prefix + ".harmonics";
            node.capability_key = "harmonics";
            node.inputs.push_back(edge_buffer("edge_vectors", "1x1o"));
            node.outputs.push_back(
                edge_buffer("edge_harmonics", data.edge_irreps));
            node.parameters.emplace_back("l_max", static_cast<double>(edge_l_max));
            return node;
        }());
        append([&] {
            OperatorNode node;
            node.kind = OperatorKind::tensor_product;
            node.id = prefix + ".edge";
            node.capability_key = "edge_tensor_product";
            node.inputs.push_back(node_buffer(
                "source_node_values", data.node_irreps_in));
            node.inputs.push_back(edge_buffer("edge_features", data.edge_irreps));
            node.inputs.push_back(edge_buffer("radial_weights", "1x0e"));
            node.outputs.push_back(node_buffer(
                "edge_messages", data.tp_output_irreps));
            node.parameters.emplace_back("layer", static_cast<double>(layer));
            return node;
        }());
        append([&] {
            OperatorNode node;
            node.kind = OperatorKind::linear;
            node.id = prefix + ".pre_linear";
            node.capability_key = "pre_linear";
            node.inputs.push_back(node_buffer("node_features", data.node_irreps_in));
            node.outputs.push_back(node_buffer(
                "source_node_values", data.node_irreps_in));
            node.parameters.emplace_back("layer", static_cast<double>(layer));
            return node;
        }());
        if (!data.self_connection.is_null()) {
            const bool bilinear = data.self_connection.contains("irreps_in1");
            append([&] {
                OperatorNode node;
                node.kind = bilinear ? OperatorKind::tensor_product
                                     : OperatorKind::linear;
                node.id = prefix + ".self_connection";
                node.capability_key = "self_connection";
                node.inputs.push_back(node_buffer(
                    "node_features", data.node_irreps_in));
                node.outputs.push_back(node_buffer(
                    "self_features", data.node_irreps_pre_gate));
                return node;
            }());
        }
        append([&] {
            OperatorNode node;
            node.kind = OperatorKind::gate;
            node.id = prefix + ".gate";
            node.capability_key = "gate";
            node.inputs.push_back(node_buffer(
                "pre_gate_features", data.node_irreps_pre_gate));
            node.outputs.push_back(node_buffer(
                "node_features", data.node_irreps_out));
            return node;
        }());
        append([&] {
            OperatorNode node;
            node.kind = OperatorKind::linear;
            node.id = prefix + ".post_linear";
            node.capability_key = "post_linear";
            node.inputs.push_back(node_buffer(
                "edge_messages", data.tp_output_irreps));
            node.outputs.push_back(node_buffer(
                "pre_gate_features", data.node_irreps_pre_gate));
            node.parameters.emplace_back("layer", static_cast<double>(layer));
            return node;
        }());
    }
    append([&] {
        OperatorNode node;
        node.kind = OperatorKind::readout;
        node.id = "oam.readout";
        node.capability_key = "readout";
        node.inputs.push_back(node_buffer("node_features",
            schema.layers.empty() ? "1x0e"
                                  : schema.layers.back().node_irreps_out));
        node.outputs.push_back(node_buffer("node_energy", "1x0e"));
        return node;
    }());
    append([&] {
        OperatorNode node;
        node.kind = OperatorKind::pair_energy;
        node.id = "oam.pair_energy";
        node.capability_key = "pair_energy";
        node.inputs.push_back(edge_buffer("edge_distances", "1x0e"));
        node.parameters.emplace_back("r_cut", schema.r_cut);
        return node;
    }());
    for (const char* key : {"reduce_forces", "reduce_stress"}) {
        append([&] {
            OperatorNode node;
            node.kind = OperatorKind::reduction;
            node.id = std::string("oam.") + key;
            node.capability_key = key;
            node.inputs.push_back(node_buffer("node_adjoint", "1x0e"));
            return node;
        }());
    }
    program.fingerprint = symmetrix::execution::program_fingerprint(program);
    return program;
}

template <typename Precision>
symmetrix::execution::ImplementationRegistry
NequIPOAMKokkos<Precision>::family_implementation_registry() const
{
    using symmetrix::execution::ImplementationDescriptor;
    using symmetrix::execution::ImplementationRegistry;
    using symmetrix::execution::ImplementationTier;
    using symmetrix::execution::OperatorKind;
    using symmetrix::execution::PrecisionPolicy;
    // The portable tier is the compiler-free generic Kokkos implementation; the
    // generated tier is the host or CUDA tensor-product artifact.  Every
    // predicate reads this evaluator's own readiness state.
    const std::string target = execution_target();
    const std::string precision_name =
        std::is_same_v<Precision, float> ? "float32" : "float64";
    const PrecisionPolicy policy{precision_name, "float64"};
    // The generated tier is qualified exactly when generated code serves this
    // execution: a loaded artifact the direct-mode dispatcher never runs (the
    // edge-owned ABI v1 artifact under a direct request) reports the portable
    // tier it actually serves instead.
    const bool generated = generated_edges_serve();
    ImplementationRegistry registry;
    const auto declare = [&registry, target, policy] (
        const std::string& id, const OperatorKind kind,
        const std::string& capability_key, const ImplementationTier tier,
        std::function<bool()> qualified) {
        ImplementationDescriptor descriptor;
        descriptor.id = id;
        descriptor.kind = kind;
        descriptor.capability_key = capability_key;
        descriptor.tier = tier;
        descriptor.target = target;
        descriptor.precision = policy;
        descriptor.capabilities.target = target;
        descriptor.capabilities.precision = policy.compute;
        descriptor.qualified =
            [available = std::move(qualified)] (const auto&) {
                return available();
            };
        registry.declare(std::move(descriptor));
    };
    declare("generic_oam_geometry", OperatorKind::geometry, "geometry",
        ImplementationTier::portable, [] { return true; });
    declare("generic_oam_radial", OperatorKind::radial, "radial",
        ImplementationTier::portable, [] { return true; });
    declare("generic_oam_harmonics", OperatorKind::spherical_harmonic,
        "harmonics", ImplementationTier::portable, [] { return true; });
    declare("portable_oam_kokkos_generic", OperatorKind::tensor_product,
        "edge_tensor_product", ImplementationTier::portable,
        [this] { return streamed_edges_mode_ == "reference"
            && has_portable_edge_implementation(); });
    declare(execution_implementation_label(), OperatorKind::tensor_product,
        "edge_tensor_product", ImplementationTier::generated,
        [this, generated] { return generated; });
    for (const char* key : {"pre_linear", "post_linear"}) {
        declare(std::string("generic_oam_") + key, OperatorKind::linear, key,
            ImplementationTier::portable, [] { return true; });
    }
    declare("generic_oam_readout", OperatorKind::readout, "readout",
        ImplementationTier::portable, [] { return true; });
    declare("generic_oam_self_connection", OperatorKind::tensor_product,
        "self_connection", ImplementationTier::portable, [] { return true; });
    declare("generic_oam_self_linear", OperatorKind::linear,
        "self_connection", ImplementationTier::portable, [] { return true; });
    declare("generic_oam_gate", OperatorKind::gate, "gate",
        ImplementationTier::portable, [] { return true; });
    declare("generic_oam_pair_energy", OperatorKind::pair_energy,
        "pair_energy", ImplementationTier::portable, [] { return true; });
    declare("generic_oam_reduce_forces", OperatorKind::reduction,
        "reduce_forces", ImplementationTier::portable, [] { return true; });
    // The reduction is execution-space generic and the portable reverse
    // populates the geometry adjoint it consumes, so it is the qualified
    // implementation on every backend.
    declare("generic_oam_reduce_stress", OperatorKind::reduction,
        "reduce_stress", ImplementationTier::portable, [] { return true; });
    return registry;
}

template <typename Precision>
bool NequIPOAMKokkos<Precision>::family_artifact_ready() const
{
    if (execution_space_is_cuda())
        return cuda_r1_plugin_ && static_cast<bool>(*cuda_r1_plugin_);
    return execution_space_is_host() && jit_oam_host_plugin_ready();
}

template <typename Precision>
symmetrix::execution::ArtifactReadiness
NequIPOAMKokkos<Precision>::family_artifact_readiness() const
{
    // The family-level query describes the artifact class this execution space
    // binds: a device build binds its CUDA artifact, a host build its host one.
    return artifact_readiness_for(execution_target());
}

template <typename Precision>
symmetrix::execution::ArtifactReadiness
NequIPOAMKokkos<Precision>::artifact_readiness_for(
    const std::string& target) const
{
    using symmetrix::execution::ArtifactReadiness;
    ArtifactReadiness readiness;
    readiness.target = target;
    if (target == "host") {
        readiness.ready = jit_oam_host_plugin_ready();
        readiness.artifact_id = jit_oam_host_plugin_artifact_id();
        readiness.generation_fingerprint = host_artifact_generation_fingerprint_;
        readiness.semantic_fingerprint = host_artifact_semantic_fingerprint_;
        readiness.structure_fingerprint = host_artifact_structure_fingerprint_;
        readiness.reason = readiness.ready
            ? "generated OAM host artifact admitted"
            : "no generated OAM host artifact is admitted; the portable host "
              "implementation remains available in reference mode";
        return readiness;
    }
    const bool device = cuda_r1_plugin_ && static_cast<bool>(*cuda_r1_plugin_);
    readiness.ready = device;
    readiness.artifact_id = device ? cuda_r1_artifact_id_ : std::string();
    // The generated CUDA identity publishes no separate generation
    // fingerprint: the ABI field named contract_fingerprint carries the
    // generated architecture fingerprint for OAM and is reported as the
    // semantic identity.
    readiness.semantic_fingerprint = cuda_r1_identity_.contract_fingerprint;
    readiness.structure_fingerprint = cuda_r1_identity_.structure_fingerprint;
    readiness.reason = device
        ? "generated OAM device artifact admitted"
        : "no generated OAM device artifact is admitted on target '" + target
            + "'; the portable Kokkos implementation remains available in "
              "reference mode";
    return readiness;
}

template <typename Precision>
symmetrix::execution::ArtifactReadiness
NequIPOAMKokkos<Precision>::load_family_artifact(
    const symmetrix::execution::ArtifactRequest& request)
{
    symmetrix::execution::ArtifactRequest resolved = request;
    if (resolved.target.empty())
        resolved.target = execution_target();
    // An empty path is the interface's readiness query and must not reach a
    // loader: the host loader resolves its identity manifest relative to the
    // plugin path, so an empty path would read the working directory instead.
    if (!resolved.path.empty()) {
        if (resolved.target != "host" && resolved.target != "cuda") {
            throw std::invalid_argument(
                "NequIP OAM generated artifacts are unsupported on target '"
                + resolved.target + "'.");
        }
        if (resolved.target == "host") {
            // The same loader the constructor uses: the receiver/source ABI v7
            // contract is probed first and the edge-owned v1 artifact is the
            // fallback.
            load_host_artifact(resolved.path);
        } else {
            symmetrix::execution::CudaR1ArtifactIdentity identity;
            identity.artifact_id = cuda_r1_artifact_id_;
            initialize_cuda_r1(resolved.path, identity);
        }
        // A loaded artifact changes which implementation the execution path
        // dispatches, so the plan report follows it.  The empty-path readiness
        // query loads nothing and therefore recomputes nothing.
        refresh_execution_plan_report();
    }
    return artifact_readiness_for(resolved.target);
}

template <typename Precision>
symmetrix::execution::FamilyDiagnostics
NequIPOAMKokkos<Precision>::family_diagnostics() const
{
    using symmetrix::execution::FamilyDiagnostics;
    FamilyDiagnostics diagnostics;
    diagnostics.fallback_count = cuda_r1_fallback_count_;
    diagnostics.diagnostic = last_execution_diagnostic_;
    return diagnostics;
}

template <typename Precision>
symmetrix::execution::ImplementationBindingReport
NequIPOAMKokkos<Precision>::family_binding_report() const
{
    const auto capabilities = family_capabilities();
    return symmetrix::execution::plan_program(compile_program(),
        family_implementation_registry(), capabilities.target,
        capabilities.precision, family_artifact_ready(),
        family_artifact_readiness().artifact_id);
}

template <typename Precision>
void NequIPOAMKokkos<Precision>::refresh_execution_plan_report()
{
    const auto bindings = family_binding_report();
    execution_plan_.requested_algorithm = streamed_edges_mode_ == "direct"
        ? ExecutionMode::direct
        // ``debug_v1`` is the family's explicit legacy edge-owned mode, which is
        // not the prepared direct path; it reports the neutral non-direct mode
        // while ``selected_id`` names the implementation it really dispatches.
        : ExecutionMode::generic;
    if (execution_plan_.selection_source == "request")
        execution_plan_.selection_source = "native_request";
    execution_plan_.state = "active";
    execution_plan_.boundary_attempt = false;
    if (streamed_edges_mode_ == "reference") {
        execution_plan_.selected_id = "portable_oam_kokkos_generic";
        execution_plan_.selection_reason =
            "portable reference execution is artifact-free and outside direct "
            "candidate ranking";
        execution_plan_.candidates.clear();
    } else if (streamed_edges_mode_ != "direct") {
        execution_plan_.selected_id = "edge_owned_oam_host_v1";
        execution_plan_.selection_reason =
            "legacy debug execution is outside direct candidate ranking";
        execution_plan_.candidates.clear();
    } else {
        execution_plan_.candidates = plan_candidate_space().evaluate();
        execution_plan_.selected_id.clear();
        execution_plan_.state = "unqualified";
        const auto first_qualified = std::find_if(
            execution_plan_.candidates.begin(), execution_plan_.candidates.end(),
            [] (const auto& candidate) { return candidate.qualified; });
        if (first_qualified == execution_plan_.candidates.end()) {
            execution_plan_.selection_reason =
                "no empirically qualified direct candidate matches the active "
                "backend, precision, schedule, and required capability";
        } else if (execution_plan_.requested_profile
            == symmetrix::execution::ExecutionProfile::speed) {
            execution_plan_.selected_id = first_qualified->id;
            execution_plan_.state = "active";
            execution_plan_.selection_reason =
                "selected the fastest qualified direct plan";
        } else if (!execution_plan_memory_available_) {
            execution_plan_.selected_id = first_qualified->id;
            execution_plan_.state = "active";
            execution_plan_.selection_reason =
                "selected the fastest qualified direct plan because the advisory "
                "memory budget is unavailable";
        } else {
            const std::size_t usable = execution_plan_.available_bytes
                    > execution_plan_.reserve_bytes
                ? execution_plan_.available_bytes - execution_plan_.reserve_bytes
                : 0;
            const auto fitting = std::find_if(execution_plan_.candidates.begin(),
                execution_plan_.candidates.end(), [usable] (const auto& candidate) {
                    return candidate.qualified && !candidate.fixed_workspace
                        && candidate.estimated_bytes
                        && *candidate.estimated_bytes <= usable;
                });
            if (fitting != execution_plan_.candidates.end()) {
                execution_plan_.selected_id = fitting->id;
                execution_plan_.state = "active";
                execution_plan_.selection_reason =
                    "selected the fastest qualified plan fitting the advisory budget";
            } else {
                const symmetrix::execution::ExecutionPlanCandidate* smallest = nullptr;
                for (const auto& candidate : execution_plan_.candidates) {
                    if (!candidate.qualified || candidate.fixed_workspace
                        || !candidate.estimated_bytes)
                        continue;
                    if (smallest == nullptr
                        || *candidate.estimated_bytes < *smallest->estimated_bytes)
                        smallest = &candidate;
                }
                if (smallest == nullptr) {
                    execution_plan_.selection_reason =
                        "no qualified non-fixed candidate has an exact estimate; "
                        "no boundary candidate can be chosen";
                } else {
                    execution_plan_.selected_id = smallest->id;
                    execution_plan_.state = "active";
                    execution_plan_.boundary_attempt = true;
                    execution_plan_.selection_reason =
                        "no qualified plan fits the advisory budget; attempting the "
                        "smallest qualified non-fixed plan at the capacity boundary";
                }
            }
        }
    }
    // The per-node tiers are the plan's own binding report, so a consumer sees
    // a portable node next to a generated one.
    execution_plan_.bindings = bindings.bindings;
    execution_plan_.binding_report_version = bindings.version;
}

template <typename Precision>
symmetrix::execution::PlanCandidateSpace
NequIPOAMKokkos<Precision>::plan_candidate_space() const
{
    using symmetrix::execution::PlanCandidateEvaluation;
    using symmetrix::execution::PlanCandidateSpace;
    const bool generated = streamed_edges_mode_ == "direct" && generated_edges_serve();
    const std::string current_artifact_identity =
        family_artifact_readiness().artifact_id;
    const bool speed_identity_matches =
        execution_plan_speed_qualification_.artifact_identity
        == current_artifact_identity;
    const bool speed_stale = execution_plan_speed_qualification_.admitted
        && (!speed_identity_matches
            || execution_plan_speed_qualification_.layout_identity
                != execution_memory_layout_identity(
                    symmetrix::oam::OAMRetentionPolicy::speed));
    const bool speed_layout_matches =
        execution_plan_speed_qualification_.layout_identity
        == execution_memory_layout_identity(
            symmetrix::oam::OAMRetentionPolicy::speed);
    const bool speed_admitted = generated
        && execution_plan_speed_qualification_.admitted && speed_identity_matches
        && speed_layout_matches;
    const bool capacity_identity_matches =
        execution_plan_capacity_qualification_.artifact_identity
        == current_artifact_identity;
    const bool capacity_stale = execution_plan_capacity_qualification_.admitted
        && (!capacity_identity_matches
            || execution_plan_capacity_qualification_.layout_identity
                != execution_memory_layout_identity(
                    symmetrix::oam::OAMRetentionPolicy::capacity));
    const bool capacity_layout_matches =
        execution_plan_capacity_qualification_.layout_identity
        == execution_memory_layout_identity(
            symmetrix::oam::OAMRetentionPolicy::capacity);
    const bool capacity_admitted =
        generated && execution_plan_capacity_qualification_.admitted
        && capacity_identity_matches && capacity_layout_matches;
    const std::size_t speed_estimate = execution_memory_layout(
        symmetrix::oam::OAMRetentionPolicy::speed).active_bytes;
    const std::size_t capacity_estimate = execution_memory_layout(
        symmetrix::oam::OAMRetentionPolicy::capacity).active_bytes;
    PlanCandidateSpace space;
    space.declare("oam-direct-speed",
        [this, speed_admitted, speed_stale, speed_identity_matches, generated,
            speed_estimate] {
        return PlanCandidateEvaluation{speed_admitted,
            speed_estimate,
            generated && speed_stale
                ? (speed_identity_matches
                    ? "stored qualification layout identity is stale"
                    : "stored qualification artifact identity is stale")
            : generated ? execution_plan_speed_qualification_.reason
                      : "generated artifact identity/backend/precision capability is not admitted"};
    });
    space.declare("oam-direct-capacity",
        [this, capacity_admitted, capacity_stale, capacity_identity_matches,
            generated, capacity_estimate] {
        return PlanCandidateEvaluation{capacity_admitted,
            capacity_estimate,
            generated && capacity_stale
                ? (capacity_identity_matches
                    ? "stored qualification layout identity is stale"
                    : "stored qualification artifact identity is stale")
            : generated ? execution_plan_capacity_qualification_.reason
                      : "generated artifact identity/backend/precision capability is not admitted"};
    });
    space.declare("oam-fixed-workspace-v1", [] {
        return PlanCandidateEvaluation{
            false, std::nullopt,
            "unavailable: fixed-workspace OAM execution is not implemented", true};
    });
    return space;
}

template <typename Precision>
void NequIPOAMKokkos<Precision>::permute_row_layout(
    Kokkos::View<const Precision**, Kokkos::LayoutRight> source,
    Kokkos::View<Precision**, Kokkos::LayoutRight> destination,
    Kokkos::View<int**, Kokkos::LayoutRight> plan, const bool to_lane) const
{
    const int blocks = static_cast<int>(plan.extent(0));
    Kokkos::parallel_for(
        "NequIPOAMKokkos permute node rows",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
            Kokkos::IndexType<int>>(0, static_cast<int>(source.extent(0))),
        KOKKOS_LAMBDA(const int row) {
            for (int block = 0; block < blocks; ++block) {
                const int offset = plan(block, 0);
                const int multiplicity = plan(block, 1);
                const int width = plan(block, 2);
                for (int channel = 0; channel < multiplicity; ++channel)
                    for (int component = 0; component < width; ++component) {
                        const int node_index =
                            offset + channel * width + component;
                        const int lane_index =
                            offset + component * multiplicity + channel;
                        destination(row, to_lane ? lane_index : node_index)
                            = source(row, to_lane ? node_index : lane_index);
                    }
            }
        });
}

template <typename Precision>
void NequIPOAMKokkos<Precision>::gather_attribute_rows(
    Kokkos::View<Precision**, Kokkos::LayoutRight> destination,
    Kokkos::View<int**, Kokkos::LayoutRight> plan) const
{
    const int blocks = static_cast<int>(plan.extent(0));
    const auto attributes = node_attributes_;
    const auto type_map = type_to_model_;
    const auto node_types = node_types_;
    Kokkos::parallel_for(
        "NequIPOAMKokkos gather attribute rows",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
            Kokkos::IndexType<int>>(0, num_nodes_),
        KOKKOS_LAMBDA(const int node) {
            const int model_type = type_map(node_types(node));
            for (int block = 0; block < blocks; ++block) {
                const int offset = plan(block, 0);
                const int multiplicity = plan(block, 1);
                const int width = plan(block, 2);
                for (int channel = 0; channel < multiplicity; ++channel)
                    for (int component = 0; component < width; ++component)
                        destination(node,
                            offset + component * multiplicity + channel)
                            = attributes(model_type,
                                offset + channel * width + component);
            }
        });
}

template class NequIPOAMKokkos<float>;
template class NequIPOAMKokkos<double>;

// The family-adapter conformance gate: this translation unit instantiates the
// evaluator, so a missing or mistyped interface member fails the library build
// here rather than at a consumer.
static_assert(symmetrix::execution::FamilyEvaluator<NequIPOAMKokkos<float>>);
static_assert(symmetrix::execution::FamilyEvaluator<NequIPOAMKokkos<double>>);
static_assert(symmetrix::execution::TieredFamilyEvaluator<NequIPOAMKokkos<float>>);
static_assert(symmetrix::execution::TieredFamilyEvaluator<NequIPOAMKokkos<double>>);
