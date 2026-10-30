#pragma once

#include "mace_nonlinear_jit_cuda_plugin_abi.h"
#include "mace_nonlinear_jit_host_plugin_abi.h"

#include <cstdint>
#include <string_view>

namespace symmetrix::execution {

struct MH1SplitPhaseExpectation {
    std::string_view artifact_id;
    std::string_view generation_fingerprint;
    std::string_view semantic_fingerprint;
    std::string_view structure_fingerprint;
    std::string_view runtime_layout_fingerprint;
    std::uint32_t interaction_count = 0;
    std::uint32_t scalar_size = 0;
    std::string_view target_backend;
    std::string_view target_architecture;
};

struct MH1FullModuleDescriptor {
    std::uint32_t abi_version = 0;
    std::string_view abi_tag;
    std::string_view artifact_id;
    std::string_view generation_fingerprint;
    std::string_view semantic_fingerprint;
    std::string_view structure_fingerprint;
    std::string_view runtime_layout_fingerprint;
    std::string_view target_backend;
    std::string_view target_architecture;
    std::uint32_t interaction_count = 0;
    std::uint32_t scalar_size = 0;
};

struct MH1ProductModuleExpectation {
    int layer = -1;
    int product_count = 0;
    std::string_view kind;
    std::string_view backend;
    std::string_view target;
    std::string_view artifact_id;
    std::string_view structure_fingerprint;
    std::uint32_t scalar_size = 0;
    int channels = 0;
    int input_components = 0;
    int output_components = 0;
    int correlation = 0;
    int term_count = 0;
};

struct MH1ProductModuleCandidate {
    int layer = -1;
    std::string_view kind;
    std::string_view backend;
    std::string_view target;
    std::string_view artifact_id;
    std::string_view structure_fingerprint;
    std::uint32_t scalar_size = 0;
    int channels = 0;
    int input_components = 0;
    int output_components = 0;
    int correlation = 0;
    int term_count = 0;
};

void validate_mh1_split_phase_host_descriptor(
    const SymmetrixJitMH1HostPluginV7& descriptor,
    const MH1SplitPhaseExpectation& expectation);

void validate_mh1_split_phase_cuda_descriptor(
    const SymmetrixJitMH1CudaPluginV5& descriptor,
    const MH1SplitPhaseExpectation& expectation);

void validate_mh1_full_module_descriptor(
    const MH1FullModuleDescriptor& descriptor,
    const MH1SplitPhaseExpectation& expectation);

void validate_mh1_product_module_candidate(
    const MH1ProductModuleCandidate& candidate,
    const MH1ProductModuleExpectation& expectation);

void validate_mh1_launch_plan_schema(
    std::string_view schema, std::string_view backend);

}  // namespace symmetrix::execution
