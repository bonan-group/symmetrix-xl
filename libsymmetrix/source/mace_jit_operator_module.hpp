#pragma once

#include "mace_jit_operator_module_abi.h"
#include "mace_jit_r0_contract.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace symmetrix::execution {

struct OperatorModuleExpectation {
    std::uint32_t kind = 0;
    std::string_view artifact_id;
    std::string_view structure_fingerprint;
    std::string_view target;
    std::uint32_t scalar_kind = 0;
    std::uint32_t scalar_size = 0;
    std::uint32_t required_capabilities = 0;
    std::int32_t input_components = 0;
    std::int32_t output_components = 0;
    std::int32_t correlation = 0;
    std::int32_t term_count = 0;
    std::int32_t max_threads_per_block = 0;
    std::int32_t expected_threads_per_block = 0;
    std::int32_t schedule_revision = 0;
    std::int32_t persistent_blocks_per_compute_unit = 0;
    std::int32_t r0_channels = 0;
    std::int32_t r0_l_max = 0;
    std::int32_t r0_density_extent_multiplier = 0;
    std::int32_t r0_density_owners_per_block = 0;
    std::int32_t r0_density_threads_per_block = 0;
    std::int32_t r0_density_persistent = 0;
    std::int32_t r0_forward_extent_multiplier = 0;
    std::int32_t r0_forward_owners_per_block = 0;
    std::int32_t r0_forward_threads_per_block = 0;
    std::int32_t r0_forward_persistent = 0;
    std::int32_t r0_reverse_prepare_extent_multiplier = 0;
    std::int32_t r0_reverse_prepare_owners_per_block = 0;
    std::int32_t r0_reverse_prepare_threads_per_block = 0;
    std::int32_t r0_reverse_prepare_persistent = 0;
    std::int32_t r0_coordinate_reverse_extent_multiplier = 0;
    std::int32_t r0_coordinate_reverse_owners_per_block = 0;
    std::int32_t r0_coordinate_reverse_threads_per_block = 0;
    std::int32_t r0_coordinate_reverse_persistent = 0;
    R0LogicalDescriptor r0_descriptor;
    std::int32_t channel_tile = 0;
    std::int32_t forward_node_tile = 0;
    std::int32_t forward_component_tile = 0;
    std::int32_t reverse_node_tile = 0;
    std::int32_t reverse_component_tile = 0;
    std::int32_t logical_tiles_per_block = 0;
};

struct OperatorModuleState;

class OperatorModule {
public:
    static OperatorModule load(
        std::string path, const OperatorModuleExpectation& expectation);

    OperatorModule();
    ~OperatorModule();
    OperatorModule(const OperatorModule&) = delete;
    OperatorModule& operator=(const OperatorModule&) = delete;
    OperatorModule(OperatorModule&&) noexcept;
    OperatorModule& operator=(OperatorModule&&) noexcept;

    explicit operator bool() const noexcept { return state_ != nullptr; }
    const std::string& path() const noexcept { return path_; }
    std::string_view artifact_id() const noexcept;
    std::string_view structure_fingerprint() const noexcept;
    std::string_view target() const noexcept;
    std::uint32_t kind() const noexcept;
    std::uint32_t capabilities() const noexcept;
    std::int32_t threads_per_block() const noexcept;

    void validate_m0_launch(
        const SymmetrixJitM0ArgsV1& args, bool reverse,
        std::int32_t persistent_blocks) const;
    void validate_m1_launch(
        const SymmetrixJitM1ArgsV1& args, bool reverse,
        std::int32_t persistent_blocks) const;

    void launch_m0_forward(
        const SymmetrixJitM0ArgsV1& args, void* stream,
        std::int32_t persistent_blocks) const;
    void launch_m0_reverse(
        const SymmetrixJitM0ArgsV1& args, void* stream,
        std::int32_t persistent_blocks) const;
    void launch_m0_forward_strided(
        const SymmetrixJitM0ArgsV2& args, void* stream,
        std::int32_t persistent_blocks) const;
    void launch_m0_reverse_strided(
        const SymmetrixJitM0ArgsV2& args, void* stream,
        std::int32_t persistent_blocks) const;
    void launch_m1_forward(
        const SymmetrixJitM1ArgsV1& args, void* stream,
        std::int32_t persistent_blocks) const;
    void launch_m1_reverse(
        const SymmetrixJitM1ArgsV1& args, void* stream,
        std::int32_t persistent_blocks) const;
    void launch_m1_forward_strided(
        const SymmetrixJitM1ArgsV2& args, void* stream,
        std::int32_t persistent_blocks) const;
    void launch_m1_reverse_strided(
        const SymmetrixJitM1ArgsV2& args, void* stream,
        std::int32_t persistent_blocks) const;
    void launch_r0_density_prepare(
        const SymmetrixJitR0ArgsV1& args, void* stream,
        std::int32_t persistent_blocks) const;
    void launch_r0_forward(
        const SymmetrixJitR0ArgsV1& args, void* stream,
        std::int32_t persistent_blocks) const;
    void launch_r0_reverse_prepare(
        const SymmetrixJitR0ArgsV1& args, void* stream,
        std::int32_t persistent_blocks) const;
    void launch_r0_coordinate_reverse(
        const SymmetrixJitR0ArgsV1& args, void* stream,
        std::int32_t persistent_blocks) const;

private:
    explicit OperatorModule(
        std::unique_ptr<OperatorModuleState> state, std::string path) noexcept;
    void reset() noexcept;

    std::unique_ptr<OperatorModuleState> state_;
    std::string path_;
};

} // namespace symmetrix::execution
