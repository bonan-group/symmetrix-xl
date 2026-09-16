#pragma once

#include "jit_operator_module_abi.h"

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

    void launch_m0_forward(
        const SymmetrixJitM0ArgsV1& args, void* stream,
        std::int32_t persistent_blocks) const;
    void launch_m0_reverse(
        const SymmetrixJitM0ArgsV1& args, void* stream,
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
