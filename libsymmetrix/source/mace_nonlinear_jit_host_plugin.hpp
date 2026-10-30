#pragma once

#include "mace_nonlinear_jit_host_plugin_abi.h"
#include "mace_nonlinear_split_phase_admission.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace symmetrix::execution {

struct MH1HostInteractionExpectation {
    std::int32_t input_1_dimension = -1;
    std::int32_t input_2_dimension = -1;
    std::int32_t output_dimension = -1;
    std::int32_t weight_size = -1;
    std::int32_t phi_dimension = -1;
    std::int32_t multiplicity = -1;
    std::int32_t input_1_angular_dimension = -1;
    std::int32_t instruction_count = -1;
};

struct MH1HostPluginExpectation {
    std::string_view artifact_id;
    std::string_view generation_fingerprint;
    std::string_view semantic_fingerprint;
    std::string_view structure_fingerprint;
    // The interaction count the model contract declares.  It is contract data,
    // not a compiled-in constant: the loader compares it with the count the
    // generated tables admit and with the count the descriptor announces.
    std::uint32_t interaction_count = 0;
    std::array<MH1HostInteractionExpectation,
        SYMMETRIX_JIT_MH1_HOST_PLUGIN_INTERACTIONS> interactions;
    std::uint32_t required_capabilities = 0;
};

struct MH1HostNodeProgramExpectation {
    std::int32_t element_count = -1;
    std::int32_t input_dimension = -1;
    std::int32_t up_dimension = -1;
    std::int32_t residual_dimension = -1;
    std::int32_t skip_dimension = -1;
    std::int32_t message_dimension = -1;
    std::int32_t interaction_output_dimension = -1;
    std::int32_t output_dimension = -1;
    std::int32_t product_term_count = -1;
    std::int32_t node_arena_dimension = -1;
    bool requires_tp_source_state_adjoint = false;
    std::int64_t linear_parameter_count = -1;
    std::int64_t product_parameter_count = -1;
    std::int64_t readout_parameter_count = -1;
    std::array<std::int32_t,
        SYMMETRIX_JIT_MH1_HOST_NODE_FORWARD_PHASES_V4>
        forward_owners_per_node{};
    std::array<std::int32_t,
        SYMMETRIX_JIT_MH1_HOST_NODE_REVERSE_PHASES_V4>
        reverse_owners_per_node{};
};

struct MH1HostPluginV4Expectation {
    std::string_view artifact_id;
    std::string_view generation_fingerprint;
    std::string_view semantic_fingerprint;
    std::string_view structure_fingerprint;
    std::string_view runtime_layout_fingerprint;
    // The interaction count the model contract declares; see
    // `MH1HostPluginExpectation::interaction_count`.
    std::uint32_t interaction_count = 0;
    std::array<MH1HostInteractionExpectation,
        SYMMETRIX_JIT_MH1_HOST_PLUGIN_INTERACTIONS> interactions;
    std::array<MH1HostNodeProgramExpectation,
        SYMMETRIX_JIT_MH1_HOST_PLUGIN_INTERACTIONS> node_programs;
    std::uint32_t required_capabilities = 0;
};

struct MH1HostSplineRProgramExpectation {
    std::uint32_t flags = 0;
    std::int32_t weight_function_count = -1;
    std::int32_t density_function_index = -1;
    std::int32_t coefficient_count = 4;
};

struct MH1HostPluginV5Expectation {
    MH1HostPluginV4Expectation v4;
    std::array<MH1HostSplineRProgramExpectation,
        SYMMETRIX_JIT_MH1_HOST_PLUGIN_INTERACTIONS> spline_r_programs;
    std::uint32_t scalar_size = 0;
    std::uint32_t required_capabilities = 0;
};

// The ABI-v6 expectation.  ABI v6 carries one pointer per per-interaction
// table instead of an embedded array, so the admitted count is the number of
// entries the expectation declares rather than a compiled-in bound: the loader
// checks the descriptor's announced count, the table sizes, and every entry
// against these vectors, and a topology with any layer count is admitted.
struct MH1HostPluginV6Expectation {
    std::string_view artifact_id;
    std::string_view generation_fingerprint;
    std::string_view semantic_fingerprint;
    std::string_view structure_fingerprint;
    std::string_view runtime_layout_fingerprint;
    std::vector<MH1HostInteractionExpectation> interactions;
    std::vector<MH1HostNodeProgramExpectation> node_programs;
    std::vector<MH1HostSplineRProgramExpectation> spline_r_programs;
    std::uint32_t scalar_size = 0;
    std::uint32_t required_capabilities = 0;
};

class MH1HostPlugin {
public:
    static MH1HostPlugin load(
        std::string path,
        const MH1HostPluginExpectation& expectation);

    MH1HostPlugin() = default;
    // The caller must fence its scheduler before destruction or move-assignment;
    // no owner callback may remain in flight when its library is unloaded.
    ~MH1HostPlugin();
    MH1HostPlugin(const MH1HostPlugin&) = delete;
    MH1HostPlugin& operator=(const MH1HostPlugin&) = delete;
    MH1HostPlugin(MH1HostPlugin&& other) noexcept;
    MH1HostPlugin& operator=(MH1HostPlugin&& other) noexcept;

    explicit operator bool() const noexcept { return handle_ != nullptr; }
    const std::string& path() const noexcept { return path_; }
    const SymmetrixJitMH1HostPluginV3& descriptor() const;

private:
    MH1HostPlugin(
        void* handle,
        const SymmetrixJitMH1HostPluginV3* descriptor,
        std::string path) noexcept;
    void reset() noexcept;

    void* handle_ = nullptr;
    const SymmetrixJitMH1HostPluginV3* descriptor_ = nullptr;
    std::string path_;
};

// Historical ABI-v4 compatibility handle.  The active generated host route
// uses the split-phase MH1HostPluginV7 ABI below.
class MH1HostPluginV4 {
public:
    static MH1HostPluginV4 load(
        std::string path,
        const MH1HostPluginV4Expectation& expectation);

    MH1HostPluginV4() = default;
    ~MH1HostPluginV4();
    MH1HostPluginV4(const MH1HostPluginV4&) = delete;
    MH1HostPluginV4& operator=(const MH1HostPluginV4&) = delete;
    MH1HostPluginV4(MH1HostPluginV4&& other) noexcept;
    MH1HostPluginV4& operator=(MH1HostPluginV4&& other) noexcept;

    explicit operator bool() const noexcept { return handle_ != nullptr; }
    const std::string& path() const noexcept { return path_; }
    const SymmetrixJitMH1HostPluginV4& descriptor() const;

private:
    MH1HostPluginV4(
        void* handle,
        const SymmetrixJitMH1HostPluginV4* descriptor,
        std::string path) noexcept;
    void reset() noexcept;

    void* handle_ = nullptr;
    const SymmetrixJitMH1HostPluginV4* descriptor_ = nullptr;
    std::string path_;
};

class MH1HostPluginV5 {
public:
    static MH1HostPluginV5 load(
        std::string path,
        const MH1HostPluginV5Expectation& expectation);

    MH1HostPluginV5() = default;
    ~MH1HostPluginV5();
    MH1HostPluginV5(const MH1HostPluginV5&) = delete;
    MH1HostPluginV5& operator=(const MH1HostPluginV5&) = delete;
    MH1HostPluginV5(MH1HostPluginV5&& other) noexcept;
    MH1HostPluginV5& operator=(MH1HostPluginV5&& other) noexcept;

    explicit operator bool() const noexcept { return handle_ != nullptr; }
    const std::string& path() const noexcept { return path_; }
    const SymmetrixJitMH1HostPluginV5& descriptor() const;

private:
    MH1HostPluginV5(
        void* handle,
        const SymmetrixJitMH1HostPluginV5* descriptor,
        std::string path) noexcept;
    void reset() noexcept;

    void* handle_ = nullptr;
    const SymmetrixJitMH1HostPluginV5* descriptor_ = nullptr;
    std::string path_;
};

// ABI-v6 handle.  Its descriptor carries pointers to the per-interaction
// tables, so the interaction count is the contract's rather than a compiled-in
// bound.  A consumer reads `node_programs[layer]` exactly as it does for v5.
class MH1HostPluginV6 {
public:
    static MH1HostPluginV6 load(
        std::string path,
        const MH1HostPluginV6Expectation& expectation);

    MH1HostPluginV6() = default;
    ~MH1HostPluginV6();
    MH1HostPluginV6(const MH1HostPluginV6&) = delete;
    MH1HostPluginV6& operator=(const MH1HostPluginV6&) = delete;
    MH1HostPluginV6(MH1HostPluginV6&& other) noexcept;
    MH1HostPluginV6& operator=(MH1HostPluginV6&& other) noexcept;

    explicit operator bool() const noexcept { return handle_ != nullptr; }
    const std::string& path() const noexcept { return path_; }
    const SymmetrixJitMH1HostPluginV6& descriptor() const;

private:
    MH1HostPluginV6(
        void* handle,
        const SymmetrixJitMH1HostPluginV6* descriptor,
        std::string path) noexcept;
    void reset() noexcept;

    void* handle_ = nullptr;
    const SymmetrixJitMH1HostPluginV6* descriptor_ = nullptr;
    std::string path_;
};

class MH1HostPluginV7 {
public:
    static MH1HostPluginV7 load(
        std::string path, const MH1SplitPhaseExpectation& expectation);
    MH1HostPluginV7() = default;
    ~MH1HostPluginV7();
    MH1HostPluginV7(const MH1HostPluginV7&) = delete;
    MH1HostPluginV7& operator=(const MH1HostPluginV7&) = delete;
    MH1HostPluginV7(MH1HostPluginV7&& other) noexcept;
    MH1HostPluginV7& operator=(MH1HostPluginV7&& other) noexcept;
    explicit operator bool() const noexcept { return handle_ != nullptr; }
    const std::string& path() const noexcept { return path_; }
    const SymmetrixJitMH1HostPluginV7& descriptor() const;
private:
    MH1HostPluginV7(
        void* handle, const SymmetrixJitMH1HostPluginV7* descriptor,
        std::string path) noexcept;
    void reset() noexcept;
    void* handle_ = nullptr;
    const SymmetrixJitMH1HostPluginV7* descriptor_ = nullptr;
    std::string path_;
};

}  // namespace symmetrix::execution
