#pragma once

#include "oam_jit_host_plugin_abi.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace symmetrix::execution {

struct OAMHostLayerExpectation {
    std::int32_t source_dimension = -1;
    std::int32_t edge_dimension = -1;
    std::int32_t weight_dimension = -1;
    std::int32_t output_dimension = -1;
    std::int32_t instruction_count = -1;
};

struct OAMHostPluginExpectation {
    std::string_view artifact_id;
    std::string_view generation_fingerprint;
    std::string_view semantic_fingerprint;
    std::string_view structure_fingerprint;
    std::uint32_t scalar_size = 0;
    std::vector<OAMHostLayerExpectation> layers;
    std::uint32_t required_capabilities = 0;
};

class OAMHostPlugin {
public:
    static OAMHostPlugin load(
        std::string path, const OAMHostPluginExpectation& expectation);

    OAMHostPlugin() = default;
    // The caller must fence Kokkos before destruction or move-assignment. No
    // generated callback may remain in flight while the shared object closes.
    ~OAMHostPlugin();
    OAMHostPlugin(const OAMHostPlugin&) = delete;
    OAMHostPlugin& operator=(const OAMHostPlugin&) = delete;
    OAMHostPlugin(OAMHostPlugin&& other) noexcept;
    OAMHostPlugin& operator=(OAMHostPlugin&& other) noexcept;

    explicit operator bool() const noexcept { return handle_ != nullptr; }
    const std::string& path() const noexcept { return path_; }
    const SymmetrixJitOAMHostPluginV1& descriptor() const;

private:
    OAMHostPlugin(
        void* handle, const SymmetrixJitOAMHostPluginV1* descriptor,
        std::string path) noexcept;
    void reset() noexcept;

    void* handle_ = nullptr;
    const SymmetrixJitOAMHostPluginV1* descriptor_ = nullptr;
    std::string path_;
};

struct OAMHostLayerExpectationV2 {
    std::int32_t source_dimension = -1;
    std::int32_t edge_dimension = -1;
    std::int32_t weight_dimension = -1;
    std::int32_t output_dimension = -1;
    std::int32_t instruction_count = -1;
    std::int32_t channel_tile = -1;
    std::int32_t target_dimension = -1;
};

struct OAMHostPluginExpectationV2 {
    std::string_view artifact_id;
    std::string_view generation_fingerprint;
    std::string_view semantic_fingerprint;
    std::string_view structure_fingerprint;
    std::uint32_t scalar_size = 0;
    std::vector<OAMHostLayerExpectationV2> layers;
    std::uint32_t required_capabilities = 0;
    std::string_view schedule_identity;
    std::string_view host_target;
    std::uint32_t scalar_width = 0;
    std::string_view required_isa;
    std::string_view full_reverse_reduction_policy;
    std::string_view schedule_qualification;
    std::string_view forward_owner_schedule;
    std::string_view full_reverse_owner_schedule;
    std::string_view tiling_policy;
    std::string_view vector_policy;
    std::string_view register_budget_policy;
};

// Move-only owner of a loaded ABI v7 shared object. No fence is performed or
// required by the handle itself. `load` returns an empty optional when the
// file is not an ABI v7 artifact (it does not export the v7 query symbol) and
// throws `plugin_error` for every other failure, so a rejected descriptor is
// never silently mistaken for a missing contract.
class OAMHostPluginV2Handle {
public:
    static std::optional<OAMHostPluginV2Handle> load(
        std::string path, const OAMHostPluginExpectationV2& expectation);

    OAMHostPluginV2Handle() = default;
    ~OAMHostPluginV2Handle();
    OAMHostPluginV2Handle(const OAMHostPluginV2Handle&) = delete;
    OAMHostPluginV2Handle& operator=(const OAMHostPluginV2Handle&) = delete;
    OAMHostPluginV2Handle(OAMHostPluginV2Handle&& other) noexcept;
    OAMHostPluginV2Handle& operator=(OAMHostPluginV2Handle&& other) noexcept;

    explicit operator bool() const noexcept { return handle_ != nullptr; }
    const std::string& path() const noexcept { return path_; }
    const std::string& directory() const noexcept { return directory_; }
    const SymmetrixJitOAMHostPluginV7& descriptor() const;

private:
    OAMHostPluginV2Handle(
        void* handle, const SymmetrixJitOAMHostPluginV7* descriptor,
        std::string path, std::string directory) noexcept;
    void reset() noexcept;

    void* handle_ = nullptr;
    const SymmetrixJitOAMHostPluginV7* descriptor_ = nullptr;
    std::string path_;
    std::string directory_;
};

}  // namespace symmetrix::execution
