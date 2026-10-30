#pragma once

namespace symmetrix::execution {

constexpr bool m1_standard_forward_available(
    bool standard_ready, bool host_execution, int host_channel_tile,
    bool metal_forward_available)
{
    // The host runtime-tile rollback bypasses standard kernels, but not an
    // admitted Metal forward, which has higher dispatch priority.
    return standard_ready
        && (metal_forward_available || !host_execution || host_channel_tile != 0);
}

constexpr const char* m1_dispatched_implementation_id(
    bool explicit_artifact_selected, bool recompute,
    bool admits_device_terms, bool standard_forward_available)
{
    if (explicit_artifact_selected)
        return "generated_m1";
    if (!recompute)
        return "generic_m1";
    // This ID names the forward specialization. Reverse independently uses
    // the standard module when ready, otherwise generic recomputation.
    if (admits_device_terms)
        return "device_m1_terms";
    if (standard_forward_available)
        return "builtin_m1";
    return "generic_m1";
}

}  // namespace symmetrix::execution
