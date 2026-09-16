#pragma once

#include <pybind11/pybind11.h>

#include "device_backend.hpp"

inline pybind11::dict execution_device_environment_dict()
{
    const auto environment = execution_device_execution_environment();
    pybind11::dict result;
    result["available"] = environment.available;
    result["backend"] = environment.backend;
    result["execution_space"] = environment.execution_space;
    result["memory_space"] = environment.memory_space;
    result["device_memory"] = environment.device_memory;
    result["device_ordinal"] = environment.device_ordinal;
    result["device_name"] = environment.device_name;
    result["raw_agent_target"] = environment.raw_agent_target;
    result["architecture"] = environment.architecture;
    result["target_features"] = environment.target_features;
    result["compiler_offload_target"] = environment.compiler_offload_target;
    result["native_subgroup_width"] = environment.native_subgroup_width;
    result["warp_width"] = environment.native_subgroup_width;
    result["compute_unit_count"] = environment.compute_unit_count;
    result["multiprocessor_count"] = environment.compute_unit_count;
    result["max_team_size"] = environment.max_team_size;
    result["max_shared_memory_per_block"] =
        environment.max_shared_memory_per_block;
    result["compute_capability_code"] = environment.compute_capability;
    result["runtime_version"] = environment.runtime_version;
    result["driver_version"] = environment.driver_version;
    result["compiler_version"] = environment.compiler_version;
    result["aot_available"] = environment.aot_available;
    result["jit_available"] = environment.jit_available;
    result["batched_blas_available"] = environment.batched_blas_available;
    return result;
}
