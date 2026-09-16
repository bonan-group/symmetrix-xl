"""Standalone generated kernels for the MACE-MH-1 Execution operator."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .execution_mh1_contract import (
    normalize_execution_mh1_contract,
    normalize_execution_mh1_v4_contract,
    project_execution_mh1_v3_contract,
)
from .jit import JIT_GENERATION_VERSION
from .jit_codegen import (
    CUDA_DIALECT as STANDARD_CUDA_DIALECT,
    HIP_DIALECT as STANDARD_HIP_DIALECT,
    _render_direct_harmonic_gradient_helper,
    _render_forward_owner as _render_standard_r1_forward_owner,
    _render_gpu_spline_helpers,
    _render_r1_fused_reverse_kernel as _render_standard_r1_fused_reverse_kernel,
)

MH1_HOST_PLUGIN_ABI = "symmetrix.jit.mh1.host-plugin/3"
MH1_HOST_PLUGIN_ABI_VERSION = 3
MH1_CUDA_PLUGIN_ABI = "symmetrix.jit.mh1.cuda-plugin/3"
MH1_CUDA_PLUGIN_ABI_VERSION = 3
MH1_HOST_PLUGIN_V4_ABI = "symmetrix.jit.mh1.host-plugin/4"
MH1_HOST_PLUGIN_V4_ABI_VERSION = 4
MH1_HOST_PLUGIN_V5_ABI = "symmetrix.jit.mh1.host-plugin/5"
MH1_HOST_PLUGIN_V5_ABI_VERSION = 5
MH1_CUDA_PLUGIN_V4_ABI = "symmetrix.jit.mh1.cuda-plugin/4"
MH1_CUDA_PLUGIN_V4_ABI_VERSION = 4
MH1_NODE_PROGRAM_METADATA_TAG = "symmetrix.execution.mh1.node-metadata/1"
MH1_FORWARD_POLICY_TAG = "symmetrix.jit.mh1.cuda-forward-policy/1"
MH1_SOURCE_POLICY_TAG = "symmetrix.jit.mh1.cuda-source-policy/1"
MH1_EDGE_POLICY_TAG = "symmetrix.jit.mh1.cuda-edge-policy/2"
MH1_CUDA_MODULE_ABI = "symmetrix.jit.mh1.cuda-module/2"
MH1_CUDA_MODULE_ABI_VERSION = 2
MH1_HIP_MODULE_ABI = "symmetrix.jit.mh1.hip-module/2"
MH1_HIP_MODULE_ABI_VERSION = 2
MH1_LAUNCH_PLAN_TAG = "symmetrix.jit.mh1.cuda-launch-plan/3"
MH1_HIP_LAUNCH_PLAN_TAG = "symmetrix.jit.mh1.hip-launch-plan/3"
MH1_CONDITIONER_LAYOUT_TAG = "symmetrix.execution.mh1.conditioner-layout/1"
_CUDA_FORWARD_COMPONENT_BUDGET = 8
_CUDA_FORWARD_SHARED_COMPONENT_BUDGET = 16
_CUDA_FORWARD_FUSE_COMPONENT_LIMIT = 16
_CUDA_SOURCE_COMPONENT_BUDGET = 8
_MH1_CUDA_EDGE_REVERSE_SCHEDULE = "hybrid-path-tiled-v7"
_MH1_HIP_EDGE_REVERSE_SCHEDULE = "hybrid-path-tiled-v6"
_MH1_PATH_TILED_EDGE_BLOCK_MULTIPLIER = 4
_MH1_PATH_TILED_EDGE_EDGE_BATCH = 4
_MH1_PATH_TILED_EDGE_PATH_BATCH = 4
_MH1_HOST_SPLINE_REVERSE_EDGE_TILE = 128
_MH1_HOST_SPLINE_REVERSE_PREFETCH_DISTANCE = 2
_MH1_NODE_FORWARD_SCHEDULE = "tiled-8x32-message-linear2-32x32-v7"
_MH1_NODE_REVERSE_SCHEDULE = "tiled-8x32-message-linear2-32x32-v7"
_MH1_CUDA_RETAINED_NODE_REVERSE_SCHEDULE = "monolithic-l1-retained-v1"
_MH1_CUDA_GROUPED_MESSAGE_REVERSE_SCHEDULE = "grouped-message-l1-retained-v1"
_MH1_CUDA_RECOMPUTE_GROUPED_MESSAGE_REVERSE_SCHEDULE = "grouped-message-l1-recompute-v1"
_MH1_CUDA_RECOMPUTE_GROUPED_LINEAR2_SCHEDULE = "grouped-message-linear2-recompute-v1"
_MH1_CUDA_RECOMPUTE_GROUPED_ALL_MESSAGE_SCHEDULE = (
    "grouped-all-message-reverse-linear2-recompute-v1"
)
_MH1_CUDA_RECOMPUTE_GROUPED_FORWARD_MESSAGE_SCHEDULE = (
    "grouped-all-message-linear2-recompute-v1"
)
_MH1_CUDA_RECOMPUTE_GROUPED_RESIDUAL_SCHEDULE = (
    "grouped-residual-all-message-linear2-recompute-v1"
)
_MH1_CUDA_RECOMPUTE_TILED_PRODUCT_REVERSE_SCHEDULE = (
    "tiled-product-reverse-grouped-residual-recompute-v1"
)
_MH1_CUDA_GROUPED_LINEAR2_SCHEDULE = "grouped-message-linear2-retained-v1"
_MH1_CUDA_GROUPED_ALL_MESSAGE_SCHEDULE = "grouped-message-all-linear2-retained-v1"
_MH1_CUDA_GROUPED_FORWARD_MESSAGE_SCHEDULE = "grouped-all-message-linear2-retained-v1"
_MH1_CUDA_GROUPED_RESIDUAL_SCHEDULE = "grouped-residual-all-message-linear2-retained-v1"
_MH1_CUDA_GROUPED_FORWARD_RESIDUAL_SCHEDULE = (
    "grouped-forward-residual-all-message-linear2-retained-v1"
)
_MH1_CUDA_TILED_PRODUCT_REVERSE_SCHEDULE = (
    "tiled-product-reverse-grouped-forward-residual-v1"
)
_MH1_CUDA_TILED_PRODUCT_FORWARD_REVERSE_SCHEDULE = (
    "tiled-product-forward-reverse-grouped-forward-residual-v1"
)
_MH1_CUDA_TILED_LINEAR_UP_REVERSE_SCHEDULE = (
    "tiled-linear-up-reverse-product-forward-reverse-grouped-forward-residual-v1"
)
_MH1_CUDA_TILED_LINEAR_UP_FORWARD_REVERSE_SCHEDULE = (
    "tiled-linear-up-forward-reverse-product-forward-reverse-grouped-forward-"
    "residual-v1"
)
_MH1_CUDA_GROUPED_READOUT_REVERSE_SCHEDULE = (
    "grouped-readout-reverse-linear-up-forward-reverse-product-forward-reverse-"
    "grouped-forward-residual-v1"
)
_MH1_CUDA_GROUPED_SKIP_FORWARD_SCHEDULE = (
    "grouped-skip-forward-readout-reverse-linear-up-forward-reverse-product-"
    "forward-reverse-grouped-forward-residual-v1"
)
_MH1_CUDA_GROUPED_ALL_SKIP_FORWARD_SCHEDULE = (
    "grouped-all-skip-forward-readout-reverse-linear-up-forward-reverse-product-"
    "forward-reverse-grouped-forward-residual-v1"
)
_MH1_CUDA_WIDE_PRODUCT_LINEAR_SCHEDULE = (
    "wide-product-linear-grouped-all-skip-forward-readout-reverse-linear-up-"
    "forward-reverse-product-forward-reverse-grouped-forward-residual-v1"
)
_MH1_CUDA_WIDE_L1_REVERSE_MESSAGE_SCHEDULE = (
    "wide-l1-reverse-message-wide-product-linear-grouped-all-skip-forward-"
    "readout-reverse-linear-up-forward-reverse-product-forward-reverse-grouped-"
    "forward-residual-v1"
)
_MH1_CUDA_WIDE_L1_REVERSE_LINEAR2_SCHEDULE = (
    "wide-l1-reverse-linear2-wide-l1-reverse-message-wide-product-linear-"
    "grouped-all-skip-forward-readout-reverse-linear-up-forward-reverse-product-"
    "forward-reverse-grouped-forward-residual-v1"
)
_MH1_CUDA_PAIRED_L0_PRODUCT_REVERSE_SCHEDULE = (
    "paired-l0-product-reverse-wide-l1-reverse-linear2-wide-l1-reverse-message-"
    "wide-product-linear-grouped-all-skip-forward-readout-reverse-linear-up-"
    "forward-reverse-product-forward-reverse-grouped-forward-residual-v1"
)
_MH1_CUDA_WIDE_ALL_REVERSE_LINEAR2_SCHEDULE = (
    "wide-all-reverse-linear2-paired-l0-product-reverse-wide-l1-reverse-message-"
    "wide-product-linear-grouped-all-skip-forward-readout-reverse-linear-up-"
    "forward-reverse-product-forward-reverse-grouped-forward-residual-v1"
)
_MH1_CUDA_WIDE_L0_FORWARD_LINEAR2_SCHEDULE = (
    "wide-l0-forward-linear2-wide-all-reverse-linear2-paired-l0-product-reverse-"
    "wide-l1-reverse-message-wide-product-linear-grouped-all-skip-forward-"
    "readout-reverse-linear-up-forward-reverse-product-forward-reverse-grouped-"
    "forward-residual-v1"
)
_MH1_CUDA_WIDE_ALL_FORWARD_LINEAR2_SCHEDULE = (
    "wide-all-forward-linear2-wide-all-reverse-linear2-paired-l0-product-reverse-"
    "wide-l1-reverse-message-wide-product-linear-grouped-all-skip-forward-"
    "readout-reverse-linear-up-forward-reverse-product-forward-reverse-grouped-"
    "forward-residual-v1"
)
_MH1_CUDA_PAIRED_ALL_PRODUCT_REVERSE_SCHEDULE = (
    "paired-all-product-reverse-wide-all-forward-linear2-wide-all-reverse-"
    "linear2-wide-l1-reverse-message-wide-product-linear-grouped-all-skip-"
    "forward-readout-reverse-linear-up-forward-reverse-product-forward-reverse-"
    "grouped-forward-residual-v1"
)
_MH1_CUDA_XWIDE_L0_REVERSE_LINEAR2_SCHEDULE = (
    "xwide-l0-reverse-linear2-paired-all-product-reverse-wide-all-forward-"
    "linear2-wide-all-reverse-linear2-wide-l1-reverse-message-wide-product-"
    "linear-grouped-all-skip-forward-readout-reverse-linear-up-forward-reverse-"
    "product-forward-reverse-grouped-forward-residual-v1"
)
_MH1_CUDA_XWIDE_ALL_REVERSE_LINEAR2_SCHEDULE = (
    "xwide-all-reverse-linear2-paired-all-product-reverse-wide-all-forward-"
    "linear2-wide-l1-reverse-message-wide-product-linear-grouped-all-skip-"
    "forward-readout-reverse-linear-up-forward-reverse-product-forward-reverse-"
    "grouped-forward-residual-v1"
)
_MH1_CUDA_XWIDE_L0_FORWARD_LINEAR2_SCHEDULE = (
    "xwide-l0-forward-linear2-xwide-all-reverse-linear2-paired-all-product-"
    "reverse-wide-l1-forward-linear2-wide-l1-reverse-message-wide-product-"
    "linear-grouped-all-skip-forward-readout-reverse-linear-up-forward-reverse-"
    "product-forward-reverse-grouped-forward-residual-v1"
)
_MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE = (
    "xwide-all-forward-linear2-xwide-all-reverse-linear2-paired-all-product-"
    "reverse-wide-l1-reverse-message-wide-product-linear-grouped-all-skip-"
    "forward-readout-reverse-linear-up-forward-reverse-product-forward-reverse-"
    "grouped-forward-residual-v1"
)
_MH1_SPLINE_R_SCHEDULE = "shared-r1-receiver-source-fused-v1"
MH1_NODE_STATE_FULL_RETENTION = "full-retention-v1"
MH1_NODE_STATE_RECOMPUTE = "recompute-v1"
MH1_NODE_STATE_REUSE_ADJOINTS = "reuse-adjoints-v1"
MH1_NODE_STATE_RETAIN_INTERACTION = "retain-interaction-v1"
_MH1_CONDITIONER_REVERSE_SCHEDULE = "bounded-recompute-split-v2"
_MH1_CONDITIONER_GENERIC_REVERSE_SCHEDULE = "generic-full-state-v1"
_MH1_CONDITIONER_REVERSE_SCHEDULE_TAG = (
    "symmetrix.jit.mh1.conditioner-reverse-schedule/1"
)
_MH1_SHARED_POLICY_PROFILE = "shared-v13-partitioning-v1"

# Compatibility names retain the CUDA-specific API aliases. Their serialized
# identities follow the backend-neutral constants, including launch-plan bumps.
MH1_CUDA_FORWARD_POLICY_TAG = MH1_FORWARD_POLICY_TAG
MH1_CUDA_SOURCE_POLICY_TAG = MH1_SOURCE_POLICY_TAG
MH1_CUDA_EDGE_POLICY_TAG = MH1_EDGE_POLICY_TAG
MH1_CUDA_LAUNCH_PLAN_TAG = MH1_LAUNCH_PLAN_TAG


def _resolve_node_state_policy(value: str) -> str:
    if value not in (
        MH1_NODE_STATE_FULL_RETENTION,
        MH1_NODE_STATE_RECOMPUTE,
        MH1_NODE_STATE_REUSE_ADJOINTS,
        MH1_NODE_STATE_RETAIN_INTERACTION,
    ):
        raise ValueError(
            "MH1 node-state policy must be 'full-retention-v1', "
            "'recompute-v1', 'reuse-adjoints-v1', or 'retain-interaction-v1'"
        )
    return value


def _cuda_module_node_reverse_schedule(node_state_policy: str) -> str:
    node_state_policy = _resolve_node_state_policy(node_state_policy)
    if node_state_policy in (
        MH1_NODE_STATE_FULL_RETENTION,
        MH1_NODE_STATE_REUSE_ADJOINTS,
        MH1_NODE_STATE_RETAIN_INTERACTION,
    ):
        return _MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE
    return _MH1_CUDA_RECOMPUTE_TILED_PRODUCT_REVERSE_SCHEDULE


def _hip_module_node_reverse_schedule(node_state_policy: str) -> str:
    node_state_policy = _resolve_node_state_policy(node_state_policy)
    if node_state_policy in (
        MH1_NODE_STATE_FULL_RETENTION,
        MH1_NODE_STATE_REUSE_ADJOINTS,
    ):
        return _MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE
    return _MH1_NODE_REVERSE_SCHEDULE


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class MH1Program:
    schema_version: int
    contract_json: str
    abi_version: Literal[3, 4]
    phases: tuple[str, ...] = (
        "interaction_forward",
        "conditioning_forward",
        "node_pre_forward",
        "node_post_forward",
        "node_post_reverse",
        "node_pre_reverse",
        "conditioning_reverse",
        "interaction_reverse",
    )

    @classmethod
    def from_contract(cls, contract: dict[str, Any], *, abi_version: int) -> MH1Program:
        if abi_version == 3:
            normalized = normalize_execution_mh1_contract(contract)
        elif abi_version == 4:
            normalized = normalize_execution_mh1_v4_contract(contract)
        else:
            raise ValueError("MH1 program ABI version must be 3 or 4")
        return cls(1, _canonical_json(normalized), abi_version)

    @property
    def contract(self) -> dict[str, Any]:
        return json.loads(self.contract_json)

    def canonical_json(self) -> str:
        return _canonical_json(asdict(self))


@dataclass(frozen=True)
class MH1GpuTarget:
    schema_version: int
    backend: Literal["cuda", "hip"]
    architecture: str
    native_subgroup_width: int
    logical_edge_width: int
    channel_tile_width: int
    target_features: tuple[str, ...] = ()
    compiler_offload_target: str | None = None
    compute_unit_count: int | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported MH1 GPU target schema")
        if self.backend == "cuda" and not re.fullmatch(
            r"sm_[1-9][0-9]{0,2}", self.architecture
        ):
            raise ValueError("MH1 CUDA architecture must look like 'sm_NN'")
        if self.backend == "hip" and not re.fullmatch(
            r"gfx[0-9a-f]+", self.architecture
        ):
            raise ValueError("MH1 HIP architecture must look like 'gfxNNN'")
        if self.logical_edge_width != 16:
            raise ValueError("MH1 logical edge width must remain 16")
        if self.native_subgroup_width % self.logical_edge_width:
            raise ValueError("MH1 logical edge width must divide the native subgroup")
        if self.channel_tile_width != 32:
            raise ValueError("MH1 channel tile width must remain 32")
        if tuple(sorted(set(self.target_features))) != self.target_features:
            raise ValueError("MH1 target features must be sorted and unique")
        compiler_target = self.compiler_offload_target or self.architecture
        if self.backend == "cuda" and compiler_target != self.architecture:
            raise ValueError("MH1 CUDA compiler target must match its architecture")
        if (
            self.backend == "hip"
            and compiler_target.partition(":")[0] != self.architecture
        ):
            raise ValueError("MH1 HIP compiler target must match its architecture")
        if self.compute_unit_count is not None and self.compute_unit_count <= 0:
            raise ValueError("MH1 compute-unit count must be positive")

    def canonical_json(self) -> str:
        return _canonical_json(asdict(self))

    @property
    def target_id(self) -> str:
        payload = self.canonical_json().encode("ascii")
        return "sha256:" + hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class MH1GpuDialect:
    backend: Literal["cuda", "hip"]
    shuffle_down_expression: str

    def device_prelude(self) -> str:
        expression = self.shuffle_down_expression.format(
            value="(value)", offset="(offset)", width="(width)"
        )
        return (
            "#define SYMMETRIX_JIT_MH1_SHUFFLE_DOWN(value, offset, width) "
            f"{expression}"
        )


MH1_CUDA_DIALECT = MH1GpuDialect(
    "cuda", "__shfl_down_sync(0xffffffffu, {value}, {offset}, {width})"
)
MH1_HIP_DIALECT = MH1GpuDialect("hip", "__shfl_down({value}, {offset}, {width})")


@dataclass(frozen=True)
class MH1KernelSchedule:
    schema_version: int
    forward_policy_json: str
    source_policy_json: str
    edge_policy_json: str
    forward_threads: int
    source_threads: int
    edge_threads: int
    node_threads: int = 128
    tile_nodes: int = 8
    tile_channels: int = 32
    tile_threads: int = 256
    node_state_policy: str = MH1_NODE_STATE_FULL_RETENTION

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported MH1 CUDA schedule schema")
        thread_counts = (
            self.forward_threads,
            self.source_threads,
            self.edge_threads,
            self.node_threads,
            self.tile_threads,
        )
        if any(value <= 0 or value % 32 for value in thread_counts):
            raise ValueError("MH1 CUDA thread counts must be positive warp multiples")
        if (self.tile_nodes, self.tile_channels, self.tile_threads) != (8, 32, 256):
            raise ValueError("MH1 node tile must remain 8x32 with 256 threads")
        _resolve_node_state_policy(self.node_state_policy)
        for encoded in (
            self.forward_policy_json,
            self.source_policy_json,
            self.edge_policy_json,
        ):
            if encoded != _canonical_json(json.loads(encoded)):
                raise ValueError("MH1 CUDA policies must use canonical serialization")

    def canonical_json(self) -> str:
        return _canonical_json(asdict(self))

    @property
    def schedule_id(self) -> str:
        return (
            "sha256:"
            + hashlib.sha256(self.canonical_json().encode("ascii")).hexdigest()
        )


MH1CudaSchedule = MH1KernelSchedule


@dataclass(frozen=True)
class MH1KernelLaunch:
    kernel: str
    grid: Literal["persistent_nodes", "node_channel_tiles"]
    threads: int
    channels: int | None = None
    components: int | None = None
    tile_nodes: int | None = None
    tile_channels: int | None = None

    def as_plan_entry(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "kernel": self.kernel,
            "grid": self.grid,
            "threads": self.threads,
        }
        if self.channels is not None:
            entry["channels"] = self.channels
        if self.components is not None:
            entry["components"] = self.components
        if self.tile_nodes is not None:
            entry["tile_nodes"] = self.tile_nodes
        if self.tile_channels is not None:
            entry["tile_channels"] = self.tile_channels
        return entry


@dataclass(frozen=True)
class MH1NodeProgram:
    device_source: str
    plugin_source: str
    post_forward_launches: tuple[MH1KernelLaunch, ...]
    post_reverse_launches: tuple[MH1KernelLaunch, ...]
    pre_reverse_launches: tuple[MH1KernelLaunch, ...] = ()
    pre_forward_launches: tuple[MH1KernelLaunch, ...] = ()


MH1CudaKernelLaunch = MH1KernelLaunch
MH1NodeCudaProgram = MH1NodeProgram


_HOST_PLUGIN_ABI_HEADER = (
    Path(__file__).with_name("jit_mh1_host_plugin_abi.h").read_text(encoding="ascii")
)
_CUDA_PLUGIN_ABI_HEADER = (
    Path(__file__).with_name("jit_mh1_cuda_plugin_abi.h").read_text(encoding="ascii")
)


def _convert_gpu_module_precision(source: str, precision: str) -> str:
    if precision == "float32":
        return source
    if precision != "float64":
        raise ValueError("MH1 GPU module precision must be 'float32' or 'float64'")
    converted = (
        source.replace("static_cast<float>", "static_cast<double>")
        .replace("const float*", "const double*")
        .replace("float*", "double*")
        .replace("float&", "double&")
        .replace("float ", "double ")
        .replace("sizeof(float)", "sizeof(double)")
        .replace("expf(", "exp(")
        .replace("sqrtf(", "sqrt(")
        .replace("fmaf(", "fma(")
    )
    # energy_scale is passed by value in the stable v4 packet. Keeping that
    # field float preserves packet layout; generated arithmetic promotes it.
    return converted.replace("double energy_scale;", "float energy_scale;")


def _gpu_scalar_size(precision: str) -> int:
    if precision == "float32":
        return 4
    if precision == "float64":
        return 8
    raise ValueError("MH1 GPU module precision must be 'float32' or 'float64'")


def _render_mh1_cuda_device_abi(precision: str = "float32") -> str:
    """Render only the fixed-width packets consumed by device kernels."""

    source = r"""typedef signed int int32_t;
typedef unsigned int uint32_t;
typedef signed long long int64_t;
namespace std {
using ::int32_t;
using ::uint32_t;
using ::int64_t;
using size_t = decltype(sizeof(0));
}

enum SymmetrixJitMH1CudaArgumentFlagV3 {
    SYMMETRIX_JIT_MH1_CUDA_APPLY_EDGE_CUTOFF_V3 = 1u << 0,
    SYMMETRIX_JIT_MH1_CUDA_HAS_LINEAR_BIAS_V3 = 1u << 1,
    SYMMETRIX_JIT_MH1_CUDA_HAS_FIXED_CONTRIBUTION_V3 = 1u << 2,
};

enum SymmetrixJitMH1CudaNodePhaseV4 {
    SYMMETRIX_JIT_MH1_CUDA_NODE_PRE_FORWARD_V4 = 0u,
    SYMMETRIX_JIT_MH1_CUDA_NODE_POST_FORWARD_V4 = 1u,
    SYMMETRIX_JIT_MH1_CUDA_NODE_POST_REVERSE_V4 = 2u,
    SYMMETRIX_JIT_MH1_CUDA_NODE_PRE_REVERSE_V4 = 3u,
};

typedef struct SymmetrixJitMH1CudaForwardArgsV3 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t flags;
    uint32_t active_receiver_count;
    int64_t num_nodes;
    int64_t num_edges;
    int64_t first_edge;
    int64_t samples;
    const int32_t* source_indices;
    const int32_t* active_receivers;
    const int32_t* receiver_offsets;
    const float* edge_phi;
    const float* linear_weight;
    const float* linear_bias;
    const float* edge_linear_contribution;
    const float* edge_input_2;
    const float* edge_cutoff_scale;
    const float* source_node_values;
    const float* output_mask;
    float* node_messages;
} SymmetrixJitMH1CudaForwardArgsV3;

typedef struct SymmetrixJitMH1CudaReverseArgsV3 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t flags;
    uint32_t reserved;
    int64_t num_nodes;
    int64_t num_edges;
    int64_t first_edge;
    int64_t samples;
    int64_t source_owner_count;
    const int32_t* source_indices;
    const int32_t* target_indices;
    const int32_t* source_edge_offsets;
    const int32_t* source_edge_indices;
    const float* edge_phi;
    const float* linear_weight;
    const float* linear_bias;
    const float* edge_linear_contribution;
    const float* edge_input_2;
    const float* edge_cutoff_scale;
    const float* source_node_values;
    const float* output_mask;
    const float* target_node_output_adjoint;
    float* source_node_input_adjoint;
    float* edge_phi_adjoint;
    float* edge_input_2_adjoint;
    float* edge_cutoff_scale_adjoint;
} SymmetrixJitMH1CudaReverseArgsV3;

typedef struct SymmetrixJitMH1CudaConditioningForwardArgsV3 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t flags;
    uint32_t active_receiver_count;
    int64_t num_nodes;
    int64_t num_edges;
    const int32_t* source_types;
    const int32_t* target_indices;
    const int32_t* node_types;
    const int32_t* active_receivers;
    const int32_t* receiver_offsets;
    const float* radial;
    const float* edge_cutoff_scale;
    const float* convolution_source_contributions;
    const float* convolution_target_contributions;
    const float* density_source_contributions;
    const float* density_target_contributions;
    const float* convolution_parameters;
    const float* density_parameters;
    float* edge_phi;
    float* node_density;
} SymmetrixJitMH1CudaConditioningForwardArgsV3;

typedef struct SymmetrixJitMH1CudaConditioningReverseArgsV3 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t flags;
    uint32_t reserved;
    int64_t num_nodes;
    int64_t num_edges;
    const int32_t* source_types;
    const int32_t* target_indices;
    const int32_t* node_types;
    const float* radial;
    const float* edge_cutoff_scale;
    const float* convolution_source_contributions;
    const float* convolution_target_contributions;
    const float* density_source_contributions;
    const float* density_target_contributions;
    const float* convolution_parameters;
    const float* density_parameters;
    const float* edge_phi_adjoint;
    const float* node_density_adjoint;
    float* radial_adjoint;
    float* edge_cutoff_scale_adjoint;
} SymmetrixJitMH1CudaConditioningReverseArgsV3;

typedef struct SymmetrixJitMH1CudaSplineV5 {
    uint32_t struct_size;
    uint32_t edge_types;
    uint32_t intervals;
    uint32_t functions;
    double h;
    double x0;
    const void* coefficients;
} SymmetrixJitMH1CudaSplineV5;

typedef struct SymmetrixJitMH1CudaSplineRForwardArgsV5 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t active_type_count;
    uint32_t reserved;
    int64_t num_nodes;
    int64_t num_edges;
    const int32_t* node_types;
    const int32_t* num_neigh;
    const int32_t* first_neigh;
    const int32_t* neigh_indices;
    const int32_t* neigh_types;
    const double* radius;
    SymmetrixJitMH1CudaSplineV5 radial;
    const void* harmonics_values;
    const void* neighbor_features;
    const void* output_mask;
    void* node_density;
    void* output;
} SymmetrixJitMH1CudaSplineRForwardArgsV5;

typedef struct SymmetrixJitMH1CudaSplineRSourceArgsV5 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t active_type_count;
    uint32_t reserved;
    int64_t num_nodes;
    int64_t num_edges;
    int64_t source_owner_count;
    const int32_t* node_types;
    const int32_t* neigh_indices;
    const int32_t* neigh_types;
    const int32_t* source_offsets;
    const int32_t* source_edges;
    const int32_t* edge_receivers;
    const double* radius;
    SymmetrixJitMH1CudaSplineV5 radial;
    const void* output_adjoint;
    const void* output_mask;
    const void* node_density_adjoint;
    void* source_adjoint;
} SymmetrixJitMH1CudaSplineRSourceArgsV5;

typedef struct SymmetrixJitMH1CudaSplineREdgeArgsV5 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t coordinate_scalar_size;
    uint32_t coordinates_are_unit;
    int64_t num_nodes;
    int64_t num_edges;
    const void* xyz;
    const double* radius;
    SymmetrixJitMH1CudaSplineV5 radial;
    const void* harmonics_values;
    const void* harmonics_gradients;
    const void* neighbor_features;
    const void* output_adjoint;
    double* directed_forces;
} SymmetrixJitMH1CudaSplineREdgeArgsV5;

typedef struct SymmetrixJitMH1CudaNodeForwardArgsV4 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t phase;
    uint32_t flags;
    int64_t num_nodes;
    const int32_t* element_indices;
    const float* node_density;
    const float* linear_parameters;
    const float* product_parameters;
    const float* readout_parameters;
    const float* layer_input;
    const float* up;
    const float* messages;
    float* up_output;
    float* layer_output;
    float* readout_contribution;
    float* node_arena;
    float* retained_pre_gate;
    float* retained_interaction_output;
} SymmetrixJitMH1CudaNodeForwardArgsV4;

typedef struct SymmetrixJitMH1CudaNodeReverseArgsV4 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t phase;
    uint32_t flags;
    float energy_scale;
    uint32_t reserved;
    int64_t num_nodes;
    const int32_t* element_indices;
    const float* node_density;
    const float* linear_parameters;
    const float* product_parameters;
    const float* readout_parameters;
    const float* layer_input;
    const float* up;
    const float* messages;
    const float* layer_output;
    float* node_arena;
    float* layer_output_adjoint;
    float* message_adjoint;
    float* up_adjoint;
    float* layer_input_adjoint;
    float* node_density_adjoint;
    const float* retained_pre_gate;
    const float* retained_interaction_output;
} SymmetrixJitMH1CudaNodeReverseArgsV4;"""
    return _convert_gpu_module_precision(source, precision)


def _cpp_string(value: str) -> str:
    return json.dumps(value)


def _cpp_float(value: float) -> str:
    return f"static_cast<float>({repr(float(value))})"


def _convert_host_spline_precision(source: str, precision: str) -> str:
    if precision == "float32":
        return source
    if precision != "float64":
        raise ValueError("MH1 host spline precision must be 'float32' or 'float64'")
    return (
        source.replace("static_cast<float>", "static_cast<double>")
        .replace("const float", "const double")
        .replace("float*", "double*")
        .replace("float&", "double&")
        .replace("float ", "double ")
        .replace("sizeof(float)", "sizeof(double)")
    )


def _ir_mul_index(
    block: dict[str, Any], component: int, channel: str = "channel"
) -> str:
    return (
        f"{int(block['offset'])} + {component}"
        f" * {int(block['multiplicity'])} + {channel}"
    )


def _runtime_ir_mul_index(
    blocks: list[dict[str, Any]],
    block_index: int,
    component: str = "component",
    channel: str = "channel",
) -> str:
    """Return an index into the runtime parameter block's ``mul_ir`` layout."""

    block = blocks[block_index]
    packed_offset = sum(
        int(previous["multiplicity"]) * int(previous["components"])
        for previous in blocks[:block_index]
    )
    return f"{packed_offset} + {channel} * {int(block['components'])}" f" + {component}"


def _artifact_id(contract: dict[str, Any]) -> str:
    fingerprint = contract["generation_fingerprint"]
    if not fingerprint.startswith("sha256:"):
        raise ValueError("MH1 generation fingerprint is malformed")
    return (
        f"jit-mh1-gen{JIT_GENERATION_VERSION}-v3-"
        + fingerprint.removeprefix("sha256:")[:16]
    )


def _artifact_id_v4(contract: dict[str, Any]) -> str:
    fingerprint = contract["generation_fingerprint"]
    if not fingerprint.startswith("sha256:"):
        raise ValueError("MH1 v4 generation fingerprint is malformed")
    return (
        f"jit-mh1-gen{JIT_GENERATION_VERSION}-v4-"
        + fingerprint.removeprefix("sha256:")[:16]
    )


def _validate_compute_capability(compute_capability: int) -> None:
    if (
        isinstance(compute_capability, bool)
        or not isinstance(compute_capability, int)
        or compute_capability <= 0
        or compute_capability > 999
    ):
        raise ValueError("CUDA compute capability must be an integer in [1, 999]")


def _interaction_extents(interaction: dict[str, Any]) -> dict[str, int]:
    dimensions = interaction["dimensions"]
    channels = int(dimensions["channels"])
    input_1 = int(dimensions["input_1"])
    if input_1 % channels:
        raise ValueError("MH1 input-1 dimension is not divisible by channels")
    return {
        "input_1_dimension": input_1,
        "input_2_dimension": int(dimensions["input_2"]),
        "output_dimension": int(dimensions["output"]),
        "weight_size": int(dimensions["weight"]),
        "phi_dimension": int(dimensions["prefix"]),
        "multiplicity": channels,
        "input_1_angular_dimension": input_1 // channels,
        "instruction_count": len(interaction["paths"]),
    }


def _node_arena_layout(layer: dict[str, Any]) -> dict[str, Any]:
    """Return the phase-local arena required by the generated node program."""

    residual = int(layer["linears"]["linear_res"]["dimensions"]["output"])
    gated = int(layer["gate"]["dimensions"]["irreps_out"])
    interaction_output = int(layer["linears"]["linear_2"]["dimensions"]["output"])
    contracted = int(layer["product"]["dimensions"]["output"])
    input_dimension = int(layer["linears"]["linear_up"]["dimensions"]["input"])
    readout = layer["readout"]
    readout_hidden = (
        0
        if readout["class"] == "LinearReadoutBlock"
        else int(readout["linear_1"]["dimensions"]["output"])
    )
    candidates = {
        "forward_pre_gate_and_gated": residual + gated,
        "forward_gated_and_interaction_output": gated + interaction_output,
        "forward_interaction_and_contracted": interaction_output + contracted,
        "reverse_interaction_contracted_adjoint_and_interaction_adjoint": (
            2 * interaction_output + contracted
        ),
        "reverse_gated_adjoint_and_inplace_pre_gate_adjoint": gated + residual,
        "forward_layer0_embedding_skip_and_contracted": (
            interaction_output + input_dimension + 2 * contracted
            if int(layer["index"]) == 0
            else interaction_output + 2 * contracted
        ),
        "readout_reverse": 2 * readout_hidden + contracted,
    }
    return {
        "scalar_type": "float32",
        "scope": "node,phase",
        "reused_between_phases": True,
        "inplace_gate_reverse": True,
        "candidates": candidates,
        "dimension": max(candidates.values()),
    }


def _node_layer_metadata(
    layer: dict[str, Any], *, backend: str, element_count: int, node_state_policy: str
) -> dict[str, Any]:
    index = int(layer["index"])
    linears = layer["linears"]
    residual = int(linears["linear_res"]["dimensions"]["output"])
    requires_source_adjoint = bool(
        layer["derivatives"]["requires_tp_source_state_adjoint"]
    )
    packs = layer["runtime_parameter_packs"]
    if backend == "host":
        forward_schedule = [1, 1]
        reverse_schedule = [1, 1 if requires_source_adjoint else 0]
        schedule_key = "owners_per_node"
    elif backend == "cuda":
        forward_schedule = [128, 128]
        reverse_schedule = [128, 128 if requires_source_adjoint else 0]
        schedule_key = "threads_per_block"
    else:
        raise ValueError("MH1 node metadata backend must be 'host' or 'cuda'")
    arena = _node_arena_layout(layer)
    retain_pre_gate = node_state_policy not in (
        MH1_NODE_STATE_RECOMPUTE,
        MH1_NODE_STATE_RETAIN_INTERACTION,
    )
    retain_interaction_output = node_state_policy != MH1_NODE_STATE_RECOMPUTE
    reuse_message_adjoint = node_state_policy in (
        MH1_NODE_STATE_REUSE_ADJOINTS,
        MH1_NODE_STATE_RETAIN_INTERACTION,
    )
    return {
        "index": index,
        "element_count": element_count,
        "input_dimension": int(linears["linear_up"]["dimensions"]["input"]),
        "up_dimension": int(linears["linear_up"]["dimensions"]["output"]),
        "residual_dimension": int(linears["linear_res"]["dimensions"]["output"]),
        "skip_dimension": int(linears["skip"]["dimensions"]["output"]),
        "message_dimension": int(linears["linear_1"]["dimensions"]["input"]),
        "interaction_output_dimension": int(
            linears["linear_2"]["dimensions"]["output"]
        ),
        "output_dimension": int(layer["product"]["dimensions"]["output"]),
        "product_term_count": len(layer["product"]["terms"]),
        "node_arena_dimension": int(arena["dimension"]),
        "retained_pre_gate_dimension": int(residual if retain_pre_gate else 0),
        "retained_interaction_output_dimension": int(
            int(linears["linear_2"]["dimensions"]["output"])
            if retain_interaction_output
            else 0
        ),
        "node_state_policy": node_state_policy,
        "reuse_message_adjoint": reuse_message_adjoint,
        "arena": arena,
        "requires_tp_source_state_adjoint": requires_source_adjoint,
        "linear_parameter_count": int(packs["linear"]["parameter_count"]),
        "product_parameter_count": int(packs["product"]["parameter_count"]),
        "readout_parameter_count": int(packs["readout"]["parameter_count"]),
        "forward_phases": [
            {"phase": phase, schedule_key: value, "enabled": True}
            for phase, value in enumerate(forward_schedule)
        ],
        "reverse_phases": [
            {
                "phase": phase,
                schedule_key: value,
                "enabled": value > 0,
            }
            for phase, value in enumerate(reverse_schedule)
        ],
        "runtime_parameter_packs": packs,
    }


def execution_mh1_node_program_metadata(
    contract: dict[str, Any], *, backend: str, node_state_policy: str | None = None
) -> dict[str, Any]:
    """Describe ABI-v4 node extents, schedules, and runtime pack identities."""

    normalized = normalize_execution_mh1_v4_contract(contract)
    node_program = normalized["node_program"]
    embedding = node_program["input"]["embedding"]
    element_count = int(embedding["dimensions"]["input"])
    if node_state_policy is None:
        node_state_policy = (
            MH1_NODE_STATE_FULL_RETENTION
            if backend == "cuda"
            else MH1_NODE_STATE_RECOMPUTE
        )
    node_state_policy = _resolve_node_state_policy(node_state_policy)
    layers = [
        _node_layer_metadata(
            layer,
            backend=backend,
            element_count=element_count,
            node_state_policy=node_state_policy,
        )
        for layer in node_program["layers"]
    ]
    abi = (
        MH1_HOST_PLUGIN_V4_ABI
        if backend == "host"
        else MH1_CUDA_PLUGIN_V4_ABI
        if backend == "cuda"
        else None
    )
    if abi is None:
        raise ValueError("MH1 node metadata backend must be 'host' or 'cuda'")
    return {
        "tag": MH1_NODE_PROGRAM_METADATA_TAG,
        "abi": abi,
        "abi_version": 4,
        "backend": backend,
        "artifact_id": _artifact_id_v4(normalized),
        "generation_fingerprint": normalized["generation_fingerprint"],
        "semantic_fingerprint": normalized["semantic_fingerprint"],
        "structure_fingerprint": normalized["structure_fingerprint"],
        "runtime_layout_fingerprint": normalized["runtime_layout_fingerprint"],
        "feature_layout": "ir_mul",
        "element_indexing": node_program["input"]["type_indexing"],
        "node_state_policy": node_state_policy,
        "layers": layers,
    }


def _render_ir_mul_linear_helpers(
    descriptor: dict[str, Any], *, name: str, function_qualifier: str
) -> str:
    parameters = descriptor["runtime_parameters"]
    segments = {segment["name"]: segment for segment in parameters["segments"]}
    weight_offset = int(segments["weight"]["offset"])
    bias_offset = int(segments["bias"]["offset"])
    bias_count = int(segments["bias"]["count"])
    mask_offset = int(segments["output_mask"]["offset"])
    input_dimension = int(descriptor["dimensions"]["input"])
    output_dimension = int(descriptor["dimensions"]["output"])

    forward = [
        f"{function_qualifier} void {name}_forward(",
        "    const float* input, const float* parameters, float* output)",
        "{",
        f"    for (int index = 0; index < {output_dimension}; ++index)",
        "        output[index] = 0.0f;",
    ]
    reverse = [
        f"{function_qualifier} void {name}_reverse(",
        "    const float* output_adjoint, const float* parameters,",
        "    float* input_adjoint)",
        "{",
        f"    for (int index = 0; index < {input_dimension}; ++index)",
        "        input_adjoint[index] = 0.0f;",
    ]
    add_scaled = [
        f"{function_qualifier} void {name}_forward_add_scaled(",
        "    const float* input, const float* parameters,",
        "    float scale, float* output)",
        "{",
    ]
    if bias_count:
        for block_index, block in enumerate(descriptor["blocks"]["output"]):
            multiplicity = int(block["multiplicity"])
            width = int(block["components"])
            output_base = int(block["offset"])
            parameter_index = _runtime_ir_mul_index(
                descriptor["blocks"]["output"], block_index
            )
            add_scaled.extend(
                [
                    f"    for (int component = 0; component < {width}; ++component) {{",
                    f"        for (int channel = 0; channel < {multiplicity}; ++channel)",
                    f"            output[{output_base} + component * {multiplicity} + channel]",
                    f"                += scale * parameters[{bias_offset} + {parameter_index}]",
                    f"                * parameters[{mask_offset} + {parameter_index}];",
                    "    }",
                ]
            )
    for path in descriptor["instructions"]:
        input_block = descriptor["blocks"]["input"][path["input_block"]]
        output_block = descriptor["blocks"]["output"][path["output_block"]]
        input_multiplicity = int(input_block["multiplicity"])
        output_multiplicity = int(output_block["multiplicity"])
        width = int(input_block["components"])
        input_base = int(input_block["offset"])
        output_base = int(output_block["offset"])
        output_parameter_index = _runtime_ir_mul_index(
            descriptor["blocks"]["output"],
            int(path["output_block"]),
            component="component",
            channel="target",
        )
        path_weight = _cpp_float(path["path_weight"])
        path_weight_offset = weight_offset + int(path["weight_offset"])
        forward.extend(
            [
                f"    for (int component = 0; component < {width}; ++component) {{",
                f"        for (int target = 0; target < {output_multiplicity}; ++target) {{",
                "            float value = 0.0f;",
                f"            for (int source = 0; source < {input_multiplicity}; ++source)",
                f"                value += {path_weight}",
                f"                    * parameters[{path_weight_offset}",
                f"                        + source * {output_multiplicity} + target]",
                f"                    * input[{input_base}",
                f"                        + component * {input_multiplicity} + source];",
                f"            output[{output_base}",
                f"                + component * {output_multiplicity} + target] += value;",
                "        }",
                "    }",
            ]
        )
        reverse.extend(
            [
                f"    for (int component = 0; component < {width}; ++component) {{",
                f"        for (int source = 0; source < {input_multiplicity}; ++source) {{",
                "            float value = 0.0f;",
                f"            for (int target = 0; target < {output_multiplicity}; ++target) {{",
                f"                const int output_index = {output_base}",
                f"                    + component * {output_multiplicity} + target;",
                f"                const int parameter_index = {output_parameter_index};",
                f"                value += {path_weight}",
                f"                    * parameters[{path_weight_offset}",
                f"                        + source * {output_multiplicity} + target]",
                f"                    * parameters[{mask_offset} + parameter_index]",
                "                    * output_adjoint[output_index];",
                "            }",
                f"            input_adjoint[{input_base}",
                f"                + component * {input_multiplicity} + source] += value;",
                "        }",
                "    }",
            ]
        )
        add_scaled.extend(
            [
                f"    for (int component = 0; component < {width}; ++component) {{",
                f"        for (int target = 0; target < {output_multiplicity}; ++target) {{",
                "            float value = 0.0f;",
                f"            for (int source = 0; source < {input_multiplicity}; ++source)",
                f"                value += {path_weight}",
                f"                    * parameters[{path_weight_offset}",
                f"                        + source * {output_multiplicity} + target]",
                f"                    * input[{input_base}",
                f"                        + component * {input_multiplicity} + source];",
                f"            const int output_index = {output_base}",
                f"                + component * {output_multiplicity} + target;",
                f"            const int parameter_index = {output_parameter_index};",
                "            output[output_index] += scale * value",
                f"                * parameters[{mask_offset} + parameter_index];",
                "        }",
                "    }",
            ]
        )
    for block_index, block in enumerate(descriptor["blocks"]["output"]):
        multiplicity = int(block["multiplicity"])
        width = int(block["components"])
        output_base = int(block["offset"])
        parameter_index = _runtime_ir_mul_index(
            descriptor["blocks"]["output"], block_index
        )
        bias_expression = (
            f" + parameters[{bias_offset} + {parameter_index}]" if bias_count else ""
        )
        forward.extend(
            [
                f"    for (int component = 0; component < {width}; ++component) {{",
                f"        for (int channel = 0; channel < {multiplicity}; ++channel) {{",
                f"            const int output_index = {output_base}"
                f" + component * {multiplicity} + channel;",
                f"            const int parameter_index = {parameter_index};",
                "            output[output_index] = (output[output_index]"
                f"{bias_expression}) * parameters[{mask_offset} + parameter_index];",
                "        }",
                "    }",
            ]
        )
    forward.append("}")
    reverse.append("}")
    add_scaled.append("}")
    return "\n".join(forward + [""] + reverse + [""] + add_scaled)


def _node_linear_descriptors(
    contract: dict[str, Any], layer_index: int
) -> list[tuple[str, dict[str, Any], str]]:
    node_program = contract["node_program"]
    layer = node_program["layers"][layer_index]
    result = []
    if layer_index == 0:
        result.append(("node_embedding", node_program["input"]["embedding"], "linear"))
    result.extend(
        (name, descriptor, "linear") for name, descriptor in layer["linears"].items()
    )
    result.append(("product_linear", layer["product"]["linear"], "linear"))
    readout = layer["readout"]
    if readout["class"] == "LinearReadoutBlock":
        result.append(("readout_linear", readout["linear"], "readout"))
    else:
        result.extend(
            [
                ("readout_linear_1", readout["linear_1"], "readout"),
                ("readout_linear_2", readout["linear_2"], "readout"),
            ]
        )
    return result


def _render_node_embedding_lookup_helper(
    descriptor: dict[str, Any], *, function_qualifier: str
) -> str:
    path = descriptor["instructions"][0]
    input_multiplicity = int(descriptor["blocks"]["input"][0]["multiplicity"])
    output_multiplicity = int(descriptor["blocks"]["output"][0]["multiplicity"])
    if int(path["weight_count"]) != input_multiplicity * output_multiplicity:
        raise ValueError("MH1 embedding table weight layout is inconsistent")
    segments = {
        segment["name"]: segment
        for segment in descriptor["runtime_parameters"]["segments"]
    }
    weight_offset = int(segments["weight"]["offset"])
    bias_offset = int(segments["bias"]["offset"])
    bias_count = int(segments["bias"]["count"])
    mask_offset = int(segments["output_mask"]["offset"])
    bias = f" + parameters[{bias_offset} + target]" if bias_count else ""
    return f"""{function_qualifier} void execution_mh1_node_embedding_lookup(
    int element, const float* parameters, float* output)
{{
    for (int target = 0; target < {output_multiplicity}; ++target)
        output[target] = ({_cpp_float(path['path_weight'])}
            * parameters[{weight_offset} + element * {output_multiplicity} + target]{bias})
            * parameters[{mask_offset} + target];
}}"""


def render_execution_mh1_node_linear_cpp_helpers(
    contract: dict[str, Any], *, function_qualifier: str = "inline"
) -> str:
    """Render fixed-coordinate `ir_mul` linears with runtime parameters."""

    normalized = normalize_execution_mh1_v4_contract(contract)
    helpers = []
    for layer_index in range(2):
        for name, descriptor, _ in _node_linear_descriptors(normalized, layer_index):
            helpers.append(
                _render_ir_mul_linear_helpers(
                    descriptor,
                    name=f"execution_mh1_node_layer_{layer_index}_{name}",
                    function_qualifier=function_qualifier,
                )
            )
    helpers.append(
        _render_node_embedding_lookup_helper(
            normalized["node_program"]["input"]["embedding"],
            function_qualifier=function_qualifier,
        )
    )
    return "\n\n".join(helpers) + "\n"


def _product_input_expression(product: dict[str, Any], angular_index: int) -> str:
    layout = product["angular_layout"][angular_index]
    return f"input[{int(layout['component_offset'])} + channel]"


def _render_node_product_helpers(
    layer: dict[str, Any], *, function_qualifier: str
) -> str:
    index = int(layer["index"])
    product = layer["product"]
    channels = int(product["dimensions"]["channels"])
    input_dimension = int(product["dimensions"]["input"])
    output_dimension = int(product["dimensions"]["output"])
    coefficient_count = int(product["runtime_coefficient_count"])
    linear_name = f"execution_mh1_node_layer_{index}_product_linear"
    prefix = f"execution_mh1_node_layer_{index}_product"

    forward = [
        f"{function_qualifier} void {prefix}_forward(",
        "    const float* input, const float* skip,",
        "    const float* coefficients, const float* linear_parameters,",
        "    float* contracted, float* output)",
        "{",
        f"    for (int column = 0; column < {output_dimension}; ++column)",
        "        contracted[column] = 0.0f;",
    ]
    for component in product["components"]:
        first = int(component["term_begin"])
        last = int(component["term_end"])
        if first == last:
            continue
        output_offset = int(component["output_component_offset"])
        forward.extend(
            [
                f"    for (int channel = 0; channel < {channels}; ++channel) {{",
                "        float value = 0.0f;",
            ]
        )
        for term_index in range(first, last):
            term = product["terms"][term_index]
            factors = " * ".join(
                _product_input_expression(product, int(angular_index))
                for angular_index in term["angular_indices"]
            )
            forward.append(
                f"        value += coefficients[{term_index * channels} + channel]"
                f" * {factors};"
            )
        forward.extend(
            [
                f"        contracted[{output_offset} + channel] = value;",
                "    }",
            ]
        )
    forward.extend(
        [
            f"    {linear_name}_forward(contracted, linear_parameters, output);",
            f"    for (int column = 0; column < {output_dimension}; ++column)",
            "        output[column] += skip[column];",
            "}",
        ]
    )

    reverse = [
        f"{function_qualifier} void {prefix}_reverse(",
        "    const float* input, const float* output_adjoint,",
        "    const float* coefficients, const float* linear_parameters,",
        "    float* contracted_adjoint, float* input_adjoint,",
        "    float* skip_adjoint)",
        "{",
        f"    {linear_name}_reverse(",
        "        output_adjoint, linear_parameters, contracted_adjoint);",
        f"    for (int column = 0; column < {input_dimension}; ++column)",
        "        input_adjoint[column] = 0.0f;",
        f"    for (int column = 0; column < {output_dimension}; ++column)",
        "        skip_adjoint[column] = output_adjoint[column];",
    ]
    for component in product["components"]:
        first = int(component["term_begin"])
        last = int(component["term_end"])
        if first == last:
            continue
        output_offset = int(component["output_component_offset"])
        reverse.append(f"    for (int channel = 0; channel < {channels}; ++channel) {{")
        for term_index in range(first, last):
            term = product["terms"][term_index]
            angular_indices = [int(value) for value in term["angular_indices"]]
            reverse.extend(
                [
                    f"        const float common_{term_index} =",
                    f"            coefficients[{term_index * channels} + channel]",
                    f"            * contracted_adjoint[{output_offset} + channel];",
                ]
            )
            for differentiated, angular_index in enumerate(angular_indices):
                factors = [
                    _product_input_expression(product, other_index)
                    for axis, other_index in enumerate(angular_indices)
                    if axis != differentiated
                ]
                product_expression = " * " + " * ".join(factors) if factors else ""
                input_offset = int(
                    product["angular_layout"][angular_index]["component_offset"]
                )
                reverse.append(
                    f"        input_adjoint[{input_offset} + channel]"
                    f" += common_{term_index}{product_expression};"
                )
        reverse.append("    }")
    reverse.append("}")
    if coefficient_count != len(product["terms"]) * channels:
        raise ValueError("MH1 agnostic product coefficient layout is inconsistent")
    return "\n".join(forward + [""] + reverse)


def render_execution_mh1_node_product_cpp_helpers(
    contract: dict[str, Any], *, function_qualifier: str = "inline"
) -> str:
    """Render exact correlation-three products in persistent `ir_mul`."""

    normalized = normalize_execution_mh1_v4_contract(contract)
    linear_helpers = render_execution_mh1_node_linear_cpp_helpers(
        normalized, function_qualifier=function_qualifier
    )
    product_helpers = [
        _render_node_product_helpers(layer, function_qualifier=function_qualifier)
        for layer in normalized["node_program"]["layers"]
    ]
    return linear_helpers + "\n" + "\n\n".join(product_helpers) + "\n"


def _render_node_gate_helpers(layer: dict[str, Any], *, function_qualifier: str) -> str:
    index = int(layer["index"])
    gate = layer["gate"]
    prefix = f"execution_mh1_node_layer_{index}_gate"
    input_dimension = int(gate["dimensions"]["irreps_in"])
    scalar_blocks = gate["blocks"]["irreps_scalars"]
    scalar_constants = gate["scalar_activation"]["constants"]
    gate_constants = gate["gate_activation"]["constants"]

    forward = [
        f"{function_qualifier} void {prefix}_forward(",
        "    const float* input, float* output)",
        "{",
    ]
    reverse = [
        f"{function_qualifier} void {prefix}_reverse(",
        "    const float* input, const float* output_adjoint,",
        "    float* input_adjoint)",
        "{",
        f"    for (int index = 0; index < {input_dimension}; ++index)",
        "        input_adjoint[index] = 0.0f;",
    ]
    reverse_inplace = [
        f"{function_qualifier} void {prefix}_reverse_inplace(",
        "    float* input_and_adjoint, const float* output_adjoint)",
        "{",
    ]
    scalar_offset = 0
    for block, constant in zip(scalar_blocks, scalar_constants, strict=True):
        multiplicity = int(block["multiplicity"])
        scale = _cpp_float(constant)
        forward.extend(
            [
                f"    for (int channel = 0; channel < {multiplicity}; ++channel) {{",
                f"        const float value = input[{scalar_offset} + channel];",
                "        const float sigmoid = 1.0f / (1.0f + expf(-value));",
                f"        output[{scalar_offset} + channel] = {scale}",
                "            * value * sigmoid;",
                "    }",
            ]
        )
        reverse.extend(
            [
                f"    for (int channel = 0; channel < {multiplicity}; ++channel) {{",
                f"        const float value = input[{scalar_offset} + channel];",
                "        const float sigmoid = 1.0f / (1.0f + expf(-value));",
                f"        input_adjoint[{scalar_offset} + channel] =",
                f"            output_adjoint[{scalar_offset} + channel] * {scale}",
                "            * (sigmoid + value * sigmoid * (1.0f - sigmoid));",
                "    }",
            ]
        )
        reverse_inplace.extend(
            [
                f"    for (int channel = 0; channel < {multiplicity}; ++channel) {{",
                f"        const float value = input_and_adjoint[{scalar_offset} + channel];",
                "        const float sigmoid = 1.0f / (1.0f + expf(-value));",
                f"        input_and_adjoint[{scalar_offset} + channel] =",
                f"            output_adjoint[{scalar_offset} + channel] * {scale}",
                "            * (sigmoid + value * sigmoid * (1.0f - sigmoid));",
                "    }",
            ]
        )
        scalar_offset += multiplicity
    for pairing, constant in zip(gate["pairing"], gate_constants, strict=True):
        multiplicity = int(pairing["multiplicity"])
        width = int(pairing["component_width"])
        gate_offset = int(pairing["gate_offset"])
        gated_offset = int(pairing["gated_offset"])
        output_offset = int(pairing["output_offset"])
        scale = _cpp_float(constant)
        forward.extend(
            [
                f"    for (int channel = 0; channel < {multiplicity}; ++channel) {{",
                f"        const float gate_input = input[{gate_offset} + channel];",
                "        const float sigmoid = 1.0f / (1.0f + expf(-gate_input));",
                f"        const float gate_value = {scale} * sigmoid;",
                f"        for (int component = 0; component < {width}; ++component)",
                f"            output[{output_offset} + component * {multiplicity} + channel] =",
                f"                input[{gated_offset} + component * {multiplicity} + channel]",
                "                * gate_value;",
                "    }",
            ]
        )
        reverse.extend(
            [
                f"    for (int channel = 0; channel < {multiplicity}; ++channel) {{",
                f"        const float gate_input = input[{gate_offset} + channel];",
                "        const float sigmoid = 1.0f / (1.0f + expf(-gate_input));",
                f"        const float gate_value = {scale} * sigmoid;",
                "        float gate_adjoint = 0.0f;",
                f"        for (int component = 0; component < {width}; ++component) {{",
                f"            const int input_index = {gated_offset}",
                f"                + component * {multiplicity} + channel;",
                f"            const int output_index = {output_offset}",
                f"                + component * {multiplicity} + channel;",
                "            const float adjoint = output_adjoint[output_index];",
                "            input_adjoint[input_index] = adjoint * gate_value;",
                "            gate_adjoint += adjoint * input[input_index];",
                "        }",
                f"        input_adjoint[{gate_offset} + channel] = gate_adjoint",
                f"            * {scale} * sigmoid * (1.0f - sigmoid);",
                "    }",
            ]
        )
        reverse_inplace.extend(
            [
                f"    for (int channel = 0; channel < {multiplicity}; ++channel) {{",
                f"        const float gate_input = input_and_adjoint[{gate_offset} + channel];",
                "        const float sigmoid = 1.0f / (1.0f + expf(-gate_input));",
                f"        const float gate_value = {scale} * sigmoid;",
                "        float gate_adjoint = 0.0f;",
                f"        for (int component = 0; component < {width}; ++component) {{",
                f"            const int input_index = {gated_offset}",
                f"                + component * {multiplicity} + channel;",
                f"            const int output_index = {output_offset}",
                f"                + component * {multiplicity} + channel;",
                "            const float input_value = input_and_adjoint[input_index];",
                "            const float adjoint = output_adjoint[output_index];",
                "            gate_adjoint += adjoint * input_value;",
                "            input_and_adjoint[input_index] = adjoint * gate_value;",
                "        }",
                f"        input_and_adjoint[{gate_offset} + channel] = gate_adjoint",
                f"            * {scale} * sigmoid * (1.0f - sigmoid);",
                "    }",
            ]
        )
    if scalar_offset != int(gate["dimensions"]["irreps_scalars"]):
        raise ValueError("MH1 gate scalar layout is inconsistent")
    forward.append("}")
    reverse.append("}")
    reverse_inplace.append("}")
    return "\n".join(forward + [""] + reverse + [""] + reverse_inplace)


def _runtime_pack_segment(
    layer: dict[str, Any], pack_name: str, segment_name: str
) -> dict[str, Any]:
    pack = layer["runtime_parameter_packs"][pack_name]
    for segment in pack["segments"]:
        if segment["name"] == segment_name:
            return segment
    raise ValueError(
        f"MH1 node runtime pack {pack_name!r} has no {segment_name!r} segment"
    )


def _render_linear_forward_adjoint_dot(
    descriptor: dict[str, Any],
    *,
    input_name: str,
    adjoint_name: str,
    parameter_name: str,
    accumulator_name: str,
    indent: str,
) -> list[str]:
    segments = {
        segment["name"]: segment
        for segment in descriptor["runtime_parameters"]["segments"]
    }
    bias_offset = int(segments["bias"]["offset"])
    bias_count = int(segments["bias"]["count"])
    mask_offset = int(segments["output_mask"]["offset"])
    lines = []
    if bias_count:
        for block_index, block in enumerate(descriptor["blocks"]["output"]):
            multiplicity = int(block["multiplicity"])
            width = int(block["components"])
            output_base = int(block["offset"])
            parameter_index = _runtime_ir_mul_index(
                descriptor["blocks"]["output"], block_index
            )
            lines.extend(
                [
                    f"for (int component = 0; component < {width}; ++component) {{",
                    f"    for (int channel = 0; channel < {multiplicity}; ++channel) {{",
                    f"        const int output_index = {output_base}"
                    f" + component * {multiplicity} + channel;",
                    f"        const int parameter_index = {parameter_index};",
                    f"        {accumulator_name} += {adjoint_name}[output_index]",
                    f"            * {parameter_name}[{bias_offset} + parameter_index]",
                    f"            * {parameter_name}[{mask_offset} + parameter_index];",
                    "    }",
                    "}",
                ]
            )
    for path_index, path in enumerate(descriptor["instructions"]):
        input_block = descriptor["blocks"]["input"][path["input_block"]]
        output_block = descriptor["blocks"]["output"][path["output_block"]]
        input_multiplicity = int(input_block["multiplicity"])
        output_multiplicity = int(output_block["multiplicity"])
        width = int(input_block["components"])
        input_base = int(input_block["offset"])
        output_base = int(output_block["offset"])
        output_parameter_index = _runtime_ir_mul_index(
            descriptor["blocks"]["output"],
            int(path["output_block"]),
            component="component",
            channel="target",
        )
        weight_offset = int(segments["weight"]["offset"]) + int(path["weight_offset"])
        path_weight = _cpp_float(path["path_weight"])
        value_name = f"linear_value_{path_index}"
        lines.extend(
            [
                f"for (int component = 0; component < {width}; ++component) {{",
                f"    for (int target = 0; target < {output_multiplicity}; ++target) {{",
                f"        float {value_name} = 0.0f;",
                f"        for (int source = 0; source < {input_multiplicity}; ++source)",
                f"            {value_name} += {path_weight}",
                f"                * {parameter_name}[{weight_offset}",
                f"                    + source * {output_multiplicity} + target]",
                f"                * {input_name}[{input_base}",
                f"                    + component * {input_multiplicity} + source];",
                f"        const int output_index_{path_index} = {output_base}",
                f"            + component * {output_multiplicity} + target;",
                f"        const int parameter_index_{path_index} = {output_parameter_index};",
                f"        {accumulator_name} += {adjoint_name}[output_index_{path_index}]",
                f"            * {value_name}",
                f"            * {parameter_name}[{mask_offset} + parameter_index_{path_index}];",
                "    }",
                "}",
            ]
        )
    return [indent + line for line in lines]


def _render_node_normalization_helpers(
    layer: dict[str, Any], *, function_qualifier: str
) -> str:
    index = int(layer["index"])
    prefix = f"execution_mh1_node_layer_{index}_normalization"
    residual_offset = int(
        _runtime_pack_segment(layer, "linear", "linear_res")["offset"]
    )
    linear_1_offset = int(_runtime_pack_segment(layer, "linear", "linear_1")["offset"])
    normalization_offset = int(
        _runtime_pack_segment(layer, "linear", "normalization_alpha_beta")["offset"]
    )
    residual_name = f"execution_mh1_node_layer_{index}_linear_res"
    linear_1_name = f"execution_mh1_node_layer_{index}_linear_1"
    residual_dimension = int(layer["linears"]["linear_res"]["dimensions"]["output"])
    forward = f"""{function_qualifier} void {prefix}_forward(
    const float* up, const float* messages, float density,
    const float* linear_parameters, float* pre_gate)
{{
    const float alpha = linear_parameters[{normalization_offset}];
    const float beta = linear_parameters[{normalization_offset + 1}];
    const float inverse_normalization = 1.0f / (alpha + beta * density);
    {residual_name}_forward(
        up, linear_parameters + {residual_offset}, pre_gate);
    {linear_1_name}_forward_add_scaled(
        messages, linear_parameters + {linear_1_offset},
        inverse_normalization, pre_gate);
}}"""
    reverse_lines = [
        f"{function_qualifier} void {prefix}_reverse(",
        "    const float* messages, float density,",
        "    const float* linear_parameters, float* pre_gate_adjoint,",
        "    float* message_adjoint, float* up_adjoint,",
        "    float* density_adjoint)",
        "{",
        f"    const float alpha = linear_parameters[{normalization_offset}];",
        f"    const float beta = linear_parameters[{normalization_offset + 1}];",
        "    const float normalization = alpha + beta * density;",
    ]
    if index > 0:
        reverse_lines.extend(
            [
                f"    {residual_name}_reverse(",
                f"        pre_gate_adjoint, linear_parameters + {residual_offset},",
                "        up_adjoint);",
            ]
        )
    else:
        reverse_lines.append("    (void)up_adjoint;")
    reverse_lines.append("    float density_value = 0.0f;")
    reverse_lines.extend(
        _render_linear_forward_adjoint_dot(
            layer["linears"]["linear_1"],
            input_name="messages",
            adjoint_name="pre_gate_adjoint",
            parameter_name=f"(linear_parameters + {linear_1_offset})",
            accumulator_name="density_value",
            indent="    ",
        )
    )
    reverse_lines.extend(
        [
            "    *density_adjoint = -beta * density_value",
            "        / (normalization * normalization);",
            f"    for (int index = 0; index < {residual_dimension}; ++index)",
            "        pre_gate_adjoint[index] /= normalization;",
            f"    {linear_1_name}_reverse(",
            f"        pre_gate_adjoint, linear_parameters + {linear_1_offset},",
            "        message_adjoint);",
            "}",
        ]
    )
    return forward + "\n\n" + "\n".join(reverse_lines)


def _render_node_readout_helpers(
    layer: dict[str, Any], *, function_qualifier: str
) -> str:
    index = int(layer["index"])
    prefix = f"execution_mh1_node_layer_{index}_readout"
    output_dimension = int(layer["product"]["dimensions"]["output"])
    readout = layer["readout"]
    if readout["class"] == "LinearReadoutBlock":
        offset = int(_runtime_pack_segment(layer, "readout", "linear")["offset"])
        linear_name = f"execution_mh1_node_layer_{index}_readout_linear"
        return f"""{function_qualifier} void {prefix}_forward(
    const float* layer_output, const float* readout_parameters,
    float* contribution, float*)
{{
    {linear_name}_forward(
        layer_output, readout_parameters + {offset}, contribution);
}}

{function_qualifier} void {prefix}_reverse_add(
    float energy_scale, const float*, const float* readout_parameters,
    float* layer_output_adjoint, float* arena)
{{
    {linear_name}_reverse(
        &energy_scale, readout_parameters + {offset}, arena);
    for (int column = 0; column < {output_dimension}; ++column)
        layer_output_adjoint[column] += arena[column];
}}"""

    linear_1_offset = int(_runtime_pack_segment(layer, "readout", "linear_1")["offset"])
    linear_2_offset = int(_runtime_pack_segment(layer, "readout", "linear_2")["offset"])
    hidden_dimension = int(readout["linear_1"]["dimensions"]["output"])
    linear_1_name = f"execution_mh1_node_layer_{index}_readout_linear_1"
    linear_2_name = f"execution_mh1_node_layer_{index}_readout_linear_2"
    activation_lines = []
    reverse_activation_lines = []
    hidden_offset = 0
    for block, constant in zip(
        readout["linear_1"]["blocks"]["output"],
        readout["activation"]["constants"],
        strict=True,
    ):
        count = int(block["multiplicity"]) * int(block["components"])
        scale = _cpp_float(constant)
        activation_lines.extend(
            [
                f"    for (int index = 0; index < {count}; ++index) {{",
                f"        const int column = {hidden_offset} + index;",
                "        const float value = hidden[column];",
                "        const float sigmoid = 1.0f / (1.0f + expf(-value));",
                f"        hidden[column] = {scale} * value * sigmoid;",
                "    }",
            ]
        )
        reverse_activation_lines.extend(
            [
                f"    for (int index = 0; index < {count}; ++index) {{",
                f"        const int column = {hidden_offset} + index;",
                "        const float value = hidden[column];",
                "        const float sigmoid = 1.0f / (1.0f + expf(-value));",
                f"        hidden_adjoint[column] *= {scale}",
                "            * (sigmoid + value * sigmoid * (1.0f - sigmoid));",
                "    }",
            ]
        )
        hidden_offset += count
    if hidden_offset != hidden_dimension:
        raise ValueError("MH1 nonlinear readout activation layout is inconsistent")
    return f"""{function_qualifier} void {prefix}_forward(
    const float* layer_output, const float* readout_parameters,
    float* contribution, float* arena)
{{
    float* hidden = arena;
    {linear_1_name}_forward(
        layer_output, readout_parameters + {linear_1_offset}, hidden);
{chr(10).join(activation_lines)}
    {linear_2_name}_forward(
        hidden, readout_parameters + {linear_2_offset}, contribution);
}}

{function_qualifier} void {prefix}_reverse_add(
    float energy_scale, const float* layer_output,
    const float* readout_parameters, float* layer_output_adjoint,
    float* arena)
{{
    float* hidden = arena;
    float* hidden_adjoint = arena + {hidden_dimension};
    float* output_adjoint = arena + {2 * hidden_dimension};
    {linear_1_name}_forward(
        layer_output, readout_parameters + {linear_1_offset}, hidden);
    {linear_2_name}_reverse(
        &energy_scale, readout_parameters + {linear_2_offset},
        hidden_adjoint);
{chr(10).join(reverse_activation_lines)}
    {linear_1_name}_reverse(
        hidden_adjoint, readout_parameters + {linear_1_offset},
        output_adjoint);
    for (int column = 0; column < {output_dimension}; ++column)
        layer_output_adjoint[column] += output_adjoint[column];
}}"""


def render_execution_mh1_node_nonlinear_cpp_helpers(
    contract: dict[str, Any], *, function_qualifier: str = "inline"
) -> str:
    """Render generated linears, products, and gates for both node layers."""

    normalized = normalize_execution_mh1_v4_contract(contract)
    product_helpers = render_execution_mh1_node_product_cpp_helpers(
        normalized, function_qualifier=function_qualifier
    )
    gate_helpers = [
        _render_node_gate_helpers(layer, function_qualifier=function_qualifier)
        for layer in normalized["node_program"]["layers"]
    ]
    normalization_helpers = [
        _render_node_normalization_helpers(layer, function_qualifier=function_qualifier)
        for layer in normalized["node_program"]["layers"]
    ]
    readout_helpers = [
        _render_node_readout_helpers(layer, function_qualifier=function_qualifier)
        for layer in normalized["node_program"]["layers"]
    ]
    return (
        product_helpers
        + "\n"
        + "\n\n".join(gate_helpers + normalization_helpers + readout_helpers)
        + "\n"
    )


def _render_node_phase_helpers(
    layer: dict[str, Any], *, function_qualifier: str
) -> str:
    index = int(layer["index"])
    prefix = f"execution_mh1_node_layer_{index}"
    input_dimension = int(layer["linears"]["linear_up"]["dimensions"]["input"])
    residual_dimension = int(layer["linears"]["linear_res"]["dimensions"]["output"])
    gated_dimension = int(layer["gate"]["dimensions"]["irreps_out"])
    interaction_dimension = int(layer["linears"]["linear_2"]["dimensions"]["output"])
    output_dimension = int(layer["product"]["dimensions"]["output"])
    linear_offsets = {
        name: int(_runtime_pack_segment(layer, "linear", name)["offset"])
        for name in (
            (["node_embedding"] if index == 0 else [])
            + [
                "linear_up",
                "linear_res",
                "skip",
                "linear_1",
                "linear_2",
                "product_linear",
                "normalization_alpha_beta",
            ]
        )
    }
    up_name = f"{prefix}_linear_up"
    skip_name = f"{prefix}_skip"
    linear_2_name = f"{prefix}_linear_2"
    normalization_name = f"{prefix}_normalization"
    gate_name = f"{prefix}_gate"
    product_name = f"{prefix}_product"
    readout_name = f"{prefix}_readout"

    if index == 0:
        pre_forward_body = f"""    float* embedded = arena;
    execution_mh1_node_embedding_lookup(
        element, linear_parameters + {linear_offsets['node_embedding']},
        embedded);
    {up_name}_forward(
        embedded, linear_parameters + {linear_offsets['linear_up']},
        up_output);"""
        skip_setup = f"""    float* embedded = arena + {interaction_dimension};
    float* skip = embedded + {input_dimension};
    float* contracted = skip + {output_dimension};
    execution_mh1_node_embedding_lookup(
        element, linear_parameters + {linear_offsets['node_embedding']},
        embedded);
    {skip_name}_forward(
        embedded, linear_parameters + {linear_offsets['skip']}, skip);"""
        layer_input_unused = "    (void)layer_input;\n"
        reverse_skip = (
            "    (void)element;\n    (void)layer_input;\n    (void)layer_input_adjoint;"
        )
    else:
        pre_forward_body = f"""    (void)element;
    (void)arena;
    {up_name}_forward(
        layer_input, linear_parameters + {linear_offsets['linear_up']},
        up_output);"""
        skip_setup = f"""    float* skip = arena + {interaction_dimension};
    float* contracted = skip + {output_dimension};
    {skip_name}_forward(
        layer_input, linear_parameters + {linear_offsets['skip']}, skip);"""
        layer_input_unused = "    (void)element;\n"
        reverse_skip = f"""    (void)element;
    {skip_name}_reverse(
        layer_output_adjoint, linear_parameters + {linear_offsets['skip']},
        layer_input_adjoint);"""

    pre_forward = f"""{function_qualifier} void {prefix}_pre_forward(
    int element, const float* layer_input, const float* linear_parameters,
    float* arena, float* up_output)
{{
{pre_forward_body}
}}"""
    post_forward = f"""{function_qualifier} void {prefix}_post_forward(
    int element, float density, const float* layer_input, const float* up,
    const float* messages, const float* linear_parameters,
    const float* product_parameters, const float* readout_parameters,
    float* arena, float* layer_output, float* readout_contribution)
{{
{layer_input_unused}    {normalization_name}_forward(
        up, messages, density, linear_parameters, arena);
    {gate_name}_forward(arena, arena + {residual_dimension});
    {linear_2_name}_forward(
        arena + {residual_dimension},
        linear_parameters + {linear_offsets['linear_2']}, arena);
{skip_setup}
    {product_name}_forward(
        arena, skip, product_parameters,
        linear_parameters + {linear_offsets['product_linear']},
        contracted, layer_output);
    {readout_name}_forward(
        layer_output, readout_parameters, readout_contribution, arena);
}}"""
    tile_gate_forward = f"""{function_qualifier} void {prefix}_tile_gate_forward(
    float density, const float* residual, const float* linear_1_output,
    const float* linear_parameters, float* pre_gate, float* gated)
{{
    const float alpha = linear_parameters[{linear_offsets['normalization_alpha_beta']}];
    const float beta = linear_parameters[{linear_offsets['normalization_alpha_beta'] + 1}];
    const float inverse = 1.0f / (alpha + beta * density);
    for (int column = 0; column < {residual_dimension}; ++column)
        pre_gate[column] = residual[column] + linear_1_output[column] * inverse;
    {gate_name}_forward(pre_gate, gated);
}}"""
    tile_product_forward = f"""{function_qualifier} void {prefix}_tile_product_forward(
    int element, const float* layer_input, const float* interaction_output,
    const float* linear_parameters, const float* product_parameters,
    const float* readout_parameters, float* arena, float* layer_output,
    float* readout_contribution)
{{
{layer_input_unused}{skip_setup}
    {product_name}_forward(
        interaction_output, skip, product_parameters,
        linear_parameters + {linear_offsets['product_linear']},
        contracted, layer_output);
    {readout_name}_forward(
        layer_output, readout_parameters, readout_contribution, arena);
}}"""
    post_reverse = f"""{function_qualifier} void {prefix}_post_reverse(
    int element, float density, float energy_scale,
    const float* layer_input, const float* up, const float* messages,
    const float* layer_output, const float* linear_parameters,
    const float* product_parameters, const float* readout_parameters,
    float* arena, float* layer_output_adjoint, float* message_adjoint,
    float* up_adjoint, float* layer_input_adjoint, float* density_adjoint)
{{
    {readout_name}_reverse_add(
        energy_scale, layer_output, readout_parameters,
        layer_output_adjoint, arena);
    {normalization_name}_forward(
        up, messages, density, linear_parameters, arena);
    {gate_name}_forward(arena, arena + {residual_dimension});
    {linear_2_name}_forward(
        arena + {residual_dimension},
        linear_parameters + {linear_offsets['linear_2']}, arena);
    float* contracted_adjoint = arena + {interaction_dimension};
    float* interaction_adjoint = contracted_adjoint + {output_dimension};
    {product_name}_reverse(
        arena, layer_output_adjoint, product_parameters,
        linear_parameters + {linear_offsets['product_linear']},
        contracted_adjoint, interaction_adjoint, layer_output_adjoint);
    {linear_2_name}_reverse(
        interaction_adjoint,
        linear_parameters + {linear_offsets['linear_2']}, arena);
    float* pre_gate_adjoint = arena + {gated_dimension};
    {normalization_name}_forward(
        up, messages, density, linear_parameters, pre_gate_adjoint);
    {gate_name}_reverse_inplace(pre_gate_adjoint, arena);
    {normalization_name}_reverse(
        messages, density, linear_parameters, pre_gate_adjoint,
        message_adjoint, up_adjoint, density_adjoint);
{reverse_skip}
}}"""
    tile_product_reverse = f"""{function_qualifier} void {prefix}_tile_product_reverse(
    int element, float energy_scale, const float* layer_input,
    const float* interaction_output, const float* layer_output,
    const float* linear_parameters, const float* product_parameters,
    const float* readout_parameters, float* arena,
    float* layer_output_adjoint, float* interaction_adjoint,
    float* layer_input_adjoint)
{{
    {readout_name}_reverse_add(
        energy_scale, layer_output, readout_parameters,
        layer_output_adjoint, arena);
    float* contracted_adjoint = arena;
    {product_name}_reverse(
        interaction_output, layer_output_adjoint, product_parameters,
        linear_parameters + {linear_offsets['product_linear']},
        contracted_adjoint, interaction_adjoint, layer_output_adjoint);
{reverse_skip}
}}"""
    tile_gate_reverse = f"""{function_qualifier} void {prefix}_tile_gate_reverse(
    float density, const float* residual, const float* linear_1_output,
    const float* linear_parameters, const float* gated_adjoint,
    float* pre_gate, float* pre_gate_adjoint, float* linear_1_adjoint,
    float* density_adjoint)
{{
    const float alpha = linear_parameters[{linear_offsets['normalization_alpha_beta']}];
    const float beta = linear_parameters[{linear_offsets['normalization_alpha_beta'] + 1}];
    const float normalization = alpha + beta * density;
    for (int column = 0; column < {residual_dimension}; ++column)
        pre_gate[column] = residual[column]
            + linear_1_output[column] / normalization;
    {gate_name}_reverse(pre_gate, gated_adjoint, pre_gate_adjoint);
    float density_value = 0.0f;
    for (int column = 0; column < {residual_dimension}; ++column) {{
        density_value -= beta * pre_gate_adjoint[column]
            * linear_1_output[column] / (normalization * normalization);
        linear_1_adjoint[column] = pre_gate_adjoint[column] / normalization;
    }}
    *density_adjoint = density_value;
}}"""
    helpers = [
        pre_forward,
        post_forward,
        tile_gate_forward,
        tile_product_forward,
        post_reverse,
        tile_product_reverse,
        tile_gate_reverse,
    ]
    if index > 0:
        helpers.append(
            f"""{function_qualifier} void {prefix}_pre_reverse(
    const float* up_adjoint, const float* linear_parameters,
    float* arena, float* layer_input_adjoint)
{{
    {up_name}_reverse(
        up_adjoint, linear_parameters + {linear_offsets['linear_up']}, arena);
    for (int column = 0; column < {input_dimension}; ++column)
        layer_input_adjoint[column] += arena[column];
}}"""
        )
    return "\n\n".join(helpers)


def render_execution_mh1_node_program_cpp_helpers(
    contract: dict[str, Any], *, function_qualifier: str = "inline"
) -> str:
    """Render complete per-node ABI-v4 forward and coordinate-reverse phases."""

    normalized = normalize_execution_mh1_v4_contract(contract)
    primitives = render_execution_mh1_node_nonlinear_cpp_helpers(
        normalized, function_qualifier=function_qualifier
    )
    phases = [
        _render_node_phase_helpers(layer, function_qualifier=function_qualifier)
        for layer in normalized["node_program"]["layers"]
    ]
    return primitives + "\n" + "\n\n".join(phases) + "\n"


def _conditioner_network_layout(
    descriptor: dict[str, Any],
    *,
    role: str,
    layer_count: int,
) -> dict[str, Any]:
    layers = descriptor["layers"][:layer_count]
    parameter_offset = 0
    packed_layers = []
    for program_index, layer in enumerate(layers):
        packed = {
            "program_index": program_index,
            "model_layer_index": int(layer["index"]),
            "type": layer["type"],
            "parameter_offset": parameter_offset,
        }
        if layer["type"] == "linear":
            input_dimension = int(layer["input_dimension"])
            output_dimension = int(layer["output_dimension"])
            weight_count = input_dimension * output_dimension
            packed.update(
                {
                    "input_dimension": input_dimension,
                    "output_dimension": output_dimension,
                    "weight_offset": parameter_offset,
                    "weight_count": weight_count,
                    "bias_offset": parameter_offset + weight_count,
                    "bias_count": output_dimension,
                    "parameter_count": weight_count + output_dimension,
                }
            )
        elif layer["type"] == "layer_norm":
            dimension = int(layer["dimension"])
            packed.update(
                {
                    "input_dimension": dimension,
                    "output_dimension": dimension,
                    "dimension": dimension,
                    "eps": float(layer["eps"]),
                    "gamma_offset": parameter_offset,
                    "gamma_count": dimension,
                    "beta_offset": parameter_offset + dimension,
                    "beta_count": dimension,
                    "parameter_count": 2 * dimension,
                }
            )
        else:
            dimension = int(layer["dimension"])
            packed.update(
                {
                    "input_dimension": dimension,
                    "output_dimension": dimension,
                    "dimension": dimension,
                    "parameter_count": 0,
                }
            )
        parameter_offset += int(packed["parameter_count"])
        packed_layers.append(packed)

    dynamic_input_dimension = int(descriptor["dynamic_input_dimension"])
    if packed_layers:
        output_dimension = int(packed_layers[-1]["output_dimension"])
        first_layer_contribution_dimension = int(packed_layers[0]["output_dimension"])
    else:
        output_dimension = dynamic_input_dimension
        first_layer_contribution_dimension = 0
    runtime_parameter_count = 0
    for layer in descriptor["layers"]:
        if layer["type"] == "linear":
            runtime_parameter_count += int(layer["input_dimension"]) * int(
                layer["output_dimension"]
            ) + int(layer["output_dimension"])
        elif layer["type"] == "layer_norm":
            runtime_parameter_count += 2 * int(layer["dimension"])
    return {
        "role": role,
        "conditioned_input_dimension": int(descriptor["conditioned_input_dimension"]),
        "dynamic_input_dimension": dynamic_input_dimension,
        "source_condition_dimension": int(descriptor["source_condition_dimension"]),
        "target_condition_dimension": int(descriptor["target_condition_dimension"]),
        "first_layer_contribution_dimension": (first_layer_contribution_dimension),
        "output_dimension": output_dimension,
        "layer_count": len(packed_layers),
        "parameter_count": parameter_offset,
        "consumed_parameter_count": parameter_offset,
        "runtime_parameter_count": runtime_parameter_count,
        "excluded_parameter_count": runtime_parameter_count - parameter_offset,
        "layers": packed_layers,
    }


def execution_mh1_conditioner_layout_metadata(
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Describe runtime parameter arrays for generated MH-1 conditioners.

    Each interaction owns two independent packed arrays. Linear layers store a
    row-major weight followed by bias, LayerNorm stores gamma followed by beta,
    and SiLU stores no values. The conditioned tensor-product helper consumes
    only the prefix of its full runtime pack because the final affine is owned
    directly by the UVU program. The density helper consumes its full pack.
    """

    normalized = normalize_execution_mh1_contract(contract)
    if any("density_mlp" not in item for item in normalized["interactions"]):
        raise ValueError("MH1 conditioner generation requires density-network metadata")
    interactions = []
    for interaction in normalized["interactions"]:
        conditioned = interaction["conditioned_mlp"]
        density = interaction["density_mlp"]
        interactions.append(
            {
                "index": int(interaction["index"]),
                "conditioned_prefix": _conditioner_network_layout(
                    conditioned,
                    role="conditioned_mlp_prefix",
                    layer_count=int(conditioned["factorization"]["prefix_layer_count"]),
                ),
                "density": _conditioner_network_layout(
                    density,
                    role="density_mlp",
                    layer_count=len(density["layers"]),
                ),
            }
        )
    return {
        "tag": MH1_CONDITIONER_LAYOUT_TAG,
        "generation_fingerprint": normalized["generation_fingerprint"],
        "scalar_type": "float32",
        "array_scope": "interaction,network",
        "network_order": ["conditioned_prefix", "density"],
        "linear_parameter_order": ["weight_row_major", "bias"],
        "layer_norm_parameter_order": ["gamma", "beta"],
        "interactions": interactions,
    }


def _render_conditioner_forward_evaluation(
    layout: dict[str, Any],
    *,
    indent: str,
) -> tuple[list[str], str]:
    lines: list[str] = []
    for layer in layout["layers"]:
        layer_index = int(layer["program_index"])
        output_dimension = int(layer["output_dimension"])
        output_name = f"activation_{layer_index + 1}"
        lines.append(f"float {output_name}[{output_dimension}];")
        if layer["type"] == "linear":
            input_dimension = int(layer["input_dimension"])
            active_input_dimension = (
                int(layout["dynamic_input_dimension"])
                if layer_index == 0
                else input_dimension
            )
            input_name = "radial" if layer_index == 0 else f"activation_{layer_index}"
            lines.extend(
                [
                    f"for (int row = 0; row < {output_dimension}; ++row) {{",
                    f"    float value = parameters[{int(layer['bias_offset'])} + row];",
                ]
            )
            if layer_index == 0:
                lines.append(
                    "    value += source_first_layer[row] + target_first_layer[row];"
                )
            lines.extend(
                [
                    "    for (int column = 0; column < "
                    f"{active_input_dimension}; ++column) {{",
                    "        value += parameters["
                    f"{int(layer['weight_offset'])} + row * {input_dimension} + column]"
                    f" * {input_name}[column];",
                    "    }",
                    f"    {output_name}[row] = value;",
                    "}",
                ]
            )
        elif layer["type"] == "layer_norm":
            input_name = f"activation_{layer_index}"
            dimension = int(layer["dimension"])
            mean = f"mean_{layer_index}"
            variance = f"variance_{layer_index}"
            inverse_stddev = f"inverse_stddev_{layer_index}"
            lines.extend(
                [
                    f"float {mean} = 0.0f;",
                    f"for (int column = 0; column < {dimension}; ++column) {{",
                    f"    {mean} += {input_name}[column];",
                    "}",
                    f"{mean} /= static_cast<float>({dimension});",
                    f"float {variance} = 0.0f;",
                    f"for (int column = 0; column < {dimension}; ++column) {{",
                    f"    const float centered = {input_name}[column] - {mean};",
                    f"    {variance} += centered * centered;",
                    "}",
                    f"{variance} /= static_cast<float>({dimension});",
                    f"const float {inverse_stddev} = 1.0f / sqrtf({variance} + "
                    f"{_cpp_float(layer['eps'])});",
                    f"for (int column = 0; column < {dimension}; ++column) {{",
                    f"    {output_name}[column] = ({input_name}[column] - {mean})"
                    f" * {inverse_stddev}"
                    f" * parameters[{int(layer['gamma_offset'])} + column]"
                    f" + parameters[{int(layer['beta_offset'])} + column];",
                    "}",
                ]
            )
        else:
            input_name = f"activation_{layer_index}"
            dimension = int(layer["dimension"])
            lines.extend(
                [
                    f"for (int column = 0; column < {dimension}; ++column) {{",
                    f"    const float value = {input_name}[column];",
                    "    const float sigmoid = 1.0f / (1.0f + expf(-value));",
                    f"    {output_name}[column] = value * sigmoid;",
                    "}",
                ]
            )
    return [indent + line if line else line for line in lines], (
        f"activation_{len(layout['layers'])}" if layout["layers"] else "radial"
    )


def _canonical_bounded_conditioner_layout(
    network_name: str, layout: dict[str, Any]
) -> (
    tuple[
        int,
        list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
        dict[str, Any] | None,
    ]
    | None
):
    layers = layout["layers"]
    tail = None
    if network_name == "conditioned_prefix":
        block_layers = layers
    elif network_name == "density" and len(layers) == 4:
        block_layers = layers[:3]
        tail = layers[3]
        if tail["type"] != "linear" or int(tail["output_dimension"]) != 1:
            return None
    else:
        return None
    if not block_layers or len(block_layers) % 3:
        return None
    blocks = []
    hidden_dimension = int(block_layers[0]["output_dimension"])
    for offset in range(0, len(block_layers), 3):
        linear, layer_norm, silu = block_layers[offset : offset + 3]
        if (
            [linear["type"], layer_norm["type"], silu["type"]]
            != ["linear", "layer_norm", "silu"]
            or int(linear["output_dimension"]) != hidden_dimension
            or int(layer_norm["dimension"]) != hidden_dimension
            or int(silu["dimension"]) != hidden_dimension
            or (offset > 0 and int(linear["input_dimension"]) != hidden_dimension)
        ):
            return None
        blocks.append((linear, layer_norm, silu))
    if int(layout["output_dimension"]) != (1 if tail is not None else hidden_dimension):
        return None
    if tail is not None and int(tail["input_dimension"]) != hidden_dimension:
        return None
    return hidden_dimension, blocks, tail


def _render_bounded_forward_prefix(
    layout: dict[str, Any], stop: int
) -> tuple[list[str], str, str]:
    lines = []
    input_name = "radial"
    linear_output = ""
    for layer in layout["layers"][: stop + 1]:
        layer_index = int(layer["program_index"])
        output_name = "forward_a" if layer_index % 2 == 0 else "forward_b"
        if layer["type"] == "linear":
            input_dimension = int(layer["input_dimension"])
            active_input_dimension = (
                int(layout["dynamic_input_dimension"])
                if layer_index == 0
                else input_dimension
            )
            lines.extend(
                [
                    f"for (int row = 0; row < {int(layer['output_dimension'])}; ++row) {{",
                    f"    float value = parameters[{int(layer['bias_offset'])} + row];",
                ]
            )
            if layer_index == 0:
                lines.append(
                    "    value += source_first_layer[row] + target_first_layer[row];"
                )
            lines.extend(
                [
                    f"    for (int column = 0; column < {active_input_dimension}; ++column) {{",
                    f"        value += parameters[{int(layer['weight_offset'])}"
                    f" + row * {input_dimension} + column] * {input_name}[column];",
                    "    }",
                    f"    {output_name}[row] = value;",
                    "}",
                ]
            )
            linear_output = output_name
        elif layer["type"] == "layer_norm":
            dimension = int(layer["dimension"])
            mean = f"mean_{layer_index}"
            variance = f"variance_{layer_index}"
            inverse_stddev = f"inverse_stddev_{layer_index}"
            lines.extend(
                [
                    f"float {mean} = 0.0f;",
                    f"for (int column = 0; column < {dimension}; ++column) {{",
                    f"    {mean} += {input_name}[column];",
                    "}",
                    f"{mean} /= static_cast<float>({dimension});",
                    f"float {variance} = 0.0f;",
                    f"for (int column = 0; column < {dimension}; ++column) {{",
                    f"    const float centered = {input_name}[column] - {mean};",
                    f"    {variance} += centered * centered;",
                    "}",
                    f"{variance} /= static_cast<float>({dimension});",
                    f"const float {inverse_stddev} = 1.0f / sqrtf({variance} + "
                    f"{_cpp_float(layer['eps'])});",
                    f"for (int column = 0; column < {dimension}; ++column) {{",
                    f"    {output_name}[column] = ({input_name}[column] - {mean})"
                    f" * {inverse_stddev}"
                    f" * parameters[{int(layer['gamma_offset'])} + column]"
                    f" + parameters[{int(layer['beta_offset'])} + column];",
                    "}",
                ]
            )
        else:
            dimension = int(layer["dimension"])
            lines.extend(
                [
                    f"for (int column = 0; column < {dimension}; ++column) {{",
                    f"    const float value = {input_name}[column];",
                    "    const float sigmoid = 1.0f / (1.0f + expf(-value));",
                    f"    {output_name}[column] = value * sigmoid;",
                    "}",
                ]
            )
        input_name = output_name
    return lines, linear_output, input_name


def _render_bounded_conditioner_reverse(
    interaction_index: int,
    network_name: str,
    layout: dict[str, Any],
    function_qualifier: str,
) -> str | None:
    canonical = _canonical_bounded_conditioner_layout(network_name, layout)
    if canonical is None:
        return None
    hidden_dimension, blocks, tail = canonical
    prefix = f"execution_mh1_{network_name}"
    lines = [
        f"    float forward_a[{hidden_dimension}];",
        f"    float forward_b[{hidden_dimension}];",
        f"    float current_adjoint[{hidden_dimension}];",
    ]
    if tail is None:
        lines.extend(
            [
                f"    for (int column = 0; column < {hidden_dimension}; ++column) {{",
                "        current_adjoint[column] = output_adjoint[column];",
                "    }",
            ]
        )
    else:
        lines.extend(
            [
                f"    for (int column = 0; column < {hidden_dimension}; ++column) {{",
                "        float value = 0.0f;",
                "        for (int row = 0; row < 1; ++row) {",
                "            value += output_adjoint[row] * parameters["
                f"{int(tail['weight_offset'])} + row * {hidden_dimension} + column];",
                "        }",
                "        current_adjoint[column] = value;",
                "    }",
            ]
        )
    for block_index in reversed(range(len(blocks))):
        linear, layer_norm, _ = blocks[block_index]
        stop = int(layer_norm["program_index"])
        forward_lines, linear_output, normalized_output = (
            _render_bounded_forward_prefix(layout, stop)
        )
        dimension = hidden_dimension
        lines.append("    {")
        lines.extend(f"        {line}" for line in forward_lines)
        lines.extend(
            [
                f"        for (int column = 0; column < {dimension}; ++column) {{",
                f"            const float value = {normalized_output}[column];",
                "            const float sigmoid = 1.0f / (1.0f + expf(-value));",
                "            current_adjoint[column] *= "
                "(sigmoid + value * sigmoid * (1.0f - sigmoid));",
                "        }",
                "        float reverse_mean = 0.0f;",
                f"        for (int column = 0; column < {dimension}; ++column) {{",
                f"            reverse_mean += {linear_output}[column];",
                "        }",
                f"        reverse_mean /= static_cast<float>({dimension});",
                "        float reverse_variance = 0.0f;",
                f"        for (int column = 0; column < {dimension}; ++column) {{",
                f"            const float centered = {linear_output}[column]"
                " - reverse_mean;",
                "            reverse_variance += centered * centered;",
                "        }",
                f"        reverse_variance /= static_cast<float>({dimension});",
                "        const float reverse_inverse_stddev = 1.0f / sqrtf("
                f"reverse_variance + {_cpp_float(layer_norm['eps'])});",
                "        float sum_scaled = 0.0f;",
                "        float sum_scaled_normalized = 0.0f;",
                f"        for (int column = 0; column < {dimension}; ++column) {{",
                f"            const float normalized = ({linear_output}[column]"
                " - reverse_mean) * reverse_inverse_stddev;",
                "            const float scaled_adjoint = current_adjoint[column]"
                f" * parameters[{int(layer_norm['gamma_offset'])} + column];",
                "            sum_scaled += scaled_adjoint;",
                "            sum_scaled_normalized += scaled_adjoint * normalized;",
                "        }",
                f"        for (int column = 0; column < {dimension}; ++column) {{",
                f"            const float normalized = ({linear_output}[column]"
                " - reverse_mean) * reverse_inverse_stddev;",
                "            const float scaled_adjoint = current_adjoint[column]"
                f" * parameters[{int(layer_norm['gamma_offset'])} + column];",
                f"            {normalized_output}[column] = reverse_inverse_stddev * (",
                f"                static_cast<float>({dimension}) * scaled_adjoint",
                "                - sum_scaled - normalized * sum_scaled_normalized)"
                f" / static_cast<float>({dimension});",
                "        }",
            ]
        )
        input_dimension = (
            int(layout["dynamic_input_dimension"])
            if block_index == 0
            else int(linear["input_dimension"])
        )
        destination = "radial_adjoint" if block_index == 0 else "current_adjoint"
        lines.extend(
            [
                f"        for (int column = 0; column < {input_dimension}; ++column) {{",
                "            float value = 0.0f;",
                f"            for (int row = 0; row < {dimension}; ++row) {{",
                f"                value += {normalized_output}[row] * parameters["
                f"{int(linear['weight_offset'])} + row * "
                f"{int(linear['input_dimension'])} + column];",
                "            }",
                f"            {destination}[column] = value;",
                "        }",
                "    }",
            ]
        )
    return f"""{function_qualifier} void {prefix}_radial_reverse_{interaction_index}(
    const float* radial,
    const float* source_first_layer,
    const float* target_first_layer,
    const float* parameters,
    const float* output_adjoint,
    float* radial_adjoint) noexcept
{{
{chr(10).join(lines)}
}}"""


def _resolve_conditioner_reverse_schedule(
    conditioner_layout: dict[str, Any],
) -> dict[str, Any]:
    interactions = []
    for interaction in conditioner_layout["interactions"]:
        resolved = {"index": int(interaction["index"])}
        for network_name in ("conditioned_prefix", "density"):
            resolved[network_name] = (
                _MH1_CONDITIONER_REVERSE_SCHEDULE
                if _canonical_bounded_conditioner_layout(
                    network_name, interaction[network_name]
                )
                is not None
                else _MH1_CONDITIONER_GENERIC_REVERSE_SCHEDULE
            )
        interactions.append(resolved)
    payload = {
        "tag": _MH1_CONDITIONER_REVERSE_SCHEDULE_TAG,
        "generation_fingerprint": conditioner_layout["generation_fingerprint"],
        "interactions": interactions,
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()
    return {**payload, "schedule_id": f"sha256:{digest}"}


def _render_conditioner_network_helpers(
    interaction_index: int,
    network_name: str,
    layout: dict[str, Any],
    function_qualifier: str,
    *,
    bounded_reverse: bool = False,
) -> str:
    prefix = f"execution_mh1_{network_name}"
    forward_lines, output_name = _render_conditioner_forward_evaluation(
        layout, indent="    "
    )
    output_dimension = int(layout["output_dimension"])
    if not layout["layers"]:
        unused = """    (void)source_first_layer;
    (void)target_first_layer;
    (void)parameters;"""
    else:
        unused = ""
    forward = f"""{function_qualifier} void {prefix}_forward_{interaction_index}(
    const float* radial,
    const float* source_first_layer,
    const float* target_first_layer,
    const float* parameters,
    float* output) noexcept
{{
{unused}
{chr(10).join(forward_lines)}
    for (int column = 0; column < {output_dimension}; ++column) {{
        output[column] = {output_name}[column];
    }}
}}"""

    reverse_forward_lines, _ = _render_conditioner_forward_evaluation(
        layout, indent="    "
    )
    reverse_lines = list(reverse_forward_lines)
    layer_count = int(layout["layer_count"])
    reverse_lines.extend(
        [
            f"    float adjoint_{layer_count}[{output_dimension}];",
            f"    for (int column = 0; column < {output_dimension}; ++column) {{",
            f"        adjoint_{layer_count}[column] = output_adjoint[column];",
            "    }",
        ]
    )
    for layer in reversed(layout["layers"]):
        layer_index = int(layer["program_index"])
        input_dimension = (
            int(layout["dynamic_input_dimension"])
            if layer_index == 0
            else int(layer["input_dimension"])
        )
        reverse_lines.append(f"    float adjoint_{layer_index}[{input_dimension}];")
        if layer["type"] == "linear":
            full_input_dimension = int(layer["input_dimension"])
            output_size = int(layer["output_dimension"])
            reverse_lines.extend(
                [
                    "    for (int column = 0; column < "
                    f"{input_dimension}; ++column) {{",
                    "        float value = 0.0f;",
                    f"        for (int row = 0; row < {output_size}; ++row) {{",
                    "            value += adjoint_"
                    f"{layer_index + 1}[row] * parameters["
                    f"{int(layer['weight_offset'])} + row * "
                    f"{full_input_dimension} + column];",
                    "        }",
                    f"        adjoint_{layer_index}[column] = value;",
                    "    }",
                ]
            )
        elif layer["type"] == "layer_norm":
            dimension = int(layer["dimension"])
            input_name = f"activation_{layer_index}"
            mean = f"reverse_mean_{layer_index}"
            variance = f"reverse_variance_{layer_index}"
            inverse_stddev = f"reverse_inverse_stddev_{layer_index}"
            normalized = f"normalized_{layer_index}"
            scaled_adjoint = f"scaled_adjoint_{layer_index}"
            sum_scaled = f"sum_scaled_{layer_index}"
            sum_scaled_normalized = f"sum_scaled_normalized_{layer_index}"
            reverse_lines.extend(
                [
                    f"    float {mean} = 0.0f;",
                    f"    for (int column = 0; column < {dimension}; ++column) {{",
                    f"        {mean} += {input_name}[column];",
                    "    }",
                    f"    {mean} /= static_cast<float>({dimension});",
                    f"    float {variance} = 0.0f;",
                    f"    for (int column = 0; column < {dimension}; ++column) {{",
                    f"        const float centered = {input_name}[column] - {mean};",
                    f"        {variance} += centered * centered;",
                    "    }",
                    f"    {variance} /= static_cast<float>({dimension});",
                    f"    const float {inverse_stddev} = 1.0f / sqrtf("
                    f"{variance} + {_cpp_float(layer['eps'])});",
                    f"    float {normalized}[{dimension}];",
                    f"    float {scaled_adjoint}[{dimension}];",
                    f"    float {sum_scaled} = 0.0f;",
                    f"    float {sum_scaled_normalized} = 0.0f;",
                    f"    for (int column = 0; column < {dimension}; ++column) {{",
                    f"        {normalized}[column] = ({input_name}[column] - {mean})"
                    f" * {inverse_stddev};",
                    f"        {scaled_adjoint}[column] = "
                    f"adjoint_{layer_index + 1}[column]"
                    f" * parameters[{int(layer['gamma_offset'])} + column];",
                    f"        {sum_scaled} += {scaled_adjoint}[column];",
                    f"        {sum_scaled_normalized} += {scaled_adjoint}[column]"
                    f" * {normalized}[column];",
                    "    }",
                    f"    for (int column = 0; column < {dimension}; ++column) {{",
                    f"        adjoint_{layer_index}[column] = {inverse_stddev} * (",
                    f"            static_cast<float>({dimension})"
                    f" * {scaled_adjoint}[column]",
                    f"            - {sum_scaled} - {normalized}[column]"
                    f" * {sum_scaled_normalized})"
                    f" / static_cast<float>({dimension});",
                    "    }",
                ]
            )
        else:
            dimension = int(layer["dimension"])
            input_name = f"activation_{layer_index}"
            reverse_lines.extend(
                [
                    f"    for (int column = 0; column < {dimension}; ++column) {{",
                    f"        const float value = {input_name}[column];",
                    "        const float sigmoid = 1.0f / (1.0f + expf(-value));",
                    f"        adjoint_{layer_index}[column] = "
                    f"adjoint_{layer_index + 1}[column]",
                    "            * (sigmoid + value * sigmoid * (1.0f - sigmoid));",
                    "    }",
                ]
            )
    dynamic_input_dimension = int(layout["dynamic_input_dimension"])
    reverse_lines.extend(
        [
            "    for (int column = 0; column < "
            f"{dynamic_input_dimension}; ++column) {{",
            "        radial_adjoint[column] = adjoint_0[column];",
            "    }",
        ]
    )
    reverse = f"""{function_qualifier} void {prefix}_radial_reverse_{interaction_index}(
    const float* radial,
    const float* source_first_layer,
    const float* target_first_layer,
    const float* parameters,
    const float* output_adjoint,
    float* radial_adjoint) noexcept
{{
{unused}
{chr(10).join(reverse_lines)}
}}"""
    if bounded_reverse:
        reverse = (
            _render_bounded_conditioner_reverse(
                interaction_index, network_name, layout, function_qualifier
            )
            or reverse
        )
    return forward + "\n\n" + reverse


def render_execution_mh1_conditioner_cpp_helpers(
    contract: dict[str, Any],
    *,
    function_qualifier: str = "__device__ __forceinline__",
    bounded_reverse: bool = False,
) -> str:
    """Render static per-edge conditioner forward and radial-reverse helpers."""

    if not isinstance(function_qualifier, str) or not function_qualifier.strip():
        raise ValueError("conditioner function qualifier must be a non-empty string")
    metadata = execution_mh1_conditioner_layout_metadata(contract)
    helpers = []
    for interaction in metadata["interactions"]:
        index = int(interaction["index"])
        helpers.append(
            _render_conditioner_network_helpers(
                index,
                "conditioned_prefix",
                interaction["conditioned_prefix"],
                function_qualifier.strip(),
                bounded_reverse=bounded_reverse,
            )
        )
        helpers.append(
            _render_conditioner_network_helpers(
                index,
                "density",
                interaction["density"],
                function_qualifier.strip(),
                bounded_reverse=bounded_reverse,
            )
        )
    return "\n\n".join(helpers)


def _render_conditioning_owners(
    interaction_layout: dict[str, Any],
    *,
    backend: str,
    function_qualifier: str,
) -> str:
    """Render graph-wide conditioner owners for one plugin interaction."""

    index = int(interaction_layout["index"])
    conditioned = interaction_layout["conditioned_prefix"]
    density = interaction_layout["density"]
    radial_dimension = int(conditioned["dynamic_input_dimension"])
    phi_dimension = int(conditioned["output_dimension"])
    convolution_contribution_dimension = int(
        conditioned["first_layer_contribution_dimension"]
    )
    density_contribution_dimension = int(density["first_layer_contribution_dimension"])
    density_output_dimension = int(density["output_dimension"])
    if int(density["dynamic_input_dimension"]) != radial_dimension:
        raise ValueError(
            f"MH1 interaction {index} conditioner radial dimensions disagree"
        )
    if density_output_dimension != 1:
        raise ValueError(
            f"MH1 interaction {index} density program must produce one scalar"
        )
    if backend not in ("Host", "Cuda"):
        raise ValueError("conditioner owner backend must be Host or Cuda")

    constant_prefix = f"SYMMETRIX_JIT_MH1_{backend.upper()}"
    args_prefix = f"SymmetrixJitMH1{backend}Conditioning"
    if convolution_contribution_dimension:
        convolution_source = (
            "args->convolution_source_contributions + "
            "static_cast<std::int64_t>(source_type)"
            f" * {convolution_contribution_dimension}"
        )
        convolution_target = (
            "args->convolution_target_contributions + "
            "static_cast<std::int64_t>(target_type)"
            f" * {convolution_contribution_dimension}"
        )
    else:
        convolution_source = "nullptr"
        convolution_target = "nullptr"
    if density_contribution_dimension:
        density_source = (
            "args->density_source_contributions + "
            "static_cast<std::int64_t>(source_type)"
            f" * {density_contribution_dimension}"
        )
        density_target = (
            "args->density_target_contributions + "
            "static_cast<std::int64_t>(target_type)"
            f" * {density_contribution_dimension}"
        )
    else:
        density_source = "nullptr"
        density_target = "nullptr"

    cuda_forward_edge = ""
    if backend == "Cuda":
        cuda_forward_edge = f"""{function_qualifier} float conditioning_forward_edge_{index}(
    const {args_prefix}ForwardArgsV3* args, std::int32_t edge) noexcept
{{
    const bool apply_cutoff =
        (args->flags & {constant_prefix}_APPLY_EDGE_CUTOFF_V3) != 0;
    const std::int32_t source_type = args->source_types[edge];
    const std::int32_t target = args->target_indices[edge];
    const std::int32_t target_type = args->node_types[target];
    const float* radial =
        args->radial + static_cast<std::int64_t>(edge) * {radial_dimension};
    const float* convolution_source = {convolution_source};
    const float* convolution_target = {convolution_target};
    const float* density_source = {density_source};
    const float* density_target = {density_target};
    execution_mh1_conditioned_prefix_forward_{index}(
        radial, convolution_source, convolution_target,
        args->convolution_parameters,
        args->edge_phi + static_cast<std::int64_t>(edge) * {phi_dimension});
    float density_output[{density_output_dimension}];
    execution_mh1_density_forward_{index}(
        radial, density_source, density_target,
        args->density_parameters, density_output);
    const float raw = density_output[0];
    const float density_base = tanhf(raw * raw);
    return density_base * (apply_cutoff
        ? args->edge_cutoff_scale[edge] : 1.0f);
}}

"""

    return f"""{cuda_forward_edge}{function_qualifier} void conditioning_forward_{index}(
    const {args_prefix}ForwardArgsV3* args,
    std::int32_t receiver_index) noexcept
{{
    const bool apply_cutoff =
        (args->flags & {constant_prefix}_APPLY_EDGE_CUTOFF_V3) != 0;
    const std::int32_t receiver = args->active_receivers[receiver_index];
    float density_sum = 0.0f;
    for (std::int64_t edge = args->receiver_offsets[receiver];
         edge < args->receiver_offsets[receiver + 1]; ++edge) {{
        const std::int32_t source_type = args->source_types[edge];
        const std::int32_t target = args->target_indices[edge];
        const std::int32_t target_type = args->node_types[target];
        const float* radial = args->radial + edge * {radial_dimension};
        const float* convolution_source = {convolution_source};
        const float* convolution_target = {convolution_target};
        const float* density_source = {density_source};
        const float* density_target = {density_target};
        execution_mh1_conditioned_prefix_forward_{index}(
            radial, convolution_source, convolution_target,
            args->convolution_parameters,
            args->edge_phi + edge * {phi_dimension});
        float density_output[{density_output_dimension}];
        execution_mh1_density_forward_{index}(
            radial, density_source, density_target,
            args->density_parameters, density_output);
        const float raw = density_output[0];
        const float density_base = tanhf(raw * raw);
        density_sum += density_base * (apply_cutoff
            ? args->edge_cutoff_scale[edge] : 1.0f);
    }}
    args->node_density[receiver] += density_sum;
}}

{function_qualifier} void conditioning_reverse_prefix_{index}(
    const {args_prefix}ReverseArgsV3* args,
    std::int32_t edge) noexcept
{{
    const std::int32_t source_type = args->source_types[edge];
    const std::int32_t target = args->target_indices[edge];
    const std::int32_t target_type = args->node_types[target];
    const float* radial =
        args->radial + static_cast<std::int64_t>(edge) * {radial_dimension};
    const float* convolution_source = {convolution_source};
    const float* convolution_target = {convolution_target};
    float convolution_radial_adjoint[{radial_dimension}];
    execution_mh1_conditioned_prefix_radial_reverse_{index}(
        radial, convolution_source, convolution_target,
        args->convolution_parameters,
        args->edge_phi_adjoint
            + static_cast<std::int64_t>(edge) * {phi_dimension},
        convolution_radial_adjoint);
    for (std::int32_t radial_index = 0;
         radial_index < {radial_dimension}; ++radial_index) {{
        args->radial_adjoint[
            static_cast<std::int64_t>(edge) * {radial_dimension}
                + radial_index] += convolution_radial_adjoint[radial_index];
    }}
}}

{function_qualifier} void conditioning_reverse_density_{index}(
    const {args_prefix}ReverseArgsV3* args,
    std::int32_t edge) noexcept
{{
    const bool apply_cutoff =
        (args->flags & {constant_prefix}_APPLY_EDGE_CUTOFF_V3) != 0;
    const std::int32_t source_type = args->source_types[edge];
    const std::int32_t target = args->target_indices[edge];
    const std::int32_t target_type = args->node_types[target];
    const float* radial =
        args->radial + static_cast<std::int64_t>(edge) * {radial_dimension};
    const float* density_source = {density_source};
    const float* density_target = {density_target};
    float density_output[{density_output_dimension}];
    execution_mh1_density_forward_{index}(
        radial, density_source, density_target,
        args->density_parameters, density_output);
    const float raw = density_output[0];
    const float density_base = tanhf(raw * raw);
    const float node_adjoint = args->node_density_adjoint[target];
    float density_output_adjoint[{density_output_dimension}];
    density_output_adjoint[0] =
        node_adjoint * (1.0f - density_base * density_base) * 2.0f * raw;
    if (apply_cutoff) {{
        args->edge_cutoff_scale_adjoint[edge] += node_adjoint * density_base;
        density_output_adjoint[0] *= args->edge_cutoff_scale[edge];
    }}
    float density_radial_adjoint[{radial_dimension}];
    execution_mh1_density_radial_reverse_{index}(
        radial, density_source, density_target,
        args->density_parameters, density_output_adjoint,
        density_radial_adjoint);
    for (std::int32_t radial_index = 0;
         radial_index < {radial_dimension}; ++radial_index) {{
        args->radial_adjoint[
            static_cast<std::int64_t>(edge) * {radial_dimension}
                + radial_index] += density_radial_adjoint[radial_index];
    }}
}}

{function_qualifier} void conditioning_reverse_{index}(
    const {args_prefix}ReverseArgsV3* args,
    std::int32_t edge) noexcept
{{
    conditioning_reverse_prefix_{index}(args, edge);
    conditioning_reverse_density_{index}(args, edge);
}}"""


def _render_cuda_conditioning_kernels(index: int, *, symbol_prefix: str = "") -> str:
    linkage = 'extern "C" ' if symbol_prefix else ""
    forward_name = f"{symbol_prefix}conditioning_forward_kernel_{index}"
    reverse_name = f"{symbol_prefix}conditioning_reverse_kernel_{index}"
    forward_kernel = f"""{linkage}__global__ void {forward_name}(
    SymmetrixJitMH1CudaConditioningForwardArgsV3 args)
{{
    __shared__ float density_reduction[128];
    for (std::int64_t receiver_index = blockIdx.x;
         receiver_index < args.active_receiver_count;
         receiver_index += gridDim.x) {{
        const std::int32_t receiver =
            args.active_receivers[receiver_index];
        float density_sum = 0.0f;
        for (std::int64_t edge = args.receiver_offsets[receiver] + threadIdx.x;
             edge < args.receiver_offsets[receiver + 1]; edge += blockDim.x)
            density_sum += conditioning_forward_edge_{index}(
                &args, static_cast<std::int32_t>(edge));
        density_reduction[threadIdx.x] = density_sum;
        __syncthreads();
        for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {{
            if (threadIdx.x < offset)
                density_reduction[threadIdx.x] +=
                    density_reduction[threadIdx.x + offset];
            __syncthreads();
        }}
        if (threadIdx.x == 0)
            args.node_density[receiver] += density_reduction[0];
        __syncthreads();
    }}
}}"""
    if not symbol_prefix:
        return f"""{forward_kernel}

{linkage}__global__ void {reverse_name}(
    SymmetrixJitMH1CudaConditioningReverseArgsV3 args)
{{
    const std::int64_t stride =
        static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    for (std::int64_t edge =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         edge < args.num_edges; edge += stride) {{
        conditioning_reverse_{index}(&args, static_cast<std::int32_t>(edge));
    }}
}}"""
    reverse_kernels = []
    for role in ("prefix", "density"):
        role_name = f"{symbol_prefix}conditioning_reverse_{role}_kernel_{index}"
        reverse_kernels.append(
            f"""{linkage}__global__ void {role_name}(
    SymmetrixJitMH1CudaConditioningReverseArgsV3 args)
{{
    const std::int64_t stride =
        static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    for (std::int64_t edge =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         edge < args.num_edges; edge += stride) {{
        conditioning_reverse_{role}_{index}(
            &args, static_cast<std::int32_t>(edge));
    }}
}}"""
        )
    return "\n\n".join((forward_kernel, *reverse_kernels))


def _render_cuda_conditioning_launchers(
    interaction_layout: dict[str, Any], metadata: dict[str, Any]
) -> str:
    index = int(interaction_layout["index"])
    conditioned = interaction_layout["conditioned_prefix"]
    density = interaction_layout["density"]
    convolution_contribution_dimension = int(
        conditioned["first_layer_contribution_dimension"]
    )
    density_contribution_dimension = int(density["first_layer_contribution_dimension"])
    convolution_parameter_count = int(conditioned["parameter_count"])
    density_parameter_count = int(density["parameter_count"])
    forward_threads = int(metadata["edge_threads_per_block"])
    convolution_forward_pointers = ""
    convolution_reverse_pointers = ""
    if convolution_contribution_dimension:
        convolution_forward_pointers += """
        || args->convolution_source_contributions == nullptr
        || args->convolution_target_contributions == nullptr"""
        convolution_reverse_pointers += """
        || args->convolution_source_contributions == nullptr
        || args->convolution_target_contributions == nullptr"""
    if convolution_parameter_count:
        convolution_forward_pointers += (
            "\n        || args->convolution_parameters == nullptr"
        )
        convolution_reverse_pointers += (
            "\n        || args->convolution_parameters == nullptr"
        )
    density_forward_pointers = ""
    density_reverse_pointers = ""
    if density_contribution_dimension:
        density_forward_pointers += """
        || args->density_source_contributions == nullptr
        || args->density_target_contributions == nullptr"""
        density_reverse_pointers += """
        || args->density_source_contributions == nullptr
        || args->density_target_contributions == nullptr"""
    if density_parameter_count:
        density_forward_pointers += "\n        || args->density_parameters == nullptr"
        density_reverse_pointers += "\n        || args->density_parameters == nullptr"

    return f"""std::int32_t conditioning_forward_launch_{index}(
    const SymmetrixJitMH1CudaConditioningForwardArgsV3* args,
    void* stream,
    std::int32_t persistent_blocks)
{{
    constexpr std::uint32_t known_flags =
        SYMMETRIX_JIT_MH1_CUDA_APPLY_EDGE_CUTOFF_V3;
    if (args == nullptr
        || args->struct_size
            < sizeof(SymmetrixJitMH1CudaConditioningForwardArgsV3)
        || args->interaction != {index}u || (args->flags & ~known_flags) != 0u
        || args->num_nodes < 0 || args->num_nodes > INT32_MAX
        || args->num_edges < 0 || args->num_edges > INT32_MAX
        || args->active_receiver_count > args->num_nodes
        || args->active_receiver_count > args->num_edges
        || (args->num_edges > 0 && args->num_nodes == 0)
        || (args->num_edges > 0 && args->active_receiver_count == 0)
        || persistent_blocks <= 0)
        return static_cast<std::int32_t>(cudaErrorInvalidValue);
    if (args->num_edges == 0)
        return static_cast<std::int32_t>(cudaSuccess);
    const bool apply_cutoff =
        (args->flags & SYMMETRIX_JIT_MH1_CUDA_APPLY_EDGE_CUTOFF_V3) != 0;
    if (args->source_types == nullptr || args->target_indices == nullptr
        || args->node_types == nullptr || args->active_receivers == nullptr
        || args->receiver_offsets == nullptr || args->radial == nullptr
        || (apply_cutoff && args->edge_cutoff_scale == nullptr){convolution_forward_pointers}{density_forward_pointers}
        || args->edge_phi == nullptr || args->node_density == nullptr)
        return static_cast<std::int32_t>(cudaErrorInvalidDevicePointer);
    constexpr std::int32_t threads = {forward_threads};
    static_assert(threads > 0,
        "MH-1 CUDA conditioner block size must be positive");
    const std::int32_t blocks = static_cast<std::int32_t>(
        args->active_receiver_count < persistent_blocks
            ? args->active_receiver_count : persistent_blocks);
    const auto launch_args = *args;
    conditioning_forward_kernel_{index}<<<blocks, threads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(launch_args);
    return static_cast<std::int32_t>(cudaPeekAtLastError());
}}

std::int32_t conditioning_reverse_launch_{index}(
    const SymmetrixJitMH1CudaConditioningReverseArgsV3* args,
    void* stream,
    std::int32_t persistent_blocks)
{{
    constexpr std::uint32_t known_flags =
        SYMMETRIX_JIT_MH1_CUDA_APPLY_EDGE_CUTOFF_V3;
    if (args == nullptr
        || args->struct_size
            < sizeof(SymmetrixJitMH1CudaConditioningReverseArgsV3)
        || args->interaction != {index}u || (args->flags & ~known_flags) != 0u
        || args->reserved != 0u
        || args->num_nodes < 0 || args->num_nodes > INT32_MAX
        || args->num_edges < 0 || args->num_edges > INT32_MAX
        || (args->num_edges > 0 && args->num_nodes == 0)
        || persistent_blocks <= 0)
        return static_cast<std::int32_t>(cudaErrorInvalidValue);
    if (args->num_edges == 0)
        return static_cast<std::int32_t>(cudaSuccess);
    const bool apply_cutoff =
        (args->flags & SYMMETRIX_JIT_MH1_CUDA_APPLY_EDGE_CUTOFF_V3) != 0;
    if (args->source_types == nullptr || args->target_indices == nullptr
        || args->node_types == nullptr || args->radial == nullptr
        || (apply_cutoff && args->edge_cutoff_scale == nullptr){convolution_reverse_pointers}{density_reverse_pointers}
        || args->edge_phi_adjoint == nullptr
        || args->node_density_adjoint == nullptr
        || args->radial_adjoint == nullptr
        || args->edge_cutoff_scale_adjoint == nullptr)
        return static_cast<std::int32_t>(cudaErrorInvalidDevicePointer);
    constexpr std::int32_t threads = {forward_threads};
    static_assert(threads > 0,
        "MH-1 CUDA conditioner block size must be positive");
    const std::int32_t blocks = launch_blocks(
        args->num_edges, threads, persistent_blocks);
    const auto launch_args = *args;
    conditioning_reverse_kernel_{index}<<<blocks, threads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(launch_args);
    return static_cast<std::int32_t>(cudaPeekAtLastError());
}}"""


def jit_mh1_host_plugin_metadata(contract: dict[str, Any]) -> dict[str, Any]:
    """Return cache and native-loader metadata for a host specialization."""

    normalized = normalize_execution_mh1_contract(contract)
    return {
        "abi": MH1_HOST_PLUGIN_ABI,
        "abi_version": MH1_HOST_PLUGIN_ABI_VERSION,
        "artifact_id": _artifact_id(normalized),
        "generation_fingerprint": normalized["generation_fingerprint"],
        "semantic_fingerprint": normalized["semantic_fingerprint"],
        "structure_fingerprint": normalized["structure_fingerprint"],
        "conditioner_layout": execution_mh1_conditioner_layout_metadata(normalized),
        "interactions": [
            _interaction_extents(interaction)
            for interaction in normalized["interactions"]
        ],
    }


def jit_mh1_cuda_plugin_metadata(
    contract: dict[str, Any],
    compute_capability: int,
    *,
    forward_policy: str | dict[str, Any] = "auto",
    source_policy: str | dict[str, Any] = "auto",
    edge_policy: str | dict[str, Any] = "auto",
    _edge_reverse_schedule: str = _MH1_CUDA_EDGE_REVERSE_SCHEDULE,
) -> dict[str, Any]:
    """Return cache and native-loader metadata for a CUDA specialization."""

    program, _, schedule = _mh1_cuda_program_target_schedule(
        contract,
        compute_capability,
        abi_version=3,
        forward_policy=forward_policy,
        source_policy=source_policy,
        edge_policy=edge_policy,
        edge_reverse_schedule=_edge_reverse_schedule,
    )
    normalized = program.contract
    resolved_forward_policy = json.loads(schedule.forward_policy_json)
    resolved_source_policy = json.loads(schedule.source_policy_json)
    resolved_edge_policy = json.loads(schedule.edge_policy_json)
    metadata = jit_mh1_host_plugin_metadata(normalized)
    conditioner_reverse_schedule = _resolve_conditioner_reverse_schedule(
        metadata["conditioner_layout"]
    )
    metadata.update(
        {
            "abi": MH1_CUDA_PLUGIN_ABI,
            "abi_version": MH1_CUDA_PLUGIN_ABI_VERSION,
            "target_compute_capability": compute_capability,
            "forward_threads_per_block": schedule.forward_threads,
            "source_threads_per_block": schedule.source_threads,
            "edge_threads_per_block": schedule.edge_threads,
            "forward_policy": resolved_forward_policy,
            "forward_policy_id": resolved_forward_policy["policy_id"],
            "source_policy": resolved_source_policy,
            "source_policy_id": resolved_source_policy["policy_id"],
            "edge_policy": resolved_edge_policy,
            "edge_policy_id": resolved_edge_policy["policy_id"],
            "edge_reverse_schedule": _edge_reverse_schedule,
            "conditioner_reverse_schedule": conditioner_reverse_schedule,
            "conditioner_reverse_schedule_id": conditioner_reverse_schedule[
                "schedule_id"
            ],
        }
    )
    return metadata


def _weight_expression(
    path: dict[str, Any],
    extents: dict[str, int],
    *,
    phi_major: bool = False,
) -> str:
    weight_index = f"{int(path['weight_offset'])} + channel"
    phi_dimension = extents["phi_dimension"]
    if phi_major:
        linear_weight_index = f"phi * {extents['weight_size']} + ({weight_index})"
    else:
        linear_weight_index = f"({weight_index}) * {phi_dimension} + phi"
    return f"""float weight = has_bias ? args->linear_bias[{weight_index}] : 0.0f;
            if (has_fixed) {{
                weight += args->edge_linear_contribution[
                    local_edge * {extents['weight_size']} + {weight_index}];
            }}
            for (std::int32_t phi = 0; phi < {phi_dimension}; ++phi) {{
                weight += args->linear_weight[
                    {linear_weight_index}]
                    * args->edge_phi[local_edge * {phi_dimension} + phi];
            }}"""


def _output_variables(interaction: dict[str, Any]) -> dict[tuple[int, int], str]:
    variables: dict[tuple[int, int], str] = {}
    for path in interaction["paths"]:
        output = path["output"]
        for component in range(int(output["components"])):
            key = (int(output["offset"]), component)
            variables.setdefault(key, f"value_{len(variables)}")
    return variables


def _input_variables(interaction: dict[str, Any]) -> dict[tuple[int, int], str]:
    variables: dict[tuple[int, int], str] = {}
    for path in interaction["paths"]:
        source = path["input_1"]
        for component in range(int(source["components"])):
            key = (int(source["offset"]), component)
            variables.setdefault(key, f"value_{len(variables)}")
    return variables


def _interaction_output_mask_is_identity(interaction: dict[str, Any]) -> bool:
    output_dimension = int(interaction["dimensions"]["output"])
    covered = [False] * output_dimension
    for path in interaction["paths"]:
        output = path["output"]
        offset = int(output["offset"])
        extent = int(output["components"]) * int(output["multiplicity"])
        covered[offset : offset + extent] = [True] * extent
    return all(covered)


def _output_mask_factor(
    interaction: dict[str, Any],
    output_index: str,
    *,
    identity: bool | None = None,
) -> str:
    if identity is None:
        identity = _interaction_output_mask_is_identity(interaction)
    if identity:
        return ""
    return f" * args->output_mask[{output_index}]"


def _render_host_forward(
    interaction: dict[str, Any],
    *,
    channel_owner: bool = False,
    phi_major_linear_weight: bool = False,
    output_mask_is_identity: bool | None = None,
) -> str:
    index = int(interaction["index"])
    extents = _interaction_extents(interaction)
    variables = _output_variables(interaction)
    declarations = "\n        ".join(
        f"float {name} = 0.0f;" for name in variables.values()
    )
    path_blocks = []
    for path in interaction["paths"]:
        source = path["input_1"]
        edge = path["input_2"]
        output = path["output"]
        statements = [
            _weight_expression(path, extents, phi_major=phi_major_linear_weight)
        ]
        statements.append("if (apply_cutoff) weight *= args->edge_cutoff_scale[edge];")
        for term in path["sparse_wigner"]["terms"]:
            output_component = int(term["c"])
            output_index = _ir_mul_index(output, output_component)
            source_index = _ir_mul_index(source, int(term["a"]))
            harmonic_index = int(edge["offset"]) + int(term["b"])
            value = variables[(int(output["offset"]), output_component)]
            statements.append(
                f"{value} += {_cpp_float(path['path_weight'])} * weight"
                f" * {_cpp_float(term['coefficient'])}"
                f" * args->source_node_values[source * {extents['input_1_dimension']}"
                f" + {source_index}]"
                f" * args->edge_input_2[local_edge * {extents['input_2_dimension']}"
                f" + {harmonic_index}]"
                f"{_output_mask_factor(interaction, output_index, identity=output_mask_is_identity)};"
            )
        path_blocks.append(
            "{\n            " + "\n            ".join(statements) + "\n            }"
        )
    stores = []
    for (offset, component), value in variables.items():
        multiplicity = next(
            int(path["output"]["multiplicity"])
            for path in interaction["paths"]
            if int(path["output"]["offset"]) == offset
        )
        stores.append(
            f"args->node_messages[receiver * {extents['output_dimension']}"
            f" + {offset} + {component} * {multiplicity} + channel] += {value};"
        )
    channel_argument = ",\n    std::int32_t channel" if channel_owner else ""
    channel_body = f"""{declarations}
        for (std::int64_t edge = begin; edge < end; ++edge) {{
            const std::int64_t local_edge = edge - args->first_edge;
            const std::int32_t source = args->source_indices[edge];
            {chr(10).join(path_blocks)}
        }}
        {chr(10).join(stores)}"""
    if not channel_owner:
        channel_body = f"""for (std::int32_t channel = 0; channel < {extents['multiplicity']}; ++channel) {{
        {channel_body}
    }}"""
    return f"""static void forward_{index}(
    const SymmetrixJitMH1HostForwardArgsV3* args,
    std::int32_t receiver{channel_argument}) noexcept
{{
    const std::int64_t block_end = args->first_edge + args->samples;
    const std::int64_t receiver_begin = args->receiver_offsets[receiver];
    const std::int64_t receiver_end = args->receiver_offsets[receiver + 1];
    const std::int64_t begin = receiver_begin > args->first_edge
        ? receiver_begin : args->first_edge;
    const std::int64_t end = receiver_end < block_end
        ? receiver_end : block_end;
    if (begin >= end) return;
    const bool apply_cutoff =
        (args->flags & SYMMETRIX_JIT_MH1_HOST_APPLY_EDGE_CUTOFF_V3) != 0;
    const bool has_bias =
        (args->flags & SYMMETRIX_JIT_MH1_HOST_HAS_LINEAR_BIAS_V3) != 0;
    const bool has_fixed =
        (args->flags & SYMMETRIX_JIT_MH1_HOST_HAS_FIXED_CONTRIBUTION_V3) != 0;
    {channel_body}
}}"""


def _render_host_forward_channel_tiled(interaction: dict[str, Any]) -> str:
    index = int(interaction["index"])
    extents = _interaction_extents(interaction)
    output_mask_is_identity = _interaction_output_mask_is_identity(interaction)
    path_blocks = []
    for path in interaction["paths"]:
        source = path["input_1"]
        edge = path["input_2"]
        output = path["output"]
        output_offset = int(output["offset"])
        output_multiplicity = int(output["multiplicity"])
        components = sorted({int(term["c"]) for term in path["sparse_wigner"]["terms"]})
        component_rows = {component: row for row, component in enumerate(components)}
        term_blocks = []
        for term_ordinal, term in enumerate(path["sparse_wigner"]["terms"]):
            output_component = int(term["c"])
            output_index = _ir_mul_index(output, output_component)
            source_index = _ir_mul_index(source, int(term["a"]))
            harmonic_index = int(edge["offset"]) + int(term["b"])
            mask = (
                ""
                if output_mask_is_identity
                else f" * args->output_mask[{output_index}]"
            )
            term_blocks.append(
                f"""const float factor_{term_ordinal} = {_cpp_float(path['path_weight'])}
                    * {_cpp_float(term['coefficient'])}
                    * args->edge_input_2[local_edge * {extents['input_2_dimension']}
                        + {harmonic_index}];
                #pragma omp simd
                for (std::int32_t lane = 0; lane < active_channels; ++lane) {{
                    const std::int32_t channel = channel_begin + lane;
                    values[{component_rows[output_component]}][lane] += factor_{term_ordinal}
                        * weights[lane]
                        * args->source_node_values[
                            source * {extents['input_1_dimension']} + {source_index}]
                        {mask};
                }}"""
            )
        stores = []
        for component in components:
            output_index = (
                f"{output_offset} + {component} * {output_multiplicity} + channel"
            )
            stores.append(
                f"""#pragma omp simd
            for (std::int32_t lane = 0; lane < active_channels; ++lane) {{
                const std::int32_t channel = channel_begin + lane;
                args->node_messages[
                    receiver * {extents['output_dimension']} + {output_index}]
                    += values[{component_rows[component]}][lane];
            }}"""
            )
        weight_offset = int(path["weight_offset"])
        path_blocks.append(
            f"""for (std::int32_t channel_begin = 0;
         channel_begin < {extents['multiplicity']}; channel_begin += channel_tile) {{
        const std::int32_t active_channels =
            channel_begin + channel_tile < {extents['multiplicity']}
                ? channel_tile : {extents['multiplicity']} - channel_begin;
        float values[{len(components)}][channel_tile] = {{}};
        for (std::int64_t edge = begin; edge < end; ++edge) {{
            const std::int64_t local_edge = edge - args->first_edge;
            const std::int32_t source = args->source_indices[edge];
            float weights[channel_tile];
            #pragma omp simd
            for (std::int32_t lane = 0; lane < active_channels; ++lane) {{
                const std::int32_t weight_index =
                    {weight_offset} + channel_begin + lane;
                weights[lane] = has_bias ? args->linear_bias[weight_index] : 0.0f;
                if (has_fixed)
                    weights[lane] += args->edge_linear_contribution[
                        local_edge * {extents['weight_size']} + weight_index];
            }}
            for (std::int32_t phi = 0; phi < {extents['phi_dimension']}; ++phi) {{
                const float phi_value =
                    args->edge_phi[local_edge * {extents['phi_dimension']} + phi];
                #pragma omp simd
                for (std::int32_t lane = 0; lane < active_channels; ++lane)
                    weights[lane] += args->linear_weight[
                        phi * {extents['weight_size']}
                            + {weight_offset} + channel_begin + lane] * phi_value;
            }}
            if (apply_cutoff) {{
                const float cutoff = args->edge_cutoff_scale[edge];
                #pragma omp simd
                for (std::int32_t lane = 0; lane < active_channels; ++lane)
                    weights[lane] *= cutoff;
            }}
            {chr(10).join(term_blocks)}
        }}
        {chr(10).join(stores)}
    }}"""
        )
    return f"""static void forward_{index}(
    const SymmetrixJitMH1HostForwardArgsV3* args,
    std::int32_t receiver) noexcept
{{
    const std::int64_t block_end = args->first_edge + args->samples;
    const std::int64_t receiver_begin = args->receiver_offsets[receiver];
    const std::int64_t receiver_end = args->receiver_offsets[receiver + 1];
    const std::int64_t begin = receiver_begin > args->first_edge
        ? receiver_begin : args->first_edge;
    const std::int64_t end = receiver_end < block_end
        ? receiver_end : block_end;
    if (begin >= end) return;
    const bool apply_cutoff =
        (args->flags & SYMMETRIX_JIT_MH1_HOST_APPLY_EDGE_CUTOFF_V3) != 0;
    const bool has_bias =
        (args->flags & SYMMETRIX_JIT_MH1_HOST_HAS_LINEAR_BIAS_V3) != 0;
    const bool has_fixed =
        (args->flags & SYMMETRIX_JIT_MH1_HOST_HAS_FIXED_CONTRIBUTION_V3) != 0;
#if defined(__AVX512F__)
    constexpr std::int32_t channel_tile = 16;
#else
    constexpr std::int32_t channel_tile = 8;
#endif
    {chr(10).join(path_blocks)}
}}"""


def _render_host_source_reverse(
    interaction: dict[str, Any],
    *,
    channel_owner: bool = False,
    phi_major_linear_weight: bool = False,
    output_mask_is_identity: bool | None = None,
    path_weight_tile_size: int = 1,
) -> str:
    if path_weight_tile_size <= 0:
        raise ValueError("source-reverse path weight tile size must be positive")
    index = int(interaction["index"])
    extents = _interaction_extents(interaction)
    variables = _input_variables(interaction)
    declarations = "\n        ".join(
        f"float {name} = 0.0f;" for name in variables.values()
    )

    def contraction_statements(path: dict[str, Any], weight: str) -> list[str]:
        source_block = path["input_1"]
        edge_block = path["input_2"]
        output = path["output"]
        statements = []
        for term in path["sparse_wigner"]["terms"]:
            source_component = int(term["a"])
            output_index = _ir_mul_index(output, int(term["c"]))
            harmonic_index = int(edge_block["offset"]) + int(term["b"])
            value = variables[(int(source_block["offset"]), source_component)]
            statements.append(
                f"{value} += {_cpp_float(path['path_weight'])} * {weight}"
                f" * {_cpp_float(term['coefficient'])}"
                f" * args->edge_input_2[local_edge * {extents['input_2_dimension']}"
                f" + {harmonic_index}]"
                f" * args->target_node_output_adjoint["
                f"target * {extents['output_dimension']} + {output_index}]"
                f"{_output_mask_factor(interaction, output_index, identity=output_mask_is_identity)};"
            )
        return statements

    paths = interaction["paths"]
    path_blocks = []
    if path_weight_tile_size == 1:
        for path in paths:
            statements = [
                _weight_expression(path, extents, phi_major=phi_major_linear_weight),
                "if (apply_cutoff) weight *= args->edge_cutoff_scale[edge];",
                *contraction_statements(path, "weight"),
            ]
            path_blocks.append(
                "{\n            "
                + "\n            ".join(statements)
                + "\n            }"
            )
    else:
        for tile_first in range(0, len(paths), path_weight_tile_size):
            tile = paths[tile_first : tile_first + path_weight_tile_size]
            statements = []
            for path_index, path in enumerate(tile):
                weight_index = f"{int(path['weight_offset'])} + channel"
                statements.append(
                    f"float weight_{path_index} = has_bias"
                    f" ? args->linear_bias[{weight_index}] : 0.0f;"
                )
                statements.append(
                    f"if (has_fixed) weight_{path_index} += "
                    f"args->edge_linear_contribution[local_edge"
                    f" * {extents['weight_size']} + {weight_index}];"
                )
            phi_updates = []
            for path_index, path in enumerate(tile):
                weight_index = f"{int(path['weight_offset'])} + channel"
                if phi_major_linear_weight:
                    linear_weight_index = (
                        f"phi * {extents['weight_size']} + ({weight_index})"
                    )
                else:
                    linear_weight_index = (
                        f"({weight_index}) * {extents['phi_dimension']} + phi"
                    )
                phi_updates.append(
                    f"weight_{path_index} += args->linear_weight["
                    f"{linear_weight_index}] * phi_value;"
                )
            statements.append(
                f"for (std::int32_t phi = 0; phi < {extents['phi_dimension']}; ++phi) {{\n"
                f"                const float phi_value = args->edge_phi["
                f"local_edge * {extents['phi_dimension']} + phi];\n"
                f"                {chr(10).join(phi_updates)}\n"
                "            }"
            )
            for path_index, path in enumerate(tile):
                weight = f"weight_{path_index}"
                statements.append(
                    f"if (apply_cutoff) {weight} *= args->edge_cutoff_scale[edge];"
                )
                statements.extend(contraction_statements(path, weight))
            path_blocks.append(
                "{\n            "
                + "\n            ".join(statements)
                + "\n            }"
            )
    stores = []
    for (offset, component), value in variables.items():
        multiplicity = next(
            int(path["input_1"]["multiplicity"])
            for path in interaction["paths"]
            if int(path["input_1"]["offset"]) == offset
        )
        stores.append(
            f"args->source_node_input_adjoint[source * {extents['input_1_dimension']}"
            f" + {offset} + {component} * {multiplicity} + channel] += {value};"
        )
    channel_argument = ",\n    std::int32_t channel" if channel_owner else ""
    channel_body = f"""{declarations}
        const std::int32_t first_scheduled =
            args->source_edge_offsets[source_owner];
        const std::int32_t source = args->source_indices[
            args->source_edge_indices[first_scheduled]];
        for (std::int32_t scheduled = first_scheduled;
             scheduled < args->source_edge_offsets[source_owner + 1];
             ++scheduled) {{
            const std::int64_t edge = args->source_edge_indices[scheduled];
            const std::int32_t local_edge =
                static_cast<std::int32_t>(edge - args->first_edge);
            const std::int32_t target = args->target_indices[edge];
            {chr(10).join(path_blocks)}
        }}
        {chr(10).join(stores)}"""
    if not channel_owner:
        channel_body = f"""for (std::int32_t channel = 0; channel < {extents['multiplicity']}; ++channel) {{
        {channel_body}
    }}"""
    return f"""static void source_reverse_{index}(
    const SymmetrixJitMH1HostReverseArgsV3* args,
    std::int32_t source_owner{channel_argument}) noexcept
{{
    const bool apply_cutoff =
        (args->flags & SYMMETRIX_JIT_MH1_HOST_APPLY_EDGE_CUTOFF_V3) != 0;
    const bool has_bias =
        (args->flags & SYMMETRIX_JIT_MH1_HOST_HAS_LINEAR_BIAS_V3) != 0;
    const bool has_fixed =
        (args->flags & SYMMETRIX_JIT_MH1_HOST_HAS_FIXED_CONTRIBUTION_V3) != 0;
    {channel_body}
}}"""


def _render_host_spline_r_forward(
    interaction: dict[str, Any], *, packed_messages: bool
) -> str:
    index = int(interaction["index"])
    extents = _interaction_extents(interaction)
    path_blocks = []
    for path in interaction["paths"]:
        source = path["input_1"]
        edge = path["input_2"]
        output = path["output"]
        weight_function = f"{int(path['weight_offset'])} + channel"
        statements = [
            f"const float weight = spline_r_value(args, pair, interval, x, xx, xxx, {weight_function});"
        ]
        for term in path["sparse_wigner"]["terms"]:
            output_index = _ir_mul_index(output, int(term["c"]))
            source_index = _ir_mul_index(source, int(term["a"]))
            harmonic_index = int(edge["offset"]) + int(term["b"])
            statements.append(
                f"message[{output_index}] += {_cpp_float(path['path_weight'])}"
                f" * weight * {_cpp_float(term['coefficient'])}"
                f" * source_node_values[source * {extents['input_1_dimension']}"
                f" + {source_index}]"
                f" * edge_input_2[edge_index * {extents['input_2_dimension']}"
                f" + {harmonic_index}];"
            )
        path_blocks.append(
            f"for (std::int32_t channel = 0;\n"
            f"     channel < {extents['multiplicity']}; ++channel) {{\n"
            "                "
            + "\n                ".join(statements)
            + "\n            }"
        )
    output_blocks = {
        int(path["output"]["offset"]): path["output"] for path in interaction["paths"]
    }
    packed_stores = []
    for offset, block in sorted(output_blocks.items()):
        multiplicity = int(block["multiplicity"])
        width = int(block["components"])
        packed_stores.append(
            f"""for (std::int32_t channel = 0; channel < {multiplicity}; ++channel)
        for (std::int32_t component = 0; component < {width}; ++component)
            node_messages[
                args->num_nodes * {offset}
                    + static_cast<std::int64_t>(channel) * args->num_nodes * {width}
                    + receiver * {width} + component]
                += message[{offset} + component * {multiplicity} + channel];"""
        )
    direct_store = f"""for (std::int32_t component = 0;
         component < {extents['output_dimension']}; ++component)
        node_messages[receiver * {extents['output_dimension']} + component]
            += message[component];"""
    stores = "\n    ".join(packed_stores) if packed_messages else direct_store
    return f"""static void spline_r_forward_{index}(
    const SymmetrixJitMH1HostSplineRForwardArgsV5* args,
    std::int32_t receiver) noexcept
{{
    const float* const edge_input_2 =
        static_cast<const float*>(args->edge_input_2);
    const float* const source_node_values =
        static_cast<const float*>(args->source_node_values);
    float* const node_messages = static_cast<float*>(args->node_messages);
    float* const node_density = static_cast<float*>(args->node_density);
    float message[{extents['output_dimension']}] = {{0.0f}};
    float density = 0.0f;
    for (std::int64_t edge_index = args->receiver_offsets[receiver];
         edge_index < args->receiver_offsets[receiver + 1]; ++edge_index) {{
        const std::int32_t source = args->source_indices[edge_index];
        const std::int32_t pair = args->source_types[edge_index]
            * args->type_count + args->node_types[receiver];
        const double radius = args->distances[edge_index];
        std::int32_t interval = static_cast<std::int32_t>(
            std::floor((radius - args->spline_x0) / args->spline_h));
        double delta = radius - args->spline_x0 - args->spline_h * interval;
        if (interval < 0) {{
            interval = 0;
            delta = 0.0;
        }} else if (interval >= args->interval_count) {{
            interval = args->interval_count - 1;
            delta = args->spline_h;
        }}
        const float x = static_cast<float>(delta);
        const float xx = x * x;
        const float xxx = xx * x;
        density += spline_r_value(
            args, pair, interval, x, xx, xxx, {extents['weight_size']});
        {chr(10).join(path_blocks)}
    }}
    {stores}
    node_density[receiver] += density;
}}"""


def _render_host_spline_r_reverse(interaction: dict[str, Any]) -> str:
    index = int(interaction["index"])
    extents = _interaction_extents(interaction)
    path_blocks = []
    for path_index, path in enumerate(interaction["paths"]):
        source = path["input_1"]
        edge = path["input_2"]
        output = path["output"]
        output_offset = int(output["offset"])
        harmonic_offset = int(edge["offset"])
        weight_function = f"{int(path['weight_offset'])} + channel"
        terms = path["sparse_wigner"]["terms"]
        source_components = sorted({int(term["a"]) for term in terms})
        harmonic_components = sorted({int(term["b"]) for term in terms})
        statements = [
            f"const float weight = spline_r_value(args, pair, interval, x, xx, xxx, {weight_function});",
            f"const float weight_derivative = spline_r_derivative(args, pair, interval, x, xx, {weight_function});",
            "float weight_adjoint = 0.0f;",
            *(
                f"float source_contribution_{component} = 0.0f;"
                for component in source_components
            ),
            *(
                f"float harmonic_contribution_{component} = 0.0f;"
                for component in harmonic_components
            ),
        ]
        for term in terms:
            source_component = int(term["a"])
            harmonic_component = int(term["b"])
            output_index = _ir_mul_index(output, int(term["c"]))
            source_index = _ir_mul_index(source, source_component)
            harmonic_index = int(edge["offset"]) + harmonic_component
            statements.append(
                "{\n                "
                f"const float reverse_factor = {_cpp_float(path['path_weight'])}"
                f" * {_cpp_float(term['coefficient'])}"
                f" * target_node_output_adjoint[receiver"
                f" * {extents['output_dimension']} + {output_index}];\n                "
                f"const float source_value = source_node_values[source"
                f" * {extents['input_1_dimension']} + {source_index}];\n                "
                f"const float harmonic_value = edge_input_2[edge_index"
                f" * {extents['input_2_dimension']} + {harmonic_index}];\n                "
                "const float source_reverse = reverse_factor * harmonic_value;\n                "
                f"source_contribution_{source_component} += source_reverse;\n                "
                f"harmonic_contribution_{harmonic_component} += reverse_factor"
                " * source_value;\n                "
                "weight_adjoint += source_reverse * source_value;\n                }"
            )
        statements.extend(
            f"source_adjoint[{_ir_mul_index(source, component)}] += weight"
            f" * source_contribution_{component};"
            for component in source_components
        )
        statements.extend(
            f"harmonic_adjoint_{path_index}_{component} += weight"
            f" * harmonic_contribution_{component};"
            for component in harmonic_components
        )
        statements.append(
            f"radial_contribution_{path_index} += weight_derivative"
            " * weight_adjoint;"
        )
        reduction_variables = [
            f"radial_contribution_{path_index}",
            *(
                f"harmonic_adjoint_{path_index}_{component}"
                for component in harmonic_components
            ),
        ]
        reduction_declarations = "\n            ".join(
            f"float {name} = 0.0f;" for name in reduction_variables
        )
        reduction_stores = "\n            ".join(
            f"edge_adjoint[{int(edge['offset']) + component}]"
            f" += harmonic_adjoint_{path_index}_{component};"
            for component in harmonic_components
        )
        path_blocks.append(
            f"""for (std::int32_t edge_lane = 0;
             edge_lane < active_edges; ++edge_lane) {{
            const std::int64_t edge_index = edge_indices[edge_lane];
            const std::int32_t receiver = receivers[edge_lane];
            const std::int32_t pair = pairs[edge_lane];
            const std::int32_t interval = intervals[edge_lane];
            const float x = xs[edge_lane];
            const float xx = xxs[edge_lane];
            const float xxx = xxxs[edge_lane];
            float* const edge_adjoint = edge_adjoints[edge_lane];
            float& radial_adjoint = radial_adjoints[edge_lane];
            if (edge_lane + {_MH1_HOST_SPLINE_REVERSE_PREFETCH_DISTANCE}
                    < active_edges) {{
                const std::int32_t prefetch_lane = edge_lane
                    + {_MH1_HOST_SPLINE_REVERSE_PREFETCH_DISTANCE};
                prefetch_read(target_node_output_adjoint
                    + static_cast<std::int64_t>(receivers[prefetch_lane])
                        * {extents['output_dimension']} + {output_offset});
                prefetch_read(edge_input_2
                    + edge_indices[prefetch_lane]
                        * {extents['input_2_dimension']} + {harmonic_offset});
            }}
            {reduction_declarations}
#if defined(_OPENMP)
#pragma omp simd reduction(+:{','.join(reduction_variables)})
#endif
            for (std::int32_t channel = 0;
                 channel < {extents['multiplicity']}; ++channel) {{
                """
            + "\n                ".join(statements)
            + f"""
            }}
            {reduction_stores}
            radial_adjoint += radial_contribution_{path_index};
        }}"""
        )
    return f"""static void spline_r_source_reverse_{index}(
    const SymmetrixJitMH1HostSplineRReverseArgsV5* args,
    std::int32_t source_owner) noexcept
{{
    const float* const edge_input_2 =
        static_cast<const float*>(args->edge_input_2);
    const float* const source_node_values =
        static_cast<const float*>(args->source_node_values);
    const float* const target_node_output_adjoint =
        static_cast<const float*>(args->target_node_output_adjoint);
    const float* const node_density_adjoint =
        static_cast<const float*>(args->node_density_adjoint);
    float* const source_node_input_adjoint =
        static_cast<float*>(args->source_node_input_adjoint);
    float* const edge_input_2_adjoint =
        static_cast<float*>(args->edge_input_2_adjoint);
    float* const distance_adjoint =
        static_cast<float*>(args->distance_adjoint);
    const std::int32_t first_scheduled =
        args->source_edge_offsets[source_owner];
    const std::int32_t scheduled_end =
        args->source_edge_offsets[source_owner + 1];
    if (first_scheduled == scheduled_end) return;
    const std::int32_t source = args->source_indices[
        args->source_edge_indices[first_scheduled]];
    float source_adjoint[{extents['input_1_dimension']}] = {{0.0f}};
    constexpr std::int32_t edge_tile = {_MH1_HOST_SPLINE_REVERSE_EDGE_TILE};
    for (std::int32_t tile_first = first_scheduled;
         tile_first < scheduled_end; tile_first += edge_tile) {{
        const std::int32_t active_edges = tile_first + edge_tile < scheduled_end
            ? edge_tile : scheduled_end - tile_first;
        std::int64_t edge_indices[edge_tile];
        std::int32_t receivers[edge_tile];
        std::int32_t pairs[edge_tile];
        std::int32_t intervals[edge_tile];
        float xs[edge_tile];
        float xxs[edge_tile];
        float xxxs[edge_tile];
        float edge_adjoints[edge_tile][{extents['input_2_dimension']}] = {{}};
        float radial_adjoints[edge_tile];
        for (std::int32_t edge_lane = 0;
             edge_lane < active_edges; ++edge_lane) {{
            const std::int64_t edge_index =
                args->source_edge_indices[tile_first + edge_lane];
            const std::int32_t receiver = args->target_indices[edge_index];
            const std::int32_t pair = args->source_types[edge_index]
                * args->type_count + args->node_types[receiver];
            const double radius = args->distances[edge_index];
            std::int32_t interval = static_cast<std::int32_t>(
                std::floor((radius - args->spline_x0) / args->spline_h));
            double delta = radius - args->spline_x0 - args->spline_h * interval;
            if (interval < 0) {{
                interval = 0;
                delta = 0.0;
            }} else if (interval >= args->interval_count) {{
                interval = args->interval_count - 1;
                delta = args->spline_h;
            }}
            const float x = static_cast<float>(delta);
            edge_indices[edge_lane] = edge_index;
            receivers[edge_lane] = receiver;
            pairs[edge_lane] = pair;
            intervals[edge_lane] = interval;
            xs[edge_lane] = x;
            xxs[edge_lane] = x * x;
            xxxs[edge_lane] = xxs[edge_lane] * x;
            radial_adjoints[edge_lane] = node_density_adjoint[receiver]
                * spline_r_derivative(args, pair, interval, x,
                    xxs[edge_lane], {extents['weight_size']});
        }}
        {chr(10).join(path_blocks)}
        for (std::int32_t edge_lane = 0;
             edge_lane < active_edges; ++edge_lane) {{
            const std::int64_t edge_index = edge_indices[edge_lane];
            for (std::int32_t component = 0;
                 component < {extents['input_2_dimension']}; ++component)
                edge_input_2_adjoint[
                    edge_index * {extents['input_2_dimension']} + component]
                    += edge_adjoints[edge_lane][component];
            distance_adjoint[edge_index] += radial_adjoints[edge_lane];
        }}
    }}
    for (std::int32_t component = 0;
         component < {extents['input_1_dimension']}; ++component)
        source_node_input_adjoint[
            static_cast<std::int64_t>(source)
                * {extents['input_1_dimension']} + component]
            += source_adjoint[component];
}}"""


def _render_host_source_reverse_packed(interaction: dict[str, Any]) -> str:
    index = int(interaction["index"])
    extents = _interaction_extents(interaction)
    path_blocks = []
    for path in interaction["paths"]:
        source_block = path["input_1"]
        edge_block = path["input_2"]
        output = path["output"]
        statements = [_weight_expression(path, extents)]
        statements.append("if (apply_cutoff) weight *= args->edge_cutoff_scale[edge];")
        for term in path["sparse_wigner"]["terms"]:
            source_index = _ir_mul_index(source_block, int(term["a"]))
            output_index = _ir_mul_index(output, int(term["c"]))
            harmonic_index = int(edge_block["offset"]) + int(term["b"])
            statements.append(
                f"source_adjoint[{source_index}]"
                f" += {_cpp_float(path['path_weight'])} * weight"
                f" * {_cpp_float(term['coefficient'])}"
                f" * args->edge_input_2[local_edge * {extents['input_2_dimension']}"
                f" + {harmonic_index}]"
                f" * args->target_node_output_adjoint["
                f"target * {extents['output_dimension']} + {output_index}]"
                f"{_output_mask_factor(interaction, output_index)};"
            )
        path_blocks.append(
            "{\n            " + "\n            ".join(statements) + "\n            }"
        )
    return f"""static void source_reverse_{index}(
    const SymmetrixJitMH1HostReverseArgsV3* args,
    std::int32_t source_owner) noexcept
{{
    const bool apply_cutoff =
        (args->flags & SYMMETRIX_JIT_MH1_HOST_APPLY_EDGE_CUTOFF_V3) != 0;
    const bool has_bias =
        (args->flags & SYMMETRIX_JIT_MH1_HOST_HAS_LINEAR_BIAS_V3) != 0;
    const bool has_fixed =
        (args->flags & SYMMETRIX_JIT_MH1_HOST_HAS_FIXED_CONTRIBUTION_V3) != 0;
    const std::int32_t first_scheduled =
        args->source_edge_offsets[source_owner];
    const std::int32_t source = args->source_indices[
        args->source_edge_indices[first_scheduled]];
    float source_adjoint[{extents['input_1_dimension']}] = {{0.0f}};
    for (std::int32_t scheduled = first_scheduled;
         scheduled < args->source_edge_offsets[source_owner + 1]; ++scheduled) {{
        const std::int64_t edge = args->source_edge_indices[scheduled];
        const std::int32_t local_edge =
            static_cast<std::int32_t>(edge - args->first_edge);
        const std::int32_t target = args->target_indices[edge];
        for (std::int32_t channel = 0;
             channel < {extents['multiplicity']}; ++channel) {{
            {chr(10).join(path_blocks)}
        }}
    }}
    for (std::int32_t input = 0; input < {extents['input_1_dimension']}; ++input)
        args->source_node_input_adjoint[
            source * {extents['input_1_dimension']} + input] += source_adjoint[input];
}}"""


def _render_host_source_edge_reverse_packed(interaction: dict[str, Any]) -> str:
    index = int(interaction["index"])
    extents = _interaction_extents(interaction)
    path_blocks = []
    for path in interaction["paths"]:
        source_block = path["input_1"]
        edge_block = path["input_2"]
        output = path["output"]
        statements = [_weight_expression(path, extents)]
        for term_ordinal, term in enumerate(path["sparse_wigner"]["terms"]):
            source_index = _ir_mul_index(source_block, int(term["a"]))
            output_index = _ir_mul_index(output, int(term["c"]))
            harmonic_index = int(edge_block["offset"]) + int(term["b"])
            coefficient = (
                f"{_cpp_float(path['path_weight'])}"
                f" * {_cpp_float(term['coefficient'])}"
            )
            statements.extend(
                (
                    f"const float harmonic_{term_ordinal}"
                    f" = args->edge_input_2[local_edge * {extents['input_2_dimension']}"
                    f" + {harmonic_index}];",
                    f"const float target_adjoint_{term_ordinal}"
                    f" = args->target_node_output_adjoint["
                    f"target * {extents['output_dimension']} + {output_index}]"
                    f"{_output_mask_factor(interaction, output_index)};",
                    f"const float term_common_{term_ordinal}"
                    f" = {coefficient}"
                    f" * target_adjoint_{term_ordinal};",
                    f"source_adjoint[{source_index}]"
                    f" += term_common_{term_ordinal}"
                    f" * weight * cutoff"
                    f" * harmonic_{term_ordinal};",
                    f"const float edge_common_{term_ordinal}"
                    f" = term_common_{term_ordinal}"
                    f" * args->source_node_values["
                    f"source * {extents['input_1_dimension']} + {source_index}];",
                    f"weight_adjoint += edge_common_{term_ordinal}"
                    f" * harmonic_{term_ordinal};",
                    f"harmonic_adjoint[{harmonic_index}]"
                    f" += edge_common_{term_ordinal}"
                    f" * weight * cutoff;",
                )
            )
        weight_index = f"{int(path['weight_offset'])} + channel"
        statements.append(
            f"for (std::int32_t phi = 0; phi < {extents['phi_dimension']}; ++phi) {{\n"
            f"                phi_adjoint[phi]"
            f" += weight_adjoint * cutoff * args->linear_weight["
            f"({weight_index}) * {extents['phi_dimension']} + phi];\n"
            f"            }}"
        )
        statements.append(
            "if (apply_cutoff) cutoff_adjoint += weight_adjoint * weight;"
        )
        path_blocks.append(
            "{\n                float weight_adjoint = 0.0f;\n                "
            + "\n                ".join(statements)
            + "\n            }"
        )
    return f"""static void fused_source_edge_reverse_{index}(
    const SymmetrixJitMH1HostReverseArgsV4* args,
    std::int32_t source_owner) noexcept
{{
    const bool apply_cutoff =
        (args->flags & SYMMETRIX_JIT_MH1_HOST_APPLY_EDGE_CUTOFF_V3) != 0;
    const bool has_bias =
        (args->flags & SYMMETRIX_JIT_MH1_HOST_HAS_LINEAR_BIAS_V3) != 0;
    const bool has_fixed =
        (args->flags & SYMMETRIX_JIT_MH1_HOST_HAS_FIXED_CONTRIBUTION_V3) != 0;
    const std::int32_t first_scheduled = args->source_edge_offsets[source_owner];
    const std::int32_t source = args->source_indices[
        args->source_edge_indices[first_scheduled]];
    float source_adjoint[{extents['input_1_dimension']}] = {{0.0f}};
    for (std::int32_t scheduled = first_scheduled;
         scheduled < args->source_edge_offsets[source_owner + 1]; ++scheduled) {{
        const std::int64_t edge = args->source_edge_indices[scheduled];
        const std::int32_t local_edge =
            static_cast<std::int32_t>(edge - args->first_edge);
        const std::int32_t target = args->target_indices[edge];
        const float cutoff = apply_cutoff ? args->edge_cutoff_scale[edge] : 1.0f;
        float phi_adjoint[{extents['phi_dimension']}] = {{0.0f}};
        float harmonic_adjoint[{extents['input_2_dimension']}] = {{0.0f}};
        float cutoff_adjoint = 0.0f;
        for (std::int32_t channel = 0;
             channel < {extents['multiplicity']}; ++channel) {{
            {chr(10).join(path_blocks)}
        }}
        for (std::int32_t phi = 0; phi < {extents['phi_dimension']}; ++phi)
            args->edge_phi_adjoint[
                local_edge * {extents['phi_dimension']} + phi] = phi_adjoint[phi];
        for (std::int32_t harmonic = 0;
             harmonic < {extents['input_2_dimension']}; ++harmonic)
            args->edge_input_2_adjoint[
                local_edge * {extents['input_2_dimension']} + harmonic] =
                    harmonic_adjoint[harmonic];
        args->edge_cutoff_scale_adjoint[local_edge] = cutoff_adjoint;
    }}
    for (std::int32_t input = 0; input < {extents['input_1_dimension']}; ++input)
        args->source_node_input_adjoint[
            source * {extents['input_1_dimension']} + input] += source_adjoint[input];
}}"""


def _render_host_edge_reverse(interaction: dict[str, Any]) -> str:
    index = int(interaction["index"])
    extents = _interaction_extents(interaction)
    path_blocks = []
    for path in interaction["paths"]:
        source_block = path["input_1"]
        edge_block = path["input_2"]
        output = path["output"]
        statements = [_weight_expression(path, extents), "float weight_adjoint = 0.0f;"]
        for term in path["sparse_wigner"]["terms"]:
            source_index = _ir_mul_index(source_block, int(term["a"]))
            harmonic_index = int(edge_block["offset"]) + int(term["b"])
            output_index = _ir_mul_index(output, int(term["c"]))
            common = (
                f"{_cpp_float(path['path_weight'])}"
                f" * {_cpp_float(term['coefficient'])}"
                f" * args->source_node_values[source * {extents['input_1_dimension']}"
                f" + {source_index}]"
                f" * args->target_node_output_adjoint["
                f"target * {extents['output_dimension']} + {output_index}]"
                f"{_output_mask_factor(interaction, output_index)}"
            )
            statements.append(
                f"weight_adjoint += ({common})"
                f" * args->edge_input_2[local_edge * {extents['input_2_dimension']}"
                f" + {harmonic_index}];"
            )
            statements.append(
                f"harmonic_adjoint[{harmonic_index}]"
                f" += ({common}) * weight * cutoff;"
            )
        weight_index = f"{int(path['weight_offset'])} + channel"
        statements.append(
            f"for (std::int32_t phi = 0; phi < {extents['phi_dimension']}; ++phi) {{\n"
            f"                phi_adjoint[phi]"
            f" += weight_adjoint * cutoff * args->linear_weight["
            f"({weight_index}) * {extents['phi_dimension']} + phi];\n"
            f"            }}"
        )
        statements.append(
            "if (apply_cutoff) cutoff_adjoint += weight_adjoint * weight;"
        )
        path_blocks.append(
            "{\n            " + "\n            ".join(statements) + "\n            }"
        )
    return f"""static void edge_reverse_{index}(
    const SymmetrixJitMH1HostReverseArgsV3* args,
    std::int32_t local_edge) noexcept
{{
    const std::int64_t edge = args->first_edge + local_edge;
    const std::int32_t source = args->source_indices[edge];
    const std::int32_t target = args->target_indices[edge];
    const bool apply_cutoff =
        (args->flags & SYMMETRIX_JIT_MH1_HOST_APPLY_EDGE_CUTOFF_V3) != 0;
    const bool has_bias =
        (args->flags & SYMMETRIX_JIT_MH1_HOST_HAS_LINEAR_BIAS_V3) != 0;
    const bool has_fixed =
        (args->flags & SYMMETRIX_JIT_MH1_HOST_HAS_FIXED_CONTRIBUTION_V3) != 0;
    const float cutoff = apply_cutoff ? args->edge_cutoff_scale[edge] : 1.0f;
    float phi_adjoint[{extents['phi_dimension']}] = {{0.0f}};
    float harmonic_adjoint[{extents['input_2_dimension']}] = {{0.0f}};
    float cutoff_adjoint = 0.0f;
    for (std::int32_t channel = 0; channel < {extents['multiplicity']}; ++channel) {{
        {chr(10).join(path_blocks)}
    }}
    for (std::int32_t phi = 0; phi < {extents['phi_dimension']}; ++phi)
        args->edge_phi_adjoint[local_edge * {extents['phi_dimension']} + phi] =
            phi_adjoint[phi];
    for (std::int32_t harmonic = 0;
         harmonic < {extents['input_2_dimension']}; ++harmonic)
        args->edge_input_2_adjoint[
            local_edge * {extents['input_2_dimension']} + harmonic] =
                harmonic_adjoint[harmonic];
    args->edge_cutoff_scale_adjoint[local_edge] = cutoff_adjoint;
}}"""


def render_jit_mh1_host_plugin(contract: dict[str, Any]) -> str:
    """Render one host plugin containing both MH-1 interaction kernels."""

    normalized = normalize_execution_mh1_contract(contract)
    metadata = jit_mh1_host_plugin_metadata(normalized)
    interactions = normalized["interactions"]
    conditioner_layouts = metadata["conditioner_layout"]["interactions"]
    functions = []
    for interaction in interactions:
        functions.extend(
            (
                _render_host_forward(interaction),
                _render_host_source_reverse_packed(interaction),
                _render_host_edge_reverse(interaction),
            )
        )
    conditioner_helpers = render_execution_mh1_conditioner_cpp_helpers(
        normalized, function_qualifier="static inline"
    )
    conditioner_owners = [
        _render_conditioning_owners(
            interaction_layout,
            backend="Host",
            function_qualifier="static inline",
        )
        for interaction_layout in conditioner_layouts
    ]
    descriptors = ",\n        ".join(
        "{"
        + ", ".join(
            (
                "sizeof(SymmetrixJitMH1HostInteractionV3)",
                "0u",
                str(extents["input_1_dimension"]),
                str(extents["input_2_dimension"]),
                str(extents["output_dimension"]),
                str(extents["weight_size"]),
                str(extents["phi_dimension"]),
                str(extents["multiplicity"]),
                str(extents["input_1_angular_dimension"]),
                str(extents["instruction_count"]),
            )
        )
        + "}"
        for extents in metadata["interactions"]
    )
    conditioning_descriptors = ",\n        ".join(
        "{"
        + ", ".join(
            (
                "sizeof(SymmetrixJitMH1HostConditioningOwnersV3)",
                f"{index}u",
                "0u",
                "0u",
                f"&conditioning_forward_{index}",
                f"&conditioning_reverse_{index}",
            )
        )
        + "}"
        for index in range(2)
    )
    return f"""{_HOST_PLUGIN_ABI_HEADER}

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {{

{chr(10).join(functions)}

{conditioner_helpers}

{chr(10).join(conditioner_owners)}

void forward_owner(
    const SymmetrixJitMH1HostForwardArgsV3* args,
    std::int32_t receiver) noexcept
{{
    if (args->interaction == 0u) forward_0(args, receiver);
    else if (args->interaction == 1u) forward_1(args, receiver);
}}

void source_reverse_owner(
    const SymmetrixJitMH1HostReverseArgsV3* args,
    std::int32_t source) noexcept
{{
    if (args->interaction == 0u) source_reverse_0(args, source);
    else if (args->interaction == 1u) source_reverse_1(args, source);
}}

void edge_reverse_owner(
    const SymmetrixJitMH1HostReverseArgsV3* args,
    std::int32_t local_edge) noexcept
{{
    if (args->interaction == 0u) edge_reverse_0(args, local_edge);
    else if (args->interaction == 1u) edge_reverse_1(args, local_edge);
}}

const SymmetrixJitMH1HostPluginV3 descriptor {{
    SYMMETRIX_JIT_MH1_HOST_PLUGIN_ABI_VERSION,
    sizeof(SymmetrixJitMH1HostPluginV3),
    sizeof(void*),
    SYMMETRIX_JIT_MH1_HOST_PLUGIN_BYTE_ORDER,
    SYMMETRIX_JIT_MH1_HOST_FORWARD_OWNER_V3
        | SYMMETRIX_JIT_MH1_HOST_SOURCE_REVERSE_OWNER_V3
        | SYMMETRIX_JIT_MH1_HOST_EDGE_REVERSE_OWNER_V3
        | SYMMETRIX_JIT_MH1_HOST_GRAPH_WIDE_V3
        | SYMMETRIX_JIT_MH1_HOST_IR_MUL_FEATURES_V3
        | SYMMETRIX_JIT_MH1_HOST_CONDITIONING_OWNER_V3,
    SYMMETRIX_JIT_MH1_HOST_PLUGIN_INTERACTIONS,
    sizeof(float),
    0u,
    SYMMETRIX_JIT_MH1_HOST_PLUGIN_ABI_TAG,
    {_cpp_string(metadata['artifact_id'])},
    {_cpp_string(metadata['generation_fingerprint'])},
    {_cpp_string(metadata['semantic_fingerprint'])},
    {_cpp_string(metadata['structure_fingerprint'])},
    {{
        {descriptors}
    }},
    &forward_owner,
    &source_reverse_owner,
    &edge_reverse_owner,
    {{
        {conditioning_descriptors}
    }},
}};

}}  // namespace

extern "C" SYMMETRIX_JIT_MH1_HOST_PLUGIN_EXPORT
const SymmetrixJitMH1HostPluginV3*
symmetrix_jit_mh1_host_plugin_query_v3(void)
{{
    return &descriptor;
}}
"""


def _render_host_node_phase_owners(
    layer: dict[str, Any], metadata: dict[str, Any]
) -> str:
    index = int(layer["index"])
    dimensions = metadata
    input_dimension = int(dimensions["input_dimension"])
    up_dimension = int(dimensions["up_dimension"])
    residual_dimension = int(dimensions["residual_dimension"])
    gated_dimension = int(layer["gate"]["dimensions"]["irreps_out"])
    message_dimension = int(dimensions["message_dimension"])
    interaction_dimension = int(dimensions["interaction_output_dimension"])
    output_dimension = int(dimensions["output_dimension"])
    arena_dimension = int(dimensions["node_arena_dimension"])
    input_row = (
        "nullptr" if index == 0 else f"args->layer_input + node * {input_dimension}"
    )
    input_adjoint_row = (
        "nullptr"
        if index == 0
        else f"args->layer_input_adjoint + node * {input_dimension}"
    )
    pre_reverse = ""
    if index > 0:
        pre_reverse = f"""
void node_pre_reverse_owner_{index}(
    const SymmetrixJitMH1HostNodeReverseArgsV4* args,
    std::int32_t node) noexcept
{{
    execution_mh1_node_layer_{index}_pre_reverse(
        args->up_adjoint + static_cast<std::int64_t>(node) * {up_dimension},
        args->linear_parameters,
        args->node_arena + static_cast<std::int64_t>(node) * {arena_dimension},
        args->layer_input_adjoint
            + static_cast<std::int64_t>(node) * {input_dimension});
}}
"""
    return f"""void node_pre_forward_owner_{index}(
    const SymmetrixJitMH1HostNodeForwardArgsV4* args,
    std::int32_t node) noexcept
{{
    execution_mh1_node_layer_{index}_pre_forward(
        args->element_indices[node], {input_row}, args->linear_parameters,
        args->node_arena + static_cast<std::int64_t>(node) * {arena_dimension},
        args->up_output + static_cast<std::int64_t>(node) * {up_dimension});
}}

void node_post_forward_owner_{index}(
    const SymmetrixJitMH1HostNodeForwardArgsV4* args,
    std::int32_t node) noexcept
{{
    if ((args->flags & SYMMETRIX_JIT_MH1_HOST_NODE_TILE_GATE_ARGUMENT_V4) != 0u) {{
        execution_mh1_node_layer_{index}_tile_gate_forward(
            args->node_density[node],
            args->up + static_cast<std::int64_t>(node) * {residual_dimension},
            args->messages + static_cast<std::int64_t>(node) * {residual_dimension},
            args->linear_parameters,
            args->up_output + static_cast<std::int64_t>(node) * {residual_dimension},
            args->layer_output + static_cast<std::int64_t>(node) * {gated_dimension});
        return;
    }}
    if ((args->flags & SYMMETRIX_JIT_MH1_HOST_NODE_TILE_PRODUCT_ARGUMENT_V4) != 0u) {{
        execution_mh1_node_layer_{index}_tile_product_forward(
            args->element_indices[node], {input_row},
            args->up + static_cast<std::int64_t>(node) * {interaction_dimension},
            args->linear_parameters, args->product_parameters,
            args->readout_parameters,
            args->node_arena + static_cast<std::int64_t>(node) * {arena_dimension},
            args->layer_output + static_cast<std::int64_t>(node) * {output_dimension},
            args->readout_contribution + node);
        return;
    }}
    execution_mh1_node_layer_{index}_post_forward(
        args->element_indices[node], args->node_density[node], {input_row},
        args->up + static_cast<std::int64_t>(node) * {up_dimension},
        args->messages + static_cast<std::int64_t>(node) * {message_dimension},
        args->linear_parameters, args->product_parameters,
        args->readout_parameters,
        args->node_arena + static_cast<std::int64_t>(node) * {arena_dimension},
        args->layer_output + static_cast<std::int64_t>(node) * {output_dimension},
        args->readout_contribution + node);
}}

void node_post_reverse_owner_{index}(
    const SymmetrixJitMH1HostNodeReverseArgsV4* args,
    std::int32_t node) noexcept
{{
    if ((args->flags & SYMMETRIX_JIT_MH1_HOST_NODE_TILE_PRODUCT_ARGUMENT_V4) != 0u) {{
        execution_mh1_node_layer_{index}_tile_product_reverse(
            args->element_indices[node], args->energy_scale, {input_row},
            args->up + static_cast<std::int64_t>(node) * {interaction_dimension},
            args->layer_output + static_cast<std::int64_t>(node) * {output_dimension},
            args->linear_parameters, args->product_parameters,
            args->readout_parameters,
            args->node_arena + static_cast<std::int64_t>(node) * {arena_dimension},
            args->layer_output_adjoint
                + static_cast<std::int64_t>(node) * {output_dimension},
            args->message_adjoint
                + static_cast<std::int64_t>(node) * {interaction_dimension},
            {input_adjoint_row});
        return;
    }}
    if ((args->flags & SYMMETRIX_JIT_MH1_HOST_NODE_TILE_GATE_ARGUMENT_V4) != 0u) {{
        float* row_arena = args->node_arena
            + static_cast<std::int64_t>(node) * {arena_dimension};
        execution_mh1_node_layer_{index}_tile_gate_reverse(
            args->node_density[node],
            args->up + static_cast<std::int64_t>(node) * {residual_dimension},
            args->messages + static_cast<std::int64_t>(node) * {residual_dimension},
            args->linear_parameters,
            args->layer_output + static_cast<std::int64_t>(node) * {gated_dimension},
            row_arena,
            args->up_adjoint
                + static_cast<std::int64_t>(node) * {residual_dimension},
            args->message_adjoint
                + static_cast<std::int64_t>(node) * {residual_dimension},
            args->node_density_adjoint + node);
        return;
    }}
    execution_mh1_node_layer_{index}_post_reverse(
        args->element_indices[node], args->node_density[node],
        args->energy_scale, {input_row},
        args->up + static_cast<std::int64_t>(node) * {up_dimension},
        args->messages + static_cast<std::int64_t>(node) * {message_dimension},
        args->layer_output + static_cast<std::int64_t>(node) * {output_dimension},
        args->linear_parameters, args->product_parameters,
        args->readout_parameters,
        args->node_arena + static_cast<std::int64_t>(node) * {arena_dimension},
        args->layer_output_adjoint
            + static_cast<std::int64_t>(node) * {output_dimension},
        args->message_adjoint
            + static_cast<std::int64_t>(node) * {message_dimension},
        args->up_adjoint + static_cast<std::int64_t>(node) * {up_dimension},
        {input_adjoint_row}, args->node_density_adjoint + node);
}}
{pre_reverse}"""


def render_jit_mh1_host_plugin_v4(
    contract: dict[str, Any],
    *,
    node_state_policy: str = MH1_NODE_STATE_RECOMPUTE,
) -> str:
    """Render a dual-query host artifact with complete generated node phases."""

    normalized = normalize_execution_mh1_v4_contract(contract)
    v3_contract = project_execution_mh1_v3_contract(normalized)
    source = render_jit_mh1_host_plugin(v3_contract)
    tiled_forward_helpers = "\n\n".join(
        _render_host_forward_channel_tiled(interaction).replace(
            f"forward_{int(interaction['index'])}",
            f"forward_tiled_{int(interaction['index'])}",
            1,
        )
        for interaction in v3_contract["interactions"]
    )
    interaction_metadata = jit_mh1_host_plugin_metadata(v3_contract)
    node_metadata = execution_mh1_node_program_metadata(
        normalized, backend="host", node_state_policy=node_state_policy
    )
    fused_reverse_helpers = "\n\n".join(
        _render_host_source_edge_reverse_packed(interaction)
        for interaction in v3_contract["interactions"]
    )
    node_helpers = render_execution_mh1_node_program_cpp_helpers(
        normalized, function_qualifier="static inline"
    )
    owner_helpers = "\n\n".join(
        _render_host_node_phase_owners(layer, metadata)
        for layer, metadata in zip(
            normalized["node_program"]["layers"],
            node_metadata["layers"],
            strict=True,
        )
    )
    interactions = ",\n        ".join(
        "{"
        + ", ".join(
            (
                "sizeof(SymmetrixJitMH1HostInteractionV4)",
                "0u",
                str(extents["input_1_dimension"]),
                str(extents["input_2_dimension"]),
                str(extents["output_dimension"]),
                str(extents["weight_size"]),
                str(extents["phi_dimension"]),
                str(extents["multiplicity"]),
                str(extents["input_1_angular_dimension"]),
                str(extents["instruction_count"]),
            )
        )
        + "}"
        for extents in interaction_metadata["interactions"]
    )
    conditioning = ",\n        ".join(
        "{"
        + ", ".join(
            (
                "sizeof(SymmetrixJitMH1HostConditioningOwnersV4)",
                f"{index}u",
                "0u",
                "0u",
                f"&conditioning_forward_{index}",
                f"&conditioning_reverse_{index}",
            )
        )
        + "}"
        for index in range(2)
    )
    node_descriptors = []
    for item in node_metadata["layers"]:
        index = int(item["index"])
        forward = item["forward_phases"]
        reverse = item["reverse_phases"]
        forward_descriptors = ", ".join(
            "{"
            + ", ".join(
                (
                    "sizeof(SymmetrixJitMH1HostNodeForwardPhaseV4)",
                    f"{index}u",
                    f"{phase['phase']}u",
                    "0u",
                    str(phase["owners_per_node"]),
                    "0",
                    "0",
                    "0",
                    f"&node_{'pre' if phase['phase']==0 else 'post'}_forward_owner_{index}",
                )
            )
            + "}"
            for phase in forward
        )
        reverse_descriptors = []
        for phase in reverse:
            enabled = bool(phase["enabled"])
            owner = (
                f"&node_{'post' if phase['phase']==0 else 'pre'}_reverse_owner_{index}"
                if enabled
                else "nullptr"
            )
            reverse_descriptors.append(
                "{"
                + ", ".join(
                    (
                        "sizeof(SymmetrixJitMH1HostNodeReversePhaseV4)",
                        f"{index}u",
                        f"{2+phase['phase']}u",
                        "0u",
                        str(phase["owners_per_node"]),
                        "0",
                        "0",
                        "0",
                        owner,
                    )
                )
                + "}"
            )
        flags = []
        if item["requires_tp_source_state_adjoint"]:
            flags.append(
                "SYMMETRIX_JIT_MH1_HOST_NODE_REQUIRES_TP_SOURCE_STATE_ADJOINT_V4"
            )
        if item["retained_interaction_output_dimension"]:
            flags.append("SYMMETRIX_JIT_MH1_HOST_NODE_RETAIN_INTERACTION_OUTPUT_V4")
        flags_expression = " | ".join(flags) if flags else "0u"
        node_descriptors.append(
            "{"
            + ", ".join(
                (
                    "sizeof(SymmetrixJitMH1HostNodeProgramV4)",
                    f"{index}u",
                    "2u",
                    "2u",
                    str(item["element_count"]),
                    str(item["input_dimension"]),
                    str(item["up_dimension"]),
                    str(item["residual_dimension"]),
                    str(item["skip_dimension"]),
                    str(item["message_dimension"]),
                    str(item["interaction_output_dimension"]),
                    str(item["output_dimension"]),
                    str(item["product_term_count"]),
                    str(item["node_arena_dimension"]),
                    flags_expression,
                    "0",
                    str(item["linear_parameter_count"]),
                    str(item["product_parameter_count"]),
                    str(item["readout_parameter_count"]),
                    "0",
                    "{" + forward_descriptors + "}",
                    "{" + ", ".join(reverse_descriptors) + "}",
                )
            )
            + "}"
        )
    return (
        source
        + f"""

namespace {{

{tiled_forward_helpers}

void tiled_forward_owner(
    const SymmetrixJitMH1HostForwardArgsV3* args,
    std::int32_t receiver) noexcept
{{
    if (args->interaction == 0u)
        forward_tiled_0(args, receiver);
    else if (args->interaction == 1u)
        forward_tiled_1(args, receiver);
}}

{fused_reverse_helpers}

void fused_source_edge_reverse_owner(
    const SymmetrixJitMH1HostReverseArgsV4* args,
    std::int32_t source_owner) noexcept
{{
    if (args->interaction == 0u)
        fused_source_edge_reverse_0(args, source_owner);
    else if (args->interaction == 1u)
        fused_source_edge_reverse_1(args, source_owner);
}}

{node_helpers}

{owner_helpers}

const SymmetrixJitMH1HostPluginV4 descriptor_v4 {{
    SYMMETRIX_JIT_MH1_HOST_PLUGIN_V4_ABI_VERSION,
    sizeof(SymmetrixJitMH1HostPluginV4), sizeof(void*),
    SYMMETRIX_JIT_MH1_HOST_PLUGIN_BYTE_ORDER,
    SYMMETRIX_JIT_MH1_HOST_FORWARD_OWNER_V4
        | SYMMETRIX_JIT_MH1_HOST_SOURCE_REVERSE_OWNER_V4
        | SYMMETRIX_JIT_MH1_HOST_EDGE_REVERSE_OWNER_V4
        | SYMMETRIX_JIT_MH1_HOST_GRAPH_WIDE_V4
        | SYMMETRIX_JIT_MH1_HOST_IR_MUL_FEATURES_V4
        | SYMMETRIX_JIT_MH1_HOST_CONDITIONING_OWNER_V4
        | SYMMETRIX_JIT_MH1_HOST_NODE_PROGRAM_PHASES_V4
        | SYMMETRIX_JIT_MH1_HOST_PERSISTENT_IR_MUL_NODE_STATE_V4
        | SYMMETRIX_JIT_MH1_HOST_FIXED_WEIGHT_COORDINATES_V4
        | SYMMETRIX_JIT_MH1_HOST_NODE_TILE_GATE_V4
        | SYMMETRIX_JIT_MH1_HOST_NODE_TILE_PRODUCT_V4
        | SYMMETRIX_JIT_MH1_HOST_SOURCE_OWNS_EDGE_REVERSE_V4
        | SYMMETRIX_JIT_MH1_HOST_PHI_MAJOR_FORWARD_WEIGHT_V4,
    2u, sizeof(float), 0u,
    SYMMETRIX_JIT_MH1_HOST_PLUGIN_V4_ABI_TAG,
    {_cpp_string(node_metadata['artifact_id'])},
    {_cpp_string(node_metadata['generation_fingerprint'])},
    {_cpp_string(node_metadata['semantic_fingerprint'])},
    {_cpp_string(node_metadata['structure_fingerprint'])},
    {_cpp_string(node_metadata['runtime_layout_fingerprint'])},
    {{ {interactions} }},
    &tiled_forward_owner, &fused_source_edge_reverse_owner, &edge_reverse_owner,
    {{ {conditioning} }},
    {{ {', '.join(node_descriptors)} }},
}};

}}  // namespace

extern "C" SYMMETRIX_JIT_MH1_HOST_PLUGIN_EXPORT
const SymmetrixJitMH1HostPluginV4*
symmetrix_jit_mh1_host_plugin_query_v4(void)
{{
    return &descriptor_v4;
}}
"""
    )


def render_jit_mh1_host_plugin_v5(
    contract: dict[str, Any],
    *,
    precision: str = "float32",
    node_state_policy: str = MH1_NODE_STATE_RECOMPUTE,
) -> str:
    """Render a host artifact with generated spline-aware R forward stages."""

    normalized = normalize_execution_mh1_v4_contract(contract)
    v3_contract = project_execution_mh1_v3_contract(normalized)
    source = render_jit_mh1_host_plugin_v4(
        normalized, node_state_policy=node_state_policy
    )
    packed_messages = precision in ("float32", "float64")
    spline_owners = "\n\n".join(
        _render_host_spline_r_forward(interaction, packed_messages=packed_messages)
        + "\n\n"
        + _render_host_spline_r_reverse(interaction)
        for interaction in v3_contract["interactions"]
    )
    spline_flags = (
        "SYMMETRIX_JIT_MH1_HOST_SPLINE_R_PACKED_FORWARD_MESSAGES_V5"
        if packed_messages
        else "0u"
    )
    spline_descriptors = ", ".join(
        "{"
        + ", ".join(
            (
                "sizeof(SymmetrixJitMH1HostSplineRProgramV5)",
                f"{index}u",
                spline_flags,
                "0u",
                str(_interaction_extents(interaction)["weight_size"]),
                str(_interaction_extents(interaction)["weight_size"]),
                "4",
                "0",
                f"&spline_r_forward_{index}",
                f"&spline_r_source_reverse_{index}",
            )
        )
        + "}"
        for index, interaction in enumerate(v3_contract["interactions"])
    )
    spline_extension = f"""

namespace {{

static inline void prefetch_read(const void* address) noexcept
{{
#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_LLVM_COMPILER)
    __builtin_prefetch(address, 0, 1);
#else
    (void)address;
#endif
}}

static inline float spline_r_value(
    const SymmetrixJitMH1HostSplineRForwardArgsV5* args,
    std::int32_t pair,
    std::int32_t interval,
    float x,
    float xx,
    float xxx,
    std::int32_t function) noexcept
{{
    const float* const coefficients =
        static_cast<const float*>(args->coefficients);
    const std::int64_t base =
        ((static_cast<std::int64_t>(pair) * args->interval_count + interval)
            * 4) * args->function_count + function;
    return coefficients[base]
        + coefficients[base + args->function_count] * x
        + coefficients[base + 2 * args->function_count] * xx
        + coefficients[base + 3 * args->function_count] * xxx;
}}

static inline float spline_r_value(
    const SymmetrixJitMH1HostSplineRReverseArgsV5* args,
    std::int32_t pair,
    std::int32_t interval,
    float x,
    float xx,
    float xxx,
    std::int32_t function) noexcept
{{
    const float* const coefficients =
        static_cast<const float*>(args->coefficients);
    const std::int64_t base =
        ((static_cast<std::int64_t>(pair) * args->interval_count + interval)
            * 4) * args->function_count + function;
    return coefficients[base]
        + coefficients[base + args->function_count] * x
        + coefficients[base + 2 * args->function_count] * xx
        + coefficients[base + 3 * args->function_count] * xxx;
}}

static inline float spline_r_derivative(
    const SymmetrixJitMH1HostSplineRReverseArgsV5* args,
    std::int32_t pair,
    std::int32_t interval,
    float x,
    float xx,
    std::int32_t function) noexcept
{{
    const float* const coefficients =
        static_cast<const float*>(args->coefficients);
    const std::int64_t base =
        ((static_cast<std::int64_t>(pair) * args->interval_count + interval)
            * 4) * args->function_count + function;
    return coefficients[base + args->function_count]
        + 2.0f * coefficients[base + 2 * args->function_count] * x
        + 3.0f * coefficients[base + 3 * args->function_count] * xx;
}}

{spline_owners}

const SymmetrixJitMH1HostPluginV5 descriptor_v5 = [] {{
    SymmetrixJitMH1HostPluginV5 value{{}};
    value.abi_version = SYMMETRIX_JIT_MH1_HOST_PLUGIN_V5_ABI_VERSION;
    value.struct_size = sizeof(SymmetrixJitMH1HostPluginV5);
    value.pointer_size = sizeof(void*);
    value.byte_order = SYMMETRIX_JIT_MH1_HOST_PLUGIN_BYTE_ORDER;
    value.capabilities = descriptor_v4.capabilities
        | SYMMETRIX_JIT_MH1_HOST_SPLINE_R_FORWARD_V5
        | SYMMETRIX_JIT_MH1_HOST_SPLINE_R_SOURCE_REVERSE_V5;
    value.interaction_count = SYMMETRIX_JIT_MH1_HOST_PLUGIN_INTERACTIONS;
    value.scalar_size = sizeof(float);
    value.abi_tag = SYMMETRIX_JIT_MH1_HOST_PLUGIN_V5_ABI_TAG;
    value.artifact_id = descriptor_v4.artifact_id;
    value.generation_fingerprint = descriptor_v4.generation_fingerprint;
    value.semantic_fingerprint = descriptor_v4.semantic_fingerprint;
    value.structure_fingerprint = descriptor_v4.structure_fingerprint;
    value.runtime_layout_fingerprint = descriptor_v4.runtime_layout_fingerprint;
    value.forward_owner = descriptor_v4.forward_owner;
    value.source_reverse_owner = descriptor_v4.source_reverse_owner;
    value.edge_reverse_owner = descriptor_v4.edge_reverse_owner;
    for (std::size_t interaction = 0;
         interaction < SYMMETRIX_JIT_MH1_HOST_PLUGIN_INTERACTIONS;
         ++interaction) {{
        value.interactions[interaction] = descriptor_v4.interactions[interaction];
        value.conditioning[interaction] = descriptor_v4.conditioning[interaction];
        value.node_programs[interaction] = descriptor_v4.node_programs[interaction];
    }}
    const SymmetrixJitMH1HostSplineRProgramV5 spline_programs[] = {{
        {spline_descriptors}
    }};
    for (std::size_t interaction = 0;
         interaction < SYMMETRIX_JIT_MH1_HOST_PLUGIN_INTERACTIONS;
         ++interaction)
        value.spline_r_programs[interaction] = spline_programs[interaction];
    return value;
}}();

}}  // namespace

extern "C" SYMMETRIX_JIT_MH1_HOST_PLUGIN_EXPORT
const SymmetrixJitMH1HostPluginV5*
symmetrix_jit_mh1_host_plugin_query_v5(void)
{{
    return &descriptor_v5;
}}
"""
    return source + _convert_host_spline_precision(spline_extension, precision)


def _render_cuda_device_function(function: str) -> str:
    """Translate one ownership function to the field-compatible CUDA ABI."""

    return (
        function.replace("static void ", "__device__ __forceinline__ void ", 1)
        .replace("SymmetrixJitMH1Host", "SymmetrixJitMH1Cuda")
        .replace("SYMMETRIX_JIT_MH1_HOST", "SYMMETRIX_JIT_MH1_CUDA")
    )


def _group_interaction_paths(
    interaction: dict[str, Any], block_name: str
) -> list[list[dict[str, Any]]]:
    """Group paths by one irrep block while preserving contract order."""

    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for path in interaction["paths"]:
        block = path[block_name]
        key = (int(block["offset"]), int(block["components"]))
        groups.setdefault(key, []).append(path)
    return list(groups.values())


def _batch_interaction_paths(
    interaction: dict[str, Any],
    block_name: str,
    component_budget: int,
    fuse_component_limit: int | None = None,
) -> list[list[dict[str, Any]]]:
    """Pack consecutive irrep groups without splitting a stored block."""

    groups = _group_interaction_paths(interaction, block_name)
    return [
        [path for group_index in batch for path in groups[group_index]]
        for batch in _batch_interaction_group_indices(
            groups, block_name, component_budget, fuse_component_limit
        )
    ]


def _batch_interaction_group_indices(
    groups: list[list[dict[str, Any]]],
    block_name: str,
    component_budget: int,
    fuse_component_limit: int | None = None,
) -> list[list[int]]:
    """Pack whole irrep groups and return their stable contract-order indices."""

    if component_budget <= 0:
        raise ValueError("component budget must be positive")
    total_components = sum(int(group[0][block_name]["components"]) for group in groups)
    if fuse_component_limit is not None and total_components <= fuse_component_limit:
        return [list(range(len(groups)))]
    batches: list[list[int]] = []
    batch: list[int] = []
    batch_components = 0
    for group_index, group in enumerate(groups):
        components = int(group[0][block_name]["components"])
        if batch and batch_components + components > component_budget:
            batches.append(batch)
            batch = []
            batch_components = 0
        batch.append(group_index)
        batch_components += components
    if batch:
        batches.append(batch)
    return batches


def _output_group_key(group: list[dict[str, Any]]) -> tuple[int, int]:
    output = group[0]["output"]
    return int(output["offset"]), int(output["components"])


def _policy_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _make_forward_policy(
    normalized: dict[str, Any], compute_capability: int, request: str
) -> dict[str, Any]:
    if request not in ("auto", "budget", "shared", "full"):
        raise ValueError(
            "MH1 CUDA forward policy must be 'auto', 'budget', 'shared', or 'full'"
        )
    interaction_policies = []
    for interaction in normalized["interactions"]:
        groups = _group_interaction_paths(interaction, "output")
        total_components = sum(
            int(group[0]["output"]["components"]) for group in groups
        )
        strategy = request
        if strategy == "auto":
            strategy = (
                "full"
                if total_components <= _CUDA_FORWARD_FUSE_COMPONENT_LIMIT
                else "budget"
            )
        if strategy == "full":
            partition_indices = [list(range(len(groups)))]
        else:
            component_budget = (
                _CUDA_FORWARD_COMPONENT_BUDGET
                if strategy == "budget"
                else _CUDA_FORWARD_SHARED_COMPONENT_BUDGET
            )
            partition_indices = _batch_interaction_group_indices(
                groups,
                "output",
                component_budget,
                _CUDA_FORWARD_FUSE_COMPONENT_LIMIT,
            )
        full_partition = [list(range(len(groups)))]
        budget_partitions = _batch_interaction_group_indices(
            groups,
            "output",
            _CUDA_FORWARD_COMPONENT_BUDGET,
            _CUDA_FORWARD_FUSE_COMPONENT_LIMIT,
        )
        if partition_indices == full_partition:
            strategy = "full"
        elif partition_indices == budget_partitions:
            strategy = "budget"
        else:
            strategy = "shared"
        interaction_policies.append(
            {
                "index": int(interaction["index"]),
                "strategy": strategy,
                "partitions": [
                    [list(_output_group_key(groups[group])) for group in partition]
                    for partition in partition_indices
                ],
                "component_counts": [
                    sum(
                        int(groups[group][0]["output"]["components"])
                        for group in partition
                    )
                    for partition in partition_indices
                ],
            }
        )
    payload = {
        "tag": MH1_CUDA_FORWARD_POLICY_TAG,
        "generation_fingerprint": normalized["generation_fingerprint"],
        "target_compute_capability": compute_capability,
        "interactions": interaction_policies,
    }
    return {**payload, "policy_id": _policy_digest(payload)}


def _validate_forward_policy(
    normalized: dict[str, Any], compute_capability: int, policy: dict[str, Any]
) -> dict[str, Any]:
    candidate = json.loads(json.dumps(policy))
    required_fields = {
        "tag",
        "generation_fingerprint",
        "target_compute_capability",
        "interactions",
        "policy_id",
    }
    if set(candidate) != required_fields:
        raise ValueError("MH1 CUDA forward policy fields are invalid")
    if candidate.get("tag") != MH1_CUDA_FORWARD_POLICY_TAG:
        raise ValueError("MH1 CUDA forward policy tag is unsupported")
    if candidate.get("generation_fingerprint") != normalized["generation_fingerprint"]:
        raise ValueError("MH1 CUDA forward policy model identity does not match")
    if candidate.get("target_compute_capability") != compute_capability:
        raise ValueError("MH1 CUDA forward policy target does not match CUDA target")
    entries = candidate.get("interactions")
    if not isinstance(entries, list) or len(entries) != len(normalized["interactions"]):
        raise ValueError("MH1 CUDA forward policy interaction count is invalid")
    canonical_entries = []
    for interaction, entry in zip(normalized["interactions"], entries, strict=True):
        index = int(interaction["index"])
        if not isinstance(entry, dict) or entry.get("index") != index:
            raise ValueError("MH1 CUDA forward policy interaction index is invalid")
        if set(entry) != {
            "index",
            "strategy",
            "partitions",
            "component_counts",
        }:
            raise ValueError("MH1 CUDA forward policy interaction fields are invalid")
        strategy = entry.get("strategy")
        if strategy not in ("budget", "shared", "full"):
            raise ValueError("MH1 CUDA forward policy strategy is invalid")
        groups = _group_interaction_paths(interaction, "output")
        group_by_key = {_output_group_key(group): group for group in groups}
        partitions = entry.get("partitions")
        if not isinstance(partitions, list) or not partitions:
            raise ValueError("MH1 CUDA forward policy partitions are empty")
        seen: set[tuple[int, int]] = set()
        canonical_partitions = []
        component_counts = []
        for partition in partitions:
            if not isinstance(partition, list) or not partition:
                raise ValueError("MH1 CUDA forward policy contains an empty partition")
            canonical_partition = []
            component_count = 0
            for raw_key in partition:
                if (
                    not isinstance(raw_key, list)
                    or len(raw_key) != 2
                    or any(
                        isinstance(value, bool) or not isinstance(value, int)
                        for value in raw_key
                    )
                ):
                    raise ValueError("MH1 CUDA forward policy block key is invalid")
                key = (raw_key[0], raw_key[1])
                if key not in group_by_key:
                    raise ValueError(
                        "MH1 CUDA forward policy block is not in the contract"
                    )
                if key in seen:
                    raise ValueError("MH1 CUDA forward policy repeats an output block")
                seen.add(key)
                canonical_partition.append(list(key))
                component_count += key[1]
            canonical_partitions.append(canonical_partition)
            component_counts.append(component_count)
        if seen != set(group_by_key):
            raise ValueError("MH1 CUDA forward policy omits an output block")
        canonical_entries.append(
            {
                "index": index,
                "strategy": strategy,
                "partitions": canonical_partitions,
                "component_counts": component_counts,
            }
        )
        expected_entry = next(
            item
            for item in _make_forward_policy(normalized, compute_capability, strategy)[
                "interactions"
            ]
            if item["index"] == index
        )
        if entry != expected_entry:
            raise ValueError(
                "MH1 CUDA forward policy does not match its canonical strategy"
            )
    payload = {
        "tag": MH1_CUDA_FORWARD_POLICY_TAG,
        "generation_fingerprint": normalized["generation_fingerprint"],
        "target_compute_capability": compute_capability,
        "interactions": canonical_entries,
    }
    policy_id = _policy_digest(payload)
    if candidate["policy_id"] != policy_id:
        raise ValueError("MH1 CUDA forward policy identity is stale")
    return {**payload, "policy_id": policy_id}


def resolve_execution_mh1_cuda_forward_policy(
    contract: dict[str, Any],
    compute_capability: int,
    request: str | dict[str, Any] = "auto",
) -> dict[str, Any]:
    """Resolve a canonical, model-bound CUDA receiver partition policy."""

    _validate_compute_capability(compute_capability)
    normalized = normalize_execution_mh1_contract(contract)
    if isinstance(request, str):
        return _make_forward_policy(normalized, compute_capability, request)
    if not isinstance(request, dict):
        raise TypeError("MH1 CUDA forward policy must be a string or mapping")
    return _validate_forward_policy(normalized, compute_capability, request)


def _input_group_key(group: list[dict[str, Any]]) -> tuple[int, int]:
    input_block = group[0]["input_1"]
    return int(input_block["offset"]), int(input_block["components"])


def _make_source_policy(
    normalized: dict[str, Any], compute_capability: int, request: str
) -> dict[str, Any]:
    if request not in ("auto", "budget", "grouped", "full"):
        raise ValueError(
            "MH1 CUDA source policy must be 'auto', 'budget', 'grouped', or " "'full'"
        )
    interaction_policies = []
    for interaction in normalized["interactions"]:
        groups = _group_interaction_paths(interaction, "input_1")
        strategy = request
        if strategy == "auto":
            strategy = "full" if len(groups) == 1 else "grouped"
        if strategy == "full":
            partition_indices = [list(range(len(groups)))]
        elif strategy == "grouped":
            partition_indices = [[group] for group in range(len(groups))]
        else:
            partition_indices = _batch_interaction_group_indices(
                groups,
                "input_1",
                _CUDA_SOURCE_COMPONENT_BUDGET,
            )
        full_partition = [list(range(len(groups)))]
        budget_partitions = _batch_interaction_group_indices(
            groups,
            "input_1",
            _CUDA_SOURCE_COMPONENT_BUDGET,
        )
        if partition_indices == full_partition:
            strategy = "full"
        elif partition_indices == budget_partitions:
            strategy = "budget"
        else:
            strategy = "grouped"
        interaction_policies.append(
            {
                "index": int(interaction["index"]),
                "strategy": strategy,
                "partitions": [
                    [list(_input_group_key(groups[group])) for group in partition]
                    for partition in partition_indices
                ],
                "component_counts": [
                    sum(
                        int(groups[group][0]["input_1"]["components"])
                        for group in partition
                    )
                    for partition in partition_indices
                ],
            }
        )
    payload = {
        "tag": MH1_CUDA_SOURCE_POLICY_TAG,
        "generation_fingerprint": normalized["generation_fingerprint"],
        "target_compute_capability": compute_capability,
        "interactions": interaction_policies,
    }
    return {**payload, "policy_id": _policy_digest(payload)}


def _validate_source_policy(
    normalized: dict[str, Any], compute_capability: int, policy: dict[str, Any]
) -> dict[str, Any]:
    candidate = json.loads(json.dumps(policy))
    required_fields = {
        "tag",
        "generation_fingerprint",
        "target_compute_capability",
        "interactions",
        "policy_id",
    }
    if set(candidate) != required_fields:
        raise ValueError("MH1 CUDA source policy fields are invalid")
    if candidate.get("tag") != MH1_CUDA_SOURCE_POLICY_TAG:
        raise ValueError("MH1 CUDA source policy tag is unsupported")
    if candidate.get("generation_fingerprint") != normalized["generation_fingerprint"]:
        raise ValueError("MH1 CUDA source policy model identity does not match")
    if candidate.get("target_compute_capability") != compute_capability:
        raise ValueError("MH1 CUDA source policy target does not match CUDA target")
    entries = candidate.get("interactions")
    if not isinstance(entries, list) or len(entries) != len(normalized["interactions"]):
        raise ValueError("MH1 CUDA source policy interaction count is invalid")
    canonical_entries = []
    for interaction, entry in zip(normalized["interactions"], entries, strict=True):
        index = int(interaction["index"])
        if not isinstance(entry, dict) or entry.get("index") != index:
            raise ValueError("MH1 CUDA source policy interaction index is invalid")
        if set(entry) != {
            "index",
            "strategy",
            "partitions",
            "component_counts",
        }:
            raise ValueError("MH1 CUDA source policy interaction fields are invalid")
        strategy = entry.get("strategy")
        if strategy not in ("budget", "grouped", "full"):
            raise ValueError("MH1 CUDA source policy strategy is invalid")
        groups = _group_interaction_paths(interaction, "input_1")
        group_by_key = {_input_group_key(group): group for group in groups}
        partitions = entry.get("partitions")
        if not isinstance(partitions, list) or not partitions:
            raise ValueError("MH1 CUDA source policy partitions are empty")
        seen: set[tuple[int, int]] = set()
        canonical_partitions = []
        component_counts = []
        for partition in partitions:
            if not isinstance(partition, list) or not partition:
                raise ValueError("MH1 CUDA source policy contains an empty partition")
            canonical_partition = []
            component_count = 0
            for raw_key in partition:
                if (
                    not isinstance(raw_key, list)
                    or len(raw_key) != 2
                    or any(
                        isinstance(value, bool) or not isinstance(value, int)
                        for value in raw_key
                    )
                ):
                    raise ValueError("MH1 CUDA source policy block key is invalid")
                key = (raw_key[0], raw_key[1])
                if key not in group_by_key:
                    raise ValueError(
                        "MH1 CUDA source policy block is not in the contract"
                    )
                if key in seen:
                    raise ValueError("MH1 CUDA source policy repeats an input block")
                seen.add(key)
                canonical_partition.append(list(key))
                component_count += key[1]
            canonical_partitions.append(canonical_partition)
            component_counts.append(component_count)
        if seen != set(group_by_key):
            raise ValueError("MH1 CUDA source policy omits an input block")
        canonical_entries.append(
            {
                "index": index,
                "strategy": strategy,
                "partitions": canonical_partitions,
                "component_counts": component_counts,
            }
        )
        expected_entry = next(
            item
            for item in _make_source_policy(normalized, compute_capability, strategy)[
                "interactions"
            ]
            if item["index"] == index
        )
        if entry != expected_entry:
            raise ValueError(
                "MH1 CUDA source policy does not match its canonical strategy"
            )
    payload = {
        "tag": MH1_CUDA_SOURCE_POLICY_TAG,
        "generation_fingerprint": normalized["generation_fingerprint"],
        "target_compute_capability": compute_capability,
        "interactions": canonical_entries,
    }
    policy_id = _policy_digest(payload)
    if candidate["policy_id"] != policy_id:
        raise ValueError("MH1 CUDA source policy identity is stale")
    return {**payload, "policy_id": policy_id}


def resolve_execution_mh1_cuda_source_policy(
    contract: dict[str, Any],
    compute_capability: int,
    request: str | dict[str, Any] = "auto",
) -> dict[str, Any]:
    """Resolve a canonical, model-bound CUDA source partition policy."""

    _validate_compute_capability(compute_capability)
    normalized = normalize_execution_mh1_contract(contract)
    if isinstance(request, str):
        return _make_source_policy(normalized, compute_capability, request)
    if not isinstance(request, dict):
        raise TypeError("MH1 CUDA source policy must be a string or mapping")
    return _validate_source_policy(normalized, compute_capability, request)


def _make_edge_policy(
    normalized: dict[str, Any],
    compute_capability: int,
    request: str,
    reverse_schedule: str,
) -> dict[str, Any]:
    if request not in ("auto", "compact_fused", "split", "fused"):
        raise ValueError(
            "MH1 CUDA edge policy must be 'auto', 'compact_fused', 'split', or 'fused'"
        )
    strategy = "compact_fused" if request == "auto" else request
    payload = {
        "tag": MH1_CUDA_EDGE_POLICY_TAG,
        "generation_fingerprint": normalized["generation_fingerprint"],
        "target_compute_capability": compute_capability,
        "reverse_schedule": reverse_schedule,
        "interactions": [
            {
                "index": int(interaction["index"]),
                "strategy": strategy,
                "physical_launch_count": 2 if strategy == "split" else 1,
            }
            for interaction in normalized["interactions"]
        ],
    }
    return {**payload, "policy_id": _policy_digest(payload)}


def _validate_edge_policy(
    normalized: dict[str, Any],
    compute_capability: int,
    policy: dict[str, Any],
    reverse_schedule: str,
) -> dict[str, Any]:
    candidate = json.loads(json.dumps(policy))
    if set(candidate) != {
        "tag",
        "generation_fingerprint",
        "target_compute_capability",
        "reverse_schedule",
        "interactions",
        "policy_id",
    }:
        raise ValueError("MH1 CUDA edge policy fields are invalid")
    if candidate.get("tag") != MH1_CUDA_EDGE_POLICY_TAG:
        raise ValueError("MH1 CUDA edge policy tag is unsupported")
    if candidate.get("generation_fingerprint") != normalized["generation_fingerprint"]:
        raise ValueError("MH1 CUDA edge policy model identity does not match")
    if candidate.get("target_compute_capability") != compute_capability:
        raise ValueError("MH1 CUDA edge policy target does not match CUDA target")
    if candidate.get("reverse_schedule") != reverse_schedule:
        raise ValueError("MH1 CUDA edge policy reverse schedule is unsupported")
    entries = candidate.get("interactions")
    if not isinstance(entries, list) or len(entries) != len(normalized["interactions"]):
        raise ValueError("MH1 CUDA edge policy interaction count is invalid")
    canonical_entries = []
    for interaction, entry in zip(normalized["interactions"], entries, strict=True):
        index = int(interaction["index"])
        if not isinstance(entry, dict) or set(entry) != {
            "index",
            "strategy",
            "physical_launch_count",
        }:
            raise ValueError("MH1 CUDA edge policy interaction fields are invalid")
        if entry.get("index") != index:
            raise ValueError("MH1 CUDA edge policy interaction index is invalid")
        strategy = entry.get("strategy")
        if strategy not in ("compact_fused", "split", "fused"):
            raise ValueError("MH1 CUDA edge policy strategy is invalid")
        expected_count = 2 if strategy == "split" else 1
        if entry.get("physical_launch_count") != expected_count:
            raise ValueError("MH1 CUDA edge policy launch count is invalid")
        canonical_entries.append(
            {
                "index": index,
                "strategy": strategy,
                "physical_launch_count": expected_count,
            }
        )
    payload = {
        "tag": MH1_CUDA_EDGE_POLICY_TAG,
        "generation_fingerprint": normalized["generation_fingerprint"],
        "target_compute_capability": compute_capability,
        "reverse_schedule": reverse_schedule,
        "interactions": canonical_entries,
    }
    policy_id = _policy_digest(payload)
    if candidate["policy_id"] != policy_id:
        raise ValueError("MH1 CUDA edge policy identity is stale")
    return {**payload, "policy_id": policy_id}


def resolve_execution_mh1_cuda_edge_policy(
    contract: dict[str, Any],
    compute_capability: int,
    request: str | dict[str, Any] = "auto",
    *,
    _reverse_schedule: str = _MH1_CUDA_EDGE_REVERSE_SCHEDULE,
) -> dict[str, Any]:
    """Resolve a canonical, model-bound CUDA edge-reverse launch policy."""

    _validate_compute_capability(compute_capability)
    normalized = normalize_execution_mh1_contract(contract)
    if isinstance(request, str):
        return _make_edge_policy(
            normalized, compute_capability, request, _reverse_schedule
        )
    if not isinstance(request, dict):
        raise TypeError("MH1 CUDA edge policy must be a string or mapping")
    return _validate_edge_policy(
        normalized, compute_capability, request, _reverse_schedule
    )


def _mh1_cuda_program_target_schedule(
    contract: dict[str, Any],
    compute_capability: int,
    *,
    abi_version: Literal[3, 4],
    forward_policy: str | dict[str, Any],
    source_policy: str | dict[str, Any],
    edge_policy: str | dict[str, Any],
    edge_reverse_schedule: str = _MH1_CUDA_EDGE_REVERSE_SCHEDULE,
) -> tuple[MH1Program, MH1GpuTarget, MH1KernelSchedule]:
    program = MH1Program.from_contract(contract, abi_version=abi_version)
    policy_contract = (
        program.contract
        if abi_version == 3
        else project_execution_mh1_v3_contract(program.contract)
    )
    resolved_forward = resolve_execution_mh1_cuda_forward_policy(
        policy_contract, compute_capability, forward_policy
    )
    resolved_source = resolve_execution_mh1_cuda_source_policy(
        policy_contract, compute_capability, source_policy
    )
    resolved_edge = resolve_execution_mh1_cuda_edge_policy(
        policy_contract,
        compute_capability,
        edge_policy,
        _reverse_schedule=edge_reverse_schedule,
    )
    target = MH1GpuTarget(1, "cuda", f"sm_{compute_capability}", 32, 16, 32)
    schedule = MH1KernelSchedule(
        1,
        _canonical_json(resolved_forward),
        _canonical_json(resolved_source),
        _canonical_json(resolved_edge),
        128,
        128,
        128,
    )
    return program, target, schedule


def _forward_partition_paths(
    interaction: dict[str, Any], interaction_policy: dict[str, Any]
) -> list[list[dict[str, Any]]]:
    groups = _group_interaction_paths(interaction, "output")
    group_by_key = {_output_group_key(group): group for group in groups}
    return [
        [
            path
            for raw_key in partition
            for path in group_by_key[(int(raw_key[0]), int(raw_key[1]))]
        ]
        for partition in interaction_policy["partitions"]
    ]


def _render_cuda_forward_partitions(
    interaction: dict[str, Any], interaction_policy: dict[str, Any]
) -> list[str]:
    index = int(interaction["index"])
    output_mask_is_identity = _interaction_output_mask_is_identity(interaction)
    functions = []
    for partition, paths in enumerate(
        _forward_partition_paths(interaction, interaction_policy)
    ):
        grouped_interaction = {**interaction, "paths": paths}
        function = _render_cuda_device_function(
            _render_host_forward(
                grouped_interaction,
                channel_owner=True,
                phi_major_linear_weight=True,
                output_mask_is_identity=output_mask_is_identity,
            )
        )
        functions.append(
            function.replace(
                f"void forward_{index}(",
                f"void forward_{index}_partition_{partition}(",
                1,
            )
        )
    return functions


def _source_partition_paths(
    interaction: dict[str, Any], interaction_policy: dict[str, Any]
) -> list[list[dict[str, Any]]]:
    groups = _group_interaction_paths(interaction, "input_1")
    group_by_key = {_input_group_key(group): group for group in groups}
    return [
        [
            path
            for raw_key in partition
            for path in group_by_key[(int(raw_key[0]), int(raw_key[1]))]
        ]
        for partition in interaction_policy["partitions"]
    ]


def _render_cuda_source_reverse_partitions(
    interaction: dict[str, Any],
    interaction_policy: dict[str, Any],
    *,
    path_weight_tile_size: int = 1,
) -> list[str]:
    index = int(interaction["index"])
    output_mask_is_identity = _interaction_output_mask_is_identity(interaction)
    functions = []
    for partition, paths in enumerate(
        _source_partition_paths(interaction, interaction_policy)
    ):
        grouped_interaction = {**interaction, "paths": paths}
        function = _render_cuda_device_function(
            _render_host_source_reverse(
                grouped_interaction,
                channel_owner=True,
                phi_major_linear_weight=True,
                output_mask_is_identity=output_mask_is_identity,
                path_weight_tile_size=path_weight_tile_size,
            )
        )
        functions.append(
            function.replace(
                f"void source_reverse_{index}(",
                f"void source_reverse_{index}_partition_{partition}(",
                1,
            )
        )
    return functions


def _render_cuda_edge_reverse_phi(interaction: dict[str, Any]) -> str:
    index = int(interaction["index"])
    extents = _interaction_extents(interaction)
    path_blocks = []
    for path in interaction["paths"]:
        source_block = path["input_1"]
        edge_block = path["input_2"]
        output = path["output"]
        statements = ["float weight_adjoint = 0.0f;"]
        for term in path["sparse_wigner"]["terms"]:
            source_index = _ir_mul_index(source_block, int(term["a"]))
            harmonic_index = int(edge_block["offset"]) + int(term["b"])
            output_index = _ir_mul_index(output, int(term["c"]))
            common = (
                f"{_cpp_float(path['path_weight'])}"
                f" * {_cpp_float(term['coefficient'])}"
                f" * args->source_node_values[source * {extents['input_1_dimension']}"
                f" + {source_index}]"
                f" * args->target_node_output_adjoint["
                f"target * {extents['output_dimension']} + {output_index}]"
                f"{_output_mask_factor(interaction, output_index)}"
            )
            statements.append(
                f"weight_adjoint += ({common})"
                f" * args->edge_input_2[local_edge * {extents['input_2_dimension']}"
                f" + {harmonic_index}];"
            )
        weight_index = f"{int(path['weight_offset'])} + channel"
        statements.append(
            f"for (std::int32_t phi = 0; phi < {extents['phi_dimension']}; ++phi) {{\n"
            f"                phi_adjoint[phi] += weight_adjoint * cutoff"
            f" * args->linear_weight[phi * {extents['weight_size']}"
            f" + ({weight_index})];\n"
            f"            }}"
        )
        path_blocks.append(
            "{\n            " + "\n            ".join(statements) + "\n            }"
        )
    return f"""__device__ __forceinline__ void edge_reverse_phi_{index}(
    const SymmetrixJitMH1CudaReverseArgsV3* args,
    std::int32_t local_edge,
    std::int32_t edge_lane,
    bool edge_active) noexcept
{{
    float phi_adjoint[{extents['phi_dimension']}] = {{0.0f}};
    if (edge_active) {{
        const std::int64_t edge = args->first_edge + local_edge;
        const std::int32_t source = args->source_indices[edge];
        const std::int32_t target = args->target_indices[edge];
        const bool apply_cutoff =
            (args->flags & SYMMETRIX_JIT_MH1_CUDA_APPLY_EDGE_CUTOFF_V3) != 0;
        const float cutoff =
            apply_cutoff ? args->edge_cutoff_scale[edge] : 1.0f;
        for (std::int32_t channel = edge_lane;
             channel < {extents['multiplicity']}; channel += 16) {{
            {chr(10).join(path_blocks)}
        }}
    }}
    #pragma unroll
    for (std::int32_t offset = 8; offset > 0; offset /= 2) {{
        #pragma unroll
        for (std::int32_t phi = 0; phi < {extents['phi_dimension']}; ++phi) {{
            phi_adjoint[phi] += SYMMETRIX_JIT_MH1_SHUFFLE_DOWN(
                phi_adjoint[phi], offset, 16);
        }}
    }}
    if (edge_lane == 0 && edge_active) {{
        #pragma unroll
        for (std::int32_t phi = 0; phi < {extents['phi_dimension']}; ++phi) {{
            args->edge_phi_adjoint[
                local_edge * {extents['phi_dimension']} + phi] = phi_adjoint[phi];
        }}
    }}
}}"""


def _edge_phi_schedule(interaction: dict[str, Any]) -> str:
    sparse_terms = sum(
        len(path["sparse_wigner"]["terms"]) for path in interaction["paths"]
    )
    return (
        "path_tiled"
        if sparse_terms > 32 or len(interaction["paths"]) > 6
        else "subgroup"
    )


def _render_cuda_edge_reverse_phi_path_tiled_kernel(
    interaction: dict[str, Any], *, symbol_prefix: str = ""
) -> str:
    index = int(interaction["index"])
    extents = _interaction_extents(interaction)
    paths = interaction["paths"]
    edge_batch = _MH1_PATH_TILED_EDGE_EDGE_BATCH
    batch_rows = _MH1_PATH_TILED_EDGE_PATH_BATCH
    path_batches = []
    for batch_start in range(0, len(paths), batch_rows):
        weight_blocks = []
        contribution_blocks = []
        for batch_row, path in enumerate(paths[batch_start : batch_start + batch_rows]):
            source_block = path["input_1"]
            edge_block = path["input_2"]
            output = path["output"]
            weight_statements = [
                "for (std::int32_t channel = threadIdx.x;",
                f"     channel < {extents['multiplicity']}; channel += blockDim.x) {{",
            ]
            for edge_slot in range(edge_batch):
                weight_statements.extend(
                    (
                        f"    float weight_adjoint_{edge_slot} = 0.0f;",
                        f"    if (edge_active_{edge_slot}) {{",
                    )
                )
                for term in path["sparse_wigner"]["terms"]:
                    source_index = _ir_mul_index(source_block, int(term["a"]))
                    harmonic_index = int(edge_block["offset"]) + int(term["b"])
                    output_index = _ir_mul_index(output, int(term["c"]))
                    common = (
                        f"{_cpp_float(path['path_weight'])}"
                        f" * {_cpp_float(term['coefficient'])}"
                        f" * args->source_node_values["
                        f"source_{edge_slot} * {extents['input_1_dimension']}"
                        f" + {source_index}]"
                        f" * args->target_node_output_adjoint["
                        f"target_{edge_slot} * {extents['output_dimension']}"
                        f" + {output_index}]"
                        f"{_output_mask_factor(interaction, output_index)}"
                    )
                    weight_statements.append(
                        f"        weight_adjoint_{edge_slot} += ({common})"
                        f" * args->edge_input_2["
                        f"local_edge_{edge_slot} * {extents['input_2_dimension']}"
                        f" + {harmonic_index}];"
                    )
                weight_statements.extend(
                    (
                        "    }",
                        (
                            f"    path_weight_adjoint[{batch_row}]"
                            f"[{edge_slot}][channel] = weight_adjoint_{edge_slot};"
                        ),
                    )
                )
            weight_statements.append("}")
            weight_blocks.append(
                "{\n            "
                + "\n            ".join(weight_statements)
                + "\n            }"
            )

            weight_index = f"{int(path['weight_offset'])} + channel"
            contribution_declarations = "\n                ".join(
                f"float contribution_{edge_slot} = 0.0f;"
                for edge_slot in range(edge_batch)
            )
            contribution_updates = "\n                    ".join(
                f"contribution_{edge_slot} +="
                f" path_weight_adjoint[{batch_row}][{edge_slot}][channel]"
                " * linear_weight;"
                for edge_slot in range(edge_batch)
            )
            contribution_reductions = "\n                    ".join(
                f"contribution_{edge_slot} +="
                " SYMMETRIX_JIT_MH1_SHUFFLE_DOWN("
                f"contribution_{edge_slot}, offset, 32);"
                for edge_slot in range(edge_batch)
            )
            phi_updates = "\n                    ".join(
                f"phi_adjoint[{edge_slot}][phi] +="
                f" contribution_{edge_slot} * cutoff_{edge_slot};"
                for edge_slot in range(edge_batch)
            )
            contribution_blocks.append(
                f"""{{\n                {contribution_declarations}\n                for (std::int32_t channel = lane;\n                     channel < {extents["multiplicity"]}; channel += 32) {{\n                    const float linear_weight = args->linear_weight[\n                        phi * {extents["weight_size"]} + ({weight_index})];\n                    {contribution_updates}\n                }}\n                #pragma unroll\n                for (std::int32_t offset = 16; offset > 0; offset /= 2) {{\n                    {contribution_reductions}\n                }}\n                if (lane == 0) {{\n                    {phi_updates}\n                }}\n            }}"""
            )
        path_batches.append(
            f"""{chr(10).join(weight_blocks)}\n        __syncthreads();\n        for (std::int32_t phi = warp; phi < {extents["phi_dimension"]};\n             phi += warps_per_block) {{\n            {chr(10).join(contribution_blocks)}\n        }}\n        __syncthreads();"""
        )

    edge_initializers = []
    phi_initializers = []
    output_stores = []
    for edge_slot in range(edge_batch):
        local_edge = (
            "edge_base"
            if edge_slot == 0
            else f"edge_base + static_cast<std::int64_t>({edge_slot})"
        )
        edge_initializers.append(
            f"""const std::int64_t local_edge_{edge_slot} = {local_edge};\n        const bool edge_active_{edge_slot} =\n            local_edge_{edge_slot} < packet.samples;\n        const std::int64_t edge_{edge_slot} = packet.first_edge\n            + (edge_active_{edge_slot} ? local_edge_{edge_slot} : 0);\n        const std::int32_t source_{edge_slot} = edge_active_{edge_slot}\n            ? packet.source_indices[edge_{edge_slot}] : 0;\n        const std::int32_t target_{edge_slot} = edge_active_{edge_slot}\n            ? packet.target_indices[edge_{edge_slot}] : 0;\n        const float cutoff_{edge_slot} = edge_active_{edge_slot} && apply_cutoff\n            ? packet.edge_cutoff_scale[edge_{edge_slot}] : 1.0f;"""
        )
        phi_initializers.append(
            f"""for (std::int32_t phi = threadIdx.x;\n             phi < {extents["phi_dimension"]}; phi += blockDim.x)\n            phi_adjoint[{edge_slot}][phi] = 0.0f;"""
        )
        output_stores.append(
            f"""if (edge_active_{edge_slot}) {{\n            for (std::int32_t phi = threadIdx.x;\n                 phi < {extents["phi_dimension"]}; phi += blockDim.x) {{\n                args->edge_phi_adjoint[\n                    local_edge_{edge_slot} * {extents["phi_dimension"]} + phi]\n                    = phi_adjoint[{edge_slot}][phi];\n            }}\n        }}"""
        )

    linkage = 'extern "C" ' if symbol_prefix else ""
    return f"""{linkage}__global__ void {symbol_prefix}edge_reverse_phi_kernel_{index}(\n    SymmetrixJitMH1CudaReverseArgsV3 packet)\n{{\n    __shared__ float path_weight_adjoint[{batch_rows}][{edge_batch}]\n        [{extents["multiplicity"]}];\n    __shared__ float phi_adjoint[{edge_batch}][{extents["phi_dimension"]}];\n    const std::int32_t lane = threadIdx.x & 31;\n    const std::int32_t warp = threadIdx.x / 32;\n    const std::int32_t warps_per_block = blockDim.x / 32;\n    const bool apply_cutoff =\n        (packet.flags & SYMMETRIX_JIT_MH1_CUDA_APPLY_EDGE_CUTOFF_V3) != 0;\n    const auto* args = &packet;\n    for (std::int64_t edge_base =\n             static_cast<std::int64_t>(blockIdx.x) * {edge_batch};\n         edge_base < packet.samples;\n         edge_base += static_cast<std::int64_t>(gridDim.x) * {edge_batch}) {{\n        {chr(10).join(edge_initializers)}\n        {chr(10).join(phi_initializers)}\n        {chr(10).join(path_batches)}\n        {chr(10).join(output_stores)}\n        __syncthreads();\n    }}\n}}\n
"""


def _render_hip_compact_edge_v6_producers(
    interaction: dict[str, Any],
    path: dict[str, Any],
    extents: dict[str, int],
    batch_row: int,
    edge_batch: int,
) -> list[str]:
    source_block = path["input_1"]
    edge_block = path["input_2"]
    output = path["output"]
    components = int(edge_block["components"])
    harmonic_offset = int(edge_block["offset"])
    weight_index = f"{int(path['weight_offset'])} + channel"
    producers = []
    for edge_slot in range(edge_batch):
        declarations = [
            f"float harmonic_contribution_{component} = 0.0f;"
            for component in range(components)
        ]
        declarations.append("float cutoff_contribution = 0.0f;")
        statements = [
            f"float weight = has_bias ? args->linear_bias[{weight_index}] : 0.0f;",
            "float weight_adjoint = 0.0f;",
            f"if (edge_active_{edge_slot}) {{",
            "    if (has_fixed) {",
            "        weight += args->edge_linear_contribution[",
            (
                f"            local_edge_{edge_slot} * {extents['weight_size']}"
                f" + ({weight_index})];"
            ),
            "    }",
            f"    for (std::int32_t phi = 0; phi < {extents['phi_dimension']}; ++phi) {{",
            "        weight += args->linear_weight[",
            f"            phi * {extents['weight_size']} + ({weight_index})]",
            (
                f"            * args->edge_phi[local_edge_{edge_slot}"
                f" * {extents['phi_dimension']} + phi];"
            ),
            "    }",
        ]
        for term in path["sparse_wigner"]["terms"]:
            source_index = _ir_mul_index(source_block, int(term["a"]))
            harmonic_component = int(term["b"])
            harmonic_index = harmonic_offset + harmonic_component
            output_index = _ir_mul_index(output, int(term["c"]))
            common = (
                f"{_cpp_float(path['path_weight'])}"
                f" * {_cpp_float(term['coefficient'])}"
                f" * args->source_node_values["
                f"source_{edge_slot} * {extents['input_1_dimension']}"
                f" + {source_index}]"
                " * args->target_node_output_adjoint["
                f"target_{edge_slot} * {extents['output_dimension']}"
                f" + {output_index}]"
                f"{_output_mask_factor(interaction, output_index)}"
            )
            statements.append(
                f"    weight_adjoint += ({common})"
                f" * args->edge_input_2[local_edge_{edge_slot}"
                f" * {extents['input_2_dimension']} + {harmonic_index}];"
            )
            statements.append(
                f"    harmonic_contribution_{harmonic_component}"
                f" += ({common}) * weight * cutoff_{edge_slot};"
            )
        statements.extend(
            (
                "}",
                (
                    f"path_weight_adjoint[{batch_row}][{edge_slot}][channel]"
                    " = weight_adjoint;"
                ),
                (
                    "if (apply_cutoff) cutoff_contribution"
                    " += weight_adjoint * weight;"
                ),
            )
        )
        reductions = [
            f"harmonic_contribution_{component} += "
            "SYMMETRIX_JIT_MH1_SHUFFLE_DOWN("
            f"harmonic_contribution_{component}, offset, 32);"
            for component in range(components)
        ]
        reductions.append(
            "cutoff_contribution += SYMMETRIX_JIT_MH1_SHUFFLE_DOWN("
            "cutoff_contribution, offset, 32);"
        )
        partial_stores = [
            f"harmonic_warp_partial[{edge_slot}][warp][{component}]"
            f" = harmonic_contribution_{component};"
            for component in range(components)
        ]
        partial_stores.append(
            f"cutoff_warp_partial[{edge_slot}][warp] = cutoff_contribution;"
        )
        producers.append(
            f"""{{
            {chr(10).join(declarations)}
            for (std::int32_t channel = threadIdx.x;
                 channel < {extents["multiplicity"]}; channel += blockDim.x) {{
                {chr(10).join(statements)}
            }}
            #pragma unroll
            for (std::int32_t offset = 16; offset > 0; offset /= 2) {{
                {chr(10).join(reductions)}
            }}
            if (lane == 0) {{
                {chr(10).join(partial_stores)}
            }}
        }}"""
        )
    return producers


def _render_cuda_edge_reverse_compact_fused_path_tiled_kernel(
    interaction: dict[str, Any],
    *,
    symbol_prefix: str = "",
    reverse_schedule: str = _MH1_CUDA_EDGE_REVERSE_SCHEDULE,
) -> str:
    if reverse_schedule not in (
        _MH1_CUDA_EDGE_REVERSE_SCHEDULE,
        _MH1_HIP_EDGE_REVERSE_SCHEDULE,
    ):
        raise ValueError("unsupported MH-1 compact-edge reverse schedule")
    index = int(interaction["index"])
    extents = _interaction_extents(interaction)
    paths = interaction["paths"]
    edge_batch = _MH1_PATH_TILED_EDGE_EDGE_BATCH
    batch_rows = _MH1_PATH_TILED_EDGE_PATH_BATCH
    max_path_components = max(int(path["input_2"]["components"]) for path in paths)

    path_batches = []
    for batch_start in range(0, len(paths), batch_rows):
        producer_blocks = []
        contribution_blocks = []
        for batch_row, path in enumerate(paths[batch_start : batch_start + batch_rows]):
            source_block = path["input_1"]
            edge_block = path["input_2"]
            output = path["output"]
            components = int(edge_block["components"])
            harmonic_offset = int(edge_block["offset"])
            weight_index = f"{int(path['weight_offset'])} + channel"
            contribution_declarations = []
            weight_declarations = []
            fixed_weight_updates = []
            weight_phi_updates = []
            contraction_blocks = []
            contribution_reductions = []
            partial_stores = []
            edge_reduction_blocks = []
            for edge_slot in range(edge_batch):
                contribution_declarations.extend(
                    f"float harmonic_contribution_{edge_slot}_{component} = 0.0f;"
                    for component in range(components)
                )
                contribution_declarations.append(
                    f"float cutoff_contribution_{edge_slot} = 0.0f;"
                )
                weight_declarations.extend(
                    (
                        (
                            f"float weight_{edge_slot} = has_bias"
                            f" ? args->linear_bias[{weight_index}] : 0.0f;"
                        ),
                        f"float weight_adjoint_{edge_slot} = 0.0f;",
                    )
                )
                fixed_weight_updates.append(
                    f"if (edge_active_{edge_slot} && has_fixed) {{\n"
                    f"    weight_{edge_slot} += args->edge_linear_contribution[\n"
                    f"        local_edge_{edge_slot} * {extents['weight_size']}"
                    f" + ({weight_index})];\n"
                    "}"
                )
                weight_phi_updates.append(
                    f"if (edge_active_{edge_slot})\n"
                    f"    weight_{edge_slot} += linear_weight\n"
                    f"        * args->edge_phi[local_edge_{edge_slot}"
                    f" * {extents['phi_dimension']} + phi];"
                )
                statements = [f"if (edge_active_{edge_slot}) {{"]
                for term in path["sparse_wigner"]["terms"]:
                    source_index = _ir_mul_index(source_block, int(term["a"]))
                    harmonic_component = int(term["b"])
                    harmonic_index = harmonic_offset + harmonic_component
                    output_index = _ir_mul_index(output, int(term["c"]))
                    common = (
                        f"{_cpp_float(path['path_weight'])}"
                        f" * {_cpp_float(term['coefficient'])}"
                        f" * args->source_node_values["
                        f"source_{edge_slot} * {extents['input_1_dimension']}"
                        f" + {source_index}]"
                        " * args->target_node_output_adjoint["
                        f"target_{edge_slot} * {extents['output_dimension']}"
                        f" + {output_index}]"
                        f"{_output_mask_factor(interaction, output_index)}"
                    )
                    statements.append(
                        f"    weight_adjoint_{edge_slot} += ({common})"
                        f" * args->edge_input_2[local_edge_{edge_slot}"
                        f" * {extents['input_2_dimension']} + {harmonic_index}];"
                    )
                    statements.append(
                        f"    harmonic_contribution_{edge_slot}_{harmonic_component}"
                        f" += ({common}) * weight_{edge_slot} * cutoff_{edge_slot};"
                    )
                statements.extend(
                    (
                        "}",
                        (
                            f"path_weight_adjoint[{batch_row}][{edge_slot}][channel]"
                            f" = weight_adjoint_{edge_slot};"
                        ),
                        (
                            f"if (apply_cutoff) cutoff_contribution_{edge_slot}"
                            f" += weight_adjoint_{edge_slot} * weight_{edge_slot};"
                        ),
                    )
                )
                contraction_blocks.append("\n".join(statements))
                contribution_reductions.extend(
                    f"harmonic_contribution_{edge_slot}_{component} += "
                    "SYMMETRIX_JIT_MH1_SHUFFLE_DOWN("
                    f"harmonic_contribution_{edge_slot}_{component}, offset, 32);"
                    for component in range(components)
                )
                contribution_reductions.append(
                    f"cutoff_contribution_{edge_slot} += "
                    "SYMMETRIX_JIT_MH1_SHUFFLE_DOWN("
                    f"cutoff_contribution_{edge_slot}, offset, 32);"
                )
                partial_stores.extend(
                    f"harmonic_warp_partial[{edge_slot}][warp][{component}]"
                    f" = harmonic_contribution_{edge_slot}_{component};"
                    for component in range(components)
                )
                partial_stores.append(
                    f"cutoff_warp_partial[{edge_slot}][warp]"
                    f" = cutoff_contribution_{edge_slot};"
                )
                edge_reduction_blocks.append(
                    f"""{{
            for (std::int32_t owner = threadIdx.x;
                 owner < {components}; owner += blockDim.x) {{
                float value = 0.0f;
                #pragma unroll
                for (std::int32_t source_warp = 0;
                     source_warp < warps_per_block; ++source_warp)
                    value += harmonic_warp_partial[{edge_slot}][source_warp][owner];
                harmonic_adjoint[{edge_slot}][{harmonic_offset} + owner] += value;
            }}
            if (threadIdx.x == 0) {{
                float value = 0.0f;
                #pragma unroll
                for (std::int32_t source_warp = 0;
                     source_warp < warps_per_block; ++source_warp)
                    value += cutoff_warp_partial[{edge_slot}][source_warp];
                cutoff_adjoint[{edge_slot}] += value;
            }}
        }}"""
                )

            if reverse_schedule == _MH1_HIP_EDGE_REVERSE_SCHEDULE:
                producer_scopes = _render_hip_compact_edge_v6_producers(
                    interaction, path, extents, batch_row, edge_batch
                )
            else:
                producer_scopes = [
                    f"""{{
            {chr(10).join(contribution_declarations)}
            for (std::int32_t channel = threadIdx.x;
                 channel < {extents["multiplicity"]}; channel += blockDim.x) {{
                {chr(10).join(weight_declarations)}
                {chr(10).join(fixed_weight_updates)}
                for (std::int32_t phi = 0;
                     phi < {extents["phi_dimension"]}; ++phi) {{
                    const float linear_weight = args->linear_weight[
                        phi * {extents["weight_size"]} + ({weight_index})];
                    {chr(10).join(weight_phi_updates)}
                }}
                {chr(10).join(contraction_blocks)}
            }}
            #pragma unroll
            for (std::int32_t offset = 16; offset > 0; offset /= 2) {{
                {chr(10).join(contribution_reductions)}
            }}
            if (lane == 0) {{
                {chr(10).join(partial_stores)}
            }}
        }}"""
                ]
            producer_blocks.append(
                f"""{chr(10).join(producer_scopes)}
        __syncthreads();
        {chr(10).join(edge_reduction_blocks)}
        __syncthreads();"""
            )

            contribution_declarations = "\n                ".join(
                f"float contribution_{edge_slot} = 0.0f;"
                for edge_slot in range(edge_batch)
            )
            contribution_updates = "\n                    ".join(
                f"contribution_{edge_slot} +="
                f" path_weight_adjoint[{batch_row}][{edge_slot}][channel]"
                " * linear_weight;"
                for edge_slot in range(edge_batch)
            )
            contribution_reductions = "\n                    ".join(
                f"contribution_{edge_slot} +="
                " SYMMETRIX_JIT_MH1_SHUFFLE_DOWN("
                f"contribution_{edge_slot}, offset, 32);"
                for edge_slot in range(edge_batch)
            )
            phi_updates = "\n                    ".join(
                f"phi_adjoint[{edge_slot}][phi] +="
                f" contribution_{edge_slot} * cutoff_{edge_slot};"
                for edge_slot in range(edge_batch)
            )
            contribution_blocks.append(
                f"""{{
                {contribution_declarations}
                for (std::int32_t channel = lane;
                     channel < {extents["multiplicity"]}; channel += 32) {{
                    const float linear_weight = args->linear_weight[
                        phi * {extents["weight_size"]} + ({weight_index})];
                    {contribution_updates}
                }}
                #pragma unroll
                for (std::int32_t offset = 16; offset > 0; offset /= 2) {{
                    {contribution_reductions}
                }}
                if (lane == 0) {{
                    {phi_updates}
                }}
            }}"""
            )
        path_batches.append(
            f"""{chr(10).join(producer_blocks)}
        for (std::int32_t phi = warp; phi < {extents["phi_dimension"]};
             phi += warps_per_block) {{
            {chr(10).join(contribution_blocks)}
        }}
        __syncthreads();"""
        )

    edge_initializers = []
    output_initializers = []
    output_stores = []
    for edge_slot in range(edge_batch):
        local_edge = (
            "edge_base"
            if edge_slot == 0
            else f"edge_base + static_cast<std::int64_t>({edge_slot})"
        )
        edge_initializers.append(
            f"""const std::int64_t local_edge_{edge_slot} = {local_edge};
        const bool edge_active_{edge_slot} = local_edge_{edge_slot} < packet.samples;
        const std::int64_t edge_{edge_slot} = packet.first_edge
            + (edge_active_{edge_slot} ? local_edge_{edge_slot} : 0);
        const std::int32_t source_{edge_slot} = edge_active_{edge_slot}
            ? packet.source_indices[edge_{edge_slot}] : 0;
        const std::int32_t target_{edge_slot} = edge_active_{edge_slot}
            ? packet.target_indices[edge_{edge_slot}] : 0;
        const float cutoff_{edge_slot} = edge_active_{edge_slot} && apply_cutoff
            ? packet.edge_cutoff_scale[edge_{edge_slot}] : 1.0f;"""
        )
        output_initializers.append(
            f"""for (std::int32_t phi = threadIdx.x;
             phi < {extents["phi_dimension"]}; phi += blockDim.x)
            phi_adjoint[{edge_slot}][phi] = 0.0f;
        for (std::int32_t harmonic = threadIdx.x;
             harmonic < {extents["input_2_dimension"]}; harmonic += blockDim.x)
            harmonic_adjoint[{edge_slot}][harmonic] = 0.0f;
        if (threadIdx.x == {edge_slot}) cutoff_adjoint[{edge_slot}] = 0.0f;"""
        )
        output_stores.append(
            f"""if (edge_active_{edge_slot}) {{
            for (std::int32_t phi = threadIdx.x;
                 phi < {extents["phi_dimension"]}; phi += blockDim.x)
                args->edge_phi_adjoint[
                    local_edge_{edge_slot} * {extents["phi_dimension"]} + phi]
                    = phi_adjoint[{edge_slot}][phi];
            for (std::int32_t harmonic = threadIdx.x;
                 harmonic < {extents["input_2_dimension"]}; harmonic += blockDim.x)
                args->edge_input_2_adjoint[
                    local_edge_{edge_slot} * {extents["input_2_dimension"]} + harmonic]
                    = harmonic_adjoint[{edge_slot}][harmonic];
            if (threadIdx.x == 0)
                args->edge_cutoff_scale_adjoint[local_edge_{edge_slot}]
                    = cutoff_adjoint[{edge_slot}];
        }}"""
        )

    linkage = 'extern "C" ' if symbol_prefix else ""
    return f"""{linkage}__global__ void {symbol_prefix}edge_reverse_compact_fused_kernel_{index}(
    SymmetrixJitMH1CudaReverseArgsV3 packet)
{{
    __shared__ float path_weight_adjoint[{batch_rows}][{edge_batch}]
        [{extents["multiplicity"]}];
    __shared__ float phi_adjoint[{edge_batch}][{extents["phi_dimension"]}];
    __shared__ float harmonic_adjoint[{edge_batch}][{extents["input_2_dimension"]}];
    __shared__ float cutoff_adjoint[{edge_batch}];
    __shared__ float harmonic_warp_partial[{edge_batch}][4][{max_path_components}];
    __shared__ float cutoff_warp_partial[{edge_batch}][4];
    const std::int32_t lane = threadIdx.x & 31;
    const std::int32_t warp = threadIdx.x / 32;
    const std::int32_t warps_per_block = blockDim.x / 32;
    const bool apply_cutoff =
        (packet.flags & SYMMETRIX_JIT_MH1_CUDA_APPLY_EDGE_CUTOFF_V3) != 0;
    const bool has_bias =
        (packet.flags & SYMMETRIX_JIT_MH1_CUDA_HAS_LINEAR_BIAS_V3) != 0;
    const bool has_fixed =
        (packet.flags & SYMMETRIX_JIT_MH1_CUDA_HAS_FIXED_CONTRIBUTION_V3) != 0;
    const auto* args = &packet;
    for (std::int64_t edge_base =
             static_cast<std::int64_t>(blockIdx.x) * {edge_batch};
         edge_base < packet.samples;
         edge_base += static_cast<std::int64_t>(gridDim.x) * {edge_batch}) {{
        {chr(10).join(edge_initializers)}
        {chr(10).join(output_initializers)}
        __syncthreads();
        {chr(10).join(path_batches)}
        {chr(10).join(output_stores)}
        __syncthreads();
    }}
}}
"""


def _render_cuda_edge_reverse_harmonic(interaction: dict[str, Any]) -> str:
    index = int(interaction["index"])
    extents = _interaction_extents(interaction)
    path_blocks = []
    for path in interaction["paths"]:
        source_block = path["input_1"]
        edge_block = path["input_2"]
        output = path["output"]
        statements = [
            _weight_expression(path, extents, phi_major=True),
            "float weight_adjoint = 0.0f;",
        ]
        for term in path["sparse_wigner"]["terms"]:
            source_index = _ir_mul_index(source_block, int(term["a"]))
            harmonic_index = int(edge_block["offset"]) + int(term["b"])
            output_index = _ir_mul_index(output, int(term["c"]))
            common = (
                f"{_cpp_float(path['path_weight'])}"
                f" * {_cpp_float(term['coefficient'])}"
                f" * args->source_node_values[source * {extents['input_1_dimension']}"
                f" + {source_index}]"
                f" * args->target_node_output_adjoint["
                f"target * {extents['output_dimension']} + {output_index}]"
                f"{_output_mask_factor(interaction, output_index)}"
            )
            statements.append(
                f"weight_adjoint += ({common})"
                f" * args->edge_input_2[local_edge * {extents['input_2_dimension']}"
                f" + {harmonic_index}];"
            )
            statements.append(
                f"harmonic_adjoint[{harmonic_index}]"
                f" += ({common}) * weight * cutoff;"
            )
        statements.append(
            "if (apply_cutoff) cutoff_adjoint += weight_adjoint * weight;"
        )
        path_blocks.append(
            "{\n            " + "\n            ".join(statements) + "\n            }"
        )
    return f"""__device__ __forceinline__ void edge_reverse_harmonic_{index}(
    const SymmetrixJitMH1CudaReverseArgsV3* args,
    std::int32_t local_edge,
    std::int32_t edge_lane,
    bool edge_active) noexcept
{{
    float harmonic_adjoint[{extents['input_2_dimension']}] = {{0.0f}};
    float cutoff_adjoint = 0.0f;
    if (edge_active) {{
        const std::int64_t edge = args->first_edge + local_edge;
        const std::int32_t source = args->source_indices[edge];
        const std::int32_t target = args->target_indices[edge];
        const bool apply_cutoff =
            (args->flags & SYMMETRIX_JIT_MH1_CUDA_APPLY_EDGE_CUTOFF_V3) != 0;
        const bool has_bias =
            (args->flags & SYMMETRIX_JIT_MH1_CUDA_HAS_LINEAR_BIAS_V3) != 0;
        const bool has_fixed =
            (args->flags & SYMMETRIX_JIT_MH1_CUDA_HAS_FIXED_CONTRIBUTION_V3) != 0;
        const float cutoff =
            apply_cutoff ? args->edge_cutoff_scale[edge] : 1.0f;
        for (std::int32_t channel = edge_lane;
             channel < {extents['multiplicity']}; channel += 16) {{
            {chr(10).join(path_blocks)}
        }}
    }}
    #pragma unroll
    for (std::int32_t offset = 8; offset > 0; offset /= 2) {{
        #pragma unroll
        for (std::int32_t harmonic = 0;
             harmonic < {extents['input_2_dimension']}; ++harmonic) {{
            harmonic_adjoint[harmonic] += SYMMETRIX_JIT_MH1_SHUFFLE_DOWN(
                harmonic_adjoint[harmonic], offset, 16);
        }}
        cutoff_adjoint += SYMMETRIX_JIT_MH1_SHUFFLE_DOWN(
            cutoff_adjoint, offset, 16);
    }}
    if (edge_lane == 0 && edge_active) {{
        #pragma unroll
        for (std::int32_t harmonic = 0;
             harmonic < {extents['input_2_dimension']}; ++harmonic) {{
            args->edge_input_2_adjoint[
                local_edge * {extents['input_2_dimension']} + harmonic]
                = harmonic_adjoint[harmonic];
        }}
        args->edge_cutoff_scale_adjoint[local_edge] = cutoff_adjoint;
    }}
}}"""


def _render_cuda_edge_reverse_fused(interaction: dict[str, Any]) -> str:
    index = int(interaction["index"])
    extents = _interaction_extents(interaction)
    path_blocks = []
    for path in interaction["paths"]:
        source_block = path["input_1"]
        edge_block = path["input_2"]
        output = path["output"]
        statements = [
            _weight_expression(path, extents, phi_major=True),
            "float weight_adjoint = 0.0f;",
        ]
        for term in path["sparse_wigner"]["terms"]:
            source_index = _ir_mul_index(source_block, int(term["a"]))
            harmonic_index = int(edge_block["offset"]) + int(term["b"])
            output_index = _ir_mul_index(output, int(term["c"]))
            common = (
                f"{_cpp_float(path['path_weight'])}"
                f" * {_cpp_float(term['coefficient'])}"
                f" * args->source_node_values[source * {extents['input_1_dimension']}"
                f" + {source_index}]"
                f" * args->target_node_output_adjoint["
                f"target * {extents['output_dimension']} + {output_index}]"
                f"{_output_mask_factor(interaction, output_index)}"
            )
            statements.append(
                f"weight_adjoint += ({common})"
                f" * args->edge_input_2[local_edge * {extents['input_2_dimension']}"
                f" + {harmonic_index}];"
            )
            statements.append(
                f"harmonic_adjoint[{harmonic_index}]"
                f" += ({common}) * weight * cutoff;"
            )
        weight_index = f"{int(path['weight_offset'])} + channel"
        statements.append(
            f"for (std::int32_t phi = 0; phi < {extents['phi_dimension']}; ++phi) {{\n"
            f"                phi_adjoint[phi] += weight_adjoint * cutoff"
            f" * args->linear_weight[phi * {extents['weight_size']}"
            f" + ({weight_index})];\n"
            f"            }}"
        )
        statements.append(
            "if (apply_cutoff) cutoff_adjoint += weight_adjoint * weight;"
        )
        path_blocks.append(
            "{\n            " + "\n            ".join(statements) + "\n            }"
        )
    return f"""__device__ __forceinline__ void edge_reverse_fused_{index}(
    const SymmetrixJitMH1CudaReverseArgsV3* args,
    std::int32_t local_edge,
    std::int32_t edge_lane,
    bool edge_active) noexcept
{{
    float phi_adjoint[{extents['phi_dimension']}] = {{0.0f}};
    float harmonic_adjoint[{extents['input_2_dimension']}] = {{0.0f}};
    float cutoff_adjoint = 0.0f;
    if (edge_active) {{
        const std::int64_t edge = args->first_edge + local_edge;
        const std::int32_t source = args->source_indices[edge];
        const std::int32_t target = args->target_indices[edge];
        const bool apply_cutoff =
            (args->flags & SYMMETRIX_JIT_MH1_CUDA_APPLY_EDGE_CUTOFF_V3) != 0;
        const bool has_bias =
            (args->flags & SYMMETRIX_JIT_MH1_CUDA_HAS_LINEAR_BIAS_V3) != 0;
        const bool has_fixed =
            (args->flags & SYMMETRIX_JIT_MH1_CUDA_HAS_FIXED_CONTRIBUTION_V3) != 0;
        const float cutoff =
            apply_cutoff ? args->edge_cutoff_scale[edge] : 1.0f;
        for (std::int32_t channel = edge_lane;
             channel < {extents['multiplicity']}; channel += 16) {{
            {chr(10).join(path_blocks)}
        }}
    }}
    #pragma unroll
    for (std::int32_t offset = 8; offset > 0; offset /= 2) {{
        #pragma unroll
        for (std::int32_t phi = 0;
             phi < {extents['phi_dimension']}; ++phi) {{
            phi_adjoint[phi] += SYMMETRIX_JIT_MH1_SHUFFLE_DOWN(
                phi_adjoint[phi], offset, 16);
        }}
        #pragma unroll
        for (std::int32_t harmonic = 0;
             harmonic < {extents['input_2_dimension']}; ++harmonic) {{
            harmonic_adjoint[harmonic] += SYMMETRIX_JIT_MH1_SHUFFLE_DOWN(
                harmonic_adjoint[harmonic], offset, 16);
        }}
        cutoff_adjoint += SYMMETRIX_JIT_MH1_SHUFFLE_DOWN(
            cutoff_adjoint, offset, 16);
    }}
    if (edge_lane == 0 && edge_active) {{
        #pragma unroll
        for (std::int32_t phi = 0;
             phi < {extents['phi_dimension']}; ++phi) {{
            args->edge_phi_adjoint[
                local_edge * {extents['phi_dimension']} + phi] = phi_adjoint[phi];
        }}
        #pragma unroll
        for (std::int32_t harmonic = 0;
             harmonic < {extents['input_2_dimension']}; ++harmonic) {{
            args->edge_input_2_adjoint[
                local_edge * {extents['input_2_dimension']} + harmonic]
                = harmonic_adjoint[harmonic];
        }}
        args->edge_cutoff_scale_adjoint[local_edge] = cutoff_adjoint;
    }}
}}"""


def _render_cuda_edge_kernel(
    index: int, suffix: str, *, symbol_prefix: str = ""
) -> str:
    linkage = 'extern "C" ' if symbol_prefix else ""
    return f"""{linkage}__global__ void {symbol_prefix}edge_reverse_{suffix}_kernel_{index}(
    SymmetrixJitMH1CudaReverseArgsV3 args)
{{
    constexpr std::int32_t edge_width = 16;
    constexpr std::int32_t edges_per_warp = 2;
    const std::int32_t lane = threadIdx.x & 31;
    const std::int32_t edge_lane = lane & (edge_width - 1);
    const std::int32_t edge_owner = lane / edge_width;
    const std::int32_t warp = threadIdx.x / 32;
    const std::int32_t warps_per_block = blockDim.x / 32;
    for (std::int64_t edge_base =
             (static_cast<std::int64_t>(blockIdx.x) * warps_per_block + warp)
                 * edges_per_warp;
         edge_base < args.samples;
         edge_base += static_cast<std::int64_t>(gridDim.x)
             * warps_per_block * edges_per_warp) {{
        const std::int64_t local_edge = edge_base + edge_owner;
        const bool edge_active = local_edge < args.samples;
        edge_reverse_{suffix}_{index}(
            &args, static_cast<std::int32_t>(local_edge), edge_lane, edge_active);
    }}
}}"""


def _render_cuda_interaction_kernels(
    interaction: dict[str, Any],
    forward_partition_count: int,
    source_partition_count: int,
    edge_strategy: str,
    edge_phi_schedule: str,
    edge_reverse_schedule: str,
    *,
    symbol_prefix: str = "",
    forward_launch_bounds: str = "",
) -> str:
    index = int(interaction["index"])
    multiplicity = _interaction_extents(interaction)["multiplicity"]
    linkage = 'extern "C" ' if symbol_prefix else ""
    forward_dispatch = "\n                ".join(
        f"case {partition}: forward_{index}_partition_{partition}("
        f"&args, receiver, channel); break;"
        for partition in range(forward_partition_count)
    )
    source_dispatch = "\n                ".join(
        f"case {partition}: source_reverse_{index}_partition_{partition}("
        f"&args, source_owner, channel); break;"
        for partition in range(source_partition_count)
    )
    if edge_strategy == "split":
        phi_kernel = (
            _render_cuda_edge_reverse_phi_path_tiled_kernel(
                interaction, symbol_prefix=symbol_prefix
            )
            if edge_phi_schedule == "path_tiled"
            else _render_cuda_edge_kernel(index, "phi", symbol_prefix=symbol_prefix)
        )
        edge_kernels = "\n\n".join(
            (
                phi_kernel,
                _render_cuda_edge_kernel(
                    index, "harmonic", symbol_prefix=symbol_prefix
                ),
            )
        )
    elif edge_strategy == "compact_fused":
        edge_kernels = _render_cuda_edge_reverse_compact_fused_path_tiled_kernel(
            interaction,
            symbol_prefix=symbol_prefix,
            reverse_schedule=edge_reverse_schedule,
        )
    else:
        edge_kernels = _render_cuda_edge_kernel(
            index, "fused", symbol_prefix=symbol_prefix
        )
    source_kernel = f"""{linkage}__global__ void {symbol_prefix}source_reverse_kernel_{index}(
    SymmetrixJitMH1CudaReverseArgsV3 args)
{{
    constexpr std::int32_t channel_tile_width = 32;
    constexpr std::int32_t channel_tile_count =
        ({multiplicity} + channel_tile_width - 1) / channel_tile_width;
    constexpr std::int32_t source_partition_count = {source_partition_count};
    const std::int64_t owner_count =
        args.source_owner_count * source_partition_count * channel_tile_count;
    const std::int32_t lane = threadIdx.x & (channel_tile_width - 1);
    const std::int32_t warp = threadIdx.x / channel_tile_width;
    const std::int32_t warps_per_block = blockDim.x / channel_tile_width;
    const std::int64_t stride =
        static_cast<std::int64_t>(warps_per_block) * gridDim.x;
    for (std::int64_t owner =
             static_cast<std::int64_t>(blockIdx.x) * warps_per_block + warp;
         owner < owner_count; owner += stride) {{
        const std::int32_t channel_tile = owner % channel_tile_count;
        const std::int64_t source_partition_owner = owner / channel_tile_count;
        const std::int32_t source_partition =
            source_partition_owner % source_partition_count;
        const std::int32_t source_owner = static_cast<std::int32_t>(
            source_partition_owner / source_partition_count);
        const std::int32_t channel = channel_tile * channel_tile_width + lane;
        if (channel < {multiplicity}) {{
            switch (source_partition) {{
                {source_dispatch}
            }}
        }}
    }}
}}"""
    return f"""{linkage}__global__ void {forward_launch_bounds}{symbol_prefix}forward_kernel_{index}(
    SymmetrixJitMH1CudaForwardArgsV3 args)
{{
    constexpr std::int32_t channel_tile_width = 32;
    constexpr std::int32_t channel_tile_count =
        ({multiplicity} + channel_tile_width - 1) / channel_tile_width;
    constexpr std::int32_t partition_count = {forward_partition_count};
    const std::int64_t owner_count =
        static_cast<std::int64_t>(args.active_receiver_count)
            * partition_count * channel_tile_count;
    const std::int32_t lane = threadIdx.x & (channel_tile_width - 1);
    const std::int32_t warp = threadIdx.x / channel_tile_width;
    const std::int32_t warps_per_block = blockDim.x / channel_tile_width;
    const std::int64_t stride =
        static_cast<std::int64_t>(warps_per_block) * gridDim.x;
    for (std::int64_t owner =
             static_cast<std::int64_t>(blockIdx.x) * warps_per_block + warp;
        owner < owner_count; owner += stride) {{
        const std::int32_t channel_tile = owner % channel_tile_count;
        const std::int64_t partition_owner = owner / channel_tile_count;
        const std::int32_t partition = partition_owner % partition_count;
        const std::int32_t receiver_index =
            static_cast<std::int32_t>(partition_owner / partition_count);
        const std::int32_t receiver = args.active_receivers[receiver_index];
        const std::int32_t channel = channel_tile * channel_tile_width + lane;
        if (channel < {multiplicity}) {{
            switch (partition) {{
                {forward_dispatch}
            }}
        }}
    }}
}}

{source_kernel}

{edge_kernels}
"""


def _render_cuda_interaction_launchers(
    index: int,
    extents: dict[str, int],
    metadata: dict[str, Any],
    forward_partition_count: int,
    source_partition_count: int,
    edge_strategy: str,
    edge_phi_schedule: str,
) -> str:
    forward_threads = metadata["forward_threads_per_block"]
    source_threads = metadata["source_threads_per_block"]
    edge_threads = metadata["edge_threads_per_block"]
    if edge_strategy == "split":
        edge_launches = f"""edge_reverse_phi_kernel_{index}<<<
        edge_blocks, edge_threads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(launch_args);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess)
        return static_cast<std::int32_t>(status);
    edge_reverse_harmonic_kernel_{index}<<<edge_blocks, edge_threads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(launch_args);"""
        edge_phi_block_multiplier = (
            _MH1_PATH_TILED_EDGE_BLOCK_MULTIPLIER
            if edge_phi_schedule == "path_tiled"
            else 1
        )
        edge_phi_edge_batch = (
            _MH1_PATH_TILED_EDGE_EDGE_BATCH if edge_phi_schedule == "path_tiled" else 1
        )
        edge_block_setup = f"""constexpr std::int64_t edge_phi_edge_batch =
        {edge_phi_edge_batch};
    const std::int64_t edge_phi_groups =
        args->samples / edge_phi_edge_batch
        + (args->samples % edge_phi_edge_batch != 0 ? 1 : 0);
    const std::int32_t edge_phi_blocks =
        static_cast<std::int32_t>(edge_phi_groups <
                static_cast<std::int64_t>(edge_persistent_blocks)
                    * {edge_phi_block_multiplier}
            ? edge_phi_groups
            : static_cast<std::int64_t>(edge_persistent_blocks)
                * {edge_phi_block_multiplier});"""
    elif edge_strategy == "compact_fused":
        edge_launches = f"""edge_reverse_compact_fused_kernel_{index}<<<
        edge_phi_blocks, edge_threads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(launch_args);"""
        edge_block_setup = f"""constexpr std::int64_t edge_phi_edge_batch =
        {_MH1_PATH_TILED_EDGE_EDGE_BATCH};
    const std::int64_t edge_phi_groups =
        args->samples / edge_phi_edge_batch
        + (args->samples % edge_phi_edge_batch != 0 ? 1 : 0);
    const std::int32_t edge_phi_blocks =
        static_cast<std::int32_t>(edge_phi_groups <
                static_cast<std::int64_t>(edge_persistent_blocks)
                    * {_MH1_PATH_TILED_EDGE_BLOCK_MULTIPLIER}
            ? edge_phi_groups
            : static_cast<std::int64_t>(edge_persistent_blocks)
                * {_MH1_PATH_TILED_EDGE_BLOCK_MULTIPLIER});"""
    else:
        edge_launches = f"""edge_reverse_fused_kernel_{index}<<<
        edge_blocks, edge_threads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(launch_args);"""
        edge_block_setup = ""
    return f"""std::int32_t forward_launch_{index}(
    const SymmetrixJitMH1CudaForwardArgsV3* args,
    void* stream,
    std::int32_t persistent_blocks)
{{
    constexpr std::uint32_t known_flags =
        SYMMETRIX_JIT_MH1_CUDA_APPLY_EDGE_CUTOFF_V3
        | SYMMETRIX_JIT_MH1_CUDA_HAS_LINEAR_BIAS_V3
        | SYMMETRIX_JIT_MH1_CUDA_HAS_FIXED_CONTRIBUTION_V3;
    if (args == nullptr
        || args->struct_size < sizeof(SymmetrixJitMH1CudaForwardArgsV3)
        || args->interaction != {index}u || (args->flags & ~known_flags) != 0u
        || args->num_nodes < 0 || args->num_nodes > INT32_MAX
        || args->num_edges < 0 || args->num_edges > INT32_MAX
        || args->first_edge != 0 || args->samples < 0
        || args->samples != args->num_edges
        || (args->samples > 0 && args->num_nodes == 0)
        || args->active_receiver_count > args->num_nodes
        || args->active_receiver_count > args->samples
        || (args->samples > 0 && args->active_receiver_count == 0)
        || persistent_blocks <= 0)
        return static_cast<std::int32_t>(cudaErrorInvalidValue);
    if (args->samples == 0)
        return static_cast<std::int32_t>(cudaSuccess);
    const bool apply_cutoff =
        (args->flags & SYMMETRIX_JIT_MH1_CUDA_APPLY_EDGE_CUTOFF_V3) != 0;
    const bool has_bias =
        (args->flags & SYMMETRIX_JIT_MH1_CUDA_HAS_LINEAR_BIAS_V3) != 0;
    const bool has_fixed =
        (args->flags & SYMMETRIX_JIT_MH1_CUDA_HAS_FIXED_CONTRIBUTION_V3) != 0;
    if (args->source_indices == nullptr || args->active_receivers == nullptr
        || args->receiver_offsets == nullptr
        || args->edge_phi == nullptr || args->linear_weight == nullptr
        || (has_bias && args->linear_bias == nullptr)
        || (has_fixed && args->edge_linear_contribution == nullptr)
        || args->edge_input_2 == nullptr
        || (apply_cutoff && args->edge_cutoff_scale == nullptr)
        || args->source_node_values == nullptr || args->output_mask == nullptr
        || args->node_messages == nullptr)
        return static_cast<std::int32_t>(cudaErrorInvalidDevicePointer);
    constexpr std::int32_t threads = {forward_threads};
    static_assert(threads >= 32 && threads % 32 == 0,
        "MH-1 CUDA forward block size must be a positive warp multiple");
    constexpr std::int32_t channel_tiles =
        ({extents['multiplicity']} + 31) / 32;
    const std::int32_t blocks = launch_blocks(
        static_cast<std::int64_t>(args->active_receiver_count)
            * {forward_partition_count} * channel_tiles * 32,
        threads, persistent_blocks);
    const auto launch_args = *args;
    forward_kernel_{index}<<<blocks, threads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(launch_args);
    return static_cast<std::int32_t>(cudaPeekAtLastError());
}}

std::int32_t reverse_launch_{index}(
    const SymmetrixJitMH1CudaReverseArgsV3* args,
    void* stream,
    std::int32_t source_persistent_blocks,
    std::int32_t edge_persistent_blocks)
{{
    constexpr std::uint32_t known_flags =
        SYMMETRIX_JIT_MH1_CUDA_APPLY_EDGE_CUTOFF_V3
        | SYMMETRIX_JIT_MH1_CUDA_HAS_LINEAR_BIAS_V3
        | SYMMETRIX_JIT_MH1_CUDA_HAS_FIXED_CONTRIBUTION_V3;
    if (args == nullptr
        || args->struct_size < sizeof(SymmetrixJitMH1CudaReverseArgsV3)
        || args->interaction != {index}u || (args->flags & ~known_flags) != 0u
        || args->num_nodes < 0 || args->num_nodes > INT32_MAX
        || args->num_edges < 0 || args->num_edges > INT32_MAX
        || args->first_edge != 0 || args->samples < 0
        || args->samples != args->num_edges
        || (args->samples > 0 && args->num_nodes == 0)
        || args->source_owner_count < 0
        || args->source_owner_count > args->samples
        || args->source_owner_count > args->num_nodes
        || (args->samples > 0 && args->source_owner_count == 0)
        || source_persistent_blocks <= 0 || edge_persistent_blocks <= 0)
        return static_cast<std::int32_t>(cudaErrorInvalidValue);
    if (args->samples == 0)
        return static_cast<std::int32_t>(cudaSuccess);
    const bool apply_cutoff =
        (args->flags & SYMMETRIX_JIT_MH1_CUDA_APPLY_EDGE_CUTOFF_V3) != 0;
    const bool has_bias =
        (args->flags & SYMMETRIX_JIT_MH1_CUDA_HAS_LINEAR_BIAS_V3) != 0;
    const bool has_fixed =
        (args->flags & SYMMETRIX_JIT_MH1_CUDA_HAS_FIXED_CONTRIBUTION_V3) != 0;
    if (args->source_indices == nullptr || args->target_indices == nullptr
        || args->source_edge_offsets == nullptr
        || args->source_edge_indices == nullptr
        || args->edge_phi == nullptr || args->linear_weight == nullptr
        || (has_bias && args->linear_bias == nullptr)
        || (has_fixed && args->edge_linear_contribution == nullptr)
        || args->edge_input_2 == nullptr
        || (apply_cutoff && args->edge_cutoff_scale == nullptr)
        || args->source_node_values == nullptr || args->output_mask == nullptr
        || args->target_node_output_adjoint == nullptr
        || args->source_node_input_adjoint == nullptr
        || args->edge_phi_adjoint == nullptr
        || args->edge_input_2_adjoint == nullptr
        || args->edge_cutoff_scale_adjoint == nullptr)
        return static_cast<std::int32_t>(cudaErrorInvalidDevicePointer);
    const auto launch_args = *args;
    constexpr std::int32_t source_threads = {source_threads};
    static_assert(source_threads >= 32 && source_threads % 32 == 0,
        "MH-1 CUDA source block size must be a positive warp multiple");
    constexpr std::int32_t source_channel_tiles =
        ({extents['multiplicity']} + 31) / 32;
    const std::int32_t source_blocks = launch_blocks(
        args->source_owner_count * {source_partition_count}
            * source_channel_tiles * 32,
        source_threads, source_persistent_blocks);
    source_reverse_kernel_{index}<<<source_blocks, source_threads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(launch_args);
    cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess)
        return static_cast<std::int32_t>(status);
    constexpr std::int32_t edge_threads = {edge_threads};
    static_assert(edge_threads >= 32 && edge_threads % 32 == 0,
        "MH-1 CUDA edge block size must be a positive warp multiple");
    const std::int32_t edge_blocks = launch_blocks(
        args->samples * 16, edge_threads, edge_persistent_blocks);
    {edge_block_setup}
    {edge_launches}
    return static_cast<std::int32_t>(cudaPeekAtLastError());
}}
"""


def _mh1_spline_r_path_rows(
    interaction: dict[str, Any],
) -> tuple[tuple[dict[str, Any], ...], ...]:
    """Project an MH-1 UVU interaction onto the ordinary generated R1 rows."""

    multiplicity = int(interaction["dimensions"]["channels"])
    next_row = 0
    result = []
    for path in interaction["paths"]:
        source = path["input_1"]
        edge = path["input_2"]
        output = path["output"]
        if any(
            int(block["multiplicity"]) != multiplicity for block in (source, output)
        ):
            raise ValueError("MH1 spline R generation requires tied UVU channels")
        if int(source["offset"]) % multiplicity != 0:
            raise ValueError(
                "MH1 spline R source offsets must align to tied UVU channels"
            )
        if int(output["offset"]) % multiplicity != 0:
            raise ValueError(
                "MH1 spline R output offsets must align to tied UVU channels"
            )
        if int(edge["multiplicity"]) != 1:
            raise ValueError("MH1 spline R edge irreps must have multiplicity one")
        rows: dict[tuple[int, int], dict[str, Any]] = {}
        for term in path["sparse_wigner"]["terms"]:
            key = (int(term["b"]), int(term["a"]))
            row = rows.get(key)
            if row is None:
                row = {
                    "row": next_row,
                    "path": int(path["index"]),
                    "lm1": int(edge["offset"]) + key[0],
                    "lm2": int(source["offset"]) // multiplicity + key[1],
                    "terms": [],
                }
                rows[key] = row
                next_row += 1
            row["terms"].append(
                {
                    "lme": int(output["offset"]) // multiplicity + int(term["c"]),
                    "coefficient": float(path["path_weight"])
                    * float(term["coefficient"]),
                }
            )
        result.append(tuple(rows.values()))
    return tuple(result)


def _render_mh1_spline_r_gpu_programs(
    contract: dict[str, Any], dialect: MH1GpuDialect, precision: str = "float32"
) -> str:
    normalized = normalize_execution_mh1_v4_contract(contract)
    projected = project_execution_mh1_v3_contract(normalized)
    standard_dialect = (
        STANDARD_HIP_DIALECT if dialect.backend == "hip" else STANDARD_CUDA_DIALECT
    )
    qualifier = "__device__ __attribute__((always_inline)) inline"
    programs = []
    for interaction in projected["interactions"]:
        index = int(interaction["index"])
        extents = _interaction_extents(interaction)
        channels = extents["multiplicity"]
        source_harmonics = extents["input_1_angular_dimension"]
        edge_harmonics = extents["input_2_dimension"]
        output_components = extents["output_dimension"] // channels
        density_function = extents["weight_size"]
        radial_offsets = tuple(
            int(path["weight_offset"]) for path in interaction["paths"]
        )
        rows = _mh1_spline_r_path_rows(interaction)
        namespace = f"symmetrix_mh1_spline_r_{index}"
        forward_owner = _render_standard_r1_forward_owner(
            rows,
            channels=channels,
            edge_harmonics=edge_harmonics,
            source_harmonics=source_harmonics,
            output_components=output_components,
            args_type="SymmetrixJitMH1CudaSplineRForwardArgsV5",
            qualifier=qualifier,
            scalar_type="Scalar",
            opaque_pointers=True,
            ordered_pair_types=True,
            map_type_indices=False,
            density_function=density_function,
            radial_function_offsets=radial_offsets,
            use_output_mask=True,
            filter_inactive_edges=False,
        ).replace("r1_forward_owner", f"spline_r_forward_owner_{index}")
        reverse_kernel = _render_standard_r1_fused_reverse_kernel(
            rows,
            source_type="SymmetrixJitMH1CudaSplineRSourceArgsV5",
            edge_type="SymmetrixJitMH1CudaSplineREdgeArgsV5",
            kernel_name=f"symmetrix_execution_mh1_spline_r_reverse_kernel_{index}",
            channels=channels,
            edge_harmonics=edge_harmonics,
            source_harmonics=source_harmonics,
            output_components=output_components,
            subgroup_width=32,
            dialect=standard_dialect,
            exported=True,
            ordered_pair_types=True,
            map_type_indices=False,
            compact_source_owners=True,
            density_function=density_function,
            radial_function_offsets=radial_offsets,
            use_output_mask=True,
            filter_inactive_edges=False,
        )
        spline_helpers = _render_gpu_spline_helpers(
            "SymmetrixJitMH1CudaSplineV5",
            qualifier,
            precision_matched_coordinates=dialect.backend == "hip",
            include_active_edge_helper=False,
        )
        harmonic_helpers = _render_direct_harmonic_gradient_helper(
            qualifier, edge_harmonics
        )
        scalar = "float" if precision == "float32" else "double"
        programs.append(
            f"""namespace {namespace} {{
using Scalar = {scalar};
constexpr std::int32_t reverse_threads_per_block = 64;

{spline_helpers}

{harmonic_helpers}

{forward_owner}

extern "C" __global__ void
symmetrix_execution_mh1_spline_r_forward_kernel_{index}(
    SymmetrixJitMH1CudaSplineRForwardArgsV5 args)
{{
    const std::int64_t stride =
        static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    const std::int64_t owners = args.num_nodes * {channels};
    for (std::int64_t owner =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         owner < owners; owner += stride)
        spline_r_forward_owner_{index}(
            &args, static_cast<std::int32_t>(owner / {channels}),
            static_cast<std::int32_t>(owner % {channels}),
            static_cast<Scalar*>(args.output)
                + (owner / {channels}) * {output_components * channels});
}}

{reverse_kernel}
}}  // namespace {namespace}"""
        )
    return "\n\n".join(programs)


def _render_execution_mh1_cuda_artifact(
    contract: dict[str, Any],
    compute_capability: int,
    *,
    forward_policy: str | dict[str, Any] = "auto",
    source_policy: str | dict[str, Any] = "auto",
    edge_policy: str | dict[str, Any] = "auto",
    _artifact_kind: Literal["plugin", "module"] = "plugin",
    _dialect: MH1GpuDialect = MH1_CUDA_DIALECT,
    _edge_reverse_schedule: str = _MH1_CUDA_EDGE_REVERSE_SCHEDULE,
) -> str:
    """Render a plugin or device-only CUDA artifact from one MH-1 program."""

    normalized = normalize_execution_mh1_contract(contract)
    resolved_forward_policy = resolve_execution_mh1_cuda_forward_policy(
        normalized, compute_capability, forward_policy
    )
    resolved_source_policy = resolve_execution_mh1_cuda_source_policy(
        normalized, compute_capability, source_policy
    )
    resolved_edge_policy = resolve_execution_mh1_cuda_edge_policy(
        normalized,
        compute_capability,
        edge_policy,
        _reverse_schedule=_edge_reverse_schedule,
    )
    metadata = jit_mh1_cuda_plugin_metadata(
        normalized,
        compute_capability,
        forward_policy=resolved_forward_policy,
        source_policy=resolved_source_policy,
        edge_policy=resolved_edge_policy,
        _edge_reverse_schedule=_edge_reverse_schedule,
    )
    interactions = normalized["interactions"]
    conditioner_layouts = metadata["conditioner_layout"]["interactions"]
    forward_policy_by_interaction = {
        int(policy["index"]): policy
        for policy in resolved_forward_policy["interactions"]
    }
    source_policy_by_interaction = {
        int(policy["index"]): policy
        for policy in resolved_source_policy["interactions"]
    }
    edge_policy_by_interaction = {
        int(policy["index"]): policy for policy in resolved_edge_policy["interactions"]
    }
    symbol_prefix = "symmetrix_execution_mh1_" if _artifact_kind == "module" else ""
    device_functions = []
    kernels = []
    launchers = []
    for interaction in interactions:
        index = int(interaction["index"])
        extents = _interaction_extents(interaction)
        forward_interaction_policy = forward_policy_by_interaction[index]
        source_interaction_policy = source_policy_by_interaction[index]
        edge_interaction_policy = edge_policy_by_interaction[index]
        edge_strategy = edge_interaction_policy["strategy"]
        edge_phi_schedule = _edge_phi_schedule(interaction)
        forward_partition_count = len(forward_interaction_policy["partitions"])
        source_partition_count = len(source_interaction_policy["partitions"])
        if edge_strategy == "split":
            device_functions.extend(
                (
                    _render_cuda_edge_reverse_phi(interaction),
                    _render_cuda_edge_reverse_harmonic(interaction),
                )
            )
        elif edge_strategy == "fused":
            device_functions.append(_render_cuda_edge_reverse_fused(interaction))
        device_functions.extend(
            _render_cuda_forward_partitions(interaction, forward_interaction_policy)
        )
        device_functions.extend(
            _render_cuda_source_reverse_partitions(
                interaction,
                source_interaction_policy,
                path_weight_tile_size=(
                    2 if _dialect.backend == "cuda" and index == 1 else 1
                ),
            )
        )
        kernels.append(
            _render_cuda_interaction_kernels(
                interaction,
                forward_partition_count,
                source_partition_count,
                edge_strategy,
                edge_phi_schedule,
                _edge_reverse_schedule,
                symbol_prefix=symbol_prefix,
                forward_launch_bounds=(
                    "__launch_bounds__(128, 4) "
                    if _dialect.backend == "cuda" and index == 1
                    else ""
                ),
            )
        )
        launchers.append(
            _render_cuda_interaction_launchers(
                index,
                extents,
                metadata,
                forward_partition_count,
                source_partition_count,
                edge_strategy,
                edge_phi_schedule,
            )
        )
    conditioner_helpers = render_execution_mh1_conditioner_cpp_helpers(
        normalized, bounded_reverse=True
    )
    for interaction_layout in conditioner_layouts:
        index = int(interaction_layout["index"])
        device_functions.append(
            _render_conditioning_owners(
                interaction_layout,
                backend="Cuda",
                function_qualifier="__device__ __forceinline__",
            )
        )
        kernels.append(
            _render_cuda_conditioning_kernels(index, symbol_prefix=symbol_prefix)
        )
        launchers.append(
            _render_cuda_conditioning_launchers(interaction_layout, metadata)
        )
    if _artifact_kind == "module":
        return "\n\n".join(
            (
                _dialect.device_prelude(),
                conditioner_helpers,
                *device_functions,
                *kernels,
            )
        )
    if _artifact_kind != "plugin":
        raise ValueError("MH1 CUDA artifact kind must be 'plugin' or 'module'")
    descriptors = ",\n        ".join(
        "{"
        + ", ".join(
            (
                "sizeof(SymmetrixJitMH1CudaInteractionV3)",
                "0u",
                str(extents["input_1_dimension"]),
                str(extents["input_2_dimension"]),
                str(extents["output_dimension"]),
                str(extents["weight_size"]),
                str(extents["phi_dimension"]),
                str(extents["multiplicity"]),
                str(extents["input_1_angular_dimension"]),
                str(extents["instruction_count"]),
                str(metadata["forward_threads_per_block"]),
                str(metadata["source_threads_per_block"]),
                str(metadata["edge_threads_per_block"]),
                str(edge_policy["physical_launch_count"]),
            )
        )
        + "}"
        for extents, edge_policy in zip(
            metadata["interactions"],
            resolved_edge_policy["interactions"],
            strict=True,
        )
    )
    launch_descriptors = ",\n        ".join(
        "{"
        + ", ".join(
            (
                "sizeof(SymmetrixJitMH1CudaInteractionLaunchesV3)",
                f"{index}u",
                "0u",
                "0u",
                f"&forward_launch_{index}",
                f"&reverse_launch_{index}",
            )
        )
        + "}"
        for index in range(2)
    )
    conditioning_launch_descriptors = ",\n        ".join(
        "{"
        + ", ".join(
            (
                "sizeof(SymmetrixJitMH1CudaConditioningLaunchesV3)",
                f"{index}u",
                "0u",
                "0u",
                f"&conditioning_forward_launch_{index}",
                f"&conditioning_reverse_launch_{index}",
            )
        )
        + "}"
        for index in range(2)
    )
    abi_header = _CUDA_PLUGIN_ABI_HEADER.replace("#pragma once\n", "")
    return f"""// Generated by symmetrix.mh1_jit_codegen. Do not edit.
{abi_header}

#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <cstdint>

{_dialect.device_prelude()}

namespace {{

std::int32_t launch_blocks(
    std::int64_t work_items,
    std::int32_t threads_per_block,
    std::int32_t persistent_blocks)
{{
    const std::int64_t required =
        work_items / threads_per_block
        + (work_items % threads_per_block != 0);
    const std::int64_t capped =
        required < persistent_blocks ? required : persistent_blocks;
    return static_cast<std::int32_t>(capped);
}}

{conditioner_helpers}

{chr(10).join(device_functions)}

{chr(10).join(kernels)}

{chr(10).join(launchers)}

const SymmetrixJitMH1CudaPluginV3 descriptor {{
    SYMMETRIX_JIT_MH1_CUDA_PLUGIN_ABI_VERSION,
    sizeof(SymmetrixJitMH1CudaPluginV3),
    sizeof(void*),
    SYMMETRIX_JIT_MH1_CUDA_PLUGIN_BYTE_ORDER,
    SYMMETRIX_JIT_MH1_CUDA_FORWARD_LAUNCH_V3
        | SYMMETRIX_JIT_MH1_CUDA_REVERSE_LAUNCH_V3
        | SYMMETRIX_JIT_MH1_CUDA_PHI_MAJOR_LINEAR_WEIGHT_V3
        | SYMMETRIX_JIT_MH1_CUDA_ACTIVE_RECEIVERS_V3
        | SYMMETRIX_JIT_MH1_CUDA_GRAPH_WIDE_V3
        | SYMMETRIX_JIT_MH1_CUDA_IR_MUL_FEATURES_V3
        | SYMMETRIX_JIT_MH1_CUDA_GLOBAL_SOURCE_REVERSE_V3
        | SYMMETRIX_JIT_MH1_CUDA_EDGE_LAUNCH_COUNT_V3
        | SYMMETRIX_JIT_MH1_CUDA_CONDITIONING_LAUNCH_V3,
    SYMMETRIX_JIT_MH1_CUDA_PLUGIN_INTERACTIONS,
    sizeof(float),
    0u,
    SYMMETRIX_JIT_MH1_CUDA_PLUGIN_ABI_TAG,
    {_cpp_string(metadata['artifact_id'])},
    {_cpp_string(metadata['generation_fingerprint'])},
    {_cpp_string(metadata['semantic_fingerprint'])},
    {_cpp_string(metadata['structure_fingerprint'])},
    {metadata['target_compute_capability']},
    0,
    {{
        {descriptors}
    }},
    {{
        {launch_descriptors}
    }},
    {{
        {conditioning_launch_descriptors}
    }},
}};

}}  // namespace

extern "C" SYMMETRIX_JIT_MH1_CUDA_PLUGIN_EXPORT
const SymmetrixJitMH1CudaPluginV3*
symmetrix_jit_mh1_cuda_plugin_query_v3(void)
{{
    return &descriptor;
}}
"""


def render_jit_mh1_cuda_plugin(
    contract: dict[str, Any],
    compute_capability: int,
    *,
    forward_policy: str | dict[str, Any] = "auto",
    source_policy: str | dict[str, Any] = "auto",
    edge_policy: str | dict[str, Any] = "auto",
) -> str:
    """Render generated CUDA forward and coordinate-reverse MH-1 kernels."""

    return _render_execution_mh1_cuda_artifact(
        contract,
        compute_capability,
        forward_policy=forward_policy,
        source_policy=source_policy,
        edge_policy=edge_policy,
    )


def jit_mh1_cuda_plugin_v4_metadata(
    contract: dict[str, Any],
    compute_capability: int,
    *,
    forward_policy: str | dict[str, Any] = "auto",
    source_policy: str | dict[str, Any] = "auto",
    edge_policy: str | dict[str, Any] = "auto",
    node_state_policy: str = MH1_NODE_STATE_FULL_RETENTION,
    _edge_reverse_schedule: str = _MH1_CUDA_EDGE_REVERSE_SCHEDULE,
    _node_reverse_schedule: str = _MH1_NODE_REVERSE_SCHEDULE,
) -> dict[str, Any]:
    """Return CUDA-v4 cache metadata while retaining v3 TP policies."""

    normalized = normalize_execution_mh1_v4_contract(contract)
    projected = project_execution_mh1_v3_contract(normalized)
    base = jit_mh1_cuda_plugin_metadata(
        projected,
        compute_capability,
        forward_policy=forward_policy,
        source_policy=source_policy,
        edge_policy=edge_policy,
        _edge_reverse_schedule=_edge_reverse_schedule,
    )
    node_state_policy = _resolve_node_state_policy(node_state_policy)
    if _node_reverse_schedule not in (
        _MH1_NODE_REVERSE_SCHEDULE,
        _MH1_CUDA_RECOMPUTE_GROUPED_MESSAGE_REVERSE_SCHEDULE,
        _MH1_CUDA_RECOMPUTE_GROUPED_LINEAR2_SCHEDULE,
        _MH1_CUDA_RECOMPUTE_GROUPED_ALL_MESSAGE_SCHEDULE,
        _MH1_CUDA_RECOMPUTE_GROUPED_FORWARD_MESSAGE_SCHEDULE,
        _MH1_CUDA_RECOMPUTE_GROUPED_RESIDUAL_SCHEDULE,
        _MH1_CUDA_RECOMPUTE_TILED_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_RETAINED_NODE_REVERSE_SCHEDULE,
        _MH1_CUDA_GROUPED_MESSAGE_REVERSE_SCHEDULE,
        _MH1_CUDA_GROUPED_LINEAR2_SCHEDULE,
        _MH1_CUDA_GROUPED_ALL_MESSAGE_SCHEDULE,
        _MH1_CUDA_GROUPED_FORWARD_MESSAGE_SCHEDULE,
        _MH1_CUDA_GROUPED_RESIDUAL_SCHEDULE,
        _MH1_CUDA_GROUPED_FORWARD_RESIDUAL_SCHEDULE,
        _MH1_CUDA_TILED_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_TILED_PRODUCT_FORWARD_REVERSE_SCHEDULE,
        _MH1_CUDA_TILED_LINEAR_UP_REVERSE_SCHEDULE,
        _MH1_CUDA_TILED_LINEAR_UP_FORWARD_REVERSE_SCHEDULE,
        _MH1_CUDA_GROUPED_READOUT_REVERSE_SCHEDULE,
        _MH1_CUDA_GROUPED_SKIP_FORWARD_SCHEDULE,
        _MH1_CUDA_GROUPED_ALL_SKIP_FORWARD_SCHEDULE,
        _MH1_CUDA_WIDE_PRODUCT_LINEAR_SCHEDULE,
        _MH1_CUDA_WIDE_L1_REVERSE_MESSAGE_SCHEDULE,
        _MH1_CUDA_WIDE_L1_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_PAIRED_L0_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_WIDE_ALL_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_WIDE_L0_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_WIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_PAIRED_ALL_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_XWIDE_L0_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_ALL_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_L0_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
    ):
        raise ValueError("MH1 node reverse schedule is unsupported")
    if (
        _node_reverse_schedule
        in (
            _MH1_CUDA_RETAINED_NODE_REVERSE_SCHEDULE,
            _MH1_CUDA_GROUPED_MESSAGE_REVERSE_SCHEDULE,
            _MH1_CUDA_GROUPED_LINEAR2_SCHEDULE,
            _MH1_CUDA_GROUPED_ALL_MESSAGE_SCHEDULE,
            _MH1_CUDA_GROUPED_FORWARD_MESSAGE_SCHEDULE,
            _MH1_CUDA_GROUPED_RESIDUAL_SCHEDULE,
            _MH1_CUDA_GROUPED_FORWARD_RESIDUAL_SCHEDULE,
            _MH1_CUDA_TILED_PRODUCT_REVERSE_SCHEDULE,
            _MH1_CUDA_TILED_PRODUCT_FORWARD_REVERSE_SCHEDULE,
            _MH1_CUDA_TILED_LINEAR_UP_REVERSE_SCHEDULE,
            _MH1_CUDA_TILED_LINEAR_UP_FORWARD_REVERSE_SCHEDULE,
            _MH1_CUDA_GROUPED_READOUT_REVERSE_SCHEDULE,
            _MH1_CUDA_GROUPED_SKIP_FORWARD_SCHEDULE,
            _MH1_CUDA_GROUPED_ALL_SKIP_FORWARD_SCHEDULE,
            _MH1_CUDA_WIDE_PRODUCT_LINEAR_SCHEDULE,
            _MH1_CUDA_WIDE_L1_REVERSE_MESSAGE_SCHEDULE,
            _MH1_CUDA_WIDE_L1_REVERSE_LINEAR2_SCHEDULE,
            _MH1_CUDA_PAIRED_L0_PRODUCT_REVERSE_SCHEDULE,
            _MH1_CUDA_WIDE_ALL_REVERSE_LINEAR2_SCHEDULE,
            _MH1_CUDA_WIDE_L0_FORWARD_LINEAR2_SCHEDULE,
            _MH1_CUDA_WIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
            _MH1_CUDA_PAIRED_ALL_PRODUCT_REVERSE_SCHEDULE,
            _MH1_CUDA_XWIDE_L0_REVERSE_LINEAR2_SCHEDULE,
            _MH1_CUDA_XWIDE_ALL_REVERSE_LINEAR2_SCHEDULE,
            _MH1_CUDA_XWIDE_L0_FORWARD_LINEAR2_SCHEDULE,
            _MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
        )
        and node_state_policy == MH1_NODE_STATE_RECOMPUTE
    ):
        raise ValueError("experimental MH1 reverse requires full node retention")
    node = execution_mh1_node_program_metadata(
        normalized, backend="cuda", node_state_policy=node_state_policy
    )
    return {
        **base,
        "abi": MH1_CUDA_PLUGIN_V4_ABI,
        "abi_version": MH1_CUDA_PLUGIN_V4_ABI_VERSION,
        "artifact_id": node["artifact_id"],
        "generation_fingerprint": node["generation_fingerprint"],
        "semantic_fingerprint": node["semantic_fingerprint"],
        "structure_fingerprint": node["structure_fingerprint"],
        "runtime_layout_fingerprint": node["runtime_layout_fingerprint"],
        "layers": node["layers"],
        "node_forward_schedule": _MH1_NODE_FORWARD_SCHEDULE,
        "node_reverse_schedule": _node_reverse_schedule,
        "spline_r_schedule": _MH1_SPLINE_R_SCHEDULE,
        "node_state_policy": node_state_policy,
    }


def _execution_mh1_cuda_module_identity(
    metadata: dict[str, Any], precision: str = "float32"
) -> str:
    """Bind a device module to its model, layout, target, and launch policies."""

    _gpu_scalar_size(precision)
    fields = {
        "schema": MH1_CUDA_MODULE_ABI,
        "precision": precision,
        "artifact_id": metadata["artifact_id"],
        "generation_fingerprint": metadata["generation_fingerprint"],
        "semantic_fingerprint": metadata["semantic_fingerprint"],
        "structure_fingerprint": metadata["structure_fingerprint"],
        "runtime_layout_fingerprint": metadata["runtime_layout_fingerprint"],
        "target_compute_capability": metadata["target_compute_capability"],
        "forward_policy_id": metadata["forward_policy_id"],
        "source_policy_id": metadata["source_policy_id"],
        "edge_policy_id": metadata["edge_policy_id"],
        "edge_reverse_schedule": metadata["edge_reverse_schedule"],
        "conditioner_reverse_schedule_id": metadata["conditioner_reverse_schedule_id"],
        "node_forward_schedule": _MH1_NODE_FORWARD_SCHEDULE,
        "node_reverse_schedule": metadata["node_reverse_schedule"],
        "spline_r_schedule": _MH1_SPLINE_R_SCHEDULE,
        "node_state_policy": metadata["node_state_policy"],
    }
    encoded = json.dumps(
        fields, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _execution_mh1_hip_module_identity(
    metadata: dict[str, Any], target: MH1GpuTarget, precision: str = "float32"
) -> str:
    _gpu_scalar_size(precision)
    fields = {
        "schema": MH1_HIP_MODULE_ABI,
        "precision": precision,
        "artifact_id": metadata["artifact_id"],
        "generation_fingerprint": metadata["generation_fingerprint"],
        "semantic_fingerprint": metadata["semantic_fingerprint"],
        "structure_fingerprint": metadata["structure_fingerprint"],
        "runtime_layout_fingerprint": metadata["runtime_layout_fingerprint"],
        "target": json.loads(target.canonical_json()),
        "forward_policy_id": metadata["forward_policy_id"],
        "source_policy_id": metadata["source_policy_id"],
        "edge_policy_id": metadata["edge_policy_id"],
        "edge_reverse_schedule": metadata["edge_reverse_schedule"],
        "conditioner_reverse_schedule_id": metadata["conditioner_reverse_schedule_id"],
        "node_forward_schedule": _MH1_NODE_FORWARD_SCHEDULE,
        "node_reverse_schedule": metadata["node_reverse_schedule"],
        "spline_r_schedule": _MH1_SPLINE_R_SCHEDULE,
        "node_state_policy": metadata["node_state_policy"],
    }
    encoded = _canonical_json(fields).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _mh1_hip_target(target_metadata: dict[str, Any]) -> MH1GpuTarget:
    architecture = str(target_metadata.get("architecture", ""))
    raw_features = target_metadata.get("target_features", ())
    if isinstance(raw_features, str):
        features = tuple(sorted(item for item in raw_features.split(":") if item))
    elif isinstance(raw_features, (list, tuple)):
        features = tuple(sorted(str(item) for item in raw_features))
    else:
        raise TypeError("MH1 HIP target features must be a string or sequence")
    compiler_target = target_metadata.get("compiler_offload_target")
    native_width = target_metadata.get("native_subgroup_width")
    compute_units = target_metadata.get("compute_unit_count")
    if isinstance(native_width, bool) or not isinstance(native_width, int):
        raise TypeError("MH1 HIP native subgroup width is required")
    if isinstance(compute_units, bool) or not isinstance(compute_units, int):
        raise TypeError("MH1 HIP compute-unit count is required")
    return MH1GpuTarget(
        1,
        "hip",
        architecture,
        native_width,
        16,
        32,
        features,
        str(compiler_target) if compiler_target else architecture,
        compute_units,
    )


def _mh1_hip_internal_policy(
    request: str | dict[str, Any],
    target: MH1GpuTarget,
    legacy_tag: str,
) -> str | dict[str, Any]:
    """Translate a validated public HIP policy to the internal v1 selector."""

    if isinstance(request, str):
        return request
    if not isinstance(request, dict):
        raise TypeError("MH1 HIP policy must be a string or mapping")
    candidate = json.loads(json.dumps(request))
    if candidate.get("tag") == legacy_tag:
        return candidate
    neutral_tag = legacy_tag.replace(".cuda-", ".gpu-")
    if candidate.get("tag") != neutral_tag:
        raise ValueError("MH1 HIP policy tag is unsupported")
    if candidate.get("target_id") != target.target_id:
        raise ValueError("MH1 HIP policy target does not match HIP target")
    if candidate.get("selection_profile") != _MH1_SHARED_POLICY_PROFILE:
        raise ValueError("MH1 HIP policy selection profile is unsupported")
    policy_id = candidate.get("policy_id")
    payload = {key: value for key, value in candidate.items() if key != "policy_id"}
    expected_id = (
        "sha256:" + hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()
    )
    if policy_id != expected_id:
        raise ValueError("MH1 HIP policy identity is stale")
    internal = {
        key: value
        for key, value in payload.items()
        if key not in ("target_id", "selection_profile")
    }
    internal["tag"] = legacy_tag
    internal["target_compute_capability"] = 80
    internal["policy_id"] = _policy_digest(internal)
    return internal


def _mh1_hip_internal_policies(
    target: MH1GpuTarget,
    forward_policy: str | dict[str, Any],
    source_policy: str | dict[str, Any],
    edge_policy: str | dict[str, Any],
) -> tuple[str | dict[str, Any], str | dict[str, Any], str | dict[str, Any]]:
    return (
        _mh1_hip_internal_policy(forward_policy, target, MH1_CUDA_FORWARD_POLICY_TAG),
        _mh1_hip_internal_policy(source_policy, target, MH1_CUDA_SOURCE_POLICY_TAG),
        _mh1_hip_internal_policy(edge_policy, target, MH1_CUDA_EDGE_POLICY_TAG),
    )


def execution_mh1_hip_module_v4_metadata(
    contract: dict[str, Any],
    target_metadata: dict[str, Any],
    *,
    forward_policy: str | dict[str, Any] = "auto",
    source_policy: str | dict[str, Any] = "auto",
    edge_policy: str | dict[str, Any] = "auto",
    node_state_policy: str = MH1_NODE_STATE_FULL_RETENTION,
) -> dict[str, Any]:
    """Return cache metadata for the shared MH1 program compiled by hipRTC."""

    target = _mh1_hip_target(target_metadata)
    # Version-1 partition policies do not branch on this legacy CUDA field. It
    # remains an internal selector only; HIP-facing metadata is target-neutral.
    policy_profile = 80
    node_reverse_schedule = _hip_module_node_reverse_schedule(node_state_policy)
    internal_forward, internal_source, internal_edge = _mh1_hip_internal_policies(
        target, forward_policy, source_policy, edge_policy
    )
    legacy_metadata = jit_mh1_cuda_plugin_v4_metadata(
        contract,
        policy_profile,
        forward_policy=internal_forward,
        source_policy=internal_source,
        edge_policy=internal_edge,
        node_state_policy=node_state_policy,
        _edge_reverse_schedule=_MH1_HIP_EDGE_REVERSE_SCHEDULE,
        _node_reverse_schedule=node_reverse_schedule,
    )
    metadata = {
        key: value
        for key, value in legacy_metadata.items()
        if key != "target_compute_capability"
    }
    for policy_name in ("forward_policy", "source_policy", "edge_policy"):
        legacy_policy = metadata[policy_name]
        policy = {
            key: value
            for key, value in legacy_policy.items()
            if key not in ("policy_id", "target_compute_capability")
        }
        policy["tag"] = str(policy["tag"]).replace(".cuda-", ".gpu-")
        policy["target_id"] = target.target_id
        policy["selection_profile"] = _MH1_SHARED_POLICY_PROFILE
        policy["policy_id"] = (
            "sha256:"
            + hashlib.sha256(_canonical_json(policy).encode("ascii")).hexdigest()
        )
        metadata[policy_name] = policy
        metadata[f"{policy_name}_id"] = policy["policy_id"]
    return {
        **metadata,
        "abi": MH1_HIP_MODULE_ABI,
        "abi_version": MH1_HIP_MODULE_ABI_VERSION,
        "target_backend": "hip",
        "target": json.loads(target.canonical_json()),
        "target_id": target.target_id,
        "policy_profile": _MH1_SHARED_POLICY_PROFILE,
        "node_forward_schedule": _MH1_NODE_FORWARD_SCHEDULE,
        "node_reverse_schedule": metadata["node_reverse_schedule"],
        "module_identity": _execution_mh1_hip_module_identity(metadata, target),
    }


def _cuda_block_linear_forward(
    descriptor: dict[str, Any],
    *,
    input_pointer: str,
    parameter_pointer: str,
    output_pointer: str,
    stem: str,
    add_scale: str | None = None,
) -> str:
    """Render a block-cooperative fixed-coordinate equivariant linear."""

    input_pointer = f"({input_pointer})"
    parameter_pointer = f"({parameter_pointer})"
    output_pointer = f"({output_pointer})"
    segments = {
        segment["name"]: segment
        for segment in descriptor["runtime_parameters"]["segments"]
    }
    weight_base = int(segments["weight"]["offset"])
    bias_base = int(segments["bias"]["offset"])
    bias_count = int(segments["bias"]["count"])
    mask_base = int(segments["output_mask"]["offset"])
    instructions_by_output: dict[int, list[dict[str, Any]]] = {}
    for instruction in descriptor["instructions"]:
        instructions_by_output.setdefault(int(instruction["output_block"]), []).append(
            instruction
        )
    lines: list[str] = []
    for output_block_index, output_block in enumerate(descriptor["blocks"]["output"]):
        multiplicity = int(output_block["multiplicity"])
        components = int(output_block["components"])
        output_base = int(output_block["offset"])
        work_count = multiplicity * components
        lines.extend(
            [
                "{",
                f"    for (int {stem}_work = threadIdx.x;",
                f"         {stem}_work < {work_count}; {stem}_work += blockDim.x) {{",
                f"        const int {stem}_component = {stem}_work / {multiplicity};",
                f"        const int {stem}_target = {stem}_work % {multiplicity};",
                f"        const int {stem}_output_index = {output_base}",
                f"            + {stem}_component * {multiplicity} + {stem}_target;",
                (
                    f"        float {stem}_value = {parameter_pointer}[{bias_base}"
                    f" + {stem}_output_index];"
                    if bias_count
                    else f"        float {stem}_value = 0.0f;"
                ),
            ]
        )
        for path_index, instruction in enumerate(
            instructions_by_output.get(output_block_index, [])
        ):
            input_block = descriptor["blocks"]["input"][int(instruction["input_block"])]
            input_multiplicity = int(input_block["multiplicity"])
            input_base = int(input_block["offset"])
            weight_offset = weight_base + int(instruction["weight_offset"])
            path_weight = _cpp_float(instruction["path_weight"])
            lines.extend(
                [
                    f"        for (int {stem}_source_{path_index} = 0;",
                    f"             {stem}_source_{path_index} < {input_multiplicity};",
                    f"             ++{stem}_source_{path_index})",
                    f"            {stem}_value += {path_weight}",
                    f"                * {parameter_pointer}[{weight_offset}",
                    f"                    + {stem}_source_{path_index} * {multiplicity}",
                    f"                    + {stem}_target]",
                    f"                * {input_pointer}[{input_base}",
                    f"                    + {stem}_component * {input_multiplicity}",
                    f"                    + {stem}_source_{path_index}];",
                ]
            )
        assignment = f"{output_pointer}[{stem}_output_index]"
        scaled_value = (
            f"({add_scale}) * {stem}_value"
            if add_scale is not None
            else f"{stem}_value"
        )
        operator = "+=" if add_scale is not None else "="
        lines.extend(
            [
                f"        {assignment} {operator} {scaled_value}",
                f"            * {parameter_pointer}[{mask_base} + {stem}_output_index];",
                "    }",
                "}",
            ]
        )
    lines.append("__syncthreads();")
    return "\n".join(lines)


def _cuda_block_linear_reverse(
    descriptor: dict[str, Any],
    *,
    output_adjoint_pointer: str,
    parameter_pointer: str,
    input_adjoint_pointer: str,
    stem: str,
    add: bool = False,
) -> str:
    """Render the input adjoint of a cooperative equivariant linear."""

    output_adjoint_pointer = f"({output_adjoint_pointer})"
    parameter_pointer = f"({parameter_pointer})"
    input_adjoint_pointer = f"({input_adjoint_pointer})"
    segments = {
        segment["name"]: segment
        for segment in descriptor["runtime_parameters"]["segments"]
    }
    weight_base = int(segments["weight"]["offset"])
    mask_base = int(segments["output_mask"]["offset"])
    instructions_by_input: dict[int, list[dict[str, Any]]] = {}
    for instruction in descriptor["instructions"]:
        instructions_by_input.setdefault(int(instruction["input_block"]), []).append(
            instruction
        )
    lines: list[str] = []
    for input_block_index, input_block in enumerate(descriptor["blocks"]["input"]):
        multiplicity = int(input_block["multiplicity"])
        components = int(input_block["components"])
        input_base = int(input_block["offset"])
        work_count = multiplicity * components
        lines.extend(
            [
                "{",
                f"    for (int {stem}_work = threadIdx.x;",
                f"         {stem}_work < {work_count}; {stem}_work += blockDim.x) {{",
                f"        const int {stem}_component = {stem}_work / {multiplicity};",
                f"        const int {stem}_source = {stem}_work % {multiplicity};",
                f"        float {stem}_value = 0.0f;",
            ]
        )
        for path_index, instruction in enumerate(
            instructions_by_input.get(input_block_index, [])
        ):
            output_block = descriptor["blocks"]["output"][
                int(instruction["output_block"])
            ]
            output_multiplicity = int(output_block["multiplicity"])
            output_base = int(output_block["offset"])
            weight_offset = weight_base + int(instruction["weight_offset"])
            path_weight = _cpp_float(instruction["path_weight"])
            lines.extend(
                [
                    f"        for (int {stem}_target_{path_index} = 0;",
                    f"             {stem}_target_{path_index} < {output_multiplicity};",
                    f"             ++{stem}_target_{path_index}) {{",
                    f"            const int {stem}_output_index_{path_index} = {output_base}",
                    f"                + {stem}_component * {output_multiplicity}",
                    f"                + {stem}_target_{path_index};",
                    f"            {stem}_value += {path_weight}",
                    f"                * {parameter_pointer}[{weight_offset}",
                    f"                    + {stem}_source * {output_multiplicity}",
                    f"                    + {stem}_target_{path_index}]",
                    f"                * {parameter_pointer}[{mask_base}",
                    f"                    + {stem}_output_index_{path_index}]",
                    f"                * {output_adjoint_pointer}[{stem}_output_index_{path_index}];",
                    "        }",
                ]
            )
        operator = "+=" if add else "="
        lines.extend(
            [
                f"        {input_adjoint_pointer}[{input_base}",
                f"            + {stem}_component * {multiplicity} + {stem}_source]",
                f"            {operator} {stem}_value;",
                "    }",
                "}",
            ]
        )
    lines.append("__syncthreads();")
    return "\n".join(lines)


_CUDA_NODE_TILE_NODES = 8
_CUDA_NODE_TILE_CHANNELS = 32
_CUDA_NODE_TILE_THREADS = _CUDA_NODE_TILE_NODES * _CUDA_NODE_TILE_CHANNELS
_CUDA_NODE_MESSAGE_TILE_NODES = 2 * _CUDA_NODE_TILE_NODES
_CUDA_NODE_LINEAR2_TILE_NODES = 4 * _CUDA_NODE_TILE_NODES
_CUDA_NODE_WIDE_GROUPED_TILE_NODES = 8 * _CUDA_NODE_TILE_NODES
_CUDA_NODE_XWIDE_GROUPED_TILE_NODES = 16 * _CUDA_NODE_TILE_NODES
_CUDA_NODE_PRODUCT_TILE_CHANNELS = 128


def _cuda_tiled_linear_forward(
    descriptor: dict[str, Any],
    *,
    args_type: str,
    input_pointer: str,
    input_stride: int,
    parameter_pointer: str,
    output_pointer: str,
    output_stride: int,
    stem: str,
    add_scale: str | None = None,
    tile_nodes: int = _CUDA_NODE_TILE_NODES,
    symbol_prefix: str = "",
) -> tuple[str, str, list[MH1CudaKernelLaunch]]:
    """Render graph-wide node/channel tiled equivariant linear kernels."""

    if tile_nodes not in (
        _CUDA_NODE_TILE_NODES,
        _CUDA_NODE_MESSAGE_TILE_NODES,
        _CUDA_NODE_LINEAR2_TILE_NODES,
    ):
        raise ValueError("MH1 tiled linear node count must be 8, 16, or 32")
    nodes_per_thread = tile_nodes // _CUDA_NODE_TILE_NODES

    input_pointer = f"({input_pointer})"
    parameter_pointer = f"({parameter_pointer})"
    output_pointer = f"({output_pointer})"
    segments = {
        segment["name"]: segment
        for segment in descriptor["runtime_parameters"]["segments"]
    }
    weight_base = int(segments["weight"]["offset"])
    bias_base = int(segments["bias"]["offset"])
    bias_count = int(segments["bias"]["count"])
    mask_base = int(segments["output_mask"]["offset"])
    instructions_by_output: dict[int, list[dict[str, Any]]] = {}
    for instruction in descriptor["instructions"]:
        instructions_by_output.setdefault(int(instruction["output_block"]), []).append(
            instruction
        )

    definitions: list[str] = []
    launches: list[str] = []
    launch_records: list[MH1CudaKernelLaunch] = []
    for output_block_index, output_block in enumerate(descriptor["blocks"]["output"]):
        output_multiplicity = int(output_block["multiplicity"])
        components = int(output_block["components"])
        output_base = int(output_block["offset"])
        initial_value = (
            f"{parameter_pointer}[{bias_base} + output_index]" if bias_count else "0.0f"
        )
        body: list[str] = [
            f"float value_{node_slot} = {initial_value};"
            for node_slot in range(nodes_per_thread)
        ]
        for path_index, instruction in enumerate(
            instructions_by_output.get(output_block_index, [])
        ):
            input_block = descriptor["blocks"]["input"][int(instruction["input_block"])]
            input_multiplicity = int(input_block["multiplicity"])
            input_base = int(input_block["offset"])
            weight_offset = weight_base + int(instruction["weight_offset"])
            path_weight = _cpp_float(instruction["path_weight"])
            body.extend(
                [
                    f"for (int source_base_{path_index} = 0;",
                    f"     source_base_{path_index} < {input_multiplicity};",
                    f"     source_base_{path_index} += tile_channels) {{",
                    "    for (int load = threadIdx.x; load < tile_nodes * tile_channels;",
                    "         load += blockDim.x) {",
                    "        const int shared_node = load / tile_channels;",
                    "        const int shared_source = load % tile_channels;",
                    f"    const int source_{path_index} = source_base_{path_index} + shared_source;",
                    "        input_tile[shared_node][shared_source] =",
                    f"            node_base + shared_node < args.num_nodes && source_{path_index} < {input_multiplicity}",
                    f"            ? {input_pointer}[(node_base + shared_node) * {input_stride}",
                    f"                + {input_base} + component * {input_multiplicity} + source_{path_index}]",
                    "            : 0.0f;",
                    "    }",
                    "    for (int load = threadIdx.x; load < tile_channels * tile_channels;",
                    "         load += blockDim.x) {",
                    "        const int shared_weight_source = load / tile_channels;",
                    "        const int shared_weight_target = load % tile_channels;",
                    f"        const int weight_source_{path_index} = source_base_{path_index} + shared_weight_source;",
                    "        const int weight_target = target_base + shared_weight_target;",
                    "        weight_tile[shared_weight_source][shared_weight_target] =",
                    f"            weight_source_{path_index} < {input_multiplicity}",
                    f"                && weight_target < {output_multiplicity}",
                    f"            ? {path_weight} * {parameter_pointer}[{weight_offset}",
                    f"                + weight_source_{path_index} * {output_multiplicity} + weight_target]",
                    "            : 0.0f;",
                    "    }",
                    "    __syncthreads();",
                    "    #pragma unroll",
                    "    for (int source = 0; source < tile_channels; ++source) {",
                    *(
                        f"        value_{node_slot} += input_tile[node_lane + {node_slot * _CUDA_NODE_TILE_NODES}][source]"
                        " * weight_tile[source][target_lane];"
                        for node_slot in range(nodes_per_thread)
                    ),
                    "    }",
                    "    __syncthreads();",
                    "}",
                ]
            )
        operator = "+=" if add_scale is not None else "="
        for node_slot in range(nodes_per_thread):
            node_add_scale = (
                add_scale.replace("[node]", f"[node_{node_slot}]")
                if add_scale is not None
                else None
            )
            scaled = (
                f"({node_add_scale}) * value_{node_slot}"
                if node_add_scale is not None
                else f"value_{node_slot}"
            )
            body.extend(
                [
                    f"if (node_valid_{node_slot} && target_valid)",
                    f"    {output_pointer}[node_{node_slot} * {output_stride} + output_index] {operator}",
                    f"        {scaled} * {parameter_pointer}[{mask_base} + output_index];",
                ]
            )
        kernel_name = f"{symbol_prefix}{stem}_block_{output_block_index}"
        linkage = 'extern "C" ' if symbol_prefix else ""
        definitions.append(
            f"""{linkage}__global__ void {kernel_name}({args_type} args)
{{
    constexpr int tile_nodes = {tile_nodes};
    constexpr int tile_channels = {_CUDA_NODE_TILE_CHANNELS};
    const int node_lane = threadIdx.x / tile_channels;
    const int target_lane = threadIdx.x % tile_channels;
    const std::int64_t node_base = static_cast<std::int64_t>(blockIdx.y) * tile_nodes;
{chr(10).join(f"    const std::int64_t node_{node_slot} = node_base + node_lane + {node_slot * _CUDA_NODE_TILE_NODES};" for node_slot in range(nodes_per_thread))}
    const std::int64_t node = node_0;
    const int target_base = blockIdx.x * tile_channels;
    const int target = target_base + target_lane;
    const int component = blockIdx.z;
{chr(10).join(f"    const bool node_valid_{node_slot} = node_{node_slot} < args.num_nodes;" for node_slot in range(nodes_per_thread))}
    const bool target_valid = target < {output_multiplicity};
    const int output_index = {output_base} + component * {output_multiplicity} + target;
    __shared__ float input_tile[tile_nodes][tile_channels];
    __shared__ float weight_tile[tile_channels][tile_channels + 1];
{chr(10).join('    ' + line for line in body)}
}}"""
        )
        launches.append(
            f"{kernel_name}<<<dim3(({output_multiplicity} + {_CUDA_NODE_TILE_CHANNELS - 1})"
            f" / {_CUDA_NODE_TILE_CHANNELS}, (args->num_nodes + {tile_nodes - 1})"
            f" / {tile_nodes}, {components}), {_CUDA_NODE_TILE_THREADS}, 0, cuda_stream>>>(*args);"
        )
        launch_records.append(
            MH1CudaKernelLaunch(
                kernel_name,
                "node_channel_tiles",
                _CUDA_NODE_TILE_THREADS,
                output_multiplicity,
                components,
                tile_nodes,
                _CUDA_NODE_TILE_CHANNELS,
            )
        )
    return "\n\n".join(definitions), "\n    ".join(launches), launch_records


def _cuda_tiled_linear_reverse(
    descriptor: dict[str, Any],
    *,
    args_type: str,
    output_adjoint_pointer: str,
    output_stride: int,
    parameter_pointer: str,
    input_adjoint_pointer: str,
    input_stride: int,
    stem: str,
    add: bool = False,
    tile_nodes: int = _CUDA_NODE_TILE_NODES,
    symbol_prefix: str = "",
) -> tuple[str, str, list[MH1CudaKernelLaunch]]:
    """Render graph-wide tiled transposed equivariant linear kernels."""

    if tile_nodes not in (
        _CUDA_NODE_TILE_NODES,
        _CUDA_NODE_MESSAGE_TILE_NODES,
        _CUDA_NODE_LINEAR2_TILE_NODES,
    ):
        raise ValueError("MH1 tiled linear node count must be 8, 16, or 32")
    nodes_per_thread = tile_nodes // _CUDA_NODE_TILE_NODES

    output_adjoint_pointer = f"({output_adjoint_pointer})"
    parameter_pointer = f"({parameter_pointer})"
    input_adjoint_pointer = f"({input_adjoint_pointer})"
    segments = {
        segment["name"]: segment
        for segment in descriptor["runtime_parameters"]["segments"]
    }
    weight_base = int(segments["weight"]["offset"])
    mask_base = int(segments["output_mask"]["offset"])
    instructions_by_input: dict[int, list[dict[str, Any]]] = {}
    for instruction in descriptor["instructions"]:
        instructions_by_input.setdefault(int(instruction["input_block"]), []).append(
            instruction
        )

    definitions: list[str] = []
    launches: list[str] = []
    launch_records: list[MH1CudaKernelLaunch] = []
    for input_block_index, input_block in enumerate(descriptor["blocks"]["input"]):
        input_multiplicity = int(input_block["multiplicity"])
        components = int(input_block["components"])
        input_base = int(input_block["offset"])
        body = [
            f"float value_{node_slot} = 0.0f;" for node_slot in range(nodes_per_thread)
        ]
        for path_index, instruction in enumerate(
            instructions_by_input.get(input_block_index, [])
        ):
            output_block = descriptor["blocks"]["output"][
                int(instruction["output_block"])
            ]
            output_multiplicity = int(output_block["multiplicity"])
            output_base = int(output_block["offset"])
            weight_offset = weight_base + int(instruction["weight_offset"])
            path_weight = _cpp_float(instruction["path_weight"])
            body.extend(
                [
                    f"for (int target_base_{path_index} = 0;",
                    f"     target_base_{path_index} < {output_multiplicity};",
                    f"     target_base_{path_index} += tile_channels) {{",
                    "    for (int load = threadIdx.x; load < tile_nodes * tile_channels;",
                    "         load += blockDim.x) {",
                    "        const int shared_node = load / tile_channels;",
                    "        const int shared_target = load % tile_channels;",
                    f"    const int target_{path_index} = target_base_{path_index} + shared_target;",
                    f"    const int output_index_{path_index} = {output_base}",
                    f"        + component * {output_multiplicity} + target_{path_index};",
                    "        adjoint_tile[shared_node][shared_target] =",
                    f"            node_base + shared_node < args.num_nodes && target_{path_index} < {output_multiplicity}",
                    f"            ? {output_adjoint_pointer}[(node_base + shared_node) * {output_stride}",
                    f"                + output_index_{path_index}] * {parameter_pointer}[{mask_base}",
                    f"                + output_index_{path_index}]",
                    "            : 0.0f;",
                    "    }",
                    "    for (int load = threadIdx.x; load < tile_channels * tile_channels;",
                    "         load += blockDim.x) {",
                    "        const int shared_weight_source = load / tile_channels;",
                    "        const int shared_weight_target = load % tile_channels;",
                    "        const int weight_source = source_base + shared_weight_source;",
                    f"        const int weight_target_{path_index} = target_base_{path_index} + shared_weight_target;",
                    "        weight_tile[shared_weight_source][shared_weight_target] =",
                    f"            weight_source < {input_multiplicity}",
                    f"                && weight_target_{path_index} < {output_multiplicity}",
                    f"            ? {path_weight} * {parameter_pointer}[{weight_offset}",
                    f"                + weight_source * {output_multiplicity} + weight_target_{path_index}]",
                    "            : 0.0f;",
                    "    }",
                    "    __syncthreads();",
                    "    #pragma unroll",
                    "    for (int target = 0; target < tile_channels; ++target) {",
                    *(
                        f"        value_{node_slot} += adjoint_tile[node_lane + {node_slot * _CUDA_NODE_TILE_NODES}][target]"
                        " * weight_tile[source_lane][target];"
                        for node_slot in range(nodes_per_thread)
                    ),
                    "    }",
                    "    __syncthreads();",
                    "}",
                ]
            )
        operator = "+=" if add else "="
        for node_slot in range(nodes_per_thread):
            body.extend(
                [
                    f"if (node_valid_{node_slot} && source_valid)",
                    f"    {input_adjoint_pointer}[node_{node_slot} * {input_stride} + input_index] {operator} value_{node_slot};",
                ]
            )
        kernel_name = f"{symbol_prefix}{stem}_block_{input_block_index}"
        linkage = 'extern "C" ' if symbol_prefix else ""
        definitions.append(
            f"""{linkage}__global__ void {kernel_name}({args_type} args)
{{
    constexpr int tile_nodes = {tile_nodes};
    constexpr int tile_channels = {_CUDA_NODE_TILE_CHANNELS};
    const int node_lane = threadIdx.x / tile_channels;
    const int source_lane = threadIdx.x % tile_channels;
    const std::int64_t node_base = static_cast<std::int64_t>(blockIdx.y) * tile_nodes;
{chr(10).join(f"    const std::int64_t node_{node_slot} = node_base + node_lane + {node_slot * _CUDA_NODE_TILE_NODES};" for node_slot in range(nodes_per_thread))}
    const int source_base = blockIdx.x * tile_channels;
    const int source = source_base + source_lane;
    const int component = blockIdx.z;
{chr(10).join(f"    const bool node_valid_{node_slot} = node_{node_slot} < args.num_nodes;" for node_slot in range(nodes_per_thread))}
    const bool source_valid = source < {input_multiplicity};
    const int input_index = {input_base} + component * {input_multiplicity} + source;
    __shared__ float adjoint_tile[tile_nodes][tile_channels];
    __shared__ float weight_tile[tile_channels][tile_channels + 1];
{chr(10).join('    ' + line for line in body)}
}}"""
        )
        launches.append(
            f"{kernel_name}<<<dim3(({input_multiplicity} + {_CUDA_NODE_TILE_CHANNELS - 1})"
            f" / {_CUDA_NODE_TILE_CHANNELS}, (args->num_nodes + {tile_nodes - 1})"
            f" / {tile_nodes}, {components}), {_CUDA_NODE_TILE_THREADS}, 0, cuda_stream>>>(*args);"
        )
        launch_records.append(
            MH1CudaKernelLaunch(
                kernel_name,
                "node_channel_tiles",
                _CUDA_NODE_TILE_THREADS,
                input_multiplicity,
                components,
                tile_nodes,
                _CUDA_NODE_TILE_CHANNELS,
            )
        )
    return "\n\n".join(definitions), "\n    ".join(launches), launch_records


def _cuda_grouped_tiled_linear_forward(
    descriptor: dict[str, Any],
    *,
    args_type: str,
    input_pointer: str,
    input_stride: int,
    parameter_pointer: str,
    output_pointer: str,
    output_stride: int,
    stem: str,
    add_scale: str | None = None,
    tile_nodes: int = _CUDA_NODE_LINEAR2_TILE_NODES,
    symbol_prefix: str = "",
) -> tuple[str, str, list[MH1CudaKernelLaunch]]:
    """Render one heterogeneous launch over tiled output irreps blocks."""

    if tile_nodes not in (
        _CUDA_NODE_LINEAR2_TILE_NODES,
        _CUDA_NODE_WIDE_GROUPED_TILE_NODES,
        _CUDA_NODE_XWIDE_GROUPED_TILE_NODES,
    ):
        raise ValueError(
            "MH1 grouped tiled linear forward requires 32, 64, or 128 node tiles"
        )
    nodes_per_thread = tile_nodes // _CUDA_NODE_TILE_NODES

    input_pointer = f"({input_pointer})"
    parameter_pointer = f"({parameter_pointer})"
    output_pointer = f"({output_pointer})"
    segments = {
        segment["name"]: segment
        for segment in descriptor["runtime_parameters"]["segments"]
    }
    weight_base = int(segments["weight"]["offset"])
    bias_base = int(segments["bias"]["offset"])
    bias_count = int(segments["bias"]["count"])
    mask_base = int(segments["output_mask"]["offset"])
    instructions_by_output: dict[int, list[dict[str, Any]]] = {}
    for instruction in descriptor["instructions"]:
        instructions_by_output.setdefault(int(instruction["output_block"]), []).append(
            instruction
        )

    logical_tile_base = 0
    branches: list[str] = []
    for output_block_index, output_block in enumerate(descriptor["blocks"]["output"]):
        output_multiplicity = int(output_block["multiplicity"])
        components = int(output_block["components"])
        output_base = int(output_block["offset"])
        target_tiles = (
            output_multiplicity + _CUDA_NODE_TILE_CHANNELS - 1
        ) // _CUDA_NODE_TILE_CHANNELS
        logical_tile_count = target_tiles * components
        initial_value = (
            f"{parameter_pointer}[{bias_base} + output_index]" if bias_count else "0.0f"
        )
        body = [
            f"float value_{node_slot} = {initial_value};"
            for node_slot in range(nodes_per_thread)
        ]
        for path_index, instruction in enumerate(
            instructions_by_output.get(output_block_index, [])
        ):
            input_block = descriptor["blocks"]["input"][int(instruction["input_block"])]
            input_multiplicity = int(input_block["multiplicity"])
            input_base = int(input_block["offset"])
            weight_offset = weight_base + int(instruction["weight_offset"])
            path_weight = _cpp_float(instruction["path_weight"])
            body.extend(
                [
                    f"for (int source_base_{path_index} = 0;",
                    f"     source_base_{path_index} < {input_multiplicity};",
                    f"     source_base_{path_index} += tile_channels) {{",
                    "    for (int load = threadIdx.x; load < tile_nodes * tile_channels;",
                    "         load += blockDim.x) {",
                    "        const int shared_node = load / tile_channels;",
                    "        const int shared_source = load % tile_channels;",
                    f"        const int source_{path_index} = source_base_{path_index} + shared_source;",
                    "        input_tile[shared_node][shared_source] =",
                    f"            node_base + shared_node < args.num_nodes && source_{path_index} < {input_multiplicity}",
                    f"            ? {input_pointer}[(node_base + shared_node) * {input_stride}",
                    f"                + {input_base} + component * {input_multiplicity} + source_{path_index}]",
                    "            : 0.0f;",
                    "    }",
                    "    for (int load = threadIdx.x; load < tile_channels * tile_channels;",
                    "         load += blockDim.x) {",
                    "        const int shared_weight_source = load / tile_channels;",
                    "        const int shared_weight_target = load % tile_channels;",
                    f"        const int weight_source_{path_index} = source_base_{path_index} + shared_weight_source;",
                    "        const int weight_target = target_base + shared_weight_target;",
                    "        weight_tile[shared_weight_source][shared_weight_target] =",
                    f"            weight_source_{path_index} < {input_multiplicity}",
                    f"                && weight_target < {output_multiplicity}",
                    f"            ? {path_weight} * {parameter_pointer}[{weight_offset}",
                    f"                + weight_source_{path_index} * {output_multiplicity} + weight_target]",
                    "            : 0.0f;",
                    "    }",
                    "    __syncthreads();",
                    "    #pragma unroll",
                    "    for (int source = 0; source < tile_channels; ++source) {",
                    *(
                        f"        value_{node_slot} += input_tile[node_lane + {node_slot * _CUDA_NODE_TILE_NODES}][source]"
                        " * weight_tile[source][target_lane];"
                        for node_slot in range(nodes_per_thread)
                    ),
                    "    }",
                    "    __syncthreads();",
                    "}",
                ]
            )
        operator = "+=" if add_scale is not None else "="
        for node_slot in range(nodes_per_thread):
            node_add_scale = (
                add_scale.replace("[node]", f"[node_{node_slot}]")
                if add_scale is not None
                else None
            )
            scaled = (
                f"({node_add_scale}) * value_{node_slot}"
                if node_add_scale is not None
                else f"value_{node_slot}"
            )
            body.extend(
                [
                    f"if (node_valid_{node_slot} && target_valid)",
                    f"    {output_pointer}[node_{node_slot} * {output_stride} + output_index] {operator}",
                    f"        {scaled} * {parameter_pointer}[{mask_base} + output_index];",
                ]
            )
        branch_prefix = "if" if not branches else "else if"
        branches.append(
            f"""{branch_prefix} (logical_tile < {logical_tile_base + logical_tile_count}) {{
        constexpr int target_tiles = {target_tiles};
        const int local_tile = logical_tile - {logical_tile_base};
        const int component = local_tile / target_tiles;
        const int target_base = (local_tile % target_tiles) * tile_channels;
        const int target = target_base + target_lane;
        const bool target_valid = target < {output_multiplicity};
        const int output_index = {output_base} + component * {output_multiplicity} + target;
{chr(10).join('        ' + line for line in body)}
    }}"""
        )
        logical_tile_base += logical_tile_count

    if logical_tile_base <= 0:
        raise ValueError("MH1 grouped tiled linear forward has no logical tiles")
    kernel_name = f"{symbol_prefix}{stem}_grouped"
    linkage = 'extern "C" ' if symbol_prefix else ""
    definition = f"""{linkage}__global__ void {kernel_name}({args_type} args)
{{
    constexpr int tile_nodes = {tile_nodes};
    constexpr int tile_channels = {_CUDA_NODE_TILE_CHANNELS};
    const int logical_tile = blockIdx.x;
    const int node_lane = threadIdx.x / tile_channels;
    const int target_lane = threadIdx.x % tile_channels;
    const std::int64_t node_base = static_cast<std::int64_t>(blockIdx.y) * tile_nodes;
{chr(10).join(f"    const std::int64_t node_{node_slot} = node_base + node_lane + {node_slot * _CUDA_NODE_TILE_NODES};" for node_slot in range(nodes_per_thread))}
    const std::int64_t node = node_0;
{chr(10).join(f"    const bool node_valid_{node_slot} = node_{node_slot} < args.num_nodes;" for node_slot in range(nodes_per_thread))}
    __shared__ float input_tile[tile_nodes][tile_channels];
    __shared__ float weight_tile[tile_channels][tile_channels + 1];
    {chr(10).join(branches)}
}}"""
    launch = (
        f"{kernel_name}<<<dim3({logical_tile_base}, (args->num_nodes + {tile_nodes - 1})"
        f" / {tile_nodes}, 1), {_CUDA_NODE_TILE_THREADS}, 0, cuda_stream>>>(*args);"
    )
    launch_record = MH1CudaKernelLaunch(
        kernel_name,
        "node_channel_tiles",
        _CUDA_NODE_TILE_THREADS,
        logical_tile_base * _CUDA_NODE_TILE_CHANNELS,
        1,
        tile_nodes,
        _CUDA_NODE_TILE_CHANNELS,
    )
    return definition, launch, [launch_record]


def _cuda_grouped_tiled_linear_reverse(
    descriptor: dict[str, Any],
    *,
    args_type: str,
    output_adjoint_pointer: str,
    output_stride: int,
    parameter_pointer: str,
    input_adjoint_pointer: str,
    input_stride: int,
    stem: str,
    add: bool = False,
    tile_nodes: int = _CUDA_NODE_LINEAR2_TILE_NODES,
    symbol_prefix: str = "",
    density_input_pointer: str | None = None,
    density_result_pointer: str | None = None,
) -> tuple[str, str, list[MH1CudaKernelLaunch]]:
    """Render one heterogeneous launch over tiled input irreps blocks."""

    if tile_nodes not in (
        _CUDA_NODE_LINEAR2_TILE_NODES,
        _CUDA_NODE_WIDE_GROUPED_TILE_NODES,
        _CUDA_NODE_XWIDE_GROUPED_TILE_NODES,
    ):
        raise ValueError(
            "MH1 grouped tiled linear reverse requires 32, 64, or 128 node tiles"
        )
    nodes_per_thread = tile_nodes // _CUDA_NODE_TILE_NODES

    output_adjoint_pointer = f"({output_adjoint_pointer})"
    parameter_pointer = f"({parameter_pointer})"
    input_adjoint_pointer = f"({input_adjoint_pointer})"
    if (density_input_pointer is None) != (density_result_pointer is None):
        raise ValueError("MH1 fused density pointers must be provided together")
    if density_input_pointer is not None:
        density_input_pointer = f"({density_input_pointer})"
        density_result_pointer = f"({density_result_pointer})"
    segments = {
        segment["name"]: segment
        for segment in descriptor["runtime_parameters"]["segments"]
    }
    weight_base = int(segments["weight"]["offset"])
    mask_base = int(segments["output_mask"]["offset"])
    instructions_by_input: dict[int, list[dict[str, Any]]] = {}
    for instruction in descriptor["instructions"]:
        instructions_by_input.setdefault(int(instruction["input_block"]), []).append(
            instruction
        )

    logical_tile_base = 0
    branches: list[str] = []
    for input_block_index, input_block in enumerate(descriptor["blocks"]["input"]):
        input_multiplicity = int(input_block["multiplicity"])
        components = int(input_block["components"])
        input_base = int(input_block["offset"])
        source_tiles = (
            input_multiplicity + _CUDA_NODE_TILE_CHANNELS - 1
        ) // _CUDA_NODE_TILE_CHANNELS
        logical_tile_count = source_tiles * components
        body = [
            f"float value_{node_slot} = 0.0f;" for node_slot in range(nodes_per_thread)
        ]
        for path_index, instruction in enumerate(
            instructions_by_input.get(input_block_index, [])
        ):
            output_block = descriptor["blocks"]["output"][
                int(instruction["output_block"])
            ]
            output_multiplicity = int(output_block["multiplicity"])
            output_base = int(output_block["offset"])
            weight_offset = weight_base + int(instruction["weight_offset"])
            path_weight = _cpp_float(instruction["path_weight"])
            body.extend(
                [
                    f"for (int target_base_{path_index} = 0;",
                    f"     target_base_{path_index} < {output_multiplicity};",
                    f"     target_base_{path_index} += tile_channels) {{",
                    "    for (int load = threadIdx.x; load < tile_nodes * tile_channels;",
                    "         load += blockDim.x) {",
                    "        const int shared_node = load / tile_channels;",
                    "        const int shared_target = load % tile_channels;",
                    f"        const int target_{path_index} = target_base_{path_index} + shared_target;",
                    f"        const int output_index_{path_index} = {output_base}",
                    f"            + component * {output_multiplicity} + target_{path_index};",
                    "        adjoint_tile[shared_node][shared_target] =",
                    f"            node_base + shared_node < args.num_nodes && target_{path_index} < {output_multiplicity}",
                    f"            ? {output_adjoint_pointer}[(node_base + shared_node) * {output_stride}",
                    f"                + output_index_{path_index}] * {parameter_pointer}[{mask_base}",
                    f"                + output_index_{path_index}]",
                    "            : 0.0f;",
                    "    }",
                    "    for (int load = threadIdx.x; load < tile_channels * tile_channels;",
                    "         load += blockDim.x) {",
                    "        const int shared_weight_source = load / tile_channels;",
                    "        const int shared_weight_target = load % tile_channels;",
                    "        const int weight_source = source_base + shared_weight_source;",
                    f"        const int weight_target_{path_index} = target_base_{path_index} + shared_weight_target;",
                    "        weight_tile[shared_weight_source][shared_weight_target] =",
                    f"            weight_source < {input_multiplicity}",
                    f"                && weight_target_{path_index} < {output_multiplicity}",
                    f"            ? {path_weight} * {parameter_pointer}[{weight_offset}",
                    f"                + weight_source * {output_multiplicity} + weight_target_{path_index}]",
                    "            : 0.0f;",
                    "    }",
                    "    __syncthreads();",
                    "    #pragma unroll",
                    "    for (int target = 0; target < tile_channels; ++target) {",
                    *(
                        f"        value_{node_slot} += adjoint_tile[node_lane + {node_slot * _CUDA_NODE_TILE_NODES}][target]"
                        " * weight_tile[source_lane][target];"
                        for node_slot in range(nodes_per_thread)
                    ),
                    "    }",
                    "    __syncthreads();",
                    "}",
                ]
            )
        operator = "+=" if add else "="
        for node_slot in range(nodes_per_thread):
            if density_input_pointer is not None:
                body.extend(
                    [
                        f"float density_dot_{node_slot} = node_valid_{node_slot} && source_valid",
                        f"    ? {density_input_pointer}[node_{node_slot} * {input_stride} + input_index]",
                        f"        * value_{node_slot} : 0.0f;",
                        "for (int offset = tile_channels / 2; offset > 0; offset /= 2)",
                        f"    density_dot_{node_slot} += SYMMETRIX_JIT_MH1_SHUFFLE_DOWN(",
                        f"        density_dot_{node_slot}, offset, tile_channels);",
                        f"if (source_lane == 0 && node_valid_{node_slot})",
                        f"    atomicAdd({density_result_pointer} + node_{node_slot},",
                        f"        density_dot_{node_slot});",
                    ]
                )
            body.extend(
                [
                    f"if (node_valid_{node_slot} && source_valid)",
                    f"    {input_adjoint_pointer}[node_{node_slot} * {input_stride} + input_index] {operator} value_{node_slot};",
                ]
            )
        branch_prefix = "if" if not branches else "else if"
        branches.append(
            f"""{branch_prefix} (logical_tile < {logical_tile_base + logical_tile_count}) {{
        constexpr int source_tiles = {source_tiles};
        const int local_tile = logical_tile - {logical_tile_base};
        const int component = local_tile / source_tiles;
        const int source_base = (local_tile % source_tiles) * tile_channels;
        const int source = source_base + source_lane;
        const bool source_valid = source < {input_multiplicity};
        const int input_index = {input_base} + component * {input_multiplicity} + source;
{chr(10).join('        ' + line for line in body)}
    }}"""
        )
        logical_tile_base += logical_tile_count

    if logical_tile_base <= 0:
        raise ValueError("MH1 grouped tiled linear reverse has no logical tiles")
    kernel_name = f"{symbol_prefix}{stem}_grouped"
    linkage = 'extern "C" ' if symbol_prefix else ""
    definition = f"""{linkage}__global__ void {kernel_name}({args_type} args)
{{
    constexpr int tile_nodes = {tile_nodes};
    constexpr int tile_channels = {_CUDA_NODE_TILE_CHANNELS};
    const int logical_tile = blockIdx.x;
    const int node_lane = threadIdx.x / tile_channels;
    const int source_lane = threadIdx.x % tile_channels;
    const std::int64_t node_base = static_cast<std::int64_t>(blockIdx.y) * tile_nodes;
{chr(10).join(f"    const std::int64_t node_{node_slot} = node_base + node_lane + {node_slot * _CUDA_NODE_TILE_NODES};" for node_slot in range(nodes_per_thread))}
{chr(10).join(f"    const bool node_valid_{node_slot} = node_{node_slot} < args.num_nodes;" for node_slot in range(nodes_per_thread))}
    __shared__ float adjoint_tile[tile_nodes][tile_channels];
    __shared__ float weight_tile[tile_channels][tile_channels + 1];
    {chr(10).join(branches)}
}}"""
    launch = (
        f"{kernel_name}<<<dim3({logical_tile_base}, (args->num_nodes + {tile_nodes - 1})"
        f" / {tile_nodes}, 1), {_CUDA_NODE_TILE_THREADS}, 0, cuda_stream>>>(*args);"
    )
    launch_record = MH1CudaKernelLaunch(
        kernel_name,
        "node_channel_tiles",
        _CUDA_NODE_TILE_THREADS,
        logical_tile_base * _CUDA_NODE_TILE_CHANNELS,
        1,
        tile_nodes,
        _CUDA_NODE_TILE_CHANNELS,
    )
    return definition, launch, [launch_record]


def _cuda_block_embedding_lookup(
    descriptor: dict[str, Any],
    *,
    element: str,
    parameter_pointer: str,
    output_pointer: str,
    stem: str,
) -> str:
    parameter_pointer = f"({parameter_pointer})"
    output_pointer = f"({output_pointer})"
    path = descriptor["instructions"][0]
    output_multiplicity = int(descriptor["blocks"]["output"][0]["multiplicity"])
    segments = {
        segment["name"]: segment
        for segment in descriptor["runtime_parameters"]["segments"]
    }
    weight_base = int(segments["weight"]["offset"])
    bias_base = int(segments["bias"]["offset"])
    bias_count = int(segments["bias"]["count"])
    mask_base = int(segments["output_mask"]["offset"])
    bias = f" + {parameter_pointer}[{bias_base} + {stem}_target]" if bias_count else ""
    return f"""for (int {stem}_target = threadIdx.x;
     {stem}_target < {output_multiplicity}; {stem}_target += blockDim.x)
    {output_pointer}[{stem}_target] = ({_cpp_float(path['path_weight'])}
        * {parameter_pointer}[{weight_base} + ({element}) * {output_multiplicity}
            + {stem}_target]{bias})
        * {parameter_pointer}[{mask_base} + {stem}_target];
__syncthreads();"""


def _cuda_block_gate_forward(
    gate: dict[str, Any],
    *,
    input_pointer: str,
    output_pointer: str,
    stem: str,
) -> str:
    input_pointer = f"({input_pointer})"
    output_pointer = f"({output_pointer})"
    lines: list[str] = []
    scalar_offset = 0
    for block_index, (block, constant) in enumerate(
        zip(
            gate["blocks"]["irreps_scalars"],
            gate["scalar_activation"]["constants"],
            strict=True,
        )
    ):
        multiplicity = int(block["multiplicity"])
        scale = _cpp_float(constant)
        lines.extend(
            [
                f"for (int {stem}_scalar_{block_index} = threadIdx.x;",
                f"     {stem}_scalar_{block_index} < {multiplicity};",
                f"     {stem}_scalar_{block_index} += blockDim.x) {{",
                f"    const int column = {scalar_offset} + {stem}_scalar_{block_index};",
                f"    const float value = {input_pointer}[column];",
                "    const float sigmoid = 1.0f / (1.0f + expf(-value));",
                f"    {output_pointer}[column] = {scale} * value * sigmoid;",
                "}",
            ]
        )
        scalar_offset += multiplicity
    for pairing_index, (pairing, constant) in enumerate(
        zip(gate["pairing"], gate["gate_activation"]["constants"], strict=True)
    ):
        multiplicity = int(pairing["multiplicity"])
        width = int(pairing["component_width"])
        gate_offset = int(pairing["gate_offset"])
        gated_offset = int(pairing["gated_offset"])
        output_offset = int(pairing["output_offset"])
        scale = _cpp_float(constant)
        lines.extend(
            [
                f"for (int {stem}_gate_{pairing_index} = threadIdx.x;",
                f"     {stem}_gate_{pairing_index} < {multiplicity};",
                f"     {stem}_gate_{pairing_index} += blockDim.x) {{",
                f"    const float gate_input = {input_pointer}[{gate_offset}",
                f"        + {stem}_gate_{pairing_index}];",
                "    const float sigmoid = 1.0f / (1.0f + expf(-gate_input));",
                f"    const float gate_value = {scale} * sigmoid;",
                f"    for (int component = 0; component < {width}; ++component)",
                f"        {output_pointer}[{output_offset} + component * {multiplicity}",
                f"            + {stem}_gate_{pairing_index}]",
                f"            = {input_pointer}[{gated_offset} + component * {multiplicity}",
                f"                + {stem}_gate_{pairing_index}] * gate_value;",
                "}",
            ]
        )
    lines.append("__syncthreads();")
    return "\n".join(lines)


def _cuda_block_gate_reverse_inplace(
    gate: dict[str, Any],
    *,
    input_and_adjoint_pointer: str,
    output_adjoint_pointer: str,
    stem: str,
    retained_input_pointer: str | None = None,
) -> str:
    input_and_adjoint_pointer = f"({input_and_adjoint_pointer})"
    input_pointer = (
        input_and_adjoint_pointer
        if retained_input_pointer is None
        else f"({retained_input_pointer})"
    )
    output_adjoint_pointer = f"({output_adjoint_pointer})"
    lines: list[str] = []
    scalar_offset = 0
    for block_index, (block, constant) in enumerate(
        zip(
            gate["blocks"]["irreps_scalars"],
            gate["scalar_activation"]["constants"],
            strict=True,
        )
    ):
        multiplicity = int(block["multiplicity"])
        scale = _cpp_float(constant)
        lines.extend(
            [
                f"for (int {stem}_scalar_{block_index} = threadIdx.x;",
                f"     {stem}_scalar_{block_index} < {multiplicity};",
                f"     {stem}_scalar_{block_index} += blockDim.x) {{",
                f"    const int column = {scalar_offset} + {stem}_scalar_{block_index};",
                f"    const float value = {input_pointer}[column];",
                "    const float sigmoid = 1.0f / (1.0f + expf(-value));",
                f"    {input_and_adjoint_pointer}[column] =",
                f"        {output_adjoint_pointer}[column] * {scale}",
                "        * (sigmoid + value * sigmoid * (1.0f - sigmoid));",
                "}",
            ]
        )
        scalar_offset += multiplicity
    for pairing_index, (pairing, constant) in enumerate(
        zip(gate["pairing"], gate["gate_activation"]["constants"], strict=True)
    ):
        multiplicity = int(pairing["multiplicity"])
        width = int(pairing["component_width"])
        gate_offset = int(pairing["gate_offset"])
        gated_offset = int(pairing["gated_offset"])
        output_offset = int(pairing["output_offset"])
        scale = _cpp_float(constant)
        lines.extend(
            [
                f"for (int {stem}_gate_{pairing_index} = threadIdx.x;",
                f"     {stem}_gate_{pairing_index} < {multiplicity};",
                f"     {stem}_gate_{pairing_index} += blockDim.x) {{",
                f"    const int gate_column = {gate_offset} + {stem}_gate_{pairing_index};",
                f"    const float gate_input = {input_pointer}[gate_column];",
                "    const float sigmoid = 1.0f / (1.0f + expf(-gate_input));",
                f"    const float gate_value = {scale} * sigmoid;",
                "    float gate_adjoint = 0.0f;",
                f"    for (int component = 0; component < {width}; ++component) {{",
                f"        const int input_column = {gated_offset} + component * {multiplicity}",
                f"            + {stem}_gate_{pairing_index};",
                f"        const int output_column = {output_offset} + component * {multiplicity}",
                f"            + {stem}_gate_{pairing_index};",
                f"        const float input_value = {input_pointer}[input_column];",
                f"        const float adjoint = {output_adjoint_pointer}[output_column];",
                "        gate_adjoint += adjoint * input_value;",
                f"        {input_and_adjoint_pointer}[input_column] = adjoint * gate_value;",
                "    }",
                f"    {input_and_adjoint_pointer}[gate_column] = gate_adjoint * {scale}",
                "        * sigmoid * (1.0f - sigmoid);",
                "}",
            ]
        )
    lines.append("__syncthreads();")
    return "\n".join(lines)


def _cuda_block_product_forward(
    product: dict[str, Any],
    *,
    input_pointer: str,
    coefficient_pointer: str,
    contracted_pointer: str,
    stem: str,
) -> str:
    input_pointer = f"({input_pointer})"
    coefficient_pointer = f"({coefficient_pointer})"
    contracted_pointer = f"({contracted_pointer})"
    channels = int(product["dimensions"]["channels"])
    output_dimension = int(product["dimensions"]["output"])
    lines = [
        f"for (int {stem}_column = threadIdx.x; {stem}_column < {output_dimension};",
        f"     {stem}_column += blockDim.x)",
        f"    {contracted_pointer}[{stem}_column] = 0.0f;",
        "__syncthreads();",
    ]
    for component_index, component in enumerate(product["components"]):
        first = int(component["term_begin"])
        last = int(component["term_end"])
        if first == last:
            continue
        output_offset = int(component["output_component_offset"])
        lines.extend(
            [
                f"for (int {stem}_channel_{component_index} = threadIdx.x;",
                f"     {stem}_channel_{component_index} < {channels};",
                f"     {stem}_channel_{component_index} += blockDim.x) {{",
                "    float value = 0.0f;",
            ]
        )
        for term_index in range(first, last):
            term = product["terms"][term_index]
            factors = " * ".join(
                f"{input_pointer}[{int(product['angular_layout'][int(axis)]['component_offset'])}"
                f" + {stem}_channel_{component_index}]"
                for axis in term["angular_indices"]
            )
            lines.append(
                f"    value += {coefficient_pointer}[{term_index * channels}"
                f" + {stem}_channel_{component_index}] * {factors};"
            )
        lines.extend(
            [
                f"    {contracted_pointer}[{output_offset}",
                f"        + {stem}_channel_{component_index}] = value;",
                "}",
            ]
        )
    lines.append("__syncthreads();")
    return "\n".join(lines)


def _cuda_block_product_reverse(
    product: dict[str, Any],
    *,
    input_pointer: str,
    contracted_adjoint_pointer: str,
    coefficient_pointer: str,
    input_adjoint_pointer: str,
    stem: str,
) -> str:
    input_pointer = f"({input_pointer})"
    contracted_adjoint_pointer = f"({contracted_adjoint_pointer})"
    coefficient_pointer = f"({coefficient_pointer})"
    input_adjoint_pointer = f"({input_adjoint_pointer})"
    channels = int(product["dimensions"]["channels"])
    lines: list[str] = []
    terms_by_angular: dict[int, list[tuple[int, list[int]]]] = {
        index: [] for index in range(len(product["angular_layout"]))
    }
    component_by_term: dict[int, int] = {}
    for component in product["components"]:
        output_offset = int(component["output_component_offset"])
        for term_index in range(
            int(component["term_begin"]), int(component["term_end"])
        ):
            component_by_term[term_index] = output_offset
    for term_index, term in enumerate(product["terms"]):
        angular_indices = [int(value) for value in term["angular_indices"]]
        for differentiated, angular_index in enumerate(angular_indices):
            remaining = [
                value
                for axis, value in enumerate(angular_indices)
                if axis != differentiated
            ]
            terms_by_angular[angular_index].append((term_index, remaining))
    for angular_index, entries in terms_by_angular.items():
        input_offset = int(product["angular_layout"][angular_index]["component_offset"])
        lines.extend(
            [
                f"for (int {stem}_channel_{angular_index} = threadIdx.x;",
                f"     {stem}_channel_{angular_index} < {channels};",
                f"     {stem}_channel_{angular_index} += blockDim.x) {{",
                "    float value = 0.0f;",
            ]
        )
        for term_index, remaining in entries:
            factors = "".join(
                f" * {input_pointer}[{int(product['angular_layout'][axis]['component_offset'])}"
                f" + {stem}_channel_{angular_index}]"
                for axis in remaining
            )
            output_offset = component_by_term[term_index]
            lines.append(
                f"    value += {coefficient_pointer}[{term_index * channels}"
                f" + {stem}_channel_{angular_index}]"
                f" * {contracted_adjoint_pointer}[{output_offset}"
                f" + {stem}_channel_{angular_index}]{factors};"
            )
        lines.extend(
            [
                f"    {input_adjoint_pointer}[{input_offset}",
                f"        + {stem}_channel_{angular_index}] = value;",
                "}",
            ]
        )
    lines.append("__syncthreads();")
    return "\n".join(lines)


def _cuda_tiled_product_forward(
    product: dict[str, Any],
    *,
    args_type: str,
    input_pointer: str,
    input_stride: int,
    coefficient_pointer: str,
    contracted_pointer: str,
    contracted_stride: int,
    stem: str,
    symbol_prefix: str = "",
) -> tuple[str, str, list[MH1CudaKernelLaunch]]:
    """Render product contraction over independent output and channel tiles."""

    input_pointer = f"({input_pointer})"
    coefficient_pointer = f"({coefficient_pointer})"
    contracted_pointer = f"({contracted_pointer})"
    channels = int(product["dimensions"]["channels"])
    output_dimension = int(product["dimensions"]["output"])
    if output_dimension % channels:
        raise ValueError("MH1 product output is not channel aligned")
    output_components = output_dimension // channels
    components_by_output: dict[int, dict[str, Any]] = {}
    for component in product["components"]:
        output_offset = int(component["output_component_offset"])
        if output_offset % channels:
            raise ValueError("MH1 product component offset is not channel aligned")
        components_by_output[output_offset // channels] = component

    branches: list[str] = []
    for output_component in range(output_components):
        component = components_by_output.get(output_component)
        body = ["float value = 0.0f;"]
        if component is not None:
            for term_index in range(
                int(component["term_begin"]), int(component["term_end"])
            ):
                term = product["terms"][term_index]
                factors = " * ".join(
                    f"input[{int(product['angular_layout'][int(axis)]['component_offset'])}"
                    " + channel]"
                    for axis in term["angular_indices"]
                )
                body.append(
                    f"value += {coefficient_pointer}[{term_index * channels} + channel]"
                    f" * {factors};"
                )
        body.append(f"contracted[{output_component * channels} + channel] = value;")
        prefix = "if" if not branches else "else if"
        branches.append(
            f"{prefix} (output_component == {output_component}) {{\n"
            + "\n".join(f"        {line}" for line in body)
            + "\n    }"
        )

    kernel_name = f"{symbol_prefix}{stem}_tiled"
    linkage = 'extern "C" ' if symbol_prefix else ""
    definition = f"""{linkage}__global__ void {kernel_name}({args_type} args)
{{
    constexpr int tile_channels = {_CUDA_NODE_PRODUCT_TILE_CHANNELS};
    const int channel = blockIdx.x * tile_channels + threadIdx.x;
    const std::int64_t node = blockIdx.y;
    const int output_component = blockIdx.z;
    if (node >= args.num_nodes || channel >= {channels}) return;
    const float* input = {input_pointer} + node * {input_stride};
    float* contracted = {contracted_pointer} + node * {contracted_stride};
    {chr(10).join(branches)}
}}"""
    launch = (
        f"{kernel_name}<<<dim3(({channels} + {_CUDA_NODE_PRODUCT_TILE_CHANNELS - 1})"
        f" / {_CUDA_NODE_PRODUCT_TILE_CHANNELS}, args->num_nodes, {output_components}),"
        f" {_CUDA_NODE_PRODUCT_TILE_CHANNELS}, 0, cuda_stream>>>(*args);"
    )
    record = MH1CudaKernelLaunch(
        kernel_name,
        "node_channel_tiles",
        _CUDA_NODE_PRODUCT_TILE_CHANNELS,
        channels,
        output_components,
        1,
        _CUDA_NODE_PRODUCT_TILE_CHANNELS,
    )
    return definition, launch, [record]


def _cuda_tiled_product_reverse(
    product: dict[str, Any],
    *,
    args_type: str,
    input_pointer: str,
    input_stride: int,
    contracted_adjoint_pointer: str,
    contracted_adjoint_stride: int,
    coefficient_pointer: str,
    input_adjoint_pointer: str,
    input_adjoint_stride: int,
    stem: str,
    tile_nodes: int = 1,
    symbol_prefix: str = "",
) -> tuple[str, str, list[MH1CudaKernelLaunch]]:
    """Render product reverse over independent angular and channel tiles."""

    if tile_nodes not in (1, 2, 4):
        raise ValueError("MH1 tiled product reverse requires 1, 2, or 4 node tiles")

    input_pointer = f"({input_pointer})"
    contracted_adjoint_pointer = f"({contracted_adjoint_pointer})"
    coefficient_pointer = f"({coefficient_pointer})"
    input_adjoint_pointer = f"({input_adjoint_pointer})"
    channels = int(product["dimensions"]["channels"])
    angular_layout = product["angular_layout"]
    terms_by_angular: dict[int, list[tuple[int, list[int]]]] = {
        index: [] for index in range(len(angular_layout))
    }
    component_by_term: dict[int, int] = {}
    for component in product["components"]:
        output_offset = int(component["output_component_offset"])
        for term_index in range(
            int(component["term_begin"]), int(component["term_end"])
        ):
            component_by_term[term_index] = output_offset
    for term_index, term in enumerate(product["terms"]):
        angular_indices = [int(value) for value in term["angular_indices"]]
        for differentiated, angular_index in enumerate(angular_indices):
            remaining = [
                value
                for axis, value in enumerate(angular_indices)
                if axis != differentiated
            ]
            terms_by_angular[angular_index].append((term_index, remaining))

    branches: list[str] = []
    for angular_index, entries in terms_by_angular.items():
        input_offset = int(angular_layout[angular_index]["component_offset"])
        if tile_nodes == 1:
            body = ["float value = 0.0f;"]
            for term_index, remaining in entries:
                factors = "".join(
                    f" * input[{int(angular_layout[axis]['component_offset'])} + channel]"
                    for axis in remaining
                )
                output_offset = component_by_term[term_index]
                body.append(
                    f"value += {coefficient_pointer}[{term_index * channels} + channel]"
                    f" * contracted_adjoint[{output_offset} + channel]{factors};"
                )
            body.append(f"input_adjoint[{input_offset} + channel] = value;")
        else:
            body = [
                f"float value_{node_slot} = 0.0f;" for node_slot in range(tile_nodes)
            ]
            for entry_index, (term_index, remaining) in enumerate(entries):
                body.append(
                    f"const float coefficient_{entry_index} = {coefficient_pointer}"
                    f"[{term_index * channels} + channel];"
                )
                output_offset = component_by_term[term_index]
                for node_slot in range(tile_nodes):
                    factors = "".join(
                        f" * input_{node_slot}[{int(angular_layout[axis]['component_offset'])} + channel]"
                        for axis in remaining
                    )
                    body.append(
                        f"value_{node_slot} += coefficient_{entry_index}"
                        f" * contracted_adjoint_{node_slot}[{output_offset} + channel]"
                        f"{factors};"
                    )
            body.append(f"input_adjoint_0[{input_offset} + channel] = value_0;")
            for node_slot in range(1, tile_nodes):
                body.extend(
                    [
                        f"if (node_valid_{node_slot})",
                        f"    input_adjoint_{node_slot}[{input_offset} + channel] = value_{node_slot};",
                    ]
                )
        prefix = "if" if not branches else "else if"
        branches.append(
            f"{prefix} (angular_index == {angular_index}) {{\n"
            + "\n".join(f"        {line}" for line in body)
            + "\n    }"
        )

    kernel_name = f"{symbol_prefix}{stem}_tiled"
    linkage = 'extern "C" ' if symbol_prefix else ""
    if tile_nodes == 1:
        definition = f"""{linkage}__global__ void {kernel_name}({args_type} args)
{{
    constexpr int tile_channels = {_CUDA_NODE_PRODUCT_TILE_CHANNELS};
    const int channel = blockIdx.x * tile_channels + threadIdx.x;
    const std::int64_t node = blockIdx.y;
    const int angular_index = blockIdx.z;
    if (node >= args.num_nodes || channel >= {channels}) return;
    const float* input = {input_pointer} + node * {input_stride};
    const float* contracted_adjoint = {contracted_adjoint_pointer}
        + node * {contracted_adjoint_stride};
    float* input_adjoint = {input_adjoint_pointer} + node * {input_adjoint_stride};
    {chr(10).join(branches)}
}}"""
    else:
        definition = f"""{linkage}__global__ void {kernel_name}({args_type} args)
{{
    constexpr int tile_channels = {_CUDA_NODE_PRODUCT_TILE_CHANNELS};
    constexpr int tile_nodes = {tile_nodes};
    const int channel = blockIdx.x * tile_channels + threadIdx.x;
    const std::int64_t node_0 = static_cast<std::int64_t>(blockIdx.y) * tile_nodes;
{chr(10).join(f"    const bool node_valid_{node_slot} = node_0 + {node_slot} < args.num_nodes;" for node_slot in range(1, tile_nodes))}
{chr(10).join(f"    const std::int64_t node_{node_slot} = node_0 + static_cast<std::int64_t>(node_valid_{node_slot} ? {node_slot} : 0);" for node_slot in range(1, tile_nodes))}
    const int angular_index = blockIdx.z;
    if (node_0 >= args.num_nodes || channel >= {channels}) return;
{chr(10).join(f"    const float* input_{node_slot} = {input_pointer} + node_{node_slot} * {input_stride};" for node_slot in range(tile_nodes))}
{chr(10).join(f"    const float* contracted_adjoint_{node_slot} = {contracted_adjoint_pointer} + node_{node_slot} * {contracted_adjoint_stride};" for node_slot in range(tile_nodes))}
{chr(10).join(f"    float* input_adjoint_{node_slot} = {input_adjoint_pointer} + node_{node_slot} * {input_adjoint_stride};" for node_slot in range(tile_nodes))}
    {chr(10).join(branches)}
}}"""
    launch = (
        f"{kernel_name}<<<dim3(({channels} + {_CUDA_NODE_PRODUCT_TILE_CHANNELS - 1})"
        f" / {_CUDA_NODE_PRODUCT_TILE_CHANNELS}, (args->num_nodes + {tile_nodes - 1})"
        f" / {tile_nodes}, {len(angular_layout)}),"
        f" {_CUDA_NODE_PRODUCT_TILE_CHANNELS}, 0, cuda_stream>>>(*args);"
    )
    record = MH1CudaKernelLaunch(
        kernel_name,
        "node_channel_tiles",
        _CUDA_NODE_PRODUCT_TILE_CHANNELS,
        channels,
        len(angular_layout),
        tile_nodes,
        _CUDA_NODE_PRODUCT_TILE_CHANNELS,
    )
    return definition, launch, [record]


def _cuda_block_linear_adjoint_dot(
    descriptor: dict[str, Any],
    *,
    input_pointer: str,
    output_adjoint_pointer: str,
    parameter_pointer: str,
    result_pointer: str,
    stem: str,
) -> str:
    """Recompute a linear and reduce its adjoint dot product per node."""

    input_pointer = f"({input_pointer})"
    output_adjoint_pointer = f"({output_adjoint_pointer})"
    parameter_pointer = f"({parameter_pointer})"
    segments = {
        segment["name"]: segment
        for segment in descriptor["runtime_parameters"]["segments"]
    }
    weight_base = int(segments["weight"]["offset"])
    bias_base = int(segments["bias"]["offset"])
    bias_count = int(segments["bias"]["count"])
    mask_base = int(segments["output_mask"]["offset"])
    instructions_by_output: dict[int, list[dict[str, Any]]] = {}
    for instruction in descriptor["instructions"]:
        instructions_by_output.setdefault(int(instruction["output_block"]), []).append(
            instruction
        )
    lines = [f"float {stem}_sum = 0.0f;"]
    for output_block_index, output_block in enumerate(descriptor["blocks"]["output"]):
        multiplicity = int(output_block["multiplicity"])
        components = int(output_block["components"])
        output_base = int(output_block["offset"])
        work_count = multiplicity * components
        lines.extend(
            [
                f"for (int {stem}_work_{output_block_index} = threadIdx.x;",
                f"     {stem}_work_{output_block_index} < {work_count};",
                f"     {stem}_work_{output_block_index} += blockDim.x) {{",
                f"    const int component = {stem}_work_{output_block_index} / {multiplicity};",
                f"    const int target = {stem}_work_{output_block_index} % {multiplicity};",
                f"    const int output_index = {output_base} + component * {multiplicity} + target;",
                (
                    f"    float value = {parameter_pointer}[{bias_base} + output_index];"
                    if bias_count
                    else "    float value = 0.0f;"
                ),
            ]
        )
        for path_index, instruction in enumerate(
            instructions_by_output.get(output_block_index, [])
        ):
            input_block = descriptor["blocks"]["input"][int(instruction["input_block"])]
            input_multiplicity = int(input_block["multiplicity"])
            input_base = int(input_block["offset"])
            weight_offset = weight_base + int(instruction["weight_offset"])
            path_weight = _cpp_float(instruction["path_weight"])
            lines.extend(
                [
                    f"    for (int source_{path_index} = 0; source_{path_index} < {input_multiplicity};",
                    f"         ++source_{path_index})",
                    f"        value += {path_weight} * {parameter_pointer}[{weight_offset}",
                    f"            + source_{path_index} * {multiplicity} + target]",
                    f"            * {input_pointer}[{input_base} + component * {input_multiplicity}",
                    f"                + source_{path_index}];",
                ]
            )
        lines.extend(
            [
                f"    {stem}_sum += {output_adjoint_pointer}[output_index] * value",
                f"        * {parameter_pointer}[{mask_base} + output_index];",
                "}",
            ]
        )
    lines.extend(
        [
            f"__shared__ float {stem}_reduction[128];",
            f"{stem}_reduction[threadIdx.x] = {stem}_sum;",
            "__syncthreads();",
            "for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {",
            "    if (threadIdx.x < offset)",
            f"        {stem}_reduction[threadIdx.x] += {stem}_reduction[threadIdx.x + offset];",
            "    __syncthreads();",
            "}",
            f"if (threadIdx.x == 0) *({result_pointer}) = {stem}_reduction[0];",
            "__syncthreads();",
        ]
    )
    return "\n".join(lines)


def _cuda_block_silu(
    blocks: list[dict[str, Any]],
    constants: list[float],
    *,
    values_pointer: str,
    stem: str,
    reverse: bool = False,
    adjoint_pointer: str | None = None,
) -> str:
    if reverse and adjoint_pointer is None:
        adjoint_pointer = values_pointer
    values_pointer = f"({values_pointer})"
    if adjoint_pointer is not None:
        adjoint_pointer = f"({adjoint_pointer})"
    lines: list[str] = []
    offset = 0
    for block_index, (block, constant) in enumerate(
        zip(blocks, constants, strict=True)
    ):
        count = int(block["multiplicity"]) * int(block["components"])
        scale = _cpp_float(constant)
        lines.extend(
            [
                f"for (int {stem}_index_{block_index} = threadIdx.x;",
                f"     {stem}_index_{block_index} < {count};",
                f"     {stem}_index_{block_index} += blockDim.x) {{",
                f"    const int column = {offset} + {stem}_index_{block_index};",
                f"    const float value = {values_pointer}[column];",
                "    const float sigmoid = 1.0f / (1.0f + expf(-value));",
                (
                    f"    {adjoint_pointer}[column] *= {scale}"
                    " * (sigmoid + value * sigmoid * (1.0f - sigmoid));"
                    if reverse
                    else f"    {values_pointer}[column] = {scale} * value * sigmoid;"
                ),
                "}",
            ]
        )
        offset += count
    lines.append("__syncthreads();")
    return "\n".join(lines)


def _render_cuda_node_program(
    layer: dict[str, Any],
    metadata: dict[str, Any],
    embedding: dict[str, Any],
    *,
    return_program: bool = False,
    symbol_prefix: str = "",
    node_reverse_schedule: str = _MH1_NODE_REVERSE_SCHEDULE,
) -> str | MH1NodeCudaProgram:
    """Render one-block-per-node cooperative CUDA node phases."""

    index = int(layer["index"])
    input_dimension = int(metadata["input_dimension"])
    up_dimension = int(metadata["up_dimension"])
    residual_dimension = int(metadata["residual_dimension"])
    gated_dimension = int(layer["gate"]["dimensions"]["irreps_out"])
    message_dimension = int(metadata["message_dimension"])
    interaction_dimension = int(metadata["interaction_output_dimension"])
    output_dimension = int(metadata["output_dimension"])
    arena_dimension = int(metadata["node_arena_dimension"])
    _resolve_node_state_policy(metadata["node_state_policy"])
    retain_pre_gate = int(metadata["retained_pre_gate_dimension"]) > 0
    retain_interaction_output = (
        int(metadata["retained_interaction_output_dimension"]) > 0
    )
    reuse_message_adjoint = bool(metadata["reuse_message_adjoint"])
    if node_reverse_schedule not in (
        _MH1_NODE_REVERSE_SCHEDULE,
        _MH1_CUDA_RECOMPUTE_GROUPED_MESSAGE_REVERSE_SCHEDULE,
        _MH1_CUDA_RECOMPUTE_GROUPED_LINEAR2_SCHEDULE,
        _MH1_CUDA_RECOMPUTE_GROUPED_ALL_MESSAGE_SCHEDULE,
        _MH1_CUDA_RECOMPUTE_GROUPED_FORWARD_MESSAGE_SCHEDULE,
        _MH1_CUDA_RECOMPUTE_GROUPED_RESIDUAL_SCHEDULE,
        _MH1_CUDA_RECOMPUTE_TILED_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_RETAINED_NODE_REVERSE_SCHEDULE,
        _MH1_CUDA_GROUPED_MESSAGE_REVERSE_SCHEDULE,
        _MH1_CUDA_GROUPED_LINEAR2_SCHEDULE,
        _MH1_CUDA_GROUPED_ALL_MESSAGE_SCHEDULE,
        _MH1_CUDA_GROUPED_FORWARD_MESSAGE_SCHEDULE,
        _MH1_CUDA_GROUPED_RESIDUAL_SCHEDULE,
        _MH1_CUDA_GROUPED_FORWARD_RESIDUAL_SCHEDULE,
        _MH1_CUDA_TILED_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_TILED_PRODUCT_FORWARD_REVERSE_SCHEDULE,
        _MH1_CUDA_TILED_LINEAR_UP_REVERSE_SCHEDULE,
        _MH1_CUDA_TILED_LINEAR_UP_FORWARD_REVERSE_SCHEDULE,
        _MH1_CUDA_GROUPED_READOUT_REVERSE_SCHEDULE,
        _MH1_CUDA_GROUPED_SKIP_FORWARD_SCHEDULE,
        _MH1_CUDA_GROUPED_ALL_SKIP_FORWARD_SCHEDULE,
        _MH1_CUDA_WIDE_PRODUCT_LINEAR_SCHEDULE,
        _MH1_CUDA_WIDE_L1_REVERSE_MESSAGE_SCHEDULE,
        _MH1_CUDA_WIDE_L1_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_PAIRED_L0_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_WIDE_ALL_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_WIDE_L0_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_WIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_PAIRED_ALL_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_XWIDE_L0_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_ALL_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_L0_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
    ):
        raise ValueError("MH1 node reverse schedule is unsupported")
    if (
        node_reverse_schedule
        in (
            _MH1_CUDA_RETAINED_NODE_REVERSE_SCHEDULE,
            _MH1_CUDA_GROUPED_MESSAGE_REVERSE_SCHEDULE,
            _MH1_CUDA_GROUPED_LINEAR2_SCHEDULE,
            _MH1_CUDA_GROUPED_ALL_MESSAGE_SCHEDULE,
            _MH1_CUDA_GROUPED_FORWARD_MESSAGE_SCHEDULE,
            _MH1_CUDA_GROUPED_RESIDUAL_SCHEDULE,
            _MH1_CUDA_GROUPED_FORWARD_RESIDUAL_SCHEDULE,
            _MH1_CUDA_TILED_PRODUCT_REVERSE_SCHEDULE,
            _MH1_CUDA_TILED_PRODUCT_FORWARD_REVERSE_SCHEDULE,
            _MH1_CUDA_TILED_LINEAR_UP_REVERSE_SCHEDULE,
            _MH1_CUDA_TILED_LINEAR_UP_FORWARD_REVERSE_SCHEDULE,
            _MH1_CUDA_GROUPED_READOUT_REVERSE_SCHEDULE,
            _MH1_CUDA_GROUPED_SKIP_FORWARD_SCHEDULE,
            _MH1_CUDA_GROUPED_ALL_SKIP_FORWARD_SCHEDULE,
            _MH1_CUDA_WIDE_PRODUCT_LINEAR_SCHEDULE,
            _MH1_CUDA_WIDE_L1_REVERSE_MESSAGE_SCHEDULE,
            _MH1_CUDA_WIDE_L1_REVERSE_LINEAR2_SCHEDULE,
            _MH1_CUDA_PAIRED_L0_PRODUCT_REVERSE_SCHEDULE,
            _MH1_CUDA_WIDE_ALL_REVERSE_LINEAR2_SCHEDULE,
            _MH1_CUDA_WIDE_L0_FORWARD_LINEAR2_SCHEDULE,
            _MH1_CUDA_WIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
            _MH1_CUDA_PAIRED_ALL_PRODUCT_REVERSE_SCHEDULE,
            _MH1_CUDA_XWIDE_L0_REVERSE_LINEAR2_SCHEDULE,
            _MH1_CUDA_XWIDE_ALL_REVERSE_LINEAR2_SCHEDULE,
            _MH1_CUDA_XWIDE_L0_FORWARD_LINEAR2_SCHEDULE,
            _MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
        )
        and not retain_interaction_output
    ):
        raise ValueError("experimental MH1 reverse requires full node retention")
    wide_layer_one_reverse_message = node_reverse_schedule in (
        _MH1_CUDA_WIDE_L1_REVERSE_MESSAGE_SCHEDULE,
        _MH1_CUDA_WIDE_L1_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_PAIRED_L0_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_WIDE_ALL_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_WIDE_L0_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_WIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_PAIRED_ALL_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_XWIDE_L0_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_ALL_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_L0_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
    )
    wide_layer_one_reverse_linear2 = node_reverse_schedule in (
        _MH1_CUDA_WIDE_L1_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_PAIRED_L0_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_WIDE_ALL_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_WIDE_L0_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_WIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_PAIRED_ALL_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_XWIDE_L0_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_ALL_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_L0_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
    )
    wide_layer_zero_reverse_linear2 = node_reverse_schedule in (
        _MH1_CUDA_WIDE_ALL_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_WIDE_L0_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_WIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_PAIRED_ALL_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_XWIDE_L0_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_ALL_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_L0_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
    )
    xwide_layer_zero_reverse_linear2 = node_reverse_schedule in (
        _MH1_CUDA_XWIDE_L0_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_ALL_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_L0_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
    )
    xwide_layer_one_reverse_linear2 = node_reverse_schedule in (
        _MH1_CUDA_XWIDE_ALL_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_L0_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
    )
    wide_layer_zero_forward_linear2 = node_reverse_schedule in (
        _MH1_CUDA_WIDE_L0_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_WIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_PAIRED_ALL_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_XWIDE_L0_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_ALL_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_L0_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
    )
    xwide_layer_zero_forward_linear2 = node_reverse_schedule in (
        _MH1_CUDA_XWIDE_L0_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
    )
    wide_layer_one_forward_linear2 = node_reverse_schedule in (
        _MH1_CUDA_WIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_PAIRED_ALL_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_XWIDE_L0_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_ALL_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_L0_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
    )
    xwide_layer_one_forward_linear2 = (
        node_reverse_schedule == _MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE
    )
    paired_layer_zero_product_reverse = node_reverse_schedule in (
        _MH1_CUDA_PAIRED_L0_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_WIDE_ALL_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_WIDE_L0_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_WIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_PAIRED_ALL_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_XWIDE_L0_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_ALL_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_L0_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
    )
    paired_layer_one_product_reverse = node_reverse_schedule in (
        _MH1_CUDA_PAIRED_ALL_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_XWIDE_L0_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_ALL_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_L0_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
    )
    wide_product_linear = node_reverse_schedule in (
        _MH1_CUDA_WIDE_PRODUCT_LINEAR_SCHEDULE,
        _MH1_CUDA_WIDE_L1_REVERSE_MESSAGE_SCHEDULE,
        _MH1_CUDA_WIDE_L1_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_PAIRED_L0_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_WIDE_ALL_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_WIDE_L0_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_WIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_PAIRED_ALL_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_XWIDE_L0_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_ALL_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_L0_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
    )
    grouped_layer_zero_skip_forward = node_reverse_schedule in (
        _MH1_CUDA_GROUPED_ALL_SKIP_FORWARD_SCHEDULE,
        _MH1_CUDA_WIDE_PRODUCT_LINEAR_SCHEDULE,
        _MH1_CUDA_WIDE_L1_REVERSE_MESSAGE_SCHEDULE,
        _MH1_CUDA_WIDE_L1_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_PAIRED_L0_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_WIDE_ALL_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_WIDE_L0_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_WIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_PAIRED_ALL_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_XWIDE_L0_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_ALL_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_L0_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
    )
    grouped_skip_forward = node_reverse_schedule in (
        _MH1_CUDA_GROUPED_SKIP_FORWARD_SCHEDULE,
        _MH1_CUDA_GROUPED_ALL_SKIP_FORWARD_SCHEDULE,
        _MH1_CUDA_WIDE_PRODUCT_LINEAR_SCHEDULE,
        _MH1_CUDA_WIDE_L1_REVERSE_MESSAGE_SCHEDULE,
        _MH1_CUDA_WIDE_L1_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_PAIRED_L0_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_WIDE_ALL_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_WIDE_L0_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_WIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_PAIRED_ALL_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_XWIDE_L0_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_ALL_REVERSE_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_L0_FORWARD_LINEAR2_SCHEDULE,
        _MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
    )
    if grouped_skip_forward:
        node_reverse_schedule = _MH1_CUDA_GROUPED_READOUT_REVERSE_SCHEDULE
    product = layer["product"]
    linear_offsets = {
        name: int(_runtime_pack_segment(layer, "linear", name)["offset"])
        for name in (
            (["node_embedding"] if index == 0 else [])
            + [
                "linear_up",
                "linear_res",
                "skip",
                "linear_1",
                "linear_2",
                "product_linear",
            ]
        )
    }

    def linear_parameter(name: str) -> str:
        return f"args.linear_parameters + {linear_offsets[name]}"

    pre_forward_parts: list[str] = []
    if index == 0:
        pre_forward_parts.append(
            _cuda_block_embedding_lookup(
                embedding,
                element="args.element_indices[node]",
                parameter_pointer=linear_parameter("node_embedding"),
                output_pointer="arena",
                stem=f"l{index}_pre_embedding",
            )
        )
        pre_input = "arena"
    else:
        pre_input = "args.layer_input + node * " + str(input_dimension)
    pre_forward_parts.append(
        _cuda_block_linear_forward(
            layer["linears"]["linear_up"],
            input_pointer=pre_input,
            parameter_pointer=linear_parameter("linear_up"),
            output_pointer="args.up_output + node * " + str(up_dimension),
            stem=f"l{index}_pre_up",
        )
    )
    pre_forward_body = "\n".join(pre_forward_parts)

    normalization_forward = "\n".join(
        [
            "const float normalization = args.linear_parameters["
            + str(
                int(
                    _runtime_pack_segment(layer, "linear", "normalization_alpha_beta")[
                        "offset"
                    ]
                )
            )
            + "] + args.linear_parameters["
            + str(
                int(
                    _runtime_pack_segment(layer, "linear", "normalization_alpha_beta")[
                        "offset"
                    ]
                )
                + 1
            )
            + "] * args.node_density[node];",
            "const float inverse_normalization = 1.0f / normalization;",
            _cuda_block_linear_forward(
                layer["linears"]["linear_res"],
                input_pointer="up",
                parameter_pointer=linear_parameter("linear_res"),
                output_pointer="pre_gate",
                stem=f"l{index}_norm_res",
            ),
            _cuda_block_linear_forward(
                layer["linears"]["linear_1"],
                input_pointer="messages",
                parameter_pointer=linear_parameter("linear_1"),
                output_pointer="pre_gate",
                stem=f"l{index}_norm_msg",
                add_scale="inverse_normalization",
            ),
        ]
    )
    gate_forward = _cuda_block_gate_forward(
        layer["gate"],
        input_pointer="pre_gate",
        output_pointer="gated",
        stem=f"l{index}_gate",
    )
    linear_2_forward = _cuda_block_linear_forward(
        layer["linears"]["linear_2"],
        input_pointer="gated",
        parameter_pointer=linear_parameter("linear_2"),
        output_pointer="interaction",
        stem=f"l{index}_linear2",
    )
    if index == 0:
        skip_input_setup = "\n".join(
            [
                f"float* embedded = arena + {interaction_dimension};",
                f"float* skip = embedded + {input_dimension};",
                _cuda_block_embedding_lookup(
                    embedding,
                    element="args.element_indices[node]",
                    parameter_pointer=linear_parameter("node_embedding"),
                    output_pointer="embedded",
                    stem="l0_skip_embedding",
                ),
            ]
        )
        skip_linear_setup = _cuda_block_linear_forward(
            layer["linears"]["skip"],
            input_pointer="embedded",
            parameter_pointer=linear_parameter("skip"),
            output_pointer="skip",
            stem="l0_skip",
        )
    else:
        skip_input_setup = f"float* skip = arena + {interaction_dimension};"
        skip_linear_setup = _cuda_block_linear_forward(
            layer["linears"]["skip"],
            input_pointer="layer_input",
            parameter_pointer=linear_parameter("skip"),
            output_pointer="skip",
            stem="l1_skip",
        )
    skip_setup = f"{skip_input_setup}\n{skip_linear_setup}"
    contracted_pointer = f"skip + {output_dimension}"
    product_forward = "\n".join(
        [
            f"float* contracted = {contracted_pointer};",
            _cuda_block_product_forward(
                product,
                input_pointer="interaction",
                coefficient_pointer="args.product_parameters",
                contracted_pointer="contracted",
                stem=f"l{index}_product",
            ),
            _cuda_block_linear_forward(
                product["linear"],
                input_pointer="contracted",
                parameter_pointer=linear_parameter("product_linear"),
                output_pointer="layer_output",
                stem=f"l{index}_product_linear",
            ),
            f"for (int column = threadIdx.x; column < {output_dimension};",
            "     column += blockDim.x)",
            "    layer_output[column] += skip[column];",
            "__syncthreads();",
        ]
    )

    readout = layer["readout"]
    if readout["class"] == "LinearReadoutBlock":
        readout_offset = int(
            _runtime_pack_segment(layer, "readout", "linear")["offset"]
        )
        readout_forward = _cuda_block_linear_forward(
            readout["linear"],
            input_pointer="layer_output",
            parameter_pointer=f"args.readout_parameters + {readout_offset}",
            output_pointer="args.readout_contribution + node",
            stem=f"l{index}_readout",
        )
        readout_reverse = "\n".join(
            [
                _cuda_block_linear_reverse(
                    readout["linear"],
                    output_adjoint_pointer="&args.energy_scale",
                    parameter_pointer=f"args.readout_parameters + {readout_offset}",
                    input_adjoint_pointer="arena",
                    stem=f"l{index}_readout_reverse",
                ),
                f"for (int column = threadIdx.x; column < {output_dimension};",
                "     column += blockDim.x)",
                "    layer_output_adjoint[column] += arena[column];",
                "__syncthreads();",
            ]
        )
    else:
        readout_1_offset = int(
            _runtime_pack_segment(layer, "readout", "linear_1")["offset"]
        )
        readout_2_offset = int(
            _runtime_pack_segment(layer, "readout", "linear_2")["offset"]
        )
        hidden_dimension = int(readout["linear_1"]["dimensions"]["output"])
        activation_blocks = readout["linear_1"]["blocks"]["output"]
        activation_constants = readout["activation"]["constants"]
        readout_forward = "\n".join(
            [
                "float* hidden = arena;",
                _cuda_block_linear_forward(
                    readout["linear_1"],
                    input_pointer="layer_output",
                    parameter_pointer=f"args.readout_parameters + {readout_1_offset}",
                    output_pointer="hidden",
                    stem=f"l{index}_readout1",
                ),
                _cuda_block_silu(
                    activation_blocks,
                    activation_constants,
                    values_pointer="hidden",
                    stem=f"l{index}_readout_activation",
                ),
                _cuda_block_linear_forward(
                    readout["linear_2"],
                    input_pointer="hidden",
                    parameter_pointer=f"args.readout_parameters + {readout_2_offset}",
                    output_pointer="args.readout_contribution + node",
                    stem=f"l{index}_readout2",
                ),
            ]
        )
        readout_reverse = "\n".join(
            [
                "float* hidden = arena;",
                f"float* hidden_adjoint = arena + {hidden_dimension};",
                f"float* readout_output_adjoint = arena + {2 * hidden_dimension};",
                _cuda_block_linear_forward(
                    readout["linear_1"],
                    input_pointer="layer_output",
                    parameter_pointer=f"args.readout_parameters + {readout_1_offset}",
                    output_pointer="hidden",
                    stem=f"l{index}_readout1_recompute",
                ),
                _cuda_block_linear_reverse(
                    readout["linear_2"],
                    output_adjoint_pointer="&args.energy_scale",
                    parameter_pointer=f"args.readout_parameters + {readout_2_offset}",
                    input_adjoint_pointer="hidden_adjoint",
                    stem=f"l{index}_readout2_reverse",
                ),
                _cuda_block_silu(
                    activation_blocks,
                    activation_constants,
                    values_pointer="hidden",
                    adjoint_pointer="hidden_adjoint",
                    stem=f"l{index}_readout_activation_reverse",
                    reverse=True,
                ),
                _cuda_block_linear_reverse(
                    readout["linear_1"],
                    output_adjoint_pointer="hidden_adjoint",
                    parameter_pointer=f"args.readout_parameters + {readout_1_offset}",
                    input_adjoint_pointer="readout_output_adjoint",
                    stem=f"l{index}_readout1_reverse",
                ),
                f"for (int column = threadIdx.x; column < {output_dimension};",
                "     column += blockDim.x)",
                "    layer_output_adjoint[column] += readout_output_adjoint[column];",
                "__syncthreads();",
            ]
        )

    post_forward_body = f"""float* arena = args.node_arena + node * {arena_dimension};
const float* layer_input = {('nullptr' if index == 0 else f'args.layer_input + node * {input_dimension}')};
const float* up = args.up + node * {up_dimension};
const float* messages = args.messages + node * {message_dimension};
float* pre_gate = arena;
float* gated = arena + {residual_dimension};
float* interaction = arena;
float* layer_output = args.layer_output + node * {output_dimension};
{normalization_forward}
{gate_forward}
{linear_2_forward}
{skip_setup}
{product_forward}
{readout_forward}"""

    product_reverse = "\n".join(
        [
            f"float* contracted_adjoint = arena + {interaction_dimension};",
            f"float* interaction_adjoint = contracted_adjoint + {output_dimension};",
            _cuda_block_linear_reverse(
                product["linear"],
                output_adjoint_pointer="layer_output_adjoint",
                parameter_pointer=linear_parameter("product_linear"),
                input_adjoint_pointer="contracted_adjoint",
                stem=f"l{index}_product_linear_reverse",
            ),
            _cuda_block_product_reverse(
                product,
                input_pointer="interaction",
                contracted_adjoint_pointer="contracted_adjoint",
                coefficient_pointer="args.product_parameters",
                input_adjoint_pointer="interaction_adjoint",
                stem=f"l{index}_product_reverse",
            ),
        ]
    )
    skip_reverse = ""
    if index > 0:
        skip_reverse = _cuda_block_linear_reverse(
            layer["linears"]["skip"],
            output_adjoint_pointer="layer_output_adjoint",
            parameter_pointer=linear_parameter("skip"),
            input_adjoint_pointer="layer_input_adjoint",
            stem="l1_skip_reverse",
        )
    interaction_reverse = _cuda_block_linear_reverse(
        layer["linears"]["linear_2"],
        output_adjoint_pointer="interaction_adjoint",
        parameter_pointer=linear_parameter("linear_2"),
        input_adjoint_pointer="arena",
        stem=f"l{index}_linear2_reverse",
    )
    pre_gate_recompute = "\n".join(
        [
            f"float* pre_gate_adjoint = arena + {gated_dimension};",
            _cuda_block_linear_forward(
                layer["linears"]["linear_res"],
                input_pointer="up",
                parameter_pointer=linear_parameter("linear_res"),
                output_pointer="pre_gate_adjoint",
                stem=f"l{index}_norm_res_recompute",
            ),
            _cuda_block_linear_forward(
                layer["linears"]["linear_1"],
                input_pointer="messages",
                parameter_pointer=linear_parameter("linear_1"),
                output_pointer="pre_gate_adjoint",
                stem=f"l{index}_norm_msg_recompute",
                add_scale="inverse_normalization",
            ),
            _cuda_block_gate_reverse_inplace(
                layer["gate"],
                input_and_adjoint_pointer="pre_gate_adjoint",
                output_adjoint_pointer="arena",
                stem=f"l{index}_gate_reverse",
            ),
        ]
    )
    residual_reverse = ""
    if index > 0:
        residual_reverse = _cuda_block_linear_reverse(
            layer["linears"]["linear_res"],
            output_adjoint_pointer="pre_gate_adjoint",
            parameter_pointer=linear_parameter("linear_res"),
            input_adjoint_pointer="up_adjoint",
            stem="l1_residual_reverse",
        )
    normalization_offset = int(
        _runtime_pack_segment(layer, "linear", "normalization_alpha_beta")["offset"]
    )
    density_reverse = "\n".join(
        [
            _cuda_block_linear_adjoint_dot(
                layer["linears"]["linear_1"],
                input_pointer="messages",
                output_adjoint_pointer="pre_gate_adjoint",
                parameter_pointer=linear_parameter("linear_1"),
                result_pointer="density_adjoint",
                stem=f"l{index}_density",
            ),
            "if (threadIdx.x == 0)",
            f"    *density_adjoint *= -args.linear_parameters[{normalization_offset + 1}]",
            "        / (normalization * normalization);",
            f"for (int column = threadIdx.x; column < {residual_dimension};",
            "     column += blockDim.x)",
            "    pre_gate_adjoint[column] *= inverse_normalization;",
            "__syncthreads();",
            _cuda_block_linear_reverse(
                layer["linears"]["linear_1"],
                output_adjoint_pointer="pre_gate_adjoint",
                parameter_pointer=linear_parameter("linear_1"),
                input_adjoint_pointer="message_adjoint",
                stem=f"l{index}_message_reverse",
            ),
        ]
    )
    post_reverse_preamble = f"""float* arena = args.node_arena + node * {arena_dimension};
const float* layer_input = {('nullptr' if index == 0 else f'args.layer_input + node * {input_dimension}')};
const float* up = args.up + node * {up_dimension};
const float* messages = args.messages + node * {message_dimension};
const float* layer_output = args.layer_output + node * {output_dimension};
float* layer_output_adjoint = args.layer_output_adjoint + node * {output_dimension};
float* message_adjoint = args.message_adjoint + node * {message_dimension};
float* up_adjoint = args.up_adjoint + node * {up_dimension};
float* layer_input_adjoint = {('nullptr' if index == 0 else f'args.layer_input_adjoint + node * {input_dimension}')};
float* density_adjoint = args.node_density_adjoint + node;"""
    if retain_pre_gate and retain_interaction_output:
        retained_gate_reverse = _cuda_block_gate_reverse_inplace(
            layer["gate"],
            input_and_adjoint_pointer="pre_gate_adjoint",
            output_adjoint_pointer="arena",
            stem=f"l{index}_retained_gate_reverse",
            retained_input_pointer="retained_pre_gate",
        )
        post_reverse_body = f"""{post_reverse_preamble}
{readout_reverse}
const float* interaction = args.retained_interaction_output
    + node * {interaction_dimension};
const float* retained_pre_gate = args.retained_pre_gate
    + node * {residual_dimension};
const float normalization = args.linear_parameters[{normalization_offset}]
    + args.linear_parameters[{normalization_offset + 1}] * args.node_density[node];
const float inverse_normalization = 1.0f / normalization;
{product_reverse}
{skip_reverse}
{interaction_reverse}
float* pre_gate_adjoint = arena + {gated_dimension};
{retained_gate_reverse}
{residual_reverse}
{density_reverse}"""
    elif retain_interaction_output:
        post_reverse_body = f"""{post_reverse_preamble}
{readout_reverse}
const float* interaction = args.retained_interaction_output
    + node * {interaction_dimension};
const float normalization = args.linear_parameters[{normalization_offset}]
    + args.linear_parameters[{normalization_offset + 1}] * args.node_density[node];
const float inverse_normalization = 1.0f / normalization;
{product_reverse}
{skip_reverse}
{interaction_reverse}
{pre_gate_recompute}
{residual_reverse}
{density_reverse}"""
    else:
        post_reverse_body = f"""{post_reverse_preamble}
{readout_reverse}
float* pre_gate = arena;
float* gated = arena + {residual_dimension};
float* interaction = arena;
{normalization_forward}
{gate_forward}
{linear_2_forward}
{product_reverse}
{skip_reverse}
{interaction_reverse}
{pre_gate_recompute}
{residual_reverse}
{density_reverse}"""

    tiled_definitions: list[str] = []
    tiled_launch_records: dict[str, tuple[MH1CudaKernelLaunch, ...]] = {}

    def tiled_forward(
        descriptor: dict[str, Any],
        *,
        args_type: str,
        input_pointer: str,
        input_stride: int,
        parameter_pointer: str,
        output_pointer: str,
        output_stride: int,
        stem: str,
        add_scale: str | None = None,
        tile_nodes: int = _CUDA_NODE_TILE_NODES,
    ) -> str:
        definitions, launches, records = _cuda_tiled_linear_forward(
            descriptor,
            args_type=args_type,
            input_pointer=input_pointer,
            input_stride=input_stride,
            parameter_pointer=parameter_pointer,
            output_pointer=output_pointer,
            output_stride=output_stride,
            stem=stem,
            add_scale=add_scale,
            tile_nodes=tile_nodes,
            symbol_prefix=symbol_prefix,
        )
        tiled_definitions.append(definitions)
        tiled_launch_records[launches] = tuple(records)
        return launches

    def tiled_reverse(
        descriptor: dict[str, Any],
        *,
        args_type: str,
        output_adjoint_pointer: str,
        output_stride: int,
        parameter_pointer: str,
        input_adjoint_pointer: str,
        input_stride: int,
        stem: str,
        add: bool = False,
        tile_nodes: int = _CUDA_NODE_TILE_NODES,
    ) -> str:
        definitions, launches, records = _cuda_tiled_linear_reverse(
            descriptor,
            args_type=args_type,
            output_adjoint_pointer=output_adjoint_pointer,
            output_stride=output_stride,
            parameter_pointer=parameter_pointer,
            input_adjoint_pointer=input_adjoint_pointer,
            input_stride=input_stride,
            stem=stem,
            add=add,
            tile_nodes=tile_nodes,
            symbol_prefix=symbol_prefix,
        )
        tiled_definitions.append(definitions)
        tiled_launch_records[launches] = tuple(records)
        return launches

    forward_args = "SymmetrixJitMH1CudaNodeForwardArgsV4"
    reverse_args = "SymmetrixJitMH1CudaNodeReverseArgsV4"
    pre_forward_launch = ""
    pre_forward_launch_records: list[MH1CudaKernelLaunch] = []
    if index > 0 and node_reverse_schedule in (
        _MH1_CUDA_TILED_LINEAR_UP_FORWARD_REVERSE_SCHEDULE,
        _MH1_CUDA_GROUPED_READOUT_REVERSE_SCHEDULE,
    ):
        definitions, pre_forward_launch, pre_forward_launch_records = (
            _cuda_grouped_tiled_linear_forward(
                layer["linears"]["linear_up"],
                args_type=forward_args,
                input_pointer="args.layer_input",
                input_stride=input_dimension,
                parameter_pointer=linear_parameter("linear_up"),
                output_pointer="args.up_output",
                output_stride=up_dimension,
                stem=f"node_tiled_l{index}_pre_forward_linear_up",
                tile_nodes=_CUDA_NODE_LINEAR2_TILE_NODES,
                symbol_prefix=symbol_prefix,
            )
        )
        tiled_definitions.append(definitions)
    specialized_readout_reverse_launch = ""
    specialized_readout_reverse_records: list[MH1CudaKernelLaunch] = []
    readout_reverse_middle_body = ""
    if (
        index == 1
        and node_reverse_schedule == _MH1_CUDA_GROUPED_READOUT_REVERSE_SCHEDULE
    ):
        if readout["class"] == "LinearReadoutBlock":
            raise ValueError("grouped MH1 readout reverse requires a nonlinear readout")
        definitions, readout_recompute_launch, readout_recompute_records = (
            _cuda_grouped_tiled_linear_forward(
                readout["linear_1"],
                args_type=reverse_args,
                input_pointer="args.layer_output",
                input_stride=output_dimension,
                parameter_pointer=f"args.readout_parameters + {readout_1_offset}",
                output_pointer="args.node_arena",
                output_stride=arena_dimension,
                stem=f"node_tiled_l{index}_reverse_readout1_recompute",
                tile_nodes=_CUDA_NODE_LINEAR2_TILE_NODES,
                symbol_prefix=symbol_prefix,
            )
        )
        tiled_definitions.append(definitions)
        readout_reverse_middle_body = f"""float* arena = args.node_arena
    + node * {arena_dimension};
float* hidden = arena;
float* hidden_adjoint = arena + {hidden_dimension};
{_cuda_block_linear_reverse(
    readout["linear_2"],
    output_adjoint_pointer="&args.energy_scale",
    parameter_pointer=f"args.readout_parameters + {readout_2_offset}",
    input_adjoint_pointer="hidden_adjoint",
    stem=f"l{index}_readout2_reverse_middle",
)}
{_cuda_block_silu(
    activation_blocks,
    activation_constants,
    values_pointer="hidden",
    adjoint_pointer="hidden_adjoint",
    stem=f"l{index}_readout_activation_reverse_middle",
    reverse=True,
)}"""
        definitions, readout_reverse_launch, readout_reverse_records = (
            _cuda_grouped_tiled_linear_reverse(
                readout["linear_1"],
                args_type=reverse_args,
                output_adjoint_pointer=f"args.node_arena + {hidden_dimension}",
                output_stride=arena_dimension,
                parameter_pointer=f"args.readout_parameters + {readout_1_offset}",
                input_adjoint_pointer="args.layer_output_adjoint",
                input_stride=output_dimension,
                stem=f"node_tiled_l{index}_reverse_readout1",
                add=True,
                tile_nodes=_CUDA_NODE_LINEAR2_TILE_NODES,
                symbol_prefix=symbol_prefix,
            )
        )
        tiled_definitions.append(definitions)
        definitions, skip_reverse_launch, skip_reverse_records = (
            _cuda_grouped_tiled_linear_reverse(
                layer["linears"]["skip"],
                args_type=reverse_args,
                output_adjoint_pointer="args.layer_output_adjoint",
                output_stride=output_dimension,
                parameter_pointer=linear_parameter("skip"),
                input_adjoint_pointer="args.layer_input_adjoint",
                input_stride=input_dimension,
                stem=f"node_tiled_l{index}_reverse_skip",
                tile_nodes=_CUDA_NODE_LINEAR2_TILE_NODES,
                symbol_prefix=symbol_prefix,
            )
        )
        tiled_definitions.append(definitions)
        readout_reverse_middle_name = (
            f"{symbol_prefix}node_tiled_l{index}_reverse_readout_middle"
        )
        specialized_readout_reverse_launch = "\n    ".join(
            (
                readout_recompute_launch,
                f"{readout_reverse_middle_name}<<<blocks, 128, 0, cuda_stream>>>(*args);",
                readout_reverse_launch,
                skip_reverse_launch,
            )
        )
        specialized_readout_reverse_records = (
            list(readout_recompute_records)
            + [
                MH1CudaKernelLaunch(
                    readout_reverse_middle_name, "persistent_nodes", 128
                )
            ]
            + list(readout_reverse_records)
            + list(skip_reverse_records)
        )
    inverse_normalization = (
        f"1.0f / (args.linear_parameters[{normalization_offset}]"
        f" + args.linear_parameters[{normalization_offset + 1}]"
        " * args.node_density[node])"
    )
    if node_reverse_schedule in (
        _MH1_CUDA_RECOMPUTE_GROUPED_RESIDUAL_SCHEDULE,
        _MH1_CUDA_RECOMPUTE_TILED_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_GROUPED_RESIDUAL_SCHEDULE,
        _MH1_CUDA_GROUPED_FORWARD_RESIDUAL_SCHEDULE,
        _MH1_CUDA_TILED_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_TILED_PRODUCT_FORWARD_REVERSE_SCHEDULE,
        _MH1_CUDA_TILED_LINEAR_UP_REVERSE_SCHEDULE,
        _MH1_CUDA_TILED_LINEAR_UP_FORWARD_REVERSE_SCHEDULE,
        _MH1_CUDA_GROUPED_READOUT_REVERSE_SCHEDULE,
    ):
        definitions, forward_residual, forward_residual_records = (
            _cuda_grouped_tiled_linear_forward(
                layer["linears"]["linear_res"],
                args_type=forward_args,
                input_pointer="args.up",
                input_stride=up_dimension,
                parameter_pointer=linear_parameter("linear_res"),
                output_pointer="args.node_arena",
                output_stride=arena_dimension,
                stem=f"node_tiled_l{index}_forward_residual",
                tile_nodes=_CUDA_NODE_LINEAR2_TILE_NODES,
                symbol_prefix=symbol_prefix,
            )
        )
        tiled_definitions.append(definitions)
        tiled_launch_records[forward_residual] = tuple(forward_residual_records)
    else:
        forward_residual = tiled_forward(
            layer["linears"]["linear_res"],
            args_type=forward_args,
            input_pointer="args.up",
            input_stride=up_dimension,
            parameter_pointer=linear_parameter("linear_res"),
            output_pointer="args.node_arena",
            output_stride=arena_dimension,
            stem=f"node_tiled_l{index}_forward_residual",
        )
    if node_reverse_schedule in (
        _MH1_CUDA_RECOMPUTE_GROUPED_FORWARD_MESSAGE_SCHEDULE,
        _MH1_CUDA_RECOMPUTE_GROUPED_RESIDUAL_SCHEDULE,
        _MH1_CUDA_RECOMPUTE_TILED_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_GROUPED_FORWARD_MESSAGE_SCHEDULE,
        _MH1_CUDA_GROUPED_RESIDUAL_SCHEDULE,
        _MH1_CUDA_GROUPED_FORWARD_RESIDUAL_SCHEDULE,
        _MH1_CUDA_TILED_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_TILED_PRODUCT_FORWARD_REVERSE_SCHEDULE,
        _MH1_CUDA_TILED_LINEAR_UP_REVERSE_SCHEDULE,
        _MH1_CUDA_TILED_LINEAR_UP_FORWARD_REVERSE_SCHEDULE,
        _MH1_CUDA_GROUPED_READOUT_REVERSE_SCHEDULE,
    ):
        definitions, forward_message, forward_message_records = (
            _cuda_grouped_tiled_linear_forward(
                layer["linears"]["linear_1"],
                args_type=forward_args,
                input_pointer="args.messages",
                input_stride=message_dimension,
                parameter_pointer=linear_parameter("linear_1"),
                output_pointer="args.node_arena",
                output_stride=arena_dimension,
                stem=f"node_tiled_l{index}_forward_message",
                add_scale=inverse_normalization,
                tile_nodes=_CUDA_NODE_LINEAR2_TILE_NODES,
                symbol_prefix=symbol_prefix,
            )
        )
        tiled_definitions.append(definitions)
        tiled_launch_records[forward_message] = tuple(forward_message_records)
    else:
        forward_message = tiled_forward(
            layer["linears"]["linear_1"],
            args_type=forward_args,
            input_pointer="args.messages",
            input_stride=message_dimension,
            parameter_pointer=linear_parameter("linear_1"),
            output_pointer="args.node_arena",
            output_stride=arena_dimension,
            stem=f"node_tiled_l{index}_forward_message",
            add_scale=inverse_normalization,
            tile_nodes=_CUDA_NODE_LINEAR2_TILE_NODES,
        )
    forward_pregate = "\n    ".join((forward_residual, forward_message))
    if node_reverse_schedule in (
        _MH1_CUDA_GROUPED_LINEAR2_SCHEDULE,
        _MH1_CUDA_RECOMPUTE_GROUPED_LINEAR2_SCHEDULE,
        _MH1_CUDA_RECOMPUTE_GROUPED_ALL_MESSAGE_SCHEDULE,
        _MH1_CUDA_RECOMPUTE_GROUPED_FORWARD_MESSAGE_SCHEDULE,
        _MH1_CUDA_RECOMPUTE_GROUPED_RESIDUAL_SCHEDULE,
        _MH1_CUDA_RECOMPUTE_TILED_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_GROUPED_ALL_MESSAGE_SCHEDULE,
        _MH1_CUDA_GROUPED_FORWARD_MESSAGE_SCHEDULE,
        _MH1_CUDA_GROUPED_RESIDUAL_SCHEDULE,
        _MH1_CUDA_GROUPED_FORWARD_RESIDUAL_SCHEDULE,
        _MH1_CUDA_TILED_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_TILED_PRODUCT_FORWARD_REVERSE_SCHEDULE,
        _MH1_CUDA_TILED_LINEAR_UP_REVERSE_SCHEDULE,
        _MH1_CUDA_TILED_LINEAR_UP_FORWARD_REVERSE_SCHEDULE,
        _MH1_CUDA_GROUPED_READOUT_REVERSE_SCHEDULE,
    ):
        definitions, forward_linear_2, forward_linear_2_records = (
            _cuda_grouped_tiled_linear_forward(
                layer["linears"]["linear_2"],
                args_type=forward_args,
                input_pointer=f"args.node_arena + {residual_dimension}",
                input_stride=arena_dimension,
                parameter_pointer=linear_parameter("linear_2"),
                output_pointer="args.node_arena",
                output_stride=arena_dimension,
                stem=f"node_tiled_l{index}_forward_linear2",
                tile_nodes=(
                    _CUDA_NODE_XWIDE_GROUPED_TILE_NODES
                    if (xwide_layer_zero_forward_linear2 and index == 0)
                    or (xwide_layer_one_forward_linear2 and index == 1)
                    else (
                        _CUDA_NODE_WIDE_GROUPED_TILE_NODES
                        if (wide_layer_zero_forward_linear2 and index == 0)
                        or (wide_layer_one_forward_linear2 and index == 1)
                        else _CUDA_NODE_LINEAR2_TILE_NODES
                    )
                ),
                symbol_prefix=symbol_prefix,
            )
        )
        tiled_definitions.append(definitions)
        tiled_launch_records[forward_linear_2] = tuple(forward_linear_2_records)
    else:
        forward_linear_2 = tiled_forward(
            layer["linears"]["linear_2"],
            args_type=forward_args,
            input_pointer=f"args.node_arena + {residual_dimension}",
            input_stride=arena_dimension,
            parameter_pointer=linear_parameter("linear_2"),
            output_pointer="args.node_arena",
            output_stride=arena_dimension,
            stem=f"node_tiled_l{index}_forward_linear2",
            tile_nodes=_CUDA_NODE_LINEAR2_TILE_NODES,
        )
    contracted_offset = (
        interaction_dimension + input_dimension + output_dimension
        if index == 0
        else interaction_dimension + output_dimension
    )
    forward_product_linear = tiled_forward(
        product["linear"],
        args_type=forward_args,
        input_pointer=f"args.node_arena + {contracted_offset}",
        input_stride=arena_dimension,
        parameter_pointer=linear_parameter("product_linear"),
        output_pointer="args.layer_output",
        output_stride=output_dimension,
        stem=f"node_tiled_l{index}_forward_product_linear",
        tile_nodes=(
            _CUDA_NODE_LINEAR2_TILE_NODES
            if wide_product_linear
            else _CUDA_NODE_TILE_NODES
        ),
    )

    retain_pre_gate_stage = (
        f"""float* retained_pre_gate = args.retained_pre_gate + node * {residual_dimension};
for (int column = threadIdx.x; column < {residual_dimension}; column += blockDim.x)
    retained_pre_gate[column] = arena[column];
__syncthreads();"""
        if retain_pre_gate
        else ""
    )
    gate_forward_stage = f"""float* arena = args.node_arena + node * {arena_dimension};
float* pre_gate = arena;
float* gated = arena + {residual_dimension};
{retain_pre_gate_stage}
{gate_forward}"""
    product_contraction_forward = _cuda_block_product_forward(
        product,
        input_pointer="interaction",
        coefficient_pointer="args.product_parameters",
        contracted_pointer="contracted",
        stem=f"l{index}_tiled_product",
    )
    retain_interaction = (
        f"""float* retained_interaction = args.retained_interaction_output
    + node * {interaction_dimension};
for (int column = threadIdx.x; column < {interaction_dimension}; column += blockDim.x)
    retained_interaction[column] = interaction[column];
__syncthreads();"""
        if retain_interaction_output
        else ""
    )
    skip_product_setup_stage = f"""float* arena = args.node_arena + node * {arena_dimension};
const float* layer_input = {('nullptr' if index == 0 else f'args.layer_input + node * {input_dimension}')};
float* interaction = arena;
{retain_interaction}
{skip_setup}"""
    retain_interaction_stage = f"""float* arena = args.node_arena + node * {arena_dimension};
float* interaction = arena;
{retain_interaction}"""
    retain_interaction_and_skip_input_stage = f"""float* arena = args.node_arena + node * {arena_dimension};
float* interaction = arena;
{retain_interaction}
{skip_input_setup}"""
    skip_product_stage = f"""{skip_product_setup_stage}
float* contracted = {contracted_pointer};
{product_contraction_forward}"""
    tiled_product_forward = ""
    if node_reverse_schedule in (
        _MH1_CUDA_TILED_PRODUCT_FORWARD_REVERSE_SCHEDULE,
        _MH1_CUDA_TILED_LINEAR_UP_REVERSE_SCHEDULE,
        _MH1_CUDA_TILED_LINEAR_UP_FORWARD_REVERSE_SCHEDULE,
        _MH1_CUDA_GROUPED_READOUT_REVERSE_SCHEDULE,
    ):
        definitions, tiled_product_forward, tiled_product_forward_records = (
            _cuda_tiled_product_forward(
                product,
                args_type=forward_args,
                input_pointer="args.node_arena",
                input_stride=arena_dimension,
                coefficient_pointer="args.product_parameters",
                contracted_pointer=f"args.node_arena + {contracted_offset}",
                contracted_stride=arena_dimension,
                stem=f"node_tiled_l{index}_forward_product",
                symbol_prefix=symbol_prefix,
            )
        )
        tiled_definitions.append(definitions)
        tiled_launch_records[tiled_product_forward] = tuple(
            tiled_product_forward_records
        )
    skip_offset = interaction_dimension + (input_dimension if index == 0 else 0)
    grouped_skip_forward_launch = ""
    grouped_skip_forward_records: list[MH1CudaKernelLaunch] = []
    if grouped_skip_forward and (index == 1 or grouped_layer_zero_skip_forward):
        grouped_skip_input_pointer = (
            f"args.node_arena + {interaction_dimension}"
            if index == 0
            else "args.layer_input"
        )
        grouped_skip_input_stride = arena_dimension if index == 0 else input_dimension
        definitions, grouped_skip_forward_launch, grouped_skip_forward_records = (
            _cuda_grouped_tiled_linear_forward(
                layer["linears"]["skip"],
                args_type=forward_args,
                input_pointer=grouped_skip_input_pointer,
                input_stride=grouped_skip_input_stride,
                parameter_pointer=linear_parameter("skip"),
                output_pointer=f"args.node_arena + {skip_offset}",
                output_stride=arena_dimension,
                stem=f"node_tiled_l{index}_forward_skip",
                tile_nodes=_CUDA_NODE_LINEAR2_TILE_NODES,
                symbol_prefix=symbol_prefix,
            )
        )
        tiled_definitions.append(definitions)
    finalize_forward_stage = f"""float* arena = args.node_arena + node * {arena_dimension};
float* skip = arena + {skip_offset};
float* layer_output = args.layer_output + node * {output_dimension};
for (int column = threadIdx.x; column < {output_dimension}; column += blockDim.x)
    layer_output[column] += skip[column];
__syncthreads();
{readout_forward}"""

    readout_skip_reverse_stage = f"""float* arena = args.node_arena + node * {arena_dimension};
const float* layer_input = {('nullptr' if index == 0 else f'args.layer_input + node * {input_dimension}')};
const float* layer_output = args.layer_output + node * {output_dimension};
float* layer_output_adjoint = args.layer_output_adjoint + node * {output_dimension};
float* layer_input_adjoint = {('nullptr' if index == 0 else f'args.layer_input_adjoint + node * {input_dimension}')};
{readout_reverse}
{skip_reverse}"""
    reverse_residual_forward = tiled_forward(
        layer["linears"]["linear_res"],
        args_type=reverse_args,
        input_pointer="args.up",
        input_stride=up_dimension,
        parameter_pointer=linear_parameter("linear_res"),
        output_pointer="args.node_arena",
        output_stride=arena_dimension,
        stem=f"node_tiled_l{index}_reverse_residual_forward",
    )
    reverse_message_forward = tiled_forward(
        layer["linears"]["linear_1"],
        args_type=reverse_args,
        input_pointer="args.messages",
        input_stride=message_dimension,
        parameter_pointer=linear_parameter("linear_1"),
        output_pointer="args.node_arena",
        output_stride=arena_dimension,
        stem=f"node_tiled_l{index}_reverse_message_forward",
        add_scale=inverse_normalization,
        tile_nodes=_CUDA_NODE_LINEAR2_TILE_NODES,
    )
    reverse_pregate = "\n    ".join((reverse_residual_forward, reverse_message_forward))
    reverse_linear_2_forward = tiled_forward(
        layer["linears"]["linear_2"],
        args_type=reverse_args,
        input_pointer=f"args.node_arena + {residual_dimension}",
        input_stride=arena_dimension,
        parameter_pointer=linear_parameter("linear_2"),
        output_pointer="args.node_arena",
        output_stride=arena_dimension,
        stem=f"node_tiled_l{index}_reverse_linear2_forward",
        tile_nodes=_CUDA_NODE_LINEAR2_TILE_NODES,
    )
    reverse_product_linear = tiled_reverse(
        product["linear"],
        args_type=reverse_args,
        output_adjoint_pointer="args.layer_output_adjoint",
        output_stride=output_dimension,
        parameter_pointer=linear_parameter("product_linear"),
        input_adjoint_pointer=f"args.node_arena + {interaction_dimension}",
        input_stride=arena_dimension,
        stem=f"node_tiled_l{index}_reverse_product_linear",
        tile_nodes=(
            _CUDA_NODE_LINEAR2_TILE_NODES
            if wide_product_linear
            else _CUDA_NODE_TILE_NODES
        ),
    )
    reverse_interaction_pointer = (
        f"args.retained_interaction_output + node * {interaction_dimension}"
        if retain_interaction_output
        else "arena"
    )
    product_reverse_stage = f"""float* arena = args.node_arena + node * {
        arena_dimension
    };
const float* interaction = {reverse_interaction_pointer};
float* contracted_adjoint = arena + {interaction_dimension};
float* interaction_adjoint = contracted_adjoint + {output_dimension};
{_cuda_block_product_reverse(
    product,
    input_pointer='interaction',
    contracted_adjoint_pointer='contracted_adjoint',
    coefficient_pointer='args.product_parameters',
    input_adjoint_pointer='interaction_adjoint',
    stem=f'l{index}_tiled_product_reverse',
)}"""
    tiled_product_reverse = ""
    if node_reverse_schedule in (
        _MH1_CUDA_TILED_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_RECOMPUTE_TILED_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_TILED_PRODUCT_FORWARD_REVERSE_SCHEDULE,
        _MH1_CUDA_TILED_LINEAR_UP_REVERSE_SCHEDULE,
        _MH1_CUDA_TILED_LINEAR_UP_FORWARD_REVERSE_SCHEDULE,
        _MH1_CUDA_GROUPED_READOUT_REVERSE_SCHEDULE,
    ):
        definitions, tiled_product_reverse, tiled_product_reverse_records = (
            _cuda_tiled_product_reverse(
                product,
                args_type=reverse_args,
                input_pointer=(
                    "args.retained_interaction_output"
                    if retain_interaction_output
                    else "args.node_arena"
                ),
                input_stride=(
                    interaction_dimension
                    if retain_interaction_output
                    else arena_dimension
                ),
                contracted_adjoint_pointer=(
                    f"args.node_arena + {interaction_dimension}"
                ),
                contracted_adjoint_stride=arena_dimension,
                coefficient_pointer="args.product_parameters",
                input_adjoint_pointer=(
                    f"args.node_arena + {interaction_dimension + output_dimension}"
                ),
                input_adjoint_stride=arena_dimension,
                stem=f"node_tiled_l{index}_reverse_product",
                tile_nodes=(
                    4
                    if (paired_layer_zero_product_reverse and index == 0)
                    or (paired_layer_one_product_reverse and index == 1)
                    else 1
                ),
                symbol_prefix=symbol_prefix,
            )
        )
        tiled_definitions.append(definitions)
        tiled_launch_records[tiled_product_reverse] = tuple(
            tiled_product_reverse_records
        )
    if node_reverse_schedule in (
        _MH1_CUDA_GROUPED_LINEAR2_SCHEDULE,
        _MH1_CUDA_RECOMPUTE_GROUPED_LINEAR2_SCHEDULE,
        _MH1_CUDA_RECOMPUTE_GROUPED_ALL_MESSAGE_SCHEDULE,
        _MH1_CUDA_RECOMPUTE_GROUPED_FORWARD_MESSAGE_SCHEDULE,
        _MH1_CUDA_RECOMPUTE_GROUPED_RESIDUAL_SCHEDULE,
        _MH1_CUDA_RECOMPUTE_TILED_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_GROUPED_ALL_MESSAGE_SCHEDULE,
        _MH1_CUDA_GROUPED_FORWARD_MESSAGE_SCHEDULE,
        _MH1_CUDA_GROUPED_RESIDUAL_SCHEDULE,
        _MH1_CUDA_GROUPED_FORWARD_RESIDUAL_SCHEDULE,
        _MH1_CUDA_TILED_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_TILED_PRODUCT_FORWARD_REVERSE_SCHEDULE,
        _MH1_CUDA_TILED_LINEAR_UP_REVERSE_SCHEDULE,
        _MH1_CUDA_TILED_LINEAR_UP_FORWARD_REVERSE_SCHEDULE,
        _MH1_CUDA_GROUPED_READOUT_REVERSE_SCHEDULE,
    ):
        definitions, reverse_linear_2, reverse_linear_2_records = (
            _cuda_grouped_tiled_linear_reverse(
                layer["linears"]["linear_2"],
                args_type=reverse_args,
                output_adjoint_pointer=(
                    f"args.node_arena + {interaction_dimension + output_dimension}"
                ),
                output_stride=arena_dimension,
                parameter_pointer=linear_parameter("linear_2"),
                input_adjoint_pointer="args.node_arena",
                input_stride=arena_dimension,
                stem=f"node_tiled_l{index}_reverse_linear2",
                tile_nodes=(
                    _CUDA_NODE_XWIDE_GROUPED_TILE_NODES
                    if (xwide_layer_zero_reverse_linear2 and index == 0)
                    or (xwide_layer_one_reverse_linear2 and index == 1)
                    else (
                        _CUDA_NODE_WIDE_GROUPED_TILE_NODES
                        if (wide_layer_one_reverse_linear2 and index == 1)
                        or (wide_layer_zero_reverse_linear2 and index == 0)
                        else _CUDA_NODE_LINEAR2_TILE_NODES
                    )
                ),
                symbol_prefix=symbol_prefix,
            )
        )
        tiled_definitions.append(definitions)
        tiled_launch_records[reverse_linear_2] = tuple(reverse_linear_2_records)
    else:
        reverse_linear_2 = tiled_reverse(
            layer["linears"]["linear_2"],
            args_type=reverse_args,
            output_adjoint_pointer=(
                f"args.node_arena + {interaction_dimension + output_dimension}"
            ),
            output_stride=arena_dimension,
            parameter_pointer=linear_parameter("linear_2"),
            input_adjoint_pointer="args.node_arena",
            input_stride=arena_dimension,
            stem=f"node_tiled_l{index}_reverse_linear2",
            tile_nodes=_CUDA_NODE_LINEAR2_TILE_NODES,
        )
    pregate_adjoint_offset = gated_dimension
    reverse_residual_recompute = tiled_forward(
        layer["linears"]["linear_res"],
        args_type=reverse_args,
        input_pointer="args.up",
        input_stride=up_dimension,
        parameter_pointer=linear_parameter("linear_res"),
        output_pointer=f"args.node_arena + {pregate_adjoint_offset}",
        output_stride=arena_dimension,
        stem=f"node_tiled_l{index}_reverse_residual_recompute",
    )
    reverse_message_recompute = tiled_forward(
        layer["linears"]["linear_1"],
        args_type=reverse_args,
        input_pointer="args.messages",
        input_stride=message_dimension,
        parameter_pointer=linear_parameter("linear_1"),
        output_pointer=f"args.node_arena + {pregate_adjoint_offset}",
        output_stride=arena_dimension,
        stem=f"node_tiled_l{index}_reverse_message_recompute",
        add_scale=inverse_normalization,
        tile_nodes=_CUDA_NODE_LINEAR2_TILE_NODES,
    )
    reverse_pregate_recompute = "\n    ".join(
        (reverse_residual_recompute, reverse_message_recompute)
    )
    if retain_interaction_output:
        reverse_pregate = ""
        reverse_linear_2_forward = ""
    if retain_pre_gate:
        reverse_pregate_recompute = ""
    gate_reverse_stage = f"""float* arena = args.node_arena + node * {arena_dimension};
float* pre_gate_adjoint = arena + {pregate_adjoint_offset};
{
        _cuda_block_gate_reverse_inplace(
            layer["gate"],
            input_and_adjoint_pointer="pre_gate_adjoint",
            output_adjoint_pointer="arena",
            stem=f"l{index}_tiled_gate_reverse",
            retained_input_pointer=(
                f"args.retained_pre_gate + node * {residual_dimension}"
                if retain_pre_gate
                else None
            ),
        )
    }"""
    reverse_residual = ""
    if index > 0:
        if node_reverse_schedule in (
            _MH1_CUDA_GROUPED_RESIDUAL_SCHEDULE,
            _MH1_CUDA_RECOMPUTE_GROUPED_RESIDUAL_SCHEDULE,
            _MH1_CUDA_RECOMPUTE_TILED_PRODUCT_REVERSE_SCHEDULE,
        ):
            definitions, reverse_residual, reverse_residual_records = (
                _cuda_grouped_tiled_linear_reverse(
                    layer["linears"]["linear_res"],
                    args_type=reverse_args,
                    output_adjoint_pointer=(
                        f"args.node_arena + {pregate_adjoint_offset}"
                    ),
                    output_stride=arena_dimension,
                    parameter_pointer=linear_parameter("linear_res"),
                    input_adjoint_pointer="args.up_adjoint",
                    input_stride=up_dimension,
                    stem=f"node_tiled_l{index}_reverse_residual",
                    tile_nodes=_CUDA_NODE_LINEAR2_TILE_NODES,
                    symbol_prefix=symbol_prefix,
                )
            )
            tiled_definitions.append(definitions)
            tiled_launch_records[reverse_residual] = tuple(reverse_residual_records)
        else:
            reverse_residual = tiled_reverse(
                layer["linears"]["linear_res"],
                args_type=reverse_args,
                output_adjoint_pointer=f"args.node_arena + {pregate_adjoint_offset}",
                output_stride=arena_dimension,
                parameter_pointer=linear_parameter("linear_res"),
                input_adjoint_pointer="args.up_adjoint",
                input_stride=up_dimension,
                stem=f"node_tiled_l{index}_reverse_residual",
            )
    scale_pregate_stage = f"""float* pre_gate_adjoint = args.node_arena
    + node * {arena_dimension} + {pregate_adjoint_offset};
const float normalization = args.linear_parameters[{normalization_offset}]
    + args.linear_parameters[{normalization_offset + 1}] * args.node_density[node];
for (int column = threadIdx.x; column < {residual_dimension}; column += blockDim.x)
    pre_gate_adjoint[column] /= normalization;
__syncthreads();"""
    group_message_reverse = node_reverse_schedule in (
        _MH1_CUDA_RECOMPUTE_GROUPED_ALL_MESSAGE_SCHEDULE,
        _MH1_CUDA_RECOMPUTE_GROUPED_FORWARD_MESSAGE_SCHEDULE,
        _MH1_CUDA_RECOMPUTE_GROUPED_RESIDUAL_SCHEDULE,
        _MH1_CUDA_RECOMPUTE_TILED_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_GROUPED_ALL_MESSAGE_SCHEDULE,
        _MH1_CUDA_GROUPED_FORWARD_MESSAGE_SCHEDULE,
        _MH1_CUDA_GROUPED_RESIDUAL_SCHEDULE,
        _MH1_CUDA_GROUPED_FORWARD_RESIDUAL_SCHEDULE,
        _MH1_CUDA_TILED_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_TILED_PRODUCT_FORWARD_REVERSE_SCHEDULE,
        _MH1_CUDA_TILED_LINEAR_UP_REVERSE_SCHEDULE,
        _MH1_CUDA_TILED_LINEAR_UP_FORWARD_REVERSE_SCHEDULE,
        _MH1_CUDA_GROUPED_READOUT_REVERSE_SCHEDULE,
    ) or (
        index == 1
        and node_reverse_schedule
        in (
            _MH1_CUDA_GROUPED_MESSAGE_REVERSE_SCHEDULE,
            _MH1_CUDA_RECOMPUTE_GROUPED_MESSAGE_REVERSE_SCHEDULE,
            _MH1_CUDA_GROUPED_LINEAR2_SCHEDULE,
            _MH1_CUDA_RECOMPUTE_GROUPED_LINEAR2_SCHEDULE,
        )
    )
    if group_message_reverse:
        definitions, reverse_message, reverse_message_records = (
            _cuda_grouped_tiled_linear_reverse(
                layer["linears"]["linear_1"],
                args_type=reverse_args,
                output_adjoint_pointer=(f"args.node_arena + {pregate_adjoint_offset}"),
                output_stride=arena_dimension,
                parameter_pointer=linear_parameter("linear_1"),
                input_adjoint_pointer="args.message_adjoint",
                input_stride=message_dimension,
                stem=f"node_tiled_l{index}_reverse_message",
                tile_nodes=(
                    _CUDA_NODE_WIDE_GROUPED_TILE_NODES
                    if wide_layer_one_reverse_message and index == 1
                    else _CUDA_NODE_LINEAR2_TILE_NODES
                ),
                symbol_prefix=symbol_prefix,
                density_input_pointer=(
                    "args.messages" if reuse_message_adjoint else None
                ),
                density_result_pointer=(
                    "args.node_density_adjoint" if reuse_message_adjoint else None
                ),
            )
        )
        tiled_definitions.append(definitions)
        tiled_launch_records[reverse_message] = tuple(reverse_message_records)
    else:
        reverse_message = tiled_reverse(
            layer["linears"]["linear_1"],
            args_type=reverse_args,
            output_adjoint_pointer=f"args.node_arena + {pregate_adjoint_offset}",
            output_stride=arena_dimension,
            parameter_pointer=linear_parameter("linear_1"),
            input_adjoint_pointer="args.message_adjoint",
            input_stride=message_dimension,
            stem=f"node_tiled_l{index}_reverse_message",
            tile_nodes=_CUDA_NODE_LINEAR2_TILE_NODES,
        )
    linear_1_segments = {
        segment["name"]: segment
        for segment in layer["linears"]["linear_1"]["runtime_parameters"]["segments"]
    }
    bias_base = int(linear_1_segments["bias"]["offset"])
    bias_count = int(linear_1_segments["bias"]["count"])
    mask_base = int(linear_1_segments["output_mask"]["offset"])
    bias_dot = ""
    if bias_count:
        bias_dot = f"""for (int column = threadIdx.x; column < {residual_dimension}; column += blockDim.x)
    l{index}_tiled_density_sum += {linear_parameter('linear_1')}[{bias_base} + column]
        * {linear_parameter('linear_1')}[{mask_base} + column]
        * pre_gate_adjoint[column];"""
    if reuse_message_adjoint:
        scale_bias = (
            f"""const float scaled = pre_gate_adjoint[column] / normalization;
        pre_gate_adjoint[column] = scaled;
        l{index}_tiled_density_bias_sum +=
            {linear_parameter('linear_1')}[{bias_base} + column]
            * {linear_parameter('linear_1')}[{mask_base} + column] * scaled;"""
            if bias_count
            else "pre_gate_adjoint[column] /= normalization;"
        )
        scale_pregate_stage = f"""float* pre_gate_adjoint = args.node_arena
    + node * {arena_dimension} + {pregate_adjoint_offset};
const float normalization = args.linear_parameters[{normalization_offset}]
    + args.linear_parameters[{normalization_offset + 1}] * args.node_density[node];
float l{index}_tiled_density_bias_sum = 0.0f;
for (int column = threadIdx.x; column < {residual_dimension}; column += blockDim.x) {{
    {scale_bias}
}}
__shared__ float l{index}_tiled_density_bias_reduction[128];
l{index}_tiled_density_bias_reduction[threadIdx.x] =
    l{index}_tiled_density_bias_sum;
__syncthreads();
for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {{
    if (threadIdx.x < offset)
        l{index}_tiled_density_bias_reduction[threadIdx.x] +=
            l{index}_tiled_density_bias_reduction[threadIdx.x + offset];
    __syncthreads();
}}
if (threadIdx.x == 0)
    args.node_density_adjoint[node] =
        l{index}_tiled_density_bias_reduction[0];
__syncthreads();"""
    density_stage = f"""const float* messages = args.messages + node * {message_dimension};
const float* message_adjoint = args.message_adjoint + node * {message_dimension};
const float* pre_gate_adjoint = args.node_arena
    + node * {arena_dimension} + {pregate_adjoint_offset};
float* density_adjoint = args.node_density_adjoint + node;
float l{index}_tiled_density_sum = 0.0f;
for (int column = threadIdx.x; column < {message_dimension}; column += blockDim.x)
    l{index}_tiled_density_sum += messages[column] * message_adjoint[column];
{bias_dot}
__shared__ float l{index}_tiled_density_reduction[128];
l{index}_tiled_density_reduction[threadIdx.x] = l{index}_tiled_density_sum;
__syncthreads();
for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {{
    if (threadIdx.x < offset)
        l{index}_tiled_density_reduction[threadIdx.x] +=
            l{index}_tiled_density_reduction[threadIdx.x + offset];
    __syncthreads();
}}
if (threadIdx.x == 0) {{
    const float normalization = args.linear_parameters[{normalization_offset}]
        + args.linear_parameters[{normalization_offset + 1}] * args.node_density[node];
    *density_adjoint = -args.linear_parameters[{normalization_offset + 1}]
        * l{index}_tiled_density_reduction[0] / normalization;
}}
__syncthreads();"""
    if reuse_message_adjoint:
        density_stage = f"""float* density_adjoint = args.node_density_adjoint + node;
if (threadIdx.x == 0) {{
    const float normalization = args.linear_parameters[{normalization_offset}]
        + args.linear_parameters[{normalization_offset + 1}] * args.node_density[node];
    *density_adjoint = -args.linear_parameters[{normalization_offset + 1}]
        * *density_adjoint / normalization;
}}
__syncthreads();"""

    kernel_linkage = 'extern "C" ' if symbol_prefix else ""

    def segment_kernel(name: str, args_type: str, body: str) -> str:
        return f"""{kernel_linkage}__global__ void {name}({args_type} args)
{{
    for (std::int64_t node = blockIdx.x; node < args.num_nodes;
         node += gridDim.x) {{
{body}
    }}
}}"""

    forward_gate_name = f"{symbol_prefix}node_tiled_l{index}_forward_gate"
    forward_product_name = f"{symbol_prefix}node_tiled_l{index}_forward_product"
    forward_product_setup_name = (
        f"{symbol_prefix}node_tiled_l{index}_forward_product_setup"
    )
    forward_finalize_name = f"{symbol_prefix}node_tiled_l{index}_forward_finalize"
    reverse_readout_name = f"{symbol_prefix}node_tiled_l{index}_reverse_readout"
    reverse_gate_forward_name = (
        f"{symbol_prefix}node_tiled_l{index}_reverse_gate_forward"
    )
    reverse_product_name = f"{symbol_prefix}node_tiled_l{index}_reverse_product"
    reverse_gate_name = f"{symbol_prefix}node_tiled_l{index}_reverse_gate"
    reverse_scale_name = f"{symbol_prefix}node_tiled_l{index}_reverse_scale"
    reverse_density_name = f"{symbol_prefix}node_tiled_l{index}_reverse_density"
    forward_product_definition = (
        segment_kernel(
            forward_product_setup_name,
            forward_args,
            (
                retain_interaction_and_skip_input_stage
                if index == 0 and grouped_layer_zero_skip_forward
                else (
                    retain_interaction_stage
                    if index == 1 and grouped_skip_forward
                    else skip_product_setup_stage
                )
            ),
        )
        if node_reverse_schedule
        in (
            _MH1_CUDA_TILED_PRODUCT_FORWARD_REVERSE_SCHEDULE,
            _MH1_CUDA_TILED_LINEAR_UP_REVERSE_SCHEDULE,
            _MH1_CUDA_TILED_LINEAR_UP_FORWARD_REVERSE_SCHEDULE,
            _MH1_CUDA_GROUPED_READOUT_REVERSE_SCHEDULE,
            _MH1_CUDA_GROUPED_SKIP_FORWARD_SCHEDULE,
            _MH1_CUDA_GROUPED_ALL_SKIP_FORWARD_SCHEDULE,
            _MH1_CUDA_WIDE_PRODUCT_LINEAR_SCHEDULE,
            _MH1_CUDA_WIDE_L1_REVERSE_MESSAGE_SCHEDULE,
        )
        else segment_kernel(forward_product_name, forward_args, skip_product_stage)
    )
    reverse_readout_definition = (
        segment_kernel(
            f"{symbol_prefix}node_tiled_l{index}_reverse_readout_middle",
            reverse_args,
            readout_reverse_middle_body,
        )
        if specialized_readout_reverse_records
        else segment_kernel(
            reverse_readout_name, reverse_args, readout_skip_reverse_stage
        )
    )
    segment_definitions = [
        segment_kernel(forward_gate_name, forward_args, gate_forward_stage),
        forward_product_definition,
        segment_kernel(forward_finalize_name, forward_args, finalize_forward_stage),
        reverse_readout_definition,
        segment_kernel(reverse_gate_name, reverse_args, gate_reverse_stage),
        segment_kernel(reverse_scale_name, reverse_args, scale_pregate_stage),
        segment_kernel(reverse_density_name, reverse_args, density_stage),
    ]
    if node_reverse_schedule not in (
        _MH1_CUDA_TILED_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_RECOMPUTE_TILED_PRODUCT_REVERSE_SCHEDULE,
        _MH1_CUDA_TILED_PRODUCT_FORWARD_REVERSE_SCHEDULE,
        _MH1_CUDA_TILED_LINEAR_UP_REVERSE_SCHEDULE,
        _MH1_CUDA_TILED_LINEAR_UP_FORWARD_REVERSE_SCHEDULE,
        _MH1_CUDA_GROUPED_READOUT_REVERSE_SCHEDULE,
    ):
        segment_definitions.insert(
            4,
            segment_kernel(reverse_product_name, reverse_args, product_reverse_stage),
        )
    if not retain_interaction_output:
        segment_definitions.insert(
            5,
            segment_kernel(reverse_gate_forward_name, reverse_args, gate_forward_stage),
        )
    tiled_definitions.extend(segment_definitions)
    forward_product_setup_launch = (
        f"{forward_product_setup_name}<<<blocks, 128, 0, cuda_stream>>>(*args);"
    )
    if grouped_skip_forward_launch:
        forward_product_setup_launch += f"\n    {grouped_skip_forward_launch}"
    forward_product_launch = (
        f"{forward_product_setup_launch}\n" f"    {tiled_product_forward}"
        if node_reverse_schedule
        in (
            _MH1_CUDA_TILED_PRODUCT_FORWARD_REVERSE_SCHEDULE,
            _MH1_CUDA_TILED_LINEAR_UP_REVERSE_SCHEDULE,
            _MH1_CUDA_TILED_LINEAR_UP_FORWARD_REVERSE_SCHEDULE,
            _MH1_CUDA_GROUPED_READOUT_REVERSE_SCHEDULE,
        )
        else f"{forward_product_name}<<<blocks, 128, 0, cuda_stream>>>(*args);"
    )
    tiled_post_forward_launches = f"""{forward_pregate}
    {forward_gate_name}<<<blocks, 128, 0, cuda_stream>>>(*args);
    {forward_linear_2}
    {forward_product_launch}
    {forward_product_linear}
    {forward_finalize_name}<<<blocks, 128, 0, cuda_stream>>>(*args);"""
    reverse_gate_forward_launch = (
        ""
        if retain_interaction_output
        else f"{reverse_gate_forward_name}<<<blocks, 128, 0, cuda_stream>>>(*args);"
    )
    reverse_product_launch = (
        tiled_product_reverse
        if node_reverse_schedule
        in (
            _MH1_CUDA_TILED_PRODUCT_REVERSE_SCHEDULE,
            _MH1_CUDA_RECOMPUTE_TILED_PRODUCT_REVERSE_SCHEDULE,
            _MH1_CUDA_TILED_PRODUCT_FORWARD_REVERSE_SCHEDULE,
            _MH1_CUDA_TILED_LINEAR_UP_REVERSE_SCHEDULE,
            _MH1_CUDA_TILED_LINEAR_UP_FORWARD_REVERSE_SCHEDULE,
            _MH1_CUDA_GROUPED_READOUT_REVERSE_SCHEDULE,
        )
        else f"{reverse_product_name}<<<blocks, 128, 0, cuda_stream>>>(*args);"
    )
    reverse_readout_launch = (
        specialized_readout_reverse_launch
        if specialized_readout_reverse_records
        else f"{reverse_readout_name}<<<blocks, 128, 0, cuda_stream>>>(*args);"
    )
    tiled_post_reverse_launches = f"""{reverse_readout_launch}
    {reverse_pregate}
    {reverse_gate_forward_launch}
    {reverse_linear_2_forward}
    {reverse_product_linear}
    {reverse_product_launch}
    {reverse_linear_2}
    {reverse_pregate_recompute}
    {reverse_gate_name}<<<blocks, 128, 0, cuda_stream>>>(*args);
    {reverse_residual}
    {reverse_scale_name}<<<blocks, 128, 0, cuda_stream>>>(*args);
    {reverse_message}
    {reverse_density_name}<<<blocks, 128, 0, cuda_stream>>>(*args);"""
    pre_reverse_body = ""
    reverse_pre_kernel = ""
    reverse_pre_launcher = ""
    reverse_pre_launch_records: list[MH1CudaKernelLaunch] = []
    if index > 0:
        if node_reverse_schedule in (
            _MH1_CUDA_TILED_LINEAR_UP_REVERSE_SCHEDULE,
            _MH1_CUDA_TILED_LINEAR_UP_FORWARD_REVERSE_SCHEDULE,
            _MH1_CUDA_GROUPED_READOUT_REVERSE_SCHEDULE,
        ):
            definitions, reverse_pre_launch, reverse_pre_launch_records = (
                _cuda_grouped_tiled_linear_reverse(
                    layer["linears"]["linear_up"],
                    args_type=reverse_args,
                    output_adjoint_pointer="args.up_adjoint",
                    output_stride=up_dimension,
                    parameter_pointer=linear_parameter("linear_up"),
                    input_adjoint_pointer="args.layer_input_adjoint",
                    input_stride=input_dimension,
                    stem=f"node_tiled_l{index}_pre_reverse_linear_up",
                    add=True,
                    tile_nodes=_CUDA_NODE_LINEAR2_TILE_NODES,
                    symbol_prefix=symbol_prefix,
                )
            )
            tiled_definitions.append(definitions)
        else:
            pre_reverse_body = "\n".join(
                [
                    f"float* arena = args.node_arena + node * {arena_dimension};",
                    f"const float* up_adjoint = args.up_adjoint + node * {up_dimension};",
                    f"float* layer_input_adjoint = args.layer_input_adjoint + node * {input_dimension};",
                    _cuda_block_linear_reverse(
                        layer["linears"]["linear_up"],
                        output_adjoint_pointer="up_adjoint",
                        parameter_pointer=linear_parameter("linear_up"),
                        input_adjoint_pointer="layer_input_adjoint",
                        stem="l1_up_reverse",
                        add=True,
                    ),
                ]
            )
            reverse_pre_kernel = f"""
{kernel_linkage}__global__ void {symbol_prefix}node_pre_reverse_kernel_{index}(
    SymmetrixJitMH1CudaNodeReverseArgsV4 args)
{{
    for (std::int64_t node = blockIdx.x; node < args.num_nodes;
         node += gridDim.x) {{
{pre_reverse_body}
    }}
}}
"""
            reverse_pre_launch = (
                f"{symbol_prefix}node_pre_reverse_kernel_{index}"
                "<<<blocks, threads, 0, cuda_stream>>>(*args);"
            )
            reverse_pre_launch_records = [
                MH1CudaKernelLaunch(
                    f"{symbol_prefix}node_pre_reverse_kernel_{index}",
                    "persistent_nodes",
                    128,
                )
            ]
        reverse_pre_launcher = f"""
std::int32_t node_pre_reverse_launch_{index}(
    const SymmetrixJitMH1CudaNodeReverseArgsV4* args, void* stream,
    std::int32_t persistent_blocks)
{{
    if (args == nullptr || args->interaction != {index}u
        || args->phase != SYMMETRIX_JIT_MH1_CUDA_NODE_PRE_REVERSE_V4
        || args->num_nodes < 0 || args->num_nodes > INT32_MAX
        || persistent_blocks <= 0)
        return static_cast<std::int32_t>(cudaErrorInvalidValue);
    if (args->num_nodes == 0) return static_cast<std::int32_t>(cudaSuccess);
    if (args->linear_parameters == nullptr || args->up_adjoint == nullptr
        || args->node_arena == nullptr || args->layer_input_adjoint == nullptr)
        return static_cast<std::int32_t>(cudaErrorInvalidDevicePointer);
    constexpr std::int32_t threads = 128;
    const auto blocks = static_cast<std::int32_t>(
        args->num_nodes < persistent_blocks ? args->num_nodes : persistent_blocks);
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    {reverse_pre_launch}
    return static_cast<std::int32_t>(cudaPeekAtLastError());
}}
"""
    tiled_source = "\n\n".join(tiled_definitions)
    layer_input_forward_check = "" if index == 0 else " || args->layer_input == nullptr"
    layer_input_reverse_check = (
        ""
        if index == 0
        else " || args->layer_input == nullptr"
        " || args->layer_input_adjoint == nullptr"
    )
    retained_forward_check = (
        " || args->retained_pre_gate == nullptr" if retain_pre_gate else ""
    ) + (
        " || args->retained_interaction_output == nullptr"
        if retain_interaction_output
        else ""
    )
    retained_reverse_check = retained_forward_check
    pre_forward_kernel_name = f"{symbol_prefix}node_pre_forward_kernel_{index}"
    if not pre_forward_launch_records:
        pre_forward_kernel = f"""{kernel_linkage}__global__ void {pre_forward_kernel_name}(
    SymmetrixJitMH1CudaNodeForwardArgsV4 args)
{{
    for (std::int64_t node = blockIdx.x; node < args.num_nodes;
         node += gridDim.x) {{
        float* arena = args.node_arena + node * {arena_dimension};
{pre_forward_body}
    }}
}}
"""
        pre_forward_launch = (
            f"{pre_forward_kernel_name}<<<blocks, threads, 0, cuda_stream>>>(*args);"
        )
        pre_forward_launch_records = [
            MH1CudaKernelLaunch(pre_forward_kernel_name, "persistent_nodes", 128)
        ]
    else:
        pre_forward_kernel = ""
    device_source = f"""{tiled_source}

{pre_forward_kernel}

{kernel_linkage}__global__ void {symbol_prefix}node_post_forward_kernel_{index}(
    SymmetrixJitMH1CudaNodeForwardArgsV4 args)
{{
    for (std::int64_t node = blockIdx.x; node < args.num_nodes;
         node += gridDim.x) {{
{post_forward_body}
    }}
}}

{kernel_linkage}__global__ void {symbol_prefix}node_post_reverse_kernel_{index}(
    SymmetrixJitMH1CudaNodeReverseArgsV4 args)
{{
    for (std::int64_t node = blockIdx.x; node < args.num_nodes;
         node += gridDim.x) {{
{post_reverse_body}
    }}
}}
{reverse_pre_kernel}
"""
    launcher_source = f"""std::int32_t node_pre_forward_launch_{index}(
    const SymmetrixJitMH1CudaNodeForwardArgsV4* args, void* stream,
    std::int32_t persistent_blocks)
{{
    if (args == nullptr || args->interaction != {index}u
        || args->phase != SYMMETRIX_JIT_MH1_CUDA_NODE_PRE_FORWARD_V4
        || args->num_nodes < 0 || args->num_nodes > INT32_MAX
        || persistent_blocks <= 0)
        return static_cast<std::int32_t>(cudaErrorInvalidValue);
    if (args->num_nodes == 0) return static_cast<std::int32_t>(cudaSuccess);
    if (args->element_indices == nullptr || args->linear_parameters == nullptr
        || args->up_output == nullptr || args->node_arena == nullptr{layer_input_forward_check})
        return static_cast<std::int32_t>(cudaErrorInvalidDevicePointer);
    constexpr std::int32_t threads = 128;
    const auto blocks = static_cast<std::int32_t>(
        args->num_nodes < persistent_blocks ? args->num_nodes : persistent_blocks);
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    {pre_forward_launch}
    return static_cast<std::int32_t>(cudaPeekAtLastError());
}}

std::int32_t node_post_forward_launch_{index}(
    const SymmetrixJitMH1CudaNodeForwardArgsV4* args, void* stream,
    std::int32_t persistent_blocks)
{{
    if (args == nullptr || args->interaction != {index}u
        || args->phase != SYMMETRIX_JIT_MH1_CUDA_NODE_POST_FORWARD_V4
        || args->num_nodes < 0 || args->num_nodes > INT32_MAX
        || persistent_blocks <= 0)
        return static_cast<std::int32_t>(cudaErrorInvalidValue);
    if (args->num_nodes == 0) return static_cast<std::int32_t>(cudaSuccess);
    if (args->element_indices == nullptr || args->node_density == nullptr
        || args->linear_parameters == nullptr || args->product_parameters == nullptr
        || args->readout_parameters == nullptr || args->up == nullptr
        || args->messages == nullptr || args->layer_output == nullptr
        || args->readout_contribution == nullptr || args->node_arena == nullptr{layer_input_forward_check}{retained_forward_check})
        return static_cast<std::int32_t>(cudaErrorInvalidDevicePointer);
    constexpr std::int32_t threads = 128;
    const auto blocks = static_cast<std::int32_t>(
        args->num_nodes < persistent_blocks ? args->num_nodes : persistent_blocks);
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    {tiled_post_forward_launches}
    return static_cast<std::int32_t>(cudaPeekAtLastError());
}}

std::int32_t node_post_reverse_launch_{index}(
    const SymmetrixJitMH1CudaNodeReverseArgsV4* args, void* stream,
    std::int32_t persistent_blocks)
{{
    if (args == nullptr || args->interaction != {index}u
        || args->phase != SYMMETRIX_JIT_MH1_CUDA_NODE_POST_REVERSE_V4
        || args->reserved != 0u || args->num_nodes < 0
        || args->num_nodes > INT32_MAX || persistent_blocks <= 0)
        return static_cast<std::int32_t>(cudaErrorInvalidValue);
    if (args->num_nodes == 0) return static_cast<std::int32_t>(cudaSuccess);
    if (args->element_indices == nullptr || args->node_density == nullptr
        || args->linear_parameters == nullptr || args->product_parameters == nullptr
        || args->readout_parameters == nullptr || args->up == nullptr
        || args->messages == nullptr || args->layer_output == nullptr
        || args->node_arena == nullptr || args->layer_output_adjoint == nullptr
        || args->message_adjoint == nullptr || args->up_adjoint == nullptr
        || args->node_density_adjoint == nullptr{layer_input_reverse_check}{retained_reverse_check})
        return static_cast<std::int32_t>(cudaErrorInvalidDevicePointer);
    constexpr std::int32_t threads = 128;
    const auto blocks = static_cast<std::int32_t>(
        args->num_nodes < persistent_blocks ? args->num_nodes : persistent_blocks);
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    {tiled_post_reverse_launches}
    return static_cast<std::int32_t>(cudaPeekAtLastError());
}}
{reverse_pre_launcher}"""
    if not return_program:
        return device_source + launcher_source

    def persistent(kernel: str) -> MH1CudaKernelLaunch:
        return MH1CudaKernelLaunch(kernel, "persistent_nodes", 128)

    def records(launches: str) -> list[MH1CudaKernelLaunch]:
        return list(tiled_launch_records.get(launches, ()))

    forward_records = (
        records(forward_residual)
        + records(forward_message)
        + [persistent(forward_gate_name)]
        + records(forward_linear_2)
        + (
            [persistent(forward_product_setup_name)]
            + list(grouped_skip_forward_records)
            + records(tiled_product_forward)
            if node_reverse_schedule
            in (
                _MH1_CUDA_TILED_PRODUCT_FORWARD_REVERSE_SCHEDULE,
                _MH1_CUDA_TILED_LINEAR_UP_REVERSE_SCHEDULE,
                _MH1_CUDA_TILED_LINEAR_UP_FORWARD_REVERSE_SCHEDULE,
                _MH1_CUDA_GROUPED_READOUT_REVERSE_SCHEDULE,
            )
            else [persistent(forward_product_name)]
        )
        + records(forward_product_linear)
        + [persistent(forward_finalize_name)]
    )
    replay_records = (
        records(reverse_residual_forward)
        + records(reverse_message_forward)
        + [persistent(reverse_gate_forward_name)]
        + records(reverse_linear_2_forward)
        if not retain_interaction_output
        else []
    )
    recompute_records = (
        records(reverse_residual_recompute) + records(reverse_message_recompute)
        if not retain_pre_gate
        else []
    )
    reverse_records = (
        (
            list(specialized_readout_reverse_records)
            if specialized_readout_reverse_records
            else [persistent(reverse_readout_name)]
        )
        + replay_records
        + records(reverse_product_linear)
        + (
            records(tiled_product_reverse)
            if node_reverse_schedule
            in (
                _MH1_CUDA_TILED_PRODUCT_REVERSE_SCHEDULE,
                _MH1_CUDA_RECOMPUTE_TILED_PRODUCT_REVERSE_SCHEDULE,
                _MH1_CUDA_TILED_PRODUCT_FORWARD_REVERSE_SCHEDULE,
                _MH1_CUDA_TILED_LINEAR_UP_REVERSE_SCHEDULE,
                _MH1_CUDA_TILED_LINEAR_UP_FORWARD_REVERSE_SCHEDULE,
                _MH1_CUDA_GROUPED_READOUT_REVERSE_SCHEDULE,
            )
            else [persistent(reverse_product_name)]
        )
        + records(reverse_linear_2)
        + recompute_records
        + [persistent(reverse_gate_name)]
        + records(reverse_residual)
        + [persistent(reverse_scale_name)]
        + records(reverse_message)
        + [persistent(reverse_density_name)]
    )
    return MH1NodeCudaProgram(
        device_source=device_source,
        plugin_source=device_source + launcher_source,
        post_forward_launches=tuple(forward_records),
        post_reverse_launches=tuple(reverse_records),
        pre_reverse_launches=tuple(reverse_pre_launch_records),
        pre_forward_launches=tuple(pre_forward_launch_records),
    )


def render_jit_mh1_cuda_plugin_v4(
    contract: dict[str, Any],
    compute_capability: int,
    *,
    forward_policy: str | dict[str, Any] = "auto",
    source_policy: str | dict[str, Any] = "auto",
    edge_policy: str | dict[str, Any] = "auto",
    node_state_policy: str = MH1_NODE_STATE_FULL_RETENTION,
) -> str:
    """Render a dual-query CUDA artifact with generated node launches."""

    normalized = normalize_execution_mh1_v4_contract(contract)
    projected = project_execution_mh1_v3_contract(normalized)
    source = render_jit_mh1_cuda_plugin(
        projected,
        compute_capability,
        forward_policy=forward_policy,
        source_policy=source_policy,
        edge_policy=edge_policy,
    )
    metadata = jit_mh1_cuda_plugin_v4_metadata(
        normalized,
        compute_capability,
        forward_policy=forward_policy,
        source_policy=source_policy,
        edge_policy=edge_policy,
        node_state_policy=node_state_policy,
    )
    node_helpers = ""
    node_layers = normalized["node_program"]["layers"]
    embedding = normalized["node_program"]["input"]["embedding"]
    programs = "\n\n".join(
        _render_cuda_node_program(layer, layer_metadata, embedding)
        for layer, layer_metadata in zip(node_layers, metadata["layers"], strict=True)
    )
    interactions = ", ".join(
        "{"
        + ", ".join(
            str(value)
            for value in (
                "sizeof(SymmetrixJitMH1CudaInteractionV4)",
                "0u",
                item["input_1_dimension"],
                item["input_2_dimension"],
                item["output_dimension"],
                item["weight_size"],
                item["phi_dimension"],
                item["multiplicity"],
                item["input_1_angular_dimension"],
                item["instruction_count"],
                metadata["forward_threads_per_block"],
                metadata["source_threads_per_block"],
                metadata["edge_threads_per_block"],
                metadata["edge_policy"]["interactions"][index]["physical_launch_count"],
            )
        )
        + "}"
        for index, item in enumerate(metadata["interactions"])
    )
    launches = ", ".join(
        "{"
        + ", ".join(
            (
                "sizeof(SymmetrixJitMH1CudaInteractionLaunchesV4)",
                f"{index}u",
                "0u",
                "0u",
                f"&forward_launch_{index}",
                f"&reverse_launch_{index}",
            )
        )
        + "}"
        for index in range(2)
    )
    conditioning = ", ".join(
        "{"
        + ", ".join(
            (
                "sizeof(SymmetrixJitMH1CudaConditioningLaunchesV4)",
                f"{index}u",
                "0u",
                "0u",
                f"&conditioning_forward_launch_{index}",
                f"&conditioning_reverse_launch_{index}",
            )
        )
        + "}"
        for index in range(2)
    )
    node_descriptors = []
    for item in metadata["layers"]:
        index = int(item["index"])
        forward_descriptors = ", ".join(
            "{"
            + ", ".join(
                (
                    "sizeof(SymmetrixJitMH1CudaNodeForwardPhaseV4)",
                    f"{index}u",
                    f"{phase['phase']}u",
                    "0u",
                    str(phase["threads_per_block"]),
                    "0",
                    "0",
                    "0",
                    f"&node_{'pre' if phase['phase']==0 else 'post'}_forward_launch_{index}",
                )
            )
            + "}"
            for phase in item["forward_phases"]
        )
        reverse_descriptors = []
        for phase in item["reverse_phases"]:
            owner = (
                f"&node_{'post' if phase['phase']==0 else 'pre'}_reverse_launch_{index}"
                if phase["enabled"]
                else "nullptr"
            )
            reverse_descriptors.append(
                "{"
                + ", ".join(
                    (
                        "sizeof(SymmetrixJitMH1CudaNodeReversePhaseV4)",
                        f"{index}u",
                        f"{2+phase['phase']}u",
                        "0u",
                        str(phase["threads_per_block"]),
                        "0",
                        "0",
                        "0",
                        owner,
                    )
                )
                + "}"
            )
        flag_names = []
        if item["requires_tp_source_state_adjoint"]:
            flag_names.append(
                "SYMMETRIX_JIT_MH1_CUDA_NODE_REQUIRES_TP_SOURCE_STATE_ADJOINT_V4"
            )
        if item["retained_pre_gate_dimension"]:
            flag_names.append("SYMMETRIX_JIT_MH1_CUDA_NODE_RETAIN_PRE_GATE_V4")
        if item["retained_interaction_output_dimension"]:
            flag_names.append(
                "SYMMETRIX_JIT_MH1_CUDA_NODE_RETAIN_INTERACTION_OUTPUT_V4"
            )
        if item["reuse_message_adjoint"]:
            flag_names.append("SYMMETRIX_JIT_MH1_CUDA_NODE_REUSE_MESSAGE_ADJOINT_V4")
        flags = " | ".join(flag_names) if flag_names else "0u"
        node_descriptors.append(
            "{"
            + ", ".join(
                (
                    "sizeof(SymmetrixJitMH1CudaNodeProgramV4)",
                    f"{index}u",
                    "2u",
                    "2u",
                    str(item["element_count"]),
                    str(item["input_dimension"]),
                    str(item["up_dimension"]),
                    str(item["residual_dimension"]),
                    str(item["skip_dimension"]),
                    str(item["message_dimension"]),
                    str(item["interaction_output_dimension"]),
                    str(item["output_dimension"]),
                    str(item["product_term_count"]),
                    str(item["node_arena_dimension"]),
                    flags,
                    str(item["retained_pre_gate_dimension"]),
                    str(item["linear_parameter_count"]),
                    str(item["product_parameter_count"]),
                    str(item["readout_parameter_count"]),
                    str(item["retained_interaction_output_dimension"]),
                    "{" + forward_descriptors + "}",
                    "{" + ", ".join(reverse_descriptors) + "}",
                )
            )
            + "}"
        )
    return (
        source
        + f"""

namespace {{
{node_helpers}
{programs}

const SymmetrixJitMH1CudaPluginV4 descriptor_v4 {{
    SYMMETRIX_JIT_MH1_CUDA_PLUGIN_V4_ABI_VERSION,
    sizeof(SymmetrixJitMH1CudaPluginV4), sizeof(void*),
    SYMMETRIX_JIT_MH1_CUDA_PLUGIN_BYTE_ORDER,
    SYMMETRIX_JIT_MH1_CUDA_FORWARD_LAUNCH_V4
        | SYMMETRIX_JIT_MH1_CUDA_REVERSE_LAUNCH_V4
        | SYMMETRIX_JIT_MH1_CUDA_PHI_MAJOR_LINEAR_WEIGHT_V4
        | SYMMETRIX_JIT_MH1_CUDA_ACTIVE_RECEIVERS_V4
        | SYMMETRIX_JIT_MH1_CUDA_GRAPH_WIDE_V4
        | SYMMETRIX_JIT_MH1_CUDA_IR_MUL_FEATURES_V4
        | SYMMETRIX_JIT_MH1_CUDA_GLOBAL_SOURCE_REVERSE_V4
        | SYMMETRIX_JIT_MH1_CUDA_EDGE_LAUNCH_COUNT_V4
        | SYMMETRIX_JIT_MH1_CUDA_CONDITIONING_LAUNCH_V4
        | SYMMETRIX_JIT_MH1_CUDA_NODE_PROGRAM_PHASES_V4
        | SYMMETRIX_JIT_MH1_CUDA_PERSISTENT_IR_MUL_NODE_STATE_V4
        | SYMMETRIX_JIT_MH1_CUDA_FIXED_WEIGHT_COORDINATES_V4,
    2u, sizeof(float), 0u,
    SYMMETRIX_JIT_MH1_CUDA_PLUGIN_V4_ABI_TAG,
    {_cpp_string(metadata['artifact_id'])},
    {_cpp_string(metadata['generation_fingerprint'])},
    {_cpp_string(metadata['semantic_fingerprint'])},
    {_cpp_string(metadata['structure_fingerprint'])},
    {_cpp_string(metadata['runtime_layout_fingerprint'])},
    {compute_capability}, 0,
    {{ {interactions} }}, {{ {launches} }}, {{ {conditioning} }},
    {{ {', '.join(node_descriptors)} }},
}};
}}  // namespace

extern "C" SYMMETRIX_JIT_MH1_CUDA_PLUGIN_EXPORT
const SymmetrixJitMH1CudaPluginV4*
symmetrix_jit_mh1_cuda_plugin_query_v4(void)
{{
    return &descriptor_v4;
}}
"""
    )


def execution_mh1_cuda_module_v4_launch_plan(
    contract: dict[str, Any],
    compute_capability: int,
    *,
    forward_policy: str | dict[str, Any] = "auto",
    source_policy: str | dict[str, Any] = "auto",
    edge_policy: str | dict[str, Any] = "auto",
    node_state_policy: str = MH1_NODE_STATE_FULL_RETENTION,
    precision: str = "float32",
    _edge_reverse_schedule: str = _MH1_CUDA_EDGE_REVERSE_SCHEDULE,
    _node_reverse_schedule: str | None = None,
) -> dict[str, Any]:
    """Return the closed native launch plan for an MH1 NVRTC module."""

    normalized = normalize_execution_mh1_v4_contract(contract)
    node_state_policy = _resolve_node_state_policy(node_state_policy)
    if _node_reverse_schedule is None:
        _node_reverse_schedule = _cuda_module_node_reverse_schedule(node_state_policy)
    if (
        _node_reverse_schedule
        in (
            _MH1_CUDA_RETAINED_NODE_REVERSE_SCHEDULE,
            _MH1_CUDA_GROUPED_MESSAGE_REVERSE_SCHEDULE,
            _MH1_CUDA_GROUPED_LINEAR2_SCHEDULE,
            _MH1_CUDA_GROUPED_ALL_MESSAGE_SCHEDULE,
            _MH1_CUDA_GROUPED_FORWARD_MESSAGE_SCHEDULE,
            _MH1_CUDA_GROUPED_RESIDUAL_SCHEDULE,
            _MH1_CUDA_GROUPED_FORWARD_RESIDUAL_SCHEDULE,
            _MH1_CUDA_TILED_PRODUCT_REVERSE_SCHEDULE,
            _MH1_CUDA_TILED_PRODUCT_FORWARD_REVERSE_SCHEDULE,
            _MH1_CUDA_TILED_LINEAR_UP_REVERSE_SCHEDULE,
            _MH1_CUDA_TILED_LINEAR_UP_FORWARD_REVERSE_SCHEDULE,
            _MH1_CUDA_GROUPED_READOUT_REVERSE_SCHEDULE,
            _MH1_CUDA_GROUPED_SKIP_FORWARD_SCHEDULE,
            _MH1_CUDA_GROUPED_ALL_SKIP_FORWARD_SCHEDULE,
            _MH1_CUDA_WIDE_PRODUCT_LINEAR_SCHEDULE,
            _MH1_CUDA_WIDE_L1_REVERSE_MESSAGE_SCHEDULE,
        )
        and node_state_policy == MH1_NODE_STATE_RECOMPUTE
    ):
        raise ValueError("experimental MH1 reverse requires full node retention")
    projected = project_execution_mh1_v3_contract(normalized)
    resolved_forward = resolve_execution_mh1_cuda_forward_policy(
        projected, compute_capability, forward_policy
    )
    resolved_source = resolve_execution_mh1_cuda_source_policy(
        projected, compute_capability, source_policy
    )
    resolved_edge = resolve_execution_mh1_cuda_edge_policy(
        projected,
        compute_capability,
        edge_policy,
        _reverse_schedule=_edge_reverse_schedule,
    )
    metadata = jit_mh1_cuda_plugin_v4_metadata(
        normalized,
        compute_capability,
        forward_policy=resolved_forward,
        source_policy=resolved_source,
        edge_policy=resolved_edge,
        node_state_policy=node_state_policy,
        _edge_reverse_schedule=_edge_reverse_schedule,
        _node_reverse_schedule=_node_reverse_schedule,
    )
    node_layers = normalized["node_program"]["layers"]
    embedding = normalized["node_program"]["input"]["embedding"]
    programs = [
        _render_cuda_node_program(
            layer,
            layer_metadata,
            embedding,
            return_program=True,
            symbol_prefix="symmetrix_execution_mh1_",
            node_reverse_schedule=_node_reverse_schedule,
        )
        for layer, layer_metadata in zip(node_layers, metadata["layers"], strict=True)
    ]
    interactions = []
    for index in range(2):
        edge_phi_schedule = _edge_phi_schedule(projected["interactions"][index])
        interactions.append(
            {
                "index": index,
                "forward_partitions": len(
                    resolved_forward["interactions"][index]["partitions"]
                ),
                "source_partitions": len(
                    resolved_source["interactions"][index]["partitions"]
                ),
                "edge_strategy": resolved_edge["interactions"][index]["strategy"],
                "edge_phi_schedule": edge_phi_schedule,
                "edge_phi_block_multiplier": (
                    _MH1_PATH_TILED_EDGE_BLOCK_MULTIPLIER
                    if resolved_edge["interactions"][index]["strategy"]
                    == "compact_fused"
                    or (
                        resolved_edge["interactions"][index]["strategy"] == "split"
                        and edge_phi_schedule == "path_tiled"
                    )
                    else 1
                ),
                "forward_threads": metadata["forward_threads_per_block"],
                "source_threads": metadata["source_threads_per_block"],
                "edge_threads": metadata["edge_threads_per_block"],
                "spline_r": {
                    "forward_kernel": (
                        f"symmetrix_execution_mh1_spline_r_forward_kernel_{index}"
                    ),
                    "reverse_kernel": (
                        f"symmetrix_execution_mh1_spline_r_reverse_kernel_{index}"
                    ),
                    "forward_threads": 256,
                    "reverse_threads": 64,
                    "schedule": _MH1_SPLINE_R_SCHEDULE,
                },
            }
        )
    nodes = []
    for index, program in enumerate(programs):
        pre_forward = [
            launch.as_plan_entry() for launch in program.pre_forward_launches
        ]
        reverse_pre = [
            launch.as_plan_entry() for launch in program.pre_reverse_launches
        ]
        post_reverse = [
            launch.as_plan_entry() for launch in program.post_reverse_launches
        ]
        if (
            index == 1
            and _node_reverse_schedule == _MH1_CUDA_RETAINED_NODE_REVERSE_SCHEDULE
        ):
            post_reverse = [
                {
                    "kernel": "symmetrix_execution_mh1_node_post_reverse_kernel_1",
                    "grid": "persistent_nodes",
                    "threads": 128,
                }
            ]
        nodes.append(
            {
                "index": index,
                "retained_pre_gate_dimension": metadata["layers"][index][
                    "retained_pre_gate_dimension"
                ],
                "retained_interaction_output_dimension": metadata["layers"][index][
                    "retained_interaction_output_dimension"
                ],
                "pre_forward": pre_forward,
                "post_forward": [
                    launch.as_plan_entry() for launch in program.post_forward_launches
                ],
                "post_reverse": post_reverse,
                "pre_reverse": reverse_pre,
            }
        )
    return {
        "schema": MH1_CUDA_LAUNCH_PLAN_TAG,
        "version": 1,
        "artifact_id": metadata["artifact_id"],
        "generation_fingerprint": metadata["generation_fingerprint"],
        "semantic_fingerprint": metadata["semantic_fingerprint"],
        "structure_fingerprint": metadata["structure_fingerprint"],
        "runtime_layout_fingerprint": metadata["runtime_layout_fingerprint"],
        "target_compute_capability": compute_capability,
        "module_identity": _execution_mh1_cuda_module_identity(metadata, precision),
        "scalar_size": _gpu_scalar_size(precision),
        "node_state_policy": metadata["node_state_policy"],
        "node_reverse_schedule": metadata["node_reverse_schedule"],
        "interactions": interactions,
        "nodes": nodes,
    }


def execution_mh1_hip_module_v4_launch_plan(
    contract: dict[str, Any],
    target_metadata: dict[str, Any],
    *,
    forward_policy: str | dict[str, Any] = "auto",
    source_policy: str | dict[str, Any] = "auto",
    edge_policy: str | dict[str, Any] = "auto",
    node_state_policy: str = MH1_NODE_STATE_FULL_RETENTION,
    precision: str = "float32",
) -> dict[str, Any]:
    """Return the closed launch plan for the shared MH1 hipRTC module."""

    target = _mh1_hip_target(target_metadata)
    internal_forward, internal_source, internal_edge = _mh1_hip_internal_policies(
        target, forward_policy, source_policy, edge_policy
    )
    metadata = execution_mh1_hip_module_v4_metadata(
        contract,
        target_metadata,
        forward_policy=forward_policy,
        source_policy=source_policy,
        edge_policy=edge_policy,
        node_state_policy=node_state_policy,
    )
    plan = execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        forward_policy=internal_forward,
        source_policy=internal_source,
        edge_policy=internal_edge,
        node_state_policy=node_state_policy,
        precision=precision,
        _edge_reverse_schedule=_MH1_HIP_EDGE_REVERSE_SCHEDULE,
        _node_reverse_schedule=metadata["node_reverse_schedule"],
    )
    plan.update(
        {
            "schema": MH1_HIP_LAUNCH_PLAN_TAG,
            "target": metadata["target"],
            "target_id": metadata["target_id"],
            "module_identity": _execution_mh1_hip_module_identity(
                metadata, target, precision
            ),
        }
    )
    plan.pop("target_compute_capability")
    return plan


def _render_execution_mh1_gpu_module_v4(
    contract: dict[str, Any],
    compute_capability: int,
    dialect: MH1GpuDialect,
    module_identity: str,
    *,
    forward_policy: str | dict[str, Any] = "auto",
    source_policy: str | dict[str, Any] = "auto",
    edge_policy: str | dict[str, Any] = "auto",
    node_state_policy: str = MH1_NODE_STATE_FULL_RETENTION,
    precision: str = "float32",
    edge_reverse_schedule: str = _MH1_CUDA_EDGE_REVERSE_SCHEDULE,
    node_reverse_schedule: str = _MH1_NODE_REVERSE_SCHEDULE,
) -> str:
    """Render the shared self-contained device-only MH1 ABI-v4 module."""

    normalized = normalize_execution_mh1_v4_contract(contract)
    projected = project_execution_mh1_v3_contract(normalized)
    device_base = _convert_gpu_module_precision(
        _render_execution_mh1_cuda_artifact(
            projected,
            compute_capability,
            forward_policy=forward_policy,
            source_policy=source_policy,
            edge_policy=edge_policy,
            _artifact_kind="module",
            _dialect=dialect,
            _edge_reverse_schedule=edge_reverse_schedule,
        ),
        precision,
    )
    metadata = jit_mh1_cuda_plugin_v4_metadata(
        normalized,
        compute_capability,
        forward_policy=forward_policy,
        source_policy=source_policy,
        edge_policy=edge_policy,
        node_state_policy=node_state_policy,
        _edge_reverse_schedule=edge_reverse_schedule,
        _node_reverse_schedule=node_reverse_schedule,
    )
    embedding = normalized["node_program"]["input"]["embedding"]
    programs = []
    for layer, layer_metadata in zip(
        normalized["node_program"]["layers"], metadata["layers"], strict=True
    ):
        program = _render_cuda_node_program(
            layer,
            layer_metadata,
            embedding,
            return_program=True,
            symbol_prefix="symmetrix_execution_mh1_",
            node_reverse_schedule=node_reverse_schedule,
        )
        programs.append(_convert_gpu_module_precision(program.device_source, precision))
    device_abi = _render_mh1_cuda_device_abi(precision)
    spline_r_programs = _render_mh1_spline_r_gpu_programs(
        normalized, dialect, precision
    )
    source = f"""// Generated by symmetrix.mh1_jit_codegen. Do not edit.
{device_abi}

{device_base}

{spline_r_programs}

{chr(10).join(programs)}

extern "C" __device__ __constant__ char
symmetrix_execution_mh1_module_identity[] =
    {_cpp_string(module_identity)};
"""
    return source


def render_execution_mh1_cuda_module_v4(
    contract: dict[str, Any],
    compute_capability: int,
    *,
    forward_policy: str | dict[str, Any] = "auto",
    source_policy: str | dict[str, Any] = "auto",
    edge_policy: str | dict[str, Any] = "auto",
    node_state_policy: str = MH1_NODE_STATE_FULL_RETENTION,
    precision: str = "float32",
) -> str:
    """Render the shared MH1 ABI-v4 program for NVRTC."""

    node_state_policy = _resolve_node_state_policy(node_state_policy)
    node_reverse_schedule = _cuda_module_node_reverse_schedule(node_state_policy)
    metadata = jit_mh1_cuda_plugin_v4_metadata(
        contract,
        compute_capability,
        forward_policy=forward_policy,
        source_policy=source_policy,
        edge_policy=edge_policy,
        node_state_policy=node_state_policy,
        _node_reverse_schedule=node_reverse_schedule,
    )
    return _render_execution_mh1_gpu_module_v4(
        contract,
        compute_capability,
        MH1_CUDA_DIALECT,
        _execution_mh1_cuda_module_identity(metadata, precision),
        forward_policy=forward_policy,
        source_policy=source_policy,
        edge_policy=edge_policy,
        node_state_policy=node_state_policy,
        precision=precision,
        node_reverse_schedule=node_reverse_schedule,
    )


def render_execution_mh1_hip_module_v4(
    contract: dict[str, Any],
    target_metadata: dict[str, Any],
    *,
    forward_policy: str | dict[str, Any] = "auto",
    source_policy: str | dict[str, Any] = "auto",
    edge_policy: str | dict[str, Any] = "auto",
    node_state_policy: str = MH1_NODE_STATE_FULL_RETENTION,
    precision: str = "float32",
) -> str:
    """Render the shared MH1 ABI-v4 program for hipRTC."""

    target = _mh1_hip_target(target_metadata)
    internal_forward, internal_source, internal_edge = _mh1_hip_internal_policies(
        target, forward_policy, source_policy, edge_policy
    )
    metadata = execution_mh1_hip_module_v4_metadata(
        contract,
        target_metadata,
        forward_policy=forward_policy,
        source_policy=source_policy,
        edge_policy=edge_policy,
        node_state_policy=node_state_policy,
    )
    return _render_execution_mh1_gpu_module_v4(
        contract,
        80,
        MH1_HIP_DIALECT,
        _execution_mh1_hip_module_identity(metadata, target, precision),
        forward_policy=internal_forward,
        source_policy=internal_source,
        edge_policy=internal_edge,
        node_state_policy=node_state_policy,
        precision=precision,
        edge_reverse_schedule=_MH1_HIP_EDGE_REVERSE_SCHEDULE,
        node_reverse_schedule=metadata["node_reverse_schedule"],
    )


__all__ = [
    "MH1_CONDITIONER_LAYOUT_TAG",
    "MH1_CUDA_EDGE_POLICY_TAG",
    "MH1_CUDA_FORWARD_POLICY_TAG",
    "MH1_CUDA_PLUGIN_ABI",
    "MH1_CUDA_PLUGIN_ABI_VERSION",
    "MH1_CUDA_PLUGIN_V4_ABI",
    "MH1_CUDA_PLUGIN_V4_ABI_VERSION",
    "MH1_CUDA_SOURCE_POLICY_TAG",
    "MH1_HOST_PLUGIN_ABI",
    "MH1_HOST_PLUGIN_ABI_VERSION",
    "MH1_HOST_PLUGIN_V4_ABI",
    "MH1_HOST_PLUGIN_V4_ABI_VERSION",
    "MH1_HOST_PLUGIN_V5_ABI",
    "MH1_HOST_PLUGIN_V5_ABI_VERSION",
    "MH1_NODE_PROGRAM_METADATA_TAG",
    "MH1_NODE_STATE_FULL_RETENTION",
    "MH1_NODE_STATE_RECOMPUTE",
    "MH1_NODE_STATE_REUSE_ADJOINTS",
    "MH1_NODE_STATE_RETAIN_INTERACTION",
    "MH1_SOURCE_POLICY_TAG",
    "MH1CudaKernelLaunch",
    "MH1KernelLaunch",
    "MH1KernelSchedule",
    "MH1NodeCudaProgram",
    "MH1NodeProgram",
    "MH1CudaSchedule",
    "MH1GpuTarget",
    "MH1Program",
    "render_execution_mh1_conditioner_cpp_helpers",
    "render_execution_mh1_cuda_module_v4",
    "render_execution_mh1_hip_module_v4",
    "render_jit_mh1_cuda_plugin",
    "render_jit_mh1_cuda_plugin_v4",
    "render_jit_mh1_host_plugin",
    "render_jit_mh1_host_plugin_v4",
    "render_jit_mh1_host_plugin_v5",
    "render_execution_mh1_node_linear_cpp_helpers",
    "render_execution_mh1_node_nonlinear_cpp_helpers",
    "render_execution_mh1_node_product_cpp_helpers",
    "render_execution_mh1_node_program_cpp_helpers",
    "resolve_execution_mh1_cuda_edge_policy",
    "resolve_execution_mh1_cuda_forward_policy",
    "resolve_execution_mh1_cuda_source_policy",
    "execution_mh1_conditioner_layout_metadata",
    "execution_mh1_cuda_module_v4_launch_plan",
    "execution_mh1_hip_module_v4_launch_plan",
    "execution_mh1_hip_module_v4_metadata",
    "jit_mh1_cuda_plugin_metadata",
    "jit_mh1_cuda_plugin_v4_metadata",
    "jit_mh1_host_plugin_metadata",
    "execution_mh1_node_program_metadata",
]
