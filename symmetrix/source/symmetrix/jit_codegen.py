"""Standalone host, CUDA, and HIP code generation for Execution specialization."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .execution_contract import normalize_jit_r1_contract
from .jit import JIT_GENERATION_VERSION

_HOST_R1_SOURCE_CHANNEL_TILE = 32
_HOST_R1_FORWARD_CHANNEL_TILE = 16
_HOST_R1_SOURCE_REVERSE_PREFETCH_DISTANCE = 2

HOST_PLUGIN_ABI = "symmetrix.jit.host-plugin/2"
HOST_PLUGIN_ABI_VERSION = 2
CUDA_PLUGIN_ABI = "symmetrix.jit.cuda-plugin/2"
CUDA_PLUGIN_ABI_VERSION = 2
HIP_PLUGIN_ABI = "symmetrix.jit.hip-plugin/1"
HIP_PLUGIN_ABI_VERSION = 1
GPU_CODEGEN_IDENTITY_SCHEMA_VERSION = 1
_HOST_PLUGIN_ABI_HEADER = (
    Path(__file__).with_name("jit_host_plugin_abi.h").read_text(encoding="ascii")
)
_CUDA_PLUGIN_ABI_HEADER = (
    Path(__file__)
    .with_name("jit_cuda_plugin_abi.h")
    .read_text(encoding="ascii")
    .removeprefix("#pragma once\n\n")
)
_HIP_PLUGIN_ABI_HEADER = (
    Path(__file__)
    .with_name("jit_hip_plugin_abi.h")
    .read_text(encoding="ascii")
    .removeprefix("#pragma once\n\n")
)


def _render_module_packet_header(prefix: str, version: int, compiler: str) -> str:
    return f"""// Self-contained {compiler} packet ABI.
namespace std {{
using int32_t = int;
using int64_t = long long;
using size_t = decltype(sizeof(0));
}}

struct {prefix}RadialSplineV{version} {{
    unsigned int struct_size;
    unsigned int edge_types;
    unsigned int intervals;
    unsigned int functions;
    double h;
    double x0;
    const void* coefficients;
}};

struct {prefix}R1ForwardArgsV{version} {{
    unsigned int struct_size;
    unsigned int active_type_count;
    long long num_nodes;
    long long num_edges;
    const int* node_types;
    const int* num_neigh;
    const int* first_neigh;
    const int* neigh_indices;
    const int* neigh_types;
    const int* type_to_active;
    const double* radius;
    {prefix}RadialSplineV{version} radial;
    const void* harmonics_values;
    const void* neighbor_features;
    void* output;
    double cutoff;
}};

struct {prefix}R1SourceArgsV{version} {{
    unsigned int struct_size;
    unsigned int active_type_count;
    long long num_nodes;
    long long num_edges;
    const int* node_types;
    const int* neigh_types;
    const int* source_offsets;
    const int* source_edges;
    const int* edge_receivers;
    const int* type_to_active;
    const double* radius;
    {prefix}RadialSplineV{version} radial;
    const void* harmonics_values;
    const void* output_adjoint;
    void* source_adjoint;
    double cutoff;
}};

struct {prefix}R1TiledForwardArgsV{version} {{
    unsigned int struct_size;
    unsigned int active_type_count;
    unsigned int channel_begin;
    unsigned int channel_count;
    long long num_nodes;
    long long num_edges;
    const int* node_types;
    const int* num_neigh;
    const int* first_neigh;
    const int* neigh_indices;
    const int* neigh_types;
    const int* type_to_active;
    const double* radius;
    {prefix}RadialSplineV{version} radial;
    const void* harmonics_values;
    const void* neighbor_features;
    void* output;
    double cutoff;
}};

struct {prefix}R1TiledSourceArgsV{version} {{
    unsigned int struct_size;
    unsigned int active_type_count;
    unsigned int channel_begin;
    unsigned int channel_count;
    long long num_nodes;
    long long num_edges;
    const int* node_types;
    const int* neigh_types;
    const int* source_offsets;
    const int* source_edges;
    const int* edge_receivers;
    const int* type_to_active;
    const double* radius;
    {prefix}RadialSplineV{version} radial;
    const void* harmonics_values;
    const void* output_adjoint;
    void* source_adjoint;
    double cutoff;
    long long source_owner_count;
    const int* source_ids;
}};

struct {prefix}R1TiledEdgeArgsV{version} {{
    unsigned int struct_size;
    unsigned int active_type_count;
    unsigned int coordinate_scalar_size;
    unsigned int coordinates_are_unit;
    long long num_nodes;
    long long num_edges;
    const int* node_types;
    const int* neigh_indices;
    const int* neigh_types;
    const int* edge_receivers;
    const int* type_to_active;
    const void* xyz;
    const double* radius;
    {prefix}RadialSplineV{version} radial;
    const void* harmonics_values;
    const void* harmonics_gradients;
    const void* neighbor_features;
    const void* output_adjoint;
    double* directed_forces;
    double cutoff;
}};

struct {prefix}R1ProjectedForwardArgsV{version} {{
    unsigned int struct_size;
    unsigned int active_type_count;
    long long num_nodes;
    long long num_edges;
    const int* node_types;
    const int* num_neigh;
    const int* first_neigh;
    const int* neigh_indices;
    const int* neigh_types;
    const int* type_to_active;
    const double* radius;
    {prefix}RadialSplineV{version} radial;
    const void* harmonics_values;
    const void* neighbor_features;
    const void* projection_weights;
    void* output;
    double cutoff;
}};

struct {prefix}R1ProjectedReverseArgsV{version} {{
    unsigned int struct_size;
    unsigned int active_type_count;
    unsigned int coordinate_scalar_size;
    unsigned int coordinates_are_unit;
    long long num_nodes;
    long long num_edges;
    const int* node_types;
    const int* num_neigh;
    const int* first_neigh;
    const int* neigh_indices;
    const int* neigh_types;
    const int* type_to_active;
    const void* xyz;
    const double* radius;
    {prefix}RadialSplineV{version} radial;
    const void* harmonics_values;
    const void* harmonics_gradients;
    const void* neighbor_features;
    const void* output_adjoint;
    void* source_adjoint;
    double* directed_forces;
    const void* projection_weights;
    double cutoff;
}};

struct {prefix}R1EdgeArgsV{version} {{
    unsigned int struct_size;
    unsigned int active_type_count;
    unsigned int coordinate_scalar_size;
    unsigned int coordinates_are_unit;
    long long num_nodes;
    long long num_edges;
    const int* node_types;
    const int* neigh_indices;
    const int* neigh_types;
    const int* edge_receivers;
    const int* type_to_active;
    const void* xyz;
    const double* radius;
    {prefix}RadialSplineV{version} radial;
    const void* harmonics_values;
    const void* harmonics_gradients;
    const void* neighbor_features;
    const void* output_adjoint;
    double* directed_forces;
    double cutoff;
}};"""


_CUDA_MODULE_PACKET_HEADER = _render_module_packet_header(
    "SymmetrixJitCuda", 2, "NVRTC"
)
_HIP_MODULE_PACKET_HEADER = _render_module_packet_header("SymmetrixJitHip", 1, "hipRTC")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class R1Program:
    """Immutable, backend-neutral description of an ordinary R1 program."""

    schema_version: int
    contract_json: str
    precision: Literal["float32", "float64"]
    packet_layout_version: int
    kernel_phases: tuple[str, ...] = ("forward", "source", "edge")

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported R1 program schema version")
        _precision_spec(self.precision)
        if self.packet_layout_version not in (1, 2):
            raise ValueError("R1 packet layout version must be 1 or 2")
        if self.kernel_phases != ("forward", "source", "edge"):
            raise ValueError("ordinary R1 kernel phases must be forward/source/edge")
        try:
            contract = json.loads(self.contract_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("R1 contract JSON must be valid") from exc
        normalized = normalize_jit_r1_contract(contract)
        if self.contract_json != _canonical_json(normalized):
            raise ValueError("R1 contract JSON must use canonical serialization")

    @classmethod
    def from_contract(
        cls, contract: dict, precision: str, *, packet_layout_version: int
    ) -> R1Program:
        _precision_spec(precision)
        normalized = normalize_jit_r1_contract(contract)
        _contract_groups(normalized)
        return cls(
            schema_version=1,
            contract_json=_canonical_json(normalized),
            precision=precision,
            packet_layout_version=packet_layout_version,
        )

    @property
    def contract(self) -> dict:
        return json.loads(self.contract_json)

    def canonical_json(self) -> str:
        return _canonical_json(asdict(self))


@dataclass(frozen=True)
class GpuTarget:
    schema_version: int
    backend: Literal["cuda", "hip"]
    architecture: str
    target_features: str
    native_subgroup_width: int
    compiler_target: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported GPU target schema version")
        if self.backend == "cuda":
            if not re.fullmatch(r"sm_[1-9][0-9]{0,2}", self.architecture):
                raise ValueError("CUDA architecture must look like 'sm_NN'")
            if self.target_features:
                raise ValueError("CUDA target features are not supported")
            if not re.fullmatch(r"compute_[1-9][0-9]{0,2}", self.compiler_target):
                raise ValueError("CUDA compiler target must look like 'compute_NN'")
            if self.architecture.removeprefix(
                "sm_"
            ) != self.compiler_target.removeprefix("compute_"):
                raise ValueError("CUDA architecture and compiler target disagree")
        elif self.backend == "hip":
            if not re.fullmatch(r"gfx[0-9a-f]+", self.architecture):
                raise ValueError("HIP architecture must look like 'gfxNNN'")
            features = [item for item in self.target_features.split(":") if item]
            if any(not re.fullmatch(r"[a-z][a-z0-9_]*[+-]", item) for item in features):
                raise ValueError(
                    "HIP target feature must use a name followed by + or -"
                )
            compiler_parts = self.compiler_target.split(":")
            if compiler_parts[0] != self.architecture or sorted(
                compiler_parts[1:]
            ) != sorted(features):
                raise ValueError("HIP compiler target and target features disagree")
            normalized_features = ":".join(sorted(set(features)))
            object.__setattr__(self, "target_features", normalized_features)
            object.__setattr__(
                self,
                "compiler_target",
                ":".join((self.architecture, *sorted(set(features)))),
            )
        else:
            raise ValueError("GPU backend must be 'cuda' or 'hip'")
        if (
            isinstance(self.native_subgroup_width, bool)
            or self.native_subgroup_width <= 0
            or self.native_subgroup_width & (self.native_subgroup_width - 1)
        ):
            raise ValueError("native subgroup width must be a positive power of two")

    def canonical_json(self) -> str:
        return _canonical_json(asdict(self))


@dataclass(frozen=True)
class KernelSchedule:
    schema_version: int
    edge_strategy: Literal["serial", "wave"]
    edge_logical_subgroup_width: int
    forward_threads_per_block: int
    source_threads_per_block: int
    edge_threads_per_block: int
    persistent_blocks_per_compute_unit: int
    dynamic_shared_memory: int = 0

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported kernel schedule schema version")
        if self.edge_strategy not in ("serial", "wave"):
            raise ValueError("edge strategy must be 'serial' or 'wave'")
        widths = (
            self.forward_threads_per_block,
            self.source_threads_per_block,
            self.edge_threads_per_block,
        )
        if any(
            isinstance(width, bool) or width < 1 or width > 1024 or width & (width - 1)
            for width in widths
        ):
            raise ValueError("thread block sizes must be powers of two in [1, 1024]")
        logical = self.edge_logical_subgroup_width
        if (
            isinstance(logical, bool)
            or logical < 1
            or logical > 64
            or logical & (logical - 1)
        ):
            raise ValueError("logical subgroup width must be a power of two in [1, 64]")
        if self.edge_strategy == "serial" and logical != 1:
            raise ValueError("serial edge schedules require logical subgroup width 1")
        if self.edge_strategy == "wave" and self.edge_threads_per_block % logical:
            raise ValueError("edge block size must be a logical subgroup multiple")
        if not 1 <= self.persistent_blocks_per_compute_unit <= 32:
            raise ValueError("persistent blocks per compute unit must be in [1, 32]")
        if self.dynamic_shared_memory < 0:
            raise ValueError("dynamic shared memory must be non-negative")

    def canonical_json(self) -> str:
        return _canonical_json(asdict(self))


@dataclass(frozen=True)
class LaunchPlan:
    schema_version: int
    exported_symbols: tuple[str, str, str]
    fused_reverse_symbol: str
    ordered_launches: tuple[str, str] = ("forward", "reverse_fused")
    dynamic_shared_memory: int = 0

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("unsupported launch-plan schema version")
        symbols = (*self.exported_symbols, self.fused_reverse_symbol)
        if len(set(symbols)) != 4 or any(
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", symbol) for symbol in symbols
        ):
            raise ValueError("launch plan requires four distinct C symbols")
        if self.ordered_launches != ("forward", "reverse_fused"):
            raise ValueError("fused R1 launch order must be forward/reverse_fused")
        if self.dynamic_shared_memory < 0:
            raise ValueError("dynamic shared memory must be non-negative")

    def canonical_json(self) -> str:
        return _canonical_json(asdict(self))


@dataclass(frozen=True)
class GpuDialect:
    backend: Literal["cuda", "hip"]
    packet_prefix: str
    packet_version: int
    plugin_abi: str
    plugin_abi_version: int
    runtime_header: str
    error_prefix: str
    query_symbol: str
    export_macro: str
    shuffle_down_template: str

    def shuffle_down(self, value: str, offset: str, width: str) -> str:
        return self.shuffle_down_template.format(
            value=value, offset=offset, width=width
        )


CUDA_DIALECT = GpuDialect(
    "cuda",
    "SymmetrixJitCuda",
    2,
    CUDA_PLUGIN_ABI,
    2,
    "cuda_runtime.h",
    "cuda",
    "symmetrix_jit_cuda_plugin_query_v2",
    "SYMMETRIX_JIT_CUDA_PLUGIN_EXPORT",
    "__shfl_down_sync(0xffffffffu, {value}, {offset}, {width})",
)
HIP_DIALECT = GpuDialect(
    "hip",
    "SymmetrixJitHip",
    1,
    HIP_PLUGIN_ABI,
    1,
    "hip/hip_runtime.h",
    "hip",
    "symmetrix_jit_hip_plugin_query_v1",
    "SYMMETRIX_JIT_HIP_PLUGIN_EXPORT",
    "__shfl_down({value}, {offset}, {width})",
)


def gpu_codegen_identity(
    program: R1Program,
    target: GpuTarget,
    schedule: KernelSchedule,
    launch_plan: LaunchPlan,
) -> str:
    """Return the canonical cache-key input for one generated GPU artifact."""

    if target.backend == "cuda" and program.packet_layout_version != 2:
        raise ValueError("CUDA R1 programs require packet layout version 2")
    if target.backend == "hip" and program.packet_layout_version != 1:
        raise ValueError("HIP R1 programs require packet layout version 1")
    if schedule.edge_strategy == "wave" and (
        schedule.edge_logical_subgroup_width > target.native_subgroup_width
        or target.native_subgroup_width % schedule.edge_logical_subgroup_width
    ):
        raise ValueError("logical subgroup must divide the native subgroup")
    if schedule.dynamic_shared_memory != launch_plan.dynamic_shared_memory:
        raise ValueError("schedule and launch-plan shared memory disagree")
    return _canonical_json(
        {
            "schema_version": GPU_CODEGEN_IDENTITY_SCHEMA_VERSION,
            "program": json.loads(program.canonical_json()),
            "target": json.loads(target.canonical_json()),
            "schedule": json.loads(schedule.canonical_json()),
            "launch_plan": json.loads(launch_plan.canonical_json()),
        }
    )


def _gpu_launch_plan(dialect: GpuDialect, artifact_kind: str) -> LaunchPlan:
    version = dialect.packet_version
    if artifact_kind == "module":
        symbols = tuple(
            f"symmetrix_factorized_{phase}_v{version}"
            for phase in ("forward", "source", "edge")
        )
        fused_reverse_symbol = f"symmetrix_factorized_reverse_fused_v{version}"
    elif artifact_kind == "plugin":
        symbols = tuple(f"r1_{phase}_kernel" for phase in ("forward", "source", "edge"))
        fused_reverse_symbol = "r1_reverse_fused_kernel"
    else:
        raise ValueError("GPU artifact kind must be 'plugin' or 'module'")
    return LaunchPlan(2, symbols, fused_reverse_symbol)


def _cuda_codegen_inputs(
    contract: dict,
    compute_capability: int,
    precision: str,
    artifact_kind: str,
    edge_strategy: str = "wave",
    edge_logical_subgroup_width: int | None = 32,
    edge_threads_per_block: int = 32,
    persistent_blocks_per_compute_unit: int = 8,
) -> tuple[R1Program, GpuTarget, KernelSchedule, LaunchPlan]:
    program = R1Program.from_contract(
        contract, precision, packet_layout_version=CUDA_DIALECT.packet_version
    )
    target = GpuTarget(
        1,
        "cuda",
        f"sm_{compute_capability}",
        "",
        32,
        f"compute_{compute_capability}",
    )
    if edge_strategy == "serial" and edge_logical_subgroup_width not in (None, 1):
        raise ValueError("serial edge schedules require logical subgroup width 1")
    logical_width = (
        1
        if edge_strategy == "serial"
        else 32
        if edge_logical_subgroup_width is None
        else edge_logical_subgroup_width
    )
    schedule = KernelSchedule(
        1,
        edge_strategy,
        logical_width,
        256,
        256,
        edge_threads_per_block,
        persistent_blocks_per_compute_unit,
    )
    return program, target, schedule, _gpu_launch_plan(CUDA_DIALECT, artifact_kind)


def _hip_codegen_inputs(
    contract: dict,
    target_metadata: dict,
    precision: str,
    artifact_kind: str,
    edge_strategy: str,
    edge_threads_per_block: int,
    persistent_blocks_per_compute_unit: int,
) -> tuple[R1Program, GpuTarget, KernelSchedule, LaunchPlan]:
    program = R1Program.from_contract(
        contract, precision, packet_layout_version=HIP_DIALECT.packet_version
    )
    target = GpuTarget(
        1,
        "hip",
        target_metadata["target_architecture"],
        target_metadata["target_features"],
        target_metadata["native_subgroup_width"],
        target_metadata["raw_agent_target"],
    )
    logical_width = (
        target_metadata["native_subgroup_width"] if edge_strategy == "wave" else 1
    )
    schedule = KernelSchedule(
        1,
        edge_strategy,
        logical_width,
        256,
        256,
        edge_threads_per_block,
        persistent_blocks_per_compute_unit,
    )
    return program, target, schedule, _gpu_launch_plan(HIP_DIALECT, artifact_kind)


def factorized_gpu_codegen_identity(
    contract: dict,
    backend: Literal["cuda", "hip"],
    target: int | str | dict,
    *,
    precision: str = "float32",
    artifact_kind: Literal["plugin", "module"] = "plugin",
    edge_strategy: str | None = None,
    edge_logical_subgroup_width: int | None = None,
    edge_threads_per_block: int | None = None,
    persistent_blocks_per_compute_unit: int | None = None,
) -> dict:
    """Return the complete structured identity used by an R1 GPU artifact."""

    if backend == "cuda":
        if not isinstance(target, int) or isinstance(target, bool):
            raise ValueError("CUDA codegen identity requires an integer target")
        inputs = _cuda_codegen_inputs(
            contract,
            target,
            precision,
            artifact_kind,
            "wave" if edge_strategy is None else edge_strategy,
            edge_logical_subgroup_width,
            32 if edge_threads_per_block is None else edge_threads_per_block,
            8
            if persistent_blocks_per_compute_unit is None
            else persistent_blocks_per_compute_unit,
        )
    elif backend == "hip":
        if edge_logical_subgroup_width is not None:
            raise ValueError("HIP logical subgroup width is selected by the target")
        target_metadata = _hip_target_metadata(target)
        inputs = _hip_codegen_inputs(
            contract,
            target_metadata,
            precision,
            artifact_kind,
            "wave" if edge_strategy is None else edge_strategy,
            64 if edge_threads_per_block is None else edge_threads_per_block,
            8
            if persistent_blocks_per_compute_unit is None
            else persistent_blocks_per_compute_unit,
        )
    else:
        raise ValueError("GPU backend must be 'cuda' or 'hip'")
    return json.loads(gpu_codegen_identity(*inputs))


def _contract_groups(contract: dict) -> tuple[tuple[int, int, int], ...]:
    channels = contract["channels"]
    for path in contract["paths"]:
        if path["connection_mode"] != "uvu" or path["path_shape"] != [channels, 1]:
            raise ValueError(
                "Execution R1 specialization requires uvu paths with shape [channels, 1]"
            )
    return tuple(
        (group["l"], group["components"], len(group["path_indices"]))
        for group in contract["groups"]
    )


def _fixed_weight_path_rows(contract: dict) -> tuple[tuple[dict, ...], ...]:
    path_offsets = [0]
    for path in contract["paths"]:
        path_offsets.append(
            path_offsets[-1] + (2 * path["edge_l"] + 1) * (2 * path["source_l"] + 1)
        )

    terms_by_row: dict[int, dict] = {}
    for term in contract["sparse_coupling"]["terms"]:
        row = term["row"]
        path_index = next(
            (
                index
                for index in range(len(contract["paths"]))
                if path_offsets[index] <= row < path_offsets[index + 1]
            ),
            None,
        )
        if path_index is None:
            raise ValueError(
                f"sparse coupling row {row} is outside the fixed-weight path layout"
            )
        entry = terms_by_row.setdefault(
            row,
            {
                "row": row,
                "path": path_index,
                "lm1": term["lm1"],
                "lm2": term["lm2"],
                "terms": [],
            },
        )
        if (
            entry["path"] != path_index
            or entry["lm1"] != term["lm1"]
            or entry["lm2"] != term["lm2"]
        ):
            raise ValueError(
                f"sparse coupling row {row} does not have one angular layout"
            )
        entry["terms"].append(term)

    return tuple(
        tuple(
            terms_by_row[row]
            for row in sorted(terms_by_row)
            if path_offsets[path_index] <= row < path_offsets[path_index + 1]
        )
        for path_index in range(len(contract["paths"]))
    )


def _cpp_string(value: str) -> str:
    return json.dumps(value)


def _cpp_float(value: float) -> str:
    return f"float({value!r})"


def _cpp_scalar(value: float, scalar_type: str) -> str:
    return f"{scalar_type}({value!r})"


def _scalar_pointer(field: str, scalar_type: str, *, opaque: bool) -> str:
    if opaque:
        return f"static_cast<const {scalar_type}*>({field})"
    return field


def _mutable_scalar_pointer(field: str, scalar_type: str, *, opaque: bool) -> str:
    if opaque:
        return f"static_cast<{scalar_type}*>({field})"
    return field


def _precision_spec(precision: str) -> dict:
    if precision == "float32":
        return {
            "cpp": "float",
            "suffix": "f32",
            "scalar_kind": 1,
            "scalar_size": 4,
            "host_scalar_macro": "SYMMETRIX_JIT_HOST_SCALAR_FLOAT32_V2",
            "cuda_scalar_macro": "SYMMETRIX_JIT_CUDA_SCALAR_FLOAT32_V2",
        }
    if precision == "float64":
        return {
            "cpp": "double",
            "suffix": "f64",
            "scalar_kind": 2,
            "scalar_size": 8,
            "host_scalar_macro": "SYMMETRIX_JIT_HOST_SCALAR_FLOAT64_V2",
            "cuda_scalar_macro": "SYMMETRIX_JIT_CUDA_SCALAR_FLOAT64_V2",
        }
    raise ValueError("Execution plugin precision must be 'float32' or 'float64'")


def _render_precision_v2_source(source: str, backend: str, precision: str) -> str:
    spec = _precision_spec(precision)
    marker = "#include <cmath>"
    header, body = source.split(marker, 1)
    body = marker + body
    prefix = "Host" if backend == "host" else "Cuda"
    macro_prefix = "HOST" if backend == "host" else "CUDA"
    body = body.replace(
        f"SymmetrixJit{prefix}RadialSplineV1",
        f"SymmetrixJit{prefix}RadialSplineV2",
    )
    for packet in ("ForwardArgs", "SourceArgs", "EdgeArgs", "Plugin"):
        body = body.replace(
            f"SymmetrixJit{prefix}R1{packet}V1"
            if packet != "Plugin"
            else f"SymmetrixJit{prefix}PluginV1",
            f"SymmetrixJit{prefix}R1{packet}V2"
            if packet != "Plugin"
            else f"SymmetrixJit{prefix}PluginV2",
        )
    body = body.replace(
        f"_{macro_prefix}_R1_FORWARD_OWNER_V1", f"_{macro_prefix}_R1_FORWARD_OWNER_V2"
    )
    body = body.replace(
        f"_{macro_prefix}_R1_SOURCE_OWNER_V1", f"_{macro_prefix}_R1_SOURCE_OWNER_V2"
    )
    body = body.replace(
        f"_{macro_prefix}_R1_COMPENSATED_SOURCE_OWNER_V1",
        f"_{macro_prefix}_R1_COMPENSATED_SOURCE_OWNER_V2",
    )
    body = body.replace(
        f"_{macro_prefix}_R1_EDGE_OWNER_V1", f"_{macro_prefix}_R1_EDGE_OWNER_V2"
    )
    body = body.replace(
        f"_{macro_prefix}_R1_FORWARD_LAUNCH_V1", f"_{macro_prefix}_R1_FORWARD_LAUNCH_V2"
    )
    body = body.replace(
        f"_{macro_prefix}_R1_COORDINATE_REVERSE_LAUNCH_V1",
        f"_{macro_prefix}_R1_COORDINATE_REVERSE_LAUNCH_V2",
    )
    body = body.replace(
        f"SYMMETRIX_JIT_{macro_prefix}_PLUGIN_ABI_VERSION,",
        f"SYMMETRIX_JIT_{macro_prefix}_PLUGIN_ABI_VERSION_V2,",
    )
    body = body.replace(
        f"symmetrix_jit_{backend}_plugin_query_v1",
        f"symmetrix_jit_{backend}_plugin_query_v2",
    )

    pointer_fields = {
        "radial.coefficients[": "static_cast<const Scalar*>(radial.coefficients)[",
        "args->output_adjoint[": "static_cast<const Scalar*>(args->output_adjoint)[",
        "args->harmonics_values[": "static_cast<const Scalar*>(args->harmonics_values)[",
        "args->neighbor_features[": "static_cast<const Scalar*>(args->neighbor_features)[",
        "args->harmonics_gradients[": "static_cast<const Scalar*>(args->harmonics_gradients)[",
        "args->output[": "static_cast<Scalar*>(args->output)[",
        "args->source_adjoint[": "static_cast<Scalar*>(args->source_adjoint)[",
    }
    for old, new in pointer_fields.items():
        body = body.replace(old, new)
    body = re.sub(r"\bfloat\b", "Scalar", body)
    body = body.replace(
        "namespace {\n", f"namespace {{\n\nusing Scalar = {spec['cpp']};\n", 1
    )

    if backend == "host":
        untiled_capability_tail = "        | SYMMETRIX_JIT_HOST_R1_EDGE_OWNER_V2,\n"
        capability_tail = """        | SYMMETRIX_JIT_HOST_R1_EDGE_OWNER_V2
        | SYMMETRIX_JIT_HOST_R1_SOURCE_CHANNEL_TILE_32_V2
        | SYMMETRIX_JIT_HOST_R1_FORWARD_CHANNEL_TILE_16_V2,
"""
        body = body.replace(
            untiled_capability_tail,
            capability_tail,
            1,
        )
        scalar_macro = spec["host_scalar_macro"]
    else:
        capability_tail = (
            "        | SYMMETRIX_JIT_CUDA_R1_COORDINATE_REVERSE_LAUNCH_V2,\n"
        )
        scalar_macro = spec["cuda_scalar_macro"]
    body = body.replace(
        capability_tail + "    0u,\n",
        capability_tail
        + f"    {scalar_macro},\n"
        + "    sizeof(Scalar),\n"
        + "    0u,\n",
        1,
    )
    return header + body


def _render_row_loaders(
    path_rows,
    channels: int,
    output_components: int,
    *,
    source_args_type: str = "SymmetrixJitHostR1SourceArgsV1",
    edge_args_type: str = "SymmetrixJitHostR1EdgeArgsV1",
    qualifier: str = "inline",
    scalar_type: str = "float",
    opaque_pointers: bool = False,
) -> str:
    output_adjoint = _scalar_pointer(
        "args->output_adjoint", scalar_type, opaque=opaque_pointers
    )
    loaders = []
    for rows in path_rows:
        for row in rows:
            terms = "\n        + ".join(
                f"{_cpp_scalar(term['coefficient'], scalar_type)} * {output_adjoint}["
                f"(static_cast<std::size_t>(receiver) * {output_components} + {term['lme']}) * {channels} + channel]"
                for term in row["terms"]
            )
            loaders.append(
                f"""{qualifier} {scalar_type} load_row_{row["row"]}(
    const {source_args_type}* args,
    std::int32_t receiver,
    std::int32_t channel)
{{
    return {terms};
}}

{qualifier} {scalar_type} load_row_{row["row"]}(
    const {edge_args_type}* args,
    std::int32_t receiver,
    std::int32_t channel)
{{
    return {terms};
}}"""
            )
    return "\n\n".join(loaders)


def _render_forward_owner(
    path_rows,
    *,
    channels: int,
    edge_harmonics: int,
    source_harmonics: int,
    output_components: int,
    args_type: str = "SymmetrixJitHostR1ForwardArgsV1",
    qualifier: str = "",
    scalar_type: str = "float",
    opaque_pointers: bool = False,
    ordered_pair_types: bool = False,
    map_type_indices: bool = True,
    density_function: int | None = None,
    radial_function_offsets: tuple[int, ...] | None = None,
    use_output_mask: bool = False,
    function_name: str = "r1_forward_owner",
    output_channel_stride: str | None = None,
    output_channel_index: str = "channel",
    filter_inactive_edges: bool = True,
) -> str:
    harmonics = _scalar_pointer(
        "args->harmonics_values", scalar_type, opaque=opaque_pointers
    )
    neighbors = _scalar_pointer(
        "args->neighbor_features", scalar_type, opaque=opaque_pointers
    )
    output_mask = _scalar_pointer(
        "args->output_mask", scalar_type, opaque=opaque_pointers
    )
    node_density = _mutable_scalar_pointer(
        "args->node_density", scalar_type, opaque=opaque_pointers
    )
    declarations = "\n".join(
        f"    {scalar_type} value_{lme} = {scalar_type}(0);"
        for lme in range(output_components)
    )
    if density_function is not None:
        declarations += f"\n    {scalar_type} density = {scalar_type}(0);"
    blocks = []
    for path_index, rows in enumerate(path_rows):
        radial_offset = (
            path_index * channels
            if radial_function_offsets is None
            else radial_function_offsets[path_index]
        )
        statements = [
            f"        const {scalar_type} radial_{path_index} = evaluate_radial(",
            f"            args->radial, edge_type, point, {radial_offset} + channel);",
        ]
        for row in rows:
            statements.extend(
                [
                    f"        const {scalar_type} product_{row['row']} = radial_{path_index}",
                    f"            * {harmonics}[edge_offset + {row['lm1']}]",
                    f"            * {neighbors}[(source_offset + {row['lm2']}) * {channels} + channel];",
                ]
            )
            for term in row["terms"]:
                mask = (
                    f" * {output_mask}[{term['lme']} * {channels} + channel]"
                    if use_output_mask
                    else ""
                )
                statements.append(
                    f"        value_{term['lme']} += {_cpp_scalar(term['coefficient'], scalar_type)}"
                    f" * product_{row['row']}{mask};"
                )
        blocks.append("\n".join(statements))
    output_stride = output_channel_stride or str(channels)
    stores = "\n".join(
        f"    output[{lme} * ({output_stride}) + ({output_channel_index})] = value_{lme};"
        for lme in range(output_components)
    )
    receiver_type = (
        "args->type_to_active[args->node_types[receiver]]"
        if map_type_indices
        else "args->node_types[receiver]"
    )
    source_type = (
        "args->type_to_active[args->neigh_types[edge]]"
        if map_type_indices
        else "args->neigh_types[edge]"
    )
    edge_type = (
        "source_type * static_cast<std::int32_t>(args->active_type_count)"
        " + receiver_type"
        if ordered_pair_types
        else "pair_type(\n            receiver_type, source_type,\n"
        "            static_cast<std::int32_t>(args->active_type_count))"
    )
    density_update = ""
    density_store = ""
    if density_function is not None:
        density_update = (
            "\n        if (channel == 0)\n"
            "            density += evaluate_radial(\n"
            f"                args->radial, edge_type, point, {density_function});"
        )
        density_store = f"\n    if (channel == 0) {node_density}[receiver] += density;"
    function_prefix = f"{qualifier} " if qualifier else ""
    active_edge_guard = (
        "        if (!edge_is_active(args->cutoff, args->radius[edge]))\n"
        "            continue;\n"
        if filter_inactive_edges
        else ""
    )
    return f"""{function_prefix}void {function_name}(
    const {args_type}* args,
    std::int32_t receiver,
    std::int32_t channel,
    {scalar_type}* output)
{{
{declarations}
    const std::int32_t receiver_type = {receiver_type};
    const std::int32_t edge_begin = args->first_neigh[receiver];
    const std::int32_t edge_end = edge_begin + args->num_neigh[receiver];
    for (std::int32_t edge = edge_begin; edge < edge_end; ++edge) {{
{active_edge_guard.rstrip()}
        const std::int32_t source = args->neigh_indices[edge];
        const std::size_t edge_offset =
            static_cast<std::size_t>(edge) * {edge_harmonics};
        const std::size_t source_offset =
            static_cast<std::size_t>(source) * {source_harmonics};
        const std::int32_t source_type = {source_type};
        const std::int32_t edge_type = {edge_type};
        const EvaluationPoint point = evaluation_point(
            args->radial, args->radius[edge]);{density_update}
{chr(10).join(blocks)}
    }}
{stores}{density_store}
}}"""


def _render_source_owner(
    path_rows,
    *,
    channels: int,
    edge_harmonics: int,
    source_harmonics: int,
    args_type: str = "SymmetrixJitHostR1SourceArgsV1",
    qualifier: str = "",
    scalar_type: str = "float",
    opaque_pointers: bool = False,
) -> str:
    harmonics = _scalar_pointer(
        "args->harmonics_values", scalar_type, opaque=opaque_pointers
    )
    source_adjoint = _mutable_scalar_pointer(
        "args->source_adjoint", scalar_type, opaque=opaque_pointers
    )
    blocks = []
    for path_index, rows in enumerate(path_rows):
        statements = [
            f"        const {scalar_type} radial_{path_index} = evaluate_radial(",
            f"            args->radial, edge_type, point, {path_index} * {channels} + channel);",
        ]
        for row in rows:
            statements.extend(
                [
                    f"        const {scalar_type} contribution_{row['row']} = radial_{path_index}",
                    f"            * {harmonics}[edge_offset + {row['lm1']}]",
                    f"            * load_row_{row['row']}(args, receiver, channel);",
                    "        if constexpr (Compensated) {",
                    f"            const {scalar_type} corrected_{row['row']} = contribution_{row['row']} - compensation[{row['lm2']}];",
                    f"            const {scalar_type} updated_{row['row']} = values[{row['lm2']}] + corrected_{row['row']};",
                    f"            compensation[{row['lm2']}] = (updated_{row['row']} - values[{row['lm2']}]) - corrected_{row['row']};",
                    f"            values[{row['lm2']}] = updated_{row['row']};",
                    "        } else {",
                    f"            values[{row['lm2']}] += contribution_{row['row']};",
                    "        }",
                ]
            )
        blocks.append("\n".join(statements))
    stores = "\n".join(
        f"    {source_adjoint}[(source_offset + {lm}) * {channels} + channel] += values[{lm}];"
        for lm in range(source_harmonics)
    )
    function_prefix = f"{qualifier} " if qualifier else ""
    return f"""template <bool Compensated>
{function_prefix}void r1_source_owner_impl(
    const {args_type}* args,
    std::int32_t source,
    std::int32_t channel)
{{
    const std::size_t source_offset =
        static_cast<std::size_t>(source) * {source_harmonics};
    {scalar_type} values[{source_harmonics}] = {{}};
    {scalar_type} compensation[{source_harmonics}] = {{}};
    for (std::int32_t scheduled = args->source_offsets[source];
         scheduled < args->source_offsets[source + 1]; ++scheduled) {{
        const std::int32_t edge = args->source_edges[scheduled];
        if (!edge_is_active(args->cutoff, args->radius[edge]))
            continue;
        const std::size_t edge_offset =
            static_cast<std::size_t>(edge) * {edge_harmonics};
        const std::int32_t receiver = args->edge_receivers[edge];
        const std::int32_t receiver_type =
            args->type_to_active[args->node_types[receiver]];
        const std::int32_t source_type =
            args->type_to_active[args->neigh_types[edge]];
        const std::int32_t edge_type = pair_type(
            receiver_type, source_type,
            static_cast<std::int32_t>(args->active_type_count));
        const EvaluationPoint point = evaluation_point(
            args->radial, args->radius[edge]);
{chr(10).join(blocks)}
    }}
{stores}
}}

{function_prefix}void r1_source_owner(
    const {args_type}* args,
    std::int32_t source,
    std::int32_t channel)
{{
    r1_source_owner_impl<false>(args, source, channel);
}}

{function_prefix}void r1_compensated_source_owner(
    const {args_type}* args,
    std::int32_t source,
    std::int32_t channel)
{{
    r1_source_owner_impl<true>(args, source, channel);
}}"""


def _render_tiled_host_forward_owner(
    path_rows,
    *,
    channels: int,
    edge_harmonics: int,
    source_harmonics: int,
    output_components: int,
    channel_tile: int,
    args_type: str = "SymmetrixJitHostR1ForwardArgsV1",
    scalar_type: str = "float",
) -> str:
    if channel_tile <= 0:
        raise ValueError("host forward channel tile must be positive")
    harmonics = _scalar_pointer("args->harmonics_values", scalar_type, opaque=False)
    neighbors = _scalar_pointer("args->neighbor_features", scalar_type, opaque=False)
    output = _mutable_scalar_pointer("args->output", scalar_type, opaque=False)
    blocks = []
    for path_index, rows in enumerate(path_rows):
        statements = [
            "        for (std::int32_t lane = 0; lane < channel_count; ++lane) {",
            "            const std::int32_t lane_channel = channel + lane;",
            f"            const {scalar_type} radial_{path_index} = evaluate_radial(",
            f"                args->radial, edge_type, point, {path_index} * {channels} + lane_channel);",
        ]
        for row in rows:
            statements.extend(
                [
                    f"            const {scalar_type} product_{row['row']} = radial_{path_index}",
                    f"                * {harmonics}[edge_offset + {row['lm1']}]",
                    f"                * {neighbors}[(source_offset + {row['lm2']}) * {channels} + lane_channel];",
                ]
            )
            for term in row["terms"]:
                statements.append(
                    f"            values[{term['lme']}][lane] += "
                    f"{_cpp_scalar(term['coefficient'], scalar_type)}"
                    f" * product_{row['row']};"
                )
        statements.append("        }")
        blocks.append("\n".join(statements))
    stores = "\n".join(
        f"        {output}[(receiver_offset + {lme}) * {channels} + lane_channel] = values[{lme}][lane];"
        for lme in range(output_components)
    )
    return f"""void r1_forward_owner(
    const {args_type}* args,
    std::int32_t receiver,
    std::int32_t channel)
{{
    constexpr std::int32_t channel_tile = {channel_tile};
    if (channel % channel_tile != 0)
        return;
    const std::int32_t channel_count =
        channel + channel_tile <= {channels} ? channel_tile : {channels} - channel;
    {scalar_type} values[{output_components}][channel_tile] = {{}};
    const std::size_t receiver_offset =
        static_cast<std::size_t>(receiver) * {output_components};
    const std::int32_t receiver_type =
        args->type_to_active[args->node_types[receiver]];
    const std::int32_t edge_begin = args->first_neigh[receiver];
    const std::int32_t edge_end = edge_begin + args->num_neigh[receiver];
    for (std::int32_t edge = edge_begin; edge < edge_end; ++edge) {{
        if (!edge_is_active(args->cutoff, args->radius[edge]))
            continue;
        const std::int32_t source = args->neigh_indices[edge];
        const std::size_t edge_offset =
            static_cast<std::size_t>(edge) * {edge_harmonics};
        const std::size_t source_offset =
            static_cast<std::size_t>(source) * {source_harmonics};
        const std::int32_t source_type =
            args->type_to_active[args->neigh_types[edge]];
        const std::int32_t edge_type = pair_type(
            receiver_type, source_type,
            static_cast<std::int32_t>(args->active_type_count));
        const EvaluationPoint point = evaluation_point(
            args->radial, args->radius[edge]);
{chr(10).join(blocks)}
    }}
    for (std::int32_t lane = 0; lane < channel_count; ++lane) {{
        const std::int32_t lane_channel = channel + lane;
{stores}
    }}
}}"""


def _render_tiled_host_source_owner(
    path_rows,
    *,
    channels: int,
    edge_harmonics: int,
    source_harmonics: int,
    output_components: int,
    channel_tile: int,
    args_type: str = "SymmetrixJitHostR1SourceArgsV1",
    scalar_type: str = "float",
) -> str:
    if channel_tile <= 0:
        raise ValueError("host source channel tile must be positive")
    harmonics = _scalar_pointer("args->harmonics_values", scalar_type, opaque=False)
    source_adjoint = _mutable_scalar_pointer(
        "args->source_adjoint", scalar_type, opaque=False
    )
    output_adjoint = _scalar_pointer("args->output_adjoint", scalar_type, opaque=False)
    blocks = []
    for path_index, rows in enumerate(path_rows):
        statements = [
            "        for (std::int32_t lane = 0; lane < channel_count; ++lane) {",
            "            const std::int32_t lane_channel = channel + lane;",
            f"            const {scalar_type} radial_{path_index} = evaluate_radial(",
            f"                args->radial, edge_type, point, {path_index} * {channels} + lane_channel);",
        ]
        for row in rows:
            statements.extend(
                [
                    f"            const {scalar_type} contribution_{row['row']} = radial_{path_index}",
                    f"                * {harmonics}[edge_offset + {row['lm1']}]",
                    f"                * load_row_{row['row']}(args, receiver, lane_channel);",
                    "            if constexpr (Compensated) {",
                    f"                const {scalar_type} corrected_{row['row']} = contribution_{row['row']} - compensation[{row['lm2']}][lane];",
                    f"                const {scalar_type} updated_{row['row']} = values[{row['lm2']}][lane] + corrected_{row['row']};",
                    f"                compensation[{row['lm2']}][lane] = (updated_{row['row']} - values[{row['lm2']}][lane]) - corrected_{row['row']};",
                    f"                values[{row['lm2']}][lane] = updated_{row['row']};",
                    "            } else {",
                    f"                values[{row['lm2']}][lane] += contribution_{row['row']};",
                    "            }",
                ]
            )
        statements.append("        }")
        blocks.append("\n".join(statements))
    stores = "\n".join(
        f"        {source_adjoint}[(source_offset + {lm}) * {channels} + lane_channel] += values[{lm}][lane];"
        for lm in range(source_harmonics)
    )
    return f"""template <bool Compensated>
void r1_source_owner_impl(
    const {args_type}* args,
    std::int32_t source,
    std::int32_t channel)
{{
    constexpr std::int32_t channel_tile = {channel_tile};
    if (channel % channel_tile != 0)
        return;
    const std::int32_t channel_count =
        channel + channel_tile <= {channels} ? channel_tile : {channels} - channel;
    const std::size_t source_offset =
        static_cast<std::size_t>(source) * {source_harmonics};
    {scalar_type} values[{source_harmonics}][channel_tile] = {{}};
    {scalar_type} compensation[{source_harmonics}][channel_tile] = {{}};
    for (std::int32_t scheduled = args->source_offsets[source];
         scheduled < args->source_offsets[source + 1]; ++scheduled) {{
        if (scheduled + {_HOST_R1_SOURCE_REVERSE_PREFETCH_DISTANCE}
                < args->source_offsets[source + 1]) {{
            const std::int32_t prefetch_edge = args->source_edges[
                scheduled + {_HOST_R1_SOURCE_REVERSE_PREFETCH_DISTANCE}];
            const std::int32_t prefetch_receiver =
                args->edge_receivers[prefetch_edge];
            prefetch_read({harmonics}
                + static_cast<std::size_t>(prefetch_edge) * {edge_harmonics});
            prefetch_read({output_adjoint}
                + static_cast<std::size_t>(prefetch_receiver)
                    * {output_components * channels} + channel);
        }}
        const std::int32_t edge = args->source_edges[scheduled];
        if (!edge_is_active(args->cutoff, args->radius[edge]))
            continue;
        const std::size_t edge_offset =
            static_cast<std::size_t>(edge) * {edge_harmonics};
        const std::int32_t receiver = args->edge_receivers[edge];
        const std::int32_t receiver_type =
            args->type_to_active[args->node_types[receiver]];
        const std::int32_t source_type =
            args->type_to_active[args->neigh_types[edge]];
        const std::int32_t edge_type = pair_type(
            receiver_type, source_type,
            static_cast<std::int32_t>(args->active_type_count));
        const EvaluationPoint point = evaluation_point(
            args->radial, args->radius[edge]);
{chr(10).join(blocks)}
    }}
    for (std::int32_t lane = 0; lane < channel_count; ++lane) {{
        const std::int32_t lane_channel = channel + lane;
{stores}
    }}
}}

void r1_source_owner(
    const {args_type}* args,
    std::int32_t source,
    std::int32_t channel)
{{
    r1_source_owner_impl<false>(args, source, channel);
}}

void r1_compensated_source_owner(
    const {args_type}* args,
    std::int32_t source,
    std::int32_t channel)
{{
    r1_source_owner_impl<true>(args, source, channel);
}}"""


def _render_edge_owner(
    path_rows,
    *,
    channels: int,
    edge_harmonics: int,
    source_harmonics: int,
    args_type: str = "SymmetrixJitHostR1EdgeArgsV1",
    qualifier: str = "",
    scalar_type: str = "float",
    opaque_pointers: bool = False,
) -> str:
    neighbors = _scalar_pointer(
        "args->neighbor_features", scalar_type, opaque=opaque_pointers
    )
    harmonics = _scalar_pointer(
        "args->harmonics_values", scalar_type, opaque=opaque_pointers
    )
    gradients = "gradients"
    gradient_pointer = (
        _scalar_pointer("args->harmonics_gradients", scalar_type, opaque=True)
        if opaque_pointers
        else f"static_cast<const {scalar_type}*>(args->harmonics_gradients)"
    )
    blocks = []
    for path_index, rows in enumerate(path_rows):
        statements = [
            "        {",
            f"        {scalar_type} radial_value_{path_index} = {scalar_type}(0);",
            f"        {scalar_type} radial_derivative_{path_index} = {scalar_type}(0);",
            "        evaluate_radial(",
            f"            args->radial, edge_type, point, {path_index} * {channels} + channel,",
            f"            radial_value_{path_index}, radial_derivative_{path_index});",
        ]
        for row in rows:
            statements.extend(
                [
                    "        {",
                    f"        const {scalar_type} weighted_{row['row']} =",
                    f"            {neighbors}[(source_offset + {row['lm2']}) * {channels} + channel]",
                    f"            * load_row_{row['row']}(args, receiver, channel);",
                    f"        const {scalar_type} radial_force_{row['row']} = radial_derivative_{path_index}",
                    f"            * weighted_{row['row']} * {harmonics}[edge_offset + {row['lm1']}];",
                    f"        const {scalar_type} angular_force_{row['row']} = radial_value_{path_index} * weighted_{row['row']};",
                    f"        local_force_x += radial_force_{row['row']} * x_over_r",
                    f"            + angular_force_{row['row']} * {gradients}[{row['lm1']}];",
                    f"        local_force_y += radial_force_{row['row']} * y_over_r",
                    f"            + angular_force_{row['row']} * {gradients}[{edge_harmonics + row['lm1']}];",
                    f"        local_force_z += radial_force_{row['row']} * z_over_r",
                    f"            + angular_force_{row['row']} * {gradients}[{2 * edge_harmonics + row['lm1']}];",
                    "        }",
                ]
            )
        statements.append("        }")
        blocks.append("\n".join(statements))
    function_prefix = f"{qualifier} " if qualifier else ""
    return f"""{function_prefix}void r1_edge_owner(
    const {args_type}* args,
    std::int32_t edge)
{{
    if (!edge_is_active(args->cutoff, args->radius[edge]))
        return;
    const std::int32_t receiver = args->edge_receivers[edge];
    const std::int32_t source = args->neigh_indices[edge];
    const std::size_t coordinate_offset =
        static_cast<std::size_t>(3) * edge;
    const std::size_t edge_offset =
        static_cast<std::size_t>(edge) * {edge_harmonics};
    const std::size_t gradient_offset =
        static_cast<std::size_t>(3) * edge_offset;
    const std::size_t source_offset =
        static_cast<std::size_t>(source) * {source_harmonics};
    const std::int32_t receiver_type =
        args->type_to_active[args->node_types[receiver]];
    const std::int32_t source_type =
        args->type_to_active[args->neigh_types[edge]];
    const std::int32_t edge_type = pair_type(
        receiver_type, source_type,
        static_cast<std::int32_t>(args->active_type_count));
    const EvaluationPoint point = evaluation_point(
        args->radial, args->radius[edge]);
    const {scalar_type} x_over_r = args->coordinates_are_unit != 0
        ? static_cast<{scalar_type}>(reinterpret_cast<const float*>(args->xyz)[coordinate_offset])
        : static_cast<{scalar_type}>(reinterpret_cast<const double*>(args->xyz)[coordinate_offset] / args->radius[edge]);
    const {scalar_type} y_over_r = args->coordinates_are_unit != 0
        ? static_cast<{scalar_type}>(reinterpret_cast<const float*>(args->xyz)[coordinate_offset + 1])
        : static_cast<{scalar_type}>(reinterpret_cast<const double*>(args->xyz)[coordinate_offset + 1] / args->radius[edge]);
    const {scalar_type} z_over_r = args->coordinates_are_unit != 0
        ? static_cast<{scalar_type}>(reinterpret_cast<const float*>(args->xyz)[coordinate_offset + 2])
        : static_cast<{scalar_type}>(reinterpret_cast<const double*>(args->xyz)[coordinate_offset + 2] / args->radius[edge]);
    const {scalar_type}* gradients = {gradient_pointer};
    {scalar_type} direct_gradients[3 * {edge_harmonics}];
    if (gradients == nullptr) {{
        direct_harmonic_gradients(
            args->xyz, args->coordinate_scalar_size,
            args->coordinates_are_unit, coordinate_offset, args->radius[edge],
            direct_gradients);
        gradients = direct_gradients;
    }} else {{
        gradients += gradient_offset;
    }}
    {scalar_type} local_force_x = {scalar_type}(0);
    {scalar_type} local_force_y = {scalar_type}(0);
    {scalar_type} local_force_z = {scalar_type}(0);
    for (std::int32_t channel = 0; channel < {channels}; ++channel) {{
{chr(10).join(blocks)}
    }}
    args->directed_forces[coordinate_offset] -= static_cast<double>(local_force_x);
    args->directed_forces[coordinate_offset + 1] -= static_cast<double>(local_force_y);
    args->directed_forces[coordinate_offset + 2] -= static_cast<double>(local_force_z);
}}"""


def _render_wave_edge_owner(
    path_rows,
    *,
    args_type: str,
    channels: int,
    edge_harmonics: int,
    source_harmonics: int,
    subgroup_width: int,
    dialect: GpuDialect,
    qualifier: str = "__device__ __forceinline__",
) -> str:
    blocks = []
    for path_index, rows in enumerate(path_rows):
        statements = [
            "        {",
            f"        Scalar radial_value_{path_index} = Scalar(0);",
            f"        Scalar radial_derivative_{path_index} = Scalar(0);",
            "        evaluate_radial(",
            f"            args->radial, edge_type, point, {path_index} * {channels} + channel,",
            f"            radial_value_{path_index}, radial_derivative_{path_index});",
        ]
        for row in rows:
            statements.extend(
                [
                    "        {",
                    f"        const Scalar weighted_{row['row']} =",
                    f"            static_cast<const Scalar*>(args->neighbor_features)[(source_offset + {row['lm2']}) * {channels} + channel]",
                    f"            * load_row_{row['row']}(args, receiver, channel);",
                    f"        const Scalar radial_force_{row['row']} = radial_derivative_{path_index}",
                    f"            * weighted_{row['row']} * static_cast<const Scalar*>(args->harmonics_values)[edge_offset + {row['lm1']}];",
                    f"        const Scalar angular_force_{row['row']} = radial_value_{path_index} * weighted_{row['row']};",
                    f"        local_force_x += radial_force_{row['row']} * x_over_r",
                    f"            + angular_force_{row['row']} * static_cast<const Scalar*>(args->harmonics_gradients)[gradient_offset + {row['lm1']}];",
                    f"        local_force_y += radial_force_{row['row']} * y_over_r",
                    f"            + angular_force_{row['row']} * static_cast<const Scalar*>(args->harmonics_gradients)[gradient_offset + {edge_harmonics + row['lm1']}];",
                    f"        local_force_z += radial_force_{row['row']} * z_over_r",
                    f"            + angular_force_{row['row']} * static_cast<const Scalar*>(args->harmonics_gradients)[gradient_offset + {2 * edge_harmonics + row['lm1']}];",
                    "        }",
                ]
            )
        statements.append("        }")
        blocks.append("\n".join(statements))
    shuffle_x = dialect.shuffle_down("local_force_x", "offset", str(subgroup_width))
    shuffle_y = dialect.shuffle_down("local_force_y", "offset", str(subgroup_width))
    shuffle_z = dialect.shuffle_down("local_force_z", "offset", str(subgroup_width))
    return f"""{qualifier} void r1_edge_owner(
    const {args_type}* args,
    std::int32_t edge,
    std::int32_t lane)
{{
    if (!edge_is_active(args->cutoff, args->radius[edge]))
        return;
    const std::int32_t receiver = args->edge_receivers[edge];
    const std::int32_t source = args->neigh_indices[edge];
    const std::size_t coordinate_offset =
        static_cast<std::size_t>(3) * edge;
    const std::size_t edge_offset =
        static_cast<std::size_t>(edge) * {edge_harmonics};
    const std::size_t gradient_offset =
        static_cast<std::size_t>(3) * edge_offset;
    const std::size_t source_offset =
        static_cast<std::size_t>(source) * {source_harmonics};
    const std::int32_t receiver_type =
        args->type_to_active[args->node_types[receiver]];
    const std::int32_t source_type =
        args->type_to_active[args->neigh_types[edge]];
    const std::int32_t edge_type = pair_type(
        receiver_type, source_type,
        static_cast<std::int32_t>(args->active_type_count));
    const EvaluationPoint point = evaluation_point(
        args->radial, args->radius[edge]);
    const Scalar x_over_r = args->coordinates_are_unit != 0
        ? static_cast<Scalar>(reinterpret_cast<const float*>(args->xyz)[coordinate_offset])
        : static_cast<Scalar>(reinterpret_cast<const double*>(args->xyz)[coordinate_offset]
            / args->radius[edge]);
    const Scalar y_over_r = args->coordinates_are_unit != 0
        ? static_cast<Scalar>(reinterpret_cast<const float*>(args->xyz)[coordinate_offset + 1])
        : static_cast<Scalar>(reinterpret_cast<const double*>(args->xyz)[coordinate_offset + 1]
            / args->radius[edge]);
    const Scalar z_over_r = args->coordinates_are_unit != 0
        ? static_cast<Scalar>(reinterpret_cast<const float*>(args->xyz)[coordinate_offset + 2])
        : static_cast<Scalar>(reinterpret_cast<const double*>(args->xyz)[coordinate_offset + 2]
            / args->radius[edge]);
    Scalar local_force_x = Scalar(0);
    Scalar local_force_y = Scalar(0);
    Scalar local_force_z = Scalar(0);
    for (std::int32_t channel = lane; channel < {channels};
         channel += {subgroup_width}) {{
{chr(10).join(blocks)}
    }}
    for (std::int32_t offset = {subgroup_width // 2}; offset > 0; offset /= 2) {{
        local_force_x += {shuffle_x};
        local_force_y += {shuffle_y};
        local_force_z += {shuffle_z};
    }}
    if (lane == 0) {{
        args->directed_forces[coordinate_offset] -= static_cast<double>(local_force_x);
        args->directed_forces[coordinate_offset + 1] -= static_cast<double>(local_force_y);
        args->directed_forces[coordinate_offset + 2] -= static_cast<double>(local_force_z);
    }}
}}"""


def jit_r1_host_plugin_metadata(
    contract: dict,
    artifact_id: str | None = None,
    *,
    precision: str = "float32",
) -> dict:
    precision_spec = _precision_spec(precision)
    normalized = normalize_jit_r1_contract(contract)
    groups = _contract_groups(normalized)
    edge_l_max = normalized["edge_harmonics"]["l_max"]
    if edge_l_max > 3:
        raise ValueError(
            "Execution R1 specialization coordinate reverse supports edge l_max <= 3"
        )
    generation = normalized["generation_fingerprint"]
    return {
        "abi": HOST_PLUGIN_ABI,
        "abi_version": HOST_PLUGIN_ABI_VERSION,
        "artifact_id": artifact_id
        or f"jit-r1-gen{JIT_GENERATION_VERSION}-{precision_spec['suffix']}-"
        + generation.removeprefix("sha256:")[:16],
        "contract_fingerprint": generation,
        "semantic_fingerprint": normalized["fingerprint"],
        "structure_fingerprint": normalized["structure_fingerprint"],
        "channels": normalized["channels"],
        "embedding": normalized["radial_embedding"],
        "edge_l_max": edge_l_max,
        "source_l_max": normalized["source_harmonics"]["l_max"],
        "output_components": sum(
            components * multiplicity for _, components, multiplicity in groups
        ),
        "precision": precision,
        "scalar_kind": precision_spec["scalar_kind"],
        "scalar_size": precision_spec["scalar_size"],
    }


def render_jit_r1_host_plugin(
    contract: dict,
    artifact_id: str | None = None,
    *,
    precision: str = "float32",
) -> str:
    """Render a standalone C++20 host plugin for one normalized R1 contract."""

    normalized = normalize_jit_r1_contract(contract)
    metadata = jit_r1_host_plugin_metadata(normalized, artifact_id, precision=precision)
    path_rows = _fixed_weight_path_rows(normalized)
    channels = metadata["channels"]
    edge_harmonics = (metadata["edge_l_max"] + 1) ** 2
    source_harmonics = (metadata["source_l_max"] + 1) ** 2
    output_components = metadata["output_components"]
    row_loaders = _render_row_loaders(path_rows, channels, output_components)
    forward_owner = _render_tiled_host_forward_owner(
        path_rows,
        channels=channels,
        edge_harmonics=edge_harmonics,
        source_harmonics=source_harmonics,
        output_components=output_components,
        channel_tile=_HOST_R1_FORWARD_CHANNEL_TILE,
    )
    source_owner = _render_tiled_host_source_owner(
        path_rows,
        channels=channels,
        edge_harmonics=edge_harmonics,
        source_harmonics=source_harmonics,
        output_components=output_components,
        channel_tile=_HOST_R1_SOURCE_CHANNEL_TILE,
    )
    edge_owner = _render_edge_owner(
        path_rows,
        channels=channels,
        edge_harmonics=edge_harmonics,
        source_harmonics=source_harmonics,
    )
    source = f"""// Generated by symmetrix.jit_codegen. Do not edit.
{_HOST_PLUGIN_ABI_HEADER}

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {{

struct EvaluationPoint {{
    std::int32_t interval;
    double x;
    double xx;
    double xxx;
}};

inline bool edge_is_active(double cutoff, double radius)
{{
    return radius < cutoff;
}}

inline EvaluationPoint evaluation_point(
    const SymmetrixJitHostRadialSplineV1& radial, double radius)
{{
    std::int32_t interval = static_cast<std::int32_t>(
        std::floor((radius - radial.x0) / radial.h));
    double x = radius - radial.x0 - radial.h * interval;
    if (interval < 0) {{
        interval = 0;
        x = 0.0;
    }} else if (interval >= static_cast<std::int32_t>(radial.intervals)) {{
        interval = static_cast<std::int32_t>(radial.intervals) - 1;
        x = radial.h;
    }}
    const double xx = x * x;
    return {{interval, x, xx, xx * x}};
}}

inline std::size_t radial_index(
    const SymmetrixJitHostRadialSplineV1& radial,
    std::int32_t edge_type,
    std::int32_t interval,
    std::int32_t coefficient,
    std::int32_t function)
{{
    return (((static_cast<std::size_t>(edge_type) * radial.intervals + interval)
        * 4u + coefficient) * radial.functions + function);
}}

inline float evaluate_radial(
    const SymmetrixJitHostRadialSplineV1& radial,
    std::int32_t edge_type,
    const EvaluationPoint& point,
    std::int32_t function)
{{
    const float c0 = radial.coefficients[radial_index(radial, edge_type, point.interval, 0, function)];
    const float c1 = radial.coefficients[radial_index(radial, edge_type, point.interval, 1, function)];
    const float c2 = radial.coefficients[radial_index(radial, edge_type, point.interval, 2, function)];
    const float c3 = radial.coefficients[radial_index(radial, edge_type, point.interval, 3, function)];
    return c0 + c1 * static_cast<float>(point.x)
        + c2 * static_cast<float>(point.xx)
        + c3 * static_cast<float>(point.xxx);
}}

inline void evaluate_radial(
    const SymmetrixJitHostRadialSplineV1& radial,
    std::int32_t edge_type,
    const EvaluationPoint& point,
    std::int32_t function,
    float& value,
    float& derivative)
{{
    const float c0 = radial.coefficients[radial_index(radial, edge_type, point.interval, 0, function)];
    const float c1 = radial.coefficients[radial_index(radial, edge_type, point.interval, 1, function)];
    const float c2 = radial.coefficients[radial_index(radial, edge_type, point.interval, 2, function)];
    const float c3 = radial.coefficients[radial_index(radial, edge_type, point.interval, 3, function)];
    value = c0 + c1 * static_cast<float>(point.x)
        + c2 * static_cast<float>(point.xx)
        + c3 * static_cast<float>(point.xxx);
    derivative = c1 + c2 * static_cast<float>(2.0 * point.x)
        + c3 * static_cast<float>(3.0 * point.xx);
}}

inline std::int32_t pair_type(
    std::int32_t left, std::int32_t right, std::int32_t type_count)
{{
    return left <= right
        ? left * (2 * type_count - left - 1) / 2 + right
        : right * (2 * type_count - right - 1) / 2 + left;
}}

inline void prefetch_read(const void* address) noexcept
{{
#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_LLVM_COMPILER)
    __builtin_prefetch(address, 0, 1);
#else
    (void)address;
#endif
}}

{row_loaders}

{_render_direct_harmonic_gradient_helper("inline", edge_harmonics)}

{forward_owner}

{source_owner}

{edge_owner}

const SymmetrixJitHostPluginV1 plugin = {{
    SYMMETRIX_JIT_HOST_PLUGIN_ABI_VERSION,
    sizeof(SymmetrixJitHostPluginV1),
    sizeof(void*),
    SYMMETRIX_JIT_HOST_PLUGIN_BYTE_ORDER,
    SYMMETRIX_JIT_HOST_R1_FORWARD_OWNER_V1
        | SYMMETRIX_JIT_HOST_R1_SOURCE_OWNER_V1
        | SYMMETRIX_JIT_HOST_R1_COMPENSATED_SOURCE_OWNER_V1
        | SYMMETRIX_JIT_HOST_R1_EDGE_OWNER_V1,
    0u,
    {_cpp_string(HOST_PLUGIN_ABI)},
    {_cpp_string(metadata["artifact_id"])},
    {_cpp_string(metadata["contract_fingerprint"])},
    {_cpp_string(metadata["semantic_fingerprint"])},
    {_cpp_string(metadata["structure_fingerprint"])},
    {metadata["channels"]},
    {metadata["embedding"]},
    {metadata["edge_l_max"]},
    {metadata["source_l_max"]},
    &r1_forward_owner,
    &r1_source_owner,
    &r1_compensated_source_owner,
    &r1_edge_owner,
}};

}} // namespace

extern "C" SYMMETRIX_JIT_HOST_PLUGIN_EXPORT
const SymmetrixJitHostPluginV1* symmetrix_jit_host_plugin_query_v1()
{{
    return &plugin;
}}
"""
    return _render_precision_v2_source(source, "host", precision)


def jit_r1_cuda_plugin_metadata(
    contract: dict,
    compute_capability: int,
    artifact_id: str | None = None,
    *,
    precision: str = "float32",
    edge_strategy: str = "wave",
    edge_logical_subgroup_width: int | None = 32,
    edge_threads_per_block: int = 32,
    persistent_blocks_per_compute_unit: int = 8,
) -> dict:
    """Return exact ABI and model metadata for one CUDA R1 plugin."""

    if (
        isinstance(compute_capability, bool)
        or not isinstance(compute_capability, int)
        or compute_capability <= 0
        or compute_capability > 999
    ):
        raise ValueError("CUDA compute capability must be an integer in [1, 999]")
    program, target, schedule, launch_plan = _cuda_codegen_inputs(
        contract,
        compute_capability,
        precision,
        "plugin",
        edge_strategy,
        edge_logical_subgroup_width,
        edge_threads_per_block,
        persistent_blocks_per_compute_unit,
    )
    metadata = jit_r1_host_plugin_metadata(
        program.contract, artifact_id, precision=precision
    )
    metadata.update(
        {
            "abi": CUDA_PLUGIN_ABI,
            "abi_version": CUDA_PLUGIN_ABI_VERSION,
            "target_compute_capability": compute_capability,
            "forward_threads_per_block": schedule.forward_threads_per_block,
            "source_threads_per_block": schedule.source_threads_per_block,
            "edge_threads_per_block": schedule.edge_threads_per_block,
            "reverse_threads_per_block": 64,
            "reverse_shared_memory_bytes": (
                (metadata["source_l_max"] + 1) ** 2
                * metadata["channels"]
                * metadata["scalar_size"]
            ),
            "persistent_blocks_per_compute_unit": (
                schedule.persistent_blocks_per_compute_unit
            ),
            "reverse_strategy": "fused_source_edge",
            "edge_strategy": schedule.edge_strategy,
            "edge_logical_subgroup_width": schedule.edge_logical_subgroup_width,
            "gpu_codegen": json.loads(
                gpu_codegen_identity(program, target, schedule, launch_plan)
            ),
        }
    )
    return metadata


def _render_r1_edge_kernel(
    edge_type: str, schedule: KernelSchedule, kernel_name: str, *, exported: bool
) -> str:
    prefix = 'extern "C" __global__' if exported else "__global__"
    if schedule.edge_strategy == "wave":
        width = schedule.edge_logical_subgroup_width
        return f"""{prefix} __launch_bounds__(edge_threads_per_block)
void {kernel_name}({edge_type} args)
{{
    const std::int64_t linear_thread =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int32_t lane = threadIdx.x % {width};
    const std::int64_t wave = linear_thread / {width};
    const std::int64_t wave_stride =
        static_cast<std::int64_t>(blockDim.x) * gridDim.x / {width};
    for (std::int64_t edge = wave; edge < args.num_edges;
         edge += wave_stride)
        r1_edge_owner(&args, static_cast<std::int32_t>(edge), lane);
}}"""
    return f"""{prefix} void {kernel_name}({edge_type} args)
{{
    const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    for (std::int64_t edge =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         edge < args.num_edges; edge += stride)
        r1_edge_owner(&args, static_cast<std::int32_t>(edge));
}}"""


def _render_r1_fused_reverse_kernel(
    path_rows,
    *,
    source_type: str,
    edge_type: str,
    kernel_name: str,
    channels: int,
    edge_harmonics: int,
    source_harmonics: int,
    output_components: int,
    subgroup_width: int,
    dialect: GpuDialect,
    exported: bool,
    ordered_pair_types: bool = False,
    map_type_indices: bool = True,
    compact_source_owners: bool = False,
    compact_source_ids: bool = False,
    density_function: int | None = None,
    radial_function_offsets: tuple[int, ...] | None = None,
    use_output_mask: bool = False,
    channel_tiled: bool = False,
    channel_tile_size: int | None = None,
    filter_inactive_edges: bool = True,
    shared_source_features: bool = False,
) -> str:
    if channel_tiled and (channel_tile_size is None or channel_tile_size <= 0):
        raise ValueError("channel-tiled reverse requires a positive tile size")
    if compact_source_ids and not compact_source_owners:
        raise ValueError("compact source IDs require compact source owners")
    prefix = 'extern "C" __global__' if exported else "__global__"
    reverse_threads = 64
    if reverse_threads % subgroup_width:
        raise ValueError("fused reverse block must be a native-subgroup multiple")
    subgroup_count = reverse_threads // subgroup_width
    path_rows_by_lm1 = []
    for path_index, rows in enumerate(path_rows):
        rows_by_lm1 = {}
        for row in rows:
            rows_by_lm1.setdefault(row["lm1"], []).append(row)
        path_rows_by_lm1.append(rows_by_lm1)

    def render_harmonic_bands():
        blocks = []
        # Apply each harmonic force basis once across all radial paths that
        # expose the same harmonic set.
        path_bands = []
        band_indices = {}
        for path_index, rows_by_lm1 in enumerate(path_rows_by_lm1):
            signature = tuple(rows_by_lm1)
            band_index = band_indices.get(signature)
            if band_index is None:
                band_indices[signature] = len(path_bands)
                path_bands.append((signature, [(path_index, rows_by_lm1)]))
            else:
                path_bands[band_index][1].append((path_index, rows_by_lm1))

        for band_index, (signature, band_paths) in enumerate(path_bands):
            statements = [
                "                {",
                f"                Scalar radial_force_{band_index} = Scalar(0);",
            ]
            for lm1 in signature:
                statements.extend(
                    [
                        f"                Scalar angular_force_{band_index}_{lm1} = Scalar(0);",
                    ]
                )
            band_rows = [
                row
                for _, rows_by_lm1 in band_paths
                for harmonic_rows in rows_by_lm1.values()
                for row in harmonic_rows
            ]
            band_output_components = sorted(
                {term["lme"] for row in band_rows for term in row["terms"]}
            )
            for lme in band_output_components:
                mask = (
                    f" * static_cast<const Scalar*>(source_args.output_mask)"
                    f"[{lme} * {channels} + channel]"
                    if use_output_mask
                    else ""
                )
                output_stride = (
                    str(channel_tile_size) if channel_tiled else str(channels)
                )
                output_channel = "local_channel" if channel_tiled else "channel"
                statements.extend(
                    [
                        f"                const Scalar output_adjoint_{band_index}_{lme} =",
                        "                    static_cast<const Scalar*>(source_args.output_adjoint)[",
                        f"                        (static_cast<std::size_t>(receiver) * {output_components} + {lme}) * ({output_stride}) + {output_channel}]{mask};",
                    ]
                )
            for path_index, rows_by_lm1 in band_paths:
                radial_offset = (
                    path_index * channels
                    if radial_function_offsets is None
                    else radial_function_offsets[path_index]
                )
                statements.extend(
                    [
                        "                {",
                        f"                Scalar radial_value_{path_index} = Scalar(0);",
                        f"                Scalar radial_derivative_{path_index} = Scalar(0);",
                        "                evaluate_radial(",
                        f"                    edge_args.radial, edge_type, point, {radial_offset} + channel,",
                        f"                    radial_value_{path_index}, radial_derivative_{path_index});",
                    ]
                )
                for lm1, harmonic_rows in rows_by_lm1.items():
                    statements.extend(
                        [
                            "                {",
                            f"                const Scalar harmonic_value_{path_index}_{lm1} =",
                            f"                    harmonics_values[edge_offset + {lm1}];",
                            f"                Scalar weighted_{path_index}_{lm1} = Scalar(0);",
                        ]
                    )
                    for row in harmonic_rows:
                        row_index = row["row"]
                        lm2 = row["lm2"]
                        row_adjoint = "\n                        + ".join(
                            f"Scalar({term['coefficient']!r})"
                            f" * output_adjoint_{band_index}_{term['lme']}"
                            for term in row["terms"]
                        )
                        statements.extend(
                            [
                                "                {",
                                f"                const Scalar row_adjoint_{row_index} =",
                                f"                    {row_adjoint};",
                                f"                source_values[{lm2} * {channels} + channel] +=",
                                f"                    radial_value_{path_index}",
                                f"                    * harmonic_value_{path_index}_{lm1}",
                                f"                    * row_adjoint_{row_index};",
                                f"                weighted_{path_index}_{lm1} +=",
                                (
                                    f"                    source_features[{lm2} * {channels} + channel]"
                                    if shared_source_features
                                    else f"                    neighbor_features[(source_offset + {lm2}) * {channels} + channel]"
                                ),
                                f"                    * row_adjoint_{row_index};",
                                "                }",
                            ]
                        )
                    statements.extend(
                        [
                            f"                radial_force_{band_index} +=",
                            f"                    radial_derivative_{path_index}",
                            f"                    * weighted_{path_index}_{lm1}",
                            f"                    * harmonic_value_{path_index}_{lm1};",
                            f"                angular_force_{band_index}_{lm1} +=",
                            f"                    radial_value_{path_index}",
                            f"                    * weighted_{path_index}_{lm1};",
                            "                }",
                        ]
                    )
                statements.append("                }")
            statements.extend(
                [
                    f"                local_force_x += radial_force_{band_index} * x_over_r;",
                    f"                local_force_y += radial_force_{band_index} * y_over_r;",
                    f"                local_force_z += radial_force_{band_index} * z_over_r;",
                ]
            )
            for lm1 in signature:
                statements.extend(
                    [
                        f"                local_force_x += angular_force_{band_index}_{lm1}",
                        f"                    * edge_gradients[{lm1}];",
                        f"                local_force_y += angular_force_{band_index}_{lm1}",
                        f"                    * edge_gradients[{edge_harmonics + lm1}];",
                        f"                local_force_z += angular_force_{band_index}_{lm1}",
                        f"                    * edge_gradients[{2 * edge_harmonics + lm1}];",
                    ]
                )
            statements.append("                }")
            blocks.append("\n".join(statements))
        return blocks

    blocks = render_harmonic_bands()
    # A bounded runtime dispatcher prevents RTC compilers from extending live
    # ranges across the complete unrolled reverse contraction.
    reverse_path_group_size = 5
    path_groups = [
        blocks[index : index + reverse_path_group_size]
        for index in range(0, len(blocks), reverse_path_group_size)
    ]
    dispatch_cases = []
    for path_index, path_group in enumerate(path_groups):
        dispatch_cases.extend(
            [
                f"                    case {path_index}:",
                "\n".join(path_group),
                "                        break;",
            ]
        )
    path_dispatch = "\n".join(
        [
            "                #pragma unroll 1",
            f"                for (std::int32_t reverse_path = 0; reverse_path < {len(path_groups)};",
            "                     ++reverse_path) {",
            "                    switch (reverse_path) {",
            *dispatch_cases,
            "                    }",
            "                }",
        ]
    )
    shuffle_x = dialect.shuffle_down("local_force_x", "offset", str(subgroup_width))
    shuffle_y = dialect.shuffle_down("local_force_y", "offset", str(subgroup_width))
    shuffle_z = dialect.shuffle_down("local_force_z", "offset", str(subgroup_width))
    block_shuffle_x = dialect.shuffle_down(
        "block_force_x", "offset", str(subgroup_width)
    )
    block_shuffle_y = dialect.shuffle_down(
        "block_force_y", "offset", str(subgroup_width)
    )
    block_shuffle_z = dialect.shuffle_down(
        "block_force_z", "offset", str(subgroup_width)
    )
    source_extent = (
        "(source_args.source_owner_count > 0"
        " ? source_args.source_owner_count : source_args.num_nodes)"
        if compact_source_owners
        else "source_args.num_nodes"
    )
    compact_source = (
        "source_args.source_ids[source_index]"
        if compact_source_ids
        else "source_args.neigh_indices[source_args.source_edges[first_scheduled]]"
    )
    source_setup = (
        f"""const bool compact_source_schedule =
            source_args.source_owner_count > 0;
        const std::int32_t schedule_owner =
            static_cast<std::int32_t>(source_index);
        const std::int32_t first_scheduled =
            source_args.source_offsets[schedule_owner];
        const std::int32_t scheduled_end =
            source_args.source_offsets[schedule_owner + 1];
        if (first_scheduled == scheduled_end) continue;
        const std::int32_t source = compact_source_schedule
            ? {compact_source}
            : static_cast<std::int32_t>(source_index);"""
        if compact_source_owners
        else """const std::int32_t source = static_cast<std::int32_t>(source_index);
        const std::int32_t first_scheduled = source_args.source_offsets[source];
        const std::int32_t scheduled_end = source_args.source_offsets[source + 1];"""
    )
    receiver_type = (
        "source_args.type_to_active[source_args.node_types[receiver]]"
        if map_type_indices
        else "source_args.node_types[receiver]"
    )
    source_type_value = (
        "source_args.type_to_active[source_args.neigh_types[edge]]"
        if map_type_indices
        else "source_args.neigh_types[edge]"
    )
    edge_type_value = (
        "source_type_value * static_cast<std::int32_t>(source_args.active_type_count)"
        " + receiver_type"
        if ordered_pair_types
        else "pair_type(\n                receiver_type, source_type_value,\n"
        "                static_cast<std::int32_t>(source_args.active_type_count))"
    )
    density_force = ""
    if density_function is not None:
        density_force = f"""
            if (threadIdx.x == 0) {{
                Scalar density_value = Scalar(0);
                Scalar density_derivative = Scalar(0);
                evaluate_radial(
                    edge_args.radial, edge_type, point, {density_function},
                    density_value, density_derivative);
                const Scalar density_force = density_derivative
                    * static_cast<const Scalar*>(source_args.node_density_adjoint)[receiver];
                local_force_x += density_force * x_over_r;
                local_force_y += density_force * y_over_r;
                local_force_z += density_force * z_over_r;
            }}"""
    source_zero = (
        f"""for (std::int32_t index = threadIdx.x;
             index < {source_harmonics * channel_tile_size};
             index += blockDim.x) {{
            const std::int32_t harmonic =
                index / {channel_tile_size};
            const std::int32_t local =
                index % {channel_tile_size};
            source_values[harmonic * {channels}
                + static_cast<std::int32_t>(source_args.channel_begin) + local] = Scalar(0);
        }}"""
        if channel_tiled
        else f"""for (std::int32_t index = threadIdx.x;
             index < {source_harmonics * channels}; index += blockDim.x)
            source_values[index] = Scalar(0);"""
    )
    channel_loop = (
        f"""for (std::int32_t local_channel = threadIdx.x;
                 local_channel < {channel_tile_size};
                 local_channel += blockDim.x) {{
                const std::int32_t channel =
                    static_cast<std::int32_t>(source_args.channel_begin) + local_channel;"""
        if channel_tiled
        else f"""for (std::int32_t channel = threadIdx.x; channel < {channels};
                 channel += blockDim.x) {{"""
    )
    source_store = (
        f"""for (std::int32_t index = threadIdx.x;
             index < {source_harmonics * channel_tile_size};
             index += blockDim.x) {{
            const std::int32_t harmonic =
                index / {channel_tile_size};
            const std::int32_t local =
                index % {channel_tile_size};
            const std::int32_t channel =
                static_cast<std::int32_t>(source_args.channel_begin) + local;
            const std::size_t target =
                (static_cast<std::size_t>(source) * {source_harmonics} + harmonic)
                    * {channels} + channel;
            source_adjoint[target] += source_values[harmonic * {channels} + channel];
        }}"""
        if channel_tiled
        else f"""for (std::int32_t index = threadIdx.x;
             index < {source_harmonics * channels}; index += blockDim.x) {{
            const std::size_t target =
                static_cast<std::size_t>(source) * {source_harmonics * channels}
                + index;
            source_adjoint[target] += source_values[index];
        }}"""
    )
    source_feature_declaration = (
        f"    __shared__ Scalar source_features[{source_harmonics * channels}];\n"
        if shared_source_features
        else ""
    )
    source_feature_load = (
        f"""for (std::int32_t index = threadIdx.x;
             index < {source_harmonics * channels}; index += blockDim.x) {{
            const std::size_t source_feature_offset =
                static_cast<std::size_t>(source) * {source_harmonics * channels};
            source_features[index] =
                neighbor_features[source_feature_offset + index];
        }}"""
        if shared_source_features
        else ""
    )
    if filter_inactive_edges:
        edge_setup = f"""const bool edge_active = edge_is_active(
                source_args.cutoff, source_args.radius[edge]);
            const std::int32_t receiver = edge_active
                ? source_args.edge_receivers[edge] : 0;
            const std::int32_t receiver_type = edge_active
                ? {receiver_type} : 0;
            const std::int32_t source_type_value = edge_active
                ? {source_type_value} : 0;
            const std::int32_t edge_type = edge_active
                ? {edge_type_value} : 0;
            const EvaluationPoint point = edge_active
                ? evaluation_point(source_args.radial, source_args.radius[edge])
                : EvaluationPoint{{0, 0, 0, 0}};"""
        coordinate_guard = "!edge_active ? Scalar(0)\n                : "
        direct_gradient_guard = " && edge_active"
        contraction_open = "            if (edge_active) {\n"
        contraction_close = "            }\n"
    else:
        edge_setup = f"""const std::int32_t receiver = source_args.edge_receivers[edge];
            const std::int32_t receiver_type = {receiver_type};
            const std::int32_t source_type_value = {source_type_value};
            const std::int32_t edge_type = {edge_type_value};
            const EvaluationPoint point = evaluation_point(
                source_args.radial, source_args.radius[edge]);"""
        coordinate_guard = ""
        direct_gradient_guard = ""
        contraction_open = ""
        contraction_close = ""
    return f"""{prefix} __launch_bounds__(reverse_threads_per_block)
void {kernel_name}({source_type} source_args, {edge_type} edge_args)
{{
    __shared__ Scalar source_values[{source_harmonics * channels}];
{source_feature_declaration.rstrip()}
    __shared__ Scalar force_x_partials[{subgroup_count}];
    __shared__ Scalar force_y_partials[{subgroup_count}];
    __shared__ Scalar force_z_partials[{subgroup_count}];
    __shared__ Scalar direct_gradients[{3 * edge_harmonics}];
    const std::int32_t lane = threadIdx.x % {subgroup_width};
    const std::int32_t subgroup = threadIdx.x / {subgroup_width};
    const auto* harmonics_values =
        static_cast<const Scalar*>(edge_args.harmonics_values);
    const auto* harmonics_gradients =
        static_cast<const Scalar*>(edge_args.harmonics_gradients);
    const auto* neighbor_features =
        static_cast<const Scalar*>(edge_args.neighbor_features);
    auto* source_adjoint = static_cast<Scalar*>(source_args.source_adjoint);
    for (std::int64_t source_index = blockIdx.x;
         source_index < {source_extent}; source_index += gridDim.x) {{
        {source_setup}
        {source_feature_load}
        {source_zero}
        __syncthreads();
        for (std::int32_t scheduled = first_scheduled;
             scheduled < scheduled_end; ++scheduled) {{
            const std::int32_t edge = source_args.source_edges[scheduled];
            {edge_setup}
            const std::size_t coordinate_offset =
                static_cast<std::size_t>(3) * edge;
            const std::size_t edge_offset =
                static_cast<std::size_t>(edge) * {edge_harmonics};
            const std::size_t gradient_offset =
                static_cast<std::size_t>(3) * edge_offset;
            const std::size_t source_offset =
                static_cast<std::size_t>(source) * {source_harmonics};
            const Scalar x_over_r = {coordinate_guard}edge_args.coordinates_are_unit != 0
                ? static_cast<Scalar>(reinterpret_cast<const float*>(edge_args.xyz)[coordinate_offset])
                : static_cast<Scalar>(reinterpret_cast<const double*>(edge_args.xyz)[coordinate_offset]
                    / edge_args.radius[edge]);
            const Scalar y_over_r = {coordinate_guard}edge_args.coordinates_are_unit != 0
                ? static_cast<Scalar>(reinterpret_cast<const float*>(edge_args.xyz)[coordinate_offset + 1])
                : static_cast<Scalar>(reinterpret_cast<const double*>(edge_args.xyz)[coordinate_offset + 1]
                    / edge_args.radius[edge]);
            const Scalar z_over_r = {coordinate_guard}edge_args.coordinates_are_unit != 0
                ? static_cast<Scalar>(reinterpret_cast<const float*>(edge_args.xyz)[coordinate_offset + 2])
                : static_cast<Scalar>(reinterpret_cast<const double*>(edge_args.xyz)[coordinate_offset + 2]
                    / edge_args.radius[edge]);
            const Scalar* edge_gradients;
            if (harmonics_gradients == nullptr) {{
                if (threadIdx.x == 0{direct_gradient_guard})
                    direct_harmonic_gradients(
                        edge_args.xyz, edge_args.coordinate_scalar_size,
                        edge_args.coordinates_are_unit, coordinate_offset,
                        edge_args.radius[edge], direct_gradients);
                __syncthreads();
                edge_gradients = direct_gradients;
            }} else {{
                edge_gradients = harmonics_gradients + gradient_offset;
            }}
            Scalar local_force_x = Scalar(0);
            Scalar local_force_y = Scalar(0);
            Scalar local_force_z = Scalar(0);
{contraction_open.rstrip()}
            {channel_loop}
{path_dispatch}
            }}{density_force}
{contraction_close.rstrip()}
            for (std::int32_t offset = {subgroup_width // 2}; offset > 0;
                 offset /= 2) {{
                local_force_x += {shuffle_x};
                local_force_y += {shuffle_y};
                local_force_z += {shuffle_z};
            }}
            if (lane == 0) {{
                force_x_partials[subgroup] = local_force_x;
                force_y_partials[subgroup] = local_force_y;
                force_z_partials[subgroup] = local_force_z;
            }}
            __syncthreads();
            if (subgroup == 0) {{
                Scalar block_force_x =
                    lane < {subgroup_count} ? force_x_partials[lane] : Scalar(0);
                Scalar block_force_y =
                    lane < {subgroup_count} ? force_y_partials[lane] : Scalar(0);
                Scalar block_force_z =
                    lane < {subgroup_count} ? force_z_partials[lane] : Scalar(0);
                for (std::int32_t offset = {subgroup_width // 2}; offset > 0;
                     offset /= 2) {{
                    block_force_x += {block_shuffle_x};
                    block_force_y += {block_shuffle_y};
                    block_force_z += {block_shuffle_z};
                }}
                if (lane == 0) {{
                    edge_args.directed_forces[coordinate_offset] -=
                        static_cast<double>(block_force_x);
                    edge_args.directed_forces[coordinate_offset + 1] -=
                        static_cast<double>(block_force_y);
                    edge_args.directed_forces[coordinate_offset + 2] -=
                        static_cast<double>(block_force_z);
                }}
            }}
            __syncthreads();
        }}
        {source_store}
        __syncthreads();
    }}
}}"""


def _render_r1_plugin_descriptor(
    metadata: dict,
    schedule: KernelSchedule,
    dialect: GpuDialect,
    plugin_type: str,
    macro: str,
) -> str:
    spec = _precision_spec(metadata["precision"])
    scalar_macro = (
        spec["cuda_scalar_macro"]
        if dialect.backend == "cuda"
        else f"SYMMETRIX_JIT_HIP_SCALAR_{metadata['precision'].upper()}_V1"
    )
    target_fields = (
        f"    {metadata['target_compute_capability']},\n"
        if dialect.backend == "cuda"
        else (
            f"    {_cpp_string(metadata['target_architecture'])},\n"
            f"    {_cpp_string(metadata['target_features'])},\n"
            f"    {metadata['native_subgroup_width']},\n"
        )
    )
    persistent_field = (
        ""
        if dialect.backend == "cuda"
        else f"    {schedule.persistent_blocks_per_compute_unit},\n"
    )
    return f"""const {plugin_type} plugin = {{
    {macro}_PLUGIN_ABI_VERSION_V{dialect.packet_version},
    sizeof({plugin_type}),
    sizeof(void*),
    {macro}_PLUGIN_BYTE_ORDER,
    {macro}_R1_FORWARD_LAUNCH_V{dialect.packet_version}
        | {macro}_R1_COORDINATE_REVERSE_LAUNCH_V{dialect.packet_version}
        | {macro}_R1_FUSED_REVERSE_LAUNCH_V{dialect.packet_version},
    {scalar_macro},
    sizeof(Scalar),
    0u,
    {_cpp_string(dialect.plugin_abi)},
    {_cpp_string(metadata["artifact_id"])},
    {_cpp_string(metadata["contract_fingerprint"])},
    {_cpp_string(metadata["semantic_fingerprint"])},
    {_cpp_string(metadata["structure_fingerprint"])},
    {metadata["channels"]},
    {metadata["embedding"]},
    {metadata["edge_l_max"]},
    {metadata["source_l_max"]},
{target_fields}    forward_threads_per_block,
    source_threads_per_block,
    edge_threads_per_block,
{persistent_field}    &r1_forward_launch,
    &r1_coordinate_reverse_launch,
}};"""


def _render_gpu_spline_helpers(
    radial_type: str,
    device_qualifier: str,
    *,
    precision_matched_coordinates: bool,
    include_active_edge_helper: bool = True,
) -> str:
    if precision_matched_coordinates:
        evaluation_scalar_cpp = "Scalar"
        evaluation_point_body = """    const Scalar local_radius =
        static_cast<Scalar>(radius);
    const Scalar x0 = static_cast<Scalar>(radial.x0);
    const Scalar h = static_cast<Scalar>(radial.h);
    std::int32_t interval = static_cast<std::int32_t>(
        floor((local_radius - x0) / h));
    Scalar x = local_radius - x0 - h * interval;
    if (interval < 0) {
        interval = 0;
        x = Scalar(0);
    } else if (interval >= static_cast<std::int32_t>(radial.intervals)) {
        interval = static_cast<std::int32_t>(radial.intervals) - 1;
        x = h;
    }
    const Scalar xx = x * x;
    return {interval, x, xx, xx * x};"""
    else:
        evaluation_scalar_cpp = "double"
        evaluation_point_body = """    std::int32_t interval = static_cast<std::int32_t>(
        floor((radius - radial.x0) / radial.h));
    double x = radius - radial.x0 - radial.h * interval;
    if (interval < 0) {
        interval = 0;
        x = 0.0;
    } else if (interval >= static_cast<std::int32_t>(radial.intervals)) {
        interval = static_cast<std::int32_t>(radial.intervals) - 1;
        x = radial.h;
    }
    const double xx = x * x;
    return {interval, x, xx, xx * x};"""
    active_edge_helper = (
        f"""{device_qualifier} bool edge_is_active(double cutoff, double radius)
{{
    return radius < cutoff;
}}

"""
        if include_active_edge_helper
        else ""
    )
    return f"""struct EvaluationPoint {{
    std::int32_t interval;
    {evaluation_scalar_cpp} x;
    {evaluation_scalar_cpp} xx;
    {evaluation_scalar_cpp} xxx;
}};

{active_edge_helper.rstrip()}
{device_qualifier} EvaluationPoint evaluation_point(
    const {radial_type}& radial, double radius)
{{
{evaluation_point_body}
}}

{device_qualifier} std::size_t radial_index(
    const {radial_type}& radial,
    std::int32_t edge_type,
    std::int32_t interval,
    std::int32_t coefficient,
    std::int32_t function)
{{
    return (((static_cast<std::size_t>(edge_type) * radial.intervals + interval)
        * 4u + coefficient) * radial.functions + function);
}}

{device_qualifier} Scalar evaluate_radial(
    const {radial_type}& radial,
    std::int32_t edge_type,
    const EvaluationPoint& point,
    std::int32_t function)
{{
    const Scalar c0 = static_cast<const Scalar*>(radial.coefficients)[radial_index(radial, edge_type, point.interval, 0, function)];
    const Scalar c1 = static_cast<const Scalar*>(radial.coefficients)[radial_index(radial, edge_type, point.interval, 1, function)];
    const Scalar c2 = static_cast<const Scalar*>(radial.coefficients)[radial_index(radial, edge_type, point.interval, 2, function)];
    const Scalar c3 = static_cast<const Scalar*>(radial.coefficients)[radial_index(radial, edge_type, point.interval, 3, function)];
    return c0 + c1 * static_cast<Scalar>(point.x)
        + c2 * static_cast<Scalar>(point.xx)
        + c3 * static_cast<Scalar>(point.xxx);
}}

{device_qualifier} void evaluate_radial(
    const {radial_type}& radial,
    std::int32_t edge_type,
    const EvaluationPoint& point,
    std::int32_t function,
    Scalar& value,
    Scalar& derivative)
{{
    const Scalar c0 = static_cast<const Scalar*>(radial.coefficients)[radial_index(radial, edge_type, point.interval, 0, function)];
    const Scalar c1 = static_cast<const Scalar*>(radial.coefficients)[radial_index(radial, edge_type, point.interval, 1, function)];
    const Scalar c2 = static_cast<const Scalar*>(radial.coefficients)[radial_index(radial, edge_type, point.interval, 2, function)];
    const Scalar c3 = static_cast<const Scalar*>(radial.coefficients)[radial_index(radial, edge_type, point.interval, 3, function)];
    value = c0 + c1 * static_cast<Scalar>(point.x)
        + c2 * static_cast<Scalar>(point.xx)
        + c3 * static_cast<Scalar>(point.xxx);
    derivative = c1 + c2 * static_cast<Scalar>(2.0 * point.x)
        + c3 * static_cast<Scalar>(3.0 * point.xx);
}}

{device_qualifier} std::int32_t pair_type(
    std::int32_t left, std::int32_t right, std::int32_t type_count)
{{
    return left <= right
        ? left * (2 * type_count - left - 1) / 2 + right
        : right * (2 * type_count - right - 1) / 2 + left;
}}"""


def _render_tiled_forward_kernel(
    *,
    args_type: str,
    kernel_name: str,
    output_components: int,
    owner_function: str,
    channel_tile_size: int,
    exported: bool,
) -> str:
    prefix = 'extern "C" __global__' if exported else "__global__"
    return f"""{prefix} __launch_bounds__(forward_threads_per_block)
void {kernel_name}({args_type} args)
{{
    auto* output = static_cast<Scalar*>(args.output);
    const std::int64_t owners =
        args.num_nodes * static_cast<std::int64_t>({channel_tile_size});
    const std::int64_t stride =
        static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    for (std::int64_t owner =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         owner < owners; owner += stride) {{
        const std::int32_t receiver = static_cast<std::int32_t>(
            owner / static_cast<std::int64_t>({channel_tile_size}));
        const std::int32_t local_channel = static_cast<std::int32_t>(
            owner % static_cast<std::int64_t>({channel_tile_size}));
        const std::int32_t channel =
            static_cast<std::int32_t>(args.channel_begin) + local_channel;
        {owner_function}(
            &args, receiver, channel,
            output + static_cast<std::size_t>(receiver)
                * {output_components * channel_tile_size});
    }}
}}"""


def _render_projected_forward_kernel(
    groups,
    *,
    args_type: str,
    kernel_name: str,
    channels: int,
    projected_harmonics: int,
    output_components: int,
    exported: bool,
) -> str:
    prefix = 'extern "C" __global__' if exported else "__global__"
    projections = []
    lme_offset = 0
    weight_offset = 0
    for l, components, multiplicity in groups:
        projections.append(
            f"""        for (std::int32_t index = threadIdx.x;
             index < {components * channels}; index += blockDim.x) {{
            const std::int32_t component = index / {channels};
            const std::int32_t output_channel = index % {channels};
            Scalar value = Scalar(0);
            for (std::int32_t eta = 0; eta < {multiplicity}; ++eta)
                for (std::int32_t input_channel = 0;
                     input_channel < {channels}; ++input_channel)
                    value += phi[
                        ({lme_offset} + component * {multiplicity} + eta)
                            * {channels} + input_channel]
                        * projection_weights[
                            {weight_offset}
                            + (eta * {channels} + input_channel) * {channels}
                            + output_channel];
            output[(receiver * {projected_harmonics} + {l * l} + component)
                * {channels} + output_channel] = value;
        }}"""
        )
        lme_offset += components * multiplicity
        weight_offset += multiplicity * channels * channels
    return f"""{prefix} __launch_bounds__(128)
void {kernel_name}({args_type} args)
{{
    extern __shared__ unsigned char shared_bytes[];
    auto* phi = reinterpret_cast<Scalar*>(shared_bytes);
    const auto* projection_weights =
        static_cast<const Scalar*>(args.projection_weights);
    auto* output = static_cast<Scalar*>(args.output);
    for (std::int64_t receiver = blockIdx.x;
         receiver < args.num_nodes; receiver += gridDim.x) {{
        for (std::int32_t channel = threadIdx.x; channel < {channels};
             channel += blockDim.x)
            r1_projected_forward_owner(
                &args, static_cast<std::int32_t>(receiver), channel, phi);
        __syncthreads();
{chr(10).join(projections)}
        __syncthreads();
    }}
}}"""


def _render_projected_reverse_kernel(
    path_rows,
    groups,
    *,
    args_type: str,
    kernel_name: str,
    channels: int,
    edge_harmonics: int,
    source_harmonics: int,
    projected_harmonics: int,
    output_components: int,
    subgroup_width: int,
    dialect: GpuDialect,
    exported: bool,
) -> str:
    prefix = 'extern "C" __global__' if exported else "__global__"
    reverse_threads = 128
    if reverse_threads % subgroup_width:
        raise ValueError("projected reverse block must be a native-subgroup multiple")
    subgroup_count = reverse_threads // subgroup_width

    reconstruction = []
    lme_offset = 0
    weight_offset = 0
    for l, components, multiplicity in groups:
        reconstruction.append(
            f"""        for (std::int32_t index = threadIdx.x;
             index < {components * multiplicity * channels};
             index += blockDim.x) {{
            const std::int32_t component =
                index / {multiplicity * channels};
            const std::int32_t remainder =
                index % {multiplicity * channels};
            const std::int32_t eta = remainder / {channels};
            const std::int32_t input_channel = remainder % {channels};
            Scalar value = Scalar(0);
            for (std::int32_t output_channel = 0;
                 output_channel < {channels}; ++output_channel)
                value += output_adjoint[
                    (receiver * {projected_harmonics} + {l * l} + component)
                        * {channels} + output_channel]
                    * projection_weights[
                        {weight_offset}
                        + (eta * {channels} + input_channel) * {channels}
                        + output_channel];
            phi_adjoint[
                ({lme_offset} + component * {multiplicity} + eta)
                    * {channels} + input_channel] = value;
        }}"""
        )
        lme_offset += components * multiplicity
        weight_offset += multiplicity * channels * channels

    path_blocks = []
    used_lm2 = sorted({row["lm2"] for rows in path_rows for row in rows})
    source_declarations = "\n".join(
        f"                Scalar source_value_{lm2} = Scalar(0);" for lm2 in used_lm2
    )
    source_stores = "\n".join(
        f"                if (source_value_{lm2} != Scalar(0))\n"
        f"                    atomicAdd(&source_adjoint[(static_cast<std::size_t>(source) * {source_harmonics} + {lm2})"
        f" * {channels} + channel], source_value_{lm2});"
        for lm2 in used_lm2
    )
    for path_index, rows in enumerate(path_rows):
        rows_by_lm1 = {}
        for row in rows:
            rows_by_lm1.setdefault(row["lm1"], []).append(row)
        statements = [
            "                {",
            f"                Scalar radial_value_{path_index} = Scalar(0);",
            f"                Scalar radial_derivative_{path_index} = Scalar(0);",
            "                evaluate_radial(",
            f"                    args.radial, edge_type, point, {path_index * channels} + channel,",
            f"                    radial_value_{path_index}, radial_derivative_{path_index});",
        ]
        for lm1, harmonic_rows in rows_by_lm1.items():
            statements.extend(
                [
                    "                {",
                    f"                const Scalar harmonic_value_{path_index}_{lm1} =",
                    f"                    harmonics_values[edge_offset + {lm1}];",
                    f"                Scalar weighted_{path_index}_{lm1} = Scalar(0);",
                ]
            )
            for row in harmonic_rows:
                row_adjoint = "\n                        + ".join(
                    f"Scalar({term['coefficient']!r})"
                    f" * phi_adjoint[{term['lme']} * {channels} + channel]"
                    for term in row["terms"]
                )
                statements.extend(
                    [
                        "                {",
                        f"                const Scalar row_adjoint_{row['row']} =",
                        f"                    {row_adjoint};",
                        f"                source_value_{row['lm2']} +=",
                        f"                    radial_value_{path_index}",
                        f"                    * harmonic_value_{path_index}_{lm1}",
                        f"                    * row_adjoint_{row['row']};",
                        f"                weighted_{path_index}_{lm1} +=",
                        (
                            f"                    neighbor_features[(source_offset + {row['lm2']})"
                            f" * {channels} + channel] * row_adjoint_{row['row']};"
                        ),
                        "                }",
                    ]
                )
            statements.extend(
                [
                    f"                local_force_x += radial_derivative_{path_index}",
                    f"                    * weighted_{path_index}_{lm1}",
                    f"                    * harmonic_value_{path_index}_{lm1} * x_over_r;",
                    f"                local_force_y += radial_derivative_{path_index}",
                    f"                    * weighted_{path_index}_{lm1}",
                    f"                    * harmonic_value_{path_index}_{lm1} * y_over_r;",
                    f"                local_force_z += radial_derivative_{path_index}",
                    f"                    * weighted_{path_index}_{lm1}",
                    f"                    * harmonic_value_{path_index}_{lm1} * z_over_r;",
                    f"                const Scalar angular_{path_index}_{lm1} =",
                    f"                    radial_value_{path_index} * weighted_{path_index}_{lm1};",
                    f"                local_force_x += angular_{path_index}_{lm1}",
                    f"                    * harmonics_gradients[gradient_offset + {lm1}];",
                    f"                local_force_y += angular_{path_index}_{lm1}",
                    f"                    * harmonics_gradients[gradient_offset + {edge_harmonics + lm1}];",
                    f"                local_force_z += angular_{path_index}_{lm1}",
                    f"                    * harmonics_gradients[gradient_offset + {2 * edge_harmonics + lm1}];",
                    "                }",
                ]
            )
        statements.append("                }")
        path_blocks.append("\n".join(statements))

    shuffle_x = dialect.shuffle_down("local_force_x", "offset", str(subgroup_width))
    shuffle_y = dialect.shuffle_down("local_force_y", "offset", str(subgroup_width))
    shuffle_z = dialect.shuffle_down("local_force_z", "offset", str(subgroup_width))
    block_shuffle_x = dialect.shuffle_down(
        "block_force_x", "offset", str(subgroup_width)
    )
    block_shuffle_y = dialect.shuffle_down(
        "block_force_y", "offset", str(subgroup_width)
    )
    block_shuffle_z = dialect.shuffle_down(
        "block_force_z", "offset", str(subgroup_width)
    )
    return f"""{prefix} __launch_bounds__(128)
void {kernel_name}({args_type} args)
{{
    extern __shared__ unsigned char shared_bytes[];
    auto* phi_adjoint = reinterpret_cast<Scalar*>(shared_bytes);
    __shared__ Scalar force_x_partials[{subgroup_count}];
    __shared__ Scalar force_y_partials[{subgroup_count}];
    __shared__ Scalar force_z_partials[{subgroup_count}];
    const std::int32_t lane = threadIdx.x % {subgroup_width};
    const std::int32_t subgroup = threadIdx.x / {subgroup_width};
    const auto* projection_weights =
        static_cast<const Scalar*>(args.projection_weights);
    const auto* output_adjoint =
        static_cast<const Scalar*>(args.output_adjoint);
    const auto* harmonics_values =
        static_cast<const Scalar*>(args.harmonics_values);
    const auto* harmonics_gradients =
        static_cast<const Scalar*>(args.harmonics_gradients);
    const auto* neighbor_features =
        static_cast<const Scalar*>(args.neighbor_features);
    auto* source_adjoint = static_cast<Scalar*>(args.source_adjoint);
    for (std::int64_t receiver = blockIdx.x;
         receiver < args.num_nodes; receiver += gridDim.x) {{
{chr(10).join(reconstruction)}
        __syncthreads();
        const std::int32_t receiver_type =
            args.type_to_active[args.node_types[receiver]];
        const std::int32_t edge_begin = args.first_neigh[receiver];
        const std::int32_t edge_end = edge_begin + args.num_neigh[receiver];
        for (std::int32_t edge = edge_begin; edge < edge_end; ++edge) {{
            const bool edge_active = edge_is_active(
                args.cutoff, args.radius[edge]);
            const std::int32_t source = edge_active
                ? args.neigh_indices[edge] : 0;
            const std::int32_t source_type = edge_active
                ? args.type_to_active[args.neigh_types[edge]] : 0;
            const std::int32_t edge_type = edge_active
                ? pair_type(receiver_type, source_type,
                    static_cast<std::int32_t>(args.active_type_count))
                : 0;
            const EvaluationPoint point = edge_active
                ? evaluation_point(args.radial, args.radius[edge])
                : EvaluationPoint{{0, 0, 0, 0}};
            const std::size_t coordinate_offset = static_cast<std::size_t>(3) * edge;
            const std::size_t edge_offset = static_cast<std::size_t>(edge) * {edge_harmonics};
            const std::size_t gradient_offset = static_cast<std::size_t>(3) * edge_offset;
            const std::size_t source_offset = static_cast<std::size_t>(source) * {source_harmonics};
            const Scalar x_over_r = !edge_active ? Scalar(0)
                : args.coordinates_are_unit != 0
                ? static_cast<Scalar>(reinterpret_cast<const float*>(args.xyz)[coordinate_offset])
                : static_cast<Scalar>(reinterpret_cast<const double*>(args.xyz)[coordinate_offset]
                    / args.radius[edge]);
            const Scalar y_over_r = !edge_active ? Scalar(0)
                : args.coordinates_are_unit != 0
                ? static_cast<Scalar>(reinterpret_cast<const float*>(args.xyz)[coordinate_offset + 1])
                : static_cast<Scalar>(reinterpret_cast<const double*>(args.xyz)[coordinate_offset + 1]
                    / args.radius[edge]);
            const Scalar z_over_r = !edge_active ? Scalar(0)
                : args.coordinates_are_unit != 0
                ? static_cast<Scalar>(reinterpret_cast<const float*>(args.xyz)[coordinate_offset + 2])
                : static_cast<Scalar>(reinterpret_cast<const double*>(args.xyz)[coordinate_offset + 2]
                    / args.radius[edge]);
            Scalar local_force_x = Scalar(0);
            Scalar local_force_y = Scalar(0);
            Scalar local_force_z = Scalar(0);
            if (edge_active) {{
            for (std::int32_t channel = threadIdx.x; channel < {channels};
                 channel += blockDim.x) {{
{source_declarations}
{chr(10).join(path_blocks)}
{source_stores}
            }}
            }}
            for (std::int32_t offset = {subgroup_width // 2}; offset > 0;
                 offset /= 2) {{
                local_force_x += {shuffle_x};
                local_force_y += {shuffle_y};
                local_force_z += {shuffle_z};
            }}
            if (lane == 0) {{
                force_x_partials[subgroup] = local_force_x;
                force_y_partials[subgroup] = local_force_y;
                force_z_partials[subgroup] = local_force_z;
            }}
            __syncthreads();
            if (subgroup == 0) {{
                Scalar block_force_x = lane < {subgroup_count}
                    ? force_x_partials[lane] : Scalar(0);
                Scalar block_force_y = lane < {subgroup_count}
                    ? force_y_partials[lane] : Scalar(0);
                Scalar block_force_z = lane < {subgroup_count}
                    ? force_z_partials[lane] : Scalar(0);
                for (std::int32_t offset = {subgroup_width // 2}; offset > 0;
                     offset /= 2) {{
                    block_force_x += {block_shuffle_x};
                    block_force_y += {block_shuffle_y};
                    block_force_z += {block_shuffle_z};
                }}
                if (lane == 0) {{
                    args.directed_forces[coordinate_offset] -=
                        static_cast<double>(block_force_x);
                    args.directed_forces[coordinate_offset + 1] -=
                        static_cast<double>(block_force_y);
                    args.directed_forces[coordinate_offset + 2] -=
                        static_cast<double>(block_force_z);
                }}
            }}
            __syncthreads();
        }}
        __syncthreads();
    }}
}}"""


def _render_direct_harmonic_gradient_helper(
    device_qualifier: str,
    edge_harmonics: int,
    *,
    index_type: str = "std::int32_t",
) -> str:
    if edge_harmonics not in (1, 4, 9, 16):
        raise ValueError("direct harmonic gradients require l_max from 0 through 3")
    return f"""{device_qualifier} void direct_harmonic_gradients(
    const void* coordinates,
    unsigned int coordinate_scalar_size,
    unsigned int coordinates_are_unit,
    std::size_t coordinate_offset,
    double radius,
    Scalar* gradients)
{{
    Scalar ux;
    Scalar uy;
    Scalar uz;
    if (coordinates_are_unit != 0) {{
        using CoordinateFloat = decltype(0.0f);
        if (coordinate_scalar_size == sizeof(CoordinateFloat)) {{
            const auto* values = static_cast<const CoordinateFloat*>(coordinates);
            ux = static_cast<Scalar>(values[coordinate_offset]);
            uy = static_cast<Scalar>(values[coordinate_offset + 1]);
            uz = static_cast<Scalar>(values[coordinate_offset + 2]);
        }} else {{
            const auto* values = static_cast<const double*>(coordinates);
            ux = static_cast<Scalar>(values[coordinate_offset]);
            uy = static_cast<Scalar>(values[coordinate_offset + 1]);
            uz = static_cast<Scalar>(values[coordinate_offset + 2]);
        }}
    }} else {{
        const auto* values = static_cast<const double*>(coordinates);
        const Scalar inverse_radius = Scalar(1)/static_cast<Scalar>(radius);
        ux = static_cast<Scalar>(values[coordinate_offset])*inverse_radius;
        uy = static_cast<Scalar>(values[coordinate_offset + 1])*inverse_radius;
        uz = static_cast<Scalar>(values[coordinate_offset + 2])*inverse_radius;
    }}
    const Scalar inverse_norm = Scalar(1)/sqrt(ux*ux + uy*uy + uz*uz);
    const Scalar x = uz*inverse_norm;
    const Scalar y = ux*inverse_norm;
    const Scalar z = uy*inverse_norm;
    const Scalar x2 = x*x;
    const Scalar y2 = y*y;
    const Scalar z2 = z*z;
    Scalar sph[16];
    Scalar dx[16];
    Scalar dy[16];
    Scalar dz[16];
    sph[0] = Scalar(0.282094791773878);
    sph[1] = Scalar(0.48860251190292)*y;
    sph[2] = Scalar(0.48860251190292)*z;
    sph[3] = Scalar(0.48860251190292)*x;
    const Scalar l2_tmp = Scalar(2.23606797749979)*x;
    sph[4] = l2_tmp*sph[1];
    sph[7] = l2_tmp*sph[2];
    sph[5] = Scalar(2.23606797749979)*z*sph[1];
    sph[6] = -Scalar(0.315391565252520)*(x2+y2-Scalar(2)*z2);
    sph[8] = Scalar(0.54627421529604)*(x2-y2);
    sph[9] = -Scalar(0.59004358992664)*y*(y2-Scalar(3)*x2);
    sph[10] = Scalar(2.64575131106459)*z*sph[4];
    const Scalar l3_tmp =
        -Scalar(0.457045799464466)*(x2+y2-Scalar(4)*z2);
    sph[11] = y*l3_tmp;
    sph[13] = x*l3_tmp;
    sph[12] = -Scalar(1.49270533036046)*z
        *(z2-Scalar(2.37799637856361)*sph[6]);
    sph[14] = Scalar(1.44530572132028)*z*(x2-y2);
    sph[15] = Scalar(0.59004358992664)*x*(x2-Scalar(3)*y2);
    dx[0] = dy[0] = dz[0] = Scalar(0);
    dx[1] = dx[2] = Scalar(0);
    dx[3] = Scalar(0.48860251190292);
    dy[1] = Scalar(0.48860251190292);
    dy[2] = dy[3] = Scalar(0);
    dz[1] = Scalar(0);
    dz[2] = Scalar(0.48860251190292);
    dz[3] = Scalar(0);
    dx[4] = Scalar(2.23606797749979)*sph[1];
    dx[5] = Scalar(0);
    dx[6] = -Scalar(1.29099444873581)*sph[3];
    dx[7] = Scalar(2.23606797749979)*sph[2];
    dx[8] = Scalar(2.23606797749979)*sph[3];
    dy[4] = -Scalar(1.73205080756888)*dx[6];
    dy[5] = dx[7];
    dy[6] = -Scalar(0.577350269189626)*dx[4];
    dy[7] = Scalar(0);
    dy[8] = -dx[4];
    dz[4] = dz[8] = Scalar(0);
    dz[5] = dx[4];
    dz[6] = Scalar(1.15470053837925)*dx[7];
    dz[7] = dy[4];
    dx[9] = Scalar(3.24037034920393)*sph[4];
    dx[10] = Scalar(2.64575131106459)*sph[5];
    dx[11] = -Scalar(0.83666002653408)*sph[4];
    dx[12] = -Scalar(2.04939015319192)*sph[7];
    dx[13] = Scalar(0.91409159892893)
        *(y2-z2+Scalar(4.75599275712721)*sph[6]);
    dx[14] = Scalar(2.64575131106459)*sph[7];
    dx[15] = Scalar(3.24037034920393)*sph[8];
    dy[9] = dx[15];
    dy[10] = dx[14];
    dy[11] = -Scalar(0.91409159892893)
        *(y2-z2-Scalar(1.58533091904240)*sph[6]);
    dy[12] = -Scalar(2.04939015319192)*sph[5];
    dy[13] = -Scalar(0.83666002653408)*sph[4];
    dy[14] = -dx[10];
    dy[15] = -dx[9];
    dz[9] = Scalar(0);
    dz[10] = Scalar(2.64575131106459)*sph[4];
    dz[11] = Scalar(3.34664010613630)*sph[5];
    dz[12] = Scalar(3.54964786985977)*sph[6];
    dz[13] = Scalar(3.34664010613630)*sph[7];
    dz[14] = Scalar(2.64575131106459)*sph[8];
    dz[15] = Scalar(0);
    const Scalar scale = Scalar(3.5449077018110318)
        *inverse_norm/static_cast<Scalar>(radius);
    for ({index_type} lm = 0; lm < {edge_harmonics}; ++lm) {{
        const Scalar radial = dx[lm]*x + dy[lm]*y + dz[lm]*z;
        const Scalar gx = (dx[lm]-x*radial)*scale;
        const Scalar gy = (dy[lm]-y*radial)*scale;
        const Scalar gz = (dz[lm]-z*radial)*scale;
        gradients[lm] = gy;
        gradients[{edge_harmonics}+lm] = gz;
        gradients[{2 * edge_harmonics}+lm] = gx;
    }}
}}"""


def _render_jit_r1_gpu_artifact(
    program: R1Program,
    metadata: dict,
    schedule: KernelSchedule,
    dialect: GpuDialect,
    *,
    artifact_kind: Literal["plugin", "module"] = "plugin",
    launch_plan: LaunchPlan | None = None,
) -> str:
    """Render a GPU plugin or RTC module directly from typed inputs."""

    normalized = program.contract
    path_rows = _fixed_weight_path_rows(normalized)
    groups = _contract_groups(normalized)
    channels = metadata["channels"]
    edge_harmonics = (metadata["edge_l_max"] + 1) ** 2
    source_harmonics = (metadata["source_l_max"] + 1) ** 2
    projected_harmonics = (max(group[0] for group in groups) + 1) ** 2
    output_components = metadata["output_components"]
    scalar_cpp = _precision_spec(program.precision)["cpp"]
    shared_source_features = (
        dialect.backend == "cuda"
        and program.precision == "float32"
        and channels == 128
        and source_harmonics == 4
        and edge_harmonics == 16
    )
    device_qualifier = "__device__ __attribute__((always_inline)) inline"
    packet = dialect.packet_prefix
    version = dialect.packet_version
    radial_type = f"{packet}RadialSplineV{version}"
    forward_type = f"{packet}R1ForwardArgsV{version}"
    source_type = f"{packet}R1SourceArgsV{version}"
    edge_type = f"{packet}R1EdgeArgsV{version}"
    tiled_forward_type = f"{packet}R1TiledForwardArgsV{version}"
    tiled_source_type = f"{packet}R1TiledSourceArgsV{version}"
    tiled_edge_type = f"{packet}R1TiledEdgeArgsV{version}"
    projected_forward_type = f"{packet}R1ProjectedForwardArgsV{version}"
    projected_reverse_type = f"{packet}R1ProjectedReverseArgsV{version}"
    plugin_type = f"{packet}PluginV{version}"
    macro = f"SYMMETRIX_JIT_{dialect.backend.upper()}"
    abi_header = (
        _CUDA_PLUGIN_ABI_HEADER if dialect.backend == "cuda" else _HIP_PLUGIN_ABI_HEADER
    )
    exported = artifact_kind == "module"
    spline_helpers = _render_gpu_spline_helpers(
        radial_type,
        device_qualifier,
        precision_matched_coordinates=(
            dialect.backend == "hip" and program.precision == "float32"
        ),
    )
    harmonic_helpers = _render_direct_harmonic_gradient_helper(
        device_qualifier, edge_harmonics
    )
    if exported:
        if launch_plan is None:
            raise ValueError("module rendering requires a launch plan")
        packet_header = (
            _CUDA_MODULE_PACKET_HEADER
            if dialect.backend == "cuda"
            else _HIP_MODULE_PACKET_HEADER
        )
        preamble = packet_header
        forward_kernel, source_kernel, edge_kernel = launch_plan.exported_symbols
        fused_reverse_kernel = launch_plan.fused_reverse_symbol
        tiled_forward_kernel = f"symmetrix_factorized_tiled_forward_v{version}"
        tiled_reverse_kernel = f"symmetrix_factorized_tiled_reverse_v{version}"
        projected_forward_kernel = f"symmetrix_factorized_projected_forward_v{version}"
        projected_reverse_kernel = f"symmetrix_factorized_projected_reverse_v{version}"
        kernel_namespace_end = "\n} // namespace\n"
    else:
        preamble = f"{abi_header}\n\n#include <{dialect.runtime_header}>\n\n#include <cmath>\n#include <cstddef>\n#include <cstdint>"
        forward_kernel, source_kernel, edge_kernel = (
            "r1_forward_kernel",
            "r1_source_kernel",
            "r1_edge_kernel",
        )
        fused_reverse_kernel = "r1_reverse_fused_kernel"
        tiled_forward_kernel = ""
        tiled_reverse_kernel = ""
        projected_forward_kernel = ""
        projected_reverse_kernel = ""
        kernel_namespace_end = ""
    row_loaders = _render_row_loaders(
        path_rows,
        channels,
        output_components,
        source_args_type=source_type,
        edge_args_type=edge_type,
        qualifier=device_qualifier,
        scalar_type="Scalar",
        opaque_pointers=True,
    )
    forward_owner = _render_forward_owner(
        path_rows,
        channels=channels,
        edge_harmonics=edge_harmonics,
        source_harmonics=source_harmonics,
        output_components=output_components,
        args_type=forward_type,
        qualifier=device_qualifier,
        scalar_type="Scalar",
        opaque_pointers=True,
    )
    projected_forward_owner = ""
    tiled_forward_owner = ""
    if exported:
        tiled_forward_owner = _render_forward_owner(
            path_rows,
            channels=channels,
            edge_harmonics=edge_harmonics,
            source_harmonics=source_harmonics,
            output_components=output_components,
            args_type=tiled_forward_type,
            qualifier=device_qualifier,
            scalar_type="Scalar",
            opaque_pointers=True,
            function_name="r1_tiled_forward_owner",
            output_channel_stride="64",
            output_channel_index=(
                "channel - static_cast<std::int32_t>(args->channel_begin)"
            ),
        )
        projected_forward_owner = _render_forward_owner(
            path_rows,
            channels=channels,
            edge_harmonics=edge_harmonics,
            source_harmonics=source_harmonics,
            output_components=output_components,
            args_type=projected_forward_type,
            qualifier=device_qualifier,
            scalar_type="Scalar",
            opaque_pointers=True,
            function_name="r1_projected_forward_owner",
        )
    source_owner = _render_source_owner(
        path_rows,
        channels=channels,
        edge_harmonics=edge_harmonics,
        source_harmonics=source_harmonics,
        args_type=source_type,
        qualifier=device_qualifier,
        scalar_type="Scalar",
        opaque_pointers=True,
    )
    if schedule.edge_strategy == "wave":
        edge_owner = _render_wave_edge_owner(
            path_rows,
            args_type=edge_type,
            channels=channels,
            edge_harmonics=edge_harmonics,
            source_harmonics=source_harmonics,
            subgroup_width=schedule.edge_logical_subgroup_width,
            dialect=dialect,
            qualifier=device_qualifier,
        )
    else:
        edge_owner = _render_edge_owner(
            path_rows,
            channels=channels,
            edge_harmonics=edge_harmonics,
            source_harmonics=source_harmonics,
            args_type=edge_type,
            qualifier=device_qualifier,
            scalar_type="Scalar",
            opaque_pointers=True,
        )
    source = f"""// Generated by symmetrix.jit_codegen. Do not edit.
{preamble}

namespace {{

using Scalar = {scalar_cpp};

constexpr std::int32_t forward_threads_per_block =
    {schedule.forward_threads_per_block};
constexpr std::int32_t source_threads_per_block =
    {schedule.source_threads_per_block};
constexpr std::int32_t edge_threads_per_block =
    {schedule.edge_threads_per_block};
constexpr std::int32_t reverse_threads_per_block =
    64;

{spline_helpers}

{harmonic_helpers}

{row_loaders}

{forward_owner}

{tiled_forward_owner}

{projected_forward_owner}

{source_owner}

{edge_owner}

{kernel_namespace_end}{'extern "C" ' if exported else ""}__global__ void {
        forward_kernel
    }({forward_type} args)
{{
    const std::int64_t owner_count = args.num_nodes * {channels};
    const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    for (std::int64_t owner =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         owner < owner_count; owner += stride) {{
        r1_forward_owner(
            &args,
            static_cast<std::int32_t>(owner / {channels}),
            static_cast<std::int32_t>(owner % {channels}),
            static_cast<Scalar*>(args.output)
                + (owner / {channels}) * {output_components * channels});
    }}
}}

{'extern "C" ' if exported else ""}__global__ void {source_kernel}({source_type} args)
{{
    const std::int64_t owner_count = args.num_nodes * {channels};
    const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    for (std::int64_t owner =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         owner < owner_count; owner += stride) {{
        r1_compensated_source_owner(
            &args,
            static_cast<std::int32_t>(owner / {channels}),
            static_cast<std::int32_t>(owner % {channels}));
    }}
}}

{_render_r1_edge_kernel(edge_type, schedule, edge_kernel, exported=exported)}

{
        _render_r1_fused_reverse_kernel(
            path_rows,
            source_type=source_type,
            edge_type=edge_type,
            kernel_name=fused_reverse_kernel,
            channels=channels,
            edge_harmonics=edge_harmonics,
            source_harmonics=source_harmonics,
            output_components=output_components,
            subgroup_width=metadata.get("native_subgroup_width", 32),
            dialect=dialect,
            exported=exported,
            shared_source_features=shared_source_features,
        )
    }
{
        (
            _render_tiled_forward_kernel(
                args_type=tiled_forward_type,
                kernel_name=tiled_forward_kernel,
                output_components=output_components,
                owner_function="r1_tiled_forward_owner",
                channel_tile_size=64,
                exported=True,
            )
            + chr(10)
            + chr(10)
            + _render_r1_fused_reverse_kernel(
                path_rows,
                source_type=tiled_source_type,
                edge_type=tiled_edge_type,
                kernel_name=tiled_reverse_kernel,
                channels=channels,
                edge_harmonics=edge_harmonics,
                source_harmonics=source_harmonics,
                output_components=output_components,
                subgroup_width=metadata.get("native_subgroup_width", 32),
                dialect=dialect,
                exported=True,
                channel_tiled=True,
                channel_tile_size=64,
                compact_source_owners=True,
                compact_source_ids=True,
                shared_source_features=shared_source_features,
            )
        )
        if exported
        else ""
    }
{
        (
            _render_projected_forward_kernel(
                groups,
                args_type=projected_forward_type,
                kernel_name=projected_forward_kernel,
                channels=channels,
                projected_harmonics=projected_harmonics,
                output_components=output_components,
                exported=True,
            )
            + chr(10)
            + _render_projected_reverse_kernel(
                path_rows,
                groups,
                args_type=projected_reverse_type,
                kernel_name=projected_reverse_kernel,
                channels=channels,
                edge_harmonics=edge_harmonics,
                source_harmonics=source_harmonics,
                projected_harmonics=projected_harmonics,
                output_components=output_components,
                subgroup_width=metadata.get("native_subgroup_width", 32),
                dialect=dialect,
                exported=True,
            )
        )
        if exported
        else ""
    }
"""
    if exported:
        constants = f"""
extern "C" __device__ __constant__ char
symmetrix_factorized_contract_fingerprint[] =
    {_cpp_string(metadata["contract_fingerprint"])};
extern "C" __device__ __constant__ int
symmetrix_factorized_forward_threads_per_block =
    {schedule.forward_threads_per_block};
extern "C" __device__ __constant__ int
symmetrix_factorized_source_threads_per_block =
    {schedule.source_threads_per_block};
extern "C" __device__ __constant__ int
symmetrix_factorized_edge_threads_per_block =
    {schedule.edge_threads_per_block};
extern "C" __device__ __constant__ int
symmetrix_factorized_reverse_threads_per_block =
    64;
extern "C" __device__ __constant__ int
symmetrix_factorized_reverse_physical_launch_count = 1;
extern "C" __device__ __constant__ int
symmetrix_factorized_tiled_forward_threads_per_block =
    {schedule.forward_threads_per_block};
extern "C" __device__ __constant__ int
symmetrix_factorized_tiled_reverse_threads_per_block = 64;
extern "C" __device__ __constant__ int
symmetrix_factorized_projected_threads_per_block = 128;
extern "C" __device__ __constant__ int
symmetrix_factorized_projected_shared_memory_bytes =
    {output_components * channels * metadata["scalar_size"]};
extern "C" __device__ __constant__ int
symmetrix_factorized_persistent_blocks_per_compute_unit =
    {schedule.persistent_blocks_per_compute_unit};
"""
        if dialect.backend == "hip":
            constants += f"""extern "C" __device__ __constant__ int
symmetrix_factorized_edge_logical_subgroup_width =
    {schedule.edge_logical_subgroup_width};
"""
        return source + constants
    source += f"""

inline std::int32_t launch_blocks(
    std::int64_t work_items,
    std::int32_t threads_per_block,
    std::int32_t persistent_blocks)
{{
    const std::int64_t required =
        work_items / threads_per_block
        + (work_items % threads_per_block != 0);
    return static_cast<std::int32_t>(
        required < persistent_blocks ? required : persistent_blocks);
}}

std::int32_t r1_forward_launch(
    const {forward_type}* args,
    void* stream,
    std::int32_t persistent_blocks)
{{
    if (args == nullptr
        || args->struct_size < sizeof({forward_type})
        || args->radial.struct_size < sizeof({radial_type})
        || args->num_nodes < 0 || args->num_nodes > INT32_MAX
        || args->num_edges < 0 || args->num_edges > INT32_MAX
        || !std::isfinite(args->cutoff) || args->cutoff <= 0.0
        || args->active_type_count == 0 || persistent_blocks <= 0)
        return static_cast<std::int32_t>({dialect.error_prefix}ErrorInvalidValue);
    if (args->num_nodes == 0)
        return static_cast<std::int32_t>({dialect.error_prefix}Success);
    const std::int64_t owner_count = args->num_nodes * {channels};
    const std::int32_t blocks = launch_blocks(
        owner_count, forward_threads_per_block, persistent_blocks);
    const {forward_type} launch_args = *args;
    r1_forward_kernel<<<
        blocks, forward_threads_per_block, 0,
        reinterpret_cast<{dialect.error_prefix}Stream_t>(stream)>>>(launch_args);
    return static_cast<std::int32_t>({dialect.error_prefix}PeekAtLastError());
}}

std::int32_t r1_coordinate_reverse_launch(
    const {source_type}* source_args,
    const {edge_type}* edge_args,
    void* stream,
    std::int32_t persistent_blocks)
{{
    if (source_args == nullptr || edge_args == nullptr
        || source_args->struct_size < sizeof({source_type})
        || edge_args->struct_size < sizeof({edge_type})
        || source_args->radial.struct_size
            < sizeof({radial_type})
        || edge_args->radial.struct_size
            < sizeof({radial_type})
        || source_args->num_nodes < 0 || source_args->num_nodes > INT32_MAX
        || source_args->num_edges < 0 || source_args->num_edges > INT32_MAX
        || source_args->num_edges != edge_args->num_edges
        || source_args->active_type_count != edge_args->active_type_count
        || source_args->cutoff != edge_args->cutoff
        || !std::isfinite(source_args->cutoff) || source_args->cutoff <= 0.0
        || source_args->active_type_count == 0 || persistent_blocks <= 0)
        return static_cast<std::int32_t>({dialect.error_prefix}ErrorInvalidValue);
    if (source_args->num_nodes == 0 || source_args->num_edges == 0)
        return static_cast<std::int32_t>({dialect.error_prefix}Success);

    const auto gpu_stream = reinterpret_cast<{dialect.error_prefix}Stream_t>(stream);
    const std::int32_t reverse_blocks = static_cast<std::int32_t>(
        source_args->num_nodes < persistent_blocks
            ? source_args->num_nodes : persistent_blocks);
    const {source_type} source_launch_args = *source_args;
    const {edge_type} edge_launch_args = *edge_args;
    r1_reverse_fused_kernel<<<
        reverse_blocks, reverse_threads_per_block, 0, gpu_stream>>>(
            source_launch_args, edge_launch_args);
    return static_cast<std::int32_t>({dialect.error_prefix}PeekAtLastError());
}}

{_render_r1_plugin_descriptor(metadata, schedule, dialect, plugin_type, macro)}

}} // namespace

extern "C" {dialect.export_macro}
const {plugin_type}* {dialect.query_symbol}()
{{
    return &plugin;
}}
"""
    return source


def render_jit_r1_cuda_plugin(
    contract: dict,
    compute_capability: int,
    artifact_id: str | None = None,
    *,
    precision: str = "float32",
    edge_strategy: str = "wave",
    edge_logical_subgroup_width: int | None = 32,
    edge_threads_per_block: int = 32,
    persistent_blocks_per_compute_unit: int = 8,
) -> str:
    """Render a standalone raw-CUDA plugin for one normalized R1 contract."""

    metadata = jit_r1_cuda_plugin_metadata(
        contract,
        compute_capability,
        artifact_id,
        precision=precision,
        edge_strategy=edge_strategy,
        edge_logical_subgroup_width=edge_logical_subgroup_width,
        edge_threads_per_block=edge_threads_per_block,
        persistent_blocks_per_compute_unit=persistent_blocks_per_compute_unit,
    )
    program, _, schedule, _ = _cuda_codegen_inputs(
        contract,
        compute_capability,
        precision,
        "plugin",
        edge_strategy,
        edge_logical_subgroup_width,
        edge_threads_per_block,
        persistent_blocks_per_compute_unit,
    )
    return _render_jit_r1_gpu_artifact(program, metadata, schedule, CUDA_DIALECT)


def render_jit_r1_cuda_module(
    contract: dict,
    compute_capability: int,
    artifact_id: str | None = None,
    *,
    precision: str = "float32",
    edge_strategy: str = "wave",
    edge_logical_subgroup_width: int | None = 32,
    edge_threads_per_block: int = 32,
    persistent_blocks_per_compute_unit: int = 8,
) -> str:
    """Render device-only R1 source for NVRTC cubin compilation."""

    metadata = jit_r1_cuda_plugin_metadata(
        contract,
        compute_capability,
        artifact_id,
        precision=precision,
        edge_strategy=edge_strategy,
        edge_logical_subgroup_width=edge_logical_subgroup_width,
        edge_threads_per_block=edge_threads_per_block,
        persistent_blocks_per_compute_unit=persistent_blocks_per_compute_unit,
    )
    program, _, schedule, launch_plan = _cuda_codegen_inputs(
        contract,
        compute_capability,
        precision,
        "module",
        edge_strategy,
        edge_logical_subgroup_width,
        edge_threads_per_block,
        persistent_blocks_per_compute_unit,
    )
    return _render_jit_r1_gpu_artifact(
        program,
        metadata,
        schedule,
        CUDA_DIALECT,
        artifact_kind="module",
        launch_plan=launch_plan,
    )


def _hip_target_metadata(target: str | dict) -> dict:
    if isinstance(target, str):
        raw = target
        mapped_features = None
        subgroup = 64
    elif isinstance(target, dict):
        raw = target.get("raw_agent_target", target.get("architecture"))
        mapped_features = target.get("target_features")
        subgroup = target.get("native_subgroup_width", 64)
    else:
        raise TypeError("HIP target must be an agent string or mapping")
    if not isinstance(raw, str) or not re.fullmatch(
        r"gfx[0-9a-f]+(?::[a-z][a-z0-9_]*[+-])*", raw.lower()
    ):
        raise ValueError("HIP target must look like 'gfxNNN[:feature...]'")
    parts = raw.lower().split(":")
    architecture = parts[0]
    raw_features = parts[1:]
    if mapped_features is None:
        features = raw_features
    elif isinstance(mapped_features, str):
        features = [item for item in mapped_features.split(":") if item]
    elif isinstance(mapped_features, (list, tuple)):
        features = [str(item) for item in mapped_features]
    else:
        raise ValueError("HIP target_features must be a string or sequence")
    if any(not re.fullmatch(r"[a-z][a-z0-9_]*[+-]", item) for item in features):
        raise ValueError("HIP target feature must use a name followed by + or -")
    features = sorted(set(features))
    if raw_features and sorted(raw_features) != features:
        raise ValueError("HIP raw agent and target features disagree")
    if not isinstance(subgroup, int) or isinstance(subgroup, bool) or subgroup <= 0:
        raise ValueError("HIP native subgroup width must be a positive integer")
    return {
        "raw_agent_target": raw.lower(),
        "target_architecture": architecture,
        "target_features": ":".join(features),
        "native_subgroup_width": subgroup,
    }


def factorized_hip_plugin_metadata(
    contract: dict,
    target: str | dict,
    artifact_id: str | None = None,
    *,
    precision: str = "float32",
    edge_strategy: str = "wave",
    edge_threads_per_block: int = 64,
    persistent_blocks_per_compute_unit: int = 8,
) -> dict:
    """Return exact ABI, model, and AMD target metadata for one HIP plugin."""

    program = R1Program.from_contract(
        contract, precision, packet_layout_version=HIP_DIALECT.packet_version
    )
    metadata = jit_r1_host_plugin_metadata(
        program.contract, artifact_id, precision=precision
    )
    metadata.update(_hip_target_metadata(target))
    if edge_strategy not in ("wave", "serial"):
        raise ValueError("HIP R1 edge strategy must be 'wave' or 'serial'")
    subgroup_width = metadata["native_subgroup_width"]
    if (
        isinstance(edge_threads_per_block, bool)
        or not isinstance(edge_threads_per_block, int)
        or edge_threads_per_block < 64
        or edge_threads_per_block > 1024
        or edge_threads_per_block & (edge_threads_per_block - 1)
    ):
        raise ValueError(
            "HIP R1 edge threads per block must be a power of two in [64, 1024]"
        )
    if edge_threads_per_block % subgroup_width:
        raise ValueError("HIP R1 edge threads per block must be a subgroup multiple")
    if (
        isinstance(persistent_blocks_per_compute_unit, bool)
        or not isinstance(persistent_blocks_per_compute_unit, int)
        or persistent_blocks_per_compute_unit < 1
        or persistent_blocks_per_compute_unit > 32
    ):
        raise ValueError("HIP R1 persistent blocks per compute unit must be in [1, 32]")
    if edge_strategy == "wave" and (
        subgroup_width > 64 or subgroup_width & (subgroup_width - 1)
    ):
        raise ValueError(
            "HIP R1 wave edge strategy requires a power-of-two subgroup up to 64"
        )
    metadata.update(
        {
            "abi": HIP_PLUGIN_ABI,
            "abi_version": HIP_PLUGIN_ABI_VERSION,
            "forward_threads_per_block": 256,
            "source_threads_per_block": 256,
            "edge_threads_per_block": edge_threads_per_block,
            "reverse_threads_per_block": 64,
            "reverse_shared_memory_bytes": (
                (metadata["source_l_max"] + 1) ** 2
                * metadata["channels"]
                * metadata["scalar_size"]
            ),
            "persistent_blocks_per_compute_unit": persistent_blocks_per_compute_unit,
            "reverse_strategy": "fused_source_edge",
            "edge_strategy": edge_strategy,
            "edge_logical_subgroup_width": (
                subgroup_width if edge_strategy == "wave" else 1
            ),
        }
    )
    program, gpu_target, schedule, launch_plan = _hip_codegen_inputs(
        program.contract,
        metadata,
        precision,
        "plugin",
        edge_strategy,
        edge_threads_per_block,
        persistent_blocks_per_compute_unit,
    )
    metadata["gpu_codegen"] = json.loads(
        gpu_codegen_identity(program, gpu_target, schedule, launch_plan)
    )
    return metadata


def render_jit_r1_hip_plugin(
    contract: dict,
    target: str | dict,
    artifact_id: str | None = None,
    *,
    precision: str = "float32",
    edge_strategy: str = "wave",
    edge_threads_per_block: int = 64,
    persistent_blocks_per_compute_unit: int = 8,
) -> str:
    """Render an ordinary-R1 hipcc shared plugin for one AMD target."""

    metadata = factorized_hip_plugin_metadata(
        contract,
        target,
        artifact_id,
        precision=precision,
        edge_strategy=edge_strategy,
        edge_threads_per_block=edge_threads_per_block,
        persistent_blocks_per_compute_unit=persistent_blocks_per_compute_unit,
    )
    program, _, schedule, _ = _hip_codegen_inputs(
        contract,
        metadata,
        precision,
        "plugin",
        edge_strategy,
        edge_threads_per_block,
        persistent_blocks_per_compute_unit,
    )
    return _render_jit_r1_gpu_artifact(program, metadata, schedule, HIP_DIALECT)


def render_jit_r1_hip_module(
    contract: dict,
    target: str | dict,
    artifact_id: str | None = None,
    *,
    precision: str = "float32",
    edge_strategy: str = "wave",
    edge_threads_per_block: int = 64,
    persistent_blocks_per_compute_unit: int = 8,
) -> str:
    """Render device-only R1 source for hipRTC code-object compilation."""

    metadata = factorized_hip_plugin_metadata(
        contract,
        target,
        artifact_id,
        precision=precision,
        edge_strategy=edge_strategy,
        edge_threads_per_block=edge_threads_per_block,
        persistent_blocks_per_compute_unit=persistent_blocks_per_compute_unit,
    )
    program, _, schedule, launch_plan = _hip_codegen_inputs(
        contract,
        metadata,
        precision,
        "module",
        edge_strategy,
        edge_threads_per_block,
        persistent_blocks_per_compute_unit,
    )
    return _render_jit_r1_gpu_artifact(
        program,
        metadata,
        schedule,
        HIP_DIALECT,
        artifact_kind="module",
        launch_plan=launch_plan,
    )


__all__ = [
    "CUDA_DIALECT",
    "CUDA_PLUGIN_ABI",
    "CUDA_PLUGIN_ABI_VERSION",
    "HIP_DIALECT",
    "HIP_PLUGIN_ABI",
    "HIP_PLUGIN_ABI_VERSION",
    "HOST_PLUGIN_ABI",
    "HOST_PLUGIN_ABI_VERSION",
    "GpuDialect",
    "GpuTarget",
    "KernelSchedule",
    "LaunchPlan",
    "R1Program",
    "factorized_gpu_codegen_identity",
    "factorized_hip_plugin_metadata",
    "gpu_codegen_identity",
    "jit_r1_cuda_plugin_metadata",
    "jit_r1_host_plugin_metadata",
    "render_jit_r1_cuda_module",
    "render_jit_r1_cuda_plugin",
    "render_jit_r1_hip_module",
    "render_jit_r1_hip_plugin",
    "render_jit_r1_host_plugin",
]
