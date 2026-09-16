import ctypes
import importlib.util
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
PACKAGE_SOURCE = REPOSITORY / "symmetrix" / "source" / "symmetrix"
FIXTURE_CONTRACT = (
    Path(__file__).resolve().parent
    / "data"
    / "jit_r1_fixture_c4_e3_l0_contract.json.in"
)
MH0_CONTRACT = (
    Path(__file__).resolve().parent
    / "data"
    / "execution_contracts"
    / "jit_r1_contract.json"
)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def codegen():
    package_name = "_symmetrix_jit_codegen_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(PACKAGE_SOURCE)]
    sys.modules[package_name] = package
    return _load_module(
        package_name + ".jit_codegen",
        PACKAGE_SOURCE / "jit_codegen.py",
    )


@pytest.fixture(scope="module")
def jit():
    return _load_module(
        "_symmetrix_jit_cache_codegen_test",
        PACKAGE_SOURCE / "jit.py",
    )


@pytest.fixture(scope="module")
def cxx():
    compiler = shutil.which(os.environ.get("CXX", "c++").split()[0])
    if compiler is None:
        pytest.skip("a C++ compiler is required for Execution JIT codegen tests")
    return compiler


class RadialSpline(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("edge_types", ctypes.c_uint32),
        ("intervals", ctypes.c_uint32),
        ("functions", ctypes.c_uint32),
        ("h", ctypes.c_double),
        ("x0", ctypes.c_double),
        ("coefficients", ctypes.POINTER(ctypes.c_float)),
    ]


class ForwardArgs(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("active_type_count", ctypes.c_uint32),
        ("num_nodes", ctypes.c_int64),
        ("num_edges", ctypes.c_int64),
        ("node_types", ctypes.POINTER(ctypes.c_int32)),
        ("num_neigh", ctypes.POINTER(ctypes.c_int32)),
        ("first_neigh", ctypes.POINTER(ctypes.c_int32)),
        ("neigh_indices", ctypes.POINTER(ctypes.c_int32)),
        ("neigh_types", ctypes.POINTER(ctypes.c_int32)),
        ("type_to_active", ctypes.POINTER(ctypes.c_int32)),
        ("radius", ctypes.POINTER(ctypes.c_double)),
        ("radial", RadialSpline),
        ("harmonics_values", ctypes.POINTER(ctypes.c_float)),
        ("neighbor_features", ctypes.POINTER(ctypes.c_float)),
        ("output", ctypes.POINTER(ctypes.c_float)),
        ("cutoff", ctypes.c_double),
    ]


class SourceArgs(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("active_type_count", ctypes.c_uint32),
        ("num_nodes", ctypes.c_int64),
        ("num_edges", ctypes.c_int64),
        ("node_types", ctypes.POINTER(ctypes.c_int32)),
        ("neigh_types", ctypes.POINTER(ctypes.c_int32)),
        ("source_offsets", ctypes.POINTER(ctypes.c_int32)),
        ("source_edges", ctypes.POINTER(ctypes.c_int32)),
        ("edge_receivers", ctypes.POINTER(ctypes.c_int32)),
        ("type_to_active", ctypes.POINTER(ctypes.c_int32)),
        ("radius", ctypes.POINTER(ctypes.c_double)),
        ("radial", RadialSpline),
        ("harmonics_values", ctypes.POINTER(ctypes.c_float)),
        ("output_adjoint", ctypes.POINTER(ctypes.c_float)),
        ("source_adjoint", ctypes.POINTER(ctypes.c_float)),
        ("cutoff", ctypes.c_double),
    ]


class EdgeArgs(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("active_type_count", ctypes.c_uint32),
        ("coordinate_scalar_size", ctypes.c_uint32),
        ("coordinates_are_unit", ctypes.c_uint32),
        ("num_nodes", ctypes.c_int64),
        ("num_edges", ctypes.c_int64),
        ("node_types", ctypes.POINTER(ctypes.c_int32)),
        ("neigh_indices", ctypes.POINTER(ctypes.c_int32)),
        ("neigh_types", ctypes.POINTER(ctypes.c_int32)),
        ("edge_receivers", ctypes.POINTER(ctypes.c_int32)),
        ("type_to_active", ctypes.POINTER(ctypes.c_int32)),
        ("xyz", ctypes.c_void_p),
        ("radius", ctypes.POINTER(ctypes.c_double)),
        ("radial", RadialSpline),
        ("harmonics_values", ctypes.POINTER(ctypes.c_float)),
        ("harmonics_gradients", ctypes.POINTER(ctypes.c_float)),
        ("neighbor_features", ctypes.POINTER(ctypes.c_float)),
        ("output_adjoint", ctypes.POINTER(ctypes.c_float)),
        ("directed_forces", ctypes.POINTER(ctypes.c_double)),
        ("cutoff", ctypes.c_double),
    ]


class Plugin(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("pointer_size", ctypes.c_uint32),
        ("byte_order", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("scalar_kind", ctypes.c_uint32),
        ("scalar_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("abi_tag", ctypes.c_char_p),
        ("artifact_id", ctypes.c_char_p),
        ("contract_fingerprint", ctypes.c_char_p),
        ("semantic_fingerprint", ctypes.c_char_p),
        ("structure_fingerprint", ctypes.c_char_p),
        ("channels", ctypes.c_int32),
        ("embedding", ctypes.c_int32),
        ("edge_l_max", ctypes.c_int32),
        ("source_l_max", ctypes.c_int32),
        ("r1_forward_owner", ctypes.c_void_p),
        ("r1_source_owner", ctypes.c_void_p),
        ("r1_compensated_source_owner", ctypes.c_void_p),
        ("r1_edge_owner", ctypes.c_void_p),
    ]


class CudaPlugin(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("pointer_size", ctypes.c_uint32),
        ("byte_order", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("scalar_kind", ctypes.c_uint32),
        ("scalar_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("abi_tag", ctypes.c_char_p),
        ("artifact_id", ctypes.c_char_p),
        ("contract_fingerprint", ctypes.c_char_p),
        ("semantic_fingerprint", ctypes.c_char_p),
        ("structure_fingerprint", ctypes.c_char_p),
        ("channels", ctypes.c_int32),
        ("embedding", ctypes.c_int32),
        ("edge_l_max", ctypes.c_int32),
        ("source_l_max", ctypes.c_int32),
        ("target_compute_capability", ctypes.c_int32),
        ("forward_threads_per_block", ctypes.c_int32),
        ("source_threads_per_block", ctypes.c_int32),
        ("edge_threads_per_block", ctypes.c_int32),
        ("r1_forward_launch", ctypes.c_void_p),
        ("r1_coordinate_reverse_launch", ctypes.c_void_p),
    ]


class HipPlugin(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("pointer_size", ctypes.c_uint32),
        ("byte_order", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("scalar_kind", ctypes.c_uint32),
        ("scalar_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("abi_tag", ctypes.c_char_p),
        ("artifact_id", ctypes.c_char_p),
        ("contract_fingerprint", ctypes.c_char_p),
        ("semantic_fingerprint", ctypes.c_char_p),
        ("structure_fingerprint", ctypes.c_char_p),
        ("channels", ctypes.c_int32),
        ("embedding", ctypes.c_int32),
        ("edge_l_max", ctypes.c_int32),
        ("source_l_max", ctypes.c_int32),
        ("target_architecture", ctypes.c_char_p),
        ("target_features", ctypes.c_char_p),
        ("native_subgroup_width", ctypes.c_int32),
        ("forward_threads_per_block", ctypes.c_int32),
        ("source_threads_per_block", ctypes.c_int32),
        ("edge_threads_per_block", ctypes.c_int32),
        ("persistent_blocks_per_compute_unit", ctypes.c_int32),
        ("r1_forward_launch", ctypes.c_void_p),
        ("r1_coordinate_reverse_launch", ctypes.c_void_p),
    ]


def _nvcc():
    candidates = [
        os.environ.get("NVCC"),
        os.environ.get("CUDACXX"),
        shutil.which("nvcc"),
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-13.3/bin/nvcc",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    pytest.skip("nvcc is required for CUDA Execution JIT codegen tests")


def _hipcc():
    candidate = os.environ.get("HIPCXX") or shutil.which("hipcc")
    if candidate and Path(candidate).is_file():
        resolved = Path(candidate).resolve()
        roots = [resolved.parent.parent, Path("/opt/rocm")]
        rocm_path = os.environ.get("ROCM_PATH")
        if rocm_path:
            roots.insert(0, Path(rocm_path))
        if any((root / "include/hip/hip_runtime.h").is_file() for root in roots):
            return str(resolved)
        pytest.skip("hipcc installation lacks hip/hip_runtime.h")
    pytest.skip("hipcc is required for HIP Execution JIT codegen tests")


def test_generated_plugin_compiles_and_all_r1_owners_match_reference(
    codegen, jit, cxx, tmp_path
):
    contract = json.loads(FIXTURE_CONTRACT.read_text())
    metadata = codegen.jit_r1_host_plugin_metadata(contract)
    source = codegen.render_jit_r1_host_plugin(contract)
    assert source.count("constexpr std::int32_t channel_tile = 16;") == 1
    assert source.count("constexpr std::int32_t channel_tile = 32;") == 1
    assert "if (channel % channel_tile != 0)" in source
    assert source.count("prefetch_read(") == 3
    assert "__builtin_prefetch(address, 0, 1);" in source
    result = jit.prepare_jit_artifact(
        source,
        abi={"host_plugin": metadata["abi"], "version": metadata["abi_version"]},
        build={"backend": "host", "precision": "float32", "contract": metadata},
        cpu={"machine": "test"},
        cache_root=tmp_path / "cache",
        cxx=cxx,
        cxx_flags=("-ffast-math",),
        artifact_name="fixture_r1",
    )
    assert result.available, (result.reason, result.diagnostics)

    library = ctypes.CDLL(str(result.artifact_path))
    query = library.symmetrix_jit_host_plugin_query_v2
    query.restype = ctypes.POINTER(Plugin)
    plugin = query().contents
    assert plugin.abi_version == 2
    assert plugin.struct_size == ctypes.sizeof(Plugin)
    assert plugin.pointer_size == ctypes.sizeof(ctypes.c_void_p)
    assert plugin.byte_order == 0x01020304
    assert plugin.capabilities == 0x6F
    assert (plugin.scalar_kind, plugin.scalar_size) == (1, 4)
    assert plugin.abi_tag.decode() == metadata["abi"]
    assert plugin.artifact_id.decode() == metadata["artifact_id"]
    assert plugin.contract_fingerprint.decode() == metadata["contract_fingerprint"]
    assert plugin.semantic_fingerprint.decode() == metadata["semantic_fingerprint"]
    assert plugin.structure_fingerprint.decode() == metadata["structure_fingerprint"]
    assert (plugin.channels, plugin.embedding) == (4, 3)
    assert (plugin.edge_l_max, plugin.source_l_max) == (0, 0)

    int1 = ctypes.c_int32 * 1
    int2 = ctypes.c_int32 * 2
    int3 = ctypes.c_int32 * 3
    float1 = ctypes.c_float * 1
    float3 = ctypes.c_float * 3
    float8 = ctypes.c_float * 8
    double1 = ctypes.c_double * 1
    double3 = ctypes.c_double * 3

    coefficients = (ctypes.c_float * 16)(
        2.0,
        3.0,
        4.0,
        5.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    radial = RadialSpline(ctypes.sizeof(RadialSpline), 1, 1, 4, 1.0, 0.0, coefficients)
    node_types = int2(0, 0)
    num_neigh = int2(1, 0)
    first_neigh = int2(0, 1)
    neigh_indices = int1(1)
    neigh_types = int1(0)
    edge_receivers = int1(0)
    type_to_active = int1(0)
    radius = double1(0.5)
    harmonics = float1(0.5)
    neighbor_features = float8(0.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0, 40.0)
    output = float8()

    forward_args = ForwardArgs(
        ctypes.sizeof(ForwardArgs),
        1,
        2,
        1,
        node_types,
        num_neigh,
        first_neigh,
        neigh_indices,
        neigh_types,
        type_to_active,
        radius,
        radial,
        harmonics,
        neighbor_features,
        output,
        0.75,
    )
    forward = ctypes.CFUNCTYPE(
        None, ctypes.POINTER(ForwardArgs), ctypes.c_int32, ctypes.c_int32
    )(plugin.r1_forward_owner)
    for channel in range(4):
        forward(ctypes.byref(forward_args), 0, channel)
    assert list(output) == pytest.approx([12.5, 35.0, 67.5, 110.0, 0.0, 0.0, 0.0, 0.0])

    output_adjoint = float8(1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0)
    source_offsets = int3(0, 0, 1)
    source_edges = int1(0)

    source_callback = ctypes.CFUNCTYPE(
        None, ctypes.POINTER(SourceArgs), ctypes.c_int32, ctypes.c_int32
    )
    for callback_pointer in (
        plugin.r1_source_owner,
        plugin.r1_compensated_source_owner,
    ):
        source_adjoint = float8()
        source_args = SourceArgs(
            ctypes.sizeof(SourceArgs),
            1,
            2,
            1,
            node_types,
            neigh_types,
            source_offsets,
            source_edges,
            edge_receivers,
            type_to_active,
            radius,
            radial,
            harmonics,
            output_adjoint,
            source_adjoint,
            0.75,
        )
        callback = source_callback(callback_pointer)
        for channel in range(4):
            callback(ctypes.byref(source_args), 1, channel)
        assert list(source_adjoint) == pytest.approx(
            [0.0, 0.0, 0.0, 0.0, 1.25, 3.5, 6.75, 11.0]
        )

    xyz = double3(0.5, 0.0, 0.0)
    harmonic_gradients = float3(0.1, 0.2, 0.3)
    directed_forces = double3()
    edge_args = EdgeArgs(
        ctypes.sizeof(EdgeArgs),
        1,
        ctypes.sizeof(ctypes.c_double),
        0,
        2,
        1,
        node_types,
        neigh_indices,
        neigh_types,
        edge_receivers,
        type_to_active,
        ctypes.cast(xyz, ctypes.c_void_p),
        radius,
        radial,
        harmonics,
        harmonic_gradients,
        neighbor_features,
        output_adjoint,
        directed_forces,
        0.75,
    )
    edge = ctypes.CFUNCTYPE(None, ctypes.POINTER(EdgeArgs), ctypes.c_int32)(
        plugin.r1_edge_owner
    )
    edge(ctypes.byref(edge_args), 0)
    assert list(directed_forces) == pytest.approx([-295.0, -290.0, -435.0])

    unit_direction = float3(1.0, 0.0, 0.0)
    compact_forces = double3()
    edge_args.coordinate_scalar_size = ctypes.sizeof(ctypes.c_float)
    edge_args.coordinates_are_unit = 1
    edge_args.xyz = ctypes.cast(unit_direction, ctypes.c_void_p)
    edge_args.directed_forces = compact_forces
    edge(ctypes.byref(edge_args), 0)
    assert list(compact_forces) == pytest.approx(list(directed_forces))

    y_only_forces = double3()
    edge_args.harmonics_gradients = None
    edge_args.directed_forces = y_only_forces
    edge(ctypes.byref(edge_args), 0)
    assert list(y_only_forces) == pytest.approx([-150.0, 0.0, 0.0])


def test_generated_float64_host_plugin_compiles_and_runs_forward(
    codegen, jit, cxx, tmp_path
):
    contract = json.loads(FIXTURE_CONTRACT.read_text())
    metadata = codegen.jit_r1_host_plugin_metadata(contract, precision="float64")
    source = codegen.render_jit_r1_host_plugin(contract, precision="float64")
    assert "if (coordinates_are_unit != 0)" in source
    assert "coordinate_scalar_size == sizeof(Scalar)" not in source
    result = jit.prepare_jit_artifact(
        source,
        abi={"host_plugin": metadata["abi"], "version": metadata["abi_version"]},
        build={"backend": "host", "precision": "float64", "contract": metadata},
        cpu={"machine": "test"},
        cache_root=tmp_path / "cache",
        cxx=cxx,
        cxx_flags=("-ffast-math",),
        artifact_name="fixture_r1_f64",
    )
    assert result.available, (result.reason, result.diagnostics)

    library = ctypes.CDLL(str(result.artifact_path))
    query = library.symmetrix_jit_host_plugin_query_v2
    query.restype = ctypes.POINTER(Plugin)
    plugin = query().contents
    assert (plugin.abi_version, plugin.scalar_kind, plugin.scalar_size) == (2, 2, 8)
    assert plugin.artifact_id.decode().startswith("jit-r1-gen11-f64-")

    int1 = ctypes.c_int32 * 1
    int2 = ctypes.c_int32 * 2
    double1 = ctypes.c_double * 1
    double8 = ctypes.c_double * 8
    coefficients = (ctypes.c_double * 16)(
        2.0,
        3.0,
        4.0,
        5.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )

    def as_float_pointer(value):
        return ctypes.cast(value, ctypes.POINTER(ctypes.c_float))

    radial = RadialSpline(
        ctypes.sizeof(RadialSpline),
        1,
        1,
        4,
        1.0,
        0.0,
        as_float_pointer(coefficients),
    )
    node_types = int2(0, 0)
    num_neigh = int2(1, 0)
    first_neigh = int2(0, 1)
    neigh_indices = int1(1)
    neigh_types = int1(0)
    type_to_active = int1(0)
    radius = double1(0.5)
    harmonics = double1(0.5)
    neighbor_features = double8(0.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0, 40.0)
    output = double8()
    args = ForwardArgs(
        ctypes.sizeof(ForwardArgs),
        1,
        2,
        1,
        node_types,
        num_neigh,
        first_neigh,
        neigh_indices,
        neigh_types,
        type_to_active,
        radius,
        radial,
        as_float_pointer(harmonics),
        as_float_pointer(neighbor_features),
        as_float_pointer(output),
        0.75,
    )
    forward = ctypes.CFUNCTYPE(
        None, ctypes.POINTER(ForwardArgs), ctypes.c_int32, ctypes.c_int32
    )(plugin.r1_forward_owner)
    for channel in range(4):
        forward(ctypes.byref(args), 0, channel)
    assert list(output) == pytest.approx([12.5, 35.0, 67.5, 110.0, 0.0, 0.0, 0.0, 0.0])

    radius[0] = 0.875
    for index in range(8):
        output[index] = 123.0
    for channel in range(4):
        forward(ctypes.byref(args), 0, channel)
    assert list(output[:4]) == [0.0] * 4
    assert list(output[4:]) == [123.0] * 4


@pytest.mark.parametrize("precision", (None, "float16", "double"))
def test_plugin_codegen_rejects_unsupported_precision(codegen, precision):
    contract = json.loads(FIXTURE_CONTRACT.read_text())
    with pytest.raises(ValueError, match="precision"):
        codegen.render_jit_r1_host_plugin(contract, precision=precision)


def test_host_plugin_codegen_rejects_non_uvu_contract(codegen):
    contract = json.loads(FIXTURE_CONTRACT.read_text())
    contract["paths"][0]["connection_mode"] = "uvw"
    with pytest.raises(ValueError, match="uvu paths"):
        codegen.render_jit_r1_host_plugin(contract)


def test_gpu_codegen_identity_is_deterministic_and_backend_specific(codegen):
    contract = json.loads(FIXTURE_CONTRACT.read_text())
    cuda_program = codegen.R1Program.from_contract(
        contract, "float32", packet_layout_version=2
    )
    reordered = {key: contract[key] for key in reversed(contract)}
    assert (
        cuda_program.canonical_json()
        == codegen.R1Program.from_contract(
            reordered, "float32", packet_layout_version=2
        ).canonical_json()
    )

    cuda_target = codegen.GpuTarget(1, "cuda", "sm_80", "", 32, "compute_80")
    cuda_schedule = codegen.KernelSchedule(1, "serial", 1, 256, 256, 256, 1)
    cuda_plan = codegen.LaunchPlan(
        2,
        (
            "symmetrix_factorized_forward_v2",
            "symmetrix_factorized_source_v2",
            "symmetrix_factorized_edge_v2",
        ),
        "symmetrix_factorized_reverse_fused_v2",
    )
    cuda_identity = codegen.gpu_codegen_identity(
        cuda_program, cuda_target, cuda_schedule, cuda_plan
    )
    assert cuda_identity == codegen.gpu_codegen_identity(
        cuda_program, cuda_target, cuda_schedule, cuda_plan
    )

    hip_program = codegen.R1Program.from_contract(
        contract, "float32", packet_layout_version=1
    )
    hip_target = codegen.GpuTarget(1, "hip", "gfx942", "xnack-", 64, "gfx942:xnack-")
    hip_schedule = codegen.KernelSchedule(1, "wave", 64, 256, 256, 256, 8)
    hip_plan = codegen.LaunchPlan(
        2,
        (
            "symmetrix_factorized_forward_v1",
            "symmetrix_factorized_source_v1",
            "symmetrix_factorized_edge_v1",
        ),
        "symmetrix_factorized_reverse_fused_v1",
    )
    hip_identity = codegen.gpu_codegen_identity(
        hip_program, hip_target, hip_schedule, hip_plan
    )
    assert cuda_identity != hip_identity
    assert '"backend":"cuda"' in cuda_identity
    assert '"backend":"hip"' in hip_identity


def test_gpu_codegen_identity_separates_schema_schedule_and_artifact_kind(
    codegen, monkeypatch
):
    contract = json.loads(FIXTURE_CONTRACT.read_text())
    target = {
        "raw_agent_target": "gfx942",
        "native_subgroup_width": 64,
    }
    wave = codegen.factorized_gpu_codegen_identity(
        contract, "hip", target, edge_strategy="wave", artifact_kind="plugin"
    )
    serial = codegen.factorized_gpu_codegen_identity(
        contract, "hip", target, edge_strategy="serial", artifact_kind="plugin"
    )
    module = codegen.factorized_gpu_codegen_identity(
        contract, "hip", target, edge_strategy="wave", artifact_kind="module"
    )
    assert wave != serial
    assert wave != module
    assert wave["schedule"]["edge_strategy"] == "wave"
    assert serial["schedule"]["edge_strategy"] == "serial"

    monkeypatch.setattr(codegen, "GPU_CODEGEN_IDENTITY_SCHEMA_VERSION", 2)
    schema_v2 = codegen.factorized_gpu_codegen_identity(
        contract, "hip", target, edge_strategy="wave", artifact_kind="plugin"
    )
    assert schema_v2 != wave
    assert schema_v2["schema_version"] == 2


def test_wave_owner_is_dialect_driven_without_vendor_tokens_in_common_body(codegen):
    common_source = inspect.getsource(codegen._render_wave_edge_owner)
    assert "SymmetrixJitHip" not in common_source
    assert "SymmetrixJitCuda" not in common_source
    assert "__shfl_down" not in common_source

    contract = json.loads(FIXTURE_CONTRACT.read_text())
    rows = codegen._fixed_weight_path_rows(codegen.normalize_jit_r1_contract(contract))
    arguments = {
        "channels": 4,
        "edge_harmonics": 1,
        "source_harmonics": 1,
        "subgroup_width": 32,
    }
    cuda = codegen._render_wave_edge_owner(
        rows,
        args_type="NeutralCudaPacket",
        dialect=codegen.CUDA_DIALECT,
        **arguments,
    )
    hip = codegen._render_wave_edge_owner(
        rows,
        args_type="NeutralHipPacket",
        dialect=codegen.HIP_DIALECT,
        **arguments,
    )
    assert "const NeutralCudaPacket* args" in cuda
    assert "__shfl_down_sync(0xffffffffu" in cuda
    assert "SymmetrixJitHip" not in cuda
    assert "const NeutralHipPacket* args" in hip
    assert "__shfl_down(local_force_x" in hip
    assert "SymmetrixJitCuda" not in hip


@pytest.mark.parametrize(
    "factory,error",
    (
        (
            lambda c: c.GpuTarget(1, "cuda", "gfx942", "", 32, "compute_80"),
            "CUDA architecture",
        ),
        (
            lambda c: c.GpuTarget(1, "hip", "gfx-942", "", 64, "gfx-942"),
            "HIP architecture",
        ),
        (
            lambda c: c.KernelSchedule(1, "invalid", 1, 256, 256, 256, 1),
            "edge strategy",
        ),
        (lambda c: c.KernelSchedule(1, "serial", 32, 256, 256, 256, 1), "serial edge"),
        (lambda c: c.KernelSchedule(1, "wave", 64, 256, 256, 96, 8), "block sizes"),
        (
            lambda c: c.LaunchPlan(2, ("same", "same", "edge"), "fused"),
            "distinct C symbols",
        ),
    ),
)
def test_typed_gpu_codegen_inputs_fail_closed(codegen, factory, error):
    with pytest.raises(ValueError, match=error):
        factory(codegen)


def test_cuda_plugin_codegen_metadata_and_source_contract(codegen):
    contract = json.loads(FIXTURE_CONTRACT.read_text())
    metadata = codegen.jit_r1_cuda_plugin_metadata(contract, 80)
    source = codegen.render_jit_r1_cuda_plugin(contract, 80)

    assert metadata["abi"] == "symmetrix.jit.cuda-plugin/2"
    assert metadata["abi_version"] == 2
    assert (metadata["scalar_kind"], metadata["scalar_size"]) == (1, 4)
    assert metadata["target_compute_capability"] == 80
    assert metadata["forward_threads_per_block"] == 256
    assert metadata["source_threads_per_block"] == 256
    assert metadata["edge_threads_per_block"] == 32
    assert metadata["reverse_threads_per_block"] == 64
    assert metadata["reverse_shared_memory_bytes"] == 16
    assert metadata["reverse_strategy"] == "fused_source_edge"
    assert metadata["persistent_blocks_per_compute_unit"] == 8
    assert metadata["edge_strategy"] == "wave"
    assert metadata["edge_logical_subgroup_width"] == 32
    assert (metadata["channels"], metadata["embedding"]) == (4, 3)
    assert (metadata["edge_l_max"], metadata["source_l_max"]) == (0, 0)

    assert "#pragma once" not in source
    assert "Kokkos" not in source
    assert "torch" not in source.lower()
    assert source.count("__global__ void r1_") == 2
    assert (
        "__global__ __launch_bounds__(edge_threads_per_block)\nvoid r1_edge_kernel"
        in source
    )
    assert "symmetrix_jit_cuda_plugin_query_v2" in source
    assert "r1_forward_launch" in source
    assert "r1_coordinate_reverse_launch" in source
    assert "void r1_reverse_fused_kernel" in source
    assert "__shared__ Scalar source_values[4]" in source
    assert "source_args.source_offsets[source]" in source
    assert "source_values[0 * 4 + channel] +=" in source
    assert "double x;" in source
    assert "floor((radius - radial.x0) / radial.h)" in source
    assert "local_radius" not in source
    assert "#pragma unroll 1" in source
    assert "reverse_path < 1" in source
    assert "switch (reverse_path)" in source
    assert "source_args->num_nodes < persistent_blocks" in source
    assert "source_args->cutoff != edge_args->cutoff" in source
    assert source.count("cudaPeekAtLastError()") == 2
    assert "cudaDeviceSynchronize" not in source
    assert "cudaStreamSynchronize" not in source
    assert "owner += stride" in source
    assert "edge += wave_stride" in source
    assert "ForwardArgsV2 launch_args = *args" in source
    assert "SourceArgsV2 source_launch_args = *source_args" in source
    assert "EdgeArgsV2 edge_launch_args = *edge_args" in source


@pytest.mark.parametrize(
    "backend,artifact_kind",
    (
        ("host", "plugin"),
        ("cuda", "plugin"),
        ("cuda", "module"),
        ("hip", "plugin"),
        ("hip", "module"),
    ),
)
def test_generated_global_offsets_promote_before_multiplication(
    codegen, backend, artifact_kind
):
    contract = json.loads(FIXTURE_CONTRACT.read_text())
    if backend == "host":
        source = codegen.render_jit_r1_host_plugin(contract)
    elif backend == "cuda":
        renderer = (
            codegen.render_jit_r1_cuda_plugin
            if artifact_kind == "plugin"
            else codegen.render_jit_r1_cuda_module
        )
        source = renderer(contract, 80)
    else:
        renderer = (
            codegen.render_jit_r1_hip_plugin
            if artifact_kind == "plugin"
            else codegen.render_jit_r1_hip_module
        )
        source = renderer(
            contract,
            {
                "raw_agent_target": "gfx942",
                "target_features": "",
                "native_subgroup_width": 64,
            },
        )

    unsafe_global_offsets = re.findall(
        r"\[[^\]\n]*(?:(?:receiver|source|edge)\s*\*|3\s*\*\s*edge)[^\]\n]*\]",
        source,
    )
    assert unsafe_global_offsets == []
    assert "static_cast<std::size_t>(receiver)" in source
    assert "static_cast<std::size_t>(source)" in source
    assert "static_cast<std::size_t>(edge)" in source
    if backend != "host" and artifact_kind == "plugin":
        assert "work_items / threads_per_block" in source
        assert "work_items + threads_per_block - 1" not in source


def test_mh0_cuda_reverse_uses_final_harmonic_band_shape(codegen):
    contract = json.loads(MH0_CONTRACT.read_text())
    normalized = codegen.normalize_jit_r1_contract(contract)
    metadata = codegen.jit_r1_cuda_plugin_metadata(contract, 120)
    source = codegen._render_r1_fused_reverse_kernel(
        codegen._fixed_weight_path_rows(normalized),
        source_type="SourceArgs",
        edge_type="EdgeArgs",
        kernel_name="r1_reverse_fused_kernel",
        channels=metadata["channels"],
        edge_harmonics=(metadata["edge_l_max"] + 1) ** 2,
        source_harmonics=(metadata["source_l_max"] + 1) ** 2,
        output_components=metadata["output_components"],
        subgroup_width=32,
        dialect=codegen.CUDA_DIALECT,
        exported=False,
    )

    assert source.count("evaluate_radial(") == 10
    assert source.count("Scalar radial_force_") == 4
    assert source.count("Scalar angular_force_") == 16
    assert source.count("load_row_") == 0
    assert source.count("const Scalar output_adjoint_") == 40
    assert source.count("source_values[") == 79


def test_cuda_schedule_variants_change_identity_metadata_and_source(codegen):
    contract = json.loads(FIXTURE_CONTRACT.read_text())
    default_schedule = codegen.factorized_gpu_codegen_identity(contract, "cuda", 120)
    serial_128 = codegen.factorized_gpu_codegen_identity(
        contract,
        "cuda",
        120,
        edge_strategy="serial",
        edge_threads_per_block=128,
    )
    wave = codegen.factorized_gpu_codegen_identity(
        contract,
        "cuda",
        120,
        edge_strategy="wave",
        edge_logical_subgroup_width=4,
        edge_threads_per_block=128,
        persistent_blocks_per_compute_unit=4,
    )
    wave_module = codegen.factorized_gpu_codegen_identity(
        contract,
        "cuda",
        120,
        artifact_kind="module",
        edge_strategy="wave",
        edge_logical_subgroup_width=4,
        edge_threads_per_block=128,
        persistent_blocks_per_compute_unit=4,
    )

    assert default_schedule["schedule"] == {
        "schema_version": 1,
        "edge_strategy": "wave",
        "edge_logical_subgroup_width": 32,
        "forward_threads_per_block": 256,
        "source_threads_per_block": 256,
        "edge_threads_per_block": 32,
        "persistent_blocks_per_compute_unit": 8,
        "dynamic_shared_memory": 0,
    }
    assert (
        len(
            {
                json.dumps(value, sort_keys=True)
                for value in (default_schedule, serial_128, wave, wave_module)
            }
        )
        == 4
    )

    metadata = codegen.jit_r1_cuda_plugin_metadata(
        contract,
        120,
        edge_strategy="wave",
        edge_logical_subgroup_width=4,
        edge_threads_per_block=128,
        persistent_blocks_per_compute_unit=4,
    )
    source = codegen.render_jit_r1_cuda_plugin(
        contract,
        120,
        edge_strategy="wave",
        edge_logical_subgroup_width=4,
        edge_threads_per_block=128,
        persistent_blocks_per_compute_unit=4,
    )
    assert metadata["edge_strategy"] == "wave"
    assert metadata["edge_logical_subgroup_width"] == 4
    assert metadata["edge_threads_per_block"] == 128
    assert metadata["reverse_threads_per_block"] == 64
    assert metadata["persistent_blocks_per_compute_unit"] == 4
    assert "__global__ __launch_bounds__(edge_threads_per_block)" in source
    assert "channel = lane" in source
    assert "channel += 4" in source
    assert "        {\n        Scalar radial_value_0" in source
    assert "        {\n        const Scalar weighted_0" in source
    assert "constexpr std::int32_t reverse_threads_per_block =\n    64;" in source
    assert "constexpr std::int32_t edge_threads_per_block =\n    128;" in source
    assert metadata["gpu_codegen"]["schedule"] == wave["schedule"]


@pytest.mark.parametrize(
    "kwargs,error",
    (
        ({"edge_strategy": "invalid"}, "edge strategy"),
        (
            {
                "edge_strategy": "serial",
                "edge_logical_subgroup_width": 2,
            },
            "serial edge",
        ),
        (
            {
                "edge_strategy": "wave",
                "edge_logical_subgroup_width": 8,
                "edge_threads_per_block": 4,
            },
            "logical subgroup multiple",
        ),
        (
            {
                "edge_strategy": "wave",
                "edge_logical_subgroup_width": 64,
            },
            "logical subgroup",
        ),
    ),
)
def test_cuda_schedule_variants_fail_closed(codegen, kwargs, error):
    contract = json.loads(FIXTURE_CONTRACT.read_text())
    with pytest.raises(ValueError, match=error):
        codegen.factorized_gpu_codegen_identity(contract, "cuda", 120, **kwargs)


@pytest.mark.parametrize(
    "precision,scalar_kind,scalar_size",
    (("float32", 1, 4), ("float64", 2, 8)),
)
def test_hip_plugin_codegen_metadata_and_source_contract(
    codegen, precision, scalar_kind, scalar_size
):
    contract = json.loads(FIXTURE_CONTRACT.read_text())
    target = {
        "raw_agent_target": "gfx942:xnack-:sramecc+",
        "target_features": "sramecc+:xnack-",
        "native_subgroup_width": 64,
    }
    metadata = codegen.factorized_hip_plugin_metadata(
        contract, target, precision=precision
    )
    source = codegen.render_jit_r1_hip_plugin(contract, target, precision=precision)

    assert metadata["abi"] == "symmetrix.jit.hip-plugin/1"
    assert metadata["abi_version"] == 1
    assert (metadata["scalar_kind"], metadata["scalar_size"]) == (
        scalar_kind,
        scalar_size,
    )
    assert metadata["target_architecture"] == "gfx942"
    assert metadata["target_features"] == "sramecc+:xnack-"
    assert metadata["native_subgroup_width"] == 64
    assert metadata["edge_strategy"] == "wave"
    assert metadata["edge_logical_subgroup_width"] == 64
    assert metadata["reverse_threads_per_block"] == 64
    assert metadata["reverse_shared_memory_bytes"] == 4 * scalar_size
    assert metadata["reverse_strategy"] == "fused_source_edge"
    assert metadata["persistent_blocks_per_compute_unit"] == 8
    assert "symmetrix_jit_hip_plugin_query_v1" in source
    assert source.count("__global__") == 4
    assert "__global__ __launch_bounds__(edge_threads_per_block)" in source
    assert "channel = lane" in source
    assert "channel += 64" in source
    assert "__shfl_down" in source
    assert "source_args->num_nodes < persistent_blocks" in source
    assert "__shared__ Scalar source_values[4]" in source
    assert "void r1_reverse_fused_kernel" in source
    evaluation_scalar = "Scalar" if precision == "float32" else "double"
    assert f"{evaluation_scalar} x;" in source
    if precision == "float32":
        assert (
            "const Scalar local_radius =\n        static_cast<Scalar>(radius);"
            in source
        )
    else:
        assert "floor((radius - radial.x0) / radial.h)" in source
        assert "local_radius" not in source
    assert "const Scalar harmonic_value_0_0" in source
    assert "const Scalar output_adjoint_0_0" in source
    assert "Scalar(1.0) * output_adjoint_0_0" in source
    assert "Scalar weighted_0_0 = Scalar(0)" in source
    assert "Scalar radial_force_0 = Scalar(0)" in source
    assert "radial_force_0 +=" in source
    assert "local_force_x += radial_force_0 * x_over_r;" in source
    assert "hipPeekAtLastError()" in source
    assert "cuda" not in source.lower()

    tuned_metadata = codegen.factorized_hip_plugin_metadata(
        contract,
        target,
        precision=precision,
        edge_threads_per_block=128,
        persistent_blocks_per_compute_unit=4,
    )
    tuned_source = codegen.render_jit_r1_hip_plugin(
        contract,
        target,
        precision=precision,
        edge_threads_per_block=128,
        persistent_blocks_per_compute_unit=4,
    )
    assert tuned_metadata["edge_threads_per_block"] == 128
    assert tuned_metadata["persistent_blocks_per_compute_unit"] == 4
    assert "edge_threads_per_block =\n    128;" in tuned_source
    assert tuned_source != source

    serial = codegen.render_jit_r1_hip_plugin(
        contract, target, precision=precision, edge_strategy="serial"
    )
    assert "channel = lane" not in serial
    assert "void r1_reverse_fused_kernel" in serial
    assert "__shfl_down" in serial
    assert "static_cast<std::int64_t>(edge_args->num_edges) * 64" not in serial
    assert (
        codegen.factorized_hip_plugin_metadata(
            contract, target, precision=precision, edge_strategy="serial"
        )["edge_logical_subgroup_width"]
        == 1
    )


@pytest.mark.parametrize(
    "precision,scalar_kind,scalar_size",
    (("float32", 1, 4), ("float64", 2, 8)),
)
def test_generated_hip_plugin_compiles_and_exports_descriptor(
    codegen, tmp_path, precision, scalar_kind, scalar_size
):
    contract = json.loads(FIXTURE_CONTRACT.read_text())
    target = {
        "raw_agent_target": "gfx1151",
        "target_features": "",
        "native_subgroup_width": 32,
    }
    metadata = codegen.factorized_hip_plugin_metadata(
        contract, target, precision=precision
    )
    source_path = tmp_path / f"fixture_plugin_{precision}.cpp"
    artifact_path = tmp_path / f"fixture_plugin_{precision}.so"
    source_path.write_text(
        codegen.render_jit_r1_hip_plugin(contract, target, precision=precision),
        encoding="utf-8",
    )
    subprocess.run(
        [
            _hipcc(),
            "-std=c++20",
            "-O2",
            "-shared",
            "-fPIC",
            "--offload-arch=gfx1151",
            str(source_path),
            "-o",
            str(artifact_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    symbols = subprocess.run(
        ["nm", "-D", "--defined-only", str(artifact_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "symmetrix_jit_hip_plugin_query_v1" in symbols
    library = ctypes.CDLL(str(artifact_path))
    query = library.symmetrix_jit_hip_plugin_query_v1
    query.restype = ctypes.POINTER(HipPlugin)
    plugin = query().contents
    assert plugin.abi_version == 1
    assert plugin.struct_size == ctypes.sizeof(HipPlugin)
    assert (plugin.scalar_kind, plugin.scalar_size) == (scalar_kind, scalar_size)
    assert plugin.abi_tag.decode() == metadata["abi"]
    assert plugin.artifact_id.decode() == metadata["artifact_id"]
    assert plugin.target_architecture.decode() == "gfx1151"
    assert plugin.target_features.decode() == ""
    assert plugin.native_subgroup_width == 32
    assert plugin.persistent_blocks_per_compute_unit == 8


def test_cuda_module_codegen_is_device_only_with_stable_symbols(codegen):
    contract = json.loads(FIXTURE_CONTRACT.read_text())
    source = codegen.render_jit_r1_cuda_module(contract, 80)

    assert source.count('extern "C" __global__ void symmetrix_factorized_') == 2
    assert (
        'extern "C" __global__ __launch_bounds__(edge_threads_per_block)\n'
        "void symmetrix_factorized_edge_v2" in source
    )
    assert "#include" not in source
    assert "Self-contained NVRTC packet ABI" in source
    assert "symmetrix_factorized_forward_v2" in source
    assert "symmetrix_factorized_source_v2" in source
    assert "symmetrix_factorized_edge_v2" in source
    assert "symmetrix_factorized_reverse_fused_v2" in source
    assert "symmetrix_factorized_contract_fingerprint" in source
    assert "symmetrix_factorized_forward_threads_per_block" in source
    assert "symmetrix_factorized_source_threads_per_block" in source
    assert "symmetrix_factorized_edge_threads_per_block" in source
    assert "symmetrix_factorized_reverse_threads_per_block" in source
    assert "symmetrix_factorized_reverse_physical_launch_count" in source
    assert "symmetrix_factorized_persistent_blocks_per_compute_unit" in source
    assert "unsigned int coordinate_scalar_size;" in source
    assert "unsigned int coordinates_are_unit;" in source
    assert "unsigned int coordinate_scalar_size," in source
    assert "std::uint32_t coordinate_scalar_size," not in source
    assert "const void* xyz;" in source
    assert (
        codegen.jit_r1_cuda_plugin_metadata(contract, 80)["contract_fingerprint"]
        in source
    )
    assert "<<<" not in source
    assert "cudaPeekAtLastError" not in source
    assert "std::int32_t r1_forward_launch(" not in source
    assert "std::int32_t r1_coordinate_reverse_launch(" not in source
    assert "owner += stride" in source
    assert "edge += wave_stride" in source
    assert "source_args.source_ids[source_index]" in source
    assert "bool edge_is_active(" in source
    assert "return radius < cutoff;" in source
    assert source.count("edge_is_active(") >= 9
    assert "if (threadIdx.x == 0 && edge_active)" in source
    assert "const EvaluationPoint point = edge_active" in source


def test_cuda_mh0_fp32_module_caches_source_features_in_fused_reverse(codegen):
    contract = json.loads(MH0_CONTRACT.read_text())
    source = codegen.render_jit_r1_cuda_module(contract, 120, precision="float32")

    assert "__shared__ Scalar source_features[512];" in source
    assert "source_features[index] =" in source
    fused_reverse = source.split(
        "void symmetrix_factorized_reverse_fused_v2", maxsplit=1
    )[1].split('extern "C" __global__', maxsplit=1)[0]
    assert "source_features[0 * 128 + channel]" in fused_reverse
    assert "neighbor_features[(source_offset + 0) * 128 + channel]" not in fused_reverse

    fp64 = codegen.render_jit_r1_cuda_module(contract, 120, precision="float64")
    assert "__shared__ Scalar source_features[512];" not in fp64

    generic_fp32 = codegen.render_jit_r1_cuda_module(
        json.loads(FIXTURE_CONTRACT.read_text()), 120, precision="float32"
    )
    assert "__shared__ Scalar source_features[512];" not in generic_fp32


def test_hip_module_codegen_is_device_only_with_stable_symbols(codegen):
    contract = json.loads(FIXTURE_CONTRACT.read_text())
    target = {
        "raw_agent_target": "gfx1151",
        "target_features": "",
        "native_subgroup_width": 32,
    }
    source = codegen.render_jit_r1_hip_module(contract, target)

    assert source.count('extern "C" __global__') == 8
    assert "symmetrix_factorized_forward_v1" in source
    assert "symmetrix_factorized_source_v1" in source
    assert "symmetrix_factorized_edge_v1" in source
    assert "symmetrix_factorized_reverse_fused_v1" in source
    assert "symmetrix_factorized_tiled_forward_v1" in source
    assert "symmetrix_factorized_tiled_reverse_v1" in source
    assert "symmetrix_factorized_projected_forward_v1" in source
    assert "symmetrix_factorized_projected_reverse_v1" in source
    assert "symmetrix_factorized_contract_fingerprint" in source
    assert "symmetrix_factorized_forward_threads_per_block" in source
    assert "symmetrix_factorized_source_threads_per_block" in source
    assert "symmetrix_factorized_edge_logical_subgroup_width" in source
    assert "symmetrix_factorized_edge_threads_per_block" in source
    assert "symmetrix_factorized_persistent_blocks_per_compute_unit" in source
    assert "symmetrix_factorized_tiled_forward_threads_per_block" in source
    assert "symmetrix_factorized_tiled_reverse_threads_per_block" in source
    assert "symmetrix_factorized_projected_threads_per_block" in source
    assert "symmetrix_factorized_projected_shared_memory_bytes" in source
    assert "unsigned int coordinate_scalar_size;" in source
    assert "unsigned int coordinates_are_unit;" in source
    assert "unsigned int coordinate_scalar_size," in source
    assert "std::uint32_t coordinate_scalar_size," not in source
    assert "const void* xyz;" in source
    assert "unsigned int channel_begin;" in source
    assert "unsigned int channel_count;" in source
    assert "<<<" not in source
    assert "hipPeekAtLastError" not in source
    assert "symmetrix_jit_hip_plugin_query_v1" not in source
    assert "r1_forward_launch(" not in source
    assert "r1_coordinate_reverse_launch(" not in source
    assert "owner += stride" in source
    assert "edge += wave_stride" in source


def test_hip_projected_module_uses_distinct_a1_and_source_harmonic_strides(codegen):
    contract = json.loads(MH0_CONTRACT.read_text())
    target = {
        "raw_agent_target": "gfx1151",
        "target_features": "",
        "native_subgroup_width": 32,
    }
    source = codegen.render_jit_r1_hip_module(contract, target)

    assert "output[(receiver * 16 + 0 + component)" in source
    assert "(receiver * 16 + 9 + component)" in source
    assert "source_adjoint[(static_cast<std::size_t>(source) * 4 +" in source


def test_hip_tiled_module_uses_compact_phi_and_full_feature_channel_strides(codegen):
    contract = json.loads(MH0_CONTRACT.read_text())
    target = {
        "raw_agent_target": "gfx1151",
        "target_features": "",
        "native_subgroup_width": 32,
    }
    source = codegen.render_jit_r1_hip_module(contract, target)

    assert "output[0 * (64)" in source
    assert "channel - static_cast<std::int32_t>(args->channel_begin)" in source
    assert "* (64) + local_channel" in source
    assert (
        "const std::int32_t channel =\n                    static_cast<std::int32_t>(source_args.channel_begin) + local_channel;"
        in source
    )
    assert "source_adjoint[target] += source_values[harmonic * 128 + channel]" in source


def test_gpu_modules_render_independently_of_plugin_envelopes(codegen, monkeypatch):
    contract = json.loads(FIXTURE_CONTRACT.read_text())

    def reject_plugin_render(*args, **kwargs):
        raise AssertionError("module renderer called a plugin renderer")

    monkeypatch.setattr(codegen, "render_jit_r1_cuda_plugin", reject_plugin_render)
    monkeypatch.setattr(codegen, "render_jit_r1_hip_plugin", reject_plugin_render)
    assert "symmetrix_factorized_forward_v2" in codegen.render_jit_r1_cuda_module(
        contract, 80
    )
    assert "symmetrix_factorized_forward_v1" in codegen.render_jit_r1_hip_module(
        contract,
        {
            "raw_agent_target": "gfx1151",
            "target_features": "",
            "native_subgroup_width": 32,
        },
    )


def test_gpu_artifact_renderers_do_not_translate_or_slice_completed_source(codegen):
    artifact_renderer = inspect.getsource(codegen._render_jit_r1_gpu_artifact)
    hip_renderer = inspect.getsource(codegen.render_jit_r1_hip_plugin)
    cuda_module_renderer = inspect.getsource(codegen.render_jit_r1_cuda_module)
    hip_module_renderer = inspect.getsource(codegen.render_jit_r1_hip_module)

    assert "_render_gpu_precision_source" not in artifact_renderer
    assert ".replace(" not in artifact_renderer
    assert ".split(" not in artifact_renderer
    assert "render_jit_r1_cuda_plugin" not in hip_renderer
    assert "render_jit_r1_cuda_plugin" not in cuda_module_renderer
    assert "render_jit_r1_hip_plugin" not in hip_module_renderer


@pytest.mark.parametrize("compute_capability", (True, 0, -1, 1000, 8.0, "80"))
def test_cuda_plugin_codegen_rejects_invalid_compute_capability(
    codegen, compute_capability
):
    contract = json.loads(FIXTURE_CONTRACT.read_text())
    with pytest.raises(ValueError, match="compute capability"):
        codegen.render_jit_r1_cuda_plugin(contract, compute_capability)


@pytest.mark.parametrize(
    "precision,scalar_kind,scalar_size",
    (("float32", 1, 4), ("float64", 2, 8)),
)
def test_generated_cuda_plugin_compiles_and_exports_descriptor(
    codegen, tmp_path, precision, scalar_kind, scalar_size
):
    contract = json.loads(FIXTURE_CONTRACT.read_text())
    metadata = codegen.jit_r1_cuda_plugin_metadata(contract, 80, precision=precision)
    source = codegen.render_jit_r1_cuda_plugin(contract, 80, precision=precision)
    assert "SYMMETRIX_JIT_CUDA_PLUGIN_ABI_VERSION_V2" in source
    assert "SYMMETRIX_EXECUTION_CUDA" not in source
    source_path = tmp_path / f"fixture_plugin_{precision}.cu"
    artifact_path = tmp_path / f"fixture_plugin_{precision}.so"
    source_path.write_text(source, encoding="utf-8")
    subprocess.run(
        [
            _nvcc(),
            "-std=c++17",
            "-O2",
            "--shared",
            "--cudart=shared",
            "-Xcompiler=-fPIC",
            "--generate-code=arch=compute_80,code=sm_80",
            str(source_path),
            "-o",
            str(artifact_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    library = ctypes.CDLL(str(artifact_path))
    query = library.symmetrix_jit_cuda_plugin_query_v2
    query.restype = ctypes.POINTER(CudaPlugin)
    plugin = query().contents
    assert plugin.abi_version == 2
    assert plugin.struct_size == ctypes.sizeof(CudaPlugin)
    assert plugin.pointer_size == ctypes.sizeof(ctypes.c_void_p)
    assert plugin.byte_order == 0x01020304
    assert plugin.capabilities == 0x07
    assert (plugin.scalar_kind, plugin.scalar_size) == (
        scalar_kind,
        scalar_size,
    )
    assert plugin.reserved == 0
    assert plugin.abi_tag.decode() == metadata["abi"]
    assert plugin.artifact_id.decode() == metadata["artifact_id"]
    assert plugin.contract_fingerprint.decode() == metadata["contract_fingerprint"]
    assert plugin.semantic_fingerprint.decode() == metadata["semantic_fingerprint"]
    assert plugin.structure_fingerprint.decode() == metadata["structure_fingerprint"]
    assert (plugin.channels, plugin.embedding) == (4, 3)
    assert (plugin.edge_l_max, plugin.source_l_max) == (0, 0)
    assert plugin.target_compute_capability == 80
    assert plugin.forward_threads_per_block == 256
    assert plugin.source_threads_per_block == 256
    assert plugin.edge_threads_per_block == 32
    assert plugin.r1_forward_launch
    assert plugin.r1_coordinate_reverse_launch
