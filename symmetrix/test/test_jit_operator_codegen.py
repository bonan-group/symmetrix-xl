import ctypes
import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest
from symmetrix.jit import artifact_cache_inputs, prepare_jit_artifact
from symmetrix.mace_execution_contract import (
    make_generated_m1_contract,
    make_standard_r0_contract,
)
from symmetrix.mace_jit_operator_codegen import (
    M0_CAPABILITIES_V1,
    M1_CAPABILITIES_V1,
    OPERATOR_ARTIFACT_ENVELOPE,
    R0_CAPABILITIES_V1,
    jit_m0_device_metadata,
    jit_m1_device_metadata,
    jit_r0_device_metadata,
    jit_r0_host_metadata,
    render_jit_m0_device_module,
    render_jit_m0_host_plugin,
    render_jit_m1_device_module,
    render_jit_m1_host_plugin,
    render_jit_r0_device_module,
    render_jit_r0_host_plugin,
    select_product_device_schedule,
)

ALIAS_SAFE_CAPABILITY = 1 << 2

CONTRACTS = Path(__file__).resolve().parent / "data" / "execution_contracts"


def _contract(name):
    return json.loads((CONTRACTS / name).read_text())


def _without_fingerprints(contract):
    contract = deepcopy(contract)
    for name in (
        "fingerprint",
        "semantic_fingerprint",
        "generation_fingerprint",
        "structure_fingerprint",
    ):
        contract.pop(name, None)
    return contract


@pytest.mark.parametrize("schedule", ("chunk32", "table"))
@pytest.mark.parametrize(
    "precision,scalar", (("float32", "float"), ("float64", "double"))
)
def test_m0_device_source_covers_state_free_forward_reverse(
    schedule, precision, scalar
):
    source, metadata = render_jit_m0_device_module(
        _contract("standard_m0_contract.json"),
        precision=precision,
        target="sm_120",
        schedule=schedule,
    )

    assert metadata["capabilities"] == M0_CAPABILITIES_V1
    revision = 2 if schedule == "chunk32" else 1
    assert metadata["artifact_id"].endswith(f"-{schedule}-r{revision}-sm_120")
    assert f"using Scalar = {scalar};" in source
    assert "symmetrix_m0_forward_v1" in source
    assert "symmetrix_m0_reverse_v1" in source
    assert "capture_input_scale_adjoint" in source
    assert "atomicAdd(args.input_scale_adjoint+node, scale)" in source
    assert "long long num_nodes;" in source
    assert "constexpr int channel_count = 128;" in source
    assert "const long long node = owner/channel_count;" in source
    assert "owner/args.channels" not in source
    assert 1_000_188 * 40 * 128 > 2**31 - 1
    if schedule == "chunk32":
        assert "m0_fwd_0" in source
        assert "m0_term_components" not in source
    else:
        assert "m0_term_components" in source
        assert "m0_fwd_0" not in source


def test_m0_artifact_identity_is_channel_target_and_topology_specific():
    contract = _without_fingerprints(_contract("standard_m0_contract.json"))
    baseline = jit_m0_device_metadata(
        contract, precision="float32", target="sm_120", schedule="chunk32"
    )

    resized = deepcopy(contract)
    resized["channels"] = 48
    resized["type_count"] = 7
    resized_metadata = jit_m0_device_metadata(
        resized, precision="float32", target="sm_120", schedule="chunk32"
    )
    assert (
        resized_metadata["structure_fingerprint"] == baseline["structure_fingerprint"]
    )
    assert resized_metadata["artifact_id"] != baseline["artifact_id"]

    resized["channels"] = contract["channels"]
    resized_metadata = jit_m0_device_metadata(
        resized, precision="float32", target="sm_120", schedule="chunk32"
    )
    assert resized_metadata["artifact_id"] == baseline["artifact_id"]

    assert (
        jit_m0_device_metadata(
            contract, precision="float32", target="sm_90", schedule="chunk32"
        )["artifact_id"]
        != baseline["artifact_id"]
    )
    assert (
        jit_m0_device_metadata(
            contract, precision="float64", target="sm_120", schedule="chunk32"
        )["artifact_id"]
        != baseline["artifact_id"]
    )
    assert (
        jit_m0_device_metadata(
            contract, precision="float32", target="sm_120", schedule="table"
        )["artifact_id"]
        != baseline["artifact_id"]
    )

    changed = deepcopy(contract)
    changed["monomial_groups"][0]["terms"][0] = [1]
    changed_metadata = jit_m0_device_metadata(
        changed, precision="float32", target="sm_120", schedule="chunk32"
    )
    assert (
        changed_metadata["structure_fingerprint"] != baseline["structure_fingerprint"]
    )
    assert changed_metadata["artifact_id"] != baseline["artifact_id"]


def test_m0_generated_module_accepts_correlation_four():
    contract = _without_fingerprints(_contract("standard_m0_contract.json"))
    contract["correlation"] = 4
    for group in contract["monomial_groups"]:
        group["degree_counts"].append(0)
    group = contract["monomial_groups"][0]
    group["terms"].append([0, 0, 0, 0])
    group["terms"].sort(key=lambda term: (len(term), tuple(term)))
    group["term_count"] = len(group["terms"])
    group["degree_counts"][3] = 1

    metadata = jit_m0_device_metadata(
        contract, precision="float32", target="sm_120", schedule="chunk32"
    )
    assert metadata["correlation"] == 4
    assert metadata["term_count"] == 423

    source, rendered = render_jit_m0_device_module(
        contract, precision="float64", target="sm_120", schedule="chunk32"
    )
    assert rendered["correlation"] == 4
    assert "x[0]*x[0]*x[0]*x[0]" in source

    table_source, table_metadata = render_jit_m0_device_module(
        contract, precision="float32", target="sm_120", schedule="table"
    )
    assert table_metadata["correlation"] == 4
    assert "term_count*correlation" in table_source
    assert "x[m0_term_components[term*correlation+other]]" in table_source


def test_m0_device_module_rejects_correlation_above_generated_capacity():
    contract = _without_fingerprints(_contract("standard_m0_contract.json"))
    contract["correlation"] = 5
    for group in contract["monomial_groups"]:
        group["degree_counts"].extend([0, 0])

    with pytest.raises(ValueError, match="no greater than four"):
        jit_m0_device_metadata(
            contract, precision="float32", target="sm_120", schedule="chunk32"
        )


def test_m1_device_source_has_distinct_identity_and_scalar_owner_kernels():
    contract = make_generated_m1_contract(
        channels=7,
        type_count=2,
        input_l_max=1,
        correlation=3,
        monomials=[[0], [1, 1], [0, 2, 3]],
    )

    metadata = jit_m1_device_metadata(contract, precision="float32", target="sm_120")
    source, rendered = render_jit_m1_device_module(
        contract, precision="float32", target="sm_120"
    )

    assert rendered == metadata
    assert metadata["operator"] == "M1"
    assert metadata["output_components"] == 1
    assert metadata["generated_identity"]["operator"] == "M1"
    assert "symmetrix_m1_forward_v1" in source
    assert "symmetrix_m1_reverse_v1" in source
    assert "symmetrix_operator_kind_v1 = 3u" in source
    assert "constexpr int channel_count = 7;" in source
    assert "if (args.reserved&1u)" in source
    assert "input_adjoint[input_adjoint_index] += grad[component]" in source


@pytest.mark.parametrize("operator", ("M0", "M1"))
@pytest.mark.parametrize("precision", ("float32", "float64"))
def test_product_tiled128_schedule_uses_persistent_logical_tiles(operator, precision):
    if operator == "M0":
        contract = _without_fingerprints(_contract("standard_m0_contract.json"))
        render = render_jit_m0_device_module
    else:
        contract = make_generated_m1_contract(
            channels=128,
            type_count=1,
            input_l_max=1,
            correlation=3,
            monomials=[[0], [1, 1], [0, 2, 3]],
        )
        render = render_jit_m1_device_module

    source, metadata = render(
        contract,
        precision=precision,
        target="sm_120",
        schedule="tiled128x4-v1",
    )

    assert metadata["abi_version"] == 3
    assert metadata["schedule"] == "tiled128x4-v1"
    assert metadata["schedule_revision"] == (3 if operator == "M0" else 4)
    assert metadata["capabilities"] == (
        M0_CAPABILITIES_V1 if operator == "M0" else M1_CAPABILITIES_V1
    )
    assert metadata["capabilities"] & ALIAS_SAFE_CAPABILITY == 0
    assert metadata["launch_geometry"] == {
        "channel_tile": 128,
        "forward_output_component_tile": 1,
        "forward_node_tile": 1,
        "reverse_input_component_tile": 1,
        "reverse_node_tile": 4,
        "grid": "persistent-1d-logical-tiles",
    }
    assert "constexpr int channel_tile = 128;" in source
    assert "constexpr int node_tile = 1;" in source
    assert "constexpr int node_tile = 4;" in source
    assert "logical_tile%output_components" in source
    assert "logical_tile%input_components" in source
    assert "logical_tile += gridDim.x" in source
    assert "symmetrix_operator_forward_node_tile_v1 = 1" in source
    assert "symmetrix_operator_reverse_node_tile_v1 = 4" in source
    assert "symmetrix_operator_logical_tiles_per_block_v1 = 1" in source
    assert "weights_0" in source
    assert "value_3" in source
    assert "blockIdx.y" not in source
    assert "blockIdx.z" not in source
    assert metadata["artifact_id"] in source


@pytest.mark.parametrize("operator", ("M0", "M1"))
def test_chunk32_product_schedule_is_disjoint_only(operator):
    if operator == "M0":
        contract = _without_fingerprints(_contract("standard_m0_contract.json"))
        render = render_jit_m0_device_module
    else:
        contract = make_generated_m1_contract(
            channels=33,
            type_count=1,
            input_l_max=1,
            correlation=3,
            monomials=[[0], [1, 1], [0, 2, 3]],
        )
        render = render_jit_m1_device_module

    source, metadata = render(
        contract, precision="float32", target="sm_120", schedule="chunk32"
    )

    assert metadata["schedule_revision"] == (2 if operator == "M0" else 3)
    assert metadata["capabilities"] == (
        M0_CAPABILITIES_V1 if operator == "M0" else M1_CAPABILITIES_V1
    )
    assert metadata["capabilities"] & ALIAS_SAFE_CAPABILITY == 0
    assert f"symmetrix_operator_capabilities_v1 = {metadata['capabilities']}u" in source


@pytest.mark.parametrize("operator", ("M0", "M1"))
@pytest.mark.parametrize("schedule", ("chunk32", "tiled128x4-v1"))
def test_product_device_abi_v3_indexes_explicit_node_row_strides(operator, schedule):
    if operator == "M0":
        contract = _without_fingerprints(_contract("standard_m0_contract.json"))
        render = render_jit_m0_device_module
    else:
        contract = make_generated_m1_contract(
            channels=128,
            type_count=2,
            input_l_max=1,
            correlation=3,
            monomials=[[0], [1, 1], [0, 2, 3]],
        )
        render = render_jit_m1_device_module

    source, metadata = render(
        contract, precision="float64", target="sm_120", schedule=schedule
    )

    assert metadata["abi_version"] == 3
    assert "long long input_node_stride;" in source
    assert "long long output_node_stride;" in source
    assert "long long output_adjoint_node_stride;" in source
    assert "long long input_adjoint_node_stride;" in source
    assert "node*args.input_node_stride+component*channel_count+channel" in source
    assert "node*args.output_node_stride+" in source
    assert "node*args.output_adjoint_node_stride+" in source
    assert "node*args.input_adjoint_node_stride+" in source


def test_product_schedule_selection_is_canonical_and_preserves_fallback():
    contract = _without_fingerprints(_contract("standard_m0_contract.json"))
    assert contract["channels"] == 128
    assert (
        select_product_device_schedule(
            contract, precision="float32", backend="cuda", target="sm_120"
        )
        == "chunk32"
    )
    assert (
        select_product_device_schedule(
            contract, precision="float64", backend="cuda", target="sm_120"
        )
        == "chunk32"
    )
    resized = deepcopy(contract)
    resized["channels"] = 512
    assert (
        select_product_device_schedule(
            resized, precision="float32", backend="cuda", target="sm_120"
        )
        == "chunk32"
    )
    resized = deepcopy(contract)
    resized["channels"] = 37
    assert (
        select_product_device_schedule(
            resized, precision="float32", backend="cuda", target="sm_120"
        )
        == "chunk32"
    )
    assert (
        select_product_device_schedule(
            contract, precision="float32", backend="hip", target="gfx1151"
        )
        == "chunk32"
    )


def test_m1_device_renderer_preserves_metadata_text_containing_m0_tokens():
    contract = make_generated_m1_contract(
        channels=7,
        type_count=2,
        input_l_max=1,
        correlation=3,
        monomials=[[0], [1, 1], [0, 2, 3]],
    )

    source, metadata = render_jit_m1_device_module(
        contract,
        precision="float32",
        target="gfx-m0-M0",
    )

    assert metadata["target"] == "gfx-m0-M0"
    assert f'char symmetrix_operator_target_v1[] = "{metadata["target"]}";' in source
    assert metadata["artifact_id"] in source


def test_m1_host_source_uses_distinct_scalar_plugin_abi():
    contract = make_generated_m1_contract(
        channels=7,
        type_count=2,
        input_l_max=1,
        correlation=3,
        monomials=[[0], [1, 1], [0, 2, 3]],
    )

    source, metadata = render_jit_m1_host_plugin(
        contract, precision="float64", target="x86-64-v3"
    )

    assert metadata["operator"] == "M1"
    assert metadata["generated_identity"]["operator"] == "M1"
    assert metadata["output_components"] == 1
    assert "const SymmetrixJitM1HostArgsV1* args" in source
    assert "symmetrix_jit_m1_host_plugin_query_v2" in source
    assert "SYMMETRIX_JIT_M1_HOST_PLUGIN_ABI_VERSION_V2" in source
    assert "SYMMETRIX_M1_ALWAYS_INLINE void m1_fwd_0" in source
    assert "SYMMETRIX_M1_ALWAYS_INLINE void m1_rev_0" in source
    assert "if (args->reserved&1u)" in source
    assert "input_adjoint[input_adjoint_index] += grad[component][lane]" in source


def test_m1_host_plugin_compiles_and_publishes_descriptor(tmp_path):
    cxx = shutil.which("c++")
    if cxx is None:
        pytest.skip("a C++ compiler is required for the host M1 plugin test")
    contract = make_generated_m1_contract(
        channels=7,
        type_count=2,
        input_l_max=1,
        correlation=3,
        monomials=[[0], [1, 1], [0, 2, 3]],
    )
    source, metadata = render_jit_m1_host_plugin(
        contract, precision="float64", target="x86-64-v3"
    )
    result = prepare_jit_artifact(
        source,
        abi=artifact_cache_inputs(metadata, OPERATOR_ARTIFACT_ENVELOPE),
        build={"operator": metadata},
        cache_root=tmp_path / "cache",
        cxx=cxx,
        artifact_name="m1_descriptor",
    )
    assert result.available, (result.reason, result.diagnostics)

    class DescriptorV1(ctypes.Structure):
        _fields_ = [
            ("abi_version", ctypes.c_uint32),
            ("struct_size", ctypes.c_uint32),
            ("pointer_size", ctypes.c_uint32),
            ("byte_order", ctypes.c_uint32),
            ("scalar_kind", ctypes.c_uint32),
            ("scalar_size", ctypes.c_uint32),
            ("capabilities", ctypes.c_uint32),
            ("channels", ctypes.c_int32),
            ("input_components", ctypes.c_int32),
            ("output_components", ctypes.c_int32),
            ("correlation", ctypes.c_int32),
            ("term_count", ctypes.c_int32),
            ("owner_channel_tile", ctypes.c_uint32),
            ("abi_tag", ctypes.c_char_p),
            ("artifact_id", ctypes.c_char_p),
            ("structure_fingerprint", ctypes.c_char_p),
            ("forward_owner", ctypes.c_void_p),
            ("reverse_owner", ctypes.c_void_p),
        ]

    library = ctypes.CDLL(str(result.artifact_path))

    class DescriptorV2(ctypes.Structure):
        _fields_ = [
            ("v1", DescriptorV1),
            ("generated_artifact_id", ctypes.c_char_p),
            ("target", ctypes.c_char_p),
        ]

    query = library.symmetrix_jit_m1_host_plugin_query_v2
    query.restype = ctypes.POINTER(DescriptorV2)
    descriptor_v2 = query().contents
    descriptor = descriptor_v2.v1

    assert descriptor.abi_version == 2
    assert descriptor.struct_size == ctypes.sizeof(DescriptorV2)
    assert descriptor.scalar_kind == 2
    assert descriptor.scalar_size == 8
    assert (
        descriptor.channels,
        descriptor.input_components,
        descriptor.output_components,
        descriptor.correlation,
        descriptor.term_count,
    ) == (7, 4, 1, 3, 3)
    assert descriptor.abi_tag.decode() == metadata["abi"]
    assert descriptor.artifact_id.decode() == metadata["artifact_id"]
    assert (
        descriptor.structure_fingerprint.decode() == metadata["structure_fingerprint"]
    )
    assert descriptor.forward_owner
    assert descriptor.reverse_owner
    assert (
        descriptor_v2.generated_artifact_id.decode()
        == metadata["generated_identity"]["artifact_id"]
    )
    assert descriptor_v2.target.decode() == "x86-64-v3"


@pytest.mark.parametrize("operator", ("M0", "M1"))
@pytest.mark.parametrize("precision", ("float32", "float64"))
def test_host_product_plugins_advertise_disjoint_only_buffers(
    tmp_path, operator, precision
):
    cxx = shutil.which("c++")
    if cxx is None:
        pytest.skip("a C++ compiler is required for the host alias execution test")
    if operator == "M0":
        contract = _without_fingerprints(_contract("standard_m0_contract.json"))
        contract["channels"] = 7
        contract["type_count"] = 1
        render = render_jit_m0_host_plugin
    else:
        contract = make_generated_m1_contract(
            channels=7,
            type_count=1,
            input_l_max=1,
            correlation=3,
            monomials=[[0], [1, 1], [0, 2, 3]],
        )
        render = render_jit_m1_host_plugin
    source, metadata = render(contract, precision=precision, target="host-test")
    result = prepare_jit_artifact(
        source,
        abi=artifact_cache_inputs(metadata, OPERATOR_ARTIFACT_ENVELOPE),
        build={"operator": metadata},
        cache_root=tmp_path / f"{operator}-{precision}",
        cxx=cxx,
        artifact_name=f"{operator.lower()}_{precision}_alias",
    )
    assert result.available, (result.reason, result.diagnostics)

    class Descriptor(ctypes.Structure):
        _fields_ = [
            ("abi_version", ctypes.c_uint32),
            ("struct_size", ctypes.c_uint32),
            ("pointer_size", ctypes.c_uint32),
            ("byte_order", ctypes.c_uint32),
            ("scalar_kind", ctypes.c_uint32),
            ("scalar_size", ctypes.c_uint32),
            ("capabilities", ctypes.c_uint32),
            ("channels", ctypes.c_int32),
            ("input_components", ctypes.c_int32),
            ("output_components", ctypes.c_int32),
            ("correlation", ctypes.c_int32),
            ("term_count", ctypes.c_int32),
            ("owner_channel_tile", ctypes.c_uint32),
            ("abi_tag", ctypes.c_char_p),
            ("artifact_id", ctypes.c_char_p),
            ("structure_fingerprint", ctypes.c_char_p),
            ("forward_owner", ctypes.c_void_p),
            ("reverse_owner", ctypes.c_void_p),
        ]

    library = ctypes.CDLL(str(result.artifact_path))
    query = getattr(library, f"symmetrix_jit_{operator.lower()}_host_plugin_query_v2")
    query.restype = ctypes.POINTER(Descriptor)
    descriptor = query().contents
    assert metadata["capabilities"] == M0_CAPABILITIES_V1
    assert descriptor.capabilities & ALIAS_SAFE_CAPABILITY == 0


@pytest.mark.parametrize("precision", ("float32", "float64"))
def test_m0_host_source_uses_shared_abi_types(precision):
    source, metadata = render_jit_m0_host_plugin(
        _contract("standard_m0_contract.json"), precision=precision
    )

    assert "const SymmetrixJitM0HostArgsV1* args" in source
    assert "const SymmetrixJitM0HostPluginV2*" in source
    assert "static const SymmetrixJitM0HostPluginV2 descriptor" in source
    assert "sizeof(SymmetrixJitM0HostPluginV2)" in source
    assert "SYMMETRIX_JIT_M0_HOST_PLUGIN_BYTE_ORDER_V1" in source
    assert "struct Args" not in source
    assert "struct Plugin" not in source
    assert "input_raw" not in source
    assert metadata["capabilities"] == 0x1B
    assert "constexpr int host_channel_tile" in source
    assert "owner/((channels+host_channel_tile-1)/host_channel_tile)" in source
    assert "host_capabilities" in source
    assert "descriptor_channel_tile" in source
    assert "SYMMETRIX_M0_ALWAYS_INLINE void m0_fwd_0" in source
    assert "SYMMETRIX_M0_ALWAYS_INLINE void m0_rev_0" in source


def test_m0_host_real_loader_binds_generated_identity_and_target(tmp_path):
    cxx = shutil.which("c++")
    if cxx is None:
        pytest.skip("a C++ compiler is required for the host M0 loader test")
    source, metadata = render_jit_m0_host_plugin(
        _contract("standard_m0_contract.json"),
        precision="float32",
        target="x86-64-v3",
    )
    generated_artifact_id = metadata["generated_identity"]["artifact_id"]
    assert generated_artifact_id in source
    assert '"x86-64-v3"' in source

    def compile_plugin(name, plugin_source):
        result = prepare_jit_artifact(
            plugin_source,
            abi=artifact_cache_inputs(metadata, OPERATOR_ARTIFACT_ENVELOPE),
            build={"operator": metadata, "test_variant": name},
            cache_root=tmp_path / name,
            cxx=cxx,
            artifact_name=name,
        )
        assert result.available, (result.reason, result.diagnostics)
        return result.artifact_path

    valid = compile_plugin("valid", source)
    wrong_target_source = source.replace('"x86-64-v3"', '"x86-64-v2"')
    assert wrong_target_source != source
    wrong_target = compile_plugin("wrong-target", wrong_target_source)
    wrong_identity_source = source.replace(
        json.dumps(generated_artifact_id), json.dumps(generated_artifact_id + "-stale")
    )
    assert wrong_identity_source != source
    wrong_identity = compile_plugin("wrong-identity", wrong_identity_source)
    surplus_capability_source = source.replace(
        "? 27u\n    : (SYMMETRIX_JIT_M0_HOST_FORWARD_OWNER_V1",
        "? 31u\n    : (SYMMETRIX_JIT_M0_HOST_ALIAS_SAFE_V1\n"
        "        |SYMMETRIX_JIT_M0_HOST_FORWARD_OWNER_V1",
    )
    assert surplus_capability_source != source
    surplus_capability = compile_plugin(
        "surplus-alias-capability", surplus_capability_source
    )

    probe = tmp_path / "load_m0_host.cpp"
    probe.write_text(
        f"""
#include "mace_jit_m0_host_plugin.hpp"
#include <exception>
#include <iostream>

int main(int argc, char** argv) {{
    const bool strict = argc == 3;
    symmetrix::execution::M0HostPluginExpectation expected{{
        {json.dumps(metadata["artifact_id"])},
        {json.dumps(metadata["structure_fingerprint"])},
        {metadata["scalar_kind"]}u, {metadata["scalar_size"]}u,
        {metadata["contract"]["channels"]}, {metadata["input_components"]},
        {metadata["output_components"]}, {metadata["correlation"]},
        {metadata["term_count"]}}};
    if (strict) {{
        expected.generated_artifact_id = {json.dumps(generated_artifact_id)};
        expected.target = "x86-64-v3";
    }}
    try {{
        auto plugin = symmetrix::execution::M0HostPlugin::load(argv[1], expected);
        return plugin ? 0 : 3;
    }} catch (const std::exception& error) {{
        std::cerr << error.what();
        return 2;
    }}
}}
"""
    )
    loader = tmp_path / "load_m0_host"
    completed = subprocess.run(
        [
            cxx,
            "-std=c++20",
            str(probe),
            str(
                Path(__file__).resolve().parents[2]
                / "libsymmetrix/source/mace_jit_m0_host_plugin.cpp"
            ),
            f"-I{Path(__file__).resolve().parents[2] / 'libsymmetrix/source'}",
            f"-I{Path(__file__).resolve().parents[1] / 'source/symmetrix'}",
            "-ldl",
            "-o",
            str(loader),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert subprocess.run([loader, valid, "strict"], check=False).returncode == 0
    assert subprocess.run([loader, valid], check=False).returncode == 0
    for artifact, message in (
        (wrong_target, "target"),
        (wrong_identity, "generated artifact identity"),
        (surplus_capability, "owner capabilities"),
    ):
        rejected = subprocess.run(
            [loader, artifact, "strict"], check=False, capture_output=True, text=True
        )
        assert rejected.returncode == 2
        assert message in rejected.stderr


@pytest.mark.parametrize(
    "precision,scalar", (("float32", "float"), ("float64", "double"))
)
def test_r0_device_source_covers_density_and_coordinate_reverse(precision, scalar):
    source, metadata = render_jit_r0_device_module(
        _contract("standard_r0_contract.json"),
        precision=precision,
        target="gfx1100",
    )

    assert metadata["capabilities"] == R0_CAPABILITIES_V1
    assert metadata["artifact_id"] == metadata["generated_identity"]["artifact_id"]
    assert metadata["legacy_artifact_id"].endswith("-receiver-edge16-gfx1100")
    assert f"using Scalar = {scalar};" in source
    for symbol in (
        "symmetrix_r0_density_prepare_v1",
        "symmetrix_r0_forward_v1",
        "symmetrix_r0_reverse_prepare_v1",
        "symmetrix_r0_coordinate_reverse_v1",
    ):
        assert symbol in source
    assert "coordinates_are_unit" in source
    assert "use_precomputed_scale_adjoint" in source
    assert "unsigned int receiver_base;" in source
    assert "args.edge_receivers[edge])\n                -args.receiver_base" in source
    assert "void direct_harmonic_gradients(" in source
    assert "if (lane == 0 && edge_active && gradients == nullptr)" in source
    assert "args.coordinates_are_unit ? sizeof(float) : sizeof(double)" in source
    assert "gradients+coordinate_offset*harmonic_count" in source
    assert "edge_gradients[2*harmonic_count+lm]" in source
    assert "gradients[(3*edge+axis)*harmonic_count+lm]" not in source
    assert "long long num_edges;" in source
    assert "for (long long edge_base=" in source
    assert "static_cast<long long>(edge)*harmonic_count" in source
    assert "bool edge_is_active(" in source
    assert source.count("edge_is_active(") == 4
    assert "if (!edge_is_active(args.cutoff, args.radius[edge])) continue;" in source
    assert 109_020_492 * 3 * 16 > 2**31 - 1


@pytest.mark.parametrize(
    ("target", "precision", "forward_owner", "persistent_blocks", "candidates"),
    (
        ("sm_120", "float32", "receiver-angular-order-team", 3, [2, 3, 6]),
        ("sm_120", "float64", "receiver-angular-order-team", 4, [2, 4, 8]),
        ("gfx1100", "float32", "receiver-channel-all-angular", 4, [2, 4, 8]),
        (
            "gfx1151:sramecc-:xnack-",
            "float32",
            "receiver-channel-all-angular",
            8,
            [4, 8, 16],
        ),
        ("gfx1100", "float64", "receiver-angular-order-team", 4, [2, 4, 8]),
    ),
)
def test_r0_device_schedule_metadata_is_backend_and_architecture_specific(
    target, precision, forward_owner, persistent_blocks, candidates
):
    metadata = jit_r0_device_metadata(
        _contract("standard_r0_contract.json"),
        precision=precision,
        target=target,
    )

    assert metadata["schedule"] == "receiver-edge16"
    assert metadata["schedule_revision"] == 6
    assert metadata["compatibility_version"] == 2
    assert metadata["abi_version"] == 5
    assert metadata["schedule_capabilities"] == [
        "density-receiver",
        "forward-receiver",
        "reverse-prepare-receiver",
        "coordinate-reverse-edge16",
        "persistent-grid",
    ]
    assert metadata["schedule_policy"] == {
        "density_prepare_owner": "receiver-warp",
        "forward_owner": forward_owner,
        "coordinate_reverse_owner": "edge16",
        "edge_team_size": 16,
        "edge_vector_length": 16,
        "threads_per_block": 256,
    }
    assert metadata["architecture_bounds"] == {
        "channels": 128,
        "l_max": 3,
        "harmonic_count": 16,
    }
    assert metadata["persistent_profile"] == {
        "blocks_per_compute_unit": persistent_blocks,
        "calibration_candidates": candidates,
    }
    expected_forward = (
        {
            "extent": "nodes",
            "extent_multiplier": 128,
            "owners_per_block": 256,
            "threads_per_block": 256,
        }
        if forward_owner == "receiver-channel-all-angular"
        else {
            "extent": "nodes",
            "extent_multiplier": 4,
            "owners_per_block": (1 if target.startswith("sm") else 8),
            "threads_per_block": 32 if target.startswith("sm") else 256,
        }
    )
    assert metadata["launch_geometry"] == {
        "density_prepare": {
            "extent": "nodes",
            "extent_multiplier": 1,
            "owners_per_block": 8,
            "threads_per_block": 256,
            "persistent": target.startswith("gfx"),
        },
        "forward": {
            **expected_forward,
            "persistent": target.startswith("gfx"),
        },
        "reverse_prepare": {
            "extent": "nodes",
            "extent_multiplier": 1,
            "owners_per_block": 1,
            "threads_per_block": 256,
            "persistent": target.startswith("gfx"),
        },
        "coordinate_reverse": {
            "extent": "edges",
            "extent_multiplier": 1,
            "owners_per_block": 16,
            "threads_per_block": 256,
            "persistent": True,
        },
    }


@pytest.mark.parametrize("target", ("sm_120", "gfx1100"))
def test_r0_device_edge32_schedule_changes_reverse_ownership(target):
    source, metadata = render_jit_r0_device_module(
        _contract("standard_r0_contract.json"),
        precision="float32",
        target=target,
        schedule="receiver-edge32",
    )

    assert metadata["schedule"] == "receiver-edge32"
    assert metadata["schedule_policy"]["coordinate_reverse_owner"] == "edge32"
    assert metadata["schedule_policy"]["edge_team_size"] == 8
    assert metadata["schedule_policy"]["edge_vector_length"] == 32
    assert metadata["launch_geometry"]["coordinate_reverse"] == {
        "extent": "edges",
        "extent_multiplier": 1,
        "owners_per_block": 8,
        "threads_per_block": 256,
        "persistent": True,
    }
    assert "constexpr int edge_team_size = 8;" in source
    assert "constexpr int edge_vector_length = 32;" in source
    assert "const int edge_team = threadIdx.x/edge_vector_length;" in source


@pytest.mark.parametrize(
    ("target", "precision", "schedule", "persistent_limit", "expected"),
    (
        ("sm_120", "float32", "receiver-edge16", 510, (512, 16384, 4096, 510)),
        ("sm_120", "float64", "receiver-edge32", 680, (512, 16384, 4096, 680)),
        ("gfx1100", "float32", "receiver-edge16", 480, (480, 480, 480, 480)),
    ),
)
def test_r0_medium_persistent_grid_uses_schedule_owner_geometry(
    target, precision, schedule, persistent_limit, expected
):
    metadata = jit_r0_device_metadata(
        _contract("standard_r0_contract.json"),
        precision=precision,
        target=target,
        schedule=schedule,
    )
    geometry = metadata["launch_geometry"]
    extents = {
        "density_prepare": 4096,
        "forward": 4096,
        "reverse_prepare": 4096,
        "coordinate_reverse": 65_537,
    }
    blocks = []
    for stage in (
        "density_prepare",
        "forward",
        "reverse_prepare",
        "coordinate_reverse",
    ):
        policy = geometry[stage]
        owners = extents[stage] * policy["extent_multiplier"]
        required = (owners + policy["owners_per_block"] - 1) // policy[
            "owners_per_block"
        ]
        blocks.append(
            min(required, persistent_limit) if policy["persistent"] else required
        )

    assert tuple(blocks) == expected


def test_r0_device_source_uses_contract_bounds_and_optimized_forward_families():
    cuda_source, cuda_metadata = render_jit_r0_device_module(
        _contract("standard_r0_contract.json"),
        precision="float32",
        target="sm_120",
    )
    hip_source, hip_metadata = render_jit_r0_device_module(
        _contract("standard_r0_contract.json"),
        precision="float32",
        target="gfx1151",
    )

    for source in (cuda_source, hip_source):
        assert "constexpr int channel_count = 128;" in source
        assert "constexpr int r0_l_max = 3;" in source
        assert "constexpr int harmonic_count = 16;" in source
        assert "static_assert(harmonic_count == (r0_l_max+1)*(r0_l_max+1));" in source
        assert "args.channels" not in source
        assert "args.l_max" not in source
        for symbol in (
            "symmetrix_operator_schedule_revision_v1",
            "symmetrix_operator_persistent_blocks_per_compute_unit_v1",
            "symmetrix_operator_r0_channels_v1",
            "symmetrix_operator_r0_l_max_v1",
            "symmetrix_operator_r0_harmonic_count_v1",
            "symmetrix_operator_r0_edge_weight_mode_v1",
            "symmetrix_operator_r0_compatibility_version_v1",
            "symmetrix_operator_r0_schedule_v1",
            "symmetrix_operator_r0_density_extent_multiplier_v1",
            "symmetrix_operator_r0_density_owners_per_block_v1",
            "symmetrix_operator_r0_density_threads_per_block_v1",
            "symmetrix_operator_r0_forward_extent_multiplier_v1",
            "symmetrix_operator_r0_forward_owners_per_block_v1",
            "symmetrix_operator_r0_forward_threads_per_block_v1",
            "symmetrix_operator_r0_reverse_prepare_extent_multiplier_v1",
            "symmetrix_operator_r0_reverse_prepare_owners_per_block_v1",
            "symmetrix_operator_r0_reverse_prepare_threads_per_block_v1",
            "symmetrix_operator_r0_coordinate_reverse_extent_multiplier_v1",
            "symmetrix_operator_r0_coordinate_reverse_owners_per_block_v1",
            "symmetrix_operator_r0_coordinate_reverse_threads_per_block_v1",
        ):
            assert symbol in source
    assert cuda_metadata["schedule_policy"]["forward_owner"] == (
        "receiver-angular-order-team"
    )
    assert "constexpr int forward_teams_per_block = 1;" in cuda_source
    assert "constexpr int forward_threads_per_block = 32;" in cuda_source
    assert "const int angular_order = receiver_l%(r0_l_max+1);" in cuda_source
    assert hip_metadata["schedule_policy"]["forward_owner"] == (
        "receiver-channel-all-angular"
    )
    assert "Scalar accumulator[harmonic_count] = {};" in hip_source


def test_cuda_r0_source_preserves_frozen_cooperative_kernel_mechanics():
    source, metadata = render_jit_r0_device_module(
        _contract("standard_r0_contract.json"),
        precision="float32",
        target="sm_120",
        schedule="receiver-edge16",
    )

    assert metadata["floating_point_policy"] == "cuda-fast-math"
    assert metadata["launch_geometry"]["density_prepare"] == {
        "extent": "nodes",
        "extent_multiplier": 1,
        "owners_per_block": 8,
        "threads_per_block": 256,
        "persistent": False,
    }
    assert metadata["launch_geometry"]["forward"]["owners_per_block"] == 1
    assert metadata["launch_geometry"]["forward"]["threads_per_block"] == 32
    assert "if (forward_team >= forward_teams_per_block) return;" in source
    assert metadata["launch_geometry"]["forward"]["persistent"] is False
    assert metadata["launch_geometry"]["reverse_prepare"]["persistent"] is False
    assert metadata["launch_geometry"]["coordinate_reverse"]["persistent"] is True
    assert "const int density_team = threadIdx.x/32;" in source
    assert "for (int local=lane; local<args.num_neigh[node]; local+=32)" in source
    assert "subgroup_sum(density, 32)" in source
    assert "radial_spline_coordinates(args.radial, args.radius[edge]" in source
    assert "switch (component_count)" in source
    assert "Scalar direction_x = Scalar(0);" in source
    assert "subgroup_sum(force_x, edge_vector_length)" in source
    coordinate_reverse = source.split("void symmetrix_r0_coordinate_reverse_v1", 1)[1]
    assert "extern __shared__ Scalar direct_gradient_storage[];" in coordinate_reverse
    assert "subgroup_sync(edge_vector_length);" in coordinate_reverse
    assert "__syncthreads()" not in coordinate_reverse
    assert "force_x_storage" not in coordinate_reverse
    assert "for (int offset=edge_vector_length/2" not in coordinate_reverse


def test_cuda_r0_nonwide_l4_uses_generic_component_accumulation(tmp_path):
    contract = make_standard_r0_contract(
        channels=128,
        radial_embedding=64,
        edge_l_max=4,
        source_irreps="128x0e",
        edge_irreps="1x0e+1x1o+1x2e+1x3o+1x4e",
        output_irreps="128x0e+128x1o+128x2e+128x3o+128x4e",
        output_l=range(5),
        edge_l=range(5),
        source_l=(0,) * 5,
        instructions=(
            {"connection_mode": "uvu", "has_weight": True, "path_shape": [128, 1]}
            for _ in range(5)
        ),
    )

    source, metadata = render_jit_r0_device_module(
        contract,
        precision="float32",
        target="sm_80",
        schedule="receiver-edge16",
    )

    assert metadata["schedule_policy"]["forward_owner"] == (
        "receiver-angular-order-team"
    )
    assert "default:" in source
    assert "component<component_count" in source

    compiler = shutil.which("nvcc")
    if compiler is None:
        pytest.skip("nvcc is unavailable")
    source_path = tmp_path / "l4-r0.cu"
    output_path = tmp_path / "l4-r0.ptx"
    source_path.write_text(source)
    completed = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O3",
            "--use_fast_math",
            "--ptx",
            "--generate-code=arch=compute_80,code=compute_80",
            str(source_path),
            "-o",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_cuda_r0_strict_policy_changes_identity_and_metadata():
    contract = _contract("standard_r0_contract.json")
    _, fast = render_jit_r0_device_module(
        contract, precision="float32", target="sm_120"
    )
    _, strict = render_jit_r0_device_module(
        contract,
        precision="float32",
        target="sm_120",
        floating_point_policy="strict",
    )

    assert fast["floating_point_policy"] == "cuda-fast-math"
    assert strict["floating_point_policy"] == "strict"
    assert fast["artifact_id"] != strict["artifact_id"]


def test_cuda_wide_r0_uses_compile_safe_forward_fallback(tmp_path):
    contract = _without_fingerprints(_contract("standard_r0_contract.json"))
    contract["channels"] = 1024
    for path in contract["paths"]:
        path["path_shape"][0] = 1024

    source, metadata = render_jit_r0_device_module(
        contract,
        precision="float64",
        target="sm_80",
        schedule="receiver-edge16",
    )

    assert metadata["schedule_policy"]["forward_owner"] == (
        "receiver-channel-all-angular-fallback"
    )
    assert metadata["launch_geometry"]["forward"] == {
        "extent": "nodes",
        "extent_multiplier": 1024,
        "owners_per_block": 256,
        "threads_per_block": 256,
        "persistent": False,
    }
    assert "__shared__ Scalar accumulator_storage" not in source

    compiler = shutil.which("nvcc")
    if compiler is None:
        pytest.skip("nvcc is unavailable")
    source_path = tmp_path / "wide-r0.cu"
    output_path = tmp_path / "wide-r0.ptx"
    source_path.write_text(source)
    completed = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O3",
            "--use_fast_math",
            "--ptx",
            "--generate-code=arch=compute_80,code=compute_80",
            str(source_path),
            "-o",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_hip_r0_keeps_strict_floating_point_policy():
    source, metadata = render_jit_r0_device_module(
        _contract("standard_r0_contract.json"),
        precision="float32",
        target="gfx1151",
        schedule="receiver-edge16",
    )

    assert metadata["floating_point_policy"] == "strict"
    assert "__HIP_DEVICE_COMPILE__" in source


def test_r0_device_schedule_rejects_unknown_reverse_family():
    with pytest.raises(ValueError, match="receiver-edge16.*receiver-edge32"):
        render_jit_r0_device_module(
            _contract("standard_r0_contract.json"),
            precision="float32",
            target="sm_120",
            schedule="edge-batch1",
        )


@pytest.mark.parametrize("precision", ("float32", "float64"))
def test_r0_device_source_specializes_edge_weight_mode(precision):
    contract = _contract("standard_r0_contract.json")
    ordinary_source, ordinary_metadata = render_jit_r0_device_module(
        contract,
        precision=precision,
        target="gfx1151",
        apply_edge_weights=False,
    )
    weighted_source, weighted_metadata = render_jit_r0_device_module(
        contract,
        precision=precision,
        target="gfx1151",
        apply_edge_weights=True,
    )

    assert ordinary_metadata["edge_weight_mode"] == "unity"
    assert weighted_metadata["edge_weight_mode"] == "weighted"
    assert ordinary_metadata["artifact_id"] != weighted_metadata["artifact_id"]
    assert (
        ordinary_metadata["legacy_artifact_id"]
        == weighted_metadata["legacy_artifact_id"]
    )
    assert "args.edge_weights[edge]" not in ordinary_source
    assert "args.edge_weights[edge]" in weighted_source


def test_r0_device_module_accepts_runtime_channels_and_rejects_topology_changes():
    contract = _without_fingerprints(_contract("standard_r0_contract.json"))
    baseline = jit_r0_device_metadata(contract, precision="float32", target="sm_120")

    resized = deepcopy(contract)
    resized["channels"] = 48
    for path in resized["paths"]:
        path["path_shape"][0] = 48
    resized_metadata = jit_r0_device_metadata(
        resized, precision="float32", target="sm_120"
    )
    assert (
        resized_metadata["structure_fingerprint"] == baseline["structure_fingerprint"]
    )
    assert resized_metadata["artifact_id"] != baseline["artifact_id"]
    assert resized_metadata["legacy_artifact_id"] == baseline["legacy_artifact_id"]

    assert (
        jit_r0_device_metadata(contract, precision="float32", target="sm_90")[
            "artifact_id"
        ]
        != baseline["artifact_id"]
    )

    perturbed = deepcopy(contract)
    perturbed["sparse_coupling"]["terms"][1]["coefficient"] = 0.5
    with pytest.raises(ValueError):
        jit_r0_device_metadata(perturbed, precision="float32", target="sm_120")


@pytest.mark.parametrize(
    "precision,scalar", (("float32", "float"), ("float64", "double"))
)
def test_r0_host_plugin_specializes_model_architecture(precision, scalar):
    contract = make_standard_r0_contract(
        channels=48,
        radial_embedding=64,
        edge_l_max=2,
        source_irreps="48x0e",
        edge_irreps="1x0e+1x1o+1x2e",
        output_irreps="48x0e+48x1o+48x2e",
        output_l=[0, 1, 2],
        edge_l=[0, 1, 2],
        source_l=[0, 0, 0],
        instructions=[
            {
                "connection_mode": "uvu",
                "has_weight": True,
                "path_shape": [48, 1],
            }
        ]
        * 3,
    )

    source, metadata = render_jit_r0_host_plugin(
        contract, precision=precision, apply_edge_weights=True
    )

    assert metadata == jit_r0_host_metadata(
        contract, precision=precision, apply_edge_weights=True
    )
    assert metadata["contract"]["channels"] == 48
    assert metadata["input_components"] == 9
    assert metadata["edge_weight_mode"] == "weighted"
    assert metadata["stage"] == "R0"
    assert metadata["schedule"] == "host-owner"
    assert metadata["schedule_revision"] == 4
    assert metadata["compatibility_version"] == 2
    assert metadata["abi_version"] == 2
    assert metadata["schedule_policy"] == {
        "density_prepare_owner": "receiver",
        "forward_owner": "receiver-channel-tile",
        "forward_channel_tile": 8,
        "reverse_prepare_owner": "receiver",
        "coordinate_reverse_owner": "receiver",
        "coordinate_reverse_channel_tile": 128,
    }
    assert metadata["artifact_id"] == metadata["generated_identity"]["artifact_id"]
    assert "-l2-c48-weighted" in metadata["legacy_artifact_id"]
    assert f"using Scalar = {scalar};" in source
    assert "constexpr int channel_count = 48;" in source
    assert "constexpr int r0_l_max = 2;" in source
    assert "constexpr int harmonic_count = 9;" in source
    assert "constexpr int forward_channel_tile = 8;" in source
    assert "constexpr int coordinate_reverse_channel_tile = 128;" in source
    assert "static_assert(harmonic_count == (r0_l_max+1)*(r0_l_max+1));" in source
    assert "const std::int64_t node = owner/channel_tile_count;" in source
    assert "Scalar accumulator[harmonic_count*forward_channel_tile] = {};" in source
    assert "for (int lm=l*l; lm<(l+1)*(l+1); ++lm)" in source
    assert "const SymmetrixJitR0HostArgsV1* args, std::int64_t receiver)" in source
    assert "const std::int64_t active_edge_begin = args->edge_begin;" in source
    assert "global_edge<clipped_edge_end" in source
    assert "channel_begin+=coordinate_reverse_channel_tile" in source
    assert "Scalar lane_force_x[coordinate_reverse_channel_tile] = {};" in source
    assert "SymmetrixJitR0HostPluginV2" in source
    for member in (
        "density_prepare_owner",
        "forward_owner",
        "reverse_prepare_owner",
        "coordinate_reverse_owner",
        "density_prepare_range",
        "forward_range",
        "reverse_prepare_range",
        "coordinate_reverse_range",
    ):
        assert member in source


def test_r0_host_plugin_compiles_and_publishes_specialized_descriptor(tmp_path):
    cxx = shutil.which("c++")
    if cxx is None:
        pytest.skip("a C++ compiler is required for the host R0 plugin test")
    contract = make_standard_r0_contract(
        channels=48,
        radial_embedding=64,
        edge_l_max=2,
        source_irreps="48x0e",
        edge_irreps="1x0e+1x1o+1x2e",
        output_irreps="48x0e+48x1o+48x2e",
        output_l=[0, 1, 2],
        edge_l=[0, 1, 2],
        source_l=[0, 0, 0],
        instructions=[
            {"connection_mode": "uvu", "has_weight": True, "path_shape": [48, 1]}
        ]
        * 3,
    )
    source, metadata = render_jit_r0_host_plugin(
        contract, precision="float64", apply_edge_weights=True
    )
    result = prepare_jit_artifact(
        source,
        abi=artifact_cache_inputs(metadata, OPERATOR_ARTIFACT_ENVELOPE),
        build={"operator": metadata},
        cache_root=tmp_path / "cache",
        cxx=cxx,
        artifact_name="r0_l2_c48",
    )
    assert result.available, (result.reason, result.diagnostics)

    class Descriptor(ctypes.Structure):
        _fields_ = [
            ("abi_version", ctypes.c_uint32),
            ("struct_size", ctypes.c_uint32),
            ("pointer_size", ctypes.c_uint32),
            ("byte_order", ctypes.c_uint32),
            ("scalar_kind", ctypes.c_uint32),
            ("scalar_size", ctypes.c_uint32),
            ("capabilities", ctypes.c_uint32),
            ("channels", ctypes.c_int32),
            ("l_max", ctypes.c_int32),
            ("harmonic_count", ctypes.c_int32),
            ("edge_weight_mode", ctypes.c_int32),
            ("abi_tag", ctypes.c_char_p),
            ("artifact_id", ctypes.c_char_p),
            ("structure_fingerprint", ctypes.c_char_p),
            ("density_prepare_owner", ctypes.c_void_p),
            ("forward_owner", ctypes.c_void_p),
            ("reverse_prepare_owner", ctypes.c_void_p),
            ("coordinate_reverse_owner", ctypes.c_void_p),
            ("target", ctypes.c_char_p),
            ("schedule", ctypes.c_char_p),
            ("schedule_revision", ctypes.c_int32),
            ("compatibility_version", ctypes.c_uint32),
            ("density_prepare_range", ctypes.c_void_p),
            ("forward_range", ctypes.c_void_p),
            ("reverse_prepare_range", ctypes.c_void_p),
            ("coordinate_reverse_range", ctypes.c_void_p),
        ]

    library = ctypes.CDLL(str(result.artifact_path))
    query = library.symmetrix_jit_r0_host_plugin_query_v2
    query.restype = ctypes.POINTER(Descriptor)
    descriptor = query().contents
    assert descriptor.struct_size == ctypes.sizeof(Descriptor)
    assert descriptor.artifact_id.decode() == metadata["artifact_id"]
    assert (
        descriptor.structure_fingerprint.decode() == metadata["structure_fingerprint"]
    )
    assert (descriptor.channels, descriptor.l_max, descriptor.harmonic_count) == (
        48,
        2,
        9,
    )
    assert descriptor.edge_weight_mode == 1
    assert descriptor.target.decode() == metadata["target"]
    assert descriptor.schedule.decode() == metadata["schedule"]
    assert descriptor.schedule_revision == metadata["schedule_revision"]
    assert descriptor.compatibility_version == metadata["compatibility_version"]
    assert all(
        (
            descriptor.density_prepare_owner,
            descriptor.forward_owner,
            descriptor.reverse_prepare_owner,
            descriptor.coordinate_reverse_owner,
            descriptor.density_prepare_range,
            descriptor.forward_range,
            descriptor.reverse_prepare_range,
            descriptor.coordinate_reverse_range,
        )
    )


def test_r0_host_real_loader_rejects_old_and_mutated_compatibility(tmp_path):
    cxx = shutil.which("c++")
    if cxx is None:
        pytest.skip("a C++ compiler is required for the host R0 loader test")
    contract = _contract("standard_r0_contract.json")
    source, metadata = render_jit_r0_host_plugin(contract, precision="float64")

    def compile_plugin(name, plugin_source):
        result = prepare_jit_artifact(
            plugin_source,
            abi=artifact_cache_inputs(metadata, OPERATOR_ARTIFACT_ENVELOPE),
            build={"operator": metadata, "test_variant": name},
            cache_root=tmp_path / name,
            cxx=cxx,
            artifact_name=name,
        )
        assert result.available, (result.reason, result.diagnostics)
        return result.artifact_path

    valid = compile_plugin("valid", source)
    old_source = source.replace(
        "symmetrix_jit_r0_host_plugin_query_v2",
        "symmetrix_jit_r0_host_plugin_query_v1",
    )
    assert old_source != source
    old = compile_plugin("old-query-v1", old_source)
    mutated_source = source.replace(
        "constexpr std::uint32_t r0_compatibility_version = 2u;",
        "constexpr std::uint32_t r0_compatibility_version = 1u;",
    )
    assert mutated_source != source
    mutated = compile_plugin("mutated-compatibility", mutated_source)

    probe = tmp_path / "load_r0_host.cpp"
    probe.write_text(
        f"""
#include "mace_jit_r0_host_plugin.hpp"
#include <exception>
#include <iostream>

int main(int argc, char** argv) {{
    const symmetrix::execution::R0LogicalDescriptor expected{{
        {json.dumps(metadata["artifact_id"])},
        {json.dumps(metadata["structure_fingerprint"])},
        {json.dumps(metadata["target"])},
        {json.dumps(metadata["schedule"])},
        {metadata["scalar_kind"]}u, {metadata["scalar_size"]}u,
        symmetrix::execution::r0_required_capabilities,
        {metadata["channels"]}, {metadata["l_max"]},
        {metadata["input_components"]}, {int(metadata["edge_weight_mode"] == "weighted")},
        {metadata["schedule_revision"]}, {metadata["compatibility_version"]}}};
    try {{
        auto plugin = symmetrix::execution::R0HostPlugin::load(argv[1], expected);
        return plugin ? 0 : 3;
    }} catch (const std::exception& error) {{
        std::cerr << error.what();
        return 2;
    }}
}}
"""
    )
    loader = tmp_path / "load_r0_host"
    completed = subprocess.run(
        [
            cxx,
            "-std=c++20",
            str(probe),
            str(
                Path(__file__).resolve().parents[2]
                / "libsymmetrix/source/mace_jit_r0_host_plugin.cpp"
            ),
            f"-I{Path(__file__).resolve().parents[2] / 'libsymmetrix/source'}",
            f"-I{Path(__file__).resolve().parents[1] / 'source/symmetrix'}",
            "-ldl",
            "-o",
            str(loader),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert subprocess.run([loader, valid], check=False).returncode == 0
    old_load = subprocess.run(
        [loader, old], check=False, capture_output=True, text=True
    )
    assert old_load.returncode == 2
    assert "query" in old_load.stderr
    mutated_load = subprocess.run(
        [loader, mutated], check=False, capture_output=True, text=True
    )
    assert mutated_load.returncode == 2
    assert "compatibility_version" in mutated_load.stderr


def test_r0_host_coordinate_reverse_clips_partial_receiver_rows(tmp_path):
    cxx = shutil.which("c++")
    if cxx is None:
        pytest.skip("a C++ compiler is required for the host R0 plugin test")
    contract = make_standard_r0_contract(
        channels=1,
        radial_embedding=1,
        edge_l_max=0,
        source_irreps="1x0e",
        edge_irreps="1x0e",
        output_irreps="1x0e",
        output_l=[0],
        edge_l=[0],
        source_l=[0],
        instructions=[
            {"connection_mode": "uvu", "has_weight": True, "path_shape": [1, 1]}
        ],
    )
    source, metadata = render_jit_r0_host_plugin(contract, precision="float64")
    result = prepare_jit_artifact(
        source,
        abi=artifact_cache_inputs(metadata, OPERATOR_ARTIFACT_ENVELOPE),
        build={"operator": metadata},
        cache_root=tmp_path / "cache",
        cxx=cxx,
        artifact_name="r0_partial_receiver_rows",
    )
    assert result.available, (result.reason, result.diagnostics)

    class Spline(ctypes.Structure):
        _fields_ = [
            ("struct_size", ctypes.c_uint32),
            ("edge_types", ctypes.c_uint32),
            ("intervals", ctypes.c_uint32),
            ("functions", ctypes.c_uint32),
            ("h", ctypes.c_double),
            ("x0", ctypes.c_double),
            ("coefficients", ctypes.c_void_p),
        ]

    class Args(ctypes.Structure):
        _fields_ = [
            ("struct_size", ctypes.c_uint32),
            ("receiver_base", ctypes.c_uint32),
            ("num_nodes", ctypes.c_int64),
            ("num_edges", ctypes.c_int64),
            ("active_type_count", ctypes.c_int32),
            ("channels", ctypes.c_int32),
            ("l_max", ctypes.c_int32),
            ("coordinates_are_unit", ctypes.c_int32),
            ("apply_density_scale", ctypes.c_int32),
            ("use_precomputed_scale_adjoint", ctypes.c_int32),
            ("node_types", ctypes.POINTER(ctypes.c_int32)),
            ("num_neigh", ctypes.POINTER(ctypes.c_int32)),
            ("first_neigh", ctypes.POINTER(ctypes.c_int32)),
            ("neigh_types", ctypes.POINTER(ctypes.c_int32)),
            ("edge_receivers", ctypes.POINTER(ctypes.c_int32)),
            ("type_to_active", ctypes.POINTER(ctypes.c_int32)),
            ("coordinates", ctypes.c_void_p),
            ("radius", ctypes.POINTER(ctypes.c_double)),
            ("radial", Spline),
            ("density", Spline),
            ("harmonics", ctypes.c_void_p),
            ("harmonic_gradients", ctypes.c_void_p),
            ("output", ctypes.c_void_p),
            ("output_adjoint", ctypes.c_void_p),
            ("density_state", ctypes.POINTER(ctypes.c_double)),
            ("precomputed_scale_adjoint", ctypes.POINTER(ctypes.c_double)),
            ("directed_forces", ctypes.POINTER(ctypes.c_double)),
            ("cutoff", ctypes.c_double),
            ("edge_weights", ctypes.POINTER(ctypes.c_double)),
            # Appended to preserve the offsets of every existing ABI field.
            ("edge_begin", ctypes.c_int64),
        ]

    class Descriptor(ctypes.Structure):
        _fields_ = [
            ("abi_version", ctypes.c_uint32),
            ("struct_size", ctypes.c_uint32),
            ("pointer_size", ctypes.c_uint32),
            ("byte_order", ctypes.c_uint32),
            ("scalar_kind", ctypes.c_uint32),
            ("scalar_size", ctypes.c_uint32),
            ("capabilities", ctypes.c_uint32),
            ("channels", ctypes.c_int32),
            ("l_max", ctypes.c_int32),
            ("harmonic_count", ctypes.c_int32),
            ("edge_weight_mode", ctypes.c_int32),
            ("abi_tag", ctypes.c_char_p),
            ("artifact_id", ctypes.c_char_p),
            ("structure_fingerprint", ctypes.c_char_p),
            ("density_prepare_owner", ctypes.c_void_p),
            ("forward_owner", ctypes.c_void_p),
            ("reverse_prepare_owner", ctypes.c_void_p),
            ("coordinate_reverse_owner", ctypes.c_void_p),
            ("target", ctypes.c_char_p),
            ("schedule", ctypes.c_char_p),
            ("schedule_revision", ctypes.c_int32),
            ("compatibility_version", ctypes.c_uint32),
            ("density_prepare_range", ctypes.c_void_p),
            ("forward_range", ctypes.c_void_p),
            ("reverse_prepare_range", ctypes.c_void_p),
            ("coordinate_reverse_range", ctypes.c_void_p),
        ]

    library = ctypes.CDLL(str(result.artifact_path))
    query = library.symmetrix_jit_r0_host_plugin_query_v2
    query.restype = ctypes.POINTER(Descriptor)
    owner_range = ctypes.CFUNCTYPE(
        None, ctypes.POINTER(Args), ctypes.c_int64, ctypes.c_int64
    )(query().contents.coordinate_reverse_range)

    edge_begin = 1
    active_edges = 4
    allocated_edges = 8
    node_types = (ctypes.c_int32 * 2)(0, 0)
    num_neigh = (ctypes.c_int32 * 2)(3, 3)
    first_neigh = (ctypes.c_int32 * 2)(0, 3)
    neigh_types = (ctypes.c_int32 * allocated_edges)(*([0] * allocated_edges))
    type_to_active = (ctypes.c_int32 * 1)(0)
    coordinates = (ctypes.c_double * (3 * allocated_edges))(
        *([1.0, 0.0, 0.0] * allocated_edges)
    )
    radius = (ctypes.c_double * allocated_edges)(*([1.0] * allocated_edges))
    coefficients = (ctypes.c_double * 4)(0.0, 1.0, 0.0, 0.0)
    harmonics = (ctypes.c_double * allocated_edges)(*([1.0] * allocated_edges))
    harmonic_gradients = (ctypes.c_double * (3 * allocated_edges))(
        *([0.0] * (3 * allocated_edges))
    )
    output_adjoint = (ctypes.c_double * 2)(1.0, 2.0)
    density_state = (ctypes.c_double * 2)(1.0, 1.0)
    directed_forces = (ctypes.c_double * (3 * allocated_edges))(
        *([0.0] * (3 * allocated_edges))
    )
    radial = Spline(
        ctypes.sizeof(Spline),
        1,
        1,
        1,
        2.0,
        0.0,
        ctypes.cast(coefficients, ctypes.c_void_p),
    )
    empty_spline = Spline(ctypes.sizeof(Spline), 0, 0, 0, 0.0, 0.0, None)

    def sliced_pointer(array, scalar_type, offset):
        return ctypes.cast(
            ctypes.byref(array, offset * ctypes.sizeof(scalar_type)),
            ctypes.POINTER(scalar_type),
        )

    args = Args(
        ctypes.sizeof(Args),
        0,
        2,
        active_edges,
        1,
        1,
        0,
        0,
        0,
        0,
        node_types,
        num_neigh,
        first_neigh,
        sliced_pointer(neigh_types, ctypes.c_int32, edge_begin),
        None,
        type_to_active,
        ctypes.cast(
            sliced_pointer(coordinates, ctypes.c_double, 3 * edge_begin),
            ctypes.c_void_p,
        ),
        sliced_pointer(radius, ctypes.c_double, edge_begin),
        radial,
        empty_spline,
        ctypes.cast(
            sliced_pointer(harmonics, ctypes.c_double, edge_begin), ctypes.c_void_p
        ),
        ctypes.cast(
            sliced_pointer(harmonic_gradients, ctypes.c_double, 3 * edge_begin),
            ctypes.c_void_p,
        ),
        None,
        ctypes.cast(output_adjoint, ctypes.c_void_p),
        density_state,
        None,
        sliced_pointer(directed_forces, ctypes.c_double, 3 * edge_begin),
        2.0,
        None,
        edge_begin,
    )
    owner_range(ctypes.byref(args), 0, 2)

    x_forces = [directed_forces[3 * edge] for edge in range(allocated_edges)]
    assert x_forces == pytest.approx([0.0, -1.0, -1.0, -2.0, -2.0, 0.0, 0.0, 0.0])
