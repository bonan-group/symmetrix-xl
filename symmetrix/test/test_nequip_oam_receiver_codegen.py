"""Regression coverage for OAM host plugin generation (ABI v7)."""

import copy
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from symmetrix.oam_jit_artifact import (
    OAMHostArtifactError,
    _verify_cached_identity,
    prepare_nequip_oam_receiver_host_artifact,
)
from symmetrix.nequip_schema import validate_nequip_oam_data
from symmetrix.oam_architecture import (
    oam_tensor_product_architecture_fingerprint,
    resolve_oam_architecture,
)
from symmetrix.oam_receiver_codegen import (
    OAMHostGenerationPolicy,
    RECEIVER_CODEGEN_VERSION,
    oam_receiver_jit_metadata as _oam_receiver_jit_metadata,
    render_oam_receiver_host_plugin as _render_oam_receiver_host_plugin,
)


FIXTURE = Path(__file__).parent / "data/nequip/oam_s_v2_minimal.json"
HOST_TARGET = "host-isa-x86-64-v3"


def oam_receiver_jit_metadata(*args, **kwargs):
    kwargs.setdefault("host_target", HOST_TARGET)
    return _oam_receiver_jit_metadata(*args, **kwargs)


def render_oam_receiver_host_plugin(*args, **kwargs):
    kwargs.setdefault("host_target", HOST_TARGET)
    return _render_oam_receiver_host_plugin(*args, **kwargs)


def _require_cxx():
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++ compiler is required for the generated OAM host plugin")
    return compiler


def _contract():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_receiver_metadata_is_deterministic_and_precision_specific():
    contract = _contract()
    first = oam_receiver_jit_metadata(contract, "float64")
    assert first == oam_receiver_jit_metadata(contract, "float64")
    assert first["abi_version"] == 7
    assert first["artifact_id"].startswith("oam-receiver-v7-float64-")
    assert (
        first["artifact_id"]
        != oam_receiver_jit_metadata(contract, "float32")["artifact_id"]
    )
    assert first["required_capabilities"] == 1 | 2 | 8
    for row in first["layers"]:
        assert row["target_dimension"] == row["output_dimension"] > 0
    changed = copy.deepcopy(contract)
    changed["layers"][0]["edge_tensor_product"]["instructions"][0]["path_weight"] = 2
    assert (
        oam_receiver_jit_metadata(changed)["semantic_fingerprint"]
        != first["semantic_fingerprint"]
    )


def test_receiver_floating_point_policy_is_identity_bound():
    contract = _contract()
    strict = oam_receiver_jit_metadata(
        contract, "float32", floating_point_policy="strict"
    )
    fast = oam_receiver_jit_metadata(contract, "float32", floating_point_policy="fast")

    assert strict["floating_point_policy"] == "strict"
    assert fast["floating_point_policy"] == "fast"
    assert strict["artifact_id"] != fast["artifact_id"]
    assert strict["generation_fingerprint"] != fast["generation_fingerprint"]
    assert strict["semantic_fingerprint"] == fast["semantic_fingerprint"]


def test_receiver_metadata_requires_a_canonical_host_target():
    with pytest.raises(ValueError, match="canonical host target"):
        _oam_receiver_jit_metadata(_contract(), "float64")
    with pytest.raises(ValueError, match="canonical host target"):
        _oam_receiver_jit_metadata(_contract(), "float64", host_target="x86-64-v3")


def test_receiver_metadata_exposes_only_full_message_execution():
    contract = _contract()
    metadata = oam_receiver_jit_metadata(contract, "float64")
    assert metadata["abi_version"] == 7
    assert metadata["execution_variants"] == ["full_message"]
    assert metadata["required_capabilities"] == 1 | 2 | 8
    assert "projection_layout" not in metadata
    assert metadata["backend"] == "host"
    assert metadata["ownership"] == {
        "forward": "receiver",
        "reverse": "source",
    }
    assert metadata["schedule_identity"].startswith("sha256:")
    for row in metadata["layers"]:
        assert "projected_dimension" not in row
        assert "projection_weight_count" not in row

    retrained = copy.deepcopy(contract)
    weights = retrained["layers"][0]["post_linear"]["weight"]["values"]
    weights[0] = float(weights[0]) + 0.125
    other = oam_receiver_jit_metadata(retrained, "float64")
    assert other["artifact_id"] == metadata["artifact_id"]
    assert other["generation_fingerprint"] == metadata["generation_fingerprint"]


@pytest.mark.parametrize("mutation", ("mask", "path"))
def test_post_linear_topology_does_not_change_full_message_host_artifact(
    mutation,
):
    contract = _contract()
    base = oam_receiver_jit_metadata(contract, "float64")
    changed = copy.deepcopy(contract)
    post = changed["layers"][0]["post_linear"]
    if mutation == "mask":
        post["output_mask"]["values"][0] *= -1
    else:
        post["instructions"][0]["path_weight"] *= 2
    other = oam_receiver_jit_metadata(changed, "float64")

    assert other["artifact_id"] == base["artifact_id"]
    assert other["structure_fingerprint"] == base["structure_fingerprint"]


def test_post_linear_bias_does_not_change_full_message_host_artifact():
    contract = _contract()
    architecture = resolve_oam_architecture(contract)
    base = oam_receiver_jit_metadata(contract, "float64", architecture=architecture)
    changed = copy.deepcopy(contract)
    changed["layers"][0]["post_linear"]["bias"] = {
        "shape": [0],
        "values": [],
    }
    other = oam_receiver_jit_metadata(changed, "float64", architecture=architecture)

    assert other["artifact_id"] == base["artifact_id"]
    assert other["structure_fingerprint"] == base["structure_fingerprint"]


def test_receiver_metadata_publishes_the_conservative_channel_tile():
    """The retained whole-owner control binds its no-split tile explicitly."""

    metadata = oam_receiver_jit_metadata(_contract(), "float64")
    assert metadata["channel_tile"] == 1
    for row in metadata["layers"]:
        assert row["channel_tile"] == 1
    source = render_oam_receiver_host_plugin(_contract(), "float64")
    assert "#pragma once" not in source
    row = re.search(
        r"sizeof\(SymmetrixJitOAMHostLayerV7\), 0u, 1u, 0u, "
        r"\d+, \d+, \d+, \d+, \d+, (\d+), \d+",
        source,
    )
    assert row is not None
    assert int(row.group(1)) == 1


def _weightless_then_weighted_contract():
    """One weightless instruction followed by one weighted instruction."""

    contract = _contract()
    product = contract["layers"][0]["edge_tensor_product"]
    weightless = copy.deepcopy(product["instructions"][0])
    weightless["has_weight"] = False
    weightless["path_shape"] = [1, 1]
    product["instructions"] = [weightless, copy.deepcopy(product["instructions"][0])]
    return contract


def _callback(source, name):
    """Extract one generated callback up to the next function definition."""

    body = source.split(f"void {name}(", 1)[1]
    return body.split("\nvoid ", 1)[0].split('\nextern "C"', 1)[0]


def test_receiver_reverse_advances_weight_offset_only_for_weighted_instructions():
    """A weightless instruction must not shift later weight indices.

    The forward renderer guarded its weight offset but the reverse advanced it
    unconditionally, so any contract with ``has_weight: false`` read every later
    instruction's weights and weight adjoints from the wrong slice.
    """

    contract = _weightless_then_weighted_contract()
    validate_nequip_oam_data(contract)
    metadata = oam_receiver_jit_metadata(contract, "float64")
    forward = _callback(
        render_oam_receiver_host_plugin(contract, "float64"), "oam_receiver_forward_0"
    )
    reverse = _callback(
        render_oam_receiver_host_plugin(contract, "float64"), "oam_source_reverse_0"
    )
    assert metadata["layers"][0]["weight_dimension"] == 1
    # The second instruction is the first weighted one and owns weight slot 0.
    assert "weights[0 + a]" in reverse
    assert "weights_adjoint[0 + a] += dw_acc_1;" in reverse
    assert "weights[1 + a]" not in reverse
    assert "weights_adjoint[1 + a]" not in reverse
    # The weightless instruction accumulates no weight adjoint at all.
    assert "dw_acc_0" not in reverse
    assert "weights[0 + a*1 + b]" in forward


def test_generated_receiver_source_materializes_no_edge_intermediates():
    source = render_oam_receiver_host_plugin(_contract(), "float64")
    # The embedded ABI declarations still describe the retired v1 packets, so
    # only the generated translation unit after the includes is inspected.
    body = source.split("#include <cmath>", 1)[-1]
    metadata = oam_receiver_jit_metadata(_contract(), "float64")
    assert "symmetrix_jit_oam_host_plugin_query_v7" in body
    for layer in range(metadata["layer_count"]):
        assert f"oam_receiver_forward_{layer}" in body
        assert f"oam_source_reverse_{layer}" in body
    for forbidden in (
        "edge_messages",
        "edge_weights",
        "edge_features",
        "E3TensorProductKokkos",
        "Kokkos::",
        "MultilayerPerceptron",
    ):
        assert forbidden not in body, forbidden
    assert "spline_function_count" in body
    assert "receiver_offsets" in body
    assert "source_edge_offsets" in body
    assert "omp simd" in body


def test_generated_receiver_source_exports_full_message_only_abi():
    source = render_oam_receiver_host_plugin(_contract(), "float64")
    assert "symmetrix_jit_oam_host_plugin_query_v7" in source
    assert "symmetrix_jit_oam_host_plugin_query_v3()" not in source
    assert "SymmetrixJitOAMHostPluginV7" in source
    generated = source.split("#include <cmath>", 1)[-1]
    assert "projected_forward_owners[]" not in generated
    assert "projected_reverse_owners[]" not in generated


def test_host_abi_v7_admits_the_full_message_owner_contract():
    metadata = oam_receiver_jit_metadata(_contract(), "float64")
    source = render_oam_receiver_host_plugin(_contract(), "float64")

    assert metadata["abi_version"] == 7
    assert metadata["artifact_id"].startswith("oam-receiver-v7-")
    assert metadata["ownership"] == {
        "forward": "receiver",
        "reverse": "source",
    }
    assert metadata["full_reverse_owner_schedule"] == "source-owner-v1"
    assert metadata["full_reverse_reduction_policy"] == "owner-local-deterministic-v1"
    assert "symmetrix_jit_oam_host_plugin_query_v7" in source
    assert "SymmetrixJitOAMHostPluginV7" in source


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_generated_receiver_source_compiles(precision, tmp_path):
    compiler = _require_cxx()
    source = render_oam_receiver_host_plugin(_contract(), precision)
    include = Path(__file__).resolve().parents[1] / "source" / "symmetrix"
    path = tmp_path / "plugin.cpp"
    path.write_text(source, encoding="ascii")
    completed = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-fPIC",
            "-shared",
            f"-I{include}",
            str(path),
            "-o",
            str(tmp_path / "plugin.so"),
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
    assert completed.returncode == 0
    assert (tmp_path / "plugin.so").is_file()


def test_receiver_artifact_preparation_reports_cache_hit(tmp_path):
    _require_cxx()
    first = prepare_nequip_oam_receiver_host_artifact(
        FIXTURE, "float64", cache_dir=tmp_path
    )
    second = prepare_nequip_oam_receiver_host_artifact(
        FIXTURE, "float64", cache_dir=tmp_path
    )
    assert first.artifact_path == second.artifact_path
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.metadata["abi_version"] == 7


def test_channel_tiled_receiver_artifact_preparation_is_cache_separated(tmp_path):
    _require_cxx()
    policy = OAMHostGenerationPolicy(
        owner_schedule="channel_tiled", channel_tile=3, scalar_width=8
    )
    tiled = prepare_nequip_oam_receiver_host_artifact(
        FIXTURE,
        "float64",
        cache_dir=tmp_path,
        generation_policy=policy,
    )
    conservative = prepare_nequip_oam_receiver_host_artifact(
        FIXTURE, "float64", cache_dir=tmp_path
    )

    assert tiled.artifact_path.is_file()
    assert tiled.artifact_id != conservative.artifact_id
    assert tiled.metadata["channel_tile"] == 3
    assert tiled.metadata["schedule_qualification"] == (
        "candidate-unqualified-explicit"
    )


def test_receiver_cached_artifact_is_bound_to_the_current_contract(tmp_path):
    _require_cxx()
    contract = _contract()
    first = prepare_nequip_oam_receiver_host_artifact(
        FIXTURE, "float64", cache_dir=tmp_path
    )
    assert first.metadata[
        "contract_fingerprint"
    ] == oam_tensor_product_architecture_fingerprint(contract)
    assert first.identity == {
        "artifact_id": first.artifact_id,
        "semantic_fingerprint": first.semantic_fingerprint,
        "structure_fingerprint": first.structure_fingerprint,
        "generation_fingerprint": first.generation_fingerprint,
    }
    assert first.generation_fingerprint in str(
        (first.manifest_path.parent / "manifest.json").read_text(encoding="utf-8")
    )
    second = prepare_nequip_oam_receiver_host_artifact(
        FIXTURE, "float64", cache_dir=tmp_path
    )
    assert second.cache_hit is True
    assert second.semantic_fingerprint == first.semantic_fingerprint


def test_cached_manifest_for_another_architecture_is_refused(tmp_path):
    """A cache hit must not be returned for another generated architecture.

    The guard has no other reachable path: the cache key already embeds the build
    metadata, so only a tampered or hand-assembled cache entry can disagree.
    """

    expected = {
        "artifact_id": "oam-receiver-v2-float64-0123456789abcdef",
        "generation_fingerprint": "sha256:" + "1" * 64,
        "semantic_fingerprint": "sha256:" + "2" * 64,
        "structure_fingerprint": "sha256:" + "3" * 64,
    }
    manifest = {"key_inputs": {"build": {"metadata": dict(expected)}}}
    _verify_cached_identity(FIXTURE, manifest, expected)

    divergent = {
        "key_inputs": {
            "build": {
                "metadata": {**expected, "semantic_fingerprint": "sha256:" + "9" * 64}
            }
        }
    }
    with pytest.raises(OAMHostArtifactError, match="different architecture"):
        _verify_cached_identity(FIXTURE, divergent, expected)

    with pytest.raises(OAMHostArtifactError, match="no build identity metadata"):
        _verify_cached_identity(FIXTURE, {"key_inputs": {}}, expected)


def test_receiver_codegen_version_bumped_for_channel_schedules():
    """Channel schedule identity changes generated source and cache identity."""

    assert RECEIVER_CODEGEN_VERSION == 13


def test_receiver_identity_binds_effective_host_target():
    native_target = "host-native-sha256-" + "a" * 64
    native = oam_receiver_jit_metadata(_contract(), host_target=native_target)
    portable = oam_receiver_jit_metadata(_contract(), host_target=HOST_TARGET)

    assert native["host_target"] == native_target
    assert native["generation_fingerprint"] != portable["generation_fingerprint"]
    assert native["artifact_id"] != portable["artifact_id"]
    source = render_oam_receiver_host_plugin(_contract(), host_target=native_target)
    assert f'"{native_target}"' in source
    assert '"receiver-owner-v1"' in source
    assert '"source-owner-v1"' in source


def test_channel_tiled_host_policy_is_identity_bound_and_explicitly_unqualified():
    policy = OAMHostGenerationPolicy(
        owner_schedule="channel_tiled",
        channel_tile=8,
        scalar_width=4,
        vector_policy="channel-tiled-lane-major-v1",
        required_isa="host-target-bound-v1",
        full_reverse_reduction_policy="owner-local-deterministic-v1",
        resource_policy="bounded-owner-stack-v1",
    )
    tiled = oam_receiver_jit_metadata(_contract(), "float32", generation_policy=policy)
    whole = oam_receiver_jit_metadata(_contract(), "float32")

    assert tiled["generation_fingerprint"] != whole["generation_fingerprint"]
    assert tiled["channel_tile"] == 8
    assert tiled["scalar_width"] == 4
    assert tiled["forward_owner_schedule"] == "receiver-channel-tiled-owner-v1"
    assert tiled["full_reverse_owner_schedule"] == "source-channel-tiled-owner-v1"
    assert tiled["schedule_qualification"] == "candidate-unqualified-explicit"
    assert all(row["channel_tile"] == 8 for row in tiled["layers"])

    changed = oam_receiver_jit_metadata(
        _contract(),
        "float32",
        generation_policy=OAMHostGenerationPolicy(
            owner_schedule="channel_tiled",
            channel_tile=4,
            scalar_width=4,
            vector_policy="channel-tiled-lane-major-v1",
            required_isa="host-target-bound-v1",
            full_reverse_reduction_policy="owner-local-deterministic-v1",
            resource_policy="bounded-owner-stack-v1",
        ),
    )
    assert changed["artifact_id"] != tiled["artifact_id"]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"scalar_width": 8}, "exactly match"),
        ({"vector_policy": "invented-v1"}, "vector policy"),
        ({"required_isa": "x86-64-v3"}, "required ISA"),
        ({"resource_policy": "unbounded-v1"}, "resource policy"),
    ],
)
def test_host_policy_rejects_unimplemented_or_precision_mismatched_tuple(kwargs, match):
    with pytest.raises(ValueError, match=match):
        policy = OAMHostGenerationPolicy(**kwargs)
        oam_receiver_jit_metadata(_contract(), "float32", generation_policy=policy)


def test_host_tile_and_owner_work_range_reject_signed_overflow(tmp_path):
    with pytest.raises(ValueError, match="signed 32-bit"):
        OAMHostGenerationPolicy(owner_schedule="channel_tiled", channel_tile=2**31)
    compiler = _require_cxx()
    source = tmp_path / "owner_range.cpp"
    source.write_text(
        '#include "oam_validation.hpp"\n#include <limits>\n#include <stdexcept>\n'
        "int main(){using symmetrix::execution::checked_oam_owner_work_range;"
        "const int m=std::numeric_limits<int>::max();"
        "auto one=checked_oam_owner_work_range(m,m,1,true);"
        "if(one.tile_count!=1||one.work_items!=1)return 1;"
        "auto edge=checked_oam_owner_work_range(m,1,1,true);"
        "if(edge.tile_count!=m||edge.work_items!=m)return 2;"
        "try{(void)checked_oam_owner_work_range(m,1,2,true);return 3;}"
        "catch(const std::overflow_error&){}"
        "try{(void)checked_oam_owner_work_range(0,1,1,true);return 4;}"
        "catch(const std::overflow_error&){}return 0;}",
        encoding="ascii",
    )
    executable = tmp_path / "owner_range"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-I",
            str(Path(__file__).resolve().parents[2] / "libsymmetrix" / "source"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([executable], check=True, capture_output=True, text=True)


def test_channel_tiled_host_source_handles_nondivisible_widths_and_compiles(tmp_path):
    compiler = _require_cxx()
    policy = OAMHostGenerationPolicy(
        owner_schedule="channel_tiled", channel_tile=3, scalar_width=8
    )
    source = render_oam_receiver_host_plugin(
        _contract(), "float64", generation_policy=policy
    )
    assert "const int receiver = work_item / channel_tile_count;" in source
    assert "const int source = work_item / channel_tile_count;" in source
    assert "const int channel_tile_index = work_item % channel_tile_count;" in source
    assert "if (channel_tile_index == 0)" in source
    assert "channel_begin = channel_tile_index * channel_tile" in source
    assert "channel_begin + channel_tile <" in source
    assert "receiver-channel-tiled-owner-v1" in source
    assert "source-channel-tiled-owner-v1" in source

    include = Path(__file__).resolve().parents[1] / "source" / "symmetrix"
    path = tmp_path / "channel_tiled_plugin.cpp"
    path.write_text(source, encoding="ascii")
    completed = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-fPIC",
            "-shared",
            f"-I{include}",
            str(path),
            "-o",
            str(tmp_path / "channel_tiled_plugin.so"),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"owner_schedule": "tiles"}, "owner schedule"),
        ({"channel_tile": 0}, "channel tile"),
        ({"scalar_width": 3}, "scalar width"),
        ({"full_reverse_reduction_policy": "atomic"}, "full reverse reduction"),
    ],
)
def test_host_schedule_policy_rejects_unsupported_contracts(kwargs, message):
    with pytest.raises(ValueError, match=message):
        OAMHostGenerationPolicy(**kwargs)


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_generated_instruction_math_stays_in_the_plugin_scalar(precision):
    """Mixed double/float arithmetic blocks vectorization in float32 builds.

    The shared instruction renderers historically emitted double accumulators
    and double wigner tables even for float32 plugins, which forced a
    float<->double conversion into every inner-loop product (a measured 5-8x
    slowdown of the generated callbacks on OAM-M/L).
    """

    scalar = "float" if precision == "float32" else "double"
    other = "double" if precision == "float32" else "float"
    source = render_oam_receiver_host_plugin(_contract(), precision)
    assert f"const {other} coeff" not in source
    assert f"static constexpr {other} w" not in source
    assert f"static constexpr {other} rw" not in source
    import re as _re

    assert _re.search(rf"{scalar} sum\d+_\d+ = {scalar}\(0\);", source)


def test_generated_float32_literals_are_valid_numeric_literals():
    """Integral values must not render as `1f` (invalid C++ literal)."""

    source = render_oam_receiver_host_plugin(_contract(), "float32")
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("static constexpr float"):
            assert " 1f" not in f" {stripped}" and not stripped.endswith("1f;")


@pytest.mark.parametrize("mode,shape", [("uvw", [1, 1, 1])])
def test_generated_uvw_instruction_block_compiles(mode, shape, tmp_path):
    """Every admitted connection mode must render balanced, valid codegen.

    The uvw adjoint block closes its own scope; a spurious brace silently
    truncated the enclosing callback for this mode only (production OAM
    contracts use uvu, so this needs its own compile check).
    """

    compiler = _require_cxx()
    contract = _contract()
    instruction = contract["layers"][0]["edge_tensor_product"]["instructions"][0]
    instruction["connection_mode"] = mode
    instruction["path_shape"] = shape
    source = render_oam_receiver_host_plugin(contract, "float64")
    include = Path(__file__).resolve().parents[1] / "source" / "symmetrix"
    path = tmp_path / "plugin.cpp"
    path.write_text(source, encoding="ascii")
    completed = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-fPIC",
            "-shared",
            f"-I{include}",
            str(path),
            "-o",
            str(tmp_path / "plugin.so"),
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
    assert completed.returncode == 0


def test_generated_receiver_source_carries_no_lane_staging():
    """The lane-major generation owns no staging pass or staged pointer."""

    source = render_oam_receiver_host_plugin(_contract(), "float32")
    body = source.split("#include <cmath>", 1)[-1]
    for forbidden in ("oam_stage_reverse", "oam_unstage_reverse", "staged_"):
        assert forbidden not in body, forbidden
    assert "a->source_node_values" in body
    assert "a->target_node_output_adjoint" in body
    assert "a->source_node_input_adjoint" in body
    metadata = oam_receiver_jit_metadata(_contract(), "float32")
    # lane-major | source reverse | receiver forward
    assert metadata["required_capabilities"] == 1 | 2 | 8
    assert "SYMMETRIX_JIT_OAM_HOST_LANE_MAJOR_V3" in body
