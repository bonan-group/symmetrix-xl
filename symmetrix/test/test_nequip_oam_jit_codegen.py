import copy
import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest

from symmetrix.oam_jit_codegen import oam_jit_metadata, render_oam_host_plugin
from symmetrix.oam_jit_artifact import prepare_nequip_oam_host_artifact


_FIXTURE_HELPER_SPEC = importlib.util.spec_from_file_location(
    "oam_jit_fixture",
    Path(__file__).parent / "data/nequip/oam_jit_fixture.py",
)
assert _FIXTURE_HELPER_SPEC is not None and _FIXTURE_HELPER_SPEC.loader is not None
_FIXTURE_HELPER = importlib.util.module_from_spec(_FIXTURE_HELPER_SPEC)
_FIXTURE_HELPER_SPEC.loader.exec_module(_FIXTURE_HELPER)
prepare_oam_jit_fixture = _FIXTURE_HELPER.prepare_oam_jit_fixture


FIXTURE = Path(__file__).parent / "data/nequip/oam_s_v2_minimal.json"


def _require_cxx():
    if shutil.which(os.environ.get("CXX", "c++")) is None:
        pytest.skip("a C++ compiler is required for the generated OAM host plugin")


def _contract():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_oam_metadata_is_deterministic_and_precision_specific():
    contract = _contract()
    first = oam_jit_metadata(contract, "float64")
    second = oam_jit_metadata(contract, "float64")
    assert first == second
    assert first["artifact_id"] != oam_jit_metadata(contract, "float32")["artifact_id"]
    changed = copy.deepcopy(contract)
    changed["layers"][0]["edge_tensor_product"]["instructions"][0]["path_weight"] = 2
    assert (
        oam_jit_metadata(changed)["semantic_fingerprint"]
        != first["semantic_fingerprint"]
    )


def test_generated_source_has_one_specialized_callback_per_layer():
    source = render_oam_host_plugin(_contract(), "float64")
    metadata = oam_jit_metadata(_contract(), "float64")
    assert "#pragma once" not in source
    assert "symmetrix_jit_oam_host_plugin_query_v1" in source
    for layer in range(metadata["layer_count"]):
        assert f"forward_{layer}" in source
        assert f"reverse_{layer}" in source
    assert "E3TensorProductKokkos" not in source
    assert "generic" not in source.lower()
    assert "instruction_count" in source


def test_invalid_precision_is_rejected():
    with pytest.raises(ValueError, match="precision"):
        oam_jit_metadata(_contract(), "float16")


def test_artifact_preparation_reports_cache_hit(tmp_path):
    _require_cxx()
    first = prepare_nequip_oam_host_artifact(FIXTURE, "float64", cache_dir=tmp_path)
    second = prepare_nequip_oam_host_artifact(FIXTURE, "float64", cache_dir=tmp_path)
    assert first.artifact_path == second.artifact_path
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.compiler_command


def test_fixture_helper_returns_compiled_plugin_and_metadata(tmp_path):
    _require_cxx()
    artifact_path, metadata = prepare_oam_jit_fixture(tmp_path)
    assert artifact_path.is_file()
    assert metadata["artifact_id"].startswith("oam-host-v1-")
    assert metadata["layer_count"] == len(_contract()["layers"])
    assert (artifact_path.parent / "manifest.json").is_file()


def test_v1_reverse_advances_the_weight_offset_only_for_weighted_instructions():
    """Guard the shared renderer against the receiver generation's regression.

    The receiver reverse advanced the instruction weight offset unconditionally
    while this v1 renderer guarded it; both must agree on the offset of the first
    weighted instruction that follows a weightless one.
    """

    contract = _contract()
    product = contract["layers"][0]["edge_tensor_product"]
    weightless = copy.deepcopy(product["instructions"][0])
    weightless["has_weight"] = False
    weightless["path_shape"] = [1, 1]
    product["instructions"] = [weightless, copy.deepcopy(product["instructions"][0])]
    source = render_oam_host_plugin(contract, "float64")
    forward = source.split("void forward_0(", 1)[1].split("\nvoid ", 1)[0]
    reverse = source.split("void reverse_0(", 1)[1].split("\nvoid ", 1)[0]
    # Layer 0's second instruction is the first weighted one and owns slot 0.
    assert "weights[0 + a*1 + b]" in forward
    assert "weights[1 + a*1 + b]" not in forward
    assert "dw[0 + a*1 + b] += dw_acc1;" in reverse
    assert "dw[1 + a*1 + b]" not in reverse
    # The weightless instruction accumulates no weight adjoint at all.
    assert "dw_acc0" not in reverse
