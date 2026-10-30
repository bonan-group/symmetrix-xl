from __future__ import annotations

import copy
import json
from pathlib import Path

from symmetrix.oam_architecture import (
    oam_architecture_fingerprint,
    resolve_oam_architecture,
)

FIXTURE = Path(__file__).parent / "data/nequip/oam_s_v2_minimal.json"
KNOWN_ARCHITECTURES = (
    Path(__file__).parents[1] / "source/symmetrix/foundation_oam_architectures.json"
)


def _contract():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_resolved_architecture_reports_actual_layer_widths_and_angular_order():
    architecture = resolve_oam_architecture(_contract())

    assert architecture["layer_count"] == len(architecture["layers"]) == 2
    assert [layer["index"] for layer in architecture["layers"]] == [0, 1]
    assert architecture["layers"][0]["source_dimension"] == 1
    assert architecture["layers"][0]["edge_l_max"] == 0
    assert architecture["layers"][0]["tensor_product"]["weight_numel"] == 1


def test_learned_weights_do_not_change_the_generated_architecture():
    baseline = _contract()
    retrained = copy.deepcopy(baseline)
    retrained["layers"][0]["pre_linear"]["weight"]["values"][0] += 0.25
    retrained["layers"][0]["edge_tensor_product"]["weight"]["values"][0] += 0.5
    retrained["layers"][0]["post_linear"]["weight"]["values"][0] += 0.75
    retrained["layers"][0]["post_linear"]["bias"]["values"][0] -= 0.25

    assert resolve_oam_architecture(retrained) == resolve_oam_architecture(baseline)
    assert oam_architecture_fingerprint(retrained) == oam_architecture_fingerprint(
        baseline
    )


def test_generated_math_change_changes_the_resolved_architecture():
    baseline = _contract()
    changed = copy.deepcopy(baseline)
    changed["layers"][0]["edge_tensor_product"]["instructions"][0]["path_weight"] *= 2.0

    assert resolve_oam_architecture(changed) != resolve_oam_architecture(baseline)
    assert oam_architecture_fingerprint(changed) != oam_architecture_fingerprint(
        baseline
    )


def test_projection_topology_change_changes_the_resolved_architecture():
    baseline = _contract()
    changed = copy.deepcopy(baseline)
    changed["layers"][0]["post_linear"]["instructions"][0]["path_weight"] *= 2.0

    assert resolve_oam_architecture(changed) != resolve_oam_architecture(baseline)
    assert oam_architecture_fingerprint(changed) != oam_architecture_fingerprint(
        baseline
    )


def test_known_architecture_manifest_contains_no_model_path_or_learned_tensors():
    manifest = json.loads(KNOWN_ARCHITECTURES.read_text(encoding="utf-8"))

    assert manifest["schema"] == "symmetrix.foundation-oam-architectures/1"
    assert {entry["id"] for entry in manifest["architectures"]} == {
        "oam-s-v0.1",
        "oam-m-v0.1",
        "oam-l-v0.1",
        "oam-xl-v0.1",
    }
    for entry in manifest["architectures"]:
        assert set(entry) == {
            "id",
            "source_contract_sha256",
            "architecture_fingerprint",
            "architecture",
        }
        architecture = entry["architecture"]
        assert (
            oam_architecture_fingerprint(architecture)
            == entry["architecture_fingerprint"]
        )
        for layer in architecture["layers"]:
            assert "pre_linear" not in layer
            assert "kokkos_radial" not in layer
            assert "weight" not in layer["tensor_product"]
            assert "weight" not in layer["post_linear"]
            assert "bias" not in layer["post_linear"]
    serialized = json.dumps(manifest)
    assert "/home/" not in serialized
    assert ".nequip" not in serialized
