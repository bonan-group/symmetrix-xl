from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from symmetrix.jit_codegen_common import spline_table_fingerprint
from symmetrix.nequip_schema import load_nequip_oam_data, validate_nequip_oam_data
from symmetrix.extract_nequip_data import _oam_radial_evaluator


FIXTURE = Path(__file__).parent / "data" / "nequip" / "oam_s_v2_minimal.json"


def _data() -> dict:
    return load_nequip_oam_data(FIXTURE)


def test_oam_fixture_has_one_certified_uniform_radial_table_per_layer():
    data = _data()
    for layer in data["layers"]:
        radial = layer["kokkos_radial"]
        assert radial["edge_type_count"] == 1
        assert radial["spacing"] == pytest.approx(
            (radial["cutoff"] - radial["grid_min"]) / (radial["num_spline_points"] - 1)
        )
        assert radial["certification"]["clamped_points"] == 0
        assert radial["table_fingerprint"].startswith("sha256:")
        assert len(radial["table_fingerprint"]) == 71


def test_oam_fixture_certificate_carries_the_runtime_derived_fingerprints():
    """The shipped certificate must match what the native loader re-derives.

    The loader rebuilds the spline and hashes the sampled nodal values, and
    hashes the contract's exporter fingerprint into the model fingerprint.  A
    stale fixture would therefore fail for a reason unrelated to the test using
    it, so the checked-in values are asserted against the same production
    derivation here, where no native build is required.
    """

    data = _data()
    exporter = data["model_metadata"]["exporter_fingerprint"]
    expected_model = "sha256:" + hashlib.sha256(exporter.encode()).hexdigest()
    for layer in data["layers"]:
        radial = layer["kokkos_radial"]
        evaluate, function_count = _oam_radial_evaluator(
            layer["radial_basis"], cutoff=radial["cutoff"]
        )
        assert radial["model_fingerprint"] == expected_model
        assert radial["table_fingerprint"] == spline_table_fingerprint(
            evaluate,
            function_count,
            num_points=radial["num_spline_points"],
            grid_min=radial["grid_min"],
            cutoff=radial["cutoff"],
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("spacing", 0.1, "uniform grid"),
        ("cutoff", 0.0, "positive"),
        ("cutoff", -1.0, "positive"),
        ("function_count", 2, "function_count"),
        ("edge_type_count", 2, "one"),
        ("model_fingerprint", "sha256:short", "fingerprint"),
        ("table_fingerprint", "sha256:short", "fingerprint"),
    ],
)
def test_oam_radial_certificate_rejects_invalid_grid_and_shape(field, value, message):
    data = _data()
    data["layers"][0]["kokkos_radial"][field] = value
    with pytest.raises(ValueError, match=message):
        validate_nequip_oam_data(data)


def test_oam_radial_certificate_requires_certification_and_rejects_pair_tables():
    missing = _data()
    del missing["layers"][0]["kokkos_radial"]["certification"]
    with pytest.raises(ValueError, match="certification"):
        validate_nequip_oam_data(missing)

    pairs = _data()
    pairs["layers"][0]["kokkos_radial"]["element_pair_tables"] = []
    with pytest.raises(ValueError, match="element-pair"):
        validate_nequip_oam_data(pairs)


def test_legacy_oam_v2_contract_remains_loadable_for_materialized_path():
    data = _data()
    legacy = copy.deepcopy(data)
    for layer in legacy["layers"]:
        del layer["kokkos_radial"]
    validate_nequip_oam_data(legacy)
