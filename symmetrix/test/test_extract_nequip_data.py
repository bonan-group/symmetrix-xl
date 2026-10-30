from __future__ import annotations

import os
import zipfile
from collections import namedtuple

import pytest

import symmetrix.extract_nequip_data as extraction
from symmetrix.extract_nequip_data import extract_nequip_data
from symmetrix.nequip_probe import NequIPProbeError


def test_oam_identity_is_read_from_packaged_archive_root(tmp_path):
    package = tmp_path / "renamed.nequip.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("NequIP_OAM_XL/model/config.yaml", "model: {}\n")

    assert extraction._source_declares_nequip_oam(package)


def test_unrelated_oam_substring_is_not_an_identity_marker(tmp_path):
    assert not extraction._source_declares_nequip_oam(tmp_path / "foam-model.zip")


def test_oam_namedtuple_instruction_preserves_path_metadata():
    instruction_type = namedtuple(
        "Instruction",
        "i_in1 i_in2 i_out connection_mode has_weight path_weight path_shape",
    )
    value = instruction_type(0, 0, 0, "uvw", True, 0.015625, (128, 32, 128))
    result = extraction._oam_instruction(value, 0)
    assert result["path_weight"] == pytest.approx(0.015625)
    assert result["path_shape"] == (128, 32, 128)


def test_wigner_requires_exact_coefficients(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "e3nn", None)
    monkeypatch.setitem(__import__("sys").modules, "e3nn.o3", None)
    with pytest.raises(NequIPProbeError, match="exact Wigner 3j"):
        extraction._wigner(1, 0, 1)


def test_malformed_oam_does_not_downgrade_to_legacy_contract():
    class ConvNetLayer:
        pass

    class BrokenOAM:
        metadata = {"architecture": "nequip-oam"}

        def __init__(self):
            self.layer = ConvNetLayer()

    with pytest.raises(ValueError, match="no InteractionBlock"):
        extract_nequip_data(BrokenOAM())


@pytest.mark.parametrize(
    "variable",
    [
        "SYMMETRIX_NEQUIP_OAM_S",
        "SYMMETRIX_NEQUIP_OAM_M",
        "SYMMETRIX_NEQUIP_OAM_L",
        "SYMMETRIX_NEQUIP_OAM_XL",
    ],
)
def test_optional_oam_package_export(variable):
    source = os.environ.get(variable)
    if not source:
        pytest.skip(f"{variable} is not set")
    pytest.importorskip("torch", reason="optional OAM export requires torch")
    pytest.importorskip("nequip", reason="optional OAM export requires NequIP")
    data = extract_nequip_data(source)
    assert data["model_type"] == "NequIP"
    assert data["architecture"] == "oam"
    assert data["readout"]["weight"]["shape"] == [
        data["readout"]["instructions"][0]["path_shape"][0]
    ]
    assert any(value != 0.0 for value in data["readout"]["weight"]["values"])
