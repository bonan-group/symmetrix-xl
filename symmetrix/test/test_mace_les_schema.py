import copy

import pytest
from symmetrix.mace_les_schema import (
    MACE_LES_SCHEMA,
    MACE_LES_SCHEMA_VERSION,
    validate_mace_les_parameters,
)


def _valid_mace_les_parameters():
    return {
        "schema": "symmetrix.model.mace_les",
        "version": 1,
        "kind": "monopole",
        "periodicity": "geometry",
        "ewald": {"sigma": 1.0, "dl": 2.0, "remove_self_interaction": True},
        "output_scale": 0.1,
        "mace_num_interactions": 2,
        "mace_num_channels": 1,
        "mace_contract_fingerprints": {
            "R0": "sha256:" + "1" * 64,
            "M0": "sha256:" + "2" * 64,
            "R1": "sha256:" + "3" * 64,
        },
        "readout_1_weights": [1.0],
        "readout_2_hidden_size": 1,
        "readout_2_weights_1": [2.0],
        "readout_2_weights_2": [3.0],
        "readout_2_scale_factor": 1.0,
    }


def test_validate_mace_les_parameters_accepts_exact_schema():
    normalized = validate_mace_les_parameters(_valid_mace_les_parameters())

    assert MACE_LES_SCHEMA == "symmetrix.model.mace_les"
    assert MACE_LES_SCHEMA_VERSION == 1
    assert normalized["schema"] == "symmetrix.model.mace_les"
    assert normalized["version"] == 1
    assert normalized["kind"] == "monopole"
    assert normalized["readout_1_weights"] == [1.0]
    assert normalized["mace_num_interactions"] == 2
    assert normalized["mace_num_channels"] == 1
    assert normalized["mace_contract_fingerprints"]["M0"] == "sha256:" + "2" * 64


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda data: data.update(version=3), "version"),
        (lambda data: data.update(kind="dipole"), "monopole"),
        (lambda data: data["ewald"].update(dl=0.0), "dl"),
        (lambda data: data["readout_1_weights"].append(0.0), "readout_1_weights"),
        (
            lambda data: data["mace_contract_fingerprints"].update(R1="sha256:bad"),
            "mace_contract_fingerprints.R1",
        ),
    ],
)
def test_validate_mace_les_parameters_rejects_invalid_payload(mutation, message):
    data = copy.deepcopy(_valid_mace_les_parameters())
    mutation(data)

    with pytest.raises(ValueError, match=message):
        validate_mace_les_parameters(data)


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda data: data.update(schema="symmetrix.model.other"), "schema"),
        (
            lambda data: data.update(
                base_num_interactions=data.pop("mace_num_interactions")
            ),
            "unsupported field set",
        ),
        (
            lambda data: data.update(base_num_channels=data.pop("mace_num_channels")),
            "unsupported field set",
        ),
        (
            lambda data: data.update(
                base_contract_fingerprints=data.pop("mace_contract_fingerprints")
            ),
            "unsupported field set",
        ),
    ],
)
def test_validate_mace_les_parameters_rejects_unknown_fields(mutation, message):
    data = copy.deepcopy(_valid_mace_les_parameters())
    mutation(data)

    with pytest.raises(ValueError, match=message):
        validate_mace_les_parameters(data)
