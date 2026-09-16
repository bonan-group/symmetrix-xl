import pytest
from symmetrix.prediction_heads import (
    available_prediction_heads,
    select_prediction_head,
)


def _multihead_model():
    def payload(energy):
        return {
            "atomic_energies": [energy],
            "readout_1_weights": [energy],
            "readout_2_weights_1": [energy],
            "readout_2_weights_2": [energy],
        }

    return {
        "model_type": "MACE",
        "atomic_energies": [1.0],
        "prediction_heads": {
            "schema_version": 1,
            "names": ["first", "second"],
            "default": "second",
            "parameters": {
                "first": payload(2.0),
                "second": payload(3.0),
            },
        },
    }


def test_select_prediction_head_uses_default_and_requested_payload():
    model = _multihead_model()
    default = select_prediction_head(model)
    first = select_prediction_head(model, "first")

    assert available_prediction_heads(model) == ("first", "second")
    assert default["head"] == "second"
    assert default["atomic_energies"] == [3.0]
    assert first["head"] == "first"
    assert first["atomic_energies"] == [2.0]
    assert first["available_heads"] == ["first", "second"]
    assert "head" not in model


def test_select_prediction_head_accepts_legacy_single_head_json():
    named = {"head": "only", "atomic_energies": [1.0]}
    unnamed = {"atomic_energies": [1.0]}

    assert select_prediction_head(named, "only")["selected_head"] == "only"
    assert select_prediction_head(unnamed)["selected_head"] is None
    assert select_prediction_head(unnamed)["available_heads"] == []

    with pytest.raises(ValueError, match="does not identify"):
        select_prediction_head(unnamed, "other")
    with pytest.raises(ValueError, match="contains 'only'"):
        select_prediction_head(named, "other")


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda section: section.update(schema_version=2), "schema version"),
        (lambda section: section.update(names=["first", "first"]), "unique"),
        (lambda section: section.update(default="missing"), "default"),
        (lambda section: section["parameters"].pop("first"), "every named head"),
        (
            lambda section: section["parameters"]["first"].update(shared_weight=[]),
            "supported fields",
        ),
    ],
)
def test_prediction_head_schema_rejects_invalid_metadata(mutation, message):
    model = _multihead_model()
    mutation(model["prediction_heads"])
    with pytest.raises(ValueError, match=message):
        available_prediction_heads(model)


def test_select_prediction_head_rejects_unknown_name():
    with pytest.raises(ValueError, match="available heads"):
        select_prediction_head(_multihead_model(), "missing")
