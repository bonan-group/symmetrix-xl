import shutil
import subprocess
from pathlib import Path

import pytest
from symmetrix.prediction_heads import (
    available_prediction_heads,
    select_prediction_head,
)

ROOT = Path(__file__).resolve().parents[2]


def _multihead_model(model_type="MACE"):
    def payload(energy):
        return {
            "atomic_energies": [energy],
            "readout_1_weights": [energy],
            "readout_2_weights_1": [energy],
            "readout_2_weights_2": [energy],
        }

    return {
        "model_type": model_type,
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


def test_select_prediction_head_projects_matching_maceles_parameters():
    model = _multihead_model("MACELES")
    model["les"] = {"schema": "symmetrix.model.mace_les", "version": 1}
    for index, payload in enumerate(
        model["prediction_heads"]["parameters"].values(), 1
    ):
        payload["les"] = {
            "schema": "symmetrix.model.mace_les",
            "version": 1,
            "readout_1_weights": [11.0 * index],
        }

    selected = select_prediction_head(model, "second")

    assert selected["les"]["readout_1_weights"] == [22.0]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.pop("les"),
        lambda payload: payload.update(extra_les_parameter=[]),
    ],
)
def test_maceles_prediction_head_rejects_missing_or_extra_parameters(mutation):
    model = _multihead_model("MACELES")
    model["les"] = {"schema": "symmetrix.model.mace_les", "version": 1}
    for payload in model["prediction_heads"]["parameters"].values():
        payload["les"] = {"schema": "symmetrix.model.mace_les", "version": 1}
    mutation(model["prediction_heads"]["parameters"]["first"])

    with pytest.raises(ValueError, match="supported fields"):
        available_prediction_heads(model)


def test_ordinary_mace_rejects_intrinsic_les_parameters():
    model = _multihead_model()
    model["les"] = {"schema": "symmetrix.model.mace_les", "version": 1}

    with pytest.raises(ValueError, match="MACELES"):
        available_prediction_heads(model)


@pytest.fixture(scope="module")
def prediction_head_cpp_probe(tmp_path_factory):
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for the prediction-head contract probe")
    directory = tmp_path_factory.mktemp("prediction-head-cpp")
    source = directory / "probe.cpp"
    executable = directory / "probe"
    source.write_text(
        r"""
#include <stdexcept>
#include <string>

#include "mace_prediction_heads.hpp"

int main(int argc, char** argv)
{
    const std::string mode = argc > 1 ? argv[1] : "valid";
    nlohmann::json payload = {
        {"atomic_energies", {1.0}},
        {"readout_1_weights", {2.0}},
        {"readout_2_weights_1", {3.0}},
        {"readout_2_weights_2", {4.0}},
    };
    if (mode != "missing")
        payload["les"] = {{"schema", "symmetrix.model.mace_les"}, {"version", 1}};
    if (mode == "extra")
        payload["extra_les_parameter"] = nlohmann::json::array();
    nlohmann::json data = {
        {"model_type", "MACELES"},
        {"les", {{"schema", "symmetrix.model.mace_les"}, {"version", 1}}},
        {"prediction_heads", {
            {"schema_version", 1},
            {"names", {"only"}},
            {"default", "only"},
            {"parameters", {{"only", payload}}},
        }},
    };
    const bool expect_error = mode != "valid";
    try {
        const auto selected = select_prediction_head(data);
        return expect_error || !selected.contains("les") ? 2 : 0;
    } catch (const std::invalid_argument&) {
        return expect_error ? 0 : 3;
    }
}
"""
    )
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-I",
            str(ROOT / "libsymmetrix/source"),
            "-I",
            str(ROOT / "libsymmetrix/external/json/single_include"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        cwd=ROOT,
    )
    return executable


@pytest.mark.parametrize("mode", ["valid", "missing", "extra"])
def test_cpp_maceles_prediction_head_contract(prediction_head_cpp_probe, mode):
    subprocess.run([prediction_head_cpp_probe, mode], check=True)
