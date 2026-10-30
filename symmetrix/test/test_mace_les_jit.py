import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from compact_r1_model import add_standard_base_contracts, foundation_r1_test_model
from symmetrix import Symmetrix
from symmetrix import symmetrix as native_symmetrix
from symmetrix.kernel_launch_tuning import (
    kernel_launch_tuning_cache_key,
    kernel_launch_tuning_identity,
    kernel_launch_workload_identity,
)
from symmetrix.mace_execution_contract import make_standard_m0_contract
from symmetrix.mace_jit_codegen import render_jit_r1_host_plugin
from symmetrix.prediction_heads import select_prediction_head


def mace_les_evaluator_identity(*args, **kwargs):
    from symmetrix.mace_jit_artifact_common import mace_les_evaluator_identity

    return mace_les_evaluator_identity(*args, **kwargs)


def native_evaluator(*args, **kwargs):
    from symmetrix.mace_jit_artifact_common import native_evaluator

    return native_evaluator(*args, **kwargs)


def _les_payload(*, output_scale=0.25):
    return {
        "schema": "symmetrix.model.mace_les",
        "version": 1,
        "kind": "monopole",
        "periodicity": "geometry",
        "ewald": {
            "sigma": 1.0,
            "dl": 2.0,
            "remove_self_interaction": True,
        },
        "output_scale": output_scale,
        "mace_num_interactions": 2,
        "mace_num_channels": 4,
        "mace_contract_fingerprints": {
            "R0": "sha256:" + "1" * 64,
            "M0": "sha256:" + "2" * 64,
            "R1": "sha256:" + "3" * 64,
        },
        "readout_1_weights": [0.3, -0.2, 0.5, 0.1],
        "readout_2_hidden_size": 3,
        "readout_2_weights_1": [
            0.1,
            -0.2,
            0.3,
            0.4,
            -0.3,
            0.2,
            -0.1,
            0.5,
            0.25,
            0.15,
            -0.35,
            0.45,
        ],
        "readout_2_weights_2": [0.7, -0.4, 0.2],
        "readout_2_scale_factor": 1.5,
    }


@pytest.mark.parametrize(
    ("dtype", "class_name"),
    [("float32", "MACELESKokkosFloat"), ("float64", "MACELESKokkos")],
)
def test_maceles_uses_dedicated_native_evaluator(dtype, class_name):
    evaluator_class = Symmetrix._native_evaluator_class("MACELES", dtype)

    assert evaluator_class.__name__ == class_name


def test_jit_native_evaluator_selects_maceles_class(mace_les_json):
    model_path, _ = mace_les_json

    evaluator = native_evaluator(
        model_path,
        "float64",
        model_type="MACELES",
        error_type=RuntimeError,
        streamed_edges="direct",
    )

    assert type(evaluator).__name__ == "MACELESKokkos"


def test_jit_native_evaluator_rejects_mismatched_model_type(mace_les_json):
    model_path, _ = mace_les_json

    with pytest.raises(RuntimeError, match="declared model_type"):
        native_evaluator(
            model_path,
            "float64",
            model_type="MACE",
            error_type=RuntimeError,
            streamed_edges="direct",
        )


def test_jit_native_evaluator_reports_unsupported_model_type(mace_les_json, tmp_path):
    _, model = mace_les_json
    unsupported_model = json.loads(json.dumps(model))
    unsupported_model["model_type"] = "Unsupported"
    model_path = tmp_path / "unsupported.json"
    model_path.write_text(json.dumps(unsupported_model, separators=(",", ":")))

    with pytest.raises(
        RuntimeError, match="^unsupported compact model_type 'Unsupported'$"
    ):
        native_evaluator(
            model_path,
            "float64",
            model_type="Unsupported",
            error_type=RuntimeError,
            streamed_edges="direct",
        )


def test_jit_native_evaluator_preserves_ambiguous_declaration_error(
    mace_les_json, monkeypatch
):
    from symmetrix import family_registry, mace_family

    model_path, _ = mace_les_json
    maceles_float64 = next(
        variant
        for variant in mace_family.FAMILY_ENTRY.evaluators
        if variant.model_types == ("MACELES",) and variant.precision == "float64"
    )
    ambiguous_entry = replace(
        mace_family.FAMILY_ENTRY,
        evaluators=(*mace_family.FAMILY_ENTRY.evaluators, maceles_float64),
    )
    ambiguous_registry = family_registry.FamilyRegistry((ambiguous_entry,))
    monkeypatch.setattr(
        family_registry, "declared_registry", lambda: ambiguous_registry
    )

    with pytest.raises(RuntimeError) as caught:
        native_evaluator(
            model_path,
            "float64",
            model_type="MACELES",
            error_type=RuntimeError,
            streamed_edges="direct",
        )

    assert str(caught.value) == (
        "family 'MACE' declares 2 evaluators for model_type 'MACELES' and dtype "
        "'float64'; expected exactly one"
    )


def _two_head_les_model(model):
    model = json.loads(json.dumps(model))
    supported_fields = (
        "atomic_energies",
        "readout_1_weights",
        "readout_2_weights_1",
        "readout_2_weights_2",
        "les",
    )
    second = {name: model[name] for name in supported_fields}
    first = json.loads(json.dumps(second))
    first["les"]["ewald"]["sigma"] = 1.25
    first["les"]["output_scale"] = 0.125
    second["les"]["ewald"]["sigma"] = 1.5
    second["les"]["output_scale"] = 0.5
    model.update(second)
    model["prediction_heads"] = {
        "schema_version": 1,
        "names": ["first", "second"],
        "default": "second",
        "parameters": {"first": first, "second": second},
    }
    return model


@pytest.fixture
def mace_les_json(tmp_path):
    contract_path = (
        Path(__file__).resolve().parent
        / "data"
        / "execution_contracts"
        / "jit_r1_fixture_c4_e3_l0_contract.json"
    )
    contract = json.loads(contract_path.read_text())
    model = add_standard_base_contracts(
        foundation_r1_test_model(contract, atomic_numbers=(7, 13))
    )
    model["model_type"] = "MACELES"
    model["les"] = _les_payload()
    model["les"]["mace_contract_fingerprints"] = {
        "R0": model["execution_contracts"]["R0"]["fingerprint"],
        "M0": model["execution_contracts"]["M0"]["semantic_fingerprint"],
        "R1": model["execution_contracts"]["R1"]["fingerprint"],
    }
    path = tmp_path / "mace-les.json"
    path.write_text(json.dumps(model, separators=(",", ":")))
    return path, model


@pytest.fixture(scope="module")
def standard_mace_les_json(tmp_path_factory):
    contracts = Path(__file__).resolve().parent / "data" / "execution_contracts"
    r1_contract = json.loads(
        (contracts / "jit_r1_mace_omat0_small_contract.json").read_text()
    )
    model = foundation_r1_test_model(r1_contract, atomic_numbers=(6, 8))
    model["execution_contracts"]["M0"] = make_standard_m0_contract(
        channels=model["num_channels"],
        type_count=model["num_elements"],
        input_l_max=model["l_max"],
        output_l_max=model["L_max"],
        correlation=1,
        monomials=model["M0_monomials"],
    )
    model["execution_contracts"]["R0"] = json.loads(
        (contracts / "standard_r0_contract.json").read_text()
    )
    channels = model["num_channels"]
    hidden = 3
    model["model_type"] = "MACELES"
    model["les"] = {
        **_les_payload(),
        "mace_num_channels": channels,
        "mace_contract_fingerprints": {
            "R0": model["execution_contracts"]["R0"]["fingerprint"],
            "M0": model["execution_contracts"]["M0"]["semantic_fingerprint"],
            "R1": model["execution_contracts"]["R1"]["fingerprint"],
        },
        "readout_1_weights": [0.001 * ((index % 7) - 3) for index in range(channels)],
        "readout_2_hidden_size": hidden,
        "readout_2_weights_1": [
            0.0001 * ((index % 11) - 5) for index in range(hidden * channels)
        ],
        "readout_2_weights_2": [0.7, -0.4, 0.2],
    }
    path = tmp_path_factory.mktemp("standard-mace-les-jit") / "model.json"
    path.write_text(json.dumps(model, separators=(",", ":")))
    return path, model


def test_maceles_evaluator_identity_records_model_and_canonical_parameters():
    payload = _les_payload()
    reordered = {key: payload[key] for key in reversed(payload)}

    identity = mace_les_evaluator_identity({"model_type": "MACELES", "les": payload})
    reordered_identity = mace_les_evaluator_identity(
        {"model_type": "MACELES", "les": reordered}
    )
    plain = mace_les_evaluator_identity({"model_type": "MACE"})

    assert identity == reordered_identity
    assert identity["model_type"] == "MACELES"
    assert len(identity["mace_les_parameters_sha256"]) == 64
    assert len(identity["mace_les_evaluator_fingerprint"]) == 64
    assert plain["model_type"] == "MACE"
    assert plain["mace_les_parameters_sha256"] is None


def test_distinct_les_payloads_change_evaluator_but_not_operator_identity(
    mace_les_json,
):
    _, first = mace_les_json
    second = json.loads(json.dumps(first))
    second["les"] = _les_payload(output_scale=0.5)

    first_identity = mace_les_evaluator_identity(first)
    second_identity = mace_les_evaluator_identity(second)

    assert (
        first_identity["mace_les_evaluator_fingerprint"]
        != second_identity["mace_les_evaluator_fingerprint"]
    )
    assert render_jit_r1_host_plugin(
        first["execution_contracts"]["R1"], precision="float64"
    ) == render_jit_r1_host_plugin(
        second["execution_contracts"]["R1"], precision="float64"
    )


def test_native_multihead_maceles_loads_default_and_requested_parameters(
    mace_les_json, tmp_path
):
    _, source_model = mace_les_json
    model = _two_head_les_model(source_model)
    path = tmp_path / "multihead-mace-les.json"
    path.write_text(json.dumps(model, separators=(",", ":")))
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()

    default = native_symmetrix.MACELESKokkos(str(path))
    first = native_symmetrix.MACELESKokkos(str(path), "first")
    h1 = np.ones((1, 4), dtype=np.float64)
    h2 = np.zeros((1, 4), dtype=np.float64)
    for evaluator in (default, first):
        evaluator.set_streamed_edges("generic")
        evaluator._evaluate_les_charge_readouts_for_testing(h1, h2)

    assert default.selected_head == "second"
    assert first.selected_head == "first"
    assert default.les_diagnostics["sigma"] == pytest.approx(1.5)
    assert first.les_diagnostics["sigma"] == pytest.approx(1.25)
    np.testing.assert_allclose(default.latent_charges, [0.35])
    np.testing.assert_allclose(first.latent_charges, [0.0875])


def test_native_multihead_maceles_requires_parameters_for_every_head(
    mace_les_json, tmp_path
):
    _, source_model = mace_les_json
    model = _two_head_les_model(source_model)
    model["prediction_heads"]["parameters"]["first"].pop("les")
    path = tmp_path / "incomplete-multihead-mace-les.json"
    path.write_text(json.dumps(model, separators=(",", ":")))
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()

    with pytest.raises(ValueError, match="exactly the supported fields"):
        native_symmetrix.MACELESKokkos(str(path))


def test_native_non_les_multihead_keeps_original_exact_field_set(
    mace_les_json, tmp_path
):
    _, source_model = mace_les_json
    model = json.loads(json.dumps(source_model))
    model["model_type"] = "MACE"
    model.pop("les")
    fields = (
        "atomic_energies",
        "readout_1_weights",
        "readout_2_weights_1",
        "readout_2_weights_2",
    )
    payload = {name: model[name] for name in fields}
    model["prediction_heads"] = {
        "schema_version": 1,
        "names": ["first", "second"],
        "default": "second",
        "parameters": {
            "first": json.loads(json.dumps(payload)),
            "second": json.loads(json.dumps(payload)),
        },
    }
    path = tmp_path / "multihead-mace.json"
    path.write_text(json.dumps(model, separators=(",", ":")))
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()

    evaluator = native_symmetrix.MACEKokkos(str(path), "first")

    assert evaluator.selected_head == "first"
    assert evaluator.available_heads == ["first", "second"]
    assert not hasattr(evaluator, "is_mace_les")


def test_multihead_calculator_identity_uses_selected_maceles_parameters(
    mace_les_json, tmp_path
):
    _, source_model = mace_les_json
    model = _two_head_les_model(source_model)
    path = tmp_path / "multihead-mace-les.json"
    path.write_text(json.dumps(model, separators=(",", ":")))

    default = Symmetrix(
        path,
        dtype="float64",
        streamed_edges="reference",
        neighbor_skin=0.0,
    )
    first = Symmetrix(
        path,
        dtype="float64",
        streamed_edges="reference",
        neighbor_skin=0.0,
        head="first",
    )
    selected_default = select_prediction_head(model)
    selected_first = select_prediction_head(model, "first")
    expected_default = mace_les_evaluator_identity(selected_default)
    expected_first = mace_les_evaluator_identity(selected_first)

    assert default.head == "second"
    assert first.head == "first"
    assert default.model_type == "MACELES"
    assert first.model_type == "MACELES"
    assert (
        default.mace_les_parameters_sha256
        == expected_default["mace_les_parameters_sha256"]
    )
    assert (
        first.mace_les_parameters_sha256 == expected_first["mace_les_parameters_sha256"]
    )
    assert default.mace_les_parameters_sha256 != first.mace_les_parameters_sha256
    assert (
        default.mace_les_evaluator_fingerprint
        == expected_default["mace_les_evaluator_fingerprint"]
    )
    assert (
        first.mace_les_evaluator_fingerprint
        == expected_first["mace_les_evaluator_fingerprint"]
    )
    assert (
        default.mace_les_evaluator_fingerprint != first.mace_les_evaluator_fingerprint
    )
    assert (
        default.execution_plan["mace_les_parameters_sha256"]
        == expected_default["mace_les_parameters_sha256"]
    )
    assert (
        first.execution_plan["mace_les_evaluator_fingerprint"]
        == expected_first["mace_les_evaluator_fingerprint"]
    )
    device_environment = {
        "backend": "hip",
        "architecture": "gfx1151",
        "compute_unit_count": 20,
        "native_subgroup_width": 32,
    }
    workload = kernel_launch_workload_identity(3, 6, ["energy", "forces"])

    def tuning_cache_key(calculator):
        return kernel_launch_tuning_cache_key(
            kernel_launch_tuning_identity(
                device_environment=device_environment,
                implementation_identity="shared-generated-operators",
                model_identity=calculator._kernel_launch_model_identity(),
                precision="float64",
                workload=workload,
            )
        )

    assert tuning_cache_key(default) != tuning_cache_key(first)


def test_required_jit_reuses_base_artifact_across_distinct_les_payloads(
    mace_les_json, monkeypatch, tmp_path
):
    first_path, first_model = mace_les_json
    second_model = json.loads(json.dumps(first_model))
    second_model["les"]["output_scale"] = 0.5
    second_path = tmp_path / "mace-les-second.json"
    second_path.write_text(json.dumps(second_model, separators=(",", ":")))
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))
    monkeypatch.setenv("SYMMETRIX_JIT_POLICY", "required")

    first = Symmetrix(
        first_path,
        dtype="float64",
        streamed_edges="direct",
        execution_profile="speed",
        kernel_launch_policy="static",
    )
    second = Symmetrix(
        second_path,
        dtype="float64",
        streamed_edges="direct",
        execution_profile="speed",
        kernel_launch_policy="static",
    )

    assert first.jit_status == "built"
    assert second.jit_status == "cached"
    assert first.jit_cache_key == second.jit_cache_key
    assert first.jit_artifact_path == second.jit_artifact_path
    assert first.jit_artifact_id == second.jit_artifact_id
    assert Path(first.jit_artifact_path).is_file()
    assert first.mace_les_evaluator_fingerprint != second.mace_les_evaluator_fingerprint

    first_contracts = {
        "R0": first.evaluator.standard_r0_model_contract_fingerprint,
        "M0": first.evaluator.standard_m0_model_structure_fingerprint,
        "R1": first.evaluator.factorized_model_contract_fingerprint,
    }
    second_contracts = {
        "R0": second.evaluator.standard_r0_model_contract_fingerprint,
        "M0": second.evaluator.standard_m0_model_structure_fingerprint,
        "R1": second.evaluator.factorized_model_contract_fingerprint,
    }
    assert (
        first_contracts
        == second_contracts
        == {
            "R0": first_model["execution_contracts"]["R0"]["generation_fingerprint"],
            "M0": first_model["execution_contracts"]["M0"]["structure_fingerprint"],
            "R1": first_model["execution_contracts"]["R1"]["generation_fingerprint"],
        }
    )
    assert (
        first.evaluator.family_artifact_readiness
        == second.evaluator.family_artifact_readiness
    )
    execution_space = native_symmetrix._kokkos_default_execution_space()
    artifact_target = {"Cuda": "cuda", "HIP": "hip"}.get(execution_space, "host")
    assert first.evaluator.family_artifact_readiness == {
        "ready": True,
        "target": artifact_target,
        "artifact_id": first.jit_artifact_id,
        "generation_fingerprint": first_model["execution_contracts"]["R1"][
            "generation_fingerprint"
        ],
        "semantic_fingerprint": first_model["execution_contracts"]["R1"]["fingerprint"],
        "structure_fingerprint": first_model["execution_contracts"]["R1"][
            "structure_fingerprint"
        ],
        "reason": "generated MH-0 artifact admitted",
    }
    del first, second


@pytest.mark.parametrize("mode", ["auto", "direct", "factorized", "direct_streamed"])
def test_les_direct_modes_reach_evaluator_construction(
    mace_les_json, monkeypatch, tmp_path, mode
):
    model_path, _ = mace_les_json
    jit_cache = tmp_path / "jit-cache"
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(jit_cache))

    class EvaluatorConstructionReached(Exception):
        pass

    def record_evaluator_construction(*args, **kwargs):
        raise EvaluatorConstructionReached

    monkeypatch.setattr(
        Symmetrix,
        "_native_evaluator_class",
        staticmethod(record_evaluator_construction),
    )

    with pytest.raises(EvaluatorConstructionReached):
        Symmetrix(
            model_path,
            streamed_edges=mode,
        )
    assert not jit_cache.exists()


def test_maceles_native_direct_configuration_is_model_driven(
    mace_les_json,
):
    model_path, _ = mace_les_json
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()

    evaluator = native_symmetrix.MACELESKokkos(str(model_path))
    evaluator.set_streamed_edges("direct")

    assert evaluator.is_mace_les
    assert evaluator.streamed_edges_mode == "direct"
    del evaluator


def test_maceles_rejected_execution_mode_does_not_mutate_state(mace_les_json):
    model_path, _ = mace_les_json
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()

    evaluator = native_symmetrix.MACELESKokkos(str(model_path))
    evaluator.set_streamed_edges("generic")

    with pytest.raises(
        ValueError,
        match="MACELES streamed_edges='materialized'.*reference or direct",
    ):
        evaluator.set_streamed_edges("materialized")

    assert evaluator.streamed_edges_mode == "generic"
    del evaluator


def test_maceles_native_rejects_fixed_workspace(mace_les_json):
    model_path, _ = mace_les_json
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()

    evaluator = native_symmetrix.MACELESKokkos(str(model_path))
    try:
        evaluator.set_streamed_edges("direct")
        with pytest.raises(ValueError, match="MACELES.*fixed-workspace"):
            evaluator._set_allow_fixed_workspace(True)
        assert not evaluator.allow_fixed_workspace
    finally:
        del evaluator


def test_maceles_native_rejects_alchemical_coefficients(standard_mace_les_json):
    model_path, _ = standard_mace_les_json
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    evaluator = native_symmetrix.MACELESKokkos(str(model_path))
    try:
        evaluator.set_streamed_edges("direct")
        with pytest.raises(ValueError, match="MACELES.*alchemical"):
            evaluator._set_alchemical_weights(0, np.ones(3), np.ones(6))
    finally:
        del evaluator


@pytest.mark.parametrize(
    ("loader", "arguments", "ordinary_error"),
    [
        ("_load_jit_host_plugin", ("/missing/les-r1-host.so",), "host plugin"),
        ("_load_jit_cuda_plugin", ("/missing/les-r1-cuda.so",), "CUDA"),
        ("_load_jit_hip_plugin", ("/missing/les-r1-hip.so",), "HIP"),
        ("_load_jit_device_plugin", ("/missing/les-r1-device.so",), "device"),
        ("_load_m0_host_plugin", ("/missing/les-m0-host.so",), "M0 host plugin"),
        ("_load_m0_device_module", ("/missing/les-m0-device.bin",), "M0 device"),
        ("_load_r0_device_module", ("/missing/les-r0-device.bin",), "R0 device"),
        (
            "load_family_artifact",
            ("/missing/les-family-host.so", "host"),
            "host plugin",
        ),
        (
            "load_family_artifact",
            ("/missing/les-family-device.so", "cuda"),
            "device",
        ),
    ],
)
def test_les_base_artifact_loaders_continue_to_ordinary_validation(
    mace_les_json, loader, arguments, ordinary_error
):
    model_path, _ = mace_les_json
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    evaluator = native_symmetrix.MACELESKokkos(str(model_path))
    evaluator.set_streamed_edges("direct")

    with pytest.raises((RuntimeError, ValueError)) as caught:
        getattr(evaluator, loader)(*arguments)
    message = str(caught.value)
    assert "LES artifact loader" not in message
    if ordinary_error not in message:
        execution_space = native_symmetrix._kokkos_default_execution_space()
        device_loader = {"Cuda": "Execution CUDA plugin", "HIP": "Execution HIP plugin"}
        assert ordinary_error in {"M0 device", "R0 device"}
        assert device_loader[execution_space] in message
        assert arguments[0] in message
    assert evaluator.is_mace_les
    del evaluator


@pytest.mark.parametrize(
    ("loader", "arguments"),
    [
        ("_load_metal_r1_module", ("invalid", 4, 16, 16, 1)),
        ("_load_metal_r0_module", ()),
        ("_load_metal_m0_module", ("invalid", 4, 16, 1, 1)),
        ("_load_metal_m1_module", ("invalid", 4, 16, 1, 1)),
    ],
)
def test_maceles_metal_artifact_loaders_remain_rejected(
    mace_les_json, loader, arguments
):
    model_path, _ = mace_les_json
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    evaluator = native_symmetrix.MACELESKokkos(str(model_path))
    evaluator.set_streamed_edges("generic")

    with pytest.raises(ValueError, match=rf"MACELES.*Metal.*{loader}"):
        getattr(evaluator, loader)(*arguments)
    assert evaluator.is_mace_les
    del evaluator


def test_reference_execution_diagnostic_records_maceles_identity(mace_les_json):
    model_path, model = mace_les_json
    calculator = Symmetrix(
        model_path,
        dtype="float64",
        streamed_edges="reference",
        neighbor_skin=0.0,
    )

    expected = mace_les_evaluator_identity(model)
    assert calculator.model_type == "MACELES"
    assert (
        calculator.mace_les_parameters_sha256 == expected["mace_les_parameters_sha256"]
    )
    assert (
        calculator.mace_les_evaluator_fingerprint
        == expected["mace_les_evaluator_fingerprint"]
    )
    assert calculator.execution_plan["model_type"] == "MACELES"
    assert (
        calculator.execution_plan["mace_les_parameters_sha256"]
        == expected["mace_les_parameters_sha256"]
    )
