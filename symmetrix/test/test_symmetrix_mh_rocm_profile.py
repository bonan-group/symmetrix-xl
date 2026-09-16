import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
PROFILE_PATH = REPOSITORY / "benchmarks" / "symmetrix_mh_rocm_profile.py"


@pytest.fixture(scope="module")
def profile_module():
    specification = importlib.util.spec_from_file_location(
        "symmetrix_mh_rocm_profile_test", PROFILE_PATH
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _forces_with_scalars(force_l2, force_max):
    full_entries = int((force_l2 / force_max) ** 2)
    remainder_squared = force_l2**2 - full_entries * force_max**2
    values = np.zeros(full_entries + 1)
    values[:full_entries] = force_max
    values[-1] = np.sqrt(max(0.0, remainder_squared))
    return values


@pytest.mark.parametrize("role", ["mh0", "mh1"])
def test_numerical_scalar_references_are_accepted(profile_module, role):
    reference = profile_module._NUMERICAL_REFERENCES[role]
    forces = _forces_with_scalars(
        reference["force_l2_eV_per_A"],
        reference["force_max_abs_eV_per_A"],
    )

    validation = profile_module._validate_numerics(
        role, reference["energy_eV"], forces, np
    )

    assert validation["passed"] is True
    assert validation["actual"] == pytest.approx(reference)
    assert validation["absolute_error"] == pytest.approx(
        {name: 0.0 for name in reference}, abs=1e-12
    )


@pytest.mark.parametrize(
    ("failed_scalar", "energy_offset", "l2_scale", "max_offset"),
    [
        ("energy_eV", 0.021, 1.0, 0.0),
        ("force_l2_eV_per_A", 0.0, 1.001, 0.0),
        ("force_max_abs_eV_per_A", 0.0, 1.0, 0.00021),
    ],
)
def test_numerical_scalar_reference_failures_are_rejected(
    profile_module,
    failed_scalar,
    energy_offset,
    l2_scale,
    max_offset,
):
    reference = profile_module._NUMERICAL_REFERENCES["mh1"]
    force_l2 = reference["force_l2_eV_per_A"] * l2_scale
    force_max = reference["force_max_abs_eV_per_A"] + max_offset
    forces = _forces_with_scalars(force_l2, force_max)
    actual = {
        "energy_eV": reference["energy_eV"] + energy_offset,
        "force_l2_eV_per_A": float(np.linalg.norm(forces)),
        "force_max_abs_eV_per_A": float(np.max(np.abs(forces))),
    }
    errors = {name: abs(actual[name] - reference[name]) for name in reference}
    tolerances = profile_module._NUMERICAL_ABSOLUTE_TOLERANCES
    assert errors[failed_scalar] > tolerances[failed_scalar]
    assert all(
        error <= tolerances[name]
        for name, error in errors.items()
        if name != failed_scalar
    )

    with pytest.raises(
        RuntimeError, match="profile numerical validation failed"
    ) as exc:
        profile_module._validate_numerics("mh1", actual["energy_eV"], forces, np)

    assert failed_scalar in str(exc.value)


class _FakeRoctx:
    def __init__(self, pause_status):
        self.pause_status = pause_status
        self.pause_threads = []

    def roctxProfilerPause(self, thread_id):
        self.pause_threads.append(thread_id)
        return self.pause_status


def test_initial_profiler_pause_accepts_success(profile_module):
    roctx = _FakeRoctx(0)

    profile_module._pause_profiler(roctx)

    assert roctx.pause_threads == [0]


def test_initial_profiler_pause_rejects_failure(profile_module):
    roctx = _FakeRoctx(17)

    with pytest.raises(
        RuntimeError, match="initial roctxProfilerPause failed with status 17"
    ):
        profile_module._pause_profiler(roctx)

    assert roctx.pause_threads == [0]


def test_sha256_hashes_file_contents(profile_module, tmp_path):
    model = tmp_path / "model.json"
    model.write_bytes(b'{"model_type":"MACE"}\n')

    assert (
        profile_module._sha256(model)
        == "8a2dbb76146afe4d3603b145af14ee7ac55e918f5d946e7e8a89b6e4caced359"
    )


def test_parse_args_selects_mh1_node_state_policy(profile_module, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "symmetrix_mh_rocm_profile.py",
            "mh1",
            "model.json",
            "--output",
            "report.json",
            "--node-state-policy",
            "recompute-v1",
        ],
    )

    _, args = profile_module._parse_args()

    assert args.execution_mh1_node_state_policy == "recompute-v1"


def test_parse_args_selects_reused_mh1_adjoint_policy(profile_module, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "symmetrix_mh_rocm_profile.py",
            "mh1",
            "mh1.json",
            "--output",
            "report.json",
            "--node-state-policy",
            "reuse-adjoints-v1",
        ],
    )

    _, args = profile_module._parse_args()

    assert args.execution_mh1_node_state_policy == "reuse-adjoints-v1"


def test_parse_args_selects_retain_interaction_mh1_policy(profile_module, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "symmetrix_mh_rocm_profile.py",
            "mh1",
            "mh1.json",
            "--output",
            "report.json",
            "--node-state-policy",
            "retain-interaction-v1",
        ],
    )

    _, args = profile_module._parse_args()

    assert args.execution_mh1_node_state_policy == "retain-interaction-v1"


def test_jit_metadata_reports_requested_and_active_node_state_policy(profile_module):
    calculator = type(
        "Calculator",
        (),
        {
            "execution_mh1_node_state_policy": "recompute-v1",
            "jit_node_state_policy": "recompute-v1",
        },
    )()

    metadata = profile_module._jit_metadata(calculator)

    assert metadata["node_state_policy_request"] == "recompute-v1"
    assert metadata["node_state_policy"] == "recompute-v1"


class _PartialDiagnosticsEvaluator:
    precision_workspace_bytes = 1234
    node_workspace_bytes = 456
    edge_workspace_bytes = 778
    execution_geometry_workspace_bytes = 90
    execution_mh1_scratch_budget_bytes = 1000
    execution_mh1_scratch_minimum_bytes = 100
    execution_mh1_scratch_planned_bytes = 800
    execution_mh1_scratch_retained_bytes = 600
    execution_mh1_scratch_recomputed_bytes = 200
    execution_mh1_scratch_recomputed_layer_count = 1
    execution_mh1_scratch_budget_satisfied = True


def test_workspace_diagnostics_report_available_and_missing_attributes(profile_module):
    diagnostics = profile_module._workspace_diagnostics(_PartialDiagnosticsEvaluator())

    assert diagnostics["total_bytes"] == {
        "availability": "available",
        "source_attribute": (
            "precision_workspace_bytes + node_workspace_bytes + "
            "execution_geometry_workspace_bytes"
        ),
        "value": 1780,
    }
    assert diagnostics["precision_bytes"]["value"] == 1234
    assert diagnostics["node_bytes"]["value"] == 456
    assert diagnostics["edge_bytes"]["value"] == 778
    assert diagnostics["geometry_bytes"]["value"] == 90
    assert diagnostics["graph_scratch"]["planned_bytes"]["value"] == 800
    assert diagnostics["graph_scratch"]["retained_bytes"]["value"] == 600
    assert diagnostics["graph_scratch"]["recomputed_bytes"]["value"] == 200

    missing = profile_module._workspace_diagnostics(object())
    assert missing["total_bytes"] == {
        "availability": "not_available",
        "source_attribute": None,
        "value": None,
    }
    assert all(
        entry["availability"] == "not_available" and entry["value"] is None
        for entry in missing["graph_scratch"].values()
    )


def test_counter_diagnostics_report_measured_delta_and_unavailability(profile_module):
    diagnostics = profile_module._counter_diagnostics(
        {"available_counter": 7, "missing_counter": None},
        {"available_counter": 10, "missing_counter": None},
    )

    assert diagnostics["available_counter"] == {
        "availability": "available",
        "before": 7,
        "after": 10,
        "delta": 3,
    }
    assert diagnostics["missing_counter"] == {
        "availability": "not_available",
        "before": None,
        "after": None,
        "delta": None,
    }


def test_counter_snapshot_keeps_all_requested_attributes(profile_module):
    evaluator = type("Evaluator", (), {"present": 5})()

    snapshot = profile_module._counter_snapshot(evaluator, ("present", "missing"))

    assert snapshot == {"present": 5, "missing": None}


def test_main_rejects_wrong_role_model_hash_before_runtime_loading(
    profile_module, tmp_path, monkeypatch, capsys
):
    model = tmp_path / "wrong-mh0.json"
    model.write_text(json.dumps({"model_type": "MACE"}), encoding="utf-8")
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(PROFILE_PATH), "mh0", str(model), "--output", str(output)],
    )

    with pytest.raises(SystemExit) as exc:
        profile_module.main()

    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "mh0 model SHA-256" in stderr
    assert profile_module._EXPECTED_MODEL_SHA256["mh0"] in stderr
    assert not output.exists()


class _IdentityAccepted(Exception):
    pass


def test_main_accepts_matching_role_hash_and_model_type(
    profile_module, tmp_path, monkeypatch
):
    model = tmp_path / "mh0.json"
    model.write_text(json.dumps({"model_type": "MACE"}), encoding="utf-8")
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        profile_module,
        "_sha256",
        lambda _path: profile_module._EXPECTED_MODEL_SHA256["mh0"],
    )
    monkeypatch.setattr(
        profile_module,
        "_load_symmetrix",
        lambda _extension, _source_root: (_ for _ in ()).throw(_IdentityAccepted),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [str(PROFILE_PATH), "mh0", str(model), "--output", str(output)],
    )

    with pytest.raises(_IdentityAccepted):
        profile_module.main()


def test_main_rejects_wrong_model_type_after_matching_hash(
    profile_module, tmp_path, monkeypatch, capsys
):
    model = tmp_path / "wrong-type.json"
    model.write_text(json.dumps({"model_type": "MACE_Nonlinear"}), encoding="utf-8")
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        profile_module,
        "_sha256",
        lambda _path: profile_module._EXPECTED_MODEL_SHA256["mh0"],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [str(PROFILE_PATH), "mh0", str(model), "--output", str(output)],
    )

    with pytest.raises(SystemExit) as exc:
        profile_module.main()

    assert exc.value.code == 2
    assert "mh0 requires model_type 'MACE'" in capsys.readouterr().err
    assert not output.exists()
