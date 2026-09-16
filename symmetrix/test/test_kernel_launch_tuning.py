import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from symmetrix import kernel_launch_calibrate as calibration
from symmetrix import kernel_launch_tuning as tuning
from symmetrix.calculator import Symmetrix


def _environment(**updates):
    result = {
        "backend": "cuda",
        "architecture": "sm_120",
        "target_features": "",
        "compute_unit_count": 170,
        "native_subgroup_width": 32,
        "max_team_size": 1024,
        "max_shared_memory_per_block": 49152,
        "runtime_version": "13.3",
        "driver_version": "580.65",
    }
    result.update(updates)
    return result


def _identity(**updates):
    arguments = {
        "device_environment": _environment(),
        "implementation_identity": "implementation-a",
        "model_identity": "contract-a",
        "precision": "float32",
        "workload": tuning.kernel_launch_workload_identity(
            4000, 364000, ("energy", "forces")
        ),
    }
    arguments.update(updates)
    return tuning.kernel_launch_tuning_identity(**arguments)


def _decisions():
    return [
        {
            "stage": "r1_forward",
            "profile_id": "artifact-a-cuda-float32-generic",
            "kind": "module",
            "blocks_per_compute_unit": 4,
        },
        {
            "stage": "r1_reverse",
            "profile_id": "artifact-a-cuda-float32-generic",
            "kind": "module",
            "blocks_per_compute_unit": 4,
        },
    ]


class _LaunchEvaluator:
    def __init__(self):
        self.execution_device_execution_environment = _environment(available=True)
        self.factorized_jit_artifact_id = "artifact-a"
        self.factorized_jit_contract_fingerprint = "contract-a"
        self.standard_m0_module_id = ""
        self.standard_m0_module_revision = ""
        self.standard_r0_module_id = ""
        self.standard_r0_module_revision = ""
        self.standard_m0_model_structure_fingerprint = ""
        self.standard_r0_model_structure_fingerprint = ""
        self.events = []

    def _clear_kernel_launch_profile_overrides(self):
        self.events.append(("clear",))

    def _set_kernel_launch_profile_override(self, profile_id, blocks):
        self.events.append(("module", profile_id, blocks))

    def _set_jit_device_plugin_launch_override(self, stage, blocks):
        self.events.append(("plugin", stage, blocks))


def _calculator(policy="automatic"):
    result = object.__new__(Symmetrix)
    result.evaluator = _LaunchEvaluator()
    result.kernel_launch_policy = policy
    result.kernel_launch_tuning_status = "unresolved"
    result.kernel_launch_tuning_cache_key = None
    result.kernel_launch_tuning_reason = None
    result._kernel_launch_tuning_applied_key = None
    result._kernel_launch_dtype = "float32"
    result.jit_cache_key = None
    result.jit_artifact_id = None
    return result


def test_workload_and_identity_are_stable_and_reusable():
    first = _identity()
    second = _identity()

    assert first == second
    assert tuning.kernel_launch_tuning_cache_key(first) == (
        tuning.kernel_launch_tuning_cache_key(second)
    )
    assert first["workload"]["nodes"] == {"lower": 2049, "upper": 4096}
    encoded = json.dumps(first)
    assert "uuid" not in encoded.lower()
    assert "hostname" not in encoded.lower()


@pytest.mark.parametrize(
    "updates",
    [
        {"device_environment": _environment(architecture="sm_86")},
        {"device_environment": _environment(compute_unit_count=108)},
        {"precision": "float64"},
        {
            "workload": tuning.kernel_launch_workload_identity(
                8192, 700000, ("energy", "forces")
            )
        },
    ],
)
def test_identity_separates_hardware_precision_and_workload(updates):
    assert tuning.kernel_launch_tuning_cache_key(_identity()) != (
        tuning.kernel_launch_tuning_cache_key(_identity(**updates))
    )


def test_publish_load_and_remove_round_trip(tmp_path):
    identity = _identity()
    publication = tuning.publish_kernel_launch_tuning_record(
        identity,
        _decisions(),
        {"accepted": True, "improvement_fraction": 0.12},
        cache_root=tmp_path,
    )

    assert publication.published is True
    lookup = tuning.load_kernel_launch_tuning_record(identity, cache_root=tmp_path)
    assert lookup.status == "loaded"
    assert lookup.record["publication_id"] == publication.record["publication_id"]
    assert tuning.remove_kernel_launch_tuning_record(identity, cache_root=tmp_path)
    assert (
        tuning.load_kernel_launch_tuning_record(identity, cache_root=tmp_path).status
        == "missing"
    )


def test_publish_rejects_non_string_json_object_keys(tmp_path):
    with pytest.raises(ValueError, match="JSON object keys must be strings"):
        tuning.publish_kernel_launch_tuning_record(
            _identity(),
            _decisions(),
            {"samples_ms": {4: [1.0]}},
            cache_root=tmp_path,
        )


def test_corrupt_record_is_ignored(tmp_path):
    identity = _identity()
    publication = tuning.publish_kernel_launch_tuning_record(
        identity, _decisions(), {"accepted": True}, cache_root=tmp_path
    )
    path = tmp_path / "records" / f"{publication.cache_key}.json"
    path.write_text("{truncated")

    lookup = tuning.load_kernel_launch_tuning_record(identity, cache_root=tmp_path)

    assert lookup.status == "ignored"
    assert "JSONDecodeError" in lookup.reason


def test_noncanonical_record_is_ignored(tmp_path):
    identity = _identity()
    publication = tuning.publish_kernel_launch_tuning_record(
        identity, _decisions(), {"accepted": True}, cache_root=tmp_path
    )
    path = tmp_path / "records" / f"{publication.cache_key}.json"
    record = json.loads(path.read_text())
    record["measurements"]["score"] = float("nan")
    path.write_text(json.dumps(record))

    lookup = tuning.load_kernel_launch_tuning_record(identity, cache_root=tmp_path)

    assert lookup.status == "ignored"
    assert "ValueError" in lookup.reason


def test_non_directory_cache_root_is_ignored(tmp_path):
    root = tmp_path / "not-a-directory"
    root.write_text("occupied")

    lookup = tuning.load_kernel_launch_tuning_record(_identity(), cache_root=root)

    assert lookup.status == "ignored"
    assert "FileExistsError" in lookup.reason


def test_environment_override_is_explicit(monkeypatch, tmp_path):
    monkeypatch.setenv(tuning.CACHE_ENVIRONMENT_VARIABLE, str(tmp_path))
    assert tuning.kernel_launch_tuning_cache_root() == tmp_path


def test_calculator_automatic_miss_uses_resource_default_without_writes(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(tuning.CACHE_ENVIRONMENT_VARIABLE, str(tmp_path))
    calculator = _calculator()

    changed = calculator._apply_kernel_launch_tuning(4000, 364000, ("energy", "forces"))

    assert calculator.kernel_launch_tuning_status == "resource_default"
    assert calculator.evaluator.events == []
    assert changed is False
    assert list(tmp_path.iterdir()) == []


def test_calculator_static_bypasses_lookup_and_override(monkeypatch):
    calculator = _calculator("static")
    monkeypatch.setattr(
        tuning,
        "load_kernel_launch_tuning_record",
        lambda identity: pytest.fail("static policy must not read tuning records"),
    )

    calculator._apply_kernel_launch_tuning(4000, 364000, ("energy",))

    assert calculator.kernel_launch_tuning_status == "static"
    assert calculator.evaluator.events == []


def test_calculator_openmp_backend_is_ineligible(monkeypatch):
    calculator = _calculator()
    calculator.evaluator.execution_device_execution_environment = _environment(
        available=True,
        backend="host",
        execution_space="OpenMP",
    )
    monkeypatch.setattr(
        calculator,
        "_kernel_launch_implementation_identity",
        lambda: pytest.fail("OpenMP must not build an implementation identity"),
    )
    monkeypatch.setattr(
        tuning,
        "kernel_launch_tuning_identity",
        lambda **kwargs: pytest.fail("OpenMP must not build a tuning identity"),
    )
    monkeypatch.setattr(
        tuning,
        "load_kernel_launch_tuning_record",
        lambda identity: pytest.fail("OpenMP must not read tuning records"),
    )

    calculator._apply_kernel_launch_tuning(4000, 364000, ("energy", "forces"))

    assert calculator.kernel_launch_tuning_status == "ineligible"
    assert calculator.kernel_launch_tuning_cache_key is None
    assert calculator.kernel_launch_tuning_reason == (
        "Execution launch tuning does not support backend 'host'; CUDA or HIP is required"
    )
    assert calculator.evaluator.events == []


def test_calculator_applies_valid_cached_decisions(monkeypatch, tmp_path):
    monkeypatch.setenv(tuning.CACHE_ENVIRONMENT_VARIABLE, str(tmp_path))
    calculator = _calculator()
    identity = tuning.kernel_launch_tuning_identity(
        device_environment=calculator.evaluator.execution_device_execution_environment,
        implementation_identity="artifact-a",
        model_identity="contract-a",
        precision="float32",
        workload=tuning.kernel_launch_workload_identity(
            4000, 364000, ("energy", "forces")
        ),
    )
    tuning.publish_kernel_launch_tuning_record(
        identity, _decisions(), {"accepted": True}, cache_root=tmp_path
    )

    calculator._apply_kernel_launch_tuning(4000, 364000, ("energy", "forces"))

    assert calculator.kernel_launch_tuning_status == "calibrated"
    assert calculator.evaluator.events == [
        ("clear",),
        ("module", "artifact-a-cuda-float32-generic", 4),
        ("module", "artifact-a-cuda-float32-generic", 4),
    ]


def test_calculator_missing_record_clears_previous_calibration(monkeypatch, tmp_path):
    monkeypatch.setenv(tuning.CACHE_ENVIRONMENT_VARIABLE, str(tmp_path))
    calculator = _calculator()
    identity = tuning.kernel_launch_tuning_identity(
        device_environment=calculator.evaluator.execution_device_execution_environment,
        implementation_identity="artifact-a",
        model_identity="contract-a",
        precision="float32",
        workload=tuning.kernel_launch_workload_identity(
            4000, 364000, ("energy", "forces")
        ),
    )
    tuning.publish_kernel_launch_tuning_record(
        identity, _decisions(), {"accepted": True}, cache_root=tmp_path
    )
    assert calculator._apply_kernel_launch_tuning(4000, 364000, ("energy", "forces"))
    calculator.evaluator.events.clear()

    changed = calculator._apply_kernel_launch_tuning(8192, 700000, ("energy", "forces"))

    assert changed is True
    assert calculator.kernel_launch_tuning_status == "resource_default"
    assert calculator.evaluator.events == [("clear",)]


def test_explicit_calibration_selects_and_publishes_bounded_winner(
    monkeypatch, tmp_path
):
    class Evaluator:
        execution_device_execution_environment = _environment(available=True)
        kernel_launch_profile_diagnostics = {
            "r1_forward": {
                "profile_id": "artifact-a-cuda-float32-generic",
                "default_blocks_per_compute_unit": 8,
                "calibration_candidates": [4, 8, 16],
                "calibration_permitted": True,
                "implementation_kind": "module",
            }
        }

        def __init__(self):
            self.blocks = 8

        def _set_kernel_launch_policy(self, policy):
            assert policy == "automatic"

        def _set_kernel_launch_profile_override(self, profile_id, blocks):
            assert profile_id == "artifact-a-cuda-float32-generic"
            self.blocks = blocks

    class Calculator:
        def __init__(self):
            self.evaluator = Evaluator()
            self.results = {}
            self.mace_input_calls = 0

        @property
        def kernel_launch_profile_diagnostics(self):
            return self.evaluator.kernel_launch_profile_diagnostics

        def calculate(self, atoms, properties, system_changes):
            self.results = {
                "energy": -1.0,
                "forces": np.zeros((len(atoms), 3)),
                "stress": np.zeros(6),
            }

        def _mace_inputs(self, atoms):
            self.mace_input_calls += 1
            return len(atoms), [], [], [0, 1], [], [], [], []

        def _kernel_launch_implementation_identity(self):
            return "artifact-a"

        def _kernel_launch_model_identity(self):
            return "contract-a"

    calculator = Calculator()

    def make_calculator(*args, **kwargs):
        assert kwargs["streamed_edges"] == "direct"
        return calculator

    monkeypatch.setattr(calibration, "Symmetrix", make_calculator)
    counter = 0.0

    def clock():
        nonlocal counter
        current = counter
        counter += {4: 0.007, 8: 0.010, 16: 0.012}[calculator.evaluator.blocks]
        return current

    monkeypatch.setattr(calibration.time, "perf_counter", clock)
    result = calibration.calibrate_kernel_launches(
        "model.json",
        Atoms("AlN", positions=[[0, 0, 0], [1, 1, 1]], cell=[4, 4, 4], pbc=True),
        properties=("energy", "forces", "stress"),
        warmups=0,
        samples=2,
        confirmation_samples=1,
        minimum_improvement=0.03,
        cache_root=tmp_path,
    )

    assert result.published is True
    assert calculator.mace_input_calls == 1
    identity = tuning.kernel_launch_tuning_identity(
        device_environment=calculator.evaluator.execution_device_execution_environment,
        implementation_identity="artifact-a",
        model_identity="contract-a",
        precision="float32",
        workload=tuning.kernel_launch_workload_identity(
            2, 2, ("energy", "forces", "stress")
        ),
    )
    lookup = tuning.load_kernel_launch_tuning_record(identity, cache_root=tmp_path)

    assert result.decisions[0]["blocks_per_compute_unit"] == 4
    assert lookup.status == "loaded"
    for field in ("samples_ms", "medians_ms", "rejected", "confirmation_ms"):
        assert all(
            isinstance(key, str)
            for key in lookup.record["measurements"]["groups"][0][field]
        )


def test_explicit_calibration_timeout_retains_static_default(monkeypatch, tmp_path):
    class Evaluator:
        execution_device_execution_environment = _environment(available=True)

        def __init__(self):
            self.kernel_launch_profile_diagnostics = {
                "r1_forward": {
                    "profile_id": "artifact-a-cuda-float32-generic",
                    "default_blocks_per_compute_unit": 4,
                    "calibration_candidates": [2, 4, 8],
                    "calibration_permitted": True,
                    "implementation_kind": "module",
                }
            }
            self.blocks = 4

        def _set_kernel_launch_policy(self, policy):
            assert policy == "automatic"

        def _set_kernel_launch_profile_override(self, profile_id, blocks):
            assert profile_id == "artifact-a-cuda-float32-generic"
            self.blocks = blocks

    class Calculator:
        def __init__(self):
            self.evaluator = Evaluator()
            self.results = {}

        @property
        def kernel_launch_profile_diagnostics(self):
            return self.evaluator.kernel_launch_profile_diagnostics

        def calculate(self, atoms, properties, system_changes):
            self.results = {"energy": -1.0, "forces": np.zeros((len(atoms), 3))}

        @staticmethod
        def _mace_inputs(atoms):
            return len(atoms), [], [], [0, 1], [], [], [], []

        @staticmethod
        def _has_native_field_coupling():
            return False

        @staticmethod
        def _compute_mace(mace_inputs):
            pass

        @staticmethod
        def _collect_mace_results(atoms, mace_inputs):
            return {"energy": -1.0, "forces": np.zeros((len(atoms), 3))}

        @staticmethod
        def _kernel_launch_implementation_identity():
            return "artifact-a"

        @staticmethod
        def _kernel_launch_model_identity():
            return "contract-a"

    calculator = Calculator()
    monkeypatch.setattr(calibration, "Symmetrix", lambda *args, **kwargs: calculator)
    monotonic_values = iter((0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0))
    monkeypatch.setattr(calibration.time, "monotonic", lambda: next(monotonic_values))

    result = calibration.calibrate_kernel_launches(
        "model.json",
        Atoms("Al2", positions=[[0, 0, 0], [1, 1, 1]]),
        properties=("energy", "forces"),
        warmups=0,
        samples=2,
        time_budget_seconds=1.0,
        cache_root=tmp_path,
    )

    assert result.decisions[0]["blocks_per_compute_unit"] == 4
    assert result.measurements["groups"][0]["winner"] == 4
    assert result.measurements["groups"][0]["rejected"] == {
        "2": "time budget exhausted",
        "4": "time budget exhausted",
        "8": "time budget exhausted",
    }
    assert calculator.evaluator.blocks == 4


def test_prepared_timing_excludes_host_result_collection(monkeypatch):
    elapsed = 0.0

    class Calculator:
        results = {}

        @staticmethod
        def _mace_inputs(atoms):
            return len(atoms), [], [], [], [], [], [], []

        @staticmethod
        def _has_native_field_coupling():
            return False

        @staticmethod
        def _compute_mace(mace_inputs):
            nonlocal elapsed
            elapsed += 0.010

        @staticmethod
        def _collect_mace_results(atoms, mace_inputs):
            nonlocal elapsed
            elapsed += 1.0
            return {"energy": -1.0}

    monkeypatch.setattr(calibration.time, "perf_counter", lambda: elapsed)
    evaluate, _ = calibration._prepared_evaluator(
        Calculator(), Atoms("Al"), ("energy",)
    )

    elapsed_ms, output = evaluate()

    assert elapsed_ms == pytest.approx(10.0)
    assert output == {"energy": -1.0}


@pytest.mark.skipif(os.name != "posix", reason="requires atomic POSIX hard links")
def test_fresh_processes_publish_one_complete_record(tmp_path):
    source_root = Path(__file__).resolve().parents[1] / "source"
    gate = tmp_path / "gate"
    worker = r"""
import importlib.util
import json
from pathlib import Path
import sys
import time
module_path = Path(sys.argv[1]) / "symmetrix" / "kernel_launch_tuning.py"
spec = importlib.util.spec_from_file_location("kernel_launch_tuning_worker", module_path)
tuning = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tuning
spec.loader.exec_module(tuning)
root = Path(sys.argv[2])
gate = Path(sys.argv[3])
while not gate.exists():
    time.sleep(0.002)
environment = {
    "backend": "cuda", "architecture": "sm_120",
    "target_features": "", "compute_unit_count": 170,
    "native_subgroup_width": 32, "max_team_size": 1024,
    "max_shared_memory_per_block": 49152,
    "runtime_version": "13.3", "driver_version": "580.65",
}
identity = tuning.kernel_launch_tuning_identity(
    device_environment=environment, implementation_identity="artifact-a",
    model_identity="contract-a", precision="float32",
    workload=tuning.kernel_launch_workload_identity(
        4000, 364000, ("energy", "forces")),
)
decision = [{"stage": "r1_forward", "profile_id": "profile-a",
             "kind": "module", "blocks_per_compute_unit": 4}]
result = tuning.publish_kernel_launch_tuning_record(
    identity, decision, {"accepted": True}, cache_root=root)
print(json.dumps({"published": result.published,
                  "publication_id": result.record["publication_id"]}))
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", worker, str(source_root), str(tmp_path), str(gate)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(8)
    ]
    gate.touch()
    outputs = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stderr
        outputs.append(json.loads(stdout))

    assert sum(result["published"] for result in outputs) == 1
    assert len({result["publication_id"] for result in outputs}) == 1
    assert list((tmp_path / ".staging").iterdir()) == []
