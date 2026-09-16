import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest
from ase.calculators.calculator import Calculator, all_changes


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/ase_omat_md_backend_scale.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location(
        "ase_omat_md_backend_scale_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def benchmark():
    return _load_benchmark()


class ZeroForceCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": 0.0,
            "forces": np.zeros((len(atoms), 3), dtype=float),
            "stress": np.array([0.02, 0.02, 0.02, 0.0, 0.0, 0.0]),
        }


def test_initial_state_is_deterministic_and_stationary(benchmark):
    first = benchmark.build_initial_state(2, seed=41)
    second = benchmark.build_initial_state(2, seed=41)
    different = benchmark.build_initial_state(2, seed=42)

    assert len(first["numbers"]) == 32
    assert first["state_sha256"] == second["state_sha256"]
    assert first["state_sha256"] != different["state_sha256"]
    assert np.linalg.norm(np.sum(first["momenta_eV_fs_per_A"], axis=0)) < 1.0e-12
    assert first["actual_temperature_K"] > 0.0


def test_state_round_trip_and_semantic_hash_validation(benchmark, tmp_path):
    state = benchmark.build_initial_state(1, seed=9)
    path = tmp_path / "state.npz"
    benchmark.save_state(path, state)
    restored = benchmark.load_state(path)

    assert restored["state_sha256"] == state["state_sha256"]
    np.testing.assert_array_equal(restored["positions_A"], state["positions_A"])
    restored["positions_A"][0, 0] += 1.0
    assert benchmark.state_sha256(restored) != state["state_sha256"]


def test_artifact_hash_rejection(benchmark, tmp_path):
    artifact = tmp_path / "model"
    artifact.write_bytes(b"wrong model")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        benchmark.verify_artifact(artifact, "0" * 64, "fixture")


def test_compact_model_requires_all_current_execution_contracts(benchmark, tmp_path):
    model = tmp_path / "compact.json"
    model.write_text(
        json.dumps({"model_type": "MACE", "num_channels": 128, "r_cut": 6.0})
    )
    with pytest.raises(RuntimeError, match="regenerate it with the extract subcommand"):
        benchmark.verify_compact_model(model)
    contracts = {
        name: {"generation_fingerprint": f"sha256:{name.lower()}"}
        for name in ("M0", "R0", "R1")
    }
    model.write_text(
        json.dumps(
            {
                "model_type": "MACE",
                "num_channels": 128,
                "r_cut": 6.0,
                "execution_contracts": contracts,
            }
        )
    )
    verified = benchmark.verify_compact_model(model)
    assert set(verified["execution_contract_fingerprints"]) == {"M0", "R0", "R1"}


def test_timing_summary_reports_required_statistics(benchmark):
    summary = benchmark.timing_summary([0.001, 0.002, 0.004, 0.003], 4)
    assert summary["samples_ms"] == [1.0, 2.0, 4.0, 3.0]
    assert summary["median_ms"] == 2.5
    assert summary["median_us_per_atom"] == 625.0
    assert summary["p95_ms"] == 4.0
    assert summary["median_ms_per_atom"] == 0.625
    assert summary["atom_steps_per_second"] == 1600.0


def test_worker_requires_one_evaluation_per_step_after_initialization(benchmark):
    benchmark.validate_evaluation_count(21, 20)
    with pytest.raises(RuntimeError, match="exactly 21"):
        benchmark.validate_evaluation_count(22, 20)


def test_worker_runs_exactly_twenty_npt_steps(benchmark, tmp_path):
    state_path = tmp_path / "state.npz"
    heartbeat = tmp_path / "heartbeat.json"
    benchmark.save_state(state_path, benchmark.build_initial_state(1, seed=17))

    def calculator_factory(backend, **kwargs):
        assert backend == "kokkos_all"
        return ZeroForceCalculator()

    def telemetry(backend, calculator):
        return {"probe": len(getattr(calculator, "results", {}))}

    record = benchmark.run_md_worker(
        backend="kokkos_all",
        state_path=state_path,
        checkpoint=tmp_path / "unused-checkpoint",
        compact_model=tmp_path / "unused-compact",
        calculator_factory=calculator_factory,
        telemetry_fn=telemetry,
        identity_fn=lambda backend, calculator: {"backend": backend, "fake": True},
        heartbeat_path=heartbeat,
    )

    assert record["status"] == "success"
    assert record["md"]["ensemble"] == "NPT"
    assert record["md"]["integrator"] == "ASE NPT (Melchionna)"
    assert record["md"]["external_pressure_GPa"] == 0.0
    assert record["md"]["thermostat_time_fs"] == 25.0
    assert record["md"]["barostat_time_fs"] == 75.0
    assert record["md"]["bulk_modulus_GPa"] == 210.0
    assert record["md"]["steps"] == 20
    assert record["m1_polynomial_request"] == {
        "policy": "retained",
        "recompute_tile_channels": 32,
    }
    assert record["calculator_evaluations"] == 21
    assert len(record["timing"]["samples_ms"]) == 20
    assert len(record["physical_samples"]) == 20
    assert len([item for item in record["telemetry"] if item["phase"] == "step"]) == 20
    assert json.loads(heartbeat.read_text())["last_step"] == 20
    assert record["physics"]["volume_max_A3"] > record["physics"]["volume_min_A3"]
    assert np.isfinite(record["physics"]["gibbs_drift_eV_per_atom"])
    assert record["workload"]["model_cutoff_A"] == 6.0
    assert record["workload"]["neighbor_skin_A"] == 0.5
    assert record["workload"]["effective_neighbor_cutoff_A"] == 6.5


def test_worker_runs_exactly_twenty_nve_steps(benchmark, tmp_path):
    state_path = tmp_path / "state.npz"
    benchmark.save_state(state_path, benchmark.build_initial_state(1, seed=17))

    record = benchmark.run_md_worker(
        backend="factorized",
        state_path=state_path,
        checkpoint=tmp_path / "unused-checkpoint",
        compact_model=tmp_path / "unused-compact",
        ensemble="nve",
        calculator_factory=lambda backend, **kwargs: ZeroForceCalculator(),
        telemetry_fn=lambda backend, calculator: {},
        identity_fn=lambda backend, calculator: {
            "backend": backend,
            "factorized": {
                "jit": {"status": "cached"},
                "evaluator": {
                    "factorized_ready": True,
                    "factorized_jit_ready": True,
                    "standard_r0_module_fallback_reason": "",
                    "standard_m0_module_fallback_reason": "",
                    "factorized_jit_artifact_id": "jit-r1",
                    "standard_r0_module_id": "r0",
                    "standard_m0_module_id": "m0",
                },
            },
        },
    )

    assert record["status"] == "success"
    assert record["md"]["ensemble"] == "NVE"
    assert record["md"]["integrator"] == "VelocityVerlet"
    assert record["calculator_evaluations"] == 21
    assert record["physics"]["drift_within_threshold"]
    assert record["physics"]["gibbs_drift_eV_per_atom"] is None
    assert record["physics"]["final_relative_volume_change"] == 0.0


def test_worker_rejects_non_twenty_step_contract(benchmark, tmp_path):
    state_path = tmp_path / "state.npz"
    benchmark.save_state(state_path, benchmark.build_initial_state(1))
    with pytest.raises(ValueError, match="exactly 20"):
        benchmark.run_md_worker(
            backend="kokkos_all",
            state_path=state_path,
            checkpoint=tmp_path / "checkpoint",
            compact_model=tmp_path / "compact",
            steps=19,
        )


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (RuntimeError("CUDA out of memory"), "cuda_oom"),
        (
            RuntimeError(
                "Kokkos ERROR: Cuda memory space failed to allocate 2.716 GiB"
            ),
            "cuda_oom",
        ),
        (
            RuntimeError(
                "cudaStreamSynchronize error(cudaErrorIllegalAddress): "
                "an illegal memory access was encountered"
            ),
            "cuda_illegal_address",
        ),
        (FloatingPointError("nonfinite force"), "nonfinite_physics"),
        (RuntimeError("Execution artifact missing"), "setup_rejection"),
        (
            ValueError("cannot create std::vector larger than max_size()"),
            "native_graph_size_error",
        ),
        (RuntimeError("other"), "unknown_exception"),
    ],
)
def test_exception_classification(benchmark, message, expected):
    assert benchmark.classify_exception(message) == expected


def test_confirmed_failure_requires_two_consistent_attempts(benchmark):
    one = {"status": "failure", "failure": {"class": "cuda_oom"}}
    other = {"status": "failure", "failure": {"class": "timeout"}}
    assert benchmark.confirmed_failure([one]) is None
    assert benchmark.confirmed_failure([one, other]) is None
    assert benchmark.confirmed_failure([one, other, one]) == "cuda_oom"


def test_attempt_identity_is_backfilled_and_validated(benchmark):
    attempt = benchmark.Attempt("factorized", 18, 2)
    assert benchmark.apply_attempt_identity({"status": "failure"}, attempt) == {
        "status": "failure",
        "backend": "factorized",
        "repeat": 18,
    }
    with pytest.raises(RuntimeError, match="repeat mismatch"):
        benchmark.apply_attempt_identity({"repeat": 16}, attempt)


def test_process_failure_uses_native_stderr(benchmark):
    assert (
        benchmark.classify_process_failure(
            returncode=-6,
            timed_out=False,
            host_guard=False,
            stderr_text="Assertion '__n < this->size()' failed.",
        )
        == "native_bounds_assertion"
    )
    assert (
        benchmark.classify_process_failure(
            returncode=-6,
            timed_out=False,
            host_guard=False,
            stderr_text="CUDA out of memory",
        )
        == "cuda_oom"
    )


def test_missing_stderr_is_empty(benchmark, tmp_path):
    assert benchmark.read_text_if_exists(tmp_path / "not-created.stderr") == ""


def test_successful_repeat_lookup_is_exact(benchmark, tmp_path):
    attempts = tmp_path / "attempts"
    attempts.mkdir()
    benchmark.atomic_json(
        attempts / "factorized-n30-a1.json", {"status": "success", "repeat": 30}
    )
    assert benchmark.has_successful_repeat(tmp_path, "factorized", 30)
    assert not benchmark.has_successful_repeat(tmp_path, "factorized", 10)


@pytest.mark.parametrize(
    ("last_success", "first_failure", "expected"),
    [(10, 12, 11), (10, 13, None), (None, 4, None), (10, None, None)],
)
def test_refinement_repeat(benchmark, last_success, first_failure, expected):
    assert benchmark.refinement_repeat(last_success, first_failure) == expected


@pytest.mark.parametrize(
    ("maximum", "expected"),
    [
        (4, [4]),
        (9, [4, 6, 8]),
        (12, [4, 6, 8, 10, 12]),
        (16, [4, 6, 8, 10, 12, 14, 16]),
    ],
)
def test_campaign_repeat_cap(benchmark, maximum, expected):
    assert benchmark.campaign_repeats(maximum) == expected


def test_direct_sizes_and_backend_selection(benchmark):
    assert benchmark.parse_direct_sizes("20,24,32") == [20, 24, 32]
    assert benchmark.parse_scale_backends("factorized") == ["factorized"]
    with pytest.raises(ValueError, match="strictly increasing"):
        benchmark.parse_direct_sizes("32,24")
    with pytest.raises(ValueError, match="unknown"):
        benchmark.parse_scale_backends("factorized,missing")


def test_largest_success_repeat_uses_prior_attempts(benchmark, tmp_path):
    attempts = tmp_path / "attempts"
    attempts.mkdir()
    for repeat, status in ((12, "success"), (20, "failure"), (18, "success")):
        benchmark.atomic_json(
            attempts / f"factorized-n{repeat}-a1.json",
            {"backend": "factorized", "repeat": repeat, "status": status},
        )
    assert benchmark.largest_success_repeat(tmp_path, "factorized") == 18


def test_execution_admission_rejects_fallback(benchmark):
    identity = {
        "jit": {"status": "built"},
        "evaluator": {
            "factorized_ready": True,
            "factorized_jit_ready": True,
            "standard_r0_module_fallback_reason": "",
            "standard_m0_module_fallback_reason": "generic",
            "factorized_jit_artifact_id": "jit",
            "standard_r0_module_id": "r0",
            "standard_m0_module_id": "m0",
        },
    }
    with pytest.raises(RuntimeError, match="fallback"):
        benchmark.validate_factorized_identity(identity)


def test_execution_admission_accepts_current_factorized_identity(benchmark):
    identity = {
        "jit": {"status": "cached"},
        "evaluator": {
            "factorized_ready": True,
            "factorized_jit_ready": True,
            "standard_r0_module_fallback_reason": "",
            "standard_m0_module_fallback_reason": "",
            "factorized_jit_artifact_id": "jit-r1",
            "standard_r0_module_id": "r0",
            "standard_m0_module_id": "m0",
        },
    }
    benchmark.validate_factorized_identity(identity)


def test_initial_parity_uses_per_atom_and_force_tolerances(benchmark):
    records = {
        backend: {
            "atoms": 256,
            "initial": {
                "energy_eV": 1.0,
                "forces_eV_per_A": np.zeros((256, 3)).tolist(),
            },
        }
        for backend in benchmark.BACKENDS
    }
    records["factorized"]["initial"]["energy_eV"] += 0.01
    parity = benchmark.check_initial_parity(records)
    assert parity["passed"]
    records["factorized"]["initial"]["forces_eV_per_A"][0][0] = 0.001
    assert not benchmark.check_initial_parity(records)["passed"]


def test_reports_preserve_raw_attempts_and_speedups(benchmark, tmp_path):
    attempt_dir = tmp_path / "attempts"
    attempt_dir.mkdir()
    for backend, median in zip(benchmark.BACKENDS, (2.0, 1.0, 4.0)):
        benchmark.atomic_json(
            attempt_dir / f"{backend}.json",
            {
                "status": "success",
                "backend": backend,
                "repeat": 4,
                "atoms": 256,
                "state_sha256": "same",
                "timing": {"median_ms": median, "samples_ms": [median] * 20},
            },
        )
    report = benchmark.assemble_report(
        tmp_path,
        campaign={"id": "fixture"},
        boundaries={},
        parity={"passed": True},
    )
    assert len(report["attempts"]) == 3
    assert report["speed_comparison"][0]["speedup"]["factorized_vs_all"] == 2.0
    assert report["speed_comparison"][0]["speedup"]["factorized_vs_torch_cueq"] == 4.0
    benchmark.publish_reports(tmp_path, report)
    assert (tmp_path / "omat0_medium_ase_md_backend_scale.json").is_file()
    assert (
        "Factorized/all"
        in (tmp_path / "omat0_medium_ase_md_backend_scale.md").read_text()
    )


def test_report_normalizes_historical_cuda_allocation_failure(benchmark, tmp_path):
    attempt_dir = tmp_path / "attempts"
    attempt_dir.mkdir()
    for index in (1, 2):
        benchmark.atomic_json(
            attempt_dir / f"kokkos_all-n20-a{index}.json",
            {
                "status": "failure",
                "backend": "kokkos_all",
                "repeat": 20,
                "failure": {
                    "class": "unknown_exception",
                    "message": (
                        "Kokkos ERROR: Cuda memory space failed to allocate 2.716 GiB"
                    ),
                },
            },
        )
    report = benchmark.assemble_report(
        tmp_path,
        campaign={"id": "fixture"},
        boundaries={
            "kokkos_all": {
                "largest_success_repeat": 19,
                "first_failure_repeat": 20,
                "failure_class": "unknown_exception",
            }
        },
        parity={"passed": True},
    )
    assert {item["failure"]["class"] for item in report["attempts"]} == {"cuda_oom"}
    assert report["boundaries"]["kokkos_all"]["failure_class"] == "cuda_oom"


def test_cli_rejects_wrong_step_count(benchmark, capsys):
    with pytest.raises(SystemExit) as exc:
        benchmark.main(
            [
                "worker",
                "--backend",
                "kokkos_all",
                "--state",
                "state.npz",
                "--output",
                "out.json",
                "--steps",
                "21",
            ]
        )
    assert exc.value.code == 2
    assert "--steps must be exactly 20" in capsys.readouterr().err


def test_current_python_keeps_virtual_environment_symlink(
    benchmark, monkeypatch, tmp_path
):
    base = tmp_path / "base-python"
    base.touch()
    launcher_dir = tmp_path / "venv" / "bin"
    launcher_dir.mkdir(parents=True)
    launcher = launcher_dir / "python"
    launcher.symlink_to(base)
    monkeypatch.setattr(benchmark.sys, "executable", str(launcher))
    assert benchmark.current_python_executable() == launcher


def test_worker_cli_does_not_apply_campaign_repeat_validation(
    benchmark, monkeypatch, tmp_path
):
    monkeypatch.setattr(benchmark, "finalize_runtime", lambda: None)
    model = tmp_path / "wrong-model"
    model.write_bytes(b"fixture")
    output = tmp_path / "worker.json"
    status = benchmark.main(
        [
            "worker",
            "--backend",
            "kokkos_all",
            "--state",
            str(tmp_path / "missing-state.npz"),
            "--checkpoint",
            str(model),
            "--compact-model",
            str(model),
            "--output",
            str(output),
        ]
    )
    assert status == 1
    assert benchmark.load_json(output)["failure"]["class"] == "unknown_exception"


def test_worker_parser_accepts_supervisor_compact_hash(benchmark):
    args = benchmark.make_parser().parse_args(
        [
            "worker",
            "--backend",
            "factorized",
            "--state",
            "state.npz",
            "--compact-model",
            "compact.json",
            "--compact-sha256",
            "a" * 64,
            "--m1-polynomial-policy",
            "recompute",
            "--m1-recompute-tile-channels",
            "32",
            "--output",
            "result.json",
        ]
    )
    assert args.compact_sha256 == "a" * 64
    assert args.m1_polynomial_policy == "recompute"
    assert args.m1_recompute_tile_channels == 32


def test_supervisor_forwards_campaign_timestep_to_worker(benchmark, tmp_path):
    command = benchmark._worker_command(
        python=tmp_path / "python",
        script=tmp_path / "benchmark.py",
        attempt=benchmark.Attempt("factorized", 7, 2),
        state_path=tmp_path / "state.npz",
        checkpoint=tmp_path / "checkpoint.model",
        compact_model=tmp_path / "compact.json",
        compact_sha256="a" * 64,
        output_path=tmp_path / "attempt.json",
        heartbeat_path=tmp_path / "heartbeat.json",
        timestep_fs=0.5,
        ensemble="nve",
        m1_polynomial_policy="recompute",
        m1_recompute_tile_channels=32,
    )

    timestep_index = command.index("--timestep-fs")
    assert command[timestep_index + 1] == "0.5"
    ensemble_index = command.index("--ensemble")
    assert command[ensemble_index + 1] == "nve"
    policy_index = command.index("--m1-polynomial-policy")
    assert command[policy_index + 1] == "recompute"
    tile_index = command.index("--m1-recompute-tile-channels")
    assert command[tile_index + 1] == "32"


def test_campaign_parser_accepts_factorized_only_nve(benchmark):
    args = benchmark.make_parser().parse_args(
        [
            "campaign",
            "--output-dir",
            "results",
            "--ensemble",
            "nve",
            "--backends",
            "factorized",
            "--scale-backends",
            "factorized",
        ]
    )
    assert args.ensemble == "nve"
    assert benchmark.parse_scale_backends(args.backends) == ["factorized"]


def test_single_backend_report_preserves_timings(benchmark, tmp_path):
    attempt_dir = tmp_path / "attempts"
    attempt_dir.mkdir()
    benchmark.atomic_json(
        attempt_dir / "factorized-n4-a1.json",
        {
            "status": "success",
            "backend": "factorized",
            "repeat": 4,
            "atoms": 256,
            "state_sha256": "same",
            "timing": {
                "median_ms": 2.0,
                "median_us_per_atom": 7.8125,
                "samples_ms": [2.0] * 20,
            },
        },
    )
    report = benchmark.assemble_report(
        tmp_path,
        campaign={"ensemble": "NVE", "backends": ["factorized"]},
        boundaries={},
        parity=None,
    )
    assert report["speed_comparison"] == []
    assert report["backend_timings"][0]["median_us_per_atom"] == 7.8125
    benchmark.publish_reports(tmp_path, report)
    markdown = (tmp_path / "omat0_medium_ase_md_backend_scale.md").read_text()
    assert "Velocity-Verlet NVE" in markdown
    assert "factorized | 4 | 256 | 7.812" in markdown
