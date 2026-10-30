import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from compact_r1_model import add_standard_base_contracts, foundation_r1_test_model
from symmetrix import (
    Symmetrix,
    available_backends,
    load_backend,
    selected_backend,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = REPOSITORY_ROOT / "benchmarks" / "mace_les_timing.py"
DOCUMENTATION_PATH = REPOSITORY_ROOT / "docs" / "mace_les.md"
CONTRACT_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "execution_contracts"
    / "jit_r1_fixture_c4_e3_l0_contract.json"
)


def _les_payload():
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
        "output_scale": 0.25,
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


@pytest.fixture()
def mace_les_json(tmp_path):
    contract = json.loads(CONTRACT_PATH.read_text())
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
    return path


def _active_backend_or_skip(requested):
    installed = {descriptor["backend"] for descriptor in available_backends()}
    if requested not in installed:
        pytest.skip(f"requires an installed Symmetrix {requested} backend")
    load_backend()
    selected = selected_backend()
    active = None if selected is None else selected["backend"]
    if active != requested:
        pytest.skip(
            f"requires the {requested} backend selected in this process; active "
            f"backend is {active or 'not loaded'}"
        )


@pytest.mark.parametrize("backend", ["cpu", "cuda", "hip"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("periodic", [False, True], ids=["nonperiodic", "periodic"])
def test_maceles_backend_contract(mace_les_json, backend, precision, periodic):
    _active_backend_or_skip(backend)
    atoms = Atoms(
        ["N", "Al", "N"],
        positions=[[0.1, 0.2, 0.3], [1.4, 0.4, 0.2], [0.5, 1.6, 0.7]],
        cell=[[6.2, 0.2, 0.1], [0.0, 6.0, 0.3], [0.1, 0.0, 5.8]],
        pbc=periodic,
    )
    calculator = Symmetrix(
        mace_les_json,
        dtype=precision,
        streamed_edges="reference",
        neighbor_skin=0.0,
    )
    calculator.calculate(
        atoms,
        properties=["energy", "energies", "forces", "stress"],
    )

    assert np.isfinite(calculator.results["energy"])
    assert np.all(np.isfinite(calculator.results["energies"]))
    assert np.all(np.isfinite(calculator.results["forces"]))
    assert np.all(np.isfinite(calculator.results["stress"]))
    assert np.all(np.isfinite(calculator.results["latent_charges"]))
    assert np.asarray(calculator.results["energies"]).sum() == pytest.approx(
        calculator.results["energy"], rel=2.0e-6, abs=2.0e-7
    )
    assert calculator.results["mace_les_energy"] == pytest.approx(
        calculator.results["energy"]
    )


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("mace_les_timing", BENCHMARK_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load benchmark module from {BENCHMARK_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mace_les_profile_region_brackets_only_synchronized_samples():
    benchmark = _load_benchmark_module()
    events = []

    class FakeROCtx:
        def roctxProfilerPause(self, _):
            events.append("pause")
            return 0

        def roctxProfilerResume(self, _):
            events.append("resume")
            return 0

        def roctxRangePushA(self, name):
            events.append(("push", name))
            return 0

        def roctxRangePop(self):
            events.append("pop")
            return 0

    class FakeHIP:
        def hipDeviceSynchronize(self):
            events.append("synchronize")
            return 0

    roctx = FakeROCtx()
    benchmark._roctx_state(roctx, roctx.roctxProfilerPause, "roctxProfilerPause")
    events.append("construction-and-warmup")
    samples = benchmark._time_samples(
        lambda: events.append("evaluate"), 2, hip=FakeHIP(), roctx=roctx
    )

    assert len(samples) == 2
    assert events == [
        "pause",
        "construction-and-warmup",
        "resume",
        ("push", b"symmetrix-mace-les-samples"),
        "synchronize",
        "evaluate",
        "synchronize",
        "synchronize",
        "evaluate",
        "synchronize",
        "pop",
        "pause",
    ]


def test_cuda_profile_region_brackets_and_cleans_up_after_failure():
    benchmark = _load_benchmark_module()
    events = []

    class FakeCuda:
        def cudaDeviceSynchronize(self):
            events.append("synchronize")
            return 0

        def cudaProfilerStart(self):
            events.append("start")
            return 0

        def cudaProfilerStop(self):
            events.append("stop")
            return 0

    class FakeNVTX:
        def nvtxRangePushA(self, name):
            events.append(("push", name))
            return 0

        def nvtxRangePop(self):
            events.append("pop")
            return 0

    def fail():
        events.append("evaluate")
        raise RuntimeError("evaluation failed")

    with pytest.raises(RuntimeError, match="evaluation failed"):
        benchmark._time_samples(
            fail, 1, runtime=FakeCuda(), runtime_kind="cuda", nvtx=FakeNVTX()
        )

    assert events == [
        "start",
        ("push", b"symmetrix-mace-les-samples"),
        "synchronize",
        "evaluate",
        "pop",
        "stop",
    ]


def test_hip_profile_region_cleans_up_after_failure():
    benchmark = _load_benchmark_module()
    events = []

    class FakeROCtx:
        def roctxProfilerPause(self, _):
            events.append("pause")
            return 0

        def roctxProfilerResume(self, _):
            events.append("resume")
            return 0

        def roctxRangePushA(self, name):
            events.append(("push", name))
            return 0

        def roctxRangePop(self):
            events.append("pop")
            return 0

    class FakeHIP:
        def hipDeviceSynchronize(self):
            events.append("synchronize")
            return 0

    def fail():
        events.append("evaluate")
        raise RuntimeError("evaluation failed")

    roctx = FakeROCtx()
    with pytest.raises(RuntimeError, match="evaluation failed"):
        benchmark._time_samples(
            fail, 1, runtime=FakeHIP(), runtime_kind="hip", roctx=roctx
        )
    assert events == [
        "resume",
        ("push", b"symmetrix-mace-les-samples"),
        "synchronize",
        "evaluate",
        "pop",
        "pause",
    ]


def test_structure_replication_and_cell_scale_are_explicit():
    benchmark = _load_benchmark_module()
    atoms = benchmark._structure(
        periodic=True,
        replicate=(2, 3, 1),
        cell_scale=1.25,
        path=None,
    )
    assert len(atoms) == 18
    assert np.linalg.det(atoms.cell.array) > 0
    base = benchmark._structure(
        periodic=True, replicate=(2, 3, 1), cell_scale=1.0, path=None
    )
    np.testing.assert_allclose(atoms.cell.array, 1.25 * base.cell.array)
    np.testing.assert_allclose(
        atoms.get_scaled_positions(), base.get_scaled_positions()
    )


def test_native_graph_generation_diagnostics_reject_timed_rebuild():
    benchmark = _load_benchmark_module()
    with pytest.raises(RuntimeError, match="native prepared graph"):
        benchmark._graph_reuse_diagnostics(0, 0, 0)
    diagnostics = benchmark._graph_reuse_diagnostics(0, 1, 1)
    assert diagnostics == {
        "native_graph_generation_before_warmup": 0,
        "native_graph_generation_before_timing": 1,
        "native_graph_generation_after_timing": 1,
        "native_graph_rebuilds_during_warmup": 1,
        "native_graph_rebuilds_during_timing": 0,
        "prepared_graph_count": 1,
    }
    with pytest.raises(RuntimeError, match="native prepared graph"):
        benchmark._graph_reuse_diagnostics(0, 1, 2)


def test_mace_les_benchmark_record_is_self_describing(mace_les_json, tmp_path):
    benchmark = _load_benchmark_module()
    environment = os.environ.copy()
    environment["SYMMETRIX_BENCHMARK_THREADS"] = "1"
    environment["SYMMETRIX_BENCHMARK_BLAS_THREADS"] = "1"
    environment["SYMMETRIX_JIT_CACHE"] = str(tmp_path / "jit-cache")
    environment["SYMMETRIX_JIT_POLICY"] = "required"
    completed = subprocess.run(
        [
            sys.executable,
            str(BENCHMARK_PATH),
            str(mace_les_json),
            "--precision",
            "float64",
            "--pbc",
            "nonperiodic",
            "--skin",
            "0.0",
            "--warmups",
            "1",
            "--samples",
            "1",
            "--streamed-edges",
            "direct",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    record = benchmark.parse_record(completed.stdout)

    benchmark.validate_requested_workload(
        record,
        num_atoms=3,
        pbc="nonperiodic",
        precision="float64",
    )
    assert record["schema"] == "symmetrix.mace-les-timing/4"
    assert record["backend"] == selected_backend()["backend"]
    assert record["execution_mode"] == "prepared-direct"
    assert record["streamed_edges"] == "direct"
    assert record["model_type"] == "MACELES"
    assert len(record["mace_les_parameters_sha256"]) == 64
    assert len(record["mace_les_evaluator_fingerprint"]) == 64
    assert record["mace_les_energy"] == pytest.approx(record["energy"])
    assert np.isfinite(record["les_energy"])
    assert record["jit_status"] in {"built", "cached"}
    assert Path(record["jit_artifact_path"]).is_file()
    assert record["prepared_evaluation_count"] == 2
    assert record["fallback_evaluation_count"] == 0
    assert record["graph_generation"] > 0
    assert record["device_synchronization_bracketed_samples"] is (
        record["kokkos_execution_space"] in {"HIP", "Cuda"}
    )
    assert record["samples_ns"] and len(record["samples_ns"]) == 1
    assert record["replication"] == [1, 1, 1]
    assert record["cell_scale"] == pytest.approx(1.0)
    assert record["synchronization"]["backend"] in {"none", "hip", "cuda"}
    assert record["profile_range"]["enabled"] is False
    assert record["prepared_graph_count"] == 1
    assert record["native_graph_generation_before_timing"] > 0
    assert (
        record["native_graph_generation_after_timing"]
        == record["native_graph_generation_before_timing"]
    )
    assert record["native_graph_rebuilds_during_timing"] == 0
    assert Path(record["interpreter_path"]).resolve() == Path(sys.executable).resolve()
    assert Path(record["symmetrix_package_path"]).is_file()
    assert record["python_version"]
    assert record["symmetrix_version"]
    assert record["device_provenance"]["backend"]
    assert record["source_provenance"]["source_content_sha256"]
    assert record["structure_sha256"] is None
    for name in (
        "reciprocal_candidate_count",
        "reciprocal_vector_count",
        "atom_vector_pairs",
        "reciprocal_workspace_bytes",
    ):
        assert record[name] is None
    assert record["num_atoms"] == 3
    assert record["directed_edges"] == 6
    assert record["pbc"] == "nonperiodic"
    assert record["precision"] == "float64"
    assert record["cutoff"] == pytest.approx(
        json.loads(mace_les_json.read_text())["r_cut"]
    )
    assert record["skin"] == pytest.approx(0.0)
    assert record["sigma"] == pytest.approx(1.0)
    assert record["dl"] == pytest.approx(2.0)
    assert record["us_per_atom"] > 0.0
    assert len(record["model_sha256"]) == 64
    assert record["mace_architecture"]["mace_num_interactions"] == 2
    assert len(record["native_extension_sha256"]) == 64
    assert record["samples"] == 1
    assert record["warmups"] == 1
    assert record["host_graph_constructions_before_timing"] == 1
    assert record["host_graph_constructions_during_timing"] == 0
    assert "native model forward/reverse" in record["timed_operations"]
    assert "host neighbor-list construction" in record["excluded_operations"]


def test_mace_les_benchmark_reference_mode_emits_valid_record(mace_les_json, tmp_path):
    benchmark = _load_benchmark_module()
    environment = os.environ.copy()
    environment["SYMMETRIX_BENCHMARK_THREADS"] = "1"
    environment["SYMMETRIX_BENCHMARK_BLAS_THREADS"] = "1"
    environment["SYMMETRIX_JIT_CACHE"] = str(tmp_path / "jit-cache")
    environment["SYMMETRIX_JIT_POLICY"] = "required"
    completed = subprocess.run(
        [
            sys.executable,
            str(BENCHMARK_PATH),
            str(mace_les_json),
            "--precision",
            "float64",
            "--pbc",
            "nonperiodic",
            "--skin",
            "0.0",
            "--warmups",
            "1",
            "--samples",
            "1",
            "--streamed-edges",
            "reference",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    record = benchmark.parse_record(completed.stdout)
    assert record["streamed_edges"] == "reference"
    assert record["execution_mode"] == "generic-full-graph"
    assert record["prepared_evaluation_count"] == 0
    assert record["fallback_evaluation_count"] == 0
    assert record["prepared_graph_count"] == 0
    assert record["graph_generation"] == 0
    assert record["native_graph_generation_before_warmup"] == 0
    assert record["native_graph_generation_before_timing"] == 0
    assert record["native_graph_generation_after_timing"] == 0
    assert record["native_graph_rebuilds_during_warmup"] == 0
    assert record["native_graph_rebuilds_during_timing"] == 0
    assert record["host_graph_constructions_during_timing"] == 0
    assert record["samples"] == 1
    assert len(record["samples_ns"]) == 1


def test_version_probe_reports_matching_selected_backend_provenance():
    command = Path(sys.executable).with_name("symmetrix")
    completed = subprocess.run(
        [str(command), "version", "--probe", "--json"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    report = json.loads(completed.stdout)
    assert report["probe_status"] == "ok"

    selected = report["probe"]["backend"]
    build = report["probe"]["build_info"]
    for name in (
        "source_commit",
        "source_content_sha256",
        "native_source_content_sha256",
    ):
        assert selected[name]
        assert selected[name] == build[name]
    assert selected["source_dirty"] is not None
    assert selected["source_dirty"] == build["source_dirty"]
    descriptor_path = Path(selected["descriptor_path"])
    assert descriptor_path.is_file()
    expected_name = (
        "_backend_cpu.json" if selected["backend"] == "cpu" else "backend.json"
    )
    assert descriptor_path.name == expected_name


def test_mace_les_documented_release_contract():
    documentation = DOCUMENTATION_PATH.read_text()
    normalized_documentation = " ".join(documentation.split())
    index = (REPOSITORY_ROOT / "docs" / "index.md").read_text()

    assert "part of the fitted MACELES model" in normalized_documentation
    assert "Symmetrix` always evaluates the complete model" in documentation
    assert 'model_type: "MACELES"' in documentation
    assert "symmetrix.model.mace_les" in documentation
    assert 'streamed_edges="direct"' in documentation
    assert 'streamed_edges="reference"' in documentation
    assert "generic full-graph" in documentation
    assert "Direct execution uses the ordinary required-JIT policy" in documentation
    assert "MACE backbone's R0, M0, and R1 artifacts" in documentation
    assert "fixed-workspace/tiled plans" in documentation
    assert "distributed evaluation" in documentation
    assert "Metal execution" in documentation
    assert "les_energy" in documentation
    assert "mace_les_energy" in documentation
    assert "latent_charges" in documentation
    assert (
        "`calculator.les_diagnostics` identifies the fitted LES stage and includes "
        "its schema metadata plus a `workspace` mapping from "
        "`evaluator.les_charge_workspace_diagnostics`." in normalized_documentation
    )
    assert (
        "The native evaluator's `les_monopole_diagnostics` property separately "
        "reports reciprocal-vector counts and monopole workspace state."
        in normalized_documentation
    )
    assert (
        "schema, periodic algorithm, reciprocal-vector counts, and workspace state"
        not in normalized_documentation
    )
    assert "correction_diagnostics` is reserved for D2 and D3" in documentation
    assert "explicit_sum" in documentation
    assert "particle_mesh" in documentation
    assert "MACE-Torch" in documentation
    assert "host_graph_constructions_during_timing" in documentation
    assert "mace_les" in index
