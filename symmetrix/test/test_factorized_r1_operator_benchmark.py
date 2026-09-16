import argparse
import copy
import importlib.util
import json
import pathlib
import shutil
import sys
import types

import numpy as np
import pytest

_REPOSITORY = pathlib.Path(__file__).resolve().parents[2]
_FIXTURE_MODULE_NAME = "_symmetrix_factorized_r1_operator_fixture_test"
_fixture_spec = importlib.util.spec_from_file_location(
    _FIXTURE_MODULE_NAME,
    _REPOSITORY
    / "symmetrix"
    / "source"
    / "symmetrix"
    / "factorized_r1_operator_fixture.py",
)
_fixture_module = importlib.util.module_from_spec(_fixture_spec)
sys.modules[_FIXTURE_MODULE_NAME] = _fixture_module
_fixture_spec.loader.exec_module(_fixture_module)

ATOL = _fixture_module.ATOL
LANE = _fixture_module.LANE
RTOL = _fixture_module.RTOL
SCHEMA = _fixture_module.SCHEMA
SCHEMA_VERSION = _fixture_module.SCHEMA_VERSION
FixtureValidationError = _fixture_module.FixtureValidationError
canonical_array_layouts = _fixture_module.canonical_array_layouts
canonical_contract = _fixture_module.canonical_contract
canonical_conventions = _fixture_module.canonical_conventions
load_fixture = _fixture_module.load_fixture
manifest_sha256 = _fixture_module.manifest_sha256
validate_arrays = _fixture_module.validate_arrays
write_fixture = _fixture_module.write_fixture

_MEASUREMENT = {
    "warmup_iterations": 10,
    "warmup_ms": 100.0,
    "min_samples": 20,
    "max_samples": 100,
    "min_sample_ms": 1000.0,
}
_TIMING_SEMANTICS = {
    "forward": "prepared_graph_forward_only",
    "reverse": "prepared_forward_state_reverse_only",
    "forward_backward": (
        "fresh_forward_plus_fixed_weight_backward_plus_directed_force"
    ),
}
_TIMING_PROTOCOL = {
    "device": "cuda",
    "sample_timer": "cuda_events",
    "stream_scope": "backend_active_cuda_stream",
    "cold_completion": "synchronize_after_cold_iteration",
    "warmup_completion": "single_synchronize_after_warmup_batch",
    "sample_completion": "synchronize_end_event",
    "stopping_rule": ("min_samples_and_min_sample_ms_capped_by_max_samples"),
    "preparation": "outside_timed_samples",
    "validation": "outside_timed_samples",
    "raw_samples_retained": True,
}


def _load_benchmark_module():
    path = _REPOSITORY / "benchmarks" / "factorized_operator_benchmark.py"
    spec = importlib.util.spec_from_file_location("factorized_operator_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    created_parent = "symmetrix" not in sys.modules
    if created_parent:
        parent = types.ModuleType("symmetrix")
        parent.__path__ = []
        sys.modules["symmetrix"] = parent
    sys.modules["symmetrix.factorized_r1_operator_fixture"] = _fixture_module
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if created_parent:
            del sys.modules["symmetrix"]
    return module


def _load_upstream_runner_module():
    path = _REPOSITORY / "benchmarks" / "execution_upstream_r1_runner.py"
    spec = importlib.util.spec_from_file_location("execution_upstream_r1_runner", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def benchmark_module():
    return _load_benchmark_module()


@pytest.fixture(scope="module")
def upstream_runner_module():
    return _load_upstream_runner_module()


@pytest.fixture(scope="module")
def synthetic_arrays():
    num_nodes = 2
    num_edges = 2
    return {
        "x": np.zeros((num_nodes, 4, 128), dtype=np.float32),
        "edge_index": np.array([[1, 0], [0, 1]], dtype=np.int64),
        "sh": np.zeros((num_edges, 16), dtype=np.float32),
        "phi": np.zeros((num_edges, 64), dtype=np.float32),
        "radial_linear": np.zeros((64, 10 * 128 * 128), dtype=np.float32),
        "grad_out": np.zeros((num_nodes, 16, 128), dtype=np.float32),
        "xyz": np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64),
        "r": np.ones((num_edges,), dtype=np.float64),
        "dsh_dxyz": np.zeros((num_edges, 3, 16), dtype=np.float32),
        "dphi_dr": np.zeros((num_edges, 64), dtype=np.float32),
        "expected_out": np.zeros((num_nodes, 16, 128), dtype=np.float32),
        "expected_grad_x": np.zeros((num_nodes, 4, 128), dtype=np.float32),
        "expected_directed_force": np.zeros((num_edges, 3), dtype=np.float64),
    }


@pytest.fixture(scope="module")
def written_fixture(tmp_path_factory, synthetic_arrays):
    path = tmp_path_factory.mktemp("execution-r1-fixture") / "synthetic.json"
    provenance = {
        "producer": "unit-test",
        "symmetrix_commit": "test-commit",
        "model_sha256": "sha256:" + "1" * 64,
        "graph_sha256": "sha256:" + "2" * 64,
        "graph_generation": 1,
    }
    return write_fixture(path, synthetic_arrays, provenance)


def test_exact_uvw_contract_and_path_normalization_are_frozen():
    contract = canonical_contract()
    instructions = contract["instructions"]

    assert SCHEMA == "symmetrix.execution.r1-operator-benchmark"
    assert SCHEMA_VERSION == 1
    assert LANE == "extracted_mace_exact_uvw"
    assert contract["input_irreps"] == "128x0e+128x1o"
    assert contract["sh_irreps"] == "1x0e+1x1o+1x2e+1x3o"
    assert contract["output_irreps"] == "128x0e+128x1o+128x2e+128x3o"
    assert contract["connection_mode"] == "uvw"
    assert [(item["i_in"], item["i_sh"], item["i_out"]) for item in instructions] == [
        (0, 0, 0),
        (1, 1, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 2, 1),
        (0, 2, 2),
        (1, 1, 2),
        (1, 3, 2),
        (0, 3, 3),
        (1, 2, 3),
    ]
    assert [item["raw_path_weight"] for item in instructions] == [
        256.0,
        256.0,
        384.0,
        384.0,
        384.0,
        384.0,
        384.0,
        384.0,
        256.0,
        256.0,
    ]
    assert contract["radial_linear_block_size"] == 128 * 128
    assert contract["parameter_gradients"] is False
    assert contract["density_scaling"] is False


def test_array_schema_preserves_native_precision_orientation_and_sign(synthetic_arrays):
    assert validate_arrays(synthetic_arrays, require_expected=True) == (2, 2)
    conventions = canonical_conventions()
    layouts = canonical_array_layouts()

    assert synthetic_arrays["xyz"].dtype == np.float64
    assert synthetic_arrays["r"].dtype == np.float64
    assert synthetic_arrays["dsh_dxyz"].shape == (2, 3, 16)
    assert synthetic_arrays["expected_directed_force"].dtype == np.float64
    assert conventions["xyz"] == "source-position-minus-receiver-position"
    assert conventions["directed_force"] == "expected_directed_force=-dE/dxyz"
    assert layouts["dsh_dxyz"] == "edge,coordinate,sh_lm"
    assert layouts["radial_linear"] == ("q,(path,input_channel,output_channel)-C-flat")


@pytest.mark.parametrize(
    ("name", "replacement", "message"),
    [
        ("xyz", np.zeros((2, 3), dtype=np.float32), "dtype"),
        ("r", np.zeros((2, 1), dtype=np.float64), "shape"),
        ("dsh_dxyz", np.zeros((2, 16, 3), dtype=np.float32), "shape"),
        (
            "expected_directed_force",
            np.zeros((2, 3), dtype=np.float32),
            "dtype",
        ),
    ],
)
def test_array_schema_rejects_precision_or_layout_drift(
    synthetic_arrays, name, replacement, message
):
    arrays = dict(synthetic_arrays)
    arrays[name] = replacement
    with pytest.raises(FixtureValidationError, match=message):
        validate_arrays(arrays)


def test_array_schema_rejects_non_receiver_major_graph(synthetic_arrays):
    arrays = dict(synthetic_arrays)
    arrays["edge_index"] = np.array([[0, 1], [1, 0]], dtype=np.int64)
    with pytest.raises(FixtureValidationError, match="receiver-major"):
        validate_arrays(arrays)


def test_array_schema_rejects_radius_drift(synthetic_arrays):
    arrays = dict(synthetic_arrays)
    arrays["r"] = np.full((2,), 2.0, dtype=np.float64)
    with pytest.raises(FixtureValidationError, match="norm of xyz"):
        validate_arrays(arrays)


def test_fixture_round_trip_hashes_manifest_payload_and_arrays(written_fixture):
    loaded = load_fixture(written_fixture.manifest_path, require_expected=True)

    assert loaded.manifest["manifest_sha256"] == manifest_sha256(loaded.manifest)
    assert loaded.manifest["payload"]["format"] == "npz"
    assert loaded.manifest["payload"]["path"] == "synthetic.npz"
    assert set(loaded.manifest["arrays"]) == set(loaded.arrays)
    assert loaded.manifest["dimensions"] == {
        "nodes": 2,
        "edges": 2,
        "channels": 128,
        "source_components": 4,
        "output_components": 16,
        "radial_embedding": 64,
        "paths": 10,
    }
    for entry in loaded.manifest["arrays"].values():
        assert entry["order"] == "C"
        assert entry["sha256"].startswith("sha256:")
        assert len(entry["sha256"]) == len("sha256:") + 64


def test_upstream_runner_independently_accepts_written_fixture(
    upstream_runner_module, written_fixture
):
    loaded = upstream_runner_module.load_fixture(written_fixture.manifest_path)

    assert upstream_runner_module.FIXTURE_SCHEMA == SCHEMA
    assert upstream_runner_module.FIXTURE_VERSION == SCHEMA_VERSION
    assert upstream_runner_module.FIXTURE_LANE == LANE
    assert upstream_runner_module.EXPECTED_CONTRACT == canonical_contract()
    assert upstream_runner_module.EXPECTED_CONVENTIONS == canonical_conventions()
    assert loaded.node_count == 2
    assert loaded.edge_count == 2


@pytest.mark.parametrize(
    ("actual_shape", "actual_dtype", "message"),
    [
        ((2, 16 * 128 - 1), "torch.float32", "shape drifted"),
        ((2, 16 * 128), "torch.float64", "dtype drifted"),
    ],
)
def test_upstream_comparison_rejects_actual_layout_before_reference_coercion(
    upstream_runner_module,
    actual_shape,
    actual_dtype,
    message,
):
    class DriftedActual:
        shape = actual_shape
        dtype = actual_dtype

        def new_tensor(self, _value):
            raise AssertionError("reference coercion must not run after layout drift")

    expected = np.zeros((2, 16, 128), dtype=np.float32)
    with pytest.raises(upstream_runner_module.FixtureError, match=message):
        upstream_runner_module._comparison(
            DriftedActual(),
            expected,
            native_shape=(2, 16 * 128),
            rtol=RTOL,
            atol=ATOL,
        )


def test_upstream_comparison_accepts_declared_flattened_native_layout(
    upstream_runner_module,
):
    torch = pytest.importorskip("torch")
    expected = np.arange(2 * 16 * 128, dtype=np.float32).reshape(2, 16, 128)
    actual = torch.from_numpy(expected.reshape(2, 16 * 128).copy())

    comparison = upstream_runner_module._comparison(
        actual,
        expected,
        native_shape=(2, 16 * 128),
        rtol=RTOL,
        atol=ATOL,
    )

    assert comparison["allclose"]
    assert comparison["shape"] == [2, 16, 128]


def test_fixture_rejects_rehashed_semantic_drift(written_fixture, tmp_path):
    manifest = copy.deepcopy(written_fixture.manifest)
    manifest["contract"]["instructions"][0]["raw_path_weight"] = 1.0
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    path = tmp_path / "drift.json"
    path.write_text(json.dumps(manifest), encoding="ascii")

    with pytest.raises(FixtureValidationError, match="exact UVW lane"):
        load_fixture(path)


def test_fixture_rejects_payload_byte_corruption(written_fixture, tmp_path):
    payload = tmp_path / "corrupt.npz"
    shutil.copyfile(written_fixture.payload_path, payload)
    with payload.open("r+b") as handle:
        handle.seek(-1, 2)
        last = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([last[0] ^ 1]))
    manifest = copy.deepcopy(written_fixture.manifest)
    manifest["payload"]["path"] = payload.name
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    path = tmp_path / "corrupt.json"
    path.write_text(json.dumps(manifest), encoding="ascii")

    with pytest.raises(FixtureValidationError, match="payload SHA-256"):
        load_fixture(path)


def _passing_report(fixture, backend, mode="forward_backward"):
    names = {
        "forward": ("A1",),
        "reverse": ("H1_adj", "edge_force"),
        "forward_backward": ("A1", "H1_adj", "edge_force"),
    }[mode]
    references = {
        "A1": "expected_out",
        "H1_adj": "expected_grad_x",
        "edge_force": "expected_directed_force",
    }
    outputs = {
        name: {
            "reference_array": references[name],
            "shape": list(fixture.arrays[references[name]].shape),
            "dtype": fixture.arrays[references[name]].dtype.str,
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
            "max_scaled_error": 0.0,
            "max_error_index": None,
            "allclose": True,
        }
        for name in names
    }
    timing = {
        "samples_ms": [1.0, 1.1],
        "timing_backend": "cuda_events",
        "execution_stream": (
            "factorized_execution_space.cuda_stream"
            if backend == "symmetrix"
            else "torch_current_cuda_stream"
        ),
    }
    report = {
        "schema": SCHEMA,
        "version": SCHEMA_VERSION,
        "lane": LANE,
        "status": "pass",
        "backend": backend,
        "mode": mode,
        "headline": mode == "forward_backward",
        "timing_semantics": _TIMING_SEMANTICS[mode],
        "timing_protocol": copy.deepcopy(_TIMING_PROTOCOL),
        "measurement": copy.deepcopy(_MEASUREMENT),
        "fixture": {
            "manifest_sha256": fixture.manifest["manifest_sha256"],
            "payload_sha256": fixture.manifest["payload"]["sha256"],
            "model_sha256": fixture.manifest["provenance"]["model_sha256"],
            "graph_sha256": fixture.manifest["provenance"]["graph_sha256"],
            "symmetrix_commit": fixture.manifest["provenance"]["symmetrix_commit"],
            "graph_generation": fixture.manifest["provenance"]["graph_generation"],
            "nodes": fixture.num_nodes,
            "edges": fixture.num_edges,
        },
        "contract": canonical_contract(),
        "conventions": canonical_conventions(),
        "validation": {
            "status": "pass",
            "rtol": RTOL,
            "atol": ATOL,
            "outputs": outputs,
        },
        "timing": timing,
    }
    if backend == "symmetrix":
        timing["cuda"] = {
            "name": "NVIDIA Test GPU",
            "compute_capability": "12.0",
            "runtime_version": 13030,
            "driver_version": 13030,
        }
        report["environment"] = {
            "producer": "benchmarks/factorized_operator_benchmark.py",
            "symmetrix_commit": fixture.manifest["provenance"]["symmetrix_commit"],
            "model_sha256": fixture.manifest["provenance"]["model_sha256"],
            "graph_sha256": fixture.manifest["provenance"]["graph_sha256"],
            "graph_generation": fixture.manifest["provenance"]["graph_generation"],
            "native_extension_sha256": "sha256:" + "3" * 64,
            "kokkos_execution_space": "Cuda",
            "native": {
                "factorized_selected_direct_forward_executor": "jit_all",
                "factorized_selected_direct_reverse_executor": "jit",
                "jit_artifact_id": "jit-artifact",
                "jit_contract_fingerprint": "jit-fingerprint",
            },
        }
    else:
        report["environment"] = {
            "device": "cuda",
            "gpu_name": "NVIDIA Test GPU",
            "compute_capability": "12.0",
            "torch": "2.13.0+cu130",
            "torch_cuda": "13.0",
        }
        report["upstream"] = {
            "root": "/tmp/execution",
            "commit": "4" * 40,
            "dirty": False,
            "module": "reference_r1",
        }
        report["execution"] = {
            "gemm": "ieee",
            "codegen_config": {"gemm": "ieee"},
            "parameter_gradients": False,
        }
    return report


def test_compare_accepts_only_matched_semantics_and_retains_raw_reports(
    benchmark_module, written_fixture
):
    local = _passing_report(written_fixture, "symmetrix")
    upstream = _passing_report(written_fixture, "execution")

    result = benchmark_module.compare_reports(
        written_fixture.manifest_path, local, upstream
    )

    assert result["status"] == "pass"
    assert result["matched_protocol"]["measurement"] == _MEASUREMENT
    assert result["matched_protocol"]["gpu"] == {
        "name": "NVIDIA Test GPU",
        "compute_capability": "12.0",
        "cuda_major": 13,
    }
    assert result["backends"]["symmetrix"]["timing"]["samples_ms"] == [1.0, 1.1]
    assert result["backends"]["execution"]["timing"]["samples_ms"] == [1.0, 1.1]


def test_compare_accepts_matched_validate_only_reports_without_gpu_timing(
    benchmark_module, written_fixture
):
    local = _passing_report(written_fixture, "symmetrix")
    upstream = _passing_report(written_fixture, "execution")
    local["timing"] = {"skipped": True, "reason": "validate-only"}
    upstream["timing"] = {"skipped": True, "reason": "validate-only"}

    result = benchmark_module.compare_reports(
        written_fixture.manifest_path, local, upstream
    )

    assert result["status"] == "pass"
    assert result["matched_protocol"]["timing_skipped"] is True
    assert result["matched_protocol"]["gpu"] is None


@pytest.mark.parametrize(
    "mutation",
    ("lane", "instruction", "force_sign", "fixture_hash", "failed_output"),
)
def test_compare_rejects_semantic_or_numerical_mismatch(
    benchmark_module, written_fixture, mutation
):
    local = _passing_report(written_fixture, "symmetrix")
    upstream = _passing_report(written_fixture, "execution")
    if mutation == "lane":
        upstream["lane"] = "different"
    elif mutation == "instruction":
        upstream["contract"]["instructions"][0]["i_sh"] = 3
    elif mutation == "force_sign":
        upstream["conventions"]["directed_force"] = "dE/dxyz"
    elif mutation == "fixture_hash":
        upstream["fixture"]["manifest_sha256"] = "sha256:" + "0" * 64
    else:
        upstream["validation"]["outputs"]["edge_force"]["allclose"] = False
        upstream["validation"]["status"] = "fail"
        upstream["status"] = "fail"

    with pytest.raises(FixtureValidationError):
        benchmark_module.compare_reports(written_fixture.manifest_path, local, upstream)


@pytest.mark.parametrize(
    "mutation",
    (
        "backend",
        "measurement",
        "timing_semantics",
        "sample_synchronization",
        "timing_backend",
        "execution_stream",
        "timing_skip_state",
        "timing_skip_reason",
        "native_extension",
        "execution_commit",
        "execution_dirty",
        "gpu_name",
        "compute_capability",
        "cuda_major",
        "gemm_protocol",
    ),
)
def test_compare_rejects_timing_provenance_or_gpu_protocol_mismatch(
    benchmark_module, written_fixture, mutation
):
    local = _passing_report(written_fixture, "symmetrix")
    upstream = _passing_report(written_fixture, "execution")
    if mutation == "backend":
        upstream["backend"] = "unexpected-upstream"
    elif mutation == "measurement":
        upstream["measurement"]["warmup_ms"] = 101.0
    elif mutation == "timing_semantics":
        upstream["timing_semantics"] = "different-boundary"
    elif mutation == "sample_synchronization":
        upstream["timing_protocol"]["sample_completion"] = "device_synchronize"
    elif mutation == "timing_backend":
        upstream["timing"]["timing_backend"] = "host_wall"
    elif mutation == "execution_stream":
        upstream["timing"]["execution_stream"] = "legacy_default_stream"
    elif mutation == "timing_skip_state":
        upstream["timing"] = {"skipped": True, "reason": "validate-only"}
    elif mutation == "timing_skip_reason":
        local["timing"] = {"skipped": True, "reason": "validate-only"}
        upstream["timing"] = {
            "skipped": True,
            "reason": "nvtx-single-iteration",
        }
    elif mutation == "native_extension":
        del local["environment"]["native_extension_sha256"]
    elif mutation == "execution_commit":
        upstream["upstream"]["commit"] = None
    elif mutation == "execution_dirty":
        upstream["upstream"]["dirty"] = True
    elif mutation == "gpu_name":
        upstream["environment"]["gpu_name"] = "Different GPU"
    elif mutation == "compute_capability":
        upstream["environment"]["compute_capability"] = "9.0"
    elif mutation == "cuda_major":
        upstream["environment"]["torch_cuda"] = "12.8"
    else:
        upstream["execution"]["gemm"] = "tf32"
        upstream["execution"]["codegen_config"]["gemm"] = "tf32"

    with pytest.raises(FixtureValidationError):
        benchmark_module.compare_reports(written_fixture.manifest_path, local, upstream)


def test_compare_rejects_boolean_measurement_count(benchmark_module, written_fixture):
    local = _passing_report(written_fixture, "symmetrix")
    upstream = _passing_report(written_fixture, "execution")
    local["measurement"]["warmup_iterations"] = True
    upstream["measurement"]["warmup_iterations"] = True

    with pytest.raises(FixtureValidationError, match="warmup_iterations"):
        benchmark_module.compare_reports(written_fixture.manifest_path, local, upstream)


def test_native_adapter_centralizes_mode_and_raw_timing_contract(benchmark_module):
    class FakeEvaluator:
        def _prepare_factorized_operator_benchmark(self, graph_generation, xyz, r):
            assert graph_generation == 7
            return 11

        def _run_factorized_operator_benchmark(self, token, mode, synchronize):
            self.run_args = (token, mode, synchronize)

        def _measure_factorized_operator_benchmark(self, *args):
            self.measure_args = args
            return {"samples_ms": [0.5], "median_ms": 0.5}

        def _factorized_operator_benchmark_outputs(self, token):
            return {"out": np.zeros((1, 16, 128), dtype=np.float32)}

        def _factorized_operator_export(self, token):
            return {"x": np.zeros((1, 4, 128), dtype=np.float32)}

    evaluator = FakeEvaluator()
    adapter = benchmark_module.NativeOperatorAdapter(evaluator)
    token = adapter.prepare(7, np.zeros((0, 3)), np.zeros((0,)))
    adapter.run(token, "forward_backward")
    timing = adapter.measure(
        token,
        "forward_backward",
        benchmark_module.MeasurementConfig(
            warmup_iterations=2,
            warmup_ms=3.0,
            min_samples=4,
            max_samples=5,
            min_sample_ms=6.0,
        ),
    )

    assert token == 11
    assert evaluator.run_args == (11, "forward_reverse", True)
    assert evaluator.measure_args == (11, "forward_reverse", 2, 3.0, 4, 5, 6.0)
    assert timing["samples_ms"] == [0.5]


def test_all_uses_fresh_local_process_and_preserves_requested_timing(
    benchmark_module, monkeypatch, tmp_path
):
    output = tmp_path / "local.json"
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        output.write_text('{"status":"pass"}', encoding="ascii")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(benchmark_module.subprocess, "run", fake_run)
    report = benchmark_module.run_local_subprocess(
        tmp_path / "model.json",
        tmp_path / "fixture.json",
        output,
        mode="forward_backward",
        structure=None,
        repeat=6,
        measurement=benchmark_module.MeasurementConfig(
            warmup_iterations=2,
            warmup_ms=3.0,
            min_samples=4,
            max_samples=5,
            min_sample_ms=6.0,
        ),
        validate_only=True,
    )

    command = captured["command"]
    assert command[:3] == [
        sys.executable,
        str(pathlib.Path(benchmark_module.__file__).resolve()),
        "local",
    ]
    assert command[command.index("--mode") + 1] == "forward_backward"
    assert command[command.index("--repeat") + 1] == "6"
    assert command[command.index("--warmup-iterations") + 1] == "2"
    assert "--validate-only" in command
    assert captured["kwargs"] == {
        "capture_output": True,
        "check": False,
        "text": True,
    }
    assert report == {"status": "pass"}


def test_cli_exposes_all_orchestration_commands_without_importing_execution(
    benchmark_module,
):
    parser = benchmark_module.build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    source = pathlib.Path(benchmark_module.__file__).read_text(encoding="utf-8")

    assert set(subparsers.choices) == {"export", "local", "compare", "all"}
    assert "import execution" not in source
    assert "from execution" not in source
    assert "subprocess.run(command" in source
