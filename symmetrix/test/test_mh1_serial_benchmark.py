import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import types

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/mh1_serial_benchmark.py"


def _load_benchmark(monkeypatch):
    native = types.SimpleNamespace(
        __file__=str(ROOT / "fake-symmetrix.so"),
        _kokkos_default_execution_space=lambda: "Serial",
    )
    package = types.ModuleType("symmetrix")
    package.Symmetrix = object
    package.symmetrix = native
    ase = types.ModuleType("ase")
    ase_build = types.ModuleType("ase.build")

    def fake_bulk(*_args, **_kwargs):
        return None

    ase_build.bulk = fake_bulk
    monkeypatch.setitem(sys.modules, "symmetrix", package)
    monkeypatch.setitem(sys.modules, "symmetrix.symmetrix", native)
    monkeypatch.setitem(sys.modules, "ase", ase)
    monkeypatch.setitem(sys.modules, "ase.build", ase_build)
    spec = importlib.util.spec_from_file_location("mh1_serial_benchmark_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_properties_and_schedule_overrides_are_strict(monkeypatch):
    benchmark = _load_benchmark(monkeypatch)

    assert benchmark._parse_properties("energy,forces,stress") == (
        "energy",
        "forces",
        "stress",
    )
    with pytest.raises(ValueError, match="duplicates"):
        benchmark._parse_properties("energy,energy")
    with pytest.raises(ValueError, match="unsupported"):
        benchmark._parse_properties("energy,virial")

    monkeypatch.setenv("SYMMETRIX_M0_RTC_SCHEDULE", "auto")
    with pytest.raises(RuntimeError, match="invalid"):
        benchmark._schedule_overrides()
    monkeypatch.setenv("SYMMETRIX_M0_RTC_SCHEDULE", "chunk32")
    monkeypatch.delenv("SYMMETRIX_M1_RTC_SCHEDULE", raising=False)
    assert benchmark._schedule_overrides() == {
        "M0": {
            "environment": "SYMMETRIX_M0_RTC_SCHEDULE",
            "requested": "chunk32",
            "effective": "chunk32",
        },
        "M1": {
            "environment": "SYMMETRIX_M1_RTC_SCHEDULE",
            "requested": None,
            "effective": "chunk32",
        },
    }


def test_array_hash_binds_dtype_shape_and_bytes(monkeypatch):
    benchmark = _load_benchmark(monkeypatch)
    raw = bytes(range(8))
    uint8 = np.frombuffer(raw, dtype=np.uint8)
    uint16 = np.frombuffer(raw, dtype=np.uint16)
    reshaped = uint8.reshape(2, 4)

    assert benchmark._array_sha256(uint8) != benchmark._array_sha256(uint16)
    assert benchmark._array_sha256(uint8) != benchmark._array_sha256(reshaped)


def test_benchmark_teardown_detaches_atoms_and_finalizes(monkeypatch):
    benchmark = _load_benchmark(monkeypatch)
    atoms = [types.SimpleNamespace(calc=object()) for _ in range(2)]
    state = {"live": 0, "initialized": True, "finalized": 0}
    native = types.SimpleNamespace(
        _kokkos_live_object_count=lambda: state["live"],
        _finalize_kokkos=lambda: state.update(initialized=False, finalized=1),
        _kokkos_is_initialized=lambda: state["initialized"],
    )

    benchmark._finalize_benchmark_runtime(native, atoms)

    assert [item.calc for item in atoms] == [None, None]
    assert state == {"live": 0, "initialized": False, "finalized": 1}


def test_benchmark_teardown_failure_is_nonzero_in_subprocess():
    source = f"""
import importlib.util
import sys
import types
package = types.ModuleType('symmetrix')
package.Symmetrix = object
package.symmetrix = types.SimpleNamespace(__file__='fake.so')
ase = types.ModuleType('ase')
ase_build = types.ModuleType('ase.build')
ase_build.bulk = lambda *args, **kwargs: None
sys.modules['symmetrix'] = package
sys.modules['symmetrix.symmetrix'] = package.symmetrix
sys.modules['ase'] = ase
sys.modules['ase.build'] = ase_build
spec = importlib.util.spec_from_file_location('benchmark', {str(SCRIPT)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
native = types.SimpleNamespace(
    _kokkos_live_object_count=lambda: 1,
    _finalize_kokkos=lambda: None,
    _kokkos_is_initialized=lambda: True,
)
module._finalize_benchmark_runtime(native, [types.SimpleNamespace(calc=object())])
"""
    completed = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )

    assert completed.returncode != 0
    assert "retained 1 live Kokkos object" in completed.stderr
    assert "Kokkos::Cuda ERROR" not in completed.stderr


def test_benchmark_teardown_subprocess_exits_clean_with_zero_live_objects():
    source = f"""
import importlib.util
import sys
import types
package = types.ModuleType('symmetrix')
package.Symmetrix = object
package.symmetrix = types.SimpleNamespace(__file__='fake.so')
ase = types.ModuleType('ase')
ase_build = types.ModuleType('ase.build')
ase_build.bulk = lambda *args, **kwargs: None
sys.modules['symmetrix'] = package
sys.modules['symmetrix.symmetrix'] = package.symmetrix
sys.modules['ase'] = ase
sys.modules['ase.build'] = ase_build
spec = importlib.util.spec_from_file_location('benchmark', {str(SCRIPT)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
state = {{'initialized': True}}
native = types.SimpleNamespace(
    _kokkos_live_object_count=lambda: 0,
    _finalize_kokkos=lambda: state.update(initialized=False),
    _kokkos_is_initialized=lambda: state['initialized'],
)
module._finalize_benchmark_runtime(native, [types.SimpleNamespace(calc=object())])
print(native._kokkos_live_object_count())
"""
    completed = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "0"
    assert "Kokkos::Cuda ERROR" not in completed.stderr


def test_property_outputs_hash_energy_forces_and_stress(monkeypatch):
    benchmark = _load_benchmark(monkeypatch)
    outputs = benchmark._property_outputs(
        {
            "energy": np.float64(1.25),
            "forces": np.arange(6, dtype=np.float64).reshape(2, 3),
            "stress": np.arange(6, dtype=np.float64),
        },
        ("energy", "forces", "stress"),
    )

    assert set(key for key in outputs if key.endswith("_sha256")) == {
        "energy_sha256",
        "forces_sha256",
        "stress_sha256",
    }


def test_explicit_extension_uses_selected_backend_module_name(monkeypatch):
    benchmark = _load_benchmark(monkeypatch)
    descriptor = types.SimpleNamespace(
        package="symmetrix_backend_cuda13_sm120",
        module="_native_cuda13_sm120",
    )

    assert (
        benchmark._explicit_native_module_name("_native_cuda13_sm120", descriptor)
        == "symmetrix_backend_cuda13_sm120._native_cuda13_sm120"
    )
    with pytest.raises(RuntimeError, match="does not match selected backend"):
        benchmark._explicit_native_module_name("_native_cpu", descriptor)


def test_explicit_extension_preserves_metadata_identified_legacy_cpu(monkeypatch):
    benchmark = _load_benchmark(monkeypatch)
    descriptor = types.SimpleNamespace(
        package="symmetrix",
        module="_native_cpu",
        backend="cpu",
        descriptor_path="bundled",
    )

    assert (
        benchmark._explicit_native_module_name("symmetrix", descriptor)
        == "symmetrix.symmetrix"
    )

    non_bundled = types.SimpleNamespace(
        package="symmetrix_backend_cpu_v3",
        module="_native_cpu_v3",
        backend="cpu",
        descriptor_path="/installed/backend.json",
    )
    with pytest.raises(RuntimeError, match="does not match selected backend"):
        benchmark._explicit_native_module_name("symmetrix", non_bundled)


def test_product_route_uses_common_module(monkeypatch, tmp_path):
    benchmark = _load_benchmark(monkeypatch)
    artifact = tmp_path / "execution.cubin"
    artifact.write_bytes(b"exact-module")
    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    launch_plan = {
        "split_phase_composition": True,
        "schema": "symmetrix.jit.mh1.cuda-launch-plan/6",
        "version": 6,
        "node_state_policy": "full-retention-v1",
        "module_identity": "sha256:module",
        "nodes": [{}, {}],
    }
    manifest = {
        "artifact_sha256": artifact_sha256,
        "key_inputs": {
            "abi": {"tag": "symmetrix.jit.mh1.cuda-module/5", "version": 5},
            "jit_generation_version": 19,
            "build": {"request": {"launch_plan": launch_plan}},
        },
    }
    artifact.with_name("manifest.json").write_text(json.dumps(manifest))

    class Evaluator:
        def __init__(self):
            self.execution_mh1_product_module_diagnostics = ["m0", "m1"]
            self.loads = []
            self.clears = 0

        def _execution_mh1_product_module_ready(self, _layer):
            return True

        def _execution_mh1_product_module_artifact_id(self, layer):
            return f"artifact-{layer}"

        def _load_jit_mh1_cuda_plugin_v4(self, path, plan):
            self.loads.append((path, json.loads(plan)))

        def _clear_execution_mh1_product_modules(self):
            self.clears += 1

    evaluator = Evaluator()
    calculator = types.SimpleNamespace(
        jit_artifact_path=str(artifact),
        execution_mh1_node_state_policy="full-retention-v1",
        evaluator=evaluator,
    )
    record = benchmark._configure_mh1_product_route(calculator, "shared")

    assert record["external_product"] is True
    assert record["split_phase_composition"] is True
    assert record["plan_schema"] == "symmetrix.jit.mh1.cuda-launch-plan/6"
    assert record["device_module_abi"]["version"] == 5
    assert record["jit_generation_version"] == 19
    assert record["node_state_policy"] == "full-retention-v1"
    assert record["module_sha256"] == artifact_sha256
    assert evaluator.loads == []
    assert evaluator.clears == 0


def test_retained_product_oracle_fails_clearly_after_generation_18(monkeypatch):
    benchmark = _load_benchmark(monkeypatch)
    with pytest.raises(RuntimeError, match="retired.*generation 19"):
        benchmark._configure_mh1_product_route(None, "retained-monolithic-oracle")


def test_release_product_route_rejects_generation_18_compatibility(
    monkeypatch, tmp_path
):
    benchmark = _load_benchmark(monkeypatch)
    artifact = tmp_path / "execution.cubin"
    artifact.write_bytes(b"legacy-module")
    manifest = {
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "key_inputs": {
            "abi": {"tag": "symmetrix.jit.mh1.cuda-module/4", "version": 4},
            "jit_generation_version": 18,
            "build": {
                "request": {
                    "launch_plan": {
                        "external_product": True,
                        "node_state_policy": "full-retention-v1",
                        "module_identity": "sha256:legacy",
                        "nodes": [{}, {}],
                    }
                }
            },
        },
    }
    artifact.with_name("manifest.json").write_text(json.dumps(manifest))
    calculator = types.SimpleNamespace(
        jit_artifact_path=str(artifact),
        execution_mh1_node_state_policy="full-retention-v1",
    )

    with pytest.raises(RuntimeError, match="generation-19 release contract"):
        benchmark._configure_mh1_product_route(calculator, "shared")


def test_evaluator_timing_collects_all_requested_properties_each_call(monkeypatch):
    benchmark = _load_benchmark(monkeypatch)
    properties_seen = []

    class Evaluator:
        streamed_edges_mode = "reference"

        def __init__(self):
            self.forward_count = 7
            self.reverse_count = 11

        def compute_node_energies_forces(self, *_args):
            self.forward_count += 1
            self.reverse_count += 1
            return None

        def fence(self):
            return None

        def _execution_mh1_product_module_forward_launch_count(self, _layer):
            return self.forward_count

        def _execution_mh1_product_module_reverse_launch_count(self, _layer):
            return self.reverse_count

    class Calculator:
        streamed_edges = "reference"
        cutoff = 4.0
        neighbor_skin = 0.5
        evaluator = Evaluator()
        jit_operator_modules = {"product_0": {}}

        def _mace_inputs(self, atoms):
            return (
                len(atoms),
                np.zeros(len(atoms), dtype=np.int32),
                np.zeros(len(atoms), dtype=np.int32),
                np.zeros(2, dtype=np.int32),
                np.zeros(2, dtype=np.int32),
                np.zeros((2, 3)),
                np.zeros(2, dtype=np.int32),
            )

        def _collect_mace_results(self, atoms, _inputs, properties, *_args):
            properties_seen.append(tuple(properties))
            return {
                "energy": 1.25,
                "forces": np.zeros((len(atoms), 3)),
                "stress": np.arange(6, dtype=float),
            }

    class Atoms:
        numbers = np.array([7, 13, 7, 13])
        positions = np.zeros((4, 3))
        cell = np.eye(3)
        pbc = np.array([True, True, True])

        def __len__(self):
            return 4

    atoms = Atoms()
    record = benchmark._benchmark(
        Calculator(),
        atoms,
        2,
        3,
        False,
        properties=("energy", "forces", "stress"),
    )

    assert properties_seen == [("energy", "forces", "stress")] * 5
    assert record["timed_properties"] == ["energy", "forces", "stress"]
    assert record["energy_eV"] == 1.25
    assert record["stress_eV_per_A3"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert len(record["forces_sha256"]) == 64
    assert len(record["stress_sha256"]) == 64
    assert record["evaluator"]["median_us_per_atom"] >= 0.0
    assert record["graph_dimensions"] == {
        "atoms": 4,
        "directed_edges": 2,
        "model_cutoff_A": 4.0,
        "neighbor_skin_A": 0.5,
        "effective_neighbor_cutoff_A": 4.5,
    }
    assert record["_product_launch_counter_window"] == {
        0: {
            "scope": "benchmark-warmups-and-timed-samples",
            "warmups": 2,
            "samples": 3,
            "warmup_forward_launch_count": 2,
            "warmup_reverse_launch_count": 2,
            "timed_forward_launch_count": 3,
            "timed_reverse_launch_count": 3,
            "total_forward_launch_count": 5,
            "total_reverse_launch_count": 5,
        }
    }


def test_product_evidence_keeps_full_plugin_origin_independent(monkeypatch, tmp_path):
    """A prebuilt common M0 must not relabel a runtime MH-1 plugin prebuilt."""
    benchmark = _load_benchmark(monkeypatch)
    module = tmp_path / "product-0.cubin"
    module.write_bytes(b"canonical-m0")
    # Product and full-plugin fixtures normally live in separate artifact
    # directories; retain that layout so each has its own manifest.
    plugin_directory = tmp_path / "plugin"
    plugin_directory.mkdir()
    plugin = plugin_directory / "mh1.cubin"
    plugin.write_bytes(b"mh1-plugin")
    plugin.with_name("manifest.json").write_text(
        json.dumps(
            {
                "key_inputs": {
                    "abi": {
                        "tag": "symmetrix.jit.mh1.cuda-module/5",
                        "version": 5,
                    },
                    "jit_generation_version": 19,
                    "build": {
                        "request": {
                            "launch_plan": {
                                "schema": "symmetrix.jit.mh1.cuda-launch-plan/6",
                                "version": 6,
                                "split_phase_composition": True,
                                "module_identity": "sha256:plugin",
                            }
                        }
                    },
                }
            }
        )
    )

    diagnostic = (
        "product_0: kind=M0 schedule=chunk32 target=sm_120 "
        "artifact_id=jit-m0 forward_launches=12 reverse_launches=12 "
        "blocks_per_compute_unit=8 persistent_block_cap=1360 "
        "forward_logical_tiles=32768 forward_tiles_per_block=256 "
        "forward_required_blocks=128 forward_realized_blocks=128 "
        "reverse_logical_tiles=32768 reverse_tiles_per_block=256 "
        "reverse_required_blocks=128 reverse_realized_blocks=128"
    )

    class Evaluator:
        execution_mh1_product_module_diagnostics = [diagnostic]
        factorized_fallback_evaluation_count = 0

        @staticmethod
        def _execution_mh1_product_module_ready(_layer):
            return True

        @staticmethod
        def _execution_mh1_product_module_artifact_id(_layer):
            return "jit-m0"

        @staticmethod
        def _execution_mh1_product_module_forward_launch_count(_layer):
            return 12

        @staticmethod
        def _execution_mh1_product_module_reverse_launch_count(_layer):
            return 12

    calculator = types.SimpleNamespace(
        jit_origin="runtime",
        jit_artifact_id="mh1-plugin",
        jit_artifact_path=str(plugin),
        jit_operator_modules={
            "product_0": {
                "operator": "M0",
                "origin": "prebuilt",
                "artifact_id": "jit-m0",
                "artifact_path": str(module),
                "schedule": "chunk32",
                "target": "sm_120",
                "backend": "cuda",
                "precision": "float32",
                "generation_fingerprint": "sha256:generation",
                "semantic_fingerprint": "sha256:semantic",
                "structure_fingerprint": "sha256:structure",
                "channels": 128,
                "type_count": 1,
                "input_components": 16,
                "output_components": 4,
                "correlation": 3,
                "term_count": 44,
                "coefficient_count": 5632,
            }
        },
        evaluator=Evaluator(),
    )

    catalog_entry = {
        "operator": "M0",
        "identity": {"artifact_id": "jit-m0", "generation_revision": 19},
        "abi": {"tag": "symmetrix.jit.operator-module/1", "version": 3},
        "installed_relative_path": module.name,
    }
    generated_operator_core = types.ModuleType("symmetrix.generated_operator_core")
    generated_operator_core.validated_installed_operator_catalog = (
        lambda _root, _build: (
            {
                "catalog_path": str(tmp_path / "catalog.json"),
                "catalog_sha256": "a" * 64,
            },
            {"artifacts": [catalog_entry]},
            tmp_path,
        )
    )
    monkeypatch.setitem(
        sys.modules, "symmetrix.generated_operator_core", generated_operator_core
    )
    monkeypatch.setattr(
        benchmark.native_symmetrix, "__file__", str(tmp_path / "native.so")
    )
    monkeypatch.setattr(
        benchmark.native_symmetrix,
        "_backend_build_info",
        lambda: {"prebuilt_operator_artifacts": {"mode": "full"}},
        raising=False,
    )

    evidence = benchmark._mh1_product_runtime(calculator, 1)

    assert evidence["full_plugin"]["origin"] == "runtime"
    assert evidence["full_plugin"]["artifact_id"] == "mh1-plugin"
    assert evidence["full_plugin"]["artifact_path"] == str(plugin)
    assert evidence["full_plugin"]["prebuilt"] is False
    layer = evidence["layers"][0]
    assert layer["origin"] == "prebuilt"
    assert layer["abi"]["version"] == 3
    assert layer["jit_generation_version"] == 19
    assert layer["contract_dimensions"]["coefficient_count"] == 5632
    assert layer["realized_geometry"]["forward_realized_blocks"] == 128
    assert layer["realized_geometry"]["reverse_realized_blocks"] == 128
    assert layer["artifact_sha256"] == hashlib.sha256(b"canonical-m0").hexdigest()
    assert layer["manifest_sha256"] is None
    assert layer["artifact_provenance"]["kind"] == "installed-catalog"
    assert layer["artifact_provenance"]["catalog_sha256"] == "a" * 64


def test_external_product_timings_require_complete_binding(monkeypatch, tmp_path):
    benchmark = _load_benchmark(monkeypatch)
    profile = tmp_path / "profile.nsys-rep"
    report = tmp_path / "kernel-summary.csv"
    profiled_record = tmp_path / "profiled-record.json"
    profile.write_bytes(b"NSYSREP\x00minimal-test-profile")
    report.write_text(
        '"Time (%)","Total Time (ns)","Instances","Avg (ns)","Med (ns)",'
        '"Min (ns)","Max (ns)","StdDev (ns)","Name"\n'
        '"40.0","10000","5","2000","2000","1900","2100","50",'
        '"symmetrix_mh1_m0_forward"\n'
        '"60.0","15000","5","3000","3000","2900","3100","50",'
        '"symmetrix_mh1_m0_reverse"\n'
    )
    profiled_command = (
        "python benchmarks/mh1_serial_benchmark.py model.pt --output profiled.json"
    )
    attachment_command = (
        "python benchmarks/mh1_serial_benchmark.py model.pt "
        "--product-timings-json timings.json --output release.json"
    )
    profile_command = "nsys profile --trace=cuda --output profile " + profiled_command
    profiled_record.write_text(
        json.dumps(
            {
                "metadata": {
                    "git_revision": "a" * 40,
                    "exact_command": profiled_command,
                    "native_extension_sha256": "d" * 64,
                    "native_build_info": {
                        "source_content_sha256": "b" * 64,
                        "native_source_content_sha256": "c" * 64,
                    },
                    "model": {"sha256": "5" * 64},
                    "warmups": 10,
                    "repeats": 5,
                },
                "systems": [
                    {
                        "system_sha256": "6" * 64,
                        "timed_properties": ["energy", "forces", "stress"],
                        "mh1_product_runtime": {
                            "full_plugin": {
                                "artifact_id": "mh1-plugin",
                                "artifact_sha256": "1" * 64,
                                "manifest_sha256": "2" * 64,
                                "module_identity": "sha256:plugin",
                            },
                            "layers": [
                                {
                                    "layer": 0,
                                    "origin": "runtime",
                                    "artifact_id": "m0-module",
                                    "artifact_path": "/jit/m0/module.cubin",
                                    "artifact_sha256": "3" * 64,
                                    "manifest_sha256": "4" * 64,
                                    "artifact_provenance": {
                                        "kind": "runtime-manifest",
                                        "manifest_path": "/jit/m0/manifest.json",
                                        "manifest_sha256": "4" * 64,
                                        "catalog_path": None,
                                        "catalog_sha256": None,
                                        "catalog_entry_sha256": None,
                                    },
                                }
                            ],
                        },
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="source"):
        benchmark._normalize_product_timings(
            {"layers": [{"layer": 0, "forward_us": 2.0, "reverse_us": 3.0}]}
        )

    value = {
        "source": "Nsight Systems 2026.1.3 GPU kernel summary",
        "collection_scope": "five post-warmup evaluator calls",
        "statistic": "median",
        "binding": {
            "git_revision": "a" * 40,
            "source_content_sha256": "b" * 64,
            "native_source_content_sha256": "c" * 64,
            "native_extension_sha256": "d" * 64,
            "profiled_command": profiled_command,
            "profiled_command_sha256": hashlib.sha256(
                profiled_command.encode()
            ).hexdigest(),
            "attachment_command": attachment_command,
            "attachment_command_sha256": hashlib.sha256(
                attachment_command.encode()
            ).hexdigest(),
            "profiled_record_path": str(profiled_record),
            "profiled_record_sha256": hashlib.sha256(
                profiled_record.read_bytes()
            ).hexdigest(),
            "model_sha256": "5" * 64,
            "system_sha256": "6" * 64,
            "properties": ["energy", "forces", "stress"],
            "sample_scope": {"warmups": 10, "samples": 5},
            "profile_command": profile_command,
            "profile_command_sha256": hashlib.sha256(
                profile_command.encode()
            ).hexdigest(),
            "profile_path": str(profile),
            "profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
            "report_path": str(report),
            "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            "full_plugin": {
                "artifact_id": "mh1-plugin",
                "artifact_sha256": "1" * 64,
                "manifest_sha256": "2" * 64,
                "module_identity": "sha256:plugin",
            },
            "layers": [
                {
                    "layer": 0,
                    "operator": "M0",
                    "origin": "runtime",
                    "artifact_id": "m0-module",
                    "artifact_path": "/jit/m0/module.cubin",
                    "artifact_sha256": "3" * 64,
                    "manifest_sha256": "4" * 64,
                    "artifact_provenance": {
                        "kind": "runtime-manifest",
                        "manifest_path": "/jit/m0/manifest.json",
                        "manifest_sha256": "4" * 64,
                        "catalog_path": None,
                        "catalog_sha256": None,
                        "catalog_entry_sha256": None,
                    },
                    "profile_kernels": {
                        "forward": "symmetrix_mh1_m0_forward",
                        "reverse": "symmetrix_mh1_m0_reverse",
                    },
                }
            ],
        },
        "layers": [{"layer": 0, "forward_us": 2.0, "reverse_us": 3.0}],
    }
    expected_attachment = {
        key: value["binding"][key]
        for key in (
            "git_revision",
            "source_content_sha256",
            "native_source_content_sha256",
            "native_extension_sha256",
            "attachment_command",
            "attachment_command_sha256",
            "full_plugin",
            "layers",
        )
    }
    timings = benchmark._normalize_product_timings(value, expected_attachment)
    assert timings["instrumentation"] == "external-profile"
    assert timings["evaluator_samples_include_product_work"] is True
    assert timings["profile_timings_derived_from_evaluator_samples"] is False
    assert timings["intrusive_timed_instrumentation"] is False
    assert timings["collected_by_benchmark"] is False
    with pytest.raises(ValueError, match="statistic"):
        benchmark._normalize_product_timings(
            {**value, "statistic": "unspecified"}, value["binding"]
        )
    missing_command = json.loads(json.dumps(value))
    del missing_command["binding"]["profiled_command"]
    with pytest.raises(ValueError, match="profiled_command"):
        benchmark._normalize_product_timings(missing_command)
    mismatched_command = json.loads(json.dumps(value))
    mismatched_command["binding"]["profiled_command_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="profiled command hash"):
        benchmark._normalize_product_timings(mismatched_command)
    mismatched = json.loads(json.dumps(value))
    mismatched["binding"]["full_plugin"]["artifact_id"] = "wrong"
    with pytest.raises(ValueError, match="binding"):
        benchmark._normalize_product_timings(mismatched, value["binding"])
    bad_hash = json.loads(json.dumps(value))
    bad_hash["binding"]["profile_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="profile hash"):
        benchmark._normalize_product_timings(bad_hash, bad_hash["binding"])
    forged_rows = json.loads(json.dumps(value))
    report.write_text(
        '"Time (%)","Total Time (ns)","Instances","Avg (ns)","Med (ns)",'
        '"Min (ns)","Max (ns)","StdDev (ns)","Name"\n'
        '"40.0","10000","5","2000","200000","1900","2100","50",'
        '"symmetrix_mh1_m0_forward"\n'
        '"60.0","15000","5","3000","300000","2900","3100","50",'
        '"symmetrix_mh1_m0_reverse"\n'
    )
    forged_rows["binding"]["report_sha256"] = hashlib.sha256(
        report.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="profiler report"):
        benchmark._normalize_product_timings(forged_rows, forged_rows["binding"])


def test_replay_provenance_records_interpreter_launcher_affinity_and_allowlist(
    monkeypatch,
):
    benchmark = _load_benchmark(monkeypatch)
    monkeypatch.setenv("SYMMETRIX_BACKEND", "cuda13-sm120")
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", "/task/jit")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-be-recorded")
    args = types.SimpleNamespace(cpu=3, cpus=None)

    provenance = benchmark._replay_provenance(args)

    assert provenance["python_executable"] == sys.executable
    assert provenance["launcher"]["requested_cpu_affinity"] == [3]
    assert provenance["launcher"]["effective_cpu_affinity"]
    assert provenance["environment"]["SYMMETRIX_BACKEND"] == "cuda13-sm120"
    assert provenance["environment"]["SYMMETRIX_JIT_CACHE"] == "/task/jit"
    assert "UNRELATED_SECRET" not in provenance["environment"]


def test_product_evidence_rejects_invalid_coefficient_count(monkeypatch, tmp_path):
    benchmark = _load_benchmark(monkeypatch)
    module = tmp_path / "product.cubin"
    module.write_bytes(b"module")
    module.with_name("manifest.json").write_text(
        json.dumps(
            {
                "artifact_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
                "key_inputs": {
                    "abi": {"tag": "symmetrix.jit.operator-module/1", "version": 3},
                    "jit_generation_version": 19,
                    "build": {},
                },
            }
        )
    )
    plugin_directory = tmp_path / "plugin"
    plugin_directory.mkdir()
    plugin = plugin_directory / "mh1.cubin"
    plugin.write_bytes(b"plugin")
    plugin.with_name("manifest.json").write_text(
        json.dumps(
            {
                "key_inputs": {
                    "abi": {"version": 5},
                    "jit_generation_version": 19,
                    "build": {
                        "request": {
                            "launch_plan": {
                                "module_identity": "sha256:plugin",
                                "schema": "symmetrix.jit.mh1.cuda-launch-plan/6",
                                "version": 6,
                                "split_phase_composition": True,
                            }
                        }
                    },
                }
            }
        )
    )
    evaluator = types.SimpleNamespace(
        execution_mh1_product_module_diagnostics=[],
        factorized_fallback_evaluation_count=0,
        _execution_mh1_product_module_ready=lambda _layer: True,
        _execution_mh1_product_module_artifact_id=lambda _layer: "m0",
        _execution_mh1_product_module_forward_launch_count=lambda _layer: 1,
        _execution_mh1_product_module_reverse_launch_count=lambda _layer: 1,
    )
    calculator = types.SimpleNamespace(
        jit_origin="runtime",
        jit_artifact_id="plugin",
        jit_artifact_path=str(plugin),
        jit_operator_modules={
            "product_0": {
                "operator": "M0",
                "origin": "runtime",
                "artifact_id": "m0",
                "artifact_path": str(module),
                "schedule": "chunk32",
                "backend": "cuda",
                "target": "sm_120",
                "precision": "float32",
                "generation_fingerprint": "sha256:g",
                "semantic_fingerprint": "sha256:s",
                "structure_fingerprint": "sha256:t",
                "channels": 8,
                "type_count": 1,
                "input_components": 4,
                "output_components": 1,
                "correlation": 3,
                "term_count": 7,
                "coefficient_count": 55,
            }
        },
        evaluator=evaluator,
    )

    with pytest.raises(RuntimeError, match="coefficient_count"):
        benchmark._mh1_product_runtime(calculator, 1)
