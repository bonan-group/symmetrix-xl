import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from symmetrix import generated_operator_core as operator_core
from symmetrix import installed_operator_artifact as installed_artifact
from symmetrix import jit as jit_runtime
from symmetrix import mace_jit_device_artifact as device_artifact
from symmetrix import mace_jit_operator_codegen as operator_codegen
from symmetrix.cli import extract_mace as extract_cli
from symmetrix.cli import prepare_jit_device_artifact as artifact_cli


@pytest.fixture(autouse=True)
def _accept_lightweight_prepared_results(monkeypatch):
    validate_entry = jit_runtime._validated_entry

    def validate_real_entries(entry, cache_key, inputs):
        if cache_key.startswith("cache-"):
            return None
        return validate_entry(entry, cache_key, inputs)

    monkeypatch.setattr(jit_runtime, "_validated_entry", validate_real_entries)


class _Evaluator:
    def __init__(self, backend):
        self.events = []
        self.execution_device_execution_environment = {
            "available": True,
            "backend": backend,
            "execution_space": "Cuda" if backend == "cuda" else "HIP",
            "compute_capability_code": 120,
            "architecture": "sm_120" if backend == "cuda" else "gfx1151",
            "raw_agent_target": "" if backend == "cuda" else "gfx1151",
            "native_subgroup_width": 32,
            "compute_unit_count": 8,
        }
        setattr(self, f"execution_{backend}_plugin_ready", False)
        setattr(self, f"execution_{backend}_plugin_artifact_id", "")
        self.m0_implementation = "generic"
        self.r0_implementation = "generic"
        self._r0_loader_implementation = "generic"
        self.m0_device_module_ready = False
        self.r0_device_module_ready = False
        self.m0_device_module_artifact_id = ""
        self.r0_device_module_artifact_id = ""

    def set_streamed_edges(self, mode):
        self.events.append(("mode", mode))

    def load(self, path, *arguments):
        backend = self.execution_device_execution_environment["backend"]
        self.events.append(("load", path, *arguments))
        setattr(self, f"execution_{backend}_plugin_ready", True)
        setattr(self, f"execution_{backend}_plugin_artifact_id", "artifact-a")

    def _load_jit_cuda_plugin(self, path, blocks):
        self.events.append(("load", path, blocks))
        self.jit_cuda_plugin_ready = True
        self.jit_cuda_plugin_artifact_id = "artifact-r1-installed"

    def _load_m0_device_module(self, path, schedule, blocks):
        self.events.append(("load-M0", path, schedule, blocks))
        self.m0_device_module_ready = True
        self.m0_device_module_artifact_id = "artifact-m0"

    def _load_r0_device_module(
        self, path, blocks, apply_edge_weights=False, schedule="receiver-edge16"
    ):
        self.events.append(("load-R0", path, schedule, blocks))
        self.r0_device_module_ready = True
        self.r0_device_module_artifact_id = "artifact-r0"


def _model(tmp_path, model_type="MACE"):
    path = tmp_path / "model.json"
    path.write_text(
        json.dumps(
            {
                "model_type": model_type,
                "execution_contracts": {"R1": {"fixture": "contract-a"}},
            }
        ),
        encoding="utf-8",
    )
    return path


def _full_cuda_model_data():
    contracts = Path(__file__).resolve().parent / "data" / "execution_contracts"
    return {
        "model_type": "MACE",
        "execution_contracts": {
            "R1": json.loads((contracts / "jit_r1_contract.json").read_text()),
            "M0": json.loads((contracts / "standard_m0_contract.json").read_text()),
            "R0": json.loads((contracts / "standard_r0_contract.json").read_text()),
        },
    }


def _install_full_cuda_catalog(monkeypatch, tmp_path, evaluator, model_data):
    package = tmp_path / "installed_full_backend"
    data_root = package / "_operator_artifacts"
    data_root.mkdir(parents=True)
    environment = evaluator.execution_device_execution_environment
    contracts = model_data["execution_contracts"]
    r1_plan = device_artifact._factorized_device_installed_plan(
        "cuda", environment, contracts["R1"], "float32", evaluator
    )
    _, m0_metadata = operator_codegen.render_jit_m0_device_module(
        contracts["M0"],
        precision="float32",
        target="sm_120",
        schedule="chunk32",
    )
    _, r0_metadata = operator_codegen.render_jit_r0_device_module(
        contracts["R0"],
        precision="float32",
        target="sm_120",
        schedule="receiver-edge16",
        apply_edge_weights=False,
    )
    metadata = {
        "R1": r1_plan.metadata,
        "M0": m0_metadata,
        "R0": r0_metadata,
    }
    abis = {
        "R1": {
            "tag": r1_plan.metadata["abi"],
            "version": r1_plan.metadata["abi_version"],
        },
        "M0": jit_runtime.artifact_cache_inputs(
            m0_metadata, operator_codegen.OPERATOR_ARTIFACT_ENVELOPE
        ),
        "R0": jit_runtime.artifact_cache_inputs(
            r0_metadata, operator_codegen.OPERATOR_ARTIFACT_ENVELOPE
        ),
    }
    entries = []
    paths = {}
    for operator in ("R1", "M0", "R0"):
        path = data_root / operator.lower() / f"{operator.lower()}.cubin"
        path.parent.mkdir()
        path.write_bytes(f"installed-{operator}".encode())
        paths[operator] = path
        entries.append(
            {
                "operator": operator,
                "identity": metadata[operator]["generated_identity"],
                "binary_format": "cubin",
                "installed_relative_path": str(path.relative_to(data_root)),
                "checksum": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                "abi": abis[operator],
                "capabilities": metadata[operator]["capabilities"],
                "build_provenance": {
                    "source_content_sha256": "sha256:" + "a" * 64,
                    "compiler_id": "NVRTC",
                    "compiler_version": "13.3",
                    **(
                        {
                            "floating_point_policy": metadata[operator][
                                "generated_identity"
                            ]["floating_point_policy"]
                        }
                        if operator == "R0"
                        else {}
                    ),
                },
            }
        )
    catalog = operator_core.generated_operator_catalog(entries)
    catalog_path = data_root / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    build_info = {
        "mode": "full",
        "backend": "cuda",
        "target": "sm_120",
        "catalog_relative_path": "_operator_artifacts/catalog.json",
        "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "artifact_count": 3,
        "installed_bytes": catalog_path.stat().st_size
        + sum(path.stat().st_size for path in paths.values()),
        "source_content_sha256": "a" * 64,
        "native_source_content_sha256": "b" * 64,
        "compiler_id": "NVRTC",
        "compiler_version": "13.3",
        "build_duration_seconds": 0.5,
    }
    monkeypatch.setattr(
        device_artifact.symmetrix,
        "_backend_build_info",
        lambda: {
            "source_content_sha256": "a" * 64,
            "native_source_content_sha256": "b" * 64,
            "prebuilt_operator_artifacts": build_info,
        },
    )
    monkeypatch.setattr(
        installed_artifact.resources, "files", lambda package_name: package
    )
    return paths


@pytest.mark.parametrize(
    ("backend", "precision", "model_type", "class_name", "load_arguments"),
    [
        ("cuda", "float32", "MACE", "MACEKokkosFloat", (8,)),
        ("hip", "float64", "MACEField", "MACEKokkos", ()),
        ("cuda", "float64", "MACELES", "MACELESKokkos", (8,)),
    ],
)
def test_prepare_device_artifact_without_ase_calculator(
    monkeypatch,
    tmp_path,
    backend,
    precision,
    model_type,
    class_name,
    load_arguments,
):
    evaluator = _Evaluator(backend)
    constructed = []
    initialized = []
    prepare_calls = []
    source_content_sha256 = "a" * 64
    artifact = tmp_path / ("module.cubin" if backend == "cuda" else "module.hsaco")
    manifest = tmp_path / "manifest.json"

    def evaluator_class(path):
        constructed.append(path)
        return evaluator

    monkeypatch.setattr(
        device_artifact.symmetrix, "_kokkos_is_initialized", lambda: False
    )
    monkeypatch.setattr(
        device_artifact.symmetrix, "_init_kokkos", lambda: initialized.append(True)
    )
    monkeypatch.setattr(device_artifact.symmetrix, class_name, evaluator_class)

    def prepare(source, **arguments):
        prepare_calls.append((source, arguments))
        return SimpleNamespace(
            status="built",
            available=True,
            cache_key="cache-a",
            artifact_path=artifact,
            manifest_path=manifest,
            manifest={"key_inputs": {"fixture": backend}},
            diagnostics=("compiled",),
            reason=None,
        )

    plan = SimpleNamespace(
        selected_compiler="nvrtc" if backend == "cuda" else "hiprtc",
        policy="automatic",
        metadata={
            "artifact_id": "artifact-a",
            "persistent_blocks_per_compute_unit": 8,
        },
        loader=evaluator.load,
        ready_attribute=f"execution_{backend}_plugin_ready",
        artifact_attribute=f"execution_{backend}_plugin_artifact_id",
        variant_id="variant-a",
        edge_policy={
            "strategy": "wave",
            "persistent_blocks_per_compute_unit": 8,
        },
        build_attempt=lambda compiler: SimpleNamespace(
            source=f"{backend} source",
            prepare=prepare,
            arguments={"build": {"compiler": compiler}},
        ),
    )
    monkeypatch.setattr(
        device_artifact,
        "_factorized_device_backend_plan",
        lambda selected_backend, environment, contract, dtype, selected_evaluator: plan,
    )
    installed_plan = SimpleNamespace(
        metadata={
            "generated_identity": {
                "operator": "R1",
                "artifact_id": "generated-r1-request",
            },
            "abi": f"symmetrix.jit.{backend}-plugin/4",
            "abi_version": 4,
            "capabilities": 0x3,
            "persistent_blocks_per_compute_unit": 8,
        },
        loader=evaluator.load,
        ready_attribute=f"execution_{backend}_plugin_ready",
        artifact_attribute=f"execution_{backend}_plugin_artifact_id",
        variant_id="variant-installed",
        edge_policy={"persistent_blocks_per_compute_unit": 8},
    )
    monkeypatch.setattr(
        device_artifact,
        "_factorized_device_installed_plan",
        lambda *args, **kwargs: installed_plan,
    )
    monkeypatch.setattr(
        device_artifact,
        "resolve_installed_operator_artifact",
        lambda **kwargs: SimpleNamespace(
            status="absent",
            diagnostics=(
                "origin=prebuilt status=absent operator=R1 reason=catalog-absent",
            ),
        ),
    )
    monkeypatch.setattr(
        device_artifact,
        "native_source_content_sha256",
        lambda native: source_content_sha256,
    )

    result = device_artifact.prepare_jit_device_artifact(
        _model(tmp_path, model_type),
        precision=precision,
        backend=backend,
        cache_root=tmp_path / "cache",
    )

    assert initialized == [True]
    assert constructed == [str((tmp_path / "model.json").resolve())]
    assert evaluator.events == [
        ("mode", "factorized"),
        ("load", str(artifact), *load_arguments),
    ]
    assert prepare_calls == [
        (
            f"{backend} source",
            {
                "build": {
                    "compiler": plan.selected_compiler,
                    "jit_generation_version": device_artifact.JIT_GENERATION_VERSION,
                },
                "source_content_sha256": source_content_sha256,
                "cache_root": tmp_path / "cache",
                **(
                    {"nvrtc_options": ("--split-compile=2",)}
                    if backend == "cuda"
                    else {}
                ),
            },
        )
    ]
    assert result.artifact_path == artifact
    assert result.artifact_id == "artifact-a"
    assert result.as_dict()["available"] is True
    assert result.as_dict()["backend"] == backend
    assert result.as_dict()["persistent_blocks_per_compute_unit"] == 8


def test_installed_device_r1_loads_before_compiler_selection(monkeypatch, tmp_path):
    evaluator = _Evaluator("cuda")
    artifact = tmp_path / "installed-r1.cubin"
    artifact.write_bytes(b"installed-r1")
    metadata = {
        "artifact_id": "generated-r1",
        "generated_identity": {"operator": "R1", "artifact_id": "generated-r1"},
        "abi": "symmetrix.jit.cuda-plugin/4",
        "abi_version": 4,
        "capabilities": 0x3,
        "persistent_blocks_per_compute_unit": 8,
    }
    installed_plan = SimpleNamespace(
        metadata=metadata,
        loader=evaluator.load,
        ready_attribute="execution_cuda_plugin_ready",
        artifact_attribute="execution_cuda_plugin_artifact_id",
        variant_id="variant-installed",
        edge_policy={"persistent_blocks_per_compute_unit": 8},
    )
    monkeypatch.setattr(
        device_artifact,
        "_factorized_device_installed_plan",
        lambda *args, **kwargs: installed_plan,
        raising=False,
    )
    monkeypatch.setattr(
        device_artifact,
        "_factorized_device_backend_plan",
        lambda *args, **kwargs: pytest.fail("compiler selection was attempted"),
    )
    monkeypatch.setattr(
        device_artifact,
        "resolve_installed_operator_artifact",
        lambda **arguments: (
            arguments["loader"](str(artifact))
            or SimpleNamespace(
                status="resolved",
                origin="prebuilt",
                artifact_path=artifact,
                artifact_id="generated-r1",
                diagnostics=("origin=prebuilt status=loaded operator=R1",),
            )
        ),
        raising=False,
    )

    result = device_artifact._prepare_and_validate(
        evaluator=evaluator,
        contract={"fixture": "r1"},
        precision="float32",
        environment=evaluator.execution_device_execution_environment,
        model_path=tmp_path / "model.json",
        model_type="MACE",
        model_data={"execution_contracts": {"R1": {"fixture": "r1"}}},
        include_low_memory_operators=False,
        cache_root=tmp_path / "empty-cache",
    )

    assert result.status == "loaded"
    assert result.compiler == "prebuilt"
    assert result.artifact_path == artifact
    assert evaluator.events == [("load", str(artifact), 8)]
    assert not (tmp_path / "empty-cache").exists()


@pytest.mark.parametrize("include_low_memory_operators", (False, True))
def test_full_installed_device_catalog_respects_operator_request_without_compiler(
    monkeypatch, tmp_path, include_low_memory_operators
):
    evaluator = _Evaluator("cuda")
    model_data = _full_cuda_model_data()
    paths = _install_full_cuda_catalog(monkeypatch, tmp_path, evaluator, model_data)
    monkeypatch.setattr(
        device_artifact,
        "_factorized_device_backend_plan",
        lambda *args, **kwargs: pytest.fail("runtime compiler selection was attempted"),
    )
    monkeypatch.setattr(
        jit_runtime,
        "prepare_nvrtc_jit_artifact",
        lambda *args, **kwargs: pytest.fail("runtime compilation was attempted"),
    )

    result = device_artifact._prepare_and_validate(
        evaluator=evaluator,
        contract=model_data["execution_contracts"]["R1"],
        precision="float32",
        environment=evaluator.execution_device_execution_environment,
        model_path=tmp_path / "model.json",
        model_type="MACE",
        model_data=model_data,
        include_low_memory_operators=include_low_memory_operators,
        cache_root=tmp_path / "empty-cache",
    )

    assert result.origin == "prebuilt"
    assert result.artifact_path == paths["R1"]
    assert tuple(result.operator_modules) == (
        ("M0", "R0") if include_low_memory_operators else ()
    )
    expected_loads = [("load", str(paths["R1"]), 8)]
    if include_low_memory_operators:
        expected_loads.extend(
            (
                ("load-M0", str(paths["M0"]), "chunk32", 8),
                ("load-R0", str(paths["R0"]), "receiver-edge16", 3),
            )
        )
        assert all(
            module.get("artifact_origin", module.get("origin")) == "prebuilt"
            for module in result.operator_modules.values()
        )
    assert evaluator.events == expected_loads
    assert not (tmp_path / "empty-cache").exists()


def test_runtime_device_fallback_retains_catalog_absent_diagnostic(
    monkeypatch, tmp_path
):
    evaluator = _Evaluator("cuda")
    artifact = tmp_path / "runtime-r1.cubin"
    manifest = tmp_path / "runtime-r1.json"
    installed_plan = SimpleNamespace(
        metadata={
            "generated_identity": {
                "operator": "R1",
                "artifact_id": "generated-r1-request",
            },
            "abi": "symmetrix.jit.cuda-plugin/4",
            "abi_version": 4,
            "capabilities": 0x3,
            "persistent_blocks_per_compute_unit": 8,
        },
        loader=evaluator.load,
        ready_attribute="execution_cuda_plugin_ready",
        artifact_attribute="execution_cuda_plugin_artifact_id",
        variant_id="variant-installed",
        edge_policy={"persistent_blocks_per_compute_unit": 8},
    )
    runtime_plan = SimpleNamespace(
        selected_compiler="nvrtc",
        policy="automatic",
        metadata={
            "artifact_id": "artifact-runtime",
            "persistent_blocks_per_compute_unit": 8,
        },
        loader=evaluator.load,
        ready_attribute="execution_cuda_plugin_ready",
        artifact_attribute="execution_cuda_plugin_artifact_id",
        variant_id="variant-runtime",
        edge_policy={"persistent_blocks_per_compute_unit": 8},
        build_attempt=lambda compiler: SimpleNamespace(
            source="runtime R1 source",
            prepare=lambda source, **arguments: SimpleNamespace(
                status="built",
                available=True,
                cache_key="cache-runtime",
                artifact_path=artifact,
                manifest_path=manifest,
                manifest={"key_inputs": {"fixture": "runtime"}},
                diagnostics=("origin=runtime status=compiled",),
                reason=None,
            ),
            arguments={"build": {"compiler": compiler}},
        ),
    )
    monkeypatch.setattr(
        device_artifact.symmetrix,
        "_backend_build_info",
        lambda: {"prebuilt_operator_artifacts": {"mode": "off"}},
    )
    monkeypatch.setattr(
        device_artifact,
        "_factorized_device_installed_plan",
        lambda *args, **kwargs: installed_plan,
    )
    monkeypatch.setattr(
        device_artifact,
        "_factorized_device_backend_plan",
        lambda *args, **kwargs: runtime_plan,
    )
    monkeypatch.setattr(
        device_artifact, "native_source_content_sha256", lambda native: "a" * 64
    )

    result = device_artifact._prepare_and_validate(
        evaluator=evaluator,
        contract={"fixture": "r1"},
        precision="float32",
        environment=evaluator.execution_device_execution_environment,
        model_path=tmp_path / "model.json",
        model_type="MACE",
        model_data={"execution_contracts": {"R1": {"fixture": "r1"}}},
        include_low_memory_operators=False,
        cache_root=tmp_path / "cache",
    )

    assert result.origin == "runtime"
    assert result.diagnostics == (
        "origin=prebuilt status=absent operator=R1 reason=catalog-absent",
        "origin=runtime status=compiled",
    )


def test_cuda_r1_m0_r0_compile_concurrently_then_load_in_order(monkeypatch, tmp_path):
    evaluator = _Evaluator("cuda")
    compile_barrier = threading.Barrier(3)
    compile_threads = set()
    prepare_calls = []

    def prepare(source, **arguments):
        prepare_calls.append((source, arguments))
        compile_threads.add(threading.get_ident())
        compile_barrier.wait(timeout=2)
        name = source.split("-")[0].lower()
        return SimpleNamespace(
            status="built",
            available=True,
            cache_key=f"cache-{name}",
            artifact_path=tmp_path / f"{name}.cubin",
            manifest_path=tmp_path / f"{name}.json",
            manifest={"key_inputs": {"fixture": name}},
            diagnostics=(f"compiled-{name}",),
            reason=None,
        )

    plan = SimpleNamespace(
        selected_compiler="nvrtc",
        policy="automatic",
        metadata={
            "artifact_id": "artifact-r1",
            "persistent_blocks_per_compute_unit": 8,
        },
        loader=evaluator.load,
        ready_attribute="execution_cuda_plugin_ready",
        artifact_attribute="execution_cuda_plugin_artifact_id",
        variant_id="variant-r1",
        edge_policy={"persistent_blocks_per_compute_unit": 8},
        build_attempt=lambda compiler: SimpleNamespace(
            source="R1-source",
            prepare=prepare,
            arguments={"build": {"compiler": compiler}},
        ),
    )
    monkeypatch.setattr(
        device_artifact, "_factorized_device_backend_plan", lambda *args: plan
    )
    monkeypatch.setattr(
        device_artifact,
        "_factorized_device_installed_plan",
        lambda *args, **kwargs: SimpleNamespace(
            metadata={
                "generated_identity": {
                    "operator": "R1",
                    "artifact_id": "generated-r1-request",
                },
                "abi": "symmetrix.jit.cuda-plugin/4",
                "abi_version": 4,
                "capabilities": 0x3,
                "persistent_blocks_per_compute_unit": 8,
            },
            loader=evaluator.load,
            ready_attribute="execution_cuda_plugin_ready",
            artifact_attribute="execution_cuda_plugin_artifact_id",
            variant_id="variant-installed",
            edge_policy={"persistent_blocks_per_compute_unit": 8},
        ),
    )
    monkeypatch.setattr(
        device_artifact,
        "resolve_installed_operator_artifact",
        lambda **kwargs: SimpleNamespace(
            status="absent",
            diagnostics=(
                "origin=prebuilt status=absent operator=R1 reason=catalog-absent",
            ),
        ),
    )
    monkeypatch.setattr(
        device_artifact, "native_source_content_sha256", lambda native: "a" * 64
    )
    monkeypatch.setattr(
        operator_codegen,
        "render_jit_m0_device_module",
        lambda contract, **arguments: (
            "M0-source",
            {
                "schema": "symmetrix.jit.operator-module/1",
                "abi_version": 1,
                "artifact_id": "artifact-m0",
                "structure_fingerprint": "sha256:" + "b" * 64,
                "capabilities": 0x1F,
            },
        ),
    )
    monkeypatch.setattr(
        operator_codegen,
        "render_jit_r0_device_module",
        lambda contract, **arguments: (
            "R0-source",
            {
                "schema": "symmetrix.jit.operator-module/1",
                "abi_version": 1,
                "artifact_id": "artifact-r0",
                "structure_fingerprint": "sha256:" + "c" * 64,
                "capabilities": 0x1F,
            },
        ),
    )
    monkeypatch.setattr(jit_runtime, "prepare_nvrtc_jit_artifact", prepare)

    result = device_artifact._prepare_and_validate(
        evaluator=evaluator,
        contract={"fixture": "r1"},
        precision="float32",
        environment=evaluator.execution_device_execution_environment,
        model_path=tmp_path / "model.json",
        model_type="MACE",
        model_data={
            "execution_contracts": {
                "R1": {"fixture": "r1"},
                "M0": {"fixture": "m0"},
                "R0": {"fixture": "r0"},
            }
        },
        include_low_memory_operators=True,
        cache_root=tmp_path / "cache",
    )

    assert len(compile_threads) == 3
    assert {source for source, _ in prepare_calls} == {
        "R1-source",
        "M0-source",
        "R0-source",
    }
    assert {
        source: arguments["nvrtc_options"] for source, arguments in prepare_calls
    } == {
        "R1-source": ("--split-compile=2",),
        "M0-source": ("--split-compile=2",),
        "R0-source": ("--split-compile=2", "--use_fast_math"),
    }
    assert [event[0] for event in evaluator.events] == [
        "load",
        "load-M0",
        "load-R0",
    ]
    assert tuple(result.operator_modules) == ("M0", "R0")


def test_prepare_device_artifact_rejects_available_openmp(monkeypatch, tmp_path):
    evaluator = _Evaluator("host")
    evaluator.execution_device_execution_environment["execution_space"] = "OpenMP"
    monkeypatch.setattr(
        device_artifact.symmetrix, "_kokkos_is_initialized", lambda: True
    )
    monkeypatch.setattr(
        device_artifact.symmetrix, "MACEKokkosFloat", lambda path: evaluator
    )
    monkeypatch.setattr(
        device_artifact,
        "_factorized_device_backend_plan",
        lambda *args: pytest.fail("OpenMP must not create a device plan"),
    )

    with pytest.raises(
        device_artifact.JitDeviceArtifactError,
        match="require a CUDA or HIP backend",
    ):
        device_artifact.prepare_jit_device_artifact(
            _model(tmp_path), precision="float32"
        )

    assert evaluator.events == [("mode", "factorized")]


def test_prepare_device_artifact_cli_path_only(monkeypatch, capsys, tmp_path):
    artifact = tmp_path / "module.cubin"
    result = SimpleNamespace(artifact_path=artifact)
    calls = []
    monkeypatch.setattr(
        artifact_cli.native_symmetrix, "_kokkos_is_initialized", lambda: False
    )
    monkeypatch.setattr(
        artifact_cli,
        "prepare_jit_device_artifact",
        lambda *args, **kwargs: calls.append(kwargs) or result,
    )

    status = artifact_cli.main(
        ["--model", "model.json", "--precision", "float32", "--path-only"]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert calls[0]["include_low_memory_operators"] is False
    assert captured.out == f"{artifact}\n"
    assert captured.err == ""


def test_prepare_device_artifact_cli_lammps_arguments(monkeypatch, capsys, tmp_path):
    r1_artifact = tmp_path / "r1 module.cubin"
    m0_artifact = tmp_path / "m0.cubin"
    r0_artifact = tmp_path / "r0.cubin"
    result = SimpleNamespace(
        artifact_path=r1_artifact,
        edge_policy={"persistent_blocks_per_compute_unit": 6},
        operator_modules={
            "M0": {
                "implementation": "device_module",
                "artifact_path": str(m0_artifact),
                "schedule": "table",
            },
            "R0": {
                "implementation": "generated",
                "backend": "cuda",
                "artifact_origin": "cache",
                "artifact_path": str(r0_artifact),
            },
        },
    )
    monkeypatch.setattr(
        artifact_cli.native_symmetrix, "_kokkos_is_initialized", lambda: True
    )
    monkeypatch.setattr(
        artifact_cli, "prepare_jit_device_artifact", lambda *args, **kwargs: result
    )

    status = artifact_cli.main(
        ["--model", "model.json", "--precision", "float64", "--lammps-arguments"]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert captured.out == (
        f'jit_device_artifact "{r1_artifact}" '
        "jit_device_blocks_per_compute_unit 6 "
        f"jit_m0_device_artifact {m0_artifact} jit_m0_device_schedule table "
        f"jit_r0_device_artifact {r0_artifact}\n"
    )
    assert captured.err == ""


def test_prepare_device_artifact_cli_lammps_arguments_omits_builtin_operators(
    monkeypatch, capsys, tmp_path
):
    result = SimpleNamespace(
        artifact_path=tmp_path / "r1.cubin",
        edge_policy=None,
        operator_modules={
            "M0": {"implementation": "builtin"},
            "R0": {
                "implementation": "generated",
                "backend": "cuda",
                "artifact_origin": None,
            },
        },
    )
    monkeypatch.setattr(
        artifact_cli.native_symmetrix, "_kokkos_is_initialized", lambda: True
    )
    monkeypatch.setattr(
        artifact_cli, "prepare_jit_device_artifact", lambda *args, **kwargs: result
    )

    status = artifact_cli.main(
        ["--model", "model.json", "--precision", "float32", "--lammps-arguments"]
    )

    assert status == 0
    assert capsys.readouterr().out == (
        f"jit_device_artifact {result.artifact_path} "
        "jit_device_blocks_per_compute_unit 8\n"
    )


def test_prepare_device_artifact_cli_lammps_arguments_rejects_multiple_precisions():
    with pytest.raises(SystemExit, match="2"):
        artifact_cli.main(
            [
                "--model",
                "model.json",
                "--precision",
                "float32",
                "--precision",
                "float64",
                "--lammps-arguments",
            ]
        )


def test_prepare_device_artifact_cli_reports_structured_error(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise device_artifact.JitDeviceArtifactError(
            "compile failed", diagnostics=("nvrtc error",), cache_key="cache-a"
        )

    monkeypatch.setattr(artifact_cli, "prepare_jit_device_artifact", fail)
    monkeypatch.setattr(
        artifact_cli.native_symmetrix, "_kokkos_is_initialized", lambda: False
    )

    status = artifact_cli.main(["--model", "model.json", "--precision", "float64"])

    captured = capsys.readouterr()
    assert status == 1
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "status": "error",
        "available": False,
        "cache_key": "cache-a",
        "reason": "compile failed",
        "diagnostics": ["nvrtc error"],
    }


def test_prepare_device_artifact_cli_finalizes_owned_kokkos(
    monkeypatch, capsys, tmp_path
):
    initialized = iter((False, True))
    finalized = []
    result = SimpleNamespace(artifact_path=tmp_path / "module.cubin")
    monkeypatch.setattr(
        artifact_cli.native_symmetrix,
        "_kokkos_is_initialized",
        lambda: next(initialized),
    )
    monkeypatch.setattr(
        artifact_cli.native_symmetrix, "_kokkos_live_object_count", lambda: 0
    )
    monkeypatch.setattr(
        artifact_cli.native_symmetrix,
        "_finalize_kokkos",
        lambda: finalized.append(True),
    )
    monkeypatch.setattr(
        artifact_cli, "prepare_jit_device_artifact", lambda *args, **kwargs: result
    )

    status = artifact_cli.main(
        ["--model", "model.json", "--precision", "float32", "--path-only"]
    )

    assert status == 0
    assert finalized == [True]
    assert capsys.readouterr().out == f"{result.artifact_path}\n"


def test_prepare_device_artifact_cli_prepares_multiple_precisions_in_one_runtime(
    monkeypatch, capsys, tmp_path
):
    initialized = iter((False, True))
    finalized = []
    prepare_calls = []
    monkeypatch.setattr(
        artifact_cli.native_symmetrix,
        "_kokkos_is_initialized",
        lambda: next(initialized),
    )
    monkeypatch.setattr(
        artifact_cli.native_symmetrix, "_kokkos_live_object_count", lambda: 0
    )
    monkeypatch.setattr(
        artifact_cli.native_symmetrix,
        "_finalize_kokkos",
        lambda: finalized.append(True),
    )

    def prepare(*args, precision, **kwargs):
        prepare_calls.append(precision)
        artifact_path = tmp_path / f"module-{precision}.cubin"
        return SimpleNamespace(
            artifact_path=artifact_path,
            as_dict=lambda: {
                "precision": precision,
                "artifact_path": str(artifact_path),
            },
        )

    monkeypatch.setattr(artifact_cli, "prepare_jit_device_artifact", prepare)

    status = artifact_cli.main(
        [
            "--model",
            "model.json",
            "--precision",
            "float32",
            "--precision",
            "float64",
            "--precision",
            "float32",
        ]
    )

    assert status == 0
    assert prepare_calls == ["float32", "float64"]
    assert finalized == [True]
    assert json.loads(capsys.readouterr().out) == [
        {
            "precision": "float32",
            "artifact_path": str(tmp_path / "module-float32.cubin"),
        },
        {
            "precision": "float64",
            "artifact_path": str(tmp_path / "module-float64.cubin"),
        },
    ]


def test_pair_symmetrix_exposes_explicit_device_artifact_loader():
    repository = Path(__file__).resolve().parents[2]
    source = (repository / "pair_symmetrix/pair_symmetrix_mace_kokkos.cpp").read_text()

    assert 'token == "jit_device_artifact"' in source
    assert 'token == "jit_m0_device_artifact"' in source
    assert 'token == "jit_r0_device_artifact"' in source
    assert 'token == "jit_m0_device_schedule"' in source
    assert 'token == "jit_device_blocks_per_compute_unit"' in source
    assert "consumed != value.size()" in source
    assert "jit_device_artifact, jit_device_blocks_per_compute_unit" in source
    assert "mace->load_m0_device_module(" in source
    assert "mace->load_r0_device_module(" in source
    assert source.index("mace->load_jit_device_plugin(") < source.index(
        "mace->load_m0_device_module("
    )
    assert source.index("mace->load_m0_device_module(") < source.index(
        "mace->set_execution_plan_request("
    )
    assert source.index("mace->load_r0_device_module(") < source.index(
        "mace->set_execution_plan_request("
    )


def test_pair_symmetrix_fixed_workspace_is_explicit_and_fail_closed():
    repository = Path(__file__).resolve().parents[2]
    source = (repository / "pair_symmetrix/pair_symmetrix_mace_kokkos.cpp").read_text()

    assert 'token == "low_memory"' in source
    assert 'token == "profile"' in source
    assert 'token == "allow_fixed_workspace"' in source
    assert "allow_fixed_workspace yes requires streamed_edges direct" in source
    assert (
        '"jit_host_artifact or jit_device_artifact; direct execution does not "'
        in source
    )
    assert "mace->set_allow_fixed_workspace(allow_fixed_workspace)" in source
    assert "mace->set_execution_plan_request(" in source
    assert "\"Symmetrix execution profile='{}'\\n\"" in source


def test_pair_symmetrix_publishes_device_atom_energies_to_lammps():
    repository = Path(__file__).resolve().parents[2]
    source = (repository / "pair_symmetrix/pair_symmetrix_mace_kokkos.cpp").read_text()

    assert source.count("k_eatom.sync_host();") == 3


def test_pair_symmetrix_mpi_stages_only_boundary_packets():
    repository = Path(__file__).resolve().parents[2]
    source = (repository / "pair_symmetrix/pair_symmetrix_mace_kokkos.cpp").read_text()
    header = (repository / "pair_symmetrix/pair_symmetrix_mace_kokkos.h").read_text()

    assert "create_mirror_view_and_copy(Kokkos::HostSpace(), H1)" not in source
    assert "create_mirror_view_and_copy(Kokkos::HostSpace(), H1_adj)" not in source
    assert source.count('_comm_host_staged"') == 4
    assert "Kokkos::View<double*> legacy_comm_packet" in header
    assert "Kokkos::View<int*> legacy_comm_indices" in header
    assert "checked_comm_value_count(std::size_t, const char *)" in header
    assert "std::numeric_limits<int>::max()" in source
    assert 'checked_comm_value_count(1, "per-atom")' in source
    assert "int max_ghost_atoms = 0;" in source
    assert "MPI_Allreduce(" in source
    assert "static_cast<std::size_t>(max_ghost_atoms)," in source
    assert '"per-message ghost upper bound");' in source
    assert 'num_feature_nodes_size, "local feature-state"' not in source
    for counter in (
        "symmetrix_mpi_staged_packet_d2h_bytes",
        "symmetrix_mpi_staged_packet_h2d_bytes",
        "symmetrix_mpi_staged_index_h2d_bytes",
    ):
        assert counter in source
    assert (
        source.count(
            "execution_mpi_staged_packet_d2h_bytes += num_values * sizeof(double);"
        )
        == 2
    )
    assert (
        source.count(
            "execution_mpi_staged_packet_h2d_bytes += num_values * sizeof(double);"
        )
        == 2
    )
    assert (
        source.count(
            "execution_mpi_staged_index_h2d_bytes += num_indices * sizeof(int);"
        )
        == 2
    )

    callback_start = source.index(
        "int PairSymmetrixMACEKokkos<DeviceType, Precision>::pack_forward_comm(int"
    )
    callback_end = source.index(
        "void PairSymmetrixMACEKokkos<DeviceType, Precision>::compute_no_domain_decomposition"
    )
    callbacks = source[callback_start:callback_end]
    assert "Kokkos::fence();" not in callbacks
    assert "return n*num_LM*num_channels;" not in callbacks
    assert "ii*num_LM*num_channels" not in callbacks
    assert "i*num_LM*num_channels" not in callbacks
    for operation in (
        "forward-pack",
        "forward-unpack",
        "reverse-pack",
        "reverse-unpack",
    ):
        assert callbacks.count(f'"{operation}"') == 2


def test_pair_symmetrix_contiguous_comm_uses_copy_or_scalar_conversion():
    repository = Path(__file__).resolve().parents[2]
    source = (repository / "pair_symmetrix/pair_symmetrix_mace_kokkos.cpp").read_text()

    assert "std::is_same_v<Precision, double>" in source
    assert "Kokkos::deep_copy(target, source);" in source
    assert "Kokkos::deep_copy(output, input);" in source
    assert "PairSymmetrixMACEKokkos::unpack_forward_comm_convert" in source
    assert "PairSymmetrixMACEKokkos::pack_reverse_comm_convert" in source

    forward_unpack = source[
        source.index(
            "void PairSymmetrixMACEKokkos<DeviceType, Precision>::unpack_forward_comm_kokkos"
        ) : source.index(
            "int PairSymmetrixMACEKokkos<DeviceType, Precision>::pack_reverse_comm"
        )
    ]
    reverse_pack = source[
        source.index(
            "int PairSymmetrixMACEKokkos<DeviceType, Precision>::pack_reverse_comm_kokkos"
        ) : source.index(
            "void PairSymmetrixMACEKokkos<DeviceType, Precision>::unpack_reverse_comm"
        )
    ]
    assert "MDRangePolicy" not in forward_unpack
    assert "MDRangePolicy" not in reverse_pack


def test_extract_mace_preserves_conversion_without_device_preparation(
    monkeypatch, tmp_path
):
    output_path = tmp_path / "model.json"
    model_data = {"model_type": "MACE", "execution_contracts": {"R1": {}}}
    monkeypatch.setattr(
        extract_cli, "extract_mace_data", lambda *args, **kwargs: model_data
    )
    monkeypatch.setattr(
        extract_cli,
        "prepare_jit_device_artifact",
        lambda *args, **kwargs: pytest.fail("preparation must remain opt-in"),
    )

    status = extract_cli.main(
        ["--model", "checkpoint.model", "--output", str(output_path)]
    )

    assert status == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == model_data


def test_extract_mace_prepares_device_artifact_after_conversion(
    monkeypatch,
    tmp_path,
):
    output_path = tmp_path / "model.json"
    cache_root = tmp_path / "cache"
    model_data = {"model_type": "MACEField", "execution_contracts": {"R1": {}}}
    prepare_calls = []
    monkeypatch.setattr(
        extract_cli, "extract_mace_data", lambda *args, **kwargs: model_data
    )
    monkeypatch.setattr(
        extract_cli,
        "prepare_jit_device_artifact",
        lambda arguments: prepare_calls.append(arguments) or 7,
    )

    status = extract_cli.main(
        [
            "--model",
            "checkpoint.model",
            "--output",
            str(output_path),
            "--prepare-jit-device-artifact",
            "--jit-device-backend",
            "cuda",
            "--jit-device-cache-root",
            str(cache_root),
        ]
    )

    assert status == 7
    assert json.loads(output_path.read_text(encoding="utf-8")) == model_data
    assert prepare_calls == [
        [
            "--model",
            str(output_path),
            "--precision",
            "float32",
            "--precision",
            "float64",
            "--backend",
            "cuda",
            "--cache-root",
            str(cache_root),
        ]
    ]


def test_extract_mace_rejects_device_preparation_for_pair_splines(monkeypatch):
    monkeypatch.setattr(
        extract_cli,
        "extract_mace_data",
        lambda *args, **kwargs: pytest.fail(
            "invalid options must fail before extraction"
        ),
    )

    with pytest.raises(SystemExit, match="2"):
        extract_cli.main(
            [
                "--model",
                "checkpoint.model",
                "--radial-format",
                "pair-splines",
                "--prepare-jit-device-artifact",
            ]
        )
