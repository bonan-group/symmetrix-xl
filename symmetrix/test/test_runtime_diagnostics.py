import ctypes
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from symmetrix.cli import main as cli
from symmetrix.runtime_diagnostics import (
    accelerator_runtime_diagnostics,
    openmp_runtime_diagnostics,
)


def _native(runtime):
    return SimpleNamespace(_openmp_runtime_info=lambda: runtime)


def _runtime(expected_path, loaded_path, libraries=None):
    return {
        "enabled": True,
        "compiled_openmp": 201511,
        "compiler_id": "GNU",
        "compiler_version": "12.3.0",
        "expected_runtime_path": str(expected_path),
        "expected_runtime_sha256": None,
        "loaded_runtime_path": str(loaded_path),
        "loaded_openmp_libraries": list(libraries or [loaded_path]),
        "kokkos_execution_space": "OpenMP",
        "kokkos_initialized": False,
        "kokkos_concurrency": None,
        "omp_max_threads": 8,
        "omp_max_active_levels": 1,
        "worker_sentinel": {
            "actual_team_size": 8,
            "cpu_observation_supported": True,
            "unique_current_cpus": list(range(8)),
            "workers": [
                {"thread_id": thread, "current_cpu": thread, "affinity": [thread]}
                for thread in range(8)
            ],
        },
        "host_blas": {
            "environment_policy": "automatic",
            "selected_backend": "cblas",
            "dense_environment_policy": "automatic",
            "dense_selected_backend": "cblas",
            "library_path": "/lib/libopenblas.so",
            "openblas_config": "OpenBLAS fixture",
            "openblas_parallel": 2,
            "openblas_openmp_enabled": True,
            "openblas_openmp_runtime_compatible": True,
        },
    }


def test_matching_runtime_hash_passes(tmp_path):
    runtime_path = tmp_path / "libgomp.so.1"
    runtime_path.write_bytes(b"qualified-runtime")
    runtime = _runtime(runtime_path, runtime_path)
    from symmetrix import runtime_diagnostics

    runtime["expected_runtime_sha256"] = runtime_diagnostics._sha256(runtime_path)

    report = openmp_runtime_diagnostics(native_module=_native(runtime))

    assert report["ok"]
    assert report["status"] == "ok"
    assert report["runtime"]["loaded_runtime_family"] == "libgomp"


@pytest.mark.parametrize(
    ("provider", "threading", "runtime_compatible", "selected", "expected_ok"),
    (
        ("mkl", "sequential", None, "cblas", True),
        ("mkl", "gnu_openmp", True, "cblas", True),
        ("mkl", "gnu_openmp", False, "cblas", False),
        ("mkl", "intel_openmp", False, "kokkos", True),
        ("unknown", "unknown", None, "kokkos", True),
    ),
)
def test_mkl_blas_routing_is_reported(
    tmp_path, provider, threading, runtime_compatible, selected, expected_ok
):
    runtime_path = tmp_path / "libgomp.so.1"
    runtime_path.write_bytes(b"qualified-runtime")
    runtime = _runtime(runtime_path, runtime_path)
    runtime["host_blas"] = {
        "environment_policy": "automatic",
        "selected_backend": selected,
        "dense_environment_policy": "automatic",
        "dense_selected_backend": "flat",
        "provider": provider,
        "threading": threading,
        "openmp_runtime_compatible": runtime_compatible,
    }
    from symmetrix import runtime_diagnostics

    runtime["expected_runtime_sha256"] = runtime_diagnostics._sha256(runtime_path)
    report = openmp_runtime_diagnostics(native_module=_native(runtime))

    assert report["ok"] is expected_ok
    assert report["runtime"]["host_blas"]["provider"] == provider
    assert report["runtime"]["host_blas"]["threading"] == threading


def test_relocated_runtime_with_matching_hash_passes(tmp_path):
    expected_path = tmp_path / "build" / "libgomp.so.1"
    loaded_path = tmp_path / "runtime" / "libgomp.so.1"
    expected_path.parent.mkdir()
    loaded_path.parent.mkdir()
    expected_path.write_bytes(b"same-runtime")
    loaded_path.write_bytes(b"same-runtime")
    runtime = _runtime(expected_path, loaded_path)
    from symmetrix import runtime_diagnostics

    runtime["expected_runtime_sha256"] = runtime_diagnostics._sha256(expected_path)

    report = openmp_runtime_diagnostics(native_module=_native(runtime))

    assert report["ok"]
    runtime_hash = next(
        check for check in report["checks"] if check["name"] == "runtime_hash"
    )
    assert "relocated path" in runtime_hash["message"]


def test_auditwheel_vendored_runtime_content_hash_passes(tmp_path):
    expected_path = tmp_path / "build" / "libgomp.so.1"
    expected_path.parent.mkdir()
    expected_path.write_bytes(b"original-runtime")
    from symmetrix import runtime_diagnostics

    expected_hash = runtime_diagnostics._sha256(expected_path)
    loaded_path = tmp_path / f"libgomp-{expected_hash[:8]}.so.1.0.0"
    loaded_path.write_bytes(b"auditwheel-patched-runtime")
    runtime = _runtime(expected_path, loaded_path)
    runtime["expected_runtime_sha256"] = expected_hash

    report = openmp_runtime_diagnostics(native_module=_native(runtime))

    assert report["ok"]
    runtime_hash = next(
        check for check in report["checks"] if check["name"] == "runtime_hash"
    )
    assert "auditwheel content hash" in runtime_hash["message"]


@pytest.mark.parametrize(
    "strict, expected_ok, expected_status",
    ((True, False, "error"), (False, True, "warning")),
)
def test_runtime_hash_mismatch_respects_advisory_mode(
    tmp_path, strict, expected_ok, expected_status
):
    expected_path = tmp_path / "expected" / "libgomp.so.1"
    loaded_path = tmp_path / "loaded" / "libgomp.so.1"
    expected_path.parent.mkdir()
    loaded_path.parent.mkdir()
    expected_path.write_bytes(b"expected")
    loaded_path.write_bytes(b"different")
    runtime = _runtime(expected_path, loaded_path)
    from symmetrix import runtime_diagnostics

    runtime["expected_runtime_sha256"] = runtime_diagnostics._sha256(expected_path)

    report = openmp_runtime_diagnostics(native_module=_native(runtime), strict=strict)

    assert report["ok"] is expected_ok
    assert report["status"] == expected_status


def test_duplicate_runtime_is_always_an_error(tmp_path):
    gomp = tmp_path / "libgomp.so.1"
    omp = tmp_path / "libgomp-a34b3233.so.1"
    gomp.write_bytes(b"gomp")
    omp.write_bytes(b"omp")
    runtime = _runtime(gomp, gomp, libraries=[gomp, omp])
    from symmetrix import runtime_diagnostics

    runtime["expected_runtime_sha256"] = runtime_diagnostics._sha256(gomp)

    report = openmp_runtime_diagnostics(native_module=_native(runtime), strict=False)

    assert not report["ok"]
    assert report["status"] == "error"
    duplicate = next(
        check for check in report["checks"] if check["name"] == "duplicate_runtimes"
    )
    assert duplicate["status"] == "error"


def test_worker_sentinel_rejects_effective_single_thread(tmp_path):
    runtime_path = tmp_path / "libgomp.so.1"
    runtime_path.write_bytes(b"qualified-runtime")
    runtime = _runtime(runtime_path, runtime_path)
    runtime["worker_sentinel"] = {
        "actual_team_size": 1,
        "unique_current_cpus": [0],
        "workers": [{"thread_id": 0, "current_cpu": 0, "affinity": [0]}],
    }
    from symmetrix import runtime_diagnostics

    runtime["expected_runtime_sha256"] = runtime_diagnostics._sha256(runtime_path)
    report = openmp_runtime_diagnostics(
        native_module=_native(runtime), expected_threads=8
    )

    assert not report["ok"]
    team = next(
        check for check in report["checks"] if check["name"] == "worker_team_size"
    )
    assert team["status"] == "error"


def test_worker_sentinel_validates_allocated_cpu_coverage(tmp_path):
    runtime_path = tmp_path / "libgomp.so.1"
    runtime_path.write_bytes(b"qualified-runtime")
    runtime = _runtime(runtime_path, runtime_path)
    from symmetrix import runtime_diagnostics

    runtime["expected_runtime_sha256"] = runtime_diagnostics._sha256(runtime_path)
    report = openmp_runtime_diagnostics(
        native_module=_native(runtime),
        expected_threads=8,
        allocated_cpus=range(8),
    )

    assert report["ok"]
    coverage = next(
        check for check in report["checks"] if check["name"] == "allocated_cpu_coverage"
    )
    assert coverage["status"] == "ok"


def test_worker_sentinel_rejects_duplicate_cpu_participation(tmp_path):
    runtime_path = tmp_path / "libgomp.so.1"
    runtime_path.write_bytes(b"qualified-runtime")
    runtime = _runtime(runtime_path, runtime_path)
    runtime["worker_sentinel"]["unique_current_cpus"] = [0, 1, 2, 3]
    from symmetrix import runtime_diagnostics

    runtime["expected_runtime_sha256"] = runtime_diagnostics._sha256(runtime_path)
    report = openmp_runtime_diagnostics(native_module=_native(runtime))

    assert not report["ok"]
    participation = next(
        check
        for check in report["checks"]
        if check["name"] == "worker_cpu_participation"
    )
    assert participation["status"] == "error"


def test_worker_sentinel_allows_duplicate_samples_for_unbound_team(tmp_path):
    runtime_path = tmp_path / "libgomp.so.1"
    runtime_path.write_bytes(b"qualified-runtime")
    runtime = _runtime(runtime_path, runtime_path)
    runtime["worker_sentinel"]["unique_current_cpus"] = [0, 1, 2, 3]
    for worker in runtime["worker_sentinel"]["workers"]:
        worker["affinity"] = list(range(8))
    from symmetrix import runtime_diagnostics

    runtime["expected_runtime_sha256"] = runtime_diagnostics._sha256(runtime_path)
    report = openmp_runtime_diagnostics(native_module=_native(runtime))

    assert report["ok"]
    participation = next(
        check
        for check in report["checks"]
        if check["name"] == "worker_cpu_participation"
    )
    assert participation == {
        "name": "worker_cpu_participation",
        "status": "not_applicable",
        "message": (
            "workers are not bound to disjoint CPU sets; instantaneous CPU "
            "samples do not measure worker participation"
        ),
    }


def test_worker_sentinel_accepts_automatic_openmp_multithread_blas(tmp_path):
    runtime_path = tmp_path / "libgomp.so.1"
    runtime_path.write_bytes(b"qualified-runtime")
    runtime = _runtime(runtime_path, runtime_path)
    runtime["host_blas"]["selected_backend"] = "cblas"
    runtime["host_blas"]["dense_selected_backend"] = "cblas"
    from symmetrix import runtime_diagnostics

    runtime["expected_runtime_sha256"] = runtime_diagnostics._sha256(runtime_path)
    report = openmp_runtime_diagnostics(native_module=_native(runtime))

    assert report["ok"]
    blas = next(
        check for check in report["checks"] if check["name"] == "host_worker_blas"
    )
    assert blas["status"] == "ok"
    assert "OpenMP-enabled OpenBLAS" in blas["message"]


def test_worker_sentinel_rejects_non_openmp_multithread_blas(tmp_path):
    runtime_path = tmp_path / "libgomp.so.1"
    runtime_path.write_bytes(b"qualified-runtime")
    runtime = _runtime(runtime_path, runtime_path)
    runtime["host_blas"]["environment_policy"] = "unsafe"
    runtime["host_blas"]["selected_backend"] = "cblas"
    runtime["host_blas"]["dense_selected_backend"] = "cblas"
    runtime["host_blas"]["openblas_parallel"] = 1
    runtime["host_blas"]["openblas_openmp_enabled"] = False
    from symmetrix import runtime_diagnostics

    runtime["expected_runtime_sha256"] = runtime_diagnostics._sha256(runtime_path)
    report = openmp_runtime_diagnostics(native_module=_native(runtime))

    assert not report["ok"]
    blas = next(
        check for check in report["checks"] if check["name"] == "host_worker_blas"
    )
    assert blas["status"] == "error"


def test_worker_sentinel_rejects_mismatched_openmp_blas_runtime(tmp_path):
    runtime_path = tmp_path / "libgomp.so.1"
    runtime_path.write_bytes(b"qualified-runtime")
    runtime = _runtime(runtime_path, runtime_path)
    runtime["host_blas"]["selected_backend"] = "kokkos"
    runtime["host_blas"]["dense_selected_backend"] = "flat"
    runtime["host_blas"]["openblas_openmp_runtime_compatible"] = False
    from symmetrix import runtime_diagnostics

    runtime["expected_runtime_sha256"] = runtime_diagnostics._sha256(runtime_path)
    report = openmp_runtime_diagnostics(native_module=_native(runtime))

    assert not report["ok"]
    blas_build = next(
        check for check in report["checks"] if check["name"] == "openblas_build"
    )
    assert blas_build["status"] == "error"
    assert "OpenMP runtime" in blas_build["message"]


def test_worker_sentinel_rejects_unsafe_openmp_blas_override(tmp_path):
    runtime_path = tmp_path / "libgomp.so.1"
    runtime_path.write_bytes(b"qualified-runtime")
    runtime = _runtime(runtime_path, runtime_path)
    runtime["host_blas"]["environment_policy"] = "unsafe"
    from symmetrix import runtime_diagnostics

    runtime["expected_runtime_sha256"] = runtime_diagnostics._sha256(runtime_path)
    report = openmp_runtime_diagnostics(native_module=_native(runtime))

    assert not report["ok"]
    blas = next(
        check for check in report["checks"] if check["name"] == "host_worker_blas"
    )
    assert blas["status"] == "error"
    assert "unsafe" in blas["message"]


def test_worker_sentinel_rejects_nested_openmp_multithread_blas(tmp_path):
    runtime_path = tmp_path / "libgomp.so.1"
    runtime_path.write_bytes(b"qualified-runtime")
    runtime = _runtime(runtime_path, runtime_path)
    runtime["omp_max_active_levels"] = 2
    from symmetrix import runtime_diagnostics

    runtime["expected_runtime_sha256"] = runtime_diagnostics._sha256(runtime_path)
    report = openmp_runtime_diagnostics(native_module=_native(runtime))

    assert not report["ok"]
    blas = next(
        check for check in report["checks"] if check["name"] == "host_worker_blas"
    )
    assert blas["status"] == "error"
    assert "nested OpenMP" in blas["message"]


@pytest.mark.parametrize(
    ("parallel_kind", "expected_status"),
    ((1, "warning"), (2, "ok")),
)
def test_openblas_threading_build_is_reported(tmp_path, parallel_kind, expected_status):
    runtime_path = tmp_path / "libgomp.so.1"
    runtime_path.write_bytes(b"qualified-runtime")
    runtime = _runtime(runtime_path, runtime_path)
    runtime["host_blas"]["openblas_parallel"] = parallel_kind
    runtime["host_blas"]["openblas_openmp_enabled"] = parallel_kind == 2
    from symmetrix import runtime_diagnostics

    runtime["expected_runtime_sha256"] = runtime_diagnostics._sha256(runtime_path)
    report = openmp_runtime_diagnostics(native_module=_native(runtime))

    check = next(
        check for check in report["checks"] if check["name"] == "openblas_build"
    )
    assert check["status"] == expected_status
    assert report["runtime"]["host_blas"]["openblas_openmp_enabled"] is (
        parallel_kind == 2
    )
    if parallel_kind == 1:
        assert "multithreaded CPU Kokkos job" in check["message"]


def test_worker_sentinel_allows_unavailable_cpu_observation_without_allocation(
    tmp_path,
):
    runtime_path = tmp_path / "libgomp.so.1"
    runtime_path.write_bytes(b"qualified-runtime")
    runtime = _runtime(runtime_path, runtime_path)
    runtime["worker_sentinel"]["cpu_observation_supported"] = False
    runtime["worker_sentinel"]["unique_current_cpus"] = []
    for worker in runtime["worker_sentinel"]["workers"]:
        worker["current_cpu"] = -1
        worker["affinity"] = []
    from symmetrix import runtime_diagnostics

    runtime["expected_runtime_sha256"] = runtime_diagnostics._sha256(runtime_path)
    report = openmp_runtime_diagnostics(native_module=_native(runtime))

    assert report["ok"]
    participation = next(
        check
        for check in report["checks"]
        if check["name"] == "worker_cpu_participation"
    )
    assert participation["status"] == "not_applicable"


def test_non_openmp_build_is_not_applicable():
    report = openmp_runtime_diagnostics(
        native_module=_native({"enabled": False}), strict=True
    )

    assert report["ok"]
    assert report["status"] == "not_applicable"


def test_missing_native_api_is_an_error():
    report = openmp_runtime_diagnostics(native_module=SimpleNamespace())

    assert not report["ok"]
    assert report["checks"][0]["name"] == "native_api"


def test_doctor_json_exit_status(monkeypatch, capsys):
    import symmetrix

    native = SimpleNamespace(__file__="/native.so", _backend_build_info=lambda: {})
    monkeypatch.setattr(symmetrix, "load_backend", lambda *_args: native)
    monkeypatch.setattr(symmetrix, "selected_backend", lambda: {"backend": "cpu"})
    monkeypatch.setattr(
        cli,
        "openmp_runtime_diagnostics",
        lambda strict: {
            "status": "error",
            "ok": False,
            "strict": strict,
            "python_executable": "/python",
            "checks": [],
            "runtime": None,
        },
    )

    assert cli.main(["doctor", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["strict"] is True


def test_backend_show_accepts_an_explicit_selector(monkeypatch, capsys):
    from symmetrix import backend_loader

    selected = {}

    def select(_version, request=None):
        selected["request"] = request
        return SimpleNamespace(
            to_dict=lambda: {
                "selector": request,
                "backend": "cuda",
                "architecture": "sm80",
                "distribution": "symmetrix-xl-cuda13-sm80",
            }
        )

    monkeypatch.setattr(
        backend_loader,
        "select_backend",
        select,
    )

    assert cli.main(["backend", "show", "cuda13-sm80"]) == 0
    assert selected["request"] == "cuda13-sm80"
    assert "cuda13-sm80: cuda sm80" in capsys.readouterr().out


def test_backend_install_prefers_uv_and_reports_resolution(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        cli,
        "_automatic_backend_resolution",
        lambda: (
            "cuda13-sm120",
            {
                "visible": "CUDA architectures: sm120",
                "toolkit": "CUDA major: 13 (source: nvidia-smi CUDA Version)",
            },
        ),
    )
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/uv")

    def fake_run(command, check, **kwargs):
        calls.append((command, check, kwargs))
        if kwargs.get("capture_output"):
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return None

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli.main(["backend", "install"]) == 0
    command, check, _ = calls[-1]
    assert command[:4] == [
        "/usr/bin/uv",
        "pip",
        "install",
        "--python",
    ]
    assert command[4] == cli.sys.executable
    assert command[-1] == "symmetrix-xl-cuda13-sm120"
    assert check is True
    output = capsys.readouterr().out
    assert "CUDA architectures: sm120" in output
    assert "CUDA major: 13 (source: nvidia-smi CUDA Version)" in output
    assert "selected selector: cuda13-sm120" in output


def test_backend_install_auto_resolves_visible_cuda(monkeypatch):
    from symmetrix import backend_loader

    monkeypatch.setattr(
        backend_loader,
        "_visible_targets",
        lambda: {"cuda": {"sm120"}, "hip": set()},
    )
    monkeypatch.setattr(cli, "_toolkit_resolution", lambda kind: ("13", "test"))

    assert cli._automatic_backend_selector() == "cuda13-sm120"


def test_cuda_toolkit_resolution_prefers_nvidia_smi(monkeypatch):
    calls = []
    monkeypatch.delenv("SYMMETRIX_CUDA_MAJOR", raising=False)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/local/cuda/bin/nvcc")

    def fake_run(command, **kwargs):
        calls.append(command)
        if command == ["nvidia-smi"]:
            return SimpleNamespace(returncode=0, stdout="CUDA Version: 13.0", stderr="")
        raise AssertionError(f"unexpected fallback command: {command}")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli._toolkit_resolution("cuda") == ("13", "nvidia-smi CUDA Version")
    assert calls == [["nvidia-smi"]]


def test_backend_install_falls_back_to_python_pip(monkeypatch):
    calls = []
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)

    def fake_run(command, check, **kwargs):
        calls.append((command, check, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli.main(["backend", "install", "--arch", "cuda12-sm80"]) == 0
    command, check, _ = calls[-1]
    assert command[:4] == [
        cli.sys.executable,
        "-m",
        "pip",
        "install",
    ]
    assert "--verbose" not in command
    assert command[-1] == "symmetrix-xl-cuda12-sm80"
    assert check is True


def test_backend_install_without_visible_gpu_explains_cpu_fallback(monkeypatch, capsys):
    from symmetrix import backend_loader

    monkeypatch.setattr(
        backend_loader,
        "_visible_targets",
        lambda: {"cuda": set(), "hip": set()},
    )

    assert cli.main(["backend", "install"]) == 0
    output = capsys.readouterr().out
    assert "bundled CPU backend is already available" in output
    assert "No backend download is required" in output
    assert "--arch cuda13-sm120" in output


def test_backend_install_auto_falls_back_to_cuda12_when_needed(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        cli,
        "_automatic_backend_resolution",
        lambda: (
            "cuda13-sm120",
            {"visible": "CUDA architectures: sm120", "toolkit": "CUDA major: 13"},
        ),
    )
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/uv")

    def fake_run(command, check, **kwargs):
        calls.append((command, check, kwargs))
        if kwargs.get("capture_output"):
            available = command[-1] == "symmetrix-xl-cuda12-sm120"
            return SimpleNamespace(returncode=int(not available), stdout="", stderr="")
        return None

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli.main(["backend", "install"]) == 0
    command, _, _ = calls[-1]
    assert command[-1] == "symmetrix-xl-cuda12-sm120"
    assert "using published fallback cuda12-sm120" in capsys.readouterr().out


def test_backend_install_reports_source_build_when_no_wheel_exists(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "_automatic_backend_resolution",
        lambda: ("cuda13-sm120", {"visible": "CUDA architectures: sm120"}),
    )
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda command, check, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="no matching distribution"
        ),
    )

    assert cli.main(["backend", "install"]) == 1
    error = capsys.readouterr().err
    assert "no pre-compiled backend wheel" in error
    assert "python tools/symmetrix_build.py install --backend cuda" in error


def test_backend_install_rejects_untrusted_selector(monkeypatch, capsys):
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/uv")

    assert cli.main(["backend", "install", "--arch", "cuda13-sm120;echo"]) == 1
    assert "invalid backend selector" in capsys.readouterr().err


def test_backend_show_probe_reports_openblas_runtime(monkeypatch, capsys):
    from symmetrix import backend_loader

    selected = {
        "selector": "cpu",
        "backend": "cpu",
        "architecture": "native",
        "distribution": "symmetrix",
    }

    monkeypatch.setattr(
        backend_loader,
        "select_backend",
        lambda _version, request=None: SimpleNamespace(
            to_dict=lambda: {**selected, "selector": request or "cpu"}
        ),
    )
    monkeypatch.setattr(
        "symmetrix.load_backend",
        lambda request=None: SimpleNamespace(
            _openmp_runtime_info=lambda: {
                "omp_max_active_levels": 1,
                "host_blas": {
                    "openblas_config": "OpenBLAS fixture USE_OPENMP",
                    "openblas_openmp_enabled": True,
                    "openblas_openmp_runtime_compatible": True,
                },
            }
        ),
    )

    assert cli.main(["backend", "show", "--probe"]) == 0
    human = capsys.readouterr().out
    assert "OpenBLAS OpenMP runtime compatible: True" in human
    assert "OpenMP maximum active levels: 1" in human

    assert cli.main(["backend", "show", "--probe", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["runtime"]["host_blas"]["openblas_openmp_enabled"] is True


def test_doctor_human_report_identifies_selected_non_openmp_backend(capsys):
    cli._human_report(
        {
            "status": "not_applicable",
            "python_executable": "/python",
            "checks": [],
            "runtime": {"kokkos_execution_space": "Cuda"},
            "backend": {
                "selector": "cuda13-sm120",
                "backend": "cuda",
                "architecture": "sm120",
            },
        }
    )

    output = capsys.readouterr().out
    assert "Selected backend: cuda13-sm120 (cuda sm120)" in output
    assert "OpenMP runtime diagnostic: not_applicable" in output


def _accelerator_native(*, environment, sentinel=True):
    state = {"initialized": False, "finalized": False}

    def initialize():
        state["initialized"] = True

    def finalize():
        state["finalized"] = True

    native = SimpleNamespace(
        _kokkos_is_initialized=lambda: state["initialized"],
        _init_kokkos=initialize,
        _finalize_kokkos=finalize,
        _kokkos_default_execution_space=lambda: "Cuda",
        _execution_device_execution_environment=lambda: environment,
        _kokkos_device_sentinel=lambda: sentinel,
    )
    return native, state


def test_accelerator_doctor_initializes_probes_and_finalizes_cuda():
    native, state = _accelerator_native(
        environment={"backend": "cuda", "architecture": "sm120"}
    )
    report = accelerator_runtime_diagnostics(
        native_module=native,
        backend={
            "selector": "cuda13-sm120",
            "backend": "cuda",
            "architecture": "sm120",
        },
    )

    assert report["ok"]
    assert report["runtime"]["device_sentinel"] is True
    assert state == {"initialized": True, "finalized": True}


def test_accelerator_doctor_rejects_wrong_device_architecture():
    native, _ = _accelerator_native(
        environment={"backend": "cuda", "architecture": "sm90"}
    )
    report = accelerator_runtime_diagnostics(
        native_module=native,
        backend={
            "selector": "cuda13-sm120",
            "backend": "cuda",
            "architecture": "sm120",
        },
    )

    assert not report["ok"]
    assert any(check["name"] == "device_architecture" for check in report["checks"])


def test_accelerator_doctor_human_report_is_readable(capsys):
    cli._human_report(
        {
            "schema": "symmetrix.accelerator-runtime-diagnostics",
            "status": "ok",
            "python_executable": "/python",
            "checks": [{"name": "device_sentinel", "status": "ok", "message": "done"}],
            "backend": {
                "selector": "cuda13-sm120",
                "backend": "cuda",
                "architecture": "sm120",
            },
            "native": {
                "extension": "/package/_native_cuda13_sm120.so",
                "build_info": {"compiler_id": "GNU", "compiler_version": "15"},
            },
            "runtime": {
                "execution_space": "Cuda",
                "device_environment": {"backend": "cuda", "architecture": "sm120"},
            },
        }
    )

    output = capsys.readouterr().out
    assert "Symmetrix accelerator diagnostic: ok" in output
    assert "Device: cuda sm120" in output
    assert "[OK] device_sentinel: done" in output


def test_native_openmp_runtime_report_shape():
    import symmetrix

    query = getattr(symmetrix, "_openmp_runtime_info", None)
    assert callable(query), "the native extension lacks OpenMP runtime diagnostics"
    report = dict(query())
    assert report["schema"] == "symmetrix.openmp-runtime-info"
    assert report["version"] == 1
    assert isinstance(report["loaded_openmp_libraries"], list)
    if report["enabled"]:
        assert report["compiled_openmp"] is not None
        assert report["expected_runtime_path"]
        assert len(report["expected_runtime_sha256"]) == 64
        assert report["loaded_runtime_path"]
        sentinel = report["worker_sentinel"]
        assert sentinel["actual_team_size"] > 0
        assert len(sentinel["workers"]) == sentinel["actual_team_size"]
        assert isinstance(sentinel["unique_current_cpus"], list)
        assert report["host_blas"]["selected_backend"] in ("cblas", "kokkos")
        assert report["host_blas"]["provider"] in ("openblas", "mkl", "unknown")
        assert isinstance(report["host_blas"]["threading"], str)


def test_native_concurrent_host_blas_stress_shape():
    import symmetrix

    query = getattr(symmetrix, "_concurrent_host_blas_stress", None)
    assert callable(query), "the native extension lacks the host BLAS stress probe"
    report = dict(query(2, 2, 8))
    assert report["workers"] == 2
    assert report["iterations"] == 2
    assert report["matrix_size"] == 8
    assert report["finite"]
    assert report["max_abs_error"] < 1e-12
    blas = report["blas"]
    if blas["provider"] == "openblas":
        assert blas["openblas_config"]
    elif blas["provider"] == "mkl":
        assert blas["threading"] in ("sequential", "gnu_openmp", "intel_openmp")


def test_native_mkl_gnu_route_rechecks_late_openmp_runtime():
    import symmetrix

    query = getattr(symmetrix, "_openmp_runtime_info", None)
    assert callable(query), "the native extension lacks OpenMP runtime diagnostics"
    before = dict(query())
    blas = before.get("host_blas", {})
    if blas.get("provider") != "mkl" or blas.get("threading") != "gnu_openmp":
        pytest.skip("requires a GNU-threaded MKL extension")

    configured = os.environ.get("SYMMETRIX_TEST_LIBIOMP_PATH")
    candidates = [
        Path(configured) if configured else None,
        Path("/opt/intel/oneapi/compiler/latest/lib/libiomp5.so"),
    ]
    libiomp = next(
        (candidate for candidate in candidates if candidate and candidate.is_file()),
        None,
    )
    if libiomp is None:
        pytest.skip("requires libiomp5 to exercise late duplicate-runtime loading")

    assert blas["openmp_runtime_compatible"] is True
    assert blas["selected_backend"] == "cblas"
    ctypes.CDLL(libiomp, mode=ctypes.RTLD_LOCAL)

    after = dict(query())
    assert len(after["loaded_openmp_libraries"]) == 2
    assert after["host_blas"]["openmp_runtime_compatible"] is False
    assert after["host_blas"]["selected_backend"] == "kokkos"
