import gc
import json
import os
from pathlib import Path

import pytest
from model_downloads import (
    MODEL_URLS,
    cached_model_path,
)
from model_downloads import (
    macefield_model_path as cached_macefield_model_path,
)

_REQUIRED_NODES = set()
_REQUIRED_PASSED_CALLS = set()


def _require_macefield():
    try:
        from mace.modules.extensions import MACEField
    except (ImportError, AttributeError):
        pytest.skip(
            "mace-field is required; upstream mace-torch uses the same `mace` "
            "package name but does not provide mace.modules.extensions.MACEField"
        )
    return MACEField


def _native_capabilities():
    try:
        import symmetrix
    except ImportError:
        return {"available": False, "backend": "unavailable"}
    query = getattr(symmetrix, "_execution_device_execution_environment", None)
    if not callable(query):
        return {
            "available": True,
            "backend": "cuda"
            if getattr(symmetrix, "_kokkos_default_execution_space", lambda: "")()
            == "Cuda"
            else "host",
        }
    return dict(query())


def _required_test_manifest():
    manifest_path = os.environ.get("SYMMETRIX_REQUIRED_TEST_MANIFEST")
    if not manifest_path:
        return None
    return json.loads(Path(manifest_path).read_text())


def pytest_sessionstart(session):
    manifest = _required_test_manifest()
    if manifest is None:
        return
    import symmetrix

    if not symmetrix._kokkos_is_initialized():
        symmetrix._init_kokkos()
    capabilities = _native_capabilities()
    validators = {
        "cuda_environment": lambda: (
            capabilities.get("available")
            and capabilities.get("backend") == "cuda"
            and capabilities.get("execution_space") == "Cuda"
            and capabilities.get("device_memory")
            and capabilities.get("compute_capability_code", 0) > 0
        ),
        "hip_environment": lambda: (
            capabilities.get("available")
            and capabilities.get("backend") == "hip"
            and capabilities.get("execution_space") == "HIP"
            and capabilities.get("device_memory")
            and str(capabilities.get("architecture", "")).startswith("gfx")
        ),
        "hip_sentinel": symmetrix._kokkos_device_sentinel,
    }
    required = manifest.get("required_capabilities", ())
    unknown = sorted(set(required) - validators.keys())
    if unknown:
        raise pytest.UsageError(
            "mandatory qualification capabilities have no validator: "
            + ", ".join(unknown)
        )
    failed = [name for name in required if not validators[name]()]
    if failed:
        raise pytest.UsageError(
            "mandatory qualification capabilities failed: " + ", ".join(failed)
        )


@pytest.fixture(scope="session")
def accelerator_capabilities():
    """Normalized host/CUDA/HIP capability record for test selection."""
    import symmetrix

    if not symmetrix._kokkos_is_initialized():
        symmetrix._init_kokkos()
    return _native_capabilities()


def pytest_collection_modifyitems(config, items):
    global _REQUIRED_NODES, _REQUIRED_PASSED_CALLS
    capabilities = _native_capabilities()
    backend = capabilities.get("backend", "unavailable")
    for item in items:
        if "cuda" in item.keywords and backend != "cuda":
            item.add_marker(pytest.mark.skip(reason="requires the CUDA backend"))
        if "hip" in item.keywords and backend != "hip":
            item.add_marker(pytest.mark.skip(reason="requires the HIP backend"))
        if "gpu" in item.keywords and backend not in ("cuda", "hip"):
            item.add_marker(pytest.mark.skip(reason="requires an accelerator backend"))

    manifest = _required_test_manifest()
    if manifest is None:
        return
    required = set(manifest.get("required_nodes", ()))
    _REQUIRED_NODES = required
    _REQUIRED_PASSED_CALLS = set()
    collected = {item.nodeid for item in items}
    missing = sorted(required - collected)
    if missing:
        raise pytest.UsageError(
            "mandatory qualification nodes were not collected: " + ", ".join(missing)
        )


def pytest_runtest_logreport(report):
    manifest = _required_test_manifest()
    if manifest is None:
        return
    required = set(manifest.get("required_nodes", ()))
    if report.nodeid not in required:
        return
    if (
        report.when == "call"
        and report.passed
        and not getattr(report, "wasxfail", False)
    ):
        _REQUIRED_PASSED_CALLS.add(report.nodeid)


def pytest_sessionfinish(session, exitstatus):
    missing = sorted(_REQUIRED_NODES - _REQUIRED_PASSED_CALLS)
    if missing:
        terminal = session.config.pluginmanager.get_plugin("terminalreporter")
        if terminal is not None:
            terminal.write_line(
                "mandatory qualification nodes did not pass normally: "
                + ", ".join(missing),
                red=True,
            )
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture(scope="session", autouse=True)
def finalize_kokkos_after_tests():
    yield

    import symmetrix

    gc.collect()
    if symmetrix._kokkos_is_initialized():
        symmetrix._finalize_kokkos()


@pytest.fixture(scope="session")
def model_cache():
    models = {}
    for filename, url in MODEL_URLS.items():
        if not filename.endswith(".json"):
            continue
        try:
            models[filename] = cached_model_path(filename, url)
        except RuntimeError as exc:
            pytest.skip(str(exc))
    return models


@pytest.fixture(scope="session")
def macefield_model_path():
    _require_macefield()
    try:
        return cached_macefield_model_path()
    except (FileNotFoundError, RuntimeError) as exc:
        pytest.skip(str(exc))
