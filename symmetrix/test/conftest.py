import gc
import json
import os
from pathlib import Path

import pytest
from compact_r1_model import foundation_r1_test_model
from model_downloads import MODEL_URLS, cached_model_path
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
    backend_markers = {"cuda", "hip", "gpu", "host", "oam_runtime"}
    if any(backend_markers.intersection(item.keywords) for item in items):
        capabilities = _native_capabilities()
        backend = capabilities.get("backend", "unavailable")
        for item in items:
            if "cuda" in item.keywords and backend != "cuda":
                item.add_marker(pytest.mark.skip(reason="requires the CUDA backend"))
            if "hip" in item.keywords and backend != "hip":
                item.add_marker(pytest.mark.skip(reason="requires the HIP backend"))
            if "gpu" in item.keywords and backend not in ("cuda", "hip"):
                item.add_marker(
                    pytest.mark.skip(reason="requires an accelerator backend")
                )
            if "host" in item.keywords and backend != "host":
                item.add_marker(pytest.mark.skip(reason="requires a host backend"))
            if "oam_runtime" in item.keywords and backend == "hip":
                item.add_marker(
                    pytest.mark.skip(
                        reason="OAM execution is unsupported on the HIP backend"
                    )
                )

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

    if symmetrix.selected_backend() is None:
        return
    native = symmetrix.load_backend()
    gc.collect()
    if native._kokkos_is_initialized():
        native._finalize_kokkos()


@pytest.fixture(scope="session")
def generated_standard_mace_zbl_model(tmp_path_factory):
    contract_path = (
        Path(__file__).resolve().parent
        / "data"
        / "execution_contracts"
        / "jit_r1_fixture_c4_e3_l0_contract.json"
    )
    model = foundation_r1_test_model(
        json.loads(contract_path.read_text()), atomic_numbers=(1, 8)
    )
    model.update(
        {
            "has_zbl": True,
            "zbl_a_exp": 0.3,
            "zbl_a_prefactor": 0.4543,
            "zbl_c": [0.1818, 0.5099, 0.2802, 0.02817],
            "zbl_covalent_radii": [1.0] * 119,
            "zbl_p": 6,
        }
    )
    # Keep the integration fixture's energy purely ZBL so its expected value
    # is independent of the generated MACE weights.
    model["readout_2_weights_1"] = [0.0] * len(model["readout_2_weights_1"])
    model["readout_2_weights_2"] = [0.0] * len(model["readout_2_weights_2"])
    output = tmp_path_factory.mktemp("standard-mace-zbl") / "model.json"
    output.write_text(json.dumps(model, separators=(",", ":")))
    return output


@pytest.fixture(scope="session")
def model_cache():
    """The cached fixture models, keyed by filename.

    The two `.json` entries are the original Symmetrix pair-spline format (named
    v1 here), which is retired: both routings stream edges and a v1 model cannot.
    They remain useful as the retired-format negative control; a test that needs
    a working MACE calculator uses ``compact_mace_model`` instead.
    """

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
def compact_mace_model(tmp_path_factory):
    """A compact MH-0 MACE model exported from the cached foundation checkpoint.

    The original Symmetrix pair-spline format (named v1 here) is retired, so the
    cached v1 `.json` fixtures can no longer be loaded through a calculator.
    This fixture performs the conversion the changelog documents: it exports the
    cached trained MACE-MP-0b3-medium checkpoint with
    ``radial_format="compact"``, which is the format both routings stream.

    The export uses the extractor's default spline resolution, so the model is
    the trained checkpoint and not a coarse approximation of it.  A test that
    compares against an external oracle (the PyTorch calculator, a published
    energy) should still convert its own checkpoint at the resolution its
    tolerance needs rather than assume this one.
    """

    try:
        from mace.calculators.foundations_models import download_mace_mp_checkpoint
        from symmetrix.extract_mace_data import extract_mace_data
    except ImportError as exc:  # pragma: no cover - depends on mace-torch
        pytest.skip(f"the MACE checkpoint exporter is unavailable: {exc}")

    try:
        checkpoint = download_mace_mp_checkpoint("medium")
    except Exception as exc:  # pragma: no cover - offline
        pytest.skip(f"the MACE foundation checkpoint is unavailable: {exc}")

    try:
        data = extract_mace_data(checkpoint, radial_format="compact")
    except (FileNotFoundError, RuntimeError) as exc:  # pragma: no cover - offline
        pytest.skip(f"the MACE foundation checkpoint is unusable: {exc}")

    root = tmp_path_factory.mktemp("compact-mace")
    path = root / "mace-mp-0b3-medium-compact.json"
    path.write_text(json.dumps(data, separators=(",", ":")))
    return path


@pytest.fixture(scope="session")
def macefield_model_path():
    _require_macefield()
    try:
        return cached_macefield_model_path()
    except (FileNotFoundError, RuntimeError) as exc:
        pytest.skip(str(exc))
