import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from symmetrix import backend_loader


def _installed_frontend_version():
    return backend_loader._cpu_descriptor("0.0.1").frontend_version


FRONTEND_VERSION = _installed_frontend_version()


def _descriptor(tmp_path, **overrides):
    value = {
        "schema_version": 1,
        "selector": "cuda13-sm120",
        "backend": "cuda",
        "architecture": "sm120",
        "distribution": "symmetrix-xl-cuda13-sm120",
        "frontend_version": FRONTEND_VERSION,
        "native_abi": 1,
        "package": "symmetrix_backend_cuda13_sm120",
        "module": "_native_cuda13_sm120",
        "toolkit": "cuda13",
    }
    value.update(overrides)
    path = tmp_path / "symmetrix_backend_cuda13_sm120/backend.json"
    path.parent.mkdir()
    path.write_text(json.dumps(value))
    return path


def _entry_point(path):
    root = path.parents[1]
    distribution = SimpleNamespace(locate_file=lambda relative: root / relative)
    return SimpleNamespace(
        name="cuda13-sm120",
        value="symmetrix_backend_cuda13_sm120",
        dist=distribution,
    )


def test_descriptor_discovery_does_not_import_native_module(tmp_path, monkeypatch):
    path = _descriptor(tmp_path)
    monkeypatch.setattr(backend_loader, "_entry_points", lambda: (_entry_point(path),))
    loaded_before = backend_loader._native_module

    descriptors = backend_loader.discover_backends(FRONTEND_VERSION)

    assert [item.selector for item in descriptors] == ["cpu", "cuda13-sm120"]
    assert backend_loader._native_module is loaded_before


def test_automatic_selection_requires_exact_visible_architecture(tmp_path, monkeypatch):
    path = _descriptor(tmp_path)
    monkeypatch.setattr(backend_loader, "_entry_points", lambda: (_entry_point(path),))
    monkeypatch.setattr(
        backend_loader, "_visible_targets", lambda: {"cuda": {"sm120"}, "hip": set()}
    )
    assert (
        backend_loader.select_backend(FRONTEND_VERSION, "auto").selector
        == "cuda13-sm120"
    )

    monkeypatch.setattr(
        backend_loader, "_visible_targets", lambda: {"cuda": {"sm90"}, "hip": set()}
    )
    assert backend_loader.select_backend(FRONTEND_VERSION, "auto").selector == "cpu"


def test_backend_inventory_marks_exact_accelerator_as_usable(tmp_path, monkeypatch):
    path = _descriptor(tmp_path)
    monkeypatch.setattr(backend_loader, "_entry_points", lambda: (_entry_point(path),))
    monkeypatch.setattr(
        backend_loader, "_visible_targets", lambda: {"cuda": {"sm120"}, "hip": set()}
    )

    inventory = backend_loader.backend_inventory(FRONTEND_VERSION)
    by_selector = {item["selector"]: item for item in inventory}

    assert by_selector["cuda13-sm120"]["availability"]["status"] == "usable"
    assert by_selector["cpu"]["availability"]["status"] == "fallback"


def test_backend_inventory_explains_incompatible_accelerator(tmp_path, monkeypatch):
    path = _descriptor(tmp_path)
    monkeypatch.setattr(backend_loader, "_entry_points", lambda: (_entry_point(path),))
    monkeypatch.setattr(
        backend_loader, "_visible_targets", lambda: {"cuda": {"sm90"}, "hip": set()}
    )

    inventory = backend_loader.backend_inventory(FRONTEND_VERSION)
    gpu = next(item for item in inventory if item["backend"] == "cuda")

    assert gpu["availability"] == {
        "status": "incompatible",
        "reason": "requires sm120; visible cuda targets: sm90",
    }


def test_explicit_incompatible_backend_fails_before_import(tmp_path, monkeypatch):
    path = _descriptor(tmp_path, frontend_version="9.0")
    monkeypatch.setattr(backend_loader, "_entry_points", lambda: (_entry_point(path),))

    with pytest.raises(backend_loader.BackendError, match="requires symmetrix 9.0"):
        backend_loader.select_backend(FRONTEND_VERSION, "cuda13-sm120")


def test_missing_backend_error_provides_install_command(monkeypatch):
    monkeypatch.setattr(backend_loader, "_entry_points", lambda: ())

    with pytest.raises(
        backend_loader.BackendError, match="pip install symmetrix-xl-cuda13-sm120"
    ):
        backend_loader.select_backend(FRONTEND_VERSION, "cuda13-sm120")


def test_explicit_backend_rejects_wrong_visible_device(tmp_path, monkeypatch):
    path = _descriptor(tmp_path)
    monkeypatch.setattr(backend_loader, "_entry_points", lambda: (_entry_point(path),))
    monkeypatch.setattr(
        backend_loader, "_visible_targets", lambda: {"cuda": {"sm90"}, "hip": set()}
    )

    with pytest.raises(backend_loader.BackendError, match="requires sm120"):
        backend_loader.select_backend(FRONTEND_VERSION, "cuda13-sm120")


def test_duplicate_selector_is_rejected(tmp_path, monkeypatch):
    first = _descriptor(tmp_path)
    second_root = tmp_path / "other"
    second_root.mkdir()
    second = _descriptor(second_root)
    monkeypatch.setattr(
        backend_loader,
        "_entry_points",
        lambda: (_entry_point(first), _entry_point(second)),
    )

    with pytest.raises(backend_loader.BackendError, match="duplicate backend selector"):
        backend_loader.discover_backends(FRONTEND_VERSION)


def _cpu_v4_entry_point(tmp_path):
    value = {
        "schema_version": 1,
        "selector": "cpu-avx512",
        "backend": "cpu",
        "architecture": "x86-64-v4",
        "distribution": "symmetrix-xl-cpu-avx512",
        "frontend_version": FRONTEND_VERSION,
        "native_abi": 1,
        "package": "symmetrix_backend_cpu_avx512",
        "module": "_native_cpu_avx512",
        "toolkit": "",
    }
    package = "symmetrix_backend_cpu_avx512"
    path = tmp_path / package / "backend.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    distribution = SimpleNamespace(locate_file=lambda relative: tmp_path / relative)
    return SimpleNamespace(
        name="cpu-avx512",
        value=package,
        dist=distribution,
    )


def _no_gpus():
    return lambda: {"cuda": set(), "hip": set()}


def _baseline_cpu_descriptor():
    return backend_loader.BackendDescriptor(
        selector="cpu",
        backend="cpu",
        architecture="native-or-wheel-baseline",
        distribution="symmetrix-xl",
        frontend_version=FRONTEND_VERSION,
        native_abi=1,
        package="symmetrix",
        module="_native_cpu",
    )


def test_cpu_v4_addon_is_preferred_over_baseline_on_avx512_host(tmp_path, monkeypatch):
    monkeypatch.setattr(
        backend_loader, "_entry_points", lambda: (_cpu_v4_entry_point(tmp_path),)
    )
    monkeypatch.setattr(backend_loader, "_visible_targets", _no_gpus())
    monkeypatch.setattr(
        backend_loader,
        "_host_cpu_flags",
        lambda: backend_loader._V3_CPU_FLAGS | backend_loader._V4_CPU_FLAGS,
    )
    monkeypatch.setattr(
        backend_loader, "_cpu_descriptor", lambda version: _baseline_cpu_descriptor()
    )

    assert (
        backend_loader.select_backend(FRONTEND_VERSION, "auto").selector == "cpu-avx512"
    )


def _local_native_descriptor():
    return backend_loader.BackendDescriptor(
        selector="cpu",
        backend="cpu",
        architecture="native",
        distribution="symmetrix-xl",
        frontend_version=FRONTEND_VERSION,
        native_abi=1,
        package="symmetrix",
        module="_native_cpu",
    )


def test_native_cpu_priority_exceeds_matching_v4_level():
    native = _local_native_descriptor()
    v4 = replace(
        native,
        selector="cpu-avx512",
        architecture="x86-64-v4",
        distribution="symmetrix-xl-cpu-avx512",
    )

    assert backend_loader._cpu_backend_priority(
        native, 4
    ) > backend_loader._cpu_backend_priority(v4, 4)


def test_local_native_build_wins_over_avx512_addon_on_avx512_host(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        backend_loader, "_entry_points", lambda: (_cpu_v4_entry_point(tmp_path),)
    )
    monkeypatch.setattr(backend_loader, "_visible_targets", _no_gpus())
    monkeypatch.setattr(
        backend_loader,
        "_host_cpu_flags",
        lambda: backend_loader._V3_CPU_FLAGS | backend_loader._V4_CPU_FLAGS,
    )
    monkeypatch.setattr(
        backend_loader, "_cpu_descriptor", lambda version: _local_native_descriptor()
    )

    assert backend_loader.select_backend(FRONTEND_VERSION, "auto").selector == "cpu"


def test_cpu_v4_addon_is_skipped_on_v3_host(tmp_path, monkeypatch):
    monkeypatch.setattr(
        backend_loader, "_entry_points", lambda: (_cpu_v4_entry_point(tmp_path),)
    )
    monkeypatch.setattr(backend_loader, "_visible_targets", _no_gpus())
    monkeypatch.setattr(
        backend_loader, "_host_cpu_flags", lambda: set(backend_loader._V3_CPU_FLAGS)
    )

    assert backend_loader.select_backend(FRONTEND_VERSION, "auto").selector == "cpu"

    with pytest.raises(backend_loader.BackendError, match="requires x86-64-v4"):
        backend_loader.select_backend(FRONTEND_VERSION, "cpu-avx512")


def test_backend_inventory_marks_v4_preferred_over_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(
        backend_loader, "_entry_points", lambda: (_cpu_v4_entry_point(tmp_path),)
    )
    monkeypatch.setattr(backend_loader, "_visible_targets", _no_gpus())
    monkeypatch.setattr(
        backend_loader,
        "_host_cpu_flags",
        lambda: backend_loader._V3_CPU_FLAGS | backend_loader._V4_CPU_FLAGS,
    )
    monkeypatch.setattr(
        backend_loader, "_cpu_descriptor", lambda version: _baseline_cpu_descriptor()
    )

    inventory = backend_loader.backend_inventory(FRONTEND_VERSION)
    by_selector = {item["selector"]: item for item in inventory}

    assert by_selector["cpu-avx512"]["availability"] == {
        "status": "usable",
        "reason": "preferred CPU backend for this host",
    }
    assert by_selector["cpu"]["availability"] == {
        "status": "fallback",
        "reason": "usable when explicitly selected",
    }
