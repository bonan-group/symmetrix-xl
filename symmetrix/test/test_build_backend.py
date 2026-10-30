from __future__ import annotations

import importlib.util
import builtins
import json
import sys
import types
from pathlib import Path

import pytest

TOOLS_ROOT = Path(__file__).parents[2] / "tools"
sys.path.insert(0, str(TOOLS_ROOT))

from _symmetrix_build.source_provenance import (  # noqa: E402
    symmetrix_native_source_fingerprint,
    symmetrix_source_fingerprint,
)

PROJECT_ROOT = Path(__file__).parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
BACKEND_PATH = PROJECT_ROOT / "build_backend.py"


@pytest.fixture
def build_backend(monkeypatch):
    calls = []
    fake_build = types.ModuleType("scikit_build_core.build")

    def record(name):
        def hook(*args):
            calls.append((name, args))
            return name

        return hook

    for name in (
        "build_editable",
        "build_sdist",
        "build_wheel",
        "get_requires_for_build_editable",
        "get_requires_for_build_sdist",
        "get_requires_for_build_wheel",
        "prepare_metadata_for_build_wheel",
    ):
        setattr(fake_build, name, record(name))
    fake_package = types.ModuleType("scikit_build_core")
    fake_package.build = fake_build
    monkeypatch.setitem(sys.modules, "scikit_build_core", fake_package)
    monkeypatch.setitem(sys.modules, "scikit_build_core.build", fake_build)

    spec = importlib.util.spec_from_file_location(
        "symmetrix_test_build_backend", BACKEND_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._test_calls = calls
    return module


def test_source_provenance_is_added_to_build_settings(build_backend):
    settings = build_backend._with_source_provenance({"custom.setting": "value"})

    assert settings["custom.setting"] == "value"
    assert settings[
        "cmake.define.SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256"
    ] == symmetrix_source_fingerprint(REPOSITORY_ROOT)
    assert settings[
        "cmake.define.SYMMETRIX_BUILD_NATIVE_SOURCE_CONTENT_SHA256"
    ] == symmetrix_native_source_fingerprint(REPOSITORY_ROOT)


def test_explicit_source_provenance_is_preserved(build_backend):
    settings = build_backend._with_source_provenance(
        {
            "cmake.define.SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256": "source",
            "cmake.define.SYMMETRIX_BUILD_NATIVE_SOURCE_CONTENT_SHA256": "native",
        }
    )

    assert settings["cmake.define.SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256"] == "source"
    assert (
        settings["cmake.define.SYMMETRIX_BUILD_NATIVE_SOURCE_CONTENT_SHA256"]
        == "native"
    )


def test_sdist_fingerprints_are_self_contained_without_repository_tools(
    build_backend, monkeypatch, tmp_path
):
    project = tmp_path / "symmetrix"
    for relative, content in (
        ("pyproject.toml", b"[project]\nname='example'\n"),
        ("CMakeLists.txt", b"project(example)\n"),
        ("source/symmetrix/__init__.py", b"value = 1\n"),
        ("libsymmetrix/source/example.cpp", b"int value = 1;\n"),
        ("libsymmetrix/external/kokkos/CMakeLists.txt", b"project(Kokkos)\n"),
    ):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    monkeypatch.setattr(build_backend, "_PROJECT_ROOT", project)
    monkeypatch.setattr(build_backend, "_REPOSITORY_ROOT", tmp_path)
    original_import = builtins.__import__

    def without_repository_tools(name, *args, **kwargs):
        if name == "tools" or name.startswith("tools."):
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_repository_tools)

    first = build_backend._source_hashes()
    second = build_backend._source_hashes()
    (project / "libsymmetrix/source/example.cpp").write_text("int value = 2;\n")
    changed = build_backend._source_hashes()

    assert first == second
    assert all(len(value) == 64 for value in first)
    assert changed[0] != first[0]
    assert changed[1] != first[1]


def test_sdist_revision_is_forwarded_from_the_vendored_record(
    build_backend, monkeypatch, tmp_path
):
    project = tmp_path / "symmetrix"
    project.mkdir()
    revision = "1" * 40
    (project / "VENDORED_DEPS.json").write_text(json.dumps({"source_commit": revision}))
    monkeypatch.setattr(build_backend, "_PROJECT_ROOT", project)
    monkeypatch.setattr(build_backend, "_source_hashes", lambda: ("a" * 64, "b" * 64))

    settings = build_backend._with_source_provenance(None)

    assert settings["cmake.define.SYMMETRIX_BUILD_SOURCE_COMMIT"] == revision
    assert settings["cmake.define.SYMMETRIX_BUILD_SOURCE_DIRTY"] == "0"


def test_sdist_rejects_an_invalid_vendored_revision(
    build_backend, monkeypatch, tmp_path
):
    project = tmp_path / "symmetrix"
    project.mkdir()
    (project / "VENDORED_DEPS.json").write_text(
        json.dumps({"source_commit": "not-a-commit"})
    )
    monkeypatch.setattr(build_backend, "_PROJECT_ROOT", project)
    monkeypatch.setattr(build_backend, "_source_hashes", lambda: ("a" * 64, "b" * 64))

    with pytest.raises(RuntimeError, match="invalid source_commit"):
        build_backend._with_source_provenance(None)


def test_build_wheel_forwards_fingerprinted_settings(build_backend):
    result = build_backend.build_wheel("dist", {"custom.setting": "value"})

    assert result == "build_wheel"
    name, args = build_backend._test_calls[-1]
    assert name == "build_wheel"
    assert args[0] == "dist"
    assert args[1]["custom.setting"] == "value"
    assert len(args[1]["cmake.define.SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256"]) == 64
    assert (
        len(args[1]["cmake.define.SYMMETRIX_BUILD_NATIVE_SOURCE_CONTENT_SHA256"]) == 64
    )
