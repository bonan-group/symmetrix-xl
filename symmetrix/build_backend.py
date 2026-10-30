"""PEP 517 wrapper that embeds Symmetrix source provenance in native builds."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from scikit_build_core import build as _scikit_build_core

_PROJECT_ROOT = Path(__file__).resolve().parent
_REPOSITORY_ROOT = _PROJECT_ROOT.parent
_SOURCE_HASH_SETTING = "cmake.define.SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256"
_NATIVE_HASH_SETTING = "cmake.define.SYMMETRIX_BUILD_NATIVE_SOURCE_CONTENT_SHA256"
_SOURCE_COMMIT_SETTING = "cmake.define.SYMMETRIX_BUILD_SOURCE_COMMIT"
_SOURCE_DIRTY_SETTING = "cmake.define.SYMMETRIX_BUILD_SOURCE_DIRTY"


def _source_hashes() -> tuple[str, str]:
    """Compute the hashes used by CMake backend descriptors."""

    helper = _REPOSITORY_ROOT / "tools/_symmetrix_build/source_provenance.py"
    if helper.is_file():
        # A checkout has the maintained Git-aware helper. PEP 517 backend-path
        # contains only the Python project directory, so expose its parent.
        import sys

        repository = str(_REPOSITORY_ROOT)
        if repository not in sys.path:
            sys.path.insert(0, repository)
        from tools._symmetrix_build.source_provenance import (
            symmetrix_native_source_fingerprint,
            symmetrix_source_fingerprint,
        )

        return (
            symmetrix_source_fingerprint(_REPOSITORY_ROOT),
            symmetrix_native_source_fingerprint(_REPOSITORY_ROOT),
        )
    # Published sdists deliberately contain no repository-level ``tools``
    # package. Hash their self-contained vendored source tree directly.
    return _archive_source_fingerprints(_PROJECT_ROOT)


def _hash_archive_inputs(
    project_root: Path, roots: tuple[str, ...], files: tuple[str, ...]
) -> str:
    digest = hashlib.sha256()
    candidates = [project_root / relative for relative in files]
    for relative in roots:
        root = project_root / relative
        if root.is_dir():
            candidates.extend(path for path in root.rglob("*") if path.is_file())
    for path in sorted(set(candidates)):
        if not path.is_file() or any(
            part in {".git", "__pycache__", "build"} for part in path.parts
        ):
            continue
        relative = path.relative_to(project_root).as_posix().encode("utf-8")
        content = (
            os.readlink(path).encode("utf-8")
            if path.is_symlink()
            else path.read_bytes()
        )
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.hexdigest()


def _archive_source_fingerprints(project_root: Path) -> tuple[str, str]:
    """Fingerprint a self-contained sdist without repository tooling or Git."""

    common_files = ("CMakeLists.txt", "pyproject.toml")
    native_roots = (
        "libsymmetrix/source",
        "source/cpp",
        "libsymmetrix/cmake",
        "cmake",
        "libsymmetrix/external",
        "external",
    )
    source = _hash_archive_inputs(
        project_root,
        (*native_roots, "source/symmetrix"),
        common_files,
    )
    native = _hash_archive_inputs(project_root, native_roots, common_files)
    return source, native


def _archive_source_revision(project_root: Path) -> str | None:
    """Return the superproject revision recorded by the sdist builder."""

    record_path = project_root / "VENDORED_DEPS.json"
    if not record_path.is_file():
        return None
    try:
        record = json.loads(record_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"cannot read source provenance from {record_path}"
        ) from error
    revision = record.get("source_commit")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise RuntimeError(
            f"source provenance in {record_path} has an invalid source_commit"
        )
    return revision


def _with_source_provenance(
    config_settings: dict[str, Any] | None,
) -> dict[str, Any]:
    """Add automatic provenance settings without overriding explicit values."""

    settings = dict(config_settings or {})
    source_hash, native_hash = _source_hashes()
    settings.setdefault(_SOURCE_HASH_SETTING, source_hash)
    settings.setdefault(_NATIVE_HASH_SETTING, native_hash)
    if revision := _archive_source_revision(_PROJECT_ROOT):
        settings.setdefault(_SOURCE_COMMIT_SETTING, revision)
        settings.setdefault(_SOURCE_DIRTY_SETTING, "0")
    return settings


def get_requires_for_build_wheel(config_settings=None):
    return _scikit_build_core.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_editable(config_settings=None):
    hook = getattr(_scikit_build_core, "get_requires_for_build_editable", None)
    if hook is None:
        return []
    return hook(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    return _scikit_build_core.get_requires_for_build_sdist(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    return _scikit_build_core.prepare_metadata_for_build_wheel(
        metadata_directory, _with_source_provenance(config_settings)
    )


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return _scikit_build_core.build_wheel(
        wheel_directory,
        _with_source_provenance(config_settings),
        metadata_directory,
    )


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    return _scikit_build_core.build_editable(
        wheel_directory,
        _with_source_provenance(config_settings),
        metadata_directory,
    )


def build_sdist(sdist_directory, config_settings=None):
    return _scikit_build_core.build_sdist(sdist_directory, config_settings)
