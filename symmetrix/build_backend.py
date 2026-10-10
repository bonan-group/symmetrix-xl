"""PEP 517 wrapper that embeds Symmetrix source provenance in native builds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scikit_build_core import build as _scikit_build_core

_PROJECT_ROOT = Path(__file__).resolve().parent
_REPOSITORY_ROOT = _PROJECT_ROOT.parent
_SOURCE_HASH_SETTING = "cmake.define.SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256"
_NATIVE_HASH_SETTING = "cmake.define.SYMMETRIX_BUILD_NATIVE_SOURCE_CONTENT_SHA256"


def _source_hashes() -> tuple[str, str]:
    """Compute the hashes used by CMake backend descriptors."""

    # The source checkout contains the maintained provenance helper. Import it
    # by adding the repository root explicitly because PEP 517 backend-path
    # contains only the Python project directory.
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


def _with_source_provenance(
    config_settings: dict[str, Any] | None,
) -> dict[str, Any]:
    """Add automatic provenance settings without overriding explicit values."""

    settings = dict(config_settings or {})
    source_hash, native_hash = _source_hashes()
    settings.setdefault(_SOURCE_HASH_SETTING, source_hash)
    settings.setdefault(_NATIVE_HASH_SETTING, native_hash)
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
