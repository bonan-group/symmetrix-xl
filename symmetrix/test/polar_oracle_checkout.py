"""Identity checks for the opt-in graph-electrostatics test oracle."""

from __future__ import annotations

import hashlib
import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Mapping

import pytest


GRAPH_ELECTROSTATICS_REVISION = "e0eb8945c21e203656f4e0bbece5d1da488b6753"
GRAPH_ELECTROSTATICS_FILES = {
    "graph_longrange/__init__.py": (
        "079b233934150a3eee30444ce2408e09895ec4dd958ec60a822f870fa1284b99"
    ),
    "graph_longrange/__version__.py": (
        "c70fdd622a5b6869ba695d8e3e01f6df8fce0d19cc3610d660b6aed471546804"
    ),
    "graph_longrange/gto_utils.py": (
        "9f84ee541d45160eded094628d9a27e3d4a050501d5652da4e482c190010628f"
    ),
    "graph_longrange/realspace_electrostatics.py": (
        "cee8dc184dda90ef0dff12535e4f647e43a06e138047bce37acd967bd303d0f1"
    ),
    "graph_longrange/utils.py": (
        "1f8a9b63140b6131ec89f5f9ae9674dccb66d405fb6d326389142149390f4c34"
    ),
}
GRAPH_ELECTROSTATICS_MODULES = {
    "graph_longrange": "graph_longrange/__init__.py",
    "graph_longrange.__version__": "graph_longrange/__version__.py",
    "graph_longrange.gto_utils": "graph_longrange/gto_utils.py",
    "graph_longrange.realspace_electrostatics": (
        "graph_longrange/realspace_electrostatics.py"
    ),
    "graph_longrange.utils": "graph_longrange/utils.py",
}


def optional_checkout(environment: str) -> Path | None:
    """Return an explicitly configured oracle checkout, if one was requested."""

    value = os.environ.get(environment)
    return None if not value else Path(value).expanduser()


def required_checkout(environment: str) -> Path:
    """Return an explicit oracle checkout or skip its dependent test."""

    checkout = optional_checkout(environment)
    if checkout is None:
        pytest.skip(f"set {environment} to run this private-oracle test")
    if not checkout.is_dir():
        pytest.skip(f"{environment} checkout does not exist: {checkout}")
    return checkout


def require_checkout_identity(
    checkout: Path,
    expected_revision: str,
    expected_files: Mapping[str, str],
) -> None:
    """Fail before import unless a checkout has the expected code identity."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.fail(f"cannot inspect pinned oracle checkout {checkout}: {error}")
    observed_revision = completed.stdout.strip()
    if observed_revision != expected_revision:
        pytest.fail(
            "pinned oracle revision mismatch: "
            f"expected {expected_revision}, observed {observed_revision}"
        )

    for relative_path, expected_digest in expected_files.items():
        path = checkout / relative_path
        try:
            observed_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            pytest.fail(f"cannot read pinned oracle source {relative_path}: {error}")
        if observed_digest != expected_digest:
            pytest.fail(
                "pinned oracle content mismatch for "
                f"{relative_path}: expected {expected_digest}, "
                f"observed {observed_digest}"
            )


def _require_module_path(module_name: str, module: ModuleType, expected: Path) -> None:
    resolved_file = getattr(module, "__file__", None)
    if resolved_file is None or Path(resolved_file).resolve() != expected.resolve():
        pytest.fail(
            "cached oracle module conflict for "
            f"{module_name}: expected {expected.resolve()}, observed {resolved_file}"
        )


def import_checkout_module(
    checkout: Path,
    expected_revision: str,
    expected_files: Mapping[str, str],
    module_name: str,
    expected_modules: Mapping[str, str],
) -> ModuleType:
    """Verify a checkout and import only if cached modules resolve inside it."""

    require_checkout_identity(checkout, expected_revision, expected_files)
    for cached_name, relative_path in expected_modules.items():
        cached = sys.modules.get(cached_name)
        if cached is not None:
            _require_module_path(cached_name, cached, checkout / relative_path)

    checkout_text = str(checkout.resolve())
    if checkout_text not in sys.path:
        sys.path.insert(0, checkout_text)
    module = importlib.import_module(module_name)

    for imported_name, relative_path in expected_modules.items():
        imported = sys.modules.get(imported_name)
        if imported is None:
            pytest.fail(
                f"verified oracle import did not load required module {imported_name}"
            )
        _require_module_path(imported_name, imported, checkout / relative_path)
    return module


def import_graph_electrostatics_realspace(checkout: Path | None) -> ModuleType:
    """Skip an absent oracle; otherwise return its verified real-space module."""

    if checkout is None or not checkout.is_dir():
        pytest.skip("pinned graph-electrostatics checkout is unavailable")
    return import_checkout_module(
        checkout,
        GRAPH_ELECTROSTATICS_REVISION,
        GRAPH_ELECTROSTATICS_FILES,
        "graph_longrange.realspace_electrostatics",
        GRAPH_ELECTROSTATICS_MODULES,
    )
