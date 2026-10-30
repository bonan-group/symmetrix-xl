"""Load one explicitly pinned Symmetrix extension for qualification tests."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


def _module_path(name: str) -> Path | None:
    module = sys.modules.get(name)
    value = getattr(module, "__file__", None)
    return Path(value).resolve() if isinstance(value, str) else None


def bootstrap_explicit_extension() -> None:
    """Bind the requested source package and native extension exactly once."""

    source_root = os.environ.get("SYMMETRIX_SOURCE_ROOT")
    extension = os.environ.get("SYMMETRIX_EXTENSION")
    if source_root is None and extension is None:
        return
    if source_root is None or extension is None:
        raise RuntimeError(
            "SYMMETRIX_SOURCE_ROOT and SYMMETRIX_EXTENSION must be set together"
        )
    package_dir = Path(source_root).resolve() / "symmetrix/source/symmetrix"
    package_path = package_dir / "__init__.py"
    extension_path = Path(extension).resolve()
    if not package_path.is_file():
        raise RuntimeError(f"invalid SYMMETRIX_SOURCE_ROOT: {source_root}")
    if not extension_path.is_file():
        raise RuntimeError(f"invalid SYMMETRIX_EXTENSION: {extension}")

    existing_package = _module_path("symmetrix")
    existing_native = _module_path("symmetrix._native_cpu")
    if existing_package is not None or existing_native is not None:
        installed_package_path = extension_path.parent / "__init__.py"
        allowed_package_paths = {package_path, installed_package_path.resolve()}
        if (
            existing_package not in allowed_package_paths
            or existing_native != extension_path
        ):
            raise RuntimeError(
                "explicit Symmetrix extension must be bootstrapped before any "
                "symmetrix package or native module is imported"
            )
        return

    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if type(finder).__name__ != "ScikitBuildRedirectingFinder"
    ]
    package_spec = importlib.util.spec_from_file_location(
        "symmetrix",
        package_path,
        submodule_search_locations=[str(package_dir)],
    )
    native_spec = importlib.util.spec_from_file_location(
        "symmetrix._native_cpu", extension_path
    )
    if package_spec is None or package_spec.loader is None:
        raise RuntimeError("could not create the explicit Symmetrix package spec")
    if native_spec is None or native_spec.loader is None:
        raise RuntimeError("could not create the explicit Symmetrix extension spec")
    package = importlib.util.module_from_spec(package_spec)
    sys.modules["symmetrix"] = package
    package_spec.loader.exec_module(package)
    native = importlib.util.module_from_spec(native_spec)
    sys.modules["symmetrix._native_cpu"] = native
    native_spec.loader.exec_module(native)
    from symmetrix import backend_loader

    backend_loader._native_module = native
    backend_loader._selected = backend_loader._cpu_descriptor(package.__version__)
    sys.modules["symmetrix.symmetrix"] = native


__all__ = ["bootstrap_explicit_extension"]
