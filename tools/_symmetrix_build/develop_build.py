"""Direct incremental native builds for an already installed frontend."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .command import BuildError, CommandRunner, sha256_file
from .manifest import TargetManifest
from .package_identity import PackageIdentity, package_identity
from .python_build import (
    _frontend_version,
    build_environment,
    cmake_definitions,
    default_build_directory,
    prepare_build_directory,
)
from .source_provenance import symmetrix_source_provenance

RUNTIME_COMPONENT = "symmetrix_development_runtime"


@dataclass(frozen=True)
class DevelopmentInvocation:
    commands: tuple[tuple[str, ...], ...]
    environment: dict[str, str]
    build_directory: Path
    manifest_path: Path
    platform_library: Path
    package_directory: Path
    extension_path: Path
    descriptor_path: Path
    identity: PackageIdentity
    source_provenance: dict[str, object]


def _runtime_layout(
    manifest: TargetManifest,
    source_project: Path,
    runner: CommandRunner,
) -> dict[str, str]:
    identity = package_identity(manifest)
    script = """
import importlib.metadata
import json
import pathlib
import sysconfig

frontend_name = "symmetrix-xl"
target_name = __TARGET_DISTRIBUTION__
package_name = __TARGET_PACKAGE__
try:
    frontend = importlib.metadata.distribution(frontend_name)
    target = importlib.metadata.distribution(target_name)
except importlib.metadata.PackageNotFoundError as error:
    print(json.dumps({"missing": error.name}))
else:
    platlib = pathlib.Path(sysconfig.get_path("platlib")).resolve()
    package = pathlib.Path(target.locate_file(package_name)).resolve()
    print(json.dumps({
        "platform_library": str(platlib),
        "package_directory": str(package),
        "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX") or ".so",
        "frontend_version": frontend.version,
        "target_version": target.version,
    }, sort_keys=True))
""".replace("__TARGET_DISTRIBUTION__", json.dumps(identity.distribution)).replace(
        "__TARGET_PACKAGE__", json.dumps(identity.package)
    )
    result = runner.run((manifest.python_executable, "-c", script), check=True)
    try:
        layout = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise BuildError(
            f"selected interpreter returned invalid runtime layout: {result.stdout}"
        ) from error
    if missing := layout.get("missing"):
        raise BuildError(
            f"develop requires {missing!r} to be installed in the selected "
            "interpreter; "
            "run the ordinary install command once before using the fast edit loop"
        )
    expected_version = _frontend_version(source_project)
    for field in ("frontend_version", "target_version"):
        if layout.get(field) != expected_version:
            raise BuildError(
                f"installed {field.replace('_', ' ')} is {layout.get(field)!r}; "
                f"expected {expected_version!r}"
            )
    platform_library = Path(layout["platform_library"])
    expected_package = platform_library / identity.package
    package_directory = Path(layout["package_directory"])
    if package_directory != expected_package:
        raise BuildError(
            "develop requires a normal installation in the selected interpreter's "
            f"platform-library directory; found {package_directory}, expected "
            f"{expected_package}"
        )
    if not package_directory.is_dir():
        raise BuildError(f"installed backend package is missing: {package_directory}")
    return layout


def development_build_invocation(
    manifest: TargetManifest,
    *,
    repo_root: str | Path,
    runner: CommandRunner,
    build_root: str | Path | None = None,
    jobs: int | None = None,
) -> DevelopmentInvocation:
    root = Path(repo_root).resolve()
    source_project = root / "symmetrix"
    if not (source_project / "pyproject.toml").is_file():
        raise BuildError(f"Symmetrix Python project is missing: {source_project}")
    layout = _runtime_layout(manifest, source_project, runner)
    if build_root is None:
        build_directory = default_build_directory(root, manifest)
    else:
        build_directory = Path(build_root).expanduser().resolve() / manifest.fingerprint
    manifest_path = prepare_build_directory(build_directory, manifest)
    provenance = symmetrix_source_provenance(root, runner)
    identity = package_identity(manifest)
    definitions = cmake_definitions(manifest, source_provenance=provenance)
    definitions.update(
        {
            "SYMMETRIX_ENABLE_DEVELOPMENT_RUNTIME_COMPONENT": "ON",
            "SYMMETRIX_TARGET_MANIFEST_FINGERPRINT": manifest.fingerprint,
            "SYMMETRIX_DEVELOPMENT_BACKEND": identity.backend,
            "SYMMETRIX_DEVELOPMENT_DESCRIPTOR_SELECTOR": identity.selector,
            "SYMMETRIX_DEVELOPMENT_TOOLKIT": identity.toolkit,
            "SYMMETRIX_FRONTEND_VERSION": _frontend_version(source_project),
        }
    )
    cmake = manifest.toolchain.cmake
    configure = [
        cmake,
        "-S",
        str(source_project),
        "-B",
        str(build_directory),
        "-G",
        manifest.toolchain.generator,
    ]
    configure.extend(f"-D{name}={value}" for name, value in definitions.items())
    build = (cmake, "--build", str(build_directory), "--target", "symmetrix_bindings")
    platform_library = Path(layout["platform_library"])
    install = (
        cmake,
        "--install",
        str(build_directory),
        "--component",
        RUNTIME_COMPONENT,
        "--prefix",
        str(platform_library),
    )
    package_directory = Path(layout["package_directory"])
    extension_path = (
        package_directory / f"{identity.module}{layout['extension_suffix']}"
    )
    descriptor_path = package_directory / (
        "_backend_cpu.json" if identity.base else "backend.json"
    )
    invocation = DevelopmentInvocation(
        commands=(tuple(configure), build, install),
        environment=build_environment(manifest, jobs=jobs),
        build_directory=build_directory,
        manifest_path=manifest_path,
        platform_library=platform_library,
        package_directory=package_directory,
        extension_path=extension_path,
        descriptor_path=descriptor_path,
        identity=identity,
        source_provenance=provenance,
    )
    record = development_result(invocation, validation=None)
    record["commands"] = [list(command) for command in invocation.commands]
    (build_directory / "develop-invocation.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    return invocation


def run_development_build(
    invocation: DevelopmentInvocation, runner: CommandRunner
) -> None:
    log_path = invocation.build_directory / "develop-build.log"
    for index, command in enumerate(invocation.commands):
        result = runner.run_logged(
            command,
            log_path,
            env=invocation.environment,
            append=index > 0,
        )
        if result.returncode != 0:
            raise BuildError(
                f"development build step failed ({result.returncode}); see {log_path}"
            )
    for label, path in (
        ("extension", invocation.extension_path),
        ("descriptor", invocation.descriptor_path),
    ):
        if not path.is_file():
            raise BuildError(f"development runtime {label} was not installed: {path}")


def validate_development_runtime(
    invocation: DevelopmentInvocation, runner: CommandRunner
) -> dict[str, object]:
    script = """
import hashlib
import json
import pathlib
import symmetrix
from symmetrix import symmetrix as native

path = pathlib.Path(native.__file__).resolve()
native._init_kokkos()
try:
    record = {
        "extension": str(path),
        "extension_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "selected_backend": symmetrix.selected_backend(),
        "build_info": native._backend_build_info(),
        "execution_space": native._kokkos_default_execution_space(),
        "device_sentinel": native._kokkos_device_sentinel(),
        "native_module": native.__name__,
    }
    print(json.dumps(record, sort_keys=True))
finally:
    native._finalize_kokkos()
"""
    environment = dict(os.environ)
    environment["SYMMETRIX_BACKEND"] = invocation.identity.selector
    result = runner.run(
        (
            TargetManifest.read(invocation.manifest_path).python_executable,
            "-c",
            script,
        ),
        check=True,
        env=environment,
    )
    try:
        record = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise BuildError(
            f"development runtime validation returned invalid JSON: {result.stdout}"
        ) from error
    expected = invocation.source_provenance
    build_info = record.get("build_info") or {}
    selected = record.get("selected_backend") or {}
    expected_execution_space = {
        "cpu": "OpenMP",
        "cuda": "Cuda",
        "hip": "HIP",
    }[invocation.identity.backend]
    checks = {
        "extension path": record.get("extension") == str(invocation.extension_path),
        "native module": record.get("native_module")
        == f"{invocation.identity.package}.{invocation.identity.module}",
        "selector": selected.get("selector") == invocation.identity.selector,
        "descriptor path": selected.get("descriptor_path")
        == str(invocation.descriptor_path),
        "descriptor target": selected.get("target_manifest_fingerprint")
        == TargetManifest.read(invocation.manifest_path).fingerprint,
        "descriptor source hash": selected.get("source_content_sha256")
        == expected["content_sha256"],
        "descriptor native source hash": selected.get("native_source_content_sha256")
        == expected["native_content_sha256"],
        "backend": build_info.get("backend") == invocation.identity.backend,
        "architecture": str(build_info.get("architecture", "")).lower().replace("_", "")
        == invocation.identity.architecture.lower().replace("_", ""),
        "source commit": build_info.get("source_commit")
        == str(expected.get("revision", "")),
        "source dirty": build_info.get("source_dirty") == bool(expected.get("dirty")),
        "source hash": build_info.get("source_content_sha256")
        == expected["content_sha256"],
        "native source hash": build_info.get("native_source_content_sha256")
        == expected["native_content_sha256"],
        "execution space": record.get("execution_space") == expected_execution_space,
        "device sentinel": record.get("device_sentinel") is True,
    }
    failed = [label for label, passed in checks.items() if not passed]
    if failed:
        raise BuildError(
            "development runtime validation provenance mismatch: " + ", ".join(failed)
        )
    return {
        "status": "ok",
        "extension_sha256": record.get("extension_sha256"),
        "execution_space": record.get("execution_space"),
        "selected_backend": {
            name: selected.get(name)
            for name in (
                "selector",
                "backend",
                "architecture",
                "descriptor_path",
                "target_manifest_fingerprint",
            )
        },
        "build_provenance": {
            name: build_info.get(name)
            for name in (
                "source_commit",
                "source_dirty",
                "source_content_sha256",
                "native_source_content_sha256",
            )
        },
    }


def development_result(
    invocation: DevelopmentInvocation,
    validation: dict[str, object] | None,
) -> dict[str, object]:
    provenance = {
        name: invocation.source_provenance[name]
        for name in (
            "revision",
            "dirty",
            "content_sha256",
            "native_content_sha256",
        )
    }
    result: dict[str, object] = {
        "development_only": True,
        "build_directory": str(invocation.build_directory),
        "extension_path": str(invocation.extension_path),
        "descriptor_path": str(invocation.descriptor_path),
        "target": {
            "manifest_fingerprint": TargetManifest.read(
                invocation.manifest_path
            ).fingerprint,
            "selector": invocation.identity.selector,
            "backend": invocation.identity.backend,
            "architecture": invocation.identity.architecture,
        },
        "source_provenance": provenance,
    }
    if validation is not None:
        result["extension_sha256"] = sha256_file(invocation.extension_path)
        result["validation"] = validation
    return result
