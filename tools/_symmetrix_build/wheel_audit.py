"""Reject release wheels containing private build data or stray build assets."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path, PurePosixPath, PureWindowsPath

from .command import BuildError
from .prebuilt_operator_artifacts import validate_generated_operator_catalog


_FORBIDDEN_COMPONENTS = frozenset(
    {
        ".agents",
        ".codex",
        ".codex-build",
        ".git",
        ".github",
        ".qualification-tmp",
        "__pycache__",
        "build",
        "include",
        "lib",
        "lib64",
        "test",
        "tests",
    }
)
_FORBIDDEN_FILE_RE = re.compile(
    r"(?:^|[_-])(?:agents|harness|plan|qualification)(?:[_-]|[.])", re.IGNORECASE
)
_SENSITIVE_CONTENT = (
    ("Linux user home", b"/home/", re.compile(rb"/home/[^/\x00\s]{1,128}/")),
    ("macOS user home", b"/Users/", re.compile(rb"/Users/[^/\x00\s]{1,128}/")),
    (
        "temporary wheel build",
        b"/tmp/symmetrix-",
        re.compile(rb"/tmp/symmetrix-[^/\x00\s]{1,128}/"),
    ),
    (
        "email address",
        b"@",
        re.compile(
            rb"[A-Z0-9._%+-]{1,64}@[A-Z0-9.-]{1,253}\."
            rb"(?:ai|cn|co|com|de|dev|edu|fr|gov|info|io|jp|me|net|org|tech|uk)",
            re.I,
        ),
    ),
    (
        "private key",
        b"PRIVATE KEY",
        re.compile(rb"-----BEGIN [A-Z ]{0,32}PRIVATE KEY-----"),
    ),
    (
        "GitHub credential",
        b"gh",
        re.compile(rb"(?:ghp|github_pat)_[A-Za-z0-9_]{20,255}"),
    ),
    (
        "PyPI credential",
        b"pypi-",
        re.compile(rb"pypi-[A-Za-z0-9_-]{20,255}"),
    ),
    ("AWS access key", b"AKIA", re.compile(rb"AKIA[0-9A-Z]{16}")),
)
_OPERATOR_CATALOG = "_operator_artifacts/catalog.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_ALLOWED_CPU_DEPENDENCY = re.compile(
    r"^(?:lib(?:c|m|dl|pthread|rt|stdc[+][+]|gcc_s|gomp|omp|iomp5)[.]so"
    r"(?:[.][0-9]+)*|ld-linux[^/]*[.]so(?:[.][0-9]+)*)$"
)
_OPENMP_RUNTIME = re.compile(
    r"^lib(gomp|omp|iomp5)(?:-[0-9a-f]{8,})?[.]so(?:[.]|$)", re.IGNORECASE
)


def _dynamic_dependencies(path: Path) -> tuple[str, ...]:
    readelf = shutil.which("readelf")
    if readelf is None:
        raise BuildError("CPU operator dependency audit requires readelf")
    result = subprocess.run(
        [readelf, "-d", os.fspath(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise BuildError(
            f"cannot inspect CPU operator dependencies for {path}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    dependencies = tuple(
        match.group(1)
        for line in result.stdout.splitlines()
        if (match := re.search(r"[(]NEEDED[)].*\[([^]]+)\]", line)) is not None
    )
    return dependencies


def audit_cpu_operator_dependencies(path: Path) -> tuple[str, ...]:
    """Require a CPU operator DSO to depend only on qualified system runtimes."""

    dependencies = _dynamic_dependencies(path)
    rejected = [
        dependency
        for dependency in dependencies
        if _ALLOWED_CPU_DEPENDENCY.fullmatch(dependency) is None
    ]
    if rejected:
        raise BuildError(
            f"CPU operator {path.name} has unapproved dependencies: "
            + ", ".join(rejected)
        )
    openmp = [
        dependency
        for dependency in dependencies
        if _OPENMP_RUNTIME.match(dependency) is not None
    ]
    if len(openmp) > 1:
        raise BuildError(
            f"CPU operator {path.name} links multiple OpenMP runtimes: "
            + ", ".join(openmp)
        )
    return dependencies


def _openmp_runtime_family(dependency: str) -> str | None:
    match = _OPENMP_RUNTIME.match(dependency)
    return match.group(1).lower() if match is not None else None


def audit_cpu_openmp_runtime_set(
    dependencies_by_member: dict[str, tuple[str, ...]], *, native_member: str
) -> None:
    """Require the native extension to own one runtime shared by all CPU DSOs."""

    native_dependencies = dependencies_by_member.get(native_member)
    if native_dependencies is None:
        raise BuildError("CPU operator wheel has no native extension to own OpenMP")
    native_families = {
        family
        for dependency in native_dependencies
        if (family := _openmp_runtime_family(dependency)) is not None
    }
    if len(native_families) != 1:
        raise BuildError(
            f"native extension {native_member} must name exactly one owning OpenMP runtime"
        )
    all_families = {
        family
        for dependencies in dependencies_by_member.values()
        for dependency in dependencies
        if (family := _openmp_runtime_family(dependency)) is not None
    }
    if len(all_families) > 1:
        raise BuildError(
            "CPU wheel links multiple OpenMP runtimes across native and operator DSOs: "
            + ", ".join(sorted(all_families))
        )


def _safe_installed_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise BuildError("operator catalog installed_relative_path is invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or PureWindowsPath(value).drive
        or str(path) != value
        or value.endswith("/")
        or any(ord(character) < 32 for character in value)
        or (path.parts and path.parts[0].endswith(":"))
    ):
        raise BuildError("operator catalog installed_relative_path is not relocatable")
    return path


def _audit_operator_catalog(
    archive: zipfile.ZipFile, catalog_info: zipfile.ZipInfo
) -> tuple[list[tuple[zipfile.ZipInfo, dict[str, object]]], dict[str, object]]:
    try:
        catalog = json.loads(archive.read(catalog_info))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BuildError(f"operator catalog is invalid JSON: {error}") from error
    try:
        catalog = validate_generated_operator_catalog(catalog)
    except ValueError as error:
        raise BuildError(
            f"operator catalog canonical validation failed: {error}"
        ) from error
    root = PurePosixPath(catalog_info.filename).parent
    records = []
    expected_members = {catalog_info.filename}
    archive_members = {info.filename: info for info in archive.infolist()}
    for artifact in catalog["artifacts"]:
        relative = _safe_installed_path(artifact.get("installed_relative_path"))
        member = str(root / relative)
        expected_members.add(member)
        info = archive_members.get(member)
        if info is None or info.is_dir():
            raise BuildError(f"operator catalog binary is missing: {member}")
        checksum = artifact.get("checksum")
        actual = "sha256:" + hashlib.sha256(archive.read(info)).hexdigest()
        if checksum != actual:
            raise BuildError(f"operator catalog checksum mismatch for {member}")
        records.append((info, artifact))
        for auxiliary in artifact.get("auxiliary_files", ()):
            auxiliary_relative = _safe_installed_path(
                auxiliary.get("installed_relative_path")
            )
            auxiliary_member = str(root / auxiliary_relative)
            expected_members.add(auxiliary_member)
            auxiliary_info = archive_members.get(auxiliary_member)
            if auxiliary_info is None or auxiliary_info.is_dir():
                raise BuildError(
                    f"operator catalog auxiliary file is missing: {auxiliary_member}"
                )
            auxiliary_actual = (
                "sha256:" + hashlib.sha256(archive.read(auxiliary_info)).hexdigest()
            )
            if auxiliary.get("checksum") != auxiliary_actual:
                raise BuildError(
                    "operator catalog auxiliary checksum mismatch for "
                    f"{auxiliary_member}"
                )
    actual_members = {
        info.filename
        for info in archive.infolist()
        if not info.is_dir() and PurePosixPath(info.filename).is_relative_to(root)
    }
    unexpected = sorted(actual_members - expected_members)
    if unexpected:
        raise BuildError(
            f"operator package data contains files absent from the catalog: {unexpected[0]}"
        )
    return records, catalog


def _normalized_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _audit_release_metadata(archive: zipfile.ZipFile) -> tuple[dict, str]:
    metadata_members = [
        info
        for info in archive.infolist()
        if not info.is_dir() and info.filename.endswith(".dist-info/METADATA")
    ]
    if len(metadata_members) != 1:
        raise BuildError("wheel must contain exactly one .dist-info/METADATA file")
    message = BytesParser(policy=default).parsebytes(archive.read(metadata_members[0]))
    distribution = str(message.get("Name", ""))
    version = str(message.get("Version", ""))
    if not distribution or not version:
        raise BuildError("wheel METADATA must declare Name and Version")

    descriptor_members = [
        info
        for info in archive.infolist()
        if not info.is_dir()
        and (
            info.filename == "symmetrix/_backend_cpu.json"
            or (
                info.filename.startswith("symmetrix_backend_")
                and info.filename.endswith("/backend.json")
            )
        )
    ]
    if len(descriptor_members) != 1:
        raise BuildError("wheel must contain exactly one backend descriptor")
    try:
        descriptor = json.loads(archive.read(descriptor_members[0]))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BuildError(f"backend descriptor is invalid JSON: {error}") from error
    if _normalized_distribution(str(descriptor.get("distribution", ""))) != (
        _normalized_distribution(distribution)
    ):
        raise BuildError("backend descriptor distribution does not match METADATA")
    if descriptor.get("frontend_version") != version:
        raise BuildError("backend descriptor frontend_version does not match METADATA")
    for field in ("source_content_sha256", "native_source_content_sha256"):
        if _SHA256.fullmatch(str(descriptor.get(field, ""))) is None:
            raise BuildError(f"backend descriptor has no complete {field}")
    if _COMMIT.fullmatch(str(descriptor.get("source_commit", ""))) is None:
        raise BuildError("backend descriptor has no complete source_commit")
    if descriptor.get("source_dirty") is not False:
        raise BuildError("release backend descriptor must record a clean source tree")

    requirements = [str(value) for value in message.get_all("Requires-Dist", [])]
    exact_frontend = re.compile(
        rf"^symmetrix-xl\s*==\s*{re.escape(version)}(?:\s*;|$)", re.IGNORECASE
    )
    is_addon = descriptor_members[0].filename != "symmetrix/_backend_cpu.json"
    if is_addon and not any(exact_frontend.match(value) for value in requirements):
        raise BuildError(f"backend wheel must require symmetrix-xl=={version} exactly")
    if not is_addon and any(
        _normalized_distribution(value.split()[0].split("=", 1)[0]) == "symmetrix-xl"
        for value in requirements
    ):
        raise BuildError("base wheel must not depend on itself")
    return descriptor, version


def audit_wheel(path: Path) -> None:
    findings: list[str] = []
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as error:
        raise BuildError(f"cannot inspect wheel {path}: {error}") from error

    with archive:
        cpu_operators: list[zipfile.ZipInfo] = []
        cpu_package_roots: set[PurePosixPath] = set()
        descriptor = None
        try:
            descriptor, _ = _audit_release_metadata(archive)
        except BuildError as error:
            findings.append(str(error))
        for info in archive.infolist():
            member = PurePosixPath(info.filename)
            lowered = {part.lower() for part in member.parts}
            forbidden = sorted(lowered & _FORBIDDEN_COMPONENTS)
            forbidden_name = bool(_FORBIDDEN_FILE_RE.search(member.name))
            if forbidden:
                findings.append(
                    f"{info.filename}: forbidden archive component {forbidden[0]!r}"
                )
            if forbidden_name:
                findings.append(f"{info.filename}: forbidden release-only filename")
            if info.is_dir() or forbidden or forbidden_name:
                continue
            content = archive.read(info)
            for label, marker, pattern in _SENSITIVE_CONTENT:
                if marker in content and pattern.search(content):
                    findings.append(f"{info.filename}: contains {label}")

        catalogs = [
            info
            for info in archive.infolist()
            if info.filename.endswith(_OPERATOR_CATALOG)
        ]
        if len(catalogs) > 1:
            findings.append("wheel contains multiple generated operator catalogs")
        for catalog_info in catalogs:
            try:
                records, catalog = _audit_operator_catalog(archive, catalog_info)
            except BuildError as error:
                findings.append(str(error))
                continue
            if descriptor is not None:
                expected_source = "sha256:" + str(
                    descriptor.get("source_content_sha256", "")
                )
                for _, artifact in records:
                    if (
                        artifact["build_provenance"]["source_content_sha256"]
                        != expected_source
                    ):
                        findings.append(
                            "operator catalog source_content_sha256 does not match "
                            "backend descriptor"
                        )
            host_records = [
                info
                for info, artifact in records
                if artifact["identity"]["backend"] == "host"
            ]
            if host_records:
                cpu_package_roots.add(
                    PurePosixPath(catalog_info.filename).parent.parent
                )
                cpu_operators.extend(host_records)
        if cpu_operators:
            with tempfile.TemporaryDirectory(
                prefix=".symmetrix-wheel-audit-", dir=path.parent
            ) as temporary:
                root = Path(temporary)
                native_infos = [
                    info
                    for info in archive.infolist()
                    if not info.is_dir()
                    and PurePosixPath(info.filename).parent in cpu_package_roots
                    and PurePosixPath(info.filename).name.startswith("_native")
                    and (
                        ".so" in PurePosixPath(info.filename).name
                        or PurePosixPath(info.filename).suffix in {".dylib", ".dll"}
                    )
                ]
                if len(native_infos) != 1:
                    findings.append(
                        "CPU operator wheel must contain exactly one native extension"
                    )
                dependencies_by_member = {}
                operator_names = {info.filename for info in cpu_operators}
                for index, info in enumerate([*cpu_operators, *native_infos]):
                    extracted = root / f"{index}-{PurePosixPath(info.filename).name}"
                    extracted.write_bytes(archive.read(info))
                    try:
                        dependencies_by_member[info.filename] = (
                            audit_cpu_operator_dependencies(extracted)
                            if info.filename in operator_names
                            else _dynamic_dependencies(extracted)
                        )
                    except BuildError as error:
                        findings.append(f"{info.filename}: {error}")
                if len(native_infos) == 1 and len(dependencies_by_member) == (
                    len(cpu_operators) + 1
                ):
                    try:
                        audit_cpu_openmp_runtime_set(
                            dependencies_by_member,
                            native_member=native_infos[0].filename,
                        )
                    except BuildError as error:
                        findings.append(str(error))

    if findings:
        detail = "\n".join(f"- {finding}" for finding in findings[:40])
        if len(findings) > 40:
            detail += f"\n- ... and {len(findings) - 40} more"
        raise BuildError(f"wheel content audit failed for {path.name}:\n{detail}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", nargs="+", type=Path)
    args = parser.parse_args(argv)
    for wheel in args.wheel:
        try:
            audit_wheel(wheel)
        except BuildError as error:
            parser.exit(1, f"error: {error}\n")
        print(f"wheel content audit passed: {wheel.name}")
    return 0
