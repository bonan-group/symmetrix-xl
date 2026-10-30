"""Keep a CPU wheel's original GNU runtime in one canonical private closure.

This is a post-auditwheel step, not an import/preload mechanism. Native runtime
admission remains responsible for rejecting an incompatible prior process stack.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile

from .command import BuildError

_CANONICAL = "libgomp.so.1"
_MANIFEST = "cpu-runtime.json"
_RUNTIME = re.compile(r"lib(?:gomp|omp|iomp5)(?:-[0-9a-f]+)?\.so(?:\.|$)", re.I)
_HASH = re.compile(r"[0-9a-f]{64}")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or ":" in name
        or str(path) != name.rstrip("/")
    ):
        raise BuildError(f"unsafe wheel member: {name!r}")
    return name


def _read_wheel(wheel: Path) -> dict[str, bytes]:
    files = {}
    with zipfile.ZipFile(wheel) as archive:
        seen = set()
        for info in archive.infolist():
            name = _safe_name(info.filename)
            if name.rstrip("/") in seen:
                raise BuildError(f"duplicate wheel member: {name}")
            seen.add(name.rstrip("/"))
            mode = stat.S_IFMT(info.external_attr >> 16)
            if mode not in {0, stat.S_IFREG, stat.S_IFDIR}:
                raise BuildError(f"nonregular wheel member: {name}")
            if not info.is_dir():
                files[name] = archive.read(info)
    # A regular file cannot also be an ancestor directory during extraction.
    for name in files:
        if any(str(parent) in files for parent in PurePosixPath(name).parents):
            raise BuildError(f"wheel file/directory collision: {name}")
    _verify_record(files)
    return files


def _dist_info(files: dict[str, bytes]) -> str:
    roots = [n.rsplit("/", 1)[0] for n in files if n.endswith(".dist-info/METADATA")]
    if len(roots) != 1:
        raise BuildError("CPU runtime repair requires exactly one dist-info METADATA")
    return roots[0]


def _verify_record(files: dict[str, bytes]) -> None:
    record = _dist_info(files) + "/RECORD"
    if record not in files:
        raise BuildError("wheel RECORD is missing")
    signatures = {record + suffix for suffix in (".jws", ".p7s")}
    rows = list(csv.reader(io.StringIO(files[record].decode())))
    if any(len(row) != 3 for row in rows):
        raise BuildError("malformed wheel RECORD row")
    names = [row[0] for row in rows]
    if (
        len(set(names)) != len(names)
        or set(names) - signatures != set(files) - signatures
    ):
        raise BuildError("wheel RECORD membership mismatch")
    for name, digest, size in rows:
        if name in signatures:
            continue
        if name == record:
            if digest or size:
                raise BuildError("wheel RECORD must not hash itself")
            continue
        content = files[name]
        expected = "sha256=" + base64.urlsafe_b64encode(
            bytes.fromhex(_sha(content))
        ).decode().rstrip("=")
        if digest != expected or size != str(len(content)):
            raise BuildError(f"wheel RECORD hash/size mismatch: {name}")


def _descriptor(files: dict[str, bytes]) -> tuple[str, dict, str]:
    descriptors = [
        n
        for n in files
        if n == "symmetrix/_backend_cpu.json"
        or (n.startswith("symmetrix_backend_") and n.endswith("/backend.json"))
    ]
    if len(descriptors) != 1:
        raise BuildError("CPU runtime repair requires exactly one backend descriptor")
    descriptor = json.loads(files[descriptors[0]])
    if descriptor.get("backend") != "cpu":
        raise BuildError("canonical runtime repair is CPU-only")
    for key in ("source_content_sha256", "native_source_content_sha256"):
        if not _HASH.fullmatch(str(descriptor.get(key, ""))):
            raise BuildError(f"CPU descriptor has invalid {key}")
    package = str(PurePosixPath(descriptors[0]).parent)
    natives = [
        n
        for n, c in files.items()
        if str(PurePosixPath(n).parent) == package
        and PurePosixPath(n).name.startswith("_native")
        and c.startswith(b"\x7fELF")
    ]
    if len(natives) != 1:
        raise BuildError("CPU wheel must contain exactly one native ELF extension")
    return descriptors[0], descriptor, natives[0]


def _run(*args: str) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode:
        raise BuildError(f"{args[0]} failed: {result.stderr.strip()}")
    return result.stdout


def _elf(path: Path) -> dict:
    dynamic = _run("readelf", "-dW", str(path))
    anchors = {"GOMP_parallel", "omp_get_num_threads", "__kmpc_fork_call"}
    defines_runtime = False
    for line in _run("readelf", "--dyn-syms", "--wide", str(path)).splitlines():
        fields = line.split()
        if (
            len(fields) >= 8
            and fields[6] != "UND"
            and fields[7].split("@", 1)[0] in anchors
        ):
            defines_runtime = True
    return {
        "needed": re.findall(r"\(NEEDED\).*?\[([^]]+)\]", dynamic),
        "soname": re.findall(r"\(SONAME\).*?\[([^]]+)\]", dynamic),
        "rpath": re.findall(r"\((?:RPATH|RUNPATH)\).*?\[([^]]+)\]", dynamic),
        "defines_runtime": defines_runtime,
    }


def _code_symbols(path: Path) -> tuple[str, str]:
    # Compare raw .text bytes without materializing a large readelf hex dump.
    sections = _run("readelf", "-SW", str(path))
    match = re.search(
        r"\[\s*\d+\]\s+\.text\s+\S+\s+\S+\s+([0-9a-f]+)\s+([0-9a-f]+)", sections
    )
    if match is None:
        raise BuildError(f"CPU runtime closure ELF has no .text: {path.name}")
    offset, size = (int(v, 16) for v in match.groups())
    with path.open("rb") as stream:
        stream.seek(offset)
        text = stream.read(size)
    symbols = []
    for line in _run("readelf", "--dyn-syms", "--wide", str(path)).splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[0].rstrip(":").isdigit():
            # patchelf can reorder section headers without changing symbols.
            # Preserve symbol values, ABI/version names and definedness, not a
            # layout-only section index or symbol-table enumeration order.
            section = "DEFINED" if fields[6].isdigit() else fields[6]
            symbols.append([*fields[1:6], section, fields[7]])
    return _sha(text), _sha(json.dumps(sorted(symbols)).encode())


def _catalog_files(files: dict[str, bytes]) -> set[str]:
    owned = set()
    for name, data in files.items():
        if not name.endswith("/_operator_artifacts/catalog.json"):
            continue
        root = PurePosixPath(name).parent
        for entry in json.loads(data)["artifacts"]:
            for item in (entry, *entry.get("auxiliary_files", ())):
                relative = _safe_name(item["installed_relative_path"])
                member = str(root / relative)
                if member not in files or item["checksum"] != "sha256:" + _sha(
                    files[member]
                ):
                    raise BuildError(f"operator catalog checksum mismatch: {member}")
                owned.add(member)
        owned.add(name)
    return owned


def _extract_elfs(files: dict[str, bytes], root: Path) -> dict[str, dict]:
    elfs = {}
    for name, data in files.items():
        if data.startswith(b"\x7fELF"):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            elfs[name] = _elf(path)
    return elfs


def _closure(
    elfs: dict[str, dict], native: str, runtime_name: str
) -> tuple[str, list[str]]:
    runtimes = [
        name
        for name, e in elfs.items()
        if e["defines_runtime"]
        or _RUNTIME.match(PurePosixPath(name).name)
        or any(_RUNTIME.match(s) for s in e["soname"])
    ]
    if len(runtimes) != 1:
        raise BuildError(
            "CPU wheel must contain exactly one physical GNU runtime; duplicate/alias runtime"
        )
    runtime = runtimes[0]
    parent = PurePosixPath(runtime).parent
    if len(parent.parts) != 1 or not parent.name.endswith(".libs"):
        raise BuildError("GNU runtime must be in the private wheel .libs directory")
    if PurePosixPath(runtime).name != runtime_name or elfs[runtime]["soname"] != [
        runtime_name
    ]:
        raise BuildError("GNU runtime physical filename/SONAME mismatch")
    consumers = []
    for name, e in elfs.items():
        dependencies = [n for n in e["needed"] if _RUNTIME.match(PurePosixPath(n).name)]
        if not dependencies:
            continue
        if dependencies != [runtime_name]:
            raise BuildError(f"inconsistent/external OpenMP dependency: {name}")
        consumers.append(name)
        search = []
        for rpath in e["rpath"]:
            for directory in rpath.split(":"):
                if directory != "$ORIGIN" and not directory.startswith("$ORIGIN/"):
                    raise BuildError(
                        f"CPU runtime consumer requires private origin RPATH: {name}"
                    )
                resolved = posixpath.normpath(
                    directory.replace("$ORIGIN", str(PurePosixPath(name).parent))
                )
                if resolved.startswith(("/", "../")) or resolved == "..":
                    raise BuildError(f"CPU runtime RPATH escapes wheel: {name}")
                search.append(str(PurePosixPath(resolved) / runtime_name))
        if runtime not in search:
            raise BuildError(f"CPU runtime consumer missing private RPATH: {name}")
    blas = [n for n in consumers if PurePosixPath(n).name.startswith("libopenblas")]
    if native not in consumers or len(blas) != 1:
        raise BuildError(
            "CPU native and one dynamic OpenMP OpenBLAS must own the runtime"
        )
    return runtime, sorted(consumers)


def _rewrite_record(files: dict[str, bytes]) -> None:
    name = _dist_info(files) + "/RECORD"
    for suffix in (".jws", ".p7s"):
        files.pop(name + suffix, None)
    rows = []
    for member, content in sorted(files.items()):
        digest = "sha256=" + base64.urlsafe_b64encode(
            bytes.fromhex(_sha(content))
        ).decode().rstrip("=")
        rows.append(
            [member, digest, str(len(content))] if member != name else [name, "", ""]
        )
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    files[name] = output.getvalue().encode()


def _update_sbom(files: dict[str, bytes], changed: list[str]) -> None:
    name = _dist_info(files) + "/sboms/auditwheel.cdx.json"
    if name not in files:
        raise BuildError("CPU repaired wheel is missing auditwheel SBOM provenance")
    sbom = json.loads(files[name])
    # Preserve RPM/source ancestry, but obsolete file-level evidence cannot survive
    # the name rewrite. Refuse unfamiliar file SBOMs rather than silently lie.
    if any(c.get("type") == "file" for c in sbom.get("components", ())):
        raise BuildError("preexisting file-level SBOM requires explicit reconciliation")
    sbom.setdefault("metadata", {}).setdefault("tools", []).append(
        {"name": "symmetrix-cpu-runtime-repair", "version": "1"}
    )
    for member in changed:
        sbom.setdefault("components", []).append(
            {
                "type": "file",
                "bom-ref": "symmetrix-cpu-runtime:" + member,
                "name": member,
                "hashes": [{"alg": "SHA-256", "content": _sha(files[member])}],
            }
        )
    files[name] = (json.dumps(sbom, indent=2, sort_keys=True) + "\n").encode()


def canonicalize_cpu_gomp_wheel(
    wheel: Path, runtime: Path, expected_runtime_sha256: str, output_dir: Path
) -> Path:
    """Produce a NEW wheel; never patch an installed environment or the input."""
    wheel, runtime, output_dir = Path(wheel), Path(runtime), Path(output_dir)
    if not _HASH.fullmatch(expected_runtime_sha256):
        raise BuildError("expected runtime SHA-256 must be complete")
    original_bytes = runtime.read_bytes() if runtime.is_file() else b""
    if _sha(original_bytes) != expected_runtime_sha256:
        raise BuildError("original build runtime SHA-256 mismatch or missing file")
    if shutil.which("readelf") is None or shutil.which("patchelf") is None:
        raise BuildError("CPU runtime repair requires readelf and patchelf")
    if not original_bytes.startswith(b"\x7fELF") or _elf(runtime)["soname"] != [
        _CANONICAL
    ]:
        raise BuildError("original GNU runtime must be ELF with canonical SONAME")
    header = _run("readelf", "-hW", str(runtime))
    if "Advanced Micro Devices X86-64" not in header or "DYN" not in header:
        raise BuildError("CPU runtime must be an x86-64 shared ELF")
    files = _read_wheel(wheel)
    with zipfile.ZipFile(wheel) as archive:
        attributes = {info.filename: info.external_attr for info in archive.infolist()}
    _, descriptor, native = _descriptor(files)
    owned = _catalog_files(files)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / wheel.name
    if output.resolve() == wheel.resolve() or output.exists():
        raise BuildError("CPU runtime output must be new and distinct from input")
    with tempfile.TemporaryDirectory(
        prefix=".cpu-runtime-", dir=output_dir
    ) as directory:
        root = Path(directory)
        elfs = _extract_elfs(files, root)
        candidates = [
            n
            for n in elfs
            if re.fullmatch(
                r"libgomp-[0-9a-f]{8,64}\.so(?:\.[0-9]+)*", PurePosixPath(n).name
            )
        ]
        if len(candidates) != 1:
            raise BuildError("expected exactly one auditwheel-hashed GNU runtime")
        old = candidates[0]
        old_name = PurePosixPath(old).name
        prefix = old_name.split("-", 1)[1].split(".", 1)[0]
        if not expected_runtime_sha256.startswith(prefix):
            raise BuildError(
                "auditwheel runtime ancestry hash differs from build metadata"
            )
        _, consumers = _closure(elfs, native, old_name)
        if set(consumers) & owned:
            raise BuildError(
                "affected catalog-owned runtime consumer: rebuild before catalog generation"
            )
        if _code_symbols(root / old) != _code_symbols(runtime):
            raise BuildError(
                "vendored runtime code/symbols differ from original build runtime"
            )
        for name in consumers:
            path = root / name
            before = _code_symbols(path)
            _run("patchelf", "--replace-needed", old_name, _CANONICAL, str(path))
            after = _elf(path)
            if (
                before != _code_symbols(path)
                or after["rpath"] != elfs[name]["rpath"]
                or after["soname"] != elfs[name]["soname"]
            ):
                raise BuildError(
                    f"runtime repair changed code/symbols/search policy: {name}"
                )
            if after["needed"] != [
                _CANONICAL if n == old_name else n for n in elfs[name]["needed"]
            ]:
                raise BuildError(f"runtime repair changed unrelated dependency: {name}")
            files[name] = path.read_bytes()
        new = str(PurePosixPath(old).parent / _CANONICAL)
        files.pop(old)
        files[new] = original_bytes
        manifest = {
            "schema_version": 1,
            "runtime_member": new,
            "runtime_sha256": expected_runtime_sha256,
            "native_member": native,
            "consumers": consumers,
            "source_content_sha256": descriptor["source_content_sha256"],
            "native_source_content_sha256": descriptor["native_source_content_sha256"],
        }
        files[_dist_info(files) + "/" + _MANIFEST] = (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode()
        _update_sbom(files, [*consumers, new])
        _rewrite_record(files)
        staged = root / "output.whl"
        with zipfile.ZipFile(
            staged, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for name, content in sorted(files.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = attributes.get(
                    name,
                    attributes[old] if name == new else (stat.S_IFREG | 0o644) << 16,
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, content)
        audit_cpu_wheel_runtime(staged)
        # Atomic no-clobber publication: failures never leave a candidate wheel.
        os.link(staged, output)
    return output


def audit_cpu_wheel_runtime(wheel: Path, *, required: bool = True) -> None:
    """Validate physical canonical closure, original full hash and all ELF edges."""
    files = _read_wheel(Path(wheel))
    manifest_name = _dist_info(files) + "/" + _MANIFEST
    if manifest_name not in files:
        if required:
            raise BuildError("CPU wheel missing canonical runtime manifest")
        return
    manifest = json.loads(files[manifest_name])
    _, descriptor, native = _descriptor(files)
    if manifest.get("schema_version") != 1 or manifest.get("native_member") != native:
        raise BuildError("CPU runtime manifest native/schema mismatch")
    for key in ("source_content_sha256", "native_source_content_sha256"):
        if manifest.get(key) != descriptor[key]:
            raise BuildError("CPU runtime manifest source provenance mismatch")
    owned = _catalog_files(files)
    with tempfile.TemporaryDirectory(
        prefix=".cpu-runtime-audit-", dir=Path(wheel).parent
    ) as directory:
        elfs = _extract_elfs(files, Path(directory))
        runtime, consumers = _closure(elfs, native, _CANONICAL)
    if (
        manifest.get("runtime_member") != runtime
        or manifest.get("consumers") != consumers
    ):
        raise BuildError("CPU runtime manifest dependency closure mismatch")
    expected = manifest.get("runtime_sha256", "")
    if not _HASH.fullmatch(str(expected)) or _sha(files[runtime]) != expected:
        raise BuildError("CPU runtime original SHA-256 hash mismatch")
    if set(consumers) & owned:
        raise BuildError("affected catalog-owned runtime consumer")
    sbom = json.loads(files[_dist_info(files) + "/sboms/auditwheel.cdx.json"])
    hashed_members = set()
    for component in sbom.get("components", ()):
        if component.get("type") == "file":
            name = component.get("name")
            if name not in files or component.get("hashes") != [
                {"alg": "SHA-256", "content": _sha(files[name])}
            ]:
                raise BuildError("CPU runtime SBOM file hash mismatch")
            hashed_members.add(name)
    if not {runtime, *consumers}.issubset(hashed_members):
        raise BuildError("CPU runtime SBOM missing final payload hashes")


def _build_runtime(build_root: Path, descriptor: dict) -> tuple[Path, str]:
    headers = list(build_root.rglob("generated/openmp_build_metadata.hpp"))
    if len(headers) != 1:
        raise BuildError(
            "build root must contain exactly one generated OpenMP metadata header"
        )
    text = headers[0].read_text()
    if not re.search(r"\bopenmp_enabled\s*=\s*(?:1|true)\s*;", text):
        raise BuildError("CPU build metadata does not enable OpenMP")
    values = {}
    for field in ("path", "sha256"):
        match = re.search(
            r"\bexpected_openmp_runtime_" + field + r'\s*=\s*("(?:[^"\\]|\\.)*")\s*;',
            text,
        )
        if match is None:
            raise BuildError(f"build runtime metadata is missing {field}")
        values[field] = json.loads(match.group(1))
    identity = (headers[0].parent / "build_metadata.hpp").read_text()
    backend = re.search(r'#define SYMMETRIX_BUILD_BACKEND "([^"]+)"', identity)
    if backend is None or backend.group(1) not in {"cpu", "cpu-avx512"}:
        raise BuildError("generated build identity is not a CPU backend")
    for field in ("source_content_sha256", "native_source_content_sha256"):
        match = re.search(
            r"#define SYMMETRIX_BUILD_" + field.upper() + r' "([^"]+)"', identity
        )
        if match is None or match.group(1) != descriptor[field]:
            raise BuildError("wheel does not match generated build source metadata")
    path = Path(values["path"])
    if not path.is_absolute():
        raise BuildError("build runtime path must be absolute")
    return path, values["sha256"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--build-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        _, descriptor, _ = _descriptor(_read_wheel(args.wheel))
        runtime, digest = _build_runtime(args.build_root, descriptor)
        result = canonicalize_cpu_gomp_wheel(
            args.wheel, runtime, digest, args.output_dir
        )
    except (BuildError, OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"CPU canonical runtime wheel: {result}")
    return 0
