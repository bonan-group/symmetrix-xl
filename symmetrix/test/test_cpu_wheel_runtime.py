"""Canonical CPU runtime repair must preserve provenance and reject open closure."""

import base64
import csv
import hashlib
import importlib
import io
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import zipfile

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
BuildError = importlib.import_module("_symmetrix_build.command").BuildError


def _module():
    # A failing assertion, rather than collection failure, proves the missing API.
    assert (
        TOOLS / "_symmetrix_build/cpu_wheel_runtime.py"
    ).is_file(), "CPU runtime repair is missing"
    return importlib.import_module("_symmetrix_build.cpu_wheel_runtime")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _wheel(path, members, *, symlink=None):
    members = dict(members)
    (metadata,) = [n for n in members if n.endswith(".dist-info/METADATA")]
    record = metadata.rsplit("/", 1)[0] + "/RECORD"
    rows = []
    for name, data in sorted(members.items()):
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        )
        rows.append([name, "sha256=" + digest, str(len(data))])
    rows.append([record, "", ""])
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    members[record] = output.getvalue().encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            info = zipfile.ZipInfo(name)
            info.external_attr = (
                (stat.S_IFLNK | 0o777)
                if name == symlink
                else (stat.S_IFREG | (0o755 if data.startswith(b"\x7fELF") else 0o644))
            ) << 16
            archive.writestr(info, data)
    return path


@pytest.fixture(scope="module")
def elf_inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("cpu-runtime-elf")
    cc = shutil.which("cc")
    patcher = shutil.which("patchelf")
    if not cc or not patcher:
        pytest.skip("requires standalone C compiler, readelf and patchelf")

    def compile_dso(name, source, *flags):
        src = root / (name + ".c")
        src.write_text(source)
        output = root / name
        subprocess.run(
            [cc, "-shared", "-fPIC", "-o", str(output), str(src), *flags],
            check=True,
            capture_output=True,
        )
        return output

    original = compile_dso(
        "libgomp.so.1",
        "int GOMP_parallel(void) { return 7; }",
        "-Wl,-soname,libgomp.so.1",
    )
    digest = _sha(original.read_bytes())
    private = "libgomp-" + digest[:8] + ".so.1"
    repaired = root / private
    shutil.copyfile(original, repaired)
    subprocess.run([patcher, "--set-soname", private, str(repaired)], check=True)
    blas_name = "libopenblas-12345678.so.0"
    blas = compile_dso(
        blas_name,
        "extern int GOMP_parallel(void); int cblas_dgemm(void) { return GOMP_parallel(); }",
        "-L" + str(root),
        "-Wl,--no-as-needed",
        "-l:" + private,
        "-Wl,-soname," + blas_name,
        "-Wl,-rpath,$ORIGIN",
    )
    native = compile_dso(
        "_native_cpu.so",
        "extern int GOMP_parallel(void); extern int cblas_dgemm(void); int call(void) { return GOMP_parallel()+cblas_dgemm(); }",
        "-L" + str(root),
        "-Wl,--no-as-needed",
        "-l:" + private,
        "-l:" + blas_name,
        "-Wl,-rpath,$ORIGIN/../symmetrix_xl.libs",
    )
    plugin = compile_dso("operator.so", "int operator_identity(void) { return 42; }")
    return {
        "root": root,
        "original": original,
        "digest": digest,
        "private": private,
        "runtime": repaired,
        "blas": blas,
        "native": native,
        "plugin": plugin,
        "patcher": patcher,
    }


@pytest.fixture
def wheel_inputs(tmp_path, elf_inputs):
    e = elf_inputs
    descriptor = {
        "backend": "cpu",
        "distribution": "symmetrix-xl",
        "frontend_version": "0.2.0",
        "architecture": "x86-64-v3",
        "source_content_sha256": "b" * 64,
        "native_source_content_sha256": "c" * 64,
        "source_commit": "a" * 40,
        "source_dirty": False,
    }
    members = {
        "symmetrix/_backend_cpu.json": json.dumps(descriptor).encode(),
        "symmetrix/_native_cpu.so": e["native"].read_bytes(),
        "symmetrix_xl.libs/" + e["private"]: e["runtime"].read_bytes(),
        "symmetrix_xl.libs/" + e["blas"].name: e["blas"].read_bytes(),
        "symmetrix/_operator_artifacts/operator.so": e["plugin"].read_bytes(),
        "symmetrix/_operator_artifacts/catalog.json": json.dumps(
            {
                "artifacts": [
                    {
                        "installed_relative_path": "operator.so",
                        "checksum": "sha256:" + _sha(e["plugin"].read_bytes()),
                    }
                ]
            }
        ).encode(),
        "symmetrix_xl-0.2.0.dist-info/METADATA": b"Name: symmetrix-xl\nVersion: 0.2.0\n",
        "symmetrix_xl-0.2.0.dist-info/WHEEL": b"Wheel-Version: 1.0\nTag: cp312-cp312-manylinux_2_28_x86_64\n",
        "symmetrix_xl-0.2.0.dist-info/sboms/auditwheel.cdx.json": json.dumps(
            {
                "bomFormat": "CycloneDX",
                "specVersion": "1.4",
                "version": 1,
                "metadata": {"tools": [{"name": "auditwheel", "version": "6.8.2"}]},
                "components": [
                    {
                        "type": "library",
                        "name": "libgomp",
                        "purl": "pkg:rpm/almalinux/libgomp@8.5.0",
                    }
                ],
            }
        ).encode(),
    }
    return tmp_path, members, e


def _repair(inputs, **kwargs):
    root, members, elf = inputs
    wheel = _wheel(
        root / "symmetrix_xl-0.2.0-cp312-cp312-manylinux_2_28_x86_64.whl", members
    )
    return _module().canonicalize_cpu_gomp_wheel(
        wheel,
        elf["original"],
        kwargs.pop("expected", elf["digest"]),
        root / "final",
        **kwargs,
    )


def test_repair_closes_all_edges_keeps_original_runtime_and_catalog(wheel_inputs):
    root, members, elf = wheel_inputs
    result = _repair(wheel_inputs)
    with zipfile.ZipFile(result) as archive:
        assert (
            archive.getinfo("symmetrix/_native_cpu.so").external_attr >> 16 & 0o777
            == 0o755
        )
        assert (
            archive.read("symmetrix_xl.libs/libgomp.so.1")
            == elf["original"].read_bytes()
        )
        assert "symmetrix_xl.libs/" + elf["private"] not in archive.namelist()
        for name in (
            "symmetrix/_operator_artifacts/operator.so",
            "symmetrix/_operator_artifacts/catalog.json",
            "symmetrix/_backend_cpu.json",
        ):
            assert archive.read(name) == members[name]
        for name in (
            "symmetrix/_native_cpu.so",
            "symmetrix_xl.libs/" + elf["blas"].name,
        ):
            path = root / Path(name).name
            path.write_bytes(archive.read(name))
            dynamic = subprocess.check_output(["readelf", "-dW", str(path)], text=True)
            assert "Shared library: [libgomp.so.1]" in dynamic
            assert elf["private"] not in dynamic
            assert "$ORIGIN" in dynamic
        record = list(
            csv.reader(
                io.StringIO(
                    archive.read("symmetrix_xl-0.2.0.dist-info/RECORD").decode()
                )
            )
        )
        assert {row[0] for row in record} == set(archive.namelist())
        for name, digest, size in record:
            if name.endswith("/RECORD"):
                assert digest == size == ""
            else:
                data = archive.read(name)
                assert digest == "sha256=" + base64.urlsafe_b64encode(
                    hashlib.sha256(data).digest()
                ).decode().rstrip("=")
                assert int(size) == len(data)
        sbom = json.loads(
            archive.read("symmetrix_xl-0.2.0.dist-info/sboms/auditwheel.cdx.json")
        )
        assert sbom["components"][0]["purl"] == "pkg:rpm/almalinux/libgomp@8.5.0"
        files = {c["name"]: c for c in sbom["components"] if c["type"] == "file"}
        assert files["symmetrix_xl.libs/libgomp.so.1"]["hashes"] == [
            {"alg": "SHA-256", "content": elf["digest"]}
        ]
    _module().audit_cpu_wheel_runtime(result)


def test_repair_supports_the_cpu_addon_private_namespace(wheel_inputs):
    root, members, elf = wheel_inputs
    new = {}
    for name, data in members.items():
        name = name.replace("symmetrix_xl-0.2.0", "symmetrix_xl_cpu_avx512-0.2.0")
        name = name.replace("symmetrix_xl.libs", "symmetrix_xl_cpu_avx512.libs")
        name = name.replace("symmetrix/", "symmetrix_backend_cpu_avx512/")
        name = name.replace("_backend_cpu.json", "backend.json").replace(
            "_native_cpu.so", "_native_cpu_avx512.so"
        )
        if name.endswith("/backend.json"):
            descriptor = json.loads(data)
            descriptor.update(
                distribution="symmetrix-xl-cpu-avx512", architecture="x86-64-v4"
            )
            data = json.dumps(descriptor).encode()
        elif name.endswith(".dist-info/METADATA"):
            data = b"Name: symmetrix-xl-cpu-avx512\nVersion: 0.2.0\nRequires-Dist: symmetrix-xl==0.2.0\n"
        elif name.endswith("/_native_cpu_avx512.so"):
            native = root / "addon-native.so"
            native.write_bytes(data)
            subprocess.run(
                [
                    elf["patcher"],
                    "--set-rpath",
                    "$ORIGIN/../symmetrix_xl_cpu_avx512.libs",
                    str(native),
                ],
                check=True,
            )
            data = native.read_bytes()
        new[name] = data
    result = _module().canonicalize_cpu_gomp_wheel(
        _wheel(root / "addon.whl", new),
        elf["original"],
        elf["digest"],
        root / "addon-final",
    )
    with zipfile.ZipFile(result) as archive:
        assert (
            archive.read("symmetrix_xl_cpu_avx512.libs/libgomp.so.1")
            == elf["original"].read_bytes()
        )
    _module().audit_cpu_wheel_runtime(result)


@pytest.mark.parametrize(
    "defect",
    [
        "hash",
        "alias",
        "hidden_runtime",
        "catalog",
        "rpath",
        "symlink",
        "traversal",
        "absolute",
        "runtime_soname",
        "other_family",
        "external_needed",
        "stale_record",
    ],
)
def test_repair_rejects_unsafe_or_inconsistent_closure(wheel_inputs, defect):
    root, members, elf = wheel_inputs
    if defect == "hash":
        with pytest.raises(BuildError, match="hash|SHA"):
            _repair(wheel_inputs, expected="0" * 64)
        return
    if defect == "alias":
        members["symmetrix_xl.libs/libgomp.so.1"] = elf["original"].read_bytes()
    elif defect == "hidden_runtime":
        path = root / "hidden.so"
        shutil.copyfile(elf["original"], path)
        subprocess.run(
            [elf["patcher"], "--set-soname", "hidden.so", str(path)], check=True
        )
        members["symmetrix_xl.libs/hidden.so"] = path.read_bytes()
    elif defect == "catalog":
        members["symmetrix/_operator_artifacts/operator.so"] = elf[
            "native"
        ].read_bytes()
    elif defect in {"rpath", "other_family", "external_needed"}:
        path = root / "mutated.so"
        path.write_bytes(members["symmetrix/_native_cpu.so"])
        args = (
            ["--remove-rpath"]
            if defect == "rpath"
            else [
                "--add-needed",
                "libomp.so.5" if defect == "other_family" else "libgomp-00000000.so.1",
            ]
        )
        subprocess.run([elf["patcher"], *args, str(path)], check=True)
        members["symmetrix/_native_cpu.so"] = path.read_bytes()
    elif defect == "traversal":
        members["../escape"] = b"no"
    elif defect == "absolute":
        members["/absolute"] = b"no"
    elif defect == "runtime_soname":
        other = root / "other.so"
        shutil.copyfile(elf["original"], other)
        subprocess.run(
            [elf["patcher"], "--set-soname", "wrong.so", str(other)], check=True
        )
        with pytest.raises(BuildError, match="SONAME|soname"):
            _module().canonicalize_cpu_gomp_wheel(
                _wheel(root / "input.whl", members),
                other,
                _sha(other.read_bytes()),
                root / "final",
            )
        return
    wheel = _wheel(
        root / "input.whl",
        members,
        symlink="symmetrix/_native_cpu.so" if defect == "symlink" else None,
    )
    if defect == "stale_record":
        with zipfile.ZipFile(wheel, "a") as archive:
            archive.writestr("unrecorded.txt", b"unrecorded")
    with pytest.raises(BuildError):
        _module().canonicalize_cpu_gomp_wheel(
            wheel, elf["original"], elf["digest"], root / "final"
        )
    assert not list((root / "final").glob("*.whl"))


def test_build_metadata_selects_actual_runtime_and_binds_source(wheel_inputs):
    root, members, elf = wheel_inputs
    generated = root / "build/fingerprint/generated"
    generated.mkdir(parents=True)
    (generated / "openmp_build_metadata.hpp").write_text(
        f'inline constexpr bool openmp_enabled = 1;\ninline constexpr const char* expected_openmp_runtime_path = "{elf["original"]}";\ninline constexpr const char* expected_openmp_runtime_sha256 = "{elf["digest"]}";\n'
    )
    (generated / "build_metadata.hpp").write_text(
        '#define SYMMETRIX_BUILD_BACKEND "cpu"\n#define SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256 "'
        + "b" * 64
        + '"\n#define SYMMETRIX_BUILD_NATIVE_SOURCE_CONTENT_SHA256 "'
        + "c" * 64
        + '"\n'
    )
    wheel = _wheel(root / "input.whl", members)
    assert (
        _module().main(
            [
                "--wheel",
                str(wheel),
                "--build-root",
                str(root / "build"),
                "--output-dir",
                str(root / "out"),
            ]
        )
        == 0
    )
    (generated / "build_metadata.hpp").write_text(
        '#define SYMMETRIX_BUILD_BACKEND "cuda"\n'
    )
    assert (
        _module().main(
            [
                "--wheel",
                str(wheel),
                "--build-root",
                str(root / "build"),
                "--output-dir",
                str(root / "wrong"),
            ]
        )
        != 0
    )


def test_repair_removes_invalidated_record_signatures(wheel_inputs):
    root, members, elf = wheel_inputs
    members["symmetrix_xl-0.2.0.dist-info/RECORD.jws"] = b"old signature"
    members["symmetrix_xl-0.2.0.dist-info/RECORD.p7s"] = b"old signature"
    with zipfile.ZipFile(_repair(wheel_inputs)) as archive:
        assert not any(
            n.endswith(("RECORD.jws", "RECORD.p7s")) for n in archive.namelist()
        )


def test_final_audit_rejects_runtime_payload_tampering(wheel_inputs):
    result = _repair(wheel_inputs)
    with zipfile.ZipFile(result) as archive:
        members = {
            n: archive.read(n) for n in archive.namelist() if not n.endswith("/RECORD")
        }
    members["symmetrix_xl.libs/libgomp.so.1"] += b"tampered"
    wheel = _wheel(result.parent / "tampered.whl", members)
    with pytest.raises(BuildError, match="hash|SHA"):
        _module().audit_cpu_wheel_runtime(wheel)


def test_repair_never_overwrites_input_or_existing_output(wheel_inputs):
    root, members, elf = wheel_inputs
    wheel = _wheel(root / "input.whl", members)
    before = wheel.read_bytes()
    with pytest.raises(BuildError, match="new|distinct"):
        _module().canonicalize_cpu_gomp_wheel(
            wheel, elf["original"], elf["digest"], root
        )
    assert wheel.read_bytes() == before
    out = root / "existing"
    out.mkdir()
    (out / wheel.name).write_bytes(b"prior result")
    with pytest.raises(BuildError, match="new|distinct"):
        _module().canonicalize_cpu_gomp_wheel(
            wheel, elf["original"], elf["digest"], out
        )
    assert (out / wheel.name).read_bytes() == b"prior result"


@pytest.mark.parametrize(
    "defect",
    [
        "sbom_missing_hash",
        "stale_needed",
        "missing_rpath",
        "missing_manifest",
        "manifest_source",
    ],
)
def test_final_audit_rejects_broken_final_evidence(wheel_inputs, defect):
    root, _, elf = wheel_inputs
    result = _repair(wheel_inputs)
    with zipfile.ZipFile(result) as archive:
        members = {
            n: archive.read(n) for n in archive.namelist() if not n.endswith("/RECORD")
        }
    metadata = "symmetrix_xl-0.2.0.dist-info/"
    if defect == "sbom_missing_hash":
        name = metadata + "sboms/auditwheel.cdx.json"
        sbom = json.loads(members[name])
        sbom["components"] = [c for c in sbom["components"] if c["type"] != "file"]
        members[name] = json.dumps(sbom).encode()
    elif defect in {"stale_needed", "missing_rpath"}:
        path = root / "broken-native.so"
        path.write_bytes(members["symmetrix/_native_cpu.so"])
        args = (
            ["--remove-rpath"]
            if defect == "missing_rpath"
            else ["--replace-needed", "libgomp.so.1", elf["private"]]
        )
        subprocess.run([elf["patcher"], *args, str(path)], check=True)
        members["symmetrix/_native_cpu.so"] = path.read_bytes()
    elif defect == "missing_manifest":
        del members[metadata + "cpu-runtime.json"]
    else:
        manifest = json.loads(members[metadata + "cpu-runtime.json"])
        manifest["source_content_sha256"] = "d" * 64
        members[metadata + "cpu-runtime.json"] = json.dumps(manifest).encode()
    wheel = _wheel(root / "broken-final.whl", members)
    with pytest.raises(BuildError):
        _module().audit_cpu_wheel_runtime(wheel)


def test_release_content_audit_requires_final_cpu_closure_when_requested(wheel_inputs):
    root, members, _ = wheel_inputs
    wheel = _wheel(root / "unrepaired.whl", members)
    audit = importlib.import_module("_symmetrix_build.wheel_audit")
    with pytest.raises(BuildError, match="canonical runtime manifest"):
        audit.audit_wheel(wheel, require_cpu_runtime=True)


@pytest.mark.parametrize(
    "defect", ["missing", "ambiguous", "disabled", "source", "relative_path"]
)
def test_metadata_rejects_unknown_or_wrong_build(wheel_inputs, defect):
    root, members, elf = wheel_inputs
    generated = root / "build/one/generated"
    generated.mkdir(parents=True)
    runtime = elf["original"] if defect != "relative_path" else Path("relative.so")
    (generated / "openmp_build_metadata.hpp").write_text(
        f'inline constexpr bool openmp_enabled = {0 if defect == "disabled" else 1};\ninline constexpr const char* expected_openmp_runtime_path = "{runtime}";\ninline constexpr const char* expected_openmp_runtime_sha256 = "{elf["digest"]}";\n'
    )
    (generated / "build_metadata.hpp").write_text(
        '#define SYMMETRIX_BUILD_BACKEND "cpu"\n#define SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256 "'
        + ("d" if defect == "source" else "b") * 64
        + '"\n#define SYMMETRIX_BUILD_NATIVE_SOURCE_CONTENT_SHA256 "'
        + "c" * 64
        + '"\n'
    )
    if defect == "missing":
        (generated / "openmp_build_metadata.hpp").unlink()
    elif defect == "ambiguous":
        other = root / "build/two/generated"
        other.mkdir(parents=True)
        shutil.copyfile(
            generated / "openmp_build_metadata.hpp", other / "openmp_build_metadata.hpp"
        )
    wheel = _wheel(root / "raw.whl", members)
    assert (
        _module().main(
            [
                "--wheel",
                str(wheel),
                "--build-root",
                str(root / "build"),
                "--output-dir",
                str(root / "out"),
            ]
        )
        == 1
    )
    assert not list((root / "out").glob("*.whl"))
