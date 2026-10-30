"""Exercise the same native runtime-identity policy used before Kokkos init."""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def identity_probe(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("native runtime identity regression requires a C++ compiler")
    root = Path(__file__).resolve().parents[2]
    work = tmp_path_factory.mktemp("native-openmp-identity")
    source = work / "probe.cpp"
    source.write_text(
        '#include "openmp_runtime_identity.hpp"\n'
        "#include <iostream>\n"
        "int main(int argc, char** argv) {\n"
        "  if (argc < 5) return 2;\n"
        "  symmetrix::runtime::OpenMPRuntimeSnapshot snapshot;\n"
        "  snapshot.loaded_runtime_path = argv[2];\n"
        "  for (int i=5; i<argc; ++i) snapshot.loaded_openmp_libraries.emplace_back(argv[i]);\n"
        "  const auto issues = symmetrix::runtime::openmp_runtime_issues(\n"
        '      snapshot, std::string(argv[1]) == "1", argv[3], argv[4]);\n'
        "  std::cout << issues.size() << '\\n';\n"
        "  for (const auto& issue : issues) std::cout << issue << '\\n';\n"
        "}\n"
    )
    binary = work / "probe"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(root / "symmetrix/source/cpp"),
            "-I",
            str(root / "libsymmetrix/source"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    def probe(loaded, build, expected_hash, *, mapped=None, enabled=True):
        lines = subprocess.check_output(
            [
                str(binary),
                str(int(enabled)),
                str(loaded),
                str(build),
                expected_hash,
                *map(str, [loaded] if mapped is None else mapped),
            ],
            text=True,
        ).splitlines()
        assert int(lines[0]) == len(lines) - 1
        return lines[1:]

    return probe


@pytest.mark.parametrize(
    "payload", [b"", b"abc", bytes(range(256)) * 513 + b"\x00tail"]
)
def test_relocated_runtime_matches_compiled_hash_without_build_path(
    identity_probe, tmp_path, payload
):
    loaded = tmp_path / "libgomp.so.1"
    loaded.write_bytes(payload)
    assert not identity_probe(
        loaded,
        tmp_path / "absent-build/libgomp.so.1",
        hashlib.sha256(payload).hexdigest(),
    )


def test_present_build_file_cannot_override_compiled_hash(identity_probe, tmp_path):
    loaded = tmp_path / "libgomp.so.1"
    loaded.write_bytes(b"exact recorded runtime")
    build = tmp_path / "build-libgomp.so.1"
    build.write_bytes(b"unrelated runtime now at historical path")
    assert not identity_probe(
        loaded, build, hashlib.sha256(loaded.read_bytes()).hexdigest().upper()
    )


def test_same_path_does_not_accept_changed_runtime_bytes(identity_probe, tmp_path):
    loaded = tmp_path / "libgomp.so.1"
    loaded.write_bytes(b"changed runtime")
    assert identity_probe(
        loaded, loaded, hashlib.sha256(b"recorded runtime").hexdigest()
    )


@pytest.mark.parametrize(
    "expected_hash",
    [
        "",
        "a" * 63,
        "a" * 65,
        "g" * 64,
        "sha256:" + "a" * 64,
        " " + "a" * 64,
        "a" * 64 + "\n",
        "é" * 64,
    ],
)
def test_invalid_compiled_hash_fails_even_at_build_path(
    identity_probe, tmp_path, expected_hash
):
    loaded = tmp_path / "libgomp.so.1"
    loaded.write_bytes(b"runtime")
    assert identity_probe(loaded, loaded, expected_hash)


@pytest.mark.parametrize("kind", ["missing", "directory", "fifo"])
def test_nonregular_or_missing_runtime_fails_closed(identity_probe, tmp_path, kind):
    digest = hashlib.sha256(b"runtime").hexdigest()
    loaded = tmp_path / f"libgomp-{digest[:8]}.so.1"
    if kind == "directory":
        loaded.mkdir()
    elif kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO runtime-path rejection requires mkfifo")
        os.mkfifo(loaded)
    assert identity_probe(loaded, loaded, digest)


def test_mismatched_canonical_runtime_fails(identity_probe, tmp_path):
    loaded = tmp_path / "libgomp.so.1"
    loaded.write_bytes(b"wrong runtime")
    assert identity_probe(
        loaded,
        tmp_path / "absent/libgomp.so.1",
        hashlib.sha256(b"expected runtime").hexdigest(),
    )


def test_duplicate_mappings_still_fail_for_exact_bytes(identity_probe, tmp_path):
    loaded = tmp_path / "libgomp.so.1"
    loaded.write_bytes(b"runtime")
    issues = identity_probe(
        loaded,
        loaded,
        hashlib.sha256(b"runtime").hexdigest(),
        mapped=[loaded, tmp_path / "other/libgomp.so.1"],
    )
    assert any("multiple OpenMP runtimes" in issue for issue in issues)


@pytest.mark.parametrize(
    "stem,prefix", [("libgomp", None), ("libomp", None), ("libgomp", "00000000")]
)
def test_existing_auditwheel_ancestry_requires_matching_stem_and_prefix(
    identity_probe, tmp_path, stem, prefix
):
    digest = hashlib.sha256(b"original runtime").hexdigest()
    loaded = tmp_path / f"{stem}-{prefix or digest[:8]}.so.1"
    loaded.write_bytes(b"auditwheel changes ELF metadata")
    issues = identity_probe(loaded, tmp_path / "absent/libgomp.so.1", digest)
    assert bool(issues) is not (stem == "libgomp" and prefix is None)


def test_auditwheel_prefix_cannot_replace_full_expected_hash(identity_probe, tmp_path):
    loaded = tmp_path / "libgomp-abcdef12.so.1"
    loaded.write_bytes(b"runtime")
    assert identity_probe(loaded, tmp_path / "absent/libgomp.so.1", "abcdef12")


def test_missing_symbol_owner_is_an_error(identity_probe, tmp_path):
    issues = identity_probe("", tmp_path / "absent/libgomp.so.1", "a" * 64, mapped=[])
    assert any("symbol owner is unavailable" in issue for issue in issues)


def test_disabled_openmp_is_not_applicable(identity_probe, tmp_path):
    assert not identity_probe("", tmp_path / "absent", "", mapped=[], enabled=False)
