import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPOSITORY_ROOT / "tools"
sys.path.insert(0, str(TOOLS_ROOT))

build_cli = importlib.import_module("_symmetrix_build.cli")
build_develop = importlib.import_module("_symmetrix_build.develop_build")
build_manifest = importlib.import_module("_symmetrix_build.manifest")
build_targets = importlib.import_module("_symmetrix_build.targets")
build_python = importlib.import_module("_symmetrix_build.python_build")
build_command = importlib.import_module("_symmetrix_build.command")

BuildError = build_command.BuildError
CommandResult = build_command.CommandResult
TargetManifest = build_manifest.TargetManifest
Toolchain = build_manifest.Toolchain
Architecture = build_targets.Architecture
FRONTEND_VERSION = build_python._frontend_version(REPOSITORY_ROOT / "symmetrix")


def _manifest(target="native", backend="cpu"):
    if backend == "cpu":
        architecture = Architecture(target, target, "NATIVE", target)
        host_target = target
        toolkit_root = ""
        toolkit_version = ""
    elif backend == "cuda":
        architecture = build_targets.cuda_architecture(target, "")
        host_target = "none"
        toolkit_root = "/usr/local/cuda"
        toolkit_version = "13.3"
    else:
        architecture = build_targets.hip_architecture(target, "")
        host_target = "none"
        toolkit_root = "/opt/rocm"
        toolkit_version = "7.0"
    return TargetManifest(
        backend=backend,
        architecture=architecture,
        toolchain=Toolchain(
            cxx="/usr/bin/c++",
            cxx_version="test compiler",
            host_cxx="/usr/bin/g++" if backend == "cuda" else "",
            cmake="/usr/bin/cmake",
            cmake_version="4.2.0",
            generator="Unix Makefiles",
            fortran="/usr/bin/gfortran",
            toolkit_root=toolkit_root,
            toolkit_version=toolkit_version,
        ),
        python_executable=sys.executable,
        python_abi=sys.implementation.cache_tag,
        host_target=host_target,
        blas_policy="openblas" if backend == "cpu" else "auto",
        blas_root="",
        device_probe="test",
        visible_devices=(target,),
    )


class LayoutRunner:
    def __init__(self, platlib, *, package="symmetrix", validation=None):
        self.platlib = Path(platlib)
        self.package = package
        self.validation = validation
        self.commands = []

    def run(self, args, **_kwargs):
        command = tuple(os.fspath(value) for value in args)
        self.commands.append(command)
        if command[1:2] == ("-c",) and "sysconfig.get_path" in command[2]:
            payload = {
                "platform_library": str(self.platlib),
                "package_directory": str(self.platlib / self.package),
                "extension_suffix": ".test.so",
                "frontend_version": FRONTEND_VERSION,
                "target_version": FRONTEND_VERSION,
            }
        elif command[1:2] == ("-c",) and self.validation is not None:
            payload = self.validation
        else:
            return CommandResult(command, 0, "", "")
        return CommandResult(command, 0, json.dumps(payload) + "\n", "")

    def run_logged(self, args, log_path, **_kwargs):
        command = tuple(os.fspath(value) for value in args)
        self.commands.append(command)
        Path(log_path).write_text("ok\n")
        return CommandResult(command, 0, "", "")


@pytest.fixture
def provenance():
    return {
        "revision": "1" * 40,
        "dirty": True,
        "content_sha256": "a" * 64,
        "native_content_sha256": "b" * 64,
    }


def _invocation(tmp_path, monkeypatch, provenance, manifest=None):
    manifest = manifest or _manifest()
    identity = build_develop.package_identity(manifest)
    platlib = tmp_path / "venv/lib/python/site-packages"
    (platlib / identity.package).mkdir(parents=True)
    runner = LayoutRunner(platlib, package=identity.package)
    monkeypatch.setattr(
        build_develop, "symmetrix_source_provenance", lambda *_args: provenance
    )
    invocation = build_develop.development_build_invocation(
        manifest,
        repo_root=REPOSITORY_ROOT,
        runner=runner,
        build_root=tmp_path / "build",
        jobs=3,
    )
    return invocation, runner


def test_develop_cli_and_direct_commands(tmp_path, monkeypatch, provenance):
    args = build_cli._parser().parse_args(
        ["develop", "--backend", "cpu", "--jobs", "3", "--dry-run"]
    )
    assert args.command == "develop"
    assert args.jobs == 3
    assert args.dry_run is True

    invocation, _ = _invocation(tmp_path, monkeypatch, provenance)
    configure, build, install = invocation.commands
    assert configure[:7] == (
        "/usr/bin/cmake",
        "-S",
        str(REPOSITORY_ROOT / "symmetrix"),
        "-B",
        str(invocation.build_directory),
        "-G",
        "Unix Makefiles",
    )
    assert "-DSYMMETRIX_ENABLE_DEVELOPMENT_RUNTIME_COMPONENT=ON" in configure
    assert f"-DSYMMETRIX_BUILD_SOURCE_CONTENT_SHA256={'a' * 64}" in configure
    assert build == (
        "/usr/bin/cmake",
        "--build",
        str(invocation.build_directory),
        "--target",
        "symmetrix_bindings",
    )
    assert install == (
        "/usr/bin/cmake",
        "--install",
        str(invocation.build_directory),
        "--component",
        "symmetrix_development_runtime",
        "--prefix",
        str(invocation.platform_library),
    )
    assert invocation.environment["CMAKE_BUILD_PARALLEL_LEVEL"] == "3"
    assert invocation.extension_path.name == "_native_cpu.test.so"
    assert invocation.descriptor_path.name == "_backend_cpu.json"


def test_develop_routes_accelerator_runtime_to_installed_backend_package(
    tmp_path, monkeypatch, provenance
):
    manifest = _manifest("sm120", "cuda")
    package = "symmetrix_backend_cuda13_sm120"
    platlib = tmp_path / "site-packages"
    (platlib / package).mkdir(parents=True)
    runner = LayoutRunner(platlib, package=package)
    monkeypatch.setattr(
        build_develop, "symmetrix_source_provenance", lambda *_args: provenance
    )

    invocation = build_develop.development_build_invocation(
        manifest,
        repo_root=REPOSITORY_ROOT,
        runner=runner,
        build_root=tmp_path / "build",
    )

    assert invocation.package_directory == platlib / package
    assert invocation.extension_path.name == "_native_cuda13_sm120.test.so"
    assert invocation.descriptor_path == platlib / package / "backend.json"
    assert f"-DSYMMETRIX_PYTHON_INSTALL_PACKAGE={package}" in invocation.commands[0]
    assert (
        "-DSYMMETRIX_DEVELOPMENT_DESCRIPTOR_SELECTOR=cuda13-sm120"
        in invocation.commands[0]
    )


@pytest.mark.parametrize(
    ("backend", "target"),
    (("cpu", "x86-64-v4"), ("cuda", "sm120"), ("hip", "gfx1151")),
)
def test_development_descriptor_renders_architecture_qualified_selector(
    tmp_path, backend, target
):
    manifest = _manifest(target, backend)
    identity = build_develop.package_identity(manifest)
    template = REPOSITORY_ROOT / "symmetrix/source/cpp/backend_development.json.in"
    rendered = tmp_path / f"{backend}.json"
    values = {
        "SYMMETRIX_BACKEND_ARCHITECTURE": identity.architecture,
        "SYMMETRIX_DEVELOPMENT_BACKEND": identity.backend,
        "CMAKE_CXX_COMPILER_ID": "GNU",
        "CMAKE_CXX_COMPILER_VERSION": "15.2.0",
        "SYMMETRIX_DISTRIBUTION_NAME": identity.distribution,
        "SYMMETRIX_FRONTEND_VERSION": FRONTEND_VERSION,
        "SYMMETRIX_PYTHON_MODULE_NAME": identity.module,
        "SYMMETRIX_BUILD_NATIVE_SOURCE_CONTENT_SHA256": "b" * 64,
        "SYMMETRIX_PYTHON_INSTALL_PACKAGE": identity.package,
        "SYMMETRIX_DEVELOPMENT_DESCRIPTOR_SELECTOR": identity.selector,
        "SYMMETRIX_BUILD_SOURCE_COMMIT": "1" * 40,
        "SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256": "a" * 64,
        "SYMMETRIX_BUILD_SOURCE_DIRTY": "0",
        "SYMMETRIX_TARGET_MANIFEST_FINGERPRINT": manifest.fingerprint,
        "SYMMETRIX_DEVELOPMENT_TOOLKIT": identity.toolkit,
    }
    script = tmp_path / f"render-{backend}.cmake"
    script.write_text(
        "\n".join(f'set({name} "{value}")' for name, value in values.items())
        + f'\nconfigure_file("{template}" "{rendered}" @ONLY)\n'
    )
    result = subprocess.run(
        (manifest.toolchain.cmake, "-P", str(script)),
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    descriptor = json.loads(rendered.read_text())
    assert descriptor["backend"] == identity.backend
    assert descriptor["selector"] == identity.selector
    assert descriptor["package"] == identity.package
    assert descriptor["module"] == identity.module


def test_develop_rejects_unsafe_or_mismatched_build_roots(
    tmp_path, monkeypatch, provenance
):
    manifest = _manifest()
    build_root = tmp_path / "build"
    selected = build_root / manifest.fingerprint
    selected.mkdir(parents=True)
    _manifest("x86-64-v4").write(selected / "symmetrix-target.json")
    platlib = tmp_path / "site-packages"
    (platlib / "symmetrix").mkdir(parents=True)
    runner = LayoutRunner(platlib)
    monkeypatch.setattr(
        build_develop, "symmetrix_source_provenance", lambda *_args: provenance
    )

    with pytest.raises(BuildError, match="build directory contains target"):
        build_develop.development_build_invocation(
            manifest,
            repo_root=REPOSITORY_ROOT,
            runner=runner,
            build_root=build_root,
        )

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    runner = LayoutRunner(foreign)
    with pytest.raises(BuildError, match="installed backend package is missing"):
        build_develop.development_build_invocation(
            manifest,
            repo_root=REPOSITORY_ROOT,
            runner=runner,
            build_root=tmp_path / "other-build",
        )


def test_development_component_is_runtime_only():
    cmake = (REPOSITORY_ROOT / "symmetrix/CMakeLists.txt").read_text()
    template = (
        REPOSITORY_ROOT / "symmetrix/source/cpp/backend_development.json.in"
    ).read_text()
    component = "COMPONENT symmetrix_development_runtime"
    assert cmake.count(component) == 3
    assert "install(TARGETS symmetrix_bindings" in cmake
    assert "RENAME _backend_cpu.json" in cmake
    assert "RENAME backend.json" in cmake
    assert "@SYMMETRIX_TARGET_MANIFEST_FINGERPRINT@" in template
    assert "@SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256@" in template
    assert "@SYMMETRIX_BUILD_NATIVE_SOURCE_CONTENT_SHA256@" in template
    assert "source/symmetrix" not in template


def test_initializer_provenance_uses_dedicated_generated_header():
    cmake = (REPOSITORY_ROOT / "symmetrix/CMakeLists.txt").read_text()
    initializer = (REPOSITORY_ROOT / "symmetrix/source/cpp/symmetrix.cpp").read_text()
    metadata = (
        REPOSITORY_ROOT / "symmetrix/source/cpp/build_metadata.hpp.in"
    ).read_text()

    assert "configure_file(\n  source/cpp/build_metadata.hpp.in" in cmake
    assert "set_property(SOURCE source/cpp/symmetrix.cpp" not in cmake
    assert '#include "build_metadata.hpp"' in initializer
    for definition in (
        "SYMMETRIX_PYTHON_MODULE",
        "SYMMETRIX_BUILD_SOURCE_COMMIT",
        "SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256",
        "SYMMETRIX_BUILD_NATIVE_SOURCE_CONTENT_SHA256",
        "SYMMETRIX_BUILD_FAST_MATH",
        "SYMMETRIX_FOURIER_D3_AVAILABLE",
    ):
        assert definition in metadata


def test_develop_runs_only_configure_binding_and_runtime_install(
    tmp_path, monkeypatch, provenance
):
    invocation, runner = _invocation(tmp_path, monkeypatch, provenance)
    invocation.extension_path.write_bytes(b"extension")
    invocation.descriptor_path.write_text("{}")

    build_develop.run_development_build(invocation, runner)

    assert runner.commands[-3:] == list(invocation.commands)
    assert all(
        "uv" not in command and "pip" not in command for command in runner.commands
    )


def test_develop_validation_requires_exact_fresh_provenance(
    tmp_path, monkeypatch, provenance
):
    invocation, _ = _invocation(tmp_path, monkeypatch, provenance)
    manifest = TargetManifest.read(invocation.manifest_path)
    selected = {
        "selector": "cpu",
        "descriptor_path": str(invocation.descriptor_path),
        "target_manifest_fingerprint": manifest.fingerprint,
        "source_content_sha256": provenance["content_sha256"],
        "native_source_content_sha256": provenance["native_content_sha256"],
    }
    build_info = {
        "backend": "cpu",
        "architecture": "native",
        "source_commit": provenance["revision"],
        "source_dirty": True,
        "source_content_sha256": provenance["content_sha256"],
        "native_source_content_sha256": provenance["native_content_sha256"],
    }
    record = {
        "extension": str(invocation.extension_path),
        "extension_sha256": "c" * 64,
        "selected_backend": selected,
        "build_info": build_info,
        "execution_space": "OpenMP",
        "device_sentinel": True,
        "native_module": "symmetrix._native_cpu",
    }
    runner = LayoutRunner(invocation.platform_library, validation=record)
    validated = build_develop.validate_development_runtime(invocation, runner)
    assert validated["status"] == "ok"
    assert validated["build_provenance"]["source_content_sha256"] == "a" * 64

    record["build_info"]["source_content_sha256"] = "d" * 64
    runner = LayoutRunner(invocation.platform_library, validation=record)
    with pytest.raises(BuildError, match="provenance mismatch: source hash"):
        build_develop.validate_development_runtime(invocation, runner)


@pytest.mark.parametrize(
    ("backend", "target", "expected_space"),
    (
        ("cpu", "x86-64-v4", "OpenMP"),
        ("cuda", "sm120", "Cuda"),
        ("hip", "gfx1151", "HIP"),
    ),
)
def test_develop_validation_rejects_wrong_execution_space(
    tmp_path, monkeypatch, provenance, backend, target, expected_space
):
    invocation, _ = _invocation(
        tmp_path, monkeypatch, provenance, _manifest(target, backend)
    )
    identity = invocation.identity
    manifest = TargetManifest.read(invocation.manifest_path)
    record = {
        "extension": str(invocation.extension_path),
        "extension_sha256": "c" * 64,
        "selected_backend": {
            "selector": identity.selector,
            "descriptor_path": str(invocation.descriptor_path),
            "target_manifest_fingerprint": manifest.fingerprint,
            "source_content_sha256": provenance["content_sha256"],
            "native_source_content_sha256": provenance["native_content_sha256"],
        },
        "build_info": {
            "backend": backend,
            "architecture": identity.architecture,
            "source_commit": provenance["revision"],
            "source_dirty": True,
            "source_content_sha256": provenance["content_sha256"],
            "native_source_content_sha256": provenance["native_content_sha256"],
        },
        "execution_space": f"not-{expected_space}",
        "device_sentinel": True,
        "native_module": f"{identity.package}.{identity.module}",
    }
    runner = LayoutRunner(
        invocation.platform_library,
        package=identity.package,
        validation=record,
    )
    with pytest.raises(BuildError, match="provenance mismatch: execution space"):
        build_develop.validate_development_runtime(invocation, runner)


def test_normal_build_definitions_do_not_enable_development_component():
    definitions = build_python.cmake_definitions(_manifest())
    assert definitions["SYMMETRIX_ENABLE_DEVELOPMENT_RUNTIME_COMPONENT"] == "OFF"


def test_develop_cpu_second_invocation_is_incremental_noop():
    manifest = os.environ.get("SYMMETRIX_DEVELOP_INTEGRATION_MANIFEST")
    build_root = os.environ.get("SYMMETRIX_DEVELOP_INTEGRATION_BUILD_ROOT")
    if not manifest or not build_root:
        pytest.skip("set the development integration manifest and build root")
    target = TargetManifest.read(manifest)
    if target.backend != "cpu":
        pytest.skip("the lightweight development integration test requires CPU")
    command = (
        target.python_executable,
        str(REPOSITORY_ROOT / "tools/symmetrix_build.py"),
        "develop",
        "--target-manifest",
        manifest,
        "--build-root",
        build_root,
        "--jobs",
        "2",
    )
    first = subprocess.run(
        command,
        check=False,
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    second = subprocess.run(
        command,
        check=False,
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
    )
    assert second.returncode == 0, second.stdout + second.stderr
    result = json.loads(second.stdout[second.stdout.rfind("\n{") + 1 :])
    assert result["development_only"] is True
    assert result["validation"]["status"] == "ok"
    log = (Path(result["build_directory"]) / "develop-build.log").read_text()
    assert "Building CXX object" not in log
    assert "Linking CXX shared module" not in log


def test_develop_provenance_change_rebuilds_only_initializer_object():
    manifest_path = os.environ.get("SYMMETRIX_DEVELOP_INTEGRATION_MANIFEST")
    build_root = os.environ.get("SYMMETRIX_DEVELOP_INTEGRATION_BUILD_ROOT")
    if not manifest_path or not build_root:
        pytest.skip("set the development integration manifest and build root")
    target = TargetManifest.read(manifest_path)
    if target.backend != "cpu" or target.toolchain.generator != "Unix Makefiles":
        pytest.skip("this integration test requires a CPU Unix Makefiles build")

    build_directory = Path(build_root) / target.fingerprint
    cache = (build_directory / "CMakeCache.txt").read_text()
    prefix = "SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256:UNINITIALIZED="
    matches = [
        line.removeprefix(prefix)
        for line in cache.splitlines()
        if line.startswith(prefix)
    ]
    assert len(matches) == 1
    original_hash = matches[0]
    changed_hash = ("f" if original_hash != "f" * 64 else "e") * 64
    object_root = build_directory / "CMakeFiles/symmetrix_bindings.dir/source/cpp"
    objects = tuple(sorted(object_root.rglob("*.cpp.o")))
    assert len(objects) > 1

    def configure(source_hash):
        return subprocess.run(
            (
                target.toolchain.cmake,
                "-S",
                str(REPOSITORY_ROOT / "symmetrix"),
                "-B",
                str(build_directory),
                f"-DSYMMETRIX_BUILD_SOURCE_CONTENT_SHA256={source_hash}",
            ),
            check=False,
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
        )

    def build():
        return subprocess.run(
            (
                target.toolchain.cmake,
                "--build",
                str(build_directory),
                "--target",
                "symmetrix_bindings",
                "--",
                "-j2",
            ),
            check=False,
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
        )

    before = {path: path.stat().st_mtime_ns for path in objects}
    try:
        time.sleep(1.1)
        configured = configure(changed_hash)
        assert configured.returncode == 0, configured.stdout + configured.stderr
        rebuilt = build()
        assert rebuilt.returncode == 0, rebuilt.stdout + rebuilt.stderr
        changed = {
            path.relative_to(object_root).as_posix()
            for path, timestamp in before.items()
            if path.stat().st_mtime_ns != timestamp
        }
        assert changed == {"symmetrix.cpp.o"}, rebuilt.stdout + rebuilt.stderr
        binding_compiles = [
            line
            for line in (rebuilt.stdout + rebuilt.stderr).splitlines()
            if "Building CXX object CMakeFiles/symmetrix_bindings.dir/" in line
        ]
        assert len(binding_compiles) == 1
        assert binding_compiles[0].endswith("source/cpp/symmetrix.cpp.o")
        flags = (
            build_directory / "CMakeFiles/symmetrix_bindings.dir/flags.make"
        ).read_text()
        assert changed_hash not in flags
        assert "SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256" not in flags
    finally:
        restored = configure(original_hash)
        assert restored.returncode == 0, restored.stdout + restored.stderr
        restored_build = build()
        assert restored_build.returncode == 0, (
            restored_build.stdout + restored_build.stderr
        )
