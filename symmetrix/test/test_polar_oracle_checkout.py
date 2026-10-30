from __future__ import annotations

import hashlib
import subprocess
import sys
import types
from pathlib import Path

import pytest

from polar_oracle_checkout import (
    import_checkout_module,
    optional_checkout,
    required_checkout,
    require_checkout_identity,
)


def _git(checkout: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _committed_checkout(tmp_path: Path) -> tuple[Path, str, dict[str, str]]:
    checkout = tmp_path / "oracle"
    checkout.mkdir()
    source = checkout / "oracle.py"
    source.write_text("VALUE = 7\n")
    _git(checkout, "init", "-q")
    _git(checkout, "add", "oracle.py")
    _git(
        checkout,
        "-c",
        "user.name=Symmetrix test",
        "-c",
        "user.email=symmetrix@example.invalid",
        "commit",
        "-q",
        "-m",
        "pin oracle",
    )
    revision = _git(checkout, "rev-parse", "HEAD")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return checkout, revision, {"oracle.py": digest}


def _committed_package(tmp_path: Path) -> tuple[Path, str, dict[str, str]]:
    checkout = tmp_path / "package-checkout"
    package = checkout / "oracle_package"
    package.mkdir(parents=True)
    init = package / "__init__.py"
    module = package / "realspace.py"
    init.write_text("from .realspace import VALUE\n")
    module.write_text("VALUE = 7\n")
    _git(checkout, "init", "-q")
    _git(checkout, "add", "oracle_package")
    _git(
        checkout,
        "-c",
        "user.name=Symmetrix test",
        "-c",
        "user.email=symmetrix@example.invalid",
        "commit",
        "-q",
        "-m",
        "pin oracle package",
    )
    revision = _git(checkout, "rev-parse", "HEAD")
    files = {
        str(path.relative_to(checkout)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (init, module)
    }
    return checkout, revision, files


def test_optional_checkout_requires_an_explicit_environment_path(tmp_path, monkeypatch):
    monkeypatch.delenv("SYMMETRIX_TEST_ORACLE", raising=False)
    assert optional_checkout("SYMMETRIX_TEST_ORACLE") is None

    monkeypatch.setenv("SYMMETRIX_TEST_ORACLE", str(tmp_path))
    assert optional_checkout("SYMMETRIX_TEST_ORACLE") == tmp_path


def test_required_checkout_skips_an_absent_or_invalid_path(tmp_path, monkeypatch):
    monkeypatch.delenv("SYMMETRIX_TEST_ORACLE", raising=False)
    with pytest.raises(pytest.skip.Exception, match="set SYMMETRIX_TEST_ORACLE"):
        required_checkout("SYMMETRIX_TEST_ORACLE")

    missing = tmp_path / "missing"
    monkeypatch.setenv("SYMMETRIX_TEST_ORACLE", str(missing))
    with pytest.raises(pytest.skip.Exception, match="does not exist"):
        required_checkout("SYMMETRIX_TEST_ORACLE")


def test_required_checkout_returns_an_explicit_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("SYMMETRIX_TEST_ORACLE", str(tmp_path))

    assert required_checkout("SYMMETRIX_TEST_ORACLE") == tmp_path


def test_oracle_checkout_identity_accepts_the_exact_revision_and_content(tmp_path):
    checkout, revision, files = _committed_checkout(tmp_path)

    require_checkout_identity(checkout, revision, files)


def test_oracle_checkout_identity_rejects_a_different_revision(tmp_path):
    checkout, revision, files = _committed_checkout(tmp_path)

    with pytest.raises(pytest.fail.Exception, match="revision mismatch"):
        require_checkout_identity(checkout, "0" * len(revision), files)


def test_oracle_checkout_identity_rejects_dirty_imported_content(tmp_path):
    checkout, revision, files = _committed_checkout(tmp_path)
    (checkout / "oracle.py").write_text("VALUE = 8\n")

    with pytest.raises(pytest.fail.Exception, match="content mismatch.*oracle.py"):
        require_checkout_identity(checkout, revision, files)


def test_oracle_import_resolves_every_checkout_module_from_verified_content(
    tmp_path, monkeypatch
):
    checkout, revision, files = _committed_package(tmp_path)
    for name in ("oracle_package.realspace", "oracle_package"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    module = import_checkout_module(
        checkout,
        revision,
        files,
        "oracle_package.realspace",
        {
            "oracle_package": "oracle_package/__init__.py",
            "oracle_package.realspace": "oracle_package/realspace.py",
        },
    )

    assert module.VALUE == 7
    assert (
        Path(module.__file__).resolve()
        == (checkout / "oracle_package/realspace.py").resolve()
    )


def test_oracle_import_rejects_a_conflicting_cached_package(tmp_path, monkeypatch):
    checkout, revision, files = _committed_package(tmp_path)
    conflicting = types.ModuleType("oracle_package")
    conflicting.__file__ = str(tmp_path / "unverified/oracle_package/__init__.py")
    conflicting.__path__ = [str(tmp_path / "unverified/oracle_package")]
    monkeypatch.setitem(sys.modules, "oracle_package", conflicting)
    monkeypatch.delitem(sys.modules, "oracle_package.realspace", raising=False)

    with pytest.raises(pytest.fail.Exception, match="cached oracle module conflict"):
        import_checkout_module(
            checkout,
            revision,
            files,
            "oracle_package.realspace",
            {
                "oracle_package": "oracle_package/__init__.py",
                "oracle_package.realspace": "oracle_package/realspace.py",
            },
        )


def test_every_opt_in_realspace_oracle_entrypoint_uses_the_verified_import(
    monkeypatch,
):
    # Catches bypassing the verified loader at any of the four live-oracle
    # entrypoints. The sentinel fires before upstream numerical work begins.
    import test_real_space_long_range_adjoints as adjoint_tests
    import test_real_space_long_range_kernels as forward_tests

    first_positions, first_multipoles, first_widths = (
        forward_tests.FP32_L1_ORDINARY_CASES[0]
    )
    cases = (
        (
            forward_tests,
            forward_tests.test_fp64_forward_matches_pinned_torch_cpu_oracle,
            (0, 0),
        ),
        (
            forward_tests,
            forward_tests.test_fp32_l1_forward_is_within_measured_pinned_torch_compatibility_envelope,
            (first_positions, first_multipoles, first_widths, "cpu"),
        ),
        (
            adjoint_tests,
            adjoint_tests.test_fp64_adjoints_match_pinned_torch_autograd,
            (0, 0, "owner"),
        ),
        (
            adjoint_tests,
            adjoint_tests.test_fp32_adjoints_are_compatible_with_pinned_torch_autograd,
            ("cpu", "owner"),
        ),
    )

    class VerifiedImportObserved(RuntimeError):
        pass

    def observe_verified_import(_checkout):
        raise VerifiedImportObserved

    for module, oracle_test, arguments in cases:
        with monkeypatch.context() as context:
            context.setattr(
                module,
                "import_graph_electrostatics_realspace",
                observe_verified_import,
            )
            with pytest.raises(VerifiedImportObserved):
                oracle_test(*arguments)
