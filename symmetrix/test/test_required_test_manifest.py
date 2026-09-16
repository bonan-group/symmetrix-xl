from types import SimpleNamespace

import conftest as qualification
import pytest


class _PluginManager:
    @staticmethod
    def get_plugin(name):
        assert name == "terminalreporter"


def _session():
    return SimpleNamespace(
        config=SimpleNamespace(pluginmanager=_PluginManager()),
        exitstatus=pytest.ExitCode.OK,
    )


def _report(*, when, passed=False, skipped=False, wasxfail=False):
    return SimpleNamespace(
        nodeid="test_required.py::test_required",
        when=when,
        passed=passed,
        skipped=skipped,
        wasxfail=wasxfail,
    )


@pytest.fixture
def required_manifest(monkeypatch):
    node = "test_required.py::test_required"
    monkeypatch.setattr(qualification, "_REQUIRED_NODES", {node})
    monkeypatch.setattr(qualification, "_REQUIRED_PASSED_CALLS", set())
    monkeypatch.setattr(
        qualification,
        "_required_test_manifest",
        lambda: {"required_nodes": [node]},
    )


def test_required_manifest_accepts_only_a_normal_passed_call(required_manifest):
    qualification.pytest_runtest_logreport(_report(when="setup", passed=True))
    qualification.pytest_runtest_logreport(_report(when="call", passed=True))
    session = _session()
    qualification.pytest_sessionfinish(session, pytest.ExitCode.OK)
    assert session.exitstatus == pytest.ExitCode.OK


@pytest.mark.parametrize(
    "report",
    (
        _report(when="setup", skipped=True),
        _report(when="setup", skipped=True, wasxfail=True),
        _report(when="call", skipped=True),
        _report(when="call", passed=True, wasxfail=True),
    ),
)
def test_required_manifest_rejects_skip_or_xfail(required_manifest, report):
    qualification.pytest_runtest_logreport(report)
    session = _session()
    qualification.pytest_sessionfinish(session, pytest.ExitCode.OK)
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
