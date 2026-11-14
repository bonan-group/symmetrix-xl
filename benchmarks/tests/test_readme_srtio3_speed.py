"""Behavior contracts; execution is deferred until the qualification hold lifts."""

import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import pytest


@pytest.fixture(scope="module")
def adapter():
    path = Path(__file__).parents[1] / "readme_srtio3_speed.py"
    spec = importlib.util.spec_from_file_location("readme_speed_unit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deadline_includes_setup_and_does_not_reset_between_routes(adapter):
    clock = iter([20.0, 54.0, 55.0])
    deadline = adapter.Deadline(0.0, 55.0, clock=lambda: next(clock))
    assert deadline.remaining() == 35.0
    assert deadline.remaining() == 1.0
    with pytest.raises(TimeoutError):
        deadline.remaining()


def test_cached_no_work_transaction_is_rejected(adapter):
    with pytest.raises(RuntimeError, match="work"):
        adapter.validate_work({"work": 4, "fallback": 0}, {"work": 4, "fallback": 0})


def test_direct_fallback_is_rejected_even_when_work_advances(adapter):
    with pytest.raises(RuntimeError, match="fallback"):
        adapter.validate_work({"work": 4, "fallback": 0}, {"work": 5, "fallback": 1})


def test_cached_generated_reverse_is_rejected_despite_new_forward_work(adapter):
    with pytest.raises(RuntimeError, match="generated"):
        adapter.validate_work(
            {"work": 4, "fallback": 0, "generated": {"forward": 8, "reverse": 8}},
            {"work": 5, "fallback": 0, "generated": {"forward": 9, "reverse": 8}},
        )


def test_real_transaction_keeps_raw_results_and_counter_evidence(adapter):
    state = {"work": 0, "fallback": 0}

    def compute():
        state["work"] += 1
        return {"energy": 2.0, "forces": [[1.0, 0.0, -1.0]]}

    result = adapter.transaction(
        compute, lambda: dict(state), adapter.Deadline(time.monotonic(), 55)
    )
    assert result["values"] == {"energy": 2.0, "forces": [[1.0, 0.0, -1.0]]}
    assert result["before"]["work"] == 0
    assert result["after"]["work"] == 1
    assert result["seconds"] > 0


def test_energy_per_atom_and_force_bounds_are_both_enforced(adapter):
    reference = {"energy": 0.0, "forces": [[0.0, 0.0, 0.0]] * 40}
    direct = {"energy": 0.024, "forces": [[0.0, 0.0, 0.0]] * 40}
    with pytest.raises(RuntimeError, match="parity"):
        adapter.compare_ef(reference, direct, "mace-omat-0-medium")
    direct["energy"] = 0.0
    direct["forces"] = [[0.006, 0.0, 0.0]] * 40
    with pytest.raises(RuntimeError, match="parity"):
        adapter.compare_ef(reference, direct, "mace-mh-1")


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_nonfinite_or_wrong_shape_results_cannot_earn_parity(adapter, bad):
    reference = {"energy": 0.0, "forces": [[0.0, 0.0, 0.0]] * 40}
    with pytest.raises(ValueError):
        adapter.compare_ef(
            reference, {"energy": bad, "forces": reference["forces"]}, "mace-mh-1"
        )
    with pytest.raises(ValueError):
        adapter.compare_ef(
            reference, {"energy": 0.0, "forces": [[0, 0, 0]]}, "mace-mh-1"
        )


def test_sample_summary_labels_only_same_package_reference(adapter):
    record = adapter.summarize(
        {"direct": [0.004, 0.002, 0.003], "reference": [0.009, 0.012, 0.006]}
    )
    assert record["direct"]["median_us_per_atom"] == pytest.approx(75.0)
    assert record["reference"]["median_us_per_atom"] == pytest.approx(225.0)
    assert record["symmetrix_reference_over_direct"] == pytest.approx(3.0)
    with pytest.raises(ValueError):
        adapter.summarize({"direct": [0, 0, 0], "reference": [1, 1, 1]})


def test_supervisor_timeout_retains_partial_evidence_and_reaps_worker(
    adapter, tmp_path
):
    # A real stdlib subprocess exercises the deadline/cleanup boundary, not a
    # fake scientific evaluator. No project or scientific imports are needed.
    script = tmp_path / "slow.py"
    script.write_text(
        "import time\nprint('setup-entered', flush=True)\ntime.sleep(20)\n"
    )
    result = adapter.supervise(
        [sys.executable, str(script)], tmp_path, time.monotonic(), 1.5
    )
    assert result["status"] == "timeout"
    assert result["worker_reaped"] is True
    assert result["process_group_gone"] is True
    assert result["returncode"] != 0
    assert "setup-entered" in (tmp_path / "worker.stdout").read_text()
    assert json.loads((tmp_path / "controller.json").read_text())["status"] == "timeout"


def test_supervisor_failure_is_not_retried_or_promoted(adapter, tmp_path):
    script = tmp_path / "fail.py"
    script.write_text("print('one-attempt', flush=True)\nraise SystemExit(7)\n")
    result = adapter.supervise(
        [sys.executable, str(script)], tmp_path, time.monotonic(), 5
    )
    assert result["status"] == "failed"
    assert result["returncode"] == 7
    assert (tmp_path / "worker.stdout").read_text().splitlines() == ["one-attempt"]


def test_supervisor_does_not_promote_zero_exit_without_completed_science(
    adapter, tmp_path
):
    result = adapter.supervise(
        [sys.executable, "-c", "pass"], tmp_path, time.monotonic(), 5
    )
    assert result["status"] == "failed"
    assert result["worker_reaped"] is True


def test_controller_sigterm_still_reaps_its_owned_worker(adapter, tmp_path):
    path = Path(adapter.__file__)
    wrapper = tmp_path / "controller.py"
    wrapper.write_text(
        "import importlib.util,pathlib,sys,time\n"
        f"s=importlib.util.spec_from_file_location('adapter', {str(path)!r})\n"
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
        "m.supervise([sys.executable,'-c',\"import time; print('ready',flush=True); time.sleep(20)\"], "
        f"pathlib.Path({str(tmp_path)!r}), time.monotonic(), 5)\n"
    )
    controller = subprocess.Popen([sys.executable, str(wrapper)])
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            log = tmp_path / "worker.stdout"
            if log.is_file() and "ready" in log.read_text():
                break
            time.sleep(0.01)
        else:
            pytest.fail("controlled worker did not start")
        controller.send_signal(signal.SIGTERM)
        controller.wait(timeout=3)
        result = json.loads((tmp_path / "controller.json").read_text())
        assert result["status"] == "failed"
        assert result["worker_reaped"] is True
        assert result["process_group_gone"] is True
    finally:
        if controller.poll() is None:
            controller.send_signal(signal.SIGTERM)
            controller.wait(timeout=3)


def test_mace_retained_topology_generation_and_skip_are_verified(adapter):
    before = {
        "work": 3,
        "fallback": 0,
        "graph_policy": "retained",
        "graph_generation": 7,
        "skip_count": 3,
    }
    with pytest.raises(RuntimeError, match="generation"):
        adapter.validate_work(
            before, {**before, "work": 4, "graph_generation": 8, "skip_count": 4}
        )
    with pytest.raises(RuntimeError, match="skip"):
        adapter.validate_work(before, {**before, "work": 4})


def test_equal_neighbor_counts_cannot_hide_different_captured_graphs(adapter):
    direct = {
        "cutoff_A": 4.5,
        "retained_skin_cache_graph_count": 3,
        "retained_skin_cache_graph_sha256": "s" * 64,
        "active_native_input_graph_count": 2,
        "active_native_input_graph_sha256": "a" * 64,
    }
    with pytest.raises(RuntimeError, match="graph"):
        adapter.require_same_graph(
            direct, {**direct, "active_native_input_graph_sha256": "b" * 64}
        )


def test_stop_during_popen_return_still_assigns_and_reaps_child(
    adapter, tmp_path, monkeypatch
):
    real_popen = subprocess.Popen
    children = []

    def spawn(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        # Delivered before the supervisor's Popen assignment can complete.
        os.kill(os.getpid(), signal.SIGTERM)
        os.kill(os.getpid(), signal.SIGINT)
        return child

    monkeypatch.setattr(adapter.subprocess, "Popen", spawn)
    try:
        result = adapter.supervise(
            [sys.executable, "-c", "import time; time.sleep(20)"],
            tmp_path,
            time.monotonic(),
            2,
        )
        assert result["status"] != "passed"
        assert result["worker_reaped"] is True
        assert result["returncode"] == children[0].returncode
        assert (
            json.loads((tmp_path / "controller.json").read_text())["stop_signal"]
            == signal.SIGTERM
        )
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=2)


def test_cleanup_oserror_and_repeated_stop_preserve_controller_record(
    adapter, tmp_path, monkeypatch
):
    real_killpg, real_popen = os.killpg, subprocess.Popen
    children = []

    def spawn(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        os.kill(os.getpid(), signal.SIGTERM)
        return child

    def failed_kill(pid, sig):
        if sig == signal.SIGKILL:
            os.kill(os.getpid(), signal.SIGINT)
            raise PermissionError("unit cleanup failure")
        return real_killpg(pid, sig)

    monkeypatch.setattr(adapter.subprocess, "Popen", spawn)
    monkeypatch.setattr(adapter.os, "killpg", failed_kill)
    try:
        result = adapter.supervise(
            [sys.executable, "-c", "import time; time.sleep(20)"],
            tmp_path,
            time.monotonic(),
            1.5,
        )
        saved = json.loads((tmp_path / "controller.json").read_text())
        assert saved["status"] != "passed"
        assert saved["cleanup_errors"]
        assert saved["reap_error"]
        assert result["returncode"] is None
    finally:
        for child in children:
            if child.poll() is None:
                real_killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=2)


def test_skin_cache_and_active_native_graph_are_distinct_evidence(adapter):
    retained = adapter.graph_identity([[0, 1, 2], [3, 4, 5]])
    active = adapter.graph_identity([[0, 1], [3, 4]])
    assert retained["count"] == 3
    assert active["count"] == 2
    assert retained["sha256"] != active["sha256"]


def test_process_reservation_requires_same_start_ticks_session_and_pgid(adapter):
    identity = adapter.process_identity(os.getpid())
    assert adapter.reservation_matches(identity) is True
    stale = {**identity, "starttime": str(int(identity["starttime"]) + 1)}
    assert adapter.reservation_matches(stale) is False


def test_late_stop_persists_failure_without_erasing_primary_channels(adapter, tmp_path):
    result = {
        "status": "passed",
        "primary_error": "science failed",
        "cleanup_errors": [],
        "reap_error": None,
        "write_errors": [],
    }
    updated = adapter.apply_late_stop(result, tmp_path, signal.SIGINT)
    assert updated["status"] == "failed"
    assert updated["primary_error"] == "science failed"
    saved = json.loads((tmp_path / "controller-late-stop.json").read_text())
    assert saved["late_stop_signal"] == signal.SIGINT
    assert saved["cleanup_errors"] == []
