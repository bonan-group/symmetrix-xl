"""Run a probe program under a capped address space, for refusal-before-allocation tests.

Some bounds can only be tested with a request that is *also* enormous: a buffer
count above the flat index range is at least a 16 GiB allocation, so a test that
proves the refusal happened before the allocation cannot let the request run in
the pytest process.  It runs in a child whose address space is capped below every
allocation the unguarded path would attempt, so a regression shows up as a
diagnosable failure instead of as a killed test session.

Two details make this more than a `subprocess.run` wrapper.

**Thread pinning.** The child is capped, and OpenBLAS allocates a per-thread
buffer for every thread it starts.  On a host with a large default thread count
that allocation can fail *during import*, before the code under test runs at all,
and OpenBLAS aborts the process rather than raising.  The child therefore gets
every thread-count variable set to one, so its initialization cost is fixed and
small and a capped run measures the code under test rather than the host's core
count.

**Initialization versus refusal.** The child prints :data:`INITIALIZED_MARKER`
once its imports and fixture setup have succeeded, and :data:`REPORT_MARKER` once
it has run the requests.  A child that dies before the first marker failed to
initialize; a child that dies after it, but before reporting, died *because a
request was allocated*.  Those are different failures and are reported as such,
because only the second is a regression in the bound under test.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

#: Thread-count variables pinned to one in the child.  The four vendor variables
#: are the set the repository's benchmark drivers already pin; the two OpenMP ones
#: keep Kokkos from building a wide team during the fixture setup.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "KOKKOS_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

#: Printed once the child's imports and fixture setup have succeeded.
INITIALIZED_MARKER = "CAP-INIT"
#: Printed once the child has run every request.
REPORT_MARKER = "CAP-REPORT"


def mark_initialized() -> None:
    """Signal that the probe's imports and fixture setup succeeded.

    Called by the probe program, not by the runner.  It must come *after* the
    imports that load the extension and after any fixture the probe needs, so that
    a death before this point is unambiguously an initialization failure.
    """
    print(INITIALIZED_MARKER, flush=True)


def emit(report: dict) -> None:
    """Signal the probe's result.  The runner reads the last such line."""
    print(REPORT_MARKER + json.dumps(report), flush=True)


def thread_environment() -> dict:
    """The parent's environment with every thread-count variable pinned to one."""
    environment = dict(os.environ)
    for variable in THREAD_VARIABLES:
        environment[variable] = "1"
    return environment


def require_host_address_space_cap() -> None:
    """Skip a virtual-address-space probe when a device runtime is selected.

    CUDA and HIP reserve substantial virtual address ranges while their runtime
    initializes, so ``RLIMIT_AS`` constrains the provider before the allocation
    request under test is reached.  These probes qualify host refusal-before-
    allocation behavior; device backends exercise the same deterministic bounds
    without imposing a host virtual-address cap.
    """
    import pytest
    import symmetrix

    selected = symmetrix.selected_backend() or {}
    if selected.get("backend") != "cpu":
        pytest.skip("RLIMIT_AS refusal probes are host-backend qualification tests")


def _program(body: str, address_space_bytes: int) -> str:
    """Prefixes the cap and the probe helpers to a probe program's body."""
    test_directory = str(Path(__file__).resolve().parent)
    return (
        "import json, resource, sys\n"
        # The cap first, so nothing the body imports can outgrow it.
        f"resource.setrlimit(resource.RLIMIT_AS, "
        f"({address_space_bytes}, {address_space_bytes}))\n"
        f"sys.path.insert(0, {test_directory!r})\n"
        "import capped_subprocess as _capped\n"
        "mark_initialized = _capped.mark_initialized\n"
        "emit = _capped.emit\n" + body
    )


def _marked(lines: list[str], marker: str) -> str | None:
    """The payload of the last line carrying `marker`, or None."""
    found = None
    for line in lines:
        if line.startswith(marker):
            found = line[len(marker) :]
    return found


def _diagnose(
    completed: subprocess.CompletedProcess, lines: list[str], label: str
) -> None:
    """Fails with a diagnosis that separates initialization from refusal."""
    stderr = completed.stderr[-2000:]
    if _marked(lines, INITIALIZED_MARKER) is None:
        raise AssertionError(
            f"{label} failed during initialization, before it exercised anything: it "
            f"never printed the {INITIALIZED_MARKER} marker.  That is an environment "
            f"failure - an import or a fixture setup that could not fit under the "
            f"address-space cap with one thread per pool - and not a regression in the "
            f"bound under test.  return code {completed.returncode}; "
            f"stderr tail: {stderr}"
        )
    raise AssertionError(
        f"{label} initialized and then died before reporting, so a request was "
        f"allocated instead of being refused.  That is the regression the cap exists "
        f"to catch.  return code {completed.returncode}; stderr tail: {stderr}"
    )


def run_capped_probe(
    body: str,
    *,
    address_space_bytes: int,
    timeout: float = 600,
    label: str = "the capped probe",
) -> dict:
    """Runs `body` under `address_space_bytes` and returns its report.

    `body` is probe source appended to a preamble that installs the cap, puts this
    directory on `sys.path`, and binds ``mark_initialized`` and ``emit``.  It must
    call ``mark_initialized()`` after its imports and fixture setup and ``emit()``
    once with a JSON-serializable mapping; the mapping is returned as parsed, so a
    probe can report a refusal as ``[exception name, message]`` and a count as a
    plain integer.

    Raises AssertionError with a diagnosis that distinguishes an initialization
    failure from a request that was allocated instead of refused.
    """
    completed = subprocess.run(
        [sys.executable, "-c", _program(body, address_space_bytes)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=thread_environment(),
    )
    lines = completed.stdout.splitlines()
    payload = _marked(lines, REPORT_MARKER)
    if payload is None:
        _diagnose(completed, lines, label)
        raise AssertionError("unreachable: _diagnose always raises")
    return json.loads(payload)
