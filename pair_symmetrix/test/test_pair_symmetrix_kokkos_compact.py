"""Kokkos pair style on a generated compact model, one LAMMPS process per case.

The model has the public MACE-OFF23 small R1 layout with generated weights
(``symmetrix/test/compact_r1_model.py``), so no checkpoint is downloaded. Each
case runs in a fresh process: an exception thrown from a pair compute, such as
the prepared graph checks' ``std::out_of_range``, is not caught by the LAMMPS
library interface and would abort an in-process test session.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

try:
    from lammps import lammps
except ImportError as exc:
    pytest.skip(
        f"LAMMPS Python module is not available: {exc}", allow_module_level=True
    )


_REPOSITORY = Path(__file__).resolve().parents[2]
_SYMMETRIX_TESTS = _REPOSITORY / "symmetrix" / "test"
_R1_CONTRACT = (
    _SYMMETRIX_TESTS
    / "data"
    / "execution_contracts"
    / "jit_r1_mace_off23_small_contract.json"
)

_RUN_CASE = r"""
import json
import sys

from lammps import lammps

spec = json.loads(sys.argv[1])
lmp = lammps(cmdargs=spec["cmdargs"])
try:
    lmp.commands_string(spec["setup"])
    lmp.command("run 0")
    count = lmp.get_natoms()
    record = {"natoms": count}
    for label in ("initial", "final"):
        if label == "final":
            lmp.command("run 4")
        ids = np.array(lmp.numpy.extract_atom("id", nelem=count), copy=True)
        forces = np.array(
            lmp.numpy.extract_atom("f", nelem=count, dim=3), copy=True
        )
        order = np.argsort(ids)
        record[label] = {
            "pe": lmp.get_thermo("pe"),
            "forces": forces[order].tolist(),
        }
finally:
    lmp.close()
print("RESULT " + json.dumps(record))
"""

# 24 atoms (8 H2O) in a periodic box shorter than twice the 4.5 A cutoff, so
# most neighbors are ghost images that the pair style maps back to owners.
_POSITIONS = (
    (2, 1, 1, 1),
    (1, 2, 1, 1),
    (1, 1, 2, 1),
    (2, 4, 1, 1),
    (1, 5, 1, 1),
    (1, 4, 2, 1),
    (2, 1, 4, 1),
    (1, 2, 4, 1),
    (1, 1, 5, 1),
    (2, 1, 1, 4),
    (1, 2, 1, 4),
    (1, 1, 2, 4),
    (2, 4, 4, 1),
    (1, 5, 4, 1),
    (1, 4, 5, 1),
    (2, 4, 1, 4),
    (1, 5, 1, 4),
    (1, 4, 2, 4),
    (2, 1, 4, 4),
    (1, 2, 4, 4),
    (1, 1, 5, 4),
    (2, 4, 4, 4),
    (1, 5, 4, 4),
    (1, 4, 5, 4),
)


def _lammps_kokkos_api():
    lmp = lammps(cmdargs=["-nocite", "-screen", "none", "-log", "none"])
    try:
        if not lmp.has_package("KOKKOS"):
            pytest.skip("LAMMPS was built without the KOKKOS package")
        return set(lmp.accelerator_config["KOKKOS"]["api"])
    finally:
        lmp.close()


def _device_kokkos(api):
    return bool(api & {"cuda", "hip"})


@pytest.fixture(scope="module")
def kokkos_api():
    return _lammps_kokkos_api()


@pytest.fixture(scope="module")
def compact_model_path(tmp_path_factory):
    """A compact two-layer MACE with the public MACE-OFF23 small R1 layout."""

    helper = _SYMMETRIX_TESTS / "compact_r1_model.py"
    if not helper.is_file() or not _R1_CONTRACT.is_file():
        pytest.skip(f"the compact R1 test model helper is missing: {helper}")
    spec = importlib.util.spec_from_file_location("compact_r1_model", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.foundation_r1_test_model(
        json.loads(_R1_CONTRACT.read_text()), atomic_numbers=(1, 8)
    )
    path = tmp_path_factory.mktemp("kokkos-atom-map") / "compact-mace.json"
    path.write_text(json.dumps(model, separators=(",", ":")))
    return path


@pytest.fixture(scope="module")
def jit_cache(tmp_path_factory):
    # tmp_path_factory directories are private to the user, as the JIT cache
    # root requires.
    return tmp_path_factory.mktemp("kokkos-atom-map-jit")


def _prepare_artifact(command, jit_cache):
    environment = os.environ.copy()
    environment["SYMMETRIX_JIT_CACHE"] = str(jit_cache)
    return subprocess.run(
        [sys.executable, "-m", *command, "--cache-root", str(jit_cache)],
        capture_output=True,
        text=True,
        timeout=1800,
        env=environment,
    )


@pytest.fixture(scope="module")
def host_artifact_path(compact_model_path, jit_cache, kokkos_api):
    if _device_kokkos(kokkos_api):
        pytest.skip("host R1 artifacts require a host-only Kokkos LAMMPS build")
    completed = _prepare_artifact(
        [
            "symmetrix.cli.prepare_jit_host_artifact",
            "--model",
            str(compact_model_path),
            "--precision",
            "float64",
            "--path-only",
        ],
        jit_cache,
    )
    assert completed.returncode == 0, completed.stderr
    path = Path(completed.stdout.strip().splitlines()[-1])
    assert path.is_file(), completed.stdout
    return path


def _kokkos_arguments():
    arguments = ["-k", "on", "t", "1"]
    gpus = os.environ.get("SYMMETRIX_LAMMPS_KOKKOS_GPUS")
    if gpus:
        arguments.extend(("g", gpus))
    return [*arguments, "-sf", "kk"]


def _run_case(tmp_path, label, model, pair_style, atom_map, package_options=""):
    log_path = tmp_path / f"{label}.log"
    atoms = "\n".join(
        f"create_atoms {kind} single {x} {y} {z} units box"
        for kind, x, y, z in _POSITIONS
    )
    setup = f"""
package         kokkos neigh half newton on {package_options}
units           metal
boundary        p p p
atom_style      atomic/kk
atom_modify     map {atom_map} sort 0 0
newton          on
region          box block 0 6.2085633514918648 0 6.2085633514918648 0 6.2085633514918648
create_box      2 box
{atoms}
mass            1 1.008
mass            2 15.999
pair_style      {pair_style}
pair_coeff      * * {model} H O
neighbor        1.0 bin
neigh_modify    every 2 delay 0 check no
timestep        0.0001
fix             integrate all nve
"""
    spec = {
        "cmdargs": [
            "-nocite",
            "-screen",
            "none",
            "-log",
            str(log_path),
            *_kokkos_arguments(),
        ],
        "setup": setup,
    }
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OMP_PROC_BIND": "false",
            "SYMMETRIX_JIT_POLICY": "none",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", _RUN_CASE, json.dumps(spec)],
        capture_output=True,
        text=True,
        timeout=900,
        env=environment,
    )
    log_text = log_path.read_text() if log_path.exists() else ""
    assert completed.returncode == 0, (
        f"{label} exited with {completed.returncode}\n"
        f"stderr:\n{completed.stderr[-4000:]}\nlog tail:\n{log_text[-4000:]}"
    )
    result_lines = [
        line for line in completed.stdout.splitlines() if line.startswith("RESULT ")
    ]
    assert result_lines, completed.stdout
    result = json.loads(result_lines[-1][len("RESULT ") :])
    result["log_text"] = log_text
    return result


def _pair_style(streamed_edges, host_artifact=None, extra=""):
    style = "symmetrix/mace no_domain_decomposition "
    if streamed_edges == "direct":
        style += (
            f"streamed_edges direct profile capacity jit_host_artifact {host_artifact}"
        )
    else:
        style += f"streamed_edges {streamed_edges}"
    return f"{style} {extra}".rstrip()


_BLOCK_WARNING = re.compile(
    r"jit_device_blocks_per_compute_unit (\d+) differs from the HIP R1 device "
    r"artifact's own (\d+) blocks per compute unit"
)


def test_host_direct_explicit_block_count_does_not_warn(
    tmp_path, compact_model_path, host_artifact_path
):
    result = _run_case(
        tmp_path,
        "host-direct-blocks",
        compact_model_path,
        _pair_style(
            "direct", host_artifact_path, "jit_device_blocks_per_compute_unit 3"
        ),
        "array",
    )
    assert "resolved='direct'" in result["log_text"]
    assert "blocks per compute unit" not in result["log_text"]
    assert not _BLOCK_WARNING.search(result["log_text"])


def test_hip_r1_block_count_warning_names_the_artifact_value(
    tmp_path, kokkos_api, compact_model_path, jit_cache
):
    if "hip" not in kokkos_api:
        pytest.skip("the R1 block-count warning is specific to Kokkos HIP builds")
    if not os.environ.get("SYMMETRIX_LAMMPS_KOKKOS_GPUS"):
        pytest.skip("Set SYMMETRIX_LAMMPS_KOKKOS_GPUS to run on a HIP device.")
    completed = _prepare_artifact(
        [
            "symmetrix.cli.prepare_jit_device_artifact",
            "--model",
            str(compact_model_path),
            "--precision",
            "float32",
            "--backend",
            "hip",
            "--lammps-arguments",
        ],
        jit_cache,
    )
    if completed.returncode != 0:
        pytest.skip(
            "a float32 HIP R1 device artifact could not be prepared in this "
            f"environment: {completed.stderr.strip()[-2000:]}"
        )
    rendered = completed.stdout.strip().splitlines()[-1]
    # Without the rendered value each case below states its own setting.
    fragment = re.sub(r"\bjit_device_blocks_per_compute_unit\s+\d+", "", rendered)
    fragment = " ".join(fragment.split())

    def run(label, arguments):
        style = (
            "symmetrix/mace/float32 no_domain_decomposition streamed_edges direct "
            f"profile capacity {arguments}"
        )
        return _run_case(tmp_path, label, compact_model_path, style, "array")[
            "log_text"
        ]

    # The prepared arguments of the JIT frontend and an inherited default stay
    # silent.
    for label, arguments in (("hip-rendered", rendered), ("hip-inherited", fragment)):
        log_text = run(label, arguments)
        assert "Activated Execution JIT device artifact" in log_text
        assert not _BLOCK_WARNING.search(log_text), label

    # The artifact's own value is one of [1, 32], so at most one of these
    # explicit settings equals it and at least one must warn.
    warned = {}
    for setting in (1, 32):
        log_text = run(
            f"hip-explicit-{setting}",
            f"{fragment} jit_device_blocks_per_compute_unit {setting}",
        )
        matches = _BLOCK_WARNING.findall(log_text)
        assert len(matches) <= 1, matches
        if matches:
            requested, artifact_value = (int(value) for value in matches[0])
            assert requested == setting
            assert artifact_value != setting
            warned[setting] = artifact_value
    assert warned, "an explicit block count that differs from the artifact did not warn"
    assert len(set(warned.values())) == 1, warned
    artifact_value = next(iter(warned.values()))
    for setting in (1, 32):
        assert (setting in warned) == (setting != artifact_value)
