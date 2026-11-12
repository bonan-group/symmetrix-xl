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

import numpy as np
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

import numpy as np
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
        types = np.array(lmp.numpy.extract_atom("type", nelem=count), copy=True)
        order = np.argsort(ids)
        record[label] = {
            "pe": lmp.get_thermo("pe"),
            "forces": forces[order].tolist(),
            "types": types[order].tolist(),
        }
finally:
    lmp.close()
print("RESULT " + json.dumps(record))
"""

# Runs each stage's commands in one process and records the state after it.
_RUN_STAGES = r"""
import json
import sys

import numpy as np
from lammps import lammps

spec = json.loads(sys.argv[1])
lmp = lammps(cmdargs=spec["cmdargs"])
try:
    lmp.commands_string(spec["setup"])
    count = lmp.get_natoms()
    record = {"natoms": count}
    for label, commands in spec["stages"]:
        lmp.commands_string(commands)
        ids = np.array(lmp.numpy.extract_atom("id", nelem=count), copy=True)
        forces = np.array(
            lmp.numpy.extract_atom("f", nelem=count, dim=3), copy=True
        )
        order = np.argsort(ids)
        record[label] = {
            "pe": lmp.get_thermo("pe"),
            "forces": forces[order].tolist(),
            "graph_rebuilds": lmp.extract_pair("symmetrix_graph_rebuild_count"),
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

# Atoms of a third type that a hybrid pair_coeff gives to another sub-style.
_DUMMY_POSITIONS = (
    (2.6, 2.6, 2.6),
    (5.6, 2.6, 2.6),
    (2.6, 5.6, 2.6),
    (2.6, 2.6, 5.6),
    (5.6, 5.6, 2.6),
    (5.6, 2.6, 5.6),
    (2.6, 5.6, 5.6),
    (5.6, 5.6, 5.6),
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


def _run_case(
    tmp_path, label, model, pair_style, atom_map, package_options="", coeff_style=""
):
    coeff = f"{coeff_style} {model}".strip()
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
pair_coeff      * * {coeff} H O
neighbor        1.0 bin
neigh_modify    every 2 delay 0 check no
timestep        0.0001
fix             integrate all nve
"""
    completed, log_text = _run_deck(tmp_path, label, setup)
    return _case_result(label, completed, log_text)


def _run_deck(tmp_path, label, setup, timeout=900, stages=None):
    log_path = tmp_path / f"{label}.log"
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
        "stages": stages,
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
        [
            sys.executable,
            "-c",
            _RUN_CASE if stages is None else _RUN_STAGES,
            json.dumps(spec),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environment,
    )
    log_text = log_path.read_text() if log_path.exists() else ""
    return completed, log_text


def _case_result(label, completed, log_text):
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
        # The compact model has no R0 execution contract, so the default
        # profile capacity, which needs a state-free R0, is unavailable.
        style += (
            f"streamed_edges direct profile speed jit_host_artifact {host_artifact}"
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


@pytest.mark.parametrize(
    "package_options",
    ["", "pair/only on"],
    ids=["default-map", "pair-only"],
)
@pytest.mark.parametrize("streamed_edges", ["generic", "direct"])
def test_kokkos_hashed_atom_map_matches_array_map(
    request, tmp_path, kokkos_api, compact_model_path, streamed_edges, package_options
):
    # On a host-only Kokkos build the atom map is built on the host by default;
    # pair/only on selects the same host map on device builds. A hashed map then
    # lives only in the host copy until the pair style syncs it.
    if streamed_edges == "direct":
        artifact = request.getfixturevalue("host_artifact_path")
        pair_style = _pair_style("direct", artifact)
    else:
        pair_style = _pair_style("generic")
    label = f"{streamed_edges}-{'pair-only' if package_options else 'default'}"
    results = {
        atom_map: _run_case(
            tmp_path,
            f"{label}-{atom_map}",
            compact_model_path,
            pair_style,
            atom_map,
            package_options,
        )
        for atom_map in ("array", "hash")
    }

    for result in results.values():
        assert result["natoms"] == len(_POSITIONS)
        assert (
            f"resolved='{streamed_edges}'" in result["log_text"]
        ), "the requested streamed-edge path did not run"
        if streamed_edges == "direct":
            assert "Activated Execution JIT host artifact" in result["log_text"]

    reference = results["array"]
    hashed = results["hash"]
    for stage in ("initial", "final"):
        expected_forces = np.asarray(reference[stage]["forces"])
        assert np.all(np.isfinite(expected_forces))
        assert np.max(np.abs(expected_forces)) > 0.0
        assert hashed[stage]["pe"] == pytest.approx(
            reference[stage]["pe"], rel=1.0e-12, abs=1.0e-10
        )
        np.testing.assert_allclose(
            hashed[stage]["forces"], expected_forces, rtol=1.0e-12, atol=1.0e-10
        )


def test_hybrid_overlay_message_passing_matches_the_plain_style(
    tmp_path, compact_model_path
):
    # Comm::init sizes the host communication buffers from force->pair, which
    # under pair_style hybrid/overlay is the wrapper. The wrapper copies its
    # sub-styles' message sizes in settings(), before pair_coeff loads the
    # model, so the pair style raises them in init_style(). This model sends 96
    # values per atom, against 6 for the atomic style's borders; on one rank
    # every swap goes through those buffers, and a wrapper left at the atom
    # style's size made the per-atom message passing write past them. The plain
    # style sizes the buffers itself and is the reference.
    style = "symmetrix/mace mpi_message_passing streamed_edges generic"
    plain = _run_case(tmp_path, "plain-mpi", compact_model_path, style, "array")
    hybrid = _run_case(
        tmp_path,
        "hybrid-overlay-mpi",
        compact_model_path,
        f"hybrid/overlay {style}",
        "array",
        coeff_style="symmetrix/mace",
    )

    for result in (plain, hybrid):
        assert result["natoms"] == len(_POSITIONS)
        assert "resolved='generic'" in result["log_text"]
    for stage in ("initial", "final"):
        expected_forces = np.asarray(plain[stage]["forces"])
        assert np.max(np.abs(expected_forces)) > 0.0
        assert hybrid[stage]["pe"] == pytest.approx(
            plain[stage]["pe"], rel=1.0e-12, abs=1.0e-10
        )
        np.testing.assert_allclose(
            hybrid[stage]["forces"], expected_forces, rtol=1.0e-12, atol=1.0e-10
        )


_RUN_EMPTY_FIRST = r"""
import json
import sys

import numpy as np
from lammps import lammps

spec = json.loads(sys.argv[1])
lmp = lammps(cmdargs=spec["cmdargs"])
try:
    lmp.commands_string(spec["setup"])
    record = {}
    if spec["empty_first"]:
        lmp.command("run 0")
        record["empty_pe"] = lmp.get_thermo("pe")
    lmp.commands_string(spec["atoms"])
    lmp.command("run 0")
    count = lmp.get_natoms()
    ids = np.array(lmp.numpy.extract_atom("id", nelem=count), copy=True)
    forces = np.array(lmp.numpy.extract_atom("f", nelem=count, dim=3), copy=True)
    record["pe"] = lmp.get_thermo("pe")
    record["forces"] = forces[np.argsort(ids)].tolist()
finally:
    lmp.close()
print("RESULT " + json.dumps(record))
"""


@pytest.mark.parametrize(
    "domain_mode", ["no_domain_decomposition", "mpi_message_passing"]
)
def test_direct_first_evaluation_without_atoms_matches_a_fresh_run(
    tmp_path, compact_model_path, host_artifact_path, domain_mode
):
    # A box created without atoms is evaluated by the first run with a graph
    # without receivers; the direct route must give that graph valid tensor
    # shapes and then evaluate the atoms created afterwards.
    pair_style = _pair_style("direct", host_artifact_path).replace(
        "no_domain_decomposition", domain_mode
    )
    setup = f"""
package         kokkos neigh half newton on
units           metal
boundary        p p p
atom_style      atomic/kk
atom_modify     map array sort 0 0
newton          on
region          box block 0 6.2085633514918648 0 6.2085633514918648 0 6.2085633514918648
create_box      2 box
mass            1 1.008
mass            2 15.999
pair_style      {pair_style}
pair_coeff      * * {compact_model_path} H O
neighbor        1.0 bin
"""
    atoms = "\n".join(
        f"create_atoms {kind} single {x} {y} {z} units box"
        for kind, x, y, z in _POSITIONS
    )
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OMP_PROC_BIND": "false",
            "SYMMETRIX_JIT_POLICY": "none",
        }
    )
    results = {}
    for empty_first in (True, False):
        label = f"empty-first-{domain_mode}-{empty_first}"
        log_path = tmp_path / f"{label}.log"
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
            "atoms": atoms,
            "empty_first": empty_first,
        }
        completed = subprocess.run(
            [sys.executable, "-c", _RUN_EMPTY_FIRST, json.dumps(spec)],
            capture_output=True,
            text=True,
            timeout=300,
            env=environment,
        )
        log_text = log_path.read_text() if log_path.exists() else ""
        assert completed.returncode == 0, (
            f"{label} exited with {completed.returncode}\n"
            f"stderr:\n{completed.stderr[-4000:]}\nlog tail:\n{log_text[-4000:]}"
        )
        assert "resolved='direct'" in log_text
        lines = [
            line for line in completed.stdout.splitlines() if line.startswith("RESULT ")
        ]
        assert lines, completed.stdout
        results[empty_first] = json.loads(lines[-1][len("RESULT ") :])

    empty, fresh = results[True], results[False]
    assert empty["empty_pe"] == 0.0
    expected_forces = np.asarray(fresh["forces"])
    assert np.max(np.abs(expected_forces)) > 0.0
    assert empty["pe"] == pytest.approx(fresh["pe"], rel=1.0e-12, abs=1.0e-10)
    np.testing.assert_allclose(
        empty["forces"], expected_forces, rtol=1.0e-12, atol=1.0e-10
    )


def _type_filtered_deck(model, pair_style, wrapper, placement):
    # Type 1 has no MACE element: plain hybrid gives its pairs to zero, and
    # hybrid/overlay removes them with none. The MACE atoms keep their
    # positions, so they interact as in the two-type deck.
    real = [
        f"create_atoms {kind + 1} single {x} {y} {z} units box"
        for kind, x, y, z in _POSITIONS
    ]
    dummy = [
        f"create_atoms 1 single {x} {y} {z} units box" for x, y, z in _DUMMY_POSITIONS
    ]
    atoms = {
        "first": dummy + real,
        "middle": real[:12] + dummy + real[12:],
        "last": real + dummy,
    }[placement]
    if wrapper == "hybrid":
        pair = f"""pair_style      hybrid {pair_style} zero 6.0
pair_coeff      * * symmetrix/mace {model} H H O
pair_coeff      1 * zero"""
    else:
        pair = f"""pair_style      hybrid/overlay {pair_style} zero 6.0
pair_coeff      * * symmetrix/mace {model} H H O
pair_coeff      * * zero
pair_coeff      1 * none"""
    atom_lines = "\n".join(atoms)
    return f"""
package         kokkos neigh half newton on
units           metal
boundary        p p p
atom_style      atomic/kk
atom_modify     map array sort 0 0
newton          on
region          box block 0 6.2085633514918648 0 6.2085633514918648 0 6.2085633514918648
create_box      3 box
{atom_lines}
mass            1 39.95
mass            2 1.008
mass            3 15.999
{pair}
neighbor        1.0 bin
neigh_modify    every 2 delay 0 check no
timestep        0.0001
fix             integrate all nve
"""


@pytest.mark.parametrize("wrapper", ["hybrid", "hybrid/overlay"])
@pytest.mark.parametrize("streamed_edges", ["generic", "materialized", "direct"])
def test_type_filtered_hybrid_list_matches_the_mace_only_deck(
    request, tmp_path, compact_model_path, streamed_edges, wrapper
):
    # A type-filtered list skips the atoms of the other type, so receiver rows
    # and local atom indices differ before, between or after them. The MACE
    # atoms are within the cutoff of each other, so a neighbor taken by its
    # local index reads another receiver's features or an unwritten row.
    if streamed_edges == "direct":
        artifact = request.getfixturevalue("host_artifact_path")
        pair_style = _pair_style("direct", artifact)
    else:
        pair_style = _pair_style(streamed_edges)
    label = f"filtered-{streamed_edges}-{wrapper.replace('/', '-')}"
    reference = _run_case(
        tmp_path, f"{label}-reference", compact_model_path, pair_style, "array"
    )
    for placement in ("first", "middle", "last"):
        completed, log_text = _run_deck(
            tmp_path,
            f"{label}-{placement}",
            _type_filtered_deck(compact_model_path, pair_style, wrapper, placement),
        )
        result = _case_result(f"{label}-{placement}", completed, log_text)
        assert result["natoms"] == len(_POSITIONS) + len(_DUMMY_POSITIONS)
        assert f"resolved='{streamed_edges}'" in log_text
        for stage in ("initial", "final"):
            types = np.asarray(result[stage]["types"])
            forces = np.asarray(result[stage]["forces"])
            expected_forces = np.asarray(reference[stage]["forces"])
            assert np.max(np.abs(expected_forces)) > 0.0
            assert result[stage]["pe"] == pytest.approx(
                reference[stage]["pe"], rel=1.0e-12, abs=1.0e-10
            ), placement
            np.testing.assert_allclose(
                forces[types != 1],
                expected_forces,
                rtol=1.0e-12,
                atol=1.0e-10,
                err_msg=placement,
            )
            assert np.all(forces[types == 1] == 0.0), placement


@pytest.mark.parametrize("streamed_edges", ["generic", "direct"])
def test_mpi_message_passing_stops_on_a_type_filtered_list(
    request, tmp_path, compact_model_path, streamed_edges
):
    # mpi_message_passing indexes receivers by local atom, so it requires every
    # owned atom in the list and stops the rank that lacks one instead of
    # writing past its receiver rows. The time-out turns a rank waiting in a
    # collective into a failure.
    if streamed_edges == "direct":
        artifact = request.getfixturevalue("host_artifact_path")
        pair_style = _pair_style("direct", artifact)
    else:
        pair_style = _pair_style(streamed_edges)
    pair_style = pair_style.replace("no_domain_decomposition", "mpi_message_passing")
    label = f"filtered-mpi-{streamed_edges}"
    completed, log_text = _run_deck(
        tmp_path,
        label,
        _type_filtered_deck(compact_model_path, pair_style, "hybrid/overlay", "first"),
        timeout=300,
    )
    assert completed.returncode != 0, log_text[-2000:]
    assert (
        "mpi_message_passing requires every owned atom in its neighbor list"
        in log_text + completed.stderr
    ), f"stderr:\n{completed.stderr[-2000:]}\nlog tail:\n{log_text[-2000:]}"


def _cluster_deck(model, pair_style, kinds):
    # The water cluster of _POSITIONS in a box with more than the cutoff plus
    # the skin of vacuum on every side, so the neighbor list has no ghost atom.
    atoms = "\n".join(
        f"create_atoms {kind} single {x} {y} {z} units box"
        for kind, (_, x, y, z) in zip(kinds, _POSITIONS)
    )
    return f"""
package         kokkos neigh half newton on
units           metal
boundary        p p p
atom_style      atomic/kk
atom_modify     map array sort 0 0
newton          on
region          box block -10 16 -10 16 -10 16
create_box      2 box
{atoms}
mass            1 1.008
mass            2 15.999
pair_style      {pair_style}
pair_coeff      * * {model} H O
neighbor        1.0 bin
neigh_modify    every 1 delay 0 check yes
"""


@pytest.mark.parametrize("streamed_edges", ["generic", "direct"])
def test_type_change_without_a_neighbor_build_matches_a_fresh_run(
    request, tmp_path, compact_model_path, streamed_edges
):
    # fix sgcmc changes an atom's type and evaluates the forces without a
    # neighbor list build. set followed by run 1 pre no post no reaches the same
    # pair compute; the atoms do not move, so LAMMPS does not rebuild the list.
    if streamed_edges == "direct":
        artifact = request.getfixturevalue("host_artifact_path")
        pair_style = _pair_style("direct", artifact)
    else:
        pair_style = _pair_style(streamed_edges)
    kinds = [kind for kind, _, _, _ in _POSITIONS]
    changed_kinds = list(kinds)
    changed_kinds[1] = 2
    completed, log_text = _run_deck(
        tmp_path,
        f"type-change-{streamed_edges}",
        _cluster_deck(compact_model_path, pair_style, kinds),
        stages=[
            ["initial", "run 0"],
            ["unchanged", "run 1 pre no post no"],
            ["changed", "set atom 2 type 2\nrun 1 pre no post no"],
        ],
    )
    result = _case_result(f"type-change-{streamed_edges}", completed, log_text)
    assert f"resolved='{streamed_edges}'" in log_text
    completed, log_text = _run_deck(
        tmp_path,
        f"type-change-{streamed_edges}-fresh",
        _cluster_deck(compact_model_path, pair_style, changed_kinds),
        stages=[["fresh", "run 0"]],
    )
    fresh = _case_result(f"type-change-{streamed_edges}-fresh", completed, log_text)[
        "fresh"
    ]

    initial, unchanged, changed = (
        result[stage] for stage in ("initial", "unchanged", "changed")
    )
    assert unchanged["pe"] == pytest.approx(initial["pe"], rel=1.0e-12, abs=1.0e-10)
    assert abs(fresh["pe"] - initial["pe"]) > 1.0e-7
    assert changed["pe"] == pytest.approx(fresh["pe"], rel=1.0e-12, abs=1.0e-10)
    np.testing.assert_allclose(
        changed["forces"], fresh["forces"], rtol=1.0e-12, atol=1.0e-10
    )
    if streamed_edges == "direct":
        # The prepared graph is rebuilt for the type change only.
        assert [initial["graph_rebuilds"], unchanged["graph_rebuilds"]] == [1, 1]
        assert changed["graph_rebuilds"] == 2


def _static_deck(model, pair_style):
    # The two-type deck of _run_case without integration.
    atoms = "\n".join(
        f"create_atoms {kind} single {x} {y} {z} units box"
        for kind, x, y, z in _POSITIONS
    )
    return f"""
package         kokkos neigh half newton on
units           metal
boundary        p p p
atom_style      atomic/kk
atom_modify     map array sort 0 0
newton          on
region          box block 0 6.2085633514918648 0 6.2085633514918648 0 6.2085633514918648
create_box      2 box
{atoms}
mass            1 1.008
mass            2 15.999
pair_style      {pair_style}
pair_coeff      * * {model} H O
neighbor        1.0 bin
"""


_REISSUE_BASE = "symmetrix/mace no_domain_decomposition streamed_edges generic"
_REISSUE_CHANGES = {
    "mode": "symmetrix/mace mpi_message_passing streamed_edges generic",
    "streamed-edges": "symmetrix/mace no_domain_decomposition streamed_edges materialized",
    "profile": f"{_REISSUE_BASE} profile speed",
    "ghost-mode": "symmetrix/mace no_mpi_message_passing streamed_edges generic",
}


def test_identical_pair_style_reissue_keeps_the_model(tmp_path, compact_model_path):
    completed, log_text = _run_deck(
        tmp_path,
        "reissue-identical",
        _static_deck(compact_model_path, _REISSUE_BASE),
        stages=[
            ["first", "run 0"],
            ["second", f"pair_style {_REISSUE_BASE}\nrun 0"],
        ],
    )
    result = _case_result("reissue-identical", completed, log_text)
    assert log_text.count("Loading MACEKokkos model") == 1
    assert result["second"]["pe"] == result["first"]["pe"]
    np.testing.assert_array_equal(result["second"]["forces"], result["first"]["forces"])


@pytest.mark.parametrize(
    "change",
    [*_REISSUE_CHANGES, "head"],
)
def test_changed_pair_style_options_require_pair_coeff(
    tmp_path, compact_model_path, change
):
    # LAMMPS passes a re-issued pair_style line with the same style to the
    # existing instance, which keeps the model loaded by the earlier pair_coeff
    # and the options it was loaded with.
    second = _REISSUE_CHANGES.get(change, f"{_REISSUE_BASE} head other")
    completed, log_text = _run_deck(
        tmp_path,
        f"reissue-{change}",
        _static_deck(compact_model_path, _REISSUE_BASE),
        stages=[["first", "run 0"], ["second", f"pair_style {second}\nrun 0"]],
        timeout=300,
    )
    assert completed.returncode != 0, log_text[-2000:]
    assert (
        "options changed after pair_coeff; issue pair_coeff again"
        in log_text + completed.stderr
    ), f"stderr:\n{completed.stderr[-2000:]}\nlog tail:\n{log_text[-2000:]}"


@pytest.mark.parametrize("change", list(_REISSUE_CHANGES))
def test_changed_pair_style_options_with_pair_coeff_match_a_fresh_deck(
    tmp_path, compact_model_path, change
):
    second = _REISSUE_CHANGES[change]
    completed, log_text = _run_deck(
        tmp_path,
        f"reissue-coeff-{change}",
        _static_deck(compact_model_path, _REISSUE_BASE),
        stages=[
            ["first", "run 0"],
            [
                "second",
                f"pair_style {second}\npair_coeff * * {compact_model_path} H O\nrun 0",
            ],
        ],
    )
    result = _case_result(f"reissue-coeff-{change}", completed, log_text)
    completed, log_text = _run_deck(
        tmp_path,
        f"reissue-fresh-{change}",
        _static_deck(compact_model_path, second),
        stages=[["fresh", "run 0"]],
    )
    fresh = _case_result(f"reissue-fresh-{change}", completed, log_text)["fresh"]
    assert result["second"]["pe"] == pytest.approx(
        fresh["pe"], rel=1.0e-12, abs=1.0e-10
    )
    np.testing.assert_allclose(
        result["second"]["forces"], fresh["forces"], rtol=1.0e-12, atol=1.0e-10
    )
