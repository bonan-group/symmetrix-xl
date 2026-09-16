import os
from pathlib import Path

import numpy as np
import pytest

try:
    from lammps import lammps
except ImportError:
    lammps = None

pytestmark = pytest.mark.skipif(
    lammps is None, reason="LAMMPS Python module unavailable"
)


def _compact_model():
    value = os.environ.get("SYMMETRIX_LAMMPS_COMPACT_MODEL")
    if not value:
        pytest.skip("Set SYMMETRIX_LAMMPS_COMPACT_MODEL for direct NPT tests.")
    path = Path(value).resolve()
    if not path.is_file():
        pytest.skip(f"Compact MACE model is unavailable: {path}")
    return path


def _host_artifact():
    environment_name = "SYMMETRIX_LAMMPS_JIT_HOST_ARTIFACT_FLOAT64"
    value = os.environ.get(environment_name)
    if not value:
        pytest.skip(
            "Set SYMMETRIX_LAMMPS_JIT_HOST_ARTIFACT_FLOAT64 for direct FP64 NPT tests."
        )
    path = Path(value).resolve()
    if not path.is_file():
        pytest.skip(f"FP64 Execution host artifact is unavailable: {path}")
    return path


def _run_trajectory(
    model,
    streamed_edges,
    integrator,
    steps=5,
    exhaust_skin=False,
    jit_host_artifact=None,
):
    lmp = lammps(cmdargs=["-screen", "none", "-k", "on", "-sf", "kk"])
    fix = (
        "fix dynamics all npt temp 300.0 300.0 0.1 iso 0.0 0.0 1.0"
        if integrator == "npt"
        else "fix dynamics all nve"
    )
    skin_velocity = (
        "group displaced id 1\nvelocity displaced set 6000.0 0.0 0.0 units box"
        if exhaust_skin
        else ""
    )
    artifact_option = (
        "" if jit_host_artifact is None else f" jit_host_artifact {jit_host_artifact}"
    )
    lmp.commands_string(
        f"""
        clear
        units           metal
        atom_style      atomic
        atom_modify     map yes sort 0 0
        boundary        p p p
        region          box block 0 12 0 12 0 12
        create_box      2 box
        create_atoms    1 single 1.0 1.0 1.0 units box
        create_atoms    2 single 3.0 3.0 3.0 units box
        create_atoms    1 single 5.0 5.0 1.0 units box
        create_atoms    2 single 7.0 7.0 3.0 units box
        create_atoms    1 single 1.0 5.0 5.0 units box
        create_atoms    2 single 3.0 7.0 7.0 units box
        create_atoms    1 single 5.0 1.0 5.0 units box
        create_atoms    2 single 7.0 3.0 7.0 units box
        mass            1 26.9815385
        mass            2 14.0067
        pair_style      symmetrix/mace no_domain_decomposition streamed_edges {streamed_edges}{artifact_option}
        pair_coeff      * * {model} Al N
        neighbor        1.0 bin
        neigh_modify    every 1 delay 0 check yes
        velocity        all create 300.0 20260814 mom yes rot no dist gaussian
        {skin_velocity}
        timestep        0.0001
        thermo          1
        thermo_style    custom step pe temp press pxx pyy pzz pxy pxz pyz vol
        {fix}
        run             0
        """
    )
    evaluations_before = lmp.extract_pair("symmetrix_pair_evaluation_count")
    lmp.command(f"run {steps} pre no post no")
    result = {
        "energy": lmp.get_thermo("pe"),
        "pressure": np.array(
            [
                lmp.get_thermo(name)
                for name in ("pxx", "pyy", "pzz", "pxy", "pxz", "pyz")
            ]
        ),
        "volume": lmp.get_thermo("vol"),
        "positions": lmp.numpy.extract_atom("x", nelem=8, dim=3).copy(),
        "forces": lmp.numpy.extract_atom("f", nelem=8, dim=3).copy(),
        "pair_evaluations": lmp.extract_pair("symmetrix_pair_evaluation_count")
        - evaluations_before,
        "graph_rebuilds": lmp.extract_pair("symmetrix_graph_rebuild_count"),
        "geometry_refreshes": lmp.extract_pair("symmetrix_geometry_refresh_count"),
        "geometry_only_updates": lmp.extract_pair(
            "symmetrix_geometry_only_update_count"
        ),
    }
    box = lmp.extract_box()
    result["cell"] = np.array(
        [
            [box[1][0] - box[0][0], 0.0, 0.0],
            [box[2], box[1][1] - box[0][1], 0.0],
            [box[4], box[3], box[1][2] - box[0][2]],
        ]
    )
    lmp.close()
    return result


def test_lammps_kokkos_direct_npt_matches_non_compiled():
    model = _compact_model()
    direct = _run_trajectory(model, "direct", "npt", jit_host_artifact=_host_artifact())
    reference = _run_trajectory(model, "non-compiled", "npt")

    assert direct["pair_evaluations"] == 5
    assert direct["graph_rebuilds"] == 1
    assert direct["geometry_refreshes"] == 6
    assert direct["geometry_only_updates"] == 5
    assert np.max(np.abs(direct["cell"] - 12.0 * np.eye(3))) > 1.0e-12
    assert direct["energy"] == pytest.approx(reference["energy"], abs=2.0e-5)
    assert direct["volume"] == pytest.approx(reference["volume"], rel=2.0e-7)
    np.testing.assert_allclose(
        direct["pressure"], reference["pressure"], rtol=2.0e-5, atol=2.0e-5
    )
    np.testing.assert_allclose(
        direct["positions"], reference["positions"], rtol=2.0e-6, atol=2.0e-6
    )
    np.testing.assert_allclose(
        direct["forces"], reference["forces"], rtol=2.0e-5, atol=2.0e-5
    )
    np.testing.assert_allclose(
        direct["cell"], reference["cell"], rtol=2.0e-7, atol=2.0e-7
    )


def test_lammps_kokkos_direct_nve_retains_topology():
    model = _compact_model()
    artifact = _host_artifact()
    retained = _run_trajectory(model, "direct", "nve", jit_host_artifact=artifact)
    exhausted = _run_trajectory(
        model,
        "direct",
        "nve",
        steps=1,
        exhaust_skin=True,
        jit_host_artifact=artifact,
    )

    assert retained["pair_evaluations"] == 5
    assert retained["graph_rebuilds"] == 1
    assert retained["geometry_refreshes"] == 6
    assert retained["geometry_only_updates"] == 5
    assert exhausted["pair_evaluations"] == 1
    assert exhausted["graph_rebuilds"] == 2
    assert exhausted["geometry_refreshes"] == 2
    assert exhausted["geometry_only_updates"] == 0
