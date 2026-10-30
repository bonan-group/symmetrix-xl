import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


_GLOBAL_RESULT = re.compile(
    r"^(?P<stage>INITIAL|REPEATED|REBUILT) "
    r"pe=(?P<pe>\S+) pxx=(?P<pxx>\S+) pyy=(?P<pyy>\S+) "
    r"pzz=(?P<pzz>\S+) pxy=(?P<pxy>\S+) pxz=(?P<pxz>\S+) "
    r"pyz=(?P<pyz>\S+)$",
    re.MULTILINE,
)


def _required_path(environment_name, description):
    value = os.environ.get(environment_name)
    if not value:
        pytest.skip(f"Set {environment_name} for {description}.")
    path = Path(value).resolve()
    if not path.is_file():
        pytest.skip(f"{description.capitalize()} is unavailable: {path}")
    return path


def _kokkos_arguments():
    arguments = ["-k", "on", "t", "1"]
    if kokkos_gpus := os.environ.get("SYMMETRIX_LAMMPS_KOKKOS_GPUS"):
        arguments.extend(("g", kokkos_gpus))
    arguments.extend(("-sf", "kk"))
    return arguments


def _read_dump(path):
    lines = path.read_text().splitlines()
    header = lines.index("ITEM: ATOMS id type x y z fx fy fz c_atom_energy proc")
    return np.loadtxt(lines[header + 1 :], ndmin=2)


def _read_global_results(log_path):
    results = {}
    for match in _GLOBAL_RESULT.finditer(log_path.read_text()):
        results[match.group("stage").lower()] = np.asarray(
            [
                float(match.group(name))
                for name in ("pe", "pxx", "pyy", "pzz", "pxy", "pxz", "pyz")
            ]
        )
    assert tuple(results) == ("initial", "repeated", "rebuilt")
    return results


def _run_case(tmp_path, executable, name, pair_style, pair_coefficients):
    data_path = tmp_path / f"{name}.data"
    input_path = tmp_path / f"in.{name}"
    log_path = tmp_path / f"{name}.log"
    dumps = {
        stage: tmp_path / f"{name}-{stage}.dump"
        for stage in ("initial", "repeated", "rebuilt")
    }
    data_path.write_text(
        """LAMMPS hybrid MPI communication fixture

8 atoms
2 atom types

0 12 xlo xhi
0 8 ylo yhi
0 8 zlo zhi

Atoms # atomic

1 1 2.5 4.0 4.0
2 2 3.5 4.1 4.0
3 1 4.5 3.9 4.1
4 2 5.5 4.0 3.9
5 1 6.5 4.1 4.0
6 2 7.5 3.9 4.1
7 1 8.5 4.0 3.9
8 2 9.5 4.1 4.0
"""
    )
    input_path.write_text(
        f"""
units           metal
atom_style      atomic/kk
atom_modify     map yes sort 0 0
boundary        p p p
package         kokkos neigh half newton on
newton          on
processors      2 1 1
read_data       {data_path}
mass            1 1.008
mass            2 15.999
pair_style      {pair_style}
{pair_coefficients}
neighbor        0.2 bin
neigh_modify    every 1 delay 0 check yes
compute         atom_energy all pe/atom
thermo          1
thermo_style    custom step atoms pe pxx pyy pzz pxy pxz pyz
thermo_modify   format float %.16g
run             0
write_dump      all custom {dumps['initial']} id type x y z fx fy fz c_atom_energy proc modify sort id format float %.16g
print           "INITIAL pe=$(pe:%.16g) pxx=$(pxx:%.16g) pyy=$(pyy:%.16g) pzz=$(pzz:%.16g) pxy=$(pxy:%.16g) pxz=$(pxz:%.16g) pyz=$(pyz:%.16g)"
run             0
write_dump      all custom {dumps['repeated']} id type x y z fx fy fz c_atom_energy proc modify sort id format float %.16g
print           "REPEATED pe=$(pe:%.16g) pxx=$(pxx:%.16g) pyy=$(pyy:%.16g) pzz=$(pzz:%.16g) pxy=$(pxy:%.16g) pxz=$(pxz:%.16g) pyz=$(pyz:%.16g)"
group           moved id 4
displace_atoms  moved move 0.75 0.02 0.0 units box
run             0
write_dump      all custom {dumps['rebuilt']} id type x y z fx fy fz c_atom_energy proc modify sort id format float %.16g
print           "REBUILT pe=$(pe:%.16g) pxx=$(pxx:%.16g) pyy=$(pyy:%.16g) pzz=$(pzz:%.16g) pxy=$(pxy:%.16g) pxz=$(pxz:%.16g) pyz=$(pyz:%.16g)"
"""
    )
    mpiexec = os.environ.get("MPIEXEC_EXECUTABLE") or shutil.which("mpiexec")
    if not mpiexec:
        pytest.skip("mpiexec is unavailable")
    environment = os.environ.copy()
    environment.update(
        {
            "SYMMETRIX_JIT_POLICY": "none",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_PROC_BIND": "false",
        }
    )
    subprocess.run(
        [
            mpiexec,
            "-n",
            "2",
            str(executable),
            "-screen",
            "none",
            "-log",
            str(log_path),
            *_kokkos_arguments(),
            "-in",
            str(input_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
        env=environment,
    )
    return {
        "atoms": {stage: _read_dump(path) for stage, path in dumps.items()},
        "global": _read_global_results(log_path),
    }


def _assert_same(left, right):
    for stage in ("initial", "repeated", "rebuilt"):
        np.testing.assert_array_equal(
            left["atoms"][stage][:, :2], right["atoms"][stage][:, :2]
        )
        np.testing.assert_allclose(
            left["atoms"][stage][:, 2:9],
            right["atoms"][stage][:, 2:9],
            rtol=1.0e-10,
            atol=1.0e-8,
        )
        np.testing.assert_allclose(
            left["global"][stage],
            right["global"][stage],
            rtol=1.0e-10,
            atol=1.0e-7,
        )


def _assert_repeated(case):
    np.testing.assert_array_equal(
        case["atoms"]["initial"][:, :2], case["atoms"]["repeated"][:, :2]
    )
    np.testing.assert_allclose(
        case["atoms"]["initial"][:, 2:9],
        case["atoms"]["repeated"][:, 2:9],
        rtol=1.0e-10,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        case["global"]["initial"],
        case["global"]["repeated"],
        rtol=1.0e-10,
        atol=1.0e-7,
    )


def _assert_additive(mace, correction, overlay):
    for stage in ("initial", "repeated", "rebuilt"):
        mace_atoms = mace["atoms"][stage]
        correction_atoms = correction["atoms"][stage]
        overlay_atoms = overlay["atoms"][stage]
        np.testing.assert_array_equal(mace_atoms[:, :2], correction_atoms[:, :2])
        np.testing.assert_array_equal(mace_atoms[:, :2], overlay_atoms[:, :2])
        np.testing.assert_allclose(
            mace_atoms[:, 2:5], overlay_atoms[:, 2:5], rtol=0.0, atol=1.0e-12
        )
        np.testing.assert_allclose(
            mace_atoms[:, 5:9] + correction_atoms[:, 5:9],
            overlay_atoms[:, 5:9],
            rtol=1.0e-10,
            atol=1.0e-8,
        )
        np.testing.assert_allclose(
            mace["global"][stage] + correction["global"][stage],
            overlay["global"][stage],
            rtol=1.0e-10,
            atol=1.0e-7,
        )


def _hybrid_requirements():
    executable = _required_path(
        "SYMMETRIX_LAMMPS_EXECUTABLE",
        "a Kokkos and MPI-enabled LAMMPS executable",
    )
    model = _required_path(
        "SYMMETRIX_LAMMPS_LEGACY_MODEL", "a legacy H/O standard-MACE model"
    )
    return executable, model


def test_mpi_message_passing_rejects_non_overlay_hybrid(tmp_path):
    executable = _required_path(
        "SYMMETRIX_LAMMPS_EXECUTABLE",
        "a Kokkos and MPI-enabled LAMMPS executable",
    )
    input_path = tmp_path / "in.rejected-hybrid"
    log_path = tmp_path / "rejected-hybrid.log"
    input_path.write_text(
        "pair_style hybrid symmetrix/mace/kk mpi_message_passing zero 6.0\n"
    )
    result = subprocess.run(
        [
            str(executable),
            "-screen",
            "none",
            "-log",
            str(log_path),
            *_kokkos_arguments(),
            "-in",
            str(input_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert (
        "mpi_message_passing is supported under hybrid/overlay/kk only"
        in log_path.read_text()
    )


def test_mpi_message_passing_matches_direct_under_hybrid_overlay(tmp_path):
    executable, model = _hybrid_requirements()
    direct = _run_case(
        tmp_path,
        executable,
        "direct",
        "symmetrix/mace/kk mpi_message_passing",
        f"pair_coeff * * {model} H O",
    )
    overlay = _run_case(
        tmp_path,
        executable,
        "overlay-zero",
        "hybrid/overlay/kk symmetrix/mace/kk mpi_message_passing zero 6.0",
        f"""pair_coeff * * symmetrix/mace/kk {model} H O
pair_coeff * * zero 6.0""",
    )

    _assert_same(direct, overlay)
    _assert_repeated(direct)
    _assert_repeated(overlay)
    assert direct["atoms"]["initial"][3, 9] != direct["atoms"]["rebuilt"][3, 9]


def test_host_d3_is_additive_under_mpi_hybrid_overlay(tmp_path):
    executable, model = _hybrid_requirements()
    help_result = subprocess.run(
        [str(executable), "-log", "none", "-help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if "dispersion/d3" not in help_result.stdout:
        pytest.skip("LAMMPS EXTRA-PAIR dispersion/d3 is unavailable")
    direct = _run_case(
        tmp_path,
        executable,
        "d3-direct",
        "symmetrix/mace/kk mpi_message_passing",
        f"pair_coeff * * {model} H O",
    )
    d3 = _run_case(
        tmp_path,
        executable,
        "d3-only",
        "dispersion/d3 bj pbe 3.0 3.0",
        "pair_coeff * * H O",
    )
    overlay = _run_case(
        tmp_path,
        executable,
        "overlay-d3",
        "hybrid/overlay/kk symmetrix/mace/kk mpi_message_passing "
        "dispersion/d3 bj pbe 3.0 3.0",
        f"""pair_coeff * * symmetrix/mace/kk {model} H O
pair_coeff * * dispersion/d3 H O""",
    )

    _assert_additive(direct, d3, overlay)
    _assert_repeated(direct)
    _assert_repeated(d3)
    _assert_repeated(overlay)
