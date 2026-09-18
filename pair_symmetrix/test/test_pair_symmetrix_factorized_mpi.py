import functools
import itertools
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


def _required_path(environment_name, description):
    value = os.environ.get(environment_name)
    if not value:
        pytest.skip(f"Set {environment_name} for {description}.")
    path = Path(value).resolve()
    if not path.is_file():
        pytest.skip(f"{description.capitalize()} is unavailable: {path}")
    return path


def _host_artifact_arguments(precision):
    precision_name = f"SYMMETRIX_LAMMPS_JIT_HOST_ARTIFACT_{precision.upper()}"
    environment_name = precision_name
    if precision == "float32" and not os.environ.get(environment_name):
        environment_name = "SYMMETRIX_LAMMPS_JIT_HOST_ARTIFACT"
    if os.environ.get(environment_name):
        return {
            "jit_host_artifact": _required_path(
                environment_name,
                f"a {precision} Execution host artifact",
            )
        }
    if precision == "float32" and os.environ.get("SYMMETRIX_LAMMPS_JIT_ARTIFACT"):
        arguments = {
            "jit_device_artifact": _required_path(
                "SYMMETRIX_LAMMPS_JIT_ARTIFACT",
                "a float32 Execution device artifact",
            )
        }
        for stage in ("M0", "R0"):
            environment_name = (
                f"SYMMETRIX_LAMMPS_JIT_{stage}_DEVICE_ARTIFACT_{precision.upper()}"
            )
            if os.environ.get(environment_name):
                arguments[f"jit_{stage.lower()}_device_artifact"] = _required_path(
                    environment_name,
                    f"a {precision} Execution {stage} device artifact",
                )
        schedule_name = f"SYMMETRIX_LAMMPS_JIT_M0_DEVICE_SCHEDULE_{precision.upper()}"
        if "jit_m0_device_artifact" in arguments and os.environ.get(schedule_name):
            arguments["jit_m0_device_schedule"] = os.environ[schedule_name]
        return arguments
    pytest.skip(f"Set {precision_name} for direct {precision} MPI tests.")


def _read_dump(path):
    lines = path.read_text().splitlines()
    header = lines.index("ITEM: ATOMS id type x y z fx fy fz c_atom_energy proc")
    return np.loadtxt(lines[header + 1 :], ndmin=2)


def _read_result(log_path, prefix):
    pattern = re.compile(
        rf"^{prefix} mode=\S+ "
        r"pe=(\S+) pxx=(\S+) pyy=(\S+) pzz=(\S+) "
        r"pxy=(\S+) pxz=(\S+) pyz=(\S+)$",
        re.MULTILINE,
    )
    match = pattern.search(log_path.read_text())
    assert match is not None
    return np.asarray([float(value) for value in match.groups()])


_CAPACITY_RECORD = re.compile(
    r"Symmetrix rank (?P<rank>\d+) graph capacity: "
    r"active receivers=(?P<active_receivers>\d+), "
    r"features=(?P<active_features>\d+), edges=(?P<active_edges>\d+); "
    r"planned receivers=(?P<planned_receivers>\d+), "
    r"features=(?P<planned_features>\d+), edges=(?P<planned_edges>\d+); "
    r"planned bytes=(?P<planned_bytes>\d+); policy=(?P<policy>[^;]+); "
    r"profile=(?P<profile>[^;]+); plan=(?P<plan>[^;]+); "
    r"fixed workspace=(?P<fixed_workspace>[^;]+); "
    r"workspace receivers=(?P<workspace_receivers>\d+), "
    r"edges=(?P<workspace_edges>\d+), bytes=(?P<workspace_bytes>\d+), "
    r"batches=(?P<workspace_batches>\d+); "
    r"pair edge geometry bytes=(?P<pair_geometry_bytes>\d+), "
    r"feature position bytes=(?P<feature_position_bytes>\d+); "
    r"graph replacements=(?P<graph_replacements>\d+), "
    r"graph updates=(?P<graph_updates>\d+), "
    r"geometry allocations=(?P<geometry_allocations>\d+), "
    r"result allocations=(?P<result_allocations>\d+), "
    r"M0 replacements=(?P<m0_replacements>\d+), "
    r"M0 adjoint allocations=(?P<m0_adjoint_allocations>\d+), "
    r"M0 alias detaches=(?P<m0_alias_detaches>\d+), "
    r"M0 alias active=(?P<m0_alias_active>[01]), "
    r"communicated H1 allocations=(?P<h1_allocations>[0-9.]+)"
)


def _read_capacity_records(log_text):
    records = []
    for match in _CAPACITY_RECORD.finditer(log_text):
        record = {}
        for name, value in match.groupdict().items():
            if name in {"policy", "profile", "plan", "fixed_workspace"}:
                record[name] = value
            elif name == "h1_allocations":
                record[name] = int(float(value))
            else:
                record[name] = int(value)
        records.append(record)
    return records


_TIMING_FIELDS = (
    "critical_pair_us_per_atom_eval",
    "critical_noncommunication_us_per_atom_eval",
    "critical_comm_us_per_atom_eval",
    "critical_comm_percent",
    "critical_forward_us_per_atom_eval",
    "critical_reverse_us_per_atom_eval",
    "rank_comm_percent_min",
    "rank_comm_percent_average",
    "rank_comm_percent_max",
    "comm_imbalance_max_over_mean",
    "forward_calls_per_eval_min",
    "forward_calls_per_eval_max",
    "reverse_calls_per_eval_min",
    "reverse_calls_per_eval_max",
)


def _read_timing_record(log_text):
    lines = [
        line.strip()
        for line in log_text.splitlines()
        if line.startswith("SYMMETRIX_TIMING ")
    ]
    assert len(lines) == 1
    fields = lines[0].split()[1:]
    values = {}
    for field in fields:
        name, separator, value = field.partition("=")
        assert separator
        values[name] = float(value)
    assert tuple(values) == ("scalar", *_TIMING_FIELDS)
    return values


_WURTZITE_BASIS = (
    (1, (0.0, 0.0, 0.0)),
    (2, (0.0, 1.0 / 3.0, 0.1199379857679)),
    (1, (0.0, 1.0 / 3.0, 0.5)),
    (2, (0.0, 0.0, 0.6199379857679)),
    (1, (0.5, 0.5, 0.0)),
    (2, (0.5, 5.0 / 6.0, 0.1199379857679)),
    (1, (0.5, 5.0 / 6.0, 0.5)),
    (2, (0.5, 0.5, 0.6199379857679)),
)

_WURTZITE_CELL_LENGTHS = np.asarray([6.224, 10.78028422630869, 9.964])

_PROCESSOR_GRIDS = {
    1: "1 1 1",
    2: "1 1 2",
    4: "1 2 2",
}


def _assert_physical_wurtzite(fractional, atom_types, cell):
    coordination = np.zeros(len(fractional), dtype=np.int32)
    minimum_distance = np.inf
    minimum_pair_types = None
    for receiver in range(len(fractional)):
        for source in range(len(fractional)):
            for shift in itertools.product((-1, 0, 1), repeat=3):
                if receiver == source and shift == (0, 0, 0):
                    continue
                displacement = (
                    fractional[source] - fractional[receiver] + shift
                ) @ cell
                distance = np.linalg.norm(displacement)
                if source != receiver and distance < minimum_distance:
                    minimum_distance = distance
                    minimum_pair_types = (atom_types[receiver], atom_types[source])
                if distance < 2.05:
                    coordination[receiver] += 1
                    assert atom_types[receiver] != atom_types[source]
    assert 1.8 < minimum_distance < 2.0
    assert minimum_pair_types == (1, 2) or minimum_pair_types == (2, 1)
    np.testing.assert_array_equal(coordination, np.full(len(fractional), 4))


@functools.lru_cache
def _physical_aln_fixture(triclinic):
    lengths = _WURTZITE_CELL_LENGTHS
    tilts = np.asarray([0.12, 0.08, 0.10]) if triclinic else np.zeros(3)
    xy, xz, yz = tilts
    cell = np.asarray(
        [
            [lengths[0], 0.0, 0.0],
            [xy, lengths[1], 0.0],
            [xz, yz, lengths[2]],
        ]
    )

    fractional = []
    atom_types = []
    migrant = None
    for iz, iy, ix in itertools.product(range(2), repeat=3):
        for basis_index, (atom_type, basis) in enumerate(_WURTZITE_BASIS):
            if ix == iy == iz == 0 and basis_index == 3:
                migrant = len(fractional)
            fractional.append(
                ((basis[0] + ix) / 2, (basis[1] + iy) / 2, (basis[2] + iz) / 2)
            )
            atom_types.append(atom_type)
    fractional = np.asarray(fractional)
    atom_types = np.asarray(atom_types, dtype=np.int32)

    # Translate the intact crystal so one nitrogen sits just below z/lz = 0.5.
    fractional[:, 2] = (fractional[:, 2] + 0.498 - fractional[migrant, 2]) % 1.0
    phases = np.asarray([0.7, 1.1, 1.7])
    perturbation = 0.008 * np.sin(
        (np.arange(len(fractional))[:, None] + 1) * phases[None, :]
    )
    fractional = ((fractional @ cell + perturbation) @ np.linalg.inv(cell)) % 1.0
    fractional[migrant, 2] = 0.498

    order = [index for index in range(len(fractional)) if index != migrant]
    order.insert(2, migrant)
    fractional = fractional[order]
    atom_types = atom_types[order]
    assert atom_types[2] == 2
    _assert_physical_wurtzite(fractional, atom_types, cell)

    migrated_positions = fractional @ cell
    migrated_positions[2, 2] += 0.04
    migrated_fractional = migrated_positions @ np.linalg.inv(cell)
    _assert_physical_wurtzite(migrated_fractional, atom_types, cell)
    assert fractional[2, 2] < 0.5 < migrated_fractional[2, 2]

    positions = fractional @ cell
    tilt_line = f"{xy:.16g} {xz:.16g} {yz:.16g} xy xz yz\n" if triclinic else ""
    atoms_section = "\n".join(
        "{} {} {:.16g} {:.16g} {:.16g}".format(atom_id, atom_type, *position)
        for atom_id, (atom_type, position) in enumerate(
            zip(atom_types, positions, strict=True), start=1
        )
    )
    return f"""LAMMPS data file for direct MPI qualification

{len(positions)} atoms
2 atom types

0 {lengths[0]:.16g} xlo xhi
0 {lengths[1]:.16g} ylo yhi
0 {lengths[2]:.16g} zlo zhi
{tilt_line}
Masses

1 26.9815385
2 14.0067

Atoms # atomic

{atoms_section}
"""


def _run_lammps_case(
    tmp_path,
    executable,
    model,
    mode,
    electric_field=None,
    precision="float64",
    triclinic=False,
    ranks=2,
    domain_mode="mpi_message_passing",
    profile="speed",
    low_memory=None,
    allow_fixed_workspace=False,
    jit_host_artifact=None,
    jit_device_artifact=None,
    jit_m0_device_artifact=None,
    jit_r0_device_artifact=None,
    jit_m0_device_schedule="chunk32",
    pair_comm="default",
    migrated_run_commands="run 0",
    debug_execution_plan=None,
    debug_single_layer_workspace_receivers=None,
    debug_dual_layer_workspace_receivers=None,
    pair_backend="kokkos",
    capture_timing=False,
):
    label_parts = [mode, precision, "triclinic" if triclinic else "orthogonal"]
    label_parts.append(pair_backend)
    if electric_field is not None:
        label_parts.append("field")
    if domain_mode == "no_domain_decomposition":
        label_parts.append("single-rank")
    if low_memory is not None:
        label_parts.append(f"legacy-low-memory-{'yes' if low_memory else 'no'}")
    elif profile is None:
        label_parts.append("default-profile")
    elif profile != "speed":
        label_parts.append(profile)
    if allow_fixed_workspace:
        label_parts.append("fixed-workspace")
    if pair_comm != "default":
        label_parts.append(pair_comm)
    label = "-".join(label_parts)
    initial_dump = tmp_path / f"{label}.dump"
    migrated_dump = tmp_path / f"{label}-migrated.dump"
    log_path = tmp_path / f"{label}.log"
    input_path = tmp_path / f"in.{label}"
    field_option = ""
    if electric_field is not None:
        field_option = "electric_field " + " ".join(map(str, electric_field)) + " "
    profile_option = "" if profile is None else f"profile {profile} "
    low_memory_option = (
        "" if low_memory is None else f"low_memory {'yes' if low_memory else 'no'} "
    )
    fixed_workspace_option = (
        "allow_fixed_workspace yes " if allow_fixed_workspace else ""
    )
    debug_option = ""
    if debug_execution_plan is not None:
        debug_option += f"_debug_execution_plan {debug_execution_plan} "
    if debug_single_layer_workspace_receivers is not None:
        debug_option += (
            "_debug_single_layer_workspace_receivers "
            f"{debug_single_layer_workspace_receivers} "
        )
    if debug_dual_layer_workspace_receivers is not None:
        debug_option += (
            "_debug_dual_layer_workspace_receivers "
            f"{debug_dual_layer_workspace_receivers} "
        )
    if jit_host_artifact is not None and jit_device_artifact is not None:
        raise ValueError("Specify only one JIT artifact")
    if jit_host_artifact is not None:
        artifact_option = f"jit_host_artifact {jit_host_artifact} "
    elif jit_device_artifact is not None:
        artifact_option = f"jit_device_artifact {jit_device_artifact} "
    else:
        artifact_option = ""
    if profile == "capacity" and jit_m0_device_artifact is not None:
        artifact_option += (
            f"jit_m0_device_artifact {jit_m0_device_artifact} "
            f"jit_m0_device_schedule {jit_m0_device_schedule} "
        )
    if profile == "capacity" and jit_r0_device_artifact is not None:
        artifact_option += f"jit_r0_device_artifact {jit_r0_device_artifact} "
    data_path = tmp_path / f"{label}.data"
    data_path.write_text(_physical_aln_fixture(triclinic))
    try:
        processor_grid = _PROCESSOR_GRIDS[ranks]
    except KeyError as error:
        raise ValueError(f"Unsupported MPI rank count: {ranks}") from error
    processors = f"processors      {processor_grid}"
    if pair_comm == "default":
        pair_comm_option = ""
    elif pair_comm == "legacy":
        pair_comm_option = " comm/pair/forward no comm/pair/reverse no"
    else:
        raise ValueError(f"Unsupported Kokkos pair communication mode: {pair_comm}")
    if pair_backend == "kokkos":
        pair_style = (
            "symmetrix/mace/float32/kk"
            if precision == "float32"
            else "symmetrix/mace/kk"
        )
        atom_style = "atomic/kk"
        package_command = (
            f"package         kokkos neigh half newton on{pair_comm_option}"
        )
        pair_options = (
            f"streamed_edges {mode} {profile_option}{low_memory_option}"
            f"{fixed_workspace_option}{debug_option}{artifact_option}"
        )
    elif pair_backend == "cpu":
        if precision != "float64":
            raise ValueError("The CPU LAMMPS pair style supports only float64")
        if pair_comm != "default":
            raise ValueError(
                "Kokkos pair communication options require pair_backend='kokkos'"
            )
        pair_style = "symmetrix/mace"
        atom_style = "atomic"
        package_command = ""
        pair_options = ""
    else:
        raise ValueError(f"Unsupported pair backend: {pair_backend}")
    timing_compute = (
        "compute         symmetrix_timing all symmetrix/timing"
        if capture_timing
        else ""
    )
    timing_print = ""
    if capture_timing:
        timing_fields = " ".join(
            f"{name}=$(c_symmetrix_timing[{index}]:%.17g)"
            for index, name in enumerate(_TIMING_FIELDS, start=1)
        )
        timing_print = (
            'print           "SYMMETRIX_TIMING '
            "scalar=$(c_symmetrix_timing:%.17g) "
            f'{timing_fields}"'
        )
    input_path.write_text(
        f"""
units           metal
atom_style      {atom_style}
atom_modify     map yes sort 1 0.0
boundary        p p p
{package_command}
newton          on
{processors}
read_data       {data_path}
pair_style      {pair_style} {field_option}{domain_mode} {pair_options}
pair_coeff      * * {model} Al N
neighbor        1.0 bin
neigh_modify    every 1 delay 0 check yes
compute         atom_energy all pe/atom
{timing_compute}
thermo          1
thermo_style    custom step atoms pe pxx pyy pzz pxy pxz pyz
thermo_modify   format float %.16g
dump            state all custom 1 {initial_dump} id type x y z fx fy fz c_atom_energy proc
dump_modify     state sort id format float %.16g
run             0
print           "RESULT mode={mode} pe=$(pe:%.16g) pxx=$(pxx:%.16g) pyy=$(pyy:%.16g) pzz=$(pzz:%.16g) pxy=$(pxy:%.16g) pxz=$(pxz:%.16g) pyz=$(pyz:%.16g)"
group           migrant id 3
displace_atoms  migrant move 0.0 0.0 0.04 units box
undump          state
reset_timestep  1
{migrated_run_commands}
write_dump      all custom {migrated_dump} id type x y z fx fy fz c_atom_energy proc modify sort id format float %.16g
print           "MIGRATED mode={mode} pe=$(pe:%.16g) pxx=$(pxx:%.16g) pyy=$(pyy:%.16g) pzz=$(pzz:%.16g) pxy=$(pxy:%.16g) pxz=$(pxz:%.16g) pyz=$(pyz:%.16g)"
{timing_print}
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
    kokkos_arguments = []
    if pair_backend == "kokkos":
        kokkos_arguments = ["-k", "on", "t", "1"]
        kokkos_gpus = os.environ.get("SYMMETRIX_LAMMPS_KOKKOS_GPUS")
        if kokkos_gpus:
            kokkos_arguments.extend(("g", kokkos_gpus))
        kokkos_arguments.extend(("-sf", "kk"))
    subprocess.run(
        [
            mpiexec,
            "-n",
            str(ranks),
            str(executable),
            "-screen",
            "none",
            "-log",
            str(log_path),
            *kokkos_arguments,
            "-in",
            str(input_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
        env=environment,
    )
    log_text = log_path.read_text()
    if low_memory is None:
        expected_profile = "capacity" if profile is None else profile
    else:
        expected_profile = "capacity" if low_memory else "speed"
    if pair_backend == "kokkos" and mode in {"direct", "direct_streamed"}:
        assert f"Symmetrix execution profile='{expected_profile}'" in log_text
    result = {
        "initial_atoms": _read_dump(initial_dump),
        "migrated_atoms": _read_dump(migrated_dump),
        "initial_global": _read_result(log_path, "RESULT"),
        "migrated_global": _read_result(log_path, "MIGRATED"),
        "capacity_records": _read_capacity_records(log_text),
        "timing": _read_timing_record(log_text) if capture_timing else None,
        "log_text": log_text,
    }
    energy_tolerance = 2.0e-5 if precision == "float32" else 1.0e-10
    for stage in ("initial", "migrated"):
        assert np.sum(result[f"{stage}_atoms"][:, 8]) == pytest.approx(
            result[f"{stage}_global"][0], rel=1.0e-6, abs=energy_tolerance
        )
    initial_migrant = result["initial_atoms"][result["initial_atoms"][:, 0] == 3]
    migrated_migrant = result["migrated_atoms"][result["migrated_atoms"][:, 0] == 3]
    assert initial_migrant.shape == migrated_migrant.shape == (1, 10)
    if ranks > 1:
        assert initial_migrant[0, 9] != migrated_migrant[0, 9]
    return result


def _run_timing_case(tmp_path, executable, model, pair_backend, ranks, domain_mode):
    label = f"timing-{pair_backend}-{ranks}-{domain_mode}"
    data_path = tmp_path / f"{label}.data"
    input_path = tmp_path / f"in.{label}"
    log_path = tmp_path / f"{label}.log"
    data_path.write_text(_physical_aln_fixture(False))

    if pair_backend == "kokkos":
        atom_style = "atomic/kk"
        package_command = "package kokkos neigh half newton on"
        pair_style = "symmetrix/mace/kk"
        pair_options = " streamed_edges generic"
        kokkos_arguments = ["-k", "on", "t", "1"]
        kokkos_gpus = os.environ.get("SYMMETRIX_LAMMPS_KOKKOS_GPUS")
        if kokkos_gpus:
            kokkos_arguments.extend(("g", kokkos_gpus))
        kokkos_arguments.extend(("-sf", "kk"))
    elif pair_backend == "cpu":
        atom_style = "atomic"
        package_command = ""
        pair_style = "symmetrix/mace"
        pair_options = ""
        kokkos_arguments = []
    else:
        raise ValueError(f"Unsupported pair backend: {pair_backend}")

    timing_thermo = " ".join(
        (
            "c_symmetrix_timing",
            "c_symmetrix_timing[1]",
            "c_symmetrix_timing[2]",
            "c_symmetrix_timing[3]",
            "c_symmetrix_timing[10]",
        )
    )
    timing_record = " ".join(
        ["scalar=$(c_symmetrix_timing:%.17g)"]
        + [
            f"{name}=$(c_symmetrix_timing[{index}]:%.17g)"
            for index, name in enumerate(_TIMING_FIELDS, start=1)
        ]
    )
    input_path.write_text(
        f"""
units           metal
atom_style      {atom_style}
atom_modify     map yes sort 1 0.0
boundary        p p p
{package_command}
newton          on
processors      {_PROCESSOR_GRIDS[ranks]}
read_data       {data_path}
pair_style      {pair_style} {domain_mode}{pair_options}
pair_coeff      * * {model} Al N
neighbor        1.0 bin
neigh_modify    every 1 delay 0 check yes
compute         symmetrix_timing all symmetrix/timing
thermo          2
thermo_style    custom step atoms {timing_thermo}
thermo_modify   colname auto line one format float %.17g
run             2 post no
print           "SYMMETRIX_TIMING {timing_record}"
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
            str(ranks),
            str(executable),
            "-screen",
            "none",
            "-log",
            str(log_path),
            *kokkos_arguments,
            "-in",
            str(input_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
        env=environment,
    )
    log_text = log_path.read_text()
    assert re.search(
        r"Step\s+Atoms\s+SxH1CommPct\s+SxPair\s+SxNonComm\s+SxH1Comm\s+SxH1Imbal",
        log_text,
    )
    return _read_timing_record(log_text)


def _assert_timing_common(record, ranks):
    assert all(np.isfinite(value) for value in record.values())
    assert record["scalar"] == pytest.approx(record["critical_comm_percent"])
    assert record["critical_pair_us_per_atom_eval"] == pytest.approx(
        record["critical_noncommunication_us_per_atom_eval"]
        + record["critical_comm_us_per_atom_eval"],
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert record["critical_comm_us_per_atom_eval"] == pytest.approx(
        record["critical_forward_us_per_atom_eval"]
        + record["critical_reverse_us_per_atom_eval"],
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert (
        record["rank_comm_percent_min"]
        <= record["rank_comm_percent_average"]
        <= record["rank_comm_percent_max"]
    )
    assert (
        record["rank_comm_percent_min"]
        <= record["critical_comm_percent"]
        <= record["rank_comm_percent_max"]
    )
    assert 0.0 <= record["comm_imbalance_max_over_mean"] <= ranks
    for direction in ("forward", "reverse"):
        minimum = record[f"{direction}_calls_per_eval_min"]
        maximum = record[f"{direction}_calls_per_eval_max"]
        assert 0.0 <= minimum <= maximum


def _run_direct_dynamics_case(
    tmp_path,
    executable,
    model,
    label,
    run_commands,
    precision="float32",
    ranks=2,
    **artifact_arguments,
):
    data_path = tmp_path / f"{label}.data"
    dump_path = tmp_path / f"{label}.dump"
    log_path = tmp_path / f"{label}.log"
    input_path = tmp_path / f"in.{label}"
    data_path.write_text(_physical_aln_fixture(False))
    pair_style = (
        "symmetrix/mace/float32/kk" if precision == "float32" else "symmetrix/mace/kk"
    )
    artifact_option = ""
    if artifact_arguments.get("jit_host_artifact") is not None:
        artifact_option = (
            f"jit_host_artifact {artifact_arguments['jit_host_artifact']} "
        )
    elif artifact_arguments.get("jit_device_artifact") is not None:
        artifact_option = (
            f"jit_device_artifact {artifact_arguments['jit_device_artifact']} "
        )
    if artifact_arguments.get("jit_m0_device_artifact") is not None:
        artifact_option += (
            "jit_m0_device_artifact "
            f"{artifact_arguments['jit_m0_device_artifact']} "
            "jit_m0_device_schedule "
            f"{artifact_arguments.get('jit_m0_device_schedule', 'chunk32')} "
        )
    if artifact_arguments.get("jit_r0_device_artifact") is not None:
        artifact_option += (
            f"jit_r0_device_artifact {artifact_arguments['jit_r0_device_artifact']} "
        )
    input_path.write_text(
        f"""
units           metal
atom_style      atomic/kk
atom_modify     map yes sort 1 0.0
boundary        p p p
package         kokkos neigh half newton on
newton          on
processors      {_PROCESSOR_GRIDS[ranks]}
read_data       {data_path}
pair_style      {pair_style} mpi_message_passing streamed_edges direct profile capacity {artifact_option}
pair_coeff      * * {model} Al N
neighbor        1.0 bin
neigh_modify    every 1 delay 0 check yes
compute         atom_energy all pe/atom
velocity        all create 25.0 20260831 mom yes rot no dist gaussian loop geom
fix             dynamics all nve
thermo          1
thermo_style    custom step atoms pe pxx pyy pzz pxy pxz pyz
thermo_modify   format float %.16g
{run_commands}
write_dump      all custom {dump_path} id type x y z fx fy fz c_atom_energy proc modify sort id format float %.16g
print           "RESULT mode=direct pe=$(pe:%.16g) pxx=$(pxx:%.16g) pyy=$(pyy:%.16g) pzz=$(pzz:%.16g) pxy=$(pxy:%.16g) pxz=$(pxz:%.16g) pyz=$(pyz:%.16g)"
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
    kokkos_arguments = ["-k", "on", "t", "1"]
    kokkos_gpus = os.environ.get("SYMMETRIX_LAMMPS_KOKKOS_GPUS")
    if kokkos_gpus:
        kokkos_arguments.extend(("g", kokkos_gpus))
    subprocess.run(
        [
            mpiexec,
            "-n",
            str(ranks),
            str(executable),
            "-screen",
            "none",
            "-log",
            str(log_path),
            *kokkos_arguments,
            "-sf",
            "kk",
            "-in",
            str(input_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
        env=environment,
    )
    log_text = log_path.read_text()
    assert "Symmetrix execution profile='capacity'" in log_text
    assert "profile=capacity; plan=" in log_text
    return _read_dump(dump_path), _read_result(log_path, "RESULT")


def _assert_direct_dynamics_close(blocked, monolithic):
    blocked_atoms, blocked_global = blocked
    monolithic_atoms, monolithic_global = monolithic

    np.testing.assert_array_equal(blocked_atoms[:, :2], monolithic_atoms[:, :2])
    coordinate_delta = blocked_atoms[:, 2:5] - monolithic_atoms[:, 2:5]
    coordinate_delta -= (
        np.rint(coordinate_delta / _WURTZITE_CELL_LENGTHS) * _WURTZITE_CELL_LENGTHS
    )
    np.testing.assert_allclose(coordinate_delta, 0.0, rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(
        blocked_atoms[:, 5:9], monolithic_atoms[:, 5:9], rtol=2.0e-4, atol=1.0e-4
    )
    # Run boundaries may trigger different, but physically equivalent, ownership.
    np.testing.assert_allclose(blocked_global, monolithic_global, rtol=2.0e-4, atol=0.1)


def _run_kokkos_hybrid_index_case(tmp_path, executable, model, precision, domain_mode):
    label = f"hybrid-index-{precision}-{domain_mode}"
    input_path = tmp_path / f"in.{label}"
    data_path = tmp_path / f"{label}.data"
    initial_dump = tmp_path / f"{label}.dump"
    log_path = tmp_path / f"{label}.log"
    pair_style = (
        "symmetrix/mace/float32" if precision == "float32" else "symmetrix/mace"
    )
    data_path.write_text(
        """Hybrid per-atom energy indexing fixture

4 atoms
2 atom types

0 20 xlo xhi
0 20 ylo yhi
0 20 zlo zhi

Atoms # atomic

1 1 10.0 10.0 10.0
2 2  1.5  5.0  5.0
3 1  5.0 10.0 10.0
4 2 11.5  5.0  5.0
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
read_data       {data_path}
mass            1 1.008
mass            2 15.999
pair_style      hybrid {pair_style} {domain_mode} zero 6.0
pair_coeff      * * {pair_style} {model} H O
pair_coeff      1 * zero 6.0
compute         atom_energy all pe/atom
thermo          1
thermo_style    custom step atoms pe pxx pyy pzz pxy pxz pyz
thermo_modify   format float %.16g
dump            state all custom 1 {initial_dump} id type x y z fx fy fz c_atom_energy proc
dump_modify     state sort id format float %.16g
run             0
print           "RESULT mode=generic pe=$(pe:%.16g) pxx=$(pxx:%.16g) pyy=$(pyy:%.16g) pzz=$(pzz:%.16g) pxy=$(pxy:%.16g) pxz=$(pxz:%.16g) pyz=$(pyz:%.16g)"
"""
    )
    environment = os.environ.copy()
    environment.update(
        {
            "SYMMETRIX_JIT_POLICY": "none",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_PROC_BIND": "false",
        }
    )
    kokkos_arguments = ["-k", "on", "t", "1"]
    kokkos_gpus = os.environ.get("SYMMETRIX_LAMMPS_KOKKOS_GPUS")
    if kokkos_gpus:
        kokkos_arguments.extend(("g", kokkos_gpus))
    kokkos_arguments.extend(("-sf", "kk"))
    subprocess.run(
        [
            str(executable),
            "-screen",
            "none",
            "-log",
            str(log_path),
            *kokkos_arguments,
            "-in",
            str(input_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
        env=environment,
    )
    return _read_dump(initial_dump), _read_result(log_path, "RESULT")[0]


def test_mpi_processor_grids_cover_qualified_rank_counts():
    assert _PROCESSOR_GRIDS == {
        1: "1 1 1",
        2: "1 1 2",
        4: "1 2 2",
    }


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_kokkos_per_atom_energy_uses_neighbor_list_atom_indices(tmp_path, precision):
    executable = _required_path(
        "SYMMETRIX_LAMMPS_EXECUTABLE", "a Kokkos-enabled LAMMPS executable"
    )
    model = _required_path(
        "SYMMETRIX_LAMMPS_LEGACY_MODEL", "a legacy standard-MACE model"
    )
    rtol = 2.0e-5 if precision == "float32" else 1.0e-12
    atol = 2.0e-4 if precision == "float32" else 1.0e-10
    for domain_mode in (
        "no_domain_decomposition",
        "no_mpi_message_passing",
    ):
        per_atom, energy = _run_kokkos_hybrid_index_case(
            tmp_path, executable, model, precision, domain_mode
        )
        assert np.allclose(per_atom[per_atom[:, 1] == 1, 8], 0.0)
        assert np.all(np.abs(per_atom[per_atom[:, 1] == 2, 8]) > 1.0)
        assert np.sum(per_atom[:, 8]) == pytest.approx(energy, rel=rtol, abs=atol)


@pytest.mark.parametrize("pair_backend", ["cpu", "kokkos"])
def test_no_mpi_message_passing_ghost_neighbors_match_single_rank(
    tmp_path, pair_backend
):
    executable = _required_path(
        "SYMMETRIX_LAMMPS_EXECUTABLE", "an MPI-enabled LAMMPS executable"
    )
    model = _required_path(
        "SYMMETRIX_LAMMPS_COMPACT_MODEL", "a compact standard-MACE model"
    )
    distributed = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "generic",
        domain_mode="no_mpi_message_passing",
        pair_backend=pair_backend,
    )
    single_rank = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "generic",
        ranks=1,
        domain_mode="no_domain_decomposition",
        pair_backend=pair_backend,
    )

    assert re.search(r"Nghost:\s+[1-9]", distributed["log_text"])
    for key in ("initial_atoms", "migrated_atoms"):
        np.testing.assert_allclose(
            distributed[key][:, :9],
            single_rank[key][:, :9],
            rtol=1.0e-11,
            atol=1.0e-10,
        )
    for key in ("initial_global", "migrated_global"):
        np.testing.assert_allclose(
            distributed[key], single_rank[key], rtol=1.0e-11, atol=1.0e-9
        )


def test_capacity_event_parser(tmp_path):
    log_path = tmp_path / "capacity.log"
    log_path.write_text(
        "Symmetrix rank 1 graph capacity: active receivers=100, features=112, "
        "edges=2048; planned receivers=164, features=176, edges=2112; "
        "planned bytes=123456; policy=capacity-y-only; profile=capacity; "
        "plan=mh0-direct-capacity-y-only; fixed workspace=disabled; "
        "workspace receivers=0, edges=0, bytes=0, batches=0; "
        "pair edge geometry bytes=67584, feature position bytes=0; "
        "graph replacements=1, "
        "graph updates=2, geometry allocations=1, result allocations=1, "
        "M0 replacements=1, M0 adjoint allocations=0, M0 alias detaches=0, "
        "M0 alias active=1, communicated H1 allocations=1\n"
    )
    assert _read_capacity_records(log_path.read_text()) == [
        {
            "rank": 1,
            "active_receivers": 100,
            "active_features": 112,
            "active_edges": 2048,
            "planned_receivers": 164,
            "planned_features": 176,
            "planned_edges": 2112,
            "planned_bytes": 123456,
            "policy": "capacity-y-only",
            "profile": "capacity",
            "plan": "mh0-direct-capacity-y-only",
            "fixed_workspace": "disabled",
            "workspace_receivers": 0,
            "workspace_edges": 0,
            "workspace_bytes": 0,
            "workspace_batches": 0,
            "pair_geometry_bytes": 67584,
            "feature_position_bytes": 0,
            "graph_replacements": 1,
            "graph_updates": 2,
            "geometry_allocations": 1,
            "result_allocations": 1,
            "m0_replacements": 1,
            "m0_adjoint_allocations": 0,
            "m0_alias_detaches": 0,
            "m0_alias_active": 1,
            "h1_allocations": 1,
        }
    ]


def test_timing_record_parser_preserves_public_schema():
    values = [12.0, 10.0, 2.0, 16.6666666666667, 1.25, 0.75]
    values.extend([10.0, 15.0, 20.0, 1.5, 1.0, 1.0, 1.0, 1.0])
    fields = " ".join(
        f"{name}={value:.17g}" for name, value in zip(_TIMING_FIELDS, values)
    )
    record = _read_timing_record(
        f"unrelated output\nSYMMETRIX_TIMING scalar={values[3]:.17g} {fields}\n"
    )

    assert record["scalar"] == pytest.approx(100.0 / 6.0)
    assert record["critical_pair_us_per_atom_eval"] == 12.0
    assert record["critical_noncommunication_us_per_atom_eval"] == 10.0
    assert record["critical_comm_us_per_atom_eval"] == 2.0
    assert record["critical_forward_us_per_atom_eval"] == 1.25
    assert record["critical_reverse_us_per_atom_eval"] == 0.75
    assert record["rank_comm_percent_min"] == 10.0
    assert record["rank_comm_percent_average"] == 15.0
    assert record["rank_comm_percent_max"] == 20.0
    assert record["comm_imbalance_max_over_mean"] == 1.5
    assert record["forward_calls_per_eval_min"] == 1.0
    assert record["forward_calls_per_eval_max"] == 1.0
    assert record["reverse_calls_per_eval_min"] == 1.0
    assert record["reverse_calls_per_eval_max"] == 1.0


def test_timing_collective_is_deferred_to_compute_query():
    source_root = Path(__file__).resolve().parents[1]
    compute_source = (source_root / "compute_symmetrix_timing.cpp").read_text()
    assert compute_source.count("MPI_Allgather(") == 1
    assert "if (last_reduced_step == update->ntimestep) return;" in compute_source
    for name in ("pair_symmetrix_mace.cpp", "pair_symmetrix_mace_kokkos.cpp"):
        assert "MPI_Allgather(" not in (source_root / name).read_text()


def test_timing_compute_reference_counts_pair_timing_opt_in():
    source_root = Path(__file__).resolve().parents[1]
    compute_source = (source_root / "compute_symmetrix_timing.cpp").read_text()
    assert "*timing_enabled += 1.0;" in compute_source
    assert "*timing_enabled -= 1.0;" in compute_source
    assert "if (force && timing_enabled)" in compute_source
    for name in ("pair_symmetrix_mace.cpp", "pair_symmetrix_mace_kokkos.cpp"):
        pair_source = (source_root / name).read_text()
        assert pair_source.count("execution_timing_enabled = 0.0;") == 1


def test_symmetrix_timing_survives_partial_uncompute_and_pair_replacement(tmp_path):
    executable = _required_path(
        "SYMMETRIX_LAMMPS_EXECUTABLE", "an MPI-enabled LAMMPS executable"
    )
    model = _required_path(
        "SYMMETRIX_LAMMPS_COMPACT_MODEL", "a compact standard-MACE model"
    )
    data_path = tmp_path / "timing-uncompute.data"
    input_path = tmp_path / "in.timing-uncompute"
    log_path = tmp_path / "timing-uncompute.log"
    data_path.write_text(_physical_aln_fixture(False))
    input_path.write_text(
        f"""
units           metal
atom_style      atomic
atom_modify     map yes sort 1 0.0
boundary        p p p
read_data       {data_path}
pair_style      symmetrix/mace no_domain_decomposition
pair_coeff      * * {model} Al N
compute         timing_a all symmetrix/timing
compute         timing_b all symmetrix/timing
thermo          1
thermo_style    custom step c_timing_b
thermo_modify   colname auto line one
run             1 post no
uncompute       timing_a
run             1 pre no post no
pair_style      symmetrix/mace no_domain_decomposition
pair_coeff      * * {model} Al N
run             1 post no
"""
    )
    environment = os.environ.copy()
    environment.update(
        {
            "SYMMETRIX_JIT_POLICY": "none",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    subprocess.run(
        [
            executable,
            "-screen",
            "none",
            "-log",
            log_path,
            "-in",
            input_path,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
        env=environment,
    )
    assert log_path.read_text().count("SxH1CommPct") == 3


@pytest.mark.parametrize("pair_backend", ["cpu", "kokkos"])
def test_symmetrix_timing_reports_reduced_mpi_communication(tmp_path, pair_backend):
    executable = _required_path(
        "SYMMETRIX_LAMMPS_EXECUTABLE",
        "a Kokkos and MPI-enabled LAMMPS executable",
    )
    model = _required_path(
        "SYMMETRIX_LAMMPS_COMPACT_MODEL", "a compact standard-MACE model"
    )
    record = _run_timing_case(
        tmp_path,
        executable,
        model,
        pair_backend,
        ranks=2,
        domain_mode="mpi_message_passing",
    )

    _assert_timing_common(record, ranks=2)
    assert record["critical_pair_us_per_atom_eval"] > 0.0
    assert record["critical_noncommunication_us_per_atom_eval"] > 0.0
    assert record["critical_comm_us_per_atom_eval"] > 0.0
    assert record["critical_forward_us_per_atom_eval"] > 0.0
    assert record["critical_reverse_us_per_atom_eval"] > 0.0
    assert record["scalar"] > 0.0
    assert record["rank_comm_percent_max"] > 0.0
    assert 1.0 <= record["comm_imbalance_max_over_mean"] <= 2.0
    for direction in ("forward", "reverse"):
        assert record[f"{direction}_calls_per_eval_min"] == pytest.approx(1.0)
        assert record[f"{direction}_calls_per_eval_max"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("ranks", "domain_mode"),
    (
        pytest.param(1, "no_domain_decomposition", id="single-rank"),
        pytest.param(2, "no_mpi_message_passing", id="two-rank-no-pair-comm"),
    ),
)
def test_symmetrix_timing_zero_communication_controls(tmp_path, ranks, domain_mode):
    executable = _required_path(
        "SYMMETRIX_LAMMPS_EXECUTABLE", "an MPI-enabled LAMMPS executable"
    )
    model = _required_path(
        "SYMMETRIX_LAMMPS_COMPACT_MODEL", "a compact standard-MACE model"
    )
    record = _run_timing_case(
        tmp_path,
        executable,
        model,
        "cpu",
        ranks=ranks,
        domain_mode=domain_mode,
    )

    _assert_timing_common(record, ranks=ranks)
    assert record["critical_pair_us_per_atom_eval"] > 0.0
    assert record["critical_noncommunication_us_per_atom_eval"] == pytest.approx(
        record["critical_pair_us_per_atom_eval"]
    )
    for field in (
        "scalar",
        "critical_comm_us_per_atom_eval",
        "critical_comm_percent",
        "critical_forward_us_per_atom_eval",
        "critical_reverse_us_per_atom_eval",
        "rank_comm_percent_min",
        "rank_comm_percent_average",
        "rank_comm_percent_max",
        "comm_imbalance_max_over_mean",
        "forward_calls_per_eval_min",
        "forward_calls_per_eval_max",
        "reverse_calls_per_eval_min",
        "reverse_calls_per_eval_max",
    ):
        assert record[field] == 0.0


@pytest.mark.parametrize(
    ("triclinic", "pair_comm"),
    [
        pytest.param(False, "default", id="orthogonal-kokkos-comm"),
        pytest.param(True, "legacy", id="triclinic-legacy-comm"),
    ],
)
def test_direct_single_layer_mpi_matches_profiles_and_single_rank_after_migration(
    tmp_path, triclinic, pair_comm
):
    executable = _required_path(
        "SYMMETRIX_LAMMPS_EXECUTABLE", "a Kokkos and MPI-enabled LAMMPS executable"
    )
    model = _required_path(
        "SYMMETRIX_LAMMPS_SINGLE_LAYER_MODEL",
        "a compact single-layer standard-MACE model",
    )
    speed = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "direct",
        triclinic=triclinic,
        profile="speed",
        pair_comm=pair_comm,
    )
    capacity = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "direct",
        triclinic=triclinic,
        profile="capacity",
        pair_comm=pair_comm,
    )
    default_capacity = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "direct",
        triclinic=triclinic,
        profile=None,
        pair_comm=pair_comm,
    )
    legacy_capacity = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "direct",
        triclinic=triclinic,
        profile=None,
        low_memory=True,
        pair_comm=pair_comm,
    )
    legacy_speed = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "direct",
        triclinic=triclinic,
        profile=None,
        low_memory=False,
        pair_comm=pair_comm,
    )
    single_rank = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "direct",
        triclinic=triclinic,
        ranks=1,
        domain_mode="no_domain_decomposition",
        profile="speed",
    )

    for result, profile in (
        (speed, "speed"),
        (capacity, "capacity"),
        (default_capacity, "capacity"),
        (legacy_capacity, "capacity"),
        (legacy_speed, "speed"),
    ):
        records = result["capacity_records"]
        assert records
        assert {record["profile"] for record in records} == {profile}
        assert {record["fixed_workspace"] for record in records} == {"disabled"}
        assert {record["h1_allocations"] for record in records} == {0}
        assert all(
            record["active_features"] >= record["active_receivers"]
            for record in records
        )

    for result in (legacy_capacity, legacy_speed):
        assert "low_memory is deprecated" in result["log_text"]

    for key in ("initial_atoms", "migrated_atoms"):
        np.testing.assert_allclose(
            capacity[key], speed[key], rtol=1.0e-11, atol=1.0e-10
        )
        np.testing.assert_allclose(
            default_capacity[key], capacity[key], rtol=1.0e-11, atol=1.0e-10
        )
        np.testing.assert_allclose(
            legacy_capacity[key], capacity[key], rtol=1.0e-11, atol=1.0e-10
        )
        np.testing.assert_allclose(
            legacy_speed[key], speed[key], rtol=1.0e-11, atol=1.0e-10
        )
        np.testing.assert_allclose(
            speed[key][:, :9], single_rank[key][:, :9], rtol=1.0e-11, atol=1.0e-10
        )
    for key in ("initial_global", "migrated_global"):
        np.testing.assert_allclose(capacity[key], speed[key], rtol=1.0e-11, atol=1.0e-9)
        np.testing.assert_allclose(
            default_capacity[key], capacity[key], rtol=1.0e-11, atol=1.0e-9
        )
        np.testing.assert_allclose(
            legacy_capacity[key], capacity[key], rtol=1.0e-11, atol=1.0e-9
        )
        np.testing.assert_allclose(
            legacy_speed[key], speed[key], rtol=1.0e-11, atol=1.0e-9
        )
        np.testing.assert_allclose(
            speed[key], single_rank[key], rtol=1.0e-11, atol=1.0e-9
        )


def test_direct_single_layer_fixed_workspace_mpi_matches_retained_after_migration(
    tmp_path,
):
    if not os.environ.get("SYMMETRIX_LAMMPS_KOKKOS_GPUS"):
        pytest.skip("Set SYMMETRIX_LAMMPS_KOKKOS_GPUS for CUDA MPI qualification.")
    executable = _required_path(
        "SYMMETRIX_LAMMPS_EXECUTABLE", "a CUDA and MPI-enabled LAMMPS executable"
    )
    model = _required_path(
        "SYMMETRIX_LAMMPS_SINGLE_LAYER_MODEL",
        "a compact single-layer standard-MACE model",
    )
    retained = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "direct",
        precision="float32",
        profile="capacity",
    )
    fixed_workspace = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "direct",
        precision="float32",
        profile="capacity",
        allow_fixed_workspace=True,
        migrated_run_commands="run 0\nrun 0",
        debug_execution_plan="mh0-single-layer-tiled-v1",
        debug_single_layer_workspace_receivers=4,
    )
    single_rank = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "direct",
        precision="float32",
        ranks=1,
        domain_mode="no_domain_decomposition",
        profile="capacity",
    )

    records = fixed_workspace["capacity_records"]
    assert records
    assert {record["plan"] for record in records} == {"mh0-single-layer-tiled-v1"}
    assert {record["fixed_workspace"] for record in records} == {"allowed"}
    assert {record["h1_allocations"] for record in records} == {0}
    assert all(0 < record["workspace_receivers"] <= 4 for record in records)
    assert all(record["workspace_edges"] > 0 for record in records)
    assert all(record["workspace_bytes"] > 0 for record in records)
    assert all(record["workspace_batches"] > 1 for record in records)
    assert {record["pair_geometry_bytes"] for record in records} == {0}
    assert all(
        record["feature_position_bytes"] >= 24 * record["active_features"]
        for record in records
    )

    for key in ("initial_atoms", "migrated_atoms"):
        np.testing.assert_allclose(
            fixed_workspace[key], retained[key], rtol=2.0e-4, atol=1.0e-4
        )
        np.testing.assert_allclose(
            fixed_workspace[key][:, :9],
            single_rank[key][:, :9],
            rtol=2.0e-4,
            atol=1.0e-4,
        )
    for key in ("initial_global", "migrated_global"):
        np.testing.assert_allclose(
            fixed_workspace[key], retained[key], rtol=2.0e-4, atol=0.1
        )
        np.testing.assert_allclose(
            fixed_workspace[key], single_rank[key], rtol=2.0e-4, atol=0.1
        )


@pytest.mark.parametrize("triclinic", (False, True), ids=("orthogonal", "triclinic"))
def test_direct_dual_layer_fixed_workspace_mpi_matches_retained_after_migration(
    tmp_path, triclinic
):
    if not os.environ.get("SYMMETRIX_LAMMPS_KOKKOS_GPUS"):
        pytest.skip("Set SYMMETRIX_LAMMPS_KOKKOS_GPUS for CUDA MPI qualification.")
    executable = _required_path(
        "SYMMETRIX_LAMMPS_EXECUTABLE", "a CUDA and MPI-enabled LAMMPS executable"
    )
    model = _required_path(
        "SYMMETRIX_LAMMPS_COMPACT_MODEL", "a compact standard-MACE model"
    )
    device_artifact = _required_path(
        "SYMMETRIX_LAMMPS_JIT_ARTIFACT",
        "a float32 Execution device artifact with tiled R1 support",
    )
    retained = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "direct",
        precision="float32",
        triclinic=triclinic,
        profile="capacity",
        jit_device_artifact=device_artifact,
    )
    fixed_workspace = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "direct",
        precision="float32",
        triclinic=triclinic,
        profile="capacity",
        allow_fixed_workspace=True,
        migrated_run_commands="run 1 post no\nrun 1",
        debug_execution_plan="mh0-dual-layer-tiled-v1",
        debug_dual_layer_workspace_receivers=4,
        jit_device_artifact=device_artifact,
        capture_timing=True,
    )
    single_rank = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "direct",
        precision="float32",
        triclinic=triclinic,
        ranks=1,
        domain_mode="no_domain_decomposition",
        profile="capacity",
        jit_device_artifact=device_artifact,
    )

    records = fixed_workspace["capacity_records"]
    assert records
    assert {record["plan"] for record in records} == {"mh0-dual-layer-tiled-v1"}
    assert {record["fixed_workspace"] for record in records} == {"allowed"}
    assert {record["h1_allocations"] for record in records} == {0}
    assert all(0 < record["workspace_receivers"] <= 4 for record in records)
    assert all(record["workspace_edges"] > 0 for record in records)
    assert all(record["workspace_bytes"] > 0 for record in records)
    assert all(record["workspace_batches"] > 1 for record in records)
    assert {record["pair_geometry_bytes"] for record in records} == {0}
    assert all(
        record["feature_position_bytes"] >= 24 * record["active_features"]
        for record in records
    )
    timing = fixed_workspace["timing"]
    assert timing["forward_calls_per_eval_min"] == 1.0
    assert timing["forward_calls_per_eval_max"] == 1.0
    assert timing["reverse_calls_per_eval_min"] == 1.0
    assert timing["reverse_calls_per_eval_max"] == 1.0

    for key in ("initial_atoms", "migrated_atoms"):
        np.testing.assert_allclose(
            fixed_workspace[key], retained[key], rtol=2.0e-4, atol=1.0e-4
        )
        np.testing.assert_allclose(
            fixed_workspace[key][:, :9],
            single_rank[key][:, :9],
            rtol=2.0e-4,
            atol=1.0e-4,
        )
    for key in ("initial_global", "migrated_global"):
        np.testing.assert_allclose(
            fixed_workspace[key], retained[key], rtol=2.0e-4, atol=0.1
        )
        np.testing.assert_allclose(
            fixed_workspace[key], single_rank[key], rtol=2.0e-4, atol=0.1
        )


@pytest.mark.parametrize(
    ("block_commands", "total_steps"),
    [
        pytest.param(
            "run 2 post no\nreset_timestep 0\nrun 10",
            12,
            id="two-run-blocks",
        ),
        pytest.param(
            "\n".join("run 1 post no" for _ in range(20)),
            20,
            id="twenty-short-blocks",
        ),
    ],
)
def test_direct_low_memory_repeated_run_blocks_match_monolithic(
    tmp_path, block_commands, total_steps
):
    executable = _required_path(
        "SYMMETRIX_LAMMPS_EXECUTABLE", "a Kokkos and MPI-enabled LAMMPS executable"
    )
    model = _required_path(
        "SYMMETRIX_LAMMPS_COMPACT_MODEL", "a compact standard-MACE model"
    )
    artifact_arguments = _host_artifact_arguments("float32")
    monolithic = _run_direct_dynamics_case(
        tmp_path,
        executable,
        model,
        f"monolithic-{total_steps}",
        f"run {total_steps}",
        **artifact_arguments,
    )
    blocked = _run_direct_dynamics_case(
        tmp_path,
        executable,
        model,
        f"blocked-{total_steps}",
        block_commands,
        **artifact_arguments,
    )
    _assert_direct_dynamics_close(blocked, monolithic)


@pytest.mark.parametrize(
    ("precision", "triclinic"),
    [
        pytest.param("float64", False, id="float64-orthogonal"),
        pytest.param("float32", False, id="float32-orthogonal"),
        pytest.param("float64", True, id="float64-triclinic"),
        pytest.param("float32", True, id="float32-triclinic"),
    ],
)
def test_direct_low_memory_mpi_matches_retained_after_migration(
    tmp_path, precision, triclinic
):
    executable = _required_path(
        "SYMMETRIX_LAMMPS_EXECUTABLE", "a Kokkos and MPI-enabled LAMMPS executable"
    )
    model = _required_path(
        "SYMMETRIX_LAMMPS_COMPACT_MODEL", "a compact standard-MACE model"
    )
    artifact_arguments = _host_artifact_arguments(precision)
    retained = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "direct",
        precision=precision,
        triclinic=triclinic,
        **artifact_arguments,
    )
    low_memory = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "direct",
        precision=precision,
        triclinic=triclinic,
        profile="capacity",
        **artifact_arguments,
    )
    single_rank = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "direct",
        precision=precision,
        triclinic=triclinic,
        ranks=1,
        domain_mode="no_domain_decomposition",
        profile="capacity",
        **artifact_arguments,
    )

    rtol = 2.0e-4 if precision == "float32" else 1.0e-11
    atom_atol = 1.0e-4 if precision == "float32" else 1.0e-10
    global_atol = 1.0e-1 if precision == "float32" else 1.0e-9
    for key in ("initial_atoms", "migrated_atoms"):
        np.testing.assert_allclose(
            low_memory[key], retained[key], rtol=rtol, atol=atom_atol
        )
        np.testing.assert_allclose(
            low_memory[key][:, :9],
            single_rank[key][:, :9],
            rtol=rtol,
            atol=atom_atol,
        )
    for key in ("initial_global", "migrated_global"):
        np.testing.assert_allclose(
            low_memory[key], retained[key], rtol=rtol, atol=global_atol
        )
        np.testing.assert_allclose(
            low_memory[key], single_rank[key], rtol=rtol, atol=global_atol
        )


@pytest.mark.parametrize(
    ("precision", "triclinic"),
    [
        pytest.param("float64", False, id="float64-orthogonal"),
        pytest.param("float32", False, id="float32-orthogonal"),
        pytest.param("float64", True, id="float64-triclinic"),
        pytest.param("float32", True, id="float32-triclinic"),
    ],
)
def test_generic_mpi_matches_non_compiled_after_migration(
    tmp_path, precision, triclinic
):
    executable = _required_path(
        "SYMMETRIX_LAMMPS_EXECUTABLE", "a Kokkos and MPI-enabled LAMMPS executable"
    )
    model = _required_path(
        "SYMMETRIX_LAMMPS_COMPACT_MODEL", "a compact standard-MACE model"
    )
    reference = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "non-compiled",
        precision=precision,
        triclinic=triclinic,
    )
    generic = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "generic",
        precision=precision,
        triclinic=triclinic,
    )
    single_rank = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "generic",
        precision=precision,
        triclinic=triclinic,
        ranks=1,
        domain_mode="no_domain_decomposition",
    )

    rtol = 2.0e-4 if precision == "float32" else 1.0e-11
    atom_atol = 1.0e-4 if precision == "float32" else 1.0e-10
    global_atol = 1.0e-1 if precision == "float32" else 1.0e-9
    for key in ("initial_atoms", "migrated_atoms"):
        np.testing.assert_allclose(
            generic[key], reference[key], rtol=rtol, atol=atom_atol
        )
        assert np.max(np.abs(generic[key][:, 5:8] - reference[key][:, 5:8])) < atom_atol
    for key in ("initial_global", "migrated_global"):
        np.testing.assert_allclose(
            generic[key], reference[key], rtol=rtol, atol=global_atol
        )
        if precision == "float32":
            np.testing.assert_allclose(
                generic[key][:1], single_rank[key][:1], rtol=1.0e-6, atol=1.0e-3
            )
            np.testing.assert_allclose(
                generic[key][1:], single_rank[key][1:], rtol=5.0e-4, atol=0.1
            )
        else:
            np.testing.assert_allclose(
                generic[key], single_rank[key], rtol=rtol, atol=global_atol
            )
    for key in ("initial_atoms", "migrated_atoms"):
        np.testing.assert_allclose(
            generic[key][:, :9],
            single_rank[key][:, :9],
            rtol=rtol,
            atol=atom_atol,
        )
        assert (
            np.max(np.abs(generic[key][:, 5:8] - single_rank[key][:, 5:8])) < atom_atol
        )


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_generic_mpi_legacy_pair_callbacks_match_kokkos_after_migration(
    tmp_path, precision
):
    executable = _required_path(
        "SYMMETRIX_LAMMPS_EXECUTABLE", "a Kokkos and MPI-enabled LAMMPS executable"
    )
    model = _required_path(
        "SYMMETRIX_LAMMPS_COMPACT_MODEL", "a compact standard-MACE model"
    )
    kokkos = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "generic",
        precision=precision,
    )
    legacy = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "generic",
        precision=precision,
        pair_comm="legacy",
    )

    rtol = 2.0e-4 if precision == "float32" else 1.0e-11
    atom_atol = 1.0e-4 if precision == "float32" else 1.0e-10
    global_atol = 1.0e-1 if precision == "float32" else 1.0e-9
    for key in ("initial_atoms", "migrated_atoms"):
        np.testing.assert_allclose(legacy[key], kokkos[key], rtol=rtol, atol=atom_atol)
    for key in ("initial_global", "migrated_global"):
        np.testing.assert_allclose(
            legacy[key], kokkos[key], rtol=rtol, atol=global_atol
        )


def test_generic_mpi_four_rank_migration_matches_single_rank(tmp_path):
    executable = _required_path(
        "SYMMETRIX_LAMMPS_EXECUTABLE", "a Kokkos and MPI-enabled LAMMPS executable"
    )
    model = _required_path(
        "SYMMETRIX_LAMMPS_COMPACT_MODEL", "a compact standard-MACE model"
    )
    four_rank = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "generic",
        ranks=4,
    )
    single_rank = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "generic",
        ranks=1,
        domain_mode="no_domain_decomposition",
    )

    for key in ("initial_atoms", "migrated_atoms"):
        np.testing.assert_allclose(
            four_rank[key][:, :9], single_rank[key][:, :9], rtol=1.0e-11, atol=1.0e-10
        )
    for key in ("initial_global", "migrated_global"):
        np.testing.assert_allclose(
            four_rank[key], single_rank[key], rtol=1.0e-11, atol=1.0e-9
        )


@pytest.mark.parametrize(
    ("precision", "triclinic"),
    [
        pytest.param("float64", False, id="float64-orthogonal"),
        pytest.param("float32", False, id="float32-orthogonal"),
        pytest.param("float64", True, id="float64-triclinic"),
        pytest.param("float32", True, id="float32-triclinic"),
    ],
)
def test_generic_macefield_mpi_matches_after_migration(tmp_path, precision, triclinic):
    executable = _required_path(
        "SYMMETRIX_LAMMPS_EXECUTABLE", "a Kokkos and MPI-enabled LAMMPS executable"
    )
    model = _required_path("SYMMETRIX_LAMMPS_FIELD_MODEL", "a compact MACEField model")
    field = (0.01, 0.02, -0.03)
    reference = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "non-compiled",
        field,
        precision,
        triclinic,
    )
    generic = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "generic",
        field,
        precision,
        triclinic,
    )
    single_rank = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "generic",
        field,
        precision,
        triclinic,
        ranks=1,
        domain_mode="no_domain_decomposition",
    )

    rtol = 2.0e-4 if precision == "float32" else 1.0e-11
    atom_atol = 1.0e-4 if precision == "float32" else 1.0e-10
    global_atol = 1.0e-1 if precision == "float32" else 1.0e-8
    for key in ("initial_atoms", "migrated_atoms"):
        np.testing.assert_allclose(
            generic[key], reference[key], rtol=rtol, atol=atom_atol
        )
        assert np.max(np.abs(generic[key][:, 5:8] - reference[key][:, 5:8])) < atom_atol
    for key in ("initial_global", "migrated_global"):
        np.testing.assert_allclose(
            generic[key], reference[key], rtol=rtol, atol=global_atol
        )
        np.testing.assert_allclose(
            generic[key], single_rank[key], rtol=rtol, atol=global_atol
        )
    for key in ("initial_atoms", "migrated_atoms"):
        np.testing.assert_allclose(
            generic[key][:, :9],
            single_rank[key][:, :9],
            rtol=rtol,
            atol=atom_atol,
        )
        assert (
            np.max(np.abs(generic[key][:, 5:8] - single_rank[key][:, 5:8])) < atom_atol
        )


@pytest.mark.parametrize(
    ("precision", "triclinic"),
    [
        pytest.param("float64", False, id="float64-orthogonal"),
        pytest.param("float32", False, id="float32-orthogonal"),
        pytest.param("float64", True, id="float64-triclinic"),
        pytest.param("float32", True, id="float32-triclinic"),
    ],
)
def test_direct_low_memory_macefield_mpi_matches_after_migration(
    tmp_path, precision, triclinic
):
    executable = _required_path(
        "SYMMETRIX_LAMMPS_EXECUTABLE", "a Kokkos and MPI-enabled LAMMPS executable"
    )
    model = _required_path("SYMMETRIX_LAMMPS_FIELD_MODEL", "a compact MACEField model")
    artifact_arguments = _host_artifact_arguments(precision)
    field = (0.01, 0.02, -0.03)
    reference = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "generic",
        field,
        precision,
        triclinic,
    )
    retained = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "direct",
        field,
        precision,
        triclinic,
        **artifact_arguments,
    )
    low_memory = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "direct",
        field,
        precision,
        triclinic,
        profile="capacity",
        **artifact_arguments,
    )
    assert "requested='direct' resolved='direct'" in low_memory["log_text"]
    capacity_records = low_memory["capacity_records"]
    assert len(capacity_records) >= 4
    records_by_rank = {
        rank: [record for record in capacity_records if record["rank"] == rank]
        for rank in range(2)
    }
    assert all(len(records) >= 2 for records in records_by_rank.values())
    for records in records_by_rank.values():
        first, last = records[0], records[-1]
        assert first["active_receivers"] <= first["planned_receivers"]
        assert first["active_features"] <= first["planned_features"]
        assert first["active_edges"] <= first["planned_edges"]
        assert last["active_receivers"] <= last["planned_receivers"]
        assert last["active_features"] <= last["planned_features"]
        assert last["active_edges"] <= last["planned_edges"]
        assert last["graph_updates"] >= first["graph_updates"]
        if last["policy"].startswith("capacity-"):
            # Ghost feature nodes can require a dedicated H1 adjoint allocation.
            assert last["m0_adjoint_allocations"] == 0
        assert last["h1_allocations"] >= first["h1_allocations"]
    single_rank = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "direct",
        field,
        precision,
        triclinic,
        ranks=1,
        domain_mode="no_domain_decomposition",
        profile="capacity",
        **artifact_arguments,
    )

    rtol = 2.0e-4 if precision == "float32" else 1.0e-11
    atom_atol = 1.0e-4 if precision == "float32" else 1.0e-10
    global_atol = 1.0e-1 if precision == "float32" else 1.0e-8
    for key in ("initial_atoms", "migrated_atoms"):
        np.testing.assert_allclose(
            retained[key], reference[key], rtol=rtol, atol=atom_atol
        )
        np.testing.assert_allclose(
            low_memory[key], retained[key], rtol=rtol, atol=atom_atol
        )
        np.testing.assert_allclose(
            low_memory[key][:, :9],
            single_rank[key][:, :9],
            rtol=rtol,
            atol=atom_atol,
        )
    for key in ("initial_global", "migrated_global"):
        np.testing.assert_allclose(
            retained[key], reference[key], rtol=rtol, atol=global_atol
        )
        np.testing.assert_allclose(
            low_memory[key], retained[key], rtol=rtol, atol=global_atol
        )
        np.testing.assert_allclose(
            low_memory[key], single_rank[key], rtol=rtol, atol=global_atol
        )


def test_direct_low_memory_macefield_repeated_runs_reuse_migration_capacity(
    tmp_path,
):
    executable = _required_path(
        "SYMMETRIX_LAMMPS_EXECUTABLE", "a Kokkos and MPI-enabled LAMMPS executable"
    )
    model = _required_path("SYMMETRIX_LAMMPS_FIELD_MODEL", "a compact MACEField model")
    artifact_arguments = _host_artifact_arguments("float32")
    field = (0.01, 0.02, -0.03)
    monolithic = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "direct",
        field,
        "float32",
        profile="capacity",
        migrated_run_commands="run 3",
        **artifact_arguments,
    )
    blocked = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "direct",
        field,
        "float32",
        profile="capacity",
        migrated_run_commands="run 0 post no\nrun 1 post no\nrun 2",
        **artifact_arguments,
    )

    for key in ("migrated_atoms", "migrated_global"):
        np.testing.assert_allclose(
            blocked[key], monolithic[key], rtol=2.0e-4, atol=1.0e-4
        )
    assert "requested='direct' resolved='direct'" in blocked["log_text"]
    records = blocked["capacity_records"]
    assert records
    assert all(
        record["active_receivers"] <= record["planned_receivers"]
        and record["active_features"] <= record["planned_features"]
        and record["active_edges"] <= record["planned_edges"]
        for record in records
    )
    capacity_records = [
        record for record in records if record["policy"].startswith("capacity-")
    ]
    assert all(record["m0_adjoint_allocations"] == 0 for record in capacity_records)
    records_by_rank = {
        rank: [record for record in records if record["rank"] == rank]
        for rank in {record["rank"] for record in records}
    }
    for rank_records in records_by_rank.values():
        assert (
            rank_records[-1]["graph_updates"] > rank_records[0]["graph_updates"]
            or rank_records[-1]["graph_replacements"]
            > rank_records[0]["graph_replacements"]
        )
        for previous, current in itertools.pairwise(rank_records):
            fits_previous_capacity = (
                current["active_receivers"] <= previous["planned_receivers"]
                and current["active_features"] <= previous["planned_features"]
                and current["active_edges"] <= previous["planned_edges"]
            )
            if fits_previous_capacity:
                assert current["graph_replacements"] == previous["graph_replacements"]
                assert (
                    current["geometry_allocations"] == previous["geometry_allocations"]
                )
                assert current["result_allocations"] == previous["result_allocations"]
                assert current["m0_replacements"] == previous["m0_replacements"]
                assert current["m0_alias_detaches"] == previous["m0_alias_detaches"]
                assert current["h1_allocations"] == previous["h1_allocations"]


def test_direct_low_memory_macefield_four_rank_matches_single_rank(tmp_path):
    executable = _required_path(
        "SYMMETRIX_LAMMPS_EXECUTABLE", "a Kokkos and MPI-enabled LAMMPS executable"
    )
    model = _required_path("SYMMETRIX_LAMMPS_FIELD_MODEL", "a compact MACEField model")
    artifact_arguments = _host_artifact_arguments("float32")
    field = (0.01, 0.02, -0.03)
    four_rank = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "direct",
        field,
        "float32",
        ranks=4,
        profile="capacity",
        **artifact_arguments,
    )
    single_rank = _run_lammps_case(
        tmp_path,
        executable,
        model,
        "direct",
        field,
        "float32",
        ranks=1,
        domain_mode="no_domain_decomposition",
        profile="capacity",
        **artifact_arguments,
    )

    for key in ("initial_atoms", "migrated_atoms"):
        np.testing.assert_allclose(
            four_rank[key][:, :9],
            single_rank[key][:, :9],
            rtol=2.0e-4,
            atol=1.0e-4,
        )
    for key in ("initial_global", "migrated_global"):
        np.testing.assert_allclose(
            four_rank[key], single_rank[key], rtol=2.0e-4, atol=1.0e-1
        )
