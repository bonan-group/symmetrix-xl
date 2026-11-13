"""MACE Polar in LAMMPS on the generated route, against the ASE calculator.

Both frontends run the same native evaluator. The tests name the backbone and
one product module per layer in ``pair_style``, require every layer's module
to have run, and compare energy, forces and stress with the ASE ``direct``
route for the same model, structure and precision.
"""

from __future__ import annotations

import os
import subprocess

import numpy as np
import pytest
from polar_jit_artifacts import prepare_polar_jit, skip_or_fail

#: Pressure in bar per eV/A^3 in LAMMPS metal units (``force->nktv2p``).
NKTV2P = 1.6021765e6

#: Largest difference from ASE allowed for each quantity, relative to its
#: largest reference magnitude. Both frontends run the same native evaluator,
#: so FP64 agrees to rounding.
TOLERANCE = {"float64": 1e-11, "float32": 1e-4}

#: Every pair lies well inside the synthetic model's 4 A cutoff, so no pair
#: sits in the neighbor skin shell.
SYNTHETIC_SPECIES = ("Si", "Si", "H")
SYNTHETIC_POSITIONS = [[3.0, 3.0, 3.0], [5.3, 3.0, 3.0], [3.4, 4.3, 3.2]]
WATER_SPECIES = ("O", "H", "H")
WATER_POSITIONS = [[3.0, 3.0, 3.0], [3.96, 3.0, 3.0], [2.76, 3.93, 3.0]]


@pytest.fixture(scope="module")
def polar_jit_cache(tmp_path_factory):
    """One private JIT cache for the module, as the cache root requires.

    Its path holds a space, as a deployment path may: every path reaches
    LAMMPS quoted.
    """

    cache = tmp_path_factory.mktemp("polar jit cache")
    cache.chmod(0o700)
    return cache


def _run(
    executable,
    model,
    dtype,
    species,
    positions,
    arguments,
    directory,
    *,
    device,
    error=None,
):
    """Evaluate once in LAMMPS; return energy, forces and stress, or check a stop.

    ``device`` launches Kokkos on one GPU, as a CUDA or HIP build runs the one
    rank; otherwise on one host thread.
    """

    from symmetrix.mace_nonlinear_jit_artifact import format_lammps_arguments

    directory.mkdir()
    elements = list(dict.fromkeys(species))
    style = (
        "symmetrix/mace/polar/kk"
        if dtype == "float64"
        else "symmetrix/mace/polar/float32/kk"
    )
    atoms = "\n".join(
        f"create_atoms {elements.index(element) + 1} single {x} {y} {z} units box"
        for element, (x, y, z) in zip(species, positions)
    )
    forces = " ".join(
        f"$(f{axis}[{atom}]:%.17g)"
        for atom in range(1, len(species) + 1)
        for axis in "xyz"
    )
    pressure = " ".join(f"$(c_pv[{index}]:%.17g)" for index in range(1, 7))
    settings = format_lammps_arguments(arguments)
    coefficients = format_lammps_arguments(["*", "*", model, *elements])
    deck = directory / "in.polar"
    deck.write_text(
        f"""
units metal
atom_style atomic/kk
atom_modify map yes
boundary p p p
region box block 0 12 0 12 0 12 units box
create_box {len(elements)} box
{atoms}
mass * 1.0
neighbor 0.5 bin
pair_style {style} polar_electrostatics direct_reciprocal {settings}
pair_coeff {coefficients}
run_style verlet/kk
compute pv all pressure NULL virial
thermo_style custom step pe c_pv[1] c_pv[2] c_pv[3] c_pv[4] c_pv[5] c_pv[6]
run 0
print "POLAR_JIT_ENERGY $(pe:%.17g)"
print "POLAR_JIT_FORCES {forces}"
print "POLAR_JIT_PRESSURE {pressure}"
"""
    )
    completed = subprocess.run(
        [
            executable,
            "-k",
            "on",
            *(("g", "1") if device else ("t", "1")),
            "-log",
            "none",
            "-in",
            str(deck),
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=600,
    )
    output = completed.stdout + completed.stderr
    if error is not None:
        assert completed.returncode != 0, output[-2000:]
        assert error in output, output[-2000:]
        assert "POLAR_JIT_ENERGY" not in output, output[-2000:]
        return None
    assert completed.returncode == 0, output[-2000:]
    layers = arguments.count("jit_product_artifact")
    executed = (
        f"MACE Polar JIT executed {layers}/{layers} product layers; "
        "backbone fallbacks 0"
    )
    assert executed in output, output[-2000:]
    values = {
        line.split()[0]: [float(value) for value in line.split()[1:]]
        for line in output.splitlines()
        if line.startswith("POLAR_JIT_")
    }
    xx, yy, zz, xy, xz, yz = values["POLAR_JIT_PRESSURE"]
    return (
        values["POLAR_JIT_ENERGY"][0],
        np.array(values["POLAR_JIT_FORCES"]).reshape(-1, 3),
        # The virial pressure is minus the stress; ASE orders xx yy zz yz xz xy.
        -np.array([xx, yy, zz, yz, xz, xy]) / NKTV2P,
    )


def _assert_matches_ase(result, reference, dtype):
    energy, forces, stress = result
    atoms = len(reference["forces"])
    for actual, expected in (
        (energy / atoms, reference["energy"] / atoms),
        (forces, reference["forces"]),
        (stress, reference["stress"]),
    ):
        expected = np.asarray(expected)
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=0.0,
            atol=TOLERANCE[dtype] * np.abs(expected).max(),
        )


@pytest.mark.parametrize(
    "dtype,layers",
    [
        pytest.param("float64", 1, id="fp64-s"),
        pytest.param("float32", 1, id="fp32-s"),
        pytest.param("float64", 2, id="fp64-l2"),
        pytest.param("float32", 2, id="fp32-l2"),
    ],
)
def test_polar_lammps_jit_matches_ase(tmp_path, polar_jit_cache, dtype, layers):
    executable = os.environ.get("SYMMETRIX_LAMMPS_EXECUTABLE")
    if not executable:
        skip_or_fail("Set SYMMETRIX_LAMMPS_EXECUTABLE for the Polar LAMMPS tests.")
    model = tmp_path / "polar.json"
    prepared = prepare_polar_jit(
        model,
        dtype,
        polar_jit_cache,
        species=SYNTHETIC_SPECIES,
        positions=SYNTHETIC_POSITIONS,
        layers=layers,
    )
    arguments = prepared["arguments"]
    assert arguments.count("jit_product_artifact") == layers
    device = arguments[0] == "jit_device_artifact"

    def run(name, selection, error=None):
        return _run(
            executable,
            model,
            dtype,
            SYNTHETIC_SPECIES,
            SYNTHETIC_POSITIONS,
            selection,
            tmp_path / name,
            device=device,
            error=error,
        )

    _assert_matches_ase(run("complete", arguments), prepared, dtype)
    if (dtype, layers) != ("float64", 2):
        return
    # An artifact without its complete product set stops while the model
    # loads; nothing falls back to the portable backbone.
    backbone, products = arguments[:2], arguments[2:]
    stop = "requires one jit_product_artifact per interaction layer"
    run("backbone-only", backbone, stop)
    run("first-layer-only", backbone + products[:4], stop)
    run(
        "second-layer-only",
        backbone + products[4:],
        "layers must be consecutive from 0",
    )
    if not device:
        # A CPU build has no accelerator backend for a device artifact to name.
        run(
            "device-on-host",
            ["jit_device_artifact", backbone[1], *products],
            "jit_device_artifact requires a CUDA or HIP build",
        )


@pytest.mark.parametrize("dtype", ("float64", "float32"), ids=("fp64", "fp32"))
def test_polar_lammps_jit_matches_ase_with_qualified_fixture(
    tmp_path, polar_jit_cache, dtype
):
    executable = os.environ.get("SYMMETRIX_LAMMPS_POLAR_EXECUTABLE")
    model = os.environ.get("SYMMETRIX_LAMMPS_POLAR_MODEL")
    missing = [
        name
        for name, value in (
            ("SYMMETRIX_LAMMPS_POLAR_EXECUTABLE", executable),
            ("SYMMETRIX_LAMMPS_POLAR_MODEL", model),
        )
        if not value
    ]
    if missing:
        pytest.skip("missing qualified external fixture: " + ", ".join(missing))

    prepared = prepare_polar_jit(
        model,
        dtype,
        polar_jit_cache,
        species=WATER_SPECIES,
        positions=WATER_POSITIONS,
    )
    result = _run(
        executable,
        model,
        dtype,
        WATER_SPECIES,
        WATER_POSITIONS,
        prepared["arguments"],
        tmp_path / "qualified",
        device=prepared["arguments"][0] == "jit_device_artifact",
    )
    _assert_matches_ase(result, prepared, dtype)
