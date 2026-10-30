"""The PolarMACE delta chain's state parity against the frozen oracle.

The delta layer is the family's node-local execution surface: the per-layer
spin-source multipole heads, the multi-layer feature mixer, the Fukui source
map, the field-dependent charge-update chain, and the post-loop
local-electron-energy readout, driven by the unrolled non-self-consistent
orchestration.  ``benchmarks/polar_chain_state_oracle_runner.py`` froze its
inputs and its state from the pinned torch stack, so this module gates the
native layer against those records without importing torch for the *oracle*:
the model itself still comes from the checkpoint (the repository's other
checkpoint-dependent polar suites do the same, and every one of them skips
cleanly with the missing prerequisite named).

The gate is the chain state, not the end-to-end energy.  End-to-end parity,
the channel-lane behaviour, the stress composition, and the dipole flip are the
next stage's.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import symmetrix
from polar_oracle_checkout import required_checkout

if not symmetrix._kokkos_is_initialized():
    symmetrix._init_kokkos()

DATA_DIR = Path(__file__).resolve().parent / "data" / "polar"
MANIFEST_PATH = DATA_DIR / "chain_manifest.json"

#: The parity tolerances, measured against the frozen records (fp64) and the
#: records' own dtype spread (fp32).  The fp64 floor is the *oracle's* own:
#: the released checkpoints carry fp32-derived GTO Fourier tables (the design
#: note's fp32-weight finding), so the oracle's own fp64 chain state differs
#: from an exact fp64 evaluation of the same formulas by ~3e-8 relative -- the
#: measured floor of this gate, well above the kernels' 1e-12 op-level parity.
CHAIN_TOLERANCE = {
    # absolute, on the density and its reductions (their scale is ~1).  The
    # measured floor is 1.8e-8 (fp64) and 1.8e-5 (fp32).
    "density_absolute": {"float64": 1.0e-7, "float32": 1.0e-3},
    # absolute, on the chain's feature buffers (the mixed features and the
    # per-iteration potential features).  Their fp64 floor is higher than the
    # density's -- 2.7e-7, the oracle's fp32-derived Fourier tables showing
    # through the feature composition -- and fp32's is 4.6e-5; the gates sit an
    # order of magnitude above both.
    "feature_absolute": {"float64": 1.0e-6, "float32": 1.0e-3},
    # relative, on the graph-summed local-electron energy (measured 3.5e-9 fp64,
    # 1.3e-6 fp32).
    "energy_relative": {"float64": 1.0e-7, "float32": 1.0e-3},
}


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def _record_paths() -> list[tuple[dict, Path]]:
    manifest = _load_manifest()
    return [(record, DATA_DIR / record["name"]) for record in manifest["records"]]


def _checkpoint_path(size: str) -> Path:
    """The MACE-POLAR-1 checkpoint, from the environment or the local cache."""

    configured = os.environ.get(f"SYMMETRIX_POLAR_MODEL_{size}")
    if configured:
        path = Path(configured)
    else:
        path = Path("~/.cache/symmetrix/test-models").expanduser() / (
            f"MACE-POLAR-1-{size}.model"
        )
    if not path.is_file():
        pytest.skip(
            f"the chain-state parity gate needs the MACE-POLAR-1-{size} "
            f"checkpoint (set SYMMETRIX_POLAR_MODEL_{size}, or cache it at "
            f"{path}); the fixture records themselves are in-tree."
        )
    return path


#: The pinned oracle checkouts the checkpoint's own module state refers to.
#: The chain-state gate never *evaluates* the oracle, but the released
#: checkpoints' pickles carry the graph_longrange classes, so unpickling needs
#: the checkout importable; it is added read-only, exactly as the end-to-end
#: fixture suite does.
_ORACLE_CHECKOUT_ENVIRONMENTS = (
    "SYMMETRIX_POLAR_ORACLE_MACE",
    "SYMMETRIX_POLAR_ORACLE_GRAPH_LONGRANGE",
)


def _prepare_oracle_imports() -> None:
    import sys

    for environment in _ORACLE_CHECKOUT_ENVIRONMENTS:
        checkout = required_checkout(environment)
        if str(checkout) not in sys.path:
            sys.path.insert(0, str(checkout))


def _model_json(tmp_path: Path, size: str) -> Path:
    """Extract the checkpoint to the JSON the family's evaluator reads."""

    path = _checkpoint_path(size)
    _prepare_oracle_imports()
    try:
        import torch
        from symmetrix.extract_mace_polar import extract_mace_polar_data
    except ImportError as error:  # pragma: no cover - environment-dependent
        pytest.skip(f"extracting the checkpoint requires torch/e3nn/checkout: {error}")
    torch.set_default_dtype(torch.float64)
    model = torch.load(str(path), map_location="cpu", weights_only=False)
    data = extract_mace_polar_data(model.to(torch.float64))
    target = tmp_path / f"MACE-POLAR-1-{size}.json"
    target.write_text(json.dumps(data))
    return target


@pytest.fixture(scope="module")
def evaluators(tmp_path_factory):
    """One polar evaluator per precision, parsed from the M checkpoint."""

    directory = tmp_path_factory.mktemp("polar-delta-chain")
    path = _model_json(directory, "M")
    evaluators = {
        dtype: (
            symmetrix.MACEPolarKokkos(str(path))
            if dtype == "float64"
            else symmetrix.MACEPolarKokkosFloat(str(path))
        )
        for dtype in ("float64", "float32")
    }
    yield evaluators
    # Kokkos refuses to finalize while evaluators are alive, so the fixture
    # releases them before the session ends (the other polar suites do the
    # same for their wired evaluators).
    evaluators.clear()


# ---------------------------------------------------------------------------
# The frozen records
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_released_m_checkpoint_selects_every_polar_e3linear(evaluators, dtype):
    """An unlabelled or shape-mismatched real linear must be visible.

    The released M architecture contains 23 E3Linear objects.  This inventory
    assertion is deliberately against the parsed evaluator, not a synthetic
    list, so adding a new polar linear without validated specialization
    metadata turns the direct-mode zero-fallback gate red.
    """

    diagnostics = evaluators[dtype].polar_e3linear_diagnostics()
    assert diagnostics["linear_count"] == 23
    assert diagnostics["selected_count"] == 23
    assert diagnostics["fallback_count"] == 0
    assert diagnostics["roles"]["feature_mixer"] == 2
    assert diagnostics["roles"]["update_readout_final"] == 2


def test_polar_e3linear_microbenchmark_compares_the_same_isolated_work(evaluators):
    """The timing surface must fence and compare generic/specialized work."""

    before = evaluators["float32"].polar_e3linear_diagnostics()
    record = evaluators["float32"].polar_e3linear_microbenchmark(
        nodes=4, warmups=0, repeats=1
    )
    after = evaluators["float32"].polar_e3linear_diagnostics()
    assert record["nodes"] == 4
    assert record["warmups"] == 0
    assert record["repeats"] == 1
    assert record["linear_count"] == 23
    assert record["forward_launches_per_repeat"] == 37
    assert record["reverse_launches_per_repeat"] == 23
    assert record["generic_ms"] > 0.0
    assert record["specialized_ms"] > 0.0
    # The generic half of the interleaved comparison does not affect selected
    # telemetry. With no warmup, the deltas therefore model exactly one
    # production request, including the 14 update-block forward replays.
    assert after["forward_count"] - before["forward_count"] == 37
    assert after["reverse_count"] - before["reverse_count"] == 23
    assert after["fallback_count"] - before["fallback_count"] == 0


def test_the_chain_state_records_are_complete_and_coherent():
    """The in-tree records carry the chain's inputs and its state.

    The inventory is the gate's own contract: a record that lost its inputs,
    its reductions, or its energy could not be driven, so the shapes and the
    key set are pinned here rather than discovered by the parity run.
    """

    manifest = _load_manifest()
    assert manifest["schema"] == "symmetrix.polar.oracle-chain-state"
    assert manifest["version"] == 1
    assert manifest["records"], "the fixture set is empty"
    oracle = manifest["oracle"]
    assert oracle["mace_sha"].startswith("22f08097")
    assert oracle["graph_longrange_version"] == "0.4.4"
    for record, path in _record_paths():
        assert path.is_file(), path
        with np.load(path) as arrays:
            keys = set(arrays.files)
            assert keys == {
                "node_features",
                "node_attrs",
                "positions",
                "cell",
                "graph_pbc",
                "node_graph_index",
                "total_charge",
                "total_spin",
                "external_potential",
                # The chain's own intermediates: the mixed features the update
                # chain consumes and every unrolled iteration's potential
                # features.
                "features_mixed",
                "potential_features",
                "spin_charge_density",
                "charge_density",
                "spin_density",
                "local_electron_energy",
            }
            assert arrays["features_mixed"].shape[0] == arrays["positions"].shape[0]
            assert (
                arrays["potential_features"].shape[0] == record["iterations"]
            ), record["name"]
            nodes = arrays["positions"].shape[0]
            assert arrays["node_features"].shape[0] == nodes
            assert arrays["node_attrs"].shape[0] == nodes
            assert arrays["spin_charge_density"].shape == (nodes, 8)
            assert arrays["charge_density"].shape == (nodes, 4)
            assert arrays["spin_density"].shape == (nodes, 4)
            assert arrays["local_electron_energy"].shape[0] == 1
            assert arrays["graph_pbc"].shape == (1, 3)
            assert arrays["external_potential"].shape == (1, 4)
            assert record["nodes"] == nodes


def test_the_records_carry_the_frozen_channels_and_dtypes():
    """Both dtypes exist for every geometry/channel record, and the manifest
    agrees with the files it names."""

    manifest = _load_manifest()
    by_key = {
        (record["size"], record["geometry"], record["channel"], record["dtype"]): record
        for record in manifest["records"]
    }
    for size in ("M",):
        for geometry in ("mol_h2o", "slab_w2", "bulk_w4"):
            for channel in ("singlet", "field"):
                for dtype in ("float64", "float32"):
                    assert (size, geometry, channel, dtype) in by_key, (
                        size,
                        geometry,
                        channel,
                        dtype,
                    )
    for record in manifest["records"]:
        path = DATA_DIR / record["name"]
        assert path.stat().st_size == record["bytes"]


# ---------------------------------------------------------------------------
# The parity gate
# ---------------------------------------------------------------------------


def _drive(evaluator, arrays: dict) -> dict:
    """Drive the chain on one record's inputs."""

    return evaluator.polar_delta_chain(
        node_features=np.ascontiguousarray(arrays["node_features"], dtype=np.float64),
        node_attrs=np.ascontiguousarray(arrays["node_attrs"], dtype=np.float64),
        positions=np.ascontiguousarray(arrays["positions"], dtype=np.float64),
        cell=np.ascontiguousarray(arrays["cell"], dtype=np.float64),
        graph_pbc=np.ascontiguousarray(arrays["graph_pbc"], dtype=np.int32),
        node_graph_index=np.ascontiguousarray(
            arrays["node_graph_index"], dtype=np.int32
        ),
        total_charge=[float(value) for value in arrays["total_charge"]],
        total_spin=[float(value) for value in arrays["total_spin"]],
        external_potential=np.ascontiguousarray(
            arrays["external_potential"], dtype=np.float64
        ),
    )


def test_released_m_checkpoint_composes_the_three_potential_embedding_forwards(
    evaluators,
):
    """The admitted M embedding is one device path, not three E3Linear calls.

    A complete M chain forward otherwise dispatches all 23 inventoried
    E3Linears once.  The two field updates each own the potential, node, and
    charge embedding linears, so the composed route leaves exactly 17
    individual E3Linear forwards.  Falling back to the reference schedule
    makes this delta 23.
    """

    evaluator = evaluators["float64"]
    _record, path = next(
        (record, path)
        for record, path in _record_paths()
        if record["dtype"] == "float64"
    )
    with np.load(path) as arrays:
        inputs = {key: arrays[key] for key in arrays.files}
    before = evaluator.polar_e3linear_diagnostics()
    _drive(evaluator, inputs)
    after = evaluator.polar_e3linear_diagnostics()

    assert after["forward_count"] - before["forward_count"] == 17
    assert after["fallback_count"] - before["fallback_count"] == 0


@pytest.mark.parametrize("record_index", range(len(RECORDS := _record_paths())))
def test_the_chain_state_matches_the_frozen_oracle(evaluators, record_index):
    """The driven chain reproduces the oracle's recorded chain state.

    The compared quantities are the ones the oracle records: the equilibrated
    spin-resolved density (the chain's whole output, since both unrolled
    iterations feed it), its spin-summed and spin-difference reductions, the
    graph-summed local-electron energy the readout produces, and the chain's
    own intermediates -- the mixed features the update chain consumes and every
    iteration's potential features, the buffer the end-to-end stage's dipole
    and energy paths read.  The intermediates are captured from the oracle's
    own modules by the runner (the model's forward keeps them as locals).
    """

    record, path = RECORDS[record_index]
    evaluator = evaluators[record["dtype"]]
    dtype = record["dtype"]
    with np.load(path) as arrays:
        inputs = {key: arrays[key] for key in arrays.files}
    state = _drive(evaluator, inputs)
    density_tolerance = CHAIN_TOLERANCE["density_absolute"][dtype]
    energy_tolerance = CHAIN_TOLERANCE["energy_relative"][dtype]
    label = f'{record["geometry"]}/{record["channel"]}/{dtype}'
    feature_tolerance = CHAIN_TOLERANCE["feature_absolute"][dtype]
    state_columns = (
        (
            "spin_charge_density",
            state["spin_charge_density"],
            inputs["spin_charge_density"],
        ),
        ("charge_density", state["charge_density"], inputs["charge_density"]),
        ("spin_density", state["spin_density"], inputs["spin_density"]),
    )
    for name, native, reference in state_columns:
        delta = float(np.abs(np.asarray(native) - reference).max())
        assert delta <= density_tolerance, (label, name, delta)
    # The chain's own intermediates -- the buffers the end-to-end stage's
    # dipole and energy paths read.
    feature_columns = [
        ("features_mixed", state["features_mixed"], inputs["features_mixed"]),
    ]
    for iteration, reference in enumerate(inputs["potential_features"]):
        feature_columns.append(
            (
                f"potential_features[{iteration}]",
                state["potential_features"][iteration],
                reference,
            )
        )
    for name, native, reference in feature_columns:
        delta = float(np.abs(np.asarray(native) - reference).max())
        assert delta <= feature_tolerance, (label, name, delta)
    native_energy = float(np.sum(state["local_electron_energy"]))
    reference_energy = float(np.sum(inputs["local_electron_energy"]))
    scale = max(abs(reference_energy), 1.0e-3)
    relative = abs(native_energy - reference_energy) / scale
    assert relative <= energy_tolerance, (label, native_energy, reference_energy)
