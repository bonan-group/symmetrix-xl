"""Finite-difference gates for the PolarMACE delta layer's adjoints.

The delta layer is node-local, so its derivative surface is the *feature*
adjoint: the seeds of its inputs' feature spaces.  This module gates it in the
established idiom -- a central difference of the layer's own forward along the
all-ones direction against the reverse pass's seed sum -- for every new block
and for the seeded paths of the composed chain.

The composed chain serves every seed the end-to-end stage's energy, dipole, and
stress paths own: the unrolled potential-feature seeds (per iteration), the
Fukui source seeds, the local-electron readout's node-energy seed, the final
equilibrated density's seed, and the last iteration's potential-feature seed
(the readout's own `field_feats` input).  The chain's unrolled step is
`previous + sources`, so its reverse adds the update's output seed to the
previous state's seed through that identity sum as well as through the
update's `local_charges` input; both terms are gated here.

Every comparison is a finite difference, so each one carries a floor: the
inputs whose derivative is *structurally* zero are asserted zero on both sides,
and every other input is asserted to carry signal before its agreement is
compared.  A permissive relative bound must never be the reason a zero passes.
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
TOLERANCE = 1.0e-6
_EPSILON = 1.0e-6
#: The smallest magnitude a finite difference may have before it is noise.
_DIFFERENCE_FLOOR = 1.0e-9


def _checkpoint_path(size: str = "M") -> Path:
    configured = os.environ.get(f"SYMMETRIX_POLAR_MODEL_{size}")
    path = (
        Path(configured)
        if configured
        else (
            Path("~/.cache/symmetrix/test-models").expanduser()
            / f"MACE-POLAR-1-{size}.model"
        )
    )
    if not path.is_file():
        pytest.skip(
            f"the delta adjoint gates need the MACE-POLAR-1-{size} checkpoint "
            f"(set SYMMETRIX_POLAR_MODEL_{size}, or cache it at {path})."
        )
    return path


@pytest.fixture(scope="module")
def handle(tmp_path_factory) -> dict:
    """The M checkpoint's delta layer, parsed through the family's evaluator.

    The fixture yields a container rather than the evaluator itself: pytest
    keeps the yielded value alive until the session ends, and Kokkos refuses to
    finalize while evaluators are alive, so the teardown has to be able to
    empty it.
    """

    from symmetrix.extract_mace_polar import extract_mace_polar_data

    for environment in (
        "SYMMETRIX_POLAR_ORACLE_MACE",
        "SYMMETRIX_POLAR_ORACLE_GRAPH_LONGRANGE",
    ):
        checkout = required_checkout(environment)
        import sys

        checkout_text = str(checkout)
        if checkout_text not in sys.path:
            sys.path.insert(0, checkout_text)
    import torch

    torch.set_default_dtype(torch.float64)
    model = torch.load(str(_checkpoint_path()), map_location="cpu", weights_only=False)
    data = extract_mace_polar_data(model.to(torch.float64))
    directory = tmp_path_factory.mktemp("polar-delta-adjoints")
    path = directory / "MACE-POLAR-1-M.json"
    path.write_text(json.dumps(data))
    container = {
        "evaluator": symmetrix.MACEPolarKokkos(str(path)),
        "model_path": str(path),
    }
    yield container
    container.clear()


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(20261015)


def _directional_check(name, forward, inputs, seeds, gradients, zeros=()):
    """Compare each input's reverse seed with a central difference.

    The direction is the all-ones tensor, so the numeric derivative is the sum
    of the analytic adjoint's entries -- the same scalar on both sides.
    ``zeros`` names the inputs whose derivative is *structurally* zero (the
    block ignores them, or the seed path never reaches them): those must be
    zero on both sides, so an accidental zero elsewhere cannot hide inside a
    permissive comparison.
    """

    for index, (value, seed, gradient) in enumerate(zip(inputs, seeds, gradients)):

        def loss(mutated):
            arguments = list(inputs)
            arguments[index] = mutated
            return float(np.sum(seed * np.asarray(forward(*arguments))))

        numeric = (loss(value + _EPSILON) - loss(value - _EPSILON)) / (2 * _EPSILON)
        analytic = float(np.sum(np.asarray(gradient)))
        if index in zeros:
            assert abs(numeric) <= _DIFFERENCE_FLOOR, (name, index, numeric)
            assert abs(analytic) <= _DIFFERENCE_FLOOR, (name, index, analytic)
            continue
        assert (
            abs(numeric) > _DIFFERENCE_FLOOR
        ), f"{name}: input {index} has no signal to compare against"
        assert abs(analytic - numeric) <= TOLERANCE * max(
            1.0, abs(numeric)
        ), f"{name}: input {index} analytic {analytic} numeric {numeric}"


def _shapes(evaluator) -> dict:
    return dict(evaluator.polar_delta_shapes())


def test_the_tensor_products_adjoint_agrees_with_a_difference(handle, rng):
    """The node-local tied-channel tensor product in both of its paths.

    The contraction path mixes densely across the multiplicity channels (the
    shipped checkpoints' 512x512 weight blocks), so this is the family's
    genuinely new kernel rather than a single-channel scaling.
    """

    evaluator = handle["evaluator"]
    shapes = _shapes(evaluator)
    nodes = 4
    width = shapes["node_feature_dimension"]
    invariants = shapes["invariant_dimension"]
    first = rng.normal(size=(nodes, width))
    second = rng.normal(size=(nodes, width))
    output = evaluator.polar_delta_block("dot_products", 0, [first, second])
    seed = rng.normal(size=np.asarray(output).shape)
    adjoints = evaluator.polar_delta_block_reverse(
        "dot_products", 0, [first, second], [seed]
    )
    _directional_check(
        "dot_products",
        lambda a, b: evaluator.polar_delta_block("dot_products", 0, [a, b]),
        [first, second],
        [seed, seed],
        adjoints,
    )

    second = rng.normal(size=(nodes, invariants))
    output = evaluator.polar_delta_block("tp_out", 0, [first, second])
    seed = rng.normal(size=np.asarray(output).shape)
    adjoints = evaluator.polar_delta_block_reverse("tp_out", 0, [first, second], [seed])
    _directional_check(
        "tp_out",
        lambda a, b: evaluator.polar_delta_block("tp_out", 0, [a, b]),
        [first, second],
        [seed, seed],
        adjoints,
    )


def test_the_dense_tensor_product_accepts_an_empty_batch(handle):
    """Prepared dense UVU workspaces treat zero rows as valid empty work."""

    evaluator = handle["evaluator"]
    shapes = _shapes(evaluator)
    width = shapes["node_feature_dimension"]
    first = np.empty((0, width), dtype=np.float64)
    second = np.empty((0, width), dtype=np.float64)
    output = evaluator.polar_delta_block("dot_products", 0, [first, second])
    assert np.asarray(output).shape[0] == 0
    adjoints = evaluator.polar_delta_block_reverse(
        "dot_products", 0, [first, second], [np.asarray(output)]
    )
    assert np.asarray(adjoints[0]).shape == first.shape
    assert np.asarray(adjoints[1]).shape == second.shape


def test_the_gate_accepts_an_empty_batch(handle):
    """The element-parallel gate treats zero rows as valid empty work."""

    evaluator = handle["evaluator"]
    shapes = _shapes(evaluator)
    values = np.empty((0, shapes["gate_input_dimension"]), dtype=np.float64)
    output = evaluator.polar_delta_block("gate", 0, [values])
    assert np.asarray(output).shape[0] == 0
    adjoint = evaluator.polar_delta_block_reverse(
        "gate", 0, [values], [np.asarray(output)]
    )
    assert np.asarray(adjoint).shape == values.shape


def test_the_direct_blocks_reject_mismatched_reverse_batches(handle):
    """Native gate and UVU adjoints validate every row extent before launch."""

    evaluator = handle["evaluator"]
    shapes = _shapes(evaluator)
    width = shapes["node_feature_dimension"]
    first = np.zeros((2, width), dtype=np.float64)
    second = np.zeros((2, width), dtype=np.float64)
    dot_output = evaluator.polar_delta_block("dot_products", 0, [first, second])
    with pytest.raises(ValueError, match="reverse batch dimensions"):
        evaluator.polar_delta_block_reverse(
            "dot_products", 0, [first, second], [np.asarray(dot_output)[:1]]
        )

    gate_input = np.zeros((2, shapes["gate_input_dimension"]), dtype=np.float64)
    gate_output = evaluator.polar_delta_block("gate", 0, [gate_input])
    with pytest.raises(ValueError, match="reverse batch dimensions"):
        evaluator.polar_delta_block_reverse(
            "gate", 0, [gate_input], [np.asarray(gate_output)[:1]]
        )


def test_the_gate_and_readout_adjoints_agree_with_a_difference(handle, rng):
    """The gated nonlinearity, alone and through the readout stack.

    The gate's own split (scalars, then gates, then gated channels) is gated
    directly rather than only through the layers around it; its width is the
    readout's hidden space, which the layer reports rather than a literal.
    """

    evaluator = handle["evaluator"]
    shapes = _shapes(evaluator)
    nodes = 4
    features = rng.normal(size=(nodes, shapes["node_feature_dimension"]))
    output = evaluator.polar_delta_block("gate_readout", 0, [features])
    seed = rng.normal(size=np.asarray(output).shape)
    adjoint = evaluator.polar_delta_block_reverse("gate_readout", 0, [features], [seed])
    _directional_check(
        "gate_readout",
        lambda a: evaluator.polar_delta_block("gate_readout", 0, [a]),
        [features],
        [seed],
        [adjoint],
    )

    values = rng.normal(size=(nodes, shapes["gate_input_dimension"]))
    output = evaluator.polar_delta_block("gate", 0, [values])
    seed = rng.normal(size=np.asarray(output).shape)
    adjoint = evaluator.polar_delta_block_reverse("gate", 0, [values], [seed])
    _directional_check(
        "gate",
        lambda a: evaluator.polar_delta_block("gate", 0, [a]),
        [values],
        [seed],
        [adjoint],
    )


def test_the_update_block_adjoint_agrees_with_a_difference(handle, rng):
    """Every input of the field update: features, potential, attrs, charges.

    The node-attribute path is live here (the update's source embedding reads
    it), so none of the four grades is structurally zero.
    """

    evaluator = handle["evaluator"]
    shapes = _shapes(evaluator)
    nodes = 4
    inputs = [
        rng.normal(size=(nodes, shapes["node_feature_dimension"])),
        rng.normal(size=(nodes, shapes["potential_dimension"])),
        rng.normal(size=(nodes, 83)),
        rng.normal(size=(nodes, shapes["charges_dimension"])),
    ]
    output = evaluator.polar_delta_block("field_update", 0, inputs)
    seed = rng.normal(size=np.asarray(output).shape)
    diagnostics_before = evaluator.polar_e3linear_diagnostics()
    adjoints = evaluator.polar_delta_block_reverse("field_update", 0, inputs, [seed])
    diagnostics_after = evaluator.polar_e3linear_diagnostics()
    # The update readout's three linears and its source embedding remain
    # individually dispatched.  The potential embedding's three transposes
    # are one composed device path; its reference schedule reports seven.
    assert diagnostics_after["reverse_count"] - diagnostics_before["reverse_count"] == 4
    assert (
        diagnostics_after["fallback_count"] - diagnostics_before["fallback_count"] == 0
    )
    _directional_check(
        "field_update",
        lambda *a: evaluator.polar_delta_block("field_update", 0, list(a)),
        inputs,
        [seed] * len(inputs),
        adjoints,
    )


def test_the_local_electron_readout_adjoint_agrees_with_a_difference(handle, rng):
    """The post-loop readout's four inputs, including both charge producers.

    The readout sums its two charge inputs, so both carry the same seed: the
    two entries are compared independently rather than assumed equal.
    """

    evaluator = handle["evaluator"]
    shapes = _shapes(evaluator)
    nodes = 4
    inputs = [
        rng.normal(size=(nodes, shapes["node_feature_dimension"])),
        rng.normal(size=(nodes, shapes["potential_dimension"])),
        rng.normal(size=(nodes, shapes["charges_dimension"])),
        rng.normal(size=(nodes, shapes["charges_dimension"])),
    ]
    output = evaluator.polar_delta_block("local_electron_readout", 0, inputs)
    seed = rng.normal(size=np.asarray(output).shape)
    adjoints = evaluator.polar_delta_block_reverse(
        "local_electron_readout", 0, inputs, [seed]
    )
    _directional_check(
        "local_electron_readout",
        lambda *a: evaluator.polar_delta_block("local_electron_readout", 0, list(a)),
        inputs,
        [seed] * len(inputs),
        adjoints,
    )


def test_the_per_layer_heads_mixer_and_source_map_adjoints_agree(handle, rng):
    """The three per-layer blocks the chain's own orchestration drives."""

    evaluator = handle["evaluator"]
    shapes = _shapes(evaluator)
    nodes = 4
    features = rng.normal(size=(nodes, shapes["node_feature_dimension"]))
    for name in ("spin_source_head", "feature_mixer", "fukui_source_map"):
        output = evaluator.polar_delta_block(name, 0, [features])
        seed = rng.normal(size=np.asarray(output).shape)
        adjoint = evaluator.polar_delta_block_reverse(name, 0, [features], [seed])
        _directional_check(
            name,
            lambda a, block=name: evaluator.polar_delta_block(block, 0, [a]),
            [features],
            [seed],
            [adjoint],
        )


def _chain_inputs(record: str) -> dict:
    with np.load(DATA_DIR / record) as arrays:
        return {
            "node_features": np.ascontiguousarray(arrays["node_features"]),
            "node_attrs": np.ascontiguousarray(arrays["node_attrs"]),
            "positions": np.ascontiguousarray(arrays["positions"]),
            "cell": np.ascontiguousarray(arrays["cell"]),
            "graph_pbc": np.ascontiguousarray(arrays["graph_pbc"], dtype=np.int32),
            "node_graph_index": np.ascontiguousarray(
                arrays["node_graph_index"], dtype=np.int32
            ),
            "total_charge": [float(value) for value in arrays["total_charge"]],
            "total_spin": [float(value) for value in arrays["total_spin"]],
            "external_potential": np.ascontiguousarray(arrays["external_potential"]),
        }


def _singlet_record() -> str:
    manifest = json.loads((DATA_DIR / "chain_manifest.json").read_text())
    return next(
        item["name"]
        for item in manifest["records"]
        if item["dtype"] == "float64"
        and item["geometry"] == "mol_h2o"
        and item["channel"] == "singlet"
    )


#: Which of the three grades a seed *structurally* cannot reach, and why.  The
#: source map and the potential-feature seeds never reach the update's
#: node-attribute path (its potential input feeds the embedding and the chain's
#: feature path, not the source embedding), and a source-map seed never reaches
#: the external-field coupling either (the coupling enters the features, which
#: that seed does not pass through).
_ZERO_GRADES = {
    "first_iteration_features": ("node_attrs",),
    "source_map": ("node_attrs", "external_potential"),
    "first_update_sources": (),
    "local_electron_energy": (),
    "final_density": (),
    "last_iteration_features": (),
}


@pytest.mark.parametrize(
    "seed_kind",
    (
        "first_iteration_features",
        "source_map",
        "first_update_sources",
        "local_electron_energy",
        "final_density",
        "last_iteration_features",
    ),
)
def test_the_chain_reverse_matches_a_difference(handle, rng, seed_kind):
    """The composed chain's served seeds, against a central difference.

    The seeds are the ones the delta layer's own orchestration owns: the
    unrolled iterations' potential features, the pre-loop source map's output,
    the updates' source pairs, the readout's node energy, the equilibrated
    density, and the readout's field-feature input.
    """

    evaluator = handle["evaluator"]
    shapes = _shapes(evaluator)
    inputs = _chain_inputs(_singlet_record())
    state = evaluator.polar_delta_chain(**inputs)
    steps = shapes["steps"]
    empty_potentials = [
        np.zeros((0, shapes["potential_dimension"])) for _ in range(steps)
    ]
    empty_sources = [np.zeros((0, 2)) for _ in range(steps + 1)]
    energy_seed = np.zeros((0, 1))
    density_seed = np.zeros((0, shapes["charges_dimension"]))
    if seed_kind == "first_iteration_features":
        potential_seeds = list(empty_potentials)
        potential_seeds[0] = rng.normal(
            size=np.asarray(state["potential_features"][0]).shape
        )
        source_seeds = []
    elif seed_kind == "source_map":
        potential_seeds = []
        source_seeds = list(empty_sources)
        source_seeds[0] = rng.normal(size=np.asarray(state["fukui_sources"][0]).shape)
    elif seed_kind == "first_update_sources":
        potential_seeds = []
        source_seeds = list(empty_sources)
        source_seeds[1] = rng.normal(size=np.asarray(state["fukui_sources"][1]).shape)
    elif seed_kind == "local_electron_energy":
        potential_seeds = []
        source_seeds = []
        energy_seed = rng.normal(size=np.asarray(state["local_electron_energy"]).shape)
    elif seed_kind == "final_density":
        potential_seeds = []
        source_seeds = []
        density_seed = rng.normal(size=np.asarray(state["spin_charge_density"]).shape)
    else:
        potential_seeds = list(empty_potentials)
        potential_seeds[-1] = rng.normal(
            size=np.asarray(state["potential_features"][-1]).shape
        )
        source_seeds = []

    def loss(**overrides):
        driven = evaluator.polar_delta_chain(**{**inputs, **overrides})
        value = sum(
            float(np.sum(seed * np.asarray(features)))
            for seed, features in zip(potential_seeds, driven["potential_features"])
            if np.size(seed)
        )
        for seed, sources in zip(source_seeds, driven["fukui_sources"]):
            if np.size(seed):
                value += float(np.sum(seed * np.asarray(sources)))
        if np.size(energy_seed):
            value += float(
                np.sum(energy_seed * np.asarray(driven["local_electron_energy"]))
            )
        if np.size(density_seed):
            value += float(
                np.sum(density_seed * np.asarray(driven["spin_charge_density"]))
            )
        return value

    adjoints = evaluator.polar_delta_chain_reverse(
        **inputs,
        local_electron_energy_seed=energy_seed,
        spin_charge_density_seed=density_seed,
        potential_features_seeds=potential_seeds,
        fukui_sources_seeds=source_seeds,
    )
    for name in ("node_features", "node_attrs", "external_potential"):
        array = np.ascontiguousarray(inputs[name])
        numeric = (
            loss(**{name: array + _EPSILON}) - loss(**{name: array - _EPSILON})
        ) / (2 * _EPSILON)
        analytic = float(np.sum(np.asarray(adjoints[name])))
        if name in _ZERO_GRADES[seed_kind]:
            assert abs(numeric) <= _DIFFERENCE_FLOOR, (seed_kind, name, numeric)
            assert abs(analytic) <= _DIFFERENCE_FLOOR, (seed_kind, name, analytic)
            continue
        assert abs(numeric) > _DIFFERENCE_FLOOR, (seed_kind, name, "no signal")
        assert abs(analytic - numeric) <= TOLERANCE * max(1.0, abs(numeric)), (
            seed_kind,
            name,
            analytic,
            numeric,
        )


#: The seeds whose *position* adjoint the composed chain must serve: every
#: energy, dipole, and stress path routes through them, and the delta layer's
#: own gate covered their feature grades only.
_POSITION_SEED_KINDS = (
    "local_electron_energy",
    "final_density",
    "first_iteration_features",
)


@pytest.mark.parametrize("seed_kind", _POSITION_SEED_KINDS)
def test_the_chain_reverse_position_adjoint_matches_a_difference(handle, seed_kind):
    """The composed chain's position adjoint, against a per-component difference.

    Task 13d's isolation: the family's energy is a functional of the chain's
    state, so its forces are the chain's position adjoint fed with the energy
    path's seeds.  This gate pins that adjoint for the three seed routes the
    family uses, per node and component (a directional sum would be blind here:
    the chain is translation invariant, so the all-ones direction is exactly
    zero).
    """

    evaluator = handle["evaluator"]
    shapes = _shapes(evaluator)
    inputs = _chain_inputs(_singlet_record())
    nodes = inputs["node_graph_index"].shape[0]
    steps = shapes["steps"]
    charges = shapes["charges_dimension"]
    potentials = shapes["potential_dimension"]
    empty_c = np.zeros((0, charges))
    empty_p = [np.zeros((0, potentials)) for _ in range(steps)]
    if seed_kind == "local_electron_energy":
        seeds = dict(
            local_electron_energy_seed=np.ones((nodes, 1)),
            spin_charge_density_seed=empty_c,
            potential_features_seeds=[],
            fukui_sources_seeds=[],
        )

        def loss(**over):
            return float(
                np.sum(
                    np.asarray(
                        evaluator.polar_delta_chain(**{**inputs, **over})[
                            "local_electron_energy"
                        ]
                    )
                )
            )
    elif seed_kind == "final_density":
        density_seed = np.linspace(1.0, 2.0, nodes * charges).reshape(nodes, charges)
        seeds = dict(
            local_electron_energy_seed=np.zeros((0, 1)),
            spin_charge_density_seed=density_seed,
            potential_features_seeds=[],
            fukui_sources_seeds=[],
        )

        def loss(**over):
            return float(
                np.sum(
                    density_seed
                    * np.asarray(
                        evaluator.polar_delta_chain(**{**inputs, **over})[
                            "spin_charge_density"
                        ]
                    )
                )
            )
    else:
        potential_seeds = list(empty_p)
        feature_seed = np.linspace(1.0, 2.0, nodes * potentials).reshape(
            nodes, potentials
        )
        potential_seeds[0] = feature_seed
        seeds = dict(
            local_electron_energy_seed=np.zeros((0, 1)),
            spin_charge_density_seed=empty_c,
            potential_features_seeds=potential_seeds,
            fukui_sources_seeds=[],
        )

        def loss(**over):
            return float(
                np.sum(
                    feature_seed
                    * np.asarray(
                        evaluator.polar_delta_chain(**{**inputs, **over})[
                            "potential_features"
                        ][0]
                    )
                )
            )

    adjoints = evaluator.polar_delta_chain_reverse(**inputs, **seeds)
    analytic = np.asarray(adjoints["positions"], dtype=np.float64)
    positions = np.ascontiguousarray(inputs["positions"], dtype=np.float64)
    numeric = np.zeros_like(positions)
    for node in range(nodes):
        for component in range(3):
            delta = np.zeros_like(positions)
            delta[node, component] = _EPSILON
            numeric[node, component] = (
                loss(positions=positions + delta) - loss(positions=positions - delta)
            ) / (2 * _EPSILON)
    assert np.abs(numeric).max() > _DIFFERENCE_FLOOR, (seed_kind, "no signal")
    assert np.abs(analytic - numeric).max() <= TOLERANCE * max(
        1.0, float(np.abs(numeric).max())
    ), (seed_kind, analytic, numeric)


def test_the_native_composed_reverse_matches_the_granular_reference(handle):
    """A fresh production reverse owns its complete resident forward state."""

    # A new evaluator has no composed density or graph multipoles left by a
    # prior forward.  This keeps the legacy reverse boundary from accidentally
    # passing by consuming stale resident buffers.
    evaluator = symmetrix.MACEPolarKokkos(handle["model_path"])
    inputs = _chain_inputs(_singlet_record())
    state = evaluator.polar_delta_chain(**inputs)
    density = np.ascontiguousarray(
        np.asarray(state["charge_density"], dtype=np.float64)
    )
    contract = dict(evaluator.polar_long_range_contract)
    ops = symmetrix.PolarLongRangeOps(
        kspace_cutoff=float(contract["kspace_cutoff"]),
        density_max_l=int(contract["density_max_l"]),
        density_smearing_width=float(contract["density_smearing_width"]),
        feature_max_l=int(contract["feature_max_l"]),
        feature_smearing_widths=[
            float(value) for value in contract["feature_smearing_widths"]
        ],
    )
    cache = ops.geometry_cache(
        inputs["positions"], inputs["cell"], inputs["node_graph_index"]
    )["cache"]
    grad_energy = np.ones(len(inputs["total_charge"]), dtype=np.float64)
    self_fields = None
    if not contract["include_electrostatic_self_interaction"]:
        self_fields = ops.energy_self_fields(density)
    energy = ops.adjoint_coulomb_energy(cache, grad_energy, density, self_fields)
    correction = ops.adjoint_correction_energy(
        density,
        inputs["positions"],
        inputs["cell"],
        inputs["node_graph_index"],
        inputs["graph_pbc"],
        grad_energy,
    )
    electric_field = np.asarray(inputs["external_potential"], dtype=np.float64)[:, 1:4]
    dipole = ops.adjoint_dipole_readout(
        density,
        inputs["positions"],
        inputs["node_graph_index"],
        np.zeros_like(grad_energy),
        electric_field,
    )
    density_seed = (
        np.asarray(energy["grad_charge_density"])
        + np.asarray(correction["grad_charge_density"])
        + np.asarray(dipole["grad_charge_density"])
    )
    if self_fields is not None:
        density_seed += np.asarray(
            ops.adjoint_energy_self_fields(-0.5 * density), dtype=np.float64
        )
    reference = evaluator.polar_delta_chain_reverse(
        **inputs,
        local_electron_energy_seed=np.ones((density.shape[0], 1), dtype=np.float64),
        spin_charge_density_seed=np.ascontiguousarray(
            np.hstack([density_seed, density_seed])
        ),
        potential_features_seeds=[],
        fukui_sources_seeds=[],
        table_seeds_cos=np.ascontiguousarray(energy["grad_cos_table"]).reshape(-1),
        table_seeds_sin=np.ascontiguousarray(energy["grad_sin_table"]).reshape(-1),
    )
    expected_positions = (
        np.asarray(reference["positions"])
        + np.asarray(correction["grad_positions"])
        + np.asarray(dipole["grad_positions"])
    )
    native = evaluator.polar_delta_composed_reverse(**inputs)
    np.testing.assert_allclose(
        native["positions"], expected_positions, rtol=0, atol=1e-12
    )
    for name in (
        "node_features",
        "node_attrs",
        "total_charge",
        "total_spin",
    ):
        np.testing.assert_allclose(native[name], reference[name], rtol=0, atol=1e-12)

    # The granular chain oracle above is seeded with the density derivative;
    # it intentionally does not own the composed energy's direct dipole-to-
    # field or cell-strain terms.  The complete transaction does, and is the
    # independent boundary oracle for those two compact outputs as well as a
    # guard that the legacy reverse ran the same resident forward first.
    transaction = evaluator.polar_delta_composed_forward(**inputs, compute_reverse=True)
    for transaction_name, reverse_name in (
        ("node_features_adjoint", "node_features"),
        ("node_attrs_adjoint", "node_attrs"),
        ("positions_adjoint", "positions"),
        ("external_potential_adjoint", "external_potential"),
        ("total_charge_adjoint", "total_charge"),
        ("total_spin_adjoint", "total_spin"),
        ("strain_adjoint", "strain"),
        ("stress", "stress"),
    ):
        np.testing.assert_allclose(
            transaction[transaction_name], native[reverse_name], rtol=0, atol=1e-12
        )
