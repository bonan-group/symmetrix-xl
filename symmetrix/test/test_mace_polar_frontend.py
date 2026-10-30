"""The PolarMACE family entry and the calculator surface it adds.

The family declares itself through the registry rather than forking the
frontend: detection by its own model type, its own schema and extractor, its
evaluator classes (the MH-1 backbone evaluator under the polar schema), the
backbone-delta contract its models carry, and the two routings every family
declares.  This module pins that declaration against the program the evaluator
compiles -- the family identity, the capability key set, the
interaction-count-scaled backbone assembly, and the dipole output the program
carries -- and the calculator surface the family adds: the graph-level scalars
(total charge, total spin, fermi level, and the external electric field)
threaded through the prepared lifecycle, and the dipole output produced by the
family composition.

The synthetic model is the backbone builder of
:mod:`test_mace_polar_backbone`, built once per module in both readout
variants: the plain nonlinear readout and the bias readout the released M and L
checkpoints use.  The real MACE-POLAR-1 checkpoints are exercised when the
environment names them (``SYMMETRIX_POLAR_MODEL_S``/``_M``/``_L``, following
:mod:`test_polar_checkpoint_facts`); otherwise those tests skip with the
missing prerequisite named.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import symmetrix
import symmetrix.family_registry as family_registry
import symmetrix.mace_polar_schema as polar_schema
from polar_oracle_checkout import optional_checkout
from symmetrix.calculator import Symmetrix
from symmetrix.extract_mace_polar import extract_mace_polar_data
from symmetrix.mace_polar_family import FAMILY_ENTRY
from symmetrix.mace_polar_schema import POLAR_CONTRACT_KEY, POLAR_MODEL_TYPE
from test_mace_polar_backbone import _build_polar_shaped_model

_ORACLE_MACE_CHECKOUT = optional_checkout("SYMMETRIX_POLAR_ORACLE_MACE")
_ORACLE_GRAPH_ELECTROSTATICS_CHECKOUT = optional_checkout(
    "SYMMETRIX_POLAR_ORACLE_GRAPH_LONGRANGE"
)
_FACTS_PATH = (
    Path(__file__).resolve().parent / "data" / "polar" / "checkpoint_facts.json"
)

#: The released checkpoint sizes the family's models ship in, in the order the
#: pinned facts record them.
MODEL_SIZES = ("S", "M", "L")

#: The long-range node set every polar program carries: the family's global
#: operators, which are the whole of what the backbone program does not
#: provide.  The set is a property of the program's assembly, not of the
#: backbone, so it is the same for every interaction count.
LONG_RANGE_NODES = frozenset(
    "polar.long_range." + name
    for name in (
        "kspace_geometry_cache",
        "kspace_coulomb_features",
        "kspace_coulomb_energy",
        "self_interaction_linear",
        "external_field_projection",
        "fukui_normalize",
        "dipole_readout",
        "analytic_correction_energy",
        "analytic_correction_features",
    )
)

#: A three-atom cluster over the synthetic backbone's two supported elements
#: (H and Si), with every pair inside the model's 4 Å cutoff.
POSITIONS = ((0.0, 0.0, 0.0), (2.3, 0.0, 0.0), (0.4, 1.3, 0.2))

#: The water molecule the checkpoint tests evaluate: the real models cover 83
#: elements, and a molecule in a box exercises the cell-dependent long-range
#: group's geometry without periodic images.
WATER_POSITIONS = ((0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0))
WATER_BOX = 12.0


def _molecule():
    import numpy as np
    from ase import Atoms

    # The cluster lives in the water fixture's 12 A box so both reciprocal
    # compatibility cases and real-space public stress have a reporting volume.
    return Atoms("Si2H", positions=POSITIONS, cell=np.eye(3) * WATER_BOX, pbc=False)


def _water():
    import numpy as np
    from ase import Atoms

    return Atoms(
        "OH2",
        positions=WATER_POSITIONS,
        cell=np.eye(3) * WATER_BOX,
        pbc=False,
    )


def _write_model(path: Path, data) -> Path:
    path.write_text(json.dumps(data, separators=(",", ":")))
    return path


def _native_module():
    """Load native code only for tests that actually exercise it."""

    return symmetrix.load_backend()


@pytest.fixture(scope="module")
def polar_models(tmp_path_factory) -> dict[str, Path]:
    """The family's synthetic files, one per readout variant."""

    from mace.modules.blocks import NonLinearBiasReadoutBlock

    root = tmp_path_factory.mktemp("mace-polar")
    variants = {"plain": None, "bias": NonLinearBiasReadoutBlock}
    return {
        name: _write_model(
            root / f"{name}.json",
            extract_mace_polar_data(_build_polar_shaped_model(readout_cls=readout)),
        )
        for name, readout in variants.items()
    }


@pytest.fixture(scope="module")
def polar_model_path(polar_models) -> Path:
    """The family's file with the bias readout the released M and L models use."""

    return polar_models["bias"]


@pytest.fixture(scope="module", autouse=True)
def polar_jit_cache(tmp_path_factory):
    """Reuse immutable JIT artifacts across this integration module."""

    cache_root = tmp_path_factory.mktemp("polar-jit-cache")
    cache_root.chmod(0o700)
    previous = os.environ.get("SYMMETRIX_JIT_CACHE")
    os.environ["SYMMETRIX_JIT_CACHE"] = str(cache_root)
    try:
        yield cache_root
    finally:
        if previous is None:
            os.environ.pop("SYMMETRIX_JIT_CACHE", None)
        else:
            os.environ["SYMMETRIX_JIT_CACHE"] = previous


def _polar_calculator(model_path, dtype="float64", **kwargs) -> Symmetrix:
    """The family's calculator on the reference routing it declares."""

    kwargs.setdefault("polar_electrostatics", "direct_reciprocal")
    return Symmetrix(str(model_path), dtype=dtype, streamed_edges="reference", **kwargs)


def test_cuda_public_output_launches_keep_the_polar_packer_non_public():
    """CUDA launch functors must not expose the Polar pack API on MH-1."""

    source_root = Path(__file__).resolve().parents[2] / "libsymmetrix" / "source"
    neutral_implementation = (
        source_root / "mace_nonlinear_kokkos_impl.tpp"
    ).read_text()
    polar_header = (source_root / "mace_polar_kokkos.hpp").read_text()
    for functor in (
        "PublicEnergyReducer",
        "PublicPayloadPacker",
        "PublicGraphEnergyAdder",
        "PublicDipolePacker",
    ):
        assert f"struct {functor}" in polar_header
        assert f"struct {functor}" not in neutral_implementation
    declaration = polar_header.index("pack_polar_public_outputs_device(")
    assert polar_header.rfind("private:", 0, declaration) > polar_header.rfind(
        "public:", 0, declaration
    )
    consumer = (source_root / "mace_polar_consumer.hpp").read_text()
    assert "struct PolarPublicOutputCopier" in consumer
    assert "const auto copy_output=[]" not in consumer
    assert consumer.count("copy_polar_public_output_device(") == 7


def _select_realspace_reverse_strategy(evaluator, strategy):
    evaluator._set_polar_realspace_reverse_strategy(strategy)
    assert evaluator.polar_delta_workspace()["realspace_reverse_strategy"] == strategy


def _assert_pair_atomic_was_used(evaluator):
    workspace = evaluator.polar_delta_workspace()
    assert workspace["realspace_reverse_strategy"] == "pair_atomic"
    assert workspace["realspace_feature_pair_atomic_selected_count"] > 0
    assert workspace["realspace_energy_pair_atomic_selected_count"] > 0
    assert workspace["realspace_feature_pair_league_extent"] > 0
    assert workspace["realspace_energy_pair_league_extent"] > 0


def _assert_realspace_reverse_resolution(evaluator, requested, resolved, reason):
    workspace = evaluator.polar_delta_workspace()
    assert workspace["realspace_reverse_strategy_requested"] == requested
    assert workspace["realspace_reverse_strategy_resolved"] == resolved
    assert workspace["realspace_reverse_strategy_reason"] == reason
    return workspace


def test_polar_solver_request_is_runtime_only(polar_model_path):
    """Changing evaluator policy must not mutate JSON or program fingerprints."""

    before = polar_model_path.read_bytes()
    automatic = Symmetrix(
        str(polar_model_path),
        dtype="float64",
        streamed_edges="reference",
    )
    periodic = Symmetrix(
        str(polar_model_path),
        dtype="float64",
        streamed_edges="reference",
        polar_electrostatics="periodic",
    )

    assert automatic.polar_electrostatics == "auto"
    assert periodic.polar_electrostatics == "periodic"
    assert (
        automatic.evaluator.family_program["fingerprint"]
        == (periodic.evaluator.family_program["fingerprint"])
    )
    assert automatic.evaluator.polar_long_range_contract == (
        periodic.evaluator.polar_long_range_contract
    )
    assert polar_model_path.read_bytes() == before
    serialized = json.loads(before)
    assert "polar_electrostatics" not in serialized
    assert "solver" not in serialized
    assert "pbc_handling" not in serialized


def test_polar_solver_request_rejects_unknown_spelling_before_model_loading():
    """A misspelled policy must not silently select a boundary condition."""

    with pytest.raises(ValueError, match="polar_electrostatics must be one of"):
        Symmetrix("does-not-exist.json", polar_electrostatics="ewald")


@pytest.mark.parametrize(
    "option",
    (
        {"polar_pme_mesh": (32.5, 32, 32)},
        {"polar_pme_mesh": (True, 32, 32)},
        {"polar_pme_order": 5.5},
        {"polar_pme_order": True},
    ),
)
def test_polar_pme_runtime_policy_rejects_lossy_integer_coercion(option):
    """Mesh policy must reject values that would otherwise be truncated."""

    with pytest.raises(ValueError, match="positive integer"):
        Symmetrix("does-not-exist.json", **option)


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("polar_pme_mesh", (32.5, 32, 32)),
        ("polar_pme_mesh", (True, 32, 32)),
        ("polar_pme_order", 5.5),
        ("polar_pme_order", True),
    ),
)
def test_polar_pme_runtime_policy_setters_reject_lossy_integer_coercion(
    polar_model_path, name, value
):
    calculator = _polar_calculator(polar_model_path)
    before = getattr(calculator, name)

    with pytest.raises(ValueError, match="positive integer"):
        setattr(calculator, name, value)

    assert getattr(calculator, name) == before


def _entry():
    entry = family_registry.declared_registry().detect(model_type=POLAR_MODEL_TYPE)
    assert entry is not None, "the polar family is not declared"
    return entry


def _program_nodes(program):
    return {node["id"]: node for node in program["operators"]}


def test_the_family_entry_declares_the_family_it_serves(polar_model_path):
    """The entry is the family's own declaration, resolved through the registry.

    The registry surface is the whole of the family's declaration: detection,
    schema, extractor, evaluator classes, contract, code generators, artifact
    adapters, capabilities, routings, and surfaces.  A forked calculator path
    would show up here as a declaration the entry does not carry.
    """

    entry = _entry()

    assert entry is FAMILY_ENTRY
    assert entry.key == "MACE_Polar"
    assert entry.detection.model_types == (POLAR_MODEL_TYPE,)
    assert entry.detection.architectures == ()
    assert {routing.key for routing in entry.routings} == {"direct", "reference"}
    assert family_registry.surface_of(entry, "polar_graph_inputs").methods == (
        "set_polar_graph_inputs",
    )
    assert family_registry.resolve_symbol(entry.schema) is polar_schema
    assert family_registry.resolve_symbol(entry.extractor) is extract_mace_polar_data
    assert entry.contract.json_key == POLAR_CONTRACT_KEY
    assert family_registry.resolve_symbol(entry.contract.builder).__name__ == (
        "make_execution_polar_backbone_contract"
    )

    for dtype in ("float64", "float32"):
        expected = family_registry.resolve_evaluator_class(
            entry, dtype=dtype, native=_native_module()
        )
        calculator = _polar_calculator(polar_model_path, dtype=dtype)
        assert isinstance(calculator.evaluator, expected)
        assert calculator.evaluator.has_polar_long_range_contract
        assert (
            calculator.evaluator.polar_delta_shapes()["long_range_scalar_bits"]
            == np.dtype(dtype).itemsize * 8
        )
        cache_node = _program_nodes(calculator.evaluator.family_program)[
            "polar.long_range.kspace_geometry_cache"
        ]
        phase_tables = {
            output["name"]: output["dtype"] for output in cache_node["outputs"]
        }
        assert phase_tables["cos_table"] == dtype
        assert phase_tables["sin_table"] == dtype


def test_the_declared_capabilities_are_the_ones_the_program_carries(polar_model_path):
    """The entry's family identity and capabilities match the compiled program.

    A family's capability declaration is honest only if the program it compiles
    carries the corresponding operators, so the program-derived properties are
    read off the program rather than asserted from the declaration.
    """

    entry = _entry()
    calculator = _polar_calculator(polar_model_path)
    evaluator = calculator.evaluator
    program = evaluator.family_program
    nodes = _program_nodes(program)

    assert program["family"] == entry.capabilities.family == "polar-mace"
    assert family_registry.program_family_of(entry, evaluator) == (
        entry.capabilities.family
    )
    capabilities = family_registry.capabilities_of(entry, evaluator)
    assert set(capabilities) == set(family_registry.CAPABILITY_KEYS)

    # The reference routing is single-thread execution on the default host
    # execution space, so it exists exactly where that space is the host one.
    assert capabilities["reference"] == (capabilities["target"] == "host")
    # The field coupling is a program property: the polar program carries the
    # external-field projection node, and the capability follows the program.
    assert capabilities["field"] == (
        "polar.long_range.external_field_projection" in nodes
    )
    assert capabilities["field"] is True
    # The generated tier is the MH-1 node program, which does not admit the
    # family's keep-last-layer topology or its bias readout.
    assert capabilities["generated"] is False
    assert capabilities["stress"] is True


#: The backbone topologies the family's program assembly is exercised over:
#: the interaction count with the hidden irreps that topology carries.  A
#: single layer's node features are the backbone embedding, which is the
#: hidden irreps only when they are scalar-only, so the one-interaction
#: topology is the released S checkpoint's shape (scalar hidden) and the
#: two- and three-interaction topologies are the M and L shape.
BACKBONE_TOPOLOGIES = (
    (1, "4x0e"),
    (2, "4x0e+4x1o"),
    (3, "4x0e+4x1o"),
)


@pytest.mark.parametrize(("interactions", "hidden_irreps"), BACKBONE_TOPOLOGIES)
def test_the_program_scales_its_backbone_with_the_interaction_count(
    tmp_path, interactions, hidden_irreps
):
    """The assembly is the backbone plus the family's global node set.

    The shipped checkpoints carry one, two, and three interactions; the program
    the evaluator compiles must scale its backbone nodes with the count while
    the long-range group it adds stays the family's own.
    """

    data = extract_mace_polar_data(
        _build_polar_shaped_model(
            num_interactions=interactions, hidden_irreps=hidden_irreps
        )
    )
    evaluator = _native_module().MACEPolarKokkos(
        str(_write_model(tmp_path / f"polar-{interactions}.json", data))
    )
    nodes = _program_nodes(evaluator.family_program)

    assert {
        node_id.split(".")[1]
        for node_id in nodes
        if node_id.startswith("mh1.interaction")
    } == {f"interaction{index}" for index in range(interactions)}
    assert set(nodes) & LONG_RANGE_NODES == LONG_RANGE_NODES
    assert (
        evaluator.polar_schedule["recursion_step_count"]
        == (data["polar"]["num_recursion_steps"])
    )


def test_the_shared_extractor_routes_a_polar_checkpoint_to_the_family_extractor(
    tmp_path,
):
    """A PolarMACE checkpoint reaches the family's own extractor by its deltas.

    The family's extractor is reached two ways: the entry's declared symbol, and
    the shared checkpoint front-end, which recognizes the polar backbone deltas
    structurally (``PolarMACE.__init__`` itself needs the out-of-tree
    ``graph_longrange`` package, so the probe cannot be a class check).  Both
    must stamp the family's model type, or a polar checkpoint would be written
    as an MH-1 file and never detected as the family's.
    """

    import torch

    from symmetrix.extract_mace_data import extract_mace_data
    from symmetrix.extract_mace_polar import is_polar_mace_model

    model = _build_polar_shaped_model()
    assert is_polar_mace_model(model)
    checkpoint = tmp_path / "MACE-POLAR-1-synthetic.model"
    torch.save(model, checkpoint)

    assert extract_mace_data(checkpoint)["model_type"] == POLAR_MODEL_TYPE


def test_the_calculator_threads_the_graph_scalar_channels(polar_model_path):
    """Q, S, the fermi level, and the field reach the family's input surface.

    The channels are graph-level program inputs, so they are resolved per
    evaluation and handed to the declared surface before the graph is
    prepared -- the same lifecycle the MACEField field argument travels.
    """

    explicit = _polar_calculator(
        polar_model_path,
        total_charge=-1.0,
        total_spin=0.5,
        fermi_level=-2.0,
        electric_field=[0.0, 0.0, 0.01],
    )
    atoms = _molecule()
    atoms.info.update(total_charge=1.0, total_spin=1.0, fermi_level=-1.0)
    atoms.calc = explicit
    atoms.get_potential_energy()
    assert {
        key: explicit._polar_graph_inputs[key]
        for key in ("total_charge", "total_spin", "fermi_level")
    } == {
        "total_charge": -1.0,
        "total_spin": 0.5,
        "fermi_level": -2.0,
    }
    np.testing.assert_allclose(
        explicit._polar_graph_inputs["electric_field"], [0.0, 0.0, 0.01]
    )

    from_info = _polar_calculator(polar_model_path)
    atoms = _molecule()
    atoms.info.update(
        total_charge=-2.0,
        total_spin=1.0,
        fermi_level=-3.0,
        electric_field=[0.0, 0.0, -0.02],
    )
    atoms.calc = from_info
    atoms.get_potential_energy()
    assert {
        key: from_info._polar_graph_inputs[key]
        for key in ("total_charge", "total_spin", "fermi_level")
    } == {
        "total_charge": -2.0,
        "total_spin": 1.0,
        "fermi_level": -3.0,
    }
    np.testing.assert_allclose(
        from_info._polar_graph_inputs["electric_field"], [0.0, 0.0, -0.02]
    )

    neutral = _polar_calculator(polar_model_path)
    atoms = _molecule()
    atoms.calc = neutral
    atoms.get_potential_energy()
    assert {
        key: neutral._polar_graph_inputs[key]
        for key in ("total_charge", "total_spin", "fermi_level")
    } == {
        "total_charge": 0.0,
        "total_spin": 0.0,
        "fermi_level": 0.0,
    }
    np.testing.assert_array_equal(
        neutral._polar_graph_inputs["electric_field"], np.zeros(3)
    )


def test_a_changed_graph_scalar_re_evaluates_the_structure(polar_model_path):
    """A changed channel is a changed evaluation, and the setter drops the cache."""

    calculator = _polar_calculator(polar_model_path)
    atoms = _molecule()
    atoms.info["total_charge"] = -1.0
    atoms.calc = calculator
    atoms.get_potential_energy()
    assert calculator.check_state(atoms) == []

    atoms.info["total_charge"] = -2.0
    assert calculator.check_state(atoms) == ["info"]
    atoms.get_potential_energy()
    assert calculator._polar_graph_inputs["total_charge"] == -2.0

    calculator.total_spin = 0.75
    assert calculator.results == {}
    atoms.get_potential_energy()
    assert calculator._polar_graph_inputs["total_spin"] == 0.75


def test_the_dipole_output_follows_the_program_and_composed_result(
    polar_model_path,
):
    """The program's dipole node is surfaced from the composed evaluation.

    The dipole is the polar program's ``dipole_readout`` node, so the frontend
    advertises it from the program rather than from a family list.  The public
    value must be the same owned result as the opt-in composed diagnostic, not
    a second frontend implementation of the readout.
    """

    calculator = _polar_calculator(polar_model_path)
    assert "dipole" in calculator.implemented_properties

    atoms = _molecule()
    atoms.calc = calculator
    dipole = np.asarray(atoms.get_dipole_moment(), dtype=np.float64)
    diagnostic = calculator._materialize_polar_diagnostics()
    np.testing.assert_allclose(
        dipole,
        np.asarray(diagnostic["dipole"], dtype=np.float64).reshape(3),
        rtol=0.0,
        atol=0.0,
    )


def test_the_polar_surface_is_the_family_s_own():
    """No other family declares or silently accepts the Polar-only channel."""

    declaring = [
        entry.key
        for entry in family_registry.declared_entries()
        if family_registry.surface_of(entry, "polar_graph_inputs") is not None
    ]
    assert declaring == ["MACE_Polar"]

    compact_mace_model = (
        Path(__file__).resolve().parent / "data" / "synthetic_one_interaction_mace.json"
    )
    with pytest.raises(ValueError, match="Polar-only MACE_Polar"):
        Symmetrix(
            compact_mace_model,
            dtype="float64",
            streamed_edges="reference",
            total_charge=1.0,
            total_spin=1.0,
            fermi_level=-1.0,
        )


def test_native_polar_apis_belong_only_to_the_polar_evaluator():
    """Keep family semantics off the reusable MH-1 backbone surface."""

    polar_names = (
        "_set_polar_long_range_contract",
        "has_polar_long_range_contract",
        "polar_schedule",
        "set_polar_graph_inputs",
        "polar_graph_inputs",
        "polar_delta_ready",
        "polar_delta_native_program",
        "_evaluate_polar_prepared",
        "_collect_polar_public_outputs",
    )
    neutral_backbone_names = (
        "layer_features",
        "clear_layer_product_seed",
    )

    for ordinary_name, polar_name in (
        ("MACENonlinearKokkos", "MACEPolarKokkos"),
        ("MACENonlinearKokkosFloat", "MACEPolarKokkosFloat"),
    ):
        ordinary = getattr(_native_module(), ordinary_name)
        polar = getattr(_native_module(), polar_name)

        leaked_names = [name for name in dir(ordinary) if "polar" in name.lower()]
        assert (
            leaked_names == []
        ), f"{ordinary_name} leaks Polar-only APIs: {leaked_names}"

        for name in polar_names:
            assert hasattr(polar, name), f"{polar_name} is missing {name}"

        for name in neutral_backbone_names:
            assert hasattr(ordinary, name), f"{ordinary_name} is missing {name}"
            assert hasattr(polar, name), f"{polar_name} is missing {name}"


@pytest.mark.parametrize(
    ("dtype", "value_atol"),
    [("float64", 2e-4), ("float32", 6e-3)],
)
def test_direct_loads_generated_backbone_and_default_resolves_to_reference(
    polar_model_path, monkeypatch, tmp_path, dtype, value_atol
):
    """Direct generates the local backbone; reference stays compiler-free.

    The polar whole-program identity stays distinct from its generated MH-1
    subprogram. The family still has no streamed-edge handshake, so the unset
    default and ``auto`` resolve to reference, while an explicit direct request
    selects the declared generated routing without handing a mode to native.
    """

    unset = Symmetrix(
        str(polar_model_path),
        dtype=dtype,
        polar_electrostatics="direct_reciprocal",
    )
    assert unset.streamed_edges == "reference"
    assert unset.streamed_edges_requested == "direct"
    assert unset._native_streamed_edges == "materialized"
    assert unset.jit_status == "not_applicable"
    assert (
        "the unset default selected reference because the family declares no "
        "streamed-edge mode handshake"
    ) == unset.streamed_edges_resolution_reason
    assert unset.evaluator.uses_mh1_fast_path
    assert not unset.evaluator.jit_mh1_host_plugin_ready

    reference = Symmetrix(
        str(polar_model_path),
        dtype=dtype,
        streamed_edges="reference",
        polar_electrostatics="direct_reciprocal",
    )
    assert reference.streamed_edges == "reference"
    assert reference._native_streamed_edges == "materialized"
    assert reference.jit_status == "not_applicable"
    assert reference.evaluator.uses_mh1_fast_path
    assert not reference.evaluator.jit_mh1_host_plugin_ready

    direct = Symmetrix(
        str(polar_model_path),
        dtype=dtype,
        streamed_edges="direct",
        polar_electrostatics="direct_reciprocal",
    )
    assert direct.streamed_edges == "direct"
    assert direct._native_streamed_edges == "direct"
    assert direct.jit_status in ("built", "cached")
    assert direct.evaluator.uses_mh1_fast_path
    if _native_module()._kokkos_default_execution_space() == "Cuda":
        assert direct.evaluator.jit_mh1_cuda_plugin_v5_ready
    else:
        assert direct.evaluator.jit_mh1_host_plugin_v7_ready
    properties = ["energy", "forces", "stress"]
    atoms = _molecule()
    reference.calculate(atoms.copy(), properties=properties)
    direct.calculate(atoms.copy(), properties=properties)
    for name in properties:
        np.testing.assert_allclose(
            direct.results[name],
            reference.results[name],
            rtol=0.0,
            atol=value_atol,
            err_msg=f"generated and portable polar {name} disagree",
        )
    assert "shared_products" in direct.evaluator.execution_mh1_execution_backend
    assert all(
        direct.evaluator._execution_mh1_product_module_forward_launch_count(layer) > 0
        for layer in range(2)
    )
    assert all(
        direct.evaluator._execution_mh1_product_module_reverse_launch_count(layer) > 0
        for layer in range(2)
    )
    assert direct.evaluator.factorized_fallback_evaluation_count == 0

    # Reuse the exact graph/coordinates, then invalidate positions, cell, and
    # topology on the same pair-spline evaluator. Each request must stay on the
    # generated path and agree with an independently cached portable route.
    variants = [atoms.copy()]
    moved = atoms.copy()
    moved.positions[0, 2] += 0.04
    variants.append(moved)
    strained = atoms.copy()
    strained.set_cell(
        np.asarray(atoms.cell) @ np.diag([1.02, 0.99, 1.01]), scale_atoms=True
    )
    variants.append(strained)
    from ase import Atom

    enlarged = atoms.copy()
    enlarged.append(Atom("H", position=(1.2, 1.0, 0.8)))
    variants.append(enlarged)
    product_forward_before = [
        direct.evaluator._execution_mh1_product_module_forward_launch_count(layer)
        for layer in range(2)
    ]
    product_reverse_before = [
        direct.evaluator._execution_mh1_product_module_reverse_launch_count(layer)
        for layer in range(2)
    ]
    for changed in variants:
        reference.calculate(changed.copy(), properties=properties)
        direct.calculate(changed.copy(), properties=properties)
        for name in properties:
            np.testing.assert_allclose(
                direct.results[name],
                reference.results[name],
                rtol=0.0,
                atol=value_atol,
            )
    assert all(
        direct.evaluator._execution_mh1_product_module_forward_launch_count(layer)
        > product_forward_before[layer]
        for layer in range(2)
    )
    assert all(
        direct.evaluator._execution_mh1_product_module_reverse_launch_count(layer)
        > product_reverse_before[layer]
        for layer in range(2)
    )
    assert direct.evaluator.factorized_fallback_evaluation_count == 0

    auto = Symmetrix(str(polar_model_path), dtype=dtype, streamed_edges="auto")
    assert auto.streamed_edges == "reference"
    assert auto._native_streamed_edges == "materialized"
    assert auto.jit_status == "not_applicable"
    assert auto.evaluator.uses_mh1_fast_path
    assert not auto.evaluator.jit_mh1_host_plugin_ready
    assert "no streamed-edge mode handshake" in auto.streamed_edges_resolution_reason
    assert auto.streamed_edges_resolution_reason != (
        unset.streamed_edges_resolution_reason
    )


def test_the_channel_lanes_are_the_programs_declared_inputs(polar_model_path):
    """The graph-input channels map onto declared program buffers.

    The mapping is the family's declaration, not a convention: total charge and
    total spin are the per-graph targets the Fukui equilibration node consumes,
    and the external potential is the four-lane vector the field-projection
    node consumes.  Pinning the declared buffer names here is what keeps a
    channel from being silently dropped when the program, the node contracts,
    or the binding are edited.

    The lane composition is the oracle's (``mace/modules/extensions.py:643``):
    ``external_potential = hstack((zeros_like(fermi_level), external_field))``
    -- columns 1..3 are the electric field and column 0 is a zero V lane, so
    the fermi level enters the graph as a channel the forward does not couple.
    """

    calculator = _polar_calculator(polar_model_path)
    nodes = _program_nodes(calculator.evaluator.family_program)

    fukui = nodes["polar.long_range.fukui_normalize"]
    consumers = {buffer["name"] for buffer in fukui["inputs"]}
    assert {"total_charge", "total_spin"} <= consumers
    assert {"fukui_sources", "charge_l0_alpha", "charge_l0_beta"} <= consumers

    projection = nodes["polar.long_range.external_field_projection"]
    assert "external_potential" in {buffer["name"] for buffer in projection["inputs"]}

    # The binding's channel is the four-lane vector; its layout is contract
    # data of the projection node, so a lane cannot change order unnoticed.
    potential = next(
        buffer
        for buffer in projection["inputs"]
        if buffer["name"] == "external_potential"
    )
    assert potential["shape"] == [0, 4]
    assert potential["dtype"] == "float64"

    # The channels the calculator accepts are the ones declared above: the
    # scalars are the equilibration targets and the field's three components
    # fill the lanes after the zero V lane.  Whether a set channel *changes*
    # the evaluation is the chain-execution gate (the equilibration and
    # projection nodes must run for it to), not a property of the declaration.
    assert family_registry.supports_surface(
        FAMILY_ENTRY, calculator.evaluator, "polar_graph_inputs"
    )


@pytest.mark.parametrize(
    ("variant", "dtype", "step", "force_atol", "stress_rtol", "stress_atol"),
    (
        pytest.param("plain", "float64", 1e-5, 1e-5, 1e-5, 1e-9, id="float64-plain"),
        pytest.param("bias", "float64", 1e-5, 1e-5, 1e-5, 1e-9, id="float64-bias"),
        pytest.param(
            "plain", "float32", 2.5e-3, 3.5e-2, 2e-2, 1.5e-5, id="float32-plain"
        ),
        pytest.param("bias", "float32", 3e-3, 3.2e-1, 2e-2, 2.5e-4, id="float32-bias"),
    ),
)
def test_both_gradients_are_served_and_are_the_family_energys_own(
    polar_models, variant, dtype, step, force_atol, stress_rtol, stress_atol
):
    """The family's energy, forces, and stress are one evaluation's outputs.

    The composition is landed, so the gradient is served rather than refused:
    the chain's per-layer seeds go back into the backbone's reverse (the cross
    term positions own through the retained products) and the composition's own
    position seeds and cell adjoints are added to it.  This test is the
    surface check on the small synthetic model -- both gradients are served and
    finite -- plus the self-consistency that matters most at that surface: the
    served force is the central difference of the served energy, along one
    representative atom direction.  The oracle-referenced gates, the full matrix, and
    the finite-strain stress gate live in
    ``test_polar_end_to_end_fixtures.py``.
    """

    import numpy as np

    atoms = _molecule()
    atoms.calc = _polar_calculator(polar_models[variant], dtype=dtype)
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    assert forces.shape == (len(atoms), 3)
    assert np.all(np.isfinite(forces))
    stress = np.asarray(atoms.get_stress(), dtype=np.float64)
    assert stress.shape == (6,)
    assert np.all(np.isfinite(stress))

    positions = atoms.get_positions()
    finite_difference_calculator = _polar_calculator(polar_models[variant], dtype=dtype)
    # One atom direction is enough for this frontend surface gate: it checks
    # self-consistency (the served force against the served energy's own
    # difference), while the oracle-referenced matrix and the full directional
    # finite-strain stress gate live in ``test_polar_end_to_end_fixtures.py``.
    # The fp64 step is small because the difference's truncation is O(step^2)
    # on a total energy of order a hundred eV: measured 4.5e-5 eV/A at 1e-4
    # against 4.5e-7 at 1e-5 on the bias variant. A recorded CPU/CUDA FP32
    # sweep from 1e-1 through 1.5e-3 places the plain-readout
    # truncation/quantization knee at 2.5e-3.  The bias readout amplifies
    # total-energy cancellation; its post-merge sweep has a lower error at
    # 3e-3, so it keeps the same force bound at that step.  The fp64 gate
    # remains unchanged and strict.
    for component in (0,):
        displaced = positions.copy()
        displaced[0, component] += step
        plus = _displaced_energy(
            polar_models[variant],
            displaced,
            None,
            dtype,
            calculator=finite_difference_calculator,
        )
        displaced[0, component] -= 2.0 * step
        minus = _displaced_energy(
            polar_models[variant],
            displaced,
            None,
            dtype,
            calculator=finite_difference_calculator,
        )
        numeric = -(plus - minus) / (2.0 * step)
        assert abs(forces[0, component] - numeric) <= force_atol, (
            variant,
            component,
            forces[0, component],
            numeric,
            energy,
        )

    # The same self-consistency for the stress.  The synthetic models are the
    # only place this is gated with the contract's self-interaction flag *off*
    # (every released checkpoint has it on), so the flag's adjoint -- both
    # halves of the quadratic subtraction -- is exercised here and nowhere
    # else.  Measured agreement at this step: 2.8e-8 relative on the largest
    # component, 7.0e-9 on a shear.
    from ase.stress import voigt_6_to_full_3x3_stress

    cell = np.asarray(atoms.get_cell(), dtype=np.float64)
    volume = abs(float(np.linalg.det(cell)))
    # One normal and one shear distinguish the volume and engineering-shear
    # conventions; the full six-component matrix is covered by the dedicated
    # end-to-end strain tests.
    voigt_indices = (0, 3)
    numerical = []
    for index in voigt_indices:
        strain = np.zeros(6)
        strain[index] = step
        tensor = voigt_6_to_full_3x3_stress(strain)
        plus = np.eye(3) + tensor
        minus = np.eye(3) - tensor
        forward = _displaced_energy(
            polar_models[variant],
            positions @ plus,
            cell @ plus,
            dtype,
            calculator=finite_difference_calculator,
        )
        backward = _displaced_energy(
            polar_models[variant],
            positions @ minus,
            cell @ minus,
            dtype,
            calculator=finite_difference_calculator,
        )
        numerical.append((forward - backward) / (2.0 * step) / volume)
    numerical = np.asarray(numerical, dtype=np.float64)
    # The finite difference's shear components carry the engineering factor of
    # two; the tensor's do not.
    numerical[1] *= 0.5
    np.testing.assert_allclose(
        stress[[0, 3]],
        numerical,
        rtol=stress_rtol,
        atol=stress_atol,
        err_msg=f"{variant}/{dtype}: the served stress must be the served "
        "energy's own strain derivative, with the self-interaction flag off",
    )


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_polar_stress_diagnostics_are_explicit_and_preserve_public_results(
    polar_model_path, dtype
):
    """Diagnostic stress is opt-in and cannot mutate the packed E/F/stress.

    Removing the explicit diagnostic entry point loses the pre-existing
    ``polar_stress`` relation surface. Materializing it during normal EFS would
    instead violate the production boundary, while returning evaluator-owned
    storage would let the next request mutate this snapshot.
    """

    atoms = _molecule()
    calculator = _polar_calculator(polar_model_path, dtype=dtype)
    calculator.calculate(atoms, properties=["energy", "forces", "stress"])
    public_before = {
        name: np.array(calculator.results[name], copy=True)
        for name in ("energy", "forces", "stress")
    }
    assert calculator._polar_composition is None
    assert "polar" not in calculator.results

    diagnostic = calculator._materialize_polar_diagnostics()
    assert diagnostic is calculator._polar_composition
    polar_stress = np.asarray(diagnostic["polar_stress"], dtype=np.float64)
    assert polar_stress.shape == (1, 3, 3)
    assert np.all(np.isfinite(polar_stress))
    for name, expected in public_before.items():
        np.testing.assert_array_equal(calculator.results[name], expected)
    assert "polar" not in calculator.results

    snapshot = polar_stress.copy()
    moved = atoms.copy()
    moved.positions[0, 0] += 0.02
    calculator.calculate(moved, properties=["energy", "forces", "stress"])
    np.testing.assert_array_equal(polar_stress, snapshot)


@pytest.mark.parametrize(("dtype", "atol"), (("float32", 3e-5), ("float64", 2e-11)))
@pytest.mark.parametrize(
    ("include_forces", "include_stress"),
    ((False, False), (True, False), (False, True), (True, True)),
)
def test_native_packed_outputs_match_granular_reference_and_own_their_lifetime(
    polar_model_path, dtype, atol, include_forces, include_stress
):
    """The packed native boundary is the prior granular E/F/stress algebra.

    A missing backbone or polar term, a wrong Voigt shear, or an array that
    aliases the evaluator's reusable pack makes this fail independently in
    both native precisions.
    """

    calculator = _polar_calculator(polar_model_path, dtype=dtype)
    atoms = _molecule()
    mace_inputs = calculator._mace_inputs(atoms)
    calculator._compute_mace(mace_inputs, atoms=atoms)
    diagnostic = calculator._compose_polar(
        atoms,
        mace_inputs,
        compute_reverse=True,
        materialize_diagnostics=True,
    )
    try:
        generation = calculator._compute_mace(mace_inputs, atoms=atoms)
        granular = calculator._collect_mace_results(
            atoms,
            mace_inputs,
            ["energy", "forces", "stress"],
            generation,
        )
        packed = calculator.evaluator._collect_polar_public_outputs(
            volume=float(atoms.get_volume()),
            graph_generation=generation,
            include_forces=include_forces,
            include_stress=include_stress,
        )
    finally:
        calculator.evaluator.clear_layer_product_seed()

    from ase.stress import full_3x3_to_voigt_6_stress

    assert packed["energy"] == pytest.approx(
        granular["energy"] + diagnostic["energy"], abs=atol, rel=0.0
    )
    assert ("forces" in packed) is include_forces
    assert ("stress" in packed) is include_stress
    if include_forces:
        np.testing.assert_allclose(
            packed["forces"],
            granular["forces"] - diagnostic["positions_adjoint"],
            rtol=0.0,
            atol=atol,
        )
    if include_stress:
        np.testing.assert_allclose(
            np.asarray(packed["stress"]).reshape(6),
            granular["stress"] + full_3x3_to_voigt_6_stress(diagnostic["stress"][0]),
            rtol=0.0,
            atol=atol,
        )

    snapshot = {name: np.array(value, copy=True) for name, value in packed.items()}
    moved = atoms.copy()
    moved.positions[0, 1] += 0.03
    calculator.calculate(moved, properties=["energy", "forces", "stress"])
    for name, expected in snapshot.items():
        np.testing.assert_array_equal(packed[name], expected)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_reused_polar_calculator_invalidates_position_cell_and_topology_caches(
    polar_model_path, dtype
):
    """Exact reuse and every invalidation route agree with a fresh evaluator."""

    reused = _polar_calculator(polar_model_path, dtype=dtype)
    base = _molecule()

    def evaluate(calculator, atoms, properties):
        calculator.calculate(atoms.copy(), properties=properties)
        return {
            name: np.array(calculator.results[name], copy=True) for name in properties
        }

    evaluate(reused, base, ["energy"])
    baseline = evaluate(reused, base, ["energy", "forces", "stress"])
    exact = evaluate(reused, base, ["energy", "forces", "stress"])
    reuse_atol = 3e-5 if dtype == "float32" else 5e-13
    for name in baseline:
        np.testing.assert_allclose(
            exact[name], baseline[name], rtol=0.0, atol=reuse_atol
        )

    variants = []
    moved = base.copy()
    moved.positions[0, 2] += 0.04
    variants.append(moved)
    strained = base.copy()
    strained.set_cell(
        np.asarray(base.cell) @ np.diag([1.03, 0.98, 1.01]), scale_atoms=True
    )
    variants.append(strained)
    enlarged = base.copy()
    from ase import Atom

    enlarged.append(Atom("H", position=(1.1, 1.4, 0.7)))
    variants.append(enlarged)
    for changed in variants:
        actual = evaluate(reused, changed, ["energy", "forces", "stress"])
        fresh = evaluate(
            _polar_calculator(polar_model_path, dtype=dtype),
            changed,
            ["energy", "forces", "stress"],
        )
        for name in actual:
            np.testing.assert_allclose(actual[name], fresh[name], rtol=0.0, atol=3e-5)


@pytest.mark.parametrize(("dtype", "atol"), (("float32", 3e-5), ("float64", 2e-11)))
def test_direct_native_packed_outputs_own_their_lifetime(
    polar_model_path, monkeypatch, tmp_path, dtype, atol
):
    """Direct packed E/F/stress matches granular algebra and owns its arrays.

    Constructing this test through the reference helper, losing a composition
    term, or returning reusable evaluator storage must make the test fail.
    """

    calculator = Symmetrix(
        str(polar_model_path),
        dtype=dtype,
        streamed_edges="direct",
        polar_electrostatics="direct_reciprocal",
    )
    assert calculator.streamed_edges == "direct"
    assert calculator._native_streamed_edges == "direct"

    atoms = _molecule()
    mace_inputs = calculator._mace_inputs(atoms)
    calculator._compute_mace(mace_inputs, atoms=atoms)
    diagnostic = calculator._compose_polar(
        atoms,
        mace_inputs,
        compute_reverse=True,
        materialize_diagnostics=True,
    )
    try:
        generation = calculator._compute_mace(mace_inputs, atoms=atoms)
        granular = calculator._collect_mace_results(
            atoms,
            mace_inputs,
            ["energy", "forces", "stress"],
            generation,
        )
        packed = calculator.evaluator._collect_polar_public_outputs(
            volume=float(atoms.get_volume()),
            graph_generation=generation,
            include_forces=True,
            include_stress=True,
        )
    finally:
        calculator.evaluator.clear_layer_product_seed()

    from ase.stress import full_3x3_to_voigt_6_stress

    assert packed["energy"] == pytest.approx(
        granular["energy"] + diagnostic["energy"], abs=atol, rel=0.0
    )
    np.testing.assert_allclose(
        packed["forces"],
        granular["forces"] - diagnostic["positions_adjoint"],
        rtol=0.0,
        atol=atol,
    )
    np.testing.assert_allclose(
        np.asarray(packed["stress"]).reshape(6),
        granular["stress"] + full_3x3_to_voigt_6_stress(diagnostic["stress"][0]),
        rtol=0.0,
        atol=atol,
    )
    assert all(
        calculator.evaluator._execution_mh1_product_module_forward_launch_count(layer)
        > 0
        for layer in range(2)
    )
    assert all(
        calculator.evaluator._execution_mh1_product_module_reverse_launch_count(layer)
        > 0
        for layer in range(2)
    )
    assert calculator.evaluator.factorized_fallback_evaluation_count == 0

    snapshot = {name: np.array(value, copy=True) for name, value in packed.items()}
    moved = atoms.copy()
    moved.positions[0, 1] += 0.03
    calculator.calculate(moved, properties=["energy", "forces", "stress"])
    for name, expected in snapshot.items():
        np.testing.assert_array_equal(packed[name], expected)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_direct_alternation_invalidates_position_cell_and_topology_caches(
    polar_model_path, monkeypatch, tmp_path, dtype
):
    """Direct energy-only/EFS alternation invalidates every graph cache key."""

    direct = Symmetrix(
        str(polar_model_path),
        dtype=dtype,
        streamed_edges="direct",
        polar_electrostatics="direct_reciprocal",
    )
    assert direct.streamed_edges == "direct"
    assert direct._native_streamed_edges == "direct"

    def evaluate(calculator, atoms, properties):
        calculator.calculate(atoms.copy(), properties=properties)
        return {
            name: np.array(calculator.results[name], copy=True) for name in properties
        }

    base = _molecule()
    evaluate(direct, base, ["energy"])
    assert all(
        direct.evaluator._execution_mh1_product_module_forward_launch_count(layer) > 0
        for layer in range(2)
    )
    reverse_after_energy = [
        direct.evaluator._execution_mh1_product_module_reverse_launch_count(layer)
        for layer in range(2)
    ]
    baseline = evaluate(direct, base, ["energy", "forces", "stress"])
    assert all(
        direct.evaluator._execution_mh1_product_module_reverse_launch_count(layer)
        > reverse_after_energy[layer]
        for layer in range(2)
    )
    warmed_output_replacements = (
        direct.evaluator.polar_public_outputs_storage_replacement_count
    )
    assert warmed_output_replacements > 0
    evaluate(direct, base, ["energy"])
    evaluate(direct, base, ["energy", "forces", "stress"])
    assert (
        direct.evaluator.polar_public_outputs_storage_replacement_count
        == warmed_output_replacements
    )
    assert all(
        direct.evaluator._execution_mh1_product_module_reverse_launch_count(layer)
        > reverse_after_energy[layer]
        for layer in range(2)
    )
    exact = evaluate(direct, base, ["energy", "forces", "stress"])
    reuse_atol = 3e-5 if dtype == "float32" else 5e-13
    for name in baseline:
        np.testing.assert_allclose(
            exact[name], baseline[name], rtol=0.0, atol=reuse_atol
        )

    moved = base.copy()
    moved.positions[0, 2] += 0.04
    strained = base.copy()
    strained.set_cell(
        np.asarray(base.cell) @ np.diag([1.03, 0.98, 1.01]), scale_atoms=True
    )
    from ase import Atom

    enlarged = base.copy()
    enlarged.append(Atom("H", position=(1.1, 1.4, 0.7)))
    route_atol = 5e-3 if dtype == "float32" else 2e-4
    reference = _polar_calculator(polar_model_path, dtype=dtype)
    for changed in (moved, strained, enlarged):
        actual = evaluate(direct, changed, ["energy", "forces", "stress"])
        expected = evaluate(reference, changed, ["energy", "forces", "stress"])
        for name in actual:
            comparison_atol = (
                6.1e-3 if dtype == "float32" and name == "forces" else route_atol
            )
            np.testing.assert_allclose(
                actual[name], expected[name], rtol=0.0, atol=comparison_atol
            )

    assert all(
        direct.evaluator._execution_mh1_product_module_forward_launch_count(layer) > 0
        for layer in range(2)
    )
    assert all(
        direct.evaluator._execution_mh1_product_module_reverse_launch_count(layer) > 0
        for layer in range(2)
    )
    assert direct.evaluator.factorized_fallback_evaluation_count == 0


def _displaced_energy(model_path, positions, cell, dtype="float64", *, calculator=None):
    """The family's energy on a displaced (and optionally strained) molecule."""

    atoms = _molecule()
    atoms.set_positions(positions)
    if cell is not None:
        atoms.set_cell(cell, scale_atoms=False)
    atoms.calc = calculator or _polar_calculator(model_path, dtype=dtype)
    return float(atoms.get_potential_energy())


def _synthetic_chain_inputs(evaluator):
    """Small deterministic inputs for the public composed-chain diagnostics."""

    shapes = evaluator.polar_delta_shapes()
    nodes = len(POSITIONS)
    feature_width = shapes["interaction_count"] * shapes["node_feature_dimension"]
    node_features = np.linspace(-0.2, 0.3, nodes * feature_width).reshape(
        nodes, feature_width
    )
    node_attrs = np.zeros((nodes, shapes["node_attr_dimension"]))
    node_attrs[np.arange(nodes), (0, 1, 0)] = 1.0
    return {
        "node_features": np.ascontiguousarray(node_features),
        "node_attrs": np.ascontiguousarray(node_attrs),
        "positions": np.ascontiguousarray(POSITIONS, dtype=np.float64),
        "cell": np.ascontiguousarray(np.eye(3)[None, :, :] * WATER_BOX),
        "graph_pbc": np.zeros((1, 3), dtype=np.int32),
        "node_graph_index": np.zeros(nodes, dtype=np.int32),
        "total_charge": [0.0],
        "total_spin": [0.0],
        "external_potential": np.zeros((1, 4), dtype=np.float64),
    }


@pytest.mark.parametrize(
    "entry_point", ("polar_delta_chain", "polar_delta_composed_forward")
)
@pytest.mark.parametrize(
    "route", ("auto", "realspace", "periodic", "direct_reciprocal", "pme")
)
def test_public_granular_polar_entry_points_reject_multiple_systems_early(
    polar_model_path, entry_point, route
):
    """Every public granular route rejects before geometry/cache allocation."""

    evaluator = _polar_calculator(polar_model_path).evaluator
    inputs = _synthetic_chain_inputs(evaluator)
    inputs.update(
        cell=np.ascontiguousarray(np.repeat(inputs["cell"], 2, axis=0)),
        graph_pbc=np.asarray(((0, 0, 0), (1, 1, 1)), dtype=np.int32),
        node_graph_index=np.asarray((0, 1, 1), dtype=np.int32),
        total_charge=[0.0, 0.0],
        total_spin=[0.0, 0.0],
        external_potential=np.zeros((2, 4), dtype=np.float64),
        polar_electrostatics=route,
    )
    before = evaluator.polar_delta_workspace()
    call = getattr(evaluator, entry_point)
    if entry_point == "polar_delta_composed_forward":
        inputs["compute_reverse"] = False

    with pytest.raises(ValueError, match="exactly one input system"):
        call(**inputs)

    after = evaluator.polar_delta_workspace()
    for name in (
        "workspace_replacement_count",
        "arena_view_allocation_count",
        "reciprocal_allocation_bytes",
        "realspace_cache_allocation_bytes",
        "realspace_arena_allocation_bytes",
        "realspace_retained_allocation_bytes",
    ):
        assert after[name] == before[name]


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_potential_adjoint_selects_node_parallel_staged_launch(polar_model_path, dtype):
    """The resident potential reverse must expose node-scaled CUDA work.

    A graph-owned correction kernel collapses a one-graph request to one
    block, irrespective of its node count.  Driving the real composed reverse
    here makes the selected family and its logical launch extent observable;
    the explicit reference family remains available for parity gates.
    """

    evaluator = _polar_calculator(polar_model_path, dtype=dtype).evaluator
    inputs = _synthetic_chain_inputs(evaluator)
    evaluator.polar_delta_composed_forward(**inputs, compute_reverse=True)

    diagnostics = evaluator.polar_potential_adjoint_diagnostics()
    assert diagnostics["selected_family"] == "node_parallel_staged"
    assert diagnostics["reference_family"] == "graph_serial_reference"
    assert diagnostics["selected_count"] == 4
    assert diagnostics["selected_node_extent"] == 3
    assert diagnostics["selected_graph_extent"] == 1
    assert diagnostics["model_extent_bits"] == 31
    assert diagnostics["flattened_index_bits"] == 64
    assert diagnostics["selected_stage_count"] == 7
    assert diagnostics["selected_geometry"] == (
        "range64(nodes),range64(graphs),range64(nodes),"
        "range64(nodes),range64(graphs),range64(nodes),range64(graphs)"
    )
    assert tuple(diagnostics["selected_kernels"]) == (
        "polar chain external adjoint node contributions",
        "polar chain external adjoint graph reduction",
        "polar chain external adjoint node apply",
        "polar chain correction adjoint node contributions",
        "polar chain correction adjoint graph reduction",
        "polar chain correction adjoint node apply",
        "polar chain correction adjoint strain",
    )


def test_potential_adjoint_rejects_model_extents_before_narrowing_or_allocation(
    polar_model_path,
):
    """Preparation validates int-bounded models and widened staging products.

    Removing the public-boundary check would narrow INT_MAX + 1 before the
    chain sees it.  Computing a staging product in int would overflow the
    literal values below.  This diagnostic surface performs validation only;
    it deliberately does not allocate the boundary-sized workspace.
    """

    evaluator = _polar_calculator(polar_model_path).evaluator
    int_max = np.iinfo(np.int32).max
    validated = evaluator.polar_validate_potential_adjoint_extents(int_max, int_max)
    assert validated == {
        "nodes": 2_147_483_647,
        "graphs": 2_147_483_647,
        "external_node_elements": 8_589_934_588,
        "external_graph_elements": 4_294_967_294,
        "correction_node_elements": 12_884_901_882,
        "correction_graph_elements": 12_884_901_882,
        "model_extent_bits": 31,
        "flattened_index_bits": 64,
    }
    with pytest.raises(ValueError, match="node count.*INT_MAX"):
        evaluator.polar_validate_potential_adjoint_extents(int_max + 1, 1)
    with pytest.raises(ValueError, match="graph count.*INT_MAX"):
        evaluator.polar_validate_potential_adjoint_extents(1, int_max + 1)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_potential_adjoint_selected_matches_serial_reference(polar_model_path, dtype):
    """The selected single-system transpose preserves the serial reference."""

    evaluator = _polar_calculator(polar_model_path, dtype=dtype).evaluator
    inputs = _synthetic_chain_inputs(evaluator)
    inputs["external_potential"] = np.array([[0.2, 0.3, -0.4, 0.5]], dtype=np.float64)
    state = evaluator.polar_delta_chain(**inputs)
    shapes = evaluator.polar_delta_shapes()
    potential_seeds = [
        np.linspace(0.25, 1.0, np.asarray(features).size).reshape(
            np.asarray(features).shape
        )
        for features in state["potential_features"]
    ]
    arguments = dict(
        **inputs,
        local_electron_energy_seed=np.zeros((0, 1)),
        spin_charge_density_seed=np.zeros((0, shapes["charges_dimension"])),
        potential_features_seeds=potential_seeds,
        fukui_sources_seeds=[],
    )
    selected = evaluator.polar_delta_chain_reverse(
        **arguments, potential_adjoint_route="selected"
    )
    reference = evaluator.polar_delta_chain_reverse(
        **arguments, potential_adjoint_route="reference"
    )

    tolerance = 1e-4 if dtype == "float32" else 1e-12
    output_names = (
        "node_features",
        "node_attrs",
        "positions",
        "external_potential",
        "total_charge",
        "total_spin",
        "grad_cos_table",
        "grad_sin_table",
        "strain",
    )
    for name in output_names:
        np.testing.assert_allclose(
            selected[name], reference[name], rtol=0.0, atol=tolerance
        )
    repeated = evaluator.polar_delta_chain_reverse(
        **arguments, potential_adjoint_route="selected"
    )
    for name in output_names:
        np.testing.assert_array_equal(repeated[name], selected[name])

    changed_positions = inputs["positions"].copy()
    changed_positions[2, 1] += 0.125
    changed_arguments = {
        **arguments,
        "positions": np.ascontiguousarray(changed_positions),
    }
    changed_reference = evaluator.polar_delta_chain_reverse(
        **changed_arguments, potential_adjoint_route="reference"
    )
    changed_selected = evaluator.polar_delta_chain_reverse(
        **changed_arguments, potential_adjoint_route="selected"
    )
    for name in output_names:
        np.testing.assert_allclose(
            changed_selected[name],
            changed_reference[name],
            rtol=0.0,
            atol=tolerance,
        )
    restored = evaluator.polar_delta_chain_reverse(
        **arguments, potential_adjoint_route="selected"
    )
    for name in output_names:
        np.testing.assert_array_equal(restored[name], selected[name])
    diagnostics = evaluator.polar_potential_adjoint_diagnostics()
    assert diagnostics["selected_count"] > 0
    assert diagnostics["reference_count"] > 0


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_slab_potential_adjoint_matches_reference_and_finite_differences(
    polar_model_path, dtype
):
    """The selected slab correction has serial parity and fresh derivatives."""

    evaluator = _polar_calculator(polar_model_path, dtype=dtype).evaluator
    inputs = _synthetic_chain_inputs(evaluator)
    inputs = {
        **inputs,
        "cell": np.ascontiguousarray(
            np.array([[[8.0, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 20.0]]])
        ),
        "graph_pbc": np.array([[1, 1, 0]], dtype=np.int32),
        "total_charge": [0.4],
        "external_potential": np.array([[0.2, -0.3, 0.4, 0.1]], dtype=np.float64),
    }
    state = evaluator.polar_delta_chain(**inputs)
    shapes = evaluator.polar_delta_shapes()
    potential_seeds = [
        np.linspace(0.2, 0.9, np.asarray(features).size).reshape(
            np.asarray(features).shape
        )
        for features in state["potential_features"]
    ]
    reverse_arguments = dict(
        **inputs,
        local_electron_energy_seed=np.zeros((0, 1)),
        spin_charge_density_seed=np.zeros((0, shapes["charges_dimension"])),
        potential_features_seeds=potential_seeds,
        fukui_sources_seeds=[],
    )
    selected = evaluator.polar_delta_chain_reverse(
        **reverse_arguments, potential_adjoint_route="selected"
    )
    reference = evaluator.polar_delta_chain_reverse(
        **reverse_arguments, potential_adjoint_route="reference"
    )
    tolerance = 1e-4 if dtype == "float32" else 1e-12
    for name in (
        "node_features",
        "positions",
        "external_potential",
        "total_charge",
        "grad_cos_table",
        "grad_sin_table",
        "strain",
    ):
        np.testing.assert_allclose(
            selected[name], reference[name], rtol=0.0, atol=tolerance
        )
    assert np.linalg.norm(selected["external_potential"]) > 0.0

    periodic = _polar_calculator(polar_model_path, dtype=dtype).evaluator
    periodic_state = periodic.polar_delta_chain(
        **{**inputs, "graph_pbc": np.ones((1, 3), dtype=np.int32)}
    )
    correction_delta = sum(
        np.linalg.norm(np.asarray(slab) - np.asarray(bulk))
        for slab, bulk in zip(
            state["potential_features"],
            periodic_state["potential_features"],
            strict=True,
        )
    )
    assert correction_delta > 1e-8

    step = 2e-3 if dtype == "float32" else 1e-5
    fd_tolerance = 3e-2 if dtype == "float32" else 3e-5

    def fresh_loss(changed):
        fresh = _polar_calculator(polar_model_path, dtype=dtype).evaluator
        changed_state = fresh.polar_delta_chain(**changed)
        return sum(
            float(np.sum(seed * np.asarray(features)))
            for seed, features in zip(
                potential_seeds,
                changed_state["potential_features"],
                strict=True,
            )
        )

    plus_positions = inputs["positions"].copy()
    minus_positions = inputs["positions"].copy()
    plus_positions[1, 0] += step
    minus_positions[1, 0] -= step
    numeric_position = (
        fresh_loss({**inputs, "positions": np.ascontiguousarray(plus_positions)})
        - fresh_loss({**inputs, "positions": np.ascontiguousarray(minus_positions)})
    ) / (2.0 * step)
    assert float(np.asarray(selected["positions"])[1, 0]) == pytest.approx(
        numeric_position, abs=fd_tolerance, rel=fd_tolerance
    )

    strain = np.zeros((3, 3))
    strain[0, 0] = step
    plus_transform = np.eye(3) + strain
    minus_transform = np.eye(3) - strain
    plus_inputs = {
        **inputs,
        "positions": np.ascontiguousarray(inputs["positions"] @ plus_transform),
        "cell": np.ascontiguousarray(inputs["cell"] @ plus_transform),
    }
    minus_inputs = {
        **inputs,
        "positions": np.ascontiguousarray(inputs["positions"] @ minus_transform),
        "cell": np.ascontiguousarray(inputs["cell"] @ minus_transform),
    }
    numeric_strain = (fresh_loss(plus_inputs) - fresh_loss(minus_inputs)) / (2.0 * step)
    # ``strain`` is the explicit cell contribution.  The affine transform also
    # moves positions, so its derivative includes the position-seed virial.
    analytic_strain = float(np.asarray(selected["strain"])[0, 0]) + float(
        np.dot(np.asarray(selected["positions"])[:, 0], inputs["positions"][:, 0])
    )
    assert analytic_strain == pytest.approx(
        numeric_strain, abs=fd_tolerance, rel=fd_tolerance
    )


@pytest.mark.parametrize("dtype, precision_bits", [("float32", 32), ("float64", 64)])
def test_polar_workspace_reuses_matching_extents_and_replaces_changed_extents(
    polar_model_path, dtype, precision_bits
):
    """The evaluator owns scratch keyed by graph, k-grid, and precision extents.

    Removing the extent checks, rebuilding the workspace on every call, or
    retaining an undersized workspace after a topology/cell change makes this
    fail through the public native-composition boundary.
    """

    evaluator = _polar_calculator(polar_model_path, dtype=dtype).evaluator
    inputs = _synthetic_chain_inputs(evaluator)

    first = evaluator.polar_delta_composed_forward(**inputs, compute_reverse=True)
    private_state_keys = {
        "spin_charge_density",
        "charge_density",
        "spin_density",
        "pre_loop_density",
        "density_history",
        "potential_features",
        "fukui_sources",
    }
    assert private_state_keys.isdisjoint(first)
    first_workspace = evaluator.polar_delta_workspace()
    second = evaluator.polar_delta_composed_forward(**inputs, compute_reverse=True)
    second_workspace = evaluator.polar_delta_workspace()
    np.testing.assert_allclose(second["energy"], first["energy"])
    assert first_workspace["precision_bits"] == precision_bits
    assert first_workspace["nodes"] == 3
    assert first_workspace["graphs"] == 1
    assert first_workspace["n_k"] > 0
    assert first_workspace["execution_space"]
    assert second_workspace["execution_space"] == first_workspace["execution_space"]
    assert first_workspace["arena_view_allocation_count"] > 0
    assert first_workspace["uvu_workspace_replacement_count"] >= 0
    assert (
        second_workspace["arena_view_allocation_count"]
        == first_workspace["arena_view_allocation_count"]
    )
    assert (
        second_workspace["uvu_workspace_replacement_count"]
        == first_workspace["uvu_workspace_replacement_count"]
    )
    assert (
        second_workspace["workspace_replacement_count"]
        == first_workspace["workspace_replacement_count"]
    )
    assert (
        second_workspace["workspace_reuse_count"]
        > first_workspace["workspace_reuse_count"]
    )
    assert second_workspace["diagnostic_state_materialization_count"] == 0
    assert first_workspace["resident_reverse_count"] > 0
    assert (
        second_workspace["resident_reverse_count"]
        > first_workspace["resident_reverse_count"]
    )

    same_grid_cell = inputs["cell"].copy()
    same_grid_cell[0, 0, 1] += 0.05
    same_grid_result = evaluator.polar_delta_composed_forward(
        **{**inputs, "cell": same_grid_cell}, compute_reverse=True
    )
    same_grid_reference = _polar_calculator(
        polar_model_path, dtype=dtype
    ).evaluator.polar_delta_composed_forward(
        **{**inputs, "cell": same_grid_cell}, compute_reverse=True
    )
    for key in (
        "energy",
        "local_electron_energy",
        "electrostatic_energy_kspace",
        "electrostatic_energy_correction",
        "external_field_energy",
        "dipole",
        "node_features_adjoint",
        "node_attrs_adjoint",
        "positions_adjoint",
        "external_potential_adjoint",
        "total_charge_adjoint",
        "total_spin_adjoint",
        "strain_adjoint",
        "stress",
    ):
        np.testing.assert_allclose(same_grid_result[key], same_grid_reference[key])
    same_grid_workspace = evaluator.polar_delta_workspace()
    assert same_grid_workspace["n_k"] == first_workspace["n_k"]
    assert (
        same_grid_workspace["workspace_replacement_count"]
        == second_workspace["workspace_replacement_count"]
    )
    assert (
        same_grid_workspace["arena_view_allocation_count"]
        == second_workspace["arena_view_allocation_count"]
    )
    assert (
        same_grid_workspace["uvu_workspace_replacement_count"]
        == second_workspace["uvu_workspace_replacement_count"]
    )

    smaller_cell = np.ascontiguousarray(np.eye(3)[None, :, :] * (WATER_BOX / 2))
    evaluator.polar_delta_composed_forward(
        **{**inputs, "cell": smaller_cell}, compute_reverse=True
    )
    cell_workspace = evaluator.polar_delta_workspace()
    assert cell_workspace["n_k"] != first_workspace["n_k"]
    assert (
        cell_workspace["workspace_replacement_count"]
        > second_workspace["workspace_replacement_count"]
    )
    assert (
        cell_workspace["arena_view_allocation_count"]
        > second_workspace["arena_view_allocation_count"]
    )

    expanded = {
        **inputs,
        "node_features": np.ascontiguousarray(
            np.concatenate([inputs["node_features"], inputs["node_features"]])
        ),
        "node_attrs": np.ascontiguousarray(
            np.concatenate([inputs["node_attrs"], inputs["node_attrs"]])
        ),
        "positions": np.ascontiguousarray(
            np.concatenate([inputs["positions"], inputs["positions"] + 0.1])
        ),
        "node_graph_index": np.zeros(6, dtype=np.int32),
    }
    expanded_result = evaluator.polar_delta_composed_forward(
        **expanded, compute_reverse=True
    )
    expanded_workspace = evaluator.polar_delta_workspace()
    assert expanded_workspace["nodes"] == 6
    assert expanded_workspace["graphs"] == 1
    assert (
        expanded_workspace["workspace_replacement_count"]
        > cell_workspace["workspace_replacement_count"]
    )
    assert (
        expanded_workspace["uvu_workspace_replacement_count"]
        >= cell_workspace["uvu_workspace_replacement_count"]
    )
    assert np.isfinite(expanded_result["energy"]).all()

    # Explicit diagnostics may materialize their public state, but they must
    # not corrupt or grow the warmed production workspace.
    before_diagnostic = evaluator.polar_delta_workspace()
    evaluator.polar_delta_chain(**expanded)
    after_diagnostic = evaluator.polar_delta_workspace()
    materializations_per_diagnostic = (
        after_diagnostic["diagnostic_state_materialization_count"]
        - before_diagnostic["diagnostic_state_materialization_count"]
    )
    assert materializations_per_diagnostic > 1
    evaluator.polar_delta_chain(**expanded)
    after_second_diagnostic = evaluator.polar_delta_workspace()
    assert (
        after_second_diagnostic["diagnostic_state_materialization_count"]
        - after_diagnostic["diagnostic_state_materialization_count"]
        == materializations_per_diagnostic
    )
    alternating = evaluator.polar_delta_composed_forward(
        **expanded, compute_reverse=True
    )
    alternating_reference = _polar_calculator(
        polar_model_path, dtype=dtype
    ).evaluator.polar_delta_composed_forward(**expanded, compute_reverse=True)
    alternating_workspace = evaluator.polar_delta_workspace()
    for key, expected in alternating_reference.items():
        np.testing.assert_allclose(alternating[key], expected)
    assert (
        alternating_workspace["arena_view_allocation_count"]
        == after_second_diagnostic["arena_view_allocation_count"]
    )
    assert (
        alternating_workspace["uvu_workspace_replacement_count"]
        == after_second_diagnostic["uvu_workspace_replacement_count"]
    )
    assert (
        alternating_workspace["diagnostic_state_materialization_count"]
        == after_second_diagnostic["diagnostic_state_materialization_count"]
    )
    assert (
        alternating_workspace["resident_reverse_count"]
        > after_second_diagnostic["resident_reverse_count"]
    )

    restored = evaluator.polar_delta_composed_forward(**inputs, compute_reverse=True)
    np.testing.assert_allclose(restored["energy"], first["energy"])


def test_polar_solver_routing_precedes_reciprocal_allocation(polar_model_path):
    """Rejected/real-space routes clear reciprocal state before numerical work."""

    evaluator = _polar_calculator(polar_model_path, dtype="float64").evaluator
    inputs = _synthetic_chain_inputs(evaluator)

    evaluator.polar_delta_composed_forward(
        **inputs,
        polar_electrostatics="direct_reciprocal",
    )
    direct_workspace = evaluator.polar_delta_workspace()
    assert direct_workspace["requested_solver"] == "direct_reciprocal"
    assert direct_workspace["resolved_solver"] == "direct_reciprocal"
    assert direct_workspace["solver_reason"] == "explicit_direct_reciprocal"
    assert direct_workspace["n_k"] > 0
    owner_prefixes = ("cache", "arena", "adjoint")
    for owner in owner_prefixes:
        assert direct_workspace[f"reciprocal_{owner}_allocation_elements"] > 0
        assert direct_workspace[f"reciprocal_{owner}_allocation_bytes"] > 0
    assert direct_workspace["reciprocal_allocation_elements"] == sum(
        direct_workspace[f"reciprocal_{owner}_allocation_elements"]
        for owner in owner_prefixes
    )
    assert direct_workspace["reciprocal_allocation_bytes"] == sum(
        direct_workspace[f"reciprocal_{owner}_allocation_bytes"]
        for owner in owner_prefixes
    )
    # The cache owns both floating-point and integer views. Treating every
    # element as an fp64 value recreates the inaccurate pre-fix metric.
    assert direct_workspace["reciprocal_cache_allocation_bytes"] < (
        direct_workspace["reciprocal_cache_allocation_elements"] * 8
    )
    assert (
        direct_workspace["reciprocal_table_elements"]
        == direct_workspace["reciprocal_allocation_elements"]
    )
    assert (
        direct_workspace["reciprocal_table_bytes"]
        == direct_workspace["reciprocal_allocation_bytes"]
    )

    periodic_inputs = {
        **inputs,
        "graph_pbc": np.asarray([[1, 0, 0]], dtype=np.int32),
    }
    periodic_auto = evaluator.polar_delta_composed_forward(
        **periodic_inputs,
        polar_electrostatics="auto",
    )
    periodic_auto_workspace = evaluator.polar_delta_workspace()
    periodic_compatibility = evaluator.polar_delta_composed_forward(
        **periodic_inputs,
        polar_electrostatics="periodic",
    )
    periodic_compatibility_workspace = evaluator.polar_delta_workspace()
    periodic_direct = evaluator.polar_delta_composed_forward(
        **periodic_inputs,
        polar_electrostatics="direct_reciprocal",
    )
    assert periodic_auto_workspace["resolved_solver"] == "direct_reciprocal"
    assert periodic_auto_workspace["solver_reason"] == "periodic_input_delivery_1"
    assert periodic_compatibility_workspace["solver_reason"] == (
        "periodic_compatibility_override"
    )
    np.testing.assert_allclose(periodic_auto["energy"], periodic_direct["energy"])
    np.testing.assert_allclose(
        periodic_compatibility["energy"], periodic_direct["energy"]
    )

    before_rejection = evaluator.polar_delta_workspace()
    with pytest.raises(ValueError, match="requires fully non-periodic input"):
        evaluator.polar_delta_composed_forward(
            **periodic_inputs,
            polar_electrostatics="realspace",
        )
    rejected_workspace = evaluator.polar_delta_workspace()
    assert rejected_workspace["requested_solver"] == "realspace"
    assert rejected_workspace["resolved_solver"] == "unresolved"
    assert rejected_workspace["solver_reason"] == (
        "realspace_requires_fully_nonperiodic_input"
    )
    assert (
        rejected_workspace["workspace_replacement_count"]
        == (before_rejection["workspace_replacement_count"])
    )
    assert (
        rejected_workspace["arena_view_allocation_count"]
        == (before_rejection["arena_view_allocation_count"])
    )
    assert rejected_workspace["n_k"] == 0
    for owner in (*owner_prefixes, ""):
        infix = f"{owner}_" if owner else ""
        assert rejected_workspace[f"reciprocal_{infix}allocation_elements"] == 0
        assert rejected_workspace[f"reciprocal_{infix}allocation_bytes"] == 0

    automatic_realspace = evaluator.polar_delta_composed_forward(
        **inputs,
        polar_electrostatics="auto",
    )
    realspace_workspace = evaluator.polar_delta_workspace()
    forced_realspace = evaluator.polar_delta_composed_forward(
        **inputs,
        polar_electrostatics="realspace",
    )
    np.testing.assert_allclose(
        automatic_realspace["energy"], forced_realspace["energy"]
    )
    assert np.array_equal(
        automatic_realspace["electrostatic_energy_correction"],
        np.zeros((1, 1)),
    )
    assert realspace_workspace["requested_solver"] == "auto"
    assert realspace_workspace["resolved_solver"] == "realspace"
    assert realspace_workspace["solver_reason"] == "fully_nonperiodic_input"
    assert realspace_workspace["n_k"] == 0
    for owner in (*owner_prefixes, ""):
        infix = f"{owner}_" if owner else ""
        assert realspace_workspace[f"reciprocal_{infix}allocation_elements"] == 0
        assert realspace_workspace[f"reciprocal_{infix}allocation_bytes"] == 0
    assert realspace_workspace["realspace_cache_allocation_bytes"] > 0
    assert realspace_workspace["realspace_retained_allocation_bytes"] > 0

    with pytest.raises(ValueError, match="PBC TTT"):
        evaluator.polar_delta_composed_forward(
            **inputs,
            polar_electrostatics="pme",
        )
    pme_workspace = evaluator.polar_delta_workspace()
    assert pme_workspace["requested_solver"] == "pme"
    assert pme_workspace["resolved_solver"] == "unresolved"
    assert pme_workspace["solver_reason"] == "pme_requires_fully_periodic_input"
    assert pme_workspace["reciprocal_allocation_bytes"] == 0

    fully_periodic_inputs = {
        **inputs,
        "graph_pbc": np.asarray([[1, 1, 1]], dtype=np.int32),
    }
    capability = _native_module()._particle_mesh_capability()
    if (
        capability["available"]
        and _native_module()._kokkos_default_execution_space() == "Cuda"
    ):
        admitted = evaluator.polar_delta_composed_forward(
            **fully_periodic_inputs,
            polar_electrostatics="pme",
        )
        assert np.isfinite(admitted["energy"]).all()
        pme_workspace = evaluator.polar_delta_workspace()
        assert pme_workspace["requested_solver"] == "pme"
        assert pme_workspace["resolved_solver"] == "pme"
        assert pme_workspace["solver_reason"] == "explicit_pme"
        assert pme_workspace["reciprocal_allocation_bytes"] == 0
    else:
        with pytest.raises(
            ValueError,
            match=(
                "no particle-mesh FFT provider|backend has not been qualified|"
                "currently qualified only on CUDA backends"
            ),
        ):
            evaluator.polar_delta_composed_forward(
                **fully_periodic_inputs,
                polar_electrostatics="pme",
            )


@pytest.mark.parametrize("self_enabled", (False, True), ids=("self-off", "self-on"))
@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_realspace_composition_matches_resident_stage_kernels(
    polar_model_path, tmp_path, dtype, self_enabled
):
    """The chain must compose each raw real-space stage exactly once.

    This catches dispatching the feature path through reciprocal projection,
    retaining the molecule-in-box correction, applying feature normalization
    or the half external term twice, and using reciprocal energy for the final
    spin-summed density.
    """

    model_path = polar_model_path
    if self_enabled:
        model = json.loads(polar_model_path.read_text())
        model["polar"]["field_si"] = True
        model["polar"]["include_electrostatic_self_interaction"] = True
        flags = model["execution_contracts"][POLAR_CONTRACT_KEY]["flags"]
        flags["field_si"] = True
        flags["include_electrostatic_self_interaction"] = True
        model_path = _write_model(tmp_path / "self-enabled.json", model)

    evaluator = _polar_calculator(model_path, dtype=dtype).evaluator
    inputs = _synthetic_chain_inputs(evaluator)
    inputs["external_potential"] = np.asarray(
        [[0.13, 0.07, -0.03, 0.02]], dtype=np.float64
    )
    state = evaluator.polar_delta_chain(**inputs, polar_electrostatics="realspace")
    composed = evaluator.polar_delta_composed_forward(
        **inputs,
        compute_reverse=True,
        polar_electrostatics="realspace",
    )

    contract = dict(evaluator.polar_long_range_contract)
    model = json.loads(model_path.read_text())
    polar = model["polar"]
    ops_class = (
        symmetrix.PolarLongRangeOpsFloat
        if dtype == "float32"
        else symmetrix.PolarLongRangeOps
    )
    ops = ops_class(
        kspace_cutoff=float(contract["kspace_cutoff"]),
        density_max_l=int(contract["density_max_l"]),
        density_smearing_width=float(contract["density_smearing_width"]),
        feature_max_l=int(contract["feature_max_l"]),
        feature_smearing_widths=[
            float(value) for value in contract["feature_smearing_widths"]
        ],
    )
    diagnostic_ops = symmetrix.PolarLongRangeOps(
        kspace_cutoff=float(contract["kspace_cutoff"]),
        density_max_l=int(contract["density_max_l"]),
        density_smearing_width=float(contract["density_smearing_width"]),
        feature_max_l=int(contract["feature_max_l"]),
        feature_smearing_widths=[
            float(value) for value in contract["feature_smearing_widths"]
        ],
    )
    cache = ops.realspace_geometry_cache(
        inputs["positions"], inputs["cell"], inputs["node_graph_index"]
    )
    norms = np.asarray(polar["field_feature_norms"]["values"], dtype=np.float64)
    external = np.asarray(
        diagnostic_ops.external_field_features(
            inputs["external_potential"],
            inputs["positions"],
            inputs["node_graph_index"],
            norms,
        )
    )
    density_width = int(evaluator.polar_delta_shapes()["density_width"])
    include_feature_self = bool(polar["field_si"])
    for step, actual in enumerate(state["potential_features"]):
        density = np.asarray(state["density_history"][step])
        expected_channels = []
        for channel in range(2):
            multipoles = np.ascontiguousarray(
                density[
                    :,
                    channel * density_width : (channel + 1) * density_width,
                ]
            )
            raw = np.asarray(
                ops.realspace_features(cache, multipoles, include_feature_self)
            )
            expected_channels.append(raw / norms[None, :] + external)
        expected = np.concatenate(expected_channels, axis=1)
        tolerance = 2.0e-5 if dtype == "float32" else 1.0e-11
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)

    final_density = np.ascontiguousarray(state["charge_density"])
    expected_electrostatic = np.asarray(
        ops.realspace_energy(
            cache,
            final_density,
            bool(contract["include_electrostatic_self_interaction"]),
        )
    )
    dipole = np.asarray(
        diagnostic_ops.dipole_readout(
            final_density, inputs["positions"], inputs["node_graph_index"]
        )["total_dipole"]
    )
    local = np.asarray(state["local_electron_energy"]).sum(axis=0)
    field = np.sum(inputs["external_potential"][:, 1:] * dipole, axis=1)
    expected_total = local + expected_electrostatic + field
    tolerance = 2.0e-5 if dtype == "float32" else 1.0e-10
    np.testing.assert_allclose(
        np.asarray(composed["electrostatic_energy_kspace"]).reshape(-1),
        expected_electrostatic,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        np.asarray(composed["dipole"]), dipole, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        np.asarray(composed["energy"]).reshape(-1),
        expected_total,
        rtol=tolerance,
        atol=tolerance,
    )
    assert np.array_equal(composed["electrostatic_energy_correction"], np.zeros((1, 1)))
    position_adjoint = np.asarray(composed["positions_adjoint"])
    assert position_adjoint.dtype == np.float64
    if dtype == "float32":
        assert np.any(
            position_adjoint != position_adjoint.astype(np.float32).astype(np.float64)
        ), "the public FP64 position adjoint must not be an FP32 round-trip"
    assert np.array_equal(composed["strain_adjoint"], np.zeros((1, 9)))
    workspace = evaluator.polar_delta_workspace()
    assert workspace["realspace_energy_selected_count"] > 0
    assert workspace["realspace_feature_reverse_selected_count"] > 0
    assert workspace["realspace_energy_reverse_selected_count"] > 0


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize("strategy", ("pair_atomic", "materialized_forward"))
def test_composed_realspace_can_force_retained_cuda_strategy(
    polar_model_path, dtype, strategy
):
    evaluator = _polar_calculator(polar_model_path, dtype=dtype).evaluator
    inputs = _synthetic_chain_inputs(evaluator)
    _select_realspace_reverse_strategy(evaluator, "owner")
    owner = evaluator.polar_delta_composed_forward(
        **inputs, compute_reverse=True, polar_electrostatics="realspace"
    )
    owner_outputs = {
        name: np.array(owner[name], copy=True)
        for name in (
            "energy",
            "local_electron_energy",
            "electrostatic_energy_kspace",
            "electrostatic_energy_correction",
            "external_field_energy",
            "dipole",
            "node_features_adjoint",
            "node_attrs_adjoint",
            "positions_adjoint",
            "external_potential_adjoint",
            "total_charge_adjoint",
            "total_spin_adjoint",
            "strain_adjoint",
            "stress",
        )
    }
    assert evaluator.polar_delta_workspace()["realspace_reverse_strategy"] == "owner"

    if strategy == "materialized_forward":
        evaluator._set_polar_realspace_materialized_budget_bytes(1 << 30)
    _select_realspace_reverse_strategy(evaluator, strategy)
    pair = evaluator.polar_delta_composed_forward(
        **inputs, compute_reverse=True, polar_electrostatics="realspace"
    )
    tolerance = 2.0e-4 if dtype == "float32" else 1.0e-8
    for name, expected in owner_outputs.items():
        np.testing.assert_allclose(
            pair[name], expected, rtol=tolerance, atol=tolerance, err_msg=name
        )
    workspace = evaluator.polar_delta_workspace()
    assert workspace["realspace_reverse_strategy"] == strategy
    assert workspace["realspace_atomic_accumulator_bytes"] > 0
    assert workspace["reciprocal_allocation_bytes"] == 0
    if strategy == "pair_atomic":
        _assert_pair_atomic_was_used(evaluator)
    else:
        assert workspace["realspace_materialized_forward_reducer"] == "segmented"
        assert workspace["realspace_feature_pair_atomic_selected_count"] > 0
        assert workspace["realspace_energy_pair_atomic_selected_count"] > 0
        assert workspace["realspace_full_materialized_table_build_count"] == 1
        assert workspace["realspace_full_materialized_table_reuse_count"] > 0
        assert workspace["realspace_full_materialized_retained_bytes"] > 0
        assert workspace["realspace_full_materialized_feature_directed_edges"] > 0


def test_materialized_forward_energy_only_omits_gradient_tables(polar_model_path):
    evaluator = _polar_calculator(polar_model_path, dtype="float64").evaluator
    inputs = _synthetic_chain_inputs(evaluator)
    evaluator._set_polar_realspace_materialized_budget_bytes(1 << 30)
    _select_realspace_reverse_strategy(evaluator, "materialized_forward")
    evaluator.polar_delta_composed_forward(
        **inputs, compute_reverse=False, polar_electrostatics="realspace"
    )
    workspace = evaluator.polar_delta_workspace()
    assert workspace["realspace_reverse_strategy"] == "materialized_forward"
    assert workspace["realspace_full_materialized_table_build_count"] == 1
    assert not workspace["realspace_full_materialized_has_gradients"]
    assert workspace["realspace_feature_reverse_selected_count"] == 0
    assert workspace["realspace_energy_reverse_selected_count"] == 0


def _retired_composed_full_materialized_reducer_selection(
    polar_model_path,
):
    evaluator = _polar_calculator(polar_model_path).evaluator
    before = evaluator.polar_delta_workspace()
    assert before["realspace_full_materialized_reducer"] == "serial"
    assert before["realspace_reverse_strategy_requested"] == "auto"

    evaluator._set_polar_realspace_full_materialized_reducer("atomic_scatter")
    after = evaluator.polar_delta_workspace()
    assert after["realspace_full_materialized_reducer"] == "atomic_scatter"
    assert after["realspace_reverse_strategy_requested"] == "auto"

    with pytest.raises(ValueError, match="full-materialized reducer"):
        evaluator._set_polar_realspace_full_materialized_reducer("invalid")


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_realspace_reverse_auto_selects_backend_policy_for_force_calls(
    polar_model_path, dtype
):
    evaluator = _polar_calculator(polar_model_path, dtype=dtype).evaluator
    inputs = _synthetic_chain_inputs(evaluator)
    before = _assert_realspace_reverse_resolution(
        evaluator, "auto", "unresolved", "reverse_not_requested"
    )
    assert before["realspace_feature_reverse_selected_count"] == 0
    assert before["realspace_energy_reverse_selected_count"] == 0

    evaluator.polar_delta_composed_forward(
        **inputs, compute_reverse=True, polar_electrostatics="realspace"
    )
    execution_space = evaluator.polar_delta_workspace()["execution_space"]
    if execution_space == "Cuda":
        workspace = _assert_realspace_reverse_resolution(
            evaluator, "auto", "pair_atomic", "cuda_qualified_backend"
        )
        assert workspace["realspace_feature_pair_atomic_selected_count"] > 0
        assert workspace["realspace_energy_pair_atomic_selected_count"] > 0
        assert workspace["realspace_feature_owner_selected_count"] == 0
        assert workspace["realspace_energy_owner_selected_count"] == 0
        assert workspace["realspace_atomic_accumulator_bytes"] > 0
    else:
        workspace = _assert_realspace_reverse_resolution(
            evaluator, "auto", "owner", "cpu_or_unqualified_backend"
        )
        assert workspace["realspace_feature_owner_selected_count"] > 0
        assert workspace["realspace_energy_owner_selected_count"] > 0
        assert workspace["realspace_feature_pair_atomic_selected_count"] == 0
        assert workspace["realspace_energy_pair_atomic_selected_count"] == 0
        assert workspace["realspace_atomic_accumulator_bytes"] == 0


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_realspace_energy_only_leaves_reverse_strategy_unresolved(
    polar_model_path, dtype
):
    evaluator = _polar_calculator(polar_model_path, dtype=dtype).evaluator
    inputs = _synthetic_chain_inputs(evaluator)
    evaluator.polar_delta_composed_forward(
        **inputs, compute_reverse=False, polar_electrostatics="realspace"
    )
    workspace = _assert_realspace_reverse_resolution(
        evaluator, "auto", "unresolved", "reverse_not_requested"
    )
    assert workspace["realspace_feature_reverse_selected_count"] == 0
    assert workspace["realspace_energy_reverse_selected_count"] == 0
    assert workspace["realspace_feature_owner_selected_count"] == 0
    assert workspace["realspace_energy_owner_selected_count"] == 0
    assert workspace["realspace_feature_pair_atomic_selected_count"] == 0
    assert workspace["realspace_energy_pair_atomic_selected_count"] == 0
    assert workspace["realspace_atomic_accumulator_bytes"] == 0


@pytest.mark.parametrize("strategy", ("owner", "pair_atomic"))
def test_forced_realspace_reverse_strategy_overrides_backend_policy(
    polar_model_path, strategy
):
    evaluator = _polar_calculator(polar_model_path, dtype="float64").evaluator
    _select_realspace_reverse_strategy(evaluator, strategy)
    evaluator.polar_delta_composed_forward(
        **_synthetic_chain_inputs(evaluator),
        compute_reverse=True,
        polar_electrostatics="realspace",
    )
    workspace = _assert_realspace_reverse_resolution(
        evaluator, strategy, strategy, "forced_override"
    )
    if strategy == "pair_atomic":
        assert workspace["realspace_feature_pair_atomic_selected_count"] > 0
        assert workspace["realspace_energy_pair_atomic_selected_count"] > 0
        assert workspace["realspace_feature_owner_selected_count"] == 0
        assert workspace["realspace_energy_owner_selected_count"] == 0
    else:
        assert workspace["realspace_feature_owner_selected_count"] > 0
        assert workspace["realspace_energy_owner_selected_count"] > 0
        assert workspace["realspace_feature_pair_atomic_selected_count"] == 0
        assert workspace["realspace_energy_pair_atomic_selected_count"] == 0


def test_direct_reciprocal_does_not_resolve_realspace_reverse_strategy(
    polar_model_path,
):
    evaluator = _polar_calculator(polar_model_path, dtype="float64").evaluator
    evaluator.polar_delta_composed_forward(
        **_synthetic_chain_inputs(evaluator),
        compute_reverse=True,
        polar_electrostatics="direct_reciprocal",
    )
    workspace = _assert_realspace_reverse_resolution(
        evaluator, "auto", "unresolved", "reverse_not_requested"
    )
    assert workspace["realspace_feature_reverse_selected_count"] == 0
    assert workspace["realspace_energy_reverse_selected_count"] == 0


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_realspace_calculator_auto_matches_forced_for_e_ef_and_efs(
    polar_model_path, dtype
):
    """The actual native calculator serves every FFF lane through real space."""

    model_bytes = polar_model_path.read_bytes()
    channels = (
        ("singlet", {}),
        ("charged", {"total_charge": 1.0, "total_spin": 0.0}),
        ("external-field", {"electric_field": [0.1, -0.04, 0.02]}),
    )
    property_sets = (
        ("energy",),
        ("energy", "forces"),
        ("energy", "forces", "stress"),
    )
    tolerance = 4.0e-5 if dtype == "float32" else 2.0e-11
    last_auto = None
    last_atoms = None
    for channel_name, channel_kwargs in channels:
        automatic = Symmetrix(
            str(polar_model_path),
            dtype=dtype,
            streamed_edges="reference",
            **channel_kwargs,
        )
        forced = Symmetrix(
            str(polar_model_path),
            dtype=dtype,
            streamed_edges="reference",
            polar_electrostatics="realspace",
            **channel_kwargs,
        )
        for properties in property_sets:
            atoms = _molecule()
            automatic.calculate(atoms, properties=list(properties))
            forced.calculate(atoms.copy(), properties=list(properties))
            for name in properties:
                np.testing.assert_allclose(
                    automatic.results[name],
                    forced.results[name],
                    rtol=0.0,
                    atol=tolerance,
                    err_msg=f"{dtype}/{channel_name}/{properties}",
                )
            assert (
                automatic.polar_electrostatics_diagnostics["resolved_solver"]
                == "realspace"
            )
            assert (
                forced.polar_electrostatics_diagnostics["resolved_solver"]
                == "realspace"
            )
            last_auto = automatic
            last_atoms = atoms

    assert polar_model_path.read_bytes() == model_bytes
    assert last_auto is not None and last_atoms is not None
    workspace = last_auto.evaluator.polar_delta_workspace()
    assert workspace["n_k"] == 0
    for owner in ("cache", "arena", "adjoint"):
        assert workspace[f"reciprocal_{owner}_allocation_bytes"] == 0
    assert workspace["reciprocal_allocation_bytes"] == 0
    assert workspace["realspace_retained_allocation_bytes"] > 0
    assert workspace["realspace_retained_allocation_bytes"] < 10_000 * len(last_atoms)
    assert workspace["realspace_energy_selected_count"] > 0
    assert workspace["realspace_energy_receiver_replica_league_extent"] > 0
    assert workspace["realspace_feature_reverse_selected_count"] > 0
    assert workspace["realspace_energy_reverse_selected_count"] > 0
    if workspace["execution_space"] == "Cuda":
        assert workspace["realspace_reverse_strategy_resolved"] == "pair_atomic"
        assert workspace["realspace_feature_reverse_position_league_extent"] == 0
        assert workspace["realspace_energy_reverse_position_league_extent"] == 0
        assert workspace["realspace_feature_pair_league_extent"] > 0
        assert workspace["realspace_energy_pair_league_extent"] > 0
    else:
        assert workspace["realspace_reverse_strategy_resolved"] == "owner"
        assert workspace["realspace_feature_reverse_position_league_extent"] == (
            3 * len(last_atoms)
        )
        assert workspace["realspace_energy_reverse_position_league_extent"] == (
            3 * len(last_atoms)
        )
    assert workspace["diagnostic_state_materialization_count"] == 0
    assert (
        last_auto.evaluator.polar_potential_adjoint_diagnostics()["reference_count"]
        == 0
    )

    replacements = workspace["workspace_replacement_count"]
    reuse = workspace["workspace_reuse_count"]
    last_auto.calculate(last_atoms.copy(), properties=["energy", "forces", "stress"])
    reused = last_auto.evaluator.polar_delta_workspace()
    assert reused["workspace_replacement_count"] == replacements
    assert reused["workspace_reuse_count"] > reuse

    diagnostic = last_auto._materialize_polar_diagnostics()
    assert float(diagnostic["electrostatic_energy_correction"]) == 0.0
    dipole = np.asarray(diagnostic["dipole"]).reshape(3)
    assert float(diagnostic["external_field_energy"]) == pytest.approx(
        float(np.dot(diagnostic["electric_field"], dipole)),
        rel=0.0,
        abs=tolerance,
    )
    positions = np.asarray(diagnostic["positions"], dtype=np.float64)
    position_adjoint = np.asarray(diagnostic["positions_adjoint"], dtype=np.float64)
    volume = abs(float(np.linalg.det(np.asarray(last_atoms.cell))))
    np.testing.assert_allclose(
        np.asarray(diagnostic["stress"])[0],
        positions.T @ position_adjoint / volume,
        rtol=0.0,
        atol=tolerance,
    )


def test_one_solver_neutral_json_evaluates_fff_and_periodic_inputs(
    polar_model_path,
):
    """PBC selects the runtime algorithm without changing exported model data."""

    before = polar_model_path.read_bytes()
    automatic = Symmetrix(
        str(polar_model_path), dtype="float64", streamed_edges="reference"
    )
    isolated = _molecule()
    automatic.calculate(isolated, properties=["energy", "forces", "stress"])
    isolated_results = {
        name: np.array(automatic.results[name], copy=True)
        for name in ("energy", "forces", "stress")
    }
    np.testing.assert_array_equal(isolated_results["stress"], np.zeros(6))
    assert automatic.polar_electrostatics_diagnostics["resolved_solver"] == (
        "realspace"
    )

    periodic = _molecule()
    periodic.pbc = (True, False, False)
    automatic.calculate(periodic, properties=["energy", "forces", "stress"])
    assert automatic.polar_electrostatics_diagnostics["resolved_solver"] == (
        "direct_reciprocal"
    )
    assert automatic.polar_electrostatics_diagnostics["n_k"] > 0

    direct = Symmetrix(
        str(polar_model_path),
        dtype="float64",
        streamed_edges="reference",
        polar_electrostatics="direct_reciprocal",
    )
    direct.calculate(periodic.copy(), properties=["energy", "forces", "stress"])
    for name in ("energy", "forces", "stress"):
        np.testing.assert_allclose(
            automatic.results[name], direct.results[name], rtol=0.0, atol=2.0e-11
        )

    automatic.calculate(isolated.copy(), properties=["energy", "forces", "stress"])
    assert automatic.polar_electrostatics_diagnostics["resolved_solver"] == (
        "realspace"
    )
    workspace = automatic.evaluator.polar_delta_workspace()
    assert workspace["n_k"] == 0
    assert workspace["reciprocal_allocation_bytes"] == 0
    for name, expected in isolated_results.items():
        np.testing.assert_allclose(
            automatic.results[name], expected, rtol=0.0, atol=2.0e-11
        )
    assert polar_model_path.read_bytes() == before


@pytest.mark.parametrize(
    "properties",
    (("energy",), ("energy", "forces"), ("energy", "forces", "stress")),
)
def test_realspace_accepts_an_isolated_system_without_a_cell(
    polar_model_path, properties
):
    """An isolated system does not need a fictitious reporting volume."""

    import numpy as np
    from ase import Atoms

    atoms = Atoms("Si2H", positions=POSITIONS, pbc=False)
    calculator = Symmetrix(
        str(polar_model_path), dtype="float64", streamed_edges="reference"
    )

    calculator.calculate(atoms, properties=list(properties))

    assert np.isfinite(calculator.results["energy"])
    if "forces" in properties:
        assert np.isfinite(calculator.results["forces"]).all()
    if "stress" in properties:
        np.testing.assert_array_equal(calculator.results["stress"], np.zeros(6))
        diagnostic = calculator._materialize_polar_diagnostics()
        np.testing.assert_array_equal(
            diagnostic["stress"], np.zeros_like(diagnostic["stress"])
        )
    assert calculator.polar_electrostatics_diagnostics["resolved_solver"] == (
        "realspace"
    )


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize(
    "reverse_strategy", ("owner", "pair_atomic", "materialized_forward")
)
def test_realspace_composed_reverse_is_translation_and_fixed_cell_invariant(
    polar_model_path, dtype, reverse_strategy
):
    """The composed real route keeps FP64 position seeds and no cell seed."""

    evaluator = _polar_calculator(polar_model_path, dtype=dtype).evaluator
    _select_realspace_reverse_strategy(evaluator, reverse_strategy)
    inputs = _synthetic_chain_inputs(evaluator)
    inputs["external_potential"] = np.zeros((1, 4), dtype=np.float64)
    reference = evaluator.polar_delta_composed_forward(
        **inputs, compute_reverse=True, polar_electrostatics="realspace"
    )
    translated = evaluator.polar_delta_composed_forward(
        **{
            **inputs,
            "positions": np.ascontiguousarray(
                inputs["positions"] + np.array([4.0, -2.0, 1.0])
            ),
        },
        compute_reverse=True,
        polar_electrostatics="realspace",
    )
    changed_cell = evaluator.polar_delta_composed_forward(
        **{
            **inputs,
            "cell": np.ascontiguousarray(
                np.array([[[20.0, 1.0, -2.0], [0.0, 5.0, 0.5], [0.0, 0.0, 3.0]]])
            ),
        },
        compute_reverse=True,
        polar_electrostatics="realspace",
    )
    tolerance = 8.0e-4 if dtype == "float32" else 2.0e-9
    for candidate in (translated, changed_cell):
        np.testing.assert_allclose(
            candidate["energy"], reference["energy"], rtol=0.0, atol=tolerance
        )
        np.testing.assert_allclose(
            candidate["positions_adjoint"],
            reference["positions_adjoint"],
            rtol=0.0,
            atol=tolerance,
        )
        assert np.array_equal(candidate["strain_adjoint"], np.zeros((1, 9)))
    np.testing.assert_allclose(
        np.asarray(reference["positions_adjoint"]).sum(axis=0),
        np.zeros(3),
        rtol=0.0,
        atol=tolerance,
    )

    positions = np.asarray(inputs["positions"], dtype=np.float64)
    position_adjoint = np.asarray(reference["positions_adjoint"], dtype=np.float64)
    volume = abs(float(np.linalg.det(inputs["cell"][0])))
    np.testing.assert_allclose(
        np.asarray(reference["stress"]).reshape(3, 3),
        positions.T @ position_adjoint / volume,
        rtol=0.0,
        atol=tolerance,
    )

    step = 2.0e-3 if dtype == "float32" else 1.0e-5
    displaced_plus = positions.copy()
    displaced_minus = positions.copy()
    displaced_plus[0, 0] += step
    displaced_minus[0, 0] -= step

    def energy(candidate_positions):
        fresh = _polar_calculator(polar_model_path, dtype=dtype).evaluator
        _select_realspace_reverse_strategy(fresh, reverse_strategy)
        return float(
            np.asarray(
                fresh.polar_delta_composed_forward(
                    **{
                        **inputs,
                        "positions": np.ascontiguousarray(candidate_positions),
                    },
                    polar_electrostatics="realspace",
                )["energy"]
            ).reshape(-1)[0]
        )

    numeric = (energy(displaced_plus) - energy(displaced_minus)) / (2.0 * step)
    fd_tolerance = 3.0e-3 if dtype == "float32" else 2.0e-5
    assert float(position_adjoint[0, 0]) == pytest.approx(
        numeric, rel=fd_tolerance, abs=fd_tolerance
    )
    if reverse_strategy == "pair_atomic":
        _assert_pair_atomic_was_used(evaluator)


@pytest.mark.parametrize(
    "reverse_strategy", ("owner", "pair_atomic", "materialized_forward")
)
def test_fp32_realspace_external_field_keeps_fp64_gauge_under_large_translation(
    polar_model_path, reverse_strategy
):
    """The real external-field gauge must not narrow its FP64 barycenter."""

    evaluator = _polar_calculator(polar_model_path, dtype="float32").evaluator
    _select_realspace_reverse_strategy(evaluator, reverse_strategy)
    inputs = _synthetic_chain_inputs(evaluator)
    inputs["external_potential"] = np.asarray(
        [[0.13, 0.07, -0.03, 0.02]], dtype=np.float64
    )
    translation = np.asarray([1.0e6, -1.0e6, 1.0e6], dtype=np.float64)
    translated_inputs = {
        **inputs,
        "positions": np.ascontiguousarray(inputs["positions"] + translation),
    }

    reference_state = evaluator.polar_delta_chain(
        **inputs, polar_electrostatics="realspace"
    )
    reference = evaluator.polar_delta_composed_forward(
        **inputs, compute_reverse=True, polar_electrostatics="realspace"
    )
    translated_state = evaluator.polar_delta_chain(
        **translated_inputs, polar_electrostatics="realspace"
    )
    translated = evaluator.polar_delta_composed_forward(
        **translated_inputs, compute_reverse=True, polar_electrostatics="realspace"
    )

    # These are the existing FP32 stage/final-property bounds. With an FP32
    # barycenter, the translated displacement is quantized before either path
    # reaches them and exceeds the bounds by orders of magnitude.
    for actual, expected in zip(
        translated_state["potential_features"],
        reference_state["potential_features"],
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-5)
    # A nominally neutral FP32 density can retain a model-precision residual
    # monopole. Remove only its analytically known origin term before comparing
    # dipole and field energy; the potential-feature check above remains the
    # direct gauge regression and does not use this correction.
    reference_charge = float(
        np.asarray(reference_state["charge_density"], dtype=np.float32)[:, 0].sum(
            dtype=np.float64
        )
    )
    translated_charge = float(
        np.asarray(translated_state["charge_density"], dtype=np.float32)[:, 0].sum(
            dtype=np.float64
        )
    )
    reference_center = np.asarray(inputs["positions"]).mean(axis=0)
    translated_center = np.asarray(translated_inputs["positions"]).mean(axis=0)
    reference_dipole = (
        np.asarray(reference["dipole"]) - reference_charge * reference_center
    )
    translated_dipole = (
        np.asarray(translated["dipole"]) - translated_charge * translated_center
    )
    np.testing.assert_allclose(
        translated["positions_adjoint"],
        reference["positions_adjoint"],
        rtol=0.0,
        atol=8.0e-4,
    )
    np.testing.assert_allclose(
        translated_dipole, reference_dipole, rtol=0.0, atol=4.0e-5
    )
    field = inputs["external_potential"][0, 1:]
    reference_energy = np.asarray(reference["energy"]) - np.dot(
        field, reference_charge * reference_center
    )
    translated_energy = np.asarray(translated["energy"]) - np.dot(
        field, translated_charge * translated_center
    )
    np.testing.assert_allclose(
        translated_energy, reference_energy, rtol=0.0, atol=4.0e-5
    )
    if reverse_strategy == "pair_atomic":
        _assert_pair_atomic_was_used(evaluator)


@pytest.mark.parametrize(
    "reverse_strategy", ("owner", "pair_atomic", "materialized_forward")
)
def test_fp32_realspace_large_origin_total_charge_adjoint_matches_finite_difference(
    polar_model_path, reverse_strategy
):
    """Centering the learned seed must preserve the public charge adjoint."""

    inputs = _synthetic_chain_inputs(
        _polar_calculator(polar_model_path, dtype="float32").evaluator
    )
    inputs["positions"] = np.ascontiguousarray(
        inputs["positions"] + np.asarray([1.0e6, -1.0e6, 1.0e6], dtype=np.float64)
    )
    inputs["external_potential"] = np.asarray(
        [[0.13, 0.07, -0.03, 0.02]], dtype=np.float64
    )
    analytic_evaluator = _polar_calculator(polar_model_path, dtype="float32").evaluator
    _select_realspace_reverse_strategy(analytic_evaluator, reverse_strategy)
    analytic = analytic_evaluator.polar_delta_composed_forward(
        **inputs, compute_reverse=True, polar_electrostatics="realspace"
    )

    def energy(total_charge):
        evaluator = _polar_calculator(polar_model_path, dtype="float32").evaluator
        _select_realspace_reverse_strategy(evaluator, reverse_strategy)
        result = evaluator.polar_delta_composed_forward(
            **{**inputs, "total_charge": [total_charge]},
            polar_electrostatics="realspace",
        )
        return float(np.asarray(result["energy"]).reshape(-1)[0])

    step = 0.1
    numeric = (energy(step) - energy(-step)) / (2.0 * step)
    actual = float(np.asarray(analytic["total_charge_adjoint"])[0, 0])
    # The forward perturbs an FP32 target through the full nonlinear model at
    # an O(1e4) field energy.  Its observed centered-difference noise is
    # 1.23e-4 relative; retain the established 2e-4 FP32 adjoint bound.
    assert actual == pytest.approx(numeric, rel=2.0e-4, abs=2.0e-2)
    if reverse_strategy == "pair_atomic":
        _assert_pair_atomic_was_used(analytic_evaluator)


@pytest.mark.parametrize(
    "solver,reverse_strategy",
    (
        ("realspace", "owner"),
        ("realspace", "pair_atomic"),
        ("realspace", "materialized_forward"),
        ("direct_reciprocal", "owner"),
    ),
)
def test_external_field_total_charge_adjoint_matches_finite_difference_by_route(
    polar_model_path, solver, reverse_strategy
):
    """The real compensation must preserve ordinary inputs and isolate direct."""

    dtype = "float64"
    inputs = _synthetic_chain_inputs(
        _polar_calculator(polar_model_path, dtype).evaluator
    )
    inputs["external_potential"] = np.asarray(
        [[0.13, 0.07, -0.03, 0.02]], dtype=np.float64
    )
    analytic_evaluator = _polar_calculator(polar_model_path, dtype).evaluator
    _select_realspace_reverse_strategy(analytic_evaluator, reverse_strategy)
    analytic = analytic_evaluator.polar_delta_composed_forward(
        **inputs, compute_reverse=True, polar_electrostatics=solver
    )

    def energy(total_charge):
        evaluator = _polar_calculator(polar_model_path, dtype).evaluator
        _select_realspace_reverse_strategy(evaluator, reverse_strategy)
        result = evaluator.polar_delta_composed_forward(
            **{**inputs, "total_charge": [total_charge]},
            polar_electrostatics=solver,
        )
        return float(np.asarray(result["energy"]).reshape(-1)[0])

    step = 1.0e-5
    numeric = (energy(step) - energy(-step)) / (2.0 * step)
    actual = float(np.asarray(analytic["total_charge_adjoint"])[0, 0])
    assert actual == pytest.approx(numeric, rel=2.0e-5, abs=2.0e-5)
    if reverse_strategy == "pair_atomic":
        _assert_pair_atomic_was_used(analytic_evaluator)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_reciprocal_allocation_diagnostics_match_independent_shape_oracle(
    polar_model_path, dtype
):
    """Every retained reciprocal allocation is counted with its actual type."""

    evaluator = _polar_calculator(polar_model_path, dtype=dtype).evaluator
    inputs = _synthetic_chain_inputs(evaluator)
    evaluator.polar_delta_composed_forward(
        **inputs,
        polar_electrostatics="direct_reciprocal",
    )
    actual = evaluator.polar_delta_workspace()

    shapes = evaluator.polar_delta_shapes()
    contract = dict(evaluator.polar_long_range_contract)
    n_k = actual["n_k"]
    n_nodes = inputs["positions"].shape[0]
    n_graphs = inputs["cell"].shape[0]
    density_width = (int(contract["density_max_l"]) + 1) ** 2
    feature_width = (int(contract["feature_max_l"]) + 1) ** 2
    n_feature_sigmas = len(contract["feature_smearing_widths"])
    feature_dimension = n_feature_sigmas * feature_width
    assert density_width == shapes["density_width"]
    assert feature_dimension == shapes["field_dimension"]

    def elements(allocation_shapes):
        return sum(int(np.prod(shape)) for shape in allocation_shapes)

    # This is an independent inventory of the retained cache allocations, not
    # a call to (or restatement of the categories from) the native accounting
    # helper. It deliberately separates Precision, host double, and C int.
    cache_scalar_elements = elements(
        (
            (n_k, 3),  # k_vectors
            (n_k,),  # k_norm2
            (n_k,),  # k0_mask
            (n_k,),  # volume_per_k
            (n_k,),  # k_factor_coulomb
            (n_k,),  # k_factor_proj
            (n_k, n_nodes),  # cos_table
            (n_k, n_nodes),  # sin_table
            (n_k, n_feature_sigmas, feature_width, 2),
            (n_k, 1, density_width, 2),
            (n_graphs,),  # kernel_volumes
            (n_nodes, 3),  # positions_workspace
            (n_k, n_nodes),  # phase_workspace
        )
    )
    cache_host_double_elements = elements(
        (
            (n_k, 3),  # k_vectors_host
            (n_k,),  # k_norm2_host
            (n_k,),  # volume_per_k_host
            (n_graphs,),  # volumes_host
            (n_nodes, 3),  # positions_host_values
            (n_graphs, 3, 3),  # cell_host_values
        )
    )
    cache_int_elements = elements(
        (
            (n_k,),  # k_graph_index
            (n_nodes,),  # node_graph_index
            (n_k,),  # k_graph_index_host
            (n_nodes,),  # node_graph_index_host
            (n_nodes,),  # node_graph_host_values
        )
    )

    # A forward-only composed call retains these eight reusable ops_ views.
    arena_scalar_elements = elements(
        (
            (n_k, 2),  # rho
            (n_k, density_width),  # coeff_cos
            (n_k, density_width),  # coeff_sin
            (n_k, 2),  # potential
            (n_k, feature_dimension),  # proj_a
            (n_k, feature_dimension),  # proj_b
            (feature_dimension, n_nodes),  # proj_total
            (n_nodes, feature_dimension),  # features
        )
    )
    adjoint_scalar_elements = elements(
        (
            (n_k, n_nodes),  # grad_cos_workspace
            (n_k, n_nodes),  # grad_sin_workspace
            (n_k, 3),  # grad_k_workspace
            (n_k,),  # grad_volume_workspace
        )
    )

    scalar_bytes = np.dtype(dtype).itemsize
    double_bytes = np.dtype(np.float64).itemsize
    int_bytes = np.dtype(np.intc).itemsize
    expected = {
        "cache": (
            cache_scalar_elements + cache_host_double_elements + cache_int_elements,
            cache_scalar_elements * scalar_bytes
            + cache_host_double_elements * double_bytes
            + cache_int_elements * int_bytes,
        ),
        "arena": (arena_scalar_elements, arena_scalar_elements * scalar_bytes),
        "adjoint": (
            adjoint_scalar_elements,
            adjoint_scalar_elements * scalar_bytes,
        ),
    }
    for owner, (expected_elements, expected_bytes) in expected.items():
        assert (
            actual[f"reciprocal_{owner}_allocation_elements"],
            actual[f"reciprocal_{owner}_allocation_bytes"],
        ) == (expected_elements, expected_bytes)
    assert actual["reciprocal_allocation_elements"] == sum(
        owner[0] for owner in expected.values()
    )
    assert actual["reciprocal_allocation_bytes"] == sum(
        owner[1] for owner in expected.values()
    )


def test_staged_oracle_rejects_an_extension_without_its_native_program(
    polar_model_path,
):
    """The retired materializing compatibility path must remain unavailable."""

    class WithoutNativeProgram:
        def __init__(self, evaluator):
            self._evaluator = evaluator

        def __getattr__(self, name):
            if name == "polar_delta_native_program":
                raise AttributeError(name)
            return getattr(self._evaluator, name)

    calculator = _polar_calculator(polar_model_path, dtype="float64")
    atoms = _molecule()
    mace_inputs = calculator._mace_inputs(atoms)
    calculator._compute_mace(mace_inputs, atoms=atoms)
    native_evaluator = calculator.evaluator
    calculator.evaluator = WithoutNativeProgram(native_evaluator)
    try:
        with pytest.raises(
            family_registry.MissingCapabilityError,
            match="retained-layer materialization is not a supported",
        ):
            calculator._compose_polar(
                atoms,
                mace_inputs,
                compute_reverse=False,
                materialize_diagnostics=True,
            )
    finally:
        calculator.evaluator = native_evaluator
        native_evaluator.clear_layer_product_seed()


def test_production_whole_model_does_not_depend_on_the_staged_oracle_program(
    polar_model_path,
    tmp_path,
    monkeypatch,
):
    """Removing the staged oracle capability must not affect production."""

    class WithoutNativeProgram:
        def __init__(self, evaluator):
            self._evaluator = evaluator

        def __getattr__(self, name):
            if name == "polar_delta_native_program":
                raise AttributeError(name)
            return getattr(self._evaluator, name)

    calculator = Symmetrix(
        str(polar_model_path),
        dtype="float64",
        streamed_edges="direct",
        polar_electrostatics="realspace",
        total_charge=0.0,
        total_spin=1.0,
    )
    atoms = _molecule()
    native_forces = calculator.get_forces(atoms)
    native_evaluator = calculator.evaluator
    calculator.reset()
    calculator.evaluator = WithoutNativeProgram(native_evaluator)
    try:
        without_staged_program_forces = calculator.get_forces(atoms)
    finally:
        calculator.evaluator = native_evaluator
        native_evaluator.clear_layer_product_seed()
    np.testing.assert_allclose(
        native_forces,
        without_staged_program_forces,
        rtol=0.0,
        atol=2.0e-10,
    )


def test_polar_native_program_rejects_multiple_systems_before_allocation(
    polar_model_path,
):
    """The production boundary must not turn two graphs into mixed routing."""

    calculator = _polar_calculator(polar_model_path, dtype="float64")
    atoms = _molecule()
    mace_inputs = calculator._mace_inputs(atoms)
    calculator._compute_mace(mace_inputs, atoms=atoms)
    positions = np.ascontiguousarray(atoms.get_positions(), dtype=np.float64)
    cell = np.ascontiguousarray(np.asarray(atoms.cell, dtype=np.float64)).reshape(
        1, 3, 3
    )
    before = calculator.evaluator.polar_delta_workspace()
    try:
        with pytest.raises(ValueError, match="exactly one input system"):
            calculator.evaluator.polar_delta_native_program(
                node_types=np.asarray(mace_inputs[1], dtype=np.int32),
                positions=positions,
                cell=np.concatenate([cell, cell]),
                graph_pbc=np.zeros((2, 3), dtype=np.int32),
                node_graph_index=np.zeros(len(atoms), dtype=np.int32),
                total_charge=[0.0, 0.0],
                total_spin=[0.0, 0.0],
                external_potential=np.zeros((2, 4), dtype=np.float64),
                compute_reverse=False,
                materialize_outputs=False,
                polar_electrostatics="direct_reciprocal",
            )
    finally:
        calculator.evaluator.clear_layer_product_seed()
    after = calculator.evaluator.polar_delta_workspace()
    assert (
        after["workspace_replacement_count"] == (before["workspace_replacement_count"])
    )
    assert (
        after["arena_view_allocation_count"] == (before["arena_view_allocation_count"])
    )
    assert after["reciprocal_table_bytes"] == 0


def _prepare_native_polar_topology(calculator, atoms):
    """Prepare compact graph geometry without executing the backbone."""

    evaluations_before = calculator.evaluator.factorized_prepared_evaluation_count
    mace_inputs = calculator._mace_inputs(atoms)
    num_nodes, node_types, num_neigh, j_list, neigh_types, _, _, _ = mace_inputs
    graph_generation = calculator.evaluator._prepare_factorized_graph(
        num_nodes,
        np.asarray(node_types, dtype=np.int32),
        np.asarray(num_neigh, dtype=np.int32),
        np.asarray(j_list, dtype=np.int32),
        np.asarray(neigh_types, dtype=np.int32),
    )
    cache = calculator._materialize_neighbor_cache_geometry()
    calculator.evaluator._prepare_factorized_geometry(
        graph_generation,
        np.ascontiguousarray(cache.reference_positions, dtype=np.float64).reshape(-1),
        np.ascontiguousarray(cache.reference_xyz, dtype=np.float64).reshape(-1),
        np.ascontiguousarray(cache.cell, dtype=np.float64).reshape(-1),
        np.ascontiguousarray(cache.inverse_cell, dtype=np.float64).reshape(-1),
        np.ascontiguousarray(cache.pbc, dtype=np.int32).reshape(-1),
    )
    assert (
        calculator.evaluator.factorized_prepared_evaluation_count == evaluations_before
    )
    return graph_generation, mace_inputs


def _evaluate_native_polar_whole_model(
    calculator,
    atoms,
    properties,
    *,
    polar_electrostatics=None,
):
    """Prepare topology, then hand the complete scientific transaction native."""

    graph_generation, mace_inputs = _prepare_native_polar_topology(calculator, atoms)
    graph_inputs = calculator._resolve_polar_graph_inputs(atoms)
    electric_field = np.asarray(
        graph_inputs["electric_field"], dtype=np.float64
    ).reshape(3)
    include_forces = "forces" in properties or "stress" in properties
    include_stress = "stress" in properties
    return calculator.evaluator._evaluate_polar_prepared(
        graph_generation=graph_generation,
        node_types=np.asarray(mace_inputs[1], dtype=np.int32),
        positions=np.ascontiguousarray(atoms.get_positions(), dtype=np.float64),
        cell=np.ascontiguousarray(np.asarray(atoms.cell, dtype=np.float64)).reshape(
            1, 3, 3
        ),
        graph_pbc=np.asarray(atoms.pbc, dtype=np.int32).reshape(1, 3),
        node_graph_index=np.zeros(len(atoms), dtype=np.int32),
        total_charge=[float(graph_inputs["total_charge"])],
        total_spin=[float(graph_inputs["total_spin"])],
        external_potential=np.asarray([[0.0, *electric_field]], dtype=np.float64),
        include_forces=include_forces,
        include_stress=include_stress,
        include_dipole="dipole" in properties,
        polar_electrostatics=(
            calculator.polar_electrostatics
            if polar_electrostatics is None
            else polar_electrostatics
        ),
        pme_mesh=calculator.polar_pme_mesh,
        pme_spline_order=calculator.polar_pme_order,
    )


def _evaluate_staged_polar_oracle(calculator, atoms, properties):
    """Run the retired Python transaction explicitly as a test-only oracle."""

    properties = tuple(properties)
    mace_inputs = calculator._mace_inputs(atoms)
    graph_generation = calculator._compute_mace(mace_inputs, atoms=atoms)
    reverse_requested = bool({"forces", "stress"}.intersection(properties))
    composition = calculator._compose_polar(
        atoms,
        mace_inputs,
        compute_reverse=reverse_requested,
        materialize_diagnostics="dipole" in properties,
    )
    calculator._polar_composition = composition
    reporting_volume = (
        1.0
        if composition["resolved_solver"] == "realspace"
        else float(atoms.get_volume())
    )
    if reverse_requested:
        if not composition.get("_native_layer_seed_installed", False):
            composition = calculator._compose_polar_reverse()
        if composition.get("layer_product_seed") is not None:
            calculator.evaluator.set_layer_product_seed(
                composition["layer_product_seed"]
            )
        elif not calculator.evaluator.has_layer_product_seed:
            raise RuntimeError("the staged oracle did not install its product seed")
        try:
            graph_generation = calculator._compute_mace(mace_inputs, atoms=atoms)
            results = calculator.evaluator._collect_polar_public_outputs(
                volume=reporting_volume,
                graph_generation=graph_generation,
                include_forces=True,
                include_stress="stress" in properties,
            )
        finally:
            calculator.evaluator.clear_layer_product_seed()
    else:
        results = calculator.evaluator._collect_polar_public_outputs(
            volume=reporting_volume,
            graph_generation=graph_generation,
            include_forces=False,
            include_stress=False,
        )
    results = dict(results)
    results["energy"] = float(results["energy"])
    results["free_energy"] = results["energy"]
    if "stress" in results:
        results["stress"] = np.asarray(results["stress"], dtype=float).reshape(6)
    if "dipole" in properties:
        results["dipole"] = np.asarray(composition["dipole"], dtype=np.float64)
    return results


def _native_polar_prepared_kwargs(calculator, atoms):
    """Return a valid low-level whole-model request for mutation tests."""

    graph_generation, mace_inputs = _prepare_native_polar_topology(calculator, atoms)
    graph_inputs = calculator._resolve_polar_graph_inputs(atoms)
    electric_field = np.asarray(
        graph_inputs["electric_field"], dtype=np.float64
    ).reshape(3)
    return {
        "graph_generation": graph_generation,
        "node_types": np.asarray(mace_inputs[1], dtype=np.int32),
        "positions": np.ascontiguousarray(atoms.get_positions(), dtype=np.float64),
        "cell": np.ascontiguousarray(np.asarray(atoms.cell, dtype=np.float64)),
        "graph_pbc": np.asarray(atoms.pbc, dtype=np.int32).reshape(1, 3),
        "node_graph_index": np.zeros(len(atoms), dtype=np.int32),
        "total_charge": [float(graph_inputs["total_charge"])],
        "total_spin": [float(graph_inputs["total_spin"])],
        "external_potential": np.asarray([[0.0, *electric_field]], dtype=np.float64),
        "include_forces": False,
        "include_stress": False,
        "include_dipole": False,
        "polar_electrostatics": calculator.polar_electrostatics,
        "pme_mesh": calculator.polar_pme_mesh,
        "pme_spline_order": calculator.polar_pme_order,
    }


@pytest.mark.parametrize("dtype", ("float64", "float32"))
def test_cuda_forced_pme_whole_model_matches_direct_reciprocal(polar_model_path, dtype):
    if _native_module()._kokkos_default_execution_space() != "Cuda":
        pytest.skip("POLAR PME is first qualified on CUDA")
    capability = _native_module()._particle_mesh_capability()
    if not capability["available"]:
        pytest.skip("requires the CUDA particle-mesh FFT provider")
    atoms = _molecule()
    atoms.set_cell(np.eye(3) * WATER_BOX)
    atoms.set_pbc(True)
    direct = _polar_calculator(
        polar_model_path,
        dtype=dtype,
        polar_electrostatics="direct_reciprocal",
    )
    pme = _polar_calculator(
        polar_model_path,
        dtype=dtype,
        polar_electrostatics="pme",
    )
    properties = ("energy", "forces", "stress", "dipole")

    expected = _evaluate_native_polar_whole_model(direct, atoms, properties)
    actual = _evaluate_native_polar_whole_model(pme, atoms, properties)

    tolerance = 3e-3 if dtype == "float32" else 2e-5
    for name in properties:
        np.testing.assert_allclose(
            actual[name], expected[name], rtol=tolerance, atol=tolerance
        )
    assert actual["resolved_solver"] == "pme"
    assert actual["reciprocal_table_elements"] == 0
    assert actual["particle_mesh_active_bytes"] > 0
    assert (
        actual["particle_mesh_capacity_bytes"] >= actual["particle_mesh_active_bytes"]
    )
    assert actual["particle_mesh_volume"] == 64**3
    assert actual["particle_mesh_shape"] == [64, 64, 64]
    assert actual["particle_mesh_rebuilds"] == 1
    assert actual["particle_mesh_reuses"] > 0
    assert actual["particle_mesh_forward_plan_builds"] == 1
    assert actual["particle_mesh_inverse_plan_builds"] == 1


def test_cuda_forced_pme_whole_model_reuses_and_rebuilds_runtime_mesh(
    polar_model_path,
):
    if _native_module()._kokkos_default_execution_space() != "Cuda":
        pytest.skip("POLAR PME is first qualified on CUDA")
    if not _native_module()._particle_mesh_capability()["available"]:
        pytest.skip("requires the CUDA particle-mesh FFT provider")
    atoms = _molecule()
    atoms.set_cell(np.eye(3) * WATER_BOX)
    atoms.set_pbc(True)
    calculator = _polar_calculator(
        polar_model_path,
        polar_electrostatics="pme",
        polar_pme_mesh=(32, 32, 32),
        polar_pme_order=6,
    )

    first = _evaluate_native_polar_whole_model(calculator, atoms, ("energy",))
    second = _evaluate_native_polar_whole_model(calculator, atoms, ("energy",))
    assert first["particle_mesh_volume"] == 32**3
    assert first["particle_mesh_shape"] == [32, 32, 32]
    assert second["particle_mesh_rebuilds"] == first["particle_mesh_rebuilds"]
    assert second["particle_mesh_reuses"] > first["particle_mesh_reuses"]

    calculator.polar_pme_mesh = (40, 36, 32)
    rebuilt = _evaluate_native_polar_whole_model(calculator, atoms, ("energy",))
    assert rebuilt["particle_mesh_volume"] == 40 * 36 * 32
    assert rebuilt["particle_mesh_shape"] == [40, 36, 32]
    assert rebuilt["particle_mesh_rebuilds"] == first["particle_mesh_rebuilds"] + 1
    assert (
        rebuilt["particle_mesh_capacity_bytes"] >= rebuilt["particle_mesh_active_bytes"]
    )


def test_cuda_automatic_pme_uses_atom_mode_threshold_and_scaled_mesh(
    polar_model_path, tmp_path
):
    if _native_module()._kokkos_default_execution_space() != "Cuda":
        pytest.skip("POLAR PME automatic dispatch is first qualified on CUDA")
    if not _native_module()._particle_mesh_capability()["available"]:
        pytest.skip("requires the CUDA particle-mesh FFT provider")

    from ase import Atoms

    model = json.loads(polar_model_path.read_text())
    # A high diagnostic cutoff gives this compact 100-atom system exactly the
    # 400,000 retained modes needed to exercise the production 40M crossover
    # without making a large-cell test part of the routine unit suite.
    model["polar"]["kspace_cutoff"] = 46.0
    model["execution_contracts"][POLAR_CONTRACT_KEY]["field"]["kspace_cutoff"] = 46.0
    model_path = _write_model(tmp_path / "automatic-pme.json", model)
    grid = np.stack(
        np.meshgrid(
            np.linspace(0.4, 7.6, 5),
            np.linspace(0.4, 7.6, 5),
            np.linspace(0.5, 7.5, 4),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    atoms = Atoms("H100", positions=grid, cell=np.eye(3) * 8.0, pbc=True)
    calculator = _polar_calculator(
        model_path, dtype="float32", polar_electrostatics="auto"
    )
    try:
        assert calculator.evaluator.polar_long_range_contract["kspace_cutoff"] == 46.0
        calculator.calculate(atoms, properties=["energy"])
        diagnostics = calculator.polar_electrostatics_diagnostics
        assert diagnostics["resolved_solver"] == "pme", diagnostics
        assert diagnostics["solver_reason"] == "automatic_pme_atom_mode_pairs"
        assert diagnostics["automatic_atom_count"] == 100
        assert diagnostics["automatic_reciprocal_mode_count"] == 400_000
        assert diagnostics["automatic_reciprocal_mode_count_is_lower_bound"] is True
        assert diagnostics["automatic_atom_mode_pair_count"] == 40_000_000
        assert diagnostics["automatic_atom_mode_pair_threshold"] == 40_000_000
        assert diagnostics["particle_mesh_shape"] == [64, 64, 64]
        assert diagnostics["particle_mesh_volume"] == 64**3
    finally:
        del calculator


def test_cuda_forced_pme_whole_model_force_and_triclinic_stress_finite_difference(
    polar_model_path,
):
    if _native_module()._kokkos_default_execution_space() != "Cuda":
        pytest.skip("POLAR PME is first qualified on CUDA")
    if not _native_module()._particle_mesh_capability()["available"]:
        pytest.skip("requires the CUDA particle-mesh FFT provider")
    atoms = _molecule()
    atoms.set_cell(
        np.array(
            [
                [WATER_BOX, 0.4, -0.2],
                [0.0, WATER_BOX + 0.3, 0.25],
                [0.0, 0.0, WATER_BOX - 0.2],
            ]
        )
    )
    atoms.set_pbc(True)
    calculator = _polar_calculator(
        polar_model_path,
        polar_electrostatics="pme",
        polar_pme_mesh=(64, 64, 64),
        polar_pme_order=6,
        electric_field=np.array([0.03, -0.02, 0.01]),
    )
    analytic = _evaluate_native_polar_whole_model(
        calculator, atoms, ("energy", "forces", "stress")
    )
    step = 2e-5

    def energy(changed_atoms):
        return float(
            _evaluate_native_polar_whole_model(calculator, changed_atoms, ("energy",))[
                "energy"
            ]
        )

    plus = atoms.copy()
    minus = atoms.copy()
    plus.positions[0, 1] += step
    minus.positions[0, 1] -= step
    numeric_force = -(energy(plus) - energy(minus)) / (2.0 * step)
    assert analytic["forces"][0, 1] == pytest.approx(numeric_force, rel=3e-4, abs=3e-4)

    strain = np.zeros((3, 3))
    strain[0, 0] = step
    plus_transform = np.eye(3) + strain
    minus_transform = np.eye(3) - strain
    plus = atoms.copy()
    minus = atoms.copy()
    plus.set_cell(np.asarray(atoms.cell) @ plus_transform, scale_atoms=True)
    minus.set_cell(np.asarray(atoms.cell) @ minus_transform, scale_atoms=True)
    numeric_strain = (energy(plus) - energy(minus)) / (2.0 * step)
    volume = abs(float(np.linalg.det(atoms.cell)))
    assert analytic["stress"][0] * volume == pytest.approx(
        numeric_strain, rel=5e-4, abs=5e-4
    )


def test_production_polar_uses_one_native_whole_model_transaction(
    polar_model_path, monkeypatch, tmp_path
):
    """Production must not reintroduce the staged Python scientific program."""

    calculator = Symmetrix(
        str(polar_model_path),
        dtype="float64",
        streamed_edges="direct",
        polar_electrostatics="realspace",
        total_charge=0.25,
        total_spin=1.0,
        electric_field=(0.01, -0.02, 0.03),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("production entered the staged Polar transaction")

    for name in (
        "_compute_mace",
        "_compose_polar",
        "_compose_polar_reverse",
        "_compose_polar_stress",
    ):
        if hasattr(calculator, name):
            monkeypatch.setattr(calculator, name, forbidden)
    before = calculator.evaluator.polar_whole_model_transaction_count
    assert calculator._polar_ops is None

    calculator.calculate(
        _molecule(), properties=["energy", "forces", "stress", "dipole"]
    )

    assert calculator.evaluator.polar_whole_model_transaction_count == before + 1
    assert set(calculator.results) == {
        "energy",
        "free_energy",
        "forces",
        "stress",
        "dipole",
    }
    diagnostics = calculator.polar_electrostatics_diagnostics
    assert diagnostics["requested_solver"] == "realspace"
    assert diagnostics["resolved_solver"] == "realspace"
    assert diagnostics["reciprocal_allocation_bytes"] == 0
    assert diagnostics["reciprocal_table_elements"] == 0
    assert diagnostics["reciprocal_table_bytes"] == 0
    assert diagnostics["fallback_count"] >= 0
    assert diagnostics["reverse_strategy"] in {"owner", "pair_atomic"}


@pytest.mark.parametrize(
    ("properties", "expected"),
    (
        (("energy",), {"energy", "free_energy"}),
        (("free_energy",), {"free_energy"}),
        (("forces",), {"forces"}),
        (("stress",), {"stress"}),
        (("dipole",), {"dipole"}),
        (("free_energy", "forces", "dipole"), {"free_energy", "forces", "dipole"}),
        (("energy", "stress"), {"energy", "free_energy", "stress"}),
    ),
)
def test_production_polar_returns_only_requested_public_properties(
    polar_model_path, properties, expected
):
    """Native extras stay private; free energy accompanies requested energy."""

    calculator = _polar_calculator(
        polar_model_path,
        dtype="float64",
        polar_electrostatics="realspace",
    )

    calculator.calculate(_molecule(), properties=list(properties))

    assert set(calculator.results) == expected


@pytest.mark.parametrize(
    ("name", "value", "resolved_key"),
    (
        ("total_charge", -0.5, "total_charge"),
        ("total_spin", 1.0, "total_spin"),
        ("fermi_level", -2.0, "fermi_level"),
    ),
)
def test_polar_post_construction_graph_input_mutation_invalidates_results(
    polar_model_path, name, value, resolved_key
):
    """Valid Polar graph-input mutation must invalidate and reach native input."""

    calculator = _polar_calculator(
        polar_model_path,
        dtype="float64",
        polar_electrostatics="realspace",
    )
    atoms = _molecule()
    calculator.calculate(atoms, properties=["energy"])
    assert calculator.results

    setattr(calculator, name, value)

    assert calculator.results == {}
    calculator.calculate(atoms, properties=["energy"])
    assert calculator._polar_graph_inputs[resolved_key] == value


def test_polar_post_construction_solver_mutation_invalidates_results(
    polar_model_path,
):
    """A valid solver mutation must be observed by the next native transaction."""

    calculator = _polar_calculator(
        polar_model_path,
        dtype="float64",
        polar_electrostatics="direct_reciprocal",
    )
    atoms = _molecule()
    calculator.calculate(atoms, properties=["energy"])
    assert calculator.polar_electrostatics_diagnostics["resolved_solver"] == (
        "direct_reciprocal"
    )

    calculator.polar_electrostatics = "realspace"

    assert calculator.results == {}
    calculator.calculate(atoms, properties=["energy"])
    assert calculator.polar_electrostatics_diagnostics["resolved_solver"] == (
        "realspace"
    )


def test_polar_post_construction_electric_field_mutation_invalidates_results(
    polar_model_path,
):
    """The shared field property remains a valid Polar graph input."""

    calculator = _polar_calculator(
        polar_model_path,
        dtype="float64",
        polar_electrostatics="realspace",
    )
    atoms = _molecule()
    calculator.calculate(atoms, properties=["energy"])
    assert calculator.results

    calculator.electric_field = (0.01, -0.02, 0.03)

    assert calculator.results == {}
    calculator.calculate(atoms, properties=["energy"])
    np.testing.assert_allclose(
        calculator._polar_graph_inputs["electric_field"],
        (0.01, -0.02, 0.03),
        rtol=0.0,
        atol=0.0,
    )


def test_polar_free_energy_correction_runs_once_on_the_requested_value(
    polar_model_path,
):
    """Force-consistent energy remains a first-class corrected property."""

    calculator = _polar_calculator(
        polar_model_path,
        dtype="float64",
        polar_electrostatics="realspace",
    )
    calls = []

    class FreeEnergyCorrection:
        def __bool__(self):
            return True

        def add_results(self, _atoms, results, properties):
            calls.append((tuple(results), tuple(properties)))
            results["free_energy"] = float(results["free_energy"]) + 1.25

        def diagnostics(self):
            return {"calls": len(calls)}

    calculator._correction_pipeline = FreeEnergyCorrection()
    calculator.calculate(_molecule(), properties=["free_energy"])

    assert calls == [(("free_energy",), ("free_energy",))]
    assert set(calculator.results) == {"free_energy"}
    assert np.isfinite(calculator.results["free_energy"])
    assert calculator.correction_diagnostics == {"calls": 1}


def test_periodic_polar_forced_kokkos_graph_stays_device_resident_and_reuses(
    polar_model_path, monkeypatch
):
    """A host graph releases all topology before device-owned EFS reuse."""

    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "host")
    reference = Symmetrix(
        str(polar_model_path),
        dtype="float64",
        streamed_edges="direct",
        polar_electrostatics="direct_reciprocal",
    )
    calculator = Symmetrix(
        str(polar_model_path),
        dtype="float64",
        streamed_edges="direct",
        polar_electrostatics="direct_reciprocal",
    )
    atoms = _molecule()
    atoms.pbc = True
    evaluator = calculator.evaluator
    assert callable(getattr(evaluator, "_prepare_periodic_factorized_graph", None))
    properties = ["energy", "forces", "stress"]
    calculator.calculate(atoms, properties=properties)
    assert calculator.neighbor_graph_backend == "host"
    assert calculator._neighbor_cache.num_edges > 0
    assert evaluator.factorized_prepared_host_topology_elements >= (
        2 * calculator._neighbor_cache.num_edges
    )
    assert evaluator.factorized_schedule_host_topology_elements > 0
    atoms.positions[0, 0] += calculator.neighbor_skin
    original_materialize = calculator._materialize_neighbor_cache_geometry
    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "kokkos")
    monkeypatch.setattr(
        calculator,
        "_materialize_neighbor_cache_geometry",
        lambda: (_ for _ in ()).throw(
            AssertionError("device graph fell back to host geometry")
        ),
    )
    before = evaluator.polar_whole_model_transaction_count
    schedule_builds_before = evaluator.factorized_device_schedule_build_count
    topology_transfers_before = evaluator.factorized_schedule_topology_d2h_elements

    def evaluate_and_compare(backend="kokkos"):
        monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "host")
        reference.calculate(atoms.copy(), properties=properties)
        monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", backend)
        calculator.calculate(atoms, properties=properties)
        for name in properties:
            np.testing.assert_allclose(
                calculator.results[name],
                reference.results[name],
                rtol=1.0e-11,
                atol=1.0e-11,
            )

    evaluate_and_compare()
    generation = calculator._neighbor_cache.device_graph_generation
    assert evaluator.factorized_device_schedule_build_count == (
        schedule_builds_before + 1
    )
    assert evaluator.factorized_schedule_topology_d2h_elements == (
        topology_transfers_before
    )
    assert evaluator.factorized_prepared_host_topology_elements == 0
    assert evaluator.factorized_schedule_host_topology_elements == 0
    atoms.positions[0, 0] += 0.01
    evaluate_and_compare()

    assert calculator.neighbor_graph_backend == "kokkos"
    assert generation != 0
    assert calculator._neighbor_cache.device_graph_generation == generation
    atoms.set_cell(np.eye(3) * (WATER_BOX + 0.01), scale_atoms=True)
    evaluate_and_compare()

    assert calculator._neighbor_cache.device_graph_generation != generation
    assert evaluator.factorized_device_schedule_build_count == (
        schedule_builds_before + 2
    )
    assert evaluator.factorized_schedule_topology_d2h_elements == (
        topology_transfers_before
    )
    assert evaluator.factorized_prepared_host_topology_elements == 0
    assert evaluator.factorized_schedule_host_topology_elements == 0
    assert evaluator.polar_whole_model_transaction_count == before + 3

    monkeypatch.setattr(
        calculator, "_materialize_neighbor_cache_geometry", original_materialize
    )
    atoms.pbc = False
    evaluate_and_compare("host")
    assert calculator.neighbor_graph_backend == "host"
    assert evaluator.factorized_prepared_host_topology_elements >= (
        2 * calculator._neighbor_cache.num_edges
    )
    assert evaluator.factorized_schedule_host_topology_elements > 0
    assert evaluator.polar_whole_model_transaction_count == before + 4


def test_large_periodic_polar_automatic_graph_selects_device_without_host_fallback(
    polar_model_path, monkeypatch
):
    """The qualified automatic threshold must remain available to Polar."""

    from ase import Atoms

    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "automatic")
    calculator = Symmetrix(
        str(polar_model_path),
        dtype="float64",
        streamed_edges="direct",
        polar_electrostatics="direct_reciprocal",
    )
    monkeypatch.setattr(
        _native_module(), "_kokkos_default_execution_space", lambda: "Cuda"
    )
    grid = np.indices((8, 8, 8)).reshape(3, -1).T.astype(float) * 4.0
    atoms = Atoms("H" * 512, positions=grid, cell=[32.0, 32.0, 32.0], pbc=True)
    monkeypatch.setattr(
        calculator,
        "_materialize_neighbor_cache_geometry",
        lambda: (_ for _ in ()).throw(
            AssertionError("automatic device graph fell back to host geometry")
        ),
    )
    before = calculator.evaluator.polar_whole_model_transaction_count

    calculator.calculate(atoms, properties=["energy"])

    assert calculator.neighbor_graph_backend == "kokkos"
    assert calculator._neighbor_cache.device_graph_generation != 0
    assert calculator.evaluator.polar_whole_model_transaction_count == before + 1


def test_missing_native_whole_model_surface_fails_without_materialization(
    polar_model_path,
):
    """An ABI mismatch must fail instead of downloading retained layers."""

    calculator = _polar_calculator(
        polar_model_path,
        dtype="float64",
        polar_electrostatics="realspace",
    )
    evaluator = calculator.evaluator

    class MissingWholeModelEvaluator:
        _evaluate_polar_prepared = None

        def __getattr__(self, name):
            if name == "layer_features":
                raise AssertionError("compatibility materialization was attempted")
            return getattr(evaluator, name)

    calculator.evaluator = MissingWholeModelEvaluator()

    with pytest.raises(
        family_registry.MissingCapabilityError,
        match="whole-model|whole_model|native.*Polar",
    ):
        calculator.calculate(_molecule(), properties=["energy"])


def test_pytorch_polar_checkpoint_resolves_adapter_after_extraction(
    tmp_path,
):
    """Family detection must use the extracted model type, not the JSON default."""

    import torch

    checkpoint = tmp_path / "polar.model"
    torch.save(_build_polar_shaped_model(), checkpoint)

    calculator = Symmetrix(
        checkpoint,
        dtype="float64",
        streamed_edges="reference",
        polar_electrostatics="realspace",
    )

    assert calculator._native_model_type == POLAR_MODEL_TYPE
    assert calculator._family_frontend_adapter is not None
    assert calculator._polar_ops is None


def test_polar_corrections_apply_once_after_the_native_public_result(
    polar_model_path,
):
    """The correction pipeline post-processes the complete native base once."""

    calculator = _polar_calculator(
        polar_model_path,
        dtype="float64",
        polar_electrostatics="realspace",
    )
    native_evaluator = calculator.evaluator
    native_calls = []

    class RecordingEvaluator:
        def _evaluate_polar_prepared(self, **kwargs):
            result = native_evaluator._evaluate_polar_prepared(**kwargs)
            native_calls.append(float(result["energy"]))
            return result

        def __getattr__(self, name):
            return getattr(native_evaluator, name)

    class RecordingCorrectionPipeline:
        def __init__(self):
            self.calls = []

        def __bool__(self):
            return True

        def add_results(self, _atoms, results, properties):
            assert native_calls
            assert float(results["energy"]) == native_calls[-1]
            self.calls.append((tuple(results), tuple(properties)))
            results["energy"] = float(results["energy"]) + 2.0

        def diagnostics(self):
            return {"calls": len(self.calls)}

    pipeline = RecordingCorrectionPipeline()
    calculator.evaluator = RecordingEvaluator()
    calculator._correction_pipeline = pipeline

    calculator.calculate(_molecule(), properties=["energy"])

    assert len(native_calls) == 1
    assert pipeline.calls == [(("energy",), ("energy",))]
    assert calculator.results["energy"] == native_calls[0] + 2.0
    assert set(calculator.results) == {"energy"}
    assert calculator.correction_diagnostics == {"calls": 1}


def test_polar_model_retains_alchemical_rejection(polar_model_path):
    """The refactor-owned alchemical policy rejects the Polar family early."""

    with pytest.raises(
        ValueError,
        match="standard two-interaction compact MACE model",
    ):
        Symmetrix(
            polar_model_path,
            dtype="float64",
            alchemical={1: {1: 0.5, 14: 0.5}},
        )


@pytest.mark.parametrize(("dtype", "atol"), (("float32", 3e-5), ("float64", 2e-11)))
@pytest.mark.parametrize(
    "properties",
    (("energy",), ("energy", "forces"), ("energy", "forces", "stress", "dipole")),
)
def test_native_whole_model_matches_the_staged_polar_transaction(
    polar_model_path, monkeypatch, tmp_path, dtype, atol, properties
):
    """One native call must reproduce the staged backbone/Polar transaction."""

    atoms = _molecule()
    staged = Symmetrix(
        str(polar_model_path),
        dtype=dtype,
        streamed_edges="direct",
        polar_electrostatics="realspace",
        total_charge=0.25,
        total_spin=1.0,
        electric_field=(0.01, -0.02, 0.03),
    )
    staged_results = _evaluate_staged_polar_oracle(staged, atoms.copy(), properties)
    native = Symmetrix(
        str(polar_model_path),
        dtype=dtype,
        streamed_edges="direct",
        polar_electrostatics="realspace",
        total_charge=0.25,
        total_spin=1.0,
        electric_field=(0.01, -0.02, 0.03),
    )

    native.calculate(atoms.copy(), properties=list(properties))
    result = native.results

    for name in properties:
        np.testing.assert_allclose(
            result[name], staged_results[name], rtol=0.0, atol=atol
        )
    diagnostics = native.polar_electrostatics_diagnostics
    assert diagnostics["resolved_solver"] == "realspace"
    assert diagnostics["reciprocal_allocation_bytes"] == 0
    assert diagnostics["backbone_fallback_count"] == 0
    assert diagnostics["polar_fallback_count"] > 0
    assert diagnostics["fallback_count"] == (
        diagnostics["backbone_fallback_count"] + diagnostics["polar_fallback_count"]
    )
    reverse_requested = "forces" in properties or "stress" in properties
    assert diagnostics["reverse_strategy"] in (
        {"owner", "pair_atomic"} if reverse_requested else {"unresolved"}
    )
    assert diagnostics["reverse_executed"] is reverse_requested
    expected_evaluations = 2 if reverse_requested else 1
    interaction_count = native.evaluator.polar_delta_shapes()["interaction_count"]
    assert diagnostics["backbone_evaluation_count"] == expected_evaluations
    assert diagnostics["backbone_forward_layer_count"] == (
        expected_evaluations * interaction_count
    )
    assert diagnostics["backbone_reverse_layer_count"] == (
        expected_evaluations * interaction_count
    )
    assert diagnostics["seeded_backbone_evaluation_count"] == (
        1 if reverse_requested else 0
    )
    assert diagnostics["polar_reverse_count"] == (1 if reverse_requested else 0)
    assert diagnostics["cumulative_transaction_count"] == 1
    assert diagnostics["cumulative_polar_reverse_count"] == (
        1 if reverse_requested else 0
    )
    assert "transaction_count" not in diagnostics
    assert "reverse_count" not in diagnostics
    assert diagnostics["product_seed_cleared"] is True
    assert native.evaluator.has_layer_product_seed is False
    workspace = native.evaluator.polar_delta_workspace()
    assert workspace["diagnostic_state_materialization_count"] == 0
    if not reverse_requested:
        assert workspace["resident_reverse_count"] == 0
        assert workspace["reciprocal_adjoint_allocation_bytes"] == 0
        assert workspace["realspace_atomic_accumulator_bytes"] == 0
    assert not {
        "layer_features",
        "layer_product_seed",
        "node_features_adjoint",
        "positions_adjoint",
    }.intersection(result)


@pytest.mark.parametrize(("dtype", "atol"), (("float32", 3e-5), ("float64", 2e-11)))
@pytest.mark.parametrize(
    ("route", "cell_free", "expected_solver"),
    (("auto", True, "realspace"), ("direct_reciprocal", False, "direct_reciprocal")),
)
def test_native_whole_model_routes_cell_free_and_periodic_systems(
    polar_model_path,
    monkeypatch,
    tmp_path,
    dtype,
    atol,
    route,
    cell_free,
    expected_solver,
):
    """One native transaction preserves auto/cell-free and periodic routing."""

    from ase import Atoms

    atoms = Atoms("Si2H", positions=POSITIONS, pbc=False) if cell_free else _molecule()
    if not cell_free:
        atoms.pbc = (True, True, True)
    kwargs = dict(
        dtype=dtype,
        streamed_edges="direct",
        polar_electrostatics=route,
        total_charge=-0.2,
        total_spin=1.0,
        electric_field=(-0.01, 0.015, 0.02),
    )
    staged = Symmetrix(str(polar_model_path), **kwargs)
    staged_results = _evaluate_staged_polar_oracle(
        staged, atoms.copy(), ("energy", "forces", "stress", "dipole")
    )
    native = Symmetrix(str(polar_model_path), **kwargs)
    native.calculate(atoms.copy(), properties=["energy", "forces", "stress", "dipole"])
    result = native.results

    for name in ("energy", "forces", "stress", "dipole"):
        np.testing.assert_allclose(
            result[name], staged_results[name], rtol=0.0, atol=atol
        )
    diagnostics = native.polar_electrostatics_diagnostics
    assert diagnostics["requested_solver"] == route
    assert diagnostics["resolved_solver"] == expected_solver
    assert diagnostics["backbone_evaluation_count"] == 2
    assert diagnostics["seeded_backbone_evaluation_count"] == 1
    assert diagnostics["polar_reverse_count"] == 1
    assert diagnostics["product_seed_cleared"] is True
    if cell_free:
        np.testing.assert_array_equal(result["stress"], np.zeros(6))
        assert diagnostics["reciprocal_allocation_bytes"] == 0
    else:
        assert diagnostics["reciprocal_allocation_bytes"] > 0


@pytest.mark.parametrize("route", ("auto", "realspace", "direct_reciprocal"))
def test_native_whole_model_rejects_multi_system_before_allocation(
    polar_model_path, monkeypatch, tmp_path, route
):
    """All solver requests reject graph batches before native transactions."""

    calculator = Symmetrix(
        str(polar_model_path),
        dtype="float64",
        streamed_edges="direct",
        polar_electrostatics=route,
    )
    atoms = _molecule()
    graph_generation, mace_inputs = _prepare_native_polar_topology(calculator, atoms)
    evaluator = calculator.evaluator
    before_workspace = evaluator.polar_delta_workspace()
    before_transaction = evaluator.polar_whole_model_transaction_count
    before_failure = evaluator.polar_whole_model_failure_count

    with pytest.raises(ValueError, match="exactly one input system"):
        evaluator._evaluate_polar_prepared(
            graph_generation=graph_generation,
            node_types=np.asarray(mace_inputs[1], dtype=np.int32),
            positions=np.ascontiguousarray(atoms.get_positions(), dtype=np.float64),
            cell=np.ascontiguousarray(np.asarray(atoms.cell, dtype=np.float64)),
            graph_pbc=np.zeros((1, 3), dtype=np.int32),
            node_graph_index=np.zeros(len(atoms), dtype=np.int32),
            total_charge=[0.0, 0.0],
            total_spin=[0.0, 0.0],
            external_potential=np.zeros((2, 4), dtype=np.float64),
            include_forces=False,
            include_stress=False,
            include_dipole=False,
            polar_electrostatics=route,
        )

    after_workspace = evaluator.polar_delta_workspace()
    assert evaluator.polar_whole_model_transaction_count == before_transaction
    assert evaluator.polar_whole_model_failure_count == before_failure
    assert (
        after_workspace["workspace_replacement_count"]
        == before_workspace["workspace_replacement_count"]
    )
    assert (
        after_workspace["arena_view_allocation_count"]
        == before_workspace["arena_view_allocation_count"]
    )
    assert after_workspace["reciprocal_allocation_bytes"] == 0


@pytest.mark.parametrize("route", ("auto", "realspace", "direct_reciprocal"))
def test_native_whole_model_rejects_nonzero_node_graph_with_valid_scalars(
    polar_model_path, monkeypatch, tmp_path, route
):
    """The C++ node-owner loop rejects batches after scalar extents pass."""

    atoms = _molecule()
    calculator = Symmetrix(
        str(polar_model_path),
        dtype="float64",
        streamed_edges="direct",
        polar_electrostatics=route,
    )
    evaluator = calculator.evaluator
    request = _native_polar_prepared_kwargs(calculator, atoms)
    changed_graph = np.asarray(request["node_graph_index"], dtype=np.int32).copy()
    changed_graph[0] = 1
    request["node_graph_index"] = changed_graph
    before = (
        evaluator.factorized_prepared_evaluation_count,
        evaluator.polar_whole_model_transaction_count,
        evaluator.polar_whole_model_failure_count,
    )

    with pytest.raises(ValueError, match="node graph index"):
        evaluator._evaluate_polar_prepared(**request)

    assert (
        evaluator.factorized_prepared_evaluation_count,
        evaluator.polar_whole_model_transaction_count,
        evaluator.polar_whole_model_failure_count,
    ) == before


def test_native_whole_model_rejects_periodic_forced_realspace_before_backbone(
    polar_model_path, monkeypatch, tmp_path
):
    """Route rejection must precede every backbone/Polar transaction."""

    atoms = _molecule()
    atoms.pbc = (True, True, True)
    calculator = Symmetrix(
        str(polar_model_path),
        dtype="float64",
        streamed_edges="direct",
        polar_electrostatics="realspace",
    )
    evaluator = calculator.evaluator
    request = _native_polar_prepared_kwargs(calculator, atoms)
    before_workspace = evaluator.polar_delta_workspace()
    before = (
        evaluator.factorized_prepared_evaluation_count,
        evaluator.polar_whole_model_transaction_count,
        evaluator.polar_whole_model_failure_count,
    )

    with pytest.raises(ValueError, match="fully non-periodic"):
        evaluator._evaluate_polar_prepared(**request)

    after_workspace = evaluator.polar_delta_workspace()
    assert (
        evaluator.factorized_prepared_evaluation_count,
        evaluator.polar_whole_model_transaction_count,
        evaluator.polar_whole_model_failure_count,
    ) == before
    for name in (
        "workspace_replacement_count",
        "arena_view_allocation_count",
        "reciprocal_table_bytes",
        "realspace_retained_allocation_bytes",
    ):
        assert after_workspace[name] == before_workspace[name]


@pytest.mark.parametrize("mutation", ("node_types", "cell", "pbc"))
def test_native_whole_model_rejects_state_different_from_prepared_graph(
    polar_model_path, monkeypatch, tmp_path, mutation
):
    """Backbone and Polar must consume one validated prepared system state."""

    atoms = _molecule()
    calculator = Symmetrix(
        str(polar_model_path),
        dtype="float64",
        streamed_edges="direct",
        polar_electrostatics="auto",
    )
    evaluator = calculator.evaluator
    request = _native_polar_prepared_kwargs(calculator, atoms)
    if mutation == "node_types":
        changed = np.asarray(request["node_types"], dtype=np.int32).copy()
        different = np.flatnonzero(changed != changed[0])
        assert different.size
        changed[0], changed[different[0]] = changed[different[0]], changed[0]
        request["node_types"] = changed
    elif mutation == "cell":
        changed = np.asarray(request["cell"], dtype=np.float64).copy()
        changed.reshape(-1)[0] += 0.125
        request["cell"] = changed
    else:
        changed = np.asarray(request["graph_pbc"], dtype=np.int32).copy()
        changed.reshape(-1)[0] = 1 - changed.reshape(-1)[0]
        request["graph_pbc"] = changed
    before = (
        evaluator.factorized_prepared_evaluation_count,
        evaluator.polar_whole_model_transaction_count,
        evaluator.polar_whole_model_failure_count,
    )

    with pytest.raises(ValueError, match="prepared"):
        evaluator._evaluate_polar_prepared(**request)

    assert (
        evaluator.factorized_prepared_evaluation_count,
        evaluator.polar_whole_model_transaction_count,
        evaluator.polar_whole_model_failure_count,
    ) == before


def test_native_whole_model_does_not_accept_a_caller_reporting_volume(
    polar_model_path, monkeypatch, tmp_path
):
    """Periodic stress volume is derived from the prepared native cell."""

    atoms = _molecule()
    atoms.pbc = (True, True, True)
    calculator = Symmetrix(
        str(polar_model_path),
        dtype="float64",
        streamed_edges="direct",
        polar_electrostatics="direct_reciprocal",
    )
    request = _native_polar_prepared_kwargs(calculator, atoms)
    request["reporting_volume"] = 2.0 * float(atoms.get_volume())

    with pytest.raises(TypeError, match="reporting_volume"):
        calculator.evaluator._evaluate_polar_prepared(**request)


def test_native_whole_model_clears_existing_seed_before_early_validation(
    polar_model_path, monkeypatch, tmp_path
):
    """No malformed request can leave a seed installed by an earlier consumer."""

    atoms = _molecule()
    calculator = Symmetrix(
        str(polar_model_path),
        dtype="float64",
        streamed_edges="direct",
        polar_electrostatics="realspace",
    )
    evaluator = calculator.evaluator
    base = _native_polar_prepared_kwargs(calculator, atoms)
    evaluator._evaluate_polar_prepared(**base)
    retained = evaluator.layer_features()
    assert retained

    mutations = []
    stale = dict(base)
    stale["graph_generation"] += 1
    mutations.append(stale)
    extent = dict(base)
    extent["positions"] = np.asarray(base["positions"]).reshape(-1)[:-1]
    mutations.append(extent)
    nonfinite = dict(base)
    changed_positions = np.asarray(base["positions"], dtype=np.float64).copy()
    changed_positions.reshape(-1)[0] = np.nan
    nonfinite["positions"] = changed_positions
    mutations.append(nonfinite)
    wrong_graph = dict(base)
    changed_graph = np.asarray(base["node_graph_index"], dtype=np.int32).copy()
    changed_graph[0] = 1
    wrong_graph["node_graph_index"] = changed_graph
    mutations.append(wrong_graph)

    for request in mutations:
        evaluator.set_layer_product_seed(
            [np.ones_like(layer, dtype=np.float64) for layer in retained]
        )
        assert evaluator.has_layer_product_seed is True
        with pytest.raises(ValueError):
            evaluator._evaluate_polar_prepared(**request)
        assert evaluator.has_layer_product_seed is False


def test_native_whole_model_rejects_invalid_prepared_periodic_volume(
    polar_model_path, monkeypatch, tmp_path
):
    """Periodic stress derives and validates volume from the prepared cell."""

    atoms = _molecule()
    atoms.pbc = (True, True, True)
    calculator = Symmetrix(
        str(polar_model_path),
        dtype="float64",
        streamed_edges="direct",
        polar_electrostatics="direct_reciprocal",
    )
    evaluator = calculator.evaluator
    request = _native_polar_prepared_kwargs(calculator, atoms)
    evaluator._evaluate_polar_prepared(**request)
    retained = evaluator.layer_features()
    assert retained
    evaluator.set_layer_product_seed(
        [np.ones_like(layer, dtype=np.float64) for layer in retained]
    )
    cache = calculator._materialize_neighbor_cache_geometry()
    singular_cell = np.zeros(9, dtype=np.float64)
    evaluator._prepare_factorized_geometry(
        request["graph_generation"],
        np.ascontiguousarray(cache.reference_positions, dtype=np.float64).reshape(-1),
        np.ascontiguousarray(cache.reference_xyz, dtype=np.float64).reshape(-1),
        singular_cell,
        np.eye(3, dtype=np.float64).reshape(-1),
        np.ones(3, dtype=np.int32),
    )
    request["cell"] = singular_cell
    request["graph_pbc"] = np.ones((1, 3), dtype=np.int32)
    request["include_forces"] = True
    request["include_stress"] = True
    before_failure = evaluator.polar_whole_model_failure_count
    before_reverse = evaluator.polar_whole_model_reverse_count
    before_transactions = evaluator.polar_whole_model_transaction_count

    with pytest.raises(ValueError, match="finite positive volume"):
        evaluator._evaluate_polar_prepared(**request)

    assert evaluator.polar_whole_model_transaction_count == before_transactions
    assert evaluator.polar_whole_model_reverse_count == before_reverse
    assert evaluator.polar_whole_model_failure_count == before_failure
    assert evaluator.has_layer_product_seed is False


@pytest.mark.parametrize("include_forces", (False, True), ids=("E", "EF"))
def test_native_whole_model_rejects_singular_direct_cell_before_transaction(
    polar_model_path, monkeypatch, tmp_path, include_forces
):
    """Every periodic direct route validates its cell before evaluation."""

    atoms = _molecule()
    atoms.pbc = (True, True, True)
    calculator = Symmetrix(
        str(polar_model_path),
        dtype="float64",
        streamed_edges="direct",
        polar_electrostatics="direct_reciprocal",
    )
    evaluator = calculator.evaluator
    request = _native_polar_prepared_kwargs(calculator, atoms)
    evaluator._evaluate_polar_prepared(**request)
    retained = evaluator.layer_features()
    assert retained
    singular_cell = np.zeros(9, dtype=np.float64)
    cache = calculator._materialize_neighbor_cache_geometry()
    evaluator._prepare_factorized_geometry(
        request["graph_generation"],
        np.ascontiguousarray(cache.reference_positions, dtype=np.float64).reshape(-1),
        np.ascontiguousarray(cache.reference_xyz, dtype=np.float64).reshape(-1),
        singular_cell,
        np.eye(3, dtype=np.float64).reshape(-1),
        np.ones(3, dtype=np.int32),
    )
    request["cell"] = singular_cell
    request["graph_pbc"] = np.ones((1, 3), dtype=np.int32)
    request["include_forces"] = include_forces
    evaluator.set_layer_product_seed(
        [np.ones_like(layer, dtype=np.float64) for layer in retained]
    )
    before_workspace = evaluator.polar_delta_workspace()
    before = (
        evaluator.factorized_prepared_evaluation_count,
        evaluator.polar_whole_model_transaction_count,
        evaluator.polar_whole_model_failure_count,
        evaluator.polar_whole_model_reverse_count,
    )

    with pytest.raises(ValueError, match="finite positive volume"):
        evaluator._evaluate_polar_prepared(**request)

    after_workspace = evaluator.polar_delta_workspace()
    assert (
        evaluator.factorized_prepared_evaluation_count,
        evaluator.polar_whole_model_transaction_count,
        evaluator.polar_whole_model_failure_count,
        evaluator.polar_whole_model_reverse_count,
    ) == before
    assert evaluator.has_layer_product_seed is False
    for name in (
        "workspace_replacement_count",
        "arena_view_allocation_count",
        "reciprocal_table_bytes",
        "reciprocal_adjoint_allocation_bytes",
        "realspace_retained_allocation_bytes",
        "realspace_atomic_accumulator_bytes",
    ):
        assert after_workspace[name] == before_workspace[name]


def test_failed_geometry_reprepare_preserves_the_previous_prepared_state(
    polar_model_path, monkeypatch, tmp_path
):
    """A failed same-token reprepare cannot publish hybrid device geometry."""

    atoms = _molecule()
    calculator = Symmetrix(
        str(polar_model_path),
        dtype="float64",
        streamed_edges="direct",
        polar_electrostatics="realspace",
    )
    evaluator = calculator.evaluator
    request = _native_polar_prepared_kwargs(calculator, atoms)
    baseline = evaluator._evaluate_polar_prepared(**request)
    cache = calculator._materialize_neighbor_cache_geometry()
    changed_xyz = np.ascontiguousarray(cache.reference_xyz, dtype=np.float64).reshape(
        -1
    )
    changed_xyz = changed_xyz.copy()
    assert changed_xyz.size
    changed_xyz[0] += 0.2

    evaluator._arm_factorized_geometry_prepare_failure()
    with pytest.raises(RuntimeError, match="injected prepared-geometry failure"):
        evaluator._prepare_factorized_geometry(
            request["graph_generation"],
            np.ascontiguousarray(cache.reference_positions, dtype=np.float64).reshape(
                -1
            ),
            changed_xyz,
            np.ascontiguousarray(np.asarray(atoms.cell), dtype=np.float64).reshape(-1),
            np.ascontiguousarray(np.linalg.inv(np.asarray(atoms.cell))).reshape(-1),
            np.asarray(atoms.pbc, dtype=np.int32),
        )

    perturbed_request = dict(request)
    perturbed_positions = np.asarray(request["positions"], dtype=np.float64).copy()
    perturbed_positions.reshape(-1)[0] += 1.0e-4
    perturbed_request["positions"] = perturbed_positions
    evaluator._evaluate_polar_prepared(**perturbed_request)

    # Force edge geometry to be rebuilt from the reference views retained by
    # the failed publication, then restore the original positions.
    repeated = evaluator._evaluate_polar_prepared(**request)
    assert repeated["energy"] == pytest.approx(baseline["energy"], rel=0.0, abs=2e-12)


def test_successful_geometry_reprepare_invalidates_cached_positions(
    polar_model_path, monkeypatch, tmp_path
):
    """A published same-token geometry must not reuse its predecessor's edges."""

    atoms = _molecule()
    calculator = Symmetrix(
        str(polar_model_path),
        dtype="float64",
        streamed_edges="direct",
        polar_electrostatics="realspace",
    )
    evaluator = calculator.evaluator
    request = _native_polar_prepared_kwargs(calculator, atoms)
    evaluator._evaluate_polar_prepared(**request)
    cache = calculator._materialize_neighbor_cache_geometry()
    changed_xyz = np.ascontiguousarray(cache.reference_xyz, dtype=np.float64).reshape(
        -1
    )
    changed_xyz = changed_xyz.copy()
    assert changed_xyz.size
    changed_xyz[0] += 0.2

    evaluator._prepare_factorized_geometry(
        request["graph_generation"],
        np.ascontiguousarray(cache.reference_positions, dtype=np.float64).reshape(-1),
        changed_xyz,
        np.ascontiguousarray(np.asarray(atoms.cell), dtype=np.float64).reshape(-1),
        np.ascontiguousarray(np.linalg.inv(np.asarray(atoms.cell))).reshape(-1),
        np.asarray(atoms.pbc, dtype=np.int32),
    )
    immediate = evaluator._evaluate_polar_prepared(**request)

    perturbed = dict(request)
    perturbed_positions = np.asarray(request["positions"], dtype=np.float64).copy()
    perturbed_positions[0] += 0.01
    perturbed["positions"] = perturbed_positions
    evaluator._evaluate_polar_prepared(**perturbed)
    recomputed = evaluator._evaluate_polar_prepared(**request)

    assert immediate["energy"] == pytest.approx(
        recomputed["energy"], rel=0.0, abs=2e-12
    )


def test_fp32_public_local_energy_uses_a_double_graph_reduction(polar_model_path):
    """Only the final scalar reduction widens; learned tensors stay FP32.

    Repeating the small graph makes the rounding lost by a float graph sum
    observable without involving a calculator, backbone reverse, or finite
    differences.  The diagnostic node values are the operation's independent
    reference: summing those FP32 values in FP64 must reproduce the public
    graph scalar, while first reducing them in FP32 must not.
    """

    evaluator = _polar_calculator(polar_model_path, dtype="float32").evaluator
    base = _synthetic_chain_inputs(evaluator)
    repeats = 129
    inputs = {
        **base,
        "node_features": np.ascontiguousarray(
            np.tile(base["node_features"], (repeats, 1))
        ),
        "node_attrs": np.ascontiguousarray(np.tile(base["node_attrs"], (repeats, 1))),
        "positions": np.ascontiguousarray(np.tile(base["positions"], (repeats, 1))),
        "node_graph_index": np.zeros(3 * repeats, dtype=np.int32),
    }
    diagnostic = evaluator.polar_delta_chain(**inputs)
    public = evaluator.polar_delta_composed_forward(**inputs, compute_reverse=False)
    node_values = np.asarray(
        diagnostic["local_electron_energy"], dtype=np.float32
    ).reshape(-1)
    reference = float(np.sum(node_values, dtype=np.float64))
    narrowed = float(np.sum(node_values, dtype=np.float32))
    assert float(
        np.asarray(public["local_electron_energy"]).reshape(-1)[0]
    ) == pytest.approx(reference, abs=1e-12, rel=0.0)
    assert narrowed != reference


def test_composed_graph_outputs_use_team_parallel_reductions(polar_model_path):
    """The production graph readout must expose node- and k-parallel work.

    A graph-owned serial worker has a launch extent of one for the common
    single-graph calculator request, regardless of atom or reciprocal-grid
    size.  The selected diagnostics describe the real composed forward just
    driven here, so regressing to that launch family makes this fail.
    """

    evaluator = _polar_calculator(polar_model_path, dtype="float32").evaluator
    inputs = _synthetic_chain_inputs(evaluator)
    result = evaluator.polar_delta_composed_forward(**inputs, compute_reverse=False)
    diagnostics = evaluator.polar_delta_workspace()

    assert np.isfinite(result["energy"]).all()
    assert diagnostics["graph_output_family"] == "team_parallel_reductions"
    assert diagnostics["graph_output_quantity_count"] == 8
    assert diagnostics["graph_output_league_extent"] == 8
    assert diagnostics["graph_output_node_extent"] == 3
    assert diagnostics["graph_output_k_extent"] == diagnostics["n_k"]
    assert diagnostics["graph_output_selected_count"] == 1


@pytest.mark.parametrize(
    "graph_pbc",
    (
        pytest.param([0, 0, 0], id="molecule"),
        pytest.param([1, 1, 0], id="slab"),
        pytest.param([1, 1, 1], id="bulk"),
    ),
)
@pytest.mark.parametrize("dtype,atol", (("float32", 2e-5), ("float64", 2e-11)))
def test_team_parallel_composed_outputs_match_granular_kernels(
    polar_model_path, graph_pbc, dtype, atol
):
    """Every graph-output term matches the independent granular operations."""

    evaluator = _polar_calculator(polar_model_path, dtype=dtype).evaluator
    inputs = _synthetic_chain_inputs(evaluator)
    inputs["graph_pbc"] = np.asarray([graph_pbc], dtype=np.int32)
    inputs["external_potential"] = np.asarray(
        [[0.0, 0.07, -0.03, 0.02]], dtype=np.float64
    )
    state = evaluator.polar_delta_chain(**inputs)
    density = np.ascontiguousarray(state["charge_density"], dtype=np.float64)
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
    self_fields = None
    if not contract["include_electrostatic_self_interaction"]:
        self_fields = ops.energy_self_fields(density)
    kspace = np.asarray(ops.coulomb_energy(cache, density, self_fields))
    correction = np.asarray(
        ops.analytic_correction_energy(
            density,
            inputs["positions"],
            inputs["cell"],
            inputs["node_graph_index"],
            inputs["graph_pbc"],
        )["correction_energy"]
    )
    dipole = np.asarray(
        ops.dipole_readout(density, inputs["positions"], inputs["node_graph_index"])[
            "total_dipole"
        ]
    )
    local = np.asarray(state["local_electron_energy"], dtype=np.float64).sum(axis=0)
    field = np.sum(inputs["external_potential"][:, 1:] * dipole, axis=1)
    expected = local + kspace + correction + field

    composed = evaluator.polar_delta_composed_forward(**inputs, compute_reverse=False)
    for name, reference in (
        ("energy", expected),
        ("local_electron_energy", local),
        ("electrostatic_energy_kspace", kspace),
        ("electrostatic_energy_correction", correction),
        ("external_field_energy", field),
        ("dipole", dipole),
    ):
        np.testing.assert_allclose(
            np.asarray(composed[name]).reshape(-1),
            np.asarray(reference).reshape(-1),
            rtol=0.0,
            atol=atol,
        )


def test_fp32_chain_long_range_adjoint_matches_a_difference_and_round_trips_tables(
    polar_model_path,
):
    """The float-native long-range adjoint and its public table boundary."""

    evaluator = _polar_calculator(polar_model_path, dtype="float32").evaluator
    assert evaluator.polar_delta_shapes()["long_range_scalar_bits"] == 32
    inputs = _synthetic_chain_inputs(evaluator)
    state = evaluator.polar_delta_chain(**inputs)
    shapes = evaluator.polar_delta_shapes()
    feature_seed = np.linspace(
        0.25,
        1.0,
        np.asarray(state["potential_features"][0]).size,
    ).reshape(np.asarray(state["potential_features"][0]).shape)
    potential_seeds = [feature_seed] + [
        np.zeros((0, shapes["potential_dimension"])) for _ in range(shapes["steps"] - 1)
    ]
    empty_energy = np.zeros((0, 1))
    empty_density = np.zeros((0, shapes["charges_dimension"]))

    def loss(positions):
        driven = evaluator.polar_delta_chain(**{**inputs, "positions": positions})
        return float(np.sum(feature_seed * np.asarray(driven["potential_features"][0])))

    adjoints = evaluator.polar_delta_chain_reverse(
        **inputs,
        local_electron_energy_seed=empty_energy,
        spin_charge_density_seed=empty_density,
        potential_features_seeds=potential_seeds,
        fukui_sources_seeds=[],
    )
    positions = inputs["positions"]
    delta = np.zeros_like(positions)
    delta[0, 0] = 2e-3
    numeric = (loss(positions + delta) - loss(positions - delta)) / (4e-3)
    analytic = float(np.asarray(adjoints["positions"])[0, 0])
    assert abs(numeric) > 1e-6
    assert analytic == pytest.approx(numeric, rel=5e-3, abs=5e-4)

    # With no learned-output seeds, the caller's host-double table seeds must
    # be the materialized table adjoints. This crosses both mixed-scalar
    # boundaries of the public FP32 diagnostic reverse.
    cos_seed = np.ascontiguousarray(np.asarray(adjoints["grad_cos_table"]) * 0.25)
    sin_seed = np.ascontiguousarray(np.asarray(adjoints["grad_sin_table"]) * 0.25)
    assert np.linalg.norm(cos_seed) + np.linalg.norm(sin_seed) > 0.0
    seeded = evaluator.polar_delta_chain_reverse(
        **inputs,
        local_electron_energy_seed=empty_energy,
        spin_charge_density_seed=empty_density,
        potential_features_seeds=[],
        fukui_sources_seeds=[],
        table_seeds_cos=cos_seed.reshape(-1),
        table_seeds_sin=sin_seed.reshape(-1),
    )
    np.testing.assert_allclose(seeded["grad_cos_table"], cos_seed, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(seeded["grad_sin_table"], sin_seed, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize("reverse_strategy", ("owner", "pair_atomic"))
def test_resident_composed_chain_reverse_matches_fresh_finite_differences(
    polar_model_path, dtype, reverse_strategy
):
    """Decompose the resident reverse independently of the backbone reverse."""

    evaluator = _polar_calculator(polar_model_path, dtype=dtype).evaluator
    _select_realspace_reverse_strategy(evaluator, reverse_strategy)
    inputs = _synthetic_chain_inputs(evaluator)
    analytic = evaluator.polar_delta_composed_forward(
        **inputs, compute_reverse=True, polar_electrostatics="realspace"
    )
    step = 2.5e-3 if dtype == "float32" else 1e-5
    tolerance = 3e-3 if dtype == "float32" else 2e-5

    def fresh_energy(changed):
        fresh = _polar_calculator(polar_model_path, dtype=dtype).evaluator
        _select_realspace_reverse_strategy(fresh, reverse_strategy)
        return float(
            np.sum(
                fresh.polar_delta_composed_forward(
                    **changed,
                    compute_reverse=False,
                    polar_electrostatics="realspace",
                )["energy"]
            )
        )

    probes = (
        ("node_features", (0, 0), "node_features_adjoint", (0, 0)),
        ("positions", (0, 0), "positions_adjoint", (0, 0)),
        ("total_charge", (0,), "total_charge_adjoint", (0, 0)),
        ("external_potential", (0, 1), "external_potential_adjoint", (0, 1)),
    )
    for input_name, input_index, output_name, output_index in probes:
        plus = np.asarray(inputs[input_name], dtype=np.float64).copy()
        minus = plus.copy()
        plus[input_index] += step
        minus[input_index] -= step
        numeric = (
            fresh_energy({**inputs, input_name: plus})
            - fresh_energy({**inputs, input_name: minus})
        ) / (2.0 * step)
        value = float(np.asarray(analytic[output_name])[output_index])
        assert value == pytest.approx(numeric, abs=tolerance, rel=tolerance), (
            dtype,
            input_name,
            value,
            numeric,
        )

    strain = np.zeros((3, 3))
    strain[0, 1] = step
    plus_transform = np.eye(3) + strain
    minus_transform = np.eye(3) - strain
    plus_inputs = {
        **inputs,
        "positions": np.ascontiguousarray(inputs["positions"] @ plus_transform),
        "cell": np.ascontiguousarray(inputs["cell"] @ plus_transform),
    }
    minus_inputs = {
        **inputs,
        "positions": np.ascontiguousarray(inputs["positions"] @ minus_transform),
        "cell": np.ascontiguousarray(inputs["cell"] @ minus_transform),
    }
    numeric_strain = (fresh_energy(plus_inputs) - fresh_energy(minus_inputs)) / (
        2.0 * step
    )
    volume = abs(float(np.linalg.det(inputs["cell"][0])))
    analytic_strain = float(np.asarray(analytic["stress"])[0, 1]) * volume
    assert analytic_strain == pytest.approx(
        numeric_strain, abs=tolerance, rel=tolerance
    )
    if reverse_strategy == "pair_atomic":
        _assert_pair_atomic_was_used(evaluator)


def test_both_precisions_serve_the_family(polar_model_path):
    """The family's float32 and float64 evaluators agree on one structure."""

    results = {}
    for dtype in ("float32", "float64"):
        atoms = _molecule()
        atoms.calc = _polar_calculator(polar_model_path, dtype=dtype)
        results[dtype] = atoms.get_potential_energy()

    # The family's energy now carries the long-range composition, whose fp32
    # realization spreads a little wider than the backbone's alone: the
    # measured spread is 9.0e-5 eV on a 24 eV total (3.7e-6 relative), which is
    # the fp32 evaluation's own floor on these terms.  The bound is relative
    # and an order of magnitude above it.
    spread = abs(results["float32"] - results["float64"])
    assert spread < 1e-5 * max(1.0, abs(results["float64"]))


def _checkpoint(size: str) -> Path:
    """The released checkpoint for one size, or a skip naming the prerequisite."""

    if _ORACLE_MACE_CHECKOUT is not None and _ORACLE_MACE_CHECKOUT.is_dir():
        sys.path.insert(0, str(_ORACLE_MACE_CHECKOUT))
    if (
        _ORACLE_GRAPH_ELECTROSTATICS_CHECKOUT is not None
        and _ORACLE_GRAPH_ELECTROSTATICS_CHECKOUT.is_dir()
    ):
        sys.path.insert(0, str(_ORACLE_GRAPH_ELECTROSTATICS_CHECKOUT))
    try:
        import mace  # noqa: F401  (also sets the weights-only env e3nn requires)
        import torch  # noqa: F401
        from graph_longrange.gto_utils import gto_basis_kspace_cutoff  # noqa: F401
    except ImportError as error:  # pragma: no cover - environment-dependent
        pytest.skip(
            "the released PolarMACE checkpoints require "
            f"torch/e3nn/mace/graph_longrange: {error}"
        )

    path = os.environ.get(f"SYMMETRIX_POLAR_MODEL_{size}")
    if not path:
        pytest.skip(
            f"set SYMMETRIX_POLAR_MODEL_{size} to load the MACE-POLAR-1-{size} "
            "checkpoint through the family entry"
        )
    path = Path(path)
    if not path.is_file():
        pytest.skip(f"MACE-POLAR-1-{size} checkpoint does not exist: {path}")
    return path


@pytest.mark.parametrize("size", MODEL_SIZES)
def test_the_released_checkpoints_load_and_evaluate_through_the_family_entry(
    tmp_path, size
):
    """Detection, schema, extractor, evaluator, contract, and program, end to end.

    The released checkpoints are the family's admission gate: they carry the
    spellings and the readout classes the polar schema exists for, and they
    must reach a compiled program whose backbone scales with their own
    interaction count.  All three are admitted at the evaluator level; S and M
    then evaluate through the family's calculator (energy, forces, stress),
    while L's *composition* is refused by name: its per-layer node features are
    l<=2 (512 channels x (1+3+5) = 4608) and the model-convention conversion
    the chain's input needs is landed for l<=1 only, so L parity stays a
    follow-up record.  The refusal is asserted here, not env-skipped, so a
    silent regression to "the backbone half served against the composed energy"
    cannot hide behind a skip.
    """

    import numpy as np
    import torch

    facts = json.loads(_FACTS_PATH.read_text())["models"][size]
    path = _checkpoint(size)
    model = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    data = extract_mace_polar_data(model.to(torch.float64))

    model_path = _write_model(tmp_path / f"MACE-POLAR-1-{size}.json", data)
    entry = family_registry.declared_registry().detect(model_type=data["model_type"])
    assert entry.key == "MACE_Polar"

    evaluator_class = family_registry.resolve_evaluator_class(
        entry, dtype="float64", native=_native_module()
    )
    evaluator = evaluator_class(str(model_path))
    assert isinstance(evaluator, evaluator_class)
    assert evaluator.has_polar_long_range_contract

    nodes = _program_nodes(evaluator.family_program)
    assert evaluator.family_program["family"] == entry.capabilities.family
    assert (
        len(
            {
                node_id.split(".")[1]
                for node_id in nodes
                if node_id.startswith("mh1.interaction")
            }
        )
        == facts["interaction_count"]
    )
    assert set(nodes) & LONG_RANGE_NODES == LONG_RANGE_NODES
    assert [readout["class"] for readout in data["readouts"]] == (
        facts["readout_classes"]
    )
    assert set(family_registry.capabilities_of(entry, evaluator)) == set(
        family_registry.CAPABILITY_KEYS
    )

    if size == "L":
        with pytest.raises(RuntimeError, match="l<=2"):
            Symmetrix(str(model_path), dtype="float64", streamed_edges="auto")
        return

    calculator = Symmetrix(str(model_path), dtype="float64", streamed_edges="auto")
    assert isinstance(calculator.evaluator, evaluator_class)
    atoms = _water()
    atoms.calc = calculator
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    assert np.isfinite(energy)
    assert np.isfinite(forces).all()
    assert np.isfinite(atoms.get_stress()).all()


def test_the_released_m_checkpoint_serves_the_realspace_property_matrix(tmp_path):
    """The released M model serves every FFF property/lane/precision request."""

    import torch

    path = _checkpoint("M")
    model = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    data = extract_mace_polar_data(model.to(torch.float64))
    model_path = _write_model(tmp_path / "MACE-POLAR-1-M.json", data)
    channels = (
        ("singlet", {}),
        ("charged", {"total_charge": 1.0, "total_spin": 0.0}),
        ("external-field", {"electric_field": [0.1, -0.04, 0.02]}),
    )
    property_sets = (
        ("energy",),
        ("energy", "forces"),
        ("energy", "forces", "stress"),
    )

    for dtype in ("float32", "float64"):
        tolerance = 4.0e-5 if dtype == "float32" else 2.0e-11
        for channel_name, channel_kwargs in channels:
            forced = Symmetrix(
                str(model_path),
                dtype=dtype,
                streamed_edges="reference",
                polar_electrostatics="realspace",
                **channel_kwargs,
            )
            forced.calculate(_water(), properties=["energy", "forces", "stress"])
            expected = {
                name: np.array(forced.results[name], copy=True)
                for name in ("energy", "forces", "stress")
            }

            automatic = Symmetrix(
                str(model_path),
                dtype=dtype,
                streamed_edges="reference",
                **channel_kwargs,
            )
            for properties in property_sets:
                automatic.calculate(_water(), properties=list(properties))
                for name in properties:
                    assert np.isfinite(automatic.results[name]).all()
                    np.testing.assert_allclose(
                        automatic.results[name],
                        expected[name],
                        rtol=0.0,
                        atol=tolerance,
                        err_msg=f"{dtype}/{channel_name}/{properties}",
                    )
                diagnostics = automatic.polar_electrostatics_diagnostics
                assert diagnostics["resolved_solver"] == "realspace"
                assert diagnostics["n_k"] == 0
                assert diagnostics["reciprocal_allocation_bytes"] == 0

            workspace = automatic.evaluator.polar_delta_workspace()
            assert workspace["realspace_retained_allocation_bytes"] > 0
            assert workspace["diagnostic_state_materialization_count"] == 0
