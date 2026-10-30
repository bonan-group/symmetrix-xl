"""MH-1 node generation for the bias readout.

The last layer's readout has two admitted nonlinear forms: the plain block, one
activation between its two linears, and the bias variant, which carries a second
hidden layer (``linear_mid``) between them.  The released PolarMACE checkpoints
read out through the bias variant, and the node-generation contract, the
generated host readout, the native node runtime's readout pack, and the
generated device node program must all serve it -- otherwise a checkpoint's

Everything here is built from a synthetic MH-1 model, so the gate needs no
checkpoint: the model takes the bias variant on its last layer (with the plain
block's own weights copied into the two linears both forms share), and the
generated node program -- which is the one the direct routing executes on this
host -- is compared against the family's own reference routing on the same
graph.  The device generation renders the same three-line stack over the
arena the plain form budgets, and its source is checked for the middle
linear's application and both first-linear recomputes the reverse needs.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import symmetrix

if not symmetrix._kokkos_is_initialized():
    symmetrix._init_kokkos()

#: The torch stack is imported after ``symmetrix`` (and through ``mace``
#: first), the order the other checkpoint-driven modules use: importing
#: ``e3nn`` ahead of it loads its Wigner constants with the weights-only
#: default that the released checkpoints predate.
try:  # pragma: no cover - environment-dependent import guard
    from ase import Atoms

    from symmetrix.extract_mace_nonlinear import extract_mace_nonlinear_data
    from symmetrix.mace_nonlinear_execution_contract import (
        make_execution_mh1_v4_contract,
    )

    import torch
    from e3nn import o3
    from mace.modules.blocks import NonLinearBiasReadoutBlock
    from mace.modules.blocks import NonLinearReadoutBlock
    from mace.modules.blocks import RealAgnosticResidualNonLinearInteractionBlock
    from mace.modules.models import ScaleShiftMACE

    mace_import_error = None
except ImportError as exc:  # pragma: no cover - environment-dependent
    mace_import_error = exc


#: The synthetic topology: a two-interaction MH-1 backbone with a bias readout
#: on its last layer.  Small enough to generate and run in seconds, wide enough
#: that the readout's middle linear is not a degenerate 1x1.
PARAMETERS = (4, 2, 4, 2, [])


def _bias_readout_model():
    """Build the synthetic backbone with the bias readout on its last layer."""

    if mace_import_error is not None:
        pytest.skip(f"mace-torch is not available: {mace_import_error}")
    node_channels, edge_channels, num_bessel, l_max, radial_mlp = PARAMETERS
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(16)
        model = ScaleShiftMACE(
            atomic_inter_scale=1.0,
            atomic_inter_shift=0.0,
            r_max=4.0,
            num_bessel=num_bessel,
            num_polynomial_cutoff=5,
            max_ell=l_max,
            interaction_cls=RealAgnosticResidualNonLinearInteractionBlock,
            interaction_cls_first=RealAgnosticResidualNonLinearInteractionBlock,
            num_interactions=2,
            num_elements=2,
            hidden_irreps=o3.Irreps(f"{node_channels}x0e+{node_channels}x1o"),
            MLP_irreps=o3.Irreps("4x0e"),
            atomic_energies=np.zeros((1, 2)),
            avg_num_neighbors=4.0,
            atomic_numbers=[1, 14],
            correlation=3,
            gate=torch.nn.functional.silu,
            use_agnostic_product=True,
            edge_irreps=o3.Irreps(f"{edge_channels}x0e+{edge_channels}x1o"),
            use_edge_irreps_first=True,
            radial_MLP=radial_mlp,
            heads=["test"],
        )
        plain = model.readouts[-1]
        assert isinstance(plain, NonLinearReadoutBlock)
        bias = NonLinearBiasReadoutBlock(
            irreps_in=plain.linear_1.irreps_in,
            MLP_irreps=plain.linear_1.irreps_out,
            gate=torch.nn.functional.silu,
        )
        # The two forms share their first and last linear's weights; the bias
        # variant's own middle layer stays at its initialization, which is what
        # makes the generated readout's ``linear_mid`` packing observable.  Only
        # the weights are copied: the bias variant's linears carry the biases
        # the plain block's do not.
        for target, source in (
            (bias.linear_1, plain.linear_1),
            (bias.linear_2, plain.linear_2),
        ):
            assert target.weight.shape == source.weight.shape
            with torch.no_grad():
                target.weight.copy_(source.weight)
        model.readouts[-1] = bias
    finally:
        torch.set_default_dtype(previous_dtype)
    return model


@pytest.fixture(scope="module")
def bias_model_path(tmp_path_factory):
    model = _bias_readout_model()
    data = extract_mace_nonlinear_data(model)
    path = tmp_path_factory.mktemp("mh1-bias") / "model.json"
    path.write_text(json.dumps(data, separators=(",", ":")))
    return path, data


def _atoms():
    return Atoms(
        numbers=[14, 1, 14],
        positions=[[0.0, 0.0, 0.0], [1.7, 0.2, 0.1], [0.3, 1.8, 0.4]],
        cell=[7.0, 7.0, 7.0],
        pbc=True,
    )


def test_the_node_contract_describes_the_bias_readout(bias_model_path):
    """The last layer's readout is the bias form, and its middle layer is packed."""

    path, data = bias_model_path
    assert [readout["class"] for readout in data["readouts"]] == [
        "LinearReadoutBlock",
        "NonLinearBiasReadoutBlock",
    ]
    contract = make_execution_mh1_v4_contract(data)
    readout = contract["node_program"]["layers"][-1]["readout"]
    assert readout["class"] == "NonLinearBiasReadoutBlock"
    assert "linear_mid" in readout
    assert (
        readout["linear_mid"]["irreps"]["input"]
        == readout["linear_mid"]["irreps"]["output"]
        == readout["linear_1"]["irreps"]["output"]
    )
    segments = [
        segment["name"]
        for segment in contract["node_program"]["layers"][-1][
            "runtime_parameter_packs"
        ]["readout"]["segments"]
    ]
    assert segments == ["linear_1", "linear_mid", "linear_2"]


def test_the_bias_readout_model_admits_the_generated_node_program(bias_model_path):
    """The widened readout rule admits the model to the streamed tier."""

    path, _ = bias_model_path
    evaluator = symmetrix.symmetrix.MACENonlinearKokkos(str(path))
    assert evaluator.supports_streamed_edges
    assert evaluator.supports_factorized
    assert evaluator.mh1_fast_path_rejection_reason == ""


def _generated_jit_target_is_executable():
    """Whether the generated node program this module executes can be prepared.

    The direct routing prepares the artifact its build's default JIT target
    serves: the host node program on a host build, the device node program on
    a device build.  Both generations render the bias readout, so the
    generated-execution parity runs on either backend.
    """

    query = getattr(symmetrix, "_kokkos_default_execution_space", None)
    execution_space = query() if callable(query) else None
    return execution_space is not None


@pytest.mark.parametrize("dtype", ("float64", "float32"))
def test_the_generated_readout_matches_the_reference_routing(bias_model_path, dtype):
    """The direct routing's generated execution matches the artifact-free tier.

    The host's generated node program is the float32 one, so the float64 case
    qualifies the artifact's published node-program identity (the plugin is
    prepared and loaded, and its declared readout pack has to match the native
    runtime's) while the float32 case exercises the generated readout itself.
    """

    if not _generated_jit_target_is_executable():
        pytest.skip(
            "the direct routing requires a JIT artifact this build cannot "
            "prepare (no Kokkos execution space reported)"
        )
    path, _ = bias_model_path
    atoms = _atoms()
    results = {}
    for mode in ("reference", "direct"):
        calculator = symmetrix.Symmetrix(str(path), dtype=dtype, streamed_edges=mode)
        assert calculator.streamed_edges == mode
        atoms.calc = calculator
        calculator.calculate(atoms, properties=["energy", "forces"])
        results[mode] = {
            "energy": float(calculator.results["energy"]),
            "forces": np.array(calculator.results["forces"], dtype=np.float64),
        }
        assert calculator.evaluator.factorized_fallback_evaluation_count == 0
    assert results["direct"]["energy"] == pytest.approx(
        results["reference"]["energy"], rel=1e-6 if dtype == "float32" else 1e-12
    )
    tolerance = 1e-5 if dtype == "float32" else 1e-10
    np.testing.assert_allclose(
        results["direct"]["forces"],
        results["reference"]["forces"],
        rtol=0.0,
        atol=tolerance,
    )


def test_the_device_node_program_renders_the_bias_readout(bias_model_path):
    """The device generation serves the bias readout's three-line stack.

    Rendering the device generation is pure source synthesis: it needs no GPU.
    The source has to apply the middle linear between the two activations in
    the forward, and the reverse has to recompute the first pre-activation
    twice -- once to reach the middle linear's adjoint, once for the first
    activation's own adjoint -- the same choreography the host form uses.
    """

    path, data = bias_model_path
    from symmetrix.mace_nonlinear_jit_codegen import (
        render_execution_mh1_cuda_module_v4,
    )
    from symmetrix.mace_nonlinear_execution_contract import (
        normalize_execution_mh1_v4_contract,
    )

    contract = normalize_execution_mh1_v4_contract(make_execution_mh1_v4_contract(data))
    source = render_execution_mh1_cuda_module_v4(contract, 90)
    for stem in (
        "l1_readout1",
        "l1_readoutmid",
        "l1_readout2",
        "l1_readout_activation1",
        "l1_readout_activation2",
        "l1_readout1_recompute",
        "l1_readoutmid_recompute",
        "l1_readoutmid_reverse",
        "l1_readout1_recompute2",
        "l1_readout1_reverse",
    ):
        assert stem in source, stem
    assert "hidden_mid" in source
    assert "readout_output_adjoint" in source


if __name__ == "__main__":  # pragma: no cover - manual runner
    raise SystemExit(pytest.main([__file__, "-q"]))
