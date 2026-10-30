"""Lane-major addressing for the Kokkos E3 linear kernel.

The node-major kernels store block element (channel a, component x) of a
block at offset o with multiplicity m and angular width d at o + a*d + x.
The lane-major kernels store the same element at o + x*m + a.  Both forms
must agree once the block indices are permuted accordingly.
"""

import json

import numpy as np
import pytest

try:
    from symmetrix import symmetrix as native_symmetrix
except Exception as _error:  # pragma: no cover - environment without backend
    pytest.skip(
        f"native Symmetrix package is unavailable: {_error}", allow_module_level=True
    )


IRREPS = "2x0e+3x1o+2x2e"
BLOCKS = ((2, 0), (3, 1), (2, 2))


def _lane_to_node_permutation(blocks):
    """Index permutation so that node_major[:, permutation] is lane major."""
    permutation = []
    offset = 0
    for multiplicity, angular in blocks:
        width = 2 * angular + 1
        block = [0] * (multiplicity * width)
        for channel in range(multiplicity):
            for component in range(width):
                block[component * multiplicity + channel] = channel * width + component
        permutation.extend(offset + index for index in block)
        offset += multiplicity * width
    return np.asarray(permutation, dtype=np.intp)


def _to_lane(node_major, blocks=BLOCKS):
    """Permute node-major rows to lane-major for the given block layout."""
    return np.asarray(node_major)[:, _lane_to_node_permutation(blocks)]


def _linear_definition(rng):
    # bias and output_mask are definition-order (node-major) tables.  Each
    # kernel must index them at the node-major position of the element it
    # touches, so position-dependent values expose any lane/node mix-up that
    # block-constant parameters would hide.
    instructions = []
    weights = []
    bias = []
    output_mask = []
    for index, (multiplicity, angular) in enumerate(BLOCKS):
        instructions.append(
            {
                "i_in": index,
                "i_out": index,
                "path_weight": float(rng.uniform(-2.0, 2.0)),
                "path_shape": [multiplicity, multiplicity],
            }
        )
        weights.append(rng.normal(size=multiplicity * multiplicity))
        size = multiplicity * (2 * angular + 1)
        bias.append(rng.normal(size=size))
        output_mask.append(rng.normal(size=size))
    weights = np.concatenate(weights)
    bias = np.concatenate(bias)
    output_mask = np.concatenate(output_mask)
    return {
        "irreps_in": IRREPS,
        "irreps_out": IRREPS,
        "instructions": instructions,
        "weight": {"shape": [weights.size], "values": weights.tolist()},
        "bias": {"shape": [bias.size], "values": bias.tolist()},
        "output_mask": {"shape": [output_mask.size], "values": output_mask.tolist()},
    }


def test_lane_kernels_interpret_their_input_as_lane_major():
    """The lane entry point must not accept node-major rows.

    If a lane kernel ignored the block permutation, its node-major and
    lane-major results would coincide and the parity tests below would silently
    become tautologies.  Both schedules are checked: the per-block definition
    and the float32 definition whose adjacent same-l blocks the host path
    coalesces.
    """

    if not hasattr(native_symmetrix, "E3LinearKokkos"):
        pytest.skip("Symmetrix was built without Kokkos E3 primitives")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    generator = np.random.default_rng(20240918)
    samples = 6

    module = native_symmetrix.E3LinearKokkos(json.dumps(_linear_definition(generator)))
    node_input = generator.normal(size=(samples, module.input_dimension))
    node_output = np.asarray(
        module.evaluate_batch(node_input.ravel(), samples)
    ).reshape(samples, -1)
    lane_output = np.asarray(
        module.evaluate_lane_batch(node_input.ravel(), samples)
    ).reshape(samples, -1)
    assert not np.allclose(lane_output, node_output, rtol=1e-6, atol=1e-8)

    float_module = native_symmetrix.E3LinearKokkosFloat(
        json.dumps(_float32_merge_definition(generator))
    )
    float_input = generator.normal(size=(samples, float_module.input_dimension))
    float_node_output = np.asarray(
        float_module.evaluate_batch(float_input.ravel().tolist(), samples)
    ).reshape(samples, -1)
    float_lane_output = np.asarray(
        float_module.evaluate_lane_batch(float_input.ravel().tolist(), samples)
    ).reshape(samples, -1)
    assert not np.allclose(float_lane_output, float_node_output, rtol=1e-5, atol=1e-6)


# Two adjacent same-angular input blocks that the float32 host path coalesces
# into one linear instruction.  The merged instruction is only valid for the
# node-major schedule; lane addressing must iterate the unmerged list.
FLOAT_IRREPS_IN = "2x1o+3x1o"
FLOAT_IRREPS_OUT = "5x1o"
FLOAT_BLOCKS_IN = ((2, 1), (3, 1))
FLOAT_BLOCKS_OUT = ((5, 1),)


def _float32_merge_definition(rng):
    in_weights = rng.normal(size=2 * 5).astype(np.float32)
    out_weights = rng.normal(size=3 * 5).astype(np.float32)
    weights = np.concatenate([in_weights, out_weights])
    # The output block is 5x1o, so bias and mask span multiplicity * width.
    bias = rng.normal(size=5 * 3).astype(np.float32)
    output_mask = rng.normal(size=5 * 3).astype(np.float32)
    return {
        "irreps_in": FLOAT_IRREPS_IN,
        "irreps_out": FLOAT_IRREPS_OUT,
        "instructions": [
            {
                "i_in": 0,
                "i_out": 0,
                "path_weight": 1.5,
                "path_shape": [2, 5],
            },
            {
                "i_in": 1,
                "i_out": 0,
                "path_weight": 1.5,
                "path_shape": [3, 5],
            },
        ],
        "weight": {"shape": [weights.size], "values": weights.tolist()},
        "bias": {"shape": [bias.size], "values": bias.tolist()},
        "output_mask": {"shape": [output_mask.size], "values": output_mask.tolist()},
    }


def _polar_linear_definition(
    rng,
    irreps_in,
    irreps_out,
    input_blocks,
    output_blocks,
    role,
):
    """Build an exact, role-labelled polar E3Linear fixture.

    The expected contractions below are derived directly from these literal
    block shapes.  In particular, the unequal-block mixer catches a selector
    that assumes every angular block has the first block's multiplicity.
    """

    instructions = []
    weights = []
    for input_index, (input_multiplicity, input_l) in enumerate(input_blocks):
        for output_index, (output_multiplicity, output_l) in enumerate(output_blocks):
            if input_l != output_l:
                continue
            instructions.append(
                {
                    "i_in": input_index,
                    "i_out": output_index,
                    "path_weight": 1.0,
                    "path_shape": [input_multiplicity, output_multiplicity],
                }
            )
            weights.extend(rng.normal(size=input_multiplicity * output_multiplicity))
    output_dimension = sum(
        multiplicity * (2 * angular + 1) for multiplicity, angular in output_blocks
    )
    return {
        "irreps_in": irreps_in,
        "irreps_out": irreps_out,
        "instructions": instructions,
        "weight": {"shape": [len(weights)], "values": weights},
        "bias": {"shape": [0], "values": []},
        "output_mask": {
            "shape": [output_dimension],
            "values": [1.0] * output_dimension,
        },
        "execution_specialization": {
            "family": "polar_e3linear",
            "role": role,
        },
    }


@pytest.mark.parametrize(
    ("class_name", "dtype", "rtol", "atol"),
    [
        ("E3LinearKokkos", np.float64, 2e-12, 2e-12),
        ("E3LinearKokkosFloat", np.float32, 3e-4, 3e-5),
    ],
)
@pytest.mark.parametrize(
    ("definition_args", "expected_family"),
    [
        (
            (
                "512x0e+384x1o",
                "512x0e+384x1o",
                ((512, 0), (384, 1)),
                ((512, 0), (384, 1)),
                "feature_mixer",
            ),
            "polar_e3linear",
        ),
        (
            (
                "512x0e+383x1o",
                "512x0e+383x1o",
                ((512, 0), (383, 1)),
                ((512, 0), (383, 1)),
                "feature_mixer",
            ),
            "generic",
        ),
        (
            (
                "512x0e+1x1o",
                "512x0e+1x1o",
                ((512, 0), (1, 1)),
                ((512, 0), (1, 1)),
                "feature_mixer",
            ),
            "generic",
        ),
        (
            ("512x0e", "10x0e", ((512, 0),), ((10, 0),), "update_readout_final"),
            "polar_e3linear",
        ),
        (
            ("511x0e", "10x0e", ((511, 0),), ((10, 0),), "update_readout_final"),
            "generic",
        ),
    ],
)
def test_polar_specialization_selects_only_validated_shapes(
    class_name, dtype, rtol, atol, definition_args, expected_family
):
    """Wrong role extents must fall back while accepted shapes stay exact.

    Removing either the role-specific shape validation or the unequal-block
    indexing makes this test fail: the former selects the 511-channel near
    miss, while the latter disagrees with the independently callable generic
    path for the 512/384 mixer.
    """

    if not hasattr(native_symmetrix, class_name):
        pytest.skip(f"Symmetrix was built without {class_name}")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    rng = np.random.default_rng(9173)
    definition = _polar_linear_definition(rng, *definition_args)
    module = getattr(native_symmetrix, class_name)(json.dumps(definition))
    samples = 7
    values = rng.normal(size=(samples, module.input_dimension)).astype(dtype)
    seed = rng.normal(size=(samples, module.output_dimension)).astype(dtype)

    assert module.specialization_family == expected_family
    assert module.selected_family(samples) == expected_family
    specialized = np.asarray(
        module.evaluate_batch(values.ravel().tolist(), samples), dtype=dtype
    ).reshape(samples, -1)
    reference = np.asarray(
        module.evaluate_generic_batch(values.ravel().tolist(), samples), dtype=dtype
    ).reshape(samples, -1)
    specialized_reverse = np.asarray(
        module.reverse_batch(seed.ravel().tolist(), samples), dtype=dtype
    ).reshape(samples, -1)
    reference_reverse = np.asarray(
        module.reverse_generic_batch(seed.ravel().tolist(), samples), dtype=dtype
    ).reshape(samples, -1)
    np.testing.assert_allclose(specialized, reference, rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        specialized_reverse, reference_reverse, rtol=rtol, atol=atol
    )
    if expected_family == "polar_e3linear":
        assert module.specialized_forward_count == 1
        assert module.specialized_reverse_count == 1
        assert module.specialization_fallback_count == 0
    else:
        assert module.specialized_forward_count == 0
        assert module.specialized_reverse_count == 0
        assert module.specialization_fallback_count == 2


def test_rejected_polar_calls_do_not_increment_execution_counters():
    """Telemetry counts only dimension-valid executions."""

    if not hasattr(native_symmetrix, "E3LinearKokkos"):
        pytest.skip("Symmetrix was built without Kokkos E3 primitives")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    rng = np.random.default_rng(104729)
    definition = _polar_linear_definition(
        rng,
        "512x0e",
        "10x0e",
        ((512, 0),),
        ((10, 0),),
        "update_readout_final",
    )
    module = native_symmetrix.E3LinearKokkos(json.dumps(definition))
    samples = 2
    values = rng.normal(size=(samples, module.input_dimension))
    seed = rng.normal(size=(samples, module.output_dimension))

    with pytest.raises(ValueError, match="batch dimensions"):
        module._evaluate_with_output_width_for_testing(
            values.ravel().tolist(), samples, module.output_dimension - 1
        )
    with pytest.raises(ValueError, match="reverse batch dimensions"):
        module._reverse_with_input_width_for_testing(
            seed.ravel().tolist(), samples, module.input_dimension - 1
        )
    assert module.specialized_forward_count == 0
    assert module.specialized_reverse_count == 0
    assert module.specialization_fallback_count == 0


@pytest.mark.parametrize(
    ("class_name", "dtype", "rtol", "atol", "epsilon"),
    [
        ("E3LinearKokkos", np.float64, 2e-8, 2e-9, 1e-6),
        ("E3LinearKokkosFloat", np.float32, 2e-2, 2e-3, 2e-3),
    ],
)
def test_polar_specialization_is_stable_and_its_reverse_is_a_finite_difference(
    class_name, dtype, rtol, atol, epsilon
):
    """A stale scratch buffer or a non-transposed reverse fails this gate."""

    if not hasattr(native_symmetrix, class_name):
        pytest.skip(f"Symmetrix was built without {class_name}")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    rng = np.random.default_rng(61109)
    definition = _polar_linear_definition(
        rng,
        "512x0e",
        "10x0e",
        ((512, 0),),
        ((10, 0),),
        "update_readout_final",
    )
    module = getattr(native_symmetrix, class_name)(json.dumps(definition))
    samples = 5
    values = rng.normal(size=(samples, 512)).astype(dtype)
    direction = rng.normal(size=values.shape).astype(dtype)
    seed = rng.normal(size=(samples, 10)).astype(dtype)

    first = np.asarray(
        module.evaluate_batch(values.ravel().tolist(), samples), dtype=dtype
    ).reshape(samples, 10)
    # Alternate the generic and specialized entry points.  Both reuse the
    # module-owned scratch, so this catches cross-route stale state.
    module.evaluate_generic_batch((-values).ravel().tolist(), samples)
    second = np.asarray(
        module.evaluate_batch(values.ravel().tolist(), samples), dtype=dtype
    ).reshape(samples, 10)
    np.testing.assert_allclose(second, first, rtol=0, atol=0)

    adjoint = np.asarray(
        module.reverse_batch(seed.ravel().tolist(), samples), dtype=dtype
    ).reshape(samples, 512)
    plus = np.asarray(
        module.evaluate_batch(
            (values + dtype(epsilon) * direction).ravel().tolist(), samples
        ),
        dtype=dtype,
    ).reshape(samples, 10)
    minus = np.asarray(
        module.evaluate_batch(
            (values - dtype(epsilon) * direction).ravel().tolist(), samples
        ),
        dtype=dtype,
    ).reshape(samples, 10)
    finite_difference = np.sum(seed * (plus - minus)) / (2 * epsilon)
    analytic = np.sum(adjoint * direction)
    np.testing.assert_allclose(analytic, finite_difference, rtol=rtol, atol=atol)
    assert module.specialized_forward_count == 4
    assert module.specialized_reverse_count == 1
    assert module.specialization_fallback_count == 0


@pytest.mark.parametrize("samples", [97, 3])
def test_kokkos_linear_lane_matches_node_major(samples):
    if not hasattr(native_symmetrix, "E3LinearKokkos"):
        pytest.skip("Symmetrix was built without Kokkos E3 primitives")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    rng = np.random.default_rng(20240 + samples)
    module = native_symmetrix.E3LinearKokkos(json.dumps(_linear_definition(rng)))

    node_input = rng.normal(size=(samples, module.input_dimension))
    node_seed = rng.normal(size=(samples, module.output_dimension))

    node_output = np.asarray(
        module.evaluate_batch(node_input.ravel(), samples)
    ).reshape(samples, -1)
    node_adjoint = np.asarray(module.reverse_batch(node_seed.ravel(), samples)).reshape(
        samples, -1
    )

    lane_output = np.asarray(
        module.evaluate_lane_batch(_to_lane(node_input).ravel(), samples)
    ).reshape(samples, -1)
    lane_adjoint = np.asarray(
        module.reverse_lane_batch(_to_lane(node_seed).ravel(), samples)
    ).reshape(samples, -1)

    np.testing.assert_allclose(
        lane_output, _to_lane(node_output), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        lane_adjoint, _to_lane(node_adjoint), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("samples", [97, 3])
def test_kokkos_float32_lane_matches_node_major_for_adjacent_blocks(samples):
    """The float32 host schedule coalesces adjacent same-l instructions.

    A coalesced instruction is only valid for node-major addressing; the lane
    kernels must iterate the unmerged instruction list, so this definition
    would mismatch if lane addressing reused the merged schedule.
    """
    if not hasattr(native_symmetrix, "E3LinearKokkosFloat"):
        pytest.skip("Symmetrix was built without float Kokkos E3 primitives")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    rng = np.random.default_rng(90210 + samples)
    module = native_symmetrix.E3LinearKokkosFloat(
        json.dumps(_float32_merge_definition(rng))
    )
    assert module.input_dimension == 2 * 3 + 3 * 3
    assert module.output_dimension == 5 * 3

    node_input = rng.normal(size=(samples, module.input_dimension))
    node_seed = rng.normal(size=(samples, module.output_dimension))
    node_output = np.asarray(
        module.evaluate_batch(node_input.ravel().tolist(), samples)
    ).reshape(samples, -1)
    node_adjoint = np.asarray(
        module.reverse_batch(node_seed.ravel().tolist(), samples)
    ).reshape(samples, -1)

    lane_output = np.asarray(
        module.evaluate_lane_batch(
            _to_lane(node_input, FLOAT_BLOCKS_IN).ravel().tolist(), samples
        )
    ).reshape(samples, -1)
    lane_adjoint = np.asarray(
        module.reverse_lane_batch(
            _to_lane(node_seed, FLOAT_BLOCKS_OUT).ravel().tolist(), samples
        )
    ).reshape(samples, -1)

    np.testing.assert_allclose(
        lane_output,
        _to_lane(node_output, FLOAT_BLOCKS_OUT),
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        lane_adjoint,
        _to_lane(node_adjoint, FLOAT_BLOCKS_IN),
        rtol=1e-4,
        atol=1e-4,
    )


def _block_layout(blocks):
    """Definition-order (offset, multiplicity, width) for each block."""
    layout = []
    offset = 0
    for multiplicity, angular in blocks:
        width = 2 * angular + 1
        layout.append((offset, multiplicity, width))
        offset += multiplicity * width
    return layout


def _linear_node_major_reference(definition, node_input):
    """Independent NumPy node-major linear, before and after finalization.

    The contraction is rebuilt from the JSON definition alone.  For each
    instruction the weight slice is applied over the source multiplicity axis
    of every angular component, then the definition-order bias and output mask
    are applied once per node-major element.  Nothing here goes through the
    node-major kernel, so the lane comparison below cannot agree by sharing a
    common (possibly lane-ordered) finalization step.
    """
    layout = _block_layout(BLOCKS)
    weights = np.asarray(definition["weight"]["values"], dtype=np.float64)
    bias = np.asarray(definition["bias"]["values"], dtype=np.float64)
    output_mask = np.asarray(definition["output_mask"]["values"], dtype=np.float64)
    samples = node_input.shape[0]
    output = np.zeros((samples, sum(m * w for _, m, w in layout)))
    weight_offset = 0
    for instruction in definition["instructions"]:
        in_offset, in_multiplicity, width = layout[instruction["i_in"]]
        out_offset, out_multiplicity, _ = layout[instruction["i_out"]]
        path_shape = tuple(int(extent) for extent in instruction["path_shape"])
        assert path_shape == (in_multiplicity, out_multiplicity)
        size = int(np.prod(path_shape))
        weight = weights[weight_offset : weight_offset + size].reshape(path_shape)
        weight = weight * float(instruction["path_weight"])
        weight_offset += size
        for component in range(width):
            source = node_input[
                :, in_offset + np.arange(in_multiplicity) * width + component
            ]
            output[:, out_offset + np.arange(out_multiplicity) * width + component] += (
                source @ weight
            )
    return output, (output + bias) * output_mask


@pytest.mark.parametrize("samples", [37, 3])
def test_kokkos_linear_lane_finalization_uses_definition_order_tables(samples):
    """Lane-major finalization must read bias/mask in definition order.

    ``finalize_lane_output`` stores a block element at o + x*m + a while the
    bias and output_mask tables stay in node-major order (o + a*d + x), so
    every entry has to be indexed at the node-major position of the lane it
    finalizes.  A lane-ordered or block-constant finalization still looks
    plausible, so the lane row is checked against an independently contracted
    NumPy node-major reference built from position-dependent bias and mask
    values.  ``samples=3`` reaches the small-batch path that calls
    ``finalize_lane_output`` directly on the host; ``samples=37`` reaches the
    fused unpack, and both are the scalar lane path under CUDA.
    """
    if not hasattr(native_symmetrix, "E3LinearKokkos"):
        pytest.skip("Symmetrix was built without Kokkos E3 primitives")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    rng = np.random.default_rng(20260112 + samples)
    definition = _linear_definition(rng)
    module = native_symmetrix.E3LinearKokkos(json.dumps(definition))
    node_input = rng.normal(size=(samples, module.input_dimension))

    raw_reference, node_reference = _linear_node_major_reference(definition, node_input)
    permutation = _lane_to_node_permutation(BLOCKS)
    expected_lane = node_reference[:, permutation]

    lane_output = np.asarray(
        module.evaluate_lane_batch(_to_lane(node_input).ravel(), samples)
    ).reshape(samples, -1)
    np.testing.assert_allclose(lane_output, expected_lane, rtol=1e-12, atol=1e-12)

    # Discriminating power: the tables are position dependent and the lane
    # permutation is non-trivial, so a finalization that reads bias/mask at the
    # lane position (a lane/node mix-up) produces a different row and the
    # assertion above must fail.
    bias = np.asarray(definition["bias"]["values"], dtype=np.float64)
    output_mask = np.asarray(definition["output_mask"]["values"], dtype=np.float64)
    assert np.any(permutation != np.arange(permutation.size))
    lane_indexed = (raw_reference[:, permutation] + bias) * output_mask
    assert not np.allclose(expected_lane, lane_indexed, rtol=1e-8, atol=1e-8)
