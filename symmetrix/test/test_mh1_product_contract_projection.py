import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest
from symmetrix.mace_execution_contract import (
    normalize_generated_m1_contract,
    normalize_standard_m0_contract,
)
from symmetrix.mace_nonlinear_execution_contract import (
    _product_descriptor,
    project_execution_mh1_product_contract,
)

ROOT = Path(__file__).resolve().parents[2]


def _tensor(shape, nonzero_indices):
    size = 1
    for extent in shape:
        size *= extent
    values = [0.0] * size
    for index in nonzero_indices:
        values[index] = 1.0
    return {"shape": shape, "values": values}


def _linear(irreps, channels, output_dimension, block_count):
    instructions = []
    for index in range(block_count):
        instructions.append(
            {
                "i_in": index,
                "i_out": index,
                "path_shape": [channels, channels],
                "path_weight": 1.0,
            }
        )
    weight_count = block_count * channels * channels
    return {
        "irreps_in": irreps,
        "irreps_out": irreps,
        "instructions": instructions,
        "weight": {"shape": [weight_count], "values": [0.0] * weight_count},
        "bias": {"shape": [0], "values": []},
        "output_mask": {
            "shape": [output_dimension],
            "values": [1.0] * output_dimension,
        },
    }


def _product(channels, input_l_max, *, final):
    irreps_in = "+".join(
        f"{channels}x{l_value}{'e' if l_value % 2 == 0 else 'o'}"
        for l_value in range(input_l_max + 1)
    )
    irreps_out = f"{channels}x0e" if final else f"{channels}x0e+{channels}x1o"
    angular = (input_l_max + 1) ** 2
    contractions = []
    for output_l in [0] if final else [0, 1]:
        width = 2 * output_l + 1
        tensors = []
        for degree in range(1, 4):
            shape = ([width] if output_l else []) + [angular] * degree + [1]
            tuple_count = angular**degree
            # Each output component gets two deliberately non-lexicographic
            # source tuples.  The descriptor/native plan canonicalizes them.
            nonzero = []
            for component in range(width):
                base = component * tuple_count
                nonzero.extend([base + degree, base + tuple_count - 1])
            tensors.append(_tensor(shape, nonzero))
        weight = {
            "shape": [1, 1, channels],
            "values": [float(channel + 1) for channel in range(channels)],
        }
        contractions.append(
            {
                "correlation": 3,
                "weights": [deepcopy(weight), deepcopy(weight)],
                "weights_max": deepcopy(weight),
                "u_tensors": tensors,
            }
        )
    output_dimension = channels * (1 if final else 4)
    definition = {
        "node_feats_irreps": irreps_out,
        "target_irreps": irreps_out,
        "use_sc": True,
        "use_agnostic_product": True,
        "linear": _linear(irreps_out, channels, output_dimension, 1 if final else 2),
        "symmetric_contractions": {
            "irreps_in": irreps_in,
            "irreps_out": irreps_out,
            "contractions": contractions,
        },
    }
    return _product_descriptor(0, definition, last=final)


@pytest.mark.parametrize("layer_count", [2, 3])
@pytest.mark.parametrize("input_l_max", [2, 3])
@pytest.mark.parametrize("channels", [7, 11])
def test_mh1_product_projection_covers_every_stage_and_topology(
    layer_count, input_l_max, channels
):
    products = [
        _product(channels, input_l_max, final=index == layer_count - 1)
        for index in range(layer_count)
    ]
    contracts = [
        project_execution_mh1_product_contract(product, final=index == layer_count - 1)
        for index, product in enumerate(products)
    ]

    for index, contract in enumerate(contracts):
        final = index == layer_count - 1
        assert contract["interaction"] == ("M1" if final else "M0")
        assert contract["type_count"] == 1
        assert contract["channels"] == channels
        assert contract["input_l_max"] == input_l_max
        assert contract["output_l_max"] == (0 if final else 1)
        normalizer = (
            normalize_generated_m1_contract if final else normalize_standard_m0_contract
        )
        assert normalizer(contract) == contract


def test_mh1_product_projection_uses_logical_angular_components():
    product = _product(5, 3, final=False)
    # Runtime feature offsets are channel-strided (0, 5, 10, 15), while the
    # canonical contraction ABI numbers logical angular components (0, 1, 2, 3).
    assert [item["output_component_offset"] for item in product["components"]] == [
        0,
        5,
        10,
        15,
    ]
    contract = project_execution_mh1_product_contract(product, final=False)
    assert [group["output_component"] for group in contract["monomial_groups"]] == [
        0,
        1,
        2,
        3,
    ]
    changed = deepcopy(product)
    for index, component in enumerate(changed["components"]):
        component["output_component_offset"] = 1000 + index
    assert project_execution_mh1_product_contract(changed, final=False) == contract


def test_mh1_product_projection_matches_native_term_order(tmp_path):
    product = _product(7, 2, final=False)
    contract = project_execution_mh1_product_contract(product, final=False)
    for block in product["blocks_program"]:
        for block_component in range(block["width"]):
            logical_component = block["angular_offset"] + block_component
            component_terms = [
                term
                for term in product["terms"]
                if term["output_block"] == block["index"]
                and term["block_component"] == block_component
            ]
            native_order = [
                (term["degree"], tuple(term["angular_indices"]))
                for term in component_terms
            ]
            assert native_order == sorted(native_order)
            native_terms = [term["angular_indices"] for term in component_terms]
            assert (
                contract["monomial_groups"][logical_component]["terms"] == native_terms
            )

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for the native ordering gate")
    source = tmp_path / "monomial_order.cpp"
    source.write_text(
        """
#include "e3nn_product_monomial_key.hpp"
#include <algorithm>
#include <iostream>
#include <vector>

int main() {
    using symmetrix::detail::ProductMonomialKey;
    std::vector<ProductMonomialKey> keys{
        {3, {8, 8, 8}}, {1, {1, 0, 0}}, {2, {1, 8, 0}},
        {3, {0, 1, 8}}, {2, {0, 0, 0}}, {1, {8, 0, 0}}
    };
    std::sort(keys.begin(), keys.end());
    for (const auto& key : keys) {
        std::cout << key.degree;
        for (int axis = 0; axis < key.degree; ++axis)
            std::cout << ':' << key.indices[axis];
        std::cout << '\\n';
    }
}
"""
    )
    executable = tmp_path / "monomial_order"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-I",
            str(ROOT / "libsymmetrix/source"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )
    native_order = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    python_keys = [
        (3, (8, 8, 8)),
        (1, (1,)),
        (2, (1, 8)),
        (3, (0, 1, 8)),
        (2, (0, 0)),
        (1, (8,)),
    ]
    assert native_order == [
        ":".join(str(value) for value in (degree, *indices))
        for degree, indices in sorted(python_keys)
    ]


def test_mh1_product_projection_fingerprints_are_canonical_and_sensitive():
    product = _product(7, 2, final=False)
    first = project_execution_mh1_product_contract(product, final=False)
    assert project_execution_mh1_product_contract(product, final=False) == first
    changed = _product(11, 2, final=False)
    second = project_execution_mh1_product_contract(changed, final=False)
    assert first["semantic_fingerprint"] != second["semantic_fingerprint"]
    assert first["generation_fingerprint"] != second["generation_fingerprint"]
    assert first["structure_fingerprint"] == second["structure_fingerprint"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda product: product.update(feature_layout="mul_ir"),
            "feature layout",
        ),
        (
            lambda product: product["angular_layout"][0].update(channel_stride=2),
            "angular layout",
        ),
        (
            lambda product: product["blocks_program"][1].update(angular_offset=2),
            "block program",
        ),
        (
            lambda product: product.update(correlation=2),
            "correlation must be three",
        ),
        (
            lambda product: product["blocks_program"][0]["runtime_coefficients"].update(
                count=1
            ),
            "coefficient extent",
        ),
    ],
)
def test_mh1_product_projection_rejects_inconsistent_descriptors(mutation, message):
    product = _product(7, 2, final=False)
    mutation(product)
    with pytest.raises(ValueError, match=message):
        project_execution_mh1_product_contract(product, final=False)


def test_mh1_product_projection_rejects_final_marker_topology_mismatch():
    with pytest.raises(ValueError, match="final marker"):
        project_execution_mh1_product_contract(_product(7, 2, final=False), final=True)
    with pytest.raises(ValueError, match="final marker"):
        project_execution_mh1_product_contract(_product(7, 2, final=True), final=False)


def test_mh1_product_projection_rejects_noncanonical_term_order():
    product = _product(7, 2, final=False)
    first = product["components"][0]["term_begin"]
    product["terms"][first], product["terms"][first + 1] = (
        product["terms"][first + 1],
        product["terms"][first],
    )
    product["terms"][first]["index"] = first
    product["terms"][first + 1]["index"] = first + 1
    with pytest.raises(ValueError, match="degree-then-lexicographic"):
        project_execution_mh1_product_contract(product, final=False)


def test_mh1_product_projection_rejects_duplicate_terms():
    product = _product(7, 2, final=False)
    first = product["components"][0]["term_begin"]
    product["terms"][first + 1].update(
        degree=product["terms"][first]["degree"],
        angular_indices=product["terms"][first]["angular_indices"].copy(),
    )
    with pytest.raises(ValueError, match="duplicate"):
        project_execution_mh1_product_contract(product, final=False)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda program: program.update(parameters=0), "parameters"),
        (lambda program: program.update(u_shape=[9]), "U shape"),
        (lambda program: program.update(u_shape=[8, 1]), "angular"),
        (lambda program: program.update(weight_shape=[2, 1, 7]), "element"),
        (lambda program: program.update(weight_shape=[1, 2, 7]), "parameter"),
        (lambda program: program.update(weight_shape=[1, 1, 8]), "channel"),
    ],
)
def test_mh1_product_projection_rejects_corrupt_degree_program(mutation, message):
    product = _product(7, 2, final=False)
    mutation(product["blocks_program"][0]["degree_programs"][0])
    with pytest.raises(ValueError, match=message):
        project_execution_mh1_product_contract(product, final=False)


def test_mh1_product_projection_rejects_nonmapping_degree_program():
    product = _product(7, 2, final=False)
    product["blocks_program"][0]["degree_programs"][0] = None
    with pytest.raises(ValueError, match="degree programs"):
        project_execution_mh1_product_contract(product, final=False)


def test_mh1_product_projection_rejects_vector_output_axis_width():
    product = _product(7, 2, final=False)
    product["blocks_program"][1]["degree_programs"][0]["u_shape"][0] = 2
    with pytest.raises(ValueError, match="output width"):
        project_execution_mh1_product_contract(product, final=False)


def test_mh1_product_projection_rejects_missing_weight_shape():
    product = _product(7, 2, final=False)
    product["blocks_program"][0]["degree_programs"][0]["weight_shape"] = None
    with pytest.raises(ValueError, match="weight shape"):
        project_execution_mh1_product_contract(product, final=False)
