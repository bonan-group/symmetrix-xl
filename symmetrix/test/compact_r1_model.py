import argparse
import json
from pathlib import Path

import numpy as np


def foundation_r1_test_model(
    contract, atomic_numbers=(1,), m0_contract=None, r0_contract=None
):
    atomic_numbers = list(atomic_numbers)
    num_elements = len(atomic_numbers)
    channels = contract["channels"]
    embedding = contract["radial_embedding"]
    edge_l_max = contract["edge_harmonics"]["l_max"]
    source_l_max = contract["source_harmonics"]["l_max"]
    paths = contract["paths"]
    terms = contract["sparse_coupling"]["terms"]
    num_edge_components = (edge_l_max + 1) ** 2
    num_source_components = (source_l_max + 1) ** 2
    cutoff = 4.5

    def repeated_diagonal(count, scale):
        values = [0.0] * (count * channels * channels)
        for block in range(count):
            block_scale = scale / (block + 1)
            offset = block * channels * channels
            for channel in range(channels):
                values[offset + channel * channels + channel] = block_scale
        return values

    def radial_network(output_width, scale):
        weights = [0.0] * (output_width * embedding)
        for output in range(output_width):
            row = output * embedding
            weights[row + output % embedding] = scale * (1 + output % 5)
            weights[row + (7 * output + 3) % embedding] += scale * 0.25
        return {
            "shape": [embedding, output_width],
            "weights": [weights],
            "activation": "silu",
            "activation_scale": 1.0,
        }

    path_multiplicities = [
        sum(path["output_l"] == l_value for path in paths)
        for l_value in range(edge_l_max + 1)
    ]
    if m0_contract is None:
        m0_monomials = {
            str(component): [[component % num_edge_components]]
            for component in range(num_source_components)
        }
    else:
        if m0_contract["channels"] != channels:
            raise ValueError("M0 and R1 contracts must use the same channel count")
        if m0_contract["type_count"] != num_elements:
            raise ValueError("M0 contract type count must match atomic_numbers")
        if m0_contract["input_l_max"] != edge_l_max:
            raise ValueError("M0 input l_max must match the R1 edge l_max")
        if m0_contract["output_l_max"] != source_l_max:
            raise ValueError("M0 output l_max must match the R1 source l_max")
        m0_monomials = {
            str(group["output_component"]): group["terms"]
            for group in m0_contract["monomial_groups"]
        }
    m0_weights = {
        str(element): {
            output: {
                str(channel): [
                    0.03
                    * (1 + element)
                    * (1 + (term + channel) % 5)
                    / (1 + int(output))
                    for term in range(len(monomials))
                ]
                for channel in range(channels)
            }
            for output, monomials in m0_monomials.items()
        }
        for element in range(num_elements)
    }
    m1_monomials = [[component] for component in range(num_edge_components)]
    m1_weights_for_element = {
        str(channel): [
            0.2 / (1 + component % 4) for component in range(num_edge_components)
        ]
        for channel in range(channels)
    }
    m1_weights = {
        str(element): m1_weights_for_element for element in range(num_elements)
    }
    readout_2_weights_1 = [0.0] * (16 * channels)
    for hidden in range(16):
        readout_2_weights_1[hidden * channels + hidden % channels] = 0.4

    return {
        "symmetrix_format_version": 2,
        "radial_representation": "compact",
        "model_type": "MACE",
        "num_elements": num_elements,
        "num_channels": channels,
        "r_cut": cutoff,
        "l_max": edge_l_max,
        "L_max": source_l_max,
        "atomic_numbers": atomic_numbers,
        "atomic_energies": [0.0] * num_elements,
        "H0_weights": [
            0.8 + 0.02 * (channel % 5)
            for _ in range(num_elements)
            for channel in range(channels)
        ],
        "A0_weights": [
            [
                repeated_diagonal(1, 0.4 + 0.02 * l_value)
                for l_value in range(edge_l_max + 1)
            ]
            for _ in range(num_elements)
        ],
        "A0_scaled": False,
        "compact_radial": {
            "spline_grid_min": 1e-12,
            "num_spline_points": 16,
            "basis": {
                "type": "bessel",
                "weights": [
                    float((index + 1) * np.pi / cutoff) for index in range(embedding)
                ],
                "prefactor": float(np.sqrt(2.0 / cutoff)),
            },
            "cutoff": {"type": "polynomial", "r_max": cutoff, "p": 6},
            "distance_transform": {"type": "none"},
            "networks": {
                "R0": radial_network((edge_l_max + 1) * channels, 0.03),
                "R1": radial_network(len(paths) * channels, 0.025),
            },
        },
        "execution_contracts": {
            **({"M0": m0_contract} if m0_contract is not None else {}),
            **({"R0": r0_contract} if r0_contract is not None else {}),
            "R1": contract,
        },
        "M0_monomials": m0_monomials,
        "M0_weights": m0_weights,
        "H1_weights": repeated_diagonal(source_l_max + 1, 0.5),
        "has_field_coupling": False,
        "field_couplings": [],
        "Phi1_l": [path["output_l"] for path in paths],
        "Phi1_l1": [path["edge_l"] for path in paths],
        "Phi1_l2": [path["source_l"] for path in paths],
        "Phi1_lme": [term["lme"] for term in terms],
        "Phi1_lelm1lm2": [term["row"] for term in terms],
        "Phi1_clebsch_gordan": [term["coefficient"] for term in terms],
        "A1_weights": [
            repeated_diagonal(multiplicity, 0.3) for multiplicity in path_multiplicities
        ],
        "A1_scaled": False,
        "M1_monomials": m1_monomials,
        "M1_weights": m1_weights,
        "H2_weights_for_H1": [
            [0.0] * (channels * channels) for _ in range(num_elements)
        ],
        "H2_weights_for_M1": repeated_diagonal(1, 0.5),
        "readout_1_weights": [0.0] * channels,
        "readout_2_weights_1": readout_2_weights_1,
        "readout_2_weights_2": [0.4] * 16,
        "readout_2_scale_factor": 1.0,
        "has_zbl": False,
    }


def write_foundation_r1_test_model(contract_path, output_path, atomic_numbers=(1,)):
    contract = json.loads(Path(contract_path).read_text())
    model = foundation_r1_test_model(contract, atomic_numbers)
    Path(output_path).write_text(json.dumps(model, separators=(",", ":")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("contract")
    parser.add_argument("output")
    parser.add_argument("--atomic-numbers", default="1")
    args = parser.parse_args()
    atomic_numbers = [int(value) for value in args.atomic_numbers.split(",")]
    write_foundation_r1_test_model(
        args.contract, args.output, atomic_numbers=atomic_numbers
    )


if __name__ == "__main__":
    main()
