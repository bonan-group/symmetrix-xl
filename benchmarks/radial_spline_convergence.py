"""Sweep compact-radial spline sizes and diagnose interpolation failures.

Known red gate: the nodal-consistency check below is expected to fail for a
compact model until the oracle is corrected.  The retained evaluator's ``R0``
table is the model's H0/A0-projected radial row while ``exact_network``
reimplements the raw compact radial network, so the two disagree at the spline
nodes (measured 1.07e1 after the projection, 1.63e3 before it).  The sweep's
value and derivative convergence numbers are therefore not yet trustworthy for a
compact model; a red gate here is this defect, not a regression.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import tempfile

import numpy as np

NETWORKS = ("R0", "R1")


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_int_csv(value: str) -> list[int]:
    result = sorted({int(item) for item in value.split(",")})
    if not result or result[0] < 4:
        raise argparse.ArgumentTypeError("spline points must be integers of at least 4")
    return result


def _parse_float_csv(value: str) -> np.ndarray:
    result = np.asarray([float(item) for item in value.split(",")], dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise argparse.ArgumentTypeError("expected three finite comma-separated values")
    return result


def _distance_grid(minimum: float, cutoff: float, samples: int) -> np.ndarray:
    if not 0.0 < minimum < cutoff:
        raise ValueError("minimum radius must lie strictly inside the model cutoff")
    if samples < 32:
        raise ValueError("at least 32 radial samples are required")
    # Linear coverage catches errors throughout the cutoff. Logarithmic coverage
    # resolves the rapidly varying short-range derivative that exposed the MPI
    # fixture problem.
    endpoint = np.nextafter(cutoff, minimum)
    linear = np.linspace(minimum, endpoint, samples)
    logarithmic = np.geomspace(minimum, endpoint, samples)
    return np.unique(np.concatenate((linear, logarithmic)))


def _distance_transform(
    compact: dict, type_i: int, type_j: int, radius: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    definition = compact["distance_transform"]
    if definition["type"] == "none":
        return radius, np.ones_like(radius)
    if definition["type"] != "agnesi":
        raise ValueError(f"unsupported distance transform: {definition['type']}")

    radii = np.asarray(definition["covalent_radii"], dtype=np.float64)
    r0 = 0.5 * (radii[type_i] + radii[type_j])
    a = float(definition["a"])
    q = float(definition["q"])
    p = float(definition["p"])
    scaled = radius / r0
    scaled_q = scaled**q
    scaled_qp = scaled ** (q - p)
    denominator_inner = 1.0 + scaled_qp
    ratio = a * scaled_q / denominator_inner
    transformed = 1.0 / (1.0 + ratio)
    scaled_q_derivative = q * scaled ** (q - 1.0) / r0
    scaled_qp_derivative = (q - p) * scaled ** (q - p - 1.0) / r0
    ratio_derivative = (
        a
        * (scaled_q_derivative * denominator_inner - scaled_q * scaled_qp_derivative)
        / denominator_inner**2
    )
    transformed_derivative = -ratio_derivative / (1.0 + ratio) ** 2
    return transformed, transformed_derivative


def _cutoff(compact: dict, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    definition = compact["cutoff"]
    if definition["type"] != "polynomial":
        raise ValueError(f"unsupported cutoff: {definition['type']}")
    maximum = float(definition["r_max"])
    order = int(definition["p"])
    scaled = radius / maximum
    active = radius < maximum
    value = np.zeros_like(radius)
    derivative = np.zeros_like(radius)
    x = scaled[active]
    coefficient_0 = (order + 1.0) * (order + 2.0) / 2.0
    coefficient_1 = order * (order + 2.0)
    coefficient_2 = order * (order + 1.0) / 2.0
    value[active] = (
        1.0
        - coefficient_0 * x**order
        + coefficient_1 * x ** (order + 1)
        - coefficient_2 * x ** (order + 2)
    )
    derivative[active] = (
        -coefficient_0 * order * x ** (order - 1)
        + coefficient_1 * (order + 1) * x**order
        - coefficient_2 * (order + 2) * x ** (order + 1)
    ) / maximum
    return value, derivative


def _radial_features(
    compact: dict, type_i: int, type_j: int, radius: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    transformed, transformed_derivative = _distance_transform(
        compact, type_i, type_j, radius
    )
    cutoff, cutoff_derivative = _cutoff(compact, radius)
    basis = compact["basis"]
    if basis["type"] != "bessel":
        raise ValueError(f"unsupported radial basis: {basis['type']}")
    weights = np.asarray(basis["weights"], dtype=np.float64)[None, :]
    prefactor = float(basis["prefactor"])
    transformed_column = transformed[:, None]
    phase = transformed_column * weights
    sine_over_radius = np.sin(phase) / transformed_column
    sine_over_radius_derivative = (
        weights * np.cos(phase) * transformed_column - np.sin(phase)
    ) / transformed_column**2
    values = prefactor * sine_over_radius * cutoff[:, None]
    derivatives = prefactor * (
        sine_over_radius_derivative * transformed_derivative[:, None] * cutoff[:, None]
        + sine_over_radius * cutoff_derivative[:, None]
    )
    return values, derivatives


def exact_network(
    compact: dict,
    network_name: str,
    type_i: int,
    type_j: int,
    radius: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the extracted PyTorch radial network and its exact derivative."""
    network = compact["networks"][network_name]
    shape = [int(value) for value in network["shape"]]
    values, derivatives = _radial_features(compact, type_i, type_j, radius)
    if values.shape[1] != shape[0]:
        raise ValueError(f"{network_name} input shape does not match its basis")

    activation_scale = float(network["activation_scale"])
    for layer, flattened in enumerate(network["weights"]):
        weights = np.asarray(flattened, dtype=np.float64).reshape(
            shape[layer + 1], shape[layer]
        )
        preactivation = values @ weights.T
        preactivation_derivative = derivatives @ weights.T
        if layer + 1 < len(shape) - 1:
            sigmoid = 1.0 / (1.0 + np.exp(-preactivation))
            activation_derivative = (
                activation_scale * sigmoid * (1.0 + preactivation * (1.0 - sigmoid))
            )
            values = activation_scale * preactivation * sigmoid
            derivatives = preactivation_derivative * activation_derivative
        else:
            values = preactivation
            derivatives = preactivation_derivative

    if network.get("postprocess") == "tanh-square":
        raw = values
        values = np.tanh(raw**2)
        derivatives *= 2.0 * raw * (1.0 - values**2)
    return values, derivatives


# ``R0_deriv``/``R1_deriv`` retired with the serial evaluator: the retained
# evaluators expose the radial network value only.  The runtime table is a C1
# cubic Hermite, so a centered difference of the runtime values reproduces its
# derivative orders of magnitude below the derivative gate.
_DERIVATIVE_STEP_A = 1.0e-8


def _r0_projection(model: dict, type_i: int, type_j: int) -> np.ndarray:
    """Return the per-``l`` linear map the runtime applies to the R0 network.

    The runtime R0 table is not the raw radial network output.  The compact
    upload multiplies every function by the source element's per-channel
    ``H0_weights`` and then fuses the receiver element's per-``l`` ``A0_weights``
    block, so an oracle that skips that map is comparing a different quantity.
    The element order is unchanged (``l * num_channels + channel``), so the map
    is block diagonal over ``l``.
    """

    num_channels = int(model["num_channels"])
    l_max = int(model["l_max"])
    h0 = np.asarray(model["H0_weights"], dtype=np.float64).reshape(-1, num_channels)
    blocks = np.asarray(model["A0_weights"][type_i], dtype=np.float64)
    if blocks.shape != (l_max + 1, num_channels * num_channels):
        raise ValueError(
            "A0_weights does not match the declared l_max and channel count"
        )
    return blocks.reshape(l_max + 1, num_channels, num_channels) * h0[type_j][None, :]


def _project_r0(values: np.ndarray, projection: np.ndarray, num_channels: int):
    """Apply the runtime R0 projection to an oracle value or derivative."""

    rows = values.shape[0]
    blocks = values.reshape(rows, projection.shape[0], num_channels)
    return np.einsum("rlk,lkj->rlj", blocks, projection).reshape(rows, -1)


def _runtime_network(
    evaluator,
    network_name: str,
    type_i: int,
    type_j: int,
    radius: np.ndarray,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the runtime radial network value and its radius derivative.

    ``R0``/``R1`` are exposed as flat vectors over a grow-only capacity view:
    the native allocator only reallocates when a request does not fit, so the row
    count is the high-water mark of every earlier request rather than the current
    edge count.  The width comes from the network's declared output width, and
    the first ``len(radius)`` rows are the edges this call evaluated.
    """

    def evaluate(sample_radius: np.ndarray) -> np.ndarray:
        node_types = np.asarray([type_i], dtype=np.int32)
        num_neigh = np.asarray([len(sample_radius)], dtype=np.int32)
        neigh_types = np.full(len(sample_radius), type_j, dtype=np.int32)
        getattr(evaluator, f"compute_{network_name}")(
            1, node_types, num_neigh, neigh_types, sample_radius
        )
        flat = np.asarray(getattr(evaluator, network_name), dtype=np.float64)
        if flat.size % width:
            raise RuntimeError(
                f"{network_name} holds {flat.size} elements, not a multiple of "
                f"its declared width {width}"
            )
        rows = flat.size // width
        if rows < len(sample_radius):
            raise RuntimeError(
                f"{network_name} holds {rows} rows, fewer than the "
                f"{len(sample_radius)} edges this call evaluated"
            )
        return flat.reshape(rows, width)[: len(sample_radius)]

    values = evaluate(radius)
    derivatives = (
        evaluate(radius + _DERIVATIVE_STEP_A) - evaluate(radius - _DERIVATIVE_STEP_A)
    ) / (2.0 * _DERIVATIVE_STEP_A)
    return values, derivatives


def _error_record(
    radius: np.ndarray,
    exact: np.ndarray,
    actual: np.ndarray,
    mask: np.ndarray,
    relative_floor: float,
) -> dict:
    selected_error = np.abs(actual[mask] - exact[mask])
    selected_exact = exact[mask]
    selected_actual = actual[mask]
    selected_radius = radius[mask]
    flat = int(np.argmax(selected_error))
    sample, output = np.unravel_index(flat, selected_error.shape)
    relative = selected_error / np.maximum(np.abs(selected_exact), relative_floor)
    relative_flat = int(np.argmax(relative))
    relative_sample, relative_output = np.unravel_index(relative_flat, relative.shape)
    return {
        "max_abs": float(selected_error[sample, output]),
        "max_abs_radius_A": float(selected_radius[sample]),
        "max_abs_output": int(output),
        "exact_at_max_abs": float(selected_exact[sample, output]),
        "spline_at_max_abs": float(selected_actual[sample, output]),
        "max_relative": float(relative[relative_sample, relative_output]),
        "max_relative_radius_A": float(selected_radius[relative_sample]),
        "max_relative_output": int(relative_output),
    }


def _bad_radius_regions(
    radius: np.ndarray, error: np.ndarray, threshold: float
) -> dict | None:
    bad = np.max(np.abs(error), axis=1) > threshold
    if not np.any(bad):
        return None
    indices = np.flatnonzero(bad)
    boundaries = np.flatnonzero(np.diff(indices) > 1) + 1
    groups = np.split(indices, boundaries)
    regions = [
        {
            "minimum_A": float(radius[group[0]]),
            "maximum_A": float(radius[group[-1]]),
            "sample_count": len(group),
        }
        for group in groups
    ]
    largest = sorted(regions, key=lambda item: item["sample_count"], reverse=True)[:8]
    return {
        "minimum_A": float(radius[indices[0]]),
        "maximum_A": float(radius[indices[-1]]),
        "sample_count": len(indices),
        "region_count": len(regions),
        "largest_regions": largest,
    }


def _sweep_count(
    model: dict,
    model_path: pathlib.Path,
    spline_points: int,
    radius: np.ndarray,
    physical_minimum: float,
    value_tolerance: float,
    derivative_tolerance: float,
    relative_floor: float,
) -> dict:
    from symmetrix import symmetrix as native_symmetrix

    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    candidate = json.loads(json.dumps(model))
    candidate["compact_radial"]["num_spline_points"] = spline_points
    candidate_path = model_path / f"compact-{spline_points}.json"
    candidate_path.write_text(json.dumps(candidate, separators=(",", ":")) + "\n")
    # The radial network is read through the retained Kokkos evaluator in the
    # reference routing's native mode, so the sweep measures the artifact-free
    # generic implementation.  ``MACEKokkos`` is the float64 evaluator.
    evaluator = native_symmetrix.MACEKokkos(str(candidate_path))
    evaluator.set_streamed_edges("generic")
    type_count = len(candidate["atomic_numbers"])
    evaluator.prepare_active_types(np.arange(type_count, dtype=np.int32))
    physical_mask = radius >= physical_minimum
    compact = candidate["compact_radial"]
    spline_grid = np.linspace(
        float(compact["spline_grid_min"]),
        float(model["r_cut"]),
        spline_points,
    )
    cases = []
    maximum_value_error = 0.0
    maximum_derivative_error = 0.0
    passed = True

    for network_name in NETWORKS:
        # The declared output width of the extracted network is the width of the
        # runtime view for this network as well; the sweep asserts the runtime
        # nodes against that same oracle below.
        width = int(compact["networks"][network_name]["shape"][-1])
        for type_i in range(type_count):
            for type_j in range(type_count):
                # ``R1`` is uploaded unprojected; ``R0`` carries the model's
                # H0/A0 map, so the oracle has to carry it too.
                projection = (
                    _r0_projection(candidate, type_i, type_j)
                    if network_name == "R0"
                    else None
                )
                exact_value, exact_derivative = exact_network(
                    compact, network_name, type_i, type_j, radius
                )
                exact_nodes, _ = exact_network(
                    compact, network_name, type_i, type_j, spline_grid
                )
                if projection is not None:
                    num_channels = int(candidate["num_channels"])
                    exact_value = _project_r0(exact_value, projection, num_channels)
                    exact_derivative = _project_r0(
                        exact_derivative, projection, num_channels
                    )
                    exact_nodes = _project_r0(exact_nodes, projection, num_channels)
                spline_value, spline_derivative = _runtime_network(
                    evaluator, network_name, type_i, type_j, radius, width
                )
                spline_nodes, _ = _runtime_network(
                    evaluator, network_name, type_i, type_j, spline_grid, width
                )
                # A cubic spline interpolant passes through its own nodal
                # values, so this gate is the oracle's only real check: the
                # runtime table at the grid nodes must equal the network output
                # there.  It is known red for a compact model, and the measured
                # evidence is recorded here so a red gate is not mistaken for a
                # regression from the routing migration:
                #
                #   * the retained ``R0`` table is the model's H0/A0-projected
                #     radial row (``mace_kokkos_factorized_lifecycle.cpp``), not
                #     the raw network output; ``_r0_projection`` applies that
                #     native map and reduces the disagreement from 1.63e3 to
                #     1.07e1, which is the residual reported below;
                #   * the disagreement is not a permutation of the oracle (a
                #     sorted comparison still differs by 9.35e2) and not a linear
                #     image of it either (a best-fit nodal map is dense and still
                #     leaves 2.6e-2);
                #   * the Bessel basis, the polynomial cutoff, the node grid and
                #     the MLP conventions all match the native implementation, so
                #     the remaining error is inside ``exact_network`` itself.
                nodal_consistency_error = float(
                    np.max(np.abs(spline_nodes - exact_nodes))
                )
                if nodal_consistency_error > 1.0e-9:
                    raise RuntimeError(
                        "exact compact-radial oracle disagrees with runtime spline nodes: "
                        f"{network_name} pair ({type_i}, {type_j}) error "
                        f"{nodal_consistency_error:.6e}"
                    )
                value_physical = _error_record(
                    radius,
                    exact_value,
                    spline_value,
                    physical_mask,
                    relative_floor,
                )
                derivative_physical = _error_record(
                    radius,
                    exact_derivative,
                    spline_derivative,
                    physical_mask,
                    relative_floor,
                )
                maximum_value_error = max(
                    maximum_value_error, value_physical["max_abs"]
                )
                maximum_derivative_error = max(
                    maximum_derivative_error, derivative_physical["max_abs"]
                )
                case_passed = (
                    value_physical["max_abs"] <= value_tolerance
                    and derivative_physical["max_abs"] <= derivative_tolerance
                )
                passed = passed and case_passed
                cases.append(
                    {
                        "network": network_name,
                        "receiver_type": type_i,
                        "receiver_atomic_number": candidate["atomic_numbers"][type_i],
                        "source_type": type_j,
                        "source_atomic_number": candidate["atomic_numbers"][type_j],
                        "nodal_value_consistency_error": nodal_consistency_error,
                        "passed_physical_gate": case_passed,
                        "physical": {
                            "value": value_physical,
                            "derivative": derivative_physical,
                        },
                        "full_range": {
                            "value": _error_record(
                                radius,
                                exact_value,
                                spline_value,
                                np.ones(len(radius), dtype=bool),
                                relative_floor,
                            ),
                            "derivative": _error_record(
                                radius,
                                exact_derivative,
                                spline_derivative,
                                np.ones(len(radius), dtype=bool),
                                relative_floor,
                            ),
                            "value_problem_regions_A": _bad_radius_regions(
                                radius,
                                spline_value - exact_value,
                                value_tolerance,
                            ),
                            "derivative_problem_regions_A": _bad_radius_regions(
                                radius,
                                spline_derivative - exact_derivative,
                                derivative_tolerance,
                            ),
                        },
                    }
                )

    return {
        "spline_points": spline_points,
        "grid_spacing_A": float(
            (float(model["r_cut"]) - float(compact["spline_grid_min"]))
            / (spline_points - 1)
        ),
        "passed_physical_gate": passed,
        "physical_max_value_error": maximum_value_error,
        "physical_max_derivative_error": maximum_derivative_error,
        "cases": cases,
    }


def _minimum_periodic_distance(atoms) -> float:
    from ase.neighborlist import neighbor_list

    distances = neighbor_list("d", atoms, 2.5)
    if len(distances) == 0:
        return float("inf")
    return float(np.min(distances))


def _load_end_to_end_structure(structure: pathlib.Path | None):
    if structure is None:
        from ase.build import bulk

        atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982).repeat((2, 2, 2))
        phases = np.asarray([0.7, 1.1, 1.7])
        atoms.positions += 0.008 * np.sin(
            (np.arange(len(atoms))[:, None] + 1) * phases[None, :]
        )
        provenance = {
            "kind": "generated",
            "description": "deterministically perturbed 2x2x2 wurtzite AlN",
            "lattice_a_A": 3.112,
            "lattice_c_A": 4.982,
            "maximum_perturbation_A": 0.008,
        }
    else:
        from ase.io import read

        resolved = structure.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"structure file is unavailable: {resolved}")
        atoms = read(resolved)
        provenance = {
            "kind": "file",
            "path": str(resolved),
            "sha256": _sha256(resolved),
        }
    if len(atoms) == 0:
        raise ValueError("end-to-end structure must contain at least one atom")
    provenance.update(
        {
            "formula": atoms.get_chemical_formula(),
            "atoms": len(atoms),
            "periodic": np.asarray(atoms.pbc, dtype=bool).tolist(),
        }
    )
    return atoms, provenance


def _end_to_end(
    checkpoint: pathlib.Path,
    candidate_paths: dict[int, pathlib.Path],
    model: dict,
    structure: pathlib.Path | None,
    head: str | None,
    device: str,
    field: np.ndarray,
    energy_tolerance: float,
    force_tolerance: float,
) -> dict:
    from mace.calculators.mace import MACECalculator

    from symmetrix import Symmetrix

    atoms, structure_provenance = _load_end_to_end_structure(structure)
    model_type = model.get("model_type", "MACE")
    if model_type == "MACEField":
        atoms.info["electric_field"] = field
    options = {
        "model_paths": [str(checkpoint)],
        "device": device,
        "default_dtype": "float64",
    }
    if head is not None:
        options["head"] = head
    if model_type == "MACEField":
        options["model_type"] = "MACEField"
    atoms.calc = MACECalculator(**options)
    reference_energy = float(atoms.get_potential_energy())
    reference_forces = np.asarray(atoms.get_forces(), dtype=np.float64)

    rows = []
    for spline_points, path in candidate_paths.items():
        atoms.calc = Symmetrix(
            path,
            dtype="float64",
            streamed_edges="all_interactions",
            jit="off",
        )
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        difference = np.abs(forces - reference_forces)
        flat = int(np.argmax(difference))
        atom, component = np.unravel_index(flat, difference.shape)
        rows.append(
            {
                "spline_points": spline_points,
                "energy_error_eV": abs(energy - reference_energy),
                "force_max_abs_error_eV_per_A": float(difference[atom, component]),
                "force_worst_atom": int(atom),
                "force_worst_component": int(component),
                "passed": (
                    abs(energy - reference_energy) <= energy_tolerance
                    and float(difference[atom, component]) <= force_tolerance
                ),
            }
        )
    return {
        "structure": structure_provenance,
        "minimum_distance_A": _minimum_periodic_distance(atoms),
        "model_type": model_type,
        "electric_field_V_per_A": field.tolist() if model_type == "MACEField" else None,
        "rows": rows,
    }


def _markdown(report: dict) -> str:
    lines = [
        "# Radial spline convergence",
        "",
        (
            f"Model cutoff: `{report['radius_grid']['cutoff_A']:.6g} A`; "
            f"physical gate starts at "
            f"`{report['radius_grid']['physical_minimum_A']:.6g} A`."
        ),
        "",
        "| Spline points | Spacing (A) | Max value error | Max derivative error | Gate |",
        "| ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in report["sweeps"]:
        lines.append(
            f"| {row['spline_points']} | {row['grid_spacing_A']:.6g} | "
            f"{row['physical_max_value_error']:.6e} | "
            f"{row['physical_max_derivative_error']:.6e} | "
            f"{'PASS' if row['passed_all_gates'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "Recommended minimum spline points: "
            + (
                f"`{report['recommended_minimum_spline_points']}`"
                if report["recommended_minimum_spline_points"] is not None
                else "none of the tested sizes passed"
            ),
            "",
        ]
    )
    if report.get("end_to_end"):
        structure = report["end_to_end"]["structure"]
        lines.extend(
            [
                "## End-to-end PyTorch parity",
                "",
                f"Structure: `{structure['formula']}` ({structure['atoms']} atoms).",
                "",
                "| Spline points | Energy error (eV) | Max force error (eV/A) |",
                "| ---: | ---: | ---: |",
            ]
        )
        for row in report["end_to_end"]["rows"]:
            lines.append(
                f"| {row['spline_points']} | {row['energy_error_eV']:.6e} | "
                f"{row['force_max_abs_error_eV_per_A']:.6e} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=pathlib.Path, help="compact Symmetrix JSON model")
    parser.add_argument(
        "--spline-points",
        type=_parse_int_csv,
        default=_parse_int_csv("64,128,256,512,1024"),
    )
    parser.add_argument("--samples", type=int, default=4097)
    parser.add_argument("--minimum-radius", type=float, default=0.5)
    parser.add_argument("--physical-minimum-radius", type=float, default=1.5)
    parser.add_argument("--max-value-error", type=float, default=1.0e-4)
    parser.add_argument("--max-derivative-error", type=float, default=1.0e-2)
    parser.add_argument("--relative-floor", type=float, default=1.0e-10)
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument(
        "--structure",
        type=pathlib.Path,
        help="ASE-readable end-to-end structure; defaults to generated AlN",
    )
    parser.add_argument("--head")
    parser.add_argument("--torch-device", default="cpu")
    parser.add_argument("--max-energy-error", type=float, default=1.0e-3)
    parser.add_argument("--max-force-error", type=float, default=1.0e-4)
    parser.add_argument(
        "--electric-field",
        type=_parse_float_csv,
        default=_parse_float_csv("0.01,0.02,-0.03"),
    )
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="write the report without failing when no tested size passes",
    )
    args = parser.parse_args()

    model_path = args.model.resolve()
    model = json.loads(model_path.read_text())
    if model.get("radial_representation") != "compact" or "compact_radial" not in model:
        parser.error("model must use the compact radial representation")
    if (
        args.max_value_error <= 0.0
        or args.max_derivative_error <= 0.0
        or args.max_energy_error <= 0.0
        or args.max_force_error <= 0.0
    ):
        parser.error("error tolerances must be positive")
    if not args.minimum_radius < args.physical_minimum_radius < float(model["r_cut"]):
        parser.error("radius limits must satisfy minimum < physical minimum < cutoff")

    radius = _distance_grid(args.minimum_radius, float(model["r_cut"]), args.samples)
    with tempfile.TemporaryDirectory(prefix="symmetrix-spline-sweep-") as temporary:
        temporary_path = pathlib.Path(temporary)
        sweeps = [
            _sweep_count(
                model,
                temporary_path,
                spline_points,
                radius,
                args.physical_minimum_radius,
                args.max_value_error,
                args.max_derivative_error,
                args.relative_floor,
            )
            for spline_points in args.spline_points
        ]
        candidate_paths = {
            spline_points: temporary_path / f"compact-{spline_points}.json"
            for spline_points in args.spline_points
        }
        end_to_end = None
        if args.checkpoint is not None:
            end_to_end = _end_to_end(
                args.checkpoint.resolve(),
                candidate_paths,
                model,
                args.structure,
                args.head,
                args.torch_device,
                args.electric_field,
                args.max_energy_error,
                args.max_force_error,
            )

    end_to_end_by_points = (
        {row["spline_points"]: row for row in end_to_end["rows"]}
        if end_to_end is not None
        else {}
    )
    for row in sweeps:
        physical_passed = (
            end_to_end_by_points[row["spline_points"]]["passed"]
            if end_to_end is not None
            else True
        )
        row["passed_end_to_end_gate"] = physical_passed
        row["passed_all_gates"] = row["passed_physical_gate"] and physical_passed
    passing = [row["spline_points"] for row in sweeps if row["passed_all_gates"]]
    maxima = [row["physical_max_derivative_error"] for row in sweeps]
    nonmonotonic = [
        args.spline_points[index]
        for index in range(1, len(maxima))
        if maxima[index] > maxima[index - 1] * (1.0 + 1.0e-12)
    ]
    report = {
        "schema_version": 1,
        "model": {
            "path": str(model_path),
            "sha256": _sha256(model_path),
            "model_type": model.get("model_type"),
            "atomic_numbers": model["atomic_numbers"],
        },
        "radius_grid": {
            "minimum_A": args.minimum_radius,
            "physical_minimum_A": args.physical_minimum_radius,
            "cutoff_A": float(model["r_cut"]),
            "sample_count": len(radius),
        },
        "gates": {
            "max_value_error": args.max_value_error,
            "max_derivative_error": args.max_derivative_error,
            "max_energy_error_eV": args.max_energy_error,
            "max_force_error_eV_per_A": args.max_force_error,
        },
        "recommended_minimum_spline_points": min(passing) if passing else None,
        "nonmonotonic_derivative_counts": nonmonotonic,
        "sweeps": sweeps,
        "end_to_end": end_to_end,
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded)
        output.with_suffix(".md").write_text(_markdown(report) + "\n")

    if not passing and not args.diagnostic_only:
        raise SystemExit("no tested spline size passed the physical radial-error gates")


if __name__ == "__main__":
    main()
