#!/usr/bin/env python3
"""Generate immutable JSON D2/D3 oracle cases with pinned torch-dftd."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import importlib
import json
from pathlib import Path
import platform
import re
import sys

from _torch_dftd_source import (
    EXPECTED_VERSION,
    REFERENCE_SOURCE_PATHS,
    UPSTREAM_COMMIT,
    sha256,
    validate_source,
)


# Import the pinned package without leaving cache files beside the source input.
sys.dont_write_bytecode = True


CASES = (
    {
        "name": "d2-pbe-mixed-molecule",
        "method": "d2",
        "damping": "zero",
        "functional": "pbe",
        "atomic_numbers": [8, 1, 1, 6],
        "positions_angstrom": [
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
            [2.75, 0.2, -0.1],
        ],
        "cell_angstrom": None,
        "pbc": [False, False, False],
        "cutoff_angstrom": 8.0,
        "coordination_cutoff_angstrom": 6.0,
        "coverage": ["molecular", "mixed-element"],
    },
    {
        "name": "d3-zero-pbe0-molecule",
        "method": "d3",
        "damping": "zero",
        "functional": "pbe0",
        "atomic_numbers": [6, 6, 1, 1],
        "positions_angstrom": [
            [0.0, 0.0, 0.0],
            [1.42, 0.0, 0.0],
            [-0.8, 0.7, 0.0],
            [2.22, -0.7, 0.0],
        ],
        "cell_angstrom": None,
        "pbc": [False, False, False],
        "cutoff_angstrom": 8.0,
        "coordination_cutoff_angstrom": 6.0,
        "coverage": ["molecular"],
    },
    {
        "name": "d3-bj-pbe-periodic",
        "method": "d3",
        "damping": "bj",
        "functional": "pbe",
        "atomic_numbers": [14, 8],
        "positions_angstrom": [[0.0, 0.0, 0.0], [2.1, 2.1, 2.1]],
        "cell_angstrom": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        "pbc": [True, True, True],
        "cutoff_angstrom": 5.2,
        "coordination_cutoff_angstrom": 4.5,
        "coverage": ["periodic", "mixed-element"],
    },
    {
        "name": "d3-zerom-pbe-cutoff-boundary",
        "method": "d3",
        "damping": "zerom",
        "functional": "pbe",
        "atomic_numbers": [6, 8],
        "positions_angstrom": [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        "cell_angstrom": None,
        "pbc": [False, False, False],
        "cutoff_angstrom": 5.0,
        "coordination_cutoff_angstrom": 5.0,
        "coverage": ["molecular", "cutoff-boundary", "mixed-element"],
    },
    {
        "name": "d3-zerom-pbe-interior",
        "method": "d3",
        "damping": "zerom",
        "functional": "pbe",
        "atomic_numbers": [6, 8],
        "positions_angstrom": [[0.0, 0.0, 0.0], [4.8, 0.0, 0.0]],
        "cell_angstrom": None,
        "pbc": [False, False, False],
        "cutoff_angstrom": 5.0,
        "coordination_cutoff_angstrom": 5.0,
        "coverage": ["molecular", "cutoff-interior", "mixed-element"],
    },
    {
        "name": "d3-bjm-pbe-empty-edge",
        "method": "d3",
        "damping": "bjm",
        "functional": "pbe",
        "atomic_numbers": [1, 8],
        "positions_angstrom": [[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]],
        "cell_angstrom": None,
        "pbc": [False, False, False],
        "cutoff_angstrom": 4.0,
        "coordination_cutoff_angstrom": 4.0,
        "coverage": ["molecular", "empty-edge", "mixed-element"],
    },
    {
        "name": "d3-bjm-pbe-mixed-pair",
        "method": "d3",
        "damping": "bjm",
        "functional": "pbe",
        "atomic_numbers": [6, 8],
        "positions_angstrom": [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        "cell_angstrom": None,
        "pbc": [False, False, False],
        "cutoff_angstrom": 5.0,
        "coordination_cutoff_angstrom": 5.0,
        "coverage": ["molecular", "nonzero-pair", "mixed-element"],
    },
    {
        "name": "d3-bj-pbe-high-coordination",
        "method": "d3",
        "damping": "bj",
        "functional": "pbe",
        "atomic_numbers": [26],
        "positions_angstrom": [[0.0, 0.0, 0.0]],
        "cell_angstrom": [
            [1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0],
            [0.0, 0.0, 1.5],
        ],
        "pbc": [True, True, True],
        "cutoff_angstrom": 5.0,
        "coordination_cutoff_angstrom": 5.0,
        "coverage": ["periodic", "high-coordination", "interpolation-underflow"],
    },
)


def _version(source: Path) -> str:
    contents = (source / "torch_dftd" / "_version.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', contents, re.MULTILINE)
    if match is None:
        raise RuntimeError("cannot determine torch-dftd source version")
    return match.group(1)


def _edges(case, torch, np, primitive_neighbor_list):
    positions = np.asarray(case["positions_angstrom"], dtype=np.float64)
    cutoff = case["cutoff_angstrom"]
    if any(case["pbc"]):
        cell = np.asarray(case["cell_angstrom"], dtype=np.float64)
        first, second, shifts = primitive_neighbor_list(
            "ijS", case["pbc"], cell, positions, cutoff
        )
        shift_positions = shifts @ cell
    else:
        first = []
        second = []
        shift_positions = []
        for i, position_i in enumerate(positions):
            for j, position_j in enumerate(positions):
                if i != j and np.linalg.norm(position_i - position_j) <= cutoff:
                    first.append(i)
                    second.append(j)
                    shift_positions.append([0.0, 0.0, 0.0])
        first = np.asarray(first, dtype=np.int64)
        second = np.asarray(second, dtype=np.int64)
        shift_positions = np.asarray(shift_positions, dtype=np.float64).reshape(-1, 3)
    edge_index = torch.tensor(np.stack([first, second]), dtype=torch.int64)
    return edge_index, torch.tensor(shift_positions, dtype=torch.float64)


def generate(source: Path) -> dict:
    if _version(source) != EXPECTED_VERSION:
        raise RuntimeError(
            f"reference generation requires torch-dftd {EXPECTED_VERSION}"
        )
    source_files = validate_source(source, REFERENCE_SOURCE_PATHS)
    source_hashes = {relative: sha256(path) for relative, path in source_files.items()}
    sys.path.insert(0, str(source))
    try:
        torch = importlib.import_module("torch")
        np = importlib.import_module("numpy")
        ase = importlib.import_module("ase")
        primitive_neighbor_list = importlib.import_module(
            "ase.neighborlist"
        ).primitive_neighbor_list
        get_params = importlib.import_module(
            "torch_dftd.dftd3_xc_params"
        ).get_dftd3_default_params
        DFTD2Module = importlib.import_module("torch_dftd.nn.dftd2_module").DFTD2Module
        DFTD3Module = importlib.import_module("torch_dftd.nn.dftd3_module").DFTD3Module

        output_cases = []
        for definition in CASES:
            case = dict(definition)
            edge_index, shift_positions = _edges(
                case, torch, np, primitive_neighbor_list
            )
            atomic_numbers = torch.tensor(case["atomic_numbers"], dtype=torch.int64)
            positions = torch.tensor(case["positions_angstrom"], dtype=torch.float64)
            cell = (
                None
                if case["cell_angstrom"] is None
                else torch.tensor(case["cell_angstrom"], dtype=torch.float64)
            )
            pbc = torch.tensor(case["pbc"], dtype=torch.bool)
            old = case["method"] == "d2"
            parameters = get_params(case["damping"], case["functional"], old=old)
            if old:
                module = DFTD2Module(
                    parameters,
                    cutoff=case["cutoff_angstrom"],
                    dtype=torch.float64,
                    bidirectional=True,
                )
            else:
                module = DFTD3Module(
                    parameters,
                    cutoff=case["cutoff_angstrom"],
                    cnthr=case["coordination_cutoff_angstrom"],
                    dtype=torch.float64,
                    bidirectional=True,
                )
            result = module.calc_energy_and_forces(
                Z=atomic_numbers,
                pos=positions,
                edge_index=edge_index,
                cell=cell,
                pbc=pbc,
                shift_pos=shift_positions,
                damping=case["damping"],
            )[0]
            case["directed_edges"] = int(edge_index.shape[1])
            case["energy_ev"] = float(result["energy"])
            case["forces_ev_per_angstrom"] = result["forces"].tolist()
            if "stress" in result:
                case["stress_ev_per_angstrom3"] = result["stress"].tolist()
            output_cases.append(case)
        return {
            "schema": "symmetrix.dispersion-reference/1",
            "source": {
                "torch_dftd": EXPECTED_VERSION,
                "commit": UPSTREAM_COMMIT,
                "torch": torch.__version__,
                "ase": ase.__version__,
                "python": platform.python_version(),
                "source_hashes": source_hashes,
            },
            "cases": output_cases,
        }
    finally:
        sys.path.remove(str(source))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-dftd-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        payload = generate(arguments.torch_dftd_source.resolve())
        output = arguments.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
