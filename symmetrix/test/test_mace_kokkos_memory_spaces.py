"""MACEKokkos keeps its model tables out of managed memory.

On CUDA and HIP, ``Kokkos::SharedSpace`` is managed memory.  Kernels read the
model tables at every evaluation while host code touches the same pages, so
the pages migrate back and forth.  On host backends ``SharedSpace`` is
``HostSpace``, so the placement is checked in the source.  The kernels that
index a View-of-Views read device handles to its inner Views; the growth test
checks that the handles follow a workspace that grows.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from compact_r1_model import foundation_r1_test_model
from symmetrix import Symmetrix

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "libsymmetrix" / "source"
CONTRACT_ROOT = Path(__file__).resolve().parent / "data" / "execution_contracts"


def _code(path):
    text = path.read_text()
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def _body(code, signature):
    start = code.index("{", code.index(signature))
    depth = 0
    for index in range(start, len(code)):
        depth += {"{": 1, "}": -1}.get(code[index], 0)
        if depth == 0:
            return code[start : index + 1]
    raise AssertionError(f"unterminated body after {signature!r}")


def test_one_dimensional_tables_are_device_views():
    tools = _code(SOURCE_ROOT / "tools_kokkos.hpp")
    body = _body(
        tools,
        "toKokkosView(const char* name,const std::vector<T>& stdVector)",
    )

    assert "SharedSpace" not in body


def test_mace_kokkos_allocates_nothing_in_shared_space():
    sources = sorted(SOURCE_ROOT.glob("mace_kokkos*.?pp"))

    assert sources
    offenders = [path.name for path in sources if "SharedSpace" in _code(path)]
    assert offenders == []


@pytest.fixture(scope="module")
def growth_model_path(tmp_path_factory):
    contract = json.loads((CONTRACT_ROOT / "jit_r1_contract.json").read_text())
    model = foundation_r1_test_model(contract, atomic_numbers=(7, 13))
    output = tmp_path_factory.mktemp("memory-space-growth") / "model.json"
    output.write_text(json.dumps(model, separators=(",", ":")))
    return output


def _cluster(count):
    grid = np.indices((4, 4, 4)).reshape(3, -1).T[:count]
    positions = 1.0 + 1.5 * grid + 0.05 * np.sin(np.arange(3 * count)).reshape(-1, 3)
    symbols = ["Al" if index % 2 == 0 else "N" for index in range(count)]
    return Atoms(symbols, positions=positions, cell=[12.0, 12.0, 12.0], pbc=False)


def test_reused_evaluator_matches_fresh_after_growth(growth_model_path):
    # A larger structure replaces the M0 polynomial workspace and the streamed
    # candidate storage; the kernels must then read the new buffers, and a
    # smaller structure must read only the active prefix of the storage.  The
    # runtime M0 executor runs the polynomial kernels that read the handles.
    def calculator():
        calc = Symmetrix(
            growth_model_path, dtype="float64", streamed_edges="non-compiled"
        )
        calc.evaluator._set_standard_m0_executor("runtime")
        return calc

    reused = calculator()
    for count in (8, 64, 8):
        atoms = _cluster(count)
        atoms.calc = reused
        fresh = _cluster(count)
        fresh.calc = calculator()

        assert atoms.get_potential_energy() == pytest.approx(
            fresh.get_potential_energy(), rel=1e-12, abs=1e-12
        )
        np.testing.assert_allclose(
            atoms.get_forces(), fresh.get_forces(), rtol=0.0, atol=1e-12
        )
