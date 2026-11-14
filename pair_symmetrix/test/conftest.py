import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_SYMMETRIX_TESTS = Path(__file__).resolve().parents[2] / "symmetrix" / "test"
_R1_CONTRACT = (
    _SYMMETRIX_TESTS
    / "data"
    / "execution_contracts"
    / "jit_r1_mace_off23_small_contract.json"
)


# Hacky, prevents bad exit due that doesn't seem directly related to the tests
def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 0:
        os._exit(0)


@pytest.fixture(scope="session")
def direct_compact_model(tmp_path_factory):
    """Generated compact models for the direct route, by atomic numbers.

    ``direct_compact_model((1, 8))`` writes a two-layer MACE with the public
    MACE-OFF23 small R1 layout and generated weights for those elements, and
    returns the paths of the model and of its float64 host artifact.
    """
    helper = _SYMMETRIX_TESTS / "compact_r1_model.py"
    if not helper.is_file() or not _R1_CONTRACT.is_file():
        pytest.skip(f"the compact R1 test model helper is missing: {helper}")
    spec = importlib.util.spec_from_file_location("compact_r1_model", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    contract = json.loads(_R1_CONTRACT.read_text())
    # tmp_path_factory directories are private to the user, as the JIT cache
    # root requires.
    jit_cache = tmp_path_factory.mktemp("direct-compact-jit")
    built = {}

    def build(atomic_numbers):
        key = tuple(atomic_numbers)
        if key not in built:
            model = module.foundation_r1_test_model(contract, atomic_numbers=key)
            path = tmp_path_factory.mktemp("direct-compact") / "compact-mace.json"
            path.write_text(json.dumps(model, separators=(",", ":")))
            environment = os.environ.copy()
            environment["SYMMETRIX_JIT_CACHE"] = str(jit_cache)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "symmetrix.cli.prepare_jit_host_artifact",
                    "--model",
                    str(path),
                    "--precision",
                    "float64",
                    "--path-only",
                    "--cache-root",
                    str(jit_cache),
                ],
                capture_output=True,
                text=True,
                timeout=1800,
                env=environment,
            )
            assert completed.returncode == 0, completed.stderr
            artifact = Path(completed.stdout.strip().splitlines()[-1])
            assert artifact.is_file(), completed.stdout
            built[key] = (path, artifact)
        return built[key]

    return build
