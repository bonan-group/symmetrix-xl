import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "benchmarks" / "mace_les_correctness.py"


def _module():
    spec = importlib.util.spec_from_file_location("les_correctness_driver", PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_result_document_preserves_full_arrays_and_provenance(tmp_path):
    driver = _module()
    checkpoint = tmp_path / "checkpoint.pt"
    converted = tmp_path / "converted.json"
    structure = tmp_path / "structure.xyz"
    checkpoint.write_bytes(b"checkpoint")
    converted.write_text(
        json.dumps(
            {
                "model_type": "MACELES",
                "les": {
                    "schema": "symmetrix.model.mace_les",
                    "version": 1,
                    "kind": "monopole",
                    "periodicity": "geometry",
                    "ewald": {
                        "sigma": 1.0,
                        "dl": 2.0,
                        "remove_self_interaction": True,
                    },
                    "output_scale": 0.25,
                    "mace_num_interactions": 2,
                    "mace_num_channels": 1,
                    "mace_contract_fingerprints": {
                        "R0": "sha256:" + "1" * 64,
                        "M0": "sha256:" + "2" * 64,
                        "R1": "sha256:" + "3" * 64,
                    },
                    "readout_1_weights": [0.3],
                    "readout_2_hidden_size": 1,
                    "readout_2_weights_1": [0.1],
                    "readout_2_weights_2": [0.7],
                    "readout_2_scale_factor": 1.5,
                },
            }
        )
    )
    structure.write_text("fixture")
    document = driver.build_result_document(
        implementation="symmetrix",
        precision="float32",
        backend="hip",
        mode="direct",
        checkpoint=checkpoint,
        converted_model=converted,
        structure=structure,
        frame=3,
        atom_count=2,
        results={
            "energy": 1.5,
            "forces": np.arange(6).reshape(2, 3),
            "latent_charges": np.array([0.2, -0.2]),
            "stress": np.arange(6),
        },
    )
    assert document["structure"]["frame"] == 3
    assert document["forces"] == [[0, 1, 2], [3, 4, 5]]
    assert document["model_type"] == "MACELES"
    assert len(document["mace_les_parameters_sha256"]) == 64
    assert len(document["mace_les_evaluator_fingerprint"]) == 64
    assert len(document["checkpoint"]["sha256"]) == 64
    assert Path(document["runtime_provenance"]["interpreter_path"]).is_file()
    assert document["runtime_provenance"]["python_version"]
    assert Path(document["runtime_provenance"]["package_path"]).is_file()
    assert document["runtime_provenance"]["package_version"]
