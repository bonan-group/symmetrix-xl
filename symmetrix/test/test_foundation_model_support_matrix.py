import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/foundation_model_support_matrix.py"
MANIFEST = ROOT / "benchmarks/foundation_model_support_manifest.json"


@pytest.fixture(scope="module")
def matrix():
    spec = importlib.util.spec_from_file_location("foundation_support_matrix", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_covers_release_matrix_and_classifies_chemistry(matrix):
    manifest = matrix.load_manifest(MANIFEST)
    ids = {row["id"] for row in manifest["models"]}
    assert {
        "mace-omat-0-small",
        "mace-omat-0-medium",
        "mace-mpa-0-medium",
        "mace-mp-0b-small",
        "mace-mp-0b-medium",
        "mace-mp-0b2-small",
        "mace-mp-0b2-medium",
        "mace-mp-0b2-large",
        "mace-mp-0b3-medium",
        "mace-matpes-pbe-0",
        "mace-matpes-r2scan-0",
        "mace-mh-0",
        "mace-mh-1",
        "mace-off23-small",
        "mace-off23-medium",
        "mace-off23-large",
    } == ids
    off = next(row for row in manifest["models"] if row["id"] == "mace-off23-small")
    assert matrix.chemistry_status(off, "C64") == (True, None)
    assert matrix.chemistry_status(off, "H2O") == (True, None)
    assert "STO40" not in off["chemistry"]


def test_oam_release_matrix_is_explicit_per_size_backend_and_precision(matrix):
    manifest = matrix.load_manifest(MANIFEST)
    support = manifest["nequip_support"]
    cells = support["release_matrix"]
    assert len(cells) == 16
    assert {(cell["size"], cell["backend"], cell["precision"]) for cell in cells} == {
        (size, backend, precision)
        for size in ("S", "M", "L", "XL")
        for backend in ("host", "cuda")
        for precision in ("float32", "float64")
    }
    for cell in cells:
        excluded_host_pair = (
            cell["backend"] == "host"
            and cell["precision"] == "float64"
            and cell["size"] in {"S", "L", "XL"}
        )
        assert cell["state"] == "available_unqualified"
        assert cell["runtime_qualification"] == (
            "available_not_runtime_qualified"
            if excluded_host_pair
            else "campaign_required"
        )
        if cell["backend"] == "host":
            assert cell["threads"] == ([] if excluded_host_pair else [1, 4])
            assert cell["graph_classes"] == {
                "host-100": (
                    "available_not_runtime_qualified"
                    if excluded_host_pair
                    else "required"
                )
            }
            assert cell["capacity_boundary"] == "none"
        else:
            assert cell["threads"] == [1]
            assert cell["graph_classes"] == {
                "small": "required",
                "production": "required",
            }
            assert cell["capacity_boundary"] == (
                "required"
                if cell["size"] == "XL" and cell["precision"] == "float32"
                else "not_required"
            )
        assert set(cell["properties"].values()) == {"available_unqualified"}
        assert set(cell["artifact"]) == {
            "availability",
            "kind",
            "abi_version",
            "origins",
        }
        assert cell["artifact"]["availability"] == "available"
        expected_status = "not_in_campaign" if excluded_host_pair else "required"
        assert cell["baseline"] == {"status": expected_status, "record": None}
        assert cell["evidence"]["correctness"] == {
            "status": expected_status,
            "record": None,
        }
        assert set(cell["tolerances"]) == {
            "energy_abs_eV_per_atom",
            "forces_max_abs_eV_per_A",
            "stress_max_abs_eV_per_A3",
        }
        assert cell["tested_compute_capabilities"] == []


def test_current_oam_matrix_requires_fresh_full_message_qualification(matrix):
    from symmetrix.oam_cuda_codegen import ABI_VERSION as cuda_abi_version
    from symmetrix.oam_receiver_codegen import ABI_VERSION as host_abi_version

    support = matrix.load_manifest(MANIFEST)["nequip_support"]

    assert "accepted_qualification" not in support
    for cell in support["release_matrix"]:
        key = (cell["size"], cell["backend"], cell["precision"])
        excluded_host_pair = key in {
            ("S", "host", "float64"),
            ("L", "host", "float64"),
            ("XL", "host", "float64"),
        }
        assert cell["state"] == "available_unqualified"
        assert cell["runtime_qualification"] == (
            "available_not_runtime_qualified"
            if excluded_host_pair
            else "campaign_required"
        )
        assert set(cell["properties"].values()) == {"available_unqualified"}
        assert cell["tested_compute_capabilities"] == []
        assert cell["artifact"]["abi_version"] == (
            host_abi_version if cell["backend"] == "host" else cuda_abi_version
        )


@pytest.mark.parametrize(
    "field", ("accepted_qualification", "historical_qualification")
)
def test_oam_release_matrix_rejects_superseded_qualification(matrix, field):
    support = json.loads(json.dumps(matrix.load_manifest(MANIFEST)["nequip_support"]))
    support[field] = {"status": "superseded"}
    with pytest.raises(ValueError, match="superseded accepted qualification"):
        matrix._validate_nequip_support(support, evidence_root=MANIFEST.parent)


def test_oam_release_matrix_declares_performance_gates(matrix):
    support = matrix.load_manifest(MANIFEST)["nequip_support"]
    assert support["performance_gates"] == {
        "production_min_speedup_vs_reference": 1.10,
        "small_min_speedup_vs_reference": 0.95,
        "production_geomean_min_speedup_vs_reference": 1.20,
        "approved_baseline_min_ratio": 0.95,
        "host_100_min_speedup_vs_reference": 1.10,
        "host_min_scaling_speedup": 1.0,
    }


@pytest.mark.parametrize("location", ("support", "cell"))
def test_oam_release_tolerances_are_exact_and_contradictions_fail(
    matrix, tmp_path, location
):
    manifest = matrix.load_manifest(MANIFEST)
    expected = {
        "float32": {
            "energy_abs_eV_per_atom": 1.0e-5,
            "forces_max_abs_eV_per_A": 5.0e-4,
            "stress_max_abs_eV_per_A3": 5.0e-4,
        },
        "float64": {
            "energy_abs_eV_per_atom": 5.0e-6,
            "forces_max_abs_eV_per_A": 1.0e-5,
            "stress_max_abs_eV_per_A3": 1.0e-6,
        },
    }
    assert manifest["nequip_support"]["tolerances"] == expected
    for cell in manifest["nequip_support"]["release_matrix"]:
        assert cell["tolerances"] == expected[cell["precision"]]

    mutated = json.loads(MANIFEST.read_text())
    target = (
        mutated["nequip_support"]["tolerances"]["float32"]
        if location == "support"
        else mutated["nequip_support"]["release_matrix"][0]["tolerances"]
    )
    target["forces_max_abs_eV_per_A"] *= 2
    path = tmp_path / "contradictory-tolerances.json"
    path.write_text(json.dumps(mutated))
    with pytest.raises(ValueError, match="authoritative OAM release tolerances"):
        matrix.load_manifest(path)


def test_generic_mace_worker_tolerances_remain_independent_from_oam(matrix):
    manifest = matrix.load_manifest(MANIFEST)
    assert manifest["tolerances"] == {
        "float32": {
            "energy_abs_eV_per_atom": 5.0e-4,
            "forces_max_abs_eV_per_A": 5.0e-3,
            "stress_max_abs_eV_per_A3": 8.0e-3,
        },
        "float64": {
            "energy_abs_eV_per_atom": 1.0e-9,
            "forces_max_abs_eV_per_A": 1.0e-8,
            "stress_max_abs_eV_per_A3": 1.0e-9,
        },
    }
    errors = {
        "energy_abs_eV_per_atom": 0.0,
        "forces_max_abs_eV_per_A": 1.0e-3,
        "stress_max_abs_eV_per_A3": 0.0,
    }
    assert matrix.parity_failures(errors, manifest["tolerances"]["float32"]) == []
    assert matrix.parity_failures(
        errors, manifest["nequip_support"]["tolerances"]["float32"]
    ) == ["forces_max_abs_eV_per_A"]


def test_oam_release_matrix_cannot_qualify_host_pair_outside_campaign(matrix):
    support = json.loads(json.dumps(matrix.load_manifest(MANIFEST)["nequip_support"]))
    cell = next(
        cell
        for cell in support["release_matrix"]
        if (cell["size"], cell["backend"], cell["precision"])
        == ("S", "host", "float64")
    )
    cell["state"] = "qualified"
    with pytest.raises(ValueError, match="outside host runtime qualification"):
        matrix._validate_nequip_support(support, evidence_root=MANIFEST.parent)


def test_oam_release_matrix_rejects_qualification_without_empirical_records(matrix):
    support = matrix.load_manifest(MANIFEST)["nequip_support"]
    support = json.loads(json.dumps(support))
    cell = support["release_matrix"][0]
    cell["state"] = "qualified"
    cell["runtime_qualification"] = "complete"
    cell["graph_classes"] = {name: "complete" for name in cell["graph_classes"]}
    cell["properties"] = {name: "qualified" for name in cell["properties"]}
    cell["baseline"] = {"status": "required", "record": None}
    with pytest.raises(ValueError, match="lacks complete baseline evidence"):
        matrix._validate_nequip_support(support, evidence_root=MANIFEST.parent)


def test_oam_release_matrix_rejects_relaxed_performance_gates(matrix):
    support = matrix.load_manifest(MANIFEST)["nequip_support"]
    support = json.loads(json.dumps(support))
    support["performance_gates"]["production_min_speedup_vs_reference"] = 1.0
    with pytest.raises(ValueError, match="approved thresholds"):
        matrix._validate_nequip_support(support, evidence_root=MANIFEST.parent)


def _qualified_manifest_with_evidence(
    matrix,
    tmp_path,
    *,
    target=("XL", "cuda", "float32"),
    record_override=None,
):
    manifest = json.loads(json.dumps(matrix.load_manifest(MANIFEST)))
    support = manifest["nequip_support"]
    cell = next(
        cell
        for cell in support["release_matrix"]
        if (cell["size"], cell["backend"], cell["precision"]) == target
    )
    cell["state"] = "qualified"
    cell["runtime_qualification"] = "complete"
    cell["graph_classes"] = {name: "complete" for name in cell["graph_classes"]}
    cell["properties"] = {name: "qualified" for name in cell["properties"]}
    if target == ("XL", "cuda", "float32"):
        cell["capacity_boundary"] = "complete"
    size, backend, precision = target
    identity = {
        "schema": "symmetrix.nequip-release-evidence",
        "version": 1,
        "model": f"nequip-oam-{size.lower()}",
        "backend": backend,
        "precision": precision,
        "graph_classes": list(cell["graph_classes"]),
        "properties": ["energy", "forces", "stress"],
    }
    metrics = (
        {
            "baseline": {},
            "correctness": {},
            "host_100_performance": {
                "speedup_vs_reference": 1.10,
                "ratio_vs_approved_baseline": 0.95,
            },
            "scaling": {"speedup_t1_over_t4": 1.01},
        }
        if backend == "host"
        else {
            "baseline": {},
            "correctness": {},
            "small_performance": {"speedup_vs_reference": 0.95},
            "production_performance": {
                "speedup_vs_reference": 1.10,
                "geomean_speedup_vs_reference": 1.20,
                "ratio_vs_approved_baseline": 0.95,
            },
            **(
                {"capacity": {}}
                if cell["capacity_boundary"] in {"required", "complete"}
                else {}
            ),
        }
    )
    for name in metrics:
        record = dict(identity, evidence_kind=name, metrics=metrics[name])
        if record_override is not None:
            record = record_override(name, record)
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(record, sort_keys=True))
        entry = {
            "status": "complete",
            "record": {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
        }
        if name == "baseline":
            cell["baseline"] = entry
        else:
            cell["evidence"][name] = entry
    cell["tested_compute_capabilities"] = [] if backend == "host" else [120]
    return manifest


def test_qualified_host_evidence_uses_host_100_and_strict_scaling_gate(
    matrix, tmp_path
):
    manifest = _qualified_manifest_with_evidence(
        matrix, tmp_path, target=("S", "host", "float32")
    )
    matrix._validate_manifest_nequip_support(manifest, evidence_root=tmp_path)

    scaling_path = tmp_path / "scaling.json"
    scaling = json.loads(scaling_path.read_text())
    scaling["metrics"]["speedup_t1_over_t4"] = 1.0
    scaling_path.write_text(json.dumps(scaling, sort_keys=True))
    cell = next(
        cell
        for cell in manifest["nequip_support"]["release_matrix"]
        if (cell["size"], cell["backend"], cell["precision"])
        == ("S", "host", "float32")
    )
    cell["evidence"]["scaling"]["record"]["sha256"] = hashlib.sha256(
        scaling_path.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="fails an approved performance gate"):
        matrix._validate_manifest_nequip_support(manifest, evidence_root=tmp_path)


@pytest.mark.parametrize(
    "record_override,match",
    [
        (lambda name, record: {"arbitrary": True}, "invalid evidence record"),
        (
            lambda name, record: (
                dict(record, backend="host") if name == "correctness" else record
            ),
            "identity mismatch",
        ),
        (
            lambda name, record: (
                dict(record, graph_classes=[*record["graph_classes"], "arbitrary"])
                if name == "correctness"
                else record
            ),
            "coverage mismatch",
        ),
        (
            lambda name, record: (
                dict(record, properties=[*record["properties"], "arbitrary"])
                if name == "correctness"
                else record
            ),
            "coverage mismatch",
        ),
    ],
)
def test_oam_qualification_rejects_hashed_but_wrong_evidence(
    matrix, tmp_path, record_override, match
):
    manifest = _qualified_manifest_with_evidence(
        matrix, tmp_path, record_override=record_override
    )
    with pytest.raises(ValueError, match=match):
        matrix._validate_manifest_nequip_support(manifest, evidence_root=tmp_path)


def test_oam_qualification_rejects_non_finite_evidence_metrics(matrix, tmp_path):
    def non_finite_production_metric(name, record):
        if name == "production_performance":
            record["metrics"]["speedup_vs_reference"] = float("nan")
        return record

    manifest = _qualified_manifest_with_evidence(
        matrix, tmp_path, record_override=non_finite_production_metric
    )
    with pytest.raises(ValueError, match="fails an approved performance gate"):
        matrix._validate_manifest_nequip_support(manifest, evidence_root=tmp_path)


def test_loaded_manifest_preserves_evidence_root_across_cwd(
    matrix, tmp_path, monkeypatch
):
    manifest = _qualified_manifest_with_evidence(matrix, tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    loaded = matrix.load_manifest(manifest_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    qualified = [
        cell
        for cell in matrix.nequip_support(loaded)["release_matrix"]
        if cell["state"] == "qualified"
    ]
    assert ("XL", "cuda", "float32") in {
        (cell["size"], cell["backend"], cell["precision"]) for cell in qualified
    }
    assert len(qualified) == 1


def test_requested_structures_are_deterministic_and_have_expected_size(matrix):
    manifest = matrix.load_manifest(MANIFEST)
    for name, expected in (("C64", 64), ("STO40", 40), ("H2O", 3)):
        assert manifest["structures"][name]["displacement_sigma_A"] == 0.03
        first = matrix.build_structure(name, manifest["structures"][name])
        second = matrix.build_structure(name, manifest["structures"][name])
        assert len(first) == expected
        np.testing.assert_array_equal(first.positions, second.positions)

    unshaken = dict(manifest["structures"]["STO40"], displacement_sigma_A=0.0)
    sto = matrix.build_structure("STO40", unshaken)
    expected_fractional = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
        )
    )
    np.testing.assert_allclose(sto.positions[:5] / 3.905, expected_fractional)

    water = matrix.build_structure(
        "H2O", dict(manifest["structures"]["H2O"], displacement_sigma_A=0.0)
    )
    assert water.pbc.tolist() == [False, False, False]
    np.testing.assert_allclose(
        sorted(water.get_all_distances()[0, 1:]), [0.96856502, 0.96856502]
    )


def test_model_families_select_only_their_declared_structures(matrix):
    manifest = matrix.load_manifest(MANIFEST)
    cases = list(
        matrix.matrix_cases(
            manifest,
            None,
            ["C64", "STO40", "H2O"],
            ["float32"],
            ["cpu"],
        )
    )
    structures_by_model = {}
    for model, structure, *_ in cases:
        structures_by_model.setdefault(model["id"], set()).add(structure)
    assert structures_by_model["mace-omat-0-small"] == {"C64", "STO40"}
    assert structures_by_model["mace-off23-small"] == {"C64", "H2O"}


def test_matrix_adds_only_the_declared_capacity_probe(matrix):
    manifest = matrix.load_manifest(MANIFEST)
    cases = list(
        matrix.matrix_cases(
            manifest,
            ["mace-mpa-0-medium"],
            ["STO40"],
            ["float32", "float64"],
            ["cpu", "cuda"],
        )
    )
    profiles = [(dtype, backend, profile) for _, _, dtype, backend, profile in cases]
    assert profiles.count(("float64", "cuda", "capacity")) == 1
    assert sum(profile == "capacity" for _, _, profile in profiles) == 1
    probes = manifest["capacity_probes"]
    assert {probe["model"] for probe in probes} == {
        "mace-omat-0-small",
        "mace-mpa-0-medium",
        "mace-off23-large",
    }
    off23_backends = {
        probe["backend"] for probe in probes if probe["model"] == "mace-off23-large"
    }
    assert off23_backends == {"cpu", "cuda", "hip"}


def _write_manifest_copy(tmp_path, mutate):
    manifest = json.loads(MANIFEST.read_text())
    mutate(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


# Values cite the captured molecule fixture (symmetrix/test/data/polar/
# mol5_manifest.json): n_k 62 over 5 atoms means 8 * 62 * 5 fp32 table bytes.
VALID_LONG_RANGE_BLOCK = {
    "n_k": 62,
    "table_bytes": 2480,
    "correction_mode": "makov-payne-dabo",
    "rebuild_on_cell_change": True,
}


def test_long_range_vocabulary_is_declared_and_optional_for_probes(matrix):
    manifest = matrix.load_manifest(MANIFEST)
    vocabulary = manifest["long_range_capacity"]
    assert vocabulary["version"] == matrix.LONG_RANGE_CAPACITY_VERSION
    assert set(vocabulary["correction_modes"]) == set(
        matrix.LONG_RANGE_CORRECTION_MODES
    )
    assert set(vocabulary["probe_fields"]) == {
        "n_k",
        "table_bytes",
        "correction_mode",
        "rebuild_on_cell_change",
    }
    # The vocabulary is additive: shipped probes carry no long-range block.
    assert all("long_range" not in probe for probe in manifest["capacity_probes"])


def test_probe_long_range_block_parses_and_survives_without_vocabulary(
    matrix, tmp_path
):
    def attach(manifest):
        manifest["capacity_probes"][0]["long_range"] = dict(VALID_LONG_RANGE_BLOCK)

    loaded = matrix.load_manifest(_write_manifest_copy(tmp_path, attach))
    assert loaded["capacity_probes"][0]["long_range"] == VALID_LONG_RANGE_BLOCK

    def drop_vocabulary(manifest):
        del manifest["long_range_capacity"]
        attach(manifest)

    # Old manifests without the vocabulary section still accept probe blocks.
    loaded = matrix.load_manifest(_write_manifest_copy(tmp_path, drop_vocabulary))
    assert loaded["capacity_probes"][0]["long_range"] == VALID_LONG_RANGE_BLOCK


@pytest.mark.parametrize(
    ("field", "value", "fragment"),
    [
        ("n_k", 0, "long-range n_k"),
        ("n_k", -3, "long-range n_k"),
        ("n_k", 61.5, "long-range n_k"),
        ("n_k", True, "long-range n_k"),
        ("n_k", None, "long-range n_k"),
        ("table_bytes", -8, "long-range table_bytes"),
        ("table_bytes", 2480.0, "long-range table_bytes"),
        ("table_bytes", True, "long-range table_bytes"),
        ("correction_mode", "ewald", "long-range correction mode"),
        ("correction_mode", None, "long-range correction mode"),
        ("rebuild_on_cell_change", "yes", "long-range rebuild trigger"),
        ("rebuild_on_cell_change", 1, "long-range rebuild trigger"),
    ],
)
def test_probe_long_range_fields_are_type_checked(
    matrix, tmp_path, field, value, fragment
):
    def attach(manifest):
        block = dict(VALID_LONG_RANGE_BLOCK)
        block[field] = value
        manifest["capacity_probes"][0]["long_range"] = block

    with pytest.raises(ValueError, match=fragment):
        matrix.load_manifest(_write_manifest_copy(tmp_path, attach))


def test_probe_long_range_block_requires_every_field(matrix, tmp_path):
    for field in VALID_LONG_RANGE_BLOCK:

        def attach(manifest, field=field):
            manifest["capacity_probes"][0]["long_range"] = {
                key: value
                for key, value in VALID_LONG_RANGE_BLOCK.items()
                if key != field
            }

        with pytest.raises(ValueError, match="long-range"):
            matrix.load_manifest(_write_manifest_copy(tmp_path, attach))


def test_long_range_vocabulary_version_and_modes_are_enforced(matrix, tmp_path):
    def bump_version(manifest):
        manifest["long_range_capacity"]["version"] += 1

    with pytest.raises(ValueError, match="long-range capacity vocabulary"):
        matrix.load_manifest(_write_manifest_copy(tmp_path, bump_version))

    def extend_modes(manifest):
        manifest["long_range_capacity"]["correction_modes"]["ewald3d"] = "extra"

    with pytest.raises(ValueError, match="disagree"):
        matrix.load_manifest(_write_manifest_copy(tmp_path, extend_modes))


def test_resume_keys_can_retry_only_failures(matrix, tmp_path):
    output = tmp_path / "results.jsonl"
    rows = [
        {"case_key": "a/C64/float32/cpu", "status": "passed"},
        {"case_key": "b/C64/float32/cpu", "status": "failed"},
        {"case_key": "c/STO40/float64/cuda", "status": "not_applicable"},
    ]
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert matrix.completed_keys(output) == {row["case_key"] for row in rows}
    assert matrix.completed_keys(output, retry_failures=True) == {
        rows[0]["case_key"],
        rows[2]["case_key"],
    }


def test_extraction_and_worker_commands_are_explicit(matrix, tmp_path):
    manifest = matrix.load_manifest(MANIFEST)
    model = next(row for row in manifest["models"] if row["id"] == "mace-mh-1")
    command = matrix.extraction_command(
        Path("converter-python"),
        SCRIPT,
        Path("mh1.model"),
        Path("mh1.json"),
        model,
        manifest["structures"]["STO40"],
    )
    assert command[-2:] == ["--head", "omat_pbe"]
    assert command[
        command.index("--chemical-symbols") + 1 : command.index("--output")
    ] == ["Sr", "Ti", "O"]
    args = SimpleNamespace(
        manifest=MANIFEST,
        cuda_selector="cuda13-sm120",
        hip_selector="hip-gfx1151",
        cpu_selector="cpu",
        jit_cache=tmp_path / "jit",
    )
    worker = matrix.worker_command(
        Path("cuda-python"),
        SCRIPT,
        args,
        model,
        "STO40",
        "float64",
        "cuda",
        Path("mh1.model"),
        Path("mh1.json"),
        Path("result.json"),
        "capacity",
        Path("float64-reference.json"),
    )
    assert worker[worker.index("--backend-selector") + 1] == "cuda13-sm120"
    assert worker[worker.index("--dtype") + 1] == "float64"
    assert worker[worker.index("--profile") + 1] == "capacity"
    assert worker[worker.index("--reference-output") + 1] == "float64-reference.json"

    hip_worker = matrix.worker_command(
        Path("hip-python"),
        SCRIPT,
        args,
        model,
        "STO40",
        "float32",
        "hip",
        Path("mh1.model"),
        Path("mh1.json"),
        Path("hip-result.json"),
    )
    assert hip_worker[hip_worker.index("--backend-selector") + 1] == "hip-gfx1151"
    assert hip_worker[hip_worker.index("--backend") + 1] == "hip"


def test_direct_identity_requires_generated_execution_and_zero_fallback(matrix):
    evaluator = SimpleNamespace(
        streamed_edges_mode="direct",
        factorized_jit_ready=True,
        factorized_fallback_evaluation_count=0,
        factorized_jit_forward_launch_count=1,
        factorized_jit_reverse_launch_count=2,
        factorized_selected_direct_forward_executor="jit_all",
        factorized_selected_direct_reverse_executor="jit",
        r0_implementation="generated",
        _r0_loader_implementation="host_plugin",
        r0_backend="host",
        r0_artifact_origin="runtime",
        r0_schedule="host-owner",
        r0_profile="owner-batched",
        r0_module_id="r0-artifact",
        m0_implementation="host_plugin",
        m0_module_id="m0-artifact",
        m1_recompute_tile_channels=16,
    )
    calculator = SimpleNamespace(
        evaluator=evaluator, jit_status="built", jit_artifact_id="artifact"
    )
    identity = matrix.direct_identity(calculator, "cpu")
    assert identity["gate_failures"] == []
    evaluator.factorized_fallback_evaluation_count = 1
    assert (
        "fallback evaluation count is 1"
        in matrix.direct_identity(calculator, "cpu")["gate_failures"]
    )
    evaluator.factorized_fallback_evaluation_count = 0
    evaluator.r0_implementation = "generic"
    evaluator._r0_loader_implementation = "generic"
    evaluator.r0_artifact_origin = None
    evaluator.r0_schedule = "generic"
    evaluator.r0_profile = "generic"
    evaluator.r0_module_id = ""
    assert matrix.direct_identity(calculator, "cpu", "speed")["gate_failures"] == []
    assert any(
        "R0 selected non-specialized implementation" in failure
        for failure in matrix.direct_identity(calculator, "cpu", "capacity")[
            "gate_failures"
        ]
    )


@pytest.mark.parametrize("retained_loader", ("builtin", "host_plugin", "device_module"))
def test_direct_identity_accepts_generic_dispatch_with_retained_r0_loader(
    matrix, retained_loader
):
    evaluator = SimpleNamespace(
        streamed_edges_mode="direct",
        factorized_jit_ready=True,
        factorized_fallback_evaluation_count=0,
        factorized_jit_forward_launch_count=1,
        factorized_jit_reverse_launch_count=2,
        factorized_selected_direct_forward_executor="jit_all",
        factorized_selected_direct_reverse_executor="jit",
        r0_implementation="generic",
        _r0_loader_implementation=retained_loader,
        r0_backend="host" if retained_loader != "device_module" else "cuda",
        r0_artifact_origin=None,
        r0_schedule="generic",
        r0_profile="generic",
        r0_module_id="",
        m0_implementation="host_plugin",
        m0_module_id="m0-artifact",
        m1_recompute_tile_channels=16,
    )
    calculator = SimpleNamespace(
        evaluator=evaluator, jit_status="built", jit_artifact_id="artifact"
    )
    backend = "cuda" if retained_loader == "device_module" else "cpu"

    assert matrix.direct_identity(calculator, backend)["gate_failures"] == []


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("r0_backend", "cuda"),
        ("r0_artifact_origin", None),
        ("r0_schedule", "receiver-edge16"),
        ("r0_profile", "persistent-4"),
        ("r0_module_id", ""),
        ("r0_implementation", "generic"),
        ("_r0_loader_implementation", "device_module"),
    ),
)
def test_direct_identity_rejects_inconsistent_r0_evidence(matrix, field, value):
    evaluator = SimpleNamespace(
        streamed_edges_mode="direct",
        factorized_jit_ready=True,
        factorized_fallback_evaluation_count=0,
        factorized_jit_forward_launch_count=1,
        factorized_jit_reverse_launch_count=2,
        factorized_selected_direct_forward_executor="jit_all",
        factorized_selected_direct_reverse_executor="jit",
        r0_implementation="generated",
        _r0_loader_implementation="host_plugin",
        r0_backend="host",
        r0_artifact_origin="runtime",
        r0_schedule="host-owner",
        r0_profile="owner-batched",
        r0_module_id="r0-artifact",
        m0_implementation="host_plugin",
        m0_module_id="m0-artifact",
        m1_recompute_tile_channels=16,
    )
    setattr(evaluator, field, value)
    calculator = SimpleNamespace(
        evaluator=evaluator, jit_status="built", jit_artifact_id="artifact"
    )

    assert matrix.direct_identity(calculator, "cpu")["gate_failures"]


def test_mh1_identity_requires_all_generated_reverse_families(matrix):
    evaluator = SimpleNamespace(
        streamed_edges_mode="direct",
        factorized_fallback_evaluation_count=0,
        jit_mh1_cuda_plugin_ready=True,
        execution_mh1_generated_forward_launch_count=1,
        execution_mh1_generated_source_reverse_launch_count=2,
        execution_mh1_generated_edge_reverse_launch_count=2,
        execution_mh1_execution_backend="generated_cuda",
        mh1_edge_executor="mlp_reference",
    )
    calculator = SimpleNamespace(
        evaluator=evaluator, jit_status="cached", jit_artifact_id="mh1-artifact"
    )
    assert matrix.direct_identity(calculator, "cuda")["gate_failures"] == []
    evaluator.execution_mh1_generated_edge_reverse_launch_count = 0
    assert (
        "generated forward/reverse execution was not observed"
        in matrix.direct_identity(calculator, "cuda")["gate_failures"]
    )
    evaluator.execution_mh1_generated_edge_reverse_launch_count = 2
    evaluator.execution_mh1_execution_backend = "generated_host_v4"
    assert any(
        "expected generated cuda" in failure
        for failure in matrix.direct_identity(calculator, "cuda")["gate_failures"]
    )


def test_mh1_identity_accepts_generated_host_v5_readiness(matrix):
    evaluator = SimpleNamespace(
        streamed_edges_mode="direct",
        factorized_fallback_evaluation_count=0,
        jit_mh1_host_plugin_ready=False,
        jit_mh1_host_plugin_v4_ready=False,
        jit_mh1_host_plugin_v5_ready=True,
        execution_mh1_generated_forward_launch_count=1,
        execution_mh1_generated_source_reverse_launch_count=2,
        execution_mh1_generated_edge_reverse_launch_count=2,
        execution_mh1_execution_backend="generated_host_v5",
        mh1_edge_executor="mlp_reference",
    )
    calculator = SimpleNamespace(
        evaluator=evaluator, jit_status="cached", jit_artifact_id="mh1-v5"
    )
    assert matrix.direct_identity(calculator, "cpu")["gate_failures"] == []


def test_mh1_hip_identity_accepts_device_plugin_and_hip_family(matrix):
    evaluator = SimpleNamespace(
        streamed_edges_mode="direct",
        factorized_fallback_evaluation_count=0,
        jit_mh1_cuda_plugin_ready=False,
        jit_mh1_cuda_plugin_v4_ready=True,
        execution_mh1_generated_forward_launch_count=1,
        execution_mh1_generated_source_reverse_launch_count=1,
        execution_mh1_generated_edge_reverse_launch_count=1,
        execution_mh1_execution_backend="generated_hip_v4",
        mh1_edge_executor="runtime",
    )
    calculator = SimpleNamespace(
        evaluator=evaluator, jit_status="built", jit_artifact_id="hip-artifact"
    )
    assert matrix.direct_identity(calculator, "hip")["gate_failures"] == []


def test_numerical_errors_are_normalized(matrix):
    reference = {"energy": 8.0, "forces": np.zeros((2, 3)), "stress": np.zeros(6)}
    candidate = {
        "energy": 8.2,
        "forces": np.ones((2, 3)) * 0.1,
        "stress": np.ones(6) * 0.2,
    }
    errors = matrix.numerical_errors(reference, candidate, atoms=2)
    assert errors["energy_abs_eV_per_atom"] == pytest.approx(0.1)
    assert errors["forces_max_abs_eV_per_A"] == pytest.approx(0.1)
    assert errors["stress_max_abs_eV_per_A3"] == pytest.approx(0.2)


def test_cross_precision_assessment_is_an_independent_gate(matrix):
    reference = {"energy": 8.0, "forces": np.zeros((2, 3)), "stress": np.zeros(6)}
    candidate = {
        "energy": 8.0002,
        "forces": np.ones((2, 3)) * 2e-4,
        "stress": np.ones(6) * 3e-5,
    }
    tolerances = {
        "energy_abs_eV_per_atom": 2e-4,
        "forces_max_abs_eV_per_A": 1e-4,
        "stress_max_abs_eV_per_A3": 1e-4,
    }
    assessment, failures = matrix.cross_precision_assessment(
        reference, candidate, 2, tolerances
    )
    assert assessment["status"] == "checked"
    assert failures == [
        "direct FP32 versus direct FP64 forces_max_abs_eV_per_A exceeds tolerance"
    ]


def test_known_checkpoint_hash_is_checked_before_extraction(
    matrix, monkeypatch, tmp_path
):
    manifest = matrix.load_manifest(MANIFEST)
    model = next(row for row in manifest["models"] if row["sha256"])
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / model["filename"]).write_bytes(b"wrong checkpoint")
    output = tmp_path / "results.jsonl"
    subprocess_calls = []
    monkeypatch.setattr(
        matrix.subprocess,
        "run",
        lambda *args, **kwargs: subprocess_calls.append((args, kwargs)),
    )
    args = SimpleNamespace(
        manifest=MANIFEST,
        output_jsonl=output,
        retry_failures=False,
        models=[model["id"]],
        structures=["C64"],
        dtypes=["float32"],
        backends=["cpu"],
        checkpoint_dir=checkpoint_dir,
        compact_dir=tmp_path / "compact",
        converter_python=Path("converter-python"),
        cpu_python=Path("cpu-python"),
        cuda_python=Path("cuda-python"),
        cpu_selector="cpu",
        cuda_selector="cuda13-sm120",
        jit_cache=tmp_path / "jit",
        reuse_compact=False,
    )
    assert matrix.orchestrate(args) == 1
    assert subprocess_calls == []
    record = json.loads(output.read_text())
    assert record["status"] == "failed"
    assert record["checkpoint"]["sha256"] == matrix.sha256_file(
        checkpoint_dir / model["filename"]
    )


def test_nonzero_worker_exit_invalidates_written_result(matrix, tmp_path):
    output = tmp_path / "worker.json"
    output.write_text(
        json.dumps(
            {
                "case_key": "model/C64/float32/cuda",
                "status": "passed",
                "failures": [],
            }
        )
    )
    result = SimpleNamespace(returncode=9, stderr="backend teardown failed")
    record = matrix.collect_worker_record(
        output, result, "model", "C64", "float32", "cuda"
    )
    assert record["status"] == "failed"
    assert record["failures"] == [
        "worker exited 9 after writing output: backend teardown failed"
    ]


def test_missing_capacity_worker_result_preserves_profile(matrix, tmp_path):
    result = SimpleNamespace(returncode=9, stderr="device launch failed")
    record = matrix.collect_worker_record(
        tmp_path / "missing.json",
        result,
        "model",
        "C64",
        "float64",
        "cuda",
        "capacity",
    )

    assert record["case_key"] == "model/C64/float64/cuda/capacity"
    assert record["execution_profile"] == "capacity"
