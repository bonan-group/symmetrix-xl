import copy
import hashlib
import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
VALIDATOR = ROOT / "benchmarks/validate_mh1_serial_record.py"


def _load_validator():
    assert VALIDATOR.is_file(), "the committed post-run validator is missing"
    spec = importlib.util.spec_from_file_location(
        "validate_mh1_serial_record", VALIDATOR
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record():
    return {
        "metadata": {
            "git_revision": "a" * 40,
            "git_dirty": False,
            "exact_command": "python benchmark.py --product-timings-json timing.json",
            "timed_properties": ["energy", "forces", "stress"],
            "warmups": 10,
            "repeats": 30,
            "model": {"sha256": "5" * 64},
            "native_extension_sha256": "d" * 64,
            "native_build_info": {
                "source_commit": "a" * 40,
                "source_content_sha256": "b" * 64,
                "native_source_content_sha256": "c" * 64,
                "source_dirty": False,
            },
            "mh1_product_schedule_overrides": {
                "M0": {"effective": "chunk32"},
                "M1": {"effective": "chunk32"},
            },
            "replay": {
                "python_executable": "/venv/bin/python",
                "launcher": {
                    "kind": "in-process-sched-setaffinity",
                    "requested_cpu_affinity": [0],
                    "effective_cpu_affinity": [0],
                },
                "environment": {"SYMMETRIX_JIT_POLICY": "required"},
            },
        },
        "systems": [
            {
                "atoms": 256,
                "directed_edges": 27904,
                "system_sha256": "6" * 64,
                "timed_properties": ["energy", "forces", "stress"],
                "energy_sha256": "7" * 64,
                "forces_sha256": "8" * 64,
                "stress_sha256": "9" * 64,
                "factorized_fallback_evaluation_count": 0,
                "evaluator": {"samples_ms": [1.0] * 30},
                "mh1_product_runtime": {
                    "full_plugin": {
                        "artifact_id": "plugin",
                        "artifact_path": "/artifacts/plugin/module.cubin",
                        "artifact_sha256": "e" * 64,
                        "manifest_sha256": "f" * 64,
                        "module_identity": "sha256:plugin-module",
                        "abi": {"version": 5},
                        "jit_generation_version": 19,
                        "launch_plan_schema": "symmetrix.jit.mh1.cuda-launch-plan/6",
                        "launch_plan_version": 6,
                        "split_phase_composition": True,
                    },
                    "fallback_count": 0,
                    "layers": [
                        {
                            "layer": layer,
                            "operator": operator,
                            "origin": "runtime",
                            "artifact_id": artifact,
                            "artifact_path": f"/artifacts/{artifact}/module.cubin",
                            "artifact_sha256": digest * 64,
                            "manifest_sha256": manifest * 64,
                            "artifact_provenance": {
                                "kind": "runtime-manifest",
                                "manifest_path": f"/artifacts/{artifact}/manifest.json",
                                "manifest_sha256": manifest * 64,
                                "catalog_path": None,
                                "catalog_sha256": None,
                                "catalog_entry_sha256": None,
                            },
                            "schedule": "chunk32",
                            "abi": {"version": 3},
                            "jit_generation_version": 19,
                            "forward_launch_count": 30,
                            "reverse_launch_count": 30,
                            "launch_counter_scope": {
                                "scope": "benchmark-warmups-and-timed-samples",
                                "warmups": 10,
                                "samples": 30,
                                "warmup_forward_launch_count": 10,
                                "warmup_reverse_launch_count": 10,
                                "timed_forward_launch_count": 30,
                                "timed_reverse_launch_count": 30,
                                "total_forward_launch_count": 40,
                                "total_reverse_launch_count": 40,
                            },
                            "memory_accounting": {
                                "scratch_bytes": None,
                                "retained_bytes": None,
                                "unavailable_reason": "composed arena only",
                            },
                            "realized_geometry": {
                                "forward_logical_tiles": 8,
                                "forward_tiles_per_block": 4,
                                "forward_required_blocks": 2,
                                "forward_realized_blocks": 2,
                                "reverse_logical_tiles": 8,
                                "reverse_tiles_per_block": 4,
                                "reverse_required_blocks": 2,
                                "reverse_realized_blocks": 2,
                            },
                            "contract_dimensions": {
                                "channels": 8,
                                "type_count": 1,
                                "input_components": 16,
                                "output_components": 4,
                                "correlation": 3,
                                "term_count": 7,
                                "coefficient_count": 56,
                            },
                            "identity": {
                                "generation_fingerprint": "sha256:g",
                                "semantic_fingerprint": "sha256:s",
                                "structure_fingerprint": "sha256:t",
                            },
                        }
                        for layer, operator, artifact, digest, manifest in (
                            (0, "M0", "m0", "1", "2"),
                            (1, "M1", "m1", "3", "4"),
                        )
                    ],
                },
            }
        ],
    }


def _expected_layers():
    record_layers = _record()["systems"][0]["mh1_product_runtime"]["layers"]
    return [
        {
            "artifact_sha256": layer["artifact_sha256"],
            "artifact_path": layer["artifact_path"],
            "manifest_sha256": layer["manifest_sha256"],
            "origin": layer["origin"],
            "artifact_provenance": layer["artifact_provenance"],
            "identity": layer["identity"],
            "contract_dimensions": layer["contract_dimensions"],
        }
        for layer in record_layers
    ]


def _expected_full_plugin():
    plugin = _record()["systems"][0]["mh1_product_runtime"]["full_plugin"]
    return {
        "artifact_id": plugin["artifact_id"],
        "artifact_path": plugin["artifact_path"],
        "artifact_sha256": plugin["artifact_sha256"],
        "manifest_path": "/artifacts/plugin/manifest.json",
        "manifest_sha256": plugin["manifest_sha256"],
        "module_identity": plugin["module_identity"],
    }


def _expected_outputs():
    system = _record()["systems"][0]
    return {
        name: system[name]
        for name in ("energy_sha256", "forces_sha256", "stress_sha256")
    }


def test_validator_accepts_exact_release_record():
    validator = _load_validator()
    summary = validator.validate_record(
        _record(),
        expected_commit="a" * 40,
        expected_source_sha256="b" * 64,
        expected_native_sha256="c" * 64,
        expected_extension_sha256="d" * 64,
        expected_properties=("energy", "forces", "stress"),
        expected_samples=30,
        expected_atoms=256,
        expected_directed_edges=27904,
        expected_artifact_ids=("m0", "m1"),
        expected_full_plugin=_expected_full_plugin(),
        expected_layers=_expected_layers(),
        expected_outputs=_expected_outputs(),
    )
    assert summary["status"] == "ok"


def test_validator_fails_closed_on_artifact_identity_drift():
    validator = _load_validator()
    record = _record()
    record["systems"][0]["mh1_product_runtime"]["layers"][1]["artifact_id"] = "wrong"
    with pytest.raises(validator.ValidationError, match="artifact identity"):
        validator.validate_record(
            record,
            expected_commit="a" * 40,
            expected_source_sha256="b" * 64,
            expected_native_sha256="c" * 64,
            expected_extension_sha256="d" * 64,
            expected_properties=("energy", "forces", "stress"),
            expected_samples=30,
            expected_atoms=256,
            expected_directed_edges=27904,
            expected_artifact_ids=("m0", "m1"),
            expected_full_plugin=_expected_full_plugin(),
            expected_layers=_expected_layers(),
            expected_outputs=_expected_outputs(),
        )


@pytest.mark.parametrize("name", ("energy", "forces", "stress"))
def test_validator_fails_closed_on_output_hash_drift(name):
    validator = _load_validator()
    record = _record()
    record["systems"][0][f"{name}_sha256"] = "0" * 64
    with pytest.raises(validator.ValidationError, match=f"{name} output hash"):
        validator.validate_record(
            record,
            expected_commit="a" * 40,
            expected_source_sha256="b" * 64,
            expected_native_sha256="c" * 64,
            expected_extension_sha256="d" * 64,
            expected_properties=("energy", "forces", "stress"),
            expected_samples=30,
            expected_atoms=256,
            expected_directed_edges=27904,
            expected_artifact_ids=("m0", "m1"),
            expected_full_plugin=_expected_full_plugin(),
            expected_layers=_expected_layers(),
            expected_outputs=_expected_outputs(),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda record: record["metadata"]["replay"].update(
                launcher={"kind": "inherited-affinity", "effective_cpu_affinity": [0]}
            ),
            "explicit in-process",
        ),
        (
            lambda record: record["systems"][0]["mh1_product_runtime"]["layers"][0][
                "contract_dimensions"
            ].update(type_count=2),
            "type_count",
        ),
        (
            lambda record: record["systems"][0]["mh1_product_runtime"]["layers"][
                0
            ].update(artifact_sha256="9" * 64),
            "artifact hash",
        ),
        (
            lambda record: record["systems"][0]["mh1_product_runtime"]["layers"][
                0
            ].update(artifact_path="/wrong/module.cubin"),
            "artifact path",
        ),
        (
            lambda record: record["systems"][0]["mh1_product_runtime"]["layers"][
                0
            ].update(manifest_sha256="9" * 64),
            "manifest hash",
        ),
        (
            lambda record: record["systems"][0]["mh1_product_runtime"]["layers"][
                0
            ].update(origin="prebuilt"),
            "origin",
        ),
        (
            lambda record: record["systems"][0]["mh1_product_runtime"][
                "full_plugin"
            ].update(module_identity="sha256:wrong"),
            "full-plugin",
        ),
        (
            lambda record: record["systems"][0]["mh1_product_runtime"][
                "full_plugin"
            ].update(manifest_sha256="9" * 64),
            "full-plugin",
        ),
        (
            lambda record: record["systems"][0]["mh1_product_runtime"]["layers"][0][
                "artifact_provenance"
            ].update(manifest_sha256="9" * 64),
            "provenance",
        ),
    ],
)
def test_validator_rejects_unbound_release_evidence(mutation, message):
    validator = _load_validator()
    record = _record()
    mutation(record)
    with pytest.raises(validator.ValidationError, match=message):
        validator.validate_record(
            record,
            expected_commit="a" * 40,
            expected_source_sha256="b" * 64,
            expected_native_sha256="c" * 64,
            expected_extension_sha256="d" * 64,
            expected_properties=("energy", "forces", "stress"),
            expected_samples=30,
            expected_atoms=256,
            expected_directed_edges=27904,
            expected_artifact_ids=("m0", "m1"),
            expected_full_plugin=_expected_full_plugin(),
            expected_layers=_expected_layers(),
            expected_outputs=_expected_outputs(),
        )


def test_validator_accepts_and_binds_installed_catalog_provenance():
    validator = _load_validator()
    record = _record()
    layer = record["systems"][0]["mh1_product_runtime"]["layers"][0]
    layer["manifest_sha256"] = None
    layer["origin"] = "prebuilt"
    layer["artifact_provenance"] = {
        "kind": "installed-catalog",
        "manifest_path": None,
        "manifest_sha256": None,
        "catalog_path": "/installed/catalog.json",
        "catalog_sha256": "7" * 64,
        "catalog_entry_sha256": "8" * 64,
    }
    expected_layers = _expected_layers()
    expected_layers[0] = {
        "artifact_sha256": layer["artifact_sha256"],
        "artifact_path": layer["artifact_path"],
        "manifest_sha256": layer["manifest_sha256"],
        "origin": layer["origin"],
        "artifact_provenance": copy.deepcopy(layer["artifact_provenance"]),
        "identity": layer["identity"],
        "contract_dimensions": layer["contract_dimensions"],
    }

    validator.validate_record(
        record,
        expected_commit="a" * 40,
        expected_source_sha256="b" * 64,
        expected_native_sha256="c" * 64,
        expected_extension_sha256="d" * 64,
        expected_properties=("energy", "forces", "stress"),
        expected_samples=30,
        expected_atoms=256,
        expected_directed_edges=27904,
        expected_artifact_ids=("m0", "m1"),
        expected_full_plugin=_expected_full_plugin(),
        expected_layers=expected_layers,
        expected_outputs=_expected_outputs(),
    )

    layer["artifact_provenance"]["catalog_sha256"] = "9" * 64
    with pytest.raises(validator.ValidationError, match="provenance"):
        validator.validate_record(
            record,
            expected_commit="a" * 40,
            expected_source_sha256="b" * 64,
            expected_native_sha256="c" * 64,
            expected_extension_sha256="d" * 64,
            expected_properties=("energy", "forces", "stress"),
            expected_samples=30,
            expected_atoms=256,
            expected_directed_edges=27904,
            expected_artifact_ids=("m0", "m1"),
            expected_full_plugin=_expected_full_plugin(),
            expected_layers=expected_layers,
            expected_outputs=_expected_outputs(),
        )


@pytest.mark.parametrize(
    ("origin", "provenance_kind"),
    (("prebuilt", "runtime-manifest"), ("runtime", "installed-catalog")),
)
def test_validator_rejects_cross_origin_provenance(origin, provenance_kind):
    validator = _load_validator()
    record = _record()
    layer = record["systems"][0]["mh1_product_runtime"]["layers"][0]
    layer["origin"] = origin
    if provenance_kind == "installed-catalog":
        layer["manifest_sha256"] = None
        layer["artifact_provenance"] = {
            "kind": "installed-catalog",
            "manifest_path": None,
            "manifest_sha256": None,
            "catalog_path": "/installed/catalog.json",
            "catalog_sha256": "7" * 64,
            "catalog_entry_sha256": "8" * 64,
        }
    expected_layers = _expected_layers()
    expected_layers[0] = {
        "artifact_sha256": layer["artifact_sha256"],
        "artifact_path": layer["artifact_path"],
        "manifest_sha256": layer["manifest_sha256"],
        "origin": layer["origin"],
        "artifact_provenance": copy.deepcopy(layer["artifact_provenance"]),
        "identity": layer["identity"],
        "contract_dimensions": layer["contract_dimensions"],
    }
    with pytest.raises(validator.ValidationError, match="origin.*provenance"):
        validator.validate_record(
            record,
            expected_commit="a" * 40,
            expected_source_sha256="b" * 64,
            expected_native_sha256="c" * 64,
            expected_extension_sha256="d" * 64,
            expected_properties=("energy", "forces", "stress"),
            expected_samples=30,
            expected_atoms=256,
            expected_directed_edges=27904,
            expected_artifact_ids=("m0", "m1"),
            expected_full_plugin=_expected_full_plugin(),
            expected_layers=expected_layers,
            expected_outputs=_expected_outputs(),
        )


def test_validator_rejects_arbitrary_product_timing_blob():
    validator = _load_validator()
    record = _record()
    record["systems"][0]["mh1_product_runtime"]["product_timings"] = {
        "source": "",
        "statistic": "unspecified",
        "layers": [],
    }
    with pytest.raises(validator.ValidationError, match="expectations"):
        validator.validate_record(
            record,
            expected_commit="a" * 40,
            expected_source_sha256="b" * 64,
            expected_native_sha256="c" * 64,
            expected_extension_sha256="d" * 64,
            expected_properties=("energy", "forces", "stress"),
            expected_samples=30,
            expected_atoms=256,
            expected_directed_edges=27904,
            expected_artifact_ids=("m0", "m1"),
            expected_full_plugin=_expected_full_plugin(),
            expected_layers=_expected_layers(),
            expected_outputs=_expected_outputs(),
        )


def _attach_realistic_product_timing(record, tmp_path):
    profile = tmp_path / "profile.nsys-rep"
    report = tmp_path / "cuda_gpu_kern_sum.csv"
    profiled_path = tmp_path / "profiled-record.json"
    profile.write_bytes(b"NSYSREP\x00minimal-test-profile")
    report.write_text(
        '"Time (%)","Total Time (ns)","Instances","Avg (ns)","Med (ns)",'
        '"Min (ns)","Max (ns)","StdDev (ns)","Name"\n'
        '"20","10000","5","2000","2000","1900","2100","50",'
        '"symmetrix_mh1_m0_forward"\n'
        '"30","15000","5","3000","3000","2900","3100","50",'
        '"symmetrix_mh1_m0_reverse"\n'
        '"20","20000","5","4000","4000","3900","4100","50",'
        '"symmetrix_mh1_m1_forward"\n'
        '"30","25000","5","5000","5000","4900","5100","50",'
        '"symmetrix_mh1_m1_reverse"\n'
    )
    profiled_command = "python benchmark.py --output profiled.json"
    attachment_command = record["metadata"]["exact_command"]
    profile_command = f"nsys profile --trace=cuda {profiled_command}"
    profiled = copy.deepcopy(record)
    profiled["metadata"]["exact_command"] = profiled_command
    profiled["metadata"]["repeats"] = 5
    profiled["systems"][0]["evaluator"]["samples_ms"] = [1.0] * 5
    profiled["systems"][0]["mh1_product_runtime"]["product_timings"] = None
    profiled_path.write_text(json.dumps(profiled))
    runtime = record["systems"][0]["mh1_product_runtime"]
    bound_layers = []
    timings = []
    for index, layer in enumerate(runtime["layers"]):
        operator = layer["operator"].lower()
        bound_layers.append(
            {
                "layer": index,
                "operator": layer["operator"],
                "origin": layer["origin"],
                "artifact_id": layer["artifact_id"],
                "artifact_path": layer["artifact_path"],
                "artifact_sha256": layer["artifact_sha256"],
                "manifest_sha256": layer["manifest_sha256"],
                "artifact_provenance": layer["artifact_provenance"],
                "profile_kernels": {
                    "forward": f"symmetrix_mh1_{operator}_forward",
                    "reverse": f"symmetrix_mh1_{operator}_reverse",
                },
            }
        )
        timings.append(
            {
                "layer": index,
                "forward_us": float(2 + 2 * index),
                "reverse_us": float(3 + 2 * index),
            }
        )
    binding = {
        "git_revision": "a" * 40,
        "source_content_sha256": "b" * 64,
        "native_source_content_sha256": "c" * 64,
        "native_extension_sha256": "d" * 64,
        "profiled_command": profiled_command,
        "profiled_command_sha256": hashlib.sha256(
            profiled_command.encode()
        ).hexdigest(),
        "attachment_command": attachment_command,
        "attachment_command_sha256": hashlib.sha256(
            attachment_command.encode()
        ).hexdigest(),
        "profile_command": profile_command,
        "profile_command_sha256": hashlib.sha256(profile_command.encode()).hexdigest(),
        "profiled_record_path": str(profiled_path),
        "profiled_record_sha256": hashlib.sha256(
            profiled_path.read_bytes()
        ).hexdigest(),
        "profile_path": str(profile),
        "profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
        "report_path": str(report),
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "model_sha256": "5" * 64,
        "system_sha256": "6" * 64,
        "properties": ["energy", "forces", "stress"],
        "sample_scope": {"warmups": 10, "samples": 5},
        "full_plugin": {
            key: runtime["full_plugin"][key]
            for key in (
                "artifact_id",
                "artifact_sha256",
                "manifest_sha256",
                "module_identity",
            )
        },
        "layers": bound_layers,
    }
    evidence = {
        "source": "Nsight Systems 2026.1.3 cuda_gpu_kern_sum",
        "statistic": "median",
        "binding": binding,
        "layers": timings,
    }
    runtime["product_timings"] = evidence
    return evidence


def test_validator_externally_binds_and_parses_product_timings(tmp_path):
    validator = _load_validator()
    record = _record()
    timing = _attach_realistic_product_timing(record, tmp_path)
    kwargs = {
        "expected_commit": "a" * 40,
        "expected_source_sha256": "b" * 64,
        "expected_native_sha256": "c" * 64,
        "expected_extension_sha256": "d" * 64,
        "expected_properties": ("energy", "forces", "stress"),
        "expected_samples": 30,
        "expected_atoms": 256,
        "expected_directed_edges": 27904,
        "expected_artifact_ids": ("m0", "m1"),
        "expected_full_plugin": _expected_full_plugin(),
        "expected_layers": _expected_layers(),
        "expected_outputs": _expected_outputs(),
        "expected_product_timings": copy.deepcopy(timing),
    }
    assert validator.validate_record(record, **kwargs)["status"] == "ok"

    for path, replacement, message in (
        (("binding", "profile_command"), "nsys profile forged", "profile_command"),
        (("binding", "git_revision"), "9" * 40, "git_revision"),
        (("binding", "source_content_sha256"), "9" * 64, "source_content"),
        (
            ("binding", "native_source_content_sha256"),
            "9" * 64,
            "native_source_content",
        ),
        (
            ("binding", "native_extension_sha256"),
            "9" * 64,
            "native_extension",
        ),
        (("binding", "model_sha256"), "9" * 64, "model_sha256"),
        (("binding", "system_sha256"), "9" * 64, "system_sha256"),
        (("binding", "properties"), ["energy"], "properties"),
        (("binding", "sample_scope", "samples"), 4, "sample_scope"),
        (
            ("binding", "full_plugin", "manifest_sha256"),
            "9" * 64,
            "full_plugin",
        ),
        (
            ("binding", "full_plugin", "module_identity"),
            "sha256:wrong",
            "module_identity",
        ),
        (
            ("binding", "layers", 0, "artifact_provenance", "manifest_sha256"),
            "9" * 64,
            "provenance",
        ),
        (
            ("binding", "layers", 0, "artifact_path"),
            "/jit/forged/module.cubin",
            "expectation",
        ),
        (
            ("binding", "layers", 0, "manifest_sha256"),
            "9" * 64,
            "expectation",
        ),
        (
            ("binding", "layers", 0, "origin"),
            "prebuilt",
            "expectation",
        ),
    ):
        mutated = copy.deepcopy(record)
        target = mutated["systems"][0]["mh1_product_runtime"]["product_timings"]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = replacement
        with pytest.raises(validator.ValidationError, match=message):
            validator.validate_record(mutated, **kwargs)
