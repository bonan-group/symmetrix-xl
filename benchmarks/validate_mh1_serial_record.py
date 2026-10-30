#!/usr/bin/env python3
"""Fail-closed validation for an MH-1 serial benchmark release record."""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


class ValidationError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise ValidationError(message)


def _sha256(value, label):
    _require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"{label} must be a lowercase SHA-256",
    )
    return value


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _require_expected(actual, expected, label):
    if isinstance(expected, dict):
        _require(isinstance(actual, dict), f"{label} must be an object")
        for key, value in expected.items():
            _require(key in actual, f"{label}.{key} is missing")
            _require_expected(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        _require(
            isinstance(actual, list) and len(actual) == len(expected),
            f"{label} list does not match",
        )
        for index, value in enumerate(expected):
            _require_expected(actual[index], value, f"{label}[{index}]")
    else:
        _require(actual == expected, f"{label} does not match")


def _validate_product_timings(
    timings, metadata, system, plugin, layers, expected_product_timings
):
    if timings is None:
        _require(
            expected_product_timings is None,
            "product timing expectations were provided without timings",
        )
        return
    _require(
        isinstance(expected_product_timings, dict),
        "external product timing expectations are required",
    )
    _require(
        bool(expected_product_timings.get("source"))
        and expected_product_timings.get("statistic") == "median",
        "external product timing source/statistic expectations are required",
    )
    expected_binding = expected_product_timings.get("binding")
    _require(
        isinstance(expected_binding, dict),
        "external product timing binding expectation is required",
    )
    for field in (
        "git_revision",
        "source_content_sha256",
        "native_source_content_sha256",
        "native_extension_sha256",
        "profiled_command",
        "profiled_command_sha256",
        "attachment_command",
        "attachment_command_sha256",
        "profile_command",
        "profile_command_sha256",
        "profiled_record_sha256",
        "profile_sha256",
        "report_sha256",
        "model_sha256",
        "system_sha256",
        "properties",
        "sample_scope",
        "full_plugin",
        "layers",
    ):
        _require(
            field in expected_binding,
            f"external product timing expectation lacks {field}",
        )
    _require(isinstance(timings, dict), "product timings must be an object")
    _require_expected(timings, expected_product_timings, "product timing expectation")
    _require(bool(timings.get("source")), "product timing source is missing")
    _require(
        bool(timings.get("statistic")) and timings.get("statistic") != "unspecified",
        "product timing statistic is missing",
    )
    binding = timings.get("binding")
    _require(isinstance(binding, dict), "product timing binding is missing")
    for command in ("profiled_command", "attachment_command", "profile_command"):
        _require(bool(binding.get(command)), f"product timing {command} is missing")
        _require(
            _text_sha256(binding[command])
            == _sha256(
                binding.get(f"{command}_sha256"),
                f"product timing {command} hash",
            ),
            f"product timing {command.replace('_', ' ')} hash does not match",
        )
    _require(
        binding.get("attachment_command") == metadata.get("exact_command"),
        "product timing attachment command does not match",
    )
    _require(
        binding.get("model_sha256") == metadata.get("model", {}).get("sha256"),
        "product timing model binding does not match",
    )
    _require(
        binding.get("system_sha256") == system.get("system_sha256"),
        "product timing system binding does not match",
    )
    _require(
        binding.get("properties") == system.get("timed_properties"),
        "product timing properties binding does not match",
    )
    _require(
        binding.get("full_plugin")
        == {
            key: plugin.get(key)
            for key in (
                "artifact_id",
                "artifact_sha256",
                "manifest_sha256",
                "module_identity",
            )
        },
        "product timing full-plugin binding does not match",
    )
    for name in ("profiled_record", "profile", "report"):
        path = Path(binding.get(f"{name}_path", ""))
        expected = _sha256(binding.get(f"{name}_sha256"), f"product timing {name} hash")
        _require(path.is_file(), f"product timing {name} file is missing")
        _require(
            _file_sha256(path) == expected,
            f"product timing {name} hash does not match",
        )
    try:
        profiled = json.loads(Path(binding["profiled_record_path"]).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"profiled record is unreadable: {error}") from error
    profiled_systems = profiled.get("systems", [])
    _require(len(profiled_systems) == 1, "profiled record must have one system")
    _require(
        profiled.get("metadata", {}).get("git_revision") == binding.get("git_revision"),
        "profiled record commit binding does not match",
    )
    _require(
        profiled.get("metadata", {}).get("git_dirty") is False,
        "profiled record must come from a clean tree",
    )
    profiled_build = profiled.get("metadata", {}).get("native_build_info", {})
    _require(
        profiled_build.get("source_commit") == binding.get("git_revision")
        and profiled_build.get("source_dirty") is False,
        "profiled native build commit binding does not match",
    )
    _require(
        profiled_build.get("source_content_sha256")
        == binding.get("source_content_sha256"),
        "profiled record source hash binding does not match",
    )
    _require(
        profiled_build.get("native_source_content_sha256")
        == binding.get("native_source_content_sha256"),
        "profiled record native source hash binding does not match",
    )
    _require(
        profiled.get("metadata", {}).get("native_extension_sha256")
        == binding.get("native_extension_sha256"),
        "profiled record extension hash binding does not match",
    )
    _require(
        profiled.get("metadata", {}).get("model", {}).get("sha256")
        == binding["model_sha256"],
        "profiled record model binding does not match",
    )
    _require(
        profiled.get("metadata", {}).get("exact_command")
        == binding.get("profiled_command"),
        "profiled record command binding does not match",
    )
    _require(
        profiled_systems[0].get("system_sha256") == binding["system_sha256"],
        "profiled record system binding does not match",
    )
    _require(
        profiled_systems[0].get("timed_properties") == binding["properties"],
        "profiled record properties binding does not match",
    )
    _require(
        {
            "warmups": profiled.get("metadata", {}).get("warmups"),
            "samples": profiled.get("metadata", {}).get("repeats"),
        }
        == binding.get("sample_scope"),
        "profiled record sample scope binding does not match",
    )
    profiled_runtime = profiled_systems[0].get("mh1_product_runtime", {})
    _require(
        {
            key: profiled_runtime.get("full_plugin", {}).get(key)
            for key in (
                "artifact_id",
                "artifact_sha256",
                "manifest_sha256",
                "module_identity",
            )
        }
        == binding.get("full_plugin"),
        "profiled record full-plugin binding does not match",
    )
    declared = timings.get("layers")
    _require(isinstance(declared, list), "product timing layers are missing")
    _require(
        [item.get("layer") for item in declared]
        == [item.get("layer") for item in layers],
        "product timing layer set does not match",
    )
    _require(
        [
            (
                item.get("layer"),
                item.get("artifact_id"),
                item.get("artifact_sha256"),
                item.get("artifact_path"),
                item.get("manifest_sha256"),
                item.get("origin"),
            )
            for item in binding.get("layers", [])
        ]
        == [
            (
                item.get("layer"),
                item.get("artifact_id"),
                item.get("artifact_sha256"),
                item.get("artifact_path"),
                item.get("manifest_sha256"),
                item.get("origin"),
            )
            for item in layers
        ],
        "product timing artifact binding does not match",
    )
    _require(
        [item.get("artifact_provenance") for item in binding.get("layers", [])]
        == [item.get("artifact_provenance") for item in layers],
        "product timing artifact binding provenance does not match",
    )
    _require(
        [
            (
                item.get("layer"),
                item.get("artifact_id"),
                item.get("artifact_sha256"),
                item.get("artifact_path"),
                item.get("manifest_sha256"),
                item.get("origin"),
                item.get("artifact_provenance"),
            )
            for item in profiled_runtime.get("layers", [])
        ]
        == [
            (
                item.get("layer"),
                item.get("artifact_id"),
                item.get("artifact_sha256"),
                item.get("artifact_path"),
                item.get("manifest_sha256"),
                item.get("origin"),
                item.get("artifact_provenance"),
            )
            for item in binding.get("layers", [])
        ],
        "profiled record artifact binding does not match",
    )
    with Path(binding["report_path"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    _require(
        timings["statistic"] == "median" and rows and "Med (ns)" in rows[0],
        "product timing profiler report is not median cuda_gpu_kern_sum CSV",
    )
    rows_by_name = {row.get("Name"): row for row in rows}
    _require(
        len(rows_by_name) == len(rows),
        "product timing profiler report has duplicate kernel rows",
    )
    for item, bound_layer in zip(declared, binding.get("layers", []), strict=True):
        kernels = bound_layer.get("profile_kernels", {})
        for direction in ("forward", "reverse"):
            row = rows_by_name.get(kernels.get(direction))
            _require(row is not None, "product timing profiler report lacks kernel row")
            try:
                measured = float(row["Med (ns)"]) / 1000.0
            except (TypeError, ValueError) as error:
                raise ValidationError(
                    "product timing profiler report median is invalid"
                ) from error
            _require(
                measured == float(item[f"{direction}_us"]),
                "product timing profiler report does not match declared timing",
            )


def validate_record(
    record,
    *,
    expected_commit,
    expected_source_sha256,
    expected_native_sha256,
    expected_extension_sha256,
    expected_properties,
    expected_samples,
    expected_atoms,
    expected_directed_edges,
    expected_artifact_ids,
    expected_full_plugin,
    expected_layers,
    expected_outputs,
    expected_product_timings=None,
):
    _require(
        isinstance(expected_full_plugin, dict),
        "explicit full-plugin expectation is required",
    )
    for field in (
        "artifact_id",
        "artifact_path",
        "artifact_sha256",
        "manifest_path",
        "manifest_sha256",
        "module_identity",
    ):
        _require(
            bool(expected_full_plugin.get(field)),
            f"full-plugin expectation lacks {field}",
        )
    _require(
        isinstance(expected_layers, list) and len(expected_layers) == 2,
        "two explicit layer expectations are required",
    )
    metadata = record.get("metadata")
    _require(isinstance(metadata, dict), "metadata must be an object")
    _require(metadata.get("git_revision") == expected_commit, "commit does not match")
    _require(metadata.get("git_dirty") is False, "record must come from a clean tree")
    build = metadata.get("native_build_info")
    _require(isinstance(build, dict), "native build info is missing")
    _require(
        build.get("source_commit") == expected_commit, "native commit does not match"
    )
    _require(build.get("source_dirty") is False, "native build must be clean")
    _require(
        build.get("source_content_sha256") == expected_source_sha256,
        "source hash does not match",
    )
    _require(
        build.get("native_source_content_sha256") == expected_native_sha256,
        "native source hash does not match",
    )
    _require(
        metadata.get("native_extension_sha256") == expected_extension_sha256,
        "extension hash does not match",
    )
    for value, label in (
        (expected_source_sha256, "expected source hash"),
        (expected_native_sha256, "expected native hash"),
        (expected_extension_sha256, "expected extension hash"),
    ):
        _sha256(value, label)
    properties = list(expected_properties)
    _require(metadata.get("timed_properties") == properties, "properties do not match")
    _require(metadata.get("repeats") == expected_samples, "sample count does not match")
    replay = metadata.get("replay")
    _require(isinstance(replay, dict), "replay provenance is missing")
    _require(bool(replay.get("python_executable")), "Python executable is missing")
    _require(
        bool(replay.get("launcher", {}).get("effective_cpu_affinity")),
        "launcher affinity is missing",
    )
    launcher = replay["launcher"]
    _require(
        launcher.get("kind") == "in-process-sched-setaffinity",
        "release records require explicit in-process CPU affinity",
    )
    _require(
        launcher.get("requested_cpu_affinity")
        == launcher.get("effective_cpu_affinity"),
        "requested and effective launcher affinity differ",
    )
    _require(
        replay.get("environment", {}).get("SYMMETRIX_JIT_POLICY") == "required",
        "strict required JIT was not recorded",
    )
    schedules = metadata.get("mh1_product_schedule_overrides")
    _require(isinstance(schedules, dict), "product schedules are missing")
    for operator in ("M0", "M1"):
        _require(
            schedules.get(operator, {}).get("effective") == "chunk32",
            f"{operator} schedule is not chunk32",
        )

    systems = record.get("systems")
    _require(isinstance(systems, list) and len(systems) == 1, "one system is required")
    system = systems[0]
    _require(system.get("atoms") == expected_atoms, "atom count does not match")
    _require(
        system.get("directed_edges") == expected_directed_edges,
        "directed-edge count does not match",
    )
    _require(system.get("timed_properties") == properties, "system properties drifted")
    _require(
        isinstance(expected_outputs, dict),
        "explicit output hash expectations are required",
    )
    output_hash_fields = {
        "energy": "energy_sha256",
        "forces": "forces_sha256",
        "stress": "stress_sha256",
    }
    for property_name in properties:
        field = output_hash_fields[property_name]
        expected_hash = _sha256(
            expected_outputs.get(field), f"expected {property_name} output hash"
        )
        actual_hash = _sha256(
            system.get(field), f"recorded {property_name} output hash"
        )
        _require(
            actual_hash == expected_hash,
            f"{property_name} output hash does not match expectation",
        )
    samples = system.get("evaluator", {}).get("samples_ms")
    _require(
        isinstance(samples, list) and len(samples) == expected_samples,
        "evaluator samples do not match",
    )
    _require(
        system.get("factorized_fallback_evaluation_count") == 0,
        "factorized fallback count is nonzero",
    )
    runtime = system.get("mh1_product_runtime")
    _require(isinstance(runtime, dict), "product runtime evidence is missing")
    _require(runtime.get("fallback_count") == 0, "product fallback count is nonzero")
    plugin = runtime.get("full_plugin")
    _require(isinstance(plugin, dict), "full-plugin evidence is missing")
    _require(plugin.get("abi", {}).get("version") == 5, "device module ABI is not 5")
    _require(plugin.get("jit_generation_version") == 19, "JIT generation is not 19")
    _require(
        plugin.get("launch_plan_schema") == "symmetrix.jit.mh1.cuda-launch-plan/6",
        "launch-plan schema is not /6",
    )
    _require(plugin.get("launch_plan_version") == 6, "launch-plan version is not 6")
    _require(
        plugin.get("split_phase_composition") is True,
        "split-phase composition is not enabled",
    )
    _require(
        bool(plugin.get("artifact_id")), "full-plugin artifact identity is missing"
    )
    _require(
        isinstance(plugin.get("artifact_path"), str) and plugin["artifact_path"],
        "full-plugin artifact path is missing",
    )
    _sha256(plugin.get("artifact_sha256"), "full-plugin artifact hash")
    _sha256(plugin.get("manifest_sha256"), "full-plugin manifest hash")
    actual_plugin = {
        "artifact_id": plugin.get("artifact_id"),
        "artifact_path": plugin.get("artifact_path"),
        "artifact_sha256": plugin.get("artifact_sha256"),
        "manifest_path": str(
            Path(plugin.get("artifact_path", "")).with_name("manifest.json")
        ),
        "manifest_sha256": plugin.get("manifest_sha256"),
        "module_identity": plugin.get("module_identity"),
    }
    _require_expected(actual_plugin, expected_full_plugin, "full-plugin expectation")

    layers = runtime.get("layers")
    _require(
        isinstance(layers, list) and len(layers) == 2, "exactly two layers are required"
    )
    _require(
        tuple(layer.get("artifact_id") for layer in layers)
        == tuple(expected_artifact_ids),
        "product artifact identity does not match",
    )
    for index, (layer, operator) in enumerate(zip(layers, ("M0", "M1"), strict=True)):
        label = f"layer {index}"
        expected_layer = expected_layers[index]
        _require(
            isinstance(expected_layer, dict)
            and isinstance(expected_layer.get("artifact_provenance"), dict),
            f"{label} explicit provenance expectation is required",
        )
        _require(
            bool(expected_layer.get("artifact_path"))
            and bool(expected_layer.get("origin"))
            and "manifest_sha256" in expected_layer,
            f"{label} explicit path/origin/manifest expectation is required",
        )
        _require(layer.get("layer") == index, f"{label} index does not match")
        _require(layer.get("operator") == operator, f"{label} operator does not match")
        _require(
            layer.get("origin") == expected_layer.get("origin"),
            f"{label} origin does not match expectation",
        )
        _require(
            layer.get("artifact_path") == expected_layer.get("artifact_path"),
            f"{label} artifact path does not match expectation",
        )
        _require(
            layer.get("manifest_sha256") == expected_layer.get("manifest_sha256"),
            f"{label} manifest hash does not match expectation",
        )
        _require(layer.get("schedule") == "chunk32", f"{label} schedule is not chunk32")
        _require(layer.get("abi", {}).get("version") == 3, f"{label} ABI is not 3")
        _require(
            layer.get("jit_generation_version") == 19, f"{label} generation is not 19"
        )
        _require(
            layer.get("forward_launch_count") == expected_samples,
            f"{label} forward counter drifted",
        )
        _require(
            layer.get("reverse_launch_count") == expected_samples,
            f"{label} reverse counter drifted",
        )
        scope = layer.get("launch_counter_scope")
        _require(isinstance(scope, dict), f"{label} launch counter scope is missing")
        _require(
            scope.get("scope") == "benchmark-warmups-and-timed-samples",
            f"{label} launch counter scope is invalid",
        )
        _require(
            scope.get("samples") == expected_samples,
            f"{label} counter sample count drifted",
        )
        _require(
            scope.get("timed_forward_launch_count") == expected_samples
            and scope.get("timed_reverse_launch_count") == expected_samples,
            f"{label} timed launch counters drifted",
        )
        artifact_sha256 = _sha256(
            layer.get("artifact_sha256"), f"{label} artifact hash"
        )
        _require(
            artifact_sha256 == expected_layer["artifact_sha256"],
            f"{label} artifact hash does not match expectation",
        )
        provenance = layer.get("artifact_provenance")
        _require(isinstance(provenance, dict), f"{label} provenance is missing")
        _require_expected(
            provenance,
            expected_layer.get("artifact_provenance"),
            f"{label} provenance expectation",
        )
        if provenance.get("kind") == "runtime-manifest":
            _sha256(provenance.get("manifest_sha256"), f"{label} manifest hash")
            _require(
                layer.get("origin") in {"runtime", "cache"},
                f"{label} origin and provenance are inconsistent",
            )
            _require(
                layer.get("manifest_sha256") == provenance.get("manifest_sha256"),
                f"{label} top-level manifest hash and provenance differ",
            )
        elif provenance.get("kind") == "installed-catalog":
            _sha256(provenance.get("catalog_sha256"), f"{label} catalog hash")
            _sha256(
                provenance.get("catalog_entry_sha256"),
                f"{label} catalog entry hash",
            )
            _require(
                layer.get("origin") == "prebuilt",
                f"{label} origin and provenance are inconsistent",
            )
            _require(
                layer.get("manifest_sha256") is None,
                f"{label} prebuilt catalog must not claim a sibling manifest",
            )
        else:
            raise ValidationError(f"{label} provenance kind is invalid")
        identity = layer.get("identity")
        _require(isinstance(identity, dict), f"{label} identity is missing")
        for field in (
            "generation_fingerprint",
            "semantic_fingerprint",
            "structure_fingerprint",
        ):
            _require(
                identity.get(field) == expected_layer["identity"][field],
                f"{label} {field} does not match expectation",
            )
        dimensions = layer.get("contract_dimensions")
        _require(isinstance(dimensions, dict), f"{label} dimensions are missing")
        for field in (
            "channels",
            "type_count",
            "input_components",
            "output_components",
            "correlation",
            "term_count",
            "coefficient_count",
        ):
            _require(
                isinstance(dimensions.get(field), int) and dimensions[field] > 0,
                f"{label} {field} is missing or invalid",
            )
            _require(
                dimensions[field] == expected_layer["contract_dimensions"][field],
                f"{label} {field} does not match expectation",
            )
        _require(
            dimensions.get("type_count") == 1,
            f"{label} type_count is not agnostic",
        )
        _require(
            dimensions.get("coefficient_count")
            == dimensions.get("term_count") * dimensions.get("channels"),
            f"{label} coefficient count is inconsistent",
        )
        memory = layer.get("memory_accounting")
        _require(isinstance(memory, dict), f"{label} memory accounting is missing")
        _require(
            memory.get("scratch_bytes") is None
            and memory.get("retained_bytes") is None,
            f"{label} memory accounting must not invent per-layer bytes",
        )
        _require(
            bool(memory.get("unavailable_reason")),
            f"{label} memory accounting reason is missing",
        )
        geometry = layer.get("realized_geometry")
        _require(isinstance(geometry, dict), f"{label} realized geometry is missing")
        for direction in ("forward", "reverse"):
            for field in (
                "logical_tiles",
                "tiles_per_block",
                "required_blocks",
                "realized_blocks",
            ):
                key = f"{direction}_{field}"
                _require(
                    isinstance(geometry.get(key), int) and geometry[key] >= 0,
                    f"{label} {key} is missing or invalid",
                )
    _validate_product_timings(
        runtime.get("product_timings"),
        metadata,
        system,
        plugin,
        layers,
        expected_product_timings,
    )
    return {
        "status": "ok",
        "commit": expected_commit,
        "atoms": expected_atoms,
        "directed_edges": expected_directed_edges,
        "samples": expected_samples,
        "artifact_ids": list(expected_artifact_ids),
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument("--extension-sha256", required=True)
    parser.add_argument("--properties", default="energy,forces,stress")
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--atoms", type=int, required=True)
    parser.add_argument("--directed-edges", type=int, required=True)
    parser.add_argument("--artifact-ids", required=True)
    parser.add_argument("--full-plugin-expectation-json", type=Path, required=True)
    parser.add_argument("--layer-expectations-json", type=Path, required=True)
    parser.add_argument("--output-expectations-json", type=Path, required=True)
    parser.add_argument("--product-timing-expectations-json", type=Path)
    return parser


def main():
    args = _parser().parse_args()
    summary = validate_record(
        json.loads(args.record.read_text()),
        expected_commit=args.commit,
        expected_source_sha256=args.source_sha256,
        expected_native_sha256=args.native_sha256,
        expected_extension_sha256=args.extension_sha256,
        expected_properties=tuple(args.properties.split(",")),
        expected_samples=args.samples,
        expected_atoms=args.atoms,
        expected_directed_edges=args.directed_edges,
        expected_artifact_ids=tuple(args.artifact_ids.split(",")),
        expected_full_plugin=json.loads(args.full_plugin_expectation_json.read_text()),
        expected_layers=json.loads(args.layer_expectations_json.read_text()),
        expected_outputs=json.loads(args.output_expectations_json.read_text()),
        expected_product_timings=(
            json.loads(args.product_timing_expectations_json.read_text())
            if args.product_timing_expectations_json is not None
            else None
        ),
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
