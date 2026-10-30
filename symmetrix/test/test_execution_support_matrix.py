"""Focused coverage for the generated execution support matrix.

The matrix is the documentation of the public execution surface, so this module
pins that it is *derived* rather than maintained: the committed document equals
a fresh generation from the declared family registry, the generator refuses a
declaration it cannot describe, and the qualification boundary is stated by
the generated rows themselves.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "generate_execution_support_matrix.py"
DOCUMENT = ROOT / "docs" / "reference" / "execution_support_matrix.md"


@pytest.fixture(scope="module")
def matrix():
    spec = importlib.util.spec_from_file_location(
        "generate_execution_support_matrix", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # The generator declares dataclasses, so it must be importable by name
    # before it executes for the postponed annotations to resolve.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _entries():
    from symmetrix import family_registry

    return family_registry.declared_entries()


def _with_entries(matrix, replacements):
    """Return the declared entries with ``replacements`` applied by family key."""

    return [
        replacements.get(entry.key, entry)
        for entry in matrix._registry().declared_entries()
    ]


def test_document_equals_a_fresh_generation(matrix):
    current = DOCUMENT.read_text()
    assert current == matrix.apply(current, matrix.render_matrix())


def test_generated_block_appears_once_and_regenerates_stably(matrix):
    document = DOCUMENT.read_text()

    assert document.count(matrix.BEGIN_MARKER) == 1
    assert document.count(matrix.END_MARKER) == 1
    applied = matrix.apply(document, matrix.render_matrix())
    assert matrix.apply(applied, matrix.render_matrix()) == applied


def test_generated_matrix_describes_complete_maceles_models(matrix):
    generated = matrix.render_matrix()

    assert "`model_type` `MACE`/`MACELES`/`MACEField`" in generated
    assert "Kokkos (float32, float64; `model_type` `MACE`/`MACEField`)" in generated
    assert "Kokkos (float32, float64; `model_type` `MACELES`)" in generated
    assert "MACELESKokkos" not in generated
    assert "MACELES evaluates its LES term in both routings" in generated
    assert "MACELES does not support fixed-workspace plans" in generated
    assert "MACELES is not supported in LAMMPS" in generated


def test_check_mode_reports_a_drifted_document(matrix, tmp_path, monkeypatch):
    drifted = tmp_path / "execution_support_matrix.md"
    drifted.write_text(
        DOCUMENT.read_text().replace(
            "| `OAM` (OAM v2 contract) | `stress` | requalification pending |",
            "| `OAM` (OAM v2 contract) | `stress` | experimental |",
        )
    )
    monkeypatch.setattr(matrix, "DOCUMENT", drifted)

    assert matrix.main(["--check"]) == 1
    assert matrix.main([]) == 0
    assert matrix.main(["--check"]) == 0
    assert drifted.read_text() == DOCUMENT.read_text()


def test_the_matrix_documents_every_declared_mode_and_capability_key(matrix):
    from symmetrix import family_registry

    entries = _entries()
    matrix.validate(entries)

    for entry in entries:
        documented = {row.mode for row in matrix.FAMILY_REQUESTS[entry.key]}
        assert documented == set(entry.modes.canonical_modes), entry.key
    keys = {
        row.key for entry in entries for row in matrix.FAMILY_CAPABILITIES[entry.key]
    }
    assert keys == set(family_registry.CAPABILITY_KEYS)


def test_each_family_documents_every_capability_once(matrix):
    for family, rows in matrix.FAMILY_CAPABILITIES.items():
        keys = [row.key for row in rows]

        assert len(keys) == len(set(keys)), family


def test_a_declared_mode_without_a_request_row_is_rejected(matrix):
    from symmetrix import family_registry

    entry = family_registry.declared_registry().by_key("MACE")
    extended = dataclasses.replace(
        entry,
        modes=family_registry.ModeVocabulary(
            aliases={**entry.modes.aliases, "streamed_v2": "streamed_v2"},
            primary=(*entry.modes.primary, "streamed_v2"),
            deprecated=entry.modes.deprecated,
        ),
    )

    with pytest.raises(ValueError, match="the request table documents"):
        matrix.validate(_with_entries(matrix, {"MACE": extended}))


def test_an_undeclared_capability_key_is_rejected(matrix):
    rows = matrix.FAMILY_CAPABILITIES["MACE"]
    tampered = dict(matrix.FAMILY_CAPABILITIES)
    tampered["MACE"] = (
        dataclasses.replace(rows[0], key="telemetry"),
        *rows[1:],
    )
    original = matrix.FAMILY_CAPABILITIES
    matrix.FAMILY_CAPABILITIES = tampered
    try:
        with pytest.raises(ValueError, match="undeclared capability 'telemetry'"):
            matrix.validate()
    finally:
        matrix.FAMILY_CAPABILITIES = original


def test_an_omitted_capability_key_is_rejected(matrix):
    tampered = {
        family: tuple(row for row in rows if row.key != "device_artifact")
        for family, rows in matrix.FAMILY_CAPABILITIES.items()
    }
    original = matrix.FAMILY_CAPABILITIES
    matrix.FAMILY_CAPABILITIES = tampered
    try:
        with pytest.raises(
            ValueError, match=r"omits the declared keys \['device_artifact'\]"
        ):
            matrix.validate()
    finally:
        matrix.FAMILY_CAPABILITIES = original


def test_a_rejected_request_without_a_reason_is_rejected(matrix):
    from symmetrix import family_registry

    entry = family_registry.declared_registry().by_key("MACE")
    extended = dataclasses.replace(
        entry,
        modes=family_registry.ModeVocabulary(
            aliases={**entry.modes.aliases, "streamed_v2": "streamed_v2"},
            primary=(*entry.modes.primary, "streamed_v2"),
            deprecated=entry.modes.deprecated,
        ),
    )
    rejected = matrix.RequestRow(
        mode="streamed_v2",
        label="`streamed_v2`",
        admitted=False,
        artifact="None",
        lammps="Not supported",
    )
    original = matrix.FAMILY_REQUESTS
    matrix.FAMILY_REQUESTS = {
        **original,
        "MACE": (*original["MACE"], rejected),
    }
    try:
        with pytest.raises(ValueError, match="rejects 'streamed_v2' without a reason"):
            matrix.validate(_with_entries(matrix, {"MACE": extended}))
    finally:
        matrix.FAMILY_REQUESTS = original


def test_the_oam_rows_require_current_runtime_qualification(matrix):
    from symmetrix.oam_receiver_codegen import ABI_VERSION
    from symmetrix.oam_cuda_codegen import ABI_VERSION as CUDA_ABI_VERSION

    rows = {row.key: row for row in matrix.FAMILY_CAPABILITIES["OAM"]}
    stress = rows["stress"]

    assert stress.status == "requalification pending"
    assert "implemented on the current host and CUDA routes" in stress.scope
    assert f"ABI-v{ABI_VERSION}/ABI-v{CUDA_ABI_VERSION}" in stress.limits
    assert "runtime qualification is pending" in stress.limits
    assert "HIP is unsupported" in stress.limits

    precision = rows["precision"]
    assert precision.status == "build-dependent"
    assert "float32 and float64 implementations" in precision.scope
    assert "full-message-only artifact release-matrix qualification is pending" in (
        precision.limits
    )

    reference = rows["reference"]
    assert reference.status == "build-dependent"
    assert "current release-matrix qualification is pending" in reference.limits
    assert "same-spline parity does not establish upstream-NequIP parity" in (
        reference.limits
    )
    portable = rows["portable"]
    assert portable.status == "build-dependent"
    assert "current runtime matrix is pending" in portable.limits

    host_artifact = rows["host_artifact"]
    assert f"ABI-v{ABI_VERSION}" in host_artifact.scope
    assert "fresh host runtime qualification" in host_artifact.limits
    assert "repaired-wheel inclusion gates are pending" in host_artifact.limits
    device_artifact = rows["device_artifact"]
    assert f"CUDA ABI-v{CUDA_ABI_VERSION}" in device_artifact.scope
    assert "fresh CUDA runtime qualification" in device_artifact.limits
    assert "repaired-wheel inclusion gates are pending" in device_artifact.limits

    generated = rows["generated"]
    assert f"host ABI v{ABI_VERSION}" in generated.scope
    assert "full-message-only" in generated.scope
    assert "compile-qualified" in generated.limits


def test_the_oam_request_route_names_the_current_host_abi(matrix):
    from symmetrix.oam_receiver_codegen import ABI_VERSION
    from symmetrix.oam_cuda_codegen import ABI_VERSION as CUDA_ABI_VERSION

    direct = next(row for row in matrix.FAMILY_REQUESTS["OAM"] if row.mode == "direct")

    assert f"host ABI v{ABI_VERSION}" in direct.artifact
    assert f"CUDA ABI v{CUDA_ABI_VERSION}" in direct.artifact


def test_the_document_states_the_current_oam_requalification_boundary(matrix):
    from symmetrix.oam_receiver_codegen import ABI_VERSION

    document = DOCUMENT.read_text()

    assert (
        "| `OAM` (OAM v2 contract) | `stress` | requalification pending |" in document
    )
    assert "full-message-only artifact release-matrix qualification is pending" in (
        document
    )
    assert "Host direct execution uses its ABI-v2" not in document
    assert "Direct stress is experimental" not in document
    assert f"host ABI v{ABI_VERSION}" in document
    assert (
        "fresh CUDA runtime qualification and repaired-wheel inclusion gates are pending"
        in document
    )


def test_the_document_keeps_the_recorded_public_requests(matrix):
    document = DOCUMENT.read_text()

    for spelling in (
        "`direct`",
        "`non-compiled`",
        "`materialized`",
        "`auto`",
        "`debug_v1`",
        "`mlp_reference`",
    ):
        assert spelling in document
    # The hand-maintained table's compatibility row survives as declared data.
    assert "Format-aware compatibility default" in document
    assert "original Symmetrix format (named v1 here)" in document


def test_polar_lammps_rows_state_the_qualified_one_rank_boundary(matrix):
    """Every Polar routing documents its narrow native LAMMPS contract."""

    rows = {row.mode: row for row in matrix.FAMILY_REQUESTS["MACE_Polar"]}

    for row in rows.values():
        assert "`symmetrix/mace/polar/kk`" in row.lammps
        assert "one MPI rank" in row.lammps
        assert "complete system" in row.lammps
        assert "domain decomposition" in row.lammps
    assert "portable `pair_spline_v1`" in rows["auto"].lammps
    assert "matching generated/direct artifact" in rows["direct"].lammps
    assert "artifact-free portable `pair_spline_v1`" in rows["reference"].lammps

    capabilities = {row.key: row for row in matrix.FAMILY_CAPABILITIES["MACE_Polar"]}
    assert "HIP remains unqualified" in capabilities["target"].limits
    assert capabilities["stress"].status == "qualified"
    assert "exact-zero isolated stress" in capabilities["stress"].scope


def test_polar_user_docs_describe_native_json_evaluation_and_lammps_boundary():
    """The public guides must not preserve the pre-native Polar limitations."""

    models = (ROOT / "docs/user/models.md").read_text()
    lammps = (ROOT / "docs/user/lammps.md").read_text()
    pair_readme = (ROOT / "pair_symmetrix/README.md").read_text()
    lammps_words = " ".join(lammps.split())
    pair_readme_words = " ".join(pair_readme.split())

    assert "JSON evaluation does not require Python" in models
    assert "symmetrix/mace/polar/kk" in lammps
    assert "exactly one MPI rank" in lammps
    assert "complete system" in lammps
    assert "at most one prepared" in lammps_words
    assert "portable `pair_spline_v1`" in lammps_words
    assert "generated/direct backbone" in lammps_words
    assert "distributed" in lammps
    assert "symmetrix/mace/polar/kk" in pair_readme
    assert "exactly one MPI rank" in pair_readme
    assert "at most one prepared" in pair_readme_words
    assert "portable `pair_spline_v1`" in pair_readme_words
    assert "generated/direct backbone" in pair_readme_words
    assert "LAMMPS dispatch remains a follow-on" not in models
