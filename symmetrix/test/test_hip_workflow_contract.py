from pathlib import Path
import json


WORKFLOW = Path(__file__).parents[2] / ".github/workflows/hip-qualification.yaml"


def test_hip_workflow_records_reproducible_build_provenance():
    text = WORKFLOW.read_text()
    required = (
        "submodule status --recursive",
        "git ls-files -s",
        "symmetrix_source_fingerprint",
        "symmetrix_native_source_fingerprint",
        "SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256",
        '--config-settings=cmake.define.SYMMETRIX_BUILD_NATIVE_SOURCE_CONTENT_SHA256="$SYMMETRIX_BUILD_NATIVE_SOURCE_CONTENT_SHA256"',
        "rocminfo",
        "hipcc",
        "hipfft",
        "rocfft",
        "rocblas",
        "symmetrix version --probe --json",
        "symmetrix doctor --json",
        "extension_sha256",
        "from symmetrix import symmetrix as native",
        "native._backend_build_info()",
        "native._init_kokkos()",
        "native._kokkos_default_execution_space()",
        "_execution_device_execution_environment",
        "_kokkos_device_sentinel",
        "native._finalize_kokkos()",
        'assert build_info.get("backend") == "hip"',
        'assert execution_space == "HIP"',
        "assert device_sentinel is True",
        "build-command.txt",
        "CMakeCache.txt",
        "actions/upload-artifact@v7",
        'entry_points(group="symmetrix.backends")',
        'descriptor.get("backend") == "hip"',
        'descriptor.get("architecture") == "gfx1151"',
        'descriptor.get("selector")',
        "assert entry.name == selector",
        "SYMMETRIX_BACKEND={selector}",
        "selected-backend.json",
        "Prepare isolated task runtime roots",
        "XDG_CACHE_HOME",
        "XDG_RUNTIME_DIR",
        "ROCM_COMGR_CACHE_DIR",
        "HIP_CACHE_DIR",
        "chmod 700",
        "df -i",
    )
    for marker in required:
        assert marker in text
    assert text.index("Record source and submodule provenance") < text.index(
        "Build clean HIP extension"
    )
    assert "if: always()" in text


def test_hip_workflow_uses_pinned_python_and_architecture_qualified_build():
    text = WORKFLOW.read_text()
    assert 'python-version: "3.12"' in text
    assert "--offload-arch=gfx1151" in text
    assert "actions/checkout@v6" in text
    assert "SYMMETRIX_BACKEND=hip" not in text


def test_hip_manifest_excludes_uncollected_mh1_nodes():
    manifest = json.loads(
        (WORKFLOW.parents[2] / "symmetrix/test/hip_required_tests.json").read_text()
    )
    nodes = manifest["required_nodes"]
    assert (
        "test/test_mh1.py::test_generated_retention_and_reverse_tap_match_portable[c4-e2-b4-l2-linear]"
        in nodes
    )
    assert not any(
        "[c8-e4-b8-l2]" in node or "[c12-e6-b16-l3]" in node for node in nodes
    )


def test_fixture_only_workflow_forbids_oracle_dependencies_and_downloads():
    workflow = (
        WORKFLOW.parents[2] / ".github/workflows/polar-fixture-only.yaml"
    ).read_text()
    assert "Build CPU frontend and run immutable fixture lane" in workflow
    assert "find_spec(name) is None" in workflow
    assert '"torch", "mace", "graph_longrange"' in workflow
    assert "test_polar_oracle_fixtures.py" in workflow
    assert "test_polar_end_to_end_fixtures.py" in workflow
    assert "test_mace_les_reference.py" in workflow
    assert "curl" not in workflow
    assert "download" not in workflow.lower()
