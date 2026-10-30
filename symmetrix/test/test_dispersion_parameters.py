import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

import numpy as np
import pytest

import symmetrix


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPOSITORY_ROOT / "tools" / "generate_dispersion_parameters.py"
REFERENCE_GENERATOR = REPOSITORY_ROOT / "tools" / "generate_dispersion_reference.py"
GENERATED_FILENAMES = (
    "dispersion_parameters.hpp",
    "dispersion_parameters.cpp",
    "dispersion_parameter_provenance.hpp",
)
REFERENCE_CASES = (
    Path(__file__).parent / "data" / "dispersion" / "torch_dftd_0_5_3_cases.json"
)
UPSTREAM_COMMIT = "28b43557e6d950502941990fa5a1c9a3aa9e8339"
UPSTREAM_SOURCE_HASHES = {
    "LICENSE": "7f7be264c10276c7772bdead5b3bcdb68082edf862ca0f67d97efdb36bc2d889",
    "torch_dftd/_version.py": (
        "b60ceea8729c11d28168fe7b179a17c6ae17956da7f49e058fc646bbb9c63998"
    ),
    "torch_dftd/dftd3_xc_params.py": (
        "30db7a75ffba3d8c4f65c959102c06ad60063a38d61cfcb00c93f4d73a99d71a"
    ),
    "torch_dftd/nn/params/dftd2_params.py": (
        "850d7d9fea302bc91583d09f2395afa0746c713a83425751d70898945eff0665"
    ),
    "torch_dftd/nn/params/dftd3_params.npz": (
        "52b7b35a957df55b813035ffde3a6b5a41dec585bdb1e281fb8ba67a729f3d92"
    ),
    "torch_dftd/testing/damping.py": (
        "ca221387b48666255fdcfa34269353754f901d678f19cd6f7719c38fb4f2216b"
    ),
}
UPSTREAM_REFERENCE_HASHES = {
    **UPSTREAM_SOURCE_HASHES,
    "torch_dftd/__init__.py": (
        "b0382200a901e94928a18656f8a8d98c775edeb85971b7e8f25d605a6d8b31d3"
    ),
    "torch_dftd/functions/__init__.py": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "torch_dftd/functions/dftd2.py": (
        "ce58eaab29f5ee83e707ecb77db984f2613331e9bbf4f23c4f7310a827e4c86e"
    ),
    "torch_dftd/functions/dftd3.py": (
        "0ef70d680aeb450a90af3aaeba0972ff89938c052f2309c0ec6d5347cf25f1f2"
    ),
    "torch_dftd/functions/distance.py": (
        "9835ebbd9820790411d62f7bbf6445c1cc8d5206204f83379ee6583e57c6ad6d"
    ),
    "torch_dftd/functions/smoothing.py": (
        "3e33be3edb2ab68fe6b113ddf6e5fe9cdff0ca5bbffab00e096306cf07f925a6"
    ),
    "torch_dftd/functions/triplets.py": (
        "e21781e60c2cdd5eda2677ffe49e1fdc983662a1addf5ea6795aefdd7a21525d"
    ),
    "torch_dftd/functions/triplets_kernel.py": (
        "abbb8fd293c7f6e5691791aa8437d37463c25053e35878482a5cf30f6da68a94"
    ),
    "torch_dftd/nn/__init__.py": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "torch_dftd/nn/base_dftd_module.py": (
        "9702458337e426795acfb6aae8a82a45b742a8ef96f2eee8547e12cd599005d4"
    ),
    "torch_dftd/nn/dftd2_module.py": (
        "321479cbc4ce2ae814aafb1ad831921d8c0a126ed4f7b36230c89b626e45995f"
    ),
    "torch_dftd/nn/dftd3_module.py": (
        "24970d4a267badab653b795b321b3f93c45994cb6c09b53a880cc214104fb72d"
    ),
    "torch_dftd/nn/params/__init__.py": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
}
EXPECTED_GENERATOR = {
    "name": "tools/generate_dispersion_parameters.py",
    "schema": "symmetrix.dispersion-parameter-generator/1",
    "version": "1",
}
EXPECTED_CONVERSION_CONSTANTS = {
    "bohr_to_angstrom": {"value": 0.52917726, "units": "angstrom / bohr"},
    "hartree_to_ev": {"value": 27.21138505, "units": "eV / hartree"},
    "joule_to_kilojoule": {"value": 1.0e-3, "units": "kJ / J"},
    "hartree_molar_energy": {
        "value": 2625.4999,
        "units": "kJ mol^-1 / hartree",
    },
    "bohr_to_nanometer": {"value": 0.052917726, "units": "nm / bohr"},
    "d2_c6_to_atomic_units": {
        "value": 17.345265600595098,
        "units": "hartree bohr^6 / (J nm^6 mol^-1)",
    },
    "hartree_bohr6_to_ev_angstrom6": {
        "value": 0.59752721065076475,
        "units": "eV angstrom^6 / (hartree bohr^6)",
    },
}


def _run_generator(source: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--torch-dftd-source",
            str(source),
            "--output",
            str(output),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _run_reference_generator(
    source: Path, output: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REFERENCE_GENERATOR),
            "--torch-dftd-source",
            str(source),
            "--output",
            str(output),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _copy_pinned_source(tmp_path: Path) -> Path:
    configured_source = os.environ.get("SYMMETRIX_TORCH_DFTD_SOURCE")
    original = Path(configured_source or "/tmp/torch-dftd-audit")
    if not (original / "torch_dftd" / "_version.py").is_file():
        pytest.skip("pinned torch-dftd 0.5.3 source is unavailable")
    source = tmp_path / "source"
    shutil.copytree(
        original,
        source,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    return source


def _rewrite_npz_array(
    archive_path: Path, name: str, mutation: Callable[[np.ndarray], None]
) -> None:
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
        unchanged = {key: value.copy() for key, value in arrays.items() if key != name}
    mutation(arrays[name])
    np.savez(archive_path, **arrays)
    with np.load(archive_path, allow_pickle=False) as archive:
        for key, expected in unchanged.items():
            np.testing.assert_array_equal(archive[key], expected)


def _run_generator_with_modified_archive(
    source: Path, output: Path, tmp_path: Path
) -> subprocess.CompletedProcess[str]:
    tools = tmp_path / "generator-tools"
    tools.mkdir()
    generator = tools / GENERATOR.name
    source_identity = tools / "_torch_dftd_source.py"
    shutil.copy2(GENERATOR, generator)
    shutil.copy2(REPOSITORY_ROOT / "tools" / source_identity.name, source_identity)

    archive = source / "torch_dftd" / "nn" / "params" / "dftd3_params.npz"
    original_hash = UPSTREAM_SOURCE_HASHES["torch_dftd/nn/params/dftd3_params.npz"]
    modified_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    identity_text = source_identity.read_text()
    assert identity_text.count(original_hash) == 1
    source_identity.write_text(identity_text.replace(original_hash, modified_hash))

    return subprocess.run(
        [
            sys.executable,
            str(generator),
            "--torch-dftd-source",
            str(source),
            "--output",
            str(output),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_parameter_generator_rejects_nonrectangular_d3_reference_mask(tmp_path):
    source = _copy_pinned_source(tmp_path)
    archive = source / "torch_dftd" / "nn" / "params" / "dftd3_params.npz"

    def remove_mask_corner(c6ab):
        c6ab[1, 1, 1, 1, 0] = -1.0

    _rewrite_npz_array(archive, "c6ab", remove_mask_corner)
    completed = _run_generator_with_modified_archive(
        source, tmp_path / "output", tmp_path
    )

    assert completed.returncode != 0
    assert "rectangular" in completed.stderr
    assert not (tmp_path / "output").exists()


def test_parameter_generator_rejects_partner_dependent_d3_coordination(tmp_path):
    source = _copy_pinned_source(tmp_path)
    archive = source / "torch_dftd" / "nn" / "params" / "dftd3_params.npz"

    def change_hydrogen_coordination_for_carbon(c6ab):
        c6ab[1, 6, 0, 0, 1] += 0.125

    _rewrite_npz_array(archive, "c6ab", change_hydrogen_coordination_for_carbon)
    completed = _run_generator_with_modified_archive(
        source, tmp_path / "output", tmp_path
    )

    assert completed.returncode != 0
    assert "partner-dependent" in completed.stderr
    assert not (tmp_path / "output").exists()


def test_generated_d3_reference_metadata_is_declared():
    header = (
        REPOSITORY_ROOT / "libsymmetrix/source/dispersion_parameters.hpp"
    ).read_text()
    source = (
        REPOSITORY_ROOT / "libsymmetrix/source/dispersion_parameters.cpp"
    ).read_text()

    assert "d3_reference_counts()" in header
    assert "d3_reference_coordination()" in header
    assert "d3_reference_count_data" in source
    assert "d3_reference_coordination_data" in source


def test_parameter_generator_rejects_unpinned_version(tmp_path):
    source = tmp_path / "source"
    package = source / "torch_dftd"
    package.mkdir(parents=True)
    (package / "_version.py").write_text('__version__ = "0.5.2"\n')

    completed = _run_generator(source, tmp_path / "output")

    assert completed.returncode != 0
    assert "requires torch-dftd 0.5.3" in completed.stderr
    assert not (tmp_path / "output").exists()


def test_generated_parameters_reproduce_from_pinned_source(tmp_path):
    configured_source = os.environ.get("SYMMETRIX_TORCH_DFTD_SOURCE")
    source = Path(configured_source or "/tmp/torch-dftd-audit")
    if not (source / "torch_dftd" / "_version.py").is_file():
        pytest.skip("pinned torch-dftd 0.5.3 source is unavailable")

    output = tmp_path / "generated"
    completed = _run_generator(source, output)

    assert completed.returncode == 0, completed.stderr
    for filename in GENERATED_FILENAMES:
        assert (output / filename).read_bytes() == (
            REPOSITORY_ROOT / "libsymmetrix" / "source" / filename
        ).read_bytes()


def test_reference_cases_reproduce_from_pinned_source(tmp_path):
    configured_source = os.environ.get("SYMMETRIX_TORCH_DFTD_SOURCE")
    source = Path(configured_source or "/tmp/torch-dftd-audit")
    if not (source / "torch_dftd" / "_version.py").is_file():
        pytest.skip("pinned torch-dftd 0.5.3 source is unavailable")
    pytest.importorskip("torch")

    output = tmp_path / "reference.json"
    completed = _run_reference_generator(source, output)

    assert completed.returncode == 0, completed.stderr
    assert output.read_bytes() == REFERENCE_CASES.read_bytes()


def test_generators_do_not_mutate_pinned_source(tmp_path):
    pytest.importorskip("torch")
    source = _copy_pinned_source(tmp_path)
    files_before = {path.relative_to(source) for path in source.rglob("*")}

    parameter_result = _run_generator(source, tmp_path / "parameters")
    reference_result = _run_reference_generator(source, tmp_path / "reference.json")

    assert parameter_result.returncode == 0, parameter_result.stderr
    assert reference_result.returncode == 0, reference_result.stderr
    assert {path.relative_to(source) for path in source.rglob("*")} == files_before


def test_parameter_generator_rejects_modified_pinned_source(tmp_path):
    source = _copy_pinned_source(tmp_path)
    parameter_file = source / "torch_dftd" / "dftd3_xc_params.py"
    parameter_file.write_text(parameter_file.read_text() + "\n# modified\n")

    completed = _run_generator(source, tmp_path / "output")

    assert completed.returncode != 0
    assert "SHA-256 mismatch" in completed.stderr
    assert "torch_dftd/dftd3_xc_params.py" in completed.stderr
    assert not (tmp_path / "output").exists()


def test_reference_generator_rejects_modified_algorithm_dependency(tmp_path):
    pytest.importorskip("torch")
    source = _copy_pinned_source(tmp_path)
    algorithm_file = source / "torch_dftd" / "functions" / "distance.py"
    algorithm_file.write_text(algorithm_file.read_text() + "\n# modified\n")

    completed = _run_reference_generator(source, tmp_path / "reference.json")

    assert completed.returncode != 0
    assert "SHA-256 mismatch" in completed.stderr
    assert "torch_dftd/functions/distance.py" in completed.stderr
    assert not (tmp_path / "reference.json").exists()


def test_generators_reject_wrong_git_commit_even_with_pinned_files(tmp_path):
    pytest.importorskip("torch")
    source = _copy_pinned_source(tmp_path)
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "add",
            ".",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )

    parameter_result = _run_generator(source, tmp_path / "parameters")
    reference_result = _run_reference_generator(source, tmp_path / "reference.json")

    for completed in (parameter_result, reference_result):
        assert completed.returncode != 0
        assert "Git commit mismatch" in completed.stderr


@pytest.mark.parametrize(
    ("method", "damping", "functional", "expected"),
    [
        (
            "d2",
            "zero",
            "pbe",
            {"s6": 0.75, "rs6": 1.1, "s18": 0.0, "alp": 20.0},
        ),
        (
            "d3",
            "zero",
            "pbe",
            {"s6": 1.0, "rs6": 1.217, "s18": 0.722, "rs18": 1.0, "alp": 14.0},
        ),
        (
            "d3",
            "bj",
            "pbe",
            {"s6": 1.0, "rs6": 0.4289, "s18": 0.7875, "rs18": 4.4407, "alp": 14.0},
        ),
        (
            "d3",
            "zerom",
            "pbe",
            {"s6": 1.0, "rs6": 2.340218, "s18": 0.0, "rs18": 0.129434, "alp": 14.0},
        ),
        (
            "d3",
            "bjm",
            "pbe",
            {"s6": 1.0, "rs6": 0.012092, "s18": 0.35894, "rs18": 5.938951, "alp": 14.0},
        ),
    ],
)
def test_native_parameter_lookup_reports_expected_scalars(
    method, damping, functional, expected
):
    summary = symmetrix._dispersion_parameter_summary(method, damping, functional)

    assert summary["parameters"] == pytest.approx(expected)
    assert summary["schema"] == "symmetrix.dispersion-parameters/1"
    assert summary["upstream_version"] == "0.5.3"
    assert summary["upstream_tag"] == "v0.5.3"
    assert summary["upstream_commit"] == UPSTREAM_COMMIT
    assert summary["license"] == "MIT"
    assert summary["attribution"] == (
        "torch-dftd Copyright (c) 2021 Preferred Networks, Inc.; MIT License"
    )
    assert summary["source_hashes"] == UPSTREAM_SOURCE_HASHES
    assert summary["generator"] == EXPECTED_GENERATOR
    assert summary["conversion_constants"] == EXPECTED_CONVERSION_CONSTANTS
    assert summary["kernel_constants"] == {"bohr_to_angstrom": 0.52917726}


def test_native_parameter_lookup_reports_table_coverage():
    d2 = symmetrix._dispersion_parameter_summary("d2", "zero", "pbe")
    d3 = symmetrix._dispersion_parameter_summary("d3", "bj", "pbe")

    assert d2["supported_atomic_number"] == 86
    assert d2["table_shapes"] == {"c6": [87, 87], "r0": [87, 87]}
    assert d3["supported_atomic_number"] == 94
    assert d3["table_shapes"] == {
        "c6": [95, 95, 5, 5, 3],
        "r0": [95, 95],
        "rcov": [95],
        "r2r4": [95],
    }


@pytest.mark.parametrize(
    ("method", "damping", "functional"),
    [
        ("d2", "bj", "pbe"),
        ("d3", "unknown", "pbe"),
        ("d3", "bj", "not-a-functional"),
        ("unknown", "zero", "pbe"),
    ],
)
def test_native_parameter_lookup_rejects_unsupported_combinations(
    method, damping, functional
):
    with pytest.raises(ValueError, match="unsupported dispersion parameter"):
        symmetrix._dispersion_parameter_summary(method, damping, functional)


def test_reference_cases_are_versioned_plain_json_with_required_coverage():
    payload = json.loads(REFERENCE_CASES.read_text())

    assert payload["schema"] == "symmetrix.dispersion-reference/1"
    assert payload["source"]["torch_dftd"] == "0.5.3"
    assert payload["source"]["commit"] == UPSTREAM_COMMIT
    assert payload["source"]["python"].startswith("3.12.")
    assert payload["source"]["torch"]
    assert payload["source"]["ase"]
    assert payload["source"]["source_hashes"] == UPSTREAM_REFERENCE_HASHES

    cases = payload["cases"]
    methods = {(case["method"], case["damping"]) for case in cases}
    assert methods >= {
        ("d2", "zero"),
        ("d3", "zero"),
        ("d3", "bj"),
        ("d3", "zerom"),
        ("d3", "bjm"),
    }
    assert {case["functional"] for case in cases} >= {"pbe", "pbe0"}
    tags = {tag for case in cases for tag in case["coverage"]}
    assert tags >= {
        "molecular",
        "periodic",
        "cutoff-boundary",
        "empty-edge",
        "mixed-element",
        "high-coordination",
    }
    assert any(
        case["name"] == "d3-zerom-pbe-interior"
        and case["damping"] == "zerom"
        and "cutoff-boundary" not in case["coverage"]
        and case["energy_ev"] < 0.0
        for case in cases
    )
    for case in cases:
        atom_count = len(case["atomic_numbers"])
        assert atom_count == len(case["positions_angstrom"])
        assert atom_count == len(case["forces_ev_per_angstrom"])
        assert all(len(position) == 3 for position in case["positions_angstrom"])
        assert all(len(force) == 3 for force in case["forces_ev_per_angstrom"])
        numeric_values = [
            case["energy_ev"],
            *(value for position in case["positions_angstrom"] for value in position),
            *(value for force in case["forces_ev_per_angstrom"] for value in force),
        ]
        if "stress_ev_per_angstrom3" in case:
            assert len(case["stress_ev_per_angstrom3"]) == 6
            numeric_values.extend(case["stress_ev_per_angstrom3"])
        assert all(isinstance(value, (int, float)) for value in numeric_values)
        assert all(
            value == value and abs(value) != float("inf") for value in numeric_values
        )


def test_torch_dftd_notice_is_complete_and_packaged():
    root_notice = REPOSITORY_ROOT / "THIRD_PARTY_LICENSES.md"
    package_notice = REPOSITORY_ROOT / "symmetrix" / "THIRD_PARTY_LICENSES.md"
    notice = root_notice.read_text()

    assert package_notice.read_text() == notice
    assert "torch-dftd 0.5.3" in notice
    assert "Copyright (c) 2021 Preferred Networks, Inc." in notice
    assert "Permission is hereby granted, free of charge" in notice
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in notice
    project = tomllib.loads(
        (REPOSITORY_ROOT / "symmetrix" / "pyproject.toml").read_text()
    )["project"]
    assert "THIRD_PARTY_LICENSES.md" in project["license-files"]
