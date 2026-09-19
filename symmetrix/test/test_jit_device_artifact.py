import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from symmetrix import jit_device_artifact as device_artifact
from symmetrix.cli import extract_mace as extract_cli
from symmetrix.cli import prepare_jit_device_artifact as artifact_cli


class _Evaluator:
    def __init__(self, backend):
        self.events = []
        self.execution_device_execution_environment = {
            "available": True,
            "backend": backend,
            "execution_space": "Cuda" if backend == "cuda" else "HIP",
            "compute_capability_code": 120,
            "architecture": "sm_120" if backend == "cuda" else "gfx1151",
            "raw_agent_target": "" if backend == "cuda" else "gfx1151",
            "native_subgroup_width": 32,
            "compute_unit_count": 8,
        }
        setattr(self, f"execution_{backend}_plugin_ready", False)
        setattr(self, f"execution_{backend}_plugin_artifact_id", "")

    def set_streamed_edges(self, mode):
        self.events.append(("mode", mode))

    def load(self, path, *arguments):
        backend = self.execution_device_execution_environment["backend"]
        self.events.append(("load", path, *arguments))
        setattr(self, f"execution_{backend}_plugin_ready", True)
        setattr(self, f"execution_{backend}_plugin_artifact_id", "artifact-a")


def _model(tmp_path, model_type="MACE"):
    path = tmp_path / "model.json"
    path.write_text(
        json.dumps(
            {
                "model_type": model_type,
                "execution_contracts": {"R1": {"fixture": "contract-a"}},
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("backend", "precision", "model_type", "load_arguments"),
    [
        ("cuda", "float32", "MACE", (8,)),
        ("hip", "float64", "MACEField", ()),
    ],
)
def test_prepare_device_artifact_without_ase_calculator(
    monkeypatch,
    tmp_path,
    backend,
    precision,
    model_type,
    load_arguments,
):
    evaluator = _Evaluator(backend)
    constructed = []
    initialized = []
    prepare_calls = []
    artifact = tmp_path / ("module.cubin" if backend == "cuda" else "module.hsaco")
    manifest = tmp_path / "manifest.json"

    def evaluator_class(path):
        constructed.append(path)
        return evaluator

    monkeypatch.setattr(
        device_artifact.symmetrix, "_kokkos_is_initialized", lambda: False
    )
    monkeypatch.setattr(
        device_artifact.symmetrix, "_init_kokkos", lambda: initialized.append(True)
    )
    monkeypatch.setattr(
        device_artifact.symmetrix,
        "MACEKokkosFloat" if precision == "float32" else "MACEKokkos",
        evaluator_class,
    )

    def prepare(source, **arguments):
        prepare_calls.append((source, arguments))
        return SimpleNamespace(
            status="built",
            available=True,
            cache_key="cache-a",
            artifact_path=artifact,
            manifest_path=manifest,
            diagnostics=("compiled",),
            reason=None,
        )

    plan = SimpleNamespace(
        selected_compiler="nvrtc" if backend == "cuda" else "hiprtc",
        policy="automatic",
        metadata={
            "artifact_id": "artifact-a",
            "persistent_blocks_per_compute_unit": 8,
        },
        loader=evaluator.load,
        ready_attribute=f"execution_{backend}_plugin_ready",
        artifact_attribute=f"execution_{backend}_plugin_artifact_id",
        variant_id="variant-a",
        edge_policy={
            "strategy": "wave",
            "persistent_blocks_per_compute_unit": 8,
        },
        build_attempt=lambda compiler: SimpleNamespace(
            source=f"{backend} source",
            prepare=prepare,
            arguments={"build": {"compiler": compiler}},
        ),
    )
    monkeypatch.setattr(
        device_artifact,
        "_factorized_device_backend_plan",
        lambda selected_backend, environment, contract, dtype, selected_evaluator: plan,
    )

    result = device_artifact.prepare_jit_device_artifact(
        _model(tmp_path, model_type),
        precision=precision,
        backend=backend,
        cache_root=tmp_path / "cache",
    )

    assert initialized == [True]
    assert constructed == [str((tmp_path / "model.json").resolve())]
    assert evaluator.events == [
        ("mode", "factorized"),
        ("load", str(artifact), *load_arguments),
    ]
    assert prepare_calls == [
        (
            f"{backend} source",
            {
                "build": {
                    "compiler": plan.selected_compiler,
                    "jit_generation_version": device_artifact.JIT_GENERATION_VERSION,
                },
                "cache_root": tmp_path / "cache",
            },
        )
    ]
    assert result.artifact_path == artifact
    assert result.artifact_id == "artifact-a"
    assert result.as_dict()["available"] is True
    assert result.as_dict()["backend"] == backend
    assert result.as_dict()["persistent_blocks_per_compute_unit"] == 8


def test_prepare_device_artifact_rejects_available_openmp(monkeypatch, tmp_path):
    evaluator = _Evaluator("host")
    evaluator.execution_device_execution_environment["execution_space"] = "OpenMP"
    monkeypatch.setattr(
        device_artifact.symmetrix, "_kokkos_is_initialized", lambda: True
    )
    monkeypatch.setattr(
        device_artifact.symmetrix, "MACEKokkosFloat", lambda path: evaluator
    )
    monkeypatch.setattr(
        device_artifact,
        "_factorized_device_backend_plan",
        lambda *args: pytest.fail("OpenMP must not create a device plan"),
    )

    with pytest.raises(
        device_artifact.JitDeviceArtifactError,
        match="require a CUDA or HIP backend",
    ):
        device_artifact.prepare_jit_device_artifact(
            _model(tmp_path), precision="float32"
        )

    assert evaluator.events == [("mode", "factorized")]


def test_prepare_device_artifact_cli_path_only(monkeypatch, capsys, tmp_path):
    artifact = tmp_path / "module.cubin"
    result = SimpleNamespace(artifact_path=artifact)
    calls = []
    monkeypatch.setattr(
        artifact_cli.native_symmetrix, "_kokkos_is_initialized", lambda: False
    )
    monkeypatch.setattr(
        artifact_cli,
        "prepare_jit_device_artifact",
        lambda *args, **kwargs: calls.append(kwargs) or result,
    )

    status = artifact_cli.main(
        ["--model", "model.json", "--precision", "float32", "--path-only"]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert calls[0]["include_low_memory_operators"] is False
    assert captured.out == f"{artifact}\n"
    assert captured.err == ""


def test_prepare_device_artifact_cli_lammps_arguments(monkeypatch, capsys, tmp_path):
    r1_artifact = tmp_path / "r1 module.cubin"
    m0_artifact = tmp_path / "m0.cubin"
    r0_artifact = tmp_path / "r0.cubin"
    result = SimpleNamespace(
        artifact_path=r1_artifact,
        edge_policy={"persistent_blocks_per_compute_unit": 6},
        operator_modules={
            "M0": {
                "implementation": "device_module",
                "artifact_path": str(m0_artifact),
                "schedule": "table",
            },
            "R0": {
                "implementation": "device_module",
                "artifact_path": str(r0_artifact),
            },
        },
    )
    monkeypatch.setattr(
        artifact_cli.native_symmetrix, "_kokkos_is_initialized", lambda: True
    )
    monkeypatch.setattr(
        artifact_cli, "prepare_jit_device_artifact", lambda *args, **kwargs: result
    )

    status = artifact_cli.main(
        ["--model", "model.json", "--precision", "float64", "--lammps-arguments"]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert captured.out == (
        f'jit_device_artifact "{r1_artifact}" '
        "jit_device_blocks_per_compute_unit 6 "
        f"jit_m0_device_artifact {m0_artifact} jit_m0_device_schedule table "
        f"jit_r0_device_artifact {r0_artifact}\n"
    )
    assert captured.err == ""


def test_prepare_device_artifact_cli_lammps_arguments_omits_builtin_operators(
    monkeypatch, capsys, tmp_path
):
    result = SimpleNamespace(
        artifact_path=tmp_path / "r1.cubin",
        edge_policy=None,
        operator_modules={
            "M0": {"implementation": "builtin"},
            "R0": {"implementation": "builtin"},
        },
    )
    monkeypatch.setattr(
        artifact_cli.native_symmetrix, "_kokkos_is_initialized", lambda: True
    )
    monkeypatch.setattr(
        artifact_cli, "prepare_jit_device_artifact", lambda *args, **kwargs: result
    )

    status = artifact_cli.main(
        ["--model", "model.json", "--precision", "float32", "--lammps-arguments"]
    )

    assert status == 0
    assert capsys.readouterr().out == (
        f"jit_device_artifact {result.artifact_path} "
        "jit_device_blocks_per_compute_unit 8\n"
    )


def test_prepare_device_artifact_cli_lammps_arguments_rejects_multiple_precisions():
    with pytest.raises(SystemExit, match="2"):
        artifact_cli.main(
            [
                "--model",
                "model.json",
                "--precision",
                "float32",
                "--precision",
                "float64",
                "--lammps-arguments",
            ]
        )


def test_prepare_device_artifact_cli_reports_structured_error(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise device_artifact.JitDeviceArtifactError(
            "compile failed", diagnostics=("nvrtc error",), cache_key="cache-a"
        )

    monkeypatch.setattr(artifact_cli, "prepare_jit_device_artifact", fail)
    monkeypatch.setattr(
        artifact_cli.native_symmetrix, "_kokkos_is_initialized", lambda: False
    )

    status = artifact_cli.main(["--model", "model.json", "--precision", "float64"])

    captured = capsys.readouterr()
    assert status == 1
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "status": "error",
        "available": False,
        "cache_key": "cache-a",
        "reason": "compile failed",
        "diagnostics": ["nvrtc error"],
    }


def test_prepare_device_artifact_cli_finalizes_owned_kokkos(
    monkeypatch, capsys, tmp_path
):
    initialized = iter((False, True))
    finalized = []
    result = SimpleNamespace(artifact_path=tmp_path / "module.cubin")
    monkeypatch.setattr(
        artifact_cli.native_symmetrix,
        "_kokkos_is_initialized",
        lambda: next(initialized),
    )
    monkeypatch.setattr(
        artifact_cli.native_symmetrix, "_kokkos_live_object_count", lambda: 0
    )
    monkeypatch.setattr(
        artifact_cli.native_symmetrix,
        "_finalize_kokkos",
        lambda: finalized.append(True),
    )
    monkeypatch.setattr(
        artifact_cli, "prepare_jit_device_artifact", lambda *args, **kwargs: result
    )

    status = artifact_cli.main(
        ["--model", "model.json", "--precision", "float32", "--path-only"]
    )

    assert status == 0
    assert finalized == [True]
    assert capsys.readouterr().out == f"{result.artifact_path}\n"


def test_prepare_device_artifact_cli_prepares_multiple_precisions_in_one_runtime(
    monkeypatch, capsys, tmp_path
):
    initialized = iter((False, True))
    finalized = []
    prepare_calls = []
    monkeypatch.setattr(
        artifact_cli.native_symmetrix,
        "_kokkos_is_initialized",
        lambda: next(initialized),
    )
    monkeypatch.setattr(
        artifact_cli.native_symmetrix, "_kokkos_live_object_count", lambda: 0
    )
    monkeypatch.setattr(
        artifact_cli.native_symmetrix,
        "_finalize_kokkos",
        lambda: finalized.append(True),
    )

    def prepare(*args, precision, **kwargs):
        prepare_calls.append(precision)
        artifact_path = tmp_path / f"module-{precision}.cubin"
        return SimpleNamespace(
            artifact_path=artifact_path,
            as_dict=lambda: {
                "precision": precision,
                "artifact_path": str(artifact_path),
            },
        )

    monkeypatch.setattr(artifact_cli, "prepare_jit_device_artifact", prepare)

    status = artifact_cli.main(
        [
            "--model",
            "model.json",
            "--precision",
            "float32",
            "--precision",
            "float64",
            "--precision",
            "float32",
        ]
    )

    assert status == 0
    assert prepare_calls == ["float32", "float64"]
    assert finalized == [True]
    assert json.loads(capsys.readouterr().out) == [
        {
            "precision": "float32",
            "artifact_path": str(tmp_path / "module-float32.cubin"),
        },
        {
            "precision": "float64",
            "artifact_path": str(tmp_path / "module-float64.cubin"),
        },
    ]


def test_pair_symmetrix_exposes_explicit_device_artifact_loader():
    repository = Path(__file__).resolve().parents[2]
    source = (repository / "pair_symmetrix/pair_symmetrix_mace_kokkos.cpp").read_text()

    assert 'token == "jit_device_artifact"' in source
    assert 'token == "jit_m0_device_artifact"' in source
    assert 'token == "jit_r0_device_artifact"' in source
    assert 'token == "jit_m0_device_schedule"' in source
    assert 'token == "jit_device_blocks_per_compute_unit"' in source
    assert "consumed != value.size()" in source
    assert "jit_device_artifact, jit_device_blocks_per_compute_unit" in source
    assert "mace->load_m0_device_module(" in source
    assert "mace->load_r0_device_module(" in source
    assert source.index("mace->load_jit_device_plugin(") < source.index(
        "mace->load_m0_device_module("
    )
    assert source.index("mace->load_m0_device_module(") < source.index(
        "mace->set_execution_plan_request("
    )
    assert source.index("mace->load_r0_device_module(") < source.index(
        "mace->set_execution_plan_request("
    )


def test_pair_symmetrix_fixed_workspace_is_explicit_and_fail_closed():
    repository = Path(__file__).resolve().parents[2]
    source = (repository / "pair_symmetrix/pair_symmetrix_mace_kokkos.cpp").read_text()

    assert 'token == "low_memory"' in source
    assert 'token == "profile"' in source
    assert 'token == "allow_fixed_workspace"' in source
    assert "allow_fixed_workspace yes requires streamed_edges direct" in source
    assert (
        '"jit_host_artifact or jit_device_artifact; direct execution does not "'
        in source
    )
    assert "mace->set_allow_fixed_workspace(allow_fixed_workspace)" in source
    assert "mace->set_execution_plan_request(" in source
    assert "\"Symmetrix execution profile='{}'\\n\"" in source


def test_pair_symmetrix_publishes_device_atom_energies_to_lammps():
    repository = Path(__file__).resolve().parents[2]
    source = (repository / "pair_symmetrix/pair_symmetrix_mace_kokkos.cpp").read_text()

    assert source.count("k_eatom.sync_host();") == 3


def test_pair_symmetrix_mpi_stages_only_boundary_packets():
    repository = Path(__file__).resolve().parents[2]
    source = (repository / "pair_symmetrix/pair_symmetrix_mace_kokkos.cpp").read_text()
    header = (repository / "pair_symmetrix/pair_symmetrix_mace_kokkos.h").read_text()

    assert "create_mirror_view_and_copy(Kokkos::HostSpace(), H1)" not in source
    assert "create_mirror_view_and_copy(Kokkos::HostSpace(), H1_adj)" not in source
    assert source.count('_comm_host_staged"') == 4
    assert "Kokkos::View<double*> legacy_comm_packet" in header
    assert "Kokkos::View<int*> legacy_comm_indices" in header
    assert "checked_comm_value_count(std::size_t, const char *)" in header
    assert "std::numeric_limits<int>::max()" in source
    assert 'checked_comm_value_count(1, "per-atom")' in source
    assert 'num_feature_nodes_size, "local feature-state"' in source
    for counter in (
        "symmetrix_mpi_staged_packet_d2h_bytes",
        "symmetrix_mpi_staged_packet_h2d_bytes",
        "symmetrix_mpi_staged_index_h2d_bytes",
    ):
        assert counter in source
    assert (
        source.count(
            "execution_mpi_staged_packet_d2h_bytes += num_values * sizeof(double);"
        )
        == 2
    )
    assert (
        source.count(
            "execution_mpi_staged_packet_h2d_bytes += num_values * sizeof(double);"
        )
        == 2
    )
    assert (
        source.count(
            "execution_mpi_staged_index_h2d_bytes += num_indices * sizeof(int);"
        )
        == 2
    )

    callback_start = source.index(
        "int PairSymmetrixMACEKokkos<DeviceType, Precision>::pack_forward_comm(int"
    )
    callback_end = source.index(
        "void PairSymmetrixMACEKokkos<DeviceType, Precision>::compute_no_domain_decomposition"
    )
    callbacks = source[callback_start:callback_end]
    assert "Kokkos::fence();" not in callbacks
    assert "return n*num_LM*num_channels;" not in callbacks
    assert "ii*num_LM*num_channels" not in callbacks
    assert "i*num_LM*num_channels" not in callbacks
    for operation in (
        "forward-pack",
        "forward-unpack",
        "reverse-pack",
        "reverse-unpack",
    ):
        assert callbacks.count(f'"{operation}"') == 2


def test_pair_symmetrix_contiguous_comm_uses_copy_or_scalar_conversion():
    repository = Path(__file__).resolve().parents[2]
    source = (repository / "pair_symmetrix/pair_symmetrix_mace_kokkos.cpp").read_text()

    assert "std::is_same_v<Precision, double>" in source
    assert "Kokkos::deep_copy(target, source);" in source
    assert "Kokkos::deep_copy(output, input);" in source
    assert "PairSymmetrixMACEKokkos::unpack_forward_comm_convert" in source
    assert "PairSymmetrixMACEKokkos::pack_reverse_comm_convert" in source

    forward_unpack = source[
        source.index(
            "void PairSymmetrixMACEKokkos<DeviceType, Precision>::unpack_forward_comm_kokkos"
        ) : source.index(
            "int PairSymmetrixMACEKokkos<DeviceType, Precision>::pack_reverse_comm"
        )
    ]
    reverse_pack = source[
        source.index(
            "int PairSymmetrixMACEKokkos<DeviceType, Precision>::pack_reverse_comm_kokkos"
        ) : source.index(
            "void PairSymmetrixMACEKokkos<DeviceType, Precision>::unpack_reverse_comm"
        )
    ]
    assert "MDRangePolicy" not in forward_unpack
    assert "MDRangePolicy" not in reverse_pack


def test_extract_mace_preserves_conversion_without_device_preparation(
    monkeypatch, tmp_path
):
    output_path = tmp_path / "model.json"
    model_data = {"model_type": "MACE", "execution_contracts": {"R1": {}}}
    monkeypatch.setattr(
        extract_cli, "extract_mace_data", lambda *args, **kwargs: model_data
    )
    monkeypatch.setattr(
        extract_cli,
        "prepare_jit_device_artifact",
        lambda *args, **kwargs: pytest.fail("preparation must remain opt-in"),
    )

    status = extract_cli.main(
        ["--model", "checkpoint.model", "--output", str(output_path)]
    )

    assert status == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == model_data


def test_extract_mace_prepares_device_artifact_after_conversion(
    monkeypatch,
    tmp_path,
):
    output_path = tmp_path / "model.json"
    cache_root = tmp_path / "cache"
    model_data = {"model_type": "MACEField", "execution_contracts": {"R1": {}}}
    prepare_calls = []
    monkeypatch.setattr(
        extract_cli, "extract_mace_data", lambda *args, **kwargs: model_data
    )
    monkeypatch.setattr(
        extract_cli,
        "prepare_jit_device_artifact",
        lambda arguments: prepare_calls.append(arguments) or 7,
    )

    status = extract_cli.main(
        [
            "--model",
            "checkpoint.model",
            "--output",
            str(output_path),
            "--prepare-jit-device-artifact",
            "--jit-device-backend",
            "cuda",
            "--jit-device-cache-root",
            str(cache_root),
        ]
    )

    assert status == 7
    assert json.loads(output_path.read_text(encoding="utf-8")) == model_data
    assert prepare_calls == [
        [
            "--model",
            str(output_path),
            "--precision",
            "float32",
            "--precision",
            "float64",
            "--backend",
            "cuda",
            "--cache-root",
            str(cache_root),
        ]
    ]


def test_extract_mace_rejects_device_preparation_for_pair_splines(monkeypatch):
    monkeypatch.setattr(
        extract_cli,
        "extract_mace_data",
        lambda *args, **kwargs: pytest.fail(
            "invalid options must fail before extraction"
        ),
    )

    with pytest.raises(SystemExit, match="2"):
        extract_cli.main(
            [
                "--model",
                "checkpoint.model",
                "--radial-format",
                "pair-splines",
                "--prepare-jit-device-artifact",
            ]
        )
