import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from symmetrix import jit_host_artifact as host_artifact
from symmetrix.cli import prepare_jit_host_artifact as artifact_cli


class _Evaluator:
    def __init__(self):
        self.events = []
        self.jit_host_plugin_ready = False
        self.jit_host_plugin_artifact_id = ""

    def set_streamed_edges(self, mode):
        self.events.append(("mode", mode))

    def _load_jit_host_plugin(self, path):
        self.events.append(("load", path))
        self.jit_host_plugin_ready = True
        self.jit_host_plugin_artifact_id = "artifact-a"


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


def test_prepare_host_artifact_without_ase_calculator(monkeypatch, tmp_path):
    evaluator = _Evaluator()
    initialized = []
    artifact = tmp_path / "plugin.so"
    manifest = tmp_path / "manifest.json"
    prepare_calls = []

    monkeypatch.setattr(
        host_artifact.symmetrix, "_kokkos_is_initialized", lambda: False
    )
    monkeypatch.setattr(
        host_artifact.symmetrix, "_init_kokkos", lambda: initialized.append(True)
    )
    monkeypatch.setattr(
        host_artifact.symmetrix, "_kokkos_default_execution_space", lambda: "OpenMP"
    )
    monkeypatch.setattr(
        host_artifact.symmetrix,
        "_required_jit_generation_version",
        lambda: host_artifact.JIT_GENERATION_VERSION,
    )
    monkeypatch.setattr(
        host_artifact.symmetrix, "MACEKokkosFloat", lambda path: evaluator
    )
    monkeypatch.setattr(
        host_artifact,
        "jit_r1_host_plugin_metadata",
        lambda contract, precision: {
            "abi": "abi-a",
            "abi_version": 2,
            "artifact_id": "artifact-a",
        },
    )
    monkeypatch.setattr(
        host_artifact,
        "render_jit_r1_host_plugin",
        lambda contract, precision: "host source",
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

    monkeypatch.setattr(host_artifact, "prepare_jit_artifact", prepare)

    result = host_artifact.prepare_jit_host_artifact(
        _model(tmp_path),
        precision="float32",
        cache_root=tmp_path / "cache",
        host_target="portable",
    )

    assert initialized == [True]
    assert evaluator.events == [("mode", "direct"), ("load", str(artifact))]
    assert prepare_calls[0][0] == "host source"
    assert prepare_calls[0][1]["cache_root"] == tmp_path / "cache"
    assert prepare_calls[0][1]["host_target"] == "portable"
    assert result.artifact_path == artifact
    assert result.artifact_id == "artifact-a"
    assert result.as_dict()["backend"] == "host"
    assert result.as_dict()["environment"]["execution_space"] == "OpenMP"


def test_prepare_host_artifact_rejects_device_execution(monkeypatch, tmp_path):
    evaluator = _Evaluator()
    monkeypatch.setattr(host_artifact.symmetrix, "_kokkos_is_initialized", lambda: True)
    monkeypatch.setattr(
        host_artifact.symmetrix, "MACEKokkosFloat", lambda path: evaluator
    )
    monkeypatch.setattr(
        host_artifact.symmetrix, "_kokkos_default_execution_space", lambda: "Cuda"
    )

    with pytest.raises(
        host_artifact.JitHostArtifactError,
        match="require a Serial or OpenMP Kokkos build",
    ):
        host_artifact.prepare_jit_host_artifact(_model(tmp_path), precision="float32")

    assert evaluator.events == [("mode", "direct")]


@pytest.mark.parametrize("contracts", [None, [], "invalid"])
def test_prepare_host_artifact_rejects_malformed_contract_container(
    tmp_path, contracts
):
    model = tmp_path / "model.json"
    model.write_text(
        json.dumps({"model_type": "MACE", "execution_contracts": contracts}),
        encoding="utf-8",
    )

    with pytest.raises(
        host_artifact.JitHostArtifactError,
        match="does not contain an Execution contracts object",
    ):
        host_artifact.prepare_jit_host_artifact(model, precision="float32")


def test_prepare_host_artifact_cli_path_only(monkeypatch, capsys, tmp_path):
    artifact = tmp_path / "plugin.so"
    result = SimpleNamespace(artifact_path=artifact)
    monkeypatch.setattr(
        artifact_cli.native_symmetrix, "_kokkos_is_initialized", lambda: True
    )
    monkeypatch.setattr(
        artifact_cli, "prepare_jit_host_artifact", lambda *args, **kwargs: result
    )

    status = artifact_cli.main(
        ["--model", "model.json", "--precision", "float32", "--path-only"]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert captured.out == f"{artifact}\n"
    assert captured.err == ""


def test_prepare_host_artifact_cli_reports_structured_cleanup_error(
    monkeypatch, capsys, tmp_path
):
    result = SimpleNamespace(artifact_path=tmp_path / "plugin.so")
    monkeypatch.setattr(
        artifact_cli.native_symmetrix, "_kokkos_is_initialized", lambda: False
    )
    monkeypatch.setattr(
        artifact_cli, "prepare_jit_host_artifact", lambda *args, **kwargs: result
    )

    def fail_cleanup(initialized_before):
        raise ValueError("cleanup failed")

    monkeypatch.setattr(artifact_cli, "_finalize_owned_kokkos", fail_cleanup)

    status = artifact_cli.main(["--model", "model.json", "--precision", "float32"])

    captured = capsys.readouterr()
    assert status == 1
    assert captured.out == ""
    assert json.loads(captured.err)["reason"] == (
        "could not finalize the CLI-owned Kokkos runtime: cleanup failed"
    )


def test_pair_symmetrix_exposes_explicit_host_artifact_loader():
    repository = Path(__file__).resolve().parents[2]
    source = (repository / "pair_symmetrix/pair_symmetrix_mace_kokkos.cpp").read_text()
    header = (repository / "pair_symmetrix/pair_symmetrix_mace_kokkos.h").read_text()

    assert 'token == "jit_host_artifact"' in source
    assert "mace->load_jit_host_plugin(jit_host_artifact)" in source
    assert "mace->jit_host_plugin_ready()" in source
    assert "std::string jit_host_artifact" in header
