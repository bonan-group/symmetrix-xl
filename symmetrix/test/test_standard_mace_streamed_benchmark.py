import importlib.util
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/standard_mace_streamed_benchmark.py"


def _load_benchmark(monkeypatch):
    native = types.SimpleNamespace(__file__=str(ROOT / "fake-symmetrix.so"))
    package = types.ModuleType("symmetrix")
    package.Symmetrix = object
    package.symmetrix = native
    ase = types.ModuleType("ase")
    ase_build = types.ModuleType("ase.build")
    ase_build.bulk = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "symmetrix", package)
    monkeypatch.setitem(sys.modules, "symmetrix.symmetrix", native)
    monkeypatch.setitem(sys.modules, "ase", ase)
    monkeypatch.setitem(sys.modules, "ase.build", ase_build)

    spec = importlib.util.spec_from_file_location(
        "standard_mace_streamed_benchmark_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_benchmark_help_exposes_nvtx_and_storage_policies(monkeypatch, capsys):
    benchmark = _load_benchmark(monkeypatch)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--help"])
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "--nvtx" in output
    assert "--phi1-policy" in output
    assert "--harmonic-storage-policy" in output


@pytest.mark.parametrize(
    "benchmark_name,script,option,canonical_modes",
    [
        (
            "standard",
            SCRIPT,
            "--modes",
            ("materialized", "non-compiled", "direct"),
        ),
    ],
)
def test_benchmark_cli_rejects_removed_second_interaction(
    monkeypatch, capsys, benchmark_name, script, option, canonical_modes
):
    benchmark = _load_benchmark(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(script), "missing.json", option, "second_interaction"],
    )

    with pytest.raises(SystemExit) as exc:
        benchmark.main()

    assert exc.value.code == 2
    error = capsys.readouterr().err
    for mode in canonical_modes:
        assert mode in error


def test_full_benchmark_help_exposes_per_atom_gate(monkeypatch, capsys):
    benchmark = _load_benchmark(monkeypatch)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--help"])
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 0
    assert "--max-us-per-atom" in capsys.readouterr().out


def test_full_benchmark_help_exposes_neighbor_skin(monkeypatch, capsys):
    benchmark = _load_benchmark(monkeypatch)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--help"])
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 0
    assert "--neighbor-skin" in capsys.readouterr().out


def test_full_benchmark_per_atom_gate_requires_one_mode(monkeypatch, capsys):
    benchmark = _load_benchmark(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "missing.json",
            "--modes",
            "materialized,factorized",
            "--max-us-per-atom",
            "50",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 2
    assert "--max-us-per-atom requires exactly one mode" in capsys.readouterr().err


def test_per_atom_summary_reports_microseconds(monkeypatch):
    benchmark = _load_benchmark(monkeypatch)
    summary = benchmark._per_atom_summary([12.8, 13.2], 256)
    assert summary["median_us_per_atom"] == pytest.approx(50.78125)


def test_full_benchmark_help_exposes_m0_executor(monkeypatch, capsys):
    benchmark = _load_benchmark(monkeypatch)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--help"])
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 0
    assert "--standard-m0-executor" in capsys.readouterr().out


def test_full_benchmark_help_exposes_m1_recompute(monkeypatch, capsys):
    benchmark = _load_benchmark(monkeypatch)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--help"])
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "--m1-polynomial-policy" in output
    assert "--m1-recompute-tile-channels" in output
    assert "--mh0-state-policy" in output
    assert "--mh1-node-state-policy" in output
    assert "reuse-adjoints-v1" in output


def test_full_benchmark_accepts_mh1_evaluator_without_mh0_policy_setters(
    monkeypatch,
):
    benchmark = _load_benchmark(monkeypatch)
    evaluator = types.SimpleNamespace(supports_streamed_edges=True)
    captured = {}

    def make_calculator(*args, **kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(evaluator=evaluator)

    monkeypatch.setattr(benchmark, "Symmetrix", make_calculator)
    calculator = benchmark._make_calculator(
        "mh1.json",
        "kokkos",
        "float32",
        "direct",
        execution_mh1_node_state_policy="recompute-v1",
    )

    assert calculator.evaluator is evaluator
    assert captured["execution_mh1_node_state_policy"] == "recompute-v1"


def test_m1_tile_defaults_to_runtime_selection_and_allows_explicit_pin(monkeypatch):
    benchmark = _load_benchmark(monkeypatch)
    selected_tiles = []
    evaluator = types.SimpleNamespace(
        supports_streamed_edges=True,
        _set_m1_recompute_tile_channels=selected_tiles.append,
    )
    monkeypatch.setattr(
        benchmark,
        "Symmetrix",
        lambda *args, **kwargs: types.SimpleNamespace(evaluator=evaluator),
    )

    benchmark._make_calculator("model.json", "kokkos", "float64", "direct")
    assert selected_tiles == []

    benchmark._make_calculator(
        "model.json",
        "kokkos",
        "float64",
        "direct",
        m1_recompute_tile_channels=8,
    )
    assert selected_tiles == [8]


def test_mh1_workspace_policy_prefers_selected_jit_variant(monkeypatch):
    benchmark = _load_benchmark(monkeypatch)
    calculator = types.SimpleNamespace(
        evaluator=types.SimpleNamespace(execution_mh1_node_state_policy=None),
        jit_node_state_policy="recompute-v1",
    )

    assert benchmark._mh1_node_state_policy(calculator) == "recompute-v1"


def test_mh0_state_report_tracks_reused_and_auxiliary_bytes(monkeypatch):
    benchmark = _load_benchmark(monkeypatch)
    evaluator = types.SimpleNamespace(
        mh0_state_policy_request="reuse-adjoints-v1",
        mh0_state_policy="reuse-adjoints-v1",
        mh0_state_policy_fallback_reason="",
        mh0_reused_state_bytes=17_252_352,
        mh0_auxiliary_state_bytes=13_824,
    )

    assert benchmark._mh0_state_report(evaluator) == {
        "requested_policy": "reuse-adjoints-v1",
        "selected_policy": "reuse-adjoints-v1",
        "fallback_reason": "",
        "reused_state_bytes": 17_252_352,
        "auxiliary_state_bytes": 13_824,
        "readout_workspace_bytes": 0,
        "readout_policy": "retained",
    }


def test_phi1_report_tracks_policy_and_workspace(monkeypatch):
    benchmark = _load_benchmark(monkeypatch)
    evaluator = types.SimpleNamespace(
        phi1_policy="receiver-local",
        phi1_workspace_bytes=0,
    )

    assert benchmark._phi1_report(evaluator) == {
        "selected_policy": "receiver-local",
        "workspace_bytes": 0,
        "a1_blas_flatten_workspace_bytes": 0,
    }


def test_harmonic_storage_report_tracks_policy_and_launches(monkeypatch):
    benchmark = _load_benchmark(monkeypatch)
    evaluator = types.SimpleNamespace(
        harmonic_storage_policy_request="automatic",
        harmonic_storage_policy="y-only-direct-v1",
        harmonic_storage_selection_reason="capacity bundle selected",
        harmonic_storage_fallback_reason="",
        harmonic_value_bytes=4096,
        harmonic_gradient_bytes=0,
        shuffled_coordinate_bytes=0,
        execution_direct_harmonic_launch_count=7,
    )

    assert benchmark._harmonic_storage_report(evaluator) == {
        "requested_policy": "automatic",
        "selected_policy": "y-only-direct-v1",
        "selection_reason": "capacity bundle selected",
        "fallback_reason": "",
        "value_bytes": 4096,
        "gradient_bytes": 0,
        "shuffled_coordinate_bytes": 0,
        "direct_value_launches": 7,
    }


def test_low_memory_report_distinguishes_request_from_selected_bundle(monkeypatch):
    benchmark = _load_benchmark(monkeypatch)
    evaluator = types.SimpleNamespace(
        low_memory_requested=True,
        low_memory=False,
        low_memory_policy="speed",
        low_memory_selection_reason="speed estimate fits",
        low_memory_device_free_bytes=24_000,
        low_memory_device_total_bytes=32_000,
        low_memory_reserve_bytes=2_000,
        low_memory_available_bytes=24_000,
        low_memory_speed_estimated_bytes=18_000,
        low_memory_capacity_y_only_estimated_bytes=12_000,
        low_memory_capacity_retained_estimated_bytes=15_000,
        low_memory_capacity_estimated_bytes=12_000,
        low_memory_selected_estimated_bytes=18_000,
    )

    assert benchmark._low_memory_report(evaluator) == {
        "requested": True,
        "active": False,
        "selected_policy": "speed",
        "selection_reason": "speed estimate fits",
        "device_free_bytes": 24_000,
        "device_total_bytes": 32_000,
        "reserve_bytes": 2_000,
        "available_bytes": 24_000,
        "speed_estimated_bytes": 18_000,
        "capacity_y_only_estimated_bytes": 12_000,
        "capacity_retained_estimated_bytes": 15_000,
        "capacity_estimated_bytes": 12_000,
        "selected_estimated_bytes": 18_000,
        "geometry_growth_reason": "",
    }


def test_low_memory_report_uses_calculator_owned_mh1_policy(monkeypatch):
    benchmark = _load_benchmark(monkeypatch)
    calculator = types.SimpleNamespace(
        low_memory_request=True,
        low_memory=True,
        low_memory_policy="mh1-retain-interaction-v1",
        low_memory_selection_reason="MH-1 hybrid policy selected",
        evaluator=types.SimpleNamespace(),
    )

    report = benchmark._low_memory_report(calculator)

    assert report["requested"] is True
    assert report["active"] is True
    assert report["selected_policy"] == "mh1-retain-interaction-v1"
    assert report["selection_reason"] == "MH-1 hybrid policy selected"


@pytest.mark.parametrize(
    "extra_args,expected_error",
    [
        (
            [],
            "--edge-geometry-policy=unit-f32-radius-f64-v1",
        ),
        (
            [
                "--edge-geometry-policy",
                "unit-f32-radius-f64-v1",
                "--phi1-policy",
                "receiver-local",
            ],
            "--phi1-policy=retained or channel-tiled-64",
        ),
    ],
)
def test_y_only_harmonics_require_compact_geometry_and_compatible_phi1(
    monkeypatch, capsys, extra_args, expected_error
):
    benchmark = _load_benchmark(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "missing.json",
            "--backend",
            "kokkos",
            "--dtype",
            "float32",
            "--modes",
            "direct",
            "--m1-polynomial-policy",
            "recompute",
            "--harmonic-storage-policy",
            "y-only-direct-v1",
            *extra_args,
        ],
    )

    with pytest.raises(SystemExit) as exc:
        benchmark.main()

    assert exc.value.code == 2
    assert expected_error in capsys.readouterr().err


def test_y_only_harmonics_accept_retained_phi1(monkeypatch):
    benchmark = _load_benchmark(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "missing.json",
            "--backend",
            "kokkos",
            "--dtype",
            "float32",
            "--modes",
            "direct",
            "--m1-polynomial-policy",
            "recompute",
            "--edge-geometry-policy",
            "unit-f32-radius-f64-v1",
            "--phi1-policy",
            "retained",
            "--harmonic-storage-policy",
            "y-only-direct-v1",
        ],
    )

    with pytest.raises(FileNotFoundError):
        benchmark.main()


def test_y_only_harmonics_accept_float64_cartesian_geometry(monkeypatch):
    benchmark = _load_benchmark(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "missing.json",
            "--backend",
            "kokkos",
            "--dtype",
            "float64",
            "--modes",
            "direct",
            "--edge-geometry-policy",
            "cartesian-f64-v1",
            "--phi1-policy",
            "retained",
            "--harmonic-storage-policy",
            "y-only-direct-v1",
        ],
    )

    with pytest.raises(FileNotFoundError):
        benchmark.main()


def test_full_benchmark_m0_executor_requires_fully_streamed_mode(monkeypatch, capsys):
    benchmark = _load_benchmark(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "missing.json",
            "--modes",
            "materialized",
            "--standard-m0-executor",
            "standard",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 2
    assert "--standard-m0-executor requires a streamed mode" in capsys.readouterr().err


def test_full_benchmark_rejects_jit_controls_for_non_compiled(monkeypatch, capsys):
    benchmark = _load_benchmark(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "missing.json",
            "--modes",
            "non-compiled",
            "--factorized-r1-source-strategy",
            "jit_plugin",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        benchmark.main()

    assert exc.value.code == 2
    assert (
        "--factorized-r1-source-strategy requires a prepared mode"
        in capsys.readouterr().err
    )


def test_standard_m0_report_tracks_provenance_launches_and_workspace(monkeypatch):
    benchmark = _load_benchmark(monkeypatch)
    evaluator = types.SimpleNamespace(
        standard_m0_module_ready=True,
        standard_m0_module_fallback_reason="",
        standard_m0_module_id="omat-medium-m0-v1",
        standard_m0_module_revision=2,
        standard_m0_model_structure_fingerprint="sha256:fixture",
        standard_m0_selected_executor="standard",
        standard_m0_poly_values_active_bytes=0,
        standard_m0_poly_values_capacity_bytes=1024,
        standard_m0_poly_adjoints_active_bytes=0,
        standard_m0_poly_adjoints_capacity_bytes=2048,
    )
    before = {
        "standard_m0_module_forward_launch_count": 2,
        "standard_m0_module_reverse_launch_count": 3,
    }
    after = {
        "standard_m0_module_forward_launch_count": 7,
        "standard_m0_module_reverse_launch_count": 8,
    }

    assert benchmark._standard_m0_report(evaluator, "standard", before, after) == {
        "ready": True,
        "fallback_reason": "",
        "module_id": "omat-medium-m0-v1",
        "module_revision": 2,
        "model_structure_fingerprint": "sha256:fixture",
        "requested_executor": "standard",
        "selected_executor": "standard",
        "module_forward_launches": 7,
        "module_reverse_launches": 8,
        "measured_module_forward_launches": 5,
        "measured_module_reverse_launches": 5,
        "poly_values": {"active_bytes": 0, "capacity_bytes": 1024},
        "poly_adjoints": {"active_bytes": 0, "capacity_bytes": 2048},
    }


def test_m1_polynomial_report_tracks_storage_and_launches(monkeypatch):
    benchmark = _load_benchmark(monkeypatch)
    evaluator = types.SimpleNamespace(
        m1_polynomial_policy="recompute",
        m1_recompute_tile_channels=16,
        m1_recompute_scratch_bytes=18176,
        m1_poly_values_active_bytes=0,
        m1_poly_values_capacity_bytes=0,
        m1_poly_adjoints_active_bytes=0,
        m1_poly_adjoints_capacity_bytes=0,
    )
    before = {
        "m1_recompute_forward_launch_count": 2,
        "m1_recompute_reverse_launch_count": 3,
    }
    after = {
        "m1_recompute_forward_launch_count": 9,
        "m1_recompute_reverse_launch_count": 10,
    }

    assert benchmark._m1_polynomial_report(evaluator, before, after) == {
        "policy": "recompute",
        "tile_channels": 16,
        "scratch_bytes": 18176,
        "standard_module_ready": False,
        "standard_module_forward_launches": 0,
        "standard_module_reverse_launches": 0,
        "poly_values": {"active_bytes": 0, "capacity_bytes": 0},
        "poly_adjoints": {"active_bytes": 0, "capacity_bytes": 0},
        "forward_launches": 9,
        "reverse_launches": 10,
        "measured_forward_launches": 7,
        "measured_reverse_launches": 7,
    }


def test_full_benchmark_nvtx_requires_kokkos(monkeypatch, capsys):
    benchmark = _load_benchmark(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "missing.json", "--backend", "serial", "--nvtx"],
    )
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 2
    assert "--nvtx requires the kokkos backend" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("modes", "sizes", "repeats"),
    [
        ("factorized,all_interactions", "6", "1"),
        ("factorized", "5,6", "1"),
        ("factorized", "6", "2"),
    ],
)
def test_full_benchmark_nvtx_requires_one_iteration(
    monkeypatch, capsys, modes, sizes, repeats
):
    benchmark = _load_benchmark(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "missing.json",
            "--backend",
            "kokkos",
            "--dtype",
            "float32",
            "--modes",
            modes,
            "--sizes",
            sizes,
            "--repeats",
            repeats,
            "--nvtx",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 2
    assert (
        "--nvtx requires exactly one mode, one size, and --repeats=1"
        in capsys.readouterr().err
    )
