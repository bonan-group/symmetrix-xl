import ctypes
import gc
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.neighborlist import neighbor_list
from compact_r1_model import add_standard_base_contracts, foundation_r1_test_model
from symmetrix import Symmetrix
from symmetrix import symmetrix as native_symmetrix

CONTRACT_ROOT = Path(__file__).resolve().parent / "data" / "execution_contracts"
BENCHMARK_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "les_particle_mesh_benchmark.py"
)
QUALIFICATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "les_particle_mesh_qualification.py"
)


def test_native_maceles_evaluator_types_extend_mace_evaluators():
    assert issubclass(native_symmetrix.MACELESKokkos, native_symmetrix.MACEKokkos)
    assert issubclass(
        native_symmetrix.MACELESKokkosFloat,
        native_symmetrix.MACEKokkosFloat,
    )


def _load_particle_mesh_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "les_particle_mesh_benchmark", BENCHMARK_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load benchmark module from {BENCHMARK_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_particle_mesh_qualification_module():
    spec = importlib.util.spec_from_file_location(
        "les_particle_mesh_qualification", QUALIFICATION_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot load qualification module from {QUALIFICATION_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_particle_mesh_benchmark_record_validates_workload_and_timing():
    benchmark = _load_particle_mesh_benchmark_module()
    record = benchmark.example_record_for_testing()
    benchmark.validate_record(record)

    record["mesh"] = [31, 32, 32]
    with pytest.raises(ValueError, match="mesh"):
        benchmark.validate_record(record)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sigma", 0.0, "sigma"),
        ("sigma", np.inf, "sigma"),
        ("dl", 0.0, "dl"),
        ("dl", np.nan, "dl"),
        ("directed_edges", -1, "directed_edges"),
        ("directed_edges", 1.5, "directed_edges"),
        ("warmups", -1, "warmups"),
        ("warmups", 1.5, "warmups"),
        ("sample_count", 2, "sample_count"),
    ],
)
def test_particle_mesh_benchmark_rejects_malformed_physical_record(
    field, value, message
):
    benchmark = _load_particle_mesh_benchmark_module()
    record = benchmark.example_record_for_testing()
    record["sample_count"] = len(record["samples_ns"])
    record[field] = value

    with pytest.raises(ValueError, match=message):
        benchmark.validate_record(record)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cell", [[1.0, 0.0, 0.0]], "cell"),
        (
            "cell",
            [[np.inf, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "cell",
        ),
        ("precision", "float16", "precision"),
        ("streamed_edges", "indirect", "streamed_edges"),
        ("skin", np.nan, "skin"),
        ("backend", "metal", "backend"),
        ("kokkos_execution_space", "HIP", "kokkos_execution_space"),
        ("model_sha256", "g" * 64, "model_sha256"),
        ("structure_sha256", "g" * 64, "structure_sha256"),
        (
            "mace_les_evaluator_fingerprint",
            "A" * 64,
            "mace_les_evaluator_fingerprint",
        ),
        ("native_extension_sha256", "g" * 64, "native_extension_sha256"),
    ],
)
def test_particle_mesh_benchmark_rejects_invalid_schema_identity(field, value, message):
    benchmark = _load_particle_mesh_benchmark_module()
    record = benchmark.example_record_for_testing()
    record[field] = value

    with pytest.raises(ValueError, match=message):
        benchmark.validate_record(record)


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("source_commit", None, ValueError),
        ("source_commit", "", ValueError),
        ("source_dirty", None, ValueError),
        ("source_dirty", "false", TypeError),
        ("source_content_sha256", None, ValueError),
        ("source_content_sha256", "g" * 64, ValueError),
        ("native_source_content_sha256", None, ValueError),
        ("native_source_content_sha256", "A" * 64, ValueError),
    ],
)
def test_particle_mesh_benchmark_rejects_invalid_source_provenance(
    field, value, error_type
):
    benchmark = _load_particle_mesh_benchmark_module()
    record = benchmark.example_record_for_testing()
    provenance = record["source_provenance"]
    if value is None:
        provenance.pop(field)
    else:
        provenance[field] = value

    with pytest.raises(error_type, match=field):
        benchmark.validate_record(record)


class _FakeRoctxOperation:
    def __init__(self, *statuses):
        self.statuses = list(statuses or (0,))
        self.calls = []
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        self.calls.append(args)
        result = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        if isinstance(result, BaseException):
            raise result
        return result


class _FakeRoctx:
    def __init__(self, *, pause=(0, 0), resume=(0,), push=(0,), pop=(0,)):
        self.roctxProfilerPause = _FakeRoctxOperation(*pause)
        self.roctxProfilerResume = _FakeRoctxOperation(*resume)
        self.roctxRangePushA = _FakeRoctxOperation(*push)
        self.roctxRangePop = _FakeRoctxOperation(*pop)


def test_particle_mesh_roctx_discovery_uses_configured_and_tool_roots(monkeypatch):
    benchmark = _load_particle_mesh_benchmark_module()
    candidates = getattr(benchmark, "_roctx_library_candidates", None)
    assert callable(candidates)
    monkeypatch.setenv("ROCM_PATH", "/configured/rocm")
    monkeypatch.setattr(
        benchmark.shutil,
        "which",
        lambda name: "/tool/rocm/bin/" + name if name == "rocprofv3" else None,
    )
    monkeypatch.setattr(benchmark.ctypes.util, "find_library", lambda _: "found.so")

    paths = candidates()

    assert "/configured/rocm/lib/librocprofiler-sdk-roctx.so" in paths
    assert "/configured/rocm/lib64/librocprofiler-sdk-roctx.so" in paths
    assert "/tool/rocm/lib/librocprofiler-sdk-roctx.so" in paths
    assert "found.so" in paths
    assert "core-7.14" not in inspect.getsource(benchmark)


def test_particle_mesh_roctx_loader_sets_checked_signatures(monkeypatch):
    benchmark = _load_particle_mesh_benchmark_module()
    library = _FakeRoctx()
    monkeypatch.setattr(benchmark, "_roctx_library_candidates", lambda: ["fake.so"])
    monkeypatch.setattr(benchmark.ctypes, "CDLL", lambda *_args, **_kwargs: library)

    loaded, path = benchmark._load_roctx()

    assert loaded is library
    assert path == "fake.so"
    assert library.roctxProfilerPause.restype is ctypes.c_int
    assert library.roctxProfilerResume.restype is ctypes.c_int
    assert library.roctxRangePushA.restype is ctypes.c_int
    assert library.roctxRangePop.restype is ctypes.c_int


def test_particle_mesh_roctx_successful_lifecycle():
    benchmark = _load_particle_mesh_benchmark_module()
    roctx = _FakeRoctx()
    initialize = getattr(benchmark, "_initialize_roctx_profile", None)
    assert callable(initialize)

    initialize(roctx)
    samples = benchmark._time_samples(
        lambda: None, 1, runtime=None, runtime_kind="none", roctx=roctx
    )

    assert len(samples) == 1
    assert len(roctx.roctxProfilerPause.calls) == 2
    assert len(roctx.roctxProfilerResume.calls) == 1
    assert len(roctx.roctxRangePushA.calls) == 1
    assert len(roctx.roctxRangePop.calls) == 1


@pytest.mark.parametrize(
    ("failure", "kwargs", "message"),
    [
        ("initial_pause", {"pause": (3,)}, "initial pause.*3"),
        ("resume", {"resume": (4,)}, "resume.*4"),
        ("push", {"push": (-2,)}, "roctxRangePushA.*-2"),
        (
            "push_exception",
            {"push": (OSError("push exploded"),)},
            "roctxRangePushA.*push exploded",
        ),
        ("pop", {"pop": (-3,)}, "roctxRangePop.*-3"),
        ("final_pause", {"pause": (0, 5)}, "final pause.*5"),
    ],
)
def test_particle_mesh_roctx_rejects_each_lifecycle_failure(failure, kwargs, message):
    benchmark = _load_particle_mesh_benchmark_module()
    roctx = _FakeRoctx(**kwargs)
    initialize = getattr(benchmark, "_initialize_roctx_profile", None)
    assert callable(initialize)
    if failure == "initial_pause":
        with pytest.raises(RuntimeError, match=message):
            initialize(roctx)
        return

    initialize(roctx)
    with pytest.raises(RuntimeError, match=message):
        benchmark._time_samples(
            lambda: None, 1, runtime=None, runtime_kind="none", roctx=roctx
        )
    if failure in {"push", "push_exception", "pop"}:
        assert len(roctx.roctxProfilerPause.calls) == 2


def test_particle_mesh_benchmark_compares_complete_maceles_time():
    benchmark = _load_particle_mesh_benchmark_module()
    explicit = benchmark.example_record_for_testing(
        algorithm="explicit_sum", median_ns=1_600_000
    )
    particle_mesh = benchmark.example_record_for_testing(
        algorithm="particle_mesh", median_ns=1_200_000
    )

    summary = benchmark.compare_records([explicit, particle_mesh])

    assert summary["explicit_sum_us_per_atom"] == pytest.approx(16.0)
    assert summary["particle_mesh_us_per_atom"] == pytest.approx(12.0)
    assert summary["particle_mesh_speedup"] == pytest.approx(4.0 / 3.0)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("structure_sha256", "2" * 64),
        ("mace_les_parameters_sha256", "2" * 64),
        (
            "source_provenance",
            {
                "source_commit": "different",
                "source_dirty": False,
                "source_content_sha256": "2" * 64,
                "native_source_content_sha256": "2" * 64,
            },
        ),
        ("interpreter_path", "/different/python"),
        ("cpu_affinity", [3]),
        ("synchronization", {"backend": "different", "runtime_library": None}),
    ],
)
def test_particle_mesh_benchmark_rejects_comparison_identity_mismatch(
    field, replacement
):
    benchmark = _load_particle_mesh_benchmark_module()
    explicit = benchmark.example_record_for_testing(
        algorithm="explicit_sum", median_ns=1_600_000
    )
    particle_mesh = benchmark.example_record_for_testing(
        algorithm="particle_mesh", median_ns=1_200_000
    )
    for record in (explicit, particle_mesh):
        record["structure_sha256"] = "1" * 64
    explicit[field] = replacement

    with pytest.raises(ValueError, match=field):
        benchmark.compare_records([explicit, particle_mesh])


def test_particle_mesh_benchmark_reports_particle_mesh_slowdown():
    benchmark = _load_particle_mesh_benchmark_module()
    explicit = benchmark.example_record_for_testing(
        algorithm="explicit_sum", median_ns=990_000
    )
    particle_mesh = benchmark.example_record_for_testing(
        algorithm="particle_mesh", median_ns=1_100_000
    )

    summary = benchmark.compare_records([explicit, particle_mesh])

    assert summary["explicit_sum_us_per_atom"] == pytest.approx(9.9)
    assert summary["particle_mesh_us_per_atom"] == pytest.approx(11.0)
    assert summary["particle_mesh_speedup"] == pytest.approx(0.9)


@pytest.fixture
def particle_mesh_available():
    capability = native_symmetrix._particle_mesh_capability()
    if not capability["available"]:
        pytest.skip(capability["reason"])


@pytest.fixture(scope="module")
def mace_les_model(tmp_path_factory):
    contract = json.loads(
        (CONTRACT_ROOT / "jit_r1_fixture_c4_e3_l0_contract.json").read_text()
    )
    model = add_standard_base_contracts(
        foundation_r1_test_model(contract, atomic_numbers=(7, 13))
    )
    model["model_type"] = "MACELES"
    model["les"] = {
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
        "mace_num_channels": 4,
        "mace_contract_fingerprints": {
            "R0": model["execution_contracts"]["R0"]["fingerprint"],
            "M0": model["execution_contracts"]["M0"]["semantic_fingerprint"],
            "R1": model["execution_contracts"]["R1"]["fingerprint"],
        },
        "readout_1_weights": [0.3, -0.2, 0.5, 0.1],
        "readout_2_hidden_size": 3,
        "readout_2_weights_1": [
            0.1,
            -0.2,
            0.3,
            0.4,
            -0.3,
            0.2,
            -0.1,
            0.5,
            0.25,
            0.15,
            -0.35,
            0.45,
        ],
        "readout_2_weights_2": [0.7, -0.4, 0.2],
        "readout_2_scale_factor": 1.5,
    }
    path = tmp_path_factory.mktemp("mace-les-kokkos") / "model.json"
    path.write_text(json.dumps(model, separators=(",", ":")))
    return path


@pytest.fixture(scope="module")
def official_mace_les_model(tmp_path_factory):
    torch = pytest.importorskip("torch", reason="official MACELES parity needs torch")
    pytest.importorskip("les", reason="official MACELES parity needs the LES package")
    try:
        # e3nn's installed constants.pt contains Python slice objects. Keep
        # PyTorch's restricted loader enabled and allow only that packaged type.
        with torch.serialization.safe_globals([slice]):
            from e3nn import o3
            from mace import modules
            from mace.modules.extensions import MACELES
            from symmetrix.extract_mace_data import extract_mace_data
    except (AttributeError, ImportError) as exc:
        pytest.skip(f"official MACELES parity dependencies are unavailable: {exc}")

    les_arguments = {
        "use_atomwise": False,
        "use_fixed_atomic_charges": False,
        "use_atomic_alpha": False,
        "use_epsilon_r_scaling": False,
        "sigma": 1.0,
        "dl": 2.0,
        "remove_self_interaction": True,
        "output_scale": 0.25,
    }
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        with torch.random.fork_rng():
            torch.manual_seed(123)
            model = MACELES(
                les_arguments=les_arguments,
                r_max=3.5,
                num_bessel=4,
                num_polynomial_cutoff=3,
                max_ell=3,
                interaction_cls=modules.interaction_classes[
                    "RealAgnosticResidualInteractionBlock"
                ],
                interaction_cls_first=modules.interaction_classes[
                    "RealAgnosticInteractionBlock"
                ],
                num_interactions=2,
                num_elements=2,
                hidden_irreps=o3.Irreps("4x0e + 4x1o"),
                MLP_irreps=o3.Irreps("3x0e"),
                atomic_energies=np.array([-1.2, -2.3]),
                avg_num_neighbors=3.0,
                atomic_numbers=[7, 13],
                correlation=3,
                gate=modules.gate_dict["silu"],
                pair_repulsion=False,
                distance_transform="None",
                radial_type="bessel",
                use_reduced_cg=False,
                atomic_inter_scale=1.3,
                atomic_inter_shift=0.0,
            ).to(torch.float64)
    finally:
        torch.set_default_dtype(previous_dtype)
    if not hasattr(model, "les_output_scale"):
        pytest.skip("installed MACELES predates scaled latent-charge support")
    model.eval()

    output = tmp_path_factory.mktemp("official-mace-les")
    checkpoint = output / "model.model"
    checkpoint_data = output / "model.json"
    torch.save(model, checkpoint)
    extracted = extract_mace_data(
        checkpoint,
        species=[7, 13],
        num_spline_points=128,
        radial_format="compact",
    )
    checkpoint_data.write_text(json.dumps(extracted, separators=(",", ":")))
    assert extracted["les"]["output_scale"] == pytest.approx(0.25)
    return model, checkpoint_data


def _atoms(*, periodic=False):
    return Atoms(
        ["N", "Al", "N"],
        positions=[[0.1, 0.2, 0.3], [1.4, 0.4, 0.2], [0.5, 1.6, 0.7]],
        cell=[[6.2, 0.2, 0.1], [0.0, 6.0, 0.3], [0.1, 0.0, 5.8]],
        pbc=periodic,
    )


class _NativeMACEInputBuilder:
    def __init__(self, model_path):
        model = json.loads(Path(model_path).read_text())
        self.cutoff = model["r_cut"]
        self.type_by_atomic_number = {
            atomic_number: index
            for index, atomic_number in enumerate(model["atomic_numbers"])
        }

    def _mace_inputs(self, atoms):
        receivers, sources, radii, xyz = neighbor_list("ijdD", atoms, self.cutoff)
        atomic_numbers = atoms.get_atomic_numbers()
        node_types = np.asarray(
            [self.type_by_atomic_number[value] for value in atomic_numbers],
            dtype=np.int32,
        )
        num_neigh = np.bincount(receivers, minlength=len(atoms)).astype(np.int32)
        neigh_types = node_types[sources]
        return (
            len(atoms),
            node_types,
            num_neigh,
            sources,
            neigh_types,
            xyz,
            radii,
            receivers,
        )


def _new_evaluator(model_path, class_name="MACELESKokkos"):
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    evaluator = getattr(native_symmetrix, class_name)(str(model_path))
    evaluator.set_streamed_edges("generic")
    return evaluator


def test_base_native_evaluator_rejects_maceles(mace_les_model):
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    try:
        evaluator = native_symmetrix.MACEKokkos(str(mace_les_model))
    except ValueError as error:
        assert "use MACELESKokkos" in str(error)
    else:
        del evaluator
        gc.collect()
        pytest.fail("MACEKokkos accepted a MACELES document")


def test_maceles_native_evaluator_rejects_mace(mace_les_model, tmp_path):
    data = json.loads(Path(mace_les_model).read_text())
    data["model_type"] = "MACE"
    data.pop("les")
    path = tmp_path / "mace.json"
    path.write_text(json.dumps(data, separators=(",", ":")))

    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    try:
        evaluator = native_symmetrix.MACELESKokkos(str(path))
    except ValueError as error:
        assert "use MACEKokkos" in str(error)
    else:
        del evaluator
        gc.collect()
        pytest.fail("MACELESKokkos accepted an ordinary MACE document")


@pytest.mark.parametrize(
    ("model_type", "class_name"),
    [("MACE", "MACEKokkos"), ("MACELES", "MACELESKokkos")],
)
def test_native_evaluators_preserve_multihead_metadata(
    mace_les_model, tmp_path, model_type, class_name
):
    data = json.loads(Path(mace_les_model).read_text())
    data["model_type"] = model_type
    head_fields = [
        "atomic_energies",
        "readout_1_weights",
        "readout_2_weights_1",
        "readout_2_weights_2",
    ]
    if model_type == "MACE":
        data.pop("les")
    else:
        head_fields.append("les")
    payload = {name: data[name] for name in head_fields}
    data["prediction_heads"] = {
        "schema_version": 1,
        "names": ["first", "second"],
        "default": "second",
        "parameters": {
            "first": payload,
            "second": json.loads(json.dumps(payload)),
        },
    }
    path = tmp_path / f"multihead-{model_type.lower()}.json"
    path.write_text(json.dumps(data, separators=(",", ":")))

    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    evaluator = getattr(native_symmetrix, class_name)(str(path), "first")
    try:
        assert evaluator.selected_head == "first"
        assert evaluator.available_heads == ["first", "second"]
        les_surface = (
            "is_mace_les",
            "_set_les_geometry",
            "latent_charges",
            "les_energy",
            "les_diagnostics",
            "les_monopole_diagnostics",
        )
        if model_type == "MACELES":
            for attribute in les_surface:
                assert attribute in type(evaluator).__dict__, attribute
        else:
            for attribute in les_surface:
                assert attribute not in type(evaluator).__dict__, attribute
    finally:
        del evaluator
        gc.collect()


def test_model_load_json_parse_phase_includes_document_parsing(
    mace_les_model, tmp_path
):
    data = json.loads(Path(mace_les_model).read_text())
    data["parse_timing_padding"] = "x" * 8_000_000
    path = tmp_path / "large-maceles.json"
    path.write_text(json.dumps(data, separators=(",", ":")))

    evaluator = _new_evaluator(path)
    try:
        assert evaluator.model_load_phase_ms["json_parse"] >= 1.0
    finally:
        del evaluator
        gc.collect()


@pytest.mark.parametrize("contract_name", ["R0", "M0", "R1"])
def test_maceles_load_rejects_tampered_mace_contract_fingerprint(
    mace_les_model, tmp_path, contract_name
):
    model = json.loads(Path(mace_les_model).read_text())
    model["les"]["mace_contract_fingerprints"][contract_name] = "sha256:" + "0" * 64
    path = tmp_path / f"tampered-{contract_name}.json"
    path.write_text(json.dumps(model, separators=(",", ":")))

    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    with pytest.raises(ValueError, match=rf"MACE {contract_name} contract fingerprint"):
        native_symmetrix.MACELESKokkos(str(path))


def test_native_maceles_requires_les_parameters(mace_les_model, tmp_path):
    model = json.loads(Path(mace_les_model).read_text())
    model.pop("les")
    path = tmp_path / "maceles-without-les.json"
    path.write_text(json.dumps(model, separators=(",", ":")))

    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    with pytest.raises(ValueError, match=r"MACELES.*les parameters"):
        native_symmetrix.MACELESKokkos(str(path))


def test_native_mace_rejects_les_parameters(mace_les_model, tmp_path):
    model = json.loads(Path(mace_les_model).read_text())
    model["model_type"] = "MACE"
    path = tmp_path / "mace-with-les.json"
    path.write_text(json.dumps(model, separators=(",", ":")))

    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    with pytest.raises(ValueError, match=r"MACELES.*les parameters"):
        native_symmetrix.MACEKokkos(str(path))


def test_native_maceles_rejects_unsupported_topology(mace_les_model, tmp_path):
    model = json.loads(Path(mace_les_model).read_text())
    model["num_interactions"] = 1
    model["single_layer_readout"] = True
    model["les"]["mace_num_interactions"] = 1
    path = tmp_path / "single-layer-maceles.json"
    path.write_text(json.dumps(model, separators=(",", ":")))

    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    with pytest.raises(ValueError, match=r"MACELES.*two-interaction"):
        native_symmetrix.MACELESKokkos(str(path))


def test_native_les_diagnostics_report_loaded_schema_version(mace_les_model):
    evaluator = _new_evaluator(mace_les_model)

    assert evaluator.les_diagnostics["schema_version"] == 1
    assert evaluator.les_diagnostics["schema"] == "symmetrix.model.mace_les"
    assert evaluator.is_mace_les
    del evaluator
    gc.collect()


def test_native_periodic_les_rejects_extreme_finite_reciprocal_bound(
    mace_les_model, tmp_path
):
    model = json.loads(Path(mace_les_model).read_text())
    model["les"]["ewald"]["dl"] = np.nextafter(0.0, 1.0)
    path = tmp_path / "extreme-dl.json"
    path.write_text(json.dumps(model, separators=(",", ":")))
    evaluator = _new_evaluator(path)
    calculator = Symmetrix(
        path,
        dtype="float64",
        streamed_edges="reference",
        neighbor_skin=0.0,
    )

    with pytest.raises(
        ValueError,
        match=r"LES reciprocal bound.*increase dl or reduce the periodic cell",
    ):
        _evaluate(evaluator, calculator, _atoms(periodic=True))
    del evaluator, calculator
    gc.collect()


def _evaluate(evaluator, calculator, atoms, *, collect_stress=False):
    inputs = calculator._mace_inputs(atoms)
    num_nodes, node_types, num_neigh, sources, neigh_types, xyz, radii, receivers = (
        inputs
    )
    evaluator._set_les_geometry(atoms.positions, atoms.cell.array, atoms.pbc)
    evaluator.compute_node_energies_forces(
        num_nodes,
        node_types,
        num_neigh,
        sources,
        neigh_types,
        np.asarray(xyz).reshape(-1),
        radii,
    )
    forces = np.asarray(
        evaluator._reduce_atom_forces(num_nodes, receivers, sources), dtype=float
    )
    result = {
        "energy": np.asarray(evaluator.node_energies, dtype=float).sum(),
        "node_energies": np.asarray(evaluator.node_energies, dtype=float),
        "forces": forces,
        "charges": np.asarray(evaluator.latent_charges, dtype=float),
        "les_energy": evaluator.les_energy,
    }
    if collect_stress:
        result["stress"] = np.asarray(
            evaluator._reduce_stress(atoms.get_volume(), xyz), dtype=float
        )
    return result


def _direct_les_result(model_path, atoms, monkeypatch, tmp_path, *, particle_mesh):
    tmp_path.mkdir(parents=True, exist_ok=True)
    tmp_path.chmod(0o700)
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))
    monkeypatch.setenv("SYMMETRIX_JIT_POLICY", "required")
    calculator = Symmetrix(
        model_path,
        dtype="float64",
        streamed_edges="direct",
        execution_profile="speed",
        neighbor_skin=0.0,
    )
    if particle_mesh:
        calculator.evaluator._configure_les_particle_mesh_for_testing(24, 24, 24, 4)
    calculator.calculate(atoms, properties=["energy", "energies", "forces", "stress"])
    evaluator = calculator.evaluator
    return calculator, {
        "energy": float(calculator.results["energy"]),
        "forces": calculator.results["forces"].copy(),
        "stress": calculator.results["stress"].copy(),
        "atomic_energies": calculator.results["energies"].copy(),
        "latent_charges": calculator.results["latent_charges"].copy(),
        "charge_adjoints": np.asarray(
            evaluator._les_charge_adjoints_for_testing, dtype=float
        ).copy(),
    }


def test_particle_mesh_drivers_read_algorithm_from_full_calculator_diagnostics(
    mace_les_model, monkeypatch, tmp_path
):
    calculator, _ = _direct_les_result(
        mace_les_model,
        _atoms(periodic=True),
        monkeypatch,
        tmp_path / "explicit-algorithm",
        particle_mesh=False,
    )
    benchmark = _load_particle_mesh_benchmark_module()
    qualification = _load_particle_mesh_qualification_module()
    try:
        assert benchmark._selected_periodic_algorithm(calculator) == "explicit_sum"
        assert qualification._selected_periodic_algorithm(calculator) == "explicit_sum"
    finally:
        del calculator
        gc.collect()


def test_direct_mace_particle_mesh_matches_explicit_les_end_to_end(
    mace_les_model, monkeypatch, tmp_path, particle_mesh_available
):
    atoms = _atoms(periodic=True)
    explicit, expected = _direct_les_result(
        mace_les_model, atoms, monkeypatch, tmp_path / "explicit", particle_mesh=False
    )
    particle_mesh, actual = _direct_les_result(
        mace_les_model,
        atoms,
        monkeypatch,
        tmp_path / "particle-mesh",
        particle_mesh=True,
    )

    np.testing.assert_allclose(
        actual["latent_charges"], expected["latent_charges"], rtol=0.0, atol=1.0e-13
    )
    for name in (
        "energy",
        "forces",
        "stress",
        "atomic_energies",
        "charge_adjoints",
    ):
        np.testing.assert_allclose(
            actual[name], expected[name], rtol=3.0e-4, atol=8.0e-5
        )
    assert (
        particle_mesh.evaluator._les_particle_mesh_diagnostics_for_testing[
            "periodic_algorithm"
        ]
        == "particle_mesh"
    )
    assert particle_mesh.evaluator.factorized_fallback_evaluation_count == 0
    del explicit, particle_mesh
    gc.collect()


def test_direct_mace_particle_mesh_force_and_stress_finite_difference_include_dq_dr(
    mace_les_model, monkeypatch, tmp_path, particle_mesh_available
):
    model = json.loads(Path(mace_les_model).read_text())
    # Amplify the fitted LES term so its stress-chain contribution is resolved
    # beyond the same finite-difference envelope used for the accepted result.
    model["les"]["output_scale"] = 4.0
    sensitive_model = tmp_path / "stress-sensitive-model.json"
    sensitive_model.write_text(json.dumps(model, separators=(",", ":")))
    jit_cache = tmp_path / "jit-cache"
    jit_cache.mkdir(mode=0o700)
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(jit_cache))
    monkeypatch.setenv("SYMMETRIX_JIT_POLICY", "required")
    calculator = Symmetrix(
        sensitive_model,
        dtype="float64",
        streamed_edges="direct",
        execution_profile="speed",
        neighbor_skin=0.0,
    )
    calculator.evaluator._configure_les_particle_mesh_for_testing(32, 32, 32, 6)
    atoms = _atoms(periodic=True)
    step = 2.0e-5

    def energy(candidate):
        calculator.calculate(
            candidate, properties=["energy", "energies", "forces", "stress"]
        )
        return float(calculator.results["energy"])

    force_energies = []
    for sign in (1.0, -1.0):
        displaced = atoms.copy()
        displaced.positions[0, 0] += sign * step
        force_energies.append(energy(displaced))
    numeric_force = -(force_energies[0] - force_energies[1]) / (2.0 * step)

    strain_energies = []
    for sign in (1.0, -1.0):
        deformation = np.eye(3)
        deformation[0, 0] += sign * step
        strained = atoms.copy()
        strained.set_cell(atoms.cell.array @ deformation, scale_atoms=True)
        strain_energies.append(energy(strained))
    numeric_stress = (strain_energies[0] - strain_energies[1]) / (
        2.0 * step * atoms.get_volume()
    )

    energy(atoms)
    analytic_force = float(calculator.results["forces"][0, 0])
    analytic_stress = float(calculator.results["stress"][0])
    force_tolerance = {"rel": 6.0e-5, "abs": 6.0e-8}
    stress_tolerance = {"rel": 6.0e-5, "abs": 6.0e-9}
    assert analytic_force == pytest.approx(numeric_force, **force_tolerance)
    assert analytic_stress == pytest.approx(numeric_stress, **stress_tolerance)
    assert calculator.evaluator.factorized_fallback_evaluation_count == 0

    calculator.evaluator._set_les_charge_adjoint_accumulation_for_testing(False)
    energy(atoms)
    missing_chain_force = float(calculator.results["forces"][0, 0])
    missing_chain_stress = float(calculator.results["stress"][0])
    assert missing_chain_force != pytest.approx(numeric_force, **force_tolerance)
    assert missing_chain_stress != pytest.approx(numeric_stress, **stress_tolerance)

    calculator.evaluator._set_les_charge_adjoint_accumulation_for_testing(True)
    energy(atoms)
    assert calculator.results["forces"][0, 0] == pytest.approx(
        numeric_force, **force_tolerance
    )
    assert calculator.results["stress"][0] == pytest.approx(
        numeric_stress, **stress_tolerance
    )
    del energy
    calculator = None
    gc.collect()


def _compose_particle_mesh_test_state(evaluator, atoms, h1_scalars, h2_scalars):
    nodes = len(atoms)
    evaluator._set_les_geometry(atoms.positions, atoms.cell.array, atoms.pbc)
    evaluator._prepare_les_charge_readouts_for_testing(h1_scalars, h2_scalars)
    evaluator.node_energies = np.zeros(nodes, dtype=np.float64)
    evaluator.H1_adj = np.zeros_like(np.asarray(evaluator.H1))
    evaluator.H2_adj = np.zeros_like(np.asarray(evaluator.H2))
    return evaluator._compose_prepared_les_for_testing(nodes)


def test_maceles_term_is_automatic_and_matches_reference_and_direct(
    mace_les_model,
):
    atoms = _atoms()
    h1 = np.array(
        [[0.7, -0.1, 0.3, 0.5], [-0.4, 0.8, 0.2, -0.6], [0.1, 0.9, -0.7, 0.4]],
        dtype=np.float64,
    )
    h2 = np.array(
        [[0.2, -0.5, 0.9, 0.1], [0.6, 0.3, -0.8, 0.7], [-0.2, 0.4, 0.5, -0.9]],
        dtype=np.float64,
    )
    reference = _new_evaluator(mace_les_model)
    direct = native_symmetrix.MACELESKokkos(str(mace_les_model))
    direct.set_streamed_edges("direct")

    expected = _compose_particle_mesh_test_state(reference, atoms, h1, h2)
    actual = _compose_particle_mesh_test_state(direct, atoms, h1, h2)

    assert np.any(np.asarray(expected["latent_charges"]) != 0.0)
    assert expected["les_energy"] != 0.0
    np.testing.assert_allclose(
        actual["latent_charges"], expected["latent_charges"], rtol=0.0, atol=1.0e-14
    )
    assert actual["les_energy"] == pytest.approx(
        expected["les_energy"], rel=0.0, abs=1.0e-14
    )
    np.testing.assert_allclose(
        actual["node_energies"], expected["node_energies"], rtol=0.0, atol=1.0e-14
    )
    del reference, direct
    gc.collect()


def test_mace_particle_mesh_workspace_lifecycle_tracks_geometry_mesh_and_mode(
    mace_les_model,
    particle_mesh_available,
):
    atoms = _atoms(periodic=True)
    h1 = np.array(
        [[0.7, -0.1, 0.3, 0.5], [-0.4, 0.8, 0.2, -0.6], [0.1, 0.9, -0.7, 0.4]],
        dtype=np.float64,
    )
    h2 = np.array(
        [[0.2, -0.5, 0.9, 0.1], [0.6, 0.3, -0.8, 0.7], [-0.2, 0.4, 0.5, -0.9]],
        dtype=np.float64,
    )
    evaluator = _new_evaluator(mace_les_model)
    evaluator.set_streamed_edges("direct")
    evaluator._configure_les_particle_mesh_for_testing(24, 24, 24, 4)

    _compose_particle_mesh_test_state(evaluator, atoms, h1, h2)
    initial = evaluator._les_particle_mesh_diagnostics_for_testing
    with pytest.raises(ValueError, match="active (Miller sphere|reciprocal vector)"):
        evaluator._set_les_geometry(atoms.positions, atoms.cell.array * 10.0, atoms.pbc)
    with pytest.raises(ValueError, match="fresh LES geometry"):
        evaluator._prepare_les_charge_readouts_for_testing(h1, h2)
    _compose_particle_mesh_test_state(evaluator, atoms, h1, h2)
    recovered = evaluator._les_particle_mesh_diagnostics_for_testing
    assert recovered["owner_generation"] == initial["owner_generation"]
    evaluator._configure_les_particle_mesh_for_testing(24, 24, 24, 4)
    _compose_particle_mesh_test_state(evaluator, atoms, h1, h2)
    same_request = evaluator._les_particle_mesh_diagnostics_for_testing
    moved = atoms.copy()
    moved.positions[1, 0] += 0.01
    _compose_particle_mesh_test_state(evaluator, moved, h1, h2)
    position_only = evaluator._les_particle_mesh_diagnostics_for_testing
    changed_cell = moved.copy()
    changed_cell.set_cell(moved.cell.array * 1.01, scale_atoms=True)
    _compose_particle_mesh_test_state(evaluator, changed_cell, h1, h2)
    cell_only = evaluator._les_particle_mesh_diagnostics_for_testing

    assert same_request["owner_generation"] == initial["owner_generation"]
    assert same_request["allocation_generation"] == initial["allocation_generation"]
    assert same_request["forward_plan_builds"] == initial["forward_plan_builds"]
    assert same_request["inverse_plan_builds"] == initial["inverse_plan_builds"]
    assert same_request["reciprocal_preparations"] == initial["reciprocal_preparations"]
    assert same_request["stencil_preparations"] == recovered["stencil_preparations"] + 1
    assert position_only["owner_generation"] == initial["owner_generation"]
    assert position_only["allocation_generation"] == initial["allocation_generation"]
    assert position_only["forward_plan_builds"] == initial["forward_plan_builds"]
    assert position_only["inverse_plan_builds"] == initial["inverse_plan_builds"]
    assert (
        position_only["reciprocal_preparations"] == initial["reciprocal_preparations"]
    )
    assert (
        position_only["stencil_preparations"]
        == same_request["stencil_preparations"] + 1
    )
    assert cell_only["owner_generation"] == initial["owner_generation"]
    assert cell_only["forward_plan_builds"] == initial["forward_plan_builds"]
    assert cell_only["inverse_plan_builds"] == initial["inverse_plan_builds"]
    assert (
        cell_only["reciprocal_preparations"] == initial["reciprocal_preparations"] + 1
    )
    assert cell_only["table_rebuilds"] == position_only["table_rebuilds"] + 1

    batch_size = 256
    fractional = np.column_stack(
        (
            (np.arange(batch_size) * 17 % 257) / 257.0,
            (np.arange(batch_size) * 43 % 263) / 263.0,
            (np.arange(batch_size) * 71 % 269) / 269.0,
        )
    )
    expanded = Atoms(
        ["N"] * batch_size,
        positions=fractional @ changed_cell.cell.array,
        cell=changed_cell.cell,
        pbc=True,
    )
    expanded_h1 = np.tile(h1[0], (batch_size, 1))
    expanded_h2 = np.tile(h2[0], (batch_size, 1))
    _compose_particle_mesh_test_state(evaluator, expanded, expanded_h1, expanded_h2)
    larger_batch = evaluator._les_particle_mesh_diagnostics_for_testing
    assert larger_batch["owner_generation"] == initial["owner_generation"]
    assert larger_batch["allocation_generation"] > cell_only["allocation_generation"]
    assert larger_batch["forward_plan_builds"] == initial["forward_plan_builds"]
    assert larger_batch["inverse_plan_builds"] == initial["inverse_plan_builds"]
    assert (
        larger_batch["reciprocal_preparations"]
        == cell_only["reciprocal_preparations"] + 1
    )
    assert larger_batch["stencil_preparations"] == cell_only["stencil_preparations"] + 1

    evaluator._configure_les_particle_mesh_for_testing(25, 25, 25, 4)
    _compose_particle_mesh_test_state(evaluator, changed_cell, h1, h2)
    remeshed = evaluator._les_particle_mesh_diagnostics_for_testing
    assert remeshed["owner_generation"] == initial["owner_generation"] + 1
    assert remeshed["allocation_generation"] > 0
    assert remeshed["forward_plan_builds"] == 1
    assert remeshed["inverse_plan_builds"] == 1
    assert remeshed["reciprocal_preparations"] == 1

    evaluator._use_explicit_les_periodic_for_testing()
    explicit = evaluator._les_particle_mesh_diagnostics_for_testing
    assert explicit["periodic_algorithm"] == "explicit_sum"
    evaluator._configure_les_particle_mesh_for_testing(24, 24, 24, 4)
    evaluator.set_streamed_edges("generic")
    _compose_particle_mesh_test_state(evaluator, atoms, h1, h2)
    generic = evaluator._les_particle_mesh_diagnostics_for_testing
    evaluator.set_streamed_edges("direct")
    _compose_particle_mesh_test_state(evaluator, atoms, h1, h2)
    direct_again = evaluator._les_particle_mesh_diagnostics_for_testing
    assert generic["owner_generation"] > explicit["owner_generation"]
    assert direct_again["owner_generation"] == generic["owner_generation"] + 1
    assert direct_again["execution_instance_id"] == initial["execution_instance_id"]
    del evaluator
    gc.collect()


def test_direct_mace_particle_mesh_fences_owner_before_immediate_transitions(
    mace_les_model,
    particle_mesh_available,
):
    atoms = _atoms(periodic=True)
    h1 = np.zeros((len(atoms), 4), dtype=np.float64)
    h2 = np.zeros_like(h1)
    evaluator = _new_evaluator(mace_les_model)
    evaluator.set_streamed_edges("direct")
    evaluator._configure_les_particle_mesh_for_testing(24, 24, 24, 4)

    evaluator._set_les_geometry(atoms.positions, atoms.cell.array, atoms.pbc)
    evaluator._configure_les_particle_mesh_for_testing(24, 24, 24, 6)
    evaluator._set_les_geometry(atoms.positions, atoms.cell.array, atoms.pbc)
    reordered = evaluator._les_particle_mesh_diagnostics_for_testing
    assert reordered["owner_fences"] == 1
    assert reordered["owner_generation"] == 2
    assert reordered["allocation_generation"] > 0
    assert reordered["forward_plan_builds"] == 1
    assert reordered["inverse_plan_builds"] == 1
    evaluator._configure_les_particle_mesh_for_testing(25, 25, 25, 4)
    evaluator._set_les_geometry(atoms.positions, atoms.cell.array, atoms.pbc)
    evaluator._use_explicit_les_periodic_for_testing()
    evaluator._configure_les_particle_mesh_for_testing(24, 24, 24, 4)
    evaluator._set_les_geometry(atoms.positions, atoms.cell.array, atoms.pbc)
    evaluator.set_streamed_edges("generic")

    _compose_particle_mesh_test_state(evaluator, atoms, h1, h2)
    diagnostics = evaluator._les_particle_mesh_diagnostics_for_testing
    assert diagnostics["owner_fences"] == 4
    assert diagnostics["periodic_algorithm"] == "particle_mesh"
    del evaluator
    gc.collect()


def test_direct_mace_particle_mesh_fences_immediate_count_changes(
    mace_les_model,
    particle_mesh_available,
):
    atoms = _atoms(periodic=True)
    grown = atoms.copy()
    grown += Atoms("Al", positions=[[1.83, 1.24, 0.77]])
    shrunk = atoms[:2]
    evaluator = _new_evaluator(mace_les_model)
    evaluator.set_streamed_edges("direct")
    evaluator._configure_les_particle_mesh_for_testing(24, 24, 24, 4)

    evaluator._set_les_geometry(atoms.positions, atoms.cell.array, atoms.pbc)
    invalid_grown = grown.copy()
    invalid_grown.set_cell(grown.cell.array * 10.0, scale_atoms=False)
    with pytest.raises(ValueError, match="active (Miller sphere|reciprocal vector)"):
        evaluator._set_les_geometry(
            invalid_grown.positions, invalid_grown.cell.array, invalid_grown.pbc
        )
    evaluator._set_les_geometry(grown.positions, grown.cell.array, grown.pbc)
    evaluator._set_les_geometry(shrunk.positions, shrunk.cell.array, shrunk.pbc)

    diagnostics = evaluator._les_particle_mesh_diagnostics_for_testing
    assert diagnostics["particle_count_fences"] == 2
    h1 = np.zeros((len(shrunk), 4), dtype=np.float64)
    h2 = np.zeros_like(h1)
    result = _compose_particle_mesh_test_state(evaluator, shrunk, h1, h2)
    assert np.isfinite(result["les_energy"])
    del evaluator
    gc.collect()


def test_direct_mace_particle_mesh_invalidates_abandoned_count_fence(
    mace_les_model,
    particle_mesh_available,
):
    atoms = _atoms(periodic=True)
    grown = atoms.copy()
    grown += Atoms("Al", positions=[[1.83, 1.24, 0.77]])
    invalid_grown = grown.copy()
    invalid_grown.set_cell(grown.cell.array * 10.0, scale_atoms=False)
    evaluator = _new_evaluator(mace_les_model)
    evaluator.set_streamed_edges("direct")
    evaluator._configure_les_particle_mesh_for_testing(24, 24, 24, 4)

    evaluator._set_les_geometry(atoms.positions, atoms.cell.array, atoms.pbc)
    with pytest.raises(ValueError, match="active (Miller sphere|reciprocal vector)"):
        evaluator._set_les_geometry(
            invalid_grown.positions, invalid_grown.cell.array, invalid_grown.pbc
        )
    evaluator._set_les_geometry(atoms.positions, atoms.cell.array, atoms.pbc)
    evaluator._set_les_geometry(grown.positions, grown.cell.array, grown.pbc)

    diagnostics = evaluator._les_particle_mesh_diagnostics_for_testing
    assert diagnostics["particle_count_fences"] == 2
    del evaluator
    gc.collect()


def test_direct_mace_particle_mesh_invalidates_count_fence_for_nonperiodic_work(
    mace_les_model,
    particle_mesh_available,
):
    atoms = _atoms(periodic=True)
    grown = atoms.copy()
    grown += Atoms("Al", positions=[[1.83, 1.24, 0.77]])
    invalid_grown = grown.copy()
    invalid_grown.set_cell(grown.cell.array * 10.0, scale_atoms=False)
    nonperiodic_grown = grown.copy()
    nonperiodic_grown.pbc = False
    evaluator = _new_evaluator(mace_les_model)
    evaluator.set_streamed_edges("direct")
    evaluator._configure_les_particle_mesh_for_testing(24, 24, 24, 4)

    evaluator._set_les_geometry(atoms.positions, atoms.cell.array, atoms.pbc)
    with pytest.raises(ValueError, match="active (Miller sphere|reciprocal vector)"):
        evaluator._set_les_geometry(
            invalid_grown.positions, invalid_grown.cell.array, invalid_grown.pbc
        )
    h1 = np.zeros((len(grown), 4), dtype=np.float64)
    h2 = np.zeros_like(h1)
    _compose_particle_mesh_test_state(evaluator, nonperiodic_grown, h1, h2)
    evaluator._set_les_geometry(grown.positions, grown.cell.array, grown.pbc)

    diagnostics = evaluator._les_particle_mesh_diagnostics_for_testing
    assert diagnostics["particle_count_fences"] == 2
    del evaluator
    gc.collect()


def test_direct_mace_particle_mesh_fences_nonperiodic_count_growth_before_copy(
    mace_les_model,
    particle_mesh_available,
):
    atoms = _atoms(periodic=True)
    nonperiodic_grown = atoms.copy()
    nonperiodic_grown += Atoms("Al", positions=[[1.83, 1.24, 0.77]])
    nonperiodic_grown.pbc = False
    evaluator = _new_evaluator(mace_les_model)
    evaluator.set_streamed_edges("direct")
    evaluator._configure_les_particle_mesh_for_testing(24, 24, 24, 4)

    evaluator._set_les_geometry(atoms.positions, atoms.cell.array, atoms.pbc)
    evaluator._set_les_geometry(
        nonperiodic_grown.positions,
        nonperiodic_grown.cell.array,
        nonperiodic_grown.pbc,
    )

    particle_count_fences = evaluator._les_particle_mesh_diagnostics_for_testing[
        "particle_count_fences"
    ]
    del evaluator
    gc.collect()
    assert particle_count_fences == 1


def test_direct_mace_particle_mesh_fences_abandoned_initial_geometry_copy(
    mace_les_model,
    particle_mesh_available,
):
    atoms = _atoms(periodic=True)
    invalid = atoms.copy()
    invalid.set_cell(atoms.cell.array * 10.0, scale_atoms=False)
    grown = atoms.copy()
    grown += Atoms("Al", positions=[[1.83, 1.24, 0.77]])
    evaluator = _new_evaluator(mace_les_model)
    evaluator.set_streamed_edges("direct")
    evaluator._configure_les_particle_mesh_for_testing(24, 24, 24, 4)

    with pytest.raises(ValueError, match="active (Miller sphere|reciprocal vector)"):
        evaluator._set_les_geometry(invalid.positions, invalid.cell.array, invalid.pbc)
    evaluator._set_les_geometry(grown.positions, grown.cell.array, grown.pbc)

    particle_count_fences = evaluator._les_particle_mesh_diagnostics_for_testing[
        "particle_count_fences"
    ]
    del evaluator
    gc.collect()
    assert particle_count_fences == 1


def test_direct_mace_particle_mesh_fences_after_failed_initial_reserve(
    mace_les_model,
    particle_mesh_available,
):
    atoms = _atoms(periodic=True)
    invalid_grown = atoms.copy()
    invalid_grown += Atoms("Al", positions=[[1.83, 1.24, 0.77]])
    invalid_grown.set_cell(atoms.cell.array * 10.0, scale_atoms=False)
    larger = invalid_grown.copy()
    larger += Atoms("N", positions=[[0.42, 1.91, 1.36]])
    larger.set_cell(atoms.cell.array, scale_atoms=False)
    evaluator = _new_evaluator(mace_les_model)
    evaluator.set_streamed_edges("direct")
    evaluator._configure_les_particle_mesh_for_testing(24, 24, 24, 4)

    native_symmetrix._fourier_d3_arm_allocation_fault(0)
    try:
        with pytest.raises(RuntimeError, match="injected allocation fault"):
            evaluator._set_les_geometry(atoms.positions, atoms.cell.array, atoms.pbc)
    finally:
        native_symmetrix._fourier_d3_clear_allocation_fault()
    with pytest.raises(ValueError, match="active (Miller sphere|reciprocal vector)"):
        evaluator._set_les_geometry(
            invalid_grown.positions, invalid_grown.cell.array, invalid_grown.pbc
        )
    evaluator._set_les_geometry(larger.positions, larger.cell.array, larger.pbc)

    particle_count_fences = evaluator._les_particle_mesh_diagnostics_for_testing[
        "particle_count_fences"
    ]
    del evaluator
    gc.collect()
    assert particle_count_fences == 2


def test_direct_mace_particle_mesh_retains_unknown_state_across_nonperiodic_work(
    mace_les_model,
    particle_mesh_available,
):
    atoms = _atoms(periodic=True)
    nonperiodic_grown = atoms.copy()
    nonperiodic_grown += Atoms("Al", positions=[[1.83, 1.24, 0.77]])
    nonperiodic_grown.pbc = False
    periodic_larger = nonperiodic_grown.copy()
    periodic_larger += Atoms("N", positions=[[0.42, 1.91, 1.36]])
    periodic_larger.pbc = True
    evaluator = _new_evaluator(mace_les_model)
    evaluator.set_streamed_edges("direct")
    evaluator._configure_les_particle_mesh_for_testing(24, 24, 24, 4)

    native_symmetrix._fourier_d3_arm_allocation_fault(0)
    try:
        with pytest.raises(RuntimeError, match="injected allocation fault"):
            evaluator._set_les_geometry(atoms.positions, atoms.cell.array, atoms.pbc)
    finally:
        native_symmetrix._fourier_d3_clear_allocation_fault()
    h1 = np.zeros((len(nonperiodic_grown), 4), dtype=np.float64)
    h2 = np.zeros_like(h1)
    _compose_particle_mesh_test_state(evaluator, nonperiodic_grown, h1, h2)
    evaluator._set_les_geometry(
        periodic_larger.positions, periodic_larger.cell.array, periodic_larger.pbc
    )

    particle_count_fences = evaluator._les_particle_mesh_diagnostics_for_testing[
        "particle_count_fences"
    ]
    del evaluator
    gc.collect()
    assert particle_count_fences == 2


def _official_mace_result(model, atoms):
    import torch
    from mace import data, tools
    from mace.tools import torch_geometric

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        atomic_data = data.AtomicData.from_config(
            data.config_from_atoms(atoms),
            z_table=tools.AtomicNumberTable([7, 13]),
            cutoff=float(model.r_max),
        )
    finally:
        torch.set_default_dtype(previous_dtype)
    loader = torch_geometric.dataloader.DataLoader(
        [atomic_data], batch_size=1, shuffle=False
    )
    output = model(
        next(iter(loader)).to_dict(),
        training=False,
        compute_force=True,
        compute_stress=True,
    )
    return {
        "charges": output["latent_charges"].detach().cpu().numpy(),
        "les_energy": output["les_energy"].detach().cpu().item(),
        "energy": output["energy"].detach().cpu().item(),
        "forces": output["forces"].detach().cpu().numpy(),
        "stress": output["stress"].detach().cpu().numpy()[0],
    }


def test_les_geometry_is_validated_and_consumed(mace_les_model):
    evaluator = _new_evaluator(mace_les_model)
    atoms = _atoms()
    with pytest.raises(ValueError, match="all false or all true"):
        evaluator._set_les_geometry(
            atoms.positions, atoms.cell.array, [True, False, True]
        )
    with pytest.raises(ValueError, match="boolean values"):
        evaluator._set_les_geometry(atoms.positions, atoms.cell.array, [2, 2, 2])

    calculator = Symmetrix(
        mace_les_model,
        dtype="float64",
        streamed_edges="reference",
        neighbor_skin=0.0,
    )
    _evaluate(evaluator, calculator, atoms)
    inputs = calculator._mace_inputs(atoms)
    with pytest.raises(ValueError, match="fresh LES geometry"):
        evaluator.compute_node_energies_forces(
            *inputs[:5], np.asarray(inputs[5]).reshape(-1), inputs[6]
        )


def _evaluate_split_les_test_seam(
    model_path, atoms, h1_scalars, h2_scalars, *, overwrite_h2=False
):
    evaluator = _new_evaluator(model_path)
    try:
        evaluator._set_les_geometry(atoms.positions, atoms.cell.array, atoms.pbc)
        evaluator._prepare_les_charge_readouts_for_testing(h1_scalars, h2_scalars)
        evaluator.node_energies = np.zeros(len(atoms), dtype=np.float64)
        evaluator.H1_adj = np.zeros_like(np.asarray(evaluator.H1))
        evaluator.H2_adj = np.zeros_like(np.asarray(evaluator.H2))
        if overwrite_h2:
            evaluator.H2 = np.full_like(np.asarray(evaluator.H2), 17.0)
        return evaluator._compose_prepared_les_for_testing(len(atoms))
    finally:
        del evaluator
        gc.collect()


def test_les_split_state_survives_h2_overwrite(mace_les_model):
    atoms = _atoms(periodic=True)
    h1_scalars = np.array(
        [[0.7, -0.1, 0.3, 0.5], [-0.4, 0.8, 0.2, -0.6], [0.1, 0.9, -0.7, 0.4]],
        dtype=np.float64,
    )
    h2_scalars = np.array(
        [[0.2, -0.5, 0.9, 0.1], [0.6, 0.3, -0.8, 0.7], [-0.2, 0.4, 0.5, -0.9]],
        dtype=np.float64,
    )

    retained = _evaluate_split_les_test_seam(
        mace_les_model, atoms, h1_scalars, h2_scalars
    )
    reused = _evaluate_split_les_test_seam(
        mace_les_model,
        atoms,
        h1_scalars,
        h2_scalars,
        overwrite_h2=True,
    )
    overwritten_before_prepare = _evaluate_split_les_test_seam(
        mace_les_model,
        atoms,
        h1_scalars,
        np.full_like(h2_scalars, 17.0),
    )

    for name in (
        "latent_charges",
        "node_energies",
        "h1_adjoint",
        "h2_adjoint",
        "explicit_forces",
        "strain_derivative",
    ):
        np.testing.assert_allclose(reused[name], retained[name], rtol=0.0, atol=1.0e-12)
    assert reused["les_energy"] == pytest.approx(
        retained["les_energy"], rel=0.0, abs=1.0e-12
    )
    assert not np.allclose(
        overwritten_before_prepare["latent_charges"], retained["latent_charges"]
    )


def test_les_charge_readout_state_is_required_before_composition(mace_les_model):
    evaluator = _new_evaluator(mace_les_model)
    try:
        with pytest.raises(RuntimeError, match="LES charge readout state"):
            evaluator._compose_prepared_les_for_testing(3)
    finally:
        del evaluator
        gc.collect()


def test_failed_prepare_invalidates_les_charge_readout_state(mace_les_model):
    atoms = _atoms(periodic=True)
    nodes = len(atoms)
    h1_scalars = np.arange(nodes * 4, dtype=np.float64).reshape(nodes, 4) / 10.0
    h2_scalars = np.arange(nodes * 4, dtype=np.float64).reshape(nodes, 4) / 7.0
    evaluator = _new_evaluator(mace_les_model)
    try:
        evaluator._set_les_geometry(atoms.positions, atoms.cell.array, atoms.pbc)
        evaluator._prepare_les_charge_readouts_for_testing(h1_scalars, h2_scalars)

        mismatched_h1 = np.vstack([h1_scalars, h1_scalars[:1]])
        mismatched_h2 = np.vstack([h2_scalars, h2_scalars[:1]])
        with pytest.raises(ValueError, match="fresh LES geometry"):
            evaluator._prepare_les_charge_readouts_for_testing(
                mismatched_h1, mismatched_h2
            )

        evaluator.node_energies = np.zeros(nodes, dtype=np.float64)
        num_lm = np.asarray(evaluator.H1).size // (
            len(mismatched_h1) * h1_scalars.shape[1]
        )
        evaluator.H1_adj = np.zeros(
            nodes * num_lm * h1_scalars.shape[1], dtype=np.float64
        )
        evaluator.H2_adj = np.zeros_like(h2_scalars)
        with pytest.raises(RuntimeError, match="LES charge readout state"):
            evaluator._compose_prepared_les_for_testing(nodes)
    finally:
        del evaluator
        gc.collect()


def test_les_position_count_is_checked_before_int_narrowing(mace_les_model):
    evaluator = _new_evaluator(mace_les_model)
    max_nodes = np.iinfo(np.int32).max
    assert (
        evaluator._validate_les_position_count_for_testing(3 * max_nodes) == max_nodes
    )
    with pytest.raises(ValueError, match="node count exceeds"):
        evaluator._validate_les_position_count_for_testing(3 * (max_nodes + 1))


@pytest.mark.parametrize(
    ("class_name", "dtype"),
    [("MACELESKokkosFloat", "float32"), ("MACELESKokkos", "float64")],
)
def test_official_maceles_generic_matches_upstream_end_to_end(
    official_mace_les_model, class_name, dtype
):
    model, model_path = official_mace_les_model
    atoms = _atoms(periodic=True)
    reference = _official_mace_result(model, atoms)
    calculator = Symmetrix(
        model_path,
        dtype=dtype,
        streamed_edges="reference",
        neighbor_skin=0.0,
    )
    result = _evaluate(
        _new_evaluator(model_path, class_name),
        calculator,
        atoms,
        collect_stress=True,
    )

    charge_atol = 3.0e-8 if dtype == "float32" else 2.0e-9
    np.testing.assert_allclose(
        result["charges"], reference["charges"], rtol=2.0e-5, atol=charge_atol
    )
    assert result["les_energy"] == pytest.approx(
        reference["les_energy"], rel=2.0e-5, abs=1.0e-9
    )
    assert result["energy"] == pytest.approx(
        reference["energy"], rel=2.0e-7, abs=2.0e-8
    )
    np.testing.assert_allclose(
        result["forces"], reference["forces"], rtol=4.0e-5, atol=2.0e-6
    )
    np.testing.assert_allclose(
        result["stress"], reference["stress"], rtol=4.0e-5, atol=2.0e-8
    )


@pytest.mark.parametrize(
    ("class_name", "dtype", "rtol", "atol"),
    [
        ("MACELESKokkosFloat", np.float32, 3.0e-4, 3.0e-5),
        ("MACELESKokkos", np.float64, 2.0e-6, 2.0e-7),
    ],
)
def test_nonperiodic_les_composes_energy_partition_and_total_forces(
    mace_les_model, class_name, dtype, rtol, atol
):
    atoms = _atoms()
    calculator = _NativeMACEInputBuilder(mace_les_model)
    evaluator = _new_evaluator(mace_les_model, class_name)
    result = _evaluate(evaluator, calculator, atoms)

    assert np.any(result["charges"] != 0.0)
    assert result["les_energy"] != 0.0

    operator = getattr(
        native_symmetrix,
        "_LESMonopoleKokkosFloat" if dtype is np.float32 else "_LESMonopoleKokkos",
    )(1.0, 2.0, True)
    fixed_charge = operator.compute_nonperiodic(
        result["charges"].astype(dtype), atoms.positions
    )
    assert result["les_energy"] == pytest.approx(
        fixed_charge["energy"], rel=rtol, abs=atol
    )
    np.testing.assert_allclose(
        evaluator.les_atomic_energies,
        fixed_charge["atomic_energies"],
        rtol=rtol,
        atol=atol,
    )

    step = 2.0e-4 if dtype is np.float32 else 2.0e-5
    numeric_forces = np.empty_like(result["forces"])
    for atom in range(len(atoms)):
        for component in range(3):
            displaced = []
            for sign in (1.0, -1.0):
                candidate = atoms.copy()
                candidate.positions[atom, component] += sign * step
                displaced.append(_evaluate(evaluator, calculator, candidate)["energy"])
            numeric_forces[atom, component] = -(displaced[0] - displaced[1]) / (
                2.0 * step
            )

    np.testing.assert_allclose(
        result["forces"], numeric_forces, rtol=rtol * 20.0, atol=atol * 20.0
    )
    repeated = _evaluate(evaluator, calculator, atoms)["forces"]
    np.testing.assert_allclose(
        evaluator._reduce_atom_forces(
            len(atoms),
            calculator._mace_inputs(atoms)[-1],
            calculator._mace_inputs(atoms)[3],
        ),
        repeated,
        rtol=0.0,
        atol=1.0e-18,
    )


@pytest.mark.parametrize(
    ("class_name", "dtype", "rtol", "atol", "step"),
    [
        ("MACELESKokkosFloat", "float32", 2.0e-3, 2.0e-6, 3.0e-4),
        ("MACELESKokkos", "float64", 3.0e-5, 3.0e-8, 2.0e-5),
    ],
)
def test_periodic_les_total_stress_matches_symmetric_strain_finite_difference(
    mace_les_model, class_name, dtype, rtol, atol, step
):
    atoms = _atoms(periodic=True)
    calculator = Symmetrix(
        mace_les_model,
        dtype=dtype,
        streamed_edges="reference",
        neighbor_skin=0.0,
    )
    evaluator = _new_evaluator(mace_les_model, class_name)
    result = _evaluate(evaluator, calculator, atoms, collect_stress=True)
    volume = atoms.get_volume()

    for row, column in ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)):
        energies = []
        for sign in (1.0, -1.0):
            deformation = np.eye(3)
            deformation[row, column] += sign * step
            if row != column:
                deformation[column, row] += sign * step
            strained = atoms.copy()
            strained.set_cell(atoms.cell.array @ deformation, scale_atoms=True)
            energies.append(_evaluate(evaluator, calculator, strained)["energy"])
        numeric = (energies[0] - energies[1]) / (2.0 * step * volume)
        analytic = result["stress"][row, column]
        if row != column:
            analytic += result["stress"][column, row]
        assert analytic == pytest.approx(numeric, rel=rtol, abs=atol)


def test_maceles_direct_execution_is_available_from_model_identity(mace_les_model):
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()

    evaluator = native_symmetrix.MACELESKokkos(str(mace_les_model))
    try:
        evaluator.set_streamed_edges("direct")
        assert evaluator.is_mace_les
        assert evaluator.streamed_edges_mode == "direct"
    finally:
        del evaluator


def test_native_les_rejects_distributed_prepared_evaluation(mace_les_model):
    evaluator = _new_evaluator(mace_les_model)
    try:
        evaluator.set_streamed_edges("direct")

        with pytest.raises(
            ValueError, match="MACELES does not support distributed prepared evaluation"
        ):
            evaluator._begin_factorized_distributed_evaluation_for_testing()
    finally:
        del evaluator
        gc.collect()


def test_charge_adjoint_control_detects_missing_dq_dr(mace_les_model):
    atoms = _atoms()
    calculator = Symmetrix(
        mace_les_model,
        dtype="float64",
        streamed_edges="reference",
        neighbor_skin=0.0,
    )
    evaluator = _new_evaluator(mace_les_model)
    step = 2.0e-5
    energies = []
    for sign in (1.0, -1.0):
        displaced = atoms.copy()
        displaced.positions[0, 0] += sign * step
        energies.append(_evaluate(evaluator, calculator, displaced)["energy"])
    numeric_force = -(energies[0] - energies[1]) / (2.0 * step)

    evaluator._set_les_charge_adjoint_accumulation_for_testing(False)
    missing_chain_force = _evaluate(evaluator, calculator, atoms)["forces"][0, 0]
    force_tolerance = {"rel": 3.0e-5, "abs": 3.0e-8}
    assert missing_chain_force != pytest.approx(numeric_force, **force_tolerance)

    evaluator._set_les_charge_adjoint_accumulation_for_testing(True)
    restored_force = _evaluate(evaluator, calculator, atoms)["forces"][0, 0]
    assert restored_force == pytest.approx(numeric_force, **force_tolerance)


def test_prepared_generic_les_reuses_state_across_positions_and_cell(
    mace_les_model,
):
    atoms = _atoms(periodic=True)
    calculator = Symmetrix(
        mace_les_model,
        dtype="float64",
        streamed_edges="reference",
        neighbor_skin=0.0,
    )
    inputs = calculator._mace_inputs(atoms)
    evaluator = _new_evaluator(mace_les_model)
    num_nodes, node_types, num_neigh, sources, neigh_types, xyz, _, receivers = inputs
    token = evaluator._prepare_all_interactions_graph(
        num_nodes, node_types, num_neigh, sources, neigh_types
    )
    evaluator._prepare_all_interactions_geometry(
        token,
        np.asarray(atoms.positions).reshape(-1),
        np.asarray(xyz).reshape(-1),
        np.asarray(atoms.cell).reshape(-1),
        np.linalg.inv(atoms.cell.array).reshape(-1),
        np.asarray(atoms.pbc, dtype=np.int32),
    )
    moved_once = atoms.copy()
    moved_once.positions += [[0.01, -0.02, 0.0], [0.0, 0.01, -0.01], [-0.01, 0.0, 0.02]]
    moved_twice = moved_once.copy()
    moved_twice.positions += [
        [-0.005, 0.0, 0.01],
        [0.01, -0.005, 0.0],
        [0.0, 0.01, -0.005],
    ]
    changed_cell = moved_twice.copy()
    changed_cell.set_cell(
        atoms.cell.array @ np.diag([1.01, 0.99, 1.02]), scale_atoms=True
    )

    for index, candidate in enumerate((atoms, moved_once, moved_twice, changed_cell)):
        candidate_inputs = calculator._mace_inputs(candidate)
        for prepared_graph, candidate_graph in zip(inputs[:5], candidate_inputs[:5]):
            np.testing.assert_array_equal(prepared_graph, candidate_graph)
        if index == 3:
            candidate_xyz = candidate_inputs[5]
            evaluator._prepare_all_interactions_geometry(
                token,
                np.asarray(candidate.positions).reshape(-1),
                np.asarray(candidate_xyz).reshape(-1),
                np.asarray(candidate.cell).reshape(-1),
                np.linalg.inv(candidate.cell.array).reshape(-1),
                np.asarray(candidate.pbc, dtype=np.int32),
            )
        evaluator._set_les_geometry(
            candidate.positions, candidate.cell.array, candidate.pbc
        )
        evaluator._compute_prepared_all_interactions_positions(
            token, np.asarray(candidate.positions).reshape(-1)
        )
        prepared = {
            "energy": np.asarray(evaluator.node_energies, dtype=float).sum(),
            "les_energy": evaluator.les_energy,
            "forces": np.asarray(
                evaluator._reduce_atom_forces(num_nodes, receivers, sources, token)
            ),
            "stress": np.asarray(
                evaluator._reduce_stress(candidate.get_volume(), np.empty(0), token)
            ),
            "charges": np.asarray(evaluator.latent_charges),
        }
        reference = _evaluate(
            _new_evaluator(mace_les_model),
            calculator,
            candidate,
            collect_stress=True,
        )
        for key, value in prepared.items():
            np.testing.assert_allclose(
                value, reference[key], rtol=2.0e-12, atol=2.0e-13
            )
