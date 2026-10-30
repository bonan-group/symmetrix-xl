#!/usr/bin/env python3
"""One-physical-core performance record for the PolarMACE family.

The family's evaluation is the MH-1 backbone plus the polar deltas (the
unrolled global<->local chain) plus the long-range (k-space) operator group.
This driver measures, on one pinned physical core with one Kokkos, OpenMP, and
BLAS thread:

* the whole family's ``energy``, ``energy+forces``, and
  ``energy+forces+stress`` cost in us/atom, with the full graph and long-range
  context (atom count, model cutoff, skin, effective cutoff, directed-edge
  count, ``n_k``, dense-table bytes, the pbc-selected correction mode, and the
  retention's memory cost);
* an equal-shape serialized MH-1 graph-parity control through the plain
  generic/eager evaluator; this is not the specialized direct-backbone timing,
  which is reported by the stage breakdown;
* a stage breakdown (backbone forward, delta chain, long-range kernel group,
  and for gradients the chain reverse and the second seeded backbone pass);
* a size ladder separating fixed from per-atom cost and exposing where the
  ``O(N * n_k)`` long-range term starts to matter.

``--device cuda`` re-runs the same ladder on a CUDA build: no CPU affinity is
applied (the device is the resource under measurement), the host-side thread
variables stay at one, and every timed region is bracketed by the evaluator's
own ``fence()`` so the recorded duration covers the device work rather than the
enqueue.  The device's identity (name, compute capability, multiprocessor
count, runtime and driver versions) comes from the evaluator's
``device_cuda_environment`` and is recorded beside the run.

The equal-shape MH-1 control is the M checkpoint's own backbone serialized for
the plain MH-1 evaluator: the checkpoint's extracted JSON with the family's
model-type stamp replaced by ``MACE_Nonlinear`` and the three MH-1 schema shape
spellings normalized (scalar ``scale``/``shift``, ``[1, num_elements]`` atomic
energies).  The transform is shape- and weight-preserving, and this driver
verifies it by comparing the two evaluators' energy, forces, and stress on the
same prepared graph (measured agreement: bit-identical).  The checkpoints
themselves cannot be loaded as MH-1 model files: their radial embedding is the
retired pair-spline format, which the MH-1 family's streamed-edge handshake
refuses.

The stage timers are the repo's own surfaces: the calculator's
``_compute_mace``/``_compose_polar*`` entry points, the evaluator's
``polar_delta_chain``/``polar_delta_chain_reverse`` bindings, and the
``PolarLongRangeOps`` kernels, each wrapped with a timing shim for the duration
of a measurement.  Compilation, model extraction, graph construction, and
warmup are outside the steady-state region; the neighbor graph is prepared once
per system and reused.

Usage:

    export SYMMETRIX_SOURCE_ROOT=$PWD
    python benchmarks/polar_mace_serial_benchmark.py --cpu 0 \
        --model ~/.cache/symmetrix/test-models/MACE-POLAR-1-M.model \
        --output benchmarks/.artifacts/polar-mace-serial/record.json
"""

import argparse
import gc
import hashlib
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import time

# Thread pinning happens before NumPy, Kokkos, or BLAS is imported: the
# environment the runtimes read must be final before the first import, and the
# affinity mask must be in place before Kokkos initializes its worker pool.
THREAD_VARIABLES = (
    "KOKKOS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _smt_sibling(cpu):
    """The thread sibling of ``cpu``, or None when it has none."""

    path = pathlib.Path(
        f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
    )
    if not path.exists():
        return None
    siblings = set()
    for chunk in path.read_text().strip().split(","):
        if "-" in chunk:
            low, high = chunk.split("-", 1)
            siblings.update(range(int(low), int(high) + 1))
        else:
            siblings.add(int(chunk))
    siblings.discard(cpu)
    return sorted(siblings)


def _first_physical_cpu():
    """The lowest logical CPU whose sibling set is its own physical core."""

    for cpu in range(os.cpu_count() or 1):
        path = pathlib.Path(
            f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        )
        if not path.exists():
            return 0
        if "," in path.read_text().strip() or "-" in path.read_text().strip():
            return cpu
    return 0


def _configure_runtime_before_imports():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--cpus")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    early, _ = parser.parse_known_args()
    if early.cpu is not None and early.cpus is not None:
        parser.error("--cpu and --cpus are mutually exclusive")
    if early.device == "cuda" and (early.cpu is not None or early.cpus):
        parser.error("--cpu/--cpus pin the host core for the CPU record only")

    for variable in THREAD_VARIABLES:
        os.environ[variable] = "1"
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES", "cores")
    # The oracle checkpoint's pickle references graph_longrange, so the pinned
    # read-only checkout must be importable for the extraction.
    oracle = os.environ.get("SYMMETRIX_POLAR_ORACLE_GRAPH_LONGRANGE")
    if oracle:
        if not os.path.isdir(oracle):
            parser.error(f"graph_longrange checkout not found: {oracle}")
        if oracle not in sys.path:
            sys.path.insert(0, oracle)

    if early.device == "cuda":
        # The device record measures the accelerator; the host core is not the
        # resource under test, so no affinity is applied and none is claimed.
        return set()

    if early.cpu is not None:
        requested = {early.cpu}
    elif early.cpus:
        try:
            requested = {int(value) for value in early.cpus.split(",")}
        except ValueError:
            parser.error("--cpus must be a comma-separated integer list")
        if not requested:
            parser.error("--cpus cannot be empty")
    else:
        requested = {_first_physical_cpu()}
    if not hasattr(os, "sched_setaffinity"):
        parser.error("--cpu/--cpus require sched_setaffinity support")
    try:
        os.sched_setaffinity(0, requested)
    except OSError as error:
        parser.error(f"could not apply requested CPU affinity: {error}")
    if os.sched_getaffinity(0) != requested:
        parser.error(f"could not enforce CPU affinity {sorted(requested)}")
    return requested


def _bootstrap_explicit_symmetrix():
    source_root = os.environ.get("SYMMETRIX_SOURCE_ROOT")
    extension_path = os.environ.get("SYMMETRIX_EXTENSION")
    if not source_root or not extension_path:
        return
    import importlib.util

    package_dir = pathlib.Path(source_root).resolve() / "symmetrix/source/symmetrix"
    extension = pathlib.Path(extension_path).resolve()
    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if type(finder).__name__ != "ScikitBuildRedirectingFinder"
    ]
    package_spec = importlib.util.spec_from_file_location(
        "symmetrix",
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    native_spec = importlib.util.spec_from_file_location(
        "symmetrix.symmetrix", extension
    )
    package = importlib.util.module_from_spec(package_spec)
    native = importlib.util.module_from_spec(native_spec)
    sys.modules["symmetrix"] = package
    sys.modules["symmetrix.symmetrix"] = native
    native_spec.loader.exec_module(native)
    package_spec.loader.exec_module(package)


PINNED_CPUS = _configure_runtime_before_imports()
_bootstrap_explicit_symmetrix()

import numpy as np  # noqa: E402
from ase import Atoms  # noqa: E402

import symmetrix  # noqa: E402
from symmetrix import symmetrix as native_symmetrix  # noqa: E402

if not symmetrix._kokkos_is_initialized():
    symmetrix._init_kokkos()

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPOSITORY / "symmetrix/test/data/polar"
DEFAULT_MODEL = pathlib.Path("~/.cache/symmetrix/test-models/MACE-POLAR-1-M.model")
DEFAULT_OUTPUT = REPOSITORY / "benchmarks/.artifacts/polar-mace-serial/record.json"

EXECUTION_COUNTERS = (
    "execution_mh1_generated_forward_launch_count",
    "execution_mh1_generated_source_reverse_launch_count",
    "execution_mh1_generated_edge_reverse_launch_count",
    "factorized_fallback_evaluation_count",
)


def _execution_counter_snapshot(evaluator):
    return {name: int(getattr(evaluator, name)) for name in EXECUTION_COUNTERS}


def _execution_counter_delta(before, after):
    delta = {name: after[name] - before[name] for name in EXECUTION_COUNTERS}
    if any(value < 0 for value in delta.values()):
        raise RuntimeError(f"execution counters moved backwards: {before=} {after=}")
    return delta


def _measured_execution_evidence(calculator, before, after):
    delta = _execution_counter_delta(before, after)
    direct_route = calculator.streamed_edges in ("direct", "direct_streamed")
    classification = (
        "direct"
        if direct_route
        else (
            "fallback"
            if delta["factorized_fallback_evaluation_count"] > 0
            else "non-direct"
        )
    )
    evidence = {
        "counter_scope": "from immediately before through immediately after timed workloads",
        "counter_start": before,
        "counter_end": after,
        "counter_delta": delta,
        "classification": classification,
    }
    if direct_route:
        missing = [name for name in EXECUTION_COUNTERS[:3] if delta[name] <= 0]
        if missing or delta["factorized_fallback_evaluation_count"] != 0:
            raise RuntimeError(
                "direct measured workload lacks generated-only execution evidence: "
                f"{evidence}"
            )
    return evidence


def _aggregate_measured_execution_evidence(calculator, windows):
    total = {name: 0 for name in EXECUTION_COUNTERS}
    for window in windows:
        for name, value in window["counter_delta"].items():
            total[name] += value
    evidence = _measured_execution_evidence(
        calculator, {name: 0 for name in EXECUTION_COUNTERS}, total
    )
    evidence["counter_scope"] = (
        "sum of per-system windows bracketing only primary synchronized "
        "timing warmups and repeats"
    )
    evidence["windows"] = windows
    evidence.pop("counter_start")
    evidence.pop("counter_end")
    return evidence


def _comparator_record_fields(enabled, equivalence=None):
    fields = {
        "comparator_verification": {
            "enabled": bool(enabled),
            "status": "passed" if enabled else "disabled_by_cli",
        }
    }
    if enabled:
        fields["comparator_equivalence"] = equivalence
    return fields


#: The fixture geometry record each named system takes its species, positions,
#: cell, and pbc from, plus the supercell repeat applied to it.
FIXTURE_SYSTEMS = {
    "mol_h2o": ("e2e_M_mol_h2o_singlet_float64.json", (1, 1, 1)),
    "slab_w2": ("e2e_M_slab_w2_singlet_float64.json", (1, 1, 1)),
    "bulk_w4": ("e2e_M_bulk_w4_singlet_float64.json", (1, 1, 1)),
    "bulk_w4_x8": ("e2e_M_bulk_w4_singlet_float64.json", (2, 2, 2)),
    "bulk_w4_x27": ("e2e_M_bulk_w4_singlet_float64.json", (3, 3, 3)),
    "bulk_w4_x64": ("e2e_M_bulk_w4_singlet_float64.json", (4, 4, 4)),
    "bulk_w4_x125": ("e2e_M_bulk_w4_singlet_float64.json", (5, 5, 5)),
}

#: The pbc class -> correction mode map of the design note's mixed_periodic
#: matrix, recorded per system so the capacity record states which analytic
#: O(N) terms the graph class selects.
CORRECTION_MODE = {
    (False, False, False): "makov-payne-dabo",
    (True, True, False): "slab-dipole",
    (True, True, True): "none",
}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - non-git checkouts
        return None


def _cpu_model():
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _peak_rss_bytes():
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------
# The equal-shape generic/eager MH-1 graph control
# ---------------------------------------------------------------------------


def build_mh1_backbone_json(model_path, output_dir, model_sha):
    """Serialize the checkpoint's backbone for the plain MH-1 evaluator.

    The extraction routes a PolarMACE checkpoint to the polar extractor, which
    stamps ``MACE_Polar``.  The MH-1 evaluator's schema requires its own model
    type and its own three shape spellings, so the transform replaces the stamp
    and normalizes the spellings -- and nothing else.  The ``polar`` block stays
    in place: it is the data the MH-1 contract's ``keep_last_layer_irreps``
    product rule reads, and the MH-1 evaluator ignores the rest of it.
    """

    from symmetrix.extract_mace_data import extract_mace_data

    cached = output_dir / f"mh1-backbone-{model_sha[:12]}.json"
    stamp = output_dir / f"mh1-backbone-{model_sha[:12]}.stamp.json"
    if cached.is_file() and stamp.is_file():
        recorded = json.loads(stamp.read_text())
        if recorded.get(
            "model_sha256"
        ) == model_sha and cached.stat().st_size == recorded.get("bytes"):
            return cached, recorded["seconds"]
    data = extract_mace_data(model_path)
    data["model_type"] = "MACE_Nonlinear"
    scale_shift = data["scale_shift"]
    for key in ("scale", "shift"):
        scale_shift[key]["shape"] = []
    energies = data["atomic_energies"]
    energies["shape"] = [1, len(energies["values"])]
    start = time.perf_counter()
    cached.write_text(json.dumps(data, separators=(",", ":")))
    seconds = time.perf_counter() - start
    stamp.write_text(
        json.dumps(
            {
                "model_sha256": model_sha,
                "bytes": cached.stat().st_size,
                "seconds": seconds,
            },
            indent=1,
        )
    )
    return cached, seconds


def load_fixture_system(name, output_dir):
    record_name, repeat = FIXTURE_SYSTEMS[name]
    record = json.loads((FIXTURE_DIR / record_name).read_text())
    atoms = Atoms(
        record["species"],
        positions=np.array(record["positions"], dtype=np.float64),
        cell=np.array(record["cell"], dtype=np.float64).reshape(3, 3),
        pbc=record["pbc"],
    )
    if repeat != (1, 1, 1):
        atoms = atoms.repeat(repeat)
    return atoms, {
        "fixture": record_name,
        "supercell": list(repeat),
        "graph_class": record["graph_class"],
    }


# ---------------------------------------------------------------------------
# Timing shims
# ---------------------------------------------------------------------------


class _Recorder:
    """Named duration buckets, appended in call order."""

    def __init__(self):
        self.samples = {}
        self.order = []

    def add(self, name, seconds):
        self.samples.setdefault(name, []).append(seconds)
        self.order.append(name)

    def median_ms(self, name):
        values = self.samples.get(name)
        return 1000.0 * statistics.median(values) if values else None

    def count(self, name):
        return len(self.samples.get(name, ()))

    def total_ms(self, name):
        values = self.samples.get(name)
        return 1000.0 * sum(values) if values else None


class _StageTimer:
    """Wrap the calculator's and evaluator's own entry points with timers.

    Nothing is reimplemented: the shims call the originals and record how long
    they took, so the composition the record attributes is exactly the one the
    frontend runs.  The evaluator is a pybind11 object and takes no attributes,
    so its two chain entry points are wrapped on the class for the duration of
    the measurement (and restored afterwards).
    """

    def __init__(self, calculator):
        self.calculator = calculator
        self.recorder = _Recorder()
        self._saved = {}

    def __enter__(self):
        self._saved["evaluator"] = self.calculator.evaluator
        for name in (
            "_compute_mace",
            "_compose_polar",
            "_compose_polar_reverse",
            "_compose_polar_stress",
        ):
            original = getattr(self.calculator, name)

            def wrapper(*args, __original=original, __name=name, **kwargs):
                start = time.perf_counter()
                try:
                    return __original(*args, **kwargs)
                finally:
                    self.recorder.add(__name, time.perf_counter() - start)

            setattr(self.calculator, name, wrapper)
            self._saved[name] = original

        # The long-range kernels the composition calls on its own ops object:
        # replace the attribute with a delegating proxy so every Python call
        # site in the composition is timed without touching native code.
        if self.calculator._polar_ops is not None:
            self._saved["polar_ops"] = self.calculator._polar_ops
            self.calculator._polar_ops = _OpsProxy(
                self.calculator._polar_ops, self.recorder
            )

        chain_names = ("polar_delta_chain", "polar_delta_chain_reverse")
        for cls_name in ("MACEPolarKokkos", "MACEPolarKokkosFloat"):
            cls = getattr(native_symmetrix, cls_name, None)
            if cls is None:
                continue
            for method in chain_names:
                key = (cls_name, method)
                self._saved[key] = getattr(cls, method)
                original = getattr(cls, method)

                def wrapper(self_, *args, __original=original, __key=key, **kwargs):
                    start = time.perf_counter()
                    try:
                        return __original(self_, *args, **kwargs)
                    finally:
                        self.recorder.add(__key[1], time.perf_counter() - start)

                setattr(cls, method, wrapper)
        return self

    def __exit__(self, *exc):
        for name in (
            "_compute_mace",
            "_compose_polar",
            "_compose_polar_reverse",
            "_compose_polar_stress",
        ):
            if name in self._saved:
                setattr(self.calculator, name, self._saved[name])
        if "polar_ops" in self._saved:
            self.calculator._polar_ops = self._saved["polar_ops"]
        for key, original in self._saved.items():
            if isinstance(key, tuple):
                cls = getattr(native_symmetrix, key[0])
                setattr(cls, key[1], original)
        return False


class _OpsProxy:
    """Delegate to a ``PolarLongRangeOps`` while timing the kernels."""

    _TIMED = frozenset(
        {
            "geometry_cache",
            "assemble_fourier",
            "apply_coulomb",
            "project_to_features",
            "coulomb_energy",
            "analytic_correction_energy",
            "analytic_correction_features",
            "dipole_readout",
            "fukui_normalize",
            "external_field_features",
            "si_feature_constants",
            "si_energy_constants",
            "feature_self_terms",
            "energy_self_fields",
            "external_field_matrix",
            "adjoint_coulomb_energy",
            "adjoint_correction_energy",
            "adjoint_dipole_readout",
            "adjoint_energy_self_fields",
            "adjoint_geometry",
            "coulomb_stress",
        }
    )

    def __init__(self, ops, recorder):
        self._ops = ops
        self._recorder = recorder

    def __getattr__(self, name):
        attribute = getattr(self._ops, name)
        if name not in self._TIMED or not callable(attribute):
            return attribute

        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return attribute(*args, **kwargs)
            finally:
                self._recorder.add(name, time.perf_counter() - start)

        return timed


def _native_args(mace_inputs):
    """The evaluator's reduced surface arguments, as the other drivers pass them."""

    return (
        *mace_inputs[:5],
        np.ascontiguousarray(mace_inputs[5]).reshape(-1),
        np.ascontiguousarray(mace_inputs[6]),
    )


def _collect_from_evaluator(calculator, evaluator, atoms, mace_inputs, properties):
    """Collect results off an arbitrary evaluator through the frontend's own path."""

    saved = calculator.evaluator
    calculator.evaluator = evaluator
    try:
        return calculator._collect_mace_results(atoms, mace_inputs, properties, 0)
    finally:
        calculator.evaluator = saved


#: Set to the evaluator's ``fence`` under ``--device cuda``: every timed region
#: is bracketed by it so the sample covers the device work, not the enqueue.
FENCE = None


def _model_cutoff_edges(calculator, atoms):
    """Directed-edge count the graph carries at the model cutoff, no skin.

    The measured graph is the skinned one (cutoff + ``neighbor_skin``); this is
    the same neighbor-list machinery asked for the bare model cutoff, so the two
    counts differ only by the skin.
    """

    from symmetrix.neighbor_graph import host_neighbor_list

    i_list, _j_list, _r, _xyz = host_neighbor_list(atoms, calculator.cutoff, "ijdD")
    return len(i_list)


def _device_context(evaluator):
    """The accelerator identity straight off the evaluator's own query."""

    environment = dict(evaluator.device_cuda_environment)
    if not environment.get("available"):
        return {"available": False}
    return environment


def _median_ms(call, repeats, warmups):
    global FENCE  # noqa: PLW0603 - one process-wide device fence, set in main
    for _ in range(warmups):
        call()
    if FENCE is not None:
        FENCE()
    samples = []
    for _ in range(repeats):
        if FENCE is not None:
            FENCE()
        start = time.perf_counter()
        call()
        if FENCE is not None:
            FENCE()
        samples.append(1000.0 * (time.perf_counter() - start))
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


# ---------------------------------------------------------------------------
# The per-system measurement
# ---------------------------------------------------------------------------


def _capacity_fields(calculator, composition, system, cache):
    n_k = int(cache.n_k)
    nodes = int(cache.n_nodes)
    itemsize = 8 if calculator._polar_contract["table_dtype"] == "float64" else 4
    shapes = calculator._polar_shapes
    layers = int(shapes["interaction_count"])
    width = int(shapes["node_feature_dimension"])
    pbc = tuple(bool(value) for value in composition["pbc"].reshape(-1))
    return {
        # The manifest's ``long_range`` capacity vocabulary.
        "n_k": n_k,
        "table_bytes": 2 * n_k * nodes * itemsize,
        "correction_mode": CORRECTION_MODE[pbc],
        "rebuild_on_cell_change": True,
        # The rest of the record's context: the graph currency and the
        # retention the family's chain pays on top of the tables.
        "table_dtype": calculator._polar_contract["table_dtype"],
        "table_shape": [n_k, nodes],
        "retention_bytes": nodes * layers * width * itemsize,
        "retention_shape": [nodes, layers, width],
        "retention_dtype": calculator._polar_contract["table_dtype"],
        "spin_channel_count": int(calculator._polar_contract["spin_channel_count"]),
        "recursion_step_count": int(calculator._polar_contract["recursion_step_count"]),
        "charges_dimension": int(shapes["charges_dimension"]),
        "potential_dimension": int(shapes["potential_dimension"]),
        "node_feature_dimension": width,
        "interaction_count": layers,
        "kspace_cutoff": float(calculator._polar_contract["kspace_cutoff"]),
        "graph_class": system["graph_class"],
    }


#: The stage names the record reports, in the order the evaluation runs them.
FORWARD_STAGES = (
    "backbone_forward",
    "chain_forward",
    "long_range.energy_path",
)
GRADIENT_STAGES = (
    "backbone_forward",
    "chain_forward",
    "long_range.energy_path",
    "chain_reverse",
    "long_range.adjoint",
    "backbone_seeded_pass",
)


def measure_system(name, calculator, mh1_evaluator, atoms, system, args):
    atoms.calc = calculator
    properties = {
        "energy": ["energy"],
        "energy_forces": ["energy", "forces"],
        "energy_forces_stress": ["energy", "forces", "stress"],
    }
    # Warm up the whole path once per property set so graph preparation, the
    # retention, and any artifact selection are outside the measured region.
    for names in properties.values():
        calculator.calculate(atoms, properties=names)
    mace_inputs = calculator._mace_inputs(atoms)
    native = _native_args(mace_inputs)
    edges = len(mace_inputs[3])
    edges_at_model_cutoff = _model_cutoff_edges(calculator, atoms)
    nodes = len(atoms)

    # The MH-1 baseline on the same prepared graph: one native call, collected
    # through the frontend's own result path with the MH-1 evaluator bound.
    mh1_reference = {}
    polar_reference = {}
    if args.verify_comparator:
        calculator.evaluator.set_retain_layer_features(False)
        calculator._compute_mace(mace_inputs, atoms=atoms)
        polar_reference = _collect_from_evaluator(
            calculator,
            calculator.evaluator,
            atoms,
            mace_inputs,
            ["energy", "forces", "stress"],
        )
        calculator.evaluator.set_retain_layer_features(True)
        mh1_evaluator.compute_node_energies_forces(*native)
        mh1_reference = _collect_from_evaluator(
            calculator,
            mh1_evaluator,
            atoms,
            mace_inputs,
            ["energy", "forces", "stress"],
        )

    measured_counter_start = _execution_counter_snapshot(calculator.evaluator)
    timings = {}
    for key, names in properties.items():
        timings[key] = _median_ms(
            lambda names=names: calculator.calculate(atoms, properties=names),
            args.repeats,
            args.warmups,
        )
    measured_counter_end = _execution_counter_snapshot(calculator.evaluator)
    measured_execution = _measured_execution_evidence(
        calculator, measured_counter_start, measured_counter_end
    )

    # The MH-1 baseline's three modes: the native reduced call always produces
    # energy, forces, and the virial together, so the modes differ only in what
    # the collector reads -- the same convention the family's own paths use.
    mh1_timings = {}
    if args.verify_comparator:
        for key, names in properties.items():
            mh1_timings[key] = _median_ms(
                lambda: (
                    mh1_evaluator.compute_node_energies_forces(*native),
                    _collect_from_evaluator(
                        calculator, mh1_evaluator, atoms, mace_inputs, names
                    ),
                ),
                args.repeats,
                args.warmups,
            )

    # Stage attribution: the calculator's and the evaluator's entry points are
    # wrapped, and the wrapped evaluation is repeated, so each stage's reported
    # value is a median over runs rather than a single sample.
    stage_samples = {}
    for _ in range(args.stage_runs):
        for key, names in properties.items():
            with _StageTimer(calculator) as timer:
                if FENCE is not None:
                    FENCE()
                calculator.calculate(atoms, properties=names)
                if FENCE is not None:
                    FENCE()
            recorder = timer.recorder
            bucket = stage_samples.setdefault(key, {})
            passes = list(recorder.samples.get("_compute_mace", ()))
            _push(bucket, "backbone_forward_ms", passes[0] * 1000.0 if passes else None)
            _push(
                bucket,
                "backbone_seeded_pass_ms",
                passes[1] * 1000.0 if len(passes) > 1 else None,
            )
            for stage, source in (
                ("chain_forward_ms", "polar_delta_chain"),
                ("chain_reverse_ms", "polar_delta_chain_reverse"),
                ("composition_forward_ms", "_compose_polar"),
                ("composition_reverse_ms", "_compose_polar_reverse"),
                ("composition_stress_ms", "_compose_polar_stress"),
            ):
                values = recorder.samples.get(source, ())
                _push(bucket, stage, values[0] * 1000.0 if values else None)
            bucket.setdefault("backbone_pass_count", []).append(len(passes))
            ops_bucket = bucket.setdefault("long_range", {})
            calls_bucket = bucket.setdefault("long_range_calls", {})
            for op in sorted(_OpsProxy._TIMED):
                values = recorder.samples.get(op, ())
                if values:
                    ops_bucket.setdefault(op, []).append(1000.0 * sum(values))
                    calls_bucket.setdefault(op, []).append(len(values))
    stages = {
        key: {
            **{
                name: statistics.median(values)
                for name, values in bucket.items()
                if name not in ("long_range", "long_range_calls", "backbone_pass_count")
            },
            "backbone_pass_count": int(
                statistics.median(bucket["backbone_pass_count"])
            ),
            "long_range": {
                op: statistics.median(values)
                for op, values in bucket.get("long_range", {}).items()
            },
            "long_range_calls": {
                op: int(statistics.median(values))
                for op, values in bucket.get("long_range_calls", {}).items()
            },
        }
        for key, bucket in stage_samples.items()
    }
    # The retention's cost, measured directly on the same prepared graph.
    retention = {}
    calculator.evaluator.set_retain_layer_features(False)
    retention["backbone_forward_off_ms"] = _median_ms(
        lambda: calculator._compute_mace(mace_inputs, atoms=atoms), args.repeats, 1
    )["median_ms"]
    calculator.evaluator.set_retain_layer_features(True)
    retention["backbone_forward_on_ms"] = _median_ms(
        lambda: calculator._compute_mace(mace_inputs, atoms=atoms), args.repeats, 1
    )["median_ms"]

    # The long-range kernels measured on their own, one shot each, on the
    # composition's own cache and density: this is the per-kernel cost the
    # contract's evaluation counts multiply, and the evidence the T3 decision
    # rests on.
    # Production E/F/stress deliberately leaves private chain and long-range
    # state resident in native memory.  The standalone kernel diagnostics below
    # need that state, so materialize it explicitly only after every primary
    # synchronized timing and stage-attribution sample has completed.
    composition = calculator._materialize_polar_diagnostics()
    ops = calculator._polar_ops
    positions = np.ascontiguousarray(composition["positions"], dtype=np.float64)
    cell = np.ascontiguousarray(composition["cell"], dtype=np.float64).reshape(1, 3, 3)
    index = composition["node_graph_index"]
    cache_result = ops.geometry_cache(positions, cell, index)
    cache = cache_result["cache"]
    kernels = {
        "geometry_cache": _median_ms(
            lambda: ops.geometry_cache(positions, cell, index),
            args.repeats,
            args.warmups,
        )["median_ms"]
    }
    if "charge_density" in composition:
        density = np.ascontiguousarray(composition["charge_density"], dtype=np.float64)
        multipoles = density
        structure = ops.assemble_fourier(cache, multipoles)
        potential = ops.apply_coulomb(cache, structure)
        kernels["assemble_fourier"] = _median_ms(
            lambda: ops.assemble_fourier(cache, multipoles),
            args.repeats,
            args.warmups,
        )["median_ms"]
        kernels["apply_coulomb"] = _median_ms(
            lambda: ops.apply_coulomb(cache, structure),
            args.repeats,
            args.warmups,
        )["median_ms"]
        kernels["project_to_features"] = _median_ms(
            lambda: ops.project_to_features(cache, potential),
            args.repeats,
            args.warmups,
        )["median_ms"]
        kernels["coulomb_energy"] = _median_ms(
            lambda: ops.coulomb_energy(cache, density, None),
            args.repeats,
            args.warmups,
        )["median_ms"]
        kernels["analytic_correction_energy"] = _median_ms(
            lambda: ops.analytic_correction_energy(
                density, positions, cell, index, composition["pbc"]
            ),
            args.repeats,
            args.warmups,
        )["median_ms"]
        kernels["dipole_readout"] = _median_ms(
            lambda: ops.dipole_readout(density, positions, index),
            args.repeats,
            args.warmups,
        )["median_ms"]

    capacity = _capacity_fields(calculator, composition, system, cache)
    attribution = {}
    for key in timings:
        stage = stages[key]
        parts = {
            name: stage[name]
            for name in (
                "backbone_forward_ms",
                "composition_forward_ms",
                "composition_reverse_ms",
                "backbone_seeded_pass_ms",
                "composition_stress_ms",
            )
            if stage.get(name) is not None
        }
        parts["frontend_residual_ms"] = timings[key]["median_ms"] - sum(parts.values())
        attribution[key] = parts
    record = {
        "system": name,
        "atoms": nodes,
        "directed_edges": edges,
        "directed_edges_model_cutoff": edges_at_model_cutoff,
        "model_cutoff_A": float(calculator.cutoff),
        "neighbor_skin_A": float(calculator.neighbor_skin),
        "effective_neighbor_cutoff_A": float(
            calculator.cutoff + calculator.neighbor_skin
        ),
        "cell_A": [float(value) for value in np.diag(atoms.cell)],
        "pbc": [bool(value) for value in atoms.pbc],
        "us_per_atom": {
            key: 1000.0 * timing["median_ms"] / nodes for key, timing in timings.items()
        },
        "us_per_atom_mh1_generic_comparator": {
            key: 1000.0 * timing["median_ms"] / nodes
            for key, timing in mh1_timings.items()
        },
        "polar_ms": timings,
        "mh1_generic_comparator_ms": mh1_timings,
        "polar_vs_generic_comparator_percent": {
            key: 100.0
            * (timings[key]["median_ms"] / mh1_timings[key]["median_ms"] - 1.0)
            for key in mh1_timings
        },
        "stages_ms": stages,
        "attribution_ms": attribution,
        "retention_backbone_ms": retention,
        "long_range_kernels_ms": kernels,
        "capacity": capacity,
        "measured_execution": measured_execution,
    }
    equivalence = None
    if args.verify_comparator:
        equivalence = {
            "energy_abs_diff": abs(
                float(polar_reference["energy"]) - float(mh1_reference["energy"])
            ),
            "forces_max_abs_diff": float(
                np.max(
                    np.abs(
                        np.asarray(polar_reference["forces"])
                        - np.asarray(mh1_reference["forces"])
                    )
                )
            ),
            "stress_max_abs_diff": float(
                np.max(
                    np.abs(
                        np.asarray(polar_reference["stress"])
                        - np.asarray(mh1_reference["stress"])
                    )
                )
            ),
        }
    record.update(_comparator_record_fields(args.verify_comparator, equivalence))
    return record


def _push(bucket, name, value):
    if value is not None:
        bucket.setdefault(name, []).append(value)


def _patched_spline_model(model_path, nodes, output):
    """A copy of the model JSON with its pair-spline resolution set.

    The released checkpoints write no ``num_spline_points``, so the shipped 256
    applies; the sweep needs the value the *model* would declare, which is the
    same key the extractor writes.
    """

    destination_dir = output.expanduser().resolve().parent / "spline-models"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{model_path.stem}-n{nodes}.json"
    data = json.loads(model_path.read_text())
    data.setdefault("radial_embedding", {})["num_spline_points"] = int(nodes)
    destination.write_text(json.dumps(data, separators=(",", ":")))
    return destination


def _device_identity(calculator):
    """The accelerator identity, from nvidia-smi and the evaluator's own query."""

    environment = dict(calculator.evaluator.device_cuda_environment)
    fields = {}
    if environment.get("available"):
        fields["evaluator"] = environment
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,"
                "compute_cap,utilization.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception as error:  # pragma: no cover - diagnostics only
        query = f"nvidia-smi unavailable: {error}"
    fields["nvidia_smi"] = query
    try:
        processes = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception as error:  # pragma: no cover - diagnostics only
        processes = f"nvidia-smi unavailable: {error}"
    fields["compute_processes"] = processes
    fields["native_extension"] = getattr(native_symmetrix, "__file__", None)
    fields["native_extension_sha256"] = (
        _sha256(native_symmetrix.__file__)
        if getattr(native_symmetrix, "__file__", None)
        else None
    )
    fields["kokkos_execution_space"] = (
        symmetrix._kokkos_default_execution_space()
        if hasattr(symmetrix, "_kokkos_default_execution_space")
        else None
    )
    return fields


def _model_identity(path, calculator):
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "cutoff_A": float(calculator.cutoff),
        "neighbor_skin_A": float(calculator.neighbor_skin),
        "dtype": getattr(calculator, "dtype", None)
        or getattr(calculator, "_dtype", None)
        or "float64",
        "family": calculator._family_entry.key,
        "polar_contract": dict(calculator._polar_contract),
        "polar_shapes": dict(calculator._polar_shapes),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    # Parsed and applied before the numerical imports; repeated here so the
    # documented command line is accepted by the main parser as well.
    parser.add_argument("--cpu", type=int, default=None)
    parser.add_argument("--cpus", default=None)
    parser.add_argument(
        "--systems",
        default="mol_h2o,slab_w2,bulk_w4,bulk_w4_x8",
        help="comma-separated fixture system names",
    )
    parser.add_argument(
        "--ladder",
        default="mol_h2o,slab_w2,bulk_w4,bulk_w4_x8,bulk_w4_x27",
        help="comma-separated systems measured with the reduced repeat count",
    )
    parser.add_argument("--skip-ladder", action="store_true")
    parser.add_argument("--skip-diagnostics", action="store_true")
    parser.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--stage-runs",
        type=int,
        default=3,
        help="instrumented evaluations per property set (each stage is reported as a median)",
    )
    parser.add_argument("--ladder-warmups", type=int, default=2)
    parser.add_argument("--ladder-repeats", type=int, default=5)
    parser.add_argument("--ladder-stage-runs", type=int, default=2)
    parser.add_argument(
        "--anchor-repeats",
        type=int,
        default=5,
        help="repeats of the fixed anchor workload measured around the run",
    )
    parser.add_argument(
        "--artifact-probe",
        action="store_true",
        default=True,
        help="measure whether the MH-1 host node-program generator serves the checkpoint",
    )
    parser.add_argument(
        "--no-artifact-probe", dest="artifact_probe", action="store_false"
    )
    parser.add_argument("--verify-comparator", action="store_true", default=True)
    parser.add_argument(
        "--no-verify-comparator", dest="verify_comparator", action="store_false"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="cpu pins one physical core; cuda measures the accelerator",
    )
    parser.add_argument(
        "--streamed-edges",
        default=None,
        choices=("reference", "generic", "direct", "direct_streamed"),
        help=(
            "select the calculator route through the production frontend, so "
            "the record's resolved routing and the evaluator's execution mode "
            "cannot disagree"
        ),
    )
    parser.add_argument(
        "--edge-executor",
        default=None,
        choices=("mlp_reference", "pair_spline_v1"),
        help="the MH-1 conditioned-edge executor (exact or spline)",
    )
    parser.add_argument(
        "--spline-nodes",
        type=int,
        default=None,
        help=(
            "patch the model JSON's radial_embedding.num_spline_points before "
            "loading, for a spline-resolution sweep"
        ),
    )
    args = parser.parse_args(argv)

    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        parser.error(f"model checkpoint not found: {model_path}")
    if args.device == "cuda":
        # The host-core pinning and the box/node/backend ladders are CPU-record
        # protocol; the device record replaces them with the accelerator's own
        # identity.
        args.skip_diagnostics = True
    if args.spline_nodes is not None:
        model_path = _patched_spline_model(model_path, args.spline_nodes, args.output)

    output_dir = args.output.expanduser().resolve().parent
    (output_dir / "comparator").mkdir(parents=True, exist_ok=True)

    model_sha = _sha256(model_path)
    comparator_json, comparator_seconds = build_mh1_backbone_json(
        model_path, output_dir / "comparator", model_sha
    )

    build_start = time.perf_counter()
    calculator = symmetrix.Symmetrix(
        str(model_path), dtype=args.dtype, streamed_edges=args.streamed_edges
    )
    build_seconds = time.perf_counter() - build_start
    mh1_evaluator = native_symmetrix.MACENonlinearKokkos(str(comparator_json))

    if args.edge_executor is not None:
        calculator.evaluator.set_mh1_edge_executor(args.edge_executor)

    global FENCE  # noqa: PLW0603 - the one device fence this process uses
    if args.device == "cuda":
        FENCE = calculator.evaluator.fence

    identity = _model_identity(model_path, calculator)
    identity["comparator"] = {
        "path": str(comparator_json),
        "sha256": _sha256(comparator_json),
        "bytes": comparator_json.stat().st_size,
        "serialize_seconds": comparator_seconds,
        "transform": (
            "the checkpoint's extraction with model_type set to MACE_Nonlinear and "
            "scale/shift/atomic_energies respelled to the MH-1 schema"
        ),
    }

    records = []
    tier = measure_tier(calculator, comparator_json, args)
    anchors = {"before_systems": _anchor_ms(calculator, args.anchor_repeats)}
    systems = [name for name in args.systems.split(",") if name]
    for name in systems:
        atoms, system = load_fixture_system(name, output_dir)
        records.append(
            measure_system(name, calculator, mh1_evaluator, atoms, system, args)
        )
        _print_system(records[-1])

    ladder = []
    if not args.skip_ladder:
        anchors["before_ladder"] = _anchor_ms(calculator, args.anchor_repeats)
        ladder_args = argparse.Namespace(**vars(args))
        ladder_args.warmups = args.ladder_warmups
        ladder_args.repeats = args.ladder_repeats
        ladder_args.stage_runs = args.ladder_stage_runs
        for name in [
            entry for entry in args.ladder.split(",") if entry and entry not in systems
        ]:
            atoms, system = load_fixture_system(name, output_dir)
            ladder.append(
                measure_system(
                    name, calculator, mh1_evaluator, atoms, system, ladder_args
                )
            )
            _print_system(ladder[-1], ladder=True)

    measured_execution = _aggregate_measured_execution_evidence(
        calculator,
        [entry["measured_execution"] for entry in (*records, *ladder)],
    )

    diagnostics = None
    if not args.skip_diagnostics:
        diagnostics = measure_diagnostics(calculator, args)
    anchors["after_diagnostics"] = _anchor_ms(calculator, args.anchor_repeats)
    anchor_reference = anchors["before_systems"]
    anchors["drift_percent"] = {
        name: 100.0 * (value / anchor_reference - 1.0)
        for name, value in anchors.items()
        if name != "drift_percent"
    }

    evaluator = calculator.evaluator
    jit_artifact = pathlib.Path(calculator.jit_artifact_path).resolve()
    execution_evidence = {
        "requested_route": calculator.streamed_edges_requested,
        "resolved_route": calculator.streamed_edges,
        "native_route": calculator._native_streamed_edges,
        "algorithm": calculator.execution_algorithm,
        "jit_status": calculator.jit_status,
        "jit_cache_key": calculator.jit_cache_key,
        "jit_artifact_id": calculator.jit_artifact_id,
        "jit_artifact_path": str(jit_artifact),
        "jit_artifact_sha256": _sha256(jit_artifact),
        "generated_forward_launch_count": (
            evaluator.execution_mh1_generated_forward_launch_count
        ),
        "generated_source_reverse_launch_count": (
            evaluator.execution_mh1_generated_source_reverse_launch_count
        ),
        "generated_edge_reverse_launch_count": (
            evaluator.execution_mh1_generated_edge_reverse_launch_count
        ),
        "factorized_fallback_evaluation_count": (
            evaluator.factorized_fallback_evaluation_count
        ),
        "measured_workload": measured_execution,
    }
    if calculator.streamed_edges in ("direct", "direct_streamed"):
        if execution_evidence["native_route"] != "direct":
            raise RuntimeError(
                f"direct qualification resolved another native route: {execution_evidence}"
            )
        # Generated/fallback acceptance is based on the measured window above.
        # The cumulative fields remain for consumers of version-1 records.

    record = {
        "schema": "symmetrix.polar.performance-record",
        "version": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {
            "cpu": _cpu_model(),
            "cores": os.cpu_count(),
            "platform": platform.platform(),
            "pinned_cpus": sorted(PINNED_CPUS),
            "smt_siblings": {
                str(cpu): _smt_sibling(cpu) for cpu in sorted(PINNED_CPUS)
            },
            "thread_environment": {
                name: os.environ.get(name) for name in THREAD_VARIABLES
            },
            "omp_proc_bind": os.environ.get("OMP_PROC_BIND"),
            "omp_places": os.environ.get("OMP_PLACES"),
        },
        "repository": {
            "revision": _git_revision(),
            "extension": getattr(symmetrix, "__file__", None),
            "backend_build_info": dict(native_symmetrix._backend_build_info()),
        },
        "device": (
            _device_identity(calculator)
            if args.device == "cuda"
            else {"backend": "cpu"}
        ),
        "protocol": {
            "device": args.device,
            "timing_fence": (
                "evaluator.fence() brackets every timed region"
                if args.device == "cuda"
                else "none (host execution is synchronous)"
            ),
            "streamed_edges_override": args.streamed_edges,
            "edge_executor": args.edge_executor,
            "spline_nodes": args.spline_nodes,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "stage_runs": args.stage_runs,
            "verify_comparator": args.verify_comparator,
            "ladder_warmups": args.ladder_warmups,
            "ladder_repeats": args.ladder_repeats,
            "steady_state_excludes": [
                "checkpoint extraction",
                "comparator serialization",
                "calculator construction",
                "graph construction and neighbor-list preparation",
                "artifact selection and JIT warmup",
            ],
            "graph_reuse": "one prepared neighbor graph per system, reused by every timed call",
            "note": (
                "the frontend appends 'stress' internally for rank-3 cells, so an "
                "'energy' request still evaluates the backbone virial; the polar "
                "forward composition always runs, while its native reverse runs "
                "only when forces or stress are requested"
            ),
        },
        "model": identity,
        "execution_evidence": execution_evidence,
        "systems": records,
        "ladder": ladder,
        "diagnostics": diagnostics,
        "tier": tier,
        "anchors": anchors,
        "peak_rss_bytes": _peak_rss_bytes(),
        "build_seconds": build_seconds,
    }
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().resolve().write_text(
        json.dumps(record, indent=1, sort_keys=False) + "\n"
    )
    print(f"\nrecord written to {args.output.expanduser().resolve()}")
    print(
        "tier: polar routing "
        f"{tier['polar_family']['resolved_routing']!r} "
        f"({tier['polar_family']['resolution_reason']}); comparator "
        f"{tier['comparator']['execution_backend']!r}/"
        f"{tier['comparator']['node_backend_selected']!r}, "
        f"generated launches "
        f"{tier['comparator']['generated_forward_launch_count']}"
    )
    if tier["generated_node_program"].get("attempted"):
        served = tier["generated_node_program"].get("available")
        print(
            "generated node program: "
            + (
                "available"
                if served
                else f"refused: {tier['generated_node_program']['refusal']}"
            )
        )
    print(
        "anchors (slab_w2 energy+forces): "
        + ", ".join(
            f"{name} {value:.2f} ms"
            for name, value in anchors.items()
            if name != "drift_percent"
        )
    )
    print(
        "anchor drift vs before_systems: "
        + ", ".join(
            f"{name} {value:+.2f}%"
            for name, value in anchors["drift_percent"].items()
            if name != "before_systems"
        )
    )
    if diagnostics is not None:
        print("\ndiagnostics: box probe (3 atoms, identical positions)")
        for entry in diagnostics["box_probe"]:
            print(
                f"  cell {entry['cell_A']:6.1f} A  edges {entry['directed_edges']:4d}  "
                f"backbone {entry['backbone_ms']:.3f} ms"
            )
        print("diagnostics: node ladder (water copies in a 40 A box)")
        for entry in diagnostics["node_ladder"]:
            print(
                f"  atoms {entry['atoms']:3d}  edges {entry['directed_edges']:4d}  "
                f"backbone {entry['backbone_ms']:.3f} ms"
            )
        print("diagnostics: e3 linear backend, 3 atoms")
        for entry in diagnostics["backend_probe"]:
            print(
                f"  {entry['backend']:12s} selected {entry['selected']:12s} "
                f"backbone {entry['backbone_ms']:.3f} ms"
            )
        print("diagnostics: backend boundary (bulk fixture prefix, 40 A box)")
        for entry in diagnostics["backend_boundary"]:
            print(
                f"  atoms {entry['atoms']:3d}  edges {entry['directed_edges']:4d}  "
                f"selected {entry['selected_backend']:12s} "
                f"backbone {entry['backbone_ms']:.3f} ms"
            )
        print(
            "  energy diff vs auto: "
            + ", ".join(
                f"{name} {value:.3e}"
                for name, value in diagnostics["backend_energy_abs_diff"].items()
            )
        )
    atoms.calc = None
    del atoms, evaluator, mh1_evaluator, calculator
    FENCE = None
    gc.collect()
    if symmetrix._kokkos_is_initialized():
        symmetrix._finalize_kokkos()
    return 0


def measure_tier(calculator, comparator_json, args):
    """The execution tier both sides run on, and why it is the only one today.

    Everything the qualified direct path would add is off for this family, and
    the record must say which of those switches is off and on whose decision:
    the family's own declaration for the polar side, and the checkpoint's
    topology for the generated-node-program side.  The artifact probe measures
    the second one instead of asserting it: the MH-1 host node-program
    generator either serves this checkpoint's topology or refuses it by name.
    """

    artifact = {"attempted": False}
    if args.artifact_probe:
        from symmetrix.mace_nonlinear_jit_artifact import (
            MaceNonlinearHostArtifactError,
            prepare_mh1_host_artifact,
        )

        artifact["attempted"] = True
        start = time.perf_counter()
        try:
            result = prepare_mh1_host_artifact(
                comparator_json,
                precision=args.dtype,
                cache_root=args.output.expanduser().resolve().parent / "jit-cache",
            )
            artifact["available"] = True
            artifact["abi"] = result.abi
            artifact["artifact_id"] = result.artifact_id
            artifact["artifact_path"] = str(result.artifact_path)
        except MaceNonlinearHostArtifactError as error:
            artifact["available"] = False
            artifact["refusal"] = str(error)
        artifact["seconds"] = time.perf_counter() - start

    evaluator = calculator.evaluator
    device_artifact = {"attempted": False}
    if getattr(args, "device", "cpu") == "cuda":
        # The device generation the CUDA backend would need: the same contract
        # the host generator takes, rendered for the device node program.  Pure
        # source synthesis, so a refusal here is the generator's, not a GPU's.
        from symmetrix.mace_nonlinear_execution_contract import (
            make_execution_mh1_v4_contract,
            normalize_execution_mh1_v4_contract,
        )
        from symmetrix.mace_nonlinear_jit_codegen import render_jit_mh1_cuda_plugin_v4

        device_artifact["attempted"] = True
        environment = dict(evaluator.device_cuda_environment)
        capability = int(environment.get("compute_capability_code") or 0)
        device_artifact["compute_capability_code"] = capability
        data = json.loads(pathlib.Path(comparator_json).read_text())
        start = time.perf_counter()
        try:
            contract = normalize_execution_mh1_v4_contract(
                make_execution_mh1_v4_contract(data)
            )
            render_jit_mh1_cuda_plugin_v4(contract, capability)
            device_artifact["available"] = True
        except ValueError as error:
            device_artifact["available"] = False
            device_artifact["refusal"] = str(error)
        device_artifact["seconds"] = time.perf_counter() - start
    direct_route = calculator.streamed_edges in ("direct", "direct_streamed")
    return {
        "polar_family": {
            "resolved_routing": calculator.streamed_edges,
            "native_mode": calculator._native_streamed_edges,
            "resolution_reason": calculator.streamed_edges_resolution_reason,
            "evaluation_algorithm": calculator.execution_algorithm,
        },
        "comparator": {
            "class": type(evaluator).__name__,
            "tier": "generic/eager Kokkos node program",
            "comparison_role": (
                "graph-parity control; not a specialized direct-backbone "
                "performance reference"
            ),
            "construction": (
                "raw evaluator built from the serialized JSON, so both sides sit "
                "on the same tier"
            ),
            "execution_backend": evaluator.execution_mh1_execution_backend,
            "node_backend_request": evaluator.execution_mh1_host_node_backend,
            "node_backend_selected": evaluator.selected_execution_mh1_host_node_backend,
            "generated_forward_launch_count": (
                evaluator.execution_mh1_generated_forward_launch_count
            ),
            "host_plugin_ready": evaluator.jit_mh1_host_plugin_ready,
            "supports_streamed_edges": evaluator.supports_streamed_edges,
        },
        "generated_node_program": artifact,
        "device_generated_node_program": device_artifact,
        "features": {
            "edge_streaming": (
                "on: the production frontend selected the specialized direct route"
                if direct_route
                else "off: the production frontend resolved the reference route"
            ),
            "generated_jit_kernels": (
                "on: the admitted polar backbone uses its generated MH-1-style "
                "node program"
                if direct_route
                else "off: no generated direct route was selected"
            ),
            "prepared_fused_routes": (
                "on: the specialized polar route is prepared through the "
                "production calculator"
                if direct_route
                else "off: the reference route is active"
            ),
            "comparator_prepared_artifacts": (
                "off by construction: the comparator is a raw evaluator on the "
                "same graph preparation as the family"
            ),
        },
        "on": {
            "prepared_neighbor_graph": (
                "one prepared neighbor graph per system, prepared once and "
                "reused by every timed call"
            )
        },
        "consequence": (
            "polar timings are production direct-tier numbers"
            if direct_route
            else "polar timings are reference/generic-tier numbers"
        ),
    }


def _backbone_ms(calculator, mace_inputs, atoms, repeats):
    return _median_ms(
        lambda: calculator._compute_mace(mace_inputs, atoms=atoms), repeats, 1
    )["median_ms"]


def _anchor_ms(calculator, repeats=5):
    """A cheap fixed workload the run measures on both sides of the ladder.

    One pinned physical core still shares its execution resources with the SMT
    sibling, so a foreign process on the host moves these numbers.  The anchor
    makes that visible inside the record instead of leaving it to be inferred
    from a suspicious row.
    """

    atoms, _system = load_fixture_system("slab_w2", None)
    atoms.calc = calculator
    for _ in range(2):
        calculator.calculate(atoms, properties=["energy", "forces"])
    return _median_ms(
        lambda: calculator.calculate(atoms, properties=["energy", "forces"]), repeats, 1
    )["median_ms"]


def measure_diagnostics(calculator, args):
    """The controls behind the smallest fixture's number.

    The molecular fixture is three atoms in a 58.8 A vacuum box, and its
    ``us/atom`` is dominated by fixed cost, so the record separates the three
    candidate causes rather than leaving them conflated: the box (the neighbor
    grid and the k-grid scale with cell volume), the node count (the e3 linear
    backend switches path on the sample count), and the graph size (edge
    count).

    The evaluator's own ``use_scalar_backend`` rule selects the scalar e3
    linear path for ``samples <= 4`` on the host, so a graph with four or fewer
    nodes takes a per-instruction kernel path instead of the packed GEMM.  The
    probe measures both paths on the same graph and checks that they agree
    numerically.
    """

    mol, _system = load_fixture_system("mol_h2o", None)
    species = mol.get_chemical_symbols()
    positions = mol.get_positions()
    box = []
    for length in (6.0, 10.0, 58.8):
        atoms = Atoms(species, positions=positions, cell=np.eye(3) * length, pbc=False)
        atoms.calc = calculator
        calculator.calculate(atoms, properties=["energy"])
        mace_inputs = calculator._mace_inputs(atoms)
        box.append(
            {
                "cell_A": length,
                "atoms": len(atoms),
                "directed_edges": len(mace_inputs[3]),
                "backbone_ms": _backbone_ms(
                    calculator, mace_inputs, atoms, args.repeats
                ),
            }
        )

    ladder = []
    for copies in (1, 2, 3, 4, 6, 12):
        expanded = Atoms(
            [symbol for _ in range(copies) for symbol in species],
            positions=np.array(
                [
                    point + np.array([0.0, 0.0, 3.0 * index])
                    for index in range(copies)
                    for point in positions
                ]
            ),
            cell=np.eye(3) * 40.0,
            pbc=False,
        )
        expanded.calc = calculator
        calculator.calculate(expanded, properties=["energy"])
        mace_inputs = calculator._mace_inputs(expanded)
        ladder.append(
            {
                "atoms": len(expanded),
                "directed_edges": len(mace_inputs[3]),
                "backbone_ms": _backbone_ms(
                    calculator, mace_inputs, expanded, args.repeats
                ),
            }
        )

    backend = []
    atoms = Atoms(species, positions=positions, cell=np.eye(3) * 58.8, pbc=False)
    atoms.calc = calculator
    calculator.calculate(atoms, properties=["energy"])
    mace_inputs = calculator._mace_inputs(atoms)
    energies = {}
    for name in ("auto", "scalar", "packed_gemm"):
        calculator.evaluator.set_e3_linear_backend(name)
        calculator.evaluator.set_retain_layer_features(False)
        milliseconds = _backbone_ms(calculator, mace_inputs, atoms, args.repeats)
        energies[name] = float(np.sum(np.asarray(calculator.evaluator.node_energies)))
        calculator.evaluator.set_retain_layer_features(True)
        backend.append(
            {
                "backend": name,
                "requested": calculator.evaluator.e3_linear_backend,
                "selected": calculator.evaluator.selected_e3_linear_backend(len(atoms)),
                "atoms": len(atoms),
                "backbone_ms": milliseconds,
            }
        )
    calculator.evaluator.set_e3_linear_backend("auto")

    # The exact boundary of the auto selection: the bulk fixture's first k
    # atoms, so an even atom count below the threshold is reachable.
    bulk, _system = load_fixture_system("bulk_w4", None)
    boundary = []
    for count in (1, 2, 3, 4, 5, 6, 8, 12):
        prefix = Atoms(
            bulk.get_chemical_symbols()[:count],
            positions=bulk.get_positions()[:count],
            cell=np.eye(3) * 40.0,
            pbc=False,
        )
        prefix.calc = calculator
        calculator.calculate(prefix, properties=["energy"])
        prefix_inputs = calculator._mace_inputs(prefix)
        calculator.evaluator.set_retain_layer_features(False)
        milliseconds = _backbone_ms(calculator, prefix_inputs, prefix, args.repeats)
        calculator.evaluator.set_retain_layer_features(True)
        boundary.append(
            {
                "atoms": count,
                "directed_edges": len(prefix_inputs[3]),
                "selected_backend": calculator.evaluator.selected_e3_linear_backend(
                    count
                ),
                "backbone_ms": milliseconds,
            }
        )
    return {
        "box_probe": box,
        "node_ladder": ladder,
        "backend_probe": backend,
        "backend_boundary": boundary,
        "backend_energy_abs_diff": {
            name: abs(energies[name] - energies["auto"]) for name in energies
        },
        "note": (
            "use_scalar_backend(samples) selects the scalar e3 linear path for "
            "samples <= 4 on the host (libsymmetrix/source/e3nn_kokkos.cpp); "
            "the boundary ladder locates the switch exactly, and the node "
            "ladder shows its cost"
        ),
    }


def _print_system(record, ladder=False):
    label = "ladder " if ladder else "system "
    capacity = record["capacity"]
    print(
        f"\n{label}{record['system']}: {record['atoms']} atoms, "
        f"{record['directed_edges']} directed edges "
        f"({record['directed_edges_model_cutoff']} at the model cutoff), "
        f"cutoff {record['model_cutoff_A']} A, "
        f"skin {record['neighbor_skin_A']} A, n_k {capacity['n_k']}, "
        f"table {capacity['table_bytes']} B ({capacity['correction_mode']})"
    )
    has_comparator = bool(record["mh1_generic_comparator_ms"])
    header = f"  {'mode':24s} {'polar ms':>10s} {'polar us/atom':>14s}"
    if has_comparator:
        header += f" {'generic ms':>10s} {'generic us/atom':>15s} {'delta':>9s}"
    print(header)
    for key in ("energy", "energy_forces", "energy_forces_stress"):
        line = (
            f"  {key:24s} {record['polar_ms'][key]['median_ms']:10.3f} "
            f"{record['us_per_atom'][key]:14.2f}"
        )
        if has_comparator:
            line += (
                f" {record['mh1_generic_comparator_ms'][key]['median_ms']:10.3f} "
                f"{record['us_per_atom_mh1_generic_comparator'][key]:13.2f} "
                f"{record['polar_vs_generic_comparator_percent'][key]:8.1f}%"
            )
        print(line)


if __name__ == "__main__":
    raise SystemExit(main())
