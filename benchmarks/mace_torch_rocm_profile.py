"""Profile one upstream MACE inference through e3nn or cuEquivariance."""

import argparse
import contextlib
import ctypes
import ctypes.util
import hashlib
import importlib.util
import json
import pathlib
import time

import numpy as np
import torch
from ase.build import bulk
from mace.calculators.mace import MACECalculator


def _load_graph_fingerprint():
    path = pathlib.Path(__file__).resolve().with_name("profile_graph_fingerprint.py")
    specification = importlib.util.spec_from_file_location(
        "symmetrix_profile_graph_fingerprint", path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not load graph fingerprint helper: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.canonical_graph_fingerprint, module.require_graph_fingerprint


canonical_graph_fingerprint, require_graph_fingerprint = _load_graph_fingerprint()

_MODEL_SHA256 = {
    "mh0": "d62ff8f293664e6556cfa49364b28ee72a70cf1e6150f3c111a578397fed609d",
    "mh1": "a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47",
}
_ROCTX_RANGE = "symmetrix:pytorch_mace:measured_inference"
_TORCH_FORWARD_RANGE = "symmetrix:pytorch_mace:forward"
_EXPECTED_GRAPH_FINGERPRINT = {
    "sha256": "a6990939929e747e577bbd8b65c8954a13f712b972e40cad9869f78f54a521b7",
    "input_order_edge_records_sha256": (
        "ef74d9e1e6030a13983bd5789c90cd7f12ff2af2ed1ba55e68f5d567722fd415"
    ),
}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_roctx():
    candidates = [
        "/opt/rocm/core-7.14/lib/librocprofiler-sdk-roctx.so",
        "/opt/rocm/lib/librocprofiler-sdk-roctx.so",
        "/opt/rocm/lib64/librocprofiler-sdk-roctx.so",
        ctypes.util.find_library("rocprofiler-sdk-roctx"),
        ctypes.util.find_library("roctx64"),
        "/opt/rocm/lib/libroctx64.so",
        "/opt/rocm/lib64/libroctx64.so",
        "/opt/rocm/core-7.14/lib/libroctx64.so",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            library = ctypes.CDLL(candidate)
        except OSError:
            continue
        library.roctxRangePushA.argtypes = [ctypes.c_char_p]
        library.roctxRangePushA.restype = ctypes.c_int
        library.roctxRangePop.argtypes = []
        library.roctxRangePop.restype = ctypes.c_int
        profiler_control = hasattr(library, "roctxProfilerResume") and hasattr(
            library, "roctxProfilerPause"
        )
        if profiler_control:
            library.roctxProfilerResume.argtypes = [ctypes.c_uint64]
            library.roctxProfilerResume.restype = ctypes.c_int
            library.roctxProfilerPause.argtypes = [ctypes.c_uint64]
            library.roctxProfilerPause.restype = ctypes.c_int
        return library, candidate, profiler_control
    return None, None, False


@contextlib.contextmanager
def _roctx_range(library, name, profiler_control):
    if library is None:
        yield
        return
    if profiler_control:
        result = library.roctxProfilerResume(0)
        if result != 0:
            raise RuntimeError(f"roctxProfilerResume failed with status {result}")
    level = library.roctxRangePushA(name.encode("ascii"))
    if level < 0:
        pause_result = library.roctxProfilerPause(0) if profiler_control else 0
        if pause_result != 0:
            raise RuntimeError(
                f"roctxRangePushA failed with level {level}; "
                f"roctxProfilerPause also failed with status {pause_result}"
            )
        raise RuntimeError(f"roctxRangePushA failed with level {level}")
    try:
        yield
    finally:
        popped_level = library.roctxRangePop()
        pause_result = library.roctxProfilerPause(0) if profiler_control else 0
        if popped_level < 0:
            raise RuntimeError(f"roctxRangePop failed with level {popped_level}")
        if pause_result != 0:
            raise RuntimeError(f"roctxProfilerPause failed with status {pause_result}")


def _default_model_path(generation):
    return (
        pathlib.Path.home()
        / ".cache"
        / "symmetrix"
        / "test-models"
        / f"mace-{generation.replace('mh', 'mh-')}.model"
    )


def _model_metadata(model):
    metadata = {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }
    for name in ("num_interactions", "num_elements", "r_max"):
        value = getattr(model, name, None)
        if value is None:
            continue
        if torch.is_tensor(value) and value.numel() == 1:
            value = value.item()
        if isinstance(value, (bool, int, float, str)):
            metadata[name] = value
    return metadata


@contextlib.contextmanager
def _module_record_ranges(model):
    """Expose eager MACE module names in a diagnostic Kineto trace."""
    active = []
    handles = []
    metadata = {
        "registered_module_count": 0,
        "skipped_module_count": 0,
        "skipped_module_names": [],
    }

    def make_pre_hook(name):
        def pre_hook(module, inputs):
            del inputs
            record = torch.autograd.profiler.record_function(f"mace.module:{name}")
            record.__enter__()
            active.append((module, record))

        return pre_hook

    def post_hook(module, inputs, output):
        del inputs, output
        if not active:
            return
        active_module, record = active.pop()
        if active_module is not module:
            raise RuntimeError("MACE module profiling ranges are not properly nested")
        record.__exit__(None, None, None)

    try:
        for name, module in model.named_modules():
            if not name:
                continue
            pre_handle = None
            try:
                pre_handle = module.register_forward_pre_hook(make_pre_hook(name))
                post_handle = module.register_forward_hook(post_hook, always_call=True)
            except RuntimeError:
                if pre_handle is not None:
                    pre_handle.remove()
                metadata["skipped_module_names"].append(name)
                continue
            handles.extend((pre_handle, post_handle))
            metadata["registered_module_count"] += 1
        metadata["skipped_module_count"] = len(metadata["skipped_module_names"])
        yield metadata
    finally:
        for handle in handles:
            handle.remove()
        while active:
            _, record = active.pop()
            record.__exit__(None, None, None)


def _event_time(event, preferred, legacy):
    value = getattr(event, preferred, None)
    if value is None:
        value = getattr(event, legacy, 0.0)
    return float(value)


def _operator_summary(profiler):
    operators = []
    for event in profiler.key_averages():
        self_cpu_time_us = float(event.self_cpu_time_total)
        if self_cpu_time_us <= 0.0:
            continue
        operators.append(
            {
                "key": event.key,
                "cpu_calls": int(event.count),
                "self_cpu_time_us": self_cpu_time_us,
                "self_device_time_us": _event_time(
                    event, "self_device_time_total", "self_cuda_time_total"
                ),
                "device_total_time_us": _event_time(
                    event, "device_time_total", "cuda_time_total"
                ),
            }
        )
    return sorted(operators, key=lambda operator: operator["key"])


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Profile exactly one synchronized e3nn/FP32 inference on the official "
            "864-atom AlN MH0 or MH1 workload."
        )
    )
    parser.add_argument("generation", choices=tuple(_MODEL_SHA256))
    parser.add_argument("--backend", choices=("e3nn", "cueq"), default="e3nn")
    parser.add_argument(
        "--model",
        type=pathlib.Path,
        help="Checkpoint path (defaults to the Symmetrix test-model cache)",
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--torch-trace",
        type=pathlib.Path,
        help=(
            "Export a one-step Kineto Chrome trace after the timed step; this is "
            "diagnostic and is excluded from benchmark timing"
        ),
    )
    args = parser.parse_args()
    if args.warmups < 0:
        parser.error("--warmups must be nonnegative")
    if args.model is None:
        args.model = _default_model_path(args.generation)
    return parser, args


def main():
    parser, args = _parse_args()
    if not torch.cuda.is_available():
        parser.error("a CUDA or ROCm PyTorch device is not available")

    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        parser.error(f"checkpoint does not exist: {model_path}")
    model_sha256 = _sha256(model_path)
    expected_sha256 = _MODEL_SHA256[args.generation]
    if model_sha256 != expected_sha256:
        parser.error(
            f"{args.generation} checkpoint SHA-256 is {model_sha256}, expected "
            f"{expected_sha256}"
        )

    calculator = MACECalculator(
        model_paths=model_path,
        device="cuda",
        default_dtype="float32",
        enable_cueq=args.backend == "cueq",
        head="omat_pbe",
    )
    if bool(getattr(calculator, "enable_cueq", args.backend == "cueq")) != (
        args.backend == "cueq"
    ):
        raise RuntimeError(f"MACECalculator did not select {args.backend}")
    model = calculator.models[0]
    model.eval()

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982).repeat((6, 6, 6))
    if len(atoms) != 864:
        raise RuntimeError(f"expected 864 atoms, constructed {len(atoms)}")
    batch_base = calculator._atoms_to_batch(atoms)
    edge_index = batch_base["edge_index"].detach().cpu().numpy()
    unit_shifts = batch_base["unit_shifts"].detach().cpu().numpy()
    graph_fingerprint = canonical_graph_fingerprint(
        atoms.numbers,
        atoms.positions,
        atoms.cell.array,
        edge_index[0],
        edge_index[1],
        unit_shifts,
    )
    require_graph_fingerprint(graph_fingerprint, _EXPECTED_GRAPH_FINGERPRINT)
    model_dtype = next(model.parameters()).dtype
    for key in batch_base.keys:
        value = batch_base[key]
        if torch.is_tensor(value) and torch.is_floating_point(value):
            batch_base[key] = value.to(dtype=model_dtype)

    def prepare_input():
        return calculator._clone_batch(batch_base).to_dict()

    output = None
    for _ in range(args.warmups):
        output = model(
            prepare_input(),
            compute_stress=False,
            training=False,
            compute_edge_forces=False,
            compute_atomic_stresses=False,
        )
        torch.cuda.synchronize()

    measured_input = prepare_input()
    torch.cuda.synchronize()
    roctx, roctx_library, roctx_profiler_control = _load_roctx()
    start = time.perf_counter()
    with (
        _roctx_range(roctx, _ROCTX_RANGE, roctx_profiler_control),
        torch.autograd.profiler.record_function(_TORCH_FORWARD_RANGE),
    ):
        output = model(
            measured_input,
            compute_stress=False,
            training=False,
            compute_edge_forces=False,
            compute_atomic_stresses=False,
        )
        torch.cuda.synchronize()
    elapsed_ms = 1000.0 * (time.perf_counter() - start)
    measured_output = output

    torch_profile = None
    if args.torch_trace is not None:
        trace_path = args.torch_trace.expanduser().resolve()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        profiled_input = prepare_input()
        torch.cuda.synchronize()
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        with (
            torch.profiler.profile(
                activities=activities,
                with_modules=True,
            ) as profiler,
            _module_record_ranges(model) as module_range_metadata,
            torch.autograd.profiler.record_function(_TORCH_FORWARD_RANGE),
        ):
            diagnostic_output = model(
                profiled_input,
                compute_stress=False,
                training=False,
                compute_edge_forces=False,
                compute_atomic_stresses=False,
            )
            torch.cuda.synchronize()
        del diagnostic_output
        profiler.export_chrome_trace(str(trace_path))
        operators = _operator_summary(profiler)
        torch_profile = {
            "kind": "diagnostic Kineto profile; not benchmark timing",
            "trace_path": str(trace_path),
            "activities": ["CPU", "CUDA (ROCm device activity)"],
            "measured_steps": 1,
            "module_ranges": {
                "prefix": "mace.module:<named_modules path>",
                **module_range_metadata,
            },
            "forward_range": _TORCH_FORWARD_RANGE,
            "aggregate": {
                "kernel_device_time_us": sum(
                    operator["self_device_time_us"] for operator in operators
                ),
            },
            "operators": operators,
        }

    forces = measured_output["forces"].detach().cpu().numpy()
    energy = measured_output["energy"].detach().cpu().reshape(-1)
    device = torch.cuda.current_device()
    report = {
        "workload": {
            "generation": args.generation,
            "head": calculator.head,
            "backend": args.backend,
            "dtype": str(model_dtype),
            "compute_stress": False,
            "compute_edge_forces": False,
            "compute_atomic_stresses": False,
        },
        "model": {
            "path": str(model_path),
            "size_bytes": model_path.stat().st_size,
            "sha256": model_sha256,
            **_model_metadata(model),
        },
        "graph": {
            "structure": "wurtzite AlN",
            "lattice_a_A": 3.112,
            "lattice_c_A": 4.982,
            "supercell_repeat": [6, 6, 6],
            "atoms": len(atoms),
            "al_atoms": int(np.count_nonzero(atoms.numbers == 13)),
            "n_atoms": int(np.count_nonzero(atoms.numbers == 7)),
            "directed_edges": int(batch_base["edge_index"].shape[1]),
            "canonical_fingerprint": graph_fingerprint,
        },
        "profile": {
            "warmups": args.warmups,
            "measured_steps": 1,
            "elapsed_ms": elapsed_ms,
            "roctx_range": _ROCTX_RANGE if roctx is not None else None,
            "roctx_library": roctx_library,
            "roctx_profiler_control": roctx_profiler_control,
            "timing_excludes_torch_diagnostic_profile": True,
        },
        "torch_diagnostic_profile": torch_profile,
        "numerics": {
            "energy_eV": float(energy[0]),
            "force_max_abs_eV_per_A": float(np.max(np.abs(forces))),
            "force_l2_eV_per_A": float(np.linalg.norm(forces)),
            "force_sum_eV_per_A": forces.sum(axis=0).tolist(),
        },
        "runtime": {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "torch_hip_version": torch.version.hip,
            "device_index": device,
            "device_name": torch.cuda.get_device_name(device),
        },
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
