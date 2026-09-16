"""Pair generated MH1 NVRTC and nvcc evaluations in one CUDA process."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time


def _bootstrap():
    source_root = Path(os.environ["SYMMETRIX_SOURCE_ROOT"]).resolve()
    extension = Path(os.environ["SYMMETRIX_EXTENSION"]).resolve()
    package_dir = source_root / "symmetrix/source/symmetrix"
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


_bootstrap()

import numpy as np  # noqa: E402 - requires the bootstrap above
from ase.build import bulk  # noqa: E402
from symmetrix import Symmetrix  # noqa: E402


def _calculator(model, backend, cache_root):
    os.environ["SYMMETRIX_JIT_CUDA_JIT_BACKEND"] = backend
    os.environ["SYMMETRIX_JIT_CACHE"] = str(cache_root / backend)
    calculator = Symmetrix(
        model,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="factorized",
        jit="required",
    )
    if calculator.jit_compiler_backend != backend:
        raise RuntimeError(
            f"requested {backend}, loaded {calculator.jit_compiler_backend}"
        )
    return calculator


def _prepared_evaluation(calculator, atoms):
    inputs = calculator._mace_inputs(atoms)
    generation = calculator.evaluator._prepare_factorized_graph(*inputs[:5])
    displacement = np.ascontiguousarray(inputs[5]).reshape(-1)
    distance = np.ascontiguousarray(inputs[6])

    def evaluate():
        calculator.evaluator._compute_prepared_factorized(
            generation, displacement, distance
        )

    return evaluate, inputs


def _time(evaluate, fence):
    fence()
    start = time.perf_counter_ns()
    evaluate()
    fence()
    return (time.perf_counter_ns() - start) / 1e6


def _summary(samples):
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--sizes", default="6,8")
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument(
        "--construction-order",
        choices=("nvrtc-first", "nvcc-first"),
        default="nvrtc-first",
        help="Run both values in separate processes before comparing backends.",
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sizes = [int(value) for value in args.sizes.split(",")]
    if args.warmups < 0 or args.pairs < 1 or any(size < 1 for size in sizes):
        parser.error("sizes and pairs must be positive; warmups must be nonnegative")

    calculators = {}
    construction_order = (
        ("nvrtc", "nvcc")
        if args.construction_order == "nvrtc-first"
        else ("nvcc", "nvrtc")
    )
    for backend in construction_order:
        calculators[backend] = _calculator(args.model, backend, args.cache_root)
    nvrtc = calculators["nvrtc"]
    nvcc = calculators["nvcc"]
    results = []
    for size in sizes:
        atoms = bulk("Si", "diamond", a=5.43).repeat((size,) * 3)
        evaluations = {}
        inputs = {}
        for name, calculator in (("nvrtc", nvrtc), ("nvcc", nvcc)):
            evaluations[name], inputs[name] = _prepared_evaluation(calculator, atoms)
        for _ in range(args.warmups):
            for name, calculator in (("nvrtc", nvrtc), ("nvcc", nvcc)):
                _time(evaluations[name], calculator.evaluator.fence)

        samples = {"nvrtc": [], "nvcc": []}
        ratios = []
        for pair in range(args.pairs):
            order = ("nvrtc", "nvcc") if pair % 2 == 0 else ("nvcc", "nvrtc")
            pair_samples = {}
            for name in order:
                calculator = nvrtc if name == "nvrtc" else nvcc
                pair_samples[name] = _time(
                    evaluations[name], calculator.evaluator.fence
                )
                samples[name].append(pair_samples[name])
            ratios.append(pair_samples["nvrtc"] / pair_samples["nvcc"])

        outputs = {}
        for name, calculator in (("nvrtc", nvrtc), ("nvcc", nvcc)):
            evaluations[name]()
            calculator.evaluator.fence()
            outputs[name] = calculator._collect_mace_results(atoms, inputs[name])
        results.append(
            {
                "atoms": len(atoms),
                "directed_edges": len(inputs["nvrtc"][6]),
                "nvrtc": _summary(samples["nvrtc"]),
                "nvcc": _summary(samples["nvcc"]),
                "paired_ratio": _summary(ratios),
                "median_delta_percent": 100.0 * (statistics.median(ratios) - 1.0),
                "energy_exact": bool(
                    outputs["nvrtc"]["energy"] == outputs["nvcc"]["energy"]
                ),
                "forces_exact": bool(
                    np.array_equal(
                        outputs["nvrtc"]["forces"], outputs["nvcc"]["forces"]
                    )
                ),
                "stress_exact": bool(
                    np.array_equal(
                        outputs["nvrtc"]["stress"], outputs["nvcc"]["stress"]
                    )
                ),
            }
        )

    record = {
        "method": (
            "diagnostic single-process alternating ABBA pairs with a fence per "
            "sample; backend comparison requires both construction orders"
        ),
        "construction_order": list(construction_order),
        "model": str(args.model.resolve()),
        "warmups_per_backend": args.warmups,
        "pairs": args.pairs,
        "compiler_backends": {
            "nvrtc": nvrtc.jit_compiler_backend,
            "nvcc": nvcc.jit_compiler_backend,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
