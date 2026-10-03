"""Compare Symmetrix MD on the CPU and on the Apple Metal GPU.

Runs the same NVT trajectory with ``metal=True`` and with the CPU evaluator,
then reports throughput in us/atom/step and the agreement of the two runs.
The checkpoint is downloaded on first use and converted by the calculator.

    python benchmarks/metal_md_demo.py --repeat 6 --steps 200
"""

import argparse
import time
import urllib.request
from pathlib import Path

import numpy as np
from ase import units
from ase.build import bulk
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.neighborlist import neighbor_list

from symmetrix import Symmetrix

MODEL_URL = (
    "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/"
    "mace_agnesi_medium.model"
)


def run(calculator, atoms, steps):
    atoms = atoms.copy()
    atoms.calc = calculator
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=600, force_temp=True, rng=np.random.RandomState(1)
    )
    Stationary(atoms)
    dynamics = NVTBerendsen(
        atoms, 1.0 * units.fs, temperature_K=600, taut=100 * units.fs
    )
    dynamics.run(5)  # compilation, graph construction, and warmup
    start = time.perf_counter()
    dynamics.run(steps)
    elapsed = time.perf_counter() - start
    return atoms, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="mace_agnesi_medium.model")
    parser.add_argument(
        "--repeat", type=int, default=6, help="NaCl cubic cells per side"
    )
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    model = Path(args.model)
    if not model.exists():
        print(f"downloading {MODEL_URL}")
        urllib.request.urlretrieve(MODEL_URL, model)

    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((args.repeat,) * 3)
    atoms.rattle(0.05, seed=1)
    results = {}
    for name, extra in (("CPU", {}), ("Metal", {"metal": True})):
        calculator = Symmetrix(str(model), dtype="float32", **extra)
        static = atoms.copy()
        static.calc = calculator
        energy, forces = static.get_potential_energy(), static.get_forces()
        _, elapsed = run(calculator, atoms, args.steps)
        results[name] = (energy, forces, elapsed)
        if name == "Metal":
            print(
                f"Metal device: {calculator.metal_device}, GPU stages: "
                f"{', '.join(calculator.metal_stages)}"
            )
    cutoff = float(calculator.cutoff)
    skin = float(calculator.neighbor_skin)
    edges = len(neighbor_list("i", atoms, cutoff + skin))
    print(
        f"{len(atoms)} atoms, cutoff {cutoff:.2f} A, skin {skin:.2f} A "
        f"(effective {cutoff + skin:.2f} A, {edges} directed edges), "
        f"{args.steps} steps, FP32"
    )
    labels = {"CPU": "CPU (Serial host, one core)", "Metal": "Metal GPU"}
    for name, (_, _, elapsed) in results.items():
        per_atom = elapsed / args.steps / len(atoms) * 1e6
        print(f"  {labels[name]:28s} {per_atom:8.2f} us/atom/step")
    (cpu_energy, cpu_forces, _), (gpu_energy, gpu_forces, _) = results.values()
    print(
        f"  starting structure: |dE| {abs(cpu_energy - gpu_energy) / len(atoms):.1e} "
        f"eV/atom, max |dF| {np.abs(cpu_forces - gpu_forces).max():.1e} eV/A"
    )


if __name__ == "__main__":
    main()
