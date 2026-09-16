"""Generate the qualified 2048-atom LAMMPS MACE-OMAT-0 NVE workload."""

import argparse
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.io import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--streamed-edges",
        choices=("all_interactions", "factorized"),
        default="factorized",
    )
    parser.add_argument("--factorized-device-artifact", type=Path)
    args = parser.parse_args()

    if args.jit_device_artifact is not None and args.streamed_edges != "factorized":
        parser.error(
            "--factorized-device-artifact requires --streamed-edges factorized"
        )
    artifact_option = (
        f" jit_device_artifact {args.jit_device_artifact.resolve()}"
        if args.jit_device_artifact is not None
        else ""
    )

    args.output.mkdir(parents=True, exist_ok=True)
    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982).repeat((8, 8, 8))
    rng = np.random.default_rng(20260808)
    atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)
    data_path = args.output / "aln-2048.data"
    write(
        data_path,
        atoms,
        format="lammps-data",
        atom_style="atomic",
        masses=True,
        specorder=["Al", "N"],
    )

    input_path = args.output / f"in.omat0-{args.streamed_edges}"
    force_path = args.output / f"forces-{args.streamed_edges}.dump"
    input_path.write_text(
        f"""units metal
atom_style atomic
atom_modify map yes sort 0 0
boundary p p p
newton on
read_data {data_path.resolve()}

neighbor 0.5 bin
neigh_modify delay 0 every 1 check yes
pair_style symmetrix/mace/float32/kk no_domain_decomposition streamed_edges {args.streamed_edges}{artifact_option}
pair_coeff * * {args.model.resolve()} Al N

velocity all create 300.0 20260808 mom yes rot no dist gaussian
timestep 0.001
fix integration all nve
thermo 20
thermo_style custom step atoms temp pe etotal
timer full

run 0
write_dump all custom {force_path.resolve()} id type fx fy fz modify sort id
run 20
"""
    )
    print(input_path)


if __name__ == "__main__":
    main()
