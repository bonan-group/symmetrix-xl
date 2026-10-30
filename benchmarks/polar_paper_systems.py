"""Deterministic systems representative of the MACE-POLAR-1 paper.

The water boxes reproduce the reported atom counts, cell lengths, and number
density, not an unpublished equilibrated trajectory frame.  The X23 benzene
cell is the published benchmark structure from
https://github.com/ddmms/x23/blob/main/data/Benzene.cif.
"""

from __future__ import annotations

import hashlib

import numpy as np
from ase import Atoms


WATER_POSITIONS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.96, 0.0, 0.0],
        [-0.2403648, 0.9294217, 0.0],
    ],
    dtype=np.float64,
)


def _rotation(index: int) -> np.ndarray:
    """Return a deterministic, well-conditioned orientation for one molecule."""

    a = (index * 0.7548776662466927) % 1.0 * 2.0 * np.pi
    b = (index * 0.5698402909980532) % 1.0 * 2.0 * np.pi
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])
    return rz @ ry


def water_box(
    grid: tuple[int, int, int],
    cell_length: float | tuple[float, float, float],
    *,
    molecules: int | None = None,
    pbc: bool,
    padding: float = 0.0,
) -> Atoms:
    """Place oriented waters on a regular grid without random state."""

    lengths = np.broadcast_to(np.asarray(cell_length, dtype=np.float64), (3,)).copy()
    sites = np.array(
        [
            ((i + 0.5) / grid[0], (j + 0.5) / grid[1], (k + 0.5) / grid[2])
            for i in range(grid[0])
            for j in range(grid[1])
            for k in range(grid[2])
        ],
        dtype=np.float64,
    )
    count = len(sites) if molecules is None else molecules
    if count <= 0 or count > len(sites):
        raise ValueError("molecules must be between one and the grid capacity")
    if count != len(sites):
        removed = set(np.linspace(0, len(sites) - 1, len(sites) - count, dtype=int))
        sites = np.array([site for i, site in enumerate(sites) if i not in removed])
    centres = sites * lengths
    positions = np.concatenate(
        [WATER_POSITIONS @ _rotation(i).T + centre for i, centre in enumerate(centres)]
    )
    if padding:
        positions += padding
        lengths += 2.0 * padding
    atoms = Atoms("OH2" * count, positions=positions, cell=np.diag(lengths), pbc=pbc)
    return atoms


def _water_monomer() -> Atoms:
    return Atoms(
        "OH2",
        positions=WATER_POSITIONS + 5.0,
        cell=np.diag([10.0, 10.0, 10.0]),
        pbc=False,
    )


_BENZENE_FRACTIONAL = np.array(
    [
        [0.9399657483085251, 0.1406387314225053, 0.993606208516887],
        [0.8620300419485792, 0.04524588853503185, 0.12612991042584434],
        [0.0774863193504736, 0.09524678874734607, 0.8675339632892805],
        [0.9225136847090664, 0.9047532091295117, 0.132466046989721],
        [0.13796995805142084, 0.9547541114649682, 0.873870093979442],
        [0.060034258457374826, 0.8593612696390658, 0.0063938017621145374],
        [0.5600342516914749, 0.6406387123142251, 0.993606208516887],
        [0.9399657415426252, 0.35936127388535033, 0.4936061938325991],
        [0.5600342381596751, 0.8593612675159236, 0.49360622613803234],
        [0.4399657537212449, 0.3593612845010616, 0.006393791483113069],
        [0.060034251691474964, 0.6406387261146497, 0.5063938046989721],
        [0.4399657564276048, 0.14063872929936308, 0.5063937826725404],
        [0.637969953991881, 0.5452458853503186, 0.126129906020558],
        [0.862030046008119, 0.45475410828025475, 0.6261299118942731],
        [0.6379699675236806, 0.9547541082802548, 0.6261299177679883],
        [0.36203004600811906, 0.45475411146496814, 0.8738700998531571],
        [0.13796995399188092, 0.5452458917197452, 0.3738700910425844],
        [0.3620300324763193, 0.04524588853503185, 0.3738700866372981],
        [0.4225136874154262, 0.5952467685774947, 0.8675339750367107],
        [0.0774863098782138, 0.4047532176220807, 0.3675339559471366],
        [0.4225136725304465, 0.904753212314225, 0.3675339838472834],
        [0.5774863166441138, 0.4047532292993631, 0.1324660234948605],
        [0.9225136860622463, 0.5952467802547771, 0.6324660411160059],
        [0.5774863247631935, 0.09524678556263269, 0.6324660279001468],
        [0.8936229648173207, 0.2504891995753715, 0.9883324317180617],
        [0.7536779891745602, 0.07929086305732484, 0.2240732731277533],
        [0.13725856698240865, 0.16915040233545647, 0.7630336534508076],
        [0.8627414330175914, 0.8308495966029724, 0.2369663509544787],
        [0.24632201082543978, 0.9207091358811041, 0.7759267312775331],
        [0.10637703653585925, 0.7495107993630573, 0.01166757268722467],
        [0.6063770338294993, 0.7504892101910828, 0.9883324302496329],
        [0.8936229607577808, 0.24951079511677282, 0.488332422907489],
        [0.6063770405953992, 0.7495107887473461, 0.4883324331864905],
        [0.39362296617050063, 0.24951078768577495, 0.011667574155653452],
        [0.10637703924221922, 0.7504892027600849, 0.5116675814977973],
        [0.39362295940460085, 0.25048920912951167, 0.5116675712187959],
        [0.74632199864682, 0.5792908630573248, 0.22407326284875184],
        [0.7536779945872801, 0.42070913057324844, 0.7240732628487518],
        [0.7463219999999999, 0.9207091369426751, 0.7240732569750368],
        [0.25367800135317997, 0.4207091348195329, 0.7759267415565345],
        [0.2463220054127199, 0.5792908683651805, 0.2759267415565345],
        [0.253678, 0.0792908609341826, 0.27592674743024964],
        [0.3627414330175913, 0.6691503970276009, 0.7630336475770926],
        [0.13725856968876862, 0.3308496050955414, 0.26303364317180616],
        [0.3627414235453315, 0.8308495912951168, 0.2630336417033774],
        [0.6372585669824086, 0.3308495997876858, 0.23696635535976504],
        [0.8627414303112313, 0.6691503938428875, 0.7369663597650514],
        [0.6372585764546684, 0.169150406581741, 0.736966362701909],
    ],
    dtype=np.float64,
)


def _x23_benzene() -> Atoms:
    cell = np.diag([7.39, 9.42, 6.81])
    return Atoms(
        ["C"] * 24 + ["H"] * 24,
        scaled_positions=_BENZENE_FRACTIONAL,
        cell=cell,
        pbc=True,
    )


SYSTEMS = {
    "molecule_h2o": {
        "builder": _water_monomer,
        "kind": "exact_reconstruction",
        "paper_regime": "small isolated molecule",
    },
    "finite_cluster_384": {
        "builder": lambda: water_box(
            (4, 4, 8), (12.4, 12.4, 24.8), pbc=False, padding=4.0
        ),
        "kind": "size_density_proxy",
        "paper_regime": "259-584 atom non-periodic active-site complexes",
    },
    "x23_benzene": {
        "builder": _x23_benzene,
        "kind": "published_structure",
        "paper_regime": "periodic X23 molecular crystal",
    },
    "redox_water64": {
        "builder": lambda: water_box((4, 4, 4), 12.4, pbc=True),
        "kind": "size_density_proxy",
        "paper_regime": "64-water, 12.4 A periodic redox cell",
    },
    "liquid_water333": {
        "builder": lambda: water_box((7, 7, 7), 21.5383813974, molecules=333, pbc=True),
        "kind": "size_density_proxy",
        "paper_regime": "333-water periodic liquid at 0.997 g/cm^3",
    },
}

# Scaling-only buckets extend beyond the paper's largest reported liquid box.
# Keep them opt-in so the default paper comparison remains bounded.
DEFAULT_SYSTEM_NAMES = tuple(SYSTEMS)
SYSTEMS.update(
    {
        "liquid_water1000": {
            "builder": lambda: water_box((10, 10, 10), 31.07408279771176, pbc=True),
            "kind": "scaling_proxy",
            "paper_regime": "1000-water periodic liquid at 0.997 g/cm^3",
        },
        "liquid_water1728": {
            "builder": lambda: water_box((12, 12, 12), 37.28889935725412, pbc=True),
            "kind": "scaling_proxy",
            "paper_regime": "1728-water periodic liquid at 0.997 g/cm^3",
        },
        "liquid_water2744": {
            "builder": lambda: water_box((14, 14, 14), 43.50371591679647, pbc=True),
            "kind": "scaling_proxy",
            "paper_regime": "2744-water periodic liquid at 0.997 g/cm^3",
        },
        "liquid_water4096": {
            "builder": lambda: water_box((16, 16, 16), 49.71853247633882, pbc=True),
            "kind": "scaling_proxy",
            "paper_regime": "4096-water periodic liquid at 0.997 g/cm^3",
        },
    }
)


def default_system_names() -> tuple[str, ...]:
    return DEFAULT_SYSTEM_NAMES


def system_names() -> tuple[str, ...]:
    return tuple(SYSTEMS)


def build_system(name: str) -> tuple[Atoms, dict[str, str]]:
    try:
        specification = SYSTEMS[name]
    except KeyError as error:
        raise ValueError(
            f"unknown system {name!r}; choose from {', '.join(SYSTEMS)}"
        ) from error
    atoms = specification["builder"]()
    atoms.info.update(
        charge=0.0,
        spin=1.0,
        total_charge=0.0,
        total_spin=1.0,
        external_field=np.zeros(3, dtype=np.float64),
        fermi_level=0.0,
    )
    return atoms, {
        "kind": specification["kind"],
        "paper_regime": specification["paper_regime"],
    }


def system_sha256(atoms: Atoms) -> str:
    digest = hashlib.sha256()
    for value in (
        np.asarray(atoms.numbers, dtype="<i8"),
        np.asarray(atoms.positions, dtype="<f8"),
        np.asarray(atoms.cell, dtype="<f8"),
        np.asarray(atoms.pbc, dtype=np.uint8),
    ):
        digest.update(value.tobytes(order="C"))
    for key in (
        "charge",
        "spin",
        "total_charge",
        "total_spin",
        "external_field",
        "fermi_level",
    ):
        digest.update(np.asarray(atoms.info[key], dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()
