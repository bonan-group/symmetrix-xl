"""Temporary generated OAM host-plugin fixture support.

The fixture intentionally compiles through the production artifact preparation
path.  Callers own ``cache_dir`` (normally pytest's ``tmp_path``), so no
checked-in or user cache binary is needed by the NequIP tests.
"""

from pathlib import Path

from symmetrix.oam_jit_artifact import prepare_nequip_oam_host_artifact


FIXTURE_CONTRACT = Path(__file__).with_name("oam_s_v2_minimal.json")


def prepare_oam_jit_fixture(
    cache_dir: str | Path,
    precision: str = "float64",
    *,
    contract_path: str | Path = FIXTURE_CONTRACT,
) -> tuple[Path, dict]:
    """Compile the generated fixture plugin and return ``(path, metadata)``."""

    result = prepare_nequip_oam_host_artifact(
        contract_path,
        precision=precision,
        cache_dir=cache_dir,
    )
    return result.artifact_path, dict(result.metadata)


build_oam_jit_fixture = prepare_oam_jit_fixture


__all__ = [
    "FIXTURE_CONTRACT",
    "build_oam_jit_fixture",
    "prepare_oam_jit_fixture",
]
