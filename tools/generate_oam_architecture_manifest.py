#!/usr/bin/env python3
"""Generate the model-free known-OAM architecture manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "symmetrix/source"))

from symmetrix.nequip_schema import load_nequip_oam_data  # noqa: E402
from symmetrix.oam_architecture import (  # noqa: E402
    oam_architecture_fingerprint,
    resolve_oam_architecture,
)

MANIFEST_SCHEMA = "symmetrix.foundation-oam-architectures/1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(entries: list[list[str]]) -> dict:
    architectures = []
    for identifier, expected_sha256, source_value in entries:
        source = Path(source_value)
        actual_sha256 = _sha256(source)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"{identifier} contract SHA-256 is {actual_sha256}, expected "
                f"{expected_sha256}"
            )
        architecture = resolve_oam_architecture(load_nequip_oam_data(source))
        architectures.append(
            {
                "id": identifier,
                "source_contract_sha256": actual_sha256,
                "architecture_fingerprint": oam_architecture_fingerprint(architecture),
                "architecture": architecture,
            }
        )
    architectures.sort(key=lambda entry: entry["id"])
    return {"schema": MANIFEST_SCHEMA, "architectures": architectures}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        nargs=3,
        action="append",
        metavar=("ID", "SHA256", "JSON"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = generate(args.contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
