"""Pinned torch-dftd 0.5.3 source identity shared by data generators."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


EXPECTED_VERSION = "0.5.3"
UPSTREAM_TAG = "v0.5.3"
UPSTREAM_COMMIT = "28b43557e6d950502941990fa5a1c9a3aa9e8339"

EXPECTED_SOURCE_HASHES = {
    "LICENSE": "7f7be264c10276c7772bdead5b3bcdb68082edf862ca0f67d97efdb36bc2d889",
    "torch_dftd/__init__.py": (
        "b0382200a901e94928a18656f8a8d98c775edeb85971b7e8f25d605a6d8b31d3"
    ),
    "torch_dftd/_version.py": (
        "b60ceea8729c11d28168fe7b179a17c6ae17956da7f49e058fc646bbb9c63998"
    ),
    "torch_dftd/dftd3_xc_params.py": (
        "30db7a75ffba3d8c4f65c959102c06ad60063a38d61cfcb00c93f4d73a99d71a"
    ),
    "torch_dftd/functions/__init__.py": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "torch_dftd/functions/dftd2.py": (
        "ce58eaab29f5ee83e707ecb77db984f2613331e9bbf4f23c4f7310a827e4c86e"
    ),
    "torch_dftd/functions/dftd3.py": (
        "0ef70d680aeb450a90af3aaeba0972ff89938c052f2309c0ec6d5347cf25f1f2"
    ),
    "torch_dftd/functions/distance.py": (
        "9835ebbd9820790411d62f7bbf6445c1cc8d5206204f83379ee6583e57c6ad6d"
    ),
    "torch_dftd/functions/smoothing.py": (
        "3e33be3edb2ab68fe6b113ddf6e5fe9cdff0ca5bbffab00e096306cf07f925a6"
    ),
    "torch_dftd/functions/triplets.py": (
        "e21781e60c2cdd5eda2677ffe49e1fdc983662a1addf5ea6795aefdd7a21525d"
    ),
    "torch_dftd/functions/triplets_kernel.py": (
        "abbb8fd293c7f6e5691791aa8437d37463c25053e35878482a5cf30f6da68a94"
    ),
    "torch_dftd/nn/__init__.py": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "torch_dftd/nn/base_dftd_module.py": (
        "9702458337e426795acfb6aae8a82a45b742a8ef96f2eee8547e12cd599005d4"
    ),
    "torch_dftd/nn/dftd2_module.py": (
        "321479cbc4ce2ae814aafb1ad831921d8c0a126ed4f7b36230c89b626e45995f"
    ),
    "torch_dftd/nn/dftd3_module.py": (
        "24970d4a267badab653b795b321b3f93c45994cb6c09b53a880cc214104fb72d"
    ),
    "torch_dftd/nn/params/__init__.py": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "torch_dftd/nn/params/dftd2_params.py": (
        "850d7d9fea302bc91583d09f2395afa0746c713a83425751d70898945eff0665"
    ),
    "torch_dftd/nn/params/dftd3_params.npz": (
        "52b7b35a957df55b813035ffde3a6b5a41dec585bdb1e281fb8ba67a729f3d92"
    ),
    "torch_dftd/testing/damping.py": (
        "ca221387b48666255fdcfa34269353754f901d678f19cd6f7719c38fb4f2216b"
    ),
}

PARAMETER_SOURCE_PATHS = (
    "LICENSE",
    "torch_dftd/_version.py",
    "torch_dftd/dftd3_xc_params.py",
    "torch_dftd/testing/damping.py",
    "torch_dftd/nn/params/dftd2_params.py",
    "torch_dftd/nn/params/dftd3_params.npz",
)

# Include every torch-dftd file imported when the oracle instantiates the D2 and
# D3 modules, plus the exact data and license inputs. D3's module imports the
# triplet implementation even though these two-body fixtures set abc=False.
REFERENCE_SOURCE_PATHS = tuple(EXPECTED_SOURCE_HASHES)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_source_file(root: Path, relative: str) -> Path:
    if relative != "LICENSE":
        return root / relative
    candidates = [root / "LICENSE"]
    candidates.extend(sorted(root.glob("torch_dftd-*.dist-info/licenses/LICENSE")))
    license_path = next((path for path in candidates if path.is_file()), None)
    if license_path is None:
        raise RuntimeError(f"cannot find torch-dftd LICENSE under {root}")
    return license_path


def validate_source(root: Path, relative_paths: tuple[str, ...]) -> dict[str, Path]:
    files = {
        relative: _resolve_source_file(root, relative) for relative in relative_paths
    }
    missing = [relative for relative, path in files.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing torch-dftd source files: {', '.join(missing)}")
    for relative, path in files.items():
        actual = sha256(path)
        expected = EXPECTED_SOURCE_HASHES[relative]
        if actual != expected:
            raise RuntimeError(
                f"torch-dftd 0.5.3 SHA-256 mismatch for {relative}: "
                f"expected {expected}, got {actual}"
            )

    if (root / ".git").exists():
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"cannot resolve torch-dftd Git commit under {root}: "
                f"{completed.stderr.strip()}"
            )
        actual_commit = completed.stdout.strip()
        if actual_commit != UPSTREAM_COMMIT:
            raise RuntimeError(
                "torch-dftd 0.5.3 Git commit mismatch: "
                f"expected {UPSTREAM_COMMIT}, got {actual_commit}"
            )
    return files
