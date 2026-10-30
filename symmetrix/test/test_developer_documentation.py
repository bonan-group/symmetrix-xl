"""Keep maintained documentation aligned with current execution contracts."""

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]


def _read(relative_path: str) -> str:
    return (_ROOT / relative_path).read_text()


def test_developer_index_points_to_current_matrix():
    assert "/reference/execution_support_matrix" in _read("docs/developer/index.md")


def test_current_oam_user_guide_names_the_live_artifact_boundary():
    from symmetrix.oam_cuda_codegen import ABI_VERSION as cuda_abi_version
    from symmetrix.oam_receiver_codegen import ABI_VERSION as host_abi_version

    models = _read("docs/user/models.md")

    assert f"ABI v{host_abi_version}\nfull-message artifact" in models
    assert f"full-message-only ABI v{cuda_abi_version} artifact" in models
    assert "default `direct` mode requires a receiver-owned ABI v4" not in models
    assert "fresh runtime qualification matrix" in models


def test_current_oam_guides_do_not_promote_the_retired_runtime_matrix():
    current_guides = (
        "README.md",
        "docs/user/models.md",
        "docs/user/execution.md",
        "docs/reference/python_api.md",
    )
    retired_claims = (
        "Retained OAM-v2 qualification covers",
        "The retained v0.2 release matrix qualifies",
        "CUDA `sm120` S/M/L/XL FP32/FP64 is runtime-qualified",
        "qualified release plan is `oam-direct-speed`",
        "host artifact ABI v4",
    )
    for path in current_guides:
        contents = _read(path)
        for claim in retired_claims:
            assert claim not in contents, (path, claim)
