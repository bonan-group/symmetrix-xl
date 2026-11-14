"""Keep maintained documentation aligned with current execution contracts."""

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]


def _read(relative_path: str) -> str:
    return (_ROOT / relative_path).read_text()


def test_developer_index_points_to_current_matrix():
    assert "/reference/execution_support_matrix" in _read("docs/developer/index.md")
