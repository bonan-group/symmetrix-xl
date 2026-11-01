"""Tests for the external-record README speed-table generator."""

from pathlib import Path

from tools.update_readme_speed_table import render_table, update_readme


def test_render_table_requires_the_canonical_atom_count():
    record = {
        "backend": "cuda",
        "model": "/models/mace-omat-0.json",
        "precision": "float32",
        "atoms": 5000,
        "median_us_per_atom": 1.25,
        "status": "passed",
    }
    table = render_table([("cuda", record)], atoms=5000)
    assert "| cuda | mace-omat-0.json | not recorded |" in table
    assert "1.250" in table
    assert "800,000" in table


def test_update_readme_replaces_only_marked_block(tmp_path: Path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "before\n"
        "<!-- BEGIN GENERATED SPEED TABLE -->\n"
        "old\n"
        "<!-- END GENERATED SPEED TABLE -->\n"
        "after\n",
        encoding="utf-8",
    )
    table = render_table(
        [
            (
                "cpu",
                {
                    "backend": "cpu",
                    "model": "mace-omat-0.json",
                    "atoms": 5000,
                    "median_us_per_atom": 2.5,
                    "status": "passed",
                },
            )
        ],
        atoms=5000,
    )
    update_readme(readme, table)
    result = readme.read_text(encoding="utf-8")
    assert result.startswith("before\n")
    assert result.endswith("after\n")
    assert "old" not in result
    assert "2.500" in result
