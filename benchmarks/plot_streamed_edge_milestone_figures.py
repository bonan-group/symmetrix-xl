#!/usr/bin/env python3
"""Generate the milestone report's performance and capacity figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


COLORS = {
    "direct": "#009E73",
    "generic": "#0072B2",
    "develop": "#666666",
    "cueq": "#E69F00",
    "e3nn": "#CC3311",
    "retained": "#8A8A8A",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.28,
            "svg.fonttype": "none",
            "svg.hashsalt": "symmetrix-streamed-edge-milestone",
        }
    )


def _log_tick(value: float, _position: float) -> str:
    return f"{value:g}"


def _save(fig: plt.Figure, output: Path, title: str, description: str) -> None:
    file_format = output.suffix.removeprefix(".")
    fig.savefig(
        output,
        format=file_format,
        bbox_inches="tight",
        metadata={"Title": title, "Description": description, "Date": None}
        if file_format == "svg"
        else None,
    )
    plt.close(fig)
    if file_format == "svg":
        lines = output.read_text(encoding="utf-8").splitlines()
        output.write_text(
            "\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8"
        )


def performance_figure(output: Path) -> None:
    # Source: benchmarks/streamed_edge_milestone_20260822.md.
    cpu = [
        ("Current direct\nlow memory", 262.962, "direct"),
        ("Origin/develop\nmaterialized", 4767.777, "develop"),
        ("Current generic", 5615.376, "generic"),
        ("PyTorch/e3nn", 12610.511, "e3nn"),
    ]
    cuda = [
        ("Current direct\nlow memory", 4.008, "direct"),
        ("Current generic", 6.648, "generic"),
        ("PyTorch/\ncuEquivariance", 31.052, "cueq"),
        ("Origin/develop\nmaterialized", 60.326, "develop"),
        ("PyTorch/e3nn", 98.995, "e3nn"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), constrained_layout=True)
    for ax, title, rows, limits in (
        (axes[0], "(a) One CPU core, 864 atoms", cpu, (200, 18000)),
        (axes[1], "(b) CUDA, 4,000 atoms", cuda, (3.3, 145)),
    ):
        labels = [row[0] for row in rows]
        values = [row[1] for row in rows]
        colors = [COLORS[row[2]] for row in rows]
        y = list(range(len(rows)))
        bars = ax.barh(y, values, color=colors, height=0.62)
        bars[0].set_hatch("///")
        bars[0].set_edgecolor("#00664B")
        bars[0].set_linewidth(0.7)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlim(*limits)
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(FuncFormatter(_log_tick))
        ax.grid(axis="x", which="major")
        ax.set_axisbelow(True)
        ax.set_xlabel("Median evaluation time (us/atom; lower is better)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                value * 1.06,
                bar.get_y() + bar.get_height() / 2,
                f"{value:,.3f}",
                va="center",
                ha="left",
                fontsize=7.4,
            )

    _save(
        fig,
        output,
        "OMAT-0 inference performance",
        "Log-scale bar charts compare median energy, force, and stress "
        "evaluation time in microseconds per atom. Current direct low-memory "
        "execution is fastest on both one CPU core and CUDA.",
    )


def capacity_figure(output: Path) -> None:
    # Sources: benchmarks/low_memory_capacity.md and
    # benchmarks/reference_cuda_capacity_20260823.md.
    policy_rows = [
        ("OMAT-0", 219488, 389344),
        ("MACEField", 202612, 340736),
    ]
    bracket_rows = [
        ("Current direct\nlow memory", 389344, 415292, "direct"),
        ("PyTorch/\ncuEquivariance", 19652, 23328, "cueq"),
        ("Origin/develop\nmaterialized", 13500, 16384, "develop"),
        ("PyTorch/e3nn", 4000, 5324, "e3nn"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.55), constrained_layout=True)

    ax = axes[0]
    y = list(range(len(policy_rows)))
    height = 0.31
    retained = [row[1] / 1000 for row in policy_rows]
    low_memory = [row[2] / 1000 for row in policy_rows]
    retained_bars = ax.barh(
        [value + height / 2 for value in y],
        retained,
        height=height,
        color=COLORS["retained"],
        hatch="..",
        edgecolor="#555555",
        linewidth=0.6,
        label="Direct retained",
    )
    low_memory_bars = ax.barh(
        [value - height / 2 for value in y],
        low_memory,
        height=height,
        color=COLORS["direct"],
        hatch="///",
        edgecolor="#00664B",
        linewidth=0.6,
        label="Direct low memory",
    )
    ax.set_yticks(y, [row[0] for row in policy_rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 445)
    ax.set_xlabel("Largest demonstrated success (thousand atoms)")
    ax.set_title("(a) Low-memory capacity gain", loc="left", fontweight="bold")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for bars, values in ((retained_bars, retained), (low_memory_bars, low_memory)):
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                value + 7,
                bar.get_y() + bar.get_height() / 2,
                f"{value:,.1f}k",
                va="center",
                fontsize=7.4,
            )
    for index in y:
        ax.text(8, index + height / 2, "Retained", va="center", fontsize=7.0)
        ax.text(
            8,
            index - height / 2,
            "Low memory",
            va="center",
            fontsize=7.0,
            color="white",
        )

    ax = axes[1]
    for index, (_label, success, failure, key) in enumerate(bracket_rows):
        color = COLORS[key]
        ax.plot([success, failure], [index, index], color=color, linewidth=2.2)
        ax.scatter(success, index, s=33, color=color, marker="o", zorder=3)
        ax.scatter(
            failure, index, s=40, color=color, marker="x", linewidth=1.5, zorder=3
        )
        ax.text(
            success,
            index - 0.23,
            f"{success:,}",
            ha="center",
            va="bottom",
            fontsize=7.2,
        )
        ax.text(
            failure, index + 0.23, f"{failure:,}", ha="center", va="top", fontsize=7.2
        )
    ax.set_yticks(range(len(bracket_rows)), [row[0] for row in bracket_rows])
    ax.invert_yaxis()
    ax.set_ylim(len(bracket_rows) - 0.5, -0.65)
    ax.set_xscale("log")
    ax.set_xlim(3000, 750000)
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{value / 1000:g}k")
    )
    ax.set_xlabel("Atoms (log scale; circle: success, x: first failure)")
    ax.set_title("(b) OMAT-0 boundary brackets", loc="left", fontweight="bold")
    ax.grid(axis="x", which="major")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    _save(
        fig,
        output,
        "CUDA capacity comparison",
        "Panel A compares retained and low-memory current direct execution for "
        "OMAT-0 and MACEField. Panel B shows measured OMAT-0 largest-success "
        "and first-failure brackets for current direct, origin/develop, "
        "PyTorch cuEquivariance, and PyTorch e3nn.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/figures"),
        help="Directory for generated SVG figures",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        help="Optional directory for PNG copies used during visual review",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _style()
    performance_figure(args.output_dir / "streamed_edge_performance.svg")
    capacity_figure(args.output_dir / "streamed_edge_capacity.svg")
    if args.preview_dir is not None:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
        performance_figure(args.preview_dir / "streamed_edge_performance.png")
        capacity_figure(args.preview_dir / "streamed_edge_capacity.png")


if __name__ == "__main__":
    main()
