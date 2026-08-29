"""Render the frozen 3x4 capacity--delay recovery trajectories.

The renderer uses forty observed post-change checkpoints per curve and never
interpolates or synthesizes observations.  PDF and SVG are publication assets;
the PNG is a 300-dpi preview.  A JSON sidecar records source hashes and
executable geometry/typography checks.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.text import Text


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FROZEN = ROOT / "results" / "frozen"
OUT = HERE / "generated"
SUMMARY = FROZEN / "tmc_capacity_delay_trajectory_summary_v36.csv"
SOURCE_META = FROZEN / "tmc_capacity_delay_trajectory_meta_v36.json"

FULL_WIDTH_IN = 7.16
HEIGHT_IN = 5.30
MIN_FONT_PT = 8.0
CAPACITIES = (2, 4, 8)
DELAYS = (0, 1, 3, 5)
CHECKPOINTS = tuple(range(10, 401, 10))
METHODS = (
    "cumulative_ucb_cv",
    "sco_reset_ucb",
    "ps_forced_reset_ucb",
    "inflight_sco_ucb",
)

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
GRAY = "#666666"
DARK = "#222222"
LIGHT = "#D9D9D9"
WHITE = "#FFFFFF"

METHOD_STYLE = {
    "cumulative_ucb_cv": ("Cumulative UCB-CV", GRAY, "o", ":"),
    "sco_reset_ucb": ("SCO-reset-UCB", BLUE, "D", "-"),
    "ps_forced_reset_ucb": ("Forced-reset-UCB", ORANGE, "^", "--"),
    "inflight_sco_ucb": ("PA-SCO", GREEN, "s", "-."),
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": MIN_FONT_PT,
            "axes.titlesize": 8.3,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.72,
            "lines.linewidth": 1.18,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": WHITE,
            "figure.facecolor": WHITE,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_record(path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def read_summary() -> list[dict[str, str]]:
    with SUMMARY.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "delay_slots",
        "capacity",
        "capacity_ratio",
        "method",
        "post_change_slot",
        "independent_seeds",
        "cumulative_excess_pct_mean",
        "cumulative_excess_pct_ci_low",
        "cumulative_excess_pct_ci_high",
    }
    if not rows or not required.issubset(rows[0]):
        raise AssertionError("capacity--delay summary is empty or incomplete")
    expected = len(CAPACITIES) * len(DELAYS) * len(METHODS) * len(CHECKPOINTS)
    if len(rows) != expected:
        raise AssertionError(f"expected {expected} summary rows, found {len(rows)}")
    keys: set[tuple[int, int, str, int]] = set()
    for row in rows:
        key = (
            int(row["capacity"]),
            int(row["delay_slots"]),
            row["method"],
            int(row["post_change_slot"]),
        )
        if key in keys:
            raise AssertionError(f"duplicate capacity--delay record: {key}")
        keys.add(key)
        if int(row["independent_seeds"]) != 12:
            raise AssertionError(f"unexpected inferential unit count: {key}")
        values = [
            float(row["cumulative_excess_pct_mean"]),
            float(row["cumulative_excess_pct_ci_low"]),
            float(row["cumulative_excess_pct_ci_high"]),
        ]
        if not all(math.isfinite(value) for value in values):
            raise AssertionError(f"non-finite plotted value: {key}")
        mean, low, high = values
        if low > mean or high < mean:
            raise AssertionError(f"interval does not contain mean: {key}")
    return rows


def text_qa(fig: plt.Figure) -> dict[str, object]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    minimum = math.inf
    count = 0
    overflow: list[str] = []
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        count += 1
        minimum = min(minimum, float(artist.get_fontsize()))
        box = artist.get_window_extent(renderer=renderer)
        if box.width <= 0 or box.height <= 0:
            continue
        if box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1:
            overflow.append(artist.get_text())
    if minimum < MIN_FONT_PT - 1e-9:
        raise AssertionError(f"minimum font is {minimum:.2f} pt")
    if overflow:
        raise AssertionError(f"text outside fixed canvas: {overflow[:8]}")
    return {
        "minimum_font_pt": minimum,
        "text_artist_count": count,
        "text_overflow_count": 0,
        "fixed_full_width_canvas": "pass",
    }


def build_figure(rows: list[dict[str, str]]) -> tuple[plt.Figure, list[dict[str, object]]]:
    lookup = {
        (
            int(row["capacity"]),
            int(row["delay_slots"]),
            row["method"],
            int(row["post_change_slot"]),
        ): row
        for row in rows
    }
    fig, axes = plt.subplots(
        3,
        4,
        figsize=(FULL_WIDTH_IN, HEIGHT_IN),
        sharex=True,
        sharey="row",
    )
    fig.subplots_adjust(
        left=0.105,
        right=0.975,
        bottom=0.100,
        top=0.870,
        wspace=0.16,
        hspace=0.25,
    )
    x = np.asarray(CHECKPOINTS, dtype=float)
    letters = "abcdefghijkl"
    endpoints: list[dict[str, object]] = []

    for row_index, capacity in enumerate(CAPACITIES):
        for column_index, delay in enumerate(DELAYS):
            ax = axes[row_index, column_index]
            panel_index = row_index * len(DELAYS) + column_index
            ax.text(
                0.965,
                0.955,
                f"({letters[panel_index]})",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontweight="bold",
                zorder=8,
            )
            for method in METHODS:
                label, color, marker, linestyle = METHOD_STYLE[method]
                method_rows = [
                    lookup[(capacity, delay, method, checkpoint)]
                    for checkpoint in CHECKPOINTS
                ]
                mean = np.asarray(
                    [float(row["cumulative_excess_pct_mean"]) for row in method_rows]
                )
                low = np.asarray(
                    [float(row["cumulative_excess_pct_ci_low"]) for row in method_rows]
                )
                high = np.asarray(
                    [float(row["cumulative_excess_pct_ci_high"]) for row in method_rows]
                )
                ax.fill_between(x, low, high, color=color, alpha=0.075, linewidth=0, zorder=1)
                ax.plot(
                    x,
                    mean,
                    color=color,
                    marker=marker,
                    markevery=5,
                    markersize=3.35,
                    markerfacecolor=WHITE,
                    markeredgewidth=0.60,
                    linewidth=1.18,
                    linestyle=linestyle,
                    label=label,
                    zorder=3,
                )
                endpoints.append(
                    {
                        "capacity_ratio": capacity / 20,
                        "delay_slots": delay,
                        "method": method,
                        "endpoint_mean_pct": float(mean[-1]),
                        "endpoint_ci_low_pct": float(low[-1]),
                        "endpoint_ci_high_pct": float(high[-1]),
                    }
                )
            ax.axhline(0.0, color=DARK, linewidth=0.65, linestyle=(0, (1.5, 2.0)), zorder=2)
            ax.grid(axis="y", color=LIGHT, linewidth=0.45, zorder=0)
            ax.set_axisbelow(True)
            ax.set_xlim(10, 400)
            ax.set_xticks([10, 100, 200, 300, 400])
            ax.tick_params(axis="both", pad=1.5)
            if row_index == 0:
                ax.set_title(rf"Delay $d={delay}$", fontweight="bold", pad=3.0)
            if column_index == 0:
                ax.set_ylabel(
                    rf"$N/K={100 * capacity / 20:.0f}\%$"
                    "\nImmediate-reset excess (%)"
                )

    fig.supxlabel("Slots after change", y=0.018, fontsize=8.2)
    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            markerfacecolor=WHITE,
            markeredgewidth=0.60,
            markersize=3.7,
            linewidth=1.18,
            linestyle=linestyle,
            label=label,
        )
        for label, color, marker, linestyle in METHOD_STYLE.values()
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.985),
        ncol=4,
        frameon=False,
        columnspacing=1.05,
        handlelength=1.65,
        handletextpad=0.42,
    )
    return fig, endpoints


def main() -> None:
    setup_style()
    rows = read_summary()
    fig, endpoints = build_figure(rows)
    qa = text_qa(fig)
    if tuple(map(float, fig.get_size_inches())) != (FULL_WIDTH_IN, HEIGHT_IN):
        raise AssertionError("figure canvas changed")
    OUT.mkdir(parents=True, exist_ok=True)
    stem = "fig_capacity_delay_trajectory_4x3_v36"
    outputs: dict[str, dict[str, object]] = {}
    for suffix in ("pdf", "svg", "png"):
        path = OUT / f"{stem}.{suffix}"
        kwargs: dict[str, object] = {"bbox_inches": None, "facecolor": WHITE}
        if suffix == "png":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)
        outputs[suffix] = {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    plt.close(fig)
    metadata = {
        "schema_version": 1,
        "renderer": Path(__file__).name,
        "layout": "4 columns (delay) x 3 rows (capacity ratio)",
        "panel_count": 12,
        "curves_per_panel": 4,
        "observed_checkpoints_per_curve": 40,
        "uncertainty": "pointwise normal 95% intervals over 12 independent seed clusters",
        "no_interpolation": True,
        "no_synthetic_points": True,
        "sources": [source_record(SUMMARY), source_record(SOURCE_META)],
        "render_qa": qa,
        "endpoints": endpoints,
        "outputs": outputs,
    }
    qa_path = OUT / "fig_capacity_delay_trajectory_4x3_v36_qa.json"
    qa_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figure": stem, "qa": qa, "outputs": list(outputs)}, indent=2))


if __name__ == "__main__":
    main()
