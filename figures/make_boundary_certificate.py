"""Render compact single-column TMC model-boundary and certificate figures.

This v19 renderer is additive: it reads immutable formal-result CSV files and
writes only to ``figs_tmc_v19``.  It does not edit the manuscript or any prior
figure directory.

Outputs
-------
* ``fig_model_boundary_compact.{pdf,svg,png}``
* ``fig_certificate_operating_compact.{pdf,svg,png}``
* ``render_qa.json``
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
from matplotlib.transforms import blended_transform_factory


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results" / "frozen"
OUT = HERE / "generated"

COLUMN_WIDTH_IN = 3.48
MIN_FONT_PT = 7.0

# Color-vision-safe semantic palette used by the existing TMC renderers.
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"
PURPLE = "#7B61A8"
SKY = "#56B4E9"
DARK = "#222222"
GRAY = "#666666"
MID_GRAY = "#9A9A9A"
LIGHT_GRAY = "#D9D9D9"
PALE = "#F2F2F2"
WHITE = "#FFFFFF"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.4,
            "axes.titlesize": 8.1,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.72,
            "lines.linewidth": 1.35,
            "lines.markersize": 4.2,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": WHITE,
            "figure.facecolor": WHITE,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_record(path: Path) -> dict[str, str | int]:
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def read_csv(name: str) -> tuple[list[dict[str, str]], Path]:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"Required formal result artifact is missing: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Result artifact is empty: {path}")
    return rows, path


def require_fields(
    rows: list[dict[str, str]], fields: tuple[str, ...], source: str
) -> None:
    missing = [field for field in fields if field not in rows[0]]
    if missing:
        raise KeyError(f"{source} lacks required fields: {missing}")


def num(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key} in row: {row}")
    return value


def panel_title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontweight="bold", pad=3.5)


def grid_y(ax) -> None:
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def grid_x(ax) -> None:
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def vertical_mean_ci(
    ax,
    x: np.ndarray,
    mean: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    color: str,
    marker: str,
    linestyle: str,
    label: str,
) -> None:
    ax.errorbar(
        x,
        mean,
        yerr=np.vstack([mean - low, high - mean]),
        color=color,
        marker=marker,
        linestyle=linestyle,
        label=label,
        capsize=1.9,
        linewidth=1.25,
        markersize=4.0,
        markeredgewidth=0.7,
        zorder=3,
    )


def horizontal_mean_ci(
    ax,
    mean: np.ndarray,
    y: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    color: str,
    marker: str,
    filled: bool,
    alpha: float,
    zorder: int,
) -> None:
    ax.errorbar(
        mean,
        y,
        xerr=np.vstack([mean - low, high - mean]),
        fmt=marker,
        linestyle="none",
        color=color,
        ecolor=color,
        alpha=alpha,
        markerfacecolor=color if filled else WHITE,
        markeredgecolor=color,
        markeredgewidth=0.75,
        markersize=4.2,
        capsize=1.7,
        elinewidth=0.9,
        zorder=zorder,
    )


def save_figure(
    fig,
    stem: str,
    source_paths: list[Path],
    *,
    layout: dict[str, object],
    point_counts: dict[str, int],
    statistical_boundary: list[str],
) -> dict[str, object]:
    """Export a fixed-width figure and fail on font or canvas overflow."""
    OUT.mkdir(parents=True, exist_ok=True)
    width, height = map(float, fig.get_size_inches())
    if abs(width - COLUMN_WIDTH_IN) > 1e-9:
        raise AssertionError(
            f"{stem}: canvas width {width:.3f} in is not {COLUMN_WIDTH_IN:.2f} in"
        )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas_width, canvas_height = fig.canvas.get_width_height()
    min_font = math.inf
    overflow: list[str] = []
    text_count = 0
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        text_count += 1
        min_font = min(min_font, float(artist.get_fontsize()))
        bbox = artist.get_window_extent(renderer=renderer)
        if bbox.width <= 0 or bbox.height <= 0:
            continue
        if (
            bbox.x0 < -1.0
            or bbox.y0 < -1.0
            or bbox.x1 > canvas_width + 1.0
            or bbox.y1 > canvas_height + 1.0
        ):
            overflow.append(artist.get_text())
    if min_font < MIN_FONT_PT - 1e-9:
        raise AssertionError(
            f"{stem}: minimum rendered font is {min_font:.2f} pt "
            f"(< {MIN_FONT_PT:.1f} pt)"
        )
    if overflow:
        raise AssertionError(f"{stem}: text outside fixed canvas: {overflow[:8]}")

    axes_outside = []
    for idx, ax in enumerate(fig.axes):
        bbox = ax.get_position()
        if bbox.x0 < 0 or bbox.y0 < 0 or bbox.x1 > 1 or bbox.y1 > 1:
            axes_outside.append(idx)
    if axes_outside:
        raise AssertionError(f"{stem}: axes outside canvas: {axes_outside}")

    outputs: dict[str, dict[str, str | int]] = {}
    for suffix in ("pdf", "svg", "png"):
        path = OUT / f"{stem}.{suffix}"
        kwargs: dict[str, object] = {
            "bbox_inches": None,
            "facecolor": WHITE,
        }
        if suffix == "png":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)
        outputs[suffix] = {
            "path": str(path.relative_to(HERE)).replace("\\", "/"),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    plt.close(fig)

    return {
        "stem": stem,
        "canvas": {
            "width_in": width,
            "height_in": height,
            "target_width_in": COLUMN_WIDTH_IN,
        },
        "minimum_font_pt": float(min_font),
        "text_artist_count": text_count,
        "text_overflow_count": 0,
        "axes_outside_canvas_count": 0,
        "layout": layout,
        "point_counts": point_counts,
        "statistical_boundary": statistical_boundary,
        "sources": [source_record(path) for path in source_paths],
        "outputs": outputs,
        "checks": {
            "fixed_single_column_width": "pass",
            "minimum_font_at_least_7pt": "pass",
            "no_text_outside_canvas": "pass",
            "all_axes_inside_canvas": "pass",
            "pdf_svg_png_exported": "pass",
            "png_dpi": 300,
            "vector_text_preserved_in_svg": "pass",
        },
    }


def make_model_boundary_figure() -> dict[str, object]:
    multi, multi_path = read_csv("tmc_multiaxis_formal_summary.csv")
    ca_n4, ca_n4_path = read_csv("tmc_ca_mismatch_formal_v2_n4_summary.csv")
    ca_n1, ca_n1_path = read_csv("tmc_ca_mismatch_formal_v2_n1_summary.csv")

    require_fields(
        multi,
        (
            "dimension",
            "method",
            "post_excess_mean",
            "post_excess_mean_ci_low",
            "post_excess_mean_ci_high",
        ),
        multi_path.name,
    )
    ca_fields = (
        "spatial_dimension",
        "method",
        "relative_vs_ca_index_pct_mean",
        "relative_vs_ca_index_pct_ci_low",
        "relative_vs_ca_index_pct_ci_high",
        "action_disagreement_rate_mean",
        "action_disagreement_rate_ci_low",
        "action_disagreement_rate_ci_high",
        "cost_ratio_age_8_mean",
        "cost_ratio_age_8_ci_low",
        "cost_ratio_age_8_ci_high",
        "cost_ratio_age_16_mean",
        "cost_ratio_age_16_ci_low",
        "cost_ratio_age_16_ci_high",
        "cost_ratio_age_32_mean",
        "cost_ratio_age_32_ci_low",
        "cost_ratio_age_32_ci_high",
        "cost_ratio_age_64_mean",
        "cost_ratio_age_64_ci_low",
        "cost_ratio_age_64_ci_high",
    )
    require_fields(ca_n4, ca_fields, ca_n4_path.name)
    require_fields(ca_n1, ca_fields, ca_n1_path.name)

    method_specs = [
        ("max_age", "Max age", MID_GRAY, "o", ":"),
        ("cumulative_ce", "Cumulative CE", GRAY, "o", "--"),
        ("cumulative_ucb_cv", "Cumulative UCB", PURPLE, "s", ":"),
        ("sw_whittle_cv_64", "SW-Whittle-CV", SKY, "^", "--"),
        ("sco_reset_ce", "SCO-reset-CE", GREEN, "D", "--"),
        ("sco_reset_ucb", "SCO-reset-UCB", BLUE, "D", "-"),
    ]
    dimensions = (1, 2, 3)
    for method, *_ in method_specs:
        selected = [row for row in multi if row["method"] == method]
        found = sorted(int(row["dimension"]) for row in selected)
        if found != list(dimensions):
            raise AssertionError(f"{method}: expected dimensions {dimensions}, got {found}")

    fig = plt.figure(figsize=(COLUMN_WIDTH_IN, 4.78))
    ax_a = fig.add_axes([0.18, 0.615, 0.79, 0.205])
    ax_gap = fig.add_axes([0.18, 0.345, 0.355, 0.155])
    ax_disagree = fig.add_axes([0.615, 0.345, 0.355, 0.155], sharey=ax_gap)
    ax_age = fig.add_axes([0.225, 0.075, 0.725, 0.145])

    # Evidence zone (a): preserve all 6 methods x 3 dimensions and their CIs.
    for method, label, color, marker, linestyle in method_specs:
        selected = sorted(
            [row for row in multi if row["method"] == method],
            key=lambda row: int(row["dimension"]),
        )
        x = np.asarray([int(row["dimension"]) for row in selected], dtype=float)
        mean = np.asarray([num(row, "post_excess_mean") for row in selected])
        low = np.asarray([num(row, "post_excess_mean_ci_low") for row in selected])
        high = np.asarray([num(row, "post_excess_mean_ci_high") for row in selected])
        vertical_mean_ci(
            ax_a,
            x,
            mean,
            low,
            high,
            color=color,
            marker=marker,
            linestyle=linestyle,
            label=label,
        )
    ax_a.set_xticks(dimensions, ["1-D", "2-D", "3-D"])
    ax_a.set_xlim(0.88, 3.12)

    ax_a.set_ylabel("Post-change\nexcess (%)")
    grid_y(ax_a)
    panel_title(ax_a, "(a) Separable CV generalization")

    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.2,
                markersize=4.0,
                label=label,
            )
            for _, label, color, marker, linestyle in method_specs
        ],
        loc="upper center",
        bbox_to_anchor=(0.50, 0.987),
        ncol=3,
        frameon=False,
        handlelength=1.45,
        columnspacing=0.52,
        handletextpad=0.22,
        borderaxespad=0.0,
    )

    # Evidence zone (b): aligned outcome and action-disagreement tracks.
    fig.text(
        0.18,
        0.555,
        "(b) CA boundary",
        ha="left",
        va="center",
        fontsize=8.1,
        fontweight="bold",
    )
    boundary_handles = [
        Line2D(
            [0],
            [0],
            color=BLUE,
            marker="o",
            linestyle="none",
            markersize=4.2,
            label="CV surrogate",
        ),
        Line2D(
            [0],
            [0],
            color=MID_GRAY,
            alpha=0.48,
            marker="^",
            linestyle="none",
            markersize=4.2,
            label="max-age ref.",
        ),
        Line2D(
            [0],
            [0],
            color=DARK,
            marker="s",
            markerfacecolor=DARK,
            linestyle="none",
            markersize=3.8,
            label="$N=1$",
        ),
        Line2D(
            [0],
            [0],
            color=DARK,
            marker="s",
            markerfacecolor=WHITE,
            linestyle="none",
            markersize=3.8,
            label="$N=4$",
        ),
    ]
    fig.legend(
        handles=boundary_handles,
        loc="center",
        bbox_to_anchor=(0.59, 0.545),
        ncol=4,
        frameon=False,
        handlelength=0.78,
        columnspacing=0.42,
        handletextpad=0.18,
        borderaxespad=0.0,
    )

    source_specs = [
        (ca_n4, False, -0.105),
        (ca_n1, True, 0.105),
    ]
    boundary_methods = [
        ("max_age", MID_GRAY, "^", 0.42, 2),
        ("cubic_cv_surrogate", BLUE, "o", 1.0, 4),
    ]
    for source_rows, filled, offset in source_specs:
        for method, color, marker, alpha, zorder in boundary_methods:
            selected = sorted(
                [row for row in source_rows if row["method"] == method],
                key=lambda row: int(row["spatial_dimension"]),
            )
            found = [int(row["spatial_dimension"]) for row in selected]
            if found != list(dimensions):
                raise AssertionError(
                    f"{method}: expected CA dimensions {dimensions}, got {found}"
                )
            y = np.asarray(found, dtype=float) + offset

            gap_mean = np.asarray(
                [num(row, "relative_vs_ca_index_pct_mean") for row in selected]
            )
            gap_low = np.asarray(
                [num(row, "relative_vs_ca_index_pct_ci_low") for row in selected]
            )
            gap_high = np.asarray(
                [num(row, "relative_vs_ca_index_pct_ci_high") for row in selected]
            )
            horizontal_mean_ci(
                ax_gap,
                gap_mean,
                y,
                gap_low,
                gap_high,
                color=color,
                marker=marker,
                filled=filled,
                alpha=alpha,
                zorder=zorder,
            )

            disagreement_mean = 100.0 * np.asarray(
                [num(row, "action_disagreement_rate_mean") for row in selected]
            )
            disagreement_low = 100.0 * np.asarray(
                [num(row, "action_disagreement_rate_ci_low") for row in selected]
            )
            disagreement_high = 100.0 * np.asarray(
                [num(row, "action_disagreement_rate_ci_high") for row in selected]
            )
            horizontal_mean_ci(
                ax_disagree,
                disagreement_mean,
                y,
                disagreement_low,
                disagreement_high,
                color=color,
                marker=marker,
                filled=filled,
                alpha=alpha,
                zorder=zorder,
            )

    ax_gap.set_xscale("log")
    ax_gap.set_xlim(0.24, 135.0)
    ax_gap.set_xticks([0.3, 1.0, 10.0, 100.0], ["0.3", "1", "10", "100"])
    ax_gap.set_yticks(dimensions, ["1-D", "2-D", "3-D"])
    ax_gap.set_ylim(0.66, 3.36)
    ax_gap.set_xlabel("Excess vs exact CA (%)")
    ax_gap.set_title("Scheduling-cost gap", loc="left", pad=2.5)
    grid_x(ax_gap)

    ax_disagree.set_xlim(0.0, 80.0)
    ax_disagree.set_xticks([0, 25, 50, 75])
    ax_disagree.set_xlabel("Action disagreement (%)")
    ax_disagree.set_title("Policy disagreement", loc="left", pad=2.5)
    ax_disagree.tick_params(axis="y", left=False, labelleft=False)
    grid_x(ax_disagree)

    # The age strip is an inset within evidence zone (b), not a third panel.
    ages = np.asarray([8, 16, 32, 64], dtype=float)
    n4_age_rows = sorted(
        [
            row
            for row in ca_n4
            if row["method"] == "ca_index"
            and int(row["spatial_dimension"]) in dimensions
        ],
        key=lambda row: int(row["spatial_dimension"]),
    )
    n1_age_rows = sorted(
        [
            row
            for row in ca_n1
            if row["method"] == "ca_index"
            and int(row["spatial_dimension"]) in dimensions
        ],
        key=lambda row: int(row["spatial_dimension"]),
    )
    if len(n4_age_rows) != 3 or len(n1_age_rows) != 3:
        raise AssertionError("Expected one CA age-ratio row per spatial dimension")

    age_means = []
    age_lows = []
    age_highs = []
    for n4_row, n1_row in zip(n4_age_rows, n1_age_rows):
        if n4_row["spatial_dimension"] != n1_row["spatial_dimension"]:
            raise AssertionError("N=1 and N=4 age-ratio rows are misaligned")
        means = np.asarray(
            [num(n4_row, f"cost_ratio_age_{int(age)}_mean") for age in ages]
        )
        lows = np.asarray(
            [num(n4_row, f"cost_ratio_age_{int(age)}_ci_low") for age in ages]
        )
        highs = np.asarray(
            [num(n4_row, f"cost_ratio_age_{int(age)}_ci_high") for age in ages]
        )
        n1_means = np.asarray(
            [num(n1_row, f"cost_ratio_age_{int(age)}_mean") for age in ages]
        )
        if not np.allclose(means, n1_means, rtol=0, atol=1e-12):
            raise AssertionError("Age-cost ratios unexpectedly differ by capacity")
        age_means.append(means)
        age_lows.append(lows)
        age_highs.append(highs)

    age_means_arr = np.asarray(age_means)
    age_lows_arr = np.asarray(age_lows)
    age_highs_arr = np.asarray(age_highs)
    median_dimension = np.median(age_means_arr, axis=0)
    ci_envelope_low = np.min(age_lows_arr, axis=0)
    ci_envelope_high = np.max(age_highs_arr, axis=0)
    ax_age.fill_between(
        ages,
        ci_envelope_low,
        ci_envelope_high,
        color=SKY,
        alpha=0.24,
        linewidth=0,
        label="1–3D CI envelope",
        zorder=1,
    )
    ax_age.plot(
        ages,
        median_dimension,
        color=BLUE,
        marker="o",
        markersize=3.6,
        linewidth=1.25,
        label="median dimension",
        zorder=3,
    )
    ax_age.axhline(1.0, color=GRAY, linestyle=":", linewidth=0.8, zorder=0)
    ax_age.set_xscale("log", base=2)
    ax_age.set_xlim(7.5, 68)
    ax_age.set_xticks(ages, [str(int(age)) for age in ages])
    ax_age.set_ylim(0.0, 1.08)
    ax_age.set_yticks([0.0, 0.5, 1.0], ["0", "0.5", "1.0"])
    ax_age.set_xlabel("Packet age")
    ax_age.set_ylabel("Fitted / true")
    ax_age.set_title("Long-age cost-ratio inset", loc="left", pad=2.0)
    grid_y(ax_age)
    ax_age.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 0.82),
        frameon=False,
        ncol=2,
        handlelength=1.0,
        columnspacing=0.48,
        handletextpad=0.20,
        borderpad=0.0,
    )

    return save_figure(
        fig,
        "fig_model_boundary_compact",
        [multi_path, ca_n4_path, ca_n1_path],
        layout={
            "format": "single-column",
            "width_in": COLUMN_WIDTH_IN,
            "evidence_zones": 2,
            "zone_a": "full-width 6-method x 3-dimension mean/CI trajectories",
            "zone_b": (
                "row-aligned cost-gap and action-disagreement tracks; "
                "long-age ratio rendered as an internal ribbon inset"
            ),
            "max_age_role": "low-opacity contextual reference only",
        },
        point_counts={
            "zone_a_method_dimension_mean_ci": 18,
            "zone_b_cost_gap_mean_ci": 12,
            "zone_b_action_disagreement_mean_ci": 12,
            "zone_b_age_ratio_underlying_dimension_age_cells": 12,
            "total_plotted_metric_observations": 54,
        },
        statistical_boundary=[
            "All means and normal 95% intervals are read from formal summary CSVs.",
            "N=1 and N=4 are encoded by filled and open markers; they are not pooled.",
            "The age-ratio line is the descriptive median across the three displayed "
            "dimensions; the ribbon is the min-to-max envelope of their reported "
            "normal 95% intervals, not a pooled confidence interval.",
            "Max age is rendered as a low-opacity reference and is not used to define "
            "the surrogate boundary.",
        ],
    )


def make_certificate_operating_figure() -> dict[str, object]:
    rows, source_path = read_csv("tmc_certificate_sweep_summary.csv")
    require_fields(
        rows,
        (
            "scale",
            "seeds",
            "index_joint_coverage_mean",
            "index_joint_coverage_ci_low",
            "index_joint_coverage_ci_high",
            "certificate_rate_mean",
            "certificate_rate_ci_low",
            "certificate_rate_ci_high",
            "certificate_error_rate_mean",
            "certificate_error_rate_ci_low",
            "certificate_error_rate_ci_high",
            "certified_count",
            "wrong_certified_count",
        ),
        source_path.name,
    )
    rows = sorted(rows, key=lambda row: num(row, "scale"))
    scales = np.asarray([num(row, "scale") for row in rows])
    if not np.allclose(scales, [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]):
        raise AssertionError(f"Unexpected certificate radius grid: {scales.tolist()}")

    certified = np.asarray([int(round(num(row, "certified_count"))) for row in rows])
    wrong = np.asarray(
        [int(round(num(row, "wrong_certified_count"))) for row in rows]
    )
    firing_fraction = np.asarray(
        [num(row, "certificate_rate_mean") for row in rows]
    )
    inferred_denominators = np.rint(certified / firing_fraction).astype(int)
    if not np.all(inferred_denominators == inferred_denominators[0]):
        raise AssertionError(
            f"Aggregate slot denominators differ by radius: {inferred_denominators}"
        )
    aggregate_denominator = int(inferred_denominators[0])

    fig = plt.figure(figsize=(COLUMN_WIDTH_IN, 4.18))
    ax_a = fig.add_axes([0.19, 0.59, 0.77, 0.315])
    ax_b = fig.add_axes([0.19, 0.17, 0.50, 0.29])
    ax_counts = fig.add_axes([0.735, 0.17, 0.225, 0.29], sharey=ax_b)

    # (a) A two-dimensional operating curve.  Both axes carry seed-level CIs;
    # marker area only encodes the separate aggregate certified count.
    firing = 100.0 * firing_fraction
    firing_low = 100.0 * np.asarray(
        [num(row, "certificate_rate_ci_low") for row in rows]
    )
    firing_high = 100.0 * np.asarray(
        [num(row, "certificate_rate_ci_high") for row in rows]
    )
    coverage = 100.0 * np.asarray(
        [num(row, "index_joint_coverage_mean") for row in rows]
    )
    coverage_low_raw = 100.0 * np.asarray(
        [num(row, "index_joint_coverage_ci_low") for row in rows]
    )
    coverage_high_raw = 100.0 * np.asarray(
        [num(row, "index_joint_coverage_ci_high") for row in rows]
    )
    coverage_low = np.maximum(0.0, coverage_low_raw)
    coverage_high = np.minimum(100.0, coverage_high_raw)

    ax_a.plot(firing, coverage, color=LIGHT_GRAY, linewidth=1.1, zorder=1)
    ax_a.errorbar(
        firing,
        coverage,
        xerr=np.vstack([firing - firing_low, firing_high - firing]),
        yerr=np.vstack([coverage - coverage_low, coverage_high - coverage]),
        fmt="none",
        ecolor=GRAY,
        elinewidth=0.85,
        capsize=1.8,
        zorder=2,
    )
    log_counts = np.log10(certified.astype(float))
    marker_area = 28.0 + 72.0 * (
        (log_counts - log_counts.min()) / (log_counts.max() - log_counts.min())
    )
    ax_a.scatter(
        firing,
        coverage,
        s=marker_area,
        color=BLUE,
        edgecolor=WHITE,
        linewidth=0.7,
        zorder=3,
    )
    label_offsets = {
        0.5: (-10, 7),
        1.0: (-18, 8),
        2.0: (5, 5),
        3.0: (5, -13),
        4.0: (5, -13),
        6.0: (5, -13),
    }
    for x, y, scale in zip(firing, coverage, scales):
        ax_a.annotate(
            f"{scale:g}x",
            (x, y),
            xytext=label_offsets[float(scale)],
            textcoords="offset points",
            color=DARK,
            fontweight="bold",
            zorder=4,
        )
    ax_a.set_xscale("log")
    ax_a.set_xlim(0.32, 55.0)
    ax_a.set_xticks(
        [0.5, 1, 2, 5, 10, 20, 40],
        ["0.5", "1", "2", "5", "10", "20", "40"],
    )
    ax_a.set_ylim(-3.0, 103.0)
    ax_a.set_yticks([0, 25, 50, 75, 100])
    ax_a.set_xlabel("Certificate firing rate (%)")
    ax_a.set_ylabel("Joint index coverage (%)")
    ax_a.grid(color=LIGHT_GRAY, linewidth=0.5, zorder=0)
    panel_title(ax_a, "(a) Coverage–availability operating curve")
    ax_a.text(
        0.02,
        0.05,
        "area $\\propto$ log aggregate certified",
        transform=ax_a.transAxes,
        ha="left",
        va="bottom",
        color=GRAY,
    )

    # (b) Use the reported seed-level conditional-error mean and CI directly.
    # The right-hand count text is aggregate descriptive support, not the CI.
    error_mean = 100.0 * np.asarray(
        [num(row, "certificate_error_rate_mean") for row in rows]
    )
    error_low = 100.0 * np.asarray(
        [num(row, "certificate_error_rate_ci_low") for row in rows]
    )
    error_high = 100.0 * np.asarray(
        [num(row, "certificate_error_rate_ci_high") for row in rows]
    )
    y = np.arange(len(rows) - 1, -1, -1, dtype=float)
    ax_b.axvline(0.0, color=DARK, linewidth=0.8, zorder=0)
    for idx, (mean, low, high, yy, wrong_count) in enumerate(
        zip(error_mean, error_low, error_high, y, wrong)
    ):
        ax_b.errorbar(
            mean,
            yy,
            xerr=np.asarray([[mean - low], [high - mean]]),
            fmt="o",
            color=ORANGE if wrong_count > 0 else BLUE,
            markerfacecolor=ORANGE if wrong_count > 0 else WHITE,
            markeredgecolor=ORANGE if wrong_count > 0 else BLUE,
            markeredgewidth=0.8,
            markersize=4.5,
            capsize=1.9,
            elinewidth=1.0,
            zorder=3,
        )
    ax_b.set_xscale("symlog", linthresh=0.10, linscale=0.85, base=10)
    ax_b.set_xlim(-0.14, 60.0)
    ax_b.set_xticks(
        [-0.1, 0.0, 0.1, 1.0, 10.0, 30.0],
        ["−0.1", "0", "0.1", "1", "10", "30"],
    )
    ax_b.set_yticks(y, [f"{scale:g}x" for scale in scales])
    ax_b.set_ylim(-0.65, len(rows) - 0.35)
    ax_b.set_xlabel("Conditional error among fired (%)")
    ax_b.set_ylabel("Radius")
    grid_x(ax_b)
    panel_title(ax_b, "(b) Seed-level conditional error")
    ax_counts.set_xlim(0.0, 1.0)
    ax_counts.set_ylim(ax_b.get_ylim())
    ax_counts.axis("off")
    ax_counts.text(
        0.5,
        1.04,
        "aggregate counts\nwrong / certified",
        transform=ax_counts.transAxes,
        ha="center",
        va="bottom",
        color=GRAY,
    )
    count_transform = blended_transform_factory(
        ax_counts.transAxes, ax_counts.transData
    )
    for yy, wrong_count, certified_count in zip(y, wrong, certified):
        ax_counts.text(
            0.5,
            yy,
            f"{wrong_count:,} / {certified_count:,}",
            transform=count_transform,
            ha="center",
            va="center",
            color=DARK if wrong_count > 0 else GRAY,
        )

    fig.text(
        0.19,
        0.045,
        "Forest = seed-level mean + normal 95% CI.\n"
        "Right text = aggregate counts (not the interval).",
        ha="left",
        va="bottom",
        color=GRAY,
    )

    pooled_error = wrong / certified
    max_seed_vs_pooled_gap = float(
        np.max(np.abs(error_mean / 100.0 - pooled_error))
    )
    return save_figure(
        fig,
        "fig_certificate_operating_compact",
        [source_path],
        layout={
            "format": "single-column",
            "width_in": COLUMN_WIDTH_IN,
            "panels": 2,
            "panel_a": (
                "full-width firing-vs-coverage operating curve with x/y seed-level "
                "intervals, radius labels, and log-count marker-area encoding"
            ),
            "panel_b": (
                "full-width radius forest of seed-level conditional-error means and "
                "intervals with a separate aggregate wrong/certified text column"
            ),
        },
        point_counts={
            "input_radius_rows": 6,
            "panel_a_operating_points": 6,
            "panel_b_seed_level_forest_points": 6,
            "panel_b_aggregate_count_annotations": 6,
            "total_plotted_statistical_points": 12,
        },
        statistical_boundary=[
            "Panel (a) uses seed-level mean intervals for firing and joint index "
            "coverage; probability intervals are visually clipped to [0,100]%.",
            "Panel (a) marker area uses a log transform of aggregate certified count "
            "for visibility and is not an uncertainty encoding.",
            "Panel (b) reads certificate_error_rate_mean and its reported seed-level "
            "normal 95% interval directly from the summary CSV.",
            "The right-hand wrong/certified values are aggregate counts and are not "
            "substituted for the seed-level mean or interval.",
            f"All radii share {aggregate_denominator:,} aggregate batch-slots; the "
            f"maximum absolute seed-mean versus pooled-count rate difference is "
            f"{max_seed_vs_pooled_gap:.6f}.",
            "Zero aggregate wrong certificates is displayed as an observed count, "
            "not as a zero-risk statement.",
        ],
    )


def main() -> None:
    setup_style()
    figure_records = [
        make_model_boundary_figure(),
        make_certificate_operating_figure(),
    ]
    renderer_path = Path(__file__).resolve()
    manifest = {
        "schema_version": 2,
        "renderer": source_record(renderer_path),
        "output_directory": OUT.name,
        "design_contract": {
            "target": "IEEE TMC true single-column figures",
            "canvas_width_in": COLUMN_WIDTH_IN,
            "minimum_font_pt": MIN_FONT_PT,
            "artifact_policy": "additive v19 outputs; no manuscript or v18 edits",
            "data_policy": "formal summary CSVs only; no synthetic or invented values",
            "palette_policy": "color-vision-safe semantic palette consistent with TMC figures",
            "no_fabrication": True,
        },
        "figures": figure_records,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    # Other additive v19 renderers share OUT; keep this renderer's manifest
    # namespaced so parallel runs cannot clobber one another.
    manifest_path = OUT / "fig_boundary_certificate_qa.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "figures": [record["stem"] for record in figure_records],
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
