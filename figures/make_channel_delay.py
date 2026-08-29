"""Build lower-density TMC v21 channel and random-delay evidence figures.

The renderer reads the same frozen formal result artifacts as the v19 plots.
It neither changes the manuscript nor writes into any earlier figure folder.
The visual hierarchy is intentionally simpler:

* ``fig_channel_pipeline_evidence_v21`` lightly harmonizes Fig. 6 while
  preserving its coherent three-question evidence chain.
* ``fig_random_delay_profiles_v21`` isolates the seven tail signatures as a
  true single-column figure.
* ``fig_random_delay_performance_mechanism_v21`` gives gains and mechanism
  boundaries room to breathe in a separate full-width two-panel figure.

PDF and SVG are publication assets; 300-dpi PNG files are previews.
``fig_channel_delay_v21_qa.json`` records source hashes, transformations, point counts,
plotted values, and executable canvas/typography checks.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.text import Text


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "generated"
RESULTS = ROOT / "results" / "frozen"

FULL_WIDTH_IN = 7.20
SINGLE_WIDTH_IN = 3.48
MIN_FONT_PT = 7.0

# Okabe-Ito-derived semantic palette used by the existing manuscript figures.
BLUE = "#0072B2"

ORANGE = "#E69F00"

SKY = "#56B4E9"

DARK = "#222222"
GRAY = "#666666"
MID_GRAY = "#A6A6A6"
LIGHT_GRAY = "#D9D9D9"
VERY_LIGHT_GRAY = "#F5F5F5"


WHITE = "#FFFFFF"

PROFILE_ORDER = (
    "fixed",
    "light_iid",
    "markov_burst",
    "feedback_heavy",
    "forward_heavy",
    "heavy_iid",
    "lognormal",
)
PROFILE_LABEL = {
    "fixed": "Fixed",
    "light_iid": "Light i.i.d.",
    "markov_burst": "Markov burst",
    "feedback_heavy": "Feedback-heavy",
    "forward_heavy": "Forward-heavy",
    "heavy_iid": "Heavy i.i.d.",
    "lognormal": "Log-normal",
}

CHANNEL_METHODS = {
    "cumulative_ucb_cv": ("Cumulative UCB", SKY, "o"),
    "sco_reset_ucb": ("SCO-reset-UCB", BLUE, "D"),
    "ps_forced_reset_ucb": ("Forced-reset-UCB", GRAY, "^"),
}
PIPELINE_METHODS = {
    "sco_reset_ucb": ("SCO-reset-UCB", BLUE, "D"),
    "inflight_sco_ucb": ("PA-SCO", ORANGE, "s"),
    "ps_forced_reset_ucb": ("Forced-reset-UCB", GRAY, "^"),
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.titlesize": 8.1,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.72,
            "lines.linewidth": 1.25,
            "lines.markersize": 4.4,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
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
    return digest.hexdigest().upper()


def read_csv(name: str) -> tuple[list[dict[str, str]], Path]:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"Required formal result artifact is missing: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Formal result artifact is empty: {path}")
    return rows, path


def require_fields(
    rows: Sequence[dict[str, str]], fields: Iterable[str], source: str
) -> None:
    missing = sorted(set(fields).difference(rows[0]))
    if missing:
        raise KeyError(f"{source} lacks required fields: {missing}")


def number(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key} in row: {row}")
    return value


def mean_ci(values: Iterable[float]) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size < 2:
        raise ValueError("A seed-level interval requires at least two values")
    mean = float(array.mean())
    sem = float(array.std(ddof=1) / math.sqrt(array.size))
    return mean, mean - 1.96 * sem, mean + 1.96 * sem


def draw_horizontal_intervals(
    ax,
    points: Sequence[tuple[float, float, float, float]],
    *,
    color: str,
    marker: str,
    label: str | None = None,
    zorder: int = 3,
) -> None:
    """Draw (y, mean, low, high) records as horizontal confidence intervals."""
    y = np.asarray([point[0] for point in points], dtype=float)
    mean = np.asarray([point[1] for point in points], dtype=float)
    low = np.asarray([point[2] for point in points], dtype=float)
    high = np.asarray([point[3] for point in points], dtype=float)
    if np.any(low > mean) or np.any(high < mean):
        raise AssertionError("A confidence interval does not contain its mean")
    ax.errorbar(
        mean,
        y,
        xerr=np.vstack([mean - low, high - mean]),
        fmt=marker,
        color=color,
        markerfacecolor=color,
        markeredgecolor=WHITE,
        markeredgewidth=0.55,
        capsize=2.0,
        capthick=0.8,
        linewidth=1.05,
        markersize=4.5,
        label=label,
        zorder=zorder,
    )


def panel_title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontweight="bold", pad=4.0)


def grid_x(ax) -> None:
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.50, zorder=0)
    ax.set_axisbelow(True)


def source_record(path: Path) -> dict[str, str | int]:
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def output_record(path: Path) -> dict[str, str | int]:
    return {
        "path": str(path.relative_to(HERE)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def audit_text(fig, stem: str) -> dict[str, object]:
    """Fail if rendered text is too small or lies outside the fixed canvas."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas_width, canvas_height = fig.canvas.get_width_height()
    minimum = math.inf
    text_count = 0
    overflow: list[str] = []
    boxes: list[tuple[str, object]] = []
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        text_count += 1
        minimum = min(minimum, float(artist.get_fontsize()))
        bbox = artist.get_window_extent(renderer=renderer)
        if bbox.width <= 0 or bbox.height <= 0:
            continue
        boxes.append((artist.get_text(), bbox))
        if (
            bbox.x0 < -1.0
            or bbox.y0 < -1.0
            or bbox.x1 > canvas_width + 1.0
            or bbox.y1 > canvas_height + 1.0
        ):
            overflow.append(artist.get_text())
    if minimum < MIN_FONT_PT - 1e-9:
        raise AssertionError(
            f"{stem}: minimum rendered font {minimum:.2f} pt is below "
            f"{MIN_FONT_PT:.1f} pt"
        )
    if overflow:
        raise AssertionError(f"{stem}: text outside fixed canvas: {overflow[:8]}")
    overlaps: list[dict[str, object]] = []
    for index, (left_text, left_box) in enumerate(boxes):
        left_area = max(1.0, float(left_box.width * left_box.height))
        for right_text, right_box in boxes[index + 1 :]:
            overlap_width = min(left_box.x1, right_box.x1) - max(left_box.x0, right_box.x0)
            overlap_height = min(left_box.y1, right_box.y1) - max(left_box.y0, right_box.y0)
            if overlap_width <= 1.0 or overlap_height <= 1.0:
                continue
            right_area = max(1.0, float(right_box.width * right_box.height))
            overlap_fraction = float(overlap_width * overlap_height) / min(
                left_area, right_area
            )
            if overlap_fraction >= 0.10:
                overlaps.append(
                    {
                        "left": left_text,
                        "right": right_text,
                        "overlap_fraction_of_smaller": overlap_fraction,
                    }
                )
    if overlaps:
        raise AssertionError(f"{stem}: significant text overlap: {overlaps[:5]}")
    return {
        "minimum_font_pt": float(minimum),
        "text_artist_count": text_count,
        "text_overflow_count": 0,
        "overflow_text": [],
        "significant_text_overlap_count": 0,
        "overlap_threshold_fraction": 0.10,
    }


def save_figure(
    fig,
    stem: str,
    *,
    sources: Sequence[Path],
    panel_map: dict[str, object],
    transformations: Sequence[str],
    plotted_values: dict[str, object],
    target_width_in: float = FULL_WIDTH_IN,
) -> dict[str, object]:
    width, height = map(float, fig.get_size_inches())
    if abs(width - target_width_in) > 1e-9:
        raise AssertionError(
            f"{stem}: width {width:.3f} in is not {target_width_in:.2f} in"
        )
    text_qa = audit_text(fig, stem)
    OUT.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, str | int]] = {}
    for suffix in ("pdf", "svg", "png"):
        path = OUT / f"{stem}.{suffix}"
        kwargs: dict[str, object] = {"bbox_inches": None, "facecolor": WHITE}
        if suffix == "png":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)
        outputs[suffix] = output_record(path)
    plt.close(fig)
    return {
        "stem": stem,
        "canvas": {
            "width_in": width,
            "height_in": height,
            "target_width_in": target_width_in,
        },
        "sources": [source_record(path) for path in sources],
        "panel_map": panel_map,
        "transformations": list(transformations),
        "plotted_values": plotted_values,
        "render_qa": {
            **text_qa,
            "fixed_target_width": "pass",
            "minimum_font_at_least_7pt": "pass",
            "no_text_outside_canvas": "pass",
            "pdf_svg_png_exported": "pass",
            "png_dpi": 300,
            "vector_text_preserved_in_svg": "pass",
        },
        "outputs": outputs,
    }


def same_numeric_record(
    first: dict[str, str],
    second: dict[str, str],
    fields: Sequence[str],
) -> bool:
    return all(abs(number(first, field) - number(second, field)) < 1e-12 for field in fields)


def make_channel_pipeline_figure() -> dict[str, object]:
    channel_rows, channel_path = read_csv("tmc_channel_stress_summary.csv")
    inflight_rows, inflight_path = read_csv("tmc_inflight_formal_raw.csv")
    require_fields(
        channel_rows,
        (
            "family",
            "method",
            "success_probability",
            "delay_slots",
            "capacity",
            "post_excess_pct_mean",
            "post_excess_pct_ci_low",
            "post_excess_pct_ci_high",
        ),
        channel_path.name,
    )
    require_fields(
        inflight_rows,
        (
            "seed",
            "family",
            "method",
            "success_probability",
            "delay_slots",
            "capacity",
            "post_cost",
            "redundant_attempt_rate",
        ),
        inflight_path.name,
    )
    seeds = sorted({int(number(row, "seed")) for row in inflight_rows})
    if len(seeds) != 12:
        raise AssertionError(f"Expected 12 formal inflight seeds, found {len(seeds)}")

    # These are the nine unique physical regimes in the original three
    # channel families.  The p=.9,d=0,N=4 and p=.9,d=1,N=4 anchors are
    # deliberately shown once rather than duplicated across family facets.
    regimes = [
        {
            "label": r"Delivery: $p=1.0$",
            "family": "delivery",
            "p": 1.0,
            "d": 0,
            "n": 4,
        },
        {
            "label": r"Delivery: $p=.9$ ($d=0$)",
            "family": "delivery",
            "p": 0.9,
            "d": 0,
            "n": 4,
        },
        {
            "label": r"Delivery: $p=.8$",
            "family": "delivery",
            "p": 0.8,
            "d": 0,
            "n": 4,
        },
        {
            "label": r"Delivery: $p=.7$",
            "family": "delivery",
            "p": 0.7,
            "d": 0,
            "n": 4,
        },
        {
            "label": r"Delay: $d=1$ ($N/K=.2$)",
            "family": "delay",
            "p": 0.9,
            "d": 1,
            "n": 4,
        },
        {
            "label": r"Delay: $d=3$",
            "family": "delay",
            "p": 0.9,
            "d": 3,
            "n": 4,
        },
        {
            "label": r"Delay: $d=5$",
            "family": "delay",
            "p": 0.9,
            "d": 5,
            "n": 4,
        },
        {
            "label": r"Capacity: $N/K=.4$",
            "family": "capacity",
            "p": 0.9,
            "d": 1,
            "n": 8,
        },
        {
            "label": r"Capacity: $N/K=.1$",
            "family": "capacity",
            "p": 0.9,
            "d": 1,
            "n": 2,
        },
    ]

    def channel_row(regime: dict[str, object], method: str) -> dict[str, str]:
        selected = [
            row
            for row in channel_rows
            if row["family"] == regime["family"]
            and row["method"] == method
            and abs(number(row, "success_probability") - float(regime["p"])) < 1e-12
            and int(number(row, "delay_slots")) == int(regime["d"])
            and int(number(row, "capacity")) == int(regime["n"])
        ]
        if len(selected) != 1:
            raise AssertionError(
                f"Expected one channel row for {regime['label']} / {method}, "
                f"found {len(selected)}"
            )
        return selected[0]

    # Prove that the omitted family duplicates are exact numeric duplicates.
    duplicate_fields = (
        "post_excess_pct_mean",
        "post_excess_pct_ci_low",
        "post_excess_pct_ci_high",
    )
    duplicate_pairs = [
        ("delivery", "delay", 0.9, 0, 4),
        ("delay", "capacity", 0.9, 1, 4),
    ]
    for first_family, second_family, p, d, n in duplicate_pairs:
        for method in CHANNEL_METHODS:
            pair = [
                row
                for row in channel_rows
                if row["family"] in (first_family, second_family)
                and row["method"] == method
                and abs(number(row, "success_probability") - p) < 1e-12
                and int(number(row, "delay_slots")) == d
                and int(number(row, "capacity")) == n
            ]
            if len(pair) != 2 or not same_numeric_record(
                pair[0], pair[1], duplicate_fields
            ):
                raise AssertionError(
                    "A cross-family channel anchor is not an exact duplicate: "
                    f"{first_family}/{second_family}/{method}"
                )

    settings = [
        {
            "label": r"Delay: $d=0$",
            "family": "delay",
            "p": 0.9,
            "d": 0,
            "n": 4,
        },
        {
            "label": r"Delay: $d=1$",
            "family": "delay",
            "p": 0.9,
            "d": 1,
            "n": 4,
        },
        {
            "label": r"Delay: $d=3$",
            "family": "delay",
            "p": 0.9,
            "d": 3,
            "n": 4,
        },
        {
            "label": r"Delay: $d=5$",
            "family": "delay",
            "p": 0.9,
            "d": 5,
            "n": 4,
        },
        {
            "label": r"Capacity: $N/K=.1$",
            "family": "capacity",
            "p": 0.9,
            "d": 1,
            "n": 2,
        },
        {
            "label": r"Capacity: $N/K=.4$",
            "family": "capacity",
            "p": 0.9,
            "d": 1,
            "n": 8,
        },
    ]

    def setting_rows(setting: dict[str, object]) -> list[dict[str, str]]:
        selected = [
            row
            for row in inflight_rows
            if row["family"] == setting["family"]
            and abs(number(row, "success_probability") - float(setting["p"])) < 1e-12
            and int(number(row, "delay_slots")) == int(setting["d"])
            and int(number(row, "capacity")) == int(setting["n"])
        ]
        expected = len(seeds) * 4  # true + SCO + forced + PA-SCO
        if len(selected) != expected:
            raise AssertionError(
                f"{setting['label']}: expected {expected} raw rows, "
                f"found {len(selected)}"
            )
        return selected

    gain_manifest: dict[str, dict[str, list[float]]] = {}
    rate_manifest: dict[str, dict[str, list[float]]] = {}
    gain_points: dict[str, list[tuple[float, float, float, float]]] = {
        "vs_sco": [],
        "vs_forced": [],
    }
    rate_points: dict[str, list[tuple[float, float, float, float]]] = {
        method: [] for method in PIPELINE_METHODS
    }
    y_setting = np.arange(len(settings), dtype=float)[::-1]
    for y, setting in zip(y_setting, settings):
        rows = setting_rows(setting)
        indexed: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
        for row in rows:
            indexed[int(number(row, "seed"))][row["method"]] = row
        vs_sco: list[float] = []
        vs_forced: list[float] = []
        method_rates: dict[str, list[float]] = {
            method: [] for method in PIPELINE_METHODS
        }
        for seed in seeds:
            methods = indexed[seed]
            required = set(PIPELINE_METHODS)
            if not required.issubset(methods):
                raise AssertionError(
                    f"{setting['label']} seed {seed} lacks {required - set(methods)}"
                )
            sco = number(methods["sco_reset_ucb"], "post_cost")
            forced = number(methods["ps_forced_reset_ucb"], "post_cost")
            pa = number(methods["inflight_sco_ucb"], "post_cost")
            vs_sco.append(100.0 * (sco - pa) / sco)
            vs_forced.append(100.0 * (forced - pa) / forced)
            for method in PIPELINE_METHODS:
                method_rates[method].append(
                    100.0 * number(methods[method], "redundant_attempt_rate")
                )
        sco_summary = mean_ci(vs_sco)
        forced_summary = mean_ci(vs_forced)
        gain_points["vs_sco"].append((y + 0.11, *sco_summary))
        gain_points["vs_forced"].append((y - 0.11, *forced_summary))
        gain_manifest[str(setting["label"])] = {
            "pa_gain_vs_sco_pct": list(sco_summary),
            "pa_gain_vs_forced_pct": list(forced_summary),
        }
        for method in PIPELINE_METHODS:
            summary = mean_ci(method_rates[method])
            rate_points[method].append((y, *summary))
            rate_manifest.setdefault(str(setting["label"]), {})[method] = list(summary)

    first_gain = gain_manifest[str(settings[0]["label"])]["pa_gain_vs_sco_pct"]
    if any(abs(value) > 1e-12 for value in first_gain):
        raise AssertionError(f"d=0 PA-SCO/SCO equivalence failed: {first_gain}")

    fig = plt.figure(figsize=(FULL_WIDTH_IN, 5.30))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.205,
        right=0.985,
        bottom=0.105,
        top=0.905,
        height_ratios=[1.20, 1.0],
        hspace=0.72,
        wspace=0.24,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1], sharey=ax_b)

    y_regime = np.arange(len(regimes), dtype=float)[::-1]
    channel_manifest: dict[str, dict[str, list[float]]] = {}
    method_offsets = {
        "cumulative_ucb_cv": 0.20,
        "sco_reset_ucb": 0.0,
        "ps_forced_reset_ucb": -0.20,
    }
    for method, (label, color, marker) in CHANNEL_METHODS.items():
        points: list[tuple[float, float, float, float]] = []
        for y, regime in zip(y_regime, regimes):
            row = channel_row(regime, method)
            summary = (
                number(row, "post_excess_pct_mean"),
                number(row, "post_excess_pct_ci_low"),
                number(row, "post_excess_pct_ci_high"),
            )
            points.append((y + method_offsets[method], *summary))
            channel_manifest.setdefault(str(regime["label"]), {})[method] = list(
                summary
            )
        draw_horizontal_intervals(
            ax_a,
            points,
            color=color,
            marker=marker,
            label=label,
        )
    ax_a.axvline(0, color=DARK, linewidth=0.75)
    ax_a.set_xlim(-1.25, 19.3)
    ax_a.set_xticks([0, 5, 10, 15])
    ax_a.set_yticks(y_regime, [str(regime["label"]) for regime in regimes])
    ax_a.set_ylim(-0.65, len(regimes) - 0.35)
    ax_a.set_xlabel("Post-change excess over channel-matched Whittle (%)")
    panel_title(ax_a, "(a) Channel boundary: nine unique regimes")
    grid_x(ax_a)
    ax_a.axhspan(4.5, 8.5, color=VERY_LIGHT_GRAY, alpha=0.72, zorder=-2)
    ax_a.axhspan(1.5, 4.5, color=LIGHT_GRAY, alpha=0.26, zorder=-2)
    ax_a.axhspan(-0.5, 1.5, color=VERY_LIGHT_GRAY, alpha=0.72, zorder=-2)
    ax_a.legend(
        loc="upper right",
        ncol=3,
        frameon=False,
        handletextpad=0.35,
        columnspacing=0.9,
        borderaxespad=0.15,
    )

    draw_horizontal_intervals(
        ax_b,
        gain_points["vs_sco"],
        color=BLUE,
        marker="D",
        label="PA-SCO gain vs SCO",
    )
    draw_horizontal_intervals(
        ax_b,
        gain_points["vs_forced"],
        color=GRAY,
        marker="^",
        label="PA-SCO gain vs forced",
    )
    ax_b.axvline(0, color=DARK, linewidth=0.75)
    ax_b.set_xlim(-3.0, 90.0)
    ax_b.set_xticks([0, 25, 50, 75])
    ax_b.set_yticks(y_setting, [str(setting["label"]) for setting in settings])
    ax_b.set_ylim(-0.6, len(settings) - 0.4)
    ax_b.set_xlabel("Paired post-change cost reduction (%)")
    panel_title(ax_b, "(b) PA-SCO paired gain")
    grid_x(ax_b)
    ax_b.axhline(1.5, color=LIGHT_GRAY, linewidth=0.8)
    ax_b.text(
        1.7,
        y_setting[0] + 0.11,
        "exact 0",
        ha="left",
        va="center",
        color=BLUE,
        fontsize=7.0,
    )
    gain_handles, gain_labels = ax_b.get_legend_handles_labels()

    rate_offsets = {
        "sco_reset_ucb": 0.20,
        "inflight_sco_ucb": 0.0,
        "ps_forced_reset_ucb": -0.20,
    }
    for method, (label, color, marker) in PIPELINE_METHODS.items():
        offset_points = [
            (point[0] + rate_offsets[method], point[1], point[2], point[3])
            for point in rate_points[method]
        ]
        draw_horizontal_intervals(
            ax_c,
            offset_points,
            color=color,
            marker=marker,
            label=label,
        )
    ax_c.axvline(0, color=DARK, linewidth=0.75)
    ax_c.set_xlim(-3.0, 90.0)
    ax_c.set_xticks([0, 25, 50, 75])
    ax_c.tick_params(axis="y", labelleft=False)
    ax_c.set_xlabel("Attempts issued while stream already occupied (%)")
    panel_title(ax_c, "(c) Duplicate-attempt share")
    grid_x(ax_c)
    ax_c.axhline(1.5, color=LIGHT_GRAY, linewidth=0.8)
    rate_handles, rate_labels = ax_c.get_legend_handles_labels()
    gain_box = ax_b.get_position()
    rate_box = ax_c.get_position()
    fig.legend(
        gain_handles,
        gain_labels,
        loc="lower center",
        bbox_to_anchor=(gain_box.x0 + gain_box.width / 2.0, gain_box.y1 + 0.052),
        ncol=2,
        frameon=False,
        handletextpad=0.28,
        columnspacing=0.75,
        borderaxespad=0.0,
    )
    fig.legend(
        rate_handles,
        rate_labels,
        loc="lower center",
        bbox_to_anchor=(rate_box.x0 + rate_box.width / 2.0, rate_box.y1 + 0.052),
        ncol=3,
        frameon=False,
        handletextpad=0.28,
        columnspacing=0.60,
        borderaxespad=0.0,
    )

    return save_figure(
        fig,
        "fig_channel_pipeline_evidence_v21",
        sources=(channel_path, inflight_path),
        panel_map={
            "a": {
                "question": "How do delivery, delay, and capacity affect absolute excess cost?",
                "point_count": 27,
                "dimensions": "9 unique regimes x 3 methods; supplied summary intervals",
            },
            "b": {
                "question": "Does PA-SCO improve on both SCO and forced reset?",
                "point_count": 12,
                "dimensions": "6 unique settings x 2 paired comparators; 12 seeds",
                "exact_identity": "PA-SCO equals SCO at d=0",
            },
            "c": {
                "question": "Does the repair suppress duplicate service?",
                "point_count": 18,
                "dimensions": "6 unique settings x 3 methods; 12 seeds",
            },
        },
        transformations=(
            "Retained the scientifically coherent three-panel Fig. 6 structure but shortened headings and aligned the lower evidence axes.",
            "Removed six exact cross-family duplicate summary rows by showing each physical regime once.",
            "Panel (b) computes seed-paired 100*(reference_cost-PA_cost)/reference_cost for SCO and forced-reset references.",
            "Panel (c) converts redundant-attempt fractions to percentage points and computes normal 95% seed-level intervals.",
            "The duplicated d=1,N=4 capacity-family raw block is omitted; the delay-family record is used once.",
            "The weak r=0.11 association panel is intentionally not reproduced; its manuscript caveat remains separate.",
        ),
        plotted_values={
            "panel_a": channel_manifest,
            "panel_b": gain_manifest,
            "panel_c": rate_manifest,
        },
    )


def prepare_random_delay_evidence() -> dict[str, object]:
    """Load the frozen random-delay results and reproduce v19 statistics."""
    paired_raw, paired_raw_path = read_csv("tmc_random_delay_formal_paired_raw.csv")
    paired_summary, paired_summary_path = read_csv(
        "tmc_random_delay_formal_paired_summary.csv"
    )
    he_paired, he_paired_path = read_csv(
        "tmc_he_rm_formal_addendum_paired_summary.csv"
    )
    he_summary, he_summary_path = read_csv("tmc_he_rm_formal_addendum_summary.csv")
    require_fields(
        paired_raw,
        (
            "seed",
            "profile",
            "round_trip_mean",
            "round_trip_p95",
            "round_trip_p99",
            "sco_stale_arrival_rate",
            "pa_stale_arrival_rate",
        ),
        paired_raw_path.name,
    )
    require_fields(
        paired_summary,
        (
            "profile",
            "seeds",
            "pa_reduction_vs_sco_pct_mean",
            "pa_reduction_vs_sco_pct_ci_low",
            "pa_reduction_vs_sco_pct_ci_high",
            "pa_redundant_attempt_rate_mean",
            "pa_redundant_attempt_rate_ci_low",
            "pa_redundant_attempt_rate_ci_high",
        ),
        paired_summary_path.name,
    )
    require_fields(
        he_paired,
        (
            "profile",
            "seeds",
            "pa_reduction_vs_he_pct_mean",
            "pa_reduction_vs_he_pct_ci_low",
            "pa_reduction_vs_he_pct_ci_high",
        ),
        he_paired_path.name,
    )
    require_fields(
        he_summary,
        (
            "profile",
            "method",
            "seeds",
            "capacity_utilization_mean",
            "capacity_utilization_ci_low",
            "capacity_utilization_ci_high",
        ),
        he_summary_path.name,
    )
    seed_ids = sorted({int(number(row, "seed")) for row in paired_raw})
    if len(seed_ids) != 12 or len(paired_raw) != 12 * len(PROFILE_ORDER):
        raise AssertionError(
            f"Random-delay bank mismatch: {len(seed_ids)} seeds, {len(paired_raw)} rows"
        )
    ps = {row["profile"]: row for row in paired_summary}
    hp = {row["profile"]: row for row in he_paired}
    hs = {row["profile"]: row for row in he_summary if row["method"] == "he_rm"}
    if any(
        profile not in ps or profile not in hp or profile not in hs
        for profile in PROFILE_ORDER
    ):
        raise AssertionError("One or more random-delay profiles are incomplete")

    y_profile = np.arange(len(PROFILE_ORDER), dtype=float)[::-1]
    tail_offsets = {
        "round_trip_mean": 0.18,
        "round_trip_p95": 0.0,
        "round_trip_p99": -0.18,
    }
    tail_styles = {
        "round_trip_mean": ("Mean RTT", DARK, "o"),
        "round_trip_p95": ("RTT p95", MID_GRAY, "^"),
        "round_trip_p99": ("RTT p99", GRAY, "D"),
    }
    tail_points = {field: [] for field in tail_styles}
    tail_manifest: dict[str, dict[str, list[float]]] = {}
    for y, profile in zip(y_profile, PROFILE_ORDER):
        rows = [row for row in paired_raw if row["profile"] == profile]
        if len(rows) != 12:
            raise AssertionError(f"{profile}: expected 12 tail rows, found {len(rows)}")
        for field in tail_styles:
            summary = mean_ci(number(row, field) for row in rows)
            tail_points[field].append((y + tail_offsets[field], *summary))
            tail_manifest.setdefault(profile, {})[field] = list(summary)

    gain_points = {"vs_sco": [], "vs_rm_ack": []}
    gain_manifest: dict[str, dict[str, list[float]]] = {}
    mechanism_points = {"pa_duplicate": [], "rm_idle": [], "stale_delta": []}
    mechanism_manifest: dict[str, dict[str, list[float]]] = {}
    for y, profile in zip(y_profile, PROFILE_ORDER):
        sco_gain = (
            number(ps[profile], "pa_reduction_vs_sco_pct_mean"),
            number(ps[profile], "pa_reduction_vs_sco_pct_ci_low"),
            number(ps[profile], "pa_reduction_vs_sco_pct_ci_high"),
        )
        rm_gain = (
            number(hp[profile], "pa_reduction_vs_he_pct_mean"),
            number(hp[profile], "pa_reduction_vs_he_pct_ci_low"),
            number(hp[profile], "pa_reduction_vs_he_pct_ci_high"),
        )
        gain_points["vs_sco"].append((y + 0.11, *sco_gain))
        gain_points["vs_rm_ack"].append((y - 0.11, *rm_gain))
        gain_manifest[profile] = {
            "pa_gain_vs_sco_pct": list(sco_gain),
            "pa_gain_vs_rm_ack_pct": list(rm_gain),
        }
        pa_duplicate = (
            100.0 * number(ps[profile], "pa_redundant_attempt_rate_mean"),
            100.0 * number(ps[profile], "pa_redundant_attempt_rate_ci_low"),
            100.0 * number(ps[profile], "pa_redundant_attempt_rate_ci_high"),
        )
        rm_idle = (
            100.0 * (1.0 - number(hs[profile], "capacity_utilization_mean")),
            100.0 * (1.0 - number(hs[profile], "capacity_utilization_ci_high")),
            100.0 * (1.0 - number(hs[profile], "capacity_utilization_ci_low")),
        )
        profile_raw = [row for row in paired_raw if row["profile"] == profile]
        stale_delta = mean_ci(
            100.0
            * (
                number(row, "pa_stale_arrival_rate")
                - number(row, "sco_stale_arrival_rate")
            )
            for row in profile_raw
        )
        mechanism_points["pa_duplicate"].append((y + 0.20, *pa_duplicate))
        mechanism_points["rm_idle"].append((y, *rm_idle))
        mechanism_points["stale_delta"].append((y - 0.20, *stale_delta))
        mechanism_manifest[profile] = {
            "pa_duplicate_share_pct": list(pa_duplicate),
            "rm_ack_idle_share_pct": list(rm_idle),
            "paired_stale_change_pa_minus_sco_points": list(stale_delta),
        }
    return {
        "sources": (
            paired_raw_path,
            paired_summary_path,
            he_paired_path,
            he_summary_path,
        ),
        "y_profile": y_profile,
        "tail_styles": tail_styles,
        "tail_points": tail_points,
        "tail_manifest": tail_manifest,
        "gain_points": gain_points,
        "gain_manifest": gain_manifest,
        "mechanism_points": mechanism_points,
        "mechanism_manifest": mechanism_manifest,
    }


def make_random_delay_profiles_figure() -> dict[str, object]:
    data = prepare_random_delay_evidence()
    y_profile = data["y_profile"]
    tail_manifest = data["tail_manifest"]
    fig, ax = plt.subplots(figsize=(SINGLE_WIDTH_IN, 3.25))
    fig.subplots_adjust(left=0.325, right=0.965, bottom=0.18, top=0.76)

    for y, profile in zip(y_profile, PROFILE_ORDER):
        mean_value = tail_manifest[profile]["round_trip_mean"][0]
        p99_value = tail_manifest[profile]["round_trip_p99"][0]
        ax.hlines(y, mean_value, p99_value, color=LIGHT_GRAY, linewidth=1.0, zorder=1)
    for field, (label, color, marker) in data["tail_styles"].items():
        draw_horizontal_intervals(
            ax,
            data["tail_points"][field],
            color=color,
            marker=marker,
            label=label,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlim(3.55, 39.0)
    ax.set_xticks([4, 8, 16, 32], ["4", "8", "16", "32"])
    ax.set_yticks(y_profile, [PROFILE_LABEL[profile] for profile in PROFILE_ORDER])
    ax.set_ylim(-0.65, len(PROFILE_ORDER) - 0.35)
    ax.set_xlabel("Round-trip delay (slots, log$_2$ scale)")
    panel_title(ax, "Round-trip tail signatures")
    grid_x(ax)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.50, 1.12),
        ncol=3,
        frameon=False,
        handletextpad=0.25,
        columnspacing=0.65,
        borderaxespad=0.0,
    )
    return save_figure(
        fig,
        "fig_random_delay_profiles_v21",
        sources=data["sources"],
        panel_map={
            "single": {
                "question": "Do comparable-mean delay profiles have distinct upper-tail signatures?",
                "point_count": 21,
                "dimensions": "7 profiles x mean/p95/p99; normal 95% intervals over 12 formal seeds",
                "role": "random-delay condition definition, separated from method outcomes",
            }
        },
        transformations=(
            "Reproduces the v19 all-seed RTT mean, p95, and p99 summaries without changing values.",
            "Places the condition-definition evidence in a standalone single-column figure.",
            "Light horizontal spans connect each profile mean to p99 but do not encode uncertainty.",
            "The glyphs summarize per-seed RTT statistics; they are not packet-level empirical CDFs.",
        ),
        plotted_values={"tail_signatures": tail_manifest},
        target_width_in=SINGLE_WIDTH_IN,
    )
def make_random_delay_performance_mechanism_figure() -> dict[str, object]:
    data = prepare_random_delay_evidence()
    y_profile = data["y_profile"]
    fig = plt.figure(figsize=(FULL_WIDTH_IN, 3.28))
    gs = fig.add_gridspec(
        1,
        2,
        left=0.145,
        right=0.988,
        bottom=0.18,
        top=0.76,
        width_ratios=[1.04, 1.0],
        wspace=0.34,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)

    draw_horizontal_intervals(
        ax_a,
        data["gain_points"]["vs_sco"],
        color=BLUE,
        marker="D",
        label="Gain vs SCO",
    )
    draw_horizontal_intervals(
        ax_a,
        data["gain_points"]["vs_rm_ack"],
        color=GRAY,
        marker="P",
        label="Gain vs RM-ACK",
    )
    ax_a.axvline(0, color=DARK, linewidth=0.75)
    ax_a.set_xlim(-3.0, 100.0)
    ax_a.set_xticks([0, 25, 50, 75, 100])
    ax_a.set_yticks(y_profile, [PROFILE_LABEL[profile] for profile in PROFILE_ORDER])
    ax_a.set_ylim(-0.65, len(PROFILE_ORDER) - 0.35)
    ax_a.set_xlabel("PA-SCO paired cost reduction (%)")
    panel_title(ax_a, "(a) Gain over both comparators")
    grid_x(ax_a)

    draw_horizontal_intervals(
        ax_b,
        data["mechanism_points"]["pa_duplicate"],
        color=ORANGE,
        marker="s",
        label="PA duplicate",
    )
    draw_horizontal_intervals(
        ax_b,
        data["mechanism_points"]["rm_idle"],
        color=MID_GRAY,
        marker="P",
        label="RM-ACK idle",
    )
    draw_horizontal_intervals(
        ax_b,
        data["mechanism_points"]["stale_delta"],
        color=DARK,
        marker="v",
        label=r"$\Delta$ stale: PA$-$SCO",
    )
    ax_b.axvline(0, color=DARK, linewidth=0.75, linestyle=(0, (3, 2)))
    ax_b.set_xlim(-12.0, 50.0)
    ax_b.set_xticks([-10, 0, 20, 40])
    ax_b.tick_params(axis="y", labelleft=False)
    ax_b.set_xlabel("Share or paired change (percentage points)")
    panel_title(ax_b, "(b) Concurrency/stale-arrival boundary")
    grid_x(ax_b)

    legend_handles = [
        Line2D([0], [0], color=BLUE, marker="D", linestyle="none", label="Gain vs SCO"),
        Line2D([0], [0], color=GRAY, marker="P", linestyle="none", label="Gain vs RM-ACK"),
        Line2D([0], [0], color=ORANGE, marker="s", linestyle="none", label="PA duplicate"),
        Line2D([0], [0], color=MID_GRAY, marker="P", linestyle="none", label="RM-ACK idle"),
        Line2D(
            [0],
            [0],
            color=DARK,
            marker="v",
            linestyle="none",
            label=r"$\Delta$ stale: PA$-$SCO",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.515, 0.985),
        ncol=5,
        frameon=False,
        handletextpad=0.25,
        columnspacing=0.70,
        borderaxespad=0.0,
    )
    return save_figure(
        fig,
        "fig_random_delay_performance_mechanism_v21",
        sources=data["sources"],
        panel_map={
            "a": {
                "question": "Does PA-SCO improve on both SCO and retrospective RM-ACK?",
                "point_count": 14,
                "dimensions": "7 profiles x 2 paired comparators; supplied 95% intervals",
            },
            "b": {
                "question": "What duplicate/idle trade-off remains, and can stale arrivals increase?",
                "point_count": 21,
                "dimensions": "7 profiles x PA duplicate / RM-ACK idle / paired stale change",
            },
        },
        transformations=(
            "Reproduces the v19 paired gain and mechanism values without changing inputs or inferential boundaries.",
            "Separates outcome/mechanism evidence from the delay-condition signature figure.",
            "PA duplicate and RM-ACK idle fractions are expressed as percentage points.",
            "RM-ACK idle endpoints use the monotone transform 100*(1-capacity utilization), with interval endpoints reversed.",
            "Stale change is the seed-paired 100*(PA stale rate-SCO stale rate) difference with a normal 95% interval.",
            "Positive stale-change values remain visible as a limitation; no causal mediation claim is made.",
        ),
        plotted_values={
            "performance": data["gain_manifest"],
            "mechanism": data["mechanism_manifest"],
        },
    )

def main() -> None:
    setup_style()
    records = [
        make_channel_pipeline_figure(),
        make_random_delay_profiles_figure(),
        make_random_delay_performance_mechanism_figure(),
    ]
    manifest = {
        "schema_version": 1,
        "renderer": Path(__file__).name,
        "renderer_sha256": sha256(Path(__file__)),
        "output_directory": OUT.name,
        "design_contract": {
            "target": "IEEE TMC single- and full-width figures",
            "canvas_width_in": {"single": SINGLE_WIDTH_IN, "full": FULL_WIDTH_IN},
            "minimum_font_pt": MIN_FONT_PT,
            "data_policy": "existing formal artifacts only; no invented observations or experiments",
            "palette_policy": "SCO deep blue; PA-SCO orange; learning light blue; forced/RM-ACK/conditions neutral gray; stale boundary dark-gray marker and dashed zero line",
            "palette_mapping": {
                "SCO": BLUE,
                "PA-SCO": ORANGE,
                "learning": SKY,
                "other_comparators": [GRAY, MID_GRAY],
                "condition_signatures": [DARK, MID_GRAY, GRAY],
            },
            "layout_policy": "Fig. 6 lightly harmonized; random-delay condition definition split from performance/mechanism evidence",
            "manuscript_policy": "renderer writes only figs_tmc_v21 and does not modify TeX, v20, or earlier assets",
        },
        "figures": records,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "fig_channel_delay_v21_qa.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "figures": [record["stem"] for record in records],
                "minimum_fonts": {
                    record["stem"]: record["render_qa"]["minimum_font_pt"]
                    for record in records
                },
                "overflow_counts": {
                    record["stem"]: record["render_qa"]["text_overflow_count"]
                    for record in records
                },
                "overlap_counts": {
                    record["stem"]: record["render_qa"]["significant_text_overlap_count"]
                    for record in records
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
