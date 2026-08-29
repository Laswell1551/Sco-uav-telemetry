"""Redraw TMC Figure 4 with family-consistent method encoding.

This script reads only the frozen artifacts already used by the v16 figure.
It writes a new v21 figure and a machine-readable render QA ledger; it does
not modify any manuscript or previous figure asset.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.text import Text


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results" / "frozen"
OUT = HERE / "generated"

FIG_W = 7.2
FIG_H = 4.8
MIN_FONT_PT = 7.0

# Family semantics requested for v21.
EXISTING = "#777777"
LEARNING = "#56B4E9"
PROPOSED = "#0072B2"
DARK = "#222222"
MID = "#666666"
LIGHT = "#D9D9D9"
PALE = "#F5F5F5"
LEARNING_FILL = "#EDF7FC"
PROPOSED_FILL = "#E5F1F8"
WHITE = "#FFFFFF"


def setup_style() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.4,
            "axes.titlesize": 8.6,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.25,
            "lines.markersize": 4.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": WHITE,
            "figure.facecolor": WHITE,
        }
    )


def read_json(name: str):
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen result artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_summary() -> list[dict[str, str]]:
    path = RESULTS / "tmc_confirmatory_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen paper table: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Frozen paper table is empty: {path}")
    return rows


def num(row: dict[str, str], key: str) -> float:
    return float(row[key])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    """Match the frozen v16 bootstrap recipe exactly."""
    rng = np.random.default_rng(seed)
    means = np.empty(100000, dtype=float)
    for start in range(0, len(means), 5000):
        stop = min(start + 5000, len(means))
        draw = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[draw].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975]).tolist()
    return float(values.mean()), float(low), float(high)


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color=LIGHT, linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)


def panel_title(ax, title: str) -> None:
    ax.text(
        0.0,
        1.045,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.6,
        fontweight="bold",
        color=DARK,
    )


def collect_data():
    expansion = read_json("tmc_v16_baseline_expansion.json")
    ts = read_json("tmc_ts_baseline_expansion.json")
    paired_ts = read_json("tmc_ts_paired_sco_addendum.json")
    max_age = read_json("tmc_external_baseline_addendum_v1.json")
    frozen_rows = read_summary()
    frozen = {row["method"]: row for row in frozen_rows}
    formal = expansion["formal"]["seed_cluster_bootstrap"]["summaries"]

    # Existing -> learning -> proposed. Values are unchanged from v16.
    entries = [
        (
            "Cumulative CE",
            num(frozen["cumulative_ce"], "post_excess_mean"),
            num(frozen["cumulative_ce"], "post_excess_ci_low"),
            num(frozen["cumulative_ce"], "post_excess_ci_high"),
            "Existing",
            "o",
        ),
        (
            "Cumulative UCB-CV",
            num(frozen["cumulative_ucb_cv"], "post_excess_mean"),
            num(frozen["cumulative_ucb_cv"], "post_excess_ci_low"),
            num(frozen["cumulative_ucb_cv"], "post_excess_ci_high"),
            "Existing",
            "s",
        ),
        (
            "Max age",
            float(max_age["summary"]["post_ex_mean"]),
            float(max_age["summary"]["post_ex_cluster_ci"][0]),
            float(max_age["summary"]["post_ex_cluster_ci"][1]),
            "Existing",
            "^",
        ),
        (
            "SW-CE (32)",
            num(frozen["sw_ce_32"], "post_excess_mean"),
            num(frozen["sw_ce_32"], "post_excess_ci_low"),
            num(frozen["sw_ce_32"], "post_excess_ci_high"),
            "Learning",
            "o",
        ),
        (
            "SW-Whittle-CV (64)",
            num(frozen["sw_ucb_cv_64"], "post_excess_mean"),
            num(frozen["sw_ucb_cv_64"], "post_excess_ci_low"),
            num(frozen["sw_ucb_cv_64"], "post_excess_ci_high"),
            "Learning",
            "s",
        ),
        (
            "DTS-Whittle-CV",
            float(formal["dts_whittle_cv"]["post_excess_cost_pct"]["mean"]),
            float(formal["dts_whittle_cv"]["post_excess_cost_pct"]["ci95"][0]),
            float(formal["dts_whittle_cv"]["post_excess_cost_pct"]["ci95"][1]),
            "Learning",
            "P",
        ),
        (
            "TS-Whittle-CV",
            float(ts["formal_summary"]["post_ex"]["mean"]),
            float(ts["formal_summary"]["post_ex"]["ci95"][0]),
            float(ts["formal_summary"]["post_ex"]["ci95"][1]),
            "Learning",
            "X",
        ),
        (
            "DE-CD-Whittle-CV",
            float(formal["de_cd_whittle_cv"]["post_excess_cost_pct"]["mean"]),
            float(formal["de_cd_whittle_cv"]["post_excess_cost_pct"]["ci95"][0]),
            float(formal["de_cd_whittle_cv"]["post_excess_cost_pct"]["ci95"][1]),
            "Learning",
            "v",
        ),
        (
            "Forced-reset-UCB",
            num(frozen["ps_forced_reset_ucb"], "post_excess_mean"),
            num(frozen["ps_forced_reset_ucb"], "post_excess_ci_low"),
            num(frozen["ps_forced_reset_ucb"], "post_excess_ci_high"),
            "Learning",
            "D",
        ),
        (
            "SCO-reset-CE",
            num(frozen["sco_reset_ce"], "post_excess_mean"),
            num(frozen["sco_reset_ce"], "post_excess_ci_low"),
            num(frozen["sco_reset_ce"], "post_excess_ci_high"),
            "Proposed",
            "d",
        ),
        (
            "SCO-reset-UCB",
            num(frozen["sco_reset_ucb"], "post_excess_mean"),
            num(frozen["sco_reset_ucb"], "post_excess_ci_low"),
            num(frozen["sco_reset_ucb"], "post_excess_ci_high"),
            "Proposed",
            "D",
        ),
    ]

    sco_by_seed = {
        int(row["seed"]): float(row["post_ex_seed_mean"])
        for row in paired_ts["sco_rows"]
    }
    raw_by_method_seed: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in expansion["formal"]["raw_seed_batch_rows"]:
        raw_by_method_seed[(row["method"], int(row["seed"]))].append(
            float(row["post_excess_cost_pct"])
        )
    paired_values = {
        "TS-Whittle-CV": np.asarray(
            paired_ts["comparisons"]["post_ex"]["paired_seed_values"], dtype=float
        )
    }
    for method, label in (
        ("dts_whittle_cv", "DTS-Whittle-CV"),
        ("de_cd_whittle_cv", "DE-CD-Whittle-CV"),
    ):
        paired_values[label] = np.asarray(
            [
                np.mean(raw_by_method_seed[(method, seed)]) - sco_by_seed[seed]
                for seed in sorted(sco_by_seed)
            ],
            dtype=float,
        )

    effects = [
        (
            "DTS-Whittle-CV",
            *bootstrap_mean(paired_values["DTS-Whittle-CV"], 316101),
            "P",
        ),
        (
            "TS-Whittle-CV",
            float(paired_ts["comparisons"]["post_ex"]["paired_difference_mean"]),
            float(paired_ts["comparisons"]["post_ex"]["paired_difference_ci95"][0]),
            float(paired_ts["comparisons"]["post_ex"]["paired_difference_ci95"][1]),
            "X",
        ),
        (
            "DE-CD-Whittle-CV",
            *bootstrap_mean(paired_values["DE-CD-Whittle-CV"], 316102),
            "v",
        ),
    ]

    trade_metrics = [
        (
            "Recall (%)",
            [
                100.0 * num(frozen["sco_reset_ucb"], "detection"),
                100.0 * num(frozen["ps_forced_reset_ucb"], "detection"),
                100.0 * float(formal["de_cd_whittle_cv"]["detection_fraction"]["mean"]),
            ],
            (93.5, 97.0),
        ),
        (
            "Delay (slots)",
            [
                num(frozen["sco_reset_ucb"], "observation_delay"),
                num(frozen["ps_forced_reset_ucb"], "observation_delay"),
                float(formal["de_cd_whittle_cv"]["mean_observation_delay"]["mean"]),
            ],
            (7.25, 7.70),
        ),
        (
            "Cost (%)",
            [
                num(frozen["sco_reset_ucb"], "post_excess_mean"),
                num(frozen["ps_forced_reset_ucb"], "post_excess_mean"),
                float(formal["de_cd_whittle_cv"]["post_excess_cost_pct"]["mean"]),
            ],
            (0.0, 28.5),
        ),
    ]
    return entries, paired_values, effects, trade_metrics


def render_qa(fig, source_paths: list[Path], point_counts: dict[str, int]) -> dict:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_box = fig.bbox
    texts = [
        obj
        for obj in fig.findobj(match=Text)
        if obj.get_visible() and obj.get_text().strip()
    ]
    overflow = []
    text_boxes = []
    for obj in texts:
        box = obj.get_window_extent(renderer=renderer)
        text_boxes.append((obj, box))
        sides = {
            "left": max(0.0, fig_box.x0 - box.x0),
            "right": max(0.0, box.x1 - fig_box.x1),
            "bottom": max(0.0, fig_box.y0 - box.y0),
            "top": max(0.0, box.y1 - fig_box.y1),
        }
        if max(sides.values()) > 0.75:
            overflow.append({"text": obj.get_text(), "overflow_px": sides})

    overlaps = []
    for idx, (left, left_box) in enumerate(text_boxes):
        for right, right_box in text_boxes[idx + 1 :]:
            x_overlap = min(left_box.x1, right_box.x1) - max(left_box.x0, right_box.x0)
            y_overlap = min(left_box.y1, right_box.y1) - max(left_box.y0, right_box.y0)
            if x_overlap > 1.0 and y_overlap > 1.0:
                overlaps.append(
                    {
                        "a": left.get_text(),
                        "b": right.get_text(),
                        "area_px2": round(float(x_overlap * y_overlap), 2),
                    }
                )

    fonts = [float(obj.get_fontsize()) for obj in texts]
    rel_sources = [
        {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(path),
        }
        for path in source_paths
    ]
    passed = min(fonts) >= MIN_FONT_PT and not overflow and not overlaps
    return {
        "artifact": "fig_controlled_grouped_v21",
        "target": "IEEE TMC two-column full-width figure",
        "canvas_inches": [FIG_W, FIG_H],
        "media_box_pt": [FIG_W * 72.0, FIG_H * 72.0],
        "minimum_font_pt": min(fonts),
        "required_minimum_font_pt": MIN_FONT_PT,
        "text_object_count": len(texts),
        "text_overflow_count": len(overflow),
        "text_overlap_count": len(overlaps),
        "text_overflows": overflow,
        "text_overlaps": overlaps,
        "point_counts": point_counts,
        "family_encoding": {
            "Existing": EXISTING,
            "Learning": LEARNING,
            "Proposed": PROPOSED,
        },
        "source_artifacts": rel_sources,
        "visual_contract": {
            "core_claim": "SCO remains the low-cost method after the controlled change.",
            "panel_a": "family-grouped post-change forest plot with unchanged 95% intervals",
            "panel_b": "paired 30-seed baseline-minus-SCO effects",
            "panel_c": "detector/service trade-off for SCO, forced reset, and DE-CD",
            "numeric_point_labels": 0,
            "no_fabrication": "All values are read from the five frozen v16 source artifacts.",
        },
        "pass": passed,
    }


def make_figure() -> dict:
    entries, paired_values, effects, trade_metrics = collect_data()
    family_color = {"Existing": EXISTING, "Learning": LEARNING, "Proposed": PROPOSED}

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    outer = fig.add_gridspec(
        2,
        2,
        left=0.205,
        right=0.985,
        bottom=0.125,
        top=0.935,
        height_ratios=[1.72, 1.0],
        width_ratios=[1.05, 1.0],
        hspace=0.52,
        wspace=0.42,
    )
    ax_a = fig.add_subplot(outer[0, :])
    ax_b = fig.add_subplot(outer[1, 0])
    trade_grid = outer[1, 1].subgridspec(1, 3, wspace=0.42)
    trade_axes = [fig.add_subplot(trade_grid[0, idx]) for idx in range(3)]

    # Panel (a): explicit Existing -> Learning -> Proposed grouping.
    y_positions = np.asarray([10.0, 9.0, 8.0, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5, 0.0, -1.0])
    ax_a.axhspan(7.5, 10.5, color=PALE, zorder=0)
    ax_a.axhspan(1.0, 7.0, color=LEARNING_FILL, zorder=0)
    ax_a.axhspan(-1.5, 0.5, color=PROPOSED_FILL, zorder=0)
    for yy, (method, mean, low, high, family, marker) in zip(y_positions, entries):
        color = family_color[family]
        ax_a.errorbar(
            mean,
            yy,
            xerr=np.asarray([[mean - low], [high - mean]]),
            fmt=marker,
            color=color,
            markerfacecolor=WHITE if family == "Existing" else color,
            markeredgecolor=color,
            markeredgewidth=0.9,
            capsize=2.4,
            markersize=5.0,
            linewidth=1.15,
            zorder=3,
        )
    ax_a.axvline(entries[-1][1], color=PROPOSED, linestyle=":", linewidth=0.9, alpha=0.9)
    ax_a.set_yticks(y_positions, [row[0] for row in entries])
    for tick, entry in zip(ax_a.get_yticklabels(), entries):
        tick.set_color(family_color[entry[4]])
        if entry[4] == "Proposed":
            tick.set_fontweight("bold")
    ax_a.set_ylim(-1.65, 10.65)
    ax_a.set_xlim(2.0, 33.0)
    ax_a.set_xticks([5, 10, 15, 20, 25, 30])
    ax_a.set_xlabel("Post-change excess cost over true-model Whittle (%)")
    style_axis(ax_a)
    panel_title(ax_a, "(a) Controlled-drift benchmark (95% intervals)")
    family_handles = [
        Line2D([], [], marker="o", linestyle="-", color=EXISTING, markerfacecolor=WHITE,
               label="Existing"),
        Line2D([], [], marker="o", linestyle="-", color=LEARNING, label="Learning"),
        Line2D([], [], marker="D", linestyle="-", color=PROPOSED, label="Proposed"),
    ]
    ax_a.legend(
        handles=family_handles,
        loc="lower right",
        ncol=3,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.5,
        handletextpad=0.4,
        borderaxespad=0.5,
    )

    # Panel (b): seed-level paired effects plus mean 95% intervals, no point labels.
    effect_y = np.arange(len(effects))[::-1]
    rng = np.random.default_rng(20260728)
    all_effects: list[float] = []
    for yy, (method, mean, low, high, marker) in zip(effect_y, effects):
        values = paired_values[method]
        all_effects.extend(values.tolist())
        jitter = rng.uniform(-0.12, 0.12, len(values))
        ax_b.scatter(
            values,
            yy + jitter,
            s=10,
            facecolor=WHITE,
            edgecolor=LEARNING,
            linewidth=0.6,
            alpha=0.45,
            zorder=2,
        )
        ax_b.errorbar(
            mean,
            yy,
            xerr=np.asarray([[mean - low], [high - mean]]),
            fmt=marker,
            color=LEARNING,
            markeredgecolor=PROPOSED,
            markeredgewidth=0.65,
            capsize=2.5,
            markersize=5.2,
            linewidth=1.3,
            zorder=3,
        )
    ax_b.axvline(0, color=PROPOSED, linestyle="--", linewidth=0.9)
    ax_b.set_xscale("symlog", linthresh=2.0, linscale=1.0)
    limit = max(35.0, max(all_effects) * 1.10)
    ax_b.set_xlim(-3.0, limit)
    ax_b.set_xticks([-2, 0, 2, 10, 40], ["-2", "0", "2", "10", "40"])
    ax_b.set_yticks(effect_y, [row[0] for row in effects])
    ax_b.set_xlabel("Baseline $-$ SCO post-change cost (points)")
    style_axis(ax_b)
    panel_title(ax_b, "(b) Paired 30-seed effects (95% intervals)")

    # Panel (c): three aligned metric axes with the same family colors.
    names = ["SCO", "Forced", "DE"]
    colors = [PROPOSED, LEARNING, LEARNING]
    markers = ["D", "^", "X"]
    for ax, (title, values, ylim) in zip(trade_axes, trade_metrics):
        xx = np.arange(3)
        ax.plot(xx, values, color=LIGHT, linewidth=1.0, zorder=1)
        for xval, value, color, marker in zip(xx, values, colors, markers):
            ax.scatter(
                xval,
                value,
                color=color,
                edgecolor=color,
                marker=marker,
                s=25,
                linewidth=0.7,
                zorder=3,
            )
        ax.set_xticks([])
        ax.set_xlim(-0.30, 2.30)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=7.2, pad=3.0)
        ax.grid(axis="y", color=LIGHT, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", pad=1.0)
    trade_box = trade_axes[0].get_position()
    fig.text(
        trade_box.x0,
        trade_box.y1 + 0.035,
        "(c) Detector and service trade-off",
        ha="left",
        va="bottom",
        fontsize=8.6,
        fontweight="bold",
        color=DARK,
    )

    trade_handles = [
        Line2D([], [], marker="D", linestyle="none", color=PROPOSED, label="SCO"),
        Line2D([], [], marker="^", linestyle="none", color=LEARNING, label="Forced"),
        Line2D([], [], marker="X", linestyle="none", color=LEARNING, label="DE"),
    ]
    fig.legend(
        handles=trade_handles,
        loc="lower center",
        bbox_to_anchor=(0.790, 0.046),
        ncol=3,
        frameon=False,
        handletextpad=0.35,
        columnspacing=0.85,
        borderaxespad=0.0,
    )
    source_paths = [
        RESULTS / "tmc_v16_baseline_expansion.json",
        RESULTS / "tmc_ts_baseline_expansion.json",
        RESULTS / "tmc_ts_paired_sco_addendum.json",
        RESULTS / "tmc_external_baseline_addendum_v1.json",
        RESULTS / "tmc_confirmatory_summary.csv",
    ]
    point_counts = {
        "panel_a_method_intervals": len(entries),
        "panel_b_seed_points": sum(len(paired_values[row[0]]) for row in effects),
        "panel_b_mean_intervals": len(effects),
        "panel_c_metric_points": 3 * len(trade_metrics),
    }
    qa = render_qa(fig, source_paths, point_counts)
    if not qa["pass"]:
        raise RuntimeError(json.dumps(qa, indent=2))

    stem = OUT / "fig_controlled_grouped_v21"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches=None, pad_inches=0)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches=None, pad_inches=0)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    (OUT / "fig_controlled_grouped_v21_qa.json").write_text(
        json.dumps(qa, indent=2) + "\n", encoding="utf-8"
    )
    return qa


if __name__ == "__main__":
    setup_style()
    result = make_figure()
    print(
        "Wrote Fig. 4 v21: "
        f"pass={result['pass']}, min_font={result['minimum_font_pt']:.1f} pt, "
        f"overflow={result['text_overflow_count']}, overlap={result['text_overlap_count']}"
    )

