"""Render a compact two-panel external-replay evidence figure for TMC v19.

Each dataset panel contains:

* an upper forest plot of excess cost over the full-information trace oracle;
* a lower strip of all 30 matched DTS-minus-SCO episode effects, together
  with the unchanged paired episode-bootstrap mean and 95% interval.

The script deliberately omits the 6GL assignment panels.  It reads the same
frozen replay summaries and retrospective v16 addendum used by
``make_tmc_v16_figures.py::make_figure5`` and never rewrites source results.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results" / "frozen"
OUT = HERE / "generated"

WIDTH_IN = 7.2
HEIGHT_IN = 4.35
MIN_FONT_PT = 7.0

BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"
PURPLE = "#7B61A8"
SKY = "#56B4E9"
VERMILLION = "#D55E00"
DARK = "#222222"
GRAY = "#666666"
MID_GRAY = "#9A9A9A"
LIGHT_GRAY = "#D9D9D9"
PALE_GRAY = "#F5F5F5"
WHITE = "#FFFFFF"

METHODS = [
    ("cumulative_ce", "Cumulative CE", GRAY, "o", False),
    ("cumulative_ucb_cv", "Cumulative UCB-CV", DARK, "s", False),
    ("sw_ce_32", "SW-CE (32)", SKY, "o", False),
    ("sw_ucb_cv_64", "SW-Whittle-CV (64)", SKY, "s", False),
    ("dts_whittle_cv", "DTS-Whittle-CV", PURPLE, "P", False),
    ("de_cd_whittle_cv", "DE-CD-Whittle-CV", ORANGE, "^", True),
    ("sco_reset_ce", "SCO-reset-CE", GREEN, "D", False),
    ("sco_reset_ucb", "SCO-reset-UCB", BLUE, "D", False),
    ("forced_reset_ucb", "Forced-reset-UCB", VERMILLION, "v", True),
    ("aoi", "AoI / round robin", MID_GRAY, "x", True),
]

DATASETS = [
    {
        "key": "uzh_fpv",
        "title": "(a) UZH-FPV",
        "primary": "uzh_trace_replay_v1.json",
        "xlim": (5.0, 110.0),
        "xticks": [5, 10, 20, 40, 80],
    },
    {
        "key": "m3ed_falcon",
        "title": "(b) M3ED Falcon",
        "primary": "m3ed_trace_replay_v1.json",
        "xlim": (5.0, 70.0),
        "xticks": [5, 10, 20, 40, 60],
    },
]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.6,
            "axes.titlesize": 9.0,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.25,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": WHITE,
            "figure.facecolor": WHITE,
        }
    )


def read_json(name: str) -> dict[str, Any]:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"Required result artifact is missing: {path}")
    with path.open(encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected object at top level of {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_record(name: str) -> dict[str, Any]:
    path = RESULTS / name
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def finite_triplet(values: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"{label} must be a three-element [mean, low, high] record")
    mean, low, high = (float(value) for value in values)
    if not all(math.isfinite(value) for value in (mean, low, high)):
        raise ValueError(f"{label} contains a non-finite value")
    if not low <= mean <= high:
        raise ValueError(f"{label} has invalid interval order")
    return mean, low, high


def dataset_records(
    spec: dict[str, Any],
    primary: dict[str, Any],
    expansion: dict[str, Any],
) -> dict[str, Any]:
    addendum = expansion["datasets"][spec["key"]]
    addendum_summary = addendum["summary_mean_ci95"]
    forest: list[dict[str, Any]] = []

    for method, label, color, marker, open_marker in METHODS:
        if method in {"dts_whittle_cv", "de_cd_whittle_cv"}:
            source = "retrospective_v16_addendum"
            values = addendum_summary["methods"][method]["excess_pct"]
        else:
            source = "frozen_primary_replay"
            values = primary["summary_mean_ci95"][method]["excess_pct"]
        mean, low, high = finite_triplet(
            values, f"{spec['key']}.{method}.excess_pct"
        )
        forest.append(
            {
                "method": method,
                "label": label,
                "mean": mean,
                "low": low,
                "high": high,
                "source": source,
                "color": color,
                "marker": marker,
                "open_marker": open_marker,
            }
        )

    episode_effects = np.asarray(
        [
            float(row["methods"]["dts_whittle_cv"]["excess_pct"])
            - float(row["references"]["sco_reset_ucb"]["excess_pct"])
            for row in addendum["episodes"]
        ],
        dtype=float,
    )
    if episode_effects.shape != (30,) or not np.all(np.isfinite(episode_effects)):
        raise ValueError(
            f"{spec['key']} must contain exactly 30 finite matched episode effects"
        )

    effect_mean, effect_low, effect_high = finite_triplet(
        addendum_summary["paired_method_minus_sco"]["dts_whittle_cv"][
            "mean_ci95"
        ],
        f"{spec['key']}.paired_method_minus_sco.dts_whittle_cv",
    )
    if not math.isclose(
        float(episode_effects.mean()), effect_mean, rel_tol=0.0, abs_tol=1e-12
    ):
        raise AssertionError(
            f"{spec['key']} paired mean differs from the stored summary"
        )

    return {
        "key": spec["key"],
        "title": spec["title"],
        "xlim": tuple(float(value) for value in spec["xlim"]),
        "xticks": list(spec["xticks"]),
        "forest": forest,
        "episode_effects": episode_effects,
        "paired_summary": {
            "mean": effect_mean,
            "low": effect_low,
            "high": effect_high,
        },
    }


def draw_forest(ax, records: dict[str, Any], show_method_labels: bool) -> None:
    rows = records["forest"]
    y = np.arange(len(rows))[::-1]
    for index, (yy, row) in enumerate(zip(y, rows)):
        if index % 2 == 1:
            ax.axhspan(yy - 0.48, yy + 0.48, color=PALE_GRAY, zorder=0)
        mean, low, high = row["mean"], row["low"], row["high"]
        ax.errorbar(
            mean,
            yy,
            xerr=np.asarray([[mean - low], [high - mean]]),
            fmt=row["marker"],
            color=row["color"],
            markerfacecolor=WHITE if row["open_marker"] else row["color"],
            markeredgecolor=row["color"],
            markeredgewidth=0.9,
            capsize=2.0,
            markersize=4.7,
            linewidth=1.05,
            zorder=3,
        )
        x_right = records["xlim"][1]
        if high > x_right * 0.77:
            x_text = high * 0.97
            horizontal = "right"
        else:
            x_text = high * 1.035
            horizontal = "left"
        ax.text(
            x_text,
            yy,
            f"{mean:.2f}",
            ha=horizontal,
            va="center",
            color=row["color"],
            fontsize=7.0,
        )

    ax.set_xscale("log")
    ax.set_xlim(*records["xlim"])
    ax.set_xticks(records["xticks"], [str(value) for value in records["xticks"]])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_yticks(y)
    if show_method_labels:
        ax.set_yticklabels([row["label"] for row in rows])
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0, pad=3.0)
    ax.set_xlabel("Excess cost over trace oracle (%) — log scale", labelpad=2.0)
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(records["title"], loc="left", fontweight="bold", pad=5.0)


def draw_effect_strip(
    ax,
    records: dict[str, Any],
    shared_xlim: tuple[float, float],
    jitter_seed: int,
) -> None:
    values = records["episode_effects"]
    summary = records["paired_summary"]
    rng = np.random.default_rng(jitter_seed)
    jitter = rng.uniform(-0.13, 0.13, len(values))

    ax.axvline(0.0, color=DARK, linestyle="--", linewidth=0.85, zorder=1)
    ax.scatter(
        values,
        jitter,
        s=14,
        facecolor=WHITE,
        edgecolor=PURPLE,
        linewidth=0.7,
        alpha=0.72,
        zorder=2,
    )
    ax.errorbar(
        summary["mean"],
        0.0,
        xerr=np.asarray(
            [
                [summary["mean"] - summary["low"]],
                [summary["high"] - summary["mean"]],
            ]
        ),
        fmt="P",
        color=PURPLE,
        markerfacecolor=PURPLE,
        capsize=2.8,
        markersize=5.4,
        linewidth=1.35,
        zorder=4,
    )
    ax.set_xlim(*shared_xlim)
    ax.set_ylim(-0.28, 0.34)
    ax.set_yticks([])
    ax.set_xticks(np.arange(math.ceil(shared_xlim[0]), math.floor(shared_xlim[1]) + 1))
    ax.set_xlabel("DTS minus SCO excess cost (percentage points)", labelpad=2.0)
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.text(
        0.01,
        0.93,
        "30 matched episodes",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=GRAY,
        fontsize=7.0,
    )
    ax.text(
        0.99,
        0.93,
        (
            f"mean {summary['mean']:+.3f} "
            f"[{summary['low']:+.3f}, {summary['high']:+.3f}]"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=PURPLE,
        fontsize=7.0,
        fontweight="bold",
    )
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)


def artist_qa(fig) -> dict[str, Any]:
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
            f"Minimum rendered font {min_font:.2f} pt is below {MIN_FONT_PT:.1f} pt"
        )
    if overflow:
        raise AssertionError(f"Text outside fixed canvas: {overflow[:6]}")
    return {
        "minimum_font_pt": float(min_font),
        "visible_text_count": text_count,
        "text_overflow_count": len(overflow),
        "overflow_text": overflow,
    }


def export(fig) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    width, height = fig.get_size_inches()
    if abs(float(width) - WIDTH_IN) > 1e-9:
        raise AssertionError(f"Canvas width must be {WIDTH_IN:.1f} in")
    qa = artist_qa(fig)

    outputs: dict[str, Any] = {}
    for suffix in ("pdf", "svg", "png"):
        path = OUT / f"fig_external_replay_compact.{suffix}"
        kwargs: dict[str, Any] = {"bbox_inches": None, "facecolor": WHITE}
        if suffix == "png":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)
        outputs[suffix] = {
            "path": str(path.relative_to(HERE)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    return {
        "canvas_width_in": float(width),
        "canvas_height_in": float(height),
        **qa,
        "outputs": outputs,
    }


def main() -> None:
    setup_style()
    expansion_name = "tmc_v16_trace_baseline_expansion.json"
    expansion = read_json(expansion_name)
    primary_payloads = {
        spec["key"]: read_json(spec["primary"]) for spec in DATASETS
    }
    records = [
        dataset_records(spec, primary_payloads[spec["key"]], expansion)
        for spec in DATASETS
    ]

    all_effects = np.concatenate([record["episode_effects"] for record in records])
    effect_limit = max(3.0, math.ceil(float(np.max(np.abs(all_effects))) * 2.0) / 2.0)
    shared_effect_xlim = (-effect_limit, effect_limit)

    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.19,
        right=0.985,
        bottom=0.12,
        top=0.93,
        width_ratios=[1.0, 1.0],
        height_ratios=[4.1, 1.15],
        wspace=0.25,
        hspace=0.23,
    )
    forest_axes = [fig.add_subplot(grid[0, index]) for index in range(2)]
    effect_axes = [fig.add_subplot(grid[1, index]) for index in range(2)]

    for index, record in enumerate(records):
        draw_forest(forest_axes[index], record, show_method_labels=index == 0)
        draw_effect_strip(
            effect_axes[index],
            record,
            shared_effect_xlim,
            jitter_seed=20260728 + index,
        )

    render = export(fig)
    plt.close(fig)

    plotted_forest_points = sum(len(record["forest"]) for record in records)
    plotted_episode_points = sum(len(record["episode_effects"]) for record in records)
    if plotted_forest_points != 20 or plotted_episode_points != 60:
        raise AssertionError("Unexpected plotted point count")

    qa_payload = {
        "schema_version": 1,
        "figure": "fig_external_replay_compact",
        "renderer": Path(__file__).name,
        "design_contract": {
            "target": "IEEE TMC two-column figure",
            "core_claim": (
                "External replay replication with dataset-specific uncertainty "
                "and an explicit matched DTS-minus-SCO boundary"
            ),
            "panel_map": {
                "a": "UZH-FPV forest plus 30-episode paired effect strip",
                "b": "M3ED Falcon forest plus 30-episode paired effect strip",
            },
            "removed_from_main_figure": "6GL assignment sensitivity",
            "width_in": WIDTH_IN,
            "minimum_font_pt": MIN_FONT_PT,
            "no_fabrication": True,
        },
        "sources": [
            source_record(spec["primary"]) for spec in DATASETS
        ]
        + [source_record(expansion_name)],
        "data_contract": {
            "method_order": [method for method, *_ in METHODS],
            "forest_point_count": plotted_forest_points,
            "paired_episode_point_count": plotted_episode_points,
            "matched_episodes_per_dataset": 30,
            "primary_methods_use_frozen_replay_summary": [
                method
                for method, *_ in METHODS
                if method not in {"dts_whittle_cv", "de_cd_whittle_cv"}
            ],
            "retrospective_methods_use_v16_addendum": [
                "dts_whittle_cv",
                "de_cd_whittle_cv",
            ],
            "paired_ci_source": (
                "tmc_v16_trace_baseline_expansion.json:"
                "summary_mean_ci95.paired_method_minus_sco."
                "dts_whittle_cv.mean_ci95"
            ),
            "data_and_ci_unchanged": True,
        },
        "datasets": {},
        "render_qa": render,
        "checks": {
            "fixed_7_2_in_width": render["canvas_width_in"] == WIDTH_IN,
            "minimum_font_at_least_7pt": (
                render["minimum_font_pt"] >= MIN_FONT_PT
            ),
            "zero_text_overflow": render["text_overflow_count"] == 0,
            "thirty_matched_episodes_each": all(
                len(record["episode_effects"]) == 30 for record in records
            ),
            "uzh_interval_crosses_zero": (
                records[0]["paired_summary"]["low"] <= 0.0
                <= records[0]["paired_summary"]["high"]
            ),
            "m3ed_interval_is_positive": records[1]["paired_summary"]["low"] > 0.0,
            "pdf_svg_png_exported": set(render["outputs"]) == {"pdf", "svg", "png"},
        },
    }
    for record in records:
        qa_payload["datasets"][record["key"]] = {
            "forest": [
                {
                    key: row[key]
                    for key in ("method", "mean", "low", "high", "source")
                }
                for row in record["forest"]
            ],
            "paired_episode_effects": [
                float(value) for value in record["episode_effects"]
            ],
            "paired_summary_mean_ci95": record["paired_summary"],
        }

    if not all(qa_payload["checks"].values()):
        failed = [
            key for key, value in qa_payload["checks"].items() if not value
        ]
        raise AssertionError(f"Figure QA failed: {failed}")

    qa_path = OUT / "fig_external_replay_compact_qa.json"
    with qa_path.open("w", encoding="utf-8") as handle:
        json.dump(qa_payload, handle, indent=2)
        handle.write("\n")

    print(
        json.dumps(
            {
                "figure": str(
                    (OUT / "fig_external_replay_compact.pdf").relative_to(HERE)
                ).replace("\\", "/"),
                "qa": str(qa_path.relative_to(HERE)).replace("\\", "/"),
                "checks": qa_payload["checks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
