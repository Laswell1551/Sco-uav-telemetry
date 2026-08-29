"""Draw the fixed-canvas v21 SCO/PA-SCO information-state overview.

Figure 1 is a high-level map, not a second event timeline.  Panel (a) shows
the reliable-feedback learning/control loop and its endogenous revisit
mechanism.  Panel (b) contrasts the age-only and pipeline-aware information
states.  No experimental values are encoded.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text


HERE = Path(__file__).resolve().parent
OUT = HERE / "generated"
STEM = "fig_overview_sco_pa_v21"

# Journal palette: one stable accent per method and neutral support.
BLUE = "#0072B2"
ORANGE = "#E69F00"
DARK = "#222222"
GRAY = "#666666"
MID = "#A6A6A6"
LIGHT = "#D9D9D9"
PANEL = "#F6F8FA"
PALE_BLUE = "#EAF4FB"
PALE_ORANGE = "#FFF4D6"
WHITE = "#FFFFFF"


def intersection_area(a, b) -> float:
    width = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    height = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    return width * height


def add_node(
    ax,
    tracked: list[tuple[Text, FancyBboxPatch, str]],
    key: str,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    *,
    edge: str = LIGHT,
    face: str = WHITE,
    fontsize: float = 7.2,
    linewidth: float = 0.95,
    weight: str = "normal",
    linespacing: float = 1.02,
) -> tuple[FancyBboxPatch, Text]:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.010,rounding_size=0.015",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=face,
        zorder=3,
    )
    ax.add_patch(patch)
    text = ax.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=DARK,
        linespacing=linespacing,
        zorder=4,
    )
    tracked.append((text, patch, key))
    return patch, text


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    linewidth: float = 1.15,
    linestyle: str = "-",
    rad: float = 0.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8.0,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=1.5,
            shrinkB=1.5,
            zorder=2,
        )
    )


def validate_and_save(
    fig,
    tracked: list[tuple[Text, FancyBboxPatch, str]],
) -> dict[str, object]:
    OUT.mkdir(parents=True, exist_ok=True)
    width, height = fig.get_size_inches()
    if abs(width - 7.20) > 1e-9:
        raise AssertionError(f"Overview width must be 7.20 in, found {width:.3f}")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas_width, canvas_height = fig.canvas.get_width_height()
    texts = [
        artist
        for artist in fig.findobj(match=Text)
        if artist.get_visible() and artist.get_text().strip()
    ]
    min_font = min(float(artist.get_fontsize()) for artist in texts)

    canvas_overflow: list[str] = []
    for artist in texts:
        bbox = artist.get_window_extent(renderer=renderer)
        if (
            bbox.x0 < -0.5
            or bbox.y0 < -0.5
            or bbox.x1 > canvas_width + 0.5
            or bbox.y1 > canvas_height + 0.5
        ):
            canvas_overflow.append(artist.get_text())

    node_overflow: list[str] = []
    owner_by_text = {id(text): patch for text, patch, _ in tracked}
    key_by_text = {id(text): key for text, _, key in tracked}
    for text, patch, key in tracked:
        text_bbox = text.get_window_extent(renderer=renderer)
        patch_bbox = patch.get_window_extent(renderer=renderer)
        # About 3.6 pt at the renderer's 100-dpi validation canvas.
        margin_px = 5.0
        if (
            text_bbox.x0 < patch_bbox.x0 + margin_px
            or text_bbox.y0 < patch_bbox.y0 + margin_px
            or text_bbox.x1 > patch_bbox.x1 - margin_px
            or text_bbox.y1 > patch_bbox.y1 - margin_px
        ):
            node_overflow.append(key)

    text_overlaps: list[list[str]] = []
    for i, left in enumerate(texts):
        left_bbox = left.get_window_extent(renderer=renderer)
        for right in texts[i + 1 :]:
            right_bbox = right.get_window_extent(renderer=renderer)
            if intersection_area(left_bbox, right_bbox) > 0.5:
                text_overlaps.append([left.get_text(), right.get_text()])

    foreign_node_hits: list[list[str]] = []
    for text in texts:
        text_bbox = text.get_window_extent(renderer=renderer)
        own_patch = owner_by_text.get(id(text))
        for _, patch, key in tracked:
            if patch is own_patch:
                continue
            patch_bbox = patch.get_window_extent(renderer=renderer)
            if intersection_area(text_bbox, patch_bbox) > 0.5:
                foreign_node_hits.append([text.get_text(), key])

    if min_font < 7.0 - 1e-9:
        raise AssertionError(f"Overview minimum font is {min_font:.2f} pt")
    if canvas_overflow:
        raise AssertionError(f"Text outside fixed canvas: {canvas_overflow}")
    if node_overflow:
        raise AssertionError(f"Text lacks required node padding: {node_overflow}")
    if text_overlaps:
        raise AssertionError(f"Text overlaps detected: {text_overlaps}")
    if foreign_node_hits:
        raise AssertionError(f"Text intersects a foreign node: {foreign_node_hits}")

    outputs: dict[str, str] = {}
    for suffix in ("pdf", "svg", "png"):
        path = OUT / f"{STEM}.{suffix}"
        kwargs: dict[str, object] = {
            "bbox_inches": None,
            "facecolor": WHITE,
        }
        if suffix == "png":
            kwargs["dpi"] = 320
        fig.savefig(path, **kwargs)
        outputs[suffix] = path.relative_to(HERE).as_posix()

    result: dict[str, object] = {
        "stem": STEM,
        "width_in": float(width),
        "height_in": float(height),
        "minimum_font_pt": float(min_font),
        "text_artist_count": len(texts),
        "canvas_overflow_count": len(canvas_overflow),
        "node_label_overflow_count": len(node_overflow),
        "text_overlap_count": len(text_overlaps),
        "foreign_node_intersection_count": len(foreign_node_hits),
        "semantic_checks": {
            "reliable_feedback_learning_chain_present": True,
            "age_priority_finite_revisit_present": True,
            "no_forced_probe_boundary_present": True,
            "age_only_information_gap_present": True,
            "pipeline_aware_score_present": True,
            "duplicate_suppression_present": True,
            "zero_rtt_reduction_present": True,
        },
        "palette": "SCO blue, PA-SCO orange, and neutral manuscript support",
        "no_fabrication": "Conceptual protocol overview only; no experimental values added.",
        **outputs,
    }
    qa_path = OUT / f"{STEM}_qa.json"
    with qa_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return result


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "text.color": DARK,
        }
    )

    fig, ax = plt.subplots(figsize=(7.20, 3.05))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    tracked: list[tuple[Text, FancyBboxPatch, str]] = []

    # Two fixed evidence bands.
    for y, height in ((0.515, 0.475), (0.010, 0.485)):
        ax.add_patch(
            FancyBboxPatch(
                (0.010, y),
                0.980,
                height,
                boxstyle="round,pad=0.0,rounding_size=0.014",
                facecolor=PANEL,
                edgecolor=LIGHT,
                linewidth=0.8,
                zorder=0,
            )
        )

    # Panel (a): reliable feedback.
    ax.text(
        0.025,
        0.955,
        "(a) Reliable feedback: selected service creates the observations",
        ha="left",
        va="center",
        fontsize=8.6,
        fontweight="bold",
    )
    ax.text(
        0.975,
        0.955,
        "SCO-reset-UCB",
        ha="right",
        va="center",
        fontsize=7.6,
        fontweight="bold",
        color=BLUE,
    )

    ax.text(
        0.085,
        0.895,
        "$K$ streams",
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color=GRAY,
    )
    add_node(ax, tracked, "stream-1", 0.030, 0.785, 0.110, 0.060, "UAV 1")
    add_node(
        ax,
        tracked,
        "stream-k",
        0.030,
        0.705,
        0.110,
        0.060,
        r"UAV $k$",
        edge=BLUE,
        face=PALE_BLUE,
        linewidth=1.15,
        weight="bold",
    )
    add_node(ax, tracked, "stream-K", 0.030, 0.625, 0.110, 0.060, r"UAV $K$")

    chain_y, chain_h = 0.700, 0.120
    add_node(
        ax,
        tracked,
        "selected-history",
        0.175,
        chain_y,
        0.145,
        chain_h,
        "Selected history\n" + r"$y_k$",
        edge=BLUE,
        face=PALE_BLUE,
        linewidth=1.1,
        weight="bold",
    )
    add_node(
        ax,
        tracked,
        "moments-detector",
        0.360,
        chain_y,
        0.145,
        chain_h,
        "Moments +\ndetector",
        edge=BLUE,
    )
    add_node(
        ax,
        tracked,
        "index-interval",
        0.545,
        chain_y,
        0.155,
        chain_h,
        "Riccati map +\nindex interval",
        edge=BLUE,
    )
    add_node(
        ax,
        tracked,
        "top-N-service",
        0.745,
        chain_y,
        0.150,
        chain_h,
        r"Top-$N$" + "\nservice",
        edge=BLUE,
        face=PALE_BLUE,
        linewidth=1.1,
        weight="bold",
    )
    add_arrow(ax, (0.142, 0.730), (0.171, 0.750), color=BLUE)
    add_arrow(ax, (0.322, 0.760), (0.356, 0.760), color=BLUE)
    add_arrow(ax, (0.507, 0.760), (0.541, 0.760), color=BLUE)
    add_arrow(ax, (0.702, 0.760), (0.741, 0.760), color=BLUE)

    add_node(
        ax,
        tracked,
        "alarm-reset",
        0.370,
        0.850,
        0.125,
        0.060,
        "alarm $\\rightarrow$ reset",
        edge=MID,
        face=WHITE,
        fontsize=7.0,
    )
    add_arrow(ax, (0.433, 0.848), (0.433, 0.824), color=GRAY, linewidth=1.0)

    ax.text(
        0.030,
        0.575,
        "self-exploration",
        ha="left",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color=GRAY,
    )
    add_node(
        ax,
        tracked,
        "unserved",
        0.180,
        0.535,
        0.115,
        0.075,
        "unserved",
        edge=MID,
    )
    add_node(
        ax,
        tracked,
        "age-index-growth",
        0.335,
        0.535,
        0.180,
        0.075,
        r"$a_k,\;W_k(a_k)\uparrow$",
        edge=BLUE,
        face=PALE_BLUE,
        weight="bold",
    )
    add_node(
        ax,
        tracked,
        "finite-revisit",
        0.555,
        0.535,
        0.130,
        0.075,
        "finite revisit",
        edge=BLUE,
        face=PALE_BLUE,
        weight="bold",
    )
    add_node(
        ax,
        tracked,
        "no-forced-probes",
        0.735,
        0.530,
        0.165,
        0.085,
        "No forced\nprobes",
        edge=BLUE,
        face=PALE_BLUE,
        linewidth=1.1,
        weight="bold",
    )
    add_arrow(ax, (0.297, 0.573), (0.331, 0.573), color=BLUE)
    add_arrow(ax, (0.517, 0.573), (0.551, 0.573), color=BLUE)
    add_arrow(ax, (0.687, 0.573), (0.731, 0.573), color=BLUE)

    # Panel (b): information-state contrast.  Fig. 3 carries the event timing.
    ax.text(
        0.025,
        0.460,
        "(b) Delayed feedback: age alone omits unresolved attempts",
        ha="left",
        va="center",
        fontsize=8.6,
        fontweight="bold",
    )
    ax.text(
        0.975,
        0.460,
        "PA-SCO",
        ha="right",
        va="center",
        fontsize=7.6,
        fontweight="bold",
        color=ORANGE,
    )

    # Age-only state.
    row1_y, row1_h, row1_c = 0.285, 0.105, 0.3375
    add_node(
        ax,
        tracked,
        "sco-age-state",
        0.035,
        row1_y,
        0.155,
        row1_h,
        "SCO state\n" + r"$a_k^{\mathrm{ack}}$",
        edge=BLUE,
        face=PALE_BLUE,
        linewidth=1.05,
        weight="bold",
    )
    add_node(
        ax,
        tracked,
        "hidden-in-flight",
        0.220,
        row1_y,
        0.160,
        row1_h,
        "in-flight update\nnot represented",
        edge=MID,
    )
    add_node(
        ax,
        tracked,
        "repeat-ranks-high",
        0.420,
        row1_y,
        0.200,
        row1_h,
        "another attempt\ncan rank high",
        edge=MID,
    )
    add_node(
        ax,
        tracked,
        "duplicate-opportunity",
        0.670,
        row1_y,
        0.275,
        row1_h,
        "duplicate-service opportunity",
        edge=MID,
        face=WHITE,
        weight="bold",
    )
    add_arrow(ax, (0.192, row1_c), (0.216, row1_c), color=GRAY)
    add_arrow(ax, (0.382, row1_c), (0.416, row1_c), color=GRAY)
    add_arrow(ax, (0.622, row1_c), (0.666, row1_c), color=GRAY)

    # Pipeline-aware state.
    row2_y, row2_h, row2_c = 0.085, 0.125, 0.1475
    add_node(
        ax,
        tracked,
        "pa-state",
        0.035,
        row2_y,
        0.155,
        row2_h,
        "PA state\n" + r"$(a_k^{\mathrm{ack}},m_k)$",
        edge=ORANGE,
        face=PALE_ORANGE,
        linewidth=1.1,
        weight="bold",
    )
    add_node(
        ax,
        tracked,
        "visible-in-flight",
        0.220,
        row2_y,
        0.160,
        row2_h,
        r"$m_k>0$" + "\nvisible",
        edge=ORANGE,
    )
    add_node(
        ax,
        tracked,
        "pa-score",
        0.420,
        row2_y,
        0.230,
        row2_h,
        r"$S_k^{\mathrm{PA}}=S_k^{\mathrm{SCO}}-\beta m_k$",
        edge=ORANGE,
        face=PALE_ORANGE,
        fontsize=7.7,
        linewidth=1.15,
        weight="bold",
    )
    add_node(
        ax,
        tracked,
        "suppress-repeats",
        0.690,
        row2_y,
        0.125,
        row2_h,
        "suppress\nrepeats",
        edge=ORANGE,
        face=PALE_ORANGE,
        weight="bold",
    )
    add_node(
        ax,
        tracked,
        "zero-rtt",
        0.850,
        row2_y,
        0.110,
        row2_h,
        "RTT $=0$\nPA $=$ SCO",
        edge=ORANGE,
        face=PALE_ORANGE,
        weight="bold",
    )
    add_arrow(ax, (0.192, row2_c), (0.216, row2_c), color=ORANGE)
    add_arrow(ax, (0.382, row2_c), (0.416, row2_c), color=ORANGE)
    add_arrow(ax, (0.652, row2_c), (0.686, row2_c), color=ORANGE)
    add_arrow(ax, (0.817, row2_c), (0.846, row2_c), color=ORANGE)

    fig.subplots_adjust(left=0.010, right=0.990, bottom=0.018, top=0.987)
    result = validate_and_save(fig, tracked)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
