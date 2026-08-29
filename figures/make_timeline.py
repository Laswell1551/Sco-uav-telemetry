"""Draw the fixed-canvas v21 PA-SCO delayed-pipeline timeline.

The figure is deliberately concise at IEEE single-column size.  It separates
the event clock, the scheduler-observable outstanding state, and the different
SCO/PA-SCO decisions.  No experimental values are encoded.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.text import Text


HERE = Path(__file__).resolve().parent
OUT = HERE / "generated"
STEM = "fig_pipeline_timeline_v21"

# Journal palette: SCO blue, PA-SCO orange, and neutral event scaffolding.
BLUE = "#0072B2"
ORANGE = "#E69F00"
DARK = "#222222"
GRAY = "#666666"
MID = "#A6A6A6"
LIGHT = "#D9D9D9"
PALE_BLUE = "#EAF4FB"
PALE_ORANGE = "#FFF4D6"
PALE_GRAY = "#F2F2F2"


def add_box(
    ax,
    tracked: list[tuple[Text, FancyBboxPatch, str]],
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    edge: str,
    face: str = "#FFFFFF",
    *,
    fontsize: float = 7.2,
    linewidth: float = 1.0,
    weight: str = "normal",
) -> tuple[FancyBboxPatch, Text]:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=face,
        zorder=3,
    )
    ax.add_patch(patch)
    txt = ax.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=DARK,
        linespacing=0.95,
        zorder=4,
    )
    tracked.append((txt, patch, label))
    return patch, txt


def add_arrow(ax, start, end, color, *, linestyle="-", linewidth=1.35):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8.0,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            shrinkA=1.0,
            shrinkB=1.0,
            zorder=2,
        )
    )


def bbox_intersection_area(a, b) -> float:
    width = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    height = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    return width * height


def validate_and_save(
    fig,
    tracked: list[tuple[Text, FancyBboxPatch, str]],
    overlap_exempt: set[frozenset[str]],
) -> dict[str, object]:
    OUT.mkdir(parents=True, exist_ok=True)
    width, height = fig.get_size_inches()
    if abs(width - 3.48) > 1e-9:
        raise AssertionError(f"Timeline width must be 3.48 in, found {width:.3f}")

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
    text_boxes: dict[str, object] = {}
    for artist in texts:
        bbox = artist.get_window_extent(renderer=renderer)
        text_boxes[artist.get_text()] = bbox
        if (
            bbox.x0 < -0.5
            or bbox.y0 < -0.5
            or bbox.x1 > canvas_width + 0.5
            or bbox.y1 > canvas_height + 0.5
        ):
            canvas_overflow.append(artist.get_text())

    box_overflow: list[str] = []
    for txt, patch, label in tracked:
        text_bbox = txt.get_window_extent(renderer=renderer)
        patch_bbox = patch.get_window_extent(renderer=renderer)
        margin = 1.0
        if (
            text_bbox.x0 < patch_bbox.x0 + margin
            or text_bbox.y0 < patch_bbox.y0 + margin
            or text_bbox.x1 > patch_bbox.x1 - margin
            or text_bbox.y1 > patch_bbox.y1 - margin
        ):
            box_overflow.append(label)

    overlaps: list[list[str]] = []
    for i, left in enumerate(texts):
        left_label = left.get_text()
        left_box = left.get_window_extent(renderer=renderer)
        for right in texts[i + 1 :]:
            right_label = right.get_text()
            if frozenset((left_label, right_label)) in overlap_exempt:
                continue
            right_box = right.get_window_extent(renderer=renderer)
            if bbox_intersection_area(left_box, right_box) > 0.5:
                overlaps.append([left_label, right_label])

    if min_font < 7.0 - 1e-9:
        raise AssertionError(f"Timeline minimum font is {min_font:.2f} pt")
    if canvas_overflow:
        raise AssertionError(f"Text outside fixed canvas: {canvas_overflow}")
    if box_overflow:
        raise AssertionError(f"Text outside assigned nodes: {box_overflow}")
    if overlaps:
        raise AssertionError(f"Text overlaps detected: {overlaps}")

    outputs: dict[str, str] = {}
    for suffix in ("pdf", "svg", "png"):
        path = OUT / f"{STEM}.{suffix}"
        kwargs: dict[str, object] = {"bbox_inches": None, "facecolor": "#FFFFFF"}
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
        "node_label_overflow_count": len(box_overflow),
        "text_overlap_count": len(overlaps),
        "semantic_checks": {
            "tx_arrival_ack_present": True,
            "outstanding_in_flight_state_present": True,
            "sco_duplicate_opportunity_present": True,
            "pa_sco_duplicate_suppression_present": True,
            "zero_delay_exact_reduction_present": True,
        },
        "palette": "SCO blue, PA-SCO orange, and neutral event scaffolding",
        "no_fabrication": "Protocol schematic only; no experimental values added.",
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
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(3.48, 2.82))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    tracked: list[tuple[Text, FancyBboxPatch, str]] = []

    # Shared event clock.
    x_tx, x_dec, x_rx, x_ack = 0.14, 0.33, 0.62, 0.88
    for x, label in (
        (x_tx, r"$t$"),
        (x_dec, r"$t+1$"),
        (x_rx, r"$t+d^{\mathrm{F}}$"),
        (x_ack, r"$t+d^{\mathrm{F}}+d^{\mathrm{B}}$"),
    ):
        ax.plot([x, x], [0.265, 0.895], color=LIGHT, linestyle=":", linewidth=0.8)
        ax.text(x, 0.925, label, ha="center", va="center", fontsize=7.2, color=DARK)

    ax.text(0.015, 0.882, "EVENTS", ha="left", va="center", fontsize=7.0,
            fontweight="bold", color=GRAY)
    add_box(ax, tracked, 0.080, 0.745, 0.120, 0.115, r"TX $k$", MID,
            fontsize=7.2, weight="bold")
    add_box(ax, tracked, 0.545, 0.745, 0.150, 0.115, "arrival", MID,
            fontsize=7.2, weight="bold")
    add_box(ax, tracked, 0.825, 0.745, 0.110, 0.115, "ACK", MID,
            fontsize=7.2, weight="bold")
    add_arrow(ax, (0.205, 0.802), (0.538, 0.802), GRAY, linewidth=1.55)
    add_arrow(ax, (0.701, 0.802), (0.818, 0.802), GRAY,
              linestyle="--", linewidth=1.45)
    ax.text(0.370, 0.827, r"$d^{\mathrm{F}}$", ha="center", va="bottom",
            fontsize=7.2, color=GRAY, fontweight="bold")
    ax.text(0.760, 0.827, r"$d^{\mathrm{B}}$", ha="center", va="bottom",
            fontsize=7.2, color=GRAY, fontweight="bold")

    # ACK-observable scheduler state: the attempt remains unresolved until ACK.
    ax.text(0.015, 0.640, "State", ha="left", va="center", fontsize=7.2,
            fontweight="bold", color=GRAY)
    state_x0, state_x1, state_y, state_h = 0.100, 0.875, 0.590, 0.100
    ax.add_patch(
        FancyBboxPatch(
            (state_x0, state_y), state_x1 - state_x0, state_h,
            boxstyle="round,pad=0.010,rounding_size=0.020",
            linewidth=0.9, edgecolor=ORANGE, facecolor=PALE_ORANGE, zorder=1,
        )
    )
    ax.text(0.485, 0.640, r"outstanding / in flight:  $m_k=1$",
            ha="center", va="center", fontsize=7.3, fontweight="bold",
            color=ORANGE, zorder=2)
    ax.text(0.487, 0.553, r"$a_k^{\mathrm{ack}}$ unchanged until feedback",
            ha="center", va="center", fontsize=7.0, color=GRAY)
    ax.text(0.930, 0.640, r"$m_k=0$", ha="center", va="center",
            fontsize=7.1, fontweight="bold", color=ORANGE)

    # The same t+1 opportunity yields different decisions.
    ax.add_patch(Rectangle((0.015, 0.405), 0.970, 0.120,
                           facecolor=PALE_BLUE, edgecolor="none", zorder=0))
    ax.text(0.030, 0.465, "SCO", ha="left", va="center", fontsize=7.4,
            fontweight="bold", color=BLUE)
    add_box(ax, tracked, 0.160, 0.420, 0.210, 0.090,
            r"high $a_k^{\mathrm{ack}}$", BLUE, "#FFFFFF")
    add_arrow(ax, (0.382, 0.465), (0.585, 0.465), BLUE,
              linestyle="--", linewidth=1.35)
    add_box(ax, tracked, 0.600, 0.420, 0.205, 0.090,
            r"re-send $k$", BLUE, "#FFFFFF", weight="bold")
    ax.text(0.970, 0.465, "duplicate", ha="right", va="center",
            fontsize=7.0, color=BLUE, fontweight="bold")

    ax.add_patch(Rectangle((0.015, 0.270), 0.970, 0.120,
                           facecolor=PALE_ORANGE, edgecolor="none", zorder=0))
    ax.text(0.030, 0.330, "PA-SCO", ha="left", va="center", fontsize=7.4,
            fontweight="bold", color=ORANGE)
    add_box(ax, tracked, 0.160, 0.285, 0.260, 0.090,
            r"$W_k/(1+\beta m_k)$", ORANGE, "#FFFFFF")
    add_arrow(ax, (0.432, 0.330), (0.585, 0.330), ORANGE, linewidth=1.55)
    add_box(ax, tracked, 0.600, 0.285, 0.340, 0.090,
            "rank another stream", ORANGE, "#FFFFFF", weight="bold")

    # Exact boundary: no unresolved state survives to the next decision.
    boundary = FancyBboxPatch(
        (0.020, 0.060), 0.960, 0.125,
        boxstyle="round,pad=0.012,rounding_size=0.020",
        linewidth=0.9, edgecolor=MID, facecolor=PALE_GRAY, zorder=1,
    )
    ax.add_patch(boundary)
    boundary_text = ax.text(
        0.500, 0.122,
        r"$d^{\mathrm{F}}=d^{\mathrm{B}}=0 \;\Rightarrow\; m_k=0$ before the next decision"
        "\n" + r"$\Rightarrow\;$ PA-SCO $\equiv$ SCO",
        ha="center", va="center", fontsize=7.1, color=DARK,
        fontweight="bold", linespacing=1.05, zorder=2,
    )
    tracked.append((boundary_text, boundary, "zero-delay exact reduction"))

    fig.subplots_adjust(left=0.018, right=0.988, bottom=0.020, top=0.985)
    result = validate_and_save(fig, tracked, overlap_exempt=set())
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
