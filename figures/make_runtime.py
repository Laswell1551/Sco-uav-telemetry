from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "results" / "frozen" / "tmc_runtime_scaling.csv"
OUT = HERE / "generated"
BLUE = "#0072B2"
GRAY = "#666666"


def main():
    with SOURCE.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    for method, label, color, marker in [
        ("cumulative_ucb_cv", "Cumulative UCB", GRAY, "o"),
        ("sco_reset_ucb", "SCO-reset-UCB", BLUE, "D"),
    ]:
        selected = [r for r in rows if r["method"] == method]
        x = np.array([int(r["K"]) for r in selected])
        median = np.array([float(r["median_ms_per_slot"]) for r in selected])
        q1 = np.array([float(r["q1_ms_per_slot"]) for r in selected])
        q3 = np.array([float(r["q3_ms_per_slot"]) for r in selected])
        ax.plot(x, median, color=color, marker=marker, linewidth=1.7, label=label)
        ax.fill_between(x, q1, q3, color=color, alpha=0.16, linewidth=0)
    ax.set_xlabel("Number of UAV streams, K (N = K/5)")
    ax.set_ylabel("Online time per slot (ms) ↓")
    ax.set_xticks([20, 40, 80, 160, 320])
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    ax.legend(frameon=False, loc="upper left")
    ax.annotate(
        "10.59 ms/slot",
        xy=(320, 10.5865),
        xytext=(195, 8.8),
        color=BLUE,
        fontsize=7,
        arrowprops=dict(arrowstyle="-", color=BLUE),
    )
    fig.tight_layout(pad=0.35)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_runtime_scaling.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(
        OUT / "fig_runtime_scaling.png",
        dpi=240,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
    print(OUT / "fig_runtime_scaling.pdf")


if __name__ == "__main__":
    main()
