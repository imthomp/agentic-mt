"""Generate the two figures for advisor_update_phases_1-6.tex from
reports/data/phase_quantities.json -- same SSOT as the macros, so the
figures can never drift from the numbers cited in prose.

Palette: dataviz skill's validated reference palette (references/palette.md).
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

DATA_PATH = Path(__file__).parent / "data" / "phase_quantities.json"
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Reference palette (light mode; this report is print/PDF, no dark-mode need)
BLUE = "#2a78d6"       # categorical slot 1 / sequential hue
AQUA = "#1baf7a"       # categorical slot 2
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": SECONDARY_INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRIDLINE,
    "grid.linewidth": 0.6,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def fig_exposure_floor(data):
    p6 = data["phase6"]
    points = [
        ("en-fr", p6["enfr_roots_bytes"], p6["enfr_clean_rate_pct"]),
        ("en-sw", p6["ensw_roots_bytes"], p6["ensw_clean_rate_pct"]),
        ("en-yo", p6["enyo_roots_bytes"], p6["enyo_clean_rate_pct"]),
        ("en-xh", p6["enxh_roots_bytes"], p6["enxh_clean_rate_pct"]),
        ("en-zu", p6["enzu_roots_bytes"], p6["enzu_clean_rate_pct"]),
    ]
    points.sort(key=lambda t: t[1])
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    labels = [p[0] for p in points]

    fig, ax = plt.subplots(figsize=(4.6, 3.0), dpi=200)
    ax.plot(xs, ys, color=BLUE, linewidth=2, zorder=2)
    ax.scatter(xs, ys, color=BLUE, s=42, zorder=3, edgecolors=SURFACE, linewidths=1.2)

    # en-xh/en-zu sit close together on the log axis at y=0 -- stack their
    # labels above/below to avoid collision instead of a uniform offset.
    label_offsets = {
        "en-fr": (0, -14), "en-sw": (0, 12),
        "en-yo": (0, 18), "en-xh": (0, -16), "en-zu": (-4, 8),
    }
    for lbl, x, y in points:
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=label_offsets[lbl],
                    ha="center", fontsize=8.5, color=SECONDARY_INK)

    ax.set_xscale("log")
    ax.set_xlabel("BLOOM ROOTS pretraining bytes (log scale)", fontsize=9)
    ax.set_ylabel("Clean-generation rate (%)", fontsize=9)
    ax.set_ylim(-8, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(AXIS)
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase6_exposure_floor.pdf")
    plt.close(fig)


def fig_phase3_conditions(data):
    p3 = data["phase3"]
    conditions = ["baseline", "cot", "tool", "cottool"]
    labels = ["Baseline", "CoT", "Tool", "CoT+Tool"]
    ende = [p3[f"ende_{c}_cometkiwi"] for c in conditions]
    enha = [p3[f"enha_{c}_cometkiwi"] for c in conditions]

    x = range(len(conditions))
    width = 0.36

    fig, ax = plt.subplots(figsize=(4.6, 3.0), dpi=200)
    bars1 = ax.bar([i - width/2 for i in x], ende, width, label="en-de (Aya-101)", color=BLUE)
    bars2 = ax.bar([i + width/2 for i in x], enha, width, label="en-ha (Aya-101)", color=AQUA)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("COMET-KIWI", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.tick_params(labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(AXIS)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.02, 1.15))

    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase3_conditions.pdf")
    plt.close(fig)


def main():
    data = json.loads(DATA_PATH.read_text())
    fig_exposure_floor(data)
    fig_phase3_conditions(data)
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
