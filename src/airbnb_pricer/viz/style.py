"""Shared visual style for every figure the project produces.

Color usage follows a small set of rules so all charts read as one system:

- Nominal bar charts (model names, feature names) use ONE hue — bar length
  already encodes the value, so color stays on identity duty.
- The three price tiers are *ordered* (Budget < Mid-Market < Premium), so they
  take an ordinal single-hue ramp: light -> dark blue. Validated for lightness
  monotonicity, step gaps, and surface contrast.
- Magnitude grids (confusion matrix) use the same blue as a sequential ramp.
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

# --- palette ---------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

SERIES_1 = "#2a78d6"  # primary categorical hue (blue)

# Ordinal ramp for the ordered price tiers (light -> dark = cheap -> expensive).
TIER_COLORS = {
    "Budget": "#86b6ef",
    "Mid-Market": "#2a78d6",
    "Premium": "#104281",
}

# Sequential blue ramp (steps 100 -> 700) for magnitude grids.
SEQUENTIAL_BLUES = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

BLUE_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUES)

RCPARAMS = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.color": INK,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
}


def apply_style() -> None:
    import matplotlib

    matplotlib.rcParams.update(RCPARAMS)
