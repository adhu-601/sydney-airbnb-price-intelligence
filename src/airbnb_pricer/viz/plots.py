"""Report figures: EDA, model comparison, confusion matrix, importances."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from airbnb_pricer import TIER_ORDER
from airbnb_pricer.viz.style import (
    BLUE_CMAP,
    INK,
    INK_SECONDARY,
    MUTED,
    SERIES_1,
    SURFACE,
    TIER_COLORS,
    apply_style,
)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_price_distribution(df: pd.DataFrame, path: Path, budget_max: float = 200,
                            mid_market_max: float = 400) -> Path:
    """Histogram of nightly price with tier boundaries marked."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    prices = df["price_numeric"].clip(upper=1500)
    ax.hist(prices, bins=60, color=SERIES_1, edgecolor=SURFACE, linewidth=0.4)
    boundaries = [(budget_max, "Budget | Mid-Market"), (mid_market_max, "Mid-Market | Premium")]
    for x, label in boundaries:
        ax.axvline(x, color=INK_SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(x, ax.get_ylim()[1] * 0.97, f" {label}", rotation=90, va="top",
                fontsize=8.5, color=INK_SECONDARY)
    ax.set_title("Nightly price distribution, Sydney Airbnb listings")
    ax.set_xlabel("Price (AUD, clipped at $1,500)")
    ax.set_ylabel("Listings")
    return _save(fig, path)


def plot_tier_balance(df: pd.DataFrame, path: Path) -> Path:
    """Share of listings per price tier (ordinal ramp, direct labels)."""
    apply_style()
    counts = df["price_category"].value_counts().reindex(TIER_ORDER)
    share = counts / counts.sum()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bars = ax.barh(
        counts.index[::-1],
        counts[::-1],
        color=[TIER_COLORS[t] for t in counts.index[::-1]],
        height=0.62,
    )
    for bar, n, s in zip(bars, counts[::-1], share[::-1], strict=True):
        ax.text(bar.get_width() + counts.max() * 0.015, bar.get_y() + bar.get_height() / 2,
                f"{n:,}  ({s:.0%})", va="center", fontsize=10, color=INK)
    ax.set_title("Listings per price tier")
    ax.set_xlabel("Listings")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, counts.max() * 1.22)
    return _save(fig, path)


MODEL_LABELS = {
    "baseline_majority": "Baseline (majority class)",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def plot_model_comparison(metrics: dict[str, dict], primary_metric: str, path: Path) -> Path:
    """One hue for all models — bar length carries the comparison."""
    apply_style()
    names = list(metrics.keys())
    accuracy = [metrics[m]["test"]["accuracy"] for m in names]
    f1 = [metrics[m]["test"]["f1_macro"] for m in names]

    y = np.arange(len(names))
    h = 0.36
    fig, ax = plt.subplots(figsize=(8, 0.9 * len(names) + 1.6))
    ax.barh(y - h / 2, accuracy, height=h, color=SERIES_1, label="Accuracy")
    ax.barh(y + h / 2, f1, height=h, color="#9ec5f4", label="Macro F1")
    for yi, (a, f) in zip(y, zip(accuracy, f1, strict=True), strict=True):
        ax.text(a + 0.008, yi - h / 2, f"{a:.3f}", va="center", fontsize=9, color=INK)
        ax.text(f + 0.008, yi + h / 2, f"{f:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [MODEL_LABELS.get(n, n.replace("_", " ").title()) for n in names])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_title("Held-out test performance by model")
    ax.set_xlabel("Score")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="upper right", labelcolor=INK_SECONDARY)
    return _save(fig, path)


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], path: Path) -> Path:
    """Row-normalised confusion matrix on the sequential blue ramp."""
    apply_style()
    cm = np.asarray(cm, dtype=float)
    row_share = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(row_share, cmap=BLUE_CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted tier")
    ax.set_ylabel("Actual tier")
    ax.set_title("Confusion matrix (best model, test set)")
    ax.grid(visible=False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = SURFACE if row_share[i, j] > 0.55 else INK
            ax.text(j, i, f"{int(cm[i, j]):,}\n{row_share[i, j]:.0%}",
                    ha="center", va="center", fontsize=10, color=color)
    return _save(fig, path)


def plot_feature_importance(importances: pd.Series, path: Path, top_n: int = 15) -> Path:
    """Permutation importance — one hue, nominal bars."""
    apply_style()
    top = importances.sort_values(ascending=False).head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(top) + 1.4))
    ax.barh(top.index, top.values, color=SERIES_1, height=0.62)
    ax.set_title(f"What drives the price tier — top {len(top)} features")
    ax.set_xlabel("Permutation importance (drop in macro F1 when shuffled)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelcolor=INK)
    return _save(fig, path)


def plot_price_by_distance(df: pd.DataFrame, path: Path) -> Path:
    """Median price by distance-from-CBD band, split by tier share."""
    apply_style()
    bands = pd.cut(df["distance_from_cbd_km"], bins=[0, 2, 5, 10, 20, 40, np.inf],
                   labels=["0-2", "2-5", "5-10", "10-20", "20-40", "40+"])
    grouped = df.groupby(bands, observed=True)["price_numeric"].median()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(grouped.index.astype(str), grouped.values, color=SERIES_1, width=0.62)
    for i, v in enumerate(grouped.values):
        ax.text(i, v + grouped.max() * 0.015, f"${v:,.0f}", ha="center", fontsize=9.5, color=INK)
    ax.set_title("Median nightly price by distance from the CBD")
    ax.set_xlabel("Distance from Sydney CBD (km)")
    ax.set_ylabel("Median price (AUD)")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", labelcolor=MUTED)
    return _save(fig, path)
