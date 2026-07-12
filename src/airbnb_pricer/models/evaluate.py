"""Evaluation: held-out metrics, confusion matrix, permutation importance."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

from airbnb_pricer.config import Config

logger = logging.getLogger(__name__)


def evaluate_on_test(pipe, X_test, y_test, encoder: LabelEncoder) -> dict:
    """Standard multi-class metrics on the held-out split."""
    y_pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro")),
        "roc_auc_ovr_macro": float(
            roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
        ),
        "n_test": int(len(y_test)),
    }
    # Per-class F1 keyed by tier name, for the model card.
    per_class = f1_score(y_test, y_pred, average=None)
    metrics["f1_per_class"] = {
        cls: float(score) for cls, score in zip(encoder.classes_, per_class, strict=True)
    }
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    return metrics


def compute_permutation_importance(
    artifact, X_test: pd.DataFrame, y_test: np.ndarray, cfg: Config, n_repeats: int = 5
) -> pd.Series:
    """Importance on *input* features (pre-encoding), so categoricals stay whole."""
    logger.info("Computing permutation importance (%s repeats)...", n_repeats)
    result = permutation_importance(
        artifact.pipeline,
        X_test,
        y_test,
        scoring=cfg.model.primary_metric,
        n_repeats=n_repeats,
        random_state=cfg.model.random_state,
        n_jobs=-1,
    )
    return pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)


def write_metrics_report(artifact, cfg: Config) -> Path:
    """Persist the full metrics dict alongside training metadata."""
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.reports_dir / "metrics.json"
    payload = {
        "best_model": artifact.best_model,
        "snapshot_date": artifact.snapshot_date,
        "trained_at": artifact.trained_at,
        "versions": artifact.versions,
        "models": artifact.metrics,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote %s", path)
    return path
