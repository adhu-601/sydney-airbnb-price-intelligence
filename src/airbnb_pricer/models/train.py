"""Model training and selection.

All candidates share one preprocessing pipeline (median/mode imputation,
scaling, one-hot encoding) fitted inside the sklearn ``Pipeline`` so nothing
leaks from the test split. Candidates are compared with stratified k-fold
cross-validation on the training split; the winner is refit on the full
training split and evaluated once on the held-out test set.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from airbnb_pricer import __version__
from airbnb_pricer.config import Config
from airbnb_pricer.features.engineer import feature_columns  # noqa: F401  (re-exported)

logger = logging.getLogger(__name__)


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)],
        verbose_feature_names_out=False,
    )


def model_registry(cfg: Config) -> dict[str, object]:
    """Candidate estimators, cheapest first."""
    rs = cfg.model.random_state
    return {
        "baseline_majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=3000, C=1.0, random_state=rs),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=rs,
            class_weight="balanced_subsample",
        ),
        "xgboost": XGBClassifier(
            n_estimators=600,
            learning_rate=0.06,
            max_depth=7,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=rs,
        ),
    }


@dataclass
class TrainedArtifact:
    """Everything needed to serve and audit the model."""

    pipeline: Pipeline
    label_encoder: LabelEncoder
    numeric_features: list[str]
    categorical_features: list[str]
    metrics: dict
    best_model: str
    snapshot_date: str
    trained_at: str
    versions: dict


def usable_feature_columns(df: pd.DataFrame, cfg: Config) -> tuple[list[str], list[str]]:
    """Feature lists minus columns with no observed values in this snapshot.

    Inside Airbnb's schema drifts between scrapes (e.g. the 2026-06 snapshot
    ships host_response_rate entirely empty); training must not depend on
    columns a snapshot doesn't actually populate.
    """
    numeric, categorical = feature_columns(cfg)
    dropped = [c for c in numeric + categorical if c in df.columns and df[c].isna().all()]
    if dropped:
        logger.warning("Dropping features with no observed values in this snapshot: %s", dropped)
    numeric = [c for c in numeric if c not in dropped]
    categorical = [c for c in categorical if c not in dropped]
    return numeric, categorical


def split_xy(df: pd.DataFrame, cfg: Config):
    numeric, categorical = usable_feature_columns(df, cfg)
    X = df[numeric + categorical]
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["price_category"].astype(str))
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.model.test_size,
        stratify=y,
        random_state=cfg.model.random_state,
    )
    return X_train, X_test, y_train, y_test, encoder, numeric, categorical


def train_candidates(df: pd.DataFrame, cfg: Config) -> tuple[TrainedArtifact, dict[str, Pipeline]]:
    """Cross-validate every candidate, pick a winner, refit, package artifact."""
    X_train, X_test, y_train, y_test, encoder, numeric, categorical = split_xy(df, cfg)
    skf = StratifiedKFold(
        n_splits=cfg.model.cv_folds, shuffle=True, random_state=cfg.model.random_state
    )

    from airbnb_pricer.models.evaluate import evaluate_on_test  # circular-import guard

    registry = model_registry(cfg)
    fitted: dict[str, Pipeline] = {}
    metrics: dict[str, dict] = {}

    for name in cfg.model.candidates:
        if name not in registry:
            raise ValueError(f"Unknown model candidate '{name}' in config")
        pipe = Pipeline(
            [("prep", build_preprocessor(numeric, categorical)), ("model", registry[name])]
        )
        logger.info("Cross-validating %s (%s-fold)...", name, cfg.model.cv_folds)
        cv_scores = cross_val_score(
            pipe, X_train, y_train, cv=skf, scoring=cfg.model.primary_metric, n_jobs=1
        )
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        test_metrics = evaluate_on_test(pipe, X_test, y_test, encoder)
        metrics[name] = {
            "cv": {
                "metric": cfg.model.primary_metric,
                "mean": float(cv_scores.mean()),
                "std": float(cv_scores.std()),
                "folds": [float(s) for s in cv_scores],
            },
            "test": test_metrics,
        }
        logger.info(
            "%s: CV %s = %.4f +/- %.4f | test accuracy = %.4f",
            name,
            cfg.model.primary_metric,
            cv_scores.mean(),
            cv_scores.std(),
            test_metrics["accuracy"],
        )

    # Winner = best cross-validated primary metric, excluding the baseline
    # (it exists to give every other number a floor, not to win).
    contenders = {k: v for k, v in metrics.items() if k != "baseline_majority"}
    best = max(contenders, key=lambda k: contenders[k]["cv"]["mean"])
    logger.info("Selected model: %s", best)

    artifact = TrainedArtifact(
        pipeline=fitted[best],
        label_encoder=encoder,
        numeric_features=numeric,
        categorical_features=categorical,
        metrics=metrics,
        best_model=best,
        snapshot_date=cfg.data.snapshot_date,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        versions={
            "airbnb_pricer": __version__,
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    )
    return artifact, fitted


def save_artifact(artifact: TrainedArtifact, cfg: Config) -> str:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.model_dir / "price_tier_model.joblib"
    joblib.dump(artifact, path, compress=3)
    logger.info("Saved model artifact to %s", path)
    return str(path)


def load_artifact(cfg: Config) -> TrainedArtifact:
    return joblib.load(cfg.model_dir / "price_tier_model.joblib")
