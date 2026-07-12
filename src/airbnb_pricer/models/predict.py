"""Serving interface: turn a listing description into a pricing recommendation.

This is the single entry point the CLI, the Streamlit app, and the tests all
use, so serving logic never drifts from training logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from airbnb_pricer.config import Config
from airbnb_pricer.features.engineer import engineer_features
from airbnb_pricer.models.train import TrainedArtifact, load_artifact


def tier_bands(cfg: Config) -> dict[str, str]:
    """Human-readable nightly-price band per tier, from the config thresholds."""
    b, m = int(cfg.target.budget_max), int(cfg.target.mid_market_max)
    return {
        "Budget": f"up to ${b} / night",
        "Mid-Market": f"${b} - ${m} / night",
        "Premium": f"above ${m} / night",
    }


@dataclass(frozen=True)
class TierPrediction:
    tier: str
    band: str
    probabilities: dict[str, float]

    @property
    def confidence(self) -> float:
        return self.probabilities[self.tier]


class PricingAdvisor:
    """Loads the trained artifact once and prices listing descriptions."""

    def __init__(self, artifact: TrainedArtifact, cfg: Config):
        self.artifact = artifact
        self.cfg = cfg
        self.bands = tier_bands(cfg)

    @classmethod
    def from_disk(cls, cfg: Config) -> PricingAdvisor:
        return cls(load_artifact(cfg), cfg)

    @property
    def feature_names(self) -> list[str]:
        return self.artifact.numeric_features + self.artifact.categorical_features

    def predict(self, listing: dict | pd.DataFrame) -> list[TierPrediction]:
        """Accepts a raw-ish listing dict (or frame) and returns tier advice.

        Input uses *cleaned* column names (see docs/data_dictionary.md); any
        missing fields are imputed by the training-time pipeline.
        """
        frame = pd.DataFrame([listing]) if isinstance(listing, dict) else listing.copy()

        needs_engineering = "distance_from_cbd_km" not in frame.columns
        if needs_engineering:
            if "amenities" not in frame.columns:
                frame["amenities"] = "[]"
            for col in ("latitude", "longitude", "neighbourhood_cleansed"):
                if col not in frame.columns:
                    raise ValueError(f"Listing needs '{col}' to derive location features")
            # Optional inputs the engineering step touches; the training-time
            # imputer fills whatever stays missing.
            for col in ("property_type", "accommodates", "host_listings_count"):
                if col not in frame.columns:
                    frame[col] = pd.NA
            frame = engineer_features(frame, self.cfg)

        # Missing numerics become NaN (imputed by the fitted pipeline);
        # supplied numerics are coerced so stray NA objects can't reach numpy.
        for col in self.artifact.numeric_features:
            if col not in frame.columns:
                frame[col] = np.nan
            else:
                frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
        for col in self.artifact.categorical_features:
            if col not in frame.columns:
                frame[col] = None
        X = frame[self.feature_names]

        proba = self.artifact.pipeline.predict_proba(X)
        classes = list(self.artifact.label_encoder.classes_)
        out = []
        for row in proba:
            probs = {cls: float(p) for cls, p in zip(classes, row, strict=True)}
            tier = max(probs, key=probs.get)
            out.append(TierPrediction(tier=tier, band=self.bands[tier], probabilities=probs))
        return out

    def predict_one(self, listing: dict) -> TierPrediction:
        return self.predict(listing)[0]
