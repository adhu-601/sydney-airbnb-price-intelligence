"""Feature engineering for the price-tier model.

Adds interpretable, domain-driven features on top of the cleaned table:

- ``distance_from_cbd_km`` / ``distance_to_beach_km`` — great-circle
  (haversine) distances in real kilometres. The original research prototype
  measured distance in raw lat/lon degrees, which understates east-west
  distance by ~17% at Sydney's latitude; this module fixes that.
- ``amenities_count`` and per-amenity flags (pool, air conditioning, ...).
- Host-scale, availability and popularity signals.
- Rare property types collapsed into "Other" so one-hot encoding stays sane.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from airbnb_pricer.config import Config

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0


def haversine_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray | float:
    """Great-circle distance in kilometres (vectorised)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def parse_amenities(series: pd.Series) -> pd.Series:
    """Amenities arrive as a JSON-ish list string; return a list per row."""

    def _parse(value) -> list[str]:
        if isinstance(value, list):
            return value
        if not isinstance(value, str) or not value.strip():
            return []
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            # Fall back to a rough split for malformed rows.
            return [t.strip(' "') for t in value.strip("[]{}").split(",") if t.strip(' "')]

    return series.apply(_parse)


def bucket_property_size(accommodates: pd.Series) -> pd.Series:
    bins = [0, 2, 4, 8, np.inf]
    labels = ["Small", "Medium", "Large", "Extra Large"]
    return pd.cut(accommodates, bins=bins, labels=labels, right=True)


def bucket_host_experience(host_listings_count: pd.Series) -> pd.Series:
    bins = [-np.inf, 1, 5, np.inf]
    labels = ["Single Property", "Small Portfolio", "Large Portfolio"]
    return pd.cut(host_listings_count, bins=bins, labels=labels, right=True)


def collapse_rare_categories(series: pd.Series, top_n: int) -> pd.Series:
    """Keep the ``top_n`` most frequent levels; everything else -> 'Other'."""
    top = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top), other="Other").astype("string")


def engineer_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Return the cleaned table with all model features attached."""
    out = df.copy()
    fc = cfg.features

    # --- location -----------------------------------------------------------
    out["distance_from_cbd_km"] = haversine_km(
        out["latitude"].to_numpy(), out["longitude"].to_numpy(), fc.cbd_lat, fc.cbd_lon
    )
    beach_dists = np.column_stack(
        [
            haversine_km(out["latitude"].to_numpy(), out["longitude"].to_numpy(), lat, lon)
            for lat, lon in fc.beaches.values()
        ]
    )
    out["distance_to_beach_km"] = beach_dists.min(axis=1)
    out["is_popular_area"] = out["neighbourhood_cleansed"].isin(fc.popular_areas).astype(int)

    # --- amenities ----------------------------------------------------------
    amenity_lists = parse_amenities(out["amenities"])
    out["amenities_count"] = amenity_lists.apply(len)
    lowered = amenity_lists.apply(lambda items: " | ".join(items).lower())
    for amenity in fc.key_amenities:
        col = "has_" + amenity.lower().replace(" ", "_")
        out[col] = lowered.str.contains(amenity.lower(), regex=False).astype(int)
    out = out.drop(columns=["amenities"])

    # --- size / host / availability ----------------------------------------
    out["property_size"] = bucket_property_size(out["accommodates"])
    out["host_experience"] = bucket_host_experience(out["host_listings_count"])
    out["property_type"] = collapse_rare_categories(
        out["property_type"].astype("string"), fc.max_property_types
    )

    logger.info("Engineered features: %s rows x %s cols", *out.shape)
    return out


# Columns handed to the model (everything else is id/label/bookkeeping).
NUMERIC_FEATURES = [
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "latitude",
    "longitude",
    "host_response_rate",
    "host_acceptance_rate",
    "host_listings_count",
    "review_scores_rating",
    "number_of_reviews",
    "reviews_per_month",
    "availability_365",
    "minimum_nights",
    "host_is_superhost",
    "host_identity_verified",
    "instant_bookable",
    "distance_from_cbd_km",
    "distance_to_beach_km",
    "is_popular_area",
    "amenities_count",
]

CATEGORICAL_FEATURES = [
    "property_type",
    "room_type",
    "neighbourhood_cleansed",
    "property_size",
    "host_experience",
]


def amenity_flag_columns(cfg: Config) -> list[str]:
    return ["has_" + a.lower().replace(" ", "_") for a in cfg.features.key_amenities]


def feature_columns(cfg: Config) -> tuple[list[str], list[str]]:
    """(numeric, categorical) feature lists used by the model pipeline."""
    return NUMERIC_FEATURES + amenity_flag_columns(cfg), list(CATEGORICAL_FEATURES)
