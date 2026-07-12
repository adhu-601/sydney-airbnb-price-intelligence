"""Cleaning for raw Inside Airbnb listings.

Turns the 79-column raw scrape into a tidy analysis table:
- keeps only the columns the model needs,
- parses money/percentage/boolean strings into real dtypes,
- extracts a numeric bathroom count from ``bathrooms_text``,
- removes duplicates and price outliers (data-entry errors),
- attaches the ``price_category`` target.

Missing-value *imputation* is deliberately left to the sklearn pipeline at
training time so it is fitted on the training split only (no leakage).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from airbnb_pricer.config import Config

logger = logging.getLogger(__name__)

RAW_COLUMNS = [
    "id",
    "price",
    "property_type",
    "room_type",
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "bathrooms_text",
    "amenities",
    "neighbourhood_cleansed",
    "latitude",
    "longitude",
    "host_is_superhost",
    "host_response_rate",
    "host_acceptance_rate",
    "host_listings_count",
    "host_identity_verified",
    "instant_bookable",
    "review_scores_rating",
    "number_of_reviews",
    "reviews_per_month",
    "availability_365",
    "minimum_nights",
]

NA_STRINGS = ["", "N/A", "NA", "n/a", "null", "None"]


def load_raw(path) -> pd.DataFrame:
    """Read a raw Inside Airbnb listings CSV (optionally gzipped)."""
    df = pd.read_csv(path, na_values=NA_STRINGS, keep_default_na=True, low_memory=False)
    logger.info("Loaded raw listings: %s rows x %s cols", *df.shape)
    return df


def parse_price(series: pd.Series) -> pd.Series:
    """'$1,250.00' -> 1250.0 (float, NaN preserved)."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64")
    return pd.to_numeric(
        series.astype("string").str.replace(r"[$,]", "", regex=True), errors="coerce"
    ).astype("float64")


def parse_percent(series: pd.Series) -> pd.Series:
    """'95%' -> 0.95 (proportion in [0, 1])."""
    if pd.api.types.is_numeric_dtype(series):
        out = series.astype("float64")
    else:
        out = (
            pd.to_numeric(
                series.astype("string").str.rstrip("%"),
                errors="coerce",
            ).astype("float64")
            / 100.0
        )
    return out.clip(0, 1)


def parse_bool(series: pd.Series) -> pd.Series:
    """Inside Airbnb booleans arrive as 't'/'f' strings."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    mapped = series.astype("string").str.lower().map({"t": 1.0, "f": 0.0, "true": 1.0, "false": 0.0})
    return mapped.astype("float64")


def parse_bathrooms(bathrooms: pd.Series, bathrooms_text: pd.Series) -> pd.Series:
    """Prefer the numeric column; recover from text like '1.5 shared baths'.

    'Half-bath' variants count as 0.5.
    """
    numeric = pd.to_numeric(bathrooms, errors="coerce").astype("float64")
    from_text = pd.to_numeric(
        bathrooms_text.astype("string").str.extract(r"(\d+(?:\.\d+)?)", expand=False),
        errors="coerce",
    ).astype("float64")
    half = bathrooms_text.astype("string").str.contains("half", case=False, na=False)
    from_text = from_text.mask(from_text.isna() & half, 0.5)
    return numeric.fillna(from_text)


def assign_price_category(price: pd.Series, cfg: Config) -> pd.Series:
    """Map nightly AUD price to the Budget / Mid-Market / Premium target."""
    bins = [0, cfg.target.budget_max, cfg.target.mid_market_max, np.inf]
    return pd.cut(price, bins=bins, labels=["Budget", "Mid-Market", "Premium"], right=True)


def clean_listings(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Full cleaning pass; returns one row per priceable listing."""
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Raw data is missing expected columns: {missing}")

    df = raw[RAW_COLUMNS].copy()
    n0 = len(df)

    df["price_numeric"] = parse_price(df.pop("price"))
    df["host_response_rate"] = parse_percent(df["host_response_rate"])
    df["host_acceptance_rate"] = parse_percent(df["host_acceptance_rate"])
    for col in ("host_is_superhost", "host_identity_verified", "instant_bookable"):
        df[col] = parse_bool(df[col])
    df["bathrooms"] = parse_bathrooms(df["bathrooms"], df.pop("bathrooms_text"))

    # Listings without a price can't be labelled; extreme prices are almost
    # always data-entry errors (e.g. $0 placeholders, $90,000 typos).
    df = df.dropna(subset=["price_numeric"])
    in_bounds = df["price_numeric"].between(cfg.target.min_price, cfg.target.max_price)
    df = df[in_bounds]

    df = df.drop_duplicates(subset="id", keep="first")

    df["price_category"] = assign_price_category(df["price_numeric"], cfg)

    df = df.reset_index(drop=True)
    logger.info(
        "Cleaned listings: %s -> %s rows (%.1f%% retained)",
        n0,
        len(df),
        100 * len(df) / max(n0, 1),
    )
    return df
