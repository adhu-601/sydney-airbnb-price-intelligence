import numpy as np
import pandas as pd

from airbnb_pricer.data.clean import (
    assign_price_category,
    parse_bathrooms,
    parse_bool,
    parse_percent,
    parse_price,
)


def test_parse_price_strips_currency_formatting():
    raw = pd.Series(["$1,250.00", "$85.00", None, "300"])
    out = parse_price(raw)
    assert out.tolist()[:2] == [1250.0, 85.0]
    assert np.isnan(out[2])
    assert out[3] == 300.0


def test_parse_percent_returns_proportions():
    out = parse_percent(pd.Series(["95%", "100%", None, "0%"]))
    assert out[0] == 0.95
    assert out[1] == 1.0
    assert np.isnan(out[2])
    assert out[3] == 0.0


def test_parse_bool_maps_t_f():
    out = parse_bool(pd.Series(["t", "f", None]))
    assert out[0] == 1.0
    assert out[1] == 0.0
    assert np.isnan(out[2])


def test_parse_bathrooms_recovers_from_text():
    numeric = pd.Series([2.0, np.nan, np.nan, np.nan])
    text = pd.Series(["2 baths", "1.5 shared baths", "Shared half-bath", None])
    out = parse_bathrooms(numeric, text)
    assert out.tolist()[:3] == [2.0, 1.5, 0.5]
    assert np.isnan(out[3])


def test_price_category_boundaries(cfg):
    b, m = cfg.target.budget_max, cfg.target.mid_market_max
    prices = pd.Series([b / 2, b, b + 0.01, m, m + 0.01, m * 3])
    cats = assign_price_category(prices, cfg).astype(str).tolist()
    assert cats == ["Budget", "Budget", "Mid-Market", "Mid-Market", "Premium", "Premium"]


def test_clean_listings_end_to_end(cleaned_sample, cfg):
    df = cleaned_sample
    assert len(df) > 0
    assert df["id"].is_unique
    assert df["price_numeric"].notna().all()
    assert df["price_numeric"].between(cfg.target.min_price, cfg.target.max_price).all()
    assert set(df["price_category"].dropna().astype(str)) <= {"Budget", "Mid-Market", "Premium"}
    # Parsed dtypes are numeric, not strings.
    for col in ("host_response_rate", "host_is_superhost", "bathrooms"):
        assert pd.api.types.is_numeric_dtype(df[col]), col
