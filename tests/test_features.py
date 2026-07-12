import numpy as np
import pandas as pd
import pytest

from airbnb_pricer.features.engineer import (
    bucket_host_experience,
    bucket_property_size,
    collapse_rare_categories,
    engineer_features,
    feature_columns,
    haversine_km,
    parse_amenities,
)


def test_haversine_zero_distance():
    assert haversine_km(-33.8688, 151.2093, -33.8688, 151.2093) == 0.0


def test_haversine_cbd_to_bondi_is_about_7km():
    # Sydney Town Hall -> Bondi Beach is ~7.2 km great-circle.
    d = haversine_km(-33.8688, 151.2093, -33.8908, 151.2743)
    assert d == pytest.approx(6.5, abs=1.5)


def test_haversine_is_in_km_not_degrees():
    # The original research prototype returned ~0.016 "degrees" for a listing
    # 1.5 km from the CBD; the fixed version must return real kilometres.
    d = haversine_km(-33.86767, 151.22497, -33.8688, 151.2093)
    assert 1.0 < d < 2.0


def test_parse_amenities_handles_json_and_junk():
    parsed = parse_amenities(pd.Series(['["Wifi", "Pool"]', "", None, "not json [Kitchen]"]))
    assert parsed[0] == ["Wifi", "Pool"]
    assert parsed[1] == []
    assert parsed[2] == []
    assert isinstance(parsed[3], list)


def test_property_size_buckets():
    out = bucket_property_size(pd.Series([1, 2, 3, 4, 5, 8, 9])).astype(str).tolist()
    assert out == ["Small", "Small", "Medium", "Medium", "Large", "Large", "Extra Large"]


def test_host_experience_buckets():
    out = bucket_host_experience(pd.Series([1, 2, 5, 6])).astype(str).tolist()
    assert out == ["Single Property", "Small Portfolio", "Small Portfolio", "Large Portfolio"]


def test_collapse_rare_categories_keeps_top_n():
    s = pd.Series(["a"] * 5 + ["b"] * 3 + ["c"] * 1 + ["d"] * 1)
    out = collapse_rare_categories(s, top_n=2)
    assert set(out.unique()) == {"a", "b", "Other"}


def test_engineer_features_adds_model_columns(cleaned_sample, cfg):
    df = engineer_features(cleaned_sample, cfg)
    numeric, categorical = feature_columns(cfg)
    for col in numeric + categorical:
        assert col in df.columns, f"missing feature column: {col}"
    # Distances must be plausible kilometres for the Sydney region.
    assert df["distance_from_cbd_km"].between(0, 120).all()
    assert df["distance_to_beach_km"].between(0, 120).all()
    assert df["amenities_count"].ge(0).all()
    assert not df["distance_from_cbd_km"].isna().any()


def test_amenity_flags_are_binary(cleaned_sample, cfg):
    df = engineer_features(cleaned_sample, cfg)
    flags = [c for c in df.columns if c.startswith("has_")]
    assert flags, "expected amenity flag columns"
    for col in flags:
        assert set(np.unique(df[col])) <= {0, 1}
