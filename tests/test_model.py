import pytest

from airbnb_pricer.features.engineer import engineer_features
from airbnb_pricer.models.predict import PricingAdvisor
from airbnb_pricer.models.train import save_artifact, train_candidates


@pytest.fixture(scope="module")
def trained(cleaned_sample, cfg):
    df = engineer_features(cleaned_sample, cfg)
    artifact, fitted = train_candidates(df, cfg)
    return artifact, fitted, df


def test_training_selects_a_real_model(trained, cfg):
    artifact, fitted, _ = trained
    assert artifact.best_model in cfg.model.candidates
    assert artifact.best_model != "baseline_majority"
    assert set(fitted) == set(cfg.model.candidates)


def test_model_beats_majority_baseline(trained):
    artifact, _, _ = trained
    baseline = artifact.metrics["baseline_majority"]["test"]["f1_macro"]
    best = artifact.metrics[artifact.best_model]["test"]["f1_macro"]
    assert best > baseline


def test_metrics_have_expected_shape(trained):
    artifact, _, _ = trained
    for metrics in artifact.metrics.values():
        assert 0 <= metrics["test"]["accuracy"] <= 1
        assert set(metrics["test"]["f1_per_class"]) == {"Budget", "Mid-Market", "Premium"}
        cm = metrics["test"]["confusion_matrix"]
        assert len(cm) == 3 and all(len(row) == 3 for row in cm)


def test_advisor_roundtrip_from_disk(trained, cfg):
    artifact, _, _ = trained
    save_artifact(artifact, cfg)
    advisor = PricingAdvisor.from_disk(cfg)
    pred = advisor.predict_one(
        {
            "property_type": "Entire rental unit",
            "room_type": "Entire home/apt",
            "accommodates": 4,
            "bedrooms": 2,
            "beds": 2,
            "bathrooms": 1.0,
            "neighbourhood_cleansed": "Sydney",
            "latitude": -33.8688,
            "longitude": 151.2093,
            "amenities": '["Wifi", "Kitchen", "Pool"]',
            "host_listings_count": 1,
        }
    )
    assert pred.tier in {"Budget", "Mid-Market", "Premium"}
    assert pred.probabilities[pred.tier] == pytest.approx(pred.confidence)
    assert sum(pred.probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_advisor_imputes_missing_optional_fields(trained, cfg):
    artifact, _, _ = trained
    advisor = PricingAdvisor(artifact, cfg)
    # Only the bare minimum: location + a couple of basics.
    pred = advisor.predict_one(
        {
            "neighbourhood_cleansed": "Waverley",
            "latitude": -33.8908,
            "longitude": 151.2743,
            "room_type": "Private room",
            "accommodates": 2,
            "host_listings_count": 1,
        }
    )
    assert pred.tier in {"Budget", "Mid-Market", "Premium"}
