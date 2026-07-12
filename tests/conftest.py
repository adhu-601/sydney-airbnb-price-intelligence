from dataclasses import replace
from pathlib import Path

import pytest

from airbnb_pricer.config import load_config
from airbnb_pricer.data.clean import clean_listings, load_raw

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    """Repo config with fast model settings and throwaway output dirs."""
    base = load_config(REPO_ROOT / "config" / "config.yaml")
    tmp = tmp_path_factory.mktemp("artifacts")
    model = replace(
        base.model,
        cv_folds=2,
        candidates=["baseline_majority", "logistic_regression"],
    )
    return replace(
        base,
        model=model,
        model_dir=tmp / "models",
        reports_dir=tmp / "reports",
        figures_dir=tmp / "reports" / "figures",
    )


@pytest.fixture(scope="session")
def raw_sample():
    return load_raw(FIXTURES / "listings_sample.csv.gz")


@pytest.fixture(scope="session")
def cleaned_sample(raw_sample, cfg):
    return clean_listings(raw_sample, cfg)
