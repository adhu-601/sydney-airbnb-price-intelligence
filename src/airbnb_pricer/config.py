"""Typed access to config/config.yaml.

The whole pipeline is driven by one YAML file so that price thresholds,
feature parameters, and model settings are reviewable in a single place
and every run is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


@dataclass(frozen=True)
class DataConfig:
    snapshot_date: str
    base_url: str
    raw_dir: Path
    processed_dir: Path

    @property
    def listings_url(self) -> str:
        return f"{self.base_url}/{self.snapshot_date}/data/listings.csv.gz"

    @property
    def neighbourhoods_geojson_url(self) -> str:
        return f"{self.base_url}/{self.snapshot_date}/visualisations/neighbourhoods.geojson"


@dataclass(frozen=True)
class TargetConfig:
    budget_max: float
    mid_market_max: float
    min_price: float
    max_price: float


@dataclass(frozen=True)
class FeatureConfig:
    cbd_lat: float
    cbd_lon: float
    beaches: dict[str, tuple[float, float]]
    popular_areas: list[str]
    max_property_types: int
    key_amenities: list[str]


@dataclass(frozen=True)
class ModelConfig:
    test_size: float
    random_state: int
    cv_folds: int
    candidates: list[str]
    primary_metric: str


@dataclass(frozen=True)
class Config:
    data: DataConfig
    target: TargetConfig
    features: FeatureConfig
    model: ModelConfig
    model_dir: Path
    reports_dir: Path
    figures_dir: Path
    project_root: Path = field(default_factory=Path.cwd)


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration, resolving relative paths against the repo root."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    root = cfg_path.resolve().parents[1]
    with open(cfg_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    return Config(
        data=DataConfig(
            snapshot_date=raw["data"]["snapshot_date"],
            base_url=raw["data"]["base_url"].rstrip("/"),
            raw_dir=root / raw["data"]["raw_dir"],
            processed_dir=root / raw["data"]["processed_dir"],
        ),
        target=TargetConfig(**raw["target"]),
        features=FeatureConfig(
            cbd_lat=raw["features"]["cbd_lat"],
            cbd_lon=raw["features"]["cbd_lon"],
            beaches={k: tuple(v) for k, v in raw["features"]["beaches"].items()},
            popular_areas=raw["features"]["popular_areas"],
            max_property_types=raw["features"]["max_property_types"],
            key_amenities=raw["features"]["key_amenities"],
        ),
        model=ModelConfig(**raw["model"]),
        model_dir=root / raw["paths"]["model_dir"],
        reports_dir=root / raw["paths"]["reports_dir"],
        figures_dir=root / raw["paths"]["figures_dir"],
        project_root=root,
    )
