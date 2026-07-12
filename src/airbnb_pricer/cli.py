"""Command-line entry point.

    airbnb-pricer download            # fetch the pinned Inside Airbnb snapshot
    airbnb-pricer prepare             # clean + engineer features -> parquet
    airbnb-pricer train               # cross-validate candidates, save winner
    airbnb-pricer report              # metrics.json + all report figures
    airbnb-pricer predict --demo      # price an example listing

Run `airbnb-pricer all` to do the first four in order.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from airbnb_pricer.config import Config, load_config

logger = logging.getLogger("airbnb_pricer")


def _processed_path(cfg: Config):
    return cfg.data.processed_dir / f"listings_features_{cfg.data.snapshot_date}.parquet"


def cmd_download(cfg: Config, force: bool = False) -> None:
    from airbnb_pricer.data.download import download_snapshot

    paths = download_snapshot(cfg, force=force)
    for name, path in paths.items():
        print(f"{name}: {path}")


def cmd_prepare(cfg: Config) -> None:
    import pandas as pd  # noqa: F401  (ensures friendly error before heavy work)

    from airbnb_pricer.data.clean import clean_listings, load_raw
    from airbnb_pricer.data.download import listings_path
    from airbnb_pricer.features.engineer import engineer_features

    raw_file = listings_path(cfg)
    if not raw_file.exists():
        sys.exit(f"Raw data not found at {raw_file}. Run `airbnb-pricer download` first.")

    df = engineer_features(clean_listings(load_raw(raw_file), cfg), cfg)
    out = _processed_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Processed dataset: {out} ({len(df):,} rows, {df.shape[1]} cols)")


def cmd_train(cfg: Config) -> None:
    import pandas as pd

    from airbnb_pricer.models.evaluate import write_metrics_report
    from airbnb_pricer.models.train import save_artifact, train_candidates

    processed = _processed_path(cfg)
    if not processed.exists():
        sys.exit(f"Processed data not found at {processed}. Run `airbnb-pricer prepare` first.")

    df = pd.read_parquet(processed)
    artifact, _ = train_candidates(df, cfg)
    save_artifact(artifact, cfg)
    write_metrics_report(artifact, cfg)

    best = artifact.metrics[artifact.best_model]["test"]
    print(f"\nBest model: {artifact.best_model}")
    print(f"  test accuracy : {best['accuracy']:.4f}")
    print(f"  test macro F1 : {best['f1_macro']:.4f}")
    print(f"  test ROC AUC  : {best['roc_auc_ovr_macro']:.4f}")


def cmd_report(cfg: Config) -> None:
    import numpy as np
    import pandas as pd

    from airbnb_pricer.models.evaluate import compute_permutation_importance
    from airbnb_pricer.models.train import load_artifact, split_xy
    from airbnb_pricer.viz import plots

    df = pd.read_parquet(_processed_path(cfg))
    artifact = load_artifact(cfg)
    figures = cfg.figures_dir

    made = [
        plots.plot_price_distribution(
            df,
            figures / "price_distribution.png",
            budget_max=cfg.target.budget_max,
            mid_market_max=cfg.target.mid_market_max,
        ),
        plots.plot_tier_balance(df, figures / "tier_balance.png"),
        plots.plot_price_by_distance(df, figures / "price_by_distance.png"),
        plots.plot_model_comparison(
            artifact.metrics, cfg.model.primary_metric, figures / "model_comparison.png"
        ),
        plots.plot_confusion_matrix(
            np.array(artifact.metrics[artifact.best_model]["test"]["confusion_matrix"]),
            list(artifact.label_encoder.classes_),
            figures / "confusion_matrix.png",
        ),
    ]

    _, X_test, _, y_test, *_ = split_xy(df, cfg)
    importances = compute_permutation_importance(artifact, X_test, y_test, cfg)
    importances.to_json(cfg.reports_dir / "feature_importance.json", indent=2)
    made.append(plots.plot_feature_importance(importances, figures / "feature_importance.png"))

    for path in made:
        print(f"figure: {path}")


def cmd_predict(cfg: Config, listing_json: str | None, demo: bool) -> None:
    from airbnb_pricer.models.predict import PricingAdvisor

    if demo:
        listing = {
            "property_type": "Entire rental unit",
            "room_type": "Entire home/apt",
            "accommodates": 4,
            "bedrooms": 2,
            "beds": 2,
            "bathrooms": 1.0,
            "neighbourhood_cleansed": "Waverley",
            "latitude": -33.8908,
            "longitude": 151.2743,
            "amenities": '["Wifi", "Kitchen", "Washer", "Air conditioning", "Free parking"]',
            "host_is_superhost": 1.0,
            "host_response_rate": 0.98,
            "host_acceptance_rate": 0.95,
            "host_listings_count": 2,
            "host_identity_verified": 1.0,
            "instant_bookable": 1.0,
            "review_scores_rating": 4.85,
            "number_of_reviews": 120,
            "reviews_per_month": 2.4,
            "availability_365": 200,
            "minimum_nights": 2,
        }
        print("Demo listing: 2BR entire apartment near Bondi Beach, superhost\n")
    elif listing_json:
        listing = json.loads(listing_json)
    else:
        sys.exit("Provide --listing '<json>' or --demo")

    advisor = PricingAdvisor.from_disk(cfg)
    pred = advisor.predict_one(listing)
    print(f"Recommended tier : {pred.tier}  ({pred.band})")
    print(f"Confidence       : {pred.confidence:.1%}")
    print("Tier probabilities:")
    for tier, p in sorted(pred.probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {tier:<12} {p:.1%}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="airbnb-pricer", description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_download = sub.add_parser("download", help="Fetch the Inside Airbnb snapshot")
    p_download.add_argument("--force", action="store_true", help="Re-download even if cached")
    sub.add_parser("prepare", help="Clean data and engineer features")
    sub.add_parser("train", help="Train and select the best model")
    sub.add_parser("report", help="Write metrics.json and report figures")
    sub.add_parser("all", help="download -> prepare -> train -> report")
    p_pred = sub.add_parser("predict", help="Price a listing")
    p_pred.add_argument("--listing", help="Listing attributes as a JSON object")
    p_pred.add_argument("--demo", action="store_true", help="Use a built-in example listing")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)

    if args.command == "download":
        cmd_download(cfg, force=args.force)
    elif args.command == "prepare":
        cmd_prepare(cfg)
    elif args.command == "train":
        cmd_train(cfg)
    elif args.command == "report":
        cmd_report(cfg)
    elif args.command == "all":
        cmd_download(cfg)
        cmd_prepare(cfg)
        cmd_train(cfg)
        cmd_report(cfg)
    elif args.command == "predict":
        cmd_predict(cfg, args.listing, args.demo)


if __name__ == "__main__":
    main()
