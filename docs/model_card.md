# Model Card — Sydney Airbnb Price-Tier Classifier

## Model details

| | |
|---|---|
| Model | XGBoost multi-class classifier (`multi:softprob`) inside an sklearn `Pipeline` |
| Version | 1.0.0 |
| Trained | 2026-07-12, on the Inside Airbnb Sydney snapshot dated 2026-06-16 |
| Selection | Best 5-fold cross-validated macro F1 among 4 candidates (majority baseline, logistic regression, random forest, XGBoost) |
| Preprocessing | Median imputation + standardisation (numeric); mode imputation + one-hot (categorical); fitted on the training split only |
| Artifact | `models/price_tier_model.joblib` — carries the fitted pipeline, label encoder, feature lists, metrics, snapshot date, and library versions |

## Intended use

Decision support for Sydney Airbnb hosts and market analysts: given a
listing's attributes, recommend one of three nightly-price tiers —
**Budget (≤ $200)**, **Mid-Market ($200–400)**, **Premium (> $400)** AUD —
with class probabilities. Not intended for automated pricing without human
review, for markets other than Greater Sydney, or for snapshots far from the
training date (retrain instead — it is one command).

## Training data

- 17,701 listings (86% of the raw scrape) after removing rows without a
  price, prices outside $10–$5,000/night, and duplicate IDs.
- Class balance: Budget 31% / Mid-Market 39% / Premium 30%.
- 24 features: capacity, room/property type, location (lat/lon, haversine
  distance to CBD and nearest major beach), amenities (count + 6 flags),
  host scale and status, review volume/quality, availability, stay policy.
- Three snapshot columns arrived empty (`host_response_rate`,
  `host_acceptance_rate`, `instant_bookable`) and were automatically excluded.

## Evaluation

Held-out stratified test set, n = 3,541 (20%).

| Metric | Value |
|---|---|
| Accuracy | 0.798 |
| Macro F1 | 0.802 |
| Macro precision / recall | 0.806 / 0.799 |
| Macro one-vs-rest ROC AUC | 0.930 |
| F1 — Budget | 0.872 |
| F1 — Mid-Market | 0.755 |
| F1 — Premium | 0.778 |

Cross-validated macro F1: 0.7955 ± 0.0080 (5 folds).

## Error analysis

- Misclassifications are overwhelmingly *adjacent-tier*; Budget↔Premium
  confusion is 0.5% of the test set (16 of 3,541).
- Mid-Market is the hardest class (F1 0.755) — it borders both other tiers.
- Top drivers (permutation importance on input features): `minimum_nights`,
  `bedrooms`, `room_type`, `accommodates`, longitude, distance to CBD/beach.

## Ethical considerations & caveats

- Listed prices are asking prices, not accepted bookings; the model learns
  what hosts *charge*, not what guests *pay*.
- Inside Airbnb is a public scrape published for community analysis
  (CC BY 4.0); the pipeline uses only listing-level attributes, no guest data,
  and no host names or photos.
- Tier thresholds are config values calibrated to the 2026-06 market; they
  must be revisited when the price distribution moves (see
  `config/config.yaml`).
- Predictions can reflect and reproduce existing market biases (e.g.
  neighbourhood effects correlated with demographics); treat outputs as one
  input to a human pricing decision.
