# Research provenance

This project began as a group research project for **STAT5003 — Computational
Statistical Methods** (University of Sydney, Semester 2 2025, group W14_G03):
a comparative study of five classifiers predicting Sydney Airbnb price tiers,
reaching ~77% accuracy with ensemble models on the 2025 Inside Airbnb snapshot.

These R Markdown files are the original analysis, kept for provenance:

| File | Contents |
|---|---|
| `01_exploratory_analysis.Rmd` | Data audit, cleaning, EDA, feature-engineering plan |
| `02_final_modeling_report.Rmd` | Full pipeline: 5 models, evaluation, conclusions |

## What the production rewrite changed

The Python package in `src/` re-implements and extends that work:

- **Fixed a distance bug** — the R prototype computed "distance from CBD" in
  raw lat/lon degrees; the package uses haversine kilometres (and adds
  distance-to-nearest-beach).
- **Recalibrated the price tiers** — the 2025 thresholds ($100/$200) left 69%
  of 2026 listings in one class; tiers are now round numbers at the current
  market terciles ($200/$400), set in `config/config.yaml`.
- **Made it reproducible** — pinned data snapshot, one-command pipeline,
  deterministic seeds, tests, CI.
- **Made it usable** — a serving interface and a Streamlit advisor app instead
  of a static report.
- **Better model** — XGBoost with leakage-safe preprocessing inside the
  sklearn pipeline (imputation fitted on the training split only).
