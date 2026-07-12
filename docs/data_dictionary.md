# Data dictionary

## Source

[Inside Airbnb](https://insideairbnb.com/get-the-data/) — Sydney, NSW,
Australia. Quarterly public scrape of Airbnb listings, CC BY 4.0. The snapshot
date is pinned in `config/config.yaml` (`data.snapshot_date`); the raw file is
`data/raw/listings_<date>.csv.gz` (not committed — fetch with
`airbnb-pricer download`).

## Processed table

`data/processed/listings_features_<date>.parquet` — one row per priceable
listing. Cleaned columns keep their Inside Airbnb names; engineered columns
are added by `airbnb_pricer/features/engineer.py`.

### Identifiers & target

| Column | Type | Description |
|---|---|---|
| `id` | int | Airbnb listing ID (deduplicated) |
| `price_numeric` | float | Nightly asking price, AUD, parsed from `"$1,250.00"`; bounded to $10–$5,000 |
| `price_category` | category | **Target.** Budget (≤ budget_max), Mid-Market (≤ mid_market_max), Premium (above); thresholds in config |

### Cleaned listing attributes

| Column | Type | Notes |
|---|---|---|
| `property_type` | string | Rare levels collapsed into `"Other"` (top 12 kept) |
| `room_type` | string | Entire home/apt, Private room, Shared room, Hotel room |
| `accommodates` | int | Guest capacity |
| `bedrooms`, `beds` | float | As scraped; missing values imputed at training time |
| `bathrooms` | float | Numeric column, recovered from `bathrooms_text` when missing ("1.5 shared baths" → 1.5, "Half-bath" → 0.5) |
| `neighbourhood_cleansed` | string | One of 33 Sydney council areas |
| `latitude`, `longitude` | float | Listing coordinates (Airbnb-anonymised ~150 m) |
| `host_is_superhost` | float 0/1 | Parsed from `t`/`f` |
| `host_identity_verified` | float 0/1 | Parsed from `t`/`f` |
| `host_listings_count` | float | Listings the host operates |
| `host_response_rate`, `host_acceptance_rate` | float 0–1 | Parsed from `"95%"`; **empty in the 2026-06 snapshot** (auto-dropped from the model) |
| `instant_bookable` | float 0/1 | **Empty in the 2026-06 snapshot** (auto-dropped) |
| `review_scores_rating` | float | 1–5 overall rating |
| `number_of_reviews`, `reviews_per_month` | numeric | Review volume |
| `availability_365` | int | Days bookable in the next year |
| `minimum_nights` | int | Minimum stay policy |

### Engineered features

| Column | Type | Definition |
|---|---|---|
| `distance_from_cbd_km` | float | Haversine distance to Sydney Town Hall (−33.8688, 151.2093) |
| `distance_to_beach_km` | float | Haversine distance to the nearest of Bondi, Manly, Coogee, Cronulla, Palm Beach |
| `is_popular_area` | int 0/1 | Neighbourhood in the configured high-demand list |
| `amenities_count` | int | Length of the parsed amenities list |
| `has_pool`, `has_air_conditioning`, `has_free_parking`, `has_washer`, `has_dishwasher`, `has_gym` | int 0/1 | Case-insensitive substring match over the amenities list |
| `property_size` | category | Small (≤2) / Medium (≤4) / Large (≤8) / Extra Large, by `accommodates` |
| `host_experience` | category | Single Property (≤1) / Small Portfolio (≤5) / Large Portfolio, by `host_listings_count` |

### Cleaning rules (applied in `data/clean.py`)

1. `"N/A"`, `"NA"`, empty strings, `"null"`, `"None"` → missing.
2. Drop rows without a parseable price; bound prices to $10–$5,000/night.
3. Deduplicate on `id` (keep first).
4. No imputation at the data layer — the sklearn pipeline imputes at training
   time using training-split statistics only.
