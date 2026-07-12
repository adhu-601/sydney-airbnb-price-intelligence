"""Sydney Airbnb Price Intelligence — interactive advisor.

Run from the repo root:

    streamlit run app/streamlit_app.py

Tab 1 (Price Advisor): describe a listing, get the recommended price tier
with model confidence. Tab 2 (Market Explorer): browse the Sydney market by
neighbourhood — tier mix, medians, and a listing map.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from airbnb_pricer import TIER_ORDER  # noqa: E402
from airbnb_pricer.config import load_config  # noqa: E402
from airbnb_pricer.features.engineer import (  # noqa: E402
    amenity_flag_columns,
    bucket_host_experience,
    bucket_property_size,
    haversine_km,
)
from airbnb_pricer.models.predict import PricingAdvisor  # noqa: E402
from airbnb_pricer.viz.style import GRID, INK, SERIES_1, TIER_COLORS  # noqa: E402

st.set_page_config(page_title="Sydney Airbnb Price Intelligence", page_icon="🏠", layout="wide")

PLOTLY_LAYOUT = dict(
    font=dict(family="Segoe UI, sans-serif", color=INK),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=48, b=10),
    xaxis=dict(gridcolor=GRID, zeroline=False),
    yaxis=dict(gridcolor=GRID, zeroline=False),
)


@st.cache_resource(show_spinner="Loading model...")
def get_advisor():
    cfg = load_config(ROOT / "config" / "config.yaml")
    return PricingAdvisor.from_disk(cfg), cfg


@st.cache_data(show_spinner="Loading market data...")
def get_market_data():
    cfg = load_config(ROOT / "config" / "config.yaml")
    path = cfg.data.processed_dir / f"listings_features_{cfg.data.snapshot_date}.parquet"
    return pd.read_parquet(path)


def guard_artifacts():
    try:
        advisor, cfg = get_advisor()
        df = get_market_data()
        return advisor, cfg, df
    except FileNotFoundError:
        st.error(
            "Model or data artifacts not found. From the repo root run:\n\n"
            "```\nairbnb-pricer all\n```\n"
            "then restart this app."
        )
        st.stop()


advisor, cfg, market = guard_artifacts()

st.title("Sydney Airbnb Price Intelligence")
st.caption(
    f"Trained on the {cfg.data.snapshot_date} Inside Airbnb snapshot - "
    f"{len(market):,} listings across {market['neighbourhood_cleansed'].nunique()} neighbourhoods."
)

tab_advisor, tab_market = st.tabs(["💡 Price Advisor", "🗺️ Market Explorer"])

# --------------------------------------------------------------------------
# Tab 1 — Price Advisor
# --------------------------------------------------------------------------
with tab_advisor:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Describe the listing")

        neighbourhoods = sorted(market["neighbourhood_cleansed"].dropna().unique())
        neighbourhood = st.selectbox("Neighbourhood (council area)", neighbourhoods,
                                     index=neighbourhoods.index("Waverley") if "Waverley" in neighbourhoods else 0)

        c1, c2 = st.columns(2)
        room_type = c1.selectbox("Room type", sorted(market["room_type"].dropna().unique()))
        property_type = c2.selectbox(
            "Property type", sorted(market["property_type"].dropna().unique())
        )

        c1, c2, c3, c4 = st.columns(4)
        accommodates = c1.number_input("Guests", 1, 16, 4)
        bedrooms = c2.number_input("Bedrooms", 0, 10, 2)
        beds = c3.number_input("Beds", 0, 16, 2)
        bathrooms = c4.number_input("Bathrooms", 0.0, 8.0, 1.0, step=0.5)

        amenity_labels = {
            "has_pool": "Pool",
            "has_air_conditioning": "Air conditioning",
            "has_free_parking": "Free parking",
            "has_washer": "Washer",
            "has_dishwasher": "Dishwasher",
            "has_gym": "Gym",
        }
        selected_amenities = st.multiselect(
            "Standout amenities",
            options=list(amenity_labels.values()),
            default=["Air conditioning", "Washer"],
        )
        amenities_count = st.slider("Total number of amenities listed", 0, 100, 30)

        st.markdown("**Host profile**")
        live_features = set(advisor.feature_names)
        c1, c2, c3 = st.columns(3)
        is_superhost = c1.toggle("Superhost", value=False)
        identity_verified = c2.toggle("ID verified", value=True)
        # Only offered when the training snapshot actually populated it.
        instant_bookable = (
            c3.toggle("Instant book", value=True) if "instant_bookable" in live_features else None
        )

        c1, c2, c3 = st.columns(3)
        host_listings = c1.number_input("Host's total listings", 1, 500, 1)
        review_rating = c2.slider("Review rating", 1.0, 5.0, 4.8, 0.05)
        n_reviews = c3.number_input("Number of reviews", 0, 3000, 25)

        c1, c2 = st.columns(2)
        availability = c1.slider("Days available / year", 0, 365, 180)
        min_nights = c2.number_input("Minimum nights", 1, 365, 2)

    # Location features from the neighbourhood's listing centroid.
    hood = market[market["neighbourhood_cleansed"] == neighbourhood]
    lat, lon = hood["latitude"].median(), hood["longitude"].median()

    listing = {
        "property_type": property_type,
        "room_type": room_type,
        "accommodates": accommodates,
        "bedrooms": bedrooms,
        "beds": beds,
        "bathrooms": bathrooms,
        "neighbourhood_cleansed": neighbourhood,
        "latitude": lat,
        "longitude": lon,
        "host_is_superhost": float(is_superhost),
        "host_listings_count": host_listings,
        "host_identity_verified": float(identity_verified),
        "review_scores_rating": review_rating,
        "number_of_reviews": n_reviews,
        "reviews_per_month": round(n_reviews / 24, 2),
        "availability_365": availability,
        "minimum_nights": min_nights,
        # Engineered features supplied directly (same helpers as training).
        "distance_from_cbd_km": float(
            haversine_km(lat, lon, cfg.features.cbd_lat, cfg.features.cbd_lon)
        ),
        "distance_to_beach_km": float(
            min(
                haversine_km(lat, lon, blat, blon)
                for blat, blon in cfg.features.beaches.values()
            )
        ),
        "is_popular_area": int(neighbourhood in cfg.features.popular_areas),
        "amenities_count": amenities_count,
        "property_size": str(bucket_property_size(pd.Series([accommodates]))[0]),
        "host_experience": str(bucket_host_experience(pd.Series([host_listings]))[0]),
    }
    if instant_bookable is not None:
        listing["instant_bookable"] = float(instant_bookable)
    label_to_flag = {v: k for k, v in amenity_labels.items()}
    for flag in amenity_flag_columns(cfg):
        listing[flag] = 0
    for label in selected_amenities:
        listing[label_to_flag[label]] = 1

    pred = advisor.predict_one(listing)

    with right:
        st.subheader("Recommendation")
        st.metric("Recommended price tier", pred.tier, help="Most likely market tier")
        # Escape $ so Streamlit's markdown doesn't read "$200 - $400" as LaTeX.
        band_md = pred.band.replace("$", "\\$")
        st.markdown(
            f"**Suggested nightly band:** {band_md} &nbsp;|&nbsp; "
            f"**Model confidence:** {pred.confidence:.0%}"
        )

        prob_df = pd.DataFrame(
            {"tier": TIER_ORDER, "probability": [pred.probabilities[t] for t in TIER_ORDER]}
        )
        fig = go.Figure(
            go.Bar(
                x=prob_df["probability"],
                y=prob_df["tier"],
                orientation="h",
                marker_color=[TIER_COLORS[t] for t in TIER_ORDER],
                text=[f"{p:.0%}" for p in prob_df["probability"]],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}: %{x:.1%}<extra></extra>",
            )
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title="Tier probabilities",
            height=240,
            xaxis_tickformat=".0%",
            xaxis_range=[0, 1.05],
        )
        fig.update_yaxes(categoryorder="array", categoryarray=TIER_ORDER[::-1])
        st.plotly_chart(fig, use_container_width=True)

        # Market context: comparable listings in the same neighbourhood.
        comps = hood[hood["room_type"] == room_type]
        if len(comps) >= 10:
            st.markdown(f"**Comparable listings in {neighbourhood}** ({len(comps):,} with the same room type)")
            c1, c2, c3 = st.columns(3)
            c1.metric("Median price", f"${comps['price_numeric'].median():,.0f}")
            c2.metric("25th–75th pct", f"${comps['price_numeric'].quantile(0.25):,.0f}–"
                                        f"${comps['price_numeric'].quantile(0.75):,.0f}")
            c3.metric("Median rating", f"{comps['review_scores_rating'].median():.2f} ★")
        else:
            st.info(f"Fewer than 10 comparable listings in {neighbourhood} for this room type.")

# --------------------------------------------------------------------------
# Tab 2 — Market Explorer
# --------------------------------------------------------------------------
with tab_market:
    st.subheader("Sydney market overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Listings", f"{len(market):,}")
    c2.metric("Median nightly price", f"${market['price_numeric'].median():,.0f}")
    c3.metric("Neighbourhoods", market["neighbourhood_cleansed"].nunique())
    premium_share = (market["price_category"] == "Premium").mean()
    c4.metric("Premium share", f"{premium_share:.0%}")

    col_map, col_stats = st.columns([3, 2], gap="large")

    with col_map:
        sample = market.sample(min(len(market), 4000), random_state=7)
        fig = px.scatter_map(
            sample,
            lat="latitude",
            lon="longitude",
            color="price_category",
            category_orders={"price_category": TIER_ORDER},
            color_discrete_map=TIER_COLORS,
            hover_data={
                "price_numeric": ":$,.0f",
                "room_type": True,
                "latitude": False,
                "longitude": False,
            },
            zoom=9.3,
            height=560,
            map_style="carto-positron",
        )
        fig.update_traces(marker=dict(size=6, opacity=0.75))
        fig.update_layout(
            font=PLOTLY_LAYOUT["font"],
            margin=dict(l=0, r=0, t=32, b=0),
            title="Listings by price tier (4,000-listing sample)",
            legend=dict(orientation="h", y=-0.02, title=""),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_stats:
        top = (
            market.groupby("neighbourhood_cleansed")
            .agg(median_price=("price_numeric", "median"), listings=("id", "count"))
            .query("listings >= 100")
            .sort_values("median_price", ascending=False)
            .head(12)
            .reset_index()
        )
        fig = go.Figure(
            go.Bar(
                x=top["median_price"][::-1],
                y=top["neighbourhood_cleansed"][::-1],
                orientation="h",
                marker_color=SERIES_1,
                text=[f"${v:,.0f}" for v in top["median_price"][::-1]],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}: $%{x:,.0f} median<extra></extra>",
            )
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title="Most expensive neighbourhoods (median, ≥100 listings)",
            height=560,
            xaxis_range=[0, top["median_price"].max() * 1.18],
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Neighbourhood deep dive")
    hood_pick = st.selectbox(
        "Choose a neighbourhood", sorted(market["neighbourhood_cleansed"].dropna().unique()),
        key="explorer_hood",
    )
    hd = market[market["neighbourhood_cleansed"] == hood_pick]

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        mix = hd["price_category"].value_counts(normalize=True).reindex(TIER_ORDER).fillna(0)
        fig = go.Figure()
        for tier in TIER_ORDER:
            fig.add_bar(
                x=[mix[tier]],
                y=["Tier mix"],
                orientation="h",
                name=tier,
                marker=dict(color=TIER_COLORS[tier], line=dict(color="#ffffff", width=2)),
                hovertemplate=f"{tier}: %{{x:.0%}}<extra></extra>",
                text=f"{tier} {mix[tier]:.0%}" if mix[tier] > 0.12 else "",
                textposition="inside",
                insidetextanchor="middle",
            )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            barmode="stack",
            height=170,
            title=f"{hood_pick}: price-tier mix ({len(hd):,} listings)",
            xaxis_tickformat=".0%",
            showlegend=True,
            legend=dict(orientation="h", y=-0.5, title=""),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        stats = pd.DataFrame(
            {
                "Metric": ["Listings", "Median price", "Median rating",
                           "Median distance to CBD", "Superhost share"],
                "Value": [
                    f"{len(hd):,}",
                    f"${hd['price_numeric'].median():,.0f}",
                    f"{hd['review_scores_rating'].median():.2f} ★",
                    f"{hd['distance_from_cbd_km'].median():.1f} km",
                    f"{hd['host_is_superhost'].mean():.0%}",
                ],
            }
        )
        st.dataframe(stats, hide_index=True, use_container_width=True)

st.divider()
b, m = int(cfg.target.budget_max), int(cfg.target.mid_market_max)
st.caption(
    f"Data: Inside Airbnb (insideairbnb.com), CC BY 4.0. Price tiers - Budget: ≤\\${b}, "
    f"Mid-Market: \\${b}-{m}, Premium: >\\${m} AUD/night. Predictions are decision support, "
    "not financial advice."
)
