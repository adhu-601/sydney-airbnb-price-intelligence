"""Sydney Airbnb Price Intelligence — interactive advisor & market dashboard.

Run from the repo root:

    streamlit run app/streamlit_app.py

Two sections, switched with an Airbnb-style segmented control:
  • Price Advisor  — describe a listing, get its recommended price tier, a
    suggested nightly band, confidence, revenue potential and market context.
  • Market Explorer — filterable dashboard of the whole Sydney snapshot:
    map, price distribution, distance-to-CBD decay, tier mix, amenity premiums
    and a per-neighbourhood deep dive.

The visual system (olive + cream, tier colours green/amber/blue) is applied
through one CSS block and a shared Plotly styler so every figure reads as one
product.
"""

from __future__ import annotations

import json
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

# ==========================================================================
# Design tokens  (olive + cream chrome; green/amber/blue tiers — CVD-validated)
# ==========================================================================
PAGE_BG = "#F4F1E7"     # warm cream page canvas
SURFACE = "#FFFFFF"     # card surface
SURFACE_2 = "#FBF9F1"   # soft inset surface (inputs)
INK = "#23231B"         # primary text
INK_2 = "#57564A"       # secondary text
MUTED = "#8D8B7C"       # captions, axis labels
BORDER = "#E4DECE"      # hairline borders
GRID = "#ECE7D7"        # recessive chart gridlines
OLIVE = "#5C6B2E"       # primary brand olive
OLIVE_DEEP = "#3D4820"  # deep olive (nav active, emphasis)
OLIVE_SOFT = "#8A9A5B"  # soft olive
GOLD = "#C1841A"        # warm accent (== mid-tier amber, ties the palette)

TIER_COLORS = {"Budget": "#4C7A34", "Mid-Market": "#C1841A", "Premium": "#2E6FB0"}
TIER_SOFT = {"Budget": "#E5EEDC", "Mid-Market": "#F6E8CB", "Premium": "#DCE7F4"}
TIER_MID = {"Budget": 150.0, "Mid-Market": 295.0, "Premium": 560.0}

PRETTY = {
    "minimum_nights": "Minimum nights", "bedrooms": "Bedrooms", "room_type": "Room type",
    "accommodates": "Guests", "longitude": "Longitude", "distance_from_cbd_km": "Distance to CBD",
    "distance_to_beach_km": "Distance to beach", "reviews_per_month": "Reviews / month",
    "bathrooms": "Bathrooms", "property_type": "Property type", "latitude": "Latitude",
    "host_listings_count": "Host listings", "review_scores_rating": "Rating",
    "amenities_count": "Amenity count", "availability_365": "Availability",
    "number_of_reviews": "Number of reviews",
}

FONT = "'Plus Jakarta Sans', 'Segoe UI', system-ui, -apple-system, sans-serif"

st.set_page_config(
    page_title="Sydney Airbnb Price Intelligence",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ==========================================================================
# Global styling
# ==========================================================================
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{ font-family: {FONT}; }}

        /* ---- page canvas: cream + subtle olive dot texture ---- */
        .stApp {{
            background-color: {PAGE_BG};
            background-image: radial-gradient(rgba(92,107,46,0.055) 1px, transparent 1px);
            background-size: 22px 22px;
        }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stToolbar"] {{ display: none; }}
        [data-testid="stDecoration"] {{ display: none; }}
        footer {{ visibility: hidden; }}
        .block-container {{ max-width: 1480px; padding-top: 1.1rem; padding-bottom: 2rem;
                            padding-left: 3rem; padding-right: 3rem; }}

        h1, h2, h3, h4 {{ font-family: {FONT}; color: {INK}; letter-spacing: -0.02em; }}

        /* ---- hero ---- */
        .hero {{
            display: flex; justify-content: space-between; align-items: flex-end;
            gap: 24px; flex-wrap: wrap; margin: 4px 0 18px 0;
        }}
        .hero-eyebrow {{
            display: inline-flex; align-items: center; gap: 8px;
            font-size: 12px; font-weight: 700; letter-spacing: .14em; text-transform: uppercase;
            color: {OLIVE}; margin-bottom: 8px;
        }}
        .hero-eyebrow::before {{
            content: ""; width: 26px; height: 3px; border-radius: 3px;
            background: linear-gradient(90deg, {OLIVE}, {GOLD});
        }}
        .hero h1 {{ font-size: 40px; font-weight: 800; line-height: 1.02; margin: 0; color: {INK}; }}
        .hero h1 .amp {{ color: {OLIVE}; }}
        .hero p {{ color: {INK_2}; font-size: 15px; margin: 8px 0 0 0; max-width: 640px; }}
        .badge-row {{ display: flex; gap: 10px; flex-wrap: wrap; }}
        .badge {{
            background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px;
            padding: 10px 14px; min-width: 92px; box-shadow: 0 1px 2px rgba(40,40,20,.04);
        }}
        .badge .v {{ font-size: 20px; font-weight: 800; color: {INK}; line-height: 1; }}
        .badge .l {{ font-size: 11px; font-weight: 600; color: {MUTED}; margin-top: 4px;
                     text-transform: uppercase; letter-spacing: .06em; }}
        .badge.accent {{ background: {OLIVE_DEEP}; border-color: {OLIVE_DEEP}; }}
        .badge.accent .v {{ color: #FCFBF4; }}
        .badge.accent .l {{ color: #C9D2AE; }}

        /* ---- cards (st.container(border=True)) ---- */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: {SURFACE}; border: 1px solid {BORDER} !important;
            border-radius: 18px; box-shadow: 0 2px 10px rgba(45,45,25,.045);
            padding: 6px 4px;
        }}

        /* ---- section title inside cards ---- */
        .sect {{ font-size: 15px; font-weight: 700; color: {INK}; margin: 2px 0 2px 2px; }}
        .sect-sub {{ font-size: 12.5px; color: {MUTED}; margin: 0 0 6px 2px; }}

        /* ---- stat tiles ---- */
        .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px,1fr));
                      gap: 12px; margin: 2px 0 4px 0; }}
        .stat {{
            background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px;
            padding: 15px 16px; position: relative; overflow: hidden;
            box-shadow: 0 1px 3px rgba(45,45,25,.04);
        }}
        .stat::before {{ content:""; position:absolute; left:0; top:14px; bottom:14px; width:4px;
                         border-radius: 0 4px 4px 0; background: var(--ac, {OLIVE}); }}
        .stat .v {{ font-size: 26px; font-weight: 800; color: {INK}; line-height: 1.05; }}
        .stat .l {{ font-size: 12px; font-weight: 600; color: {MUTED}; margin-top: 3px;
                    text-transform: uppercase; letter-spacing: .05em; }}
        .stat .s {{ font-size: 12px; color: {INK_2}; margin-top: 6px; }}

        /* ---- recommendation hero ---- */
        .rec {{ padding: 4px 6px 2px 6px; }}
        .rec .kicker {{ font-size: 11.5px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; color: {MUTED}; }}
        .rec .tier {{ display:flex; align-items:center; gap: 12px; margin: 6px 0 4px 0; }}
        .rec .dot {{ width: 16px; height: 16px; border-radius: 50%; box-shadow: 0 0 0 4px var(--soft); }}
        .rec .tname {{ font-size: 34px; font-weight: 800; line-height: 1; color: var(--tc); letter-spacing: -.02em; }}
        .rec .band {{ font-size: 15px; color: {INK}; font-weight: 600; margin: 8px 0 2px 0; }}
        .rec .band b {{ color: {OLIVE_DEEP}; }}
        .meter {{ height: 9px; border-radius: 6px; background: {SURFACE_2}; border:1px solid {BORDER};
                  overflow: hidden; margin-top: 10px; }}
        .meter > span {{ display:block; height:100%; border-radius:6px;
                         background: linear-gradient(90deg, var(--tc), var(--tc)); }}
        .meter-l {{ display:flex; justify-content:space-between; font-size:12px; color:{MUTED}; margin-top:6px; }}

        /* ---- insight callouts ---- */
        .insight {{
            display: flex; gap: 11px; align-items: flex-start;
            background: {TIER_SOFT["Budget"]}; border: 1px solid {BORDER};
            border-left: 4px solid {OLIVE}; border-radius: 12px;
            padding: 11px 13px; margin: 8px 0; font-size: 13.5px; color: {INK_2}; line-height: 1.45;
        }}
        .insight.gold {{ background: #F7ECD3; border-left-color: {GOLD}; }}
        .insight.blue {{ background: {TIER_SOFT["Premium"]}; border-left-color: {TIER_COLORS["Premium"]}; }}
        .insight .ic {{ font-size: 17px; line-height: 1.2; }}
        .insight b {{ color: {INK}; }}

        /* ---- Airbnb-style top nav: icon over label + animated underline ---- */
        [data-testid="stButtonGroup"] {{
            display: flex !important; justify-content: center !important; gap: 52px; width: 100%;
            margin: 8px 0 22px 0; background: transparent; border: none;
            padding: 0; box-shadow: none;
        }}
        /* center the segmented-control's element wrapper too */
        [data-testid="stElementContainer"]:has(> [data-testid="stButtonGroup"]) {{ width: 100% !important; }}
        [data-testid="stButtonGroup"] button {{
            display: flex !important; flex-direction: column !important; align-items: center !important;
            gap: 7px !important; background: transparent !important; border: none !important;
            box-shadow: none !important; border-radius: 0 !important; min-height: 0 !important;
            height: auto !important; overflow: visible !important;
            padding: 8px 10px 16px 10px !important; position: relative !important; color: {MUTED} !important;
            transition: transform .25s ease, color .25s ease !important;
        }}
        [data-testid="stButtonGroup"] button > div,
        [data-testid="stButtonGroup"] button [data-testid="stMarkdownContainer"] {{
            height: auto !important; overflow: visible !important; display: block !important;
        }}
        [data-testid="stButtonGroup"] button::before {{
            font-size: 31px; line-height: 1; display: block; filter: grayscale(12%);
            transition: transform .3s cubic-bezier(.34,1.56,.64,1);
        }}
        [data-testid="stButtonGroup"] button:nth-of-type(1)::before {{ content: "🏷️"; }}
        [data-testid="stButtonGroup"] button:nth-of-type(2)::before {{ content: "🗺️"; }}
        [data-testid="stButtonGroup"] button p,
        [data-testid="stButtonGroup"] button div {{
            font-size: 15px !important; font-weight: 700 !important; margin: 0 !important;
            color: inherit !important; letter-spacing: -.01em;
        }}
        [data-testid="stButtonGroup"] button::after {{
            content: ""; position: absolute; left: 6px; right: 6px; bottom: 0; height: 3px;
            border-radius: 3px; background: {OLIVE_DEEP}; transform: scaleX(0); transform-origin: center;
            transition: transform .32s cubic-bezier(.4,0,.2,1);
        }}
        [data-testid="stButtonGroup"] button:hover {{ color: {INK} !important; transform: translateY(-2px); }}
        [data-testid="stButtonGroup"] button:hover::before {{ transform: scale(1.14) rotate(-3deg); }}
        [data-testid="stButtonGroup"] button:hover::after {{ transform: scaleX(.55); background: {OLIVE_SOFT}; }}
        [data-testid="stButtonGroup"] button[aria-checked="true"] {{ color: {OLIVE_DEEP} !important; }}
        [data-testid="stButtonGroup"] button[aria-checked="true"]::before {{ transform: scale(1.08); filter: grayscale(0%); }}
        [data-testid="stButtonGroup"] button[aria-checked="true"]::after {{ transform: scaleX(1); background: {OLIVE_DEEP}; }}
        [data-testid="stButtonGroup"] button[aria-checked="true"] p {{ color: {OLIVE_DEEP} !important; }}

        /* ---- inputs ---- */
        [data-baseweb="select"] > div, .stNumberInput input, [data-baseweb="input"] {{
            background: {SURFACE_2} !important; border-radius: 11px !important;
        }}
        .stSlider [data-baseweb="slider"] {{ padding-top: 4px; }}
        div[data-testid="stMetricValue"] {{ font-weight: 800; color: {INK}; }}
        hr {{ border-color: {BORDER}; }}
        /* section band divider */
        .band-title {{ font-size: 22px; font-weight: 800; color: {INK}; margin: 6px 0 2px 0; letter-spacing:-.02em; }}
        .band-desc {{ font-size: 13.5px; color: {INK_2}; margin: 0 0 12px 0; }}

        /* ---- KPI infographic tiles (icon + value) ---- */
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr));
                     gap: 12px; margin: 2px 0 6px 0; }}
        .kpi {{ display: flex; align-items: center; gap: 13px; background: {SURFACE};
                border: 1px solid {BORDER}; border-top: 3px solid var(--ac, {OLIVE});
                border-radius: 16px; padding: 14px 15px; box-shadow: 0 1px 3px rgba(45,45,25,.05); }}
        .kpi-ic {{ width: 44px; height: 44px; border-radius: 13px; display: flex; align-items: center;
                   justify-content: center; font-size: 21px; flex-shrink: 0; }}
        .kpi-v {{ font-size: 24px; font-weight: 800; color: {INK}; line-height: 1.05; }}
        .kpi-l {{ font-size: 11px; font-weight: 700; color: {MUTED}; text-transform: uppercase;
                  letter-spacing: .06em; margin-top: 3px; }}

        /* ---- tier guide ---- */
        .tg {{ padding: 2px 4px; }}
        .tg-row {{ display: grid; grid-template-columns: 14px 1fr 96px 42px; align-items: center;
                   gap: 11px; padding: 9px 2px; border-bottom: 1px dashed {BORDER}; }}
        .tg-row:last-child {{ border-bottom: none; }}
        .tg-dot {{ width: 13px; height: 13px; border-radius: 50%; }}
        .tg-name {{ font-weight: 700; color: {INK}; font-size: 14px; display: flex; flex-direction: column; }}
        .tg-band {{ font-weight: 500; color: {MUTED}; font-size: 11.5px; }}
        .tg-bar {{ height: 8px; background: {SURFACE_2}; border: 1px solid {BORDER};
                   border-radius: 6px; overflow: hidden; }}
        .tg-bar > span {{ display: block; height: 100%; border-radius: 6px; }}
        .tg-pct {{ font-weight: 800; color: {INK_2}; font-size: 13px; text-align: right; }}

        /* ---- market explorer section band ---- */
        .mx-band {{ font-size: 17px; font-weight: 800; color: {INK}; margin: 22px 0 10px 0;
                    display: flex; align-items: center; gap: 10px; letter-spacing: -.01em; }}
        .mx-band::before {{ content: ""; width: 5px; height: 19px; border-radius: 3px;
                            background: linear-gradient(180deg, {OLIVE}, {GOLD}); }}
        .mx-band small {{ font-size: 12.5px; font-weight: 500; color: {MUTED}; letter-spacing: 0; }}

        /* ---- centered footer ---- */
        .foot {{ text-align: center; color: {MUTED}; font-size: 12.5px; line-height: 1.6;
                 max-width: 860px; margin: 8px auto 0 auto; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================================
# Data / model loading
# ==========================================================================
@st.cache_resource(show_spinner="Loading model…")
def get_advisor():
    cfg = load_config(ROOT / "config" / "config.yaml")
    return PricingAdvisor.from_disk(cfg), cfg


@st.cache_data(show_spinner="Loading market data…")
def get_market_data():
    cfg = load_config(ROOT / "config" / "config.yaml")
    path = cfg.data.processed_dir / f"listings_features_{cfg.data.snapshot_date}.parquet"
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def get_reports():
    out = {"metrics": {}, "importance": {}}
    try:
        out["metrics"] = json.loads((ROOT / "reports" / "metrics.json").read_text())
    except Exception:
        pass
    try:
        out["importance"] = json.loads((ROOT / "reports" / "feature_importance.json").read_text())
    except Exception:
        pass
    return out


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


# ==========================================================================
# Reusable UI helpers
# ==========================================================================
def styled(fig: go.Figure, height: int = 320, title: str | None = None,
           legend: bool = False, ytitle: str | None = None, xtitle: str | None = None) -> go.Figure:
    fig.update_layout(
        font=dict(family=FONT, color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=10, t=44 if title else 12, b=8),
        height=height,
        title=dict(text=title or "", font=dict(size=15, color=INK, family=FONT),
                   x=0.01, xanchor="left", y=0.97),
        showlegend=legend,
        legend=dict(orientation="h", y=-0.16, x=0, title="", font=dict(size=12, color=INK_2)),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=BORDER,
                        font=dict(family=FONT, color=INK, size=12.5)),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER,
                     tickfont=dict(color=MUTED, size=11), title=xtitle,
                     title_font=dict(color=INK_2, size=12))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER,
                     tickfont=dict(color=MUTED, size=11), title=ytitle,
                     title_font=dict(color=INK_2, size=12))
    return fig


def sect(title: str, sub: str | None = None) -> None:
    st.markdown(f"<div class='sect'>{title}</div>", unsafe_allow_html=True)
    if sub:
        st.markdown(f"<div class='sect-sub'>{sub}</div>", unsafe_allow_html=True)


def stat_row(cards: list[dict]) -> None:
    """cards: list of {value, label, sub?, color?}."""
    html = "<div class='stat-grid'>"
    for c in cards:
        ac = c.get("color", OLIVE)
        sub = f"<div class='s'>{c['sub']}</div>" if c.get("sub") else ""
        html += (f"<div class='stat' style='--ac:{ac}'>"
                 f"<div class='v'>{c['value']}</div>"
                 f"<div class='l'>{c['label']}</div>{sub}</div>")
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def insight(text: str, icon: str = "💡", tone: str = "olive") -> None:
    cls = {"olive": "", "gold": " gold", "blue": " blue"}.get(tone, "")
    st.markdown(
        f"<div class='insight{cls}'><span class='ic'>{icon}</span><span>{text}</span></div>",
        unsafe_allow_html=True,
    )


def kpi_row(cards: list[dict]) -> None:
    """cards: list of {icon, value, label, color, soft}."""
    html = "<div class='kpi-grid'>"
    for c in cards:
        html += (
            f"<div class='kpi' style='--ac:{c['color']}'>"
            f"<div class='kpi-ic' style='background:{c['soft']};color:{c['color']}'>{c['icon']}</div>"
            f"<div><div class='kpi-v'>{c['value']}</div><div class='kpi-l'>{c['label']}</div></div>"
            f"</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def mx_band(title: str, sub: str | None = None) -> None:
    extra = f" <small>· {sub}</small>" if sub else ""
    st.markdown(f"<div class='mx-band'>{title}{extra}</div>", unsafe_allow_html=True)


def tier_guide() -> None:
    counts = market["price_category"].value_counts(normalize=True)
    bands = {
        "Budget": f"≤ ${int(cfg.target.budget_max)}",
        "Mid-Market": f"${int(cfg.target.budget_max)}–{int(cfg.target.mid_market_max)}",
        "Premium": f"> ${int(cfg.target.mid_market_max)}",
    }
    rows = ""
    for t in TIER_ORDER:
        share = float(counts.get(t, 0))
        rows += (
            f"<div class='tg-row'>"
            f"<span class='tg-dot' style='background:{TIER_COLORS[t]}'></span>"
            f"<div class='tg-name'>{t}<span class='tg-band'>{bands[t]} / night</span></div>"
            f"<div class='tg-bar'><span style='width:{share*100:.0f}%;background:{TIER_COLORS[t]}'></span></div>"
            f"<span class='tg-pct'>{share:.0%}</span></div>"
        )
    st.markdown(f"<div class='tg'>{rows}</div>", unsafe_allow_html=True)


def money(x: float) -> str:
    return f"${x:,.0f}"


# ==========================================================================
# Bootstrap
# ==========================================================================
inject_css()
advisor, cfg, market = guard_artifacts()
reports = get_reports()

xgb = reports["metrics"].get("models", {}).get("xgboost", {}).get("test", {})
acc = xgb.get("accuracy")
n_hoods = market["neighbourhood_cleansed"].nunique()

# ---- Hero -----------------------------------------------------------------
acc_badge = (
    f"<div class='badge accent'><div class='v'>{acc*100:.0f}%</div>"
    f"<div class='l'>Model accuracy</div></div>" if acc else ""
)
st.markdown(
    f"""
    <div class="hero">
      <div>
        <div class="hero-eyebrow">Sydney · Inside Airbnb {cfg.data.snapshot_date}</div>
        <h1>Airbnb Price <span class="amp">Intelligence</span></h1>
        <p>Price any Sydney listing into its market tier, then explore what moves
           nightly rates across {n_hoods} neighbourhoods — distance to the harbour,
           room type, host signals and amenities.</p>
      </div>
      <div class="badge-row">
        <div class="badge"><div class="v">{len(market):,}</div><div class="l">Listings</div></div>
        <div class="badge"><div class="v">{n_hoods}</div><div class="l">Neighbourhoods</div></div>
        <div class="badge"><div class="v">{money(market['price_numeric'].median())}</div><div class="l">Median / night</div></div>
        {acc_badge}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Airbnb-style top nav (icon over label, animated underline via CSS) ----
section = st.segmented_control(
    "Section", options=["Price Advisor", "Market Explorer"], default="Price Advisor",
    label_visibility="collapsed", key="nav",
)
if not section:
    section = "Price Advisor"


# ==========================================================================
# SECTION 1 — PRICE ADVISOR
# ==========================================================================
def render_advisor() -> None:
    st.markdown("<div class='band-title'>Price a listing</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='band-desc'>Describe a property and the model places it in a market "
        "tier with a suggested nightly band, confidence and revenue outlook.</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1.05], gap="large")

    with left:
        with st.container(border=True):
            sect("Describe the listing", "Inputs mirror the trained feature set.")
            neighbourhoods = sorted(market["neighbourhood_cleansed"].dropna().unique())
            neighbourhood = st.selectbox(
                "Neighbourhood (council area)", neighbourhoods,
                index=neighbourhoods.index("Waverley") if "Waverley" in neighbourhoods else 0,
            )
            c1, c2 = st.columns(2)
            room_type = c1.selectbox("Room type", sorted(market["room_type"].dropna().unique()))
            property_type = c2.selectbox("Property type", sorted(market["property_type"].dropna().unique()))

            c1, c2, c3, c4 = st.columns(4)
            accommodates = c1.number_input("Guests", 1, 16, 4)
            bedrooms = c2.number_input("Bedrooms", 0, 10, 2)
            beds = c3.number_input("Beds", 0, 16, 2)
            bathrooms = c4.number_input("Bathrooms", 0.0, 8.0, 1.0, step=0.5)

            amenity_labels = {
                "has_pool": "Pool", "has_air_conditioning": "Air conditioning",
                "has_free_parking": "Free parking", "has_washer": "Washer",
                "has_dishwasher": "Dishwasher", "has_gym": "Gym",
            }
            selected_amenities = st.multiselect(
                "Standout amenities", options=list(amenity_labels.values()),
                default=["Air conditioning", "Washer"],
            )
            amenities_count = st.slider("Total number of amenities listed", 0, 100, 30)

            st.markdown("**Host profile**")
            live_features = set(advisor.feature_names)
            c1, c2, c3 = st.columns(3)
            is_superhost = c1.toggle("Superhost", value=False)
            identity_verified = c2.toggle("ID verified", value=True)
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

        with st.container(border=True):
            sect("Price tiers in Sydney", "How the market splits, and the band each tier covers.")
            tier_guide()

        with st.container(border=True):
            sect("Under the hood", "XGBoost model · held-out test performance.")
            if xgb:
                stat_row([
                    {"value": f"{xgb.get('accuracy', 0):.0%}", "label": "Accuracy", "color": OLIVE},
                    {"value": f"{xgb.get('f1_macro', 0):.2f}", "label": "Macro F1", "color": GOLD},
                    {"value": f"{xgb.get('roc_auc_ovr_macro', 0):.2f}", "label": "ROC-AUC",
                     "color": TIER_COLORS["Premium"]},
                ])
            if reports["importance"]:
                top3 = sorted(reports["importance"].items(), key=lambda kv: kv[1], reverse=True)[:3]
                drivers = ", ".join(PRETTY.get(k, k).lower() for k, _ in top3)
                insight(
                    f"Trained on 35 features across {len(market):,} listings. "
                    f"The strongest price signals are <b>{drivers}</b>.",
                    icon="🧠", tone="blue")

    # ---- assemble the listing feature dict (unchanged interface) ----
    hood = market[market["neighbourhood_cleansed"] == neighbourhood]
    lat, lon = hood["latitude"].median(), hood["longitude"].median()
    listing = {
        "property_type": property_type, "room_type": room_type,
        "accommodates": accommodates, "bedrooms": bedrooms, "beds": beds, "bathrooms": bathrooms,
        "neighbourhood_cleansed": neighbourhood, "latitude": lat, "longitude": lon,
        "host_is_superhost": float(is_superhost), "host_listings_count": host_listings,
        "host_identity_verified": float(identity_verified), "review_scores_rating": review_rating,
        "number_of_reviews": n_reviews, "reviews_per_month": round(n_reviews / 24, 2),
        "availability_365": availability, "minimum_nights": min_nights,
        "distance_from_cbd_km": float(haversine_km(lat, lon, cfg.features.cbd_lat, cfg.features.cbd_lon)),
        "distance_to_beach_km": float(min(
            haversine_km(lat, lon, blat, blon) for blat, blon in cfg.features.beaches.values())),
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
    tc = TIER_COLORS[pred.tier]
    comps = hood[hood["room_type"] == room_type]
    rep_price = float(comps["price_numeric"].median()) if len(comps) >= 10 else TIER_MID[pred.tier]

    with right:
        # ---- recommendation hero ----
        with st.container(border=True):
            band_txt = pred.band.replace("$", "&#36;")
            st.markdown(
                f"""
                <div class="rec" style="--tc:{tc}; --soft:{TIER_SOFT[pred.tier]}">
                  <div class="kicker">Recommended price tier</div>
                  <div class="tier"><span class="dot" style="background:{tc}"></span>
                       <span class="tname">{pred.tier}</span></div>
                  <div class="band">Suggested nightly band&nbsp; <b>{band_txt}</b></div>
                  <div class="meter"><span style="width:{pred.confidence*100:.0f}%"></span></div>
                  <div class="meter-l"><span>Model confidence</span><span><b>{pred.confidence:.0%}</b></span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            prob_df = pd.DataFrame(
                {"tier": TIER_ORDER, "p": [pred.probabilities[t] for t in TIER_ORDER]})
            fig = go.Figure(go.Bar(
                x=prob_df["p"], y=prob_df["tier"], orientation="h",
                marker=dict(color=[TIER_COLORS[t] for t in TIER_ORDER],
                            line=dict(color=SURFACE, width=2)),
                text=[f"{p:.0%}" for p in prob_df["p"]], textposition="outside",
                textfont=dict(color=INK_2, size=12), cliponaxis=False,
                hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
            fig.update_yaxes(categoryorder="array", categoryarray=TIER_ORDER[::-1])
            fig = styled(fig, height=180, title="Tier probabilities")
            fig.update_layout(xaxis_tickformat=".0%", xaxis_range=[0, 1.12])
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ---- revenue outlook ----
        with st.container(border=True):
            sect("Revenue outlook", "Illustrative — nightly rate × booked nights.")
            open_nights = min(int(availability), 365)
            est = rep_price * open_nights * 0.65
            lo = rep_price * open_nights * 0.55
            hi = rep_price * open_nights * 0.75
            stat_row([
                {"value": money(rep_price), "label": "Nightly rate (est.)", "color": tc,
                 "sub": "median of comps" if len(comps) >= 10 else "tier midpoint"},
                {"value": f"{open_nights}", "label": "Open nights / yr", "color": OLIVE,
                 "sub": "at 65% occupancy"},
                {"value": money(est), "label": "Annual revenue (est.)", "color": GOLD,
                 "sub": f"range {money(lo)}–{money(hi)}"},
            ])

        # ---- market context ----
        with st.container(border=True):
            if len(comps) >= 10:
                sect(f"Where you'd sit in {neighbourhood}",
                     f"{len(comps):,} comparable {room_type.lower()} listings.")
                cp = comps["price_numeric"].clip(upper=comps["price_numeric"].quantile(0.98))
                fig = go.Figure(go.Histogram(
                    x=cp, nbinsx=34, marker=dict(color=OLIVE_SOFT, line=dict(color=SURFACE, width=1)),
                    hovertemplate="$%{x:.0f}/night<br>%{y} listings<extra></extra>"))
                fig.add_vline(x=rep_price, line=dict(color=OLIVE_DEEP, width=2.5, dash="dot"),
                              annotation_text=f"you · {money(rep_price)}",
                              annotation_position="top",
                              annotation_font=dict(color=OLIVE_DEEP, size=12))
                for thr, col in [(cfg.target.budget_max, TIER_COLORS["Budget"]),
                                 (cfg.target.mid_market_max, TIER_COLORS["Premium"])]:
                    if cp.min() <= thr <= cp.max():
                        fig.add_vline(x=thr, line=dict(color=col, width=1, dash="dash"))
                fig = styled(fig, height=210, title="Nightly price distribution of comps",
                             xtitle="AUD / night", ytitle="listings")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                pctile = (comps["price_numeric"] < rep_price).mean()
                insight(
                    f"A nightly rate of <b>{money(rep_price)}</b> sits above <b>{pctile:.0%}</b> "
                    f"of comparable {room_type.lower()} listings in {neighbourhood}.",
                    icon="📈", tone="gold")

                stat_row([
                    {"value": money(comps["price_numeric"].median()), "label": "Median price", "color": tc},
                    {"value": f"${comps['price_numeric'].quantile(.25):,.0f}–{comps['price_numeric'].quantile(.75):,.0f}",
                     "label": "25th–75th pct", "color": OLIVE},
                    {"value": f"{comps['review_scores_rating'].median():.2f} ★", "label": "Median rating", "color": GOLD},
                ])
            else:
                st.info(f"Fewer than 10 comparable listings in {neighbourhood} for this room type.")

        # ---- location + host insights ----
        closer = (market["distance_from_cbd_km"] > listing["distance_from_cbd_km"]).mean()
        insight(
            f"At <b>{listing['distance_from_cbd_km']:.1f} km</b> from the CBD and "
            f"<b>{listing['distance_to_beach_km']:.1f} km</b> from the nearest major beach, "
            f"this location is closer to town than <b>{closer:.0%}</b> of Sydney listings.",
            icon="📍")
        sh_med = market.groupby(market["host_is_superhost"] > 0)["price_numeric"].median()
        if True in sh_med.index and False in sh_med.index:
            gap = sh_med[True] - sh_med[False]
            tone = "blue" if is_superhost else "olive"
            verb = "You benefit from the" if is_superhost else "Turning Superhost could add to a"
            insight(
                f"{verb} <b>{money(abs(gap))}</b> median nightly gap Superhosts command across Sydney.",
                icon="⭐", tone=tone)


# ==========================================================================
# SECTION 2 — MARKET EXPLORER
# ==========================================================================
def render_market() -> None:
    st.markdown("<div class='band-title'>Explore the market</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='band-desc'>Filter the full snapshot and watch every panel update — "
        "map, price distribution, distance decay, room-type mix and amenity premiums.</div>",
        unsafe_allow_html=True,
    )

    # ---- filters ----
    with st.container(border=True):
        sect("Filters")
        f1, f2, f3 = st.columns([1.2, 1.4, 1.2])
        rooms = f1.multiselect("Room type", sorted(market["room_type"].dropna().unique()),
                               default=sorted(market["room_type"].dropna().unique()))
        pmin, pmax = int(market["price_numeric"].min()), int(market["price_numeric"].quantile(0.99))
        prange = f2.slider("Nightly price band (AUD)", pmin, pmax, (pmin, pmax), step=10)
        tiers = f3.multiselect("Price tier", TIER_ORDER, default=TIER_ORDER)

    fdf = market[
        market["room_type"].isin(rooms if rooms else market["room_type"].unique())
        & market["price_numeric"].between(*prange)
        & market["price_category"].isin(tiers if tiers else TIER_ORDER)
    ]
    if fdf.empty:
        st.warning("No listings match these filters — widen the range.")
        return

    # ---- KPI infographic row ----
    mx_band("Market pulse", "live totals for the current filter")
    prem = (fdf["price_category"] == "Premium").mean()
    kpi_row([
        {"icon": "🏠", "value": f"{len(fdf):,}", "label": "Listings",
         "color": OLIVE, "soft": "#E9EBD9"},
        {"icon": "💵", "value": money(fdf["price_numeric"].median()), "label": "Median / night",
         "color": GOLD, "soft": "#F6EACB"},
        {"icon": "📍", "value": f"{fdf['neighbourhood_cleansed'].nunique()}", "label": "Neighbourhoods",
         "color": OLIVE_SOFT, "soft": "#EAEEDD"},
        {"icon": "💎", "value": f"{prem:.0%}", "label": "Premium share",
         "color": TIER_COLORS["Premium"], "soft": TIER_SOFT["Premium"]},
        {"icon": "🏅", "value": f"{fdf['host_is_superhost'].mean():.0%}", "label": "Superhosts",
         "color": TIER_COLORS["Budget"], "soft": TIER_SOFT["Budget"]},
        {"icon": "⭐", "value": f"{fdf['review_scores_rating'].median():.2f}", "label": "Median rating",
         "color": "#B8860B", "soft": "#F5E9C8"},
    ])

    # ---- map + top neighbourhoods ----
    mx_band("Where the listings are", "geography of price across Sydney")
    col_map, col_bar = st.columns([1.55, 1], gap="large")
    with col_map:
        with st.container(border=True):
            sect("Listings map", "Coloured by price tier · hover any point for details.")
            sample = fdf.sample(min(len(fdf), 4000), random_state=7)
            fig = px.scatter_map(
                sample, lat="latitude", lon="longitude", color="price_category",
                category_orders={"price_category": TIER_ORDER}, color_discrete_map=TIER_COLORS,
                custom_data=["neighbourhood_cleansed", "price_numeric", "room_type",
                             "bedrooms", "review_scores_rating", "distance_from_cbd_km"],
                zoom=9.4, height=520, map_style="carto-positron")
            fig.update_traces(marker=dict(size=7, opacity=0.8), hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "<b>$%{customdata[1]:,.0f}</b> / night · %{customdata[2]}<br>"
                "%{customdata[3]:.0f} bd · ★ %{customdata[4]:.2f}<br>"
                "%{customdata[5]:.1f} km to CBD<extra></extra>"))
            fig.update_layout(font=dict(family=FONT, color=INK), margin=dict(l=0, r=0, t=6, b=0),
                              legend=dict(orientation="h", y=-0.02, x=0, title="",
                                          font=dict(size=12, color=INK_2)),
                              hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=BORDER,
                                              font=dict(family=FONT, color=INK, size=12.5)))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_bar:
        with st.container(border=True):
            sect("Priciest neighbourhoods", "Median nightly rate · ≥100 listings.")
            top = (fdf.groupby("neighbourhood_cleansed")
                   .agg(median_price=("price_numeric", "median"), listings=("id", "count"))
                   .query("listings >= 100").sort_values("median_price").tail(11).reset_index())
            fig = go.Figure(go.Bar(
                x=top["median_price"], y=top["neighbourhood_cleansed"], orientation="h",
                marker=dict(color=OLIVE, line=dict(color=SURFACE, width=1)),
                text=[money(v) for v in top["median_price"]], textposition="outside",
                textfont=dict(color=INK_2, size=11), cliponaxis=False,
                hovertemplate="%{y}: $%{x:,.0f} median<extra></extra>"))
            fig = styled(fig, height=520)
            fig.update_layout(xaxis_range=[0, top["median_price"].max() * 1.2])
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    if len(top):
        insight(f"<b>{top.iloc[-1]['neighbourhood_cleansed']}</b> is the priciest large market at "
                f"<b>{money(top.iloc[-1]['median_price'])}</b> median — "
                f"{top.iloc[-1]['median_price']/fdf['price_numeric'].median()-1:+.0%} vs the city median.",
                icon="🏆", tone="gold")

    # ---- distribution + distance decay ----
    mx_band("What drives nightly price", "distribution, location and features")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        with st.container(border=True):
            sect("Price distribution by tier", "Where the market's nightly rates concentrate.")
            fig = go.Figure()
            for t in TIER_ORDER:
                s = fdf.loc[fdf["price_category"] == t, "price_numeric"]
                s = s[s <= fdf["price_numeric"].quantile(0.98)]
                fig.add_trace(go.Histogram(
                    x=s, name=t, marker=dict(color=TIER_COLORS[t], line=dict(color=SURFACE, width=1)),
                    opacity=0.85, hovertemplate=f"{t}<br>$%{{x:.0f}}: %{{y}}<extra></extra>"))
            fig = styled(fig, height=300, legend=True, xtitle="AUD / night", ytitle="listings")
            fig.update_layout(barmode="stack", bargap=0.04)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c2:
        with st.container(border=True):
            sect("Distance to CBD vs price", "Median nightly rate by distance ring.")
            d = fdf.copy()
            d["ring"] = pd.cut(d["distance_from_cbd_km"], bins=[0, 2, 4, 6, 8, 10, 15, 20, 100],
                               labels=["0–2", "2–4", "4–6", "6–8", "8–10", "10–15", "15–20", "20+"])
            ring = d.groupby("ring", observed=True)["price_numeric"].median().reset_index()
            fig = go.Figure(go.Scatter(
                x=ring["ring"], y=ring["price_numeric"], mode="lines+markers",
                line=dict(color=OLIVE, width=3), marker=dict(size=9, color=OLIVE_DEEP),
                fill="tozeroy", fillcolor="rgba(92,107,46,0.10)",
                hovertemplate="%{x} km<br>$%{y:,.0f} median<extra></extra>"))
            fig = styled(fig, height=300, xtitle="km from CBD", ytitle="median AUD / night")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            near = fdf.loc[fdf["distance_from_cbd_km"] <= 5, "price_numeric"].median()
            far = fdf.loc[fdf["distance_from_cbd_km"] >= 15, "price_numeric"].median()
            if pd.notna(near) and pd.notna(far) and far:
                insight(f"Listings within 5 km of the CBD run <b>{money(near)}</b> median vs "
                        f"<b>{money(far)}</b> beyond 15 km — a <b>{near/far-1:+.0%}</b> premium for being central.",
                        icon="🧭")

    # ---- tier mix by room type + amenity premiums ----
    c1, c2 = st.columns(2, gap="large")
    with c1:
        with st.container(border=True):
            sect("Tier mix by room type", "Share of each tier within a room type.")
            mix = (fdf.groupby("room_type", observed=True)["price_category"]
                   .value_counts(normalize=True).rename("share").reset_index())
            fig = go.Figure()
            for t in TIER_ORDER:
                sub = mix[mix["price_category"] == t]
                fig.add_trace(go.Bar(
                    x=sub["share"], y=sub["room_type"], orientation="h", name=t,
                    marker=dict(color=TIER_COLORS[t], line=dict(color=SURFACE, width=2)),
                    hovertemplate=f"{t}: %{{x:.0%}}<extra></extra>"))
            fig = styled(fig, height=300, legend=True, xtitle="share of listings")
            fig.update_layout(barmode="stack", xaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c2:
        with st.container(border=True):
            sect("Amenity price premium", "Median nightly rate: with vs without.")
            amen = {"has_pool": "Pool", "has_air_conditioning": "Air con", "has_gym": "Gym",
                    "has_free_parking": "Parking", "has_dishwasher": "Dishwasher", "has_washer": "Washer"}
            rows = []
            for col, lab in amen.items():
                with_m = fdf.loc[fdf[col] == 1, "price_numeric"].median()
                wo_m = fdf.loc[fdf[col] == 0, "price_numeric"].median()
                if pd.notna(with_m) and pd.notna(wo_m):
                    rows.append((lab, with_m - wo_m))
            rows.sort(key=lambda r: r[1])
            labs = [r[0] for r in rows]
            vals = [r[1] for r in rows]
            colors = [TIER_COLORS["Budget"] if v >= 0 else TIER_COLORS["Premium"] for v in vals]
            fig = go.Figure(go.Bar(
                x=vals, y=labs, orientation="h",
                marker=dict(color=colors, line=dict(color=SURFACE, width=1)),
                text=[f"{'+' if v >= 0 else '−'}${abs(v):,.0f}" for v in vals],
                textposition="outside", textfont=dict(color=INK_2, size=11), cliponaxis=False,
                hovertemplate="%{y}: %{x:+,.0f} median diff<extra></extra>"))
            fig = styled(fig, height=300, xtitle="median price difference (AUD)")
            rng = max(abs(min(vals)), abs(max(vals))) * 1.35 if vals else 1
            fig.update_layout(xaxis_range=[min(0, min(vals) * 1.35), rng])
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            if rows:
                best = rows[-1]
                insight(f"A <b>{best[0].lower()}</b> tracks the biggest gap — "
                        f"<b>+{money(best[1])}</b> median nightly vs listings without one.",
                        icon="🏊", tone="gold")

    # ---- model transparency + market composition ----
    mx_band("How the model decides", "and what the market is made of")
    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        with st.container(border=True):
            sect("What the model weighs most", "Global XGBoost feature importance (top 10).")
            if reports["importance"]:
                imp = sorted(reports["importance"].items(), key=lambda kv: kv[1], reverse=True)[:10][::-1]
                labs = [PRETTY.get(k, k) for k, _ in imp]
                vals = [v for _, v in imp]
                fig = go.Figure(go.Bar(
                    x=vals, y=labs, orientation="h",
                    marker=dict(color=GOLD, line=dict(color=SURFACE, width=1)),
                    hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
                fig = styled(fig, height=320, xtitle="importance (gain)")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Feature importance report not available.")
    with c2:
        with st.container(border=True):
            sect("Listings by room type", "Composition of the filtered market.")
            comp = fdf["room_type"].value_counts()
            room_palette = [OLIVE_DEEP, OLIVE, OLIVE_SOFT, GOLD]
            fig = go.Figure(go.Pie(
                labels=comp.index.tolist(), values=comp.values.tolist(), hole=0.58,
                marker=dict(colors=room_palette[:len(comp)], line=dict(color=SURFACE, width=2)),
                sort=False, textinfo="percent", textfont=dict(color="#FFFFFF", size=12),
                hovertemplate="%{label}<br>%{value:,} listings (%{percent})<extra></extra>"))
            fig = styled(fig, height=320, legend=True)
            fig.update_layout(annotations=[dict(
                text=f"<b>{len(fdf):,}</b><br><span style='font-size:11px;color:{MUTED}'>listings</span>",
                x=0.5, y=0.5, font=dict(size=18, color=INK, family=FONT), showarrow=False)])
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ---- neighbourhood deep dive ----
    mx_band("Neighbourhood deep dive", "zoom into one council area")
    with st.container(border=True):
        sect("Pick a neighbourhood")
        hood_pick = st.selectbox("Choose a neighbourhood",
                                 sorted(fdf["neighbourhood_cleansed"].dropna().unique()),
                                 key="explorer_hood")
        hd = fdf[fdf["neighbourhood_cleansed"] == hood_pick]
        city_med = fdf["price_numeric"].median()
        stat_row([
            {"value": f"{len(hd):,}", "label": "Listings", "color": OLIVE},
            {"value": money(hd["price_numeric"].median()), "label": "Median / night", "color": GOLD,
             "sub": f"{hd['price_numeric'].median()/city_med-1:+.0%} vs city"},
            {"value": f"{hd['review_scores_rating'].median():.2f}★", "label": "Median rating", "color": INK_2},
            {"value": f"{hd['distance_from_cbd_km'].median():.1f} km", "label": "To CBD", "color": OLIVE_SOFT},
            {"value": f"{hd['host_is_superhost'].mean():.0%}", "label": "Superhosts", "color": TIER_COLORS["Budget"]},
        ])
        mix = hd["price_category"].value_counts(normalize=True).reindex(TIER_ORDER).fillna(0)
        fig = go.Figure()
        for t in TIER_ORDER:
            fig.add_bar(x=[mix[t]], y=["Tier mix"], orientation="h", name=t,
                        marker=dict(color=TIER_COLORS[t], line=dict(color=SURFACE, width=2)),
                        text=f"{t} · {mix[t]:.0%}" if mix[t] > 0.10 else "",
                        textposition="inside", insidetextanchor="middle",
                        textfont=dict(color="#FFFFFF", size=12),
                        hovertemplate=f"{t}: %{{x:.0%}}<extra></extra>")
        fig = styled(fig, height=150, legend=True)
        fig.update_layout(barmode="stack", xaxis_tickformat=".0%",
                          xaxis_range=[0, 1], yaxis_showticklabels=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ==========================================================================
# Render selected section + footer
# ==========================================================================
if section == "Price Advisor":
    render_advisor()
else:
    render_market()

st.divider()
b, m = int(cfg.target.budget_max), int(cfg.target.mid_market_max)
st.markdown(
    f"<div class='foot'>Data: Inside Airbnb (insideairbnb.com), CC BY 4.0 · snapshot "
    f"{cfg.data.snapshot_date}. Tiers — Budget ≤ ${b}, Mid-Market ${b}–{m}, Premium &gt; ${m} "
    f"AUD/night. Model: XGBoost classifier. Estimates are decision support, not financial advice.</div>",
    unsafe_allow_html=True,
)
