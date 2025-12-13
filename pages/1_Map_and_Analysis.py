# pages/1_Map_and_Analysis.py

import streamlit as st
import json, random, time

import numpy as np
import pandas as pd

import folium
from streamlit_folium import st_folium
from folium.plugins import Draw

from sklearn.linear_model import LinearRegression  # kept for possible future use

import os
import sys
import requests
from shapely.geometry import shape, Point
import math

from ui_theme import inject_global_css  # new import

# ------------------------------------------------------------------
# Make sure we can import gee.py from project root
# ------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from gee import compute_class_areas  # our GEE helper

# Apply global dark theme
inject_global_css()

# ------------------------------------------------------------------
# Page title and explanation
# ------------------------------------------------------------------
st.title("🗺️ Map & Analysis")

# Short explanation for your professor or user
st.markdown(
    """
    **Step 1.** Draw the area of interest on the map.  
    **Step 2.** Choose the input year and the target year.  
    **Step 3.** Run the analysis to compute land cover change and a future projection for that area.
    """,
)

st.markdown("#### 1 · Draw your region of interest")



# ------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------
LC_CLASSES = [
    "Tree cover",
    "Shrubland",
    "Grassland",
    "Cropland",
    "Built-up",
    "Bare / sparse",
    "Snow/Ice",
    "Permanent water",
    "Herbaceous wetland",
    "Mangroves",
    "Moss/Lichen",
]

# Actual GEE reference years (WorldCover)
REF_START_YEAR = 2020
REF_LATEST_YEAR = 2021
REF_YEAR_GAP = REF_LATEST_YEAR - REF_START_YEAR  # should be 1

NOMINATIM_USER_AGENT = "TerraTrackAI/1.0 (capstone project)"


# ------------------------------------------------------------------
# Base map (cached)
# ------------------------------------------------------------------
@st.cache_resource
def create_base_map():
    """Base folium map with drawing tools."""
    m = folium.Map(
        location=[18.5204, 73.8567],  # Pune-ish center
        zoom_start=7,
        tiles="CartoDB positron",
    )
    Draw(export=False).add_to(m)
    return m


# Cache GEE calls per AOI+year to speed up repeat runs
@st.cache_data(show_spinner=False)
def get_class_areas_cached(aoi_geojson_str: str, year: int) -> dict:
    """Cached wrapper around compute_class_areas using a stable string key."""
    geo = json.loads(aoi_geojson_str)
    return compute_class_areas(geo, year=year)


@st.cache_data(show_spinner=False)
def reverse_geocode_point(lat: float, lon: float):
    """
    Reverse geocode a single lat/lon to (city, state, country) using OSM Nominatim.

    Returns (city, state, country) – any of them can be None on failure.
    """
    try:
        url = (
            "https://nominatim.openstreetmap.org/reverse"
            f"?format=json&lat={lat}&lon={lon}&zoom=10&addressdetails=1"
        )
        headers = {"User-Agent": NOMINATIM_USER_AGENT}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None, None, None
        data = resp.json()
        addr = data.get("address", {})

        city = (
            addr.get("city")
            or addr.get("town")
            or addr.get("village")
            or addr.get("county")
        )
        state = addr.get("state")
        country = addr.get("country")
        return city, state, country
    except Exception:
        return None, None, None


def get_main_city_and_covering_cities(aoi_geojson: dict, max_points: int = 9):
    """
    Given an AOI GeoJSON geometry, compute:

    - main_city:   city at AOI centroid (nearest main city)
    - state_name:  state of that main city (if available)
    - covering_cities: list of unique cities touched by AOI
    - area_label:  human-readable label (city, state, country) or "Selected AOI"
    """
    try:
        poly = shape(aoi_geojson)
    except Exception:
        return None, None, [], "Selected AOI"

    if poly.is_empty:
        return None, None, [], "Selected AOI"

    # 1) Centroid → nearest main city
    centroid = poly.centroid
    main_city, state_name, country = reverse_geocode_point(centroid.y, centroid.x)

    parts = [p for p in [main_city, state_name, country] if p]
    area_label = ", ".join(parts) if parts else "Selected AOI"

    # 2) Grid sampling within AOI to find all covering cities
    covering_cities = set()
    if main_city:
        covering_cities.add(main_city)

    minx, miny, maxx, maxy = poly.bounds
    if max_points < 1:
        max_points = 1
    n_side = max(1, int(math.sqrt(max_points)))
    step_x = (maxx - minx) / n_side if n_side > 0 else 0
    step_y = (maxy - miny) / n_side if n_side > 0 else 0

    for i in range(n_side):
        for j in range(n_side):
            x = minx + (i + 0.5) * step_x
            y = miny + (j + 0.5) * step_y
            pt = Point(x, y)
            if not poly.contains(pt):
                continue

            city, _, _ = reverse_geocode_point(y, x)
            if city:
                covering_cities.add(city)

    covering_cities_list = sorted(covering_cities)
    return main_city, state_name, covering_cities_list, area_label


# ------------------------------------------------------------------
# Layout: Map (left) + Analysis panel (right)
# ------------------------------------------------------------------
aoi_geojson = None

map_col, form_col = st.columns([5, 3], vertical_alignment="top")

with map_col:
    fmap = create_base_map()
    map_out = st_folium(
        fmap,
        height=420,  # shorter map so it does not push a huge white area below
        use_container_width=True,
        returned_objects=["all_drawings"],
        key="roi_map",
    )


    if map_out and isinstance(map_out.get("all_drawings"), list) and map_out["all_drawings"]:
        last_feat = map_out["all_drawings"][-1]
        if isinstance(last_feat, dict) and "geometry" in last_feat:
            aoi_geojson = last_feat["geometry"]

with form_col:
    st.markdown("#### 2 · Configure years and run analysis")
    # Wrap the form in a soft card for nicer UI
    st.markdown(
        "<div style='padding: 1rem; border-radius: 16px; background-color: #f9fafb; border: 1px solid #e5e7eb;'>",
        unsafe_allow_html=True,
    )

    with st.form(key="params_form", clear_on_submit=False):
        # Demo toggle (checkbox so text doesn't wrap vertically)
        demo_mode = st.checkbox(
            "Use Demo Mode (mock data, no GEE)",
            value=True,
            help="Uncheck this to fetch real WorldCover land-cover data from Google Earth Engine.",
        )
        st.caption(
            "Internally, TerraTrack uses ESA WorldCover 2020–2021 as reference years. "
            "You can still choose any baseline and future year below for forecasting."
        )

        # User-facing years: only two numbers
        # NOTE: Logic does not support pre-2020, so we clamp min and default to 2020.
        baseline_year = st.number_input(
            "Baseline year (starting point)",
            min_value=2020,
            max_value=2100,
            value=2020,
        )

        target_year = st.number_input(
            "Forecast year (future)",
            min_value=baseline_year + 1,
            max_value=2100,
            value=2040,
        )

        submitted = st.form_submit_button("Run Analysis 🚀", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def mock_stats(seed=0) -> dict:
    """Mock km² per class for Demo Mode."""
    random.seed(seed)
    base = {
        "Tree cover": random.uniform(60, 140),
        "Shrubland": random.uniform(5, 30),
        "Grassland": random.uniform(10, 40),
        "Cropland": random.uniform(25, 80),
        "Built-up": random.uniform(15, 50),
        "Bare / sparse": random.uniform(30, 100),
        "Snow/Ice": random.uniform(0, 5),
        "Permanent water": random.uniform(20, 90),
        "Herbaceous wetland": random.uniform(0, 10),
        "Mangroves": random.uniform(0, 5),
        "Moss/Lichen": random.uniform(0, 5),
    }
    return {k: round(v, 1) for k, v in base.items()}


def forecast_from_reference(
    baseline_year_user: int,
    target_year_user: int,
    ref_start_stats: dict,
    ref_latest_stats: dict,
    ref_gap_years: int = 1,
) -> dict:
    """
    Forecast future areas for each class using two reference years from GEE.

    We measure the per-year change between REF_START_YEAR and REF_LATEST_YEAR,
    then apply that rate from baseline_year_user to target_year_user.
    """
    horizon_years = target_year_user - baseline_year_user
    if ref_gap_years <= 0:
        ref_gap_years = 1

    proj = {}
    for cls in LC_CLASSES:
        a0 = float(ref_start_stats.get(cls, 0.0))
        a1 = float(ref_latest_stats.get(cls, 0.0))

        # per-year change measured between the two reference years
        slope_per_year = (a1 - a0) / float(ref_gap_years)

        # forecast from baseline_year_user to target_year_user
        proj_area = a0 + slope_per_year * float(horizon_years)
        proj[cls] = round(proj_area, 2)

    return proj


# ------------------------------------------------------------------
# Placeholders so we don't stack charts on rerun
# ------------------------------------------------------------------
progress_ph = st.empty()
notice_ph = st.empty()
chart_ph = st.empty()
table_ph = st.empty()


# ------------------------------------------------------------------
# Main analysis flow
# ------------------------------------------------------------------
if submitted:

    # If in real (GEE) mode, we must have an AOI
    if not demo_mode and aoi_geojson is None:
        st.warning("Please draw a polygon/rectangle AOI on the map before running analysis.")
        st.stop()

    # Short progress bar (just for UX)
    pbar = progress_ph.progress(0)
    for pct in range(0, 101, 10):
        time.sleep(0.02)
        pbar.progress(pct)
    progress_ph.empty()

    # Prepare AOI string for caching
    aoi_str = json.dumps(aoi_geojson, sort_keys=True) if aoi_geojson is not None else None

    # Location details defaults
    main_city = None
    state_name = None
    covering_cities: list[str] = []

    # 1) Get reference stats
    if demo_mode or aoi_str is None:
        ref_start_stats = mock_stats(seed=1)
        ref_latest_stats = mock_stats(seed=2)
        area_label = "Selected AOI"
    else:
        # Try real stats from GEE; if anything fails, fall back to demo mode
        try:
            ref_start_stats = get_class_areas_cached(aoi_str, year=REF_START_YEAR)
            ref_latest_stats = get_class_areas_cached(aoi_str, year=REF_LATEST_YEAR)

            # Derive detailed location info (nearest main city + covering cities)
            main_city, state_name, covering_cities, area_label = get_main_city_and_covering_cities(
                aoi_geojson
            )
        except Exception as e:
            st.error(
                "Google Earth Engine is not ready or this project is not fully configured yet.\n\n"
                f"Details: {e}\n\n"
                "Falling back to Demo Mode so you can continue testing the app."
            )
            ref_start_stats = mock_stats(seed=1)
            ref_latest_stats = mock_stats(seed=2)
            demo_mode = True
            area_label = "Selected AOI"
            main_city = None
            state_name = None
            covering_cities = []

    # 2) Forecast from reference years to user's target year
    proj_stats = forecast_from_reference(
        baseline_year_user=int(baseline_year),
        target_year_user=int(target_year),
        ref_start_stats=ref_start_stats,
        ref_latest_stats=ref_latest_stats,
        ref_gap_years=REF_YEAR_GAP,
    )

    notice_ph.success(
        "✅ Analysis completed using {}.".format(
            "Demo Mode (mock data)"
            if demo_mode
            else f"WorldCover {REF_START_YEAR}–{REF_LATEST_YEAR} as reference years"
        )
    )

    # Show AOI location details under the notice
    if main_city or covering_cities:
        lines = []
        if main_city and state_name:
            lines.append(f"Nearest main city: **{main_city}, {state_name}**")
        elif main_city:
            lines.append(f"Nearest main city: **{main_city}**")
        if covering_cities:
            lines.append("Cities inside AOI: " + ", ".join(covering_cities))
        st.markdown("  \n".join(lines))

    # 3) Build dataframe for charts – baseline & forecast only
    df = pd.DataFrame(
        {
            "Class": LC_CLASSES,
            f"{int(baseline_year)}_km²": [ref_start_stats.get(k, 0.0) for k in LC_CLASSES],
            f"{int(target_year)}_proj_km²": [proj_stats.get(k, 0.0) for k in LC_CLASSES],
        }
    )

    chart_ph.bar_chart(df.set_index("Class"))
    table_ph.dataframe(df, use_container_width=True)

    # 4) Save normalized structure for Auto Report page
    internal_latest_year = int(baseline_year) + 1

    stats_for_report = {}
    for cls in LC_CLASSES:
        start_val = float(ref_start_stats.get(cls, 0.0))
        latest_val = float(ref_latest_stats.get(cls, 0.0))
        proj_val = float(proj_stats.get(cls, 0.0))
        stats_for_report[cls] = {
            "start": start_val,
            "latest": latest_val,
            "proj": proj_val,
        }

    st.session_state["terratrack_stats"] = json.dumps(stats_for_report)
    st.session_state["terratrack_meta"] = {
        "area_label": area_label,
        "start_year": int(baseline_year),
        "latest_year": internal_latest_year,
        "target_year": int(target_year),
        "demo_mode": bool(demo_mode),
        "ref_start_year": REF_START_YEAR,
        "ref_latest_year": REF_LATEST_YEAR,
        "main_city": main_city,
        "state_name": state_name,
        "covering_cities": covering_cities,
    }

    st.info(f"Area: **{area_label}**  ·  Open **📄 Auto Report** to generate the summary & PDF.")

# On rerun without new submit: re-display last results (if any)
elif "terratrack_stats" in st.session_state:
    try:
        stats = json.loads(st.session_state["terratrack_stats"])
        meta = st.session_state.get("terratrack_meta", {})
        sy = meta.get("start_year", 2020)
        ty = meta.get("target_year", 2040)

        df = pd.DataFrame(
            {
                "Class": list(stats.keys()),
                f"{sy}_km²": [v["start"] for v in stats.values()],
                f"{ty}_proj_km²": [v["proj"] for v in stats.values()],
            }
        )
        chart_ph.bar_chart(df.set_index("Class"))
        table_ph.dataframe(df, use_container_width=True)
    except Exception:
        pass

