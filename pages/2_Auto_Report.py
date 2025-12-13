import streamlit as st
import os
import json

from utils.pdf import write_pdf
from services.report import render_summary  # kept in case you use it somewhere
from ui_theme import inject_global_css  # new

# Apply global theme
inject_global_css()

st.title("📄 Auto Report")


# If there is no analysis data, stop early with a clear message
if "terratrack_stats" not in st.session_state:
    st.warning(
        "No analysis data found. Please open the Map and Analysis page, "
        "run the analysis for your AOI, then come back here."
    )
    st.stop()

st.markdown(
    """
    <style>
    .tt-card {
        padding: 1.3rem 1.5rem;
        border-radius: 0.9rem;
        background: #020617;
        border: 1px solid rgba(148, 163, 184, 0.35);
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.8);
        color: #e5e7eb;
    }
    .tt-card-title {
        font-size: 1.0rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #f9fafb;
    }
    .tt-card-body {
        font-size: 0.9rem;
        color: #e5e7eb;
    }
    .tt-pill {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        background: rgba(52, 211, 153, 0.12);
        color: #6ee7b7;
        border: 1px solid rgba(52, 211, 153, 0.4);
    }

    @keyframes typing { from {width:0} to {width:100%} }
    @keyframes blink { 50% { border-color: transparent } }
    .typewriter h4 {
      overflow: hidden;
      border-right: .15em solid #0E7490;
      white-space: nowrap;
      animation: typing 3s steps(30,end), blink .75s step-end infinite;
      font-size: 1.05rem;
      margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="typewriter"><h4>AI generated layman report for your selected area</h4></div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "This page turns the land cover statistics from the Map and Analysis page "
    "into a plain language summary and an exportable PDF report."
)


def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return 0.0


def build_detailed_report(
    area_label,
    start_year,
    latest_year,
    target_year,
    stats_json,
    main_city=None,
    state_name=None,
    covering_cities=None,
):
    covering_cities = covering_cities or []

    try:
        stats = json.loads(stats_json)
    except Exception:
        return "Could not parse statistics JSON. Please run the analysis on the Map page again."

    rows = []
    for cls_name, vals in stats.items():
        start_val = _safe_float(vals.get("start"))
        latest_val = _safe_float(vals.get("latest"))
        proj_val = _safe_float(vals.get("proj"))

        delta_obs = latest_val - start_val
        delta_future = proj_val - latest_val

        pct_obs = (delta_obs / start_val * 100.0) if start_val > 0 else None
        pct_future = (delta_future / latest_val * 100.0) if latest_val > 0 else None

        rows.append(
            {
                "class": cls_name,
                "start": start_val,
                "latest": latest_val,
                "proj": proj_val,
                "delta_obs": delta_obs,
                "delta_future": delta_future,
                "pct_obs": pct_obs,
                "pct_future": pct_future,
            }
        )

    if not rows:
        return "No land cover statistics were found. Please run the analysis on the Map page again."

    largest_obs_inc = max(rows, key=lambda r: r["delta_obs"])
    largest_obs_dec = min(rows, key=lambda r: r["delta_obs"])
    largest_future_inc = max(rows, key=lambda r: r["delta_future"])
    largest_future_dec = min(rows, key=lambda r: r["delta_future"])

    def fmt_delta(row, kind):
        if kind == "obs":
            d = row["delta_obs"]
            p = row["pct_obs"]
            period = f"{start_year} to {latest_year}"
        else:
            d = row["delta_future"]
            p = row["pct_future"]
            period = f"{latest_year} to {target_year}"
        if p is None:
            return f"{row['class']} changes by {d:.1f} km² over {period}"
        sign = "+" if d >= 0 else "-"
        direction = "increases" if d >= 0 else "decreases"
        return (
            f"{row['class']} {direction} by {abs(d):.1f} km² "
            f"({sign}{abs(p):.1f} percent) from {period}"
        )

    location_lines = []
    if main_city and state_name:
        location_lines.append(
            f"This analysis focuses on the region around {main_city}, {state_name} ({area_label})."
        )
    else:
        location_lines.append(
            f"This analysis focuses on the selected area: {area_label}."
        )
    if covering_cities:
        city_list = ", ".join(sorted(set(covering_cities)))
        location_lines.append(
            f"The area includes parts of the following administrative units: {city_list}."
        )

    method_lines = [
        "1. Data and method",
        f"   - Land cover information is derived from ESA WorldCover maps for {start_year} and {latest_year}, accessed via Google Earth Engine.",
        "   - For your selected polygon, the model computes the area in square kilometres of each land cover class (tree cover, cropland, built up, water, and others) for both years.",
        "   - The observed change per year is calculated from the difference between the two reference years.",
        f"   - A simple linear trend is then extended from {start_year} out to {target_year} to estimate future land cover areas.",
        "   - The numbers below come directly from these area calculations. No random values are introduced in this report.",
    ]

    snapshot_lines = ["2. Snapshot of land cover (area in km²)"]
    for r in rows:
        snapshot_lines.append(
            f"   - {r['class']}: {r['start']:.1f} km² in {start_year}, "
            f"{r['latest']:.1f} km² in {latest_year}, projected {r['proj']:.1f} km² by {target_year}."
        )

    observed_lines = [
        f"3. Observed changes between {start_year} and {latest_year}",
        f"   - Largest increase: {fmt_delta(largest_obs_inc, 'obs')}.",
        f"   - Largest decrease: {fmt_delta(largest_obs_dec, 'obs')}.",
    ]

    future_lines = [
        f"4. Projected changes towards {target_year}",
        f"   - Strongest projected increase: {fmt_delta(largest_future_inc, 'future')}.",
        f"   - Strongest projected decrease: {fmt_delta(largest_future_dec, 'future')}.",
        "   - These projections assume that the recent trend continues at a similar pace. They are not a climate or socio economic scenario, only an extension of recent behaviour.",
    ]

    recommendation_lines = ["5. Planning and management hints"]

    def has_increase(keyword):
        for r in rows:
            if keyword.lower() in r["class"].lower() and (
                r["delta_obs"] > 0 or r["delta_future"] > 0
            ):
                return True
        return False

    def has_decrease(keyword):
        for r in rows:
            if keyword.lower() in r["class"].lower() and (
                r["delta_obs"] < 0 or r["delta_future"] < 0
            ):
                return True
        return False

    if has_increase("Built"):
        recommendation_lines.append(
            "   - Built up surfaces are increasing. Consider guiding new development towards existing urban cores, protecting key green spaces, and reserving corridors for drainage and drainage."
        )
    if has_decrease("Tree") or has_decrease("Forest"):
        recommendation_lines.append(
            "   - Tree covered areas are under pressure. Targeted tree planting, roadside and riparian buffers, and protection of remaining patches can help maintain shade, biodiversity, and local cooling."
        )
    if has_decrease("water") or has_decrease("Permanent water"):
        recommendation_lines.append(
            "   - Water bodies are shrinking. Safeguard lake and river buffers, reduce hard paving near channels, and improve stormwater capture and recharge structures."
        )
    if has_increase("Cropland"):
        recommendation_lines.append(
            "   - Cropland expansion is visible. Promoting soil conserving practices and checking conversion of sensitive wetlands or forest patches can reduce long term risk."
        )

    if len(recommendation_lines) == 1:
        recommendation_lines.append(
            "   - Overall changes are modest. Continuing to monitor land cover every few years will help detect emerging risks early."
        )

    explanation_lines = [
        "6. How to read this report",
        "   - Every number in this report is derived from the computed land cover areas inside your drawn polygon.",
        "   - Observed changes describe the difference between the two reference years.",
        "   - Projected values simply extend that recent change rate into the future target year.",
        "   - Suggested actions are rule of thumb recommendations linked to which land cover classes are increasing or decreasing. They support planning but do not replace detailed field studies.",
    ]

    sections = [
        "Location and coverage",
        "---------------------",
        *location_lines,
        "",
        *method_lines,
        "",
        *snapshot_lines,
        "",
        *observed_lines,
        "",
        *future_lines,
        "",
        *recommendation_lines,
        "",
        *explanation_lines,
    ]
    return "\n".join(sections)


# ------------------------------------------------------------------
# Read meta and stats from session state
# ------------------------------------------------------------------
meta = st.session_state.get("terratrack_meta", {})
stats_json = st.session_state.get("terratrack_stats", "{}")

area_label_default = meta.get("area_label", "Selected area")
main_city = meta.get("main_city")
state_name = meta.get("state_name")
covering_cities = meta.get("covering_cities", [])

existing_summary = st.session_state.get("terratrack_summary", "")

# ------------------------------------------------------------------
# Layout: left controls card, right explanation card
# ------------------------------------------------------------------
left_col, right_col = st.columns([1.4, 1.0])

with left_col:
    st.markdown('<div class="tt-card">', unsafe_allow_html=True)
    st.markdown('<div class="tt-card-title">Configure report</div>', unsafe_allow_html=True)
    st.markdown('<div class="tt-card-body">', unsafe_allow_html=True)

    area_label = st.text_input("Area label", value=area_label_default)

    if main_city or covering_cities:
        details_lines = []
        if main_city and state_name:
            details_lines.append(f"Nearest main city: {main_city}, {state_name}")
        elif main_city:
            details_lines.append(f"Nearest main city: {main_city}")
        if covering_cities:
            unique_cities = ", ".join(sorted(set(covering_cities)))
            details_lines.append(f"Units inside AOI: {unique_cities}")
        st.markdown("  \n".join(details_lines))

    c1, c2, c3 = st.columns(3)
    with c1:
        start_year_default = meta.get("start_year", 2020)
        start_year = st.number_input("Starting year", 2000, 2100, start_year_default)
    with c2:
        latest_year_default = meta.get("latest_year", start_year_default + 1)
        latest_year = st.number_input("Latest year", 2000, 2100, latest_year_default)
    with c3:
        target_year_default = meta.get("target_year", latest_year_default + 20)
        target_year = st.number_input("Projected year", 2000, 2100, target_year_default)

    st.text_area(
        "Stats JSON (auto filled)",
        value=stats_json,
        key="stats_json_area",
        height=180,
    )

    if st.button("Generate summary ✨", use_container_width=True):
        if not st.session_state.get("stats_json_area", "").strip():
            st.warning("No stats found in the box above. Please run analysis on the Map page again.")
        else:
            with st.spinner("Drafting layman summary..."):
                txt = build_detailed_report(
                    area_label=area_label,
                    start_year=int(start_year),
                    latest_year=int(latest_year),
                    target_year=int(target_year),
                    stats_json=st.session_state["stats_json_area"],
                    main_city=main_city,
                    state_name=state_name,
                    covering_cities=covering_cities,
                )
                st.session_state["terratrack_summary"] = txt
                existing_summary = txt

    if st.button("Export PDF 📄", use_container_width=True):
        if "terratrack_summary" not in st.session_state:
            st.warning("Please generate the summary first.")
        else:
            pdf_path = write_pdf(
                area_label,
                int(start_year),
                int(latest_year),
                int(target_year),
                st.session_state["terratrack_summary"],
            )
            st.success(f"PDF generated: {os.path.basename(pdf_path)}")
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "⬇️ Download report",
                    f,
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf",
                )

    st.markdown("</div></div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="tt-card">', unsafe_allow_html=True)
    st.markdown('<div class="tt-card-title">What this report tells you</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="tt-card-body">
        <p><span class="tt-pill">For presentation</span></p>
        <ul style="padding-left:1.2rem; margin-top:0.4rem;">
          <li>Summarises how land cover in your selected AOI has changed between the input year and target year.</li>
          <li>Explains which land cover types increased or decreased the most.</li>
          <li>Extends the recent trend to the target year to give a simple future projection.</li>
          <li>Provides rule of thumb planning hints you can mention during viva.</li>
        </ul>
        <p style="font-size:0.85rem; color:#9ca3af; margin-top:0.6rem;">
          Tip: During the demo you can say that all numbers come directly from satellite based area calculations,
          so the text is grounded in real data, not random AI text.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.subheader("Generated summary")

st.text_area(
    "AI generated layman summary",
    value=existing_summary,
    height=420,
    key="terratrack_summary_display",
)
