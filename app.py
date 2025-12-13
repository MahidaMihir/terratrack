import streamlit as st
from dotenv import load_dotenv

from ui_theme import inject_global_css


load_dotenv()

# --------------------------------------------------
# Page config (only once)
# --------------------------------------------------
st.set_page_config(
    page_title="TerraTrack AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global dark theme
inject_global_css()

# --------------------------------------------------
# Sidebar content
# --------------------------------------------------
st.sidebar.title("TerraTrack AI")
st.sidebar.markdown("Environmental Change Monitoring & Future Simulation 🌱")
st.sidebar.divider()
st.sidebar.page_link("pages/1_Map_and_Analysis.py", label="🗺️ Map and Analysis")
st.sidebar.page_link("pages/2_Auto_Report.py", label="📄 Auto Report")
st.sidebar.page_link("pages/3_change_viewer.py", label="🔁 Change viewer")

# --------------------------------------------------
# Top banner
# --------------------------------------------------
st.markdown(
    """
    <div style="
        text-align:center;
        background: linear-gradient(90deg,#0E7490,#06b6d4);
        padding: 1rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 5px 25px rgba(0,0,0,0.25);
    " class="pulse">
        <b>🌍 TerraTrack AI</b> - Environmental Change Monitoring with Future Simulation
    </div>
    """,
    unsafe_allow_html=True,
)


def main():
    # hero section
    st.markdown(
        """
        <h1 style="font-size: 2.4rem; margin-bottom: 0.2rem;">
            Welcome to TerraTrack AI
        </h1>
        <p style="font-size: 0.95rem; color: #cbd5f5; margin-bottom: 1.5rem;">
            TerraTrack AI is a satellite driven tool that measures land cover change between an input year and a target year,
            generates a plain language report, and provides a simulated future view of your area of interest.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Navigate to **🗺️ Map and Analysis** to begin analysis, "
        "or open **📄 Auto Report** to read the generated summary for your selected area."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.markdown(
            """
            <div class="tt-card">
              <div class="tt-card-title">How the workflow looks</div>
              <div class="tt-card-body">
                <ol style="padding-left: 1.2rem; margin-top: 0.3rem;">
                  <li>Draw an area of interest on the map.</li>
                  <li>Run the analysis to compute land cover for the input year and the target year.</li>
                  <li>Open the Auto Report page to get a detailed summary in simple language.</li>
                  <li>Open the Change Viewer page to see example tiles and a simulated satellite view.</li>
                </ol>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="tt-card">
              <div class="tt-card-title">Quick navigation</div>
              <div class="tt-card-body">
                <p style="margin-bottom: 0.6rem;">
                  <span class="tt-pill">Step 1</span> Map and Analysis<br/>
                  Draw the AOI, choose years, and compute statistics.
                </p>
                <p style="margin-bottom: 0.6rem;">
                  <span class="tt-pill">Step 2</span> Auto Report<br/>
                  Read the automatically generated summary, trends, and projections.
                </p>
                <p style="margin-bottom: 0.1rem;">
                  <span class="tt-pill">Step 3</span> Change Viewer<br/>
                  Browse sample tiles and view the simulated satellite image for the target year.
                </p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <hr style="border:none; height:1px; background:#1f2933; margin-top:2rem;">
        <p style="text-align:center; color:#9ca3af; font-size:13px;">
        © 2025 TerraTrack AI - Capstone Project, MIT-WPU Pune
        </p>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
