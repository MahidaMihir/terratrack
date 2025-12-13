import streamlit as st

def inject_global_css() -> None:
    """Apply a clean blue theme similar to earlier TerraTrack UI."""
    st.markdown(
        """
        <style>
        /* Remove Streamlit system UI, but keep header so sidebar toggle works */
        #MainMenu, footer {visibility: hidden;}

        /* Style the header to match background (no white bar) */
        header[data-testid="stHeader"] {
            background: #0b1e39 !important;   /* same as app background */
            color: #e8eef7 !important;
            border-bottom: none !important;
        }
        header[data-testid="stHeader"] > div {
            margin-bottom: 0rem;
        }

        /* Light navy app background */
        .stApp {
            background: #0b1e39 !important;
            color: #e8eef7;
        }

        /* Center container */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }

        /* Slight card effect for main area */
        .main > div {
            background: rgba(255, 255, 255, 0.04) !important;
            padding: 1rem 2rem;
            border-radius: 10px;
            border: 1px solid rgba(170, 200, 255, 0.15);
        }

        /* Headings */
        h1, h2, h3, h4 {
            color: #d9e7ff !important;
            font-weight: 700;
            letter-spacing: -0.01em;
        }

        /* Text */
        p, span, div, label, li {
            color: #e6edf7 !important;
            font-size: 0.95rem;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: #102544 !important;
            border-right: 1px solid rgba(120, 160, 220, 0.25);
        }
        section[data-testid="stSidebar"] * {
            color: #dde8f9 !important;
        }

        /* Sidebar active item */
        section[data-testid="stSidebar"] .stSidebarNav [aria-selected="true"] {
            background-color: rgba(80, 130, 200, 0.35) !important;
            border-radius: 8px !important;
            color: white !important;
        }

        /* Buttons */
        .stButton>button, button[kind="primary"] {
            background: linear-gradient(90deg, #3482f6, #4da3ff) !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.45rem 1.3rem;
            font-weight: 600;
            color: white !important;
        }

        /* Inputs */
        .stTextInput>div>div>input,
        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] > div {
            background: #e8eef7 !important;
            color: #0b1e39 !important;
            border-radius: 8px !important;
            border: 1px solid #aac1e8 !important;
        }

        /* Checkbox label color */
        div[data-testid="stCheckbox"] label {
            color: #e8eef7 !important;
        }

        /* Generic card used in app.py */
        .tt-card {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(120, 160, 220, 0.35);
            padding: 1.3rem;
            border-radius: 12px;
            color: #e6edf7 !important;
        }
        .tt-card-title {
            font-size: 1.05rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #d9e7ff;
        }
        .tt-card-body {
            font-size: 0.9rem;
        }

        /* Progress bar blue */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg,#3482f6,#4da3ff) !important;
        }

        /* Slider styling */
        .stSlider > div > div > div {
            background: #3482f6 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
