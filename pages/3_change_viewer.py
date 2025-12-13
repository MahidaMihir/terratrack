import numpy as np
from pathlib import Path

import rasterio
from PIL import Image
import streamlit as st
from ui_theme import inject_global_css

# Unified image size for consistent layout
IMAGE_WIDTH = 400
IMAGE_HEIGHT = 400


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "ml" / "data" / "raw" / "terratrack_training"
RAW_INPUT_PATTERN = "train_2020_tile_{}.tif"
RAW_TARGET_PATTERN = "train_2021_tile_{}.tif"

CHANGE_CLEAN_DIR = PROJECT_ROOT / "ml" / "data" / "change_from_seg_clean"


def stretch_contrast(img, low=0.02, high=0.30):
    img = np.clip((img - low) / (high - low), 0, 1)
    return img


@st.cache_data(show_spinner=False)
def load_rgb_from_tif(tif_path: Path) -> np.ndarray:
    with rasterio.open(tif_path) as src:
        red = src.read(3).astype("float32")
        green = src.read(2).astype("float32")
        blue = src.read(1).astype("float32")

    rgb = np.stack([red, green, blue], axis=-1)
    rgb = stretch_contrast(rgb, 0.02, 0.30)
    rgb = np.clip(rgb * 255, 0, 255).astype("uint8")
    return rgb


@st.cache_data(show_spinner=False)
def list_available_tiles():
    ids = []
    for f in sorted(CHANGE_CLEAN_DIR.glob("train_change_tile_*_mask.npy")):
        tid = (
            f.stem.replace("train_change_tile_", "")
            .replace("_mask", "")
        )
        ids.append(tid)
    return ids


@st.cache_data(show_spinner=False)
def load_mask(tile):
    f = CHANGE_CLEAN_DIR / f"train_change_tile_{tile}_mask.npy"
    if f.exists():
        return np.load(f).astype("uint8")
    return None


def make_overlay(rgb_target, mask, alpha):
    h, w, _ = rgb_target.shape
    base = Image.fromarray(rgb_target).convert("RGBA")
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[mask == 1] = (255, 0, 0, int(alpha * 255))
    overlay_img = Image.fromarray(overlay, mode="RGBA")
    combined = Image.alpha_composite(base, overlay_img)
    return combined.convert("RGB")


def dilate_mask(mask, iters=1):
    m = mask.astype(bool)
    for _ in range(iters):
        padded = np.pad(m, 1, mode="edge")
        m = (
            padded[:-2, :-2] | padded[:-2, 1:-1] | padded[:-2, 2:] |
            padded[1:-1, :-2] | padded[1:-1, 1:-1] | padded[1:-1, 2:] |
            padded[2:, :-2] | padded[2:, 1:-1] | padded[2:, 2:]
        )
    return m.astype("uint8")


def make_future(rgb_input, rgb_target, mask, growth):
    grown = dilate_mask(mask, growth)
    out = rgb_input.copy()
    out[grown == 1] = rgb_target[grown == 1]
    return Image.fromarray(out)


def resize_image(img: Image.Image):
    return img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.BILINEAR)


def main():
    st.set_page_config(page_title="Terratrack Viewer", layout="wide")
    inject_global_css()  # apply dark theme for this page

    st.title("Terratrack - Land Cover Change Viewer")
    st.markdown(
        "Explore land cover change between the **input year** and the **target year** "
        "for a selected area of interest."
    )

    # Sidebar
    with st.sidebar:
        st.header("Controls")

        tiles = list_available_tiles()
        tile_id = st.selectbox("Tile (area of interest)", tiles)

        alpha = st.slider("Overlay opacity", 0.2, 0.9, 0.5, 0.05)
        growth = st.slider("Future expansion level", 1, 4, 2)

        show_stats = st.checkbox("Show tile statistics", True)

    tif_input = RAW_DIR / RAW_INPUT_PATTERN.format(tile_id)
    tif_target = RAW_DIR / RAW_TARGET_PATTERN.format(tile_id)

    rgb_input = load_rgb_from_tif(tif_input)
    rgb_target = load_rgb_from_tif(tif_target)
    mask = load_mask(tile_id)

    overlay_img = make_overlay(rgb_target, mask, alpha)
    future_img = make_future(rgb_input, rgb_target, mask, growth)

    # Top layout: 3 equal images
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Observed pattern (overlay)")
        st.image(resize_image(overlay_img))

    with c2:
        st.subheader("Input year image")
        st.image(resize_image(Image.fromarray(rgb_input)))

    with c3:
        st.subheader("Target year image")
        st.image(resize_image(Image.fromarray(rgb_target)))

    # Bottom layout: future image (centered)
    st.markdown("---")
    st.subheader("Simulated satellite view for the target year")

    st.image(resize_image(future_img))

    # Statistics
    if show_stats:
        st.markdown("---")
        st.subheader("Tile statistics")

        total = mask.size
        changed = int((mask == 1).sum())
        pct = 100 * changed / total
        area = changed * 0.0001  # km² (10m pixels)

        m1, m2, m3 = st.columns(3)
        m1.metric("Changed pixels", changed)
        m2.metric("Change percent", f"{pct:.2f} %")
        m3.metric("Changed area", f"{area:.3f} km²")


if __name__ == "__main__":
    main()

st.markdown(
    "<script>window.addEventListener('load', () => { window.parent.postMessage({type: 'streamlit:setFrameHeight'}, '*') })</script>",
    unsafe_allow_html=True,
)