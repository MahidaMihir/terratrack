import argparse
from pathlib import Path
import numpy as np
import rasterio
from PIL import Image


def stretch_contrast(img, low=0.02, high=0.30):
    """
    Stretch reflectance image for better visualization.
    Any value <= low becomes 0
    Any value >= high becomes 1
    Linear scale in between.
    """
    img = np.clip((img - low) / (high - low), 0, 1)
    return img


def load_rgb_from_tif(tif_path: Path) -> np.ndarray:
    """
    Load Sentinel-style tile (B2,B3,B4,B8) and return bright RGB (H,W,3).
    """
    with rasterio.open(tif_path) as src:
        red = src.read(3).astype("float32")
        green = src.read(2).astype("float32")
        blue = src.read(1).astype("float32")

    rgb = np.stack([red, green, blue], axis=-1)

    # apply contrast stretching
    rgb = stretch_contrast(rgb, low=0.02, high=0.30)

    # convert to uint8
    rgb = np.clip(rgb * 255, 0, 255).astype("uint8")
    return rgb


def make_overlay(base_rgb: np.ndarray, change_mask: np.ndarray, alpha: float = 0.5) -> Image.Image:
    h, w, _ = base_rgb.shape
    if change_mask.shape != (h, w):
        raise ValueError("Mask/base image shapes do not match")

    base_img = Image.fromarray(base_rgb, mode="RGB").convert("RGBA")

    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[change_mask == 1] = (255, 0, 0, int(alpha * 255))

    overlay_img = Image.fromarray(overlay, mode="RGBA")
    combined = Image.alpha_composite(base_img, overlay_img)
    return combined.convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, help="Folder with original 2021 .tif tiles")
    parser.add_argument("--change-dir", required=True, help="Folder with cleaned change *_mask.npy")
    parser.add_argument("--out-dir", required=True, help="Folder to write overlays")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay opacity 0..1")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    change_dir = Path(args.change_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Raw dir     :", raw_dir.resolve())
    print("Change dir  :", change_dir.resolve())
    print("Output dir  :", out_dir.resolve())

    change_files = sorted(change_dir.glob("train_change_tile_*_mask.npy"))
    print("Found", len(change_files), "change masks")

    for f_change in change_files:
        tile_index = f_change.stem.replace("train_change_tile_", "").replace("_mask", "")
        tif_2021 = raw_dir / f"train_2021_tile_{tile_index}.tif"

        if not tif_2021.exists():
            print("Missing 2021 image for tile:", tile_index)
            continue

        print("Processing tile:", tile_index)

        base_rgb = load_rgb_from_tif(tif_2021)
        change_mask = np.load(f_change).astype("uint8")

        overlay = make_overlay(base_rgb, change_mask, alpha=args.alpha)

        out_png = out_dir / f"train_change_tile_{tile_index}_overlay.png"
        overlay.save(out_png)

        print("Saved overlay:", out_png.name)

    print("Done.")


if __name__ == "__main__":
    main()
