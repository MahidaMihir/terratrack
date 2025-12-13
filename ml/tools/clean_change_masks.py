import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def clean_mask(mask: np.ndarray, min_size: int = 50, hole_size: int = 50) -> np.ndarray:
    """
    Clean a binary change mask.

    mask: 0 = no change, 1 = change

    min_size  : remove connected red blobs smaller than this (in pixels)
    hole_size : fill white holes inside red blobs smaller than this
    """
    # ensure boolean
    m = mask.astype(bool)

    # 1. remove small red components
    labeled, num = ndi.label(m)
    # sizes[i] is the size of component with label i+1
    sizes = ndi.sum(m, labeled, index=range(1, num + 1))

    m_clean = m.copy()
    for label, size in enumerate(sizes, start=1):
        if size < min_size:
            m_clean[labeled == label] = False

    # 2. fill small white holes inside red regions
    inv = ~m_clean
    labeled_inv, num_inv = ndi.label(inv)
    sizes_inv = ndi.sum(inv, labeled_inv, index=range(1, num_inv + 1))

    for label, size in enumerate(sizes_inv, start=1):
        if size < hole_size:
            # this small white region becomes red
            m_clean[labeled_inv == label] = True

    return m_clean.astype("uint8")


def mask_to_png(mask: np.ndarray) -> Image.Image:
    """
    Convert binary mask to RGB image.
    White = no change, red = change.
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[mask == 0] = (255, 255, 255)
    rgb[mask == 1] = (255, 0, 0)
    return Image.fromarray(rgb, mode="RGB")


def main():
    print(">>> clean_change_masks.py started")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        required=True,
        help="Folder with raw change *_mask.npy from compute_change_from_segmentation",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Folder to write cleaned masks",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=50,
        help="Remove red blobs smaller than this many pixels",
    )
    parser.add_argument(
        "--hole-size",
        type=int,
        default=50,
        help="Fill white holes smaller than this many pixels",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Input dir :", in_dir.resolve())
    print("Output dir:", out_dir.resolve())
    print("min_size  :", args.min_size)
    print("hole_size :", args.hole_size)

    files = sorted(in_dir.glob("train_change_tile_*_mask.npy"))
    print("Found", len(files), "change masks")

    if not files:
        print("No change masks found. Check in_dir path.")
        return

    for f in files:
        print("Processing", f.name)
        mask = np.load(f)
        cleaned = clean_mask(mask, min_size=args.min_size, hole_size=args.hole_size)

        # save npy
        out_npy = out_dir / f.name
        np.save(out_npy, cleaned)

        # save png
        out_png = out_dir / f.name.replace(".npy", ".png")
        img = mask_to_png(cleaned)
        img.save(out_png)

        print("Saved:", out_npy.name, "and", out_png.name)

    print(">>> clean_change_masks.py finished")


if __name__ == "__main__":
    main()
