import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def colorize_change(mask: np.ndarray) -> Image.Image:
    """
    mask: 0 = no change, 1 = change
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # white = no change, red = change
    rgb[mask == 0] = (255, 255, 255)
    rgb[mask == 1] = (255, 0, 0)

    return Image.fromarray(rgb, mode="RGB")


def main():
    print(">>> compute_change_from_segmentation.py started")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preds-2020",
        required=True,
        help="Folder with 2020 *_mask.npy files",
    )
    parser.add_argument(
        "--preds-2021",
        required=True,
        help="Folder with 2021 *_mask.npy files",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Folder to write change masks (npy + png)",
    )
    args = parser.parse_args()

    dir_2020 = Path(args.preds_2020)
    dir_2021 = Path(args.preds_2021)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("2020 dir:", dir_2020.resolve())
    print("2021 dir:", dir_2021.resolve())
    print("Output dir:", out_dir.resolve())

    files_2020 = sorted(dir_2020.glob("train_2020_tile_*_mask.npy"))
    print("Found", len(files_2020), "2020 mask files")

    if not files_2020:
        print("No 2020 mask files found. Check folder path and filenames.")
        return

    for f20 in files_2020:
        tile_id = f20.name.replace("_mask.npy", "")            # train_2020_tile_XX
        tile_index = tile_id.replace("train_2020_tile_", "")   # XX

        f21 = dir_2021 / f"train_2021_tile_{tile_index}_mask.npy"
        if not f21.exists():
            print("Missing 2021 mask for tile", tile_index, "->", f21.name)
            continue

        print("Processing tile", tile_index)

        m20 = np.load(f20)
        m21 = np.load(f21)

        if m20.shape != m21.shape:
            print("Shape mismatch for tile", tile_index, m20.shape, m21.shape)
            continue

        change = (m20 != m21).astype("uint8")  # 0=no change, 1=change

        out_npy = out_dir / f"train_change_tile_{tile_index}_mask.npy"
        np.save(out_npy, change)

        img = colorize_change(change)
        out_png = out_dir / f"train_change_tile_{tile_index}_mask.png"
        img.save(out_png)

        print("Saved:", out_npy.name, "and", out_png.name)

    print(">>> compute_change_from_segmentation.py finished")


if __name__ == "__main__":
    main()
