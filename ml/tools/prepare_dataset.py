# ml/tools/prepare_dataset.py

import os
from pathlib import Path
import numpy as np
import rasterio


# 1. Paths (edit SOURCE_DIR to match your project layout)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Folder where you moved all the exported GeoTIFF tiles
SOURCE_DIR = PROJECT_ROOT / "ml" / "data" / "raw" / "terratrack_training"

# Output folders
PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
VAL_DIR.mkdir(parents=True, exist_ok=True)


def find_tif_files(source_dir: Path):
    tif_files = sorted(source_dir.glob("*.tif"))
    if not tif_files:
        raise RuntimeError(f"No .tif files found in {source_dir}")
    return tif_files


def load_tile(path: Path):
    """Read GeoTIFF and split into X (4 bands) and y (label band)."""
    with rasterio.open(path) as src:
        # Rasterio returns (bands, height, width)
        arr = src.read()  # shape: (5, H, W) expected

    if arr.shape[0] < 5:
        raise ValueError(f"Expected 5 bands (B2,B3,B4,B8,LC), got {arr.shape[0]} for {path}")

    # First 4 bands: Sentinel 2 reflectance
    x = arr[:4, :, :].astype("float32")

    # Last band: land cover classes; convert back to integer labels
    lc = arr[4, :, :]
    y = np.round(lc).astype("int64")

    return x, y


def main(train_ratio: float = 0.8, seed: int = 42):
    tif_files = find_tif_files(SOURCE_DIR)
    print(f"Found {len(tif_files)} GeoTIFF tiles in {SOURCE_DIR}")

    # Shuffle with a fixed seed for reproducibility
    rng = np.random.RandomState(seed)
    indices = np.arange(len(tif_files))
    rng.shuffle(indices)

    split_idx = int(len(indices) * train_ratio)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    print(f"Using {len(train_idx)} tiles for TRAIN and {len(val_idx)} tiles for VAL")

    def save_subset(idxs, out_dir: Path, prefix: str):
        for count, i in enumerate(idxs, start=1):
            tif_path = tif_files[i]
            x, y = load_tile(tif_path)

            tile_id = f"{prefix}_{count:04d}"
            x_path = out_dir / f"{tile_id}_X.npy"
            y_path = out_dir / f"{tile_id}_y.npy"

            np.save(x_path, x)
            np.save(y_path, y)

            print(f"Saved {tile_id} from {tif_path.name}")

    save_subset(train_idx, TRAIN_DIR, "train")
    save_subset(val_idx, VAL_DIR, "val")

    print("Done. Numpy tiles written to:")
    print(f"  Train: {TRAIN_DIR}")
    print(f"  Val:   {VAL_DIR}")


if __name__ == "__main__":
    main()
