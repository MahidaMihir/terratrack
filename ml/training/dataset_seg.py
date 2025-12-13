# ml/training/dataset_seg.py

from pathlib import Path
from typing import Tuple, List, Union

import numpy as np
import torch
from torch.utils.data import Dataset


def center_crop_2d(arr: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    """
    Center crop a 2D array (H, W) or 3D array (C, H, W) to crop_h x crop_w.
    """
    if arr.ndim == 2:
        h, w = arr.shape
    elif arr.ndim == 3:
        _, h, w = arr.shape
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")

    if h < crop_h or w < crop_w:
        raise ValueError(f"Cannot crop {h}x{w} to {crop_h}x{crop_w}")

    top = (h - crop_h) // 2
    left = (w - crop_w) // 2

    if arr.ndim == 2:
        return arr[top : top + crop_h, left : left + crop_w]
    else:
        return arr[:, top : top + crop_h, left : left + crop_w]


class LandCoverSegmentationDataset(Dataset):
    """
    Loads X/y NumPy tiles for segmentation.

    Assumptions:
      - X files end with '_X.npy'
      - y files end with '_y.npy'
      - X shape is (4, H, W)
      - y shape is (H, W) with WorldCover codes 10,20,30,40,50,60
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        crop_size: int = 256,
        normalize: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.crop_size = crop_size
        self.normalize = normalize

        self.x_files: List[Path] = sorted(self.root_dir.glob("*_X.npy"))
        if not self.x_files:
            raise RuntimeError(f"No *_X.npy files found in {self.root_dir}")

        self.y_files: List[Path] = []
        for xf in self.x_files:
            yf = self.root_dir / xf.name.replace("_X.npy", "_y.npy")
            if not yf.exists():
                raise RuntimeError(f"Missing label file for {xf.name}")
            self.y_files.append(yf)

        # You checked earlier: labels are 10,20,30,40,50,60 -> 6 classes
        self.num_classes = 6

    def __len__(self) -> int:
        return len(self.x_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = np.load(self.x_files[idx]).astype("float32")  # (4, H, W)
        y = np.load(self.y_files[idx]).astype("int64")    # (H, W)

        # Center crop to 256x256
        x = center_crop_2d(x, self.crop_size, self.crop_size)
        y = center_crop_2d(y, self.crop_size, self.crop_size)

        # Inputs already scaled to ~0..1 in GEE; just clip to be safe
        if self.normalize:
            x = np.clip(x, 0.0, 1.0)

        # WorldCover codes 10,20,30,40,50,60 -> indices 0..5
        y_idx = (y // 10) - 1
        y_idx = np.clip(y_idx, 0, self.num_classes - 1)

        x_tensor = torch.from_numpy(x)                 # (4, 256, 256), float32
        y_tensor = torch.from_numpy(y_idx.astype("int64"))  # (256, 256), int64

        return x_tensor, y_tensor
