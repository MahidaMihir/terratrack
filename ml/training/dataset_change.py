from pathlib import Path
from typing import Union, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset_seg import center_crop_2d   # reuse your existing function


class LandCoverChangeDataset(Dataset):
    """
    Dataset for binary land cover change detection.
    
    Expects files:
        *_2020_X.npy  -> shape (4,H,W)
        *_2021_X.npy  -> shape (4,H,W)
        *_y.npy       -> shape (H,W) label (0=no change, 1=change)
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

        # load all *_2020_X.npy files
        self.x2020_files = sorted(self.root_dir.glob("*_2020_X.npy"))
        if not self.x2020_files:
            raise RuntimeError(f"No *_2020_X.npy files found in {self.root_dir}")

        self.x2021_files = []
        self.y_files = []

        for xf in self.x2020_files:
            base = xf.name.replace("_2020_X.npy", "")

            f2021 = self.root_dir / f"{base}_2021_X.npy"
            fy = self.root_dir / f"{base}_y.npy"

            if not f2021.exists():
                raise RuntimeError(f"Missing file: {f2021}")
            if not fy.exists():
                raise RuntimeError(f"Missing label file: {fy}")

            self.x2021_files.append(f2021)
            self.y_files.append(fy)

    def __len__(self) -> int:
        return len(self.x2020_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x2020 = np.load(self.x2020_files[idx]).astype("float32")
        x2021 = np.load(self.x2021_files[idx]).astype("float32")
        y = np.load(self.y_files[idx]).astype("int64")

        # Center crop to consistent size
        x2020 = center_crop_2d(x2020, self.crop_size, self.crop_size)
        x2021 = center_crop_2d(x2021, self.crop_size, self.crop_size)
        y = center_crop_2d(y, self.crop_size, self.crop_size)

        # Inputs are already scaled 0..1 from GEE (same as segmentation training)
        if self.normalize:
            x2020 = np.clip(x2020, 0.0, 1.0)
            x2021 = np.clip(x2021, 0.0, 1.0)

        # Prepare tensors
        # Stack along channel dimension -> (8,256,256)
        x = np.concatenate([x2020, x2021], axis=0)
        x_tensor = torch.from_numpy(x)  # float32

        y_tensor = torch.from_numpy(y.astype("int64"))  # 0 or 1

        return x_tensor, y_tensor
