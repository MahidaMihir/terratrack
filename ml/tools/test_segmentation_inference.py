# ml/tools/test_segmentation_inference.py

from pathlib import Path
import numpy as np
import torch

from ml.models.unet import UNet


def main():
    project_root = Path(__file__).resolve().parents[2]
    ckpt_path = project_root / "ml" / "models" / "unet_seg_worldcover.pt"
    train_dir = project_root / "ml" / "data" / "processed" / "train"

    print("Project root:", project_root)
    print("Checkpoint:", ckpt_path)

    # pick first tile
    x_files = sorted(train_dir.glob("*_X.npy"))
    if not x_files:
        raise RuntimeError(f"No *_X.npy files found in {train_dir}")

    x_path = x_files[0]
    y_path = train_dir / x_path.name.replace("_X.npy", "_y.npy")

    print("Using tile:", x_path.name)

    x = np.load(x_path).astype("float32")  # (4, H, W)
    y = np.load(y_path).astype("int64")    # (H, W)

    # same preprocessing as in dataset_seg
    from ml.training.dataset_seg import center_crop_2d

    x = np.nan_to_num(x, nan=0.0)
    y = np.nan_to_num(y, nan=10)

    x = center_crop_2d(x, 256, 256)
    y = center_crop_2d(y, 256, 256)

    x = np.clip(x, 0.0, 1.0)

    # WorldCover IDs to indices
    num_classes = 6
    y_idx = (y // 10) - 1
    y_idx = np.clip(y_idx, -1, num_classes - 1)
    y_idx[y_idx < 0] = 0

    x_tensor = torch.from_numpy(x).unsqueeze(0)  # (1, 4, 256, 256)

    model = UNet(in_channels=4, num_classes=num_classes)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        logits = model(x_tensor)
        preds = logits.argmax(dim=1).squeeze(0).numpy()  # (256, 256)

    # simple stats
    print("Input tile shape:", x.shape)
    print("True labels (indices) min/max:", y_idx.min(), y_idx.max())
    unique_true, counts_true = np.unique(y_idx, return_counts=True)
    print("True label distribution:", dict(zip(unique_true.tolist(), counts_true.tolist())))

    unique_pred, counts_pred = np.unique(preds, return_counts=True)
    print("Pred label distribution:", dict(zip(unique_pred.tolist(), counts_pred.tolist())))

    pixel_acc = (preds == y_idx).mean()
    print("Pixel accuracy on this tile:", pixel_acc)


if __name__ == "__main__":
    main()
