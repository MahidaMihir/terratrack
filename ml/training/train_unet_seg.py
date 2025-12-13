# ml/training/train_unet_seg.py

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

from ml.training.dataset_seg import LandCoverSegmentationDataset
from ml.models.unet import UNet


def train_unet_seg(
    project_root: str | Path,
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 1e-3,
    device_str: str = "cuda",
):
    project_root = Path(project_root)
    train_dir = project_root / "ml" / "data" / "processed" / "train"
    val_dir = project_root / "ml" / "data" / "processed" / "val"

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # datasets and loaders
    train_ds = LandCoverSegmentationDataset(train_dir, crop_size=256)
    val_ds = LandCoverSegmentationDataset(val_dir, crop_size=256)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    in_channels = 4
    num_classes = train_ds.num_classes

    model = UNet(in_channels=in_channels, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    ckpt_dir = project_root / "ml" / "models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "unet_seg_worldcover.pt"

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # ---------- Train ----------
        model.train()
        running_loss = 0.0

        for x, y in tqdm(train_loader, desc="Train", ncols=80):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)  # (B, C, H, W)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_ds)

        # ---------- Validation ----------
        model.eval()
        val_loss = 0.0
        total_correct = 0
        total_pixels = 0

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Val  ", ncols=80):
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)

                preds = logits.argmax(dim=1)
                total_correct += (preds == y).sum().item()
                total_pixels += y.numel()

        val_loss /= len(val_ds)
        pix_acc = total_correct / max(1, total_pixels)

        print(
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val pixel acc: {pix_acc:.4f}"
        )

        # Save checkpoint each epoch
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict()},
            ckpt_path,
        )
        print(f"Checkpoint saved to {ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    print("Starting U-Net segmentation training with project root:", project_root)
    train_unet_seg(project_root)
