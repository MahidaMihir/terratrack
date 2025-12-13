# ml/tools/infer_seg_on_tif.py

import argparse
import os
import inspect

import numpy as np
import torch
import rasterio
from PIL import Image

from ml.models.unet import UNet  # adjust import if your UNet path or name differs


def load_model(
    checkpoint_path: str,
    in_channels: int = 4,
    num_classes: int = 6,
    device: str | None = None,
) -> torch.nn.Module:
    """
    Load UNet model and weights.

    Tries to adapt to different UNet.__init__ signatures.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Inspect UNet signature to understand which parameter names it expects
    sig = inspect.signature(UNet)
    params = sig.parameters

    # Build kwargs based on what the UNet constructor supports
    kwargs = {}

    # Handle channels parameter name
    if "in_channels" in params:
        kwargs["in_channels"] = in_channels
    elif "n_channels" in params:
        kwargs["n_channels"] = in_channels
    elif "channels" in params:
        kwargs["channels"] = in_channels

    # Handle classes parameter name
    if "n_classes" in params:
        kwargs["n_classes"] = num_classes
    elif "num_classes" in params:
        kwargs["num_classes"] = num_classes
    elif "classes" in params:
        kwargs["classes"] = num_classes

    # Fallback if nothing matched
    if not kwargs:
        raise RuntimeError(
            f"Could not map in_channels / num_classes to UNet constructor params: {list(params.keys())}. "
            f"Please update load_model to match your UNet.__init__ signature."
        )

    model = UNet(**kwargs)

    state = torch.load(checkpoint_path, map_location=device)
    # If you used a wrapper like {"model_state_dict": ...}, adapt this:
    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model



def normalize_input(x: np.ndarray) -> np.ndarray:
    """
    Normalization stub.

    x shape: (C, H, W), dtype usually float32 or uint16.

    Replace this with the same preprocessing used in ml/training/dataset_seg.py.
    For now, simple scaling if values look like Sentinel 2 reflectance (0 to 10000).
    """
    x = x.astype(np.float32)

    # Simple heuristic normalization
    # Option 1: scale to 0 to 1
    x = np.clip(x, 0.0, 1.0)

    # Option 2 (alternative): per band mean std normalization
    # x = (x - x.mean(axis=(1, 2), keepdims=True)) / (x.std(axis=(1, 2), keepdims=True) + 1e-6)

    return x


def tile_image(x: np.ndarray, tile_size: int = 256):
    """
    Split (C, H, W) array into tiles of size tile_size x tile_size.

    Handles non divisible sizes by padding at bottom and right edges.

    Returns:
        tiles: list of (c, tile_size, tile_size) arrays
        meta: list of (y, x) top left indices for reconstruction
        H_pad, W_pad: padded height and width
        H_orig, W_orig: original height and width
    """
    _, H, W = x.shape
    H_orig, W_orig = H, W

    # Compute padded sizes (next multiple of tile_size)
    H_pad = ((H + tile_size - 1) // tile_size) * tile_size
    W_pad = ((W + tile_size - 1) // tile_size) * tile_size

    if H_pad != H or W_pad != W:
        print(f"Padding image from ({H}, {W}) to ({H_pad}, {W_pad}) for tiling")
        pad_y = H_pad - H
        pad_x = W_pad - W

        # Pad only bottom and right
        x = np.pad(
            x,
            pad_width=((0, 0), (0, pad_y), (0, pad_x)),
            mode="constant",
            constant_values=0,
        )

    tiles = []
    meta = []
    for y in range(0, H_pad, tile_size):
        for x0 in range(0, W_pad, tile_size):
            patch = x[:, y : y + tile_size, x0 : x0 + tile_size]
            tiles.append(patch)
            meta.append((y, x0))

    return tiles, meta, H_pad, W_pad, H_orig, W_orig



def reconstruct_mask(
    tile_masks: list[np.ndarray],
    meta: list[tuple[int, int]],
    H_pad: int,
    W_pad: int,
    tile_size: int = 256,
) -> np.ndarray:
    """
    Reconstruct full padded mask (H_pad, W_pad) from per tile predictions.
    """
    full_mask = np.zeros((H_pad, W_pad), dtype=np.uint8)
    for patch_mask, (y, x0) in zip(tile_masks, meta):
        full_mask[y : y + tile_size, x0 : x0 + tile_size] = patch_mask
    return full_mask



def colorize_mask(mask: np.ndarray) -> Image.Image:
    """
    Map class indices to RGB colors and return a PIL Image.

    mask shape: (H, W), dtype int or uint8.
    """
    # Adjust these colors to match your class definitions
    palette = {
        0: (0, 0, 0),         # background
        1: (0, 0, 255),       # water
        2: (0, 128, 0),       # vegetation
        3: (255, 0, 0),       # built up
        4: (255, 255, 0),     # cropland
        5: (181, 101, 29),    # bare / barren
    }

    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in palette.items():
        rgb[mask == cls] = color

    return Image.fromarray(rgb, mode="RGB")


@torch.no_grad()
def run_inference_on_tif(
    input_tif: str,
    checkpoint_path: str,
    output_prefix: str,
    tile_size: int = 256,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure output directory exists (if prefix contains a directory)
    out_dir = os.path.dirname(output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 1. Load model
    model = load_model(checkpoint_path, in_channels=4, num_classes=6, device=device)


    # 2. Read GeoTIFF (assume 4 bands, first 4 are what you trained on)
    with rasterio.open(input_tif) as src:
        # Read first 4 bands. Adjust if your training used different bands.
        x = src.read(indexes=[1, 2, 3, 4])  # shape: (4, H, W)
        profile = src.profile

    # 3. Normalize
    x = normalize_input(x)  # (4, H, W)

    # 4. Tile into 256 x 256 patches (with padding if needed)
    tiles, meta, H_pad, W_pad, H_orig, W_orig = tile_image(x, tile_size=tile_size)

    # 5. Run inference tile by tile
    tile_masks = []
    for tile in tiles:
        # tile shape: (4, 256, 256)
        inp = torch.from_numpy(tile).unsqueeze(0).to(device)  # (1, C, H, W)
        logits = model(inp)
        # logits shape: (1, num_classes, H, W)
        pred = torch.argmax(logits, dim=1)  # (1, H, W)
        tile_masks.append(pred.squeeze(0).cpu().numpy().astype(np.uint8))

    # 6. Reconstruct full padded mask, then crop back to original size
    full_mask_padded = reconstruct_mask(
        tile_masks, meta, H_pad, W_pad, tile_size=tile_size
    )
    full_mask = full_mask_padded[:H_orig, :W_orig]


    # 7. Save raw mask as numpy and as GeoTIFF aligned with input (optional)
    npy_path = output_prefix + "_mask.npy"
    np.save(npy_path, full_mask)
    print(f"Saved raw mask to {npy_path}")

    # Optional: write mask GeoTIFF with same transform and crs
    gtiff_path = output_prefix + "_mask.tif"
    profile_mask = profile.copy()
    profile_mask.update(
        {
            "count": 1,
            "dtype": "uint8",
        }
    )
    with rasterio.open(gtiff_path, "w", **profile_mask) as dst:
        dst.write(full_mask, 1)
    print(f"Saved mask GeoTIFF to {gtiff_path}")

    # 8. Colorize mask and save PNG
    color_img = colorize_mask(full_mask)
    png_path = output_prefix + "_mask.png"
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    color_img.save(png_path)
    print(f"Saved colored PNG to {png_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run UNet segmentation on a GeoTIFF tile and save colored PNG."
    )
    parser.add_argument(
        "--input-tif", type=str, required=True, help="Path to input GeoTIFF tile."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ml/models/unet_seg_worldcover.pt",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Output prefix (without extension), e.g. outputs/demo_tile",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="Tile size expected by the model.",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device: cuda or cpu. Default: auto."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_inference_on_tif(
        input_tif=args.input_tif,
        checkpoint_path=args.checkpoint,
        output_prefix=args.output_prefix,
        tile_size=args.tile_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
