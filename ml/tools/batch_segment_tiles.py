import argparse
from pathlib import Path
import subprocess
import sys


def run_command(cmd: list[str]):
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    print("Return code:", result.returncode)
    if result.returncode != 0:
        print("Command failed:", cmd)
        sys.exit(result.returncode)


def main():
    print(">>> batch_segment_tiles.py started")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Folder with .tif tiles")
    parser.add_argument("--output-dir", required=True, help="Folder to write masks")
    parser.add_argument("--year-tag", required=True, help="Tag: 2020 or 2021 (for naming only)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Input dir:", input_dir.resolve())
    print("Output dir:", output_dir.resolve())
    print("Year tag:", args.year_tag)

    # find all tif tiles
    pattern = f"train_{args.year_tag}_tile_*.tif"
    tif_files = sorted(input_dir.glob(pattern))
    print(f"Looking for tiles with pattern: {pattern}")
    print(f"Found {len(tif_files)} tiles")

    if not tif_files:
        print("No tiles found. Check folder path and filenames.")
        return

    for tif in tif_files:
        tile_id = tif.stem  # e.g. train_2020_tile_1
        out_prefix = output_dir / tile_id
        cmd = [
            sys.executable,
            "-m",
            "ml.tools.infer_seg_on_tif",
            "--input-tif",
            str(tif),
            "--output-prefix",
            str(out_prefix),
        ]
        run_command(cmd)

    print(">>> batch_segment_tiles.py finished")


if __name__ == "__main__":
    main()
