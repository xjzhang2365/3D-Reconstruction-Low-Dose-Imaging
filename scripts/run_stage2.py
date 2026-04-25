"""Run Stage 2 initialization from the packaged source layout."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graphene3d.stage2.pipeline import Stage2Config, run_stage2_initialization


DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "stage2"

# Preferred candidates in priority order: simulated frame first, then legacy fallbacks.
_IMAGE_CANDIDATES = [
    REPO_ROOT / "outputs" / "preprocessing" / "preprocessed_frame2.tif",
    REPO_ROOT / "outputs" / "preprocessing" / "preprocessed_frame21.tif",
    REPO_ROOT / "data" / "preprocessed" / "preprocessed_frame21.npy",
    REPO_ROOT / "data" / "synthetic" / "img_noisy.tif",
]


def _choose_default_image() -> Path:
    for path in _IMAGE_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("No default Stage 2 image found. Run generate_simulated_dataset.py and run_preprocessing.py first.")


def _load_stage2_image(path: Path):
    if path.suffix.lower() == ".npy":
        return np.load(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix", default="stage2_initialization")
    args = parser.parse_args()

    image_path = args.image if args.image is not None else _choose_default_image()
    config = Stage2Config(output_prefix=args.prefix)
    run_stage2_initialization(
        _load_stage2_image(image_path),
        config=config,
        output_dir=args.output_dir,
    )
    print(f"Stage 2 input      : {image_path}")
    print(f"Stage 2 output dir : {args.output_dir}")


if __name__ == "__main__":
    main()
