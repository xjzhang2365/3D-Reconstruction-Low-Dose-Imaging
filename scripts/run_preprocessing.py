"""Run the preprocessing workflow from the repository root.

This runner keeps generated preprocessing artifacts under outputs/preprocessing
and leaves src/ plus scripts/ as code-only folders.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

import numpy as np
import tifffile


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graphene3d.preprocessing.averaging import temporal_average
from graphene3d.preprocessing.corrections import (
    correct_flat_field_stack,
    remove_dead_pixels_stack,
)
from graphene3d.preprocessing.denoising import BM3DDenoiser


DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "simulated" / "raw_frames"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "preprocessing"


def _frame_id(path: Path) -> int:
    match = re.search(r"raw_(\d+)\.tif$", path.name)
    if not match:
        raise ValueError(f"Expected raw_<frame>.tif filename, got {path.name}")
    return int(match.group(1))


def _load_raw_stack(input_dir: Path) -> tuple[np.ndarray, list[int]]:
    paths = sorted(input_dir.glob("raw_*.tif"), key=_frame_id)
    if not paths:
        raise FileNotFoundError(f"No raw_*.tif files found in {input_dir}")

    frame_ids = [_frame_id(path) for path in paths]
    frames = [tifffile.imread(path).astype(np.float64) for path in paths]
    return np.stack(frames, axis=0), frame_ids


def run_preprocessing(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stack_raw, frame_ids = _load_raw_stack(Path(args.input_dir))
    if args.target_frame not in frame_ids:
        raise ValueError(
            f"Target frame {args.target_frame} was not found. "
            f"Available frame IDs include {frame_ids[:5]} ... {frame_ids[-5:]}"
        )

    target_idx = frame_ids.index(args.target_frame)
    scale = float(np.max(stack_raw))
    stack_norm = stack_raw / scale if scale > 0 else stack_raw

    corrected = correct_flat_field_stack(stack_norm, sigma=args.flat_sigma)
    cleaned, dead_masks = remove_dead_pixels_stack(
        corrected,
        threshold_sigma=args.dead_threshold_sigma,
    )
    averaged = temporal_average(
        cleaned,
        target_idx=target_idx,
        window_size=args.window_size,
    )

    if args.denoise == "bm3d":
        preprocessed = BM3DDenoiser().denoise(averaged)
    else:
        preprocessed = averaged

    stem = f"frame{args.target_frame}"
    np.save(output_dir / f"single_corrected_{stem}.npy", cleaned[target_idx])
    np.save(output_dir / f"dead_pixel_mask_{stem}.npy", dead_masks[target_idx])
    np.save(output_dir / f"averaged_{stem}.npy", averaged)
    np.save(output_dir / f"preprocessed_{stem}.npy", preprocessed)
    tifffile.imwrite(output_dir / f"preprocessed_{stem}.tif", preprocessed.astype(np.float32))

    print("Preprocessing complete")
    print(f"input frames      : {Path(args.input_dir)}")
    print(f"target frame      : raw_{args.target_frame}.tif")
    print(f"window size       : {args.window_size}")
    print(f"denoise           : {args.denoise}")
    print(f"output directory  : {output_dir}")
    print(f"stage2 input      : {output_dir / f'preprocessed_{stem}.tif'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-frame", type=int, default=21)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--flat-sigma", type=float, default=20.0)
    parser.add_argument("--dead-threshold-sigma", type=float, default=5.0)
    parser.add_argument("--denoise", choices=["bm3d", "none"], default="bm3d")
    return parser


def main() -> None:
    run_preprocessing(build_parser().parse_args())


if __name__ == "__main__":
    main()
