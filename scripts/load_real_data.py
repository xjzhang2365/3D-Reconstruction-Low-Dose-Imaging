"""
load_real_data.py
=================
Load a small window of TEM frames around the target and run Stage 1 preprocessing.

Folder layout expected
----------------------
3D-Reconstruction-Low-Dose-Imaging/   <- project root
    src/
        graphene3d/
            preprocessing/
                __init__.py
                corrections.py
                averaging.py
                denoising.py
    scripts/
        load_real_data.py              <- this file
    data/
        raw/
            raw_1.tif
            raw_2.tif
            ...
            raw_100.tif
        preprocessed/

Usage
-----
    python scripts/load_real_data.py
"""

import sys
import numpy as np
from pathlib import Path

# -- locate project root and add to sys.path ------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import tifffile
from graphene3d.preprocessing import preprocess_stack
from graphene3d.preprocessing.denoising import BM3DDenoiser


# -------------------------------------------------------------------------
# CONFIGURE THESE FOR YOUR DATA
# -------------------------------------------------------------------------

# Folder containing raw_XX.tif files
DATA_FOLDER = PROJECT_ROOT / "data" / "raw"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "preprocessed"

# The frame number you want to reconstruct (matches the number in filename)
# e.g. TARGET_FRAME = 21 means raw_21.tif
TARGET_FRAME = 21

# How many frames to average (must be odd; centred on TARGET_FRAME)
TEMPORAL_WINDOW = 5

# Preprocessing parameters (paper values)
FLAT_FIELD_SIGMA = 20.0   # Gaussian sigma for background estimation (pixels)
DEAD_PIXEL_SIGMA = 5.0    # Outlier threshold for dead-pixel detection (x std)


# -------------------------------------------------------------------------
# LOADERS
# -------------------------------------------------------------------------

def load_single_frame(tif_path) -> np.ndarray:
    """Load one TIFF frame as float64 array of shape (H, W)."""
    arr = tifffile.imread(str(tif_path)).astype(np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2-D frame, got shape {arr.shape} from {tif_path}"
        )
    return arr


def load_window(target_frame: int,
                window: int = TEMPORAL_WINDOW,
                folder=None,
                pattern: str = "raw_{}.tif") -> tuple:
    """Load only the frames in the temporal window around target_frame.

    For target_frame=21, window=5: loads raw_19.tif, raw_20.tif, raw_21.tif,
    raw_22.tif, raw_23.tif and returns them as a (5, H, W) stack.

    Parameters
    ----------
    target_frame : frame number to reconstruct (matches number in filename)
    window       : number of frames to load (must be odd)
    folder       : directory containing TIFF files (default: DATA_FOLDER)
    pattern      : filename pattern, {} is replaced by frame number

    Returns
    -------
    frames   : np.ndarray, shape (window, H, W), dtype float64
    target_idx : int, index of target_frame within the returned stack
                 (always window // 2 for a full window)
    """
    if folder is None:
        folder = DATA_FOLDER
    folder = Path(folder)

    if window % 2 == 0:
        raise ValueError(f"TEMPORAL_WINDOW must be odd, got {window}")

    half   = window // 2
    start  = target_frame - half   # e.g. 21 - 2 = 19
    end    = target_frame + half   # e.g. 21 + 2 = 23

    frame_numbers = list(range(start, end + 1))   # [19, 20, 21, 22, 23]
    target_idx    = half                           # index 2 in the window

    print(f"Loading {window}-frame window around raw_{target_frame}.tif")
    print(f"  Frames : {[f'raw_{n}.tif' for n in frame_numbers]}")

    frames = []
    for n in frame_numbers:
        path = folder / pattern.format(n)
        if not path.exists():
            raise FileNotFoundError(
                f"Frame not found: {path}\n"
                f"Check that raw_{n}.tif exists in {folder}"
            )
        frames.append(load_single_frame(path))

    stack = np.stack(frames)   # (window, H, W)
    print(f"Loaded stack : shape={stack.shape}  dtype=float64")
    _print_stats(stack)
    return stack, target_idx


# -------------------------------------------------------------------------
# PIPELINE ENTRY POINT
# -------------------------------------------------------------------------

def run_pipeline(frames: np.ndarray,
                 target_idx: int,
                 flat_field_sigma: float = FLAT_FIELD_SIGMA,
                 dead_pixel_sigma: float = DEAD_PIXEL_SIGMA) -> dict:
    """Run the complete Stage 1 preprocessing pipeline.

    Parameters
    ----------
    frames           : (N, H, W) float64 stack from load_window()
    target_idx       : index of the target frame within the stack
    flat_field_sigma : Gaussian sigma for background subtraction
    dead_pixel_sigma : outlier threshold for dead-pixel detection

    Returns
    -------
    dict with keys
        'corrected'  : (N, H, W) -- after flat-field + dead-pixel correction
        'denoised'   : (N, H, W) -- after BM3D denoising
        'averaged'   : (H, W)    -- final averaged frame -> input to Stage 2
        'dead_masks' : (N, H, W) -- bool, True where pixel was replaced
    """
    N = len(frames)
    if not (0 <= target_idx < N):
        raise ValueError(
            f"target_idx={target_idx} out of range for stack of {N} frames"
        )

    print(f"\nRunning Stage 1 pipeline")
    print(f"  target_idx       = {target_idx}  (frame {TARGET_FRAME})")
    print(f"  flat_field_sigma = {flat_field_sigma}")
    print(f"  dead_pixel_sigma = {dead_pixel_sigma}")
    print(f"  temporal_window  = {N}")

    result = preprocess_stack(
        frames,
        target_idx       = target_idx,
        flat_field_sigma = flat_field_sigma,
        dead_pixel_sigma = dead_pixel_sigma,
        denoiser         = BM3DDenoiser(),
        window_size      = N,
    )

    n_dead = int(result['dead_masks'].sum())
    print(f"\n  Dead pixels corrected : {n_dead}")
    print(f"  Output frame shape    : {result['averaged'].shape}")
    print(f"  Output intensity range: "
          f"[{result['averaged'].min():.4f}, {result['averaged'].max():.4f}]")
    print(f"\n  result['averaged'] is ready for Stage 2")

    return result


# -------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------

def _print_stats(frames: np.ndarray):
    flat = frames.ravel()
    std  = flat.std()
    snr  = flat.mean() / std if std > 0 else float('inf')
    print(f"  Intensity : min={flat.min():.0f}  max={flat.max():.0f}  "
          f"mean={flat.mean():.1f}  std={std:.1f}")
    print(f"  SNR approx {snr:.2f}  "
          f"{'(confirmed low-dose regime, SNR < 3)' if snr < 3 else '(higher dose)'}")


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

if __name__ == '__main__':

    print("=" * 60)
    print("  Stage 1 -- Preprocessing pipeline")
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Data folder  : {DATA_FOLDER}")
    print(f"  Target frame : raw_{TARGET_FRAME}.tif")
    print(f"  Window size  : {TEMPORAL_WINDOW} frames")
    print("=" * 60)

    # Load only the relevant window of frames
    frames, target_idx = load_window(
        target_frame = TARGET_FRAME,
        window       = TEMPORAL_WINDOW,
    )

    # Run the pipeline
    result = run_pipeline(frames, target_idx=target_idx)

    # Save the output for downstream Stage 2 initialization
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_FOLDER / f"preprocessed_frame{TARGET_FRAME}.npy"
    np.save(str(out_path), result['averaged'])
    print(f"\nSaved preprocessed frame -> {out_path}")
    print("Next step: pass this array into Stage 2 (atomic coordinate estimation).")
