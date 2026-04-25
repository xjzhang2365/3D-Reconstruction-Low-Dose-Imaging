"""
Generate a fully reproducible simulated TEM dataset for demonstrating the
end-to-end reconstruction pipeline. All outputs are derived from code alone —
no proprietary experimental data is used.

The script produces five simulated TEM frames from the bundled calibrated
clean-image base (data/synthetic/image_clean.tif, an abTEM simulation of the
640-atom graphene ground truth at 0.183 A/px and 80 kV). Each frame has
fresh Poisson shot noise at the calibrated dose, multiplicative flat-field
shading, and 2-3 dead pixels, giving Stage 1 real preprocessing work.

The image shape (242, 262) is fixed by the PDB cell / pixel size:
    width  = round(47.946 A / 0.183 A/px) = 262 px
    height = round(44.286 A / 0.183 A/px) = 242 px
This ensures ground-truth PDB positions map correctly to pixel coordinates.

Parameters tuned to reproduce paper-regime contrast on simulated data.
See scripts/tune_simulated_dataset.py for the parameter sweep that selected
these values.

Outputs
-------
data/simulated/raw_frames/raw_0.tif  ... raw_4.tif  -- five noisy frames
data/simulated/clean_reference.npy                   -- noise-free reference
data/ground_truth/graphene_640.csv                   -- ground-truth xy, z (A)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

PDB_PATH = REPO_ROOT / "data" / "simulated" / "ground_xyz.pdb"
CLEAN_BASE_PATH = REPO_ROOT / "data" / "synthetic" / "image_clean.tif"
RAW_FRAMES_DIR = REPO_ROOT / "data" / "simulated" / "raw_frames"
CLEAN_REF_PATH = REPO_ROOT / "data" / "simulated" / "clean_reference.npy"
GT_DIR = REPO_ROOT / "data" / "ground_truth"
GT_CSV_PATH = GT_DIR / "graphene_640.csv"

# Parameters tuned to reproduce paper-regime contrast on simulated data.
# IMAGE_SHAPE is derived from PDB cell (47.946 x 44.286 A) / pixel_size (0.183 A/px).
# DOSE_SCALE is chosen so Poisson noise matches the bundled synthetic reference
# (804 counts/pixel mean, std ~34). This corresponds to an effective dose of
# ~24 000 e-/A^2, higher than the paper's 8 000 e-/A^2, to give Stage 2
# sufficient SNR for hole detection.
IMAGE_SHAPE = (242, 262)   # (height, width) in pixels
PIXEL_SIZE_ANG = 0.183
DOSE_SCALE = 804.0         # electrons per pixel (effective dose ~24 000 e-/A^2)
SHADING_AMP = 0.1          # flat-field shading amplitude (0 = none, 0.4 = strong)
N_FRAMES = 5
TARGET_FRAME = 2
SEED = 42


def _load_pdb(pdb_path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    xyz, cell = [], None
    with open(pdb_path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("CRYST1"):
                cell = (float(line[6:15]), float(line[15:24]), float(line[24:33]))
            elif line.startswith(("ATOM", "HETATM")):
                xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if cell is None or not xyz:
        raise ValueError(f"Invalid PDB: {pdb_path}")
    return np.array(xyz, dtype=np.float64), cell


def _load_clean_base(path: Path) -> np.ndarray:
    """Load the calibrated noise-free abTEM simulation used as the frame base."""
    img = tifffile.imread(path).astype(np.float64)
    if img.shape != IMAGE_SHAPE:
        raise ValueError(
            f"Clean base image shape {img.shape} does not match "
            f"IMAGE_SHAPE {IMAGE_SHAPE}. Check CLEAN_BASE_PATH."
        )
    return img


def _flat_field_shading(rng: np.random.Generator) -> np.ndarray:
    H, W = IMAGE_SHAPE
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    shading = (1.0 - SHADING_AMP / 2.0) + SHADING_AMP * np.exp(
        -((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (2 * (W * 0.31) ** 2)
    )
    tilt_x = rng.uniform(-0.01, 0.01)
    tilt_y = rng.uniform(-0.01, 0.01)
    shading *= 1.0 + tilt_x * (xx / W - 0.5) + tilt_y * (yy / H - 0.5)
    return shading.astype(np.float64)


def main() -> None:
    print("=== generate_simulated_dataset.py ===")
    print(f"Loading ground truth from: {PDB_PATH}")
    positions, cell = _load_pdb(PDB_PATH)
    n_atoms = len(positions)
    print(f"Loaded {n_atoms} atoms  cell={cell[0]:.3f} x {cell[1]:.3f} x {cell[2]:.3f} A")
    print(f"Image shape: {IMAGE_SHAPE}  pixel size: {PIXEL_SIZE_ANG} A/px")

    print(f"\nLoading clean base image from: {CLEAN_BASE_PATH}")
    clean = _load_clean_base(CLEAN_BASE_PATH)
    print(f"Clean image: shape={clean.shape}, mean={clean.mean():.4f}, std={clean.std():.4f}")

    # Save noise-free reference
    np.save(CLEAN_REF_PATH, clean.astype(np.float32))
    print(f"Saved: {CLEAN_REF_PATH}")

    # Save ground-truth CSV (nominal positions, no thermal displacement)
    GT_DIR.mkdir(parents=True, exist_ok=True)
    np.savetxt(GT_CSV_PATH, positions, delimiter=",", header="x_ang,y_ang,z_ang", comments="")
    print(f"Saved: {GT_CSV_PATH}  ({n_atoms} rows)")

    # Generate N_FRAMES noisy frames
    RAW_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    H, W = IMAGE_SHAPE

    print(f"\nGenerating {N_FRAMES} noisy frames -> {RAW_FRAMES_DIR}")
    print(f"  dose scale: {DOSE_SCALE:.0f} counts/pixel  shading amplitude: {SHADING_AMP}")
    for frame_idx in range(N_FRAMES):
        # Tiny per-frame thermal perturbation on clean image values
        # (sigma = 0.1 % of dynamic range, sub-dominant vs Poisson noise)
        thermal_noise = rng.normal(0.0, clean.std() * 0.001, clean.shape)
        clean_frame = np.clip(clean + thermal_noise, 0, None)

        # Flat-field shading artifact (brighter at centre than edges)
        if SHADING_AMP > 0:
            shading = _flat_field_shading(rng)
            counts = clean_frame * DOSE_SCALE * shading
        else:
            counts = clean_frame * DOSE_SCALE

        # Poisson shot noise
        noisy = rng.poisson(np.maximum(counts, 0)).astype(np.float32)

        # 2-3 dead pixels (value = 0)
        n_dead = int(rng.integers(2, 4))
        noisy[rng.integers(0, H, n_dead), rng.integers(0, W, n_dead)] = 0.0

        out_path = RAW_FRAMES_DIR / f"raw_{frame_idx}.tif"
        tifffile.imwrite(out_path, noisy)
        print(
            f"  Frame {frame_idx}: range=[{noisy.min():.1f}, {noisy.max():.1f}]"
            f"  mean={noisy.mean():.1f}  std={noisy.std():.1f}  -> {out_path.name}"
        )

    print("\nDone. Quick-start pipeline commands:")
    print("  python scripts/run_preprocessing.py --target-frame 2")
    print("  python scripts/run_stage2.py")
    print("  python scripts/validate_stage2_against_ground_truth.py")


if __name__ == "__main__":
    main()
