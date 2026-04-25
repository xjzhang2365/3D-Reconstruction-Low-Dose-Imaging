"""
Parameter sweep to tune generate_simulated_dataset.py for Stage 2 compatibility.

Each combination generates 5 noisy frames in a temp directory, runs the full
Stage 1 preprocessing pipeline (flat-field + dead-pixel + temporal avg + BM3D),
runs Stage 2 initialization, and evaluates against the ground-truth PDB.

Target: ~630 atoms detected, >600 matched, xy RMSE < 1.0 px (paper regime).

Usage:
    python scripts/tune_simulated_dataset.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile
from scipy.spatial import KDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

PDB_PATH = REPO_ROOT / "data" / "simulated" / "ground_xyz.pdb"
PIXEL_SIZE_ANG = 0.183
N_FRAMES = 5
SEED = 42

# ---------------------------------------------------------------------------
# Sweep configurations: (label, image_shape, defocus_ang, dose, shading_amp)
# ---------------------------------------------------------------------------
CONFIGS = [
    # Baseline: current (wrong) shape
    ("baseline_256x256",   (256, 256), -80,  8000, 0.4),
    # Primary fix: correct shape matching PDB cell / pixel_size
    ("shape_fix_242x262",  (242, 262), -80,  8000, 0.4),
    # Shape fixed, reduce shading artifact
    ("shape_shade0.2",     (242, 262), -80,  8000, 0.2),
    ("shape_shade0.1",     (242, 262), -80,  8000, 0.1),
    # Shape fixed, sweep defocus
    ("defocus_-50",        (242, 262), -50,  8000, 0.2),
    ("defocus_-150",       (242, 262), -150, 8000, 0.2),
    ("defocus_-200",       (242, 262), -200, 8000, 0.2),
    # Shape fixed, sweep dose
    ("dose_4000",          (242, 262), -80,  4000, 0.2),
    ("dose_12000",         (242, 262), -80, 12000, 0.2),
]


# ---------------------------------------------------------------------------
# PDB / ground-truth helpers (duplicated from benchmark_raw_pcd_z for
# self-containedness)
# ---------------------------------------------------------------------------

def _load_pdb(pdb_path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    xyz, cell = [], None
    with open(pdb_path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("CRYST1"):
                cell = (float(line[6:15]), float(line[15:24]), float(line[24:33]))
            elif line.startswith(("ATOM", "HETATM")):
                xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(xyz, dtype=np.float64), cell


def _pdb_to_pixels(
    xyz: np.ndarray,
    cell: tuple[float, float, float],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Map PDB Angstrom coordinates to pixel (x, y) given image shape."""
    H, W = image_shape
    cx, cy, _ = cell
    x_pix = xyz[:, 0] / (cx / W)
    y_pix = (cy - xyz[:, 1]) / (cy / H)
    return np.column_stack([x_pix, y_pix])


def _match_xy(pred_xy, true_xy, radius=3.0):
    if len(pred_xy) == 0 or len(true_xy) == 0:
        return 0, len(true_xy), len(pred_xy), np.nan
    tree = KDTree(pred_xy)
    used_p, used_t = set(), set()
    matched_dists = []
    cands = []
    for ti, tp in enumerate(true_xy):
        for pi in tree.query_ball_point(tp, radius):
            cands.append((float(np.linalg.norm(tp - pred_xy[pi])), pi, ti))
    for dist, pi, ti in sorted(cands):
        if pi in used_p or ti in used_t:
            continue
        used_p.add(pi)
        used_t.add(ti)
        matched_dists.append(dist)
    n_match = len(matched_dists)
    n_miss = len(true_xy) - n_match
    n_extra = len(pred_xy) - n_match
    rmse = float(np.sqrt(np.mean(np.array(matched_dists) ** 2))) if matched_dists else np.nan
    return n_match, n_miss, n_extra, rmse


# ---------------------------------------------------------------------------
# Frame generation helpers
# ---------------------------------------------------------------------------

def _generate_frames(
    positions: np.ndarray,
    image_shape: tuple[int, int],
    defocus_ang: float,
    dose: float,
    shading_amp: float,
    rng: np.random.Generator,
    sim_cache: dict,
) -> list[np.ndarray]:
    """Generate N_FRAMES noisy frames and return as list of float32 arrays."""
    key = (image_shape, defocus_ang)
    if key not in sim_cache:
        from graphene3d.stage3.pcd_ctf_simulator import AbtemSimulator
        sim_cache[key] = AbtemSimulator(
            pixel_size_ang=PIXEL_SIZE_ANG,
            image_shape=image_shape,
            defocus_ang=defocus_ang,
            slice_thickness=1.0,
            dose=dose,
            add_noise=False,
        )
    sim = sim_cache[key]
    H, W = image_shape

    frames = []
    for _ in range(N_FRAMES):
        thermal = rng.normal(0.0, 0.05, positions.shape)
        pos_frame = (positions + thermal).astype(np.float32)
        clean = sim.simulate(pos_frame)

        # Flat-field shading
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        shading = (1.0 - shading_amp / 2.0) + shading_amp * np.exp(
            -((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (2 * (W * 0.31) ** 2)
        )
        tilt_x = rng.uniform(-0.01, 0.01)
        tilt_y = rng.uniform(-0.01, 0.01)
        shading = shading * (1.0 + tilt_x * (xx / W - 0.5) + tilt_y * (yy / H - 0.5))
        shaded = (clean * shading).astype(np.float32)

        # Poisson noise
        counts = np.maximum(shaded, 0) * dose * PIXEL_SIZE_ANG ** 2
        noisy = rng.poisson(counts).astype(np.float32)

        # Dead pixels
        n_dead = int(rng.integers(2, 4))
        noisy[rng.integers(0, H, n_dead), rng.integers(0, W, n_dead)] = 0.0
        frames.append(noisy)
    return frames


# ---------------------------------------------------------------------------
# Preprocessing helpers (inline, no file I/O)
# ---------------------------------------------------------------------------

def _preprocess_stack(frames: list[np.ndarray], target_idx: int) -> np.ndarray:
    """Flat-field correct, dead-pixel clean, temporal average, BM3D denoise."""
    from graphene3d.preprocessing.corrections import (
        correct_flat_field_stack,
        remove_dead_pixels_stack,
    )
    from graphene3d.preprocessing.averaging import temporal_average
    from graphene3d.preprocessing.denoising import BM3DDenoiser

    stack = np.stack(frames, axis=0).astype(np.float64)
    scale = stack.max()
    stack_norm = stack / scale if scale > 0 else stack

    corrected = correct_flat_field_stack(stack_norm, sigma=20.0)
    cleaned, _ = remove_dead_pixels_stack(corrected, threshold_sigma=5.0)
    averaged = temporal_average(cleaned, target_idx=target_idx, window_size=N_FRAMES)
    denoised = BM3DDenoiser().denoise(averaged)
    return denoised


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep() -> list[dict]:
    from graphene3d.stage2.pipeline import Stage2Config, run_stage2_initialization

    positions, cell = _load_pdb(PDB_PATH)
    n_truth = len(positions)
    rng_master = np.random.default_rng(SEED)
    sim_cache: dict = {}

    results = []
    for cfg in CONFIGS:
        label, image_shape, defocus, dose, shading_amp = cfg
        print(f"\n{'='*60}")
        print(f"Config: {label}")
        print(f"  shape={image_shape}, defocus={defocus} A, dose={dose}, shading={shading_amp}")

        rng = np.random.default_rng(SEED)   # same seed every config for fair comparison

        print("  Generating frames...", end=" ", flush=True)
        frames = _generate_frames(positions, image_shape, defocus, dose, shading_amp, rng, sim_cache)
        print("done")

        print("  Preprocessing...", end=" ", flush=True)
        preprocessed = _preprocess_stack(frames, target_idx=2)
        print("done")

        print("  Running Stage 2...", end=" ", flush=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config = Stage2Config(
                output_prefix="tune",
                verbose=False,
                save_outputs=False,
            )
            result = run_stage2_initialization(preprocessed, config=config, output_dir=tmp)
        print(f"done  ({len(result.atom_xy_pixels)} atoms)")

        true_xy = _pdb_to_pixels(positions, cell, image_shape)
        n_match, n_miss, n_extra, xy_rmse = _match_xy(result.atom_xy_pixels, true_xy)

        row = {
            "label": label,
            "image_shape": image_shape,
            "defocus": defocus,
            "dose": dose,
            "shading": shading_amp,
            "n_detected": len(result.atom_xy_pixels),
            "n_matched": n_match,
            "n_missing": n_miss,
            "n_extra": n_extra,
            "xy_rmse_px": xy_rmse,
        }
        results.append(row)
        print(
            f"  RESULT: detected={row['n_detected']}  matched={n_match}/{n_truth}"
            f"  missing={n_miss}  extra={n_extra}  xy_rmse={xy_rmse:.3f} px"
        )

    return results


def print_table(results: list[dict]) -> None:
    header = (
        f"{'Config':<22} {'Shape':>10} {'Def':>6} {'Dose':>6} {'Shd':>4} "
        f"{'Det':>5} {'Match':>6} {'Miss':>5} {'Extra':>6} {'RMSE':>6}"
    )
    print("\n" + "=" * len(header))
    print("PARAMETER SWEEP RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        rmse_str = f"{r['xy_rmse_px']:.3f}" if np.isfinite(r["xy_rmse_px"]) else "  nan"
        print(
            f"{r['label']:<22} {str(r['image_shape']):>10} {r['defocus']:>6} "
            f"{r['dose']:>6} {r['shading']:>4} "
            f"{r['n_detected']:>5} {r['n_matched']:>6} {r['n_missing']:>5} "
            f"{r['n_extra']:>6} {rmse_str:>6}"
        )
    print("=" * len(header))

    best = min(
        (r for r in results if np.isfinite(r["xy_rmse_px"])),
        key=lambda r: (r["xy_rmse_px"], r["n_missing"]),
    )
    print(f"\nBest config by xy RMSE: {best['label']}")
    print(f"  image_shape={best['image_shape']}, defocus={best['defocus']} A, "
          f"dose={best['dose']}, shading={best['shading']}")
    print(f"  detected={best['n_detected']}, matched={best['n_matched']}, "
          f"missing={best['n_missing']}, extra={best['n_extra']}, "
          f"xy_rmse={best['xy_rmse_px']:.3f} px")


if __name__ == "__main__":
    results = run_sweep()
    print_table(results)
