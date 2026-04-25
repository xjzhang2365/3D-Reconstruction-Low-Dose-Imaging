"""
validate_stage2_against_ground_truth.py
=======================================
Validation harness for the integrated Stage 2 initialization pipeline.

This script compares Stage 2 output against the simulated ground-truth xyz.
It is diagnostic only: no SA/MD and no algorithm redesign.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
import tifffile
from scipy.spatial import KDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from benchmark_raw_pcd_z import load_pdb_xyz, xyz_angstrom_to_image_pixels
from graphene3d.stage2.pipeline import Stage2Config, run_stage2_initialization

# Prefer the preprocessed Stage 1 output; fall back to legacy simulated images.
def _resolve_image_path() -> Path:
    candidates = [
        REPO_ROOT / "outputs" / "preprocessing" / "preprocessed_frame2.tif",
        REPO_ROOT / "outputs" / "preprocessing" / "preprocessed_frame21.tif",
        REPO_ROOT / "data" / "simulated" / "img_noisy.tif",
        REPO_ROOT / "data" / "synthetic" / "img_noisy.tif",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No Stage 2 input image found. Run generate_simulated_dataset.py "
        "and run_preprocessing.py first."
    )

def _resolve_pdb_path() -> Path:
    candidates = [
        REPO_ROOT / "data" / "simulated" / "ground_xyz.pdb",
        REPO_ROOT / "data" / "synthetic" / "ground_xyz.pdb",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Ground-truth PDB not found in data/simulated/ or data/synthetic/.")

IMAGE_PATH = _resolve_image_path()
PDB_PATH = _resolve_pdb_path()


OUTPUT_DIR = REPO_ROOT / "outputs" / "stage2" / "validation"
DIAGNOSTIC_PATH = OUTPUT_DIR / "stage2_gt_diagnostic_overlay.png"
REPORT_PATH = OUTPUT_DIR / "stage2_gt_validation_report.txt"


def match_xy(pred_xy: np.ndarray,
             true_xy: np.ndarray,
             match_radius: float = 3.0) -> dict:
    """Greedy one-to-one xy matching within match_radius pixels."""
    pred = np.asarray(pred_xy, dtype=np.float64).reshape(-1, 2)
    true = np.asarray(true_xy, dtype=np.float64).reshape(-1, 2)

    if len(pred) == 0 or len(true) == 0:
        return {
            "matches": np.empty((0, 3), dtype=np.float64),
            "missing_true_idx": np.arange(len(true), dtype=int),
            "extra_pred_idx": np.arange(len(pred), dtype=int),
        }

    tree = KDTree(pred)
    candidates = []
    for true_idx, true_pos in enumerate(true):
        pred_idxs = tree.query_ball_point(true_pos, match_radius)
        for pred_idx in pred_idxs:
            dist = float(np.linalg.norm(true_pos - pred[pred_idx]))
            candidates.append((dist, pred_idx, true_idx))

    used_pred = set()
    used_true = set()
    matches = []
    for dist, pred_idx, true_idx in sorted(candidates):
        if pred_idx in used_pred or true_idx in used_true:
            continue
        used_pred.add(pred_idx)
        used_true.add(true_idx)
        matches.append((pred_idx, true_idx, dist))

    missing = np.array([idx for idx in range(len(true)) if idx not in used_true], dtype=int)
    extra = np.array([idx for idx in range(len(pred)) if idx not in used_pred], dtype=int)
    return {
        "matches": np.array(matches, dtype=np.float64),
        "missing_true_idx": missing,
        "extra_pred_idx": extra,
    }


def _centered_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """RMSE after removing mean offsets from matched z arrays."""
    valid = np.isfinite(pred) & np.isfinite(true)
    if not valid.any():
        return np.nan
    p = pred[valid] - np.mean(pred[valid])
    t = true[valid] - np.mean(true[valid])
    return float(np.sqrt(np.mean((p - t) ** 2)))


def _diagnose_missing(missing_xy: np.ndarray,
                      holes_xy: np.ndarray,
                      image_shape: tuple[int, int],
                      boundary_margin: float = 8.0,
                      hole_radius: float = 16.0) -> dict:
    """Heuristic diagnosis of where missing atoms sit."""
    if len(missing_xy) == 0:
        return {
            "near_boundary": 0,
            "low_hole_support": 0,
            "interior_with_hole_support": 0,
        }

    height, width = image_shape
    x = missing_xy[:, 0]
    y = missing_xy[:, 1]
    near_boundary_mask = (
        (x < boundary_margin) |
        (x > width - 1 - boundary_margin) |
        (y < boundary_margin) |
        (y > height - 1 - boundary_margin)
    )

    if len(holes_xy):
        hole_tree = KDTree(holes_xy)
        nearby_counts = np.array([
            len(hole_tree.query_ball_point(pos, hole_radius))
            for pos in missing_xy
        ])
    else:
        nearby_counts = np.zeros(len(missing_xy), dtype=int)

    low_hole_support_mask = nearby_counts < 3
    interior_with_hole_support = (~near_boundary_mask) & (~low_hole_support_mask)
    return {
        "near_boundary": int(np.sum(near_boundary_mask)),
        "low_hole_support": int(np.sum(low_hole_support_mask)),
        "interior_with_hole_support": int(np.sum(interior_with_hole_support)),
        "mean_nearby_holes_for_missing": float(np.mean(nearby_counts)),
    }


def save_diagnostic_overlay(image: np.ndarray,
                            pred_xy: np.ndarray,
                            true_xy: np.ndarray,
                            match_info: dict,
                            path: Path) -> None:
    """Save matched/missing/extra atom overlay."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    matches = match_info["matches"]
    missing_idx = match_info["missing_true_idx"]
    extra_idx = match_info["extra_pred_idx"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap="gray", interpolation="nearest")

    if len(matches):
        pred_match_idx = matches[:, 0].astype(int)
        true_match_idx = matches[:, 1].astype(int)
        ax.scatter(true_xy[true_match_idx, 0], true_xy[true_match_idx, 1],
                   s=18, facecolors="none", edgecolors="lime",
                   linewidths=0.8, label=f"Matched truth ({len(matches)})")
        ax.scatter(pred_xy[pred_match_idx, 0], pred_xy[pred_match_idx, 1],
                   s=8, c="cyan", marker=".", label="Matched prediction")

    if len(missing_idx):
        ax.scatter(true_xy[missing_idx, 0], true_xy[missing_idx, 1],
                   s=36, facecolors="none", edgecolors="red",
                   linewidths=1.1, label=f"Missing truth ({len(missing_idx)})")

    if len(extra_idx):
        ax.scatter(pred_xy[extra_idx, 0], pred_xy[extra_idx, 1],
                   s=22, c="orange", marker="x",
                   linewidths=0.9, label=f"Extra prediction ({len(extra_idx)})")

    ax.set_title("Stage 2 validation against simulated ground truth")
    ax.legend(fontsize=8, loc="upper right")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def format_report(metrics: dict, diagnosis: dict, output_paths: dict) -> str:
    """Create a plain-text validation report."""
    lines = [
        "Stage 2 validation against simulated ground truth",
        "=================================================",
        f"matched atom count : {metrics['matched']}",
        f"missing atom count : {metrics['missing']}",
        f"extra atom count   : {metrics['extra']}",
        f"x-y RMSE           : {metrics['xy_rmse']:.6f} px",
        f"raw z RMSE         : {metrics['z_raw_rmse']:.6f} Angstrom (centered)",
        f"smoothed z RMSE    : {metrics['z_smooth_rmse']:.6f} Angstrom (centered)",
        "",
        "Diagnosis",
        "---------",
        f"missing near boundary        : {diagnosis['near_boundary']}",
        f"missing with low hole support : {diagnosis['low_hole_support']}",
        f"interior missing with hole support: {diagnosis['interior_with_hole_support']}",
        f"mean nearby holes for missing: {diagnosis.get('mean_nearby_holes_for_missing', np.nan):.3f}",
        "",
        "Interpretation hints",
        "--------------------",
        "- Boundary missing atoms point toward border clipping or edge geometry.",
        "- Low hole support points toward hole detection as the bottleneck.",
        "- Interior missing atoms with hole support point toward triplet-generation rules or lack of MAP.",
        "",
        "Outputs",
        "-------",
    ]
    for key, value in output_paths.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def main() -> None:
    image = tifffile.imread(IMAGE_PATH).astype(np.float64)
    xyz_true, cell = load_pdb_xyz(PDB_PATH)
    true_xy = xyz_angstrom_to_image_pixels(xyz_true, cell, image.shape)
    true_z = xyz_true[:, 2]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = Stage2Config(
        output_prefix="stage2_validation_init",
        lowess_frac=0.08,
        verbose=True,
        save_outputs=True,
    )
    result = run_stage2_initialization(
        IMAGE_PATH,
        config=config,
        output_dir=OUTPUT_DIR,
    )

    match_info = match_xy(result.atom_xy_pixels, true_xy, match_radius=3.0)
    matches = match_info["matches"]
    pred_idx = matches[:, 0].astype(int) if len(matches) else np.array([], dtype=int)
    true_idx = matches[:, 1].astype(int) if len(matches) else np.array([], dtype=int)
    xy_errors = matches[:, 2] if len(matches) else np.array([], dtype=np.float64)

    metrics = {
        "matched": int(len(matches)),
        "missing": int(len(match_info["missing_true_idx"])),
        "extra": int(len(match_info["extra_pred_idx"])),
        "xy_rmse": float(np.sqrt(np.mean(xy_errors ** 2))) if len(xy_errors) else np.nan,
        "z_raw_rmse": _centered_rmse(result.z_raw[pred_idx], true_z[true_idx]),
        "z_smooth_rmse": _centered_rmse(result.z_smooth[pred_idx], true_z[true_idx]),
    }

    missing_xy = true_xy[match_info["missing_true_idx"]]
    diagnosis = _diagnose_missing(missing_xy, result.holes_xy, image.shape)
    save_diagnostic_overlay(image, result.atom_xy_pixels, true_xy,
                            match_info, DIAGNOSTIC_PATH)

    output_paths = {
        "diagnostic_overlay": str(DIAGNOSTIC_PATH),
        "validation_report": str(REPORT_PATH),
        **result.output_paths,
    }
    report = format_report(metrics, diagnosis, output_paths)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print("")
    print(report)


if __name__ == "__main__":
    main()
