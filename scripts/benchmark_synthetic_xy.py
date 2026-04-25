"""
benchmark_synthetic_xy.py
=========================
Minimal synthetic-only benchmark for Stage 2 x-y initialization.

This script checks only the bright-hole -> atom-xy initialization path:
  1. generate a synthetic graphene-like image with ground-truth holes/atoms
  2. detect bright hole centres
  3. infer atom x-y candidates from hole triplets
  4. evaluate predicted atom x-y against synthetic ground truth
  5. save a predicted-vs-ground-truth overlay

It intentionally does not run MAP, PCD, LOWESS, or downstream optimization.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
from scipy.spatial import KDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graphene3d.stage2.find_xy import (
    build_atom_positions_from_holes,
    evaluate_xy_predictions,
    plot_xy_comparison,
)
from graphene3d.stage2.hole_finding import find_holes


OUTPUT_DIR = REPO_ROOT / "outputs" / "stage2" / "synthetic_xy"
OVERLAY_CLEAN_PATH = OUTPUT_DIR / "synthetic_xy_overlay_clean_gt_holes.png"
OVERLAY_DETECTED_PATH = OUTPUT_DIR / "synthetic_xy_overlay_detected_holes.png"


def make_hex_image(n_rings: int = 5,
                   a: float = 14.0,
                   img_size: int = 160,
                   noise: float = 0.02,
                   seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic graphene-like image.

    Contrast convention:
      - hole centres are bright
      - atom columns are dark
    """
    img = np.ones((img_size, img_size), dtype=np.float64) * 0.5

    hole_positions = []
    for i in range(n_rings + 2):
        for j in range(n_rings + 2):
            x = i * a + (j % 2) * (a / 2) + 20
            y = j * a * np.sqrt(3) / 2 + 20
            if 5 < x < img_size - 5 and 5 < y < img_size - 5:
                hole_positions.append([x, y])

    for hx, hy in hole_positions:
        xi, yi = int(round(hx)), int(round(hy))
        if 2 <= xi < img_size - 2 and 2 <= yi < img_size - 2:
            img[yi - 1:yi + 2, xi - 1:xi + 2] += 0.35

    atom_positions = []
    holes = np.array(hole_positions, dtype=np.float64)
    tree = KDTree(holes)
    seen = set()
    for i, hi in enumerate(holes):
        nbrs = tree.query_ball_point(hi, a * 1.2)
        nbrs = [n for n in nbrs if n != i]
        for j, k in combinations(nbrs, 2):
            key = tuple(sorted([i, j, k]))
            if key in seen:
                continue
            if np.linalg.norm(holes[j] - holes[k]) > a * 1.2:
                continue
            seen.add(key)
            centroid = (holes[i] + holes[j] + holes[k]) / 3.0
            ax, ay = centroid
            if 3 < ax < img_size - 3 and 3 < ay < img_size - 3:
                atom_positions.append([ax, ay])
                xi, yi = int(round(ax)), int(round(ay))
                if 2 <= xi < img_size - 2 and 2 <= yi < img_size - 2:
                    img[yi - 1:yi + 2, xi - 1:xi + 2] -= 0.25

    rng = np.random.default_rng(seed)
    img += rng.normal(0, noise, img.shape)
    img = np.clip(img, 0.0, 1.0)

    return img, holes, np.array(atom_positions, dtype=np.float64)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    img, true_holes, true_atoms = make_hex_image()

    detected_holes, hole_scores = find_holes(img, sigma=2.0, return_scores=True)
    predicted_atoms_clean = build_atom_positions_from_holes(
        true_holes,
        nn_radius=16.0,
        max_side_ratio=1.35,
        duplicate_radius=1.5,
    )
    predicted_atoms_detected = build_atom_positions_from_holes(
        detected_holes,
        nn_radius=16.0,
        max_side_ratio=1.35,
        duplicate_radius=1.5,
    )

    metrics_clean = evaluate_xy_predictions(
        predicted_atoms_clean,
        true_atoms,
        match_radius=3.0,
    )
    metrics_holes = evaluate_xy_predictions(
        detected_holes,
        true_holes,
        match_radius=3.0,
    )
    metrics_detected = evaluate_xy_predictions(
        predicted_atoms_detected,
        true_atoms,
        match_radius=3.0,
    )

    plot_xy_comparison(
        img,
        predicted_atoms_clean,
        true_atoms,
        save_path=str(OVERLAY_CLEAN_PATH),
        match_radius=3.0,
        title="Synthetic xy from ground-truth holes",
    )
    plot_xy_comparison(
        img,
        predicted_atoms_detected,
        true_atoms,
        save_path=str(OVERLAY_DETECTED_PATH),
        match_radius=3.0,
        title="Synthetic xy from detected holes",
    )

    print("Synthetic x-y benchmark")
    print("=======================")
    print(f"true holes       : {len(true_holes)}")
    print(f"detected holes   : {len(detected_holes)}")
    print(f"hole score range : [{hole_scores.min():.4f}, {hole_scores.max():.4f}]"
          if len(hole_scores) else "hole score range : []")
    print(f"true atoms       : {len(true_atoms)}")
    print(f"clean overlay    : {OVERLAY_CLEAN_PATH}")
    print(f"detected overlay : {OVERLAY_DETECTED_PATH}")
    print("")
    print("Clean find_xy path: ground-truth holes -> atom xy")
    print(f"predicted atoms  : {len(predicted_atoms_clean)}")
    print(f"tp               : {metrics_clean['tp']}")
    print(f"fp               : {metrics_clean['fp']}")
    print(f"fn               : {metrics_clean['fn']}")
    print(f"precision        : {metrics_clean['precision']:.4f}")
    print(f"recall           : {metrics_clean['recall']:.4f}")
    print(f"RMSE             : {metrics_clean['rmse']:.4f} px")
    print(f"mean error       : {metrics_clean['mean_error']:.4f} px")
    print("")
    print("Hole detection diagnostic")
    print(f"tp               : {metrics_holes['tp']}")
    print(f"fp               : {metrics_holes['fp']}")
    print(f"fn               : {metrics_holes['fn']}")
    print(f"precision        : {metrics_holes['precision']:.4f}")
    print(f"recall           : {metrics_holes['recall']:.4f}")
    print(f"RMSE             : {metrics_holes['rmse']:.4f} px")
    print(f"mean error       : {metrics_holes['mean_error']:.4f} px")
    print("")
    print("End-to-end diagnostic: detected holes -> atom xy")
    print(f"predicted atoms  : {len(predicted_atoms_detected)}")
    print(f"tp               : {metrics_detected['tp']}")
    print(f"fp               : {metrics_detected['fp']}")
    print(f"fn               : {metrics_detected['fn']}")
    print(f"precision        : {metrics_detected['precision']:.4f}")
    print(f"recall           : {metrics_detected['recall']:.4f}")
    print(f"RMSE             : {metrics_detected['rmse']:.4f} px")
    print(f"mean error       : {metrics_detected['mean_error']:.4f} px")


if __name__ == "__main__":
    main()
