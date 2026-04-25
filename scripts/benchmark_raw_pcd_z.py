"""
benchmark_raw_pcd_z.py
======================
Raw PCD z-initialization benchmark for the real simulated TEM case.

This runner benchmarks:
  - raw unsmoothed PCD z on the preprocessed Stage 2 image
  - a diagnostic LOWESS span sweep applied after raw PCD
  - no MAP
  - no x-y refinement/tuning

The atom x-y inputs come from the ground-truth PDB so the benchmark isolates
the z-initialization behavior.  Bright hole centres are detected from the noisy
preprocessed simulated TEM image and used as the local PCD reference intensity.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graphene3d.stage2.hole_finding import find_holes
from graphene3d.stage2.pcd_z import estimate_z_pcd, lowess_smooth_z


SYNTHETIC_DIR = ROOT / "data" / "synthetic"
IMAGE_PATH = SYNTHETIC_DIR / "img_noisy.tif"
PDB_PATH = SYNTHETIC_DIR / "ground_xyz.pdb"
OUTPUT_DIR = ROOT / "outputs" / "stage2" / "raw_pcd_z"

SCATTER_PATH = OUTPUT_DIR / "raw_pcd_z_pred_vs_truth.png"
HIST_PATH = OUTPUT_DIR / "raw_pcd_z_histogram.png"
SCATTER3D_PATH = OUTPUT_DIR / "raw_pcd_z_3d_scatter.png"
LOWESS_SWEEP_PATH = OUTPUT_DIR / "raw_pcd_z_lowess_span_sweep.png"
LOWESS_SCATTER_PATH = OUTPUT_DIR / "raw_pcd_z_lowess_best_pred_vs_truth.png"
LOWESS_HIST_PATH = OUTPUT_DIR / "raw_pcd_z_lowess_best_histogram.png"
LOWESS_SCATTER3D_PATH = OUTPUT_DIR / "raw_pcd_z_lowess_best_3d_scatter.png"
LOWESS_SPANS = np.round(np.arange(0.08, 0.181, 0.01), 2)


def load_pdb_xyz(pdb_path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load xyz atom coordinates and CRYST1 cell lengths from a simple PDB."""
    xyz = []
    cell = None
    with open(pdb_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("CRYST1"):
                cell = (
                    float(line[6:15]),
                    float(line[15:24]),
                    float(line[24:33]),
                )
            elif line.startswith(("ATOM", "HETATM")):
                xyz.append([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ])

    if cell is None:
        raise ValueError(f"Missing CRYST1 cell in {pdb_path}")
    if not xyz:
        raise ValueError(f"No ATOM/HETATM coordinates found in {pdb_path}")

    return np.array(xyz, dtype=np.float64), cell


def xyz_angstrom_to_image_pixels(xyz: np.ndarray,
                                 cell: tuple[float, float, float],
                                 image_shape: tuple[int, int]) -> np.ndarray:
    """
    Convert PDB x/y Angstrom coordinates into image pixel coordinates.

    PDB y increases upward in the simulation cell, while image rows increase
    downward, so y is flipped into row coordinates.
    """
    height, width = image_shape
    cell_x, cell_y, _ = cell
    pixel_x = cell_x / width
    pixel_y = cell_y / height

    x_pix = xyz[:, 0] / pixel_x
    y_pix = (cell_y - xyz[:, 1]) / pixel_y
    return np.column_stack([x_pix, y_pix])


def centered_z_metrics(z_pred: np.ndarray,
                       z_true: np.ndarray) -> dict:
    """
    Compute raw PCD z metrics after removing arbitrary mean offsets.

    The simplified PCD estimate is a height initialization, so this benchmark
    compares centered z variation rather than the absolute PDB z baseline.
    """
    valid = np.isfinite(z_pred) & np.isfinite(z_true)
    if not valid.any():
        raise ValueError("No valid z predictions to benchmark")

    pred = z_pred[valid]
    true = z_true[valid]
    pred_centered = pred - np.mean(pred)
    true_centered = true - np.mean(true)
    err = pred_centered - true_centered

    if len(pred_centered) > 1 and np.std(pred_centered) > 0 and np.std(true_centered) > 0:
        corr = float(np.corrcoef(pred_centered, true_centered)[0, 1])
    else:
        corr = np.nan

    return {
        "valid": int(valid.sum()),
        "n_total": int(len(z_true)),
        "z_pred_centered": pred_centered,
        "z_true_centered": true_centered,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "corr": corr,
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred)),
        "true_mean": float(np.mean(true)),
        "true_std": float(np.std(true)),
        "valid_mask": valid,
    }


def save_z_plots(xy_pix: np.ndarray,
                 z_pred: np.ndarray,
                 z_true: np.ndarray,
                 metrics: dict,
                 scatter_path: Path = SCATTER_PATH,
                 hist_path: Path = HIST_PATH,
                 scatter3d_path: Path = SCATTER3D_PATH,
                 label: str = "Raw PCD") -> None:
    """Save scatter, histogram, and 3D colored scatter diagnostics."""
    import matplotlib.pyplot as plt

    zp = metrics["z_pred_centered"]
    zt = metrics["z_true_centered"]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(zt, zp, s=10, alpha=0.65, edgecolors="none")
    lim_min = float(min(np.min(zt), np.min(zp)))
    lim_max = float(max(np.max(zt), np.max(zp)))
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="black", linewidth=1)
    ax.set_xlabel("Ground-truth z, centered (Angstrom)")
    ax.set_ylabel(f"{label} z, centered (Angstrom)")
    ax.set_title(
        f"{label} z vs ground truth\n"
        f"RMSE={metrics['rmse']:.3f} A, MAE={metrics['mae']:.3f} A, "
        f"r={metrics['corr']:.3f}"
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(
        float(min(np.min(zt), np.min(zp))),
        float(max(np.max(zt), np.max(zp))),
        35,
    )
    ax.hist(zt, bins=bins, alpha=0.55, label="Ground truth centered")
    ax.hist(zp, bins=bins, alpha=0.55, label=f"{label} centered")
    ax.set_xlabel("z (Angstrom, centered)")
    ax.set_ylabel("Atom count")
    ax.set_title(f"{label} z histogram comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(hist_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    valid_xy = xy_pix[metrics["valid_mask"]]
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(valid_xy[:, 0], valid_xy[:, 1], zp, c=zt,
                    s=10, cmap="viridis", alpha=0.75)
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    ax.set_zlabel(f"{label} z centered (A)")
    ax.set_title(f"{label} z field, colored by ground-truth z")
    fig.colorbar(sc, ax=ax, shrink=0.7, label="ground-truth z centered (A)")
    fig.tight_layout()
    fig.savefig(scatter3d_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_lowess_sweep(atom_xy_pix: np.ndarray,
                     z_pred_raw: np.ndarray,
                     z_true: np.ndarray) -> list[dict]:
    """Run LOWESS span sweep on raw PCD z without changing pcd_z.py."""
    rows = []
    for span in LOWESS_SPANS:
        z_smooth = lowess_smooth_z(
            atom_xy_pix,
            z_pred_raw,
            frac=float(span),
            n_iter=3,
        )
        metrics = centered_z_metrics(z_smooth, z_true)
        rows.append({
            "span": float(span),
            "z_smooth": z_smooth,
            "metrics": metrics,
        })
    return rows


def save_lowess_sweep_plot(sweep_rows: list[dict]) -> None:
    """Save RMSE/MAE/correlation versus LOWESS span."""
    import matplotlib.pyplot as plt

    spans = np.array([row["span"] for row in sweep_rows], dtype=np.float64)
    rmse = np.array([row["metrics"]["rmse"] for row in sweep_rows], dtype=np.float64)
    mae = np.array([row["metrics"]["mae"] for row in sweep_rows], dtype=np.float64)
    corr = np.array([row["metrics"]["corr"] for row in sweep_rows], dtype=np.float64)

    fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
    ax1.plot(spans, rmse, "o-", label="RMSE")
    ax1.plot(spans, mae, "s-", label="MAE")
    ax1.set_xlabel("LOWESS span / frac")
    ax1.set_ylabel("z error (Angstrom, centered)")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(spans, corr, "^-", color="tab:green", label="Correlation")
    ax2.set_ylabel("Correlation")
    ax2.legend(loc="upper right")

    fig.suptitle("LOWESS span sweep after raw PCD z")
    fig.tight_layout()
    fig.savefig(LOWESS_SWEEP_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image = tifffile.imread(IMAGE_PATH).astype(np.float64)
    xyz_true, cell = load_pdb_xyz(PDB_PATH)
    atom_xy_pix = xyz_angstrom_to_image_pixels(xyz_true, cell, image.shape)

    holes = find_holes(
        image,
        sigma=2.0,
        min_response=0.6,
        min_distance=5.0,
    )

    z_pred_raw = estimate_z_pcd(
        atom_xy_pix,
        holes,
        image,
        search_radius=13.0,
    )
    z_true = xyz_true[:, 2]
    metrics = centered_z_metrics(z_pred_raw, z_true)
    save_z_plots(atom_xy_pix, z_pred_raw, z_true, metrics, label="Raw PCD")

    sweep_rows = run_lowess_sweep(atom_xy_pix, z_pred_raw, z_true)
    save_lowess_sweep_plot(sweep_rows)
    best_row = min(sweep_rows, key=lambda row: row["metrics"]["rmse"])
    best_span = best_row["span"]
    best_metrics = best_row["metrics"]
    save_z_plots(
        atom_xy_pix,
        best_row["z_smooth"],
        z_true,
        best_metrics,
        scatter_path=LOWESS_SCATTER_PATH,
        hist_path=LOWESS_HIST_PATH,
        scatter3d_path=LOWESS_SCATTER3D_PATH,
        label=f"LOWESS span {best_span:.2f}",
    )

    print("Raw PCD z benchmark: real simulated TEM")
    print("=======================================")
    print(f"image             : {IMAGE_PATH}")
    print("image role        : preprocessed Stage 2 input")
    print(f"ground truth xyz  : {PDB_PATH}")
    print(f"image shape       : {image.shape}")
    print(f"atoms             : {len(xyz_true)}")
    print(f"detected holes    : {len(holes)}")
    print(f"valid z estimates : {metrics['valid']}/{metrics['n_total']}")
    print("")
    print("Centered raw PCD z metrics")
    print(f"z RMSE            : {metrics['rmse']:.6f} Angstrom")
    print(f"mean abs z error  : {metrics['mae']:.6f} Angstrom")
    print(f"z correlation     : {metrics['corr']:.6f}")
    print("")
    print("Distribution summary")
    print(f"pred z mean/std   : {metrics['pred_mean']:.6f} / {metrics['pred_std']:.6f}")
    print(f"true z mean/std   : {metrics['true_mean']:.6f} / {metrics['true_std']:.6f}")
    print("")
    print("LOWESS span sweep after raw PCD")
    print("span     RMSE(A)     MAE(A)      corr")
    for row in sweep_rows:
        m = row["metrics"]
        print(f"{row['span']:.2f}     {m['rmse']:.6f}   {m['mae']:.6f}   {m['corr']:.6f}")
    print("")
    print(f"best LOWESS span  : {best_span:.2f}")
    print(f"best z RMSE       : {best_metrics['rmse']:.6f} Angstrom")
    print(f"best mean abs err : {best_metrics['mae']:.6f} Angstrom")
    print(f"best correlation  : {best_metrics['corr']:.6f}")
    print("")
    print(f"scatter plot      : {SCATTER_PATH}")
    print(f"histogram plot    : {HIST_PATH}")
    print(f"3D scatter plot   : {SCATTER3D_PATH}")
    print(f"LOWESS sweep plot : {LOWESS_SWEEP_PATH}")
    print(f"best LOWESS scatter: {LOWESS_SCATTER_PATH}")
    print(f"best LOWESS hist  : {LOWESS_HIST_PATH}")
    print(f"best LOWESS 3D    : {LOWESS_SCATTER3D_PATH}")


if __name__ == "__main__":
    main()
