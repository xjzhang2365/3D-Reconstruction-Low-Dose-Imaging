#!/usr/bin/env python
"""
Regenerate Stage 3 production-run convergence plot at publication
quality. Reads checkpoint data; does not re-run SA.
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_history_from_checkpoints(checkpoint_dir):
    files = sorted(Path(checkpoint_dir).glob("checkpoint_outer_*.npz"))
    if not files:
        raise FileNotFoundError(f"No checkpoints in {checkpoint_dir}")
    latest = files[-1]
    data = np.load(latest, allow_pickle=True)
    return {
        "chi2": data["history_chi2"].tolist(),
        "z_rmsd": [None if np.isnan(v) else float(v)
                   for v in data["history_z_rmsd"]],
        "acceptance": data["history_acceptance"].tolist(),
    }


def make_publication_plot(history, output_path, paper_target_z=0.45):
    # === Publication style settings ===
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    # Colors: colorblind-safe, print well in grayscale
    color_chi2   = "#2b6cb0"   # deep blue
    color_zrmsd  = "#c53030"   # deep red
    color_target = "#718096"   # neutral grey

    chi2 = np.array(history["chi2"])
    z_rmsd_raw = history["z_rmsd"]
    iters = np.arange(len(chi2))

    # Single-column figure: 89 mm wide (Nature single-col), 3.5 x 2.5 in
    fig, ax1 = plt.subplots(figsize=(3.5, 2.5))

    # Left axis: chi-squared as line + markers
    ln_chi2 = ax1.plot(
        iters, chi2,
        color=color_chi2, marker="o", markersize=4.5,
        linewidth=1.4, markeredgewidth=0,
        label=r"$\chi^2$",
        zorder=3,
    )
    ax1.set_xlabel("Outer iteration")
    ax1.set_ylabel(r"$\chi^2$", color=color_chi2)
    ax1.tick_params(axis="y", colors=color_chi2)
    ax1.spines["left"].set_color(color_chi2)

    ax1.set_xticks(iters)
    ax1.set_xlim(-0.3, len(iters) - 0.7)

    chi2_range = chi2.max() - chi2.min()
    ax1.set_ylim(
        chi2.min() - 0.15 * chi2_range,
        chi2.max() + 0.15 * chi2_range,
    )

    # Right axis: z-RMSD
    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    z_valid = [(i, v) for i, v in enumerate(z_rmsd_raw) if v is not None]
    ln_z = []
    if z_valid:
        ix, zv = zip(*z_valid)
        ln_z = ax2.plot(
            ix, zv,
            color=color_zrmsd, marker="s", markersize=4.5,
            linewidth=1.4, markeredgewidth=0,
            label="z-RMSD",
            zorder=3,
        )

    ax2.axhline(
        paper_target_z, color=color_target,
        linestyle=(0, (4, 3)), linewidth=1.0, zorder=2,
    )

    ax2.set_ylabel(r"z-RMSD ($\rm{\AA}$)", color=color_zrmsd)
    ax2.tick_params(axis="y", colors=color_zrmsd)
    ax2.spines["right"].set_color(color_zrmsd)

    z_vals = [v for v in z_rmsd_raw if v is not None]
    if z_vals:
        z_min_data = min(min(z_vals), paper_target_z)
        z_max_data = max(z_vals)
        margin = 0.05 * (z_max_data - z_min_data + 0.1)
        ax2.set_ylim(z_min_data - margin, z_max_data + margin)

    # Combined legend
    all_lines = list(ln_chi2) + list(ln_z)
    all_labels = [l.get_label() for l in all_lines]
    target_handle = mpl.lines.Line2D(
        [0], [0],
        color=color_target, linestyle=(0, (4, 3)), linewidth=1.0,
    )
    all_lines.append(target_handle)
    all_labels.append(f"Paper target ({paper_target_z:.2f} " + r"$\rm{\AA}$)")

    ax1.legend(
        all_lines, all_labels,
        loc="lower left",
        frameon=True, framealpha=0.95,
        edgecolor="none",
        fancybox=False,
        borderpad=0.4,
        handletextpad=0.5,
        labelspacing=0.3,
    )

    plt.tight_layout()
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")


def main():
    checkpoint_dir = REPO_ROOT / "runs" / "stage3_production"
    output = REPO_ROOT / "docs" / "images" / "stage3_production_convergence.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoints from: {checkpoint_dir}")
    history = load_history_from_checkpoints(str(checkpoint_dir))
    print(f"  {len(history['chi2'])} outer iterations")
    print(f"  chi2: {history['chi2']}")
    print(f"  z_rmsd: {history['z_rmsd']}")

    make_publication_plot(history, str(output))


if __name__ == "__main__":
    main()
