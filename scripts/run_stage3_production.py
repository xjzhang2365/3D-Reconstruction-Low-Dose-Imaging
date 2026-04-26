#!/usr/bin/env python
"""
Stage 3 paper-aligned production run.
Target: sigma_z approaching 0.45 A on the simulated dataset.

Estimated runtime: 3-5 hours.
Checkpointed every outer iteration.
Resumable via --resume flag.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graphene3d.stage3.sa_refine import (
    make_paper_aligned_config,
    run_sa_refinement_paper_aligned,
)
from graphene3d.stage3.pcd_ctf_simulator import AbtemSimulator


def load_ground_truth(pdb_path: Path) -> np.ndarray | None:
    if not pdb_path.exists():
        print(f"  Ground truth PDB not found: {pdb_path}")
        return None
    positions = []
    with open(pdb_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                positions.append([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ])
    return np.array(positions) if positions else None


def _resolve_target(repo_root: Path) -> np.ndarray:
    candidates = [
        repo_root / "outputs" / "preprocessing" / "preprocessed_frame2.tif",
        repo_root / "outputs" / "preprocessing" / "preprocessed_frame21.tif",
        repo_root / "data" / "simulated" / "target_preprocessed_like_raw21.npy",
    ]
    for p in candidates:
        if p.exists():
            print(f"  Target image: {p}")
            if p.suffix == ".npy":
                return np.load(p)
            import tifffile
            return tifffile.imread(p).astype(np.float64)
    raise FileNotFoundError(
        "No preprocessed target found. Run scripts/run_preprocessing.py first."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3 paper-aligned production run"
    )
    parser.add_argument("--max-outer", type=int, default=30)
    parser.add_argument("--max-inner", type=int, default=400)
    parser.add_argument("--checkpoint-dir", type=str,
                        default="runs/stage3_production")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--md-mode", type=str, default="lammps",
                        choices=["lammps", "none"])
    parser.add_argument("--lammps-exe", type=str, required=False)
    parser.add_argument("--potential-file", type=str, required=False)
    args = parser.parse_args()

    sa_input_path = (REPO_ROOT / "outputs" / "stage2" / "validation"
                     / "stage2_validation_init_sa_input.npz")
    if not sa_input_path.exists():
        raise FileNotFoundError(
            f"Stage 2 SA input not found: {sa_input_path}\n"
            "Run scripts/run_stage2.py first."
        )
    data = np.load(sa_input_path)
    positions = data["xyz_angstrom"]

    target = _resolve_target(REPO_ROOT)
    gt = load_ground_truth(REPO_ROOT / "data" / "simulated" / "ground_xyz.pdb")

    print("Production run starting:")
    print(f"  positions: {positions.shape}")
    print(f"  target:    {target.shape}")
    print(f"  outer iterations: {args.max_outer}")
    print(f"  inner steps:      {args.max_inner}")
    print(f"  total SA+MD steps: {args.max_outer * args.max_inner}")
    print(f"  estimated runtime: "
          f"{args.max_outer * args.max_inner * 0.8 / 3600:.1f} hours")
    if args.resume:
        print(f"  mode: RESUME from {args.checkpoint_dir}")

    sim = AbtemSimulator(
        pixel_size_ang=0.183,
        image_shape=target.shape,
        defocus_ang=-80.0,
        slice_thickness=1.0,
    )

    config_kwargs = dict(
        max_outer_iters=args.max_outer,
        max_inner_steps=args.max_inner,
        T_final_fraction=0.01,
        step_size_xy_ang=0.08,
        step_size_z_ang=0.15,
        md_mode=args.md_mode,
        lammps_max_displacement_angstrom=0.5,
        outer_tol=1e-5,
    )
    if args.md_mode == "lammps":
        if not args.lammps_exe:
            parser.error("--lammps-exe is required when --md-mode=lammps")
        if not args.potential_file:
            parser.error("--potential-file is required when --md-mode=lammps")
        config_kwargs["lammps_executable"] = args.lammps_exe
        config_kwargs["potential_file"] = args.potential_file
        config_kwargs["lammps_workdir"] = str(
            Path(args.checkpoint_dir) / "lammps_work"
        )

    config = make_paper_aligned_config(**config_kwargs)

    t_start = time.time()
    final_positions, history = run_sa_refinement_paper_aligned(
        config, positions, target, sim,
        ground_truth=gt,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
    )
    elapsed = time.time() - t_start

    print(f"\n=== PRODUCTION RUN COMPLETE ===")
    print(f"Total wall-clock: {elapsed / 3600:.2f} hours")

    print(f"\nchi2 trajectory:")
    for i, c in enumerate(history['chi2']):
        z = history['z_rmsd'][i]
        z_str = f"{z:.4f}" if z is not None else "N/A"
        a = history['acceptance'][i]
        print(f"  outer {i:2d}: chi2={c:.6f}, z_rmsd={z_str} A, accept={a:.2f}")

    final_z = history['z_rmsd'][-1]
    init_z  = history['z_rmsd'][0]
    print(f"\nFinal z-RMSD: {final_z:.4f} A" if final_z is not None else "\nFinal z-RMSD: N/A")
    print("Paper target: 0.4500 A")
    print(f"Initial:      {init_z:.4f} A" if init_z is not None else "Initial:      N/A")

    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "final_result.npz",
        positions=final_positions,
        history_chi2=np.array(history['chi2']),
        history_z_rmsd=np.array(
            [v if v is not None else np.nan for v in history['z_rmsd']]
        ),
    )
    print(f"\nFinal positions saved: {out_dir / 'final_result.npz'}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(8, 5))
    iters = list(range(len(history['chi2'])))
    ax1.bar(iters, history['chi2'], color='green', alpha=0.6, label='chi2')
    ax1.set_xlabel('Outer iteration')
    ax1.set_ylabel('chi2', color='green')
    ax2 = ax1.twinx()
    valid = [(i, v) for i, v in enumerate(history['z_rmsd']) if v is not None]
    if valid:
        ix, zv = zip(*valid)
        ax2.plot(ix, zv, 'r-o', linewidth=2, label='z-RMSD')
    ax2.set_ylabel('z-RMSD (A)', color='red')
    ax2.axhline(0.45, color='red', linestyle='--', alpha=0.5,
                label='Paper target (0.45 A)')
    ax2.legend(loc='upper right')
    plt.title('Stage 3 Production Run Convergence')
    plt.tight_layout()
    plot_path = out_dir / "convergence.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Convergence plot: {plot_path}")


if __name__ == "__main__":
    main()
