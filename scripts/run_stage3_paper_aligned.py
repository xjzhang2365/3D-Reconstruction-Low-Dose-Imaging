"""
Full Stage 3 SA+MD run using paper-aligned config.

Tests both no-MD and LAMMPS modes, reporting chi2 and z-RMSD per outer iteration.
Usage:
    python scripts/run_stage3_paper_aligned.py [--lammps] [--outer N] [--inner N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

LAMMPS_EXE = Path(
    r"C:\Users\xzhan\AppData\Local\LAMMPS 64-bit 22Jul2025\bin\lmp.exe"
)
POTENTIAL_FILE = Path(
    r"C:\Users\xzhan\AppData\Local\LAMMPS 64-bit 22Jul2025\Potentials\BNC.tersoff"
)

SA_INPUT = (
    REPO_ROOT / "outputs" / "stage2" / "validation"
    / "stage2_validation_init_sa_input.npz"
)
GT_PDB = REPO_ROOT / "data" / "simulated" / "ground_xyz.pdb"

TARGET_IMG_NPY = (
    REPO_ROOT / "data" / "simulated" / "target_preprocessed_like_raw21.npy"
)


def load_target_image() -> np.ndarray:
    return np.load(str(TARGET_IMG_NPY)).astype(np.float32)


def load_sa_input() -> np.ndarray:
    data = np.load(str(SA_INPUT))
    return data["xyz_angstrom"].copy()


def load_ground_truth() -> np.ndarray | None:
    if not GT_PDB.exists():
        return None
    positions = []
    with open(GT_PDB) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                positions.append([x, y, z])
    return np.array(positions) if positions else None


def build_simulator(positions: np.ndarray, target: np.ndarray, slice_thickness: float = 1.0):
    from graphene3d.stage3.pcd_ctf_simulator import AbtemSimulator

    pixel_size = 0.456  # Angstroms — matches generate_simulated_target.py
    image_shape = target.shape[:2]
    return AbtemSimulator(
        pixel_size_ang=pixel_size,
        image_shape=image_shape,
        voltage_kV=80.0,
        defocus_ang=-80.0,
        Cs_mm=0.001,
        dose=8000.0,
        slice_thickness=slice_thickness,
        add_noise=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lammps", action="store_true",
                        help="Enable LAMMPS MD relaxation on accepted moves")
    parser.add_argument("--outer", type=int, default=4,
                        help="Number of outer SA iterations (default 4)")
    parser.add_argument("--inner", type=int, default=600,
                        help="Number of inner SA steps per outer (default 600)")
    parser.add_argument("--output", type=str, default=None,
                        help="JSON output file for history (optional)")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("Stage 3 SA+MD — paper-aligned run")
    print(f"  outer={args.outer}, inner={args.inner}, lammps={args.lammps}")
    print("=" * 60)

    positions = load_sa_input()
    target = load_target_image()
    gt = load_ground_truth()

    print(f"Loaded {len(positions)} atoms from SA input")
    print(f"Target image shape: {target.shape}, dtype: {target.dtype}")
    if gt is not None:
        print(f"Ground truth: {len(gt)} atoms from {GT_PDB.name}")

    print("\nBuilding AbtemSimulator ...")
    t0 = time.time()
    simulator = build_simulator(positions, target)
    print(f"  Simulator ready in {time.time() - t0:.1f}s")

    from graphene3d.stage3.sa_refine import (
        make_paper_aligned_config,
        run_sa_refinement_paper_aligned,
    )

    md_mode = "lammps" if args.lammps else "none"
    config = make_paper_aligned_config(
        simulator_kind="abtem",
        md_mode=md_mode,
        max_outer_iters=args.outer,
        max_inner_steps=args.inner,
        step_size_xy_ang=0.08,
        step_size_z_ang=0.0,   # chi2 is z-blind; disable z SA to prevent drift
        initial_temperature="auto",
        T_final_fraction=0.01,
        outer_tol=0.001,
        lammps_executable=str(LAMMPS_EXE) if args.lammps else None,
        potential_file=str(POTENTIAL_FILE) if args.lammps else None,
        lammps_workdir=str(REPO_ROOT / "runs" / "lammps_paper_aligned"),
        lammps_timeout_seconds=20.0,
        lammps_max_displacement_angstrom=0.05,
    )

    t_run = time.time()
    final_pos, history = run_sa_refinement_paper_aligned(
        config=config,
        initial_positions=positions,
        target_image=target,
        simulator=simulator,
        ground_truth=gt,
    )
    elapsed = time.time() - t_run

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total run time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"chi2 history:   {[f'{v:.6f}' for v in history['chi2']]}")
    print(f"z_rmsd history: {[f'{v:.4f}' if v is not None else 'N/A' for v in history['z_rmsd']]}")
    print(f"acceptance:     {[f'{v:.2f}' for v in history['acceptance']]}")
    if args.lammps:
        print(f"MD calls:       {history['n_md_calls']}")
        print(f"MD failures:    {history['n_md_failures']}")
        total_md = sum(history['n_md_calls'])
        total_fail = sum(history['n_md_failures'])
        if total_md > 0:
            print(f"MD fail rate:   {total_fail / total_md:.1%}")

    # Check improvement
    if len(history['chi2']) >= 2:
        chi2_drop = history['chi2'][0] - history['chi2'][-1]
        print(f"\nchi2 drop: {chi2_drop:.6f} ({chi2_drop / history['chi2'][0] * 100:.1f}%)")

    if history['z_rmsd'] and history['z_rmsd'][0] is not None:
        zr = [v for v in history['z_rmsd'] if v is not None]
        print(f"z_rmsd: {zr[0]:.4f} -> {zr[-1]:.4f} A "
              f"({'improved' if zr[-1] < zr[0] else 'worsened'})")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "config": config,
                "history": {k: [float(v) if v is not None else None for v in vs]
                            for k, vs in history.items()},
                "elapsed_s": elapsed,
            }, f, indent=2)
        print(f"\nHistory written to {out_path}")


if __name__ == "__main__":
    main()
