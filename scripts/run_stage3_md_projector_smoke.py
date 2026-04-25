"""
Smoke test for the MD-projector SA architecture.

LAMMPS is applied to every candidate before chi2 evaluation.
The fix vs. the old 0.94-acceptance bug: positions is also relaxed
at the start of each outer iteration, so both current and candidate
are in the same (relaxed) space when computing delta.

Usage:
    python scripts/run_stage3_md_projector_smoke.py
"""

from __future__ import annotations

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
GT_PDB   = REPO_ROOT / "data" / "simulated" / "ground_xyz.pdb"
TARGET_NPY = REPO_ROOT / "data" / "simulated" / "target_preprocessed_like_raw21.npy"


def load_gt(path):
    if not path.exists():
        return None
    pos = []
    with open(path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                pos.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(pos) if pos else None


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    positions = np.load(str(SA_INPUT))["xyz_angstrom"].copy()
    target    = np.load(str(TARGET_NPY)).astype(np.float32)
    gt        = load_gt(GT_PDB)

    print(f"Atoms: {len(positions)}, target shape: {target.shape}")

    from graphene3d.stage3.pcd_ctf_simulator import AbtemSimulator
    sim = AbtemSimulator(
        pixel_size_ang=0.456,
        image_shape=target.shape[:2],
        voltage_kV=80.0,
        defocus_ang=-80.0,
        Cs_mm=0.001,
        dose=8000.0,
        slice_thickness=1.0,
        add_noise=False,
    )

    from graphene3d.stage3.sa_refine import (
        make_paper_aligned_config,
        run_sa_refinement_paper_aligned,
    )

    config = make_paper_aligned_config(
        simulator_kind="abtem",
        md_mode="lammps",
        max_outer_iters=3,
        max_inner_steps=200,
        step_size_xy_ang=0.08,
        step_size_z_ang=0.15,
        initial_temperature="auto",
        T_final_fraction=0.01,
        outer_tol=0.005,
        lammps_executable=str(LAMMPS_EXE),
        potential_file=str(POTENTIAL_FILE),
        lammps_workdir=str(REPO_ROOT / "runs" / "lammps_sa_no_reset"),
        lammps_timeout_seconds=20.0,
        lammps_max_displacement_angstrom=0.5,
        lammps_xy_cap_angstrom=0.05,
        lammps_z_cap_angstrom=0.50,
    )

    t0 = time.time()
    final_pos, hist = run_sa_refinement_paper_aligned(
        config=config,
        initial_positions=positions,
        target_image=target,
        simulator=sim,
        md_relaxer=None,
        ground_truth=gt,
    )
    elapsed = time.time() - t0

    print(f"\nchi2:   {[f'{v:.4f}' for v in hist['chi2']]}")
    print(f"z_rmsd: {[f'{v:.4f}' for v in hist['z_rmsd']]}")
    print(f"accept: {[f'{v:.2f}' for v in hist['acceptance']]}")
    print(f"MD calls: {hist['n_md_calls']}, failures: {hist['n_md_failures']}")
    print(f"Total time: {elapsed:.1f}s")

    # Bond length check: 3D nearest-neighbour distances
    from scipy.spatial import cKDTree
    tree = cKDTree(final_pos)
    dists, _ = tree.query(final_pos, k=4)
    nn = dists[:, 1]   # closest neighbour (exclude self)
    valid = (nn > 1.0) & (nn < 1.8)
    if valid.any():
        print(f"\nMean NN (3D): {nn[valid].mean():.4f} A  "
              f"min={nn[valid].min():.4f}  max={nn[valid].max():.4f}  "
              f"(paper: 1.42 A)")
    else:
        print("\nNo valid NN distances found in 1.0-1.8 A range")

    # Z spread check
    z_std_before = positions[:, 2].std()
    z_std_after  = final_pos[:, 2].std()
    print(f"z std before/after: {z_std_before:.4f} / {z_std_after:.4f} A")

    # Acceptance rate verdict
    accept_avg = np.mean(hist['acceptance'])
    if accept_avg > 0.85:
        print("\nWARNING: acceptance > 0.85 — both-in-relaxed-space fix may not be working")
    elif accept_avg < 0.30:
        print("\nWARNING: acceptance < 0.30 — T0 may be too low")
    else:
        print(f"\nAcceptance {accept_avg:.2f} in range [0.30, 0.85] — OK")

    # z_rmsd verdict
    z_list = [v for v in hist['z_rmsd'] if v is not None]
    if len(z_list) >= 2:
        if z_list[-1] < z_list[0]:
            print(f"z_rmsd DECREASED: {z_list[0]:.4f} -> {z_list[-1]:.4f} A  SUCCESS")
        else:
            print(f"z_rmsd did not decrease: {z_list[0]:.4f} -> {z_list[-1]:.4f} A")


if __name__ == "__main__":
    main()
