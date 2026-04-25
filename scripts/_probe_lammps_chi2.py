"""Quick probe: chi2 before/after LAMMPS, and uphill/downhill sample from relaxed minimum."""
import sys
sys.path.insert(0, "src")
import numpy as np

positions = np.load("outputs/stage2/validation/stage2_validation_init_sa_input.npz")["xyz_angstrom"].copy()
target    = np.load("data/simulated/target_preprocessed_like_raw21.npy").astype("float32")

from graphene3d.stage3.pcd_ctf_simulator import AbtemSimulator
sim = AbtemSimulator(pixel_size_ang=0.456, image_shape=(256,256),
                     voltage_kV=80, defocus_ang=-80, Cs_mm=0.001,
                     dose=8000, slice_thickness=1.0)

from graphene3d.stage3.lammps_minimizer import LammpsMinimizer
LAMMPS_EXE = r"C:\Users\xzhan\AppData\Local\LAMMPS 64-bit 22Jul2025\bin\lmp.exe"
POT_FILE   = r"C:\Users\xzhan\AppData\Local\LAMMPS 64-bit 22Jul2025\Potentials\BNC.tersoff"
md = LammpsMinimizer(
    lammps_executable=LAMMPS_EXE,
    potential_file=POT_FILE,
    working_dir="runs/lammps_probe",
    max_displacement_angstrom=0.5,
)

chi2_before = sim.chi2(positions, target)
relaxed     = md.relax(positions)
chi2_after  = sim.chi2(relaxed, target)

print(f"chi2 before LAMMPS: {chi2_before:.6f}")
print(f"chi2 after  LAMMPS: {chi2_after:.6f}")
print(f"chi2 delta:         {chi2_after - chi2_before:+.6f}  ({'UP' if chi2_after > chi2_before else 'DOWN'})")

disp = np.linalg.norm(relaxed - positions, axis=1)
print(f"\nAtom displacement after LAMMPS:")
print(f"  mean={disp.mean():.4f}  max={disp.max():.4f}  fraction>0.1A: {(disp>0.1).mean():.1%}")
print(f"  xy max: {np.linalg.norm(relaxed[:,:2]-positions[:,:2],axis=1).max():.4f}")
print(f"  z  max: {np.abs(relaxed[:,2]-positions[:,2]).max():.4f}")

# Sample from relaxed minimum
chi2_base = chi2_after
rng = np.random.default_rng(42)
n_up = n_down = 0
uphill = []
for _ in range(20):
    p = relaxed.copy()
    idx = rng.integers(len(p))
    p[idx, 0] += rng.normal(0, 0.08)
    p[idx, 1] += rng.normal(0, 0.08)
    p[idx, 2] += rng.normal(0, 0.15)
    p2 = md.relax(p)
    delta = sim.chi2(p2, target) - chi2_base
    if delta > 0:
        n_up += 1
        uphill.append(delta)
    else:
        n_down += 1

print(f"\nFrom LAMMPS minimum, 20 samples: {n_up} uphill, {n_down} downhill")
if uphill:
    print(f"Mean uphill delta: {np.mean(uphill):.6f}")
    print(f"Implied T0 (50% accept): {np.mean(uphill)/np.log(2):.6e}")
