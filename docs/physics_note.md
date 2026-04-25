# Why the simulator z-sensitivity is weak, and why that is OK

At 80 kV, graphene is a near-ideal weak phase object. The depth of
focus is 100-200 Å, and the phase shift from ~1 Å out-of-plane
corrugation is on the order of 0.01 rad, producing intensity
contrast changes below 0.5% after noise.

This means the forward model's chi-squared objective cannot by
itself provide strong z-direction gradients. This is a physics
limit, not a code bug.

The paper's framework handles this by using three complementary
signals:

  1. PCD intensity ratios (Stage 2) give direct per-atom z estimates
     with ~1 Å accuracy — this is the dominant z signal.
  2. MD relaxation (Stage 3 LAMMPS) enforces C-C = 1.42 Å bond
     length and proper bond angles. Once xy positions are correct,
     the bond-length constraint dictates z values.
  3. SA with image matching (Stage 3 simulator) refines the full
     3D model by matching honeycomb contrast patterns. Primarily
     corrects xy, with weak z contribution.

Quantitative sensitivities (abTEM, 80 kV, defocus=-80, st=1.0):
  - Half-atom +1Å corrugation: +0.02% chi2 change
  - One-atom +1Å:              +0.005% chi2 change
  - Single-atom +0.1Å xy:      ~1% chi2 change (much larger!)

The simulator is strongly sensitive in xy and weakly sensitive in z.
That matches the paper's σ_x = 0.082 Å, σ_y = 0.096 Å (strong xy
recovery) and σ_z = 0.45 Å (larger error — limited by z signal
strength, refined mainly by MD).

## SA z-drift: why step_z must be 0 in the SA loop

Empirical observation (6 outer × 600 inner, step_z = 0.25 Å):
  - chi2 drops 6.7% over 8 outers × 600 inner (xy improvement working)
  - z_rmsd WORSENS from 1.42 → 1.44 Å (z drifts upward)

Explanation: chi2 sensitivity to z is ~5×10⁻⁵ per atom per 1 Å.
For a step_z = 0.25 Å proposal, the chi2 change is ~3×10⁻⁶, which
is << SA temperature T₀ ≈ 3×10⁻⁴. The Metropolis criterion cannot
distinguish a good z move from a bad one. z effectively undergoes a
random walk, with RMSD growing as sqrt(N_accepted_z_moves).

Fix: set step_z = 0.0 in the SA inner loop. SA optimizes xy only.
z is kept at the Stage 2 PCD estimates (which have ~1 Å accuracy).
LAMMPS, applied after sufficient xy convergence, corrects z via
bond-length constraints.

## Why LAMMPS cannot help z during short SA runs

LAMMPS energy minimization of graphene with the Tersoff potential
drives toward the flat free-standing ground state. Real graphene
corrugation (~0.45 Å) comes from substrate interactions — it is not
the Tersoff energy minimum. Applying LAMMPS before xy is converged
moves atoms away from chi2-optimal positions (worsening chi2) without
meaningfully correcting z (because z errors of ~1.4 Å dominate the
bond stretching, causing LAMMPS to make large corrections that disrupt
both xy and z).

LAMMPS z-correction only becomes effective after:
  1. SA has significantly converged xy (chi2 close to minimum)
  2. The residual z error is ~0.2–0.5 Å (in the regime where bond-length
     deviations from correct xy clearly encode z)

For the full paper result (σ_z = 0.45 Å), SA must run for many more
iterations (estimated ~70,000 SA steps to converge xy) before LAMMPS
can effectively correct z.

## Recommended Stage 3 configuration

Short runs / smoke tests (verified working):
  step_size_xy_ang = 0.08
  step_size_z_ang  = 0.0    # prevent z drift
  max_outer_iters  = 6
  max_inner_steps  = 600
  md_mode          = 'none'

Expected: chi2 drops ~1.5% per outer, z_rmsd stable at Stage 2 value.

Long production run (target σ_z = 0.45 Å):
  step_size_xy_ang = 0.08
  step_size_z_ang  = 0.0
  max_outer_iters  = ~100 (until chi2 converges)
  max_inner_steps  = 600
  md_mode          = 'lammps'    # enabled after xy converged
  lammps_max_displacement_angstrom = 0.5  # larger z freedom at end
