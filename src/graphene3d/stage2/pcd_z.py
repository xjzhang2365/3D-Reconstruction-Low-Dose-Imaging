"""
pcd_z.py  –  Stage 2d: z-height estimation via the simplified PCD method

Physical background
-------------------
The Projected Charge Density (PCD) approximation (Lynch et al. 1975;
Chen et al. Scientific Reports 2017) states that, for a thin phase object,
a small vertical displacement Δz of an atom produces a proportional change
in the local image intensity:

    ΔI(x, y, Δz) = Δz · (λσ / 2π) · ρ(x,y) / (ε·ε₀)

where λ is the electron wavelength, σ is the interaction constant, and
ρ(x,y) is the projected charge density at the atom site.

In practice — and as implemented in p5_z.m — this reduces to a simpler
empirical formula:

    z_i = (mean_hole_intensity_i  -  atom_intensity_i) / (I_mean · 0.0049)

where:
  - mean_hole_intensity_i  = mean of the 3×3 pixel neighbourhood around
                             each of the nearby ring centres (holes) for atom i
  - atom_intensity_i       = mean of the 3×3 pixel neighbourhood around atom i
  - I_mean                 = mean intensity of the entire image
  - 0.0049                 = empirical calibration constant from the Stobbs-
                             factor-corrected PCD slope for this dataset
                             (≈ λσ·ρ / (2π·ε·ε₀·I_mean) in consistent units)

The resulting z values are then divided by 9.3 to convert from raw intensity
units to Angstroms (this factor absorbs the pixel calibration and the
Stobbs factor).

Note: this is a *simplified* version of the full PCD method.  The full
version (Chen et al. 2017) also calibrates the Stobbs factor from a
bilayer reference region.  That calibration is not performed here.

After z estimation, LOWESS smoothing is applied to reduce noise in the
z-height map before feeding the structure into the SA optimisation.
"""

import numpy as np
from scipy.spatial import KDTree
from typing import Optional, Tuple


# =============================================================================
# 1.  3×3 neighbourhood mean  (replicates the surround_pix loop in p5_z.m)
# =============================================================================

def neighbourhood_mean(image: np.ndarray, x: float, y: float) -> float:
    """
    Compute the mean intensity in a 3×3 pixel window centred at (x, y).

    Matches the MATLAB surround_pix loop in p5_z.m:
        surround_pix(1) = im(py-1, px-1)
        ...
        surround_pix(9) = im(py,   px  )
        h_inten = mean(surround_pix)

    Parameters
    ----------
    image : 2-D image array (any numeric dtype)
    x     : column coordinate (float, will be rounded)
    y     : row coordinate    (float, will be rounded)

    Returns
    -------
    float : mean intensity of the 9-pixel neighbourhood, or NaN if out of bounds
    """
    h, w = image.shape
    px = int(round(x))
    py = int(round(y))

    # Check that a full 3×3 window fits inside the image
    if px < 1 or px > w - 2 or py < 1 or py > h - 2:
        return np.nan

    window = image[py - 1:py + 2, px - 1:px + 2]   # 3×3
    return float(np.mean(window))


# =============================================================================
# 2.  Assign nearby holes to each atom  (replicates the distance loop in p5_z.m)
# =============================================================================

def assign_holes_to_atoms(atom_positions: np.ndarray,
                          hole_positions: np.ndarray,
                          image: np.ndarray,
                          search_radius: float = 13.0
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each atom, find all holes within search_radius pixels and record
    their mean 3×3 intensity.

    In p5_z.m:
        if dis < 13
            intensity = h_inten(eee)
            inten = [inten; intensity]

    The mean hole intensity per atom is then used as the "background reference"
    from which the atom's own dark intensity is subtracted to get Δz.

    Parameters
    ----------
    atom_positions : (N_atoms, 2) [x, y] array
    hole_positions : (N_holes, 2) [x, y] array
    image          : 2-D image array
    search_radius  : maximum distance (pixels) to associate a hole with an atom

    Returns
    -------
    atom_intensities  : (N_atoms,)  mean 3×3 intensity at each atom position
    mean_hole_intens  : (N_atoms,)  mean intensity of nearby holes per atom
                        (NaN if no holes found within search_radius)
    """
    N_atoms = len(atom_positions)
    N_holes = len(hole_positions)

    # Pre-compute 3×3 mean intensity at every hole position
    hole_intensities = np.array([
        neighbourhood_mean(image, h[0], h[1])
        for h in hole_positions
    ], dtype=np.float64)

    # Pre-compute 3×3 mean intensity at every atom position
    atom_intensities = np.array([
        neighbourhood_mean(image, a[0], a[1])
        for a in atom_positions
    ], dtype=np.float64)

    # Use KDTree for efficient radius search
    if N_holes == 0:
        return atom_intensities, np.full(N_atoms, np.nan)

    hole_tree = KDTree(hole_positions)
    mean_hole_intens = np.full(N_atoms, np.nan, dtype=np.float64)

    for i, atom in enumerate(atom_positions):
        neighbour_idxs = hole_tree.query_ball_point(atom, search_radius)
        if not neighbour_idxs:
            continue
        nearby_h_intens = hole_intensities[neighbour_idxs]
        # Remove NaN entries (holes too close to border)
        nearby_h_intens = nearby_h_intens[~np.isnan(nearby_h_intens)]
        if len(nearby_h_intens) > 0:
            mean_hole_intens[i] = float(np.mean(nearby_h_intens))

    return atom_intensities, mean_hole_intens


# =============================================================================
# 3.  PCD z-height estimation  (core formula from p5_z.m)
# =============================================================================

def estimate_z_pcd(atom_positions: np.ndarray,
                   hole_positions: np.ndarray,
                   image: np.ndarray,
                   pcd_slope: float = 0.0049,
                   z_scale: float = 9.3,
                   search_radius: float = 13.0
                   ) -> np.ndarray:
    """
    Estimate z-height for every atom using the simplified PCD formula.

    Formula (from p5_z.m):
        z_raw(i) = (mean_hole_intensity(i) - atom_intensity(i))
                   / (I_mean * pcd_slope)
        z_ang(i) = z_raw(i) / z_scale          [convert to Angstroms]

    Parameters
    ----------
    atom_positions : (N, 2) [x, y] array of atom positions
    hole_positions : (M, 2) [x, y] array of hole (ring-centre) positions
    image          : 2-D TEM image (uint8 or float)
    pcd_slope      : empirical PCD calibration constant (default 0.0049)
                     This encapsulates (λσ·ρ)/(2π·ε·ε₀) normalised by I_mean.
                     Adjust for different accelerating voltages / detectors.
    z_scale        : scale factor to convert raw z units to Angstroms (default 9.3)
                     Absorbs pixel calibration and Stobbs factor.
    search_radius  : pixels within which a hole is considered "nearby" an atom

    Returns
    -------
    z_angstrom : (N,) array of z-heights in Angstroms.
                 Atoms with no nearby holes will have z = NaN.
    """
    img = image.astype(np.float64)
    img_mean = float(np.mean(img))

    atom_intens, hole_intens = assign_holes_to_atoms(
        atom_positions, hole_positions, img, search_radius
    )

    # Core PCD formula
    z_raw = (hole_intens - atom_intens) / (img_mean * pcd_slope)

    # Convert to Angstroms
    z_angstrom = z_raw / z_scale

    return z_angstrom


# =============================================================================
# 4.  LOWESS smoothing of the z-height field
# =============================================================================

def lowess_smooth_z(atom_positions: np.ndarray,
                    z_values: np.ndarray,
                    frac: float = 0.1,
                    n_iter: int = 3
                    ) -> np.ndarray:
    """
    Apply LOWESS (Locally Weighted Scatterplot Smoothing) to the z-height field.

    LOWESS fits a local weighted polynomial at each atom position, using only
    the spatially nearest atoms as neighbours.  This suppresses high-frequency
    noise in the z estimates (which arise from shot noise in the image
    intensities) while preserving the genuine low-frequency ripple structure.

    This is the final step before the structure is passed to Simulated
    Annealing: the smoothed structure is physically reasonable and close
    enough to the true structure for SA to converge efficiently.

    Implementation
    --------------
    We implement a simple 2D LOWESS:
        For each atom i:
            1. Find the frac * N nearest neighbours by Euclidean distance.
            2. Compute tricubic weights w_j = (1 - (d_j/d_max)³)³
            3. Fit a weighted least-squares plane z = a + b·x + c·y to neighbours.
            4. z_smooth(i) = a + b·x_i + c·y_i

    Repeated n_iter times for robustness (each iteration down-weights residual
    outliers, as in the standard LOWESS algorithm).

    Parameters
    ----------
    atom_positions : (N, 2) [x, y] array
    z_values       : (N,) raw z-heights (may contain NaN)
    frac           : fraction of atoms used as neighbours (default 0.1)
    n_iter         : number of robustness iterations (default 3)

    Returns
    -------
    z_smoothed : (N,) smoothed z-heights (NaN atoms remain NaN)
    """
    N = len(atom_positions)
    if N == 0:
        return z_values.copy()

    # Handle NaN: work only on atoms with valid z
    valid = ~np.isnan(z_values)
    if not valid.any():
        return z_values.copy()

    pos_valid = atom_positions[valid]
    z_valid   = z_values[valid].copy()
    n_valid   = len(pos_valid)

    k = max(3, int(frac * n_valid))   # number of neighbours

    tree = KDTree(pos_valid)
    z_smooth = z_valid.copy()
    robustness_weights = np.ones(n_valid)

    for iteration in range(n_iter):
        z_new = np.zeros(n_valid)

        for i, pos_i in enumerate(pos_valid):
            # Find k nearest neighbours
            dists, idxs = tree.query(pos_i, k=min(k, n_valid))
            d_max = dists[-1] + 1e-10

            # Tricubic kernel weights (spatial)
            u = dists / d_max
            spatial_w = np.maximum(0.0, 1.0 - u ** 3) ** 3

            # Combined weight = spatial × robustness
            w = spatial_w * robustness_weights[idxs]

            # Weighted least-squares fit: z = a + b*x + c*y
            X_nb = pos_valid[idxs]
            z_nb = z_valid[idxs]

            # Design matrix [1, x, y]
            A = np.column_stack([np.ones(len(idxs)), X_nb])
            W = np.diag(w)

            try:
                AtWA = A.T @ W @ A
                AtWz = A.T @ W @ z_nb
                coeffs, _, _, _ = np.linalg.lstsq(AtWA, AtWz, rcond=None)
                z_new[i] = coeffs[0] + coeffs[1] * pos_i[0] + coeffs[2] * pos_i[1]
            except np.linalg.LinAlgError:
                z_new[i] = z_valid[i]    # fallback: keep original

        # Update robustness weights based on residuals
        residuals = np.abs(z_new - z_valid)
        median_res = np.median(residuals)
        if median_res < 1e-10:
            robustness_weights = np.ones(n_valid)
        else:
            u_r = residuals / (6.0 * median_res)
            robustness_weights = np.maximum(0.0, 1.0 - u_r ** 2) ** 2

        z_valid = z_new

    # Put smoothed values back into full array
    z_out = z_values.copy()
    z_out[valid] = z_valid
    return z_out


# =============================================================================
# 5.  Full z-pipeline: PCD + LOWESS
# =============================================================================

def get_z_positions(atom_positions: np.ndarray,
                    hole_positions: np.ndarray,
                    image: np.ndarray,
                    pcd_slope: float = 0.0049,
                    z_scale: float = 9.3,
                    search_radius: float = 13.0,
                    lowess_frac: float = 0.1,
                    lowess_iter: int = 3
                    ) -> np.ndarray:
    """
    Full z-position pipeline: PCD estimation → LOWESS smoothing.

    Parameters
    ----------
    (see estimate_z_pcd and lowess_smooth_z for parameter descriptions)

    Returns
    -------
    z_smoothed : (N,) smoothed z-heights in Angstroms, ready for SA input
    """
    z_raw = estimate_z_pcd(
        atom_positions, hole_positions, image,
        pcd_slope=pcd_slope,
        z_scale=z_scale,
        search_radius=search_radius
    )

    z_smoothed = lowess_smooth_z(atom_positions, z_raw,
                                 frac=lowess_frac,
                                 n_iter=lowess_iter)
    return z_smoothed
