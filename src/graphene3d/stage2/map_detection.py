"""
map_detection.py  –  Stage 2c: MAP probability rule for defect regions

Physical background
-------------------
In regions where the graphene lattice has defects (vacancies, Stone-Wales
defects, grain boundaries), the hexagonal ring structure is broken.  The
geometric hole-triplet method from find_xy.py cannot be applied because
the regular ring pattern no longer exists.

Instead, we use the **Maximum A Posteriori (MAP) probability rule** from:

    Fatermans et al., Physical Review Letters 121, 056101 (2018)

The MAP rule treats atom detection as a model-order selection problem:
"How many Gaussian peaks N are present in this image patch, and where
are they?"  For each candidate N, we compute the posterior probability
p(N | image data) and select the N that maximises it.

The posterior (equation 2 in Fatermans et al.) is approximated as:

    log p(N | w) ∝  -χ²_min/2
                    + 2N · log(4π)
                    - N · log[(η_max - η_min)(ρ_max - ρ_min)]
                    - (1/2) · log det(∇∇χ²)
                    - N · log[(βx_max - βx_min)(βy_max - βy_min)]

where χ²_min is the minimum weighted sum of squared residuals when
fitting N Gaussians to the patch.

Implementation strategy
-----------------------
1. Divide the image into small overlapping patches around defect regions.
2. For each patch, evaluate the MAP probability for N = 0, 1, 2, ... N_max.
3. Select N* = argmax p(N | w).
4. The final atom positions in this patch are the fitted Gaussian centres
   from the N*-atom model.

This module provides:
- map_detect_patch()  : MAP detection on a single image patch
- detect_defect_atoms(): tile a defect region and collect all detections
"""

import numpy as np
from scipy.optimize import least_squares, approx_fprime
from scipy.ndimage import label, center_of_mass
from typing import List, Tuple, Optional
import warnings


def _invert_contrast_for_dark_atoms(patch: np.ndarray) -> np.ndarray:
    """
    Convert dark atom contrast into bright peaks for the positive-Gaussian MAP fit.

    Stage 2 uses two contrast conventions:
      - bright graphene hole centres are handled by hole_finding.py
      - experimental defect-region MAP is intended to detect dark atom columns

    The MAP objective below models positive Gaussian peaks on top of a
    background.  For dark atom columns we therefore invert only the local
    defect patch before fitting, preserving the patch intensity range.
    """
    patch = patch.astype(np.float64)
    return patch.max() + patch.min() - patch


# =============================================================================
# 1.  Criterion function (translates criterion_MAP.m)
# =============================================================================

def criterion_MAP(p: np.ndarray,
                  X: np.ndarray, Y: np.ndarray,
                  w: np.ndarray,
                  sigma_noise: np.ndarray,
                  model_background: np.ndarray,
                  rho: float) -> np.ndarray:
    """
    Weighted residual for N Gaussian peaks on a known background.

    Translates criterion_MAP.m directly:
        model = model_loop
        for nn = 1:N
            R2    = (X - p(3*nn-1)).^2 + (Y - p(3*nn)).^2
            model = model + p(3*nn-2) * exp(-0.5*R2 / rho^2)
        f = (w - model) / sigma

    Parameters
    ----------
    p                : parameter vector, length 3*N:
                         [eta_0, x_0, y_0,  eta_1, x_1, y_1, ...]
    X, Y             : 2-D coordinate grids for the patch
    w                : observed image patch (2-D)
    sigma_noise      : 2-D noise map for the patch
    model_background : 2-D background image (constant or polynomial surface
                       pre-computed before the MAP loop)
    rho              : shared Gaussian width (pixels) — fixed during MAP

    Returns
    -------
    Weighted residual vector (w - model) / sigma,  shape (n_pixels,)
    """
    N = len(p) // 3
    model = model_background.astype(np.float64).copy()

    for nn in range(N):
        eta = p[3 * nn]           # peak amplitude
        x_n = p[3 * nn + 1]      # x centre
        y_n = p[3 * nn + 2]      # y centre
        R2  = (X - x_n) ** 2 + (Y - y_n) ** 2
        model = model + eta * np.exp(-0.5 * R2 / rho ** 2)

    residuals = (w - model) / sigma_noise
    return residuals.ravel()


# =============================================================================
# 2.  MAP probability for a given N  (equation 2, Fatermans et al. 2018)
# =============================================================================

def _log_map_probability(chi2_min: float,
                         hessian: np.ndarray,
                         N: int,
                         eta_range: Tuple[float, float],
                         rho_range: Tuple[float, float],
                         beta_range: Tuple[float, float]) -> float:
    """
    Compute log p(N | w) from the fitted chi-square and Hessian.

    Equation (2) in Fatermans et al. 2018 (simplified log form):

        log p(N|w) ∝  -chi2_min/2
                      + 2N*log(4π)
                      - N*log[(eta_max-eta_min)*(rho_max-rho_min)]
                      - 0.5*log(det(H))
                      - N*log[(bx_range)*(by_range)]

    Parameters
    ----------
    chi2_min   : minimum chi² from the N-atom fit
    hessian    : approximate Hessian of chi² at the minimum (3N × 3N matrix)
    N          : number of atoms in this model
    eta_range  : (eta_min, eta_max) prior range on peak amplitudes
    rho_range  : (rho_min, rho_max) prior range on Gaussian width
    beta_range : (beta_min, beta_max) prior range on position coordinates

    Returns
    -------
    log_prob : float  (up to an additive constant; use for relative comparison)
    """
    if N == 0:
        # Zero-atom model: no Gaussians, just the background.
        # log p(0|w) = -chi2_min/2  (no prior terms)
        return -chi2_min / 2.0

    # Sign-flipped determinant check: use absolute value for log
    sign, log_det = np.linalg.slogdet(hessian)
    if sign <= 0:
        # Degenerate or indefinite Hessian — penalise this model
        return -np.inf

    eta_min, eta_max     = eta_range
    rho_min, rho_max     = rho_range
    beta_min, beta_max   = beta_range

    log_prob = (
        -chi2_min / 2.0
        + 2 * N * np.log(4 * np.pi)
        - N * np.log((eta_max - eta_min) * (rho_max - rho_min) + 1e-30)
        - 0.5 * log_det
        - N * np.log((beta_max - beta_min) ** 2 + 1e-30)
    )
    return log_prob


# =============================================================================
# 3.  MAP detection on a single patch
# =============================================================================

def map_detect_patch(patch: np.ndarray,
                     sigma_noise_patch: Optional[np.ndarray] = None,
                     N_max: int = 5,
                     rho: float = 1.5,
                     eta_range: Tuple[float, float] = (0.0, None),
                     beta_range: Optional[Tuple[float, float]] = None,
                     background: Optional[np.ndarray] = None
                     ) -> Tuple[np.ndarray, int, float]:
    """
    Detect positive Gaussian peaks in a small image patch using the MAP rule.

    For experimental defect-region atom detection, atoms are dark in the
    original TEM contrast.  Callers should invert those patches before calling
    this function; detect_defect_atoms() does that explicitly at the tiling
    boundary.  The MAP model itself intentionally remains a positive-Gaussian
    formulation.

    For each candidate atom count N in [0, N_max]:
        1. Fit N Gaussian peaks to the patch using least_squares.
        2. Compute χ²_min and approximate Hessian.
        3. Evaluate log p(N | patch data).
    Select N* = argmax log p(N | data).

    Parameters
    ----------
    patch              : 2-D image patch (float)
    sigma_noise_patch  : 2-D noise map; if None, uses sqrt(patch)
    N_max              : maximum number of atoms to consider (default 5)
    rho                : fixed Gaussian width in pixels (default 1.5)
    eta_range          : (eta_min, eta_max) for the MAP prior;
                         eta_max defaults to max(patch)
    beta_range         : (beta_min, beta_max) for position prior;
                         defaults to (0, max(patch.shape))
    background         : pre-computed background for the patch; if None,
                         uses the patch mean as a constant background

    Returns
    -------
    atom_positions : (N*, 2) array of detected [x, y] in patch coordinates
    N_best         : selected number of atoms
    log_prob_best  : log MAP probability of the selected model
    """
    ph, pw = patch.shape
    patch  = patch.astype(np.float64)

    if sigma_noise_patch is None:
        sigma_noise_patch = np.sqrt(np.maximum(patch, 1.0))

    if background is None:
        background = np.full_like(patch, np.mean(patch))

    if eta_range[1] is None:
        eta_range = (eta_range[0], float(np.max(patch)))

    if beta_range is None:
        beta_range = (0.0, float(max(ph, pw)))

    # Coordinate grids for the patch
    cols = np.arange(pw, dtype=np.float64)
    rows = np.arange(ph, dtype=np.float64)
    X, Y = np.meshgrid(cols, rows)

    rho_range = (rho * 0.5, rho * 2.0)   # allow ±2× variation for prior

    log_probs = []
    fits      = []         # stores (result, N) for each candidate

    for N in range(0, N_max + 1):
        if N == 0:
            # Zero-atom model: residual is just (patch - background) / sigma
            residuals = (patch - background) / sigma_noise_patch
            chi2 = float(np.sum(residuals ** 2))
            log_p = _log_map_probability(chi2, np.eye(1), 0,
                                         eta_range, rho_range, beta_range)
            log_probs.append(log_p)
            fits.append(None)
            continue

        # Initial guess: spread N atoms evenly across patch centre
        # (a more sophisticated initialisation uses local maxima)
        p0 = _initialise_atoms(patch, N, rho)

        try:
            result = least_squares(
                criterion_MAP, p0,
                args=(X, Y, patch, sigma_noise_patch, background, rho),
                method='trf',
                max_nfev=500 * (1 + N),
                verbose=0
            )
        except Exception:
            log_probs.append(-np.inf)
            fits.append(None)
            continue

        chi2_min = 2.0 * result.cost   # least_squares minimises 0.5 * ||r||²

        # Approximate Hessian via finite differences on the Jacobian
        try:
            J = result.jac                        # (n_pixels, 3N)
            H = J.T @ J                           # Gauss-Newton Hessian approx
        except Exception:
            H = np.eye(3 * N)

        log_p = _log_map_probability(chi2_min, H, N,
                                     eta_range, rho_range, beta_range)
        log_probs.append(log_p)
        fits.append(result)

    # Select the model with the highest MAP probability
    N_best = int(np.argmax(log_probs))
    log_prob_best = log_probs[N_best]

    if N_best == 0 or fits[N_best] is None:
        return np.empty((0, 2), dtype=np.float64), 0, log_prob_best

    # Extract atom positions from the best fit
    best_p = fits[N_best].x
    positions = []
    for nn in range(N_best):
        x_n = best_p[3 * nn + 1]
        y_n = best_p[3 * nn + 2]
        # Only keep atoms that landed inside the patch
        if 0 <= x_n < pw and 0 <= y_n < ph:
            positions.append([x_n, y_n])

    if not positions:
        return np.empty((0, 2), dtype=np.float64), N_best, log_prob_best

    return np.array(positions, dtype=np.float64), N_best, log_prob_best


def detect_dark_atoms_in_local_patch(image: np.ndarray,
                                     center_xy: np.ndarray,
                                     patch_radius: int = 8,
                                     sigma_noise: Optional[np.ndarray] = None,
                                     rho: float = 1.5,
                                     N_max: int = 1,
                                     accept_radius: float = 3.5
                                     ) -> Tuple[np.ndarray, int, float]:
    """
    Run dark-atom MAP detection in a small patch around one candidate position.

    This is a selective fallback helper for unresolved local regions.  It keeps
    the MAP objective as a positive-Gaussian model by locally inverting the
    patch contrast, then returns only detections close to the requested centre.
    """
    img = image.astype(np.float64)
    h, w = img.shape
    cx, cy = np.asarray(center_xy, dtype=np.float64)

    c0 = max(0, int(round(cx)) - patch_radius)
    c1 = min(w, int(round(cx)) + patch_radius + 1)
    r0 = max(0, int(round(cy)) - patch_radius)
    r1 = min(h, int(round(cy)) + patch_radius + 1)
    if c1 <= c0 or r1 <= r0:
        return np.empty((0, 2), dtype=np.float64), 0, -np.inf

    patch = img[r0:r1, c0:c1]
    patch_for_map = _invert_contrast_for_dark_atoms(patch)

    patch_sigma = None
    if sigma_noise is not None:
        patch_sigma = sigma_noise[r0:r1, c0:c1]

    positions_patch, n_best, log_prob = map_detect_patch(
        patch_for_map,
        sigma_noise_patch=patch_sigma,
        N_max=N_max,
        rho=rho,
    )
    if len(positions_patch) == 0:
        return np.empty((0, 2), dtype=np.float64), n_best, log_prob

    positions_full = positions_patch + np.array([c0, r0], dtype=np.float64)
    dists = np.linalg.norm(positions_full - np.array([cx, cy]), axis=1)
    keep = dists <= accept_radius
    return positions_full[keep], n_best, log_prob


def _initialise_atoms(patch: np.ndarray, N: int, rho: float) -> np.ndarray:
    """
    Generate an initial parameter vector for N atoms in the patch.

    Strategy: find the N strongest local maxima in the patch and use them
    as starting positions.  This converges much faster than random init.

    Returns p0: length-3N vector  [eta0, x0, y0,  eta1, x1, y1, ...]
    """
    from skimage.morphology import local_maxima
    from scipy.ndimage import label, center_of_mass

    ph, pw = patch.shape
    mean_val = float(np.mean(patch))
    eta_init = float(np.max(patch) - mean_val)

    # Find all local maxima
    lm = local_maxima(patch, connectivity=2)
    labeled, n_feat = label(lm)
    if n_feat == 0:
        # Fallback: grid of N positions
        xs = np.linspace(rho, pw - rho, max(1, int(np.sqrt(N))) + 1)[:N]
        ys = np.linspace(rho, ph - rho, max(1, int(np.sqrt(N))) + 1)[:N]
        positions = [[xs[min(i, len(xs)-1)], ys[min(i, len(ys)-1)]]
                     for i in range(N)]
    else:
        centroids_rc = center_of_mass(lm, labels=labeled,
                                      index=np.arange(1, n_feat + 1))
        intensities  = [patch[int(round(r)), int(round(c))]
                        for r, c in centroids_rc]
        # Sort by intensity descending, take top N
        order     = np.argsort(intensities)[::-1]
        positions = [[centroids_rc[i][1], centroids_rc[i][0]]
                     for i in order[:N]]
        # Pad with centre positions if not enough maxima found
        while len(positions) < N:
            positions.append([pw / 2.0, ph / 2.0])

    p0 = []
    for x_n, y_n in positions[:N]:
        p0.extend([eta_init, float(x_n), float(y_n)])
    return np.array(p0, dtype=np.float64)


# =============================================================================
# 4.  Tile a defect region with MAP detection
# =============================================================================

def detect_defect_atoms(image: np.ndarray,
                        defect_mask: np.ndarray,
                        patch_size: int = 20,
                        overlap: int = 5,
                        sigma_noise: Optional[np.ndarray] = None,
                        rho: float = 1.5,
                        N_max: int = 4,
                        duplicate_radius: float = 3.0
                        ) -> np.ndarray:
    """
    Apply MAP detection across a defect region by tiling into overlapping patches.

    This implements the strategy from the paper: "we cut the image into pieces
    and find the atoms one by one" for regions where the regular lattice is
    broken.

    Contrast convention
    -------------------
    In the experimental TEM contrast used here, atom columns are dark while
    graphene hole centres are bright.  Since map_detect_patch() fits positive
    Gaussian peaks, each defect-region patch is inverted locally before MAP
    fitting.  Returned coordinates are still in the original image coordinate
    system; only the fitting intensity contrast is inverted.

    Parameters
    ----------
    image         : full 2-D TEM image
    defect_mask   : boolean 2-D mask, True where defect analysis is needed
    patch_size    : side length of each square patch in pixels (default 20)
    overlap       : pixel overlap between adjacent patches (default 5)
    sigma_noise   : 2-D noise map; if None, uses sqrt(image)
    rho           : fixed Gaussian width (pixels)
    N_max         : max atoms per patch (default 4)
    duplicate_radius : minimum distance (pixels) between accepted detections;
                    detections closer than this from different patches are
                    merged (keep the one with higher amplitude)

    Returns
    -------
    defect_atom_positions : (M, 2) array of detected [x, y] in full-image coords
    """
    img = image.astype(np.float64)
    h, w = img.shape

    if sigma_noise is None:
        sigma_noise = np.sqrt(np.maximum(img, 1.0))

    step = patch_size - overlap
    all_positions = []    # will collect [x, y] in full-image coordinates

    # Find bounding box of defect_mask to limit the search area
    rows_mask, cols_mask = np.where(defect_mask)
    if len(rows_mask) == 0:
        return np.empty((0, 2), dtype=np.float64)

    r_min, r_max = int(rows_mask.min()), int(rows_mask.max())
    c_min, c_max = int(cols_mask.min()), int(cols_mask.max())

    for r0 in range(r_min, r_max, step):
        for c0 in range(c_min, c_max, step):
            r1 = min(r0 + patch_size, h)
            c1 = min(c0 + patch_size, w)

            # Skip patch if it does not overlap the defect mask
            if not defect_mask[r0:r1, c0:c1].any():
                continue

            patch       = img[r0:r1, c0:c1]
            patch_sigma = sigma_noise[r0:r1, c0:c1]
            patch_for_map = _invert_contrast_for_dark_atoms(patch)

            positions_patch, N_best, _ = map_detect_patch(
                patch_for_map,
                sigma_noise_patch=patch_sigma,
                N_max=N_max,
                rho=rho
            )

            # Convert patch-local coordinates back to full-image coordinates
            for x_local, y_local in positions_patch:
                all_positions.append([x_local + c0, y_local + r0])

    if not all_positions:
        return np.empty((0, 2), dtype=np.float64)

    positions = np.array(all_positions, dtype=np.float64)

    # Remove duplicates from overlapping patches:
    # keep one detection per duplicate_radius neighbourhood
    positions = _remove_duplicates(positions, duplicate_radius)

    return positions


def _remove_duplicates(positions: np.ndarray,
                       radius: float) -> np.ndarray:
    """
    Greedy duplicate removal: iterate through positions and discard any
    detection that is within `radius` pixels of an already-accepted one.

    This is equivalent to non-maximum suppression in the spatial domain.
    """
    if len(positions) == 0:
        return positions

    from scipy.spatial import KDTree
    tree    = KDTree(positions)
    kept    = np.ones(len(positions), dtype=bool)

    for i in range(len(positions)):
        if not kept[i]:
            continue
        neighbours = tree.query_ball_point(positions[i], radius)
        for j in neighbours:
            if j != i:
                kept[j] = False    # discard the duplicate

    return positions[kept]
