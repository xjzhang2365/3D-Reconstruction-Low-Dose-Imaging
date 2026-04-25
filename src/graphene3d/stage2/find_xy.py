"""
find_xy.py  –  Stage 2b: Derive atom (x, y) positions from hole triplets
               and refine with Gaussian model fitting

Physical background
-------------------
In a hexagonal graphene lattice, each carbon atom is at the centre of
three adjacent six-membered rings.  Equivalently, each ring centre (hole)
is shared by three atoms.  The dual relationship means:

    atom position ≈ centroid of its three nearest hole neighbours

This geometric step gives us a first estimate of every atom's (x, y)
position in defect-free regions.  We then refine the positions — and
simultaneously fit the peak amplitudes (eta) — using the linearised
multi-Gaussian model from criterion_lin.m (flat background) or
criterion_lin_polyback.m (polynomial background for tilted specimens).

Module structure
----------------
1. build_atom_positions_from_holes()
   Nearest-neighbour triplet search → raw atom (x, y) positions.

2. GaussianModelFitter
   Wraps the criterion_lin / criterion_lin_polyback objective function
   and calls scipy.optimize.least_squares to jointly optimise:
     - background level (flat or polynomial)
     - shared Gaussian width (sigma_g)
     - atom (x, y) positions  [only in the 'full' variant]

3. refine_positions()
   High-level entry point: runs either fixed-position or full refinement.
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import least_squares
from typing import Tuple, Optional
import warnings


# =============================================================================
# 1.  Geometric step: atom positions from hole triplets
# =============================================================================

def build_atom_positions_from_holes(hole_positions: np.ndarray,
                                    nn_radius: float = 8.0,
                                    min_triplet_angle_deg: float = 40.0,
                                    max_side_ratio: float = 1.35,
                                    duplicate_radius: float = 1.5,
                                    nn_radius_tolerance: float = 1.06,
                                    ) -> np.ndarray:
    """
    Estimate atom (x, y) positions as centroids of adjacent hole triplets.

    Algorithm
    ---------
    For each hole H_i:
        1. Find all holes within nn_radius pixels  (nearest-neighbour search).
        2. For every pair (H_j, H_k) among those neighbours, form the triplet
           (H_i, H_j, H_k).
        3. Accept the triplet only if the three interior angles are all
           reasonably close to 60° (indicating a genuine hexagonal ring)
           and the triplet has not already been counted from a different
           starting hole.
        4. The atom position is the centroid (mean) of the three hole centres.

    In the graphene lattice the nearest-hole distance is approximately
    1.42 × √3 ≈ 2.46 Å projected — corresponding to a few pixels depending
    on magnification.  nn_radius should be set to ~1.5× this distance.

    Parameters
    ----------
    hole_positions      : (N, 2) array of [x, y] hole coordinates
    nn_radius           : maximum centre-to-centre distance (pixels) between
                          holes that form a ring.  Tune to your pixel size.
    min_triplet_angle_deg : minimum interior angle (degrees) to accept a
                          triplet as a genuine hexagonal ring (default 40°).

    max_side_ratio      : reject triplets whose longest side is more than this
                          multiple of the shortest side. Set to None to
                          disable this geometric consistency check.
    duplicate_radius    : merge atom candidates within this radius in pixels.
                          Set to None or <= 0 to disable merging.
    nn_radius_tolerance : multiply nn_radius by this factor during neighbour
                          search. This absorbs small subpixel hole-localisation
                          and lattice-distortion errors while the angle and
                          side-ratio checks keep false positives controlled.

    Returns
    -------
    atom_positions : (M, 2) float array, unique atom [x, y] estimates.
    """
    if len(hole_positions) < 3:
        return np.empty((0, 2), dtype=np.float64)

    tree = KDTree(hole_positions)
    effective_nn_radius = nn_radius * max(1.0, float(nn_radius_tolerance))
    atom_list = []
    seen_triplets = set()          # avoid counting same ring from 3 starting holes

    for i, h_i in enumerate(hole_positions):
        # Neighbours of h_i within nn_radius (excluding h_i itself)
        neighbour_idxs = tree.query_ball_point(h_i, effective_nn_radius)
        neighbour_idxs = [idx for idx in neighbour_idxs if idx != i]

        if len(neighbour_idxs) < 2:
            continue

        # Try every pair among the neighbours
        for a, j in enumerate(neighbour_idxs):
            for k in neighbour_idxs[a + 1:]:
                # Canonical triplet key (sorted) to avoid duplicates
                key = tuple(sorted([i, j, k]))
                if key in seen_triplets:
                    continue

                h_j = hole_positions[j]
                h_k = hole_positions[k]

                # Check that j and k are also close to each other
                if np.linalg.norm(h_j - h_k) > effective_nn_radius:
                    continue

                # Verify the three interior angles are ≥ min_triplet_angle_deg
                # (rejects very flat or degenerate triplets)
                if not _triplet_angle_ok(h_i, h_j, h_k, min_triplet_angle_deg):
                    continue
                if not _triplet_side_ratio_ok(h_i, h_j, h_k, max_side_ratio):
                    continue

                seen_triplets.add(key)
                centroid = (h_i + h_j + h_k) / 3.0
                atom_list.append(centroid)

    if not atom_list:
        return np.empty((0, 2), dtype=np.float64)

    atoms = np.array(atom_list, dtype=np.float64)
    return merge_close_positions(atoms, radius=duplicate_radius)


def _triplet_angle_ok(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                      min_angle_deg: float) -> bool:
    """
    Return True if all three interior angles of triangle (p1, p2, p3)
    are ≥ min_angle_deg.  Rejects degenerate / very elongated triangles
    that cannot correspond to a hexagonal ring.
    """
    pts = [p1, p2, p3]
    for i in range(3):
        a = pts[i]
        b = pts[(i + 1) % 3]
        c = pts[(i + 2) % 3]
        v1 = b - a
        v2 = c - a
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        if angle_deg < min_angle_deg:
            return False
    return True


def _triplet_side_ratio_ok(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                           max_side_ratio: Optional[float]) -> bool:
    """
    Return True if the hole triplet is close enough to an equilateral triangle.

    Bright hole centres in defect-free graphene form a triangular lattice, so a
    valid synthetic-data atom candidate should come from three neighbouring
    holes with similar pairwise distances.
    """
    if max_side_ratio is None or max_side_ratio <= 0:
        return True

    sides = np.array([
        np.linalg.norm(p1 - p2),
        np.linalg.norm(p2 - p3),
        np.linalg.norm(p3 - p1),
    ], dtype=np.float64)

    shortest = float(np.min(sides))
    if shortest <= 1e-12:
        return False
    return float(np.max(sides)) / shortest <= max_side_ratio


def merge_close_positions(positions: np.ndarray,
                          radius: Optional[float] = 1.5) -> np.ndarray:
    """
    Merge near-duplicate xy positions by connected-component averaging.

    This geometric cleanup is used after hole-triplet atom generation.  It does
    not use dark-atom positive-Gaussian fitting on the raw image, so it stays
    consistent with the Stage 2 contrast convention.
    """
    pts = np.asarray(positions, dtype=np.float64)
    if pts.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    pts = pts.reshape(-1, 2)

    if radius is None or radius <= 0 or len(pts) == 1:
        return pts.copy()

    tree = KDTree(pts)
    visited = np.zeros(len(pts), dtype=bool)
    merged = []

    for start_idx in range(len(pts)):
        if visited[start_idx]:
            continue

        stack = [start_idx]
        visited[start_idx] = True
        cluster = []

        while stack:
            idx = stack.pop()
            cluster.append(idx)
            for nb in tree.query_ball_point(pts[idx], radius):
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)

        merged.append(pts[cluster].mean(axis=0))

    return np.array(merged, dtype=np.float64)


def evaluate_xy_predictions(predicted_xy: np.ndarray,
                            true_xy: np.ndarray,
                            match_radius: float = 3.0) -> dict:
    """
    Evaluate synthetic-data xy initialization against ground truth.

    Matching is one-to-one and greedy by distance within match_radius.  The
    returned metrics are for synthetic validation only and do not alter the
    reconstruction pipeline.
    """
    pred = np.asarray(predicted_xy, dtype=np.float64).reshape(-1, 2)
    true = np.asarray(true_xy, dtype=np.float64).reshape(-1, 2)

    if len(pred) == 0 or len(true) == 0:
        tp = 0
        fp = len(pred)
        fn = len(true)
        return {
            'n_pred': len(pred),
            'n_true': len(true),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': 0.0 if len(pred) else 1.0,
            'recall': 0.0 if len(true) else 1.0,
            'rmse': np.nan,
            'mean_error': np.nan,
            'matches': np.empty((0, 3), dtype=np.float64),
        }

    tree = KDTree(pred)
    candidates = []
    for true_idx, true_pos in enumerate(true):
        pred_idxs = tree.query_ball_point(true_pos, match_radius)
        for pred_idx in pred_idxs:
            dist = float(np.linalg.norm(true_pos - pred[pred_idx]))
            candidates.append((dist, pred_idx, true_idx))

    used_pred = set()
    used_true = set()
    matches = []
    for dist, pred_idx, true_idx in sorted(candidates):
        if pred_idx in used_pred or true_idx in used_true:
            continue
        used_pred.add(pred_idx)
        used_true.add(true_idx)
        matches.append((pred_idx, true_idx, dist))

    matches = np.array(matches, dtype=np.float64)
    tp = len(matches)
    fp = len(pred) - tp
    fn = len(true) - tp
    errors = matches[:, 2] if tp else np.array([], dtype=np.float64)

    return {
        'n_pred': len(pred),
        'n_true': len(true),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': tp / len(pred) if len(pred) else 1.0,
        'recall': tp / len(true) if len(true) else 1.0,
        'rmse': float(np.sqrt(np.mean(errors ** 2))) if tp else np.nan,
        'mean_error': float(np.mean(errors)) if tp else np.nan,
        'matches': matches,
    }


def plot_xy_comparison(image: np.ndarray,
                       predicted_xy: np.ndarray,
                       true_xy: np.ndarray,
                       save_path: Optional[str] = None,
                       match_radius: float = 3.0,
                       title: str = "Synthetic xy initialization") -> dict:
    """
    Overlay predicted and ground-truth xy atom positions for synthetic checks.

    Returns the same metrics as evaluate_xy_predictions().  If save_path is
    provided, the overlay is written to disk.
    """
    metrics = evaluate_xy_predictions(predicted_xy, true_xy,
                                      match_radius=match_radius)

    import matplotlib.pyplot as plt

    pred = np.asarray(predicted_xy, dtype=np.float64).reshape(-1, 2)
    true = np.asarray(true_xy, dtype=np.float64).reshape(-1, 2)

    fig, ax = plt.subplots(figsize=(6, 6))
    if image is not None:
        ax.imshow(image, cmap='gray', interpolation='nearest')

    if len(true):
        ax.scatter(true[:, 0], true[:, 1], s=32, facecolors='none',
                   edgecolors='lime', linewidths=0.9, label='Ground truth')
    if len(pred):
        ax.scatter(pred[:, 0], pred[:, 1], s=12, c='red', marker='.',
                   label='Predicted', zorder=3)

    for pred_idx, true_idx, _ in metrics['matches']:
        p = pred[int(pred_idx)]
        t = true[int(true_idx)]
        ax.plot([p[0], t[0]], [p[1], t[1]], color='white',
                linewidth=0.4, alpha=0.6)

    ax.set_title(
        f"{title}\n"
        f"precision={metrics['precision']:.2f}, "
        f"recall={metrics['recall']:.2f}, "
        f"rmse={metrics['rmse']:.2f}px"
    )
    ax.legend(fontsize=8)
    ax.axis('off')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return metrics


# =============================================================================
# 2.  Gaussian model fitting  (Python translation of criterion_lin*.m)
# =============================================================================

def _build_gaussian_matrix(X: np.ndarray, Y: np.ndarray,
                            sigma_g: float,
                            xy: np.ndarray,
                            sigma_noise: np.ndarray) -> np.ndarray:
    """
    Build the design matrix Ga where Ga[:, i] = g_i / sigma_noise.

    g_i(x, y) = exp(-0.5 * R² / sigma_g²)
    R²(x, y)  = (x - xi)² + (y - yi)²

    This is the inner loop from criterion_lin.m:
        R2 = (X - xy(i,1)).^2 + (Y - xy(i,2)).^2;
        g  = exp(-0.5 * R2 / p(2)^2);
        Ga(:, i) = g(:) ./ sigma_vec;

    Parameters
    ----------
    X, Y        : 2-D coordinate grids (psizey × psizex)
    sigma_g     : Gaussian width in pixels
    xy          : (N, 2) atom position estimates
    sigma_noise : 2-D noise map (same shape as X, Y)

    Returns
    -------
    Ga : (n_pixels, N_atoms) matrix
    """
    N = len(xy)
    sigma_vec = sigma_noise.ravel()          # (n_pixels,)
    Ga = np.zeros((len(sigma_vec), N), dtype=np.float64)

    for ii in range(N):
        R2 = (X - xy[ii, 0]) ** 2 + (Y - xy[ii, 1]) ** 2
        g  = np.exp(-0.5 * R2 / sigma_g ** 2)
        Ga[:, ii] = g.ravel() / sigma_vec

    return Ga


def _solve_eta(Ga: np.ndarray,
               m_vec: np.ndarray,
               etamin: float = 0.0) -> np.ndarray:
    """
    Solve for peak amplitudes eta using the linear least-squares system
        GaTGa · eta = GaTY
    and clamp to etamin.

    This is the linearisation trick from criterion_lin.m:
        GaTGa = Ga' * Ga
        GaTY  = Ga' * m_vec
        eta   = GaTGa \\ GaTY     (least squares solve)
        eta(eta < etamin) = etamin

    Using np.linalg.lstsq for numerical stability.
    """
    GaTGa = Ga.T @ Ga                                # (N, N)
    GaTY  = Ga.T @ m_vec                             # (N,)
    eta, _, _, _ = np.linalg.lstsq(GaTGa, GaTY, rcond=None)
    eta = np.maximum(eta, etamin)
    return eta


def criterion_lin(p: np.ndarray,
                  X: np.ndarray, Y: np.ndarray,
                  w: np.ndarray,
                  sigma_noise: np.ndarray,
                  xy: np.ndarray,
                  etamin: float = 0.0) -> np.ndarray:
    """
    Residual vector for the *fixed-position* Gaussian model with flat background.

    Translates criterion_lin.m directly.

    Model:
        fb(x, y) = p[0]  +  Σ_i eta_i · exp(-0.5 · R_i² / p[1]²)

    where p[0] = background level, p[1] = Gaussian sigma (shared),
    and eta_i are solved analytically by linear least squares.

    Parameters
    ----------
    p           : [background, sigma_g]
    X, Y        : coordinate grids (psizey × psizex)
    w           : observed image patch
    sigma_noise : pixel-wise noise estimate
    xy          : (N, 2) fixed atom positions
    etamin      : minimum allowed amplitude (prevents negative peaks)

    Returns
    -------
    Weighted residual vector  (w - fb) / sigma_noise,  shape (n_pixels,)
    """
    background = p[0]
    sigma_g    = p[1]

    w_vec     = w.ravel()
    sigma_vec = sigma_noise.ravel()
    m_vec     = (w_vec - background) / sigma_vec      # background-subtracted, normalised

    Ga  = _build_gaussian_matrix(X, Y, sigma_g, xy, sigma_noise)
    eta = _solve_eta(Ga, m_vec, etamin)

    # Reconstruct model image
    fb = np.full_like(w, background, dtype=np.float64)
    for jj in range(len(xy)):
        R2 = (X - xy[jj, 0]) ** 2 + (Y - xy[jj, 1]) ** 2
        fb += eta[jj] * np.exp(-0.5 * R2 / sigma_g ** 2)

    residuals = (w - fb) / sigma_noise
    return residuals.ravel()


def criterion_lin_polyback(p: np.ndarray,
                           X: np.ndarray, Y: np.ndarray,
                           w: np.ndarray,
                           sigma_noise: np.ndarray,
                           xy: np.ndarray,
                           etamin: float = 0.0) -> np.ndarray:
    """
    Residual vector for fixed-position Gaussian model with **polynomial background**.

    Translates criterion_lin_polyback.m.

    Why polynomial background?
    --------------------------
    When the graphene specimen is tilted relative to the electron beam, the
    projected thickness — and therefore the mean image intensity — varies
    smoothly across the field of view.  A constant background model
    (criterion_lin) would absorb some of this gradient into the peak
    amplitudes, biasing the fit.  A 2nd-order polynomial:

        zeta(x, y) = p[0] + p[1]·x + p[2]·y + p[3]·x² + p[4]·y²

    captures the smooth tilt-induced gradient, leaving the Gaussians to
    model only the atomic contrast.

    Parameters
    ----------
    p           : [a0, a1, a2, a3, a4, sigma_g]
                  (5 polynomial coefficients + 1 shared Gaussian width)
    X, Y        : coordinate grids
    w           : observed image patch
    sigma_noise : pixel-wise noise estimate
    xy          : (N, 2) fixed atom positions
    etamin      : minimum allowed amplitude

    Returns
    -------
    Weighted residual vector, shape (n_pixels,)
    """
    # Polynomial background surface
    zeta = (p[0]
            + p[1] * X + p[2] * Y
            + p[3] * X ** 2 + p[4] * Y ** 2)

    sigma_g   = p[5]
    w_vec     = w.ravel()
    sigma_vec = sigma_noise.ravel()
    m_vec     = (w_vec - zeta.ravel()) / sigma_vec

    Ga  = _build_gaussian_matrix(X, Y, sigma_g, xy, sigma_noise)
    eta = _solve_eta(Ga, m_vec, etamin)

    fb = zeta.copy().astype(np.float64)
    for jj in range(len(xy)):
        R2 = (X - xy[jj, 0]) ** 2 + (Y - xy[jj, 1]) ** 2
        fb += eta[jj] * np.exp(-0.5 * R2 / sigma_g ** 2)

    residuals = (w - fb) / sigma_noise
    return residuals.ravel()


def criterion_lin_full(p: np.ndarray,
                       X: np.ndarray, Y: np.ndarray,
                       w: np.ndarray,
                       sigma_noise: np.ndarray,
                       N: int,
                       etamin: float = 0.0) -> np.ndarray:
    """
    Residual vector for the *free-position* Gaussian model with flat background.

    Translates criterion_lin_full.m.

    Unlike criterion_lin, here the atom (x, y) positions are also optimisation
    variables inside p:
        p[0]          = background level
        p[1]          = shared Gaussian sigma
        p[2], p[3]    = x, y of atom 0
        p[4], p[5]    = x, y of atom 1
        ...
        p[2+2i], p[3+2i]  = x, y of atom i

    This allows scipy.optimize.least_squares to jointly refine positions
    *and* the Gaussian width.

    Parameters
    ----------
    p           : parameter vector  [background, sigma_g, x0, y0, x1, y1, ...]
    X, Y        : coordinate grids
    w           : observed image patch
    sigma_noise : pixel-wise noise estimate
    N           : number of atoms (needed to unpack p)
    etamin      : minimum allowed amplitude

    Returns
    -------
    Weighted residual vector, shape (n_pixels,)
    """
    background = p[0]
    sigma_g    = p[1]

    # Unpack atom positions from the parameter vector
    xy = np.array([[p[2 + 2 * ii], p[3 + 2 * ii]] for ii in range(N)],
                  dtype=np.float64)

    w_vec     = w.ravel()
    sigma_vec = sigma_noise.ravel()
    m_vec     = (w_vec - background) / sigma_vec

    Ga  = _build_gaussian_matrix(X, Y, sigma_g, xy, sigma_noise)
    eta = _solve_eta(Ga, m_vec, etamin)

    fb = np.full_like(w, background, dtype=np.float64)
    for jj in range(N):
        R2 = (X - xy[jj, 0]) ** 2 + (Y - xy[jj, 1]) ** 2
        fb += eta[jj] * np.exp(-0.5 * R2 / sigma_g ** 2)

    residuals = (w - fb) / sigma_noise
    return residuals.ravel()


# =============================================================================
# 3.  High-level refinement entry point
# =============================================================================

def refine_positions(image: np.ndarray,
                     atom_positions_init: np.ndarray,
                     sigma_noise: Optional[np.ndarray] = None,
                     sigma_g_init: float = 1.5,
                     etamin: float = 0.0,
                     use_polyback: bool = False,
                     refine_xy: bool = False,
                     verbose: bool = False
                     ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Refine atom (x, y) positions and fit Gaussian amplitudes over the full image.

    Chooses between four criterion functions based on the flags:
        use_polyback=False, refine_xy=False  →  criterion_lin           (most common)
        use_polyback=True,  refine_xy=False  →  criterion_lin_polyback  (tilted specimen)
        use_polyback=False, refine_xy=True   →  criterion_lin_full
        use_polyback=True,  refine_xy=True   →  criterion_lin_full_polyback  (todo)

    Parameters
    ----------
    image               : full 2-D TEM image (float, same frame as atom_positions)
    atom_positions_init : (N, 2) initial atom [x, y] from build_atom_positions_from_holes()
    sigma_noise         : 2-D noise map; if None, taken as sqrt(image) (shot noise)
    sigma_g_init        : initial guess for the shared Gaussian width (pixels)
    etamin              : minimum allowed peak amplitude
    use_polyback        : if True, use polynomial background model
    refine_xy           : if True, also optimise atom (x, y) coordinates
    verbose             : if True, print optimiser summary

    Returns
    -------
    refined_xy  : (N, 2) refined atom positions (same as init if refine_xy=False)
    eta         : (N,) fitted peak amplitudes
    info        : dict with keys 'sigma_g', 'background_params', 'cost', 'success'
    """
    img = image.astype(np.float64)
    h, w_img = img.shape
    N = len(atom_positions_init)

    # Coordinate grids  (X → columns/x,  Y → rows/y)
    # This matches MATLAB's meshgrid convention used in criterion_lin.m
    cols = np.arange(w_img, dtype=np.float64)
    rows = np.arange(h, dtype=np.float64)
    X, Y = np.meshgrid(cols, rows)          # both shape (h, w_img)

    # Default noise model: Poisson (shot noise) → sigma ≈ sqrt(intensity)
    if sigma_noise is None:
        # Add small floor to avoid division by zero
        sigma_noise = np.sqrt(np.maximum(img, 1.0))

    if not use_polyback and not refine_xy:
        # --- criterion_lin: p = [background, sigma_g] ----------------------
        p0 = np.array([np.mean(img), sigma_g_init])
        bounds_lo = [0.0,  0.5]
        bounds_hi = [np.max(img) * 2, 10.0]

        result = least_squares(
            criterion_lin, p0,
            args=(X, Y, img, sigma_noise, atom_positions_init, etamin),
            bounds=(bounds_lo, bounds_hi),
            method='trf',
            verbose=2 if verbose else 0
        )
        refined_xy = atom_positions_init.copy()
        sigma_g_fit = result.x[1]
        bg_params   = result.x[0]

    elif use_polyback and not refine_xy:
        # --- criterion_lin_polyback: p = [a0,a1,a2,a3,a4, sigma_g] --------
        p0 = np.array([np.mean(img), 0.0, 0.0, 0.0, 0.0, sigma_g_init])
        bounds_lo = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.5]
        bounds_hi = [ np.inf,  np.inf,  np.inf,  np.inf,  np.inf, 10.0]

        result = least_squares(
            criterion_lin_polyback, p0,
            args=(X, Y, img, sigma_noise, atom_positions_init, etamin),
            bounds=(bounds_lo, bounds_hi),
            method='trf',
            verbose=2 if verbose else 0
        )
        refined_xy = atom_positions_init.copy()
        sigma_g_fit = result.x[5]
        bg_params   = result.x[:5]

    elif not use_polyback and refine_xy:
        # --- criterion_lin_full: p = [bg, sigma_g, x0,y0, x1,y1, ...] -----
        p0 = np.concatenate([
            [np.mean(img), sigma_g_init],
            atom_positions_init.ravel()
        ])
        result = least_squares(
            criterion_lin_full, p0,
            args=(X, Y, img, sigma_noise, N, etamin),
            method='trf',
            verbose=2 if verbose else 0
        )
        # Unpack refined positions from result
        refined_xy  = result.x[2:].reshape(N, 2)
        sigma_g_fit = result.x[1]
        bg_params   = result.x[0]

    else:
        raise NotImplementedError(
            "criterion_lin_full_polyback not yet implemented in this module."
        )

    # Recover eta from the final parameter values
    # Re-run the Gaussian matrix with the final parameters to get eta
    if not use_polyback:
        background_final = bg_params if np.isscalar(bg_params) else bg_params
        sigma_g_final    = sigma_g_fit
        m_vec = (img.ravel() - float(bg_params)) / sigma_noise.ravel()
    else:
        zeta = (bg_params[0]
                + bg_params[1] * X + bg_params[2] * Y
                + bg_params[3] * X ** 2 + bg_params[4] * Y ** 2)
        m_vec = (img.ravel() - zeta.ravel()) / sigma_noise.ravel()
        sigma_g_final = sigma_g_fit

    Ga_final = _build_gaussian_matrix(X, Y, sigma_g_final,
                                      refined_xy, sigma_noise)
    eta = _solve_eta(Ga_final, m_vec, etamin)

    info = {
        'sigma_g':           sigma_g_fit,
        'background_params': bg_params,
        'cost':              result.cost,
        'success':           result.success if hasattr(result, 'success') else True,
        'message':           result.message if hasattr(result, 'message') else ''
    }

    if verbose:
        print(f"  Optimiser: cost={result.cost:.4f}, "
              f"sigma_g={sigma_g_fit:.3f}px, "
              f"N_atoms={N}")

    return refined_xy, eta, info
