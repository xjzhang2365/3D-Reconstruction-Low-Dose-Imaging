"""
Iterative 2D Gaussian fitting for Stage 2 atom-position refinement.

Atoms in graphene TEM images are dark columns. The refiner inverts
each local patch before fitting so that atom centres appear as peaks,
then applies a positive-amplitude 2D Gaussian model.

Coordinate convention (internal): (row, col).
The pipeline caller converts from its (col, row) convention before
calling refine() and converts back afterwards.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree


def gaussian_2d(
    coords: tuple[np.ndarray, np.ndarray],
    amplitude: float,
    row: float,
    col: float,
    sigma_r: float,
    sigma_c: float,
    background: float,
) -> np.ndarray:
    r, c = coords
    return amplitude * np.exp(
        -((r - row) ** 2 / (2 * sigma_r ** 2) + (c - col) ** 2 / (2 * sigma_c ** 2))
    ) + background


class GaussianAtomRefiner:
    """
    Refines atom positions from hole-triplet detection using per-atom
    2D Gaussian fitting on a local patch of the denoised image.

    Provides:
      - Sub-pixel position refinement (reduces xy RMSE)
      - Amplitude-based rejection of ghost atoms
      - Lattice-completion recovery of missing atoms in defect regions

    Atoms in STEM/TEM dark-field images of graphene are darker than
    the background. Each patch is inverted before fitting so the atom
    centre appears as a positive-amplitude peak.

    Parameters
    ----------
    patch_radius_px : int
        Half-size of the square fitting window. The full patch is
        (2*R+1) × (2*R+1) pixels.
    min_amplitude_snr : float
        Minimum ratio of Gaussian amplitude to patch noise std.
        Fits below this threshold are treated as ghost atoms and
        discarded.
    sigma_bounds_px : tuple
        (min, max) allowed fitted Gaussian width in pixels.
    bond_length_ang : float
        Expected C-C bond length in Angstroms. Used to compute the
        expected nearest-neighbour distance in pixels for completion.
    pixel_size_ang : float
        Å per pixel.
    completion_search_radius_px : float
        Candidate predictions closer than this to an existing atom
        are skipped (already covered).
    n_completion_passes : int
        Number of lattice-completion iterations.
    """

    def __init__(
        self,
        patch_radius_px: int = 4,
        min_amplitude_snr: float = 0.2,
        sigma_bounds_px: tuple[float, float] = (0.3, 3.0),
        bond_length_ang: float = 1.42,
        pixel_size_ang: float = 0.183,
        completion_search_radius_px: float = 2.0,
        n_completion_passes: int = 2,
        refine_position: bool = False,
    ):
        self.R = patch_radius_px
        self.min_snr = min_amplitude_snr
        self.sig_min, self.sig_max = sigma_bounds_px
        self.bond_length_ang = bond_length_ang
        self.pixel_size = pixel_size_ang
        self.bond_length_px = bond_length_ang / pixel_size_ang
        self.search_radius_px = completion_search_radius_px
        self.n_completion_passes = n_completion_passes
        self.refine_position = refine_position

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refine(
        self,
        image: np.ndarray,
        positions_px: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Refine atom positions and recover missing lattice sites.

        Parameters
        ----------
        image : (H, W) float array
            Denoised TEM image. Atoms appear as dark columns.
        positions_px : (N, 2) array
            Initial atom positions in (row, col) pixel coordinates.

        Returns
        -------
        refined_positions : (M, 2) array
            Refined sub-pixel positions in (row, col).
        fit_quality : (M,) array
            Per-atom amplitude / noise_std ratio.
        status : dict
            Diagnostic counts.
        """
        n_initial = len(positions_px)
        refined, quality = self._fit_all(image, positions_px)
        n_after_fit = len(refined)

        n_added_total = 0
        for _ in range(self.n_completion_passes):
            added, added_quality = self._complete_missing(image, refined)
            if len(added) == 0:
                break
            refined = np.vstack([refined, added])
            quality = np.concatenate([quality, added_quality])
            n_added_total += len(added)

        refined, quality = self._remove_duplicates(refined, quality)

        status = {
            "n_initial": int(n_initial),
            "n_after_fit_rejection": int(n_after_fit),
            "n_rejected_bad_fit": int(n_initial - n_after_fit),
            "n_added_by_completion": int(n_added_total),
            "n_final": int(len(refined)),
        }
        return refined, quality, status

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fit_single(
        self, image: np.ndarray, r0: float, c0: float
    ) -> tuple[float, float, float] | None:
        """Fit one inverted 2D Gaussian centred at (r0, c0).

        Returns (refined_row, refined_col, quality) or None if rejected.
        quality = amplitude / patch_noise_std.
        """
        H, W = image.shape
        R = self.R
        r0i, c0i = int(round(r0)), int(round(c0))
        if r0i - R < 0 or r0i + R + 1 > H:
            return None
        if c0i - R < 0 or c0i + R + 1 > W:
            return None

        raw = image[r0i - R : r0i + R + 1, c0i - R : c0i + R + 1].astype(np.float64)

        # Invert so dark atoms appear as positive peaks
        patch = raw.max() - raw

        # Estimate background noise from patch border pixels
        border = np.concatenate([
            patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]
        ])
        noise_std = float(border.std()) if len(border) > 1 else 1e-3
        noise_std = max(noise_std, 1e-6)

        coords = np.indices(patch.shape)
        flat_coords = (coords[0].ravel(), coords[1].ravel())

        p0 = [
            float(patch.max() - patch.min()),
            float(R), float(R),
            1.5, 1.5,
            float(patch.min()),
        ]
        max_disp = 0.8   # pixels; sub-pixel refinement only, no inter-atom jumps
        bounds_lo = [0.0, R - max_disp, R - max_disp, self.sig_min, self.sig_min, 0.0]
        bounds_hi = [
            float(2.0 * patch.max() + 1e-6),
            float(R + max_disp), float(R + max_disp),
            self.sig_max, self.sig_max,
            float(patch.mean() + 1e-6),
        ]

        try:
            popt, _ = curve_fit(
                gaussian_2d, flat_coords, patch.ravel(),
                p0=p0, bounds=(bounds_lo, bounds_hi), maxfev=500,
            )
        except (RuntimeError, ValueError):
            return None

        A, dr, dc, sr, sc, _bg = popt
        quality = A / noise_std
        if quality < self.min_snr:
            return None
        if not (self.sig_min < sr < self.sig_max):
            return None
        if not (self.sig_min < sc < self.sig_max):
            return None

        if self.refine_position:
            return r0i - R + dr, c0i - R + dc, quality
        return r0, c0, quality   # keep original position; only use fit for quality

    def _fit_all(
        self, image: np.ndarray, positions_px: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        refined, quality = [], []
        for pos in positions_px:
            result = self._fit_single(image, pos[0], pos[1])
            if result is not None:
                r, c, q = result
                refined.append([r, c])
                quality.append(q)
        if not refined:
            return np.empty((0, 2)), np.empty(0)
        return np.array(refined), np.array(quality)

    def _complete_missing(
        self, image: np.ndarray, positions_px: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict and fit atoms at missing graphene lattice sites."""
        if len(positions_px) < 3:
            return np.empty((0, 2)), np.empty(0)

        tree = cKDTree(positions_px)
        dists, idx = tree.query(positions_px, k=4)   # self + 3 neighbors
        neighbor_dists = dists[:, 1:]
        tolerance = 0.20 * self.bond_length_px

        candidates: list[np.ndarray] = []
        for i, pos in enumerate(positions_px):
            real_neighbors = np.abs(neighbor_dists[i] - self.bond_length_px) < tolerance
            if real_neighbors.sum() >= 3:
                continue    # all three neighbors present

            known_idxs = idx[i, 1:][real_neighbors]
            if len(known_idxs) == 0:
                continue
            known_neighbors = positions_px[known_idxs]
            vectors = known_neighbors - pos
            angles = np.arctan2(vectors[:, 0], vectors[:, 1])

            for known_angle in angles:
                for offset in [2 * np.pi / 3, -2 * np.pi / 3]:
                    angle = known_angle + offset
                    pred = pos + self.bond_length_px * np.array(
                        [np.sin(angle), np.cos(angle)]
                    )
                    d, _ = tree.query(pred, k=1)
                    if d < self.search_radius_px:
                        continue
                    candidates.append(pred)

        if not candidates:
            return np.empty((0, 2)), np.empty(0)

        # Deduplicate predictions within search_radius
        cands = np.array(candidates)
        cand_tree = cKDTree(cands)
        pairs = cand_tree.query_pairs(r=self.search_radius_px)
        to_remove: set[int] = set()
        for a, b in pairs:
            to_remove.add(b)
        unique_cands = cands[[i for i in range(len(cands)) if i not in to_remove]]

        accepted, accepted_quality = [], []
        for cand in unique_cands:
            result = self._fit_single(image, cand[0], cand[1])
            if result is not None:
                r, c, q = result
                # Reject if too close to any already-refined atom
                d, _ = tree.query([r, c], k=1)
                if d < self.search_radius_px:
                    continue
                accepted.append([r, c])
                accepted_quality.append(q)

        if not accepted:
            return np.empty((0, 2)), np.empty(0)
        return np.array(accepted), np.array(accepted_quality)

    def _remove_duplicates(
        self,
        positions: np.ndarray,
        quality: np.ndarray,
        min_sep_ang: float = 0.7,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remove pairs closer than min_sep_ang Å; keep higher quality."""
        if len(positions) < 2:
            return positions, quality
        min_sep_px = min_sep_ang / self.pixel_size
        tree = cKDTree(positions)
        pairs = tree.query_pairs(r=min_sep_px)
        to_remove: set[int] = set()
        for i, j in pairs:
            to_remove.add(i if quality[i] < quality[j] else j)
        keep = np.array([i for i in range(len(positions)) if i not in to_remove])
        if len(keep) == 0:
            return np.empty((0, 2)), np.empty(0)
        return positions[keep], quality[keep]
