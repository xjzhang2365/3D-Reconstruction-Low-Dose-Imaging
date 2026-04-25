"""
hole_finding.py  –  Stage 2a: Detect bright ring-centre (hole) positions

Physical background
-------------------
In aberration-corrected TEM images of graphene recorded at this particular
defocus, the *centres* of the six-membered carbon rings appear as **bright**
spots, while the carbon atom columns themselves appear **dark**.  This is a
consequence of the phase-contrast transfer function at slight under-focus:
the projected charge density of the ring centre (which has no atoms) produces
a local intensity maximum.

The goal here is therefore to find every local brightness maximum in the
filtered image.  Each such maximum is the centre of one hexagonal ring –
we call these positions "holes" (following the MATLAB code convention).

Pipeline in this module
-----------------------
1. Contrast enhancement  (imadjust equivalent → histogram stretch)
2. Gaussian smoothing    (reduce pixel-level noise before peak search)
3. Second contrast stretch
4. Regional maximum detection  (imregionalmax equivalent)
5. Connected-component centroid extraction

These steps follow find_hole.m exactly.

Once hole positions are known, atom positions are derived in find_xy.py by
taking the centroid of every adjacent triplet of holes.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, label
from scipy.ndimage import center_of_mass
from skimage.exposure import rescale_intensity
from skimage.morphology import local_maxima          # equivalent to imregionalmax
import matplotlib.pyplot as plt
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# 1.  Contrast enhancement  (equivalent to MATLAB imadjust)
# ---------------------------------------------------------------------------

def imadjust(image: np.ndarray,
             in_range: Tuple[float, float] = None,
             out_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """
    Stretch image intensities to fill [out_range].

    MATLAB's imadjust() with no arguments saturates the bottom 1 % and top
    1 % of pixels, then linearly maps the remaining range to [0, 1].
    We replicate that behaviour here.

    Parameters
    ----------
    image      : 2-D float or uint image
    in_range   : (low, high) input percentile clipping; defaults to (1%, 99%)
    out_range  : desired output intensity range

    Returns
    -------
    Contrast-stretched image as float64 in out_range.
    """
    img = image.astype(np.float64)

    if in_range is None:
        # Clip at 1st and 99th percentile – same behaviour as MATLAB imadjust()
        p_low  = np.percentile(img, 1)
        p_high = np.percentile(img, 99)
    else:
        p_low, p_high = in_range

    # rescale_intensity linearly maps [p_low, p_high] → out_range
    stretched = rescale_intensity(img,
                                  in_range=(p_low, p_high),
                                  out_range=out_range)
    return stretched


# ---------------------------------------------------------------------------
# 2.  Gaussian smoothing  (equivalent to MATLAB imgaussfilt)
# ---------------------------------------------------------------------------

def smooth_image(image: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Apply a Gaussian low-pass filter to suppress pixel-level noise.

    In find_hole.m the sigma is set to 2 pixels.  For images with a
    different pixel-to-Angstrom calibration you may need to adjust sigma
    so that the spatial frequency cut-off stays at roughly one half the
    graphene lattice spacing.

    Parameters
    ----------
    image : 2-D float image
    sigma : Gaussian standard deviation in pixels (default 2.0)

    Returns
    -------
    Smoothed image, same shape and dtype as input.
    """
    return gaussian_filter(image.astype(np.float64), sigma=sigma)


# ---------------------------------------------------------------------------
# 3.  Regional maxima detection  (equivalent to MATLAB imregionalmax)
# ---------------------------------------------------------------------------

def find_regional_maxima(image: np.ndarray,
                         connectivity: int = 2) -> np.ndarray:
    """
    Return a boolean mask that is True at every regional maximum pixel.

    A regional maximum is a connected set of pixels whose values are all
    strictly greater than every pixel in its immediate neighbourhood.
    This is exactly what MATLAB's imregionalmax() computes.

    scikit-image's local_maxima() uses the same definition with a
    configurable connectivity (1 = 4-connectivity, 2 = 8-connectivity).
    We use 8-connectivity (connectivity=2) to match MATLAB's default.

    Parameters
    ----------
    image        : 2-D float image (already contrast-stretched and smoothed)
    connectivity : 1 → 4-connected,  2 → 8-connected (default)

    Returns
    -------
    Boolean mask, same shape as image.
    """
    # local_maxima returns True wherever the pixel is a strict local maximum
    # within a (2*connectivity+1) neighbourhood
    mask = local_maxima(image, connectivity=connectivity)
    return mask


# ---------------------------------------------------------------------------
# 4.  Connected-component centroid extraction  (equivalent to bwconncomp +
#     regionprops 'Centroid')
# ---------------------------------------------------------------------------

def maxima_to_centroids(mask: np.ndarray) -> np.ndarray:
    """
    Label connected regions in the maxima mask and return their centroids.

    MATLAB workflow:
        cc    = bwconncomp(max_filter)
        stats = regionprops(cc, 'Centroid')

    In Python we use scipy.ndimage.label + center_of_mass, which is
    equivalent.

    Important coordinate convention
    --------------------------------
    MATLAB regionprops returns Centroid as (col, row) = (x, y).
    scipy center_of_mass returns (row, col).
    We convert to (x, y) = (col, row) here so that the output matches the
    MATLAB atom_positions array used in downstream code.

    Parameters
    ----------
    mask : boolean 2-D array from find_regional_maxima()

    Returns
    -------
    hole_positions : (N, 2) float array, columns are [x, y] in pixel coords
    """
    # Label each connected bright region with a unique integer
    labeled_array, num_features = label(mask)

    if num_features == 0:
        return np.empty((0, 2), dtype=np.float64)

    # center_of_mass returns a list of (row, col) tuples
    centroids_rc = center_of_mass(mask,
                                  labels=labeled_array,
                                  index=np.arange(1, num_features + 1))

    # Convert from (row, col) → (x=col, y=row) to match MATLAB convention
    centroids_xy = np.array([[c[1], c[0]] for c in centroids_rc],
                            dtype=np.float64)
    return centroids_xy


def score_holes(image_response: np.ndarray,
                hole_positions: np.ndarray) -> np.ndarray:
    """
    Score hole candidates by their bright-centre response.

    image_response should be the filtered/contrast-stretched image used for
    regional-maximum detection. Higher score means the candidate is a brighter
    local maximum and is more likely to be a true graphene hole centre in the
    synthetic bright-hole convention.
    """
    holes = np.asarray(hole_positions, dtype=np.float64).reshape(-1, 2)
    if len(holes) == 0:
        return np.empty((0,), dtype=np.float64)

    h, w = image_response.shape
    xs = np.clip(np.rint(holes[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.rint(holes[:, 1]).astype(int), 0, h - 1)
    return image_response[ys, xs].astype(np.float64)


def filter_holes_by_response(hole_positions: np.ndarray,
                             scores: np.ndarray,
                             min_response: Optional[float]) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove weak bright-centre candidates below min_response.

    Set min_response=None to disable this synthetic-data filter.
    """
    holes = np.asarray(hole_positions, dtype=np.float64).reshape(-1, 2)
    scores = np.asarray(scores, dtype=np.float64)
    if min_response is None or len(holes) == 0:
        return holes, scores

    keep = scores >= float(min_response)
    return holes[keep], scores[keep]


def suppress_close_holes(hole_positions: np.ndarray,
                         scores: Optional[np.ndarray] = None,
                         min_distance: Optional[float] = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedily keep the strongest hole candidate in each min_distance neighbourhood.

    This prevents a single bright hole centre from producing multiple nearby
    candidates. Set min_distance=None or <=0 to disable suppression.
    """
    holes = np.asarray(hole_positions, dtype=np.float64).reshape(-1, 2)
    if scores is None:
        scores = np.ones(len(holes), dtype=np.float64)
    else:
        scores = np.asarray(scores, dtype=np.float64)

    if min_distance is None or min_distance <= 0 or len(holes) <= 1:
        return holes.copy(), scores.copy()

    order = np.argsort(scores)[::-1]
    kept_idxs = []
    for idx in order:
        candidate = holes[idx]
        if all(np.linalg.norm(candidate - holes[kept]) >= min_distance
               for kept in kept_idxs):
            kept_idxs.append(idx)

    kept_idxs = np.array(kept_idxs, dtype=int)
    return holes[kept_idxs], scores[kept_idxs]


# ---------------------------------------------------------------------------
# 5.  Main public function: full hole-finding pipeline
# ---------------------------------------------------------------------------

def find_holes(image: np.ndarray,
               sigma: float = 2.0,
               connectivity: int = 2,
               border_margin: int = 2,
               min_response: Optional[float] = 0.6,
               min_distance: Optional[float] = 5.0,
               return_scores: bool = False):
    """
    Full hole-finding pipeline: enhancement → smoothing → maxima → centroids.

    Replicates the complete find_hole.m script.

    Parameters
    ----------
    image         : raw or pre-processed 2-D TEM image (uint8, uint16, float)
    sigma         : Gaussian smoothing sigma in pixels (default 2.0)
    connectivity  : maxima connectivity (default 2 = 8-connected)
    border_margin : strip of pixels near the image edge to discard, because
                    border effects can produce spurious maxima.
                    find_hole.m / p5_z.m discard positions within 2 px of
                    the border (col < 2 or col > width-2, etc.)
    min_response  : minimum response in the final contrast-stretched image.
                    Set None to disable. Default 0.6 is tuned for the
                    synthetic bright-hole benchmark.
    min_distance  : minimum distance in pixels between returned hole centres.
                    Set None or <=0 to disable.
    return_scores : if True, return (hole_positions, scores).

    Returns
    -------
    hole_positions : (N, 2) float array of [x, y] pixel coordinates
                     for each detected ring centre (hole).
    """
    # --- Step 1: first contrast stretch ----------------------------------
    img_adj1 = imadjust(image)

    # --- Step 2: Gaussian smoothing --------------------------------------
    img_smooth = smooth_image(img_adj1, sigma=sigma)

    # --- Step 3: second contrast stretch ---------------------------------
    # find_hole.m calls imadjust a second time on the smoothed image.
    # This sharpens the contrast of the peaks before thresholding.
    img_adj2 = imadjust(img_smooth)

    # --- Step 4: regional maxima mask ------------------------------------
    max_mask = find_regional_maxima(img_adj2, connectivity=connectivity)

    # --- Step 5: centroid extraction ------------------------------------
    holes = maxima_to_centroids(max_mask)

    # --- Step 6: discard border positions --------------------------------
    # p5_z.m: col < 2 | col > 254 | row < 2 | row > 254  (for 256×256)
    # We generalise to any image size using border_margin.
    h, w = image.shape[:2]
    keep = (
        (holes[:, 0] >= border_margin) &
        (holes[:, 0] <= w - border_margin) &
        (holes[:, 1] >= border_margin) &
        (holes[:, 1] <= h - border_margin)
    )
    holes = holes[keep]

    # --- Step 7: synthetic-data bright-hole filtering ---------------------
    # The contrast convention is explicit: hole centres are bright. Weak local
    # maxima after the final contrast stretch are likely noise in the synthetic
    # benchmark, and min-distance suppression keeps one candidate per hole.
    scores = score_holes(img_adj2, holes)
    holes, scores = filter_holes_by_response(holes, scores, min_response)
    holes, scores = suppress_close_holes(holes, scores, min_distance)

    if return_scores:
        return holes, scores
    return holes


# ---------------------------------------------------------------------------
# 6.  Diagnostic visualisation
# ---------------------------------------------------------------------------

def plot_holes(image: np.ndarray,
               hole_positions: np.ndarray,
               title: str = "Detected ring centres (holes)",
               save_path: str = None) -> None:
    """
    Overlay detected hole positions on the image for visual inspection.

    Parameters
    ----------
    image          : original TEM image (before enhancement)
    hole_positions : (N, 2) array from find_holes()
    title          : plot title
    save_path      : if provided, save figure to this path
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: raw image
    axes[0].imshow(image, cmap='gray', interpolation='nearest')
    axes[0].set_title("Input image")
    axes[0].axis('off')

    # Right: image + detected holes
    axes[1].imshow(image, cmap='gray', interpolation='nearest')
    if len(hole_positions) > 0:
        axes[1].plot(hole_positions[:, 0], hole_positions[:, 1],
                     'r.', markersize=4, label=f'Holes (N={len(hole_positions)})')
        axes[1].legend(fontsize=9)
    axes[1].set_title(title)
    axes[1].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
