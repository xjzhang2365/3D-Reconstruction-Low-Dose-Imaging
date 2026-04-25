"""
corrections.py
==============
Flat-field correction and dead-pixel removal for raw TEM frames.

Paper reference (Methods):
  "Raw frames underwent flat-field correction (Gaussian background estimation,
   σ = 20 pixels for 256×256 frames; subtracted), dead-pixel removal (> 5σ from
   local mean; replaced by eight-neighbour mean)."
"""

import numpy as np
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Flat-field correction
# ---------------------------------------------------------------------------

def correct_flat_field(
    frame: np.ndarray,
    sigma: float = 20.0,
) -> np.ndarray:
    """Remove low-frequency background shading from a single TEM frame.

    The background is estimated by heavily blurring the image with a Gaussian
    filter (large σ captures only slow illumination gradients, not atomic
    features).  Subtracting this from the raw frame leaves only high-frequency
    structural contrast.

    Parameters
    ----------
    frame : np.ndarray
        2-D array of raw pixel intensities (float or int).
    sigma : float
        Standard deviation of the Gaussian used for background estimation.
        Default 20 px matches the paper (256×256 frame size).

    Returns
    -------
    corrected : np.ndarray
        Background-subtracted frame, same shape as input, dtype float64.
        Values are shifted so the minimum is zero.

    How it works
    ------------
    Raw TEM frames have a smooth, spatially varying brightness caused by
    non-uniform illumination and detector response.  A Gaussian blur with
    σ >> atomic spacing (< 2 px) acts as a low-pass filter — it captures
    the illumination envelope while completely blurring out atomic features.
    Subtracting this envelope isolates the structural signal.

    Example
    -------
    >>> import numpy as np
    >>> frame = np.random.rand(256, 256).astype(np.float64)
    >>> corrected = correct_flat_field(frame, sigma=20.0)
    >>> corrected.shape
    (256, 256)
    """
    frame = np.asarray(frame, dtype=np.float64)

    # Estimate background: slow illumination envelope
    background = gaussian_filter(frame, sigma=sigma)

    # Subtract background, shift minimum to zero
    corrected = frame - background
    corrected -= corrected.min()

    return corrected


def correct_flat_field_stack(
    frames: np.ndarray,
    sigma: float = 20.0,
) -> np.ndarray:
    """Apply flat-field correction to every frame in a stack.

    Parameters
    ----------
    frames : np.ndarray
        3-D array of shape (N, H, W).
    sigma : float
        Gaussian σ passed to :func:`correct_flat_field`.

    Returns
    -------
    np.ndarray
        Corrected stack, shape (N, H, W), dtype float64.
    """
    return np.stack(
        [correct_flat_field(f, sigma=sigma) for f in frames],
        axis=0,
    )


# ---------------------------------------------------------------------------
# Dead-pixel removal
# ---------------------------------------------------------------------------

def _eight_neighbour_mean(frame: np.ndarray, row: int, col: int) -> float:
    """Return the mean of the up-to-8 valid neighbours of pixel (row, col).

    Pixels outside the frame boundary are excluded from the average.
    """
    H, W = frame.shape
    neighbours = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue          # skip the pixel itself
            r, c = row + dr, col + dc
            if 0 <= r < H and 0 <= c < W:
                neighbours.append(frame[r, c])
    return float(np.mean(neighbours)) if neighbours else 0.0


def remove_dead_pixels(
    frame: np.ndarray,
    threshold_sigma: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect and replace dead (hot/cold) pixels by 8-neighbour interpolation.

    A pixel is flagged as dead if its value deviates from the local mean by
    more than ``threshold_sigma`` × local standard deviation.  Here "local"
    means the global frame statistics — consistent with the paper's method.

    Parameters
    ----------
    frame : np.ndarray
        2-D array of pixel intensities (float64 recommended).
    threshold_sigma : float
        Detection threshold in units of standard deviations.  Default 5.0
        matches the paper.

    Returns
    -------
    cleaned : np.ndarray
        Frame with dead pixels replaced.
    dead_mask : np.ndarray
        Boolean mask, True where a pixel was identified as dead.

    How it works
    ------------
    Dead pixels fall into two categories:
      - Hot pixels: stuck at a very high value (detector saturation)
      - Cold pixels: stuck at a very low value (non-responsive element)
    Both are detected as outliers relative to the frame mean and standard
    deviation.  Replacing with the local 8-neighbour mean preserves edge
    continuity better than simple median replacement.

    Example
    -------
    >>> frame = np.ones((64, 64), dtype=np.float64)
    >>> frame[10, 10] = 999.0          # inject a hot pixel
    >>> cleaned, mask = remove_dead_pixels(frame, threshold_sigma=5.0)
    >>> mask[10, 10]
    True
    >>> abs(cleaned[10, 10] - 1.0) < 0.01
    True
    """
    frame = np.asarray(frame, dtype=np.float64)
    mean = frame.mean()
    std  = frame.std()

    # Flag outliers
    dead_mask = np.abs(frame - mean) > threshold_sigma * std

    cleaned = frame.copy()
    dead_rows, dead_cols = np.where(dead_mask)
    for r, c in zip(dead_rows, dead_cols):
        cleaned[r, c] = _eight_neighbour_mean(cleaned, r, c)

    return cleaned, dead_mask


def remove_dead_pixels_stack(
    frames: np.ndarray,
    threshold_sigma: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply dead-pixel removal to every frame in a stack.

    Parameters
    ----------
    frames : np.ndarray
        Shape (N, H, W).
    threshold_sigma : float
        Passed to :func:`remove_dead_pixels`.

    Returns
    -------
    cleaned_stack : np.ndarray
        Shape (N, H, W).
    mask_stack : np.ndarray
        Boolean array, shape (N, H, W).
    """
    cleaned_list, mask_list = [], []
    for f in frames:
        cleaned, mask = remove_dead_pixels(f, threshold_sigma)
        cleaned_list.append(cleaned)
        mask_list.append(mask)
    return np.stack(cleaned_list, axis=0), np.stack(mask_list, axis=0)
