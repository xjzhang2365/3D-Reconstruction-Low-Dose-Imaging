"""
preprocessing/
==============
Stage 1 of the TEM reconstruction pipeline.

Modules
-------
corrections   flat-field correction, dead-pixel removal
averaging     temporal averaging of frame sequences
denoising     BM3D, K-SVD, and U-Net denoising methods

Quick start
-----------
    from preprocessing import preprocess_stack, BM3DDenoiser

    # Full pipeline on a raw frame stack
    result = preprocess_stack(raw_frames, target_idx=25)
    averaged_frame = result['averaged']

    # Or run individual steps
    from preprocessing import correct_flat_field, temporal_average
"""

from .corrections import (
    correct_flat_field,
    correct_flat_field_stack,
    remove_dead_pixels,
    remove_dead_pixels_stack,
)
from .averaging import (
    temporal_average,
    temporal_average_all,
)
from .denoising import (
    BM3DDenoiser,
    KSVDDenoiser,
    UNetDenoiser,
    benchmark_denoisers,
)

import numpy as np


def preprocess_stack(
    raw_frames: np.ndarray,
    target_idx: int,
    flat_field_sigma: float = 20.0,
    dead_pixel_sigma: float = 5.0,
    denoiser=None,
    window_size: int = 5,
) -> dict:
    """Run the complete preprocessing pipeline on a raw frame stack.

    This is the top-level function that replicates the paper's Methods:
      1. Flat-field correction
      2. Dead-pixel removal
      3. Denoising (BM3D by default)
      4. Temporal averaging (5-frame window centred on target_idx)

    Parameters
    ----------
    raw_frames : np.ndarray
        Raw TEM image stack, shape (N, H, W).
    target_idx : int
        Index of the frame to reconstruct.
    flat_field_sigma : float
        Gaussian sigma for background estimation. Default 20 px (paper value).
    dead_pixel_sigma : float
        Detection threshold for dead pixels in sigma. Default 5 (paper value).
    denoiser : object or None
        Denoiser instance with a .denoise() method. Defaults to BM3DDenoiser().
    window_size : int
        Temporal averaging window. Default 5 (paper value).

    Returns
    -------
    dict with keys:
        'corrected'  : np.ndarray (N, H, W) -- after flat-field + dead pixel
        'denoised'   : np.ndarray (N, H, W) -- after denoising
        'averaged'   : np.ndarray (H, W)    -- final averaged frame
        'dead_masks' : np.ndarray (N, H, W) -- boolean dead-pixel locations
    """
    if denoiser is None:
        denoiser = BM3DDenoiser()

    corrected = correct_flat_field_stack(raw_frames, sigma=flat_field_sigma)
    cleaned, dead_masks = remove_dead_pixels_stack(corrected, dead_pixel_sigma)
    denoised = np.stack([denoiser.denoise(f) for f in cleaned], axis=0)
    averaged = temporal_average(denoised, target_idx=target_idx, window_size=window_size)

    return {
        'corrected':  corrected,
        'denoised':   denoised,
        'averaged':   averaged,
        'dead_masks': dead_masks,
    }


__all__ = [
    'correct_flat_field', 'correct_flat_field_stack',
    'remove_dead_pixels', 'remove_dead_pixels_stack',
    'temporal_average', 'temporal_average_all',
    'BM3DDenoiser', 'KSVDDenoiser', 'UNetDenoiser', 'benchmark_denoisers',
    'preprocess_stack',
]
