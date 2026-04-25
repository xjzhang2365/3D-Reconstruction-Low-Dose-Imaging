"""
averaging.py
============
Temporal averaging of TEM image sequences.

Paper reference (Methods):
  "BM3D denoising and five-frame averaging for initial model estimation."
  "The target image was pre-processed by averaging over five consecutive
   images."

The key design choice: we average a symmetric window of frames *around* the
target frame (not a causal window), because all frames are available
off-line.  This minimises motion blur relative to the target while maximising
noise reduction.
"""

import numpy as np


def temporal_average(
    frames: np.ndarray,
    target_idx: int,
    window_size: int = 5,
) -> np.ndarray:
    """Average a symmetric window of frames centred on ``target_idx``.

    Parameters
    ----------
    frames : np.ndarray
        Stack of TEM images, shape (N, H, W), already preprocessed
        (flat-field corrected, dead pixels removed, denoised).
    target_idx : int
        Index of the frame to reconstruct.  The window is centred here.
    window_size : int
        Total number of frames to include in the average (must be odd
        for a symmetric window).  Default 5 matches the paper.

    Returns
    -------
    averaged : np.ndarray
        Single averaged frame, shape (H, W), dtype float64.

    Raises
    ------
    ValueError
        If ``window_size`` is even or ``target_idx`` is out of bounds.

    How it works
    ------------
    Shot noise in TEM images follows Poisson statistics: variance ∝ signal.
    Averaging K independent frames reduces noise variance by a factor of K
    (standard error of the mean), improving SNR by √K.  For K=5 this gives
    a √5 ≈ 2.2× SNR improvement — critical when the raw SNR < 3.

    The trade-off is temporal blur: averaging across time smears out features
    that change between frames.  Using only 5 frames at 1 ms intervals limits
    this smearing to ~4 ms, acceptable for the slow ripple dynamics observed.
    The averaged result is used only for *initialising* the atomic model, not
    as the reconstruction target itself.

    Boundary behaviour
    ------------------
    If the window extends beyond the start or end of the stack, it is
    clipped to the available range.  A warning is printed in this case.

    Example
    -------
    >>> import numpy as np
    >>> frames = np.random.rand(10, 64, 64)
    >>> avg = temporal_average(frames, target_idx=4, window_size=5)
    >>> avg.shape
    (64, 64)
    """
    frames = np.asarray(frames, dtype=np.float64)
    N = frames.shape[0]

    if window_size % 2 == 0:
        raise ValueError(
            f"window_size must be odd for a symmetric window; got {window_size}."
        )
    if not (0 <= target_idx < N):
        raise ValueError(
            f"target_idx {target_idx} out of range for stack of {N} frames."
        )

    half = window_size // 2
    start = target_idx - half
    end   = target_idx + half + 1   # exclusive

    # Clip to valid range and warn if boundary is hit
    clipped_start = max(0, start)
    clipped_end   = min(N, end)
    if clipped_start != start or clipped_end != end:
        actual = clipped_end - clipped_start
        print(
            f"[temporal_average] Window clipped at boundary: "
            f"using {actual} frames instead of {window_size}."
        )

    window = frames[clipped_start:clipped_end]
    averaged = window.mean(axis=0)
    return averaged


def temporal_average_all(
    frames: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """Compute temporal averages for every frame in the stack.

    Useful for batch processing: returns a full averaged stack where
    averaged[i] = temporal_average(frames, target_idx=i, window_size).

    Parameters
    ----------
    frames : np.ndarray
        Shape (N, H, W).
    window_size : int
        Passed to :func:`temporal_average`.

    Returns
    -------
    np.ndarray
        Shape (N, H, W) — one averaged frame per target index.

    Example
    -------
    >>> frames = np.random.rand(10, 64, 64)
    >>> averaged_stack = temporal_average_all(frames, window_size=5)
    >>> averaged_stack.shape
    (10, 64, 64)
    """
    return np.stack(
        [temporal_average(frames, i, window_size) for i in range(len(frames))],
        axis=0,
    )
