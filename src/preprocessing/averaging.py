"""
Temporal averaging for noise reduction in time-series imaging.

Based on methodology from:
Zhang et al., "Revealing 3D Atomic Dynamics of Graphene via High-Speed 
Low-Dose TEM Imaging" (2025, in preparation)

This implements the temporal averaging approach where 5 consecutive frames
were averaged to improve SNR while maintaining millisecond temporal resolution.

Key parameters from paper:
- Frame rate: 1ms intervals
- Window size: 5 frames
- Effective temporal resolution: 5ms
- Dose per frame: ~8×10³ e⁻/Å²
"""

import numpy as np
from typing import List, Optional
from tqdm import tqdm


class TemporalAverager:
    """
    Temporal averaging for low-dose sequential imaging.
    
    Reduces noise by averaging consecutive frames, trading some temporal
    resolution for improved signal-to-noise ratio (SNR).
    
    Attributes
    ----------
    window_size : int
        Number of frames to average (default: 5, as in paper)
    mode : str
        Averaging mode ('uniform' for equal weights, as in paper)
        
    Examples
    --------
    >>> averager = TemporalAverager(window_size=5)
    >>> images = [load_image(f) for f in range(50)]  # Load sequence
    >>> averaged_img = averager.average_sequence(images, target_idx=25)
    >>> print(f"SNR improved by {averager.snr_improvement:.2f}x")
    
    Notes
    -----
    This is a standard technique in time-resolved imaging, particularly
    effective for low-dose electron microscopy. The paper used this as
    the first step in the preprocessing pipeline (see Fig 2a).
    """
    
    def __init__(self, window_size: int = 5, mode: str = 'uniform'):
        """
        Initialize temporal averager.
        
        Parameters
        ----------
        window_size : int
            Number of consecutive frames to average
        mode : str
            Averaging mode: 'uniform' (equal weights)
        """
        self.window_size = window_size
        self.mode = mode
        self.snr_improvement = None
        self._validate_params()
    
    def _validate_params(self):
        """Validate initialization parameters"""
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if self.window_size % 2 == 0:
            print(f"Warning: Even window_size ({self.window_size}) may cause "
                  f"temporal misalignment. Odd numbers recommended.")
    
    def average_sequence(self,
                        images: List[np.ndarray],
                        target_idx: int) -> np.ndarray:
        """
        Average frames surrounding target frame.
        
        This implements the averaging shown in Fig 2a of the manuscript,
        where frames surrounding the target are averaged to create a
        cleaner initial estimate.
        
        Parameters
        ----------
        images : list of np.ndarray
            Sequence of images, each shape (H, W)
        target_idx : int
            Index of target frame (becomes center of averaging window)
            
        Returns
        -------
        averaged : np.ndarray
            Averaged image, same shape as input images
            
        Raises
        ------
        ValueError
            If target_idx is outside valid range
            
        Examples
        --------
        >>> images = [np.random.rand(256, 256) for _ in range(20)]
        >>> averager = TemporalAverager(window_size=5)
        >>> result = averager.average_sequence(images, target_idx=10)
        >>> print(result.shape)  # (256, 256)
        """
        n_frames = len(images)
        
        # Validate input
        if target_idx < 0 or target_idx >= n_frames:
            raise ValueError(
                f"target_idx {target_idx} out of range [0, {n_frames})"
            )
        
        # Determine window boundaries
        half_window = self.window_size // 2
        start_idx = max(0, target_idx - half_window)
        end_idx = min(n_frames, target_idx + half_window + 1)
        
        # Extract frames in window
        frames = [images[i] for i in range(start_idx, end_idx)]
        n_frames_actual = len(frames)
        
        # Compute weights
        weights = self._compute_weights(n_frames_actual)
        
        # Weighted average
        averaged = np.zeros_like(frames[0], dtype=np.float64)
        for frame, weight in zip(frames, weights):
            averaged += weight * frame.astype(np.float64)
        
        # Estimate SNR improvement
        if len(frames) > 1:
            self.snr_improvement = self._estimate_snr_improvement(
                images[target_idx], averaged
            )
        
        return averaged
    
    def _compute_weights(self, n_frames: int) -> np.ndarray:
        """
        Compute averaging weights based on mode.
        
        Parameters
        ----------
        n_frames : int
            Number of frames to weight
            
        Returns
        -------
        weights : np.ndarray
            Normalized weights for each frame
        """
        if self.mode == 'uniform':
            # Equal weights (as used in paper)
            weights = np.ones(n_frames) / n_frames
        else:
            raise ValueError(f"Unknown averaging mode: {self.mode}")
        
        return weights
    
    def _estimate_snr_improvement(self, 
                                  original: np.ndarray,
                                  averaged: np.ndarray) -> float:
        """
        Estimate SNR improvement from averaging.
        
        Uses robust estimation: signal is low-frequency component,
        noise is high-frequency component.
        """
        from scipy import ndimage
        
        # Estimate signal (smooth, low-frequency)
        signal_orig = ndimage.gaussian_filter(original.astype(np.float64), sigma=2)
        signal_avg = ndimage.gaussian_filter(averaged, sigma=2)
        
        # Estimate noise (high-frequency residual)
        noise_orig = original.astype(np.float64) - signal_orig
        noise_avg = averaged - signal_avg
        
        # Calculate SNR = signal_mean / noise_std
        snr_orig = np.mean(np.abs(signal_orig)) / (np.std(noise_orig) + 1e-10)
        snr_avg = np.mean(np.abs(signal_avg)) / (np.std(noise_avg) + 1e-10)
        
        return snr_avg / snr_orig
    
    def process_full_sequence(self,
                             images: List[np.ndarray],
                             show_progress: bool = True) -> List[np.ndarray]:
        """
        Apply temporal averaging to entire image sequence.
        
        Each frame becomes the center of an averaging window, enabling
        noise reduction across the entire sequence while maintaining
        temporal sampling.
        
        Parameters
        ----------
        images : list of np.ndarray
            Full image sequence
        show_progress : bool
            Whether to display progress bar
            
        Returns
        -------
        averaged_sequence : list of np.ndarray
            Averaged image for each frame
            
        Examples
        --------
        >>> images = [load_frame(i) for i in range(100)]
        >>> averager = TemporalAverager(window_size=5)
        >>> averaged_all = averager.process_full_sequence(images)
        >>> # Now each frame in averaged_all has reduced noise
        """
        n_frames = len(images)
        averaged = []
        
        iterator = range(n_frames)
        if show_progress:
            iterator = tqdm(iterator, desc="Temporal averaging", unit="frame")
        
        for idx in iterator:
            avg_frame = self.average_sequence(images, idx)
            averaged.append(avg_frame)
        
        return averaged


# Utility functions

def estimate_noise_std(image: np.ndarray) -> float:
    """
    Estimate noise standard deviation using robust method.
    
    Uses Median Absolute Deviation (MAD) on high-frequency component.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
        
    Returns
    -------
    sigma : float
        Estimated noise standard deviation
    """
    from scipy import ndimage
    
    # High-frequency component (noise dominated)
    laplacian = ndimage.laplace(image.astype(np.float64))
    
    # Robust estimate using MAD
    mad = np.median(np.abs(laplacian - np.median(laplacian)))
    sigma = 1.4826 * mad  # MAD to std conversion
    
    return sigma


if __name__ == '__main__':
    """Test temporal averaging module"""
    
    print("Temporal Averaging Module - Test")
    print("=" * 60)
    
    # Create synthetic test sequence
    print("\n1. Generating synthetic test sequence...")
    n_frames = 20
    img_size = 256
    
    # Base structure
    y, x = np.mgrid[0:img_size, 0:img_size]
    base_image = np.zeros((img_size, img_size))
    
    # Add "atomic columns"
    np.random.seed(42)
    for i in range(15):
        cx, cy = np.random.randint(50, img_size-50, 2)
        base_image += np.exp(-((x-cx)**2 + (y-cy)**2) / (2*5**2))
    
    # Create noisy sequence (simulating low-dose imaging)
    images = []
    for i in range(n_frames):
        # Poisson noise (shot noise)
        noisy = np.random.poisson(base_image * 100 + 50).astype(float)
        # Gaussian readout noise
        noisy += np.random.normal(0, 5, noisy.shape)
        images.append(noisy)
    
    print(f"  ✓ Created {n_frames} frames of size {img_size}×{img_size}")
    
    # Test averaging
    print("\n2. Testing temporal averaging...")
    averager = TemporalAverager(window_size=5)
    target_idx = n_frames // 2
    
    print(f"  - Target frame: {target_idx}")
    print(f"  - Window size: {averager.window_size}")
    
    averaged = averager.average_sequence(images, target_idx)
    
    print(f"\n3. Results:")
    print(f"  - SNR improvement: {averager.snr_improvement:.2f}x")
    
    noise_orig = estimate_noise_std(images[target_idx])
    noise_avg = estimate_noise_std(averaged)
    noise_reduction = (1 - noise_avg/noise_orig) * 100
    
    print(f"  - Noise std (original): {noise_orig:.2f}")
    print(f"  - Noise std (averaged): {noise_avg:.2f}")
    print(f"  - Noise reduction: {noise_reduction:.1f}%")
    
    print("\n" + "=" * 60)
    print("✓ Temporal averaging module test completed successfully!")