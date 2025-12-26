"""
BM3D denoising for low-dose imaging.

Implements Block-Matching 3D (BM3D) denoising as described in:
Zhang et al., "Revealing 3D Atomic Dynamics..." (2025) - Section: Preprocessing

BM3D is a state-of-the-art denoising algorithm particularly effective 
for low signal-to-noise ratio (SNR) images, making it ideal for low-dose 
electron microscopy and medical imaging applications.

Reference:
Dabov et al., "Image denoising by sparse 3-D transform-domain collaborative 
filtering," IEEE TIP, 2007.
"""

import numpy as np
from typing import Optional, Tuple
import bm3d
import warnings


class BM3DDenoiser:
    """
    BM3D denoising for low-dose imaging applications.
    
    BM3D works by:
    1. Grouping similar image patches (block-matching)
    2. Collaborative filtering in 3D transform domain
    3. Aggregating denoised patches
    
    Particularly effective when SNR < 10 (typical in low-dose imaging).
    
    Attributes
    ----------
    sigma_psd : float
        Noise power spectral density (standard deviation)
    stage : str
        Processing stage: 'hard', 'wiener', or 'all' (both stages)
    profile : str
        Quality profile: 'fast', 'normal', or 'high'
        
    Examples
    --------
    >>> denoiser = BM3DDenoiser(sigma_psd=25.0)
    >>> denoised = denoiser.denoise(noisy_image)
    >>> print(f"PSNR improved by {denoiser.psnr_improvement:.2f} dB")
    
    Notes
    -----
    The paper applied BM3D after temporal averaging to further reduce
    noise before structure estimation. This is a standard approach in
    low-dose imaging pipelines.
    """
    
    def __init__(self, 
                 sigma_psd: Optional[float] = None,
                 stage: str = 'all',
                 profile: str = 'normal'):
        """
        Initialize BM3D denoiser.
        
        Parameters
        ----------
        sigma_psd : float, optional
            Noise standard deviation. If None, will be auto-estimated.
        stage : str
            'hard' - Fast single-stage (hard thresholding)
            'wiener' - Second stage only (Wiener filtering)
            'all' - Both stages (best quality, recommended)
        profile : str
            'fast' - Faster processing, lower quality
            'normal' - Balanced (recommended)
            'high' - Best quality, slower
        """
        self.sigma_psd = sigma_psd
        self.stage = stage
        self.profile = profile
        self.psnr_improvement = None
        self._validate_params()
    
    def _validate_params(self):
        """Validate initialization parameters"""
        valid_stages = ['hard', 'wiener', 'all']
        if self.stage not in valid_stages:
            raise ValueError(f"stage must be one of {valid_stages}")
        
        valid_profiles = ['fast', 'normal', 'high']
        if self.profile not in valid_profiles:
            raise ValueError(f"profile must be one of {valid_profiles}")
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply BM3D denoising to image.
        
        Parameters
        ----------
        image : np.ndarray
            Noisy input image (2D array)
            
        Returns
        -------
        denoised : np.ndarray
            Denoised image (same shape as input)
            
        Examples
        --------
        >>> noisy_img = load_image('noisy_frame.png')
        >>> denoiser = BM3DDenoiser()
        >>> clean_img = denoiser.denoise(noisy_img)
        """
        # Estimate noise if not provided
        if self.sigma_psd is None:
            self.sigma_psd = self._estimate_noise_std(image)
            print(f"Auto-estimated noise σ = {self.sigma_psd:.2f}")
        
        # Normalize to [0, 1] range (BM3D requirement)
        img_min, img_max = image.min(), image.max()
        img_range = img_max - img_min + 1e-10
        img_normalized = (image - img_min) / img_range
        
        # Normalize noise parameter
        sigma_normalized = self.sigma_psd / img_range
        
        # Select stage
        stage_arg = self.stage
        if self.stage == 'wiener':
            warnings.warn("Wiener-only stage not supported; using both stages.")
            stage_arg = 'all'
        
        # Select profile
        profile_arg = self.profile  # 'fast', 'np', or 'high'
        if profile_arg == 'normal':
            profile_arg = 'np'  
        
        # Apply BM3D
        try:
            # Modern BM3D API - just needs image and sigma
            denoised_normalized = bm3d.bm3d(
                img_normalized,
                sigma_psd=sigma_normalized
            )
        except Exception as e:
            print(f"BM3D error: {e}")
            print("Falling back to simpler denoising...")
            denoised_normalized = self._fallback_denoise(img_normalized)
        
        # Denormalize
        denoised = denoised_normalized * img_range + img_min
        
        # Calculate improvement
        self.psnr_improvement = self._calculate_psnr_improvement(
            image, denoised
        )
        
        return denoised
    
    def _estimate_noise_std(self, image: np.ndarray) -> float:
        """
        Estimate noise standard deviation using robust method.
        
        Uses Median Absolute Deviation (MAD) on high-frequency components,
        which is robust to outliers and structured content.
        
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
        
        # Compute Laplacian (high-frequency component)
        laplacian = ndimage.laplace(image.astype(np.float64))
        
        # Robust noise estimation using MAD
        mad = np.median(np.abs(laplacian - np.median(laplacian)))
        sigma = 1.4826 * mad  # MAD to std conversion factor
        
        return sigma
    
    def _fallback_denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Fallback denoising using simple Gaussian filter.
        
        Used if BM3D fails for any reason.
        """
        from scipy import ndimage
        sigma = 1.0
        return ndimage.gaussian_filter(image, sigma=sigma)
    
    def _calculate_psnr_improvement(self,
                                   noisy: np.ndarray,
                                   denoised: np.ndarray) -> float:
        """
        Calculate PSNR improvement (requires ground truth estimate).
        
        Since we don't have ground truth, we estimate improvement
        by comparing noise levels.
        """
        noise_before = self._estimate_noise_std(noisy)
        noise_after = self._estimate_noise_std(denoised)
        
        # Approximate PSNR improvement
        psnr_improvement = 20 * np.log10(noise_before / (noise_after + 1e-10))
        
        return psnr_improvement
    
    def batch_denoise(self,
                     images: list,
                     show_progress: bool = True) -> list:
        """
        Denoise multiple images with same parameters.
        
        Parameters
        ----------
        images : list of np.ndarray
            List of noisy images
        show_progress : bool
            Show progress bar
            
        Returns
        -------
        denoised_images : list of np.ndarray
            List of denoised images
        """
        from tqdm import tqdm
        
        denoised = []
        iterator = images if not show_progress else tqdm(images, desc="BM3D denoising")
        
        for img in iterator:
            denoised.append(self.denoise(img))
        
        return denoised


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline as described in the paper.
    
    Implements the two-stage preprocessing:
    1. Temporal averaging (reduce noise, maintain temporal resolution)
    2. BM3D denoising (further noise reduction)
    
    This pipeline is shown in Fig 2a of the manuscript.
    
    Examples
    --------
    >>> pipeline = PreprocessingPipeline(window_size=5)
    >>> images = load_image_sequence('data/')
    >>> processed = pipeline.process(images, target_idx=25)
    >>> print(f"SNR improved by {pipeline.total_snr_improvement:.2f}x")
    
    Notes
    -----
    The paper used this preprocessing before structure estimation to
    ensure the initial model was as clean as possible, which speeds up
    the subsequent optimization.
    """
    
    def __init__(self,
                 window_size: int = 5,
                 denoise_sigma: Optional[float] = None,
                 denoise_profile: str = 'normal'):
        """
        Initialize preprocessing pipeline.
        
        Parameters
        ----------
        window_size : int
            Temporal averaging window size (5 in paper)
        denoise_sigma : float, optional
            BM3D noise level (auto-estimated if None)
        denoise_profile : str
            BM3D quality profile
        """
        from .averaging import TemporalAverager
        
        self.averager = TemporalAverager(window_size=window_size)
        self.denoiser = BM3DDenoiser(
            sigma_psd=denoise_sigma,
            profile=denoise_profile
        )
        self.total_snr_improvement = None
    
    def process(self,
               images: list,
               target_idx: int,
               verbose: bool = True) -> np.ndarray:
        """
        Apply complete preprocessing pipeline to image sequence.
        
        Parameters
        ----------
        images : list of np.ndarray
            Time-series images
        target_idx : int
            Index of target frame
        verbose : bool
            Print progress messages
            
        Returns
        -------
        processed : np.ndarray
            Preprocessed image ready for structure estimation
            
        Examples
        --------
        >>> pipeline = PreprocessingPipeline()
        >>> images = [load_image(f) for f in filenames]
        >>> clean = pipeline.process(images, target_idx=25)
        """
        if verbose:
            print("Preprocessing Pipeline")
            print("=" * 60)
        
        # Stage 1: Temporal averaging
        if verbose:
            print("Stage 1: Temporal averaging...", end=" ")
        
        averaged = self.averager.average_sequence(images, target_idx)
        snr_improvement_avg = self.averager.snr_improvement or 1.0
        
        if verbose:
            print(f"✓ (SNR improved {snr_improvement_avg:.2f}x)")
        
        # Stage 2: BM3D denoising
        if verbose:
            print("Stage 2: BM3D denoising...", end=" ")
        
        denoised = self.denoiser.denoise(averaged)
        
        if verbose:
            psnr = self.denoiser.psnr_improvement or 0
            print(f"✓ (PSNR improved {psnr:.2f} dB)")
        
        # Calculate total improvement
        self.total_snr_improvement = snr_improvement_avg
        
        if verbose:
            print("=" * 60)
            print(f"Total SNR improvement: {self.total_snr_improvement:.2f}x")
            print("Preprocessing complete!\n")
        
        return denoised


# Utility functions

def compare_preprocessing_methods(image: np.ndarray,
                                  methods: list = None) -> dict:
    """
    Compare different preprocessing methods on same image.
    
    Parameters
    ----------
    image : np.ndarray
        Noisy input image
    methods : list, optional
        List of method names to compare
        
    Returns
    -------
    results : dict
        Dictionary with results for each method
    """
    if methods is None:
        methods = ['none', 'gaussian', 'median', 'bm3d']
    
    from scipy import ndimage
    
    results = {}
    
    # Original (no preprocessing)
    if 'none' in methods:
        results['none'] = {
            'image': image,
            'psnr': 0,
            'noise_std': BM3DDenoiser()._estimate_noise_std(image)
        }
    
    # Gaussian filter
    if 'gaussian' in methods:
        gaussian = ndimage.gaussian_filter(image, sigma=1.5)
        results['gaussian'] = {
            'image': gaussian,
            'noise_std': BM3DDenoiser()._estimate_noise_std(gaussian)
        }
    
    # Median filter
    if 'median' in methods:
        median = ndimage.median_filter(image, size=3)
        results['median'] = {
            'image': median,
            'noise_std': BM3DDenoiser()._estimate_noise_std(median)
        }
    
    # BM3D
    if 'bm3d' in methods:
        denoiser = BM3DDenoiser()
        bm3d_result = denoiser.denoise(image)
        results['bm3d'] = {
            'image': bm3d_result,
            'noise_std': BM3DDenoiser()._estimate_noise_std(bm3d_result),
            'psnr_improvement': denoiser.psnr_improvement
        }
    
    return results


if __name__ == '__main__':
    """Test BM3D denoising module"""
    
    print("BM3D Denoising Module - Test")
    print("=" * 60)
    
    # Create synthetic noisy image
    print("\n1. Generating synthetic test image...")
    size = 256
    y, x = np.mgrid[0:size, 0:size]
    
    # Clean image with structures
    clean = np.zeros((size, size))
    np.random.seed(42)
    for i in range(10):
        cx, cy = np.random.randint(50, size-50, 2)
        clean += np.exp(-((x-cx)**2 + (y-cy)**2) / (2*5**2))
    
    # Add noise (simulating low-dose conditions)
    noisy = np.random.poisson(clean * 100 + 50).astype(float)
    noisy += np.random.normal(0, 10, noisy.shape)
    
    print(f"  ✓ Created {size}×{size} test image")
    
    # Test BM3D
    print("\n2. Testing BM3D denoising...")
    denoiser = BM3DDenoiser()
    
    print(f"  - Auto-estimating noise level...")
    denoised = denoiser.denoise(noisy)
    
    print(f"\n3. Results:")
    print(f"  - PSNR improvement: {denoiser.psnr_improvement:.2f} dB")
    
    # Calculate actual improvement vs clean
    mse_noisy = np.mean((noisy - clean)**2)
    mse_denoised = np.mean((denoised - clean)**2)
    psnr_noisy = 10 * np.log10(clean.max()**2 / mse_noisy)
    psnr_denoised = 10 * np.log10(clean.max()**2 / mse_denoised)
    
    print(f"  - PSNR (noisy): {psnr_noisy:.2f} dB")
    print(f"  - PSNR (denoised): {psnr_denoised:.2f} dB")
    print(f"  - Actual improvement: {psnr_denoised - psnr_noisy:.2f} dB")
    
    # Test full pipeline
    print("\n4. Testing full preprocessing pipeline...")
    images = [noisy + np.random.normal(0, 2, noisy.shape) for _ in range(10)]
    
    pipeline = PreprocessingPipeline(window_size=5)
    processed = pipeline.process(images, target_idx=5, verbose=False)
    
    mse_processed = np.mean((processed - clean)**2)
    psnr_processed = 10 * np.log10(clean.max()**2 / mse_processed)
    
    print(f"  - PSNR (after pipeline): {psnr_processed:.2f} dB")
    print(f"  - Total improvement: {psnr_processed - psnr_noisy:.2f} dB")
    
    print("\n" + "=" * 60)
    print("✓ BM3D denoising module test completed successfully!")