"""
Preprocessing module for low-dose imaging.

Implements the preprocessing pipeline from Zhang et al. (2025):
1. Temporal averaging - Reduce noise across time series
2. BM3D denoising - Advanced spatial denoising

These preprocessing steps are critical for:
- Improving initial structure estimation
- Speeding up optimization convergence
- Achieving better final reconstruction accuracy

Example Usage
-------------
>>> from src.preprocessing import PreprocessingPipeline
>>> 
>>> # Load image sequence
>>> images = [load_image(f) for f in image_files]
>>> 
>>> # Create pipeline with paper parameters
>>> pipeline = PreprocessingPipeline(window_size=5)
>>> 
>>> # Process target frame
>>> cleaned = pipeline.process(images, target_idx=25)
>>> 
>>> # Now ready for structure estimation
>>> initial_structure = estimate_structure(cleaned)
"""

from .averaging import TemporalAverager, estimate_noise_std
from .denoising import BM3DDenoiser, PreprocessingPipeline, compare_preprocessing_methods

__version__ = "1.0.0"

__all__ = [
    # Main classes
    'TemporalAverager',
    'BM3DDenoiser',
    'PreprocessingPipeline',
    
    # Utility functions
    'estimate_noise_std',
    'compare_preprocessing_methods',
]

# Quick reference for paper parameters
PAPER_PARAMETERS = {
    'temporal_window': 5,  # frames
    'frame_interval': 1,   # milliseconds
    'electron_dose': 8e3,  # e-/Å²
    'imaging_voltage': 80, # kV
}