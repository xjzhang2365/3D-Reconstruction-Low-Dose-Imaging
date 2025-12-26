"""
Structure estimation module.

Implements estimation pipeline from Zhang et al. (2025):
1. Gaussian fitting - Localize atomic columns (x, y)
2. PCD method - Estimate z-heights from intensities
3. Bayesian refinement - Refine positions with uncertainties

Note: Bayesian implementation is conceptual framework only.
Full details pending publication.

Example Usage
-------------
>>> from src.estimation import StructureEstimator
>>> 
>>> estimator = StructureEstimator()
>>> structure = estimator.estimate(preprocessed_image)
>>> 
>>> print(f"Detected {structure['n_atoms']} atoms")
>>> positions_3d = structure['positions_3d']  # (N, 3) array
"""

from .gaussian_fitting import GaussianFitter, gaussian_2d, fit_lattice_model
from .bayesian_estimation import (
    BayesianEstimator, 
    PCDMethod, 
    StructureEstimator
)

__version__ = "1.0.0"

__all__ = [
    # Main classes
    'GaussianFitter',
    'BayesianEstimator',
    'PCDMethod',
    'StructureEstimator',
    
    # Utility functions
    'gaussian_2d',
    'fit_lattice_model',
]