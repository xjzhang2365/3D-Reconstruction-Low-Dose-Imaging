"""
Gaussian fitting for atomic column localization.

Standard approach in electron microscopy: model each atomic column
as a 2D Gaussian peak. This determines x,y positions of atoms.

This is a well-established technique (not proprietary):
- Van Aert et al., Ultramicroscopy, 2005
- De Backer et al., Ultramicroscopy, 2013
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter, label
from skimage.feature import peak_local_max


def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, offset, theta=0):
    """
    2D Gaussian function.
    
    Parameters
    ----------
    coords : tuple of arrays
        (x, y) coordinate arrays
    amplitude : float
        Peak amplitude
    x0, y0 : float
        Center position
    sigma_x, sigma_y : float
        Gaussian widths
    offset : float
        Background offset
    theta : float
        Rotation angle (radians)
        
    Returns
    -------
    values : array
        Gaussian values at coordinates
    """
    x, y = coords
    
    # Rotate coordinates
    x_rot = (x - x0) * np.cos(theta) - (y - y0) * np.sin(theta)
    y_rot = (x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)
    
    # Gaussian
    gaussian = amplitude * np.exp(
        -(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2))
    )
    
    return gaussian + offset


class GaussianFitter:
    """
    Fit 2D Gaussians to atomic columns in TEM images.
    
    This is a standard technique in quantitative electron microscopy.
    Each atomic column is modeled as a 2D Gaussian peak, and fitting
    determines the (x, y) positions.
    
    Examples
    --------
    >>> fitter = GaussianFitter()
    >>> positions = fitter.fit_image(image)
    >>> print(f"Found {len(positions)} atoms")
    """
    
    def __init__(self, 
                 expected_sigma: float = 2.0,
                 min_distance: int = 5,
                 threshold_rel: float = 0.1):
        """
        Parameters
        ----------
        expected_sigma : float
            Expected Gaussian width (pixels)
        min_distance : int
            Minimum distance between peaks (pixels)
        threshold_rel : float
            Relative threshold for peak detection (0-1)
        """
        self.expected_sigma = expected_sigma
        self.min_distance = min_distance
        self.threshold_rel = threshold_rel
    
    def detect_peaks(self, image: np.ndarray) -> np.ndarray:
        """
        Detect atomic column positions using peak detection.
        
        Parameters
        ----------
        image : np.ndarray
            Input TEM image
            
        Returns
        -------
        peaks : np.ndarray
            Peak positions, shape (N, 2) with columns [y, x]
        """
        # Smooth image slightly
        smoothed = gaussian_filter(image, sigma=0.5)
        
        # Find local maxima
        peaks = peak_local_max(
            smoothed,
            min_distance=self.min_distance,
            threshold_rel=self.threshold_rel,
            exclude_border=True
        )
        
        return peaks
    
    def fit_single_peak(self,
                       image: np.ndarray,
                       initial_pos: Tuple[float, float],
                       window_size: int = 15) -> dict:
        """
        Fit single 2D Gaussian to peak.
        
        Parameters
        ----------
        image : np.ndarray
            Input image
        initial_pos : tuple
            Initial (y, x) position
        window_size : int
            Size of fitting window
            
        Returns
        -------
        params : dict
            Fitted parameters {'x': ..., 'y': ..., 'amplitude': ..., etc.}
        """
        y0, x0 = initial_pos
        
        # Extract window around peak
        half_size = window_size // 2
        y_min = max(0, int(y0 - half_size))
        y_max = min(image.shape[0], int(y0 + half_size + 1))
        x_min = max(0, int(x0 - half_size))
        x_max = min(image.shape[1], int(x0 + half_size + 1))
        
        window = image[y_min:y_max, x_min:x_max]
        
        # Create coordinate grid
        y_coords = np.arange(window.shape[0]) + y_min
        x_coords = np.arange(window.shape[1]) + x_min
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        
        # Initial guess
        amplitude_guess = window.max() - window.min()
        offset_guess = window.min()
        
        p0 = [
            amplitude_guess,
            x0, y0,
            self.expected_sigma, self.expected_sigma,
            offset_guess
        ]
        
        # Bounds
        bounds = (
            [0, x_min, y_min, 0.5, 0.5, 0],  # Lower bounds
            [np.inf, x_max, y_max, 10, 10, np.inf]  # Upper bounds
        )
        
        try:
            # Fit
            popt, pcov = curve_fit(
                lambda coords, amp, x, y, sx, sy, off: 
                    gaussian_2d(coords, amp, x, y, sx, sy, off).ravel(),
                (x_grid, y_grid),
                window.ravel(),
                p0=p0,
                bounds=bounds,
                maxfev=1000
            )
            
            # Extract parameters
            params = {
                'amplitude': popt[0],
                'x': popt[1],
                'y': popt[2],
                'sigma_x': popt[3],
                'sigma_y': popt[4],
                'offset': popt[5],
                'success': True,
                'covariance': pcov
            }
            
        except Exception as e:
            # Fitting failed, use initial guess
            params = {
                'x': x0,
                'y': y0,
                'amplitude': amplitude_guess,
                'sigma_x': self.expected_sigma,
                'sigma_y': self.expected_sigma,
                'offset': offset_guess,
                'success': False,
                'error': str(e)
            }
        
        return params
    
    def fit_image(self, image: np.ndarray) -> List[dict]:
        """
        Fit all atomic columns in image.
        
        Parameters
        ----------
        image : np.ndarray
            Input TEM image
            
        Returns
        -------
        results : list of dict
            Fitted parameters for each detected atom
        """
        # Detect peaks
        peaks = self.detect_peaks(image)
        
        print(f"Detected {len(peaks)} potential atomic columns")
        
        # Fit each peak
        results = []
        for i, (y, x) in enumerate(peaks):
            params = self.fit_single_peak(image, (y, x))
            params['id'] = i
            results.append(params)
        
        # Filter unsuccessful fits
        successful = [r for r in results if r['success']]
        print(f"Successfully fitted {len(successful)}/{len(peaks)} columns")
        
        return successful


def fit_lattice_model(positions: np.ndarray,
                     lattice_constant: float) -> dict:
    """
    Fit periodic lattice model to positions.
    
    For graphene: hexagonal lattice with a = 1.42 Å bond length
    
    Parameters
    ----------
    positions : np.ndarray
        Atomic positions, shape (N, 2)
    lattice_constant : float
        Expected lattice constant
        
    Returns
    -------
    lattice_params : dict
        Lattice parameters and fit quality
    """
    from scipy.spatial import distance_matrix
    
    # Calculate all pairwise distances
    distances = distance_matrix(positions, positions)
    
    # Find nearest neighbors
    nn_distances = []
    for i in range(len(positions)):
        dists = distances[i]
        # Exclude self (distance = 0)
        dists_nonzero = dists[dists > 0]
        if len(dists_nonzero) > 0:
            nn_distances.append(np.min(dists_nonzero))
    
    measured_constant = np.median(nn_distances)
    lattice_std = np.std(nn_distances)
    
    return {
        'measured_constant': measured_constant,
        'expected_constant': lattice_constant,
        'deviation': abs(measured_constant - lattice_constant),
        'std': lattice_std,
        'relative_error': abs(measured_constant - lattice_constant) / lattice_constant
    }


if __name__ == '__main__':
    """Test Gaussian fitting"""
    
    print("Gaussian Fitting Module - Test")
    print("=" * 60)
    
    # Create synthetic image with Gaussian peaks
    size = 256
    y, x = np.mgrid[0:size, 0:size]
    
    image = np.zeros((size, size))
    true_positions = []
    
    # Add peaks on approximate hexagonal lattice
    lattice_spacing = 20  # pixels
    np.random.seed(42)
    
    for i in range(10):
        for j in range(10):
            x_pos = 50 + i * lattice_spacing + np.random.randn() * 0.5
            y_pos = 50 + j * lattice_spacing + (i % 2) * lattice_spacing/2 + np.random.randn() * 0.5
            
            if x_pos < size-50 and y_pos < size-50:
                amplitude = 1.0 + np.random.randn() * 0.1
                image += gaussian_2d((x, y), amplitude, x_pos, y_pos, 2.0, 2.0, 0)
                true_positions.append([x_pos, y_pos])
    
    # Add noise
    image += np.random.normal(0, 0.05, image.shape)
    image += 0.1  # offset
    
    print(f"\n1. Created test image with {len(true_positions)} atoms")
    
    # Fit
    fitter = GaussianFitter(expected_sigma=2.0, min_distance=10)
    results = fitter.fit_image(image)
    
    print(f"\n2. Fitting results:")
    print(f"   Detected: {len(results)} atoms")
    
    # Analyze accuracy
    fitted_positions = np.array([[r['x'], r['y']] for r in results])
    
    if len(fitted_positions) > 0:
        # Find nearest true position for each fit
        errors = []
        for fit_pos in fitted_positions:
            distances = [np.linalg.norm(fit_pos - true_pos) 
                        for true_pos in true_positions]
            errors.append(min(distances))
        
        print(f"   Mean localization error: {np.mean(errors):.3f} pixels")
        print(f"   Std error: {np.std(errors):.3f} pixels")
        
        # Lattice analysis
        lattice = fit_lattice_model(fitted_positions, lattice_spacing)
        print(f"\n3. Lattice analysis:")
        print(f"   Expected spacing: {lattice_spacing:.2f} pixels")
        print(f"   Measured spacing: {lattice['measured_constant']:.2f} pixels")
        print(f"   Error: {lattice['relative_error']*100:.2f}%")
    
    print("\n" + "=" * 60)
    print("✓ Gaussian fitting module test complete!")