"""
Structure analysis tools for reconstructed 3D atomic structures.

These are standard analysis methods, fully implemented and safe to share.
Used to analyze the reconstructed structures from the optimization.

Analyses performed in the paper:
- Bond length distribution
- Strain mapping  
- Height distribution
- Gradient analysis
- Statistical metrics
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.spatial import distance_matrix
from scipy.interpolate import griddata
import warnings


class StructureAnalyzer:
    """
    Analyze reconstructed 3D atomic structures.
    
    Provides various structural analyses as shown in Fig 3 of the paper:
    - Bond lengths and distributions
    - Strain calculations
    - Height distributions (rippling)
    - Geometric gradients (curvature)
    
    Examples
    --------
    >>> analyzer = StructureAnalyzer()
    >>> structure = np.load('reconstructed_structure.npy')  # (N, 3)
    >>> results = analyzer.analyze_full(structure)
    >>> print(f"Mean bond length: {results['bond_length_mean']:.3f} Å")
    """
    
    def __init__(self, 
                 expected_bond_length: float = 1.42,
                 lattice_type: str = 'graphene'):
        """
        Initialize analyzer.
        
        Parameters
        ----------
        expected_bond_length : float
            Expected bond length in Angstroms (1.42 for graphene)
        lattice_type : str
            Type of lattice structure
        """
        self.expected_bond_length = expected_bond_length
        self.lattice_type = lattice_type
    
    def calculate_bond_lengths(self, 
                               positions: np.ndarray,
                               max_bond_length: float = 2.0) -> Dict:
        """
        Calculate bond lengths between neighboring atoms.
        
        Parameters
        ----------
        positions : np.ndarray
            Atomic positions, shape (N, 3)
        max_bond_length : float
            Maximum distance to consider as bond
            
        Returns
        -------
        results : dict
            Bond length statistics
        """
        # Calculate distance matrix
        distances = distance_matrix(positions, positions)
        
        # Find bonds (neighbors within max_bond_length, excluding self)
        bonds = []
        for i in range(len(positions)):
            neighbor_distances = distances[i]
            # Exclude self (distance = 0)
            neighbor_distances = neighbor_distances[neighbor_distances > 0]
            # Keep only within max distance
            valid_bonds = neighbor_distances[neighbor_distances < max_bond_length]
            bonds.extend(valid_bonds)
        
        bonds = np.array(bonds)
        
        if len(bonds) == 0:
            warnings.warn("No bonds found within max_bond_length")
            return {'bond_lengths': np.array([]), 'mean': 0, 'std': 0}
        
        # Calculate relative change from expected
        relative_change = (bonds - self.expected_bond_length) / self.expected_bond_length
        
        return {
            'bond_lengths': bonds,
            'mean': np.mean(bonds),
            'std': np.std(bonds),
            'min': np.min(bonds),
            'max': np.max(bonds),
            'median': np.median(bonds),
            'relative_change': relative_change,
            'mean_relative_change': np.mean(relative_change),
            'n_bonds': len(bonds)
        }
    
    def calculate_height_distribution(self, 
                                     positions: np.ndarray) -> Dict:
        """
        Analyze z-height distribution (rippling).
        
        As shown in Fig 3a of the paper.
        
        Parameters
        ----------
        positions : np.ndarray
            Atomic positions, shape (N, 3)
            
        Returns
        -------
        results : dict
            Height distribution statistics
        """
        z_heights = positions[:, 2]
        
        # Fit plane to find average orientation
        from scipy.optimize import minimize
        
        def plane_error(params):
            a, b, c = params
            z_plane = a * positions[:, 0] + b * positions[:, 1] + c
            return np.sum((z_heights - z_plane)**2)
        
        result = minimize(plane_error, [0, 0, np.mean(z_heights)])
        a_fit, b_fit, c_fit = result.x
        
        # Calculate deviation from fitted plane
        z_plane = a_fit * positions[:, 0] + b_fit * positions[:, 1] + c_fit
        deviations = z_heights - z_plane
        
        return {
            'z_raw': z_heights,
            'z_deviations': deviations,
            'mean': np.mean(deviations),
            'std': np.std(deviations),
            'min': np.min(deviations),
            'max': np.max(deviations),
            'range': np.max(deviations) - np.min(deviations),
            'plane_params': (a_fit, b_fit, c_fit),
            'rms_roughness': np.sqrt(np.mean(deviations**2))
        }
    
    def calculate_strain_map(self,
                            positions: np.ndarray,
                            reference_lattice_constant: Optional[float] = None) -> Dict:
        """
        Calculate local strain at each atomic position.
        
        Strain = (measured - expected) / expected
        
        Parameters
        ----------
        positions : np.ndarray
            Atomic positions, shape (N, 3)
        reference_lattice_constant : float, optional
            Reference lattice constant (uses self.expected_bond_length if None)
            
        Returns
        -------
        results : dict
            Strain values and statistics
        """
        if reference_lattice_constant is None:
            reference_lattice_constant = self.expected_bond_length
        
        n_atoms = len(positions)
        strains = np.zeros(n_atoms)
        
        # Calculate distance matrix
        distances = distance_matrix(positions, positions)
        
        # For each atom, calculate local strain
        for i in range(n_atoms):
            # Find nearest neighbors
            neighbor_dists = distances[i]
            neighbor_dists_nonzero = neighbor_dists[neighbor_dists > 0]
            
            if len(neighbor_dists_nonzero) > 0:
                # Use mean of 3 nearest neighbors
                nearest = np.sort(neighbor_dists_nonzero)[:min(3, len(neighbor_dists_nonzero))]
                local_spacing = np.mean(nearest)
                
                # Calculate strain
                strains[i] = (local_spacing - reference_lattice_constant) / reference_lattice_constant
        
        return {
            'strains': strains,
            'mean': np.mean(strains),
            'std': np.std(strains),
            'min': np.min(strains),
            'max': np.max(strains),
            'positions': positions  # Include for mapping
        }
    
    def calculate_geometric_gradient(self,
                                    positions: np.ndarray,
                                    grid_resolution: int = 50) -> Dict:
        """
        Calculate geometric gradient (curvature) map.
        
        As shown in Fig 3c of the paper.
        
        Parameters
        ----------
        positions : np.ndarray
            Atomic positions, shape (N, 3)
        grid_resolution : int
            Resolution of interpolation grid
            
        Returns
        -------
        results : dict
            Gradient magnitude and components
        """
        # Create regular grid
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        
        grid_x, grid_y = np.meshgrid(
            np.linspace(x_min, x_max, grid_resolution),
            np.linspace(y_min, y_max, grid_resolution)
        )
        
        # Interpolate z values onto grid
        grid_z = griddata(
            positions[:, :2],
            positions[:, 2],
            (grid_x, grid_y),
            method='cubic',
            fill_value=np.nan
        )
        
        # Calculate gradients
        grad_y, grad_x = np.gradient(grid_z)
        
        # Gradient magnitude
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Mask NaN values
        valid_mask = ~np.isnan(grad_magnitude)
        
        return {
            'grid_x': grid_x,
            'grid_y': grid_y,
            'grid_z': grid_z,
            'gradient_x': grad_x,
            'gradient_y': grad_y,
            'gradient_magnitude': grad_magnitude,
            'mean_gradient': np.nanmean(grad_magnitude),
            'max_gradient': np.nanmax(grad_magnitude),
            'valid_mask': valid_mask
        }
    
    def analyze_full(self, positions: np.ndarray) -> Dict:
        """
        Perform complete structural analysis.
        
        Runs all analysis methods and combines results.
        
        Parameters
        ----------
        positions : np.ndarray
            Atomic positions, shape (N, 3)
            
        Returns
        -------
        results : dict
            Complete analysis results
        """
        print("Performing complete structural analysis...")
        print("=" * 60)
        
        results = {}
        
        # Bond length analysis
        print("\n1. Analyzing bond lengths...")
        results['bonds'] = self.calculate_bond_lengths(positions)
        print(f"   Mean bond length: {results['bonds']['mean']:.3f} Å")
        print(f"   Std: {results['bonds']['std']:.3f} Å")
        
        # Height distribution
        print("\n2. Analyzing height distribution...")
        results['heights'] = self.calculate_height_distribution(positions)
        print(f"   RMS roughness: {results['heights']['rms_roughness']:.3f} Å")
        print(f"   Height range: {results['heights']['range']:.3f} Å")
        
        # Strain analysis
        print("\n3. Calculating strain map...")
        results['strain'] = self.calculate_strain_map(positions)
        print(f"   Mean strain: {results['strain']['mean']*100:.2f}%")
        print(f"   Strain range: {results['strain']['min']*100:.2f}% to {results['strain']['max']*100:.2f}%")
        
        # Geometric gradient
        print("\n4. Computing geometric gradients...")
        results['gradient'] = self.calculate_geometric_gradient(positions)
        print(f"   Mean gradient: {results['gradient']['mean_gradient']:.4f}")
        print(f"   Max gradient: {results['gradient']['max_gradient']:.4f}")
        
        print("\n" + "=" * 60)
        print("✓ Analysis complete!\n")
        
        return results


def compare_structures(structure1: np.ndarray,
                      structure2: np.ndarray) -> Dict:
    """
    Compare two structures (e.g., different time points).
    
    Calculates RMSD and other comparison metrics.
    
    Parameters
    ----------
    structure1, structure2 : np.ndarray
        Atomic positions, shape (N, 3)
        
    Returns
    -------
    comparison : dict
        Comparison metrics
    """
    if structure1.shape != structure2.shape:
        raise ValueError("Structures must have same shape")
    
    # Calculate RMSD
    differences = structure1 - structure2
    rmsd_xyz = np.sqrt(np.mean(differences**2, axis=0))
    rmsd_total = np.sqrt(np.mean(np.sum(differences**2, axis=1)))
    
    # Per-atom displacements
    displacements = np.sqrt(np.sum(differences**2, axis=1))
    
    return {
        'rmsd_x': rmsd_xyz[0],
        'rmsd_y': rmsd_xyz[1],
        'rmsd_z': rmsd_xyz[2],
        'rmsd_total': rmsd_total,
        'displacements': displacements,
        'mean_displacement': np.mean(displacements),
        'max_displacement': np.max(displacements),
    }


if __name__ == '__main__':
    """Test structure analysis"""
    
    print("Structure Analysis Module - Test")
    print("=" * 60)
    
    # Create synthetic graphene-like structure
    n_x, n_y = 10, 10
    a = 1.42  # Bond length
    
    positions = []
    for i in range(n_x):
        for j in range(n_y):
            x = i * a * np.sqrt(3)
            y = j * a * 1.5 + (i % 2) * a * 0.75
            # Add rippling
            z = 0.5 * np.sin(x / 5) * np.cos(y / 5)
            # Add small random perturbations
            x += np.random.randn() * 0.05
            y += np.random.randn() * 0.05
            z += np.random.randn() * 0.1
            positions.append([x, y, z])
    
    positions = np.array(positions)
    
    print(f"\n1. Created test structure with {len(positions)} atoms")
    
    # Analyze
    analyzer = StructureAnalyzer(expected_bond_length=a)
    results = analyzer.analyze_full(positions)
    
    print("\n✓ Structure analysis module test complete!")