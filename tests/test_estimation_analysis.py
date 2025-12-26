"""
Tests for estimation and analysis modules.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from estimation import GaussianFitter, PCDMethod, StructureEstimator
from analysis import StructureAnalyzer, compare_structures


def create_test_structure(n_atoms=50):
    """Create test atomic structure"""
    positions = []
    np.random.seed(42)
    
    for i in range(n_atoms):
        x = np.random.uniform(10, 100)
        y = np.random.uniform(10, 100)
        z = np.random.uniform(-2, 2)
        positions.append([x, y, z])
    
    return np.array(positions)


class TestGaussianFitting:
    """Tests for Gaussian fitting"""
    
    @staticmethod
    def test_gaussian_fitting():
        """Test basic Gaussian fitting"""
        # Create synthetic image
        size = 128
        y, x = np.mgrid[0:size, 0:size]
        image = np.zeros((size, size))
        
        # Add Gaussian peaks
        for i in range(5):
            cx, cy = np.random.randint(20, size-20, 2)
            image += np.exp(-((x-cx)**2 + (y-cy)**2) / (2*3**2))
        
        # Fit
        fitter = GaussianFitter(expected_sigma=3.0, min_distance=10)
        results = fitter.fit_image(image)
        
        assert len(results) > 0, "Should detect some peaks"
        print(f"  ✓ Gaussian fitting test passed (found {len(results)} peaks)")


class TestPCDMethod:
    """Tests for PCD z-estimation"""
    
    @staticmethod
    def test_pcd_estimation():
        """Test PCD z-height estimation"""
        # Create image with varying intensities
        size = 128
        y, x = np.mgrid[0:size, 0:size]
        image = np.zeros((size, size))
        
        xy_positions = []
        true_z = []
        
        for i in range(10):
            cx, cy = np.random.randint(20, size-20, 2)
            cz = np.random.uniform(-2, 2)
            intensity = 1.0 + 0.5 * cz  # Intensity varies with z
            
            image += intensity * np.exp(-((x-cx)**2 + (y-cy)**2) / (2*3**2))
            xy_positions.append([cx, cy])
            true_z.append(cz)
        
        xy_positions = np.array(xy_positions)
        
        # Estimate z
        pcd = PCDMethod()
        z_estimated = pcd.estimate_z(image, xy_positions)
        
        assert len(z_estimated) == len(xy_positions)
        assert np.mean(z_estimated) < 1.0  # Should be normalized
        
        print(f"  ✓ PCD estimation test passed")


class TestStructureAnalyzer:
    """Tests for structure analysis"""
    
    @staticmethod
    def test_bond_length_analysis():
        """Test bond length calculation"""
        # Create regular lattice
        positions = create_test_structure(20)
        
        analyzer = StructureAnalyzer()
        results = analyzer.calculate_bond_lengths(positions)
        
        assert results['n_bonds'] > 0
        assert results['mean'] > 0
        
        print(f"  ✓ Bond length analysis test passed")
    
    @staticmethod
    def test_height_distribution():
        """Test height distribution analysis"""
        positions = create_test_structure(30)
        
        analyzer = StructureAnalyzer()
        results = analyzer.calculate_height_distribution(positions)
        
        assert 'rms_roughness' in results
        assert results['rms_roughness'] >= 0
        
        print(f"  ✓ Height distribution test passed")
    
    @staticmethod
    def test_strain_calculation():
        """Test strain mapping"""
        positions = create_test_structure(25)
        
        analyzer = StructureAnalyzer()
        results = analyzer.calculate_strain_map(positions)
        
        assert len(results['strains']) == len(positions)
        
        print(f"  ✓ Strain calculation test passed")
    
    @staticmethod
    def test_gradient_calculation():
        """Test geometric gradient"""
        positions = create_test_structure(40)
        
        analyzer = StructureAnalyzer()
        results = analyzer.calculate_geometric_gradient(positions)
        
        assert 'gradient_magnitude' in results
        assert results['mean_gradient'] >= 0
        
        print(f"  ✓ Gradient calculation test passed")


class TestStructureComparison:
    """Tests for structure comparison"""
    
    @staticmethod
    def test_structure_comparison():
        """Test RMSD calculation"""
        struct1 = create_test_structure(30)
        struct2 = struct1 + np.random.randn(*struct1.shape) * 0.1
        
        comparison = compare_structures(struct1, struct2)
        
        assert 'rmsd_total' in comparison
        assert comparison['rmsd_total'] > 0
        assert comparison['rmsd_total'] < 1.0  # Should be small
        
        print(f"  ✓ Structure comparison test passed (RMSD={comparison['rmsd_total']:.3f})")


def run_all_tests():
    """Run all tests"""
    print("Running Estimation & Analysis Module Tests")
    print("=" * 60)
    
    print("\n1. Gaussian Fitting Tests:")
    TestGaussianFitting.test_gaussian_fitting()
    
    print("\n2. PCD Method Tests:")
    TestPCDMethod.test_pcd_estimation()
    
    print("\n3. Structure Analysis Tests:")
    TestStructureAnalyzer.test_bond_length_analysis()
    TestStructureAnalyzer.test_height_distribution()
    TestStructureAnalyzer.test_strain_calculation()
    TestStructureAnalyzer.test_gradient_calculation()
    
    print("\n4. Structure Comparison Tests:")
    TestStructureComparison.test_structure_comparison()
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")


if __name__ == '__main__':
    run_all_tests()