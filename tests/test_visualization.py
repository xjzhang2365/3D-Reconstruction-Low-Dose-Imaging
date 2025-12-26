"""
Tests for visualization modules.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from visualization import (
    StructureVisualizer,
    ResultsPlotter,
    plot_height_map
)


def create_test_structure(n_atoms=50):
    """Create test structure"""
    positions = []
    np.random.seed(42)
    
    for i in range(n_atoms):
        x = np.random.uniform(0, 20)
        y = np.random.uniform(0, 20)
        z = np.random.uniform(-1, 1)
        positions.append([x, y, z])
    
    return np.array(positions)


class TestStructureVisualizer:
    """Tests for structure visualization"""
    
    @staticmethod
    def test_3d_plot():
        """Test 3D structure plot"""
        structure = create_test_structure(30)
        viz = StructureVisualizer()
        
        fig = viz.plot_3d(structure)
        assert fig is not None
        plt.close(fig)
        
        print("  ✓ 3D plot test passed")
    
    @staticmethod
    def test_height_map():
        """Test height map plot"""
        structure = create_test_structure(40)
        
        fig = plot_height_map(structure)
        assert fig is not None
        plt.close(fig)
        
        print("  ✓ Height map test passed")
    
    @staticmethod
    def test_comparison_plot():
        """Test structure comparison"""
        struct1 = create_test_structure(30)
        struct2 = struct1 + np.random.randn(*struct1.shape) * 0.1
        
        viz = StructureVisualizer()
        fig = viz.plot_comparison(struct1, struct2)  # ← Removed show=False
        assert fig is not None
        plt.close(fig)
        
        print("  ✓ Comparison plot test passed")


class TestResultsPlotter:
    """Tests for results plotting"""
    
    @staticmethod
    def test_convergence_plot():
        """Test convergence plot"""
        convergence = [150, 135, 125, 122, 121, 120.5]
        
        plotter = ResultsPlotter()
        fig = plotter.plot_convergence(convergence)
        assert fig is not None
        plt.close(fig)
        
        print("  ✓ Convergence plot test passed")
    
    @staticmethod
    def test_dose_plot():
        """Test dose vs accuracy plot"""
        doses = np.array([2.3e3, 4.6e3, 6.4e3, 9.1e3, 2.7e4])
        accuracies = np.array([1.5, 0.87, 0.54, 0.45, 0.33])
        
        plotter = ResultsPlotter()
        fig = plotter.plot_accuracy_vs_dose(doses, accuracies, threshold_dose=4.6e3)
        assert fig is not None
        plt.close(fig)
        
        print("  ✓ Dose vs accuracy plot test passed")
    
    @staticmethod
    def test_comparison_bar():
        """Test method comparison bar chart"""
        methods = ['Method A', 'Method B', 'Method C']
        accuracies = [0.45, 0.8, 1.2]
        
        plotter = ResultsPlotter()
        fig = plotter.plot_comparison_bar(methods, accuracies)
        assert fig is not None
        plt.close(fig)
        
        print("  ✓ Comparison bar test passed")
    
    @staticmethod
    def test_bond_distribution():
        """Test bond length distribution plot"""
        bonds = np.random.normal(1.42, 0.05, 100)
        
        plotter = ResultsPlotter()
        fig = plotter.plot_bond_length_distribution(bonds)
        assert fig is not None
        plt.close(fig)
        
        print("  ✓ Bond distribution plot test passed")
    
    @staticmethod
    def test_height_distribution():
        """Test height distribution plot"""
        heights = np.random.normal(0, 0.5, 100)
        
        plotter = ResultsPlotter()
        fig = plotter.plot_height_distribution(heights)
        assert fig is not None
        plt.close(fig)
        
        print("  ✓ Height distribution plot test passed")


def run_all_tests():
    """Run all tests"""
    print("Running Visualization Module Tests")
    print("="*60)
    
    print("\n1. Structure Visualization Tests:")
    TestStructureVisualizer.test_3d_plot()
    TestStructureVisualizer.test_height_map()
    TestStructureVisualizer.test_comparison_plot()
    
    print("\n2. Results Plotting Tests:")
    TestResultsPlotter.test_convergence_plot()
    TestResultsPlotter.test_dose_plot()
    TestResultsPlotter.test_comparison_bar()
    TestResultsPlotter.test_bond_distribution()
    TestResultsPlotter.test_height_distribution()
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("\nNote: Plots created but not displayed (testing mode)")


if __name__ == '__main__':
    run_all_tests()