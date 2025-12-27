"""
Basic Usage Examples

Quick-start examples for common tasks.
"""

import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, '../src')

from preprocessing import PreprocessingPipeline, BM3DDenoiser
from estimation import GaussianFitter
from analysis import StructureAnalyzer
from visualization import StructureVisualizer


def example_1_preprocessing():
    """
    Example 1: Preprocess a noisy image sequence
    """
    print("="*60)
    print("EXAMPLE 1: Preprocessing Pipeline")
    print("="*60)
    
    # Create synthetic noisy images
    print("\n1. Creating synthetic noisy images...")
    images = []
    for i in range(10):
        # Simulate noisy image
        img = np.random.rand(256, 256) * 0.5
        img += np.random.normal(0, 0.1, (256, 256))
        images.append(img)
    
    print(f"   ✓ Created {len(images)} noisy images")
    
    # Apply preprocessing pipeline
    print("\n2. Applying preprocessing...")
    pipeline = PreprocessingPipeline(window_size=5)
    
    # Process middle frame
    target_idx = len(images) // 2
    cleaned = pipeline.process(images, target_idx)
    
    print("   ✓ Preprocessing complete!")
    print(f"   - Temporal averaging: {pipeline.averager.snr_improvement:.2f}× SNR improvement")
    print(f"   - BM3D denoising: {pipeline.denoiser.psnr_improvement:.1f} dB PSNR gain")
    
    print("\n" + "="*60)
    return cleaned


def example_2_structure_estimation():
    """
    Example 2: Estimate atomic positions from clean image
    """
    print("="*60)
    print("EXAMPLE 2: Structure Estimation")
    print("="*60)
    
    # Create synthetic atomic image
    print("\n1. Creating synthetic atomic structure...")
    
    size = 256
    y, x = np.mgrid[0:size, 0:size]
    img = np.zeros((size, size))
    
    # Add Gaussian peaks (simulating atoms)
    n_atoms = 30
    positions = []
    
    np.random.seed(42)
    for i in range(n_atoms):
        cx = np.random.uniform(30, size-30)
        cy = np.random.uniform(30, size-30)
        
        # Gaussian peak
        img += np.exp(-((x-cx)**2 + (y-cy)**2) / (2*3**2))
        positions.append([cx, cy])
    
    print(f"   ✓ Created image with {n_atoms} atoms")
    
    # Estimate positions
    print("\n2. Estimating atomic positions...")
    fitter = GaussianFitter()
    
    results = fitter.fit_image(img, min_distance=10, threshold=0.1)
    
    print(f"   ✓ Found {len(results['positions'])} atoms")
    print(f"   - Mean intensity: {np.mean(results['intensities']):.3f}")
    print(f"   - Mean sigma: {np.mean(results['sigmas']):.3f} pixels")
    
    print("\n" + "="*60)
    return results


def example_3_structure_analysis():
    """
    Example 3: Analyze 3D structure
    """
    print("="*60)
    print("EXAMPLE 3: Structure Analysis")
    print("="*60)
    
    # Create synthetic 3D structure
    print("\n1. Creating synthetic 3D structure...")
    
    # Simple graphene-like lattice
    n_atoms = 50
    structure_3d = np.zeros((n_atoms, 3))
    
    # Hexagonal lattice with rippling
    idx = 0
    for i in range(7):
        for j in range(7):
            if idx >= n_atoms:
                break
            
            x = i * 2.5 + (j % 2) * 1.25
            y = j * 2.2
            z = 0.3 * np.sin(i * 0.5) * np.cos(j * 0.5)  # Rippling
            
            structure_3d[idx] = [x, y, z]
            idx += 1
    
    # Add small noise
    structure_3d += np.random.normal(0, 0.05, structure_3d.shape)
    
    print(f"   ✓ Created structure with {len(structure_3d)} atoms")
    
    # Analyze structure
    print("\n2. Analyzing structure...")
    analyzer = StructureAnalyzer()
    
    results = analyzer.analyze_full(structure_3d)
    
    print("\n3. Analysis Results:")
    print(f"   Bond Lengths:")
    print(f"   - Mean: {results['bonds']['mean']:.3f} Å")
    print(f"   - Std:  {results['bonds']['std']:.3f} Å")
    
    print(f"\n   Heights (z-coordinates):")
    print(f"   - RMS roughness: {results['heights']['rms_roughness']:.3f} Å")
    print(f"   - Peak-to-valley: {results['heights']['peak_to_valley']:.3f} Å")
    
    print(f"\n   Strain:")
    print(f"   - Mean: {results['strain']['mean']*100:.2f}%")
    print(f"   - Max:  {results['strain']['max']*100:.2f}%")
    
    print("\n" + "="*60)
    return structure_3d, results


def example_4_visualization():
    """
    Example 4: Visualize 3D structure
    """
    print("="*60)
    print("EXAMPLE 4: Visualization")
    print("="*60)
    
    # Create structure (reuse from example 3)
    print("\n1. Creating structure...")
    structure_3d, _ = example_3_structure_analysis()
    
    # Visualize
    print("\n2. Creating visualizations...")
    viz = StructureVisualizer()
    
    # 3D plot
    viz.plot_3d(
        structure_3d,
        color_by='z',
        save_path='example_3d_structure.png',
        show=False
    )
    print("   ✓ Saved: example_3d_structure.png")
    
    # Height map
    viz.plot_height_map(
        structure_3d,
        save_path='example_height_map.png',
        show=False
    )
    print("   ✓ Saved: example_height_map.png")
    
    print("\n" + "="*60)


def example_5_complete_workflow():
    """
    Example 5: Complete workflow from raw data to analysis
    """
    print("="*60)
    print("EXAMPLE 5: Complete Workflow")
    print("="*60)
    
    print("\n1. Preprocessing...")
    cleaned = example_1_preprocessing()
    
    print("\n2. Structure Estimation...")
    results = example_2_structure_estimation()
    
    print("\n3. Analysis...")
    structure_3d, analysis = example_3_structure_analysis()
    
    print("\n4. Visualization...")
    example_4_visualization()
    
    print("\n" + "="*60)
    print("COMPLETE WORKFLOW FINISHED!")
    print("="*60)
    print("\nGenerated files:")
    print("  - example_3d_structure.png")
    print("  - example_height_map.png")
    print("\nThis demonstrates the complete pipeline:")
    print("  Raw Data → Preprocessing → Estimation → Analysis → Visualization")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run usage examples')
    parser.add_argument(
        'example',
        type=int,
        choices=[1, 2, 3, 4, 5],
        nargs='?',
        default=5,
        help='Example number to run (1-5, default: 5 = all)'
    )
    
    args = parser.parse_args()
    
    examples = {
        1: example_1_preprocessing,
        2: example_2_structure_estimation,
        3: example_3_structure_analysis,
        4: example_4_visualization,
        5: example_5_complete_workflow,
    }
    
    print("\n" + "="*70)
    print("3D RECONSTRUCTION - USAGE EXAMPLES")
    print("="*70)
    
    examples[args.example]()
    
    print("\n✓ Example complete!")