"""
Basic Usage Examples - Minimal Working Version

Demonstrates core functionality without complex dependencies.
"""

import numpy as np
import sys

# Add src to path
sys.path.insert(0, '../src')

from preprocessing import PreprocessingPipeline, BM3DDenoiser
from analysis import StructureAnalyzer
from visualization import StructureVisualizer


def example_1_preprocessing():
    """Example 1: Preprocess a noisy image sequence"""
    print("="*60)
    print("EXAMPLE 1: Preprocessing Pipeline")
    print("="*60)
    
    print("\n1. Creating synthetic noisy images...")
    images = []
    for i in range(10):
        img = np.random.rand(256, 256) * 0.5
        img += np.random.normal(0, 0.1, (256, 256))
        images.append(img)
    
    print(f"   ✓ Created {len(images)} noisy images")
    
    print("\n2. Applying preprocessing pipeline...")
    pipeline = PreprocessingPipeline(window_size=5)
    target_idx = len(images) // 2
    cleaned = pipeline.process(images, target_idx)
    
    print("   ✓ Preprocessing complete!")
    print(f"   - Temporal averaging: {pipeline.averager.snr_improvement:.2f}× improvement")
    print(f"   - BM3D denoising: {pipeline.denoiser.psnr_improvement:.1f} dB gain")
    
    print("\n" + "="*60)
    return cleaned


def example_2_denoising():
    """Example 2: BM3D denoising demonstration"""
    print("="*60)
    print("EXAMPLE 2: Denoising with BM3D")
    print("="*60)
    
    print("\n1. Creating noisy test image...")
    size = 256
    y, x = np.mgrid[0:size, 0:size]
    
    # Clean image with peaks
    clean = np.zeros((size, size))
    np.random.seed(42)
    for i in range(20):
        cx = np.random.uniform(30, size-30)
        cy = np.random.uniform(30, size-30)
        clean += np.exp(-((x-cx)**2 + (y-cy)**2) / (2*3**2))
    
    # Add noise
    noisy = clean + np.random.normal(0, 0.15, clean.shape)
    
    print("   ✓ Created noisy image")
    
    print("\n2. Denoising...")
    denoiser = BM3DDenoiser()
    denoised = denoiser.denoise(noisy)
    
    # Calculate improvement
    mse_before = np.mean((noisy - clean)**2)
    mse_after = np.mean((denoised - clean)**2)
    improvement = mse_before / mse_after
    
    print("   ✓ Denoising complete!")
    print(f"   - Error reduction: {improvement:.2f}×")
    print(f"   - MSE before: {mse_before:.6f}")
    print(f"   - MSE after: {mse_after:.6f}")
    
    print("\n" + "="*60)
    return denoised


def example_3_structure_analysis():
    """Example 3: Analyze 3D structure"""
    print("="*60)
    print("EXAMPLE 3: Structure Analysis")
    print("="*60)
    
    print("\n1. Creating synthetic 3D structure...")
    
    # Hexagonal lattice with rippling
    n_atoms = 50
    structure_3d = np.zeros((n_atoms, 3))
    
    idx = 0
    for i in range(7):
        for j in range(7):
            if idx >= n_atoms:
                break
            
            x = i * 2.5 + (j % 2) * 1.25
            y = j * 2.2
            z = 0.3 * np.sin(i * 0.5) * np.cos(j * 0.5)
            
            structure_3d[idx] = [x, y, z]
            idx += 1
    
    # Add noise
    structure_3d += np.random.normal(0, 0.05, structure_3d.shape)
    
    print(f"   ✓ Created structure with {len(structure_3d)} atoms")
    
    print("\n2. Analyzing structure...")
    analyzer = StructureAnalyzer()
    results = analyzer.analyze_full(structure_3d)
    
    print("\n3. Results:")
    
    # Safely print results
    if 'bonds' in results:
        bonds = results['bonds']
        print(f"   Bond Lengths:")
        if 'mean' in bonds:
            print(f"   - Mean: {bonds['mean']:.3f} Å")
        if 'std' in bonds:
            print(f"   - Std:  {bonds['std']:.3f} Å")
        if 'min' in bonds:
            print(f"   - Min:  {bonds['min']:.3f} Å")
        if 'max' in bonds:
            print(f"   - Max:  {bonds['max']:.3f} Å")
    
    if 'heights' in results:
        heights = results['heights']
        print(f"\n   Surface Properties:")
        if 'rms_roughness' in heights:
            print(f"   - RMS roughness: {heights['rms_roughness']:.3f} Å")
        if 'range' in heights:
            print(f"   - Height range: {heights['range']:.3f} Å")
        if 'mean' in heights:
            print(f"   - Mean height: {heights['mean']:.3f} Å")
        if 'std' in heights:
            print(f"   - Std height: {heights['std']:.3f} Å")
    
    if 'strain' in results:
        strain = results['strain']
        print(f"\n   Strain Analysis:")
        if 'mean' in strain:
            print(f"   - Mean: {strain['mean']*100:.2f}%")
        if 'std' in strain:
            print(f"   - Std:  {strain['std']*100:.2f}%")
        if 'max' in strain:
            print(f"   - Max:  {strain['max']*100:.2f}%")
    
    if 'gradients' in results:
        gradients = results['gradients']
        print(f"\n   Gradients:")
        if 'mean' in gradients:
            print(f"   - Mean: {gradients['mean']:.3f} Å/Å")
        if 'max' in gradients:
            print(f"   - Max:  {gradients['max']:.3f} Å/Å")
    
    print("\n" + "="*60)
    return structure_3d, results

def example_4_visualization():
    """Example 4: Visualize 3D structure"""
    print("="*60)
    print("EXAMPLE 4: Visualization")
    print("="*60)
    
    print("\n1. Creating structure...")
    structure_3d, _ = example_3_structure_analysis()
    
    print("\n2. Creating 3D visualization...")
    viz = StructureVisualizer()
    
    # Basic 3D plot (always available)
    viz.plot_3d(
        structure_3d,
        color_by='z',
        save_path='example_3d_structure.png',
        show=False
    )
    print("   ✓ Saved: example_3d_structure.png")
    
    # Try comparison plot if available
    if hasattr(viz, 'plot_comparison'):
        try:
            structure_2 = structure_3d + np.random.normal(0, 0.02, structure_3d.shape)
            viz.plot_comparison(
                structure_3d,
                structure_2,
                labels=['Original', 'Perturbed'],
                save_path='example_comparison.png'
            )
            print("   ✓ Saved: example_comparison.png")
        except Exception as e:
            print(f"   ⚠ Comparison plot skipped: {str(e)[:50]}")
    
    # Try interactive plot if available
    if hasattr(viz, 'plot_interactive'):
        try:
            viz.plot_interactive(
                structure_3d,
                save_path='example_interactive.html'
            )
            print("   ✓ Saved: example_interactive.html")
        except Exception as e:
            print(f"   ⚠ Interactive plot skipped: {str(e)[:50]}")
    
    print("\n" + "="*60)


def example_5_complete_workflow():
    """Example 5: Complete workflow"""
    print("="*60)
    print("COMPLETE WORKFLOW")
    print("="*60)
    
    print("\n" + "Step 1: Preprocessing")
    example_1_preprocessing()
    
    print("\n" + "Step 2: Denoising")
    example_2_denoising()
    
    print("\n" + "Step 3: Analysis")
    example_3_structure_analysis()
    
    print("\n" + "Step 4: Visualization")
    example_4_visualization()
    
    print("\n" + "="*60)
    print("WORKFLOW COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - example_3d_structure.png")
    print("  - example_height_map.png")
    print("="*60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run usage examples')
    parser.add_argument(
        'example',
        type=int,
        choices=[1, 2, 3, 4, 5],
        nargs='?',
        default=5,
        help='Example number (1-5, default: 5 = all)'
    )
    
    args = parser.parse_args()
    
    examples = {
        1: example_1_preprocessing,
        2: example_2_denoising,
        3: example_3_structure_analysis,
        4: example_4_visualization,
        5: example_5_complete_workflow,
    }
    
    print("\n" + "="*70)
    print("3D RECONSTRUCTION - USAGE EXAMPLES")
    print("="*70)
    print()
    
    examples[args.example]()
    
    print("\n✓ Example complete!\n")