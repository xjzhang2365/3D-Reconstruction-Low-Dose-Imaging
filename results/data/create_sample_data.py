import numpy as np
import pandas as pd
import json
from pathlib import Path

print("Creating sample data structure...")
print("=" * 60)

# Parameters from your paper
N_ATOMS = 747  # Number of carbon atoms in your sample
DOSE_LEVELS = [2.3e3, 4.6e3, 6.4e3, 9.1e3, 2.7e4]  # e-/Å²
ACCURACIES = [1.50, 0.87, 0.54, 0.45, 0.33]  # Angstroms (from Fig 4)

# 1. CREATE SAMPLE RECONSTRUCTED STRUCTURES
print("\n1. Creating sample structure files...")

Path("reconstructed_structures").mkdir(exist_ok=True)

# Create sample structure (random for demonstration)
sample_structure = np.random.randn(N_ATOMS, 3) * [5, 5, 2]  # xyz coordinates

# Save multiple time points
for time_ms in [0, 1, 2, 3, 4]:
    # Add small perturbation for each time point
    structure_t = sample_structure + np.random.randn(N_ATOMS, 3) * 0.1
    filename = f"sample_structure_{time_ms}ms.npy"
    np.save(f"reconstructed_structures/{filename}", structure_t)
    print(f"  ✓ {filename}")

# Create metadata file
metadata = {
    "experiment_info": {
        "n_atoms": N_ATOMS,
        "dose_level": "8.0e3 e-/Å²",
        "acquisition_interval": "1ms",
        "total_frames": 5
    },
    "reconstruction_accuracy": {
        "rmsd_x": "0.08Å",
        "rmsd_y": "0.09Å", 
        "rmsd_z": "0.45Å",
        "overall": "0.46Å"
    },
    "description": "Sample reconstructed structures (format demonstration)",
    "note": "Coordinates are in Angstroms. Z-axis is perpendicular to nominal graphene plane."
}

with open('reconstructed_structures/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("  ✓ metadata.json")

# 2. CREATE METRICS DATA
print("\n2. Creating metrics files...")

Path("metrics").mkdir(exist_ok=True)

# Accuracy vs. dose (from your Fig 4)
accuracy_df = pd.DataFrame({
    'dose_e_per_angstrom2': DOSE_LEVELS,
    'rmsd_total_angstrom': ACCURACIES,
    'rmsd_x_angstrom': [a * 0.18 for a in ACCURACIES],
    'rmsd_y_angstrom': [a * 0.20 for a in ACCURACIES],
    'rmsd_z_angstrom': [a * 0.90 for a in ACCURACIES],
    'snr_estimate': [2.1, 3.8, 5.2, 6.7, 12.5],
})
accuracy_df.to_csv('metrics/accuracy_vs_dose.csv', index=False)
print("  ✓ accuracy_vs_dose.csv")

# Timing benchmarks
timing_df = pd.DataFrame({
    'stage': ['Preprocessing', 'Estimation', 'Optimization', 'Validation', 'Total'],
    'time_seconds': [2.3, 5.1, 142.5, 18.2, 168.1],
    'percentage': [1.4, 3.0, 84.8, 10.8, 100.0],
    'notes': [
        'Averaging + BM3D denoising',
        'Gaussian fitting + Bayesian inference',
        'Simulated Annealing iterations',
        'MD relaxation check',
        'End-to-end reconstruction'
    ]
})
timing_df.to_csv('metrics/timing_benchmarks.csv', index=False)
print("  ✓ timing_benchmarks.csv")

# Method comparison
comparison_df = pd.DataFrame({
    'method': ['Our Method', 'Standard SA', 'Exit Wave Recon.', 'Tilt Series', 'Deep Learning'],
    'accuracy_angstrom': [0.45, 1.2, 0.8, 0.6, 0.9],
    'dose_required': ['8×10³', '2×10⁴', '5×10⁴', '3×10⁴', '1×10⁴'],
    'time_resolution_ms': [1, 10, 100, 50, 5],
    'images_required': [1, 1, 20, 30, 1],
    'requires_prior_knowledge': ['No', 'No', 'Yes', 'No', 'Yes']
})
comparison_df.to_csv('metrics/method_comparison.csv', index=False)
print("  ✓ method_comparison.csv")

# Statistical summary
stats_df = pd.DataFrame({
    'metric': [
        'Total atoms analyzed',
        'Frames processed', 
        'Average SNR (input)',
        'Average SNR (after preprocessing)',
        'Success rate',
        'Mean convergence iterations'
    ],
    'value': [747, 50, 3.2, 8.5, 96, 4.2],
    'unit': ['atoms', 'frames', 'ratio', 'ratio', '%', 'iterations']
})
stats_df.to_csv('metrics/statistical_summary.csv', index=False)
print("  ✓ statistical_summary.csv")

print("\n" + "=" * 60)
print("✓ Sample data structure created successfully!")
print("\nFiles created:")
print("  - 5 structure files (.npy)")
print("  - 1 metadata file (.json)")
print("  - 4 metrics files (.csv)")
print("\nNext step: Replace with your actual research data as needed.")