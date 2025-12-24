# Research Results Data

Data from: **Zhang et al. (2025, in preparation)**

## Directory Structure
```
data/
├── reconstructed_structures/  # 3D atomic coordinates
├── metrics/                    # Performance measurements
├── visualizations/             # Plots and renderings
└── README.md                  # This file
```

## Files Description

### Reconstructed Structures

**Format**: NumPy `.npy` files containing (N, 3) arrays
- N = number of atoms (747 for graphene sample)
- 3 columns = x, y, z coordinates in Angstroms

**Files**:
- `sample_structure_0ms.npy` through `sample_structure_4ms.npy`
- Time series showing structural dynamics

**Load in Python**:
```python
import numpy as np
structure = np.load('reconstructed_structures/sample_structure_0ms.npy')
print(f"Shape: {structure.shape}")  # (747, 3)
print(f"X range: {structure[:, 0].min():.2f} to {structure[:, 0].max():.2f} Å")
```

### Metrics

**accuracy_vs_dose.csv**
- Reconstruction accuracy at different electron doses
- Data from Figure 4 of manuscript
- Critical threshold: 4.6×10³ e⁻/Å²

**timing_benchmarks.csv**
- Computational time for each pipeline stage
- Optimization takes ~85% of total time

**method_comparison.csv**
- Comparison with other reconstruction methods
- Our method: Best accuracy at lowest dose

**statistical_summary.csv**
- Overall statistics from the study

## Usage Examples

### Load and visualize structure
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load structure
structure = np.load('reconstructed_structures/sample_structure_0ms.npy')

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(structure[:, 0], structure[:, 1], structure[:, 2], 
           c=structure[:, 2], cmap='viridis', s=20)
ax.set_xlabel('X (Å)')
ax.set_ylabel('Y (Å)')
ax.set_zlabel('Z (Å)')
plt.show()
```

### Analyze metrics
```python
import pandas as pd

# Load accuracy data
df = pd.read_csv('metrics/accuracy_vs_dose.csv')

# Plot dose vs accuracy
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.plot(df['dose_e_per_angstrom2'], df['rmsd_total_angstrom'], 'o-')
plt.xlabel('Electron Dose (e⁻/Å²)')
plt.ylabel('RMSD (Å)')
plt.title('Reconstruction Accuracy vs. Electron Dose')
plt.grid(True)
plt.show()
```

## Note

These are sample data structures demonstrating the format of research results. Actual experimental data from the manuscript will be made available upon publication.