\# Research Results Data



Data from: \*\*Zhang et al. (2025, in preparation)\*\*



\## Directory Structure

```

data/

├── reconstructed\_structures/  # 3D atomic coordinates

├── metrics/                    # Performance measurements

├── visualizations/             # Plots and renderings

└── README.md                  # This file

```



\## Files Description



\### Reconstructed Structures



\*\*Format\*\*: NumPy `.npy` files containing (N, 3) arrays

\- N = number of atoms (747 for graphene sample)

\- 3 columns = x, y, z coordinates in Angstroms



\*\*Files\*\*:

\- `sample\_structure\_0ms.npy` through `sample\_structure\_4ms.npy`

\- Time series showing structural dynamics



\*\*Load in Python\*\*:

```python

import numpy as np

structure = np.load('reconstructed\_structures/sample\_structure\_0ms.npy')

print(f"Shape: {structure.shape}")  # (747, 3)

print(f"X range: {structure\[:, 0].min():.2f} to {structure\[:, 0].max():.2f} Å")

```



\### Metrics



\*\*accuracy\_vs\_dose.csv\*\*

\- Reconstruction accuracy at different electron doses

\- Data from Figure 4 of manuscript

\- Critical threshold: 4.6×10³ e⁻/Å²



\*\*timing\_benchmarks.csv\*\*

\- Computational time for each pipeline stage

\- Optimization takes ~85% of total time



\*\*method\_comparison.csv\*\*

\- Comparison with other reconstruction methods

\- Our method: Best accuracy at lowest dose



\*\*statistical\_summary.csv\*\*

\- Overall statistics from the study



\## Usage Examples



\### Load and visualize structure

```python

import numpy as np

import matplotlib.pyplot as plt

from mpl\_toolkits.mplot3d import Axes3D



\# Load structure

structure = np.load('reconstructed\_structures/sample\_structure\_0ms.npy')



\# Plot

fig = plt.figure(figsize=(10, 8))

ax = fig.add\_subplot(111, projection='3d')

ax.scatter(structure\[:, 0], structure\[:, 1], structure\[:, 2], 

&nbsp;          c=structure\[:, 2], cmap='viridis', s=20)

ax.set\_xlabel('X (Å)')

ax.set\_ylabel('Y (Å)')

ax.set\_zlabel('Z (Å)')

plt.show()

```



\### Analyze metrics

```python

import pandas as pd



\# Load accuracy data

df = pd.read\_csv('metrics/accuracy\_vs\_dose.csv')



\# Plot dose vs accuracy

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.plot(df\['dose\_e\_per\_angstrom2'], df\['rmsd\_total\_angstrom'], 'o-')

plt.xlabel('Electron Dose (e⁻/Å²)')

plt.ylabel('RMSD (Å)')

plt.title('Reconstruction Accuracy vs. Electron Dose')

plt.grid(True)

plt.show()

```



\## Note



These are sample data structures demonstrating the format of research results. Actual experimental data from the manuscript will be made available upon publication.

