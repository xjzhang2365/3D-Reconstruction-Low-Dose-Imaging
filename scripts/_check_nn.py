import sys; sys.path.insert(0,'src')
import numpy as np
from scipy.spatial import cKDTree

pos = np.load('outputs/stage2/validation/stage2_validation_init_sa_input.npz')['xyz_angstrom'].copy()

tree2d = cKDTree(pos[:, :2])
d2d, _ = tree2d.query(pos[:, :2], k=4)
nn2d = d2d[:, 1]
print('INITIAL SA INPUT 2D NN:')
print(f'  mean={nn2d.mean():.4f}  min={nn2d.min():.4f}  max={nn2d.max():.4f}')
print(f'  fraction < 1.3 A: {(nn2d < 1.3).mean():.1%}')

tree3d = cKDTree(pos)
d3d, _ = tree3d.query(pos, k=4)
nn3d = d3d[:, 1]
print('INITIAL SA INPUT 3D NN:')
print(f'  mean={nn3d.mean():.4f}  min={nn3d.min():.4f}  max={nn3d.max():.4f}')
print(f'  fraction < 1.3 A: {(nn3d < 1.3).mean():.1%}')
print(f'  z std: {pos[:,2].std():.4f} A')
