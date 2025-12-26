
"""
Visualization module for structures and results.

Full implementation of standard visualization techniques.

Components:
-----------
- StructureVisualizer: 3D structure visualization
- ResultsPlotter: Publication-quality result plots
- Utility functions: Height maps, comparisons, etc.

Example Usage
-------------
>>> from src.visualization import StructureVisualizer, ResultsPlotter
>>> 
>>> # Visualize 3D structure
>>> viz = StructureVisualizer()
>>> structure = np.load('structure.npy')
>>> viz.plot_3d(structure, save_path='structure.png')
>>> 
>>> # Plot results
>>> plotter = ResultsPlotter()
>>> plotter.plot_accuracy_vs_dose(doses, accuracies)
"""

from .structure_viz import StructureVisualizer, plot_height_map
from .results_plotting import ResultsPlotter

# Note: create_paper_figure_1 available but not exported by default
# Import directly if needed: from visualization.results_plotting import create_paper_figure_1

__version__ = "1.0.0"

__all__ = [
    'StructureVisualizer',
    'ResultsPlotter',
    'plot_height_map',
]