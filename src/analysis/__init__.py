"""
Analysis module for structural characterization.

Standard analysis tools for reconstructed structures.
All methods are fully implemented (not proprietary).

Analyses include:
- Bond length distributions
- Strain mapping
- Height distributions (rippling)
- Geometric gradients (curvature)
- Structure comparisons

Example Usage
-------------
>>> from src.analysis import StructureAnalyzer
>>> 
>>> analyzer = StructureAnalyzer()
>>> structure = np.load('reconstructed_structure.npy')
>>> results = analyzer.analyze_full(structure)
>>> 
>>> print(f"RMS roughness: {results['heights']['rms_roughness']:.3f} Å")
"""

from .structure_analysis import StructureAnalyzer, compare_structures

__version__ = "1.0.0"

__all__ = [
    'StructureAnalyzer',
    'compare_structures',
]