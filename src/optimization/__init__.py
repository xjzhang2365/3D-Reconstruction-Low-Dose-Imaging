"""
Optimization module for structure refinement.

⚠️ CONCEPTUAL FRAMEWORK - Full implementation proprietary.

This module demonstrates the optimization approach from Zhang et al. (2025)
without revealing proprietary algorithm details.

Components:
-----------
- SimulatedAnnealingOptimizer: Main optimization framework
- EnergyFunction: Energy calculation interface
- ForwardModel: TEM image simulation interface

Key Innovation (Proprietary):
----------------------------
- Custom energy function combining image matching + physics
- Optimized cooling schedule for low-dose data
- Integration with MD validation

For the full implementation:
- Contact authors for collaboration
- Wait for publication release

Example Usage
-------------
>>> from src.optimization import SimulatedAnnealingOptimizer
>>> 
>>> # For demonstration, uses pre-computed results
>>> optimizer = SimulatedAnnealingOptimizer()
>>> result = optimizer.load_precomputed_result('path/to/result.npy')
"""

from .simulated_annealing import (
    SimulatedAnnealingOptimizer,
    EnergyFunction,
    ForwardModel
)

__version__ = "1.0.0"
__status__ = "Conceptual Framework - Full implementation pending publication"

__all__ = [
    'SimulatedAnnealingOptimizer',
    'EnergyFunction',
    'ForwardModel',
]