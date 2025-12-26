"""
Validation module for physical plausibility checks.

⚠️ CONCEPTUAL FRAMEWORK - Requires MD software.

Shows the validation approach without requiring full MD installation.

Example Usage
-------------
>>> from src.validation import MDValidator
>>> 
>>> # Conceptual framework (no actual MD)
>>> validator = MDValidator(backend='conceptual')
>>> result = validator.validate(structure)
"""

from .md_validator import MDValidator

__version__ = "1.0.0"

__all__ = ['MDValidator']