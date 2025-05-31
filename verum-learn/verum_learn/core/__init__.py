"""
Core Learning Module

Core algorithms for cross-domain learning and personal AI model training.
This module implements the fundamental learning capabilities of Verum:

- Cross-domain pattern recognition and transfer
- Personal AI model training from multi-domain data
- Fear response learning and implementation
- Pattern validation and performance measurement
"""

from .cross_domain import CrossDomainLearner
from .pattern_transfer import PatternTransfer
from .personal_model import PersonalModelTrainer
from .validation import ValidationFramework

__all__ = [
    "CrossDomainLearner",
    "PatternTransfer",
    "PersonalModelTrainer", 
    "ValidationFramework",
] 