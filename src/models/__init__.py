"""
Models package for HMM implementation and hyperparameter optimization.

This package provides:
- HiddenMarkovModel: Original backward-compatible HMM class
- DiscreteHMM: New refactored discrete HMM implementation
- BaseHMM: Base class for HMM implementations
- HMMTrainer: Training logic separated from model
- RegimeMapper: Utilities for mapping states to market regimes
- Initialization utilities: Various initialization strategies
"""

# Backward compatibility: export original class
from .hmm_model import HiddenMarkovModel

# New refactored classes
from .base_hmm import BaseHMM
from .discrete_hmm import DiscreteHMM
from .trainer import HMMTrainer
from .utils import RegimeMapper, initialize_random_params, initialize_structured_params, initialize_kmeans_params

__all__ = [
    # Original API (backward compatible)
    'HiddenMarkovModel',
    # New API
    'BaseHMM',
    'DiscreteHMM',
    'HMMTrainer',
    'RegimeMapper',
    'initialize_random_params',
    'initialize_structured_params',
    'initialize_kmeans_params',
]
