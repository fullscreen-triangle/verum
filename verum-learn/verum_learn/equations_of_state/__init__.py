"""
Equations of State for Vehicular Trajectory Completion
=======================================================

Validation module for the vehicular equations of state derived from
bounded phase space. Implements all ten validation experiments from
the paper "Equations of State for Vehicular Trajectory Completion
in Bounded Phase Space".
"""

from .validation import run_all

__all__ = ["run_all"]
