"""
Molecular Navigation Module
============================

Validates molecular navigation claims using virtual gas ensembles
derived from hardware oscillator timing — the same approach as the
Maxwell categorical computing framework.

The computer's timing jitter IS the gas. Each timing deviation maps
to an S-entropy coordinate. Virtual molecules are real categorical
states from real physical measurements.
"""

from .hardware_oscillator import HardwareOscillator
from .virtual_atmosphere import VirtualAtmosphere, VirtualMolecule
from .exhaust_trail import ExhaustTrail, RoadTrailMap
from .road_network import PartitionRoad

__all__ = [
    "HardwareOscillator",
    "VirtualAtmosphere", "VirtualMolecule",
    "ExhaustTrail", "RoadTrailMap",
    "PartitionRoad",
]
