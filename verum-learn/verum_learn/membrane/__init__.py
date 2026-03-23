"""
Membrane Computing Module
=========================

Lipid membrane signal transduction and biological semiconductor circuits
for autonomous vehicle sensing via categorical trajectory completion.

Implements the complete membrane computing stack derived from first principles:
  Lipid oscillators → Carriers → P-N Junctions → BMD Transistors →
  Logic Gates → ALU → Memory → Sensor Circuit

All parameters derived from bounded phase space axiom with zero free parameters.
"""

from .lipid import Lipid, LipidArray, LipidPhase
from .carriers import OscillatoryHole, MolecularCarrier, CarrierPopulation
from .junction import PNJunction
from .transistor import BMDTransistor
from .logic_gates import TriDimensionalGate
from .alu import VirtualALU
from .memory import SDictionaryMemory
from .s_entropy import SEntropyCoordinate, compute_s_entropy
from .ensemble import PhaseLockedEnsemble, O2Ensemble
from .sensor_circuit import SensorCircuit
from .validation import validate_signal_transduction, run_all_validations

__all__ = [
    "Lipid", "LipidArray", "LipidPhase",
    "OscillatoryHole", "MolecularCarrier", "CarrierPopulation",
    "PNJunction",
    "BMDTransistor",
    "TriDimensionalGate",
    "VirtualALU",
    "SDictionaryMemory",
    "SEntropyCoordinate", "compute_s_entropy",
    "PhaseLockedEnsemble", "O2Ensemble",
    "SensorCircuit",
    "validate_signal_transduction", "run_all_validations",
]
