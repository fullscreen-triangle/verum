"""
Philharmonic — F1 Telemetry Validation for the Vehicle Circuit Graph Framework

This package validates the Philharmonic vehicle circuit graph framework using
real Formula 1 telemetry data. It builds a 20-node circuit graph representing
an F1 car's major subsystems, applies trajectory completion from partial
observations, and validates against known F1 behaviour.

Four validation tests:
    1. State Reconstruction — reconstruct hidden states from public telemetry
    2. Fault Prediction — detect failures before they happen
    3. Tire Degradation — predict the tire performance cliff
    4. Racing Line — extract optimal path from S-entropy analysis
"""

__version__ = "0.1.0"
