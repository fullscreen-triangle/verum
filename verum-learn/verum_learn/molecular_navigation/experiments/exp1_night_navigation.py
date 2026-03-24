"""
Experiment 1: Photon-Independent Navigation
=============================================

Validates: Vehicles detectable in total darkness via atmospheric
S-entropy perturbations (thermal + molecular + pressure).

Setup: Virtual atmosphere with vehicles at known positions.
Test: Detect vehicles using only S-entropy (no photons).
Pass criterion: Detection at 50+ distance units in zero-light.
"""

import numpy as np
from ..virtual_atmosphere import VirtualAtmosphere


def run() -> dict:
    """Run the night navigation experiment."""

    # Create virtual atmosphere
    atm = VirtualAtmosphere(width=200, height=50, base_temperature=300.0)
    atm.populate(n_molecules=2000)

    # Measure baseline S-entropy at observer position
    observer_x, observer_y = 100.0, 25.0
    baseline = atm.measure_at(observer_x, observer_y)

    # Place vehicles at increasing distances
    distances = [10, 20, 30, 50, 70, 100]
    results = []

    for dist in distances:
        # Reset atmosphere
        atm_test = VirtualAtmosphere(width=200, height=50)
        atm_test.populate(n_molecules=2000)

        # Inject vehicle at distance from observer
        vehicle_x = observer_x + dist
        vehicle_y = observer_y
        atm_test.inject_vehicle(vehicle_x, vehicle_y, engine_temp=90, speed=30)

        # Let perturbation propagate
        atm_test.diffuse(dt=0.5, D=2.0)

        # Measure baseline (far from vehicle)
        bl = atm_test.measure_at(0, 25)

        # Detect at observer position
        detected, d_cat = atm_test.detect_perturbation(
            observer_x, observer_y, bl, threshold=0.015
        )

        results.append({
            "distance": dist,
            "detected": detected,
            "d_cat": d_cat,
        })

    # Also test human detection
    atm_human = VirtualAtmosphere(width=200, height=50)
    atm_human.populate(n_molecules=2000)
    bl_h = atm_human.measure_at(observer_x, observer_y)

    atm_human.inject_human(observer_x + 8, observer_y)
    atm_human.diffuse(dt=0.3, D=1.0)
    human_detected, human_d = atm_human.detect_perturbation(
        observer_x, observer_y, bl_h, threshold=0.01
    )

    # Evaluate
    max_detection_distance = 0
    for r in results:
        if r["detected"]:
            max_detection_distance = max(max_detection_distance, r["distance"])

    passed = max_detection_distance >= 50

    return {
        "name": "Photon-Independent Navigation",
        "passed": passed,
        "max_detection_distance": max_detection_distance,
        "target_distance": 50,
        "human_detected_at_8m": human_detected,
        "human_d_cat": human_d,
        "detail": results,
    }
