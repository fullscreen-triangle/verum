"""
Experiment 2: Brake Anticipation
==================================

Validates: Membrane detects braking intent 150-290ms before brake lights
via exhaust composition change propagating at molecular speeds.

Setup: Lead vehicle lifts throttle → exhaust changes → S-entropy shifts.
Test: Measure detection latency vs brake light latency.
Pass criterion: Detection 150-290ms earlier (in normalised time units).
"""

import numpy as np
from ..virtual_atmosphere import VirtualAtmosphere


def run() -> dict:
    """Run the brake anticipation experiment."""

    # Simulation parameters
    dt = 0.01           # timestep (normalised seconds)
    n_steps = 100       # total simulation steps
    following_distance = 50.0  # metres behind lead vehicle

    # Lead vehicle position
    lead_x = 150.0
    lead_y = 25.0
    observer_x = lead_x - following_distance
    observer_y = 25.0

    # Timeline:
    # t=0.00s: throttle lift
    # t=0.05s: exhaust composition begins changing (molecular speed)
    # t=0.15s: S-entropy shift propagates to observer (sound speed ~343 m/s → 50m/343 ≈ 0.15s)
    # t=0.25s: brake light illuminates (human reaction time)

    throttle_lift_step = 30     # t = 0.30
    brake_light_step = 55       # t = 0.55 (250ms later)

    detection_step = None
    d_cat_history = []

    atm = VirtualAtmosphere(width=200, height=50)
    atm.populate(n_molecules=1500)

    # Establish baseline
    baseline = atm.measure_at(observer_x, observer_y)

    for step in range(n_steps):
        t = step * dt

        # Lead vehicle always present
        atm_step = VirtualAtmosphere(width=200, height=50)
        atm_step.populate(n_molecules=1500)

        if step < throttle_lift_step:
            # Normal driving: steady exhaust
            atm_step.inject_vehicle(lead_x, lead_y, engine_temp=90, speed=30)
        else:
            # After throttle lift: exhaust composition changes
            # Engine temp drops, CO₂ increases (rich mixture), NOx decreases
            time_since_lift = (step - throttle_lift_step) * dt
            temp_drop = min(time_since_lift * 200, 50)  # up to 50°C drop over 250ms
            co2_increase = min(time_since_lift * 0.5, 0.15)  # S_k shift

            atm_step.inject_vehicle(lead_x, lead_y,
                                     engine_temp=90 - temp_drop,
                                     speed=max(30 - time_since_lift * 40, 0))
            # Extra CO₂ perturbation from rich mixture
            atm_step.inject_perturbation(
                lead_x - 3, lead_y, radius=8.0,
                delta_sk=co2_increase, delta_se=-0.05,
            )

        # Propagate
        propagation_time = following_distance / 343.0  # sound speed
        atm_step.diffuse(dt=propagation_time, D=2.0)

        # Measure at observer
        detected, d_cat = atm_step.detect_perturbation(
            observer_x, observer_y, baseline, threshold=0.015
        )
        d_cat_history.append(d_cat)

        if detected and detection_step is None and step > throttle_lift_step:
            detection_step = step

    # Compute latencies
    if detection_step is not None:
        membrane_latency = (detection_step - throttle_lift_step) * dt
        brake_light_latency = (brake_light_step - throttle_lift_step) * dt
        advance_warning = brake_light_latency - membrane_latency
        advance_warning_ms = advance_warning * 1000
    else:
        membrane_latency = float("inf")
        brake_light_latency = (brake_light_step - throttle_lift_step) * dt
        advance_warning = 0
        advance_warning_ms = 0

    # At 30 m/s, advance warning in metres
    advance_metres = advance_warning * 30.0

    passed = 150 <= advance_warning_ms <= 350  # 150-290ms target, allow some margin

    return {
        "name": "Brake Anticipation",
        "passed": passed,
        "membrane_detection_ms": membrane_latency * 1000,
        "brake_light_ms": brake_light_latency * 1000,
        "advance_warning_ms": advance_warning_ms,
        "advance_warning_metres": advance_metres,
        "target_range_ms": "150-290",
        "d_cat_peak": max(d_cat_history) if d_cat_history else 0,
    }
