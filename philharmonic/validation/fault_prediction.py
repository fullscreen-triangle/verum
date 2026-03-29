"""
Test 2: Fault Prediction — detect failures before they happen.

Load a session with known retirements (or use synthetic data with injected
faults), run Philharmonic on each lap, track backward trajectories, and
detect when a node escapes its healthy attractor.
"""

import numpy as np
from typing import Dict, List

from ..circuit import F1CircuitGraph
from ..trajectory import TrajectoryCompletion
from ..telemetry import (
    load_session, get_lap_telemetry, get_retirement_info,
    telemetry_to_observations, generate_synthetic_telemetry,
)


def run(use_fastf1: bool = True, verbose: bool = True) -> Dict:
    """
    Run fault prediction validation.

    Strategy:
        1. Generate a baseline (healthy) stint of ~20 laps
        2. Inject a fault at lap 15 (30% conductance drop in ICE→Turbo edge)
        3. Run completion on every lap
        4. Detect when the faulty node's trajectory diverges from healthy

    Returns dict with:
        - healthy_baseline: dict of node -> list of per-lap values
        - faulty_trajectory: dict of node -> list of per-lap values
        - deviations: per-lap deviation dict
        - detection_lap: lap at which fault is first detected
        - fault_lap: actual fault injection lap
        - detection_lead_time: fault_lap - detection_lap
        - per_node_deviation: final deviation per node
    """
    graph = F1CircuitGraph()
    tc_healthy = TrajectoryCompletion(graph)
    tc_faulty = TrajectoryCompletion(graph)

    # --- Generate synthetic telemetry (20 laps) ---------------------------
    n_laps = 20
    fault_lap = 15
    fault_node_edge = ("ICE", "Turbo")  # simulated fuel system issue
    conductance_drop = 0.30  # 30% reduction

    all_samples = generate_synthetic_telemetry(n_laps=n_laps, sample_rate=4.0, seed=42)

    # Organise by lap
    laps_data = {}
    for s in all_samples:
        lap = s["LapNumber"]
        if lap not in laps_data:
            laps_data[lap] = []
        laps_data[lap].append(s)

    if verbose:
        print(f"  Generated {n_laps} synthetic laps")
        print(f"  Fault injection: {fault_node_edge[0]}->{fault_node_edge[1]} "
              f"conductance -{conductance_drop*100:.0f}% at lap {fault_lap}")

    # --- Build faulty graph -----------------------------------------------
    faulty_graph = F1CircuitGraph()
    ia = faulty_graph.node_index[fault_node_edge[0]]
    ib = faulty_graph.node_index[fault_node_edge[1]]

    # --- Run healthy and faulty completions per lap -----------------------
    healthy_per_lap = {n: [] for n in graph.node_names}
    faulty_per_lap = {n: [] for n in graph.node_names}
    deviations_per_lap = []
    healthy_refs = {}

    for lap in range(1, n_laps + 1):
        samples = laps_data.get(lap, [])
        if not samples:
            continue

        # Use lap midpoint sample as representative
        mid = samples[len(samples) // 2]
        obs = telemetry_to_observations(mid)

        # --- Healthy completion ---
        tc_healthy.set_observations(obs)
        state_h, _ = tc_healthy.complete(max_iter=30, tol=1e-5)

        # --- Faulty completion ---
        if lap >= fault_lap:
            # Build a fresh faulty graph each lap (progressive degradation)
            faulty_graph = F1CircuitGraph()
            laps_since_fault = lap - fault_lap
            degradation = conductance_drop * (1.0 + 0.15 * laps_since_fault)
            degradation = min(degradation, 0.90)  # cap at 90%
            faulty_graph._adj[ia, ib] *= (1.0 - degradation)
            faulty_graph._adj[ib, ia] = faulty_graph._adj[ia, ib]
            tc_faulty_g = TrajectoryCompletion(faulty_graph)
        else:
            tc_faulty_g = TrajectoryCompletion(graph)  # same as healthy

        tc_faulty_g.set_observations(obs)
        state_f, _ = tc_faulty_g.complete(max_iter=30, tol=1e-5)

        for n in graph.node_names:
            healthy_per_lap[n].append(state_h.get(n, 0.0))
            faulty_per_lap[n].append(state_f.get(n, 0.0))

        # Store first healthy lap as reference
        if lap == 1:
            healthy_refs = dict(state_h)

        # Compute deviation
        dev = tc_faulty_g.detect_fault(healthy_refs, threshold=0.1)
        deviations_per_lap.append(dev)

    # --- Detection: find first lap where max deviation exceeds threshold --
    threshold = 0.15
    detection_lap = None
    for lap_idx, dev in enumerate(deviations_per_lap):
        max_dev = max(dev.values()) if dev else 0.0
        if max_dev > threshold:
            detection_lap = lap_idx + 1
            break

    if detection_lap is None:
        detection_lap = n_laps  # not detected

    lead_time = fault_lap - detection_lap
    if lead_time < 0:
        lead_time = 0

    # Per-node final deviation
    final_dev = deviations_per_lap[-1] if deviations_per_lap else {}

    if verbose:
        print(f"\n  Detection threshold: {threshold}")
        print(f"  First detection at lap: {detection_lap}")
        print(f"  Actual fault lap: {fault_lap}")
        print(f"  Lead time: {lead_time} laps")
        print(f"\n  Per-node deviation at final lap:")
        for n, d in sorted(final_dev.items(), key=lambda x: -x[1])[:8]:
            marker = " <<<" if n in fault_node_edge else ""
            print(f"    {n:15s}: {d:.4f}{marker}")

    return {
        "healthy_baseline": healthy_per_lap,
        "faulty_trajectory": faulty_per_lap,
        "deviations": deviations_per_lap,
        "detection_lap": detection_lap,
        "fault_lap": fault_lap,
        "detection_lead_time": lead_time,
        "per_node_deviation": final_dev,
        "n_laps": n_laps,
        "threshold": threshold,
    }


if __name__ == "__main__":
    results = run(use_fastf1=True, verbose=True)
