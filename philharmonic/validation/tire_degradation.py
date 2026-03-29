"""
Test 3: Tire Degradation — predict the tire performance cliff.

Build a circuit graph with tire-wear-dependent conductance, track tire
node categorical depth over a stint, identify the "cliff" (rapid
increase in depth = loss of grip), and compare with actual pit stop lap.
"""

import numpy as np
from typing import Dict

from ..circuit import F1CircuitGraph
from ..trajectory import TrajectoryCompletion
from ..telemetry import (
    generate_synthetic_telemetry, generate_synthetic_lap_times,
    telemetry_to_observations,
)


def run(use_fastf1: bool = True, verbose: bool = True) -> Dict:
    """
    Run tire degradation validation.

    Returns dict with:
        - lap_times: array of lap times
        - tire_depth: per-lap categorical depth of tire nodes
        - predicted_cliff_lap: lap where cliff is predicted
        - actual_pit_lap: lap where pit stop actually occurs
        - degradation_curve: normalised degradation values
        - tire_state_surface: (lap, speed, brake) -> tire state  (for 3D plot)
    """
    n_laps = 30
    base_time = 90.0
    actual_pit_lap = 25  # simulate: driver pits at lap 25

    # --- Generate lap times with degradation curve -------------------------
    lap_times = generate_synthetic_lap_times(n_laps=n_laps, base_time=base_time, seed=42)

    if verbose:
        print(f"  Generated {n_laps} synthetic laps (pit at lap {actual_pit_lap})")
        print(f"  Lap time range: {lap_times.min():.2f} - {lap_times.max():.2f} s")

    # --- Generate telemetry for each lap -----------------------------------
    all_samples = generate_synthetic_telemetry(n_laps=n_laps, sample_rate=4.0, seed=42)
    laps_data = {}
    for s in all_samples:
        lap = s["LapNumber"]
        if lap not in laps_data:
            laps_data[lap] = []
        laps_data[lap].append(s)

    # --- Build wear-dependent graph and run completion per lap -------------
    tire_nodes = ["FL_Wheel", "FR_Wheel", "RL_Wheel", "RR_Wheel"]
    tire_depth = {n: [] for n in tire_nodes}
    tire_depth_mean = []

    # Surface data for 3D plot: (lap, avg_speed, avg_brake) -> mean_tire_state
    surface_laps = []
    surface_speeds = []
    surface_brakes = []
    surface_tire = []

    for lap in range(1, n_laps + 1):
        # Tire wear factor: conductance degrades with lap
        wear = 1.0 - 0.015 * lap  # linear part
        if lap > 22:
            wear -= 0.05 * (lap - 22) ** 1.5  # cliff
        wear = max(wear, 0.1)

        # Build a worn graph
        graph = F1CircuitGraph()
        for tn in tire_nodes:
            ti = graph.node_index[tn]
            # Reduce all conductances touching tire nodes
            graph._adj[ti, :] *= wear
            graph._adj[:, ti] *= wear

        tc = TrajectoryCompletion(graph)

        samples = laps_data.get(lap, [])
        if not samples:
            for tn in tire_nodes:
                tire_depth[tn].append(0.0)
            tire_depth_mean.append(0.0)
            continue

        # Average observation across the lap
        obs_list = [telemetry_to_observations(s) for s in samples]
        avg_obs = {}
        for key in obs_list[0]:
            avg_obs[key] = np.mean([o.get(key, 0.0) for o in obs_list])

        tc.set_observations(avg_obs)
        state, _ = tc.complete(max_iter=30, tol=1e-5)

        # Categorical depth for tire nodes
        potentials = graph.node_potentials(state)
        depths = []
        for tn in tire_nodes:
            d = potentials[graph.node_index[tn]]
            tire_depth[tn].append(float(d))
            depths.append(float(d))
        tire_depth_mean.append(np.mean(depths))

        # Surface data
        avg_speed = np.mean([s["Speed"] for s in samples])
        avg_brake = np.mean([s["Brake"] for s in samples])
        surface_laps.append(lap)
        surface_speeds.append(avg_speed)
        surface_brakes.append(avg_brake)
        surface_tire.append(np.mean(depths))

    # --- Detect cliff: largest second derivative of tire_depth_mean --------
    tdm = np.array(tire_depth_mean)
    if len(tdm) > 4:
        # Smooth first
        kernel = np.ones(3) / 3.0
        tdm_smooth = np.convolve(tdm, kernel, mode="same")
        d1 = np.diff(tdm_smooth)
        d2 = np.diff(d1)
        # Cliff = lap with maximum acceleration in depth
        cliff_idx = np.argmax(d2) + 2  # offset for double diff
        predicted_cliff_lap = int(cliff_idx + 1)
    else:
        predicted_cliff_lap = n_laps

    cliff_error = abs(predicted_cliff_lap - actual_pit_lap)

    # --- Degradation curve (normalised) -----------------------------------
    degradation_curve = (tdm - tdm.min()) / (tdm.max() - tdm.min() + 1e-12)

    if verbose:
        print(f"\n  Predicted cliff lap: {predicted_cliff_lap}")
        print(f"  Actual pit lap: {actual_pit_lap}")
        print(f"  Prediction error: {cliff_error} laps")
        print(f"  Tire depth (mean) range: {tdm.min():.4f} - {tdm.max():.4f}")

    return {
        "lap_times": lap_times,
        "tire_depth": tire_depth,
        "tire_depth_mean": tire_depth_mean,
        "predicted_cliff_lap": predicted_cliff_lap,
        "actual_pit_lap": actual_pit_lap,
        "cliff_error": cliff_error,
        "degradation_curve": degradation_curve.tolist(),
        "surface_data": {
            "laps": surface_laps,
            "speeds": surface_speeds,
            "brakes": surface_brakes,
            "tire_state": surface_tire,
        },
        "n_laps": n_laps,
    }


if __name__ == "__main__":
    results = run(use_fastf1=True, verbose=True)
