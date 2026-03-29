"""
Test 4: Racing Line — extract optimal path from S-entropy analysis.

For each mini-sector, compute the S-entropy of speed/throttle/brake traces.
The fastest lap's S-entropy trace IS the optimal path in S-space.
Compare with the aggregate of all laps (the "molecular trail").
"""

import numpy as np
from typing import Dict, List

from ..circuit import F1CircuitGraph
from ..telemetry import generate_synthetic_qualifying


def _s_entropy(x: np.ndarray, n_bins: int = 20) -> float:
    """
    Compute Shannon entropy of a 1D signal.

    S = -sum p_i log(p_i)    (base e)
    """
    if len(x) == 0 or np.std(x) < 1e-12:
        return 0.0
    hist, _ = np.histogram(x, bins=n_bins, density=True)
    hist = hist[hist > 0]
    # Normalise to probability
    p = hist / hist.sum()
    return float(-np.sum(p * np.log(p + 1e-30)))


def _normalised_entropy(x: np.ndarray, n_bins: int = 20) -> float:
    """S-entropy normalised to [0, 1] by dividing by log(n_bins)."""
    s = _s_entropy(x, n_bins)
    return s / (np.log(n_bins) + 1e-30)


def run(use_fastf1: bool = True, verbose: bool = True) -> Dict:
    """
    Run racing line validation.

    Returns dict with:
        - fastest_lap_idx: index of fastest lap
        - speed_traces: list of speed arrays per lap
        - entropy_traces: dict with S_k (speed), S_t (throttle), S_e (brake) per sector
        - fastest_entropy: S-entropy trace for fastest lap
        - aggregate_entropy: average S-entropy across all laps
        - sector_times: per-lap sector time estimates
        - optimal_sectors: predicted optimal sector times
    """
    # --- Load qualifying data ----------------------------------------------
    n_laps = 8
    laps_data, fastest_idx = generate_synthetic_qualifying(
        n_laps=n_laps, lap_time=87.0, sample_rate=4.0, seed=99
    )

    if verbose:
        print(f"  Generated {n_laps} qualifying laps")
        print(f"  Fastest lap: #{fastest_idx + 1}")

    # --- Divide each lap into sectors (3 sectors) -------------------------
    n_sectors = 3
    speed_traces = []
    throttle_traces = []
    brake_traces = []

    for lap in laps_data:
        speeds = np.array([s["Speed"] for s in lap])
        throttles = np.array([s["Throttle"] for s in lap])
        brakes = np.array([s["Brake"] for s in lap])
        speed_traces.append(speeds)
        throttle_traces.append(throttles)
        brake_traces.append(brakes)

    # --- Compute S-entropy per sector per lap -----------------------------
    # S_k = entropy of speed, S_t = entropy of throttle, S_e = entropy of brake
    entropy_S_k = []  # shape: (n_laps, n_sectors)
    entropy_S_t = []
    entropy_S_e = []

    for lap_idx in range(n_laps):
        speeds = speed_traces[lap_idx]
        throttles = throttle_traces[lap_idx]
        brakes = brake_traces[lap_idx]
        n = len(speeds)
        sector_len = n // n_sectors

        sk_lap = []
        st_lap = []
        se_lap = []
        for s in range(n_sectors):
            start = s * sector_len
            end = (s + 1) * sector_len if s < n_sectors - 1 else n
            sk_lap.append(_normalised_entropy(speeds[start:end]))
            st_lap.append(_normalised_entropy(throttles[start:end]))
            se_lap.append(_normalised_entropy(brakes[start:end]))

        entropy_S_k.append(sk_lap)
        entropy_S_t.append(st_lap)
        entropy_S_e.append(se_lap)

    entropy_S_k = np.array(entropy_S_k)
    entropy_S_t = np.array(entropy_S_t)
    entropy_S_e = np.array(entropy_S_e)

    # --- Fastest lap's S-entropy IS the optimal path ----------------------
    fastest_Sk = entropy_S_k[fastest_idx]
    fastest_St = entropy_S_t[fastest_idx]
    fastest_Se = entropy_S_e[fastest_idx]

    # Aggregate across all laps
    agg_Sk = entropy_S_k.mean(axis=0)
    agg_St = entropy_S_t.mean(axis=0)
    agg_Se = entropy_S_e.mean(axis=0)

    # --- Sector time estimation -------------------------------------------
    # Approximate sector time ~ base_time / n_sectors * (1 + deviation_factor)
    base_sector = 87.0 / n_sectors
    sector_times = []
    for lap_idx in range(n_laps):
        speeds = speed_traces[lap_idx]
        n = len(speeds)
        sector_len = n // n_sectors
        st = []
        for s in range(n_sectors):
            start = s * sector_len
            end = (s + 1) * sector_len if s < n_sectors - 1 else n
            avg_speed = np.mean(speeds[start:end])
            # Time proportional to 1/speed
            st.append(base_sector * (200.0 / (avg_speed + 1e-6)))
        sector_times.append(st)

    sector_times = np.array(sector_times)
    total_times = sector_times.sum(axis=1)

    # Optimal = fastest actual sector time for each sector (theoretical best)
    optimal_sectors = sector_times.min(axis=0)
    fastest_sectors = sector_times[fastest_idx]

    # --- Speed trace comparison: fastest vs average -----------------------
    avg_speed = np.mean(speed_traces, axis=0) if len(speed_traces) > 0 else np.array([])
    fastest_speed = speed_traces[fastest_idx] if fastest_idx < len(speed_traces) else np.array([])

    # --- Full S-entropy trajectory for 3D plot ----------------------------
    # Fine-grained: compute S-entropy in rolling windows along the lap
    n_windows = 50
    fastest_speeds = speed_traces[fastest_idx]
    fastest_throttles = throttle_traces[fastest_idx]
    fastest_brakes = brake_traces[fastest_idx]
    win_len = len(fastest_speeds) // n_windows

    fine_Sk = []
    fine_St = []
    fine_Se = []
    for w in range(n_windows):
        start = w * win_len
        end = start + win_len
        fine_Sk.append(_normalised_entropy(fastest_speeds[start:end], n_bins=10))
        fine_St.append(_normalised_entropy(fastest_throttles[start:end], n_bins=10))
        fine_Se.append(_normalised_entropy(fastest_brakes[start:end], n_bins=10))

    if verbose:
        print(f"\n  Sector S-entropy (fastest lap):")
        for s in range(n_sectors):
            print(f"    Sector {s+1}: S_k={fastest_Sk[s]:.3f}  "
                  f"S_t={fastest_St[s]:.3f}  S_e={fastest_Se[s]:.3f}")
        print(f"\n  Sector times (fastest vs optimal):")
        for s in range(n_sectors):
            print(f"    Sector {s+1}: fastest={fastest_sectors[s]:.2f}s  "
                  f"optimal={optimal_sectors[s]:.2f}s")
        print(f"  Total: fastest={total_times[fastest_idx]:.2f}s  "
              f"optimal={optimal_sectors.sum():.2f}s")

    return {
        "fastest_lap_idx": int(fastest_idx),
        "speed_traces": [s.tolist() for s in speed_traces],
        "avg_speed": avg_speed.tolist() if len(avg_speed) > 0 else [],
        "fastest_speed": fastest_speed.tolist() if len(fastest_speed) > 0 else [],
        "entropy_traces": {
            "S_k": entropy_S_k.tolist(),
            "S_t": entropy_S_t.tolist(),
            "S_e": entropy_S_e.tolist(),
        },
        "fastest_entropy": {
            "S_k": fastest_Sk.tolist(),
            "S_t": fastest_St.tolist(),
            "S_e": fastest_Se.tolist(),
        },
        "aggregate_entropy": {
            "S_k": agg_Sk.tolist(),
            "S_t": agg_St.tolist(),
            "S_e": agg_Se.tolist(),
        },
        "fine_entropy": {
            "S_k": fine_Sk,
            "S_t": fine_St,
            "S_e": fine_Se,
        },
        "sector_times": sector_times.tolist(),
        "optimal_sectors": optimal_sectors.tolist(),
        "total_times": total_times.tolist(),
        "n_laps": n_laps,
        "n_sectors": n_sectors,
    }


if __name__ == "__main__":
    results = run(use_fastf1=True, verbose=True)
