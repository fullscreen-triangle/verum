"""
Experiment 4: Sweet Spot Discovery (Optimal Path Extraction)
=============================================================

Validates: After N >> 1 vehicles, cumulative exhaust trail C(x,y)
encodes the collectively-discovered optimal driving path.

C(x,y) ∝ N · P_optimal(x,y)

Setup: 500 virtual drivers traverse a road segment, each following
near-optimal path (Gaussian around true optimal, σ ~ 0.5m).
Test: Extract optimal path from trail peak, compare with ground truth.
Pass criterion: Extracted path within 1σ of true optimal.
"""

import numpy as np
from ..exhaust_trail import RoadTrailMap


def run() -> dict:
    """Run the sweet spot discovery experiment."""

    road_length = 100.0
    road_width = 10.0
    n_drivers = 500
    sigma_driver = 0.5  # driver path variability (metres)

    # True optimal path: a smooth curve through the road
    # This represents the racing line / safest line / most efficient path
    x_coords = np.linspace(0, road_length, 200)
    true_optimal = 5.0 + 1.5 * np.sin(2 * np.pi * x_coords / road_length)
    # Ensure within road bounds
    true_optimal = np.clip(true_optimal, 0.5, road_width - 0.5)

    # Create trail map
    trail_map = RoadTrailMap(road_length=road_length, road_width=road_width,
                              resolution=0.5)

    # Simulate N drivers, each following near-optimal path
    for driver in range(n_drivers):
        # Each driver follows optimal + Gaussian noise
        driver_path = true_optimal + np.random.normal(0, sigma_driver, len(x_coords))
        driver_path = np.clip(driver_path, 0.1, road_width - 0.1)

        # Resample to match trail map grid
        path_resampled = np.interp(
            trail_map.x_coords,
            x_coords,
            driver_path,
        )
        trail_map.add_vehicle_path(path_resampled, sigma=sigma_driver)

    # Extract optimal path from trail concentration peak
    extracted_path = trail_map.optimal_path()

    # Compare with true optimal (resampled to same grid)
    true_optimal_resampled = np.interp(
        trail_map.x_coords,
        x_coords,
        true_optimal,
    )

    # Compute error: RMS deviation between extracted and true
    error = np.sqrt(np.mean((extracted_path - true_optimal_resampled)**2))

    # Also check: trail peak quality
    trail_quality = trail_map.path_quality(extracted_path)
    random_quality = trail_map.path_quality(
        np.random.uniform(1, road_width - 1, len(extracted_path))
    )
    quality_ratio = trail_quality / (random_quality + 1e-10)

    # Hazard detection: add a gap (hazard) at one position
    # Drivers avoid x=50, y=5 (pothole)
    hazard_x_idx = trail_map.nx // 2
    hazard_y_idx = trail_map.ny // 2
    # Reduce concentration at hazard
    trail_map.grid[hazard_x_idx - 2:hazard_x_idx + 2,
                   hazard_y_idx - 1:hazard_y_idx + 1] *= 0.05
    hazard_map = trail_map.hazard_map(threshold_fraction=0.15)
    hazard_detected = hazard_map[hazard_x_idx, hazard_y_idx]

    passed = error < sigma_driver and quality_ratio > 3.0

    return {
        "name": "Sweet Spot Discovery",
        "passed": passed,
        "n_drivers": n_drivers,
        "rms_error_m": error,
        "sigma_driver_m": sigma_driver,
        "error_within_1sigma": error < sigma_driver,
        "trail_quality_optimal": trail_quality,
        "trail_quality_random": random_quality,
        "quality_ratio": quality_ratio,
        "hazard_detected": bool(hazard_detected),
    }
