"""
Exhaust Trail: Molecular Memory in Road Networks
==================================================

Exhaust trails persist for hours in the atmospheric boundary layer.
After N >> 1 vehicles: C(x,y) ∝ N · P_optimal(x,y).
The trail IS the solved optimization problem.
Hazards are encoded as GAPS in the trail distribution.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrailPoint:
    """A single exhaust emission point in space-time."""
    x: float
    y: float
    time: float
    s_k: float            # exhaust composition signature
    s_t: float            # velocity signature
    s_e: float            # energy signature
    concentration: float = 1.0  # relative concentration
    species: str = "CO2"

    def diffused_concentration(self, x_obs: float, y_obs: float,
                                t_obs: float, D: float = 1.0) -> float:
        """Concentration at (x_obs, y_obs, t_obs) from this emission.

        2D Gaussian diffusion: C = C₀/(4πDt) × exp(-r²/(4Dt))
        """
        dt = t_obs - self.time
        if dt <= 0:
            return 0.0
        r2 = (x_obs - self.x)**2 + (y_obs - self.y)**2
        denom = 4.0 * np.pi * D * dt
        if denom < 1e-30:
            return 0.0
        return self.concentration / denom * np.exp(-r2 / (4.0 * D * dt))


class ExhaustTrail:
    """A single vehicle's exhaust trail over time."""

    def __init__(self, vehicle_id: int = 0):
        self.vehicle_id = vehicle_id
        self.points: list[TrailPoint] = []

    def emit(self, x: float, y: float, time: float,
             s_k: float = 0.6, s_t: float = 0.5, s_e: float = 0.55,
             concentration: float = 1.0) -> None:
        """Emit exhaust at position (x,y) at given time."""
        self.points.append(TrailPoint(
            x=x, y=y, time=time,
            s_k=s_k, s_t=s_t, s_e=s_e,
            concentration=concentration,
        ))

    def concentration_at(self, x: float, y: float, t: float,
                          D: float = 1.0) -> float:
        """Total diffused concentration at (x,y) at time t from this trail."""
        return sum(p.diffused_concentration(x, y, t, D) for p in self.points)

    def s_entropy_at(self, x: float, y: float, t: float,
                      D: float = 1.0) -> Optional[tuple[float, float, float]]:
        """Concentration-weighted S-entropy at (x,y,t).

        Returns weighted average of S-entropy from all contributing trail points.
        """
        total_c = 0.0
        weighted_sk = 0.0
        weighted_st = 0.0
        weighted_se = 0.0
        for p in self.points:
            c = p.diffused_concentration(x, y, t, D)
            if c > 1e-15:
                total_c += c
                weighted_sk += c * p.s_k
                weighted_st += c * p.s_t
                weighted_se += c * p.s_e
        if total_c < 1e-15:
            return None
        return weighted_sk / total_c, weighted_st / total_c, weighted_se / total_c

    def age_at(self, x: float, y: float, t: float, D: float = 1.0) -> float:
        """Estimate age of dominant trail contribution at (x,y).

        Weighted average emission time → age = t - t_emission.
        """
        total_c = 0.0
        weighted_t = 0.0
        for p in self.points:
            c = p.diffused_concentration(x, y, t, D)
            total_c += c
            weighted_t += c * p.time
        if total_c < 1e-15:
            return float("inf")
        return t - weighted_t / total_c


class RoadTrailMap:
    """Cumulative trail map from many vehicles on a road segment.

    After N vehicles: C(x,y) ∝ N · P_optimal(x,y)
    The concentration distribution IS the collective optimal path.
    """

    def __init__(self, road_length: float = 100.0, road_width: float = 10.0,
                 resolution: float = 0.5):
        self.road_length = road_length
        self.road_width = road_width
        self.resolution = resolution

        self.nx = int(road_length / resolution)
        self.ny = int(road_width / resolution)
        self.grid = np.zeros((self.nx, self.ny))  # cumulative concentration
        self.n_vehicles = 0

    @property
    def x_coords(self) -> np.ndarray:
        return np.linspace(0, self.road_length, self.nx)

    @property
    def y_coords(self) -> np.ndarray:
        return np.linspace(0, self.road_width, self.ny)

    def add_vehicle_trail(self, trail: ExhaustTrail, observation_time: float,
                          D: float = 1.0) -> None:
        """Add a vehicle's trail to the cumulative map."""
        for i, x in enumerate(self.x_coords):
            for j, y in enumerate(self.y_coords):
                self.grid[i, j] += trail.concentration_at(x, y, observation_time, D)
        self.n_vehicles += 1

    def add_vehicle_path(self, path_y: np.ndarray, sigma: float = 0.5) -> None:
        """Add a vehicle path directly as Gaussian trail on the grid.

        path_y: y-position at each x-coordinate (same length as self.nx)
        sigma: trail width (standard deviation in metres)
        """
        for i in range(self.nx):
            for j in range(self.ny):
                y = self.y_coords[j]
                self.grid[i, j] += np.exp(-0.5 * ((y - path_y[i]) / sigma)**2)
        self.n_vehicles += 1

    def optimal_path(self) -> np.ndarray:
        """Extract optimal path as argmax_y C(x, y) at each x.

        This IS the sweet spot — the path collectively discovered
        by all previous drivers.
        """
        path = np.zeros(self.nx)
        for i in range(self.nx):
            j_max = np.argmax(self.grid[i, :])
            path[i] = self.y_coords[j_max]
        return path

    def hazard_map(self, threshold_fraction: float = 0.1) -> np.ndarray:
        """Detect hazards as gaps in the trail distribution.

        Positions where C(x,y) < threshold × max(C) are likely hazards
        (drivers avoid them → no exhaust → gap).
        """
        max_c = self.grid.max()
        if max_c < 1e-15:
            return np.zeros_like(self.grid, dtype=bool)
        return self.grid < threshold_fraction * max_c

    def path_quality(self, path_y: np.ndarray) -> float:
        """Evaluate how well a path follows the cumulative trail.

        Returns mean concentration along the path (higher = better).
        """
        total = 0.0
        for i in range(min(len(path_y), self.nx)):
            j = int(np.clip(
                (path_y[i] / self.road_width) * self.ny, 0, self.ny - 1
            ))
            total += self.grid[i, j]
        return total / len(path_y)

    def concentration_profile(self, x_index: int) -> np.ndarray:
        """Get lateral concentration profile at a given x position."""
        return self.grid[x_index, :]
