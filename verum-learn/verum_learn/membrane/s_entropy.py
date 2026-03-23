"""
S-Entropy Coordinate System
============================

Three-dimensional entropy coordinate system S = (S_k, S_t, S_e) ∈ [0,1]³
mapping oscillatory parameters (ω, φ, A) to categorical state space.

From the oscillation-to-S mapping (Sachikonye 2025, Eq. 15-17):
    S_k = ln(1 + ω) / ln(ω_max)     knowledge entropy (frequency)
    S_t = φ / (2π)                    temporal entropy (phase)
    S_e = tanh(A)                     evolution entropy (amplitude)

where ω_max ≈ 10¹⁵ Hz is the maximum observable frequency.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

# Maximum observable frequency (optical oscillations)
OMEGA_MAX = 1e15  # Hz
LN_OMEGA_MAX = np.log(OMEGA_MAX)


@dataclass
class SEntropyCoordinate:
    """A point in S-entropy space [0,1]³."""
    s_k: float  # knowledge entropy
    s_t: float  # temporal entropy
    s_e: float  # evolution entropy

    def __post_init__(self):
        self.s_k = float(np.clip(self.s_k, 0.0, 1.0))
        self.s_t = float(np.clip(self.s_t, 0.0, 1.0))
        self.s_e = float(np.clip(self.s_e, 0.0, 1.0))

    def to_array(self) -> np.ndarray:
        return np.array([self.s_k, self.s_t, self.s_e])

    @staticmethod
    def from_array(arr: np.ndarray) -> "SEntropyCoordinate":
        return SEntropyCoordinate(s_k=arr[0], s_t=arr[1], s_e=arr[2])

    def categorical_distance(self, other: "SEntropyCoordinate") -> float:
        """Euclidean distance in S-space. Bounded by √3."""
        return float(np.linalg.norm(self.to_array() - other.to_array()))

    def ternary_address(self, depth: int = 20) -> list[int]:
        """Convert S-coordinate to ternary address at given depth.

        Each trit refines one coordinate axis cyclically:
            depth 0,3,6,... → S_k
            depth 1,4,7,... → S_t
            depth 2,5,8,... → S_e
        """
        coords = [self.s_k, self.s_t, self.s_e]
        address = []
        ranges = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]

        for d in range(depth):
            axis = d % 3
            lo, hi = ranges[axis]
            third = (hi - lo) / 3.0
            val = coords[axis]

            if val < lo + third:
                address.append(0)
                ranges[axis] = [lo, lo + third]
            elif val < lo + 2 * third:
                address.append(1)
                ranges[axis] = [lo + third, lo + 2 * third]
            else:
                address.append(2)
                ranges[axis] = [lo + 2 * third, hi]

        return address


def compute_s_entropy(
    omega: float | np.ndarray,
    phi: float | np.ndarray,
    amplitude: float | np.ndarray,
) -> SEntropyCoordinate | list[SEntropyCoordinate]:
    """Map oscillation parameters to S-entropy coordinates.

    Args:
        omega: Angular frequency (rad/s)
        phi: Phase (rad)
        amplitude: Oscillation amplitude (dimensionless or normalized)

    Returns:
        SEntropyCoordinate or list thereof for array inputs.
    """
    scalar = np.isscalar(omega)
    omega = np.atleast_1d(np.asarray(omega, dtype=np.float64))
    phi = np.atleast_1d(np.asarray(phi, dtype=np.float64))
    amplitude = np.atleast_1d(np.asarray(amplitude, dtype=np.float64))

    s_k = np.log(1.0 + np.abs(omega)) / LN_OMEGA_MAX
    s_t = (phi % (2.0 * np.pi)) / (2.0 * np.pi)
    s_e = np.tanh(np.abs(amplitude))

    s_k = np.clip(s_k, 0.0, 1.0)
    s_t = np.clip(s_t, 0.0, 1.0)
    s_e = np.clip(s_e, 0.0, 1.0)

    if scalar:
        return SEntropyCoordinate(s_k=s_k[0], s_t=s_t[0], s_e=s_e[0])

    return [SEntropyCoordinate(s_k=s_k[i], s_t=s_t[i], s_e=s_e[i])
            for i in range(len(omega))]


def inverse_s_entropy(
    coord: SEntropyCoordinate,
) -> tuple[float, float, float]:
    """Recover oscillation parameters (ω, φ, A) from S-entropy coordinates.

    Inverse of compute_s_entropy:
        ω = exp(S_k · ln(ω_max)) - 1
        φ = 2π · S_t
        A = arctanh(S_e)
    """
    omega = np.exp(coord.s_k * LN_OMEGA_MAX) - 1.0
    phi = 2.0 * np.pi * coord.s_t
    amplitude = np.arctanh(np.clip(coord.s_e, 0.0, 1.0 - 1e-10))
    return float(omega), float(phi), float(amplitude)


def precision_by_difference(
    t_ref: float | np.ndarray,
    t_local: float | np.ndarray,
) -> float | np.ndarray:
    """Compute precision-by-difference ΔP = T_ref - t_local.

    The sign and magnitude encode categorical position:
        ΔP > 0 → local clock slow (Branch 0)
        ΔP ≈ 0 → synchronized (Branch 1)
        ΔP < 0 → local clock fast (Branch 2)
    """
    return t_ref - t_local


def delta_p_to_branch(delta_p: float, scale: float = 1e9) -> int:
    """Map precision-by-difference to ternary branch index.

    b = floor(3 · |ΔP · scale|) mod 3
    """
    return int(np.floor(3.0 * abs(delta_p * scale))) % 3


def delta_p_to_s_coordinates(
    delta_p_sequence: np.ndarray,
) -> SEntropyCoordinate:
    """Convert a sequence of ΔP values to S-entropy coordinates.

    From Proposition 4.2 (Signature-to-Coordinate Conversion):
        S_k = std(∇ΔP)        (rate of change variability)
        S_t = mean(ΔP)         (central tendency)
        S_e = H(ΔP)            (histogram entropy)
    """
    grad = np.diff(delta_p_sequence)
    s_k = float(np.std(grad)) if len(grad) > 0 else 0.0

    s_t_raw = float(np.mean(delta_p_sequence))
    # Normalize to [0,1] via sigmoid-like transform
    s_t = 1.0 / (1.0 + np.exp(-s_t_raw * 1e6))

    # Histogram entropy
    hist, _ = np.histogram(delta_p_sequence, bins=min(50, len(delta_p_sequence) // 2 + 1))
    hist = hist[hist > 0].astype(np.float64)
    hist /= hist.sum()
    entropy = -np.sum(hist * np.log(hist))
    max_entropy = np.log(len(hist)) if len(hist) > 1 else 1.0
    s_e = entropy / max_entropy if max_entropy > 0 else 0.0

    # Normalize s_k
    s_k = min(s_k / (np.std(delta_p_sequence) + 1e-30), 1.0)

    return SEntropyCoordinate(s_k=s_k, s_t=s_t, s_e=s_e)
