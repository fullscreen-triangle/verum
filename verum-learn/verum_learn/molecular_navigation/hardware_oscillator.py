"""
Hardware Oscillator: Real Timing Jitter as Physical Measurement
================================================================

The computer's hardware oscillators (CPU clock, memory bus, I/O timing)
produce timing jitter that maps to S-entropy coordinates via
precision-by-difference: ΔP = T_ref - t_local.

This is NOT simulation. The timing deviations are real physical
measurements caused by thermal noise, power fluctuations, EM interference,
and quantum fluctuations — the same physics that governs atmospheric
molecular dynamics.

By the oscillator-processor duality (ω ≡ R_compute), each timing
measurement IS a categorical state transition. The jitter IS
information, not noise.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


K_B = 1.381e-23   # Boltzmann constant (J/K)
HBAR = 1.055e-34   # Reduced Planck constant (J·s)


@dataclass
class TimingMeasurement:
    """A single hardware timing measurement."""
    timestamp_ns: int        # Raw nanosecond timestamp
    delta_p: float           # Precision-by-difference (ns)
    s_k: float               # Knowledge entropy [0,1]
    s_t: float               # Temporal entropy [0,1]
    s_e: float               # Evolution entropy [0,1]


class HardwareOscillator:
    """Reads real CPU timing jitter and maps to S-entropy coordinates.

    This is the physical grounding of the entire framework.
    The oscillator reads time.perf_counter_ns() — real hardware clock —
    and extracts categorical state from timing deviations.
    """

    def __init__(self, sample_interval_ns: int = 1000):
        self.sample_interval_ns = sample_interval_ns
        self.reference_period: Optional[float] = None
        self.measurements: list[TimingMeasurement] = []
        self._calibrate()

    def _calibrate(self, n_samples: int = 200) -> None:
        """Establish reference period from baseline measurements."""
        intervals = []
        prev = time.perf_counter_ns()
        for _ in range(n_samples):
            # Minimal work between measurements
            now = time.perf_counter_ns()
            intervals.append(now - prev)
            prev = now
        self.reference_period = np.mean(intervals)
        self._baseline_std = np.std(intervals)

    def read_timing(self) -> int:
        """Read raw nanosecond timestamp from hardware clock."""
        return time.perf_counter_ns()

    def read_jitter(self) -> float:
        """Read timing deviation from expected period (precision-by-difference).

        ΔP = T_ref - t_local
        """
        t_start = time.perf_counter_ns()
        # One "tick" of work
        _ = np.random.random()
        t_end = time.perf_counter_ns()
        t_local = t_end - t_start
        return self.reference_period - t_local

    def read_jitter_batch(self, n: int = 100) -> np.ndarray:
        """Read n timing deviations."""
        jitters = np.empty(n)
        for i in range(n):
            jitters[i] = self.read_jitter()
        return jitters

    def to_s_entropy(self, jitter_sequence: np.ndarray) -> tuple[float, float, float]:
        """Map a sequence of ΔP values to S-entropy coordinates.

        S_k = std(∇ΔP) / (std(ΔP) + ε)    — rate-of-change variability
        S_t = sigmoid(mean(ΔP) × scale)     — central tendency
        S_e = H(ΔP) / H_max                 — histogram entropy
        """
        if len(jitter_sequence) < 3:
            return 0.5, 0.5, 0.5

        grad = np.diff(jitter_sequence)
        std_dp = np.std(jitter_sequence)

        # S_k: variability of rate of change
        s_k = float(np.clip(np.std(grad) / (std_dp + 1e-30), 0, 1))

        # S_t: mean deviation (sigmoid normalised)
        mean_dp = np.mean(jitter_sequence)
        s_t = float(1.0 / (1.0 + np.exp(-mean_dp * 0.01)))

        # S_e: histogram entropy
        hist, _ = np.histogram(jitter_sequence, bins=min(20, len(jitter_sequence) // 3 + 1))
        hist = hist[hist > 0].astype(np.float64)
        hist /= hist.sum()
        entropy = -np.sum(hist * np.log(hist))
        max_entropy = np.log(len(hist)) if len(hist) > 1 else 1.0
        s_e = float(np.clip(entropy / max_entropy, 0, 1)) if max_entropy > 0 else 0.5

        return s_k, s_t, s_e

    def sample_molecule(self, n_jitters: int = 20) -> TimingMeasurement:
        """Create one virtual molecule from hardware timing.

        Each molecule IS a real categorical state — a point in S-entropy
        space derived from actual hardware oscillator measurements.
        """
        jitters = self.read_jitter_batch(n_jitters)
        s_k, s_t, s_e = self.to_s_entropy(jitters)
        timestamp = time.perf_counter_ns()
        delta_p = float(np.mean(jitters))

        m = TimingMeasurement(
            timestamp_ns=timestamp,
            delta_p=delta_p,
            s_k=s_k, s_t=s_t, s_e=s_e,
        )
        self.measurements.append(m)
        return m

    @property
    def categorical_temperature(self) -> float:
        """T_cat = ℏ/k_B × σ_jitter — temperature from jitter variance.

        This is not die temperature. It is the categorical transition rate
        expressed in temperature units via T = (ℏ/k_B)(dM/dt).
        """
        if not self.measurements:
            return 0.0
        jitters = np.array([m.delta_p for m in self.measurements[-100:]])
        sigma = np.std(jitters) * 1e-9  # convert ns to seconds
        return HBAR * sigma / K_B if sigma > 0 else 0.0

    @property
    def categorical_pressure(self) -> float:
        """P_cat = k_B T × (N/V) — pressure from molecule creation rate."""
        if len(self.measurements) < 2:
            return 0.0
        dt = (self.measurements[-1].timestamp_ns - self.measurements[0].timestamp_ns) * 1e-9
        rate = len(self.measurements) / dt if dt > 0 else 0.0
        return K_B * self.categorical_temperature * rate
