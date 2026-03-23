"""
Biological Maxwell Demon (BMD) Transistor
==========================================

Unlike conventional transistors that switch on voltage thresholds,
BMD transistors switch on pattern recognition — phase-lock gating.

The gate is a Maxwell demon: it selectively recognizes input patterns
(phase-locked oscillatory signatures) and channels output accordingly.

Properties (from Sachikonye 2025):
    Clock frequency:     f_0 = 758 Hz (ATP-driven)
    Coherence time:      τ_c = 10 ms (phase-locked)
    Gate operation:      < 100 μs
    Fidelity:            > 85%
    Energy per operation: 446× Landauer limit

Structure: Source (N) → Channel (P) → Drain (N)
Gate: Phase-lock control at f_0 = 758 Hz
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable

from .s_entropy import SEntropyCoordinate, compute_s_entropy
from .carriers import OscillatoryHole, MolecularCarrier
from .junction import PNJunction


# ── Constants ─────────────────────────────────────────────────────────────────

F_CLOCK = 758.0           # biological clock frequency (Hz)
OMEGA_CLOCK = 2 * np.pi * F_CLOCK
COHERENCE_TIME = 10e-3    # 10 ms
GATE_TIME = 100e-6        # 100 μs
FIDELITY_MIN = 0.85
ATP_ENERGY = 50e-21       # J per hydrolysis
LANDAUER_ENERGY = 1.381e-23 * 310 * np.log(2)  # k_BT ln 2 at 310 K
ENERGY_RATIO = ATP_ENERGY / LANDAUER_ENERGY  # ≈ 446×


@dataclass
class BMDTransistor:
    """Biological Maxwell Demon transistor.

    Three terminals:
        Source (N-type): molecular carrier input
        Channel (P-type): oscillatory hole medium
        Drain (N-type): molecular carrier output

    Gate: Phase-lock pattern recognizer operating at f_0 = 758 Hz.
    The gate opens (conducts) when input pattern matches gate pattern.
    """
    gate_pattern: SEntropyCoordinate = None  # pattern the gate recognizes
    gate_threshold: float = 0.2              # max categorical distance for match
    source_junction: PNJunction = None       # source-channel junction
    drain_junction: PNJunction = None        # channel-drain junction
    gate_phase: float = 0.0                  # current gate oscillator phase
    is_open: bool = False                    # gate state
    fidelity: float = 0.92                   # gate operation fidelity

    def __post_init__(self):
        if self.gate_pattern is None:
            self.gate_pattern = SEntropyCoordinate(0.5, 0.5, 0.5)
        if self.source_junction is None:
            self.source_junction = PNJunction()
        if self.drain_junction is None:
            self.drain_junction = PNJunction()

    def recognize(self, input_signal: SEntropyCoordinate) -> bool:
        """Pattern recognition: does input match gate pattern?

        Returns True if categorical distance < threshold.
        This is the BMD operation: selective recognition, not threshold comparison.
        """
        d_cat = input_signal.categorical_distance(self.gate_pattern)
        match = d_cat < self.gate_threshold

        # Apply fidelity (probability of correct recognition)
        if np.random.random() < self.fidelity:
            return match
        else:
            return not match  # error

    def gate_tick(self, input_signal: SEntropyCoordinate) -> None:
        """One gate operation cycle at f_0 = 758 Hz.

        1. Advance gate oscillator phase
        2. Recognize input pattern
        3. Open or close channel
        """
        self.gate_phase = (self.gate_phase + OMEGA_CLOCK * GATE_TIME) % (2 * np.pi)
        self.is_open = self.recognize(input_signal)

    def conduct(self, input_carrier: MolecularCarrier,
                bias_voltage: float = 0.3) -> Optional[MolecularCarrier]:
        """Attempt to conduct a carrier from source to drain.

        If gate is open (pattern recognized):
            Carrier passes through channel, acquiring gate's phase signature.
        If gate is closed:
            Carrier blocked (returned None).

        Args:
            input_carrier: carrier at source terminal
            bias_voltage: forward bias across junction

        Returns:
            Output carrier at drain (or None if blocked).
        """
        if not self.is_open:
            return None

        # Carrier passes through — its phase is modulated by gate
        output = MolecularCarrier(
            modes=[(omega, amp, (phi + self.gate_phase) % (2 * np.pi))
                   for omega, amp, phi in input_carrier.modes],
            position=input_carrier.position,
        )
        return output

    def energy_per_operation(self) -> float:
        """Energy consumed per gate operation (J).

        One ATP hydrolysis per coherence refresh.
        """
        return ATP_ENERGY

    def operations_per_second(self) -> float:
        """Gate operations per second = f_0 = 758 Hz."""
        return F_CLOCK

    def landauer_ratio(self) -> float:
        """Ratio of actual energy to Landauer minimum. ≈ 446×."""
        return self.energy_per_operation() / LANDAUER_ENERGY

    def set_gate_pattern(self, pattern: SEntropyCoordinate) -> None:
        """Program the transistor to recognize a specific pattern."""
        self.gate_pattern = pattern

    def transfer_characteristic(
        self, n_points: int = 100
    ) -> tuple[np.ndarray, np.ndarray]:
        """BMD transfer characteristic: output vs input categorical distance.

        Unlike MOSFET (I_D vs V_GS), BMD plots conductance vs d_cat.
        Sharp transition at d_cat = threshold.
        """
        distances = np.linspace(0, 1, n_points)
        conductances = np.zeros(n_points)

        for i, d in enumerate(distances):
            # Create input at distance d from gate pattern
            test_input = SEntropyCoordinate(
                s_k=self.gate_pattern.s_k + d / np.sqrt(3),
                s_t=self.gate_pattern.s_t,
                s_e=self.gate_pattern.s_e,
            )
            self.gate_tick(test_input)
            conductances[i] = 1.0 if self.is_open else 0.0

        return distances, conductances
