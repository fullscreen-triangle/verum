"""
Virtual Arithmetic Logic Unit (ALU)
====================================

Arithmetic via oscillatory operations:
    Addition:        frequency sum (ω_a + ω_b)
    Multiplication:  phase product (φ_a + φ_b mod 2π → frequency domain)
    Phase Shift:     direct phase addition
    Freq Modulation: amplitude-modulated frequency change

The ALU operates on S-entropy coordinates, performing arithmetic
in categorical space rather than binary register space.

Performance (from Sachikonye 2025, Section 5.4):
    Clock:           758 Hz
    Operations:      add, multiply, phase_shift, freq_mod
    Precision:       determined by partition depth (20 trits → 3⁻²⁰ ≈ 10⁻¹⁰)
"""

import numpy as np
from dataclasses import dataclass

from .s_entropy import SEntropyCoordinate, compute_s_entropy, OMEGA_MAX


@dataclass
class VirtualALU:
    """Arithmetic Logic Unit operating on oscillatory S-entropy states.

    All operations are defined on (ω, φ, A) triples, which map to
    S-entropy coordinates via the standard mapping.
    """
    precision_depth: int = 20  # ternary digits of precision

    @property
    def resolution(self) -> float:
        """Minimum distinguishable difference: 3⁻ᵈ."""
        return 3.0 ** (-self.precision_depth)

    def add(self, a: SEntropyCoordinate, b: SEntropyCoordinate) -> SEntropyCoordinate:
        """Addition: combine oscillatory modes.

        In the oscillatory picture:
            ω_result = ω_a + ω_b (frequency sum)
            φ_result = (φ_a + φ_b) / 2 (mean phase)
            A_result = √(A_a² + A_b²) (energy conservation)

        In S-coordinates: normalized combination.
        """
        s_k = min(a.s_k + b.s_k, 1.0)  # frequency sum, capped
        s_t = (a.s_t + b.s_t) / 2.0     # mean phase
        s_e = np.sqrt(a.s_e**2 + b.s_e**2) / np.sqrt(2)  # RMS amplitude
        return SEntropyCoordinate(s_k=s_k, s_t=s_t, s_e=min(s_e, 1.0))

    def multiply(self, a: SEntropyCoordinate, b: SEntropyCoordinate) -> SEntropyCoordinate:
        """Multiplication: frequency-domain product.

        In the oscillatory picture:
            ω_result = ω_a · ω_b / ω_ref (normalized product)
            φ_result = (φ_a + φ_b) mod 2π (phase addition)
            A_result = A_a · A_b (amplitude product)

        In S-coordinates: product in each dimension.
        """
        s_k = a.s_k * b.s_k         # frequency product (normalized)
        s_t = (a.s_t + b.s_t) % 1.0  # phase sum mod 1
        s_e = a.s_e * b.s_e          # amplitude product
        return SEntropyCoordinate(s_k=s_k, s_t=s_t, s_e=s_e)

    def phase_shift(self, a: SEntropyCoordinate, delta_phi: float) -> SEntropyCoordinate:
        """Phase shift: rotate in temporal dimension.

        S_t → (S_t + δφ/(2π)) mod 1.0
        """
        new_s_t = (a.s_t + delta_phi / (2 * np.pi)) % 1.0
        return SEntropyCoordinate(s_k=a.s_k, s_t=new_s_t, s_e=a.s_e)

    def frequency_modulate(self, carrier: SEntropyCoordinate,
                           modulator: SEntropyCoordinate,
                           depth: float = 0.1) -> SEntropyCoordinate:
        """Frequency modulation: modulator shifts carrier frequency.

        S_k_result = S_k_carrier + depth · S_e_modulator
        """
        new_s_k = np.clip(carrier.s_k + depth * modulator.s_e, 0, 1)
        return SEntropyCoordinate(s_k=new_s_k, s_t=carrier.s_t, s_e=carrier.s_e)

    def subtract(self, a: SEntropyCoordinate, b: SEntropyCoordinate) -> SEntropyCoordinate:
        """Subtraction: frequency difference."""
        s_k = max(a.s_k - b.s_k, 0.0)
        s_t = (a.s_t - b.s_t) % 1.0
        s_e = abs(a.s_e - b.s_e)
        return SEntropyCoordinate(s_k=s_k, s_t=s_t, s_e=s_e)

    def compare(self, a: SEntropyCoordinate, b: SEntropyCoordinate) -> int:
        """Compare: return -1 (a<b), 0 (a≈b), 1 (a>b) by categorical distance."""
        diff = np.linalg.norm(a.to_array() - b.to_array())
        if diff < self.resolution:
            return 0
        # Compare by dominant coordinate
        if a.s_k > b.s_k:
            return 1
        elif a.s_k < b.s_k:
            return -1
        return 0

    def categorical_distance(self, a: SEntropyCoordinate,
                              b: SEntropyCoordinate) -> float:
        """Compute categorical distance between two states."""
        return a.categorical_distance(b)
