"""
Biological P-N Junction
=========================

Junction forms at interface between P-type (hole-dominated) and N-type
(carrier-dominated) regions of the biological membrane.

Parameters (from Sachikonye 2025, Section 3.3):
    Built-in potential:    V_bi = k_BT/e · ln(N_A·N_D/n_i²) = 0.78 V
    Saturation current:    I_s = 1.2 × 10⁻¹² A
    Ideality factor:       n = 1.8
    Rectification ratio:   >42 at |V| = 0.5 V

I-V characteristic follows Shockley diode equation:
    I = I_s [exp(eV/(nk_BT)) - 1]
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .carriers import (
    CarrierPopulation, E_CHARGE, K_B, T_PHYSIOL,
    P_DENSITY, N_DENSITY, INTRINSIC_DENSITY, MU_P, MU_N,
)


@dataclass
class PNJunction:
    """Biological P-N junction with Shockley diode characteristics.

    The junction enables directional information flow — forward bias
    injects carriers for computation, reverse bias blocks unwanted flow.
    """
    n_a: float = P_DENSITY          # acceptor (hole) concentration (m⁻³)
    n_d: float = N_DENSITY          # donor (carrier) concentration (m⁻³)
    n_i: float = INTRINSIC_DENSITY  # intrinsic carrier concentration (m⁻³)
    temperature: float = T_PHYSIOL
    ideality_factor: float = 1.8
    saturation_current: float = 1.2e-12  # A
    area: float = 1e-12             # junction area (m²), ~1 μm²

    @property
    def thermal_voltage(self) -> float:
        """V_T = k_BT/e."""
        return K_B * self.temperature / E_CHARGE

    @property
    def built_in_potential(self) -> float:
        """V_bi = V_T · ln(N_A · N_D / n_i²)."""
        return self.thermal_voltage * np.log(self.n_a * self.n_d / self.n_i**2)

    @property
    def depletion_width(self, applied_voltage: float = 0.0) -> float:
        """W = sqrt(2ε(V_bi - V)/e · (1/N_A + 1/N_D)).

        Using biological permittivity ε ≈ 80ε₀ (aqueous).
        """
        epsilon = 80.0 * 8.854e-12  # F/m
        v_eff = self.built_in_potential - applied_voltage
        if v_eff < 0:
            v_eff = 0.0
        return np.sqrt(2 * epsilon * v_eff / E_CHARGE *
                       (1.0 / self.n_a + 1.0 / self.n_d))

    def current(self, voltage: float) -> float:
        """Shockley diode equation: I = I_s [exp(eV/(nk_BT)) - 1].

        Args:
            voltage: Applied voltage (V). Positive = forward bias.

        Returns:
            Current in Amperes.
        """
        vt = self.thermal_voltage
        exponent = voltage / (self.ideality_factor * vt)
        # Clip to prevent overflow
        exponent = np.clip(exponent, -100, 100)
        return self.saturation_current * (np.exp(exponent) - 1.0)

    def current_density(self, voltage: float) -> float:
        """Current density J = I/A (A/m²)."""
        return self.current(voltage) / self.area

    def iv_curve(self, v_min: float = -1.0, v_max: float = 1.0,
                 n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """Generate I-V characteristic curve.

        Returns:
            (voltages, currents) arrays.
        """
        voltages = np.linspace(v_min, v_max, n_points)
        currents = np.array([self.current(v) for v in voltages])
        return voltages, currents

    def rectification_ratio(self, voltage: float = 0.5) -> float:
        """Rectification ratio: I_forward(V) / |I_reverse(-V)|.

        Paper predicts >42 at |V| = 0.5 V.
        """
        i_forward = self.current(abs(voltage))
        i_reverse = abs(self.current(-abs(voltage)))
        if i_reverse < 1e-30:
            return float("inf")
        return i_forward / i_reverse

    def differential_resistance(self, voltage: float) -> float:
        """r_d = dV/dI = nk_BT / (eI) for forward bias."""
        i = self.current(voltage)
        if abs(i) < 1e-30:
            return float("inf")
        return self.ideality_factor * self.thermal_voltage / abs(i)

    def capacitance(self, voltage: float = 0.0) -> float:
        """Junction capacitance C = εA/W."""
        epsilon = 80.0 * 8.854e-12
        w = self.depletion_width
        if w < 1e-15:
            return float("inf")
        return epsilon * self.area / w

    def forward_voltage_at_current(self, target_current: float) -> float:
        """Solve for V given I: V = nV_T · ln(I/I_s + 1)."""
        if target_current <= 0:
            return 0.0
        return (self.ideality_factor * self.thermal_voltage *
                np.log(target_current / self.saturation_current + 1.0))
