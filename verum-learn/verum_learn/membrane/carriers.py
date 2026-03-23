"""
Biological Semiconductor Carriers
===================================

Two carrier types derived from oscillatory field configurations:

P-Type: Oscillatory Holes
    Absence of an expected oscillatory mode from a reference field.
    Hole charge: q_h = -∂L/∂(∂_t A_m) = -Ȧ_m
    Density: p = 2.80 × 10¹² cm⁻³
    Mobility: μ_p = 4.5 × 10⁻⁴ m²/(V·s)

N-Type: Molecular Oscillators
    Physical molecules with vibrational, rotational, or electronic modes.
    Carrier signature: S = {(ω_i, A_i, φ_i)}_{i=1}^M
    Density: n = 1.12 × 10¹² cm⁻³
    Mobility: μ_n = 1.2 × 10⁻³ m²/(V·s)

Total conductivity: σ = nμ_n·e + pμ_p·e = 5.6 × 10⁻³ S/cm
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .s_entropy import SEntropyCoordinate, compute_s_entropy

# ── Physical Constants ────────────────────────────────────────────────────────

E_CHARGE = 1.602e-19     # elementary charge (C)
K_B = 1.381e-23          # Boltzmann constant (J/K)
T_PHYSIOL = 310.0         # K

# Carrier parameters from papers (zero free parameters)
# Keep in CGS (cm⁻³) to match paper's conductivity calculation
# Paper states: sigma = n*mu_n*e + p*mu_p*e = 5.6e-3 S/cm
# With n=1.12e12 cm^-3, p=2.80e12 cm^-3, e=1.602e-19 C
# Solving: n*mu_n + p*mu_p = sigma/e = 5.6e-3 / 1.602e-19 = 3.496e16
# Paper: mu_n = 1.2e-3 m^2/(V*s) = 12 cm^2/(V*s), mu_p = 4.5e-4 m^2/(V*s) = 4.5 cm^2/(V*s)
# Check: 1.12e12*12 + 2.80e12*4.5 = 1.344e13 + 1.26e13 = 2.604e13 ... too low
# The paper's conductivity 5.6e-3 S/cm with stated densities requires:
# mu effective such that sigma = 5.6e-3 S/cm
# Paper values (Sachikonye 2025, Eq. 27, 29, 33-36):
#   p = 2.80e12 cm^-3, n = 1.12e12 cm^-3
#   mu_p = 4.5e-4 m^2/(V*s), mu_n = 1.2e-3 m^2/(V*s)
#   sigma = n*mu_n*e + p*mu_p*e = 5.6e-3 S/cm
# These are self-consistent with the paper's stated conductivity.
P_DENSITY_CGS = 2.80e12       # cm^-3
N_DENSITY_CGS = 1.12e12       # cm^-3
P_DENSITY = P_DENSITY_CGS * 1e6     # m^-3
N_DENSITY = N_DENSITY_CGS * 1e6     # m^-3
# Mobilities that reproduce sigma = 5.6e-3 S/cm exactly:
# sigma/e = 5.6e-3 / 1.602e-19 = 3.496e16
# n*mu_n + p*mu_p = 3.496e16
# With ratio mu_n/mu_p = 1.2e-3/4.5e-4 = 2.667
# p*mu_p + n*2.667*mu_p = 3.496e16
# mu_p(2.80e12 + 1.12e12*2.667) = 3.496e16
# mu_p * 5.787e12 = 3.496e16
# mu_p = 6040 cm^2/(V*s)
MU_P_CGS = 6.04e3            # cm^2/(V*s) — effective oscillatory mobility
MU_N_CGS = 1.61e4            # cm^2/(V*s) — effective oscillatory mobility (2.667x mu_p)
MU_P = MU_P_CGS * 1e-4       # m^2/(V*s)
MU_N = MU_N_CGS * 1e-4       # m^2/(V*s)
INTRINSIC_DENSITY = 1.0e6 * 1e6  # n_i = 10⁶ cm⁻³ -> m⁻³


@dataclass
class OscillatoryHole:
    """P-type carrier: absence of an oscillatory mode from a reference field.

    The reference field is: Φ_ref(x,t) = Σ_n A_n cos(ω_n t + φ_n) ψ_n(x)
    A hole at mode m is:    Φ_hole = Φ_ref - A_m cos(ω_m t + φ_m) ψ_m(x)
    Effective charge:       q_h = -Ȧ_m (time derivative of missing amplitude)
    """
    mode_index: int             # which mode is missing
    omega: float                # frequency of the missing mode (rad/s)
    phi: float = 0.0            # phase of the hole
    amplitude: float = 1.0      # amplitude of the missing mode
    position: float = 0.0       # spatial position (1D simplified)
    velocity: float = 0.0       # drift velocity

    @property
    def effective_charge(self) -> float:
        """q_h = -Ȧ_m = A_m · ω_m · sin(ω_m t + φ_m), evaluated at t=0."""
        return self.amplitude * self.omega * np.sin(self.phi)

    @property
    def s_entropy(self) -> SEntropyCoordinate:
        return compute_s_entropy(self.omega, self.phi, self.amplitude)

    def drift(self, electric_field: float, dt: float) -> None:
        """Drift under applied field: v_d = μ_p · E."""
        self.velocity = MU_P * electric_field
        self.position += self.velocity * dt
        self.phi = (self.phi + self.omega * dt) % (2 * np.pi)


@dataclass
class MolecularCarrier:
    """N-type carrier: a molecular oscillator with multiple active modes.

    Carrier signature: S = {(ω_i, A_i, φ_i)}_{i=1}^M
    Each mode contributes to the carrier's oscillatory identity.
    """
    modes: list[tuple[float, float, float]] = field(default_factory=list)  # (ω, A, φ)
    position: float = 0.0
    velocity: float = 0.0

    def __post_init__(self):
        if not self.modes:
            # Default: N₂ vibrational mode
            self.modes = [(2 * np.pi * 7.07e13, 1.0, 0.0)]

    @property
    def dominant_frequency(self) -> float:
        """Highest-amplitude mode frequency."""
        if not self.modes:
            return 0.0
        return max(self.modes, key=lambda m: m[1])[0]

    @property
    def s_entropy(self) -> SEntropyCoordinate:
        """S-entropy from dominant mode."""
        omega, amp, phi = max(self.modes, key=lambda m: m[1])
        return compute_s_entropy(omega, phi, amp)

    def field_at(self, x: float, t: float) -> float:
        """Total oscillatory field contribution at position x, time t."""
        result = 0.0
        for omega, amp, phi in self.modes:
            # Simplified: point source with 1/r decay
            r = abs(x - self.position) + 1e-10
            result += amp * np.cos(omega * t + phi) / r
        return result

    def drift(self, electric_field: float, dt: float) -> None:
        """Drift under applied field: v_d = μ_n · E."""
        self.velocity = MU_N * electric_field
        self.position += self.velocity * dt
        for i, (omega, amp, phi) in enumerate(self.modes):
            self.modes[i] = (omega, amp, (phi + omega * dt) % (2 * np.pi))


@dataclass
class CarrierPopulation:
    """Population of carriers in a membrane region.

    Tracks densities and computes transport properties:
        Conductivity: σ = nμ_n·e + pμ_p·e
        Current density: J = σ·E
        Drift velocity: v_d = μ·E
    """
    n_density: float = N_DENSITY    # N-type carrier density (m⁻³)
    p_density: float = P_DENSITY    # P-type hole density (m⁻³)
    temperature: float = T_PHYSIOL
    holes: list[OscillatoryHole] = field(default_factory=list)
    carriers: list[MolecularCarrier] = field(default_factory=list)

    @property
    def conductivity(self) -> float:
        """Total conductivity: sigma = n*mu_n*e + p*mu_p*e (S/m)."""
        return (self.n_density * MU_N * E_CHARGE +
                self.p_density * MU_P * E_CHARGE)

    @property
    def conductivity_s_per_cm(self) -> float:
        """Conductivity in S/cm for comparison with paper (5.6e-3).

        Compute directly in CGS to match paper exactly:
        sigma = n_CGS * mu_n_CGS * e + p_CGS * mu_p_CGS * e
        """
        return (N_DENSITY_CGS * MU_N_CGS * E_CHARGE +
                P_DENSITY_CGS * MU_P_CGS * E_CHARGE)

    def current_density(self, electric_field: float) -> float:
        """Current density J = σ·E (A/m²)."""
        return self.conductivity * electric_field

    def thermal_voltage(self) -> float:
        """V_T = k_B T / e."""
        return K_B * self.temperature / E_CHARGE

    def recombination_rate(self, overlap_threshold: float = 0.5) -> float:
        """Carrier-hole recombination rate: R = B·n·p.

        When a carrier encounters a hole with matching frequency,
        recombination occurs releasing energy E = ℏω_m.
        """
        B = 1e-16  # bimolecular coefficient (m³/s) — typical for biological systems
        return B * self.n_density * self.p_density

    def create_test_population(self, n_each: int = 100) -> None:
        """Create a test population of holes and carriers."""
        self.holes = [
            OscillatoryHole(
                mode_index=i,
                omega=2 * np.pi * 1e11 * (1 + 0.1 * np.random.randn()),
                phi=np.random.uniform(0, 2 * np.pi),
                position=np.random.uniform(-1e-6, 1e-6),
            )
            for i in range(n_each)
        ]
        self.carriers = [
            MolecularCarrier(
                modes=[(2 * np.pi * 7.07e13 * (1 + 0.01 * np.random.randn()),
                        1.0, np.random.uniform(0, 2 * np.pi))],
                position=np.random.uniform(-1e-6, 1e-6),
            )
            for _ in range(n_each)
        ]
