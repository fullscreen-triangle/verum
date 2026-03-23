"""
Phase-Locked O₂ Ensemble Model
================================

Atmospheric O₂ molecules form phase-locked ensembles via Van der Waals
and paramagnetic coupling. These ensembles encode complete environmental
state in their phase structure.

Parameters (from Sachikonye 2025):
    Coherence length:   ξ_coh ≈ √(D/Δω) ≈ 14 nm
    Ensemble size:      N ≈ ρ · (4π/3)ξ³ ≈ 10⁴ molecules
    O₂ ground state:    ³Σ_g⁻ (triplet, S=1, μ=2μ_B)
    Spin substates:     m_S = +1, 0, -1

Environmental encoding:
    τ_coh          → Temperature
    n_ensemble     → Pressure
    ξ_coh          → Volume/confinement
    φ̃(ω)          → Chemistry (paramagnetic fingerprinting)
    ∇φ             → Gravity
    φ_drift        → Flow velocity
    γ_decay        → Viscosity
"""

import numpy as np
from dataclasses import dataclass, field

from .s_entropy import SEntropyCoordinate


# ── Physical Constants ────────────────────────────────────────────────────────

K_B = 1.381e-23           # Boltzmann constant
MU_B = 9.274e-24          # Bohr magneton (J/T)
O2_MAGNETIC_MOMENT = 2 * MU_B  # O₂ triplet state μ = 2μ_B
O2_DIFFUSION = 2e-5       # m²/s at STP
O2_FREQ_MISMATCH = 1e6    # Hz typical frequency spread
O2_DENSITY_STP = 1e25     # m⁻³ at STP
O2_VDW_C6 = 1.57e-77      # J·m⁶ (Van der Waals coefficient)
O2_MEAN_SPACING = 3.3e-9  # m at STP


@dataclass
class PhaseLockedEnsemble:
    """A phase-locked ensemble of molecules.

    All molecules in the ensemble satisfy |φ_i(t) - φ_j(t)| < π/4.
    The collective phase φ_ensemble and coherence quality Δφ encode
    the ensemble's information content.
    """
    n_molecules: int = 10000         # ~10⁴ per ensemble
    collective_phase: float = 0.0    # φ_ensemble (rad)
    coherence_quality: float = 0.95  # 1 - Δφ/π (1.0 = perfect)
    coherence_length: float = 14e-9  # ξ_coh (m)
    lifetime: float = 1e-9           # τ_coh (s), temperature-dependent
    temperature: float = 300.0       # K
    pressure: float = 1e5            # Pa

    @property
    def ensemble_density(self) -> float:
        """Number density of ensembles per unit volume (m⁻³)."""
        v_ensemble = (4 / 3) * np.pi * self.coherence_length**3
        return 1.0 / v_ensemble

    @property
    def information_bits(self) -> float:
        """Information content: ~15 bits per ensemble.

        3 parameters (φ, N, Δφ) each with ~5 bits resolution.
        """
        phase_bits = 5.0  # ~32 distinguishable phases
        size_bits = 5.0   # ~32 distinguishable sizes
        quality_bits = 5.0  # ~32 quality levels
        return phase_bits + size_bits + quality_bits

    @property
    def compression_ratio(self) -> float:
        """Reality compression: individual vs ensemble description.

        Individual: N × log₂(M) bits for N molecules, M states each
        Ensemble:   ~15 bits
        Ratio:      N × log₂(M) / 15 ≈ 10⁴ × 20 / 15 ≈ 1.3 × 10⁴
        """
        individual = self.n_molecules * 20  # ~20 bits per molecule
        return individual / self.information_bits


@dataclass
class O2Ensemble(PhaseLockedEnsemble):
    """O₂-specific phase-locked ensemble with environmental encoding.

    O₂ is uniquely suited because:
    1. Triplet ground state (³Σ_g⁻) → paramagnetic → magnetic coupling
    2. S=1, three m_S substates → ternary encoding natural
    3. 21% of atmosphere → ubiquitous
    4. Primary information carrier (not just metabolic substrate)
    """

    def coherence_lifetime(self, temperature: float = None) -> float:
        """τ_coh(T) = τ₀ · exp(E_bind / k_BT).

        Arrhenius dependence on temperature — ensemble IS a thermometer.
        """
        T = temperature or self.temperature
        tau_0 = 1e-13  # molecular collision timescale (s)
        E_bind = 0.6e-3 * 1.602e-19  # ~0.6 meV effective binding (J)
        return tau_0 * np.exp(E_bind / (K_B * T))

    def temperature_from_lifetime(self, tau_coh: float) -> float:
        """Inverse: T = E_bind / (k_B · ln(τ_coh/τ₀)).

        No thermometer needed — the ensemble IS the thermometer.
        """
        tau_0 = 1e-13
        E_bind = 0.6e-3 * 1.602e-19
        if tau_coh <= tau_0:
            return float("inf")
        return E_bind / (K_B * np.log(tau_coh / tau_0))

    def pressure_from_density(self) -> float:
        """P = k_BT · n_ensemble · N_ensemble.

        Pressure measurable by counting ensembles per unit volume.
        """
        return K_B * self.temperature * self.ensemble_density * self.n_molecules

    def chemical_fingerprint(self, species_g_factors: dict[str, float] = None
                             ) -> dict[str, float]:
        """Paramagnetic chemical fingerprinting via Zeeman splitting.

        Different paramagnetic species have distinct Larmor frequencies:
            ω_X = g_X · μ_B · B / ℏ

        O₂:  ³Σ, S=1, μ=2μ_B
        NO:  ²Π, S=1/2, μ=μ_B
        N₂:  ¹Σ, S=0, diamagnetic (no signal)
        """
        if species_g_factors is None:
            species_g_factors = {"O2": 2.0, "NO": 1.0, "N2": 0.0}

        B_earth = 50e-6  # Earth's field ~50 μT
        hbar = 1.055e-34

        fingerprint = {}
        for species, g in species_g_factors.items():
            if g > 0:
                omega = g * MU_B * B_earth / hbar
                fingerprint[species] = omega
            else:
                fingerprint[species] = 0.0
        return fingerprint

    def encode_environment(
        self,
        temperature: float = 300.0,
        pressure: float = 1e5,
        flow_velocity: float = 0.0,
        magnetic_field: float = 50e-6,
    ) -> SEntropyCoordinate:
        """Encode complete environmental state as S-entropy coordinate.

        The mapping:
            S_k ← chemistry + pressure (configurational state)
            S_t ← flow + temperature (dynamical state)
            S_e ← magnetic field + energy distribution
        """
        # Update internal state
        self.temperature = temperature
        self.pressure = pressure
        self.lifetime = self.coherence_lifetime(temperature)

        # S_k: from pressure (ensemble density)
        p_ref = 1e5  # reference pressure
        s_k = np.clip(np.log(1 + pressure / p_ref) / np.log(1 + 10), 0, 1)

        # S_t: from temperature and flow
        t_ref = 300.0
        flow_term = np.tanh(flow_velocity / 10.0)  # normalized flow
        s_t = np.clip(0.5 * (temperature / t_ref) + 0.5 * flow_term, 0, 1)

        # S_e: from magnetic field and lifetime
        b_ref = 50e-6
        s_e = np.clip(np.tanh(magnetic_field / b_ref), 0, 1)

        return SEntropyCoordinate(s_k=s_k, s_t=s_t, s_e=s_e)

    def detect_perturbation(
        self,
        baseline: SEntropyCoordinate,
        current: SEntropyCoordinate,
        threshold: float = 0.05,
    ) -> tuple[bool, float]:
        """Detect environmental perturbation (e.g., another vehicle).

        A perturbation is detected when categorical distance between
        baseline and current S-entropy exceeds threshold.

        Returns (detected, distance).
        """
        d = baseline.categorical_distance(current)
        return d > threshold, d
