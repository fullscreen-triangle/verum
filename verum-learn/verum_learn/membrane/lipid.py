"""
Lipid Oscillator Model
=======================

Each lipid is modeled as a harmonic oscillator with frequency ω, phase φ,
and amplitude A. By the oscillator-processor duality (ω ≡ R_compute),
each lipid IS a processor computing at rate R = ω/(2π).

Physical parameters (zero free parameters, derived from bounded phase space):
    Bilayer thickness:       d = 4.0 nm
    Area per lipid:          A_L = 0.64 nm²
    Bending modulus:         κ = 19 k_BT
    Chain isomerization:     ~10¹¹ /s per lipid
    Biological clock:        f_0 = 758 Hz (ATP-driven)
    Coherence time:          τ_c = 10 ms

The lipid's equation of motion:
    d²x/dt² + γ(dx/dt) + ω²x = F_env(t) + F_coupling(t)

where γ is damping, ω is natural frequency, F_env is environmental forcing,
and F_coupling is inter-lipid coupling (Van der Waals + paramagnetic).
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .s_entropy import SEntropyCoordinate, compute_s_entropy


# ── Physical Constants ────────────────────────────────────────────────────────

K_B = 1.381e-23          # Boltzmann constant (J/K)
T_PHYSIOL = 310.0         # Physiological temperature (K)
K_BT = K_B * T_PHYSIOL    # Thermal energy at 310 K

BILAYER_THICKNESS = 4.0e-9       # m
AREA_PER_LIPID = 0.64e-18       # m² (0.64 nm²)
BENDING_MODULUS_KBT = 19.0       # in units of k_BT
BENDING_MODULUS = BENDING_MODULUS_KBT * K_BT  # J

CHAIN_ISOMERIZATION_RATE = 1e11  # Hz (per lipid)
BIOLOGICAL_CLOCK_FREQ = 758.0    # Hz (ATP-driven)
COHERENCE_TIME = 10e-3           # s (10 ms)

ATP_ENERGY = 50e-21              # J (50 zJ per hydrolysis)


class LipidPhase(Enum):
    """Lipid phase state — partition extinction at gel-fluid transition."""
    GEL = "gel"         # Ordered chains, single conformational state
    FLUID = "fluid"     # Disordered chains, 3 conformations per bond
    RAFT = "raft"       # Local partition extinction domain


@dataclass
class Lipid:
    """A single lipid modeled as an oscillator-processor.

    State: (ω, φ, A) maps to S-entropy (S_k, S_t, S_e) via the S-mapping.
    Processing rate: R = ω/(2π) operations per second.
    """
    omega: float = CHAIN_ISOMERIZATION_RATE * 2 * np.pi  # angular frequency
    phi: float = 0.0                                       # phase
    amplitude: float = 1.0                                 # normalized amplitude
    damping: float = 1e9                                   # damping coefficient γ
    phase_state: LipidPhase = LipidPhase.FLUID
    temperature: float = T_PHYSIOL

    @property
    def frequency(self) -> float:
        """Oscillation frequency in Hz."""
        return self.omega / (2.0 * np.pi)

    @property
    def processing_rate(self) -> float:
        """Computational rate by oscillator-processor duality: R = ω/(2π)."""
        return self.frequency

    @property
    def period(self) -> float:
        """Oscillation period in seconds."""
        return 1.0 / self.frequency if self.frequency > 0 else float("inf")

    @property
    def s_entropy(self) -> SEntropyCoordinate:
        """Current S-entropy coordinate."""
        return compute_s_entropy(self.omega, self.phi, self.amplitude)

    def displacement(self, t: float) -> float:
        """Oscillatory displacement at time t: x(t) = A·cos(ωt + φ)."""
        return self.amplitude * np.cos(self.omega * t + self.phi)

    def velocity(self, t: float) -> float:
        """Oscillatory velocity at time t: v(t) = -Aω·sin(ωt + φ)."""
        return -self.amplitude * self.omega * np.sin(self.omega * t + self.phi)

    def energy(self) -> float:
        """Total oscillator energy: E = ½mω²A² (m=1 in normalized units)."""
        return 0.5 * self.omega**2 * self.amplitude**2

    def evolve(self, dt: float, forcing: float = 0.0) -> None:
        """Advance the oscillator by dt seconds under optional forcing.

        Damped driven oscillator: φ → φ + ωdt, A → A·exp(-γdt/2) + forcing response.
        """
        self.phi = (self.phi + self.omega * dt) % (2.0 * np.pi)
        self.amplitude *= np.exp(-self.damping * dt / (2.0 * self.omega))
        if forcing != 0.0:
            # Forcing adds energy proportional to resonance overlap
            self.amplitude += forcing * dt / self.omega

    def is_phase_locked_with(self, other: "Lipid", threshold: float = np.pi / 4) -> bool:
        """Check phase-locking: |φ_self - φ_other| < threshold."""
        delta = abs(self.phi - other.phi)
        delta = min(delta, 2 * np.pi - delta)  # handle wraparound
        return delta < threshold


@dataclass
class LipidArray:
    """An array of lipids forming a membrane patch.

    For a patch of area A_patch:
        N_lipids = 2 · A_patch / A_L  (factor 2 for bilayer)
        Processing power = N_lipids × R_per_lipid

    A 1 mm² patch: N ≈ 3.1 × 10¹² lipids → R ≈ 3.1 × 10²³ ops/s
    A 1 m² patch:  N ≈ 3.1 × 10¹⁸ lipids → R ≈ 3.1 × 10²⁹ ops/s
    """
    area: float = 1e-6              # patch area in m² (default 1 mm²)
    temperature: float = T_PHYSIOL
    lipids: list[Lipid] = field(default_factory=list)

    def __post_init__(self):
        if not self.lipids:
            n = self.lipid_count
            # Initialize with random phases, uniform frequency
            self.lipids = [
                Lipid(
                    phi=np.random.uniform(0, 2 * np.pi),
                    temperature=self.temperature,
                )
                for _ in range(min(n, 10000))  # cap simulation size
            ]

    @property
    def lipid_count(self) -> int:
        """Number of lipids in bilayer: N = 2·A/A_L."""
        return int(2.0 * self.area / AREA_PER_LIPID)

    @property
    def total_processing_rate(self) -> float:
        """Total computational throughput in ops/s."""
        return self.lipid_count * CHAIN_ISOMERIZATION_RATE

    @property
    def mean_phase(self) -> float:
        """Mean phase across all lipids (Kuramoto order parameter angle)."""
        phases = np.array([l.phi for l in self.lipids])
        z = np.mean(np.exp(1j * phases))
        return float(np.angle(z)) % (2 * np.pi)

    @property
    def phase_coherence(self) -> float:
        """Kuramoto order parameter r ∈ [0,1]. r=1 means perfect phase-lock."""
        phases = np.array([l.phi for l in self.lipids])
        return float(np.abs(np.mean(np.exp(1j * phases))))

    @property
    def mean_s_entropy(self) -> SEntropyCoordinate:
        """Average S-entropy across the array."""
        coords = np.array([[l.s_entropy.s_k, l.s_entropy.s_t, l.s_entropy.s_e]
                           for l in self.lipids])
        mean = coords.mean(axis=0)
        return SEntropyCoordinate(s_k=mean[0], s_t=mean[1], s_e=mean[2])

    def evolve(self, dt: float, environmental_forcing: Optional[np.ndarray] = None) -> None:
        """Evolve all lipids by dt, with optional per-lipid forcing."""
        for i, lipid in enumerate(self.lipids):
            f = environmental_forcing[i] if environmental_forcing is not None else 0.0
            lipid.evolve(dt, forcing=f)

    def apply_coupling(self, coupling_strength: float = 0.01) -> None:
        """Apply nearest-neighbor Kuramoto coupling between lipids.

        dφ_i/dt += (K/N) Σ_j sin(φ_j - φ_i)

        This models Van der Waals + paramagnetic inter-lipid coupling.
        """
        phases = np.array([l.phi for l in self.lipids])
        n = len(phases)
        mean_sin = np.mean(np.sin(phases[:, None] - phases[None, :]), axis=1)
        for i, lipid in enumerate(self.lipids):
            lipid.phi += coupling_strength * mean_sin[i]
            lipid.phi %= (2 * np.pi)

    def transduce(self, signal: float) -> SEntropyCoordinate:
        """Transduce an environmental signal through the lipid array.

        The signal modifies lipid oscillation parameters, and the resulting
        S-entropy change IS the transduced output.

        This is the core claim: lipid arrays act as signal transducers.
        """
        # Signal modulates lipid frequencies via forcing
        forcing = np.full(len(self.lipids), signal * 1e-3)
        self.evolve(dt=1.0 / BIOLOGICAL_CLOCK_FREQ, environmental_forcing=forcing)
        self.apply_coupling()
        return self.mean_s_entropy
