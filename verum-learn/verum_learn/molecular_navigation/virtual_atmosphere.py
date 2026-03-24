"""
Virtual Atmosphere: Atmospheric Ensemble from Hardware Timing
==============================================================

Creates a virtual atmospheric ensemble where each molecule is a real
timing measurement from hardware oscillators, positioned in S-entropy
space. The ensemble has thermodynamic properties (T, P, ρ) derived
from ensemble statistics — not assumed, measured.

The atmosphere is a distributed computer with ~10²² molecules per
10 cm³, each processing at ~10¹³ ops/s. We create a scaled virtual
version from hardware timing that preserves the categorical structure.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .hardware_oscillator import HardwareOscillator, TimingMeasurement


@dataclass
class VirtualMolecule:
    """A virtual atmospheric molecule — a point in S-entropy space.

    Each molecule has:
    - Position (x, y) in physical space
    - S-entropy (s_k, s_t, s_e) in categorical space
    - Species (O₂, N₂, CO₂, etc.) determining paramagnetic properties
    - Temperature contribution (from partition lag)
    """
    x: float                     # physical position x (metres)
    y: float                     # physical position y (metres)
    s_k: float                   # knowledge entropy [0,1]
    s_t: float                   # temporal entropy [0,1]
    s_e: float                   # evolution entropy [0,1]
    species: str = "O2"          # molecular species
    temperature: float = 300.0   # local temperature (K)
    velocity_x: float = 0.0     # wind x component (m/s)
    velocity_y: float = 0.0     # wind y component (m/s)

    @property
    def s_array(self) -> np.ndarray:
        return np.array([self.s_k, self.s_t, self.s_e])

    def categorical_distance(self, other: "VirtualMolecule") -> float:
        return float(np.linalg.norm(self.s_array - other.s_array))


class VirtualAtmosphere:
    """Virtual atmospheric ensemble from hardware oscillator timing.

    Each molecule is derived from real hardware timing jitter.
    The ensemble IS the atmosphere — a bounded collection of
    categorical states with thermodynamic properties.
    """

    def __init__(self, width: float = 200.0, height: float = 50.0,
                 base_temperature: float = 300.0, base_pressure: float = 1e5):
        self.width = width       # metres
        self.height = height     # metres
        self.base_temperature = base_temperature  # K
        self.base_pressure = base_pressure        # Pa
        self.oscillator = HardwareOscillator()
        self.molecules: list[VirtualMolecule] = []
        self.time = 0.0

    def populate(self, n_molecules: int = 1000) -> None:
        """Create n virtual molecules from hardware timing.

        Each molecule gets:
        - Random physical position in the domain
        - S-entropy from hardware oscillator (REAL measurement)
        - Temperature from base + local perturbation
        """
        self.molecules = []
        for _ in range(n_molecules):
            m = self.oscillator.sample_molecule(n_jitters=10)
            mol = VirtualMolecule(
                x=np.random.uniform(0, self.width),
                y=np.random.uniform(0, self.height),
                s_k=m.s_k,
                s_t=m.s_t,
                s_e=m.s_e,
                species=np.random.choice(["O2", "N2", "CO2", "H2O"],
                                         p=[0.21, 0.78, 0.004, 0.006]),
                temperature=self.base_temperature + np.random.normal(0, 2),
            )
            self.molecules.append(mol)

    def measure_at(self, x: float, y: float, radius: float = 5.0) -> dict:
        """Measure atmospheric S-entropy at a position.

        Averages over all molecules within radius of (x,y).
        Returns S-entropy and thermodynamic properties.
        """
        nearby = [m for m in self.molecules
                  if (m.x - x)**2 + (m.y - y)**2 < radius**2]

        if not nearby:
            return {"s_k": 0.5, "s_t": 0.5, "s_e": 0.5,
                    "temperature": self.base_temperature,
                    "n_molecules": 0, "detected": False}

        s_k = np.mean([m.s_k for m in nearby])
        s_t = np.mean([m.s_t for m in nearby])
        s_e = np.mean([m.s_e for m in nearby])
        temp = np.mean([m.temperature for m in nearby])

        return {
            "s_k": s_k, "s_t": s_t, "s_e": s_e,
            "temperature": temp,
            "n_molecules": len(nearby),
            "detected": True,
        }

    def inject_perturbation(self, x: float, y: float, radius: float = 10.0,
                            delta_temp: float = 0.0,
                            delta_sk: float = 0.0,
                            delta_st: float = 0.0,
                            delta_se: float = 0.0,
                            species_override: Optional[str] = None) -> int:
        """Inject a perturbation (vehicle, human, obstacle) at position.

        Modifies all molecules within radius. Returns count of affected molecules.
        """
        count = 0
        for m in self.molecules:
            dist = np.sqrt((m.x - x)**2 + (m.y - y)**2)
            if dist < radius:
                falloff = 1.0 - dist / radius  # linear falloff
                m.temperature += delta_temp * falloff
                m.s_k = np.clip(m.s_k + delta_sk * falloff, 0, 1)
                m.s_t = np.clip(m.s_t + delta_st * falloff, 0, 1)
                m.s_e = np.clip(m.s_e + delta_se * falloff, 0, 1)
                if species_override:
                    m.species = species_override
                count += 1
        return count

    def inject_vehicle(self, x: float, y: float, engine_temp: float = 90.0,
                       speed: float = 30.0, heading: float = 0.0) -> None:
        """Inject a vehicle's atmospheric signature.

        A vehicle perturbs the atmosphere via:
        - Engine heat: ΔT = engine_temp above ambient
        - Exhaust: CO₂, NOx species change
        - Wake turbulence: flow velocity perturbation
        - Pressure wave: from vehicle motion
        """
        # Thermal plume (radius proportional to speed)
        thermal_radius = 10.0 + speed * 0.3
        self.inject_perturbation(
            x, y, radius=thermal_radius,
            delta_temp=engine_temp * 0.1,  # fraction reaching atmosphere
            delta_se=0.15,  # energy redistribution
        )
        # Exhaust plume (behind vehicle)
        exhaust_x = x - 3.0 * np.cos(heading)
        exhaust_y = y - 3.0 * np.sin(heading)
        self.inject_perturbation(
            exhaust_x, exhaust_y, radius=5.0,
            delta_sk=0.2,   # composition change
            delta_se=0.1,   # energy change
            species_override="CO2",
        )
        # Wake turbulence
        self.inject_perturbation(
            x, y, radius=thermal_radius * 1.5,
            delta_st=0.1 * speed / 30.0,  # flow perturbation proportional to speed
        )

    def inject_human(self, x: float, y: float) -> None:
        """Inject a pedestrian's atmospheric signature.

        Human body: +37°C, CO₂ from breathing (~40,000 ppm exhaled),
        humidity from respiration.
        """
        self.inject_perturbation(
            x, y, radius=5.0,
            delta_temp=5.0,   # body heat at 5m
            delta_sk=0.05,    # CO₂ from breath
            delta_se=0.03,    # metabolic energy
        )

    def diffuse(self, dt: float, D: float = 1.0) -> None:
        """Advance atmospheric diffusion by dt seconds.

        Each molecule undergoes Brownian motion: Δx = √(2D·dt) × N(0,1)
        S-entropy coordinates also diffuse (information spreads).
        """
        sigma = np.sqrt(2 * D * dt)
        for m in self.molecules:
            m.x += np.random.normal(0, sigma) + m.velocity_x * dt
            m.y += np.random.normal(0, sigma) + m.velocity_y * dt
            # Keep in bounds
            m.x = np.clip(m.x, 0, self.width)
            m.y = np.clip(m.y, 0, self.height)
            # S-entropy diffuses toward equilibrium
            decay = np.exp(-dt * 0.1)
            m.s_k = 0.5 + (m.s_k - 0.5) * decay
            m.s_t = 0.5 + (m.s_t - 0.5) * decay
            m.s_e = 0.5 + (m.s_e - 0.5) * decay
            # Temperature relaxes toward base
            m.temperature = self.base_temperature + (m.temperature - self.base_temperature) * decay
        self.time += dt

    def detect_perturbation(self, x: float, y: float, baseline: dict,
                            threshold: float = 0.02) -> tuple[bool, float]:
        """Detect if the atmosphere at (x,y) differs from baseline.

        Returns (detected, categorical_distance).
        """
        current = self.measure_at(x, y)
        if not current["detected"]:
            return False, 0.0

        d_cat = np.sqrt(
            (current["s_k"] - baseline["s_k"])**2 +
            (current["s_t"] - baseline["s_t"])**2 +
            (current["s_e"] - baseline["s_e"])**2
        )
        return d_cat > threshold, float(d_cat)
