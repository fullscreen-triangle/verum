"""
F1 Car Circuit Graph — 20-node Philharmonic graph for a Formula 1 car.

Nodes represent major subsystems (ICE, turbo, MGU-K/H, battery, gearbox,
differential, 4 wheels, 4 brakes, 4 suspensions, aero). Edges represent
physical couplings (mechanical, electrical, thermal, aerodynamic).

The adjacency / conductance matrix and graph Laplacian follow the standard
Philharmonic vehicle-circuit-graph conventions.
"""

import numpy as np
from scipy import linalg


# ---------------------------------------------------------------------------
# Node catalogue
# ---------------------------------------------------------------------------

NODE_NAMES = [
    "ICE",          # 0
    "Turbo",        # 1
    "MGU-K",        # 2
    "MGU-H",        # 3
    "Battery",      # 4
    "Gearbox",      # 5
    "Differential",  # 6
    "FL_Wheel",     # 7
    "FR_Wheel",     # 8
    "RL_Wheel",     # 9
    "RR_Wheel",     # 10
    "FL_Brake",     # 11
    "FR_Brake",     # 12
    "RL_Brake",     # 13
    "RR_Brake",     # 14
    "FL_Susp",      # 15
    "FR_Susp",      # 16
    "RL_Susp",      # 17
    "RR_Susp",      # 18
    "Aero",         # 19
]

NODE_INDEX = {name: i for i, name in enumerate(NODE_NAMES)}
N_NODES = len(NODE_NAMES)

# Characteristic frequencies (Hz) at nominal operating point
NODE_FREQUENCIES = {
    "ICE":          250.0,    # ~15 000 RPM
    "Turbo":       2000.0,    # ~120 000 RPM
    "MGU-K":        500.0,
    "MGU-H":       2000.0,
    "Battery":     1000.0,    # DC with ~1 kHz ripple
    "Gearbox":      180.0,    # varies with gear
    "Differential": 100.0,
    "FL_Wheel":      25.0,    # ~300 km/h, r ≈ 0.33 m
    "FR_Wheel":      25.0,
    "RL_Wheel":      25.0,
    "RR_Wheel":      25.0,
    "FL_Brake":       1.0,    # thermal oscillation
    "FR_Brake":       1.0,
    "RL_Brake":       1.0,
    "RR_Brake":       1.0,
    "FL_Susp":        4.0,
    "FR_Susp":        4.0,
    "RL_Susp":        4.0,
    "RR_Susp":        4.0,
    "Aero":          10.0,    # speed-dependent effective
}

# Coupling type tags (for display / transport formula)
COUPLING_TYPES = {
    "mechanical": 1.0,
    "electrical": 1.2,
    "thermal":    0.6,
    "aerodynamic": 0.8,
}

# ---------------------------------------------------------------------------
# Edge list: (node_a, node_b, coupling_type, base_conductance)
# ---------------------------------------------------------------------------

EDGES = [
    # Power unit
    ("ICE",    "Turbo",       "thermal",     0.7),
    ("ICE",    "MGU-K",       "mechanical",  0.9),
    ("Turbo",  "MGU-H",       "mechanical",  0.85),
    ("MGU-K",  "Battery",     "electrical",  1.0),
    ("MGU-H",  "Battery",     "electrical",  1.0),
    ("Battery","MGU-K",       "electrical",  1.0),   # bidirectional boost
    ("Battery","MGU-H",       "electrical",  1.0),

    # Drivetrain
    ("ICE",        "Gearbox",      "mechanical", 0.95),
    ("Gearbox",    "Differential", "mechanical", 0.9),
    ("Differential","RL_Wheel",    "mechanical", 0.85),
    ("Differential","RR_Wheel",    "mechanical", 0.85),

    # Wheel ↔ Brake (friction / thermal)
    ("FL_Wheel", "FL_Brake", "thermal", 0.5),
    ("FR_Wheel", "FR_Brake", "thermal", 0.5),
    ("RL_Wheel", "RL_Brake", "thermal", 0.5),
    ("RR_Wheel", "RR_Brake", "thermal", 0.5),

    # Wheel ↔ Suspension (mechanical)
    ("FL_Wheel", "FL_Susp", "mechanical", 0.7),
    ("FR_Wheel", "FR_Susp", "mechanical", 0.7),
    ("RL_Wheel", "RL_Susp", "mechanical", 0.7),
    ("RR_Wheel", "RR_Susp", "mechanical", 0.7),

    # Aero → all suspensions (load-dependent)
    ("Aero", "FL_Susp", "aerodynamic", 0.6),
    ("Aero", "FR_Susp", "aerodynamic", 0.6),
    ("Aero", "RL_Susp", "aerodynamic", 0.65),
    ("Aero", "RR_Susp", "aerodynamic", 0.65),
]


# ---------------------------------------------------------------------------
# F1CircuitGraph class
# ---------------------------------------------------------------------------

class F1CircuitGraph:
    """
    20-node circuit graph for a Formula 1 car.

    The graph mirrors the physical subsystem topology.  Observable nodes
    (those that map to public FastF1 telemetry channels) form a strict
    subset; hidden nodes must be inferred via trajectory completion.
    """

    def __init__(self):
        self.node_names = list(NODE_NAMES)
        self.node_index = dict(NODE_INDEX)
        self.n = N_NODES
        self.frequencies = np.array(
            [NODE_FREQUENCIES[n] for n in self.node_names], dtype=np.float64
        )
        self.edges = list(EDGES)

        # Build adjacency / conductance matrix
        self._adj = np.zeros((self.n, self.n), dtype=np.float64)
        for a, b, ctype, g_base in self.edges:
            ia, ib = self.node_index[a], self.node_index[b]
            g = g_base * COUPLING_TYPES[ctype]
            self._adj[ia, ib] = max(self._adj[ia, ib], g)
            self._adj[ib, ia] = max(self._adj[ib, ia], g)

        # Observable ↔ hidden split
        self._observable_names = [
            "ICE", "FL_Wheel", "FR_Wheel", "RL_Wheel", "RR_Wheel",
            "FL_Brake", "FR_Brake", "RL_Brake", "RR_Brake",
            "Gearbox", "Aero",
        ]
        self._hidden_names = [
            n for n in self.node_names if n not in self._observable_names
        ]

    # ---- properties -------------------------------------------------------

    @property
    def adjacency_matrix(self) -> np.ndarray:
        """20x20 conductance-weighted adjacency matrix."""
        return self._adj.copy()

    @property
    def laplacian(self) -> np.ndarray:
        """Graph Laplacian  L = D - A."""
        D = np.diag(self._adj.sum(axis=1))
        return D - self._adj

    @property
    def observable_nodes(self):
        return list(self._observable_names)

    @property
    def hidden_nodes(self):
        return list(self._hidden_names)

    @property
    def observable_indices(self):
        return [self.node_index[n] for n in self._observable_names]

    @property
    def hidden_indices(self):
        return [self.node_index[n] for n in self._hidden_names]

    # ---- methods ----------------------------------------------------------

    def node_potentials(self, state_dict: dict) -> np.ndarray:
        """
        Compute categorical depth for each node from raw telemetry values.

        *state_dict* maps node names to raw scalar values.  Categorical
        depth is defined as  phi_i = log(1 + |v_i| / f_i)  where f_i is
        the characteristic frequency — a dimensionless depth measure.
        """
        phi = np.zeros(self.n)
        for name, val in state_dict.items():
            if name in self.node_index:
                i = self.node_index[name]
                phi[i] = np.log1p(np.abs(val) / self.frequencies[i])
        return phi

    def kirchhoff_balance(self, potentials: np.ndarray,
                          external_drive: np.ndarray) -> np.ndarray:
        """
        Solve Kirchhoff balance  L x = I_ext  for node potentials.

        Uses pseudo-inverse because L is singular (constant mode).
        *external_drive* is the 20-vector of external current injection
        (throttle → ICE, brake → brakes, etc.).
        """
        L = self.laplacian
        # Regularise: pin the Aero node as ground reference
        L_reg = L.copy()
        L_reg[-1, :] = 0.0
        L_reg[-1, -1] = 1.0
        rhs = external_drive.copy()
        rhs[-1] = potentials[-1] if np.any(potentials) else 0.0
        return np.linalg.solve(L_reg, rhs)

    def laplacian_eigenvalues(self) -> np.ndarray:
        """Return sorted eigenvalues of the graph Laplacian."""
        return np.sort(np.real(linalg.eigvalsh(self.laplacian)))

    def coupling_degree(self) -> np.ndarray:
        """Weighted degree of each node (row sum of adjacency)."""
        return self._adj.sum(axis=1)
