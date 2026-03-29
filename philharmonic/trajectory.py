"""
Backward Trajectory + Completion for the Philharmonic F1 circuit graph.

Given partial observations (public FastF1 telemetry channels mapped to
observable nodes), reconstruct the full 20-node state via:

    1. Kirchhoff propagation from observed to hidden nodes
    2. Backward trajectory inference (simplified Viterbi on graph)
    3. Thermodynamic projection (ensure Xi >= 0, energy conservation)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .circuit import F1CircuitGraph
from .transport import conductance_matrix, transport_coefficient


class TrajectoryCompletion:
    """
    Trajectory completion engine for the F1 circuit graph.

    Given observations at a subset of nodes (the *observable* nodes that
    map to public FastF1 channels), reconstruct the full state vector
    of all 20 nodes.
    """

    def __init__(self, graph: F1CircuitGraph):
        self.graph = graph
        self.n = graph.n
        self.observations: Dict[str, float] = {}
        self._history: List[np.ndarray] = []
        self._G = conductance_matrix(graph)

    def set_observations(self, obs_dict: Dict[str, float]):
        """
        Set known node states from telemetry.

        *obs_dict* maps node names (e.g. 'ICE', 'FL_Wheel') to scalar
        telemetry values (RPM, angular velocity, etc.).
        """
        self.observations = dict(obs_dict)

    def complete(self, max_iter: int = 50,
                 tol: float = 1e-6) -> Tuple[Dict[str, float], List[float]]:
        """
        Run the 3-step completion algorithm.

        Returns
        -------
        state : dict
            Complete state dict for all 20 nodes.
        convergence : list of float
            Residual at each iteration.
        """
        obs_idx = []
        obs_val = []
        for name, val in self.observations.items():
            if name in self.graph.node_index:
                obs_idx.append(self.graph.node_index[name])
                obs_val.append(val)
        obs_idx = np.array(obs_idx, dtype=int)
        obs_val = np.array(obs_val, dtype=float)

        hid_idx = np.array(
            [i for i in range(self.n) if i not in obs_idx], dtype=int
        )

        # --- Step 1: Kirchhoff propagation ---------------------------------
        # Partition Laplacian into observed / hidden blocks.
        L = self.graph.laplacian.copy()
        # L_hh x_h = -L_ho x_o
        L_hh = L[np.ix_(hid_idx, hid_idx)]
        L_ho = L[np.ix_(hid_idx, obs_idx)]

        # Regularise L_hh (may be singular for isolated hidden components)
        reg = 1e-8 * np.eye(len(hid_idx))
        x_h = np.linalg.solve(L_hh + reg, -L_ho @ obs_val)

        # Assemble full state
        x = np.zeros(self.n)
        x[obs_idx] = obs_val
        x[hid_idx] = x_h

        convergence: List[float] = []

        # --- Step 2: Backward trajectory (iterative refinement) ------------
        # Simplified Viterbi-like: at each step, hidden nodes relax toward
        # the Kirchhoff equilibrium while observed nodes are clamped.
        G = self._G
        for it in range(max_iter):
            x_prev = x.copy()
            # Weighted average update for hidden nodes
            for i in hid_idx:
                neighbours = np.where(G[i] > 0)[0]
                if len(neighbours) == 0:
                    continue
                weights = G[i, neighbours]
                x[i] = np.dot(weights, x_prev[neighbours]) / weights.sum()
            # Re-clamp observed
            x[obs_idx] = obs_val

            residual = np.linalg.norm(x - x_prev) / (np.linalg.norm(x) + 1e-12)
            convergence.append(float(residual))
            if residual < tol:
                break

        # --- Step 3: Thermodynamic projection ------------------------------
        # Ensure transport coefficient Xi >= 0 (physical constraint).
        # Project hidden node values to maintain positive entropy production.
        xi = self._compute_xi(x)
        if xi < 0:
            # Flip sign of hidden nodes that contribute negatively
            for i in hid_idx:
                if x[i] < 0:
                    x[i] = abs(x[i])
            # Minimal projection: scale hidden states uniformly
            scale = 1.0
            for _ in range(10):
                xi = self._compute_xi(x)
                if xi >= 0:
                    break
                scale *= 0.9
                x[hid_idx] *= scale

        self._history.append(x.copy())

        state = {self.graph.node_names[i]: float(x[i]) for i in range(self.n)}
        return state, convergence

    def detect_fault(self, healthy_reference: Dict[str, float],
                     threshold: float = 0.1) -> Dict[str, float]:
        """
        Compare current state with a healthy baseline.

        Returns a dict of node_name -> deviation (fraction of healthy value).
        Deviations above *threshold* indicate potential faults.
        """
        if not self._history:
            return {}

        current = self._history[-1]
        deviations = {}
        for name, ref_val in healthy_reference.items():
            i = self.graph.node_index.get(name)
            if i is None:
                continue
            if abs(ref_val) < 1e-12:
                dev = abs(current[i])
            else:
                dev = abs(current[i] - ref_val) / abs(ref_val)
            deviations[name] = float(dev)
        return deviations

    def backward_trajectory(self, node_name: str,
                            history_length: int = 10) -> np.ndarray:
        """
        Return the MAP backward path for a node over the last
        *history_length* completion steps.
        """
        idx = self.graph.node_index.get(node_name)
        if idx is None:
            return np.array([])

        traj = []
        for state in self._history[-history_length:]:
            traj.append(state[idx])
        return np.array(traj)

    def complete_timeseries(self, obs_series: List[Dict[str, float]],
                            max_iter: int = 30,
                            tol: float = 1e-5) -> List[Dict[str, float]]:
        """
        Run completion on a time series of observations (e.g. one per sample).

        Returns list of complete state dicts.
        """
        results = []
        for obs in obs_series:
            self.set_observations(obs)
            state, _ = self.complete(max_iter=max_iter, tol=tol)
            results.append(state)
        return results

    # ---- internal ---------------------------------------------------------

    def _compute_xi(self, x: np.ndarray) -> float:
        """Transport coefficient for a given state vector."""
        G = self._G
        total = 0.0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if G[i, j] > 0:
                    total += G[i, j] * (x[i] - x[j]) ** 2
        return total / self.n
