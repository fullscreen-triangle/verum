"""
Universal Transport Formula for the Philharmonic circuit graph.

    Xi = N^{-1}  sum_{edges} tau_{p,ij} * g_{ij}

where tau_p is the partition lag between two oscillators and g_ij is the
coupling strength (conductance).
"""

import numpy as np

from .circuit import F1CircuitGraph, COUPLING_TYPES


# ---------------------------------------------------------------------------
# Partition lag
# ---------------------------------------------------------------------------

def partition_lag(freq_i: float, freq_j: float, T: float = 300.0) -> float:
    """
    Compute the partition lag tau_p between two oscillators.

    tau_p = (1 / |f_i - f_j + epsilon|) * exp(-|f_i - f_j| / (k_B T_eff))

    where T_eff is an effective thermal scale (in frequency units) and
    epsilon prevents division by zero.  This captures the idea that
    strongly detuned oscillators communicate with larger delay.
    """
    k_B_T = T  # effective thermal frequency scale
    delta = abs(freq_i - freq_j)
    eps = 1e-3
    return (1.0 / (delta + eps)) * np.exp(-delta / k_B_T)


# ---------------------------------------------------------------------------
# Coupling strength
# ---------------------------------------------------------------------------

def coupling_strength(g_base: float, coupling_type: str) -> float:
    """
    Coupling strength  g_ij = g_base * type_factor.
    """
    return g_base * COUPLING_TYPES.get(coupling_type, 1.0)


# ---------------------------------------------------------------------------
# Transport coefficient
# ---------------------------------------------------------------------------

def transport_coefficient(graph: F1CircuitGraph,
                          T: float = 300.0) -> float:
    """
    Compute the universal transport coefficient for the full graph:

        Xi = (1/N) * sum_{(i,j) in edges} tau_p(f_i, f_j, T) * g_ij
    """
    total = 0.0
    seen = set()
    for a, b, ctype, g_base in graph.edges:
        key = tuple(sorted([a, b]))
        if key in seen:
            continue
        seen.add(key)
        ia = graph.node_index[a]
        ib = graph.node_index[b]
        fi = graph.frequencies[ia]
        fj = graph.frequencies[ib]
        tau = partition_lag(fi, fj, T)
        g = coupling_strength(g_base, ctype)
        total += tau * g
    return total / graph.n


def conductance_matrix(graph: F1CircuitGraph,
                       T: float = 300.0) -> np.ndarray:
    """
    Compute the full N x N effective conductance matrix G_ij,
    weighted by partition lag:

        G_ij = tau_p(f_i, f_j) * g_ij
    """
    G = np.zeros((graph.n, graph.n), dtype=np.float64)
    for a, b, ctype, g_base in graph.edges:
        ia = graph.node_index[a]
        ib = graph.node_index[b]
        fi = graph.frequencies[ia]
        fj = graph.frequencies[ib]
        tau = partition_lag(fi, fj, T)
        g = coupling_strength(g_base, ctype)
        val = tau * g
        G[ia, ib] = max(G[ia, ib], val)
        G[ib, ia] = max(G[ib, ia], val)
    return G
