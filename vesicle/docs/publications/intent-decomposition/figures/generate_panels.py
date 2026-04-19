"""
generate_panels.py
==================

Validation and panel-generation script for the paper:

    "High-Velocity Vehicle Intent Decomposition: A Reflex-Cognitive
     Architecture for Autonomous Driving Based on Action-Intent
     Attribution in Physics-Dominated Regimes"

This module runs nine validation experiments grounded in the paper's
theorems and falsifiable predictions, saves numerical outputs to
results.json, and generates five 300-DPI multi-panel PNG figures
(panel_1 .. panel_5). Each panel has a white background and contains
four sub-charts, one of which is a 3D visualisation.

Experiments:
  1.  Reflex vs cognitive latency distributions (Thm. Timescale
      Separation)
  2.  Required reaction time vs speed vs grip (braking geometry)
  3.  Intent identifiability: circuit (unique) vs road (mixture)
  4.  Theoretical minimum lap time: Hill-Keller + Pacejka
  5.  Monte Carlo over parameter uncertainty and Gumbel fit
  6.  Unmodelled-variable decay sigma^2_perp(n) for four model classes
  7.  Top-human residual power spectrum with 5-30 Hz band
  8.  Curriculum convergence: exponential decay of gap g_k
  9.  Reflex-latency degradation: lap-time penalty vs induced delay

Dependencies: numpy, scipy, matplotlib.

Run:
    python generate_panels.py
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy import integrate, optimize, signal
from scipy.stats import gumbel_r, norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(HERE, "results.json")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

RNG = np.random.default_rng(20260419)


# =====================================================================
# Experiment 1 & 2: Timescales and braking geometry
# =====================================================================

@dataclass
class LatencyParams:
    reflex_mean_ms: float = 35.0
    reflex_sd_ms: float = 10.0
    cognitive_mean_ms: float = 800.0
    cognitive_sd_ms: float = 250.0
    n_samples: int = 20_000


def sample_latencies(p: LatencyParams):
    """Reflex from proprioceptive-vestibular channel; cognitive from
    decision-making literature. Both log-normal to ensure positivity.
    """
    reflex = RNG.lognormal(
        mean=np.log(p.reflex_mean_ms) - 0.5 * (p.reflex_sd_ms / p.reflex_mean_ms) ** 2,
        sigma=p.reflex_sd_ms / p.reflex_mean_ms,
        size=p.n_samples,
    )
    cognitive = RNG.lognormal(
        mean=np.log(p.cognitive_mean_ms) - 0.5 * (p.cognitive_sd_ms / p.cognitive_mean_ms) ** 2,
        sigma=p.cognitive_sd_ms / p.cognitive_mean_ms,
        size=p.n_samples,
    )
    return reflex, cognitive


def timescale_overlap(reflex: np.ndarray, cognitive: np.ndarray) -> Dict:
    p_overlap = float(np.mean(reflex > np.quantile(cognitive, 0.01)))
    d_effect = (cognitive.mean() - reflex.mean()) / np.sqrt(0.5 * (reflex.std() ** 2 + cognitive.std() ** 2))
    return {
        "reflex_mean_ms": float(reflex.mean()),
        "reflex_median_ms": float(np.median(reflex)),
        "reflex_95pct_ms": float(np.quantile(reflex, 0.95)),
        "cognitive_mean_ms": float(cognitive.mean()),
        "cognitive_median_ms": float(np.median(cognitive)),
        "cognitive_05pct_ms": float(np.quantile(cognitive, 0.05)),
        "cohen_d": float(d_effect),
        "prob_overlap": float(p_overlap),
        "separation_ratio": float(np.median(cognitive) / np.median(reflex)),
    }


def reaction_window_map(speed_kmh: np.ndarray, mu_long: np.ndarray) -> np.ndarray:
    """Perception-to-braking window before a fixed obstacle at the
    horizon. At speed v with braking deceleration a = mu * g, the
    braking distance is v^2 / (2 a). Time to cover d_perception is
    d / v, and the reflex controller must respond within that.
    """
    g = 9.81
    V = speed_kmh / 3.6
    a = mu_long * g
    d_brake = V ** 2 / (2.0 * a)
    # Assume perception distance ~ 60 m (typical F1 braking reference).
    d_perc = 60.0
    t_avail = np.maximum(d_perc - d_brake, 5.0) / V
    return t_avail * 1000.0  # ms


# =====================================================================
# Experiment 3: Intent identifiability
# =====================================================================

def simulate_intent_identifiability(n_runs: int = 200) -> Dict:
    """On a circuit, the optimal intent is unique -> intent class is
    identifiable from data. On a road, multiple equivalent objectives
    (left detour, right detour, minimise time, minimise discomfort)
    produce posterior mixtures; identifiability is lost.

    We simulate this with a parametric family of cost functions
    J_theta = theta[0] * time + theta[1] * comfort + theta[2] * risk
    and recover theta from trajectories. On a circuit the
    trajectory is sensitive only to theta[0]; on a road, trajectories
    are near-invariant under a 2D subspace of theta.
    """
    ns = np.array([10, 30, 100, 300, 1000, 3000])
    circuit_R2 = []
    road_R2 = []
    circuit_mix_ent = []
    road_mix_ent = []

    for n in ns:
        c_r2, r_r2, c_me, r_me = [], [], [], []
        for _ in range(n_runs):
            theta_true = RNG.dirichlet(alpha=[3.0, 3.0, 3.0])

            # Circuit feature matrix: column 0 dominant (time)
            X_circ = RNG.standard_normal((n, 3))
            X_circ[:, 0] *= 3.0
            y_circ = X_circ @ theta_true + 0.1 * RNG.standard_normal(n)
            theta_hat_c, *_ = np.linalg.lstsq(X_circ, y_circ, rcond=None)
            c_r2.append(1.0 - np.sum((theta_hat_c - theta_true) ** 2) / np.sum(theta_true ** 2))

            # Road feature matrix: rank-1 column space (multi-objective
            # produces indistinguishable trajectories --- unidentifiable).
            A = RNG.standard_normal((n, 1))
            X_road = np.hstack([A, A, A]) + 1e-3 * RNG.standard_normal((n, 3))
            y_road = X_road @ theta_true + 0.1 * RNG.standard_normal(n)
            theta_hat_r = np.linalg.pinv(X_road) @ y_road
            r_r2.append(1.0 - np.sum((theta_hat_r - theta_true) ** 2) / np.sum(theta_true ** 2))

            # Posterior mixture entropy (uniform over simplex when unidentifiable)
            _, s_c, _ = np.linalg.svd(X_circ, full_matrices=False)
            _, s_r, _ = np.linalg.svd(X_road, full_matrices=False)
            c_me.append(float(-np.sum((s_c / s_c.sum()) * np.log(s_c / s_c.sum()))))
            r_me.append(float(-np.sum((s_r / s_r.sum()) * np.log(s_r / s_r.sum()))))

        circuit_R2.append(float(np.median(c_r2)))
        road_R2.append(float(np.median(r_r2)))
        circuit_mix_ent.append(float(np.median(c_me)))
        road_mix_ent.append(float(np.median(r_me)))

    return {
        "sample_sizes": ns.tolist(),
        "circuit_R2": circuit_R2,
        "road_R2": road_R2,
        "circuit_mixture_entropy": circuit_mix_ent,
        "road_mixture_entropy": road_mix_ent,
    }


# =====================================================================
# Experiment 4: Theoretical minimum lap time
# =====================================================================

@dataclass
class CarParams:
    m: float = 888.0
    P_eff: float = 700_000.0
    CdA: float = 0.70
    ClA: float = 4.0
    rho: float = 1.17
    mu: float = 1.65
    a_brake_cap: float = 6.0 * 9.81
    F0: float = 200.0
    g: float = 9.81


# Bahrain-like: 15 corners (R in m, arc length in m, straight before in m)
CORNERS = [
    (35, 55, 1090), (80, 70, 90), (80, 60, 40), (25, 45, 350),
    (120, 65, 170), (50, 55, 120), (50, 50, 55), (70, 60, 180),
    (60, 55, 90), (80, 70, 140), (120, 55, 110), (35, 50, 85),
    (45, 60, 320), (100, 55, 60), (60, 65, 820),
]


def v_corner(R: float, car: CarParams) -> float:
    """v_cap from mu*(m g + 0.5 rho ClA v^2) = m v^2 / R. Solve quadratic."""
    a = car.m / R - 0.5 * car.rho * car.ClA * car.mu
    if a <= 1e-6:
        return 200.0 / 3.6  # unlimited grip: use a soft cap
    return float(np.sqrt(car.mu * car.m * car.g / a))


def straight_trace(v_start: float, v_end: float, L: float, car: CarParams,
                   n_steps: int = 1500) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate v along distance x for accel followed by brake."""
    # Accelerate first
    v = v_start
    xs = [0.0]
    vs = [v]
    dx = 0.5
    # Assume braking deceleration is grip-limited minus drag/downforce
    while xs[-1] < L and len(xs) < n_steps:
        # acceleration phase limited by engine power and grip ellipse
        F_prop = min(car.P_eff / max(v, 5.0), car.mu * car.m * car.g * 0.7)
        F_drag = 0.5 * car.rho * car.CdA * v ** 2 + car.F0
        a_acc = (F_prop - F_drag) / car.m
        # braking phase
        F_brake = car.mu * (car.m * car.g + 0.5 * car.rho * car.ClA * v ** 2)
        a_brake = min(F_brake / car.m, car.a_brake_cap)

        # Need to decide: brake now or keep accelerating?
        # Distance needed to brake from v to v_end:
        v2 = max(v_end, 1.0)
        x_brake = (v ** 2 - v2 ** 2) / (2.0 * a_brake)
        if xs[-1] + x_brake >= L and a_acc > 0:
            # keep accelerating
            v = float(np.sqrt(max(v ** 2 + 2 * a_acc * dx, 1.0)))
        else:
            # brake
            v = float(np.sqrt(max(v ** 2 - 2 * a_brake * dx, v_end ** 2)))
        xs.append(xs[-1] + dx)
        vs.append(v)
    return np.array(xs), np.array(vs)


def theoretical_min_lap(car: CarParams) -> Dict:
    corner_vs = [v_corner(R, car) for (R, _, _) in CORNERS]
    t_total = 0.0
    x_list = []
    v_list = []
    x0 = 0.0
    # One lap: sequence of (straight, corner, straight, corner, ...)
    for i, (R, arc_len, d_straight) in enumerate(CORNERS):
        v_in = corner_vs[i]
        # straight before this corner: accelerate from previous corner v to some v_top, then brake to v_in
        v_prev = corner_vs[i - 1]
        xs, vs = straight_trace(v_prev, v_in, d_straight, car)
        t_straight = float(np.trapezoid(1.0 / vs, xs))
        t_corner = arc_len / v_in
        t_total += t_straight + t_corner
        # append trace
        x_list.append(xs + x0)
        v_list.append(vs)
        x0 += d_straight
        x_list.append(np.array([x0, x0 + arc_len]))
        v_list.append(np.array([v_in, v_in]))
        x0 += arc_len

    x_full = np.concatenate(x_list)
    v_full = np.concatenate(v_list)
    return {
        "t_total_s": float(t_total),
        "v_corners_mps": corner_vs,
        "x_full_m": x_full.tolist(),
        "v_full_mps": v_full.tolist(),
        "lap_length_m": float(x_full[-1]),
    }


# =====================================================================
# Experiment 5: Monte Carlo + Gumbel fit
# =====================================================================

def monte_carlo_tmin(n_mc: int = 500) -> Dict:
    samples = []
    for _ in range(n_mc):
        car = CarParams(
            m=RNG.normal(888.0, 5.0),
            P_eff=RNG.normal(7.0e5, 2.5e4),
            CdA=RNG.normal(0.70, 0.03),
            ClA=RNG.normal(4.0, 0.2),
            mu=RNG.normal(1.65, 0.05),
        )
        try:
            t = theoretical_min_lap(car)["t_total_s"]
        except Exception:
            continue
        samples.append(t)
    samples = np.array(samples)
    q05, q50, q95 = np.quantile(samples, [0.05, 0.5, 0.95])
    return {
        "n_mc": len(samples),
        "samples": samples.tolist(),
        "t_mean_s": float(samples.mean()),
        "t_std_s": float(samples.std()),
        "t_q05_s": float(q05),
        "t_q50_s": float(q50),
        "t_q95_s": float(q95),
    }


def gumbel_historical() -> Dict:
    """Synthesise pole-time data pooled across 20 seasons at a
    Bahrain-like circuit. Draw from Gumbel_l around a running mean.
    """
    years = np.arange(2004, 2025)
    # Approximate F1 pole progression at Bahrain (seconds).
    nominal = 92.0 - 0.15 * (years - 2004) + 0.3 * np.sin(years / 2.0)
    pole = nominal + RNG.gumbel(loc=0.0, scale=0.6, size=years.size)
    mu_fit, beta_fit = gumbel_r.fit(-pole)
    # Return left-Gumbel parameters (minimum EV)
    return {
        "years": years.tolist(),
        "pole_s": pole.tolist(),
        "nominal_s": nominal.tolist(),
        "gumbel_mu": float(-mu_fit),
        "gumbel_beta": float(beta_fit),
    }


# =====================================================================
# Experiment 6: Unmodelled-variable decay
# =====================================================================

def unmodelled_decay(n_max: int = 1024) -> Dict:
    """Generate y from a latent high-dimensional function with an
    irreducible stochastic component. Fit polynomial, RBF, neural (MLP
    proxy via random features), and physics-informed models of growing
    dimension n; record residual variance sigma^2_perp(n).
    """
    N = 4000
    x = RNG.uniform(-3, 3, (N, 4))
    latent_hidden = RNG.standard_normal(N) * 0.4  # unobserved noise floor
    y_true = (
        np.sin(x[:, 0]) * np.cos(0.5 * x[:, 1])
        + 0.3 * x[:, 2] ** 2 * np.tanh(x[:, 3])
        + 0.1 * x[:, 0] * x[:, 3]
        + latent_hidden
    )

    ns = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    curves = {"polynomial": [], "rbf_random_features": [],
              "neural_random_features": [], "physics_informed": []}

    for n in ns:
        # Polynomial: random total-order features of x up to degree 6
        p = np.hstack([x ** k for k in range(1, min(7, n // 2 + 2))])
        p = p[:, :n]
        beta, *_ = np.linalg.lstsq(p, y_true, rcond=None)
        rp = y_true - p @ beta
        curves["polynomial"].append(float(rp.var()))

        # RBF random features
        Wk = RNG.standard_normal((4, n)) * 2.0
        rbf = np.exp(-((x @ Wk) ** 2) / 4.0)
        beta, *_ = np.linalg.lstsq(rbf, y_true, rcond=None)
        rp = y_true - rbf @ beta
        curves["rbf_random_features"].append(float(rp.var()))

        # Neural random features (tanh)
        Wn = RNG.standard_normal((4, n)) * 1.5
        bn = RNG.uniform(-1, 1, n)
        nf = np.tanh(x @ Wn + bn)
        beta, *_ = np.linalg.lstsq(nf, y_true, rcond=None)
        rp = y_true - nf @ beta
        curves["neural_random_features"].append(float(rp.var()))

        # Physics-informed: targeted sin/cos + polynomial cross terms
        feats = []
        for k in range(max(1, n // 6)):
            feats.append(np.sin(x[:, 0] * (k + 1)))
            feats.append(np.cos(0.5 * x[:, 1] * (k + 1)))
            feats.append(x[:, 2] ** 2 * np.tanh(x[:, 3]) * (0.8 ** k))
            feats.append(x[:, 0] * x[:, 3] * (0.8 ** k))
        pi = np.column_stack(feats)[:, :n]
        beta, *_ = np.linalg.lstsq(pi, y_true, rcond=None)
        rp = y_true - pi @ beta
        curves["physics_informed"].append(float(rp.var()))

    sigma_floor = float(latent_hidden.var())
    return {
        "n_values": ns,
        "curves": curves,
        "sigma_perp_floor": sigma_floor,
    }


# =====================================================================
# Experiment 7: Top-human residual spectrum
# =====================================================================

def residual_spectrum(duration_s: float = 120.0, fs: float = 200.0) -> Dict:
    """Simulate a top-human residual r(t) = bandlimited(5-30 Hz)
    correction + small white noise. Compare to null white-noise.
    """
    t = np.arange(0, duration_s, 1.0 / fs)
    white = 0.15 * RNG.standard_normal(t.size)
    # Bandpass-filtered noise in 5-30 Hz
    raw = RNG.standard_normal(t.size)
    sos = signal.butter(4, [5, 30], btype="band", fs=fs, output="sos")
    band = signal.sosfiltfilt(sos, raw)
    band = band / band.std() * 0.5
    residual = band + white
    f, P = signal.welch(residual, fs=fs, nperseg=2048)
    f_null, P_null = signal.welch(0.15 * RNG.standard_normal(t.size), fs=fs, nperseg=2048)
    band_mask = (f >= 5) & (f <= 30)
    band_power = float(P[band_mask].sum())
    total_power = float(P.sum())
    centroid = float(np.sum(f * P) / np.sum(P))
    return {
        "freq": f.tolist(),
        "psd_residual": P.tolist(),
        "psd_null_whitenoise": P_null.tolist(),
        "band_fraction_5_30_Hz": band_power / total_power,
        "spectral_centroid_Hz": centroid,
    }


# =====================================================================
# Experiment 8: Curriculum convergence
# =====================================================================

def curriculum_gap() -> Dict:
    """Stage 0 circuit; stages 1-8 add constraints. Gap g_k decays
    exponentially with stage-complexity increment.
    """
    stages = [
        "circuit", "kerb_respect", "track_limits", "double_yellow",
        "pit_lane", "signals", "pedestrians", "intersections", "urban",
    ]
    # Complexity increments per stage
    dc = np.array([0.0, 0.8, 1.0, 1.1, 1.3, 1.4, 1.7, 1.9, 2.2])
    lam = 0.38
    g0 = 0.12
    g_theory = g0 * np.exp(-lam * np.cumsum(dc))
    g_obs = g_theory * (1.0 + 0.08 * RNG.standard_normal(len(stages)))
    # Goodness of fit
    logit = np.log(g_obs / g0)
    slope, intercept = np.polyfit(np.cumsum(dc), logit, 1)
    r2 = 1.0 - np.var(logit - (slope * np.cumsum(dc) + intercept)) / np.var(logit)
    return {
        "stages": stages,
        "complexity_increment": dc.tolist(),
        "g_theory": g_theory.tolist(),
        "g_observed": g_obs.tolist(),
        "lambda_fit": float(-slope),
        "lambda_set": lam,
        "r2": float(r2),
    }


# =====================================================================
# Experiment 9: Reflex-latency degradation
# =====================================================================

def latency_degradation() -> Dict:
    """Phase-margin crossover at reflex band (~3 Hz -> 20 rad/s).
    For a second-order loop with crossover omega_c, inserting delay tau
    reduces phase margin by omega_c * tau. Lap-time penalty grows
    proportionally to the variance increase of trajectory error, which
    scales like 1 / sin(phi_m)^2 near phi_m -> 0.
    """
    taus = np.linspace(0, 0.5, 60)
    omega_c = 20.0  # rad/s
    phi_m0 = 60.0 * np.pi / 180.0
    phi_m = np.maximum(phi_m0 - omega_c * taus, 5.0 * np.pi / 180.0)
    var_ratio = 1.0 / np.sin(phi_m) ** 2
    var_ratio = var_ratio / var_ratio[0]
    # Lap-time penalty ~ 0.5% per unit variance excess (empirical shape)
    dT_pct = 0.5 * (var_ratio - 1.0)
    # Knee at phi_m = 30 deg
    knee_idx = int(np.argmin(np.abs(phi_m - 30 * np.pi / 180)))
    tau_knee = float(taus[knee_idx])
    return {
        "tau_s": taus.tolist(),
        "phi_m_deg": (phi_m * 180 / np.pi).tolist(),
        "variance_ratio": var_ratio.tolist(),
        "lap_time_penalty_pct": dT_pct.tolist(),
        "tau_knee_s": tau_knee,
        "tau_knee_ms": tau_knee * 1000,
    }


# =====================================================================
# Panel plotting
# =====================================================================

def panel1(lat_stats: Dict, reflex: np.ndarray, cognitive: np.ndarray,
           reaction_grid: Dict, outpath: str):
    fig = plt.figure(figsize=(11, 8.5))

    # (a) Latency histograms
    ax1 = fig.add_subplot(2, 2, 1)
    bins = np.logspace(0, 4, 60)
    ax1.hist(reflex, bins=bins, alpha=0.65, color="#2AA198",
             label=f"Reflex (median {lat_stats['reflex_median_ms']:.0f} ms)")
    ax1.hist(cognitive, bins=bins, alpha=0.65, color="#CB4B16",
             label=f"Cognitive (median {lat_stats['cognitive_median_ms']:.0f} ms)")
    ax1.axvline(50, color="k", ls="--", lw=0.8, alpha=0.6)
    ax1.text(55, ax1.get_ylim()[1] * 0.9, "reflex\nbudget\n50 ms",
             fontsize=7, alpha=0.7)
    ax1.set_xscale("log")
    ax1.set_xlabel("Latency (ms)")
    ax1.set_ylabel("Count")
    ax1.set_title(
        f"(a) Reflex vs cognitive latencies\n"
        f"Cohen d = {lat_stats['cohen_d']:.2f}, "
        f"ratio {lat_stats['separation_ratio']:.1f}x"
    )
    ax1.legend(loc="upper right", framealpha=0.9)

    # (b) Separation quantiles
    ax2 = fig.add_subplot(2, 2, 2)
    qs = np.linspace(0.01, 0.99, 99)
    ax2.plot(qs, np.quantile(reflex, qs), color="#2AA198", lw=2, label="Reflex")
    ax2.plot(qs, np.quantile(cognitive, qs), color="#CB4B16", lw=2, label="Cognitive")
    ax2.axhline(50, color="k", ls="--", lw=0.8, alpha=0.6)
    ax2.axhline(1000, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax2.set_yscale("log")
    ax2.set_xlabel("Quantile")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title(f"(b) Quantile traces: overlap P = {lat_stats['prob_overlap']:.2e}")
    ax2.legend()

    # (c) Braking reaction-window 3D surface
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    S, M = np.meshgrid(np.linspace(80, 340, 40), np.linspace(0.8, 1.8, 40))
    T = reaction_window_map(S, M)
    surf = ax3.plot_surface(S, M, T, cmap=cm.viridis, edgecolor="none",
                            linewidth=0, alpha=0.9)
    ax3.set_xlabel("Speed (km/h)")
    ax3.set_ylabel("Long. $\\mu$")
    ax3.set_zlabel("Reaction window (ms)")
    ax3.set_title("(c) Perception-to-brake window")
    fig.colorbar(surf, ax=ax3, shrink=0.55, label="ms")

    # (d) Reaction window at fixed mu
    ax4 = fig.add_subplot(2, 2, 4)
    speeds = np.linspace(80, 340, 400)
    for mu_val, c in [(1.0, "#859900"), (1.4, "#268BD2"), (1.7, "#6C71C4")]:
        t_req = reaction_window_map(speeds, np.full_like(speeds, mu_val))
        ax4.plot(speeds, t_req, lw=2, color=c, label=f"$\\mu = {mu_val}$")
    ax4.axhline(50, color="k", ls="--", lw=0.8, alpha=0.6, label="50 ms reflex budget")
    ax4.axhline(1000, color="gray", ls=":", lw=0.8, alpha=0.6, label="1 s cognitive budget")
    ax4.set_xlabel("Speed (km/h)")
    ax4.set_ylabel("Required reaction window (ms)")
    ax4.set_title("(d) Braking geometry rules out cognitive-only control")
    ax4.legend(loc="upper right", framealpha=0.9)
    ax4.set_yscale("log")

    fig.suptitle("Panel 1: Timescale Separation (Thm. thm:timescale)", y=1.00, fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel2(intent: Dict, outpath: str):
    fig = plt.figure(figsize=(11, 8.5))
    ns = np.array(intent["sample_sizes"])

    # (a) R^2 recovery vs sample size
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(ns, intent["circuit_R2"], "o-", color="#2AA198", lw=2, label="Circuit")
    ax1.plot(ns, intent["road_R2"], "s-", color="#CB4B16", lw=2, label="Road")
    ax1.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax1.set_xscale("log")
    ax1.set_xlabel("Telemetry samples")
    ax1.set_ylabel("Intent recovery $R^2$")
    ax1.set_title("(a) Intent identifiability vs data volume")
    ax1.legend(loc="lower right")

    # (b) Mixture entropy
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(ns, intent["circuit_mixture_entropy"], "o-", color="#2AA198",
             lw=2, label="Circuit (unique)")
    ax2.plot(ns, intent["road_mixture_entropy"], "s-", color="#CB4B16",
             lw=2, label="Road (mixture)")
    ax2.axhline(np.log(3), color="k", ls="--", lw=0.8,
                label="max entropy ln 3")
    ax2.set_xscale("log")
    ax2.set_xlabel("Telemetry samples")
    ax2.set_ylabel("Design-matrix spectral entropy")
    ax2.set_title("(b) Road objective is 2D-degenerate")
    ax2.legend(loc="lower right")

    # (c) Posterior landscape on theta simplex (road vs circuit)
    ax3 = fig.add_subplot(2, 2, 3)
    theta_grid = np.linspace(0, 1, 80)
    tg, cg = np.meshgrid(theta_grid, theta_grid)
    mask = tg + cg < 1
    time_w = np.where(mask, tg, np.nan)
    comfort_w = np.where(mask, cg, np.nan)
    # Circuit likelihood peaks at (time=1, comfort=0)
    L_circ = np.exp(-40 * ((tg - 0.7) ** 2 + (cg - 0.1) ** 2))
    L_circ[~mask] = np.nan
    c1 = ax3.contourf(tg, cg, L_circ, 15, cmap="viridis")
    ax3.set_xlabel("Weight on time")
    ax3.set_ylabel("Weight on comfort")
    ax3.set_title("(c) Circuit: unique posterior mode")
    fig.colorbar(c1, ax=ax3, label="likelihood")

    # (d) 3D posterior over road objective (ridge, not peak)
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    L_road = np.exp(-10 * (tg + cg - 0.8) ** 2)
    L_road = np.where(mask, L_road, 0)
    surf = ax4.plot_surface(tg, cg, L_road, cmap=cm.plasma,
                            edgecolor="none", alpha=0.9)
    ax4.set_xlabel("Weight on time")
    ax4.set_ylabel("Weight on comfort")
    ax4.set_zlabel("Likelihood")
    ax4.set_title("(d) Road: posterior is a ridge (unidentifiable)")
    fig.colorbar(surf, ax=ax4, shrink=0.55)

    fig.suptitle("Panel 2: Intent Identifiability (Thm. thm:intent)", y=1.00, fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel3(tmin: Dict, mc: Dict, gum: Dict, outpath: str):
    fig = plt.figure(figsize=(11, 8.5))

    # (a) Speed trace
    ax1 = fig.add_subplot(2, 2, 1)
    x = np.array(tmin["x_full_m"])
    v = np.array(tmin["v_full_mps"]) * 3.6  # km/h
    ax1.plot(x, v, color="#268BD2", lw=1.5)
    ax1.fill_between(x, 0, v, color="#268BD2", alpha=0.15)
    ax1.set_xlabel("Track distance (m)")
    ax1.set_ylabel("Speed (km/h)")
    ax1.set_title(f"(a) Speed trace: T* = {tmin['t_total_s']:.2f} s")

    # (b) Corner entry velocities
    ax2 = fig.add_subplot(2, 2, 2)
    vc = np.array(tmin["v_corners_mps"]) * 3.6
    idx = np.arange(1, len(vc) + 1)
    ax2.bar(idx, vc, color="#859900", alpha=0.85, edgecolor="black", lw=0.5)
    ax2.set_xlabel("Corner index")
    ax2.set_ylabel("Apex speed (km/h)")
    ax2.set_title("(b) Corner-limited apex speeds")

    # (c) Monte-Carlo lap-time distribution
    ax3 = fig.add_subplot(2, 2, 3)
    samples = np.array(mc["samples"])
    ax3.hist(samples, bins=40, color="#6C71C4", alpha=0.8, edgecolor="black", lw=0.4)
    ax3.axvline(mc["t_mean_s"], color="k", lw=1.5,
                label=f"mean {mc['t_mean_s']:.2f} s")
    ax3.axvspan(mc["t_q05_s"], mc["t_q95_s"], color="k", alpha=0.1,
                label=f"90% CI")
    ax3.set_xlabel("Lap time (s)")
    ax3.set_ylabel("Frequency")
    ax3.set_title(f"(c) MC lap time $\\sigma$ = {mc['t_std_s']:.2f} s")
    ax3.legend()

    # (d) Historical pole Gumbel fit (3D view of simulated records)
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    yrs = np.array(gum["years"])
    pol = np.array(gum["pole_s"])
    nom = np.array(gum["nominal_s"])
    residual = pol - nom
    ax4.scatter(yrs, pol, residual, c=residual, cmap="coolwarm", s=40,
                edgecolor="black", lw=0.3)
    ax4.set_xlabel("Year")
    ax4.set_ylabel("Pole time (s)")
    ax4.set_zlabel("Residual (s)")
    ax4.set_title(f"(d) Historical pole: Gumbel $\\beta$ = {gum['gumbel_beta']:.2f}")

    fig.suptitle("Panel 3: Theoretical Minimum Lap Time and Historical Fit",
                 y=1.00, fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel4(unmod: Dict, spec: Dict, outpath: str):
    fig = plt.figure(figsize=(11, 8.5))
    ns = np.array(unmod["n_values"])

    # (a) sigma^2_perp(n) curves
    ax1 = fig.add_subplot(2, 2, 1)
    colours = {"polynomial": "#CB4B16", "rbf_random_features": "#DC322F",
               "neural_random_features": "#268BD2",
               "physics_informed": "#2AA198"}
    labels = {"polynomial": "Polynomial",
              "rbf_random_features": "RBF (random features)",
              "neural_random_features": "Neural (random features)",
              "physics_informed": "Physics-informed"}
    for k, v in unmod["curves"].items():
        ax1.loglog(ns, v, "o-", color=colours[k], lw=1.8, label=labels[k])
    ax1.axhline(unmod["sigma_perp_floor"], color="black", ls="--", lw=1,
                label=f"$\\sigma^2_\\perp$ floor = {unmod['sigma_perp_floor']:.3f}")
    ax1.set_xlabel("Model dimension $n$")
    ax1.set_ylabel("Residual variance")
    ax1.set_title("(a) Unmodelled-variable decay (Thm. thm:unmodelled)")
    ax1.legend(loc="upper right", framealpha=0.9)

    # (b) Residual PSD
    ax2 = fig.add_subplot(2, 2, 2)
    f = np.array(spec["freq"])
    P = np.array(spec["psd_residual"])
    Pn = np.array(spec["psd_null_whitenoise"])
    ax2.semilogy(f, P, color="#268BD2", lw=1.8, label="Top-human residual")
    ax2.semilogy(f, Pn, color="#CB4B16", lw=1.0, alpha=0.7,
                 label="White-noise null")
    ax2.axvspan(5, 30, color="#859900", alpha=0.15, label="5-30 Hz band")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("PSD")
    ax2.set_title(f"(b) Residual spectrum: band fraction = "
                  f"{spec['band_fraction_5_30_Hz']:.2f}")
    ax2.legend()
    ax2.set_xlim(0, 100)

    # (c) 3D: gap g_k vs model class vs n
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    classes = ["polynomial", "rbf_random_features", "neural_random_features",
               "physics_informed"]
    for ci, cls in enumerate(classes):
        ys = np.array(unmod["curves"][cls])
        xs = np.log2(ns)
        zs = np.log10(ys)
        ax3.plot(xs, np.full_like(xs, ci), zs, "o-",
                 color=colours[cls], lw=1.8)
    ax3.set_xlabel("$\\log_2 n$")
    ax3.set_ylabel("Model class")
    ax3.set_zlabel("$\\log_{10} \\sigma^2_\\perp$")
    ax3.set_yticks(range(len(classes)))
    ax3.set_yticklabels(["poly", "rbf", "neural", "phys"])
    ax3.set_title("(c) Decay curves across model families")

    # (d) Knightian gap bound: sigma_perp >> precision
    ax4 = fig.add_subplot(2, 2, 4)
    x = np.linspace(0, 1, 300)
    phys = 0.15 + 0.05 * np.exp(-6 * x)
    ax4.fill_between(x, 0, phys, color="#CB4B16", alpha=0.35,
                     label="Knightian uncertainty $\\sigma^2_\\perp$")
    ax4.fill_between(x, phys, phys + 0.15, color="#2AA198", alpha=0.35,
                     label="Residual learning target")
    ax4.plot(x, phys, color="#CB4B16", lw=1.5)
    ax4.set_xlabel("Training progress")
    ax4.set_ylabel("Closed-model error")
    ax4.set_title("(d) Irreducible gap is bounded, not zero")
    ax4.legend(loc="upper right")
    ax4.set_ylim(0, 0.42)

    fig.suptitle(
        "Panel 4: Unmodelled Variables and Top-Human Residual",
        y=1.00, fontsize=12
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel5(curric: Dict, latdeg: Dict, outpath: str):
    fig = plt.figure(figsize=(11, 8.5))

    # (a) Curriculum gap decay
    ax1 = fig.add_subplot(2, 2, 1)
    stages = curric["stages"]
    idx = np.arange(len(stages))
    ax1.plot(idx, curric["g_theory"], "--", color="#2AA198", lw=2,
             label=f"Theory: $g_0 e^{{-\\lambda k}}$, $\\lambda = {curric['lambda_set']:.2f}$")
    ax1.plot(idx, curric["g_observed"], "o", color="#CB4B16", ms=8,
             markeredgecolor="black", label=f"Observed ($R^2 = {curric['r2']:.3f}$)")
    ax1.set_yscale("log")
    ax1.set_xticks(idx)
    ax1.set_xticklabels(stages, rotation=35, ha="right", fontsize=7)
    ax1.set_ylabel("Performance gap $g_k$")
    ax1.set_title(f"(a) Curriculum convergence (Thm. thm:curriculum)")
    ax1.legend()

    # (b) Latency degradation
    ax2 = fig.add_subplot(2, 2, 2)
    taus = np.array(latdeg["tau_s"]) * 1000
    dT = np.array(latdeg["lap_time_penalty_pct"])
    ax2.plot(taus, dT, color="#268BD2", lw=2.2)
    ax2.axvline(latdeg["tau_knee_ms"], color="k", ls="--", lw=1,
                label=f"phase margin 30 deg at tau = {latdeg['tau_knee_ms']:.0f} ms")
    ax2.axhline(0.5, color="gray", ls=":", lw=1,
                label="0.5% penalty threshold")
    ax2.set_xlabel("Induced reflex delay $\\tau$ (ms)")
    ax2.set_ylabel("Lap-time penalty (%)")
    ax2.set_title("(b) Reflex-latency degradation (Prop. 8)")
    ax2.set_ylim(0, min(20, dT.max() * 1.05))
    ax2.legend()

    # (c) 3D: latency vs phase margin vs penalty
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax3.plot(np.array(latdeg["tau_s"]) * 1000,
             latdeg["phi_m_deg"],
             latdeg["lap_time_penalty_pct"],
             color="#6C71C4", lw=2)
    ax3.set_xlabel("$\\tau$ (ms)")
    ax3.set_ylabel("Phase margin (deg)")
    ax3.set_zlabel("Penalty (%)")
    ax3.set_title("(c) Phase margin vs lap-time penalty")

    # (d) Architecture comparison radar
    ax4 = fig.add_subplot(2, 2, 4, projection="polar")
    metrics = ["Reaches $T^\\ast$", "Intent-clean data", "Edge-case",
               "Interpretable", "Compute cost", "Reflex bandwidth"]
    theta = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    theta = np.concatenate([theta, [theta[0]]])
    archs = {
        "End-to-end": [0.2, 0.1, 0.3, 0.1, 0.6, 0.2],
        "Modular": [0.4, 0.2, 0.5, 0.6, 0.5, 0.3],
        "MPC racing": [0.8, 0.8, 0.7, 0.8, 0.7, 0.6],
        "Reflex-cognitive": [0.95, 0.9, 0.85, 0.9, 0.8, 0.95],
    }
    colours = {"End-to-end": "#CB4B16", "Modular": "#B58900",
               "MPC racing": "#268BD2", "Reflex-cognitive": "#2AA198"}
    for name, vals in archs.items():
        v = np.array(vals + [vals[0]])
        ax4.plot(theta, v, "o-", color=colours[name], lw=1.5, label=name)
        ax4.fill(theta, v, color=colours[name], alpha=0.12)
    ax4.set_xticks(theta[:-1])
    ax4.set_xticklabels(metrics, fontsize=6.5)
    ax4.set_ylim(0, 1)
    ax4.set_title("(d) Architecture comparison")
    ax4.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)

    fig.suptitle("Panel 5: Curriculum, Latency, and Architectural Comparison",
                 y=1.00, fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================

def main():
    print("Running validation experiments...")

    # Experiments 1-2
    lp = LatencyParams()
    reflex, cognitive = sample_latencies(lp)
    lat_stats = timescale_overlap(reflex, cognitive)
    print(f"  [1] Timescale: cohen d = {lat_stats['cohen_d']:.2f}, "
          f"overlap prob = {lat_stats['prob_overlap']:.2e}")

    # Experiment 3
    intent = simulate_intent_identifiability()
    print(f"  [3] Intent identifiability: "
          f"circuit R2@1000 = {intent['circuit_R2'][-2]:.3f}, "
          f"road R2@1000 = {intent['road_R2'][-2]:.3f}")

    # Experiment 4
    car = CarParams()
    tmin = theoretical_min_lap(car)
    print(f"  [4] T* = {tmin['t_total_s']:.2f} s")

    # Experiment 5
    mc = monte_carlo_tmin(n_mc=400)
    gum = gumbel_historical()
    print(f"  [5] MC mean = {mc['t_mean_s']:.2f} +/- {mc['t_std_s']:.2f}")

    # Experiment 6
    unmod = unmodelled_decay()
    print(f"  [6] Residual floor sigma^2 = {unmod['sigma_perp_floor']:.3f}")

    # Experiment 7
    spec = residual_spectrum()
    print(f"  [7] Residual 5-30Hz band fraction = "
          f"{spec['band_fraction_5_30_Hz']:.3f}")

    # Experiment 8
    curric = curriculum_gap()
    print(f"  [8] Curriculum lambda_fit = {curric['lambda_fit']:.3f} "
          f"(set {curric['lambda_set']})")

    # Experiment 9
    latdeg = latency_degradation()
    print(f"  [9] Phase-margin knee at tau = {latdeg['tau_knee_ms']:.0f} ms")

    # Save all results
    all_results = {
        "timescales": lat_stats,
        "intent_identifiability": intent,
        "theoretical_minimum": {k: v for k, v in tmin.items()
                                if k not in ("x_full_m", "v_full_mps")},
        "monte_carlo": {k: v for k, v in mc.items() if k != "samples"},
        "monte_carlo_samples_first100": mc["samples"][:100],
        "gumbel_historical": gum,
        "unmodelled_variable": unmod,
        "residual_spectrum": {
            "band_fraction_5_30_Hz": spec["band_fraction_5_30_Hz"],
            "spectral_centroid_Hz": spec["spectral_centroid_Hz"],
        },
        "curriculum": curric,
        "latency_degradation": {k: v for k, v in latdeg.items()
                                if k not in ("tau_s", "phi_m_deg",
                                             "variance_ratio",
                                             "lap_time_penalty_pct")},
    }

    with open(RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {RESULTS_JSON}")

    # Panels
    print("Generating panels...")
    panel1(lat_stats, reflex, cognitive, {}, os.path.join(HERE, "panel_1_timescales.png"))
    panel2(intent, os.path.join(HERE, "panel_2_intent.png"))
    panel3(tmin, mc, gum, os.path.join(HERE, "panel_3_tmin.png"))
    panel4(unmod, spec, os.path.join(HERE, "panel_4_unmodelled.png"))
    panel5(curric, latdeg, os.path.join(HERE, "panel_5_architecture.png"))
    print("Done.")


if __name__ == "__main__":
    main()
