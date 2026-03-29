"""
Generate 5 publication-quality panels (20 charts total) for the
Philharmonic F1 telemetry validation.

Each panel is a 4-subplot figure saved at 300 DPI with white background.
"""

import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Ensure project root on path
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
TEAL   = "#2AA198"
GOLD   = "#D4AF37"
CORAL  = "#ff6b6b"
CYAN   = "#58E6D9"
DARK   = "#2c3e50"
LIGHT  = "#ecf0f1"

OUT_DIR = os.path.join(os.path.dirname(__file__))


def _savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved: {path}")


# ===================================================================
# Panel 1: F1 Circuit Graph
# ===================================================================

def panel_1_circuit(results):
    from philharmonic.circuit import F1CircuitGraph

    graph = F1CircuitGraph()
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="white")

    # (A) Adjacency matrix heatmap
    ax = axes[0]
    A = graph.adjacency_matrix
    im = ax.imshow(A, cmap="YlOrBr", aspect="equal", interpolation="nearest")
    ax.set_xticks(range(graph.n))
    ax.set_xticklabels(graph.node_names, rotation=90, fontsize=5)
    ax.set_yticks(range(graph.n))
    ax.set_yticklabels(graph.node_names, fontsize=5)
    ax.set_title("(A) Adjacency Matrix", fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Conductance")

    # (B) Node frequency spectrum (bar chart, log scale)
    ax = axes[1]
    freqs = graph.frequencies
    colors = [TEAL if n in graph.observable_nodes else GOLD for n in graph.node_names]
    bars = ax.barh(range(graph.n), freqs, color=colors, edgecolor="none")
    ax.set_yticks(range(graph.n))
    ax.set_yticklabels(graph.node_names, fontsize=6)
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)", fontsize=8)
    ax.set_title("(B) Characteristic Frequencies", fontsize=10, fontweight="bold")
    # Legend
    from matplotlib.patches import Patch
    ax.legend([Patch(facecolor=TEAL), Patch(facecolor=GOLD)],
              ["Observable", "Hidden"], fontsize=7, loc="lower right")

    # (C) 3D: frequency vs categorical_depth vs coupling_degree
    ax = fig.add_subplot(1, 4, 3, projection="3d")
    axes[2].set_visible(False)
    # Use nominal potentials for visualisation
    nominal_state = {n: f for n, f in zip(graph.node_names, graph.frequencies)}
    potentials = graph.node_potentials(nominal_state)
    degrees = graph.coupling_degree()
    obs_mask = np.array([n in graph.observable_nodes for n in graph.node_names])
    ax.scatter(np.log10(freqs[obs_mask] + 1), potentials[obs_mask], degrees[obs_mask],
               c=TEAL, s=50, label="Observable", depthshade=True)
    ax.scatter(np.log10(freqs[~obs_mask] + 1), potentials[~obs_mask], degrees[~obs_mask],
               c=GOLD, s=50, label="Hidden", depthshade=True)
    for i, name in enumerate(graph.node_names):
        ax.text(np.log10(freqs[i] + 1), potentials[i], degrees[i], name,
                fontsize=4, ha="left")
    ax.set_xlabel("log10(freq)", fontsize=7)
    ax.set_ylabel("Categorical depth", fontsize=7)
    ax.set_zlabel("Coupling degree", fontsize=7)
    ax.set_title("(C) Node State Space", fontsize=10, fontweight="bold")

    # (D) Laplacian eigenvalue spectrum
    ax = axes[3]
    eigs = graph.laplacian_eigenvalues()
    ax.bar(range(len(eigs)), eigs, color=CYAN, edgecolor="none")
    ax.set_xlabel("Eigenvalue index", fontsize=8)
    ax.set_ylabel("Eigenvalue", fontsize=8)
    ax.set_title("(D) Laplacian Spectrum", fontsize=10, fontweight="bold")
    ax.axhline(y=0, color="grey", linewidth=0.5, linestyle="--")

    fig.suptitle("Panel 1: F1 Circuit Graph (20 Nodes)", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    _savefig(fig, "panel_1_f1_circuit.png")


# ===================================================================
# Panel 2: State Reconstruction
# ===================================================================

def panel_2_reconstruction(results):
    r = results.get("state_reconstruction", {})
    obs_ts = r.get("observed_timeseries", {})
    rec_ts = r.get("reconstructed_timeseries", {})
    errors = r.get("reconstruction_errors", {})

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="white")

    # (A) Observed vs reconstructed ICE (RPM proxy) over samples
    ax = axes[0]
    ice_obs = np.array(obs_ts.get("ICE", [0]))
    ice_rec = np.array(rec_ts.get("ICE", [0]))
    t = np.arange(len(ice_obs))
    ax.plot(t, ice_obs, color=TEAL, linewidth=1.0, label="Observed", alpha=0.8)
    ax.plot(t, ice_rec, color=GOLD, linewidth=1.0, label="Reconstructed",
            alpha=0.8, linestyle="--")
    ax.set_xlabel("Sample", fontsize=8)
    ax.set_ylabel("ICE frequency (Hz)", fontsize=8)
    ax.set_title("(A) ICE: Observed vs Reconstructed", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=7)

    # (B) Reconstructed battery SOC over stint
    ax = axes[1]
    battery = np.array(rec_ts.get("Battery", [0]))
    ax.plot(np.arange(len(battery)), battery, color=CORAL, linewidth=1.2)
    ax.fill_between(np.arange(len(battery)), battery,
                    alpha=0.15, color=CORAL)
    ax.set_xlabel("Sample", fontsize=8)
    ax.set_ylabel("Battery state", fontsize=8)
    ax.set_title("(B) Reconstructed Battery SOC", fontsize=10,
                 fontweight="bold")

    # (C) 3D: all node states — observed (teal) vs reconstructed (gold)
    ax = fig.add_subplot(1, 4, 3, projection="3d")
    axes[2].set_visible(False)
    from philharmonic.circuit import F1CircuitGraph
    graph = F1CircuitGraph()
    n_points = min(50, len(ice_obs))
    step = max(1, len(ice_obs) // n_points)
    for i in range(0, len(ice_obs), step):
        obs_vals = [obs_ts.get(n, [0] * (i + 1))[i] if len(obs_ts.get(n, [])) > i else 0
                    for n in graph.observable_nodes[:3]]
        rec_vals = [rec_ts.get(n, [0] * (i + 1))[i] if len(rec_ts.get(n, [])) > i else 0
                    for n in graph.observable_nodes[:3]]
        if len(obs_vals) >= 3:
            ax.scatter(*obs_vals[:3], c=TEAL, s=8, alpha=0.4)
        if len(rec_vals) >= 3:
            ax.scatter(*rec_vals[:3], c=GOLD, s=8, alpha=0.4)
    ax.set_xlabel(graph.observable_nodes[0], fontsize=6)
    ax.set_ylabel(graph.observable_nodes[1], fontsize=6)
    ax.set_zlabel(graph.observable_nodes[2], fontsize=6)
    ax.set_title("(C) State Space", fontsize=10, fontweight="bold")

    # (D) Per-node reconstruction error bars
    ax = axes[3]
    nodes = list(errors.keys())
    nrmse_vals = [errors[n].get("nrmse", 0.0) for n in nodes]
    y_pos = range(len(nodes))
    colors = [TEAL if v < 0.01 else GOLD if v < 0.05 else CORAL for v in nrmse_vals]
    ax.barh(y_pos, nrmse_vals, color=colors, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(nodes, fontsize=6)
    ax.set_xlabel("NRMSE", fontsize=8)
    ax.set_title("(D) Reconstruction Error", fontsize=10, fontweight="bold")

    fig.suptitle("Panel 2: State Reconstruction from Partial Telemetry",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _savefig(fig, "panel_2_reconstruction.png")


# ===================================================================
# Panel 3: Fault Prediction
# ===================================================================

def panel_3_fault(results):
    r = results.get("fault_prediction", {})
    healthy = r.get("healthy_baseline", {})
    faulty = r.get("faulty_trajectory", {})
    deviations = r.get("deviations", [])
    detection_lap = r.get("detection_lap", 0)
    fault_lap = r.get("fault_lap", 15)
    n_laps = r.get("n_laps", 20)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="white")
    laps = np.arange(1, n_laps + 1)

    # (A) Categorical depth vs lap: healthy baseline + faulty diverging
    ax = axes[0]
    for node, color, label in [("ICE", TEAL, "ICE (healthy)"),
                                ("Turbo", GOLD, "Turbo (healthy)")]:
        h = np.array(healthy.get(node, [0] * n_laps))[:n_laps]
        f_vals = np.array(faulty.get(node, [0] * n_laps))[:n_laps]
        ax.plot(laps[:len(h)], h, color=color, linewidth=1.2,
                label=label, alpha=0.7)
        ax.plot(laps[:len(f_vals)], f_vals, color=color, linewidth=1.2,
                linestyle="--", label=f"{node} (faulty)", alpha=0.9)
    ax.axvline(x=fault_lap, color=CORAL, linestyle=":", linewidth=1.5,
               label=f"Fault @ lap {fault_lap}")
    if detection_lap:
        ax.axvline(x=detection_lap, color=CYAN, linestyle="-.",
                   linewidth=1.5, label=f"Detected @ lap {detection_lap}")
    ax.set_xlabel("Lap", fontsize=8)
    ax.set_ylabel("Node state", fontsize=8)
    ax.set_title("(A) Healthy vs Faulty Trajectory", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=6)

    # (B) Per-node deviation at final lap
    ax = axes[1]
    final_dev = r.get("per_node_deviation", {})
    if final_dev:
        nodes = sorted(final_dev.keys(), key=lambda k: final_dev[k], reverse=True)[:12]
        vals = [final_dev[n] for n in nodes]
        colors = [CORAL if n in ("ICE", "Turbo") else TEAL for n in nodes]
        ax.barh(range(len(nodes)), vals, color=colors, edgecolor="none")
        ax.set_yticks(range(len(nodes)))
        ax.set_yticklabels(nodes, fontsize=6)
    ax.set_xlabel("Deviation from healthy", fontsize=8)
    ax.set_title("(B) Fault Localization", fontsize=10, fontweight="bold")

    # (C) 3D: backward trajectory through state space
    ax = fig.add_subplot(1, 4, 3, projection="3d")
    axes[2].set_visible(False)
    ice_h = np.array(healthy.get("ICE", [0] * n_laps))[:n_laps]
    turbo_h = np.array(healthy.get("Turbo", [0] * n_laps))[:n_laps]
    mguk_h = np.array(healthy.get("MGU-K", [0] * n_laps))[:n_laps]
    ice_f = np.array(faulty.get("ICE", [0] * n_laps))[:n_laps]
    turbo_f = np.array(faulty.get("Turbo", [0] * n_laps))[:n_laps]
    mguk_f = np.array(faulty.get("MGU-K", [0] * n_laps))[:n_laps]
    ax.plot(ice_h, turbo_h, mguk_h, color=TEAL, linewidth=1.5,
            label="Healthy", alpha=0.8)
    ax.plot(ice_f, turbo_f, mguk_f, color=CORAL, linewidth=1.5,
            label="Faulty", alpha=0.8, linestyle="--")
    ax.scatter([ice_f[-1]], [turbo_f[-1]], [mguk_f[-1]], c=CORAL, s=60,
               marker="x", zorder=5)
    ax.set_xlabel("ICE", fontsize=7)
    ax.set_ylabel("Turbo", fontsize=7)
    ax.set_zlabel("MGU-K", fontsize=7)
    ax.set_title("(C) State-Space Trajectory", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6)

    # (D) Detection lead time
    ax = axes[3]
    lead = r.get("detection_lead_time", 0)
    threshold = r.get("threshold", 0.15)
    # Show cumulative max deviation per lap
    max_devs = []
    for dev in deviations:
        if isinstance(dev, dict) and dev:
            max_devs.append(max(dev.values()))
        else:
            max_devs.append(0.0)
    ax.plot(laps[:len(max_devs)], max_devs, color=GOLD, linewidth=1.5)
    ax.axhline(y=threshold, color=CORAL, linestyle="--", linewidth=1,
               label=f"Threshold ({threshold})")
    ax.axvline(x=fault_lap, color=CORAL, linestyle=":", linewidth=1,
               label=f"Fault @ lap {fault_lap}")
    if detection_lap:
        ax.axvline(x=detection_lap, color=CYAN, linestyle="-.",
                   linewidth=1, label=f"Detection @ lap {detection_lap}")
    ax.fill_between([detection_lap or fault_lap, fault_lap], 0,
                    max(max_devs) if max_devs else 1,
                    color=CYAN, alpha=0.15, label=f"Lead: {lead} laps")
    ax.set_xlabel("Lap", fontsize=8)
    ax.set_ylabel("Max deviation", fontsize=8)
    ax.set_title("(D) Detection Confidence", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6)

    fig.suptitle("Panel 3: Fault Prediction", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    _savefig(fig, "panel_3_fault.png")


# ===================================================================
# Panel 4: Tire Degradation
# ===================================================================

def panel_4_tires(results):
    r = results.get("tire_degradation", {})
    lap_times = np.array(r.get("lap_times", [90] * 30))
    tire_depth_mean = np.array(r.get("tire_depth_mean", [0] * 30))
    tire_depth = r.get("tire_depth", {})
    predicted_cliff = r.get("predicted_cliff_lap", 25)
    actual_pit = r.get("actual_pit_lap", 25)
    surface = r.get("surface_data", {})
    n_laps = r.get("n_laps", 30)
    laps = np.arange(1, n_laps + 1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="white")

    # (A) Lap time vs lap number
    ax = axes[0]
    ax.plot(laps[:len(lap_times)], lap_times[:n_laps], color=TEAL,
            linewidth=1.5, marker="o", markersize=3)
    ax.axvline(x=actual_pit, color=CORAL, linestyle="--", linewidth=1,
               label=f"Pit @ lap {actual_pit}")
    ax.set_xlabel("Lap", fontsize=8)
    ax.set_ylabel("Lap time (s)", fontsize=8)
    ax.set_title("(A) Lap Time Degradation", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)

    # (B) Tire categorical depth vs lap
    ax = axes[1]
    tdm = tire_depth_mean[:n_laps]
    ax.plot(laps[:len(tdm)], tdm, color=GOLD, linewidth=1.5, label="Mean tire depth")
    for tn, color_t in [("FL_Wheel", TEAL), ("RR_Wheel", CORAL)]:
        td = np.array(tire_depth.get(tn, [0] * n_laps))[:n_laps]
        ax.plot(laps[:len(td)], td, color=color_t, linewidth=0.8,
                alpha=0.6, label=tn)
    ax.axvline(x=predicted_cliff, color=CYAN, linestyle="-.", linewidth=1.5,
               label=f"Predicted cliff @ {predicted_cliff}")
    ax.axvline(x=actual_pit, color=CORAL, linestyle="--", linewidth=1,
               label=f"Pit @ {actual_pit}")
    ax.set_xlabel("Lap", fontsize=8)
    ax.set_ylabel("Categorical depth", fontsize=8)
    ax.set_title("(B) Tire Depth (Cliff)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6)

    # (C) 3D surface: tire state vs (lap, speed)
    ax = fig.add_subplot(1, 4, 3, projection="3d")
    axes[2].set_visible(False)
    s_laps = np.array(surface.get("laps", []))
    s_speeds = np.array(surface.get("speeds", []))
    s_brakes = np.array(surface.get("brakes", []))
    s_tire = np.array(surface.get("tire_state", []))
    if len(s_laps) > 0:
        ax.scatter(s_laps, s_speeds, s_tire, c=s_tire, cmap="YlOrRd",
                   s=30, alpha=0.8, edgecolors="none")
    ax.set_xlabel("Lap", fontsize=7)
    ax.set_ylabel("Avg speed", fontsize=7)
    ax.set_zlabel("Tire state", fontsize=7)
    ax.set_title("(C) Tire State Surface", fontsize=10, fontweight="bold")

    # (D) Predicted vs actual pit lap
    ax = axes[3]
    labels = ["Predicted\nCliff", "Actual\nPit"]
    values = [predicted_cliff, actual_pit]
    bars = ax.bar(labels, values, color=[CYAN, CORAL], edgecolor="none",
                  width=0.5)
    ax.set_ylabel("Lap number", fontsize=8)
    ax.set_title("(D) Cliff Prediction Accuracy", fontsize=10,
                 fontweight="bold")
    error = abs(predicted_cliff - actual_pit)
    ax.text(0.5, max(values) * 0.5, f"Error: {error} laps",
            ha="center", fontsize=12, fontweight="bold",
            transform=ax.transAxes)

    fig.suptitle("Panel 4: Tire Degradation", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    _savefig(fig, "panel_4_tires.png")


# ===================================================================
# Panel 5: Racing Line
# ===================================================================

def panel_5_racing_line(results):
    r = results.get("racing_line", {})
    fastest_speed = np.array(r.get("fastest_speed", []))
    avg_speed = np.array(r.get("avg_speed", []))
    fine = r.get("fine_entropy", {})
    fine_Sk = np.array(fine.get("S_k", []))
    fine_St = np.array(fine.get("S_t", []))
    fine_Se = np.array(fine.get("S_e", []))
    sector_times = np.array(r.get("sector_times", []))
    optimal_sectors = np.array(r.get("optimal_sectors", []))
    fastest_idx = r.get("fastest_lap_idx", 0)
    n_sectors = r.get("n_sectors", 3)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="white")

    # (A) Speed trace: fastest vs average
    ax = axes[0]
    if len(fastest_speed) > 0:
        t = np.linspace(0, 1, len(fastest_speed))
        ax.plot(t, fastest_speed, color=TEAL, linewidth=1.2,
                label="Fastest lap")
    if len(avg_speed) > 0:
        t2 = np.linspace(0, 1, len(avg_speed))
        ax.plot(t2, avg_speed, color=GOLD, linewidth=1.0, alpha=0.7,
                label="Average")
    ax.set_xlabel("Normalised distance", fontsize=8)
    ax.set_ylabel("Speed (km/h)", fontsize=8)
    ax.set_title("(A) Speed Trace", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)

    # (B) S-entropy traces
    ax = axes[1]
    if len(fine_Sk) > 0:
        t3 = np.linspace(0, 1, len(fine_Sk))
        ax.plot(t3, fine_Sk, color=TEAL, linewidth=1.2, label="S_k (speed)")
        ax.plot(t3, fine_St, color=GOLD, linewidth=1.2, label="S_t (throttle)")
        ax.plot(t3, fine_Se, color=CORAL, linewidth=1.2, label="S_e (brake)")
    ax.set_xlabel("Normalised distance", fontsize=8)
    ax.set_ylabel("S-entropy (norm.)", fontsize=8)
    ax.set_title("(B) S-Entropy Trace", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)

    # (C) 3D: S-entropy trajectory in [0,1]^3
    ax = fig.add_subplot(1, 4, 3, projection="3d")
    axes[2].set_visible(False)
    if len(fine_Sk) > 0:
        ax.plot(fine_Sk, fine_St, fine_Se, color=TEAL, linewidth=1.5, alpha=0.8)
        ax.scatter(fine_Sk, fine_St, fine_Se, c=np.linspace(0, 1, len(fine_Sk)),
                   cmap="viridis", s=15, alpha=0.6)
        ax.scatter([fine_Sk[0]], [fine_St[0]], [fine_Se[0]], c="green",
                   s=40, marker="^", label="Start")
        ax.scatter([fine_Sk[-1]], [fine_St[-1]], [fine_Se[-1]], c="red",
                   s=40, marker="v", label="Finish")
    ax.set_xlabel("S_k", fontsize=7)
    ax.set_ylabel("S_t", fontsize=7)
    ax.set_zlabel("S_e", fontsize=7)
    ax.set_title("(C) S-Entropy Trajectory", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6)

    # (D) Sector time comparison
    ax = axes[3]
    if len(sector_times) > 0 and len(optimal_sectors) > 0:
        x = np.arange(n_sectors)
        width = 0.35
        fastest_sec = sector_times[fastest_idx] if fastest_idx < len(sector_times) else optimal_sectors
        ax.bar(x - width / 2, fastest_sec, width, color=TEAL,
               label="Fastest actual", edgecolor="none")
        ax.bar(x + width / 2, optimal_sectors, width, color=GOLD,
               label="Optimal (best sectors)", edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{i+1}" for i in range(n_sectors)], fontsize=8)
    ax.set_ylabel("Sector time (s)", fontsize=8)
    ax.set_title("(D) Sector Comparison", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)

    fig.suptitle("Panel 5: Racing Line (S-Entropy Analysis)", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    _savefig(fig, "panel_5_racing_line.png")


# ===================================================================
# Entry point
# ===================================================================

def generate_all(results: dict):
    """Generate all 5 panels from validation results."""
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"  Output directory: {OUT_DIR}")
    panel_1_circuit(results)
    panel_2_reconstruction(results)
    panel_3_fault(results)
    panel_4_tires(results)
    panel_5_racing_line(results)
    print(f"  All 5 panels generated.")


if __name__ == "__main__":
    # Standalone: run all validations then generate panels
    import json

    json_path = os.path.join(OUT_DIR, "validation_results.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            results = json.load(f)
        generate_all(results)
    else:
        print("No validation_results.json found. Run validation first:")
        print("  python -m philharmonic.validation.run_all")
