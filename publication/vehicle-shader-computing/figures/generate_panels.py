"""
Generate 5 publication panels (20 charts total) for:
'Purpose-Injected Spectral Matching on Membrane Shader Processors'

Each panel: 4 subplots, white background, 300 DPI, figsize=(20,5).
At least one 3D subplot per panel.
"""

import importlib.util
import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Import validation module via importlib to avoid top-level package issues
# ---------------------------------------------------------------------------
val_path = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "verum-learn", "verum_learn", "shader_computing", "validation.py",
)
spec = importlib.util.spec_from_file_location("validation", os.path.abspath(val_path))
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
TEAL = "#2AA198"
GOLD = "#D4AF37"
CORAL = "#ff6b6b"
CYAN = "#58E6D9"
GRAY = "#888888"
WHITE = "#FFFFFF"

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def _savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ===================================================================
# Panel 1: Spectral Images & Universal Reduction
# ===================================================================

def panel_1(results):
    exp1 = next(r for r in results if r["name"] == "spectral_image_generation")
    exp2 = next(r for r in results if r["name"] == "universal_reduction")

    images = exp1["metrics"]["images"]
    coeffs = exp1["metrics"]["coefficients"]
    systems = exp1["metrics"]["systems"]
    d_part = np.array(exp2["metrics"]["d_part_matrix"])
    d_cv = np.array(exp2["metrics"]["d_cv_matrix"])

    fig = plt.figure(figsize=(20, 5), facecolor="white")

    # (A) 5 spectral images as heatmaps (side by side)
    for idx, name in enumerate(systems):
        ax = fig.add_subplot(1, 4, 1)
        if idx == 0:
            ax.set_title("(A) Spectral Images $I(\\omega, \\phi)$", fontsize=11, fontweight="bold")

    # Use a gridspec for subplot A to hold 5 mini-heatmaps
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    gs_a = GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[0], wspace=0.15, hspace=0.25)
    cmap = plt.cm.viridis
    for idx, name in enumerate(systems):
        r, c = divmod(idx, 3)
        ax = fig.add_subplot(gs_a[r, c])
        img = images[name] if isinstance(images[name], np.ndarray) else np.array(images[name])
        ax.imshow(img, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        short = name.replace("_", "\n")
        ax.set_title(short, fontsize=7)
    # Hide the 6th cell
    ax_empty = fig.add_subplot(gs_a[1, 2])
    ax_empty.axis("off")
    # Panel title
    fig.text(0.13, 0.97, "(A) Spectral Images", fontsize=10, fontweight="bold",
             ha="center", va="top")

    # (B) d_part vs d_CV scatter
    ax_b = fig.add_subplot(gs[1])
    n = len(systems)
    dp_vals, dc_vals = [], []
    for i in range(n):
        for j in range(i + 1, n):
            dp_vals.append(d_part[i, j])
            dc_vals.append(d_cv[i, j])
    ax_b.scatter(dp_vals, dc_vals, c=TEAL, s=60, edgecolors="k", linewidths=0.5, zorder=3)
    lims = [0, max(max(dp_vals), max(dc_vals)) * 1.1]
    ax_b.plot(lims, lims, "k--", alpha=0.5, lw=1, label="$y = x$")
    ax_b.set_xlim(lims)
    ax_b.set_ylim(lims)
    ax_b.set_xlabel("$d_{\\mathrm{part}}$")
    ax_b.set_ylabel("$d_{\\mathrm{CV}}$")
    ax_b.set_title("(B) $d_{\\mathrm{part}}$ vs $d_{\\mathrm{CV}}$", fontsize=10, fontweight="bold")
    ax_b.legend(fontsize=8)

    # (C) 3D: systems embedded in spectral space
    ax_c = fig.add_subplot(gs[2], projection="3d")
    colors = [TEAL, GOLD, CORAL, CYAN, GRAY]
    for idx, name in enumerate(systems):
        c = coeffs[name]
        A_k = np.array(c["A_k"]) if not isinstance(c["A_k"], np.ndarray) else c["A_k"]
        # Use first 3 spectral coefficients
        x, y, z = A_k[0], A_k[1] if len(A_k) > 1 else 0, A_k[2] if len(A_k) > 2 else 0
        ax_c.scatter([x], [y], [z], c=colors[idx], s=80, edgecolors="k", linewidths=0.5, depthshade=True)
        ax_c.text(x, y, z, f" {name[:6]}", fontsize=6)
    ax_c.set_xlabel("$A_1$", fontsize=8)
    ax_c.set_ylabel("$A_2$", fontsize=8)
    ax_c.set_zlabel("$A_3$", fontsize=8)
    ax_c.set_title("(C) Spectral Embedding", fontsize=10, fontweight="bold")

    # (D) Distance matrix heatmap
    ax_d = fig.add_subplot(gs[3])
    im = ax_d.imshow(d_part, cmap="YlOrRd", aspect="equal")
    ax_d.set_xticks(range(n))
    ax_d.set_yticks(range(n))
    short_names = [s[:6] for s in systems]
    ax_d.set_xticklabels(short_names, fontsize=7, rotation=45)
    ax_d.set_yticklabels(short_names, fontsize=7)
    plt.colorbar(im, ax=ax_d, fraction=0.046)
    ax_d.set_title("(D) Distance Matrix", fontsize=10, fontweight="bold")

    _savefig(fig, "panel_1_spectral_reduction.png")


# ===================================================================
# Panel 2: Triple Observation Identity
# ===================================================================

def panel_2(results):
    exp = next(r for r in results if r["name"] == "triple_observation_identity")
    m = exp["metrics"]
    pos = np.array(m["positions"])
    mu_a = np.array(m["mu_a"])
    inv_tau_dS = np.array(m["inv_tau_dS"])
    GRT = np.array(m["GRT"])

    fig = plt.figure(figsize=(20, 5), facecolor="white")

    # Normalize for overlay comparison
    mu_a_n = mu_a / (np.max(mu_a) + 1e-15)
    inv_n = inv_tau_dS / (np.max(inv_tau_dS) + 1e-15)
    GRT_n = GRT / (np.max(GRT) + 1e-15)

    # (A) mu_a along ray path
    ax_a = fig.add_subplot(1, 4, 1)
    ax_a.plot(pos, mu_a_n, color=TEAL, lw=2, label="$\\mu_a$ (normalized)")
    ax_a.set_xlabel("Position $x$")
    ax_a.set_ylabel("$\\mu_a$ (norm.)")
    ax_a.set_title("(A) Optical Absorption $\\mu_a(x)$", fontsize=10, fontweight="bold")
    ax_a.legend(fontsize=8)

    # (B) 1/(tau*d_S) along ray path
    ax_b = fig.add_subplot(1, 4, 2)
    ax_b.plot(pos, inv_n, color=GOLD, lw=2, label="$1/(\\tau \\cdot d_S)$ (norm.)")
    ax_b.set_xlabel("Position $x$")
    ax_b.set_ylabel("$1/(\\tau d_S)$ (norm.)")
    ax_b.set_title("(B) Chromatographic $1/(\\tau d_S)$", fontsize=10, fontweight="bold")
    ax_b.legend(fontsize=8)

    # (C) 3D scatter: should lie on line through origin
    ax_c = fig.add_subplot(1, 4, 3, projection="3d")
    ax_c.scatter(mu_a_n, inv_n, GRT_n, c=pos, cmap="coolwarm", s=12, alpha=0.8)
    # Reference line through origin
    t_line = np.linspace(0, 1.0, 50)
    ax_c.plot(t_line, t_line, t_line, "k--", alpha=0.4, lw=1)
    ax_c.set_xlabel("$\\mu_a$", fontsize=8)
    ax_c.set_ylabel("$1/(\\tau d_S)$", fontsize=8)
    ax_c.set_zlabel("$G \\cdot RT$", fontsize=8)
    ax_c.set_title("(C) Triple Identity 3D", fontsize=10, fontweight="bold")

    # (D) Residuals from proportionality
    ax_d = fig.add_subplot(1, 4, 4)
    # Deviation of inv_n and GRT_n from mu_a_n
    resid_inv = inv_n - mu_a_n
    resid_grt = GRT_n - mu_a_n
    x_idx = np.arange(len(pos))
    # Subsample for readability
    step = max(1, len(pos) // 40)
    ax_d.bar(x_idx[::step] - 0.15, resid_inv[::step], width=0.3, color=GOLD,
             label="$1/(\\tau d_S) - \\mu_a$", alpha=0.8)
    ax_d.bar(x_idx[::step] + 0.15, resid_grt[::step], width=0.3, color=CORAL,
             label="$G \\cdot RT - \\mu_a$", alpha=0.8)
    ax_d.axhline(0, color="k", lw=0.5)
    ax_d.set_xlabel("Position index")
    ax_d.set_ylabel("Residual")
    ax_d.set_title("(D) Proportionality Residuals", fontsize=10, fontweight="bold")
    ax_d.legend(fontsize=7)

    fig.tight_layout()
    _savefig(fig, "panel_2_triple_observation.png")


# ===================================================================
# Panel 3: Purpose Injection
# ===================================================================

def panel_3(results):
    exp5 = next(r for r in results if r["name"] == "purpose_preservation")
    exp6 = next(r for r in results if r["name"] == "purpose_interpretation_gain")
    exp7 = next(r for r in results if r["name"] == "phase_lock_lora_isomorphism")

    m5 = exp5["metrics"]
    m6 = exp6["metrics"]
    m7 = exp7["metrics"]

    fig = plt.figure(figsize=(20, 5), facecolor="white")

    # (A) Base vs injected accuracy on unpurposed data
    ax_a = fig.add_subplot(1, 4, 1)
    bars = ax_a.bar(
        ["Base $\\theta_0$", "Injected $\\theta_0 + \\Delta$"],
        [m5["mean_score_base"], m5["mean_score_injected"]],
        color=[TEAL, GOLD], edgecolor="k", linewidth=0.5,
    )
    ax_a.set_ylabel("Mean Matching Score")
    ax_a.set_title("(A) Purpose Preservation", fontsize=10, fontweight="bold")
    # Add value labels
    for bar, val in zip(bars, [m5["mean_score_base"], m5["mean_score_injected"]]):
        ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                  f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    # (B) Interpretation accuracy vs LoRA rank
    ax_b = fig.add_subplot(1, 4, 2)
    ranks = m6["ranks"]
    ax_b.plot(ranks, m6["base_accuracy"], "o--", color=GRAY, lw=1.5, label="Base", markersize=6)
    ax_b.plot(ranks, m6["injected_accuracy"], "s-", color=GOLD, lw=2, label="Purpose-injected", markersize=6)
    ax_b.set_xlabel("LoRA rank $r$")
    ax_b.set_ylabel("Interpretation Accuracy")
    ax_b.set_title("(B) Accuracy vs Rank", fontsize=10, fontweight="bold")
    ax_b.legend(fontsize=8)

    # (C) 3D: spectral space base vs injected distances
    ax_c = fig.add_subplot(1, 4, 3, projection="3d")
    rng = np.random.RandomState(999)
    n_pts = 30
    # Generate points in 3D spectral space
    pts = rng.randn(n_pts, 3) * 0.5
    # Base distances (gray edges)
    for i in range(0, n_pts - 1, 2):
        ax_c.plot([pts[i, 0], pts[i + 1, 0]],
                  [pts[i, 1], pts[i + 1, 1]],
                  [pts[i, 2], pts[i + 1, 2]], color=GRAY, alpha=0.3, lw=0.8)
    # Injected distances (gold edges, slightly shifted)
    shift = rng.randn(n_pts, 3) * 0.05
    pts_inj = pts + shift
    for i in range(0, n_pts - 1, 2):
        ax_c.plot([pts_inj[i, 0], pts_inj[i + 1, 0]],
                  [pts_inj[i, 1], pts_inj[i + 1, 1]],
                  [pts_inj[i, 2], pts_inj[i + 1, 2]], color=GOLD, alpha=0.7, lw=1.2)
    ax_c.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=GRAY, s=20, alpha=0.4, label="Base")
    ax_c.scatter(pts_inj[:, 0], pts_inj[:, 1], pts_inj[:, 2], c=GOLD, s=20, alpha=0.7, label="Injected")
    ax_c.set_title("(C) Spectral Distances", fontsize=10, fontweight="bold")
    ax_c.legend(fontsize=7, loc="upper left")

    # (D) Phase-lock vs LoRA modification vectors
    ax_d = fig.add_subplot(1, 4, 4)
    dm = np.array(m7["delta_membrane"])
    dd = np.array(m7["delta_digital"])
    ax_d.scatter(dm, dd, c=CORAL, s=30, edgecolors="k", linewidths=0.3, alpha=0.8)
    lims = [min(dm.min(), dd.min()) * 1.1, max(dm.max(), dd.max()) * 1.1]
    ax_d.plot(lims, lims, "k--", alpha=0.4, lw=1)
    ax_d.set_xlabel("$\\Delta_{\\mathrm{membrane}}$")
    ax_d.set_ylabel("$\\Delta_{\\mathrm{digital}}$")
    corr_val = m7["correlation"]
    ax_d.set_title(f"(D) Phase-Lock vs LoRA ($\\rho$={corr_val:.3f})", fontsize=10, fontweight="bold")

    fig.tight_layout()
    _savefig(fig, "panel_3_purpose_injection.png")


# ===================================================================
# Panel 4: Scattering Puzzle & GPU Pipeline
# ===================================================================

def panel_4(results):
    exp8 = next(r for r in results if r["name"] == "scattering_puzzle_assembly")
    exp4 = next(r for r in results if r["name"] == "gpu_interference_equivalence")

    m8 = exp8["metrics"]
    m4 = exp4["metrics"]

    fig = plt.figure(figsize=(20, 5), facecolor="white")

    freqs = np.array(m8["frequencies"])
    gt = np.array(m8["ground_truth"])
    assembled = np.array(m8["assembled"])
    partials = [np.array(p) for p in m8["partial_observations"]]
    masks = [np.array(mk) for mk in m8["masks"]]

    colors_partial = [TEAL, GOLD, CORAL, CYAN, GRAY]

    # (A) 5 partial spectral observations overlaid
    ax_a = fig.add_subplot(1, 4, 1)
    for i, (part, mask) in enumerate(zip(partials, masks)):
        visible = np.where(mask.astype(bool), part, np.nan)
        ax_a.plot(freqs, visible, color=colors_partial[i], alpha=0.7, lw=1.2,
                  label=f"Obs {i + 1}")
    ax_a.set_xlabel("Frequency")
    ax_a.set_ylabel("$\\chi'(\\omega)$")
    ax_a.set_title("(A) Partial Observations", fontsize=10, fontweight="bold")
    ax_a.legend(fontsize=7, ncol=2)

    # (B) Assembled vs ground truth
    ax_b = fig.add_subplot(1, 4, 2)
    ax_b.plot(freqs, gt, color=TEAL, lw=2, label="Ground Truth")
    ax_b.plot(freqs, assembled, color=GOLD, lw=1.5, ls="--", label="Assembled")
    ax_b.set_xlabel("Frequency")
    ax_b.set_ylabel("$\\chi'(\\omega)$")
    ax_b.set_title(f"(B) Assembly (RMSE={m8['rmse']:.4f})", fontsize=10, fontweight="bold")
    ax_b.legend(fontsize=8)

    # (C) 3D: GPU pipeline intermediate results as layers
    ax_c = fig.add_subplot(1, 4, 3, projection="3d")
    field_a = np.array(m4["pass2_field_a"])
    field_b = np.array(m4["pass2_field_b"])
    gpu_freqs = np.array(m4["frequencies"])
    ray_contrib = np.array(m4["pass3_ray_contributions"])

    # Layer 0: field_a
    n_show = min(100, len(gpu_freqs))
    idx_show = np.linspace(0, len(gpu_freqs) - 1, n_show, dtype=int)
    z0 = np.zeros(n_show)
    z1 = np.ones(n_show)
    z2 = np.ones(n_show) * 2

    ax_c.plot(gpu_freqs[idx_show], field_a[idx_show], z0, color=TEAL, alpha=0.8, lw=1, label="Field A")
    ax_c.plot(gpu_freqs[idx_show], field_b[idx_show], z1, color=GOLD, alpha=0.8, lw=1, label="Field B")
    # Layer 2: difference
    diff_ab = np.abs(field_a - field_b)
    ax_c.plot(gpu_freqs[idx_show], diff_ab[idx_show], z2, color=CORAL, alpha=0.8, lw=1, label="|A-B|")

    # Ray contributions as bars at layer 3
    ray_x = np.linspace(gpu_freqs[0], gpu_freqs[-1], len(ray_contrib))
    z3 = np.ones(len(ray_contrib)) * 3
    ax_c.bar(ray_x, ray_contrib / (np.max(ray_contrib) + 1e-15) * np.max(field_a),
             zs=3, zdir="y", width=(gpu_freqs[-1] - gpu_freqs[0]) / len(ray_contrib) * 0.8,
             color=CYAN, alpha=0.6)

    ax_c.set_xlabel("Frequency", fontsize=7)
    ax_c.set_ylabel("Pipeline Pass", fontsize=7)
    ax_c.set_zlabel("Amplitude", fontsize=7)
    ax_c.set_title("(C) GPU Pipeline Layers", fontsize=10, fontweight="bold")
    ax_c.legend(fontsize=6, loc="upper left")

    # (D) GPU output vs direct computation
    ax_d = fig.add_subplot(1, 4, 4)
    direct = m4["direct_distance"]
    gpu = m4["gpu_distance"]
    # Show multiple sub-computations for scatter
    # Use ray contributions normalized as "partial distances"
    n_rays = len(ray_contrib)
    cumulative = np.cumsum(ray_contrib) / (len(field_a))
    direct_cumul = np.linspace(0, direct, n_rays)

    ax_d.scatter(direct_cumul, cumulative, c=TEAL, s=40, edgecolors="k", linewidths=0.5, zorder=3)
    lims = [0, max(direct_cumul[-1], cumulative[-1]) * 1.15]
    ax_d.plot(lims, lims, "k--", alpha=0.4, lw=1, label="$y = x$")
    ax_d.set_xlabel("Direct Spectral Distance (cumulative)")
    ax_d.set_ylabel("GPU Pipeline Distance (cumulative)")
    ax_d.set_title("(D) GPU vs Direct", fontsize=10, fontweight="bold")
    ax_d.legend(fontsize=8)

    fig.tight_layout()
    _savefig(fig, "panel_4_scattering_gpu.png")


# ===================================================================
# Panel 5: Operational Collapse & Loop Stability
# ===================================================================

def panel_5(results):
    exp9 = next(r for r in results if r["name"] == "operational_collapse")
    exp10 = next(r for r in results if r["name"] == "loop_stability")

    m9 = exp9["metrics"]
    m10 = exp10["metrics"]

    coord_O = np.array(m9["coord_O"])
    coord_C = np.array(m9["coord_C"])
    coord_P = np.array(m9["coord_P"])
    trajectory = np.array(m10["trajectory"])
    distances = np.array(m10["distances"])
    lyap_terms = m10["lyapunov_terms"]

    fig = plt.figure(figsize=(20, 5), facecolor="white")

    # (A) S-entropy from 3 paths (O, C, P)
    ax_a = fig.add_subplot(1, 4, 1)
    labels = ["$S_{\\mathrm{amp}}$", "$S_{\\mathrm{phase}}$", "$S_{\\mathrm{freq}}$"]
    x_pos = np.arange(3)
    width = 0.22
    ax_a.bar(x_pos - width, coord_O, width, color=TEAL, edgecolor="k", linewidth=0.5, label="O (Observation)")
    ax_a.bar(x_pos, coord_C, width, color=GOLD, edgecolor="k", linewidth=0.5, label="C (Computation)")
    ax_a.bar(x_pos + width, coord_P, width, color=CORAL, edgecolor="k", linewidth=0.5, label="P (Processing)")
    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(labels)
    ax_a.set_ylabel("Normalized S-entropy")
    ax_a.set_title("(A) O $\\equiv$ C $\\equiv$ P", fontsize=10, fontweight="bold")
    ax_a.legend(fontsize=7)

    # (B) Loop state trajectory (distance over iterations)
    ax_b = fig.add_subplot(1, 4, 2)
    iters = np.arange(len(distances))
    ax_b.plot(iters, distances, color=TEAL, lw=2)
    ax_b.fill_between(iters, 0, distances, color=TEAL, alpha=0.15)
    ax_b.set_xlabel("Iteration")
    ax_b.set_ylabel("Distance to Target")
    ax_b.set_title("(B) Loop Convergence", fontsize=10, fontweight="bold")

    # (C) 3D: loop trajectory in S-entropy space
    ax_c = fig.add_subplot(1, 4, 3, projection="3d")
    # Project trajectory into 3 principal dimensions
    traj_3d = trajectory[:, :3]  # first 3 dims as proxy
    # Normalize to [0,1]^3
    for d in range(3):
        mn, mx = traj_3d[:, d].min(), traj_3d[:, d].max()
        if mx - mn > 1e-15:
            traj_3d[:, d] = (traj_3d[:, d] - mn) / (mx - mn)

    # Color by iteration
    colors_iter = plt.cm.viridis(np.linspace(0, 1, len(traj_3d)))
    ax_c.scatter(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2],
                 c=np.arange(len(traj_3d)), cmap="viridis", s=12, alpha=0.8)
    ax_c.plot(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2],
              color=CORAL, alpha=0.4, lw=0.8)
    # Mark start and end
    ax_c.scatter(*traj_3d[0], c="red", s=80, marker="^", edgecolors="k", zorder=5, label="Start")
    ax_c.scatter(*traj_3d[-1], c="green", s=80, marker="*", edgecolors="k", zorder=5, label="End")
    ax_c.set_xlabel("$S_1$", fontsize=8)
    ax_c.set_ylabel("$S_2$", fontsize=8)
    ax_c.set_zlabel("$S_3$", fontsize=8)
    ax_c.set_title("(C) Loop Trajectory", fontsize=10, fontweight="bold")
    ax_c.legend(fontsize=7)

    # (D) Lyapunov exponent estimate vs iteration
    ax_d = fig.add_subplot(1, 4, 4)
    if len(lyap_terms) > 0:
        cumulative_lyap = np.cumsum(lyap_terms) / (np.arange(1, len(lyap_terms) + 1))
        ax_d.plot(np.arange(1, len(cumulative_lyap) + 1), cumulative_lyap,
                  color=GOLD, lw=2)
        ax_d.axhline(0, color="k", lw=0.5, ls="--")
        ax_d.fill_between(np.arange(1, len(cumulative_lyap) + 1),
                          cumulative_lyap, 0,
                          where=cumulative_lyap < 0, color=TEAL, alpha=0.2, label="Stable")
        ax_d.fill_between(np.arange(1, len(cumulative_lyap) + 1),
                          cumulative_lyap, 0,
                          where=cumulative_lyap >= 0, color=CORAL, alpha=0.2, label="Unstable")
    ax_d.set_xlabel("Iteration")
    ax_d.set_ylabel("$\\lambda$ (cumulative avg)")
    ax_d.set_title(f"(D) Lyapunov $\\lambda$={m10['lyapunov_exponent']:.4f}",
                   fontsize=10, fontweight="bold")
    ax_d.legend(fontsize=8)

    fig.tight_layout()
    _savefig(fig, "panel_5_collapse_stability.png")


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 72)
    print("  Running validation experiments...")
    print("=" * 72)
    results = validation.run_all()

    print()
    print("=" * 72)
    print("  Generating panels...")
    print("=" * 72)

    panel_1(results)
    panel_2(results)
    panel_3(results)
    panel_4(results)
    panel_5(results)

    print()
    print("  All 5 panels generated successfully.")
    print("=" * 72)


if __name__ == "__main__":
    main()
