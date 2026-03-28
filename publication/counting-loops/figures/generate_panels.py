"""
Generate 5 publication panels for the Counting Loops paper.
Each panel: 4 charts in a row, white background, 300 DPI, at least one 3D chart.

Panel 1: Counting Loops & Harmonic Network
Panel 2: Precision & Fusion
Panel 3: Sufficiency & Transplanckian
Panel 4: Positioning & Detection
Panel 5: Duality & Commutation
"""

import sys
import os
import types
import json

# ── Mock verum_learn to avoid top-level import issues ─────────────────────────
pkg = types.ModuleType('verum_learn')
pkg.__path__ = ['verum_learn']
pkg.__package__ = 'verum_learn'
sys.modules['verum_learn'] = pkg

cl_pkg = types.ModuleType('verum_learn.counting_loops')
cl_pkg.__path__ = ['verum_learn/counting_loops']
cl_pkg.__package__ = 'verum_learn.counting_loops'
sys.modules['verum_learn.counting_loops'] = cl_pkg

# ── Import validation module via importlib ────────────────────────────────────
import importlib.util

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
VALIDATION_PATH = os.path.join(PROJECT_ROOT, 'verum-learn', 'verum_learn',
                               'counting_loops', 'validation.py')

spec = importlib.util.spec_from_file_location("validation", VALIDATION_PATH)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

FIGDIR = SCRIPT_DIR

# ── Color Scheme ──────────────────────────────────────────────────────────────
C_TEAL = '#2AA198'
C_GOLD = '#D4AF37'
C_CORAL = '#ff6b6b'
C_CYAN = '#58E6D9'
C_DARK = '#0a0a0a'

C_PALETTE = [C_TEAL, C_GOLD, C_CORAL, C_CYAN, '#6C71C4', '#859900', '#CB4B16',
             '#D33682', '#268BD2', '#DC322F', '#B58900', '#2AA198',
             '#93a1a1', '#586e75', '#002b36']


def _style_axis(ax, title, xlabel, ylabel, fontsize_title=10, fontsize_label=9):
    """Apply consistent styling."""
    ax.set_title(title, fontsize=fontsize_title, fontweight='bold', color=C_DARK)
    ax.set_xlabel(xlabel, fontsize=fontsize_label, color=C_DARK)
    ax.set_ylabel(ylabel, fontsize=fontsize_label, color=C_DARK)
    ax.tick_params(labelsize=8, colors=C_DARK)
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_color('#cccccc')


def _style_3d(ax, title, xlabel, ylabel, zlabel):
    """Apply consistent 3D styling."""
    ax.set_title(title, fontsize=10, fontweight='bold', color=C_DARK)
    ax.set_xlabel(xlabel, fontsize=7, labelpad=5)
    ax.set_ylabel(ylabel, fontsize=7, labelpad=5)
    ax.set_zlabel(zlabel, fontsize=7, labelpad=5)
    ax.tick_params(labelsize=6)
    ax.set_facecolor('white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False


# ═══════════════════════════════════════════════════════════════════════════════
# Panel 1: Counting Loops & Harmonic Network
# ═══════════════════════════════════════════════════════════════════════════════

def panel_1(results_dict):
    """Panel 1: Counting Loops & Harmonic Network
    (A) dM/dt measured vs predicted (scatter with y=x)
    (B) Harmonic coincidence adjacency matrix heatmap
    (C) 3D: oscillator network in (frequency, phase, coupling_degree)
    (D) Enhancement factor per mechanism (bar chart, log scale)
    """
    m_cl = results_dict['counting_loop_identity']['metrics']
    m_hc = results_dict['harmonic_coincidence']['metrics']
    m_tp = results_dict['transplanckian_enhancement']['metrics']

    freqs = np.array(m_cl['frequencies'])
    dM_measured = np.array(m_cl['dM_dt_measured'])
    dM_predicted = np.array(m_cl['dM_dt_predicted'])
    names = m_cl['oscillator_names']
    adj = np.array(m_hc['adjacency_matrix'])
    coupling_deg = np.array(m_hc['coupling_degrees'])
    phases = np.array(m_hc['phases'])

    fig = plt.figure(figsize=(20, 5), facecolor='white')
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    # ── (A) Counting loop identity: measured vs predicted ─────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(dM_predicted, dM_measured, c=C_TEAL, s=60, edgecolors='white',
                linewidth=0.5, zorder=5, label='Oscillators')
    # y=x reference line
    lims = [min(dM_predicted.min(), dM_measured.min()) * 0.5,
            max(dM_predicted.max(), dM_measured.max()) * 2]
    ax1.plot(lims, lims, '--', color=C_CORAL, linewidth=1.5, alpha=0.7, label='y = x')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    for i, name in enumerate(names):
        ax1.annotate(name[:4], (dM_predicted[i], dM_measured[i]),
                     fontsize=5, ha='left', va='bottom', color=C_DARK)
    _style_axis(ax1, '(A) Counting Loop Identity\ndM/dt: Measured vs Predicted',
                'Predicted dM/dt', 'Measured dM/dt')
    ax1.legend(fontsize=7, loc='lower right')

    # ── (B) Harmonic coincidence adjacency matrix ─────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(adj, cmap='YlOrRd', interpolation='nearest', aspect='equal')
    ax2.set_xticks(range(len(names)))
    ax2.set_yticks(range(len(names)))
    ax2.set_xticklabels([n[:4] for n in names], rotation=90, fontsize=6)
    ax2.set_yticklabels([n[:4] for n in names], fontsize=6)
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Connected', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax2.set_title('(B) Harmonic Coincidence Graph\n(adjacency matrix)',
                  fontsize=10, fontweight='bold', color=C_DARK)

    # ── (C) 3D: oscillator network ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    log_freqs = np.log10(freqs)
    # Phase: sum of phase relationships
    phase_sums = phases.sum(axis=1)
    sc = ax3.scatter(log_freqs, phase_sums, coupling_deg,
                     c=C_TEAL, s=80, edgecolors='white', linewidth=0.5,
                     depthshade=True)
    # Draw edges
    for i in range(len(freqs)):
        for j in range(i + 1, len(freqs)):
            if adj[i, j] > 0:
                ax3.plot([log_freqs[i], log_freqs[j]],
                         [phase_sums[i], phase_sums[j]],
                         [coupling_deg[i], coupling_deg[j]],
                         color=C_GOLD, alpha=0.3, linewidth=0.8)
    for i, name in enumerate(names):
        ax3.text(log_freqs[i], phase_sums[i], coupling_deg[i], f' {name[:5]}',
                 fontsize=5, color=C_DARK)
    _style_3d(ax3, '(C) Oscillator Network',
              r'$\log_{10}$ Freq (Hz)', 'Phase Sum', 'Coupling Degree')

    # ── (D) Enhancement factors (bar chart, log scale) ────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    enh_names = m_tp['enhancement_names']
    log10_vals = np.array(m_tp['log10_values'])
    short_names = ['Ternary', 'Multi-modal', 'Harmonic', 'Poincare', 'Continuous']
    bar_colors = [C_TEAL, C_GOLD, C_CORAL, C_CYAN, '#6C71C4']
    bars = ax4.bar(range(len(short_names)), log10_vals, color=bar_colors,
                   edgecolor='white', linewidth=0.5)
    ax4.set_xticks(range(len(short_names)))
    ax4.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
    for i, (bar, val) in enumerate(zip(bars, log10_vals)):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=7, color=C_DARK)
    _style_axis(ax4, f'(D) Enhancement Mechanisms\n'
                f'Total: $10^{{{m_tp["log10_total"]:.1f}}}$',
                'Mechanism', r'$\log_{10}$ Enhancement')

    fig.savefig(os.path.join(FIGDIR, 'panel_1_counting_loops.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved panel_1_counting_loops.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Panel 2: Precision & Fusion
# ═══════════════════════════════════════════════════════════════════════════════

def panel_2(results_dict):
    """Panel 2: Precision & Fusion
    (A) Precision-by-difference: delta_P distribution (histogram)
    (B) S_t distribution vs uniform baseline
    (C) 3D: 5 observer S-entropy points in [0,1]^3
    (D) Pairwise observer distances (bar chart)
    """
    m_pbd = results_dict['precision_by_difference']['metrics']
    m_mmf = results_dict['multi_modal_fusion']['metrics']

    delta_P = np.array(m_pbd['delta_P'])
    S_t = np.array(m_pbd['S_t'])
    kl_div = m_pbd['KL_divergence']
    obs_pts = np.array(m_mmf['observer_points'])
    fused_pt = np.array(m_mmf['fused_point'])
    obs_names = m_mmf['observer_names']
    pw_labels = m_mmf['pairwise_labels']
    pw_values = np.array(m_mmf['pairwise_values'])

    fig = plt.figure(figsize=(20, 5), facecolor='white')
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    # ── (A) delta_P distribution ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(delta_P * 1e12, bins=60, color=C_TEAL, edgecolor='white',
             linewidth=0.3, alpha=0.85, density=True)
    _style_axis(ax1, r'(A) Precision-by-Difference' '\n' r'$\Delta P = T_{ref} - t_{local}$',
                r'$\Delta P$ (ps)', 'Density')
    ax1.axvline(0, color=C_CORAL, linewidth=1.0, linestyle='--', alpha=0.7)

    # ── (B) S_t distribution vs uniform ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    n_bins = 50
    counts, edges, patches = ax2.hist(S_t, bins=n_bins, range=(0, 1),
                                       color=C_GOLD, edgecolor='white',
                                       linewidth=0.3, alpha=0.85, density=True,
                                       label=r'$S_t$ distribution')
    # Uniform baseline
    ax2.axhline(1.0, color=C_CORAL, linewidth=1.5, linestyle='--', alpha=0.7,
                label='Uniform baseline')
    _style_axis(ax2, f'(B) S-Entropy from Timing Jitter\nKL = {kl_div:.4f}',
                r'$S_t = \phi / 2\pi$', 'Density')
    ax2.legend(fontsize=7)

    # ── (C) 3D: observer S-entropy points ─────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    obs_colors = [C_TEAL, C_GOLD, C_CORAL, C_CYAN, '#6C71C4']
    for i, (name, color) in enumerate(zip(obs_names, obs_colors)):
        ax3.scatter(obs_pts[i, 0], obs_pts[i, 1], obs_pts[i, 2],
                    c=color, s=100, edgecolors='white', linewidth=0.5,
                    label=name, depthshade=True, zorder=5)
    # Fused point (larger, black outline)
    ax3.scatter(fused_pt[0], fused_pt[1], fused_pt[2],
                c='white', s=200, edgecolors=C_DARK, linewidth=2,
                marker='*', label='Fused', zorder=10)
    # Draw lines from each observer to fused point
    for i in range(len(obs_names)):
        ax3.plot([obs_pts[i, 0], fused_pt[0]],
                 [obs_pts[i, 1], fused_pt[1]],
                 [obs_pts[i, 2], fused_pt[2]],
                 color='gray', alpha=0.4, linewidth=0.8)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_zlim(0, 1)
    _style_3d(ax3, '(C) Multi-Modal S-Entropy\n(5 observers + fused)',
              r'$S_1$', r'$S_2$', r'$S_3$')
    ax3.legend(fontsize=5, loc='upper left', ncol=2)

    # ── (D) Pairwise observer distances ───────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    x_pos = np.arange(len(pw_labels))
    bar_colors_pw = [C_TEAL if v < 0.2 else C_CORAL for v in pw_values]
    ax4.bar(x_pos, pw_values, color=bar_colors_pw, edgecolor='white', linewidth=0.5)
    ax4.axhline(0.2, color=C_CORAL, linewidth=1.5, linestyle='--', alpha=0.7,
                label='Threshold (0.2)')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(pw_labels, rotation=90, fontsize=6)
    _style_axis(ax4, '(D) Pairwise Observer Distances',
                'Observer pair', 'Distance')
    ax4.legend(fontsize=7)

    fig.savefig(os.path.join(FIGDIR, 'panel_2_precision_fusion.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved panel_2_precision_fusion.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Panel 3: Sufficiency & Transplanckian
# ═══════════════════════════════════════════════════════════════════════════════

def panel_3(results_dict):
    """Panel 3: Sufficiency & Transplanckian
    (A) Triple convergence: eps_osc, eps_cat, eps_par for 20 scenarios
    (B) Confusion matrix heatmap
    (C) 3D surface: enhancement as function of (oscillator count, integration time)
    (D) Cumulative enhancement product on log scale
    """
    m_sf = results_dict['sufficiency_recognition']['metrics']
    m_tp = results_dict['transplanckian_enhancement']['metrics']

    eps_osc = np.array(m_sf['eps_osc'])
    eps_cat = np.array(m_sf['eps_cat'])
    eps_par = np.array(m_sf['eps_par'])
    ground_truth = np.array(m_sf['ground_truth'])
    confusion = np.array(m_sf['confusion_matrix'])
    threshold = m_sf['threshold']
    accuracy = m_sf['accuracy']

    enh_names = m_tp['enhancement_names']
    log10_vals = np.array(m_tp['log10_values'])
    cumulative = np.array(m_tp['cumulative_log10'])

    fig = plt.figure(figsize=(20, 5), facecolor='white')
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    # ── (A) Triple convergence for 20 scenarios ──────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    # Show first 20 scenarios, sorted by ground truth
    idx_sorted = np.argsort(~ground_truth)[:20]  # safe first
    x = np.arange(20)
    ax1.plot(x, eps_osc[idx_sorted], 'o-', color=C_TEAL, markersize=4,
             linewidth=1.0, label=r'$\varepsilon_{osc}$', alpha=0.8)
    ax1.plot(x, eps_cat[idx_sorted], 's-', color=C_GOLD, markersize=4,
             linewidth=1.0, label=r'$\varepsilon_{cat}$', alpha=0.8)
    ax1.plot(x, eps_par[idx_sorted], '^-', color=C_CORAL, markersize=4,
             linewidth=1.0, label=r'$\varepsilon_{par}$', alpha=0.8)
    ax1.axhline(threshold, color='gray', linewidth=1.0, linestyle='--',
                label=f'Threshold ({threshold})')
    # Shade safe/unsafe regions
    n_safe_shown = sum(ground_truth[idx_sorted[:20]])
    ax1.axvspan(-0.5, n_safe_shown - 0.5, alpha=0.05, color=C_TEAL)
    ax1.axvspan(n_safe_shown - 0.5, 19.5, alpha=0.05, color=C_CORAL)
    _style_axis(ax1, '(A) Triple Convergence\n(20 scenarios)',
                'Scenario', 'Gap magnitude')
    ax1.legend(fontsize=6, loc='upper left')

    # ── (B) Confusion matrix heatmap ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(confusion, cmap='Blues', interpolation='nearest', aspect='equal')
    for (ii, jj), val in np.ndenumerate(confusion):
        ax2.text(jj, ii, str(int(val)), ha='center', va='center',
                 fontsize=14, fontweight='bold',
                 color='white' if val > confusion.max() * 0.5 else C_DARK)
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Predicted\nUnsafe', 'Predicted\nSafe'], fontsize=8)
    ax2.set_yticklabels(['Actual\nUnsafe', 'Actual\nSafe'], fontsize=8)
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    ax2.set_title(f'(B) Sufficiency Classification\nAccuracy: {accuracy*100:.1f}%',
                  fontsize=10, fontweight='bold', color=C_DARK)

    # ── (C) 3D surface: enhancement vs (N_osc, T_int) ────────────────────
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    N_osc_range = np.linspace(5, 50, 20)
    T_int_range = np.logspace(-3, 3, 20)  # 1 ms to 1000 s
    N_grid, T_grid = np.meshgrid(N_osc_range, T_int_range)
    # Enhancement model: log10(E) = a * N_osc + b * log10(T_int) + c
    # Based on harmonic coincidence (scales with N) and continuous refinement (scales with T)
    log10_E_surface = (0.5 * N_grid +
                       21.7 * np.log10(T_grid + 1e-10) / np.log10(1e43) * 43.4 +
                       3.5 + 5.0 + 66.0)
    surf = ax3.plot_surface(N_grid, np.log10(T_grid), log10_E_surface,
                            cmap='viridis', alpha=0.85, edgecolor='none')
    _style_3d(ax3, '(C) Enhancement Surface',
              r'$N_{osc}$', r'$\log_{10} T_{int}$ (s)', r'$\log_{10} E$')

    # ── (D) Cumulative enhancement (stacked bar) ─────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    short_names = ['Ternary', 'Multi-modal', 'Harmonic', 'Poincare', 'Continuous']
    bar_colors = [C_TEAL, C_GOLD, C_CORAL, C_CYAN, '#6C71C4']
    # Cumulative: stacked bars
    bottoms = np.zeros(5)
    bottoms[1:] = cumulative[:-1]
    ax4.bar(range(5), log10_vals, bottom=bottoms, color=bar_colors,
            edgecolor='white', linewidth=0.5)
    # Cumulative line
    ax4.plot(range(5), cumulative, 'ko-', markersize=6, linewidth=2, zorder=5)
    for i, (cum_val, name) in enumerate(zip(cumulative, short_names)):
        ax4.text(i, cum_val + 2, f'{cum_val:.1f}', ha='center', va='bottom',
                 fontsize=7, fontweight='bold', color=C_DARK)
    ax4.set_xticks(range(5))
    ax4.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
    _style_axis(ax4, f'(D) Cumulative Enhancement\n'
                f'Total: $10^{{{m_tp["log10_total"]:.1f}}}$',
                'Mechanism', r'Cumulative $\log_{10} E$')
    ax4.axhline(100, color=C_CORAL, linewidth=1.0, linestyle='--', alpha=0.7,
                label=r'$10^{100}$ threshold')
    ax4.legend(fontsize=7)

    fig.savefig(os.path.join(FIGDIR, 'panel_3_sufficiency.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved panel_3_sufficiency.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Panel 4: Positioning & Detection
# ═══════════════════════════════════════════════════════════════════════════════

def panel_4(results_dict):
    """Panel 4: Positioning & Detection
    (A) Recovered vs true position (scatter with y=x)
    (B) Positioning error distribution (histogram)
    (C) 3D: atmospheric S-entropy field with vehicle perturbation
    (D) Detection range: perturbation magnitude vs distance
    """
    m_gps = results_dict['gps_free_positioning']['metrics']
    m_vpd = results_dict['vehicle_perturbation_detection']['metrics']

    positions = np.array(m_gps['positions'])
    recovered = np.array(m_gps['recovered_positions'])
    errors = np.array(m_gps['errors'])
    s_points = np.array(m_gps['s_points'])

    distances = np.array(m_vpd['distances'])
    pert_mag = np.array(m_vpd['perturbation_magnitudes'])
    detected = np.array(m_vpd['detected'])
    threshold_det = m_vpd['threshold']
    S_bg = np.array(m_vpd['S_background'])
    S_pert = np.array(m_vpd['S_perturbed_all'])

    fig = plt.figure(figsize=(20, 5), facecolor='white')
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    # ── (A) Recovered vs true position (x-coordinate) ────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(positions[:, 0], recovered[:, 0], c=C_TEAL, s=15,
                edgecolors='white', linewidth=0.3, alpha=0.7, label='x-coord')
    ax1.scatter(positions[:, 1], recovered[:, 1], c=C_GOLD, s=15,
                edgecolors='white', linewidth=0.3, alpha=0.7, label='y-coord')
    lims = [positions.min() - 50, positions.max() + 50]
    ax1.plot(lims, lims, '--', color=C_CORAL, linewidth=1.5, alpha=0.7, label='y = x')
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    _style_axis(ax1, '(A) GPS-Free Positioning\nRecovered vs True',
                'True position (m)', 'Recovered position (m)')
    ax1.legend(fontsize=7)

    # ── (B) Positioning error distribution ────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(errors, bins=30, color=C_GOLD, edgecolor='white', linewidth=0.3,
             alpha=0.85, density=True)
    mean_err = errors.mean()
    ax2.axvline(mean_err, color=C_CORAL, linewidth=1.5, linestyle='--',
                label=f'Mean = {mean_err:.1f} m')
    _style_axis(ax2, '(B) Positioning Error Distribution',
                'Error (m)', 'Density')
    ax2.legend(fontsize=7)

    # ── (C) 3D: S-entropy field with perturbation ────────────────────────
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    # Background: sample of S-points from positioning data
    ax3.scatter(s_points[:, 0], s_points[:, 1], s_points[:, 2],
                c=C_CYAN, s=8, alpha=0.3, label='Background')
    # Perturbed points
    for i in range(len(distances)):
        color = C_CORAL if detected[i] else C_GOLD
        marker = 'D' if detected[i] else 'o'
        ax3.scatter(S_pert[i, 0], S_pert[i, 1], S_pert[i, 2],
                    c=color, s=80, marker=marker, edgecolors='white',
                    linewidth=0.5, zorder=10,
                    label=f'{distances[i]:.0f}m' if i < 4 else None)
    # Background reference
    ax3.scatter(S_bg[0], S_bg[1], S_bg[2], c=C_DARK, s=120, marker='*',
                edgecolors='white', linewidth=1, zorder=15, label='Background ref')
    _style_3d(ax3, '(C) S-Entropy Field\nwith Vehicle Perturbation',
              r'$S_T$', r'$S_P$', r'$S_H$')
    ax3.legend(fontsize=5, loc='upper left', ncol=2)

    # ── (D) Detection range ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    det_colors = [C_TEAL if d else C_CORAL for d in detected]
    ax4.scatter(distances, pert_mag, c=det_colors, s=80, edgecolors='white',
                linewidth=0.5, zorder=5)
    # Fit decay curve
    r_smooth = np.linspace(distances.min(), distances.max(), 200)
    # Model: perturbation ~ A/r
    A_fit = pert_mag[0] * distances[0]
    pert_smooth = A_fit / r_smooth
    ax4.plot(r_smooth, pert_smooth, '-', color=C_GOLD, linewidth=1.5,
             alpha=0.7, label=r'$\propto 1/r$ decay')
    ax4.axhline(threshold_det, color=C_CORAL, linewidth=1.5, linestyle='--',
                alpha=0.7, label=f'Threshold ({threshold_det:.3f})')
    # Label detected / undetected
    for i, (d, pm) in enumerate(zip(distances, pert_mag)):
        status = 'det' if detected[i] else 'miss'
        ax4.annotate(f'{d:.0f}m\n({status})', (d, pm),
                     fontsize=6, ha='center', va='bottom')
    _style_axis(ax4, '(D) Detection Range\nPerturbation vs Distance',
                'Distance (m)', 'Perturbation magnitude')
    ax4.legend(fontsize=7)

    fig.savefig(os.path.join(FIGDIR, 'panel_4_positioning.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved panel_4_positioning.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Panel 5: Duality & Commutation
# ═══════════════════════════════════════════════════════════════════════════════

def panel_5(results_dict):
    """Panel 5: Duality & Commutation
    (A) R = omega/(2pi) vs dM/dt (scatter, y=x)
    (B) Total processing vs number of oscillators (linear)
    (C) 3D: pre vs post measurement physical states
    (D) Perturbation magnitude across 1000 measurements
    """
    m_opd = results_dict['oscillator_processor_duality']['metrics']
    m_cpc = results_dict['categorical_physical_commutation']['metrics']

    freqs = np.array(m_opd['frequencies'])
    R_vals = np.array(m_opd['R_values'])
    dMdt_per_state = np.array(m_opd['dMdt_per_state'])
    names = m_opd['oscillator_names']
    total_proc = m_opd['total_processing']
    full_freqs = np.array(m_opd['full_vehicle_freqs'])

    pre_states = np.array(m_cpc['pre_measurement_states'])
    post_states = np.array(m_cpc['post_measurement_states'])
    max_pert = m_cpc['max_perturbation']
    pert_per_meas = np.array(m_cpc['perturbation_per_measurement'])
    n_meas = m_cpc['n_measurements']

    fig = plt.figure(figsize=(20, 5), facecolor='white')
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    # ── (A) Oscillator-processor duality: R vs dM/dt per state ────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(R_vals, dMdt_per_state, c=C_TEAL, s=60, edgecolors='white',
                linewidth=0.5, zorder=5, label='Oscillators')
    lims = [R_vals.min() * 0.5, R_vals.max() * 2]
    ax1.plot(lims, lims, '--', color=C_CORAL, linewidth=1.5, alpha=0.7,
             label=r'$R = \omega/2\pi = dM/dt \cdot M^{-1}$')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    for i, name in enumerate(names):
        ax1.annotate(name[:4], (R_vals[i], dMdt_per_state[i]),
                     fontsize=5, ha='left', va='bottom', color=C_DARK)
    _style_axis(ax1, r'(A) Oscillator-Processor Duality' '\n' r'$R = \omega / 2\pi$ vs $dM/dt \cdot M^{-1}$',
                r'Processing rate $R$ (Hz)', r'Counting rate $dM/dt \cdot M^{-1}$')
    ax1.legend(fontsize=6)

    # ── (B) Total processing vs number of oscillators ─────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    # Build cumulative: add oscillators one by one (sorted by frequency)
    sorted_freqs = np.sort(full_freqs)
    n_osc_range = np.arange(1, len(sorted_freqs) + 1)
    cumulative_proc = np.cumsum(sorted_freqs)
    ax2.plot(n_osc_range, cumulative_proc, 'o-', color=C_TEAL, markersize=4,
             linewidth=1.5)
    ax2.set_yscale('log')
    ax2.axhline(total_proc, color=C_GOLD, linewidth=1.0, linestyle='--',
                label=f'10 oscillators: {total_proc:.2e}')
    _style_axis(ax2, '(B) Total Processing Rate\nvs Oscillator Count',
                'Number of oscillators', 'Total R (Hz)')
    ax2.legend(fontsize=7)

    # ── (C) 3D: pre vs post measurement states ───────────────────────────
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    # Plot first 200 points: (pre_x, pre_y, pre_z) and (post_x, post_y, post_z)
    n_show = min(200, n_meas)
    ax3.scatter(pre_states[:n_show, 0], pre_states[:n_show, 1],
                pre_states[:n_show, 2], c=C_TEAL, s=10, alpha=0.5,
                label='Pre-measurement')
    ax3.scatter(post_states[:n_show, 0], post_states[:n_show, 1],
                post_states[:n_show, 2], c=C_CORAL, s=10, alpha=0.5,
                marker='x', label='Post-measurement')
    # Diagonal plane: pre = post
    plane_range = np.linspace(-100, 100, 5)
    Xp, Yp = np.meshgrid(plane_range, plane_range)
    Zp = Xp  # z = x on the diagonal
    ax3.plot_surface(Xp, Yp, Zp, alpha=0.05, color=C_GOLD)
    _style_3d(ax3, '(C) Pre vs Post Measurement\n(should overlay)',
              'x (m)', 'y (m)', 'z (m)')
    ax3.legend(fontsize=6, loc='upper left')

    # ── (D) Perturbation magnitudes ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.scatter(range(n_meas), pert_per_meas, c=C_TEAL, s=3, alpha=0.5)
    ax4.axhline(1e-10, color=C_CORAL, linewidth=1.5, linestyle='--',
                label=r'Threshold ($10^{-10}$)')
    ax4.set_ylim(-0.1 * max(pert_per_meas.max(), 1e-10),
                 max(pert_per_meas.max(), 1e-10) * 1.5 + 1e-11)
    _style_axis(ax4, f'(D) Categorical-Physical Commutation\n'
                f'Max perturbation: {max_pert:.1e}',
                'Measurement index', 'Perturbation magnitude')
    ax4.legend(fontsize=7)
    # Annotate
    ax4.annotate(r'$[\hat{O}_{cat}, \hat{O}_{phys}] = 0$',
                 xy=(n_meas * 0.5, max_pert * 0.5 + 1e-11),
                 fontsize=10, ha='center', color=C_DARK, fontstyle='italic')

    fig.savefig(os.path.join(FIGDIR, 'panel_5_duality.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved panel_5_duality.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 78)
    print("Counting Loops — Panel Generation")
    print("=" * 78)

    # ── Step 1: Run validation ────────────────────────────────────────────
    print("\n[1/2] Running validation experiments...")
    results_json_path = os.path.join(FIGDIR, 'results.json')
    results = validation.run_all(save_json=True, json_path=results_json_path)

    # Build dict indexed by experiment name
    results_dict = {}
    for r in results:
        results_dict[r.name] = r.to_dict()

    # ── Step 2: Generate panels ───────────────────────────────────────────
    print("\n[2/2] Generating panels...")
    panel_1(results_dict)
    panel_2(results_dict)
    panel_3(results_dict)
    panel_4(results_dict)
    panel_5(results_dict)

    print("\n" + "=" * 78)
    n_pass = sum(1 for r in results if r.passed)
    print(f"Done. {n_pass}/{len(results)} experiments passed.")
    print(f"Panels saved to: {FIGDIR}")
    print(f"Results saved to: {results_json_path}")
    print("=" * 78)


if __name__ == '__main__':
    main()
