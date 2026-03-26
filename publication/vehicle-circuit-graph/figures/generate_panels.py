"""
Generate 5 publication panels for the Vehicle Oscillatory Circuit Graph paper.
Each panel: 4 charts in a row, white background, 300 DPI, at least one 3D chart.

Panel 1: Graph Structure
Panel 2: Kirchhoff Validation
Panel 3: Fuzzy State & Trajectory Completion
Panel 4: Backward Trajectory & Fault Detection
Panel 5: Transport & Propagation
"""

import sys
import os
import types
import json

# ── Mock verum_learn to avoid torch dependency on top-level import ────────────
pkg = types.ModuleType('verum_learn')
pkg.__path__ = ['verum_learn']
pkg.__package__ = 'verum_learn'
sys.modules['verum_learn'] = pkg

cg_pkg = types.ModuleType('verum_learn.circuit_graph')
cg_pkg.__path__ = ['verum_learn/circuit_graph']
cg_pkg.__package__ = 'verum_learn.circuit_graph'
sys.modules['verum_learn.circuit_graph'] = cg_pkg

# ── Import validation module via importlib to avoid top-level package issues ──
import importlib.util

# Resolve path to validation.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
VALIDATION_PATH = os.path.join(PROJECT_ROOT, 'verum-learn', 'verum_learn',
                               'circuit_graph', 'validation.py')

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

# Additional palette for multi-series plots
C_PALETTE = [C_TEAL, C_GOLD, C_CORAL, C_CYAN, '#6C71C4', '#859900', '#CB4B16',
             '#D33682', '#268BD2', '#DC322F', '#B58900', '#2AA198',
             '#93a1a1', '#586e75', '#002b36']

# Category colors
CAT_COLORS = {
    'powertrain': C_TEAL,
    'chassis': C_GOLD,
    'electrical': C_CYAN,
    'thermal': C_CORAL,
}


def _style_axis(ax, title, xlabel, ylabel, fontsize_title=10, fontsize_label=9):
    """Apply consistent styling."""
    ax.set_title(title, fontsize=fontsize_title, fontweight='bold', color=C_DARK)
    ax.set_xlabel(xlabel, fontsize=fontsize_label, color=C_DARK)
    ax.set_ylabel(ylabel, fontsize=fontsize_label, color=C_DARK)
    ax.tick_params(labelsize=8, colors=C_DARK)
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_color('#cccccc')


# ═══════════════════════════════════════════════════════════════════════════════
# Panel 1: Graph Structure
# ═══════════════════════════════════════════════════════════════════════════════

def panel_1(results_dict):
    """Panel 1: Graph Structure
    (A) Adjacency matrix heatmap (15x15, colored by conductance)
    (B) Laplacian eigenvalue spectrum
    (C) 3D node layout (frequency, categorical_depth, coupling_degree)
    (D) Degree distribution
    """
    m = results_dict['graph_construction']['metrics']
    A = np.array(m['adjacency_matrix'])
    eigenvalues = np.array(m['laplacian_eigenvalues'])
    names = m['node_names']
    freqs = np.array(m['node_frequencies'])
    depths = np.array(m['node_depths'])
    degrees = np.array(m['node_degrees'])
    categories = m['node_categories']

    fig = plt.figure(figsize=(20, 5), facecolor='white')
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    # ── (A) Adjacency matrix heatmap ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(A, cmap='YlOrRd', interpolation='nearest', aspect='equal')
    ax1.set_xticks(range(len(names)))
    ax1.set_yticks(range(len(names)))
    ax1.set_xticklabels([n[:4] for n in names], rotation=90, fontsize=6)
    ax1.set_yticklabels([n[:4] for n in names], fontsize=6)
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Conductance', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax1.set_title('(A) Adjacency Matrix\n(conductance)', fontsize=10,
                  fontweight='bold', color=C_DARK)

    # ── (B) Laplacian eigenvalue spectrum ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    colors_eig = [C_CORAL if abs(e) < 1e-10 else C_TEAL for e in eigenvalues]
    bars = ax2.bar(range(len(eigenvalues)), eigenvalues, color=colors_eig,
                   edgecolor='white', linewidth=0.5)
    _style_axis(ax2, '(B) Laplacian Eigenvalue Spectrum', 'Eigenvalue index',
                'Eigenvalue')
    ax2.axhline(y=0, color=C_CORAL, linewidth=1.0, linestyle='--', alpha=0.6)
    ax2.annotate(r'$\lambda_1 = 0$ (connected)', xy=(0, eigenvalues[0]),
                xytext=(2, eigenvalues[3] * 0.5),
                arrowprops=dict(arrowstyle='->', color=C_CORAL),
                fontsize=7, color=C_CORAL)

    # ── (C) 3D node layout ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    node_colors = [CAT_COLORS.get(c, C_DARK) for c in categories]
    # Coupling degree = sum of conductances
    coupling_degree = A.sum(axis=1)
    sc = ax3.scatter(freqs, depths, coupling_degree, c=node_colors,
                     s=80, edgecolors='white', linewidth=0.5, depthshade=True)
    for i, name in enumerate(names):
        ax3.text(freqs[i], depths[i], coupling_degree[i], f' {name[:5]}',
                fontsize=5, color=C_DARK)
    ax3.set_xlabel('Frequency (Hz)', fontsize=7, labelpad=5)
    ax3.set_ylabel('Categorical Depth', fontsize=7, labelpad=5)
    ax3.set_zlabel('Coupling Degree', fontsize=7, labelpad=5)
    ax3.set_title('(C) 3D Node Layout', fontsize=10, fontweight='bold',
                  color=C_DARK)
    ax3.tick_params(labelsize=6)
    ax3.set_facecolor('white')
    ax3.xaxis.pane.fill = False
    ax3.yaxis.pane.fill = False
    ax3.zaxis.pane.fill = False

    # ── (D) Degree distribution ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    bar_colors = [CAT_COLORS.get(c, C_DARK) for c in categories]
    ax4.barh(range(len(names)), degrees, color=bar_colors,
             edgecolor='white', linewidth=0.5)
    ax4.set_yticks(range(len(names)))
    ax4.set_yticklabels(names, fontsize=7)
    _style_axis(ax4, '(D) Node Degree Distribution', 'Degree (connections)', '')
    ax4.invert_yaxis()

    fig.savefig(os.path.join(FIGDIR, 'panel_1_graph_structure.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved panel_1_graph_structure.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Panel 2: Kirchhoff Validation
# ═══════════════════════════════════════════════════════════════════════════════

def panel_2(results_dict):
    """Panel 2: Kirchhoff Validation
    (A) KCL: current balance at each node
    (B) KVL: voltage around each cycle
    (C) 3D surface: potentials vs (node, driving_amplitude)
    (D) Steady-state node potentials
    """
    m_kcl = results_dict['kirchhoff_current_law']['metrics']
    m_kvl = results_dict['kirchhoff_voltage_law']['metrics']

    names = validation.NODE_NAMES
    categories = [validation.NODE_CATEGORIES[n] for n in names]

    imbalance = np.array(m_kcl['node_imbalance'])
    potentials = np.array(m_kcl['node_potentials'])
    drive_amps = np.array(m_kcl['drive_amplitudes'])
    pot_surface = np.array(m_kcl['potential_surface'])
    cycle_voltages = np.array(m_kvl['cycle_voltages'])

    fig = plt.figure(figsize=(20, 5), facecolor='white')
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    # ── (A) KCL: current balance ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(range(len(names)), imbalance, color=C_TEAL, edgecolor='white',
            linewidth=0.5)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels([n[:4] for n in names], rotation=90, fontsize=6)
    ax1.set_yscale('log')
    _style_axis(ax1, r'(A) KCL: Current Imbalance', 'Node',
                r'$|\sum I_{ij} - I_{ext}|$')
    ax1.axhline(y=1e-10, color=C_CORAL, linewidth=1.0, linestyle='--',
                label='Threshold')
    ax1.legend(fontsize=7)

    # ── (B) KVL: voltage around cycles ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    n_cycles = len(cycle_voltages)
    ax2.bar(range(n_cycles), np.abs(cycle_voltages), color=C_GOLD,
            edgecolor='white', linewidth=0.5)
    _style_axis(ax2, r'(B) KVL: Cycle Voltage Sum', 'Cycle index',
                r'$|\sum_{\mathrm{loop}} \Delta V|$')
    ax2.set_yscale('log')
    ax2.axhline(y=1e-10, color=C_CORAL, linewidth=1.0, linestyle='--',
                label='Threshold')
    ax2.legend(fontsize=7)

    # ── (C) 3D surface: potentials vs driving amplitude ──────────────────
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    X, Y = np.meshgrid(range(len(names)), drive_amps)
    Z = pot_surface.T  # shape: (n_amps, n_nodes)
    surf = ax3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.85,
                            edgecolor='none')
    ax3.set_xlabel('Node index', fontsize=7, labelpad=5)
    ax3.set_ylabel('Drive amplitude', fontsize=7, labelpad=5)
    ax3.set_zlabel('Potential', fontsize=7, labelpad=5)
    ax3.set_title('(C) Potentials vs Driving', fontsize=10, fontweight='bold',
                  color=C_DARK)
    ax3.tick_params(labelsize=6)
    ax3.set_facecolor('white')
    ax3.xaxis.pane.fill = False
    ax3.yaxis.pane.fill = False
    ax3.zaxis.pane.fill = False

    # ── (D) Steady-state potentials ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    bar_colors = [CAT_COLORS.get(c, C_DARK) for c in categories]
    ax4.bar(range(len(names)), potentials, color=bar_colors,
            edgecolor='white', linewidth=0.5)
    ax4.set_xticks(range(len(names)))
    ax4.set_xticklabels([n[:4] for n in names], rotation=90, fontsize=6)
    _style_axis(ax4, '(D) Steady-State Potentials', 'Node', 'Potential')
    ax4.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

    fig.savefig(os.path.join(FIGDIR, 'panel_2_kirchhoff.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved panel_2_kirchhoff.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Panel 3: Fuzzy State & Trajectory Completion
# ═══════════════════════════════════════════════════════════════════════════════

def panel_3(results_dict):
    """Panel 3: Fuzzy State & Trajectory Completion
    (A) Fuzzy width narrowing for 5 representative nodes
    (B) Hausdorff distance vs iteration (log scale)
    (C) 3D scatter: ground truth vs reconstructed vs node_index
    (D) Reconstruction error per node
    """
    m_fuzzy = results_dict['fuzzy_state_propagation']['metrics']
    m_traj = results_dict['trajectory_completion']['metrics']

    names = validation.NODE_NAMES

    rep_names = m_fuzzy['representative_nodes']
    rep_widths = m_fuzzy['representative_widths']
    hausdorff = np.array(m_fuzzy['hausdorff_history'])

    truth = np.array(m_traj['ground_truth'])
    recon = np.array(m_traj['reconstructed'])
    errors = np.array(m_traj['normalized_errors'])
    is_observed = np.array(m_traj['is_observed'])

    fig = plt.figure(figsize=(20, 5), facecolor='white')
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    # ── (A) Fuzzy width narrowing ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for idx, rname in enumerate(rep_names):
        widths = rep_widths[rname]
        ax1.plot(range(len(widths)), widths, color=C_PALETTE[idx],
                linewidth=1.5, label=rname[:8])
    _style_axis(ax1, '(A) Fuzzy Width Narrowing', 'Iteration',
                'Fuzzy interval width')
    ax1.legend(fontsize=6, loc='upper right')
    ax1.set_yscale('log')

    # ── (B) Hausdorff distance ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(range(len(hausdorff)), hausdorff, color=C_TEAL, linewidth=2.0,
                marker='o', markersize=3)
    _style_axis(ax2, '(B) Hausdorff Distance', 'Iteration',
                'Hausdorff distance')
    ax2.axhline(y=1e-8, color=C_CORAL, linewidth=1.0, linestyle='--',
                label='Tolerance')
    ax2.legend(fontsize=7)

    # ── (C) 3D scatter: truth vs reconstructed ───────────────────────────
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    node_idx = np.arange(len(names))
    colors = [C_TEAL if obs else C_CORAL for obs in is_observed]
    ax3.scatter(truth, recon, node_idx, c=colors, s=60,
               edgecolors='white', linewidth=0.5, depthshade=True)
    # Reference line (truth = reconstructed)
    t_range = np.linspace(min(truth.min(), recon.min()),
                          max(truth.max(), recon.max()), 50)
    ax3.plot(t_range, t_range, np.full_like(t_range, len(names) / 2),
             color=C_GOLD, linewidth=1.5, linestyle='--', alpha=0.7)
    ax3.set_xlabel('Ground Truth', fontsize=7, labelpad=5)
    ax3.set_ylabel('Reconstructed', fontsize=7, labelpad=5)
    ax3.set_zlabel('Node Index', fontsize=7, labelpad=5)
    ax3.set_title('(C) Truth vs Reconstructed', fontsize=10,
                  fontweight='bold', color=C_DARK)
    ax3.tick_params(labelsize=6)
    ax3.set_facecolor('white')
    ax3.xaxis.pane.fill = False
    ax3.yaxis.pane.fill = False
    ax3.zaxis.pane.fill = False

    # ── (D) Reconstruction error per node ────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    bar_colors = [C_TEAL if obs else C_CORAL for obs in is_observed]
    ax4.bar(range(len(names)), errors, color=bar_colors,
            edgecolor='white', linewidth=0.5)
    ax4.set_xticks(range(len(names)))
    ax4.set_xticklabels([n[:4] for n in names], rotation=90, fontsize=6)
    _style_axis(ax4, '(D) Reconstruction Error per Node', 'Node',
                'Normalized error')
    ax4.axhline(y=0.10, color=C_GOLD, linewidth=1.0, linestyle='--',
                label='10% threshold')
    # Legend for observed / inferred
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=C_TEAL, label='Observed'),
                       Patch(facecolor=C_CORAL, label='Inferred')]
    ax4.legend(handles=legend_elements, fontsize=7, loc='upper right')

    fig.savefig(os.path.join(FIGDIR, 'panel_3_fuzzy_completion.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved panel_3_fuzzy_completion.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Panel 4: Backward Trajectory & Fault Detection
# ═══════════════════════════════════════════════════════════════════════════════

def panel_4(results_dict):
    """Panel 4: Backward Trajectory & Fault Detection
    (A) MAP backward path through state space
    (B) Time-invariance: overlapping trajectories at t1, t2, t3
    (C) 3D surface: healthy vs faulty potentials
    (D) Fault detection: deviation per node
    """
    m_bt = results_dict['backward_trajectory_time_invariance']['metrics']
    m_fd = results_dict['fault_detection']['metrics']

    names = validation.NODE_NAMES

    traj_states = np.array(m_bt['traj_states'])  # (n_steps, 2) -> depth, freq
    depths_t1 = np.array(m_bt['depths_t1'])
    depths_t2 = np.array(m_bt['depths_t2'])
    depths_t3 = np.array(m_bt['depths_t3'])
    traj_names = m_bt['traj_node_names']

    healthy = np.array(m_fd['healthy_potentials'])
    faulty = np.array(m_fd['faulty_potentials'])
    deviation = np.array(m_fd['deviation'])
    detected = m_fd['detected_node_name']

    fig = plt.figure(figsize=(20, 5), facecolor='white')
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    # ── (A) MAP backward path ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(traj_states[:, 0], traj_states[:, 1], color=C_TEAL, linewidth=2.0,
             marker='o', markersize=6, markerfacecolor=C_GOLD,
             markeredgecolor='white', markeredgewidth=0.5)
    for i, name in enumerate(traj_names):
        ax1.annotate(name[:6], (traj_states[i, 0], traj_states[i, 1]),
                    fontsize=6, xytext=(5, 5), textcoords='offset points',
                    color=C_DARK)
    _style_axis(ax1, '(A) MAP Backward Trajectory', 'Categorical Depth',
                'Frequency (Hz)')

    # ── (B) Time-invariance ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    steps = range(len(depths_t1))
    ax2.plot(steps, depths_t1, color=C_TEAL, linewidth=2.5, label='t1',
             marker='o', markersize=5)
    ax2.plot(steps, depths_t2, color=C_GOLD, linewidth=1.8, label='t2',
             marker='s', markersize=4, linestyle='--')
    ax2.plot(steps, depths_t3, color=C_CORAL, linewidth=1.2, label='t3',
             marker='^', markersize=4, linestyle=':')
    _style_axis(ax2, '(B) Time-Invariance of Trajectory', 'Step',
                'Categorical Depth')
    ax2.legend(fontsize=8)

    # ── (C) 3D: healthy vs faulty potentials ─────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    node_idx = np.arange(len(names))
    # Two "slices": healthy (y=0) and faulty (y=1)
    ax3.bar3d(node_idx - 0.2, np.zeros(len(names)), np.zeros(len(names)),
              0.4, 0.4, healthy, color=C_TEAL, alpha=0.8, label='Healthy')
    ax3.bar3d(node_idx + 0.2, np.ones(len(names)) * 0.6, np.zeros(len(names)),
              0.4, 0.4, faulty, color=C_CORAL, alpha=0.8, label='Faulty')
    ax3.set_xlabel('Node index', fontsize=7, labelpad=5)
    ax3.set_ylabel('Condition', fontsize=7, labelpad=5)
    ax3.set_zlabel('Potential', fontsize=7, labelpad=5)
    ax3.set_title('(C) Healthy vs Faulty Potentials', fontsize=10,
                  fontweight='bold', color=C_DARK)
    ax3.tick_params(labelsize=6)
    ax3.set_facecolor('white')
    ax3.xaxis.pane.fill = False
    ax3.yaxis.pane.fill = False
    ax3.zaxis.pane.fill = False

    # ── (D) Fault detection deviation ────────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    bar_colors = [C_CORAL if names[i] == detected else C_TEAL
                  for i in range(len(names))]
    ax4.bar(range(len(names)), deviation, color=bar_colors,
            edgecolor='white', linewidth=0.5)
    ax4.set_xticks(range(len(names)))
    ax4.set_xticklabels([n[:4] for n in names], rotation=90, fontsize=6)
    _style_axis(ax4, '(D) Fault Detection: Deviation', 'Node',
                '|Healthy model - Faulty actual|')
    # Highlight detected node
    det_idx = names.index(detected)
    ax4.annotate(f'Detected: {detected}', xy=(det_idx, deviation[det_idx]),
                xytext=(det_idx + 2, deviation[det_idx] * 1.2),
                arrowprops=dict(arrowstyle='->', color=C_CORAL),
                fontsize=7, color=C_CORAL, fontweight='bold')

    fig.savefig(os.path.join(FIGDIR, 'panel_4_trajectory_fault.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved panel_4_trajectory_fault.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Panel 5: Transport & Propagation
# ═══════════════════════════════════════════════════════════════════════════════

def panel_5(results_dict):
    """Panel 5: Transport & Propagation
    (A) Transport formula: computed vs analytical for 4 types
    (B) Signal propagation: time to reach each node from engine
    (C) 3D: propagation wavefront at 3 time snapshots
    (D) Contraction mapping convergence
    """
    m_tr = results_dict['transport_formula_consistency']['metrics']
    m_sp = results_dict['signal_propagation_velocity']['metrics']
    m_cm = results_dict['contraction_mapping']['metrics']

    names = validation.NODE_NAMES

    transport_types = m_tr['transport_types']
    computed = np.array(m_tr['computed'])
    analytical = np.array(m_tr['analytical'])

    prop_times = np.array(m_sp['propagation_times'])
    velocities_phys = np.array(m_sp['velocities_physical'])
    wavefront_snaps = m_sp['wavefront_snapshots']
    t_snaps = m_sp['t_snapshots']
    node_freqs = np.array(m_sp['node_frequencies'])

    errors_1 = np.array(m_cm['errors_1'])
    errors_2 = np.array(m_cm['errors_2'])
    lip_consts = np.array(m_cm['lipschitz_constants'])

    fig = plt.figure(figsize=(20, 5), facecolor='white')
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    # ── (A) Transport formula: computed vs analytical ────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    # Normalize to log scale for comparison
    log_comp = np.log10(np.abs(computed) + 1e-50)
    log_anal = np.log10(np.abs(analytical) + 1e-50)
    scatter_colors = [C_TEAL, C_GOLD, C_CORAL, C_CYAN]
    for i, ttype in enumerate(transport_types):
        ax1.scatter(log_anal[i], log_comp[i], c=scatter_colors[i], s=100,
                   edgecolors='white', linewidth=0.5, label=ttype, zorder=5)
    # y = x reference line
    all_vals = np.concatenate([log_comp, log_anal])
    vmin, vmax = all_vals.min() - 1, all_vals.max() + 1
    ax1.plot([vmin, vmax], [vmin, vmax], color='gray', linewidth=1.0,
             linestyle='--', alpha=0.6)
    ax1.set_xlim(vmin, vmax)
    ax1.set_ylim(vmin, vmax)
    _style_axis(ax1, r'(A) Transport: Computed vs Analytical',
                r'$\log_{10}$(Analytical)', r'$\log_{10}$(Computed)')
    ax1.legend(fontsize=7, loc='upper left')
    ax1.set_aspect('equal')

    # ── (B) Signal propagation times ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    # Sort by propagation time (exclude source = 0)
    valid_mask = (prop_times > 0) & (prop_times < 1e20)
    sorted_idx = np.argsort(prop_times)
    sorted_idx = [i for i in sorted_idx if valid_mask[i]]
    sorted_times = prop_times[sorted_idx]
    sorted_names = [names[i][:6] for i in sorted_idx]

    ax2.barh(range(len(sorted_names)), sorted_times, color=C_CYAN,
             edgecolor='white', linewidth=0.5)
    ax2.set_yticks(range(len(sorted_names)))
    ax2.set_yticklabels(sorted_names, fontsize=7)
    _style_axis(ax2, '(B) Propagation Time from Engine', 'Time (abstract units)',
                '')
    ax2.invert_yaxis()

    # ── (C) 3D propagation wavefront ────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    depths = np.array([validation._categorical_depth(n) for n in names])
    snap_colors = [C_TEAL, C_GOLD, C_CORAL]
    snap_labels = [f't={t:.3f}' for t in t_snaps]
    for si, (snap, col, lab) in enumerate(zip(wavefront_snaps, snap_colors,
                                              snap_labels)):
        if len(snap) > 0:
            snap_arr = np.array(snap)
            ax3.scatter(node_freqs[snap_arr], depths[snap_arr],
                       np.full(len(snap), si),
                       c=col, s=70, edgecolors='white', linewidth=0.5,
                       label=lab, depthshade=True)
            for ni in snap:
                ax3.text(node_freqs[ni], depths[ni], si,
                        f' {names[ni][:4]}', fontsize=5, color=C_DARK)
    ax3.set_xlabel('Frequency (Hz)', fontsize=7, labelpad=5)
    ax3.set_ylabel('Categorical Depth', fontsize=7, labelpad=5)
    ax3.set_zlabel('Time Snapshot', fontsize=7, labelpad=5)
    ax3.set_title('(C) Signal Wavefront', fontsize=10, fontweight='bold',
                  color=C_DARK)
    ax3.tick_params(labelsize=6)
    ax3.legend(fontsize=6, loc='upper left')
    ax3.set_facecolor('white')
    ax3.xaxis.pane.fill = False
    ax3.yaxis.pane.fill = False
    ax3.zaxis.pane.fill = False

    # ── (D) Contraction mapping convergence ──────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.semilogy(range(len(errors_1)), errors_1, color=C_TEAL, linewidth=2.0,
                label='Init 1 (zeros)', marker='o', markersize=3)
    ax4.semilogy(range(len(errors_2)), errors_2, color=C_GOLD, linewidth=2.0,
                label='Init 2 (random)', marker='s', markersize=3)
    _style_axis(ax4, '(D) Contraction: Convergence', 'Iteration',
                r'$\|x - x^*\|$')
    ax4.legend(fontsize=7)

    # Inset: Lipschitz constants
    if len(lip_consts) > 2:
        ax_inset = ax4.inset_axes([0.55, 0.55, 0.40, 0.35])
        ax_inset.plot(range(len(lip_consts)), lip_consts, color=C_CORAL,
                     linewidth=1.5)
        ax_inset.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8)
        ax_inset.set_title(r'$\lambda$ (Lipschitz)', fontsize=6)
        ax_inset.tick_params(labelsize=5)
        ax_inset.set_facecolor('white')

    fig.savefig(os.path.join(FIGDIR, 'panel_5_transport_propagation.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved panel_5_transport_propagation.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 78)
    print("Vehicle Oscillatory Circuit Graph — Panel Generation")
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
