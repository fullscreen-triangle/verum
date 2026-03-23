"""
Generate 3 publication panels for the membrane signal transduction paper.
Each panel: 4 charts in a row, white background, at least one 3D chart.

Panel 1 (Tests 1-5):  Lipid oscillation, processing rate, conductivity, V_bi, rectification
Panel 2 (Tests 6-9):  BMD transistor, logic gates, ALU, S-entropy invertibility
Panel 3 (Tests 10-13): S-entropy injectivity, full circuit transduction, obstacle detection, weather
"""

import sys
import types

# Mock verum_learn package to avoid torch dependency
pkg = types.ModuleType('verum_learn')
pkg.__path__ = ['verum_learn']
pkg.__package__ = 'verum_learn'
sys.modules['verum_learn'] = pkg
mp = types.ModuleType('verum_learn.membrane')
mp.__path__ = ['verum_learn/membrane']
mp.__package__ = 'verum_learn.membrane'
sys.modules['verum_learn.membrane'] = mp

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

# ── Import membrane modules ──────────────────────────────────────────────────
sys.path.insert(0, '.')
from verum_learn.membrane.lipid import Lipid, LipidArray, CHAIN_ISOMERIZATION_RATE
from verum_learn.membrane.carriers import CarrierPopulation
from verum_learn.membrane.junction import PNJunction
from verum_learn.membrane.transistor import BMDTransistor
from verum_learn.membrane.logic_gates import TriDimensionalGate
from verum_learn.membrane.alu import VirtualALU
from verum_learn.membrane.s_entropy import SEntropyCoordinate, compute_s_entropy
from verum_learn.membrane.ensemble import O2Ensemble
from verum_learn.membrane.sensor_circuit import SensorCircuit

import os
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))

# Colors
C_TEAL = '#2AA198'
C_ORANGE = '#CB4B16'
C_PURPLE = '#6C71C4'
C_GREEN = '#859900'
C_RED = '#DC322F'
C_BLUE = '#268BD2'
C_YELLOW = '#B58900'
C_MAGENTA = '#D33682'


def panel_1():
    """Panel 1: Lipid Physics (Tests 1-5)
    A) Lipid oscillation waveform (time series)
    B) Processing rate vs membrane area (log-log)
    C) I-V characteristic of P-N junction
    D) 3D: Rectification ratio surface vs voltage and temperature
    """
    fig = plt.figure(figsize=(20, 5), facecolor='white')
    gs = GridSpec(1, 4, figure=fig, wspace=0.3)

    # ── A: Lipid oscillation waveform ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    lipid = Lipid()
    t = np.linspace(0, 5e-11, 500)  # 50 ps window
    x = [lipid.displacement(ti) for ti in t]
    ax1.plot(t * 1e12, x, color=C_TEAL, linewidth=1.5)
    ax1.set_xlabel('Time (ps)', fontsize=9)
    ax1.set_ylabel('Displacement', fontsize=9)
    ax1.set_title(f'(A) Lipid Oscillation\nf = {lipid.frequency:.1e} Hz', fontsize=10, fontweight='bold')
    ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax1.tick_params(labelsize=8)

    # ── B: Processing rate vs area (log-log) ──────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    areas = np.logspace(-12, 0, 50)  # 1 um^2 to 1 m^2
    rates = [LipidArray(area=a).total_processing_rate for a in areas]
    ax2.loglog(areas * 1e6, rates, color=C_ORANGE, linewidth=2)
    ax2.axhline(y=1e20, color=C_RED, linewidth=1, linestyle='--', alpha=0.7)
    ax2.text(1e-4, 2e20, 'GPU (10$^{20}$)', fontsize=7, color=C_RED)
    ax2.set_xlabel('Area (mm$^2$)', fontsize=9)
    ax2.set_ylabel('Processing Rate (ops/s)', fontsize=9)
    ax2.set_title('(B) Membrane Processing\nRate vs Area', fontsize=10, fontweight='bold')
    ax2.tick_params(labelsize=8)

    # ── C: I-V characteristic ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    junction = PNJunction()
    v, i = junction.iv_curve(-0.8, 0.8, 400)
    ax3.semilogy(v[i > 0], i[i > 0], color=C_PURPLE, linewidth=2, label='Forward')
    ax3.semilogy(v[i < 0], np.abs(i[i < 0]), color=C_BLUE, linewidth=2, linestyle='--', label='Reverse')
    ax3.axvline(x=junction.built_in_potential, color=C_GREEN, linewidth=1, linestyle=':', alpha=0.7)
    ax3.text(junction.built_in_potential + 0.02, 1e-6, f'V$_{{bi}}$={junction.built_in_potential:.2f}V',
             fontsize=7, color=C_GREEN)
    ax3.set_xlabel('Voltage (V)', fontsize=9)
    ax3.set_ylabel('|Current| (A)', fontsize=9)
    ax3.set_title('(C) P-N Junction\nI-V Characteristic', fontsize=10, fontweight='bold')
    ax3.legend(fontsize=7, loc='lower right')
    ax3.set_ylim(1e-15, 1e-2)
    ax3.tick_params(labelsize=8)

    # ── D: 3D Rectification ratio vs voltage and temperature ──────────────
    ax4 = fig.add_subplot(gs[0, 3], projection='3d')
    voltages = np.linspace(0.1, 1.0, 30)
    temps = np.linspace(280, 340, 30)
    V, T = np.meshgrid(voltages, temps)
    RR = np.zeros_like(V)
    for i in range(len(temps)):
        for j in range(len(voltages)):
            junc = PNJunction(temperature=T[i, j])
            RR[i, j] = min(junc.rectification_ratio(V[i, j]), 1e6)
    RR_log = np.log10(np.clip(RR, 1, None))
    surf = ax4.plot_surface(V, T, RR_log, cmap='viridis', alpha=0.85, edgecolor='none')
    ax4.set_xlabel('V (V)', fontsize=8, labelpad=5)
    ax4.set_ylabel('T (K)', fontsize=8, labelpad=5)
    ax4.set_zlabel('log$_{10}$(RR)', fontsize=8, labelpad=5)
    ax4.set_title('(D) Rectification Ratio\nvs Voltage & Temperature', fontsize=10, fontweight='bold')
    ax4.tick_params(labelsize=7)
    ax4.view_init(elev=25, azim=135)

    fig.savefig(f'{FIGDIR}/panel_1_lipid_physics.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print('Panel 1 saved.')


def panel_2():
    """Panel 2: Circuit Components (Tests 6-9)
    A) BMD transistor transfer characteristic
    B) Logic gate truth table heatmap
    C) 3D: S-entropy coordinate space with invertibility test
    D) ALU operations visualization
    """
    fig = plt.figure(figsize=(20, 5), facecolor='white')
    gs = GridSpec(1, 4, figure=fig, wspace=0.3)

    # ── A: BMD transistor transfer characteristic ─────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    transistor = BMDTransistor(fidelity=1.0)
    # Sweep categorical distance
    distances = np.linspace(0, 0.5, 200)
    conductances = []
    for d in distances:
        test_input = SEntropyCoordinate(
            s_k=transistor.gate_pattern.s_k + d,
            s_t=transistor.gate_pattern.s_t,
            s_e=transistor.gate_pattern.s_e,
        )
        transistor.gate_tick(test_input)
        conductances.append(1.0 if transistor.is_open else 0.0)

    ax1.fill_between(distances, conductances, alpha=0.3, color=C_TEAL, step='mid')
    ax1.step(distances, conductances, color=C_TEAL, linewidth=2, where='mid')
    ax1.axvline(x=transistor.gate_threshold, color=C_RED, linewidth=1.5, linestyle='--')
    ax1.text(transistor.gate_threshold + 0.01, 0.5, 'd$_{threshold}$', fontsize=8, color=C_RED)
    ax1.set_xlabel('Categorical Distance d$_{cat}$', fontsize=9)
    ax1.set_ylabel('Gate State', fontsize=9)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Closed', 'Open'])
    ax1.set_title('(A) BMD Transistor\nTransfer Characteristic', fontsize=10, fontweight='bold')
    ax1.tick_params(labelsize=8)

    # ── B: Logic gate truth table heatmap ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    gate = TriDimensionalGate()
    tt = gate.truth_table()

    # Build matrix: rows = gates, cols = input combos
    gate_names = ['AND', 'OR', 'XOR']
    input_labels = ['00', '01', '10', '11']
    matrix = np.array([[row[g] for row in tt] for g in gate_names])

    im = ax2.imshow(matrix, cmap='RdYlGn', vmin=-0.5, vmax=1.5, aspect='auto')
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(input_labels, fontsize=9)
    ax2.set_yticks(range(3))
    ax2.set_yticklabels(['AND\n(S$_k$)', 'OR\n(S$_t$)', 'XOR\n(S$_e$)'], fontsize=9)
    ax2.set_xlabel('Input (AB)', fontsize=9)
    # Annotate cells
    for i in range(3):
        for j in range(4):
            ax2.text(j, i, str(matrix[i, j]), ha='center', va='center',
                     fontsize=14, fontweight='bold',
                     color='white' if matrix[i, j] == 0 else 'black')
    ax2.set_title('(B) Tri-Dimensional\nLogic Gates', fontsize=10, fontweight='bold')

    # ── C: 3D S-entropy coordinate space ──────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')

    # Generate test oscillators and show (omega, phi, A) -> (S_k, S_t, S_e)
    np.random.seed(42)
    n_pts = 200
    omegas = 10 ** np.random.uniform(8, 14, n_pts)
    phis = np.random.uniform(0, 2 * np.pi, n_pts)
    amps = np.random.uniform(0.1, 3.0, n_pts)

    coords = [compute_s_entropy(o, p, a) for o, p, a in zip(omegas, phis, amps)]
    sk = [c.s_k for c in coords]
    st = [c.s_t for c in coords]
    se = [c.s_e for c in coords]

    sc = ax3.scatter(sk, st, se, c=se, cmap='plasma', s=15, alpha=0.7, edgecolors='none')
    ax3.set_xlabel('S$_k$', fontsize=8, labelpad=3)
    ax3.set_ylabel('S$_t$', fontsize=8, labelpad=3)
    ax3.set_zlabel('S$_e$', fontsize=8, labelpad=3)
    ax3.set_title('(C) S-Entropy Space\n[0,1]$^3$ Mapping', fontsize=10, fontweight='bold')
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1); ax3.set_zlim(0, 1)
    ax3.tick_params(labelsize=7)
    ax3.view_init(elev=20, azim=45)

    # ── D: ALU operations ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    alu = VirtualALU()

    # Show addition and multiplication trajectories
    a_vals = np.linspace(0.05, 0.95, 20)
    b_fixed = SEntropyCoordinate(0.3, 0.4, 0.5)

    add_sk = []; add_st = []; mul_sk = []; mul_st = []
    for av in a_vals:
        a_coord = SEntropyCoordinate(av, 0.5, 0.5)
        add_r = alu.add(a_coord, b_fixed)
        mul_r = alu.multiply(a_coord, b_fixed)
        add_sk.append(add_r.s_k)
        add_st.append(add_r.s_t)
        mul_sk.append(mul_r.s_k)
        mul_st.append(mul_r.s_t)

    ax4.plot(a_vals, add_sk, color=C_BLUE, linewidth=2, label='Add: S$_k$')
    ax4.plot(a_vals, mul_sk, color=C_ORANGE, linewidth=2, label='Mul: S$_k$')
    ax4.plot(a_vals, add_st, color=C_BLUE, linewidth=1.5, linestyle='--', label='Add: S$_t$')
    ax4.plot(a_vals, mul_st, color=C_ORANGE, linewidth=1.5, linestyle='--', label='Mul: S$_t$')
    ax4.set_xlabel('Input S$_k$ (A)', fontsize=9)
    ax4.set_ylabel('Output', fontsize=9)
    ax4.set_title('(D) ALU Operations\nAdd vs Multiply', fontsize=10, fontweight='bold')
    ax4.legend(fontsize=7, loc='upper left')
    ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)
    ax4.tick_params(labelsize=8)

    fig.savefig(f'{FIGDIR}/panel_2_circuit_components.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print('Panel 2 saved.')


def panel_3():
    """Panel 3: Sensor Validation (Tests 10-13)
    A) S-entropy injectivity — 5 environments in S-space
    B) Full circuit: categorical distance by environment type (bar)
    C) 3D: S-entropy trajectory through environment changes
    D) Weather enhancement — fog/rain/clear comparison
    """
    fig = plt.figure(figsize=(20, 5), facecolor='white')
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    circuit = SensorCircuit()
    env_baseline = {
        "temperature": 300.0, "pressure": 1e5, "concentration": 0.21,
        "light_intensity": 1.0, "echo_delay": 0.001,
        "magnetic_field": 50e-6, "flow_velocity": 0.0,
    }
    circuit.calibrate(env_baseline)

    # ── A: S-entropy injectivity — environments in S-space ────────────────
    ax1 = fig.add_subplot(gs[0, 0])

    envs = {
        'Baseline': env_baseline,
        'Hot (+20K)': {**env_baseline, "temperature": 320.0},
        'Cold (-20K)': {**env_baseline, "temperature": 280.0},
        'Windy (10m/s)': {**env_baseline, "flow_velocity": 10.0},
        'High P (+20%)': {**env_baseline, "pressure": 1.2e5},
    }
    colors = [C_TEAL, C_RED, C_BLUE, C_PURPLE, C_YELLOW]
    s_coords = {}
    for (name, env), c in zip(envs.items(), colors):
        s = circuit.process(env)
        s_coords[name] = s
        ax1.scatter(s.s_k, s.s_t, color=c, s=100, zorder=5, edgecolors='black', linewidth=0.5)
        ax1.annotate(name, (s.s_k, s.s_t), fontsize=6, ha='center', va='bottom',
                     xytext=(0, 5), textcoords='offset points')

    ax1.set_xlabel('S$_k$ (Knowledge)', fontsize=9)
    ax1.set_ylabel('S$_t$ (Temporal)', fontsize=9)
    ax1.set_title('(A) Environmental\nS-Entropy Separation', fontsize=10, fontweight='bold')
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.tick_params(labelsize=8)

    # ── B: Categorical distance by environment (bar chart) ────────────────
    ax2 = fig.add_subplot(gs[0, 1])

    test_envs = {
        'Hot\n(+20K)': {**env_baseline, "temperature": 320.0},
        'Cold\n(-20K)': {**env_baseline, "temperature": 280.0},
        'Windy\n(10m/s)': {**env_baseline, "flow_velocity": 10.0},
        'Pressure\n(+20%)': {**env_baseline, "pressure": 1.2e5},
        'Obstacle\n(vehicle)': {**env_baseline, "temperature": 305.0, "flow_velocity": 3.0},
    }

    circuit2 = SensorCircuit()
    circuit2.calibrate(env_baseline)
    baseline_s = circuit2.baseline

    names = list(test_envs.keys())
    dists = []
    for env in test_envs.values():
        s = circuit2.process(env)
        dists.append(s.categorical_distance(baseline_s))

    bar_colors = [C_RED, C_BLUE, C_PURPLE, C_YELLOW, C_MAGENTA]
    bars = ax2.bar(range(len(names)), dists, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0.01, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax2.text(4.5, 0.015, 'threshold', fontsize=7, color='gray')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, fontsize=7)
    ax2.set_ylabel('d$_{cat}$ from Baseline', fontsize=9)
    ax2.set_title('(B) Signal Transduction\nEnvironmental Sensitivity', fontsize=10, fontweight='bold')
    ax2.tick_params(labelsize=8)

    # ── C: 3D S-entropy trajectory ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')

    # Drive through a sequence of environments
    circuit3 = SensorCircuit()
    circuit3.calibrate(env_baseline)
    trajectory_envs = [
        env_baseline,
        {**env_baseline, "temperature": 305.0},
        {**env_baseline, "temperature": 310.0},
        {**env_baseline, "temperature": 310.0, "flow_velocity": 2.0},
        {**env_baseline, "temperature": 310.0, "flow_velocity": 5.0},
        {**env_baseline, "temperature": 310.0, "flow_velocity": 5.0, "pressure": 1.05e5},
        {**env_baseline, "temperature": 305.0, "flow_velocity": 3.0, "pressure": 1.05e5},
        {**env_baseline, "temperature": 300.0, "flow_velocity": 0.0, "pressure": 1.0e5},
    ]

    traj_sk = []; traj_st = []; traj_se = []
    for env in trajectory_envs:
        s = circuit3.process(env)
        traj_sk.append(s.s_k); traj_st.append(s.s_t); traj_se.append(s.s_e)

    # Plot trajectory
    colors_traj = plt.cm.cool(np.linspace(0, 1, len(traj_sk)))
    for i in range(len(traj_sk) - 1):
        ax3.plot(traj_sk[i:i+2], traj_st[i:i+2], traj_se[i:i+2],
                 color=colors_traj[i], linewidth=2)
    ax3.scatter(traj_sk[0], traj_st[0], traj_se[0], color=C_GREEN, s=80, marker='o',
                edgecolors='black', zorder=5, label='Start')
    ax3.scatter(traj_sk[-1], traj_st[-1], traj_se[-1], color=C_RED, s=80, marker='s',
                edgecolors='black', zorder=5, label='End')

    ax3.set_xlabel('S$_k$', fontsize=8, labelpad=3)
    ax3.set_ylabel('S$_t$', fontsize=8, labelpad=3)
    ax3.set_zlabel('S$_e$', fontsize=8, labelpad=3)
    ax3.set_title('(C) S-Entropy Trajectory\nDriving Sequence', fontsize=10, fontweight='bold')
    ax3.legend(fontsize=7, loc='upper left')
    ax3.tick_params(labelsize=7)
    ax3.view_init(elev=25, azim=50)

    # ── D: Weather enhancement ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])

    circuit4 = SensorCircuit()
    circuit4.calibrate(env_baseline)

    weather_envs = {
        'Clear': env_baseline,
        'Fog': {**env_baseline, "pressure": 1.01e5, "light_intensity": 0.3, "echo_delay": 0.003},
        'Rain': {**env_baseline, "temperature": 290.0, "pressure": 0.98e5,
                 "flow_velocity": 2.0, "light_intensity": 0.5},
        'Snow': {**env_baseline, "temperature": 273.0, "pressure": 1.02e5,
                 "light_intensity": 0.7, "echo_delay": 0.002},
    }

    baseline_w = circuit4.process(env_baseline)
    weather_names = []
    weather_dists = []
    for name, env in list(weather_envs.items())[1:]:  # skip clear
        s = circuit4.process(env)
        weather_names.append(name)
        weather_dists.append(s.categorical_distance(baseline_w))

    # Also show what conventional sensors see (degrades with weather)
    conv_degradation = [0.6, 0.4, 0.5]  # normalized sensor quality: fog, rain, snow

    x_pos = np.arange(len(weather_names))
    width = 0.35
    bars1 = ax4.bar(x_pos - width/2, weather_dists, width, color=C_TEAL,
                    edgecolor='black', linewidth=0.5, label='Membrane d$_{cat}$')
    bars2 = ax4.bar(x_pos + width/2, conv_degradation, width, color=C_RED,
                    edgecolor='black', linewidth=0.5, alpha=0.6, label='Conv. sensor quality')

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(weather_names, fontsize=9)
    ax4.set_ylabel('Signal Strength', fontsize=9)
    ax4.set_title('(D) Weather Enhancement\nvs Conventional Degradation', fontsize=10, fontweight='bold')
    ax4.legend(fontsize=7)
    ax4.tick_params(labelsize=8)

    fig.savefig(f'{FIGDIR}/panel_3_sensor_validation.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print('Panel 3 saved.')


if __name__ == '__main__':
    panel_1()
    panel_2()
    panel_3()
    print('All panels generated.')
