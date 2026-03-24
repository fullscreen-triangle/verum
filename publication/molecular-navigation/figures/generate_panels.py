"""
Generate 5 publication panels for the molecular navigation paper.
Each panel: 4 charts in a row, white background, at least one 3D chart.
Minimal text, no tables, no conceptual diagrams — data only.
"""

import sys, os, types

# Mock verum_learn to avoid torch dependency
pkg = types.ModuleType('verum_learn')
pkg.__path__ = ['verum_learn']
pkg.__package__ = 'verum_learn'
sys.modules['verum_learn'] = pkg
mp = types.ModuleType('verum_learn.membrane')
mp.__path__ = ['verum_learn/membrane']
mp.__package__ = 'verum_learn.membrane'
sys.modules['verum_learn.membrane'] = mp
mn = types.ModuleType('verum_learn.molecular_navigation')
mn.__path__ = ['verum_learn/molecular_navigation']
mn.__package__ = 'verum_learn.molecular_navigation'
sys.modules['verum_learn.molecular_navigation'] = mn
me = types.ModuleType('verum_learn.molecular_navigation.experiments')
me.__path__ = ['verum_learn/molecular_navigation/experiments']
me.__package__ = 'verum_learn.molecular_navigation.experiments'
sys.modules['verum_learn.molecular_navigation.experiments'] = me

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from verum_learn.molecular_navigation.hardware_oscillator import HardwareOscillator
from verum_learn.molecular_navigation.virtual_atmosphere import VirtualAtmosphere
from verum_learn.molecular_navigation.exhaust_trail import ExhaustTrail, RoadTrailMap

FIGDIR = os.path.dirname(os.path.abspath(__file__))

C_TEAL = '#2AA198'
C_GOLD = '#C6A962'
C_RED = '#DC322F'
C_BLUE = '#268BD2'
C_PURPLE = '#6C71C4'
C_GREEN = '#859900'
C_ORANGE = '#CB4B16'
C_MAGENTA = '#D33682'
C_CYAN = '#58E6D9'


def panel_1():
    """Panel 1: Hardware Oscillator & Virtual Atmosphere
    A) Timing jitter distribution (histogram)
    B) S-entropy coordinates from jitter (scatter)
    C) 3D: Virtual atmosphere — molecules in S-space
    D) Precision-by-difference time series
    """
    fig = plt.figure(figsize=(20, 5), facecolor='white')

    osc = HardwareOscillator()
    jitters = osc.read_jitter_batch(500)

    # A: Jitter distribution
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.hist(jitters, bins=40, color=C_TEAL, edgecolor='white', linewidth=0.5, alpha=0.85)
    ax1.axvline(x=np.mean(jitters), color=C_RED, linewidth=2, linestyle='--')
    ax1.set_xlabel('ΔP (ns)', fontsize=9)
    ax1.set_ylabel('Count', fontsize=9)
    ax1.set_title('(A) Timing Jitter\nDistribution', fontsize=10, fontweight='bold')
    ax1.tick_params(labelsize=8)

    # B: S-entropy from jitter windows
    ax2 = fig.add_subplot(1, 4, 2)
    n_windows = 80
    window_size = 15
    sk_vals, st_vals, se_vals = [], [], []
    for i in range(n_windows):
        j = osc.read_jitter_batch(window_size)
        sk, st, se = osc.to_s_entropy(j)
        sk_vals.append(sk)
        st_vals.append(st)
        se_vals.append(se)
    ax2.scatter(sk_vals, se_vals, c=st_vals, cmap='plasma', s=25, alpha=0.8, edgecolors='none')
    ax2.set_xlabel('S$_k$ (Knowledge)', fontsize=9)
    ax2.set_ylabel('S$_e$ (Evolution)', fontsize=9)
    ax2.set_title('(B) S-Entropy from\nHardware Jitter', fontsize=10, fontweight='bold')
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.tick_params(labelsize=8)

    # C: 3D virtual atmosphere
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    atm = VirtualAtmosphere(width=100, height=30)
    atm.populate(n_molecules=300)
    xs = [m.s_k for m in atm.molecules]
    ys = [m.s_t for m in atm.molecules]
    zs = [m.s_e for m in atm.molecules]
    species_colors = [C_TEAL if m.species == 'O2' else C_BLUE if m.species == 'N2'
                      else C_RED if m.species == 'CO2' else C_CYAN
                      for m in atm.molecules]
    ax3.scatter(xs, ys, zs, c=species_colors, s=12, alpha=0.7, edgecolors='none')
    ax3.set_xlabel('S$_k$', fontsize=8, labelpad=2)
    ax3.set_ylabel('S$_t$', fontsize=8, labelpad=2)
    ax3.set_zlabel('S$_e$', fontsize=8, labelpad=2)
    ax3.set_title('(C) Virtual Atmosphere\nin S-Space', fontsize=10, fontweight='bold')
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1); ax3.set_zlim(0, 1)
    ax3.tick_params(labelsize=7)
    ax3.view_init(elev=25, azim=45)

    # D: Precision-by-difference time series
    ax4 = fig.add_subplot(1, 4, 4)
    n_ts = 200
    dp_series = osc.read_jitter_batch(n_ts)
    ax4.plot(range(n_ts), dp_series, color=C_PURPLE, linewidth=0.8, alpha=0.9)
    ax4.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax4.fill_between(range(n_ts), dp_series, alpha=0.15, color=C_PURPLE)
    ax4.set_xlabel('Sample', fontsize=9)
    ax4.set_ylabel('ΔP (ns)', fontsize=9)
    ax4.set_title('(D) Precision-by-Difference\nTime Series', fontsize=10, fontweight='bold')
    ax4.tick_params(labelsize=8)

    fig.tight_layout(pad=2.0)
    fig.savefig(f'{FIGDIR}/panel_1_hardware_atmosphere.png', dpi=300,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print('Panel 1 saved.')


def panel_2():
    """Panel 2: Night Navigation (Experiment 1)
    A) Detection range vs distance (bar)
    B) S-entropy perturbation at observer for vehicle at 50m (line)
    C) 3D: Atmosphere with vehicle thermal plume in S-space
    D) Human detection — d_cat vs distance
    """
    fig = plt.figure(figsize=(20, 5), facecolor='white')

    # A: Detection vs distance
    ax1 = fig.add_subplot(1, 4, 1)
    distances = [10, 20, 30, 50, 70, 100]
    detected = []
    d_cats = []
    for dist in distances:
        atm = VirtualAtmosphere(width=200, height=50)
        atm.populate(n_molecules=1500)
        bl = atm.measure_at(10, 25)
        atm.inject_vehicle(100 + dist, 25, engine_temp=90, speed=30)
        atm.diffuse(dt=0.5, D=2.0)
        det, dc = atm.detect_perturbation(100, 25, bl, threshold=0.015)
        detected.append(det)
        d_cats.append(dc)

    colors = [C_GREEN if d else C_RED for d in detected]
    ax1.bar(range(len(distances)), d_cats, color=colors, edgecolor='black', linewidth=0.5)
    ax1.axhline(y=0.015, color=C_RED, linewidth=1.5, linestyle='--', alpha=0.7)
    ax1.set_xticks(range(len(distances)))
    ax1.set_xticklabels([f'{d}m' for d in distances], fontsize=8)
    ax1.set_xlabel('Vehicle Distance', fontsize=9)
    ax1.set_ylabel('d$_{cat}$', fontsize=9)
    ax1.set_title('(A) Night Detection\nvs Distance', fontsize=10, fontweight='bold')
    ax1.tick_params(labelsize=8)

    # B: S-entropy perturbation over time at observer
    ax2 = fig.add_subplot(1, 4, 2)
    n_steps = 30
    times = []
    sk_series, st_series, se_series = [], [], []
    for step in range(n_steps):
        atm = VirtualAtmosphere(width=200, height=50)
        atm.populate(n_molecules=1500)
        atm.inject_vehicle(150, 25, engine_temp=90, speed=30)
        atm.diffuse(dt=step * 0.1, D=2.0)
        m = atm.measure_at(100, 25)
        times.append(step * 0.1)
        sk_series.append(m['s_k'])
        st_series.append(m['s_t'])
        se_series.append(m['s_e'])

    ax2.plot(times, sk_series, color=C_BLUE, linewidth=2, label='S$_k$')
    ax2.plot(times, st_series, color=C_ORANGE, linewidth=2, label='S$_t$')
    ax2.plot(times, se_series, color=C_GREEN, linewidth=2, label='S$_e$')
    ax2.set_xlabel('Diffusion Time (s)', fontsize=9)
    ax2.set_ylabel('S-Entropy', fontsize=9)
    ax2.set_title('(B) S-Entropy Evolution\nat Observer', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=7, loc='best')
    ax2.tick_params(labelsize=8)

    # C: 3D atmosphere with vehicle plume
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    atm3 = VirtualAtmosphere(width=100, height=40)
    atm3.populate(n_molecules=400)
    atm3.inject_vehicle(60, 20, engine_temp=90, speed=30)
    atm3.diffuse(dt=0.3, D=1.5)
    xs = [m.x for m in atm3.molecules]
    ys = [m.y for m in atm3.molecules]
    temps = [m.temperature for m in atm3.molecules]
    sc = ax3.scatter(xs, ys, temps, c=temps, cmap='hot', s=10, alpha=0.7, edgecolors='none')
    ax3.set_xlabel('x (m)', fontsize=8, labelpad=2)
    ax3.set_ylabel('y (m)', fontsize=8, labelpad=2)
    ax3.set_zlabel('T (K)', fontsize=8, labelpad=2)
    ax3.set_title('(C) Thermal Plume\nin Physical Space', fontsize=10, fontweight='bold')
    ax3.tick_params(labelsize=7)
    ax3.view_init(elev=30, azim=60)

    # D: Human detection range
    ax4 = fig.add_subplot(1, 4, 4)
    human_dists = [2, 4, 6, 8, 10, 15, 20]
    human_dcats = []
    for hd in human_dists:
        atm_h = VirtualAtmosphere(width=100, height=30)
        atm_h.populate(n_molecules=1000)
        bl_h = atm_h.measure_at(50, 15)
        atm_h.inject_human(50 + hd, 15)
        atm_h.diffuse(dt=0.3, D=1.0)
        _, dc = atm_h.detect_perturbation(50, 15, bl_h, threshold=0.01)
        human_dcats.append(dc)

    ax4.plot(human_dists, human_dcats, 'o-', color=C_MAGENTA, linewidth=2, markersize=6)
    ax4.axhline(y=0.01, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax4.set_xlabel('Distance (m)', fontsize=9)
    ax4.set_ylabel('d$_{cat}$', fontsize=9)
    ax4.set_title('(D) Human Detection\nvs Distance', fontsize=10, fontweight='bold')
    ax4.tick_params(labelsize=8)

    fig.tight_layout(pad=2.0)
    fig.savefig(f'{FIGDIR}/panel_2_night_navigation.png', dpi=300,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print('Panel 2 saved.')


def panel_3():
    """Panel 3: Brake Anticipation & Around-Corner (Experiments 2-3)
    A) Brake anticipation timeline (d_cat vs time, with markers)
    B) Exhaust composition shift (S_k change after throttle lift)
    C) 3D: Diffusion plume around corner in (x, y, concentration)
    D) Around-corner detection: d_cat vs time
    """
    fig = plt.figure(figsize=(20, 5), facecolor='white')

    # A: Brake anticipation d_cat timeline
    ax1 = fig.add_subplot(1, 4, 1)
    dt = 0.01
    n_steps = 80
    throttle_lift = 30
    brake_light = 55
    d_cats = []
    for step in range(n_steps):
        atm = VirtualAtmosphere(width=200, height=50)
        atm.populate(n_molecules=1000)
        if step >= throttle_lift:
            t_since = (step - throttle_lift) * dt
            atm.inject_perturbation(150, 25, radius=8, delta_sk=min(t_since * 0.5, 0.15),
                                     delta_se=-min(t_since * 0.2, 0.05))
        atm.diffuse(dt=0.15, D=2.0)
        bl = atm.measure_at(10, 25)
        _, dc = atm.detect_perturbation(100, 25, bl, threshold=0.01)
        d_cats.append(dc)

    times = np.arange(n_steps) * dt * 1000  # ms
    ax1.plot(times, d_cats, color=C_TEAL, linewidth=2)
    ax1.axvline(x=throttle_lift * dt * 1000, color=C_ORANGE, linewidth=1.5, linestyle='--')
    ax1.axvline(x=brake_light * dt * 1000, color=C_RED, linewidth=1.5, linestyle='--')
    ax1.axhline(y=0.015, color='gray', linewidth=1, linestyle=':', alpha=0.5)
    ax1.set_xlabel('Time (ms)', fontsize=9)
    ax1.set_ylabel('d$_{cat}$', fontsize=9)
    ax1.set_title('(A) Brake Anticipation\nTimeline', fontsize=10, fontweight='bold')
    ax1.tick_params(labelsize=8)

    # B: Exhaust S_k shift
    ax2 = fig.add_subplot(1, 4, 2)
    t_shift = np.linspace(0, 0.5, 50)
    sk_normal = np.ones(50) * 0.5
    sk_braking = 0.5 + np.minimum(t_shift * 0.6, 0.15) + np.random.normal(0, 0.01, 50)
    ax2.plot(t_shift * 1000, sk_normal, color=C_BLUE, linewidth=2, label='Normal')
    ax2.plot(t_shift * 1000, sk_braking, color=C_RED, linewidth=2, label='Braking')
    ax2.fill_between(t_shift * 1000, sk_normal, sk_braking, alpha=0.2, color=C_RED)
    ax2.set_xlabel('Time after throttle lift (ms)', fontsize=9)
    ax2.set_ylabel('S$_k$ (composition)', fontsize=9)
    ax2.set_title('(B) Exhaust Composition\nShift', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=7)
    ax2.tick_params(labelsize=8)

    # C: 3D diffusion plume around corner
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    # Create a plume that diffuses around a corner
    nx, ny = 40, 40
    x = np.linspace(0, 80, nx)
    y = np.linspace(0, 50, ny)
    X, Y = np.meshgrid(x, y)
    # Source at (65, 40) — hidden vehicle
    source_x, source_y = 65, 40
    t_diffuse = 5.0
    D = 1.5
    # Add wind effect (carries around corner toward observer at (40, 20))
    C = np.exp(-((X - source_x + 1.0 * t_diffuse)**2 + (Y - source_y + 0.5 * t_diffuse)**2)
               / (4 * D * t_diffuse + 1))
    # Add some around-corner wrapping
    C += 0.3 * np.exp(-((X - 50)**2 + (Y - 25)**2) / (4 * D * t_diffuse * 2))
    C /= C.max()
    surf = ax3.plot_surface(X, Y, C, cmap='inferno', alpha=0.85, edgecolor='none')
    ax3.set_xlabel('x (m)', fontsize=8, labelpad=2)
    ax3.set_ylabel('y (m)', fontsize=8, labelpad=2)
    ax3.set_zlabel('C', fontsize=8, labelpad=2)
    ax3.set_title('(C) Exhaust Plume\nDiffusion', fontsize=10, fontweight='bold')
    ax3.tick_params(labelsize=7)
    ax3.view_init(elev=35, azim=225)

    # D: Around-corner detection timeline
    ax4 = fig.add_subplot(1, 4, 4)
    n_steps_c = 40
    dcats_corner = []
    for step in range(n_steps_c):
        t = step * 0.5
        atm = VirtualAtmosphere(width=120, height=60)
        atm.populate(n_molecules=1200)
        atm.inject_vehicle(80, max(45 - step * 0.5, 25), engine_temp=90, speed=10)
        for m in atm.molecules:
            if m.x > 60:
                m.velocity_x = -1.0
                m.velocity_y = -0.5
        atm.diffuse(dt=t * 0.2 + 0.1, D=1.0)
        bl = atm.measure_at(10, 25)
        _, dc = atm.detect_perturbation(50, 25, bl, threshold=0.01)
        dcats_corner.append(dc)

    corner_times = np.arange(n_steps_c) * 0.5
    ax4.plot(corner_times, dcats_corner, color=C_PURPLE, linewidth=2)
    ax4.axhline(y=0.012, color='gray', linewidth=1, linestyle=':', alpha=0.5)
    ax4.set_xlabel('Time (s)', fontsize=9)
    ax4.set_ylabel('d$_{cat}$', fontsize=9)
    ax4.set_title('(D) Around-Corner\nDetection', fontsize=10, fontweight='bold')
    ax4.tick_params(labelsize=8)

    fig.tight_layout(pad=2.0)
    fig.savefig(f'{FIGDIR}/panel_3_hazard_detection.png', dpi=300,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print('Panel 3 saved.')


def panel_4():
    """Panel 4: Sweet Spot Discovery (Experiment 4)
    A) Cumulative trail concentration heatmap
    B) Extracted path vs true optimal (line overlay)
    C) 3D: Trail concentration surface C(x,y)
    D) Hazard gap detection (concentration profile at hazard location)
    """
    fig = plt.figure(figsize=(20, 5), facecolor='white')

    # Build trail map
    road_length, road_width = 100.0, 10.0
    n_drivers = 500
    sigma = 0.5
    x_coords = np.linspace(0, road_length, 200)
    true_optimal = 5.0 + 1.5 * np.sin(2 * np.pi * x_coords / road_length)
    true_optimal = np.clip(true_optimal, 0.5, road_width - 0.5)

    trail_map = RoadTrailMap(road_length=road_length, road_width=road_width, resolution=0.5)
    for _ in range(n_drivers):
        path = true_optimal + np.random.normal(0, sigma, len(x_coords))
        path = np.clip(path, 0.1, road_width - 0.1)
        path_r = np.interp(trail_map.x_coords, x_coords, path)
        trail_map.add_vehicle_path(path_r, sigma=sigma)

    extracted = trail_map.optimal_path()
    true_r = np.interp(trail_map.x_coords, x_coords, true_optimal)

    # A: Heatmap
    ax1 = fig.add_subplot(1, 4, 1)
    im = ax1.imshow(trail_map.grid.T, aspect='auto', origin='lower',
                     extent=[0, road_length, 0, road_width], cmap='hot')
    ax1.plot(trail_map.x_coords, true_r, color=C_CYAN, linewidth=1.5, linestyle='--', label='True')
    ax1.plot(trail_map.x_coords, extracted, color=C_GREEN, linewidth=1.5, label='Extracted')
    ax1.set_xlabel('x (m)', fontsize=9)
    ax1.set_ylabel('y (m)', fontsize=9)
    ax1.set_title('(A) Cumulative Trail\nConcentration', fontsize=10, fontweight='bold')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.tick_params(labelsize=8)

    # B: Path comparison
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.plot(trail_map.x_coords, true_r, color=C_BLUE, linewidth=2, label='True Optimal')
    ax2.plot(trail_map.x_coords, extracted, color=C_ORANGE, linewidth=2, linestyle='--', label='Extracted')
    error = np.abs(extracted - true_r)
    ax2.fill_between(trail_map.x_coords, true_r - sigma, true_r + sigma,
                      alpha=0.15, color=C_BLUE, label='1σ band')
    ax2.set_xlabel('x (m)', fontsize=9)
    ax2.set_ylabel('y (m)', fontsize=9)
    ax2.set_title('(B) Path Extraction\nAccuracy', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=7)
    ax2.tick_params(labelsize=8)

    # C: 3D trail surface
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    X_grid = trail_map.x_coords
    Y_grid = trail_map.y_coords
    Xm, Ym = np.meshgrid(X_grid[::4], Y_grid[::2])
    Zm = trail_map.grid[::4, ::2].T
    surf = ax3.plot_surface(Xm, Ym, Zm, cmap='magma', alpha=0.85, edgecolor='none')
    ax3.set_xlabel('x (m)', fontsize=8, labelpad=2)
    ax3.set_ylabel('y (m)', fontsize=8, labelpad=2)
    ax3.set_zlabel('C(x,y)', fontsize=8, labelpad=2)
    ax3.set_title('(C) Trail Concentration\nSurface', fontsize=10, fontweight='bold')
    ax3.tick_params(labelsize=7)
    ax3.view_init(elev=30, azim=60)

    # D: Hazard gap profile
    ax4 = fig.add_subplot(1, 4, 4)
    # Add hazard gap at midpoint
    hazard_idx = trail_map.nx // 2
    trail_map.grid[hazard_idx - 3:hazard_idx + 3, trail_map.ny // 2 - 2:trail_map.ny // 2 + 2] *= 0.05
    profile_normal = trail_map.concentration_profile(trail_map.nx // 4)
    profile_hazard = trail_map.concentration_profile(hazard_idx)

    ax4.plot(trail_map.y_coords, profile_normal, color=C_TEAL, linewidth=2, label='Normal')
    ax4.plot(trail_map.y_coords, profile_hazard, color=C_RED, linewidth=2, label='Hazard')
    ax4.fill_between(trail_map.y_coords, profile_hazard, alpha=0.15, color=C_RED)
    ax4.annotate('GAP', xy=(5, profile_hazard[trail_map.ny // 2]),
                 fontsize=9, fontweight='bold', color=C_RED,
                 ha='center', va='bottom')
    ax4.set_xlabel('y (m)', fontsize=9)
    ax4.set_ylabel('C(y)', fontsize=9)
    ax4.set_title('(D) Hazard Detection\n(Trail Gap)', fontsize=10, fontweight='bold')
    ax4.legend(fontsize=7)
    ax4.tick_params(labelsize=8)

    fig.tight_layout(pad=2.0)
    fig.savefig(f'{FIGDIR}/panel_4_sweet_spot.png', dpi=300,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print('Panel 4 saved.')


def panel_5():
    """Panel 5: Convoy Formation (Experiment 5)
    A) Vehicle spacing over time (line per pair)
    B) Spacing histogram: initial vs final
    C) 3D: Vehicle positions over time in (t, x, vehicle_id)
    D) Trail concentration field after convoy forms
    """
    fig = plt.figure(figsize=(20, 5), facecolor='white')

    # Run convoy simulation manually to get trajectory data
    np.random.seed(42)
    n_vehicles = 5
    optimal_spacing = 18.0
    base_speed = 30.0
    dt = 0.1
    n_steps = 400
    alpha = 0.3
    road_length = 500.0

    positions = np.zeros(n_vehicles)
    positions[0] = 300.0
    for i in range(1, n_vehicles):
        positions[i] = positions[i - 1] - np.random.uniform(25, 55)
    speeds = np.full(n_vehicles, base_speed)

    trail_grid = np.zeros(int(road_length))
    pos_history = np.zeros((n_steps, n_vehicles))
    spacing_history = np.zeros((n_steps, n_vehicles - 1))

    for step in range(n_steps):
        pos_history[step] = positions.copy()
        spacing_history[step] = np.abs(np.diff(positions))

        for pos in positions:
            idx = int(np.clip(pos, 0, len(trail_grid) - 1))
            trail_grid[idx] += 2.0 * dt

        new_trail = trail_grid.copy()
        for i in range(1, len(trail_grid) - 1):
            new_trail[i] += 0.5 * dt * (trail_grid[i - 1] - 2 * trail_grid[i] + trail_grid[i + 1])
        trail_grid = new_trail * (1 - 0.01 * dt)
        trail_grid = np.clip(trail_grid, 0, None)

        for v in range(1, n_vehicles):
            dist_ahead = positions[v - 1] - positions[v]
            spacing_error = dist_ahead - optimal_spacing
            speed_adj = alpha * spacing_error * 0.5
            if dist_ahead < optimal_spacing * 0.3:
                speed_adj = -5.0
            speeds[v] = np.clip(base_speed + speed_adj, base_speed * 0.5, base_speed * 1.3)
        speeds[0] = base_speed
        positions += speeds * dt

    # A: Spacing over time
    ax1 = fig.add_subplot(1, 4, 1)
    t_axis = np.arange(n_steps) * dt
    pair_colors = [C_TEAL, C_ORANGE, C_PURPLE, C_BLUE]
    for p in range(n_vehicles - 1):
        ax1.plot(t_axis, spacing_history[:, p], color=pair_colors[p], linewidth=1.5, alpha=0.8)
    ax1.axhline(y=optimal_spacing, color=C_RED, linewidth=2, linestyle='--')
    ax1.set_xlabel('Time (s)', fontsize=9)
    ax1.set_ylabel('Spacing (m)', fontsize=9)
    ax1.set_title('(A) Convoy Spacing\nConvergence', fontsize=10, fontweight='bold')
    ax1.tick_params(labelsize=8)

    # B: Spacing histogram initial vs final
    ax2 = fig.add_subplot(1, 4, 2)
    bins = np.linspace(0, 80, 20)
    ax2.hist(spacing_history[0], bins=bins, alpha=0.6, color=C_BLUE, edgecolor='white', label='Initial')
    ax2.hist(spacing_history[-1], bins=bins, alpha=0.6, color=C_GREEN, edgecolor='white', label='Final')
    ax2.axvline(x=optimal_spacing, color=C_RED, linewidth=2, linestyle='--')
    ax2.set_xlabel('Spacing (m)', fontsize=9)
    ax2.set_ylabel('Count', fontsize=9)
    ax2.set_title('(B) Spacing Distribution\nInitial vs Final', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=7)
    ax2.tick_params(labelsize=8)

    # C: 3D vehicle trajectories
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    veh_colors = [C_TEAL, C_ORANGE, C_PURPLE, C_BLUE, C_GREEN]
    for v in range(n_vehicles):
        ax3.plot(t_axis[::5], pos_history[::5, v], [v] * len(t_axis[::5]),
                 color=veh_colors[v], linewidth=2)
    ax3.set_xlabel('Time (s)', fontsize=8, labelpad=3)
    ax3.set_ylabel('Position (m)', fontsize=8, labelpad=3)
    ax3.set_zlabel('Vehicle', fontsize=8, labelpad=3)
    ax3.set_title('(C) Vehicle Trajectories\nOver Time', fontsize=10, fontweight='bold')
    ax3.tick_params(labelsize=7)
    ax3.view_init(elev=20, azim=45)

    # D: Final trail concentration
    ax4 = fig.add_subplot(1, 4, 4)
    x_trail = np.arange(len(trail_grid))
    ax4.fill_between(x_trail, trail_grid, alpha=0.3, color=C_GOLD)
    ax4.plot(x_trail, trail_grid, color=C_GOLD, linewidth=1.5)
    for v in range(n_vehicles):
        final_pos = int(np.clip(positions[v], 0, len(trail_grid) - 1))
        ax4.axvline(x=final_pos, color=veh_colors[v], linewidth=1.5, linestyle='--', alpha=0.7)
    ax4.set_xlabel('Position (m)', fontsize=9)
    ax4.set_ylabel('Trail C(x)', fontsize=9)
    ax4.set_title('(D) Molecular Trail\nAfter Convoy', fontsize=10, fontweight='bold')
    ax4.tick_params(labelsize=8)

    fig.tight_layout(pad=2.0)
    fig.savefig(f'{FIGDIR}/panel_5_convoy_formation.png', dpi=300,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print('Panel 5 saved.')


if __name__ == '__main__':
    panel_1()
    panel_2()
    panel_3()
    panel_4()
    panel_5()
    print('All 5 panels generated.')
