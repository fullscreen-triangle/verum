"""
FastF1 Data Fetching + Preprocessing for the Philharmonic F1 validation.

If FastF1 cannot fetch data (no internet, API issues), we fall back to
synthetic F1-like telemetry that preserves realistic physical behaviour.
"""

import os
import warnings
import numpy as np
from typing import Optional, Dict, List, Tuple

# Suppress FastF1 warnings during import
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WHEEL_DIAMETER = 0.66  # metres (front and rear F1 tyre)
WHEEL_RADIUS = WHEEL_DIAMETER / 2.0

# Gear ratios (approximate, typical F1 8-speed)
GEAR_RATIOS = {
    0: 0.0,     # neutral
    1: 3.50,
    2: 2.60,
    3: 2.00,
    4: 1.63,
    5: 1.35,
    6: 1.15,
    7: 1.00,
    8: 0.88,
}


# ---------------------------------------------------------------------------
# FastF1 wrappers with fallback
# ---------------------------------------------------------------------------

def _init_cache():
    """Enable FastF1 cache in the project directory."""
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    try:
        import fastf1
        fastf1.Cache.enable_cache(cache_dir)
    except Exception:
        pass


def load_session(year: int, race, session: str = "R"):
    """
    Load an F1 session via FastF1.

    Parameters
    ----------
    year : int
    race : str or int
    session : str   ('R', 'Q', 'FP1', etc.)

    Returns
    -------
    session object, or None on failure.
    """
    _init_cache()
    try:
        import fastf1
        sess = fastf1.get_session(year, race, session)
        sess.load()
        return sess
    except Exception as exc:
        warnings.warn(f"FastF1 load failed ({exc}); will use synthetic data.")
        return None


def get_driver_telemetry(session, driver: str, lap: Optional[int] = None):
    """
    Get telemetry DataFrame for a driver.

    Returns a DataFrame with columns:
        Speed, RPM, Throttle, Brake, nGear, DRS, (plus timestamps)
    or None on failure.
    """
    if session is None:
        return None
    try:
        drv = session.laps.pick_driver(driver)
        if lap is not None:
            drv = drv[drv["LapNumber"] == lap]
        if drv.empty:
            return None
        # Use the fastest lap if no specific lap requested
        if lap is None:
            fastest = drv.pick_fastest()
        else:
            fastest = drv.iloc[0]
        tel = fastest.get_telemetry()
        return tel
    except Exception:
        return None


def get_lap_telemetry(session, driver: str, lap_number: int):
    """Get telemetry for a specific lap number."""
    if session is None:
        return None
    try:
        drv = session.laps.pick_driver(driver)
        lap_data = drv[drv["LapNumber"] == lap_number]
        if lap_data.empty:
            return None
        tel = lap_data.iloc[0].get_telemetry()
        return tel
    except Exception:
        return None


def get_retirement_info(session) -> List[Dict]:
    """
    Find drivers who retired and the reason.

    Returns list of dicts with 'Driver', 'Status', 'LapsCompleted'.
    """
    if session is None:
        return []
    try:
        results = session.results
        retired = results[results["Status"] != "Finished"]
        infos = []
        for _, row in retired.iterrows():
            infos.append({
                "Driver": row.get("Abbreviation", "UNK"),
                "Status": row.get("Status", "Unknown"),
                "LapsCompleted": int(row.get("LapsCompleted", 0)) if not np.isnan(row.get("LapsCompleted", 0)) else 0,
            })
        return infos
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Telemetry → circuit graph observation mapping
# ---------------------------------------------------------------------------

def telemetry_to_observations(row) -> Dict[str, float]:
    """
    Map a single telemetry sample (row / dict) to circuit graph observable
    node values.

    Mapping:
        Speed  → wheel nodes  (omega = v / (pi * d))
        RPM    → ICE node
        Throttle → ICE drive  (will be treated as external input)
        Brake  → brake nodes  (proportional to brake %)
        nGear  → Gearbox node (gear ratio → frequency)
        DRS    → Aero node    (open/closed modifies downforce)
    """
    obs = {}
    # Speed in km/h
    speed_kmh = _get(row, "Speed", 0.0)
    speed_ms = speed_kmh / 3.6
    omega_wheel = speed_ms / (np.pi * WHEEL_DIAMETER)  # rev/s
    for wname in ("FL_Wheel", "FR_Wheel", "RL_Wheel", "RR_Wheel"):
        obs[wname] = omega_wheel

    # RPM
    rpm = _get(row, "RPM", 0.0)
    obs["ICE"] = rpm / 60.0  # convert to Hz

    # Brake (0-100 or boolean)
    brake = _get(row, "Brake", 0.0)
    # FastF1 Brake can be bool or 0-100
    if isinstance(brake, bool):
        brake = 100.0 if brake else 0.0
    brake_val = float(brake)
    for bname in ("FL_Brake", "FR_Brake", "RL_Brake", "RR_Brake"):
        obs[bname] = brake_val

    # Gear
    gear = int(_get(row, "nGear", 0))
    ratio = GEAR_RATIOS.get(gear, 1.0)
    obs["Gearbox"] = ratio * (rpm / 60.0) if rpm else 0.0

    # DRS (0, 1, or 10-14 in some years)
    drs = _get(row, "DRS", 0)
    # Aero downforce is reduced when DRS is open
    drs_open = float(drs) > 0 if isinstance(drs, (int, float)) else False
    aero_factor = 0.6 if drs_open else 1.0
    # Downforce ~ speed^2
    obs["Aero"] = aero_factor * (speed_ms ** 2) / 100.0

    return obs


def telemetry_to_external_drive(row) -> np.ndarray:
    """
    Build the 20-element external drive vector from a telemetry sample.

    Throttle injects current into ICE; braking injects into brake nodes.
    """
    from .circuit import N_NODES, NODE_INDEX
    drive = np.zeros(N_NODES)
    throttle = _get(row, "Throttle", 0.0) / 100.0  # normalise to [0, 1]
    drive[NODE_INDEX["ICE"]] = throttle

    brake = _get(row, "Brake", 0.0)
    if isinstance(brake, bool):
        brake = 1.0 if brake else 0.0
    else:
        brake = float(brake) / 100.0
    for bname in ("FL_Brake", "FR_Brake", "RL_Brake", "RR_Brake"):
        drive[NODE_INDEX[bname]] = brake

    return drive


# ---------------------------------------------------------------------------
# Synthetic telemetry fallback
# ---------------------------------------------------------------------------

def generate_synthetic_telemetry(n_laps: int = 10,
                                  lap_time: float = 90.0,
                                  sample_rate: float = 4.0,
                                  seed: int = 42) -> List[Dict]:
    """
    Generate realistic F1-like telemetry without network access.

    Returns a list of dicts, each representing one time sample with keys:
        Speed, RPM, Throttle, Brake, nGear, DRS, LapNumber, Time

    The synthetic circuit has:
        - Long straight   (accel to ~340 km/h, DRS zone)
        - Heavy braking   (340 → 80 km/h)
        - Medium corner   (80-120 km/h)
        - Short straight  (accel to ~280 km/h)
        - Chicane          (250 → 150 → 250 km/h)
    repeated once per lap.
    """
    rng = np.random.RandomState(seed)
    samples_per_lap = int(lap_time * sample_rate)
    t_lap = np.linspace(0, 1, samples_per_lap)

    data = []
    for lap in range(1, n_laps + 1):
        # Tire degradation factor: lap time increases ~0.1 s/lap
        deg = 1.0 + 0.003 * (lap - 1)

        for idx, t in enumerate(t_lap):
            sample = _synthetic_sample(t, deg, rng, lap, idx / sample_rate + (lap - 1) * lap_time)
            data.append(sample)
    return data


def generate_synthetic_lap_times(n_laps: int = 30, base_time: float = 90.0,
                                  seed: int = 42) -> np.ndarray:
    """
    Generate realistic lap times with tire degradation curve.

    Shows a fuel-effect improvement early, then linear degradation,
    then a sharp cliff near the end.
    """
    rng = np.random.RandomState(seed)
    laps = np.arange(1, n_laps + 1, dtype=float)
    # Fuel effect: ~0.06 s/lap improvement
    fuel = -0.06 * laps
    # Tire degradation: quadratic
    tire = 0.005 * (laps ** 1.5)
    # Cliff at lap ~25
    cliff = np.where(laps > 24, 0.5 * (laps - 24) ** 2, 0.0)
    noise = rng.normal(0, 0.15, n_laps)
    return base_time + fuel + tire + cliff + noise


def generate_synthetic_qualifying(n_laps: int = 5,
                                   lap_time: float = 87.0,
                                   sample_rate: float = 4.0,
                                   seed: int = 99) -> Tuple[List[List[Dict]], int]:
    """
    Generate qualifying-like telemetry: multiple push laps.

    Returns (list_of_laps, fastest_lap_index).
    Each lap is a list of telemetry dicts.
    """
    rng = np.random.RandomState(seed)
    samples_per_lap = int(lap_time * sample_rate)
    t_lap = np.linspace(0, 1, samples_per_lap)

    laps = []
    lap_times = []
    for lap_idx in range(n_laps):
        # Variation: each lap has slightly different execution
        speed_offset = rng.uniform(-3, 3)
        lap_data = []
        for idx, t in enumerate(t_lap):
            sample = _synthetic_sample(
                t, 1.0, rng, lap_idx + 1,
                idx / sample_rate,
                speed_offset=speed_offset
            )
            lap_data.append(sample)
        laps.append(lap_data)
        # Approximate lap time from speed variance
        lap_times.append(lap_time + speed_offset * 0.1 + rng.normal(0, 0.05))

    fastest = int(np.argmin(lap_times))
    return laps, fastest


def _synthetic_sample(t: float, deg: float, rng, lap: int,
                      time_s: float, speed_offset: float = 0.0) -> Dict:
    """Generate one synthetic telemetry sample at normalised position t in [0,1]."""
    # Circuit profile (speed in km/h as function of normalised distance)
    # Sector 1: straight (0-0.25)
    # Sector 2: heavy braking + slow corner (0.25-0.40)
    # Sector 3: medium speed section (0.40-0.60)
    # Sector 4: back straight (0.60-0.80)
    # Sector 5: chicane (0.80-1.00)

    noise = rng.normal(0, 2)

    if t < 0.25:
        # Acceleration on main straight
        frac = t / 0.25
        speed = 80 + 260 * frac  # 80 → 340 km/h
        throttle = 100.0
        brake = 0.0
        gear = min(8, int(2 + frac * 6.5))
        drs = 1 if frac > 0.3 else 0
    elif t < 0.32:
        # Heavy braking
        frac = (t - 0.25) / 0.07
        speed = 340 - 260 * frac  # 340 → 80
        throttle = 0.0
        brake = 100.0 * (1 - 0.3 * frac)
        gear = max(1, int(8 - frac * 6))
        drs = 0
    elif t < 0.40:
        # Slow corner
        frac = (t - 0.32) / 0.08
        speed = 80 + 40 * frac  # 80 → 120
        throttle = 30 + 40 * frac
        brake = 0.0
        gear = 3
        drs = 0
    elif t < 0.60:
        # Medium speed section
        frac = (t - 0.40) / 0.20
        speed = 120 + 160 * frac  # 120 → 280
        throttle = 80 + 20 * frac
        brake = 0.0
        gear = min(8, int(3 + frac * 5))
        drs = 0
    elif t < 0.80:
        # Back straight
        frac = (t - 0.60) / 0.20
        speed = 280 + 50 * frac  # 280 → 330
        throttle = 100.0
        brake = 0.0
        gear = 8
        drs = 1 if frac > 0.2 else 0
    elif t < 0.88:
        # Chicane entry braking
        frac = (t - 0.80) / 0.08
        speed = 330 - 180 * frac  # 330 → 150
        throttle = 0.0
        brake = 90 * (1 - 0.4 * frac)
        gear = max(2, int(8 - frac * 5))
        drs = 0
    else:
        # Chicane exit
        frac = (t - 0.88) / 0.12
        speed = 150 + 40 * frac  # 150 → 190 → run to start/finish
        throttle = 60 + 40 * frac
        brake = 0.0
        gear = min(5, int(2 + frac * 3))
        drs = 0

    speed = max(0, speed + noise + speed_offset) / deg
    rpm = _speed_to_rpm(speed, gear)

    return {
        "Speed": speed,
        "RPM": rpm,
        "Throttle": np.clip(throttle + rng.normal(0, 1), 0, 100),
        "Brake": max(0.0, brake + rng.normal(0, 0.5)),
        "nGear": gear,
        "DRS": drs,
        "LapNumber": lap,
        "Time": time_s,
    }


def _speed_to_rpm(speed_kmh: float, gear: int) -> float:
    """Convert speed + gear to approximate engine RPM."""
    if gear <= 0 or speed_kmh <= 0:
        return 4000.0  # idle
    speed_ms = speed_kmh / 3.6
    wheel_rps = speed_ms / (np.pi * WHEEL_DIAMETER)
    ratio = GEAR_RATIOS.get(gear, 1.0)
    # Final drive ratio ~3.0
    engine_rps = wheel_rps * ratio * 3.0
    rpm = engine_rps * 60.0
    return np.clip(rpm, 4000, 15000)


def _get(row, key, default=0.0):
    """Safely get a value from a dict or DataFrame row."""
    try:
        val = row[key]
        if hasattr(val, "item"):
            return val.item()
        return val
    except (KeyError, TypeError, IndexError):
        return default
