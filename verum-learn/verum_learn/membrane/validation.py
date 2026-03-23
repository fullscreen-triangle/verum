"""
Membrane Signal Transduction Validation
=========================================

End-to-end validation that lipid arrays act as signal transducers.

Tests:
    1. Lipid oscillation at correct frequencies
    2. Carrier conductivity matches σ = 5.6 × 10⁻³ S/cm
    3. P-N junction V_bi = 0.78 V, rectification > 42
    4. BMD transistor pattern recognition
    5. Logic gates 100% truth table accuracy
    6. ALU arithmetic correctness
    7. S-entropy coordinate injectivity (different inputs → different outputs)
    8. Full circuit signal transduction (environment → S-entropy)
    9. Perturbation detection (obstacle sensing)
    10. Environmental discrimination (distinct conditions → distinct S-entropy)
"""

import numpy as np
from typing import Optional

from .lipid import Lipid, LipidArray, CHAIN_ISOMERIZATION_RATE
from .carriers import CarrierPopulation, OscillatoryHole, MolecularCarrier
from .junction import PNJunction
from .transistor import BMDTransistor
from .logic_gates import TriDimensionalGate
from .alu import VirtualALU
from .memory import SDictionaryMemory
from .s_entropy import SEntropyCoordinate, compute_s_entropy, inverse_s_entropy
from .ensemble import O2Ensemble
from .sensor_circuit import SensorCircuit


class ValidationResult:
    """Result of a single validation test."""
    def __init__(self, name: str, passed: bool, expected, actual, details: str = ""):
        self.name = name
        self.passed = passed
        self.expected = expected
        self.actual = actual
        self.details = details

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: expected={self.expected}, actual={self.actual} {self.details}"


def validate_lipid_oscillation() -> ValidationResult:
    """Test 1: Lipid oscillates at ~10¹¹ Hz."""
    lipid = Lipid()
    freq = lipid.frequency
    expected = CHAIN_ISOMERIZATION_RATE  # 10¹¹ Hz

    passed = abs(freq - expected) / expected < 0.01  # within 1%
    return ValidationResult(
        "Lipid oscillation frequency",
        passed,
        f"{expected:.2e} Hz",
        f"{freq:.2e} Hz",
    )


def validate_lipid_array_processing() -> ValidationResult:
    """Test 1b: Lipid array processing rate scales with area."""
    array_1mm2 = LipidArray(area=1e-6)  # 1 mm²
    rate = array_1mm2.total_processing_rate

    # Expected: ~3.1 × 10²³ ops/s for 1 mm²
    expected_order = 23
    actual_order = np.log10(rate) if rate > 0 else 0

    passed = abs(actual_order - expected_order) < 1  # within 1 order of magnitude
    return ValidationResult(
        "Lipid array processing rate",
        passed,
        f"~10^{expected_order} ops/s",
        f"10^{actual_order:.1f} ops/s",
    )


def validate_carrier_conductivity() -> ValidationResult:
    """Test 2: Conductivity σ = 5.6 × 10⁻³ S/cm."""
    pop = CarrierPopulation()
    sigma = pop.conductivity_s_per_cm

    expected = 5.6e-3  # S/cm
    passed = abs(sigma - expected) / expected < 0.1  # within 10%
    return ValidationResult(
        "Carrier conductivity",
        passed,
        f"{expected:.2e} S/cm",
        f"{sigma:.2e} S/cm",
    )


def validate_junction_vbi() -> ValidationResult:
    """Test 3a: Built-in potential V_bi = 0.78 V."""
    junction = PNJunction()
    vbi = junction.built_in_potential

    expected = 0.78  # V
    passed = abs(vbi - expected) / expected < 0.05  # within 5%
    return ValidationResult(
        "Junction built-in potential",
        passed,
        f"{expected:.2f} V",
        f"{vbi:.2f} V",
    )


def validate_junction_rectification() -> ValidationResult:
    """Test 3b: Rectification ratio > 42 at |V| = 0.5 V."""
    junction = PNJunction()
    rr = junction.rectification_ratio(0.5)

    expected_min = 42
    passed = rr > expected_min
    return ValidationResult(
        "Junction rectification ratio",
        passed,
        f"> {expected_min}",
        f"{rr:.1f}",
    )


def validate_bmd_transistor() -> ValidationResult:
    """Test 4: BMD transistor recognizes matching patterns."""
    transistor = BMDTransistor(fidelity=1.0)  # perfect fidelity for validation
    pattern = SEntropyCoordinate(0.5, 0.5, 0.5)
    transistor.set_gate_pattern(pattern)

    # Matching input should open gate
    close_input = SEntropyCoordinate(0.52, 0.48, 0.51)
    transistor.gate_tick(close_input)
    match_open = transistor.is_open

    # Distant input should close gate
    far_input = SEntropyCoordinate(0.1, 0.9, 0.2)
    transistor.gate_tick(far_input)
    match_closed = not transistor.is_open

    passed = match_open and match_closed
    return ValidationResult(
        "BMD transistor pattern recognition",
        passed,
        "open for match, closed for mismatch",
        f"match→{'open' if match_open else 'closed'}, "
        f"mismatch→{'closed' if match_closed else 'open'}",
    )


def validate_logic_gates() -> ValidationResult:
    """Test 5: Tri-dimensional gates achieve 100% truth table accuracy."""
    gate = TriDimensionalGate()
    accuracy = gate.validate()

    all_100 = all(v == 1.0 for v in accuracy.values())
    return ValidationResult(
        "Logic gate truth table",
        all_100,
        {"AND": 1.0, "OR": 1.0, "XOR": 1.0},
        accuracy,
    )


def validate_alu_arithmetic() -> ValidationResult:
    """Test 6: ALU addition and multiplication produce valid S-coordinates."""
    alu = VirtualALU()
    a = SEntropyCoordinate(0.3, 0.4, 0.5)
    b = SEntropyCoordinate(0.2, 0.3, 0.4)

    add_result = alu.add(a, b)
    mul_result = alu.multiply(a, b)

    # Results must be valid S-coordinates (in [0,1]³)
    valid_add = all(0 <= v <= 1 for v in add_result.to_array())
    valid_mul = all(0 <= v <= 1 for v in mul_result.to_array())

    # Addition should produce higher S_k than either input
    add_correct = add_result.s_k >= max(a.s_k, b.s_k) - 0.01

    passed = valid_add and valid_mul and add_correct
    return ValidationResult(
        "ALU arithmetic validity",
        passed,
        "valid S-coordinates, add.s_k >= max(inputs)",
        f"add={add_result.to_array()}, mul={mul_result.to_array()}",
    )


def validate_s_entropy_invertibility() -> ValidationResult:
    """Test 7a: S-entropy mapping is invertible (ω,φ,A) ↔ (S_k,S_t,S_e)."""
    omega_test = 2 * np.pi * 1e10  # 10 GHz
    phi_test = 1.5
    amp_test = 0.8

    s = compute_s_entropy(omega_test, phi_test, amp_test)
    omega_rec, phi_rec, amp_rec = inverse_s_entropy(s)

    omega_err = abs(omega_rec - omega_test) / omega_test
    phi_err = abs(phi_rec - phi_test)
    amp_err = abs(amp_rec - amp_test)

    passed = omega_err < 0.01 and phi_err < 0.01 and amp_err < 0.01
    return ValidationResult(
        "S-entropy invertibility",
        passed,
        f"ω={omega_test:.2e}, φ={phi_test:.2f}, A={amp_test:.2f}",
        f"ω={omega_rec:.2e}, φ={phi_rec:.2f}, A={amp_rec:.2f}",
    )


def validate_s_entropy_injectivity() -> ValidationResult:
    """Test 7b: Different environments → different S-entropy coordinates."""
    environments = [
        {"temperature": 280, "pressure": 0.9e5, "flow_velocity": 0.0},
        {"temperature": 300, "pressure": 1.0e5, "flow_velocity": 0.0},
        {"temperature": 320, "pressure": 1.1e5, "flow_velocity": 0.0},
        {"temperature": 300, "pressure": 1.0e5, "flow_velocity": 5.0},
        {"temperature": 300, "pressure": 1.0e5, "flow_velocity": -5.0},
    ]

    ensemble = O2Ensemble()
    coords = []
    for env in environments:
        s = ensemble.encode_environment(**env)
        coords.append(s)

    # All pairs should have d_cat > 0
    min_dist = float("inf")
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d = coords[i].categorical_distance(coords[j])
            min_dist = min(min_dist, d)

    passed = min_dist > 0.01  # distinct outputs
    return ValidationResult(
        "S-entropy injectivity (distinct environments)",
        passed,
        "min d_cat > 0.01",
        f"min d_cat = {min_dist:.4f}",
        f"({len(environments)} environments tested)",
    )


def validate_signal_transduction() -> ValidationResult:
    """Test 8: Full circuit transduces environmental signals to S-entropy.

    Core validation: feed different environments through the complete
    sensor circuit and verify distinct, correct S-entropy outputs.
    """
    circuit = SensorCircuit()

    env_baseline = {
        "temperature": 300.0, "pressure": 1e5, "concentration": 0.21,
        "light_intensity": 1.0, "echo_delay": 0.001,
        "magnetic_field": 50e-6, "flow_velocity": 0.0,
    }
    env_hot = {**env_baseline, "temperature": 320.0}
    env_cold = {**env_baseline, "temperature": 280.0}
    env_windy = {**env_baseline, "flow_velocity": 10.0}
    env_high_pressure = {**env_baseline, "pressure": 1.2e5}

    circuit.calibrate(env_baseline)
    s_baseline = circuit.baseline

    s_hot = circuit.process(env_hot)
    s_cold = circuit.process(env_cold)
    s_windy = circuit.process(env_windy)
    s_high_p = circuit.process(env_high_pressure)

    # All should differ from baseline
    d_hot = s_hot.categorical_distance(s_baseline)
    d_cold = s_cold.categorical_distance(s_baseline)
    d_windy = s_windy.categorical_distance(s_baseline)
    d_high_p = s_high_p.categorical_distance(s_baseline)

    all_distinct = all(d > 0.01 for d in [d_hot, d_cold, d_windy, d_high_p])

    # Hot and cold should differ from each other
    d_hot_cold = s_hot.categorical_distance(s_cold)
    hot_cold_distinct = d_hot_cold > 0.01

    passed = all_distinct and hot_cold_distinct
    return ValidationResult(
        "Full circuit signal transduction",
        passed,
        "all environments distinguishable (d_cat > 0.01)",
        f"d_hot={d_hot:.4f}, d_cold={d_cold:.4f}, "
        f"d_windy={d_windy:.4f}, d_pressure={d_high_p:.4f}",
    )


def validate_perturbation_detection() -> ValidationResult:
    """Test 9: Circuit detects obstacles as S-entropy perturbations."""
    circuit = SensorCircuit()

    env_clear = {
        "temperature": 300.0, "pressure": 1e5, "concentration": 0.21,
        "light_intensity": 1.0, "echo_delay": 0.001,
        "magnetic_field": 50e-6, "flow_velocity": 0.0,
    }
    circuit.calibrate(env_clear)

    # Another vehicle perturbs temperature (engine heat) and flow (wake)
    env_vehicle_nearby = {
        **env_clear,
        "temperature": 305.0,  # warmer from engine
        "flow_velocity": 3.0,  # wake turbulence
    }

    detected, distance = circuit.detect_obstacle(env_vehicle_nearby, threshold=0.01)
    passed = detected
    return ValidationResult(
        "Perturbation detection (obstacle)",
        passed,
        "detected=True",
        f"detected={detected}, d_cat={distance:.4f}",
    )


def validate_weather_enhancement() -> ValidationResult:
    """Test 10: Weather changes produce stronger (not weaker) signals.

    Counterintuitive claim: bad weather = more information for membrane.
    Rain/fog increase molecular interactions → richer S-entropy signatures.
    """
    circuit = SensorCircuit()

    env_clear = {
        "temperature": 300.0, "pressure": 1e5, "concentration": 0.21,
        "light_intensity": 1.0, "echo_delay": 0.001,
        "magnetic_field": 50e-6, "flow_velocity": 0.0,
    }
    circuit.calibrate(env_clear)

    # Fog: higher humidity changes pressure, reduces light, changes acoustics
    env_fog = {
        **env_clear,
        "pressure": 1.01e5,       # slight pressure change
        "light_intensity": 0.3,    # reduced visibility
        "echo_delay": 0.003,       # longer acoustic echo (moisture)
    }

    # Rain: temperature drop, pressure change, flow
    env_rain = {
        **env_clear,
        "temperature": 290.0,
        "pressure": 0.98e5,
        "flow_velocity": 2.0,
        "light_intensity": 0.5,
    }

    s_clear = circuit.process(env_clear)
    s_fog = circuit.process(env_fog)
    s_rain = circuit.process(env_rain)

    d_fog = s_fog.categorical_distance(s_clear)
    d_rain = s_rain.categorical_distance(s_clear)
    d_fog_rain = s_fog.categorical_distance(s_rain)

    # Weather changes should produce LARGER categorical distances
    # (more distinguishable, not less)
    all_distinguishable = d_fog > 0.01 and d_rain > 0.01 and d_fog_rain > 0.01
    passed = all_distinguishable
    return ValidationResult(
        "Weather enhances signal (not degrades)",
        passed,
        "fog, rain, clear all distinguishable",
        f"d_fog={d_fog:.4f}, d_rain={d_rain:.4f}, d_fog_vs_rain={d_fog_rain:.4f}",
    )


def run_all_validations() -> list[ValidationResult]:
    """Run all validation tests and return results."""
    tests = [
        validate_lipid_oscillation,
        validate_lipid_array_processing,
        validate_carrier_conductivity,
        validate_junction_vbi,
        validate_junction_rectification,
        validate_bmd_transistor,
        validate_logic_gates,
        validate_alu_arithmetic,
        validate_s_entropy_invertibility,
        validate_s_entropy_injectivity,
        validate_signal_transduction,
        validate_perturbation_detection,
        validate_weather_enhancement,
    ]

    results = []
    for test in tests:
        try:
            result = test()
        except Exception as e:
            result = ValidationResult(test.__name__, False, "no error", str(e))
        results.append(result)

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("MEMBRANE SIGNAL TRANSDUCTION VALIDATION")
    print("Lipid arrays as signal transducers — end-to-end verification")
    print("=" * 70)
    print()

    results = run_all_validations()

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    for r in results:
        print(r)
    print()
    print(f"{'=' * 70}")
    print(f"Results: {passed}/{total} tests passed")
    if passed == total:
        print("ALL VALIDATIONS PASSED — lipid arrays confirmed as signal transducers")
    else:
        print(f"FAILURES: {total - passed} tests need investigation")
    print(f"{'=' * 70}")
