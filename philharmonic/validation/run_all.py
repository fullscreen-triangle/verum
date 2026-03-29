"""
Run all Philharmonic F1 validation tests and generate publication panels.

Usage:
    python -m philharmonic.validation.run_all
"""

import json
import os
import sys
import time
import traceback

# Ensure the project root is on the path
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)


def main():
    print("=" * 72)
    print("  PHILHARMONIC  —  F1 Telemetry Validation Suite")
    print("=" * 72)

    results = {}
    t0 = time.time()

    # ------------------------------------------------------------------
    # Test 1: State Reconstruction
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  TEST 1: State Reconstruction")
    print("-" * 60)
    try:
        from philharmonic.validation import state_reconstruction
        r1 = state_reconstruction.run(use_fastf1=True, verbose=True)
        results["state_reconstruction"] = _serialisable(r1)
        results["state_reconstruction"]["status"] = "PASS"
    except Exception as exc:
        print(f"  ERROR: {exc}")
        traceback.print_exc()
        results["state_reconstruction"] = {"status": "FAIL", "error": str(exc)}

    # ------------------------------------------------------------------
    # Test 2: Fault Prediction
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  TEST 2: Fault Prediction")
    print("-" * 60)
    try:
        from philharmonic.validation import fault_prediction
        r2 = fault_prediction.run(use_fastf1=True, verbose=True)
        results["fault_prediction"] = _serialisable(r2)
        results["fault_prediction"]["status"] = "PASS"
    except Exception as exc:
        print(f"  ERROR: {exc}")
        traceback.print_exc()
        results["fault_prediction"] = {"status": "FAIL", "error": str(exc)}

    # ------------------------------------------------------------------
    # Test 3: Tire Degradation
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  TEST 3: Tire Degradation")
    print("-" * 60)
    try:
        from philharmonic.validation import tire_degradation
        r3 = tire_degradation.run(use_fastf1=True, verbose=True)
        results["tire_degradation"] = _serialisable(r3)
        results["tire_degradation"]["status"] = "PASS"
    except Exception as exc:
        print(f"  ERROR: {exc}")
        traceback.print_exc()
        results["tire_degradation"] = {"status": "FAIL", "error": str(exc)}

    # ------------------------------------------------------------------
    # Test 4: Racing Line
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  TEST 4: Racing Line")
    print("-" * 60)
    try:
        from philharmonic.validation import racing_line
        r4 = racing_line.run(use_fastf1=True, verbose=True)
        results["racing_line"] = _serialisable(r4)
        results["racing_line"]["status"] = "PASS"
    except Exception as exc:
        print(f"  ERROR: {exc}")
        traceback.print_exc()
        results["racing_line"] = {"status": "FAIL", "error": str(exc)}

    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    for test_name, res in results.items():
        status = res.get("status", "UNKNOWN")
        print(f"  {test_name:30s}  [{status}]")
    print(f"\n  Total time: {elapsed:.1f} s")

    # Save results JSON
    out_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "validation_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to: {json_path}")

    # ------------------------------------------------------------------
    # Generate panels
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  Generating publication panels...")
    print("-" * 60)
    try:
        from philharmonic.figures import generate_panels
        generate_panels.generate_all(results)
        print("  Panels generated successfully.")
    except Exception as exc:
        print(f"  Panel generation error: {exc}")
        traceback.print_exc()

    print("\n" + "=" * 72)
    print("  DONE")
    print("=" * 72)


def _serialisable(obj):
    """Make a result dict JSON-serialisable (convert numpy arrays, etc.)."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _serialisable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialisable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj


if __name__ == "__main__":
    main()
