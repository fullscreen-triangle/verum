"""
Run All Molecular Navigation Validation Experiments
=====================================================

Executes all 5 experiments and reports results.
"""

from . import exp1_night_navigation
from . import exp2_brake_anticipation
from . import exp3_around_corner
from . import exp4_sweet_spot
from . import exp5_convoy_formation


def run_all() -> list[dict]:
    """Run all experiments and return results."""
    experiments = [
        ("Exp 1", exp1_night_navigation.run),
        ("Exp 2", exp2_brake_anticipation.run),
        ("Exp 3", exp3_around_corner.run),
        ("Exp 4", exp4_sweet_spot.run),
        ("Exp 5", exp5_convoy_formation.run),
    ]

    results = []
    print("=" * 70)
    print("MOLECULAR NAVIGATION VALIDATION")
    print("Virtual Gas Ensemble Experiments")
    print("=" * 70)
    print()

    for label, run_fn in experiments:
        try:
            result = run_fn()
        except Exception as e:
            result = {"name": label, "passed": False, "error": str(e)}
        results.append(result)

        status = "PASS" if result.get("passed") else "FAIL"
        name = result.get("name", label)
        print(f"[{status}] {name}")

        # Print key metrics
        for k, v in result.items():
            if k not in ("name", "passed", "detail", "error"):
                print(f"       {k}: {v}")
        print()

    passed = sum(1 for r in results if r.get("passed"))
    total = len(results)

    print("=" * 70)
    print(f"Results: {passed}/{total} experiments passed")
    if passed == total:
        print("ALL EXPERIMENTS PASSED")
        print("Molecular navigation validated via virtual gas ensembles")
    else:
        print(f"FAILURES: {total - passed}")
        for r in results:
            if not r.get("passed"):
                print(f"  -> {r.get('name', '?')}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_all()
