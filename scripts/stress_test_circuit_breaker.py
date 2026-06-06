"""
Stress Test: Circuit Breaker & Anomaly Detector Validation (Tahap 6)
=====================================================================
Validates that the full safety-net pipeline responds correctly when
subjected to synthetic market crises:

  Test 1  - VIX Extreme Panic  : vix_percentile_252d forced to 0.999
  Test 2  - Asset Flash Crash  : return_zscore_90d forced to -4.0
  Test 3  - Asset Spike        : return_zscore_90d forced to +4.0
  Test 4  - Covid Black Swan   : VIX=0.99, Z=-5, Corr=-0.7 simultaneously
  Test 5  - Euphoric Bull      : VIX=0.05, Z=+0.3, Corr=+0.9 (low-risk)
  Test 6  - AnomalyDetector    : detect_asset_anomaly() self-check
  Test 7  - DynamicStrength    : CorrelationEnforcer dynamic tiers
  Test 8  - MonotonicityCheck  : Divergence flag when corr < -0.3

Run:
    (.venv) python scripts/stress_test_circuit_breaker.py
"""

import sys
import os
import numpy as np
import pandas as pd

# Ensure project root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.anomaly_detector import detect_asset_anomaly
from utils.correlation_enforcer import CorrelationEnforcer

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"

passed = 0
failed = 0


def run_test(name: str, condition: bool, detail: str = ""):
    global passed, failed
    status = PASS if condition else FAIL
    print(f"  {status}  {name}")
    if detail:
        print(f"         {detail}")
    if condition:
        passed += 1
    else:
        failed += 1
    return condition


def _synthetic_price_series(n: int = 300, seed: int = 42,
                             crash_end: bool = False,
                             spike_end: bool = False) -> pd.Series:
    """Generate a synthetic close price series with optional tail event."""
    np.random.seed(seed)
    rets = np.random.normal(0.001, 0.015, n)
    if crash_end:
        rets[-1] = -0.12   # -12% flash crash
    if spike_end:
        rets[-1] = +0.14   # +14% spike
    prices = 100 * np.cumprod(1 + rets)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(prices, index=idx, name="PRICE")


def _synthetic_feature_row(vix_pct: float = 0.5,
                             z_score: float = 0.0,
                             roll_corr: float = 0.5) -> dict:
    """Build a minimal feature dict for injection tests."""
    return {
        "vix_percentile_252d": vix_pct,
        "return_zscore_90d": z_score,
        "roll_corr_spy_90d": roll_corr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 - VIX Extreme Panic Regime
# ─────────────────────────────────────────────────────────────────────────────

def test_vix_extreme_panic():
    print("\n" + "="*60)
    print("TEST 1 - VIX Extreme Panic (vix_percentile = 0.999)")
    print("="*60)

    row = _synthetic_feature_row(vix_pct=0.999, z_score=0.5, roll_corr=0.6)
    vix = row["vix_percentile_252d"]

    # Regime classification logic (mirrors Dashboard display)
    if vix >= 0.90:
        regime = "EXTREME"
    elif vix >= 0.70:
        regime = "ELEVATED"
    else:
        regime = "CALM"

    run_test("VIX percentile=0.999 => regime=EXTREME",
             regime == "EXTREME",
             f"regime={regime}, vix_pct={vix}")

    run_test("VIX percentile is in valid [0.0, 1.0] range",
             0.0 <= vix <= 1.0,
             f"value={vix}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 - Flash Crash Detection (Z-Score -4 sigma)
# ─────────────────────────────────────────────────────────────────────────────

def test_flash_crash_detection():
    print("\n" + "="*60)
    print("TEST 2 - Flash Crash Detection (return_zscore_90d approx -4.0)")
    print("="*60)

    prices = _synthetic_price_series(n=300, crash_end=True)
    df = pd.DataFrame({"BTC": prices})

    result = detect_asset_anomaly("btc", df, price_col="BTC")

    print(f"  {INFO}  z_score={result['z_score']:.3f}, "
          f"regime={result['regime']}, "
          f"direction={result['direction']}, "
          f"threshold={result['threshold']}")

    run_test("detect_asset_anomaly fires is_anomaly=True",
             result["is_anomaly"] is True,
             f"is_anomaly={result['is_anomaly']}")

    run_test("Direction is CRASH (negative z_score)",
             result["direction"] == "CRASH",
             f"direction={result['direction']}")

    run_test("Regime is ANOMALY",
             result["regime"] == "ANOMALY",
             f"regime={result['regime']}")

    run_test("Z-Score < -2.0 (sufficiently extreme)",
             result["z_score"] < -2.0,
             f"z_score={result['z_score']:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 - Euphoric Spike Detection (Z-Score +4 sigma)
# ─────────────────────────────────────────────────────────────────────────────

def test_spike_detection():
    print("\n" + "="*60)
    print("TEST 3 - Spike Detection (return_zscore_90d approx +4.0)")
    print("="*60)

    prices = _synthetic_price_series(n=300, spike_end=True)
    df = pd.DataFrame({"GOLD": prices})

    result = detect_asset_anomaly("gold", df, price_col="GOLD")

    print(f"  {INFO}  z_score={result['z_score']:.3f}, "
          f"regime={result['regime']}, "
          f"direction={result['direction']}")

    run_test("detect_asset_anomaly fires is_anomaly=True",
             result["is_anomaly"] is True,
             f"is_anomaly={result['is_anomaly']}")

    run_test("Direction is SPIKE (positive z_score)",
             result["direction"] == "SPIKE",
             f"direction={result['direction']}")

    run_test("Z-Score > +2.0",
             result["z_score"] > 2.0,
             f"z_score={result['z_score']:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 - Covid Black Swan Simulation (Combined Worst-Case)
# ─────────────────────────────────────────────────────────────────────────────

def test_black_swan_combined():
    print("\n" + "="*60)
    print("TEST 4 - Covid Black Swan (VIX=0.99, Z=-5.0, Corr=-0.7)")
    print("="*60)

    row = _synthetic_feature_row(vix_pct=0.99, z_score=-5.0, roll_corr=-0.7)

    # 1) VIX regime
    vix = row["vix_percentile_252d"]
    vix_regime = "EXTREME" if vix >= 0.90 else "ELEVATED" if vix >= 0.70 else "CALM"
    run_test("VIX regime=EXTREME at 99th percentile",
             vix_regime == "EXTREME", f"regime={vix_regime}")

    # 2) Z-Score anomaly flag
    z = row["return_zscore_90d"]
    z_anomaly = abs(z) > 3.0
    run_test("Z-Score anomaly flag fires at |z|>3",
             z_anomaly, f"|z|={abs(z):.1f}")

    # 3) Dynamic correlation strength: at corr=-0.7 => strength=MINIMUM (0.20)
    enforcer = CorrelationEnforcer.__new__(CorrelationEnforcer)
    enforcer.betas = {}
    enforcer.correlations = {}
    strength = enforcer._compute_dynamic_strength(row["roll_corr_spy_90d"])
    run_test("Dynamic strength = 0.20 (minimum) when corr=-0.7",
             strength == 0.20, f"strength={strength:.2f}")

    # 4) Monotonicity check: BTC predicted +10% while corr=-0.7 => DIVERGENCE_WARNING
    enforcer.reference_ticker = "SPY"
    spy_prices = [500.0, 495.0]          # SPY going DOWN
    btc_prices = [65000.0, 71500.0]      # BTC going UP +10% -- contradictory!
    predictions = {"SPY": spy_prices, "BTC": btc_prices}

    labels = enforcer.monotonicity_check(predictions,
                                          roll_corr_90d=row["roll_corr_spy_90d"],
                                          divergence_threshold=0.05)
    run_test("MonotonicityCheck flags BTC as HIGH_UNCERTAINTY or DIVERGENCE_WARNING",
             labels.get("BTC") in ("HIGH_UNCERTAINTY", "DIVERGENCE_WARNING"),
             f"BTC label={labels.get('BTC')}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 - Euphoric Bull (Low Risk Baseline)
# ─────────────────────────────────────────────────────────────────────────────

def test_euphoric_bull():
    print("\n" + "="*60)
    print("TEST 5 - Euphoric Bull (VIX=0.05, Z=+0.3, Corr=+0.92)")
    print("="*60)

    row = _synthetic_feature_row(vix_pct=0.05, z_score=0.3, roll_corr=0.92)

    vix = row["vix_percentile_252d"]
    vix_regime = "EXTREME" if vix >= 0.90 else "ELEVATED" if vix >= 0.70 else "CALM"
    run_test("VIX regime=CALM when vix_pct=0.05",
             vix_regime == "CALM", f"regime={vix_regime}")

    z = row["return_zscore_90d"]
    z_anomaly = abs(z) > 3.0
    run_test("No anomaly at Z=+0.3",
             not z_anomaly, f"|z|={abs(z):.1f}")

    enforcer = CorrelationEnforcer.__new__(CorrelationEnforcer)
    strength = enforcer._compute_dynamic_strength(row["roll_corr_spy_90d"])
    run_test("Dynamic strength=0.85 (maximum) at corr=+0.92",
             strength == 0.85, f"strength={strength:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 - AnomalyDetector: Normal Conditions (No Anomaly)
# ─────────────────────────────────────────────────────────────────────────────

def test_no_anomaly_normal():
    print("\n" + "="*60)
    print("TEST 6 - AnomalyDetector: Normal Price Action (No Anomaly)")
    print("="*60)

    prices = _synthetic_price_series(n=300, crash_end=False, spike_end=False)
    df = pd.DataFrame({"SPY": prices})

    result = detect_asset_anomaly("SPY", df, price_col="SPY")

    print(f"  {INFO}  z_score={result['z_score']:.3f}, regime={result['regime']}")

    run_test("is_anomaly=False under normal conditions",
             result["is_anomaly"] is False,
             f"is_anomaly={result['is_anomaly']}, regime={result['regime']}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7 - CorrelationEnforcer: All 4 Dynamic Strength Tiers
# ─────────────────────────────────────────────────────────────────────────────

def test_dynamic_strength_tiers():
    print("\n" + "="*60)
    print("TEST 7 - Dynamic Strength: All 4 Correlation Tiers")
    print("="*60)

    enforcer = CorrelationEnforcer.__new__(CorrelationEnforcer)

    cases = [
        (0.90,  0.85, "Very Strong coupling (corr>=0.80)"),
        (0.60,  0.60, "Strong coupling (corr 0.50-0.79)"),
        (0.40,  0.35, "Weak coupling (corr 0.30-0.49)"),
        (0.10,  0.20, "Decoupling (corr<0.30)"),
        (-0.50, 0.20, "Inverse regime (corr<0.30)"),
    ]

    for corr, expected_strength, label in cases:
        strength = enforcer._compute_dynamic_strength(corr)
        run_test(f"{label}: strength={expected_strength:.2f}",
                 abs(strength - expected_strength) < 1e-6,
                 f"got={strength:.2f}, expected={expected_strength:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 8 - Monotonicity: NORMAL when corr is positive
# ─────────────────────────────────────────────────────────────────────────────

def test_monotonicity_positive_corr():
    print("\n" + "="*60)
    print("TEST 8 - MonotonicityCheck: No Warning when corr is Positive")
    print("="*60)

    enforcer = CorrelationEnforcer.__new__(CorrelationEnforcer)
    enforcer.reference_ticker = "SPY"
    enforcer.betas = {}

    # Both going UP -- consistent with positive correlation
    predictions = {
        "SPY": [500.0, 510.0],
        "BTC": [65000.0, 69000.0],
    }

    labels = enforcer.monotonicity_check(predictions, roll_corr_90d=0.75)

    run_test("BTC=NORMAL when corr=+0.75 and both move UP",
             labels.get("BTC") == "NORMAL",
             f"label={labels.get('BTC')}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  STRESS TEST SUITE - Market Intelligence Circuit Breakers")
    print("  Tahap 6: Scenario Simulator & Stress Testing")
    print("=" * 60)

    test_vix_extreme_panic()
    test_flash_crash_detection()
    test_spike_detection()
    test_black_swan_combined()
    test_euphoric_bull()
    test_no_anomaly_normal()
    test_dynamic_strength_tiers()
    test_monotonicity_positive_corr()

    print("\n" + "=" * 60)
    total = passed + failed
    pct = (passed / total * 100) if total > 0 else 0
    print(f"  RESULTS: {passed}/{total} tests passed ({pct:.0f}%)")
    if failed == 0:
        print("  All circuit breaker tests PASSED")
    else:
        print(f"  {failed} test(s) FAILED -- review output above")
    print("=" * 60 + "\n")

    sys.exit(0 if failed == 0 else 1)
