"""Tests for resilience metrics module."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from utils.metrics import ResilienceAnalyzer


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _make_synthetic(n_days=200, attack_start=50, attack_end=100,
                    baseline=1000.0, peak_factor=1.5, recovery_days=30):
    """Build a synthetic TSTT trajectory with known properties.

    - Pre-attack: constant baseline
    - Attack: linear ramp up to peak_factor * baseline
    - Post-attack: exponential decay back toward baseline
    """
    tstt = np.full(n_days, baseline)

    # Attack phase: linear ramp
    attack_len = attack_end - attack_start
    for i in range(attack_len):
        frac = (i + 1) / attack_len
        tstt[attack_start + i] = baseline * (1.0 + (peak_factor - 1.0) * frac)

    # Recovery phase: exponential decay
    peak = baseline * peak_factor
    for i in range(attack_end, n_days):
        decay = (peak - baseline) * np.exp(-3.0 * (i - attack_end) / recovery_days)
        tstt[i] = baseline + decay

    # Trust trajectory: 3 classes
    trust = np.full((n_days, 3), 0.85)
    for i in range(attack_start, attack_end):
        frac = (i - attack_start + 1) / attack_len
        trust[i] = 0.85 - 0.5 * frac
    for i in range(attack_end, n_days):
        decay = 0.5 * np.exp(-2.0 * (i - attack_end) / recovery_days)
        trust[i] = 0.85 - decay

    return tstt, trust


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic():
    tstt, trust = _make_synthetic()
    return ResilienceAnalyzer(tstt, trust, attack_start=50, attack_end=100)


@pytest.fixture(scope="module")
def no_attack():
    """Flat TSTT, no attack effect."""
    tstt = np.full(200, 1000.0)
    trust = np.full((200, 3), 0.85)
    return ResilienceAnalyzer(tstt, trust, attack_start=50, attack_end=100)


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

class TestPriceOfAttack:
    """Price of Attack metrics (sustained, peak, legacy)."""

    def test_sustained_poatt(self, synthetic):
        poa = synthetic.sustained_poatt()
        # Linear ramp from 1.0 to 1.5 baseline: mean = 1.255
        assert abs(poa - 1.255) < 0.02

    def test_peak_poatt(self, synthetic):
        poa = synthetic.peak_poatt()
        # Peak is 1.5 * baseline at end of attack ramp
        assert abs(poa - 1.5) < 0.01

    def test_price_of_attack_is_sustained(self, synthetic):
        assert synthetic.price_of_attack() == synthetic.sustained_poatt()

    def test_legacy_poatt(self, synthetic):
        poa = synthetic.price_of_attack_legacy()
        assert abs(poa - 1.5) < 0.01

    def test_transient_overshoot(self, synthetic):
        ratio = synthetic.transient_overshoot_ratio()
        # peak/sustained = 1.5/1.255 ~ 1.195
        assert 1.1 < ratio < 1.3

    def test_no_attack_poa_is_one(self, no_attack):
        assert abs(no_attack.price_of_attack() - 1.0) < 1e-10


class TestTimeToRecover:
    """Days after attack for TSTT to return within threshold of baseline."""

    def test_recovery_is_finite(self, synthetic):
        ttr = synthetic.time_to_recover(threshold=0.05)
        # Exponential decay with rate=3/30 => should recover in ~30 days
        assert 0 < ttr < 100

    def test_no_attack_immediate_recovery(self, no_attack):
        assert no_attack.time_to_recover() == 0


class TestResilienceTriangle:
    """Integral of excess TSTT, normalized by baseline."""

    def test_triangle_positive(self, synthetic):
        area = synthetic.resilience_triangle_area()
        assert area > 0

    def test_no_attack_triangle_zero(self, no_attack):
        assert no_attack.resilience_triangle_area() == 0.0


class TestRobustness:
    """1 - (max increase / baseline)."""

    def test_robustness_value(self, synthetic):
        rob = synthetic.robustness()
        # peak_factor = 1.5 => max_increase = 0.5*baseline => robustness = 0.5
        assert abs(rob - 0.5) < 0.01

    def test_no_attack_full_robustness(self, no_attack):
        assert abs(no_attack.robustness() - 1.0) < 1e-10


class TestTrustRecoveryDelay:
    """Days for mean trust to recover to near pre-attack level."""

    def test_trust_delay_finite(self, synthetic):
        delay = synthetic.trust_recovery_delay(threshold=0.10)
        assert delay >= 0

    def test_no_attack_immediate_trust(self, no_attack):
        assert no_attack.trust_recovery_delay() == 0


class TestHysteresisGap:
    """Permanent degradation at end of simulation."""

    def test_hysteresis_small_after_long_recovery(self, synthetic):
        gap = synthetic.hysteresis_gap()
        # After 100 days of exponential recovery, gap should be small
        assert gap < 0.05

    def test_no_attack_zero_hysteresis(self, no_attack):
        assert abs(no_attack.hysteresis_gap()) < 1e-10


class TestRecoveryAmplification:
    """time_to_recover / attack_duration."""

    def test_amplification_reasonable(self, synthetic):
        ratio = synthetic.recovery_amplification_ratio()
        # Recovery takes some fraction of the 50-day attack duration
        assert ratio >= 0

    def test_no_attack_zero_amplification(self, no_attack):
        assert no_attack.recovery_amplification_ratio() == 0.0


class TestSummary:
    """summary() returns all metrics."""

    def test_summary_keys(self, synthetic):
        s = synthetic.summary()
        expected_keys = {
            "sustained_poatt", "peak_poatt", "transient_overshoot_ratio",
            "price_of_attack", "time_to_recover", "resilience_triangle",
            "robustness", "trust_recovery_delay", "hysteresis_gap",
            "recovery_amplification_ratio",
        }
        assert set(s.keys()) == expected_keys

    def test_summary_values_finite(self, synthetic):
        s = synthetic.summary()
        for k, v in s.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"


class TestBaselineAutoDetect:
    """Baseline auto-detection from pre-attack mean."""

    def test_auto_baseline(self):
        tstt = np.concatenate([np.full(50, 800.0), np.full(150, 1200.0)])
        trust = np.full((200, 3), 0.8)
        ra = ResilienceAnalyzer(tstt, trust, attack_start=50, attack_end=100)
        assert abs(ra.baseline - 800.0) < 1e-10

    def test_explicit_baseline(self):
        tstt = np.full(200, 1000.0)
        trust = np.full((200, 3), 0.8)
        ra = ResilienceAnalyzer(tstt, trust, attack_start=50, attack_end=100,
                                baseline_tstt=900.0)
        assert abs(ra.baseline - 900.0) < 1e-10
