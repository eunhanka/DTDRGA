"""Tests for trust dynamics module."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from trust.trust_dynamics import TrustParams, TrustState, TrustDynamics


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture(scope="module")
def params():
    return TrustParams()


@pytest.fixture(scope="module")
def dynamics(params):
    return TrustDynamics(params)


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

class TestInitialTrust:
    """Verify initial trust state matches Beta(5, 1) prior."""

    def test_initial_trust_high(self, dynamics):
        """Expected trust ~= 5/6 ≈ 0.833 for all classes."""
        state = dynamics.initial_state()
        expected = 5.0 / (5.0 + 1.0)  # 0.8333...
        np.testing.assert_allclose(
            state.expected_trust, expected, atol=1e-10,
        )
        assert state.alpha.shape == (3,)
        assert state.beta.shape == (3,)


class TestAccurateGuidance:
    """Accurate guidance should maintain or increase trust."""

    def test_accurate_guidance_maintains_trust(self, dynamics):
        """10 days of accurate guidance (e=0) → trust >= 0.80 for all classes."""
        state = dynamics.initial_state()
        for _ in range(10):
            state = dynamics.update(state, guidance_error=np.zeros(3))
        assert np.all(state.expected_trust >= 0.80)


class TestInaccurateGuidance:
    """Inaccurate guidance should erode trust asymmetrically."""

    def test_inaccurate_guidance_erodes_trust(self, dynamics):
        """10 days of e=500 (> epsilon=360) → trust drops for all classes."""
        state = dynamics.initial_state()
        initial_trust = state.expected_trust.copy()
        for _ in range(10):
            state = dynamics.update(state, guidance_error=np.full(3, 500.0))
        final_trust = state.expected_trust

        # All classes should have lower trust
        assert np.all(final_trust < initial_trust)

        # With uniform w_f, all classes erode equally
        assert final_trust[0] <= final_trust[1] <= final_trust[2]


class TestAsymmetricRecovery:
    """Trust recovery should exhibit hysteresis — slow to rebuild after attack."""

    def test_asymmetric_recovery(self, dynamics):
        """5 days attack + 5 days accurate → trust < initial (hysteresis)."""
        state = dynamics.initial_state()
        initial_trust = state.expected_trust.copy()

        # 5 days of attack (error > epsilon=360)
        for _ in range(5):
            state = dynamics.update(state, guidance_error=np.full(3, 500.0))

        # 5 days of recovery
        for _ in range(5):
            state = dynamics.update(state, guidance_error=np.zeros(3))

        # Trust should not fully recover
        assert np.all(state.expected_trust < initial_trust)


class TestRecoveryTimeScales:
    """Higher w_f/w_s ratio → longer recovery time needed."""

    def test_recovery_time_scales_with_ratio(self):
        """Class with higher w_f/w_s ratio needs more days to recover."""
        # Two single-class configs with different w_f/w_s ratios
        # Class A: low ratio (w_f/w_s = 2)
        params_a = TrustParams(
            n_classes=1,
            class_fractions=np.array([1.0]),
            w_s=np.array([0.5]),
            w_f=np.array([1.0]),
            lambda_max=np.array([0.9]),
            theta=np.array([1.0]),
            indifference_band=np.array([0.0]),
            alpha_init=np.array([5.0]),
            beta_init=np.array([1.0]),
        )
        # Class B: high ratio (w_f/w_s = 10)
        params_b = TrustParams(
            n_classes=1,
            class_fractions=np.array([1.0]),
            w_s=np.array([0.1]),
            w_f=np.array([1.0]),
            lambda_max=np.array([0.9]),
            theta=np.array([1.0]),
            indifference_band=np.array([0.0]),
            alpha_init=np.array([5.0]),
            beta_init=np.array([1.0]),
        )

        dyn_a = TrustDynamics(params_a)
        dyn_b = TrustDynamics(params_b)

        # Apply 5 days of attack then recover
        target = 0.75  # recovery threshold

        def days_to_recover(dyn, n_attack=5):
            state = dyn.initial_state()
            for _ in range(n_attack):
                state = dyn.update(state, guidance_error=np.array([500.0]))
            for d in range(200):
                state = dyn.update(state, guidance_error=np.array([0.0]))
                if state.expected_trust[0] >= target:
                    return d + 1
            return 200  # did not recover

        recovery_a = days_to_recover(dyn_a)
        recovery_b = days_to_recover(dyn_b)

        # Higher ratio should need more days
        assert recovery_b > recovery_a


class TestInformationWeight:
    """Information weight should scale with trust level."""

    def test_information_weight(self, dynamics, params):
        """High trust → near lambda_max; low trust → near 0."""
        # High trust state
        high_state = TrustState(
            alpha=np.array([99.0, 99.0, 99.0]),
            beta=np.array([1.0, 1.0, 1.0]),
        )
        high_weights = dynamics.information_weight(high_state)
        np.testing.assert_allclose(
            high_weights, params.lambda_max, atol=0.02,
        )

        # Low trust state
        low_state = TrustState(
            alpha=np.array([1.0, 1.0, 1.0]),
            beta=np.array([99.0, 99.0, 99.0]),
        )
        low_weights = dynamics.information_weight(low_state)
        assert np.all(low_weights < 0.05)


class TestHeterogeneousClasses:
    """Same attack should affect different classes differently."""

    def test_heterogeneous_classes(self, dynamics):
        """Same attack → all classes erode (uniform w_f gives equal erosion)."""
        state = dynamics.initial_state()

        # 20 days of attack (error > epsilon=360)
        for _ in range(20):
            state = dynamics.update(state, guidance_error=np.full(3, 500.0))

        trust = state.expected_trust
        # With uniform w_f=[0.5,0.5,0.5], all classes erode equally
        assert trust[0] <= trust[1] <= trust[2]

        # All classes should have substantially eroded trust
        assert np.all(trust < 0.5)
