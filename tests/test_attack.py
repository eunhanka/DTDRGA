"""Tests for attack models module."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from attack.attack_models import (
    AttackConfig,
    NoAttack,
    ConstantFDI,
    AsymmetricFDI,
    EscalatingAttack,
    IntermittentAttack,
    TargetedBottleneckAttack,
    StrategicSybilAttack,
)
import scipy.sparse as sp


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture(scope="module")
def true_costs():
    """Synthetic true path costs [n_paths=20, n_windows=5]."""
    rng = np.random.default_rng(42)
    return rng.uniform(100, 500, size=(20, 5))


@pytest.fixture(scope="module")
def od_path_set():
    """4 OD pairs, 5 paths each."""
    return [
        np.array([0, 1, 2, 3, 4]),
        np.array([5, 6, 7, 8, 9]),
        np.array([10, 11, 12, 13, 14]),
        np.array([15, 16, 17, 18, 19]),
    ]


@pytest.fixture(scope="module")
def path_link_matrix():
    """Synthetic path-link matrix [n_paths=20, max_links=4]."""
    plm = -np.ones((20, 4), dtype=np.intp)
    for i in range(20):
        plm[i, 0] = i % 10
        if i >= 5:
            plm[i, 1] = 10 + i % 5
    return plm


@pytest.fixture(scope="module")
def state(path_link_matrix, od_path_set):
    """State dict with path_link_matrix, link_flows, and od_path_set."""
    return {
        'path_link_matrix': path_link_matrix,
        'link_flows': np.arange(15, dtype=float),
        'od_path_set': od_path_set,
    }


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

class TestNoAttack:
    """NoAttack should always return zeros."""

    def test_no_attack_zero(self, true_costs):
        """NoAttack always returns zeros regardless of day."""
        policy = NoAttack()
        for day in [0, 50, 75, 100, 200]:
            result = policy.generate(day, true_costs)
            np.testing.assert_array_equal(result, np.zeros_like(true_costs))


class TestConstantFDIActivePeriod:
    """ConstantFDI should only be active during [start_day, end_day)."""

    def test_constant_fdi_active_period(self, true_costs, state):
        """Returns nonzero only during [start, end)."""
        config = AttackConfig(start_day=50, end_day=100, bias_fraction=0.3)
        policy = ConstantFDI(config)

        # Before attack
        assert np.all(policy.generate(49, true_costs, state) == 0)

        # During attack — asymmetric perturbation should produce nonzero values
        result = policy.generate(50, true_costs, state)
        assert np.any(result != 0)

        # After attack
        assert np.all(policy.generate(100, true_costs, state) == 0)


class TestConstantFDIMagnitude:
    """ConstantFDI reverses cost signal: good paths inflated, bad deflated."""

    def test_constant_fdi_asymmetric(self, true_costs, state):
        """Perturbation is -bias_fraction * (cost - mean_cost) per OD."""
        config = AttackConfig(start_day=0, end_day=200, bias_fraction=0.3)
        policy = ConstantFDI(config)
        result = policy.generate(50, true_costs, state)

        # Verify per-OD: cheap paths get positive perturbation (inflated),
        # expensive paths get negative (deflated)
        for path_indices in state['od_path_set']:
            od_costs = true_costs[path_indices]
            mean_cost = od_costs.mean(axis=0, keepdims=True)
            expected = -0.3 * (od_costs - mean_cost)
            np.testing.assert_allclose(result[path_indices], expected)

    def test_perturbation_sums_to_zero_per_od(self, true_costs, state):
        """Asymmetric perturbation sums to ~zero within each OD pair."""
        config = AttackConfig(start_day=0, end_day=200, bias_fraction=0.3)
        policy = ConstantFDI(config)
        result = policy.generate(50, true_costs, state)

        for path_indices in state['od_path_set']:
            od_perturbation = result[path_indices]
            np.testing.assert_allclose(
                od_perturbation.sum(axis=0), 0.0, atol=1e-10,
            )


class TestAsymmetricFDI:
    """AsymmetricFDI inflates the cheapest paths per OD only."""

    def test_only_best_paths_perturbed(self, true_costs, state):
        """Only top-K cheapest paths per OD are inflated; others are zero."""
        config = AttackConfig(start_day=0, end_day=200, bias_fraction=0.3)
        policy = AsymmetricFDI(config, top_k=2)
        result = policy.generate(50, true_costs, state)

        for path_indices in state['od_path_set']:
            od_costs = true_costs[path_indices]
            mean_per_path = od_costs.mean(axis=-1)
            best_local = np.argsort(mean_per_path)[:2]
            worst_local = np.argsort(mean_per_path)[2:]

            # Best paths should have positive perturbation
            best_paths = np.asarray(path_indices)[best_local]
            assert np.all(result[best_paths] > 0)

            # Other paths should be zero
            other_paths = np.asarray(path_indices)[worst_local]
            np.testing.assert_array_equal(result[other_paths], 0)

    def test_asymmetric_magnitude(self, true_costs, state):
        """Inflated paths get bias_fraction * true_cost."""
        config = AttackConfig(start_day=0, end_day=200, bias_fraction=0.3)
        policy = AsymmetricFDI(config, top_k=2)
        result = policy.generate(50, true_costs, state)

        for path_indices in state['od_path_set']:
            od_costs = true_costs[path_indices]
            mean_per_path = od_costs.mean(axis=-1)
            best_local = np.argsort(mean_per_path)[:2]
            best_paths = np.asarray(path_indices)[best_local]
            np.testing.assert_allclose(
                result[best_paths], 0.3 * true_costs[best_paths],
            )


class TestEscalatingTwoPhases:
    """Escalating attack has lower perturbation in phase 1 than phase 2."""

    def test_escalating_two_phases(self, true_costs):
        """Phase 1 perturbation < Phase 2 perturbation."""
        config = AttackConfig(
            start_day=50, end_day=100, bias_fraction=0.3,
        )
        policy = EscalatingAttack(config, phase1_fraction=0.1, mid_day=75)

        phase1 = policy.generate(60, true_costs)
        phase2 = policy.generate(80, true_costs)

        assert np.linalg.norm(phase1) > 0
        assert np.linalg.norm(phase2) > 0

        np.testing.assert_allclose(phase1, 0.1 * true_costs)
        np.testing.assert_allclose(phase2, 0.3 * true_costs)

        assert np.linalg.norm(phase1) < np.linalg.norm(phase2)


class TestIntermittentPattern:
    """Intermittent attack follows on/off cycle."""

    def test_intermittent_pattern(self, true_costs):
        """On/off pattern matches (on_days=3, off_days=2)."""
        config = AttackConfig(start_day=50, end_day=100, bias_fraction=0.3)
        policy = IntermittentAttack(config, on_days=3, off_days=2)

        expected_on = [True, True, True, False, False] * 10
        for i, day in enumerate(range(50, 100)):
            result = policy.generate(day, true_costs)
            if expected_on[i]:
                assert np.any(result > 0), f"Day {day} should be ON"
            else:
                assert np.all(result == 0), f"Day {day} should be OFF"


class TestStrategicSybilAttack:
    """StrategicSybilAttack selects targets once and uses fixed path mask."""

    def test_targets_selected_once(self, true_costs):
        """Targets are computed on first active day and remain fixed."""
        config = AttackConfig(start_day=0, end_day=200, bias_fraction=0.3)
        policy = StrategicSybilAttack(config, n_target_links=3)

        # Build a fake incidence matrix (10 links x 20 paths)
        rng = np.random.default_rng(99)
        data = (rng.random((10, 20)) > 0.7).astype(float)
        delta = sp.csr_matrix(data)

        # Fake network-like object with link_capacity
        class FakeNet:
            link_capacity = rng.uniform(0.5, 2.0, size=10)

        state = {
            'incidence_matrix': delta,
            'network': FakeNet(),
        }

        r1 = policy.generate(0, true_costs, state)
        targets_first = policy.targets.copy()
        r2 = policy.generate(1, true_costs, state)
        np.testing.assert_array_equal(policy.targets, targets_first)
        assert np.any(r1 > 0)

    def test_inactive_days_zero(self, true_costs):
        """Returns zero outside [start, end)."""
        config = AttackConfig(start_day=50, end_day=100, bias_fraction=0.3)
        policy = StrategicSybilAttack(config, n_target_links=3)
        assert np.all(policy.generate(49, true_costs) == 0)
        assert np.all(policy.generate(100, true_costs) == 0)

    def test_path_mask_applied(self, true_costs):
        """Only paths using target links get perturbation."""
        config = AttackConfig(start_day=0, end_day=200, bias_fraction=0.3)
        policy = StrategicSybilAttack(config, n_target_links=2)

        # Build sparse incidence: only link 0 connects to paths 0,1,2
        # link 1 connects to paths 3,4
        data = np.zeros((5, 20))
        data[0, 0] = data[0, 1] = data[0, 2] = 1.0
        data[1, 3] = data[1, 4] = 1.0
        data[2, 10] = 1.0
        data[3, 15] = 1.0
        data[4, 18] = 1.0
        delta = sp.csr_matrix(data)

        class FakeNet:
            link_capacity = np.array([0.5, 0.5, 2.0, 2.0, 2.0])

        state = {
            'incidence_matrix': delta,
            'network': FakeNet(),
        }

        result = policy.generate(0, true_costs, state)
        # Paths not on any target link should be zero
        mask = policy._path_mask
        for i in range(20):
            if mask[i] == 0:
                np.testing.assert_array_equal(result[i], 0)
            else:
                assert np.all(result[i] > 0)


class TestAttackBudgetConstraint:
    """Total perturbation norm should not exceed budget."""

    def test_attack_budget_constraint(self, true_costs, state):
        """Total perturbation norm <= budget."""
        budget = 50.0
        config = AttackConfig(
            start_day=0, end_day=200, bias_fraction=0.5, budget=budget,
        )
        policy = ConstantFDI(config)
        result = policy.generate(50, true_costs, state)

        # Unconstrained norm should exceed budget for this to be a valid test
        config_unbounded = AttackConfig(
            start_day=0, end_day=200, bias_fraction=0.5,
        )
        policy_unbounded = ConstantFDI(config_unbounded)
        unconstrained = policy_unbounded.generate(50, true_costs, state)
        assert np.linalg.norm(unconstrained) > budget

        # With budget, should be clipped
        assert np.linalg.norm(result) <= budget + 1e-10
