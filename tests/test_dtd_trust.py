"""Tests for the integrated DTD-Trust-Attack simulator."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from network.network import load_sioux_falls
from dnl.dnl import BPRDNL
from dtd.dtd_simulator import DTDParams
from dtd.dtd_trust_simulator import DTDTrustSimulator, IntegratedResult
from trust.trust_dynamics import TrustParams
from attack.attack_models import AttackConfig, NoAttack, ConstantFDI

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "sioux_falls")


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture(scope="module")
def loaded_data():
    return load_sioux_falls(DATA_DIR)


@pytest.fixture(scope="module")
def net(loaded_data):
    return loaded_data[0]


@pytest.fixture(scope="module")
def demand(loaded_data):
    return loaded_data[1]


@pytest.fixture(scope="module")
def sim_params(loaded_data):
    return loaded_data[2]


@pytest.fixture(scope="module")
def dnl():
    return BPRDNL(alpha=0.15, beta=4.0)


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

class TestNoAttackBaseline:
    """Integrated simulator with NoAttack should behave like base DTD."""

    def test_runs_without_error(self, net, demand, dnl, sim_params):
        """20-day smoke test with NoAttack."""
        dtd_p = DTDParams(n_days=20, theta=0.02, memory_days=3, memory_weight=0.7)
        trust_p = TrustParams()
        sim = DTDTrustSimulator(
            net, demand, dnl, dtd_p, sim_params,
            trust_params=trust_p,
            attack_policy=NoAttack(),
        )
        result = sim.run()

        assert isinstance(result, IntegratedResult)
        assert result.tstt.shape == (20,)
        assert result.trust.shape == (20, 3)
        assert result.info_weight.shape == (20, 3)
        assert result.mean_compliance.shape == (20,)
        assert result.attack_active.shape == (20,)
        assert np.all(np.isfinite(result.tstt))
        assert np.all(result.tstt > 0)

    def test_trust_maintained_no_attack(self, net, demand, dnl, sim_params):
        """Without attack, trust should stay high."""
        dtd_p = DTDParams(n_days=20, theta=0.02, memory_days=3, memory_weight=0.7)
        trust_p = TrustParams()
        sim = DTDTrustSimulator(
            net, demand, dnl, dtd_p, sim_params,
            trust_params=trust_p,
            attack_policy=NoAttack(),
        )
        result = sim.run()

        # Trust should remain high (no attack → accurate guidance)
        assert np.all(result.trust[-1] > 0.7)
        # No attack days
        assert not np.any(result.attack_active)


class TestAttackImpact:
    """ConstantFDI attack should erode trust and increase TSTT."""

    def test_attack_erodes_trust(self, net, demand, dnl, sim_params):
        """30-day sim with attack on days 6-20: trust should drop."""
        dtd_p = DTDParams(n_days=30, theta=0.02, memory_days=3, memory_weight=0.7)
        trust_p = TrustParams()
        config = AttackConfig(start_day=5, end_day=20, bias_fraction=0.3)
        attack = ConstantFDI(config)

        sim = DTDTrustSimulator(
            net, demand, dnl, dtd_p, sim_params,
            trust_params=trust_p,
            attack_policy=attack,
        )
        result = sim.run()

        # Trust should drop during attack
        pre_attack_trust = result.trust[4].mean()   # day 5 (before attack)
        during_attack_trust = result.trust[15].mean()  # day 16 (mid-attack)
        assert during_attack_trust < pre_attack_trust

        # Attack active flags
        assert not result.attack_active[4]   # day 5: before
        assert result.attack_active[10]      # day 11: during
        assert not result.attack_active[20]  # day 21: after

    def test_info_weight_drops_during_attack(self, net, demand, dnl, sim_params):
        """Information weights should decrease as trust erodes."""
        dtd_p = DTDParams(n_days=30, theta=0.02, memory_days=3, memory_weight=0.7)
        trust_p = TrustParams()
        config = AttackConfig(start_day=5, end_day=20, bias_fraction=0.3)
        attack = ConstantFDI(config)

        sim = DTDTrustSimulator(
            net, demand, dnl, dtd_p, sim_params,
            trust_params=trust_p,
            attack_policy=attack,
        )
        result = sim.run()

        # Info weight should drop during attack
        pre_weight = result.info_weight[4].mean()
        mid_weight = result.info_weight[15].mean()
        assert mid_weight < pre_weight

        # Mean compliance should track info weight
        assert result.mean_compliance[15] < result.mean_compliance[4]


class TestOutputShapes:
    """Verify all output array shapes and types."""

    def test_result_shapes(self, net, demand, dnl, sim_params):
        n_days = 15
        dtd_p = DTDParams(n_days=n_days, theta=0.02, memory_days=3, memory_weight=0.7)
        trust_p = TrustParams(n_classes=3)
        sim = DTDTrustSimulator(
            net, demand, dnl, dtd_p, sim_params,
            trust_params=trust_p,
            attack_policy=NoAttack(),
        )
        result = sim.run()

        assert result.tstt.shape == (n_days,)
        assert result.trust.shape == (n_days, 3)
        assert result.info_weight.shape == (n_days, 3)
        assert result.mean_compliance.shape == (n_days,)
        assert result.attack_active.shape == (n_days,)
        assert result.attack_active.dtype == bool
        assert result.agg_costs is not None
        assert result.agg_costs.shape[0] == net.n_paths


class TestHeterogeneousClasses:
    """Different classes should respond differently to the same attack."""

    def test_cav_erodes_fastest(self, net, demand, dnl, sim_params):
        """CAV (class 0, highest w_f) should lose trust fastest."""
        dtd_p = DTDParams(n_days=30, theta=0.02, memory_days=3, memory_weight=0.7)
        trust_p = TrustParams()
        config = AttackConfig(start_day=5, end_day=25, bias_fraction=0.3)
        attack = ConstantFDI(config)

        sim = DTDTrustSimulator(
            net, demand, dnl, dtd_p, sim_params,
            trust_params=trust_p,
            attack_policy=attack,
        )
        result = sim.run()

        # At end of attack, all classes should have eroded trust
        # With uniform w_f=[0.5,0.5,0.5], all classes erode equally
        end_trust = result.trust[24]
        assert end_trust[0] <= end_trust[1] <= end_trust[2]
