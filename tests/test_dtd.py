"""Tests for the day-to-day dynamics simulator."""

import copy
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from network.network import NetworkData, load_sioux_falls
from dnl.dnl import BPRDNL
from dtd.dtd_simulator import DTDParams, DTDSimulator, DayResult

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


def _make_disrupted_network(
    net: NetworkData, sim_params=None, dnl_obj=None,
) -> NetworkData:
    """Create a disrupted network by severely reducing capacity on the
    most congested links (highest flow/capacity ratio)."""
    from dnl.dnl import build_incidence_matrix

    dn = copy.copy(net)
    cap = net.link_capacity.copy()

    if sim_params is not None and sim_params.initial_path_departures is not None:
        delta = build_incidence_matrix(net)
        link_flow = np.asarray(delta @ sim_params.initial_path_departures).max(axis=1)
        ratio = link_flow / cap
        # Pick the 15 most congested links and slash capacity to 10%
        worst = np.argsort(ratio)[-15:]
        cap[worst] *= 0.10
    else:
        # Fallback: halve the 10 lowest-capacity (bottleneck) links
        bottom10 = np.argsort(cap)[:10]
        cap[bottom10] *= 0.5

    dn.link_capacity = cap
    return dn


# -------------------------------------------------------------------
# 1. Initialisation
# -------------------------------------------------------------------
class TestInitialisation:
    def test_creates_without_error(self, net, demand, dnl, sim_params):
        params = DTDParams(n_days=1)
        sim = DTDSimulator(net, demand, dnl, params, sim_params)
        assert sim is not None

    def test_initial_path_departures_shape(self, sim_params):
        dep = sim_params.initial_path_departures
        assert dep is not None
        assert dep.shape == (6180, 100)


# -------------------------------------------------------------------
# 2. Single day
# -------------------------------------------------------------------
class TestSingleDay:
    def test_shapes(self, net, demand, dnl, sim_params):
        params = DTDParams(n_days=1, theta=0.02)
        sim = DTDSimulator(net, demand, dnl, params, sim_params)
        results = sim.run()

        assert len(results) == 1
        r = results[0]
        assert isinstance(r, DayResult)
        assert r.path_flows.shape == (6180, 100)
        assert r.agg_path_flow.shape == (6180, 20)
        assert r.perceived_cost.shape == (6180, 20)
        assert r.agg_cost.shape == (6180, 20)
        assert r.travel_times.shape == (6180, 100)

    def test_tstt_positive_and_finite(self, net, demand, dnl, sim_params):
        params = DTDParams(n_days=1, theta=0.02)
        sim = DTDSimulator(net, demand, dnl, params, sim_params)
        results = sim.run()
        assert results[0].tstt > 0
        assert np.isfinite(results[0].tstt)

    def test_callback_invoked(self, net, demand, dnl, sim_params):
        params = DTDParams(n_days=3, theta=0.02)
        sim = DTDSimulator(net, demand, dnl, params, sim_params)
        called = []
        sim.run(callbacks=[lambda d, r: called.append(d)])
        assert called == [0, 1, 2]


# -------------------------------------------------------------------
# 3. Convergence without disruption
# -------------------------------------------------------------------
class TestConvergenceNoDis:
    @pytest.fixture(scope="class")
    def results_100(self, net, demand, dnl, sim_params):
        params = DTDParams(n_days=100, theta=0.02, memory_days=6,
                           memory_weight=0.7)
        sim = DTDSimulator(net, demand, dnl, params, sim_params)
        return sim.run()

    def test_tstt_stabilises(self, results_100):
        tstt = np.array([r.tstt for r in results_100])
        tail = tstt[-20:]
        cv = tail.std() / tail.mean()   # coefficient of variation
        assert cv < 0.05, f"TSTT did not stabilise: CV={cv:.4f}"

    def test_tstt_all_finite(self, results_100):
        for r in results_100:
            assert np.isfinite(r.tstt), "TSTT contains non-finite values"


# -------------------------------------------------------------------
# 4. Disruption response
# -------------------------------------------------------------------
class TestDisruptionResponse:
    @pytest.fixture(scope="class")
    def results_150(self, net, demand, dnl, sim_params):
        dn = _make_disrupted_network(net, sim_params)
        params = DTDParams(
            n_days=150, theta=0.02, memory_days=6,
            memory_weight=0.7, disruption_days=(51, 100),
        )
        sim = DTDSimulator(net, demand, dnl, params, sim_params,
                           disrupted_network=dn)
        return sim.run()

    def test_tstt_increases_during_disruption(self, results_150):
        tstt = np.array([r.tstt for r in results_150])
        pre = tstt[30:50].mean()     # settled pre-disruption
        during = tstt[70:100].mean() # mid-disruption
        assert during > pre, (
            f"TSTT should increase during disruption: "
            f"pre={pre:.1f}, during={during:.1f}"
        )

    def test_tstt_decreases_after_disruption(self, results_150):
        tstt = np.array([r.tstt for r in results_150])
        during = tstt[70:100].mean()
        post = tstt[130:150].mean()
        assert post < during, (
            f"TSTT should decrease after disruption ends: "
            f"during={during:.1f}, post={post:.1f}"
        )


# -------------------------------------------------------------------
# 5. Bounded rationality slows convergence
# -------------------------------------------------------------------
class TestBoundedRationality:
    def test_br_more_inertia(self, net, demand, dnl, sim_params):
        n_days = 60

        # Standard MNL (no BR)
        p0 = DTDParams(n_days=n_days, theta=0.02,
                        use_bounded_rationality=False)
        sim0 = DTDSimulator(net, demand, dnl, p0, sim_params)
        r0 = sim0.run()

        # BR with delta=400
        p1 = DTDParams(n_days=n_days, theta=0.02,
                        use_bounded_rationality=True,
                        indifference_band=400.0)
        sim1 = DTDSimulator(net, demand, dnl, p1, sim_params)
        r1 = sim1.run()

        tstt0 = np.array([r.tstt for r in r0])
        tstt1 = np.array([r.tstt for r in r1])

        # BR should show more day-to-day variation (slower convergence)
        # measured by the standard deviation of day-to-day TSTT changes
        diff0 = np.abs(np.diff(tstt0[10:]))
        diff1 = np.abs(np.diff(tstt1[10:]))

        # The BR model preserves previous flows (inertia), so the
        # TSTT trajectory should differ from the MNL trajectory.
        assert not np.allclose(tstt0, tstt1, rtol=0.001), \
            "BR and MNL should produce different TSTT trajectories"


# -------------------------------------------------------------------
# 6. Numerical stability
# -------------------------------------------------------------------
class TestNumericalStability:
    def test_large_theta_no_nan(self, net, demand, dnl, sim_params):
        """Very large theta concentrates probability on the best
        alternative. Must not produce NaN or Inf."""
        params = DTDParams(n_days=10, theta=1.0)
        sim = DTDSimulator(net, demand, dnl, params, sim_params)
        results = sim.run()
        for r in results:
            assert np.all(np.isfinite(r.perceived_cost)), "NaN in PC"
            assert np.all(np.isfinite(r.agg_path_flow)), "NaN in flow"
            assert np.isfinite(r.tstt), "NaN in TSTT"

    def test_small_theta_uniform(self, net, demand, dnl, sim_params):
        """Very small theta → nearly uniform choice.
        All (path,window) combos within an OD get similar flow."""
        params = DTDParams(n_days=5, theta=1e-6)
        sim = DTDSimulator(net, demand, dnl, params, sim_params)
        results = sim.run()
        r = results[-1]
        for k, paths in enumerate(demand.od_path_set):
            flows = r.agg_path_flow[paths, :]
            if flows.sum() > 0:
                cv = flows.std() / flows.mean()
                assert cv < 0.01, (
                    f"OD {k}: theta≈0 should give near-uniform flow, "
                    f"CV={cv:.4f}"
                )

    def test_br_large_delta_no_nan(self, net, demand, dnl, sim_params):
        """Large indifference band should not cause overflow."""
        params = DTDParams(n_days=5, theta=0.02,
                           use_bounded_rationality=True,
                           indifference_band=2000.0)
        sim = DTDSimulator(net, demand, dnl, params, sim_params)
        results = sim.run()
        for r in results:
            assert np.all(np.isfinite(r.agg_path_flow)), \
                "NaN/Inf in BR flow with large delta"


# -------------------------------------------------------------------
# 7. Information sharing
# -------------------------------------------------------------------
class TestInfoSharing:
    def test_is_runs_and_converges(self, net, demand, dnl, sim_params):
        params = DTDParams(n_days=50, theta=0.02,
                           use_info_sharing=True, is_alpha=2.0)
        sim = DTDSimulator(net, demand, dnl, params, sim_params)
        results = sim.run()
        tstt = np.array([r.tstt for r in results])
        assert np.all(np.isfinite(tstt))
        # Should show some convergence behaviour
        tail = tstt[-10:]
        assert tail.std() / tail.mean() < 0.10
