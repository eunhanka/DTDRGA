"""Tests for the DNL module."""

import os
import sys

import numpy as np
import pytest
import scipy.sparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from network.network import load_sioux_falls
from dnl.dnl import (
    BPRDNL,
    build_incidence_matrix,
    compute_arrival_penalty,
    aggregate_to_windows,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "sioux_falls")


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
def params(loaded_data):
    return loaded_data[2]


@pytest.fixture(scope="module")
def delta(net):
    return build_incidence_matrix(net)


# -------------------------------------------------------------------
# Incidence matrix
# -------------------------------------------------------------------
class TestIncidenceMatrix:
    def test_shape(self, delta, net):
        assert delta.shape == (net.n_links, net.n_paths)
        assert delta.shape == (76, 6180)

    def test_sparse_type(self, delta):
        assert scipy.sparse.issparse(delta)

    def test_binary_values(self, delta):
        data = delta.data
        assert np.all((data == 0) | (data == 1))

    def test_column_sum_equals_path_link_count(self, delta, net):
        """Each column of Delta sums to the number of links in that path."""
        col_sums = np.asarray(delta.sum(axis=0)).flatten()
        np.testing.assert_array_equal(col_sums, net.path_link_count)

    def test_no_empty_columns(self, delta):
        """Every path must use at least one link."""
        col_sums = np.asarray(delta.sum(axis=0)).flatten()
        assert (col_sums >= 1).all()

    def test_row_sums_positive(self, delta):
        """Every link must be used by at least one path."""
        row_sums = np.asarray(delta.sum(axis=1)).flatten()
        assert (row_sums >= 1).all()

    def test_specific_path(self, net, delta):
        """Spot-check: first path with 1 link should have a single entry."""
        # Path 0 has path_link_count[0] links
        n_links_p0 = net.path_link_count[0]
        col0 = np.asarray(delta[:, 0].todense()).flatten()
        assert col0.sum() == n_links_p0
        # The non-zero entries should correspond to the links in path 0
        expected_links = net.path_link_matrix[0, :n_links_p0]
        assert set(np.where(col0 > 0)[0]) == set(expected_links)


# -------------------------------------------------------------------
# BPR DNL — zero flow
# -------------------------------------------------------------------
class TestBPRDNLZeroFlow:
    def test_zero_flow_returns_free_flow_sum(self, net, params, delta):
        """With zero departures, path cost = sum of free-flow times."""
        dnl = BPRDNL(alpha=0.15, beta=4.0)
        zero_dep = np.zeros((net.n_paths, params.n_steps), dtype=np.float64)
        tt = dnl.compute_costs(zero_dep, net, params)

        assert tt.shape == (net.n_paths, params.n_steps)

        # Expected: Delta.T @ fft  (same for every time step)
        expected_fft = scipy.sparse.csr_matrix(delta.T) @ net.link_fft
        np.testing.assert_allclose(tt[:, 0], expected_fft, rtol=1e-12)
        # Constant across all time steps
        np.testing.assert_allclose(tt[:, -1], expected_fft, rtol=1e-12)

    def test_all_costs_positive(self, net, params):
        dnl = BPRDNL()
        zero_dep = np.zeros((net.n_paths, params.n_steps))
        tt = dnl.compute_costs(zero_dep, net, params)
        assert (tt > 0).all()


# -------------------------------------------------------------------
# BPR DNL — low uniform flow
# -------------------------------------------------------------------
class TestBPRDNLLowFlow:
    def test_low_flow_close_to_free_flow(self, net, params, delta):
        """With very low flow, costs should be very close to free-flow."""
        dnl = BPRDNL(alpha=0.15, beta=4.0)
        # Use a tiny uniform departure rate: 0.001 veh/s on every path
        low_dep = np.full((net.n_paths, params.n_steps), 1e-3)
        tt = dnl.compute_costs(low_dep, net, params)

        expected_ff = scipy.sparse.csr_matrix(delta.T) @ net.link_fft
        # Relative increase should be small at such low flows.
        # With 6180 paths aggregating onto 76 links, even 1e-3 veh/s per
        # path yields non-trivial link flow, so allow up to 2%.
        max_rel_increase = np.max((tt[:, 0] - expected_ff) / expected_ff)
        assert max_rel_increase < 0.02

    def test_higher_flow_increases_cost(self, net, params):
        """Doubling flow should increase travel times."""
        dnl = BPRDNL()
        low = np.full((net.n_paths, params.n_steps), 0.01)
        high = np.full((net.n_paths, params.n_steps), 0.02)
        tt_low = dnl.compute_costs(low, net, params)
        tt_high = dnl.compute_costs(high, net, params)
        assert (tt_high >= tt_low).all()
        # At least some paths should be strictly more expensive
        assert (tt_high > tt_low).any()

    def test_bpr_parameters_affect_output(self, net, params):
        """Different alpha/beta should yield different costs."""
        dep = np.full((net.n_paths, params.n_steps), 0.5)
        dnl_a = BPRDNL(alpha=0.15, beta=4.0)
        dnl_b = BPRDNL(alpha=0.50, beta=2.0)
        tt_a = dnl_a.compute_costs(dep, net, params)
        tt_b = dnl_b.compute_costs(dep, net, params)
        assert not np.allclose(tt_a, tt_b)


# -------------------------------------------------------------------
# Arrival penalty
# -------------------------------------------------------------------
class TestArrivalPenalty:
    def test_output_shape(self, net, params, demand):
        dnl = BPRDNL()
        dep = np.zeros((net.n_paths, params.n_steps))
        tt = dnl.compute_costs(dep, net, params)
        eff = compute_arrival_penalty(tt, params, demand, net)
        assert eff.shape == (net.n_paths, params.n_steps)

    def test_effective_delay_ge_travel_time(self, net, params, demand):
        """Effective delay >= raw travel time (penalty is non-negative)."""
        dnl = BPRDNL()
        dep = np.zeros((net.n_paths, params.n_steps))
        tt = dnl.compute_costs(dep, net, params)
        eff = compute_arrival_penalty(tt, params, demand, net)
        assert (eff >= tt - 1e-10).all()

    def test_zero_penalty_at_exact_arrival(self, net, params, demand):
        """If arrival == target, penalty should be zero → eff == tt."""
        # Construct fake travel_times such that arrival = target for path 0
        # target_arrival[0] corresponds to the OD of path 0
        ta0 = demand.target_arrival[0]
        # departure time at step 0 is time_horizon[0] = 0
        # We want 0 + tt = ta0, so tt = ta0
        fake_tt = np.full((net.n_paths, params.n_steps), ta0)
        # Only test the first time step (departure at t=0)
        eff = compute_arrival_penalty(fake_tt, params, demand, net,
                                      gamma_early=0.8, gamma_late=1.8)
        # For path 0 at step 0: arrival = 0 + ta0 = ta0 → penalty = 0
        np.testing.assert_allclose(eff[0, 0], fake_tt[0, 0], atol=1e-10)


# -------------------------------------------------------------------
# Window aggregation
# -------------------------------------------------------------------
class TestAggregateToWindows:
    def test_shape(self, params):
        n_paths = 10
        fine = np.random.rand(n_paths, params.n_steps)
        agg = aggregate_to_windows(fine, params)
        assert agg.shape == (n_paths, params.n_windows)

    def test_constant_input(self, params):
        """Constant across time steps → same value in every window."""
        n_paths = 5
        val = 42.0
        fine = np.full((n_paths, params.n_steps), val)
        agg = aggregate_to_windows(fine, params)
        np.testing.assert_allclose(agg, val)

    def test_mean_is_correct(self, params):
        """Each window average should match manual calculation."""
        n_paths = 3
        fine = np.arange(params.n_steps, dtype=np.float64)
        fine = np.tile(fine, (n_paths, 1))  # [3, 100]
        agg = aggregate_to_windows(fine, params)
        spw = params.steps_per_window
        for w in range(params.n_windows):
            expected = np.mean(fine[0, w * spw:(w + 1) * spw])
            np.testing.assert_allclose(agg[0, w], expected)

    def test_sioux_falls_dimensions(self, net, params):
        fine = np.zeros((net.n_paths, params.n_steps))
        agg = aggregate_to_windows(fine, params)
        assert agg.shape == (6180, 20)
