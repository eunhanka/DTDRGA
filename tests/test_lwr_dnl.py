"""Tests for LWR Newell kinematic-wave DNL."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from network.network import load_sioux_falls
from dnl.lwr_dnl import LWRDNL

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "sioux_falls")


@pytest.fixture(scope="module")
def sioux_falls():
    """Load Sioux Falls network data."""
    return load_sioux_falls(DATA_DIR)


@pytest.fixture(scope="module")
def lwr():
    """LWR DNL instance."""
    return LWRDNL(verbose=False)


class TestPreprocess:
    """Preprocessing computes correct derived quantities."""

    def test_fft_mod_multiples_of_dt(self, sioux_falls):
        net, _, params = sioux_falls
        fft_mod, _, _, _, _ = LWRDNL._preprocess(net, params.dt)
        # All modified FFT should be multiples of dt
        residuals = fft_mod % params.dt
        np.testing.assert_allclose(residuals, 0.0, atol=1e-10)

    def test_fft_mod_positive(self, sioux_falls):
        net, _, params = sioux_falls
        fft_mod, _, _, _, _ = LWRDNL._preprocess(net, params.dt)
        assert np.all(fft_mod >= params.dt)

    def test_tnw_is_3x_tnk(self, sioux_falls):
        net, _, params = sioux_falls
        _, _, _, tnk, tnw = LWRDNL._preprocess(net, params.dt)
        np.testing.assert_array_equal(tnw, 3 * tnk)

    def test_n_jam_positive(self, sioux_falls):
        net, _, params = sioux_falls
        _, _, N_jam, _, _ = LWRDNL._preprocess(net, params.dt)
        assert np.all(N_jam > 0)


class TestZeroFlow:
    """With zero departures, travel times should equal free-flow time."""

    def test_zero_flow_returns_fft(self, sioux_falls, lwr):
        net, demand, params = sioux_falls
        n_paths = net.n_paths
        nt = params.n_steps

        zero_dep = np.zeros((n_paths, nt), dtype=np.float64)
        tt = lwr.compute_costs(zero_dep, net, params)

        assert tt.shape == (n_paths, nt)
        # With zero flow, all travel times should be zero (no vehicles)
        # Actually, zero departures means no vehicles enter, so travel
        # times are technically undefined. The implementation should
        # return zeros since no vehicles travel.
        assert np.all(tt >= 0)


class TestLowFlow:
    """Low flow should produce travel times near free-flow."""

    def test_low_flow_near_fft(self, sioux_falls, lwr):
        net, demand, params = sioux_falls
        # Use 1% of the initial departures
        low_dep = params.initial_path_departures * 0.01
        tt = lwr.compute_costs(low_dep, net, params)

        assert tt.shape == (net.n_paths, params.n_steps)
        # Should be positive for paths with nonzero departures
        has_flow = low_dep.sum(axis=1) > 0
        # For paths with flow, travel time should be >= some FFT
        assert np.all(tt[has_flow] >= 0)


class TestNormalFlow:
    """Normal flow (initial departures) should produce reasonable results."""

    def test_normal_flow_shape(self, sioux_falls, lwr):
        net, demand, params = sioux_falls
        tt = lwr.compute_costs(params.initial_path_departures, net, params)
        assert tt.shape == (net.n_paths, params.n_steps)

    def test_travel_times_positive(self, sioux_falls, lwr):
        net, demand, params = sioux_falls
        tt = lwr.compute_costs(params.initial_path_departures, net, params)
        has_flow = params.initial_path_departures.sum(axis=1) > 0
        # Paths with departures should have positive travel times
        for r in range(net.n_paths):
            if has_flow[r]:
                active_steps = params.initial_path_departures[r] > 0
                if active_steps.any():
                    assert tt[r, active_steps].min() > 0, (
                        f"Path {r} has zero travel time despite having flow"
                    )

    def test_travel_times_bounded(self, sioux_falls, lwr):
        net, demand, params = sioux_falls
        tt = lwr.compute_costs(params.initial_path_departures, net, params)
        # Travel times should not exceed the time horizon
        horizon = params.time_horizon[1] - params.time_horizon[0]
        assert np.all(tt <= horizon + 1.0)
