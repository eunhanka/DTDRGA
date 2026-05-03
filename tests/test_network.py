"""Tests for the network data loader."""

import os
import sys

import numpy as np
import pytest

# allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from network.network import load_sioux_falls

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


# -------------------------------------------------------------------
# NetworkData
# -------------------------------------------------------------------
class TestNetworkData:
    def test_counts(self, net):
        assert net.n_nodes == 24
        assert net.n_links == 76
        assert net.n_paths == 6180
        assert net.n_od == 528
        assert net.n_sources == 24
        assert net.n_sinks == 24

    def test_path_link_matrix_shape(self, net):
        assert net.path_link_matrix.shape == (6180, 19)

    def test_path_link_matrix_0indexed(self, net):
        # valid links are 0..75, padding is -1
        valid = net.path_link_matrix[net.path_link_matrix >= 0]
        assert valid.min() >= 0
        assert valid.max() <= net.n_links - 1
        # padding values should be -1
        assert (net.path_link_matrix[net.path_link_matrix < 0] == -1).all()

    def test_adjacency_list_shape(self, net):
        assert net.adjacency_list.shape == (76, 2)

    def test_adjacency_list_0indexed(self, net):
        assert net.adjacency_list.min() >= 0
        assert net.adjacency_list.max() <= net.n_nodes - 1

    def test_link_arrays_shape(self, net):
        assert net.link_capacity.shape == (76,)
        assert net.link_fft.shape == (76,)
        assert net.link_length.shape == (76,)

    def test_link_positive_values(self, net):
        assert (net.link_capacity > 0).all()
        assert (net.link_fft > 0).all()
        assert (net.link_length > 0).all()

    def test_links_in_out_length(self, net):
        assert len(net.links_in) == 24
        assert len(net.links_out) == 24

    def test_links_in_out_0indexed(self, net):
        # all link indices in links_in/out should be valid
        total_elements = net.n_links + net.n_sources + net.n_sinks
        for arr in net.links_in:
            if arr.size > 0:
                assert arr.min() >= 0
                assert arr.max() < total_elements
        for arr in net.links_out:
            if arr.size > 0:
                assert arr.min() >= 0
                assert arr.max() < total_elements

    def test_num_links_in_matches(self, net):
        for i in range(net.n_nodes):
            assert len(net.links_in[i]) == net.num_links_in[i]
            assert len(net.links_out[i]) == net.num_links_out[i]

    def test_source_sink_nodes(self, net):
        assert net.source_nodes.shape == (24,)
        assert net.sink_nodes.shape == (24,)
        assert net.source_nodes.min() >= 0
        assert net.source_nodes.max() <= 23
        assert net.sink_nodes.min() >= 0
        assert net.sink_nodes.max() <= 23

    def test_node_coordinates(self, net):
        assert net.node_x.shape == (24,)
        assert net.node_y.shape == (24,)

    def test_signal_priorities(self, net):
        assert len(net.signal_priorities) == 24
        for i in range(24):
            assert len(net.signal_priorities[i]) == net.num_links_in[i]

    def test_path_source_sink_nodes(self, net):
        assert net.path_source_node.shape == (6180,)
        assert net.path_sink_node.shape == (6180,)
        assert net.path_source_node.min() >= 0
        assert net.path_sink_node.min() >= 0

    def test_path_link_count(self, net):
        assert net.path_link_count.shape == (6180,)
        assert net.path_link_count.min() >= 1
        # verify link_count matches actual non-padding entries in path_link_matrix
        for i in range(0, 6180, 500):  # spot-check every 500th path
            row = net.path_link_matrix[i]
            n_valid = np.sum(row >= 0)
            assert n_valid == net.path_link_count[i]

    def test_path_links_in_out(self, net):
        assert len(net.path_links_in) == 24
        assert len(net.path_links_out) == 24
        # check dimensions match num_links_in/out
        for i in range(24):
            assert len(net.path_links_in[i]) == net.num_links_in[i]
            assert len(net.path_links_in[i][0]) == net.num_links_out[i]

    def test_link_index(self, net):
        assert net.link_index.shape == (76,)
        # should be 0..75 (sequential)
        np.testing.assert_array_equal(net.link_index, np.arange(76))


# -------------------------------------------------------------------
# DemandData
# -------------------------------------------------------------------
class TestDemandData:
    def test_od_set_shape(self, demand):
        assert demand.od_set.shape == (528, 3)

    def test_od_set_0indexed(self, demand):
        assert demand.od_set[:, :2].min() >= 0
        assert demand.od_set[:, :2].max() <= 23

    def test_od_path_set_length(self, demand):
        assert len(demand.od_path_set) == 528

    def test_od_path_set_0indexed(self, demand):
        for arr in demand.od_path_set:
            assert arr.min() >= 0
            assert arr.max() <= 6179

    def test_od_path_set_covers_all_paths(self, demand):
        all_paths = np.concatenate(demand.od_path_set)
        assert len(np.unique(all_paths)) == 6180

    def test_od_demand(self, demand):
        assert demand.od_demand.shape == (528,)
        assert (demand.od_demand > 0).all()

    def test_target_arrival_in_seconds(self, demand):
        assert demand.target_arrival.shape == (528,)
        # should be in seconds (original is hours * 3600)
        assert demand.target_arrival.min() >= 3600.0   # >= 1 hour
        assert demand.target_arrival.max() <= 18000.0   # <= 5 hours


# -------------------------------------------------------------------
# SimulationParams
# -------------------------------------------------------------------
class TestSimulationParams:
    def test_dt(self, params):
        assert params.dt == 180.0

    def test_departure_window(self, params):
        assert params.departure_window == 900.0

    def test_n_steps(self, params):
        assert params.n_steps == 100

    def test_n_windows(self, params):
        assert params.n_windows == 20

    def test_steps_per_window(self, params):
        assert params.steps_per_window == 5

    def test_time_horizon(self, params):
        assert params.time_horizon == (0.0, 18000.0)

    def test_initial_path_departures(self, params):
        assert params.initial_path_departures is not None
        assert params.initial_path_departures.shape == (6180, 100)
        assert params.initial_path_departures.min() >= 0.0
