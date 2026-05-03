"""
Dynamic Network Loading (DNL) module.

Tier 1: BPR-based quasi-static DNL — maps path departure rates to path
travel times using Bureau of Public Roads volume-delay functions and the
path-link incidence matrix.

Tier 2 (future): Full LWR Newell kinematic wave model ported from
DYNAMIC_NETWORK_LOADING.m.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse

from network.network import DemandData, NetworkData, SimulationParams


# ---------------------------------------------------------------------------
# Path-link incidence matrix
# ---------------------------------------------------------------------------

def build_incidence_matrix(network: NetworkData) -> scipy.sparse.csr_matrix:
    """Build the path-link incidence matrix Delta.

    Delta[a, r] = 1 if link *a* appears on path *r*, else 0.

    Parameters
    ----------
    network : NetworkData with ``path_link_matrix`` [n_paths, max_links]
              (0-indexed link ids, -1 = padding) and ``path_link_count``.

    Returns
    -------
    scipy.sparse.csr_matrix of shape [n_links, n_paths].
    """
    plm = network.path_link_matrix          # [P, max_links]
    n_paths, max_links = plm.shape
    n_links = network.n_links

    # Collect (link, path) pairs for all valid (non-padding) entries
    rows = []  # link indices
    cols = []  # path indices
    for r in range(n_paths):
        n_l = network.path_link_count[r]
        links = plm[r, :n_l]               # valid link ids for this path
        rows.append(links)
        cols.append(np.full(n_l, r, dtype=np.intp))

    row_idx = np.concatenate(rows)
    col_idx = np.concatenate(cols)
    data = np.ones(len(row_idx), dtype=np.float64)

    return scipy.sparse.csr_matrix(
        (data, (row_idx, col_idx)), shape=(n_links, n_paths)
    )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseDNL(ABC):
    """Interface for all DNL implementations."""

    @abstractmethod
    def compute_costs(
        self,
        path_departures: np.ndarray,
        network: NetworkData,
        params: SimulationParams,
    ) -> np.ndarray:
        """Compute path travel times from path departure rates.

        Parameters
        ----------
        path_departures : [n_paths, n_steps]  departure rates in veh/s.
        network : static network data.
        params  : temporal discretisation.

        Returns
        -------
        path_travel_times : [n_paths, n_steps]  in seconds.
        """


# ---------------------------------------------------------------------------
# Tier 1 — BPR quasi-static DNL
# ---------------------------------------------------------------------------

class BPRDNL(BaseDNL):
    """BPR volume-delay based DNL.

    For every time step *t* independently:

    1. Aggregate path flows to link flows via the incidence matrix.
    2. Apply the BPR function to obtain link travel times.
    3. Sum link travel times along each path.

    The incidence matrix is built once on the first call and cached.
    """

    def __init__(self, alpha: float = 0.15, beta: float = 4.0):
        self.alpha = alpha
        self.beta = beta
        self._delta: scipy.sparse.csr_matrix | None = None
        self._delta_T: scipy.sparse.csc_matrix | None = None

    def _get_incidence(self, network: NetworkData):
        """Lazily build and cache the incidence matrix and its transpose."""
        if self._delta is None:
            self._delta = build_incidence_matrix(network)
            self._delta_T = self._delta.T.tocsr()
        return self._delta, self._delta_T

    def compute_costs(
        self,
        path_departures: np.ndarray,
        network: NetworkData,
        params: SimulationParams,
    ) -> np.ndarray:
        delta, delta_T = self._get_incidence(network)

        # --- Step 1: path flows → link flows  [n_links, n_steps] ----------
        # path_departures is [n_paths, n_steps] in veh/s.
        # Multiply by dt to get vehicles per time step, which is the
        # flow volume to feed into BPR.  BPR capacity is also converted
        # to vehicles-per-step so the ratio is unitless.
        link_flow = delta @ path_departures              # [n_links, n_steps]

        # --- Step 2: BPR link travel time ----------------------------------
        # t_a = fft_a * (1 + alpha * (x_a / c_a)^beta)
        cap = network.link_capacity[:, np.newaxis]       # [n_links, 1]
        fft = network.link_fft[:, np.newaxis]            # [n_links, 1]

        # Guard against division by zero (shouldn't happen with real data)
        ratio = np.divide(
            link_flow, cap,
            out=np.zeros_like(link_flow),
            where=(cap > 0),
        )
        link_tt = fft * (1.0 + self.alpha * np.power(ratio, self.beta))

        # --- Step 3: path travel time = sum of link travel times -----------
        path_tt = delta_T @ link_tt                      # [n_paths, n_steps]

        return path_tt


# ---------------------------------------------------------------------------
# Arrival penalty / schedule delay
# ---------------------------------------------------------------------------

def compute_arrival_penalty(
    travel_times: np.ndarray,
    params: SimulationParams,
    demand: DemandData,
    network: NetworkData,
    gamma_early: float = 0.8,
    gamma_late: float = 1.8,
) -> np.ndarray:
    """Compute schedule-delay penalty (early + late arrival).

    Mirrors the MATLAB logic::

        arrival_time(r, t) = t + travel_time(r, t)
        penalty = gamma_early * max(0, T_A(od) - arrival)   [early]
                + gamma_late  * max(0, arrival - T_A(od))    [late]

    Parameters
    ----------
    travel_times : [n_paths, n_steps]  path travel times in seconds.
    params       : simulation parameters (provides the time grid).
    demand       : demand data (provides target arrival per OD).
    network      : network data (provides path→OD mapping via
                   ``path_sink_node``).
    gamma_early  : penalty coefficient for early arrival.
    gamma_late   : penalty coefficient for late arrival.

    Returns
    -------
    effective_delay : [n_paths, n_steps]  travel_time + penalty.
    """
    # Build time grid: departure times for each time step
    time_grid = np.linspace(
        params.time_horizon[0], params.time_horizon[1], params.n_steps,
        endpoint=False,
    )  # [n_steps]

    # Arrival time for every (path, time_step)
    arrival_time = time_grid[np.newaxis, :] + travel_times  # [n_paths, n_steps]

    # Build per-path target arrival vector from OD data
    # Each path belongs to exactly one OD pair; use od_path_set to map
    target_per_path = np.empty(network.n_paths, dtype=np.float64)
    for k, path_indices in enumerate(demand.od_path_set):
        target_per_path[path_indices] = demand.target_arrival[k]

    ta = target_per_path[:, np.newaxis]  # [n_paths, 1]

    # Penalty
    diff = arrival_time - ta
    penalty = np.where(diff > 0, gamma_late * diff, -gamma_early * diff)

    return travel_times + penalty


# ---------------------------------------------------------------------------
# Window aggregation
# ---------------------------------------------------------------------------

def aggregate_to_windows(
    costs_fine: np.ndarray,
    params: SimulationParams,
) -> np.ndarray:
    """Average fine-grained costs into departure-time windows.

    Parameters
    ----------
    costs_fine : [n_paths, n_steps]  per-time-step costs.
    params     : provides ``n_windows`` and ``steps_per_window``.

    Returns
    -------
    costs_agg : [n_paths, n_windows]  window-averaged costs.
    """
    n_paths = costs_fine.shape[0]
    spw = params.steps_per_window
    nw = params.n_windows

    # Reshape into (n_paths, n_windows, steps_per_window) and mean
    truncated = costs_fine[:, : nw * spw]
    return truncated.reshape(n_paths, nw, spw).mean(axis=2)
