"""
Day-to-Day (DTD) dynamic traffic assignment simulator.

Ports the logic from Base_Model_I.m, Base_Model_I_BR.m, and
Base_Model_I_IS.m into a unified Python class.

Reference
---------
Yu, Y., Han, K., Ochieng, W.Y., 2020. Day-to-Day Dynamic Traffic
Assignment with Imperfect Information, Bounded Rationality and
Information Sharing. Transportation Research Part C, 114, 59-83.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from dnl.dnl import BaseDNL, aggregate_to_windows, compute_arrival_penalty
from network.network import DemandData, NetworkData, SimulationParams


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DTDParams:
    """Day-to-day simulation parameters."""

    n_days: int = 200
    memory_days: int = 3            # N in MATLAB
    memory_weight: float = 0.7      # lambda in MATLAB
    theta: float = 0.02             # logit dispersion
    demand_factor: float = 1.0      # OD demand scaling factor

    # Bounded rationality (Model I + BR)
    use_bounded_rationality: bool = False
    indifference_band: float = 400.0   # delta [s]

    # Information sharing (Model I + IS)
    use_info_sharing: bool = False
    is_alpha: float = 2.0              # flow-split power

    # Schedule delay coefficients
    gamma_early: float = 0.8
    gamma_late: float = 1.8

    # Disruption window (1-based day numbers, matching MATLAB convention)
    disruption_days: tuple[int, int] = (51, 100)


# ---------------------------------------------------------------------------
# Per-day result
# ---------------------------------------------------------------------------

@dataclass
class DayResult:
    """State snapshot for one simulated day."""

    path_flows: np.ndarray          # [n_paths, n_steps] departures used today
    agg_path_flow: np.ndarray       # [n_paths, n_windows] aggregated flow
    perceived_cost: np.ndarray      # [n_paths, n_windows]
    agg_cost: np.ndarray            # [n_paths, n_windows] effective delay
    travel_times: np.ndarray        # [n_paths, n_steps] raw travel times
    tstt: float                     # total system travel time [veh-hours]


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class DTDSimulator:
    """Unified DTD simulator supporting MNL, BR, and IS variants.

    Parameters
    ----------
    network : NetworkData
        Base (undisrupted) network.
    demand : DemandData
        OD demand structure.
    dnl : BaseDNL
        Dynamic network loading implementation (e.g. BPRDNL).
    dtd_params : DTDParams
        Day-to-day model parameters.
    sim_params : SimulationParams
        Within-day temporal discretisation.
    disrupted_network : NetworkData, optional
        Network with degraded link properties for disruption days.
    """

    def __init__(
        self,
        network: NetworkData,
        demand: DemandData,
        dnl: BaseDNL,
        dtd_params: DTDParams,
        sim_params: SimulationParams,
        disrupted_network: Optional[NetworkData] = None,
    ):
        self.network = network
        self.demand = demand
        self.dnl = dnl
        self.dtd_params = dtd_params
        self.sim_params = sim_params
        self.disrupted_network = disrupted_network

        # Scaled OD demand
        self._od_demand = demand.od_demand * dtd_params.demand_factor

        # Pre-build per-path target arrival (avoids recomputing each day)
        self._target_per_path = np.empty(network.n_paths, dtype=np.float64)
        for k, paths in enumerate(demand.od_path_set):
            self._target_per_path[paths] = demand.target_arrival[k]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        callbacks: Optional[Sequence[Callable[[int, DayResult], None]]] = None,
    ) -> list[DayResult]:
        """Execute the full day-to-day simulation.

        Parameters
        ----------
        callbacks : list of callables ``(day_index, DayResult) -> None``
            Invoked after each day for logging / early stopping.

        Returns
        -------
        list[DayResult] of length ``dtd_params.n_days``.
        """
        p = self.dtd_params
        sp = self.sim_params

        # --- Initialise from the .mat initial path departures --------
        path_dep = sp.initial_path_departures.copy() * p.demand_factor
        current_agg_flow = aggregate_to_windows(path_dep, sp)

        # History buffers
        agg_cost_history: list[np.ndarray] = []
        flow_split_history: list[np.ndarray] = []

        results: list[DayResult] = []

        for day in range(p.n_days):
            # 1. Select network (disruption check) --------------------
            is_disrupted = (
                self.disrupted_network is not None
                and p.disruption_days[0] <= (day + 1) <= p.disruption_days[1]
            )
            net = self.disrupted_network if is_disrupted else self.network

            # 2. DNL ---------------------------------------------------
            travel_times = self.dnl.compute_costs(path_dep, net, sp)

            # 3. Arrival penalty → effective delay ---------------------
            eff_delay = compute_arrival_penalty(
                travel_times, sp, self.demand, self.network,
                gamma_early=p.gamma_early, gamma_late=p.gamma_late,
            )

            # 4. Aggregate to departure windows -----------------------
            agg_cost = aggregate_to_windows(eff_delay, sp)
            agg_cost_history.append(agg_cost)

            # 5. Flow split (needed for IS) ---------------------------
            if p.use_info_sharing:
                fs = self._compute_flow_split(current_agg_flow)
                flow_split_history.append(fs)

            # 6. Perceived cost ---------------------------------------
            if p.use_info_sharing:
                perceived = self._perceived_cost_is(
                    day, agg_cost_history, flow_split_history,
                )
            else:
                perceived = self._perceived_cost_standard(
                    day, agg_cost_history,
                )

            # 7. Choice model → next day's aggregated flow ------------
            if p.use_bounded_rationality:
                new_agg_flow = self._route_choice_br(
                    perceived, current_agg_flow,
                )
            else:
                new_agg_flow = self._route_choice_mnl(perceived)

            # 8. Disaggregate to fine time-steps ----------------------
            new_path_dep = self._disaggregate(new_agg_flow)

            # 9. TSTT -------------------------------------------------
            tstt = self.compute_tstt(travel_times, path_dep)

            # 10. Record ----------------------------------------------
            result = DayResult(
                path_flows=path_dep,
                agg_path_flow=current_agg_flow,
                perceived_cost=perceived,
                agg_cost=agg_cost,
                travel_times=travel_times,
                tstt=tstt,
            )
            results.append(result)

            if callbacks:
                for cb in callbacks:
                    cb(day, result)

            # 11. Advance to next day ---------------------------------
            current_agg_flow = new_agg_flow
            path_dep = new_path_dep

        return results

    # ------------------------------------------------------------------
    # TSTT
    # ------------------------------------------------------------------

    @staticmethod
    def compute_tstt(
        travel_times: np.ndarray,
        path_departures: np.ndarray,
        dt: float = 180.0,
    ) -> float:
        """Total System Travel Time in vehicle-hours.

        TSTT = sum_{r,t} travel_time(r,t) * path_departures(r,t) * dt / 3600
        """
        return float(np.sum(travel_times * path_departures) * dt / 3600.0)

    # ------------------------------------------------------------------
    # Perceived cost — standard exponential memory (Model I / I+BR)
    # ------------------------------------------------------------------

    def _perceived_cost_standard(
        self,
        day: int,
        agg_cost_history: list[np.ndarray],
    ) -> np.ndarray:
        """Exponential-memory perceived cost.

        PC = [(1-λ)/(1-λ^K)] * Σ_{d=0}^{K-1} λ^d * aggE[day-d]

        where K = min(day+1, memory_days).

        Matches Base_Model_I.m lines 110-122.
        """
        lam = self.dtd_params.memory_weight
        N = self.dtd_params.memory_days
        n_use = min(day + 1, N)

        weighted = agg_cost_history[day].copy()
        for d in range(1, n_use):
            weighted += lam ** d * agg_cost_history[day - d]

        # Normalisation: handles λ=1 edge case
        if abs(lam - 1.0) < 1e-15:
            norm = 1.0 / n_use
        else:
            norm = (1.0 - lam) / (1.0 - lam ** n_use)

        return norm * weighted

    # ------------------------------------------------------------------
    # Perceived cost — information sharing (Model I + IS)
    # ------------------------------------------------------------------

    def _perceived_cost_is(
        self,
        day: int,
        agg_cost_history: list[np.ndarray],
        flow_split_history: list[np.ndarray],
    ) -> np.ndarray:
        """Flow-weighted perceived cost for the IS variant.

        weight(d) = flow_split[day-d]^α * λ^d
        PC = Σ weight(d) * aggE[day-d]  /  Σ weight(d)

        Matches Base_Model_I_IS.m lines 110-140.

        Note: the original MATLAB ``else`` branch (T < N) uses λ^(T-1)
        instead of λ^(T-i).  This appears to be a bug; we use the
        correct λ^d form in both branches for consistency.
        """
        if day == 0:
            return agg_cost_history[0].copy()

        lam = self.dtd_params.memory_weight
        N = self.dtd_params.memory_days
        alpha = self.dtd_params.is_alpha
        n_use = min(day + 1, N)

        w0 = np.power(flow_split_history[day], alpha)  # λ^0 = 1
        weighted_cost = w0 * agg_cost_history[day]
        sum_weight = w0.copy()

        for d in range(1, n_use):
            w = np.power(flow_split_history[day - d], alpha) * (lam ** d)
            weighted_cost += w * agg_cost_history[day - d]
            sum_weight += w

        # Avoid 0/0 where flow_split was zero for all remembered days
        return np.divide(
            weighted_cost, sum_weight,
            out=agg_cost_history[day].copy(),
            where=(sum_weight > 1e-30),
        )

    # ------------------------------------------------------------------
    # Route choice — joint MNL (Model I, Model I + IS)
    # ------------------------------------------------------------------

    def _route_choice_mnl(self, perceived_cost: np.ndarray) -> np.ndarray:
        """Joint multinomial logit over (path, window) pairs.

        P(j,k) = exp(-θ·PC(j,k)) / Σ exp(-θ·PC(j',k'))
        flow(j,k) = OD_demand · P(j,k) / DT

        Matches Base_Model_I.m lines 124-132.
        """
        theta = self.dtd_params.theta
        DT = self.sim_params.departure_window
        n_paths = self.network.n_paths
        n_win = self.sim_params.n_windows

        new_flow = np.zeros((n_paths, n_win), dtype=np.float64)

        for i, paths in enumerate(self.demand.od_path_set):
            pc_od = perceived_cost[paths, :]          # [n_p, n_w]
            v = -theta * pc_od
            v -= v.max()                              # numerical stability
            exp_v = np.exp(v)
            prob = exp_v / exp_v.sum()
            new_flow[paths, :] = self._od_demand[i] * prob / DT

        return new_flow

    # ------------------------------------------------------------------
    # Route choice — bounded rationality (Model I + BR)
    # ------------------------------------------------------------------

    def _route_choice_br(
        self,
        perceived_cost: np.ndarray,
        current_agg_flow: np.ndarray,
    ) -> np.ndarray:
        """Bounded-rationality switching model.

        Each traveller has an indifference band δ making their current
        choice appear δ seconds cheaper, creating inertia.

        Vectorised form (per OD pair):
            shift = exp(θ·δ)
            w[m]  = flow[m] / (Den + exp_v[m]·(shift − 1))
            new_flow[m] = exp_v[m] · (w[m]·(shift − 1) + Σw)

        Matches Base_Model_I_BR.m lines 127-146.
        """
        theta = self.dtd_params.theta
        delta = self.dtd_params.indifference_band
        n_paths = self.network.n_paths
        n_win = self.sim_params.n_windows

        shift = np.exp(theta * delta)          # constant across alternatives
        shift_m1 = shift - 1.0

        new_flow = np.zeros((n_paths, n_win), dtype=np.float64)

        for i, paths in enumerate(self.demand.od_path_set):
            pc_od = perceived_cost[paths, :]   # [n_p, n_w]
            fl_od = current_agg_flow[paths, :] # [n_p, n_w]

            pc_flat = pc_od.ravel()
            fl_flat = fl_od.ravel()

            v = -theta * pc_flat
            v_max = v.max()
            exp_v = np.exp(v - v_max)

            Den = exp_v.sum()

            # Modified denominator per alternative:
            #   Den - exp_v[m] + exp_v[m]*shift = Den + exp_v[m]*(shift-1)
            mod_den = Den + exp_v * shift_m1

            w = fl_flat / mod_den
            sum_w = w.sum()

            # new_flow[m] = w[m]·exp_v_delta[m] + exp_v[m]·(Σw − w[m])
            #             = w[m]·(exp_v_delta[m] − exp_v[m]) + exp_v[m]·Σw
            #             = exp_v[m]·(w[m]·shift_m1 + sum_w)
            new_flat = exp_v * (w * shift_m1 + sum_w)

            new_flow[paths, :] = new_flat.reshape(pc_od.shape)

        return new_flow

    # ------------------------------------------------------------------
    # Disaggregation
    # ------------------------------------------------------------------

    def _disaggregate(self, agg_flow: np.ndarray) -> np.ndarray:
        """Distribute window flows uniformly across fine time steps.

        Each window's constant rate is replicated ``steps_per_window``
        times.  Matches Base_Model_I.m lines 135-139.
        """
        return np.repeat(agg_flow, self.sim_params.steps_per_window, axis=1)

    # ------------------------------------------------------------------
    # Flow split (for IS model)
    # ------------------------------------------------------------------

    def _compute_flow_split(self, agg_flow: np.ndarray) -> np.ndarray:
        """Per-OD normalised flow proportions.

        flow_split[paths_of_OD, :] = agg_flow[...] / total_OD_flow

        Matches Base_Model_I_IS.m lines 111-115.
        """
        flow_split = np.zeros_like(agg_flow)
        for k, paths in enumerate(self.demand.od_path_set):
            total = agg_flow[paths, :].sum()
            if total > 0:
                flow_split[paths, :] = agg_flow[paths, :] / total
        return flow_split
