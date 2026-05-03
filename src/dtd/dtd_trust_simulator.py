"""
Integrated DTD-Trust-Attack simulator.

Extends the base DTD day-to-day dynamics with:
- Multi-class heterogeneous drivers (Beta-distribution trust)
- Attack injection (false data injection on information signals)
- Trust-weighted perceived cost (experience vs. information blending)
- Per-class route choice with class-specific indifference bands

Perceived cost update follows Eq. 2 of the paper:
    ĉ_r^{k,(d+1)} = λ_m · ĉ_r^{k,(d)}
                   + (1 - λ_m) · [(1 - λ_k(T_k)) · c̃_r + λ_k(T_k) · I_r]

where c̃_r is the (optionally IS-blended) experienced cost, I_r is the
information signal, and λ_k = λ̄_k · T_k is the trust-dependent reliance.
The blended signal is smoothed into per-class perceived cost memory so
that the information signal accumulates in memory across days.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from dnl.dnl import (
    BaseDNL,
    aggregate_to_windows,
    build_incidence_matrix,
    compute_arrival_penalty,
)
from network.network import DemandData, NetworkData, SimulationParams
from trust.trust_dynamics import TrustDynamics, TrustParams, TrustState
from attack.attack_models import AttackConfig, AttackPolicy, NoAttack


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class IntegratedResult:
    """Results from the integrated DTD-Trust-Attack simulation."""

    tstt: np.ndarray              # [n_days]
    trust: np.ndarray             # [n_days, n_classes]
    info_weight: np.ndarray       # [n_days, n_classes]
    mean_compliance: np.ndarray   # [n_days] weighted-avg info weight
    attack_active: np.ndarray     # [n_days] boolean
    agg_costs: Optional[np.ndarray] = None   # last day [n_paths, n_windows]
    class_flows: Optional[list] = None       # per-class trajectories (optional)
    link_flows_at: Optional[dict] = None     # {day: link_flow_array} for checkpoint days
    guidance_error: Optional[np.ndarray] = None  # [n_days] flow-weighted error


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class DTDTrustSimulator:
    """Integrated DTD simulator with trust dynamics and attack injection.

    Parameters
    ----------
    network : NetworkData
        Base (undisrupted) network.
    demand : DemandData
        OD demand structure.
    dnl : BaseDNL
        Dynamic network loading implementation.
    dtd_params : DTDParams
        Day-to-day model parameters (n_days, theta, memory, disruption, ...).
    sim_params : SimulationParams
        Within-day temporal discretisation.
    trust_params : TrustParams
        Heterogeneous driver class trust parameters.
    attack_policy : AttackPolicy
        Attack strategy (use NoAttack for baseline).
    disrupted_network : NetworkData, optional
        Network with degraded links for physical disruption days.
    """

    def __init__(
        self,
        network: NetworkData,
        demand: DemandData,
        dnl: BaseDNL,
        dtd_params,
        sim_params: SimulationParams,
        trust_params: TrustParams,
        attack_policy: AttackPolicy,
        disrupted_network: Optional[NetworkData] = None,
        use_info_sharing: bool = False,
        is_alpha: float = 2.0,
    ):
        self.network = network
        self.demand = demand
        self.dnl = dnl
        self.dtd_params = dtd_params
        self.sim_params = sim_params
        self.trust_params = trust_params
        self.attack = attack_policy
        self.disrupted_network = disrupted_network
        self.use_info_sharing = use_info_sharing
        self.is_alpha = is_alpha

        # Trust dynamics engine
        self.trust_dyn = TrustDynamics(trust_params)

        # Scaled OD demand
        self._od_demand = demand.od_demand * dtd_params.demand_factor

        # Pre-build incidence matrix for link-flow computation
        self._delta = build_incidence_matrix(network)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        callbacks: Optional[Sequence[Callable]] = None,
        capture_link_flow_days: Optional[Sequence[int]] = None,
    ) -> IntegratedResult:
        """Execute the integrated DTD-Trust-Attack simulation.

        Perceived cost update (paper Eq. 2):
            ĉ_r^{k,(d+1)} = λ_m · ĉ_r^{k,(d)}
                           + (1-λ_m) · [(1-λ_k)·c̃_r + λ_k·I_r]

        The blended signal (experience + trust-weighted info) enters memory,
        so past attack signals persist in perceived costs after attack ends.

        Returns
        -------
        IntegratedResult with full trajectories.
        """
        p = self.dtd_params
        sp = self.sim_params
        tp = self.trust_params
        n_classes = tp.n_classes
        lam_m = p.memory_weight  # λ_m from Eq. 2

        # --- Initialise ---------------------------------------------------
        path_dep = sp.initial_path_departures.copy() * p.demand_factor
        current_agg_flow = aggregate_to_windows(path_dep, sp)
        trust_state = self.trust_dyn.initial_state()

        n_paths = self.network.n_paths
        n_win = sp.n_windows

        # Per-class perceived cost state: ĉ_k^{(d)}
        # Initialised to None; set from first day's experienced cost.
        perceived_cost_class: list[Optional[np.ndarray]] = [None] * n_classes

        # Secondary 6-day rolling average buffer — paper §5.1.2
        # "we apply a secondary six-day rolling average of experienced
        #  costs to smooth within-day stochasticity"
        _rolling_K = p.memory_days          # defaults to 6
        _agg_cost_buffer: list[np.ndarray] = []

        # IS memory buffers (Han et al. 2020 temporal flow-weighted IS)
        # Per-class blended signal history and flow-split history
        _is_N = p.memory_days               # IS memory window = N
        _blended_history: list[list[np.ndarray]] = [[] for _ in range(n_classes)]
        _flow_split_history: list[np.ndarray] = []

        # Output storage
        tstt_out = np.zeros(p.n_days)
        trust_out = np.zeros((p.n_days, n_classes))
        info_weight_out = np.zeros((p.n_days, n_classes))
        compliance_out = np.zeros(p.n_days)
        attack_active_out = np.zeros(p.n_days, dtype=bool)
        guidance_error_out = np.zeros(p.n_days)
        link_flow_capture: dict = {}
        capture_set = set(capture_link_flow_days) if capture_link_flow_days else set()
        last_agg_cost = None

        for day in range(p.n_days):
            # 1. Select network (disruption check) -------------------------
            is_disrupted = (
                self.disrupted_network is not None
                and p.disruption_days[0] <= (day + 1) <= p.disruption_days[1]
            )
            net = self.disrupted_network if is_disrupted else self.network

            # 2. DNL -------------------------------------------------------
            travel_times = self.dnl.compute_costs(path_dep, net, sp)

            # 3. Arrival penalty -> effective delay -------------------------
            eff_delay = compute_arrival_penalty(
                travel_times, sp, self.demand, self.network,
                gamma_early=p.gamma_early, gamma_late=p.gamma_late,
            )

            # 4. Aggregate to departure windows -----------------------------
            raw_agg_cost = aggregate_to_windows(eff_delay, sp)

            # 4b. Secondary rolling average — paper §5.1.2 -----------------
            #     Smooths within-day LWR stochasticity (CV 6.2% → 0.13%)
            _agg_cost_buffer.append(raw_agg_cost)
            if len(_agg_cost_buffer) > _rolling_K:
                _agg_cost_buffer.pop(0)
            agg_cost = np.mean(_agg_cost_buffer, axis=0)

            # 5. Experienced cost — no spatial IS blend needed
            #    Han's IS is applied in the perceived cost memory (step 10)
            experienced_cost = agg_cost

            # 6. Attack injection — paper Eq. 13 ----------------------------
            link_flow = np.asarray(self._delta @ path_dep).max(axis=1)
            if day in capture_set:
                link_flow_capture[day] = link_flow.copy()
            state_dict = {
                'path_link_matrix': self.network.path_link_matrix,
                'link_flows': link_flow,
                'od_path_set': self.demand.od_path_set,
                'network': self.network,
                'incidence_matrix': self._delta,
            }
            perturbation = self.attack.generate(day, agg_cost, state_dict)
            info_signal = agg_cost + perturbation
            # I_r^{(d)} = c_r(f^{(d)}) + a_r(f^{(d)})

            # 7. Guidance error per class — paper Eq. 5 --------------------
            errors = self._compute_class_errors(
                agg_cost, info_signal, current_agg_flow,
            )
            guidance_error_out[day] = float(errors[0])

            # 8. Information weights BEFORE trust update -------------------
            lambda_k = self.trust_dyn.information_weight(trust_state)

            # 14a. Record PRE-update trust ---------------------------------
            trust_out[day] = trust_state.expected_trust

            # 9. Trust update — paper Eq. 8 --------------------------------
            trust_state = self.trust_dyn.update(trust_state, errors)

            # 10. Per-class trust-weighted blending + perceived cost --------
            #     blended_k = (1-λ_k)·c_exp + λ_k·I_r
            #     Then: IS memory (Han 2020) or simple EMA

            # 10a. Compute flow split for IS memory
            if self.use_info_sharing:
                fs = self._compute_flow_split(current_agg_flow)
                _flow_split_history.append(fs)
                if len(_flow_split_history) > _is_N:
                    _flow_split_history.pop(0)

            # 10b. Per-class blending and memory
            for k in range(n_classes):
                lk = lambda_k[k]
                blended_k = (1.0 - lk) * experienced_cost + lk * info_signal

                if self.use_info_sharing:
                    # Store blended signal in per-class history
                    _blended_history[k].append(blended_k)
                    if len(_blended_history[k]) > _is_N:
                        _blended_history[k].pop(0)

                    # Han's IS: flow-weighted temporal averaging
                    perceived_cost_class[k] = self._perceived_cost_is_han(
                        _blended_history[k], _flow_split_history,
                    )
                else:
                    # Standard exponential memory (EMA)
                    if perceived_cost_class[k] is None:
                        perceived_cost_class[k] = blended_k.copy()
                    else:
                        perceived_cost_class[k] = (
                            lam_m * perceived_cost_class[k]
                            + (1.0 - lam_m) * blended_k
                        )

            # 11. Multi-class route choice using perceived costs ------------
            new_agg_flow = self._multi_class_choice_from_perceived(
                perceived_cost_class, current_agg_flow,
            )

            # 12. Disaggregate to fine time-steps ---------------------------
            new_path_dep = np.repeat(
                new_agg_flow, sp.steps_per_window, axis=1,
            )

            # 13. TSTT ------------------------------------------------------
            tstt = float(
                np.sum(travel_times * path_dep) * sp.dt / 3600.0
            )

            # 14b. Record remaining outputs ---------------------------------
            tstt_out[day] = tstt
            info_weight_out[day] = lambda_k
            compliance_out[day] = float(np.dot(tp.class_fractions, lambda_k))
            attack_active_out[day] = self.attack.is_active(day)
            last_agg_cost = agg_cost

            if callbacks:
                for cb in callbacks:
                    cb(day, tstt, trust_state)

            # 15. Advance to next day ---------------------------------------
            current_agg_flow = new_agg_flow
            path_dep = new_path_dep

        return IntegratedResult(
            tstt=tstt_out,
            trust=trust_out,
            info_weight=info_weight_out,
            mean_compliance=compliance_out,
            attack_active=attack_active_out,
            agg_costs=last_agg_cost,
            link_flows_at=link_flow_capture if link_flow_capture else None,
            guidance_error=guidance_error_out,
        )

    # ------------------------------------------------------------------
    # Han (2020) IS: flow-weighted temporal memory
    # ------------------------------------------------------------------

    def _perceived_cost_is_han(
        self,
        blended_history: list[np.ndarray],
        flow_split_history: list[np.ndarray],
    ) -> np.ndarray:
        """Han et al. (2020) IS perceived cost (Base_Model_I_IS.m).

        weight(d) = flow_split[current-d]^α × λ_m^d
        PC = Σ weight(d) × blended(current-d)  /  Σ weight(d)

        α is a power exponent on flow shares — NOT a linear blend.
        Stable for any α ≥ 0. Higher α gives more weight to
        high-traffic paths in the temporal memory.

        Parameters
        ----------
        blended_history : recent trust-blended signals (newest = [-1])
        flow_split_history : recent flow splits (newest = [-1])
        """
        n_use = min(len(blended_history), len(flow_split_history))
        if n_use == 0:
            raise ValueError("Empty history")
        if n_use == 1:
            return blended_history[-1].copy()

        alpha = self.is_alpha
        lam = self.dtd_params.memory_weight

        # d=0 is current day (index -1)
        w0 = np.power(flow_split_history[-1], alpha)
        weighted_cost = w0 * blended_history[-1]
        sum_weight = w0.copy()

        for d in range(1, n_use):
            w = np.power(flow_split_history[-1 - d], alpha) * (lam ** d)
            weighted_cost += w * blended_history[-1 - d]
            sum_weight += w

        return np.divide(
            weighted_cost, sum_weight,
            out=blended_history[-1].copy(),
            where=(sum_weight > 1e-30),
        )

    # ------------------------------------------------------------------
    # Guidance error
    # ------------------------------------------------------------------

    def _compute_class_errors(
        self,
        true_cost: np.ndarray,
        info_signal: np.ndarray,
        current_agg_flow: np.ndarray,
    ) -> np.ndarray:
        """Compute guidance error per class — paper Eq. 5.

        e_k = (1/Q_k) Σ_w Σ_r f_r^{k,(d)} |I_r - c_r|

        With proportional class flows (f_r^k = π_k · f_r), the error
        becomes a flow-weighted mean that is class-independent:
        e_k = Σ f_r |I_r - c_r| / Σ f_r  ∀k

        Returns
        -------
        np.ndarray, shape [n_classes]
        """
        abs_perturbation = np.abs(info_signal - true_cost)  # [n_paths, n_windows]
        total_flow = current_agg_flow.sum()
        if total_flow > 1e-30:
            flow_weighted_error = float(
                np.sum(current_agg_flow * abs_perturbation) / total_flow
            )
        else:
            flow_weighted_error = float(np.mean(abs_perturbation))
        return np.full(self.trust_params.n_classes, flow_weighted_error)

    # ------------------------------------------------------------------
    # Multi-class route choice using per-class perceived costs
    # ------------------------------------------------------------------

    def _multi_class_choice_from_perceived(
        self,
        perceived_cost_class: list[np.ndarray],
        current_agg_flow: np.ndarray,
    ) -> np.ndarray:
        """Multi-class route choice from per-class perceived costs.

        Each class k uses its own perceived cost ĉ_k (already blended
        and smoothed via Eq. 2) for route choice.

        Total flow = sum_k (π_k · flow_k(ĉ_k))
        """
        tp = self.trust_params
        total_flow = np.zeros_like(perceived_cost_class[0])

        for k in range(tp.n_classes):
            perceived_k = perceived_cost_class[k]
            frac = tp.class_fractions[k]
            delta_k = tp.indifference_band[k]

            if delta_k > 0:
                # Bounded rationality: redistribute class-proportional flow
                class_flow = frac * current_agg_flow
                flow_k = self._route_choice_br(
                    perceived_k, class_flow, delta_k,
                )
            else:
                # Pure MNL: distribute fraction of OD demand
                flow_k = frac * self._route_choice_mnl(perceived_k)

            total_flow += flow_k

        return total_flow

    # ------------------------------------------------------------------
    # Route choice — MNL
    # ------------------------------------------------------------------

    def _route_choice_mnl(self, perceived_cost: np.ndarray) -> np.ndarray:
        """Joint multinomial logit over (path, window) pairs."""
        theta = self.dtd_params.theta
        DT = self.sim_params.departure_window
        n_paths = self.network.n_paths
        n_win = self.sim_params.n_windows

        new_flow = np.zeros((n_paths, n_win), dtype=np.float64)

        for i, paths in enumerate(self.demand.od_path_set):
            pc_od = perceived_cost[paths, :]
            v = -theta * pc_od
            v -= v.max()
            exp_v = np.exp(v)
            prob = exp_v / exp_v.sum()
            new_flow[paths, :] = self._od_demand[i] * prob / DT

        return new_flow

    # ------------------------------------------------------------------
    # Route choice — bounded rationality
    # ------------------------------------------------------------------

    def _route_choice_br(
        self,
        perceived_cost: np.ndarray,
        current_flow: np.ndarray,
        delta: float,
    ) -> np.ndarray:
        """Bounded-rationality switching model for a single class."""
        theta = self.dtd_params.theta
        n_paths = self.network.n_paths
        n_win = self.sim_params.n_windows

        shift = np.exp(theta * delta)
        shift_m1 = shift - 1.0

        new_flow = np.zeros((n_paths, n_win), dtype=np.float64)

        for i, paths in enumerate(self.demand.od_path_set):
            pc_od = perceived_cost[paths, :]
            fl_od = current_flow[paths, :]

            pc_flat = pc_od.ravel()
            fl_flat = fl_od.ravel()

            v = -theta * pc_flat
            v_max = v.max()
            exp_v = np.exp(v - v_max)

            Den = exp_v.sum()
            mod_den = Den + exp_v * shift_m1

            w = fl_flat / mod_den
            sum_w = w.sum()

            new_flat = exp_v * (w * shift_m1 + sum_w)
            new_flow[paths, :] = new_flat.reshape(pc_od.shape)

        return new_flow

    # ------------------------------------------------------------------
    # Flow split (for IS model)
    # ------------------------------------------------------------------

    def _compute_flow_split(self, agg_flow: np.ndarray) -> np.ndarray:
        """Per-OD normalised flow proportions."""
        flow_split = np.zeros_like(agg_flow)
        for k, paths in enumerate(self.demand.od_path_set):
            total = agg_flow[paths, :].sum()
            if total > 0:
                flow_split[paths, :] = agg_flow[paths, :] / total
        return flow_split
