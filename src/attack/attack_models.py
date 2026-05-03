"""Attack models for false data injection on route guidance information.

The attack modifies the information signal that drivers receive:
    I(d) = C(d) + a(d)
where C(d) is the true cost and a(d) is the adversarial perturbation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np


@dataclass
class AttackConfig:
    """Common attack configuration."""

    target_links: Optional[np.ndarray] = None  # indices of targeted links (None=all)
    bias_fraction: float = 0.3       # perturbation as fraction of true cost
    start_day: int = 50
    end_day: int = 100
    budget: float = np.inf           # max total perturbation norm

    def __post_init__(self):
        if self.target_links is not None:
            self.target_links = np.asarray(self.target_links, dtype=np.intp)


class AttackPolicy(ABC):
    """Base class for attack strategies."""

    def __init__(self, config: AttackConfig):
        self.config = config

    @abstractmethod
    def generate(self, day: int, true_costs: np.ndarray,
                 state: Optional[dict] = None) -> np.ndarray:
        """Return perturbation a(d) with same shape as true_costs.

        Parameters
        ----------
        day : int
            Current simulation day (0-indexed).
        true_costs : np.ndarray
            True path costs, shape [n_paths, n_windows] or [n_paths, n_steps].
        state : dict, optional
            Additional state info. May contain:
            - 'path_link_matrix': [n_paths, max_links], 0-indexed, -1=pad
            - 'link_flows': [n_links] current link flows

        Returns
        -------
        np.ndarray
            Perturbation array, same shape as true_costs.
            Outside [start_day, end_day), returns zeros.
        """

    def is_active(self, day: int) -> bool:
        """Check if attack is active on the given day."""
        return self.config.start_day <= day < self.config.end_day

    def _apply_budget(self, perturbation: np.ndarray) -> np.ndarray:
        """Clip perturbation so its L2 norm does not exceed budget."""
        if not np.isfinite(self.config.budget):
            return perturbation
        norm = np.linalg.norm(perturbation)
        if norm > self.config.budget:
            perturbation = perturbation * (self.config.budget / norm)
        return perturbation

    def _path_mask(self, state: Optional[dict]) -> np.ndarray:
        """Build a boolean mask of paths that use any targeted link.

        Parameters
        ----------
        state : dict
            Must contain 'path_link_matrix' [n_paths, max_links].

        Returns
        -------
        np.ndarray
            Boolean array of shape [n_paths].
        """
        if state is None or 'path_link_matrix' not in state:
            raise ValueError(
                "Targeted attack requires state['path_link_matrix']"
            )
        plm = state['path_link_matrix']  # [n_paths, max_links], -1=pad
        target_set = set(self.config.target_links.tolist())
        # For each path, check if any of its links is in the target set
        mask = np.zeros(plm.shape[0], dtype=bool)
        for i in range(plm.shape[0]):
            links = plm[i]
            links = links[links >= 0]  # remove padding
            if any(l in target_set for l in links):
                mask[i] = True
        return mask


class NoAttack(AttackPolicy):
    """Baseline: no attack."""

    def __init__(self, config: Optional[AttackConfig] = None):
        if config is None:
            config = AttackConfig()
        super().__init__(config)

    def generate(self, day, true_costs, state=None):
        return np.zeros_like(true_costs)


class ConstantFDI(AttackPolicy):
    """Asymmetric false data injection that reverses cost signals.

    For each OD pair, inverts the relative cost ordering:
      - Efficient (below-mean) paths are INFLATED  -> push traffic away
      - Inefficient (above-mean) paths are DEFLATED -> attract traffic

    perturbation[paths_of_OD] = -bias_fraction * (C - mean_C)

    Requires state['od_path_set'] (list of arrays of path indices per OD).
    """

    def generate(self, day, true_costs, state=None):
        if not self.is_active(day):
            return np.zeros_like(true_costs)

        perturbation = np.zeros_like(true_costs)

        if state is not None and 'od_path_set' in state:
            for path_indices in state['od_path_set']:
                od_costs = true_costs[path_indices]
                mean_cost = od_costs.mean(axis=0, keepdims=True)
                diff = od_costs - mean_cost
                perturbation[path_indices] = (
                    -self.config.bias_fraction * diff
                )
        else:
            # Fallback: treat all paths as one group
            mean_cost = true_costs.mean(axis=0, keepdims=True)
            perturbation = -self.config.bias_fraction * (
                true_costs - mean_cost
            )

        perturbation = self._apply_budget(perturbation)
        return perturbation


class AsymmetricFDI(AttackPolicy):
    """Simple asymmetric attack: inflate the best paths per OD pair.

    For each OD pair, identifies the top-K cheapest paths and inflates
    their reported costs by +bias_fraction, leaving others unchanged.
    Easier to explain than full cost-reversal.

    Parameters
    ----------
    config : AttackConfig
    top_k : int
        Number of cheapest paths per OD to inflate (default 2).
    """

    def __init__(self, config: AttackConfig, top_k: int = 2):
        super().__init__(config)
        self.top_k = top_k

    def generate(self, day, true_costs, state=None):
        if not self.is_active(day):
            return np.zeros_like(true_costs)

        perturbation = np.zeros_like(true_costs)

        if state is not None and 'od_path_set' in state:
            for path_indices in state['od_path_set']:
                od_costs = true_costs[path_indices]
                # Mean cost across windows for ranking
                mean_per_path = od_costs.mean(axis=-1) if od_costs.ndim > 1 else od_costs
                k = min(self.top_k, len(path_indices))
                best_local = np.argsort(mean_per_path)[:k]
                best_paths = np.asarray(path_indices)[best_local]
                perturbation[best_paths] = (
                    self.config.bias_fraction * true_costs[best_paths]
                )
        else:
            mean_per_path = true_costs.mean(axis=-1) if true_costs.ndim > 1 else true_costs
            k = min(self.top_k, len(true_costs))
            best = np.argsort(mean_per_path)[:k]
            perturbation[best] = self.config.bias_fraction * true_costs[best]

        perturbation = self._apply_budget(perturbation)
        return perturbation


class EscalatingAttack(AttackPolicy):
    """Two-phase attack: low intensity first (erode trust),
    then high intensity.

    Phase 1: days [start, mid) at phase1_fraction
    Phase 2: days [mid, end) at full bias_fraction
    """

    def __init__(self, config: AttackConfig, phase1_fraction: float = 0.1,
                 mid_day: Optional[int] = None):
        super().__init__(config)
        self.phase1_fraction = phase1_fraction
        self.mid_day = mid_day if mid_day is not None else (
            config.start_day + config.end_day) // 2

    def generate(self, day, true_costs, state=None):
        if not self.is_active(day):
            return np.zeros_like(true_costs)

        if day < self.mid_day:
            fraction = self.phase1_fraction
        else:
            fraction = self.config.bias_fraction

        perturbation = fraction * true_costs

        if self.config.target_links is not None:
            mask = self._path_mask(state)
            if true_costs.ndim == 1:
                perturbation *= mask
            else:
                perturbation *= mask[:, np.newaxis]

        perturbation = self._apply_budget(perturbation)
        return perturbation


class IntermittentAttack(AttackPolicy):
    """Attack on/off pattern.

    Active for on_days, then off for off_days, repeating
    within the [start_day, end_day) window.
    """

    def __init__(self, config: AttackConfig, on_days: int = 3,
                 off_days: int = 2):
        super().__init__(config)
        self.on_days = on_days
        self.off_days = off_days

    def _is_on(self, day: int) -> bool:
        """Check if the attack is in an 'on' phase on the given day."""
        if not self.is_active(day):
            return False
        cycle_pos = (day - self.config.start_day) % (self.on_days + self.off_days)
        return cycle_pos < self.on_days

    def generate(self, day, true_costs, state=None):
        if not self._is_on(day):
            return np.zeros_like(true_costs)

        perturbation = self.config.bias_fraction * true_costs

        if self.config.target_links is not None:
            mask = self._path_mask(state)
            if true_costs.ndim == 1:
                perturbation *= mask
            else:
                perturbation *= mask[:, np.newaxis]

        perturbation = self._apply_budget(perturbation)
        return perturbation


class TargetedBottleneckAttack(AttackPolicy):
    """Automatically targets the most congested links.

    Uses current flow state to identify high-impact targets,
    then applies FDI on paths using those links.
    """

    def __init__(self, config: AttackConfig, n_targets: int = 10):
        super().__init__(config)
        self.n_targets = n_targets

    def generate(self, day, true_costs, state=None):
        if not self.is_active(day):
            return np.zeros_like(true_costs)

        perturbation = self.config.bias_fraction * true_costs

        # Identify top-n congested links from state
        if state is not None and 'link_flows' in state:
            link_flows = state['link_flows']
            top_links = np.argsort(link_flows)[-self.n_targets:]

            if 'path_link_matrix' in state:
                plm = state['path_link_matrix']
                target_set = set(top_links.tolist())
                mask = np.zeros(plm.shape[0], dtype=bool)
                for i in range(plm.shape[0]):
                    links = plm[i]
                    links = links[links >= 0]
                    if any(l in target_set for l in links):
                        mask[i] = True
                if true_costs.ndim == 1:
                    perturbation *= mask
                else:
                    perturbation *= mask[:, np.newaxis]

        perturbation = self._apply_budget(perturbation)
        return perturbation


class LinkDeflationAttack(AttackPolicy):
    """Legacy deflation attack (for comparison).

    Finds top-n most congested links (highest flow/capacity ratio)
    from state dict, then DEFLATES costs on paths using those links
    by -bias_fraction, making congested links appear less congested
    to attract more traffic and worsen congestion.

    Requires state dict with keys:
    - 'link_flows': current link flow array
    - 'network': NetworkData object (for capacity)
    - 'incidence_matrix': sparse path-link incidence matrix
    """

    def __init__(self, config: AttackConfig, n_target_links: int = 10):
        super().__init__(config)
        self.n_target_links = n_target_links

    def generate(self, day, true_costs, state=None):
        if not self.is_active(day):
            return np.zeros_like(true_costs)
        if state is None or 'link_flows' not in state:
            return np.zeros_like(true_costs)

        link_flows = state['link_flows']
        capacities = state['network'].link_capacity
        ratios = link_flows / np.maximum(capacities, 1e-10)
        targets = np.argsort(ratios)[-self.n_target_links:]

        delta_mat = state['incidence_matrix']
        path_mask = np.zeros(true_costs.shape[0])
        for link_idx in targets:
            paths_using = (
                np.asarray(delta_mat[link_idx].toarray()).flatten() > 0
            )
            path_mask[paths_using] = 1.0

        perturbation = (
            -self.config.bias_fraction * true_costs
            * path_mask[:, np.newaxis]
        )
        return self._apply_budget(perturbation)


# Keep backward-compatible alias
LinkLevelFDI = LinkDeflationAttack


class StrategicSybilAttack(AttackPolicy):
    """Realistic Sybil attack based on link structural vulnerability.

    Target selection: links with high (route_importance / capacity).
    - route_importance = number of paths traversing the link
    - capacity = link capacity (low capacity = easier to fake)

    This captures the realistic Sybil attacker's trade-off:
    - High importance: disruption affects more users
    - Low capacity: fewer Sybil devices needed to forge congestion

    Targets are selected ONCE before attack begins and remain FIXED.
    This matches real Sybil attacks where devices are pre-positioned.

    References:
    - Wang et al. (2018): single virtual vehicle creates hotspot on low-traffic road
    - Soderhall et al. (2025): benign users counteract Sybils on busy roads
    - Ben Sinai et al. (2014): targeted attacks on specific roads/businesses
    """

    def __init__(self, config, n_target_links=None):
        super().__init__(config)
        self.n_target_links = n_target_links  # None = auto-determine
        self.targets = None  # selected once, fixed for attack duration
        self._path_mask = None

    def _select_targets(self, state):
        """One-time target selection based on structural vulnerability."""
        delta_mat = state['incidence_matrix']  # sparse, links x paths
        capacities = state['network'].link_capacity
        n_links = delta_mat.shape[0]

        # Route importance: number of paths using each link
        # This is computable from network structure alone (public info)
        route_importance = np.array(
            (delta_mat > 0).sum(axis=1)
        ).flatten().astype(float)

        # Structural vulnerability score
        # High importance + low capacity = best target
        vulnerability = route_importance / np.maximum(capacities, 1e-10)

        # Auto-determine number of targets if not specified
        if self.n_target_links is None:
            n_paths = delta_mat.shape[1]
            sorted_indices = np.argsort(vulnerability)[::-1]
            covered_paths = set()
            selected = []
            for idx in sorted_indices:
                paths_on_link = set(
                    np.where(np.asarray(delta_mat[idx].toarray()).flatten() > 0)[0]
                )
                new_coverage = len(covered_paths | paths_on_link) / n_paths
                selected.append(idx)
                covered_paths |= paths_on_link
                if new_coverage >= 0.50:  # 50% path coverage threshold
                    break
            self.targets = np.array(selected)
        else:
            sorted_indices = np.argsort(vulnerability)[::-1]
            self.targets = sorted_indices[:self.n_target_links]

        # Pre-compute path mask (fixed for entire attack)
        path_mask = np.zeros(delta_mat.shape[1])
        for link_idx in self.targets:
            paths_using = np.asarray(
                delta_mat[link_idx].toarray()
            ).flatten() > 0
            path_mask[paths_using] = 1.0
        self._path_mask = path_mask

        n_paths = delta_mat.shape[1]
        print(f"  StrategicSybil: selected {len(self.targets)} target links "
              f"(vulnerability range: {vulnerability[self.targets[-1]]:.1f} - "
              f"{vulnerability[self.targets[0]]:.1f})")
        print(f"  Path coverage: {self._path_mask.sum():.0f}/{n_paths} "
              f"({100*self._path_mask.sum()/n_paths:.1f}%)")

    def generate(self, day, true_costs, state=None):
        if not self.is_active(day):
            return np.zeros_like(true_costs)
        if state is None or 'incidence_matrix' not in state:
            return np.zeros_like(true_costs)

        # Select targets once (first active day)
        if self.targets is None:
            self._select_targets(state)

        # Fixed inflation on fixed targets
        perturbation = (self.config.bias_fraction * true_costs
                       * self._path_mask[:, np.newaxis])
        return perturbation


class RandomSybilAttack(AttackPolicy):
    """Baseline: attacker with no network knowledge.

    Randomly selects N links as targets (fixed for entire attack).
    This represents a Sybil attacker without strategic information.
    """

    def __init__(self, config, n_target_links=10, seed=42):
        super().__init__(config)
        self.n_target_links = n_target_links
        self.seed = seed
        self.targets = None
        self._path_mask = None

    def _select_targets(self, state):
        rng = np.random.RandomState(self.seed)
        delta_mat = state['incidence_matrix']
        n_links = delta_mat.shape[0]
        self.targets = rng.choice(n_links, self.n_target_links, replace=False)
        # Build path mask (same as StrategicSybil)
        path_mask = np.zeros(delta_mat.shape[1])
        for link_idx in self.targets:
            paths_using = np.asarray(delta_mat[link_idx].toarray()).flatten() > 0
            path_mask[paths_using] = 1.0
        self._path_mask = path_mask
        print(f"  RandomSybil: selected {len(self.targets)} random links")
        print(f"  Path coverage: {self._path_mask.sum():.0f}/{delta_mat.shape[1]}")

    def generate(self, day, true_costs, state=None):
        if not self.is_active(day):
            return np.zeros_like(true_costs)
        if state is None:
            return np.zeros_like(true_costs)
        if self.targets is None:
            self._select_targets(state)
        perturbation = self.config.bias_fraction * true_costs * self._path_mask[:, np.newaxis]
        return perturbation


class SybilInflationAttack(AttackPolicy):
    """Realistic Sybil attack based on demonstrated attacks on Waze/Google Maps.

    Mechanism: Attacker injects fake slow-speed GPS data on target links,
    making them appear congested. The navigation platform inflates the
    reported travel times for those links. Users avoid targeted links
    and reroute to alternatives, causing congestion elsewhere.

    Target selection: Top-N links by flow/capacity ratio (bottlenecks).
    These are publicly observable from navigation app congestion displays.

    References:
    - Wang et al. (2018) IEEE/ACM ToN: Ghost Riders, Sybil attacks on Waze
    - Eryonucu & Papadimitratos (2022) WiSec: Sybil attacks on Google Maps
    - Soderhall et al. (2025): 3% Sybils cause 20% travel time increase
    """

    def __init__(self, config: AttackConfig, n_target_links: int = 10):
        super().__init__(config)
        self.n_target_links = n_target_links

    def generate(self, day, true_costs, state=None):
        if not self.is_active(day):
            return np.zeros_like(true_costs)
        if state is None or 'link_flows' not in state:
            return np.zeros_like(true_costs)

        link_flows = state['link_flows']
        capacities = state['network'].link_capacity
        ratios = link_flows / np.maximum(capacities, 1e-10)
        targets = np.argsort(ratios)[-self.n_target_links:]

        delta_mat = state['incidence_matrix']
        path_mask = np.zeros(true_costs.shape[0])
        for link_idx in targets:
            paths_using = np.asarray(delta_mat[link_idx].toarray()).flatten() > 0
            path_mask[paths_using] = 1.0

        # INFLATE: positive perturbation on paths using targeted links
        # Makes targeted paths appear MORE expensive -> users avoid them
        # -> reroute to alternatives -> alternatives get congested
        perturbation = self.config.bias_fraction * true_costs * path_mask[:, np.newaxis]
        return self._apply_budget(perturbation)


class BCSybilInflationAttack(AttackPolicy):
    """BC-based Sybil attack: INFLATE travel times on high-BC links.

    Targets links with highest betweenness centrality (path-based).
    BC is computed from network topology alone — no real-time traffic needed.

    Mechanism: Fake congestion reported on backbone links.
    Effect: Users abandon efficient backbone routes → forced onto worse alternatives.

    References:
    - Waniek et al. (2021) Sci Rep: suboptimal heuristic for NP-hard target selection
    - Wang et al. (2018) IEEE/ACM ToN: Ghost Riders demonstrate Sybil feasibility
    - Söderhäll et al. (2025): 3% Sybil → 20% TT increase on alternatives
    """
    def __init__(self, config, n_target_links=10):
        super().__init__(config)
        self.n_target_links = n_target_links
        self.targets = None
        self._path_mask_arr = None

    def _select_targets(self, state):
        delta_mat = state['incidence_matrix']  # sparse, links × paths
        network = state['network']

        # Topological edge betweenness centrality on FFT-weighted digraph
        G = nx.DiGraph()
        for i in range(network.n_links):
            u = int(network.adjacency_list[i, 0])
            v = int(network.adjacency_list[i, 1])
            G.add_edge(u, v, weight=max(float(network.link_fft[i]), 0.001),
                       link_idx=i)
        topo_bc_dict = nx.edge_betweenness_centrality(G, weight='weight',
                                                      normalized=False)
        link_bc = np.zeros(network.n_links)
        for (u, v), bc_val in topo_bc_dict.items():
            link_bc[G[u][v]['link_idx']] = bc_val

        sorted_idx = np.argsort(link_bc)[::-1]
        self.targets = sorted_idx[:self.n_target_links]

        # Build path mask
        n_paths = delta_mat.shape[1]
        path_mask = np.zeros(n_paths)
        for link_idx in self.targets:
            paths_using = np.asarray(delta_mat[link_idx].toarray()).flatten() > 0
            path_mask[paths_using] = 1.0
        self._path_mask_arr = path_mask

        print(f"  BCSybilInflation: {len(self.targets)} high-topoBC links")
        print(f"  Path coverage: {self._path_mask_arr.sum():.0f}/{n_paths} ({100*self._path_mask_arr.sum()/n_paths:.1f}%)")
        print(f"  Target links (0-idx): {list(self.targets)}")

    def generate(self, day, true_costs, state=None):
        if not self.is_active(day):
            return np.zeros_like(true_costs)
        if state is None or 'incidence_matrix' not in state:
            return np.zeros_like(true_costs)
        if self.targets is None:
            self._select_targets(state)
        # INFLATE: make backbone links appear more expensive
        perturbation = self.config.bias_fraction * true_costs * self._path_mask_arr[:, np.newaxis]
        return perturbation


class BCSybilDeflationAttack(AttackPolicy):
    """BC-based Sybil attack: DEFLATE travel times on high-BC links.

    Same BC-based target selection, but OPPOSITE perturbation direction.

    Mechanism: Fake free-flow reported on backbone links.
    Effect: Users over-concentrate on backbone links → overload → congestion.

    Technical feasibility: Wang et al. (2018) showed ghost riders can report
    arbitrary GPS speed — reporting fast speed is identical difficulty to slow speed.

    References:
    - Lin et al. (2018): explicitly modeled deflation attacks on route guidance
    - Wang et al. (2018): ghost riders manipulate GPS speed freely
    """
    def __init__(self, config, n_target_links=10):
        super().__init__(config)
        self.n_target_links = n_target_links
        self.targets = None
        self._path_mask_arr = None

    def _select_targets(self, state):
        delta_mat = state['incidence_matrix']
        network = state['network']

        # Topological edge betweenness centrality on FFT-weighted digraph
        G = nx.DiGraph()
        for i in range(network.n_links):
            u = int(network.adjacency_list[i, 0])
            v = int(network.adjacency_list[i, 1])
            G.add_edge(u, v, weight=max(float(network.link_fft[i]), 0.001),
                       link_idx=i)
        topo_bc_dict = nx.edge_betweenness_centrality(G, weight='weight',
                                                      normalized=False)
        link_bc = np.zeros(network.n_links)
        for (u, v), bc_val in topo_bc_dict.items():
            link_bc[G[u][v]['link_idx']] = bc_val

        sorted_idx = np.argsort(link_bc)[::-1]
        self.targets = sorted_idx[:self.n_target_links]

        n_paths = delta_mat.shape[1]
        path_mask = np.zeros(n_paths)
        for link_idx in self.targets:
            paths_using = np.asarray(delta_mat[link_idx].toarray()).flatten() > 0
            path_mask[paths_using] = 1.0
        self._path_mask_arr = path_mask

        print(f"  BCSybilDeflation: {len(self.targets)} high-topoBC links")
        print(f"  Path coverage: {self._path_mask_arr.sum():.0f}/{n_paths} ({100*self._path_mask_arr.sum()/n_paths:.1f}%)")
        print(f"  Target links (0-idx): {list(self.targets)}")

    def generate(self, day, true_costs, state=None):
        if not self.is_active(day):
            return np.zeros_like(true_costs)
        if state is None or 'incidence_matrix' not in state:
            return np.zeros_like(true_costs)
        if self.targets is None:
            self._select_targets(state)
        # DEFLATE: make backbone links appear cheaper (more attractive)
        perturbation = -self.config.bias_fraction * true_costs * self._path_mask_arr[:, np.newaxis]
        return perturbation


class BCLinkAdditiveDeflationAttack(AttackPolicy):
    """Scenario II: Link-additive deflation on high-BC links — paper Eq. 16.

    a_r(f) = -γ · Σ_{a ∈ r ∩ A_att} c_a(f)

    Only the attacked links' cost contributions are perturbed.
    A path with 1 targeted link out of 5 total gets a smaller discount
    than under Scenario I (route-level).

    Link costs c_a(f) are estimated via BPR from current link flows,
    since the LWR kinematic wave model does not produce individual link costs.

    References:
    - Wang et al. (2018): Ghost Riders, Sybil attacks on Waze
    - Söderhäll et al. (2025): 3% Sybils cause 20% TT increase
    """
    def __init__(self, config, n_target_links=10,
                 bpr_alpha=0.15, bpr_beta=4.0):
        super().__init__(config)
        self.n_target_links = n_target_links
        self.bpr_alpha = bpr_alpha
        self.bpr_beta = bpr_beta
        self.targets = None

    def _select_targets(self, state):
        """Same BC-based target selection as Scenario I."""
        delta_mat = state['incidence_matrix']
        network = state['network']

        G = nx.DiGraph()
        for i in range(network.n_links):
            u = int(network.adjacency_list[i, 0])
            v = int(network.adjacency_list[i, 1])
            G.add_edge(u, v, weight=max(float(network.link_fft[i]), 0.001),
                       link_idx=i)
        topo_bc_dict = nx.edge_betweenness_centrality(G, weight='weight',
                                                      normalized=False)
        link_bc = np.zeros(network.n_links)
        for (u, v), bc_val in topo_bc_dict.items():
            link_bc[G[u][v]['link_idx']] = bc_val

        sorted_idx = np.argsort(link_bc)[::-1]
        self.targets = set(sorted_idx[:self.n_target_links].tolist())

        print(f"  BCLinkAdditiveDeflation: {len(self.targets)} high-topoBC links")
        print(f"  Target links (0-idx): {sorted(self.targets)}")

    def generate(self, day, true_costs, state=None):
        if not self.is_active(day):
            return np.zeros_like(true_costs)
        if state is None or 'incidence_matrix' not in state:
            return np.zeros_like(true_costs)
        if self.targets is None:
            self._select_targets(state)

        network = state['network']
        link_flows = state['link_flows']

        # Estimate link costs via BPR: c_a = FFT_a * (1 + α(v/c)^β)
        fft = network.link_fft
        cap = np.maximum(network.link_capacity, 1e-10)
        vc_ratio = link_flows / cap
        link_costs = fft * (1.0 + self.bpr_alpha * np.power(vc_ratio, self.bpr_beta))

        # For each path, sum attacked-link costs: Σ_{a ∈ r ∩ A_att} c_a
        plm = network.path_link_matrix      # [n_paths, max_links], -1=pad
        plc = network.path_link_count        # [n_paths]
        n_paths = plm.shape[0]
        target_link_cost_sum = np.zeros(n_paths)
        for r in range(n_paths):
            for j in range(plc[r]):
                a = plm[r, j]
                if a >= 0 and a in self.targets:
                    target_link_cost_sum[r] += link_costs[a]

        # a_r = -γ · Σ c_a  (link-additive deflation)
        perturbation = -self.config.bias_fraction * target_link_cost_sum[:, np.newaxis]
        # Broadcast to [n_paths, n_windows]
        perturbation = np.broadcast_to(perturbation, true_costs.shape).copy()
        return perturbation


class PathBCDeflationAttack(AttackPolicy):
    """Demand-aware attack using path-based betweenness centrality.

    Targets links with highest path-based BC, which weights each edge
    by the number of enumerated OD-pair paths that traverse it.
    This represents a more informed attacker with knowledge of travel
    demand patterns.

    Paper: "path-based BC yields PoAtt = 1.057" (vs 1.046 for topo-BC).
    """
    def __init__(self, config, n_target_links=10):
        super().__init__(config)
        self.n_target_links = n_target_links
        self.targets = None
        self._path_mask_arr = None

    def _select_targets(self, state):
        delta_mat = state['incidence_matrix']  # sparse [n_links, n_paths*n_steps]
        network = state['network']

        # Path-based BC: count how many paths use each link
        # Using the path_link_matrix for exact path enumeration
        plm = network.path_link_matrix    # [n_paths, max_links]
        plc = network.path_link_count     # [n_paths]
        n_links = network.n_links
        path_count = np.zeros(n_links, dtype=np.float64)
        for r in range(network.n_paths):
            for j in range(plc[r]):
                a = plm[r, j]
                if a >= 0:
                    path_count[a] += 1.0

        sorted_idx = np.argsort(path_count)[::-1]
        self.targets = sorted_idx[:self.n_target_links]

        n_paths = delta_mat.shape[1]
        path_mask = np.zeros(n_paths)
        for link_idx in self.targets:
            paths_using = np.asarray(delta_mat[link_idx].toarray()).flatten() > 0
            path_mask[paths_using] = 1.0
        self._path_mask_arr = path_mask

        print(f"  PathBCDeflation: {len(self.targets)} high-pathBC links")
        print(f"  Path coverage: {self._path_mask_arr.sum():.0f}/{n_paths} ({100*self._path_mask_arr.sum()/n_paths:.1f}%)")
        print(f"  Target links (0-idx): {list(self.targets)}")

    def generate(self, day, true_costs, state=None):
        if not self.is_active(day):
            return np.zeros_like(true_costs)
        if state is None or 'incidence_matrix' not in state:
            return np.zeros_like(true_costs)
        if self.targets is None:
            self._select_targets(state)
        # DEFLATE: same as Scenario I but with path-BC targets
        perturbation = -self.config.bias_fraction * true_costs * self._path_mask_arr[:, np.newaxis]
        return perturbation
