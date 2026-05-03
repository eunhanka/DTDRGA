"""Resilience metrics for DTD-Trust-Attack analysis.

Quantifies system performance degradation and recovery after
information attacks, using TSTT and trust trajectories.
"""

from __future__ import annotations

import numpy as np


class ResilienceAnalyzer:
    """Compute resilience metrics from simulation trajectories.

    Parameters
    ----------
    tstt_trajectory : np.ndarray, shape [n_days]
        Total system travel time per day.
    trust_trajectory : np.ndarray, shape [n_days, n_classes]
        Expected trust per day per driver class.
    attack_start : int
        First day of the attack (0-indexed).
    attack_end : int
        First day after the attack (0-indexed, exclusive).
    baseline_tstt : float, optional
        Legacy baseline TSTT (mean of all pre-attack days including
        convergence).  Used by price_of_attack_legacy().
    baseline_converged : float, optional
        Converged baseline TSTT = mean(TSTT[30:50]) from a separate
        no-attack baseline run.  Used by sustained_poatt / peak_poatt.
        Falls back to baseline_tstt if not provided.
    """

    def __init__(
        self,
        tstt_trajectory: np.ndarray,
        trust_trajectory: np.ndarray,
        attack_start: int,
        attack_end: int,
        baseline_tstt: float = None,
        baseline_converged: float = None,
    ):
        self.tstt = np.asarray(tstt_trajectory, dtype=float)
        self.trust = np.asarray(trust_trajectory, dtype=float)
        self.attack_start = attack_start
        self.attack_end = attack_end
        self.n_days = len(self.tstt)

        if baseline_tstt is not None:
            self.baseline = float(baseline_tstt)
        else:
            pre = self.tstt[:attack_start]
            self.baseline = float(pre.mean()) if len(pre) > 0 else float(self.tstt[0])

        if baseline_converged is not None:
            self.baseline_converged = float(baseline_converged)
        else:
            self.baseline_converged = self.baseline

        # Pre-attack trust (mean across classes at last pre-attack day)
        if attack_start > 0:
            self._pre_attack_trust = self.trust[attack_start - 1].copy()
        else:
            self._pre_attack_trust = self.trust[0].copy()

    # ------------------------------------------------------------------
    # New corrected PoAtt metrics
    # ------------------------------------------------------------------

    def sustained_poatt(self) -> float:
        """Mean TSTT during attack period / converged baseline TSTT."""
        attack_tstt = self.tstt[self.attack_start:self.attack_end]
        if len(attack_tstt) == 0:
            return 1.0
        return float(attack_tstt.mean() / self.baseline_converged)

    def peak_poatt(self) -> float:
        """Max TSTT during attack period / converged baseline TSTT."""
        attack_tstt = self.tstt[self.attack_start:self.attack_end]
        if len(attack_tstt) == 0:
            return 1.0
        return float(attack_tstt.max() / self.baseline_converged)

    def transient_overshoot_ratio(self) -> float:
        """peak_poatt / sustained_poatt.

        Measures how much the transient spike exceeds the steady
        attacked state.  Large ratio = system near stability boundary.
        """
        sus = self.sustained_poatt()
        if sus <= 0:
            return float('inf')
        return self.peak_poatt() / sus

    def price_of_attack(self) -> float:
        """Default PoAtt = sustained_poatt (corrected metric)."""
        return self.sustained_poatt()

    def price_of_attack_legacy(self) -> float:
        """Legacy: max TSTT during attack / baseline (includes convergence)."""
        attack_tstt = self.tstt[self.attack_start:self.attack_end]
        if len(attack_tstt) == 0:
            return 1.0
        return float(attack_tstt.max() / self.baseline)

    # ------------------------------------------------------------------
    # Other metrics (unchanged)
    # ------------------------------------------------------------------

    def time_to_recover(self, threshold: float = 0.05) -> int:
        """Days after attack_end until TSTT within threshold of baseline."""
        target = self.baseline_converged * (1.0 + threshold)
        post = self.tstt[self.attack_end:]
        for i, val in enumerate(post):
            if val <= target:
                return i
        return len(post)

    def resilience_triangle_area(self) -> float:
        """Integral of (TSTT(d) - baseline) from attack_start to end of sim.

        Normalized by converged baseline for comparability.
        """
        region = self.tstt[self.attack_start:]
        excess = np.maximum(region - self.baseline_converged, 0.0)
        return float(excess.sum() / self.baseline_converged)

    def robustness(self) -> float:
        """1 - (max_TSTT - baseline) / baseline."""
        attack_tstt = self.tstt[self.attack_start:self.attack_end]
        if len(attack_tstt) == 0:
            return 1.0
        max_increase = attack_tstt.max() - self.baseline_converged
        return float(1.0 - max_increase / self.baseline_converged)

    def trust_recovery_delay(self, threshold: float = 0.10) -> int:
        """Days after attack_end for mean trust to reach
        (pre_attack_trust - threshold)."""
        target = self._pre_attack_trust.mean() - threshold
        post_trust = self.trust[self.attack_end:]
        for i in range(len(post_trust)):
            if post_trust[i].mean() >= target:
                return i
        return len(post_trust)

    def hysteresis_gap(self) -> float:
        """(TSTT at end of simulation - baseline) / baseline."""
        return float((self.tstt[-1] - self.baseline_converged) / self.baseline_converged)

    def recovery_amplification_ratio(self) -> float:
        """time_to_recover / attack_duration."""
        attack_duration = self.attack_end - self.attack_start
        if attack_duration == 0:
            return 0.0
        return float(self.time_to_recover() / attack_duration)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """All metrics in one dict."""
        return {
            "sustained_poatt": self.sustained_poatt(),
            "peak_poatt": self.peak_poatt(),
            "transient_overshoot_ratio": self.transient_overshoot_ratio(),
            "price_of_attack": self.price_of_attack(),
            "time_to_recover": self.time_to_recover(),
            "resilience_triangle": self.resilience_triangle_area(),
            "robustness": self.robustness(),
            "trust_recovery_delay": self.trust_recovery_delay(),
            "hysteresis_gap": self.hysteresis_gap(),
            "recovery_amplification_ratio": self.recovery_amplification_ratio(),
        }
