"""Trust dynamics module: Beta-distribution trust with heterogeneous driver classes.

Models how driver trust in route guidance information evolves over time,
based on the accuracy of past guidance.  Trust is represented as a
Beta(alpha, beta) distribution per driver class, updated daily via
Bayesian-inspired increments with asymmetric success/failure weights.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class TrustParams:
    """Parameters governing trust dynamics for heterogeneous driver classes.

    Default driver classes:
        0 - CAV-connected: high information reliance (10%)
        1 - App-reliant: moderate information reliance (50%)
        2 - Experience-based: low information reliance (40%)
    """

    n_classes: int = 3
    class_names: Optional[List[str]] = None
    class_fractions: Optional[np.ndarray] = None

    # Per-class trust update weights
    w_s: Optional[np.ndarray] = None   # success weight (accurate guidance)
    w_f: Optional[np.ndarray] = None   # failure weight (inaccurate guidance)

    # Per-class information weight parameters
    lambda_max: Optional[np.ndarray] = None   # max information reliance
    theta: Optional[np.ndarray] = None        # sensitivity to trust level

    # Per-class indifference band (tolerance for guidance error)
    indifference_band: Optional[np.ndarray] = None

    # Global parameters
    lambda_forget: float = 0.95   # forgetting factor (shrinks alpha, beta toward prior)
    epsilon: float = 360.0        # threshold for accurate vs. inaccurate guidance
                                  # = 0.1 h = 6 min in seconds (paper §5.1.3)

    # Smooth trust mode: xi = exp(-eta * |error|) instead of threshold
    smooth_mode: bool = False
    eta: float = 10.0             # smooth sensitivity (higher → sharper transition)

    # Initial Beta distribution parameters
    alpha_init: Optional[np.ndarray] = None
    beta_init: Optional[np.ndarray] = None

    def __post_init__(self):
        n = self.n_classes

        if self.class_names is None:
            self.class_names = ["CAV-connected", "App-reliant", "Experience-based"][:n]

        if self.class_fractions is None:
            self.class_fractions = np.array([0.10, 0.60, 0.30][:n])
        else:
            self.class_fractions = np.asarray(self.class_fractions, dtype=float)

        if self.w_s is None:
            self.w_s = np.array([0.16, 0.10, 0.06][:n])
        else:
            self.w_s = np.asarray(self.w_s, dtype=float)

        if self.w_f is None:
            self.w_f = np.array([0.8, 0.5, 0.3][:n])
        else:
            self.w_f = np.asarray(self.w_f, dtype=float)

        if self.lambda_max is None:
            self.lambda_max = np.array([0.9, 0.7, 0.3][:n])
        else:
            self.lambda_max = np.asarray(self.lambda_max, dtype=float)

        if self.theta is None:
            # Per-class theta reserved for future use; not used in route choice.
            # Route choice uses a single scalar theta (default 0.004) from DTD params.
            self.theta = np.array([0.004, 0.004, 0.004][:n])
        else:
            self.theta = np.asarray(self.theta, dtype=float)

        if self.indifference_band is None:
            self.indifference_band = np.array([0.0, 200.0, 400.0][:n])
        else:
            self.indifference_band = np.asarray(self.indifference_band, dtype=float)

        if self.alpha_init is None:
            self.alpha_init = np.full(n, 5.0)
        else:
            self.alpha_init = np.asarray(self.alpha_init, dtype=float)

        if self.beta_init is None:
            self.beta_init = np.full(n, 1.0)
        else:
            self.beta_init = np.asarray(self.beta_init, dtype=float)

        # Validate class fractions sum to 1
        frac_sum = self.class_fractions.sum()
        if not np.isclose(frac_sum, 1.0):
            raise ValueError(
                f"class_fractions must sum to 1.0, got {frac_sum:.6f}"
            )


@dataclass
class TrustState:
    """Current trust state: Beta distribution parameters per driver class."""

    alpha: np.ndarray   # shape [n_classes]
    beta: np.ndarray    # shape [n_classes]

    @property
    def expected_trust(self) -> np.ndarray:
        """E[T] = alpha / (alpha + beta)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> np.ndarray:
        """Var[T] = alpha * beta / ((alpha+beta)^2 * (alpha+beta+1))."""
        ab = self.alpha + self.beta
        return (self.alpha * self.beta) / (ab ** 2 * (ab + 1))

    @property
    def confidence(self) -> np.ndarray:
        """Confidence = alpha + beta (higher → more concentrated distribution)."""
        return self.alpha + self.beta


class TrustDynamics:
    """Evolves trust states day-to-day based on guidance accuracy.

    Usage::

        params = TrustParams()
        dynamics = TrustDynamics(params)
        state = dynamics.initial_state()

        for day in range(n_days):
            guidance_error = compute_error(...)   # shape [n_classes]
            state = dynamics.update(state, guidance_error)
            weights = dynamics.information_weight(state)
    """

    def __init__(self, params: TrustParams):
        self.params = params

    def initial_state(self) -> TrustState:
        """Create the initial trust state from parameters."""
        return TrustState(
            alpha=self.params.alpha_init.copy(),
            beta=self.params.beta_init.copy(),
        )

    def update(self, state: TrustState, guidance_error: np.ndarray) -> TrustState:
        """Update trust based on guidance error for each driver class.

        Parameters
        ----------
        state : TrustState
            Current trust state.
        guidance_error : np.ndarray, shape [n_classes]
            Absolute guidance error per class (e.g. |info_signal - experienced_cost|).

        Returns
        -------
        TrustState
            New trust state after one day's update.
        """
        guidance_error = np.asarray(guidance_error, dtype=float)
        p = self.params

        # Step 1: Forgetting — shrink toward uniform prior
        new_alpha = p.lambda_forget * state.alpha
        new_beta = p.lambda_forget * state.beta

        # Step 2: Bayesian-inspired update based on accuracy
        if p.smooth_mode:
            # Smooth model: xi = exp(-eta * |error|), continuous in [0, 1]
            xi = np.exp(-p.eta * np.abs(guidance_error))
            new_alpha = new_alpha + xi * p.w_s
            new_beta = new_beta + (1.0 - xi) * p.w_f
        else:
            # Threshold model: binary accurate/inaccurate
            accurate = guidance_error <= p.epsilon   # boolean [n_classes]
            new_alpha = np.where(accurate, new_alpha + p.w_s, new_alpha)
            new_beta = np.where(accurate, new_beta, new_beta + p.w_f)

        return TrustState(alpha=new_alpha, beta=new_beta)

    def information_weight(self, state: TrustState) -> np.ndarray:
        """Compute information reliance weight lambda(T) per class.

        lambda_k = T_k * lambda_max_k

        Parameters
        ----------
        state : TrustState
            Current trust state.

        Returns
        -------
        np.ndarray, shape [n_classes]
            Information weight in [0, lambda_max] per class.
        """
        return state.expected_trust * self.params.lambda_max
