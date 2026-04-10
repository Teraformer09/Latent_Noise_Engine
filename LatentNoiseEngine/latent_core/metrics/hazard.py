"""
latent_core/metrics/hazard.py
================================
Survival / hazard formulation for qubit failure.

Math
----
Instantaneous risk:   ρ(θ_t, x_t, a_t) = F_L(t) - F_L(t+1)
Cumulative hazard:    H_t = Σ_{τ=0}^{t-1} ρ_τ = 1 - F_L(t)
Failure time:         T_fail = inf { t : H_t ≥ H_crit }

DP loss:
    ℒ = 𝔼[Σ_t γ^t H_t]       discounted cumulative hazard
"""

from __future__ import annotations

from dataclasses import dataclass, field
import jax.numpy as jnp
import numpy as np


@dataclass
class HazardTracker:
    """
    Tracks cumulative hazard H_t = 1 - F_L(t) over a trajectory.
    """

    h_crit: float = 0.3
    gamma: float = 0.99

    _fidelities: list[float] = field(default_factory=list)
    _hazards: list[float] = field(default_factory=list)
    _failure_time: int | None = None

    def update(self, fidelity: float) -> None:
        """Push new fidelity value and update hazard accumulation."""
        self._fidelities.append(fidelity)
        H_t = 1.0 - fidelity
        self._hazards.append(H_t)

        if self._failure_time is None and H_t >= self.h_crit:
            self._failure_time = len(self._hazards) - 1

    @property
    def current_hazard(self) -> float:
        return self._hazards[-1] if self._hazards else 0.0

    @property
    def failure_time(self) -> int | None:
        return self._failure_time

    @property
    def survived(self) -> bool:
        return self._failure_time is None

    @property
    def discounted_loss(self) -> float:
        """ℒ = Σ_t γ^t H_t"""
        total = 0.0
        for t, h in enumerate(self._hazards):
            total += (self.gamma**t) * h
        return total

    def to_arrays(self) -> dict:
        return {
            "fidelities": np.array(self._fidelities),
            "hazards": np.array(self._hazards),
            "failure_time": self._failure_time,
        }


def compute_hazard_trajectory(fidelities: np.ndarray) -> np.ndarray:
    """H_t = 1 - F_L(t) for array of fidelities."""
    return 1.0 - fidelities


def discounted_hazard_loss(
    hazards: np.ndarray, gamma: float = 0.99
) -> float:
    """ℒ = Σ_t γ^t H_t — DP-style discounted cost."""
    t = np.arange(len(hazards))
    return float(np.sum((gamma**t) * hazards))


def failure_time(hazards: np.ndarray, h_crit: float = 0.3) -> int | None:
    """T_fail = inf{t : H_t ≥ H_crit}. Returns None if no failure."""
    indices = np.where(hazards >= h_crit)[0]
    return int(indices[0]) if len(indices) > 0 else None
