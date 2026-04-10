"""
latent_core/metrics/blp.py
============================
Breuer-Laine-Piilo (BLP) measure of non-Markovianity.

Math
----
Trace distance:   D(ρ₁, ρ₂) = ½ ‖ρ₁ - ρ₂‖₁

BLP theorem:
    dD/dt > 0  →  information backflow  →  non-Markovian

Measure:   𝒩 = Σ_{dD/dt > 0} (dD/dt)    (integral over all backflow episodes)
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax import Array


def trace_distance(rho1: Array, rho2: Array) -> float:
    """D(ρ₁, ρ₂) = ½ ‖ρ₁ - ρ₂‖₁"""
    diff = rho1 - rho2
    evals = jnp.linalg.eigvalsh(diff)
    return float(0.5 * jnp.sum(jnp.abs(evals)))


def blp_measure(
    rho1_trajectory: list[Array],
    rho2_trajectory: list[Array],
    dt: float = 1.0,
) -> dict:
    """
    Compute BLP non-Markovianity measure from two state trajectories.

    Parameters
    ----------
    rho1_trajectory : list of (d,d) density matrices (initial state 1)
    rho2_trajectory : list of (d,d) density matrices (initial state 2)
    dt              : time step

    Returns
    -------
    dict with:
        N           : BLP measure (non-Markovianity)
        D           : trace distance trajectory
        dDdt        : derivative of trace distance
        backflow    : boolean array (True when dD/dt > 0)
    """
    D = np.array([
        trace_distance(r1, r2)
        for r1, r2 in zip(rho1_trajectory, rho2_trajectory)
    ])

    dDdt = np.gradient(D, dt)
    backflow = dDdt > 0.0

    N = float(np.trapz(np.where(backflow, dDdt, 0.0)))

    return {
        "N": N,
        "D": D,
        "dDdt": dDdt,
        "backflow": backflow,
        "is_non_markovian": N > 0,
    }
