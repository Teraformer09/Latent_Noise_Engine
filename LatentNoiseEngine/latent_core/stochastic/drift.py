"""
latent_core/stochastic/drift.py
=================================
Non-stationary slow drift process.

Math
----
λ̄_α(t+1) = λ̄_α(t) + ν_α(t),   ν_α(t) ~ 𝒩(0, σ_ν²)

Var[λ̄(t)] = t σ_ν²  → unit-root non-stationarity (random walk).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def drift_step(key: Array, drift: Array, sigma: float) -> Array:
    """
    One step of the random walk drift.

    Parameters
    ----------
    key   : JAX PRNGKey
    drift : current drift value(s), shape (n_axes,)
    sigma : innovation std σ_ν

    Returns
    -------
    drift_next : shape (n_axes,)
    """
    noise = jax.random.normal(key, shape=drift.shape) * sigma
    return drift + noise


def generate_drift_trajectory(
    key: Array, T: int, n_axes: int, sigma: float, init: float = 0.0
) -> Array:
    """
    Generate (T, n_axes) drift trajectory.
    """
    init_drift = jnp.full((n_axes,), init)

    def scan_fn(drift, k):
        drift_next = drift_step(k, drift, sigma)
        return drift_next, drift_next

    keys = jax.random.split(key, T)
    _, trajectory = jax.lax.scan(scan_fn, init_drift, keys)
    return trajectory  # shape (T, n_axes)
