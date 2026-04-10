"""
latent_core/stochastic/arfima.py
==================================
ARFIMA(0, d, 0) long-memory model as alternative to fGn.

This is the discrete-time counterpart. d = H - 0.5 ∈ (-0.5, 0.5).
For d > 0: long memory (non-summable ACF).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


def arfima_weights(d: float, max_lag: int) -> np.ndarray:
    """
    Compute truncated AR(∞) coefficients for ARFIMA(0, d, 0).

    π_k = Γ(k - d) / (Γ(k+1) Γ(-d))   for k ≥ 1
    """
    from scipy.special import gamma

    ks = np.arange(1, max_lag + 1)
    weights = np.array(
        [gamma(k - d) / (gamma(k + 1) * gamma(-d)) for k in ks]
    )
    return weights


def generate_arfima(
    key: Array, T: int, d: float, sigma: float = 1.0, max_lag: int = 200
) -> Array:
    """
    Generate T samples of ARFIMA(0, d, 0) via truncated AR representation.

    Parameters
    ----------
    key     : JAX PRNGKey
    T       : number of samples
    d       : fractional differencing parameter (d = H - 0.5)
    sigma   : innovation std
    max_lag : AR truncation order

    Returns
    -------
    x : (T,) JAX array
    """
    weights = jnp.array(arfima_weights(d, max_lag))
    innovations = jax.random.normal(key, shape=(T + max_lag,)) * sigma

    # Convolve via scan for memory efficiency
    def scan_fn(history, eps):
        x_t = eps - jnp.dot(weights, history)
        history_next = jnp.roll(history, shift=1).at[0].set(x_t)
        return history_next, x_t

    init_history = jnp.zeros(max_lag)
    _, xs = jax.lax.scan(scan_fn, init_history, innovations)
    return xs[max_lag:]  # discard burn-in
