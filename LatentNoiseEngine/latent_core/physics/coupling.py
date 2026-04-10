"""
latent_core/physics/coupling.py
=================================
Latent state dynamics with control back-action.

Math
----
θ_{t+1} = A θ_t + B a_t + η_t,   ρ(A) < 1,   η_t ~ 𝒩(0, Σ_η)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def latent_step(
    key: Array,
    theta: Array,
    action: Array,
    A: Array,
    B: Array,
    noise_cov: Array,
) -> Array:
    """
    One step of the latent state transition:
        θ_{t+1} = A θ_t + B a_t + η_t

    Parameters
    ----------
    key       : JAX PRNGKey
    theta     : (m,) current latent state
    action    : (k,) control action
    A         : (m, m) state transition matrix, ρ(A) < 1
    B         : (m, k) control input matrix
    noise_cov : (m, m) process noise covariance Σ_η

    Returns
    -------
    theta_next : (m,) next latent state
    """
    eta = jax.random.multivariate_normal(key, mean=jnp.zeros(theta.shape[0]), cov=noise_cov)
    return A @ theta + B @ action + eta


def build_stable_A(dim: int, spectral_radius: float = 0.95, key=None) -> Array:
    """
    Build a random stable matrix A with ρ(A) ≈ spectral_radius.
    Ensures latent dynamics are mean-reverting.
    """
    import jax
    if key is None:
        key = jax.random.PRNGKey(42)

    # Random orthogonal-ish matrix scaled to desired radius
    M = jax.random.normal(key, shape=(dim, dim))
    U, _, Vt = jnp.linalg.svd(M)
    A = U @ jnp.diag(jnp.full(dim, spectral_radius)) @ Vt
    return A
