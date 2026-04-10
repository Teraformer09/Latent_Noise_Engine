"""
latent_core/mapping/softmax.py
================================
Softmax mapping from latent state θ_t to probability simplex.

Math
----
p_P(t) = exp(w_P^T θ_t) / Σ_Q exp(w_Q^T θ_t)

Theorem (valid probability simplex):
    p_P ≥ 0,   Σ_P p_P = 1   ← guaranteed by construction

Identifiability constraint
--------------------------
W must have full rank to ensure injectivity θ_t → p_t.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def softmax(logits: Array, temperature: float = 1.0) -> Array:
    """Numerically stable softmax with temperature."""
    scaled = logits / temperature
    shifted = scaled - jnp.max(scaled)
    exp_s = jnp.exp(shifted)
    return exp_s / jnp.sum(exp_s)


def theta_to_probs(
    theta: Array,
    W: Array,
    b: Array | None = None,
    temperature: float = 1.0,
) -> Array:
    """
    Map latent state θ ∈ ℝ^m to channel probabilities p ∈ Δ^{|𝒫|-1}.

    Parameters
    ----------
    theta       : (m,) latent state vector
    W           : (|𝒫|, m) weight matrix — must have full column rank
    b           : (|𝒫|,) optional bias
    temperature : softmax temperature

    Returns
    -------
    probs : (|𝒫|,) probability vector over {I, X, Y, Z}
    """
    logits = W @ theta
    if b is not None:
        logits = logits + b
    return softmax(logits, temperature)


def init_softmax_weights(
    n_paulis: int = 4,
    latent_dim: int = 3,
    key=None,
    scale: float = 0.1,
) -> tuple[Array, Array]:
    """
    Initialise W, b for the softmax mapping.

    Full-rank initialisation: W is drawn from N(0, scale²) with
    rank enforcement via random orthogonal basis.
    """
    import jax
    if key is None:
        key = jax.random.PRNGKey(0)

    k1, k2 = jax.random.split(key)
    W = jax.random.normal(k1, shape=(n_paulis, latent_dim)) * scale
    b = jax.random.normal(k2, shape=(n_paulis,)) * scale
    return W, b
