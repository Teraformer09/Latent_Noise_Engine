"""
latent_core/jax_engine/random.py
==================================
Functional random key management for JAX.

Rules
-----
- NEVER reuse a key
- ALWAYS split before consuming
- Log seed for reproducibility
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def make_key(seed: int) -> Array:
    """Create a fresh PRNGKey from integer seed."""
    return jax.random.PRNGKey(seed)


def split_key(key: Array) -> tuple[Array, Array]:
    """Split key → (key_next, subkey). Use subkey for operations."""
    return jax.random.split(key)


def split_keys(key: Array, n: int) -> Array:
    """Split key into n independent subkeys. Shape (n, 2)."""
    return jax.random.split(key, n)


def make_trajectory_keys(key: Array, T: int) -> Array:
    """Generate T independent keys for a scan loop."""
    return jax.random.split(key, T)


def make_batch_keys(key: Array, batch_size: int, T: int) -> Array:
    """
    Generate (batch_size, T) key array for batched trajectory simulation.
    Each trajectory gets its own independent key sequence.
    """
    batch_keys = jax.random.split(key, batch_size)
    traj_keys = jax.vmap(lambda k: jax.random.split(k, T))(batch_keys)
    return traj_keys  # shape (batch_size, T, 2)
