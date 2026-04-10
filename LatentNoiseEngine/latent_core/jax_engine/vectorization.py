"""
latent_core/jax_engine/vectorization.py
=========================================
vmap wrappers for batched trajectory simulation.

Math
----
Batched evolution:
    {ρ_{t+1}^(i)}_{i=1}^N = vmap(step_fn)(keys_i)

Complexity: O(T × d²) parallelised over N trajectories.
"""

from __future__ import annotations

import jax
from jax import Array
from typing import Callable


def batch_trajectories(
    traj_fn: Callable,
    batch_keys: Array,
    init_state,
) -> tuple:
    """
    Run N independent trajectories in parallel via vmap.

    Parameters
    ----------
    traj_fn    : function(init_state, keys_T) -> (final_state, outputs)
    batch_keys : (N, T, 2) array of per-trajectory per-step keys
    init_state : pytree of initial state (shared across batch)

    Returns
    -------
    (final_states, outputs) — each with leading batch dimension N
    """
    batched = jax.vmap(traj_fn, in_axes=(None, 0))
    return batched(init_state, batch_keys)


def scan_trajectory(
    step_fn: Callable,
    init_carry,
    keys: Array,
) -> tuple:
    """
    Run a single trajectory of T steps via jax.lax.scan.

    Parameters
    ----------
    step_fn  : function(carry, key) -> (carry_next, output)
    init_carry : initial carry state
    keys     : (T, 2) per-step keys

    Returns
    -------
    (final_carry, stacked_outputs)
    """
    return jax.lax.scan(step_fn, init_carry, keys)
