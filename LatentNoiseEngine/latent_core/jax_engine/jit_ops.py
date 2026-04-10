"""
latent_core/jax_engine/jit_ops.py
====================================
JIT-compiled core operations.

All functions decorated with @jax.jit must:
  - accept only JAX arrays + static Python scalars
  - be pure (no side effects)
  - avoid Python control flow on traced values
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from latent_core.linalg.operators import PAULI_LIST
from latent_core.mapping.softmax import softmax


@jax.jit
def jit_softmax(logits: Array, temperature: float = 1.0) -> Array:
    """JIT-compiled softmax."""
    return softmax(logits, temperature)


@jax.jit
def jit_apply_pauli_channel(rho: Array, probs: Array) -> Array:
    """JIT-compiled Pauli channel application."""
    paulis = jnp.stack([jnp.array(P) for P in PAULI_LIST])  # (4, 2, 2)

    def _term(P, p):
        return p * (P @ rho @ jnp.conj(P).T)

    terms = jax.vmap(_term)(paulis, probs)
    return jnp.sum(terms, axis=0)


@jax.jit
def jit_fidelity(rho: Array, psi: Array) -> Array:
    """JIT-compiled pure state fidelity."""
    return jnp.real(jnp.conj(psi) @ rho @ psi)


@jax.jit
def jit_trace_distance(rho1: Array, rho2: Array) -> Array:
    """JIT-compiled trace distance."""
    diff = rho1 - rho2
    evals = jnp.linalg.eigvalsh(diff)
    return 0.5 * jnp.sum(jnp.abs(evals))
