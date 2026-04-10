"""
latent_core/mapping/pauli_channel.py
======================================
Pauli channel construction and application.

Math
----
ℰ_t(ρ) = Σ_{P ∈ {I,X,Y,Z}} p_P P ρ P

Kraus operators:  K_P = √p_P  P

CPTP condition:   Σ_P K_P† K_P = Σ_P p_P I = I  ✓  (since Σ p_P = 1)
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from latent_core.linalg.operators import PAULI_LIST, PAULI_LABELS
from latent_core.linalg.superoperator import validate_cptp


# ---------------------------------------------------------------------------
# Channel application
# ---------------------------------------------------------------------------

def apply_pauli_channel(rho: Array, probs: Array) -> Array:
    """
    Apply Pauli channel:  ρ' = Σ_P p_P P ρ P

    Parameters
    ----------
    rho   : (2, 2) density matrix
    probs : (4,) probability vector for [I, X, Y, Z]

    Returns
    -------
    rho_next : (2, 2)
    """
    rho_next = jnp.zeros_like(rho)
    for p, P in zip(probs, PAULI_LIST):
        rho_next = rho_next + p * (P @ rho @ jnp.conj(P).T)
    return rho_next


def kraus_operators(probs: Array) -> list[Array]:
    """
    Build Kraus operators K_P = √p_P P.
    """
    return [jnp.sqrt(p) * P for p, P in zip(probs, PAULI_LIST)]


def check_channel(probs: Array, tol: float = 1e-8) -> dict:
    """
    Validate the Pauli channel defined by probs.

    Returns dict: cptp_valid, cptp_error, prob_sum
    """
    kraus = kraus_operators(probs)
    err = validate_cptp(kraus, tol)

    # Also check via simple sum (should be 1.0)
    return {
        "cptp_valid": err < tol,
        "cptp_error": err,
        "prob_sum": float(jnp.sum(probs)),
    }


# ---------------------------------------------------------------------------
# Combined channel + unitary evolution
# ---------------------------------------------------------------------------

def evolve_step(
    rho: Array,
    probs: Array,
    U: Array | None = None,
    ordering: str = "channel_first",
) -> Array:
    """
    One evolution step:

        channel_first:  ρ' = ℰ(U ρ U†)
        unitary_first:  ρ' = U ℰ(ρ) U†

    Parameters
    ----------
    rho      : (d, d) density matrix
    probs    : (4,) Pauli probabilities
    U        : (d, d) unitary (None → identity)
    ordering : application order

    Returns
    -------
    rho_next : (d, d) density matrix
    """
    if U is None:
        U = jnp.eye(rho.shape[0], dtype=jnp.complex128)

    if ordering == "channel_first":
        rho_u = U @ rho @ jnp.conj(U).T
        return apply_pauli_channel(rho_u, probs)
    else:  # unitary_first
        rho_e = apply_pauli_channel(rho, probs)
        return U @ rho_e @ jnp.conj(U).T
