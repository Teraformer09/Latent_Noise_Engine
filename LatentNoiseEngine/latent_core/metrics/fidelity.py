"""
latent_core/metrics/fidelity.py
=================================
Fidelity metrics for quantum state tracking.

Math
----
Pure state fidelity:   F(ρ) = ⟨ψ|ρ|ψ⟩
Logical fidelity:      F_L = Tr(P_code ρ)
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def pure_fidelity(rho: Array, psi: Array) -> float:
    """
    F(ρ, ψ) = ⟨ψ|ρ|ψ⟩

    Parameters
    ----------
    rho : (d, d) density matrix
    psi : (d,) pure state vector

    Returns
    -------
    F ∈ [0, 1]
    """
    psi_c = jnp.conj(psi)
    return float(jnp.real(psi_c @ rho @ psi))


def mixed_fidelity(rho: Array, sigma: Array) -> float:
    """
    Uhlmann fidelity F(ρ, σ) via sqrt approach.
    Approximate as Tr(ρσ) when sigma is pure.
    """
    return float(jnp.real(jnp.trace(rho @ sigma)))


def logical_fidelity(rho: Array, code_projector: Array) -> float:
    """
    F_L = Tr(P_code ρ) — fidelity within the code space.

    Parameters
    ----------
    rho            : (d, d) density matrix
    code_projector : (d, d) projector onto logical code subspace

    Returns
    -------
    F_L ∈ [0, 1]
    """
    return float(jnp.real(jnp.trace(code_projector @ rho)))


def compute_fidelity(
    rho: Array,
    target: Array,
    mode: str = "pure",
    code_projector: Array | None = None,
) -> float:
    """
    Unified fidelity computation.

    Parameters
    ----------
    rho    : current density matrix
    target : (d,) state vector or (d,d) density matrix
    mode   : 'pure' | 'mixed' | 'logical'
    """
    if mode == "pure":
        return pure_fidelity(rho, target)
    elif mode == "mixed":
        return mixed_fidelity(rho, target)
    elif mode == "logical":
        assert code_projector is not None
        return logical_fidelity(rho, code_projector)
    else:
        raise ValueError(f"Unknown fidelity mode: {mode}")
