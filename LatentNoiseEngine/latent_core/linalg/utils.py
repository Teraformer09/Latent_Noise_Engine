"""latent_core/linalg/utils.py — Shared linear algebra utilities."""

from __future__ import annotations
import jax.numpy as jnp
from jax import Array


def partial_trace(rho: Array, dims: tuple[int, int], keep: int) -> Array:
    """
    Partial trace over a bipartite system.

    Parameters
    ----------
    rho  : (d_A * d_B, d_A * d_B) density matrix
    dims : (d_A, d_B)
    keep : 0 → keep subsystem A; 1 → keep subsystem B
    """
    d_A, d_B = dims
    rho_r = rho.reshape(d_A, d_B, d_A, d_B)
    if keep == 0:
        return jnp.einsum("ibjb->ij", rho_r)
    else:
        return jnp.einsum("aibj->ij", rho_r)


def fidelity(rho: Array, sigma: Array) -> float:
    """
    Uhlmann fidelity F(ρ, σ) = (Tr√(√ρ σ √ρ))².
    Simplified to Tr(ρσ) when one state is pure.
    """
    return float(jnp.real(jnp.trace(rho @ sigma)))


def trace_distance(rho: Array, sigma: Array) -> float:
    """D(ρ, σ) = ½ ‖ρ - σ‖_1"""
    diff = rho - sigma
    evals = jnp.linalg.eigvalsh(diff)
    return float(0.5 * jnp.sum(jnp.abs(evals)))


def purity(rho: Array) -> float:
    """γ = Tr(ρ²) ∈ [1/d, 1]."""
    return float(jnp.real(jnp.trace(rho @ rho)))


def bloch_vector(rho: Array) -> Array:
    """Extract Bloch vector [r_x, r_y, r_z] from single-qubit ρ."""
    from latent_core.linalg.operators import X, Y, Z
    rx = float(jnp.real(jnp.trace(rho @ X)))
    ry = float(jnp.real(jnp.trace(rho @ Y)))
    rz = float(jnp.real(jnp.trace(rho @ Z)))
    return jnp.array([rx, ry, rz])


def assert_valid_density_matrix(rho: Array, tol: float = 1e-8) -> None:
    """Raise AssertionError if ρ is not a valid density matrix."""
    assert rho.ndim == 2 and rho.shape[0] == rho.shape[1], "ρ must be square"
    assert abs(float(jnp.real(jnp.trace(rho))) - 1.0) < tol, "Tr(ρ) ≠ 1"
    evals = jnp.linalg.eigvalsh(rho)
    assert float(jnp.min(evals)) >= -tol, "ρ not positive semidefinite"
