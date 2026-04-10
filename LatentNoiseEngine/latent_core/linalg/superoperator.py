"""
latent_core/linalg/superoperator.py
=====================================
Superoperator (Liouville) representation of quantum channels.

Math
----
Vectorise density matrix:  |ρ⟩⟩ = vec(ρ) ∈ ℂ^{d²}

Channel as superoperator:
    ℰ = Σ_i K_i ⊗ K_i*

Evolution:
    |ρ_{t+1}⟩⟩ = ℰ |ρ_t⟩⟩

Spectral radius of ℰ governs stability.
"""

from __future__ import annotations
import jax
# Force x64 globally before any jax.numpy usage
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import Array


# ---------------------------------------------------------------------------
# Vectorisation helpers
# ---------------------------------------------------------------------------

def vec(rho: Array) -> Array:
    """Flatten density matrix ρ (d×d) → |ρ⟩⟩ (d²,) column-major."""
    return rho.flatten(order="F")


def unvec(v: Array, d: int) -> Array:
    """Reshape |ρ⟩⟩ (d²,) → ρ (d×d) column-major."""
    return v.reshape(d, d, order="F")


# ---------------------------------------------------------------------------
# Superoperator construction
# ---------------------------------------------------------------------------

def kraus_to_superop(kraus_ops: list[Array]) -> Array:
    """
    Build the superoperator matrix ℰ ∈ ℂ^{d²×d²} from Kraus operators.

        ℰ = Σ_i K_i ⊗ K_i*
    """
    d = kraus_ops[0].shape[0]
    superop = jnp.zeros((d * d, d * d), dtype=jnp.complex128)
    for K in kraus_ops:
        superop = superop + jnp.kron(K, jnp.conj(K))
    return superop


def apply_superop(superop: Array, rho: Array) -> Array:
    """Apply superoperator to density matrix: ρ' = unvec(ℰ vec(ρ))."""
    d = rho.shape[0]
    rho_vec = vec(rho)
    rho_next_vec = superop @ rho_vec
    return unvec(rho_next_vec, d)


# ---------------------------------------------------------------------------
# CPTP validation
# ---------------------------------------------------------------------------

def cptp_error(kraus_ops: list[Array]) -> float:
    """
    Compute ‖Σ_i K_i† K_i - I‖_F — should be zero for valid CPTP.
    """
    d = kraus_ops[0].shape[0]
    total = jnp.zeros((d, d), dtype=jnp.complex128)
    for K in kraus_ops:
        total = total + jnp.conj(K).T @ K
    diff = total - jnp.eye(d, dtype=jnp.complex128)
    return float(jnp.linalg.norm(diff, ord="fro"))


def validate_cptp(kraus_ops: list[Array], tol: float = 1e-8) -> bool:
    """Return True if Kraus operators satisfy completeness relation."""
    return cptp_error(kraus_ops) < tol


# ---------------------------------------------------------------------------
# Eigenspectrum analysis
# ---------------------------------------------------------------------------

def superop_spectrum(superop: Array) -> Array:
    """Eigenvalues of the superoperator (governs decoherence rates)."""
    return jnp.linalg.eigvals(superop)


def spectral_radius(superop: Array) -> float:
    """ρ(ℰ) = max |λ_i|.  Must be ≤ 1 for physical channels."""
    return float(jnp.max(jnp.abs(superop_spectrum(superop))))
