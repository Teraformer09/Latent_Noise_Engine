"""
latent_core/physics/hamiltonian.py
=====================================
System-environment Hamiltonian construction.

Math
----
H_SE(t) = Σ_{α ∈ {X,Z}} λ_α(t) σ_α ⊗ B_α

For single-qubit simulations we model B_α ≈ scalar → effective system H:
    H_eff(t) = Σ_α λ_α(t) σ_α
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from latent_core.linalg.operators import X, Y, Z, I2, normalize_spectrum


# ---------------------------------------------------------------------------
# Effective single-qubit Hamiltonian
# ---------------------------------------------------------------------------

def build_hamiltonian(
    lambda_X: float | Array,
    lambda_Z: float | Array,
    lambda_Y: float | Array = 0.0,
    normalise: bool = True,
) -> Array:
    """
    Build effective Hamiltonian:
        H = λ_X σ_X + λ_Y σ_Y + λ_Z σ_Z

    Parameters
    ----------
    lambda_X, lambda_Z, lambda_Y : coupling strengths
    normalise : scale to spec ⊆ [-1,1] for QSP compatibility

    Returns
    -------
    H : (2, 2) Hermitian matrix
    """
    H = (
        jnp.real(lambda_X) * X
        + jnp.real(lambda_Y) * Y
        + jnp.real(lambda_Z) * Z
    )
    if normalise:
        H = normalize_spectrum(H)
    return H


def hamiltonian_from_latent(theta: Array, normalise: bool = True) -> Array:
    """
    Build Hamiltonian from latent state vector θ = [λ_X², λ_Z², c_t].

    Uses sqrt to recover coupling magnitudes: λ_α = √(θ_α).
    """
    lam_X = jnp.sqrt(jnp.abs(theta[0]) + 1e-12)
    lam_Z = jnp.sqrt(jnp.abs(theta[1]) + 1e-12)
    return build_hamiltonian(lam_X, lam_Z, normalise=normalise)
