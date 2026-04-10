"""
latent_core/qsp/transform.py
================================
Apply QSP transformation to produce effective Hamiltonian P(H).

Math
----
H_eff = P(H̃)     where H̃ = H/‖H‖  (spectrum in [-1,1])

U_Φ implements P via:
    ⟨0| U_Φ(H̃) |0⟩ = P(H̃)    (in the signal oracle model)

For simulation we use the polynomial applied to H eigenvalues.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
import numpy as np

from latent_core.linalg.operators import normalize_spectrum
from latent_core.qsp.polynomial import eval_polynomial, chebyshev_basis


def apply_polynomial_to_operator(H: Array, coeffs: Array) -> Array:
    """
    Compute P(H̃) where P is defined by Chebyshev coefficients.

    Uses spectral decomposition:
        H̃ = V Λ V†
        P(H̃) = V diag(P(λ_i)) V†

    Parameters
    ----------
    H      : (d, d) Hermitian operator
    coeffs : (deg+1,) Chebyshev coefficients

    Returns
    -------
    P_H : (d, d) Hermitian operator
    """
    H_tilde = normalize_spectrum(H)
    evals, evecs = jnp.linalg.eigh(H_tilde)

    # Evaluate polynomial at eigenvalues
    P_evals = eval_polynomial(coeffs, evals)

    # Reconstruct operator: P(H̃) = V diag(P(λ)) V†
    P_H = evecs @ jnp.diag(P_evals.astype(jnp.complex128)) @ jnp.conj(evecs).T
    return P_H


def qsp_effective_hamiltonian(
    H: Array,
    coeffs: Array,
) -> Array:
    """
    Compute effective Hamiltonian H_eff = P(H̃) for use in evolution.

    Returns
    -------
    H_eff : (d, d) Hermitian operator (eigenvalues ∈ range of P)
    """
    return apply_polynomial_to_operator(H, coeffs)
