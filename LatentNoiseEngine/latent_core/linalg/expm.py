"""
latent_core/linalg/expm.py
============================
Matrix exponential for Hamiltonian evolution.

U(t) = e^{-i H t}

Uses scipy.linalg.expm via JAX's pure_callback for correctness,
with a fast analytic path for 2×2 Hermitian matrices (single qubit).
"""

from __future__ import annotations
import jax
# Force x64 globally before any jax.numpy usage
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import Array
import numpy as np


# ---------------------------------------------------------------------------
# Analytic 2×2 path (single qubit — very fast)
# ---------------------------------------------------------------------------

def expm_hermitian_2x2(H: Array, t: float = 1.0) -> Array:
    """
    Compute e^{-i H t} analytically for 2×2 Hermitian H.

    Uses:  e^{-iHt} = cos(|h|t)I - i sin(|h|t)/|h| * H
    where  H = h·σ (decompose into Bloch vector).
    """
    H = H.astype(jnp.complex128)
    # Bloch vector components
    hx = jnp.real(H[0, 1])
    hy = jnp.imag(H[1, 0])
    hz = jnp.real(H[0, 0])
    norm_h = jnp.sqrt(hx**2 + hy**2 + hz**2) + 1e-15

    angle = norm_h * t
    c = jnp.cos(angle)
    s = jnp.sin(angle) / norm_h

    I2 = jnp.eye(2, dtype=jnp.complex128)
    sigma_vec = jnp.array(
        [[hz, hx - 1j * hy], [hx + 1j * hy, -hz]], dtype=jnp.complex128
    )
    return c * I2 - 1j * s * sigma_vec


# ---------------------------------------------------------------------------
# General path via scipy (works for arbitrary dimension)
# ---------------------------------------------------------------------------

def _scipy_expm(M: np.ndarray) -> np.ndarray:
    from scipy.linalg import expm
    return expm(M)


def expm_hamiltonian(H: Array, t: float = 1.0) -> Array:
    """
    Compute U = e^{-i H t} for Hermitian H of arbitrary dimension.

    Routes:
      - 2×2 → analytic (fast, JIT-compatible)
      - else → scipy.linalg.expm via numpy bridge
    """
    if H.shape == (2, 2):
        return expm_hermitian_2x2(H, t)

    # General: numpy bridge (not JIT-compatible but correct)
    M = np.array(-1j * t * H)
    U_np = _scipy_expm(M)
    return jnp.array(U_np, dtype=jnp.complex128)


# ---------------------------------------------------------------------------
# Unitary evolution
# ---------------------------------------------------------------------------

def unitary_evolution(rho: Array, H: Array, dt: float) -> Array:
    """
    Apply unitary evolution:  ρ(t+dt) = U ρ U†
    where U = e^{-i H dt}.
    """
    U = expm_hamiltonian(H, t=dt)
    return U @ rho @ jnp.conj(U).T
