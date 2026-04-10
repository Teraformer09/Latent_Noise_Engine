"""
latent_core/stochastic/fgn.py
================================
Fractional Gaussian Noise (fGn) generation.

Math
----
Covariance:
    𝔼[ζ(t) ζ(t+τ)] = σ²/2 (|τ+1|^{2H} - 2|τ|^{2H} + |τ-1|^{2H})

For H > 0.5: Σ_τ 𝔼[ζ(t)ζ(t+τ)] = ∞  → long memory.

Power spectrum:
    S(f) ∝ 1/f^β,   β = 2H - 1

Algorithm: Davies–Harte (exact, O(N log N)).
"""

from __future__ import annotations
import jax
# Force x64 globally before any jax.numpy usage
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax import Array


# ---------------------------------------------------------------------------
# Autocovariance of fGn
# ---------------------------------------------------------------------------

def fgn_autocov(tau: np.ndarray, H: float, sigma: float = 1.0) -> np.ndarray:
    """
    Theoretical autocovariance γ(τ) for fGn(H, σ).
    τ is an integer array of lags.
    """
    h2 = 2 * H
    return (sigma**2 / 2) * (
        np.abs(tau + 1) ** h2 - 2 * np.abs(tau) ** h2 + np.abs(tau - 1) ** h2
    )


# ---------------------------------------------------------------------------
# Davies–Harte algorithm (exact, O(N log N))
# ---------------------------------------------------------------------------

def generate_fgn(key: Array, N: int, H: float, sigma: float = 1.0) -> Array:
    """
    Generate N samples of fGn with Hurst exponent H using Davies–Harte.

    Parameters
    ----------
    key   : JAX PRNGKey
    N     : number of samples
    H     : Hurst exponent ∈ (0, 1)
    sigma : standard deviation

    Returns
    -------
    fgn   : (N,) JAX array
    """
    # Build circulant row
    tau = np.arange(N)
    cov = fgn_autocov(tau, H, sigma=1.0)
    # Embed into circulant of length 2N
    row = np.concatenate([cov, cov[-2:0:-1]])
    M = len(row)

    # Eigenvalues of circulant = FFT of first row
    evals = np.real(np.fft.fft(row))

    # Ensure non-negative (numerical noise at zero)
    evals = np.maximum(evals, 0.0)

    # Draw standard normals via JAX
    key1, key2 = jax.random.split(key)
    w_real = jax.random.normal(key1, shape=(M,))
    w_imag = jax.random.normal(key2, shape=(M,))

    # Scale by sqrt of eigenvalues
    sqrt_evals = jnp.array(np.sqrt(evals / M))
    z = sqrt_evals * (w_real + 1j * w_imag)

    # IFFT → take first N real components
    fgn_full = jnp.real(jnp.fft.ifft(z) * M)
    fgn_samples = fgn_full[:N] * sigma

    return fgn_samples


# ---------------------------------------------------------------------------
# Batch generation (pre-generate all time steps)
# ---------------------------------------------------------------------------

def generate_fgn_batch(
    key: Array, batch_size: int, T: int, H: float, sigma: float = 1.0
) -> Array:
    """
    Generate (batch_size, T) matrix of fGn trajectories.
    Uses vmap over batch dimension.
    """
    keys = jax.random.split(key, batch_size)

    def _single(k):
        return generate_fgn(k, T, H, sigma)

    return jax.vmap(_single)(keys)
