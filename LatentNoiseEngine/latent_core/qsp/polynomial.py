"""
latent_core/qsp/polynomial.py
================================
Polynomial construction from spectral features for QSP input.

Math
----
P(x) = Σ_{k=0}^d a_k T_k(x)     (Chebyshev basis on [-1,1])

QSP requirement:  |P(x)| ≤ 1  for all x ∈ [-1,1]

Two construction modes:
  1. lowpass  — damped exponential coefficients driven by PSD slope β
  2. fitted   — least-squares fit to a target response function
"""

from __future__ import annotations
import jax
# Force x64 globally before any jax.numpy usage
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import Array
import numpy as np


# ---------------------------------------------------------------------------
# Chebyshev evaluation
# ---------------------------------------------------------------------------

def chebyshev_basis(x: Array, degree: int) -> Array:
    """
    Evaluate Chebyshev polynomials T_0(x), ..., T_d(x) at x.

    Uses recurrence:  T_0=1, T_1=x, T_{k+1}=2x T_k - T_{k-1}

    Returns
    -------
    T : (degree+1, len(x)) array
    """
    x = jnp.atleast_1d(x)
    T = [jnp.ones_like(x), x]
    for k in range(2, degree + 1):
        T.append(2 * x * T[-1] - T[-2])
    return jnp.stack(T[:degree + 1], axis=0)  # (d+1, N)


def eval_polynomial(coeffs: Array, x: Array) -> Array:
    """
    Evaluate P(x) = Σ_k a_k T_k(x) at points x ∈ [-1, 1].

    Parameters
    ----------
    coeffs : (d+1,) Chebyshev coefficients
    x      : (N,) evaluation points

    Returns
    -------
    Px : (N,) values
    """
    T = chebyshev_basis(x, degree=len(coeffs) - 1)
    return jnp.dot(coeffs, T)


# ---------------------------------------------------------------------------
# Domain mapping:  frequency → Chebyshev domain [-1, 1]
# ---------------------------------------------------------------------------

def freq_to_cheb_domain(f: Array, f_min: float, f_max: float) -> Array:
    """
    Map f ∈ [f_min, f_max] → x ∈ [-1, 1].

        x = 2(f - f_min)/(f_max - f_min) - 1
    """
    return 2.0 * (f - f_min) / (f_max - f_min + 1e-12) - 1.0


# ---------------------------------------------------------------------------
# Mode 1 — Low-pass polynomial from PSD slope β
# ---------------------------------------------------------------------------

def lowpass_coeffs(beta: float, degree: int, alpha_scale: float = 1.0) -> Array:
    """
    Build damped Chebyshev coefficients for a low-pass filter.

        a_k = e^{-α k}   where  α = f(β)

    Larger β (steeper 1/f spectrum) → faster coefficient decay.
    Coefficients are then normalised so ‖P‖_∞ ≤ 1.
    """
    alpha = alpha_scale * max(beta, 0.1)
    ks = jnp.arange(degree + 1, dtype=jnp.float64)
    raw = jnp.exp(-alpha * ks)

    # Normalise so max |P(x)| ≤ 1 on a grid
    x_grid = jnp.linspace(-1.0, 1.0, 500)
    P_vals = eval_polynomial(raw, x_grid)
    max_val = jnp.max(jnp.abs(P_vals)) + 1e-12
    return raw / max_val


# ---------------------------------------------------------------------------
# Mode 2 — Least-squares fitted polynomial
# ---------------------------------------------------------------------------

def fit_polynomial(
    target_fn,
    degree: int,
    n_points: int = 500,
    reg: float = 1e-3,
) -> Array:
    """
    Fit Chebyshev coefficients to a target function on [-1, 1].

        min_a ‖P(x) - f(x)‖² + reg ‖a‖²

    with clipping to enforce |P(x)| ≤ 1.

    Parameters
    ----------
    target_fn : callable, maps (N,) array → (N,) array
    degree    : polynomial degree
    n_points  : number of Gauss-Chebyshev quadrature points
    reg       : L2 regularisation on coefficients

    Returns
    -------
    coeffs : (degree+1,) Chebyshev coefficients, normalised
    """
    # Chebyshev nodes (Gauss-Chebyshev quadrature points)
    k = np.arange(1, n_points + 1)
    x_nodes = np.cos((2 * k - 1) * np.pi / (2 * n_points))

    T = np.array(chebyshev_basis(jnp.array(x_nodes), degree))  # (d+1, N)
    y = np.array(target_fn(x_nodes))

    # Regularised least squares
    A = T.T  # (N, d+1)
    coeffs_np, _, _, _ = np.linalg.lstsq(
        A.T @ A + reg * np.eye(degree + 1), A.T @ y, rcond=None
    )

    coeffs = jnp.array(coeffs_np)

    # Normalise
    x_grid = jnp.linspace(-1.0, 1.0, 1000)
    P_vals = eval_polynomial(coeffs, x_grid)
    max_val = jnp.max(jnp.abs(P_vals)) + 1e-12
    return coeffs / max_val


# ---------------------------------------------------------------------------
# Polynomial construction dispatcher
# ---------------------------------------------------------------------------

def build_polynomial(
    beta: float,
    degree: int,
    method: str = "lowpass",
    reg: float = 1e-3,
) -> Array:
    """
    Build polynomial coefficients from spectral exponent β.

    Parameters
    ----------
    beta   : PSD slope exponent (β = 2H - 1)
    degree : Chebyshev degree d
    method : 'lowpass' | 'fitted'

    Returns
    -------
    coeffs : (d+1,) Chebyshev coefficients, |P(x)| ≤ 1 enforced
    """
    if method == "lowpass":
        coeffs = lowpass_coeffs(beta, degree)
    elif method == "fitted":
        # Target: rational low-pass  P(x) ≈ 1/(1 + γ x²),  γ driven by β
        gamma = max(beta, 0.1)
        target = lambda x: 1.0 / (1.0 + gamma * x**2)
        coeffs = fit_polynomial(target, degree, reg=reg)
    else:
        raise ValueError(f"Unknown polynomial method: {method}")

    # Ensure polynomial values are bounded within [-1,1] on a grid
    x_grid = jnp.linspace(-1.0, 1.0, 500)
    vals = eval_polynomial(coeffs, x_grid)
    coeffs = coeffs / (jnp.max(jnp.abs(vals)) + 1e-12)
    return coeffs


def enforce_odd_polynomial(coeffs: Array) -> Array:
    """
    Zero out even Chebyshev coefficients to enforce odd parity.
    """
    coeffs = coeffs.copy()
    # Chebyshev index k=0 is even (constant) — clear even indices
    for k in range(len(coeffs)):
        if k % 2 == 0:
            coeffs = coeffs.at[k].set(0.0)
    return coeffs
