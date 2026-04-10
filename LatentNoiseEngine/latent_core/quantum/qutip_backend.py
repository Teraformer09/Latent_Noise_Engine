"""
latent_core/quantum/qutip_backend.py
======================================
Optional QuTiP backend for validation and cross-checking JAX results.

This module is NOT used in the main JAX pipeline.
It serves as:
  1. Ground-truth validator for the JAX quantum channel implementation
  2. Lindblad master equation solver (continuous-time cross-check)
  3. Visualisation utilities (Bloch sphere, density matrix plots)

Requires: qutip >= 5.0
"""

from __future__ import annotations

import warnings
import numpy as np
from typing import Callable

try:
    import qutip as qt
    HAS_QUTIP = True
except ImportError:
    HAS_QUTIP = False
    warnings.warn("QuTiP not installed. qutip_backend unavailable.", ImportWarning)


def _require_qutip():
    if not HAS_QUTIP:
        raise ImportError(
            "QuTiP is required for this function. "
            "Install via: pip install qutip"
        )


# ---------------------------------------------------------------------------
# Conversion utilities
# ---------------------------------------------------------------------------

def jax_to_qobj(arr: np.ndarray, dims: list | None = None) -> "qt.Qobj":
    """Convert numpy/JAX array to QuTiP Qobj."""
    _require_qutip()
    M = np.array(arr, dtype=complex)
    if dims is None:
        n = M.shape[0]
        dims = [[n], [n]]
    return qt.Qobj(M, dims=dims)


def qobj_to_numpy(q: "qt.Qobj") -> np.ndarray:
    """Convert QuTiP Qobj to numpy array."""
    return np.array(q.full(), dtype=complex)


# ---------------------------------------------------------------------------
# Pauli channel via QuTiP
# ---------------------------------------------------------------------------

def pauli_channel_qutip(
    rho_np: np.ndarray,
    probs: np.ndarray,
) -> np.ndarray:
    """
    Apply Pauli channel using QuTiP for ground-truth validation.

    ℰ(ρ) = Σ_P p_P P ρ P

    Parameters
    ----------
    rho_np : (2,2) numpy density matrix
    probs  : (4,) probabilities [I, X, Y, Z]

    Returns
    -------
    rho_next : (2,2) numpy array
    """
    _require_qutip()
    rho_q = qt.Qobj(rho_np)
    paulis = [qt.identity(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    result = sum(p * (P * rho_q * P.dag()) for p, P in zip(probs, paulis))
    return qobj_to_numpy(result)


# ---------------------------------------------------------------------------
# Lindblad master equation (continuous-time validation)
# ---------------------------------------------------------------------------

def lindblad_evolve(
    rho_np: np.ndarray,
    H_np: np.ndarray,
    collapse_ops: list[tuple[np.ndarray, float]],
    tlist: np.ndarray,
) -> list[np.ndarray]:
    """
    Solve Lindblad master equation using QuTiP mesolve.

    dρ/dt = -i[H, ρ] + Σ_k γ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})

    Parameters
    ----------
    rho_np       : (d,d) initial density matrix
    H_np         : (d,d) Hamiltonian
    collapse_ops : list of (operator_np, rate) tuples
    tlist        : time points to evaluate

    Returns
    -------
    states : list of (d,d) numpy density matrices at each time point
    """
    _require_qutip()
    rho0 = qt.Qobj(rho_np)
    H = qt.Qobj(H_np)
    c_ops = [np.sqrt(rate) * qt.Qobj(op) for op, rate in collapse_ops]

    result = qt.mesolve(H, rho0, tlist, c_ops=c_ops)
    return [qobj_to_numpy(s) for s in result.states]


# ---------------------------------------------------------------------------
# Bloch sphere visualisation
# ---------------------------------------------------------------------------

def plot_bloch_trajectory(bloch_vectors: np.ndarray, title: str = "Bloch sphere trajectory"):
    """
    Plot a trajectory of Bloch vectors on the Bloch sphere.

    Parameters
    ----------
    bloch_vectors : (T, 3) array of [rx, ry, rz] values
    """
    _require_qutip()
    import matplotlib.pyplot as plt

    b = qt.Bloch()
    b.add_points(bloch_vectors.T)
    b.render()
    plt.title(title)
    plt.tight_layout()
    return b


# ---------------------------------------------------------------------------
# Cross-validation utility
# ---------------------------------------------------------------------------

def validate_against_qutip(
    rho_jax: np.ndarray,
    rho_qutip: np.ndarray,
    tol: float = 1e-6,
) -> dict:
    """
    Compare JAX and QuTiP density matrices.

    Returns
    -------
    dict with: max_abs_diff, trace_distance, is_consistent
    """
    diff = rho_jax - rho_qutip
    max_diff = float(np.max(np.abs(diff)))
    evals = np.linalg.eigvalsh(diff)
    td = float(0.5 * np.sum(np.abs(evals)))
    return {
        "max_abs_diff": max_diff,
        "trace_distance": td,
        "is_consistent": max_diff < tol,
    }
