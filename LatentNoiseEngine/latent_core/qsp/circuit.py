"""
latent_core/qsp/circuit.py
============================
Circuit-level representation and execution of QSP sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
import jax.numpy as jnp
from jax import Array

from latent_core.linalg.operators import Rz, signal_op
from latent_core.qsp.phase_solver import qsp_unitary


@dataclass
class QSPCircuit:
    """Holds a solved QSP circuit: phases + polynomial metadata."""

    raw_phases: Array     # (d+1,) unconstrained phases
    degree: int
    loss: float           # solver residual
    is_valid: bool        # passed unitary check

    @property
    def phases(self) -> Array:
        """Constrained phases φ_k ∈ (-π, π)."""
        return jnp.pi * jnp.tanh(self.raw_phases)

    def apply(self, x: float | Array) -> Array:
        """Evaluate U_Φ(x) at signal value x."""
        return qsp_unitary(self.raw_phases, x)

    def apply_batch(self, xs: Array) -> Array:
        """Evaluate U_Φ(x) for a batch of x values. Returns (N, 2, 2) unitaries."""
        import jax
        return jax.vmap(lambda xi: qsp_unitary(self.raw_phases, xi))(xs)

    def __repr__(self) -> str:
        return (
            f"QSPCircuit(degree={self.degree}, "
            f"loss={self.loss:.2e}, valid={self.is_valid})"
        )
