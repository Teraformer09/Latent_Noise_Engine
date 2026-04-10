"""
latent_core/linalg/operators.py
================================
Pauli basis operators and tensor-product algebra.

Mathematical spine
------------------
Pauli basis:  𝒫 = {I, X, Y, Z}
n-qubit basis: 𝒫_n = 𝒫^⊗n

All matrices are returned as complex128 JAX arrays.
"""

from __future__ import annotations
import jax
# Force x64 globally before any jax.numpy usage
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import Array
from functools import reduce

# ---------------------------------------------------------------------------
# Fundamental Pauli matrices (complex128)
# ---------------------------------------------------------------------------

I2: Array = jnp.eye(2, dtype=jnp.complex128)
X: Array = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
Y: Array = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
Z: Array = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)

PAULIS: dict[str, Array] = {"I": I2, "X": X, "Y": Y, "Z": Z}
PAULI_LIST: list[Array] = [I2, X, Y, Z]
PAULI_LABELS: list[str] = ["I", "X", "Y", "Z"]


# ---------------------------------------------------------------------------
# Tensor products
# ---------------------------------------------------------------------------

def kron(a: Array, b: Array) -> Array:
    """Kronecker product of two matrices."""
    return jnp.kron(a, b)


def tensor_power(op: Array, n: int) -> Array:
    """Compute op^⊗n via repeated Kronecker products."""
    if n == 1:
        return op
    return reduce(jnp.kron, [op] * n)


def pauli_tensor(labels: list[str]) -> Array:
    """
    Build multi-qubit Pauli operator from a list of single-qubit labels.

    Example
    -------
    >>> pauli_tensor(["X", "Z"])  # X ⊗ Z on 2 qubits
    """
    ops = [PAULIS[lbl] for lbl in labels]
    return reduce(jnp.kron, ops)


def n_qubit_paulis(n: int) -> list[Array]:
    """
    Return all 4^n n-qubit Pauli operators as a flat list.
    Order: lexicographic over {I,X,Y,Z}^n.
    """
    from itertools import product as iproduct

    paulis = []
    for combo in iproduct(PAULI_LABELS, repeat=n):
        paulis.append(pauli_tensor(list(combo)))
    return paulis


# ---------------------------------------------------------------------------
# Operator properties
# ---------------------------------------------------------------------------

def commutator(a: Array, b: Array) -> Array:
    """[A, B] = AB - BA"""
    return a @ b - b @ a


def anticommutator(a: Array, b: Array) -> Array:
    """{A, B} = AB + BA"""
    return a @ b + b @ a


def dagger(op: Array) -> Array:
    """Conjugate transpose."""
    return jnp.conj(op).T


def is_hermitian(op: Array, tol: float = 1e-10) -> bool:
    """Check if op == op†."""
    return bool(jnp.allclose(op, dagger(op), atol=tol))


def is_unitary(op: Array, tol: float = 1e-10) -> bool:
    """Check if op†op == I."""
    n = op.shape[0]
    return bool(jnp.allclose(dagger(op) @ op, jnp.eye(n, dtype=op.dtype), atol=tol))


def normalize_spectrum(H: Array) -> Array:
    """
    Scale Hermitian H so that spec(H) ⊆ [-1, 1].

    QSP requires this.  Uses operator norm (largest eigenvalue magnitude).
    """
    evals = jnp.linalg.eigvalsh(H)
    norm = jnp.max(jnp.abs(evals))
    return H / (norm + 1e-12)


# ---------------------------------------------------------------------------
# Signal (rotation) operators used by QSP
# ---------------------------------------------------------------------------

def Rx(theta: float | Array) -> Array:
    """Rotation about X: e^{-i θ/2 X}"""
    c = jnp.cos(theta / 2)
    s = jnp.sin(theta / 2)
    return jnp.array([[c, -1j * s], [-1j * s, c]], dtype=jnp.complex128)


def Rz(phi: float | Array) -> Array:
    """Rotation about Z: e^{-i φ/2 Z}"""
    return jnp.array(
        [[jnp.exp(-1j * phi / 2), 0], [0, jnp.exp(1j * phi / 2)]],
        dtype=jnp.complex128,
    )


def signal_op(x: float | Array) -> Array:
    """
    QSP signal operator W(x) = e^{i arccos(x) X}
    for x ∈ [-1, 1].
    """
    theta = jnp.arccos(jnp.clip(x, -1.0, 1.0))
    return Rx(2 * theta)
