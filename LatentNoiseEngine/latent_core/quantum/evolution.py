"""
latent_core/quantum/evolution.py
==================================
Full quantum evolution step combining Hamiltonian + channel.

Math
----
ρ_{t+1} = ℰ_t( P(H_t) ρ_t P(H_t)† )

where:
    H_t = H(λ(t))          — instantaneous Hamiltonian
    P(H_t) = QSP transform  — effective operator
    ℰ_t(·) = Pauli channel  — noise channel
"""

from __future__ import annotations
import jax
# Force x64 globally before any jax.numpy usage
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import Array

from latent_core.linalg.expm import unitary_evolution
from latent_core.mapping.pauli_channel import apply_pauli_channel, evolve_step
from latent_core.qsp.transform import qsp_effective_hamiltonian


def evolution_step(
    rho: Array,
    H: Array,
    probs: Array,
    qsp_coeffs: Array | None = None,
    dt: float = 1.0,
    ordering: str = "channel_first",
    use_qsp: bool = True,
) -> Array:
    """
    Full single-step evolution:

        1. QSP transform H → H_eff = P(H)   (if use_qsp)
        2. Unitary evolution U = e^{-i H_eff dt}
        3. Channel application ℰ(U ρ U†)

    Parameters
    ----------
    rho        : (d, d) density matrix
    H          : (d, d) Hermitian Hamiltonian
    probs      : (4,) Pauli channel probabilities
    qsp_coeffs : (deg+1,) Chebyshev coefficients (None → skip QSP)
    dt         : time step
    ordering   : 'channel_first' or 'unitary_first'
    use_qsp    : whether to apply QSP transformation

    Returns
    -------
    rho_next : (d, d) evolved density matrix
    """
    # 1. QSP operator transformation
    if use_qsp and qsp_coeffs is not None:
        H_eff = qsp_effective_hamiltonian(H, qsp_coeffs)
        # scaling boost to amplify QSP action
        H_eff = 3.0 * H_eff
    else:
        H_eff = H

    # 2. Unitary
    U = _compute_unitary(H_eff, dt)

    # 3. Channel + unitary (respecting ordering)
    return evolve_step(rho, probs, U=U, ordering=ordering)


def _compute_unitary(H: Array, dt: float) -> Array:
    """e^{-i H dt} via analytic 2×2 path or scipy for larger dims."""
    from latent_core.linalg.expm import expm_hamiltonian
    return expm_hamiltonian(H, t=dt)
