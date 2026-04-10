import pytest
import numpy as np
import sys
import os
import jax.numpy as jnp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from latent_core.qsp.transform import qsp_effective_hamiltonian
from latent_core.linalg.operators import X, Y, Z, I2

def test_31_heff_hermitian():
    H = 0.5*X + 0.3*Y - 0.2*Z
    coeffs = jnp.array([1.0, 0.5, 0.2])
    H_eff = qsp_effective_hamiltonian(H, coeffs)
    assert jnp.allclose(H_eff, jnp.conj(H_eff).T)

def test_32_eigenvalues_transformed():
    H = 0.5 * Z
    coeffs = jnp.array([0.0, 1.0]) # P(x) = x -> identity mapping
    H_eff = qsp_effective_hamiltonian(H, coeffs)
    evals_H = jnp.linalg.eigvalsh(H)
    evals_Heff = jnp.linalg.eigvalsh(H_eff)
    # H_tilde = H / ||H|| = Z.  P(Z) = Z.
    assert np.allclose(evals_Heff, jnp.linalg.eigvalsh(Z))

def test_33_spectrum_bounded():
    H = 10.0 * X
    # P(x) = T_2(x) = 2x^2 - 1.  Max value in [-1,1] is 1, min is -1.
    coeffs = jnp.array([-1.0, 0.0, 2.0]) 
    H_eff = qsp_effective_hamiltonian(H, coeffs)
    evals = jnp.linalg.eigvalsh(H_eff)
    assert jnp.max(jnp.abs(evals)) <= 1.0 + 1e-6

def test_36_degree_difference():
    H = 0.5 * X + 0.5 * Z
    coeffs1 = jnp.array([0.0, 1.0])
    coeffs3 = jnp.array([0.0, -0.5, 0.0, 0.8])
    H_eff1 = qsp_effective_hamiltonian(H, coeffs1)
    H_eff3 = qsp_effective_hamiltonian(H, coeffs3)
    assert not jnp.allclose(H_eff1, H_eff3)

def test_39_identity_polynomial():
    H = 0.5 * Y
    coeffs = jnp.array([0.0, 1.0]) # P(x) = x
    H_eff = qsp_effective_hamiltonian(H, coeffs)
    # H_tilde = Y
    assert jnp.allclose(H_eff, Y)

def test_40_no_spectral_explosion():
    H = 100.0 * Z
    coeffs = jnp.array([1.0, 1.0, 1.0, 1.0])
    H_eff = qsp_effective_hamiltonian(H, coeffs)
    assert not jnp.any(jnp.isnan(H_eff))
    assert jnp.max(jnp.abs(H_eff)) < 10.0 # Bounded by sum of coeffs

def test_41_stable_across_lambda():
    for l in [-10.0, -1.0, 0.0, 1.0, 10.0]:
        H = l * X
        H_eff = qsp_effective_hamiltonian(H, jnp.array([1.0, 0.5]))
        assert not jnp.any(jnp.isnan(H_eff))
