import pytest
import numpy as np
import sys
import os
import jax.numpy as jnp
from scipy.linalg import expm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from latent_core.linalg.operators import X, Y, Z, I2

def test_51_H_hermitian():
    H = 0.5*X + 0.5*Y + 0.1*Z
    assert np.allclose(H, np.conj(H).T)

def test_52_unitary_evolution():
    H = 0.5*X + 0.5*Y
    U = expm(-1j * H * 0.1)
    assert np.allclose(U @ np.conj(U).T, np.eye(2))

def test_53_rho_stays_psd():
    H = 0.5*Z
    U = expm(-1j * H * 0.1)
    rho_0 = np.array([[1, 0], [0, 0]]) # Pure state |0><0|
    rho_1 = U @ rho_0 @ np.conj(U).T
    evals = np.linalg.eigvalsh(rho_1)
    assert np.all(evals > -1e-10)

def test_54_trace_preserved():
    H = X + Y + Z
    U = expm(-1j * H * 0.1)
    rho_0 = np.array([[0.5, 0.5], [0.5, 0.5]]) # Pure state |+><+|
    rho_1 = U @ rho_0 @ np.conj(U).T
    assert np.isclose(np.trace(rho_1), 1.0)

def test_55_no_evolution_when_H_zero():
    H = np.zeros((2, 2))
    U = expm(-1j * H * 0.1)
    rho_0 = np.array([[1, 0], [0, 0]])
    rho_1 = U @ rho_0 @ np.conj(U).T
    assert np.allclose(rho_0, rho_1)

def test_56_evolution_when_H_nonzero():
    H = X
    U = expm(-1j * H * 0.5) # pi/2 pulse if t=pi/2, here just some rotation
    rho_0 = np.array([[1, 0], [0, 0]])
    rho_1 = U @ rho_0 @ np.conj(U).T
    assert not np.allclose(rho_0, rho_1)

def test_58_multi_axis_coupling():
    H_z = Z
    H_xyz = X + Y + Z
    U_z = expm(-1j * H_z * 0.1)
    U_xyz = expm(-1j * H_xyz * 0.1)
    rho_0 = np.array([[1, 0], [0, 0]])
    rho_z = U_z @ rho_0 @ np.conj(U_z).T
    rho_xyz = U_xyz @ rho_0 @ np.conj(U_xyz).T
    assert not np.allclose(rho_z, rho_xyz)

def test_60_long_run_evolution_stable():
    H = 0.1*X + 0.2*Y - 0.1*Z
    rho = np.array([[1, 0], [0, 0]])
    U = expm(-1j * H * 0.01)
    
    for _ in range(1000):
        rho = U @ rho @ np.conj(U).T
        
    assert np.isclose(np.trace(rho), 1.0)
    assert np.all(np.linalg.eigvalsh(rho) > -1e-10)
