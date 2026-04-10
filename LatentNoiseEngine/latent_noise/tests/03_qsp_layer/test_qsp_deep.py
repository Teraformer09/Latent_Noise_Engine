import pytest
import numpy as np
import sys
import os
import jax.numpy as jnp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from latent_core.qsp.transform import qsp_effective_hamiltonian
from latent_core.linalg.operators import X, Y, Z, normalize_spectrum
from latent_noise.engine.pipeline import LatentNoisePipeline

def test_34_qsp_on_off_diff():
    cfg = {
        "steps": 5, "d": 3, "sigma_temporal": 0.1, "shots": 1000, "seed": 42,
        "base_alpha": 1.0, "kp": 10.0, "ki": 2.0, "target_hazard": 0.1
    }
    pipe_off = LatentNoisePipeline({**cfg, "use_qsp": False})
    pipe_on = LatentNoisePipeline({**cfg, "use_qsp": True, "qsp_coeffs": [0.1, 0.5, 0.1]})
    
    h_off = np.mean(pipe_off.run()["hazards"])
    h_on = np.mean(pipe_on.run()["hazards"])
    assert abs(h_on - h_off) > 0.01

def test_35_different_polynomials():
    H = 0.5 * X + 0.5 * Y
    c1 = jnp.array([1.0, 0.0])
    c2 = jnp.array([0.0, 1.0])
    H1 = qsp_effective_hamiltonian(H, c1)
    H2 = qsp_effective_hamiltonian(H, c2)
    assert not jnp.allclose(H1, H2)

def test_37_stability_across_coeffs():
    H = Z
    # Random large coeffs
    coeffs = jnp.array(np.random.normal(0, 10, 5))
    H_eff = qsp_effective_hamiltonian(H, coeffs)
    assert not jnp.any(jnp.isnan(H_eff))

def test_38_spectrum_normalization():
    H = 100.0 * X
    H_norm = normalize_spectrum(H)
    evals = jnp.linalg.eigvalsh(H_norm)
    assert jnp.max(jnp.abs(evals)) <= 1.0 + 1e-9

def test_42_eigenvalue_tracking():
    H = X + Y + Z
    coeffs = jnp.array([0.0, 1.0]) # P(x) = x
    H_eff = qsp_effective_hamiltonian(H, coeffs)
    ev_H = jnp.linalg.eigvalsh(normalize_spectrum(H))
    ev_Heff = jnp.linalg.eigvalsh(H_eff)
    assert jnp.allclose(ev_H, ev_Heff)
