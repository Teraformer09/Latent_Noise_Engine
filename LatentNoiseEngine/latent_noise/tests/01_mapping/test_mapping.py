import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from latent_noise.noise.mapping import lambda_to_pauli_probs

def get_error_sum(p):
    return p['px'] + p['py'] + p['pz']

def test_01_sensitivity_small():
    p1 = lambda_to_pauli_probs([0.1, 0.1, 0.1])
    p2 = lambda_to_pauli_probs([0.1, 0.1, 0.101])
    assert abs(p1['pz'] - p2['pz']) > 1e-5

def test_02_sensitivity_large():
    p1 = lambda_to_pauli_probs([0.1, 0.1, 0.1])
    p2 = lambda_to_pauli_probs([0.1, 0.1, 0.11])
    assert abs(p1['pz'] - p2['pz']) > 1e-4

def test_03_random_perturbation():
    l1 = np.random.rand(3)
    l2 = l1 + np.random.normal(0, 0.01, 3)
    p1 = lambda_to_pauli_probs(l1)
    p2 = lambda_to_pauli_probs(l2)
    assert not np.allclose([p1['px'], p1['py'], p1['pz']], [p2['px'], p2['py'], p2['pz']])

def test_04_gradient_monotonicity():
    # In Gibbs mapping: p ~ exp(-alpha * E)
    # So larger lambda (energy) -> lower probability
    p1 = lambda_to_pauli_probs([0.1, 0.1, 0.1], scale=1.0)
    p2 = lambda_to_pauli_probs([0.1, 0.1, 0.2], scale=1.0)
    assert p2['pz'] < p1['pz']

def test_05_numerical_stability():
    p = lambda_to_pauli_probs([10.0, -10.0, 0.0])
    assert not np.isnan(p['px'])

def test_06_sum_p_less_than_1():
    p = lambda_to_pauli_probs(np.random.rand(3))
    assert get_error_sum(p) <= 1.0 + 1e-6

def test_07_no_negative_probs():
    p = lambda_to_pauli_probs(np.random.rand(3))
    assert all(v >= 0 for v in p.values())

def test_08_identity_dominance():
    p = lambda_to_pauli_probs([100.0, 100.0, 100.0]) # High energy -> Identity dominance
    assert get_error_sum(p) < 0.1

def test_09_collapse_large_alpha():
    p = lambda_to_pauli_probs([0.5, 0.5, 0.5], scale=100.0)
    assert get_error_sum(p) < 0.05

def test_10_zero_lambda():
    p = lambda_to_pauli_probs([0.0, 0.0, 0.0])
    assert np.isclose(p['px'], p['py'])
    assert np.isclose(p['py'], p['pz'])

def test_11_large_positive_lambda():
    p = lambda_to_pauli_probs([100, 100, 100])
    assert all(0 <= v <= 1 for v in p.values())

def test_12_large_negative_lambda():
    p = lambda_to_pauli_probs([-100, -100, -100])
    assert all(0 <= v <= 1 for v in p.values())

def test_13_asymmetric_lambda():
    p = lambda_to_pauli_probs([1.0, 0.1, 0.1])
    assert p['px'] < p['py'] # Higher energy -> lower probability

def test_14_control_sensitivity():
    p1 = lambda_to_pauli_probs([0.1, 0.2, 0.3], scale=1.0)
    p2 = lambda_to_pauli_probs([0.1, 0.2, 0.3], scale=2.0)
    assert p1 != p2

def test_15_monotonic_pI_alpha():
    p_I_vals = []
    for a in [1.0, 5.0, 10.0]:
        p = lambda_to_pauli_probs([0.5, 0.5, 0.5], scale=a)
        p_I_vals.append(p['pi'])
    assert p_I_vals[0] < p_I_vals[1] < p_I_vals[2]
