import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from latent_noise.noise.ou_process import PhysicalNoiseModel, SpatialNoiseField

def test_16_mean_reversion():
    nm = PhysicalNoiseModel(phi=0.5, sigma_zeta=0.5, sigma_v=0.0)
    vals = []
    nm.zeta = np.array([5.0, 5.0, 5.0])
    for _ in range(50):
        vals.append(nm.step())
    # Should revert towards 0
    assert np.mean(np.abs(vals[-1])) < np.mean(np.abs(vals[0]))

def test_17_variance_bounded():
    nm = PhysicalNoiseModel(sigma_v=0.0, phi=0.9)
    vals = np.array([nm.step() for _ in range(1000)])
    assert np.var(vals) < 2.0

def test_18_no_divergence():
    nm = PhysicalNoiseModel(sigma_v=0.01)
    vals = np.array([nm.step() for _ in range(10000)])
    assert not np.any(np.isnan(vals))
    assert np.max(np.abs(vals)) < 1000 # Random walk diverges but shouldn't explode

def test_19_autocorrelation():
    nm = PhysicalNoiseModel(sigma_v=0.0, phi=0.95) # pure AR(1)
    vals = np.array([nm.step()[0] for _ in range(1000)])
    # lag-1 autocorrelation
    corr = np.corrcoef(vals[:-1], vals[1:])[0, 1]
    assert corr > 0.8

def test_20_ar_persistence():
    nm = PhysicalNoiseModel(sigma_v=0.0, phi=0.9)
    nm.zeta = np.array([10.0, 10.0, 10.0])
    val1 = nm.step()
    assert np.all(val1 > 5.0)

def test_22_neighbor_correlation():
    coords = [(0,0), (0,1), (0,2), (10,10)]
    sn = SpatialNoiseField(coords=coords)
    W = sn.W
    # Node 0 and 1 are neighbors, 0 and 3 are far
    assert W[0, 1] > W[0, 3]

def test_23_kernel_normalization():
    coords = [(0,0), (0,1), (1,0), (1,1)]
    sn = SpatialNoiseField(coords=coords)
    assert np.allclose(sn.W.sum(axis=1), 1.0)

def test_24_spatial_variance():
    coords = [(i,j) for i in range(5) for j in range(5)]
    sn = SpatialNoiseField(coords=coords)
    for _ in range(10):
        field = sn.step()
    assert np.var(field) > 0

def test_25_heavy_tail_distribution():
    sn = SpatialNoiseField(coords=[(0,0)], burst_prob=1.0) # Always burst
    vals = np.array([sn.step()[0,0] for _ in range(100)])
    # Cauchy has high variance/outliers
    assert np.max(np.abs(vals)) > 5.0

def test_26_outlier_frequency():
    sn = SpatialNoiseField(coords=[(0,0)], burst_prob=0.1)
    vals = np.array([sn.step()[0,0] for _ in range(1000)])
    outliers = np.sum(np.abs(vals) > 3.0)
    assert outliers > 0

def test_27_no_nans():
    sn = SpatialNoiseField(coords=[(0,0)])
    for _ in range(100):
        sn.step()
    assert not np.any(np.isnan(sn.field))

def test_29_seed_reproducibility():
    sn1 = SpatialNoiseField(coords=[(0,0)], seed=42)
    sn2 = SpatialNoiseField(coords=[(0,0)], seed=42)
    assert np.allclose(sn1.step(), sn2.step())

def test_30_multirun_consistency():
    coords = [(0,0)]
    results = []
    for _ in range(3):
        sn = SpatialNoiseField(coords=coords, seed=1)
        results.append(sn.step())
    assert np.allclose(results[0], results[1])
    assert np.allclose(results[1], results[2])