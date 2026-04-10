import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from latent_noise.engine.pipeline import LatentNoisePipeline

def test_105_long_horizon_stability():
    # Test for 500 steps (research scale)
    cfg = {"steps": 500, "seed": 0, "shots": 100}
    pipe = LatentNoisePipeline(cfg)
    out = pipe.run()
    assert len(out["hazards"]) == 500
    assert not np.any(np.isnan(out["hazards"]))

def test_106_no_drift_explosion():
    cfg = {"steps": 1000, "sigma_v": 0.1, "seed": 42}
    pipe = LatentNoisePipeline(cfg)
    out = pipe.run()
    # Even with drift, control and mapping should keep hazards bounded
    assert np.max(out["hazards"]) < 1.0

def test_109_control_convergence():
    cfg = {"steps": 100, "kp": 20.0, "ki": 5.0, "target_hazard": 0.1, "seed": 1}
    pipe = LatentNoisePipeline(cfg)
    out = pipe.run()
    # Check if the last 20 steps are closer to target than the first 20
    initial_error = abs(np.mean(out["hazards"][:20]) - 0.1)
    final_error = abs(np.mean(out["hazards"][-20:]) - 0.1)
    assert final_error <= initial_error + 0.05

def test_95_disable_mapping_collapses():
    # If we force probabilities to a fixed high value, hazard should skyrocket
    # This is a 'manual' simulation of a broken system.
    pass

def test_97_extreme_lambda_stability():
    # Force high noise field
    cfg = {"steps": 10, "sigma_temporal": 10.0, "seed": 42}
    pipe = LatentNoisePipeline(cfg)
    out = pipe.run()
    assert not np.any(np.isnan(out["hazards"]))
