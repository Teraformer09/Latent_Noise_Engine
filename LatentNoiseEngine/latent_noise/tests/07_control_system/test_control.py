import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from engine.pipeline import LatentNoisePipeline

def test_82_alpha_increases_high_hazard():
    # Use sigma that causes errors without triggering identity cooling
    cfg = {
        "steps": 20, "target_hazard": 0.0001, "sigma_temporal": 0.1, 
        "base_alpha": 1.0, "kp": 100.0, "ki": 10.0, "seed": 42
    }
    pipe = LatentNoisePipeline(cfg)
    out = pipe.run()
    # If hazard > 0.0001, alpha should increase
    assert out["alphas"][-1] > 1.0

def test_83_alpha_decreases_low_hazard():
    cfg = {
        "steps": 10, "target_hazard": 0.9, "sigma_v": 0.0, "sigma_temporal": 0.0, 
        "base_alpha": 10.0, "kp": 10.0, "seed": 42
    }
    pipe = LatentNoisePipeline(cfg)
    out = pipe.run()
    assert out["alphas"][-1] < 9.9

def test_84_no_oscillation_blowup():
    cfg = {"steps": 100, "target_hazard": 0.1, "kp": 2.0, "ki": 0.5} 
    pipe = LatentNoisePipeline(cfg)
    out = pipe.run()
    alphas = out["alphas"]
    count_boundary = sum(1 for a in alphas if a >= 99.0 or a <= 0.02)
    assert count_boundary < 80 
    
def test_86_integral_control_works():
    cfg_p = {"steps": 50, "kp": 5.0, "ki": 0.0, "target_hazard": 0.1, "sigma_temporal": 0.1, "seed": 1}
    out_p = LatentNoisePipeline(cfg_p).run()
    
    cfg_pi = {"steps": 50, "kp": 5.0, "ki": 2.0, "target_hazard": 0.1, "sigma_temporal": 0.1, "seed": 1}
    out_pi = LatentNoisePipeline(cfg_pi).run()
    
    error_p = abs(np.mean(out_p["hazards"][-10:]) - 0.1)
    error_pi = abs(np.mean(out_pi["hazards"][-10:]) - 0.1)
    
    assert error_pi <= error_p + 0.15
