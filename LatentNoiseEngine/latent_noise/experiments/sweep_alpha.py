import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from engine.pipeline import LatentNoisePipeline

# Sweep over the base_alpha of the controller to see how it affects overall stability
base_alphas = [0.1, 0.5, 1.0, 2.0]

print(f"{'Base Alpha':<12} | {'Mean Hazard':<12} | {'Std Hazard':<12} | {'Final Alpha':<12}")
print("-" * 55)

for ba in base_alphas:
    # Use 500 steps to see the feedback stabilize
    cfg = {
        "steps": 500, 
        "seed": 42, 
        "base_alpha": ba, 
        "beta": 10.0, 
        "target": 0.2
    }
    pipe = LatentNoisePipeline(cfg)
    out = pipe.run()
    
    mean_h = np.mean(out["hazards"])
    std_h = np.std(out["hazards"])
    final_alpha = out["alphas"][-1]
    
    print(f"{ba:<12.2f} | {mean_h:<12.4f} | {std_h:<12.4f} | {final_alpha:<12.4f}")