import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from engine.pipeline import LatentNoisePipeline

def run_benchmark():
    print("="*60)
    print("   LATENT NOISE ENGINE: RESEARCH BENCHMARK (d=3, 5, 7)")
    print("="*60)
    
    distances = [3, 5, 7]
    results = {}
    
    cfg_base = {
        "steps": 25,
        "seed": 42,
        "base_alpha": 1.0,
        "kp": 10.0,
        "ki": 2.0,
        "target_hazard": 0.01, 
        "sigma_spatial": 1.0,
        "sigma_temporal": 0.015, # Slightly more noise to avoid floor
        "gamma_corr": 0.05,
        "shots": 5000 # More shots for resolution
    }
    
    for d in distances:
        print(f"Running distance d={d}...")
        pipe = LatentNoisePipeline({**cfg_base, "d": d})
        out = pipe.run()
        results[d] = np.mean(out["hazards"])
        
    print("\n" + "-"*40)
    print(f"{'Distance':<10} | {'Mean Hazard':<12}")
    print("-"*40)
    for d, h in results.items():
        print(f"{d:<10} | {h:<12.6f}")
    
    # Validation
    if results[3] > results[5] and results[5] > results[7]:
        print("\n✅ DISTANCE SCALING: PASS (d=3 > d=5 > d=7)")
    else:
        print("\n❌ DISTANCE SCALING: FAIL")

    # QSP Influence Check
    print("\n🧪 QSP INFLUENCE CHECK")
    pipe_off = LatentNoisePipeline({**cfg_base, "d": 3, "use_qsp": False})
    pipe_on = LatentNoisePipeline({**cfg_base, "d": 3, "use_qsp": True, "qsp_coeffs": [0.1, 0.5, 0.1]})
    
    h_off = np.mean(pipe_off.run()["hazards"])
    h_on = np.mean(pipe_on.run()["hazards"])
    diff = abs(h_on - h_off)
    
    print(f"Hazard (QSP OFF): {h_off:.6f}")
    print(f"Hazard (QSP ON):  {h_on:.6f}")
    print(f"Difference:      {diff:.6f}")
    
    if diff > 0.01:
        print("✅ QSP INFLUENCE: PASS (Significant spectral shift)")
    else:
        print("❌ QSP INFLUENCE: FAIL")

if __name__ == "__main__":
    run_benchmark()
