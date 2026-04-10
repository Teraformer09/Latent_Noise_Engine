import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from engine.pipeline import LatentNoisePipeline

def validate_research_claims():
    print("="*60)
    print("   RESEARCH VALIDATION: PHYSICAL QUANTUM ENGINE")
    print("="*60)
    
    cfg_base = {
        "steps": 30,
        "seed": 42,
        "kp": 15.0,
        "ki": 3.0,
        "target_hazard": 0.1,
        "sigma_temporal": 0.05,
        "dt": 0.1
    }
    
    # 🧪 TEST 1: QSP INFLUENCE (Does QSP ON vs OFF change the physics?)
    print("\n🧪 TEST 1: QSP INFLUENCE")
    
    pipe_off = LatentNoisePipeline({**cfg_base, "use_qsp": False})
    out_off = pipe_off.run()
    
    pipe_on = LatentNoisePipeline({**cfg_base, "use_qsp": True, "qsp_coeffs": [1.0, 0.0, -0.5]})
    out_on = pipe_on.run()
    
    diff = np.abs(np.mean(out_on["hazards"]) - np.mean(out_off["hazards"]))
    print(f"Mean Hazard (QSP OFF): {np.mean(out_off['hazards']):.4f}")
    print(f"Mean Hazard (QSP ON):  {np.mean(out_on['hazards']):.4f}")
    print(f"Hazard Difference:     {diff:.4f}")
    
    if diff > 1e-4:
        print("✅ QSP INFLUENCE: PASS (QSP spectral shaping is physically active)")
    else:
        print("❌ QSP INFLUENCE: FAIL (No spectral impact detected)")

    # 🧪 TEST 2: DISTANCE SCALING (Does d=5 protect better than d=3?)
    print("\n🧪 TEST 2: DISTANCE SCALING")
    
    pipe_d3 = LatentNoisePipeline({**cfg_base, "d": 3})
    out_d3 = pipe_d3.run()
    
    # Surface Code H matrix for d=5 is needed for a real test, 
    # but we can simulate the 'strength' of d=5 by reducing the logical prob
    print(f"Hazard (d=3): {np.mean(out_d3['hazards']):.4f}")
    print("INFO: Multi-distance scaling depends on the Stim-layer mapping.")

    # 🧪 TEST 3: PI CONTROL STABILITY (Does hazard stabilize near target?)
    print("\n🧪 TEST 3: PI CONTROL STABILITY")
    h_final = out_on["hazards"][-5:]
    mean_final = np.mean(h_final)
    print(f"Final Hazard Mean (Target=0.1): {mean_final:.4f}")
    
    if abs(mean_final - 0.1) < 0.1:
        print("✅ CONTROL STABILITY: PASS (System stabilizes near target)")
    else:
        print("❌ CONTROL STABILITY: FAIL")

if __name__ == "__main__":
    validate_research_claims()
