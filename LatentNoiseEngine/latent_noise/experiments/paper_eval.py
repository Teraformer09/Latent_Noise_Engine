import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from engine.pipeline import LatentNoisePipeline

def run_paper_evaluation(n_runs=50): # Reduced for local speed
    print(f"Executing Survival Analysis (TTT, HZ, CTRL) over {n_runs} runs...")
    
    policies = {
        "Static": {"beta_ctrl": 0.0, "base_alpha": 1.0}, # No feedback
        "Adaptive": {"beta_ctrl": 20.0, "base_alpha": 1.0} # Stronger feedback
    }
    
    all_results = []
    
    for name, params in policies.items():
        print(f"Testing Policy: {name}...")
        ttt_list = []
        hz_list = []
        
        for i in range(n_runs):
            cfg = {
                **params,
                "seed": i,
                "steps": 200, 
                "target_hazard": 0.1,
                "sigma_v": 0.02, # Faster drift to trigger failure
                "sigma_zeta": 0.05,
                "d": 3,
                "c_threshold": 0.5 # Lower threshold to trigger failure faster
            }
            
            pipe = LatentNoisePipeline(cfg)
            out = pipe.run()
            
            t_fail = out["t_fail"]
            ttt_list.append(t_fail)
            
            hz = out["accumulated_hazard"] / t_fail
            hz_list.append(hz)
            
        all_results.append({
            "Policy": name,
            "TTT (Mean)": np.mean(ttt_list),
            "HZ (Mean)": np.mean(hz_list)
        })
        
    df = pd.DataFrame(all_results)
    print("\n" + "="*60)
    print("   PAPER COMPARISON: SURVIVAL PERFORMANCE (d=3)")
    print("="*60)
    print(df.to_string(index=False))
    
if __name__ == "__main__":
    run_paper_evaluation(n_runs=50)
