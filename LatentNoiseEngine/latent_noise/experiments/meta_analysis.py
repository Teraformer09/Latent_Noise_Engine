import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

# Ensure local imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from engine.pipeline import LatentNoisePipeline

def run_comprehensive_analysis(n_runs=500, steps=100):
    print(f"Starting Meta-Analysis: {n_runs} Independent Monte Carlo Runs...")
    
    # Storage for meta-parameters
    results = {
        "final_hazard": [],
        "mean_hazard": [],
        "final_alpha": [],
        "hazard_variance": [],
        "settling_time": [] # Time to stay within 10% of target
    }
    
    # Configuration
    target_hazard = 0.2
    cfg = {
        "steps": steps,
        "base_alpha": 1.0,
        "beta": 15.0, # Increased sensitivity for analysis
        "target": target_hazard,
        "sigma": 0.3,  # Higher noise floor to test suppression
        "theta": 0.5
    }

    # Run Simulation
    for i in range(n_runs):
        if (i+1) % 50 == 0:
            print(f"Completed {i+1}/{n_runs} runs...")
            
        pipe = LatentNoisePipeline({**cfg, "seed": i})
        out = pipe.run()
        
        h_series = np.array(out["hazards"])
        a_series = np.array(out["alphas"])
        
        results["final_hazard"].append(h_series[-1])
        results["mean_hazard"].append(np.mean(h_series))
        results["final_alpha"].append(a_series[-1])
        results["hazard_variance"].append(np.var(h_series))
        
        # Calculate approximate settling time
        within_tolerance = np.abs(h_series - target_hazard) < 0.05
        settling_idx = np.where(within_tolerance)[0]
        results["settling_time"].append(settling_idx[0] if len(settling_idx) > 0 else steps)

    # Statistical Aggregation
    df = pd.DataFrame(results)
    
    print("\n" + "="*50)
    print("      LATENT NOISE ENGINE: 500-RUN META-ANALYSIS")
    print("="*50)
    
    stats = df.describe().loc[['mean', 'std', 'min', 'max']]
    print(stats)
    
    # Check for heavy-tail distribution (Kurtosis)
    kurt = df['mean_hazard'].kurtosis()
    print(f"\nDistribution Kurtosis (Mean Hazard): {kurt:.4f}")
    if kurt > 3:
        print("ALERT: Heavy right tail detected in survival/hazard distribution.")
    else:
        print("INFO: Distribution is approximately Mesokurtic/Normal.")

    # Domain 3: Scaling Significance (Sample d=3 result)
    print("\nDomain 3: Scaling Significance (d=3)")
    success_rate = np.mean(df['final_hazard'] < target_hazard)
    print(f"Control Success Rate (Final Hazard < {target_hazard}): {success_rate*100:.2f}%")

    # Save summary
    df.to_csv("latent_noise_meta_analysis.csv", index=False)
    print("\nResults exported to latent_noise_meta_analysis.csv")

if __name__ == "__main__":
    run_comprehensive_analysis(n_runs=500, steps=100)
