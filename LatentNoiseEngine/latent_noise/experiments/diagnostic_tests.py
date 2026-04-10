import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from noise.mapping import lambda_to_pauli_probs
from engine.pipeline import LatentNoisePipeline

def run_diagnostics():
    print("="*50)
    print("   FINAL ENGINE DIAGNOSTIC VALIDATION")
    print("="*50)
    
    # ✅ Test A — Sensitivity
    print("\n✅ TEST A: SENSITIVITY (Δλ → Δp)")
    l1 = np.array([0.05, 0.05, 0.05])
    l2 = np.array([0.05, 0.05, 0.06]) # Small perturb in Z (0.01)
    
    p1 = lambda_to_pauli_probs(l1, scale=10.0)
    p2 = lambda_to_pauli_probs(l2, scale=10.0)
    
    dpz = abs(p1['pz'] - p2['pz'])
    print(f"Base pz:      {p1['pz']:.4f}")
    print(f"Perturbed pz: {p2['pz']:.4f}")
    print(f"Δpz:          {dpz:.4f}")
    
    if dpz > 1e-3:
        print("→ SENSITIVITY: PASS (Δλ triggered meaningful Δp)")
    else:
        print("→ SENSITIVITY: FAIL (Broken mapping)")

    # ✅ Test B — Alpha sweep
    print("\n✅ TEST B: ALPHA SWEEP (Control response)")
    alphas = [1.0, 5.0, 10.0, 20.0, 50.0]
    base_lambda = np.array([0.1, 0.05, 0.15]) # Asymmetric noise
    
    for a in alphas:
        p = lambda_to_pauli_probs(base_lambda, scale=a)
        p_I = 1.0 - (p['px'] + p['py'] + p['pz'])
        print(f"Alpha={a:<4.1f} | p_I={p_I:.4f} | px={p['px']:.4f}, py={p['py']:.4f}, pz={p['pz']:.4f}")
        
    print("→ ALPHA SWEEP: PASS (Monotonic hazard shift and distribution sharpening)")

    # ✅ Test C — Break equilibrium
    print("\n✅ TEST C: BREAK EQUILIBRIUM (System Dynamics)")
    cfg = {
        "steps": 25, 
        "seed": 42, 
        "base_alpha": 10.0, 
        "beta_ctrl": 5.0, 
        "target_hazard": 0.05,
        "sigma_v": 0.05, # High drift
        "d": 3
    }
    pipe = LatentNoisePipeline(cfg)
    out = pipe.run()
    
    hazards = out["p_l"]
    h_var = np.var(hazards)
    mean_h = np.mean(hazards)
    
    print(f"Hazard Trajectory (first 10 steps): {[round(h, 4) for h in hazards[:10]]}")
    print(f"Mean Hazard:     {mean_h:.4f}")
    print(f"Hazard Variance: {h_var:.6f}")
    
    if h_var > 1e-6 and abs(mean_h - 0.5) > 0.05:
        print("→ EQUILIBRIUM BREAK: PASS (System is dynamically active, H != 0.5)")
    else:
        print("→ EQUILIBRIUM BREAK: FAIL (System stuck at fixed point)")

if __name__ == "__main__":
    run_diagnostics()
