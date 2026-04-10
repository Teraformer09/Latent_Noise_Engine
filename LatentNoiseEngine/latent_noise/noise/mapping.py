import numpy as np

def lambda_to_pauli_probs(lambda_vec, scale=10.0):
    """
    Numerically stable Gibbs-based mapping.
    Fixes overflow by clamping energy values and using stable sigmoid.
    """
    alpha = scale
    beta = 5.0    
    theta = 0.1   
    
    # 1. Physical Energy
    E_xyz = np.array(lambda_vec)
    
    # 2. Gibbs Distribution
    # logits = -alpha * E. Clamped to safe range.
    logits = np.clip(-alpha * E_xyz, -20, 20)
    
    # Stable softmax
    shift_logits = logits - np.max(logits)
    exp_logits = np.exp(shift_logits)
    p_xyz_tilde = exp_logits / np.sum(exp_logits)
    
    # 3. Identity Gating (Cooling)
    # g_logit is proportional to how far the effective noise is from threshold.
    # No constant alpha offset — that drives p_I -> 1 regardless of noise level.
    lambda_eff = np.mean(E_xyz)
    g_logit = np.clip(beta * (lambda_eff - theta), -20, 20)
    
    # p_I = 1 / (1 + exp(-g_logit))
    p_I = 1.0 / (1.0 + np.exp(-g_logit))
    
    # 4. Final Pauli Probabilities
    p_xyz = (1.0 - p_I) * p_xyz_tilde
    
    return {
        "px": float(p_xyz[0]),
        "py": float(p_xyz[1]),
        "pz": float(p_xyz[2]),
        "pi": float(p_I)
    }
