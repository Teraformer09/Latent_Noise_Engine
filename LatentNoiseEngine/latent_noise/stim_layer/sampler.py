import numpy as np
from latent_noise.decoder.matching import get_rotated_surface_code_hx, get_logical_z_op

def sample_surface_code_scaling(d, probs, coords, gamma=0.3, shots=1000):
    N = d * d
    errors = (np.random.random((shots, N)) < probs).astype(int)
    
    # Fast spatial correlation check (nearest neighbors only)
    # This avoids the O(N^2) pair loop
    for i in range(N):
        xi, yi = coords[i]
        for j in range(i + 1, N):
            xj, yj = coords[j]
            # Manhattan distance for grid adjacency
            dist2 = (xi - xj)**2 + (yi - yj)**2
            if dist2 < 1.1: 
                p_corr = gamma * min(probs[i], probs[j])
                if p_corr > 1e-6:
                    # Apply correlated flip to the whole batch at once
                    corr_flips = (np.random.random(shots) < p_corr).astype(int)
                    errors[:, i] ^= corr_flips
                    errors[:, j] ^= corr_flips
    
    H = get_rotated_surface_code_hx(d)
    syndromes = (errors @ H.T) % 2
    return errors, syndromes

def decode_and_check_logical_scaling(d, errors, syndromes, matching):
    logical_op = get_logical_z_op(d)
    logical_errors = []
    
    for i in range(len(errors)):
        prediction = matching.decode(syndromes[i])
        flip = int((np.sum((errors[i] + prediction) * logical_op)) % 2)
        logical_errors.append(flip)
        
    return np.array(logical_errors)
