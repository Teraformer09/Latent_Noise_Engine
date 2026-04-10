import numpy as np

def compute_hazard(logical_errors):
    return float(np.mean(logical_errors))