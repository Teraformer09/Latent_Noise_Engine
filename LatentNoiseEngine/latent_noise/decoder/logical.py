import numpy as np

def compute_logical_error(samples):
    """
    Return the per-shot logical error array.

    ``samples`` is a 1-D array of per-shot flip bits (0 or 1) produced by
    ``decode_and_check_logical_scaling``.  Each element already encodes
    whether a logical error occurred on that shot; no further majority vote
    is needed.
    """
    return np.asarray(samples, dtype=int)