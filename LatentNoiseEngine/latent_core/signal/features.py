"""
latent_core/signal/features.py
================================
Normalised feature vector extracted from signal/PSD for polynomial mapping.

z_t = [β_t, log_psd_mean_t, H_est_t]  (normalised to zero mean, unit variance)

These feed into polynomial_step → QSP.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

from latent_core.signal.fft import compute_psd, estimate_beta


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    signal_window: Array,
    window: str = "hann",
    running_stats: dict | None = None,
) -> dict:
    """
    Extract normalised features from signal window.

    Parameters
    ----------
    signal_window  : (chunk_size,) signal array
    window         : spectral window type
    running_stats  : optional dict with 'mean' and 'std' arrays (shape (3,))
                     for online normalisation

    Returns
    -------
    dict with:
        raw   : (3,) raw features [beta, log_psd_mean, signal_std]
        norm  : (3,) normalised features
    """
    psd = compute_psd(signal_window, window)
    beta = estimate_beta(psd)
    log_psd_mean = float(jnp.mean(jnp.log(psd + 1e-12)))
    signal_std = float(jnp.std(signal_window))

    raw = np.array([beta, log_psd_mean, signal_std])

    if running_stats is not None:
        mu = running_stats["mean"]
        sigma = running_stats["std"] + 1e-8
        norm = (raw - mu) / sigma
    else:
        # Rough default normalisation
        norm = (raw - np.array([1.0, -5.0, 0.01])) / np.array([1.0, 2.0, 0.01])

    return {
        "raw": jnp.array(raw),
        "norm": jnp.array(norm),
        "beta": beta,
        "log_psd_mean": log_psd_mean,
    }


# ---------------------------------------------------------------------------
# Running statistics (online mean/variance for normalisation)
# ---------------------------------------------------------------------------

class RunningStats:
    """Welford online algorithm for streaming mean/variance."""

    def __init__(self, dim: int = 3):
        self.n = 0
        self.mean = np.zeros(dim)
        self.M2 = np.zeros(dim)

    def update(self, x: np.ndarray) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def std(self) -> np.ndarray:
        if self.n < 2:
            return np.ones_like(self.mean)
        return np.sqrt(self.M2 / (self.n - 1))

    def to_dict(self) -> dict:
        return {"mean": self.mean.copy(), "std": self.std.copy()}
