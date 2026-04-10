"""
latent_core/signal/psd.py
===========================
Power spectral density computation and feature extraction.
Supports chunked / online PSD for use inside the pipeline scan loop.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

from latent_core.signal.fft import compute_psd, estimate_beta, frequency_axis


# ---------------------------------------------------------------------------
# Chunked PSD (for streaming / online use)
# ---------------------------------------------------------------------------

class PSDAccumulator:
    """
    Accumulates a sliding window of signal samples and computes PSD on demand.
    Not JAX-traced — used at Python level between scan calls.
    """

    def __init__(self, chunk_size: int = 64, window: str = "hann"):
        self.chunk_size = chunk_size
        self.window = window
        self._buffer: list[float] = []

    def push(self, sample: float) -> None:
        self._buffer.append(float(sample))
        if len(self._buffer) > self.chunk_size:
            self._buffer.pop(0)

    def ready(self) -> bool:
        return len(self._buffer) >= self.chunk_size

    def compute(self) -> dict:
        """Return dict with psd array and estimated β."""
        signal = jnp.array(self._buffer[-self.chunk_size:])
        psd = compute_psd(signal, self.window)
        beta = estimate_beta(psd)
        return {"psd": psd, "beta": beta}


# ---------------------------------------------------------------------------
# Stateless (pure-function) PSD step for JAX pipeline
# ---------------------------------------------------------------------------

def psd_step(signal_window: Array, window: str = "hann") -> dict:
    """
    Compute PSD features from a fixed-length signal window.

    Parameters
    ----------
    signal_window : (chunk_size,) array — latest signal samples

    Returns
    -------
    dict with keys: psd, beta, log_psd_mean
    """
    psd = compute_psd(signal_window, window)
    beta = estimate_beta(psd)
    log_psd_mean = float(jnp.mean(jnp.log(psd + 1e-12)))

    return {
        "psd": psd,
        "beta": beta,
        "log_psd_mean": log_psd_mean,
    }
