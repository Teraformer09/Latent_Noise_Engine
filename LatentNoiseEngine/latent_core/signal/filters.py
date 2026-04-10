"""
latent_core/signal/filters.py
================================
Digital filters for pre-processing signal before spectral analysis.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
import numpy as np
from scipy.signal import butter, sosfilt


def butter_lowpass(signal: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """Apply Butterworth low-pass filter."""
    nyq = 0.5 * fs
    sos = butter(order, cutoff / nyq, btype="low", output="sos")
    return sosfilt(sos, signal)


def butter_highpass(signal: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """Apply Butterworth high-pass filter."""
    nyq = 0.5 * fs
    sos = butter(order, cutoff / nyq, btype="high", output="sos")
    return sosfilt(sos, signal)


def moving_average(x: Array, window: int) -> Array:
    """Causal moving average (no future leakage)."""
    kernel = jnp.ones(window) / window
    return jnp.convolve(x, kernel, mode="full")[: len(x)]
