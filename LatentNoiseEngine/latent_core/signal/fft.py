"""
latent_core/signal/fft.py
===========================
FFT-based spectral analysis with windowing to prevent spectral leakage.

Math
----
ŝ(f) = Σ_t s_w(t) e^{-i2πft}      windowed DFT
S(f)  = |ŝ(f)|²                     power spectral density

For fGn:   S(f) ∝ 1/f^β,   β = 2H - 1
"""

from __future__ import annotations
import jax
# Force x64 globally before any jax.numpy usage
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import Array
import numpy as np


# ---------------------------------------------------------------------------
# Window functions (applied before FFT to reduce spectral leakage)
# ---------------------------------------------------------------------------

WINDOWS = {
    "hann":    np.hanning,
    "hamming": np.hamming,
    "blackman": np.blackman,
    "rect":    np.ones,
}


def apply_window(signal: Array, window: str = "hann") -> Array:
    """Multiply signal by window function w(t)."""
    N = signal.shape[-1]
    w = jnp.array(WINDOWS[window](N))
    return signal * w


# ---------------------------------------------------------------------------
# Core FFT / PSD
# ---------------------------------------------------------------------------

def compute_fft(signal: Array, window: str = "hann") -> Array:
    """
    Windowed FFT.  Returns complex spectrum ŝ(f) of length N//2+1 (one-sided).
    """
    s_w = apply_window(signal, window)
    full = jnp.fft.rfft(s_w)
    return full


def compute_psd(signal: Array, window: str = "hann") -> Array:
    """
    One-sided PSD:  S(f) = |ŝ(f)|².

    Returns
    -------
    psd : (N//2+1,) real array
    """
    spectrum = compute_fft(signal, window)
    psd = jnp.abs(spectrum) ** 2
    # Normalise by window power
    N = signal.shape[-1]
    w = jnp.array(WINDOWS[window](N))
    psd = psd / (jnp.sum(w ** 2) + 1e-12)
    return psd


def frequency_axis(N: int, dt: float = 1.0) -> Array:
    """Return one-sided frequency axis for signal of length N."""
    return jnp.fft.rfftfreq(n=N, d=dt)


# ---------------------------------------------------------------------------
# PSD slope estimation (β for 1/f^β noise)
# ---------------------------------------------------------------------------

def estimate_beta(psd: Array, dt: float = 1.0, f_min_frac: float = 0.05) -> float:
    """
    Estimate spectral exponent β from log-log slope of PSD.

    Fits:  log S(f) = -β log f + const

    Parameters
    ----------
    psd        : one-sided PSD array
    dt         : time step
    f_min_frac : fraction of Nyquist to use as lower frequency cutoff
                 (avoids DC and near-DC artifacts)
    """
    N = (len(psd) - 1) * 2
    freqs = np.array(frequency_axis(N, dt))
    psd_np = np.array(psd)

    # Frequency band selection
    f_min = f_min_frac * 0.5 / dt
    f_max = 0.45 / dt
    mask = (freqs > f_min) & (freqs < f_max) & (psd_np > 0)

    if mask.sum() < 5:
        return 0.0  # not enough points

    log_f = np.log(freqs[mask])
    log_s = np.log(psd_np[mask])

    # Ordinary least squares
    A = np.column_stack([log_f, np.ones_like(log_f)])
    coeffs, _, _, _ = np.linalg.lstsq(A, log_s, rcond=None)
    beta = -coeffs[0]
    return float(beta)
