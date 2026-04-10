"""
latent_core/signal/autocorr.py
================================
Autocorrelation and Hurst exponent estimation.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array


def autocorrelation(x: Array, max_lag: int) -> Array:
    """
    Unbiased sample autocorrelation up to max_lag.

    ρ(τ) = Σ_{t=0}^{N-τ-1} (x_t - x̄)(x_{t+τ} - x̄) / (N var(x))
    """
    x = x - jnp.mean(x)
    N = len(x)
    var = jnp.var(x) * N
    lags = []
    for tau in range(max_lag + 1):
        c = jnp.sum(x[:N - tau] * x[tau:]) / (var + 1e-15)
        lags.append(float(c))
    return jnp.array(lags)


def hurst_rs(x: np.ndarray) -> float:
    """
    R/S statistic estimate of Hurst exponent.
    Classic method: H = log(R/S) / log(N).
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    mean_x = np.mean(x)
    deviations = np.cumsum(x - mean_x)
    R = np.max(deviations) - np.min(deviations)
    S = np.std(x, ddof=1)
    if S == 0:
        return 0.5
    return float(np.log(R / S) / np.log(N))


def hurst_dfa(x: np.ndarray, scales: list[int] | None = None) -> float:
    """
    Detrended Fluctuation Analysis (DFA) Hurst exponent.
    More robust than R/S for non-stationary series.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    y = np.cumsum(x - np.mean(x))

    if scales is None:
        scales = [int(s) for s in np.logspace(1, np.log10(N // 4), 20)]

    fluctuations = []
    valid_scales = []
    for n in scales:
        if n >= N:
            continue
        n_segments = N // n
        if n_segments < 2:
            continue
        F = []
        for i in range(n_segments):
            seg = y[i * n: (i + 1) * n]
            t = np.arange(n)
            p = np.polyfit(t, seg, 1)
            trend = np.polyval(p, t)
            F.append(np.mean((seg - trend) ** 2))
        fluctuations.append(np.sqrt(np.mean(F)))
        valid_scales.append(n)

    if len(valid_scales) < 3:
        return 0.5

    log_n = np.log(valid_scales)
    log_F = np.log(fluctuations)
    H, _ = np.polyfit(log_n, log_F, 1)
    return float(np.clip(H, 0.0, 1.0))
