"""Signal processing: windowed FFT, PSD estimation, spectral features."""
from .fft import compute_fft, compute_psd, frequency_axis, estimate_beta
from .features import extract_features, RunningStats
from .autocorr import autocorrelation, hurst_dfa, hurst_rs
