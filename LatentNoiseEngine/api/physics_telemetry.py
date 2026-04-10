"""
physics_telemetry.py
--------------------
Utility functions for computing eigenvalues, power spectral density, and
building mock telemetry state dicts suitable for msgpack serialisation.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_eigenvalues(H: np.ndarray) -> list[float]:
    """
    Compute sorted eigenvalues of a 2×2 Hermitian matrix.

    Parameters
    ----------
    H : np.ndarray, shape (2, 2), dtype complex or float
        A Hermitian matrix.

    Returns
    -------
    list[float]
        Eigenvalues in ascending order as a plain Python list.
    """
    H_arr = np.asarray(H, dtype=complex)
    if H_arr.shape != (2, 2):
        raise ValueError(f"H must be 2×2, got shape {H_arr.shape}")
    eigenvalues, _ = np.linalg.eigh(H_arr)
    return sorted(float(v) for v in eigenvalues)


def compute_psd(
    noise_history: np.ndarray,
    dt: float = 0.05,
) -> tuple[list[float], list[float]]:
    """
    FFT-based Power Spectral Density over the provided noise history.

    Parameters
    ----------
    noise_history : np.ndarray, shape (N,)
        A 1-D array of scalar noise samples.
    dt : float
        Sampling interval in seconds (default 0.05 s → 20 Hz).

    Returns
    -------
    (freqs, amps) : tuple[list[float], list[float]]
        Non-negative one-sided frequency bins and their PSD amplitudes,
        both as plain Python lists.
    """
    signal = np.asarray(noise_history, dtype=float)
    n = len(signal)
    if n < 2:
        return [0.0], [0.0]

    freqs = np.fft.rfftfreq(n, d=dt)
    spectrum = np.fft.rfft(signal)
    amps = (np.abs(spectrum) ** 2).real

    return freqs.tolist(), amps.tolist()


def build_mock_state(
    step: int,
    rng: np.random.Generator,
    d: int = 3,
) -> dict:
    """
    Build a complete telemetry state dict suitable for msgpack serialisation.

    All numpy arrays are converted to plain Python lists so that msgpack can
    serialise them without a custom encoder.

    Parameters
    ----------
    step : int
        Current simulation step index.
    rng : np.random.Generator
        Seeded NumPy random generator for reproducibility.
    d : int
        Surface-code distance (default 3).

    Returns
    -------
    dict
        Wire-format telemetry frame.
    """
    # --- Pauli probabilities (sum to 1) ------------------------------------
    raw_pauli = rng.dirichlet([1.0, 1.0, 1.0, 10.0])  # biased toward I
    px, py, pz, pi = float(raw_pauli[0]), float(raw_pauli[1]), float(raw_pauli[2]), float(raw_pauli[3])

    # --- General probability vector (d² elements) --------------------------
    n_qubits = d * d
    raw_probs = rng.dirichlet(np.ones(n_qubits))
    probabilities: list[float] = raw_probs.tolist()

    # --- Hazard & alpha ----------------------------------------------------
    hazard = float(np.clip(rng.normal(0.05, 0.02), 0.0, 1.0))
    alpha = float(np.clip(rng.normal(0.5, 0.1), 0.0, 1.0))

    # --- Lambda field: (N, 3) — one [lx, ly, lz] row per qubit -------------
    n_qubits = d * d
    lambda_field: list[list[float]] = rng.uniform(-1.0, 1.0, (n_qubits, 3)).tolist()

    # --- QEC metrics: fixed distance keys {3, 5, 7} ----------------------
    qec_metrics: dict[int, float] = {
        k: float(np.clip(rng.exponential(0.01 * k), 0.0, 1.0))
        for k in (3, 5, 7)
    }

    # --- Eigenvalues of a mock 2×2 Hamiltonian ----------------------------
    omega = float(rng.uniform(0.5, 2.0))
    delta = float(rng.uniform(-0.3, 0.3))
    H = np.array([[delta, omega], [omega, -delta]], dtype=complex)
    eigenvalues: list[float] = compute_eigenvalues(H)

    # --- PSD over a short noise window ------------------------------------
    window_size = max(64, min(256, step + 1))
    noise_history = rng.standard_normal(window_size)
    psd_freqs, psd_amps = compute_psd(noise_history, dt=0.05)

    # --- State vector (2-element complex → [[re, im], [re, im]]) ----------
    raw_sv = rng.standard_normal(4).view(complex)  # 2 complex numbers
    norm = float(np.linalg.norm(raw_sv))
    if norm > 0:
        raw_sv = raw_sv / norm
    state_vector: list[list[float]] = [
        [float(c.real), float(c.imag)] for c in raw_sv
    ]

    return {
        "step": int(step),
        "lambda_field": lambda_field,
        "probabilities": probabilities,
        "pauli_probs": {
            "px": px,
            "py": py,
            "pz": pz,
            "pi": pi,
        },
        "hazard": hazard,
        "alpha": alpha,
        "qec_metrics": qec_metrics,
        "use_qsp": True,
        "d": int(d),
        "eigenvalues": eigenvalues,
        "psd_freqs": psd_freqs,
        "psd_amps": psd_amps,
        "state_vector": state_vector,
    }
