"""
api/models.py
=============
Pydantic v2-compatible parameter models for the Latent Noise Engine API.

Every field has strict numeric bounds. FastAPI returns 422 Unprocessable
Entity if a client sends an out-of-range value — the JAX engine never sees
malformed inputs.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class NoiseType(str, Enum):
    ornstein_uhlenbeck = "ornstein_uhlenbeck"
    flicker            = "flicker"
    white              = "white"


class TargetFunction(str, Enum):
    sign      = "sign"
    step      = "step"
    linear    = "linear"
    inversion = "inversion"


# ---------------------------------------------------------------------------
# A. Quantum Signal Processing
# ---------------------------------------------------------------------------

class QSPParams(BaseModel):
    """Controls the QSP phase-gate sequence and polynomial approximation."""

    degree: int = Field(
        default=3,
        ge=1, le=128,
        description="Number of phase gates in the QSP sequence (d). Higher d → "
                    "sharper polynomial approximation of target Hamiltonian.",
    )
    phi_vector: Optional[List[float]] = Field(
        default=None,
        description="Explicit phase angles φ₀…φ_d in (-π, π). When None the "
                    "vector is auto-generated from target_function.",
    )
    target_function: TargetFunction = Field(
        default=TargetFunction.sign,
        description="Auto-generate φ vector to approximate this function.",
    )
    rescaling_factor: float = Field(
        default=1.0,
        ge=0.01, le=10.0,
        description="Rescales the input signal before the QSP transform to "
                    "prevent operator saturation (α).",
    )

    @field_validator("phi_vector")
    @classmethod
    def _validate_phi(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is None:
            return v
        import math
        for angle in v:
            if not (-math.pi <= angle <= math.pi):
                raise ValueError(
                    f"Phase angle {angle:.4f} out of (-π, π). "
                    "QSP solver produces phases via tanh reparameterisation in (-π, π)."
                )
        return v


# ---------------------------------------------------------------------------
# B. Latent Noise Engine
# ---------------------------------------------------------------------------

class NoiseParams(BaseModel):
    """Controls the non-Markovian latent noise dynamics."""

    noise_type: NoiseType = Field(
        default=NoiseType.ornstein_uhlenbeck,
        description="Statistical process driving the noise field.",
    )
    tau_corr: float = Field(
        default=0.05,
        ge=0.001, le=10.0,
        description="Temporal correlation time τ (seconds). Controls non-Markovian "
                    "memory depth — how long noise 'remembers' its previous state.",
    )
    xi_spatial: float = Field(
        default=1.5,
        ge=0.1, le=20.0,
        description="Spatial correlation length ξ (lattice units). Determines "
                    "whether noise is localized or correlated across the lattice.",
    )
    beta_exponent: float = Field(
        default=1.0,
        ge=0.0, le=2.0,
        description="Spectral exponent β for the power-law autocorrelation decay "
                    "⟨ξ(t)ξ(0)⟩ ~ τ^(-β). β=0 → white noise, β=1 → 1/f, β=2 → brown.",
    )
    burst_amplitude: float = Field(
        default=0.5,
        ge=0.0, le=10.0,
        description="Amplitude of burst/glitch events (Cauchy scale parameter). "
                    "Simulates cosmic ray hits or hardware glitches.",
    )
    burst_prob: float = Field(
        default=0.01,
        ge=0.0, le=0.5,
        description="Per-step probability of a burst event occurring.",
    )


# ---------------------------------------------------------------------------
# C. Quantum Error Correction & Controller
# ---------------------------------------------------------------------------

class QECParams(BaseModel):
    """Controls the surface code lattice, decoder, and adaptive PI controller."""

    distance: int = Field(
        default=3,
        description="Surface code lattice distance d. Must be one of 3, 5, 7, 9, 11.",
    )
    p_measure: float = Field(
        default=0.0,
        ge=0.0, le=0.1,
        description="Measurement error rate p_m. Simulates noisy stabilizer "
                    "readout — errors are injected into syndromes with this probability.",
    )
    kp: float = Field(
        default=10.0,
        ge=0.0, le=1000.0,
        description="Proportional gain K_p of the PID controller.",
    )
    ki: float = Field(
        default=2.0,
        ge=0.0, le=500.0,
        description="Integral gain K_i of the PID controller.",
    )
    kd: float = Field(
        default=0.0,
        ge=0.0, le=100.0,
        description="Derivative gain K_d of the PID controller.",
    )
    target_hazard: float = Field(
        default=0.1,
        ge=0.0, le=1.0,
        description="Target hazard setpoint — the logical error rate the "
                    "controller tries to maintain.",
    )

    @field_validator("distance")
    @classmethod
    def _validate_distance(cls, v: int) -> int:
        if v not in (3, 5, 7, 9, 11):
            raise ValueError(
                f"distance={v} is invalid. Must be one of 3, 5, 7, 9, 11."
            )
        return v


# ---------------------------------------------------------------------------
# Composite model — the full parameter payload
# ---------------------------------------------------------------------------

class SimParams(BaseModel):
    """
    Full simulation parameter payload.

    POST to /config/params to atomically update all three sub-systems.
    Any sub-section can be omitted; existing values are preserved.
    """
    qsp: Optional[QSPParams]   = Field(default=None)
    noise: Optional[NoiseParams] = Field(default=None)
    qec: Optional[QECParams]   = Field(default=None)


# ---------------------------------------------------------------------------
# Flat config snapshot returned by GET /config
# ---------------------------------------------------------------------------

class ConfigSnapshot(BaseModel):
    """What the API returns when you GET /config — a flat dict of all params."""
    # QSP
    degree: int
    phi_vector: Optional[List[float]]
    target_function: str
    rescaling_factor: float
    # Noise
    noise_type: str
    tau_corr: float
    xi_spatial: float
    beta_exponent: float
    burst_amplitude: float
    burst_prob: float
    # QEC
    distance: int
    p_measure: float
    kp: float
    ki: float
    kd: float
    target_hazard: float
    # Runtime
    decimation: int
    dt: float
