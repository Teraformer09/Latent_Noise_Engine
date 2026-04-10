"""
latent_core/engine/pipeline.py
================================
Master pipeline: orchestrates the full stochastic → signal → QSP → evolution → metrics chain.

Full system equation:
    ρ_{t+1} = ℰ_t( P(H(λ(t))) ρ_t P(H(λ(t)))† )

Computational graph:
    key → λ(t) → signal → PSD → β → polynomial → QSP → H_eff → evolution → metrics

Design
------
- Python-level orchestration (no global JAX trace across all modules)
- JAX used for inner loops: stochastic steps, channel application, fidelity
- QSP solver called at Python level (optimization-based, not traced)
- Outputs logged per step: fidelity, hazard, beta, probs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import jax
import jax.numpy as jnp
from jax import Array

from latent_core.stochastic.fgn import generate_fgn
from latent_core.stochastic.drift import drift_step
from latent_core.signal.fft import compute_psd, estimate_beta
from latent_core.signal.features import extract_features, RunningStats
from latent_core.qsp.polynomial import build_polynomial
# phase-based QSP solver removed: operate on polynomial coefficients directly
from latent_core.qsp.transform import qsp_effective_hamiltonian
from latent_core.physics.hamiltonian import build_hamiltonian
from latent_core.mapping.softmax import theta_to_probs, init_softmax_weights
from latent_core.mapping.pauli_channel import evolve_step
from latent_core.metrics.fidelity import pure_fidelity
from latent_core.metrics.hazard import HazardTracker
from latent_core.linalg.expm import expm_hamiltonian
from latent_core.linalg.operators import I2
from latent_core.utils.config import load_config, SimConfig
from latent_core.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------

@dataclass
class PipelineState:
    """All mutable state carried forward across time steps."""

    rho: Array                    # (d, d) density matrix
    drift: Array                  # (n_axes,) current drift
    theta: Array                  # (m,) latent state vector
    signal_buffer: list[float]    # rolling signal window
    step: int = 0
    qsp_coeffs: Array | None = None  # cached polynomial coefficients


@dataclass
class StepOutput:
    """Logged quantities for one time step."""

    t: int
    fidelity: float
    hazard: float
    beta: float
    probs: Array
    lambda_vals: Array
    qsp_loss: float = 0.0


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class LatentNoisePipeline:
    """
    Full stochastic quantum noise simulation pipeline.

    Usage
    -----
    >>> pipe = LatentNoisePipeline.from_config("configs/default.yaml")
    >>> outputs = pipe.run(seed=42)
    """

    def __init__(self, cfg: dict):
        # Allow passing a partial config dict; merge with defaults
        if isinstance(cfg, dict):
            # Convert to typed SimConfig with defaults filled
            simcfg = SimConfig.from_dict(cfg)
        elif isinstance(cfg, SimConfig):
            simcfg = cfg
        else:
            raise TypeError("cfg must be dict or SimConfig")

        # Keep both typed and raw dict forms
        self.simcfg = simcfg
        self.cfg = cfg
        self._setup(simcfg)

    @classmethod
    def from_config(cls, path: str) -> "LatentNoisePipeline":
        cfg = load_config(path)
        return cls(cfg)

    def _setup(self, cfg: SimConfig) -> None:
        # Dimensions — use typed SimConfig attributes
        self.d = cfg.hilbert_dim
        self.T = cfg.horizon
        self.dt = cfg.dt
        self.warmup = cfg.warmup

        # Noise
        self.hurst = cfg.hurst
        self.drift_sigma = cfg.drift_sigma
        self.chunk_size = cfg.chunk_size
        self.window = cfg.signal_window

        # Polynomial & QSP
        self.poly_degree = cfg.poly_degree
        self.poly_method = cfg.poly_method
        self.poly_reg = cfg.poly_reg
        # qsp settings as accessible mapping
        self.qsp_cfg = {
            "n_iter": cfg.qsp_n_iter,
            "lr": cfg.qsp_lr,
            "reinit_trials": cfg.qsp_reinit_trials,
            "convergence_tol": cfg.qsp_tol,
            "unitary_check": cfg.qsp_unitary_check,
        }

        # Hazard
        self.h_crit = cfg.h_crit
        self.gamma = cfg.gamma

        # Softmax weights (latent → probs)
        key = jax.random.PRNGKey(cfg.seed)
        # Uninformative initialisation: near-uniform channel probabilities
        self.W, _ = init_softmax_weights(n_paulis=4, latent_dim=3, key=key, scale=0.01)
        self.b = jnp.zeros(4)

        # Target state |0⟩ for fidelity
        self.psi_target = jnp.array([1.0, 0.0], dtype=jnp.complex128)

        # Running normalisation stats for signal features
        self.running_stats = RunningStats(dim=3)

        # Operator ordering and runtime flags (allow raw cfg dict to override)
        if isinstance(self.cfg, dict):
            self.operator_ordering = self.cfg.get("system", {}).get(
                "operator_ordering", getattr(self.simcfg, "ordering", "channel_first")
            )
            self.use_qsp_flag = self.cfg.get("use_qsp", True)
            # mode: 'phases' (solve phases) or 'poly' (apply polynomial directly)
            self.use_qsp_mode = self.cfg.get("use_qsp_mode", "poly")
        else:
            self.operator_ordering = getattr(self.simcfg, "ordering", "channel_first")
            self.use_qsp_flag = True
            self.use_qsp_mode = "poly"

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _init_state(self, key: Array) -> PipelineState:
        # Start in |0⟩⟨0| pure state
        rho = jnp.outer(self.psi_target, jnp.conj(self.psi_target))
        drift = jnp.zeros(2)  # [λ_X, λ_Z]
        theta = jnp.zeros(3)  # [λ_X², λ_Z², c_t]

        # Pre-generate fGn for full trajectory (Davies-Harte, efficient)
        key, k1, k2 = jax.random.split(key, 3)
        self._fgn_X = np.array(generate_fgn(k1, self.T + self.chunk_size, self.hurst))
        self._fgn_Z = np.array(generate_fgn(k2, self.T + self.chunk_size, self.hurst))

        return PipelineState(
            rho=rho,
            drift=drift,
            theta=theta,
            signal_buffer=[],
        )

    # -----------------------------------------------------------------------
    # Step functions
    # -----------------------------------------------------------------------

    def _stochastic_step(self, state: PipelineState, key: Array, t: int) -> tuple[Array, Array]:
        """Update drift + fGn → λ(t) → θ_t."""
        key, subkey = jax.random.split(key)
        drift_next = drift_step(subkey, state.drift, self.drift_sigma)

        # Retrieve pre-generated fGn at time t
        fgn_X = float(self._fgn_X[t])
        fgn_Z = float(self._fgn_Z[t])

        lambda_X = float(drift_next[0]) + fgn_X
        lambda_Z = float(drift_next[1]) + fgn_Z

        # Latent state: [λ_X², λ_Z², correlation estimate]
        # Accumulate a short history to estimate X-Z cross-correlation.
        if not hasattr(self, "_lambda_hist_X"):
            self._lambda_hist_X: list[float] = []
            self._lambda_hist_Z: list[float] = []
        self._lambda_hist_X.append(lambda_X)
        self._lambda_hist_Z.append(lambda_Z)
        if len(self._lambda_hist_X) > 32:
            self._lambda_hist_X.pop(0)
            self._lambda_hist_Z.pop(0)

        if len(self._lambda_hist_X) >= 4:
            corr_mat = float(np.corrcoef(self._lambda_hist_X, self._lambda_hist_Z)[0, 1])
            corr = 0.0 if np.isnan(corr_mat) else corr_mat
        else:
            corr = 0.0

        theta = jnp.array([lambda_X**2, lambda_Z**2, corr])
        return drift_next, theta, jnp.array([lambda_X, lambda_Z])

    def _signal_step(self, lambda_vals: Array, state: PipelineState) -> dict:
        """Accumulate signal buffer and compute PSD features."""
        signal_val = float(jnp.mean(jnp.abs(lambda_vals)))
        state.signal_buffer.append(signal_val)
        if len(state.signal_buffer) > self.chunk_size:
            state.signal_buffer.pop(0)

        if len(state.signal_buffer) < self.chunk_size:
            # Not enough data yet — return defaults
            return {"beta": 1.0, "log_psd_mean": -5.0, "signal_std": 0.01}

        sig_arr = jnp.array(state.signal_buffer)
        feats = extract_features(sig_arr, self.window, self.running_stats.to_dict())
        self.running_stats.update(np.array(feats["raw"]))
        return feats

    def _polynomial_step(self, feats: dict) -> Array:
        """Map signal features → Chebyshev polynomial coefficients.

        We build a smooth lowpass polynomial and normalise its values to lie
        within [-1, 1] which is required for QSP feasibility.
        """
        beta = float(feats.get("beta", 1.0))
        coeffs = build_polynomial(beta, self.poly_degree, method="lowpass", reg=self.poly_reg)
        # normalise polynomial values to be within [-1, 1]
        x_grid = jnp.linspace(-1.0, 1.0, 500)
        from latent_core.qsp.polynomial import eval_polynomial
        vals = eval_polynomial(coeffs, x_grid)
        coeffs = coeffs / (jnp.max(jnp.abs(vals)) + 1e-12)
        return coeffs

    def _qsp_step(self, coeffs: Array) -> tuple[Array, float]:
        """Phase-based QSP solver intentionally disabled; keep signature.

        The pipeline applies polynomials directly (poly mode). This
        placeholder preserves the API for callers but does not perform
        any optimization.
        """
        # No-op placeholder
        return None, 0.0

    def _evolution_step(
        self,
        state: PipelineState,
        theta: Array,
        lambda_vals: Array,
        qsp_coeffs: Array,
        key: Array,
    ) -> tuple[Array, Array]:
        """Full quantum evolution: ρ_{t+1} = ℰ_t(P(H_t) ρ_t P(H_t)†).

        Returns
        -------
        rho_next, probs
        """

        # Build Hamiltonian
        H = build_hamiltonian(float(lambda_vals[0]), float(lambda_vals[1]), normalise=True)

        # QSP effective Hamiltonian: apply polynomial only when enabled
        if self.use_qsp_flag and qsp_coeffs is not None:
            H_eff = qsp_effective_hamiltonian(H, qsp_coeffs)
        else:
            H_eff = H

        # Scale QSP effect by configured alpha
        alpha = getattr(self.simcfg, "alpha", 64.0)
        H_eff = alpha * H_eff

        # Unitary
        U = expm_hamiltonian(H_eff, t=self.dt)

        # Channel probabilities from latent state
        probs = theta_to_probs(theta, self.W, self.b)
        # Ensure valid probability simplex
        probs = jnp.clip(probs, 0.0, jnp.inf)
        probs = probs / (jnp.sum(probs) + 1e-12)

        # Evolve — ensure we use the effective Hamiltonian
        if self.use_qsp_flag and qsp_coeffs is not None:
            # reduce spam: only log periodically
            if state.step % 100 == 0:
                logger.info("QSP active (poly)")
        rho_next = evolve_step(state.rho, probs, U=U, ordering=self.operator_ordering)
        return rho_next, probs

    # -----------------------------------------------------------------------
    # Main simulation loop
    # -----------------------------------------------------------------------

    def run(self, seed: int | None = None) -> dict:
        """
        Run the full pipeline for T steps.

        Returns
        -------
        dict with keys: fidelities, hazards, betas, outputs
        """
        # determine seed (respect explicit 0)
        if seed is None:
            seed = (self.cfg.get("seed") if isinstance(self.cfg, dict) else getattr(self.simcfg, "seed", 42))
        key = jax.random.PRNGKey(int(seed))

        # initialise state and trackers
        state = self._init_state(key)
        hazard_tracker = HazardTracker(h_crit=self.h_crit, gamma=self.gamma)
        outputs: list[StepOutput] = []
        rhos: list[np.ndarray] = []

        # Initial polynomial (before warmup)
        qsp_coeffs = build_polynomial(1.0, self.poly_degree, self.poly_method)
        qsp_loss_val = 0.0

        # logging frequency
        log_every = getattr(self.simcfg, "log_every", 50)

        for t in range(self.T):
            key, subkey = jax.random.split(key)

            # 1. Stochastic update
            state.drift, theta, lambda_vals = self._stochastic_step(state, subkey, t)
            state.theta = theta

            # 2. Signal features
            feats = self._signal_step(lambda_vals, state)

            # 3. Polynomial (update every chunk_size steps after warmup)
            if t >= self.warmup and t % self.chunk_size == 0:
                qsp_coeffs = self._polynomial_step(feats)
                state.qsp_coeffs = qsp_coeffs

                # 4. QSP solver (expensive — only update periodically)
                if self.use_qsp_flag:
                    if self.use_qsp_mode == "phases":
                        _, qsp_loss_val = self._qsp_step(qsp_coeffs)
                    else:
                        # poly mode: skip phase solving, operate directly on polynomial
                        qsp_loss_val = 0.0

            active_coeffs = state.qsp_coeffs if state.qsp_coeffs is not None else qsp_coeffs

            # 5. Evolution
            rho_next, probs = self._evolution_step(state, theta, lambda_vals, active_coeffs, subkey)
            state.rho = rho_next
            state.step = t

            # 6. Metrics
            F = pure_fidelity(state.rho, self.psi_target)
            hazard_tracker.update(F)

            out = StepOutput(
                t=t,
                fidelity=F,
                hazard=hazard_tracker.current_hazard,
                beta=float(feats.get("beta", 1.0)),
                probs=probs,
                lambda_vals=lambda_vals,
                qsp_loss=qsp_loss_val,
            )
            outputs.append(out)

            if t % log_every == 0:
                logger.info(
                    f"t={t:4d} | F={F:.4f} | H={hazard_tracker.current_hazard:.4f} "
                    f"| β={out.beta:.3f} | p_I={float(probs[0]):.3f}"
                )

            # mark failure but continue to collect full trajectory
            if hazard_tracker.failure_time == t:
                logger.info(f"Failure at t={t} (H_t ≥ H_crit={self.h_crit})")

            rhos.append(np.array(state.rho))

        result = hazard_tracker.to_arrays()
        # metrics time series
        fidelities = np.array([float(o.fidelity) for o in outputs])
        hazards = np.array([float(o.hazard) for o in outputs])
        betas = np.array([float(o.beta) for o in outputs])
        probs_arr = np.stack([np.array(o.probs) for o in outputs])

        # von Neumann entropy for each rho
        entropies = []
        for r in rhos:
            try:
                vals = np.linalg.eigvalsh(r)
                vals = np.clip(vals, 1e-12, 1.0)
                S = -np.sum(vals * np.log(vals))
            except Exception:
                S = np.nan
            entropies.append(S)
        entropies = np.array(entropies)

        result.update({
            "outputs": outputs,
            "fidelities": fidelities,
            "hazards": hazards,
            "betas": betas,
            "probs": probs_arr,
            "rhos": np.stack(rhos),
            "entropies": entropies,
        })

        return result
