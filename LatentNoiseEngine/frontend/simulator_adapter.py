"""
frontend/simulator_adapter.py
==============================
Optimized adapter with lazy decoder initialization and support for d=3,5,7,9,11.
"""
from __future__ import annotations
import sys, os, threading
import numpy as np
import jax
jax.config.update("jax_platform_name", "cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_BACKEND_ROOT = _REPO
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

def _try_import_backend():
    try:
        from latent_noise.noise.ou_process import SpatialNoiseField
        from latent_noise.noise.mapping import lambda_to_pauli_probs
        from latent_noise.control.controller import AdaptiveController
        from latent_noise.control.hazard import compute_hazard
        from latent_noise.stim_layer.sampler import (
            sample_surface_code_scaling,
            decode_and_check_logical_scaling,
        )
        from latent_noise.decoder.matching import build_decoder_for_distance
        import jax.numpy as jnp
        from latent_core.qsp.transform import qsp_effective_hamiltonian
        from latent_core.linalg.operators import X, Y, Z
        return dict(
            SpatialNoiseField=SpatialNoiseField,
            lambda_to_pauli_probs=lambda_to_pauli_probs,
            AdaptiveController=AdaptiveController,
            compute_hazard=compute_hazard,
            sample_surface_code_scaling=sample_surface_code_scaling,
            decode_and_check_logical_scaling=decode_and_check_logical_scaling,
            build_decoder_for_distance=build_decoder_for_distance,
            jnp=jnp,
            qsp_effective_hamiltonian=qsp_effective_hamiltonian,
            PAULIS=[X, Y, Z],
        )
    except Exception: return None

DEFAULT_CONFIG = {
    "d": 3, "sigma_spatial": 1.5, "theta_noise": 0.95, "sigma_temporal": 0.05,
    "burst_prob": 0.01, "seed": 42, "base_alpha": 1.0, "kp": 10.0, "ki": 2.0,
    "target_hazard": 0.1, "steps": 10000, "gamma_corr": 0.3, "shots": 200,
    "use_qsp": True, "qsp_coeffs": [1.0, -0.5, 0.1],
}

class NoiseSimulator:
    def __init__(self, config: dict | None = None):
        self._lock = threading.Lock()
        self._config = {**DEFAULT_CONFIG, **(config or {})}
        self._backend = _try_import_backend()
        self._step_count = 0
        self._current_alpha = self._config["base_alpha"]
        self._last_hazard = 0.0
        self._is_paused = False
        self._matching = {} # Lazy loaded

        if self._backend: self._init_real()
        else: self._init_mock()

    def _init_real(self):
        B, cfg = self._backend, self._config
        d = cfg["d"]
        self.coords = [(i, j) for i in range(d) for j in range(d)]
        self._noise_field = B["SpatialNoiseField"](
            coords=self.coords, sigma_spatial=cfg["sigma_spatial"],
            theta=cfg["theta_noise"], sigma_temporal=cfg["sigma_temporal"],
            burst_prob=cfg["burst_prob"], seed=cfg["seed"],
        )
        self._ctrl = B["AdaptiveController"](
            base_alpha=cfg["base_alpha"], kp=cfg["kp"], ki=cfg["ki"],
            target=cfg["target_hazard"],
        )
        self._use_real = True

    def _init_mock(self):
        d = self._config["d"]
        self.coords = [(i, j) for i in range(d) for j in range(d)]
        self._rng = np.random.default_rng(42)
        self._use_real = False

    def step(self, force: bool = False) -> dict:
        if not force:
            with self._lock:
                if self._is_paused: return self._fallback_state("Paused")
        try:
            return self._step_real(force=force) if self._use_real else self._step_mock()
        except Exception as e: return self._fallback_state(str(e))

    def update_params(self, params: dict):
        with self._lock:
            cfg = self._config
            if "base_alpha" in params:
                cfg["base_alpha"] = np.clip(float(params["base_alpha"]), 0.01, 100.0)
                if self._use_real: self._ctrl.base_alpha = self._ctrl.alpha = cfg["base_alpha"]
            if "sigma" in params:
                cfg["sigma_temporal"] = np.clip(float(params["sigma"]), 0.001, 1.0)
                if self._use_real: self._noise_field.sigma_temporal = cfg["sigma_temporal"]
            if "burst_prob" in params:
                cfg["burst_prob"] = np.clip(float(params["burst_prob"]), 0.0, 0.5)
                if self._use_real: self._noise_field.burst_prob = cfg["burst_prob"]
            if "distance" in params:
                val = int(params["distance"])
                if val != cfg["d"]:
                    cfg["d"] = val
                    self.coords = [(i, j) for i in range(val) for j in range(val)]
                    if self._use_real:
                        B = self._backend
                        self._noise_field = B["SpatialNoiseField"](
                            coords=self.coords, sigma_spatial=cfg["sigma_spatial"],
                            theta=cfg["theta_noise"], sigma_temporal=cfg["sigma_temporal"],
                            burst_prob=cfg["burst_prob"], seed=cfg["seed"],
                        )
            if "use_qsp" in params: cfg["use_qsp"] = bool(params["use_qsp"])
            if "target_hazard" in params:
                cfg["target_hazard"] = np.clip(float(params["target_hazard"]), 0.01, 0.99)
                if self._use_real: self._ctrl.target = cfg["target_hazard"]

    def _step_real(self, force: bool = False) -> dict:
        B, cfg = self._backend, self._config
        with self._lock:
            d, use_qsp, shots = cfg["d"], cfg["use_qsp"], cfg["shots"]
            gamma, alpha = cfg["gamma_corr"], self._current_alpha
            lambda_field = self._noise_field.step()
        probs_list = []
        pauli_list = []
        qsp_coeffs = B["jnp"].array(cfg.get("qsp_coeffs", [1.0, -0.5, 0.1]))
        qsp_eigenvalues = None

        for i in range(len(self.coords)):
            l_vec = lambda_field[i]
            if use_qsp:
                H_i = sum(l_vec[j] * B["PAULIS"][j] for j in range(3))
                H_eff = B["qsp_effective_hamiltonian"](H_i, qsp_coeffs)
                evals = np.linalg.eigvalsh(np.array(H_eff))
                l_vec_eff = l_vec * (float(np.max(np.abs(evals))) / (np.linalg.norm(l_vec) + 1e-12))
                if i == 0:
                    # Capture QSP-effective eigenvalues for telemetry
                    qsp_eigenvalues = sorted(float(v) for v in evals)
            else:
                l_vec_eff = l_vec
            p_dict = B["lambda_to_pauli_probs"](l_vec_eff, scale=alpha)
            probs_list.append(p_dict["pz"])
            pauli_list.append(p_dict)

        probs_arr = np.clip(np.array(probs_list), 0.0, 1.0)

        # Lazy Decoder
        if d not in self._matching: self._matching[d] = B["build_decoder_for_distance"](d)

        try:
            batch_shots = max(10, shots // 20)
            errors, syndromes = B["sample_surface_code_scaling"](d=d, probs=probs_arr, coords=self.coords, gamma=gamma, shots=batch_shots)
            log_errs = B["decode_and_check_logical_scaling"](d, errors, syndromes, self._matching[d])
            hazard = float(np.mean(log_errs))
        except Exception: hazard = float(np.mean(probs_arr))

        with self._lock: self._current_alpha = self._ctrl.update(hazard)
        self._step_count += 1

        # Compute mean Pauli probs across all qubits for telemetry
        mean_pauli = {
            k: float(np.mean([p.get(k, 0.0) for p in pauli_list]))
            for k in ("px", "py", "pz", "pi")
        }

        result = {
            "lambda_field": lambda_field, "probabilities": probs_arr, "hazard": hazard,
            "alpha": float(self._current_alpha), "step": self._step_count, "d": d,
            "pauli_probs": mean_pauli,
        }
        if qsp_eigenvalues is not None:
            result["eigenvalues"] = qsp_eigenvalues
        return result

    def _step_mock(self) -> dict:
        d = self._config["d"]
        self._step_count += 1
        probs = 0.05 + 0.02 * np.sin(self._step_count * 0.1) + self._rng.normal(0, 0.01, d*d)
        return {
            "lambda_field": np.zeros((d*d, 3)), "probabilities": np.clip(probs, 0, 1),
            "hazard": float(np.mean(probs)), "alpha": 1.0, "step": self._step_count, "d": d,
        }

    def _fallback_state(self, err: str) -> dict:
        d = self._config["d"]
        return {"step": self._step_count, "d": d, "hazard": 0, "probabilities": np.zeros(d*d), "_error": err}
