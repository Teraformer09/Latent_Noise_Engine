"""
api/worker.py
=============
Simulation worker executed in a separate thread.

Kept in its own module so that Windows multiprocessing (spawn method) does NOT
re-import api.main when forking — that would re-run SimulationManager() and
deadlock before the first loop iteration.
"""
import logging as _log
import os
import sys
import time
from collections import deque
from typing import Any

import msgpack
import numpy as np


# ---------------------------------------------------------------------------
# Backend import helper
# ---------------------------------------------------------------------------

def _try_import_real_backend():
    try:
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from frontend.simulator_adapter import NoiseSimulator
        return NoiseSimulator
    except Exception as exc:
        return None


# ---------------------------------------------------------------------------
# Live config translator
# ---------------------------------------------------------------------------

def _apply_live_config_update(engine: Any, new_cfg: dict, old_cfg: dict, log: Any) -> None:
    params: dict = {}
    if new_cfg.get("rescaling_factor") != old_cfg.get("rescaling_factor"):
        params["base_alpha"] = float(new_cfg["rescaling_factor"])
    if new_cfg.get("tau_corr") != old_cfg.get("tau_corr"):
        params["sigma"] = float(new_cfg["tau_corr"])
    if new_cfg.get("burst_prob") != old_cfg.get("burst_prob"):
        params["burst_prob"] = float(new_cfg["burst_prob"])
    if new_cfg.get("target_hazard") != old_cfg.get("target_hazard"):
        params["target_hazard"] = float(new_cfg["target_hazard"])
    if new_cfg.get("use_qsp") != old_cfg.get("use_qsp"):
        params["use_qsp"] = bool(new_cfg["use_qsp"])

    new_d = int(new_cfg.get("distance", new_cfg.get("d", 3)))
    old_d = int(old_cfg.get("distance", old_cfg.get("d", 3)))
    if new_d != old_d:
        params["distance"] = new_d

    if params:
        try:
            engine.update_params(params)
            log.info("Live update: %s", list(params.keys()))
        except Exception as exc:
            log.warning("Update fail: %s", exc)


# ---------------------------------------------------------------------------
# State coercion
# ---------------------------------------------------------------------------

def _coerce_state(raw: Any, step: int) -> dict:
    def _to_list(v: Any) -> Any:
        if hasattr(v, "tolist"):
            return v.tolist()
        if isinstance(v, (list, tuple)):
            return list(v)
        return v

    if isinstance(raw, dict):
        out: dict = {}
        for k, v in raw.items():
            out[k] = _to_list(v)
        out.setdefault("step", step)
        return out
    return {"step": step, "raw": str(raw)}


# ---------------------------------------------------------------------------
# Main worker entry point
# ---------------------------------------------------------------------------

def simulation_worker(shared_cfg: Any, shared_status: Any, use_redis: bool) -> None:
    _log.basicConfig(level=_log.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    worker_logger = _log.getLogger("sim_worker")
    worker_logger.info("Worker start.")

    # Imports that are safe inside the worker thread
    from api.physics_telemetry import compute_eigenvalues, compute_psd, build_mock_state

    rng = np.random.default_rng(seed=42)
    redis_pub = None
    if use_redis:
        try:
            import redis as _r
            redis_pub = _r.Redis(host="localhost", port=6379)
        except Exception:
            redis_pub = None

    real_engine_cls = _try_import_real_backend()
    real_engine = None
    cfg_snapshot = dict(shared_cfg)
    step = 0
    shared_status["step"] = 0

    # Rolling noise history for PSD computation (circular deque)
    _NOISE_HIST_MAX = 128
    noise_hist: deque = deque(maxlen=_NOISE_HIST_MAX)

    if real_engine_cls:
        try:
            real_engine = real_engine_cls(config=cfg_snapshot)
            worker_logger.info("Engine ready: d=%s", cfg_snapshot.get("d", 3))
        except Exception as exc:
            worker_logger.error("Engine init fail: %s", exc)
            real_engine = None

    while shared_status.get("running", False):
        try:
            cfg = dict(shared_cfg)
            if real_engine and cfg != cfg_snapshot:
                _apply_live_config_update(real_engine, cfg, cfg_snapshot, worker_logger)
                cfg_snapshot = cfg

            if shared_status.get("reinit_requested", False):
                shared_status["reinit_requested"] = False
                if real_engine_cls:
                    try:
                        worker_logger.info("Re-init start.")
                        real_engine = real_engine_cls(config=cfg)
                        cfg_snapshot = cfg
                        noise_hist.clear()
                        worker_logger.info("Re-init done.")
                    except Exception:
                        real_engine = None

            if real_engine:
                raw = real_engine.step(force=True)
                lf = raw.get("lambda_field", [])

                if len(lf) > 0:
                    lf_arr = np.asarray(lf, dtype=float)  # shape (N, 3)

                    # --- Eigenvalues -------------------------------------------
                    # Use QSP eigenvalues if returned by adapter, else compute
                    # from first qubit's raw Hamiltonian.
                    if "eigenvalues" not in raw:
                        l0 = lf_arr[0]
                        H = np.array(
                            [[l0[2], l0[0] - 1j * l0[1]],
                             [l0[0] + 1j * l0[1], -l0[2]]],
                            dtype=complex,
                        )
                        raw["eigenvalues"] = compute_eigenvalues(H)

                    # --- Rolling noise history for PSD -------------------------
                    # Scalar noise metric: mean L2 norm across all qubit lambda vecs
                    noise_scalar = float(np.mean(np.linalg.norm(lf_arr, axis=1)))
                    noise_hist.append(noise_scalar)

                    if len(noise_hist) >= 8:
                        history_arr = np.array(noise_hist)
                        psd_freqs, psd_amps = compute_psd(
                            history_arr,
                            dt=float(cfg.get("dt", 0.05)),
                        )
                        raw["psd_freqs"] = psd_freqs
                        raw["psd_amps"] = psd_amps

                raw.setdefault("psd_freqs", [0.0, 1.0])
                raw.setdefault("psd_amps", [0.1, 0.2])
                raw.setdefault("state_vector", [[1.0, 0.0], [0.0, 0.0]])
                state = _coerce_state(raw, step)
            else:
                state = build_mock_state(step, rng, int(cfg.get("d", 3)))

            step += 1
            shared_status["step"] = step

            frame = msgpack.packb(state, use_bin_type=True)
            if redis_pub:
                try:
                    redis_pub.publish("sim_telemetry", frame)
                except Exception:
                    redis_pub = None
            shared_status["last_frame"] = frame

            time.sleep(float(cfg.get("dt", 0.05)))

        except Exception as exc:
            worker_logger.error("Loop error: %s", exc)
            time.sleep(1.0)
