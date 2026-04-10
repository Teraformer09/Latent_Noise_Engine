"""
latent_core/utils/config.py
=============================
Config loading with YAML merge support.

Usage
-----
>>> cfg = load_config("configs/default.yaml")
>>> cfg = load_config("configs/non_markov.yaml", base="configs/default.yaml")
"""

from __future__ import annotations

import copy
import yaml
from pathlib import Path


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str, base: str | None = None) -> dict:
    """
    Load YAML config, optionally merging with a base config.

    Parameters
    ----------
    path : path to config file (overrides)
    base : optional path to base config (defaults applied first)

    Returns
    -------
    cfg : merged config dict
    """
    path = Path(path)
    assert path.exists(), f"Config not found: {path}"

    with open(path) as f:
        cfg = yaml.safe_load(f)

    if base is not None:
        base_path = Path(base)
        assert base_path.exists(), f"Base config not found: {base_path}"
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f)
        cfg = _deep_merge(base_cfg, cfg)

    return cfg


def dump_config(cfg: dict, path: str) -> None:
    """Save config dict to YAML file."""
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=True)


# ---------------------------------------------------------------------------
# SimConfig — typed dataclass derived from config dict
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class SimConfig:
    """
    Typed configuration object for the simulation pipeline.
    Derived from config YAML dict via SimConfig.from_dict() or SimConfig.from_yaml().
    """

    # Time
    dt: float = 0.001
    horizon: int = 1000
    warmup: int = 50

    # System
    hilbert_dim: int = 2
    n_qubits: int = 1
    normalize_H: bool = True
    ordering: str = "channel_first"

    # Noise
    hurst: float = 0.7
    drift_sigma: float = 0.01
    coupling_axes: list = None  # type: ignore

    # Signal
    signal_window: str = "hann"
    chunk_size: int = 64
    normalize_signal: bool = True

    # Polynomial
    poly_degree: int = 5
    poly_method: str = "lowpass"
    poly_reg: float = 1e-3

    # QSP
    qsp_n_iter: int = 500
    qsp_lr: float = 0.01
    qsp_reinit_trials: int = 5
    qsp_tol: float = 1e-6
    qsp_unitary_check: bool = True

    # Channel
    softmax_temp: float = 1.0

    # Latent
    latent_dim: int = 3

    # Training / metrics
    gamma: float = 0.99
    h_crit: float = 0.3
    # QSP / polynomial scaling
    alpha: float = 64.0
    # noise added to softmax probabilities to break equilibrium
    prob_noise: float = 0.05

    # Logging
    log_every: int = 50
    seed: int = 42

    def __post_init__(self):
        if self.coupling_axes is None:
            self.coupling_axes = ["X", "Z"]

    @classmethod
    def from_dict(cls, cfg: dict) -> "SimConfig":
        """Build SimConfig from a nested config dict."""
        flat = {}
        # time
        flat["dt"] = cfg.get("time", {}).get("dt", 0.01)
        flat["horizon"] = cfg.get("time", {}).get("horizon", 1000)
        flat["warmup"] = cfg.get("time", {}).get("warmup", 50)
        # system
        flat["hilbert_dim"] = cfg.get("system", {}).get("hilbert_dim", 2)
        flat["n_qubits"] = cfg.get("system", {}).get("n_qubits", 1)
        flat["normalize_H"] = cfg.get("system", {}).get("normalize_H", True)
        flat["ordering"] = cfg.get("system", {}).get("operator_ordering", "channel_first")
        # noise
        flat["hurst"] = cfg.get("noise", {}).get("hurst", 0.7)
        flat["drift_sigma"] = cfg.get("noise", {}).get("drift_sigma", 0.01)
        flat["coupling_axes"] = cfg.get("noise", {}).get("coupling_axes", ["X", "Z"])
        # signal
        flat["signal_window"] = cfg.get("signal", {}).get("window", "hann")
        flat["chunk_size"] = cfg.get("signal", {}).get("chunk_size", 64)
        flat["normalize_signal"] = cfg.get("signal", {}).get("normalize", True)
        # polynomial
        flat["poly_degree"] = cfg.get("polynomial", {}).get("degree", 5)
        flat["poly_method"] = cfg.get("polynomial", {}).get("construction", "lowpass")
        flat["poly_reg"] = cfg.get("polynomial", {}).get("regularization", 1e-3)
        # qsp
        flat["qsp_n_iter"] = cfg.get("qsp", {}).get("n_iter", 500)
        flat["qsp_lr"] = cfg.get("qsp", {}).get("lr", 0.01)
        flat["qsp_reinit_trials"] = cfg.get("qsp", {}).get("reinit_trials", 5)
        flat["qsp_tol"] = cfg.get("qsp", {}).get("convergence_tol", 1e-6)
        flat["qsp_unitary_check"] = cfg.get("qsp", {}).get("unitary_check", True)
        # channel
        flat["softmax_temp"] = cfg.get("channel", {}).get("softmax_temp", 1.0)
        # latent
        flat["latent_dim"] = cfg.get("latent", {}).get("dim", 3)
        # training
        flat["gamma"] = cfg.get("training", {}).get("gamma", 0.99)
        # metrics
        flat["h_crit"] = cfg.get("metrics", {}).get("hazard_crit", 0.3)
        # qsp / poly scaling
        flat["alpha"] = cfg.get("qsp", {}).get("alpha", 64.0)
        flat["prob_noise"] = cfg.get("channel", {}).get("prob_noise", 0.05)
        # logging
        flat["log_every"] = cfg.get("logging", {}).get("log_every", 50)
        flat["seed"] = cfg.get("seed", 42)
        return cls(**flat)

    @classmethod
    def from_yaml(cls, path: str, base: str = "configs/default.yaml") -> "SimConfig":
        """Load from YAML file, optionally merging with base config."""
        cfg_dict = load_config(path, base=base if Path(base).exists() else None)
        return cls.from_dict(cfg_dict)

    @classmethod
    def default(cls) -> "SimConfig":
        """Return default configuration."""
        return cls()
