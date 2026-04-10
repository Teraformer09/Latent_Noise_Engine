"""
latent_core/engine/generator.py
=================================
Batch trajectory generator: runs N independent simulations in parallel.

Usage
-----
>>> gen = BatchGenerator(cfg, n_trajectories=128)
>>> results = gen.run(seed=42)
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from latent_core.engine.pipeline import LatentNoisePipeline
from latent_core.utils.config import SimConfig
from latent_core.utils.logging import get_logger

logger = get_logger(__name__)


class BatchGenerator:
    """
    Runs N independent LatentNoisePipeline simulations.

    Each trajectory gets an independent seed derived from the base seed.
    Results are aggregated into arrays of shape (N, T).
    """

    def __init__(self, cfg: dict, n_trajectories: int = 32):
        self.cfg = cfg
        self.N = n_trajectories

    def run(self, seed: int = 42, progress: bool = True) -> dict:
        """
        Run N trajectories and return aggregated statistics.

        Returns
        -------
        dict with:
            fidelities    : (N, T) array
            hazards       : (N, T) array
            betas         : (N, T) array
            failure_times : (N,) array (None → survived)
            mean_fidelity : (T,) mean fidelity across trajectories
            std_fidelity  : (T,) std fidelity across trajectories
        """
        all_fidelities = []
        all_hazards = []
        all_betas = []
        failure_times = []

        T = SimConfig.from_dict(self.cfg).horizon

        iterator = range(self.N)
        if progress:
            iterator = tqdm(iterator, desc="Trajectories")

        for i in iterator:
            pipe = LatentNoisePipeline(self.cfg)
            result = pipe.run(seed=seed + i)

            F = result["fidelities"]
            H = result["hazards"]
            B = result["betas"]

            # Pad to T if trajectory terminated early
            F = _pad_to(F, T)
            H = _pad_to(H, T)
            B = _pad_to(B, T)

            all_fidelities.append(F)
            all_hazards.append(H)
            all_betas.append(B)
            failure_times.append(result["failure_time"])

        fidelities = np.stack(all_fidelities)   # (N, T)
        hazards = np.stack(all_hazards)
        betas = np.stack(all_betas)

        return {
            "fidelities": fidelities,
            "hazards": hazards,
            "betas": betas,
            "failure_times": np.array(
                [ft if ft is not None else T for ft in failure_times]
            ),
            "mean_fidelity": np.mean(fidelities, axis=0),
            "std_fidelity": np.std(fidelities, axis=0),
            "survival_fraction": np.mean([ft is None for ft in failure_times]),
        }


def _pad_to(arr: np.ndarray, T: int) -> np.ndarray:
    """Pad array to length T with its last value."""
    if len(arr) >= T:
        return arr[:T]
    pad_val = arr[-1] if len(arr) > 0 else 0.0
    pad = np.full(T - len(arr), pad_val)
    return np.concatenate([arr, pad])
