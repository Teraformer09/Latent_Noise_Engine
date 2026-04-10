"""
latent_core/engine/scheduler.py
=================================
Ablation and baseline experiment scheduler.

Runs structured comparisons:
  - Baseline A: no QSP, Markovian noise (H = 0.5)
  - Baseline B: no QSP, non-Markovian noise (H > 0.5)
  - System:     QSP active, non-Markovian noise

These three must be compared to prove QSP adds value.
"""

from __future__ import annotations

import copy
import numpy as np
from dataclasses import dataclass
from typing import Any

from latent_core.engine.generator import BatchGenerator
from latent_core.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExperimentSpec:
    name: str
    config_overrides: dict  # nested dict of overrides applied to base cfg


STANDARD_ABLATIONS = [
    ExperimentSpec(
        name="markov_no_qsp",
        config_overrides={
            "noise": {"hurst": 0.5},
            "system": {"use_qsp": False},
        },
    ),
    ExperimentSpec(
        name="non_markov_no_qsp",
        config_overrides={
            "noise": {"hurst": 0.7},
            "system": {"use_qsp": False},
        },
    ),
    ExperimentSpec(
        name="non_markov_with_qsp",
        config_overrides={
            "noise": {"hurst": 0.7},
            "system": {"use_qsp": True},
        },
    ),
]


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into a copy of base."""
    result = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class AblationScheduler:
    """
    Runs a battery of ablation experiments and returns comparative results.

    Usage
    -----
    >>> sched = AblationScheduler(base_cfg, n_trajectories=64)
    >>> results = sched.run(seed=0)
    """

    def __init__(
        self,
        base_cfg: dict,
        n_trajectories: int = 32,
        experiments: list[ExperimentSpec] | None = None,
    ):
        self.base_cfg = base_cfg
        self.N = n_trajectories
        self.experiments = experiments or STANDARD_ABLATIONS

    def run(self, seed: int = 0) -> dict[str, Any]:
        """
        Run all experiments and return comparative summary.

        Returns
        -------
        dict: experiment_name → BatchGenerator result dict
        """
        results = {}

        for spec in self.experiments:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running: {spec.name}")
            logger.info(f"Overrides: {spec.config_overrides}")

            cfg = _deep_merge(self.base_cfg, spec.config_overrides)
            gen = BatchGenerator(cfg, n_trajectories=self.N)
            result = gen.run(seed=seed, progress=True)

            results[spec.name] = result
            logger.info(
                f"  survival_fraction={result['survival_fraction']:.3f} "
                f"  mean_final_fidelity={result['mean_fidelity'][-1]:.4f}"
            )

        return results

    def summary_table(self, results: dict) -> str:
        """Format a simple comparison table."""
        lines = [
            f"{'Experiment':<30} {'Survival':>10} {'Mean F(T)':>12} {'Mean TTF':>10}",
            "-" * 65,
        ]
        for name, r in results.items():
            survived = r["survival_fraction"]
            final_F = r["mean_fidelity"][-1]
            mean_ttf = np.mean(r["failure_times"])
            lines.append(f"{name:<30} {survived:>10.3f} {final_F:>12.4f} {mean_ttf:>10.1f}")
        return "\n".join(lines)
