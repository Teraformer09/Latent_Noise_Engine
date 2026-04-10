"""
frontend/experiment/manager.py
================================
Experiment reproducibility and batch execution manager.

Features
--------
- save_config / load_config   → JSON round-trip
- run_batch                   → N seeds, returns mean±std per metric
- export_csv / export_json    → file output
- deterministic via seed

Does NOT call UI primitives — pure data layer.
"""

from __future__ import annotations

import json
import csv
import copy
import time
import hashlib
import numpy as np
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Config I/O
# ---------------------------------------------------------------------------

def save_config(cfg: dict, path: str | Path) -> str:
    """Serialise config dict to JSON. Returns absolute path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(cfg, f, indent=2, default=_json_serialise)
    return str(p.resolve())


def load_config(path: str | Path) -> dict:
    """Load config from JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p) as f:
        return json.load(f)


def config_hash(cfg: dict) -> str:
    """Deterministic hash of a config dict (for experiment ID)."""
    s = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

class ExperimentManager:
    """
    Manages batch experiment runs, aggregation, and export.

    Usage
    -----
    em = ExperimentManager(simulator)
    results = em.run_batch(n_runs=5, steps_per_run=200)
    em.export_csv(results, "results/batch_01.csv")
    em.export_json(results, "results/batch_01.json")
    """

    def __init__(self, simulator, output_dir: str = "results"):
        self._sim = simulator
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Run management
    # ------------------------------------------------------------------

    def run_batch(
        self,
        n_runs: int = 5,
        steps_per_run: int = 200,
        seeds: list[int] | None = None,
        progress_cb=None,
    ) -> dict:
        """
        Run N independent simulations, aggregate metrics.

        Parameters
        ----------
        n_runs        : number of independent runs
        steps_per_run : steps per run
        seeds         : explicit seeds list (generated if None)
        progress_cb   : optional callable(run_idx, total) for UI progress

        Returns
        -------
        dict with keys:
          runs        : list of per-run result dicts
          mean_hazard : float
          std_hazard  : float
          mean_alpha  : float
          std_alpha   : float
          qec_mean    : dict {3: float, 5: float, 7: float}
          qec_std     : dict {3: float, 5: float, 7: float}
          config      : dict
          timestamp   : str
        """
        if seeds is None:
            rng = np.random.default_rng(42)
            seeds = [int(rng.integers(0, 10000)) for _ in range(n_runs)]

        base_cfg = self._sim.get_config()
        runs = []

        for run_idx, seed in enumerate(seeds[:n_runs]):
            cfg = copy.deepcopy(base_cfg)
            cfg["seed"] = seed

            # Run single simulation
            run_result = self._run_single(cfg, steps_per_run)
            run_result["seed"] = seed
            runs.append(run_result)

            if progress_cb:
                progress_cb(run_idx + 1, n_runs)

        # Aggregate
        all_hazards = [float(np.mean(r["hazards"])) for r in runs]
        all_alphas = [float(np.mean(r["alphas"])) for r in runs]

        qec_mean = {}
        qec_std = {}
        for d in [3, 5, 7]:
            vals = [float(np.mean(r["qec_per_d"].get(d, [0.0]))) for r in runs]
            qec_mean[d] = float(np.mean(vals))
            qec_std[d] = float(np.std(vals))

        result = {
            "runs": runs,
            "mean_hazard": float(np.mean(all_hazards)),
            "std_hazard": float(np.std(all_hazards)),
            "mean_alpha": float(np.mean(all_alphas)),
            "std_alpha": float(np.std(all_alphas)),
            "qec_mean": qec_mean,
            "qec_std": qec_std,
            "config": base_cfg,
            "n_runs": n_runs,
            "steps_per_run": steps_per_run,
            "seeds": seeds[:n_runs],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "experiment_id": config_hash(base_cfg),
        }

        self._history.append(result)
        return result

    def _run_single(self, cfg: dict, steps: int) -> dict:
        """Run one simulation using the high-speed batch method."""
        # Use the simulator's high-speed batch method to run these steps
        # This will be run in the background thread of the experiment manager
        batch_results = self._sim.run_batch_simulation(steps)
        
        return {
            "hazards": batch_results["hazards"],
            "alphas": batch_results["alphas"],
            "qec_per_d": batch_results["qec_per_d"],
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_json(self, result: dict, path: str | Path | None = None) -> str:
        if path is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = self._output_dir / f"experiment_{ts}.json"
        path = Path(path).resolve()
        # Guard against path traversal: ensure export stays within output_dir.
        output_dir_resolved = self._output_dir.resolve()
        try:
            path.relative_to(output_dir_resolved)
        except ValueError:
            raise ValueError(
                f"Export path '{path}' is outside the allowed output directory "
                f"'{output_dir_resolved}'. Refusing to write."
            )
        save_config(result, path)
        return str(path)

    def export_csv(self, result: dict, path: str | Path | None = None) -> str:
        if path is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = self._output_dir / f"experiment_{ts}.csv"
        path = Path(path).resolve()
        # Guard against path traversal: ensure export stays within output_dir.
        output_dir_resolved = self._output_dir.resolve()
        try:
            path.relative_to(output_dir_resolved)
        except ValueError:
            raise ValueError(
                f"Export path '{path}' is outside the allowed output directory "
                f"'{output_dir_resolved}'. Refusing to write."
            )
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for run_idx, run in enumerate(result.get("runs", [])):
            seed = run.get("seed", run_idx)
            for t, (h, a) in enumerate(
                zip(run.get("hazards", []), run.get("alphas", []))
            ):
                row = {
                    "run": run_idx,
                    "seed": seed,
                    "step": t,
                    "hazard": h,
                    "alpha": a,
                }
                for d in [3, 5, 7]:
                    qd = run.get("qec_per_d", {}).get(d, [])
                    row[f"pL_d{d}"] = qd[t] if t < len(qd) else ""
                rows.append(row)

        if rows:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        return str(path)

    def list_results(self) -> list[dict]:
        """Return summary of all batch runs this session."""
        return [
            {
                "experiment_id": r.get("experiment_id"),
                "timestamp": r.get("timestamp"),
                "n_runs": r.get("n_runs"),
                "mean_hazard": r.get("mean_hazard"),
                "std_hazard": r.get("std_hazard"),
            }
            for r in self._history
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_serialise(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Not serialisable: {type(obj)}")
