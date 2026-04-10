"""
frontend/experiment/panel_ui.py
Panel UI for the experiment manager.
Compatible with Panel 0.x–1.x (no pn.Divider dependency).
"""
from __future__ import annotations
import threading
import numpy as np

try:
    import panel as pn
    _PANEL_OK = True
except ImportError:
    _PANEL_OK = False

from .manager import ExperimentManager


class ExperimentPanel:
    def __init__(self, simulator):
        self._manager = ExperimentManager(simulator)

        if not _PANEL_OK:
            self.panel = None
            return

        self._n_runs = pn.widgets.IntSlider(
            name="N runs", start=1, end=20, value=3,
            sizing_mode="stretch_width")
        self._steps = pn.widgets.IntSlider(
            name="Steps per run", start=50, end=500, step=50, value=100,
            sizing_mode="stretch_width")
        self._run_btn = pn.widgets.Button(
            name="▶ Run Batch", button_type="primary",
            sizing_mode="stretch_width")
        
        self._csv_download = pn.widgets.FileDownload(
            label="Download CSV", button_type="default",
            sizing_mode="stretch_width", disabled=True,
            filename="experiment_results.csv",
            callback=self._get_csv_data)
        
        self._json_download = pn.widgets.FileDownload(
            label="Download JSON", button_type="default",
            sizing_mode="stretch_width", disabled=True,
            filename="experiment_results.json",
            callback=self._get_json_data)

        self._status       = pn.pane.Markdown("_No runs yet_",    width=300)
        self._results_pane = pn.pane.Markdown("",                 width=300)
        self._last_result  = None

        self._run_btn.on_click(self._on_run)

        self.panel = pn.Column(
            pn.pane.Markdown("### Experiment Manager"),
            self._n_runs,
            self._steps,
            self._run_btn,
            pn.Row(self._csv_download, self._json_download),
            self._status,
            self._results_pane,
            width=310)

    # ------------------------------------------------------------------
    def _on_run(self, event):
        self._run_btn.disabled = True
        self._status.object = "⏳ Running batch…"

        def _run():
            try:
                result = self._manager.run_batch(
                    n_runs=self._n_runs.value,
                    steps_per_run=self._steps.value,
                    progress_cb=self._on_progress)
                self._last_result = result
                self._status.object = (
                    f"✓ Done  |  mean hazard={result['mean_hazard']:.4f}"
                    f"  ±{result['std_hazard']:.4f}")
                self._render_summary(result)
                self._csv_download.disabled  = False
                self._json_download.disabled = False
            except Exception as e:
                self._status.object = f"❌ Error: {e}"
            finally:
                self._run_btn.disabled = False

        threading.Thread(target=_run, daemon=True).start()

    def _on_progress(self, done: int, total: int):
        self._status.object = f"⏳ Run {done}/{total}…"

    def _get_csv_data(self):
        if self._last_result:
            import io
            path = self._manager.export_csv(self._last_result)
            with open(path, "rb") as f:
                return io.BytesIO(f.read())
        return None

    def _get_json_data(self):
        if self._last_result:
            import io
            path = self._manager.export_json(self._last_result)
            with open(path, "rb") as f:
                return io.BytesIO(f.read())
        return None

    def _render_summary(self, result: dict):
        lines = [
            f"**Experiment:** `{result.get('experiment_id','?')}`",
            f"**Runs:** {result['n_runs']}  |  **Steps:** {result['steps_per_run']}",
            f"**Mean hazard:** {result['mean_hazard']:.4f} ± {result['std_hazard']:.4f}",
            f"**Mean alpha:** {result['mean_alpha']:.4f} ± {result['std_alpha']:.4f}",
            "", "**QEC logical error rates:**"]
        for d in [3, 5, 7]:
            m = result["qec_mean"].get(d, 0.0)
            s = result["qec_std"].get(d, 0.0)
            lines.append(f"  d={d}: {m:.4f} ± {s:.4f}")
        self._results_pane.object = "\n".join(lines)
