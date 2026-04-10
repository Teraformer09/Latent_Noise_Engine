"""
frontend/app.py
================
Latent Noise Engine — Diagnostic Frontend
Run:  python frontend/app.py
Open: http://localhost:5006
"""
from __future__ import annotations

# Enforce JAX CPU backend before any other import — prevents Windows TPU
# plugin errors (LoadPjrtPlugin not implemented on Windows) and forces
# XLA to use optimal AVX/AVX2 multi-core CPU execution.
import jax
jax.config.update("jax_platform_name", "cpu")

import os, sys, queue, threading, time, logging

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for p in [_REPO, _HERE]:
    if p not in sys.path:
        sys.path.insert(0, p)

import panel as pn

# Panel extension — use only universally supported kwargs
try:
    pn.extension(sizing_mode="stretch_width", notifications=True)
except Exception:
    try:
        pn.extension(sizing_mode="stretch_width")
    except Exception:
        pn.extension()

from frontend.simulator_adapter import NoiseSimulator, DEFAULT_CONFIG
from frontend.viewer.pyvista_view import PyVistaViewer
from frontend.plots.time_series import TimeSeriesPlot
from frontend.plots.stats import StatsPanel
from frontend.plots.qec_view import QECView
from frontend.plots.qsp_poly import QSPPolyView
from frontend.plots.prob_dist import ProbDistView
from frontend.controls.config_panel import ConfigPanel
from frontend.experiment.panel_ui import ExperimentPanel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("latent_frontend")

# ── Config ────────────────────────────────────────────────────────────────
CONFIG = {
    **DEFAULT_CONFIG,
    "d": 3, "shots": 200, "steps": 100_000, "use_qsp": True,
}

# ── Simulator + queue ────────────────────────────────────────────────────
simulator   = NoiseSimulator(CONFIG)
state_queue: queue.Queue = queue.Queue(maxsize=10)
_running    = True

def _simulation_loop():
    while _running:
        try:
            state = simulator.step()
            if not state_queue.full():
                state_queue.put_nowait(state)
        except Exception as e:
            log.error(f"FATAL Simulation step error: {e}", exc_info=True)
            break
        time.sleep(0.05)

_sim_thread = threading.Thread(target=_simulation_loop,
                                name="sim-backend", daemon=True)
_sim_thread.start()
log.info("Background simulation thread started.")

# ── Status bar ───────────────────────────────────────────────────────────
def _format_status(state: dict) -> str:
    step = state.get("step", 0)
    h    = state.get("hazard", 0.0)
    a    = state.get("alpha",  1.0)
    qsp  = "ON" if state.get("use_qsp", False) else "OFF"
    d    = state.get("d", 3)
    err  = state.get("_error", None)
    err_s= f"  ⚠ {str(err)[:60]}" if err else ""
    return (f"**Step:** {step}  |  **H(t):** {h:.4f}  |  "
            f"**α:** {a:.3f}  |  **QSP:** {qsp}  |  **d:** {d}{err_s}")

# ── UI components ─────────────────────────────────────────────────────────
def _get_app():
    _d_init         = CONFIG["d"]
    viewer          = PyVistaViewer(d=_d_init)
    time_plot       = TimeSeriesPlot(maxlen=5000)
    stats           = StatsPanel(maxlen=3000)
    qec_view        = QECView()
    qsp_view        = QSPPolyView()
    prob_view       = ProbDistView()
    controls        = ConfigPanel(simulator)
    experiment_panel= ExperimentPanel(simulator)

    _status_bar = pn.pane.Markdown(
        "**Status:** initialising...",
        sizing_mode="stretch_width",
        styles={"background": "#0f0f1a", "color": "#aaaaff",
                "padding": "4px 12px", "border-radius": "4px"})

    def _ui_update():
        if state_queue.empty():
            return
        try:
            state = state_queue.get_nowait()
        except queue.Empty:
            return
        
        for fn, name in [(viewer.update, "viewer"),
                         (time_plot.update, "time_plot"),
                         (stats.update, "stats"),
                         (qec_view.update, "qec_view"),
                         (qsp_view.update, "qsp_view"),
                         (prob_view.update, "prob_view")]:
            try:
                fn(state)
            except Exception as e:
                log.debug(f"{name}.update error: {e}")
        
        _status_bar.object = _format_status(state)

    pn.state.add_periodic_callback(_ui_update, period=100)

    # ── Layout ────────────────────────────────────────────────────────────────
    _header = pn.pane.Markdown(
        "# 🔬 Latent Noise Engine — Diagnostic Interface",
        sizing_mode="stretch_width",
        styles={"background": "#12122a", "color": "#ccccff",
                "padding": "10px 16px", "border-radius": "6px"})

    _left_col = pn.Column(
        controls.panel,
        experiment_panel.panel,
        width=320)

    _centre_col = pn.Column(
        viewer.panel,
        time_plot.panel,
        qsp_view.panel, # NEW PLOT
        sizing_mode="stretch_width")

    _right_col = pn.Column(
        stats.panel,
        prob_view.panel, # NEW PLOT
        qec_view.panel,
        width=500)

    layout = pn.Column(
        _header,
        _status_bar,
        pn.Row(_left_col, _centre_col, _right_col,
               sizing_mode="stretch_width"),
        sizing_mode="stretch_width")
    
    return layout

# ── Serve ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Starting Panel server → http://localhost:5006")
    pn.serve(_get_app, port=5006, show=False,
             title="Latent Noise Engine — Diagnostic Frontend",
             autoreload=False)
