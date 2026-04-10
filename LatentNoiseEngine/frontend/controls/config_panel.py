"""
frontend/controls/config_panel.py
===================================
Parameter control interface. Sliders bind to simulator.update_params() immediately.
Compatible with Panel 0.x through 1.x — uses pn.layout.Divider instead of pn.Divider.
"""
from __future__ import annotations

try:
    import panel as pn
    # Resolve Divider across Panel versions
    try:
        _Divider = pn.layout.Divider
    except AttributeError:
        try:
            _Divider = pn.Divider
        except AttributeError:
            _Divider = None          # very old Panel — skip dividers
    _PANEL_OK = True
except ImportError:
    _PANEL_OK = False

import numpy as np


def _divider():
    """Return a Panel divider widget, or None if unavailable."""
    if not _PANEL_OK or _Divider is None:
        return None
    try:
        return _Divider()
    except Exception:
        return None


class ConfigPanel:
    def __init__(self, simulator):
        self._sim = simulator

        if _PANEL_OK:
            self._alpha_slider = pn.widgets.FloatSlider(
                name="Base alpha (QSP strength)",
                start=0.01, end=50.0, step=0.1, value=1.0,
                bar_color="#4b9fff", sizing_mode="stretch_width")
            self._sigma_slider = pn.widgets.FloatSlider(
                name="Sigma temporal (noise intensity)",
                start=0.001, end=0.5, step=0.001, value=0.05,
                bar_color="#ff4b5c", sizing_mode="stretch_width")
            self._burst_slider = pn.widgets.FloatSlider(
                name="Burst probability",
                start=0.0, end=0.2, step=0.005, value=0.01,
                bar_color="#f7b731", sizing_mode="stretch_width")
            self._target_slider = pn.widgets.FloatSlider(
                name="Target hazard (controller setpoint)",
                start=0.01, end=0.9, step=0.01, value=0.1,
                bar_color="#20bf6b", sizing_mode="stretch_width")
            self._dist_selector = pn.widgets.RadioButtonGroup(
                name="QEC distance",
                options=[3, 5, 7], value=3,
                button_type="default", sizing_mode="stretch_width")
            self._qsp_toggle = pn.widgets.Toggle(
                name="QSP enabled",
                value=True, button_type="primary",
                sizing_mode="stretch_width")
            self._play_pause = pn.widgets.Toggle(
                name="▶ Play", value=False, button_type="success",
                sizing_mode="stretch_width")
            self._status = pn.pane.Markdown(
                "_Ready — adjust sliders_", width=280)

            self._play_pause.param.watch(self._on_play_pause, "value")
            self._alpha_slider.param.watch(self._on_alpha,    "value")
            self._sigma_slider.param.watch(self._on_sigma,    "value")
            self._burst_slider.param.watch(self._on_burst,    "value")
            self._target_slider.param.watch(self._on_target,  "value")
            self._dist_selector.param.watch(self._on_distance,"value")
            self._qsp_toggle.param.watch(self._on_qsp,        "value")

            # Build layout items, inserting dividers only when available
            items = [pn.pane.Markdown("### Simulation Controls")]
            items += [self._play_pause]
            div = _divider()
            if div is not None:
                items.append(div)
            items += [pn.pane.Markdown("**QSP**"), self._qsp_toggle, self._alpha_slider]
            div = _divider()
            if div is not None:
                items.append(div)
            items += [pn.pane.Markdown("**Noise**"), self._sigma_slider, self._burst_slider]
            div = _divider()
            if div is not None:
                items.append(div)
            items += [pn.pane.Markdown("**QEC / Controller**"), self._dist_selector, self._target_slider]
            div = _divider()
            if div is not None:
                items.append(div)
            items.append(self._status)

            self.panel = pn.Column(*items, width=300)
        else:
            self._alpha_slider  = type("W", (), {"value": 1.0})()
            self._sigma_slider  = type("W", (), {"value": 0.05})()
            self._burst_slider  = type("W", (), {"value": 0.01})()
            self._target_slider = type("W", (), {"value": 0.1})()
            self._dist_selector = type("W", (), {"value": 3})()
            self._qsp_toggle    = type("W", (), {"value": True, "name": "QSP enabled"})()
            self._status        = type("T", (), {"object": ""})()
            self.panel = None

    def set_enabled(self, enabled: bool):
        """Enable or disable all configuration widgets."""
        self._alpha_slider.disabled = not enabled
        self._sigma_slider.disabled = not enabled
        self._burst_slider.disabled = not enabled
        self._target_slider.disabled = not enabled
        self._dist_selector.disabled = not enabled
        self._qsp_toggle.disabled = not enabled

    # ------------------------------------------------------------------
    def _on_play_pause(self, event):
        val = bool(event.new)
        # Toggle value: True = "Playing" → simulator NOT paused (set_paused=False)
        #               False = "Paused"  → simulator IS  paused (set_paused=True)
        self._sim.set_paused(not val)
        self.set_enabled(not val)  # Disable sliders while simulation is running
        if val:
            self._play_pause.name = "⏸ Pause"
            self._play_pause.button_type = "danger"
            self._status.object = "▶ Simulation running..."
        else:
            self._play_pause.name = "▶ Play"
            self._play_pause.button_type = "success"
            self._status.object = "⏸ Simulation paused."

    def _on_alpha(self, event):
        val = float(event.new)
        if 0.01 <= val <= 50.0:
            self._sim.update_params({"base_alpha": val})
            self._status.object = f"✓ alpha → {val:.3f}"

    def _on_sigma(self, event):
        val = float(event.new)
        if 0.001 <= val <= 0.5:
            self._sim.update_params({"sigma": val})
            self._status.object = f"✓ sigma → {val:.4f}"

    def _on_burst(self, event):
        val = float(event.new)
        if 0.0 <= val <= 0.2:
            self._sim.update_params({"burst_prob": val})
            self._status.object = f"✓ burst_prob → {val:.4f}"

    def _on_target(self, event):
        val = float(event.new)
        if 0.01 <= val <= 0.9:
            self._sim.update_params({"target_hazard": val})
            self._status.object = f"✓ target_hazard → {val:.3f}"

    def _on_distance(self, event):
        val = int(event.new)
        if val in [3, 5, 7]:
            self._sim.update_params({"distance": val})
            self._status.object = f"✓ QEC distance → d={val}"

    def _on_qsp(self, event):
        val = bool(event.new)
        self._sim.update_params({"use_qsp": val})
        label = "ON" if val else "OFF"
        self._status.object = f"✓ QSP → {label}"
        self._qsp_toggle.name = f"QSP [{label}]"
