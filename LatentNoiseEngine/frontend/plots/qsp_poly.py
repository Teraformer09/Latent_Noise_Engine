"""
frontend/plots/qsp_poly.py
==========================
Visualize the QSP spectral transformation polynomial P(x) vs x.
Shows how the QSP layer maps raw Hamiltonian strength to effective control.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import panel as pn
    _PANEL_OK = True
except ImportError:
    _PANEL_OK = False

class QSPPolyView:
    def __init__(self):
        self._coeffs = [1.0, -0.5, 0.1]
        self._x = np.linspace(-1, 1, 100)
        
        if _PANEL_OK:
            self._plot_pane = pn.pane.Matplotlib(
                self._make_figure(), height=260, width=480)
            self.panel = pn.Column(
                pn.pane.Markdown("### QSP Spectral Response"),
                self._plot_pane)
        else:
            self.panel = None

    def update(self, state: dict):
        if not _PANEL_OK: return
        
        coeffs = state.get("qsp_coeffs", self._coeffs)
        self._coeffs = coeffs
        
        fig = self._make_figure()
        self._plot_pane.object = fig
        plt.close(fig)

    def _make_figure(self):
        fig, ax = plt.subplots(figsize=(5, 2.5), facecolor="#1a1a2e", constrained_layout=True)
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors="lightgrey", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#333355")
            
        # P(x) = sum(c_i * T_i(x))
        def eval_cheby(c, x):
            if len(c) == 0: return np.zeros_like(x)
            if len(c) == 1: return c[0] * np.ones_like(x)
            t0 = np.ones_like(x)
            t1 = x.copy()
            res = c[0] * t0 + c[1] * t1
            for i in range(2, len(c)):
                t2 = 2 * x * t1 - t0
                res += c[i] * t2
                t0, t1 = t1, t2
            return res

        y = eval_cheby(self._coeffs, self._x)
        ax.plot(self._x, self._x, color="grey", linestyle="--", alpha=0.5, label="Identity")
        ax.plot(self._x, y, color="#4b9fff", linewidth=1.5, label="QSP P(x)")
        
        ax.set_title("QSP Mapping: H_raw → H_eff", color="lightgrey", fontsize=9)
        ax.set_xlabel("Signal Strength (x)", color="lightgrey", fontsize=8)
        ax.set_ylabel("P(x)", color="lightgrey", fontsize=8)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1.1, 1.1)
        ax.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white")
        
        return fig
