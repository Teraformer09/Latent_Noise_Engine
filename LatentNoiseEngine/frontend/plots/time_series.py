"""
frontend/plots/time_series.py
Live temporal plots for hazard(t) and alpha(t).
NaN-safe, bounded deque memory, matplotlib-only rendering.
"""
from __future__ import annotations
import collections, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import panel as pn
    _PANEL_OK = True
except ImportError:
    _PANEL_OK = False

_MAX_POINTS = 5000


class TimeSeriesPlot:
    def __init__(self, maxlen: int = _MAX_POINTS):
        self._hazard: collections.deque = collections.deque(maxlen=maxlen)
        self._alpha:  collections.deque = collections.deque(maxlen=maxlen)
        self._steps:  collections.deque = collections.deque(maxlen=maxlen)
        self._maxlen = maxlen

        if _PANEL_OK:
            self._pane = pn.pane.Matplotlib(
                self._make_figure(), height=280, width=680)
            self.panel = pn.Column(
                pn.pane.Markdown("### Temporal Evolution"), self._pane)
        else:
            self._pane = None
            self.panel = None

    def update(self, state: dict):
        h = state.get("hazard", None)
        a = state.get("alpha", None)
        s = state.get("step", len(self._steps))
        if h is not None and np.isfinite(float(h)):
            self._hazard.append(float(np.clip(h, 0.0, 1.0)))
        if a is not None and np.isfinite(float(a)):
            self._alpha.append(float(a))
        self._steps.append(int(s))
        if _PANEL_OK and self._pane is not None:
            fig = self._make_figure()
            self._pane.object = fig
            plt.close(fig)

    def reset(self):
        self._hazard.clear(); self._alpha.clear(); self._steps.clear()
        if _PANEL_OK and self._pane is not None:
            fig = self._make_figure()
            self._pane.object = fig
            plt.close(fig)

    def _make_figure(self):
        haz   = np.array(self._hazard)
        alp   = np.array(self._alpha)
        steps = np.array(self._steps)
        n = min(len(haz), len(alp), len(steps))
        haz, alp, steps = haz[-n:], alp[-n:], steps[-n:]

        fig, (ax_h, ax_a) = plt.subplots(2, 1, figsize=(7, 3.5),
                                          facecolor="#1a1a2e",
                                          constrained_layout=True)

        for ax in [ax_h, ax_a]:
            ax.set_facecolor("#0f0f1a")
            ax.tick_params(colors="lightgrey", labelsize=8)
            for sp in ax.spines.values():
                sp.set_color("#333355")

        if len(haz) > 1:
            ax_h.plot(steps, haz, color="#ff4b5c", linewidth=1.0, alpha=0.9)
            ax_h.fill_between(steps, haz, alpha=0.15, color="#ff4b5c")
            ax_h.axhline(0.3, color="orange", linewidth=0.8,
                         linestyle="--", label="H_crit=0.3")
            ax_h.legend(fontsize=7, loc="upper right",
                        labelcolor="white", facecolor="#1a1a2e")
        ax_h.set_ylabel("Hazard", color="lightgrey", fontsize=9)
        ax_h.set_ylim(0.0, 1.05)
        ax_h.set_title(
            f"hazard(t)  last={haz[-1]:.4f}" if len(haz) else "hazard(t)",
            color="lightgrey", fontsize=9)

        if len(alp) > 1:
            ax_a.plot(steps, alp, color="#4b9fff", linewidth=1.0, alpha=0.9)
            ax_a.fill_between(steps, alp, alpha=0.15, color="#4b9fff")
        ax_a.set_ylabel("α (control)", color="lightgrey", fontsize=9)
        ax_a.set_xlabel("step", color="lightgrey", fontsize=9)
        ax_a.set_title(
            f"alpha(t)  last={alp[-1]:.4f}" if len(alp) else "alpha(t)",
            color="lightgrey", fontsize=9)

        return fig
