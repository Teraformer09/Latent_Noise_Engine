"""
frontend/plots/stats.py
Live statistical diagnostics: histogram, mean/var, ACF.
NaN-safe. Bounded memory. constrained_layout replaces tight_layout.
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

_MAX_HIST_POINTS = 3000
_N_BINS = 30


class StatsPanel:
    def __init__(self, maxlen: int = _MAX_HIST_POINTS):
        self._hazard_buf: collections.deque = collections.deque(maxlen=maxlen)
        self._prob_buf:   collections.deque = collections.deque(maxlen=200)

        if _PANEL_OK:
            self._text_pane = pn.pane.Markdown(
                "**Step:** 0  |  **Mean hazard:** —  |  **Var:** —",
                width=680)
            self._plot_pane = pn.pane.Matplotlib(
                self._make_figure(), height=300, width=680)
            self.panel = pn.Column(
                pn.pane.Markdown("### Statistical Diagnostics"),
                self._text_pane, self._plot_pane)
        else:
            self._text_pane = type("T", (), {"object": ""})()
            self._plot_pane = None
            self.panel = None

    def update(self, state: dict):
        h     = state.get("hazard", None)
        probs = state.get("probabilities", None)
        step  = state.get("step", 0)

        if h is not None and np.isfinite(float(h)):
            self._hazard_buf.append(float(np.clip(h, 0.0, 1.0)))
        if probs is not None:
            p = np.asarray(probs, dtype=float)
            p = p[np.isfinite(p)]
            if len(p):
                self._prob_buf.append(p)

        haz    = np.array(self._hazard_buf)
        mean_h = float(np.mean(haz)) if len(haz) else 0.0
        var_h  = float(np.var(haz))  if len(haz) else 0.0

        self._text_pane.object = (
            f"**Step:** {step}  |  **Mean hazard:** {mean_h:.4f}  |  "
            f"**Var:** {var_h:.6f}  |  **N samples:** {len(haz)}"
        )
        if _PANEL_OK and self._plot_pane is not None:
            fig = self._make_figure()
            self._plot_pane.object = fig
            plt.close(fig)

    def reset(self):
        self._hazard_buf.clear(); self._prob_buf.clear()
        if _PANEL_OK and self._plot_pane is not None:
            fig = self._make_figure()
            self._plot_pane.object = fig
            plt.close(fig)

    def _make_figure(self):
        haz = np.array(self._hazard_buf)

        fig, (ax_h, ax_a) = plt.subplots(1, 2, figsize=(7, 3.5),
                                          facecolor="#1a1a2e",
                                          constrained_layout=True)

        for ax in [ax_h, ax_a]:
            ax.set_facecolor("#0f0f1a")
            ax.tick_params(colors="lightgrey", labelsize=8)
            for sp in ax.spines.values():
                sp.set_color("#333355")

        # Histogram
        if len(haz) > 5:
            ax_h.hist(haz, bins=_N_BINS, color="#ff4b5c", alpha=0.75,
                      edgecolor="#ff8888", density=True)
            ax_h.axvline(np.mean(haz), color="orange", linewidth=1.5,
                         linestyle="--", label=f"μ={np.mean(haz):.3f}")
            ax_h.axvline(np.mean(haz)+np.std(haz), color="yellow",
                         linewidth=0.8, linestyle=":",
                         label=f"σ={np.std(haz):.3f}")
            ax_h.axvline(np.mean(haz)-np.std(haz), color="yellow",
                         linewidth=0.8, linestyle=":")
            ax_h.legend(fontsize=7, labelcolor="white", facecolor="#1a1a2e")
        ax_h.set_title("Hazard distribution", color="lightgrey", fontsize=9)
        ax_h.set_xlabel("hazard", color="lightgrey", fontsize=8)
        ax_h.set_ylabel("density", color="lightgrey", fontsize=8)
        ax_h.set_xlim(0.0, 1.0)

        # ACF
        if len(haz) > 20:
            max_lag = min(50, len(haz) // 2)
            lags    = np.arange(max_lag)
            acf     = self._compute_acf(haz, max_lag)
            ax_a.bar(lags, acf, color="#4b9fff", alpha=0.7, width=0.8)
            conf = 1.96 / np.sqrt(len(haz))
            ax_a.axhline( conf, color="orange", linewidth=0.8, linestyle="--")
            ax_a.axhline(-conf, color="orange", linewidth=0.8, linestyle="--")
            ax_a.axhline(0,     color="white",  linewidth=0.5)
        ax_a.set_title("Hazard autocorrelation", color="lightgrey", fontsize=9)
        ax_a.set_xlabel("lag",  color="lightgrey", fontsize=8)
        ax_a.set_ylabel("ACF",  color="lightgrey", fontsize=8)
        ax_a.set_ylim(-1.05, 1.05)

        return fig

    @staticmethod
    def _compute_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
        x   = x - np.mean(x)
        var = np.var(x)
        if var < 1e-12:
            return np.zeros(max_lag)
        acf = [1.0]
        for lag in range(1, max_lag):
            acf.append(float(np.mean(x[:-lag] * x[lag:])) / var)
        return np.array(acf)
