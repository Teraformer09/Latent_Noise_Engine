"""
frontend/plots/qec_view.py
QEC outcome visualization: logical error rates for d=3,5,7.
constrained_layout replaces tight_layout to silence UserWarning.
"""
from __future__ import annotations
import collections, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

try:
    import panel as pn
    _PANEL_OK = True
except ImportError:
    _PANEL_OK = False

_DISTANCES  = [3, 5, 7]
_COLORS     = {3: "#ff4b5c", 5: "#f7b731", 7: "#20bf6b"}
_MAX_TS_LEN = 2000


class QECView:
    def __init__(self):
        self._qec_ts    = {d: collections.deque(maxlen=_MAX_TS_LEN) for d in _DISTANCES}
        self._last_probs = None
        self._last_d     = 3
        self._step       = 0

        if _PANEL_OK:
            self._plot_pane   = pn.pane.Matplotlib(
                self._make_figure({}), height=360, width=480)
            self._metric_text = pn.pane.Markdown(
                "**QEC Metrics** — waiting for data…", width=480)
            self.panel = pn.Column(
                pn.pane.Markdown("### QEC Diagnostics"),
                self._metric_text, self._plot_pane)
        else:
            self._plot_pane   = None
            self._metric_text = type("T", (), {"object": ""})()
            self.panel        = None

    # ------------------------------------------------------------------
    def update(self, state: dict):
        qec = state.get("qec_metrics", {})
        self._last_probs = np.asarray(
            state.get("probabilities", np.zeros(9)), dtype=float)
        self._last_d = state.get("d", 3)
        self._step   = state.get("step", 0)

        for d in _DISTANCES:
            val = qec.get(d, None)
            if val is not None and np.isfinite(float(val)):
                self._qec_ts[d].append(float(np.clip(val, 0.0, 1.0)))

        parts = [f"**d={d}:** {self._qec_ts[d][-1]:.4f}"
                 for d in _DISTANCES if self._qec_ts[d]]
        self._metric_text.object = "  |  ".join(parts) if parts else "waiting…"

        if _PANEL_OK and self._plot_pane is not None:
            fig = self._make_figure(qec)
            self._plot_pane.object = fig
            plt.close(fig)

    def reset(self):
        for d in _DISTANCES:
            self._qec_ts[d].clear()
        if _PANEL_OK and self._plot_pane is not None:
            fig = self._make_figure({})
            self._plot_pane.object = fig
            plt.close(fig)

    # ------------------------------------------------------------------
    def _make_figure(self, qec_metrics: dict):
        # Use constrained_layout=True — avoids tight_layout UserWarning
        fig = plt.figure(figsize=(5, 4.5), facecolor="#1a1a2e",
                         constrained_layout=True)
        gs     = GridSpec(2, 2, figure=fig, wspace=0.45, hspace=0.55)
        ax_bar = fig.add_subplot(gs[0, :])
        ax_map = fig.add_subplot(gs[1, 0])
        ax_ts  = fig.add_subplot(gs[1, 1])

        for ax in [ax_bar, ax_map, ax_ts]:
            ax.set_facecolor("#0f0f1a")
            ax.tick_params(colors="lightgrey", labelsize=7)
            for sp in ax.spines.values():
                sp.set_color("#333355")

        # Bar chart — current logical error rate per distance
        bars_h   = [float(np.clip(qec_metrics.get(d, 0.0), 0.0, 1.0))
                    for d in _DISTANCES]
        bar_cols = [_COLORS[d] for d in _DISTANCES]
        rects    = ax_bar.bar(range(len(_DISTANCES)), bars_h,
                              color=bar_cols, alpha=0.85, width=0.5)
        for rect, val in zip(rects, bars_h):
            ax_bar.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom",
                color="white", fontsize=7)
        ax_bar.set_xticks(range(len(_DISTANCES)))
        ax_bar.set_xticklabels([f"d={d}" for d in _DISTANCES],
                                color="lightgrey", fontsize=8)
        ax_bar.set_ylim(0.0, 1.05)
        ax_bar.set_ylabel("Logical error rate", color="lightgrey", fontsize=8)
        ax_bar.set_title(f"QEC  [step {self._step}]",
                         color="lightgrey", fontsize=9)

        # Heatmap — per-qubit pZ at current distance
        d = self._last_d
        p = (np.resize(self._last_probs, d * d).reshape(d, d)
             if self._last_probs is not None else np.zeros((d, d)))
        im = ax_map.imshow(p, cmap="hot", aspect="equal", vmin=0.0, vmax=0.5)
        ax_map.set_title(f"pZ heatmap d={d}", color="lightgrey", fontsize=8)
        ax_map.set_xlabel("col", color="lightgrey", fontsize=7)
        ax_map.set_ylabel("row", color="lightgrey", fontsize=7)
        plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)

        # Time series — p_L(t) per distance
        for d_plot in _DISTANCES:
            buf = np.array(self._qec_ts[d_plot])
            if len(buf) > 1:
                ax_ts.plot(buf[-200:], color=_COLORS[d_plot],
                           linewidth=0.8, alpha=0.85, label=f"d={d_plot}")
        if any(len(self._qec_ts[d]) > 1 for d in _DISTANCES):
            ax_ts.legend(fontsize=7, labelcolor="white",
                         facecolor="#1a1a2e", loc="upper right")
        ax_ts.set_ylim(0.0, 1.0)
        ax_ts.set_title("p_L(t) vs d",       color="lightgrey", fontsize=8)
        ax_ts.set_xlabel("recent steps",      color="lightgrey", fontsize=7)
        ax_ts.set_ylabel("p_L",               color="lightgrey", fontsize=7)

        return fig
