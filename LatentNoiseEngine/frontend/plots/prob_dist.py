"""
frontend/plots/prob_dist.py
===========================
Probability distribution change over time.
Visualizes how the controller (alpha) shifts the Pauli probability budget.
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

_MAX_TS_LEN = 1000

class ProbDistView:
    def __init__(self):
        self._px = collections.deque(maxlen=_MAX_TS_LEN)
        self._py = collections.deque(maxlen=_MAX_TS_LEN)
        self._pz = collections.deque(maxlen=_MAX_TS_LEN)
        self._pi = collections.deque(maxlen=_MAX_TS_LEN)
        self._steps = collections.deque(maxlen=_MAX_TS_LEN)

        if _PANEL_OK:
            self._plot_pane = pn.pane.Matplotlib(
                self._make_figure(), height=280, width=480)
            self.panel = pn.Column(
                pn.pane.Markdown("### Probability Budget Evolution"),
                self._plot_pane)
        else:
            self.panel = None

    def update(self, state: dict):
        if not _PANEL_OK: return
        
        step = state.get("step", 0)
        # Assuming the backend returns the full mapping dict in 'pauli_probs'
        probs = state.get("pauli_probs", {"px": 0.05, "py": 0.05, "pz": 0.05, "pi": 0.85})
        
        self._px.append(probs.get("px", 0))
        self._py.append(probs.get("py", 0))
        self._pz.append(probs.get("pz", 0))
        self._pi.append(probs.get("pi", 0))
        self._steps.append(step)
        
        fig = self._make_figure()
        self._plot_pane.object = fig
        plt.close(fig)

    def _make_figure(self):
        fig, ax = plt.subplots(figsize=(5, 3), facecolor="#1a1a2e", constrained_layout=True)
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors="lightgrey", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#333355")
            
        if len(self._steps) > 1:
            steps = np.array(self._steps)
            px = np.array(self._px)
            py = np.array(self._py)
            pz = np.array(self._pz)
            pi = np.array(self._pi)
            
            # Stacked plot
            ax.stackplot(steps, px, py, pz, pi, 
                         labels=["pX", "pY", "pZ", "pI"],
                         colors=["#ff4b5c", "#f7b731", "#20bf6b", "#4b9fff"],
                         alpha=0.8)
            
        ax.set_title("Pauli Probability Gating (Mean)", color="lightgrey", fontsize=9)
        ax.set_xlabel("step", color="lightgrey", fontsize=8)
        ax.set_ylabel("Probability sum=1.0", color="lightgrey", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white", loc="lower left")
        
        return fig
