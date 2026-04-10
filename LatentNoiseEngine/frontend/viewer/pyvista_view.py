"""
frontend/viewer/pyvista_view.py
================================
3D voxel visualization of the spatial noise field.
Modes: "lambda" | "prob" | "qsp"
Falls back to matplotlib heatmap if pyvista unavailable.
constrained_layout avoids tight_layout warnings.
"""
from __future__ import annotations
import numpy as np

try:
    import pyvista as pv
    _PYVISTA_OK = True # RESTORE 3D CAPABILITY
except ImportError:
    _PYVISTA_OK = False

try:
    import panel as pn
    _PANEL_OK = True
except ImportError:
    _PANEL_OK = False


def _make_heatmap(data_2d, title, cmap="viridis"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="#1a1a2e",
                           constrained_layout=True)
    ax.set_facecolor("#0f0f1a")
    im = ax.imshow(data_2d, cmap=cmap, aspect="equal", vmin=0, vmax=1)
    ax.set_title(title, fontsize=11, color="lightgrey")
    ax.set_xlabel("qubit column", color="lightgrey")
    ax.set_ylabel("qubit row",    color="lightgrey")
    ax.tick_params(colors="lightgrey")
    plt.colorbar(im, ax=ax, fraction=0.046)
    return fig


class PyVistaViewer:
    """Real-time 3D (or 2D fallback) viewer of the spatial noise field."""

    def __init__(self, d: int = 3):
        self.d = d
        self.N = d * d
        self._last_state = None

        # Simple mode selector (works with or without Panel)
        class _Sel:
            def __init__(self_, val):
                self_.value = val
        self._mode_sel = _Sel("lambda")

        if _PANEL_OK:
            self._build_panel_ui()
        else:
            self.panel = None

    def _build_panel_ui(self):
        if _PYVISTA_OK:
            self._init_pyvista()
            self._pane = pn.pane.VTK(
                self._plotter.ren_win, height=420, width=460, sizing_mode="fixed")
        else:
            self._pane = pn.pane.Matplotlib(
                _make_heatmap(np.zeros((self.d, self.d)), "λ field"),
                height=420, width=460)

        self._mode_sel_widget = pn.widgets.RadioButtonGroup(
            name="View mode", options=["lambda", "prob", "qsp"],
            value="lambda", button_type="default", sizing_mode="stretch_width")
        self._mode_sel_widget.param.watch(self._on_mode_change, "value")

        self.panel = pn.Column(
            pn.pane.Markdown("### 3D Noise Field"),
            self._mode_sel_widget,
            self._pane)

    def _init_pyvista(self):
        pv.global_theme.background = "#1a1a2e"
        self._grid = pv.ImageData()
        self._grid.dimensions = (self.d + 1, self.d + 1, 2)
        self._grid.spacing = (1.0, 1.0, 0.5)
        self._grid.origin  = (0.0, 0.0, 0.0)
        self._grid.cell_data["field"] = np.zeros(self.N)
        self._plotter = pv.Plotter(off_screen=True, window_size=(460, 420))
        self._plotter.add_mesh(
            self._grid, scalars="field", cmap="plasma",
            clim=[0.0, 1.0], show_edges=True, edge_color="#333355")
        self._plotter.view_isometric()
        self._plotter.reset_camera()
        self._plotter.add_axes()

    # ── Public API ────────────────────────────────────────────────────────
    def update(self, state: dict):
        new_d = state.get("d", self.d)
        if new_d != self.d:
            self.d = new_d
            self.N = new_d * new_d
            if _PYVISTA_OK:
                self._init_pyvista()

        self._last_state = state
        self._render(self._extract_field(state))

    def _extract_field(self, state: dict) -> np.ndarray:
        mode = self._mode_sel.value
        lf   = np.array(state.get("lambda_field", np.zeros((self.N, 3))),
                        dtype=float)

        # Ensure 2-D (N, 3)
        if lf.ndim == 1:
            lf = lf.reshape(-1, 3) if (lf.size % 3 == 0) else np.zeros((self.N, 3))

        if mode == "lambda":
            raw = np.linalg.norm(lf, axis=1)
        elif mode == "prob":
            raw = np.asarray(
                state.get("probabilities", np.zeros(self.N)), dtype=float)
        else:  # qsp
            raw = np.abs(lf[:, 2]) if lf.shape[1] > 2 else np.zeros(self.N)

        raw = np.asarray(raw, dtype=float).flatten()
        raw = np.where(np.isfinite(raw), raw, 0.0)
        if len(raw) != self.N:
            raw = np.resize(raw, self.N)

        lo, hi = raw.min(), raw.max()
        return (raw - lo) / (hi - lo) if (hi - lo) > 1e-10 else np.zeros(self.N)

    def _render(self, field: np.ndarray):
        if not _PANEL_OK:
            return
        step  = (self._last_state or {}).get("step", 0)
        title = f"Mode: {self._mode_sel.value}  | step {step}"
        if _PYVISTA_OK:
            self._grid.cell_data["field"] = field
            self._plotter.update_scalars(field, mesh=self._grid)
            self._plotter.render()
            self._pane.object = self._plotter.ren_win
        else:
            fig = _make_heatmap(field.reshape(self.d, self.d), title)
            self._pane.object = fig
            plt.close(fig)

    def _on_mode_change(self, event):
        self._mode_sel.value = getattr(event, "new", event)
        if self._last_state is not None:
            self.update(self._last_state)
