# Latent Noise Engine — Diagnostic Frontend

A research-grade, local, real-time diagnostic interface for the stochastic
quantum noise simulation pipeline.

```
λ(t) → QSP transform → Pauli probabilities → QEC decoder → hazard(t)
```

---

## Structure

```
LatentNoise-Core/
├── latent_core/           ← physics backend (do NOT modify)
├── latent_noise/          ← simulation engine (do NOT modify)
├── frontend/
│   ├── app.py             ← main entrypoint
│   ├── simulator_adapter.py   ← thread-safe backend bridge
│   ├── requirements.txt
│   ├── viewer/
│   │   └── pyvista_view.py    ← 3D noise field (λ / prob / QSP modes)
│   ├── controls/
│   │   └── config_panel.py    ← parameter sliders
│   ├── plots/
│   │   ├── time_series.py     ← hazard(t) + alpha(t)
│   │   ├── stats.py           ← histogram + ACF diagnostics
│   │   └── qec_view.py        ← QEC logical error rates (d=3,5,7)
│   ├── experiment/
│   │   ├── manager.py         ← batch runs, CSV/JSON export
│   │   └── panel_ui.py        ← experiment UI panel
│   └── tests/
│       └── run_tests.py       ← 210-test self-contained suite
```

---

## Setup

### 1. Install dependencies

```bash
cd LatentNoise-Core
pip install -r frontend/requirements.txt
```

### 2. Run the frontend

```bash
python frontend/app.py
```

Opens at **http://localhost:5006**

---

## Running Tests

### Full suite (210 tests, ~2s)

```bash
python frontend/tests/run_tests.py
```

### Fast mode (skips slow stress tests, ~1s)

```bash
python frontend/tests/run_tests.py --fast
```

### Verbose (show all test names)

```bash
python frontend/tests/run_tests.py --verbose
```

Expected output:
```
══════════════════════════════════════════════════════════════════════
  Latent Noise Engine — Frontend Test Suite  (200+ tests)
══════════════════════════════════════════════════════════════════════
...
──────────────────────────────────────────────────────────────────────
  Total:210  PASS:210  FAIL:0  SKIP:0  Time:1.5s
──────────────────────────────────────────────────────────────────────
  ✅  ALL TESTS PASSED — deployment ready
```

---

## Architecture

```
Backend thread          UI thread (Panel @ 50ms)
──────────────          ────────────────────────
simulator.step()   →    state_queue.get()
     ↓                       ↓
state_queue.put()   →   viewer.update(state)
                        time_plot.update(state)
                        stats.update(state)
                        qec_view.update(state)
```

**Critical rules:**
- `simulator.step()` is ONLY called from the background thread
- UI thread ONLY reads from the queue — never touches the simulator directly
- All parameter updates use `simulator.update_params()` which is lock-protected

---

## Controls

| Slider | Effect |
|--------|--------|
| Base alpha | QSP signal strength (PI controller base) |
| Sigma temporal | Noise field intensity |
| Burst probability | Non-Gaussian burst events |
| Target hazard | PI controller setpoint |
| QEC distance | Primary d=3/5/7 selection |
| QSP toggle | Enable/disable QSP transformation |

---

## 3D View Modes

| Mode | Shows |
|------|-------|
| `lambda` | Raw λ vector magnitude per qubit |
| `prob` | Per-qubit Pauli-Z error probability |
| `qsp` | QSP-transformed λ_Z component |

---

## Experiment Manager

1. Set parameters via control sliders
2. Click **▶ Run Batch**
3. Choose N runs and steps per run
4. Click **Export CSV** or **Export JSON**

Results saved to `results/` directory.

---

## Fallback Behaviour

| Missing dependency | Fallback |
|--------------------|----------|
| `pyvista` | matplotlib heatmap (2D) |
| `panel` | headless mode (tests still pass) |
| backend unavailable | mock simulator with realistic dynamics |

---

## Test Coverage

| Category | Tests | Description |
|----------|-------|-------------|
| State flow integrity | 25 | Queue, fields, types, NaN |
| 3D view correctness | 30 | Modes, shapes, normalisation |
| Control accuracy | 25 | Params, clamping, thread safety |
| Time series | 25 | Bounds, NaN, oscillation |
| Statistics | 20 | ACF, histogram, mean/var |
| QEC visualisation | 20 | Distance ordering, heatmap |
| Experiment | 20 | Save/load, CSV/JSON, seeds |
| Performance | 20 | Memory, threads, concurrency |
| System causality | 25 | Pipeline, QSP effect, control |
| **Total** | **210** | **All pass** |
