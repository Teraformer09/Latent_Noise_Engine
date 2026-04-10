# Latent Noise Engine

A real-time quantum noise simulation and visualization platform. It models non-Markovian noise on a surface-code lattice, applies Quantum Signal Processing (QSP) to transform the effective Hamiltonian, runs a minimum-weight perfect matching (MWPM) decoder, and streams live telemetry to a browser-based dashboard.

---

## Architecture

```
LatentNoiseEngine_frontend_v2/
├── LatentNoiseEngine/          # Python backend
│   ├── api/                    # FastAPI server + WebSocket stream
│   │   ├── main.py             # App, SimulationManager, Broadcaster
│   │   ├── worker.py           # Simulation thread (noise → QSP → QEC → telemetry)
│   │   ├── models.py           # Pydantic request/response models
│   │   ├── physics_telemetry.py# PSD computation, eigenvalue utils, mock state
│   │   └── requirements.txt
│   ├── latent_core/            # Core physics engine (JAX-based)
│   │   ├── qsp/                # QSP phase solver, circuit, polynomial
│   │   ├── linalg/             # Pauli operators, matrix exponentials
│   │   ├── signal/             # FFT, PSD, autocorrelation, filters
│   │   ├── stochastic/         # Fractional Gaussian noise, ARFIMA
│   │   └── quantum/            # Hamiltonian evolution, Lindblad
│   ├── latent_noise/           # Surface-code noise pipeline
│   │   ├── noise/              # Ornstein-Uhlenbeck spatial noise field
│   │   ├── decoder/            # MWPM decoder, logical Z operator
│   │   ├── stim_layer/         # Stim circuit builder and sampler
│   │   └── control/            # Adaptive PID hazard controller
│   ├── frontend/
│   │   └── simulator_adapter.py# NoiseSimulator — bridges backend to API worker
│   └── configs/                # YAML configuration files
└── frontend-ui/                # React/TypeScript dashboard
    └── src/
        ├── components/
        │   ├── ControlPanel.tsx    # Parameter sliders, start/stop, export CSV
        │   ├── Lattice3D.tsx       # Three.js surface-code lattice
        │   ├── BlochSphere.tsx     # Bloch sphere state visualization
        │   ├── EigenvaluePanel.tsx # QSP Hamiltonian spectrum bar chart
        │   ├── FidelityPanel.tsx   # Hazard & alpha time-series chart
        │   ├── BottomPanels.tsx    # Noise PSD, autocorrelation, threshold gauge
        │   ├── MemoryHeatmap.tsx   # Qubit error correlation heatmap
        │   └── WebSocketManager.tsx# msgpack WebSocket consumer
        └── store/
            ├── useSimulationStore.ts  # Live telemetry state + CSV export
            └── useConfigStore.ts      # QSP/noise/QEC parameter store
```

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.10+ |
| Node.js | 18+ |
| npm | 9+ |

Python dependencies (from `LatentNoiseEngine/api/requirements.txt`):

```
fastapi, uvicorn, msgpack, numpy, jax, jaxlib, scipy, pymatching, pyyaml, pydantic, redis (optional)
```

---

## Setup

### 1. Backend

```bash
cd LatentNoiseEngine_frontend_v2/LatentNoiseEngine

pip install -r api/requirements.txt
```

### 2. Frontend

```bash
cd LatentNoiseEngine_frontend_v2/frontend-ui

npm install
```

---

## Running

Open two terminals.

**Terminal 1 — Backend:**

```bash
cd LatentNoiseEngine_frontend_v2/LatentNoiseEngine

python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — Frontend:**

```bash
cd LatentNoiseEngine_frontend_v2/frontend-ui

npm run dev
```

Then open **http://localhost:5173** in your browser.

---

## Usage

1. **Start the simulation** — click `▶ START` in the control panel.
2. **Adjust parameters** — sliders update live; click `⚡ COMMIT & SYNC` to push changes to the running engine.
3. **Stop** — click `■ STOP`.
4. **Export data** — click `↓ EXPORT CSV` to download a timestamped CSV of all recorded frames.

### Control Panel Sections

| Section | Parameters |
|---------|-----------|
| **QSP Controls** | Polynomial degree, target function (sign/step/linear), rescaling factor α, custom φ phase vector |
| **Noise Dynamics** | Noise type (Ornstein-Uhlenbeck / Flicker / White), temporal correlation τ, spatial correlation ξ, burst probability |
| **QEC & Controller** | Surface code distance d (3/5/7/9/11), PID gain Kp, target hazard rate |

---

## Dashboard Panels

| Panel | Description |
|-------|-------------|
| **Lattice 3D** | Surface-code qubit lattice coloured by per-qubit error probability |
| **Bloch Sphere** | State vector visualized on the Bloch sphere |
| **Eigenvalue Spectrum** | Sorted eigenvalues of the QSP-effective Hamiltonian |
| **Fidelity Monitor** | Time-series of logical hazard rate and adaptive gain α |
| **Noise PSD** | Power spectral density of the noise field (log-log) |
| **Autocorrelation** | Empirical autocorrelation function of the PSD amplitudes |
| **Threshold Gauge** | Radial gauge showing current hazard vs the 10% critical threshold |
| **Memory Heatmap** | Qubit error correlation matrix weighted by geometric mean of error probs |

---

## Telemetry Wire Format

Frames are sent over WebSocket as **msgpack**-encoded dicts at each simulation step (`dt = 0.05 s` default):

| Field | Type | Description |
|-------|------|-------------|
| `step` | int | Simulation step index |
| `hazard` | float | Current logical error rate |
| `alpha` | float | Adaptive controller gain |
| `d` | int | Active surface code distance |
| `probabilities` | float[] | Per-qubit Z-error probabilities (length d²) |
| `lambda_field` | float[][] | Per-qubit noise vector [λx, λy, λz] (shape d²×3) |
| `pauli_probs` | object | Mean Pauli channel probs `{px, py, pz, pi}` |
| `eigenvalues` | float[] | Sorted eigenvalues of QSP-effective Hamiltonian |
| `psd_freqs` | float[] | Frequency bins (Hz) |
| `psd_amps` | float[] | PSD amplitudes |
| `state_vector` | float[][] | Qubit state as `[[re, im], [re, im]]` |

---

## Export CSV

Click `↓ EXPORT CSV` in the control panel footer at any time. The downloaded file contains one row per telemetry frame:

```
step,hazard,alpha,px,py,pz,pi,eig0,eig1
0,0.0412,1.0,0.0018,0.0085,0.0347,0.9551,-0.60,0.60
1,0.0389,1.02,...
```

Up to 5000 rows are retained in memory during a session.

---

## Configuration

Default simulation parameters live in `LatentNoiseEngine/configs/default.yaml`. Key fields:

```yaml
time:
  dt: 0.01          # physical time step
  horizon: 1000     # total simulation steps

noise:
  type: fgn         # fractional Gaussian noise (hurst: 0.7)

qsp:
  method: optimization
  n_iter: 500
  lr: 0.01

metrics:
  hazard_crit: 0.3  # critical hazard threshold
```

Runtime parameters (degree, noise type, distance, PID gains) can be changed live via the control panel without restarting.

---

## Optional: Redis

If a Redis server is running on `localhost:6379`, the backend publishes telemetry frames to the `sim_telemetry` channel and the WebSocket endpoint subscribes via pub/sub instead of the in-process broadcaster. This is transparent to the frontend.

```bash
# Windows (scoop)
scoop install redis
redis-server

# or via Docker
docker run -p 6379:6379 redis:alpine
```

---

## Project Structure Notes

- **`api/worker.py`** runs the simulation in a `threading.Thread` (not a subprocess) to avoid Windows multiprocessing spawn issues with JAX and asyncio.
- **`frontend/simulator_adapter.py`** (`NoiseSimulator`) is the bridge between the physics backend packages and the API worker. It handles lazy decoder initialization, live parameter updates, and graceful fallback to a mock state when backend imports fail.
- **`api/physics_telemetry.py`** provides `compute_psd()` and `build_mock_state()` used both by the worker and in tests.
