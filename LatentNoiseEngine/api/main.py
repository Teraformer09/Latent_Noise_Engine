import jax; jax.config.update("jax_platform_name", "cpu")  # noqa: E402 — must be first

import asyncio
import logging
import threading
from typing import Any, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api.models import QSPParams, NoiseParams, QECParams, SimParams
from api.worker import simulation_worker

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("latent_noise_api")

# ---------------------------------------------------------------------------
# Redis availability probe
# ---------------------------------------------------------------------------
_REDIS_AVAILABLE = False
try:
    import redis as _redis_sync
    _probe = _redis_sync.Redis(host="localhost", port=6379)
    _probe.ping()
    _REDIS_AVAILABLE = True
    _probe.close()
    logger.info("Redis is available.")
except Exception:
    logger.warning("Redis unavailable.")

REDIS_CHANNEL = "sim_telemetry"


# ---------------------------------------------------------------------------
# SimulationManager  (threading-based — avoids Windows multiprocessing/asyncio issues)
# ---------------------------------------------------------------------------
class _ThreadSafeDict:
    """Minimal dict-like proxy backed by a threading.Lock."""

    def __init__(self, data: dict):
        self._lock = threading.Lock()
        self._data = dict(data)

    def get(self, key, default=None):
        with self._lock:
            return self._data.get(key, default)

    def __getitem__(self, key):
        with self._lock:
            return self._data[key]

    def __setitem__(self, key, value):
        with self._lock:
            self._data[key] = value

    def __iter__(self):
        with self._lock:
            return iter(dict(self._data))

    def __eq__(self, other):
        with self._lock:
            return self._data == other

    def keys(self):
        with self._lock:
            return list(self._data.keys())

    def items(self):
        with self._lock:
            return list(self._data.items())


class SimulationManager:
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._shared_cfg = _ThreadSafeDict({
            "degree": 3, "phi_vector": None, "target_function": "sign",
            "rescaling_factor": 1.0, "noise_type": "ornstein_uhlenbeck",
            "tau_corr": 0.05, "xi_spatial": 1.5, "beta_exponent": 1.0,
            "burst_amplitude": 0.5, "burst_prob": 0.01, "distance": 3,
            "p_measure": 0.0, "kp": 10.0, "ki": 2.0, "kd": 0.0,
            "target_hazard": 0.1, "dt": 0.05, "use_qsp": True, "d": 3,
        })
        self._shared_status = _ThreadSafeDict({
            "running": False, "step": 0, "last_frame": None, "reinit_requested": False
        })

    @property
    def is_running(self) -> bool:
        return bool(self._shared_status.get("running", False))

    @property
    def step(self) -> int:
        return int(self._shared_status.get("step", 0))

    def get_config(self) -> dict:
        return dict(self._shared_cfg._data)

    def apply_sim_params(self, params: SimParams) -> dict:
        old_d = int(self._shared_cfg.get("d", 3))
        old_degree = int(self._shared_cfg.get("degree", 3))
        needs_reinit = False
        if params.qsp:
            q = params.qsp
            self._shared_cfg["degree"] = q.degree
            self._shared_cfg["phi_vector"] = q.phi_vector
            self._shared_cfg["target_function"] = q.target_function.value
            self._shared_cfg["rescaling_factor"] = q.rescaling_factor
            if q.degree != old_degree:
                needs_reinit = True
        if params.noise:
            n = params.noise
            self._shared_cfg["noise_type"] = n.noise_type.value
            self._shared_cfg["tau_corr"] = n.tau_corr
            self._shared_cfg["xi_spatial"] = n.xi_spatial
            self._shared_cfg["beta_exponent"] = n.beta_exponent
            self._shared_cfg["burst_amplitude"] = n.burst_amplitude
            self._shared_cfg["burst_prob"] = n.burst_prob
        if params.qec:
            c = params.qec
            self._shared_cfg["distance"] = c.distance
            self._shared_cfg["d"] = c.distance
            self._shared_cfg["p_measure"] = c.p_measure
            self._shared_cfg["kp"] = c.kp
            self._shared_cfg["ki"] = c.ki
            self._shared_cfg["kd"] = c.kd
            self._shared_cfg["target_hazard"] = c.target_hazard
            if c.distance != old_d:
                needs_reinit = True
        if needs_reinit:
            self._shared_status["reinit_requested"] = True
        return dict(self._shared_cfg._data)

    def start(self) -> None:
        if self.is_running:
            return
        self._shared_status["running"] = True
        self._shared_status["step"] = 0
        self._thread = threading.Thread(
            target=simulation_worker,
            args=(self._shared_cfg, self._shared_status, _REDIS_AVAILABLE),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._shared_status["running"] = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._thread = None

    def pop_last_frame(self) -> Optional[bytes]:
        f = self._shared_status.get("last_frame")
        if f is not None:
            self._shared_status["last_frame"] = None
        return f

# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
app = FastAPI()
_ALLOWED_ORIGINS = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
sim_manager = SimulationManager()

class Broadcaster:
    def __init__(self): self.queues: Set[asyncio.Queue] = set()
    def register(self): q = asyncio.Queue(maxsize=10); self.queues.add(q); return q
    def unregister(self, q): self.queues.discard(q)
    def broadcast(self, frame):
        for q in self.queues:
            if q.full(): q.get_nowait()
            q.put_nowait(frame)

broadcaster = Broadcaster()

async def frame_poller():
    while True:
        try:
            f = sim_manager.pop_last_frame()
            if f is not None:
                broadcaster.broadcast(f)
        except Exception as exc:
            logger.debug("Frame poller error: %s", exc)
        await asyncio.sleep(0.01)

@app.on_event("startup")
async def on_startup():
    if not _REDIS_AVAILABLE:
        asyncio.create_task(frame_poller())

@app.on_event("shutdown")
async def on_shutdown():
    sim_manager.stop()

@app.get("/")
async def root(): return {"status": "ok", "running": sim_manager.is_running}
@app.get("/config")
async def get_config(): return sim_manager.get_config()
@app.post("/config/params")
async def post_sim_params(params: SimParams): return {"updated": True, "config": sim_manager.apply_sim_params(params)}
@app.post("/start")
async def start_sim(): sim_manager.start(); return {"started": True}
@app.post("/stop")
async def stop_sim(): sim_manager.stop(); return {"stopped": True}
@app.get("/status")
async def status(): return {"running": sim_manager.is_running, "step": sim_manager.step}

@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    if _REDIS_AVAILABLE:
        import redis.asyncio as aioredis
        r = aioredis.Redis(host="localhost", port=6379)
        pubsub = r.pubsub()
        await pubsub.subscribe(REDIS_CHANNEL)
        try:
            async for m in pubsub.listen():
                if m["type"] == "message":
                    await websocket.send_bytes(m["data"])
        except (WebSocketDisconnect, Exception) as exc:
            logger.debug("Redis WS closed: %s", exc)
        finally:
            await pubsub.unsubscribe(REDIS_CHANNEL)
            await r.aclose()
    else:
        q = broadcaster.register()
        try:
            while True:
                f = await q.get()
                await websocket.send_bytes(f)
        except (WebSocketDisconnect, Exception) as exc:
            logger.debug("WS closed: %s", exc)
        finally:
            broadcaster.unregister(q)
