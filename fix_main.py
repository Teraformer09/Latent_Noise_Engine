import re

with open(r'C:\Users\ncclab\Downloads\LatentNoiseEngine_frontend_v2\LatentNoiseEngine\api\main.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Fix the import
content = content.replace(
    'from latent_core.engine import LatentNoiseEngine  # type: ignore',
    'from frontend.simulator_adapter import NoiseSimulator as LatentNoiseEngine  # type: ignore'
)

# 2. Fix the instantiation before the loop
content = content.replace(
    'real_engine = real_engine_cls(**cfg) if real_engine_cls else None',
    'real_engine = real_engine_cls(config=cfg) if real_engine_cls else None'
)

# 3. Fix the instantiation inside the loop (re-init)
content = content.replace(
    'real_engine = real_engine_cls(**cfg)',
    'real_engine = real_engine_cls(config=cfg)'
)

# 4. Inject physics telemetry into step
old_step = """        # --- compute state ---
        try:
            if real_engine is not None:
                raw = real_engine.step()
                state = _coerce_state(raw, step)"""

new_step = """        # --- compute state ---
        try:
            if real_engine is not None:
                raw = real_engine.step(force=True)
                
                from api.physics_telemetry import compute_eigenvalues, compute_psd
                import numpy as np
                lf = raw.get("lambda_field", [])
                if len(lf) > 0:
                    l0 = lf[0]
                    H = np.array([[l0[2], l0[0] - 1j*l0[1]], [l0[0] + 1j*l0[1], -l0[2]]], dtype=complex)
                    raw["eigenvalues"] = compute_eigenvalues(H)
                else:
                    raw["eigenvalues"] = [0.0, 0.0]
                
                raw.setdefault("psd_freqs", [0.0, 1.0])
                raw.setdefault("psd_amps", [0.1, 0.2])
                raw.setdefault("state_vector", [[1.0, 0.0], [0.0, 0.0]])
                
                state = _coerce_state(raw, step)"""

content = content.replace(old_step, new_step)

with open(r'C:\Users\ncclab\Downloads\LatentNoiseEngine_frontend_v2\LatentNoiseEngine\api\main.py', 'w', encoding='utf-8') as f:
    f.write(content)
