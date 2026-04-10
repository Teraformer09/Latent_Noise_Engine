import numpy as np
import sys
import os
import jax.numpy as jnp

# Bridge to latent_core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from latent_core.qsp.transform import qsp_effective_hamiltonian
from latent_core.linalg.operators import X, Y, Z

from latent_noise.noise.ou_process import SpatialNoiseField
from latent_noise.noise.mapping import lambda_to_pauli_probs
from latent_noise.control.controller import AdaptiveController
from latent_noise.control.hazard import compute_hazard
from latent_noise.stim_layer.sampler import sample_surface_code_scaling, decode_and_check_logical_scaling
from latent_noise.decoder.matching import build_decoder_for_distance


class LatentNoisePipeline:
    def __init__(self, config):
        # 1. Define Geometry
        self.d = config.get("d", 3)
        self.coords = [(i, j) for i in range(self.d) for j in range(self.d)]
        self.N = len(self.coords)
        
        # 2. Spatial Noise Field
        self.noise_field = SpatialNoiseField(
            coords=self.coords,
            sigma_spatial=config.get("sigma_spatial", 1.5),
            theta=config.get("theta_noise", 0.95),
            sigma_temporal=config.get("sigma_temporal", 0.05),
            burst_prob=config.get("burst_prob", 0.01),
            seed=config.get("seed", 0)
        )
        
        # 3. Adaptive Controller (PI)
        self.ctrl = AdaptiveController(
            base_alpha=config.get("base_alpha", 1.0),
            kp=config.get("kp", 10.0),
            ki=config.get("ki", 2.0),
            target=config.get("target_hazard", 0.1)
        )
        
        self.matching = build_decoder_for_distance(self.d)
        self.steps = config.get("steps", 100)
        self.gamma = config.get("gamma_corr", 0.3)
        self.shots = config.get("shots", 1000)
        
        self.use_qsp = config.get("use_qsp", True)
        # Default: degree-1 identity polynomial; override via config "qsp_coeffs"
        degree = config.get("degree", 3)
        default_coeffs = [0.0] * degree + [1.0]  # Chebyshev T_d, degree=d
        self.qsp_coeffs = jnp.array(config.get("qsp_coeffs", default_coeffs))
        
        self.current_alpha = config.get("base_alpha", 1.0)
        self.PAULIS_3D = [X, Y, Z]

    def run(self):
        hazards = []
        alphas = []

        for t in range(self.steps):
            # 1. Evolve Spatial Noise Field
            lambda_field = self.noise_field.step() # (N, 3)

            # 2. QSP Transformation
            probs = []
            for i in range(self.N):
                H_i = 0
                for j in range(3):
                    H_i += lambda_field[i, j] * self.PAULIS_3D[j]
                
                if self.use_qsp:
                    H_eff = qsp_effective_hamiltonian(H_i, self.qsp_coeffs)
                else:
                    H_eff = H_i
                
                # Spectral mapping: scale each axis by the spectral norm
                evals = np.linalg.eigvalsh(H_eff)
                spectral_scale = float(np.max(np.abs(evals))) / (np.linalg.norm(lambda_field[i]) + 1e-12)
                l_vec_scaled = lambda_field[i] * spectral_scale

                # 3. Stable Mapping — use all three Pauli channels
                p_dict = lambda_to_pauli_probs(l_vec_scaled, scale=self.current_alpha)
                # Combined error probability: sum of non-identity channels
                p_err = p_dict['px'] + p_dict['py'] + p_dict['pz']
                probs.append(float(np.clip(p_err, 0.0, 1.0)))

            # 4. Scaling Simulation
            errors, syndromes = sample_surface_code_scaling(
                d=self.d,
                probs=np.array(probs),
                coords=self.coords,
                gamma=self.gamma,
                shots=self.shots
            )
            
            # 5. Decode
            logical_errors = decode_and_check_logical_scaling(self.d, errors, syndromes, self.matching)
            h = compute_hazard(logical_errors)

            # 6. Control Update
            self.current_alpha = self.ctrl.update(h)

            hazards.append(h)
            alphas.append(self.current_alpha)

        return {"hazards": hazards, "alphas": alphas}
