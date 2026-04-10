import numpy as np

class PhysicalNoiseModel:
    def __init__(self, dim=3, sigma_v=0.01, sigma_zeta=0.05, phi=0.95, seed=None):
        self.dim = dim
        self.sigma_v = sigma_v
        self.sigma_zeta = sigma_zeta
        self.phi = phi
        self.rng = np.random.default_rng(seed)
        self.drift = np.zeros(dim)
        self.zeta = np.zeros(dim)

    def step(self):
        self.drift += self.rng.normal(0, self.sigma_v, size=self.dim)
        self.zeta = self.phi * self.zeta + self.rng.normal(0, self.sigma_zeta, size=self.dim)
        return (self.drift + self.zeta).copy()

class SpatialNoiseField:
    """
    Implements a 3D spatially structured noise field λ(x, y, t) ∈ R^3.
    """
    def __init__(self, coords, sigma_spatial=1.5, theta=0.95, sigma_temporal=0.05, burst_prob=0.01, seed=None):
        self.coords = coords
        self.N = len(coords)
        self.theta = theta
        self.sigma_temporal = sigma_temporal
        self.burst_prob = burst_prob
        self.rng = np.random.default_rng(seed)
        
        # λ(x,y,t) is now (N, 3) representing [λx, λy, λz]
        self.field = np.zeros((self.N, 3))
        self.W = self._build_spatial_kernel(coords, sigma_spatial)

    def _build_spatial_kernel(self, coords, sigma):
        N = len(coords)
        W = np.zeros((N, N))
        for i, (x1, y1) in enumerate(coords):
            for j, (x2, y2) in enumerate(coords):
                dist2 = (x1 - x2)**2 + (y1 - y2)**2
                W[i, j] = np.exp(-dist2 / (2 * sigma**2))
        return W / W.sum(axis=1, keepdims=True)

    def step(self):
        # 1. Temporal white noise for 3 axes
        noise = self.rng.normal(0, self.sigma_temporal, size=(self.N, 3))

        # 2. Apply spatial correlation across the grid
        correlated_noise = self.W @ noise

        # 3. OU drift
        self.field = self.theta * self.field + correlated_noise

        # 4. Non-Gaussian bursts
        if self.rng.random() < self.burst_prob:
            self.field += self.rng.standard_cauchy(size=(self.N, 3)) * 0.2

        return self.field.copy()
