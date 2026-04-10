import { create } from "zustand";

// ---------------------------------------------------------------------------
// Domain interfaces
// ---------------------------------------------------------------------------

export interface QSPConfig {
  degree: number;
  phiVector: number[] | null;
  targetFunction: "sign" | "step" | "linear" | "inversion";
  rescalingFactor: number;
}

export interface NoiseConfig {
  noiseType: "ornstein_uhlenbeck" | "flicker" | "white";
  tauCorr: number;
  xiSpatial: number;
  betaExponent: number;
  burstAmplitude: number;
  burstProb: number;
}

export interface QECConfig {
  distance: 3 | 5 | 7 | 9 | 11;
  pMeasure: number;
  kp: number;
  ki: number;
  kd: number;
  targetHazard: number;
}

// ---------------------------------------------------------------------------
// Store interface
// ---------------------------------------------------------------------------

interface ConfigStore {
  qsp: QSPConfig;
  noise: NoiseConfig;
  qec: QECConfig;
  isDirty: boolean;
  isSyncing: boolean;
  lastSyncError: string | null;

  updateQSP: (partial: Partial<QSPConfig>) => void;
  updateNoise: (partial: Partial<NoiseConfig>) => void;
  updateQEC: (partial: Partial<QECConfig>) => void;
  pushToEngine: () => Promise<void>;
  setPhiFromJSON: (json: string) => void;
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

const defaultQSP: QSPConfig = {
  degree: 3,
  phiVector: null,
  targetFunction: "sign",
  rescalingFactor: 1.0,
};

const defaultNoise: NoiseConfig = {
  noiseType: "ornstein_uhlenbeck",
  tauCorr: 0.05,
  xiSpatial: 1.5,
  betaExponent: 1.0,
  burstAmplitude: 0.5,
  burstProb: 0.01,
};

const defaultQEC: QECConfig = {
  distance: 3,
  pMeasure: 0.0,
  kp: 10.0,
  ki: 2.0,
  kd: 0.0,
  targetHazard: 0.1,
};

const API_BASE = (import.meta as any).env?.VITE_API_BASE ?? "http://localhost:8000";

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

export const useConfigStore = create<ConfigStore>()((set, get) => ({
  qsp: { ...defaultQSP },
  noise: { ...defaultNoise },
  qec: { ...defaultQEC },
  isDirty: false,
  isSyncing: false,
  lastSyncError: null,

  updateQSP: (partial) =>
    set((state) => ({
      qsp: { ...state.qsp, ...partial },
      isDirty: true,
    })),

  updateNoise: (partial) =>
    set((state) => ({
      noise: { ...state.noise, ...partial },
      isDirty: true,
    })),

  updateQEC: (partial) =>
    set((state) => ({
      qec: { ...state.qec, ...partial },
      isDirty: true,
    })),

  pushToEngine: async () => {
    const { qsp, noise, qec } = get();

    set({ isSyncing: true });

    const payload = {
      qsp: {
        degree: qsp.degree,
        phi_vector: qsp.phiVector,
        target_function: qsp.targetFunction,
        rescaling_factor: qsp.rescalingFactor,
      },
      noise: {
        noise_type: noise.noiseType,
        tau_corr: noise.tauCorr,
        xi_spatial: noise.xiSpatial,
        beta_exponent: noise.betaExponent,
        burst_amplitude: noise.burstAmplitude,
        burst_prob: noise.burstProb,
      },
      qec: {
        distance: qec.distance,
        p_measure: qec.pMeasure,
        kp: qec.kp,
        ki: qec.ki,
        kd: qec.kd,
        target_hazard: qec.targetHazard,
      },
    };

    try {
      const res = await fetch(`${API_BASE}/config/params`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`HTTP ${res.status}: ${text}`);
      }

      set({ isDirty: false, isSyncing: false, lastSyncError: null });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      set({ isSyncing: false, lastSyncError: msg });
    }
  },

  setPhiFromJSON: (json) => {
    let parsed: unknown;
    try {
      parsed = JSON.parse(json);
    } catch {
      set({ lastSyncError: "Invalid phi JSON: must be an array of numbers" });
      return;
    }

    if (
      !Array.isArray(parsed) ||
      parsed.some((v) => typeof v !== "number")
    ) {
      set({ lastSyncError: "Invalid phi JSON: must be an array of numbers" });
      return;
    }

    set((state) => ({
      qsp: { ...state.qsp, phiVector: parsed as number[] },
      isDirty: true,
      lastSyncError: null,
    }));
  },
}));
