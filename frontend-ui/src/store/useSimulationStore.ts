import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";

interface PauliProbs {
  px: number;
  py: number;
  pz: number;
  pi: number;
}

interface HistoryRow {
  step: number;
  hazard: number;
  alpha: number;
  px: number;
  py: number;
  pz: number;
  pi: number;
  eig0: number | null;
  eig1: number | null;
}

interface SimulationState {
  step: number;
  hazard: number;
  alpha: number;
  d: number;                 // active surface code distance from last telemetry frame
  probabilities: number[];
  pauliProbs: PauliProbs;
  eigenvalues: number[];
  psdFreqs: number[];
  psdAmps: number[];
  /** Each element is [re, im] */
  stateVector: number[][];
  qecMetrics: { [d: number]: number };
  isConnected: boolean;
  isRunning: boolean;
  history: HistoryRow[];

  // Actions
  updateFromTelemetry: (data: Record<string, unknown>) => void;
  setConnected: (v: boolean) => void;
  setRunning: (v: boolean) => void;
  startStatusPolling: (apiBase: string) => () => void;
  exportCSV: () => void;
}

export const useSimulationStore = create<SimulationState>()(
  subscribeWithSelector((set) => ({
    step: 0,
    hazard: 0,
    alpha: 0,
    d: 3,
    probabilities: [],
    pauliProbs: { px: 0.25, py: 0.25, pz: 0.25, pi: 0.25 },
    eigenvalues: [],
    psdFreqs: [],
    psdAmps: [],
    stateVector: [[1, 0], [0, 0]],
    qecMetrics: {},
    isConnected: false,
    isRunning: false,
    history: [],

    updateFromTelemetry: (data) => {
      const raw = data as Record<string, unknown>;

      const pauliRaw = raw["pauli_probs"] as Record<string, number> | undefined;
      const pauliProbs: PauliProbs = pauliRaw
        ? {
            px: Number(pauliRaw["px"] ?? 0.25),
            py: Number(pauliRaw["py"] ?? 0.25),
            pz: Number(pauliRaw["pz"] ?? 0.25),
            pi: Number(pauliRaw["pi"] ?? 0.25),
          }
        : { px: 0.25, py: 0.25, pz: 0.25, pi: 0.25 };

      const qecRaw = raw["qec_metrics"] as Record<string, number> | undefined;
      const qecMetrics: { [d: number]: number } = {};
      if (qecRaw) {
        for (const [k, v] of Object.entries(qecRaw)) {
          qecMetrics[Number(k)] = Number(v);
        }
      }

      const probabilities = Array.isArray(raw["probabilities"])
        ? (raw["probabilities"] as number[])
        : [];

      const eigenvalues = Array.isArray(raw["eigenvalues"])
        ? (raw["eigenvalues"] as number[])
        : [];

      const psdFreqs = Array.isArray(raw["psd_freqs"])
        ? (raw["psd_freqs"] as number[])
        : [];

      const psdAmps = Array.isArray(raw["psd_amps"])
        ? (raw["psd_amps"] as number[])
        : [];

      const stateVector = Array.isArray(raw["state_vector"])
        ? (raw["state_vector"] as number[][])
        : [[1, 0], [0, 0]];

      const newStep = Number(raw["step"] ?? 0);
      const newHazard = Number(raw["hazard"] ?? 0);
      const newAlpha = Number(raw["alpha"] ?? 0);

      const row: HistoryRow = {
        step: newStep,
        hazard: newHazard,
        alpha: newAlpha,
        px: pauliProbs.px,
        py: pauliProbs.py,
        pz: pauliProbs.pz,
        pi: pauliProbs.pi,
        eig0: eigenvalues[0] ?? null,
        eig1: eigenvalues[1] ?? null,
      };

      set((s) => ({
        step: newStep,
        hazard: newHazard,
        alpha: newAlpha,
        d: Number(raw["d"] ?? 3),
        probabilities,
        pauliProbs,
        eigenvalues,
        psdFreqs,
        psdAmps,
        stateVector,
        qecMetrics,
        history: s.history.length >= 5000
          ? [...s.history.slice(-4999), row]
          : [...s.history, row],
      }));
    },

    setConnected: (v) => set({ isConnected: v }),
    setRunning: (v) => set({ isRunning: v }),

    exportCSV: () => {
      const { history } = useSimulationStore.getState();
      if (history.length === 0) return;
      const header = "step,hazard,alpha,px,py,pz,pi,eig0,eig1";
      const rows = history.map((r) =>
        [r.step, r.hazard, r.alpha, r.px, r.py, r.pz, r.pi,
          r.eig0 ?? "", r.eig1 ?? ""].join(",")
      );
      const csv = [header, ...rows].join("\n");
      const blob = new Blob([csv], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `latent_noise_sim_${Date.now()}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    },

    startStatusPolling: (apiBase) => {
      const interval = setInterval(async () => {
        try {
          const res = await fetch(`${apiBase}/status`);
          if (res.ok) {
            const data = (await res.json()) as { running: boolean; step: number };
            set((s) => ({
              isRunning: data.running,
              // only forward step if simulation is idle (telemetry not flowing)
              ...(data.running ? {} : { step: data.step }),
            }));
          }
        } catch {
          // Network unavailable — leave current state
        }
      }, 3000);
      return () => clearInterval(interval);
    },
  }))
);
