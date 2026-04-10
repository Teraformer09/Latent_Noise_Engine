import WebSocketManager from "./components/WebSocketManager";
import ControlPanel from "./components/ControlPanel";
import FidelityPanel from "./components/FidelityPanel";
import Lattice3D from "./components/Lattice3D";
import BlochSphere from "./components/BlochSphere";
import EigenvaluePanel from "./components/EigenvaluePanel";
import MemoryHeatmap from "./components/MemoryHeatmap";
import BottomPanels from "./components/BottomPanels";
import { useSimulationStore } from "./store/useSimulationStore";

function StatusBar() {
  const { step, hazard, isConnected, isRunning } = useSimulationStore();

  return (
    <div className="col-span-12 flex items-center gap-6 px-4 bg-slate-900 border-b border-slate-800 text-xs font-mono overflow-hidden">
      <span className="font-bold tracking-widest text-cyan-400 uppercase">
        Latent Noise Engine
      </span>
      <span className="text-slate-500">
        Step:{" "}
        <span className="text-slate-200">{step}</span>
      </span>
      <span className={hazard > 0.1 ? "text-rose-400 font-bold" : "text-slate-500"}>
        Hazard:{" "}
        <span className={hazard > 0.1 ? "text-rose-300" : "text-slate-200"}>
          {hazard.toFixed(4)}
        </span>
      </span>
      <span className="flex items-center gap-1.5 text-slate-500">
        Status:{" "}
        <span
          className={`inline-block w-1.5 h-1.5 rounded-full ${
            isConnected ? "bg-green-400 shadow-[0_0_4px_#4ade80]" : "bg-red-500"
          }`}
        />
        <span className="text-slate-200">
          {!isConnected ? "Disconnected" : isRunning ? "Running" : "Connected"}
        </span>
      </span>
    </div>
  );
}

export default function App() {
  return (
    <div
      className="grid grid-cols-12 grid-rows-[60px_1fr_1fr_1fr] h-screen w-screen bg-slate-950 gap-1 p-1 overflow-hidden"
    >
      {/* Invisible WebSocket manager */}
      <WebSocketManager />

      {/* ── Row 0: Status bar (all 12 cols) ─────────────────────── */}
      <StatusBar />

      {/* ── ControlPanel: col 1-3, row-span-4 (full height) ──────── */}
      <div className="col-span-3 row-span-3 bg-slate-900 rounded overflow-hidden">
        <ControlPanel />
      </div>

      {/* ── Row 1: Lattice3D cols 4-9 ─────────────────────────────── */}
      <div className="col-span-6 bg-[#0f0f1a] rounded overflow-hidden">
        <Lattice3D />
      </div>

      {/* ── Row 1: FidelityPanel cols 10-12 ───────────────────────── */}
      <div className="col-span-3 bg-slate-900 rounded overflow-hidden">
        <FidelityPanel />
      </div>

      {/* ── Row 2: BlochSphere cols 4-7 ───────────────────────────── */}
      <div className="col-span-4 bg-[#0a0a18] rounded overflow-hidden">
        <BlochSphere />
      </div>

      {/* ── Row 2: EigenvaluePanel cols 8-9 ───────────────────────── */}
      <div className="col-span-2 bg-[#0a0a18] rounded overflow-hidden">
        <EigenvaluePanel />
      </div>

      {/* ── Row 2: MemoryHeatmap cols 10-12 ───────────────────────── */}
      <div className="col-span-3 bg-[#0a0a18] rounded overflow-hidden">
        <MemoryHeatmap />
      </div>

      {/* ── Row 3: BottomPanels cols 4-12 ─────────────────────────── */}
      <div className="col-span-9 bg-slate-900 rounded overflow-hidden">
        <BottomPanels />
      </div>
    </div>
  );
}
