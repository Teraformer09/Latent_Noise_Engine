import { useSimulationStore } from "../store/useSimulationStore";

const API_BASE = "http://localhost:8000";

async function postAction(path: string): Promise<void> {
  try {
    await fetch(`${API_BASE}${path}`, { method: "POST" });
  } catch (err) {
    console.error(`[SidebarControls] POST ${path} failed:`, err);
  }
}

export default function SidebarControls() {
  const { isConnected, isRunning, step, hazard, alpha, pauliProbs, setRunning } =
    useSimulationStore();

  const handleStart = async () => {
    await postAction("/start");
    setRunning(true);
  };

  const handleStop = async () => {
    await postAction("/stop");
    setRunning(false);
  };

  const pauliEntries: { label: string; value: number; color: string }[] = [
    { label: "Px", value: pauliProbs.px, color: "#ef4444" },
    { label: "Py", value: pauliProbs.py, color: "#f97316" },
    { label: "Pz", value: pauliProbs.pz, color: "#3b82f6" },
    { label: "Pi", value: pauliProbs.pi, color: "#22c55e" },
  ];

  const totalPauli = pauliEntries.reduce((acc, e) => acc + e.value, 0) || 1;

  return (
    <div className="h-full flex flex-col px-3 py-4 text-slate-100 text-sm select-none">
      {/* Title */}
      <h1 className="text-base font-bold mb-4 leading-tight bg-gradient-to-r from-cyan-400 to-violet-400 bg-clip-text text-transparent">
        Latent Noise Engine
      </h1>

      {/* Connection status */}
      <div className="flex items-center gap-2 mb-4">
        <span
          className={`inline-block w-2.5 h-2.5 rounded-full flex-shrink-0 ${
            isConnected ? "bg-green-400 shadow-[0_0_6px_#4ade80]" : "bg-red-500"
          }`}
        />
        <span className="text-slate-400 text-xs">
          {isConnected ? "Connected" : "Disconnected"}
        </span>
      </div>

      {/* Metrics */}
      <div className="space-y-2 mb-5">
        <MetricRow label="Step" value={String(step)} />
        <MetricRow label="Hazard" value={hazard.toFixed(4)} highlight={hazard > 0.1} />
        <MetricRow label="Alpha" value={alpha.toFixed(3)} />
      </div>

      {/* Pauli breakdown */}
      <div className="mb-5">
        <p className="text-slate-500 text-xs uppercase tracking-wider mb-2">
          Pauli Probabilities
        </p>
        {/* Stacked bar */}
        <div className="flex h-3 rounded overflow-hidden mb-2">
          {pauliEntries.map((e) => (
            <div
              key={e.label}
              style={{
                width: `${(e.value / totalPauli) * 100}%`,
                backgroundColor: e.color,
              }}
              title={`${e.label}: ${(e.value * 100).toFixed(1)}%`}
            />
          ))}
        </div>
        {/* Legend */}
        <div className="grid grid-cols-2 gap-x-2 gap-y-0.5">
          {pauliEntries.map((e) => (
            <div key={e.label} className="flex items-center gap-1 text-xs text-slate-400">
              <span
                className="inline-block w-2 h-2 rounded-sm flex-shrink-0"
                style={{ backgroundColor: e.color }}
              />
              {e.label}: {(e.value * 100).toFixed(1)}%
            </div>
          ))}
        </div>
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Start / Stop buttons */}
      <div className="space-y-2">
        <button
          onClick={handleStart}
          disabled={isRunning}
          className="w-full py-2 rounded text-xs font-semibold uppercase tracking-wider transition-colors
            bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed"
        >
          Start
        </button>
        <button
          onClick={handleStop}
          disabled={!isRunning}
          className="w-full py-2 rounded text-xs font-semibold uppercase tracking-wider transition-colors
            bg-rose-700 hover:bg-rose-600 disabled:bg-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed"
        >
          Stop
        </button>
      </div>
    </div>
  );
}

function MetricRow({
  label,
  value,
  highlight = false,
}: {
  label: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-slate-500 text-xs uppercase tracking-wider">{label}</span>
      <span
        className={`font-mono text-xs ${
          highlight ? "text-rose-400 font-bold" : "text-slate-200"
        }`}
      >
        {value}
      </span>
    </div>
  );
}
